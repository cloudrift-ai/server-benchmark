"""Shared helper for the ``050_fold_into_constant`` / ``060_fold_reshape_into_constant`` rules.

Both rules absorb a single layout op (``TransposeOp`` / ``ReshapeOp``)
whose only input is a parameter ``ConstantOp`` into the constant's
``load_ops`` chain. The rewrite is uniform; the rule files differ only
in their ``PATTERN`` op type, so the body lives here.

Gated on sm_90+ (the TMA path). Folding under the default cp.async path
changes shapes in ways other passes (RoPE, SDPA, register-tile) don't
expect — the consumer Load index already matches the smem layout via
``120_transpose`` / ``130_reshape``'s IndexMap fusion. Folding under
the TMA path is what enables LDS.128 vectorization on the asymmetric
``(BN, BM)`` tile shape (see ``tuning.py``). The gate is a behavior
preservation choice; the ``load_ops`` field itself is general-purpose
and is honored by the loader regardless of how a chain got recorded.

**Matvec exception (all caps)**: a transposed weight feeding ONLY M=1
matmuls (the decode gemv tier) folds below sm_90 too. The ``b<n>t``
transposed REDUCE band — the tier's recorded golden winners on sm_89 —
assumes k-major B storage, which only the fold produces; composing the
transpose into the read indices (the sub-sm_90 default) leaves B in the
``F.linear`` layout where the band is (correctly) never offered, so the
recorded ``.m1.t`` goldens could never realize in-model on those cards.
The matvec shape has no RoPE/SDPA/register-tile consumer to disturb —
the behavior-preservation concern above doesn't apply to it.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.context import Context
from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.base import ConstantOp
from emmy.compiler.pipeline import RuleSkipped
from emmy.compiler.pipeline.passes.frontend.decomposition._helpers import open_fragment


def _feeds_only_matvecs(graph: Graph, root: Node) -> bool:
    """``True`` iff every matmul consuming ``root``'s output is a MATVEC —
    its (keepdim) output has static extent 1 on every axis but the last.
    Probed on whichever form the consumers are in when the fold fires: a
    still-frontend ``MatmulOp``, or ``040_linear``'s already-decomposed
    unsqueeze → broadcast → multiply → ``ReduceOp`` chain (bounded walk).
    No matmul consumer found → ``False`` (conservative: no fold)."""
    from emmy.compiler.ir.frontend.ir import MatmulOp  # noqa: PLC0415
    from emmy.compiler.ir.tensor.ir import ReduceOp  # noqa: PLC0415

    def _is_rowvec(shape) -> bool:
        return len(shape) >= 1 and all(d.is_static and d.as_static() == 1 for d in shape[:-1])

    seen: set[str] = set()
    frontier = [root.id]
    verdicts: list[bool] = []
    for _ in range(5):  # transpose → unsqueeze → broadcast → multiply → reduce
        nxt: list[str] = []
        for nid in frontier:
            for cid in graph.users(nid):
                if cid in seen:
                    continue
                seen.add(cid)
                node = graph.nodes.get(cid)
                if node is None:
                    continue
                if isinstance(node.op, (ReduceOp, MatmulOp)):
                    verdicts.append(_is_rowvec(node.output.shape))
                else:
                    nxt.append(cid)
        frontier = nxt
    return bool(verdicts) and all(verdicts)


def fold_into_constant(graph: Graph, root: Node, inp_x: Node, out: Tensor, ctx: Context) -> Graph | None:
    """Append ``root.op`` to ``inp_x.op.load_ops`` and rebuild the constant.

    Skips scalar constants (``value is not None``) — the loader never
    visits them. Skips activations — only parameter/buffer constants
    get a load_ops chain.
    """
    from emmy.compiler.ir.frontend.ir import TransposeOp  # noqa: PLC0415

    if not ctx.has_tma and not (isinstance(root.op, TransposeOp) and _feeds_only_matvecs(graph, root)):
        raise RuleSkipped("TMA path inactive and not an M=1 matvec weight — fold not needed")
    if not isinstance(inp_x.op, ConstantOp):
        raise RuleSkipped("input is not a ConstantOp")
    if inp_x.op.value is not None:
        raise RuleSkipped("scalar constants are not folded into")

    # ``replace`` rather than a field-list reconstruction so every ConstantOp
    # field — including ones added later — propagates by construction.
    new_op = replace(inp_x.op, load_ops=inp_x.op.load_ops + (root.op,))
    frag = open_fragment(graph, [])
    new_id = frag.add_node(op=new_op, inputs=[], output=Tensor(out.name, out.shape, out.dtype))
    frag.outputs = [new_id]
    return frag
