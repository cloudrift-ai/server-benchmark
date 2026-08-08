"""Lift a HAT-BASIS ``TrellisDecodeOp`` to a ``LoopOp`` of per-element ``TrellisLoad`` reads.

The hat-basis (``hadamard=False``) trellis decode is the form with a kernel realization: its
output element ``(n, k)`` is a pure function of the packed codes buffer alone (the 16-bit
window is directly addressable in the tile's circular bit stream — no carried walk state), so
the op lifts to the elementwise copy-kernel shape every other lift produces:

    for n: for k: w = trellis_decode(codes @ (k, n)); out[n, k] = w

Loop fusion then inlines this producer into its consuming matmul exactly as it inlines an fp8
decode cone, handing recognition a contraction whose B operand is a computed trellis cone —
the ``bind_contraction`` trellis arm binds it, and the warp tier's compute fill decodes the B
tile in-kernel while the codes stay compressed in device memory.

This rule only ever fires when the cone SURVIVED ``032_fold_constant_subgraphs`` — an
input-rooted cone (compressed weights as program inputs), or a constant-rooted one under
``EMMY_TRELLIS_EXPAND``. A checkpoint-basis (``hadamard=True``) op is never lifted: it has no
per-element form (the 128-block Hadamard mixes rows), and 032 folds it unconditionally.
"""

from __future__ import annotations

from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.frontend.ir import TrellisDecodeOp
from emmy.compiler.ir.loop import Axis, Loop, LoopOp, Write
from emmy.compiler.ir.stmt import TrellisLoad
from emmy.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", TrellisDecodeOp)]


def rewrite(root: Node, inp_codes: Node, out: Tensor) -> Graph | None:
    op: TrellisDecodeOp = root.op
    if op.hadamard:
        raise RuleSkipped("checkpoint-basis trellis decode has no per-element form — 032 folds it")
    t_shape = tuple(d.as_static() for d in inp_codes.output.shape)
    k_bits = t_shape[-1] // 16
    n_ax = Axis(name="a0", extent=op.out_features)
    k_ax = Axis(name="a1", extent=op.in_features)
    inner = (
        TrellisLoad(name="w", input=inp_codes.id, index=(Var(k_ax.name), Var(n_ax.name)), cb=op.cb, k_bits=k_bits, n_tiles=t_shape[1]),
        Write(output=f"kernel_{root.id}", index=(Var(n_ax.name), Var(k_ax.name)), value="w"),
    )
    kernel = LoopOp(body=(Loop(axis=n_ax, body=(Loop(axis=k_ax, body=inner),)),))

    frag = Graph()
    frag.add_node(InputOp(), [], Tensor(inp_codes.id, inp_codes.output.shape, inp_codes.output.dtype), node_id=inp_codes.id)
    out_id = frag.add_node(kernel, list(kernel.inputs), Tensor(out.name, out.shape, out.dtype), node_id=f"kernel_{root.id}")
    frag.outputs = [out_id]
    return frag
