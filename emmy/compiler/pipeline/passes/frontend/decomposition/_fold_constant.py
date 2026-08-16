"""Shared helper for the ``050_fold_into_constant`` / ``060_fold_reshape_into_constant`` rules.

Both rules absorb a single layout op (``TransposeOp`` / ``ReshapeOp``)
whose only input is a parameter ``ConstantOp`` into the constant's
``load_ops`` chain. The rewrite is uniform; the rule files differ only
in their ``PATTERN`` op type, so the body lives here.

The fold is an UNCONDITIONAL structural canonicalization: a parameter's
layout op always dissolves into its load chain, on every capability and
every consumer shape, so one weight has one stored form. The
earlier sub-sm_90 gate — and the M=1 matvec
carve-out punched through it for the ``.m1.t`` golden layout — were
shape/hardware special-casing in a pass, exactly what the no-gates
doctrine forbids: which realization of a layout wins is measured
evidence's call, never a pass condition. Records captured under the
gated behavior re-record against the folded form (the bench campaign's
re-record worklist); the ``load_ops`` field itself is general-purpose
and is honored by the loader regardless of how a chain got recorded.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.base import ConstantOp
from emmy.compiler.pipeline import RuleSkipped
from emmy.compiler.pipeline.passes.frontend.decomposition._helpers import open_fragment


def fold_into_constant(graph: Graph, root: Node, inp_x: Node, out: Tensor) -> Graph | None:
    """Append ``root.op`` to ``inp_x.op.load_ops`` and rebuild the constant.

    Skips scalar constants (``value is not None``) — the loader never
    visits them. Skips activations — only parameter/buffer constants
    get a load_ops chain.
    """
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
