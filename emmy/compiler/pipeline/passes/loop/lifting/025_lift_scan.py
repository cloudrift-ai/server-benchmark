"""Lift an additive ``ScanOp`` to a serial-prefix ``LoopOp``.

The scan axis carries one accumulator across its ordered loop. Writing that accumulator after each update preserves
the cumulative value at every input coordinate; free axes remain ordinary enclosing loops.
"""

from __future__ import annotations

from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.elementwise import ElementwiseImpl, reduce_canon
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.loop import Accum, Axis, Load, Loop, LoopOp, Write
from emmy.compiler.ir.stmt import Body
from emmy.compiler.ir.tensor.ir import ScanOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped

PATTERN = [Pattern("root", ScanOp)]


def rewrite(match: Match, root: Node) -> Graph | None:
    graph = match.graph
    src_id = root.inputs[0]
    src_node = graph.producer(src_id)
    if src_node is None:
        raise RuleSkipped(f"scan input {src_id!r} no longer in graph")
    src_shape = tuple(src_node.output.shape)
    if not src_shape:
        raise RuleSkipped("scan input is scalar")
    if tuple(root.output.shape) != src_shape or root.output.dtype != src_node.output.dtype:
        raise RuleSkipped("scan must preserve input shape and dtype")
    if reduce_canon(root.op.name) != "add":
        raise RuleSkipped(f"only additive scans are supported, got {root.op.name!r}")

    axis_raw = root.op.axis
    if not isinstance(axis_raw, int) or isinstance(axis_raw, bool):
        raise RuleSkipped(f"scan axis must be a static integer, got {axis_raw!r}")
    axis = axis_raw if axis_raw >= 0 else axis_raw + len(src_shape)
    if axis < 0 or axis >= len(src_shape):
        raise RuleSkipped(f"scan axis {axis_raw} out of range for rank {len(src_shape)}")

    axes = tuple(Axis(name=f"a{i}", extent=dim) for i, dim in enumerate(src_shape))
    scan_axis = axes[axis]
    index = tuple(Var(item.name) for item in axes)
    inner: Body = (
        Loop(
            axis=scan_axis,
            body=(
                Load(name="in0", input=src_id, index=index),
                Accum(name="acc", value="in0", op=ElementwiseImpl("add"), axes=(scan_axis.name,)),
                Write(output=f"lift_{root.id}", index=index, value="acc"),
            ),
        ),
    )
    body = inner
    for free_axis in reversed([item for item in axes if item != scan_axis]):
        body = (Loop(axis=free_axis, body=body),)
    kernel = LoopOp(body=body)

    frag = Graph()
    frag.add_node(InputOp(), [], Tensor(src_id, src_shape, src_node.output.dtype), node_id=src_id)
    out_id = frag.add_node(
        kernel,
        list(kernel.inputs),
        Tensor(root.output.name, root.output.shape, root.output.dtype),
        node_id=f"lift_{root.id}",
    )
    frag.outputs = [out_id]
    return frag
