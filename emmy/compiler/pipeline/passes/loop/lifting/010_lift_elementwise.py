"""Lift pointwise tensor compute to a trivial single-op ``LoopOp``.

Every ``ElementwiseOp`` and same-width ``BitcastOp`` in the post-decomposition
graph is wrapped as a one-op kernel that reads its inputs via identity Ports,
applies the op, and writes the result. Broadcasts on input buffers are handled by right-aligning non-
size-1 dims onto kernel axes (matching the iteration space of the output).

Mergeable pairs (producer/consumer LoopOp) are collapsed later by the merge
rule; this pass only introduces the LoopOp wrapper.
"""

from __future__ import annotations

from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Expr, Literal, Var
from emmy.compiler.ir.loop import Assign, Axis, Load, Loop, LoopOp, Stmt, Write
from emmy.compiler.ir.stmt import Body
from emmy.compiler.ir.stmt.base import dtype_promote
from emmy.compiler.ir.tensor.ir import BitcastOp, ElementwiseOp
from emmy.compiler.pipeline import Match, Pattern

PATTERN = [Pattern("root", (ElementwiseOp, BitcastOp))]


def rewrite(match: Match, root: Node) -> Graph | None:
    graph = match.graph
    out_shape = tuple(root.output.shape)
    if isinstance(root.op, BitcastOp):
        inp = graph.buffer(root.inputs[0])
        if inp is None or inp.dtype.nbytes != root.output.dtype.nbytes:
            raise ValueError(f"BitcastOp requires equal element widths, got {None if inp is None else inp.dtype} and {root.output.dtype}")
    # Symbolic dims on FREE axes are fine — they pass through as ``Axis(extent=Dim("name"))``
    # and ``LoopOp.forward`` / launch geometry resolve them from input array shapes at run time.
    axes = tuple(Axis(name=f"a{i}", extent=d) for i, d in enumerate(out_shape))

    load_stmts: list[Stmt] = []
    load_names: list[str] = []
    for i, inp_id in enumerate(root.inputs):
        inp_t = graph.buffer(inp_id)
        inp_shape = tuple(inp_t.shape) if inp_t is not None else ()
        idx = _identity_index(inp_shape, axes)
        name = f"in{i}"
        load_names.append(name)
        load_stmts.append(Load(name=name, input=inp_id, index=idx))

    write_index = tuple(Var(a.name) for a in axes)
    if isinstance(root.op, ElementwiseOp):
        op = root.op.op
    elif isinstance(root.op, BitcastOp):
        op = ElementwiseImpl("bitcast")
    # ``copy`` is also the generic spelling used by rewrites that deliberately
    # keep a numerical precision boundary inside a fused contraction. Unlike
    # ordinary pointwise arithmetic, its declared output dtype is the operation
    # itself, not merely the eventual store type. Stamp the scalar assignment so
    # fusion cannot erase a widening cast (for example u32 -> u64 before a
    # packed shift).
    input_dtypes = [graph.buffer(nid).dtype.name for nid in root.inputs if graph.buffer(nid) is not None]
    promoted = dtype_promote(op.name, input_dtypes) if isinstance(root.op, ElementwiseOp) else None
    assign_dtype = root.output.dtype if op.name == "bitcast" or root.output.dtype.name != promoted else None
    inner: Body = (
        *load_stmts,
        Assign(name="v", op=op, args=tuple(load_names), dtype=assign_dtype),
        Write(output=f"lift_{root.id}", index=write_index, value="v"),
    )
    # Nest the body in free-axis Loops (outer axis wraps the innermost).
    body: Body = inner
    for a in reversed(axes):
        body = (Loop(axis=a, body=body),)
    kernel = LoopOp(body=body)

    frag = Graph()
    for inp_id in root.inputs:
        if inp_id in frag.nodes:
            continue
        ext_t = graph.buffer(inp_id)
        shape = ext_t.shape if ext_t is not None else ()
        dtype = ext_t.dtype if ext_t is not None else "f32"
        frag.add_node(InputOp(), [], Tensor(inp_id, shape, dtype), node_id=inp_id)

    out_id = frag.add_node(
        kernel,
        list(kernel.inputs),
        Tensor(root.output.name, root.output.shape, root.output.dtype),
        node_id=f"lift_{root.id}",
    )
    frag.outputs = [out_id]
    return frag


def _identity_index(src_shape: tuple, axes: tuple[Axis, ...]) -> tuple[Expr, ...]:
    """Build an identity read index for ``src_shape`` under ``axes``.

    Walks right-to-left so non-size-1 source dims latch onto the rightmost
    matching-extent axis. Size-1 source dims become ``Literal(0)`` (broadcast).
    Missing leading axes (scalar / fewer-dim inputs) contribute nothing.
    """
    if not src_shape:
        return ()

    result: list[Expr] = [Literal(0, "int")] * len(src_shape)
    cursor = len(axes) - 1

    for i in range(len(src_shape) - 1, -1, -1):
        dim = src_shape[i]
        if dim.is_static and dim.as_static() == 1:
            continue
        while cursor >= 0 and axes[cursor].extent != dim:
            cursor -= 1
        if cursor < 0:
            break
        result[i] = Var(axes[cursor].name)
        cursor -= 1
    return tuple(result)
