"""Lift bounded static Sinkhorn normalization to one register-resident LoopOp."""

from __future__ import annotations

from collections.abc import Sequence

from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.loop import Assign, Axis, Load, Loop, LoopOp, Write
from emmy.compiler.ir.stmt import Body
from emmy.compiler.ir.tensor.ir import FixedSinkhornOp
from emmy.compiler.pipeline import Match, Pattern

PATTERN = [Pattern("root", FixedSinkhornOp)]


class _StraightLine:
    """Build unique scalar SSA assignments for one statically sized matrix."""

    def __init__(self) -> None:
        self.body: list = []
        self._next = 0

    def assign(self, op: str, *args: str) -> str:
        name = f"v{self._next}"
        self._next += 1
        self.body.append(Assign(name=name, op=ElementwiseImpl(op), args=tuple(args)))
        return name

    def fold(self, op: str, values: Sequence[str]) -> str:
        value = values[0]
        for other in values[1:]:
            value = self.assign(op, value, other)
        return value


def _normalize(builder: _StraightLine, cells: list[list[str]], eps: str, *, columns: bool) -> list[list[str]]:
    size = len(cells)
    result = [list(row) for row in cells]
    groups = ([cells[row][col] for row in range(size)] for col in range(size)) if columns else iter(cells)
    for group_index, group in enumerate(groups):
        denominator = builder.assign("add", builder.fold("add", group), eps)
        for element_index, value in enumerate(group):
            normalized = builder.assign("divide", value, denominator)
            if columns:
                result[element_index][group_index] = normalized
            else:
                result[group_index][element_index] = normalized
    return result


def _sinkhorn_body(source: str, output: str, eps_source: str, size: int, iterations: int) -> Body:
    builder = _StraightLine()
    batch = Var("a0")
    cells: list[list[str]] = []
    for row in range(size):
        values = []
        for col in range(size):
            name = f"in_{row}_{col}"
            builder.body.append(Load(name=name, input=source, index=(batch, Literal(row, "int"), Literal(col, "int"))))
            values.append(name)
        cells.append(values)
    builder.body.append(Load(name="eps", input=eps_source, index=(Literal(0, "int"),)))

    softmax: list[list[str]] = []
    for row in cells:
        maximum = builder.fold("maximum", row)
        exponentials = [builder.assign("exp", builder.assign("subtract", value, maximum)) for value in row]
        denominator = builder.fold("add", exponentials)
        softmax.append([builder.assign("add", builder.assign("divide", value, denominator), "eps") for value in exponentials])

    values = _normalize(builder, softmax, "eps", columns=True)
    for _ in range(iterations - 1):
        values = _normalize(builder, values, "eps", columns=False)
        values = _normalize(builder, values, "eps", columns=True)

    for row in range(size):
        for col in range(size):
            builder.body.append(
                Write(
                    output=output,
                    index=(batch, Literal(row, "int"), Literal(col, "int")),
                    value=values[row][col],
                )
            )
    return Body(builder.body)


def rewrite(match: Match, root: Node) -> Graph | None:
    graph = match.graph
    source = root.inputs[0]
    source_tensor = graph.buffer(source)
    if source_tensor is None:
        raise ValueError(f"FixedSinkhornOp input {source!r} is not a graph buffer")
    if source_tensor.dtype.name != "f32" or root.output.dtype.name != "f32":
        raise TypeError(f"FixedSinkhornOp lowering requires f32 input/output, got {source_tensor.dtype.name}/{root.output.dtype.name}")
    shape = tuple(source_tensor.shape)
    size = root.op.matrix_size(shape)

    output = f"lift_{root.id}"
    eps_source = f"{root.id}__eps"
    batch_axis = Axis(name="a0", extent=shape[0])
    kernel = LoopOp(
        body=Body(
            (
                Loop(
                    axis=batch_axis,
                    body=_sinkhorn_body(source, output, eps_source, size, root.op.iterations),
                ),
            )
        )
    )

    fragment = Graph()
    fragment.add_node(InputOp(), [], Tensor(source, shape, source_tensor.dtype), node_id=source)
    fragment.add_node(
        ConstantOp(name=eps_source, value=root.op.eps),
        [],
        Tensor(eps_source, (1,), "f32"),
        node_id=eps_source,
    )
    output_id = fragment.add_node(
        kernel,
        list(kernel.inputs),
        Tensor(root.output.name, root.output.shape, root.output.dtype),
        node_id=output,
    )
    fragment.outputs = [output_id]
    return fragment
