"""Lift row RMSNorm plus suffix GPT-J RoPE to one ``LoopOp``."""

from __future__ import annotations

from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var
from emmy.compiler.ir.loop import Accum, Assign, Axis, Cond, Load, Loop, LoopOp, Select, SelectBranch, Write
from emmy.compiler.ir.stmt import Body
from emmy.compiler.ir.tensor.ir import RowRmsNormRopeOp
from emmy.compiler.pipeline import Match, Pattern

PATTERN = [Pattern("root", RowRmsNormRopeOp)]


def rewrite(match: Match, root: Node) -> Graph | None:
    graph = match.graph
    q_id, positions_id, cache_id = root.inputs
    q = graph.buffer(q_id)
    positions = graph.buffer(positions_id)
    cache = graph.buffer(cache_id)
    if q is None or positions is None or cache is None:
        raise ValueError("RowRmsNormRopeOp inputs must be graph buffers")
    dtypes = (q.dtype.name, positions.dtype.name, cache.dtype.name, root.output.dtype.name)
    if dtypes != ("f16", "i64", "f32", "f16"):
        raise TypeError(f"RowRmsNormRopeOp lowering requires f16/i64/f32 -> f16, got {dtypes}")

    rows, heads, head_dim_value = tuple(q.shape)
    if not head_dim_value.is_static:
        raise ValueError("RowRmsNormRopeOp lowering requires a static head dimension")
    head_dim = head_dim_value.as_static()
    rope_dim = root.op.rope_dim
    nope_dim = head_dim - rope_dim
    half_rope = rope_dim // 2

    row = Axis(name="a0", extent=rows)
    head = Axis(name="a1", extent=heads)
    feature = Axis(name="a2", extent=head_dim_value)
    output = f"lift_{root.id}"
    mean_count = f"{root.id}__mean_count"
    eps = f"{root.id}__eps"

    reduce_body: Body = (
        Load(name="q_reduce", input=q_id, index=(Var("a0"), Var("a1"), Var("a2"))),
        Assign(name="q_square", op=ElementwiseImpl("multiply"), args=("q_reduce", "q_reduce")),
        Accum(name="sum_square", value="q_square", op=ElementwiseImpl("add"), axes=("a2",)),
    )
    rope_body: Body = (
        Load(name="position", input=positions_id, index=(Var("a0"),)),
        Load(
            name="q_partner",
            input=q_id,
            index=(
                Var("a0"),
                Var("a1"),
                TernaryExpr(
                    BinaryExpr(
                        "==", BinaryExpr("%", BinaryExpr("-", Var("a2"), Literal(nope_dim, "int")), Literal(2, "int")), Literal(0, "int")
                    ),
                    BinaryExpr("+", Var("a2"), Literal(1, "int")),
                    BinaryExpr("-", Var("a2"), Literal(1, "int")),
                ),
            ),
        ),
        Assign(name="partner_norm", op=ElementwiseImpl("multiply"), args=("q_partner", "rrms")),
        Load(
            name="rope_cos",
            input=cache_id,
            index=(Var("position"), BinaryExpr("//", BinaryExpr("-", Var("a2"), Literal(nope_dim, "int")), Literal(2, "int"))),
        ),
        Load(
            name="rope_sin",
            input=cache_id,
            index=(
                Var("position"),
                BinaryExpr(
                    "+",
                    Literal(half_rope, "int"),
                    BinaryExpr("//", BinaryExpr("-", Var("a2"), Literal(nope_dim, "int")), Literal(2, "int")),
                ),
            ),
        ),
        Assign(name="current_cos", op=ElementwiseImpl("multiply"), args=("q_normalized", "rope_cos")),
        Assign(name="partner_sin", op=ElementwiseImpl("multiply"), args=("partner_norm", "rope_sin")),
        Assign(name="rope_even", op=ElementwiseImpl("subtract"), args=("current_cos", "partner_sin")),
        Assign(name="rope_odd", op=ElementwiseImpl("add"), args=("current_cos", "partner_sin")),
        Select(
            name="q_roped",
            branches=(
                SelectBranch(
                    value="rope_even",
                    select=BinaryExpr(
                        "==", BinaryExpr("%", BinaryExpr("-", Var("a2"), Literal(nope_dim, "int")), Literal(2, "int")), Literal(0, "int")
                    ),
                ),
                SelectBranch(value="rope_odd", select=Literal(1, "int")),
            ),
        ),
        Write(output=output, index=(Var("a0"), Var("a1"), Var("a2")), value="q_roped"),
    )
    output_body: Body = (
        Load(name="q_output", input=q_id, index=(Var("a0"), Var("a1"), Var("a2"))),
        Assign(name="q_normalized", op=ElementwiseImpl("multiply"), args=("q_output", "rrms")),
        Cond(
            cond=BinaryExpr("<", Var("a2"), Literal(nope_dim, "int")),
            body=(Write(output=output, index=(Var("a0"), Var("a1"), Var("a2")), value="q_normalized"),),
            else_body=rope_body,
        ),
    )
    head_body: Body = (
        Loop(axis=feature, body=reduce_body),
        Load(name="mean_count", input=mean_count, index=(Literal(0, "int"),)),
        Load(name="eps", input=eps, index=(Literal(0, "int"),)),
        Assign(name="mean_square", op=ElementwiseImpl("divide"), args=("sum_square", "mean_count")),
        Assign(name="mean_eps", op=ElementwiseImpl("add"), args=("mean_square", "eps")),
        Assign(name="rrms", op=ElementwiseImpl("rsqrt"), args=("mean_eps",)),
        Loop(axis=feature, body=output_body),
    )
    kernel = LoopOp(body=Body((Loop(axis=row, body=(Loop(axis=head, body=head_body),)),)))

    fragment = Graph()
    for input_id, tensor in ((q_id, q), (positions_id, positions), (cache_id, cache)):
        fragment.add_node(InputOp(), [], Tensor(input_id, tensor.shape, tensor.dtype), node_id=input_id)
    fragment.add_node(ConstantOp(name=mean_count, value=float(head_dim)), [], Tensor(mean_count, (1,), "f32"), node_id=mean_count)
    fragment.add_node(ConstantOp(name=eps, value=root.op.eps), [], Tensor(eps, (1,), "f32"), node_id=eps)
    output_id = fragment.add_node(
        kernel,
        list(kernel.inputs),
        Tensor(root.output.name, root.output.shape, root.output.dtype),
        node_id=output,
    )
    fragment.outputs = [output_id]
    return fragment
