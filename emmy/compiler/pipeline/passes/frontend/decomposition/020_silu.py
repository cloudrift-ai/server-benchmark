"""Decompose silu(x) into x * recip(1 + exp(-x)) to enable SiLU+Mul fusion."""

from emmy.compiler.dtype import BF16, F16, F32
from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.tensor.ir import ElementwiseOp
from emmy.compiler.pipeline import Match, Pattern
from emmy.compiler.pipeline.passes.frontend.decomposition._helpers import const_bc, open_fragment

PATTERN = [Pattern("root", ElementwiseOp, {"fn": "silu"})]


def rewrite(match: Match, inp_x: Node, out: Tensor) -> Graph | None:
    graph = match.graph
    """Replace silu(x) with x * recip(1 + exp(-x))."""
    frag = open_fragment(graph, [inp_x])
    opmath_dtype = F32 if inp_x.output.dtype in (F16, BF16) else out.dtype

    work_x = inp_x
    if inp_x.output.dtype != opmath_dtype:
        work_id = frag.add_node(
            op=ElementwiseOp(op="copy"),
            inputs=[inp_x],
            output=Tensor(f"{out.name}_opmath", out.shape, opmath_dtype),
        )
        work_x = frag.nodes[work_id]

    neg_id = frag.add_node(op=ElementwiseOp(op="negative"), inputs=[work_x], output=Tensor(f"{out.name}_neg", out.shape, opmath_dtype))
    exp_id = frag.add_node(op=ElementwiseOp(op="exp"), inputs=[neg_id], output=Tensor(f"{out.name}_exp", out.shape, opmath_dtype))
    one_bc = const_bc(frag, name=f"{out.name}_one", value=1.0, target_shape=out.shape, dtype=opmath_dtype)
    add_id = frag.add_node(
        op=ElementwiseOp(op="add"),
        inputs=[one_bc, exp_id],
        output=Tensor(f"{out.name}_denom", out.shape, opmath_dtype),
    )
    recip_id = frag.add_node(
        op=ElementwiseOp(op="reciprocal"),
        inputs=[add_id],
        output=Tensor(f"{out.name}_sigmoid", out.shape, opmath_dtype),
    )
    mul_id = frag.add_node(
        op=ElementwiseOp(op="multiply"),
        inputs=[work_x, recip_id],
        output=Tensor(out.name, out.shape, out.dtype),
    )

    frag.outputs = [mul_id]
    return frag
