"""Decompose default softplus into an overflow-stable primitive chain.

``max(x, 0) + log(1 + exp(-abs(x)))`` is algebraically equivalent to
``log(1 + exp(x))`` but never exponentiates a large positive value.  The
tracer accepts only PyTorch's default beta/threshold parameters.
"""

from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.tensor.ir import ElementwiseOp
from emmy.compiler.pipeline import Match, Pattern
from emmy.compiler.pipeline.passes.frontend.decomposition._helpers import const_bc, open_fragment

PATTERN = [Pattern("root", ElementwiseOp, {"fn": "softplus"})]


def rewrite(match: Match, inp_x: Node, out: Tensor) -> Graph | None:
    frag = open_fragment(match.graph, [inp_x])
    abs_id = frag.add_node(
        op=ElementwiseOp("abs"),
        inputs=[inp_x],
        output=Tensor(f"{out.name}_abs", out.shape, out.dtype),
    )
    neg_id = frag.add_node(
        op=ElementwiseOp("negative"),
        inputs=[abs_id],
        output=Tensor(f"{out.name}_neg_abs", out.shape, out.dtype),
    )
    exp_id = frag.add_node(
        op=ElementwiseOp("exp"),
        inputs=[neg_id],
        output=Tensor(f"{out.name}_exp", out.shape, out.dtype),
    )
    one = const_bc(frag, name=f"{out.name}_one", value=1.0, target_shape=out.shape, dtype=out.dtype)
    denom_id = frag.add_node(
        op=ElementwiseOp("add"),
        inputs=[one, exp_id],
        output=Tensor(f"{out.name}_one_plus_exp", out.shape, out.dtype),
    )
    log_id = frag.add_node(
        op=ElementwiseOp("log"),
        inputs=[denom_id],
        output=Tensor(f"{out.name}_log", out.shape, out.dtype),
    )
    zero = const_bc(frag, name=f"{out.name}_zero", value=0.0, target_shape=out.shape, dtype=out.dtype)
    positive_id = frag.add_node(
        op=ElementwiseOp("maximum"),
        inputs=[inp_x, zero],
        output=Tensor(f"{out.name}_positive", out.shape, out.dtype),
    )
    result_id = frag.add_node(
        op=ElementwiseOp("add"),
        inputs=[positive_id, log_id],
        output=Tensor(out.name, out.shape, out.dtype),
    )
    frag.outputs = [result_id]
    return frag
