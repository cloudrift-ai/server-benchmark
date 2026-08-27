"""Decompose a two-operand EinsumOp into transpose → MatmulOp → transpose.

An einsum over two operands is a permutation followed by a contraction, so the
whole rule is: line the shared batch labels up in the same order on both sides,
put the contracted label last on A and first on B, hand that to ``MatmulOp``, and
permute the result into the requested output order.

Deliberately no reshape. Grouping several free labels into one matrix axis would
need the product of their extents, which is not expressible when one of them is
symbolic — and a linear-attention einsum is exactly where a symbolic sequence
length shows up. The tracer therefore admits only the single-label form this rule
handles, so an equation that would need reshaping never reaches here.
"""

from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.frontend.ir import EinsumOp, MatmulOp, TransposeOp
from emmy.compiler.pipeline import Match, Pattern
from emmy.compiler.pipeline.passes.frontend.decomposition._helpers import open_fragment

PATTERN = [Pattern("root", EinsumOp)]


def _aligned(frag: Graph, src: Node, labels: str, wanted: str, *, name: str) -> Node | str:
    """Permute ``src`` from ``labels`` order into ``wanted`` order (identity stays put)."""
    axes = tuple(labels.index(label) for label in wanted)
    if axes == tuple(range(len(axes))):
        return src
    shape = tuple(src.output.shape[axis] for axis in axes)
    nid = frag.add_node(
        op=TransposeOp(axes=axes),
        inputs=[src],
        output=Tensor(name, shape, src.output.dtype),
    )
    return frag.nodes[nid]


def rewrite(match: Match, root: Node, inp_a: Node, inp_b: Node, out: Tensor) -> Graph | None:
    graph = match.graph
    a_labels, b_labels, out_labels = root.op._terms()

    contracted = [label for label in a_labels if label in b_labels and label not in out_labels]
    batch = [label for label in out_labels if label in a_labels and label in b_labels]
    free_a = [label for label in out_labels if label in a_labels and label not in b_labels]
    free_b = [label for label in out_labels if label in b_labels and label not in a_labels]

    frag = open_fragment(graph, [inp_a, inp_b])
    batch_order = "".join(batch)
    lhs = _aligned(frag, inp_a, a_labels, batch_order + "".join(free_a) + "".join(contracted), name=f"{out.name}_a")
    rhs = _aligned(frag, inp_b, b_labels, batch_order + "".join(contracted) + "".join(free_b), name=f"{out.name}_b")

    # MatmulOp contracts A's last axis with B's second-to-last and broadcasts the batch
    # prefix, so the product is already in ``batch + free_a + free_b`` label order.
    product_labels = batch_order + "".join(free_a) + "".join(free_b)
    lhs_node = lhs if isinstance(lhs, Node) else frag.nodes[lhs]
    rhs_node = rhs if isinstance(rhs, Node) else frag.nodes[rhs]
    product_shape = tuple(lhs_node.output.shape[:-1]) + (rhs_node.output.shape[-1],)
    needs_permute = product_labels != out_labels
    mm_id = frag.add_node(
        op=MatmulOp(),
        inputs=[lhs, rhs],
        output=Tensor(f"{out.name}_mm" if needs_permute else out.name, product_shape, out.dtype),
    )

    if not needs_permute:
        frag.outputs = [mm_id]
        return frag

    axes = tuple(product_labels.index(label) for label in out_labels)
    out_id = frag.add_node(
        op=TransposeOp(axes=axes),
        inputs=[mm_id],
        output=Tensor(out.name, out.shape, out.dtype),
    )
    frag.outputs = [out_id]
    return frag
