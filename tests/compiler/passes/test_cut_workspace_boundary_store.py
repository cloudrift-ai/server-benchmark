"""A placement cut re-spells the consumer's kernel-boundary stores, not only its term.

The cut replaces a cone with a workspace read named after the WORKSPACE — the cone's result name
tagged with the seam — so the value it materialized cannot collide with a structurally equal cone
still computed in place beside it. A term's readers follow that rename on their own: a consumer's
params are spelled as the result names of the edge they bind, so replacing the edge re-spells every
read of it at once.

A kernel-boundary store is not part of the term. ``TileOp.output_specs`` names its stored value as a
plain string and is only reconstituted into a body at lowering, where some arms re-spell it through
the term's applied params and some hand it to ``apply_output_specs`` as it stands. Cutting a branch
the kernel stores WHOLE leaves that store as the only reader the value has, so on the arms that do
not re-spell it named a value the consumer no longer defines: the per-cell rename then suffixed it
once per register cell and nvcc rejected every one — ``identifier "v120__c<i>_<j>" is undefined``,
a hundred errors deep in the fused W4A4 MLP kernel of the Qwen3-8B-NVFP4 decode trunk, which cost
that tune candidate its measurement.

The oracle is the cut's own postcondition, asked where the rename is minted rather than at whichever
arm consumes it: every value the consumer stores must be one its term defines.
"""

from __future__ import annotations

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.schedule import Placement
from emmy.compiler.ir.stmt import Assign, Write
from emmy.compiler.ir.stmt.leaves import OutputSpec
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline.passes.lowering.tile._cut import cuttable_seams, output_map, realize
from tests.compiler.terms import projection, reduction, slab

_ROW, _COL = Axis("m", Dim(8)), Axis("k", Dim(32))


def _kernel() -> tuple[Graph, Node]:
    """A two-output kernel over one input: a row sum, and the square of that sum.

    ``total`` is the branch the kernel stores WHOLE — cutting it materializes exactly the value one
    boundary store is written from — while ``squared`` keeps a second output reading it, so the
    consumer survives the cut with work of its own.
    """
    total = reduction(_COL, (slab("cell", "x", "m", "k"),), (Assign(name="total__v", op="copy", args=("cell",)),), ("total",))
    root = projection((total,), (Assign(name="squared", op="multiply", args=("total", "total")),), results=("squared",))
    tile = TileOp(
        op=root,
        place=Placement(free=(_ROW,)),
        axes=(_ROW, _COL),
        output_specs=(
            OutputSpec(Write(output="sum", index=(Var("m"),), value="total")),
            OutputSpec(Write(output="square", index=(Var("m"),), value="squared")),
        ),
    )
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (8, 32), dtype=F32), node_id="x")
    graph.add_node(tile, ["x"], outputs=[Tensor("sum", (8,), dtype=F32), Tensor("square", (8,), dtype=F32)], node_id="sum")
    graph.inputs, graph.outputs = ["x"], ["sum", "square"]
    return graph, graph.nodes["sum"]


class _Match:
    """The one thing ``realize`` asks a match for: the graph it looks input buffers up in."""

    def __init__(self, graph: Graph) -> None:
        self.graph = graph


def _defined(tile: TileOp) -> set[str]:
    """Every value the consumer's term defines — its body lowered with NO stores spliced in, so the
    answer is what the kernel has to store FROM, never what a store happens to name."""
    names: set[str] = set()
    pending = list(tile.op.lower(frozenset(axis.name for axis in tile.place.free), (), tile.axes))
    while pending:
        stmt = pending.pop()
        names.update(stmt.defines())
        pending.extend(inner for body in stmt.nested() for inner in body)
    return names


def test_a_cut_branch_the_kernel_stores_whole_keeps_its_boundary_store_readable() -> None:
    graph, node = _kernel()
    seams = [seam for seam in cuttable_seams(node.op) if seam.node.axis is not None]
    assert seams, "the reducing branch must be offered as a cuttable seam for this shape to arise"

    fragment = realize(_Match(graph), node, (seams[0],), output_map(node))
    consumer = next(n for n in fragment.nodes.values() if isinstance(n.op, TileOp) and n.op.op.axis is None)

    defined = _defined(consumer.op)
    stored = {value for spec in consumer.op.output_specs for value in spec.write.values}
    assert stored <= defined, f"the consumer stores values its term never defines: {sorted(stored - defined)}"
