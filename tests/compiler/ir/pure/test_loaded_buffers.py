"""The gmem buffers a stored term reads — ``_cut._buffer_reads``, read off the tree through its
operand edges, never off a lowered view."""

from __future__ import annotations

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.stmt import Assign, Load
from emmy.compiler.pipeline.passes.lowering.tile._cut import _buffer_reads
from tests.compiler.terms import contraction, projection, slab


def _root_holding_a_cone():
    """A cone whose OPERAND edge loads ``ws``, held as an operand of the root.

    This is DeepSeek-V4 ``post4096``'s shape, minimized: repeated placement leaves the fused root
    holding its cut cones, and each cone reads its workspace through an operand edge. The cone is
    an EDGE and not a body member — a term composes through ``operands``, and a body holds
    statements — so a reader that walks statements alone cannot see beneath it."""
    source = projection(body=(Load(name="w", input="ws", index=(Var("j"),)),), results=("w",))
    cone = projection((source,), (Assign(name="c", op="relu", args=("w",)),), ("c",))
    return projection((cone,), (Assign(name="out", op="copy", args=("c",)),), ("out",))


def test_a_cone_held_as_an_edge_still_reports_its_operand_buffers() -> None:
    """The lowered body cannot answer this, which is why the cut may not ask it.

    ``Body.loads`` walks ``Stmt.nested()``, and a Fold's operand edges are not nested statements —
    so a region that keeps its cone as a term hides every buffer beneath that edge. A cut that
    declared the lowered view named fewer graph inputs than the kernel the materializer built from
    the same tree went on to read; the workspace producers lost their only consumer edge, were
    pruned as orphans, and the launch asked for a buffer nothing had allocated
    (``KeyError: '<node>__place_<token>_0'`` on DeepSeek-V4's TP8xPP2 boot)."""
    root = _root_holding_a_cone()

    assert not root.lift.body.loads, "the root's own body loads nothing — every buffer is beneath an edge"
    assert _buffer_reads(root) == {"ws"}


def test_buffer_reads_see_a_contraction_through_its_edges() -> None:
    """A contraction's lift is its products alone — its algebra IS its operand edges, and the slabs
    are where the buffers are read."""
    node = contraction(Axis("k", 8), slab("av", "x", "m", "k"), (slab("bv", "w", "n", "k"), "acc"))

    assert not node.lift.body.loads
    assert _buffer_reads(node) == {"x", "w"}
