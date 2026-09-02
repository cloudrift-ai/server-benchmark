"""``loaded_buffers`` — every buffer a stored term reads, operand edges included."""

from __future__ import annotations

from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.pure import Fold
from emmy.compiler.ir.pure.fold import loaded_buffers
from emmy.compiler.ir.stmt import Assign, Body, Load


def _root_holding_a_cone() -> Fold:
    """A cone whose OPERAND edge loads ``ws``, held as an operand of the root.

    This is DeepSeek-V4 ``post4096``'s shape, minimized: repeated placement leaves the fused root
    holding its cut cones, and each cone reads its workspace through an operand edge. The cone is
    an EDGE and not a body member — a term composes through ``operands``, and a body holds
    statements — but the reading the buffer walk has to do is unchanged: an edge is not a nested
    statement, so the lowered view cannot see beneath it."""
    source = Fold.projection(body=Body((Load(name="w", input="ws", index=(Var("j"),)),)), results=("w",))
    cone = Fold.projection(operands=(source,), body=Body((Assign(name="c", op="relu", args=("w",)),)), results=("c",))
    return Fold.projection(operands=(cone,), body=Body((Assign(name="out", op="copy", args=("c",)),)), results=("out",))


def test_a_cone_held_as_an_edge_still_reports_its_operand_buffers() -> None:
    """The lowered body cannot answer this, which is why the cut may not ask it.

    ``Body.loads`` walks ``Stmt.nested()``, and a Fold's operand edges are not nested statements —
    so a region that keeps its cone as a term hides every buffer beneath that edge. A cut that
    declared the lowered view named fewer graph inputs than the kernel the materializer built from
    the same tree went on to read; the workspace producers lost their only consumer edge, were
    pruned as orphans, and the launch asked for a buffer nothing had allocated
    (``KeyError: '<node>__place_<token>_0'`` on DeepSeek-V4's TP8xPP2 boot)."""
    root = _root_holding_a_cone()

    assert loaded_buffers(root) == {"ws"}


def test_loaded_buffers_reads_a_contraction_through_its_edges() -> None:
    """A contraction's ``nested()`` is empty by design — its algebra IS its operand edges."""
    from emmy.compiler.ir.pure import Channel

    contraction = Fold.contraction(
        k_axis=Axis("k", Dim(8)),
        a=Load(name="av", input="x", index=(Var("m"), Var("k"))),
        channels=(Channel(b=Load(name="bv", input="w", index=(Var("n"), Var("k"))), acc="acc"),),
    )

    assert contraction.nested() == ()
    assert loaded_buffers(contraction) == {"x", "w"}
