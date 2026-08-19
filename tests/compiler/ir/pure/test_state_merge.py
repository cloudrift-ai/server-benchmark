"""``StateMerge`` — a PURE term with exactly one statement realization.

The state⊕state combine used to travel as a renderable ``Stmt`` carrying its own merge program
and its own neutral elements. Both were second spellings of things the IR already has: the program
is the fold's stored ``combine`` :class:`Lambda`, and the neutral element is what the identity
placement already reads off an ``Accum``. These tests pin the replacement contract — the term
carries the algebra, :meth:`StateMerge.stmts` is where it becomes statements, and the statements
it produces are ordinary ones that the generic machinery can see through.
"""

from __future__ import annotations

import pytest

from emmy.compiler.ir.pure import Lambda, M, StateMerge, exp_combine_states
from emmy.compiler.ir.stmt import Accum, Assign, Body


def _degenerate() -> Lambda:
    """The two-component additive/max monoid — ``M``'s componentwise combine."""
    _, combine = M("add", "maximum", names=("acc0", "acc1"))
    return combine


def _twisted() -> Lambda:
    """The exp-family ``(m, l)`` carrier's generated state⊕state combine."""
    names, other = ("m", "l"), ("m__o", "l__o")
    return Lambda(params=names + other, body=Body(exp_combine_states(names, other)), results=names)


def test_of_renames_only_the_second_operand() -> None:
    """``of`` is the fold's own combine with its right operand pointed at the partial actually
    being merged — the state (and so the results) is untouched."""
    sm = StateMerge.of(_degenerate(), ("acc0__r1", "acc1__r1"))
    assert sm.state == ("acc0", "acc1")
    assert sm.state_b == ("acc0__r1", "acc1__r1")


def test_stmts_realizes_every_component_as_an_accum() -> None:
    """Every state component leaves as an ``Accum`` — the form that reassigns in place AND whose
    ``op.identity`` the ONE identity placement seeds with. A degenerate ⊕ needs no temps."""
    stmts = StateMerge.of(_degenerate(), ("acc0__p", "acc1__p")).stmts()
    assert [type(s).__name__ for s in stmts] == ["Accum", "Accum"]
    assert [(s.name, s.op.name, s.value) for s in stmts] == [("acc0", "add", "acc0__p"), ("acc1", "maximum", "acc1__p")]
    assert [s.op.identity for s in stmts] == [0.0, -1e30], "the seed the identity placement will emit"


def test_twisted_stmts_are_rescale_temps_plus_base_accums() -> None:
    """A twisted combine realizes as its ψ-rescale ``Assign`` temps followed by one ``Accum`` per
    component — the pivot a plain ``maximum`` fold, the accumulator channels ``base``-redirected
    onto their rescaled old state."""
    stmts = StateMerge.of(_twisted(), ("m__p", "l__p")).stmts()
    accums = [s for s in stmts if isinstance(s, Accum)]
    assert {a.name for a in accums} == {"m", "l"}
    assert all(isinstance(s, (Assign, Accum)) for s in stmts)
    pivot = next(a for a in accums if a.name == "m")
    assert pivot.op.name == "maximum" and pivot.base is None
    channel = next(a for a in accums if a.name == "l")
    assert channel.op.name == "add" and channel.base is not None, "the ψ rescale redirects the left operand"


def test_twisted_temps_are_keyed_on_the_partial_being_merged() -> None:
    """Two merges of DIFFERENT partials into the same state must not share temp names — a REG-tree
    fold emits one per copy back to back, and colliding temps would cross-wire them."""
    a = StateMerge.of(_twisted(), ("m__r1", "l__r1")).stmts()
    b = StateMerge.of(_twisted(), ("m__r2", "l__r2")).stmts()
    temps_a = {s.name for s in a if isinstance(s, Assign)}
    temps_b = {s.name for s in b if isinstance(s, Assign)}
    assert temps_a and not (temps_a & temps_b)


def test_reads_are_visible_through_ordinary_stmt_deps() -> None:
    """The merge's reads reach read counters / liveness / rename through the per-stmt ``deps()``,
    with no special channel: the partial is read, the program's own temps are not."""
    stmts = StateMerge.of(_twisted(), ("m__p", "l__p")).stmts()
    defined = {s.name for s in stmts}
    reads = {r for s in stmts for r in s.deps()} - defined
    assert reads == {"m__p", "l__p"}


def test_formation_rejects_a_lambda_that_is_not_state_times_state() -> None:
    """The S × S → S shape is the invariant consumers read ``state`` / ``state_b`` back off — a
    lambda that does not have it is rejected at construction, not misread later."""
    with pytest.raises(ValueError, match="S . S"):
        StateMerge(Lambda(params=("a", "b", "c"), body=Body(()), results=("a",)))
    with pytest.raises(ValueError, match="reassign its state"):
        StateMerge(Lambda(params=("a", "b"), body=Body((Assign(name="c", op="add", args=("a", "b")),)), results=("c",)))
