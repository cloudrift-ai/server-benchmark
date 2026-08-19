"""``merge_stmts`` — the state⊕state combine's ONE statement realization.

The cross-partition combine used to be a renderable ``Stmt`` carrying its own merge program and
its own neutral elements. Both were second spellings of things the IR already has: the program is
the fold's stored ``combine`` :class:`Lambda`, and the neutral element is what the identity
placement already reads off an ``Accum``. It is now a function over that stored combine — these
tests pin what it renders and why the ``Accum`` form is the load-bearing part.
"""

from __future__ import annotations

from emmy.compiler.ir.pure import Lambda, M, exp_combine_states, merge_stmts
from emmy.compiler.ir.stmt import Accum, Assign, Body


def _degenerate() -> Lambda:
    """The two-component additive/max monoid — ``M``'s componentwise combine."""
    _, combine = M("add", "maximum", names=("acc0", "acc1"))
    return combine


def _twisted() -> Lambda:
    """The exp-family ``(m, l)`` carrier's generated state⊕state combine."""
    names, other = ("m", "l"), ("m__o", "l__o")
    return Lambda(params=names + other, body=Body(exp_combine_states(names, other)), results=names)


def test_every_component_leaves_as_an_accum() -> None:
    """Each state component is reassigned by an ``Accum`` — the form that folds in place AND whose
    ``op.identity`` the ONE identity placement seeds with. A degenerate ⊕ needs no temps."""
    stmts = merge_stmts(_degenerate(), ("acc0__p", "acc1__p"))
    assert [(s.name, s.op.name, s.value) for s in stmts] == [("acc0", "add", "acc0__p"), ("acc1", "maximum", "acc1__p")]
    assert [s.op.identity for s in stmts] == [0.0, -1e30], "the seed the identity placement will emit"


def test_twisted_renders_rescale_temps_plus_base_accums() -> None:
    """A twisted combine renders as its ψ-rescale ``Assign`` temps followed by one ``Accum`` per
    component — the pivot a plain ``maximum`` fold, the accumulator channels ``base``-redirected
    onto their rescaled old state."""
    stmts = merge_stmts(_twisted(), ("m__p", "l__p"))
    assert all(isinstance(s, (Assign, Accum)) for s in stmts)
    accums = [s for s in stmts if isinstance(s, Accum)]
    assert {a.name for a in accums} == {"m", "l"}
    pivot = next(a for a in accums if a.name == "m")
    assert pivot.op.name == "maximum" and pivot.base is None
    channel = next(a for a in accums if a.name == "l")
    assert channel.op.name == "add" and channel.base is not None, "the ψ rescale redirects the left operand"


def test_twisted_temps_are_keyed_on_the_partial_being_merged() -> None:
    """Two merges of DIFFERENT partials into the same state must not share temp names — a REG-tree
    fold emits one per copy back to back, and colliding temps would cross-wire them."""
    temps_a = {s.name for s in merge_stmts(_twisted(), ("m__r1", "l__r1")) if isinstance(s, Assign)}
    temps_b = {s.name for s in merge_stmts(_twisted(), ("m__r2", "l__r2")) if isinstance(s, Assign)}
    assert temps_a and not (temps_a & temps_b)


def test_reads_are_visible_through_ordinary_stmt_deps() -> None:
    """The merge's reads reach read counters / liveness / rename through the per-stmt ``deps()``,
    with no special channel: the partial is read, the program's own temps are not."""
    stmts = merge_stmts(_twisted(), ("m__p", "l__p"))
    reads = {r for s in stmts for r in s.deps()} - {s.name for s in stmts}
    assert reads == {"m__p", "l__p"}
