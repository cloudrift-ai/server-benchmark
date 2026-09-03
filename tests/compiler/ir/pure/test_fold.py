"""``Fold.lower`` — the nest a term flattens to, placed by ONE rule.

A term is materialized at the SHALLOWEST scope on its path that binds every coordinate it is
evaluated over (:attr:`Fold.free_axes`). The scopes are the loops the term opens for the free
coordinates its caller left unbound (``bound``) and the reduce loop of every term on the way down.
Nothing is walked for free names: the declaration is compared against the scopes above, and the
same comparison hoists an operand ahead of its reader's reduce loop, ahead of an output sweep, or
past the very term that reads it.

These pin the binding contract (``None`` binds every free coordinate — the open body; ``frozenset()``
binds none — the closed program), the loop order (the tree's, not the caller's), the placement rule
at each of its three depths, where a boundary store lands (after the term defining its value), the
memo per binding, and ``Fold.merge`` — the stored combine applied at a second state, of which the
serial step is the instance at the injected singleton.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.pure import Fold, Lambda
from emmy.compiler.ir.pure.twist import SOFTMAX
from emmy.compiler.ir.stmt import Accum, Assign, Body, Const, Loop, OutputSpec, Write
from tests.compiler.terms import contraction, reduction, slab

M_AXIS, N_AXIS, K_AXIS = Axis("m", Dim(8)), Axis("n", Dim(4)), Axis("k", Dim(16))
SCOPE = (M_AXIS, N_AXIS, K_AXIS)


def _reduce(operands: tuple[Fold, ...], body: tuple, acc: str, op: str = "add") -> Fold:
    """A one-state reducing term over ``k``."""
    return reduction(K_AXIS, operands, body, (acc,), op)


def _matmul(a: Fold, b: Fold) -> Fold:
    return contraction(K_AXIS, a, (b, "acc"))


def _normalized_sum() -> tuple[Fold, Fold]:
    """``Σ_k exp(x[m,n,k] − t[m])`` over a row total ``t[m] = Σ_k y[m,k]`` — the total is an
    operand of the swept sum, evaluated over ``m`` alone."""
    total = _reduce((slab("y", "y", "m", "k"),), (Assign(name="tot__v", op="copy", args=("y",)),), "tot")
    x = slab("x", "x", "m", "n", "k")
    body = (Assign(name="d", op="subtract", args=("x", "tot")), Assign(name="acc__v", op="exp", args=("d",)))
    return total, _reduce((x, total), body, "acc")


def _chain(body) -> list[str]:
    """The axes of the trailing loops, outermost first — the nest a body spells."""
    out = []
    while body and isinstance(body[-1], Loop):
        out.append(body[-1].axis.name)
        body = body[-1].body
    return out


# --- formation: a bilinear term orients itself ---------------------------------------------------- #


def test_a_bilinear_term_puts_its_k_last_operand_first_at_formation() -> None:
    """``operands[0]`` IS A by construction: with one product, the slab whose reduction axis is
    its last index coordinate leads, and the lift's params move with the operands."""
    w, x = slab("r", "w", "k", "n"), slab("l", "x", "m", "k")
    mm = _matmul(w, x)
    assert mm.operands == (x, w) and mm.lift.params == ("k", "l", "r")
    assert mm.as_contraction() is not None and mm.as_contraction().left == "m"


def test_a_multi_channel_term_puts_the_shared_operand_first_at_formation() -> None:
    """With several products A is the argument they share, whatever the slab layouts say."""
    x, g, u = slab("l", "x", "k", "m"), slab("g", "wg", "n", "k"), slab("u", "wu", "n", "k")
    init, combine = (0.0, 0.0), Lambda.componentwise(("add", "add"), ("acc_g", "acc_u"))
    body = (Assign(name="acc_g__v", op="multiply", args=("g", "l")), Assign(name="acc_u__v", op="multiply", args=("u", "l")))
    lift = Lambda.closing(("k", "g", "u", "l"), Body(body), ("acc_g__v", "acc_u__v"))
    fold = Fold(operands=(g, u, x), lift=lift, init=init, combine=combine)
    assert fold.operands == (x, g, u) and fold.lift.params == ("k", "l", "g", "u")
    assert fold.as_contraction() is not None


# --- the binding contract ------------------------------------------------------------------------ #


def test_the_open_body_binds_every_free_coordinate() -> None:
    """``lower()`` is the body a term spells inside a scope binding all of its coordinates: the
    reduce loop alone, ``m`` and ``n`` read free."""
    mm = _matmul(slab("l", "x", "m", "k"), slab("r", "w", "k", "n"))
    assert mm.lower(axes=SCOPE) == mm.lower(mm.free_axes, axes=SCOPE)
    assert _chain(mm.lower(axes=SCOPE)) == ["k"]


def test_the_closed_program_opens_a_loop_per_free_coordinate_in_declaration_order() -> None:
    """Nothing bound: one plain loop per free coordinate, ordered as the tree first declares them
    when no coordinate is shared more than another, the reduce loop innermost with the whole step
    inside it."""
    mm = _matmul(slab("l", "x", "m", "k"), slab("r", "w", "k", "n"))
    assert _chain(mm.lower(frozenset(), axes=SCOPE)) == ["m", "n", "k"]
    (outer,) = mm.lower(frozenset(), axes=SCOPE)
    assert [type(stmt).__name__ for stmt in outer.body[-1].body[-1].body] == ["Load", "Load", "Assign", "Accum"]
    # The order is the TREE's: the first-declared coordinate opens first, whichever the caller's grid says.
    swapped = _matmul(slab("r", "w", "n", "k"), slab("l", "x", "k", "m"))
    assert _chain(swapped.lower(frozenset(), axes=SCOPE)) == ["n", "m", "k"]


# --- the placement rule, at each depth ----------------------------------------------------------- #


def test_an_operand_that_does_not_index_the_reduce_axis_lands_ahead_of_the_loop() -> None:
    scale = slab("s", "scale", "m")
    scaled = _reduce((slab("l", "x", "m", "k"), scale), (Assign(name="acc__v", op="multiply", args=("l", "s")),), "acc")
    assert [type(stmt).__name__ for stmt in scaled.lower(axes=SCOPE)] == ["Load", "Loop"]
    assert scaled.lower(axes=SCOPE)[0].input == "scale"


def test_a_term_lands_at_the_shallowest_scope_binding_its_coordinates() -> None:
    """The row total is an operand of the swept sum, but it is evaluated over ``m`` alone, so the
    closed program materializes it under ``m`` and ahead of the ``n`` loop — not inside ``n``
    where its reader sits."""
    total, swept = _normalized_sum()
    (m_loop,) = swept.lower(frozenset(), axes=SCOPE)
    assert m_loop.axis.name == "m"
    hoisted, n_loop = m_loop.body
    assert hoisted == total.lower(frozenset({"m"}), axes=SCOPE)[0] and n_loop.axis.name == "n"
    assert _chain(n_loop.body) == ["k"]


def test_the_callers_binding_decides_what_hoists() -> None:
    """The same term under two grids: binding ``m`` leaves ``n`` to open and the total ahead of
    it; binding ``n`` leaves ``m`` to open and pins the total under it, with no ``n`` loop at all."""
    total, swept = _normalized_sum()
    ahead, n_loop = swept.lower(frozenset({"m"}), axes=SCOPE)
    assert ahead == total.lower(axes=SCOPE)[0] and n_loop.axis.name == "n"
    (m_loop,) = swept.lower(frozenset({"n"}), axes=SCOPE)
    assert m_loop.axis.name == "m" and _chain(m_loop.body) == ["k"] and len(m_loop.body) == 2


def test_an_operand_hoists_past_the_term_that_reads_it() -> None:
    """The scale is read by a zero-axis cone that rides the reduce loop; the scale itself is
    evaluated over ``m`` alone, so it leaves the loop even though its reader stays inside."""
    x, scale = slab("l", "x", "m", "k"), slab("s", "scale", "m")
    cone = Fold(operands=(x, scale), lift=Lambda.closing(("l", "s"), Body((Assign(name="y", op="multiply", args=("l", "s")),)), ("y",)))
    mx = _reduce((cone,), (Assign(name="mx__v", op="copy", args=("y",)),), "mx", op="maximum")
    ahead, loop = mx.lower(axes=SCOPE)
    assert ahead.input == "scale"
    assert [type(stmt).__name__ for stmt in loop.body] == ["Load", "Assign", "Assign", "Accum"]


def test_sweeps_no_term_shares_are_sibling_loops_and_a_reader_makes_them_a_chain() -> None:
    """Two reduces over different output coordinates under a wrapper with no step of its own take
    their own paths — sibling ``q`` and ``n`` loops under ``m``. A reader evaluated over both
    puts them on ITS path, so what it reads is in scope: the second is recomputed inside the first."""
    q = Axis("q", Dim(2))
    scope = (M_AXIS, N_AXIS, q, K_AXIS)
    a, b = slab("a", "x", "m", "q", "k"), slab("b", "y", "m", "n", "k")
    over_q = _reduce((a,), (Assign(name="sq__v", op="copy", args=("a",)),), "sq")
    over_n = _reduce((b,), (Assign(name="sn__v", op="copy", args=("b",)),), "sn")
    forest = Fold(operands=(over_q, over_n), lift=Lambda.closing(("sq", "sn"), Body(), ("sq", "sn")))
    (m_loop,) = forest.lower(frozenset(), axes=scope)
    assert [loop.axis.name for loop in m_loop.body] == ["q", "n"]
    assert all(_chain(loop.body) == ["k"] for loop in m_loop.body)

    total = Body((Assign(name="t", op="add", args=("sq", "sn")),))
    reader = Fold(operands=(over_q, over_n), lift=Lambda.closing(("sq", "sn"), total, ("t",)))
    (m_loop,) = reader.lower(frozenset(), axes=scope)
    (q_loop,) = m_loop.body
    assert q_loop.axis.name == "q" and [type(stmt).__name__ for stmt in q_loop.body] == ["Loop", "Loop"]
    assert q_loop.body[-1].axis.name == "n" and [type(stmt).__name__ for stmt in q_loop.body[-1].body] == ["Loop", "Assign"]


# --- boundary stores ----------------------------------------------------------------------------- #


def test_a_store_follows_the_term_defining_its_value_at_that_terms_scope() -> None:
    """Closed, the ``[m, n]`` store rides the ``n`` loop after the reduce; at kernel scope, where
    the grid binds both, it is the kernel tail."""
    mm = _matmul(slab("l", "x", "m", "k"), slab("r", "w", "k", "n"))
    store = OutputSpec(write=Write(output="out", index=(Var("m"), Var("n")), value="acc"))
    (m_loop,) = mm.lower(frozenset(), (store,), axes=SCOPE)
    (n_loop,) = m_loop.body
    assert [type(stmt).__name__ for stmt in n_loop.body] == ["Loop", "Write"] and n_loop.body[-1] == store.write
    assert mm.lower(mm.free_axes, (store,), axes=SCOPE) == Body((*mm.lower(axes=SCOPE), store.write))


def test_a_sweep_store_rides_the_loop_the_term_opened() -> None:
    """At kernel scope the term opens its output sweep itself and the store follows the swept sum
    inside it; the row total, evaluated over ``m`` alone, stays ahead."""
    total, swept = _normalized_sum()
    store = OutputSpec(write=Write(output="out", index=(Var("m"), Var("n")), value="acc"), sweep=(N_AXIS,))
    ahead, sweep = swept.lower(frozenset({"m"}), (store,), axes=SCOPE)
    assert ahead == total.lower(axes=SCOPE)[0] and sweep.axis is N_AXIS
    assert [type(stmt).__name__ for stmt in sweep.body] == ["Loop", "Write"]


def test_a_store_lands_in_the_sibling_loop_of_its_term() -> None:
    q = Axis("q", Dim(2))
    scope = (M_AXIS, N_AXIS, q, K_AXIS)
    a, b = slab("a", "x", "m", "q", "k"), slab("b", "y", "m", "n", "k")
    over_q = _reduce((a,), (Assign(name="sq__v", op="copy", args=("a",)),), "sq")
    over_n = _reduce((b,), (Assign(name="sn__v", op="copy", args=("b",)),), "sn")
    forest = Fold(operands=(over_q, over_n), lift=Lambda.closing(("sq", "sn"), Body(), ("sq", "sn")))
    stores = (
        OutputSpec(write=Write(output="oq", index=(Var("m"), Var("q")), value="sq")),
        OutputSpec(write=Write(output="on", index=(Var("m"), Var("n")), value="sn")),
    )
    (m_loop,) = forest.lower(frozenset(), stores, axes=scope)
    by_axis = {loop.axis.name: loop for loop in m_loop.body}
    assert tuple(by_axis["q"].body)[-1] == stores[0].write and tuple(by_axis["n"].body)[-1] == stores[1].write


def test_a_broadcast_store_opens_the_sweep_axis_its_spec_names() -> None:
    """``o[m, j] = tot`` with nothing computed over ``j``: the store alone is evaluated over it, so
    the term opens a ``j`` loop under the total, from the spec's axis."""
    j = Axis("j", Dim(3))
    total = _reduce((slab("y", "y", "m", "k"),), (Assign(name="tot__v", op="copy", args=("y",)),), "tot")
    store = OutputSpec(write=Write(output="o", index=(Var("m"), Var("j")), value="tot"), sweep=(j,))
    (m_loop,) = total.lower(frozenset(), (store,), axes=SCOPE)
    reduce_loop, j_loop = m_loop.body
    assert reduce_loop.axis is K_AXIS and j_loop.axis is j and tuple(j_loop.body) == (store.write,)


def test_an_observed_store_rides_the_reduce_loop_after_the_observer() -> None:
    init, combine = (0.0,), Lambda.componentwise(("add",), ("acc",))
    observe = Lambda(params=("k", "acc"), body=Body((Assign(name="acc__obs", op="copy", args=("acc",)),)), results=("acc__obs",))
    lift = Lambda.closing(("k", "y"), Body((Assign(name="acc__v", op="copy", args=("y",)),)), ("acc__v",))
    scan = Fold(operands=(slab("y", "y", "m", "k"),), lift=lift, init=init, combine=combine, observe=observe)
    store = OutputSpec(write=Write(output="o", index=(Var("m"), Var("k")), value="acc__obs"))
    (loop,) = scan.lower(scan.free_axes, (store,), axes=SCOPE)
    assert [type(stmt).__name__ for stmt in loop.body] == ["Load", "Assign", "Accum", "Assign", "Write"]


# --- the merge: the combine applied at a second state -------------------------------------------- #


def _twisted(states: tuple[str, str] = ("m", "l")) -> Fold:
    """The exp-family ``(m, l)`` carrier: the softmax recipe's program over a ``(score, 1)`` singleton."""
    body = Body((Assign(name="s", op="copy", args=("y",)), Const(name="one", value=1.0)))
    lift = Lambda.closing(("k", "y"), body, ("s", "one"))
    return Fold(operands=(slab("y", "y", "m", "k"),), lift=lift, init=(-1e30, 0.0), combine=SOFTMAX.program(states))


def test_a_twisted_state_spelling_never_reaches_the_canonical_form() -> None:
    """The combine's own names — its second operand, its rescale temps — renumber after the term's,
    so two folds equal up to what they called their accumulators have EQUAL canonical forms."""
    assert _twisted().canonical() == _twisted(("p", "q")).canonical()


def test_a_componentwise_merge_is_one_accum_per_state() -> None:
    """Each state component is reassigned by an ``Accum`` — the form that folds in place AND whose
    ``op.identity`` the ONE identity placement seeds with. A planar ⊕ needs no temps."""
    init, combine = (0.0, -1e30), Lambda.componentwise(("add", "maximum"), ("acc0", "acc1"))
    body = Body((Assign(name="a0", op="copy", args=("y",)), Assign(name="a1", op="negative", args=("y",))))
    lift = Lambda.closing(("k", "y"), body, ("a0", "a1"))
    fold = Fold(operands=(slab("y", "y", "m", "k"),), lift=lift, init=init, combine=combine)
    stmts = fold.merge(("acc0__p", "acc1__p"))
    assert [(s.name, s.op.name, s.value) for s in stmts] == [("acc0", "add", "acc0__p"), ("acc1", "maximum", "acc1__p")]
    assert [s.op.identity for s in stmts] == [0.0, -1e30], "the seed the identity placement will emit"
    assert fold.as_reduction().ops is not None and not fold.as_reduction().twisted


def test_a_twisted_merge_is_rescale_temps_then_base_accums() -> None:
    """A twisted combine renders as its ψ-rescale ``Assign`` temps followed by one ``Accum`` per
    component — the pivot a plain ``maximum`` fold, the accumulator channel ``base``-redirected
    onto its rescaled old state."""
    fold = _twisted()
    assert fold.as_reduction().twisted
    stmts = fold.merge(("m__p", "l__p"))
    assert all(isinstance(s, (Assign, Accum)) for s in stmts)
    accums = [s for s in stmts if isinstance(s, Accum)]
    assert {a.name for a in accums} == {"m", "l"}
    pivot = next(a for a in accums if a.name == "m")
    assert pivot.op.name == "maximum" and pivot.base is None
    channel = next(a for a in accums if a.name == "l")
    assert channel.op.name == "add" and channel.base is not None, "the ψ rescale redirects the left operand"


def test_merge_temps_are_keyed_on_the_partial_being_merged() -> None:
    """Two merges of DIFFERENT partials into the same state must not share temp names — a REG-tree
    fold emits one per copy back to back, and colliding temps would cross-wire them."""
    fold = _twisted()
    temps_a = {s.name for s in fold.merge(("m__r1", "l__r1")) if isinstance(s, Assign)}
    temps_b = {s.name for s in fold.merge(("m__r2", "l__r2")) if isinstance(s, Assign)}
    assert temps_a and not (temps_a & temps_b)


def test_merge_reads_are_ordinary_stmt_deps() -> None:
    """The merge's reads reach read counters / liveness / rename through the per-stmt ``deps()``,
    with no special channel: the partial is read, the program's own temps are not."""
    stmts = _twisted().merge(("m__p", "l__p"))
    reads = {r for s in stmts for r in s.deps()} - {s.name for s in stmts}
    assert reads == {"m__p", "l__p"}


def test_the_step_is_the_merge_at_the_injected_singleton() -> None:
    """One derivation: the serial step applies the same program at the lift's results, its
    ``Accum`` forms folding over the reduce axis, after the lift body."""
    fold = _twisted()
    step, merged = fold.step(), fold.merge(fold.lift.results)
    assert tuple(step[: len(fold.lift.body)]) == tuple(fold.lift.body)
    assert tuple(step[len(fold.lift.body) :]) == tuple(replace(s, axes=("k",)) if isinstance(s, Accum) else s for s in merged)


# --- the memo ------------------------------------------------------------------------------------ #


def test_lowering_is_memoized_per_binding() -> None:
    mm = _matmul(slab("l", "x", "m", "k"), slab("r", "w", "k", "n"))
    assert mm.lower(axes=SCOPE) is mm.lower(axes=SCOPE)
    assert mm.lower(frozenset(), axes=SCOPE) is mm.lower(frozenset(), axes=SCOPE)
    assert mm.lower(axes=SCOPE) is not mm.lower(frozenset(), axes=SCOPE)


# --- the binding: positional, the consumer's names; rendering spells the operands ---------------- #


def test_a_consumer_names_what_it_binds_and_rendering_spells_the_operand() -> None:
    """A term's lift params are its own names, bound in order to its operands' result components;
    only the rendered statements (``step`` / ``lower``) read the operands' spelling, and a store
    over a passed-through operand value renders with it. Two spellings of one binding are one
    term: equal canonical forms, equal lowered bodies."""
    y = slab("y", "y", "m", "k")
    lift = Lambda.closing(("k", "value"), Body((Assign(name="acc__v", op="copy", args=("value",)),)), ("acc__v",))
    fold = Fold(operands=(y,), lift=lift, init=(0.0,), combine=Lambda.componentwise(("add",), ("acc",)))
    assert fold.bindings == (("value", y, 0),) and fold.applied.params == ("k", "y")
    (loop,) = fold.lower(axes=SCOPE)
    assert [stmt.args for stmt in loop.body if isinstance(stmt, Assign)] == [("y",)]
    spelled = replace(fold, lift=lift.rename({"value": "y"}))
    assert fold.canonical() == spelled.canonical() and fold.lower(axes=SCOPE) == spelled.lower(axes=SCOPE)
    # A projection passing the state through exposes the state's own name, and its store follows.
    passthrough = Fold(operands=(fold,), lift=Lambda(params=("total",), body=Body(()), results=("total",)))
    assert passthrough.exposes == ("acc",)
    store = OutputSpec(write=Write(output="o", index=(Var("m"),), value="total"))
    (m_loop,) = passthrough.lower(frozenset(), (store,), axes=SCOPE)
    assert m_loop.body[-1] == Write(output="o", index=(Var("m"),), value="acc")
