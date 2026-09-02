"""``Fold.lower`` — the nest a term flattens to, placed by ONE rule.

A term is materialized at the SHALLOWEST scope on its path that binds every coordinate it is
evaluated over (:attr:`Fold.free_axes`). The scopes are the loops the term opens for the free
coordinates its caller left unbound (``bound``) and the reduce loop of every term on the way down.
Nothing is walked for free names: the declaration is compared against the scopes above, and the
same comparison hoists an operand ahead of its reader's reduce loop, ahead of an output sweep, or
past the very term that reads it.

These pin the binding contract (``None`` binds every free coordinate — the open body; ``frozenset()``
binds none — the closed program), the loop order (the tree's, not the caller's), the placement rule
at each of its three depths, where a boundary store lands (after the term defining its value), and
the memo per binding.
"""

from __future__ import annotations

from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.pure import Fold, Lambda, M
from emmy.compiler.ir.stmt import Assign, Body, Load, Loop, OutputSpec, Write

M_AXIS, N_AXIS, K_AXIS = Axis("m", Dim(8)), Axis("n", Dim(4)), Axis("k", Dim(16))
SCOPE = (M_AXIS, N_AXIS, K_AXIS)


def _slab(name: str, buffer: str, *index: str) -> Fold:
    return Fold.slab(Load(name=name, input=buffer, index=tuple(Var(v) for v in index)), SCOPE)


def _reduce(operands: tuple[Fold, ...], body: tuple, acc: str, op: str = "add") -> Fold:
    """A reducing term over ``k``: one ``⊕`` state, the lift's results its per-step value."""
    bound = tuple(name for edge in operands for name in edge.exposes)
    init, combine = M(op, names=(acc,))
    lift = Lambda.closing((K_AXIS.name, *bound), Body(body), (f"{acc}__v",))
    return Fold(axes=(K_AXIS,), operands=operands, lift=lift, init=init, combine=combine)


def _matmul(a: Fold, b: Fold) -> Fold:
    return _reduce((a, b), (Assign(name="acc__v", op="multiply", args=(a.exposes[0], b.exposes[0])),), "acc")


def _normalized_sum() -> tuple[Fold, Fold]:
    """``Σ_k exp(x[m,n,k] − t[m])`` over a row total ``t[m] = Σ_k y[m,k]`` — the total is an
    operand of the swept sum, evaluated over ``m`` alone."""
    total = _reduce((_slab("y", "y", "m", "k"),), (Assign(name="tot__v", op="copy", args=("y",)),), "tot")
    x = _slab("x", "x", "m", "n", "k")
    body = (Assign(name="d", op="subtract", args=("x", "tot")), Assign(name="acc__v", op="exp", args=("d",)))
    return total, _reduce((x, total), body, "acc")


def _chain(body) -> list[str]:
    """The axes of the trailing loops, outermost first — the nest a body spells."""
    out = []
    while body and isinstance(body[-1], Loop):
        out.append(body[-1].axis.name)
        body = body[-1].body
    return out


# --- the binding contract ------------------------------------------------------------------------ #


def test_the_open_body_binds_every_free_coordinate() -> None:
    """``lower()`` is the body a term spells inside a scope binding all of its coordinates: the
    reduce loop alone, ``m`` and ``n`` read free."""
    mm = _matmul(_slab("l", "x", "m", "k"), _slab("r", "w", "k", "n"))
    assert mm.lower() == mm.lower(mm.free_axes)
    assert _chain(mm.lower()) == ["k"]


def test_the_closed_program_opens_a_loop_per_free_coordinate_in_declaration_order() -> None:
    """Nothing bound: one plain loop per free coordinate, ordered as the tree first declares them
    when no coordinate is shared more than another, the reduce loop innermost with the whole step
    inside it."""
    mm = _matmul(_slab("l", "x", "m", "k"), _slab("r", "w", "k", "n"))
    assert _chain(mm.lower(frozenset())) == ["m", "n", "k"]
    (outer,) = mm.lower(frozenset())
    assert [type(stmt).__name__ for stmt in outer.body[-1].body[-1].body] == ["Load", "Load", "Assign", "Accum"]
    # The order is the TREE's: the first-declared coordinate opens first, whichever the caller's grid says.
    swapped = _matmul(_slab("r", "w", "n", "k"), _slab("l", "x", "k", "m"))
    assert _chain(swapped.lower(frozenset())) == ["n", "m", "k"]


# --- the placement rule, at each depth ----------------------------------------------------------- #


def test_an_operand_that_does_not_index_the_reduce_axis_lands_ahead_of_the_loop() -> None:
    scale = _slab("s", "scale", "m")
    scaled = _reduce((_slab("l", "x", "m", "k"), scale), (Assign(name="acc__v", op="multiply", args=("l", "s")),), "acc")
    assert [type(stmt).__name__ for stmt in scaled.lower()] == ["Load", "Loop"]
    assert scaled.lower()[0].input == "scale"


def test_a_term_lands_at_the_shallowest_scope_binding_its_coordinates() -> None:
    """The row total is an operand of the swept sum, but it is evaluated over ``m`` alone, so the
    closed program materializes it under ``m`` and ahead of the ``n`` loop — not inside ``n``
    where its reader sits."""
    total, swept = _normalized_sum()
    (m_loop,) = swept.lower(frozenset())
    assert m_loop.axis.name == "m"
    hoisted, n_loop = m_loop.body
    assert hoisted == total.lower(frozenset({"m"}))[0] and n_loop.axis.name == "n"
    assert _chain(n_loop.body) == ["k"]


def test_the_callers_binding_decides_what_hoists() -> None:
    """The same term under two grids: binding ``m`` leaves ``n`` to open and the total ahead of
    it; binding ``n`` leaves ``m`` to open and pins the total under it, with no ``n`` loop at all."""
    total, swept = _normalized_sum()
    ahead, n_loop = swept.lower(frozenset({"m"}))
    assert ahead == total.lower()[0] and n_loop.axis.name == "n"
    (m_loop,) = swept.lower(frozenset({"n"}))
    assert m_loop.axis.name == "m" and _chain(m_loop.body) == ["k"] and len(m_loop.body) == 2


def test_an_operand_hoists_past_the_term_that_reads_it() -> None:
    """The scale is read by a zero-axis cone that rides the reduce loop; the scale itself is
    evaluated over ``m`` alone, so it leaves the loop even though its reader stays inside."""
    x, scale = _slab("l", "x", "m", "k"), _slab("s", "scale", "m")
    cone = Fold(operands=(x, scale), lift=Lambda.closing(("l", "s"), Body((Assign(name="y", op="multiply", args=("l", "s")),)), ("y",)))
    mx = _reduce((cone,), (Assign(name="mx__v", op="copy", args=("y",)),), "mx", op="maximum")
    ahead, loop = mx.lower()
    assert ahead.input == "scale"
    assert [type(stmt).__name__ for stmt in loop.body] == ["Load", "Assign", "Assign", "Accum"]


def test_sweeps_no_term_shares_are_sibling_loops_and_a_reader_makes_them_a_chain() -> None:
    """Two reduces over different output coordinates under a wrapper with no step of its own take
    their own paths — sibling ``q`` and ``n`` loops under ``m``. A reader evaluated over both
    puts them on ITS path, so what it reads is in scope: the second is recomputed inside the first."""
    q = Axis("q", Dim(2))
    scope = (M_AXIS, N_AXIS, q, K_AXIS)
    a = Fold.slab(Load(name="a", input="x", index=(Var("m"), Var("q"), Var("k"))), scope)
    b = Fold.slab(Load(name="b", input="y", index=(Var("m"), Var("n"), Var("k"))), scope)
    over_q = _reduce((a,), (Assign(name="sq__v", op="copy", args=("a",)),), "sq")
    over_n = _reduce((b,), (Assign(name="sn__v", op="copy", args=("b",)),), "sn")
    forest = Fold(operands=(over_q, over_n), lift=Lambda.closing(("sq", "sn"), Body(), ("sq", "sn")))
    (m_loop,) = forest.lower(frozenset())
    assert [loop.axis.name for loop in m_loop.body] == ["q", "n"]
    assert all(_chain(loop.body) == ["k"] for loop in m_loop.body)

    total = Body((Assign(name="t", op="add", args=("sq", "sn")),))
    reader = Fold(operands=(over_q, over_n), lift=Lambda.closing(("sq", "sn"), total, ("t",)))
    (m_loop,) = reader.lower(frozenset())
    (q_loop,) = m_loop.body
    assert q_loop.axis.name == "q" and [type(stmt).__name__ for stmt in q_loop.body] == ["Loop", "Loop"]
    assert q_loop.body[-1].axis.name == "n" and [type(stmt).__name__ for stmt in q_loop.body[-1].body] == ["Loop", "Assign"]


# --- boundary stores ----------------------------------------------------------------------------- #


def test_a_store_follows_the_term_defining_its_value_at_that_terms_scope() -> None:
    """Closed, the ``[m, n]`` store rides the ``n`` loop after the reduce; at kernel scope, where
    the grid binds both, it is the kernel tail."""
    mm = _matmul(_slab("l", "x", "m", "k"), _slab("r", "w", "k", "n"))
    store = OutputSpec(write=Write(output="out", index=(Var("m"), Var("n")), value="acc"))
    (m_loop,) = mm.lower(frozenset(), (store,))
    (n_loop,) = m_loop.body
    assert [type(stmt).__name__ for stmt in n_loop.body] == ["Loop", "Write"] and n_loop.body[-1] is store.write
    assert mm.lower(mm.free_axes, (store,)) == Body((*mm.lower(), store.write))


def test_a_sweep_store_rides_the_loop_the_term_opened() -> None:
    """At kernel scope the term opens its output sweep itself and the store follows the swept sum
    inside it; the row total, evaluated over ``m`` alone, stays ahead."""
    total, swept = _normalized_sum()
    store = OutputSpec(write=Write(output="out", index=(Var("m"), Var("n")), value="acc"), sweep=N_AXIS)
    ahead, sweep = swept.lower(frozenset({"m"}), (store,))
    assert ahead == total.lower()[0] and sweep.axis is N_AXIS
    assert [type(stmt).__name__ for stmt in sweep.body] == ["Loop", "Write"]


def test_a_store_lands_in_the_sibling_loop_of_its_term() -> None:
    q = Axis("q", Dim(2))
    scope = (M_AXIS, N_AXIS, q, K_AXIS)
    a = Fold.slab(Load(name="a", input="x", index=(Var("m"), Var("q"), Var("k"))), scope)
    b = Fold.slab(Load(name="b", input="y", index=(Var("m"), Var("n"), Var("k"))), scope)
    over_q = _reduce((a,), (Assign(name="sq__v", op="copy", args=("a",)),), "sq")
    over_n = _reduce((b,), (Assign(name="sn__v", op="copy", args=("b",)),), "sn")
    forest = Fold(operands=(over_q, over_n), lift=Lambda.closing(("sq", "sn"), Body(), ("sq", "sn")))
    stores = (
        OutputSpec(write=Write(output="oq", index=(Var("m"), Var("q")), value="sq")),
        OutputSpec(write=Write(output="on", index=(Var("m"), Var("n")), value="sn")),
    )
    (m_loop,) = forest.lower(frozenset(), stores)
    by_axis = {loop.axis.name: loop for loop in m_loop.body}
    assert tuple(by_axis["q"].body)[-1] is stores[0].write and tuple(by_axis["n"].body)[-1] is stores[1].write


def test_a_broadcast_store_opens_the_sweep_axis_its_spec_names() -> None:
    """``o[m, j] = tot`` with nothing computed over ``j``: the store alone is evaluated over it, so
    the term opens a ``j`` loop under the total, from the spec's axis."""
    j = Axis("j", Dim(3))
    total = _reduce((_slab("y", "y", "m", "k"),), (Assign(name="tot__v", op="copy", args=("y",)),), "tot")
    store = OutputSpec(write=Write(output="o", index=(Var("m"), Var("j")), value="tot"), sweep=j)
    (m_loop,) = total.lower(frozenset(), (store,))
    reduce_loop, j_loop = m_loop.body
    assert reduce_loop.axis is K_AXIS and j_loop.axis is j and tuple(j_loop.body) == (store.write,)


def test_an_observed_store_rides_the_reduce_loop_after_the_observer() -> None:
    init, combine = M("add", names=("acc",))
    observe = Lambda(params=("k", "acc"), body=Body((Assign(name="acc__obs", op="copy", args=("acc",)),)), results=("acc__obs",))
    lift = Lambda.closing(("k", "y"), Body((Assign(name="acc__v", op="copy", args=("y",)),)), ("acc__v",))
    scan = Fold(axes=(K_AXIS,), operands=(_slab("y", "y", "m", "k"),), lift=lift, init=init, combine=combine, observe=observe)
    store = OutputSpec(write=Write(output="o", index=(Var("m"), Var("k")), value="acc__obs"))
    (loop,) = scan.lower(scan.free_axes, (store,))
    assert [type(stmt).__name__ for stmt in loop.body] == ["Load", "Assign", "Accum", "Assign", "Write"]


# --- the memo ------------------------------------------------------------------------------------ #


def test_lowering_is_memoized_per_binding() -> None:
    mm = _matmul(_slab("l", "x", "m", "k"), _slab("r", "w", "k", "n"))
    assert mm.lower() is mm.lower()
    assert mm.lower(frozenset()) is mm.lower(frozenset())
    assert mm.lower() is not mm.lower(frozenset())
