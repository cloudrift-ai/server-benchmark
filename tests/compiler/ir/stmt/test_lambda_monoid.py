"""The λ-foldMap primitives (1m): :class:`Lambda` formation + α-invariance, the flat ``(init, combine)`` pair (``M``) —
the true ``(init, combine)`` monoid — and the executable SPEC (the denotational
:func:`foldmap_eval` evaluator).

Three test families pin the algebra so every later purification step refactors toward an oracle
that already runs:

- **Formation** — the ``Stmt.pure`` trait gate (conservative default: an effectful stmt kind is
  rejected from a lambda body), results-defined, arity checks.
- **ASSOCIATIVITY** (the split/coop legality certificate) — ``combine(a, combine(b, c)) ==
  combine(combine(a, b), c)`` on random states, for the componentwise monoids AND the generated
  exp/LSE family.
- **AGREEMENT** — ``⟦tree⟧ == lowered loop`` on random inputs: the denotational foldMap of a
  hand-built ``(Monoid, lift)`` equals a mini-interpretation of the corresponding stored
  :class:`Fold`'s derived ``loop`` (sum, x² statistic, dot product, online softmax).
"""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.stmt import Accum, Assign, Body, Init, Lambda, Load, Loop, M, Write
from emmy.compiler.ir.stmt.algebra import Algebra, component_ops, degenerate, eval_lambda, foldmap_eval
from emmy.compiler.ir.stmt.carrier import exp_combine_states
from emmy.compiler.ir.tile.ir import Fold

# --- the mini loop-IR interpreter (the agreement test's right-hand side) ------------------------- #


def _exec(stmts, env: dict, bufs: dict[str, np.ndarray]) -> None:
    for s in stmts:
        if isinstance(s, Load):
            idx = tuple(int(e.eval(env)) for e in s.index)
            env[s.name] = float(bufs[s.input][idx if len(idx) > 1 else idx[0]])
        elif isinstance(s, Assign):
            env[s.name] = s.op(*(env[a] if a in env else float(a) for a in s.args))
        elif isinstance(s, Accum):
            cur = env[s.base] if s.base is not None else env[s.name]
            env[s.name] = s.op(cur, env[s.value])
        elif isinstance(s, Loop):
            _run_loop(s, env, bufs)
        else:  # pragma: no cover — the agreement shapes only use the above
            raise AssertionError(f"mini interpreter: unexpected {type(s).__name__}")


def _run_loop(loop: Loop, env: dict, bufs: dict[str, np.ndarray]) -> dict:
    """Execute a reduce ``Loop`` exactly as ``Loop.render`` spells it: seed each immediate-body
    ``Accum`` at its op identity, then stream the body over the axis."""
    for s in loop.body:
        if isinstance(s, Accum):
            env.setdefault(s.name, s.op.identity)
    for k in range(loop.axis.extent.as_static()):
        env[loop.axis.name] = k
        _exec(loop.body, env, bufs)
    return env


# --- Lambda formation: the Stmt.pure gate + results-defined ------------------------------------- #


def test_lambda_accepts_pure_stmts_and_defines_results() -> None:
    lam = Lambda(
        params=("k", "x"),
        body=Body((Assign(name="x2", op="multiply", args=("x", "x")),)),
        results=("x2",),
    )
    assert lam.results == ("x2",)
    assert lam.free_names() == frozenset()


@pytest.mark.parametrize(
    "stmt",
    [
        Accum(name="acc", value="x", op="add"),
        Write(output="out", index=(Var("m"),), values=("x",)),
        Init(name="acc", identity=0.0, dtype=None),
        Loop(axis=Axis("k", 4), body=Body(())),
    ],
    ids=lambda s: type(s).__name__,
)
def test_lambda_rejects_impure_stmt_kinds(stmt) -> None:
    """The conservative ``Stmt.pure`` default: effectful / scope-bound kinds never enter a stored
    lambda body — the purity-erosion guard."""
    with pytest.raises(ValueError, match="pure"):
        Lambda(params=("x",), body=Body((stmt,)), results=("x",))


def test_lambda_rejects_undefined_results() -> None:
    with pytest.raises(ValueError, match="not defined"):
        Lambda(params=("x",), body=Body(()), results=("y",))


def test_lambda_allows_param_results_and_literal_results() -> None:
    """ι is spelled in the lift: softmax's singleton is ``(x, 1)`` — the literal component has
    no def to name and passes through evaluation verbatim."""
    lam = Lambda(params=("k", "x"), body=Body(()), results=("x", 1.0))
    assert eval_lambda(lam, (0, 3.5)) == (3.5, 1.0)


def test_lambda_free_names_are_the_contextual_read() -> None:
    lam = Lambda(
        params=("x",),
        body=Body((Assign(name="y", op="add", args=("x", "outer")),)),
        results=("y",),
    )
    assert lam.free_names() == frozenset({"outer"})


# --- α-invariance: canonical renumbering -------------------------------------------------------- #


def test_alpha_equality_under_bound_renaming() -> None:
    a = Lambda(
        params=("k", "x"),
        body=Body((Assign(name="t", op="multiply", args=("x", "x")),)),
        results=("t",),
    )
    b = Lambda(
        params=("j", "v"),
        body=Body((Assign(name="sq", op="multiply", args=("v", "v")),)),
        results=("sq",),
    )
    assert a.alpha_eq(b)
    assert a.canonical() == b.canonical()
    assert hash(a.canonical()) == hash(b.canonical())


def test_alpha_inequality_on_structure_and_free_names() -> None:
    sq = Lambda(params=("x",), body=Body((Assign(name="t", op="multiply", args=("x", "x")),)), results=("t",))
    dbl = Lambda(params=("x",), body=Body((Assign(name="t", op="add", args=("x", "x")),)), results=("t",))
    free = Lambda(params=("x",), body=Body((Assign(name="t", op="multiply", args=("x", "w")),)), results=("t",))
    assert not sq.alpha_eq(dbl)
    assert not sq.alpha_eq(free)  # a FREE name must match exactly — never renumbered
    assert sq.canonical().canonical() == sq.canonical()  # idempotent


# --- Monoid: the componentwise constructor + the derived degenerate predicate -------------------- #


def test_monoid_of_is_degenerate_with_identity_seeds() -> None:
    init, combine = M("add", "maximum")
    assert len(init) == 2
    assert init == (0.0, -1e30)
    ops = component_ops(combine)
    assert ops is not None and [o.name for o in ops] == ["add", "maximum"]
    assert degenerate(combine)


def test_fold_arity_mismatch_rejected() -> None:
    from emmy.compiler.ir.axis import Axis as _Axis
    from emmy.compiler.ir.tile.ir import Fold as _Fold

    lam = Lambda(params=("a", "b"), body=Body((Assign(name="c", op="add", args=("a", "b")),)), results=("c",))
    lift = Lambda(params=("k",), body=Body(()), results=(1.0, 1.0))
    with pytest.raises(ValueError, match="S × S → S"):
        _Fold(axis=_Axis("k", 4), lift=lift, init=(0.0, 0.0), combine=lam)


def _lse_monoid(n_expect: int = 0) -> tuple[tuple, Lambda]:
    """The exp/LSE-family ``(init, combine)`` at arity ``2 + n_expect`` — combine generated by the
    carrier machinery (``exp_combine_states``), the ONE stored program."""
    state = ("m", "l") + tuple(f"o{i}" for i in range(n_expect))
    other = tuple(f"{s}__b" for s in state)
    prog = exp_combine_states(state, other)
    combine = Lambda(params=state + other, body=Body(prog), results=state)
    return (float("-inf"), 0.0) + (0.0,) * n_expect, combine


def test_lse_monoid_is_not_degenerate() -> None:
    _, combine = _lse_monoid()
    assert component_ops(combine) is None
    assert not degenerate(combine)


# --- ASSOCIATIVITY (the split/coop legality certificate) ----------------------------------------- #


@pytest.mark.parametrize(
    "pair",
    [M("add"), M("maximum"), M("add", "add"), _lse_monoid(0), _lse_monoid(1)],
    ids=["add", "max", "add2", "lse2", "lse3"],
)
def test_combine_is_associative_on_random_states(pair) -> None:
    init, combine = pair
    rng = np.random.default_rng(7)
    n = len(init)
    for _ in range(50):
        a, b, c = (tuple(rng.normal(size=n) * 3.0) for _ in range(3))
        lhs = eval_lambda(combine, (*a, *eval_lambda(combine, (*b, *c))))
        rhs = eval_lambda(combine, (*eval_lambda(combine, (*a, *b)), *c))
        np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-12)


def test_init_is_the_neutral_element() -> None:
    rng = np.random.default_rng(3)
    for init, combine in (M("add"), M("add", "add"), _lse_monoid(1)):
        s = tuple(rng.normal(size=len(init)))
        left = eval_lambda(combine, (*init, *s))
        right = eval_lambda(combine, (*s, *init))
        np.testing.assert_allclose(left, s, rtol=1e-12)
        np.testing.assert_allclose(right, s, rtol=1e-12)


# --- AGREEMENT: ⟦tree⟧ == lowered loop ----------------------------------------------------------- #


def _id_fold(op: str, value_stmts, acc: str, axis: Axis, value_name: str) -> Fold:
    """A stored degenerate fold in today's vocabulary — ``from_loop`` over the annotated dissolved
    loop (the λ spelling is the ONE spelling since step 7; the byte-identity gate settles at
    construction, so the derived ``loop`` reproduces this exact body)."""
    accum = Accum(name=acc, value=value_name, op=op, axes=(axis.name,))
    loop = Loop(axis=axis, body=Body((*value_stmts, accum)), role=AxisRole.PLANAR, carrier=accum.as_algebra())
    fold = Fold.from_loop(loop)
    assert fold is not None
    return fold


def test_agreement_bare_sum() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=16)
    axis = Axis("k", 16)
    fold = _id_fold("add", (Load(name="x0", input="x", index=(Var("k"),)),), "acc", axis, "x0")
    got = _run_loop(fold.loop, {}, {"x": x})["acc"]
    spec = foldmap_eval(
        *M("add"),
        Lambda(params=("k", "x0"), body=Body(()), results=("x0",)),
        [(k, x[k]) for k in range(16)],
    )
    np.testing.assert_allclose(got, spec[0], rtol=1e-12)
    np.testing.assert_allclose(spec[0], x.sum(), rtol=1e-12)


def test_agreement_square_statistic() -> None:
    """RMSNorm's statistic: ``Σ x²`` — the lift carries the per-element compute."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=12)
    axis = Axis("k", 12)
    stmts = (
        Load(name="x0", input="x", index=(Var("k"),)),
        Assign(name="x2", op="multiply", args=("x0", "x0")),
    )
    fold = _id_fold("add", stmts, "acc", axis, "x2")
    got = _run_loop(fold.loop, {}, {"x": x})["acc"]
    lift = Lambda(
        params=("k", "x0"),
        body=Body((Assign(name="x2", op="multiply", args=("x0", "x0")),)),
        results=("x2",),
    )
    spec = foldmap_eval(*M("add"), lift, [(k, x[k]) for k in range(12)])
    np.testing.assert_allclose(got, spec[0], rtol=1e-12)


def test_agreement_dot_product() -> None:
    """The contraction's scalar cell: ``Σ_k a·b`` — the bilinear lift into the additive monoid."""
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=8), rng.normal(size=8)
    axis = Axis("k", 8)
    stmts = (
        Load(name="a0", input="a", index=(Var("k"),)),
        Load(name="b0", input="b", index=(Var("k"),)),
        Assign(name="v", op="multiply", args=("a0", "b0")),
    )
    fold = _id_fold("add", stmts, "acc", axis, "v")
    got = _run_loop(fold.loop, {}, {"a": a, "b": b})["acc"]
    lift = Lambda(
        params=("k", "a0", "b0"),
        body=Body((Assign(name="v", op="multiply", args=("a0", "b0")),)),
        results=("v",),
    )
    spec = foldmap_eval(*M("add"), lift, [(k, a[k], b[k]) for k in range(8)])
    np.testing.assert_allclose(got, spec[0], rtol=1e-12)
    np.testing.assert_allclose(spec[0], float(a @ b), rtol=1e-12)


def test_agreement_online_softmax() -> None:
    """The twisted fold: today's stored dissolved merge (the streaming step) agrees with the
    TRUE-monoid denotation — combine folded over the ``(x, 1)`` singletons. This is the exact
    consistency 1p makes correct by construction (the serial step derives from combine)."""
    rng = np.random.default_rng(4)
    x = rng.normal(size=20) * 2.0
    axis = Axis("k", 20)
    carrier = Algebra.exp_family("x0", (1.0,), ("m_i", "l_i"))
    step = Body((Load(name="x0", input="x", index=(Var("k"),)), *carrier.merge))
    fold = Fold.from_loop(Loop(axis=axis, body=step, role=AxisRole.TWISTED, carrier=carrier))
    assert fold is not None
    env = _run_loop(fold.loop, {"m_i": float("-inf"), "l_i": 0.0}, {"x": x})
    lift = Lambda(params=("k", "x0"), body=Body(()), results=("x0", 1.0))
    spec = foldmap_eval(*_lse_monoid(0), lift, [(k, x[k]) for k in range(20)])
    np.testing.assert_allclose((env["m_i"], env["l_i"]), spec, rtol=1e-12)
    # And both equal the direct LSE reference.
    m = x.max()
    np.testing.assert_allclose(spec, (m, np.exp(x - m).sum()), rtol=1e-12)


def test_agreement_flash_arity3() -> None:
    """Flash's ``(m, l, O)`` — the expectation channel rides the same combine."""
    rng = np.random.default_rng(5)
    s, v = rng.normal(size=10) * 2.0, rng.normal(size=10)
    axis = Axis("j", 10)
    carrier = Algebra.exp_family("s0", (1.0, "v0"), ("m_i", "l_i", "O_i"))
    step = Body(
        (
            Load(name="s0", input="s", index=(Var("j"),)),
            Load(name="v0", input="v", index=(Var("j"),)),
            *carrier.merge,
        )
    )
    fold = Fold.from_loop(Loop(axis=axis, body=step, role=AxisRole.TWISTED, carrier=carrier))
    assert fold is not None
    env = _run_loop(fold.loop, {"m_i": float("-inf"), "l_i": 0.0, "O_i": 0.0}, {"s": s, "v": v})
    lift = Lambda(params=("j", "s0", "v0"), body=Body(()), results=("s0", 1.0, "v0"))
    spec = foldmap_eval(*_lse_monoid(1), lift, [(j, s[j], v[j]) for j in range(10)])
    np.testing.assert_allclose((env["m_i"], env["l_i"], env["O_i"]), spec, rtol=1e-10)
    m = s.max()
    p = np.exp(s - m)
    np.testing.assert_allclose(spec, (m, p.sum(), float(p @ v)), rtol=1e-10)
