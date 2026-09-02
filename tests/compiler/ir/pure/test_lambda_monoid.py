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

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.pure import Lambda, M
from emmy.compiler.ir.pure.algebra import component_ops, degenerate, eval_lambda, foldmap_eval
from emmy.compiler.ir.pure.carrier import exp_combine_states
from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.stmt import Accum, Assign, Body, Const, Init, Load, Loop, Write
from emmy.compiler.pipeline.passes.lowering.tile._fromloop import fold_from_loop

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
    # CLOSED: every name the body reads is a param or one of its own defs, so there is no
    # residual for the former to append and the two spellings coincide.
    assert lam.defined == frozenset({"k", "x", "x2"})
    assert Lambda.closing(("k", "x"), lam.body, lam.results) == lam


def test_lambda_post_init_canonicalizes_body_order() -> None:
    left = Load(name="left", input="x", index=(Var("m"), Var("k")))
    right = Load(name="right", input="w", index=(Var("n"), Var("k")))
    product = Assign(name="product", op="multiply", args=("left", "right"))

    first = Lambda.closing(("k",), Body((left, right, product)), ("product",))
    second = Lambda.closing(("k",), Body((right, left, product)), ("product",))

    assert first == second


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


def test_lambda_allows_param_results_and_constant_results() -> None:
    """ι is spelled in the lift: softmax's singleton is ``(x, 1)`` — the constant component is a
    ``Const`` def, a name like any other result."""
    lam = Lambda(params=("k", "x"), body=Body((Const(name="one", value=1.0),)), results=("x", "one"))
    assert eval_lambda(lam, (0, 3.5)) == (3.5, 1.0)


def test_lambda_refuses_an_open_body_and_closing_binds_the_residual() -> None:
    """A term carries no free names. What the retired ``free_names`` reported as a contextual read
    is now a FORMATION ERROR, and :meth:`Lambda.closing` is the former that turns it into a
    trailing param — trailing, never interleaved, so the operand prefix keeps its positions."""
    body = Body((Assign(name="y", op="add", args=("x", "outer")),))

    with pytest.raises(ValueError, match=r"reads \['outer'\]"):
        Lambda(params=("x",), body=body, results=("y",))

    lam = Lambda.closing(("x",), body, ("y",))
    assert lam.params == ("x", "outer")
    assert lam.defined == frozenset({"x", "outer", "y"})
    assert lam.results == ("y",)


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


def test_alpha_inequality_on_structure_and_residual_arity() -> None:
    sq = Lambda(params=("x",), body=Body((Assign(name="t", op="multiply", args=("x", "x")),)), results=("t",))
    dbl = Lambda(params=("x",), body=Body((Assign(name="t", op="add", args=("x", "x")),)), results=("t",))
    closed = Lambda.closing(("x",), Body((Assign(name="t", op="multiply", args=("x", "w")),)), ("t",))
    renamed = Lambda.closing(("x",), Body((Assign(name="t", op="multiply", args=("x", "z")),)), ("t",))
    assert not sq.alpha_eq(dbl)
    # The residual read is BOUND, so it changes the arity — ``x·w`` is a two-param function.
    assert closed.params == ("x", "w") and not sq.alpha_eq(closed)
    # And being bound, it renumbers like any other param: its spelling no longer has to match.
    assert closed.alpha_eq(renamed)
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
    from emmy.compiler.ir.pure.fold import Fold as _Fold

    lam = Lambda(params=("a", "b"), body=Body((Assign(name="c", op="add", args=("a", "b")),)), results=("c",))
    lift = Lambda(params=("k",), body=Body((Const(name="one", value=1.0), Const(name="two", value=1.0))), results=("one", "two"))
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
    loop = Loop(axis=axis, body=Body((*value_stmts, accum)))
    fold = fold_from_loop(loop)
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
    names = ("m_i", "l_i")
    other = tuple(f"{name}__o" for name in names)
    init = (float("-inf"), 0.0)
    combine = Lambda(params=names + other, body=Body(exp_combine_states(names, other)), results=names)
    fold = Fold(
        axis=axis,
        lift=Lambda(
            params=("k",),
            body=Body((Load(name="x0", input="x", index=(Var("k"),)), Const(name="one", value=1.0))),
            results=("x0", "one"),
        ),
        init=init,
        combine=combine,
    )
    env = _run_loop(fold.loop, {"m_i": float("-inf"), "l_i": 0.0}, {"x": x})
    lift = Lambda(params=("k", "x0"), body=Body((Const(name="one", value=1.0),)), results=("x0", "one"))
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
    names = ("m_i", "l_i", "O_i")
    other = tuple(f"{name}__o" for name in names)
    init = (float("-inf"), 0.0, 0.0)
    combine = Lambda(params=names + other, body=Body(exp_combine_states(names, other)), results=names)
    fold = Fold(
        axis=axis,
        lift=Lambda(
            params=("j",),
            body=Body(
                (
                    Load(name="s0", input="s", index=(Var("j"),)),
                    Load(name="v0", input="v", index=(Var("j"),)),
                    Const(name="one", value=1.0),
                )
            ),
            results=("s0", "one", "v0"),
        ),
        init=init,
        combine=combine,
    )
    env = _run_loop(fold.loop, {"m_i": float("-inf"), "l_i": 0.0, "O_i": 0.0}, {"s": s, "v": v})
    lift = Lambda(params=("j", "s0", "v0"), body=Body((Const(name="one", value=1.0),)), results=("s0", "one", "v0"))
    spec = foldmap_eval(*_lse_monoid(1), lift, [(j, s[j], v[j]) for j in range(10)])
    np.testing.assert_allclose((env["m_i"], env["l_i"], env["O_i"]), spec, rtol=1e-10)
    m = s.max()
    p = np.exp(s - m)
    np.testing.assert_allclose(spec, (m, p.sum(), float(p @ v)), rtol=1e-10)


# --- The MONOID-FAMILY registry: membership by program equality + declared-property checks ------- #


def test_family_of_claims_componentwise_and_exp_and_rejects_foreign() -> None:
    from emmy.compiler.ir.pure.algebra import Componentwise, ExpFamily, family_of

    _, add_max = M("add", "maximum")
    fam = family_of(add_max)
    assert isinstance(fam, Componentwise) and not fam.twisted and fam.commutative and fam.observable
    assert [op.name for op in fam.ops] == ["add", "maximum"]
    assert fam.program(("s0", "s1")) == add_max  # membership IS generator-output equality

    # Membership is the STORED spelling — the ``__o`` second-operand naming formation pins —
    # so the family's own generator output is the member; the property suite's local ``__b``
    # spelling (`_lse_monoid`) is evaluation material, not a stored program, and is NOT claimed.
    lse_stored = ExpFamily().program(("m", "l", "o0"))
    exp_fam = family_of(lse_stored)
    assert isinstance(exp_fam, ExpFamily) and exp_fam.twisted and exp_fam.commutative and not exp_fam.observable
    assert family_of(_lse_monoid(1)[1]) is None

    # A cross-component read fails the componentwise shape and matches no twisted generator.
    foreign = Lambda(
        params=("a", "b", "a__o", "b__o"),
        body=Body(
            (
                Assign(name="a2", op="add", args=("a", "b__o")),
                Assign(name="b2", op="add", args=("b", "a__o")),
            )
        ),
        results=("a2", "b2"),
    )
    assert family_of(foreign) is None


def test_fold_formation_rejects_a_family_less_combine() -> None:
    """The formation gate is the registry claim: a combine no registered generator would emit is
    rejected at construction, never stored."""
    foreign = Lambda(
        params=("a", "b", "a__o", "b__o"),
        body=Body(
            (
                Assign(name="a2", op="add", args=("a", "b__o")),
                Assign(name="b2", op="add", args=("b", "a__o")),
            )
        ),
        results=("a2", "b2"),
    )
    body = Body((Load(name="x0", input="x", index=(Var("k"),)), Const(name="one", value=1.0)))
    lift = Lambda(params=("k",), body=body, results=("x0", "one"))
    with pytest.raises(AssertionError, match="no registered monoid family"):
        Fold(axis=Axis("k", 4), lift=lift, init=(0.0, 0.0), combine=foreign)


def test_declared_commutativity_holds_on_random_states() -> None:
    """Each registered entry's ``commutative`` claim, checked against the executable spec — the
    same discipline as the associativity certificate above."""
    from emmy.compiler.ir.pure.algebra import ExpFamily, family_of

    rng = np.random.default_rng(11)
    entries = (
        (2, M("add", "maximum")[1]),
        (2, ExpFamily().program(("m", "l"))),
        (3, ExpFamily().program(("m", "l", "o0"))),
    )
    for n, combine in entries:
        family = family_of(combine)
        assert family is not None and family.commutative
        for _ in range(25):
            a, b = (tuple(rng.normal(size=n) * 3.0) for _ in range(2))
            np.testing.assert_allclose(eval_lambda(combine, (*a, *b)), eval_lambda(combine, (*b, *a)), rtol=1e-12, atol=1e-12)


def test_family_merge_matches_the_retired_merge_arms() -> None:
    """``merge_stmts`` now dispatches through the claiming family; the realized programs must be
    byte-identical to the two retired inline arms."""
    from emmy.compiler.ir.pure.algebra import family_of, merge_stmts
    from emmy.compiler.ir.pure.carrier import exp_combine_states as _exp

    _, add2 = M("add", "add", names=("p", "q"))
    other = ("p__r1", "q__r1")
    got = merge_stmts(add2, other)
    assert all(isinstance(s, Accum) and s.op.name == "add" for s in got)
    assert tuple((s.name, s.value) for s in got) == (("p", "p__r1"), ("q", "q__r1"))

    from emmy.compiler.ir.pure.algebra import ExpFamily

    names = ("m", "l", "o0")
    lse = ExpFamily().program(names)
    partial = tuple(f"{n}__p" for n in names)
    assert merge_stmts(lse, partial) == _exp(names, partial, key=partial[0], accum=True)
    assert family_of(lse).program(names) == lse
