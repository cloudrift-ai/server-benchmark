"""Carrier-algebra generator + per-family stabilizer (transport of structure).

A streaming reduce carrier (online softmax, flash attention) is a **twisted monoid**: a
base monoid ``(max, +, +, …)`` conjugated by a bijection ψ. Rather than hand-author the
numerically-stable combine programs, we *generate* them:

1. **Generate** the naive symmetric combine ``ψ ∘ base_combine ∘ (ψ⁻¹ × ψ⁻¹)`` — associativity
   is inherited from the base monoid for free ("association via inverse ψ"). The naive form
   carries overflowing ``exp(m)`` factors.
2. **Stabilize** (per family) by algebraic rewriting — distribute the ψ-rescale over the base
   add, fuse exponentials (``e^a·e^b → e^{a+b}``), fold multiplicative identities, DCE the dead
   overflowing temps, CSE shared rescales. For the exp/LSE family this lands on the stable form
   where every surviving ``exp`` has a provably ``≤ 0`` argument.
3. **Certify** stability structurally: every ``exp`` arg is ``x − max(…, x, …)``.

``combine_states`` (state⊕state, the cross-partition fold) and ``merge`` (the streaming
single-element fold) are both derived from one injection spec — ``merge`` is ``combine_states``
with the second operand replaced by the per-element injected terms, its final per-component
writes retagged to seed-riding ``base``-``Accum``\\ s.

Scope: the **exp/LSE family** (covers attention + online softmax — they differ only in channel
count). The generation (1) is family-agnostic; only the stabilizer (2) is per-family.
"""

from __future__ import annotations

from dataclasses import dataclass

from emmy.compiler.dtype import F32
from emmy.compiler.ir.stmt.base import Stmt
from emmy.compiler.ir.stmt.leaves import Accum, Assign


class UnstableCarrierError(ValueError):
    """A generated combine failed the stability certificate — an ``exp`` whose argument is not
    provably ``≤ 0``. Raised instead of silently emitting overflow-prone code."""


# --------------------------------------------------------------------------------------------
# Term — a tiny symbolic algebra used ONLY to generate + stabilize the combine programs.
# (Distinct from ir.expr.Expr, which is for index/predicate codegen.)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _T:
    op: str  # 'leaf' | 'lit' | 'exp' | 'neg' | 'maximum' | 'add' | 'multiply' | 'subtract'
    a: tuple  # leaf -> (name,), lit -> (value,), else tuple[_T, ...]


def _leaf(n: str) -> _T:
    return _T("leaf", (n,))


def _lit(v: float) -> _T:
    return _T("lit", (v,))


def _term(t: str | float) -> _T:
    return _lit(t) if isinstance(t, (int, float)) else _leaf(t)


# Term-op -> ElementwiseImpl name (the only place the spelling is chosen).
_OP = {"exp": "exp", "neg": "negative", "maximum": "maximum", "add": "add", "multiply": "multiply", "subtract": "subtract"}


@dataclass(frozen=True)
class Family:
    """The exp/LSE family's OP VOCABULARY — the ψ, its pivot, and the base semiring.

    The generator below emits this spelling and the Tile-IR recognizer
    (``lowering/tile/020_twisted``) reads it back. Sharing the table is the point: a recognizer
    with its own copy of ``"exp"`` / ``"subtract"`` / ``"reciprocal"`` drifts from the generator
    the day either is respelled, and the failure is silent — the rewrite simply declines and the
    kernel demotes to the planar fold.

    The family is genuinely exp-specific (see the module docstring: generation is family-agnostic,
    the stabilizer is not), so this is a NAMED SCOPE rather than an op-name list standing in for a
    trait. What it must not be is a scope spelled twice.
    """

    #: ψ and its inverse's ingredients — ``exp(score − pivot)`` is the stable weight.
    psi: str = "exp"
    #: the pivot's fold combine — the ``max`` of ``(max, Σ)``.
    pivot: str = "maximum"
    #: how the weight's exponent is formed against the pivot.
    shift: str = "subtract"
    #: the multiplicative inverse the projection applies (``· 1/denominator``).
    inverse: str = "reciprocal"
    #: the base monoid the twisted carrier conjugates, as ``(⊗, ⊕)``.
    product: str = "multiply"
    plus: str = "add"
    #: the transparent alias a carried value may be read through.
    alias: str = "copy"


#: The one exp/LSE vocabulary. Recognition and generation both read it.
EXP_FAMILY = Family()


def _flatten_mul(t: _T) -> list[_T]:
    if t.op == "multiply":
        return _flatten_mul(t.a[0]) + _flatten_mul(t.a[1])
    return [t]


def _fold_exponents(args: list[_T]) -> _T:
    """Combine a product of exponentials' exponents into one: ``[a, neg(b)] → a − b``."""
    acc = args[0]
    if acc.op == "neg":  # leading negation (our carriers never do this, but stay total)
        acc = _T("subtract", (_lit(0.0), acc.a[0]))
    for x in args[1:]:
        acc = _T("subtract", (acc, x.a[0])) if x.op == "neg" else _T("add", (acc, x))
    return acc


def _combine_exps(factors: list[_T]) -> _T:
    """A product with no ``add`` factors: drop multiplicative-identity ``1.0``, fuse all ``exp``
    factors into one ``exp(Σ exponents)``, rebuild the product."""
    nonexp = [f for f in factors if f.op != "exp" and not (f.op == "lit" and f.a[0] == 1.0)]
    exps = [f for f in factors if f.op == "exp"]
    out = list(nonexp)
    if exps:
        out.append(_T("exp", (_fold_exponents([e.a[0] for e in exps]),)))
    if not out:
        return _lit(1.0)
    acc = out[0]
    for f in out[1:]:
        acc = _T("multiply", (acc, f))
    return acc


def _expand_product(factors: list[_T]) -> _T:
    """Normalize a product into a sum of products: distribute over the first ``add`` factor,
    else fuse the exponentials. Recurses until no ``add`` factor remains."""
    flat: list[_T] = []
    for f in factors:
        flat += _flatten_mul(f)
    for i, f in enumerate(flat):
        if f.op == "add":
            rest = flat[:i] + flat[i + 1 :]
            return _T("add", (_expand_product([f.a[0]] + rest), _expand_product([f.a[1]] + rest)))
    return _combine_exps(flat)


def _simplify(t: _T) -> _T:
    """Algebraic stabilization of a generated term: bottom-up, expanding products of sums and
    fusing exponentials so the overflowing ``exp(m)`` cancels against the ``exp(−M)`` rescale."""
    if t.op in ("leaf", "lit"):
        return t
    if t.op == "exp":
        return _T("exp", (_simplify(t.a[0]),))
    if t.op == "neg":
        return _T("neg", (_simplify(t.a[0]),))
    if t.op in ("maximum", "subtract", "add"):
        return _T(t.op, (_simplify(t.a[0]), _simplify(t.a[1])))
    if t.op == "multiply":
        return _expand_product([_simplify(t.a[0]), _simplify(t.a[1])])
    raise AssertionError(f"_simplify: unexpected term op {t.op!r}")


def _reads(t: _T, name: str) -> bool:
    if t.op == "leaf":
        return t.a[0] == name
    if t.op == "lit":
        return False
    return any(_reads(x, name) for x in t.a)


# --------------------------------------------------------------------------------------------
# Generation: naive ψ ∘ base_combine ∘ (ψ⁻¹ × ψ⁻¹) for the exp/LSE family.
# --------------------------------------------------------------------------------------------


def _gen_outputs(state: tuple[str, ...], b0: _T, b_rest: list[_T]) -> list[_T]:
    """Per-component naive→simplified output term. ``state`` = operand A (carried) names; ``b0`` =
    operand B pivot term; ``b_rest`` = operand B per-accumulator terms (state names for
    combine_states, injected values for merge). Component 0 is the max pivot."""
    m_a = _leaf(state[0])
    M = _T("maximum", (m_a, b0))
    outs = [M]  # pivot
    for i in range(1, len(state)):
        a_i = _leaf(state[i])
        b_i = b_rest[i - 1]
        lifted_a = _T("multiply", (a_i, _T("exp", (m_a,))))  # ψ⁻¹: a_i · e^{m_a}
        lifted_b = _T("multiply", (b_i, _T("exp", (b0,))))  # ψ⁻¹: b_i · e^{b0}
        base_sum = _T("add", (lifted_a, lifted_b))
        proj = _T("multiply", (base_sum, _T("exp", (_T("neg", (M,)),))))  # ψ: · e^{−M}
        outs.append(_simplify(proj))
    outs[0] = _simplify(outs[0])
    return outs


# --------------------------------------------------------------------------------------------
# Emission: lower the simplified terms to an Assign/Accum program with CSE.
# --------------------------------------------------------------------------------------------


def _emit(outs: list[_T], state: tuple[str, ...], key: str, *, accum: bool, dtype=F32) -> tuple[Stmt, ...]:
    """Emit the combine program in one of the two FORMS the same algebra takes. ``accum`` retags
    each channel's final write into a seed-riding ``base``-``Accum`` — the STATEMENT form, whose
    seed the identity placement derives from ``op.identity``; else a plain ``Assign``
    reassignment — the PURE form a stored ``combine`` :class:`Lambda` holds. Temps are namespaced
    on ``key`` so distinct folds never collide."""
    memo: dict[_T, str] = {}
    body: list[Stmt] = []
    n = [0]

    def fresh() -> str:
        name = f"{key}__t{n[0]}"
        n[0] += 1
        return name

    def realize(t: _T) -> str:
        if t.op == "leaf":
            return t.a[0]
        if t.op == "lit":
            raise AssertionError(f"literal {t.a[0]} survived stabilization")
        if t in memo:
            return memo[t]
        args = tuple(realize(x) for x in t.a)
        name = fresh()
        body.append(Assign(name, _OP[t.op], args))
        memo[t] = name
        return name

    writes: list[Stmt] = []
    # Accumulator channels.
    for sname, out in list(zip(state[1:], outs[1:], strict=True)):
        assert out.op == "add", f"accumulator channel must reduce to a sum, got {out.op}"
        p, q = out.a
        if not accum:
            writes.append(Assign(sname, "add", (realize(p), realize(q))))
            continue
        base_t, val_t = (p, q) if _reads(p, sname) else (q, p)
        base = realize(base_t)  # the rescaled old carried state (e.g. l·alpha)
        val = realize(val_t)  # this element's contribution (e.g. p or p·v)
        writes.append(Accum(name=sname, value=val, op="add", base=base, dtype=dtype))
    # Pivot (channel 0).
    pivot = outs[0]
    assert pivot.op == "maximum"
    if not accum:
        writes.append(Assign(state[0], "copy", (realize(pivot),)))
    else:
        realize(pivot)  # the max temp the rescales read
        a0, b0 = pivot.a
        other = b0 if a0.op == "leaf" and a0.a[0] == state[0] else a0
        writes.append(Accum(name=state[0], value=realize(other), op="maximum", dtype=dtype))
    return tuple(body + writes)


# --------------------------------------------------------------------------------------------
# Stability certificate.
# --------------------------------------------------------------------------------------------


def _max_operands(name: str, defs: dict[str, Assign]) -> set[str] | None:
    """The (recursively flattened) operand names of a ``maximum`` defining ``name``, or ``None``
    if ``name`` is not a maximum temp."""
    d = defs.get(name)
    if d is None or d.op.name != "maximum":
        return None
    ops: set[str] = set()
    for a in d.args:
        nested = _max_operands(a, defs)
        ops |= nested if nested is not None else {a}
    return ops


def _certify(prog: tuple[Stmt, ...]) -> None:
    """Every ``exp`` argument must be ``subtract(x, R)`` with ``R`` a ``maximum`` whose operand
    set contains ``x`` — so ``arg = x − max(…, x, …) ≤ 0`` and ``exp(arg) ≤ 1``."""
    defs = {s.name: s for s in prog if isinstance(s, Assign)}
    for s in prog:
        if not (isinstance(s, Assign) and s.op.name == "exp"):
            continue
        arg = defs.get(s.args[0])
        if arg is None or arg.op.name != "subtract":
            raise UnstableCarrierError(f"exp arg {s.args[0]!r} is not a subtract — cannot prove ≤ 0")
        x, r = arg.args
        ops = _max_operands(r, defs)
        if ops is None or x not in ops:
            raise UnstableCarrierError(f"exp({x} − {r}): {r!r} is not a max over a set containing {x!r}")


# --------------------------------------------------------------------------------------------
# Public: build the exp-family carrier.
# --------------------------------------------------------------------------------------------


def exp_combine_states(
    state: tuple[str, ...], state_b: tuple[str, ...], *, key: str | None = None, accum: bool = False, dtype=F32
) -> tuple[Stmt, ...]:
    """The cross-partition state⊕state combine for an exp-family carrier of arity ``len(state)``.
    Temps namespaced on ``key`` (defaults to ``state_b[0]`` so distinct REG-tier folds — which
    rename ``state_b`` — never collide). ``accum`` selects the STATEMENT form (final writes as
    ``base``-``Accum``, so the identity placement seeds them) over the pure ``Assign`` form a
    stored ``combine`` holds — one algebra, two spellings. ``dtype=None`` leaves that statement
    form in canonical Loop IR for total lift; kernel lowering passes an explicit accumulator dtype."""
    outs = _gen_outputs(state, _leaf(state_b[0]), [_leaf(n) for n in state_b[1:]])
    prog = _emit(outs, state, key or state_b[0], accum=accum, dtype=dtype)
    _certify(prog)
    return prog


def exp_merge(state: tuple[str, ...], terms: tuple, *, key: str | None = None) -> tuple[Stmt, ...]:
    """The streaming single-element fold for an exp-family algebra. The injection singleton is
    ``terms`` — one per component (pivot ← the score name, denominator ← ``1.0``, expectation ←
    the value name)."""
    score = terms[0]
    assert isinstance(score, str), "pivot term (the score) must be an SSA name"
    outs = _gen_outputs(state, _leaf(score), [_term(t) for t in terms[1:]])
    prog = _emit(outs, state, key or state[0], accum=True)
    _certify(prog)
    return prog


def exp_rescale(rescale: str, pivot: str, arrival: str, *, key: str) -> tuple[tuple[Stmt, ...], str]:
    """The exp-family **pivot advance** — the new pivot ``max(pivot, arrival)`` plus the ψ-RESCALE
    factor ``exp(pivot − new)`` that every carried non-pivot channel takes when the pivot moves
    there.

    It is the same factor :func:`exp_combine_states` puts on each channel's ``Accum`` ``base``,
    named on its own for a channel the merge cannot reach: an accumulator living OUTSIDE the fold's
    state. Attention's streaming sweep is that case — the ``(pivot, denominator)`` pair rides the
    state while the expectation it weights is a tensor-core output tile held in registers, rescaled
    per KV block by this factor. Certified by the same rule as every generated combine.

    Returns ``(program, the new pivot's name)`` — hand the pivot back as the arriving pivot of the
    merge that follows, so the merge's own ``maximum`` is idempotent on it."""
    new, diff = f"{key}__p", f"{key}__pd"
    prog = (Assign(new, "maximum", (pivot, arrival)), Assign(diff, "subtract", (pivot, new)), Assign(rescale, "exp", (diff,)))
    _certify(prog)
    return prog, new
