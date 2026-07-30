"""The fold algebra's IR core — the TRUE monoid ``(init, combine)``, spelled ONCE.

The algebra of a kernel lives on the :class:`~emmy.compiler.ir.tile.ir.Fold` node as the flat
``combine`` :class:`Lambda` (``S × S → S`` — ONE program, its params ``(s₁…sₙ, s₁′…sₙ′)``, its
results the merged state) plus the stored ``init`` seeds; every derived program — the streaming
step, the cross-partition combine, the cooperative tree — is read off it structurally where it
is consumed. This module holds only what the STORED TERM itself needs:

- :func:`M` — the componentwise free constructor (the one construction site of a plain fold's
  combine program; recognition threads the real accumulator names through it).
- :func:`component_ops` / :func:`degenerate` — the DEGENERATE-vs-TWISTED shape test on a stored
  combine (the family discriminator; no family annotation exists).
- :func:`rename_combine` — the SSA-rename lockstep for the stored program (a generated twisted
  program is REGENERATED over the renamed state, keeping the formation invariant).
- :func:`eval_lambda` / :func:`foldmap_eval` — the denotational spec oracle the agreement +
  associativity property tests run against.

The exp/LSE-family program GENERATORS live in :mod:`~emmy.compiler.ir.stmt.carrier`; the
lowering-side derivations (state⊕state re-emission, finalize seeds) live with their one consumer
(``pipeline/passes/lowering/_reduction``). The old ``Monoid`` / ``Semiring`` node wrappers, the
ψ-conjugation apparatus (``Carrier`` / ``Twist`` / ``State``) and the loop-annotation ``Algebra``
bundle are all retired: the node's stored combine is the single spelling of ⊕.
"""

from __future__ import annotations

from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.stmt.body import Body, Lambda
from emmy.compiler.ir.stmt.leaves import Assign

# --------------------------------------------------------------------------------------------
# The TRUE monoid — ``(init, combine)``, ONE program, stored FLAT on the ``Fold`` node.
# ``combine : S × S → S`` is a pure :class:`Lambda` whose params are ``(s₁…sₙ, s₁′…sₙ′)`` and
# whose results are the merged state — state enters as params and leaves as results, no ``Accum``
# in any stored program. The serial streaming step is NOT stored: it is ``s′ = combine(s,
# lift(k))`` — combine specialized at the singleton state the lift produces — so
# update-vs-combine consistency is correct BY CONSTRUCTION. The pair is generated ONCE at
# construction: :func:`M` (the componentwise free constructor) for a plain fold, recognition's
# pattern builders for a twisted one — the family name dissolves there and downstream reads the
# program, never a name. DEGENERATE is the DERIVED shape predicate :func:`component_ops` /
# :func:`degenerate`; the S × S → S arity check lives in ``Fold.__post_init__``. ASSOCIATIVITY is
# TEST-enforced: ``combine(a, combine(b, c)) == combine(combine(a, b), c)`` on random states.
# --------------------------------------------------------------------------------------------


def M(*ops, names: tuple[str, ...] | None = None) -> tuple[tuple[float, ...], Lambda]:
    """The componentwise ``(init, combine)`` constructor — the ``M(op…)`` spelling: one
    independent self-fold ⊕ᵢ per state component (``sᵢ = ⊕ᵢ(sᵢ, sᵢ′)`` — the reassignment shape
    ``id_combine_states`` spells), seeds the op identities. The ONE construction site of a plain
    fold's combine program. ``names`` are the state component names — recognition threads its
    REAL accumulator names through here (the byte-identity requirement: the derived serial
    step's ``Accum``\\ s carry these names); the generic ``s{i}`` default serves nameless algebra
    (the spec/property tests)."""
    impls = tuple(ElementwiseImpl(o) if isinstance(o, str) else o for o in ops)
    idents = []
    for op in impls:
        if op.identity is None:
            raise ValueError(f"M: op {op.name!r} has no identity — not a monoid ⊕")
        idents.append(op.identity)
    n = len(impls)
    s = tuple(names) if names is not None else tuple(f"s{i}" for i in range(n))
    if len(s) != n:
        raise ValueError(f"M: {n} ops but {len(s)} state names")
    o = tuple(f"{nm}__o" for nm in s)
    body = Body(tuple(Assign(name=s[i], op=impls[i], args=(s[i], o[i])) for i in range(n)))
    return tuple(idents), Lambda(params=s + o, body=body, results=s)


def component_ops(combine: Lambda) -> tuple[ElementwiseImpl, ...] | None:
    """The DEGENERATE shape test on a stored ``combine`` — every result ``sᵢ″ = ⊕ᵢ(sᵢ, sᵢ′)``,
    independently and in order (the ``Accum``-form read). Returns the per-component ⊕ᵢ handles
    (what the trait/legality queries consume), or ``None`` for a twisted monoid (any
    cross-component read, rescale temp, or reordering fails the shape)."""
    n = len(combine.results)
    if len(combine.body) != n or len(combine.params) != 2 * n:
        return None
    s, o = combine.params[:n], combine.params[n:]
    ops: list[ElementwiseImpl] = []
    for i, st in enumerate(combine.body):
        if not isinstance(st, Assign) or st.name != combine.results[i] or st.args != (s[i], o[i]):
            return None
        ops.append(st.op)
    return tuple(ops)


def degenerate(combine: Lambda) -> bool:
    """True iff ``combine`` is componentwise (a plain fold) — a derived predicate, never a
    storage arm."""
    return component_ops(combine) is not None


def rename_combine(combine: Lambda, rename_ssa) -> Lambda:
    """A copy of a stored ``combine`` with every state/temp name mapped through ``rename_ssa`` —
    the lockstep the ``Fold`` rewrite applies so the stored algebra tracks its fold's SSA renames.
    A second-operand ``<n>__o`` spelling follows
    its component name so the S × S shape survives canonical renumbering. A GENERATED twisted
    program (the exp/LSE family) is REGENERATED over the renamed state names rather than patched —
    its internal temps are namespaced on the state spelling, and regeneration is the
    deterministic rule that keeps the stored program equal to the generator's output (the
    formation invariant the consuming ``Fold`` asserts)."""
    from emmy.compiler.ir.stmt.carrier import exp_combine_states  # noqa: PLC0415

    def rn(name: str) -> str:
        if name.endswith("__o"):
            return f"{rename_ssa(name[:-3])}__o"
        return rename_ssa(name)

    if component_ops(combine) is None:
        old = tuple(r for r in combine.results if isinstance(r, str))
        old_other = tuple(f"{n}__o" for n in old)
        if combine.params == old + old_other and tuple(combine.body) == tuple(exp_combine_states(old, old_other)):
            names = tuple(rename_ssa(n) for n in old)
            other = tuple(f"{n}__o" for n in names)
            return Lambda(params=names + other, body=Body(exp_combine_states(names, other)), results=names)
    return Lambda(
        params=tuple(rn(p) for p in combine.params),
        body=Body(tuple(st.rewrite(rn) for st in combine.body)),
        results=tuple(rn(r) if isinstance(r, str) else r for r in combine.results),
    )


# --------------------------------------------------------------------------------------------
# The executable SPEC — the denotational foldMap evaluator. ⟦Fold⟧ = ⊕_{k ∈ axis} ι(lift(k)),
# seeded at init: the ~20-line oracle the agreement + associativity property tests run against.
# --------------------------------------------------------------------------------------------


def eval_lambda(lam: Lambda, args: tuple) -> tuple:
    """Evaluate a pure ``Lambda`` denotationally on numeric ``args`` (positional binding).
    Covers the ANF ``Assign`` chain — the stored combine/lift vocabulary; an arg spelling that
    is not a bound name evaluates as a float literal (the ``str(term)`` convention), and a
    ``float`` result passes through (ι's literal components — softmax's ``(x, 1)``)."""
    env = dict(zip(lam.params, args, strict=True))
    for s in lam.body:
        assert isinstance(s, Assign), f"spec evaluator covers the ANF Assign chain, got {type(s).__name__}"
        env[s.name] = s.op(*(env[a] if a in env else float(a) for a in s.args))
    return tuple(env[r] if isinstance(r, str) else r for r in lam.results)


def foldmap_eval(init: tuple, combine: Lambda, lift: Lambda, elements) -> tuple:
    """⟦Fold⟧ — the denotational foldMap: fold ``combine`` over the per-element singleton states
    ``lift(k, v₁…vₙ)``, seeded at ``init``. Each element of ``elements`` is the positional arg
    tuple of one ``lift`` application (the iteration var first, then the operand values)."""
    state = tuple(init)
    for el in elements:
        state = eval_lambda(combine, (*state, *eval_lambda(lift, tuple(el))))
    return state


__all__ = ["M", "component_ops", "degenerate", "eval_lambda", "foldmap_eval", "rename_combine"]
