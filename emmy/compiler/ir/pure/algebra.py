"""The fold algebra's IR core — the TRUE monoid ``(init, combine)``, spelled ONCE.

The algebra of a kernel lives on the :class:`~emmy.compiler.ir.pure.fold.Fold` node as the flat
``combine`` :class:`Lambda` (``S × S → S`` — ONE program, its params ``(s₁…sₙ, s₁′…sₙ′)``, its
results the merged state) plus the stored ``init`` seeds; every derived program — the streaming
step, the cross-partition combine, the cooperative tree — is read off it structurally where it
is consumed. This module holds only what the STORED TERM itself needs:

- :func:`M` — the componentwise free constructor (the one construction site of a plain fold's
  combine program; recognition threads the real accumulator names through it).
- :func:`component_ops` / :func:`degenerate` — the PLANAR-vs-TWISTED shape test on a stored
  combine (a derived reading; no family annotation exists).
- :func:`rename_combine` — the SSA-rename lockstep for the stored program.
- :func:`eval_lambda` / :func:`foldmap_eval` — the denotational spec oracle the agreement +
  associativity property tests run against.

The exp/LSE-family program GENERATORS live in :mod:`~emmy.compiler.ir.pure.carrier`; the
statement forms of a stored combine are the term's own (``Fold.merge`` / ``Fold.step``). The old
``Monoid`` / ``Semiring`` node wrappers, the ψ-conjugation apparatus (``Carrier`` / ``Twist`` /
``State``) and the loop-annotation ``Algebra`` bundle are all retired: the node's stored combine
is the single spelling of ⊕.

A TWISTED combine — online softmax's rescaling program — is never hand-authored on a term: a
recipe (:mod:`~emmy.compiler.ir.pure.twist`) states the twisted monoid as data, and
``Fold.twist`` fuses a two-pass reduce pair into it when a recipe clicks. Associativity is proven
at recipe-authoring time by transport of structure (a bijection ψ on the carrier makes the
conjugated combine associative for free) and pinned by the property tests; nothing algebraic runs
at compile time.
"""

from __future__ import annotations

from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.pure.lam import Lambda
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.leaves import Assign, Const

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
    # By NAME, not position: ``Lambda`` canonicalizes statement order, so a multi-state planar
    # combine spells its components in whatever order their ops sort.
    defs = {stmt.name: stmt for stmt in combine.body if isinstance(stmt, Assign)}
    ops: list[ElementwiseImpl] = []
    for i, result in enumerate(combine.results):
        st = defs.get(result)
        if st is None or st.args != (s[i], o[i]):
            return None
        ops.append(st.op)
    return tuple(ops)


def degenerate(combine: Lambda) -> bool:
    """True iff ``combine`` is componentwise (a plain fold) — a derived predicate, never a
    storage arm."""
    return component_ops(combine) is not None


def rename_combine(combine: Lambda, rename_ssa) -> Lambda:
    """A copy of a stored ``combine`` with every state / temp name mapped through ``rename_ssa`` —
    the lockstep the ``Fold`` rewrite applies so the stored algebra tracks its fold's SSA renames.
    A second-operand ``<n>__o`` spelling follows its component name so the S × S shape survives
    canonical renumbering; the program's own temps rename like any other definition."""

    def rn(name: str) -> str:
        if name.endswith("__o"):
            return f"{rename_ssa(name[:-3])}__o"
        return rename_ssa(name)

    return Lambda(
        params=tuple(rn(p) for p in combine.params),
        body=Body(tuple(st.rewrite(rn) for st in combine.body)),
        results=tuple(rn(r) for r in combine.results),
    )


# --------------------------------------------------------------------------------------------
# The executable SPEC — the denotational foldMap evaluator. ⟦Fold⟧ = ⊕_{k ∈ axis} ι(lift(k)),
# seeded at init: the ~20-line oracle the agreement + associativity property tests run against.
# --------------------------------------------------------------------------------------------


def eval_lambda(lam: Lambda, args: tuple) -> tuple:
    """Evaluate a pure ``Lambda`` denotationally on numeric ``args`` (positional binding).
    Covers the ANF ``Assign`` chain and a ``Const`` def — the stored combine/lift vocabulary
    (ι's constant component, softmax's ``(x, 1)``, is a def); an arg spelling that is not a
    bound name evaluates as a float literal (the ``str(term)`` convention)."""
    env = dict(zip(lam.params, args, strict=True))
    for s in lam.body:
        if isinstance(s, Const):
            env[s.name] = s.value
            continue
        assert isinstance(s, Assign), f"spec evaluator covers the ANF Assign chain, got {type(s).__name__}"
        env[s.name] = s.op(*(env[a] if a in env else float(a) for a in s.args))
    return tuple(env[r] for r in lam.results)


def foldmap_eval(init: tuple, combine: Lambda, lift: Lambda, elements) -> tuple:
    """⟦Fold⟧ — the denotational foldMap: fold ``combine`` over the per-element singleton states
    ``lift(k, v₁…vₙ)``, seeded at ``init``. Each element of ``elements`` is the positional arg
    tuple of one ``lift`` application (the iteration var first, then the operand values)."""
    state = tuple(init)
    for el in elements:
        state = eval_lambda(combine, (*state, *eval_lambda(lift, tuple(el))))
    return state


def product_spine(defs: dict, name: str, *, divide: bool = False):
    """Flatten the ``⊗`` spine defining ``name`` into ``(leaf names, spine statements)``.

    The ONE reading of a product tree. The spine is recognized by the ``semiring_product`` TRAIT,
    never an op-name list, so a newly registered ⊗ is covered without touching this. ``divide``
    additionally admits a division node on the numerator side — ``(Σ x)/c`` equals ``Σ (x/c)`` for
    a fold-invariant ``c``, but nothing licenses moving a fold into a denominator, so the divisor
    is recorded as a leaf and only the numerator continues the spine.

    Returns ``None`` when a spine node is not binary; a name with no product above it is the
    degenerate one-leaf product.
    """
    spine: list = []
    leaves: list[str] = []

    def walk(current: str) -> bool:
        stmt = defs.get(current)
        if isinstance(stmt, Assign):
            if stmt.op.semiring_product:
                if len(stmt.args) != 2:
                    return False
                spine.append(stmt)
                return all(walk(arg) for arg in stmt.args)
            if divide and stmt.op.name == "divide" and len(stmt.args) == 2:
                spine.append(stmt)
                leaves.append(stmt.args[1])
                return walk(stmt.args[0])
        leaves.append(current)
        return True

    return (tuple(leaves), tuple(spine)) if walk(name) else None


__all__ = [
    "M",
    "component_ops",
    "degenerate",
    "eval_lambda",
    "foldmap_eval",
    "product_spine",
    "rename_combine",
]
