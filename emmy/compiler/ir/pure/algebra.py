"""The fold algebra's IR core — the TRUE monoid ``(init, combine)``, spelled ONCE.

The algebra of a kernel lives on the :class:`~emmy.compiler.ir.pure.fold.Fold` node as the flat
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
- :func:`merge_stmts` — the one statement realization of a complete state⊕state combine.
- :func:`eval_lambda` / :func:`foldmap_eval` — the denotational spec oracle the agreement +
  associativity property tests run against.

The exp/LSE-family program GENERATORS live in :mod:`~emmy.compiler.ir.pure.carrier`; the
kernel-level partition helpers live in ``pipeline/passes/lowering/_reduction``. The old
``Monoid`` / ``Semiring`` node wrappers, the ψ-conjugation apparatus (``Carrier`` / ``Twist`` /
``State``) and the loop-annotation ``Algebra`` bundle are all retired: the node's stored combine
is the single spelling of ⊕.

The FAMILY REGISTRY (:func:`family_of`, :class:`Componentwise`, :class:`ExpFamily`) is the one
dispatch over that spelling. A family claims a stored combine iff its generator would have
emitted exactly that program — membership is program equality, never an annotation — and the
claiming family answers every family-shaped question (the twist read the kernel tiers make,
the cross-partition merge realization, the rename regeneration, the per-family legality
properties). Associativity is proven at family-AUTHORING time by transport of structure (a
bijection ψ on the carrier makes the conjugated combine associative for free) and pinned by the
property tests; nothing algebraic runs at compile time. A new family — the affine recurrence, the
Welford carrier — is a new registered entry, not a formation change.
"""

from __future__ import annotations

from dataclasses import dataclass

from emmy.compiler.dtype import F32
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.pure.carrier import exp_combine_states
from emmy.compiler.ir.pure.lam import Lambda
from emmy.compiler.ir.stmt.base import Stmt
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Const

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


# --------------------------------------------------------------------------------------------
# The MONOID-FAMILY registry — the one dispatch over a stored combine. A family is a base
# componentwise monoid, or its conjugation by a twist ψ (transport of structure: ψ a bijection
# on the carrier ⇒ the twisted combine is associative by construction). Membership is PROGRAM
# EQUALITY — a family claims a combine iff its generator would have emitted exactly it — so the
# family is derived, never stored, and no annotation can lie. Registering a new family (the
# affine recurrence, the Welford carrier) is a new entry in ``_TWISTED_FAMILIES`` plus its
# property-test rows; formation code does not change.
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Componentwise:
    """The untwisted family — one independent ⊕ᵢ per state component, exactly what :func:`M`
    constructs. Parametrized by its op vector, so the instance IS the family's payload (the
    per-component handles the trait/legality queries consume)."""

    ops: tuple[ElementwiseImpl, ...]
    name = "componentwise"
    twisted = False
    #: A per-step observer is meaningful: the serial stream visits elements in axis order, so the
    #: running state at step k is the k-prefix reduction (a scan).
    observable = True

    @property
    def commutative(self) -> bool:
        """Whether the ⊕ reorders freely — what every order-scrambling combine move (the SHFL
        butterfly, the cross-CTA atomic finalize) requires of the algebra."""
        return all(op.commutative for op in self.ops)

    def program(self, names: tuple[str, ...]) -> Lambda:
        """The canonical ``S × S → S`` combine for these state names — :func:`M`'s program."""
        return M(*self.ops, names=names)[1]

    def merge(self, names: tuple[str, ...], other: tuple[str, ...], *, dtype=F32) -> tuple[Stmt, ...]:
        return tuple(Accum(name=n, value=o, op=op, dtype=dtype) for n, op, o in zip(names, self.ops, other, strict=True))


@dataclass(frozen=True)
class ExpFamily:
    """The exp/LSE twisted family — the base ``(max, +, +, …)`` monoid conjugated by
    ψ(m, D, O…) = (m, D·e⁻ᵐ, O·e⁻ᵐ…), generated + stabilized by
    :mod:`~emmy.compiler.ir.pure.carrier` (whose ``EXP_FAMILY`` table is the op vocabulary the
    recognizer shares). Covers online softmax and flash attention — they differ only in channel
    count."""

    name = "exp"
    twisted = True
    #: The symmetric merge reorders freely (``max`` commutes, the rescaled adds commute).
    commutative = True
    #: No observer support yet: the per-step state is well-defined, but no customer exists and
    #: the smaller supported surface keeps the formation gate meaningful.
    observable = False

    def program(self, names: tuple[str, ...]) -> Lambda:
        other = tuple(f"{n}__o" for n in names)
        return Lambda(params=names + other, body=Body(exp_combine_states(names, other)), results=names)

    def merge(self, names: tuple[str, ...], other: tuple[str, ...], *, dtype=F32) -> tuple[Stmt, ...]:
        return exp_combine_states(names, other, key=other[0], accum=True, dtype=dtype)


#: The registered twisted entries, tried in order after the componentwise fast path.
_TWISTED_FAMILIES = (ExpFamily(),)


def family_of(combine: Lambda):
    """The registered family claiming ``combine`` — :class:`Componentwise` over its op vector,
    a twisted entry whose generated program equals the stored one, or ``None`` (no registered
    family; formation rejects). The ONE family read behind the merge
    realization, the rename regeneration and the observer/partition legality gates."""
    ops = component_ops(combine)
    if ops is not None:
        return Componentwise(ops)
    names = combine.results
    for family in _TWISTED_FAMILIES:
        if combine == family.program(names):
            return family
    return None


def rename_combine(combine: Lambda, rename_ssa) -> Lambda:
    """A copy of a stored ``combine`` with every state/temp name mapped through ``rename_ssa`` —
    the lockstep the ``Fold`` rewrite applies so the stored algebra tracks its fold's SSA renames.
    A second-operand ``<n>__o`` spelling follows
    its component name so the S × S shape survives canonical renumbering. A GENERATED twisted
    program (the exp/LSE family) is REGENERATED over the renamed state names rather than patched —
    its internal temps are namespaced on the state spelling, and regeneration is the
    deterministic rule that keeps the stored program equal to the generator's output (the
    formation invariant the consuming ``Fold`` asserts)."""

    def rn(name: str) -> str:
        if name.endswith("__o"):
            return f"{rename_ssa(name[:-3])}__o"
        return rename_ssa(name)

    family = family_of(combine)
    if family is not None and family.twisted:
        old = combine.results
        return family.program(tuple(rename_ssa(n) for n in old))
    return Lambda(
        params=tuple(rn(p) for p in combine.params),
        body=Body(tuple(st.rewrite(rn) for st in combine.body)),
        results=tuple(rn(r) for r in combine.results),
    )


# --------------------------------------------------------------------------------------------
# The one STATEMENT realization of a stored combine — a pure term never occupies a statement
# position, it renders into one (``ir/ARCHITECTURE.md``, "Pure terms vs statements").
# --------------------------------------------------------------------------------------------


def merge_stmts(combine: Lambda, other: tuple[str, ...], *, dtype=F32) -> tuple[Stmt, ...]:
    """The cross-partition state⊕state combine, realized as loop-IR statements: ``combine``
    applied at ``S × S → S`` with its second operand naming ``other`` — a second FULLY-REDUCED
    state (a REG copy ``<n>__r1``, a tree neighbour's partial, a workspace slice ``<n>__p``).
    Emitted wherever a partitioned reduce has to fold its partials together: the REG-tree merge,
    the cooperative tail, the cross-CTA finalize loop.

    Both families land on the same shape — ``Assign`` rescale temps followed by one ``Accum`` per
    state component. The ``Accum`` is doing two jobs: it renders the in-place reassignment
    ``s = ⊕(base, value)``, and it carries the neutral element, so the seed comes from the ONE
    identity placement (``Loop.render``) and never has to travel beside the combine. A DEGENERATE
    ⊕ needs no temps at all — one self-``Accum`` per component. A TWISTED one is REGENERATED in
    ``Accum`` form keyed on ``other[0]``: its temps are namespaced on the second operand's
    spelling, so two merges of different partials into one state cannot collide. Regenerating
    rather than patching is the same rule :func:`rename_combine` follows — a generated program is
    the deterministic function of its state names. ``dtype=None`` produces canonical Loop IR;
    kernel-level consumers keep the default f32 accumulator stamp."""
    names = combine.results
    family = family_of(combine)
    if family is None:
        raise ValueError("merge_stmts: no registered monoid family claims this combine")
    return family.merge(names, other, dtype=dtype)


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
    "Componentwise",
    "ExpFamily",
    "M",
    "component_ops",
    "degenerate",
    "eval_lambda",
    "family_of",
    "foldmap_eval",
    "merge_stmts",
    "product_spine",
    "rename_combine",
]
