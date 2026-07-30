"""The fold algebra — the loop-carried ⊕, spelled ONCE.

The algebra of a kernel is **in the body**: a reduce ``Loop`` carries an :class:`Algebra`
(``loop.carrier``) — the TRUE monoid's flat ``combine`` :class:`Lambda` (the SAME one-program
spelling a ``Fold`` node stores) plus the per-element injected ``terms`` — and every derived
program (the streaming merge, the cross-partition combine, the ``Accum`` forms, the one-shot
:class:`StateMerge`) reads off it structurally. A contraction (matmul) is just a reduce ``Loop``
whose ``AxisRole`` is ``CONTRACTION`` — the ``⊗`` lift sits in the loop body, recognized
structurally on demand, never stored as a node kind. The old ``Monoid`` / ``Semiring`` node
wrappers AND the ψ-conjugation apparatus (``Carrier`` / ``Twist`` / ``State`` — a family string
+ channel spec with a second, generated spelling of ⊕) are retired: there is no family
annotation (DEGENERATE vs TWISTED is the :func:`component_ops` shape test) and no second
spelling (``combine_states`` IS the stored combine's body).

- :class:`Algebra` — the loop-carried associative combine (⊕) + injection terms.
- :class:`StateMerge` — the renderable cross-partition state⊕state combine it emits.

The lift / projection wrapper itself — :class:`~emmy.compiler.ir.tile.ir.Map` — is a
*tile-IR* op-tree node (it carries the kernel's schedule), so it lives in ``ir/tile/ir.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.stmt.base import RenderCtx, Stmt, render_merge_program
from emmy.compiler.ir.stmt.body import Body, Lambda
from emmy.compiler.ir.stmt.carrier import exp_combine_states, exp_merge
from emmy.compiler.ir.stmt.leaves import Accum, Assign


def _rename_assign_args(a: Assign, sub: dict[str, str]) -> Assign:
    return Assign(name=a.name, op=a.op, args=tuple(sub.get(x, x) for x in a.args), dtype=a.dtype)


@dataclass(frozen=True)
class Algebra:
    """The fold **algebra** of a reduce — the flat monoid ``combine`` (``S × S → S``, the ONE ⊕
    program, the same :class:`Lambda` spelling a ``Fold`` node stores) plus the per-element
    injected ``terms`` (one per state component: the folded SSA name, or a float literal —
    softmax's denominator ``1.0``) — decoupled from any op-tree node or loop position.

    It derives the streaming fold (:attr:`merge`), the cross-partition fold
    (:attr:`combine_states` — the stored combine's body, re-emitted with the accumulator dtypes
    for a degenerate ⊕), the ``Accum`` form (:meth:`as_accums`) and the one-shot cross-partition
    combine (:meth:`as_state_merge`). It is NOT a ``Stmt`` and carries no ``partial`` / ``axis``:
    a reduce ``Loop`` carries one (``loop.carrier``) so the streaming / cooperative / cross-CTA
    materializers read the combine off the loop, not a node. DEGENERATE (a plain
    ``sum`` / ``max`` / ``mean`` — each component folded by its own self-op) vs TWISTED
    (online-softmax / flash's exp/LSE rescale) is the :func:`component_ops` shape test on the
    stored combine — no family annotation. A contraction's algebra is the degenerate algebra of
    its additive fold (the ``⊗`` lift sits in the loop body). ``dtypes`` is the per-component
    accumulator-dtype side-tuple (precision, not algebra — it keeps the emitted programs
    byte-identical); empty means default per component."""

    combine: Lambda
    terms: tuple = ()
    dtypes: tuple = ()

    @classmethod
    def degenerate(cls, folds: tuple[ElementwiseImpl, ...], names: tuple[str, ...], terms: tuple, dtypes: tuple = ()) -> Algebra:
        """The componentwise (plain-fold) algebra — combine spelled exactly as :func:`M` spells
        it (``sᵢ = ⊕ᵢ(sᵢ, sᵢ__o)``), with no identity requirement (the seed is the consuming
        fold's ``op.identity``, derived where needed)."""
        other = tuple(f"{n}__o" for n in names)
        body = Body(tuple(Assign(name=n, op=op, args=(n, o)) for n, op, o in zip(names, folds, other, strict=True)))
        return cls(combine=Lambda(params=names + other, body=body, results=names), terms=terms, dtypes=dtypes)

    @classmethod
    def exp_family(cls, score: str, extra_terms: tuple, names: tuple[str, ...]) -> Algebra:
        """The exp/LSE-family (twisted) algebra over ``names`` (pivot first) — the combine IS the
        generated cross-partition program (``exp_combine_states``), the terms the injection
        singleton (``score``, then ``1.0`` for a denominator / a value name for an expectation)."""
        other = tuple(f"{n}__o" for n in names)
        combine = Lambda(params=names + other, body=Body(exp_combine_states(names, other)), results=names)
        return cls(combine=combine, terms=(score, *extra_terms))

    @property
    def names(self) -> tuple[str, ...]:
        """The carried state's SSA names — the combine's results."""
        return tuple(self.combine.results)

    @property
    def state_b(self) -> tuple[str, ...]:
        """The second-operand state names for the cross-partition combine — the combine's
        second param half (the ``"<n>__o"`` spelling both constructors use)."""
        return tuple(self.combine.params[len(self.combine.results) :])

    @cached_property
    def ops(self) -> tuple[ElementwiseImpl, ...] | None:
        """The per-component ⊕ handles of a DEGENERATE combine (:func:`component_ops`), or
        ``None`` for a twisted (exp-family) one — the family discriminator."""
        return component_ops(self.combine)

    @property
    def component_dtypes(self) -> tuple:
        """The per-component accumulator dtype, ``None``-filled when unset."""
        return self.dtypes or (None,) * len(self.combine.results)

    @property
    def out(self) -> str:
        """The bound output name — the primary carried state component."""
        return self.combine.results[0]

    def rename(self, rename_ssa) -> Algebra:
        """A copy with every SSA name mapped through ``rename_ssa`` — the combine (via
        :func:`rename_combine`, which regenerates a twisted program over the renamed state) and
        the injected term names. The ``Loop`` / ``StridedLoop`` rewrites call this so an
        annotated reduce's ALGEBRA tracks its body through SSA renaming: a verbatim algebra
        after ``rename_ssa_sequential`` left the cooperative combine reading a state name the
        renamed body no longer defines (the M=1 cut-consumer's ``acc1``-undefined miscompile).
        σ / axis substitutions still never touch the algebra — only names move here."""
        terms = tuple(rename_ssa(t) if isinstance(t, str) else t for t in self.terms)
        return Algebra(combine=rename_combine(self.combine, rename_ssa), terms=terms, dtypes=self.dtypes)

    @cached_property
    def merge(self) -> tuple[Stmt, ...]:
        """The streaming single-element fold program — the combine specialized at the injection
        singleton: the componentwise self-fold for a degenerate ⊕, the generated exp-family
        streaming form for a twisted one."""
        if self.ops is None:
            return exp_merge(self.names, self.terms, key=self.names[0])
        return tuple(
            Assign(name=n, op=op, args=(n, str(t)), dtype=dt)
            for n, op, t, dt in zip(self.names, self.ops, self.terms, self.component_dtypes, strict=True)
        )

    @cached_property
    def combine_states(self) -> tuple[Assign, ...]:
        """The cross-partition state⊕state fold program — the stored combine's body (re-emitted
        with the accumulator dtypes for a degenerate ⊕, whose stored program is dtype-free)."""
        if self.ops is None:
            return tuple(self.combine.body)
        return tuple(
            Assign(name=n, op=op, args=(n, o), dtype=dt)
            for n, op, o, dt in zip(self.names, self.ops, self.state_b, self.component_dtypes, strict=True)
        )

    def partial_names(self) -> tuple[str, ...]:
        """The bound name of each partial the merge folds in (its external reads)."""
        return _merge_reads(self.merge, self.names)

    def dissolve(self) -> list[Stmt]:
        """The loose fold stmts this algebra lowers to — bare ``Accum``\\ s for a degenerate ⊕
        (:meth:`as_accums`), else the streaming ``merge``."""
        accums = self.as_accums()
        return list(accums) if accums is not None else list(self.merge)

    def as_accums(self) -> list[Accum] | None:
        """If this is a **degenerate** algebra (componentwise — each state component folded by
        its own self-op, no rescale temps), the equivalent ``Accum`` folds; else ``None``."""
        if self.ops is None:
            return None
        return [
            Accum(name=n, value=str(t), op=op, dtype=dt)
            for n, op, t, dt in zip(self.names, self.ops, self.terms, self.component_dtypes, strict=True)
        ]

    def as_state_merge(self, other: tuple[str, ...]) -> StateMerge:
        """A one-shot :class:`StateMerge` stmt that folds this algebra's state with a second
        fully-reduced state named ``other`` (the cross-partition combine's right operand). The
        merge program IS ``combine_states`` with ``state_b`` renamed to ``other`` (a twisted
        program regenerates keyed on ``other[0]`` so distinct folds' temps never collide), so the
        cooperative-tree / cross-CTA reduce renders it through the same machinery as a streaming
        step."""
        if self.ops is None:
            merged: tuple[Stmt, ...] = exp_combine_states(self.names, other, key=other[0])
        else:
            sub = dict(zip(self.state_b, other, strict=True))
            merged = tuple(_rename_assign_args(a, sub) for a in self.combine_states)
        return StateMerge(state=self.names, merge=merged, state_b=other)


@dataclass(frozen=True)
class StateMerge(Stmt):
    """The cross-partition state⊕state combine, as a **renderable** loop-IR stmt (its right
    operand is a second fully-reduced state named :attr:`state_b`). Emitted by
    :meth:`Algebra.as_state_merge` for the REG tree / cooperative-tree / cross-CTA finalize; it
    renders the ψ-rescale state reassignment via ``render_merge_program``. Unlike ``Accum`` it is
    not a fold carrier — it sits in a combine region, not a streaming fold loop, so it never
    makes its enclosing loop ``is_reduce``."""

    state: tuple[str, ...]
    merge: tuple[Stmt, ...]
    state_b: tuple[str, ...]

    def deps(self) -> tuple[str, ...]:
        """Every external name the render references: ``state_b`` plus any other outer name a
        merge-program stmt reads (:func:`_merge_reads` — carried state and program-internal temps
        excluded, matching the ``Accum`` convention that read-modify-written names live in
        ``defines()``). ``deps`` must be the COMPLETE read set — read counters / liveness / the
        splicer's rename resolve references through it, and the render walks the merge program
        directly (``merge`` is not a nested ``Body``), so a read absent here is invisible to them."""
        seen = set(self.state_b)
        extra = tuple(n for n in _merge_reads(self.merge, self.state) if n not in seen)
        return self.state_b + extra

    def defines(self) -> tuple[str, ...]:
        return self.state

    def pretty(self, indent: str = "") -> list[str]:
        lines = [f"{indent}({', '.join(self.state)}) <- combine_states({', '.join(self.state_b)})"]
        for a in self.merge:
            lines += a.pretty(indent + "    ")
        return lines

    def render(self, ctx: RenderCtx) -> list[str]:
        return render_merge_program(self.merge, self.state, ctx)


# --------------------------------------------------------------------------------------------
# The TRUE monoid — ``(init, combine)``, ONE program, stored FLAT on the ``Fold`` node (the
# ``Monoid`` wrapper class dissolved at 1r). ``combine : S × S → S`` is a pure :class:`Lambda`
# whose params are ``(s₁…sₙ, s₁′…sₙ′)`` and whose results are the merged state — state enters as
# params and leaves as results, no ``Accum`` in any stored program. The serial streaming step is
# NOT stored: it is ``s′ = combine(s, lift(k))`` — combine specialized at the singleton state the
# lift produces — so update-vs-combine consistency is correct BY CONSTRUCTION. The pair is
# generated ONCE at construction: :func:`M` (the componentwise free constructor) for a plain
# fold, recognition's pattern builders for a twisted one — the family name dissolves there and
# downstream reads the program, never a name. DEGENERATE is the DERIVED shape predicate
# :func:`component_ops` / :func:`degenerate`; the S × S → S arity check lives in
# ``Fold.__post_init__`` (the wrapper's old formation invariant, relocated). ASSOCIATIVITY is
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
    independently and in order (the ``as_accums`` read). Returns the per-component ⊕ᵢ handles
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
    the lockstep the ``Fold`` rewrite applies so the stored algebra tracks its fold's SSA renames
    (the same contract as :meth:`Algebra.rename`). A second-operand ``<n>__o`` spelling follows
    its component name so the S × S shape survives canonical renumbering. A GENERATED twisted
    program (the exp/LSE family) is REGENERATED over the renamed state names rather than patched —
    its internal temps are namespaced on the state spelling, and regeneration is the
    deterministic rule that keeps the stored program equal to the generator's output (the
    formation invariant the consuming ``Fold`` asserts)."""

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


def _stmt_reads(a: Stmt) -> tuple[str, ...]:
    """The arg reads of one merge-program stmt. An ``Assign`` reads its ``args``; an
    ``Accum`` reads its folded ``value`` and (when redirected) its rescaled ``base`` — its
    carried ``name`` is the loop-carried state, not a same-program read."""
    if isinstance(a, Accum):
        return (a.base, a.value) if a.base is not None and a.base != a.name else (a.value,)
    return a.args


def _merge_reads(merge: tuple[Stmt, ...], state_names: tuple[str, ...]) -> tuple[str, ...]:
    """The external read names of a merge program — args read but neither carried state
    nor a temp defined within the program — in first-use order. These are the partials the
    merge folds into the state. The program is a mix of ``Assign`` temps/rescales and ``Accum``
    folds (a twisted carrier's streaming merge); both expose their reads via :func:`_stmt_reads`
    and their def via ``name``."""
    state, defined, seen, reads = set(state_names), set(), set(), []
    for a in merge:
        for arg in _stmt_reads(a):
            if arg not in state and arg not in defined and arg not in seen:
                seen.add(arg)
                reads.append(arg)
        defined.add(a.name)
    return tuple(reads)


__all__ = ["Algebra", "M", "StateMerge", "component_ops", "degenerate", "eval_lambda", "foldmap_eval", "rename_combine"]
