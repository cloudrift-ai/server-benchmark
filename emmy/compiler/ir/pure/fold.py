"""``Fold`` — the ONE reduce term: ``reduce(⊕) ∘ map(f)`` in the λ-foldMap spelling.

The whole stored vocabulary of Tile IR, and a PURE term throughout: an optional iteration
``axis``, a pure ``lift`` :class:`~emmy.compiler.ir.pure.lam.Lambda`, the monoid's flat
``(init, combine)`` pair, and a tuple of ``operands`` — the closed inputs, each an edge bound
positionally to a lift param. Every reading (``Map`` at zero axes, the bilinear ``Contraction``,
the ``AxisRole``, the serial step) is DERIVED from those params; nothing else is stored.

Nothing here is a :class:`~emmy.compiler.ir.stmt.base.Stmt`. A composed step — flash's ``Σ Q·K``
ahead of its ``Σ_j P·V``, split-K's sliced contraction — is reached through ``operands``, and its
POSITION in the emitted step stream is produced by the derivation (:func:`_twisted_derived_step`
heads the inline-node edges; :func:`splice_operands` places each edge's body before the first read
of its bound name), not by sitting in a statement list. The term becomes statements in exactly one
place, :meth:`Fold.lower` / :attr:`Fold.loop`. See ``ir/ARCHITECTURE.md``, "Pure terms vs
statements".

The schedule is deliberately absent: an accepted, site-indexed ``Schedule`` lives on the
``TileOp`` boundary (``ir/tile/ir.py``), so the term is IMMUTABLE across the whole schedule search
and kernel identity (:meth:`Fold.structural_key`) is the algebra alone.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field, replace
from functools import cached_property

from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.pure.algebra import M, component_ops, family_of, merge_stmts, rename_combine
from emmy.compiler.ir.pure.carrier import exp_merge
from emmy.compiler.ir.pure.lam import Lambda
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Stmt
from emmy.compiler.ir.stmt.base import _axis_identity
from emmy.compiler.ir.stmt.body import _member_reads


def splice_operands(operands: tuple, stmts: tuple[Stmt, ...]) -> tuple[Stmt, ...]:
    """Splice each operand edge's producing stmts into ``stmts`` immediately BEFORE the first stmt
    that reads the operand's bound name, directly or through another operand. A provider inherits
    its dependent's insertion point and precedes it; otherwise an unread edge appends. Independent
    ties retain operand TUPLE order. This is the one lowering rule that turns the stored operands +
    derived step back into the flat loop body — deterministic, so the derived loop (and with it
    ``identity_key(with_io=True, with_knobs=True)``) depends only on the stored params."""
    operands = _unique_edges(operands)
    if not operands:
        return stmts

    results = tuple(frozenset(_operand_result_names(edge)) for edge in operands)
    dependencies = []
    for index, edge in enumerate(operands):
        body = Body(operand_body(edge))
        external = body.backward_cone(_operand_result_names(edge)).external_reads
        dependencies.append(tuple(provider for provider, names in enumerate(results) if provider != index and names & external))

    incoming = [set(providers) for providers in dependencies]
    outgoing: list[list[int]] = [[] for _ in operands]
    for dependent, providers in enumerate(dependencies):
        for provider in providers:
            outgoing[provider].append(dependent)
    ready = [index for index, providers in enumerate(incoming) if not providers]
    heapq.heapify(ready)
    order: list[int] = []
    while ready:
        index = heapq.heappop(ready)
        order.append(index)
        for dependent in outgoing[index]:
            incoming[dependent].remove(index)
            if not incoming[dependent]:
                heapq.heappush(ready, dependent)
    assert len(order) == len(operands), "operand edges are acyclic SSA cones — a cyclic provider dependency is not constructible"

    indexes = []
    for edge in operands:
        names = set(_operand_result_names(edge))
        indexes.append(next((i for i, st in enumerate(stmts) if names & deep_reads([st])), len(stmts)))
    for dependent in reversed(order):
        for provider in dependencies[dependent]:
            indexes[provider] = min(indexes[provider], indexes[dependent])

    at: dict[int, list] = {}
    for index in order:
        at.setdefault(indexes[index], []).append(operands[index])
    out: list[Stmt] = []
    for i, st in enumerate((*stmts, None)):
        for edge in at.get(i, ()):
            out.extend(operand_body(edge))
        if st is not None:
            out.append(st)
    return tuple(out)


def _unique_edges(operands: tuple) -> tuple:
    """Maximal operand result sets, preserving their relative order.

    A projection edge returning ``(a, b)`` subsumes a sibling returning only ``(a)``.  Keeping
    both would bind and emit the same SSA value twice; the larger pure edge is the one source.
    """
    out = []
    keys = tuple(_operand_result_names(edge) for edge in operands)
    for index, (edge, key) in enumerate(zip(operands, keys, strict=True)):
        names = set(key)
        if any(names < set(other) for other in keys):
            continue
        if any(key == earlier for earlier in keys[:index]):
            continue
        out.append(edge)
    return tuple(out)


def _flatten_nodes(steps: tuple) -> tuple[Stmt, ...]:
    """Flatten any nested **structural node** — any :class:`Fold`, at any role — that sits in the
    derived step sequence to its own lowered loop nest; plain stmts pass through. ``steps`` is a
    TUPLE, not a ``Body``: a composed step is a TERM in a term-level sequence, never a statement
    in a statement list. That is what lets flash's ``Σ_dd Q·K`` score and its
    ``Σ_j P·V``, split-K's sliced contraction, and the fused sibling group (a zero-axis ``Fold``)
    inside a split reduce's partial. This is the single
    **node-walk** the ``.loop`` splice and the kernel materializer's recursion share, and it yields
    the same loop nest whether a node was pre-flattened or reached structurally.

    A composed step's POSITION in the body is semantic, not incidental: flash's PV reads the softmax
    weight the merge stmts of that same loop step produce, so the step cannot be hoisted ahead of
    them. That is why composition rides the sequence rather than a hoisted operand tuple."""
    out: list[Stmt] = []
    for s in steps:
        if isinstance(s, Fold):
            out.extend(s.lower())
        else:
            out.append(s)
    return tuple(out)


def _derived_expect_fold(o: str, p_name: str, v_edge: Load) -> Fold:
    """The synthesized expectation contraction of a twisted fold's DERIVED blocked evaluation —
    flash's ``Oblk = Σ_j P·V``, a contraction. A is the
    register-resident softmax weight ``P`` — a one-stmt cone node whose one legal capture is the
    running max the same derived merge updates — and B is the fold's own expectation operand edge
    (the value ``Load``).

    A's ``copy`` is the REFERENCE, not a no-op: an operand edge's two inhabitants are a gmem
    ``Load`` and an inline node, with no let table and no name-reference arm, so pointing at an
    already-computed register value means wrapping it in the smallest node that yields one. The
    rename is load-bearing too — ``p_name`` is a positional temp of the generated twist program,
    while ``{o}__p`` is derived from the accumulator name and is stable.

    There is no wrapping ``Fold.projection(body=(), operands=(<this>,))`` cone shape around it: an empty cell
    carries no information (``cone_seam`` bridges no stats either way, both spellings lower to
    this one stmt, and the lowered CUDA is byte-identical), so the edge IS the one-stmt node."""
    cone = Fold.projection(body=Body((Assign(name=f"{o}__p", op="copy", args=(p_name,)),)))
    k = Axis(name="pj", extent=Dim(1))  # block=1: a singleton intra-block reduce
    return Fold.contraction(k_axis=k, a=cone, channels=(Channel(b=v_edge, acc=f"{o}__pv"),))


def _split_expect(merge: list[Stmt], o: str, v: str, v_edge: Load) -> list[Stmt]:
    """Split the generated streaming ``merge`` around the synthesized expectation contraction:
    the fused ``v·P`` product becomes :func:`_derived_expect_fold`'s output, and the state fold consumes it
    (``O = O·α + Oblk``; the ``O·α`` base is untouched)."""
    o_accum = next(s for s in merge if isinstance(s, Accum) and s.name == o)
    defs = {s.name: s for s in merge if isinstance(s, Assign)}
    prod = defs[o_accum.value]  # the fused ``v·P`` (multiply(v, P))
    p_name = next(a for a in prod.args if a != v)  # the softmax weight P (register-resident)
    pv = _derived_expect_fold(o, p_name, v_edge)
    out: list[Stmt] = []
    for s in merge:
        if s is prod:
            continue  # the inline v·P is dropped — the synthesized contraction computes it
        if s is o_accum:
            out.append(pv)
            out.append(replace(o_accum, value=f"{o}__pv"))
            continue
        out.append(s)
    return out


def _composes_state(inner, names: tuple[str, ...], ops) -> bool:
    """``inner`` carries EXACTLY the outer fold's accumulator state — the identity-lift
    reassociation's precondition (split-K): a contraction whose (additive by
    construction) channel accumulators are the names and whose outer ⊕ is componentwise
    ``add``."""
    return is_contraction(inner) and tuple(inner.defines()) == names and ops == (ElementwiseImpl("add"),) * len(names)


def _identity_lift(fold: Fold) -> bool:
    """Whether the lift passes complete operand state through unchanged."""
    bound = tuple(name for edge in fold.operands for name in _operand_result_names(edge))
    return not fold.lift.body and tuple(fold.lift.results) == bound and all(isinstance(result, str) for result in fold.lift.results)


def _operand_binding(fold: Fold) -> dict:
    """The bound-name → operand-edge map (positional binding, one lift param per operand RESULT
    COMPONENT — a product edge binds every component to the same edge)."""
    out: dict = {}
    for e in fold.operands:
        for n in _operand_result_names(e):
            out[n] = e
    return out


def _expectation_bindings(fold: Fold) -> tuple[tuple[str, str, object], ...]:
    """The twisted carrier's ``(state, injected term, operand edge)`` expectation channels.

    This is the one structural reading shared by derived evaluation and placement: every named
    non-pivot lift result bound to an operand becomes a synthesized expectation contraction when
    that edge is materialized as a :class:`Load`. A computed edge is the same future contraction
    operand before placement cuts it.
    """
    if fold.axis is None or not fold.family.twisted:
        return ()
    by_param = _operand_binding(fold)
    return tuple(
        (state, term, by_param[term])
        for state, term in zip(fold.combine.results[1:], fold.lift.results[1:], strict=True)
        if isinstance(term, str) and term in by_param
    )


def _twisted_derived_step(fold: Fold) -> tuple[Stmt, ...]:
    """The DERIVED blocked evaluation of a λ-spelled TWISTED fold: the INLINE-NODE operand edges
    at the head in operand order (flash's ``Σ_dd Q·K`` score, ahead of the lift body), the lift
    body (the scale / mask stmts), then the generated streaming merge
    with each Load-bound EXPECTATION operand split out as a synthesized contraction
    (:func:`_split_expect` — flash's PV). Deterministic from the stored params only; the operand
    edges consumed here are excluded from the generic first-use splice
    (:meth:`Fold._splice_edges`)."""
    lam = fold.lift
    names = tuple(fold.combine.results)
    terms = tuple(lam.results)
    merge = list(exp_merge(names, terms, key=names[0]))
    for nm, term, edge in _expectation_bindings(fold):
        if isinstance(edge, Load):
            merge = _split_expect(merge, nm, term, edge)
    # The inline-node edges are PLACED, not prepended: each lands immediately before the first
    # stmt (lift body or merge) that reads its bound name, ties in operand order — the same
    # first-use rule :func:`splice_operands` applies to every other edge, with the node itself
    # riding the sequence (``_flatten_nodes`` lowers it later). Prepending unconditionally would
    # reorder a step whose pure prologue precedes the producer (a loop-invariant scale ``Load``
    # ahead of attention's score contraction), and the byte-identity gate reads that as a
    # different program.
    steps: list[Stmt] = [*lam.body, *merge]
    for edge in reversed([e for e in fold.operands if isinstance(e, Fold)]):
        names = set(_operand_result_names(edge))
        steps.insert(next((i for i, st in enumerate(steps) if names & deep_reads([st])), 0), edge)
    return tuple(steps)


def _fold_derived_step(fold: Fold) -> tuple[Stmt, ...]:
    """Derive ``s′ = combine(s, lift(k))`` from the stored Fold.

    A componentwise monoid specializes a singleton lift to ``Accum`` statements. An exp-family
    singleton uses its registered streaming generator. An identity lift is different: its
    operands are already complete monoid states, so it realizes the stored ``S × S → S`` combine
    directly. That is the generic cross-partition merge used by split-reduce.

    A split-K identity lift over one contraction is reassociated by embedding that contraction;
    its additive accumulators already carry the outer state. Every case is deterministic from the
    stored parameters, so kernel identity depends on no classified view."""
    lam = fold.lift
    names = fold.combine.results
    family = fold.family
    ops = None if family.twisted else family.ops
    if _identity_lift(fold):
        if len(fold.operands) == 1 and _composes_state(fold.operands[0], tuple(names), ops):
            # Split-K's inner contraction already updates the shared accumulators directly.
            return (fold.operands[0],)
        # A fully reduced state is an ordinary monoid element, not a singleton injection. Apply
        # the stored S × S → S combine verbatim; this is the generic split-finalize shape.
        return tuple(
            replace(stmt, axes=(fold.axis.name,)) if isinstance(stmt, Accum) else stmt
            for stmt in merge_stmts(fold.combine, tuple(lam.results), dtype=None)
        )
    if ops is None:  # the twisted (exp-family) serial step — the derived state's channels
        # carry the singleton terms (the lift results), so the generated streaming merge is the
        # singleton specialization of the stored combine, names included.
        return fold._derived_twisted
    accums = tuple(Accum(name=names[i], value=str(lam.results[i]), op=ops[i], axes=(fold.axis.name,)) for i in range(len(names)))
    after: dict[str, list[int]] = {}
    for i, r in enumerate(lam.results):
        if isinstance(r, str):
            after.setdefault(r, []).append(i)
    out: list[Stmt] = []
    placed: set[int] = set()
    for s in lam.body:
        out.append(s)
        for d in s.defines():
            for i in after.get(d, ()):
                out.append(accums[i])
                placed.add(i)
    out.extend(accums[i] for i in range(len(accums)) if i not in placed)
    if fold.observe is not None:
        # The observer taps the post-combine state: its stmts run AFTER every Accum, so reading a
        # state name yields iteration k's inclusive prefix. The streamed store itself stays a
        # kernel-boundary OutputSpec; only the pure tap rides the step.
        out.extend(fold.observe.body)
    return tuple(out)


@dataclass(frozen=True)
class Fold:
    """A scheduled reduce — the typed successor of the bare annotated reduce
    ``Loop`` (``ir/pure/algebra``). It splits the reduce's **algebra** (the loop-carried
    flat ⊕ — degenerate/componentwise for a plain
    ``sum`` / ``max`` / ``mean``, twisted (exp-family) for online-softmax / flash) from its **structure**
    (the reduce ``axis`` + the per-element ``step`` it folds). Its :class:`AxisRole`
    (``PLANAR`` / ``TWISTED`` / ``CONTRACTION``) is **derived** from those params (:attr:`role`),
    never stored. The fold ``Loop`` is **synthesized on
    demand** (:attr:`loop`), never stored — so the same node tiles under any
    :class:`~emmy.compiler.ir.schedule.Reduce`, which is not a field here: the reduce
    partition is a site choice in ``TileOp.schedule``, read through ``ops.Sched``.

    A reduce whose per-step partial COMPOSES another node — split-K's ``Fold ⊃ Fold``
    (whose ``axis`` ``ksplit`` differs from the inner ``k_axis`` ``kslice``, so no double-reduce),
    flash's ``Σ Q·K`` score at the head of its kv step — spells it ONE way: the node sits in
    ``step`` and :func:`_flatten_nodes` flattens it in place. There is no second ``source`` edge.

    It holds **no projection**: a bare reduce (``sum`` / ``max``) is the kernel root (its grid
    ``Write`` is glue); a reduce with a post-fold sweep (softmax / RMSNorm) is an operand of a
    zero-axis :class:`Fold` whose lift body is that projection. A nested term may occupy a position
    in the lift's structural sequence without becoming a ``Stmt``; :meth:`lower` is the one boundary
    that flattens it to the synthesized loop.

    The reduce PARTITION (:class:`Reduce` — GRID split / BLOCK coop / REG ILP) is the schedule's,
    not the node's: it is selected for the node site in ``TileOp.schedule`` and read through
    ``ops.Sched``, which is why ``lower`` cannot see it and ``identity_key(with_io=True, with_knobs=True)`` stays byte-identical
    whichever partition the fork picked. See the NO-schedule-fields note on ``operands`` below."""

    pure = True  # a term is a value — its internals are its own; legal inside a stored ``Lambda``

    # The reduce axis — ``None`` is the ZERO-AXIS node (what zero-axis ``Fold`` was): no iteration, no monoid,
    # its ``lift`` the per-cell projection. The BILINEAR reading is a READING of this one
    # stored kind (the derived accessors below), never separate storage.
    axis: Axis | None = None
    unroll: bool = False
    # The CLOSED inputs, each an operand edge (a gmem ``Load`` or an inline node) — the 1k fold
    # vocabulary. Sharing is edge reuse: the step reads an operand's bound name as many times as it
    # needs. ``lower`` splices each edge's body before its first use (:func:`splice_operands`).
    operands: tuple = ()
    # NO schedule fields: node and edge choices live in ``TileOp.schedule`` — the term is pure
    # algebra, IMMUTABLE across the whole schedule search (a fork is a different assignment,
    # never a rebuilt tree).
    # A cross-CTA SLICE of the stream (flash split-KV) is not spelled here: ``030_cut``
    # shrinks ``axis`` to the slice length and the slice's absolute base / end ride that axis's
    # :class:`~emmy.compiler.ir.axis.Window` — ONE windowing vocabulary, the same one an axis's
    # split parentage uses, read by the realizer and the mask machinery alike.
    #
    # ---- the λ-foldMap spelling — the fold's storage: a PURE ``lift`` ``λ(k, v₁…vₙ) → S``
    # (params: the iteration var first, then one per operand edge, bound POSITIONALLY) plus the
    # TRUE monoid's flat ``(init, combine)`` pair whose combine carries the REAL accumulator
    # names (its results). The serial step, the ``Accum`` forms and the ``carrier`` annotation
    # are DERIVED (:func:`_fold_derived_step` / ``__post_init__``). ------------------------------ #
    lift: Lambda = field(kw_only=True)  # CLOSED by ``Lambda.__post_init__``; formed by :meth:`Lambda.closing`
    init: tuple = ()  # the ⊕ seeds — op identities for a plain fold; (−inf, 0, …) LSE
    combine: Lambda | None = field(kw_only=True, default=None)  # S × S → S — THE ⊕; None at zero axes
    # The per-step OBSERVER — the scan spelling: a pure λ(k, s₁…sₙ) over the carried state,
    # evaluated AFTER iteration k's combine (inclusive; exclusive is an init/index shift, never a
    # stored flag), binding the iteration var then the state positionally. Its results are FRESH
    # names (disjoint from the state) that only kernel-boundary ``OutputSpec`` writes consume —
    # the effect stays at the boundary, the term stays pure. Part of the ALGEBRA: it keys into
    # ``structural_key`` (a cumsum is not a sum), and it makes the stream order-visible, so an
    # observed fold offers exactly the serial reduce plan (every partitioned combine — coop band,
    # ILP register partials, the cross-CTA split — changes which prefixes exist).
    observe: Lambda | None = field(kw_only=True, default=None)

    def __post_init__(self) -> None:
        if not isinstance(self.init, tuple):
            object.__setattr__(self, "init", tuple(self.init))
        if self.axis is None:
            # The ZERO-AXIS node: no iteration and no monoid, so the only formation fact is the
            # positional binding — one lift param per operand RESULT COMPONENT, no leading
            # iteration var. (The projection (zero-axis) fold was exactly this, with ``fn`` for ``lift``.)
            assert self.combine is None and not self.init, "a zero-axis Fold carries no monoid"
            assert self.observe is None, "a zero-axis Fold carries no per-step state to observe"
            bound = tuple(n for e in self.operands for n in _operand_result_names(e))
            assert self.lift.params[: len(bound)] == bound, f"lift params {self.lift.params} must bind the operands {bound} positionally"
            return
        # Formation validates the positional binding and the S × S → S arity; the ``carrier``
        # annotation is a DERIVED read (:attr:`carrier`), never a second stored spelling.
        n = len(self.init)
        if len(self.combine.params) != 2 * n or len(self.combine.results) != n:
            raise ValueError(f"Fold combine must be S × S → S at arity {n}: params={self.combine.params} results={self.combine.results}")
        lam = self.lift
        assert lam.params[:1] == (self.axis.name,), f"lift param 0 must be the iteration var {self.axis.name!r}: {lam.params}"
        # One lift param per operand RESULT COMPONENT (a product edge — split-K's sliced
        # multi-channel fold — binds every component), positionally.
        bound = tuple(n for e in self.operands for n in _operand_result_names(e))
        assert lam.params[1 : 1 + len(bound)] == bound, f"lift params {lam.params[1:]} must bind the operand edges {bound} positionally"
        assert len(lam.results) == n, "one lift result per monoid component"
        # The FAMILY claim is the formation gate: membership is program equality against the
        # registry (:func:`family_of`), so a foreign combine — twisted or otherwise — with no
        # registered derivation is rejected loudly, never stored. The claiming family is memoized
        # (:attr:`family`) for every downstream family-shaped read.
        family = self.family
        assert family is not None, (
            "no registered monoid family claims this Fold's combine — the stored program must be a family generator's exact output"
        )
        if self.observe is not None:
            assert family.observable, f"the {family.name} family does not support a per-step observer"
            assert tuple(self.observe.params) == (self.axis.name, *self.combine.results), (
                f"observer params {self.observe.params} must bind the iteration var then the carried state "
                f"{(self.axis.name, *self.combine.results)} positionally"
            )
            defined = {name for stmt in self.observe.body for name in stmt.defines()}
            assert all(isinstance(r, str) and r in defined for r in self.observe.results), (
                "observer results must be FRESH names its body defines — never the carried state itself "
                "(the boundary distinguishes a streamed store from a post-fold store by the name)"
            )
            assert not any(isinstance(stmt, Fold) for stmt in self.observe.body), "an observer body holds plain stmts, never a nested node"
            assert not _identity_lift(self), "an identity-lift fold (a split partial/finalize) carries no observer"
        if not family.twisted:
            return  # the componentwise family — nothing further to validate
        # TWISTED: the state-component ROLE decision is shape-derived off the lift's injected
        # singleton, no annotation: the pivot is component 0 (its injected term the score), a
        # literal-1 injection is a denominator, a value injection an expectation.
        assert isinstance(lam.results[0], str), "the twisted lift's pivot component must inject the score name"

    @property
    def environment(self) -> tuple[str, ...]:
        """The trailing lift params NO operand supplies — what this term must be applied to.

        The declared successor of the free-name walk :meth:`deps` used to perform: it is read off
        the stored params, not discovered by scanning an enclosing scope."""
        lead = 1 if self.axis is not None else 0
        bound = sum(len(_operand_result_names(edge)) for edge in self.operands)
        return tuple(self.lift.params[lead + bound :])

    @property
    def role(self) -> AxisRole:
        """The fold's :class:`AxisRole`, DERIVED from the stored params — never stored:

        - ``FREE`` iff there is no axis (the zero-axis node — a pure pointwise cell or the
          projection over a source node; what zero-axis ``Fold`` was).
        - ``TWISTED`` iff the stored combine's twist family is non-degenerate (``exp`` — online
          softmax / flash).
        - ``CONTRACTION`` iff the bilinear reading holds (:attr:`_contraction` — a ``⊗`` lift
          distributed over ≥ 2 operand edges under a componentwise-additive ⊕). Split-K's outer
          reduce is NOT one: it tiles nothing and has no operand pair, so it derives ``PLANAR``
          like any other additive fold and :attr:`composed` — a structural probe, not a role —
          stays the one read that recognizes the reassociation.
        - ``PLANAR`` otherwise — including an unbindable contraction (matvec-shaped 1-D output,
          no ``(m, n)`` loads, the zero-legal-rows fallback): recognition keeps its loads inline
          in the lift instead of building the node, so there are no edges for the bilinear
          reading to bind and the fold takes the reduce tiers at schedule dispatch. The demotion
          is a FORMATION fact and there is no role rewrite anywhere: recognition keeps an
          unbindable contraction's loads inline in the lift, so there are no edges to bind."""
        if self.axis is None:
            return AxisRole.FREE
        if self.family.twisted:
            return AxisRole.TWISTED
        if self._contraction is not None:
            return AxisRole.CONTRACTION  # the bilinear cell itself — the node kind that was
        return AxisRole.PLANAR

    @cached_property
    def family(self):
        """The registered monoid family claiming this fold's combine
        (:func:`~emmy.compiler.ir.pure.algebra.family_of`) — ``None`` at zero axes (no monoid).
        Memoized: the term is immutable, and a twisted membership check regenerates the family
        program. The ONE family-shaped read (the ``TWISTED`` role, the derived-step dispatch,
        the observer/partition legality gates)."""
        return family_of(self.combine) if self.combine is not None else None

    @cached_property
    def _contraction(self) -> tuple[object, tuple[Channel, ...]] | None:
        """The BILINEAR reading — ``(a, channels)`` — or ``None`` when this fold is not a
        contraction. This is the derived successor of the retired ``Contraction`` kind's ``a`` /
        ``channels`` fields: the shape it recognizes is exactly the
        one :meth:`Fold.contraction` builds, so ``a`` / ``channels`` / ``b_trans`` read back
        off any fold recognition stored as a contraction.

        The canonical operand order starts ``(b₀, a, …)``; later channel edges occur once, even
        when several channels reuse one. Node-locally the two sides are symmetric: each carries the
        reduction axis plus one free axis. Telling physical M from N needs the PLACEMENT, which is a
        caller fact living on the ``TileOp`` and deliberately absent here.

        The reading is SEMIRING-GENERIC, by the traits and never by op name: the carrier must be a
        product of ONE commutative-monoid ⊕ and every lift stmt one shared two-arg ⊗ with
        ``⊗.distributes_over(⊕)`` — the registered-semiring table. Today that table admits exactly
        ``(multiply, add)``, so the reading is the matmul it always was; a new semiring instance
        registers in ``ElementwiseImpl._SEMIRING`` and reads back here without change (the mma
        tier gates on the ``(·, +)`` instance via :attr:`semiring`)."""
        if self.axis is None or len(self.operands) < 2 or self.combine is None:
            return None
        ring = self.semiring
        n = len(self.combine.results)
        if ring is None:
            return None
        product, _ = ring
        defs = self.lift.body.definitions
        body = tuple(defs.get(result) for result in self.lift.results if isinstance(result, str))
        if len(body) != n or tuple(self.lift.results) != tuple(f"{r}__v" for r in self.combine.results):
            return None
        names = tuple(operand_name(edge) for edge in self.operands)
        by_name = dict(zip(names, self.operands, strict=True))
        shared = set(body[0].args).intersection(*(stmt.args for stmt in body[1:]))
        a_name = names[1] if len(names) > 1 and names[1] in shared else next(iter(shared), None)
        if a_name is None:
            return None
        b_names = []
        for index, stmt in enumerate(body):
            if not isinstance(stmt, Assign) or stmt.op != product or a_name not in stmt.args:
                return None
            others = tuple(arg for arg in stmt.args if arg != a_name)
            if not others and stmt.args == (a_name, a_name):
                others = (a_name,)
            if len(others) != 1 or others[0] not in by_name:
                return None
            b_name = others[0]
            expected = tuple(sorted((a_name, b_name))) if stmt.op.commutative else ((b_name, a_name) if index == 0 else (a_name, b_name))
            if stmt.args != expected:
                return None
            b_names.append(b_name)
        if set(names) != {a_name, *b_names}:
            return None
        chans = tuple(Channel(b=by_name[name], acc=acc) for name, acc in zip(b_names, self.combine.results, strict=True))
        return by_name[a_name], chans

    @cached_property
    def semiring(self) -> tuple | None:
        """The componentwise ``(⊗, ⊕)`` instance carried by this Fold, independent of operand sharing.

        A semiring Fold has one distributive binary product per lift result and one shared
        commutative-monoid combine. Whether those products share the A operand shape required by
        the current MMA emitter is the narrower :attr:`_contraction` reading; it is deliberately
        not part of this algebraic question.
        """
        if self.axis is None or self.combine is None:
            return None
        pluses = component_ops(self.combine)
        if pluses is None or not pluses or len(set(pluses)) != 1:
            return None
        plus = pluses[0]
        if not (plus.associative and plus.commutative and plus.has_identity):
            return None
        if self.init != (plus.identity,) * len(pluses):
            return None
        defs = self.lift.body.definitions
        products = [defs.get(result) if isinstance(result, str) else None for result in self.lift.results]
        if len(products) != len(pluses) or any(not isinstance(stmt, Assign) or len(stmt.args) != 2 for stmt in products):
            return None
        product = products[0].op
        if any(stmt.op != product or not stmt.op.distributes_over(plus) for stmt in products):
            return None
        if {id(stmt) for stmt in products} != {id(stmt) for stmt in self.lift.body}:
            return None
        return product, plus

    @property
    def composed(self) -> Fold | None:
        """The single sliced contraction this outer reduce COMPOSES (split-K's
        reassociation ``fold_k = fold_{ksplit} ∘ fold_{kslice}``), or ``None`` — the identity-lift
        λ spelling (one inline node operand carrying the outer's exact accumulator state). The
        structural probe the derived :attr:`role` reads (``030_cut`` builds its sliced
        partial directly, so the composition is a recognized FORM here, never a required input)."""
        if len(self.lift.body) or len(self.operands) != 1:
            return None
        inner = self.operands[0]
        return inner if is_contraction(inner) else None

    # ---- the DERIVED READINGS. ``Map`` and ``Contraction`` are no longer stored kinds (the
    # collapse); every field they carried reads back off the one stored term here, so their old
    # accessors keep their exact meanings and their consumers keep their exact spellings. ------- #
    @property
    def body(self) -> Body:
        """The projection body — ``lift.body`` (the stmts live on the lambda)."""
        return self.lift.body

    @property
    def a(self):
        """The shared operand edge of the bilinear reading (``operands[1]``).

        Placement resolves which physical output axis this edge carries.
        """
        v = self._contraction
        assert v is not None, f"not a contraction fold (role={self.role.value}) — no `a` reading"
        return v[0]

    @property
    def channels(self) -> tuple[Channel, ...]:
        """The product channels ``(bᵢ, accᵢ)`` of the bilinear reading — arity 1 is a plain
        matmul, arity N the fused gate⊗up edge over ONE shared ``a``."""
        v = self._contraction
        assert v is not None, f"not a contraction fold (role={self.role.value}) — no `channels` reading"
        return v[1]

    @property
    def b(self):
        """The primary channel's streamed operand edge (``channels[0].b``)."""
        return self.channels[0].b

    @property
    def acc(self) -> str:
        """The primary channel's fold accumulator (``channels[0].acc``)."""
        return self.channels[0].acc

    @property
    def b_trans(self) -> bool:
        """B stored N×K (the K axis last in its index) vs the canonical ``B[k, n]`` — a gmem
        LAYOUT question, so it is meaningful only for a materialized B; a computed B answers
        ``False`` (every tier that would act on the layout gates on ``isinstance(c.b, Load)``)."""
        return isinstance(self.b, Load) and self.axis.name in self.b.index[-1].free_vars()

    @classmethod
    def contraction(
        cls, *, k_axis: Axis, a, channels: tuple[Channel, ...], product="multiply", fold_op="add", axes: tuple[str, ...] = ()
    ) -> Fold:
        """A BILINEAR fold over the ``(⊗, ⊕)`` semiring — the matmul cell at the default
        ``(multiply, add)`` instance. Unlike :meth:`projection` this constructor GENERATES
        algebra: unique operands in first-use order from `(b₀, a, b₁…)`, the lift
        ``λ(k, b, a, b₂…). (b⊗a, a⊗b₂, …)`` and the
        componentwise ⊕ over the channel accumulators. That generated shape is exactly what
        :attr:`_contraction` reads back, so ``a`` / ``channels`` / ``b_trans`` / :attr:`semiring`
        and the ``CONTRACTION`` role all follow from it.

        The FORMATION GATE asserts the laws every consumer relies on, where the node is built:
        ``⊕`` a commutative monoid (associative + commutative + identity — the reassociation
        license behind split-K, the tree combine and the atomic partition store) and ``⊗``
        distributing over it (``ElementwiseImpl.distributes_over``, the registered-semiring
        table). A caller REBUILDING an existing node (a σ-sliced split, a decode-boundary cone
        rewrap) must thread the node's own :attr:`semiring`, never assume the default.

        Arity N ≥ 2 is the fused sibling edge (gate⊗up): N channels over ONE shared operand,
        scheduled and lowered as one unit. Any edge reused by several terms occupies one operand
        slot; sharing never duplicates its load.

        Placement and schedule live nowhere here: the ``(m, n)`` axes ride ``TileOp.place`` and the
        typed assignment rides ``TileOp.schedule``, so a node's identity is its algebra alone."""
        mul = ElementwiseImpl(product) if isinstance(product, str) else product
        plus = ElementwiseImpl(fold_op) if isinstance(fold_op, str) else fold_op
        if not (plus.associative and plus.commutative and plus.has_identity and mul.distributes_over(plus)):
            raise ValueError(f"contraction: ({mul.name}, {plus.name}) is not a registered semiring (⊗ over a commutative-monoid ⊕)")
        channels = tuple(channels)
        prim = channels[0]
        operands = [prim.b, a]
        seen = {operand_name(prim.b), operand_name(a)}
        for channel in channels[1:]:
            name = operand_name(channel.b)
            if name not in seen:
                seen.add(name)
                operands.append(channel.b)
        operands = tuple(operands)
        a_name = operand_name(a)
        body: list[Stmt] = [Assign(name=f"{prim.acc}__v", op=mul, args=(operand_name(prim.b), a_name))]
        body += [Assign(name=f"{ch.acc}__v", op=mul, args=(a_name, operand_name(ch.b))) for ch in channels[1:]]
        accs = tuple(ch.acc for ch in channels)
        # ``axes`` is the enclosing scope, supplied by the binder. Declared beside the operand
        # names so canonicalizing a contraction re-forms its lift WITHOUT resetting the caller's
        # declaration — this constructor is the one the semiring canonicalization builds through.
        bound = tuple(operand_name(edge) for edge in operands)
        scope = tuple(axis for axis in axes if axis != k_axis.name and axis not in bound)
        lift = Lambda.closing((k_axis.name, *bound, *scope), Body(tuple(body)), tuple(f"{acc}__v" for acc in accs))
        init, combine = M(*([plus] * len(accs)), names=accs)
        return cls(axis=k_axis, operands=operands, lift=lift, init=init, combine=combine)

    @classmethod
    def projection(cls, operands: tuple = (), *, body=None, results: tuple | None = None, axes: tuple[str, ...] = ()) -> Fold:
        """A ZERO-AXIS fold — the pointwise / projection cell (what the zero-axis fold kind was).
        No axis and no monoid: the synthesized binder IS the ``lift`` and IS the per-cell compute,
        so softmax's normalize, the relu epilogue and flash's ``divide(O, l)`` are this node over
        the reducing fold rather than a wrapper kind around it.

        ``operands`` bind POSITIONALLY, one lift param per RESULT COMPONENT — a product operand
        binds every channel accumulator, so the geglu combine's second read is a bound param and
        never a free name. The body is pure; synthesized Loop IR must pass through total lift
        before it can enter a projection. Results default to the body's last definition unless
        the caller explicitly names the values passed through to a consumer."""
        members = tuple(body) if body is not None else ()
        if any(isinstance(stmt, Fold) for stmt in members):
            # A term handed in as a BODY member is an operand edge that has not been spelled as
            # one yet. Separating here — rather than leaving it for a later rewrite — is what
            # keeps a lambda body free of nested terms: a Fold tree composes through operands,
            # and two composition mechanisms are one too many.
            names = tuple(results) if results is not None else (_map_results(members) or ())
            return _ordered_projection((*operands, *members), names, axes)
        b = Body(members)
        operands = _unique_edges(tuple(operands))
        params = tuple(n for s in operands for n in _operand_result_names(s))
        if results is None:
            results = _map_results(b) or params[:1]
        # ``axes`` is the enclosing iteration scope, supplied BY THE BINDER — the term cannot tell
        # an axis from a value, so the caller that bound them says which is which. Declared beside
        # the operand results; a caller with no scope to declare passes none and nothing changes.
        scope = tuple(axis for axis in axes if axis not in params)
        return cls(axis=None, operands=operands, lift=Lambda.closing((*params, *scope), b, tuple(results)))

    @cached_property
    def _derived_twisted(self) -> tuple[Stmt, ...]:
        """The MEMOIZED derived blocked evaluation of a λ-spelled twisted fold — cached so the
        synthesized expectation contraction (flash's PV) has ONE identity per stored fold: the
        path walker's derived sites, the schedule accessors (``Sched.tile_of``) and every
        realizer read the same node object (site matching is by identity)."""
        return _twisted_derived_step(self)

    def step_stmts(self) -> tuple[Stmt, ...]:
        """The DERIVED per-cell stmt sequence (:func:`_fold_derived_step`): the lift body with
        each component's ``Accum`` — the combine specialized at the singleton; the full blocked
        evaluation for a twisted fold with operand edges (flash's derived head is its score
        operand edge, its PV the memoized synthesized contraction); the embedded operand for
        split-K composition; or the stored combine for an identity lift of complete states."""
        return _fold_derived_step(self)

    def _splice_edges(self) -> tuple:
        """The operand edges the GENERIC first-use splice places — every edge not already
        consumed by the derived step: a λ-spelled TWISTED fold's derived blocked evaluation
        embeds its inline-node edges at the head and its Load-bound expectation edges inside
        the synthesized contraction (:func:`_twisted_derived_step`); the identity-lift
        composition (split-K) embeds its one fold operand verbatim. None of those splice
        twice."""
        consumed = {id(s) for s in self.step_stmts()}
        if self.family.twisted and not _identity_lift(self):
            for _, _, edge in _expectation_bindings(self):
                if isinstance(edge, Load):
                    consumed.add(id(edge))
        return _unique_edges(tuple(e for e in self.operands if id(e) not in consumed))

    @cached_property
    def loop(self) -> Loop:
        """The synthesized annotated reduce ``Loop`` — reconstructed from the params: byte-identical
        to the loop :meth:`from_loop` captured (the λ spelling's construction gate guarantees it;
        a retained ``step`` reproduces trivially). Any **nested structural node inside the
        step** — any nested :class:`Fold` — is flattened to its own loop nest in place by the shared :func:`_flatten_nodes`
        node-walk (so flash's kv loop holds the head ``Σ Q·K`` score contraction loop and the embedded
        ``Σ_j P·V`` PV contraction loop, and split-K's kslice contraction its own nest — exactly the
        loop-in-body form the scalar tier expands): ONE structural rule for a reduce whose per-step
        partial composes other nodes. Operand edges splice in ahead of the first read of their
        bound param (:func:`splice_operands` — positional binding names the edges, ties in
        operand order), so a fold with hoisted inputs lowers byte-identically to the flat form
        that carried them in its body."""
        stmts = splice_operands(self._step_edges(), _flatten_nodes(self.step_stmts()))
        return Loop(axis=self.axis, body=Body(stmts), unroll=self.unroll, role=self.role)

    def _hoisted_edges(self) -> tuple:
        """The operand edges the loop does NOT re-evaluate per step: a computed edge whose subtree
        never reads the fold's iteration var (a chained producer — the normalized row's
        statistic, a sibling fold's projected result) is loop-invariant and lowers ONCE, ahead of
        the loop. A gmem ``Load`` edge always rides the step (its index is the operand slab)."""
        if self.axis is None:
            return ()
        # Read off the edge's DECLARATION, not by lowering it and scanning free names: an edge
        # declares the enclosing coordinates it reads as lift params, less the axis it binds (an
        # edge reducing over its own ``k`` shadows this one and is not varying with it). The
        # lowered-body scan this replaces is the same inversion removed from `_operand_roles` —
        # asking a term to re-derive what it already states — and it re-lowered every edge on
        # every call. Inlined rather than imported: ``ir/pure`` may not reach the schedule layer
        # where the general reading lives (``Closure.over_edge`` inlines it for the same reason).
        return tuple(
            edge
            for edge in self._splice_edges()
            if isinstance(edge, Fold) and self.axis.name not in (set(edge.lift.params) - edge.binds_axes())
        )

    def _step_edges(self) -> tuple:
        hoisted = {id(e) for e in self._hoisted_edges()}
        return tuple(e for e in self._splice_edges() if id(e) not in hoisted)

    def spliced_step(self) -> tuple[Stmt, ...]:
        """The (derived) step with every operand edge's body spliced before its first use — the
        stmt sequence the emit-side node walk consumes (nested structural nodes NOT flattened;
        :attr:`loop` additionally flattens them). Edges the derived blocked evaluation already
        consumed (a twisted fold's head node / expectation Load) never splice twice."""
        return splice_operands(self._step_edges(), self.step_stmts())

    @property
    def out(self) -> str:
        """The bound output name — the carried state's primary component (the combine's first
        result; a bare reduce's grid ``Write`` is glue). At zero axes there is no carried state,
        so it is the projection's primary result (what a projection's ``out`` read)."""
        if self.axis is None:
            r = self.lift.results
            assert r and isinstance(r[0], str), f"zero-axis Fold has no named result: {r!r}"
            return r[0]
        return self.combine.results[0]

    def lower(self) -> list[Stmt]:
        """Flatten to the loop-IR body the materializer expands — the synthesized reduce ``Loop``,
        or, at zero axes, the operand nests followed by the projection body (a ``Load`` edge is
        the CUT TERMINAL: the seam value arrives materialized, so the "nest" is the load itself).

        The ONE lowering spelling: every caller that consumes a body calls this method (there is no
        free ``ops.lower`` wrapper duplicating it), which is also what keeps this module free of any
        import back into :mod:`~emmy.compiler.ir.tile.ops`."""
        if self.axis is None:
            prefix = [s for e in _unique_edges(self.operands) for s in operand_body(e)]
            return [*prefix, *_flatten_nodes(tuple(self.body))]
        return [*(s for e in self._hoisted_edges() for s in operand_body(e)), self.loop]

    # ---- the STRUCTURAL protocol — children, defs, reads, bound axes. Spelled with the stmt
    # vocabulary's names on purpose: one canonicalizer and one deep walk then serve a term and its
    # stmt siblings without dispatching on which they are. Nothing here is statement behaviour —
    # a term has no scope to seed, no effect to order and no ``render``; it becomes statements
    # once, through :meth:`lower`. ---------- #
    @property
    def observed(self) -> bool:
        """Whether this fold carries a per-step observer — the structural probe (like
        :attr:`composed`, not a role) the schedule gates and the boundary read."""
        return self.observe is not None

    def defines(self) -> tuple[str, ...]:
        """The result names this node exposes to its containing lambda — the combine's state,
        plus an observer's results: the streamed store reads them at the stream position after
        the node, so they are interface names (protected through canonicalization like any
        result), relocated into the loop only at reconstitution."""
        results = self.lift.results if self.axis is None else self.combine.results
        observed = self.observe.results if self.observe is not None else ()
        return tuple(result for result in (*results, *observed) if isinstance(result, str))

    def nested(self) -> tuple[Body, ...]:
        """The one nested body is the lift's — EXCEPT under the bilinear reading, whose lift is
        pure algebra reached through the operand edges (the retired ``Contraction`` kind had no nested
        body, and generic walks must keep seeing none, or a contraction's multiply args start
        reading as body deps)."""
        return () if self._contraction is not None else (self.lift.body,)

    def with_bodies(self, bodies: tuple[Body, ...]) -> Fold:
        if not bodies:
            return self
        (partial,) = bodies
        return replace(self, lift=Lambda(self.lift.params, Body.coerce(partial), self.lift.results))

    def rewrite(self, rename_ssa, sigma=None, axis_fn=None):
        """α-rename this term — the TERM operation that happens to share the stmt vocabulary's
        spelling, so the shared canonicalizers (``rename_ssa_sequential`` and friends) reach a
        node and its stmt siblings through one call. Dispatches to the registered handler at the
        bottom of this module, which threads the rename through the operand edges, the lift and
        the combine in lockstep."""
        return _rewrite(self, rename_ssa, Sigma.IDENTITY if sigma is None else sigma, _axis_identity if axis_fn is None else axis_fn)

    def __getstate__(self):
        """Pickle the stored params only. Every memo riding ``__dict__`` (the derived loop, deps,
        the normalize stamp, the codec's spell cache) recomputes after transport — and an
        id-keyed cache carried across processes could collide with a fresh object's id."""
        return {name: self.__dict__[name] for name in self.__dataclass_fields__ if name in self.__dict__}

    def structural_key(self) -> str:
        """The α-invariant identity digest of this term — the
        :class:`~emmy.compiler.structural.Structural` implementation: the EXACT-flavor canonical
        digest of the Loop-IR body the term lowers to (``Body.structural_key(structural=False)``
        — SSA / axis / buffer spelling normalized away, op kinds kept, since consumers like the
        sharing unification replace occurrences with one representative and must never merge
        distinct computations). The term is pure algebra and its lowered body is its normal
        form, so no separate term hasher exists. Cached: the term is immutable across the whole
        schedule search."""
        return self._lowered_key

    @cached_property
    def _lowered_key(self) -> str:
        # Through the ONE reconstitution spelling (with no output specs — a bare term), so a
        # ``ProjectionRegion`` in a projection body expands exactly as materialization expands it.
        # The per-step observer is folded in beside the body: a bare lowering carries observer
        # stmts only when reconstituted with their stream store, and a scan must never key as its
        # plain fold (the sharing unification would merge them).
        from emmy.compiler.ir.tile.ir import lower_with_output_specs  # noqa: PLC0415 — region expansion lives with the region type
        from emmy.compiler.structural import digest  # noqa: PLC0415

        body = Body.coerce(lower_with_output_specs(self, ())).structural_key(structural=False)
        observed = "" if self.observe is None else Body.coerce(self.observe.body).structural_key(structural=False)
        return digest(body, observed)

    def deps(self) -> tuple[str, ...]:
        """What this term needs SUPPLIED — its own :attr:`environment` plus its operand edges',
        recursively, less the axes it binds.

        A DECLARATION read, not a context question: it reads stored params and edge indices, never
        a lowered body, and never asks what the enclosing scope happens to hold. A term states what
        it must be applied to; that is its own business, and the enclosing binder needs it — the
        generic walks reach a term this way (``stmt/body._member_reads``), and an operand edge is
        NOT a child (:meth:`nested` yields the lift body only), so without the roll-up a capture
        living on an edge is invisible to every caller that walks statements.

        Returning a bare ``()`` here was wrong and silently so: `_ordered_projection` could not see
        that a term's edge read a name its prefix defined, and a legitimate shape tripped its
        "separated pure prefix must feed its suffix" assertion.
        """
        return self._deps

    @cached_property
    def _deps(self) -> tuple[str, ...]:
        out = set(self.environment)
        for edge in self.operands:
            if isinstance(edge, Fold):
                out |= set(edge.deps())
            else:
                out |= {name for expr in edge.exprs() for name in expr.free_vars()}
        return tuple(sorted(out - set(self.binds_axes())))

    # The above is the whole story. A term is closed: its VALUES arrive through operand edges bound positionally
    # to lift params, so there is no SSA name it reads from an enclosing scope — the base
    # ``Stmt.deps`` default of ``()`` is exactly right and overriding it would be a claim to the
    # contrary. The ITERATION SPACE is a separate channel and always was: an axis is not a value
    # the term depends on but the space it is evaluated over, carried by :attr:`free_axes` and, on
    # the edges themselves, by each ``Load``'s index ``exprs`` (which already duplicated every name
    # the old ``deps`` reported — that duplication is what made this look load-bearing).

    def exprs(self):
        """No index / predicate ``Expr`` of its own — a term's coordinates live on the ``Load``
        edges and the stmts inside its lift, which the walks reach as children.

        Spelled for the same reason as :meth:`deps`: a ``Fold`` duck-types the stmt protocol
        rather than inheriting it, so there is no base default to fall back on."""
        return ()

    def binds_axes(self) -> frozenset[str]:
        """The iteration var this term binds (empty at zero axes) — what scopes an axis-name read
        so a nested fold's ``k`` shadows an enclosing one of the same name."""
        return frozenset() if self.axis is None else frozenset({self.axis.name})


def is_contraction(x) -> bool:
    """The BILINEAR reading of ``x`` — the predicate that replaced ``isinstance(_, Contraction)``
    when the kind dissolved. A predicate, not a kind: it cannot be constructed, subclassed or
    annotated, and it answers ``False`` for a non-node (a ``Load`` edge, a plain stmt in a step
    stream) exactly as a type test would. Prefer this over a bare
    ``x.role is AxisRole.CONTRACTION`` anywhere ``x`` is not already known to be a ``Fold``."""
    return isinstance(x, Fold) and x._contraction is not None


def deep_defines(s: Stmt) -> set[str]:
    """Every SSA name defined in ``s`` (deep — a stat reduce ``Loop``'s ``Accum`` counts)."""
    out = set(s.defines())
    for b in s.nested():
        for child in b:
            out |= deep_defines(child)
    return out


def deep_reads(stmts: list[Stmt]) -> set[str]:
    """Every SSA name read anywhere in ``stmts`` (deep)."""
    out: set[str] = set()
    for s in stmts:
        out |= set(s.deps())
        for b in s.nested():
            out |= deep_reads(list(b))
    return out


def loaded_buffers(node) -> set[str]:
    """Every graph BUFFER loaded under ``node`` — a term, a stmt, or a body of them.

    ``Body.loads`` walks ``Stmt.nested()``, and a Fold's operand EDGES are not nested statements:
    :meth:`Fold.nested` yields the lift body, and nothing at all for a contraction, whose algebra
    is meant to read as edges rather than body deps. That is fine for a fully flattened stream,
    but a lowered body still carries Folds as TERMS wherever a region kept them
    (:class:`~emmy.compiler.ir.tile.ir.ProjectionRegion` holds its cones as terms), so asking the
    lowered body alone silently under-reports every buffer beneath such an edge.

    Ask this whenever the answer must cover what a consumer of the STORED tree will reach — the
    cut declaring a piece's graph inputs, ordering pieces by the workspaces they read. A cut that
    declared the lowered view instead named fewer inputs than the kernel the materializer built
    from the same tree went on to read, and the workspace producers, unreferenced, were pruned as
    orphans."""
    out: set[str] = set()
    seen: set[int] = set()
    stack = [node]
    while stack:
        item = stack.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, Load):
            out.add(item.input)
        elif isinstance(item, Fold):
            stack.extend(item.operands)
            stack.extend(item.lift.body)
        elif isinstance(item, (list, tuple, Body)):
            stack.extend(item)
        else:
            for body in item.nested():
                stack.extend(body)
    return out


def _ordered_projection(members, results: tuple[str, ...], axes: tuple[str, ...] = ()) -> Fold:
    """Factor an ordered pure cone without moving a Fold ahead of an earlier scalar producer.

    A projection evaluates every operand before its scalar body, so a source sequence
    ``Fold; scalar; Fold`` cannot flatten both terms into sibling operands — the later one reads
    the scalar, and as an edge it would splice ahead of its own provider. The prefix becomes a
    source projection of the later Fold instead.

    It lives beside :meth:`Fold.projection`, the constructor that needs it: separating terms out
    of a body is a FORMATION rule, not something a later pass repairs.
    """
    # A PLAIN TUPLE, never a Body: the incoming sequence is exactly the mixed stmt/term stream a
    # body may not hold (``Body.__new__`` refuses a non-``Stmt``), and separating it is this
    # function's whole job. Only the term-free remainder becomes a Body, at the end.
    members = tuple(members)
    scalar_seen = False
    split = None
    for index, stmt in enumerate(members):
        if isinstance(stmt, Fold):
            if scalar_seen:
                split = index
                break
        else:
            scalar_seen = True

    if split is not None:
        prefix, suffix = members[:split], members[split:]
        needed = set(results)
        for stmt in suffix:
            needed.update(_member_reads(stmt))
        bridge = tuple(name for stmt in prefix for name in stmt.defines() if name in needed)
        # A prefix that feeds nothing needs no nesting: it is DEAD with respect to the suffix, so
        # there is no ordering to preserve and the plain separation below is correct — the terms
        # become operands and the dead stmts stay body members for a later pass to judge. Asserting
        # a bridge here rejected that shape; the ordering rule only has something to say when the
        # prefix actually provides.
        if bridge:
            source = _ordered_projection(prefix, bridge, axes)
            return _ordered_projection((source, *suffix), results, axes)

    operands = _unique_edges(tuple(stmt for stmt in members if isinstance(stmt, Fold)))
    body = Body(stmt for stmt in members if not isinstance(stmt, Fold))
    params = tuple(name for edge in operands for name in _operand_result_names(edge))
    scope = tuple(axis for axis in axes if axis not in params)
    return Fold(axis=None, operands=operands, lift=Lambda.closing((*params, *scope), body, tuple(results)))


def operand_body(op) -> tuple[Stmt, ...]:
    """An operand edge's producing stmts — the singleton gmem ``Load``, or the inline node
    flattened (:meth:`Fold.lower`). A free function, not a per-role helper on the node: an edge is
    an edge, and whether it is the shared or a channel operand is the Fold's reading of operand
    order, not a property of the edge itself."""
    return (op,) if isinstance(op, Load) else tuple(op.lower())


def operand_name(op) -> str:
    """An operand edge's bound SSA name — the inline node's ``out``, or the ``Load``'s def. Free
    for the same reason as :func:`operand_body`."""
    return op.defines()[-1] if isinstance(op, Load) else op.out


def _operand_result_names(op) -> tuple[str, ...]:
    """An operand edge's bound RESULT names — one per produced component. A single-valued edge
    (a gmem ``Load``) is the 1-tuple of :func:`operand_name`; a product fold / multi-channel
    node exposes EVERY component, so a wrapping zero-axis fold's synthesized params bind one name per
    component (the 1q params-flattening fix: the geglu combine reads BOTH ``acc_g`` and
    ``acc_u`` from one source — before the flattening the second component reached the lambda
    as a free name)."""
    if isinstance(op, Load):
        return (op.defines()[-1],)
    if isinstance(op, Fold):
        if op.axis is None:
            # The zero-axis reading: the projection's NAMED results (a store-only tail names
            # none, and falls back to its primary bound value).
            named = tuple(r for r in op.lift.results if isinstance(r, str))
            return named if named else (op.out,)
        # An observed fold's results include the observer's fresh names — the same interface
        # ``defines()`` exposes, so the enclosing scope's rename/identity walks see them.
        return tuple(op.defines()) if op.observe is not None else tuple(op.combine.results)
    return (operand_name(op),)


def subst_free(stmt: Stmt, sigma: Sigma) -> Stmt:
    """:func:`~emmy.compiler.ir.stmt.passes.rewrite`'s σ-substitution made **hygienic**, the way
    ``rename_free`` makes the SSA rename hygienic: the substitution stops at a nested scope whose
    binder re-binds a substituted name.

    ``rewrite``'s σ descends into every nested body. But axis names collide across a tree by
    design (see :func:`~emmy.compiler.ir.pure.scope.edge_axes`), so an occurrence under a ``Loop`` / reducing ``Fold``
    that re-binds a substituted name is a DIFFERENT variable: substituting it rewires the inner
    reduction onto the outer coordinate (the k-norm inside attention's K operand cone re-binds
    the contraction axis; a blind σ made its 128-element reduce read one slab element 128 times).
    Use this — not a bare ``rewrite`` — wherever σ carries axis coordinates into stmts that may
    re-bind them, such as the smem compute fill's per-cell cone evaluation."""
    if not sigma.mapping:
        return stmt
    bound = stmt.binds_axes() & sigma.mapping.keys()
    inner = Sigma({k: v for k, v in sigma.mapping.items() if k not in bound}) if bound else sigma
    if isinstance(stmt, Fold):
        # Every part of a reducing Fold — operand edges included — evaluates per step, inside its
        # binder; and a contraction's ``nested()`` is empty by design, so the generic body walk
        # below could not reach its cones.
        if not inner.mapping:
            return stmt
        lift = Lambda(stmt.lift.params, Body(tuple(subst_free(s, inner) for s in stmt.lift.body)), stmt.lift.results)
        observe = stmt.observe
        if observe is not None:
            observe = Lambda(observe.params, Body(tuple(subst_free(s, inner) for s in observe.body)), observe.results)
        return replace(stmt, operands=tuple(subst_free(e, inner) for e in stmt.operands), lift=lift, observe=observe)
    bodies = stmt.nested()
    renamed = stmt.rewrite(lambda nm: nm, sigma)  # header exprs (a StridedLoop's start/step) sit outside the binder
    if not bodies:
        return renamed
    return renamed.with_bodies(tuple(Body(tuple(subst_free(c, inner) for c in b)) for b in bodies))


@dataclass(frozen=True)
class Channel:
    """One product channel of a contraction — the streamed K×N operand edge ``b`` plus the
    additive fold accumulator ``acc`` that channel produces. A plain matmul is one channel; the
    fused gate⊗up MLP edge is two channels over the node's single shared ``a`` (sharing is arity,
    not naming — the product-carrier contraction outputs a tuple)."""

    b: Load | Fold  # the streamed operand edge — MATERIALIZED or COMPUTED
    acc: str  # this channel's fold accumulator


def _map_results(body: Body) -> tuple[str, ...]:
    """The synthesized projection result: the body's last definition, if any."""
    for s in reversed(body):
        d = s.defines()
        if d:
            return (d[-1],)
    return ()


# ``Body.structural_key()`` dispatches :func:`emmy.compiler.ir.stmt.passes.rewrite` over every
# stmt for SSA / Expr / axis canonicalization. Register the structural node's handler here — an
# INLINE node operand dispatches back through the same registry, so a stored computed operand
# (the cone, flash's ``P``) canonicalizes like any other subtree.
from emmy.compiler.ir.stmt.passes import rewrite as _rewrite  # noqa: E402


@_rewrite.register
def _(s: Fold, rename, sigma, axis_fn):
    # ONE handler for the one stored kind (the collapse retired the Map / Contraction arms —
    # singledispatch keys on the stored type, and there is now only one). Every operand edge
    # dispatches back through the registry; the fold renames its lift / monoid in lockstep
    # (params track the operand names positionally, the combine's results ARE the accumulator
    # names). At zero axes there is no iteration var to rename and no monoid to thread.
    operands = tuple(_rewrite(edge, rename, sigma, axis_fn) for edge in s.operands)
    axis = axis_fn(s.axis) if s.axis is not None else None
    lead = (axis.name,) if axis is not None else ()
    from emmy.compiler.ir.expr import Var  # noqa: PLC0415 — the rewrite tail's axis check

    def _param(name: str) -> str:
        # The environment tail may hold AXIS names, which rename through sigma (the body's own
        # coordinate substitution), not through the SSA renamer. Taking sigma's answer when it has
        # one keeps a param and every use of it in the body spelled the same way.
        mapped = sigma.get(name) if sigma is not None else None
        return mapped.name if isinstance(mapped, Var) else rename(name)

    lift = Lambda(
        params=(*lead, *(_param(p) for p in s.lift.params[len(lead) :])),
        body=Body(tuple(_rewrite(st, rename, sigma, axis_fn) for st in s.lift.body)),
        results=tuple(rename(r) if isinstance(r, str) else r for r in s.lift.results),
    )
    combine = rename_combine(s.combine, rename) if s.combine is not None else None
    observe = None
    if s.observe is not None:
        # The observer renames in lockstep: param 0 tracks the axis, the state params track the
        # combine's renamed results, the body/results are ordinary SSA material.
        observe = Lambda(
            params=(axis.name, *(rename(p) for p in s.observe.params[1:])),
            body=Body(tuple(_rewrite(st, rename, sigma, axis_fn) for st in s.observe.body)),
            results=tuple(rename(r) if isinstance(r, str) else r for r in s.observe.results),
        )
    return replace(s, axis=axis, operands=operands, lift=lift, combine=combine, observe=observe)


__all__ = [
    "Channel",
    "deep_defines",
    "deep_reads",
    "Fold",
    "is_contraction",
    "loaded_buffers",
    "operand_body",
    "operand_name",
    "splice_operands",
    "subst_free",
]
