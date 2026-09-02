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

from dataclasses import dataclass, field, replace
from functools import cached_property

from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.pure.algebra import component_ops, family_of, rename_combine
from emmy.compiler.ir.pure.lam import Lambda
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Stmt

# ``Body.structural_key()`` dispatches :func:`emmy.compiler.ir.stmt.passes.rewrite` over every
# stmt for SSA / Expr / axis canonicalization. Register the structural node's handler here — an
# INLINE node operand dispatches back through the same registry, so a stored computed operand
# (the cone, flash's ``P``) canonicalizes like any other subtree.
from emmy.compiler.ir.stmt.passes import _rewrite_kind
from emmy.compiler.ir.stmt.passes import rewrite as _rewrite  # noqa: E402


@dataclass(frozen=True)
class ContractionView:
    """A Fold's BILINEAR reading, as geometry: the axis its operands share and the free axis each
    one contributes.

    ``a[m,k] × b[n,k]`` reads as ``axis=k, left=m, right=n``. That is the whole of the recognition
    — no algebra walk, no product-argument matching, no canonical form to compare against. Which
    of ``left`` / ``right`` is physically M or N is the PLACEMENT's answer, not the term's.
    """

    axis: Axis
    left: str
    right: str
    b_trans: bool = False
    """Whether the streamed operand is stored N-major — its reduction axis LAST, ``B[n, k]``,
    against the canonical ``B[k, n]``. A gmem LAYOUT fact, so it is meaningful only for a
    materialized slab; a computed B answers ``False``. A stored on the same side is not a
    question: ``operands[0]`` is A by canonical form, k-last, which is what normalization
    guarantees by swapping the pair."""


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
    axes: tuple[Axis, ...] = ()
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
            arity = sum(len(edge.exposes) for edge in self.operands)
            assert len(self.lift.params) >= arity, f"lift binds {len(self.lift.params)} params for {arity} operand result components"
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
        # POSITIONAL, so checked positionally: param i+1 is operand result component i. Comparing
        # NAMES here is what coupled a consumer's params to its producers' spelling — it is why an
        # edge could not be canonicalized without breaking the term above it, and why reading an
        # edge's result names was load-bearing at the constructor.
        arity = sum(len(edge.exposes) for edge in self.operands)
        assert len(lam.params) >= 1 + arity, f"lift binds {len(lam.params) - 1} params after the axis for {arity} operand result components"
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
        if not family.twisted:
            return  # the componentwise family — nothing further to validate
        # TWISTED: the state-component ROLE decision is shape-derived off the lift's injected
        # singleton, no annotation: the pivot is component 0 (its injected term the score), a
        # literal-1 injection is a denominator, a value injection an expectation.
        assert isinstance(lam.results[0], str), "the twisted lift's pivot component must inject the score name"

    @property
    def exposes(self) -> tuple[str, ...]:
        """The result names this term binds into its consumer — one per produced component.

        Asked of the TERM, not recovered by a free function branching on what the edge happens to
        be: a zero-axis term exposes its named lift results, a reducing one its carried state, and
        an observed one its observer's fresh names beside it. The arity is what the positional
        binding law is about; the names are what the consumer's lift still spells, and they go once
        the binding is positional in fact as well as in intent.
        """
        if self.combine is None:
            return tuple(result for result in self.lift.results if isinstance(result, str))
        state = tuple(self.combine.results)
        return state if self.observe is None else (*state, *(result for result in self.observe.results if isinstance(result, str)))

    @cached_property
    def index_space(self) -> frozenset[str]:
        """Every coordinate this term is evaluated over — its own :attr:`axes` plus its operands'.

        DECLARED, not discovered: a term states its index space, so a reader intersects instead of
        walking a lowered body for names that look axis-shaped. Every operand is a term — a gmem
        read is a :meth:`slab` — so this is a union of declarations with nothing to dispatch on.
        """
        space = {axis.name for axis in self.axes}
        for edge in self.operands:
            space |= edge.index_space
        return frozenset(space)

    def as_contraction(self) -> ContractionView | None:
        """The :class:`ContractionView` of this term, or ``None`` when it is not bilinear.

        ALGEBRA names the pair, GEOMETRY names the axes. The two readings are not
        interchangeable: ``sum_k(a[m,k] * b[k,n])`` and ``sum_k(a[m,k] + b[k,n])`` have identical
        axes and identical free roles, so reading coordinates alone hands an addition to the
        tensor-core tier. :attr:`semiring` settles what the carrier is — one shared ⊗ per result
        distributing over a commutative-monoid ⊕ — and each product's ARGUMENTS say which operand
        edges it multiplies.

        The reduction is then what that pair SHARES, and each side's own free axis is the output
        role it carries. Sharing is not exclusive to the reduction: a batch axis rides both
        operands and stays free (``Q[b,h,m,d] x K[b,h,n,d]`` shares ``{b,h,d}``, reduces ``d``),
        so the fold's axis must be AMONG the shared axes rather than all of them.

        Every product reads ``operands[0]`` — A by canonical form — which is what makes the fused
        multi-channel edge one contraction over one shared A rather than several.
        """
        if self.axis is None or len(self.operands) < 2 or self.semiring is None:
            return None
        by_name = {name: edge for edge in self.operands for name in edge.exposes}
        a_edge = self.operands[0]
        a_names = set(a_edge.exposes)
        streamed = []
        for product in self.lift.body:
            other = set(product.args) - a_names
            if len(set(product.args) & a_names) != 1 or len(other) != 1:
                return None  # a product that does not multiply A by exactly one other edge
            edge = by_name.get(next(iter(other)))
            if edge is None or edge is a_edge:
                return None
            streamed.append(edge)

        b_edge = streamed[0]
        a_space, b_space = a_edge.index_space, b_edge.index_space
        if self.axis.name not in a_space & b_space:
            return None
        left_only, right_only = a_space - b_space, b_space - a_space
        if len(left_only) != 1 or len(right_only) != 1:
            return None
        b_trans = b_edge.is_slab and self.axis.name in b_edge.lift.body[0].index[-1].free_vars()
        return ContractionView(
            axis=self.axis, left=next(iter(left_only)), right=next(iter(right_only)), b_trans=b_trans
        )

    @property
    def is_slab(self) -> bool:
        """Whether this term is a wrapped ``Load`` — a gmem read that declares its coordinates.

        A leaf, not a computed cone: no operands, no monoid, a body of one ``Load``. This is what
        ``isinstance(edge, Load)`` used to ask, back when a statement could sit in ``operands``.
        Placement never offers one as a seam: there is nothing to materialize that the load does
        not already do.
        """
        return not self.operands and self.combine is None and len(self.lift.body) == 1 and isinstance(self.lift.body[0], Load)

    @property
    def loads(self) -> tuple[Stmt, ...]:
        """Every ``Load`` beneath this term — its operands', recursively, then its lift body's.

        The term's own :attr:`Body.loads`, which cannot reach them: a body walks ``Stmt.nested()``
        and a term's operands are EDGES, not nested statements. This is the reading a consumer of
        the STORED tree needs — the cut declaring a piece's graph inputs — and it answers without
        lowering, so it costs a walk of the tree rather than a construction of its statements.
        """
        return tuple(load for edge in self.operands for load in edge.loads) + tuple(self.lift.body.loads)

    @property
    def axis(self) -> Axis | None:
        """The REDUCTION axis — the innermost of :attr:`axes`, ``None`` when this term binds none.

        A property, not a field: :attr:`axes` is the term's whole iteration space, and every
        existing reading of ``axis`` (``axis is None`` meaning "does not reduce") keeps answering
        unchanged while the space itself grows a place to live. A term that iterates WITHOUT
        reducing — a wrapped ``Load``, whose coordinates are its own binders rather than free
        names — is what ``axes`` exists for, and it is told apart by :attr:`combine`, not here.
        """
        return self.axes[-1] if self.axes and self.combine is not None else None

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
        if self.as_contraction() is not None:
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
        return inner if inner.as_contraction() is not None else None

    # ---- the DERIVED READINGS. ``Map`` and ``Contraction`` are no longer stored kinds (the
    # collapse); every field they carried reads back off the one stored term here, so their old
    # accessors keep their exact meanings and their consumers keep their exact spellings. ------- #
    @classmethod
    def slab(cls, load: Load, axes: tuple[Axis, ...]) -> Fold:
        """Wrap one ``Load`` as a term that DECLARES the coordinates it reads.

        A ``Load`` is a statement, and a statement sitting in a term tree is the one leaf whose
        coordinates are still free names — the last place a coordinate escapes its binder. Wrapped,
        the leaf binds them: :attr:`axes` is its index space, taken from the enclosing ``axes`` it
        actually indexes, in their binding order. Every edge in the tree then answers ``axes`` as a
        FIELD, and a consumer's reduction axis is the intersection of its operands' — no walk.

        The coordinates are the lift's PARAM PREFIX, one per declared axis and in the same order,
        because a leaf cannot tell an axis from a value — a gather index ``weight[(int)in0, a]``
        genuinely reads ``in0`` — and only the binder can. Declaring them as axes is what the old
        untyped trailing residue could not do: the term binds them, applies them by iterating, and
        every reader gets the answer from :attr:`axes` instead of inferring it. No ``combine``: a
        slab iterates, it does not reduce.
        """
        coordinates = {name for expr in load.exprs() for name in expr.free_vars()}
        declared = tuple(axis for axis in axes if axis.name in coordinates)
        missing = coordinates - {axis.name for axis in declared}
        if missing:
            raise ValueError(f"Load indexes {sorted(missing)}, which the enclosing binder does not supply as an axis")
        return cls(
            axes=declared,
            lift=Lambda(params=tuple(axis.name for axis in declared), body=Body((load,)), results=tuple(load.names)),
        )

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
        """Flatten this term to the Loop IR body the materializer expands.

        Three parts, read straight off the representation:

        * every operand is a TERM, so it lowers by the same method — the tree is homogeneous and
          there is nothing to dispatch on;
        * an operand whose :attr:`index_space` does not contain this fold's axis does not vary
          with the step, so it lowers ONCE ahead of the loop. That is the whole of the hoist: a
          declaration compared against an axis, not a body walked for free names;
        * without a ``combine`` the term is a map and the step IS the answer; with one, the step
          folds into an ``Accum`` per carried component and the loop binds the axis.

        The ONE lowering spelling — every consumer of a term's statements calls this.
        """
        axis = self.axis
        rides = [edge for edge in self.operands if axis is not None and axis.name in edge.index_space]
        ridden = {id(edge) for edge in rides}
        prologue = [stmt for edge in self.operands if id(edge) not in ridden for stmt in edge.lower()]
        step = [stmt for edge in rides for stmt in edge.lower()] + list(self.lift.body)
        if self.combine is None:
            return [*prologue, *step]
        ops = component_ops(self.combine)
        assert ops is not None, "a lowerable fold carries a componentwise combine"
        step += [
            Accum(name=state, value=value, op=op, axes=(axis.name,))
            for state, value, op in zip(self.combine.results, self.lift.results, ops, strict=True)
        ]
        return [*prologue, Loop(axis=axis, body=Body(step), unroll=self.unroll, role=self.role)]

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


@dataclass(frozen=True)
class Channel:
    """One product channel of a contraction — the streamed K×N operand edge ``b`` plus the
    additive fold accumulator ``acc`` that channel produces. A plain matmul is one channel; the
    fused gate⊗up MLP edge is two channels over the node's single shared ``a`` (sharing is arity,
    not naming — the product-carrier contraction outputs a tuple)."""

    b: Load | Fold  # the streamed operand edge — MATERIALIZED or COMPUTED
    acc: str  # this channel's fold accumulator


# ``Body.structural_key()`` dispatches :func:`emmy.compiler.ir.stmt.passes.rewrite` over every
# stmt for SSA / Expr / axis canonicalization. Register the structural node's handler here — an
# INLINE node operand dispatches back through the same registry, so a stored computed operand
# (the cone, flash's ``P``) canonicalizes like any other subtree.


@_rewrite_kind.register
def _(s: Fold, rename, sigma, axis_fn):
    # ONE handler for the one stored kind (the collapse retired the Map / Contraction arms —
    # singledispatch keys on the stored type, and there is now only one). Every operand edge
    # dispatches back through the registry; the fold renames its lift / monoid in lockstep
    # (params track the operand names positionally, the combine's results ARE the accumulator
    # names). At zero axes there is no iteration var to rename and no monoid to thread.
    operands = tuple(_rewrite(edge, rename, sigma, axis_fn) for edge in s.operands)
    axes = tuple(axis_fn(axis) for axis in s.axes)
    axis = axes[-1] if axes else None
    lead = (axis.name,) if axis is not None else ()

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
    return replace(s, axes=axes, operands=operands, lift=lift, combine=combine, observe=observe)


__all__ = [
    "Channel",
    "Fold",
]
