"""``Fold`` — the ONE reduce term: ``reduce(⊕) ∘ map(f)`` in the λ-foldMap spelling.

The whole stored vocabulary of Tile IR, and a PURE term throughout: an optional iteration
``axis``, a pure ``lift`` :class:`~emmy.compiler.ir.pure.lam.Lambda`, the monoid's ``(init, base)``
pair with the optional ``twist`` recipe that conjugates it, and a tuple of ``operands`` — the
closed inputs, each an edge bound positionally to a lift param. Every reading (the ⊕ itself, the
map at zero axes, the bilinear :class:`ContractionView`, the :class:`SlabView` leaf, the serial
step) is DERIVED from those params; nothing else is stored.

The fold's algebra is the twisted monoid of :mod:`~emmy.compiler.ir.pure.twist`, spelled in one
vocabulary with the recipes: ``lift`` is the per-element contribution to the BASE state, ``base``
the base monoid's componentwise ⊕, and ``twist`` the recipe carrying the bijection ``psi`` onto the
stable carrier. The ⊕ the fold actually folds with is then derived —
``combine(x, y) = psi(psi_inv(x) base psi_inv(y))`` — in the stable spelling the recipe certifies,
and ``twist=None`` is the planar case where ``psi`` is the identity and ``combine`` IS ``base``.

Nothing here is a :class:`~emmy.compiler.ir.stmt.base.Stmt`. A composed step — flash's ``Σ Q·K``
ahead of its ``Σ_j P·V``, split-K's sliced contraction — is reached through ``operands``, and its
POSITION in the emitted nest is produced by the derivation (:meth:`Fold.lower` places every term
at the shallowest scope binding its free coordinates, operands ahead of their readers), not by
sitting in a statement list. The term becomes statements in exactly one place, :meth:`Fold.lower`,
which also places the kernel's boundary stores, each after the term defining its value.
See ``ir/ARCHITECTURE.md``, "Pure terms vs statements".

The schedule is deliberately absent: an accepted, site-indexed ``Schedule`` lives on the
``TileOp`` boundary (``ir/tile/ir.py``), so the term is IMMUTABLE across the whole schedule search
and kernel identity — the Loop IR the term lowers to — is the algebra alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.pure.lam import Lambda
from emmy.compiler.ir.pure.twist import Recipe, Twist
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, OutputSpec, Stmt, StridedLoop
from emmy.compiler.ir.stmt.body import free_names

# ``Body.structural_key()`` dispatches :func:`emmy.compiler.ir.stmt.passes.rewrite` over every
# stmt for SSA / Expr / axis canonicalization. Register the structural node's handler here — an
# INLINE node operand dispatches back through the same registry, so a stored computed operand
# (the cone, flash's ``P``) canonicalizes like any other subtree.
from emmy.compiler.ir.stmt.passes import _rewrite_kind
from emmy.compiler.ir.stmt.passes import rewrite as _rewrite  # noqa: E402
from emmy.utils import cached_method


@dataclass(frozen=True)
class SlabView:
    """A Fold's reading as one gmem read — a leaf evaluated over the coordinates it indexes.

    ``load`` is the read itself. No operands, no monoid, a body of one ``Load``: what
    ``isinstance(edge, Load)`` used to ask, back when a statement could sit in ``operands``.
    Placement never offers one as a seam — there is nothing to materialize that the load does not
    already do.
    """

    load: Load


@dataclass(frozen=True)
class ContractionView:
    """A Fold's BILINEAR reading, as geometry: the axis its operands share and the free axis each
    one contributes.

    ``a[m,k] × b[n,k]`` reads as ``axis=k, left=m, right=n``. That is the whole of the recognition
    — no algebra walk, no product-argument matching, no canonical form to compare against. Which
    of ``left`` / ``right`` is physically M or N is the PLACEMENT's answer, not the term's.
    """

    axis: str
    left_axes: frozenset[str]
    """The free axes A carries and B does not — the output role it strides, plus any broadcast
    batch coordinate riding A alone (``x[b, m, k]`` against ``w[n, k]``): which of them is the
    row is the placement's answer, the others are grid offsets."""
    right_axes: frozenset[str]
    """The free axes the streamed operand carries alone. Empty for a MATVEC, whose B is a vector
    over the reduction alone: a contraction does not stop being one for want of a second output
    axis; whether the pair can be ORIENTED as (m, n) is the placement's question, not
    recognition's. A pair with NO output role on either side (a row's dot product with itself,
    the RMS statistic spelled as two casts of one load) is not a contraction at all: it reads as a
    planar reduce."""
    shared_axes: frozenset[str] = frozenset()
    """The free axes BOTH operands read besides the reduction: a batch the pair rides (attention's
    ``b, h``), a split-K partition coordinate, a value-dead reshape residue — or, when a side
    carries no role of its own, the row that side's every element moves with. Which of those a
    shared axis is takes the kernel's extents to tell, so it is the tile's question
    (:meth:`TileOp.contracts`), not the term's."""
    product: ElementwiseImpl | None = None
    """The ⊗ this contraction multiplies its operand pair with."""
    plus: ElementwiseImpl | None = None
    """The commutative-monoid ⊕ the products fold through. Together with :attr:`product` this is
    the semiring INSTANCE — the mma tier gates on ``(multiply, add)``, and a new instance reads
    back here without change."""
    b_trans: bool = False
    """Whether the streamed operand is stored N-major — its reduction axis LAST, ``B[n, k]``,
    against the canonical ``B[k, n]``. A gmem LAYOUT fact, so it is meaningful only for a
    materialized slab; a computed B answers ``False``. A stored on the same side is not a
    question: ``operands[0]`` is A by canonical form, k-last, which formation guarantees by
    orienting the pair."""

    @property
    def left(self) -> str | None:
        """A's one own axis, or ``None`` when it carries none or several (a broadcast batch)."""
        return next(iter(self.left_axes)) if len(self.left_axes) == 1 else None

    @property
    def right(self) -> str | None:
        """B's one own axis, or ``None`` when it carries none or several."""
        return next(iter(self.right_axes)) if len(self.right_axes) == 1 else None


@dataclass(frozen=True)
class ReductionView:
    """A Fold's reading as a MONOID fold — the algebra its combine spells, read off the lambda.

    ``states`` are the carried state's names (the combine's results), ``other`` the names the
    combine binds its second operand to, ``terms`` the injection singleton (the lift's results,
    one per component). ``ops`` is the componentwise op vector when the combine has that shape —
    every result ``sᵢ = ⊕ᵢ(sᵢ, sᵢ′)`` independently, a PLANAR fold (sum, max, mean) — and ``None``
    for a TWISTED one (online softmax's rescaling program): the one discrimination every partition
    legality question asks. No family name: the program itself is the reading, and
    :meth:`Fold.merge` applies it.
    """

    axis: str
    states: tuple[str, ...]
    other: tuple[str, ...]
    terms: tuple[str, ...]
    ops: tuple[ElementwiseImpl, ...] | None

    @property
    def twisted(self) -> bool:
        return self.ops is None


@dataclass(frozen=True)
class Fold:
    """The ONE reduce term — ``reduce(⊕) ∘ map(f)``, the typed successor of the annotated reduce
    ``Loop``. It splits the reduce's **algebra** (the base monoid ``base``, plus the ``twist``
    recipe whose ``psi`` conjugates it into the rescaling program online-softmax / flash folds
    with — :attr:`combine` derives that ⊕, and a planar ``sum`` / ``max`` / ``mean`` is the
    identity case) from its **structure** (the axis the lift binds and the per-element ``step`` it
    folds). Every reading is **derived** from the stored params (:attr:`combine`,
    :meth:`as_contraction`, :meth:`as_slab`, :meth:`as_reduction`, :meth:`step`), never stored, and the loop nest is
    **synthesized on demand** (:meth:`lower`), never stored — so the same term tiles under any
    :class:`~emmy.compiler.ir.schedule.Reduce`, which is not a field here: the reduce partition is
    a site choice in ``TileOp.schedule``, read through ``ops.Sched``.

    Everything a term reads arrives through its ``operands``, bound POSITIONALLY to the lift's
    params (:attr:`bindings`) — a gmem read as a slab, a nested reduce as the term itself, a
    projection as a zero-axis term — so a reduce that reads another (flash's ``Σ Q·K`` score ahead
    of its ``Σ_j P·V``) is one term with the other among its operands, and its position in the
    emitted nest is :meth:`lower`'s to place. A term holds no projection of its own: a bare reduce
    (``sum`` / ``max``) is the kernel root, and a reduce with a post-fold sweep (softmax / RMSNorm)
    is an operand of the zero-axis term whose lift is that projection.

    The reduce PARTITION (:class:`Reduce` — GRID split / BLOCK coop / REG ILP) is the schedule's,
    not the term's: it is selected for the term's site in ``TileOp.schedule`` and read through
    ``ops.Sched``, which is why ``lower`` cannot see it and ``identity_key(with_io=True, with_knobs=True)``
    stays byte-identical whichever partition the fork picked."""

    pure = True  # a term is a value — its internals are its own; legal inside a stored ``Lambda``
    deps_deep = True  # :meth:`deps` is the memoized scoped rollup — read walks must not re-walk the lift

    # The CLOSED inputs, each an operand edge (a slab or an inline node) — the 1k fold
    # vocabulary. Sharing is edge reuse: the step reads an operand's bound name as many times as it
    # needs. ``lower`` places each edge at the shallowest scope binding its coordinates, ahead of its reader.
    operands: tuple[Fold, ...] = ()
    # NO schedule fields: node and edge choices live in ``TileOp.schedule`` — the term is pure
    # algebra, IMMUTABLE across the whole schedule search (a fork is a different assignment,
    # never a rebuilt tree).
    # A cross-CTA SLICE of the stream (flash split-KV) is not spelled here: ``030_cut``
    # shrinks ``axis`` to the slice length and the slice's absolute base / end ride that axis's
    # :class:`~emmy.compiler.ir.axis.Window` — ONE windowing vocabulary, the same one an axis's
    # split parentage uses, read by the realizer and the mask machinery alike.
    #
    # ---- the λ-foldMap spelling — the fold's storage: a PURE ``lift`` ``λ(k, v₁…vₙ) → B``
    # (params: the iteration var first, then one per operand result component, bound POSITIONALLY —
    # the names are this term's own, :attr:`bindings` pairs them with the edges, and :attr:`applied`
    # spells them as the operands' results for every renderer) plus the monoid's ``(init, base)``
    # pair, whose ``base`` carries the REAL accumulator names (its results), and the ``twist``
    # recipe that conjugates it. The ⊕ itself (:attr:`combine`), the serial step and the ``Accum``
    # forms are DERIVED (:attr:`combine` / :attr:`step` / ``__post_init__``). ------------------- #
    lift: Lambda = field(kw_only=True)  # the per-element contribution; CLOSED by ``Lambda.__post_init__``
    init: tuple[float, ...] = ()  # the ⊕ seeds — op identities for a plain fold; (−inf, 0, …) LSE
    # The BASE monoid's ⊕, as the componentwise program ``base = componentwise(ops, states)`` — one
    # op per carried component, and its RESULTS are the carrier's state names. ``None`` at zero
    # axes, where there is no monoid at all.
    base: Lambda | None = field(kw_only=True, default=None)
    # The recipe this carrier INSTANTIATES — the schema plus this term's spelling of its roles
    # (:class:`~emmy.compiler.ir.pure.twist.Twist`) — or ``None`` for a planar fold. It supplies
    # the bijection ``psi`` onto the stable carrier, and with it the two halves the term does not
    # store: ``combine(x, y) = psi(psi_inv(x) base psi_inv(y))`` (:attr:`combine`) and
    # ``psi ∘ lift``, the singleton that ⊕ folds (:attr:`injected`). NAMED rather than restated:
    # stability is not preserved by conjugation, so neither can be computed from ``psi`` alone.
    twist: Twist | None = field(kw_only=True, default=None)
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
        # EVERY OPERAND IS A TERM. A gmem read is a term over one ``Load``, so the tree is homogeneous and a
        # per-edge question is an attribute rather than a helper dispatching on what an edge
        # happens to be. Stated here because a statement in ``operands`` does not announce itself:
        # an ``isinstance(edge, Load)`` against a type that can no longer appear reads as False,
        # not as an error, so the check it guards disappears silently. Formation is the one place
        # that can say no — this caught the cut's workspace reads, which were bare ``Load``s.
        stray = [type(edge).__name__ for edge in self.operands if not isinstance(edge, Fold)]
        if stray:
            raise TypeError(f"Fold operands must be terms, got {stray}; a gmem read is a term over one Load")
        if self.base is None:
            # The ZERO-AXIS node: no iteration and no monoid, so the only formation fact is the
            # positional binding — one lift param per operand RESULT COMPONENT, no leading
            # iteration var. (The projection (zero-axis) fold was exactly this, with ``fn`` for ``lift``.)
            assert not self.init and self.twist is None, "a zero-axis Fold carries no monoid to seed or twist"
            assert self.observe is None, "a zero-axis Fold carries no per-step state to observe"
            # An operand that another operand is built OVER and passes through whole (the small
            # cone beside the larger one computed from it, both bound by their own names) is that
            # operand's shadow: the maximal one stays, its params go with it, and the body reads
            # the pass-through by the same name.
            slots, cursor = [], 0
            for edge in self.operands:
                slots.append(tuple(self.lift.params[cursor : cursor + len(edge.exposes)]))
                cursor += len(edge.exposes)
            shadowed = [
                index
                for index, edge in enumerate(self.operands)
                if slots[index] == edge.exposes
                and any(
                    other is not edge
                    and any(edge is held for held in other.operands)
                    and set(edge.exposes) < set(other.exposes)
                    and slots[position] == other.exposes
                    for position, other in enumerate(self.operands)
                )
            ]
            if shadowed:
                kept, params, cursor = [], [], 0
                for index, edge in enumerate(self.operands):
                    width = len(edge.exposes)
                    if index not in shadowed:
                        kept.append(edge)
                        params.extend(self.lift.params[cursor : cursor + width])
                    cursor += width
                object.__setattr__(self, "operands", tuple(kept))
                object.__setattr__(self, "lift", replace(self.lift, params=(*params, *self.lift.params[cursor:])))
            arity = sum(len(edge.exposes) for edge in self.operands)
            assert len(self.lift.params) >= arity, f"lift binds {len(self.lift.params)} params for {arity} operand result components"
            return
        # Formation validates the positional binding and the S × S → S arity; the planar-vs-twisted
        # reading is DERIVED (:meth:`as_reduction`), never a second stored spelling.
        n = len(self.init)
        if len(self.base.params) != 2 * n or len(self.base.results) != n or self.base.components() is None:
            raise ValueError(f"Fold base must be the componentwise ⊕ at arity {n}: params={self.base.params} results={self.base.results}")
        lam = self.lift
        assert lam.params, "a reducing lift binds its iteration var first"
        # One lift param per operand RESULT COMPONENT (a product edge — split-K's sliced
        # multi-channel fold — binds every component), positionally.
        # POSITIONAL, so checked positionally: param i+1 is operand result component i. Comparing
        # NAMES here is what coupled a consumer's params to its producers' spelling — it is why an
        # edge could not be canonicalized without breaking the term above it, and why reading an
        # edge's result names was load-bearing at the constructor.
        arity = sum(len(edge.exposes) for edge in self.operands)
        assert len(lam.params) >= 1 + arity, f"lift binds {len(lam.params) - 1} params after the axis for {arity} operand result components"
        assert len(lam.results) == n, "one lift result per monoid component"
        # CANONICAL ORIENTATION of a bilinear term: A — the operand every product multiplies —
        # comes first, so ``operands[0]`` IS A by construction. With several channels (the fused
        # gate⊗up edge) A is the argument the products SHARE, and that names it outright; with one
        # product both arguments are shared, so the layout rule decides: A reads ``[…, k]``, its
        # reduction axis last, which lets a fragment load stride its rows contiguously. Binding is
        # positional, so the lift's params move with the operands; the body reads by name.
        if len(self.operands) >= 2 and lam.body and all(isinstance(stmt, Assign) and len(stmt.args) == 2 for stmt in lam.body):
            by_name = {param: edge for param, edge, _ in self.bindings}
            arguments = [product.args for product in lam.body]
            if len(arguments) > 1:
                shared = [by_name[name] for name in arguments[0] if name in by_name and all(name in args for args in arguments[1:])]
                a_edge = shared[0] if len(shared) == 1 else None
            else:
                pair = [by_name[name] for name in arguments[0] if name in by_name]
                # Two SLABS orient by layout; a computed operand keeps the order its former chose.
                slabs = len(pair) == 2 and all(e.as_slab() is not None for e in pair)
                k_last = [e for e in pair if slabs and self.axis in e.as_slab().load.index[-1].free_vars()]
                # Both k-last (a matmul): A is the one reading the earlier free coordinate — the
                # row, under the lift's declaration order — so alpha-equal terms orient alike.
                k_last.sort(key=lambda e: min((n for n in e.free_axes if n != self.axis), default=""))
                a_edge = k_last[0] if len(pair) == 2 and k_last else None
            if a_edge is not None and self.operands[0] is not a_edge:
                reordered = (a_edge, *(edge for edge in self.operands if edge is not a_edge))
                bound = tuple(param for edge in reordered for param, held, _ in self.bindings if held is edge)
                object.__setattr__(self, "operands", reordered)
                object.__setattr__(self, "lift", replace(lam, params=(lam.params[0], *bound, *lam.params[1 + arity :])))
                del self.__dict__["bindings"]  # read before the reorder; the memo is stale
        if self.observe is not None:
            reading = self.as_reduction()
            assert reading is not None and not reading.twisted, (
                "a twisted carrier does not support a per-step observer: its state is rescaled, never a stream of partials"
            )
            assert tuple(self.observe.params) == (self.axis, *self.combine.results), (
                f"observer params {self.observe.params} must bind the iteration var then the carried state "
                f"{(self.axis, *self.combine.results)} positionally"
            )
            defined = {name for stmt in self.observe.body for name in stmt.defines()}
            assert all(r in defined for r in self.observe.results), (
                "observer results must be FRESH names its body defines — never the carried state itself "
                "(the boundary distinguishes a streamed store from a post-fold store by the name)"
            )
            assert not any(isinstance(stmt, Fold) for stmt in self.observe.body), "an observer body holds plain stmts, never a nested node"

    @cached_property
    def exposes(self) -> tuple[str, ...]:
        """The result names this term DEFINES for its consumer — one per produced component, in the
        spelling its rendered statements define: a reducing term its carried state (an observed one
        its observer's fresh names beside it), a zero-axis term its lift results, a result that
        merely passes an operand through spelled as that operand's own result. The arity is what
        the positional binding law is about; a consumer names what it binds itself
        (:attr:`bindings`), and never reads these names before the term is rendered.
        """
        if self.base is None:
            return self.applied.results
        state = self.base.results
        return state if self.observe is None else (*state, *self.observe.results)

    @cached_property
    def bindings(self) -> tuple[tuple[str, Fold, int], ...]:
        """The positional binding of the lift to the operands — for every operand-bound param in
        order, ``(param, edge, component)``: the lift's params past the iteration var bind the
        operands' result components in order. The names are this term's own; how an operand spells
        its results is nobody else's business until the term is rendered (:attr:`applied`).
        """
        params = self.lift.params[1 if self.base is not None else 0 :]
        out: list[tuple[str, Fold, int]] = []
        for edge in self.operands:
            out.extend((params[len(out)], edge, index) for index in range(len(edge.exposes)))
        return tuple(out)

    @cached_property
    def applied(self) -> Lambda:
        """The lift APPLIED to the operands: the same lambda with every operand-bound param spelled
        as the operand result it binds — the one reading a renderer of this term's statements takes
        (:meth:`step`, :attr:`exposes`, :meth:`lower`), so a lowered body reads producer names
        throughout and the binding is resolved exactly once.
        """
        return self.lift.rename({param: edge.exposes[index] for param, edge, index in self.bindings})

    @cached_property
    def read_operands(self) -> tuple[Fold, ...]:
        """The operands the term's rendered statements actually READ — what :meth:`lower` places.

        Every operand is a SITE the schedule may take, and a twisted carrier's weight cone is one
        the serial nest has no use for: ψ folds each element against the running pivot, so the
        absolute ``exp(score)`` the contraction channel multiplies is never evaluated there.
        Placing it anyway would emit a transcendental per element for nothing. An edge whose value
        this term passes through (a wrapper's operand, named among :attr:`exposes`) is read.
        """
        seen = set(self.step().ssa_uses) | set(self.exposes)
        return tuple(edge for edge in self.operands if not seen.isdisjoint(edge.exposes))

    def binds_axes(self) -> frozenset[str]:
        """The axis this term binds — what the statement-door ``rewrite`` drops from σ for the subtree."""
        return frozenset() if self.axis is None else frozenset({self.axis})

    @cached_property
    def combine(self) -> Lambda | None:
        """THE ⊕ this fold folds with — ``S × S → S``, ``None`` at zero axes. DERIVED, never
        stored, so the carrier has ONE spelling of its algebra:

        - planar (``twist is None``): ``combine = base``, the componentwise ⊕ itself;
        - twisted: ``combine = psi ∘ base ∘ (psi_inv × psi_inv)`` over the same states — the
          conjugate, in the numerically stable spelling the recipe authored and certifies.

        Memoized on the term, which is immutable."""
        if self.base is None or self.twist is None:
            return self.base
        return self.twist.program(self.base.results)

    @cached_property
    def injected(self) -> Lambda:
        """The lift SEEN THROUGH ψ — ``injected = psi ∘ lift``, the singleton the ⊕ folds, in the
        operands' spelling. ``applied`` itself for a planar fold, where ψ is the identity.

        For a twisted one the stored ``lift`` computes the BASE contribution, which is what makes a
        channel read as a contraction and what a schedule tiles; it is NOT what a serial step may
        emit, because ``Sum exp(score)`` overflows. So the step folds this instead: the score's own
        cone, then the recipe's authored injections (:meth:`Twist.inject`) — the simplified form ψ
        takes where the pivot IS the score. Nothing else may see through ψ.
        """
        applied = self.applied
        if self.twist is None or not self.twist.roles:
            return applied  # planar, or a merge whose elements are carrier states already
        spelled = dict(zip(self.lift.params, applied.params, strict=True))
        roles = tuple((role, spelled.get(name, name)) for role, name in self.twist.roles)
        injection = self.twist.inject(roles, self.base.results)
        score = injection.results[0]
        return Lambda(params=applied.params, body=Body((*applied.cone(score).body, *injection.body)), results=injection.results)

    @property
    def axis(self) -> str | None:
        """The name of the axis this term BINDS — its lift's iteration var, the first param — or
        ``None`` for a zero-axis term. A NAME: the extent and window are the evaluator's, held in
        the kernel's axis table (``TileOp.axes``) and handed to :meth:`lower`; a sum over 128 and
        one over 256 are the same term over different domains, like a slab under any M."""
        return self.lift.params[0] if self.base is not None else None

    @cached_property
    def free_axes(self) -> frozenset[str]:
        """The coordinates this term is evaluated over as seen from OUTSIDE — its free coordinates.

        DECLARED by the lift, not discovered: a lift is closed, so past the axis and the operand
        binding its params are exactly the coordinates its body reads (a closed lift captures no
        value — the loop lift turns one into an operand). Those, its operands' free coordinates,
        less the axis the term BINDS: a reduce over ``k`` does not vary with ``k`` to the fold that
        holds it, and a nested reduce that happens to bind the same name as the loop above still
        hoists. Their extents are the evaluator's, never the term's (:meth:`lower` takes them for
        the closed program).
        """
        space = set(self.lift.params[(self.axis is not None) + len(self.bindings) :])
        for edge in self.operands:
            space |= edge.free_axes
        return frozenset(space - ({self.axis} if self.axis is not None else set()))

    @cached_method
    def as_contraction(self) -> ContractionView | None:
        """The :class:`ContractionView` of this term, or ``None`` when it is not bilinear.

        ALGEBRA names the pair, GEOMETRY names the axes. The two readings are not
        interchangeable: ``sum_k(a[m,k] * b[k,n])`` and ``sum_k(a[m,k] + b[k,n])`` have identical
        axes and identical free roles, so reading coordinates alone hands an addition to the
        tensor-core tier. The semiring settles what the carrier is — one shared ⊗ per result
        distributing over a commutative-monoid ⊕, the lift body nothing but those products — and
        each product's ARGUMENTS say which operand edges it multiplies.

        The reduction is then what that pair SHARES, and each side's own free axis is the output
        role it carries. Sharing is not exclusive to the reduction: a batch axis rides both
        operands and stays free (``Q[b,h,m,d] x K[b,h,n,d]`` shares ``{b,h,d}``, reduces ``d``),
        so the fold's axis must be AMONG the shared axes rather than all of them.

        Every product reads ``operands[0]`` — A by canonical form — which is what makes the fused
        multi-channel edge one contraction over one shared A rather than several. Memoized on the
        term: it is immutable and every role, schedule and emission read
        asks this.
        """
        if self.axis is None or self.combine is None or len(self.operands) < 2:
            return None
        pluses = self.as_reduction().ops
        if not pluses or len(set(pluses)) != 1:
            return None
        plus = pluses[0]
        if not (plus.associative and plus.commutative and plus.has_identity) or self.init != (plus.identity,) * len(pluses):
            return None
        defs = self.lift.body.definitions
        products = [defs.get(result) for result in self.lift.results]
        if len(products) != len(pluses) or any(not isinstance(stmt, Assign) or len(stmt.args) != 2 for stmt in products):
            return None
        product = products[0].op
        if any(stmt.op != product or not stmt.op.distributes_over(plus) for stmt in products):
            return None
        if {id(stmt) for stmt in products} != {id(stmt) for stmt in self.lift.body}:
            return None
        by_name = {param: edge for param, edge, _ in self.bindings}
        a_edge = self.operands[0]
        a_names = {param for param, edge, _ in self.bindings if edge is a_edge}
        streamed = []
        for stmt in products:
            other = set(stmt.args) - a_names
            if len(set(stmt.args) & a_names) != 1 or len(other) != 1:
                return None  # a product that does not multiply A by exactly one other edge
            edge = by_name.get(next(iter(other)))
            if edge is None or edge is a_edge:
                return None
            streamed.append(edge)

        b_edge = streamed[0]
        a_space, b_space = a_edge.free_axes, b_edge.free_axes
        if self.axis not in a_space & b_space:
            return None
        left_only, right_only = a_space - b_space, b_space - a_space
        if len(left_only) > 1 and len(right_only) > 1:
            return None  # several own axes on BOTH sides: an outer product over batches, not an orientable pair
        if not left_only and not right_only:
            return None  # a dot product over shared axes only carries no output role to tile: a planar reduce
        slab = b_edge.as_slab()
        b_trans = slab is not None and self.axis in slab.load.index[-1].free_vars()
        return ContractionView(
            axis=self.axis,
            left_axes=frozenset(left_only),
            right_axes=frozenset(right_only),
            shared_axes=frozenset((a_space & b_space) - {self.axis}),
            product=product,
            plus=plus,
            b_trans=b_trans,
        )

    @cached_method
    def as_reduction(self) -> ReductionView | None:
        """The :class:`ReductionView` of this term — its combine read as a monoid fold — or
        ``None`` for a term without one (a slab, a projection). Memoized on the term."""
        if self.combine is None:
            return None
        states = self.combine.results
        return ReductionView(
            axis=self.axis,
            states=states,
            other=self.combine.params[len(states) :],
            terms=self.injected.results,
            ops=self.combine.components(),
        )

    @cached_method
    def as_slab(self) -> SlabView | None:
        """The :class:`SlabView` of this term — its one gmem read and the coordinates it declares —
        or ``None`` for a computed cone. Memoized on the term."""
        if self.operands or self.base is not None or len(self.lift.body) != 1 or not isinstance(self.lift.body[0], Load):
            return None
        return SlabView(load=self.lift.body[0])

    # ---- the DERIVED READINGS. ``Map`` and ``Contraction`` are no longer stored kinds (the
    # collapse); every field they carried reads back off the one stored term here, so their old
    # accessors keep their exact meanings and their consumers keep their exact spellings. ------- #
    @classmethod
    def slab(cls, load: Load) -> Fold:
        """One gmem read as a term — a lift of one ``Load`` over the coordinates the load indexes,
        which :meth:`Lambda.closing` binds as its params. No operands, no ``combine``: a slab
        iterates, it does not reduce."""
        return cls(lift=Lambda.closing((), Body((load,)), load.names))

    @cached_method
    def merge(self, other: tuple[str, ...]) -> Body:
        """The combine APPLIED at ``(state, other)`` — a second, fully reduced state named
        ``other`` (a register copy ``acc__r1``, a tree neighbour's partial, a workspace slice
        ``acc__p``) — as loop-IR statements: the combine's temps, then one in-place ``Accum`` per
        state component, the form whose seed is the ⊕'s identity (what the identity placement
        emits). A temp a state's new value is built from (a twisted combine's ψ-rescale) becomes
        that ``Accum``'s ``base``. The temps are renamed onto ``other``'s spelling, so two merges
        into one state never collide.

        The ONE derivation of the stored ⊕ as statements; :meth:`step` is its instance at the
        injected singleton. Memoized on the term per ``other``.
        """
        view = self.as_reduction()
        prefix = f"{view.other[0]}__"
        named = dict(zip(view.other, other, strict=True))
        for stmt in self.combine.body:
            if stmt.name not in view.states:
                named[stmt.name] = f"{other[0]}__{stmt.name.removeprefix(prefix)}"
        applied = list(self.combine.rename(named).body)
        definitions = {stmt.name: stmt for stmt in applied if stmt.name not in view.states}  # temps; a state's rewrite is not a def

        def reads(name: str, state: str) -> bool:
            stmt = definitions.get(name)
            return name == state or (stmt is not None and any(reads(arg, state) for arg in stmt.args))

        out: list[Stmt] = []
        for stmt in applied:
            if stmt.name not in view.states:
                out.append(stmt)
                continue
            state, args = stmt.name, stmt.args
            if stmt.op.name == "copy" and (pivot := definitions.get(args[0])) is not None and state in pivot.args:
                # The pivot's final write copies its own ``maximum`` temp: accumulate that maximum.
                stmt, args = pivot, pivot.args
            if state in args:
                value = next(arg for arg in args if arg != state) if args != (state, state) else state
                out.append(Accum(name=state, value=value, op=stmt.op))
            else:
                base, value = args if reads(args[0], state) else (args[1], args[0])
                out.append(Accum(name=state, value=value, op=stmt.op, base=base))
        return Body(tuple(out))

    @cached_method
    def step(self) -> Body:
        """The per-step statements this fold DERIVES from its stored parameters: the lift body
        SEEN THROUGH ψ (:attr:`injected` — the lift itself for a planar fold), then the combine
        applied at that injected singleton (:meth:`merge` at its results, each ``Accum`` folding
        over the reduce axis), then an observer's pure tap, so a streamed store reads the
        post-combine (inclusive-prefix) state.

        Without a combine the term is a map and the step is the lift body. Deterministic from the
        stored parameters, so kernel identity depends on no classified view. Memoized on the term.
        """
        lift = self.injected
        if self.combine is None:
            return lift.body
        merged = [replace(stmt, axes=(self.axis,)) if isinstance(stmt, Accum) else stmt for stmt in self.merge(lift.results)]
        observed = self.observe.body if self.observe is not None else ()
        return Body((*lift.body, *merged, *observed))

    def fuse(self, recipe: Recipe, axes) -> Fold | None:
        """This reduce fused onto the reduce it reads, by ``recipe`` — one fold carrying both
        states under the recipe's twisted ⊕ — or ``None`` when no channel of the recipe clicks.
        The ONE generic algorithm over the declarative recipes (:mod:`~emmy.compiler.ir.pure.twist`).
        The pivot is found among this term's operands: a reduce folding the recipe's pivot ⊕, or a
        fold this recipe already fused (whose channel 0 is the pivot). The pivot's state is the
        lift param bound to it, positionally; the score is the sub-cone of the lift alpha-equal to
        the pivot's own per-element map, operand for operand (a projection's component by its own
        cone); and what remains of the lift with that cone cut out, its params in role order, must
        equal a channel's pattern by canonical form. A click gives the role-to-name map, and the
        fused fold is the recipe instantiated by renaming: the pivot's operands plus what the
        extras bind, the pivot's lift with the channel's injection appended, the states
        concatenated, the recipe's ⊕ program over them. The same recipe fuses online softmax and
        flash attention alike; the caller rewrites the tree's operands onto the result.
        """
        view = self.as_reduction()
        if view is None or self.observe is not None or view.ops is None or len(view.states) != 1:
            return None
        for pivot in self.operands:
            pview = pivot.as_reduction()
            if pview is None or pivot.observe is not None:
                continue
            if pview.ops is not None:
                if len(pview.states) != 1 or pview.ops[0].reduce_canon != recipe.pivot:
                    continue
            elif pivot.twist is None or pivot.twist.recipe is not recipe:
                continue  # a twisted pivot must already carry THIS recipe: another one's carrier is not its pivot
            if axes[self.axis].extent != axes[pivot.axis].extent or axes[self.axis].window != axes[pivot.axis].window:
                continue
            fused = self._twist(pivot, recipe)
            if fused is not None:
                return fused
        return None

    def _twist(self, pivot: Fold, recipe) -> Fold | None:
        view, pview = self.as_reduction(), pivot.as_reduction()

        def cone(fold: Fold, name: str) -> tuple:
            # The closed cone defining ``name`` in the lift, and the VALUE each of its params binds
            # (``None`` for a coordinate: a reduce's state binds that term, a projection's result
            # binds its own cone).
            by_param = {param: edge for param, edge, _ in fold.bindings}
            fn = fold.lift.cone(name)
            values = tuple(
                None if (edge := by_param.get(param)) is None else edge if edge.axis is not None else cone(edge, param)
                for param in fn.params
            )
            return fn, values

        def same(a: tuple, b: tuple) -> bool:
            if a[0].canonical() != b[0].canonical() or len(a[1]) != len(b[1]):
                return False
            for x, y in zip(a[1], b[1], strict=True):
                if (x is None) != (y is None) or isinstance(x, tuple) != isinstance(y, tuple):
                    return False
                if x is not None and (not same(x, y) if isinstance(x, tuple) else x.canonical() != y.canonical()):
                    return False
            return True

        bound = tuple((param, edge) for param, edge, _ in self.bindings)
        by_param = dict(bound)
        g = next(param for param, edge in bound if edge is pivot)
        pivot_params = {param for param, edge in bound if edge is pivot}
        score = cone(pivot, pivot.lift.results[0])
        candidates = [p for p in self.lift.params[1:] if p not in pivot_params] + [n for s in self.lift.body for n in s.defines()]
        for x in candidates:
            found = cone(self, x)
            # The cone may read coordinates free (the causal mask's ``key <= query``): this fold's
            # own axis is the pivot's once fused, so it is COMPARED under the pivot's name, while
            # the residual below is cut by the cone's own statements.
            spelled = found if self.axis == pivot.axis else (found[0].rename({self.axis: pivot.axis}), found[1])
            if not same(spelled, score):
                continue
            # What remains of the lift with the score cut out, closed over what it still reads —
            # ``(score, pivot, *extras)`` in role order, the extras operand-bound values.
            residual = tuple(stmt for stmt in self.lift.body if stmt not in found[0].body)
            fn = Lambda.closing((x, g), Body(residual), self.lift.results)
            extras = fn.params[2:]
            if any(p not in by_param or p in pivot_params for p in extras):
                continue
            # A channel whose base ⊕ is this fold's and whose pattern is what remains.
            plus = view.ops[0].reduce_canon
            channel = next(
                (
                    c
                    for index, c in enumerate(recipe.channels)
                    if c.pattern is not None and recipe.base[1 + index] == plus and fn.canonical() == c.pattern.canonical()
                ),
                None,
            )
            if channel is None:
                continue
            # Instantiate: every operand an extra binds joins, its axis spelled as the pivot's.
            old, new = self.axis, pivot.axis
            extra_edges = tuple(dict.fromkeys(edge for p, edge in bound if p in extras))
            extra_params = tuple(p for edge in extra_edges for p, e in bound if e is edge)
            if old != new:
                extra_edges = tuple(
                    _rewrite_kind(
                        edge, lambda n: n, Sigma({old: Var(new)}), lambda a, old=old, new=new: replace(a, name=new) if a.name == old else a
                    )
                    for edge in extra_edges
                )
            # The carrier's states: the pivot's, then every channel in recipe order — the matched
            # one is this fold's own state, one without a pattern a state the recipe adds. Each
            # takes the recipe's BASE contribution for it (``lift``'s own result), instantiated at
            # the score and this channel's extras: the term stores what the recipe declares, and ψ
            # is applied at lowering (:attr:`injected`). The SEEDS stay the carrier's — the state
            # is the stable one, only the per-element contribution is the base's.
            state = view.states[0]
            score = pivot.lift.results[0]
            added = [
                (state if c is channel else f"{state}__{c.name}", index, self.init[0] if c is channel else c.init)
                for index, c in enumerate(recipe.channels)
                if c is channel or c.pattern is None
            ]
            roles = {recipe.lift.params[0]: score, **dict(zip(channel.pattern.params[2:], extras, strict=True))}
            arity = sum(len(edge.exposes) for edge in pivot.operands)
            held = [
                *zip(pivot.operands, _slots(pivot.lift.params[1 : 1 + arity], pivot.operands), strict=True),
                *zip(extra_edges, _slots(extra_params, extra_edges), strict=True),
            ]
            body, results, inits = list(pivot.lift.body), list(pivot.lift.results), list(pivot.init)
            for name, index, init in added:
                taken = {n for stmt in body for n in stmt.defines()} | {p for _, slot in held for p in slot}
                cone = recipe.lift.cone(recipe.lift.results[1 + index])
                names = {param: roles[param] for param in cone.params}
                names.update((stmt.name, stmt.name if stmt.name not in taken else f"{name}__{stmt.name}") for stmt in cone.body)
                instance = cone.rename(names)
                body.extend(instance.body)
                results.append(instance.results[0])
                inits.append(init)
            results = _factor_weights(body, results, held, len(pivot.lift.results))
            lift = Lambda(
                params=(pivot.axis, *(p for _, slot in held for p in slot), *pivot.lift.params[1 + arity :]),
                body=Body(tuple(Body(tuple(body)).backward_cone(tuple(dict.fromkeys(results))).members)),
                results=tuple(results),
            )
            states = (*pview.states, *(name for name, _, _ in added))
            ops = (*pivot.base.components(), *(recipe.base[1 + index] for _, index, _ in added))
            carried = () if pivot.twist is None else pivot.twist.channels
            bindings = dict(() if pivot.twist is None else pivot.twist.roles)
            twist = Twist(
                recipe=recipe,
                roles=tuple({**bindings, **{role: name for role, name in roles.items()}}.items()),
                channels=(*carried, *(index for _, index, _ in added)),
            )
            return Fold(
                operands=tuple(edge for edge, _ in held),
                lift=lift,
                init=tuple(inits),
                base=Lambda.componentwise(ops, states),
                twist=twist,
            )
        return None

    @cached_method
    def canonical(self) -> Fold:
        """The α-canonical form of this TERM — a ``Fold``, the same kind that went in.

        The whole term, not its lift: a Fold's value also depends on its axis extent, its monoid
        and its operand edges, none of which live in ``lift``. Every bound name renames
        positionally — the axis it binds (``_a0``), the lift's internal defs (``_v``), what the term exposes
        (``_r``: a projection's results, a reduce's carried state and observed values) and what it
        binds from each operand (``_e``), each operand canonicalized first and its own ``_r``
        interface re-spelled onto the binding — so two terms equal up to the choice of every
        bound name have EQUAL canonical forms, whatever their accumulators were called. FREE
        names pass through, so equal canonical forms mean equal value under the SAME environment.
        """
        lead = () if self.axis is None else (self.axis,)
        mapping = {name: f"_a{index}" for index, name in enumerate((*lead, *self.lift.params[len(lead) + len(self.bindings) :]))}
        own = self.lift.results if self.base is None else self.exposes
        mapping.update((name, f"_r{index}") for index, name in enumerate(own))
        for index, (param, _, _) in enumerate(self.bindings):
            mapping.setdefault(param, f"_e{index}")
        counter = 0
        for stmt in self.lift.body.iter():
            for name in stmt.defines():
                if name not in mapping:
                    mapping[name] = f"_v{counter}"
                    counter += 1
        base = None
        if self.base is not None:
            # The ⊕'s own names — its second operand — are nobody else's: they renumber after the
            # term's, so how a fold spelled its accumulators never reaches the form. The twisted
            # conjugate needs no arm here: it is DERIVED from these states, so it follows them.
            own = dict(mapping)
            for name in self.base.params:
                own.setdefault(name, f"_c{len(own)}")
            base = self.base.rename(own)
        return replace(
            self,
            operands=tuple(edge.canonical() for edge in self.operands),
            lift=self.lift.rename(mapping),
            base=base,
            # The role binding is names of THIS term, so it renames in lockstep like every other.
            twist=None if self.twist is None else replace(self.twist, roles=_renamed_roles(self.twist.roles, lambda n: mapping.get(n, n))),
            # The observer binds the iteration var and reads the carried state, so it renames in
            # LOCKSTEP: renaming the axis without it leaves the observer reading a name that no
            # longer exists, and a scan would then canonicalize to something that is not a term.
            observe=None if self.observe is None else self.observe.rename(mapping),
        )

    @cached_method
    def lower(self, bound: frozenset[str] | None = None, stores: tuple[OutputSpec, ...] = (), axes: tuple[Axis, ...] = ()) -> Body:
        """Flatten this term to the Loop IR nest the materializer expands, the kernel's boundary
        ``stores`` placed in it.

        ``bound`` names the coordinates the CALLER binds — the kernel grid, an enclosing loop's
        scope; ``None`` binds every free coordinate (the open body a term spells inside an
        enclosing scope) and ``frozenset()`` the closed program; ``axes`` is the kernel's axis table
        — the extent and window of every axis the tree binds or reads, which a term carries none
        of: a reduce loop takes its axis from it, and the closed program the coordinates it opens
        loops for. One rule then places every term
        of the tree, read straight off the representation: a term is materialized at the
        SHALLOWEST scope on its path that binds all of its :attr:`free_axes`. The scopes are the
        plain loops this term opens for the free coordinates left unbound — outermost the
        coordinate the most terms are evaluated over, so what siblings share is hoisted above
        them, ties in declaration order — and the reduce loop of each term on the way down. That
        is the whole of the hoist: a declaration compared against the scopes above, not a body
        walked for free names. An operand that does not index its reader's reduce axis lands
        ahead of that loop; one that reads no coordinate a deeper loop binds lands ahead of every
        such loop, past the term that reads it.

        A boundary store follows the term that DEFINES the value it writes, at that term's scope:
        a projection's result after its step, a reduce's carried state after its loop, an observed
        per-step value inside the loop after the observer. A store alone evaluated over a
        coordinate no term declares (a broadcast, ``o[j] = acc``) opens that coordinate under its
        term, the spec's ``sweep`` naming the axis. Nothing is walked for the place a store goes:
        the term defining its value was just placed.

        The free loops form a TREE, not a chain. A term's operands sit on ITS path — the shallowest
        prefix of its own loops that binds them — so what a term reads is always in scope; only a
        wrapper with no step of its own leaves its operands free to take their own paths, which is
        how two output sweeps no term shares become sibling loops. Operands are placed before
        the statements that read them, a reduce term's step follows its operands inside the loop
        that binds its axis, and a SHARED term — one object reached through several operand
        positions — defines its names once per scope.

        The ONE lowering spelling — every consumer of a term's statements calls this. Memoized on
        the term per binding.
        """
        if bound is None:
            bound = self.free_axes
        # The stores are spelled in this term's own vocabulary; they render with the lift.
        spelled = dict(zip(self.lift.params, self.applied.params, strict=True))
        stores = tuple(replace(spec, write=spec.write.rewrite(lambda name: spelled.get(name, name))) for spec in stores)
        coordinates: dict[str, Axis] = {axis.name: axis for axis in axes}
        declared: list[str] = []  # the order the tree first reads its coordinates — the tie order below
        internal: set[str] = set()  # the axes terms of the tree bind: a store under a reduce loop may index one
        readers: dict[str, int] = {}
        origin: dict[str, tuple[int, str]] = {}  # a value's defining term and what it is there
        seen: set[int] = set()
        pending = [self]
        while pending:
            term = pending.pop()
            if id(term) in seen:
                continue
            seen.add(id(term))
            declared.extend(name for name in term.lift.params[(term.axis is not None) + len(term.bindings) :] if name not in declared)
            if term.axis is not None:
                internal.add(term.axis)
            for name in term.free_axes:
                readers[name] = readers.get(name, 0) + 1
            if term.base is None:
                origin.update((name, (id(term), "step")) for stmt in term.lift.body for name in stmt.defines())
            else:
                origin.update((name, (id(term), "state")) for name in term.base.results)
                if term.observe is not None:
                    origin.update((name, (id(term), "observed")) for name in term.observe.results)
            pending.extend(reversed(term.operands))
        owned: dict[tuple[int, str], list[OutputSpec]] = {}

        def placed(term: Fold) -> tuple[Fold, ...]:
            # An operand is placed when the term READS it, or when the tree below it defines a
            # value the kernel STORES — a nested output sweep is materialized for its own store,
            # not for its reader. Everything else is a site the schedule may take and the nest has
            # no use for (:attr:`read_operands`).
            kept = {id(edge) for edge in term.read_operands}
            return tuple(edge for edge in term.operands if id(edge) in kept or _writes_under(edge, writing))

        for spec in stores:
            key = origin.get(spec.write.values[0])
            assert key is not None and all(origin.get(name) == key for name in spec.write.values), (
                f"a store over {spec.write.values} writes values no one term defines"
            )
            owned.setdefault(key, []).append(spec)
        for spec in stores:
            for axis in spec.sweep:
                coordinates.setdefault(axis.name, axis)
        declared.extend(name for name in coordinates if name not in declared)
        read = self.free_axes | {name for spec in stores for expr in spec.write.index for name in expr.free_vars()}
        if missing := read - bound - internal - set(coordinates):
            raise ValueError(f"lower: no extent for coordinates {sorted(missing)} — the closed program takes them as axes")
        # A name a nested reduce binds is still a free coordinate here when the tree reads it free
        # (a stat over the row beside a slab of the same row): the reduce loop shadows it inside.
        opened = sorted(
            (name for name in coordinates if name in read and name not in bound and (name not in internal or name in self.free_axes)),
            key=lambda name: (-readers.get(name, 0), declared.index(name)),
        )
        nest: dict[tuple[str, ...], list[Stmt]] = {(): []}

        def path_of(free: frozenset[str], path: tuple[str, ...] | None) -> tuple[str, ...]:
            # The free loops a term sits under: the shallowest prefix of its reader's path binding
            # its coordinates, or — unconstrained — those coordinates in the opened order.
            needed = {name for name in opened if name in free}
            if path is None:
                return tuple(name for name in opened if name in needed)
            return next(path[:depth] for depth in range(len(path) + 1) if needed <= set(path[:depth]))

        def sink(path: tuple[str, ...]) -> list[Stmt]:
            for depth in range(len(path) + 1):
                nest.setdefault(path[:depth], [])
            return nest[path]

        def attach(term: Fold, kind: str, target: list[Stmt], node: tuple[str, ...], scope: frozenset[str]) -> None:
            # The stores over what ``term`` defines as ``kind`` — its step's results, its carried
            # state, or its observer's per-step values — land right after it, in ``target``.
            for spec in owned.get((id(term), kind), ()):
                extra = {name for expr in spec.write.index for name in expr.free_vars()} - scope
                if not extra:
                    target.append(spec.write)
                    continue
                assert extra <= set(opened), f"a store reads coordinates {sorted(extra - set(opened))} no term declares and no sweep names"
                sink((*node, *(name for name in opened if name in extra))).append(spec.write)

        def place(term: Fold, loops: list[tuple[str, frozenset[str], list[Stmt]]], path: tuple[str, ...] | None) -> None:
            # ``loops``: the reduce loops enclosing this position, outermost first, as (axis, scope, stmts).
            if any(axis in term.free_axes for axis, _, _ in loops):
                depth = next(depth for depth, (_, scope, _) in enumerate(loops) if term.free_axes <= scope)
                _, scope, stmts = loops[depth]
                loops, node = loops[: depth + 1], path
            else:
                node = path_of(term.free_axes, path)
                scope, stmts, loops = frozenset(bound) | set(node), None, []
            if term.axis is None:
                step = term.step()
                for edge in placed(term):
                    place(edge, loops, node if step else path)
                if step:
                    target = stmts if stmts is not None else sink(node)
                    target.extend(step)
                    attach(term, "step", target, node, scope)
                return
            inner: list[Stmt] = []
            for edge in placed(term):
                place(edge, [*loops, (term.axis, scope | {term.axis}, inner)], node)
            inner.extend(term.step())
            attach(term, "observed", inner, node, scope | {term.axis})
            target = stmts if stmts is not None else sink(node)
            if term.axis not in coordinates:
                raise ValueError(f"lower: no extent for reduce axis {term.axis!r} — the kernel's axis table names it")
            target.append(Loop(axis=coordinates[term.axis], body=_scope(inner)))
            attach(term, "state", target, node, scope)

        def assemble(path: tuple[str, ...]) -> Body:
            body = list(nest[path])
            for name in opened[opened.index(path[-1]) + 1 :] if path else opened:
                if (*path, name) in nest:
                    body.append(Loop(axis=coordinates[name], body=assemble((*path, name))))
            return _scope(body)

        writing = {term for term, _ in owned}
        place(self, [], None)
        return assemble(())


def _scope(stmts) -> Body:
    """One scope's statements — a term reached through several operand positions defining its
    names once, and sibling terms folding ONE coordinate iterating together.

    The dedup is the shared-term rule. The FUSE is the same rule a level up: two loops over one
    axis where neither reads what the other defines are two passes over one stream, and their
    union is one pass, computing what they both read once. A loop that DOES read the loop above it (a
    statistic's pass, then the pass that normalizes by it) reads a FINISHED accumulator, and
    iterating together would hand it the in-flight one; that pair stays two loops.
    """
    out: list[Stmt] = []
    for stmt in dict.fromkeys(stmts):
        prior = out[-1] if out else None
        if (
            isinstance(stmt, (Loop, StridedLoop))
            and isinstance(prior, (Loop, StridedLoop))
            and prior.is_reduce
            and stmt.is_reduce
            and replace(prior, body=stmt.body) == stmt
            and not (free_names(stmt) & prior.body.ssa_defs)
        ):
            stmt = replace(prior, body=_scope((*prior.body, *stmt.body)))
            out.pop()
        out.append(stmt)
    return Body(tuple(out))


@_rewrite_kind.register
def _(s: Fold, rename, sigma, axis_fn):
    # ONE handler for the one stored kind (the collapse retired the Map / Contraction arms —
    # singledispatch keys on the stored type, and there is now only one). Every operand edge
    # dispatches back through the registry; the fold renames its lift / monoid in lockstep
    # (params track the operand names positionally, the combine's results ARE the accumulator
    # names). At zero axes there is no iteration var to rename and no monoid to thread.
    # σ is applied hygienically to the term's own binder: the reduce axis this term binds is a
    # different variable from any σ key spelled alike, so its mapping is dropped for the subtree.
    # Operand edges are terms, so they rewrite through this same registry, never the statement entry.
    if s.axis is not None and sigma is not None and s.axis in sigma.mapping:
        sigma = Sigma({name: value for name, value in sigma.mapping.items() if name != s.axis})
    operands = tuple(_rewrite_kind(edge, rename, sigma, axis_fn) for edge in s.operands)
    lead = (rename(s.axis),) if s.axis is not None else ()  # the iteration var — a slab has none

    def _param(name: str) -> str:
        # The environment tail may hold AXIS names, which rename through sigma (the body's own
        # coordinate substitution), not through the SSA renamer. Taking sigma's answer when it has
        # one keeps a param and every use of it in the body spelled the same way.
        mapped = sigma.get(name) if sigma is not None else None
        return mapped.name if isinstance(mapped, Var) else rename(name)

    lift = Lambda(
        params=(*lead, *(_param(p) for p in s.lift.params[len(lead) :])),
        body=Body(tuple(_rewrite(st, rename, sigma, axis_fn) for st in s.lift.body)),
        results=tuple(rename(r) for r in s.lift.results),
    )
    base = s.base.rename(rename) if s.base is not None else None
    twist = None if s.twist is None else replace(s.twist, roles=_renamed_roles(s.twist.roles, rename))
    observe = None
    if s.observe is not None:
        # The observer renames in lockstep: param 0 tracks the axis, the state params track the
        # combine's renamed results, the body/results are ordinary SSA material.
        observe = Lambda(
            params=(*lead, *(rename(p) for p in s.observe.params[1:])),
            body=Body(tuple(_rewrite(st, rename, sigma, axis_fn) for st in s.observe.body)),
            results=tuple(rename(r) for r in s.observe.results),
        )
    return replace(s, operands=operands, lift=lift, base=base, twist=twist, observe=observe)


def _writes_under(term: Fold, writing: set[int]) -> bool:
    """Whether any term of ``term``'s subtree defines a value the kernel's boundary stores."""
    pending, seen = [term], set()
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if id(node) in writing:
            return True
        pending.extend(node.operands)
    return False


def _slots(params: tuple[str, ...], edges) -> list[tuple[str, ...]]:
    """``params`` cut into one group per edge — the positional binding, one param per result component."""
    out, cursor = [], 0
    for edge in edges:
        out.append(tuple(params[cursor : cursor + len(edge.exposes)]))
        cursor += len(edge.exposes)
    return out


def _factor_weights(body: list[Stmt], results: list[str], held: list[tuple[Fold, tuple[str, ...]]], start: int) -> list[str]:
    """Make every factor of a channel's PRODUCT an operand, so the channel reads as a contraction.

    A twisted carrier's expectation channel contributes ``weight ⊗ value`` — one monomial, but the
    weight is a cone over the score, so as it stands the product multiplies a body definition by an
    operand and no bilinear reading applies. Hoisting that cone out as its own zero-axis operand is
    what the semiring formation already does for a matmul's decoded A, and it leaves the channel a
    bare product of two operand edges. A channel whose whole contribution IS that cone (the
    denominator, ``Sum exp(score)``) then finds it already held and binds it rather than computing a
    second copy, which is why one score node serves the whole carrier.

    Only the results from ``start`` on are the channels this fusion added; the ones before them are
    the pivot's own, already factored when it was built. ``body`` and ``held`` are extended in
    place; the new result names are returned.
    """
    bound = {param for _, slot in held for param in slot}

    def operand(name: str, *, create: bool) -> str:
        # ``name``'s closed cone as an operand — the param of an equal edge already held, or, with
        # ``create``, a fresh zero-axis term appended under the cone's own result name. Either way
        # the statement defining it leaves the lift: it is the operand's program now, not the step's.
        members = tuple(Body(tuple(body)).backward_cone((name,)).members)
        reads = {n for stmt in members for n in stmt.deps()} - {n for stmt in members for n in stmt.defines()}
        inner = [(edge, slot) for edge, slot in held if not reads.isdisjoint(slot)]
        cone = Fold(
            operands=tuple(edge for edge, _ in inner),
            lift=Lambda.closing(tuple(param for _, slot in inner for param in slot), Body(members), (name,)),
        )
        for edge, slot in held:
            if edge.canonical() == cone.canonical():
                body[:] = [stmt for stmt in body if name not in stmt.defines()]
                return slot[0]
        if not create:
            return name
        body[:] = [stmt for stmt in body if name not in stmt.defines()]
        held.append((cone, (name,)))
        return name

    definitions = {stmt.name: stmt for stmt in body}
    out = list(results)
    for position, result in enumerate(out[start:], start):
        product = None if result in bound else definitions.get(result)
        if product is None:
            continue
        if isinstance(product, Assign) and len(product.args) == 2:
            # A factor that is itself a carried contribution stays in the lift: Welford's ``x·x``
            # multiplies the pivot's own value by itself, which is one edge and no contraction.
            free = [arg for arg in product.args if arg in definitions and arg not in out]
            if not free:
                continue
            renamed = {arg: operand(arg, create=True) for arg in free}
            body[body.index(product)] = replace(product, args=tuple(renamed.get(arg, arg) for arg in product.args))
            continue
        # Not a product: the whole contribution may still BE a cone another channel already holds.
        out[position] = operand(result, create=False)
    return out


def _renamed_roles(roles: tuple[tuple[str, str], ...], rename) -> tuple[tuple[str, str], ...]:
    """A twist's role binding under one renamer — the recipe's role name is the SCHEMA's and never
    renames; the term's own name for it does, like every other name the term binds."""
    return tuple((role, rename(name)) for role, name in roles)


__all__ = [
    "ReductionView",
    "Fold",
]
