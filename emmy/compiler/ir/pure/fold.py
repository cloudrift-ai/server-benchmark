"""``Fold`` — the ONE reduce term: ``reduce(⊕) ∘ map(f)`` in the λ-foldMap spelling.

The whole stored vocabulary of Tile IR, and a PURE term throughout: an optional iteration
``axis``, a pure ``lift`` :class:`~emmy.compiler.ir.pure.lam.Lambda`, the monoid's flat
``(init, combine)`` pair, and a tuple of ``operands`` — the closed inputs, each an edge bound
positionally to a lift param. Every reading (the map at zero axes, the bilinear
:class:`ContractionView`, the :class:`SlabView` leaf, the serial step) is DERIVED from those
params; nothing else is stored.

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
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, OutputSpec, Stmt

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
    left: str | None
    """The free axis A carries — the output role it strides. ``None`` where A brings none."""
    right: str | None
    """The free axis the streamed operand carries. ``None`` for a MATVEC, whose B is a vector over
    the reduction alone: a contraction does not stop being one for want of a second output axis;
    whether the pair can be ORIENTED as (m, n) is the placement's question, not recognition's. A
    pair with NO output role on either side (a row's dot product with itself, the RMS statistic
    spelled as two casts of one load) is not a contraction at all: it reads as a planar reduce."""
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
    ``Loop``. It splits the reduce's **algebra** (the loop-carried flat ⊕ — componentwise for a
    plain ``sum`` / ``max`` / ``mean``, a rescaling program for online-softmax / flash — a plain
    :class:`Lambda` either way) from its **structure** (the axis the lift binds and the per-element
    ``step`` it folds). Every reading is **derived** from the stored params (:meth:`as_contraction`,
    :meth:`as_slab`, :meth:`as_reduction`, :meth:`step`), never stored, and the loop nest is
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
    # ---- the λ-foldMap spelling — the fold's storage: a PURE ``lift`` ``λ(k, v₁…vₙ) → S``
    # (params: the iteration var first, then one per operand result component, bound POSITIONALLY —
    # the names are this term's own, :attr:`bindings` pairs them with the edges, and :attr:`applied`
    # spells them as the operands' results for every renderer) plus the TRUE monoid's flat
    # ``(init, combine)`` pair whose combine carries the REAL accumulator names (its results). The
    # serial step and the ``Accum`` forms are DERIVED (:attr:`step` / ``__post_init__``). --------- #
    lift: Lambda = field(kw_only=True)  # CLOSED by ``Lambda.__post_init__``; formed by :meth:`Lambda.closing`
    init: tuple[float, ...] = ()  # the ⊕ seeds — op identities for a plain fold; (−inf, 0, …) LSE
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
        # EVERY OPERAND IS A TERM. A gmem read is a term over one ``Load``, so the tree is homogeneous and a
        # per-edge question is an attribute rather than a helper dispatching on what an edge
        # happens to be. Stated here because a statement in ``operands`` does not announce itself:
        # an ``isinstance(edge, Load)`` against a type that can no longer appear reads as False,
        # not as an error, so the check it guards disappears silently. Formation is the one place
        # that can say no — this caught the cut's workspace reads, which were bare ``Load``s.
        stray = [type(edge).__name__ for edge in self.operands if not isinstance(edge, Fold)]
        if stray:
            raise TypeError(f"Fold operands must be terms, got {stray}; a gmem read is a term over one Load")
        if self.combine is None:
            # The ZERO-AXIS node: no iteration and no monoid, so the only formation fact is the
            # positional binding — one lift param per operand RESULT COMPONENT, no leading
            # iteration var. (The projection (zero-axis) fold was exactly this, with ``fn`` for ``lift``.)
            assert self.combine is None and not self.init, "a zero-axis Fold carries no monoid"
            assert self.observe is None, "a zero-axis Fold carries no per-step state to observe"
            arity = sum(len(edge.exposes) for edge in self.operands)
            assert len(self.lift.params) >= arity, f"lift binds {len(self.lift.params)} params for {arity} operand result components"
            return
        # Formation validates the positional binding and the S × S → S arity; the planar-vs-twisted
        # reading is DERIVED (:meth:`as_reduction`), never a second stored spelling.
        n = len(self.init)
        if len(self.combine.params) != 2 * n or len(self.combine.results) != n:
            raise ValueError(f"Fold combine must be S × S → S at arity {n}: params={self.combine.params} results={self.combine.results}")
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
        if self.combine is None:
            return self.applied.results
        state = self.combine.results
        return state if self.observe is None else (*state, *self.observe.results)

    @cached_property
    def bindings(self) -> tuple[tuple[str, Fold, int], ...]:
        """The positional binding of the lift to the operands — for every operand-bound param in
        order, ``(param, edge, component)``: the lift's params past the iteration var bind the
        operands' result components in order. The names are this term's own; how an operand spells
        its results is nobody else's business until the term is rendered (:attr:`applied`).
        """
        params = self.lift.params[1 if self.combine is not None else 0 :]
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

    def binds_axes(self) -> frozenset[str]:
        """The axis this term binds — what the statement-door ``rewrite`` drops from σ for the subtree."""
        return frozenset() if self.axis is None else frozenset({self.axis})

    @property
    def axis(self) -> str | None:
        """The name of the axis this term BINDS — its lift's iteration var, the first param — or
        ``None`` for a zero-axis term. A NAME: the extent and window are the evaluator's, held in
        the kernel's axis table (``TileOp.axes``) and handed to :meth:`lower`; a sum over 128 and
        one over 256 are the same term over different domains, like a slab under any M."""
        return self.lift.params[0] if self.combine is not None else None

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
        if len(left_only) > 1 or len(right_only) > 1:
            return None  # more than one free axis a side is not an orientable output role
        if not left_only and not right_only:
            return None  # a dot product over shared axes only carries no output role to tile: a planar reduce
        slab = b_edge.as_slab()
        b_trans = slab is not None and self.axis in slab.load.index[-1].free_vars()
        return ContractionView(
            axis=self.axis,
            left=next(iter(left_only), None),
            right=next(iter(right_only), None),
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
            terms=self.applied.results,
            ops=self.combine.components(),
        )

    @cached_method
    def as_slab(self) -> SlabView | None:
        """The :class:`SlabView` of this term — its one gmem read and the coordinates it declares —
        or ``None`` for a computed cone. Memoized on the term."""
        if self.operands or self.combine is not None or len(self.lift.body) != 1 or not isinstance(self.lift.body[0], Load):
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
        """The per-step statements this fold DERIVES from its stored parameters: the lift body,
        then the combine applied at the injected singleton (:meth:`merge` at the lift's results,
        each ``Accum`` folding over the reduce axis), then an observer's pure tap, so a streamed
        store reads the post-combine (inclusive-prefix) state.

        Without a combine the term is a map and the step is the lift body. Deterministic from the
        stored parameters, so kernel identity depends on no classified view. Memoized on the term.
        """
        lift = self.applied
        if self.combine is None:
            return lift.body
        merged = [replace(stmt, axes=(self.axis,)) if isinstance(stmt, Accum) else stmt for stmt in self.merge(lift.results)]
        observed = self.observe.body if self.observe is not None else ()
        return Body((*lift.body, *merged, *observed))

    def twist(self, recipe, axes) -> Fold | None:
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
            elif recipe.combine is not None and len(pview.states) != len(recipe.combine.results):
                continue  # a fixed-arity recipe's ⊕ is over exactly its carrier; another recipe's fold is not its pivot
            elif not pivot.combine.alpha_eq(recipe.program(pview.states)):
                continue
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
            if not same(found, score):
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
            operands = (*pivot.operands, *extra_edges)
            # The carrier's states: the pivot's, then every channel in recipe order — the matched one
            # is this fold's own state, one without a pattern a state the recipe adds; each injection
            # instantiated at the score, its temps namespaced on the state it feeds.
            state = view.states[0]
            score = pivot.lift.results[0]
            injections = [
                (state, channel.injection, self.init[0]) if c is channel else (f"{state}__{c.name}", c.injection, c.init)
                for c in recipe.channels
                if c is channel or c.pattern is None
            ]
            roles = dict(zip(channel.pattern.params[2:], extras, strict=True))  # the channel's extras, by role
            body, results, inits = list(pivot.lift.body), list(pivot.lift.results), list(pivot.init)
            for name, injection, init in injections:
                names = {injection.params[0]: score, **{param: roles[param] for param in injection.params[1:]}}
                names.update((stmt.name, f"{name}__{stmt.name}") for stmt in injection.body)
                instance = injection.rename(names)
                body.extend(instance.body)
                results.extend(instance.results)
                inits.append(init)
            arity = sum(len(edge.exposes) for edge in pivot.operands)
            lift = Lambda(
                params=(pivot.axis, *pivot.lift.params[1 : 1 + arity], *extra_params, *pivot.lift.params[1 + arity :]),
                body=Body(body),
                results=tuple(results),
            )
            states = (*pview.states, *(name for name, _, _ in injections))
            return Fold(operands=operands, lift=lift, init=tuple(inits), combine=recipe.program(states))
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
        own = self.lift.results if self.combine is None else self.exposes
        mapping.update((name, f"_r{index}") for index, name in enumerate(own))
        for index, (param, _, _) in enumerate(self.bindings):
            mapping.setdefault(param, f"_e{index}")
        counter = 0
        for stmt in self.lift.body.iter():
            for name in stmt.defines():
                if name not in mapping:
                    mapping[name] = f"_v{counter}"
                    counter += 1
        combine = None
        if self.combine is not None:
            # The combine's own names — its second operand, its temps — are nobody else's: they
            # renumber after the term's, so how a fold spelled its accumulators never reaches the form.
            own = dict(mapping)
            for name in (*self.combine.params, *(name for stmt in self.combine.body for name in stmt.defines())):
                own.setdefault(name, f"_c{len(own)}")
            combine = self.combine.rename(own)
        return replace(
            self,
            operands=tuple(edge.canonical() for edge in self.operands),
            lift=self.lift.rename(mapping),
            combine=combine,
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
            if term.combine is None:
                origin.update((name, (id(term), "step")) for stmt in term.lift.body for name in stmt.defines())
            else:
                origin.update((name, (id(term), "state")) for name in term.combine.results)
                if term.observe is not None:
                    origin.update((name, (id(term), "observed")) for name in term.observe.results)
            pending.extend(reversed(term.operands))
        owned: dict[tuple[int, str], list[OutputSpec]] = {}
        for spec in stores:
            key = origin.get(spec.write.values[0])
            assert key is not None and all(origin.get(name) == key for name in spec.write.values), (
                f"a store over {spec.write.values} writes values no one term defines"
            )
            owned.setdefault(key, []).append(spec)
        for spec in stores:
            if spec.sweep is not None:
                coordinates.setdefault(spec.sweep.name, spec.sweep)
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
                for edge in term.operands:
                    place(edge, loops, node if step else path)
                if step:
                    target = stmts if stmts is not None else sink(node)
                    target.extend(step)
                    attach(term, "step", target, node, scope)
                return
            inner: list[Stmt] = []
            for edge in term.operands:
                place(edge, [*loops, (term.axis, scope | {term.axis}, inner)], node)
            inner.extend(term.step())
            attach(term, "observed", inner, node, scope | {term.axis})
            target = stmts if stmts is not None else sink(node)
            if term.axis not in coordinates:
                raise ValueError(f"lower: no extent for reduce axis {term.axis!r} — the kernel's axis table names it")
            target.append(Loop(axis=coordinates[term.axis], body=Body(tuple(dict.fromkeys(inner)))))
            attach(term, "state", target, node, scope)

        def assemble(path: tuple[str, ...]) -> Body:
            body = list(nest[path])
            for name in opened[opened.index(path[-1]) + 1 :] if path else opened:
                if (*path, name) in nest:
                    body.append(Loop(axis=coordinates[name], body=assemble((*path, name))))
            return Body(tuple(dict.fromkeys(body)))

        place(self, [], None)
        return assemble(())


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
    combine = s.combine.rename(rename) if s.combine is not None else None
    observe = None
    if s.observe is not None:
        # The observer renames in lockstep: param 0 tracks the axis, the state params track the
        # combine's renamed results, the body/results are ordinary SSA material.
        observe = Lambda(
            params=(*lead, *(rename(p) for p in s.observe.params[1:])),
            body=Body(tuple(_rewrite(st, rename, sigma, axis_fn) for st in s.observe.body)),
            results=tuple(rename(r) for r in s.observe.results),
        )
    return replace(s, operands=operands, lift=lift, combine=combine, observe=observe)


__all__ = [
    "ReductionView",
    "Fold",
]
