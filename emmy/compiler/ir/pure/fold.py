"""``Fold`` — the ONE reduce term: ``reduce(⊕) ∘ map(f)`` in the λ-foldMap spelling.

The whole stored vocabulary of Tile IR, and a PURE term throughout: an optional iteration
``axis``, a pure ``lift`` :class:`~emmy.compiler.ir.pure.lam.Lambda`, the monoid's flat
``(init, combine)`` pair, and a tuple of ``operands`` — the closed inputs, each an edge bound
positionally to a lift param. Every reading (the map at zero axes, the bilinear
:class:`ContractionView`, the :class:`SlabView` leaf, the serial step) is DERIVED from those
params; nothing else is stored.

Nothing here is a :class:`~emmy.compiler.ir.stmt.base.Stmt`. A composed step — flash's ``Σ Q·K``
ahead of its ``Σ_j P·V``, split-K's sliced contraction — is reached through ``operands``, and its
POSITION in the emitted step stream is produced by the derivation (:meth:`Fold.step`
heads the inline-node edges; :func:`splice_operands` places each edge's body before the first read
of its bound name), not by sitting in a statement list. The term becomes statements in exactly one
place, :meth:`Fold.lower` / :attr:`Fold.loop`. See ``ir/ARCHITECTURE.md``, "Pure terms vs
statements".

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
from emmy.compiler.ir.pure.algebra import component_ops, rename_combine
from emmy.compiler.ir.pure.lam import Lambda
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Stmt

# ``Body.structural_key()`` dispatches :func:`emmy.compiler.ir.stmt.passes.rewrite` over every
# stmt for SSA / Expr / axis canonicalization. Register the structural node's handler here — an
# INLINE node operand dispatches back through the same registry, so a stored computed operand
# (the cone, flash's ``P``) canonicalizes like any other subtree.
from emmy.compiler.ir.stmt.passes import _rewrite_kind
from emmy.compiler.ir.stmt.passes import rewrite as _rewrite  # noqa: E402
from emmy.utils import cached_method


@dataclass(frozen=True)
class SlabView:
    """A Fold's reading as one gmem read — a leaf that DECLARES the coordinates it indexes.

    ``load`` is the read itself, ``axes`` the coordinates it binds (the term's own). No operands,
    no monoid, a body of one ``Load``: what ``isinstance(edge, Load)`` used to ask, back when a
    statement could sit in ``operands``. Placement never offers one as a seam — there is nothing
    to materialize that the load does not already do.
    """

    load: Load
    axes: tuple[Axis, ...]


@dataclass(frozen=True)
class ContractionView:
    """A Fold's BILINEAR reading, as geometry: the axis its operands share and the free axis each
    one contributes.

    ``a[m,k] × b[n,k]`` reads as ``axis=k, left=m, right=n``. That is the whole of the recognition
    — no algebra walk, no product-argument matching, no canonical form to compare against. Which
    of ``left`` / ``right`` is physically M or N is the PLACEMENT's answer, not the term's.
    """

    axis: Axis
    left: str | None
    """The free axis A carries — the output role it strides. ``None`` where A brings none."""
    right: str | None
    """The free axis the streamed operand carries. ``None`` for a MATVEC, whose B is a vector over
    the reduction alone: a contraction does not stop being one for want of a second output axis;
    whether the pair can be ORIENTED as (m, n) is the placement's question, not recognition's."""
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
    question: ``operands[0]`` is A by canonical form, k-last, which is what normalization
    guarantees by swapping the pair."""


@dataclass(frozen=True)
class Fold:
    """A scheduled reduce — the typed successor of the bare annotated reduce
    ``Loop`` (``ir/pure/algebra``). It splits the reduce's **algebra** (the loop-carried
    flat ⊕ — componentwise for a plain ``sum`` / ``max`` / ``mean``, a rescaling program for
    online-softmax / flash — a plain :class:`Lambda` either way) from its **structure** (the
    reduce ``axis`` + the per-element ``step`` it folds). Every reading is **derived** from those
    params (:meth:`as_contraction`, :meth:`as_slab`, :meth:`step`), never stored. The fold
    ``Loop`` is **synthesized on demand** (:meth:`lower`), never stored — so the same node tiles under any
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
    # (params: the iteration var first, then one per operand edge, bound POSITIONALLY) plus the
    # TRUE monoid's flat ``(init, combine)`` pair whose combine carries the REAL accumulator
    # names (its results). The serial step, the ``Accum`` forms and the ``carrier`` annotation
    # are DERIVED (:attr:`step` / ``__post_init__``). ------------------------------------------- #
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
            self._declares(self.lift.params[arity:])
            return
        # Formation validates the positional binding and the S × S → S arity; the ``carrier``
        # annotation is a DERIVED read (:attr:`carrier`), never a second stored spelling.
        n = len(self.init)
        if len(self.combine.params) != 2 * n or len(self.combine.results) != n:
            raise ValueError(f"Fold combine must be S × S → S at arity {n}: params={self.combine.params} results={self.combine.results}")
        lam = self.lift
        assert lam.params and lam.params[0] in {axis.name for axis in self.axes}, (
            f"lift param 0 must be the iteration var, one of the term's axes {[axis.name for axis in self.axes]}: {lam.params}"
        )
        # One lift param per operand RESULT COMPONENT (a product edge — split-K's sliced
        # multi-channel fold — binds every component), positionally.
        # POSITIONAL, so checked positionally: param i+1 is operand result component i. Comparing
        # NAMES here is what coupled a consumer's params to its producers' spelling — it is why an
        # edge could not be canonicalized without breaking the term above it, and why reading an
        # edge's result names was load-bearing at the constructor.
        arity = sum(len(edge.exposes) for edge in self.operands)
        assert len(lam.params) >= 1 + arity, f"lift binds {len(lam.params) - 1} params after the axis for {arity} operand result components"
        assert len(lam.results) == n, "one lift result per monoid component"
        self._declares((self.axis.name, *lam.params[1 + arity :]))
        if self.observe is not None:
            assert tuple(self.observe.params) == (self.axis.name, *self.combine.results), (
                f"observer params {self.observe.params} must bind the iteration var then the carried state "
                f"{(self.axis.name, *self.combine.results)} positionally"
            )
            defined = {name for stmt in self.observe.body for name in stmt.defines()}
            assert all(r in defined for r in self.observe.results), (
                "observer results must be FRESH names its body defines — never the carried state itself "
                "(the boundary distinguishes a streamed store from a post-fold store by the name)"
            )
            assert not any(isinstance(stmt, Fold) for stmt in self.observe.body), "an observer body holds plain stmts, never a nested node"

    def _declares(self, coordinates: tuple[str, ...]) -> None:
        # A TERM DECLARES THE COORDINATES IT READS. The lift's params past the operand binding —
        # what ``Lambda.closing`` left unbound — are exactly the names of this term's ``axes``: a
        # slab's declared coordinates, a reduce's iteration var beside any coordinate its step
        # reads outright, a projection's grid axes. One spelling, so :attr:`free_axes` is a
        # declaration and never a walk, and a coordinate that reaches a lift with no axis to bind
        # it fails here rather than as an undefined name at nvcc.
        declared = {axis.name for axis in self.axes}
        assert set(coordinates) == declared, (
            f"the lift reads coordinates {sorted(coordinates)} but the term declares axes {sorted(declared)}"
        )

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
            return self.lift.results
        state = self.combine.results
        return state if self.observe is None else (*state, *self.observe.results)

    @cached_property
    def free_axes(self) -> frozenset[str]:
        """The coordinates this term is evaluated over as seen from OUTSIDE — its free coordinates.

        DECLARED, not discovered: a slab declares the coordinates its load indexes, a term's
        operands contribute theirs, and the axis a term BINDS (its reduce axis) is not free above
        it. So a reduce over ``k`` does not vary with ``k`` to the fold that holds it, and a
        nested reduce that happens to bind the same name as the loop above still hoists. Every
        reader asks this one question — does the term vary with an axis — and intersects instead
        of walking a lowered body for names that look axis-shaped.
        """
        space = {axis.name for axis in self.axes}
        for edge in self.operands:
            space |= edge.free_axes
        return frozenset(space - ({self.axis.name} if self.axis is not None else set()))

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
        pluses = component_ops(self.combine)
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
        by_name = {name: edge for edge in self.operands for name in edge.exposes}
        a_edge = self.operands[0]
        a_names = set(a_edge.exposes)
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
        if self.axis.name not in a_space & b_space:
            return None
        left_only, right_only = a_space - b_space, b_space - a_space
        if len(left_only) > 1 or len(right_only) > 1:
            return None  # more than one free axis a side is not an orientable output role
        slab = b_edge.as_slab()
        b_trans = slab is not None and self.axis.name in slab.load.index[-1].free_vars()
        return ContractionView(
            axis=self.axis,
            left=next(iter(left_only), None),
            right=next(iter(right_only), None),
            product=product,
            plus=plus,
            b_trans=b_trans,
        )

    @cached_method
    def as_slab(self) -> SlabView | None:
        """The :class:`SlabView` of this term — its one gmem read and the coordinates it declares —
        or ``None`` for a computed cone. Memoized on the term."""
        if self.operands or self.combine is not None or len(self.lift.body) != 1 or not isinstance(self.lift.body[0], Load):
            return None
        return SlabView(load=self.lift.body[0], axes=self.axes)

    @property
    def axis(self) -> Axis | None:
        """The REDUCTION axis — the one the lift iterates, named by its leading param — or
        ``None`` when this term binds none: a slab's axes are all coordinates, and a projection has
        no iteration. Read off the lift, never off a position in :attr:`axes`: which of a term's
        declared axes it reduces is what the lift says, and the rest are coordinates it reads.
        """
        if self.combine is None:
            return None
        return next(axis for axis in self.axes if axis.name == self.lift.params[0])

    @property
    def composed(self) -> Fold | None:
        """The single sliced contraction this outer reduce COMPOSES (split-K's
        reassociation ``fold_k = fold_{ksplit} ∘ fold_{kslice}``), or ``None`` — the identity-lift
        λ spelling (one inline node operand carrying the outer's exact accumulator state). The
        structural probe :meth:`step` reads (``030_cut`` builds its sliced partial directly, so
        the composition is a recognized FORM here, never a required input)."""
        if len(self.lift.body) or len(self.operands) != 1:
            return None
        inner = self.operands[0]
        return inner if inner.as_contraction() is not None else None

    # ---- the DERIVED READINGS. ``Map`` and ``Contraction`` are no longer stored kinds (the
    # collapse); every field they carried reads back off the one stored term here, so their old
    # accessors keep their exact meanings and their consumers keep their exact spellings. ------- #
    @classmethod
    def slab(cls, load: Load, axes: tuple[Axis, ...]) -> Fold:
        """One gmem read as a term that DECLARES the coordinates it indexes.

        A ``Load`` is a statement, and a statement sitting in a term tree is the one leaf whose
        coordinates are still free names — the last place a coordinate escapes its binder. Wrapped,
        the leaf binds them: ``axes`` are the enclosing axes the load actually reads, in binding
        order, and the lift's params are those same names — the formation rule every term states.
        A coordinate no enclosing axis supplies fails formation: the lambda reads a name it does
        not bind. No ``combine``: a slab iterates, it does not reduce.
        """
        read = {name for expr in load.exprs() for name in expr.free_vars()}
        declared = tuple(axis for axis in axes if axis.name in read)
        return cls(axes=declared, lift=Lambda(params=tuple(axis.name for axis in declared), body=Body((load,)), results=load.names))

    @cached_method
    def step(self) -> Body:
        """The per-step statements this fold DERIVES from its stored parameters: the lift body,
        then the combine APPLIED at the injected singleton — its second-operand params bound to
        the lift's results — with each result-defining assign spelled as the in-place ``Accum`` over the carried
        state, the loop-IR form whose seed is the ⊕'s identity. An observer's pure tap runs last,
        so a streamed store reads the post-combine (inclusive-prefix) state.

        A reduce that COMPOSES a sliced contraction (split-K) has no step of its own: the inner
        contraction already updates the shared accumulators, so the reassociation is the
        embedding itself. Without a combine the term is a map and the step is the lift body.
        Deterministic from the stored parameters, so kernel identity depends on no classified view.
        Memoized on the term.
        """
        if self.combine is None:
            return self.lift.body
        if self.composed is not None:
            return Body()
        states = self.combine.results
        named = dict(zip(self.combine.params[len(states) :], self.lift.results, strict=True))
        out: list[Stmt] = [*self.lift.body]
        applied = [stmt.rename(named) for stmt in self.combine.body]
        definitions = {stmt.name: stmt for stmt in applied if stmt.name not in states}  # temps; a state's rewrite is not a def

        def reads(name: str, state: str) -> bool:
            stmt = definitions.get(name)
            return name == state or (stmt is not None and any(reads(arg, state) for arg in stmt.args))

        for stmt in applied:
            if stmt.name not in states:
                out.append(stmt)
                continue
            state, args = stmt.name, stmt.args
            if stmt.op.name == "copy" and (pivot := definitions.get(args[0])) is not None and state in pivot.args:
                # The pivot's final write copies its own ``maximum`` temp: accumulate that maximum.
                stmt, args = pivot, pivot.args
            if state in args:
                value = next(arg for arg in args if arg != state) if args != (state, state) else state
                out.append(Accum(name=state, value=value, op=stmt.op, axes=(self.axis.name,)))
            else:
                base, value = args if reads(args[0], state) else (args[1], args[0])
                out.append(Accum(name=state, value=value, op=stmt.op, base=base, axes=(self.axis.name,)))
        if self.observe is not None:
            out.extend(self.observe.body)
        return Body(tuple(out))

    @cached_method
    def canonical(self) -> Fold:
        """The α-canonical form of this TERM — a ``Fold``, the same kind that went in.

        The whole term, not its lift: a Fold's value also depends on its axis extent, its monoid
        and its operand edges, none of which live in ``lift``. Renames only what the term PRIVATELY
        binds — its axes and its lift's internal defs. The operand interface names are shared with
        the edges that produce them, and renaming one side without the other is how a lambda ends
        up not defining its own result; leaving them alone under-merges, which is the safe
        direction for a sharing or comparison key.

        FREE names pass through, so equal canonical forms mean equal value under the SAME
        environment.
        """
        mapping = {axis.name: f"_a{index}" for index, axis in enumerate(self.axes)}
        counter = 0
        for stmt in self.lift.body.iter():
            for name in stmt.defines():
                if name not in mapping:
                    mapping[name] = f"_v{counter}"
                    counter += 1

        def renamed(fn: Lambda) -> Lambda:
            return Lambda(
                params=tuple(mapping.get(name, name) for name in fn.params),
                body=Body(tuple(stmt.rename(mapping) for stmt in fn.body)),
                results=tuple(mapping.get(result, result) for result in fn.results),
            )

        return replace(
            self,
            axes=tuple(replace(axis, name=mapping[axis.name]) for axis in self.axes),
            operands=tuple(edge.canonical() for edge in self.operands),
            lift=renamed(self.lift),
            # The observer binds the iteration var and reads the carried state, so it renames in
            # LOCKSTEP: renaming the axis without it leaves the observer reading a name that no
            # longer exists, and a scan would then canonicalize to something that is not a term.
            observe=None if self.observe is None else renamed(self.observe),
        )

    @cached_method
    def lower(self) -> Body:
        """Flatten this term to the Loop IR body the materializer expands.

        Three parts, read straight off the representation:

        * every operand is a TERM, so it lowers by the same method — the tree is homogeneous and
          there is nothing to dispatch on; operands lower before the statements that read them;
        * an operand whose :attr:`free_axes` does not contain this fold's axis does not vary
          with the step, so it lowers ONCE ahead of the loop. That is the whole of the hoist: a
          declaration compared against an axis, not a body walked for free names;
        * the :attr:`step` follows, and the loop binds the axis.

        A SHARED term — one object reached through several operand positions — defines its names
        once per scope: the same object lowers to the same statements, and a repeat is dropped.

        The ONE lowering spelling — every consumer of a term's statements calls this. Memoized on
        the term.
        """
        axis = self.axis
        rides = [edge for edge in self.operands if axis is not None and axis.name in edge.free_axes]
        ridden = {id(edge) for edge in rides}
        prologue = [stmt for edge in self.operands if id(edge) not in ridden for stmt in edge.lower()]
        step = [*(stmt for edge in rides for stmt in edge.lower()), *self.step()]
        if axis is None:
            return Body(tuple(dict.fromkeys([*prologue, *step])))
        return Body((*dict.fromkeys(prologue), Loop(axis=axis, body=Body(tuple(dict.fromkeys(step))), unroll=self.unroll)))


@_rewrite_kind.register
def _(s: Fold, rename, sigma, axis_fn):
    # ONE handler for the one stored kind (the collapse retired the Map / Contraction arms —
    # singledispatch keys on the stored type, and there is now only one). Every operand edge
    # dispatches back through the registry; the fold renames its lift / monoid in lockstep
    # (params track the operand names positionally, the combine's results ARE the accumulator
    # names). At zero axes there is no iteration var to rename and no monoid to thread.
    operands = tuple(_rewrite(edge, rename, sigma, axis_fn) for edge in s.operands)
    axes = tuple(axis_fn(axis) for axis in s.axes)
    axis = axis_fn(s.axis) if s.combine is not None else None  # the iteration var — a slab has none
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
        results=tuple(rename(r) for r in s.lift.results),
    )
    combine = rename_combine(s.combine, rename) if s.combine is not None else None
    observe = None
    if s.observe is not None:
        # The observer renames in lockstep: param 0 tracks the axis, the state params track the
        # combine's renamed results, the body/results are ordinary SSA material.
        observe = Lambda(
            params=(axis.name, *(rename(p) for p in s.observe.params[1:])),
            body=Body(tuple(_rewrite(st, rename, sigma, axis_fn) for st in s.observe.body)),
            results=tuple(rename(r) for r in s.observe.results),
        )
    return replace(s, axes=axes, operands=operands, lift=lift, combine=combine, observe=observe)


__all__ = [
    "Fold",
]
