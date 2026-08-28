"""Tile IR — a map/reduce kernel with its *schedule* made explicit.

One :class:`TileOp` is the article's reduction skeleton — ``project ∘ reduce(⊕, e) ∘ map(f)`` —
scheduled but not yet bound to hardware threads. It sits between Loop IR (pure iteration) and
Kernel IR (threads / smem):

    Loop IR ──lowering/tile──▶ Tile IR ──lowering/kernel──▶ Kernel IR

The whole point of the layer is the article's thesis: **the schedule is separate from the
combine.** The combine is not defined here — it is the :class:`~emmy.compiler.ir.pure.fold.Fold`
term (``ir/pure/fold.py``), which a ``TileOp`` holds whole in ``op``. What this module owns is
everything the term deliberately does not carry:

- the root-global schedule fields — the free-axis → grid :class:`~.schedule.Placement` (``place``),
  the ONE worker inventory (``work``) and the warp-spec split (``workers``);
- the per-node schedule SLICES in ``TileOp.schedule`` (``{codec key → resolved TilePlan /
  ReducePlan / Stage}``, keyed by the tree-path codec and read through ``ops.Sched``);
- the kernel's EFFECTS — the :class:`OutputSpec` decorations and the ``apply_output_specs`` /
  ``extract_output_specs`` pair that reconstitutes the effectful stmt stream from them.

That split is the layer's invariant, not a convenience. The stored term is pure algebra, IMMUTABLE
across the whole schedule search — a fork is a different slice map, never a rebuilt tree — which is
what makes kernel identity (``TileOp.structural_key``) the algebra alone, with placement, slices,
workers and output specifications all excluded. Tile IR stores only pure terms; statements appear when the term is
lowered, never inside it (``ir/ARCHITECTURE.md``, "Pure terms vs statements").

There is no per-kind kernel/schedule type: dispatch reads the role structurally off the node (a
fold's role derives), so a projection, a reduction and a contraction all ride the same ``TileOp``.
The kernel materializer reads the schedule off the slice beside the node — it never re-recognizes
structure the tile IR already holds.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import Op
from emmy.compiler.ir.expr import Literal
from emmy.compiler.ir.pure import Lambda
from emmy.compiler.ir.pure.fold import Fold, deep_defines, edge_refs_axis, is_contraction, operand_body
from emmy.compiler.ir.schedule import Placement, WarpSpec
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Body, Loop, Stmt, Write, pretty_body
from emmy.compiler.ir.stmt.base import _axis_identity
from emmy.compiler.ir.stmt.body import _member_reads
from emmy.compiler.ir.tile.normalize import normalize_fold_tree
from emmy.compiler.ir.tile.path import sites
from emmy.compiler.structural import digest


@dataclass(frozen=True)
class OutputSpec:
    """One output write specification at the kernel boundary — the effect the stored term no
    longer carries. ``write`` is the store verbatim (target buffer, index template, stored value
    names, the atomic flag — holding the ``Write`` whole keeps every field lossless), and it is
    NOT part of the term: ``TileOp.output_specs`` owns the tuple, and consumers reconstitute the
    effectful stmt stream via :func:`apply_output_specs`. Consecutive ``sweep`` stores on one axis ride
    one per-cell output ``Loop`` (rms/softmax's normalize sweep, ``unroll`` preserved); the swept
    members are the trailing projection stmts reading the axis (:func:`_sweep_start`). Conversion sites go
    through :func:`extract_output_specs`, whose reconstitution round-trip gate is what keeps kernel
    sources byte-identical to the stored-``Write`` era."""

    write: Write
    sweep: Axis | None = None
    unroll: bool = False


@dataclass(frozen=True)
class ProjectionRegion:
    """A pure output projection repeated over one local free axis.

    A maximally fused kernel may have several sibling output loops with different extents.  Their
    computation remains in pure lambdas while :class:`OutputSpec` owns the writes.  The region is
    therefore structural map material, not a reduction and not an effectful Loop IR fallback.
    """

    axis: Axis
    lift: Lambda
    unroll: bool = False
    pure = True

    @property
    def body(self) -> Body:
        return self.lift.body

    @property
    def results(self) -> tuple[str, ...]:
        """Values observed by this region's output specifications."""
        return tuple(result for result in self.lift.results if isinstance(result, str))

    def defines(self) -> tuple[str, ...]:
        """A projection loop does not expose its per-iteration values to the enclosing scope."""
        return ()

    def deps(self) -> tuple[str, ...]:
        return self.lift.params[1:]

    def exprs(self) -> tuple:
        return ()

    def nested(self) -> tuple[Body, ...]:
        return (self.lift.body,)

    def with_bodies(self, bodies: tuple[Body, ...]) -> ProjectionRegion:
        (body,) = bodies
        return replace(self, lift=Lambda(self.lift.params, body, self.lift.results))

    def binds_axes(self) -> frozenset[str]:
        return frozenset({self.axis.name})

    def rewrite(self, rename_ssa, sigma=None, axis_fn=None):
        """Rename the pure region through the shared statement rewrite."""
        return _rewrite(
            self,
            rename_ssa,
            Sigma.IDENTITY if sigma is None else sigma,
            _axis_identity if axis_fn is None else axis_fn,
        )

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}project {self.axis.name} in 0..{self.axis.extent}", *pretty_body(self.body, indent + "    ")]


def _sweep_start(stmts, axis_name: str) -> int:
    """The first index of the trailing projection run a ``sweep`` store's output ``Loop``
    wraps — the earliest stmt reading the sweep axis (SSA deps + Expr free vars, deep). The
    trailing-RUN rule (everything from that stmt on is swept) is deliberately simple; the
    :func:`extract_output_specs` round-trip gate is what proves it reproduces the captured loop."""
    for i, s in enumerate(stmts):
        if axis_name in _member_reads(s):
            return i
    return len(stmts)


def observed_result_names(op) -> frozenset[str]:
    """Every observer result name on the term's folds — the STREAMED stores' values. A boundary
    write over one of these is the scan store: it rides each iteration of the observed fold's
    reduce loop (after the observer stmts) rather than the kernel tail. Derived from the term at
    each reconstitution site — never stored on the spec (store only what extraction destroyed)."""
    names: set[str] = set()
    stack = [op]
    while stack:
        node = stack.pop()
        if not isinstance(node, Fold):
            continue
        if node.observe is not None:
            names.update(node.observe.results)
        stack.extend(node.operands)
        stack.extend(s for s in node.lift.body if isinstance(s, Fold))
    return frozenset(names)


def _splice_streamed(stmts: list, write: Write) -> bool:
    """Splice a streamed store into the (deepest) reduce ``Loop`` whose body defines its observed
    values — at the body end, after the observer stmts the derived step placed last. ``False``
    when no such loop is present (a term-level stream, where the store stays at its post-node
    position)."""
    values = set(write.values)
    for i, stmt in enumerate(stmts):
        if not isinstance(stmt, Loop):
            continue
        inner = list(stmt.body)
        if _splice_streamed(inner, write):
            stmts[i] = stmt.with_bodies((Body(tuple(inner)),))
            return True
        if stmt.is_reduce and values <= set().union(*(deep_defines(s) for s in inner)):
            stmts[i] = stmt.with_bodies((Body((*inner, write)),))
            return True
    return False


def apply_output_specs(stmts, specs, *, observed: frozenset = frozenset()) -> list[Stmt]:
    """Reassemble the EFFECTFUL projection stmt stream from a pure projection body + the
    kernel-boundary output specifications — the ONE reconstitution rule the scheduler's tail gates, the
    materializer's zero-axis ``Fold`` peel and ``035_split_reduce`` share, so the lowered kernels stay
    byte-identical to the stored-``Write`` era. A plain store appends its ``Write``; consecutive
    ``sweep`` stores on one axis wrap the trailing run of stmts reading that axis
    (:func:`_sweep_start`) into one per-cell output ``Loop``, with the ``Write`` run last."""
    specs = tuple(specs)
    claimed: set[int] = set()

    def expand(body) -> list:
        out = []
        for stmt in body:
            if not isinstance(stmt, ProjectionRegion):
                out.append(stmt)
                continue
            inner = expand(stmt.body)
            results = set(stmt.results)
            owned = [spec for spec in specs if set(spec.write.values) <= results]
            claimed.update(id(spec) for spec in owned)
            inner.extend(spec.write for spec in owned)
            out.append(Loop(axis=stmt.axis, body=Body(inner), unroll=stmt.unroll))
        return out

    out = expand(stmts)
    stores = [spec for spec in specs if id(spec) not in claimed]
    index = 0
    while index < len(stores):
        st = stores[index]
        if st.sweep is None:
            # A store over OBSERVED values streams into its reduce loop when the loop is present
            # (a lowered kernel body); at term level — the fold not yet a ``Loop`` — it keeps its
            # post-node position, which is also where extraction's round-trip gate expects it.
            if not (observed and set(st.write.values) <= observed and _splice_streamed(out, st.write)):
                out.append(st.write)
            index += 1
            continue
        end = index + 1
        while end < len(stores) and stores[end].sweep == st.sweep and stores[end].unroll == st.unroll:
            end += 1
        start = _sweep_start(out, st.sweep.name)
        writes = tuple(store.write for store in stores[index:end])
        out = [*out[:start], Loop(axis=st.sweep, body=Body((*out[start:], *writes)), unroll=st.unroll)]
        index = end
    return out


def _projection_results(body) -> set[str]:
    out = set()
    for member in body:
        if isinstance(member, ProjectionRegion):
            out.update(member.results)
            out.update(_projection_results(member.body))
    return out


def _implicit_unit_row(specs: tuple[OutputSpec, ...], free: tuple[Axis, ...]) -> Axis | None:
    """Recover an elided matrix row when every boundary write proves ``[0, n]``.

    The column axis may already be free or may still be the one shared output sweep that
    contraction canonicalization will promote. A pure reshape of the same column coordinate
    proves the same boundary: all varying index expressions still read only ``n`` while a
    literal-zero coordinate carries the elided unit row.
    """
    if not specs:
        return None
    if len(free) == 1 and all(spec.sweep is None for spec in specs):
        n_name = free[0].name
    elif not free and all(spec.sweep is not None for spec in specs) and len({spec.sweep.name for spec in specs}) == 1:
        n_name = specs[0].sweep.name
    else:
        return None
    for spec in specs:
        index = spec.write.index
        variables = frozenset().union(*(expr.free_vars() for expr in index))
        if not (n_name in variables and variables <= {n_name} and any(isinstance(expr, Literal) and expr.value == 0 for expr in index)):
            return None
    return Axis("_um", Dim(1))


def lower_with_output_specs(op, specs) -> list[Stmt]:
    """Lower one pure Tile term and attach every output specification at its owning scope."""
    specs = tuple(specs)

    def lower_body(body) -> list[Stmt]:
        out = []
        for member in body:
            if isinstance(member, Fold):
                out.extend(member.lower())
                continue
            if isinstance(member, ProjectionRegion):
                inner = lower_body(member.body)
                results = set(member.results)
                inner.extend(spec.write for spec in specs if set(spec.write.values) <= results)
                out.append(Loop(axis=member.axis, body=Body(inner), unroll=member.unroll))
                continue
            out.append(member)
        return out

    if isinstance(op, Fold) and op.axis is None:
        body = [*(stmt for edge in op.operands for stmt in operand_body(edge)), *lower_body(op.body)]
        root_specs = tuple(spec for spec in specs if not set(spec.write.values) <= _projection_results(op.body))
        return apply_output_specs(body, root_specs, observed=observed_result_names(op))
    return apply_output_specs(op.lower(), specs, observed=observed_result_names(op))


def extract_output_specs(stmts) -> tuple[tuple[Stmt, ...], tuple[OutputSpec, ...]] | None:
    """Split an effectful projection stmt stream into ``(pure stmts, OutputSpec decorations)`` — the
    conversion-side inverse of :func:`apply_output_specs`, valid ONLY when the reconstitution
    round-trips byte-identically (checked here; ``None`` otherwise). It handles flat root writes, the
    legacy single output sweep, and recursively nested sibling output loops. Each sibling loop becomes a
    pure :class:`ProjectionRegion`; every write becomes an :class:`OutputSpec`. An already-pure stream
    returns ``(stmts, ())``."""
    original = list(stmts)
    rest = list(stmts)
    stores: list[OutputSpec] = []
    while rest and isinstance(rest[-1], Write):
        stores.insert(0, OutputSpec(write=rest.pop()))
    if not stores and rest and isinstance(rest[-1], Loop) and not rest[-1].is_reduce:
        loop = rest[-1]
        inner = list(loop.body)
        writes = []
        while inner and isinstance(inner[-1], Write):
            writes.insert(0, inner.pop())
        if writes and all(s.pure for s in inner):
            stores.extend(OutputSpec(write=write, sweep=loop.axis, unroll=loop.unroll) for write in writes)
            rest = [*rest[:-1], *inner]
    if all(s.pure for s in rest) and apply_output_specs(rest, stores) == original:
        return tuple(rest), tuple(stores)

    def extract(body: Body) -> tuple[Body, list[OutputSpec], list[OutputSpec]] | None:
        pure = []
        outputs: list[OutputSpec] = []
        direct: list[OutputSpec] = []
        for stmt in body:
            if isinstance(stmt, Write):
                spec = OutputSpec(write=stmt)
                outputs.append(spec)
                direct.append(spec)
                continue
            if isinstance(stmt, Loop) and not stmt.is_reduce:
                child = extract(stmt.body)
                if child is None:
                    return None
                child_body, child_outputs, child_direct = child
                results = tuple(dict.fromkeys(value for spec in child_direct for value in spec.write.values))
                provisional = Lambda(params=(stmt.axis.name,), body=child_body, results=results)
                captures = tuple(sorted(provisional.free_names()))
                region = ProjectionRegion(
                    axis=stmt.axis,
                    lift=Lambda(params=(stmt.axis.name, *captures), body=child_body, results=results),
                    unroll=stmt.unroll,
                )
                pure.append(region)
                outputs.extend(child_outputs)
                continue
            if not stmt.pure:
                return None
            pure.append(stmt)
        return Body(pure), outputs, direct

    extracted = extract(Body.coerce(stmts))
    if extracted is None:
        return None
    body, outputs, _ = extracted
    if apply_output_specs(body, outputs) != original:
        return None
    return tuple(body), tuple(outputs)


@dataclass
class TileOp(Op):
    """One scheduled map/reduce kernel (see module docstring).

    Holds the structural-IR root ``op`` (a :class:`Fold`, at any role, or ``None`` for a
    placeholder node) plus the schedule fields — not a pre-lowered body. The per-cell loop-IR
    body is generated at materialize time by ``op.lower()``, and a bare reduction / contraction's
    output ``Write`` is glue generated there too (from ``place.grid`` + the graph node's output
    buffer; see ``lowering/kernel/010_materialize``). ``inputs`` / ``outputs`` come from the base
    :meth:`Op.populate_io` (graph edges) — no body walk.

    Schedule fields (all defaulted, so a fresh / placeholder node is well-formed):

    - ``place`` — the free-axis → grid binding (:class:`~.schedule.Placement`); root-global.
    - ``workers`` — the warp-specialization split (:class:`~.schedule.WarpSpec`); root-global, ``None`` =
      uniform SIMT.

    There is **no** let table: a computed operand is stored inline on its edge, and sharing is the
    product contraction's arity (see the module docstring), so stored trees are already
    resolved and every walk is a plain tree walk. The per-node schedule SLICES live in
    ``schedule``: ``{codec key → resolved TilePlan / ReducePlan / Stage}``, keyed by the
    tree-path codec's canonical key (:mod:`~emmy.compiler.ir.tile.path` — a fold may carry all
    three families at once, so the path alone cannot key the map; the family selects the slice
    kind, so key and value agree by construction). The ``op`` term is pure algebra, IMMUTABLE
    across the whole schedule search — a fork is a different map, never a rebuilt tree. Read /
    write through :class:`~emmy.compiler.ir.tile.ops.Sched` (``ops.reduce_plan`` is the plan
    accessor); ``lower`` never sees the slices, so kernel identity (``Op.cache_key``) is untouched."""

    op: object = None
    name: str = ""
    place: Placement = field(default_factory=Placement)
    workers: WarpSpec | None = None
    schedule: dict = field(default_factory=dict)
    # The kernel's output specifications: every explicit ``Write`` (and the legacy rms/softmax
    # output-sweep spelling) as a kernel-boundary fact beside ``place``. Empty for a
    # bare reduction / contraction — its grid-cell store
    # stays the materializer's default glue (``_factor.with_store``). Consumers reconstitute
    # the effectful stmt stream via ``apply_output_specs`` — never read a ``Write`` out of the term.
    output_specs: tuple[OutputSpec, ...] = ()
    # The ONE worker inventory (``ir.schedule.Workers``): the ``w``/``n`` worker
    # tokens factored out of the per-site TILE values, derived at option assembly
    # (``ops.Sched.seal_workers`` — loud on cross-site disagreement). ``None`` = the per-cell /
    # pure-reduce forms (derived launch geometry). The wire format spells the inventory ONCE, in
    # ``WORK``; the site values carry no worker tokens and the retired embedded spellings raise.
    work: object = None
    # Whether the graph-level Fold-edge placement fork kept this kernel fused. Cut pieces are
    # fresh TileOps with the default ``False`` and may expose their own smaller seam set.
    placement_decided: bool = False
    # Whether the split QUESTION is consumed for this kernel: the structural cross-CTA fork
    # (``035_split_reduce``) declined it (the unsplit arm), or the kernel is a realized split's
    # independent projection SIBLING — which has no sliced axis, so it carries this flag as its
    # consumed-split receipt (a ``REDUCE`` pin's ``g`` half strips on it). The partial / finalize
    # pieces need no flag: their sliced axis's partition ``Window`` is the receipt. Widening the
    # flag past "declined" is safe because the partition receipt is an explicit pool-key term of
    # the schedule memo — a receipt-bearing twin can never serve a receipt-free one.
    split_consumed: bool = False

    def __post_init__(self) -> None:
        scope_axes = (*self.place.free, *(store.sweep for store in self.output_specs if store.sweep is not None))
        axes = tuple(dict.fromkeys(axis.name for axis in scope_axes))
        normalized = normalize_fold_tree(self.op, axes)
        unit_row = _implicit_unit_row(self.output_specs, self.place.free)
        if unit_row is not None:
            candidate_free = (unit_row, *self.place.free)
            candidate_scope = (*candidate_free, *(store.sweep for store in self.output_specs if store.sweep is not None))
            candidate_axes = tuple(dict.fromkeys(axis.name for axis in candidate_scope))
            candidate = normalize_fold_tree(self.op, candidate_axes, implicit_axes=(unit_row.name,))
            if any(is_contraction(site.node) for site in sites(candidate)):
                normalized = candidate
                self.place = replace(self.place, free=candidate_free)
        if self.schedule and normalized != self.op:
            raise ValueError("cannot canonicalize a TileOp after schedule slices have been attached")
        self.op = normalized
        if self.place.is_mapped:
            return

        contractions = tuple(site.node for site in sites(normalized) if is_contraction(site.node))
        promoted = {
            store.sweep.name
            for store in self.output_specs
            if store.sweep is not None
            and any(any(edge_refs_axis(edge, store.sweep.name) for edge in contraction.operands) for contraction in contractions)
        }
        if not promoted:
            return
        free_names = {axis.name for axis in self.place.free}
        extra = tuple(
            {
                store.sweep.name: store.sweep
                for store in self.output_specs
                if store.sweep is not None and store.sweep.name in promoted - free_names
            }.values()
        )
        if extra:
            self.place = Placement(free=(*self.place.free, *extra))
        self.output_specs = tuple(
            replace(store, sweep=None) if store.sweep is not None and store.sweep.name in promoted else store for store in self.output_specs
        )
        # Promotion changes the enclosing-axis context that closes computed contraction operands.
        # Normalize once under the final scope so reconstructing this TileOp cannot expose a
        # different Fold tree or placement seam.
        scope_axes = (*self.place.free, *(store.sweep for store in self.output_specs if store.sweep is not None))
        final_axes = tuple(dict.fromkeys(axis.name for axis in scope_axes))
        renormalized = normalize_fold_tree(self.op, final_axes)
        if self.schedule and renormalized != self.op:
            raise ValueError("cannot canonicalize a TileOp after schedule slices have been attached")
        self.op = renormalized

    def pretty_body(self) -> str:
        """The structural dump — delegated to :mod:`~emmy.compiler.ir.tile._dump`, which owns
        every presentation concern in the layer."""
        from emmy.compiler.ir.tile._dump import tile_body  # noqa: PLC0415 — presentation, loaded on demand

        return tile_body(self)

    def structural_key(self) -> str:
        """Kernel identity — the stored term's α-invariant digest (``""`` for a placeholder).
        Placement, schedule slices, workers and output specifications are deliberately EXCLUDED: identity is
        the algebra alone (the NO-schedule-fields rule above), so every fork sibling of one term
        shares the key and no emission path can leak a schedule into it."""
        return self.op.structural_key() if self.op is not None else ""

    def cache_key(self) -> str | None:
        return digest(type(self).__name__, self.structural_key(), self._knob_key())


__all__ = [
    "OutputSpec",
    "ProjectionRegion",
    "TileOp",
    "apply_output_specs",
    "extract_output_specs",
    "lower_with_output_specs",
    "observed_result_names",
]


from emmy.compiler.ir.stmt.passes import rewrite as _rewrite  # noqa: E402


@_rewrite.register
def _(region: ProjectionRegion, rename, sigma, axis_fn):
    axis = axis_fn(region.axis)
    lift = Lambda(
        params=(axis.name, *(rename(param) for param in region.lift.params[1:])),
        body=Body(_rewrite(stmt, rename, sigma, axis_fn) for stmt in region.body),
        results=tuple(rename(result) if isinstance(result, str) else result for result in region.lift.results),
    )
    return replace(region, axis=axis, lift=lift)
