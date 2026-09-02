"""Tile IR — a map/reduce kernel with its *schedule* made explicit.

One :class:`TileOp` is the article's reduction skeleton — ``project ∘ reduce(⊕, e) ∘ map(f)`` —
scheduled but not yet bound to hardware threads. It sits between Loop IR (pure iteration) and
Kernel IR (threads / smem):

    Loop IR ──lowering/tile──▶ Tile IR ──lowering/kernel──▶ Kernel IR

The whole point of the layer is the article's thesis: **the schedule is separate from the
combine.** The combine is not defined here — it is the :class:`~emmy.compiler.ir.pure.fold.Fold`
term (``ir/pure/fold.py``), which a ``TileOp`` holds whole in ``op``. What this module owns is
everything the term deliberately does not carry:

- the free-axis → grid :class:`~.schedule.Placement` (``place``), an accepted site-indexed
  :class:`Schedule`, and its separate materialization;
- the kernel's EFFECTS — the :class:`OutputSpec` decorations and the ``apply_output_specs`` /
  ``extract_output_specs`` pair that reconstitutes the effectful stmt stream from them.

That split is the layer's invariant, not a convenience. The stored term is pure algebra, IMMUTABLE
across the whole schedule search — a fork is a different assignment, never a rebuilt tree — which is
what keeps kernel identity (``Op.identity_key`` over the derived ``loop_body``) schedule-free, with
the schedule, materialization, placement binding and workers excluded. Tile IR stores only
pure terms; statements appear when the term is
lowered, never inside it (``ir/ARCHITECTURE.md``, "Pure terms vs statements").

There is no per-kind kernel/schedule type: dispatch reads the role structurally off the node (a
fold's role derives), so a projection, a reduction and a contraction all ride the same ``TileOp``.
The kernel materializer reads the schedule by site — it never re-recognizes structure the tile IR already holds.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property

from frozendict import frozendict

from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import Op
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.pure.tree import Visit, walk
from emmy.compiler.ir.schedule import Placement, WarpSpec
from emmy.compiler.ir.schedule.base import Schedule
from emmy.compiler.ir.schedule.packing import packed_readings
from emmy.compiler.ir.schedule.views import (
    ContractionFacts,
    EdgeSite,
    NodeId,
    contraction_facts,
)
from emmy.compiler.ir.stmt import Body, Loop, OutputSpec, Stmt, Write
from emmy.compiler.ir.stmt.body import free_names
from emmy.compiler.ir.tile.normalize import normalize_fold_tree
from emmy.compiler.ir.tile.path import sites


def _sweep_start(stmts, axis_name: str) -> int:
    """The first index of the trailing projection run a ``sweep`` store's output ``Loop``
    wraps — the earliest stmt reading the sweep axis. The trailing-RUN rule (everything from that
    stmt on is swept) is deliberately simple; the :func:`extract_output_specs` round-trip gate is
    what proves it reproduces the captured loop."""
    for i, s in enumerate(stmts):
        if axis_name in free_names(s):
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
        if stmt.is_reduce and values <= set().union(*(Body((s,)).ssa_defs for s in inner)):
            stmts[i] = stmt.with_bodies((Body((*inner, write)),))
            return True
    return False


def apply_output_specs(stmts, specs, *, observed: frozenset = frozenset()) -> Body:
    """Reassemble the EFFECTFUL stmt stream from a pure statement STREAM + the kernel-boundary
    output specifications — the materializer's spelling, where the grid binds every free axis
    and nothing is a term: a store appends as the kernel tail, consecutive ``sweep`` stores on one
    axis wrap the trailing run of stmts reading that axis (:func:`_sweep_start`) into one per-cell
    output ``Loop``, and a store over OBSERVED values streams into its reduce loop when that loop
    is present. The ONE reconstitution rule the scheduler's tail gates, the materializer's
    zero-axis ``Fold`` peel and ``030_cut`` share, so the lowered kernels stay byte-identical to
    the stored-``Write`` era. A TERM places its stores itself (:meth:`Fold.lower`)."""
    out = list(stmts)
    stores = list(specs)
    index = 0
    while index < len(stores):
        st = stores[index]
        if st.sweep is None:
            # At term level — the fold not yet a ``Loop`` — an observed store keeps its post-node
            # position, which is also where extraction's round-trip gate expects it.
            if not (observed and set(st.write.values) <= observed and _splice_streamed(out, st.write)):
                out.append(st.write)
            index += 1
            continue
        end = index + 1
        while end < len(stores) and stores[end].sweep == st.sweep:
            end += 1
        start = _sweep_start(out, st.sweep.name)
        out = [*out[:start], Loop(axis=st.sweep, body=Body((*out[start:], *(store.write for store in stores[index:end]))))]
        index = end
    return Body(tuple(out))


def _dense_axis_suffix(index: tuple, name: str) -> bool:
    """Whether ``index`` is one dense coordinate, directly or split by a row-major reshape."""
    if not index:
        return False
    stride = 1
    for position in reversed(range(len(index))):
        expr = index[position]
        dim = None
        if position:
            if not (
                isinstance(expr, BinaryExpr)
                and expr.op == "%"
                and isinstance(expr.right, Literal)
                and isinstance(expr.right.value, int)
                and expr.right.value > 0
            ):
                return False
            expr, dim = expr.left, expr.right.value
        if stride != 1:
            if not (
                isinstance(expr, BinaryExpr) and expr.op in ("/", "//") and isinstance(expr.right, Literal) and expr.right.value == stride
            ):
                return False
            expr = expr.left
        if expr != Var(name):
            return False
        if dim is not None:
            stride *= dim
    return True


def _implicit_unit_row(specs: tuple[OutputSpec, ...], free: tuple[Axis, ...]) -> Axis | None:
    """Recover an elided matrix row when every boundary write proves ``[0..., n]``.

    The column axis may already be free or may still be the one shared output sweep that
    contraction canonicalization will promote. The unit coordinates must be a non-empty leading
    zero prefix, followed by the dense column coordinate directly or through a row-major reshape.
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
        split = next((position for position, expr in enumerate(index) if not (isinstance(expr, Literal) and expr.value == 0)), len(index))
        if split == 0 or not _dense_axis_suffix(index[split:], n_name):
            return None
    return Axis("_um", Dim(1))


def extract_output_specs(stmts) -> tuple[tuple[Stmt, ...], tuple[OutputSpec, ...]] | None:
    """Split an effectful projection stmt stream into ``(pure stmts, OutputSpec decorations)`` — the
    conversion-side inverse of :func:`apply_output_specs`, valid ONLY when the reconstitution
    round-trips byte-identically (checked here; ``None`` otherwise). The trailing run of root
    ``Write`` stmts and output loops splits off: a write is a plain spec, an output loop (a non-reduce
    ``Loop`` whose writes end its body — sibling sweeps included) gives one ``sweep`` spec per write
    over its axis, its pure prefix rejoining the stream. An already-pure stream returns
    ``(stmts, ())``."""
    original = tuple(stmts)
    rest = list(stmts)
    stores: list[OutputSpec] = []
    while rest:
        last = rest[-1]
        if isinstance(last, Write):
            stores.insert(0, OutputSpec(write=rest.pop()))
            continue
        if isinstance(last, Loop) and not last.is_reduce:
            inner = list(last.body)
            writes: list[Write] = []
            while inner and isinstance(inner[-1], Write):
                writes.insert(0, inner.pop())
            if writes and all(s.pure for s in inner):
                stores[0:0] = [OutputSpec(write=write, sweep=last.axis) for write in writes]
                rest = [*rest[:-1], *inner]
                continue
        break
    if all(s.pure for s in rest) and apply_output_specs(rest, stores) == original:
        return tuple(rest), tuple(stores)
    return None


@dataclass(frozen=True)
class TileOp(Op):
    """One scheduled map/reduce kernel (see module docstring).

    Holds the structural-IR root ``op`` (a :class:`Fold`, at any role, or ``None`` for a
    placeholder node) plus the schedule fields — not a pre-lowered body. The per-cell loop-IR
    body is generated at materialize time by ``op.lower()``, and a bare reduction / contraction's
    output ``Write`` is glue generated there too (from ``place.grid`` + the graph node's output
    buffer; see ``lowering/kernel/010_materialize``). ``inputs`` / ``outputs`` come from the base
    :meth:`Op.with_io` (graph edges) — no body walk.

    Schedule fields (all defaulted, so a fresh / placeholder node is well-formed):

    - ``place`` — the free-axis → grid binding (:class:`~.schedule.Placement`); root-global.
    - ``workers`` — the warp-specialization split (:class:`~.schedule.WarpSpec`); root-global, ``None`` =
      uniform SIMT.

    There is **no** let table: a computed operand is stored inline on its edge, and sharing is the
    product contraction's arity (see the module docstring), so stored trees are already
    resolved and every walk is a plain tree walk. An accepted ``schedule`` assignment contains
    choices only; ``materialization`` separately contains placed geometry and resolved transport
    facts. There is no second schedule map or per-node schedule field. The ``op`` term is pure
    algebra, IMMUTABLE across the whole schedule search. Read through
    :class:`~emmy.compiler.ir.tile.ops.Sched`; ``lower`` never sees the schedule, so kernel identity
    (``identity_key(with_io=True, with_knobs=True)``) is untouched."""

    op: object = None
    name: str = ""
    place: Placement = field(default_factory=Placement)
    workers: WarpSpec | None = None
    # The accepted semantic assignment and its derived lowering facts. Unscheduled Tile IR carries
    # neither; scheduling installs both together.
    schedule: Schedule | None = field(default=None, compare=False, repr=False)
    materialization: object | None = field(default=None, compare=False, repr=False)
    # The kernel's output specifications: every explicit ``Write`` (and the legacy rms/softmax
    # output-sweep spelling) as a kernel-boundary fact beside ``place``. Empty for a
    # bare reduction / contraction — its grid-cell store
    # stays the materializer's default glue (``_factor.with_store``). Consumers reconstitute
    # the effectful stmt stream via ``apply_output_specs`` — never read a ``Write`` out of the term.
    output_specs: tuple[OutputSpec, ...] = ()
    # Whether the graph-level Fold-edge placement decision is consumed. Unpinned cut pieces keep
    # the default ``False`` and may expose their own smaller seam set; a pinned cut sets it on both
    # pieces because its authoritative decision cannot name a fresh tree.
    placement_decided: bool = False
    # Whether the split QUESTION is consumed for this kernel: the structural cross-CTA fork
    # (``030_cut``) declined it (the unsplit arm), or the kernel is a realized split's
    # independent projection SIBLING — which has no sliced axis, so it carries this flag as its
    # consumed-split receipt (a ``REDUCE`` pin's ``g`` half strips on it). The partial / finalize
    # pieces need no flag: their sliced axis's partition ``Window`` is the receipt. Widening the
    # flag past "declined" is safe because the partition receipt is an explicit pool-key term of
    # the schedule memo — a receipt-bearing twin can never serve a receipt-free one.
    split_consumed: bool = False

    def __post_init__(self) -> None:
        Op.__post_init__(self)
        scope_axes = (*self.place.free, *(store.sweep for store in self.output_specs if store.sweep is not None))
        axes = tuple(dict.fromkeys(axis.name for axis in scope_axes))
        free_names = {axis.name for axis in self.place.free}
        sweep_axes = frozenset(name for name in axes if name not in free_names)
        normalized = normalize_fold_tree(self.op, axes, sweep_axes=sweep_axes)
        unit_row = _implicit_unit_row(self.output_specs, self.place.free)
        if unit_row is not None:
            candidate_free = (unit_row, *self.place.free)
            candidate_scope = (*candidate_free, *(store.sweep for store in self.output_specs if store.sweep is not None))
            candidate_axes = tuple(dict.fromkeys(axis.name for axis in candidate_scope))
            candidate_sweeps = frozenset(name for name in candidate_axes if name not in {axis.name for axis in candidate_free})
            candidate = normalize_fold_tree(self.op, candidate_axes, implicit_axes=(unit_row.name,), sweep_axes=candidate_sweeps)
            if any(site.node.as_contraction() is not None for site in sites(candidate)):
                normalized = candidate
                object.__setattr__(self, "place", replace(self.place, free=candidate_free))
        if self.schedule is not None and normalized != self.op:
            raise ValueError("cannot canonicalize a TileOp after a schedule has been attached")
        object.__setattr__(self, "op", normalized)

        contractions = tuple(site.node for site in sites(normalized) if site.node.as_contraction() is not None)
        promoted = {
            store.sweep.name
            for store in self.output_specs
            if store.sweep is not None and any(any(store.sweep.name in edge.free_axes for edge in con.operands) for con in contractions)
        }
        if not promoted:
            self._validate_schedule()
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
            free = (*self.place.free, *extra)
            if self.place.is_mapped:
                grid = self.place.grid or self.place.free
                object.__setattr__(self, "place", replace(self.place, free=free, grid=(*grid, *extra), mapped=True))
            else:
                object.__setattr__(self, "place", replace(self.place, free=free))
        object.__setattr__(
            self,
            "output_specs",
            tuple(
                replace(store, sweep=None) if store.sweep is not None and store.sweep.name in promoted else store
                for store in self.output_specs
            ),
        )
        # Promotion changes the enclosing-axis context that closes computed contraction operands.
        # Normalize once under the final scope so reconstructing this TileOp cannot expose a
        # different Fold tree or placement seam.
        scope_axes = (*self.place.free, *(store.sweep for store in self.output_specs if store.sweep is not None))
        final_axes = tuple(dict.fromkeys(axis.name for axis in scope_axes))
        final_sweeps = frozenset(name for name in final_axes if name not in {axis.name for axis in self.place.free})
        renormalized = normalize_fold_tree(self.op, final_axes, sweep_axes=final_sweeps)
        if self.schedule is not None and renormalized != self.op:
            raise ValueError("cannot canonicalize a TileOp after a schedule has been attached")
        object.__setattr__(self, "op", renormalized)
        self._validate_schedule()

    def _validate_schedule(self) -> None:
        """Enforce the schedule/materialization boundary on construction."""
        if self.schedule is None and self.materialization is None:
            return
        if self.schedule is None or self.materialization is None:
            raise ValueError("a scheduled TileOp requires both a schedule and materialization")
        validate = getattr(self.materialization, "validate", None)
        if not callable(validate):
            raise TypeError("schedule materialization must provide validate(schedule, source, place=..., workers=...)")
        validate(self.schedule, self, place=self.place, workers=self.workers)

    @cached_property
    def sites(self) -> tuple[Visit, ...]:
        """The ONE walk over this kernel's term, one record per node identity, indexed by node id.

        Every structural reading is a FIELD of these records — the node, the parent that reached
        it, the axes in scope there, its segment path, whether it is derived evaluation — so the
        walk runs once per kernel and nothing re-derives a label it already carries. Those labels
        are POSITIONS in this kernel's tree, which is why they live here and never on the shared
        subterms the tree is built from: one Fold reached down two paths has two parents, and keeps
        the first that reached it.
        """
        out, seen = [], set()
        for visit in walk(self.op) if isinstance(self.op, Fold) else ():
            if id(visit.node) not in seen:
                seen.add(id(visit.node))
                out.append(visit)
        return tuple(out)

    @cached_property
    def node_sites(self) -> tuple[NodeId, ...]:
        """Every node identity, in stable schedule order."""
        return tuple(range(len(self.sites)))

    @cached_property
    def edge_sites(self) -> tuple[EdgeSite, ...]:
        """Every consumer operand position in stable schedule order."""
        return tuple((consumer, operand) for consumer, site in enumerate(self.sites) for operand in range(len(site.node.operands)))

    @cached_property
    def _node_ids(self) -> dict[int, NodeId]:
        return {id(site.node): node_id for node_id, site in enumerate(self.sites)}

    def node_id(self, node: Fold) -> NodeId:
        """Return ``node``'s schedule identity by object identity."""
        try:
            return self._node_ids[id(node)]
        except KeyError:
            raise KeyError("Fold is not a node of this TileOp") from None

    @cached_property
    def incident_edges(self) -> frozendict[NodeId, tuple[EdgeSite, ...]]:
        """Each consumer's operand positions."""
        out: dict[NodeId, list[EdgeSite]] = {site: [] for site in range(len(self.sites))}
        for edge in self.edge_sites:
            out[edge[0]].append(edge)
        return frozendict({site: tuple(edges) for site, edges in out.items()})

    @cached_property
    def views(self) -> tuple[Fold, ...]:
        """The TERM at each node site — a term is its own classification.

        ``axis is None`` is the projection reading, :meth:`Fold.as_contraction` the bilinear one,
        so a wrapper kind restating them bought nothing. Indexed BY node id, which is a dense
        ordinal — a tuple, not a map keyed by the integers it is already ordered by.
        """
        return tuple(site.node for site in self.sites)

    def contracts(self, site: NodeId) -> bool:
        """Whether one site is a contraction-capable reduction — the shape TILE and STAGE want."""
        view = self.views[site]
        return view.as_contraction() is not None

    @cached_property
    def family_sites(self) -> frozendict[str, tuple[NodeId, ...]]:
        """The node sites each classic node family can address, by family name.

        One classification, filtered once per family, so a reader dispatches on the family NAME it
        already holds instead of picking a differently-named member per family.
        """
        return frozendict(
            {
                "TILE": tuple(
                    site
                    for site in self.node_sites
                    if self.contracts(site) or (self.views[site].axis is None and site == 0 and not self.sites[site].node.operands)
                ),
                "REDUCE": tuple(site for site in self.node_sites if self.views[site].axis is not None),
            }
        )

    @cached_property
    def stage_edges(self) -> tuple[EdgeSite, ...]:
        """The operand positions a ``STAGE`` transport can address."""
        return tuple(edge for edge in self.edge_sites if self.contracts(edge[0]))

    @cached_property
    def _packed_readings(self) -> frozendict:
        return packed_readings(tuple(site.node for site in self.sites if site.node.as_contraction() is not None), self.inputs)

    def packed_reading(self, node) -> tuple:
        """One node's ``(B copy, pair)`` packed-operand readings.

        Only a contraction carries the operand roles the readings match on, so anything else
        answers ``(None, None)`` rather than being absent."""
        return self._packed_readings.get(id(node), (None, None))

    @cached_property
    def contractions(self) -> frozendict[NodeId, ContractionFacts]:
        """The per-contraction structure every schedule choice over this kernel shares."""
        return contraction_facts(self) if isinstance(self.op, Fold) else frozendict()

    def __getstate__(self):
        """Pickle stored fields only; derived schedule inventories recompute after transport."""
        return {name: self.__dict__[name] for name in self.__dataclass_fields__ if name in self.__dict__}

    def pretty_body(self) -> str:
        """The structural dump — delegated to :mod:`~emmy.compiler.ir.tile._dump`, which owns
        every presentation concern in the layer."""
        from emmy.compiler.ir.tile._dump import tile_body  # noqa: PLC0415 — presentation, loaded on demand

        return tile_body(self)

    @cached_property
    def loop_body(self) -> Body | None:
        """The complete schedule-free Loop-IR body this kernel executes, derived from the term
        — what the identity lattice digests. The closed program: ``Fold.lower`` with nothing bound
        and the boundary stores handed in, so the term binds every free coordinate with its own
        loops and places each store after the term defining its value — the extents, the store
        program (index spelling, ``atomicAdd``, width, output sweeps) and a cut child's typed seam
        ``Load`` are all in the body. Schedule-free by construction: ``lower`` never reads the
        classic assignment, and ``place`` stays out entirely — which coordinates the grid binds,
        and in what order the source nest spelled them, is execution choice, not identity. A bare
        reduction carries no ``Write`` — its grid-cell store is materializer glue derived from
        ``place.grid``, so the empty store stream is itself derivable. Cached: the term and the
        kernel-boundary fields are immutable across the schedule search. ``None`` for a
        placeholder."""
        if self.op is None:
            return None
        return self.op.lower(frozenset(), self.output_specs)

    def _body_identity(self, *, structural: bool = True) -> str | None:
        """Override :meth:`Op._body_identity` with the DERIVED body: :attr:`loop_body`'s
        canonical digest, so a golden record derives the SAME key from its persisted program
        (both sides lower through the one spelling) and term re-spellings that lower alike
        share it."""
        body = self.loop_body
        return None if body is None else body.structural_key(structural=structural)


__all__ = [
    "OutputSpec",
    "TileOp",
    "apply_output_specs",
    "extract_output_specs",
    "observed_result_names",
]
