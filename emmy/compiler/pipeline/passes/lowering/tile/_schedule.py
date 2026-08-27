r"""Schedule a lifted ``TileOp`` as a lazy product over its complete Fold tree.

Every Fold remains an addressable site. Each site contributes a small local catalog, and
``_RowProduct`` composes those catalogs through generic interfaces: one worker inventory and equal
tile geometry on shared physical axes. A derived contraction with a unit marker axis inherits its
enclosing Fold's reduction domain; this is a parent/child interface fact, not an operation-family
match.

**A site's catalog is a function of the site.** ``WORK`` is not an input to it — it is FOLDED OUT
of a row's own slices (:func:`derive_inventory`) and carried on the row as its claim, so the
kernel-global inventory is a JOIN KEY over the catalogs rather than a loop around them. One
enumeration answers for every inventory: ``_ComboSpace`` joins the sites on the claim, and
``_space`` partitions the single product into one segment per inventory. The alternative — asking
each site what it offers under each candidate inventory — rebuilds every stage and reduce catalog
once per member of a list those same catalogs produce.

The product, compatible joins, worker segments, and launch-order product are all addressable
sequences. A live compile retains that space through the fork and creates a candidate dictionary
only when search visits its index. There is no flat candidate list or row-count budget. Offline
sampling uses the same index path.

Catalogs contain choices only; legality filters them, and the evidence hierarchy ranks them.
Materialization re-resolves the selected spellings against the same stored Fold nodes. An empty
space leaves the term unmapped.
"""

from __future__ import annotations

import logging
from bisect import bisect_right
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from itertools import accumulate, product
from types import MappingProxyType

from emmy.compiler.dim import Dim
from emmy.compiler.ir.atom import atoms_for
from emmy.compiler.ir.axis import Axis, Window
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.pure import Lambda, M
from emmy.compiler.ir.pure.fold import Fold, deep_reads, edge_refs_axis, is_contraction
from emmy.compiler.ir.schedule import Level as _ReduceLevel
from emmy.compiler.ir.schedule import (
    Raster,
    ReducePlan,
    Stage,
    TilePlan,
    WarpSpec,
    Workers,
    derive_inventory,
    plan_workers,
    resolve_site_tile,
)
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Stmt, Write
from emmy.compiler.ir.stmt.passes import has_contraction_tail
from emmy.compiler.ir.tile import OutputSpec, Placement, TileOp
from emmy.compiler.ir.tile.identity import hint_extent, pool_key
from emmy.compiler.ir.tile.ops import Sched, cone_seam, edge_dtypes, projection_tail, scheduled
from emmy.compiler.ir.tile.path import Site, sites
from emmy.compiler.pipeline.fork import Fork, Level, build_fork_tree
from emmy.compiler.pipeline.knob import family_of, schedule_pin_fingerprint, values_equal
from emmy.compiler.pipeline.passes.lowering.tile import _legality as legal
from emmy.compiler.pipeline.passes.lowering.tile._pool import Block, PoolSpace, Segment
from emmy.compiler.pipeline.search.space import (
    RASTER,
    REDUCE,
    STAGE,
    TILE,
    WORK,
    coop_reduce_moves,
    map_tile_moves,
    raster_moves,
    scalar_tile_moves,
    splitk_moves,
    stage_moves,
    warp_tile_moves,
)
from emmy.compiler.structural import digest

logger = logging.getLogger(__name__)

#: The per-site schedule families this enumeration decides, IN THE ORDER their keys lead the fork
#: levels. ``WORK`` and ``RASTER`` are kernel-global and bracket them; ``PLACE`` is the seam
#: family — resolved from ``PLACE`` pins, never enumerated here.
#:
#: Not a copy of ``path.SLICE_FAMILIES`` even though the members match: that one answers "which
#: families key a slice" (a set) and this one "in what order do their levels nest" (a sequence).
#: Aliasing them would make a fork-level reordering look like an edit to the addressing vocabulary.
FAMILIES = ("TILE", "STAGE", "REDUCE")

#: The ``Knob`` each family pins through.
_KNOBS = {"TILE": TILE, "STAGE": STAGE, "REDUCE": REDUCE}

#: The most compatible interface combinations one inventory's pass may enumerate
#: (:meth:`_ComboSpace.build`). The product runs over the SITES, so it is exponential in their
#: number: a term whose sites each offer two compatible geometries has 2^sites of them, and terms
#: with hundreds of sites exist — a whole fused model reaches here as one term. Exceeding this is a
#: LOUD refusal, never a truncation: a truncated product reads as "these are the compatible
#: schedules" while dropping whichever the walk reached last. Measured headroom: the widest live
#: term (a fused flash pair, every tier) composes ~24k.
#:
#: This bounds the COMPOSITION, not the space. How many candidates the composed product addresses
#: is a prefix sum over it, and unbounded by design.
MAX_COMBINATIONS = 1 << 20

# ---- structural reads over the stored term ----------------------------------------------------- #


def _node_loads(node) -> list[Load]:
    """Every gmem ``Load`` the term reads, as a deep walk over the STORED structure: an operand
    edge's MATERIALIZED inhabitant plus the loads sitting inline in a lift body, recursing through
    a COMPUTED edge's own node.

    Deliberately hand-written rather than a ``Body`` walk: an operand edge is not a body
    dependency, and ``Fold.nested()`` withholds a contraction's edges precisely so generic walkers
    do not read its multiply arguments as statements. Crossing the edges is the whole point here,
    so the walk alternates node-wise and statement-wise and visits each node exactly once."""
    out: list[Load] = []

    def walk_stmts(stmts) -> None:
        for s in stmts:
            if isinstance(s, Fold):
                walk(s)  # an inline COMPUTED edge is a node, not a statement — one visit, over there
                continue
            if isinstance(s, Load):
                out.append(s)
            for b in s.nested():
                walk_stmts(list(b))

    def walk(n) -> None:
        if isinstance(n, Load):
            out.append(n)
        elif isinstance(n, Fold):
            for e in n.operands:
                walk(e)
            # A contraction's lift IS the synthesized multiply — its reads are the edges above.
            if not is_contraction(n):
                walk_stmts(list(n.lift.body))

    walk(node)
    return out


def _projection(op) -> Body:
    """The kernel's per-cell projection — the wrapping zero-axis fold's body, or empty when the
    term is a bare node. A projection has ONE home (the wrapper's lift), never a node field."""
    return op.lift.body if isinstance(op, Fold) and op.axis is None and op.operands else Body(())


def _inner_free(place: Placement) -> Axis | None:
    """The innermost NON-UNIT free axis — a synthesized unit axis can sit
    innermost, and it is not the axis the transposed emitter sweeps."""
    if not place.free:
        return None
    return next((a for a in reversed(place.free) if not (a.extent.is_static and a.extent.as_static() == 1)), None)


def _shared_row_buf(carrier_loads, tail, grid_vars, raxis: Axis, inputs) -> str | None:
    """The input buffer reused as a CTA-shared ROW across the reduce + a contraction tail — read in
    the carrier at ``(grid…, raxis)`` AND in the tail at ``(grid…, k)``, its trailing dim the
    (static) reduce extent. ``None`` ⇒ no eligible operand (stay gmem-direct)."""
    if not raxis.extent.is_static or not has_contraction_tail(tail):
        return None
    n = len(grid_vars)
    carrier_bufs = {
        s.input for s in carrier_loads if len(s.index) == n + 1 and tuple(s.index[:n]) == grid_vars and s.index[-1] == Var(raxis.name)
    }
    for s in (ld for ld in Body(tail).loads if ld.is_scalar):
        if s.input in carrier_bufs and len(s.index) == n + 1 and tuple(s.index[:n]) == grid_vars:
            t = inputs.get(s.input)
            if t is not None and t.shape and t.shape[-1].is_static and t.shape[-1].as_static() == raxis.extent.as_static():
                return s.input
    return None


def _row_stage(term: _Term, node) -> Stage | None:
    """The shared-row :class:`Stage` a COOPERATIVE reduce over ``node`` can drive, or ``None``. It
    is a RESOLVER, not a choice: the row spells ``d1/smem`` exactly when the shape carries an
    operand the CTA can hold as a shared row across the reduce and its contraction tail, and the
    materializer re-resolves the same buffer off the same term."""
    tail = projection_tail(term.tile)
    if not has_contraction_tail(tail):
        return None
    grid_vars = tuple(Var(a.name) for a in term.place.grid)
    carrier_loads = [ld for ld in _node_loads(node) if ld.is_scalar]
    buf = _shared_row_buf(carrier_loads, tail, grid_vars, node.axis, term.tile.inputs)
    return Stage(transport="smem", smem=(buf,)) if buf is not None else None


def _strippable(term: _Term) -> bool:
    """Whether the pointwise cell admits the register strip: a pure zero-axis fold with no operands
    whose body is FLAT elementwise (per-cell ``Load`` / ``Assign`` + boundary root stores, no nested
    ``Loop`` / carried state), over a static innermost free axis."""
    op, place = term.tile.op, term.place
    if not (isinstance(op, Fold) and op.axis is None and not op.operands) or not place.free:
        return False
    if not place.free[-1].extent.is_static:
        return False
    return all(isinstance(s, (Load, Assign, Write)) for s in op.body) and all(st.sweep is None for st in term.tile.output_specs)


# ---- the site tree ------------------------------------------------------------------------------ #


@dataclass(frozen=True)
class _Node:
    """One node of the SITE TREE the enumeration walks: a schedule site, the canonical key each
    family spells it with, and the sites nested under it."""

    site: Site
    keys: dict[str, str]
    children: tuple[_Node, ...]


def _site_tree(op, key) -> tuple[_Node, ...]:
    """``op``'s scheduling sites as a TREE — the topmost ones first, each carrying the sites nested
    under it. The walker is ``path.sites``; this only groups its output by containment, so a term
    shape never gets a site list of its own.

    A node IS a site exactly when some family can key it, and ``key`` (``_Term.key`` over
    ``Sched.key``) already answers that — it spells a family site and returns ``None`` otherwise —
    so the key dict each node carries doubles as the membership test, and there is no second site
    predicate to keep in step with the codec's."""
    keyed = [(site, {family: spelled for family in FAMILIES if (spelled := key(family, site.node)) is not None}) for site in sites(op)]
    keyed = [(site, keys) for site, keys in keyed if keys]

    # ``sites`` is preorder. Build containment from that order + depth, not from path-prefix set
    # membership: sibling roots may have the same canonical segment path (and differ only by the
    # codec ordinal), so a prefix-only query attaches the first sibling's children to both roots.
    roots: list[tuple[Site, dict, list]] = []
    stack: list[tuple[Site, dict, list]] = []
    for site, keys in keyed:
        while stack and (stack[-1][0].depth >= site.depth or site.segments[: stack[-1][0].depth] != stack[-1][0].segments):
            stack.pop()
        record = (site, keys, [])
        (stack[-1][2] if stack else roots).append(record)
        stack.append(record)

    def freeze(record) -> _Node:
        site, keys, children = record
        return _Node(site=site, keys=keys, children=tuple(freeze(child) for child in children))

    return tuple(freeze(record) for record in roots)


def _kids(node: _Node) -> tuple[_Node, ...]:
    """The site's complete Fold children, shared by enumeration and materialization."""
    return node.children


@dataclass(frozen=True)
class _FoldInterface:
    """A Fold plus the reduction domain it presents to its parent schedule."""

    fold: Fold
    axis: Axis
    seam: tuple

    def __getattr__(self, name: str):
        return getattr(self.fold, name)


def _scheduled_node(node: _Node, parent: _Node | None):
    """The Fold interface a site presents to scheduling.

    A contraction in a parent's derived step can carry a unit axis merely to bind its result. Its
    reduction domain is the enclosing Fold axis; replacing that axis for catalog and legality
    queries exposes the actual parent/child interface without rewriting the stored Fold tree.
    """
    fold = node.site.node
    enclosing = parent.site.node if parent is not None else None
    if (
        node.site.derived
        and is_contraction(fold)
        and fold.axis.extent.is_static
        and fold.axis.extent.as_static() == 1
        and isinstance(enclosing, Fold)
        and enclosing.axis is not None
    ):
        carried = tuple(enclosing.combine.results[: -len(fold.combine.results)])
        return _FoldInterface(fold, enclosing.axis, ((), (), carried))
    return fold


# ---- the candidate values, per site ------------------------------------------------------------- #


class _Term:
    """Everything enumeration reads about one stored term: its grid placement, target, schedule
    key speller, and per-site tile catalogs grouped by worker inventory."""

    def __init__(self, tile: TileOp, place: Placement, ctx) -> None:
        self.tile = tile
        self.place = place
        self.ctx = ctx
        self.sched = Sched(tile.op, {}, place=place)
        self.proj = _projection(tile.op)
        self._tiles: dict[int, dict[str, list[TilePlan]]] = {}
        self._dtypes: dict[int, tuple] = {}
        self._seams: dict[int, tuple] = {}
        self._producers: dict[int, Fold | None] = {}
        self._interfaces: dict[tuple[int, int | None], Fold | _FoldInterface] = {}
        self._partitioned: bool | None = None
        self.tree = _site_tree(tile.op, self.key)
        self.tile_nodes = {node.keys["TILE"]: node.site.node for root in self.tree for node in _walk_nodes(root) if "TILE" in node.keys}
        self.fragment_edges = self._fragment_edges()
        #: The refusal a schedule PIN drew, kept until the walk is done. One inventory declining
        #: a pin is ordinary (the widths are read OFF the inventory, so the pin names a different
        #: plan under each); a pin NO inventory could spell is malformed, and that is loud.
        self.pin_error: ValueError | None = None
        self.pin_spelled = False
        #: Set when any site offers a tensor-core row — a structural fact about the KERNEL, stamped
        #: on EVERY row so the priors can price "a scalar tile where tensor cores were on offer".
        self.warp_eligible = False

    def key(self, family: str, node) -> str | None:
        """The canonical key ``family`` spells for ``node``, or ``None`` when it has no site."""
        return self.sched.key(family, node.fold if isinstance(node, _FoldInterface) else node)

    def scheduled_node(self, node: _Node, parent: _Node | None = None):
        """The stable scheduling interface for one stored site and its parent."""
        key = (id(node.site.node), id(parent.site.node) if parent is not None else None)
        if key not in self._interfaces:
            self._interfaces[key] = _scheduled_node(node, parent)
        return self._interfaces[key]

    def seam(self, node: Fold) -> tuple:
        """The computed-A seam cached once per immutable contraction node."""
        if isinstance(node, _FoldInterface):
            return node.seam
        if id(node) not in self._seams:
            self._seams[id(node)] = cone_seam(node.a, node.axis.name) if not isinstance(node.a, Load) else ((), (), ())
        return self._seams[id(node)]

    def producer(self, node: Fold) -> Fold | None:
        """The single contraction producer nested in a computed edge, when one exists."""
        if id(node) not in self._producers:
            edge = node.a
            candidates = (
                tuple(site.node for site in sites(edge) if is_contraction(site.node) and edge_refs_axis(site.node, node.axis.name))
                if isinstance(edge, Fold)
                else ()
            )
            self._producers[id(node)] = candidates[0] if len(candidates) == 1 else None
        return self._producers[id(node)]

    def _fragment_edges(self) -> tuple[tuple[str, str], ...]:
        """Scheduled contraction producer/consumer edges, keyed by their TILE sites."""
        out: list[tuple[str, str]] = []

        def link(consumer, producer) -> None:
            pair = self.key("TILE", consumer), self.key("TILE", producer)
            if None not in pair and pair not in out:
                out.append(pair)

        def walk(current: _Node, parent: _Node | None = None) -> None:
            node = self.scheduled_node(current, parent)
            producer = self.producer(node) if is_contraction(node) else None
            if producer is not None:
                link(node, producer)

            # A Fold may compute a fragment operand through sibling contractions in its stored
            # step.  The dependency is the backward cone of the consumer's computed edges; this
            # is the same generic dataflow relation the fragment evaluator follows, with no
            # operation-family recognition.
            if isinstance(node, Fold) and node.axis is not None and not is_contraction(node):
                steps = node.step_stmts()
                states = set(node.combine.results)
                for index, consumer in ((i, stmt) for i, stmt in enumerate(steps) if is_contraction(stmt)):
                    accumulated = any(
                        isinstance(stmt, Accum) and stmt.name in states and stmt.value in consumer.defines() for stmt in steps[index + 1 :]
                    )
                    reads = {name for edge in consumer.operands if isinstance(edge, Fold) for name in deep_reads(edge.lower())}
                    if not accumulated or not reads:
                        continue
                    cone = Body(tuple(steps[:index])).backward_cone(reads)
                    producers = tuple(stmt for stmt in cone.members if is_contraction(stmt))
                    if len(producers) == 1:
                        link(consumer, producers[0])
            for child in current.children:
                walk(child, current)

        for root in self.tree:
            walk(root)
        return tuple(out)

    def carries_partition(self) -> bool:
        """Whether this immutable kernel already contains a split receipt, cached for the catalog."""
        if self._partitioned is None:
            self._partitioned = _carries_partition(self.tile.op)
        return self._partitioned

    def pin(self, family: str, node) -> str | None:
        """The live env pin for this site's ``family`` key — ``EMMY_KNOBS``'s ``FAMILY@<element>``
        entry, falling back to the bare ``EMMY_<FAMILY>``. Unset reads ``None``, which is the
        distinction the enumeration keys on: an unset ``TILE`` offers the catalog, a set one is
        authoritative."""
        key = self.key(family, node)
        if key is None:
            return None
        knob, element = _KNOBS[family], key.partition("@")[2]
        return knob.narrow_at(element) if element else knob.raw()

    def tiles(self, node) -> dict[str, list[TilePlan]]:
        """The contraction node's ``TILE`` catalog, grouped by the ``WORK`` spelling each candidate
        implies (``""`` for the untiled per-cell tile, which composes with any inventory a
        cooperative ``REDUCE`` claims)."""
        if id(node) not in self._tiles:
            self._tiles[id(node)] = self._build_tiles(node)
        return self._tiles[id(node)]

    def _build_tiles(self, node) -> dict[str, list[TilePlan]]:
        atoms = _warp_atoms(self, node)
        warp = [p for p in warp_tile_moves(atoms) if _tile_ok(self, node, p)] if atoms else []
        self.warp_eligible = self.warp_eligible or bool(warp)
        # A computed edge is ordinary scalar code: the register tile evaluates its body once per
        # operand row/column and reuses the result across the sibling cells. Only a multi-channel
        # product needs the warp compute fill; the scalar emitter carries one accumulator channel.
        scalar = (
            [plan for plan in scalar_tile_moves() if not plan.is_tiled or _placed(self, node, plan).axes is not None]
            if _supports_scalar(node)
            else []
        )
        grouped: dict[str, list[TilePlan]] = {}
        for plan in scalar + warp:
            w = plan_workers(plan)
            grouped.setdefault(w.spell() if w is not None else "", []).append(plan)
        # A ``TILE`` pin is authoritative over the VALUES but not over the inventories: its unit
        # widths are read OFF ``WORK``, so the pin names a different plan under each one and is
        # re-resolved against each in :func:`_pinned_tiles`. The catalog still answers "which
        # inventories can this site spell against", pinned or not.
        return grouped


def _has_computed_operand(node) -> bool:
    """Whether either role is an inline *pointwise cone* eligible for smem compute fill.

    A nonzero-axis Fold is a nested scheduling site, not a scalar producer evaluated at each
    contraction cell.  Keeping that distinction preserves multi-site reduce enumeration while the
    zero-axis cones created by ``make_cone`` take the fused transport.
    """

    def eligible(edge) -> bool:
        return isinstance(edge, Fold) and edge.axis is None

    return eligible(node.a) or any(eligible(ch.b) for ch in node.channels)


def _requires_sync_fill(node) -> bool:
    """Whether a warp contraction must use the synchronous shared-memory fill.

    A computed edge must be evaluated into a slab. A product contraction with more than one B/C
    channel must copy one shared A slab plus every compatible B slab before one A fragment feeds
    all MMA accumulator channels; the gmem-direct, async-copy, and scalar emitters are deliberately
    single-channel.
    """
    return _has_computed_operand(node) or len(node.channels) > 1


def _supports_scalar(node) -> bool:
    """Whether the scalar atom can carry this contraction.

    Inline operand cones are evaluated directly by the scalar register tile. A multi-channel
    product needs one accumulator family per channel, which remains a warp compute-fill form.
    """
    return len(node.channels) == 1


def _converting_a(node, atom, inputs) -> bool:
    """Whether the ``a`` edge is a MATERIALIZED load whose dtype the atom cannot bind directly —
    the CONVERTING smem compute fill's case (Gemma's erased ``.float()`` cast ahead of an f16
    projection): the synchronous fill evaluates the load per slab cell and the typed slab store
    performs the conversion. A byte transport moves raw bits and cannot, so such an edge takes the
    fill or nothing. ``False`` for computed edges (the fill's native case), matching dtypes, and
    1-byte loads (the f8 tiers move raw bits by design)."""
    if not isinstance(node.a, Load) or not inputs:
        return False
    if atom.operand_dtype("a").nbytes < 2:
        return False
    t = inputs.get(node.a.input)
    return t is not None and t.dtype.nbytes >= 2 and t.dtype != atom.operand_dtype("a")


def _needs_fill(term: _Term, node, plan: TilePlan) -> bool:
    """Whether this warp candidate's operands take the mandatory smem compute fill — a computed
    edge, a multi-channel product, or a materialized ``a`` the fill must convert. The ONE predicate
    every fill dispatch reads (tile legality, stage enumeration, the resolver, the split-K
    partial), so the four cannot drift."""
    return _requires_sync_fill(node) or (plan.is_warp and _converting_a(node, plan.atom, term.tile.inputs))


def _placed(term: _Term, node, plan: TilePlan) -> TilePlan:
    """Bind a candidate plan to the physical axes of its stored Fold site."""
    stored = node.fold if isinstance(node, _FoldInterface) else node
    return term.sched.placed(stored, plan)


def _tile_ok(term: _Term, node, plan: TilePlan) -> bool:
    """Whether a warp tile candidate is realizable on ``node`` — the K-step divisibility every warp
    row needs, plus the exact-cover geometry the smem compute fill adds. Both are ``_legality``
    predicates, dropped here and RAISED on a pin (:func:`_contraction_values`)."""
    placed = _placed(term, node, plan)
    if placed.axes is None:
        return False
    if not legal.enforce(legal.warp_atom_target(plan.atom, term.ctx), pinned=False):
        return False
    shapes = {**term.tile.inputs, **term.tile.outputs}
    if not legal.enforce(legal.warp_split_store(projection_tail(term.tile), term.place.free, plan.atom.shape, shapes), pinned=False):
        return False
    conv = _converting_a(node, plan.atom, term.tile.inputs)
    # The converting fill reads A per element through its own σ — the fragment loader's contiguous
    # K-column requirement is a gmem-direct/byte-transport fact and does not apply to it.
    if not conv and not legal.enforce(legal.warp_a_columns(node, plan, term.tile.inputs), pinned=False):
        return False
    if not legal.enforce(legal.warp_k_step(node, plan), pinned=False):
        return False
    if not _requires_sync_fill(node) and not conv:
        return True
    return legal.enforce(legal.computed_operand_cover(node, placed, converting_a=conv), pinned=False) and legal.enforce(
        legal.computed_operand_copy_dtype(node, placed, term.tile.inputs, converting_a=conv), pinned=False
    )


def _f16acc_allowed(ctx) -> bool:  # noqa: ARG001 — ctx kept for call-site symmetry with the other precision policies
    """Whether the f16-accumulate atom forks may be OFFERED — a precision-trading policy, off by
    default: the precise ``F16_MMA_F32_ACC`` pin is authoritative on every target; unset, the
    ``FAST_MATH`` umbrella offers the family everywhere it is legal, and tuning evidence or the
    prior ranks it against the f32-accumulate siblings per shape and card. A ``TILE`` pin naming
    the atom bypasses this policy entirely (pins are authoritative)."""
    from emmy.compiler.pipeline.search.space import F16_MMA_F32_ACC, precision_pin  # noqa: PLC0415

    raw = F16_MMA_F32_ACC.raw()
    if raw is not None:
        return F16_MMA_F32_ACC.parse(raw)
    return precision_pin(F16_MMA_F32_ACC)


def _f8_mma_allowed(ctx) -> bool:
    """Whether the native fp8 mma atom forks may be OFFERED — a precision-trading gate, off by
    default: the instruction's effective accumulation precision is arch-dependent (reduced on
    sm_89, ~3e-4 rel vs the exact f32 decode-and-fma scalar path; true-f32 on sm_120), so the
    precise ``FP8_MMA`` pin — or the ``FAST_MATH`` umbrella — must offer it. The sm_89 hardware
    floor is absolute: below it the bare ``mma...e4m3`` form does not compile, and no pin
    overrides that. A ``TILE`` pin naming the atom bypasses this gate (pins are authoritative)."""
    from emmy.compiler.pipeline.search.space import FP8_MMA, precision_pin  # noqa: PLC0415

    if not ctx.has_fp8_mma:
        return False
    return bool(precision_pin(FP8_MMA))


def _a_dtype(term: _Term, node):
    """The ``a`` edge's produced element dtype — the value stored to or read by the A slab."""
    return edge_dtypes(node.a, term.tile.inputs, term._dtypes)[0]


def _channel_dtype(term: _Term, node):
    """The unambiguous tensor-core dtype supplied by the B channels, if any."""
    dts = {edge_dtypes(ch.b, term.tile.inputs, term._dtypes)[0] for ch in node.channels}
    if len(dts) == 1:
        return next(iter(dts))
    eligible = {dtype for dtype in dts if dtype is not None and atoms_for(dtype, ctx=term.ctx)}
    return next(iter(eligible)) if len(eligible) == 1 else None


def _warp_atoms(term: _Term, node) -> tuple[str, ...]:
    """The dtype-eligible tensor-core atom names for this contraction, ``()`` when the warp tier
    doesn't apply (a non-16-bit operand dtype, a fragment-unrealizable gather epilogue), extended
    with the f16-accumulate siblings when :func:`_f16acc_allowed`. Reads pure algebra off the STORED
    node — the placement / tile would be unread.

    An ``a`` edge whose value is f32 — a computed cone's leaf or a plain materialized load (the
    model's own erased ``.float()`` rounding) — still rides the CHANNELS' 16-bit atom: the smem
    compute fill converts on the slab store, and stage resolution mandates the fill for exactly
    these edges (:func:`_needs_fill`).

    This is the CHOICE half of the dtype rule; a ``TILE`` pin bypasses the choice layer by design, so
    it re-asks the same question as a CHECK (``_legality.warp_operand_dtype``)."""
    # The mma atom realizes ONLY the (·, +) semiring instance — the bilinear reading is
    # semiring-generic (``Fold.semiring``), so any other registered instance takes the
    # scalar / reduce tiers rather than silently reaching a tensor core that sums products.
    ring = node.semiring
    if ring is None or tuple(o.name for o in ring) != ("multiply", "add"):
        return ()
    inputs = term.tile.inputs
    # Boundary stores are outside the algebraic term. Reconstitute them before asking whether
    # the projection is a straight-line fragment epilogue, or a swept stack tail reaches RegStore.
    if not inputs or legal.fragment_epilogue(Body(tuple(projection_tail(term.tile)))) is not None:
        return ()
    ab = _a_dtype(term, node)
    if ab is not None and ab.nbytes == 1:
        # The native fp8 tier (M3): offered only under the precision gate, on a MATERIALIZED f8
        # ``a`` whose channels all carry the SAME f8 dtype (the byte-gather loaders move raw
        # bits — a mismatched operand would be read at the wrong width) and a static K (no
        # masked-K byte gather). Outside that, an f8 ``a`` has no warp tier at all: the sync
        # compute fill would DEMOTE the cone's value on a 1-byte slab store. The STRUCTURAL
        # requirements hold under any pin; the precision gate alone is bypassed by a ``TILE``
        # pin naming an f8 atom (pins are authoritative, the ``_f16acc_allowed`` convention —
        # the pin must also open the WORK inventories this site spells against).
        atoms = atoms_for(ab, ctx=term.ctx)
        pin = term.pin("TILE", node)
        ok = (
            isinstance(node.a, Load)
            and _channel_dtype(term, node) == ab
            and node.axis.extent.is_static
            and (_f8_mma_allowed(term.ctx) or (pin is not None and any(a in pin for a in atoms)))
        )
        return atoms if ok else ()
    if not atoms_for(ab, ctx=term.ctx):
        ab = _channel_dtype(term, node)  # the demoting compute fill — an f32 a (cone leaf or plain load) on 16-bit B
        if ab is not None and ab.nbytes == 1:
            return ()  # an f8 channel under a demoting fill stays off the warp tier (the fill would demote to f8)
    atoms = atoms_for(ab, ctx=term.ctx)
    if not atoms or not _f16acc_allowed(term.ctx):
        return atoms
    return atoms + atoms_for(ab, acc=ab, ctx=term.ctx)  # the f16-accumulate siblings, registry order preserved


# --- the pointwise cell: the register strip ---


def _strip_width(plan: TilePlan) -> int:
    """The strip ratio ``r`` a strip row's ``TILE`` names — the inner register width. A warp codec
    names none (there is no fragment on a pointwise cell), so it reads ``0`` and is dropped."""
    return 0 if plan.is_warp else plan.reg_n


def _strip_blocks(term: _Term, node) -> list[Block]:
    """The register-strip values: the flat per-cell tile and every ladder width the cell can carry.
    ``r`` IS the spelled ``TILE=f<r>`` — the strip is a TERM VARIANT applied at materialization, a
    function of the ROW, not a member of a pre-enumerated variant set."""
    pin = term.pin("TILE", node)
    ext = term.place.free[-1].extent.as_static() if _strippable(term) else 0
    try:
        plans = [resolve_site_tile(pin, None)] if pin is not None else [TilePlan(), *map_tile_moves()]
    except ValueError as e:
        # A pin the strip site cannot SPELL — a warp atom, which needs an inventory a pointwise
        # cell never has. Same rule as everywhere: the candidate is simply not in
        # ``values(site, work)``, so the cell degrades to the flat per-cell tile. This is PIN BLEED
        # (one env pin, several kernels in the graph, and this is not the one it was written for),
        # which is why it degrades rather than emptying the fork; ``_enumerate`` still raises the
        # recorded error if NOTHING in the term could spell it.
        term.pin_error = e
        plans = []
    out = []
    for plan in plans:
        # A strip WIDTH the cell cannot carry (a stateful / sweep body, a symbolic or indivisible
        # inner extent, a warp codec on a pointwise cell) drops the row; the flat per-cell base
        # below is always offered, so a narrowing pin degrades to it.
        if legal.enforce(legal.strip_width(ext, _strip_width(plan)), pinned=False):
            out.append(Block({"TILE": plan}, (None,)))
    return out or [Block({"TILE": TilePlan()}, (None,))]


# --- the reduce partition ---


def _splittable_axis(term: _Term, node) -> bool:
    """Whether a cross-CTA partition may still be offered here — false once THIS KERNEL already
    carries one.

    A cross-CTA split is consumed by the rewrite that realizes it. The pieces are brand-new
    kernels reaching this enumeration with no knobs of their own, so nothing but the IR records
    that the partition happened — and what records it is a reduce axis that is already a slice of
    a parent (``_factor_k`` / ``_slice_loop`` build it as a ``Window``). No provenance flag: the
    axis's own shape is the receipt.

    The scope is the KERNEL, not the axis, because that is the scope of the decision being
    consumed. ``REDUCE`` is one pin and a bare one fans out to every eligible site
    (``_Term.pin``), so reading the receipt per-axis lets the same pin fire again on a DIFFERENT
    reduce axis of the piece: a fused cone's partial still holds the cone's per-row statistic
    fold, never sliced, and ``g4k`` split that too — a third kernel from one pinned split. Reading
    it per kernel is what makes ``g4k`` mean one split.

    Per-axis alone also never terminates on its own axis: a pinned split re-applies to its own
    partial, halving the extent every sweep until it runs out (found on a K=64 matmul:
    64 → 32 → … → 1)."""
    if node.axis is None:
        return False
    return not term.carries_partition()


def _carries_partition(op) -> bool:
    """Whether this kernel's IR already records a realized cross-CTA split — the ``Window``
    receipt :func:`_splittable_axis` reads.

    The receipt sits on the sliced axis, and that axis is not always a NODE: on a computed-A cone
    a computed-A cone can keep its sliced contraction inside the lift as a plain ``Loop``, so a
    ``sites``-only scan misses it and the ambient pin splits the piece a second time — the
    statistic fold, which no partition ever touched (measured on the gate⊗up twin: three kernels
    from one ``g4k``, the doubly-split partial off the mma tier). Scan the loop bodies too; the
    receipt is in the IR either way."""

    def loops(stmts):
        for s in stmts:
            if isinstance(s, Loop):
                yield s
                yield from loops(s.body)

    for site in sites(op):
        node = site.node
        ax = getattr(node, "axis", None)
        if ax is not None and ax.window is not None and ax.window.partition:
            return True
        bodies = [node.body, *([node.lift.body] if getattr(node, "lift", None) is not None else [])]
        if any(lp.axis.window is not None and lp.axis.window.partition for b in bodies for lp in loops(b)):
            return True
    return False


def _consumed_split(term: _Term, node, plan: ReducePlan) -> ReducePlan:
    """``plan`` with its cross-CTA stage dropped when ``node``'s axis is already a slice — the pin
    half of :func:`_splittable_axis`. A pin is a statement about a kernel's schedule; the split it
    asked for was realized on the kernel that was split, so what reaches the pieces is the rest of
    the row (``g2k/coop`` on a sliced axis is ``coop``)."""
    if not plan.needs_split or _splittable_axis(term, node):
        return plan
    return ReducePlan(tuple(st for st in plan.stages if st.level is not _ReduceLevel.GRID))


def _reduce_specs(term: _Term, node) -> list[ReducePlan]:
    """Every reduce partition a non-contraction fold can legally spell: the serial fold, plus each
    :func:`coop_reduce_moves` entry this node admits. No candidate is preferred, promoted or
    dropped for speed — the catalog is filtered by LEGALITY
    alone (the band's geometry, its epilogue, and a width the reduce extent can actually feed), so
    the 16- / 32-wide reduce goldens and the wide normalizer bands are all reachable and the
    evidence hierarchy ranks them. An env pin is authoritative — minus any cross-CTA stage this
    axis already consumed (:func:`_consumed_split`).

    A cross-CTA split is offered here only in COMPOSITE with the transposed band — every split
    candidate in the catalog is one, so the loop below states the catalog's shape rather than
    adding a rule."""
    inner = _inner_free(term.place)
    k_static = node.axis.extent.as_static() if node.axis.extent.is_static else None
    # Term-wide, so it is asked ONCE, not per candidate.
    epilogue = legal.coop_band_epilogue(projection_tail(term.tile))

    def band_legal(p: ReducePlan, *, pinned: bool) -> bool:
        # The transposed band's own requirements: the geometry (shared with the contraction tier)
        # plus this tier's epilogue condition. A pin meets the same test, as a refusal.
        if not p.coop_transposed:
            return True
        return legal.enforce(legal.coop_band_geometry(p, k_static, inner), pinned=pinned) and legal.enforce(epilogue, pinned=pinned)

    atomic = legal.atomic_finalize(tuple(node.combine.results), projection_tail(term.tile), term.tile.outputs)

    pin = term.pin("REDUCE", node)
    if pin is not None:
        plan = _consumed_split(term, node, ReducePlan.parse(pin, Workers.parse(WORK.raw())))
        band_legal(plan, pinned=True)
        if plan.needs_split and plan.finalize == "atomic":
            # The SAME gate the catalog applies below. A pin that skipped it would reach
            # ``030_split_reduce`` and crash the emitter instead of reporting a refusal.
            legal.enforce(atomic, pinned=True)
        return [plan]
    extent = hint_extent(node.axis)
    cands = [ReducePlan()]
    for p in coop_reduce_moves():
        if p.needs_split and not _splittable_axis(term, node):
            continue  # the axis is already a slice — its cross-CTA partition was consumed
        if not band_legal(p, pinned=False):
            continue
        if p.finalize == "atomic" and not legal.enforce(atomic, pinned=False):
            continue
        if p.coop <= extent and p.reg <= extent and p not in cands:
            cands.append(p)
    return cands


def _reduce_blocks(term: _Term, node) -> list[Block]:
    """The reduce-partition values a non-contraction fold offers: the partition itself plus the
    shared-row ``STAGE`` a cooperative band can drive (a resolver, not a choice — see
    :func:`_row_stage`). Each block is a rectangle ONE stage deep: the transport here is a function
    of the partition, never a free axis beside it. Which of them SPELL against the kernel's chosen
    inventory is the ROW's question, not this site's (:func:`_block_rows` reads the claim off the
    block) — a serial fold claims no workers at all, so at a NESTED site it composes with any
    parent inventory."""
    out = []
    for plan in _reduce_specs(term, node):
        shared = _row_stage(term, node) if plan.coop > 1 else None
        out.append(Block({"REDUCE": plan}, (None, shared) if shared is not None else (None,)))
    return out


# --- the contraction: tile x stage x reduce ---


def _resolve_stage(
    term: _Term,
    node,
    tile: TilePlan,
    want: Stage | None,
    why: list[str] | None = None,
    *,
    k_axis: Axis | None = None,
    seam: tuple | None = None,
) -> Stage | None:
    """The ONE transport-resolver dispatch — which operand edges and tier select.

    Any COMPUTED contraction operand and every multi-channel product take the smem compute fill,
    which is MANDATORY (no byte transport can evaluate a cone or carry several B/C channels), so
    ``want=None`` still resolves and only the DEPTH is ever free. A single-channel, fully
    MATERIALIZED contraction takes the mma resolver on a warp tile and the scalar one otherwise,
    with ``want=None`` the gmem-direct baseline; TMA declines below sm_90 rather than failing to
    compile. ``tile`` is the PLACED slice.

    Enumeration, the split-K composition and re-materialization all reach the resolvers through
    here, so a row's resolved spelling is reproducible BY CONSTRUCTION rather than by three copies
    of the dispatch staying in step."""
    budget = term.ctx.max_dynamic_smem
    if want is not None and legal.stage_target(want, term.ctx) is not None:
        return None
    if _needs_fill(term, node, tile):
        # A computed edge, a multi-channel product, or a converting materialized edge takes only
        # the ``smem`` compute fill — a want naming an asynchronous byte transport declines rather
        # than silently resolving to the fill. The fill IS asynchronous on its B slabs; that ring
        # is its own depth 2, so the decline names that spelling instead of leaving the caller
        # hunting a smem budget.
        if want is not None and want.transport != "smem":
            legal.decline(
                why,
                f"the smem compute fill has no {want.transport} sibling: a computed operand cannot ride a byte "
                f"transport, and the fill's own asynchronous B-slab prefetch ring is spelled d2/smem",
            )
            return None
        return legal.resolve_fill_stage(
            node,
            tile,
            budget,
            want.depth if want is not None else 1,
            inputs=term.tile.inputs,
            why=why,
            seam=term.seam(node) if seam is None else seam,
            k_axis=k_axis,
            producer=term.producer(node),
        )
    if want is None or (want.transport == "smem-tma" and not term.ctx.has_tma):
        return None
    if tile.is_warp:
        return legal.resolve_warp_stage(node, tile, want, budget, term.tile.inputs)
    return legal.resolve_scalar_stage(node, tile, want, term.tile.inputs, budget)


def _resolved(moves, resolve, *, gmem_direct: bool = True) -> list[Stage | None]:
    """``moves`` resolved against the term and deduped on the RESOLVED spelling — the shape every
    stage-value site shares. Dedupe is on the resolved spelling, never the catalog move: a depth
    that clamps under the smem budget spells identically to its shallower sibling and must yield
    ONE row, or the fork carries two leaves naming one kernel.

    ``gmem_direct`` seeds the no-intermediate candidate (``None``, no slab) — a legality fact about
    the tier, not a preference. The compute-fill tier has no gmem-direct sibling — a computed ``a``
    edge must land somewhere — so it seeds nothing, and a caller that declines every move returns
    the empty list rather than a silent fallback."""
    out: list[Stage | None] = [None] if gmem_direct else []
    spelled = {""} if gmem_direct else set()
    for move in moves:
        r = resolve(move)
        if r is not None and r.spell() not in spelled:
            spelled.add(r.spell())
            out.append(r)
    return out


def _fill_values(
    term: _Term,
    node,
    tile: TilePlan,
    *,
    k_axis: Axis | None = None,
    seam: tuple | None = None,
) -> list[Stage | None]:
    """The RESOLVED compute-fill stages a computed operand or multi-channel product offers — its
    depths, and nothing else: the fill is MANDATORY (there is no gmem-direct ``None`` sibling and
    no byte transport can evaluate a cone or carry several B/C channels), so a ``STAGE`` pin can
    only choose the depth. ``d1`` and the asynchronous-peer prefetch ring ``d2`` are fork siblings
    — the ring is measured per shape (see
    :func:`_legality.resolve_fill_stage`) — and a ``d2`` that clamps back to ``d1`` under the smem
    budget spells identically, so it dedupes to one row."""
    pin = term.pin("STAGE", node)
    if pin:
        # A pinned spelling names a kernel, so its TRANSPORT cannot be quietly dropped and read as
        # depth alone: the fill has no byte-transport sibling, and its own asynchronous B-slab
        # prefetch ring is the depth-2 ``smem`` row, not ``smem-async``.
        pinned_stage = Stage.parse(pin)
        if pinned_stage.transport != "smem":
            legal.enforce(
                f"the smem compute fill has no {pinned_stage.transport} sibling: a computed operand cannot ride a "
                f"byte transport (nothing but the fill can evaluate a producer cone). Its own asynchronous B-slab "
                f"prefetch ring is spelled d2/smem.",
                pinned=True,
            )
    depths = [Stage.parse(pin).depth] if pin else [1, 2]

    def resolve(st: Stage) -> Stage | None:
        why: list[str] = []
        r = _resolve_stage(term, node, tile, st, why=why, k_axis=k_axis, seam=seam)
        if r is None:  # per DECLINED depth, so a pin that fits no depth names the gate it hit
            legal.enforce(
                f"the smem compute fill does not resolve at depth {st.depth}: "
                + (why[-1] if why else f"its slabs must fit the {term.ctx.max_dynamic_smem} B smem budget"),
                pinned=pin is not None,
            )
        return r

    return _resolved((Stage(depth=d) for d in depths), resolve, gmem_direct=False)


def _stage_values(term: _Term, node, plan: TilePlan) -> list[Stage | None]:
    """The RESOLVED operand stages for one tile candidate — gmem-direct ``None`` first, then every
    catalog move that RESOLVES against the node with this ``plan``, so the leaf identity, the
    stamped knobs and the kernel agree. A pinned ``STAGE`` is authoritative: the resolved pin alone,
    or gmem-direct when it declines."""
    if not plan.is_tiled:
        return [None]  # per-cell / unbindable — no operand slab to stage
    if not plan.is_warp and _has_computed_operand(node):
        # The scalar atom evaluates inline cones directly in its register row/column reads. The
        # warp-only compute fill is unnecessary here, and byte transports cannot evaluate it.
        return [None]
    tile = _placed(term, node, plan)
    if plan.is_warp and _needs_fill(term, node, tile):
        return _fill_values(term, node, tile)

    def resolve(st: Stage) -> Stage | None:
        return _resolve_stage(term, node, tile, st)

    pinned = term.pin("STAGE", node)
    if pinned is not None:
        # A malformed pin RAISES through ``Stage.parse`` — this used to be swallowed into
        # gmem-direct, which made it the only silently-ignored pin in the family.
        if not pinned:
            return [None]
        wanted = Stage.parse(pinned)
        # SM70 pins are strict: do not silently turn a newer copy instruction into the
        # gmem-direct sibling.
        if term.ctx.compute_capability < (8, 0):
            legal.enforce(legal.stage_target(wanted, term.ctx), pinned=True)
        return [resolve(wanted)]
    return _resolved(stage_moves(warp=plan.is_warp), resolve)


def _contraction_reduces(term: _Term, node, plan: TilePlan) -> list[ReducePlan]:
    """The contraction's ``REDUCE`` candidates — the serial fold, the legal coop / ILP moves
    (per-cell tier only — the non-output-tiled contract) and the divisor-legal split-K
    moves. An ATOMIC split is offered only on a single-channel node whose FULL
    projection tail distributes over the add; the deferred kernel finalize stays legal for any
    epilogue."""
    pin = term.pin("REDUCE", node)
    if pin is not None:
        pinned = _consumed_split(term, node, ReducePlan.parse(pin, Workers.parse(WORK.raw())))
        ext = node.axis.extent
        # A pin meets the transposed band's geometry as a refusal, not an emitter crash.
        legal.enforce(legal.coop_band_geometry(pinned, ext.as_static() if ext.is_static else None, _inner_free(term.place)), pinned=True)
        if pinned.needs_split:
            if pinned.finalize == "atomic":
                # The SAME gate ``atomic_ok`` applies to the catalog below — a pin refuses here
                # rather than reaching ``030_split_reduce`` with a carrier it cannot realize.
                atomic = legal.atomic_finalize(tuple(node.combine.results), projection_tail(term.tile), term.tile.outputs)
                legal.enforce(atomic, pinned=True)
            return [pinned]
        if pinned.coop > 1 or pinned.reg > 1:
            # A tiled candidate contracts K serially per register cell — the coop / ILP partition is
            # the NON-output-tiled tier's, so a tiled tile has nothing to honor the pin with.
            return [] if plan.is_tiled else [pinned]
        return [ReducePlan()]
    out = [ReducePlan()]
    ext = node.axis.extent
    k = ext.as_static() if ext.is_static else None
    # A cross-CTA split factors a STATIC K; either edge's σ-reindex then rides ``_sliced_edge``
    # (a gmem index, or a computed cone's own k coordinate).
    splittable = k is not None
    if k is not None and not plan.is_tiled:
        inner = _inner_free(term.place)
        for p in coop_reduce_moves():
            if not (p.coop <= k and p.reg <= k):
                continue
            if p.needs_split and not splittable:
                continue
            # The transposed lane swap also needs the structure its emitter assumes — the SAME
            # geometry the reduce tier requires, stated once in ``_legality``.
            if not legal.enforce(legal.coop_band_geometry(p, k, inner), pinned=False):
                continue
            out.append(p)
    if splittable and _splittable_axis(term, node) and len(term.place.free) >= 2:
        step = plan.atom.atom_k * plan.bk if plan.is_warp else 1
        atomic_ok = legal.enforce(
            legal.atomic_finalize(tuple(node.combine.results), projection_tail(term.tile), term.tile.outputs), pinned=False
        )
        for sp in splitk_moves():
            if sp.finalize == "atomic" and not atomic_ok:
                continue  # the carrier, projection, or destination cannot realize a direct atomic finalize
            if k % sp.cta == 0 and (k // sp.cta) % step == 0:
                out.append(sp)
    return out


def _pinned_tile_legal(term: _Term, node, plan: TilePlan) -> bool:
    """Whether the pinned ``TILE`` resolved to ``plan`` may stand — the same predicates the unpinned
    catalog drops on, RAISED here because a pin names a kernel the user asked for."""
    if plan.is_warp:
        # A PIN with an indivisible K-step or a gather epilogue RAISES — the same predicates
        # the unpinned catalog drops on, one home each.
        legal.enforce(legal.warp_atom_target(plan.atom, term.ctx), pinned=True)
        conv = _converting_a(node, plan.atom, term.tile.inputs)
        if not conv:
            legal.enforce(legal.warp_a_columns(node, plan, term.tile.inputs), pinned=True)
        legal.enforce(legal.warp_k_step(node, plan), pinned=True)
        legal.enforce(legal.fragment_epilogue(term.proj), pinned=True)
        shapes = {**term.tile.inputs, **term.tile.outputs}
        legal.enforce(legal.warp_split_store(projection_tail(term.tile), term.place.free, plan.atom.shape, shapes), pinned=True)
        if _requires_sync_fill(node) or conv:
            placed = _placed(term, node, plan)
            legal.enforce(legal.computed_operand_cover(node, placed, converting_a=conv), pinned=True)
            legal.enforce(legal.computed_operand_copy_dtype(node, placed, term.tile.inputs, converting_a=conv), pinned=True)
            return True
        # Fully materialized contractions use the ordinary operand-dtype rule. Inline-edge
        # contractions were checked above by the sync copy-dtype rule, which also tolerates
        # scheduler-only fixtures that carry no Tensor metadata.
        return legal.enforce(legal.warp_operand_dtype(node, plan, _a_dtype(term, node)), pinned=False)
    if not _supports_scalar(node):
        return False  # the scalar emitter carries one accumulator channel
    # The CTA thread budget, raised HERE rather than left to materialization: a pinned tile the
    # hardware cannot launch is a user error, and `Pipeline.run`'s validity retry would otherwise
    # catch the materializer's raise and quietly deploy the next leaf — the pin says yes, the
    # deploy says something else.
    legal.enforce(legal.scalar_block_threads(plan), pinned=True)
    return True


def _pinned_tiles(term: _Term, node, pin: str) -> list[TilePlan]:
    """The pinned ``TILE`` resolved against every inventory this SITE can spell it against.

    A ``TILE`` value's unit widths are read OFF ``WORK`` (:meth:`TilePlan.parse`), so one pin names
    a different plan under each inventory: the pin is authoritative over the VALUE, never over the
    inventory. A pinned ``WORK`` therefore leaves exactly ONE candidate, and it is authoritative
    whether or not a catalog implies it — the widths it fixes are what no catalog can predict, and
    resolving the tile pin against inventories the user excluded would make "this pin is illegal" a
    statement about a kernel nobody asked for. Unpinned, the candidates are the site's own."""
    raw = WORK.raw()
    # The site's catalog is asked whether or not the pin narrows the candidates: it is what answers
    # "were tensor cores on offer here" (``_Term.warp_eligible``), a structural fact about the
    # KERNEL that a pin must not be able to erase from the rows it does enumerate.
    catalog = term.tiles(node)
    spells = [raw] if raw is not None else [*catalog]
    reduce_pin = term.pin("REDUCE", node) or ""
    out: list[TilePlan] = []
    for spell in spells:
        work = Workers.parse(spell) if spell else None
        try:
            plan = resolve_site_tile(pin, work, ReducePlan.parse(reduce_pin, work).coop)
            if plan in out:
                continue
            spellable = _pinned_tile_legal(term, node, plan)
        except ValueError as e:
            # The pin does not reach a kernel at THIS inventory — a warp atom needs a warp
            # ``WORK``, and a plan that resolves here can still be illegal here and legal at a
            # sibling. Either way the candidate is simply not in the site's catalog at this
            # inventory, the same rule every other value follows. A pin that reaches NO inventory
            # is a different failure: :func:`_enumerate` raises the recorded refusal rather than
            # quietly emptying the fork, so the message survives whichever inventory drew it.
            term.pin_error = e
            continue
        if spellable:
            term.pin_spelled = True
            out.append(plan)
    return out


def _contraction_blocks(term: _Term, node) -> list[Block]:
    """The contraction's values: the tile × stage × reduce legal product, over EITHER inhabitant of
    the ``a`` edge — a materialized ``Load`` (both tiers, every transport) or a COMPUTED cone
    (inline on scalar register tiles, or over the mandatory warp compute fill).

    The catalog is a function of the SITE alone. Which kernel inventory a rectangle can inhabit is
    the rectangle's OWN claim (:func:`_block_rows`), not an input here — ``_Term.tiles`` already
    groups the tile candidates by the inventory each implies, so asking per inventory would only
    re-resolve the same stages and reduces once per member of a list this catalog produces.

    The product is emitted as one BLOCK per ``(TILE, REDUCE)`` pair rather than one value per
    ``(TILE, REDUCE, STAGE)`` triple. Same catalog calls, same legality calls, same order — only
    the return SHAPE differs, and it is the shape that says what the walk already knew: the
    transport is a free axis over the pair, not a third coupled dimension. A pair whose every stage
    was refused is no block at all, exactly as it used to be no rows at all."""
    pin = term.pin("TILE", node)
    plans = _pinned_tiles(term, node, pin) if pin is not None else [p for group in term.tiles(node).values() for p in group]
    out = []
    for plan in plans:
        # The transport catalog is a function of ``(node, plan)`` — hoisted out of the reduce loop
        # it does not depend on, as the reduce catalog is out of the inventory it never took.
        values = _stage_values(term, node, plan)
        for red in _contraction_reduces(term, node, plan):
            stages = tuple(
                stage
                for stage in values
                if red.needs_split
                or legal.enforce(legal.paired_fragment_register_budget(node, term.producer(node), plan, stage), pinned=pin is not None)
            )
            if stages:
                out.append(Block({"TILE": plan, "REDUCE": red}, stages))
    return out


def _raster_values(term: _Term) -> list[str]:
    """The ``RASTER`` candidates — kernel-global, and CONTRACTION-scoped: only a 2-D-tiled
    contraction grid decodes the swizzle. A symbolic-axis (masked-tile) grid renders through the
    dynamic decode path, which does not carry it, so offering ``gm8`` there would stamp a launch
    order the kernel doesn't realize."""
    if not any(is_contraction(n.site.node) for n in term.tree):
        return [""]
    if any(not ax.extent.is_static for ax in term.place.free):
        return [""]
    return list(RASTER.narrow(raster_moves()))


def _site_blocks(term: _Term, current: _Node, parent: _Node | None = None) -> list[Block]:
    """The values ``site`` offers, as :class:`Block` rectangles — TYPED schedule slices, keyed by
    family. Dispatch is the two stored-param predicates on the node, never the ``AxisRole``.

    The site-tree context travels with it: ``parent`` selects the reduction domain a derived child
    presents to its enclosing Fold. The kernel inventory does NOT travel with it — a site offers
    what it can spell, and :func:`_block_rows` reads each rectangle's claim off its own slices."""
    node = current.site.node
    if node.axis is None:
        return _strip_blocks(term, node)
    if is_contraction(node):
        return _contraction_blocks(term, term.scheduled_node(current, parent))
    return _reduce_blocks(term, node)


# ---- the recursion: one row is a joint assignment across the site tree --------------------------- #


def _spell(value) -> str:
    """A slice's stored spelling — ``""`` is the DECIDED empty (the per-cell tile, the serial fold,
    gmem-direct), never an absent key."""
    return value.spell() if value is not None else ""


@dataclass(frozen=True)
class _Row:
    """One enumerated row — its spelled knob dict and worker/physical interfaces.

    The ``TILE`` slices are kept BY KEY, not as a flat tuple: reading a site's slice back out of a
    flattened list by position is how a two-site term silently swaps its sites. No OTHER family's
    resolved slice is carried — the row is the kernel's complete identity and :func:`_materialize`
    re-resolves every slice from its spelling, so a second copy could only ever disagree."""

    knobs: dict
    plans: dict = field(default_factory=dict)
    stages: dict = field(default_factory=dict)
    #: The worker inventory THIS row's own slices claim — :func:`derive_inventory` folded over them
    #: once, where the row is built. ``None`` claims nothing and composes with any inventory, which
    #: is what lets a serial fold sit under a warp parent. The claim is what makes ``WORK`` a JOIN
    #: KEY over the catalogs rather than an input to them: no row is enumerated per inventory.
    work: Workers | None = None

    @property
    def width(self) -> int:
        """One fully spelled row occupies one address in a pool segment."""
        return 1

    @classmethod
    def union(cls, parts: Iterable[_Row]) -> _Row | None:
        """Several rows as one, or ``None`` when their worker claims conflict.

        The claim is a CONSISTENCY, not a maximum: two sites claiming DIFFERENT inventories name
        kernels the wire format cannot tell apart, so they are not one row. A part claiming
        ``None`` composes with any other. This is the SAME rule :func:`_merge_interfaces` applies
        to the interface, one level up — it is stated once, in :func:`derive_inventory`, and
        applied here to the already-derived claims."""
        knobs: dict = {}
        plans: dict = {}
        stages: dict = {}
        work: Workers | None = None
        for part in parts:
            knobs.update(part.knobs)
            plans.update(part.plans)
            stages.update(part.stages)
            if part.work is not None:
                if work is not None and part.work != work:
                    return None  # two sites, two inventories, one WORK entry to spell them in
                work = part.work
        return cls(knobs=knobs, plans=plans, stages=stages, work=work)


def _producer_bands(work: Workers | None, stage: Stage | None) -> tuple[Workers, ...]:
    """The producer-band inventories a row ALSO claims.

    The band is kernel-global, but every condition on it is a fact about the ROW: it drives a
    resolved TMA stage (:func:`legal.producer_transport`) and needs a warp inventory wide enough to
    spare it (:func:`legal.producer_band`). Claiming it here is what lets the join build the band's
    segment out of the rows that can inhabit it, instead of re-enumerating the whole term once per
    band — and it makes the old term-wide "no band beside a synchronous compute fill" gate fall
    out: a fill stage is not TMA, so a fill site claims no band and the join finds no partner."""
    if work is None or work.kind != "warp" or legal.producer_transport(stage) is not None:
        return ()
    return tuple(
        replace(work, producer=band) for band in (1, 2) if legal.enforce(legal.producer_band(WarpSpec(band), work.count * 32), pinned=False)
    )


def _block_rows(node: _Node, block: Block) -> list[_Row]:
    """Spell one local block into complete per-stage rows at its canonical site keys, each carrying
    the worker inventory its own slices claim."""
    red = block.values.get("REDUCE")
    tile = block.values.get("TILE")
    try:
        work = derive_inventory((tile,) if tile is not None else (), coop=red.coop if red is not None else 1)
    except ValueError:
        return []  # this block's own TILE and REDUCE name two inventories — no kernel spells it
    key = node.keys.get("STAGE")
    tile_key = node.keys.get("TILE")
    base = _Row(
        knobs={k: _spell(block.values.get(family)) for family, k in node.keys.items() if family != "STAGE"},
        plans={tile_key: tile} if tile is not None and tile_key is not None else {},
        work=work,
    )
    out: list[_Row] = []
    for stage in block.stages:
        row = replace(
            base,
            knobs={**base.knobs, **({key: _spell(stage)} if key is not None else {})},
            stages={tile_key: stage} if tile_key is not None else {},
        )
        out.append(row)
        out.extend(replace(row, work=band) for band in _producer_bands(work, stage))
    return out


@dataclass(frozen=True)
class _ComboSpace(Sequence[tuple[int, ...]]):
    """Compatible signature-group products, concatenated as an addressable sequence."""

    groups: tuple[tuple[Sequence[int], ...], ...]
    interfaces: tuple[tuple, ...]
    offsets: tuple[int, ...]

    @classmethod
    def build(cls, term: _Term, parts: tuple[Sequence[_Row], ...]) -> _ComboSpace:
        """The compatible products of the sites' interfaces, one pass per worker inventory.

        The inventory RESTRICTS THE ROWS, it does not key the interface. That distinction is the
        whole tractability of this product: a term can carry hundreds of sites, and the product
        runs over each site's interface GROUPS, so one extra group per site multiplies the whole
        enumeration. Under a chosen inventory a site offers the rows claiming it plus the rows
        claiming nothing, which collapse to the few interfaces they always did; keying the group by
        the claim instead would split every site's catalog once per inventory it can spell and turn
        a 491-site term into an unenumerable product.

        A pass keeps a combination when the merged interface CLAIMS workers exactly when the pass's
        inventory does — which is what keeps the all-claims-nothing combination out of every
        concrete inventory's pass and in its own. The combination is then recorded against that
        inventory: the pass knows the claim, so nothing has to re-derive it."""
        if not parts:
            return cls(((),), ((None, (), (), ()),), (0, 1))
        buckets, claims = [], {}
        for rows in parts:
            per_claim: dict = {}
            for index in range(len(rows)):
                row = rows[index]
                per_claim.setdefault(row.work, {}).setdefault(_row_interface(term, row), []).append(index)
            buckets.append(per_claim)
            claims.update(dict.fromkeys(claim for claim in per_claim if claim is not None))
        products = []
        product_interfaces = []
        # ``None`` leads: the per-cell / pure-reduce geometry is the inventory a term always has.
        for claim in (None, *claims):
            grouped = []
            for per_claim in buckets:
                # Claiming rows lead, then the rows that claim nothing — the order the site's own
                # catalog puts them in, and the order the walk below reads a branch's first row in.
                # The two halves cannot collide: an interface records WHETHER its row claims
                # workers, so a claiming group and a claims-nothing group are never the same key.
                unclaimed = per_claim.get(None, {})
                offered = {**per_claim.get(claim, {}), **unclaimed} if claim is not None else unclaimed
                if not offered:
                    break  # this site cannot inhabit the inventory at all, so no row of it can
                grouped.append(tuple(offered.items()))
            if len(grouped) != len(buckets):
                continue
            size = 1
            for groups in grouped:
                size *= len(groups)
                if size > MAX_COMBINATIONS:
                    raise ValueError(
                        f"schedule enumeration is intractable: this term's {len(parts)} sites compose more than "
                        f"{MAX_COMBINATIONS} compatible interface combinations at WORK={_spell(claim)!r} — the "
                        f"composition is exponential in the number of independently scheduled sites"
                    )
            for choices in product(*grouped):
                interfaces, indices = zip(*choices, strict=True)
                interface = _merge_interfaces(term, interfaces)
                if interface is None or interface[0] != (claim is not None):
                    continue
                products.append(indices)
                product_interfaces.append((claim, *interface[1:]))
        sizes = [_product_size(group) for group in products]
        return cls(tuple(products), tuple(product_interfaces), (0, *accumulate(sizes)))

    def __len__(self) -> int:
        return self.offsets[-1]

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        size = len(self)
        if index < 0:
            index += size
        if not 0 <= index < size:
            raise IndexError(index)
        group_index = bisect_right(self.offsets, index) - 1
        local = index - self.offsets[group_index]
        group = self.groups[group_index]
        result = [0] * len(group)
        for i in range(len(group) - 1, -1, -1):
            local, digit = divmod(local, len(group[i]))
            result[i] = group[i][digit]
        return tuple(result)


def _product_size(parts) -> int:
    size = 1
    for part in parts:
        size *= len(part)
    return size


def _row_interface(term: _Term, row: _Row) -> tuple:
    axes = {}
    for key, plan in row.plans.items():
        node = term.tile_nodes.get(key)
        if node is None or not plan.is_tiled:
            continue
        placed = term.sched.placed(node, plan)
        if placed.axes is not None:
            axes.update((side.axis.name, (side.tile, side.units)) for side in placed.mn)
    # WHETHER the row claims workers, never WHICH inventory: the inventory restricts which rows a
    # pass offers (:meth:`_ComboSpace.build`), and keying it here would split every site's catalog
    # once per inventory it can spell — one more group per site, multiplied over every site.
    return row.work is not None, tuple(sorted(axes.items())), tuple(sorted(row.plans.items())), tuple(sorted(row.stages.items()))


def _merge_interfaces(term: _Term, interfaces) -> tuple | None:
    """The one interface several sites present together, or ``None`` when they cannot compose.

    Two agreements are checked here — the physical tile axes and the fragment seam. The third, the
    worker inventory, is structural: :meth:`_ComboSpace.build` offers a pass only the rows that
    claim its inventory or claim nothing, so no combination reaching here can name two. What comes
    out is whether the combination claims workers AT ALL, which is what tells the pass whether the
    combination belongs to it."""
    axes = {}
    plans = {}
    stages = {}
    claims_workers = False
    for claim, physical_axes, local_plans, local_stages in interfaces:
        claims_workers = claims_workers or claim
        plans.update(local_plans)
        stages.update(local_stages)
        for axis, value in physical_axes:
            previous = axes.setdefault(axis, value)
            if previous != value:
                return None

    for consumer_key, producer_key in term.fragment_edges:
        if consumer_key not in plans or producer_key not in plans:
            continue
        consumer, producer = plans[consumer_key], plans[producer_key]
        # An untiled producer is evaluated elementwise into the consumer's synchronous slab.  It
        # has no fragment interface to compose, so this is the ordinary inline/register residence.
        # A tiled producer, conversely, produces fragments and therefore requires a compatible
        # tiled consumer to receive them.
        if not producer.is_tiled:
            continue
        stage = stages.get(consumer_key)
        if not (consumer.is_warp and producer.is_warp and stage is not None and stage.transport == "smem"):
            return None
        if producer.atom.shape != consumer.atom.shape or producer.atom.fragment_layout != consumer.atom.fragment_layout:
            return None
        producer_node = term.tile_nodes[producer_key]
        placed = term.sched.placed(producer_node, producer)
        if placed.axes is None or placed.n.units != 1 or placed.n.tile != stage.bk_elems:
            return None
    return claims_workers, tuple(sorted(axes.items())), tuple(sorted(plans.items())), tuple(sorted(stages.items()))


@dataclass(frozen=True)
class _RowProduct(Sequence[_Row]):
    """A mixed-radix compatible product of schedule row spaces."""

    parts: tuple[Sequence[_Row], ...]
    combos: _ComboSpace
    closed = True

    @classmethod
    def build(cls, term: _Term, parts: tuple[Sequence[_Row], ...]) -> _RowProduct:
        return cls(tuple(parts), _ComboSpace.build(term, tuple(parts)))

    def __len__(self) -> int:
        return len(self.combos)

    def _regrouped(self, groups: dict[str, list[tuple]]) -> list[tuple[str, _RowProduct]]:
        """One ``{value: [(combo, interface)]}`` grouping rebuilt as addressable sub-products — the
        tail both partitions below share. A group keeps the PARTS and narrows only the combos, so
        the sub-product addresses a subset of this one without a row being visited."""
        out = []
        for value, entries in groups.items():
            combos, interfaces = zip(*entries, strict=True)
            sizes = [_product_size(group) for group in combos]
            out.append((value, _RowProduct(self.parts, _ComboSpace(tuple(combos), tuple(interfaces), (0, *accumulate(sizes))))))
        return out

    def work_groups(self) -> list[tuple[str, _RowProduct]]:
        """This product's rows grouped by the inventory they claim — the ``WORK`` segments.

        A regroup of the combo interfaces, which already carry the claim, so no Cartesian row is
        visited and no catalog is read again. This is the whole of what the per-inventory
        enumeration loop used to produce, one enumeration later."""
        groups: dict[str, list[tuple]] = {}
        for combo, interface in zip(self.combos.groups, self.combos.interfaces, strict=True):
            groups.setdefault(_spell(interface[0]), []).append((combo, interface))
        return self._regrouped(groups)

    def partition(self, key: str):
        """Partition this compatible product by one site-local schedule key.

        The key belongs to one leaf catalog. Restricting that catalog inside each compatibility
        group preserves the product symbolically; no Cartesian row is visited.
        """
        owners = [index for index, part in enumerate(self.parts) if any(key in row.knobs for row in part)]
        if not owners:
            return (("", self),)
        if len(owners) != 1:
            raise ValueError(f"schedule key {key!r} belongs to several row catalogs")
        owner = owners[0]
        part = self.parts[owner]
        groups: dict[str, list[tuple[tuple[Sequence[int], ...], tuple]]] = {}
        for combo, interface in zip(self.combos.groups, self.combos.interfaces, strict=True):
            by_value: dict[str, list[int]] = {}
            for index in combo[owner]:
                by_value.setdefault(part[index].knobs.get(key, ""), []).append(index)
            for value, indices in by_value.items():
                narrowed = (*combo[:owner], tuple(indices), *combo[owner + 1 :])
                groups.setdefault(value, []).append((narrowed, interface))
        return tuple(self._regrouped(groups))

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        size = len(self)
        if index < 0:
            index += size
        if not 0 <= index < size:
            raise IndexError(index)
        indices = self.combos[index]
        row = _Row.union(part[part_index] for part, part_index in zip(self.parts, indices, strict=True))
        if row is None:
            raise ValueError("schedule catalogs admitted incompatible worker claims")
        return row


def _site_catalogs(term: _Term, node: _Node, parent: _Node | None = None) -> Iterator[Sequence[_Row]]:
    """Every site's own row catalog in the subtree rooted at ``node``, children first.

    One catalog per SITE — not one per site per inventory, and not one product per subtree. A
    subtree product would only be flattened away by its parent's, so the sites are collected here
    and :func:`_term_rows` forms the one product over them."""
    for child in _kids(node):
        yield from _site_catalogs(term, child, node)
    yield tuple(row for block in _site_blocks(term, node, parent) for row in _block_rows(node, block))


def _walk_nodes(node: _Node):
    yield node
    for child in _kids(node):
        yield from _walk_nodes(child)


def _work_groups(term: _Term, rows: _RowProduct) -> list[tuple[str, Sequence[_Row]]]:
    """The kernel's ``WORK`` segments: the row product partitioned by the inventory its rows claim.

    The offered inventories are not scanned for and then enumerated against — they ARE the claims
    the rows make, so what a segment is stamped with and what its rows spell cannot disagree.

    A live ``WORK`` pin narrows to the matching group, and a ``+p`` pin whose BASE inventory the
    term does claim narrows to no group at all — the band is part of the inventory, so a term with
    no row that can drive it stays unmapped rather than deploying without it.

    THE ONE PLACE A PIN DOES NOT NARROW is the PIN-BLEED rule: one env pin, several kernels in the
    graph, and this is not the one it was written for (a recognition fork's reduce sibling seeing a
    matmul's warp pin). The pinned inventory then LEADS, carrying the rows that claim nothing — the
    only rows any inventory can spell — and the term's own groups stay as siblings, so it still
    maps rather than being left unmapped over a pin that was never about it. Both halves of that
    are pinned by ``test_work_pin_widens_only_where_the_site_offers_no_warp_inventory``."""
    groups = rows.work_groups()
    raw = WORK.raw()
    if raw is None:
        return groups
    kept = [group for group in groups if values_equal(WORK.name, raw, group[0])]
    if kept:
        return kept
    pinned = Workers.parse(raw)
    if pinned is not None and pinned.producer and any(spell == _spell(replace(pinned, producer=0)) for spell, _ in groups):
        # The term CAN spell the pinned inventory; it is the BAND no row of it can drive. A band is
        # PART of the inventory, so this is a pin the term simply has no row at — it stays unmapped
        # rather than quietly deploying without the band the pin asked for. Not pin bleed: bleed is
        # for a pin that is not about this kernel at all, which is the branch below.
        return []
    logger.warning(
        "WORK pin %r matches no candidate's worker inventory (%s offered); offering it beside the full fork",
        raw,
        ", ".join(repr(spell) for spell, _ in groups) or "none",
    )
    unclaimed = [sub for spell, sub in groups if spell == ""]
    return [(_spell(pinned), unclaimed[0]), *groups] if unclaimed else groups


def _level_keys(term: _Term) -> list[str]:
    """The site keys ``term``'s own tree decides, family by family, in fork-level order."""
    decided: dict[str, list[str]] = {f: [] for f in FAMILIES}

    def walk(node: _Node) -> None:
        for family, key in node.keys.items():
            decided[family].append(key)
        for child in _kids(node):
            walk(child)

    for node in term.tree:
        walk(node)
    return [k for family in FAMILIES for k in decided[family]]


def _keys(term: _Term) -> list[str]:
    """The stored tree's site keys between ``WORK`` and ``RASTER``, in fork-level order. A family
    with no keyed site keeps its bare key so every schedule row has the same family vocabulary."""
    seen: dict[str, list[str]] = {f: [] for f in FAMILIES}
    for key in _level_keys(term):
        fam = family_of(key)
        if key not in seen[fam]:
            seen[fam].append(key)
    return [k for family in FAMILIES for k in (seen[family] or [family])]


def _term_rows(term: _Term) -> _RowProduct:
    """The lazy compatible product over EVERY site of the term.

    ``_RowProduct`` joins the sites on the interface they present one another: the physical tile
    axes, the fragment seam, and the worker inventory each row claims. Parent and child, and the
    independent roots of a forest, therefore compose by one rule — and no inventory is chosen
    before the rows exist, so no catalog is built twice. A term with no site still has one row,
    which is the empty product."""
    return _RowProduct.build(term, tuple(catalog for root in term.tree for catalog in _site_catalogs(term, root)))


def _space(term: _Term) -> PoolSpace:
    """Every legal schedule candidate for the stored term, as an addressable space with one
    segment per worker inventory.

    ONE product over the sites, partitioned by the inventory its rows claim through the same
    combo regrouping (:meth:`_RowProduct._regrouped`) that opens a fork level, so the segment
    layout and the fork levels narrow the space the same way."""
    keys = _keys(term)
    rows = _term_rows(term)
    #: Every key the tree spells, decided-empty until a row supplies it.
    base = {k: "" for k in keys}
    if term.warp_eligible:
        # ``S_``-prefixed - not a schedule family, so tile identity and prefix-consistency are
        # untouched (``canonical_row_key`` reads the tuning-knob view); it prices "a scalar tile
        # where tensor cores were on offer". It rides the BASE dict rather than a closing pass over
        # the rows: the walk above already asked every site for its tile catalog, which is the one
        # thing that sets the flag, so the answer is known before the first candidate exists.
        base["S_warp_eligible"] = 1.0
    rasters = [{RASTER.name: r} for r in _raster_values(term)]
    segments = [Segment.build(sub, {WORK.name: spell}, rasters) for spell, sub in _work_groups(term, rows)]
    return PoolSpace.build(keys, base, segments)


def _enumerate(term: _Term, sample=None) -> tuple[Sequence[dict], list[str], int]:
    """The addressable legal schedule space, its site keys, and its exact size.

    ``sample`` is the Context's ``search.pool.PoolSample``, ``None`` on every live compile. It
    draws rows by index. A live compile keeps the space itself: candidate dictionaries are created
    only as the fork walk addresses them, so schedule cardinality is not a materialization limit."""
    space = _space(term)
    total = len(space)
    rows = space if sample is None else sample.take(space)
    if not rows and term.pin_error is not None and not term.pin_spelled:
        raise term.pin_error  # NO inventory could spell the pin - a pin names a specific kernel
    return rows, list(space.keys), total


# ---- materialization: one builder per form, all fed by the same row ------------------------------ #


def _stamp(term: _Term, op, name, knobs: dict, slices, workers=None) -> TileOp:
    """Build the scheduled ``TileOp`` — :func:`ops.scheduled` over this term's placement and root
    stores. The term stays pure algebra; no slice is ever a node field."""

    return scheduled(
        op,
        name=name,
        place=term.place,
        knobs=knobs,
        output_specs=term.tile.output_specs,
        slices=slices,
        workers=workers,
    )


def _strip_variant(term: _Term, plan: TilePlan, name: str, knobs: dict) -> TileOp:
    """The pointwise register-STRIP term variant: hand each thread ``r`` CONTIGUOUS inner-axis
    elements. The inner free axis shrinks to ``extent/r`` (the grid walks it) and the cell body is
    unrolled ``r`` times — copy ``i`` reads/writes ``inner·r + i`` with its SSA names suffixed —
    then regrouped as ``r`` loads · ``r`` computes · ``r`` writes so the unit-stride runs feed
    ``050_vectorize_loads`` / ``080_vectorize_stores``. A different term, hence a different
    ``structural_key`` and ``Op.cache_key`` — which is why it is applied HERE and not at recognition."""
    inner = term.place.free[-1]
    r = plan.reg_n
    op = term.tile.op
    ssa: set[str] = set()
    for s in op.body:
        ssa.update(s.defines())
    loads: list[Stmt] = []
    computes: list[Stmt] = []
    stores: list[OutputSpec] = []
    for i in range(r):

        def rename(n: str, i: int = i) -> str:  # suffix only the body's SSA names; axis vars stay
            return f"{n}__u{i}" if n in ssa else n

        sigma = Sigma({inner.name: BinaryExpr("+", BinaryExpr("*", Var(inner.name), Literal(r, "int")), Literal(i, "int"))})
        for s in op.body:
            s2 = s.rewrite(rename, sigma)
            (loads if isinstance(s2, Load) else computes).append(s2)
        stores.extend(OutputSpec(write=st.write.rewrite(rename, sigma)) for st in term.tile.output_specs)
    new_inner = replace(inner, extent=Dim(inner.extent.as_static() // r))
    new_free = (*term.place.free[:-1], new_inner)
    new_place = Placement(free=new_free, grid=new_free)
    return scheduled(Fold.projection(body=Body((*loads, *computes))), name=name, place=new_place, knobs=knobs, output_specs=tuple(stores))


def _free_option(term: _Term, plan: TilePlan, name: str, knobs: dict, nested: Sequence[tuple] = ()) -> TileOp:
    """One zero-axis row: the flat per-cell map, or the strip variant when the row's ``TILE``
    names a register width. A zero-axis fold with no operands has no nested sites, so the strip
    arm takes none."""
    if _strip_width(plan) > 1:
        return _strip_variant(term, plan, name, knobs)
    return _stamp(term, term.tile.op, name, knobs, nested)


def _factor_k(k_axis: Axis, w: int) -> tuple[Axis, Axis, Sigma]:
    """Factor a STATIC contraction axis into ``ksplit × kslice``. ``ksplit`` (extent ``w``, name
    ``<k>_ks``) becomes the outer :class:`Fold`'s reduce axis, parallelized across CTAs and summed
    in the finalize; ``kslice`` (extent ``K/w``, the ORIGINAL name) stays the inner contraction's.
    The ``sigma`` maps the original ``k`` to ``ksplit·(K/w) + kslice`` so the operand loads
    reconstruct the absolute index; distinct names are what avoid a double-reduce."""
    legal.enforce(legal.splitk_width(k_axis, w), pinned=True)
    b = k_axis.extent.as_static() // w
    # Keep the established generated spelling; free-loop canonicalization now uses the partial
    # workspace's output-coordinate order, not this name, to retain the partition as a lead axis.
    ksplit = Axis(name=f"_{k_axis.name}_ks", extent=Dim(w))
    # The slice carries its parentage: a cross-CTA split is CONSUMED by the rewrite that realizes
    # it, and an axis that is already a window of a parent is one nothing may partition again.
    kslice = replace(k_axis, extent=Dim(b), window=Window(parent=k_axis.source_axis or k_axis, partition=True))
    sigma = Sigma({k_axis.name: BinaryExpr("+", BinaryExpr("*", Var(ksplit.name), Literal(b, "int")), Var(k_axis.name))})
    return ksplit, kslice, sigma


def _sliced_edge(edge, sigma: Sigma, k_name: str):
    """An operand edge σ-reindexed to absolute k for a split partition — the SAME rule on either
    edge. A MATERIALIZED edge rewrites its gmem index; a COMPUTED cone rewrites its per-cell BODY
    and every K-VARYING producer edge it composes (attention's per-cell score contraction — the
    slice's own k coordinate reaches gmem through that node, so leaving it unreindexed makes every
    partition recompute partition 0's scores). The cone's row-invariant prologue (the per-row
    statistic, the K seam ``ops.cone_seam`` reads off the node boundary) spans the whole row and
    stays FULL-ROW in every partition, each recomputing it — the REDUNDANT-STATISTIC split. That
    redundancy is what the split trades for parallelism; whether it pays on a given shape is
    evidence's decision."""
    if isinstance(edge, Load):
        return replace(edge, index=tuple(sigma.apply(e) for e in edge.index))
    ops = tuple(e.rewrite(lambda nm: nm, sigma) if edge_refs_axis(e, k_name) else e for e in edge.operands)
    return replace(edge, operands=ops).with_bodies((Body(tuple(s.rewrite(lambda nm: nm, sigma) for s in edge.body)),))


def _splitk_option(term: _Term, plan: TilePlan, node, rplan: ReducePlan, name: str, knobs: dict) -> TileOp:
    """One SPLIT-K contraction row — the structural ``Fold(axis=ksplit) ⊃ Fold(axis=kslice)``
    composition ``030_split_reduce`` consumes into the cross-CTA partial + finalize. The inner node
    is the SAME contraction a non-split matmul builds, over ``kslice`` with operands σ-reindexed to
    absolute k; the outer reduce is the IDENTITY-lift composition over it (``Fold.composed``).

    It resolves NO STAGE. The split mints brand-new kernels that schedule themselves, so a
    transport resolved here for the partial would be discarded at the splice — including the
    smem-budget refusal a declining compute fill used to raise, which is the partial's own fork's
    to make about the partial's own K. The TILE slice stays because the ROW needs it: ``WORK`` is
    the inventory derived from a row's tile slices, and a row that seals none spells an empty
    inventory."""
    if not plan.is_warp:
        legal.enforce(legal.scalar_block_threads(plan), pinned=True)
    ksplit, kslice, sigma = _factor_k(node.axis, rplan.cta)
    mul, plus = node.semiring
    inner = Fold.contraction(
        k_axis=kslice,
        a=_sliced_edge(node.a, sigma, node.axis.name),
        channels=tuple(replace(ch, b=_sliced_edge(ch.b, sigma, node.axis.name)) for ch in node.channels),
        product=mul,
        fold_op=plus,
    )
    # ONE composition rule: the outer reduce is the IDENTITY lift over the sliced contraction
    # operand, its combine the componentwise ⊕ (the node's OWN semiring ⊕ — the reassociation
    # ``fold_k = fold_{ksplit} ∘ fold_{kslice}`` is licensed by that monoid's associativity)
    # over the same accumulator names.
    accs = tuple(inner.defines())
    outer = Fold(
        axis=ksplit,
        operands=(inner,),
        lift=Lambda(params=(ksplit.name, *accs), body=Body(()), results=accs),
        **dict(zip(("init", "combine"), M(*([plus] * len(accs)), names=accs), strict=True)),
    )
    op = Fold.projection(body=term.proj, operands=(outer,)) if len(term.proj) else outer
    # Re-indexing the computed cone can expose another canonical operand split. Normalize first,
    # then key the schedule against the nodes that are actually stored. No nested slice survives
    # this choice: 030 replaces the split carrier with fresh kernels which schedule themselves.
    normalized = TileOp(op=op, place=term.place, output_specs=term.tile.output_specs).op
    actual_outer = normalized.operands[0] if normalized.axis is None else normalized
    actual_inner = actual_outer.composed
    assert actual_inner is not None
    return _stamp(term, normalized, name, knobs, [("REDUCE", actual_outer, rplan), ("TILE", actual_inner, plan)])


def _materialize(term: _Term, row: dict, name: str, knobs: dict) -> TileOp:
    """One row → its ``TileOp``, every slice RE-RESOLVED from the row's spellings through the same
    dispatches the enumeration used (:func:`_resolve_stage`, ``resolve_site_tile``,
    ``ReducePlan.parse``) — the row is the kernel's complete identity, and decode-by-spelling is
    the replayability invariant enforced at its one seam. The FORM is the two node predicates
    again, never a role.

    A family a site left at its DECIDED EMPTY spells ``""`` and is still resolved against the
    inventory: an empty ``TILE`` beside a thread inventory is a real unit-register tile and only
    ``resolve_site_tile`` knows it."""
    work = Workers.parse(row.get(WORK.name) or None)
    # Structural stamps (``S_warp_eligible``) ride onto the op: fork rows carry them for branch
    # identity, but the MATERIALIZED op is what ``realized_knobs`` reads, and dropping them here
    # left leaf/evidence rows unstamped while fork rows were stamped — fracturing the ``S_*``
    # evidence signature (the 2026-07-07 5090 gate's 330× fp16 miss).
    op_knobs = {**knobs, **{k: v for k, v in row.items() if k.startswith("S_")}}
    raster_spec = row.get(RASTER.name, "")
    Raster.parse(raster_spec)  # loud pin contract — a malformed spelling fails the row here
    op_knobs = {**op_knobs, RASTER.name: raster_spec, **{k: v for k, v in row.items() if family_of(k) in FAMILIES}}

    if not term.tree:
        return _free_option(term, resolve_site_tile(row.get("TILE", "") or "", work), name, op_knobs)

    slices: list[tuple] = []
    split = None
    workers = None
    for root in term.tree:
        keys = root.keys

        def value(family: str, keys: dict = keys) -> str:
            return row.get(keys.get(family, family), "") or ""

        node = root.site.node
        nested = _nested_slices(term, root, row, work)
        if node.axis is None:
            plan = resolve_site_tile(value("TILE"), work)
            if _strip_width(plan) > 1:
                if len(term.tree) != 1:
                    raise ValueError("a register strip cannot materialize one member of a multi-root term")
                return _strip_variant(term, plan, name, op_knobs)
            slices.extend(nested)
            continue

        rplan = ReducePlan.parse(value("REDUCE"), work)
        # An empty spelling is a unit register tile only when THIS root owns a TILE site. A
        # nested-only term has no root TILE key at all; borrowing its shared thread inventory there
        # invents a slice the codec cannot address.
        plan = resolve_site_tile(value("TILE"), work, rplan.coop) if "TILE" in keys else TilePlan()
        if is_contraction(node) and rplan.needs_split:
            if len(term.tree) > 1:
                # Keep each selected partition on its root. The split rewrite consumes the
                # resulting scheduled forest.
                slices.extend((("REDUCE", node, rplan), *nested))
                continue
            split = (root, plan, rplan)
            continue
        stage = _stage_of(term, node, node, plan, value("STAGE")) if value("STAGE") else None
        if plan.is_tiled:
            legal.enforce(legal.warp_k_step(node, plan) if plan.is_warp else legal.scalar_block_threads(plan), pinned=True)
            slices.extend((("TILE", node, plan), ("STAGE", node, stage)))
            if work is not None and work.producer:
                workers = WarpSpec(work.producer)
        else:
            slices.extend((("REDUCE", node, rplan if rplan.stages else None), ("STAGE", node, stage)))
        slices.extend(nested)

    if split is not None:
        root, plan, rplan = split
        return _splitk_option(term, plan, root.site.node, rplan, name, op_knobs)
    return _stamp(term, term.tile.op, name, op_knobs, slices, workers=workers)


def _nested_slices(term: _Term, node: _Node, row: dict, work: Workers | None) -> list[tuple]:
    """Every NESTED site's resolved slices, as the ``(family, node, value)`` triples ``scheduled``
    keys — materialization's half of the recursion :func:`_site_catalogs` already does.

    The enumeration walks the whole site tree, so a row DECIDES every site; stamping the root alone
    left a nested key as a knob no kernel realized — the row said ``REDUCE@j=r2`` and the op's
    schedule came back empty. The walk descends through :func:`_kids`, the same accessor
    ``_site_catalogs`` uses, so what materializes is what was enumerated. A site whose value is the decided empty resolves
    to ``None`` and ``scheduled`` skips it, which is why the corpus terms — whose one nested site is
    the cone statistic the parent fill realizes — stamp nothing new."""
    out: list[tuple] = []
    for child in _kids(node):
        cnode, keys = child.site.node, child.keys

        def spec(family: str, keys: dict = keys) -> str:
            return row.get(keys.get(family), "") or ""

        rplan = ReducePlan.parse(spec("REDUCE"), work)
        tile_spec = spec("TILE")
        plan = resolve_site_tile(tile_spec, work, rplan.coop)
        scheduled_node = term.scheduled_node(child, node)
        stage = _stage_of(term, scheduled_node, cnode, plan, spec("STAGE")) if spec("STAGE") else None
        out.append(("REDUCE", cnode, rplan if rplan.stages else None))
        out.append(("STAGE", cnode, stage))
        if "TILE" in keys and tile_spec:
            # Stored UNPLACED: which ``(m, n)`` pair a nested site tiles is a function of its
            # POSITION (``Sched._mn_for`` — the parent fold's axis for a hoisted edge, the trailing
            # free pair for a derived one), so binding it here with the ROOT's rule would name the
            # wrong axes. ``Sched.tile_of`` binds at read, through the one home.
            out.append(("TILE", cnode, plan))
        out.extend(_nested_slices(term, child, row, work))
    return out


def _stage_of(term: _Term, node, stored_node, plan: TilePlan, spec: str) -> Stage | None:
    """The row's ``STAGE`` re-resolved against the node — the operand pipeline on a tiled
    contraction, the shared ROW buffer on any other fold, dispatched by the same predicate the
    enumeration used. The row carries what the enumeration RESOLVED, so this reproduces the slice
    the leaf identity was built from, through :func:`_resolve_stage`'s one dispatch."""
    if not spec:
        return None
    if not is_contraction(node.fold if isinstance(node, _FoldInterface) else node):
        return _row_stage(term, node)
    if not plan.is_tiled:
        return None
    placed = term.sched.placed(stored_node, plan)
    return _resolve_stage(term, node, placed, Stage.parse(spec))


# ---- the pool cache and the entry point ---------------------------------------------------------- #


@dataclass(frozen=True)
class _Pool:
    """One term's enumerated schedule pool — everything :func:`schedule` derives that is
    OP-INDEPENDENT: the fork's site keys and the rows. A row is a complete spelled identity and
    carries no resolved slices — materialization re-resolves them from the spelling. Shared through ``ctx.session_cache``
    across ops with equal ``cache_key`` and across tune trajectories (the pipeline re-runs
    ``020_schedule`` per trajectory), so it sits BELOW the search policies: greedy and MCTS hit
    it alike, and it holds NO ranking and consults NO evidence — only what evidence cannot
    change belongs here. Rows are read-only mappings (every consumer that mutates already
    copies — ``_Leaf``, the greedy row merge)."""

    keys: tuple[str, ...]
    rows: Sequence
    #: The size of the SPACE the rows came from - ``len(rows)`` unless the Context asked for a
    #: sample. Memoized with them because it is the same pure function of the term, and because a
    #: rank is only interpretable next to what it was ranked among.
    total: int

    @classmethod
    def build(cls, rows: Sequence[dict], keys: list[str], total: int) -> _Pool:
        stored = rows if isinstance(rows, PoolSpace) else tuple(MappingProxyType(dict(row)) for row in rows)
        return cls(tuple(keys), stored, total)


def schedule(tile: TileOp, name: str, knobs: dict, ctx) -> Fork | list[TileOp] | TileOp:
    """Map a newly lifted, unmapped ``tile`` onto the grid and offer its scheduling fork.

    Returns the lazy fork tree over the enumerated rows (levels ``[WORK, *site keys, RASTER]`` — the
    kernel-global worker inventory leads, so every deeper prefix row is self-decoding; the
    launch-order codec closes), a single ``TileOp`` when the space collapses to one row, or ``[]``
    when nothing is enumerable (the guardrail contract — the caller leaves the term unmapped; an
    empty pool is cached too, so the guardrail answers from the memo on repeat).

    The enumeration itself is memoized in ``ctx.session_cache`` (:class:`_Pool`): the rows are a
    pure function of ``(term, ctx, pins, hints, ctx.pool_sample)``, so N same-shape ops — and every tune trajectory
    after the first — pay one enumeration. The term is rebuilt per op so materialization always stamps against THIS
    op's placement and stores."""
    term = _Term(tile, tile.place.on_grid(), ctx)
    cache = getattr(ctx, "session_cache", None)
    sample = getattr(ctx, "pool_sample", None)
    # The sample is part of the KEY, not merely of the Context: ``dataclasses.replace`` SHARES the
    # session cache, so a sampled Context and the live one it came from sit on one memo and a
    # Context-only flag would serve a sampled pool to a live compile.
    key = (
        digest(pool_key(tile, pins=schedule_pin_fingerprint()), sample.key if sample is not None else "")
        if cache is not None or sample is not None
        else None
    )
    pool = cache.get(key) if cache is not None else None
    if pool is None:
        pool = _Pool.build(*_enumerate(term, sample))
        if cache is not None:
            cache.put(key, pool)
    if sample is not None:
        sample.totals[key] = pool.total  # the sampled rows cannot carry it; the caller reads it here
    if not pool.rows:
        return []

    def materialize(row: dict) -> TileOp:
        return _materialize(term, row, name, knobs)

    if len(pool.rows) == 1:
        return materialize(pool.rows[0])

    def _level(key: str) -> Level:
        return Level((key,), key=lambda r: (r.get(key, ""),), partition_key=key)

    levels = [_level(WORK.name), *(_level(k) for k in pool.keys), _level(RASTER.name)]
    return build_fork_tree(params=pool.rows, levels=levels, materialize=materialize)


__all__ = ["FAMILIES", "schedule"]
