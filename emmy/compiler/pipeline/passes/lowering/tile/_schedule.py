r"""Schedule a lifted (UNMAPPED) ``TileOp`` — the generic row enumerator.

**Every role emits rows through ONE recursive walk of the site tree; no role builds ``TileOp``\ s
directly, and no term shape gets its own path.** A row is a joint assignment across every scheduling
SITE of a term, and the tree that generates it is the term's own:

.. code-block:: text

    for work in _inventories(term)            # the kernel's ONE inventory, CHOSEN at the root
      Segment(rows=_term_rows(term, work),     # one WORK rectangle of the space
              rasters=_raster_values(term))   # kernel-global, like work - a free axis of the segment

    _rows_at(site, work) = for combo in product(_rows_at(child, work) for child in children)
                             for block in _site_blocks(site, work)      # RESOLVED against work
                               _merge(site, block, combo)               # spells each slice ONCE

The walk emits :class:`._pool.Block` rectangles, not values: a block fixes everything a site decides
except ``STAGE`` and carries the stages legal for it. That factoring is a fact about the FILTER -
:func:`_work_holds` and :meth:`_Row.union` read the resolved tiles and the cooperative width alone,
so the transport and the launch order multiply through them unconditionally. A ``_Row`` therefore
stands for ``width x len(rasters)`` candidates, the validation runs once per legal
``(TILE, REDUCE)`` assignment, and :mod:`._pool` turns the whole thing into an addressable space
whose size is known before any candidate dict exists.

``WORK`` leads because the codec says so: :meth:`TilePlan.parse` and :meth:`ReducePlan.parse` both
take the inventory as an INPUT — a ``TILE`` value's unit widths and a ``REDUCE`` value's coop width
are READ OFF it — so the dependency runs work → slice, and a candidate that cannot spell against the
chosen inventory is simply not in ``_site_blocks(site, work)``. :func:`derive_inventory` stays, as
the VALIDATION that a row's own slices imply the inventory it claims.

Three layers, each with one job:

- the candidate DOMAIN is generated from its bounds in ``search/space.py`` (the tile spaces) or
  listed in its catalog there (the families with no multiplicative coupling — stages, split widths,
  the coop partitions, the raster orders);
- per-node LEGALITY — what a domain cannot know because it depends on this term's K, N, dtype and
  smem cap — is :mod:`._legality`, one predicate per rule, raise-vs-drop chosen by ``pinned``;
- THIS module decides which families a SITE offers and how a row becomes a ``TileOp``.

**Legality is the only limit, and this module ranks nothing.** Every partition a site can legally
spell is enumerated; no default, ordering or filter here exists to make an unmeasured compile land
on a particular row. The result is a SET, and its emission order is an implementation detail of the
recursion — the deploy evidence hierarchy (recorded goldens, then measurements, then the fitted
prior) is what ranks it. A compile with no evidence and no prior therefore takes whatever row the
walk emitted first, and that row being slow is an accepted outcome, never a reason to add a rule
back.

**Dispatch is two stored-param predicates on the node, never the** :class:`AxisRole`: ``axis is
None`` selects the register-strip values, :func:`is_contraction` the tile × stage × reduce product,
and everything else falls through to the reduce partition. The role is a LOOP annotation and a
materializer read; it never selects a catalog here.

What the walk covers:

- the pointwise cell: the register-strip ladder (``TILE=f<r>``, a TERM VARIANT applied at
  materialization);
- the reduce partition (``REDUCE``): the serial fold plus every legal partition in the coop / ILP
  catalog;
- the contraction: the ``TILE × STAGE × REDUCE × RASTER`` legal product over the scalar and warp
  (mma) tiers, split-K rows routing through the structural ``Fold ⊃ Fold`` composition that
  ``030_split_reduce`` consumes;
- a COMPUTED pointwise edge on either contraction operand (including fused norm→linear / gate⊗up
  on A and an expanding pure producer on B): the warp tier over the mandatory ``reg``
  compute-fill, with the cone's own statistic site under the same inventory — a
  ``_site_values`` entry plus legality, not an emitter of its own;
- a Fold step containing two contractions: each child carries its own tile and the parent accepts
  exactly the geometries that share one blocked reduce axis.

A term this walk cannot schedule yields NO rows, and ``020_schedule`` leaves it unmapped rather than
guessing. That is the guardrail contract: empty enumeration returns ``[]``, never raises.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from itertools import product
from types import MappingProxyType

from emmy.compiler.dim import DEFAULT_SEQ_HINT, Dim
from emmy.compiler.ir.atom import atoms_for
from emmy.compiler.ir.axis import Axis, Window
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.pure import Lambda, M, component_ops
from emmy.compiler.ir.pure.fold import Fold, edge_refs_axis, is_contraction, operand_body
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
from emmy.compiler.ir.stmt import Assign, Body, Load, Loop, Stmt, Write
from emmy.compiler.ir.stmt.passes import has_contraction_tail, projection_distributes
from emmy.compiler.ir.tile import Placement, Store, TileOp
from emmy.compiler.ir.tile.ops import Sched, projection_tail, scheduled
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

#: The most rows one kernel's enumeration may MATERIALIZE. The product across sites is GENERATED,
#: so a term that widens it silently would hand the search a space it cannot walk and the prior a
#: feature space it cannot cover. Exceeding it is a LOUD failure, never a truncation — a truncated
#: enumeration reads as "covered everything" while dropping whichever rows the walk reached last.
#: Measured headroom: the widest live term (a static f16 square matmul, both tiers, every stage /
#: split / raster) enumerates ~133k rows.
#:
#: The budget is checked against :meth:`PoolSpace.__len__`, which is a prefix-sum lookup — so an
#: over-budget term now fails BEFORE its first candidate dict exists rather than after 400k of
#: them. The space itself is unbounded: it is the MATERIALIZATION this bounds.
MAX_ROWS = 400_000


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


def _hint_extent(ax) -> int:
    """An axis's static extent, or its ``Dim`` hint when symbolic."""
    e = ax.extent
    return e.as_static() if e.is_static else (e.hint or DEFAULT_SEQ_HINT)


def _hint_fingerprint(tile: TileOp) -> tuple[int, ...]:
    """The hint-resolved extents of the term's SYMBOLIC axes, in walk order. ``Dim.hint`` is
    deliberately excluded from identity (``Op.cache_key`` stays hint-independent), but the
    enumeration SIZES against it (:func:`_hint_extent` → which coop bands the reduce extent can
    feed), so the pool cache's key must carry it — two same-key ops traced at different
    ``--seq-len`` hints enumerate different pools."""
    out: list[int] = []

    def note(ax) -> None:
        if ax is not None and not ax.extent.is_static:
            out.append(_hint_extent(ax))

    def walk(node) -> None:
        if not isinstance(node, Fold):
            return
        note(node.axis)
        for e in node.operands:
            walk(e)
        for s in node.lift.body:
            walk(s)

    for a in tile.place.free:
        note(a)
    walk(tile.op)
    return tuple(out)


def _extent_fingerprint(tile: TileOp) -> tuple[str, ...]:
    """Every axis extent of the recognized term in walk order — the free grid, then each
    ``Fold`` axis: a static extent as its integer, a symbolic axis as the bare ``sym`` marker
    (identity stays hint-free — a symbolic record is the symbolic kernel's identity at every
    hint). Part of :func:`deploy_identity` because the α-invariant algebra digest canonicalizes
    sizes away: without extents every same-algebra cone on a card shares one key, and the
    fastest record of ANY shape decides them all (an m32 scalar row deploying onto every M)."""
    out: list[str] = []

    def note(ax) -> None:
        if ax is not None:
            out.append(str(ax.extent.as_static()) if ax.extent.is_static else "sym")

    def walk(node) -> None:
        if not isinstance(node, Fold):
            return
        note(node.axis)
        for e in node.operands:
            walk(e)
        for s in node.lift.body:
            walk(s)

    for a in tile.place.free:
        note(a)
    walk(tile.op)
    return tuple(out)


def _inner_free(place: Placement) -> Axis | None:
    """The innermost NON-UNIT free axis — the m1 recognizer's synthesized unit axis can sit
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
    return all(isinstance(s, (Load, Assign, Write)) for s in op.body) and all(st.sweep is None for st in term.tile.stores)


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
    keyed = [(s, {f: k for f in FAMILIES if (k := key(f, s.node)) is not None}) for s in sites(op)]
    keyed = [(s, keys) for s, keys in keyed if keys]

    def under(parent: Site, child: Site) -> bool:
        return len(child.segments) > len(parent.segments) and child.segments[: len(parent.segments)] == parent.segments

    def build(parent: Site | None) -> tuple[_Node, ...]:
        kids = [(s, keys) for s, keys in keyed if parent is None or under(parent, s)]
        tops = [(s, keys) for s, keys in kids if not any(t is not s and under(t, s) for t, _ in kids)]
        return tuple(_Node(site=s, keys=keys, children=build(s)) for s, keys in tops)

    return build(None)


def _step_contractions(node) -> tuple[Fold, ...]:
    """The direct bilinear children in one Fold step, in evaluation order."""
    if not isinstance(node, Fold) or node.axis is None:
        return ()
    return tuple(stmt for stmt in node.step_stmts() if isinstance(stmt, Fold) and stmt.semiring is not None)


def _blocked_pair(node) -> tuple[Fold, ...]:
    """The two contractions in an exp-family Fold's blocked step, if present."""
    pair = _step_contractions(node)
    if (
        len(pair) == 2
        and component_ops(node.combine) is None
        and len(node.init) >= 3
        and len(node.combine.results) >= 3
        and tuple(node.lift.results[1:2]) == (1.0,)
    ):
        return pair
    return ()


def _keeps_children(site: Site) -> bool:
    """Whether a site's nested algebra remains independently schedulable."""
    return is_contraction(site.node) or bool(_blocked_pair(site.node))


def _kids(node: _Node) -> tuple[_Node, ...]:
    """The children the site tree descends into — :func:`_keeps_children` applied, in ONE place.
    Four walks need it (the row product, the inventory scan, the key spelling and the
    materializer's slice stamping) and they must agree exactly: what materializes is what was
    enumerated, so a walk that pruned differently would stamp a key no row decided."""
    return node.children if _keeps_children(node.site) else ()


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
        self.tree = _site_tree(tile.op, self.key)
        self._tiles: dict[int, dict[str, list[TilePlan]]] = {}
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
        return self.sched.key(family, node)

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

    def keyed_pin(self, family: str, node) -> str | None:
        """The EXPLICIT ``FAMILY@<element>`` pin for this site — no bare fallback. The two reads
        are different questions where a family has SEVERAL sites: an explicit key names one site
        and is authoritative there, while a bare pin fans out to every eligible site and cannot say
        which it meant (``knob.pin_key_matches``), so it narrows by MATCHING each site's own
        catalog and leaves a site it names nothing at alone."""
        key = self.key(family, node)
        element = key.partition("@")[2] if key is not None else ""
        return _KNOBS[family].pin_at(element) if element else None

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
        # A synchronous-fill node is warp-ONLY: computed edges need evaluation, while a product
        # node has several B/C channels that the gmem-direct and scalar emitters cannot carry.
        scalar = scalar_tile_moves() if not _requires_sync_fill(node) else []
        grouped: dict[str, list[TilePlan]] = {}
        for plan in scalar + warp:
            w = plan_workers(plan)
            grouped.setdefault(w.spell() if w is not None else "", []).append(plan)
        # A ``TILE`` pin is authoritative over the VALUES but not over the inventories: its unit
        # widths are read OFF ``WORK``, so the pin names a different plan under each one and is
        # re-resolved per inventory in :func:`_contraction_values`. The catalog still answers
        # "which inventories can this site spell against".
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


def _tile_ok(term: _Term, node, plan: TilePlan) -> bool:
    """Whether a warp tile candidate is realizable on ``node`` — the K-step divisibility every warp
    row needs, plus the exact-cover geometry the smem compute fill adds. Both are ``_legality``
    predicates, dropped here and RAISED on a pin (:func:`_contraction_values`)."""
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
    placed = plan.placed_on(term.place)
    if placed.axes is None:
        return False  # no (m, n) pair on the grid — nothing to place a compute-filled tile on
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


def _a_dtype(node, inputs):
    """The ``a`` edge's element dtype — the value the mma fragment reads. A MATERIALIZED edge reads
    its gmem tensor's; a COMPUTED cone reads its K-indexed leaf ``Load``'s, which is the value the
    smem compute fill stores to the A slab."""
    ld = node.a
    if not isinstance(ld, Load):
        k = node.axis.name
        ld = next((s for s in operand_body(node.a) if isinstance(s, Load) and k in {v for e in s.index for v in e.free_vars()}), None)
    t = inputs.get(ld.input) if ld is not None else None
    return t.dtype if t is not None else None


def _channel_dtype(node, inputs):
    """The one element dtype every channel's B agrees on, or ``None`` — the dtype an f32 ``a`` still
    rides when the smem compute fill DEMOTES it on the slab store."""
    bs = [ch.b for ch in node.channels]
    if not bs or not all(isinstance(b, Load) for b in bs):
        return None
    dts = {getattr(inputs.get(b.input), "dtype", None) for b in bs}
    return next(iter(dts)) if len(dts) == 1 else None


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
    ab = _a_dtype(node, inputs)
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
            and _channel_dtype(node, inputs) == ab
            and node.axis.extent.is_static
            and (_f8_mma_allowed(term.ctx) or (pin is not None and any(a in pin for a in atoms)))
        )
        return atoms if ok else ()
    if not atoms_for(ab, ctx=term.ctx):
        ab = _channel_dtype(node, inputs)  # the demoting compute fill — an f32 a (cone leaf or plain load) on 16-bit B
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
    return not _carries_partition(term.tile.op)


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

    pin = term.pin("REDUCE", node)
    if pin is not None:
        plan = _consumed_split(term, node, ReducePlan.parse(pin, Workers.parse(WORK.raw())))
        band_legal(plan, pinned=True)
        if plan.needs_split and plan.finalize == "atomic":
            legal.enforce(legal.direct_atomic_output(term.tile.outputs), pinned=True)
        return [plan]
    extent = _hint_extent(node.axis)
    cands = [ReducePlan()]
    for p in coop_reduce_moves():
        if p.needs_split and not _splittable_axis(term, node):
            continue  # the axis is already a slice — its cross-CTA partition was consumed
        if not band_legal(p, pinned=False):
            continue
        if p.finalize == "atomic" and not legal.enforce(legal.direct_atomic_output(term.tile.outputs), pinned=False):
            continue
        if p.coop <= extent and p.reg <= extent and p not in cands:
            cands.append(p)
    return cands


def _reduce_blocks(term: _Term, node) -> list[Block]:
    """The reduce-partition values a non-contraction fold offers: the partition itself plus the
    shared-row ``STAGE`` a cooperative band can drive (a resolver, not a choice — see
    :func:`_row_stage`). Each block is a rectangle ONE stage deep: the transport here is a function
    of the partition, never a free axis beside it. Which of them SPELL against the kernel's chosen
    inventory is the row's question, not this site's (:func:`_work_holds`) — a serial fold claims
    no workers at all, so at a NESTED site it composes with any parent inventory."""
    return [Block({"REDUCE": plan}, (_row_stage(term, node) if plan.coop > 1 else None,)) for plan in _reduce_specs(term, node)]


def _fill_realized(parent: _Node | None, site: Site) -> bool:
    """Whether the PARENT form realizes this nested fold's partition ITSELF, leaving the site's own
    value the decided empty. One form does today: the smem compute fill's per-row statistic
    prologue stripes a cone's statistic ONE ROW PER WARP, the warp's 32 lanes striding the fold and
    closing it on the shuffle butterfly (``lowering/kernel/_stage.sync_stat_fill``) — a single
    hardwired partition, so any value here would stamp a knob no kernel realizes."""
    if parent is None or not is_contraction(parent.site.node):
        return False
    depth = len(parent.site.segments)
    if len(site.segments) <= depth:
        return False
    role = site.segments[depth]
    if role == "a":
        return not isinstance(parent.site.node.a, Load)
    # A computed B CONE (a zero-axis projection) is evaluated by the same fill per slab cell, its
    # statistic with it; an inline B fold WITH an axis is a real nested schedule site.
    return role == "b" and all(isinstance(ch.b, Fold) and ch.b.axis is None for ch in parent.site.node.channels if isinstance(ch.b, Fold))


def _band_of(plan: ReducePlan) -> Workers | None:
    """The inventory a reduce partition implies — the 1-D cooperative band, or ``None`` (a serial /
    register-ILP fold keeps the derived per-cell launch geometry)."""
    return Workers(kind="thread", units=(plan.coop, 1)) if plan.coop > 1 else None


# --- the contraction: tile x stage x reduce ---


def _resolve_stage(term: _Term, node, tile: TilePlan, want: Stage | None, why: list[str] | None = None) -> Stage | None:
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
        return legal.resolve_fill_stage(node, tile, budget, want.depth if want is not None else 1, inputs=term.tile.inputs, why=why)
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


def _fill_values(term: _Term, node, tile: TilePlan) -> list[Stage | None]:
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
        r = _resolve_stage(term, node, tile, st, why=why)
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
    tile = plan.placed_on(term.place)
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
                legal.enforce(legal.direct_atomic_output(term.tile.outputs), pinned=True)
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
    splittable = k is not None and legal.enforce(legal.splitk_computed_b_site(node), pinned=False)
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
        tail = tuple(projection_tail(term.tile))
        atomic_ok = (
            len(node.channels) == 1
            and (len(tail) == 0 or projection_distributes(tail, (node.acc,)))
            and legal.enforce(legal.direct_atomic_output(term.tile.outputs), pinned=False)
        )
        for sp in splitk_moves():
            if sp.finalize == "atomic" and not atomic_ok:
                continue  # the carrier, projection, or destination cannot realize a direct atomic finalize
            if k % sp.cta == 0 and (k // sp.cta) % step == 0:
                out.append(sp)
    return out


def _contraction_blocks(term: _Term, node, work: Workers | None) -> list[Block]:
    """The contraction's values at ``work``: the tile × stage × reduce legal product, over EITHER
    inhabitant of the ``a`` edge — a materialized ``Load`` (both tiers, every transport) or a
    COMPUTED cone (the warp tier alone, over the mandatory compute fill).

    The product is emitted as one BLOCK per ``(TILE, REDUCE)`` pair rather than one value per
    ``(TILE, REDUCE, STAGE)`` triple. Same catalog calls, same legality calls, same order — only
    the return SHAPE differs, and it is the shape that says what the walk already knew: the
    transport is a free axis over the pair, not a third coupled dimension. A pair whose every stage
    was refused is no block at all, exactly as it used to be no rows at all."""
    pin = term.pin("TILE", node)
    if pin is not None:
        try:
            plans = [resolve_site_tile(pin, work, ReducePlan.parse(term.pin("REDUCE", node) or "", work).coop)]
        except ValueError as e:
            # The pin cannot SPELL against this inventory (a warp atom needs a warp ``WORK``), so
            # the candidate is simply not in ``values(site, work)`` — the same rule every other
            # value follows. A pin that spells against NO inventory is a different failure and
            # :func:`_enumerate` raises it rather than quietly emptying the fork.
            term.pin_error = e
            return []
        term.pin_spelled = True
        for plan in plans:
            if plan.is_warp:
                # A PIN with an indivisible K-step or a gather epilogue RAISES — the same predicates
                # the unpinned catalog above drops on, one home each.
                legal.enforce(legal.warp_atom_target(plan.atom, term.ctx), pinned=True)
                conv = _converting_a(node, plan.atom, term.tile.inputs)
                if not conv:
                    legal.enforce(legal.warp_a_columns(node, plan, term.tile.inputs), pinned=True)
                legal.enforce(legal.warp_k_step(node, plan), pinned=True)
                legal.enforce(legal.fragment_epilogue(term.proj), pinned=True)
                shapes = {**term.tile.inputs, **term.tile.outputs}
                legal.enforce(legal.warp_split_store(projection_tail(term.tile), term.place.free, plan.atom.shape, shapes), pinned=True)
                if _requires_sync_fill(node) or conv:
                    legal.enforce(legal.computed_operand_cover(node, plan.placed_on(term.place), converting_a=conv), pinned=True)
                    legal.enforce(
                        legal.computed_operand_copy_dtype(node, plan.placed_on(term.place), term.tile.inputs, converting_a=conv),
                        pinned=True,
                    )
                # Fully materialized contractions use the ordinary operand-dtype rule. Inline-edge
                # contractions were checked above by the sync copy-dtype rule, which also tolerates
                # scheduler-only fixtures that carry no Tensor metadata.
                elif not legal.enforce(legal.warp_operand_dtype(node, plan, _a_dtype(node, term.tile.inputs)), pinned=False):
                    return []
            elif _requires_sync_fill(node):
                return []  # the compute-filled edge has no scalar realization
            else:
                # The CTA thread budget, raised HERE rather than left to materialization: a pinned
                # tile the hardware cannot launch is a user error, and `Pipeline.run`'s validity
                # retry would otherwise catch the materializer's raise and quietly deploy the next
                # leaf — the pin says yes, the deploy says something else.
                legal.enforce(legal.scalar_block_threads(plan), pinned=True)
    else:
        base = replace(work, producer=0) if work is not None else None
        grouped = term.tiles(node)
        plans = grouped.get(base.spell() if base is not None else "", []) + (grouped.get("", []) if base is not None else [])
    out = []
    for plan in plans:
        for red in _contraction_reduces(term, node, plan):
            pinned = pin is not None
            stages = tuple(
                stage
                for stage in _stage_values(term, node, plan)
                if not (work is not None and work.producer) or legal.enforce(legal.producer_transport(stage), pinned=False)
                if red.needs_split or legal.enforce(legal.paired_fragment_register_budget(node, plan, stage), pinned=pinned)
            )
            if stages:
                out.append(Block({"TILE": plan, "REDUCE": red}, stages))
    return out


def _blocked_plans(term: _Term, parent, node, work: Workers | None) -> list[TilePlan]:
    """Tiles for a contraction evaluated as one block of its enclosing Fold.

    The first contraction produces a block of the enclosing axis; the second consumes that same
    block while covering its output axis exactly.  The rule is expressed only in tile geometry,
    so any Fold tree with this composition gets the same rows.
    """
    pair = _blocked_pair(parent)
    if len(pair) != 2 or not any(node is child for child in pair):
        return []
    if work is None or work.kind != "warp":
        return [TilePlan()]
    projected = list(term.proj)
    states = tuple(parent.combine.results)
    reciprocal = next(
        (stmt for stmt in projected if isinstance(stmt, Assign) and stmt.op.name == "reciprocal" and stmt.args == (states[1],)), None
    )
    if (
        reciprocal is None
        or len(projected) != 2
        or not isinstance(projected[-1], Assign)
        or projected[-1].op.name != "multiply"
        or set(projected[-1].args) != {states[2], reciprocal.name}
    ):
        return []
    if work.units[1] != 1 or not parent.axis.extent.is_static:
        return []
    score, value = pair
    score_mn = term.sched.placed(score, TilePlan()).axes
    value_mn = term.sched.placed(value, TilePlan()).axes
    if (
        score_mn is None
        or value_mn is None
        or not score.axis.extent.is_static
        or not score_mn[0].extent.is_static
        or not value_mn[1].extent.is_static
    ):
        return []
    stream = parent.axis.extent.as_static()
    rows = score_mn[0].extent.as_static()
    columns = value_mn[1].extent.as_static()
    base = term.tiles(score).get(work.spell(), ())
    plans: list[TilePlan] = []
    for candidate in base:
        atom = candidate.atom
        if candidate.units_n != 1 or rows % candidate.tile_m:
            continue
        score_plan = replace(candidate, bk=-(-score.axis.extent.as_static() // atom.atom_k))
        if stream % score_plan.tile_n or columns % atom.atom_n:
            continue
        value_plan = TilePlan(
            atom=atom,
            units=candidate.units,
            regs=(candidate.reg_m, columns // atom.atom_n),
            bk=max(1, score_plan.tile_n // atom.atom_k),
        )
        score_k = score_plan.bk * atom.atom_k
        shared = atom.operand_dtype("a").nbytes * (
            score_plan.tile_m * score_k
            + score_plan.tile_n * score_k
            + score_plan.tile_m * score_plan.tile_n
            + score_plan.tile_n * value_plan.tile_n
        )
        if shared > term.ctx.max_dynamic_smem:
            continue
        plan = score_plan if node is score else value_plan
        if plan not in plans:
            plans.append(plan)
    return plans


def _blocked_child_blocks(term: _Term, site: Site, work: Workers | None, parent: _Node) -> list[Block]:
    plans = _blocked_plans(term, parent.site.node, site.node, work)
    pin = term.pin("TILE", site.node)
    if pin is not None:
        try:
            wanted = resolve_site_tile(pin, work)
        except ValueError as e:
            term.pin_error = e
            return []
        term.pin_spelled = True
        plans = [plan for plan in plans if plan == wanted]
    if any(plan.is_warp for plan in plans):
        term.warp_eligible = True
    return [Block({"TILE": plan}, (None,)) for plan in plans]


def _blocked_tiles(node: _Node, combo: tuple[_Row, ...]) -> tuple[TilePlan | None, TilePlan | None]:
    pair = _blocked_pair(node.site.node)
    found = {
        id(child.site.node): plan if (plan := row.plans.get(child.keys.get("TILE"))) is not None and plan.is_tiled else None
        for child, row in zip(_kids(node), combo, strict=True)
    }
    return tuple(found.get(id(child)) for child in pair) if len(pair) == 2 else (None, None)


def _blocked_parent_blocks(term: _Term, node: _Node, combo: tuple[_Row, ...]) -> list[Block]:
    first, second = _blocked_tiles(node, combo)
    if first is None and second is None:
        return _reduce_blocks(term, node.site.node)
    if not (first is not None and second is not None and first.is_warp and second.is_warp):
        return []
    compatible = (
        first.atom == second.atom
        and first.units == second.units
        and first.reg_m == second.reg_m
        and first.tile_n == second.bk * second.atom.atom_k
    )
    return [Block({"REDUCE": ReducePlan()}, (None,))] if compatible else []


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


def _site_blocks(
    term: _Term,
    current: _Node,
    work: Workers | None,
    parent: _Node | None = None,
    combo: tuple[_Row, ...] = (),
) -> list[Block]:
    """The values ``site`` offers under the chosen inventory, as :class:`Block` rectangles — TYPED
    schedule slices, keyed by family. Dispatch is the two stored-param predicates on the node,
    never the ``AxisRole``.

    The site-tree context travels with it: ``parent`` selects a child tile domain, while ``combo``
    lets a composed Fold verify that its children chose compatible block geometry."""
    site, node = current.site, current.site.node
    if parent is not None and _blocked_pair(parent.site.node):
        return _blocked_child_blocks(term, site, work, parent)
    if node.axis is None:
        return _strip_blocks(term, node)
    if is_contraction(node):
        return _contraction_blocks(term, node, work)
    if _blocked_pair(node):
        return _blocked_parent_blocks(term, current, combo)
    if _fill_realized(parent, site):
        # The one site that offers nothing but the decided empty: its PARENT form realizes the
        # partition itself, so there is no choice left here to spell.
        return [Block({"REDUCE": ReducePlan()}, (None,))]
    return _reduce_blocks(term, node)


# ---- the recursion: one row is a joint assignment across the site tree --------------------------- #


def _spell(value) -> str:
    """A slice's stored spelling — ``""`` is the DECIDED empty (the per-cell tile, the serial fold,
    gmem-direct), never an absent key."""
    return value.spell() if value is not None else ""


@dataclass(frozen=True)
class _Row:
    """One enumerated row — the SPELLED knob dict, plus the two facts the kernel's ONE worker
    inventory derives from: the resolved ``TILE`` slices the row carries and the cooperative
    ``REDUCE`` band it claims. ``derive_inventory`` over exactly those is what ``ops.seal_workers``
    computes at materialization, so :func:`_work_holds` and the seal cannot answer differently.

    The ``TILE`` slices are kept BY KEY, not as a flat tuple: reading a site's slice back out of a
    flattened list by position is how a two-site term silently swaps its sites. No OTHER family's
    resolved slice is carried — the row is the kernel's complete identity and :func:`_materialize`
    re-resolves every slice from its spelling, so a second copy could only ever disagree.

    A row is PARTLY decided: it stands for :attr:`width` candidates, not one. The transport axis of
    the site whose values vary FASTEST stays open in :attr:`stages`, because legality never reads
    it — :func:`_work_holds` and :meth:`union` see ``plans`` and ``coop`` alone — so the filter runs
    once per row instead of once per stage. Leaving the FASTEST site's axis open is what keeps
    emission order untouched: its stage already varied immediately above ``RASTER``, which is where
    the space multiplies it back in."""

    knobs: dict
    plans: dict = field(default_factory=dict)
    coop: int = 1
    #: One ``{key: spelling}`` stamp per still-open ``STAGE`` value, or ``()`` when the row decided
    #: its transport already (every site but the fastest one, which spells it into ``knobs``).
    stages: tuple[dict, ...] = ()

    @property
    def tiles(self) -> tuple:
        """The row's resolved ``TILE`` slices — what the inventory folds out of."""
        return tuple(self.plans.values())

    @property
    def width(self) -> int:
        """How many candidates this row stands for, before ``RASTER`` multiplies through."""
        return len(self.stages) or 1

    @classmethod
    def union(cls, parts: Iterable[_Row]) -> _Row | None:
        """Several rows as ONE — knobs and tile slices unioned, the still-open transport axis
        carried through, the cooperative claim RECONCILED. ``None`` when the parts cannot share one
        inventory.

        The claim is a CONSISTENCY, not a maximum. Since step 7 a ``REDUCE`` value spells no coop
        width — the width lives once in ``WORK`` — so two sites claiming DIFFERENT cooperative
        bands spell identically while naming kernels the wire format cannot tell apart. Folding
        them with ``max`` admitted all of them: on the two-site fixture four child widths rode one
        ``t32`` parent as four byte-identical rows. A serial part claims nothing (``coop == 1``)
        and still composes with any other, which is what lets a nested serial fold sit under a warp
        inventory at all.

        Used at BOTH levels a row is assembled — a site with its children (:func:`_merge`) and the
        forest of a term's root sites (:func:`_term_rows`). They stated the rule differently until
        this existed, and the looser of the two was the one that ran on multi-root terms."""
        knobs: dict = {}
        plans: dict = {}
        stages: tuple[dict, ...] = ()
        coop = 1
        for part in parts:
            knobs.update(part.knobs)
            plans.update(part.plans)
            if part.stages:
                stages = part.stages  # at most one part is open — the fastest site's
            if part.coop > 1:
                if coop > 1 and part.coop != coop:
                    return None  # two sites, two widths, one WORK entry to spell them in
                coop = part.coop
        return cls(knobs=knobs, plans=plans, coop=coop, stages=stages)


def _merge(node: _Node, block: Block, combo: tuple[_Row, ...], *, open_stage: bool) -> list[_Row]:
    """One site's rows: each family's slice spelled at ITS canonical path key (``Sched.key`` spells
    ANY site, so there are no new keys and no new codec), unioned with the child rows — and with
    them the inventory claim, which is a fact about the whole row, never one site's. Empty when the
    sites cannot share ONE inventory (:meth:`_Row.union` owns that rule).

    ``open_stage`` decides whether the block's transport axis stays open on the row (ONE row, of
    :attr:`_Row.width` candidates) or is spelled out into one row per stage. Only the site whose
    values vary fastest may leave it open — see :class:`_Row`."""
    red = block.values.get("REDUCE")
    tile = block.values.get("TILE")
    key = node.keys.get("STAGE")
    stamps = tuple({key: _spell(stage)} if key is not None else {} for stage in block.stages)
    own = _Row(
        knobs={k: _spell(block.values.get(family)) for family, k in node.keys.items() if family != "STAGE"},
        plans={node.keys["TILE"]: tile} if tile is not None and "TILE" in node.keys else {},
        coop=red.coop if red is not None else 1,
        stages=stamps if open_stage else (),
    )
    row = _Row.union((own, *combo))
    if row is None:
        return []
    return [row] if open_stage else [replace(row, knobs={**row.knobs, **stamp}) for stamp in stamps]


def _rows_at(term: _Term, node: _Node, work: Workers | None, parent: _Node | None = None, *, open_stage: bool = False) -> list[_Row]:
    """Every row the subtree rooted at ``node`` offers under ``work`` — this site's values crossed
    with each child's own rows. The children are enumerated ONCE per inventory, not once per parent
    value: under a fixed ``work`` a child's candidates do not depend on what the parent chose (that
    is what choosing the inventory at the root buys). Neither direction has a dependency left —
    :func:`_site_values` reads ``work`` and the parent form, never what the subtree decided — so
    the walk is a clean PRODUCT of the site tree, with no dependent-product escape hatch."""
    children = _kids(node)
    child_rows = [_rows_at(term, c, work, node) for c in children]
    out: list[_Row] = []
    for combo in product(*child_rows):
        for block in _site_blocks(term, node, work, parent, combo):
            out.extend(_merge(node, block, combo, open_stage=open_stage))
    return out


def _work_holds(row: _Row, work: Workers | None) -> bool:
    """Whether the row's own slices really imply the inventory it claims — :func:`derive_inventory`
    as the VALIDATION the work-first order turns it into, stated ONCE over the whole row. A serial
    fold or an untiled cell claims nothing, so it composes with any parent inventory; a genuine
    conflict (tiled ``TILE`` workers beside a differing coop width, a producer band with no warp
    inventory) is not co-representable and the row is never built."""
    try:
        return derive_inventory(row.tiles, coop=row.coop, producer=work.producer if work is not None else 0) == work
    except ValueError:
        return False  # the enumerator DROPS what ``seal_workers`` raises on — same rule, one home


def _site_inventories(term: _Term, node: _Node, parent: _Node | None = None) -> list[Workers | None]:
    """Every inventory the subtree rooted at ``node`` can spell a value against. The list is a SET
    of legal candidates; the position an entry lands in carries no meaning."""
    site = node.site
    out: list[Workers | None] = []
    if parent is not None and _blocked_pair(parent.site.node):
        return out
    if site.node.axis is None:
        out.append(None)
    elif is_contraction(site.node):
        out.append(None)  # the derived per-cell geometry — the per-cell tile beside a serial fold
        out.extend(Workers.parse(spell) for spell in term.tiles(site.node) if spell)
        # The non-output-tiled tier folds K across a cooperative band, so a contraction claims
        # those inventories too — at the per-cell tile, where the coop moves are offered.
        out.extend(_band_of(p) for p in _contraction_reduces(term, site.node, TilePlan()))
    else:
        out.extend(_band_of(p) for p in _reduce_specs(term, site.node))
        if _blocked_pair(site.node):
            for units_m in (1, 2, 4):
                work = Workers(kind="warp", units=(units_m, 1))
                pair = _blocked_pair(site.node)
                if all(_blocked_plans(term, site.node, child, work) for child in pair):
                    out.append(work)
    for child in _kids(node):
        out.extend(_site_inventories(term, child, node))
    return out


def _inventories(term: _Term) -> list[Workers | None]:
    """The kernel's ``WORK`` candidates from every site in the stored tree, with ``None`` as the
    per-cell / pure-reduce geometry and producer bands derived from warp inventories."""
    out: list[Workers | None] = []
    seen: set[str] = set()
    for node in term.tree:
        for w in _site_inventories(term, node):
            if (spell := w.spell() if w is not None else "") not in seen:
                seen.add(spell)
                out.append(w)
    if "" not in seen:
        out.append(None)  # a term with no site still maps its placement — one all-empty row
    for w in list(out):
        if w is None or w.kind != "warp":
            continue
        for band in (1, 2):
            spec = WarpSpec(band)
            if legal.enforce(legal.producer_band(spec, w.count * 32), pinned=False):
                out.append(replace(w, producer=band))
    # The live ``WORK`` pin is AUTHORITATIVE, so the pinned inventory is offered whether or not a
    # catalog implies it — the unit widths a ``TILE`` pin reads off it are exactly what no catalog
    # can predict.
    raw = WORK.raw()
    if raw is None:
        return out
    kept = [w for w in out if values_equal(WORK.name, raw, w.spell() if w is not None else "")]
    if kept:
        return kept
    # THE ONE PLACE A PIN DOES NOT NARROW, and it is the PIN-BLEED rule: one env pin, several
    # kernels in the graph, and this is not the one it was written for (a recognition fork's reduce
    # sibling seeing a matmul's warp pin). The catalog's own inventories stay as siblings so the
    # term still maps — emptying the fork would leave a term unmapped over a pin that was never
    # about it, which is the same degrade the strip site applies to a warp ``TILE`` pin it cannot
    # spell. A composed Fold enumerates its own warp geometry, so a ``w<M>x<N>`` pin narrows there
    # like anywhere else.
    # ``test_work_pin_widens_only_where_the_site_offers_no_warp_inventory`` pins both halves.
    logger.warning(
        "WORK pin %r matches no candidate's worker inventory (%s offered); offering it beside the full fork",
        raw,
        ", ".join(repr(w.spell() if w is not None else "") for w in out) or "none",
    )
    return [Workers.parse(raw), *out]


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


def _term_rows(term: _Term, work: Workers | None) -> list[_Row]:
    """The partly decided rows at one inventory - the site product over the term's ROOT
    sites, filtered by the row-level inventory validation. The roots reconcile through the same
    :meth:`_Row.union` a site uses for its children: one rule, whichever level of the tree
    assembles the row.

    The kernel-global ``RASTER`` and the fastest site's ``STAGE`` are NOT closed here: they
    multiply through that filter unconditionally (it reads ``plans`` and ``coop`` alone), so the
    segment carries them as EXTENTS and the space spells them per candidate. That is the whole
    saving - the validation runs once per legal ``(TILE, REDUCE)`` assignment instead of once per
    stage per launch order."""
    roots = term.tree
    out: list[_Row] = []
    for combo in product(*(_rows_at(term, node, work, open_stage=node is roots[-1]) for node in roots)):
        row = _Row.union(combo)
        if row is None or not _work_holds(row, work):
            continue
        out.append(row)
    return out


def _space(term: _Term) -> PoolSpace:
    """Every legal schedule candidate for the stored term, as an addressable space with one
    segment per worker inventory."""
    keys = _keys(term)
    works = _inventories(term)
    #: Every key the tree spells, decided-empty until a row supplies it.
    base = {k: "" for k in keys}
    if term.warp_eligible:
        # ``S_``-prefixed - not a schedule family, so tile identity and prefix-consistency are
        # untouched (``canonical_row_key`` reads the tuning-knob view); it prices "a scalar tile
        # where tensor cores were on offer". It rides the BASE dict rather than a closing pass over
        # the rows: :func:`_inventories` above already asked every site for its tile catalog, which
        # is the one thing that sets the flag, so the answer is known before the first row exists.
        base["S_warp_eligible"] = 1.0
    segments: list[Segment] = []
    for work in works:
        spelled = {WORK.name: work.spell() if work is not None else ""}
        rows = _term_rows(term, work)
        if rows:
            segments.append(Segment.build(rows, spelled, [{RASTER.name: r} for r in _raster_values(term)]))
    return PoolSpace.build(*_decided(keys, base, segments))


def _enumerate(term: _Term, sample=None) -> tuple[list[dict], list[str], int]:
    """The space MATERIALIZED - every legal schedule row in the site value grammar, the fork's site
    keys, and the EXACT size of the space they came from. An empty result is the guardrail
    contract, never a raise: the caller leaves the term unmapped.

    ``sample`` is the Context's ``search.pool.PoolSample``, ``None`` on every live compile. It
    draws its rows out of the space rather than materializing it, which is only possible because
    the space knows its own size and can address a member without building its neighbours.

    This is where :data:`MAX_ROWS` belongs, and the space is what lets it be asked before the
    answer is built: an over-budget term fails on a prefix-sum lookup instead of after 400k dicts.
    A SAMPLED enumeration materializes nothing, so the budget has nothing to bind - but a widened
    product is a real finding, so the size is reported instead of swallowed. The abort dies there;
    the signal does not."""
    space = _space(term)
    total = len(space)
    if total > MAX_ROWS:
        why = (
            f"schedule enumeration for {term.tile.name!r} offers {total} rows, past the {MAX_ROWS}-row "
            f"budget ({len(space.keys)} site keys) - the product across sites widened; "
            f"narrow a catalog or add the legality predicate that bounds it, never truncate"
        )
        if sample is None:
            raise ValueError(why)
        logger.warning("%s. Sampling %d of them.", why, sample.rows)
    rows = list(space) if sample is None else sample.take(space)
    if not rows and term.pin_error is not None and not term.pin_spelled:
        raise term.pin_error  # NO inventory could spell the pin - a pin names a specific kernel
    return rows, list(space.keys), total


def _decided(keys: list[str], base: dict, segments: list[Segment]) -> tuple[list[str], dict, list[Segment]]:
    """The fork's keys, base and segments with the addressed ``REDUCE`` / ``STAGE`` keys NO row
    decides removed.

    A FOLD over the rows, not a scan over the candidates: every row stands for at least one
    candidate and every still-open stage appears in at least one of them, so "does any candidate
    decide this key" is exactly "does any row or any of its open stamps spell it". The trim is
    applied to the base and the rows ONCE, so no candidate is ever built and rebuilt.

    The uniform-key obligation is that every leaf of one fork spells the SAME family keys — not that
    every SITE gets one per family. A site whose partition and transport are not its own to decide
    (the streaming pair's two contractions: their K-step rides ``TILE``, their operands ride the
    stream's ``STAGE``) decides nothing there, and dropping those keys keeps the FEATURIZER honest.
    It reads one node GROUP per distinct ``@<axis>`` element and gives each group the reduce
    geometry when the slice carries a ``REDUCE`` key at all — so a decided-empty ``REDUCE@dd``
    fabricates a partitioned reduce at a site that has none and sum-pools its occupancy into the
    row. Measured on a flash term: ``D_threads`` and ``D_splitk`` tripled and ``D_log2_ctas`` read
    18 instead of 6, which cost the chain and warp forms their cold deploy.

    ``TILE`` keys stay whatever the rows decide: that family is what NAMES the node group, and a
    site offering no tile on this shape is still the site a golden joins against. Bare family keys
    stay too — a bare key is the row's "this family declined" stamp, which the featurizer,
    ``stamp_schedule_families`` and the golden matcher all expect on every row."""
    live: set[str] = set()
    for seg in segments:
        for row in seg.rows:
            live.update(k for k, v in row.knobs.items() if v)
            live.update(k for stamp in row.stages for k, v in stamp.items() if v)
    dead = {k for k in keys if "@" in k and family_of(k) != "TILE" and k not in live}
    if not dead:
        return keys, base, segments

    def trim(d: dict) -> dict:
        return {k: v for k, v in d.items() if k not in dead}

    trimmed = [
        Segment.build(
            [replace(row, knobs=trim(row.knobs), stages=tuple(trim(stamp) for stamp in row.stages)) for row in seg.rows],
            seg.knobs,
            seg.rasters,
        )
        for seg in segments
    ]
    return [k for k in keys if k not in dead], trim(base), trimmed


# ---- materialization: one builder per form, all fed by the same row ------------------------------ #


def _stamp(term: _Term, op, name, knobs: dict, slices, workers=None) -> TileOp:
    """Build the scheduled ``TileOp`` — :func:`ops.scheduled` over this term's placement and root
    stores. The term stays pure algebra; no slice is ever a node field."""

    return scheduled(
        op,
        name=name,
        place=term.place,
        knobs=knobs,
        stores=term.tile.stores,
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
    stores: list[Store] = []
    for i in range(r):

        def rename(n: str, i: int = i) -> str:  # suffix only the body's SSA names; axis vars stay
            return f"{n}__u{i}" if n in ssa else n

        sigma = Sigma({inner.name: BinaryExpr("+", BinaryExpr("*", Var(inner.name), Literal(r, "int")), Literal(i, "int"))})
        for s in op.body:
            s2 = s.rewrite(rename, sigma)
            (loads if isinstance(s2, Load) else computes).append(s2)
        stores.extend(Store(write=st.write.rewrite(rename, sigma)) for st in term.tile.stores)
    new_inner = replace(inner, extent=Dim(inner.extent.as_static() // r))
    new_free = (*term.place.free[:-1], new_inner)
    new_place = Placement(free=new_free, grid=new_free)
    return scheduled(Fold.projection(body=Body((*loads, *computes))), name=name, place=new_place, knobs=knobs, stores=tuple(stores))


def _free_option(term: _Term, plan: TilePlan, name: str, knobs: dict, nested: Sequence[tuple] = ()) -> TileOp:
    """One zero-axis row: the flat per-cell map (also the raw-loop-IR escape's one row), or the
    strip variant when the row's ``TILE`` names a register width. A zero-axis fold with no operands
    has no nested sites, so the strip arm takes none."""
    if _strip_width(plan) > 1:
        return _strip_variant(term, plan, name, knobs)
    return _stamp(term, term.tile.op, name, knobs, nested)


def _node_option(
    term: _Term, node, plan: TilePlan, rplan: ReducePlan, stage: Stage | None, work, name: str, knobs: dict, nested: Sequence[tuple] = ()
) -> TileOp:
    """One un-split row whose compute is a single fold. What it stores is a property of the
    resolved plan, not of a role:

    - an UNTILED output stores its K partition on the node. That is a plain reduce's cooperative /
      ILP band, and equally a non-output-tiled contraction's — the contraction is the degenerate
      carrier of its own additive fold, so ``_factor._tile_reduce_axis`` folds it identically;
    - a TILED output stores its tile + transport instead, and contracts K serially per register
      cell. The two tiers differ only in which budget the tile must fit.

    ``scheduled`` skips a ``None`` slice, so a declined resolver and a serial fold need no guard."""
    if not plan.is_tiled:
        own = [("REDUCE", node, rplan if rplan.stages else None), ("STAGE", node, stage)]
        return _stamp(term, term.tile.op, name, knobs, [*own, *nested])
    legal.enforce(legal.warp_k_step(node, plan) if plan.is_warp else legal.scalar_block_threads(plan), pinned=True)
    # The producer band is INVENTORY, and the enumeration only offered this inventory to rows whose
    # stage can drive it (``_legality.producer_transport``) — so there is nothing left to re-check.
    workers = WarpSpec(work.producer) if work is not None and work.producer else None
    own = [("TILE", node, plan.placed_on(term.place)), ("STAGE", node, stage)]
    return _stamp(term, term.tile.op, name, knobs, [*own, *nested], workers=workers)


def _factor_k(k_axis: Axis, w: int) -> tuple[Axis, Axis, Sigma]:
    """Factor a STATIC contraction axis into ``ksplit × kslice``. ``ksplit`` (extent ``w``, name
    ``<k>_ks``) becomes the outer :class:`Fold`'s reduce axis, parallelized across CTAs and summed
    in the finalize; ``kslice`` (extent ``K/w``, the ORIGINAL name) stays the inner contraction's.
    The ``sigma`` maps the original ``k`` to ``ksplit·(K/w) + kslice`` so the operand loads
    reconstruct the absolute index; distinct names are what avoid a double-reduce."""
    legal.enforce(legal.splitk_width(k_axis, w), pinned=True)
    b = k_axis.extent.as_static() // w
    # LEADING UNDERSCORE, and it is load-bearing: ``normalize_body``'s ``canonicalize_free_axis_order``
    # sorts a body's outer free-loop chain by axis NAME, so a partition axis spelled ``a3_ks`` sorts
    # BELOW the row / column axes it must dominate and ``hoist_loop_invariants`` then sinks it between
    # the column sweep and the K fold — a shape ``bind_prologue_contraction`` cannot parse, which
    # costs a re-recognized split piece its computed-A binding (and its warp rows). ``_`` sorts ahead
    # of every ``aN``, keeping the partition a LEAD grid axis, the same convention the residual path's
    # ``_ksplit`` already relies on.
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


def _splitk_option(term: _Term, plan: TilePlan, node, rplan: ReducePlan, name: str, knobs: dict, nested: Sequence[tuple] = ()) -> TileOp:
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
    # The enumeration asks the same question with ``pinned=False`` (a dropped row); asked again
    # here because a PINNED split never goes through it, and a computed-B cone the σ-reindex cannot
    # carry must raise rather than silently mis-lower.
    legal.enforce(legal.splitk_computed_b_site(node), pinned=True)
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
    return _stamp(term, op, name, knobs, [("REDUCE", outer, rplan), ("TILE", inner, plan.placed_on(term.place)), *nested])


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

    # The row's own keys — spelled ONCE, when the site tree was built. A family the site does not
    # carry keys the BARE name, which is the decided empty every row spells.
    #
    # ONE root, and it is checked rather than assumed: ``_term_rows`` products over EVERY root of
    # ``term.tree``, so a second root would contribute knobs to the row and then be dropped here —
    # its nested slices never stamped, form dispatch reading the wrong node, and both silently. No
    # live term has one; if one appears, this says so instead of mis-materializing.
    if len(term.tree) > 1:
        raise ValueError(
            f"{term.tile.name!r}: {len(term.tree)} root site trees — materialization stamps ONE. "
            "Walk the forest here (as _term_rows does) before producing this shape."
        )
    root = term.tree[0] if term.tree else None
    keys = root.keys if root is not None else {}

    def value(family: str) -> str:
        return row.get(keys.get(family, family), "") or ""

    site = root.site if root is not None else None
    nested = _nested_slices(term, root, row, work) if root is not None else []
    if site is None or site.node.axis is None:
        return _free_option(term, resolve_site_tile(value("TILE"), work), name, op_knobs, nested)
    node = site.node
    rplan = ReducePlan.parse(value("REDUCE"), work)
    # An empty spelling is a unit register tile only when THIS root owns a TILE site. A
    # nested-only term has no root TILE key at all; borrowing its shared thread inventory there
    # invents a slice the codec cannot address.
    plan = resolve_site_tile(value("TILE"), work, rplan.coop) if "TILE" in keys else TilePlan()
    if is_contraction(node) and rplan.needs_split:
        return _splitk_option(term, plan, node, rplan, name, op_knobs, nested)
    stage = _stage_of(term, node, plan, value("STAGE")) if value("STAGE") else None
    return _node_option(term, node, plan, rplan, stage, work, name, op_knobs, nested)


def _nested_slices(term: _Term, node: _Node, row: dict, work: Workers | None) -> list[tuple]:
    """Every NESTED site's resolved slices, as the ``(family, node, value)`` triples ``scheduled``
    keys — materialization's half of the recursion :func:`_rows_at` already does.

    The enumeration walks the whole site tree, so a row DECIDES every site; stamping the root alone
    left a nested key as a knob no kernel realized — the row said ``REDUCE@j=r2`` and the op's
    schedule came back empty. The walk descends through :func:`_kids`, the same accessor
    ``_rows_at`` uses, so what materializes is what was enumerated. A site whose value is the decided empty resolves
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
        stage = _stage_of(term, cnode, plan, spec("STAGE")) if spec("STAGE") else None
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


def _stage_of(term: _Term, node, plan: TilePlan, spec: str) -> Stage | None:
    """The row's ``STAGE`` re-resolved against the node — the operand pipeline on a tiled
    contraction, the shared ROW buffer on any other fold, dispatched by the same predicate the
    enumeration used. The row carries what the enumeration RESOLVED, so this reproduces the slice
    the leaf identity was built from, through :func:`_resolve_stage`'s one dispatch."""
    if not spec:
        return None
    if not is_contraction(node):
        return _row_stage(term, node)
    if not plan.is_tiled:
        return None
    return _resolve_stage(term, node, plan.placed_on(term.place), Stage.parse(spec))


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
    rows: tuple
    #: The size of the SPACE the rows came from - ``len(rows)`` unless the Context asked for a
    #: sample. Memoized with them because it is the same pure function of the term, and because a
    #: rank is only interpretable next to what it was ranked among.
    total: int

    @classmethod
    def build(cls, rows: list[dict], keys: list[str], total: int) -> _Pool:
        return cls(tuple(keys), tuple(MappingProxyType(r) for r in rows), total)


def _dtype_fingerprint(tile: TileOp) -> tuple[str, ...]:
    """The operand dtypes as the enumeration reads them — each term ``Load``'s buffer dtype in
    first-use walk order, plus the output dtypes. NAME-FREE (a buffer's graph id never enters),
    so two same-shape kernels still share a pool, while an f16 and an f32 trace of one shape —
    equal terms, different atom eligibility — key apart. Explicit rather than via the stamped
    ``S_dtype_*`` knobs because not every path that reaches scheduling carries the stamps."""
    seen: set[str] = set()
    out: list[str] = []

    def note_stmt(s) -> None:
        if isinstance(s, Fold):
            walk(s)
            return
        if isinstance(s, Load) and s.input not in seen:
            seen.add(s.input)
            t = tile.inputs.get(s.input)
            out.append(str(t.dtype) if t is not None else "?")
        for b in s.nested():
            for c in b:
                note_stmt(c)

    def walk(node) -> None:
        if not isinstance(node, Fold):
            return
        for e in node.operands:
            note_stmt(e)
        for s in node.lift.body:
            note_stmt(s)

    walk(tile.op)
    return (*out, "->", *(str(t.dtype) for t in tile.outputs.values()))


def deploy_identity(tile: TileOp) -> str:
    """The verified-tier join key — the recognized term's α/buffer-invariant algebra digest
    (:meth:`TileOp.structural_key`) folded with the operand/output dtype fingerprint and the
    axis-extent fingerprint the term deliberately omits (:func:`_extent_fingerprint` — static
    sizes and symbolic markers, never hints). A golden record derives the SAME key from its own
    persisted program through the shared total lift (``_lift.lift_tile``), so the
    join is exact structural identity — no classified shape, no matching heuristic. Unlike
    :func:`pool_key` it excludes knobs, symbolic hints and live pins: identity is what the
    kernel IS; the strict row decode (exact spelled-row equality) is what guarantees a record
    still realizes."""
    return digest(tile.structural_key(), _dtype_fingerprint(tile), _extent_fingerprint(tile))


def pool_key(tile: TileOp) -> str:
    """The pool cache key — everything the enumeration reads that the Context does not pin.
    ``tile.cache_key()`` covers the term (the bottom-up ``structural_key``) and the knobs; the
    three identity-excluded inputs are folded in explicitly — the operand/output dtypes
    (:func:`_dtype_fingerprint` — the atom-eligibility input the term deliberately omits), the
    symbolic-axis hints (:func:`_hint_fingerprint`) and the live env pins
    (:func:`schedule_pin_fingerprint`). The ctx facts (target, smem cap, TMA, the f16acc gate)
    need no key part: the cache lives ON the Context, so one instance never spans two fact
    sets."""
    return digest(tile.cache_key(), _dtype_fingerprint(tile), _hint_fingerprint(tile), schedule_pin_fingerprint())


def schedule(tile: TileOp, name: str, knobs: dict, ctx) -> Fork | list[TileOp] | TileOp:
    """Map a freshly-recognized (UNMAPPED) ``tile`` onto the grid and offer its scheduling fork.

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
    key = digest(pool_key(tile), sample.key if sample is not None else "") if cache is not None or sample is not None else None
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
        return Level((key,), key=lambda r: (r.get(key, ""),))

    levels = [_level(WORK.name), *(_level(k) for k in pool.keys), _level(RASTER.name)]
    return build_fork_tree(params=list(pool.rows), levels=levels, materialize=materialize)


__all__ = ["FAMILIES", "MAX_ROWS", "deploy_identity", "pool_key", "schedule"]
