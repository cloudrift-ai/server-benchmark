r"""Schedule a recognized (UNMAPPED) ``TileOp`` — the generic row enumerator.

**Every role emits rows through ONE recursive walk of the site tree; no role builds ``TileOp``\ s
directly, and no term shape gets its own path.** A row is a joint assignment across every scheduling
SITE of a term, and the tree that generates it is the term's own:

.. code-block:: text

    for work in _inventories(term)            # the kernel's ONE inventory, CHOSEN at the root
        for raster in _raster_values(term)    # kernel-global, like work
            for row in _rows_at(root, work)   # the product over the site tree

    _rows_at(site, work) = for value in _site_values(site, work)      # RESOLVED against work
                             for combo in product(_rows_at(child, work) for child in children)
                               _merge(site, value, combo)             # spells each slice ONCE

``WORK`` leads because the codec says so: :meth:`TilePlan.parse` and :meth:`ReducePlan.parse` both
take the inventory as an INPUT — a ``TILE`` value's unit widths and a ``REDUCE`` value's coop width
are READ OFF it — so the dependency runs work → slice, and a candidate that cannot spell against the
chosen inventory is simply not in ``_site_values(site, work)``. :func:`derive_inventory` stays, as
the VALIDATION that a row's own slices imply the inventory it claims.

Three layers, each with one job:

- the candidate DOMAIN is generated from its bounds in ``search/space.py`` (the tile spaces) or
  listed in its catalog there (the families with no multiplicative coupling — stages, split widths,
  the coop partitions, the raster orders);
- per-node LEGALITY — what a domain cannot know because it depends on this term's K, N, dtype and
  smem cap — is :mod:`._legality`, one predicate per rule, raise-vs-drop chosen by ``pinned``;
- THIS module chooses: which families a SITE offers, the conservative option-0 each leads with, and
  how a row becomes a ``TileOp``.

**Dispatch is two stored-param predicates on the node, never the** :class:`AxisRole`: ``axis is
None`` selects the register-strip values, :func:`is_contraction` the tile × stage × reduce product,
and everything else falls through to the reduce partition. The role is a LOOP annotation and a
materializer read; it never selects a catalog here.

Scope of THIS cut — the SINGLE-SITE terms, whose operand edges are all MATERIALIZED:

- the pointwise cell: the register-strip ladder (``TILE=f<r>``, a TERM VARIANT applied at
  materialization);
- the reduce partition (``REDUCE``): the conservative heuristic pick, then the coop / ILP catalog;
- the contraction: the ``TILE × STAGE × REDUCE × RASTER`` legal product over the scalar and warp
  (mma) tiers, split-K rows routing through the structural ``Fold ⊃ Fold`` composition that
  ``030_split_reduce`` consumes.

A term this cut cannot schedule — a COMPUTED operand edge (the fused norm→linear / gate⊗up cone) or
the flash streaming pair — yields NO rows, and ``020_schedule`` leaves it unmapped rather than
guessing. That is the guardrail contract: empty enumeration returns ``[]``, never raises.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from itertools import product
from math import prod
from types import SimpleNamespace

from emmy.compiler.dim import DEFAULT_SEQ_HINT, Dim
from emmy.compiler.ir.atom import atoms_for
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
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
from emmy.compiler.ir.stmt import Assign, Body, Lambda, Load, Loop, Stmt, Write
from emmy.compiler.ir.stmt.algebra import M
from emmy.compiler.ir.stmt.passes import has_contraction_tail, projection_distributes
from emmy.compiler.ir.tile import Fold, Placement, Store, TileOp
from emmy.compiler.ir.tile.ir import is_contraction
from emmy.compiler.ir.tile.ops import Sched, projection_tail, scheduled
from emmy.compiler.ir.tile.path import Site, family_sites, sites
from emmy.compiler.pipeline.fork import Fork, Level, build_fork_tree
from emmy.compiler.pipeline.knob import canon_family_value, family_of, values_equal
from emmy.compiler.pipeline.passes.lowering._addr import gmem_row_stride
from emmy.compiler.pipeline.passes.lowering.tile import _legality as legal
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

logger = logging.getLogger(__name__)

#: The per-site schedule families this enumeration decides, in the order their keys lead the fork
#: levels. ``WORK`` and ``RASTER`` are kernel-global and bracket them; ``PLACE`` is the seam
#: family — resolved from routing goldens / pins, never enumerated here.
FAMILIES = ("TILE", "STAGE", "REDUCE")

#: The most rows one kernel's enumeration may produce. The product across sites is GENERATED, so a
#: term that widens it silently would hand the search a space it cannot walk and the prior a
#: feature space it cannot cover. Exceeding it is a LOUD failure, never a truncation — a truncated
#: enumeration reads as "covered everything" while dropping whichever rows the walk reached last.
#: Measured headroom: the widest live term (a static f16 square matmul, both tiers, every stage /
#: split / raster) enumerates ~133k rows.
MAX_ROWS = 400_000


# ---- pin reads ---------------------------------------------------------------------------------- #


def _pin(knob, key: str) -> str | None:
    """The live env pin for one SITE KEY — ``EMMY_KNOBS``'s ``FAMILY@<element>`` entry, falling back
    to the bare ``EMMY_<FAMILY>``. A family whose pin is unset reads ``None``, which is the
    distinction the enumeration keys on: an unset ``TILE`` offers the catalog, a set one is
    authoritative."""
    element = key.split("@", 1)[1] if "@" in key else None
    return knob.raw() if element is None else knob.narrow_at(element)


def _narrow_work(options: list[Workers | None]) -> list[Workers | None]:
    """Narrow the inventory candidates by the live ``WORK`` env pin — authoritative, so the pinned
    inventory is offered whether or not a catalog implies it (a pin names a specific kernel, and
    the unit widths a ``TILE`` pin reads OFF it are exactly what no catalog can predict).

    A pin no candidate matches keeps the catalog's own inventories as siblings, with a warning:
    the term the pin was written for may not be the one being scheduled (a recognition fork's
    reduce sibling sees a matmul's warp pin), and emptying its fork would leave it unmapped."""
    raw = WORK.raw()
    if raw is None or not options:
        return options
    kept = [w for w in options if values_equal(WORK.name, raw, w.spell() if w is not None else "")]
    if kept:
        return kept
    logger.warning("WORK pin %r matches no candidate's worker inventory; offering it beside the full fork", raw)
    return [Workers.parse(raw), *options]


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


def _free_cells(place: Placement) -> int:
    """How many output cells the kernel's free grid covers (hint-resolved)."""
    return prod(_hint_extent(a) for a in place.free) if place.free else 1


def _inner_free(place: Placement) -> Axis | None:
    """The innermost NON-UNIT free axis — the m1 recognizer's synthesized unit axis can sit
    innermost, and it is not the axis the transposed emitter sweeps."""
    if not place.free:
        return None
    return next((a for a in reversed(place.free) if not (a.extent.is_static and a.extent.as_static() == 1)), None)


def _matvec_b_kstride(term: _Term, carrier) -> int | None:
    """B's gmem stride along the reduce axis at the per-cell MATVEC tier, or ``None`` when no
    layout gate applies. A contraction demoted to PLANAR carries BOTH a vector operand (a load
    along the reduce axis touching no non-unit free axis — A) and a matrix operand indexed by the
    reduce axis AND a non-unit free axis (B); only that two-operand shape is gated. ``1`` means the
    reduce axis is B's fastest-varying dimension (the serving ``F.linear`` N×K layout); ``>1`` is
    k-major (canonical ``B[k, n]``)."""
    nonunit = {a.name for a in term.place.free if not (a.extent.is_static and a.extent.as_static() == 1)}
    k_name = carrier.axis.name
    a_seen = False
    strides = set()
    for ld in _node_loads(term.tile.op):
        used = set().union(*(e.free_vars() for e in ld.index)) if ld.index else set()
        if k_name not in used:
            continue
        if used & nonunit:
            strides.add(gmem_row_stride(ld, k_name, term.tile.inputs))
        else:
            a_seen = True
    return strides.pop() if a_seen and len(strides) == 1 else None


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
    is a RESOLVER, not a choice: the row spells ``d1/sync`` exactly when the shape carries an
    operand the CTA can hold as a shared row across the reduce and its contraction tail, and the
    materializer re-resolves the same buffer off the same term."""
    tail = projection_tail(term.tile)
    if not has_contraction_tail(tail):
        return None
    grid_vars = tuple(Var(a.name) for a in term.place.grid)
    carrier_loads = [ld for ld in _node_loads(node) if ld.is_scalar]
    buf = _shared_row_buf(carrier_loads, tail, grid_vars, node.axis, term.tile.inputs)
    return Stage(transport="sync", smem=(buf,)) if buf is not None else None


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


def _is_site(s: Site, per_family: dict[str, tuple[Site, ...]]) -> bool:
    return any(s in fam for fam in per_family.values())


def _site_tree(op, sched: Sched) -> tuple[tuple[_Node, ...], dict[str, tuple[Site, ...]]]:
    """``op``'s scheduling sites as a TREE — the topmost ones first, each carrying the sites nested
    under it. The walker is ``path.sites``; this only groups its output by containment, so a term
    shape never gets a site list of its own."""
    all_sites = sites(op)
    per_family = {f: family_sites(f, all_sites) for f in FAMILIES}
    sched_sites = [s for s in all_sites if _is_site(s, per_family)]

    def under(parent: Site, child: Site) -> bool:
        return len(child.segments) > len(parent.segments) and child.segments[: len(parent.segments)] == parent.segments

    def build(parent: Site | None) -> tuple[_Node, ...]:
        kids = [s for s in sched_sites if (under(parent, s) if parent is not None else True)]
        tops = [s for s in kids if not any(t is not s and under(t, s) for t in kids)]
        return tuple(
            _Node(
                site=s,
                keys={f: k for f in FAMILIES if (k := sched.key(f, s.node)) is not None},
                children=build(s),
            )
            for s in tops
        )

    return build(None), per_family


def _keeps_children(site: Site) -> bool:
    """Whether the site's nested sites stay sites under the values this cut offers. Only a
    CONTRACTION keeps them: its COMPUTED operand edges carry their own families (a MATERIALIZED
    operand is not a site — its transport is the parent's ``STAGE``). Under every other tier the
    fold's edges lower INLINE in its body, so they are not separately scheduled — which is what
    leaves flash's hoisted QK edge and its derived PV to the cut that gives the streaming site its
    own values."""
    return is_contraction(site.node)


# ---- the candidate values, per site ------------------------------------------------------------- #


class _Term:
    """Everything the enumeration reads about ONE term — the op, its grid placement, the target and
    the key speller — plus the per-site catalogs, built once and grouped by the inventory each
    candidate implies (the enumeration visits every site once per inventory)."""

    def __init__(self, tile: TileOp, place: Placement, ctx) -> None:
        self.tile = tile
        self.place = place
        self.ctx = ctx
        self.sched = Sched(tile.op, {}, place=place)
        self.proj = _projection(tile.op)
        self.tree, self.per_family = _site_tree(tile.op, self.sched)
        self._tiles: dict[int, dict[str, list[TilePlan]]] = {}
        #: The refusal a schedule PIN drew, kept until the walk is done. One inventory declining
        #: a pin is ordinary (the widths are read OFF the inventory, so the pin names a different
        #: plan under each); a pin NO inventory could spell is malformed, and that is loud.
        self.pin_error: ValueError | None = None
        self.pin_spelled = False
        #: Set when any site offers a tensor-core row — a structural fact about the KERNEL, stamped
        #: on EVERY row so the priors can price "a scalar tile where tensor cores were on offer".
        self.warp_eligible = False

    def pin(self, family: str, node) -> str | None:
        key = self.sched.key(family, node)
        return None if key is None else _pin({"TILE": TILE, "STAGE": STAGE, "REDUCE": REDUCE}[family], key)

    def tiles(self, node) -> dict[str, list[TilePlan]]:
        """The contraction node's ``TILE`` catalog, grouped by the ``WORK`` spelling each candidate
        implies (``""`` for the untiled per-cell tile, which composes with any inventory a
        cooperative ``REDUCE`` claims)."""
        if id(node) not in self._tiles:
            self._tiles[id(node)] = self._build_tiles(node)
        return self._tiles[id(node)]

    def _build_tiles(self, node) -> dict[str, list[TilePlan]]:
        atoms = _warp_atoms(self, node)
        warp = [p for p in warp_tile_moves(atoms) if legal.enforce(legal.warp_k_step(node, p), pinned=False)] if atoms else []
        self.warp_eligible = self.warp_eligible or bool(warp)
        grouped: dict[str, list[TilePlan]] = {}
        for plan in scalar_tile_moves() + warp:
            w = plan_workers(plan)
            grouped.setdefault(w.spell() if w is not None else "", []).append(plan)
        # A ``TILE`` pin is authoritative over the VALUES but not over the inventories: its unit
        # widths are read OFF ``WORK``, so the pin names a different plan under each one and is
        # re-resolved per inventory in :func:`_contraction_values`. The catalog still answers
        # "which inventories can this site spell against".
        return grouped


def _f16acc_allowed(ctx) -> bool:
    """Whether the f16-accumulate atom forks may be OFFERED — a precision-trading gate, off by
    default: the precise ``F16_MMA_F32_ACC`` pin is authoritative on every target; unset, the
    ``FAST_MATH`` umbrella offers it on the consumer dies (``Context.f16acc_is_faster``) where the
    f32-accumulate half-rate nerf makes it profitable. A ``TILE`` pin naming the atom bypasses this
    gate entirely (pins are authoritative)."""
    from emmy.compiler.pipeline.search.space import F16_MMA_F32_ACC, precision_pin  # noqa: PLC0415

    raw = F16_MMA_F32_ACC.raw()
    if raw is not None:
        return F16_MMA_F32_ACC.parse(raw)
    return precision_pin(F16_MMA_F32_ACC) and ctx.f16acc_is_faster


def _warp_atoms(term: _Term, node) -> tuple[str, ...]:
    """The dtype-eligible tensor-core atom names for this contraction, ``()`` when the warp tier
    doesn't apply (a non-16-bit operand dtype, a COMPUTED A edge, or a fragment-unrealizable gather
    epilogue), extended with the f16-accumulate siblings when :func:`_f16acc_allowed`. Reads pure
    algebra off the STORED node — the placement / tile would be unread."""
    inputs = term.tile.inputs
    if not inputs or legal.fragment_epilogue(term.proj) is not None:
        return ()
    if not isinstance(node.a, Load):
        return ()  # a computed cone is out of this cut's scope (see the module docstring)
    t = inputs.get(node.a.input)
    ab = t.dtype if t is not None else None
    atoms = atoms_for(ab)
    if not atoms or not _f16acc_allowed(term.ctx):
        return atoms
    return atoms + atoms_for(ab, acc=ab)  # the f16-accumulate siblings, registry order preserved


# --- the pointwise cell: the register strip ---


def _strip_width(plan: TilePlan) -> int:
    """The strip ratio ``r`` a strip row's ``TILE`` names — the inner register width. A warp codec
    names none (there is no fragment on a pointwise cell), so it reads ``0`` and is dropped."""
    return 0 if plan.is_warp else plan.regs[0]


def _strip_values(term: _Term, node) -> list[dict]:
    """The register-strip values: the flat per-cell tile (option-0), then the catalog's ladder.
    ``r`` IS the spelled ``TILE=f<r>`` — the strip is a TERM VARIANT applied at materialization, a
    function of the ROW, not a member of a pre-enumerated variant set."""
    pin = term.pin("TILE", node)
    ext = term.place.free[-1].extent.as_static() if _strippable(term) else 0
    plans = [resolve_site_tile(pin, None)] if pin is not None else [TilePlan(), *map_tile_moves()]
    out = []
    for plan in plans:
        # A strip WIDTH the cell cannot carry (a stateful / sweep body, a symbolic or indivisible
        # inner extent, a warp codec on a pointwise cell) drops the row; the flat per-cell base
        # below is always offered, so a narrowing pin degrades to option-0.
        if legal.enforce(legal.strip_width(ext, _strip_width(plan)), pinned=False):
            out.append({"TILE": plan})
    return out or [{"TILE": TilePlan()}]


# --- the reduce partition ---

# Conservative cooperative-reduce selection constants (the default when REDUCE is unpinned).
_COOP_MIN_EXTENT = 128  # only cooperate when the reduce axis is at least this wide
_SERIAL_TARGET = 8  # aim for ~this many serial steps per cooperating thread
_MAX_COOP = 256  # cap on cooperative threads per CTA (power of two)
_FREE_CAP = 256  # only cooperate when the output grid is at most this many cells


def _prevpow2(n: int) -> int:
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def _pick_coop(extent: int, free: int, *, has_tail: bool = False) -> int:
    """The conservative whole-CTA cooperative-thread count for a reduce of static ``extent`` over
    ``free`` output cells, or ``1`` (stay serial). Cooperate only on a wide reduce feeding a small
    grid — otherwise the scalar tier already saturates the GPU; ``has_tail`` lifts the grid cap (a
    fused contraction tail multiplies each cell's work by its column extent)."""
    if extent < _COOP_MIN_EXTENT or (free > _FREE_CAP and not has_tail):
        return 1
    coop = min(_prevpow2(extent // _SERIAL_TARGET), _MAX_COOP)
    return coop if coop >= 2 else 1


def _reduce_specs(term: _Term, node) -> list[ReducePlan]:
    """The reduce-partition candidates for a non-contraction fold — option-0 is the conservative
    heuristic pick (:func:`_pick_coop`, so a cold greedy compile keeps its historical deploy), then
    the legal :func:`coop_reduce_moves` catalog + serial as fork siblings. The catalog rows are what
    keep the 16- / 32-wide reduce goldens reachable. An env pin is authoritative."""
    pin = term.pin("REDUCE", node)
    if pin is not None:
        return [ReducePlan.parse(pin, _coop_work(pin))]
    extent = _hint_extent(node.axis)
    free = _free_cells(term.place)
    tail = projection_tail(term.tile)
    coop = _pick_coop(extent, free, has_tail=has_contraction_tail(tail))
    # The layout gate (WS5, the cold-poison hardening): at the matvec tier the coop bands are only
    # coalesced on ONE B orientation — the plain band interleaves lanes along K, the transposed
    # band sweeps lanes along the output axis. Enumeration is the single choke point every tier
    # resolves through, so the gate lives here; an env pin stays authoritative and un-gated.
    kstride = _matvec_b_kstride(term, node)
    if kstride == 1 and extent >= _COOP_MIN_EXTENT and free >= _FREE_CAP:
        # The wide-K COALESCED matvec: a 32-wide band is the measured deploy, and option-0 is what
        # the prior-free paths take. It is a RANKING decision, so it leads the list rather than
        # replacing it — the row SET stays a function of the term alone, never of a context flag.
        coop = 32
    elif coop > 1 and kstride is not None and kstride != 1:
        coop = 1  # the heuristic option-0 is a plain band too — uncoalesced on a k-major B
    cands = [ReducePlan.of(coop=coop)]
    tail_scalar = not any(isinstance(s, Loop) for s in tail)
    inner = _inner_free(term.place)
    k_static = node.axis.extent.as_static() if node.axis.extent.is_static else None
    bt_ok = (
        k_static is not None
        and tail_scalar
        and inner is not None
        and inner.extent.is_static
        and inner.extent.as_static() % 32 == 0
        and _row_stage(term, node) is None
    )
    for p in coop_reduce_moves():
        if p.coop_transposed or p.needs_split:
            if not (bt_ok and p.coop_transposed and p.coop % 32 == 0 and (not p.needs_split or k_static % p.cta == 0)):
                continue
            if kstride == 1:
                continue  # the transposed band lane-sweeps the output axis — uncoalesced there
        elif p.coop > 1 and kstride is not None and kstride != 1:
            continue  # the plain band interleaves lanes along K — uncoalesced on a k-major B
        if p.coop <= extent and p.reg <= extent and p not in cands:
            cands.append(p)
    if ReducePlan() not in cands:
        cands.append(ReducePlan())
    return cands


def _coop_work(spec: str | None) -> Workers | None:
    """The inventory a cooperative ``REDUCE`` pin spells against — the pin names a width only
    through ``WORK``, so a bare ``coop`` pin is read against the pinned inventory."""
    return Workers.parse(WORK.raw()) if WORK.raw() else None


def _reduce_values(term: _Term, node, work: Workers | None) -> list[dict]:
    """The reduce-partition values at ``work``: the partition itself plus the shared-row ``STAGE``
    a cooperative band can drive (a resolver, not a choice — see :func:`_row_stage`)."""
    out = []
    for plan in _reduce_specs(term, node):
        if _band_of(plan) != work:
            continue  # the candidate cannot spell against the chosen inventory
        stage = _row_stage(term, node) if plan.coop > 1 else None
        out.append({"REDUCE": plan, "STAGE": stage})
    return out


def _band_of(plan: ReducePlan) -> Workers | None:
    """The inventory a reduce partition implies — the 1-D cooperative band, or ``None`` (a serial /
    register-ILP fold keeps the derived per-cell launch geometry)."""
    return Workers(kind="thread", units=(plan.coop, 1)) if plan.coop > 1 else None


# --- the contraction: tile x stage x reduce ---

# Emit unpinned split-K candidates only when the output grid alone leaves the GPU under-occupied.
_SPLITK_MAX_CTAS = 1024


def _tile_area(plan: TilePlan) -> int:
    """The output cells one CTA covers under ``plan`` — the occupancy denominator."""
    am, an = (plan.atom.atom_m, plan.atom.atom_n) if plan.is_warp else (1, 1)
    return max(plan.units_m * plan.reg_m * am * plan.units_n * plan.reg_n * an, 1)


def _stage_values(term: _Term, node, plan: TilePlan) -> list[Stage | None]:
    """The RESOLVED operand stages for one tile candidate — gmem-direct ``None`` first, then every
    catalog move that RESOLVES against the node with this ``plan``, so the leaf identity, the
    stamped knobs and the kernel agree. A pinned ``STAGE`` is authoritative: the resolved pin alone,
    or gmem-direct when it declines."""
    if not plan.is_tiled:
        return [None]  # per-cell / unbindable — no operand slab to stage
    tile = plan.placed_on(term.place)
    budget = term.ctx.max_dynamic_smem

    def resolve(st: Stage) -> Stage | None:
        if st.transport == "tma" and not term.ctx.has_tma:
            return None  # TMA is Hopper+ (sm_90) — decline below it rather than fail to compile
        if plan.is_warp:
            return legal.resolve_warp_stage(node, tile, st, budget)
        return legal.resolve_scalar_stage(node, tile, st, term.tile.inputs, budget)

    pinned = term.pin("STAGE", node)
    if pinned is not None:
        # A malformed pin RAISES through ``Stage.parse`` — this used to be swallowed into
        # gmem-direct, which made it the only silently-ignored pin in the family.
        return [resolve(Stage.parse(pinned)) if pinned else None]
    out: list[Stage | None] = [None]
    spelled = {""}
    for move in stage_moves(warp=plan.is_warp):
        r = resolve(move)
        if r is not None and r.spell() not in spelled:
            spelled.add(r.spell())
            out.append(r)
    return out


def _splitk_reduces(term: _Term, node, plan: TilePlan) -> list[ReducePlan]:
    """The contraction's ``REDUCE`` candidates — serial first (option-0), then the legal coop / ILP
    moves (per-cell tier only — the non-output-tiled contract) and the divisor- and occupancy-
    guarded split-K moves. An ATOMIC split is offered only on a single-channel node whose FULL
    projection tail distributes over the add; the deferred kernel finalize stays legal for any
    epilogue."""
    pin = term.pin("REDUCE", node)
    if pin is not None:
        pinned = ReducePlan.parse(pin, _coop_work(pin))
        if pinned.needs_split:
            return [pinned]
        if pinned.coop > 1 or pinned.reg > 1:
            # A tiled candidate contracts K serially per register cell — the coop / ILP partition is
            # the NON-output-tiled tier's, so a tiled tile has nothing to honor the pin with.
            return [] if plan.is_tiled else [pinned]
        return [ReducePlan()]
    out = [ReducePlan()]
    ext = node.axis.extent
    k = ext.as_static() if ext.is_static else None
    if k is not None and not plan.is_tiled:
        inner = _inner_free(term.place)
        for p in coop_reduce_moves():
            if not (p.coop <= k and p.reg <= k):
                continue
            if p.coop_transposed:
                # The transposed lane swap needs the structure its emitter assumes: a static
                # innermost free axis divisible by the 32-lane sweep and a 32-multiple coop, plus
                # (for a split composite) the split divisibility. Layout is a GATE too (WS5): the
                # band lane-sweeps the output axis, so it is coalesced only on k-major B.
                if not (
                    not node.b_trans
                    and p.coop % 32 == 0
                    and inner is not None
                    and inner.extent.is_static
                    and inner.extent.as_static() % 32 == 0
                    and (not p.needs_split or k % p.cta == 0)
                ):
                    continue
            elif p.coop > 1 and not node.b_trans:
                continue  # the plain band interleaves lanes along K — uncoalesced on canonical B[k, n]
            out.append(p)
    if k is not None and _free_cells(term.place) // _tile_area(plan) <= _SPLITK_MAX_CTAS:
        step = plan.atom.atom_k * plan.bk if plan.is_warp else 1
        tail = tuple(projection_tail(term.tile))
        atomic_ok = len(node.channels) == 1 and (len(tail) == 0 or projection_distributes(tail, (node.acc,)))
        for sp in splitk_moves():
            if sp.finalize == "atomic" and not atomic_ok:
                continue  # a non-distributive projection would raise at 030_split_reduce
            if k % sp.cta == 0 and (k // sp.cta) % step == 0:
                out.append(sp)
    return out


def _contraction_values(term: _Term, node, work: Workers | None) -> list[dict]:
    """The contraction's values at ``work``: the tile × stage × reduce legal product. A COMPUTED
    operand edge is out of this cut's scope and yields none."""
    if not isinstance(node.a, Load):
        return []
    pin = term.pin("TILE", node)
    if pin is not None:
        try:
            plans = [resolve_site_tile(pin, work, term.pin("REDUCE", node) or "")]
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
                legal.enforce(legal.warp_k_step(node, plan), pinned=True)
                legal.enforce(legal.fragment_epilogue(term.proj), pinned=True)
    else:
        base = replace(work, producer=0) if work is not None else None
        grouped = term.tiles(node)
        plans = grouped.get(base.spell() if base is not None else "", []) + (grouped.get("", []) if base is not None else [])
    out = []
    for plan in plans:
        for red in _splitk_reduces(term, node, plan):
            if not _inventory_holds(plan, red, work):
                continue
            for stage in _stage_values(term, node, plan):
                if work is not None and work.producer and not legal.enforce(legal.producer_transport(stage, red), pinned=False):
                    continue
                out.append({"TILE": plan, "STAGE": stage, "REDUCE": red})
    return out


def _inventory_holds(plan: TilePlan, red: ReducePlan, work: Workers | None) -> bool:
    """Whether ``(plan, red)`` really implies the inventory the row claims — :func:`derive_inventory`
    as the VALIDATION the work-first order turns it into. A genuine conflict (tiled TILE workers
    beside a differing coop width, a producer band with no warp inventory) is not co-representable,
    so the combination is never built: ``WORK`` is one kernel's one inventory."""
    try:
        return derive_inventory((plan,), coop=red.coop, producer=work.producer if work is not None else 0) == work
    except ValueError:
        return False  # the enumerator DROPS what ``seal_workers`` raises on — same rule, one home


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


def _site_values(term: _Term, site: Site, work: Workers | None) -> list[dict]:
    """The values ``site`` offers under the chosen inventory — TYPED schedule slices, keyed by
    family. Dispatch is the two stored-param predicates on the node, never the ``AxisRole``."""
    node = site.node
    if node.axis is None:
        return _strip_values(term, node) if work is None else []
    if is_contraction(node):
        return _contraction_values(term, node, work)
    return _reduce_values(term, node, work)


# ---- the recursion: one row is a joint assignment across the site tree --------------------------- #


def _spell(family: str, value) -> str:
    """A slice's stored spelling — ``""`` is the DECIDED empty (the per-cell tile, the serial fold,
    gmem-direct), never an absent key."""
    return value.spell() if value is not None else ""


def _merge(node: _Node, value: dict, combo: tuple[dict, ...]) -> dict:
    """One site's row: each family's slice spelled at ITS canonical path key (``Sched.key`` spells
    ANY site, so there are no new keys and no new codec), unioned with the child rows."""
    row = {key: _spell(family, value.get(family)) for family, key in node.keys.items()}
    for child in combo:
        row.update(child)
    return row


def _rows_at(term: _Term, node: _Node, work: Workers | None) -> list[dict]:
    """Every row the subtree rooted at ``node`` offers under ``work`` — this site's values crossed
    with each child's own rows. The children are enumerated ONCE per inventory, not once per parent
    value: under a fixed ``work`` a child's candidates do not depend on what the parent chose (that
    is what choosing the inventory at the root buys)."""
    child_rows = [_rows_at(term, c, work) for c in (node.children if _keeps_children(node.site) else ())]
    return [_merge(node, value, combo) for value in _site_values(term, node.site, work) for combo in product(*child_rows)]


def _site_inventories(term: _Term, node: _Node) -> list[Workers | None]:
    """The inventories the subtree rooted at ``node`` can spell a value against, **its own option-0
    first** — which is what makes the enumeration's leading row the conservative one every
    prior-free path deploys."""
    site = node.site
    out: list[Workers | None] = []
    if site.node.axis is None:
        out.append(None)
    elif is_contraction(site.node):
        out.append(None)  # the per-cell tile and the serial fold — the contraction's option-0
        out.extend(Workers.parse(spell) for spell in term.tiles(site.node) if spell)
        # The non-output-tiled tier folds K across a cooperative band, so a contraction claims
        # those inventories too — at the per-cell tile, where the coop moves are offered.
        out.extend(_band_of(p) for p in _splitk_reduces(term, site.node, TilePlan()))
    else:
        out.extend(_band_of(p) for p in _reduce_specs(term, site.node))
    for child in node.children if _keeps_children(site) else ():
        out.extend(_site_inventories(term, child))
    return out


def _inventories(term: _Term) -> list[Workers | None]:
    """The kernel's ``WORK`` candidates — every inventory the term's own catalogs imply, the
    OPTION-0 one leading (the reduce tier's conservative cooperative band, or ``None`` for the
    per-cell / chain / pure-reduce tiers — a first-class inventory), then the ``+p`` producer bands
    a warp inventory can carry. CHOSEN at the root: every site resolves against it, so three of the
    parent/child couplings stop being rules at all."""
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
            spec = WarpSpec.parse(f"p{band}")
            if legal.enforce(legal.producer_band(spec, w.count * 32), pinned=False):
                out.append(replace(w, producer=band))
    return out


def _level_keys(term: _Term) -> list[str]:
    """The site keys the fork levels between ``WORK`` and ``RASTER``, family by family. A family
    with no decided site keys the BARE name — the decided-empty every row carries, which is what
    keeps one fork's leaves prefix-consistent."""
    decided: dict[str, list[str]] = {f: [] for f in FAMILIES}

    def walk(node: _Node) -> None:
        for family, key in node.keys.items():
            decided[family].append(key)
        for child in node.children if _keeps_children(node.site) else ():
            walk(child)

    for node in term.tree:
        walk(node)
    return [k for family in FAMILIES for k in (decided[family] or [family])]


def _enumerate(term: _Term) -> tuple[list[dict], list[str]]:
    """Every legal schedule row for ``term``, in the site value grammar, plus the fork's site keys.
    An empty result is the guardrail contract, never a raise: the caller leaves the term unmapped."""
    keys = _level_keys(term)
    #: The families no site decided — each stamps the BARE key with the decided empty, so one
    #: fork's leaves all spell the same family keys (an ABSENT key would read as "free" and let a
    #: gmem-direct leaf inherit a staged row's measurement).
    bare = {family: "" for family in FAMILIES if family in keys}
    rasters = _raster_values(term)
    rows: list[dict] = []
    for work in _narrow_work(_inventories(term)):
        spelled = work.spell() if work is not None else ""
        for combo in product(*(_rows_at(term, node, work) for node in term.tree)):
            base = {**bare}
            for part in combo:
                base.update(part)
            for raster in rasters:
                rows.append({**base, WORK.name: spelled, RASTER.name: raster})
            if len(rows) > MAX_ROWS:
                raise ValueError(
                    f"schedule enumeration for {term.tile.name!r} exceeds the {MAX_ROWS}-row budget "
                    f"({len(term.tree)} root sites, {len(keys)} site keys) — the product across sites widened; "
                    f"narrow a catalog or add the legality predicate that bounds it, never truncate."
                )
    if not rows and term.pin_error is not None and not term.pin_spelled:
        raise term.pin_error  # NO inventory could spell the pin — a pin names a specific kernel
    if term.warp_eligible:
        # ``S_``-prefixed — not a schedule family, so tile identity and prefix-consistency are
        # untouched; it prices "a scalar tile where tensor cores were on offer".
        for row in rows:
            row["S_warp_eligible"] = 1.0
    return rows, keys


# ---- materialization: one builder per form, all fed by the same row ------------------------------ #


def _site_knobs(stamped: dict) -> dict:
    """A row's knob dict flipped to the site grammar: TILE/REDUCE values re-spell SITE-LOCAL — the
    worker halves live in the ``WORK`` entry ``seal_workers`` stamps."""
    return {k: canon_family_value(k, v) if isinstance(v, str) and family_of(k) in ("TILE", "REDUCE") else v for k, v in stamped.items()}


def _stamp(term: _Term, op, name, knobs: dict, slices, workers=None) -> TileOp:
    """Build the scheduled ``TileOp`` — :func:`ops.scheduled` with this module's knob grammar
    applied. The term stays pure algebra; no slice is ever a node field."""
    return scheduled(op, name=name, place=term.place, knobs=_site_knobs(knobs), stores=term.tile.stores, slices=slices, workers=workers)


def _strip_variant(term: _Term, plan: TilePlan, name: str, knobs: dict) -> TileOp:
    """The pointwise register-STRIP term variant: hand each thread ``r`` CONTIGUOUS inner-axis
    elements. The inner free axis shrinks to ``extent/r`` (the grid walks it) and the cell body is
    unrolled ``r`` times — copy ``i`` reads/writes ``inner·r + i`` with its SSA names suffixed —
    then regrouped as ``r`` loads · ``r`` computes · ``r`` writes so the unit-stride runs feed
    ``050_vectorize_loads`` / ``080_vectorize_stores``. A different term, hence a different
    ``term_key`` and ``op_cache_key`` — which is why it is applied HERE and not at recognition."""
    inner = term.place.free[-1]
    r = plan.regs[0]
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
    return scheduled(
        Fold.projection(body=Body((*loads, *computes))), name=name, place=new_place, knobs=_site_knobs(knobs), stores=tuple(stores)
    )


def _free_option(term: _Term, plan: TilePlan, name: str, knobs: dict) -> TileOp:
    """One zero-axis row: the flat per-cell map (also the raw-loop-IR escape's one row), or the
    strip variant when the row's ``TILE`` names a register width."""
    if _strip_width(plan) > 1:
        return _strip_variant(term, plan, name, knobs)
    return _stamp(term, term.tile.op, name, knobs, ())


def _reduce_option(term: _Term, node, plan: ReducePlan, stage: Stage | None, name: str, knobs: dict) -> TileOp:
    """One reduce-partition row: the resolved :class:`ReducePlan` stored on the fold, plus the
    shared-row operand :class:`Stage` a cooperative band drives."""
    slices = []
    if plan.stages:
        slices.append(("REDUCE", node, plan))
    if stage is not None:
        slices.append(("STAGE", node, stage))
    return _stamp(term, term.tile.op, name, knobs, slices)


def _tile_option(term: _Term, plan: TilePlan, node, name: str, knobs: dict, stage: Stage | None) -> TileOp:
    """One scalar-tier contraction row. A tiled candidate contracts K serially per register cell, so
    a coop / ILP partition is DROPPED rather than stamped onto a kernel that doesn't fold it."""
    legal.enforce(legal.scalar_block_threads(plan), pinned=True)
    slices: list = []
    if plan.is_tiled:
        slices = [("TILE", node, plan.placed_on(term.place)), ("STAGE", node, stage)]
    return _stamp(term, term.tile.op, name, knobs, slices)


def _coop_contraction_option(term: _Term, node, rplan: ReducePlan, name: str, knobs: dict) -> TileOp:
    """One NON-output-tiled contraction row carrying a coop / ILP K partition — the contraction is
    the degenerate carrier of its own additive fold, so ``_factor._tile_reduce_axis`` folds the
    partition off the node exactly as it does for a plain reduce."""
    return _stamp(term, term.tile.op, name, knobs, [("REDUCE", node, rplan)])


def _warp_option(term: _Term, plan: TilePlan, node, name: str, knobs: dict, stage: Stage | None, work: Workers | None) -> TileOp:
    """One warp (tensor-core) contraction row. The producer band rides ORTHOGONAL to the resolved
    tile/stage: it is gated on the RESOLVED stage, so an ineligible inventory degrades to uniform
    rather than claiming a pipeline that never ran."""
    legal.enforce(legal.warp_k_step(node, plan), pinned=True)
    workers = None
    if work is not None and work.producer:
        spec = WarpSpec.parse(f"p{work.producer}")
        workers = spec if spec.is_legal(SimpleNamespace(stage=stage)) else None
    return _stamp(term, term.tile.op, name, knobs, [("TILE", node, plan.placed_on(term.place)), ("STAGE", node, stage)], workers=workers)


def _factor_k(k_axis: Axis, w: int) -> tuple[Axis, Axis, Sigma]:
    """Factor a STATIC contraction axis into ``ksplit × kslice``. ``ksplit`` (extent ``w``, name
    ``<k>_ks``) becomes the outer :class:`Fold`'s reduce axis, parallelized across CTAs and summed
    in the finalize; ``kslice`` (extent ``K/w``, the ORIGINAL name) stays the inner contraction's.
    The ``sigma`` maps the original ``k`` to ``ksplit·(K/w) + kslice`` so the operand loads
    reconstruct the absolute index; distinct names are what avoid a double-reduce."""
    legal.enforce(legal.splitk_width(k_axis, w), pinned=True)
    b = k_axis.extent.as_static() // w
    ksplit = Axis(name=f"{k_axis.name}_ks", extent=Dim(w))
    kslice = replace(k_axis, extent=Dim(b))
    sigma = Sigma({k_axis.name: BinaryExpr("+", BinaryExpr("*", Var(ksplit.name), Literal(b, "int")), Var(k_axis.name))})
    return ksplit, kslice, sigma


def _splitk_option(term: _Term, plan: TilePlan, node, rplan: ReducePlan, name: str, knobs: dict, stage_spec: str) -> TileOp:
    """One SPLIT-K contraction row — the structural ``Fold(axis=ksplit) ⊃ Fold(axis=kslice)``
    composition ``030_split_reduce`` consumes into the cross-CTA partial + finalize. The inner node
    is the SAME contraction a non-split matmul builds, over ``kslice`` with operands σ-reindexed to
    absolute k; the outer reduce is the IDENTITY-lift composition over it (``Fold.composed``).

    Knob keying stamps against the PRE-SPLIT tree, keeping the kernel single-eligible-axis so the
    golden bare-collapse and the prior featurizer stay invariant."""
    if not plan.is_warp:
        legal.enforce(legal.scalar_block_threads(plan), pinned=True)
    w = rplan.cta
    legal.enforce(legal.splitk_slice_k_step(node, plan, w), pinned=True)
    legal.enforce(legal.splitk_materialized_b(node), pinned=True)
    ksplit, kslice, sigma = _factor_k(node.axis, w)
    inner = Fold.contraction(
        k_axis=kslice,
        a=replace(node.a, index=tuple(sigma.apply(e) for e in node.a.index)),
        channels=tuple(replace(ch, b=replace(ch.b, index=tuple(sigma.apply(e) for e in ch.b.index))) for ch in node.channels),
    )
    placed = plan.placed_on(term.place)
    stage = None
    if stage_spec:
        st = Stage.parse(stage_spec)
        budget = term.ctx.max_dynamic_smem
        stage = (
            legal.resolve_warp_stage(inner, placed, st, budget)
            if plan.is_warp
            else legal.resolve_scalar_stage(inner, placed, st, term.tile.inputs, budget)
        )
    # ONE composition rule: the outer reduce is the IDENTITY lift over the sliced contraction
    # operand, its combine the componentwise additive ⊕ over the same accumulator names — the
    # reassociation ``fold_k = fold_{ksplit} ∘ fold_{kslice}``.
    accs = tuple(inner.defines())
    outer = Fold(
        axis=ksplit,
        operands=(inner,),
        lift=Lambda(params=(ksplit.name, *accs), body=Body(()), results=accs),
        **dict(zip(("init", "combine"), M(*(["add"] * len(accs)), names=accs), strict=True)),
    )
    op = Fold.projection(body=term.proj, operands=(outer,)) if len(term.proj) else outer
    return _stamp(term, op, name, knobs, [("REDUCE", outer, rplan), ("TILE", inner, placed), ("STAGE", inner, stage)])


def _materialize(term: _Term, row: dict, name: str, knobs: dict) -> TileOp:
    """One row → its ``TileOp``. Each family resolves ONCE against the row's ``WORK`` inventory,
    per key, through the same :class:`Sched` spelling the enumeration used — and the FORM is the
    two node predicates again, never a role."""
    work = Workers.parse(row.get(WORK.name) or None)
    # Structural stamps (``S_warp_eligible``) ride onto the op: fork rows carry them for branch
    # identity, but the MATERIALIZED op is what ``realized_knobs`` reads, and dropping them here
    # left leaf/evidence rows unstamped while fork rows were stamped — fracturing the ``S_*``
    # evidence signature (the 2026-07-07 5090 gate's 330× fp16 miss).
    op_knobs = {**knobs, **{k: v for k, v in row.items() if k.startswith("S_")}}
    raster_spec = row.get(RASTER.name, "")
    Raster.parse(raster_spec)  # loud pin contract — a malformed spelling fails the row here
    op_knobs = {**op_knobs, RASTER.name: raster_spec, **{k: v for k, v in row.items() if family_of(k) in FAMILIES}}

    site = term.tree[0].site if term.tree else None
    if site is None or site.node.axis is None:
        return _free_option(term, resolve_site_tile(row.get(_key_of(term, site, "TILE"), ""), work), name, op_knobs)
    node = site.node
    rplan = ReducePlan.parse(row.get(_key_of(term, site, "REDUCE"), ""), work)
    stage_spec = row.get(_key_of(term, site, "STAGE"), "") or ""
    if not is_contraction(node):
        stage = _row_stage(term, node) if stage_spec else None
        return _reduce_option(term, node, rplan, stage, name, op_knobs)
    plan = resolve_site_tile(row.get(_key_of(term, site, "TILE"), ""), work, rplan.spell())
    if rplan.stages and rplan.needs_split:
        return _splitk_option(term, plan, node, rplan, name, op_knobs, stage_spec)
    stage = _resolve_stage(term, node, plan, stage_spec)
    if plan.is_warp:
        return _warp_option(term, plan, node, name, op_knobs, stage, work)
    if rplan.stages:
        return _coop_contraction_option(term, node, rplan, name, op_knobs)
    return _tile_option(term, plan, node, name, op_knobs, stage)


def _key_of(term: _Term, site: Site | None, family: str) -> str:
    """The row key ``family`` carries for ``site`` — the canonical site key, or the BARE family name
    when the node is not a site of it (the decided empty)."""
    if site is None:
        return family
    return term.sched.key(family, site.node) or family


def _resolve_stage(term: _Term, node, plan: TilePlan, spec: str) -> Stage | None:
    """Re-resolve the row's ``STAGE`` spelling against the node — the row carries what the
    enumeration RESOLVED, so this reproduces the same slice the leaf identity was built from."""
    if not spec or not plan.is_tiled:
        return None
    st = Stage.parse(spec)
    budget = term.ctx.max_dynamic_smem
    placed = plan.placed_on(term.place)
    if plan.is_warp:
        return legal.resolve_warp_stage(node, placed, st, budget)
    return legal.resolve_scalar_stage(node, placed, st, term.tile.inputs, budget)


# ---- the entry point ----------------------------------------------------------------------------- #


def schedule(tile: TileOp, name: str, knobs: dict, ctx) -> Fork | list[TileOp] | TileOp:
    """Map a freshly-recognized (UNMAPPED) ``tile`` onto the grid and offer its scheduling fork.

    Returns the lazy fork tree over the enumerated rows (levels ``[WORK, *site keys, RASTER]`` — the
    kernel-global worker inventory leads, so every deeper prefix row is self-decoding; the
    launch-order codec closes), a single ``TileOp`` when the space collapses to one row, or ``[]``
    when nothing is enumerable (the guardrail contract — the caller leaves the term unmapped)."""
    term = _Term(tile, tile.place.on_grid(), ctx)
    rows, keys = _enumerate(term)
    if not rows:
        return []

    def materialize(row: dict) -> TileOp:
        return _materialize(term, row, name, knobs)

    if len(rows) == 1:
        return materialize(rows[0])

    def _level(key: str) -> Level:
        return Level((key,), key=lambda r: (r.get(key, ""),))

    levels = [_level(WORK.name), *(_level(k) for k in keys), _level(RASTER.name)]
    return build_fork_tree(params=rows, levels=levels, materialize=materialize)


__all__ = ["FAMILIES", "MAX_ROWS", "schedule"]
