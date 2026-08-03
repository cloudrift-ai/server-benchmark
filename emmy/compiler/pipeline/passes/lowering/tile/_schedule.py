"""Schedule a recognized (UNMAPPED) ``TileOp`` — the generic row enumerator.

**Every role emits rows; no role builds ``TileOp``\\ s directly.** The whole schedule step is one
pipeline, shared by every :class:`~emmy.compiler.ir.axis.AxisRole`:

.. code-block:: text

    sites   = path.family_sites(family, path.sites(term))   # the ONE node walk
    values  = _values(family, site, …)                      # per-family typed slices
    rows    = _assemble(...)                                # spell each row ONCE, derive WORK
    fork    = build_fork_tree(rows, levels=[WORK, *site keys, RASTER], materialize=_materialize)

:func:`_values` is keyed on the site's ``AxisRole`` (never on a node type — there is only one
stored node kind), and the candidate DOMAIN is the move catalog in ``search/space.py``: this module
owns only the per-node legality that a catalog value cannot know (the warp static-K divisibility,
the stage resolvers, coop eligibility). :func:`_assemble` derives the ONE ``WORK`` inventory from
the row's own slices and drops the row when they disagree, so a row is self-consistent before it
ever reaches materialization.

Scope of THIS cut — the three roles whose operand edges are all MATERIALIZED:

- ``FREE`` — the pointwise cell: the register-strip ladder (``TILE=f<r>``, a TERM VARIANT applied
  at materialization).
- ``PLANAR`` / ``TWISTED`` — the reduce partition (``REDUCE``): the conservative heuristic pick,
  then the coop / ILP catalog.
- ``CONTRACTION`` with materialized edges — the ``TILE × STAGE × REDUCE × RASTER`` product over
  the scalar and warp (mma) tiers, split-K rows routing through the structural
  ``Fold ⊃ Fold`` fork that ``030_split_reduce`` consumes.

A term this cut cannot schedule — a COMPUTED operand edge (the fused norm→linear / gate⊗up cone)
or the flash streaming pair — yields NO rows, and ``020_schedule`` leaves it unmapped rather than
guessing. That is the guardrail contract: empty enumeration returns ``[]``, never raises.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from math import prod
from types import SimpleNamespace

from emmy.compiler.context import STATIC_SMEM_CAP
from emmy.compiler.dim import DEFAULT_SEQ_HINT, Dim
from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.schedule import (
    Raster,
    Stage,
    TilePlan,
    WarpSpec,
    Workers,
    plan_workers,
    resolve_site_tile,
)
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Assign, Body, Lambda, Load, Loop, Stmt, Write
from emmy.compiler.ir.stmt.algebra import M
from emmy.compiler.ir.stmt.passes import projection_distributes
from emmy.compiler.ir.tile import Fold, Placement, ReducePlan, Store, TileOp
from emmy.compiler.ir.tile.ir import is_contraction
from emmy.compiler.ir.tile.ops import Sched, axis_role, head, projection_tail, sched_of, seal_workers
from emmy.compiler.pipeline.fork import Fork, Level, build_fork_tree
from emmy.compiler.pipeline.knob import canon_family_value, family_of, values_equal
from emmy.compiler.pipeline.passes.lowering._addr import gmem_row_stride
from emmy.compiler.pipeline.search.space import (
    MAX_BLOCK_THREADS,
    RASTER,
    REDUCE,
    STAGE,
    TILE,
    WORK,
    WSPEC,
    coop_reduce_moves,
    map_tile_moves,
    raster_moves,
    scalar_tile_moves,
    splitk_moves,
    stage_moves,
    warp_tile_moves,
    wspec_moves,
)

logger = logging.getLogger(__name__)


# ---- target / pin reads ------------------------------------------------------------------------ #


def _smem_budget(ctx) -> int:
    """The per-block smem budget the stage resolvers size against — the device's dynamic-smem
    opt-in cap, or the 48 KiB static floor when no ``Context`` reaches the schedule."""
    return ctx.max_dynamic_smem if ctx is not None else STATIC_SMEM_CAP


def _tma_allowed(ctx) -> bool:
    """Whether ``d*/tma*`` stage moves may be offered. TMA (``cp.async.bulk.tensor``) is a Hopper
    (sm_90) feature — Ada / Ampere have none, and nvcc has no ``sm_89a``, so a TMA stage there fails
    to compile. ``ctx is None`` (a direct unit-test drive) allows it: those paths never reach nvcc."""
    return ctx is None or ctx.compute_capability >= (9, 0)


def _pinned_workers() -> Workers | None:
    """The live ``WORK`` env pin's inventory, or ``None`` (unset / unparseable)."""
    raw = WORK.raw()
    if not raw:
        return None
    try:
        return Workers.parse(raw)
    except ValueError:
        return None


def _pinned_tile() -> TilePlan | None:
    """The live ``TILE`` env pin resolved into a :class:`TilePlan` against the ``WORK`` (and
    ``REDUCE``, for the empty-value ambiguity rule) pins, or ``None``. Raises on a malformed pin."""
    raw = TILE.raw()
    if raw is None:
        return None
    return resolve_site_tile(raw, _pinned_workers(), REDUCE.raw() or "")


def _pinned_reduce() -> ReducePlan | None:
    """The live ``REDUCE`` env pin resolved into a :class:`ReducePlan` against the ``WORK`` pin."""
    raw = REDUCE.raw()
    if raw is None:
        return None
    return ReducePlan.parse(raw, _pinned_workers())


def warp_tile_pinned() -> bool:
    """A live warp (atom-naming) ``TILE`` env pin. Exposed as a function so ``020_schedule`` never
    imports the ``Knob`` objects themselves (``Pass.load`` OFF-fills any bare ``Knob`` attr it
    finds on a rule module onto every variant of the pass)."""
    try:
        plan = _pinned_tile()
    except ValueError:
        return False
    return plan is not None and plan.is_warp


def _filter_work(options: list, work_of) -> list:
    """Narrow ``options`` by the live ``WORK`` env pin. No pin / no survivor ⇒ the full list (a
    no-match pin degrades with a warning rather than emptying the enumeration)."""
    raw = WORK.raw()
    if raw is None or not options:
        return options
    kept = [o for o in options if values_equal(WORK.name, raw, work_of(o))]
    if not kept:
        logger.warning("WORK pin %r matches no candidate's worker inventory; keeping the full fork", raw)
        return options
    return kept


# ---- structural reads over the stored term ----------------------------------------------------- #


def _node_loads(node) -> list[Load]:
    """Every gmem ``Load`` the term reads, as a deep walk over the STORED structure: an operand
    edge's MATERIALIZED inhabitant plus the loads sitting inline in a lift body, recursing through
    a COMPUTED edge's own node. The node-native equivalent of scanning a lowered nest."""
    out: list[Load] = []

    def walk_stmts(stmts) -> None:
        for s in stmts:
            if isinstance(s, Fold):
                walk(s)
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
            if n._contraction is None:
                walk_stmts(list(n.lift.body))

    walk(node)
    return out


def _projection(op) -> Body:
    """The kernel's per-cell projection — the wrapping zero-axis fold's body, or empty when the
    term is a bare node. A projection has ONE home (the wrapper's lift), never a node field."""
    return op.lift.body if isinstance(op, Fold) and op.axis is None and op.operands else Body(())


def _a_computed(node) -> bool:
    """The contraction's shared A edge is COMPUTED (an inline cone) rather than a gmem ``Load``."""
    return not isinstance(node.a, Load)


def _placed(place: Placement, plan: TilePlan) -> TilePlan:
    """``plan`` bound to the ROOT contraction's ``(m, n)`` — the kernel grid's trailing pair, the
    same rule ``ops.Sched._mn_for`` states for a depth-1 site (so a reader never re-derives it)."""
    grid = tuple(place.grid)
    return plan.at(grid[-2], grid[-1]) if len(grid) >= 2 else plan


def _family_key(op, family: str, node) -> str:
    """The CANONICAL knob key for ``family`` addressing ``node`` on ``op``'s tree. A node that is
    not a site of the family (a PLANAR fold's ``TILE``) keys the bare family — the decided-empty."""
    return Sched(op, {}).key(family, node) or family


def _site_keys(op) -> tuple[str, str, str]:
    """The ``(TILE, STAGE, REDUCE)`` key triple for this kernel's primary node — bare when there is
    no node to key on (a pure pointwise cell / the raw-loop-IR escape)."""
    node = head(op)
    if node is None:
        return (TILE.name, STAGE.name, REDUCE.name)
    return tuple(_family_key(op, f.name, node) for f in (TILE, STAGE, REDUCE))  # type: ignore[return-value]


# ---- the row grammar: SITE-LOCAL TILE/REDUCE values + ONE derived WORK entry -------------------- #


def _site_knobs(stamped: dict) -> dict:
    """A builder's stamped knob dict flipped to the site grammar: TILE/REDUCE values re-spell
    SITE-LOCAL (the worker halves live in the ``WORK`` entry ``seal_workers`` stamps), and the
    ``WSPEC`` key never stamps (the producer band is ``WORK``'s ``+p`` suffix)."""
    return {
        k: canon_family_value(k, v) if isinstance(v, str) and family_of(k) in ("TILE", "REDUCE") else v
        for k, v in stamped.items()
        if k != WSPEC.name
    }


def _assemble(keys: tuple[str, str, str], plan: TilePlan, stage: str, rplan: ReducePlan, aux: int, raster: str, stamp: dict) -> dict | None:
    """One assembled fork row. TILE/REDUCE spell their SITE halves and the worker geometry (the
    tile's units, a coop ``REDUCE`` width, the ``aux`` producer band) is DERIVED into ONE ``WORK``
    entry — the row never carries a ``WSPEC`` key. A genuine inventory conflict (tiled TILE workers
    vs a differing coop width, a producer band with no warp inventory) drops the row (``None``):
    ``WORK`` is derived from the slices, so a row whose slices disagree is not co-representable."""
    tkey, skey, rkey = keys
    work = plan_workers(plan)
    if rplan.coop > 1:
        coop = Workers(kind="thread", units=(rplan.coop, 1))
        if work is not None and work != coop:
            return None
        work = coop
    if aux:
        if work is None or work.kind != "warp":
            return None
        work = replace(work, producer=aux)
    return {
        tkey: plan.spell(),
        skey: stage,
        rkey: rplan.spell(),
        WORK.name: work.spell() if work is not None else "",
        RASTER.name: raster,
        **stamp,
    }


# ---- the reduce family ------------------------------------------------------------------------- #

# Conservative cooperative-reduce selection constants (the default when REDUCE is unpinned).
_COOP_MIN_EXTENT = 128  # only cooperate when the reduce axis is at least this wide
_SERIAL_TARGET = 8  # aim for ~this many serial steps per cooperating thread
_MAX_COOP = 256  # cap on cooperative threads per CTA (power of two)
_FREE_CAP = 256  # only cooperate when the output grid is at most this many cells


def _hint_extent(ax) -> int:
    """An axis's static extent, or its ``Dim`` hint when symbolic."""
    e = ax.extent
    return e.as_static() if e.is_static else (e.hint or DEFAULT_SEQ_HINT)


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


def _has_accum(stmts) -> bool:
    from emmy.compiler.ir.stmt import Accum  # noqa: PLC0415

    return any(isinstance(s, Accum) or any(_has_accum(list(b)) for b in s.nested()) for s in stmts)


def _has_contraction_tail(stmts) -> bool:
    """The post-reduce tail contracts over a NEW free axis — a ``Loop`` whose body holds an inner
    reduce ``Loop``. This is the fused norm→linear shape, distinguished from a plain softmax tail
    (a single sweep over the SAME axis)."""
    for s in stmts:
        if isinstance(s, Loop) and any(isinstance(c, Loop) and _has_accum(list(c.body)) for c in s.body):
            return True
        if any(_has_contraction_tail(list(b)) for b in s.nested()):
            return True
    return False


def _matvec_b_kstride(kernel, carrier, place) -> int | None:
    """B's gmem stride along the reduce axis at the per-cell MATVEC tier, or ``None`` when no
    layout gate applies. A contraction demoted to PLANAR carries BOTH a vector operand (a load
    along the reduce axis touching no non-unit free axis — A) and a matrix operand indexed by the
    reduce axis AND a non-unit free axis (B); only that two-operand shape is gated. ``1`` means the
    reduce axis is B's fastest-varying dimension (the serving ``F.linear`` N×K layout); ``>1`` is
    k-major (canonical ``B[k, n]``)."""
    nonunit = {a.name for a in place.free if not (a.extent.is_static and a.extent.as_static() == 1)}
    k_name = carrier.axis.name
    a_seen = False
    strides = set()
    for ld in _node_loads(kernel.op):
        used = set().union(*(e.free_vars() for e in ld.index)) if ld.index else set()
        if k_name not in used:
            continue
        if used & nonunit:
            strides.add(gmem_row_stride(ld, k_name, kernel.inputs))
        else:
            a_seen = True
    return strides.pop() if a_seen and len(strides) == 1 else None


def _shared_row_buf(carrier_loads, tail, grid_vars, raxis: Axis, inputs) -> str | None:
    """The input buffer reused as a CTA-shared ROW across the reduce + a contraction tail — read in
    the carrier at ``(grid…, raxis)`` AND in the tail at ``(grid…, k)``, its trailing dim the
    (static) reduce extent. ``None`` ⇒ no eligible operand (stay gmem-direct)."""
    if not raxis.extent.is_static or not _has_contraction_tail(tail):
        return None
    n = len(grid_vars)
    carrier_bufs = {
        s.input for s in carrier_loads if len(s.index) == n + 1 and tuple(s.index[:n]) == grid_vars and s.index[-1] == Var(raxis.name)
    }
    for s in _node_loads_in(tail):
        if s.input in carrier_bufs and len(s.index) == n + 1 and tuple(s.index[:n]) == grid_vars:
            t = inputs.get(s.input)
            if t is not None and t.shape and t.shape[-1].is_static and t.shape[-1].as_static() == raxis.extent.as_static():
                return s.input
    return None


def _node_loads_in(stmts) -> list[Load]:
    """Every scalar ``Load`` reachable in ``stmts`` (deep) — the projection tail's own reads."""
    out: list[Load] = []
    for s in stmts:
        if isinstance(s, Load) and s.is_scalar:
            out.append(s)
        for b in s.nested():
            out.extend(_node_loads_in(list(b)))
    return out


def _row_stage(tile, place) -> Stage | None:
    """The shared-row :class:`Stage` for a COOPERATIVE reduce, or ``None``. Not a knob — it fires
    whenever the cooperative partition is chosen and the shape qualifies (a pure perf transform),
    stored as a schedule slice only."""
    red = head(tile.op)
    if red is None:
        return None
    tail = projection_tail(tile)
    if not _has_contraction_tail(tail):
        return None
    grid_vars = tuple(Var(a.name) for a in place.grid)
    carrier_loads = [ld for ld in _node_loads(red) if ld.is_scalar]
    buf = _shared_row_buf(carrier_loads, tail, grid_vars, red.axis, tile.inputs)
    return Stage(transport="sync", smem=(buf,)) if buf is not None else None


def _reduce_specs(kernel, place, ctx=None) -> list[ReducePlan]:
    """The ``PLANAR`` / ``TWISTED`` reduce-partition candidates — option-0 is the conservative
    heuristic pick (:func:`_pick_coop`, so a cold greedy compile keeps its historical deploy), then
    the legal :func:`coop_reduce_moves` catalog + serial as fork siblings. The catalog rows are what
    keep the 16- / 32-wide reduce goldens reachable. An env pin is authoritative."""
    carrier = head(kernel.op)
    if carrier is None or carrier.role not in (AxisRole.PLANAR, AxisRole.TWISTED):
        return [ReducePlan()]  # not cooperative-eligible — the scalar serial fold
    extent = _hint_extent(carrier.axis)
    free = prod(_hint_extent(a) for a in place.free) if place.free else 1
    tail = projection_tail(kernel)
    coop = _pick_coop(extent, free, has_tail=_has_contraction_tail(tail))
    # The layout gate (WS5, the cold-poison hardening): at the matvec tier the coop bands are only
    # coalesced on ONE B orientation — the plain band interleaves lanes along K, the transposed
    # band sweeps lanes along the output axis. Enumeration is the single choke point every tier
    # resolves through, so the gate lives here; an env pin stays authoritative and un-gated.
    kstride = _matvec_b_kstride(kernel, carrier, place)
    deploy = ctx is None or ctx.validate_pins
    if deploy and REDUCE.raw() is None and kstride == 1 and extent >= _COOP_MIN_EXTENT and free >= _FREE_CAP:
        return [ReducePlan.of(coop=32)]
    if coop > 1 and kstride is not None and kstride != 1:
        coop = 1  # the heuristic option-0 is a plain band too — uncoalesced on a k-major B
    cands = [ReducePlan.of(coop=coop)]
    tail_scalar = not any(isinstance(s, Loop) for s in tail)
    inner = next((a for a in reversed(place.free) if not (a.extent.is_static and a.extent.as_static() == 1)), None) if place.free else None
    k_static = carrier.axis.extent.as_static() if carrier.axis.extent.is_static else None
    bt_ok = (
        k_static is not None
        and tail_scalar
        and inner is not None
        and inner.extent.is_static
        and inner.extent.as_static() % 32 == 0
        and _row_stage(kernel, place) is None
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
    pin = _pinned_reduce()
    return cands if pin is None else [pin]


# ---- the contraction families ------------------------------------------------------------------ #

# The mma atoms eligible per operand dtype — the warp tier's dtype gate (16-bit operands only).
_ATOMS_BY_DTYPE = {"f16": ("mma_m16n8k16_f16_f32",), "bf16": ("mma_m16n8k16_bf16_f32",)}

# Emit unpinned split-K candidates only when the output grid alone leaves the GPU under-occupied.
_SPLITK_MAX_CTAS = 1024


def _fragment_epilogue_ok(epilogue: Body) -> bool:
    """The mma store folds the projection into a ``RegEpilogue`` whose leaf ``Load``\\ s are
    evaluated independently per fragment element — a load whose INDEX reads a name an earlier
    epilogue stmt defined (an embedding gather) cannot be threaded through that form."""
    defs: set[str] = set()
    for s in epilogue:
        if isinstance(s, Load) and {v for e in s.index for v in e.free_vars()} & defs:
            return False
        defs.update(s.defines())
    return True


# The consumer-die compute capabilities where f32-accumulate HMMA runs at HALF the f16-accumulate
# rate (GA102/AD102/GB202 silicon). On the datacenter parts f32-accumulate is full rate, so the
# f16acc fork is pure search noise there.
_F16ACC_CCS = frozenset({(8, 6), (8, 9), (12, 0)})

# The f16-accumulate sibling of each base atom (f16 only — mma.sync has no bf16-accumulate form).
_F16ACC_ATOMS = {"mma_m16n8k16_f16_f32": "mma_m16n8k16_f16_f16"}


def _f16acc_allowed(ctx) -> bool:
    """Whether the f16-accumulate atom forks may be OFFERED — a precision-trading gate, off by
    default: the precise ``F16_MMA_F32_ACC`` pin is authoritative on every target; unset, the
    ``FAST_MATH`` umbrella offers it on the consumer dies (:data:`_F16ACC_CCS`) where the
    f32-accumulate half-rate nerf makes it profitable. ``ctx is None`` stays off — enumeration must
    not grow under a bare umbrella with no known target. A ``TILE`` pin naming the atom bypasses
    this gate entirely (pins are authoritative)."""
    from emmy.compiler.pipeline.search.space import F16_MMA_F32_ACC, precision_pin  # noqa: PLC0415

    raw = F16_MMA_F32_ACC.raw()
    if raw is not None:
        return F16_MMA_F32_ACC.parse(raw)
    if not precision_pin(F16_MMA_F32_ACC):
        return False
    return ctx is not None and ctx.compute_capability in _F16ACC_CCS


def _with_f16acc(atoms: tuple[str, ...], ctx) -> tuple[str, ...]:
    """Extend the dtype-eligible ``atoms`` with their f16-accumulate siblings when
    :func:`_f16acc_allowed` — the extended tuple rides :func:`warp_tile_moves` unchanged, so the
    f16acc forks are ordinary ``TILE`` rows identified by the value's atom token."""
    if not atoms or not _f16acc_allowed(ctx):
        return atoms
    return atoms + tuple(_F16ACC_ATOMS[a] for a in atoms if a in _F16ACC_ATOMS)


def _warp_atoms(kernel, node, proj: Body) -> tuple[str, ...]:
    """The dtype-eligible tensor-core atom names for this contraction, ``()`` when the warp tier
    doesn't apply (no node, a non-16-bit operand dtype, or a fragment-unrealizable gather
    epilogue). Reads pure algebra off the STORED node — the placement / tile would be unread."""
    if node is None or not kernel.inputs or not _fragment_epilogue_ok(proj):
        return ()
    if not isinstance(node.a, Load):
        return ()  # a computed cone is out of this cut's scope (see the module docstring)
    t = kernel.inputs.get(node.a.input)
    return _ATOMS_BY_DTYPE.get(getattr(getattr(t, "dtype", None), "name", None), ())


def _check_warp_static_k(node, wt: TilePlan) -> None:
    """Reject a warp move whose **static** contraction K is not a multiple of the inner mma K-step
    (``atom_k · bk``). The warp K-loop has no static-K tail handling — a partial final step reads
    past the operand and silently corrupts the result. A SYMBOLIC K is fine (it reaches the masked
    tier), so guard only the static case."""
    ext = node.axis.extent
    if not ext.is_static:
        return
    k = ext.as_static()
    step = wt.atom.atom_k * wt.bk
    if k % step:
        raise ValueError(
            f"warp TILE pin K-step {step} (atom_k={wt.atom.atom_k}·bk={wt.bk}) does not divide the "
            f"static contraction K={k}; the warp K-loop has no static-K tail masking yet, so a "
            f"partial final step corrupts the result. Pin a K that is a multiple of {step}, or "
            f"drop the atom token to use the scalar tier."
        )


def _warp_move_ok(node, plan: TilePlan) -> bool:
    """The enumeration-side (filtering) form of :func:`_check_warp_static_k` — an unpinned warp move
    whose K-step doesn't divide the static K is silently dropped; a PIN with the same defect
    raises. ONE predicate, one home; the caller picks raise-vs-drop from ``pinned``."""
    try:
        _check_warp_static_k(node, plan)
    except ValueError:
        return False
    return True


def _tile_area(plan: TilePlan) -> int:
    """The output cells one CTA covers under ``plan`` — the occupancy denominator."""
    am, an = (plan.atom.atom_m, plan.atom.atom_n) if plan.is_warp else (1, 1)
    return max(plan.units_m * plan.reg_m * am * plan.units_n * plan.reg_n * an, 1)


def _can_stage_warp(stage, k_axis: Axis, tile_n: int, bk: int, atom_k: int, mask_n: bool, b_trans: bool) -> bool:
    """cp.async staging eligibility: a STATIC, tile-divisible K. A masked / symbolic M is fine (the
    A-slab fill clamp-reads the overhang); a masked N or a non-divisible K stays gmem-direct.
    cp.async needs a ≥4 B contiguous chunk, so the inner slab dim must be even at 2 B/elem."""
    if stage is None or stage.transport != "cp.async" or mask_n:
        return False
    if not k_axis.extent.is_static:
        return False
    bk_elems = bk * atom_k
    if k_axis.extent.as_static() % bk_elems != 0:
        return False
    return (bk_elems % 2 == 0) and (b_trans or tile_n % 2 == 0)


def _tma_operand_rank_ok(index: tuple, tile_name: str, k_name: str) -> bool:
    """Whether TMA's box can encode this operand's gmem index. The box's data plane is the TRAILING
    2 dims; extra LEADING dims ride as extent-1 box dims whose origin coordinate is evaluated once
    per fill — so those exprs must not move with the tile or the K loop. Rank caps at 4 so the
    swizzle-split box stays within TMA's 5-dim hardware limit."""
    if not 2 <= len(index) <= 4:
        return False
    return all(not ({tile_name, k_name} & e.free_vars()) for e in index[:-2])


def _can_stage_warp_tma(stage, k_axis: Axis, n_axis: Axis, tile_n: int, bk: int, atom_k: int, elem_bytes: int, mask_n, b_trans) -> bool:
    """TMA staging eligibility: a STATIC, tile-divisible K and N. The box's inner dim and the
    source's inner global stride must be 16 B-aligned (the NONE-swizzle box-copy rule); a
    transposed B boxes N-major, so N drops out of the alignment gate."""
    if stage is None or stage.transport != "tma" or mask_n:
        return False
    if not (k_axis.extent.is_static and n_axis.extent.is_static):
        return False
    bk_elems = bk * atom_k
    k, n = k_axis.extent.as_static(), n_axis.extent.as_static()
    if k % bk_elems != 0:
        return False
    inner = (bk_elems, k) if b_trans else (bk_elems, tile_n, k, n)
    return all((x * elem_bytes) % 16 == 0 for x in inner)


def _resolve_warp_stage(c: Fold, tile: TilePlan, stage: Stage, budget: int = STATIC_SMEM_CAP) -> Stage | None:
    """Resolve an operand ``Stage`` against the warp (mma) contraction ``c`` — TMA > cp.async >
    gmem-direct (``None``). The resolved stage carries ``bk_elems``, ``depth`` clamped so the ring's
    slots fit ``budget`` (dropping ``ring`` when the clamp leaves nothing to cycle) and
    ``reg_depth`` clamped to ``bk``. A tile whose single depth-1 slot already exceeds ``budget``
    DECLINES — unlike the scalar resolver it cannot shrink the slab."""
    if stage.alt:
        return None  # the alternating single-slab pipeline is the warp-flash stream's
    atom = tile.atom
    a_nbytes = atom.operand_dtype("a").nbytes
    bk = tile.bk
    m, n = tile.m, tile.n
    tma_rank_ok = (
        isinstance(c.a, Load)
        and isinstance(c.b, Load)  # a descriptor needs a gmem address on BOTH edges
        and _tma_operand_rank_ok(c.a.index, m.axis.name, c.axis.name)
        and _tma_operand_rank_ok(c.b.index, n.axis.name, c.axis.name)
    )
    # TMA hardware: every box dim must be 1..256 — an oversized warp register tile must decline.
    tma_box_ok = max(m.tile, n.tile, bk * atom.atom_k) <= 256
    tma_ok = tma_rank_ok and tma_box_ok and _can_stage_warp_tma(stage, c.axis, n.axis, n.tile, bk, atom.atom_k, a_nbytes, n.mask, c.b_trans)
    cp_ok = (not tma_ok) and _can_stage_warp(stage, c.axis, n.tile, bk, atom.atom_k, n.mask, c.b_trans)
    if not (tma_ok or cp_ok):
        return None
    bk_elems = bk * atom.atom_k
    slot_bytes = (m.tile + n.tile) * bk_elems * a_nbytes
    if slot_bytes > budget:
        return None
    depth = min(stage.depth, budget // slot_bytes)
    return replace(stage, depth=depth, ring=stage.ring and depth >= 2, reg_depth=min(stage.reg_depth, bk), bk_elems=bk_elems)


def _resolve_scalar_stage(c: Fold, tile: TilePlan, stage: Stage, inputs, budget: int = STATIC_SMEM_CAP) -> Stage | None:
    """Resolve an operand ``Stage`` against the scalar register-tile contraction ``c``, or ``None``
    (gmem-direct). The slab K-chunk ``bk_elems`` is DERIVED to fit ``depth`` operand slots in the
    smem ``budget`` (largest power of two dividing K) — not codec-spelled, so no schema change;
    when no chunk fits at the requested depth the depth steps down, single-buffer last."""
    if stage.alt:
        return None
    if not c.axis.extent.is_static or stage.transport not in ("tma", "cp.async"):
        return None
    # A masked-N B-slab fill would clamp a chunk-start column into a row-crossing gmem address and
    # hang on the misaligned copy; a transposed B has no scalar drain variant (the warp tier stages
    # it into an N-major slab).
    if tile.n.mask or c.b_trans:
        return None
    if not inputs or not isinstance(c.a, Load) or not isinstance(c.b, Load) or c.a.input not in inputs:
        return None
    if stage.transport == "tma" and not (
        _tma_operand_rank_ok(c.a.index, tile.m.axis.name, c.axis.name) and _tma_operand_rank_ok(c.b.index, tile.n.axis.name, c.axis.name)
    ):
        return None
    # Staging needs the CTA to BE one (tile_m × tile_n) output tile (the cooperative fill / drain
    # contract). A register-only tile launches the scalar default block over unrelated cells.
    if tile.launch_threads is None:
        return None
    K = c.axis.extent.as_static()
    elem_bytes = inputs[c.a.input].dtype.nbytes
    if stage.transport == "tma" and max(tile.m.tile, tile.n.tile) > 256:
        return None
    # Every staged transport needs 16 B-aligned inner global strides — A's is K, B's is N. An
    # odd-stride shape faults at descriptor encode (TMA) or at runtime (cp.async), so gate both.
    n_ext = tile.n.axis.extent
    if not n_ext.is_static or ((K * elem_bytes) % 16 or (n_ext.as_static() * elem_bytes) % 16):
        return None
    b_bytes = inputs[c.b.input].dtype.nbytes if c.b.input in inputs else elem_bytes
    depth, bk_elems = max(1, stage.depth), 0
    while depth >= 1:
        cap = budget // (depth * max(1, tile.m.tile * elem_bytes + tile.n.tile * b_bytes))
        bk_elems = next((v for v in (128, 64, 32, 16, 8, 4) if v <= cap and K % v == 0), 0)
        if bk_elems >= 4:
            break
        depth -= 1
    if bk_elems < 4:
        return None
    return replace(stage, depth=depth, ring=stage.ring and depth >= 2, reg_depth=1, bk_elems=bk_elems)


def _stage_spec() -> str:
    """The pinned ``STAGE`` codec spelling, or ``""``. A pin that doesn't parse as the codec is
    structurally invalid for this tier and degrades to gmem-direct rather than failing lowering."""
    pinned = STAGE.narrow([""])[0]
    if not pinned:
        return ""
    try:
        Stage.parse(pinned)
    except ValueError:
        return ""
    return pinned


def _stage_values(kernel, node, place, plan: TilePlan, budget: int, tma_ok: bool) -> list[str]:
    """The RESOLVED ``STAGE`` spellings for one tile candidate — gmem-direct ``""`` first, then
    every catalog move that RESOLVES against the node with this ``plan``, so the leaf identity, the
    stamped knobs and the kernel agree. A pinned ``STAGE`` is authoritative: the resolved pin alone,
    or gmem-direct when it declines."""
    if node is None or not plan.is_tiled:
        return [""]  # per-cell / unbindable — no operand slab to stage
    tile = _placed(place, plan)

    def resolve(spec: str) -> str | None:
        st = Stage.parse(spec)
        if st.transport == "tma" and not tma_ok:
            return None  # TMA is Hopper+ (sm_90) — decline below it rather than fail to compile
        r = _resolve_warp_stage(node, tile, st, budget) if plan.is_warp else _resolve_scalar_stage(node, tile, st, kernel.inputs, budget)
        return r.spell() if r is not None else None

    if STAGE.raw() is not None:
        pinned = _stage_spec()
        r = resolve(pinned) if pinned else None
        return [r] if r else [""]
    out = [""]
    for move in stage_moves(warp=plan.is_warp):
        r = resolve(move) if move else None
        if r and r not in out:
            out.append(r)
    return out


def _splitk_pin() -> ReducePlan | None:
    """The pinned cross-CTA split-K partition (or ``None``) — a ``REDUCE`` pin resolving to a GRID
    split. A non-split coop / ILP pin is not a split-K request: ignore it rather than fail."""
    try:
        plan = _pinned_reduce()
    except ValueError:
        return None
    return plan if plan is not None and plan.needs_split else None


def _coop_reduce_spec() -> ReducePlan | None:
    """The pinned cooperative / ILP K partition a NON-output-tiled contraction honors (the split-K
    takes the structural fork instead); ``None`` when the pin is a foreign codec."""
    try:
        plan = _pinned_reduce()
    except ValueError:
        return None
    if plan is None or plan.needs_split or not (plan.coop > 1 or plan.reg > 1):
        return None
    return plan


def _reduce_values(kernel, place, plan: TilePlan, node, channels: int = 1) -> list[ReducePlan]:
    """The ``REDUCE`` candidates for one contraction tile candidate — serial first (option-0), then
    the legal coop / ILP moves (per-cell tier only — the non-output-tiled contract) and the divisor-
    and occupancy-guarded split-K moves. An ATOMIC split is offered only on a single-channel node
    whose FULL projection tail distributes over the add; the deferred kernel finalize stays legal
    for any epilogue."""
    ext = node.axis.extent
    if REDUCE.raw() is not None:
        split = _splitk_pin()
        if split is not None:
            return [split]
        coop = _coop_reduce_spec()
        if coop is not None:
            return [coop] if not plan.is_tiled else []
        return [ReducePlan()]
    out = [ReducePlan()]
    k = ext.as_static() if ext.is_static else None
    if k is not None and not plan.is_tiled:
        # The innermost NON-UNIT free axis — the m1 recognizer's synthesized unit axis can sit
        # innermost (extent 1), and it is not the axis the transposed emitter sweeps.
        inner = next((a for a in reversed(place.free) if not (a.extent.is_static and a.extent.as_static() == 1)), None)
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
    free = prod(_hint_extent(a) for a in place.free) if place.free else 1
    if k is not None and free // _tile_area(plan) <= _SPLITK_MAX_CTAS:
        step = plan.atom.atom_k * plan.bk if plan.is_warp else 1
        tail = tuple(projection_tail(kernel))
        atomic_ok = channels == 1 and (len(tail) == 0 or projection_distributes(tail, (node.acc,)))
        for sp in splitk_moves(warp=plan.is_warp):
            if sp.finalize == "atomic" and not atomic_ok:
                continue  # a non-distributive projection would raise at 030_split_reduce
            if k % sp.cta == 0 and (k // sp.cta) % step == 0:
                out.append(sp)
    return out


def _raster_values(place) -> list[str]:
    """The ``RASTER`` candidates for one contraction row. A symbolic-axis (masked-tile) grid renders
    through the dynamic decode path, which does not carry the swizzle — offering ``gm8`` there would
    stamp a launch order the kernel doesn't realize, so a symbolic grid decides the flat ``""``."""
    if any(not ax.extent.is_static for ax in place.free):
        return [""]
    return list(RASTER.narrow(raster_moves()))


def _wspec_values(plan: TilePlan, stage_spelling: str, red: ReducePlan) -> list[str]:
    """The ``WSPEC`` candidates for one enumerated row — uniform ``""`` alone unless the row can
    drive a producer band: a warp tile over a resolved TMA stage and no cross-CTA split."""
    if not (plan.is_warp and "tma" in stage_spelling and not red.needs_split):
        return [""]
    return list(WSPEC.narrow(wspec_moves()))


def _aux_warps(wspec: str) -> int:
    """The dedicated producer-warp count a ``WSPEC`` candidate asks for (``0`` for uniform / an
    unparseable spelling) — the band the row's ``WORK`` inventory carries as its ``+p`` suffix."""
    if not wspec:
        return 0
    try:
        return WarpSpec.parse(wspec).aux_warps
    except ValueError:
        return 0


def _wspec_workers(spec: str, stage, block_threads: int | None) -> tuple[WarpSpec | None, str]:
    """The resolved ``WSPEC`` worker split, or ``(None, "")`` — uniform SIMT. A spec that doesn't
    parse, names no role, carries a reserved per-role param, or whose roles are illegal (a producer
    drives a resolved TMA stage only) degrades to uniform silently. Two thread-budget gates:
    ``block_threads + 32·aux`` fits the CTA limit, and ``32·aux ≤ block_threads``."""
    if not spec:
        return None, ""
    try:
        ws = WarpSpec.parse(spec)
    except ValueError:
        return None, ""
    if not ws.roles or any(a.params for a in ws.roles):
        return None, ""
    if not ws.is_legal(SimpleNamespace(stage=stage)):
        return None, ""
    aux = 32 * ws.aux_warps
    if block_threads is None or aux > block_threads or block_threads + aux > MAX_BLOCK_THREADS:
        return None, ""
    return ws, spec


# ---- rows: the ONE enumeration ----------------------------------------------------------------- #


def _free_tile_values() -> list[TilePlan]:
    """The ``FREE`` ``TILE`` candidates — the per-cell tile (option-0), then the catalog's
    register-strip ladder. ``r`` IS the spelled ``TILE=f<r>``: the strip is a TERM VARIANT applied
    at materialization, a function of the ROW, not a member of a pre-enumerated variant set."""
    return [TilePlan(), *map_tile_moves()]


def _strip_width(plan: TilePlan) -> int:
    """The strip ratio ``r`` a ``FREE`` row's ``TILE`` names — the inner register width. A warp
    codec names none (there is no fragment on a pointwise cell), so it reads ``0`` and is dropped."""
    return 0 if plan.is_warp else plan.regs[0]


def _strippable(tile: TileOp, place: Placement) -> bool:
    """Whether the pointwise cell admits the register strip: a pure zero-axis fold with no operands
    whose body is FLAT elementwise (per-cell ``Load`` / ``Assign`` + boundary root stores, no nested
    ``Loop`` / carried state), over a static innermost free axis."""
    op = tile.op
    if not (isinstance(op, Fold) and op.axis is None and not op.operands) or not place.free:
        return False
    if not place.free[-1].extent.is_static:
        return False
    return all(isinstance(s, (Load, Assign, Write)) for s in op.body) and all(st.sweep is None for st in tile.stores)


def _rows(tile: TileOp, place: Placement, ctx) -> tuple[list[dict], tuple[str, str, str]]:
    """Every legal schedule row for ``tile``, in the site value grammar, plus the
    ``(TILE, STAGE, REDUCE)`` key triple. ONE walk, one row shape, three role arms — each family's
    values come from :func:`_values`-style emitters keyed on the site's ``AxisRole``, and
    :func:`_assemble` derives ``WORK`` and drops a row whose slices disagree.

    An empty result is the guardrail contract, never a raise: the caller leaves the term unmapped.
    """
    keys = _site_keys(tile.op)
    role = axis_role(tile.op)
    node = head(tile.op)
    if role is AxisRole.FREE:
        pin = _pinned_tile()
        strip_ok = _strippable(tile, place)
        ext = place.free[-1].extent.as_static() if strip_ok else 0
        rows = []
        for plan in [pin] if pin is not None else _free_tile_values():
            # A strip WIDTH the cell cannot carry (a stateful / sweep body, a symbolic or
            # indivisible inner extent, a warp codec on a pointwise cell) drops the row; the flat
            # per-cell base below is always offered, so a narrowing pin degrades to option-0.
            if _strip_width(plan) > 1 and not (ext and ext % _strip_width(plan) == 0):
                continue
            row = _assemble(keys, plan, "", ReducePlan(), 0, "", {})
            if row is not None:
                rows.append(row)
        if not rows:
            rows = [_assemble(keys, TilePlan(), "", ReducePlan(), 0, "", {})]
        return _filter_work(rows, lambda r: r.get(WORK.name, "")), keys
    if role is not AxisRole.CONTRACTION:
        # PLANAR / TWISTED — the reduce partition is the only family with a choice here.
        rows = []
        for rplan in _reduce_specs(tile, place, ctx):
            row = _assemble(keys, TilePlan(), "", rplan, 0, "", {})
            if row is not None:
                rows.append(row)
        return _filter_work(rows, lambda r: r.get(WORK.name, "")), keys
    # CONTRACTION — the tile × stage × reduce × wspec × raster legal product.
    if node is None or not is_contraction(node) or _a_computed(node):
        return [], keys  # a COMPUTED operand edge is out of this cut's scope
    proj = _projection(tile.op)
    budget, tma_ok = _smem_budget(ctx), _tma_allowed(ctx)
    tiles = scalar_tile_moves()
    atoms = _with_f16acc(_warp_atoms(tile, node, proj), ctx)
    warp_moves = [p for p in warp_tile_moves(atoms) if _warp_move_ok(node, p)] if atoms else []
    tiles += warp_moves
    pin = _pinned_tile()
    if pin is not None:
        tiles = [pin]
    # Warp-eligibility is a structural fact about the KERNEL: when the enumeration offers any
    # tensor-core row, EVERY row carries ``S_warp_eligible`` so the priors can price "a scalar tile
    # where tensor cores were on offer". ``S_``-prefixed — not a schedule family, so tile identity
    # and prefix-consistency are untouched.
    stamp: dict = {"S_warp_eligible": 1.0} if warp_moves else {}
    rows: list[dict] = []
    for plan in tiles:
        if plan.is_warp and pin is not None:
            _check_warp_static_k(node, plan)  # a PIN with an indivisible K-step raises
            if not _fragment_epilogue_ok(proj):
                raise ValueError(
                    "warp TILE pin: the projection epilogue gathers through another epilogue load "
                    "(a data-dependent index) — the fragment epilogue cannot thread it; drop the "
                    "atom token to use the scalar tier."
                )
        for stage in _stage_values(tile, node, place, plan, budget, tma_ok):
            for red in _reduce_values(tile, place, plan, node, len(node.channels)):
                for wspec in _wspec_values(plan, stage, red):
                    for raster in _raster_values(place):
                        row = _assemble(keys, plan, stage, red, _aux_warps(wspec), raster, stamp)
                        if row is not None:
                            rows.append(row)
    return _filter_work(rows, lambda r: r.get(WORK.name, "")), keys


# ---- materialization: one builder per role, all fed by the same row ----------------------------- #


def _stamp(tile: TileOp, op, place, name, knobs: dict, slices, workers=None) -> TileOp:
    """Build the scheduled ``TileOp``: the term + placement + the resolved slices in
    ``TileOp.schedule`` (keyed through :class:`Sched`, the canonical codec spelling) + the sealed
    ``WORK`` inventory. The term stays pure algebra — no slice is ever a node field."""
    out = TileOp(op=op, name=name, place=place, workers=workers, knobs=_site_knobs(knobs), stores=tile.stores)
    sched = sched_of(out)
    for family, node, value in slices:
        if value is not None:
            sched.put(family, node, value)
    seal_workers(out)
    return out


def _strip_variant(tile: TileOp, place: Placement, plan: TilePlan, name: str, knobs: dict) -> TileOp:
    """The pointwise register-STRIP term variant: hand each thread ``r`` CONTIGUOUS inner-axis
    elements. The inner free axis shrinks to ``extent/r`` (the grid walks it) and the cell body is
    unrolled ``r`` times — copy ``i`` reads/writes ``inner·r + i`` with its SSA names suffixed —
    then regrouped as ``r`` loads · ``r`` computes · ``r`` writes so the unit-stride runs feed
    ``050_vectorize_loads`` / ``080_vectorize_stores``. A different term, hence a different
    ``term_key`` and ``op_cache_key`` — which is why it is applied HERE and not at recognition."""
    inner = place.free[-1]
    r = plan.regs[0]
    op = tile.op
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
        stores.extend(Store(write=st.write.rewrite(rename, sigma)) for st in tile.stores)
    new_inner = replace(inner, extent=Dim(inner.extent.as_static() // r))
    new_free = (*place.free[:-1], new_inner)
    new_place = Placement(free=new_free, grid=new_free)
    out = TileOp(
        op=Fold.projection(body=Body((*loads, *computes))), name=name, place=new_place, knobs=_site_knobs(knobs), stores=tuple(stores)
    )
    seal_workers(out)
    return out


def _free_option(tile: TileOp, place: Placement, plan: TilePlan, name: str, knobs: dict) -> TileOp:
    """One ``FREE`` (pointwise) row: the flat per-cell map, or the strip variant when the row's
    ``TILE`` names a register width."""
    if _strip_width(plan) > 1:
        return _strip_variant(tile, place, plan, name, knobs)
    out = TileOp(op=tile.op, name=name, place=place, knobs=_site_knobs(knobs), stores=tile.stores)
    seal_workers(out)
    return out


def _reduce_option(tile: TileOp, place: Placement, plan: ReducePlan, name: str, knobs: dict) -> TileOp:
    """One ``PLANAR`` / ``TWISTED`` row: the resolved :class:`ReducePlan` stored on the primary
    fold. A cooperative partition also derives the shared-row operand :class:`Stage` (a derived perf
    transform, stored as a slice only — never a knob)."""
    node = head(tile.op)
    slices = []
    if node is not None:
        if plan.stages:
            slices.append(("REDUCE", node, plan))
        if plan.coop > 1:
            slices.append(("STAGE", node, _row_stage(tile, place)))
    return _stamp(tile, tile.op, place, name, knobs, slices)


def _tile_option(tile: TileOp, place: Placement, plan: TilePlan, node, name: str, knobs: dict, stage_spec: str, budget: int) -> TileOp:
    """One scalar-tier contraction row. A tiled candidate contracts K serially per register cell, so
    a coop / ILP partition is DROPPED rather than stamped onto a kernel that doesn't fold it."""
    if plan.block_threads > MAX_BLOCK_THREADS:
        raise ValueError(
            f"TILE parallel block {plan.units_n}×{plan.units_m}={plan.block_threads} threads exceeds "
            f"the {MAX_BLOCK_THREADS}-thread/CTA limit; shrink n/m or move work to the f register sub-tile."
        )
    slices: list = []
    if plan.is_tiled:
        placed = _placed(place, plan)
        stage = _resolve_scalar_stage(node, placed, Stage.parse(stage_spec), tile.inputs, budget) if stage_spec else None
        slices = [("TILE", node, placed), ("STAGE", node, stage)]
    return _stamp(tile, tile.op, place, name, knobs, slices)


def _coop_contraction_option(tile: TileOp, place: Placement, node, rplan: ReducePlan, name: str, knobs: dict) -> TileOp:
    """One NON-output-tiled contraction row carrying a coop / ILP K partition — the contraction is
    the degenerate carrier of its own additive fold, so ``_factor._tile_reduce_axis`` folds the
    partition off the node exactly as it does for a plain reduce."""
    return _stamp(tile, tile.op, place, name, knobs, [("REDUCE", node, rplan)])


def _warp_option(tile: TileOp, place, plan: TilePlan, node, name: str, knobs: dict, stage_spec: str, budget: int, wspec: str) -> TileOp:
    """One warp (tensor-core) contraction row. Warp specialization rides ORTHOGONAL to the resolved
    tile/stage: the band is gated on the RESOLVED stage, so an ineligible spec degrades to uniform
    rather than claiming a pipeline that never ran."""
    _check_warp_static_k(node, plan)
    placed = _placed(place, plan)
    stage = _resolve_warp_stage(node, placed, Stage.parse(stage_spec), budget) if stage_spec else None
    workers, _ = _wspec_workers(wspec, stage, placed.launch_threads)
    return _stamp(tile, tile.op, place, name, knobs, [("TILE", node, placed), ("STAGE", node, stage)], workers=workers)


def _factor_k(k_axis: Axis, w: int) -> tuple[Axis, Axis, Sigma]:
    """Factor a STATIC contraction axis into ``ksplit × kslice``. ``ksplit`` (extent ``w``, name
    ``<k>_ks``) becomes the outer :class:`Fold`'s reduce axis, parallelized across CTAs and summed
    in the finalize; ``kslice`` (extent ``K/w``, the ORIGINAL name) stays the inner contraction's.
    The ``sigma`` maps the original ``k`` to ``ksplit·(K/w) + kslice`` so the operand loads
    reconstruct the absolute index; distinct names are what avoid a double-reduce."""
    big_k = k_axis.extent.as_static()
    if big_k % w:
        raise ValueError(f"split-K width {w} does not divide K={big_k}; pick a dividing split width.")
    b = big_k // w
    ksplit = Axis(name=f"{k_axis.name}_ks", extent=Dim(w))
    kslice = replace(k_axis, extent=Dim(b))
    sigma = Sigma({k_axis.name: BinaryExpr("+", BinaryExpr("*", Var(ksplit.name), Literal(b, "int")), Var(k_axis.name))})
    return ksplit, kslice, sigma


def _splitk_option(
    tile: TileOp, place, plan: TilePlan, node, rplan: ReducePlan, name: str, knobs: dict, stage_spec: str, budget: int
) -> TileOp:
    """One SPLIT-K contraction row — the structural ``Fold(axis=ksplit) ⊃ Fold(axis=kslice)``
    composition ``030_split_reduce`` consumes into the cross-CTA partial + finalize. The inner node
    is the SAME contraction a non-split matmul builds, over ``kslice`` with operands σ-reindexed to
    absolute k; the outer reduce is the IDENTITY-lift composition over it (``Fold.composed``), so
    its role DERIVES as ``CONTRACTION`` with no rewrite.

    Knob keying stamps against the PRE-SPLIT tree, keeping the kernel single-eligible-axis so the
    golden bare-collapse and the prior featurizer stay invariant."""
    if not plan.is_warp and plan.block_threads > MAX_BLOCK_THREADS:
        raise ValueError(
            f"TILE parallel block {plan.units_n}×{plan.units_m}={plan.block_threads} threads exceeds "
            f"the {MAX_BLOCK_THREADS}-thread/CTA limit; shrink n/m or move work to the f register sub-tile."
        )
    w = rplan.cta
    if plan.is_warp:
        step = plan.atom.atom_k * plan.bk
        ks = node.axis.extent.as_static() // w
        if ks % step:
            raise ValueError(
                f"split-K slice K={ks} (K/{w}) is not a multiple of the mma K-step {step} "
                f"(atom_k={plan.atom.atom_k}·bk={plan.bk}); pick a split width whose slice is divisible."
            )
    if any(not isinstance(ch.b, Load) for ch in node.channels):
        raise ValueError("split-K needs a materialized B on every channel — a computed B has no gmem index to σ-reindex")
    ksplit, kslice, sigma = _factor_k(node.axis, w)
    inner = Fold.contraction(
        k_axis=kslice,
        a=replace(node.a, index=tuple(sigma.apply(e) for e in node.a.index)),
        channels=tuple(replace(ch, b=replace(ch.b, index=tuple(sigma.apply(e) for e in ch.b.index))) for ch in node.channels),
    )
    placed = _placed(place, plan)
    stage = None
    if stage_spec:
        st = Stage.parse(stage_spec)
        stage = (
            _resolve_warp_stage(inner, placed, st, budget)
            if plan.is_warp
            else _resolve_scalar_stage(inner, placed, st, tile.inputs, budget)
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
    proj = _projection(tile.op)
    op = Fold.projection(body=proj, operands=(outer,)) if len(proj) else outer
    return _stamp(tile, op, place, name, knobs, [("REDUCE", outer, rplan), ("TILE", inner, placed), ("STAGE", inner, stage)])


# ---- the entry point --------------------------------------------------------------------------- #


def schedule(tile: TileOp, name: str, knobs: dict, ctx=None) -> Fork | list[TileOp] | TileOp:
    """Map a freshly-recognized (UNMAPPED) ``tile`` onto the grid and offer its scheduling fork.

    Returns the lazy fork tree over the enumerated rows (levels ``[WORK, *site keys, RASTER]`` — the
    kernel-global worker inventory leads, so every deeper prefix row is self-decoding; the
    launch-order codec closes), a single ``TileOp`` when the space collapses to one row, or ``[]``
    when nothing is enumerable (the guardrail contract — the caller leaves the term unmapped)."""
    place = tile.place.on_grid()
    rows, keys = _rows(tile, place, ctx)
    if not rows:
        return []
    role = axis_role(tile.op)
    node = head(tile.op)
    budget = _smem_budget(ctx)

    def _materialize(row: dict) -> TileOp:
        # The row carries SITE values + the ONE ``WORK`` entry: resolve each family ONCE against
        # that inventory and hand the builders typed plans. The producer band rides WORK's ``+p``.
        tkey, skey, rkey = keys
        work = Workers.parse(row.get(WORK.name) or "")
        plan = resolve_site_tile(row.get(tkey, ""), work, row.get(rkey, ""))
        stage_spec = row.get(skey, "")
        rplan = ReducePlan.parse(row.get(rkey, ""), work)
        wspec = f"p{work.producer}" if work is not None and work.producer else ""
        # Structural stamps (``S_warp_eligible``) ride onto the op: fork rows carry them for branch
        # identity, but the MATERIALIZED op is what ``realized_knobs`` reads, and dropping them here
        # left leaf/evidence rows unstamped while fork rows were stamped — fracturing the ``S_*``
        # evidence signature (the 2026-07-07 5090 gate's 330× fp16 miss).
        op_knobs = {**knobs, **{k: v for k, v in row.items() if k.startswith("S_")}}
        raster_spec = row.get(RASTER.name, "")
        Raster.parse(raster_spec)  # loud pin contract — a malformed spelling fails the row here
        op_knobs = {**op_knobs, RASTER.name: raster_spec, tkey: plan.spell(), skey: stage_spec, rkey: rplan.spell()}
        if role is AxisRole.FREE:
            return _free_option(tile, place, plan, name, op_knobs)
        if role is not AxisRole.CONTRACTION:
            return _reduce_option(tile, place, rplan, name, op_knobs)
        if rplan.stages and rplan.needs_split:
            return _splitk_option(tile, place, plan, node, rplan, name, op_knobs, stage_spec, budget)
        if plan.is_warp:
            return _warp_option(tile, place, plan, node, name, op_knobs, stage_spec, budget, wspec)
        if rplan.stages:
            return _coop_contraction_option(tile, place, node, rplan, name, op_knobs)
        return _tile_option(tile, place, plan, node, name, op_knobs, stage_spec, budget)

    if len(rows) == 1:
        return _materialize(rows[0])

    def _level(key: str) -> Level:
        return Level((key,), key=lambda r: (r.get(key, ""),))

    levels = [_level(WORK.name), *(_level(k) for k in keys), _level(RASTER.name)]
    return build_fork_tree(params=rows, levels=levels, materialize=_materialize)


__all__ = ["schedule", "warp_tile_pinned"]
