"""Schedule a recognized (UNMAPPED) ``TileOp`` — the generic row enumerator.

**Every role emits rows; no role builds ``TileOp``\\ s directly.** The whole schedule step is one
pipeline, shared by every :class:`~emmy.compiler.ir.axis.AxisRole`:

.. code-block:: text

    role   = ops.axis_role(term)                 # derived from arity, never a node type
    rows   = _ROWS[role](...)                    # per-family typed slices, assembled site-local
    #                                            # (_assemble spells each row ONCE, derives WORK)
    fork   = build_fork_tree(rows, levels=[WORK, *site keys, RASTER], materialize=_materialize)

Three layers, each with one job:

- the candidate DOMAIN is generated from its bounds in ``search/space.py`` (the tile spaces) or
  listed in its catalog there (the families with no multiplicative coupling — stages, split widths,
  the coop partitions, the raster orders);
- per-node LEGALITY — what a domain cannot know because it depends on this term's K, N, dtype and
  smem cap — is :mod:`._legality`, one predicate per rule, raise-vs-drop chosen by ``pinned``;
- THIS module chooses: which families a role offers, the conservative option-0 each leads with, and
  how a row becomes a ``TileOp``.

:func:`_assemble` derives the ONE ``WORK`` inventory from the row's own slices and drops the row
when they disagree, so a row is self-consistent before it ever reaches materialization.

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

from emmy.compiler.dim import DEFAULT_SEQ_HINT, Dim
from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.schedule import (
    Raster,
    Stage,
    TilePlan,
    WarpSpec,
    Workers,
    derive_inventory,
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
from emmy.compiler.pipeline.passes.lowering.tile import _legality as legal
from emmy.compiler.pipeline.search.space import (
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


def _pinned_workers() -> Workers | None:
    """The live ``WORK`` env pin's inventory, or ``None`` when unset. A MALFORMED pin raises — one
    severity per pin, the same doctrine :func:`._legality.enforce` states."""
    raw = WORK.raw()
    return Workers.parse(raw) if raw else None


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


def _placed(place: Placement, plan: TilePlan) -> TilePlan:
    """``plan`` bound to the ROOT contraction's ``(m, n)`` — the kernel grid's trailing pair, the
    same rule ``ops.Sched._mn_for`` states for a depth-1 site (so a reader never re-derives it)."""
    grid = tuple(place.grid)
    return plan.at(grid[-2], grid[-1]) if len(grid) >= 2 else plan


def _site_keys(op) -> tuple[str, str, str]:
    """The ``(TILE, STAGE, REDUCE)`` key triple for this kernel's primary node — bare when there is
    no node to key on (a pure pointwise cell / the raw-loop-IR escape). A node that is not a site of
    the family (a PLANAR fold's ``TILE``) keys the bare family — the decided-empty."""
    node = head(op)
    if node is None:
        return (TILE.name, STAGE.name, REDUCE.name)
    sched = Sched(op, {})
    return tuple(sched.key(f.name, node) or f.name for f in (TILE, STAGE, REDUCE))  # type: ignore[return-value]


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


def _has_contraction_tail(stmts) -> bool:
    """The post-reduce tail contracts over a NEW free axis — a ``Loop`` whose body holds an inner
    reduce ``Loop``. This is the fused norm→linear shape, distinguished from a plain softmax tail
    (a single sweep over the SAME axis). ``Body.accums`` supplies the deep accumulator scan."""
    for s in stmts:
        if isinstance(s, Loop) and any(isinstance(c, Loop) and Body(c.body).accums for c in s.body):
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
    for s in (ld for ld in Body(tail).loads if ld.is_scalar):
        if s.input in carrier_bufs and len(s.index) == n + 1 and tuple(s.index[:n]) == grid_vars:
            t = inputs.get(s.input)
            if t is not None and t.shape and t.shape[-1].is_static and t.shape[-1].as_static() == raxis.extent.as_static():
                return s.input
    return None


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
    try:
        work = derive_inventory((plan,), coop=rplan.coop, producer=aux)
    except ValueError:
        return None  # the enumerator DROPS what ``seal_workers`` raises on — same rule, one home
    return {
        tkey: plan.spell(),
        skey: stage,
        rkey: rplan.spell(),
        WORK.name: work.spell() if work is not None else "",
        RASTER.name: raster,
        **stamp,
    }


# ---- FREE: the pointwise cell ------------------------------------------------------------------ #


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


def _free_rows(tile: TileOp, place: Placement, keys, ctx) -> list[dict]:
    """The ``FREE`` rows: the flat per-cell tile (option-0), then the catalog's register-strip
    ladder. ``r`` IS the spelled ``TILE=f<r>`` — the strip is a TERM VARIANT applied at
    materialization, a function of the ROW, not a member of a pre-enumerated variant set."""
    del ctx
    pin = _pinned_tile()
    strip_ok = _strippable(tile, place)
    ext = place.free[-1].extent.as_static() if strip_ok else 0
    rows = []
    for plan in [pin] if pin is not None else [TilePlan(), *map_tile_moves()]:
        # A strip WIDTH the cell cannot carry (a stateful / sweep body, a symbolic or indivisible
        # inner extent, a warp codec on a pointwise cell) drops the row; the flat per-cell base
        # below is always offered, so a narrowing pin degrades to option-0.
        if not legal.enforce(legal.strip_width(ext, _strip_width(plan)), pinned=False):
            continue
        row = _assemble(keys, plan, "", ReducePlan(), 0, "", {})
        if row is not None:
            rows.append(row)
    return rows or [_assemble(keys, TilePlan(), "", ReducePlan(), 0, "", {})]


# ---- PLANAR / TWISTED: the reduce partition ---------------------------------------------------- #

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


def _reduce_specs(kernel, place, ctx) -> list[ReducePlan]:
    """The ``PLANAR`` / ``TWISTED`` reduce-partition candidates — option-0 is the conservative
    heuristic pick (:func:`_pick_coop`, so a cold greedy compile keeps its historical deploy), then
    the legal :func:`coop_reduce_moves` catalog + serial as fork siblings. The catalog rows are what
    keep the 16- / 32-wide reduce goldens reachable. An env pin is authoritative."""
    carrier = head(kernel.op)
    if carrier is None or carrier.role not in (AxisRole.PLANAR, AxisRole.TWISTED):
        return [ReducePlan()]  # not cooperative-eligible — the scalar serial fold
    extent = _hint_extent(carrier.axis)
    free = _free_cells(place)
    tail = projection_tail(kernel)
    coop = _pick_coop(extent, free, has_tail=_has_contraction_tail(tail))
    # The layout gate (WS5, the cold-poison hardening): at the matvec tier the coop bands are only
    # coalesced on ONE B orientation — the plain band interleaves lanes along K, the transposed
    # band sweeps lanes along the output axis. Enumeration is the single choke point every tier
    # resolves through, so the gate lives here; an env pin stays authoritative and un-gated.
    kstride = _matvec_b_kstride(kernel, carrier, place)
    deploy = ctx.validate_pins
    if deploy and REDUCE.raw() is None and kstride == 1 and extent >= _COOP_MIN_EXTENT and free >= _FREE_CAP:
        return [ReducePlan.of(coop=32)]
    if coop > 1 and kstride is not None and kstride != 1:
        coop = 1  # the heuristic option-0 is a plain band too — uncoalesced on a k-major B
    cands = [ReducePlan.of(coop=coop)]
    tail_scalar = not any(isinstance(s, Loop) for s in tail)
    inner = _inner_free(place)
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


def _reduce_rows(tile: TileOp, place: Placement, keys, ctx) -> list[dict]:
    """The ``PLANAR`` / ``TWISTED`` rows — the reduce partition is the only family with a choice."""
    rows = []
    for rplan in _reduce_specs(tile, place, ctx):
        row = _assemble(keys, TilePlan(), "", rplan, 0, "", {})
        if row is not None:
            rows.append(row)
    return rows


# ---- CONTRACTION: the tile × stage × reduce × wspec × raster product --------------------------- #

# The mma atoms eligible per operand dtype — the warp tier's dtype gate (16-bit operands only).
_ATOMS_BY_DTYPE = {"f16": ("mma_m16n8k16_f16_f32",), "bf16": ("mma_m16n8k16_bf16_f32",)}

# Emit unpinned split-K candidates only when the output grid alone leaves the GPU under-occupied.
_SPLITK_MAX_CTAS = 1024

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
    return ctx.compute_capability in _F16ACC_CCS


def _warp_atoms(kernel, node, proj: Body, ctx) -> tuple[str, ...]:
    """The dtype-eligible tensor-core atom names for this contraction, ``()`` when the warp tier
    doesn't apply (no node, a non-16-bit operand dtype, or a fragment-unrealizable gather
    epilogue), extended with the f16-accumulate siblings when :func:`_f16acc_allowed`. Reads pure
    algebra off the STORED node — the placement / tile would be unread."""
    if node is None or not kernel.inputs or legal.fragment_epilogue(proj) is not None:
        return ()
    if not isinstance(node.a, Load):
        return ()  # a computed cone is out of this cut's scope (see the module docstring)
    t = kernel.inputs.get(node.a.input)
    atoms = _ATOMS_BY_DTYPE.get(getattr(getattr(t, "dtype", None), "name", None), ())
    if not atoms or not _f16acc_allowed(ctx):
        return atoms
    return atoms + tuple(_F16ACC_ATOMS[a] for a in atoms if a in _F16ACC_ATOMS)


def _tile_area(plan: TilePlan) -> int:
    """The output cells one CTA covers under ``plan`` — the occupancy denominator."""
    am, an = (plan.atom.atom_m, plan.atom.atom_n) if plan.is_warp else (1, 1)
    return max(plan.units_m * plan.reg_m * am * plan.units_n * plan.reg_n * an, 1)


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
        r = (
            legal.resolve_warp_stage(node, tile, st, budget)
            if plan.is_warp
            else legal.resolve_scalar_stage(node, tile, st, kernel.inputs, budget)
        )
        return r.spell() if r is not None else None

    if STAGE.raw() is not None:
        # A malformed pin RAISES through ``Stage.parse`` — this used to be swallowed into
        # gmem-direct, which made it the only silently-ignored pin in the family.
        pinned = STAGE.narrow([""])[0]
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
    split. A well-formed but FOREIGN codec (a coop pin where split-K is wanted) is legitimately not
    a split-K request and returns ``None``; a MALFORMED one raises, as it does everywhere else."""
    plan = _pinned_reduce()
    return plan if plan is not None and plan.needs_split else None


def _coop_reduce_spec() -> ReducePlan | None:
    """The pinned cooperative / ILP K partition a NON-output-tiled contraction honors (the split-K
    takes the structural fork instead); ``None`` when the pin is a foreign codec."""
    plan = _pinned_reduce()
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
        inner = _inner_free(place)
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
    if k is not None and _free_cells(place) // _tile_area(plan) <= _SPLITK_MAX_CTAS:
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


def _wspec_bands(plan: TilePlan, placed: TilePlan, stage_spelling: str, red: ReducePlan) -> list[int]:
    """The producer-band widths (in warps) one enumerated row may carry — ``[0]`` (uniform SIMT)
    alone unless the row can drive a band: a warp tile over a resolved TMA stage and no cross-CTA
    split, whose thread budget the band fits.

    The budget is checked HERE, at enumeration, not at materialization. ``_assemble`` spells the
    band into the row's ``WORK`` as ``+p<n>``, so an over-budget band checked later yields a row
    whose stamped inventory claims a producer while its op carries none — the knob row and the
    kernel disagree, which is the ``S_warp_eligible`` fracture in another costume.

    The catalog's own ``""`` IS the uniform band, and ``Knob.narrow`` drops it under a ``WSPEC``
    pin — so the pin stays authoritative and uniform is never re-offered beside it."""
    if not (plan.is_warp and "tma" in stage_spelling and not red.needs_split):
        return [0]
    out = []
    for spec in WSPEC.narrow(wspec_moves()):
        if not spec:
            out.append(0)
            continue
        ws = WarpSpec.parse(spec)
        if legal.enforce(legal.producer_band(ws, placed.launch_threads), pinned=WSPEC.raw() is not None):
            out.append(ws.aux_warps)
    return out or [0]


def _wspec_workers(spec: str, stage) -> WarpSpec | None:
    """The ``WSPEC`` worker split a materialized row carries, or ``None`` (uniform SIMT). The spec
    is one this module SPELLED (``p<n>``, from :func:`_wspec_bands`), so only the stage-role
    legality — a producer drives a resolved TMA stage only — can still refuse it here."""
    if not spec:
        return None
    ws = WarpSpec.parse(spec)
    return ws if ws.is_legal(SimpleNamespace(stage=stage)) else None


def _contraction_rows(tile: TileOp, place: Placement, keys, ctx) -> list[dict]:
    """The ``CONTRACTION`` rows: the tile × stage × reduce × wspec × raster legal product over the
    scalar and warp tiers. A COMPUTED operand edge is out of this cut's scope and yields none."""
    node = head(tile.op)
    if node is None or not is_contraction(node) or not isinstance(node.a, Load):
        return []
    proj = _projection(tile.op)
    budget, tma_ok = ctx.max_dynamic_smem, ctx.has_tma
    pin = _pinned_tile()
    atoms = _warp_atoms(tile, node, proj, ctx)
    warp_moves = [p for p in warp_tile_moves(atoms) if legal.enforce(legal.warp_k_step(node, p), pinned=False)] if atoms else []
    tiles = [pin] if pin is not None else scalar_tile_moves() + warp_moves
    # Warp-eligibility is a structural fact about the KERNEL: when the enumeration offers any
    # tensor-core row, EVERY row carries ``S_warp_eligible`` so the priors can price "a scalar tile
    # where tensor cores were on offer". ``S_``-prefixed — not a schedule family, so tile identity
    # and prefix-consistency are untouched.
    stamp: dict = {"S_warp_eligible": 1.0} if warp_moves else {}
    rows: list[dict] = []
    for plan in tiles:
        if plan.is_warp and pin is not None:
            # A PIN with an indivisible K-step or a gather epilogue RAISES — same predicates the
            # unpinned path above drops on, one home each.
            legal.enforce(legal.warp_k_step(node, plan), pinned=True)
            legal.enforce(legal.fragment_epilogue(proj), pinned=True)
        placed = _placed(place, plan)
        for stage in _stage_values(tile, node, place, plan, budget, tma_ok):
            for red in _reduce_values(tile, place, plan, node, len(node.channels)):
                for aux in _wspec_bands(plan, placed, stage, red):
                    for raster in _raster_values(place):
                        row = _assemble(keys, plan, stage, red, aux, raster, stamp)
                        if row is not None:
                            rows.append(row)
    return rows


# The ONE role dispatch — keyed on the derived ``AxisRole``, never on a node type (there is only one
# stored node kind). Every role emits rows; no role builds ``TileOp``s.
_ROWS = {AxisRole.FREE: _free_rows, AxisRole.CONTRACTION: _contraction_rows}


def _rows(tile: TileOp, place: Placement, ctx) -> tuple[list[dict], tuple[str, str, str]]:
    """Every legal schedule row for ``tile``, in the site value grammar, plus the
    ``(TILE, STAGE, REDUCE)`` key triple. An empty result is the guardrail contract, never a raise:
    the caller leaves the term unmapped."""
    keys = _site_keys(tile.op)
    emit = _ROWS.get(axis_role(tile.op), _reduce_rows)
    rows = [r for r in emit(tile, place, keys, ctx) if r is not None]
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
    legal.enforce(legal.scalar_block_threads(plan), pinned=True)
    slices: list = []
    if plan.is_tiled:
        placed = _placed(place, plan)
        stage = legal.resolve_scalar_stage(node, placed, Stage.parse(stage_spec), tile.inputs, budget) if stage_spec else None
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
    legal.enforce(legal.warp_k_step(node, plan), pinned=True)
    placed = _placed(place, plan)
    stage = legal.resolve_warp_stage(node, placed, Stage.parse(stage_spec), budget) if stage_spec else None
    workers = _wspec_workers(wspec, stage)
    return _stamp(tile, tile.op, place, name, knobs, [("TILE", node, placed), ("STAGE", node, stage)], workers=workers)


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
    placed = _placed(place, plan)
    stage = None
    if stage_spec:
        st = Stage.parse(stage_spec)
        stage = (
            legal.resolve_warp_stage(inner, placed, st, budget)
            if plan.is_warp
            else legal.resolve_scalar_stage(inner, placed, st, tile.inputs, budget)
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


def schedule(tile: TileOp, name: str, knobs: dict, ctx) -> Fork | list[TileOp] | TileOp:
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
    budget = ctx.max_dynamic_smem

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


__all__ = ["schedule"]
