"""Schedule a lifted kernel onto the thread grid (+ pick the reduce partition / output tile).

The scheduling **half** of the merged ``010_recognize`` tile-lowering pass — recognition
builds an UNMAPPED :class:`~emmy.compiler.ir.tile.ir.TileOp` (the structural-IR root ``op`` +
a ``place`` carrying just the free axes) and calls :func:`schedule` here in the same rewrite (no
separate ``020`` pass). Scheduling binds the placement's ``free`` axes onto the grid
(``Placement.on_grid``) and offers the per-axis
scheduling forks — the reduce-axis **partition** (:class:`~...schedule.ReducePlan`, the
``REDUCE`` codec) for a reduce axis and the output **tile** (:class:`~...schedule.TilePlan`,
the ``TILE`` codec) for a contraction — read off the axes' :class:`~...axis.AxisRole`, never a
kernel kind. This is a helper module (``_``-prefixed, not a standalone rule); its knob
constants still register (``knob._walk_modules`` walks every imported module under the package).

This cut picks a **whole-CTA cooperative** partition for a **static, scalar-output,
degenerate-monoid** reduce (plain ``sum`` / ``max`` / ``mean``) when the reduce axis is
wide and the output grid is small enough to leave the GPU under-occupied — one CTA per
output cell, ``coop`` threads cooperatively folding the reduce axis (the combine is
materialized in ``lowering/kernel``). Everything else (pointwise ``Map``, twisted /
full-row reductions like online-softmax & RMSNorm, contractions, symbolic axes) keeps the
**scalar serial** fold (``ReducePlan()`` — one thread per output cell).

The selection here is **conservative module constants** standing in for the eventual
``REDUCE`` knob + prior-driven choice. ``# TODO``: replace the constants with
``knob.py::_reduce_decomp`` (BR→coop, BK→serial, FK→reg, SPLITK→cta) + the learned /
analytic prior. The cross-CTA ``g<n>`` split (``030_split``) and the ``r<n>`` (ILP) reg
fold are built and honored for an additive carrier via an explicit ``REDUCE`` pin (the
split emits the partial + finalize kernels / atomicAdd; the reg fold emits the ILP
accumulators). Strided-cooperative rows (a small whole free axis packed alongside the coop
lanes), the symbolic-axis cooperative tier, the twisted-carrier (flash) cross-CTA split,
and flash cooperative-KV remain future steps.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from math import prod
from types import SimpleNamespace

from emmy.compiler.context import STATIC_SMEM_CAP
from emmy.compiler.dim import DEFAULT_SEQ_HINT, Dim
from emmy.compiler.dtype import F32
from emmy.compiler.ir.atom import ATOM_REGISTRY
from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.schedule import Stage, WarpSpec, is_warp_codec
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Stmt
from emmy.compiler.ir.tile import Contraction, Map, Placement, ReducePlan, Reduction, TileOp, TilePlan
from emmy.compiler.ir.tile.ops import axis_role, nodify_reduce, reduce_loop
from emmy.compiler.pipeline.fork import Fork, Level, build_fork_tree
from emmy.compiler.pipeline.passes.lowering.tile._atomize import map_cone, semiring_binding
from emmy.compiler.pipeline.passes.lowering.tile._carrier import projection_distributes
from emmy.compiler.pipeline.pipeline import LoweringError
from emmy.compiler.pipeline.search.space import (
    MAX_BLOCK_THREADS,
    REDUCE,
    STAGE,
    TILE,
    WSPEC,
    coop_reduce_moves,
    scalar_tile_moves,
    splitk_moves,
    stage_moves,
    warp_tile_moves,
    wspec_moves,
)

logger = logging.getLogger(__name__)

# The schedule codec knobs (``REDUCE`` / ``TILE`` / ``STAGE`` / ``WSPEC``) and the enumeration
# value grids are declared in ``search/space.py`` (the one search-space file) and imported here,
# where they are resolved into the schedule slices. The decision hierarchy for each is the env
# pin (via ``Knob.narrow``) > the search/prior fork > the conservative default below.


def _smem_budget(ctx) -> int:
    """The per-block smem budget the stage resolvers size against — the device's dynamic-smem
    opt-in cap (``ctx.max_dynamic_smem``; the backend declares an ``extern __shared__`` pool and
    sets the func attribute past the static cap), or the 48 KiB static floor when no ``Context``
    reaches the schedule (direct unit-test drives)."""
    return ctx.max_dynamic_smem if ctx is not None else STATIC_SMEM_CAP


def _at(knob, axis_name: str) -> str:
    """The axis-named knob key ``FAMILY@<axis>`` (e.g. ``TILE@d``) — the per-node schedule codec keyed
    by the reduce/contraction axis it schedules, so a multi-node kernel addresses each node."""
    return f"{knob.name}@{axis_name}"


# Conservative cooperative-reduce selection constants (the default when REDUCE is unpinned).
_COOP_MIN_EXTENT = 128  # only cooperate when the reduce axis is at least this wide
_SERIAL_TARGET = 8  # aim for ~this many serial steps per cooperating thread
_MAX_COOP = 256  # cap on cooperative threads per CTA (power of two)
_FREE_CAP = 256  # only cooperate when the output grid is at most this many cells (under-occupied)


def _hint_extent(ax) -> int:
    """An axis's static extent, or its ``Dim`` hint when symbolic (the occupancy heuristic
    sizes a dynamic axis by its hint; the kernel still deploys over the runtime extent)."""
    e = ax.extent
    return e.as_static() if e.is_static else (e.hint or DEFAULT_SEQ_HINT)


def _prevpow2(n: int) -> int:
    """The largest power of two ≤ ``n`` (≥ 1)."""
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def _pick_coop(extent: int, free: int, *, has_tail: bool = False) -> int:
    """The conservative whole-CTA cooperative-thread count for a reduce of static
    ``extent`` over ``free`` output cells, or ``1`` (stay scalar/serial). Cooperate only on
    a wide reduce (``extent ≥ _COOP_MIN_EXTENT``) feeding a small grid (``free ≤
    _FREE_CAP`` — otherwise the scalar tier already saturates the GPU); the count targets
    ``_SERIAL_TARGET`` serial steps, capped at ``_MAX_COOP``, rounded to a power of two (the
    butterfly / tree reorder). ``has_tail`` lifts the free-grid cap: a fused contraction
    tail multiplies each cell's work by its column extent, so "the scalar tier saturates"
    does not hold — the fused final-norm → lm_head (151k columns) runs MINUTES one-thread-
    per-row, while the cooperative row distributes the tail across the lanes."""
    if extent < _COOP_MIN_EXTENT or (free > _FREE_CAP and not has_tail):
        return 1
    coop = min(_prevpow2(extent // _SERIAL_TARGET), _MAX_COOP)
    return coop if coop >= 2 else 1


def _coop_carrier(kernel):
    """The cooperative-eligible reduce ``Loop`` of ``kernel`` (read for its ``axis``), or ``None``
    (keep serial).

    Eligible: a ``PLANAR`` / ``TWISTED`` reduce loop — **degenerate** (plain ``sum`` / ``max`` /
    ``mean``) AND **twisted** (online-softmax ``(m, d)``, flash ``(m, l, O)``) alike, since the
    cross-thread combine is carrier-generic (it drives off the carrier's ``combine_states``, which
    a twisted carrier authors). Both **scalar** outputs (flash's ``O/l`` per ``(m, d)`` cell — ``d``
    is a grid axis) and **full-row** outputs (softmax / RMSNorm — the post-reduce sweep is
    distributed across the coop lanes by the materializer) are handled. The reduce axis may be
    **symbolic** (dynamic ``seq_len``): each lane strides it to the runtime extent (the ``< seq_len``
    bound is the masked tail). A ``CONTRACTION`` (its output tile is ``_tile_option`` / ``_warp_option``;
    a cross-CTA split-K is the ``_splitk_option`` fork) or a flat-``Map`` fallback (multi /
    nested-non-flash reduce — no annotated reduce loop) is not eligible here and keeps the serial fold."""
    rl = reduce_loop(kernel.op)
    if rl is None or rl.role not in (AxisRole.PLANAR, AxisRole.TWISTED):
        return None
    return rl


def _reduce_specs(kernel, place) -> list[str]:
    """The candidate ``REDUCE`` codec strings for ``kernel``, applying the decision
    hierarchy. A kernel the cooperative tier can't partition (pointwise, or a twisted /
    full-row / contraction reduce) is the lone scalar fold ``[""]`` — the ``REDUCE`` pin is
    ignored there, since it only governs the cooperative reduce tier. An eligible reduce
    offers ``[conservative coop, scalar]`` (a fork the search / prior ranks, option-0 = the
    conservative pick so a cold greedy compile keeps cooperating), with an env pin
    (``EMMY_REDUCE``) authoritative over the candidates (``Knob.narrow``)."""
    carrier = _coop_carrier(kernel)
    if carrier is None:
        return [""]  # not cooperative-eligible — scalar serial fold; the pin doesn't apply
    # A symbolic reduce axis is sized by its ``Dim`` hint for the conservative pick (the
    # kernel deploys at the hint and strides to the runtime extent); a pin overrides it.
    extent = _hint_extent(carrier.axis)
    # A symbolic free axis (dynamic-grid tier) is sized by its ``Dim`` hint for the occupancy
    # heuristic — the kernel still deploys over the runtime grid. A fused contraction tail
    # (the norm→linear shape — ``_has_contraction_tail``) lifts the free-grid cap: per-cell
    # work scales with the tail's columns, so a big grid does NOT mean the serial tier saturates.
    free = prod(_hint_extent(a) for a in place.free) if place.free else 1
    tail = list(kernel.op.body) if isinstance(kernel.op, Map) else []
    coop = _pick_coop(extent, free, has_tail=_has_contraction_tail(tail))
    cands = [f"b{coop}", ""] if coop > 1 else [""]  # conservative coop first (cold greedy → option-0)
    return list(REDUCE.narrow(cands))


def _with_reduce(op, plan: ReducePlan):
    """Stamp the chosen ``ReducePlan`` onto the op's :class:`Reduction` node (bare, or wrapped under a
    projecting :class:`Map`). The reduce partition lives **on the node**, not the ``TileSchedule`` —
    read back via ``ops.reduce_plan``. ``_option`` only schedules a PLANAR / TWISTED reduce, whose op
    recognition always emits as a bare ``Reduction`` or a projecting ``Map(source=Reduction)``."""
    if isinstance(op, Reduction):
        return replace(op, reduce=plan)
    assert isinstance(op, Map) and isinstance(op.source, Reduction), f"reduce op must nodify to Reduction, got {type(op).__name__}"
    return replace(op, source=replace(op.source, reduce=plan))


# ---- shared-row operand staging (the fused norm→linear prologue) -------------------------------- #
# The reduce tier's one staging move: when an input row is folded by the cooperative reduce AND
# re-read per output column of a contraction tail (the fused RMSNorm→linear shape), stage it into
# smem once and share it across both readers. The DETECTION lives here — stamped as a first-class
# ``sync`` :class:`Stage` whose ``smem`` names the row buffer — and ``_factor._tile_reduce_axis`` only
# APPLIES it (fill + load-rewrite), the same Stage → apply path the contraction tiers follow. Not a
# knob: it fires whenever the cooperative partition is chosen and the shape qualifies (a pure perf
# transform), so nothing is spelled on ``knobs`` and the prior featurization is untouched.


def _scalar_loads(stmts: list[Stmt]) -> list[Load]:
    """Every scalar ``Load`` reachable in ``stmts`` (deep)."""
    out: list[Load] = []
    for s in stmts:
        if isinstance(s, Load) and s.is_scalar:
            out.append(s)
        for b in s.nested():
            out.extend(_scalar_loads(list(b)))
    return out


def _has_accum(stmts: list[Stmt]) -> bool:
    return any(isinstance(s, Accum) or any(_has_accum(list(b)) for b in s.nested()) for s in stmts)


def _has_contraction_tail(stmts: list[Stmt]) -> bool:
    """The post-reduce tail contracts over a NEW free axis — a ``Loop`` (the free output
    axis) whose body holds an inner reduce ``Loop`` (an ``Accum``). This is the fused
    norm→linear shape (``for n: for k: acc += …``), and it distinguishes it from a plain
    softmax tail (a single ``for k`` sum over the SAME reduce axis, no nested contraction).
    Only the former benefits from staging the shared input row — and only it is staged."""
    for s in stmts:
        if isinstance(s, Loop) and any(isinstance(c, Loop) and _has_accum(list(c.body)) for c in s.body):
            return True
        if any(_has_contraction_tail(list(b)) for b in s.nested()):
            return True
    return False


def _shared_row_buf(carrier_body, tail: list[Stmt], grid_vars: tuple, raxis: Axis, inputs: dict) -> str | None:
    """The input buffer reused as a CTA-shared ROW across the reduce + a contraction tail — an
    input read in the carrier reduce at ``(grid…, raxis)`` AND in the tail at ``(grid…, k)``,
    whose trailing dim is the (static) reduce extent. That row (e.g. RMSNorm's ``x[m, :]``,
    folded by the mean reduce then re-read per output column of the fused linear) is the one
    operand worth staging into smem. ``None`` ⇒ no eligible operand (stay gmem-direct)."""
    if not raxis.extent.is_static or not _has_contraction_tail(tail):
        return None
    n = len(grid_vars)
    carrier_bufs = {
        s.input
        for s in _scalar_loads(list(carrier_body))
        if len(s.index) == n + 1 and tuple(s.index[:n]) == grid_vars and s.index[-1] == Var(raxis.name)
    }
    for s in _scalar_loads(tail):
        if s.input in carrier_bufs and len(s.index) == n + 1 and tuple(s.index[:n]) == grid_vars:
            t = inputs.get(s.input)
            if t is not None and t.shape and t.shape[-1].is_static and t.shape[-1].as_static() == raxis.extent.as_static():
                return s.input
    return None


def _row_stage(tile, place) -> Stage | None:
    """The shared-row :class:`Stage` for a **cooperative** reduce ``tile``, or ``None`` (no eligible
    row — gmem-direct). Reads the reduce loop / projection tail off the node tree (the same stmts the
    materializer emits) and the operand shapes off ``tile.inputs`` (seeded from the recognized
    ``LoopOp``); the stamped stage is the depth-1 ``sync`` transport with ``smem`` naming the row."""
    rloop = reduce_loop(tile.op)
    tail = list(tile.op.body) if isinstance(tile.op, Map) else []
    grid_vars = tuple(Var(a.name) for a in place.grid)
    buf = _shared_row_buf(rloop.body, tail, grid_vars, rloop.axis, tile.inputs)
    return Stage(transport="sync", smem=(buf,)) if buf is not None else None


def _option(tile, place, spec: str, name: str, knobs: dict) -> TileOp:
    """One scheduled ``TileOp``: ``place`` mapped onto the grid + the ``REDUCE`` spec resolved into the
    :class:`Reduction` node's ``ReducePlan`` (the ephemeral knob → materialized plan stamped **on the
    node**), with the spec stamped on ``knobs`` for the prior. The spec is keyed ``REDUCE@<axis>``
    (the reduce axis this node partitions), so a multi-node kernel addresses each reduce. A
    cooperative partition also derives the shared-row operand :class:`Stage` (:func:`_row_stage`,
    stamped on the schedule field only — a derived perf transform, never a knob)."""
    plan = ReducePlan.parse(spec)
    op = _with_reduce(tile.op, plan)
    raxis = reduce_loop(tile.op).axis.name
    stage = _row_stage(tile, place) if plan.coop > 1 else None
    return TileOp(op=op, name=name, place=place, stage=stage, knobs={**knobs, _at(REDUCE, raxis): spec})


# The mma atoms eligible per operand dtype — the warp tier's dtype gate (16-bit operands only).
_ATOMS_BY_DTYPE = {"f16": ("mma_m16n8k16_f16",), "bf16": ("mma_m16n8k16_bf16",)}

# Emit unpinned split-K candidates only when the output grid alone leaves the GPU under-occupied —
# split-K beyond the ~2-wave occupancy need is pure combine/workspace waste (the prior's
# ``D_splitk_excess`` prices the remainder; this gate keeps the obviously-pointless rows out).
_SPLITK_MAX_CTAS = 512


def _fragment_epilogue_ok(epilogue: Body) -> bool:
    """The mma store folds the projection into a :class:`RegEpilogue` whose leaf ``Load``\\ s are
    evaluated independently per fragment element — a load whose INDEX reads a name defined by an
    earlier epilogue stmt (an embedding-gather chain, ``emb[(int)ids[m], n]``) cannot be threaded
    through that form. Gate in the negative: walk the epilogue and refuse on the first
    data-dependent index; everything else folds."""
    defs: set[str] = set()
    for s in epilogue:
        if isinstance(s, Load) and {v for e in s.index for v in e.free_vars()} & defs:
            return False
        defs.update(s.defines())
    return True


def _warp_atoms(kernel, probe) -> tuple[str, ...]:
    """The dtype-eligible tensor-core atom names for this contraction, ``()`` when the warp tier
    doesn't apply (unbindable node, a non-16-bit operand dtype, or a fragment-unrealizable gather
    epilogue). A computed-A (fused-cone) node reads its operand dtype off the cone's K-indexed
    ``Load`` — the value the sync compute-fill stores to the A slab."""
    if probe is None or not kernel.inputs:
        return ()
    if not _fragment_epilogue_ok(probe.epilogue):
        return ()
    if isinstance(probe.a_operand, Load):
        ld = probe.a_operand
    else:
        kname = probe.k_axis.name
        ld = next((st for st in probe.a_body if isinstance(st, Load) and kname in {v for e in st.index for v in e.free_vars()}), None)
    t = kernel.inputs.get(ld.input) if ld is not None else None
    return _ATOMS_BY_DTYPE.get(getattr(getattr(t, "dtype", None), "name", None), ())


def _warp_move_ok(kernel, spec: str) -> bool:
    """The enumeration-side (filtering) form of :func:`_check_warp_static_k` — an unpinned warp
    move whose K-step doesn't divide the static contraction K is silently dropped (a PIN with the
    same defect still raises, in :func:`_tile_rows`)."""
    try:
        _check_warp_static_k(kernel, TilePlan.parse(spec))
    except ValueError:
        return False
    return True


def _tile_area(plan: TilePlan) -> int:
    """The output cells one CTA covers under ``plan`` — the occupancy denominator."""
    am, an = (plan.atom.atom_m, plan.atom.atom_n) if plan.is_warp else (1, 1)
    return max(plan.units_m * plan.reg_m * am * plan.units_n * plan.reg_n * an, 1)


def _stage_candidates(kernel, probe, plan: TilePlan, budget: int = STATIC_SMEM_CAP) -> list[str]:
    """The RESOLVED operand-stage spellings for one tile candidate — gmem-direct ``""`` first, then
    every grid move that resolves against the node with this ``plan`` (:func:`_resolve_warp_stage` /
    :func:`_resolve_scalar_stage`); the row carries the resolved spelling so the leaf identity, the
    stamped knobs, and the kernel agree. A pinned ``STAGE`` is authoritative: the resolved pin
    alone, or gmem-direct when it declines (the standard pin-validity degrade)."""
    if probe is None or not plan.is_tiled:
        return [""]  # per-cell / unbindable — no operand slab to stage
    node = replace(probe, tile=plan)

    def resolve(spec: str) -> str | None:
        st = Stage.parse(spec)
        r = _resolve_warp_stage(node, st, budget) if plan.is_warp else _resolve_scalar_stage(node, st, kernel.inputs, budget)
        return r.spell() if r is not None else None

    if STAGE.raw() is not None:
        pinned = _stage_spec(kernel)
        r = resolve(pinned) if pinned else None
        return [r] if r else [""]
    out = [""]
    for move in stage_moves(warp=plan.is_warp):
        r = resolve(move) if move else None
        if r and r not in out:
            out.append(r)
    return out


def _reduce_candidates(kernel, place, plan: TilePlan, probe: Contraction | None = None) -> list[str]:
    """The ``REDUCE`` codec candidates for one tile candidate — serial ``""`` first (option-0),
    then the legal coop / ILP moves (per-cell tier only — the non-output-tiled contract) and the
    divisor- and occupancy-guarded split-K moves (deferred-only on the warp tier). An **atomic**
    split (``g<w>a``) is offered only when the kernel's projection epilogue distributes over the
    add (``projection_distributes`` off the ``probe`` node) — a non-distributive fused projection
    would raise at ``030_split`` and waste a search slot; the deferred ``g<w>k`` finalize stays
    legal for any epilogue. A pinned ``REDUCE`` is authoritative and keeps the pin contract: a
    ``g`` split rides every tile (an invalid warp slice / atomic-on-non-distributive raises in
    :func:`_splitk_option` / ``030_split``, as a pin should), a ``b``/``r`` partition applies to
    the per-cell tier only (a tiled candidate has no row under it)."""
    ext = reduce_loop(kernel.op).axis.extent
    if REDUCE.raw() is not None:
        split = _splitk_pin()
        if split:
            return [split]
        coop = _coop_reduce_spec()
        if coop:
            return [coop] if not plan.is_tiled else []
        return [""]
    out = [""]
    k = ext.as_static() if ext.is_static else None
    if k is not None and not plan.is_tiled:
        for move in coop_reduce_moves():
            p = ReducePlan.parse(move)
            if p.coop <= k and p.reg <= k:
                out.append(move)
    free = prod(_hint_extent(a) for a in place.free) if place.free else 1
    # A computed-A (fused-cone) contraction offers no split-K: the split σ-reindexes the operand
    # Loads, and a producer-cone A cannot be sliced over K (its statistic prologue spans the row).
    if k is not None and free // _tile_area(plan) <= _SPLITK_MAX_CTAS and (probe is None or not probe.a_computed):
        step = plan.atom.atom_k * plan.bk if plan.is_warp else 1
        atomic_ok = probe is not None and (len(probe.epilogue) == 0 or projection_distributes(probe.epilogue, (probe.acc,)))
        for move in splitk_moves(warp=plan.is_warp):
            sp = ReducePlan.parse(move)
            if sp.finalize == "atomic" and not atomic_ok:
                continue  # non-distributive fused projection — 030_split would raise; don't offer
            if k % sp.cta == 0 and (k // sp.cta) % step == 0:
                out.append(move)
    return out


def _wspec_candidates(plan: TilePlan, stage_spelling: str, red: str) -> list[str]:
    """The ``WSPEC`` candidates for one enumerated (tile, stage, reduce) row — uniform ``""``
    alone unless the row can drive a producer band: a warp tile over a resolved **TMA** stage
    (:func:`_wspec_workers`'s legality, pre-filtered here so the fork doesn't spawn rows that
    materialize identically) and no cross-CTA split (the split partial re-resolves its own
    pipeline; wiring WSPEC through ``030_split`` is a follow-up). The env pin narrows as usual —
    and only eligible rows offer it, so a pinned split on an ineligible row degrades to uniform
    at materialization rather than multiplying dead rows here."""
    try:
        needs_split = bool(red) and ReducePlan.parse(red).needs_split
    except ValueError:
        needs_split = False
    if not (plan.is_warp and "tma" in stage_spelling and not needs_split):
        return [""]
    return list(WSPEC.narrow(wspec_moves()))


def _tile_rows(kernel, place, ctx=None) -> tuple[list[dict], str]:
    """The contraction's enumerated knob rows (the tile × stage × reduce × wspec legal product,
    each schedule family keyed ``FAMILY@<k_axis>``, ``WSPEC`` bare/root-global) and the k-axis
    name. Env pins narrow each family (``Knob.narrow``); the unpinned families come from the
    ``search/space.py`` move catalog, legality-guarded here (the per-node half of the space)."""
    kaxis = reduce_loop(kernel.op).axis.name
    try:
        probe = _contraction_node(kernel.op, place, TilePlan())
    except LoweringError:
        probe = None
    if probe is not None and probe.a_computed:
        return _computed_a_rows(kernel, place, probe, kaxis, _smem_budget(ctx)), kaxis
    tiles = scalar_tile_moves() if probe is not None else [""]
    if probe is not None:
        atoms = _warp_atoms(kernel, probe)
        if atoms:
            tiles += [s for s in warp_tile_moves(atoms) if _warp_move_ok(kernel, s)]
    tiles = list(TILE.narrow(tiles))
    rows: list[dict] = []
    for spec in tiles:
        plan = TilePlan.parse(spec)
        if plan.is_warp and TILE.raw() is not None:
            _check_warp_static_k(kernel, plan)  # a PIN with an indivisible K-step raises (the pin contract)
            if probe is not None and not _fragment_epilogue_ok(probe.epilogue):
                raise ValueError(
                    "warp TILE pin: the projection epilogue gathers through another epilogue "
                    "load (a data-dependent index) — the fragment epilogue cannot thread it; "
                    "drop the a:<atom> token to use the scalar tier."
                )
        for stage in _stage_candidates(kernel, probe, plan, _smem_budget(ctx)):
            for red in _reduce_candidates(kernel, place, plan, probe):
                # A staged split row is legal: ``_splitk_option`` re-resolves the stage against the
                # SLICED inner node (the warp slice divisibility already held in
                # ``_reduce_candidates``) and ``030_split`` threads it onto the partial kernel.
                # Every family key is explicit — ``""`` is a DECIDED empty (per-cell / serial /
                # gmem-direct), distinguishable from an absent (never-offered) family. The
                # evidence pick's prefix-consistency depends on it: an absent key reads as
                # "free" and would let a gmem-direct leaf inherit a staged row's measurement.
                for wspec in _wspec_candidates(plan, stage, red):
                    rows.append({_at(TILE, kaxis): spec, _at(STAGE, kaxis): stage, _at(REDUCE, kaxis): red, WSPEC.name: wspec})
    return rows, kaxis


def _splitk_pin() -> str:
    """The pinned ``g<w>[a|k]`` split-K spec (or ``""``) — the cross-CTA K partition a
    ``CONTRACTION`` honors through the structural ``Reduction ⊃ Contraction`` fork
    (:func:`_splitk_option`), consumed by ``030_split``. Reads the ``REDUCE`` pin and returns it
    only when it parses to a **GRID split** (``needs_split``); a non-split ``b`` / ``r`` pin or
    another codec is not a split-K request — ignore it rather than fail."""
    pinned = REDUCE.narrow([""])[0]
    try:
        plan = ReducePlan.parse(pinned)
    except ValueError:
        return ""
    return pinned if plan.needs_split else ""


def _coop_reduce_spec() -> str:
    """The pinned cooperative (``b``) / ILP (``r``) K partition a **non-output-tiled** ``CONTRACTION``
    honors — folded through ``_factor._tile_reduce_axis`` (a contraction is the degenerate carrier of
    its additive fold), riding the residual ``reduce`` field on the still-``Map`` scalar tier. Returns
    the ``REDUCE`` pin iff it parses to a coop / reg partition WITHOUT a GRID split (the split-K ``g``
    takes the structural :func:`_splitk_option` fork instead); ``""`` otherwise (a foreign codec is
    not ours — ignore it rather than fail)."""
    pinned = REDUCE.narrow([""])[0]
    try:
        plan = ReducePlan.parse(pinned)
    except ValueError:
        return ""
    return pinned if (not plan.needs_split and (plan.coop > 1 or plan.reg > 1)) else ""


def _stage_spec(kernel) -> str:
    """The pinned ``STAGE`` codec for ``kernel`` — only a ``CONTRACTION`` contraction stages its
    operands today (everything else is ``""``, the pin doesn't apply). Returns the authoritative
    ``EMMY_STAGE`` pin (``Knob.narrow``) or ``""`` (unpinned — the enumeration's resolver-gated
    grid takes over, see :func:`_stage_candidates`). A pin that doesn't parse as the ``STAGE`` codec (e.g. a bare operand
    binmask ``"11"``) is **structurally invalid** for this tier, so it degrades to ``""``
    (gmem-direct) rather than failing the lowering — the same pin-validity rule the other
    codecs follow. The returned spec is only the requested *spelling*; each option builder RESOLVES
    it against its built node (:func:`_resolve_warp_stage` / :func:`_resolve_scalar_stage`) into the
    ``Stage`` it stamps, and ``knobs`` records that resolved spelling (or nothing, when declined)."""
    if axis_role(kernel.op) is not AxisRole.CONTRACTION:
        return ""
    pinned = STAGE.narrow([""])[0]
    if not pinned:
        return ""
    try:
        Stage.parse(pinned)
    except ValueError:
        return ""
    return pinned


# ---- contraction operand-stage RESOLUTION (eligibility + sizing, once, scheduler-side) ---------- #
# A ``STAGE`` pin on a contraction is resolved HERE against the built :class:`Contraction` node —
# transport eligibility, the slab K-chunk (``bk_elems``), and the depth clamps — and the RESOLVED
# :class:`Stage` (or ``None``, gmem-direct) is stamped on the ``TileOp``. The materializer
# (``_atom._staged``) applies it verbatim, deciding nothing — the same stamp-then-apply shape as the
# reduce tier's shared-row stage (:func:`_row_stage`). ``knobs`` carries the RESOLVED spelling
# (``Stage.spell()``), and the explicit OFF value ``""`` when resolution declines — the DB row /
# feature vector describes the pipeline the kernel actually has, never the pin as requested (and
# a decided-empty is spelled, so the evidence pick can tell it from a never-offered family).


def _can_stage_warp(stage, k_axis: Axis, tile_m: int, tile_n: int, bk: int, atom_k: int, mask_m: bool, mask_n: bool, b_trans: bool) -> bool:
    """cp.async staging eligibility: a ``cp.async`` stage over a contraction with a STATIC,
    tile-divisible K axis and a canonical (non-transposed) B operand. A masked / symbolic **M**
    (output rows) is fine — the A-slab fill clamp-reads the overhanging rows in-bounds and the
    ``RegStore`` guards their store. A masked **N** (the B-slab inner dim) and a symbolic /
    non-divisible **K** stay gmem-direct (K zero-fill is a follow-up). Staging only ever *adds* a
    faster lowering, so an ineligible kernel silently falls back to gmem-direct."""
    if stage is None or stage.transport != "cp.async" or b_trans or mask_n:
        return False
    if not k_axis.extent.is_static:
        return False
    bk_elems = bk * atom_k
    if k_axis.extent.as_static() % bk_elems != 0:
        return False
    # cp.async needs a ≥4-byte contiguous chunk; the 16-bit mma operands give 2 B/elem, so the
    # inner slab dim must be even (A's BK, B's tile_n). Odd ⇒ fall back.
    return (bk_elems % 2 == 0) and (tile_n % 2 == 0)


def _can_stage_warp_tma(
    stage, k_axis: Axis, n_axis: Axis, tile_n: int, bk: int, atom_k: int, elem_bytes: int, mask_n: bool, b_trans: bool
) -> bool:
    """TMA (``cp.async.bulk.tensor``) staging eligibility: a ``tma`` stage over a contraction with a
    STATIC, tile-divisible K and a canonical B. A masked / symbolic **M** is fine — the descriptor's
    globalDim is the runtime M and TMA zero-fills the box overhang past it (no fill clamp needed). A
    masked **N** and a symbolic / non-divisible **K** stay gmem-direct. The box's inner dim (A's BK,
    B's tile_n) and the source's inner global stride (A's K, B's N) must be 16 B-aligned (the
    NONE-swizzle TMA box-copy rule)."""
    if stage is None or stage.transport != "tma" or b_trans or mask_n:
        return False
    if not (k_axis.extent.is_static and n_axis.extent.is_static):
        return False
    bk_elems = bk * atom_k
    k, n = k_axis.extent.as_static(), n_axis.extent.as_static()
    if k % bk_elems != 0:
        return False
    return all((x * elem_bytes) % 16 == 0 for x in (bk_elems, tile_n, k, n))


def _resolve_warp_stage(c: Contraction, stage: Stage, budget: int = STATIC_SMEM_CAP) -> Stage | None:
    """Resolve a pinned operand ``Stage`` against the warp (mma) contraction ``c`` — TMA > cp.async >
    gmem-direct (``None``). The resolved stage carries ``bk_elems`` (the codec-spelled ``TilePlan.bk``
    in elements), ``depth`` clamped so the ring's slots fit the smem ``budget`` (the device's dynamic
    opt-in cap when a ``Context`` reaches the schedule, else the 48 KiB static floor; dropping
    ``ring`` when the clamp leaves nothing to cycle), and ``reg_depth`` clamped to ``bk`` (nothing to
    ping-pong past the resident chunk)."""
    atom = c.atom
    a_nbytes = atom.operand_dtype("a").nbytes
    bk = c.tile.bk
    m, n = c.m, c.n
    # The TMA descriptor's box is 2-D over the operand's own array — a batched (or
    # leading-literal-indexed) operand has more gmem dims than the box and cannot encode. cp.async
    # has no descriptor (its fill closure carries the extra index dims verbatim), so it stays
    # eligible for those.
    tma_rank_ok = isinstance(c.a_operand, Load) and len(c.a_operand.index) == 2 and len(c.b_load.index) == 2
    tma_ok = tma_rank_ok and _can_stage_warp_tma(stage, c.k_axis, n.axis, n.tile, bk, atom.atom_k, a_nbytes, n.mask, c.b_trans)
    cp_ok = (not tma_ok) and _can_stage_warp(stage, c.k_axis, m.tile, n.tile, bk, atom.atom_k, m.mask, n.mask, c.b_trans)
    if not (tma_ok or cp_ok):
        return None
    bk_elems = bk * atom.atom_k
    slot_bytes = (m.tile + n.tile) * bk_elems * a_nbytes
    depth = min(stage.depth, max(1, budget // slot_bytes))
    return replace(stage, depth=depth, ring=stage.ring and depth >= 2, reg_depth=min(stage.reg_depth, bk), bk_elems=bk_elems)


def _resolve_scalar_stage(c: Contraction, stage: Stage, inputs, budget: int = STATIC_SMEM_CAP) -> Stage | None:
    """Resolve a pinned operand ``Stage`` against the scalar register-tile contraction ``c``, or
    ``None`` (gmem-direct). Staging is **opt-in behind a ``STAGE`` pin**: eligible when the transport
    is ``tma`` / ``cp.async`` and K is static (a computed-A contraction never reaches here — it keeps
    the ``Map`` form). A masked (overhanging) M / N is fine — the drain reads the slab by LOCAL tile
    coords and the overhanging store is guarded, so TMA zero-fills the box overhang and cp.async
    clamps the gmem read. The slab K-chunk ``bk_elems`` is **derived** to fit ``depth``
    ``tile_m×bk + bk×tile_n`` operand slots in the smem ``budget`` (largest power-of-two dividing K; ``inputs``
    supplies the element dtype) — not spelled by a codec, so no schema change. ``depth >= 2`` is the
    scalar gmem→smem prefetch ring — the same ``staged_kloop`` phases the warp tier runs, the atom
    contributing only the slab drain; when no K-chunk fits at the requested depth, the depth steps
    down (a smaller ring beats gmem-direct), single-buffer last. ``reg_depth`` stays 1 (the
    smem→register double-buffer is an ``ldmatrix`` transform, no scalar counterpart)."""
    if not c.k_axis.extent.is_static or stage.transport not in ("tma", "cp.async"):
        return None
    if not inputs or c.a_operand.input not in inputs:
        return None
    # TMA's 2-D descriptor box cannot encode a batched / leading-literal-indexed operand (extra
    # gmem dims); cp.async's fill closure carries them verbatim, so only TMA is gated on rank.
    if stage.transport == "tma" and (len(c.a_operand.index) != 2 or len(c.b_load.index) != 2):
        return None
    # Staging needs the CTA to BE one (tile_m × tile_n) output tile (the cooperative fill / drain
    # contract). A register-only tile (units 1×1, ``block_threads`` None) launches the scalar
    # default block over unrelated cells — no CTA-shared slab to fill; stay gmem-direct.
    if c.block_threads is None:
        return None
    K = c.k_axis.extent.as_static()
    elem_bytes = inputs[c.a_operand.input].dtype.nbytes
    # TMA hardware: every box dim must be 1..256 — the slot shapes are A (tile_m, bk) / B (bk,
    # tile_n), and bk never exceeds 128, so the gate is on the tile widths (an oversized scalar
    # register tile like tile_n=832 must decline TMA; cp.async has no box).
    if stage.transport == "tma" and max(c.m.tile, c.n.tile) > 256:
        return None
    # cuTensorMapEncodeTiled: every global stride must be 16 B-aligned — A's inner global
    # stride is K (row-major (M,K)), B's is N ((K,N)). The warp resolver's
    # ``_can_stage_warp_tma`` gates the same; without it an odd-width shape (e.g. N=5 fp32,
    # a 20 B row stride) resolver-accepts and then crashes at descriptor encode time.
    if stage.transport == "tma":
        n_ext = c.n.axis.extent
        if not n_ext.is_static or ((K * elem_bytes) % 16 or (n_ext.as_static() * elem_bytes) % 16):
            return None
    depth, bk_elems = max(1, stage.depth), 0
    while depth >= 1:
        cap = budget // (depth * max(1, c.m.tile + c.n.tile) * elem_bytes)
        bk_elems = next((v for v in (128, 64, 32, 16, 8, 4) if v <= cap and K % v == 0), 0)
        if bk_elems >= 4:
            break
        depth -= 1
    if bk_elems < 4:
        return None
    return replace(stage, depth=depth, ring=stage.ring and depth >= 2, reg_depth=1, bk_elems=bk_elems)


def warp_tile_pinned() -> bool:
    """A live warp (``a:<atom>``) ``TILE`` env pin. Exposed as a function so ``010_recognize`` never
    imports the ``Knob`` objects themselves — ``Pass.load`` scans rule modules for ``Knob`` attrs and
    OFF-fills any it finds onto every variant of the pass (bare ``TILE: ""`` stamps on every kernel)."""
    return is_warp_codec(TILE.narrow([""])[0])


def prologue_knob_bases(k2: str, stat_axis: str) -> tuple[dict, dict]:
    """The recognizer merge's two knob bases — ``(contraction-form, map-form)``. Each form carries
    the OTHER form's family keys as decided-empty stamps so every leaf row of the merged fork spells
    the same key set (the evidence pick's prefix-consistency: an absent key reads as "free"). Lives
    here (not the rule module) for the same ``Knob``-import reason as :func:`warp_tile_pinned`."""
    con = {"PLACE@cone": "fuse", _at(REDUCE, stat_axis): ""}
    map_ = {_at(TILE, k2): "", _at(STAGE, k2): "", _at(REDUCE, k2): ""}
    return con, map_


def _resolve_sync_stage(c: Contraction, budget: int = STATIC_SMEM_CAP, want_depth: int = 1) -> Stage | None:
    """The ``sync`` compute-fill :class:`Stage` for a **computed-A** warp contraction with tile plan
    ``c.tile`` — MANDATORY for this form (the gmem-direct mma leaf refuses a computed A; cp.async /
    TMA are copy transports that cannot evaluate a producer cone), so there is no gmem-direct ``""``
    sibling and a ``STAGE`` pin degrades to this resolved row. ``None`` when the slabs don't fit
    the 48 KiB smem budget: the A/B operand slabs plus one fp32 row per bridged statistic
    (``sync_stat_fill``'s decls — the same ``Contraction.stat_prologue`` seam the materializer
    fills through). ``want_depth >= 2`` (a ``STAGE`` ``d<n>/sync`` pin — pin / tune-explorable,
    never the unpinned default: the ring costs occupancy and measured slower on the reference
    shapes) rings the slabs when a canonical B has cp.async copies to overlap and the ring fits
    the budget; a transposed-B (all-sync) pipeline has nothing async to overlap and stays
    single-buffer. ``budget`` is the device's per-block dynamic-smem opt-in cap
    (``ctx.max_dynamic_smem`` — the backend declares an ``extern __shared__`` pool and sets the
    func attribute past the 48 KiB static cap), falling back to the static cap when no context
    reaches the schedule."""
    atom = c.atom
    bk_elems = c.tile.bk * atom.atom_k
    a_nbytes = atom.operand_dtype("a").nbytes
    _, _, stats = c.stat_prologue()
    # One A slab + one B slab per fold channel (the multi-channel gate/up node fills a B slab per
    # projection) + one fp32 stat row per bridged statistic.
    slot_bytes = (c.m.tile + len(c.folds) * c.n.tile) * bk_elems * a_nbytes
    stat_bytes = len(stats) * c.m.tile * 4
    if slot_bytes + stat_bytes > budget:
        return None
    depth = want_depth if want_depth >= 2 and not c.b_trans and want_depth * slot_bytes + stat_bytes <= budget else 1
    return Stage(depth=depth, transport="sync", smem=(c.a_name,), bk_elems=bk_elems)


def _computed_a_rows(kernel, place, probe: Contraction, kaxis: str, budget: int = STATIC_SMEM_CAP) -> list[dict]:
    """The knob rows for a **computed-A (fused-cone)** contraction — warp (mma) rows only, each
    riding the mandatory resolved ``sync`` stage. No scalar / per-cell ``""`` row: a per-cell
    expansion would re-run the embedded producer cone (the norm→linear statistic reduce) on every
    K step, and the serial / cooperative coverage already rides the ``Map``-form fork siblings the
    recognizer emits alongside this node. Geometry: K must be static and tile-divisible
    (``_warp_move_ok`` — the ``_staged`` driver reads a static K) and N exactly covered (the sync
    B fill has no N clamp); a masked / symbolic M is fine (the fill σ clamps, the store guards).
    Zero rows (fp32, no atoms, bad geometry, over-budget slabs) is the graceful fallback — the
    ``Map`` rows stand alone. Pins: a warp ``TILE`` pin is checked loudly (the pin contract); a
    scalar / empty ``TILE`` pin asks for a tier this form doesn't offer (zero rows — the ``Map``
    sibling's business); ``REDUCE`` rides :func:`_reduce_candidates` (split-K is excluded unpinned;
    a ``g<w>`` pin raises in :func:`_splitk_option`); a ``STAGE`` pin is ignored in favor of the
    resolved sync row."""
    # The projection rides the recognizer's ``Map`` wrapper (plus any node epilogue) — the store
    # folds it into the fragment epilogue, so it must be fragment-realizable (no gather chains).
    tail = (*probe.epilogue, *kernel.op.body) if isinstance(kernel.op, Map) else tuple(probe.epilogue)
    if not _fragment_epilogue_ok(Body(tail)):
        return []
    if TILE.raw() is not None:
        spec = TILE.narrow([""])[0]
        if not is_warp_codec(spec) or not _warp_atoms(kernel, probe):
            return []  # a scalar/empty pin, or no dtype-eligible atom (fp32) — the reduce sibling's business
        wt = TilePlan.parse(spec)
        _check_warp_static_k(kernel, wt)
        if not probe.k_axis.extent.is_static:
            raise ValueError("warp TILE pin on a fused-cone contraction: K must be static (the sync compute-fill has no K mask).")
        if replace(probe, tile=wt).n.mask:
            raise ValueError(
                f"warp TILE pin on a fused-cone contraction: the tile's N width must exactly cover the static "
                f"output columns (N={probe.axes[1].extent}, no N mask in the sync compute-fill); pick a dividing tile."
            )
        tiles = [spec]
    else:
        atoms = _warp_atoms(kernel, probe)
        tiles = [
            s
            for s in warp_tile_moves(atoms)
            if _warp_move_ok(kernel, s) and probe.k_axis.extent.is_static and not replace(probe, tile=TilePlan.parse(s)).n.mask
        ]
    want_depth = Stage.parse(_stage_spec(kernel)).depth if _stage_spec(kernel) else 1
    rows: list[dict] = []
    for spec in tiles:
        stage = _resolve_sync_stage(replace(probe, tile=TilePlan.parse(spec)), budget, want_depth)
        if stage is None:
            if TILE.raw() is not None:
                raise ValueError(f"warp TILE pin {spec!r} on a fused-cone contraction: the sync slabs exceed the {budget} B smem budget.")
            continue
        for red in _reduce_candidates(kernel, place, TilePlan.parse(spec), probe):
            rows.append({_at(TILE, kaxis): spec, _at(STAGE, kaxis): stage.spell(), _at(REDUCE, kaxis): red})
    return rows


def _wspec_workers(spec: str, stage, block_threads: int | None) -> tuple[WarpSpec | None, str]:
    """The resolved ``WSPEC`` worker split for a pipeline with the given ``stage`` and compute-band
    ``block_threads``, or ``(None, "")`` — uniform SIMT. A spec that doesn't parse, names no role,
    carries a reserved per-role param (the producer ``q`` window — inert this cut, so stamping it
    would claim a pipeline that never ran), or whose roles are illegal (a producer drives a resolved
    **TMA** stage only — ``RoleKind.legal``) degrades to uniform silently — the same pin-validity
    rule the other codecs follow. Two thread-budget gates: ``block_threads + 32·aux_warps`` must fit
    the CTA limit, and the aux band must not exceed the compute band (``32·aux ≤ block_threads`` —
    the wrapped aux decode elects the fill thread via ``threadIdx % block_threads == 0``, which must
    match exactly one aux thread)."""
    if not spec:
        return None, ""
    try:
        ws = WarpSpec.parse(spec)
    except ValueError:
        return None, ""
    if not ws.roles or any(a.params for a in ws.roles):
        return None, ""
    # ``is_legal`` reads only ``.stage`` off its arg (the producer-drives-TMA rule) — pass a probe.
    if not ws.is_legal(SimpleNamespace(stage=stage)):
        return None, ""
    aux = 32 * ws.aux_warps
    if block_threads is None or aux > block_threads or block_threads + aux > MAX_BLOCK_THREADS:
        return None, ""
    return ws, spec


def _check_warp_static_k(kernel, wt) -> None:
    """Reject a warp pin whose **static** contraction K is not a multiple of the inner mma
    K-step (``atom_k · bk``). The warp K-loop has no static-K tail handling — a partial final
    K-step reads past the operand and silently corrupts the result (max error ≫ tol, yet the
    output's *mean* error stays small so the accuracy gate passes it). A **symbolic** K is
    fine: it reaches the masked tier (ceil-div grid + boundary ``Cond`` + zero-filled partial
    slab), so guard only the static case. Raising here surfaces a clean compile error instead
    of a numerically-wrong kernel."""
    ext = reduce_loop(kernel.op).axis.extent
    if not ext.is_static:
        return
    k = ext.as_static()
    step = wt.atom.atom_k * wt.bk
    if k % step:
        raise ValueError(
            f"warp TILE pin K-step {step} (atom_k={wt.atom.atom_k}·bk={wt.bk}) does not divide the "
            f"static contraction K={k}; the warp K-loop has no static-K tail masking yet, so a "
            f"partial final step corrupts the result. Pin a K that is a multiple of {step}, or "
            f"drop the a:<atom> token to use the scalar tier."
        )


def _contraction_node(node, place, tile_plan: TilePlan) -> Contraction:
    """The high-level :class:`Contraction` structural node for a tiled ``CONTRACTION`` leaf. A
    kernel recognition already nodified (the per-cell scalar contraction — ``_nodify_contraction``
    in ``010_recognize``) only swaps the ``tile`` schedule field; a still-``Map`` form (a fused /
    flash-side contraction) is bound here at fork-emit (seam #1): the ``(a_load, b_load, acc,
    epilogue)`` operand→role facts resolve structurally (:func:`semiring_binding`) — raising
    ``LoweringError`` on an unbindable atom — plus the resolved ``tile_plan`` from the schedule
    fork, and the (m, n) output / K axes off the ``Map``. The projection ``epilogue`` is the
    binding's body verbatim — the synthesized grid-``Write`` for a bare contraction stays a
    materialize concern (it needs ``root.output``), appended there when the epilogue carries no
    ``Write``."""
    if isinstance(node, Map) and isinstance(node.source, Contraction):
        # The recognizer's ``Map(body=projection, source=Contraction)`` spelling — the node under
        # the wrapper is the schedulable contraction; the projection stays on the ``Map`` (the
        # option builders re-wrap, materialize peels it into the store tail).
        return replace(node.source, tile=tile_plan)
    if isinstance(node, Contraction):
        return replace(node, tile=tile_plan)
    grid = list(place.grid)
    a_load, b_load, acc, epilogue = semiring_binding(node, place.grid)
    return Contraction(
        axes=(grid[-2], grid[-1]),
        k_axis=reduce_loop(node).axis,
        a_operand=a_load,
        folds=((b_load, acc),),
        tile=tile_plan,
        lead_axes=tuple(grid[:-2]),
        epilogue=epilogue,
    )


def _factor_k(k_axis: Axis, w: int) -> tuple[Axis, Axis, Sigma]:
    """Factor a **static** contraction axis ``k`` into ``ksplit × kslice`` for split-K.

    ``ksplit`` (extent ``w``, name ``<k>_ks``) is the outer *partition index* — becomes the
    :class:`Reduction`'s reduce axis, parallelized across CTAs and summed in the finalize; ``kslice``
    (extent ``K/w``, the **original** name) is the per-partition chunk — stays the inner
    :class:`Contraction`'s ``k_axis``. The returned ``sigma`` maps the original ``k`` var to
    ``ksplit·(K/w) + kslice`` so the operand loads reconstruct the absolute index. Distinct names
    (``k`` vs ``<k>_ks``) are what avoid a double-reduce ``for k:[for k:]`` — every original ``k`` is
    visited once (``kslice`` folded into a partial, ``ksplit`` summed across partials)."""
    big_k = k_axis.extent.as_static()
    b = big_k // w
    ksplit = Axis(name=f"{k_axis.name}_ks", extent=Dim(w))
    kslice = replace(k_axis, extent=Dim(b))
    sigma = Sigma({k_axis.name: BinaryExpr("+", BinaryExpr("*", Var(ksplit.name), Literal(b, "int")), Var(k_axis.name))})
    return ksplit, kslice, sigma


def _splitk_option(
    tile, place, tile_spec: str, split_spec: str, name: str, knobs: dict, stage_spec: str = "", budget: int = STATIC_SMEM_CAP
) -> TileOp:
    """One scheduled **split-K** contraction ``TileOp``: the structural ``Reduction(axis=ksplit,
    source=Contraction(k_axis=kslice))``. The inner :class:`Contraction` is the **same** node a
    non-split matmul builds (:func:`_contraction_node`, so it factorizes through ``_factor`` to mma or
    scalar per the ``tile_spec`` atom) but over ``kslice`` with operands reindexed to
    ``ksplit·(K/w) + kslice``; the outer additive :class:`Reduction` carries the ``g<w>[a|k]`` GRID
    partition (:class:`ReducePlan`) that ``030_split`` consumes into the cross-CTA partial + finalize.

    The additive carrier is built exactly as ``contraction_loop`` / a plain-sum reduce does — an
    ``Accum(op="add").as_carrier()`` (identity ``0.0``, 1 component) — so ``030_split``'s finalize
    (which reads the carrier's identity + ``as_state_merge``) needs no change. The output tile
    (``tier``) rides the inner ``Contraction``; the ``Reduction`` holds only the K partition.

    An operand ``stage_spec`` is RESOLVED against the **sliced** inner node (its ``kslice`` extent +
    offset operand indices), so eligibility is judged on the pipeline the partial kernel actually
    runs; ``030_split`` threads the resolved ``Stage`` onto its partial ``TileOp``. The honest-
    stamping rule applies (the resolved spelling, or the decided-empty ``""`` on decline).

    Knob keying: ``TILE`` / ``REDUCE`` / ``STAGE`` are stamped on the **original** k-axis name (not
    ``ksplit`` / ``kslice``), keeping the kernel single-eligible-axis so golden bare-collapse + the
    prior featurizer stay invariant vs the residual/golden spelling."""
    wt = TilePlan.parse(tile_spec)
    # Same 1024-thread/CTA guard as ``_tile_option`` — the split partial launches the same
    # ``par_n · par_m`` block, so an over-limit pinned tile must not escape through the split
    # arm to an opaque late CUDA_ERROR_INVALID_VALUE.
    if not wt.is_warp and wt.block_threads > MAX_BLOCK_THREADS:
        raise ValueError(
            f"TILE parallel block {wt.units_n}×{wt.units_m}={wt.block_threads} threads exceeds the "
            f"{MAX_BLOCK_THREADS}-thread/CTA limit; shrink n/m or move work to the f register sub-tile."
        )
    inner = _contraction_node(tile.op, place, wt)
    if inner.a_computed:
        raise ValueError(
            "split-K REDUCE pin on a fused-cone (computed-A) contraction: the producer cone cannot be "
            "sliced over K (its per-row statistic prologue spans the whole row); drop the g<w> pin."
        )
    w = ReducePlan.parse(split_spec).cta
    # A warp (mma) slice must keep the inner K-step dividing K/w — the warp K-loop has no static-K
    # tail masking (same guard as ``_check_warp_static_k``, but on the post-split slice).
    if wt.is_warp:
        step = wt.atom.atom_k * wt.bk
        ks = inner.k_axis.extent.as_static() // w
        if ks % step:
            raise ValueError(
                f"split-K slice K={ks} (K/{w}) is not a multiple of the mma K-step {step} "
                f"(atom_k={wt.atom.atom_k}·bk={wt.bk}); pick a split width whose slice is divisible."
            )
    ksplit, kslice, sigma = _factor_k(inner.k_axis, w)
    inner = replace(
        inner,
        k_axis=kslice,
        a_operand=replace(inner.a_operand, index=tuple(sigma.apply(e) for e in inner.a_operand.index)),
        folds=tuple((replace(bl, index=tuple(sigma.apply(e) for e in bl.index)), acc) for bl, acc in inner.folds),
    )
    stage = None
    if stage_spec:
        st = Stage.parse(stage_spec)
        stage = _resolve_warp_stage(inner, st, budget) if wt.is_warp else _resolve_scalar_stage(inner, st, tile.inputs, budget)
    carrier = Accum(name=inner.acc, value=f"{inner.acc}__v", op=ElementwiseImpl("add"), dtype=F32).as_carrier()
    op = Reduction(carrier=carrier, axis=ksplit, role=AxisRole.CONTRACTION, source=inner, reduce=ReducePlan.parse(split_spec))
    kaxis = reduce_loop(tile.op).axis.name  # the ORIGINAL k-axis name — single-eligible-axis keying
    stamped = {**knobs, _at(TILE, kaxis): tile_spec, _at(REDUCE, kaxis): split_spec}
    stamped[_at(STAGE, kaxis)] = stage.spell() if stage is not None else ""
    return TileOp(op=op, name=name, place=place, tier=inner.tile, stage=stage, knobs=stamped)


def _warp_option(
    tile, place, spec: str, name: str, knobs: dict, stage_spec: str = "", budget: int = STATIC_SMEM_CAP, wspec_spec: str = ""
) -> TileOp:
    """One scheduled warp-tier contraction ``TileOp``: ``place`` mapped onto the grid + the warp
    form of the ``TILE`` spec resolved into the warp-atom :class:`TilePlan`, plus an optional operand
    ``STAGE`` resolved into a :class:`Stage`. The tiled :class:`Contraction` leaf is built here (``op``),
    so materialize only ``factorize``\\ s. The packed ``TILE`` codec is the sole on-dict spelling — the
    learned-prior featurizer parses it directly (one codec, not a per-knob ``WM``/``WN``/``MMA`` explosion)."""
    wt = TilePlan.parse(spec)
    _check_warp_static_k(tile, wt)
    # Build the tiled Contraction node here — it resolves the operand→role facts internally, so an
    # unbindable atom (a non-Load operand: a computed-cone / demoted matmul) raises and is rejected
    # at fork construction, like the static-K check.
    op = _contraction_node(tile.op, place, wt)
    # A computed-A node's stage is the mandatory resolved ``sync`` compute-fill (its ``smem`` /
    # ``bk_elems`` are derived, not codec-spelled, so the row's ``"d1/sync"`` re-resolves here);
    # a Load-operand node resolves the copy transports as usual.
    if op.a_computed:
        stage = _resolve_sync_stage(op, budget, Stage.parse(stage_spec).depth if stage_spec else 1)
        assert stage is not None, "computed-A row enumerated past its smem budget"  # _computed_a_rows resolved this
    else:
        stage = _resolve_warp_stage(op, Stage.parse(stage_spec), budget) if stage_spec else None
    # Re-wrap the recognizer's projecting ``Map`` around the tiled node (materialize peels it into
    # the store tail — the same ``project ∘ contract`` spelling the Reduction tiers use).
    emitted = replace(tile.op, source=op) if isinstance(tile.op, Map) and isinstance(tile.op.source, Contraction) else op
    # Warp specialization rides ORTHOGONAL to the tile/stage just resolved: an optional WSPEC row /
    # pin splits the warps into roles over this fixed pipeline (gated on the RESOLVED ``stage`` — an
    # ineligible spec leaves no pipeline for a producer to drive, so WSPEC degrades to uniform).
    workers, wspec_spec = _wspec_workers(wspec_spec, stage, op.block_threads)
    # The per-node schedule codecs key ``@<k_axis>`` (the contraction axis this node schedules), so a
    # multi-node kernel can address each node; ``WSPEC`` stays root-global (bare).
    kaxis = op.k_axis.name
    stamped = {**knobs, _at(TILE, kaxis): spec}
    # Honest stamping: the RESOLVED spelling (depth clamps, dropped ring) — the DB row / feature
    # vector must describe the pipeline the kernel actually has. A declined / absent stage stamps
    # the explicit OFF value ``""`` (decided: gmem-direct), never the raw pin — and never nothing:
    # an absent family key means "not offered", and the evidence pick's prefix-consistency reads an
    # absent key as free, letting a gmem-direct leaf inherit a STAGED row's measurement.
    stamped[_at(STAGE, kaxis)] = stage.spell() if stage is not None else ""
    if wspec_spec:
        stamped[WSPEC.name] = wspec_spec
    return TileOp(op=emitted, name=name, place=place, tier=wt, stage=stage, workers=workers, knobs=stamped)


def _tile_option(
    tile, place, spec: str, name: str, knobs: dict, reduce_spec: str = "", stage_spec: str = "", budget: int = STATIC_SMEM_CAP
) -> TileOp:
    """One scheduled scalar-tier contraction ``TileOp``: ``place`` mapped onto the grid + the ``TILE``
    spec resolved into the ``TilePlan`` (an optional cooperative / ILP ``REDUCE`` spec **nodifying** the
    contraction to a :class:`Reduction` node carrying the K partition — the per-cell tier only, a tiled
    candidate drops it; an optional operand ``STAGE`` into the :class:`Stage`), the applied specs stamped
    on ``knobs`` for the prior. ``reduce_spec`` is the ``b`` / ``r`` K partition only — the cross-CTA
    split-K ``g`` rides the separate structural :func:`_splitk_option` fork."""
    plan = TilePlan.parse(spec)
    # The scalar tile's CTA launches ``par_n · par_m`` threads (one per parallel output cell,
    # each owning a ``reg_n · reg_m`` register sub-tile). Reject a parallel tile over the
    # 1024-thread/CTA hardware limit — otherwise the launch fails late with an opaque
    # ``CUDA_ERROR_INVALID_VALUE`` instead of a clear compile-time error.
    block = plan.block_threads
    if block > MAX_BLOCK_THREADS:
        raise ValueError(
            f"TILE parallel block {plan.units_n}×{plan.units_m}={block} threads exceeds the "
            f"{MAX_BLOCK_THREADS}-thread/CTA limit; shrink n/m or move work to the f register sub-tile."
        )
    # A tiled register-tile leaf (a ``TILE`` pin) becomes a :class:`Contraction` node here, so
    # materialize only ``factorize``\\ s. An unbindable contraction (a non-``Load`` operand) keeps the
    # ``Map`` form — materialize's per-cell scalar tier lowers it. A coop / ILP ``reduce_spec``
    # **nodifies** the flat ``Map`` contraction to a :class:`Reduction` node carrying the K partition
    # (:func:`nodify_reduce`), so the plan rides the node — not a residual ``TileOp.reduce`` field —
    # and ``_factor._tile_reduce_axis`` folds it off the node.
    op = tile.op
    stage = None
    if plan.is_tiled:
        # The coop / ILP ``REDUCE`` partition rides the NON-output-tiled tier only
        # (``_coop_reduce_spec``'s contract — ``_tile_reduce_axis`` folds one cell per thread): a
        # tiled candidate contracts K serially per register cell, so the partition is DROPPED here
        # rather than stamped onto a kernel that doesn't fold it (an honest row, not a claimed one).
        reduce_spec = ""
        try:
            op = _contraction_node(tile.op, place, plan)
        except LoweringError:
            pass  # an unbindable contraction (a non-Load operand) keeps the Map form
        else:
            # Only a built Contraction node can engage operand staging — resolve the pin against it
            # (per-cell / coop-K / unbindable forms stamp None: nothing downstream would read a stage).
            if stage_spec:
                stage = _resolve_scalar_stage(op, Stage.parse(stage_spec), tile.inputs, budget)
    elif reduce_spec:
        op = nodify_reduce(tile.op, ReducePlan.parse(reduce_spec))
    # ``TILE`` / ``REDUCE`` / ``STAGE`` key ``@<k_axis>`` (the contraction axis this node schedules),
    # unifying the schedule onto the axis-named family. STAGE stamps the RESOLVED spelling, and only
    # when resolution took (see ``_warp_option`` — the same honest-stamping rule).
    kaxis = reduce_loop(tile.op).axis.name
    stamped = {**knobs, _at(TILE, kaxis): spec, _at(REDUCE, kaxis): reduce_spec}
    stamped[_at(STAGE, kaxis)] = stage.spell() if stage is not None else ""
    return TileOp(op=op, name=name, place=place, tier=plan, stage=stage, knobs=stamped)


def schedule(tile: TileOp, name: str, knobs: dict, ctx=None) -> Fork | list[TileOp] | TileOp:
    """Map a freshly-recognized (UNMAPPED) ``tile`` onto the grid and offer its scheduling forks —
    the scheduling half of ``010_recognize``, called inline once recognition has built the tile op.
    ``tile`` is an unmapped :class:`TileOp` (its ``op`` set, ``place`` carrying just the free axes).
    Returns a single scheduled ``TileOp`` (no fork) or a list of candidate ``TileOp``\\ s (the search /
    prior ranks them). ``knobs`` is the recognized kernel's knob base (empty for a fresh kernel)."""
    place = tile.place.on_grid()
    # Dispatch on the axes' role, not a kernel kind: a pointwise (FREE) kernel has no reduce
    # decision — just map the grid (the off-default stamps ``REDUCE=""``). A reduction offers its
    # ``REDUCE`` candidate(s); a contraction offers its output ``TILE``. One candidate applies
    # directly; multiple fork for the search / prior to rank.
    role = axis_role(tile.op)
    if role is AxisRole.FREE:
        return TileOp(op=tile.op, name=name, place=place)
    # A contraction picks its free-axis output tile (``TILE``); a reduction picks its reduce
    # partition (``REDUCE``). Each offers its candidate(s): one applies directly, multiple fork.
    # A contraction ALSO honors a cross-CTA split-K (``g``) / cooperative (``b``/``r``) ``REDUCE``
    # pin — orthogonal to the output tile (``reduce`` = the K partition; ``g`` is consumed by
    # ``030_split``, ``b``/``r`` by ``_factor._tile_reduce_axis`` on the non-tiled scalar tier).
    # ``TILE`` is the unified output-fragment knob: a candidate whose codec names an atom
    # (``a:<atom>`` — :func:`is_warp_codec`) builds the tensor-core warp option, otherwise the
    # scalar register-tile option (the either-ness — a kernel is one fragment or the other).
    if role is AxisRole.CONTRACTION:
        # The RESTORED enumeration: the tile × stage × reduce legal product (rows keyed
        # ``FAMILY@<k_axis>``), offered as a lazy hierarchical fork tree — greedy descent flattens
        # the rows for one prior-scoring pass; MCTS pays one level per pop. Env pins narrow each
        # family (a fully-pinned space collapses to the single materialized option, no fork). A
        # split ``g`` row routes through the structural ``Reduction ⊃ Contraction`` fork
        # (:func:`_splitk_option`, consumed by ``030_split``); a warp row through
        # :func:`_warp_option`; the rest through :func:`_tile_option`.
        rows, kaxis = _tile_rows(tile, place, ctx)
        if not rows:
            # A computed-A (fused-cone) contraction with no legal warp row (fp32, no atoms, bad
            # geometry, over-budget slabs) contributes nothing — the recognizer's ``Map``-form
            # fork siblings stand alone. Never a raising row: the guardrail contract.
            return []

        def _materialize(row: dict) -> TileOp:
            spec = row.get(_at(TILE, kaxis), "")
            stage_spec = row.get(_at(STAGE, kaxis), "")
            red = row.get(_at(REDUCE, kaxis), "")
            if red and ReducePlan.parse(red).needs_split:
                return _splitk_option(tile, place, spec, red, name, knobs, stage_spec, _smem_budget(ctx))
            if is_warp_codec(spec):
                return _warp_option(tile, place, spec, name, knobs, stage_spec, _smem_budget(ctx), row.get(WSPEC.name, ""))
            return _tile_option(tile, place, spec, name, knobs, red, stage_spec, _smem_budget(ctx))

        if len(rows) == 1:
            return _materialize(rows[0])

        def _level(key: str) -> Level:
            return Level((key,), key=lambda r: (r.get(key, ""),))

        # The worker split (``WSPEC``, bare/root-global) is the fourth level, under the pipeline it
        # splits — option-0 ``""`` is uniform SIMT.
        levels = [_level(_at(k, kaxis)) for k in (TILE, STAGE, REDUCE)] + [_level(WSPEC.name)]
        return build_fork_tree(params=rows, levels=levels, materialize=_materialize)
    # A TWISTED streaming reduce whose per-step partial is a contraction pair takes the WARP
    # (fragment-resident) tier when the mma atom is eligible, then the scalar register-vector CHAIN
    # (the FA-2 shared-score form) when the column axis fits the register budget — DETERMINISTIC
    # conservative picks, not fork siblings: the e2e contract pins these as the cold unpinned
    # schedules, and the cold AnalyticPrior cannot yet rank structurally-different flash forms
    # (a featureless serial row scores the neutral 1.0 against a featured warp/chain row — the
    # asymmetry would flip the pick per shape). Offering warp/chain/coop/serial as one prior-ranked
    # fork is the anticipated follow-up gated on the AnalyticPrior cold-start refit; the ``REDUCE``
    # pin stays the scalar escape (it asks for a reduce partition, which only the scalar tiers
    # honor).
    if not REDUCE.narrow([""])[0]:
        warp = _twisted_warp_option(tile, name, knobs)
        if warp is not None:
            return warp
        chain = _twisted_chain_option(tile, place, name, knobs)
        if chain is not None:
            return chain
        # A PLANAR ⊗-fold over a computed MAP cone — the fused producer → matmul edge — honors a
        # warp ``TILE`` pin (pin-driven, like the matmul warp tier): the demoted contraction
        # nodifies with its computed A and the sync compute-fill stage.
        demoted = _demoted_warp_option(tile, place, name, knobs)
        if demoted is not None:
            return demoted
        # (The MONOID-producer cone — the fused norm→linear edge — no longer needs a rescue here:
        # recognition nodifies it to a computed-A Contraction fork sibling, ``010_recognize``'s
        # ``bind_prologue_contraction`` merge, so it arrives at this dispatch as CONTRACTION.)
    specs = _reduce_specs(tile, place)
    return [_option(tile, place, spec, name, knobs) for spec in specs]


_CHAIN_MAX_D = 64  # register-vector budget: the chain holds the whole output row per thread


def _twisted_chain_option(tile: TileOp, place, name: str, knobs: dict) -> TileOp | None:
    """The scalar register-vector (CHAIN) schedule for a ``TWISTED`` streaming contraction pair —
    the FA-2 shared-score form: the expect contraction's output column axis leaves the grid and
    rides a per-thread register vector (a scalar ``TilePlan`` register tile on the node), so the
    score computes ONCE per streamed key and is shared across the columns (vs the per-cell tier's
    redundant recompute per column). The conservative deterministic pick when the warp tier did not
    take the tree and the column axis is small + static (``≤ _CHAIN_MAX_D``, the register budget) —
    stamped on the schedule fields only, never a knob."""
    op = tile.op
    red = op.source if isinstance(op, Map) and isinstance(op.source, Reduction) else (op if isinstance(op, Reduction) else None)
    if red is None or red.role is not AxisRole.TWISTED or red.carrier.twist.family != "exp" or len(red.partial) == 0:
        return None
    if not isinstance(red.partial[0], Contraction):
        return None
    tail_contractions = [st for st in list(red.partial)[1:] if isinstance(st, Contraction)]
    if len(tail_contractions) != 1 or not tail_contractions[0].a_computed:
        return None
    pv = tail_contractions[0]
    d_ax = pv.n_axis
    grid = list(place.grid)
    if not d_ax.extent.is_static or not grid or grid[-1].name != d_ax.name:
        return None
    d = d_ax.extent.as_static()
    if d > _CHAIN_MAX_D:
        return None
    pv2 = replace(pv, tile=TilePlan(regs=(d, 1)))  # scalar reg order (reg_n, reg_m): the column vector
    partial = tuple(pv2 if st is pv else st for st in red.partial)
    red2 = replace(red, partial=type(red.partial)(partial))
    op2 = replace(op, source=red2) if isinstance(op, Map) else red2
    # The chain is now a fork SIBLING of the warp / reduce-partition schedules, so its resolved
    # register-vector plan is stamped (keyed on the PV contraction's k axis, like every per-node
    # schedule codec) — the row identity the DB / prior separate it from the per-cell serial by.
    stamped = {**knobs, _at(TILE, pv.k_axis.name): pv2.tile.spell()}
    return TileOp(op=op2, name=name, place=Placement(free=tile.place.free, grid=tuple(grid[:-1])), knobs=stamped)


def _demoted_warp_option(tile: TileOp, place, name: str, knobs: dict) -> TileOp | None:
    """The warp (mma) candidate for a **demoted-cone contraction** — a ``PLANAR`` ⊗-fold whose
    lift multiplies a gmem ``Load`` B with a computed pure-MAP cone A (the fused producer → matmul
    edge: ``f(x, …) @ w``), or ``None`` (stay scalar). PIN-DRIVEN like the matmul warp tier: fires
    only under a warp ``TILE`` pin. Nodifies the fold to a computed-A :class:`Contraction` (the
    same ``a_operand = Body`` the flash P@V rides) and stamps the ``sync`` compute-fill
    :class:`Stage` — the producer cone materializes the A tile straight into the smem slab the
    ``ldmatrix`` drain reads (the fused edge IS the mma tier's ``sync`` transport). First cut:
    exact-cover geometry only (static M/N/K divisible by the tile / K-chunk — no masked overhang),
    and the cone may read the ``(m, k)`` axes only."""
    spec = TILE.narrow([""])[0]
    if not is_warp_codec(spec):
        return None
    op = tile.op
    red = op.source if isinstance(op, Map) and isinstance(op.source, Reduction) else (op if isinstance(op, Reduction) else None)
    if red is None or red.role is not AxisRole.PLANAR or red.source is not None or red.carrier.twist.family != "id":
        return None
    body = list(red.partial)
    accums = [st for st in body if isinstance(st, Accum)]
    if len(accums) != 1 or accums[0].op.name != "add":
        return None
    acc = accums[0]
    defs = {st.name: st for st in body if isinstance(st, Assign)}
    lift = defs.get(acc.value)
    if lift is None or lift.op.name != "multiply" or len(lift.args) != 2:
        return None
    grid = list(place.grid)
    if len(grid) < 2:
        return None
    m_ax, n_ax, k_ax = grid[-2], grid[-1], red.axis
    loads = {st.names[0]: st for st in body if isinstance(st, Load)}

    def _load_vars(nm: str) -> set | None:
        ld = loads.get(nm)
        return {v for e in ld.index for v in e.free_vars()} if ld is not None else None

    b_name = next((a for a in lift.args if (vs := _load_vars(a)) and n_ax.name in vs and k_ax.name in vs), None)
    if b_name is None:
        return None
    a_name = next(a for a in lift.args if a != b_name)
    cone = map_cone(body, a_name)
    if cone is None or not cone:
        return None
    for st in cone:
        if isinstance(st, Load) and n_ax.name in {v for e in st.index for v in e.free_vars()}:
            return None  # the cone must be (m, k)-indexed — an n-dependent producer isn't the A tile
    wt = TilePlan.parse(spec)
    atom = wt.atom
    atom_m, atom_n, atom_k = atom.shape
    q_tensor = tile.inputs.get(next(st.input for st in cone if isinstance(st, Load))) if tile.inputs else None
    if getattr(getattr(q_tensor, "dtype", None), "name", None) != atom.ab_dtype:
        return None
    exts = (m_ax.extent, n_ax.extent, k_ax.extent)
    if not all(e.is_static for e in exts):
        return None
    M, N, K = (e.as_static() for e in exts)
    bk_elems = wt.bk * atom_k
    if K % bk_elems or M % (wt.units_m * wt.reg_m * atom_m) or N % (wt.units_n * wt.reg_n * atom_n):
        return None
    epilogue = Body(tuple(op.body)) if isinstance(op, Map) else Body(())
    if not _fragment_epilogue_ok(epilogue):
        return None  # a gather epilogue is fragment-unrealizable — stay scalar
    node = Contraction(
        axes=(m_ax, n_ax),
        k_axis=k_ax,
        a_operand=Body(tuple(cone)),
        folds=((loads[b_name], acc.name),),
        tile=wt,
        lead_axes=tuple(grid[:-2]),
        epilogue=epilogue,
    )
    stage = Stage(transport="sync", smem=(node.a_name,), bk_elems=bk_elems)
    # ``PLACE@cone=fuse`` is the RESOLVED producer-cone placement this option realizes — the cone
    # compute-fills the A slab instead of round-tripping a gmem intermediate. The one live producer
    # of the cone element (the cut side — materialize the producer as its own kernel — has no
    # emitter in the rebuilt tree, so only ``fuse`` is ever stamped today).
    stamped = {**knobs, _at(TILE, k_ax.name): spec, "PLACE@cone": "fuse"}
    return TileOp(op=node, name=name, place=place, tier=wt, stage=stage, knobs=stamped)


def _twisted_warp_option(tile: TileOp, name: str, knobs: dict) -> TileOp | None:
    """The fragment-resident (tensor-core) candidate for a ``TWISTED`` streaming reduce, or ``None``
    (not eligible — the scalar options stand alone). Eligible when the tree is the streaming
    contraction pair — a head :class:`Contraction` with gmem ``Load`` operands producing the score
    and an expect :class:`Contraction` consuming a computed (register-resident) weight, under an
    exp-family carrier — and the mma atom's own demands hold (a 16-bit operand dtype; the head's
    contraction axis and the expect's output axis divisible by the atom; a static stream / query
    extent divisible by the block, since a static ragged tail has no fragment mask — the symbolic
    path masks at the fragment and guards the gmem reads). The same-per-node stamping rule as
    ``_warp_option``: the two contractions get their mma :class:`TilePlan`\\ s (one warp, the score
    block ``2·atom_n`` keys wide, the value dim folded into the expect tile), and the placement maps
    one warp per ``atom_m`` query rows — the value axis leaves the grid. An additive ``(m, kv)``
    score bias is not realizable at the fragment tier → ``None``."""
    op = tile.op
    red = op.source if isinstance(op, Map) and isinstance(op.source, Reduction) else (op if isinstance(op, Reduction) else None)
    if red is None or red.role is not AxisRole.TWISTED or red.carrier.twist.family != "exp" or len(red.partial) == 0:
        return None
    head = red.partial[0]
    if not isinstance(head, Contraction) or not isinstance(head.a_operand, Load):
        return None
    tail_contractions = [s for s in list(red.partial)[1:] if isinstance(s, Contraction)]
    if len(tail_contractions) != 1 or not tail_contractions[0].a_computed:
        return None
    pv = tail_contractions[0]
    channels = red.carrier.twist.channels
    if len(channels) != 3 or channels[1].lift is not None or channels[2].lift is None:
        return None
    q_tensor = tile.inputs.get(head.a_operand.input) if tile.inputs else None
    atom_name = {"f16": "mma_m16n8k16_f16", "bf16": "mma_m16n8k16_bf16"}.get(getattr(getattr(q_tensor, "dtype", None), "name", None))
    if atom_name is None:
        return None
    atom = ATOM_REGISTRY[atom_name]
    atom_m, atom_n, atom_k = atom.shape
    head_dim, d_v = head.k_axis.extent, pv.n_axis.extent
    if not (head_dim.is_static and head_dim.as_static() % atom_k == 0 and d_v.is_static and d_v.as_static() % atom_n == 0):
        return None
    bn = 2 * atom_n  # the streaming block: one double-atom key step
    kv_ext, m_ext = red.axis.extent, head.m_axis.extent
    if (kv_ext.is_static and kv_ext.as_static() % bn != 0) or (m_ext.is_static and m_ext.as_static() % atom_m != 0):
        return None
    m_name, kv_name = head.m_axis.name, red.axis.name
    for s in list(red.partial)[1:]:
        if isinstance(s, Load) and s.index and {m_name, kv_name} <= {v for e in s.index for v in e.free_vars()}:
            return None  # an additive (m, kv) score bias — fragment-unrealizable, stay scalar
    qk_plan = TilePlan(atom=atom, units=(1, 1), regs=(1, bn // atom_n), bk=head_dim.as_static() // atom_k)
    pv_plan = TilePlan(atom=atom, units=(1, 1), regs=(1, d_v.as_static() // atom_n), bk=1)
    partial = tuple(replace(s, tile=qk_plan) if s is head else (replace(s, tile=pv_plan) if s is pv else s) for s in red.partial)
    red2 = replace(red, partial=type(red.partial)(partial))
    op2 = replace(op, source=red2) if isinstance(op, Map) else red2
    # One warp per atom_m query rows: the query axis shrinks to its block count; the value (expect
    # output) axis folds into the fragment tile and leaves the grid.
    grid = tuple(
        Axis(name=ax.name, extent=ax.extent.ceil_div(atom_m), source_axis=ax.source_axis or ax) if ax.name == m_name else ax
        for ax in tile.place.free
        if ax.name != pv.n_axis.name
    )
    place = Placement(free=tile.place.free, grid=grid)
    return TileOp(op=op2, name=name, place=place, knobs={**knobs, _at(TILE, head.k_axis.name): qk_plan.spell()})
