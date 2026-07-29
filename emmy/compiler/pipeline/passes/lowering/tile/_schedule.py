"""Schedule a lifted kernel onto the thread grid (+ pick the reduce partition / output tile).

The scheduling **half** of the merged ``010_recognize`` tile-lowering pass — recognition
builds an UNMAPPED :class:`~emmy.compiler.ir.tile.ir.TileOp` (the structural-IR root ``op`` +
a ``place`` carrying just the free axes) and calls :func:`schedule` here in the same rewrite (no
separate ``020`` pass). Scheduling binds the placement's ``free`` axes onto the grid
(``Placement.on_grid``) and offers the per-axis
scheduling forks — the reduce-axis **partition** (:class:`~...schedule.ReducePlan`, the
``REDUCE`` codec) for a reduce axis and the output **tile** (:class:`~...schedule.TilePlan`,
the ``TILE`` codec) for a contraction — read off the axes' :class:`~...axis.AxisRole`, never a
kernel kind. This is a helper module (``_``-prefixed, not a standalone rule); the knobs it
decides are imported from ``search/space.py`` (registration happens at declaration there).

This cut picks a **whole-CTA cooperative** partition for a **static, scalar-output,
degenerate-monoid** reduce (plain ``sum`` / ``max`` / ``mean``) when the reduce axis is
wide and the output grid is small enough to leave the GPU under-occupied — one CTA per
output cell, ``coop`` threads cooperatively folding the reduce axis (the combine is
materialized in ``lowering/kernel``). Everything else (pointwise ``Map``, twisted /
full-row reductions like online-softmax & RMSNorm, contractions, symbolic axes) keeps the
**scalar serial** fold (``ReducePlan()`` — one thread per output cell).

The selection here is **conservative module constants** standing in for the eventual
``REDUCE`` knob + prior-driven choice. ``# TODO``: replace the constants with
``knob.py::_reduce_decomp`` (BR→coop, BK→serial, FK→reg, SPLITK→cta) + the online /
offline prior. The cross-CTA ``g<n>`` split (``030_split_reduce``) and the ``r<n>`` (ILP) reg
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
from emmy.compiler.ir.atom import ATOM_REGISTRY
from emmy.compiler.ir.axis import Axis, AxisRole, Window
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.schedule import Raster, Stage, WarpSpec, has_scalar_atom_alias, is_warp_codec
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Stmt, Write
from emmy.compiler.ir.tile import (
    Channel,
    Contraction,
    Map,
    Placement,
    ReducePlan,
    Reduction,
    TileOp,
    TilePlan,
    contraction_view,
    is_contraction_fold,
)
from emmy.compiler.ir.tile.ir import gmem_row_stride
from emmy.compiler.ir.tile.ops import axis_role, cone_seam, nodify_reduce, reduce_loop
from emmy.compiler.ir.tile.ops import lower as lower_op
from emmy.compiler.pipeline.fork import Fork, Level, build_fork_tree
from emmy.compiler.pipeline.passes.lowering.tile._atomize import make_cone, map_cone, semiring_binding
from emmy.compiler.pipeline.passes.lowering.tile._carrier import projection_distributes
from emmy.compiler.pipeline.pipeline import LoweringError
from emmy.compiler.pipeline.search.space import (
    F16_MMA_F32_ACC,
    MAX_BLOCK_THREADS,
    RASTER,
    REDUCE,
    STAGE,
    TILE,
    WSPEC,
    coop_reduce_moves,
    map_tile_moves,
    precision_pin,
    raster_moves,
    scalar_tile_moves,
    splitk_moves,
    stage_moves,
    twisted_warp_moves,
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


def _tma_allowed(ctx) -> bool:
    """Whether ``TMA`` (``cp.async.bulk.tensor``) stage moves may be offered for this target. TMA is
    a Hopper (sm_90) feature — Ada / Ampere and older have no TMA, and nvcc has no ``sm_89a`` target,
    so a TMA stage there fails to compile (``nvcc fatal: Unsupported gpu architecture 'sm_89a'``).
    Gate the ``d*/tma*`` moves off below sm_90, mirroring the frontend TMA-fold gate in
    :mod:`~emmy.compiler.pipeline.passes.frontend.decomposition._fold_constant` (``cc < (9, 0)``).
    ``ctx is None`` (direct unit-test drive, no target) allows it — those paths never reach nvcc."""
    return ctx is None or ctx.compute_capability >= (9, 0)


# The consumer-die compute capabilities where f32-accumulate HMMA runs at HALF the f16-accumulate
# rate (GA102/AD102/GB202 silicon — RTX 3090 / 4090 / 5090 and their workstation SKUs). On the
# datacenter parts (sm_90 H100, sm_100 B200) f32-accumulate is full rate, so the f16acc fork is
# pure search noise there.
_F16ACC_CCS = frozenset({(8, 6), (8, 9), (12, 0)})


def _f16acc_allowed(ctx) -> bool:
    """Whether the f16-accumulate atom forks (``a:mma_m16n8k16_f16_f16``) may be OFFERED. A
    precision-trading gate, off by default: the precise ``EMMY_F16_MMA_F32_ACC`` pin is
    authoritative on every target (1 offers everywhere — e.g. to measure the no-win case — 0
    never); unset, the ``EMMY_FAST_MATH`` umbrella offers it on the consumer-die targets
    (:data:`_F16ACC_CCS`) where the f32-accumulate half-rate nerf makes it profitable. ``ctx is
    None`` (direct unit-test drive, no target) stays off — enumeration must not grow under a
    bare umbrella with no known target. A ``TILE`` pin naming the atom bypasses this gate
    entirely (pins are authoritative; :func:`_warp_atoms` only checks dtype eligibility)."""
    raw = F16_MMA_F32_ACC.raw()
    if raw is not None:
        return F16_MMA_F32_ACC.parse(raw)
    if not precision_pin(F16_MMA_F32_ACC):
        return False
    return ctx is not None and ctx.compute_capability in _F16ACC_CCS


# The f16-accumulate sibling of each base atom (f16 only — mma.sync has no bf16-accumulate form).
_F16ACC_ATOMS = {"mma_m16n8k16_f16_f32": "mma_m16n8k16_f16_f16"}


def _with_f16acc(atoms: tuple[str, ...], ctx) -> tuple[str, ...]:
    """Extend the dtype-eligible ``atoms`` with their f16-accumulate siblings when
    :func:`_f16acc_allowed` — the extended tuple rides :func:`warp_tile_moves` unchanged, so the
    f16acc forks are ordinary ``TILE`` rows (identified by the ``a:<atom>`` token, priced by the
    ``MMA_acc_bits`` feature)."""
    if not atoms or not _f16acc_allowed(ctx):
        return atoms
    return atoms + tuple(_F16ACC_ATOMS[a] for a in atoms if a in _F16ACC_ATOMS)


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


def _matvec_b_kstride(kernel, carrier, place) -> int | None:
    """B's gmem stride along the reduce axis at the per-cell MATVEC tier, or ``None`` when no
    layout gate applies. A contraction demoted to PLANAR (the M=1 recognizer fallback) carries
    BOTH a vector operand — a load along the reduce axis touching no non-unit free axis (A) —
    and a matrix operand indexed by the reduce axis AND a non-unit free axis (B); only that
    two-operand shape is gated. A plain rowwise monoid reduce (softmax / rms_norm / bare mean:
    ONE input, indexed by row and reduce alike) has no vector operand and returns ``None``, as
    do disagreeing multi-B strides and ``gmem_row_stride``'s underivable cases. ``1`` means the
    reduce axis is B's fastest-varying dimension (the serving ``F.linear`` N×K layout); ``>1``
    means k-major (canonical ``B[k, n]``)."""
    nonunit = {a.name for a in place.free if not (a.extent.is_static and a.extent.as_static() == 1)}
    k_name = carrier.axis.name

    def loads(stmts):
        for s in stmts:
            if isinstance(s, Load):
                yield s
            for b in s.nested():
                yield from loads(b)

    a_seen = False
    strides = set()
    for ld in loads(lower_op(kernel.op)):
        used = set().union(*(e.free_vars() for e in ld.index)) if ld.index else set()
        if k_name not in used:
            continue
        if used & nonunit:
            strides.add(gmem_row_stride(ld, k_name, kernel.inputs))
        else:
            a_seen = True
    return strides.pop() if a_seen and len(strides) == 1 else None


def _reduce_specs(kernel, place, ctx=None) -> list[str]:
    """The candidate ``REDUCE`` codec strings for ``kernel``, applying the decision
    hierarchy. A kernel the cooperative tier can't partition (pointwise, or a twisted /
    full-row / contraction reduce) is the lone scalar fold ``[""]`` — the ``REDUCE`` pin is
    ignored there, since it only governs the cooperative reduce tier. An eligible reduce
    normally offers option-0 = the conservative heuristic pick (``_pick_coop`` — so a cold
    greedy compile keeps its historical deploy), then the full legal
    :func:`coop_reduce_moves` catalog + serial as fork siblings for the search / prior to
    rank. A deterministic deploy narrows structurally dominated wide F.linear MATVECs and
    the measured RTX-4080 DiT LayerNorm shape to their cooperative defaults; tune mode
    (``ctx.validate_pins=False``) retains the catalog. The catalog rows are
    what keep the reduce goldens (``b16``/``b32``) reachable: the heuristic alone offered a
    single spec on any free grid past ``_FREE_CAP`` (no fork), so a bare 2048-row sum tuned
    exactly one serial variant, 53× behind its pinned golden (eighth-sweep finding 3). An env
    pin (``EMMY_REDUCE``) stays authoritative over the candidates (``Knob.narrow``)."""
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
    # The layout gate (WS5, the cold-poison hardening): at the matvec tier the coop bands are only
    # coalesced on ONE B orientation — plain ``b<n>`` interleaves lanes along K (fast only when K is
    # B's fastest-varying stride, the serving ``F.linear`` layout), the transposed ``b<n>t`` sweeps
    # lanes along the output axis (fast only on k-major B, the canonical ``.t`` form). The old
    # "measurement decides" stance let a cold/tied pick land a band on the wrong operand (three
    # 10-100× incidents in one day: qk_global_cat/gate_up m256 ``.lin`` on the ``b<n>t`` band, the
    # gate_up_cat.m64.lin 16.9 ms catalog case) because ShapeKey is layout-blind — cross-orientation
    # golden/evidence rows tie. Enumeration is the single choke point every tier resolves through
    # (goldens and evidence pick among OFFERED rows), so the gate lives here; an env pin stays
    # authoritative and un-gated (exploratory pinned benches on either layout keep working).
    kstride = _matvec_b_kstride(kernel, carrier, place)
    # A row-major F.linear MATVEC (M=1 before the contraction recognizer's
    # demotion) with a wide output grid must distribute K across at least one
    # warp.  Leaving the serial sibling in the deploy fork is catastrophically
    # fragile: the generic prior can prefer one thread per output even though
    # that thread then walks the entire K axis (DiT's conditioning projections
    # measured 27--116 us serial versus 2--7 us with b32 on AD103).  This is a
    # structural dominance rule, not a model-specific schedule: K is contiguous
    # in B, there are enough output cells to fill the device, and every CTA owns
    # one output cell, so a single-warp fold is the conservative deployment.
    # An explicit REDUCE pin remains authoritative and can request any catalog
    # row for tuning/experimentation.
    deploy = ctx is None or ctx.validate_pins
    if deploy and REDUCE.raw() is None and kstride == 1 and extent >= _COOP_MIN_EXTENT and free >= _FREE_CAP:
        return ["b32"]
    # Keep the RTX-4080 DiT LayerNorm rows deterministic too. The local online
    # prior is mutable (``tune`` retrains it), and one bad checkpoint selected
    # the 480-us serial row instead of this measured 1.7-us b128 fold. Scope the
    # guard to the exact SKU/shape; other cards and row widths retain their
    # recorded golden candidates.
    if (
        deploy
        and REDUCE.raw() is None
        and ctx is not None
        and ctx.gpu_name == "NVIDIA GeForce RTX 4080"
        and kstride is None
        and extent == 1152
        and free == 256
    ):
        return ["b128"]
    if coop > 1 and kstride is not None and kstride != 1:
        coop = 1  # the heuristic option-0 is a plain band too — uncoalesced on a k-major B
    cands = [f"b{coop}" if coop > 1 else ""]  # conservative heuristic pick first (cold greedy → option-0)
    # The transposed band (and its g-split composites) is the k-major MATVEC partition. The
    # M=1 cut consumers land on THIS tier (a contraction at M=1 demotes to PLANAR — the
    # recognizer's documented fallback), so the band is offered here under the structural
    # conditions the transposed emitter assumes: no shared-row ``sync`` stage, a scalar
    # projection tail (no distributed sweep ``Loop``), a STATIC reduce extent (the ``g``
    # composite splits it), and a 32-divisible non-unit inner free axis. Softmax / rms rows
    # fail the tail/row-stage conditions and never see the band.
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
    for move in coop_reduce_moves():
        p = ReducePlan.parse(move)
        if p.coop_transposed or p.needs_split:
            if not (bt_ok and p.coop_transposed and p.coop % 32 == 0 and (not p.needs_split or k_static % p.cta == 0)):
                continue
            if kstride == 1:
                continue  # WS5: ``b<n>t`` lane-sweeps the output axis — uncoalesced when K is B's fastest stride
        elif p.coop > 1 and kstride is not None and kstride != 1:
            continue  # WS5: plain ``b<n>`` interleaves lanes along K — uncoalesced on a k-major B
        if p.coop <= extent and p.reg <= extent and move not in cands:
            cands.append(move)
    if "" not in cands:
        cands.append("")
    return list(REDUCE.narrow(cands))


def _with_reduce(op, plan: ReducePlan, stage: Stage | None = None):
    """Stamp the chosen ``ReducePlan`` onto the op's :class:`Reduction` node (bare, or wrapped under a
    projecting :class:`Map`). The reduce partition lives **on the node**, not the ``TileSchedule`` —
    read back via ``ops.reduce_plan``. ``_option`` only schedules a PLANAR / TWISTED reduce, whose op
    recognition always emits as a bare ``Reduction`` or a projecting ``Map(source=Reduction)``."""
    if isinstance(op, Reduction):
        return replace(op, reduce=plan, stage=stage)
    assert isinstance(op, Map) and isinstance(op.source, Reduction), f"reduce op must nodify to Reduction, got {type(op).__name__}"
    return replace(op, sources=(replace(op.source, reduce=plan, stage=stage),))


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
    op = _with_reduce(tile.op, plan, _row_stage(tile, place) if plan.coop > 1 else None)
    raxis = reduce_loop(tile.op).axis.name
    return TileOp(op=op, name=name, place=place, knobs={**knobs, _at(REDUCE, raxis): spec})


# The mma atoms eligible per operand dtype — the warp tier's dtype gate (16-bit operands only).
_ATOMS_BY_DTYPE = {"f16": ("mma_m16n8k16_f16_f32",), "bf16": ("mma_m16n8k16_bf16_f32",)}

# Measured deploy defaults for the four contraction shapes in
# facebook/DiT-XL-2-256's first block on an RTX 4080. These are deliberately
# SKU- and shape-exact: the generic prior remains responsible for every other
# contraction, and tune mode still sees the complete legal row set. A plain
# run uses these stable winners instead of depending on mutable machine-local
# prior state. Keys are ``(computed_a, b_trans, M, N, K)``.
_RTX4080_DIT_CONTRACTION_DEFAULTS: dict[tuple[bool, bool, int, int, int], tuple[str, str, str]] = {
    # LayerNorm -> packed QKV.
    (True, True, 256, 3456, 1152): ("a:mma_m16n8k16_f16_f32/w4x1/f2x4/k4", "d1/sync", ""),
    # Attention output projection + residual.
    (False, True, 256, 1152, 1152): ("a:mma_m16n8k16_f16_f32/w1x4/f4x2/k4", "d2/cp/ring/p2", ""),
    # LayerNorm -> feed-forward input projection.
    (True, True, 256, 4608, 1152): ("a:mma_m16n8k16_f16_f32/w1x2/f4x4/k4", "d1/sync", ""),
    # Feed-forward output projection + residual; four K slices restore enough
    # CTA parallelism for this small-M, long-K shape.
    (False, True, 256, 1152, 4608): ("a:mma_m16n8k16_f16_f32/w1x2/f4x4/k2", "d2/cp/ring/p2", "g4k"),
}

# Emit unpinned split-K candidates only when the output grid alone leaves the GPU under-occupied —
# split-K far past the occupancy need is pure combine/workspace waste (the prior's
# ``D_splitk_excess`` prices the remainder; this gate keeps the obviously-pointless rows out).
# 1024, not 512: the gemma s2048 goldens (q_proj.s2048 ``g2k`` at a 1024-CTA grid, 313 vs 402 µs
# un-split on the 5090) sit right past the old cap — a recorded golden must stay a member of the
# unpinned enumeration, and the measured win refutes "~2 waves is enough" at that boundary.
_SPLITK_MAX_CTAS = 1024


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


def _warp_atoms(kernel, probe, proj: Body | None = None) -> tuple[str, ...]:
    """The dtype-eligible tensor-core atom names for this contraction, ``()`` when the warp tier
    doesn't apply (unbindable node, a non-16-bit operand dtype, or a fragment-unrealizable gather
    epilogue). A computed-A (fused-cone) node reads its operand dtype off the cone's K-indexed
    ``Load`` — the value the sync compute-fill stores to the A slab. A cone whose leaf is **f32**
    may still ride the folds' 16-bit atom (:func:`_demoted_atoms`) — the fill demotes on the slab
    store; the plain-``Load`` form stays 16-bit-only here (copy transports move raw bytes and can
    never convert) and reaches the warp tier through :func:`_demote_mixed_a`'s cone wrap instead."""
    if probe is None or not kernel.inputs:
        return ()
    if not _fragment_epilogue_ok(Body(()) if proj is None else proj):
        return ()
    if isinstance(probe.a, Load):
        ld = probe.a
    else:
        kname = probe.k_axis.name
        cone = probe.a_body  # the inline cone's K-indexed leaf carries the dtype
        ld = next((st for st in cone if isinstance(st, Load) and kname in {v for e in st.index for v in e.free_vars()}), None)
    t = kernel.inputs.get(ld.input) if ld is not None else None
    atoms = _ATOMS_BY_DTYPE.get(getattr(getattr(t, "dtype", None), "name", None), ())
    if atoms or isinstance(probe.a, Load) or getattr(getattr(t, "dtype", None), "name", None) != "f32":
        return atoms
    return _demoted_atoms(kernel, probe)


def _demoted_atoms(kernel, con) -> tuple[str, ...]:
    """The 16-bit atoms an **f32-A** contraction may ride by demoting A on the sync compute-fill's
    slab store: the single 16-bit dtype shared by every fold's B ``Load``, or ``()``. The
    ``f32-A × 16-bit-B`` signature can only enter a traced graph through an erased dtype cast —
    torch cannot execute a mixed-dtype matmul, so the model itself cast one side (Gemma's
    ``self._norm(x.float()).type_as(x)`` rounds A back to f16 before every projection; the tracer
    maps ``to``/``type_as`` to identity pass-throughs, leaving the f32 tensor feeding the f16
    weights). B's values genuinely carry 16 bits, the accumulate stays f32, and this is a fork
    SIBLING (the scalar rows remain), so the demotion is searchable, pinnable, and costs ~2^-11
    relative noise on A — the rounding the model performed anyway in the dominant erased-downcast
    case. (The converse erased-upcast graph — ``w.float()`` on a 16-bit weight — shows B=f32 and
    never triggers this.)"""
    b_loads = [ch.b for ch in con.channels]
    if not b_loads or not all(isinstance(b, Load) for b in b_loads):
        return ()
    b_names = {getattr(getattr(kernel.inputs.get(b.input), "dtype", None), "name", None) for b in b_loads}
    if len(b_names) == 1 and b_names <= set(_ATOMS_BY_DTYPE):
        return _ATOMS_BY_DTYPE[next(iter(b_names))]
    return ()


def _demote_mixed_a(kernel, con):
    """A mixed-dtype contraction — plain f32-A ``Load`` against 16-bit folds
    (:func:`_demoted_atoms`) — re-expressed as a computed-A cone (a one-``Load`` ``Map`` stored
    inline on the ``a`` edge), so it rides the mandatory ``sync`` compute-fill whose slab
    store demotes the value to the atom dtype. The copy transports (gmem-direct ldmatrix / cp.async
    / TMA) move raw bytes and cannot convert, which is why the demotion routes through the cone form
    instead of the plain warp tier. Anything else (already-computed A, non-Load A, 16-bit A,
    non-16-bit folds) returns unchanged."""
    if con is None or not isinstance(con.a, Load):
        return con
    a_t = kernel.inputs.get(con.a.input)
    if getattr(getattr(a_t, "dtype", None), "name", None) != "f32" or not _demoted_atoms(kernel, con):
        return con
    return replace(con, a=make_cone([con.a], con.k_axis.name))


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


def _stage_candidates(kernel, probe, plan: TilePlan, budget: int = STATIC_SMEM_CAP, tma_ok: bool = True) -> list[str]:
    """The RESOLVED operand-stage spellings for one tile candidate — gmem-direct ``""`` first, then
    every grid move that resolves against the node with this ``plan`` (:func:`_resolve_warp_stage` /
    :func:`_resolve_scalar_stage`); the row carries the resolved spelling so the leaf identity, the
    stamped knobs, and the kernel agree. A pinned ``STAGE`` is authoritative: the resolved pin
    alone, or gmem-direct when it declines (the standard pin-validity degrade). ``tma_ok`` is the
    target's TMA availability (:func:`_tma_allowed`): below sm_90 a ``d*/tma*`` move / pin declines
    here (stays gmem-direct) rather than being offered and failing to compile."""
    if probe is None or not plan.is_tiled:
        return [""]  # per-cell / unbindable — no operand slab to stage
    node = replace(probe, tile=plan)

    def resolve(spec: str) -> str | None:
        st = Stage.parse(spec)
        if st.transport == "tma" and not tma_ok:
            return None  # TMA is Hopper+ (sm_90); nvcc has no sm_89a — decline below it
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


def _reduce_candidates(kernel, place, plan: TilePlan, probe: Contraction | None = None, channels: int = 1) -> list[str]:
    """The ``REDUCE`` codec candidates for one tile candidate — serial ``""`` first (option-0),
    then the legal coop / ILP moves (per-cell tier only — the non-output-tiled contract) and the
    divisor- and occupancy-guarded split-K moves (both finalizes, both tiers). An **atomic**
    split (``g<w>a``) is offered only on a single-fold node whose FULL projection tail (the node
    epilogue + a computed-A ``Map`` wrapper's body, exactly what ``_splitk_option`` folds into
    the partial) distributes over the add — a non-distributive projection or a multi-channel
    ⊗-combine would raise at ``030_split_reduce`` and waste a search slot; the deferred ``g<w>k``
    finalize stays legal for any epilogue. A pinned ``REDUCE`` is authoritative and keeps the pin
    contract: a ``g`` split rides every tile (an invalid warp slice / atomic-on-non-distributive
    raises in :func:`_splitk_option` / ``030_split_reduce``, as a pin should), a ``b``/``r``
    partition applies to the per-cell tier only (a tiled candidate has no row under it)."""
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
            if not (p.coop <= k and p.reg <= k):
                continue
            if p.coop_transposed:
                # The ``b<n>t`` lane swap needs the structure its emitter assumes: a plain
                # CONTRACTION per-cell tier (``probe`` built — the fused monoid rows ride
                # shared-row staging the transposed layout can't), a static innermost free
                # axis divisible by the 32-lane sweep, and a 32-multiple coop. A composite
                # ``g<w>k/b<n>t`` (the deployable long-K form) additionally needs the split
                # divisibility the plain split moves check. Layout is a GATE too (WS5): the
                # band lane-sweeps the output axis, so it is only coalesced on k-major B —
                # offering it on an N-major (``F.linear``) operand let cold/tied picks land
                # 10-100× rows (the layout-blind ShapeKey cold-poison incidents).
                # The innermost NON-UNIT free axis — the m1 recognizer's synthesized
                # ``_um`` unit axis can sit innermost (extent 1), and it is not the axis
                # the transposed emitter sweeps.
                inner = next((a for a in reversed(place.free) if not (a.extent.is_static and a.extent.as_static() == 1)), None)
                if not (
                    probe is not None
                    and not probe.b_trans
                    and p.coop % 32 == 0
                    and inner is not None
                    and inner.extent.is_static
                    and inner.extent.as_static() % 32 == 0
                    and (not p.needs_split or k % p.cta == 0)
                ):
                    continue
            elif p.coop > 1 and probe is not None and not probe.b_trans:
                continue  # WS5: plain ``b<n>`` interleaves lanes along K — uncoalesced on canonical ``B[k, n]``
            out.append(move)
    free = prod(_hint_extent(a) for a in place.free) if place.free else 1
    # A computed-A (fused-cone) contraction splits over K via the REDUNDANT-STATISTIC form: the
    # k-invariant stat prologue spans the whole row and stays full-row in every partition (each
    # recomputes it — cheap exactly where split-K pays, the small-free decode shapes this offer
    # is gated to), while only the per-cell cone is σ-reindexed (``_splitk_option``). Multi-channel
    # nodes (the gate/up fused edge) split too: each channel's fold is an independent additive
    # state, so the deferred finalize folds the N-component carrier exactly like flash's (m,l,O)
    # and applies the ⊗-combine projection once after the cross-partition sums.
    if k is not None and free // _tile_area(plan) <= _SPLITK_MAX_CTAS:
        step = plan.atom.atom_k * plan.bk if plan.is_warp else 1
        # The atomic gate judges the FULL projection the split partial will carry: a computed-A
        # node's ``Map``-wrapper body is folded into the inner epilogue at ``_splitk_option``, so
        # it must distribute too. Multi-fold (gate/up) nodes never offer ``a`` — the per-channel
        # ⊗-combine needs the deferred finalize kernel.
        tail = tuple(kernel.op.body) if isinstance(kernel.op, Map) else ()
        atomic_ok = probe is not None and channels == 1 and (len(tail) == 0 or projection_distributes(tail, (probe.acc,)))
        for move in splitk_moves(warp=plan.is_warp):
            sp = ReducePlan.parse(move)
            if sp.finalize == "atomic" and not atomic_ok:
                continue  # non-distributive fused projection — 030_split_reduce would raise; don't offer
            if k % sp.cta == 0 and (k // sp.cta) % step == 0:
                out.append(move)
    return out


def _raster_candidates(place) -> list[str]:
    """The ``RASTER`` codec candidates for one contraction row — every matmul-tier output is a
    2-D block-tile grid, so eligibility here is the grid's STATIC-ness (the reduce/pointwise
    tiers never build these rows; the flash fork spells its own key set and stays flat). A
    symbolic-axis (masked-tile) grid renders through the dynamic decode path, which does not
    carry the swizzle yet — offering ``gm8`` there would stamp a launch order the kernel doesn't
    realize (the silent-degrade family), so a symbolic grid decides the flat ``""`` only; an
    explicit pin on one degrades likewise and the replay integrity gate reports it. The env pin
    narrows as usual on static grids (``EMMY_RASTER=gm8`` / ``gn4`` — pins may spell values
    outside :func:`raster_moves`)."""
    if any(not ax.extent.is_static for ax in place.free):
        return [""]
    return list(RASTER.narrow(raster_moves()))


def _wspec_candidates(plan: TilePlan, stage_spelling: str, red: str) -> list[str]:
    """The ``WSPEC`` candidates for one enumerated (tile, stage, reduce) row — uniform ``""``
    alone unless the row can drive a producer band: a warp tile over a resolved **TMA** stage
    (:func:`_wspec_workers`'s legality, pre-filtered here so the fork doesn't spawn rows that
    materialize identically) and no cross-CTA split (the split partial re-resolves its own
    pipeline; wiring WSPEC through ``030_split_reduce`` is a follow-up). The env pin narrows as usual —
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
        probe, proj = _contraction_node(kernel.op, place, TilePlan())
    except LoweringError:
        probe, proj = None, Body(())
    if probe is not None and probe.a_computed:
        return _computed_a_rows(kernel, place, probe, kaxis, _smem_budget(ctx), ctx, proj=proj), kaxis
    tiles = scalar_tile_moves() if probe is not None else [""]
    warp_offered = False
    demoted_rows: list[dict] = []
    if probe is not None:
        atoms = _with_f16acc(_warp_atoms(kernel, probe, proj=proj), ctx)
        if atoms:
            warp_moves = [s for s in warp_tile_moves(atoms) if _warp_move_ok(kernel, s)]
            tiles += warp_moves
            warp_offered = bool(warp_moves)
        else:
            # Mixed-dtype (f32-A × 16-bit-B) contraction: the warp tier rides the demoting
            # sync compute-fill through the cone form — pre-assembled rows (TILE+STAGE+REDUCE
            # resolved, warp pins handled inside); the scalar rows below stay fork siblings.
            demoted = _demote_mixed_a(kernel, probe)
            if demoted is not probe:
                demoted_rows = _computed_a_rows(kernel, place, demoted, kaxis, _smem_budget(ctx), ctx, proj)
                warp_offered = bool(demoted_rows)
    # A pinned ``a:scalar`` / ``a:none`` is the explicit scalar-tier spelling — canonicalize it to the
    # bare scalar codec (``""`` / ``n../f..``) so the pin-only alias never rides a stored knob row (it
    # would otherwise leak into the prior/DB key and the golden YAML). Non-alias specs pass through
    # untouched — no blanket re-spell.
    tiles = [TilePlan.parse(t).spell() if has_scalar_atom_alias(t) else t for t in TILE.narrow(tiles)]
    if demoted_rows:
        # A warp TILE pin is already honored (or rejected) by the demoted rows — the plain loop
        # must not also emit it over the copy transports, which cannot convert the f32 A.
        tiles = [t for t in tiles if not is_warp_codec(t)]
    # Warp-eligibility is a structural fact about the KERNEL: when the enumeration offers any
    # tensor-core row, EVERY row (scalar and warp alike) carries ``S_warp_eligible`` so the
    # priors can price "a scalar tile where tensor cores were on offer"
    # (``features.D_scalar_on_warp_eligible``). ``S_``-prefixed — rides ``knob_features``'
    # structural pass-through; not a schedule family, so tile identity / prefix-consistency
    # and ``tile_signature`` are untouched.
    stamp: dict = {"S_warp_eligible": 1.0} if warp_offered else {}
    rows: list[dict] = []
    for spec in tiles:
        plan = TilePlan.parse(spec)
        if plan.is_warp and TILE.raw() is not None:
            _check_warp_static_k(kernel, plan)  # a PIN with an indivisible K-step raises (the pin contract)
            if probe is not None and not _fragment_epilogue_ok(proj):
                raise ValueError(
                    "warp TILE pin: the projection epilogue gathers through another epilogue "
                    "load (a data-dependent index) — the fragment epilogue cannot thread it; "
                    "drop the a:<atom> token to use the scalar tier."
                )
        for stage in _stage_candidates(kernel, probe, plan, _smem_budget(ctx), _tma_allowed(ctx)):
            for red in _reduce_candidates(kernel, place, plan, probe, len(probe.channels) if probe else 1):
                # A staged split row is legal: ``_splitk_option`` re-resolves the stage against the
                # SLICED inner node (the warp slice divisibility already held in
                # ``_reduce_candidates``) and ``030_split_reduce`` threads it onto the partial kernel.
                # Every family key is explicit — ``""`` is a DECIDED empty (per-cell / serial /
                # gmem-direct), distinguishable from an absent (never-offered) family. The
                # evidence pick's prefix-consistency depends on it: an absent key reads as
                # "free" and would let a gmem-direct leaf inherit a staged row's measurement.
                for wspec in _wspec_candidates(plan, stage, red):
                    for raster in _raster_candidates(place):
                        rows.append(
                            {
                                _at(TILE, kaxis): spec,
                                _at(STAGE, kaxis): stage,
                                _at(REDUCE, kaxis): red,
                                WSPEC.name: wspec,
                                RASTER.name: raster,
                                **stamp,
                            }
                        )
    rows += [{**r, **stamp} for r in demoted_rows]
    return rows, kaxis


def _is_splitk_spec(spec: str) -> bool:
    """True when a ``REDUCE`` spelling is a cross-CTA GRID split (``g<w>[a|k]``) — the partition
    ``_splitk_option`` realizes into a partial/finalize pair. An unparseable or non-split spelling
    (``""`` / ``b`` / ``r``) is not a split."""
    if not spec:
        return False
    try:
        return ReducePlan.parse(spec).needs_split
    except ValueError:
        return False


def _splitk_pin() -> str:
    """The pinned ``g<w>[a|k]`` split-K spec (or ``""``) — the cross-CTA K partition a
    ``CONTRACTION`` honors through the structural ``Reduction ⊃ Contraction`` fork
    (:func:`_splitk_option`), consumed by ``030_split_reduce``. Reads the ``REDUCE`` pin and returns it
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
    tile-divisible K axis. A masked / symbolic **M** (output rows) is fine — the A-slab fill
    clamp-reads the overhanging rows in-bounds and the ``RegStore`` guards their store. A masked
    **N** and a symbolic / non-divisible **K** stay gmem-direct (K zero-fill is a follow-up).
    A transposed B (the serving ``F.linear`` layout, K gmem-contiguous) stages into an N-MAJOR
    slab (``tile_n × bk``) whose inner dim maps stride-1 to gmem K exactly like A's — the fill's
    chunk contiguity and row-base alignment hold automatically (B's row stride K is a multiple
    of ``bk_elems``, which the chunk width divides), so only the K-chunk evenness gates it.
    Staging only ever *adds* a faster lowering, so an ineligible kernel silently falls back to
    gmem-direct."""
    if stage is None or stage.transport != "cp.async" or mask_n:
        return False
    if not k_axis.extent.is_static:
        return False
    bk_elems = bk * atom_k
    if k_axis.extent.as_static() % bk_elems != 0:
        return False
    # cp.async needs a ≥4-byte contiguous chunk; the 16-bit mma operands give 2 B/elem, so the
    # inner slab dim must be even (A's BK, and B's tile_n — or BK again on a transposed B, whose
    # N-major slab keeps K inner). Odd ⇒ fall back.
    return (bk_elems % 2 == 0) and (b_trans or tile_n % 2 == 0)


def _tma_operand_rank_ok(index: tuple, tile_name: str, k_name: str) -> bool:
    """Whether TMA's box can encode this operand's gmem index. The box's data plane is the
    TRAILING 2 dims (A ``(m, k)`` / B ``(k, n)``); any extra LEADING dims ride as extent-1 box
    dims whose origin coordinate is the operand's own index expr, evaluated once per fill (the
    flash K/V ``(B, H, S, D)`` convention) — so those exprs must not move with the tile or the
    K loop, or the rank-2 plane the box copies would be the wrong one. Rank caps at 4 so the
    swizzle-split box (+1 dim) stays within TMA's 5-dim hardware limit."""
    if not 2 <= len(index) <= 4:
        return False
    return all(not ({tile_name, k_name} & e.free_vars()) for e in index[:-2])


def _can_stage_warp_tma(
    stage, k_axis: Axis, n_axis: Axis, tile_n: int, bk: int, atom_k: int, elem_bytes: int, mask_n: bool, b_trans: bool
) -> bool:
    """TMA (``cp.async.bulk.tensor``) staging eligibility: a ``tma`` stage over a contraction with a
    STATIC, tile-divisible K. A masked / symbolic **M** is fine — the descriptor's globalDim is the
    runtime M and TMA zero-fills the box overhang past it (no fill clamp needed). A masked **N**
    and a symbolic / non-divisible **K** stay gmem-direct. The box's inner dim and the source's
    inner global stride must be 16 B-aligned (the NONE-swizzle TMA box-copy rule): canonical B
    boxes ``(bk, tile_n)`` over strides (A's K, B's N); a transposed B (the serving ``F.linear``
    layout) boxes N-major ``(tile_n, bk)`` — both operands' inner dim is then the K chunk and both
    inner strides are K, so N drops out of the alignment gate."""
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


def _resolve_warp_stage(c: Contraction, stage: Stage, budget: int = STATIC_SMEM_CAP) -> Stage | None:
    """Resolve a pinned operand ``Stage`` against the warp (mma) contraction ``c`` — TMA > cp.async >
    gmem-direct (``None``). The resolved stage carries ``bk_elems`` (the codec-spelled ``TilePlan.bk``
    in elements), ``depth`` clamped so the ring's slots fit the smem ``budget`` (the device's dynamic
    opt-in cap when a ``Context`` reaches the schedule, else the 48 KiB static floor; dropping
    ``ring`` when the clamp leaves nothing to cycle), and ``reg_depth`` clamped to ``bk`` (nothing to
    ping-pong past the resident chunk). A tile whose single depth-1 slot already exceeds ``budget``
    DECLINES (``None``, gmem-direct) — unlike the scalar resolver it cannot shrink the slab
    (``bk_elems`` is codec-spelled here, not derived), and a resolved-but-unfittable stage would
    sail through the fork only to be rejected at materialize (the issue-#327 unlowered-``TileOp``
    bench fails)."""
    if stage.alt:
        return None  # the alternating single-slab pipeline is the warp-flash stream's (K/V phase split + staged Q)

    atom = c.atom
    a_nbytes = atom.operand_dtype("a").nbytes
    bk = c.tile.bk
    m, n = c.m, c.n
    # The TMA box's data plane is the operand's TRAILING 2 gmem dims; extra leading (batch)
    # dims ride as extent-1 box dims with the operand's own origin exprs — eligible when those
    # exprs don't move with the tile or the K loop (:func:`_tma_operand_rank_ok`; the model's
    # ``[1, seq, K]`` unit-batch views were the motivating decline). cp.async has no descriptor
    # (its fill closure carries the extra index dims verbatim), so it never gated on rank.
    tma_rank_ok = (
        isinstance(c.a, Load)
        and isinstance(c.b, Load)  # a descriptor needs a gmem address on BOTH edges
        and _tma_operand_rank_ok(c.a.index, m.axis.name, c.k_axis.name)
        and _tma_operand_rank_ok(c.b.index, n.axis.name, c.k_axis.name)
    )
    # TMA hardware: every box dim must be 1..256 — the slot shapes are A (tile_m, bk) / B (bk,
    # tile_n), so an oversized warp register tile (e.g. tile_m = 512 at w4/f8) must decline TMA
    # (the scalar resolver gates the same; cp.async has no box).
    tma_box_ok = max(m.tile, n.tile, bk * atom.atom_k) <= 256
    tma_ok = (
        tma_rank_ok and tma_box_ok and _can_stage_warp_tma(stage, c.k_axis, n.axis, n.tile, bk, atom.atom_k, a_nbytes, n.mask, c.b_trans)
    )
    cp_ok = (not tma_ok) and _can_stage_warp(stage, c.k_axis, m.tile, n.tile, bk, atom.atom_k, m.mask, n.mask, c.b_trans)
    if not (tma_ok or cp_ok):
        return None
    bk_elems = bk * atom.atom_k
    slot_bytes = (m.tile + n.tile) * bk_elems * a_nbytes
    if slot_bytes > budget:
        return None
    depth = min(stage.depth, budget // slot_bytes)
    return replace(stage, depth=depth, ring=stage.ring and depth >= 2, reg_depth=min(stage.reg_depth, bk), bk_elems=bk_elems)


def _resolve_scalar_stage(c: Contraction, stage: Stage, inputs, budget: int = STATIC_SMEM_CAP) -> Stage | None:
    """Resolve a pinned operand ``Stage`` against the scalar register-tile contraction ``c``, or
    ``None`` (gmem-direct). Staging is **opt-in behind a ``STAGE`` pin**: eligible when the transport
    is ``tma`` / ``cp.async`` and K is static (a computed-A contraction never reaches here — it keeps
    the ``Map`` form). A masked (overhanging) **M** is fine — the drain reads the slab by LOCAL tile
    coords and the overhanging store is guarded. A masked **N** or a transposed **B** stays gmem-direct:
    the masked-N B-slab fill would clamp a chunk-start column into a row-crossing gmem address and hang
    the kernel on the misaligned cp.async / TMA copy (the warp tier refuses the same), and the scalar
    drain has no transposed-slab variant (the warp tier DOES stage a transposed B, via its N-major
    slab). The slab K-chunk ``bk_elems`` is **derived** to fit ``depth``
    ``tile_m×bk + bk×tile_n`` operand slots in the smem ``budget`` (largest power-of-two dividing K; ``inputs``
    supplies the element dtype) — not spelled by a codec, so no schema change. ``depth >= 2`` is the
    scalar gmem→smem prefetch ring — the same ``staged_kloop`` phases the warp tier runs, the atom
    contributing only the slab drain; when no K-chunk fits at the requested depth, the depth steps
    down (a smaller ring beats gmem-direct), single-buffer last. ``reg_depth`` stays 1 (the
    smem→register double-buffer is an ``ldmatrix`` transform, no scalar counterpart)."""
    if stage.alt:
        return None  # the alternating single-slab pipeline is the warp-flash stream's (K/V phase split + staged Q)

    if not c.k_axis.extent.is_static or stage.transport not in ("tma", "cp.async"):
        return None
    # A masked-N (overhanging inner dim) B-slab fill would clamp a chunk-start column into a
    # row-crossing gmem address and hang on the misaligned 16 B copy — the warp tier refuses the
    # same (:func:`_can_stage_warp` / :func:`_can_stage_warp_tma`). A transposed B stays
    # gmem-direct on THIS tier only: the warp tier stages it into an N-major slab, but the scalar
    # drain reads the slab by ``(k, n)`` coords and has no transposed variant (pin-only tier, no
    # serving fork rides it).
    if c.n.mask or c.b_trans:
        return None
    # Staging transports a gmem OPERAND: both edges must be materialized. A computed B (a fused
    # per-column prologue) has no gmem address to copy from, so the tier declines and the node stays
    # gmem-direct — the schedule stating the A/B asymmetry the structural type does not.
    if not inputs or not isinstance(c.a, Load) or not isinstance(c.b, Load) or c.a.input not in inputs:
        return None
    # TMA's box encodes extra leading (batch) dims as extent-1 dims when they are tile/K-invariant
    # (:func:`_tma_operand_rank_ok`); cp.async's fill closure carries them verbatim, no rank gate.
    if stage.transport == "tma" and not (
        _tma_operand_rank_ok(c.a.index, c.m.axis.name, c.k_axis.name) and _tma_operand_rank_ok(c.b.index, c.n.axis.name, c.k_axis.name)
    ):
        return None
    # Staging needs the CTA to BE one (tile_m × tile_n) output tile (the cooperative fill / drain
    # contract). A register-only tile (units 1×1, ``block_threads`` None) launches the scalar
    # default block over unrelated cells — no CTA-shared slab to fill; stay gmem-direct.
    if c.block_threads is None:
        return None
    K = c.k_axis.extent.as_static()
    elem_bytes = inputs[c.a.input].dtype.nbytes
    # TMA hardware: every box dim must be 1..256 — the slot shapes are A (tile_m, bk) / B (bk,
    # tile_n), and bk never exceeds 128, so the gate is on the tile widths (an oversized scalar
    # register tile like tile_n=832 must decline TMA; cp.async has no box).
    if stage.transport == "tma" and max(c.m.tile, c.n.tile) > 256:
        return None
    # Every staged transport needs 16 B-aligned inner global strides — A's is K (row-major
    # (M,K)), B's is N ((K,N)). TMA: cuTensorMapEncodeTiled's global-stride rule (crashes at
    # descriptor encode otherwise — the N=5 fp32 / 20 B case). cp.async: the fill's vectorized
    # copies inherit the row base alignment, so a 12 B-stride operand (N=3 fp32) issues
    # ``cp.async`` at misaligned addresses and faults at RUNTIME (``CUDA_ERROR_MISALIGNED_ADDRESS``
    # + watchdog hang — the ninth-sweep e2e regression on a (8,3) fp32 B; the Gemma
    # ``k_linear_reduce`` bench_fail cluster shares the signature and likely the class).
    # Gate both transports: an odd-stride shape stays gmem-direct.
    n_ext = c.n.axis.extent
    if not n_ext.is_static or ((K * elem_bytes) % 16 or (n_ext.as_static() * elem_bytes) % 16):
        return None
    # Per-operand slot bytes: A's slab is (tile_m × bk) at A's element size, B's (bk × tile_n) at
    # B's — the operands may differ (fp32 split partials × fp16 weights), so sizing both with A's
    # element over-books the budget on the mixed shape.
    b_bytes = inputs[c.b.input].dtype.nbytes if c.b.input in inputs else elem_bytes
    depth, bk_elems = max(1, stage.depth), 0
    while depth >= 1:
        cap = budget // (depth * max(1, c.m.tile * elem_bytes + c.n.tile * b_bytes))
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
    con = {_at(REDUCE, stat_axis): ""}
    map_ = {_at(TILE, k2): "", _at(STAGE, k2): "", _at(REDUCE, k2): ""}
    return con, map_


def _resolve_sync_stage(c: Contraction, budget: int = STATIC_SMEM_CAP, want_depth: int = 1) -> Stage | None:
    """The ``sync`` compute-fill :class:`Stage` for a **computed-A** warp contraction with tile plan
    ``c.tile`` — MANDATORY for this form (the gmem-direct mma leaf refuses a computed A; cp.async /
    TMA are copy transports that cannot evaluate a producer cone), so there is no gmem-direct ``""``
    sibling and a ``STAGE`` pin degrades to this resolved row. ``None`` when the slabs don't fit
    the 48 KiB smem budget: the A/B operand slabs plus one fp32 row per bridged statistic
    (``sync_stat_fill``'s decls — the same ``ops.cone_seam`` node boundary the materializer
    fills through). ``want_depth >= 2`` is the **asymmetric B-only prefetch ring**: only the
    B cp.async slabs ring (their copies for chunk ``i+d-1`` fly under chunk ``i``'s
    compute fill AND drain), while the compute-filled A slab and the stat rows stay
    single-buffer — ringing a compute fill buys no overlap (it runs on the drain's own
    threads). Measured on the gemma gate_up fused edge at M=512 (5090) the B-only ring loses
    (897 vs 665 µs at d2) — the extra B slot alone crosses the smem occupancy quantization
    (3 → 2 CTAs/SM), the same cliff that killed the historical full-slab ring — but at decode
    M (tile_m ≤ 32) the A slab + stat rows are tiny and the tradeoff inverts, so
    :func:`_computed_a_rows` enumerates ``d1`` and ``d2`` as fork siblings (measured per
    shape) and a ``STAGE`` pin's depth stays authoritative. A transposed B rides the same
    async B fills through its N-major slab (``_sync_operands`` — its own gmem orientation, K
    stride-1), so the ring is enumerable on the serving ``F.linear`` fused edges too. ``budget`` is the
    device's per-block dynamic-smem opt-in cap (``ctx.max_dynamic_smem`` — the backend declares
    an ``extern __shared__`` pool and sets the func attribute past the 48 KiB static cap),
    falling back to the static cap when no context reaches the schedule."""
    atom = c.atom
    bk_elems = c.tile.bk * atom.atom_k
    a_nbytes = atom.operand_dtype("a").nbytes
    _, _, stats = cone_seam(c.a) if c.a_computed else ((), (), ())
    # One A slab + one B slab per fold channel (the multi-channel gate/up node fills a B slab per
    # projection) + one fp32 stat row per bridged statistic. Only the B slabs multiply by the
    # ring depth (the asymmetric ring above).
    a_bytes = c.m.tile * bk_elems * a_nbytes
    b_bytes = len(c.channels) * c.n.tile * bk_elems * a_nbytes
    stat_bytes = len(stats) * c.m.tile * 4
    if a_bytes + b_bytes + stat_bytes > budget:
        return None
    depth = want_depth if want_depth >= 2 and a_bytes + stat_bytes + want_depth * b_bytes <= budget else 1
    return Stage(depth=depth, transport="sync", smem=(c.a_name,), bk_elems=bk_elems)


def _computed_a_rows(
    kernel,
    place,
    probe: Contraction,
    kaxis: str,
    budget: int = STATIC_SMEM_CAP,
    ctx=None,
    proj=(),
) -> list[dict]:
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
    tail = tuple(kernel.op.body) if isinstance(kernel.op, Map) else tuple(proj)
    if not _fragment_epilogue_ok(Body(tail)):
        return []
    if TILE.raw() is not None:
        spec = TILE.narrow([""])[0]
        if not is_warp_codec(spec) or not _warp_atoms(kernel, probe, Body(tail)):
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
        atoms = _with_f16acc(_warp_atoms(kernel, probe, Body(tail)), ctx)
        tiles = [
            s
            for s in warp_tile_moves(atoms)
            if _warp_move_ok(kernel, s) and probe.k_axis.extent.is_static and not replace(probe, tile=TilePlan.parse(s)).n.mask
        ]
    # Depths: a STAGE pin is authoritative; unpinned rows enumerate the d1 compute-fill AND the
    # asymmetric B-only prefetch ring at d2 as fork siblings. The ring was measured a LOSS on the
    # M=512 gate⊗up edge (the extra B slot crosses the smem occupancy quantization — the
    # :func:`_resolve_sync_stage` note), but the tradeoff INVERTS at decode M (tile_m ≤ 32: the
    # compute-filled A slab and stat rows are tiny, so the d2 slot rarely moves occupancy while the
    # B prefetch hides the dominant weight stream) — so the depth is measured per shape, not
    # hardwired. A d2 that clamps back to d1 (budget) spells identically and dedupes below.
    depths = [Stage.parse(_stage_spec(kernel)).depth] if _stage_spec(kernel) else [1, 2]
    rows: list[dict] = []
    # The launch-order codec rides these rows exactly like the plain contraction's
    # (:func:`_raster_candidates` — a computed-A kernel's output is the same static 2-D block-tile
    # grid, and the grouped decode is kernel-scoped launch metadata the transport never sees; a
    # symbolic-M fused edge decides the flat ``""`` through the same gate). The fused edge is the
    # codec's best customer: its B stripes re-stream per M-tile row (64.6% DRAM on the gemma
    # gate_up shape), which is precisely the L2 reuse a grouped order buys.
    for spec in tiles:
        spellings: list[str] = []
        for want_depth in depths:
            stage = _resolve_sync_stage(replace(probe, tile=TilePlan.parse(spec)), budget, want_depth)
            if stage is None:
                if TILE.raw() is not None:
                    raise ValueError(
                        f"warp TILE pin {spec!r} on a fused-cone contraction: the sync slabs exceed the {budget} B smem budget."
                    )
                continue
            if stage.spell() in spellings:
                continue  # d2 clamped to d1 — same resolved pipeline, one row
            spellings.append(stage.spell())
            for red in _reduce_candidates(kernel, place, TilePlan.parse(spec), probe, len(probe.channels)):
                for raster in _raster_candidates(place):
                    rows.append({_at(TILE, kaxis): spec, _at(STAGE, kaxis): stage.spell(), _at(REDUCE, kaxis): red, RASTER.name: raster})
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


def _contraction_node(node, place, tile_plan: TilePlan) -> tuple[Contraction, Body]:
    """The high-level :class:`Contraction` structural node for a tiled ``CONTRACTION`` leaf. A
    kernel recognition already nodified (the per-cell scalar contraction — ``_nodify_contraction``
    in ``010_recognize``) only swaps the ``tile`` schedule field; a still-``Map`` form (a fused /
    flash-side contraction) is bound here at fork-emit (seam #1): the ``(a_load, b_load, acc,
    epilogue)`` operand→role facts resolve structurally (:func:`semiring_binding`) — raising
    ``LoweringError`` on an unbindable atom — plus the resolved ``tile_plan`` from the schedule
    fork, and the (m, n) output / K axes off the ``Map``. Returns ``(node, projection)`` — the
    projection has ONE home, the wrapping ``Map``'s body, so the option builders re-wrap it and
    materialize peels it into the store tail (the synthesized grid-``Write`` for a bare contraction
    stays a materialize concern — it needs ``root.output``). A computed-A cone the binding resolves
    is stored INLINE on the ``a`` edge (:func:`make_cone`)."""
    grid = list(place.grid)
    if isinstance(node, Map) and len(node.sources) == 1 and len(grid) >= 2:
        # The recognizer's ``Map(body=projection, sources=(fold,))`` spelling — the ONE bilinear
        # fold under the wrapper is the schedulable unit, the projection its ``Map`` body. The view
        # (``contraction_view``) is DERIVED here, its output axes the placement's trailing grid.
        view = contraction_view(node.sources[0], grid[-2], grid[-1], tuple(grid[:-2]))
        if view is not None:
            return replace(view, tile=tile_plan), node.body
    if len(grid) >= 2:
        view = contraction_view(node, grid[-2], grid[-1], tuple(grid[:-2]))
        if view is not None:
            return replace(view, tile=tile_plan), Body(())
    if isinstance(node, Contraction):
        return replace(node, tile=tile_plan), Body(())
    a_load, b_load, acc, epilogue = semiring_binding(node, place.grid)
    kaxis = reduce_loop(node).axis
    return (
        Contraction(
            axes=(grid[-2], grid[-1]),
            k_axis=kaxis,
            a=a_load if isinstance(a_load, Load) else make_cone(a_load, kaxis.name),
            channels=(Channel(b=b_load, acc=acc),),
            tile=tile_plan,
            lead_axes=tuple(grid[:-2]),
        ),
        epilogue,
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
    # Only a dividing width factorizes losslessly — the enumeration path filters ``k % cta``
    # upstream, so a non-divisor can only arrive via an EMMY_REDUCE pin, and truncating here
    # would silently drop the ``K − w·⌊K/w⌋`` remainder columns of the contraction (the scalar
    # tier has no step check to catch it). Refuse loudly, as a pin should.
    if big_k % w:
        raise ValueError(f"split-K width {w} does not divide K={big_k}; pick a dividing split width.")
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
    partition (:class:`ReducePlan`) that ``030_split_reduce`` consumes into the cross-CTA partial + finalize.

    The additive carrier is built exactly as ``contraction_loop`` / a plain-sum reduce does — an
    ``Accum(op="add").as_carrier()`` (identity ``0.0``, 1 component) — so ``030_split_reduce``'s finalize
    (which reads the carrier's identity + ``as_state_merge``) needs no change. The output tile
    (``tier``) rides the inner ``Contraction``; the ``Reduction`` holds only the K partition.

    An operand ``stage_spec`` is RESOLVED against the **sliced** inner node (its ``kslice`` extent +
    offset operand indices), so eligibility is judged on the pipeline the partial kernel actually
    runs; ``030_split_reduce`` threads the resolved ``Stage`` onto its partial ``TileOp``. The honest-
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
    inner, epi = _contraction_node(tile.op, place, wt)
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
    # Every channel's B is σ-reindexed to its K slice below, which needs a gmem index to rewrite. A
    # computed B would have to slice its producing subtree instead (the mirror of the cone's
    # redundant-statistic split); nothing builds one yet, so the option declines rather than
    # silently slicing the wrong operand.
    if any(not isinstance(ch.b, Load) for ch in inner.channels):
        raise ValueError("split-K needs a materialized B on every channel — a computed B has no gmem index to σ-reindex")
    ksplit, kslice, sigma = _factor_k(inner.k_axis, w)
    if inner.a_computed:
        # REDUNDANT-STATISTIC split: the leading k-invariant run of the cone (the per-row stat
        # prologue — the same K seam ``ops.cone_seam`` reads off the node boundary) stays FULL-ROW in every
        # partition, recomputed redundantly; only the k-indexed per-cell remainder is σ-reindexed
        # to absolute k. The sync fill's own σ (``k := k0 + col``) then composes to
        # ``ksplit·(K/w) + k0 + col``. The projection riding the recognizer's ``Map`` wrapper is
        # folded into the node epilogue so ``030_split_reduce``'s deferred finalize re-applies it
        # after the cross-partition sum (the partial's epilogue becomes the workspace write).
        cone = inner.a
        # The cone's SOURCE node is its row-invariant prologue — the per-row statistic and anything
        # else that does not index K — so it stays whole in every partition (that redundancy is what
        # the split trades for parallelism). Only the cone's ``body``, the per-cell normalize, is
        # σ-reindexed to absolute k. No stmt scan: the K seam is the node boundary.
        inner = replace(
            inner,
            k_axis=kslice,
            a=replace(cone, body=Body(tuple(st.rewrite(lambda nm: nm, sigma) for st in cone.body))),
            channels=tuple(replace(ch, b=replace(ch.b, index=tuple(sigma.apply(e) for e in ch.b.index))) for ch in inner.channels),
        )
    else:
        inner = replace(
            inner,
            k_axis=kslice,
            a=replace(inner.a, index=tuple(sigma.apply(e) for e in inner.a.index)),
            channels=tuple(replace(ch, b=replace(ch.b, index=tuple(sigma.apply(e) for e in ch.b.index))) for ch in inner.channels),
        )
    stage = None
    if inner.a_computed:
        # The sync compute-fill is MANDATORY for a computed-A partial (same contract as
        # ``_warp_option``); resolve it against the SLICED node at the row's spelled depth.
        stage = _resolve_sync_stage(inner, budget, Stage.parse(stage_spec).depth if stage_spec else 1)
        if stage is None:
            raise ValueError(f"split-K on a fused-cone contraction: the sync slabs exceed the {budget} B smem budget at TILE {tile_spec!r}")
    elif stage_spec:
        st = Stage.parse(stage_spec)
        stage = _resolve_warp_stage(inner, st, budget) if wt.is_warp else _resolve_scalar_stage(inner, st, tile.inputs, budget)
    # The resolved pipeline rides the node it decorates — ``030_split_reduce`` carries the sliced
    # contraction into the partial kernel and the stage travels with it.
    inner = replace(inner, stage=stage)
    # The carrier is the sliced fold's own — the 1-component additive ``Accum`` for a plain matmul,
    # the N-component product-monoid state for a multi-channel (gate/up) node — so the split
    # finalize folds exactly the state the kernel accumulates.
    inner_fold = inner.as_fold()
    # ONE composition rule: the sliced fold (multi-channel included) rides the reduce's ``partial``.
    op = Reduction(
        carrier=inner_fold.carrier, axis=ksplit, role=AxisRole.CONTRACTION, partial=Body((inner_fold,)), reduce=ReducePlan.parse(split_spec)
    )
    # The projection rides the ``Map`` wrapper — its ONE home; ``030_split_reduce`` reads it there and
    # retargets it (per-partition atomic store / a deferred finalize after the cross-partition sums).
    if len(epi):
        op = Map(body=epi, sources=(op,))
    kaxis = reduce_loop(tile.op).axis.name  # the ORIGINAL k-axis name — single-eligible-axis keying
    stamped = {**knobs, _at(TILE, kaxis): tile_spec, _at(REDUCE, kaxis): split_spec}
    stamped[_at(STAGE, kaxis)] = stage.spell() if stage is not None else ""
    return TileOp(op=op, name=name, place=place, knobs=stamped)


def _warp_option(
    tile, place, spec: str, name: str, knobs: dict, stage_spec: str = "", budget: int = STATIC_SMEM_CAP, wspec_spec: str = ""
) -> TileOp:
    """One scheduled warp-tier contraction ``TileOp``: ``place`` mapped onto the grid + the warp
    form of the ``TILE`` spec resolved into the warp-atom :class:`TilePlan`, plus an optional operand
    ``STAGE`` resolved into a :class:`Stage`. The tiled :class:`Contraction` leaf is built here (``op``),
    so materialize only ``factorize``\\ s. The packed ``TILE`` codec is the sole on-dict spelling — the
    online-prior featurizer parses it directly (one codec, not a per-knob ``WM``/``WN``/``MMA`` explosion)."""
    wt = TilePlan.parse(spec)
    _check_warp_static_k(tile, wt)
    # Build the tiled Contraction node here — it resolves the operand→role facts internally, so an
    # unbindable atom (a non-Load operand: a computed-cone / demoted matmul) raises and is rejected
    # at fork construction, like the static-K check.
    node, proj = _contraction_node(tile.op, place, wt)
    node = _demote_mixed_a(tile, node)
    # A computed-A node's stage is the mandatory resolved ``sync`` compute-fill (its ``smem`` /
    # ``bk_elems`` are derived, not codec-spelled, so the row's ``"d1/sync"`` re-resolves here);
    # a Load-operand node resolves the copy transports as usual.
    if node.a_computed:
        stage = _resolve_sync_stage(node, budget, Stage.parse(stage_spec).depth if stage_spec else 1)
        assert stage is not None, "computed-A row enumerated past its smem budget"  # _computed_a_rows resolved this
    else:
        stage = _resolve_warp_stage(node, Stage.parse(stage_spec), budget) if stage_spec else None
    # Re-wrap the recognizer's projecting ``Map`` around the tiled node (materialize peels it into
    # the store tail — the same ``project ∘ contract`` spelling the Reduction tiers use).
    node = replace(node, stage=stage)
    emitted = Map(body=proj, sources=(node.as_fold(),)) if len(proj) else node.as_fold()
    # Warp specialization rides ORTHOGONAL to the tile/stage just resolved: an optional WSPEC row /
    # pin splits the warps into roles over this fixed pipeline (gated on the RESOLVED ``stage`` — an
    # ineligible spec leaves no pipeline for a producer to drive, so WSPEC degrades to uniform).
    workers, wspec_spec = _wspec_workers(wspec_spec, stage, node.block_threads)
    # The per-node schedule codecs key ``@<k_axis>`` (the contraction axis this node schedules), so a
    # multi-node kernel can address each node; ``WSPEC`` stays root-global (bare).
    kaxis = node.k_axis.name
    stamped = {**knobs, _at(TILE, kaxis): spec}
    # Honest stamping: the RESOLVED spelling (depth clamps, dropped ring) — the DB row / feature
    # vector must describe the pipeline the kernel actually has. A declined / absent stage stamps
    # the explicit OFF value ``""`` (decided: gmem-direct), never the raw pin — and never nothing:
    # an absent family key means "not offered", and the evidence pick's prefix-consistency reads an
    # absent key as free, letting a gmem-direct leaf inherit a STAGED row's measurement.
    stamped[_at(STAGE, kaxis)] = stage.spell() if stage is not None else ""
    if wspec_spec:
        stamped[WSPEC.name] = wspec_spec
    return TileOp(op=emitted, name=name, place=place, workers=workers, knobs=stamped)


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
            node, proj = _contraction_node(tile.op, place, plan)
        except LoweringError:
            pass  # an unbindable contraction (a non-Load operand) keeps the Map form
        else:
            # Only a built Contraction node can engage operand staging — resolve the pin against it
            # (per-cell / coop-K / unbindable forms stamp None: nothing downstream would read a stage).
            if stage_spec:
                stage = _resolve_scalar_stage(node, Stage.parse(stage_spec), tile.inputs, budget)
            node = replace(node, stage=stage)
            op = Map(body=proj, sources=(node.as_fold(),)) if len(proj) else node.as_fold()
    elif reduce_spec:
        op = nodify_reduce(tile.op, ReducePlan.parse(reduce_spec))
    # ``TILE`` / ``REDUCE`` / ``STAGE`` key ``@<k_axis>`` (the contraction axis this node schedules),
    # unifying the schedule onto the axis-named family. STAGE stamps the RESOLVED spelling, and only
    # when resolution took (see ``_warp_option`` — the same honest-stamping rule).
    kaxis = reduce_loop(tile.op).axis.name
    stamped = {**knobs, _at(TILE, kaxis): spec, _at(REDUCE, kaxis): reduce_spec}
    stamped[_at(STAGE, kaxis)] = stage.spell() if stage is not None else ""
    return TileOp(op=op, name=name, place=place, knobs=stamped)


def _map_reg_width(spec: str) -> int:
    """The inner-axis register width of a scalar ``TILE`` codec applied to a pointwise ``Map`` — the
    ``f<fn>`` register sub-tile's leading count (``regs[0]``), or ``0`` for a warp / atom / unparseable
    codec (a map has no fragment tier — those don't apply)."""
    try:
        plan = TilePlan.parse(spec)
    except (ValueError, KeyError):
        return 0
    return 0 if plan.is_warp else plan.regs[0]


def _map_body_defs(body: Body) -> set[str]:
    """The SSA value names a pointwise ``Map`` body defines (``Load`` names + ``Assign`` names) —
    the set the register-strip unroll renames per copy (axis vars stay put)."""
    defs: set[str] = set()
    for s in body:
        defs.update(s.defines())
    return defs


def _map_strip_option(tile: TileOp, place: Placement, inner: Axis, r: int, spec: str, name: str) -> TileOp:
    """One register-strip candidate: hand each thread ``r`` **contiguous** inner-axis elements. The
    inner free axis shrinks to ``extent/r`` (the grid walks it) and the ``Map`` body is unrolled ``r``
    times — copy ``i`` reads/writes ``inner·r + i`` (blocked layout, contiguous per thread) with its SSA
    names suffixed — then regrouped as ``r`` loads · ``r`` computes · ``r`` writes so the unit-stride
    runs feed ``050_vectorize_loads`` / ``080_vectorize_stores`` (→ one ``float<r>`` access each). The
    width rides the scalar ``TILE`` codec (``f<r>``), keyed by the tiled inner axis."""
    op = tile.op
    ssa = _map_body_defs(op.body)
    loads: list[Stmt] = []
    computes: list[Stmt] = []
    writes: list[Stmt] = []
    for i in range(r):

        def rename(n: str, i: int = i) -> str:  # suffix only the body's SSA names per copy; axis vars stay
            return f"{n}__u{i}" if n in ssa else n

        sigma = Sigma({inner.name: BinaryExpr("+", BinaryExpr("*", Var(inner.name), Literal(r, "int")), Literal(i, "int"))})
        for s in op.body:
            s2 = s.rewrite(rename, sigma)
            (loads if isinstance(s2, Load) else writes if isinstance(s2, Write) else computes).append(s2)
    body = Body((*loads, *computes, *writes))
    new_inner = replace(inner, extent=Dim(inner.extent.as_static() // r))
    new_free = (*place.free[:-1], new_inner)
    new_place = Placement(free=new_free, grid=new_free)
    return TileOp(op=Map(body=body), name=name, place=new_place, knobs={_at(TILE, inner.name): spec})


def _map_strip_fork(tile: TileOp, place: Placement, name: str) -> list[TileOp] | TileOp:
    """The pointwise-map register-strip fork (the ``FREE`` dispatch): option-0 is one element per
    thread (today's flat map, ``TILE=""``); ``f2``/``f4``/``f8`` hand each thread that many contiguous
    inner-axis elements (:func:`_map_strip_option`). Reuses the scalar ``TILE`` codec's ``f<fn>``
    register sub-tile — a pure ``Map`` is the degenerate output tile (no ``n`` unit-tile / atom).
    Offered only for a pure ``Map`` (``source=None``) whose innermost free axis is static and divisible
    by the width; a ``TILE`` pin narrows the ladder (``Knob.narrow``), one surviving candidate applies
    directly (no fork)."""
    op = tile.op
    if not (isinstance(op, Map) and op.source is None) or not place.free:
        return TileOp(op=op, name=name, place=place)
    inner = place.free[-1]
    base = TileOp(op=op, name=name, place=place, knobs={_at(TILE, inner.name): ""})
    # Only a FLAT elementwise body (per-cell Load/Assign/Write, no nested Loop / Accum / carried
    # state) is safely unrollable: every output cell is independent, so replicating adjacent cells
    # and regrouping preserves semantics. A sweep / stateful-fallback Map (which also reaches the
    # FREE dispatch) has cross-cell dependencies the naive copy would break — leave it at option-0.
    if not inner.extent.is_static or not all(isinstance(s, (Load, Assign, Write)) for s in op.body):
        return base
    ext = inner.extent.as_static()
    opts = [base]
    for spec in TILE.narrow(map_tile_moves()):
        r = _map_reg_width(spec)
        if r > 1 and ext % r == 0:
            opts.append(_map_strip_option(tile, place, inner, r, spec, name))
    if TILE.raw() is not None:  # a live pin — the narrowed strip alone (option-0 only when the pin is "")
        return opts[-1] if len(opts) > 1 else base
    return opts if len(opts) > 1 else base


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
        return _map_strip_fork(tile, place, name)
    # A contraction picks its free-axis output tile (``TILE``); a reduction picks its reduce
    # partition (``REDUCE``). Each offers its candidate(s): one applies directly, multiple fork.
    # A contraction ALSO honors a cross-CTA split-K (``g``) / cooperative (``b``/``r``) ``REDUCE``
    # pin — orthogonal to the output tile (``reduce`` = the K partition; ``g`` is consumed by
    # ``030_split_reduce``, ``b``/``r`` by ``_factor._tile_reduce_axis`` on the non-tiled scalar tier).
    # ``TILE`` is the unified output-fragment knob: a candidate whose codec names an atom
    # (``a:<atom>`` — :func:`is_warp_codec`) builds the tensor-core warp option, otherwise the
    # scalar register-tile option (the either-ness — a kernel is one fragment or the other).
    if role is AxisRole.CONTRACTION:
        # The RESTORED enumeration: the tile × stage × reduce legal product (rows keyed
        # ``FAMILY@<k_axis>``), offered as a lazy hierarchical fork tree — greedy descent flattens
        # the rows for one prior-scoring pass; MCTS pays one level per pop. Env pins narrow each
        # family (a fully-pinned space collapses to the single materialized option, no fork). A
        # split ``g`` row routes through the structural ``Reduction ⊃ Contraction`` fork
        # (:func:`_splitk_option`, consumed by ``030_split_reduce``); a warp row through
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
            # Thread the row's structural stamps (``S_warp_eligible``) onto the op. Fork rows carry
            # them for branch identity, but the MATERIALIZED op is what ``realized_knobs`` reads —
            # dropping them here left leaf/evidence rows unstamped while fork rows (deploy
            # candidates) were stamped, fracturing the ``S_*`` evidence signature: deploy-time
            # ``evidence_pick`` never joined the measured -O3 rows, and greedy shipped the online
            # model's unbenched per-cell extrapolation (the 2026-07-07 5090 gate's 330x fp16 miss).
            op_knobs = {**knobs, **{k: v for k, v in row.items() if k.startswith("S_")}}
            # The rasterization codec rides op-level knobs (kernel-scoped launch-order metadata,
            # not a per-node schedule structure) — the kernel materializer's ``grid_tile`` seal
            # reads it back off the TileOp and stamps the grouped decode on the ``Tile``.
            raster_spec = row.get(RASTER.name, "")
            Raster.parse(raster_spec)  # loud pin contract — a malformed spelling fails the row here
            op_knobs = {**op_knobs, RASTER.name: raster_spec}
            if red and ReducePlan.parse(red).needs_split:
                return _splitk_option(tile, place, spec, red, name, op_knobs, stage_spec, _smem_budget(ctx))
            if is_warp_codec(spec):
                return _warp_option(tile, place, spec, name, op_knobs, stage_spec, _smem_budget(ctx), row.get(WSPEC.name, ""))
            return _tile_option(tile, place, spec, name, op_knobs, red, stage_spec, _smem_budget(ctx))

        if len(rows) == 1:
            return _materialize(rows[0])

        def _level(key: str) -> Level:
            return Level((key,), key=lambda r: (r.get(key, ""),))

        # The worker split (``WSPEC``, bare/root-global) is the fourth level, under the pipeline it
        # splits — option-0 ``""`` is uniform SIMT. The launch-order codec (``RASTER``, also bare —
        # one grid, one order) is the fifth — option-0 ``""`` is the flat N-fastest raster.
        levels = [_level(_at(k, kaxis)) for k in (TILE, STAGE, REDUCE)] + [_level(WSPEC.name), _level(RASTER.name)]
        return build_fork_tree(params=rows, levels=levels, materialize=_materialize)
    # A TWISTED streaming reduce whose per-step partial is a contraction pair (the flash tree,
    # :func:`_twisted_pair`) offers its structurally-different schedules as ONE prior-ranked fork:
    # the WARP (fragment-resident) move grid (option-0 = the conservative one-warp / ``2·atom_n``
    # key block — the historical deterministic stamp), the scalar register-vector CHAIN (the FA-2
    # shared-score form), then the reduce-partition tiers (cooperative ``b<n>`` / per-cell serial —
    # the redundant-recompute forms) as :func:`_option` siblings. Every leaf row spells the same
    # key set (``TILE@<qk_k>`` / ``TILE@<pv_k>`` / ``REDUCE@<kv>``, decided-empty where a form
    # doesn't tile) — the evidence pick's prefix-consistency. Pins keep their contracts: a
    # non-empty ``REDUCE`` pin stays the scalar escape (it asks for a reduce partition only the
    # scalar tiers honor), a warp ``TILE`` pin keeps the mma rows alone.
    # A cross-CTA ``g<n>k`` REDUCE pin on the flash tree selects the SPLIT-KV warp rows (the
    # partial keeps fragment residence; ``030_split_reduce`` realizes partial + LSE-combine finalize) —
    # pin-driven, like the demoted warp tier. Any other non-empty REDUCE pin keeps its scalar
    # escape below (it asks for a reduce partition only the scalar tiers honor), as does a split
    # pin no warp row can legally carry (symbolic / non-divisible kv, atomic finalize).
    reduce_pin = REDUCE.raw()
    if reduce_pin:
        try:
            rplan = ReducePlan.parse(reduce_pin)
        except Exception:  # noqa: BLE001 — an unparseable pin keeps the historical scalar escape
            rplan = None
        if rplan is not None and rplan.needs_split and rplan.finalize == "kernel":
            pair = _twisted_pair(tile.op, tile.place.free)
            if pair is not None:
                red, _, _, head, pv = pair
                warps = _twisted_warp_options(tile, name, knobs, _smem_budget(ctx), _tma_allowed(ctx), _f16acc_allowed(ctx))
                rows = _stamp_twisted_split(warps, red.axis.name, rplan)
                if rows:
                    rows = _narrow_flash_forms(rows, head, pv, keyed_only=True)
                    return rows if len(rows) > 1 else rows[0]
    if not REDUCE.narrow([""])[0]:
        pair = _twisted_pair(tile.op, tile.place.free)
        if pair is not None:
            red, _, _, head, pv = pair
            warps = _twisted_warp_options(tile, name, knobs, _smem_budget(ctx), _tma_allowed(ctx), _f16acc_allowed(ctx))
            measured_flash = _rtx4080_dit_flash_deploy(warps, red, head, pv, ctx)
            if measured_flash is not None:
                return measured_flash
            # A live **warp** ``TILE`` pin (``a:<atom>…``) OR a non-empty ``STAGE`` pin keeps the mma
            # rows alone: ONLY the warp tier stages, so a staging pin (the ``--ab STAGE=…`` /
            # ``emmy tune`` staging probe, or ``EMMY_STAGE``) must not fall through to the chain /
            # scalar reduce-partition siblings and let the prior bury the (necessarily
            # lower-occupancy) staged warp form under a higher-occupancy scalar form — the
            # scalar-fallback that made a staged-flash A/B row read as a 100× regression.
            # ``warps == []`` (a non-warp-eligible flash) still degrades to scalar below, so a stage
            # pin on such a shape stays a graceful no-op. A NON-warp ``TILE`` pin (``a:scalar`` /
            # ``""`` / the chain's ``f<d>``) falls through to the scalar forms and narrows them
            # below (:func:`_narrow_flash_forms`) — routing on ANY pin locked the chain row out.
            stage_pinned = STAGE.raw() is not None and bool(STAGE.narrow([""])[0])
            warp_pinned = TILE.raw() is not None and is_warp_codec(TILE.raw())
            if warps and (warp_pinned or stage_pinned):
                # The axis-keyed ``TILE@<dd>``/``TILE@<pj>`` pins (a static attention golden's
                # spelling) must still select their geometry among the kept mma rows — this early
                # return used to skip the narrowing entirely, so a golden that also pins ``STAGE``
                # benched the prior's tile under the pinned stage (the findings-5 F2 coercion).
                warps = _narrow_flash_forms(warps, head, pv, keyed_only=True)
                return warps if len(warps) > 1 else warps[0]
            chain = _twisted_chain_option(tile, place, name, knobs)
            forms = [*warps, *([chain] if chain is not None else [])]
            if forms:
                empty = {_at(TILE, head.k_axis.name): "", _at(TILE, pv.k_axis.name): "", _at(STAGE, red.axis.name): ""}
                forms += [_option(tile, place, spec, name, {**knobs, **empty}) for spec in _reduce_specs(tile, place, ctx)]
                forms = _narrow_flash_forms(forms, head, pv)
                return forms if len(forms) > 1 else forms[0]
        else:
            # A PLANAR ⊗-fold over a computed MAP cone — the fused producer → matmul edge — honors
            # a warp ``TILE`` pin (pin-driven, like the matmul warp tier): the demoted contraction
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


def _canon_tile_spec(spec: str) -> str:
    """A ``TILE`` spelling canonicalized through the codec (``a:scalar`` ≡ ``""``, ``f64x1`` ≡
    ``f64``) so pin-only aliases compare equal to stamped row spellings; an unparseable pin passes
    through (it just won't match, and :func:`_narrow_flash_forms` degrades gracefully)."""
    if not spec:
        return ""
    try:
        return TilePlan.parse(spec).spell()
    except Exception:  # noqa: BLE001 — an invalid pin must not crash the fork build
        return spec


def _narrow_flash_forms(forms: list[TileOp], head: Contraction, pv: Contraction, *, keyed_only: bool = False) -> list[TileOp]:
    """Narrow the flash fork's leaf rows by the live per-node ``TILE`` pins. Every flash row spells
    the same key set (``TILE@<qk_k>`` / ``TILE@<pv_k>``), so a pin selects rows by their stamped
    spelling: ``f<d>`` on the PV k-axis keeps the CHAIN row, ``""`` / ``a:scalar`` the per-cell
    rows, a warp codec the mma rows (the bare-warp-pin fast path returns before this). A pin that
    matches NO row keeps the full fork — the same graceful degrade as a warp pin that doesn't fit
    the flash form (warned, greedy picks by prior). ``keyed_only`` reads the explicit
    ``TILE@<axis>`` pins alone, skipping ``narrow_at``'s bare-``TILE`` fallback — the warp-rows
    caller has already consumed a bare pin in ``_twisted_warp_options``, and matching it against
    the *derived* pj spelling would spuriously miss every row."""
    from emmy import config  # noqa: PLC0415

    live: dict[str, str] = {}
    for ax in (head.k_axis.name, pv.k_axis.name):
        pin = config.knob_raw(f"{TILE.name}@{ax}") if keyed_only else TILE.narrow_at(ax)
        if pin is not None:
            live[ax] = _canon_tile_spec(pin)
    if not live:
        return forms
    kept = [f for f in forms if all(_canon_tile_spec(f.knobs.get(_at(TILE, ax), "")) == p for ax, p in live.items())]
    if not kept:
        logger.warning("TILE pin(s) %s match no flash form (warp / chain / per-cell); keeping the full fork", live)
        return forms
    return kept


def _rtx4080_dit_flash_deploy(forms: list[TileOp], red: Reduction, head: Contraction, pv: Contraction, ctx) -> TileOp | None:
    """The measured DiT S256/Hd72 flash winner on RTX 4080, or ``None``.

    Like the contraction defaults above, this applies only to a deterministic
    unpinned deploy; autotuning retains every flash geometry."""
    if (
        ctx is None
        or not ctx.validate_pins
        or ctx.gpu_name != "NVIDIA GeForce RTX 4080"
        or any(k.raw() is not None for k in (TILE, STAGE, REDUCE))
        or not (red.axis.extent.is_static and head.m_axis.extent.is_static and head.k_axis.extent.is_static and pv.n_axis.extent.is_static)
        or (
            red.axis.extent.as_static(),
            head.m_axis.extent.as_static(),
            head.k_axis.extent.as_static(),
            pv.n_axis.extent.as_static(),
        )
        != (256, 256, 72, 72)
    ):
        return None
    qk = "a:mma_m16n8k16_f16_f32/w2x1/f2x8/k5"
    expect = "a:mma_m16n8k16_f16_f32/w2x1/f2x9/k4"
    for form in forms:
        if (
            form.knobs.get(_at(TILE, head.k_axis.name), "") == qk
            and form.knobs.get(_at(TILE, pv.k_axis.name), "") == expect
            and form.knobs.get(_at(REDUCE, red.axis.name), "") == ""
            and form.knobs.get(_at(STAGE, red.axis.name), "") == ""
        ):
            return form
    logger.warning("measured RTX 4080 DiT flash schedule is no longer enumerable")
    return None


def _stamp_twisted_split(rows: list[TileOp], kv_name: str, plan: ReducePlan) -> list[TileOp]:
    """The flash split-KV rows: each warp row with the pinned cross-CTA partition stamped on its
    :class:`Reduction` node (``030_split_reduce`` consumes it) and spelled on ``REDUCE@<kv>``. Per-row
    legality for a STATIC kv: an extent divisible by ``cta`` and a slice divisible by the row's own
    streaming key block (the staged chunking + fragment masks assume block-whole slices); an
    illegal row is dropped, not degraded. A SYMBOLIC kv always stamps — ``030_split_reduce`` builds
    the bn-aligned runtime slice width and the absolute ``bound`` the realizer stops/masks against."""
    out: list[TileOp] = []
    for r in rows:
        red = r.op.source if isinstance(r.op, Map) else r.op
        head = red.partial[0]
        bn = head.tile.regs[1] * head.tile.atom.shape[1]
        ext = red.axis.extent
        if ext.is_static and (ext.as_static() % plan.cta != 0 or (ext.as_static() // plan.cta) % bn != 0):
            continue
        op2 = _with_reduce(r.op, plan)
        out.append(replace(r, op=op2, knobs={**r.knobs, _at(REDUCE, kv_name): plan.spell()}))
    return out


def _twisted_pair(op, free) -> tuple[Reduction, Reduction, Reduction, Contraction, Contraction] | None:
    """The flash-shaped ``TWISTED`` streaming contraction pair — ``(reduction, head_fold, pv_fold,
    head, pv)``: the STORED ``role=CONTRACTION`` folds (the score at the partial's head, the single
    computed-A expect later in the sequence) plus their DERIVED :class:`Contraction` views (output
    axes off the placement's trailing ``free`` — the query axis, and the stream axis for the score /
    the value axis for the expect). ``None`` when not a streaming pair (an online-softmax / RMSNorm
    ``TWISTED`` reduce takes the reduce-partition tiers). The one structural guard the warp / chain /
    scalar flash forms share; each form's own demands (a gmem-``Load`` A, the mma atom's dtype /
    divisibility, the chain's register budget) stay with its builder. Stamping targets the FOLDS
    (`s is head_fold` in the partial); reads go through the views."""
    red = op.source if isinstance(op, Map) and isinstance(op.source, Reduction) else (op if isinstance(op, Reduction) else None)
    if red is None or red.role is not AxisRole.TWISTED or red.carrier.twist.family != "exp" or len(red.partial) == 0:
        return None
    head_fold = red.partial[0]
    if not is_contraction_fold(head_fold) or len(free) < 2:
        return None
    tail_folds = [st for st in list(red.partial)[1:] if is_contraction_fold(st)]
    if len(tail_folds) != 1:
        return None
    pv_fold = tail_folds[0]
    head = contraction_view(head_fold, free[-2], red.axis)
    pv = contraction_view(pv_fold, free[-2], free[-1])
    if head is None or pv is None or not pv.a_computed:
        return None
    return red, head_fold, pv_fold, head, pv


def _twisted_chain_option(tile: TileOp, place, name: str, knobs: dict) -> TileOp | None:
    """The scalar register-vector (CHAIN) schedule for a ``TWISTED`` streaming contraction pair —
    the FA-2 shared-score form: the expect contraction's output column axis leaves the grid and
    rides a per-thread register vector (a scalar ``TilePlan`` register tile on the node), so the
    score computes ONCE per streamed key and is shared across the columns (vs the per-cell tier's
    redundant recompute per column). A fork SIBLING of the warp / reduce-partition schedules,
    offered when the column axis is small + static (``≤ _CHAIN_MAX_D``, the register budget)."""
    pair = _twisted_pair(tile.op, tile.place.free)
    if pair is None:
        return None
    red, _, pv_fold, head, pv = pair
    op = tile.op
    d_ax = pv.n_axis
    grid = list(place.grid)
    if not d_ax.extent.is_static or not grid or grid[-1].name != d_ax.name:
        return None
    d = d_ax.extent.as_static()
    if d > _CHAIN_MAX_D:
        return None
    pv2 = replace(pv_fold, tile=TilePlan(regs=(d, 1)))  # scalar reg order (reg_n, reg_m): the column vector
    partial = tuple(pv2 if st is pv_fold else st for st in red.partial)
    red2 = replace(red, partial=type(red.partial)(partial))
    op2 = replace(op, sources=(red2,)) if isinstance(op, Map) else red2
    # A fork SIBLING of the warp / reduce-partition schedules: the resolved register-vector plan is
    # stamped (keyed on the PV contraction's k axis, like every per-node schedule codec) — the row
    # identity the DB / prior separate it from the per-cell serial by — and the sibling families it
    # does NOT schedule are decided-empty so every flash leaf row spells the same key set.
    stamped = {
        **knobs,
        _at(TILE, pv.k_axis.name): pv2.tile.spell(),
        _at(TILE, head.k_axis.name): "",
        _at(REDUCE, red.axis.name): "",
        _at(STAGE, red.axis.name): "",
    }
    return TileOp(op=op2, name=name, place=Placement(free=tile.place.free, grid=tuple(grid[:-1])), knobs=stamped)


def _composes(red: Reduction) -> bool:
    """True when the reduce's per-step ``partial`` composes another structural node (split-K's
    sliced contraction, flash's score) — the ONE spelling for a composed reduce, so a "bare
    statistic reduce" test is just its negation."""
    return any(isinstance(s, (Map, Reduction, Contraction)) for s in red.partial)


def _with_twisted_stage(op, stage: Stage | None):
    """Stamp the resolved K/V pipeline onto the flash tree's :class:`Reduction` — the node it
    decorates (the streaming reduce whose per-step partial reads the staged slabs)."""
    red = op.sources[0] if isinstance(op, Map) and op.sources else op
    staged = replace(red, stage=stage)
    return replace(op, sources=(staged,)) if isinstance(op, Map) and op.sources else staged


def _demoted_warp_option(tile: TileOp, place, name: str, knobs: dict) -> TileOp | None:
    """The warp (mma) candidate for a **demoted-cone contraction** — a ``PLANAR`` ⊗-fold whose
    lift multiplies a gmem ``Load`` B with a computed pure-MAP cone A (the fused producer → matmul
    edge: ``f(x, …) @ w``), or ``None`` (stay scalar). PIN-DRIVEN like the matmul warp tier: fires
    only under a warp ``TILE`` pin. Nodifies the fold to a computed-A :class:`Contraction` (the
    same ``a = Body`` the flash P@V rides) and stamps the ``sync`` compute-fill
    :class:`Stage` — the producer cone materializes the A tile straight into the smem slab the
    ``ldmatrix`` drain reads (the fused edge IS the mma tier's ``sync`` transport). First cut:
    exact-cover geometry only (static M/N/K divisible by the tile / K-chunk — no masked overhang),
    and the cone may read the ``(m, k)`` axes only."""
    spec = TILE.narrow([""])[0]
    if not is_warp_codec(spec):
        return None
    op = tile.op
    red = op.source if isinstance(op, Map) and isinstance(op.source, Reduction) else (op if isinstance(op, Reduction) else None)
    if red is None or red.role is not AxisRole.PLANAR or _composes(red) or red.carrier.twist.family != "id":
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
    # A square fold (``multiply(x, x)`` — a mean-of-squares reduce) has no distinct A arg; it is
    # not a matmul edge, so the pin degrades gracefully instead of raising StopIteration.
    a_name = next((a for a in lift.args if a != b_name), None)
    if a_name is None:
        return None
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
        a=make_cone(list(cone), k_ax.name),
        channels=(Channel(b=loads[b_name], acc=acc.name),),
        tile=wt,
        lead_axes=tuple(grid[:-2]),
    )
    stage = Stage(transport="sync", smem=(node.a_name,), bk_elems=bk_elems)
    stamped = {**knobs, _at(TILE, k_ax.name): spec}
    node = replace(node, stage=stage)
    emitted = Map(body=epilogue, sources=(node.as_fold(),)) if len(epilogue) else node.as_fold()
    return TileOp(op=emitted, name=name, place=place, knobs=stamped)


def _resolve_twisted_stage(
    stage: Stage, kv_extent, bn: int, head_dim: int, d_v: int, elem_bytes: int, budget: int, m_rows: int = 0
) -> Stage | None:
    """Resolve an operand ``Stage`` against a warp-flash streaming pair — the K/V slabs of one
    ``bn``-key streaming step (K ``bn × head_dim`` in its native N-major layout, V ``bn × d_v``
    K-major; both verbatim row copies, so staging stays bit-identical to gmem-direct). Two
    transports resolve: **cp.async** (cooperative fills, +16 B padded rows for the bank-conflict
    break) and **TMA** (the batched K/V encode as rank-N boxes with leading extent-1 dims —
    ``(1, 1, bn, head_dim)`` — the load's own batch/head index exprs supplying the origin coords;
    the slabs stay dense and take the hardware swizzle + drain XOR instead of padding; box dims
    cap at the 256 hardware limit). ``sync`` has nothing to overlap. A **symbolic** kv stages
    under both transports — TMA rides the runtime globalDim and zero-fills the box overhang;
    cp.async clamp-reads the tail's key rows to the last valid key — and the streaming drain's
    tail-key masks (the same clamp the gmem-direct symbolic path makes) keep either bit-identical.
    A static, non-block-divisible kv has no tail mask, so it stays gmem-direct on both.
    ``depth`` clamps so the ring's K+V slot pairs fit the smem
    ``budget``; ``reg_depth`` clamps to 1 (no ldmatrix ping-pong on the streaming drain yet);
    ``bk_elems`` records the streamed keys per step."""
    if stage.transport not in ("cp.async", "tma"):
        return None
    if stage.alt:
        # The ALTERNATING single-slab pipeline (``d1/tma/alt``): one K slab + one V slab, each on
        # its own mbarrier, refilled in the phase that no longer reads it (K under softmax + P·V,
        # V under the next step's Q·K) — the FA-2 choreography that overlaps a WIDE (64-key)
        # streaming block's copies in HALF the paired ring's smem. The Q (query) tile stages
        # through smem too — a padded row-major slab filled once, its A fragments ldmatrix'd per
        # atom-K chunk — which frees the resident Q registers that made the wide block spill.
        # A SYMBOLIC kv rides the ring's tail story unchanged (TMA zero-fills the box overhang,
        # cp.async clamp-reads the tail's key rows, the drain masks zero the overhanging P
        # columns — bit-identical either way); the kill-point refill clamps onto the runtime
        # last chunk exactly as the ring prefetch does, and a symbolic M clamp-reads the
        # staged-Q fill's overhanging query rows (their outputs are store-guarded). A static
        # non-block-divisible kv has no tail mask, so it stays gmem-direct (the ring's gate).
        # Both async transports ride the same seam: TMA arms per-operand mbarriers
        # (``d1/tma/alt``); cp.async commits each fill into its own group and a uniform
        # ``wait_group(1)`` completes the older sibling (``d1/cp/alt`` — the sm_89 form).
        if m_rows <= 0 or (kv_extent.is_static and kv_extent.as_static() % bn != 0):
            return None
        if stage.transport == "tma":
            if bn > 256 or head_dim > 256 or d_v > 256 or (head_dim * elem_bytes) % 16 or (d_v * elem_bytes) % 16:
                return None
            kv_pad = 0  # dense hardware-swizzled slabs
        else:
            kv_pad = 16  # the two K/V slabs' padded rows (2 x _twist._PAD elems)
        q_bytes = m_rows * (head_dim + 8) * elem_bytes  # +8 elem row pad (the flash cp-path bank break)
        if q_bytes + bn * (head_dim + d_v + kv_pad) * elem_bytes > budget:
            return None
        return replace(stage, reg_depth=1, bk_elems=bn)
    if stage.transport == "cp.async":
        # A symbolic kv stages: the fill clamp-reads the overhanging tail rows to the last valid
        # key (cp.async has no OOB zero-fill) and the drain's tail masks zero their P columns, so
        # the duplicates contribute exactly 0 — bit-identical to gmem-direct, the cp.async
        # counterpart of TMA's box zero-fill. The static non-divisible guard below is defensive:
        # the warp form's own divisibility gate rejects that geometry before any stage resolves.
        if kv_extent.is_static and kv_extent.as_static() % bn != 0:
            return None
    elif kv_extent.is_static and kv_extent.as_static() % bn != 0:
        # TMA + a STATIC non-block-divisible kv has no tail mask (masking is symbolic-only) —
        # stay gmem-direct. A symbolic kv falls through: TMA zero-fills its box overhang.
        return None
    if stage.transport == "tma":
        # Box dims must fit the 1..256 hardware range, and the inner (contiguous) row spans must
        # be 16 B-aligned for the box copy (guaranteed ≥ the mma divisibility gates, checked for
        # form). Rank ≤ 5 holds structurally: SDPA operands are (batch…, S, D) rank ≤ 4.
        if bn > 256 or head_dim > 256 or d_v > 256 or (head_dim * elem_bytes) % 16 or (d_v * elem_bytes) % 16:
            return None
        pad = 0  # TMA deposits dense — the hardware swizzle replaces the row pad
    else:
        pad = 16  # the two slabs' padded rows (2 × _twist._PAD)
    slot_bytes = bn * (head_dim + d_v + pad) * elem_bytes
    if slot_bytes > budget:
        return None
    depth = min(stage.depth, budget // slot_bytes)
    # ``reg_depth`` ≤ 2: the streaming drains support the two-slot ldmatrix ping-pong (the next
    # atom-K step's B fragments load while the current step's mmas consume — breaking the WAR
    # hazard on the fragment registers); deeper ping-pongs cost registers the fm chains don't have.
    return replace(stage, depth=depth, ring=stage.ring and depth >= 2, reg_depth=min(stage.reg_depth, 2), bk_elems=bn)


def _twisted_stage_candidates(
    kv_extent, bn: int, head_dim: int, d_v: int, elem_bytes: int, budget: int, tma_ok: bool = True, m_rows: int = 0
) -> list[Stage | None]:
    """The K/V operand-stage candidates for one warp-flash geometry row — gmem-direct ``None``
    first (the conservative option-0), then every ``stage_moves`` entry that RESOLVES against the
    stream (:func:`_resolve_twisted_stage`), deduped on the resolved spelling (a ``p2`` move clamps
    to the same pipeline as its ``p1`` sibling and drops; an over-budget depth clamps down the same
    way). A pinned ``STAGE`` is authoritative: the resolved pin alone, or gmem-direct with a log
    line when it declines — the same pin-validity degrade as the warp ``TILE`` pin. ``tma_ok`` is
    the target's TMA availability (:func:`_tma_allowed`): below sm_90 a ``d*/tma*`` move / pin
    declines here rather than being offered and failing to compile."""

    def resolve(spec: str) -> Stage | None:
        st = Stage.parse(spec)
        if st.transport == "tma" and not tma_ok:
            return None  # TMA is Hopper+ (sm_90); nvcc has no sm_89a — decline below it
        return _resolve_twisted_stage(st, kv_extent, bn, head_dim, d_v, elem_bytes, budget, m_rows)

    if STAGE.raw() is not None:
        pinned = STAGE.narrow([""])[0]
        if pinned:
            try:
                r = resolve(pinned)
            except ValueError:
                r = None
            if r is not None:
                return [r]
            logger.warning(
                "STAGE pin %r does not resolve against the warp-flash stream at this geometry "
                "(%d-key block x head_dim %d: needs an async transport, a %d-key-block-divisible static kv, "
                "TMA box dims <= 256, and K/V(+staged-Q for alt) slabs within the smem budget); "
                "the flash kernel stays gmem-direct",
                pinned,
                bn,
                head_dim,
                bn,
            )
        return [None]
    out: list[Stage | None] = [None]
    spelled: set[str] = set()
    # ``d1/tma/alt`` (the alternating single-slab pipeline) rides the flash candidate list only —
    # it is not a ``stage_moves`` entry (the matmul resolvers decline it), and it resolves only
    # where the wide-block Q+K+V slabs fit (:func:`_resolve_twisted_stage`).
    for move in [*stage_moves(warp=True), "d1/tma/alt", "d1/cp/alt"]:
        if not move:
            continue
        r = resolve(move)
        if r is not None and r.spell() not in spelled:
            spelled.add(r.spell())
            out.append(r)
    return out


def _twisted_warp_options(
    tile: TileOp, name: str, knobs: dict, budget: int = STATIC_SMEM_CAP, tma_ok: bool = True, f16acc_ok: bool = False
) -> list[TileOp]:
    """The fragment-resident (tensor-core) candidates for a ``TWISTED`` streaming reduce — the
    warp-flash MOVE GRID over :func:`~emmy.compiler.pipeline.search.space.twisted_warp_moves`'s
    ``(warps_m, key_atoms)`` geometry — or ``[]`` (not eligible: the scalar options stand alone).
    Eligible when the tree is the streaming contraction pair (:func:`_twisted_pair`) with a gmem
    ``Load`` A operand producing the score, and the mma atom's own demands hold (a 16-bit operand
    dtype; a static head contraction axis (a non-divisible final atom is gmem-zero-filled) and the
    expect's output axis divisible by the atom; a static
    stream / query extent divisible by the candidate's block, since a static ragged tail has no
    fragment mask — the symbolic path masks at the fragment and guards the gmem reads). The
    same-per-node stamping rule as ``_warp_option``: the two contractions get their mma
    :class:`TilePlan`\\ s — the Q@K score block ``key_atoms·atom_n`` keys wide, the value dim folded
    into the expect tile whose ``bk`` covers the block — and the placement maps ``warps_m`` warps
    per CTA, each owning ``atom_m`` query rows; the value axis leaves the grid. Each geometry row
    crosses with its K/V operand-stage candidates (:func:`_twisted_stage_candidates` — gmem-direct
    option-0, then the resolver-gated cp.async ring depths, keyed ``STAGE@<kv>``). An additive
    ``(m, kv)`` score bias (the explicit SDPA ``attn_mask``) realizes as a per-element fragment
    bias load (``FragmentBiasAdd``); a bias indexed beyond ``(m, kv)`` declines. A warp ``TILE`` pin
    narrows the grid to the pinned geometry (loud on a divisibility violation — the pin contract);
    a pin that doesn't fit the flash form (wrong atom / a column split / a foreign ``bk``) declines
    the tier with a log line — the standard pin-validity degrade, since a bare warp pin may target
    another kernel in the same graph."""
    pair = _twisted_pair(tile.op, tile.place.free)
    if pair is None:
        return []
    red, head_fold, pv_fold, head, pv = pair
    op = tile.op
    if not isinstance(head.a, Load):
        return []
    channels = red.carrier.twist.channels
    if len(channels) != 3 or channels[1].lift is not None or channels[2].lift is None:
        return []
    q_tensor = tile.inputs.get(head.a.input) if tile.inputs else None
    atom_name = {"f16": "mma_m16n8k16_f16_f32", "bf16": "mma_m16n8k16_bf16_f32"}.get(
        getattr(getattr(q_tensor, "dtype", None), "name", None)
    )
    if atom_name is None:
        return []
    atom = ATOM_REGISTRY[atom_name]
    if not atom.c_to_a_repack:
        return []  # the fragment realizer feeds P@V via the C→A register repack — an atom without it has no tier
    atom_m, atom_n, atom_k = atom.shape
    head_dim, d_v = head.k_axis.extent, pv.n_axis.extent
    if not (head_dim.is_static and head_dim.as_static() > 0 and d_v.is_static and d_v.as_static() % atom_n == 0):
        return []
    kv_ext, m_ext = red.axis.extent, head.m_axis.extent
    m_name, kv_name = head.m_axis.name, red.axis.name
    for s in list(red.partial)[1:]:
        if isinstance(s, Load) and s.index and not {v for e in s.index for v in e.free_vars()} <= {m_name, kv_name}:
            return []  # a score bias indexed beyond (m, kv) — no fragment realization for it
        # An additive (m, kv) score bias (the explicit SDPA attn_mask) IS realizable: the fragment
        # realizer loads the bias tile per element at its absolute coordinates (FragmentBiasAdd).

    # TMA staging derives its box-origin coords POSITIONALLY (batch dims = the load index's leading
    # components, the kv row at [-2], the contiguous head_dim last) — a K/V trace whose kv axis sits
    # elsewhere (the un-transposed (B, S, H, D) layout: kv at position 1) would leak the raw axis
    # var into the emitted coords (an undefined identifier in the kernel). cp.async is unaffected —
    # its fill substitutes by axis NAME. Decline the tma moves for such layouts.
    def _kv_penultimate(load) -> bool:
        idx = load.index
        return (
            len(idx) >= 2
            and isinstance(idx[-2], Var)
            and idx[-2].name == kv_name
            and not any(kv_name in e.free_vars() for e in (*idx[:-2], idx[-1]))
        )

    tma_ok = tma_ok and _kv_penultimate(head.b) and _kv_penultimate(pv.b)
    # The fragment loaders step gmem rows at the buffer's REAL row stride, derived from the load
    # index + buffer shape (``gmem_row_stride`` — H·D on an un-transposed (B, S, H, D) trace,
    # where the old trailing-extent assumption read the wrong rows: the gemma layer-0 NaN).
    # Underivable strides (an axis split across index components, a non-affine use, a symbolic
    # trailing extent) have no fragment realization — decline the tier.
    strides = (
        gmem_row_stride(head.a, m_name, tile.inputs),
        gmem_row_stride(head.b, head.n_axis.name if head.b_trans else head.k_axis.name, tile.inputs),
        gmem_row_stride(pv.b, pv.n_axis.name if pv.b_trans else kv_name, tile.inputs),
    )
    if any(s is None for s in strides):
        return []
    bk = (head_dim.as_static() + atom_k - 1) // atom_k
    # The f16-accumulate PV sibling (``f16acc_ok`` — the F16_MMA_F32_ACC / FAST_MATH gate): each
    # geometry row doubles with a variant whose **PV plan** rides the ``_f16acc`` atom (the O
    # accumulator promotes per streaming block in the realizer; scores stay f32-accumulate). An
    # axis-keyed ``TILE@<pv_k>`` pin naming the sibling (a recorded STATIC golden's spelling) also
    # offers it, as does a BARE ``TILE`` pin spelling the sibling PV plan (the masked-flash golden
    # form — see the pinned branch below), so pinned rows replay without the gate — pins are
    # authoritative.
    from emmy import config  # noqa: PLC0415 — the same deferred import _narrow_flash_forms uses

    sibling = _F16ACC_ATOMS.get(atom_name)
    pv_pin = config.knob_raw(f"{TILE.name}@{pv.k_axis.name}")
    pv_atoms = [atom]
    if sibling is not None and (f16acc_ok or (pv_pin is not None and f"a:{sibling}" in pv_pin)):
        pv_atoms.append(ATOM_REGISTRY[sibling])
    pinned = TILE.raw() is not None
    if pinned:
        spec = TILE.narrow([""])[0]
        if not is_warp_codec(spec):
            return []  # a scalar / empty pin asks for a tier this form doesn't offer
        want = TilePlan.parse(spec)
        if sibling is not None and want.atom.name == sibling:
            # A bare pin naming the f16-accumulate SIBLING spells the **PV plan** verbatim — the
            # masked-flash fast-math golden form: a symbolic trace resolves no ``TILE@<axis>`` key,
            # so a dynamic ``[fm]`` golden records its PV plan (the exact string the static twin
            # stamps on ``TILE@<pv_k>``) as its one bare ``TILE``, and the realized stamp then
            # equals the pin — the replay integrity gate holds. Scores stay on the base
            # f32-accumulate atom as always; the geometry recovers from the PV plan (``um``/``fm``
            # shared, ``nt`` from its K-chunk — ``bk·atom_k/atom_n``, the streamed key block).
            if want.units[1] != 1 or want.regs[1] != d_v.as_static() // atom_n or (want.bk * atom_k) % atom_n:
                logger.warning(
                    "bare f16-accumulate TILE pin %r does not spell this flash form's PV plan "
                    "(w<um>x1/f<fm>x%d/k<block/%d>); the flash kernel stays scalar",
                    spec,
                    d_v.as_static() // atom_n,
                    atom_k,
                )
                return []
            pv_atoms = [ATOM_REGISTRY[sibling]]
            triples = [(want.units[0], want.bk * atom_k // atom_n, want.regs[0])]
        else:
            if want.atom.name != atom_name or want.units[1] != 1 or want.bk != bk:
                logger.warning(
                    "warp TILE pin %r does not fit the flash form (atom %s, w<um>x1/f<fm>x<nt>/k%d); the flash kernel stays scalar",
                    spec,
                    atom_name,
                    bk,
                )
                return []
            if (want.regs[1] * atom_n) % atom_k:
                # The streamed key block must be a multiple of the P@V atom-K step — an odd nt leaves a
                # partial expect fold (the C→A repack pairs k-adjacent score fragments). Loud, per the
                # pin contract (divisibility violation).
                raise ValueError(f"warp TILE pin: key block {want.regs[1] * atom_n} is not a multiple of the P@V atom K-step {atom_k}")
            triples = [(want.units[0], want.regs[1], want.regs[0])]
    else:
        triples = twisted_warp_moves()
    out: list[TileOp] = []
    elem_bytes = atom.operand_dtype("b").nbytes
    for um, nt, fm in triples:
        bn = nt * atom_n  # the streaming block: nt key atoms per step
        if kv_ext.is_static and kv_ext.as_static() % bn != 0:
            if pinned:
                raise ValueError(f"warp TILE pin: key block {bn} does not divide the static KV extent {kv_ext.as_static()}")
            continue
        if m_ext.is_static and m_ext.as_static() % (um * fm * atom_m) != 0:
            if pinned:
                raise ValueError(
                    f"warp TILE pin: {um} warps × {fm} query tiles × {atom_m} rows do not divide the static M extent {m_ext.as_static()}"
                )
            continue
        qk_plan = TilePlan(atom=atom, units=(um, 1), regs=(fm, nt), bk=bk)
        variants: list[tuple[TilePlan, object]] = []
        for pv_atom in pv_atoms:
            pv_plan = TilePlan(atom=pv_atom, units=(um, 1), regs=(fm, d_v.as_static() // atom_n), bk=max(1, bn // atom_k))
            partial = tuple(
                replace(s, tile=qk_plan) if s is head_fold else (replace(s, tile=pv_plan) if s is pv_fold else s) for s in red.partial
            )
            red2 = replace(red, partial=type(red.partial)(partial))
            variants.append((pv_plan, replace(op, sources=(red2,)) if isinstance(op, Map) else red2))
        # ``um`` warps per CTA, each owning ``fm`` register query tiles of ``atom_m`` rows: the
        # query axis shrinks to its CTA-block count; the value (expect output) axis folds into the
        # fragment tile and leaves the grid.
        grid = tuple(
            Axis(name=ax.name, extent=ax.extent.ceil_div(um * fm * atom_m), window=Window(parent=ax.source_axis or ax))
            if ax.name == m_name
            else ax
            for ax in tile.place.free
            if ax.name != pv.n_axis.name
        )
        place = Placement(free=tile.place.free, grid=grid)

        # Both contractions' plans are stamped (each keyed on its node's k axis), the reduce
        # partition decided-empty, and the K/V operand stage on the STREAM axis (the resolved
        # spelling, or the explicit OFF ``""`` — the honest-stamping rule), so every flash leaf
        # row spells the same key set. A resolved **TMA** stage additionally offers the WSPEC
        # producer-band splits (the same legality as the matmul tier's ``_wspec_candidates``:
        # ``_wspec_workers`` gates the thread budgets — the aux band must not exceed the
        # ``32·um`` compute band); a degraded candidate is skipped rather than duplicating the
        # uniform row.
        # Staging byte-copies the operands into slabs typed at the ATOM's operand width — an
        # operand traced at a different dtype (the gemma layer's f32 V intermediate) would deposit
        # wrong-sized elements and drain garbage (the gemma layer-0 NaN's second head). Gmem-direct
        # fragment loads convert per element, so a dtype-mismatched operand keeps the warp tier but
        # declines its stage rows; ``alt`` additionally stages Q, so it also needs a matching A.
        def _buf_dtype_name(load) -> str | None:
            t = tile.inputs.get(load.input) if tile.inputs else None
            return getattr(getattr(t, "dtype", None), "name", None)

        b_dt = atom.operand_dtype("b").name
        kv_stage_ok = _buf_dtype_name(head.b) == b_dt and _buf_dtype_name(pv.b) == b_dt
        q_stage_ok = _buf_dtype_name(head.a) == atom.operand_dtype("a").name
        stage_cands = (
            [None]
            if not kv_stage_ok or head_dim.as_static() % atom_k
            else _twisted_stage_candidates(
                kv_ext, bn, head_dim.as_static(), d_v.as_static(), elem_bytes, budget, tma_ok, m_rows=um * fm * atom_m
            )
        )
        for stage in stage_cands:
            if stage is not None and stage.alt and not q_stage_ok:
                continue  # alt stages Q through smem too — a dtype-mismatched Q cannot byte-copy
            wspecs = list(WSPEC.narrow(wspec_moves())) if (stage is not None and stage.transport == "tma" and not stage.alt) else [""]
            for wspec_cand in wspecs:
                workers, wspec_spec = _wspec_workers(wspec_cand, stage, 32 * um)
                if wspec_cand and workers is None:
                    continue  # over-budget / illegal split — identical to the uniform row
                for pv_plan, op2 in variants:
                    stamped = {
                        **knobs,
                        _at(TILE, head.k_axis.name): qk_plan.spell(),
                        _at(TILE, pv.k_axis.name): pv_plan.spell(),
                        _at(REDUCE, kv_name): "",
                        _at(STAGE, kv_name): stage.spell() if stage is not None else "",
                    }
                    if wspec_spec:
                        stamped[WSPEC.name] = wspec_spec
                    op2 = _with_twisted_stage(op2, stage)
                    out.append(TileOp(op=op2, name=name, place=place, workers=workers, knobs=stamped))
    return out
