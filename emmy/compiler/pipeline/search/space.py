"""The **search space** — every ``Knob`` declaration plus the enumeration value grids, in one file.

**INVARIANT: every ``Knob`` instance is declared here, and nowhere else.**
:mod:`~emmy.compiler.pipeline.knob` owns the ``Knob`` *descriptor* (the dataclass), the registry,
and the env plumbing; :mod:`~emmy.compiler.pipeline.search.features` owns the featurizers; this
module owns the concrete *declarations* AND the candidate-value generators — so the whole tunable
surface (dimensions × values) is visible in one place. A rule that decides a knob imports it from here
(``from emmy.compiler.pipeline.search.space import VECTORIZE_LOADS``) rather than declaring its
own. ``knob.registry()`` still discovers them via ``knob._walk_modules`` (any package module with
module-level ``Knob`` attributes is walked, and this module is imported at pipeline startup by the
rules that consume its knobs). When adding a knob, declare it here and import it into the owning rule.

Scope note: this module holds the **static** space only — the declared dimensions and their bounded
candidate grids. Per-kernel legality (the warp static-K divisibility check, the stage resolvers, coop
eligibility, the ``_COOP_*`` constants) stays with the scheduler in
``passes/lowering/tile/_schedule.py`` — the legal subset is a function of the node.

Three groups:

- **Schedule codec knobs** (``REDUCE`` / ``TILE`` / ``STAGE`` / ``WSPEC`` / ``RASTER``) — the tile-lowering schedule
  fork points that spell the ir schedule codecs (:mod:`emmy.compiler.ir.schedule`). Decided in
  the ``_schedule`` helper inside ``lowering/tile/010_recognize`` and materialized in
  ``lowering/kernel/010_materialize``. Each is the **ephemeral** codec spelling: it resolves into a
  schedule slice (``ReducePlan`` / ``TilePlan`` / ``Stage`` / ``WarpSpec``) and rides on ``TileOp.knobs``
  so the online prior featurizes / tunes the decision. ``off=""`` (the conservative serial / per-cell /
  gmem-direct / uniform default) is auto-stamped on kernels the pass doesn't schedule.
- **The structural placement pin** (``PLACE``) — pin-only: where an intermediate edge lives, registers
  (``fuse``) or memory (``cut``), per edge-class element (``PLACE@<element>`` via ``EMMY_KNOBS``).
- **Kernel-lowering policy knobs** (``VECTORIZE_LOADS`` / ``INTERLEAVE_LOADS``) — boolean codegen
  policies recorded on the kernel op (idempotence + env override), on by default and not search
  dimensions (``hints=(True,)``).
"""

from __future__ import annotations

import logging

from emmy.compiler.ir.schedule import TilePlan
from emmy.compiler.pipeline.knob import Knob, KnobType

logger = logging.getLogger(__name__)

# --- Schedule codec knobs ---------------------------------------------------

# The reduce-axis partition codec. ``off=""`` = the scalar serial fold.
REDUCE = Knob(
    "REDUCE",
    KnobType.STR,
    help="Reduce-axis partition codec (g<n> cta / b<n> coop / r<n> reg; empty=serial). "
    "Decided in lowering/tile/010_recognize (the _schedule helper), materialized in lowering/kernel/010_materialize.",
    off="",
)

# The free-axis output tile — the **unified output-fragment** knob. A contraction's output tile is
# *either* the scalar register sub-tile (``n<N>[x<M>]`` parallel thread-tile / ``f<fn>[x<fm>]``
# register sub-tile) *or* the tensor-core warp mma tile (``a:<atom>/w<WM>x<WN>/f<FM>x<FN>/k<bk>``),
# never both. The value self-discriminates: an ``a:<atom>`` token selects the warp fragment (see
# ``schedule.is_warp_codec``); otherwise the scalar fragment. Only a ``CONTRACTION`` tiles its output
# today; ``off=""`` auto-stamps everything else. The codec is the sole on-dict spelling — the
# online-prior featurizer (``features.mma_atom`` / ``is_warp`` / ``_free_slots`` / ``tile_signature``)
# parses it directly (no legacy ``WM``/``WN``/``MMA`` keys).
TILE = Knob(
    "TILE",
    KnobType.STR,
    help="Output-fragment codec — scalar tile (n<N>[x<M>]/f<fn>[x<fm>]) OR warp mma tile "
    "(a:<atom>/w<WM>x<WN>/f<FM>x<FN>/k<bk>, selected by the a:<atom> token); empty=per-cell. "
    "Decided in lowering/tile/010_recognize (the _schedule helper), materialized in lowering/kernel/010_materialize.",
    off="",
)

# Operand staging — the reused gmem operands (matmul A/B, a fused prologue's read) ride a
# shared-memory slab + double-buffered producer (``sync`` plain copy / ``cp.async`` / ``tma``) over
# the serial reduce loop, instead of the gmem-direct register baseline. Resolved into the schedule's
# :class:`Stage` (``None`` = gmem-direct). Composes with both fragments of the unified ``TILE`` knob.
STAGE = Knob(
    "STAGE",
    KnobType.STR,
    help="Operand-staging codec (d<depth>/sync|cp|tma[/ring][/alt][/p<reg_depth>]; empty=gmem-direct). "
    "Decided in lowering/tile/010_recognize (the _schedule helper), materialized in lowering/kernel/010_materialize.",
    off="",
)

# Warp specialization — the worker-mapping sibling of ``REDUCE``/``TILE``/``STAGE`` and ORTHOGONAL to
# all three: the pipeline (what's staged, the mma tile, the reduce partition) is fixed by those pins;
# ``WSPEC`` only splits the warps that run it into roles (``p<np>`` producer warps drive the ``STAGE``
# load half; the compute warps stay on the mma). ``off=""`` is uniform SIMT (every warp does both
# halves). Resolved into the schedule's :class:`WarpSpec` (``None`` = uniform) and gated on a warp
# ``TILE`` + a resolved **TMA** ``STAGE`` (the producer band drives the box-copy mbarrier ring;
# cp.async's wait-group is issuing-thread-scoped and a sync compute-fill has no async load half).


def _wspec_features(val) -> dict[str, float]:
    """The ``WSPEC`` sub-features for the online prior — the dedicated (non-COMPUTE) warp count
    (``0.0`` = uniform SIMT). The producer ``q`` window is reserved (inert) and not featurized."""
    if not val:
        return {"D_wspec_warps": 0.0}
    from emmy.compiler.ir.schedule import WarpSpec  # noqa: PLC0415 — schedule imports this module's knobs' consumers

    try:
        ws = WarpSpec.parse(str(val))
    except ValueError:
        return {"D_wspec_warps": 0.0}
    return {"D_wspec_warps": float(ws.aux_warps)}


WSPEC = Knob(
    "WSPEC",
    KnobType.STR,
    help="Warp-specialization codec — role→warp split over the fixed pipeline "
    "(p<np> producer[:q<window>, reserved], s<ns> sfu, …; compute warps implicit = TilePlan.units; empty=uniform SIMT). "
    "Decided in lowering/tile/010_recognize (the _schedule helper), materialized in lowering/kernel/010_materialize "
    "(the staged K-loop's producer/compute band split; warp TILE + TMA STAGE only).",
    features=_wspec_features,
    off="",
)


def wspec_moves() -> list[str]:
    """The warp-specialization ``WSPEC`` codec candidates — uniform ``""`` first (the conservative
    option-0), then the producer-band splits. Per-row legality (a warp tile over a resolved TMA
    stage, the ``block_threads + 32·aux ≤ 1024`` and ``32·aux ≤ block_threads`` thread budgets) is
    the scheduler's (``_schedule._wspec_candidates`` / ``_wspec_workers``)."""
    return ["", "p1", "p2"]


def _raster_features(val) -> dict[str, float]:
    """The ``RASTER`` sub-features for the priors — the stripe group size (``0.0`` = the flat
    N-fastest order) and the orientation flag (``1.0`` = ``gn``, the transposed grouping)."""
    if not val:
        return {"D_raster_group": 0.0, "D_raster_gn": 0.0}
    from emmy.compiler.ir.schedule import Raster  # noqa: PLC0415 — same deferred pattern as _wspec_features

    try:
        r = Raster.parse(str(val))
    except ValueError:
        return {"D_raster_group": 0.0, "D_raster_gn": 0.0}
    if r is None:
        return {"D_raster_group": 0.0, "D_raster_gn": 0.0}
    return {"D_raster_group": float(r.group), "D_raster_gn": 1.0 if r.orient == "n" else 0.0}


RASTER = Knob(
    "RASTER",
    KnobType.STR,
    help="CTA rasterization codec — the launch-order mapping of flat CTA ids onto the 2-D "
    "(m, n) block-tile grid (gm<G>: G M block-tiles iterate fastest per stripe, L2 reuse of the "
    "streamed B operand; gn<G>: the transpose, A streamed; empty = flat N-fastest row-major). "
    "Kernel-scoped like WSPEC (no @<axis> key); changes no per-CTA work or layout, only the "
    "block-id decode. Decided in lowering/tile/010_recognize (the _schedule row product), applied "
    "at the kernel materializer's grid_tile seal; 2-D-tiled contraction grids only.",
    features=_raster_features,
    off="",
)


def raster_moves() -> list[str]:
    """The ``RASTER`` codec candidates — the flat order ``""`` first (the conservative option-0,
    byte-identical to the historical codegen), then the CUTLASS/Triton-conventional grouped-M
    stripe of 8. Wall-time effect is shape-dependent and small (±2–4% measured on sm_89:
    qkv −4%, gate_up fm +2.4%, most shapes neutral) while DRAM traffic on wide-N shapes halves
    (503.6 → 261.6 MB on mlp_gate_up, the theoretical floor) — so the family is enumerated for
    the search/goldens to arbitrate per shape, never a blanket policy. ``gn<G>`` spellings are
    pin-only until a shape wants them."""
    return ["", "gm8"]


def map_tile_moves() -> list[str]:
    """The pointwise-map register-strip candidates — spelled in the **scalar ``TILE`` codec** (the same
    ``f<fn>`` register sub-tile a contraction's output rides, here with no ``n`` unit-tile / atom since
    the grid already parallelizes a pure ``Map``). ``f<r>`` hands each thread ``r`` **contiguous**
    inner-axis elements (blocked: thread t owns ``[t·r, t·r+r)``) as ``r`` grouped loads + ``r`` grouped
    writes, which ``050_vectorize_loads`` / ``080_vectorize_stores`` merge into one ``float<r>`` access
    — matching torch's ``vectorized_elementwise_kernel<r>``. Option-0 (``""``, 1 elem/thread) leads.
    Legality (a static inner free axis divisible by r) is the scheduler's (``_schedule._map_strip_fork``).
    The ladder stops at ``f4``: ``f8`` regressed both pointwise goldens (register pressure — 22 vs 14
    regs — outweighs the wider access), so it's left out until a shape wants it."""
    return ["f2", "f4"]


# --- The structural placement pin (PLACE) -----------------------------------
#
# ONE pin-only family controlling structural emission: where an intermediate edge lives — registers
# (``fuse``) or memory (``cut``). Elements are named by the MOVE, not the shape:
#
#   PLACE@cone   producer-cone inlining (the fused producer → matmul edge). ``cut`` is realized
#                by ``lowering/tile/020_cut_edge`` — a ``030_split_reduce``-style graph rewrite that
#                splits the recognized cone kernel at the A seam (the producer materializes the
#                cone value to a workspace; the matmul re-lowers with a plain gmem A) — and both
#                halves re-enter recognition on the pass-scan restart, so each gets its own
#                schedule / fork. The recognizer's nodification gate is the ``fuse`` half.
#   PLACE@fold   downstream-fold absorption (flash vs separate softmax + P@V kernels)
#   PLACE@tuple  sibling-fold tupling (online softmax vs two-pass stats)
#
# Vocabulary: ``auto`` (the built-in per-element default) / ``fuse`` / ``cut``. Precedence:
# ``PLACE@<element>`` > bare ``PLACE`` > built-in ``auto`` (read via :func:`place_decision` /
# ``Knob.narrow_at``). ``auto`` never appears in a knob dict — it is pin vocabulary; the stamped
# value is the *resolved* decision (``fuse`` / ``cut``), stamped for ``fold`` / ``cone`` only
# (``tuple`` is pure policy — dominance — and is never stamped). ``fuse`` is a request, not a
# guarantee: a forced fuse on an uncertifiable kernel (e.g. RoPE'd QK, which flash recognition
# rejects) degrades to ``cut`` with a log line — the standard pin-validity rule. Since ``@`` is not
# a valid shell var name character, per-element pins ride the ``EMMY_KNOBS`` aggregate
# (``EMMY_KNOBS="PLACE@fold=cut"``); the bare ``EMMY_PLACE`` env var pins every element.
PLACE = Knob(
    "PLACE",
    KnobType.STR,
    help="Structural placement of an intermediate edge — auto|fuse|cut, per element via "
    "PLACE@cone (producer-cone inlining) / PLACE@fold (flash vs multi-kernel attention) / "
    "PLACE@tuple (online softmax vs two-pass stats) / PLACE@stat (auto|fuse|sink — a norm's "
    "row statistic staying local vs sunk into its producer's epilogue); bare PLACE pins every "
    "eligible edge. Pin-only (never enumerated); read in lowering/tile/010_recognize.",
    off="",
)

# The built-in ``auto`` defaults per element — today's emission behavior (fuse everywhere: flash,
# online softmax, and producer-cone inlining are all on when recognizable; a row statistic stays
# LOCAL to its kernel). Flipping a default is a behavior change gated on the validation suite, not
# a spelling change. ``stat``'s alternative is ``sink`` (not ``cut``): the statistic reduce
# migrates into the producer kernel's epilogue (``025_sink_row_reduce``) instead of splitting out.
_PLACE_DEFAULTS = {"cone": "fuse", "fold": "fuse", "tuple": "fuse", "stat": "fuse"}
_PLACE_VALUES = {"cone": ("fuse", "cut"), "fold": ("fuse", "cut"), "tuple": ("fuse", "cut"), "stat": ("fuse", "sink")}


def place_decision(element: str) -> str:
    """The resolved ``PLACE`` decision (``"fuse"`` / ``"cut"`` — ``"sink"`` for ``stat``) for
    ``element`` — the pin (``PLACE@<element>`` > bare ``PLACE``, via ``Knob.narrow_at``) with the
    explicit ``auto`` token (and no pin at all) resolving to the built-in per-element default. An
    unknown pin value degrades to the default with a log line — the standard pin-validity rule."""
    default = _PLACE_DEFAULTS[element]
    allowed = _PLACE_VALUES[element]
    pin = PLACE.narrow_at(element)
    if pin is None or pin in ("", "auto"):
        return default
    if pin in allowed:
        return pin
    logger.warning("PLACE@%s pin %r is not auto|%s; using the built-in %r", element, pin, "|".join(allowed), default)
    return default


# --- Kernel-lowering policy knobs -------------------------------------------
#
# Boolean codegen policies recorded on the kernel op — on by default, not search dimensions
# (``hints=(True,)``); a rule records its knob for idempotence and honors the ``EMMY_<NAME>``
# env override. Consumed by ``lowering/kernel/050_vectorize_loads`` / ``095_interleave_loads``.

VECTORIZE_LOADS = Knob(
    "VECTORIZE_LOADS",
    KnobType.BOOL,
    hints=(True,),  # on by default; not a search dimension — manual override only via the env var
    help="Fold runs of consecutive scalar Loads into one wide vector Load (float4 / __half2).",
    off=False,
)

INTERLEAVE_LOADS = Knob(
    "INTERLEAVE_LOADS",
    KnobType.BOOL,
    hints=(True,),  # on by default; not a search dimension — manual override only via the env var
    help="Sink each Load to just before its first SSA-consumer in flat compute blocks.",
    off=False,
)

FAST_EXP = Knob(
    "FAST_EXP",
    KnobType.BOOL,
    # Off by default and not a search dimension — a precision-trading knob (__expf ≈ 2 ulp vs
    # correctly-rounded expf; numerically benign for the softmax family — the α rescale never
    # amplifies and the carrier stays fp32 — but it must be a deliberate, pinnable choice, never
    # a silent default). Enabled via EMMY_FAST_EXP=1 or the FAST_MATH umbrella.
    hints=(False,),
    help="Lower f32 exp through the SFU fast path (__expf: one FMUL + MUFU.EX2) instead of libm expf.",
    off=False,
)


# --- Precision-trading knobs (the FAST_MATH family) ---------------------------
#
# Knobs that trade numerical precision for throughput are NEVER silently on: each is off by
# default and enabled by its own ``EMMY_<NAME>`` pin, or batch-enabled by the ``FAST_MATH``
# umbrella (the ``-use_fast_math`` / ``-O3`` analogue). Precedence per knob: its own pin >
# ``FAST_MATH`` > off (:func:`precision_pin`). The umbrella is a meta gate, not a kernel
# property — the realized fork is already fully identified by what it enables (``FAST_EXP``'s
# stamped BOOL, the ``TILE`` codec's ``a:<atom>`` token) — so it is ``unfeatured`` and never
# stamped, enumerated, or featurized.

FAST_MATH = Knob(
    "FAST_MATH",
    KnobType.BOOL,
    hints=(False,),
    help="Umbrella pin for the precision-trading knobs (FAST_EXP, F16_MMA_F32_ACC): "
    "EMMY_FAST_MATH=1 enables each one not individually pinned; individual pins win.",
    unfeatured=True,  # a meta gate over other knobs — must never enter the feature vector
)

F16_MMA_F32_ACC = Knob(
    "F16_MMA_F32_ACC",
    KnobType.BOOL,
    hints=(False,),
    help="Offer the f16-mma / chunked-f32-accumulate atom forks (a:mma_m16n8k16_f16_f16 — the mma "
    "chain accumulates in f16 at the full HMMA rate, with a periodic register promote into f32 "
    "shadows; ~2x mma throughput on consumer dies where f32-accumulate is half rate). "
    "Pin 1 to offer on every target, 0 never; unset follows FAST_MATH (consumer-die targets only). "
    "Enumeration-gate only — the realized fork is identified by the TILE codec's atom token.",
)


def precision_pin(knob: Knob) -> bool | None:
    """The effective pin for a precision-trading BOOL ``knob``: its own ``EMMY_<NAME>`` pin when
    set, else the ``FAST_MATH`` umbrella pin, else ``None`` (neither set — the caller applies its
    conservative default, and may keep target gates that an *individual* pin overrides)."""
    raw = knob.raw()
    if raw is not None:
        return knob.parse(raw)
    raw = FAST_MATH.raw()
    if raw is not None:
        return FAST_MATH.parse(raw)
    return None


LOOPIFY = Knob(
    "LOOPIFY",
    KnobType.INT,
    # Off by default (0) and not a search dimension — a readability-only codegen policy: re-roll a
    # maximal run of parallel per-fragment ``FragmentApply`` (the flash mma epilogue's element-wise
    # C-fragment arithmetic) into a ``#pragma unroll`` loop over an arrayed fragment family. The value
    # is the MINIMUM run length to re-roll: ``0`` / unset (and any value < 2) → off, byte-identical
    # CUDA; ``EMMY_LOOPIFY=4`` re-rolls the 8-long O rescale / divide runs (most of the win) while
    # skipping the 2-long QK scale; ``EMMY_LOOPIFY=2`` re-rolls every run ≥ 2. Identical SASS (nvcc
    # unrolls the pragma) — purely a listing-shrink for blog / ``--ir cuda`` inspection.
    hints=(0,),
    help="Min parallel FragmentApply run length to re-roll into a #pragma-unroll loop (0 = off, byte-identical).",
    off=0,
    unfeatured=True,  # SASS-identical listing re-spell — excluded from the feature vector; batch-enabled by EMMY_READABLE
)


# --- Enumeration value grids -------------------------------------------------
#
# The permitted-move catalog: the bounded, legality-guarded candidate values the ``_schedule`` emit
# enumerates into the scheduling fork. Each move is a codec spelling under the node's axis-named key
# (``TILE@<k_axis>`` / ``REDUCE@<axis>`` / ``STAGE@<axis>``) that the existing ``parse`` / ``spell``
# grammar, the prior featurizer, and the perf DB already consume — the move is only the **generator**,
# not new syntax. Two invariants keep a cold greedy compile stable and correct:
#
# - **Conservative option-0.** The per-cell / serial / gmem-direct pick leads every list (the reduce
#   tier deliberately emits its conservative *cooperative* pick first — the option-0 rule is
#   per-family, naming that family's safe default), so the emission-order fallback (no prior loaded)
#   keeps today's behavior.
# - **Static-value legality only.** Guards evaluable from the values alone (the scalar block-thread
#   budget) apply here; per-node guards (warp static-K divisibility, stage resolver eligibility) live
#   with their moves in ``_schedule``. An env pin still wins via ``Knob.narrow`` at the call site —
#   the catalog is the *unpinned* candidate set.

# The scalar block-thread budget (CUDA's 1024-thread/CTA hardware limit); a scalar tile launches
# ``par_n·par_m`` threads (one per parallel output cell). The same limit ``_schedule`` enforces on a
# pinned tile (imported there — one constant, two enforcement points).
MAX_BLOCK_THREADS = 1024

# The scalar register-tile candidate grid: ``(par_n, par_m)`` parallel thread-tile widths ×
# ``(reg_n, reg_m)`` per-thread register sub-tile widths. Bounded and hand-computable — the product the
# structural-coverage test recomputes independently. The parallel widths stay inside the thread budget
# (``64·16 = 1024 ≤ 1024``); the register widths span the square + skewed sub-tiles the prior ranks by
# occupancy / reuse PLUS the golden-informed deep-FM points — every ``(reg_n, reg_m)`` here is a
# recorded golden winner on some card (the ``f2x14`` / ``f4x8`` / ``f4x10`` / ``f4x26`` family that the
# post-rebuild grid orphaned — the sixth sweep's 1.29-1.49× reachability losses). The permanence test
# (``tests/compiler/test_golden_configs.py``) asserts every golden TILE stays a member of this product.
_SCALAR_PAR: tuple[tuple[int, int], ...] = ((16, 8), (16, 16), (32, 8), (32, 16), (64, 16))  # (par_n, par_m)
_SCALAR_REG: tuple[tuple[int, int], ...] = (
    (1, 1), (2, 2), (4, 4), (2, 4), (4, 2),  # the square / skewed core
    (2, 6), (2, 8), (2, 14),  # golden-informed deep-FM, narrow-par rows
    (4, 6), (4, 8), (4, 10), (4, 12), (4, 14), (4, 26),  # golden-informed deep-FM, wide rows
)  # fmt: skip  # (reg_n, reg_m)


def scalar_tile_moves() -> list[str]:
    """The scalar-contraction output-tile ``TILE`` codec candidates: per-cell (``""``) first — the
    conservative option-0 — then the register-tile grid (:data:`_SCALAR_PAR` × :data:`_SCALAR_REG`)
    filtered by the ``par_n·par_m ≤ 1024`` thread budget. Each is spelled through :class:`TilePlan`
    so it round-trips the codec grammar exactly."""
    moves = [""]
    for par in _SCALAR_PAR:
        if par[0] * par[1] > MAX_BLOCK_THREADS:
            continue
        for reg in _SCALAR_REG:
            moves.append(TilePlan(units=par, regs=reg).spell())
    return moves


# The warp (tensor-core) tile candidate grid: ``(WM, WN)`` warp counts × ``(FM, FN)`` per-warp
# register fragments × ``bk`` K-chunks, spelled ``a:<atom>/w..x../f..x../k..``. Bounded to shapes the
# golden sweeps have deployed (``FM·FN ≤ 32`` C-fragment cells, shallow pipelined bk; ``(8, 2)`` and
# ``(2, 8)`` are recorded golden winners on the RTX 4090 / PRO 6000 — the permanence test keeps them).
# ``(1, 16)`` is the thin-M / wide-N decode geometry (16 warps down the N axis, 1 down M — the same
# 16-warp CTA as ``(8, 2)``): a decode-M computed-A (fused norm→linear) contraction wants its warps
# spread across the wide output columns, and it beat the ``(1, 8)`` sibling ~5% on both the q-proj
# (N=4096) and gate/up (N=15360) fused edges at M=32 (5090). ``(2, 8)`` is its M=64 sibling (a second
# M warp-unit once the decode bucket doubles): the lm_head.m64 golden winner — w1x8 leaves 2x on the
# table there (2392 vs 1215 µs, 5090) — and the best fused-geglu tile at the same M. Per-node legality — the atom's operand
# dtype and the ``_check_warp_static_k`` K-divisibility — is the scheduler's (``_schedule``), not the grid's.
# (WM, WN) / (FM, FN)
_WARP_UNITS: tuple[tuple[int, int], ...] = (
    (1, 1),
    (2, 1),
    (1, 2),
    (2, 2),
    (4, 1),
    (1, 4),
    (2, 4),
    (4, 2),
    (4, 4),
    (1, 8),
    (2, 8),
    (8, 2),
    (1, 16),
)
_WARP_REGS: tuple[tuple[int, int], ...] = ((1, 1), (2, 2), (1, 4), (4, 1), (2, 4), (4, 2), (4, 4), (4, 8), (2, 8))
_WARP_BK: tuple[int, ...] = (1, 2, 4, 8)


def warp_tile_moves(atom_names: tuple[str, ...]) -> list[str]:
    """The warp-contraction output-tile ``TILE`` codec candidates over the (already
    dtype-eligible) ``atom_names``: the :data:`_WARP_UNITS` × :data:`_WARP_REGS` × :data:`_WARP_BK`
    grid per atom. No conservative option-0 of its own — these EXTEND :func:`scalar_tile_moves`
    (whose per-cell ``""`` leads the combined list)."""
    from emmy.compiler.ir.atom import ATOM_REGISTRY  # noqa: PLC0415

    moves = []
    for name in atom_names:
        atom = ATOM_REGISTRY[name]
        for units in _WARP_UNITS:
            for regs in _WARP_REGS:
                for bk in _WARP_BK:
                    moves.append(TilePlan(atom=atom, units=units, regs=regs, bk=bk).spell())
    return moves


# The warp-flash (fragment-resident TWISTED) tile grid: warps per CTA over the query rows ×
# score n-atoms per streaming key block. The conservative pair leads (one warp, the ``2·atom_n``
# key block — today's deterministic stamp), so a cold tie keeps the historical geometry. Per-node
# legality (kv / query-row divisibility, the dtype atom) is the scheduler's
# (``_schedule._twisted_warp_options``), not the grid's.
_FLASH_WARPS: tuple[int, ...] = (1, 2, 4)  # warps per CTA, each owning its own query-row block
_FLASH_KEY_ATOMS: tuple[int, ...] = (2, 4, 8, 16)  # score n-atoms per streaming block (bn = n·atom_n keys)
_FLASH_QTILES: tuple[int, ...] = (1, 2)  # register query tiles per warp (reg_m — FA-2's in-flight ILP)


def twisted_warp_moves() -> list[tuple[int, int, int]]:
    """The warp-flash geometry candidates ``(warps_m, key_atoms, q_tiles)`` — the
    :data:`_FLASH_WARPS` × :data:`_FLASH_KEY_ATOMS` × :data:`_FLASH_QTILES` grid, conservative
    option-0 ``(1, 2, 1)`` first. ``q_tiles`` is the register query-tile count per warp (the
    ``TILE`` codec's ``f<FM>x<FN>`` reg_m): each warp streams ``q_tiles`` independent ``(m, l, O)``
    chains against shared K/V fragments — FA-2's in-flight ILP, hiding the per-step
    mma → rowmax → exp → rescale dependency chain without more warps. Each triple resolves into
    the Q@K / P@V mma :class:`TilePlan`\\ s in ``_schedule._twisted_warp_options`` (the ``TILE``
    codec spells the full plan; this grid only generates the free geometry — ``bk`` is shape-derived)."""
    return [(um, nt, fm) for fm in _FLASH_QTILES for um in _FLASH_WARPS for nt in _FLASH_KEY_ATOMS]


def stage_moves(*, warp: bool) -> list[str]:
    """The operand-staging ``STAGE`` codec candidates — gmem-direct ``""`` first (the conservative
    option-0), then the transport / depth / double-buffer variants. Both tiers offer the gmem→smem
    prefetch ring depths (the scalar ring lands on the same ``staged_kloop`` phases; its slab
    K-chunk is depth-aware, derived in ``_resolve_scalar_stage``); the ``p2`` smem→register
    double-buffer is an ``ldmatrix`` transform, warp-only. Emission is resolver-gated in
    ``_schedule`` — a candidate is offered only when it RESOLVES against the built node, and the
    row carries the resolved spelling."""
    ring = ["", "d1/cp", "d2/cp/ring", "d3/cp/ring", "d4/cp/ring", "d1/tma", "d2/tma/ring", "d3/tma/ring", "d4/tma/ring"]
    return [*ring, "d2/cp/ring/p2"] if warp else ring


# Cross-CTA split-K widths (the ``REDUCE`` codec's ``g<w>`` field). Divisor / occupancy legality is
# the scheduler's.
SPLITK_WIDTHS: tuple[int, ...] = (2, 4, 8)


def splitk_moves(*, warp: bool) -> list[str]:
    """The cross-CTA split-K ``REDUCE`` codec candidates, both tiers each: the deferred-kernel
    finalize (``g<w>k``, an f32 workspace + sibling combine kernel) and the in-place atomic
    (``g<w>a``, one kernel — the partial ``atomicAdd``\\ s into the zero-init'd output; the mma
    tier rides ``RegStore.atomic``'s packed-pair red). The scheduler's ``atomic_ok`` gate
    (``_reduce_candidates``) keeps ``a`` rows off multi-fold / non-distributive-projection nodes.
    These EXTEND the serial ``""`` option-0."""
    del warp  # both tiers share the catalog; per-node legality lives in the scheduler's gates
    return [f"g{w}{f}" for w in SPLITK_WIDTHS for f in ("k", "a")]


def coop_reduce_moves() -> list[str]:
    """The cooperative / ILP K-partition ``REDUCE`` codec candidates for a NON-output-tiled
    contraction (``_coop_reduce_spec``'s contract — the per-cell tier folds K across ``b`` coop
    threads / ``r`` ILP register chains). These EXTEND the serial ``""`` option-0. ``b16`` /
    ``b32`` are recorded reduce-golden winners (the wide-row coop folds) — kept enumerable so
    the reduce goldens stay reachable. The wide ``b64``–``b512`` folds are the memory-bound
    normalizer band: a wide-K softmax / rms_norm saturates bandwidth only with a full-block coop
    row (``softmax.k2048`` wants ``b512`` — 2.6× over ``b32``). The scheduler's ``_coop_reduce_spec``
    declines a ``b<n>`` wider than the row has work for, so enumerating them is safe on small K.
    The ``b<n>t`` transposed band is the k-major-B matvec partition (warp lanes sweep the output
    axis — the M=1 gemv tier's coalescing fix); ``_reduce_candidates`` gates it structurally
    (plain contraction, 32-divisible inner free axis) and MEASUREMENT decides the layout — a
    row-major-B shape simply benches it slower."""
    return ["b4", "b8", "b16", "b32", "b64", "b128", "b256", "b512", "r2", "r4", "r2/b4", "b32t", "b64t", "b128t", "b256t"]
