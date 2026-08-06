"""The **search space** — every ``Knob`` declaration plus the enumeration value grids, in one file.

**INVARIANT: every ``Knob`` instance is declared here, and nowhere else.**
:mod:`~emmy.compiler.pipeline.knob` owns the ``Knob`` *descriptor* (the dataclass), the registry,
and the env plumbing; :mod:`~emmy.compiler.pipeline.search.features` owns the featurizers; this
module owns the concrete *declarations* AND the candidate-value generators — so the whole tunable
surface (dimensions × values) is visible in one place. A rule that decides a knob imports it from here
(``from emmy.compiler.pipeline.search.space import VECTORIZE_LOADS``) rather than declaring its
own. Registration is construction (``Knob.__post_init__``), and ``knob.registry()`` imports this
module before answering — so the registry is complete in any process that can ask, however little
of the pipeline it loaded. When adding a knob, declare it here and import it into the owning rule.

Scope note: this module holds the **static** space only — the declared dimensions and their bounded
candidate grids. Per-kernel legality (the warp static-K divisibility check, the stage resolvers, coop
eligibility, the ``_COOP_*`` constants) stays with the tile scheduler — the legal subset is a
function of the node.

The ``TILE`` / ``REDUCE`` grids hand out the **typed schedule slices** themselves
(:class:`~emmy.compiler.ir.schedule.TilePlan` / :class:`~emmy.compiler.ir.schedule.ReducePlan`,
built structurally — never a parsed literal), so the enumeration never speaks a codec spelling; the
scheduler spells each row ONCE, site-local, where it becomes stored state. ``STAGE`` hands out typed
:class:`~emmy.compiler.ir.schedule.Stage` slices in the same currency; ``RASTER`` stays a codec string
(it has no slice type of its own).

Two groups:

- **Schedule codec knobs** (``WORK`` / ``REDUCE`` / ``TILE`` / ``STAGE`` / ``RASTER``) — the tile-lowering schedule
  fork points that spell the ir schedule codecs (:mod:`emmy.compiler.ir.schedule`). Decided by the
  tile schedule and materialized in ``lowering/kernel/010_materialize``. Each is the **ephemeral** codec spelling: it resolves into a
  schedule slice (``ReducePlan`` / ``TilePlan`` / ``Stage`` / ``WarpSpec``) and rides on ``TileOp.knobs``
  so the online prior featurizes / tunes the decision. ``off=""`` (the conservative serial / per-cell /
  gmem-direct / uniform default) is auto-stamped on kernels the pass doesn't schedule.
- **Kernel-lowering policy knobs** (``VECTORIZE_LOADS`` / ``INTERLEAVE_LOADS``) — boolean codegen
  policies recorded on the kernel op (idempotence + env override), on by default and not search
  dimensions (``hints=(True,)``).
"""

from __future__ import annotations

import logging

from emmy.compiler.ir.schedule import ReducePlan, Stage, TilePlan
from emmy.compiler.pipeline.knob import Knob, KnobType
from emmy.compiler.pipeline.search.domain import Bound, Dimension, Space

logger = logging.getLogger(__name__)

# --- Schedule codec knobs ---------------------------------------------------

# The reduce-axis partition codec. ``off=""`` = the scalar serial fold.
REDUCE = Knob(
    "REDUCE",
    KnobType.STR,
    help="Reduce-axis partition codec, site-local (g<n>[a|k] cta / coop[-t] / r<n> reg; empty=serial; "
    "the coop WIDTH lives in WORK). "
    "Decided in the tile schedule, materialized in lowering/kernel/010_materialize.",
    off="",
)

# The free-axis output tile — the **unified output-fragment** knob. A contraction's output tile is
# *either* the scalar register sub-tile (``f<fn>[x<fm>]``) *or* the tensor-core warp mma tile
# (``<atom>/f<FM>x<FN>[/k<bk>]``), never both. The value is SITE-LOCAL — the unit widths live in
# ``WORK``, and the tier discriminator IS the worker kind, with the leading atom token naming the
# fragment. Only a ``CONTRACTION`` tiles its output today; ``off=""`` auto-stamps everything else.
# The codec is the sole on-dict spelling — the online-prior featurizer (``features.mma_atom`` /
# ``is_warp`` / ``_free_slots`` / ``tile_signature``) resolves it against the row's ``WORK``.
TILE = Knob(
    "TILE",
    KnobType.STR,
    help="Output-fragment codec, site-local — scalar tile (f<fn>[x<fm>]) OR warp mma tile "
    "(<atom>/f<FM>x<FN>[/k<bk>]); empty=per-cell, the worker widths live in WORK. "
    "Decided in the tile schedule, materialized in lowering/kernel/010_materialize.",
    off="",
)

# Operand staging — the reused gmem operands (matmul A/B, a fused prologue's read) ride a
# shared-memory slab + double-buffered producer (``sync`` plain copy / ``cp.async`` / ``tma``) over
# the serial reduce loop, instead of the gmem-direct register baseline. Resolved into the schedule's
# :class:`Stage` (``None`` = gmem-direct). Composes with both fragments of the unified ``TILE`` knob.
STAGE = Knob(
    "STAGE",
    KnobType.STR,
    help="Operand-staging codec (d<depth>/sync|cp|tma[/split][/p<reg_depth>]; empty=gmem-direct). "
    "Decided in the tile schedule, materialized in lowering/kernel/010_materialize.",
    off="",
)

# The kernel-global worker inventory (the step-7 value-grammar family): the w/n worker tokens
# factored out of the per-site TILE values, the coop width out of REDUCE, and the producer band the
# retired WSPEC family used to spell, absorbed as ``+p<n>``. Stamped by ``ops.seal_workers`` on every
# assembled option row; ``off=""``
# = the per-cell / pure-reduce forms' derived launch geometry.


def _work_features(val) -> dict[str, float]:
    """The ``WORK`` sub-features for the online prior — ONLY the ``+p`` producer band, as the
    same ``D_wspec_warps`` the retired per-row ``WSPEC`` key spelled (name/semantics preserved;
    a legacy row's ``WSPEC`` key writes the identical value). The inventory's tile geometry is
    NOT re-featurized here — the per-node featurizers already fold it in by resolving the site
    ``TILE``/``REDUCE`` values against WORK (``features._tile_plan`` / ``_reduce_decomp``)."""
    if not val:
        return {"D_wspec_warps": 0.0}
    from emmy.compiler.ir.schedule import Workers  # noqa: PLC0415 — deferred: schedule imports this module

    try:
        w = Workers.parse(str(val))
    except ValueError:
        return {"D_wspec_warps": 0.0}
    return {"D_wspec_warps": float(w.producer if w is not None else 0)}


WORK = Knob(
    "WORK",
    KnobType.STR,
    help="Kernel-global worker inventory (w<M>x<N>[+p<np>] warps / t<N>[x<M>] threads; empty=derived per-cell geometry). "
    "The step-7 value-grammar family — TILE/REDUCE values become site-local; the tier discriminator IS the worker kind.",
    features=_work_features,
    off="",
)


def _raster_features(val) -> dict[str, float]:
    """The ``RASTER`` sub-features for the priors — the stripe group size (``0.0`` = the flat
    N-fastest order) and the orientation flag (``1.0`` = ``gn``, the transposed grouping)."""
    if not val:
        return {"D_raster_group": 0.0, "D_raster_gn": 0.0}
    from emmy.compiler.ir.schedule import Raster  # noqa: PLC0415 — deferred: schedule imports this module

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
    "Kernel-scoped (no @<axis> key); changes no per-CTA work or layout, only the "
    "block-id decode. Decided by the tile schedule (the row product), applied "
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

#: The unroll budget — the max static loop trip count eligible for ``#pragma unroll``. Pin-only
#: (``off`` defaults to ``_UNSET`` → never stamped / featurized / enumerated, so it can't perturb the
#: search or the goldens): ``EMMY_UNROLL=0`` keeps every extent-driven loop **rolled** (compact,
#: readable kernels — e.g. for a blog listing), a high value unrolls more. Unset → each call site's
#: built-in cap (64 for an inner reduce, 128 for the flash KV fold, uncapped for the tensor-core
#: K-chunk), so the default codegen is byte-identical. Consumed by ``lowering/kernel/_atom.unroll_ok``.
UNROLL = Knob(
    "UNROLL",
    KnobType.INT,
    help="Max static loop trip count to #pragma-unroll (the unroll budget); pin 0 to keep loops rolled. Unset = per-site cap.",
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
    help="Umbrella pin for the precision-trading knobs (FAST_EXP, F16_MMA_F32_ACC, FP8_MMA): "
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


FP8_MMA = Knob(
    "FP8_MMA",
    KnobType.BOOL,
    hints=(False,),
    help="Offer the native fp8 tensor-core atom forks (a:mma_m16n8k32_e4m3_f32 / _e5m2_f32 — both "
    "multiplicands consumed as raw f8 bytes at k32, scale factors on the f32 epilogue). The "
    "instruction's effective accumulation precision is arch-dependent (reduced on sm_89, ~3e-4 rel "
    "vs the exact f32 decode-and-fma scalar path), so the fork family is precision-trading: pin 1 "
    "to offer, 0 never; unset follows FAST_MATH. The sm_89 hardware floor is absolute — below it "
    "the instruction does not compile, and no pin overrides that. "
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
# enumerates into the scheduling fork. A move is the **typed schedule slice** itself — a
# :class:`TilePlan` / :class:`ReducePlan` built structurally, never a parsed literal — so the
# enumeration never speaks a codec spelling: ``_schedule`` spells each row ONCE, site-local, at the
# boundary where it becomes stored state. Two invariants keep a cold greedy compile stable and correct:
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

# The lanes one warp spends of that budget — the coefficient a warp-tile bound multiplies by.
WARP_LANES = 32

# The mma C fragment's per-warp register budget, in accumulator cells (``FM·FN``). Above this the
# fragment no longer fits the register file at any useful occupancy.
MAX_FRAGMENT_CELLS = 32

# The scalar register-tile candidate SPACE: ``(par_n, par_m)`` parallel thread-tile widths ×
# ``(reg_n, reg_m)`` per-thread register sub-tile widths, generated from the one constraint that
# bounds it — the CTA thread budget, which a scalar tile spends ``par_n·par_m`` of. The register
# widths carry no budget (the register file is unmodeled everywhere in scheduling), so their
# dimensions are the value ladders themselves: the square / skewed core plus the deep-FM points
# (``f2x14`` / ``f4x8`` / ``f4x10`` / ``f4x26``) that are recorded golden winners.
#
# ``tests/compiler/test_golden_configs.py`` asserts every golden TILE stays a member of this space;
# ``test_move_catalog.py`` recomputes the product independently.
_SCALAR_TILE_SPACE = Space(
    dims=(
        Dimension("par_n", (16, 32, 64)),
        Dimension("par_m", (8, 16)),
        Dimension("reg_n", (1, 2, 4)),
        Dimension("reg_m", (1, 2, 4, 6, 8, 10, 12, 14, 26)),
    ),
    bounds=(Bound(("par_n", "par_m"), limit=MAX_BLOCK_THREADS),),
)


def scalar_tile_moves() -> list[TilePlan]:
    """The scalar-contraction output-tile candidates: the per-cell tile first — the conservative
    option-0 — then :data:`_SCALAR_TILE_SPACE`'s points, parallel widths varying slowest."""
    moves = [TilePlan()]
    moves.extend(TilePlan(units=(p["par_m"], p["par_n"]), regs=(p["reg_m"], p["reg_n"])) for p in _SCALAR_TILE_SPACE)
    return moves


# The warp (tensor-core) tile candidate SPACE: ``(WM, WN)`` warp counts × ``(FM, FN)`` per-warp
# register fragments × ``bk`` K-chunks, spelled ``a:<atom>/w..x../f..x../k..``. Two multiplicative
# budgets bound it and both are real hardware/codegen limits rather than deployment history: a warp
# tile spends ``32·WM·WN`` threads of the CTA budget, and the C fragment holds ``FM·FN ≤ 32`` cells.
# ``bk`` carries no budget of its own — the staged slab it implies is sized by the stage resolvers
# against the live smem cap, per node.
#
# Recorded golden winners, for why the ladders reach as far as they do: ``(8, 2)`` and
# ``(2, 8)`` (RTX 4090 / PRO 6000); ``(1, 16)``, the thin-M / wide-N decode geometry — a decode-M
# computed-A (fused norm->linear) contraction wants its warps spread across the wide output columns,
# and it beat the ``(1, 8)`` sibling ~5% on both the q_proj (N=4096) and gate/up (N=15360) fused
# edges at M=32 (5090); ``(2, 8)``, its M=64 sibling, the lm_head.m64 winner where ``w1x8`` leaves 2x
# on the table (2392 vs 1215 us, 5090) and the best fused-geglu tile at the same M.
#
# Per-node legality — the atom's operand dtype, the warp static-K divisibility, whether a slab fits
# smem — stays the scheduler's, not the space's.
_WARP_TILE_SPACE = Space(
    dims=(
        Dimension("wm", (1, 2, 4, 8, 16)),
        Dimension("wn", (1, 2, 4, 8, 16)),
        Dimension("fm", (1, 2, 4, 8)),
        Dimension("fn", (1, 2, 4, 8)),
        Dimension("bk", (1, 2, 4, 8)),
    ),
    bounds=(
        Bound(("wm", "wn"), limit=MAX_BLOCK_THREADS, coeff=WARP_LANES),  # the CTA thread budget
        Bound(("fm", "fn"), limit=MAX_FRAGMENT_CELLS),  # the C-fragment register budget
    ),
)


def warp_tile_moves(atom_names: tuple[str, ...]) -> list[TilePlan]:
    """The warp-contraction output-tile candidates over the (already dtype-eligible)
    ``atom_names``: :data:`_WARP_TILE_SPACE`'s points per atom. No conservative option-0 of its own
    — these EXTEND :func:`scalar_tile_moves` (whose per-cell tile leads the combined list)."""
    from emmy.compiler.ir.atom import ATOM_REGISTRY  # noqa: PLC0415

    moves = []
    for name in atom_names:
        atom = ATOM_REGISTRY[name]
        moves.extend(TilePlan(atom=atom, units=(p["wm"], p["wn"]), regs=(p["fm"], p["fn"]), bk=p["bk"]) for p in _WARP_TILE_SPACE)
    return moves


# The warp-flash (fragment-resident TWISTED) tile grid: warps per CTA over the query rows ×
# score n-atoms per streaming key block. The conservative pair leads (one warp, the ``2·atom_n``
# key block — today's deterministic stamp), so a cold tie keeps the historical geometry. Per-node
# legality (kv / query-row divisibility, the dtype atom) is the scheduler's
# (the twisted warp options), not the grid's.
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
    the Q@K / P@V mma :class:`TilePlan`\\ s in the twisted warp options (the ``TILE``
    codec spells the full plan; this grid only generates the free geometry — ``bk`` is shape-derived)."""
    return [(um, nt, fm) for fm in _FLASH_QTILES for um in _FLASH_WARPS for nt in _FLASH_KEY_ATOMS]


def map_tile_moves() -> list[TilePlan]:
    """The pointwise-map register-strip candidates — the **scalar output tile** (the same ``f<fn>``
    register sub-tile a contraction's output rides, here with no ``n`` unit-tile / atom since the
    grid already parallelizes a pure pointwise cell). ``f<r>`` hands each thread ``r`` CONTIGUOUS
    inner-axis elements (blocked: thread t owns ``[t·r, t·r+r)``) as ``r`` grouped loads + ``r``
    grouped writes, which ``050_vectorize_loads`` / ``080_vectorize_stores`` merge into one
    ``float<r>`` access — matching torch's ``vectorized_elementwise_kernel<r>``. These EXTEND the
    per-cell option-0 (``""``, 1 elem/thread) the scheduler leads with; legality (a static inner
    free axis divisible by r) is the scheduler's. The ladder stops at ``f4``: ``f8`` regressed both
    pointwise goldens (register pressure — 22 vs 14 regs — outweighs the wider access)."""
    return [TilePlan(regs=(1, 2)), TilePlan(regs=(1, 4))]  # (m, n): the strip widens the INNER axis


def stage_moves(*, warp: bool) -> list[Stage]:
    """The operand-staging candidates as TYPED :class:`Stage` slices — the transport / depth /
    double-buffer variants. Both tiers offer the gmem→smem prefetch ring depths (the scalar ring
    lands on the same ``staged_kloop`` phases; its slab K-chunk is depth-aware, derived by the
    scalar stage resolver); the ``p2`` smem→register double-buffer is an ``ldmatrix`` transform,
    warp-only.

    ``split`` — one transport PER staged edge instead of one over all of them — joins the warp list.
    It resolves only where the fold's edges are consumed at DISTINCT positions of its derived
    evaluation (``_legality.stage_split_groups``), which a contraction's single multiply never is,
    so the matmul resolvers decline it and the streaming pair is where it lands.

    Gmem-direct is the ABSENCE of a stage (``None``), so it is not a member here — the enumeration
    leads with it as the conservative option-0. Emission is resolver-gated: a candidate is offered
    only when it RESOLVES against the built node, and the row carries the RESOLVED spelling."""
    depths = [Stage.parse(s) for s in ("d1/cp", "d2/cp", "d3/cp", "d4/cp", "d1/tma", "d2/tma", "d3/tma", "d4/tma")]
    if not warp:
        return depths
    return [*depths, Stage.parse("d2/cp/p2"), Stage.parse("d1/cp/split"), Stage.parse("d1/tma/split")]


# Cross-CTA split-K widths (the ``REDUCE`` codec's ``g<w>`` field). Divisor / occupancy legality is
# the scheduler's.
SPLITK_WIDTHS: tuple[int, ...] = (2, 4, 8)


def splitk_moves() -> list[ReducePlan]:
    """The cross-CTA split-K ``REDUCE`` candidates, both tiers each: the deferred-kernel finalize
    (an f32 workspace + sibling combine kernel) and the in-place atomic (one kernel — the partial
    ``atomicAdd``\\ s into the zero-init'd output; the mma tier rides ``RegStore.atomic``'s
    packed-pair red). The scheduler's ``atomic_ok`` gate keeps atomic rows
    off multi-fold / non-distributive-projection nodes. These EXTEND the serial option-0. Both tiers share the
    catalog — per-node legality lives with the enumeration's gates, not here."""
    return [ReducePlan.of(cta=w, finalize=f) for w in SPLITK_WIDTHS for f in ("kernel", "atomic")]


def coop_reduce_moves() -> list[ReducePlan]:
    """The cooperative / ILP K-partition ``REDUCE`` candidates for a NON-output-tiled contraction
    (the coop reduce spec's contract — the per-cell tier folds K across the coop threads / ILP
    register chains). These EXTEND the serial option-0. The 16- / 32-wide coop folds are recorded
    reduce-golden winners (the wide-row folds) — kept enumerable so the reduce goldens stay
    reachable. The 64–512-wide folds are the memory-bound normalizer band: a wide-K softmax /
    rms_norm saturates bandwidth only with a full-block coop row (``softmax.k2048`` wants 512 —
    2.6× over 32). The scheduler's coop reduce spec declines a band wider than the row has
    work for, so enumerating them is safe on small K. The TRANSPOSED band is the k-major-B matvec
    partition (warp lanes sweep the output axis — the M=1 gemv tier's coalescing fix);
    The reduce-candidate enumeration gates it structurally (plain contraction, 32-divisible inner free axis)
    AND by layout: the band is offered only on k-major B, and the plain band only on K-contiguous B
    at the matvec tier. Measurement used to decide the layout, but ShapeKey is layout-blind —
    cross-orientation golden/evidence rows tie, and a cold/tied pick landed the band on the wrong
    operand three times in one day (10-100× regressions; the WS5 cold-poison hardening). An env pin
    bypasses the gate (exploration)."""
    return [
        *(ReducePlan.of(coop=n) for n in (4, 8, 16, 32, 64, 128, 256, 512)),
        ReducePlan.of(reg=2),
        ReducePlan.of(reg=4),
        ReducePlan.of(coop=4, reg=2),
        # The transposed band + its grid-split composites: a bare transposed fold is latency-bound
        # on long-K matvecs (120 CTAs of serial K), so the deployable winners pair it with a
        # deferred-kernel grid split (down g32k/b256t 75.7 us = the row-major floor on k-major B).
        *(ReducePlan.of(coop=n, coop_transposed=True) for n in (32, 64, 128, 256)),
        ReducePlan.of(cta=8, coop=128, coop_transposed=True),
        *(ReducePlan.of(cta=w, coop=256, coop_transposed=True) for w in (8, 16, 32)),
    ]
