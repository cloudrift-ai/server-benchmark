"""The search-space knob declarations.

**INVARIANT: every ``Knob`` instance is declared here, and nowhere else.**
:mod:`~emmy.compiler.pipeline.knob` owns the ``Knob`` *descriptor* (the dataclass), the registry,
and the env plumbing; :mod:`~emmy.compiler.pipeline.search.features` owns the featurizers; this
module owns the concrete declarations. A rule that decides a knob imports it from here
(``from emmy.compiler.pipeline.search.space import VECTORIZE_LOADS``) rather than declaring its
own. Registration is construction (``Knob.__post_init__``), and ``knob.registry()`` imports this
module before answering — so the registry is complete in any process that can ask, however little
of the pipeline it loaded. When adding a knob, declare it here and import it into the owning rule.

The classic schedule's typed move catalogs live with its model in
:mod:`emmy.compiler.ir.schedule.catalog`; search consumes encoded rows but does not define the
candidate schedules.

Two groups:

- **Schedule codec knobs** (``WORK`` / ``REDUCE`` / ``TILE`` / ``STAGE`` / ``RASTER``) — the tile-lowering schedule
  fork points serialized by ``ClassicScheduleCodec``. The typed assignment is materialized in
  ``lowering/kernel/010_materialize``; its encoded row rides on ``TileOp.knobs`` so the online
  prior can featurize and tune the decision. ``off=""`` is the explicit direct leaf value.
- **Kernel-lowering policy knobs** (``VECTORIZE_LOADS`` / ``INTERLEAVE_LOADS``) — boolean codegen
  policies recorded on the kernel op (idempotence + env override), on by default and not search
  dimensions (``hints=(True,)``).
"""

from __future__ import annotations

import logging

from emmy.compiler.pipeline.knob import Knob, KnobType

logger = logging.getLogger(__name__)

# --- Schedule codec knobs ---------------------------------------------------

PLACE = Knob(
    "PLACE",
    KnobType.STR,
    hints=("fuse", "cut"),
    help="Stored Fold-edge placement (fuse into the consumer or cut to a fresh workspace kernel).",
)

# The exp-family reduce pair a twist recipe fuses: its twisted carrier (one pass over the axis), or
# the two-pass tree the lift reconstructs from that carrier's loop, whose value channel is a
# contraction node with a ``TILE`` site of its own. Decided by ``lowering/tile/020_twisted``.
TWIST = Knob(
    "TWIST",
    KnobType.STR,
    hints=("twisted", "two-pass"),
    help="A twist recipe's reduce pair: the fused twisted carrier (one pass) or the two-pass tree.",
)

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
# shared-memory slab + double-buffered producer (``smem`` synchronous thread fill / ``smem-async``
# cp.async / ``smem-tma`` TMA) over the serial reduce loop, instead of the no-intermediate baseline.
# Resolved into the schedule's :class:`Stage` (``None`` = no intermediate). Composes with both
# fragments of the unified ``TILE`` knob.
STAGE = Knob(
    "STAGE",
    KnobType.STR,
    help="Operand-staging codec (d<depth>/smem|smem-async|smem-tma[/p<reg_depth>]; empty=no intermediate). "
    "Decided in the tile schedule, materialized in lowering/kernel/010_materialize.",
    off="",
)

# The kernel-global worker inventory (the step-7 value-grammar family): the w/n worker tokens
# factored out of the per-site TILE values, the coop width out of REDUCE, and the producer band the
# retired WSPEC family used to spell, absorbed as ``+p<n>``. Encoded from the accepted kernel
# choice on every complete row; ``off=""``
# = the per-cell / pure-reduce forms' derived launch geometry.


def _work_features(val) -> dict[str, float]:
    """The ``WORK`` sub-features for the online prior — ONLY the ``+p`` producer band, as the
    same ``D_wspec_warps`` the retired per-row ``WSPEC`` key spelled (name/semantics preserved;
    a legacy row's ``WSPEC`` key writes the identical value). The inventory's tile geometry is
    NOT re-featurized here — the per-node featurizers already fold it in by resolving the site
    ``TILE``/``REDUCE`` values against WORK (``features._tile_plan`` / ``_reduce_decomp``)."""
    if not val:
        return {"D_wspec_warps": 0.0}
    from emmy.compiler.ir.schedule import Work  # noqa: PLC0415 — deferred: schedule imports this module

    try:
        w = Work.parse(str(val))
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
    if r.is_direct:
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

VECTORIZE_STORES = Knob(
    "VECTORIZE_STORES",
    KnobType.BOOL,
    hints=(True,),  # on by default; not a search dimension — manual override only via the env var
    help="Fold runs of consecutive scalar Writes into one wide vector Write (float4 / __half2).",
    off=False,
)

PAIR_LDMATRIX = Knob(
    "PAIR_LDMATRIX",
    KnobType.BOOL,
    hints=(True,),  # on by default; not a search dimension — manual override only via the env var
    help="Pair slab-adjacent staged ldmatrix.x2 B-fragment loads into one ldmatrix.x4.",
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
# stamped BOOL, the ``TILE`` codec's bare atom token) — so it is ``unfeatured`` and never
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
    help="Offer the f16-mma / chunked-f32-accumulate atom forks (mma_m16n8k16_f16_f16 — the mma "
    "chain accumulates in f16 at the full HMMA rate, with a periodic register promote into f32 "
    "shadows; ~2x mma throughput on consumer dies where f32-accumulate is half rate). "
    "Pin 1 to offer on every target, 0 never; unset follows FAST_MATH (consumer-die targets only). "
    "Enumeration-gate only — the realized fork is identified by the TILE codec's atom token.",
)


FP8_MMA = Knob(
    "FP8_MMA",
    KnobType.BOOL,
    hints=(False,),
    help="Offer the native fp8 tensor-core atom forks (mma_m16n8k32_e4m3_f32 / _e5m2_f32 — both "
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
