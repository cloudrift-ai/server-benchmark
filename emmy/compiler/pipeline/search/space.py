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
scheduler spells each row ONCE, site-local, where it becomes stored state. The ``STAGE`` / ``WSPEC``
/ ``RASTER`` grids stay codec strings — those families have no worker half to factor out.

Two groups:

- **Schedule codec knobs** (``REDUCE`` / ``TILE`` / ``STAGE`` / ``WSPEC`` / ``RASTER``) — the tile-lowering schedule
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

from emmy.compiler.ir.schedule import ReducePlan, TilePlan
from emmy.compiler.pipeline.knob import Knob, KnobType

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
    help="Operand-staging codec (d<depth>/sync|cp|tma[/ring][/alt][/p<reg_depth>]; empty=gmem-direct). "
    "Decided in the tile schedule, materialized in lowering/kernel/010_materialize.",
    off="",
)

# Warp specialization — an env-pin alias: the role→warp split (``p<np>`` producer warps drive the
# ``STAGE`` load half; compute warps stay on the mma) lives in the WORK inventory's ``+p`` suffix.
# Gated on a warp ``TILE`` + a resolved **TMA** ``STAGE`` (the producer band drives the box-copy
# mbarrier ring; cp.async's wait-group is issuing-thread-scoped and a sync compute-fill has no
# async load half).
WSPEC = Knob(
    "WSPEC",
    KnobType.STR,
    help="Warp-specialization env-pin ALIAS — the producer band now rides WORK's +p suffix; a "
    "WSPEC pin (p<np> producer[:q<window>, reserved]; empty=uniform SIMT) narrows the WORK "
    "inventory via seal_workers. No realized row carries the key.",
)


# The kernel-global worker inventory (the step-7 value-grammar family): the w/n worker tokens
# factored out of the per-site TILE values, the coop width out of REDUCE, the WSPEC producer band
# absorbed as ``+p<n>``. Stamped by ``ops.seal_workers`` on every assembled option row; ``off=""``
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
    "Kernel-scoped like WSPEC (no @<axis> key); changes no per-CTA work or layout, only the "
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


def scalar_tile_moves() -> list[TilePlan]:
    """The scalar-contraction output-tile candidates: the per-cell tile first — the conservative
    option-0 — then the register-tile grid (:data:`_SCALAR_PAR` × :data:`_SCALAR_REG`) filtered by
    the ``par_n·par_m ≤ 1024`` thread budget."""
    moves = [TilePlan()]
    for par in _SCALAR_PAR:
        if par[0] * par[1] > MAX_BLOCK_THREADS:
            continue
        for reg in _SCALAR_REG:
            moves.append(TilePlan(units=par, regs=reg))
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


def warp_tile_moves(atom_names: tuple[str, ...]) -> list[TilePlan]:
    """The warp-contraction output-tile candidates over the (already dtype-eligible)
    ``atom_names``: the :data:`_WARP_UNITS` × :data:`_WARP_REGS` × :data:`_WARP_BK` grid per atom.
    No conservative option-0 of its own — these EXTEND :func:`scalar_tile_moves` (whose per-cell
    tile leads the combined list)."""
    from emmy.compiler.ir.atom import ATOM_REGISTRY  # noqa: PLC0415

    moves = []
    for name in atom_names:
        atom = ATOM_REGISTRY[name]
        for units in _WARP_UNITS:
            for regs in _WARP_REGS:
                for bk in _WARP_BK:
                    moves.append(TilePlan(atom=atom, units=units, regs=regs, bk=bk))
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


def splitk_moves(*, warp: bool) -> list[ReducePlan]:
    """The cross-CTA split-K ``REDUCE`` candidates, both tiers each: the deferred-kernel finalize
    (an f32 workspace + sibling combine kernel) and the in-place atomic (one kernel — the partial
    ``atomicAdd``\\ s into the zero-init'd output; the mma tier rides ``RegStore.atomic``'s
    packed-pair red). The scheduler's ``atomic_ok`` gate (``_reduce_candidates``) keeps atomic rows
    off multi-fold / non-distributive-projection nodes. These EXTEND the serial option-0."""
    del warp  # both tiers share the catalog; per-node legality lives in the scheduler's gates
    return [ReducePlan.of(cta=w, finalize=f) for w in SPLITK_WIDTHS for f in ("kernel", "atomic")]


def coop_reduce_moves() -> list[ReducePlan]:
    """The cooperative / ILP K-partition ``REDUCE`` candidates for a NON-output-tiled contraction
    (``_coop_reduce_spec``'s contract — the per-cell tier folds K across the coop threads / ILP
    register chains). These EXTEND the serial option-0. The 16- / 32-wide coop folds are recorded
    reduce-golden winners (the wide-row folds) — kept enumerable so the reduce goldens stay
    reachable. The 64–512-wide folds are the memory-bound normalizer band: a wide-K softmax /
    rms_norm saturates bandwidth only with a full-block coop row (``softmax.k2048`` wants 512 —
    2.6× over 32). The scheduler's ``_coop_reduce_spec`` declines a band wider than the row has
    work for, so enumerating them is safe on small K. The TRANSPOSED band is the k-major-B matvec
    partition (warp lanes sweep the output axis — the M=1 gemv tier's coalescing fix);
    ``_reduce_candidates`` gates it structurally (plain contraction, 32-divisible inner free axis)
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
