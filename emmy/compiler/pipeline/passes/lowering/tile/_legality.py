"""Per-node schedule legality — **one predicate, one home, one severity**.

Every function here answers ONE question about whether a candidate is realizable against a
particular node, and returns the REFUSAL REASON as a string (or ``None`` for "legal"). The caller
picks the severity through :func:`enforce`: an env PIN raises the reason, the unpinned enumeration
silently drops the candidate. That split is why the predicates live together — the same rule stated
twice, once as a raise and once as a drop, is what produces "the pin says yes and the enumeration
says no".

The rules here are ORDINARY ARITHMETIC over facts read off the node — a thread budget, a K-step
that must tile a static extent, a 16 B-aligned inner stride. They are deliberately NOT expressed as
:mod:`~emmy.compiler.pipeline.search.domain` ``Bound``\\ s: that module GENERATES a candidate
domain, and nothing here is ever installed into a ``Space``. Routing a scalar divisibility test
through a constructed ``Bound`` shares the currency in name only and costs a second file open to
read ``k % step == 0``. If a legality rule ever does become a generated bound, constructing one is
a line.

What does NOT live here: anything that CHOOSES rather than checks. The conservative cooperative
pick, the atom ladder and the stage catalog are the schedule's.

**With ONE stated exception: the stage RESOLVERS** (``resolve_warp_stage`` / ``resolve_scalar_stage``
/ ``resolve_sync_stage`` / ``resolve_twisted_stage``). They return a SIZED :class:`Stage` rather
than a reason, because for the smem budget the legal answer is a size and not a yes/no — the
largest depth and slab chunk that fit. Handing back the largest legal stage is what keeps an
over-budget row out of the fork instead of failing at materialization, and splitting "how big may
this be" from "is this legal" would put the budget in two places. They are resolvers, they are
named so, and they clamp through the one shared :func:`clamp_depth`; everything else here refuses.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.schedule import ReducePlan, Stage, TilePlan, WarpSpec
from emmy.compiler.ir.stmt import Body, Load, Loop
from emmy.compiler.ir.stmt.passes import has_contraction_tail
from emmy.compiler.ir.tile import Fold
from emmy.compiler.ir.tile.ir import is_contraction, operand_name
from emmy.compiler.ir.tile.ops import cone_seam
from emmy.compiler.pipeline.passes.lowering._addr import BYTE_SLAB_PAD, gmem_axis_step
from emmy.compiler.pipeline.search.space import MAX_BLOCK_THREADS, WARP_LANES

# TMA hardware: every box dim must fall in 1..256, and the swizzle-split box caps the operand rank
# at 4 so it stays within the 5-dim limit.
_TMA_MAX_BOX = 256
_TMA_ALIGN = 16  # the NONE-swizzle box-copy rule: 16 B-aligned inner dim and inner global stride

# cp.async needs a >= 4 B contiguous chunk, so a 2 B/elem slab's inner dim must be even.
_CP_ASYNC_MIN_ELEMS = 2

# The cp.async streaming slabs' padded rows, in elements — two slabs at ``2 * _twist._PAD`` each,
# the bank-conflict break the hardware swizzle replaces on the TMA path.
_STREAM_PAD_ELEMS = 16

# The staged QUERY slab's row pad, in elements (the ``split`` arm stages Q through smem too).
_STREAM_Q_PAD_ELEMS = 8


def enforce(reason: str | None, *, pinned: bool) -> bool:
    """Resolve a refusal into a decision: ``True`` when legal, ``False`` when the candidate should
    be dropped, and a ``ValueError`` when it was PINNED — a pin names a specific kernel, so refusing
    it silently would deploy something the user did not ask for."""
    if reason is None:
        return True
    if pinned:
        raise ValueError(reason)
    return False


# ---- thread budgets ---------------------------------------------------------------------------- #


def scalar_block_threads(plan: TilePlan) -> str | None:
    """A scalar tile launches one thread per parallel output cell, so its parallel widths spend
    ``par_n·par_m`` of the CTA budget."""
    if plan.units_n * plan.units_m <= MAX_BLOCK_THREADS:
        return None
    return (
        f"TILE parallel block {plan.units_n}x{plan.units_m}={plan.block_threads} threads exceeds the "
        f"{MAX_BLOCK_THREADS}-thread/CTA limit; shrink n/m or move work to the f register sub-tile."
    )


def producer_band(spec: WarpSpec, block_threads: int) -> str | None:
    """A dedicated producer band adds ``32·p`` threads ON TOP of the compute warps. Two budgets:
    the total fits the CTA limit, and the band does not outnumber the compute half.

    The only rule in this module that is NOT a bound — it mixes a sum with a product, which no
    coordinate change linearizes — so it stays an ordinary arithmetic predicate.
    """
    aux = WARP_LANES * spec.producer_warps
    if aux > block_threads:
        return f"producer band {aux} threads outnumbers the {block_threads} compute threads"
    if block_threads + aux > MAX_BLOCK_THREADS:
        return f"producer band {aux} + {block_threads} compute exceeds the {MAX_BLOCK_THREADS}-thread/CTA limit"
    return None


def producer_transport(stage: Stage | None, reduce: ReducePlan) -> str | None:
    """What a producer band can actually drive: a RESOLVED TMA stage (the band arms the box-copy
    mbarrier ring — cp.async's wait-group is issuing-thread-scoped and a sync compute-fill has no
    async load half) on a kernel that is not split across CTAs."""
    if stage is None or stage.transport != "tma":
        return "a producer band drives a resolved TMA stage; this row has none"
    if stage.split:
        return "a producer band arms ONE mbarrier ring; the per-edge transport split arms one per edge"
    if reduce.needs_split:
        return "a producer band and a cross-CTA split-K are not co-representable"
    return None


def coop_band_layout(plan: ReducePlan, k_contiguous: bool | None) -> str | None:
    """Whether a cooperative band's lane mapping is COALESCED on B's layout. The plain band
    interleaves lanes along K, so it needs K contiguous in B (the serving ``F.linear`` N×K layout);
    the transposed band sweeps lanes along the output axis, so it needs the canonical k-major
    ``B[k, n]``. ``k_contiguous is None`` (no B to classify) gates neither.

    Measurement cannot decide this: ``ShapeKey`` is layout-blind, so cross-orientation golden /
    evidence rows tie, and a cold or tied pick landed the band on the wrong operand three times in
    one day — 10-100x regressions (the WS5 cold-poison hardening). An env pin bypasses the gate."""
    if k_contiguous is None or plan.coop <= 1:
        return None
    if plan.coop_transposed and k_contiguous:
        return "the transposed coop band lane-sweeps the output axis — uncoalesced on a K-contiguous B"
    if not plan.coop_transposed and not k_contiguous:
        return "the plain coop band interleaves lanes along K — uncoalesced on a k-major B[k, n]"
    return None


def coop_band_geometry(plan: ReducePlan, k: int | None, inner: Axis | None) -> str | None:
    """The TRANSPOSED coop band's structural requirements — the same fact for every tier that
    offers it. The band swaps the lane mapping so 32 lanes sweep the innermost FREE axis while
    each lane walks K serially, which fixes three things about the term: the coop width must be a
    whole number of warps, the swept axis must be static and 32-divisible (there is no partial-warp
    sweep), and a split composite must divide K.

    One home because it had two: the reduce tier and the contraction tier each spelled a version of
    this inline, with different conditions and neither of them here — exactly the raise-vs-drop
    defect class this module exists to remove. The reduce tier's EXTRA requirement, on the shape of
    the epilogue, is :func:`coop_band_epilogue`; ``coop_band_layout`` still owns the B-orientation
    half. A plain (non-transposed) band answers ``None`` — it has no such geometry."""
    if not plan.coop_transposed:
        return None
    if plan.coop % WARP_LANES:
        return f"the transposed coop band sweeps whole warps — coop={plan.coop} is not a multiple of {WARP_LANES}"
    if inner is None or not inner.extent.is_static or inner.extent.as_static() % WARP_LANES:
        return f"the transposed coop band needs a static {WARP_LANES}-divisible innermost free axis to sweep"
    if plan.needs_split and (k is None or k % plan.cta):
        return f"the transposed split composite g{plan.cta} must divide a static K"
    return None


def coop_band_epilogue(tail) -> str | None:
    """The reduce tier's EXTRA requirement on a transposed band: the projection epilogue must be
    per-cell straight-line code. A swept lane owns output cells rather than a row, so a tail that
    is itself a loop — a sweep, or the nested contraction of a fused norm→linear — has no per-cell
    form for the swapped mapping to run, and the fused shape's shared-row slab is addressed for the
    plain band's row-per-CTA layout.

    A statement-SHAPE question, which is why it takes the tail rather than the term: the reduce
    tier used to ask it by calling the shared-row STAGE resolver and testing for ``None``, wiring a
    transport decision into a REDUCE candidate gate for a fact about the epilogue alone."""
    if any(isinstance(s, Loop) for s in tail):
        return "the transposed coop band needs a per-cell epilogue — this tail sweeps"
    if has_contraction_tail(tail):
        return "the transposed coop band needs a per-cell epilogue — this tail contracts over a new axis"
    return None


# ---- the warp tier's K step -------------------------------------------------------------------- #


def warp_atom_target(atom, ctx) -> str | None:
    """Whether ``atom`` belongs to the target's selected MMA instruction family."""
    if atom.available_on(ctx):
        return None
    return (
        f"warp TILE: atom {atom.name} requires target feature {atom.target_feature}, which is unavailable "
        f"on sm_{ctx.compute_capability[0]}{ctx.compute_capability[1]}"
    )


def warp_atom_edges(node: Fold, atom) -> str | None:
    """Whether the atom can consume this contraction's materialized/computed operand edges."""
    if not atom.materialized_edges_only:
        return None
    if not isinstance(node.a, Load) or any(not isinstance(ch.b, Load) for ch in node.channels):
        return f"warp TILE: atom {atom.name} accepts only materialized A and B edges"
    return None


def warp_a_columns(node: Fold, tile: TilePlan, inputs) -> str | None:
    """Whether a materialized A edge has the contiguous K columns an mma fragment loader reads.

    A gmem fragment load always advances one element at a time within an ``atom_k``-wide row.  A
    blocked index is still representable when every fragment stays inside one of its contiguous
    runs; an unknown or strided index is not.  Computed A is exempt because its sync fill writes a
    dense shared-memory slab rather than asking a gmem loader to interpret the source index.
    """
    if not isinstance(node.a, Load):
        return None
    step = gmem_axis_step(node.a, node.axis.name, inputs)
    width = tile.atom.atom_k
    if step is not None and step[0] == 1 and (step[1] == 0 or step[1] % width == 0):
        return None
    motion = "unknown" if step is None else f"{step[0]} elements per column"
    return (
        f"warp TILE: A fragment loaders read {width} contraction columns CONTIGUOUSLY, but this "
        f"operand's gmem index moves {motion}; drop the atom token to use the scalar tier."
    )


def stage_target(stage: Stage, ctx) -> str | None:
    """Whether ``stage`` names a copy instruction family present on ``ctx``."""
    if stage.transport == "cp.async" and not ctx.has_cp_async:
        return f"STAGE {stage.spell()}: cp.async requires sm_80 or newer"
    if stage.transport == "tma" and not ctx.has_tma:
        return f"STAGE {stage.spell()}: TMA requires sm_90 or newer"
    return None


def warp_atom_stage(atom, stage: Stage) -> str | None:
    """Whether ``atom`` has a shared-memory fragment drain for ``stage``."""
    if atom.materialized_edges_only and not (stage.transport == "sync" and atom.sync_copy_staging):
        return f"STAGE {stage.spell()}: atom {atom.name} supports only synchronous-copy staging"
    return None


def warp_k_step(node: Fold, plan: TilePlan) -> str | None:
    """The inner mma K-step ``atom_k·bk`` must tile a STATIC contraction K: the warp K-loop has no
    static-K tail masking, so a partial final step reads past the operand and silently corrupts the
    result. A SYMBOLIC K reaches the masked tier and is fine — except on the fp8 atoms, whose
    byte-gather fragment loaders have no masked-K zero-fill family."""
    ext = node.axis.extent
    if not ext.is_static:
        if plan.is_warp and plan.atom.operand_dtype("a").nbytes == 1:
            return f"atom {plan.atom.name}: the fp8 byte-gather loaders have no masked-K zero-fill — a symbolic K stays off the fp8 tier"
        return None
    k, step = ext.as_static(), plan.atom.atom_k * plan.bk
    if k % step == 0:
        return None
    return (
        f"warp TILE K-step {step} (atom_k={plan.atom.atom_k}*bk={plan.bk}) does not divide the static "
        f"contraction K={k}; the warp K-loop has no static-K tail masking yet, so a partial final step "
        f"corrupts the result. Pin a K that is a multiple of {step}, or drop the atom token to use the "
        f"scalar tier."
    )


def splitk_slice_k_step(node: Fold, plan: TilePlan, width: int) -> str | None:
    """After a cross-CTA split the inner contraction sees ``K/width``, which must still be a
    multiple of the mma K-step."""
    if not plan.is_warp:
        return None
    step = plan.atom.atom_k * plan.bk
    ks = node.axis.extent.as_static() // width
    if ks % step == 0:
        return None
    return (
        f"split-K slice K={ks} (K/{width}) is not a multiple of the mma K-step {step} "
        f"(atom_k={plan.atom.atom_k}*bk={plan.bk}); pick a split width whose slice is divisible."
    )


def splitk_width(k_axis: Axis, width: int) -> str | None:
    """A cross-CTA split must divide the contraction axis evenly — the σ-reindex reconstructs an
    absolute k from ``ksplit·(K/w) + kslice``, which is only a bijection when ``w`` divides K."""
    big_k = k_axis.extent.as_static()
    if big_k % width == 0:
        return None
    return f"split-K width {width} does not divide K={big_k}; pick a dividing split width."


def splitk_materialized_b(node: Fold) -> str | None:
    """Every channel's B must be a gmem ``Load`` — a computed B has no index to σ-reindex."""
    if all(isinstance(ch.b, Load) for ch in node.channels):
        return None
    return "split-K needs a materialized B on every channel — a computed B has no gmem index to σ-reindex"


# ---- the register strip ------------------------------------------------------------------------ #


def strip_width(extent: int, width: int) -> str | None:
    """The pointwise strip hands each thread ``width`` CONTIGUOUS inner-axis elements, so the width
    must tile the inner free extent."""
    if width <= 1 or (extent and extent % width == 0):
        return None
    return f"register strip width {width} does not divide the inner free extent {extent}"


# ---- the fragment epilogue --------------------------------------------------------------------- #


def fragment_epilogue(epilogue: Body) -> str | None:
    """The mma store folds the projection into a ``RegEpilogue`` whose leaf ``Load``\\ s are
    evaluated independently per fragment element, so a load whose INDEX reads a name an earlier
    epilogue stmt defined (an embedding gather) cannot be threaded through it. A nested output
    sweep is likewise not a per-fragment straight-line epilogue."""
    defs: set[str] = set()
    for s in epilogue:
        if isinstance(s, Loop):
            return "warp TILE: the projection epilogue contains an output sweep — use the scalar tier"
        if isinstance(s, Load) and {v for e in s.index for v in e.free_vars()} & defs:
            return (
                "warp TILE: the projection epilogue gathers through another epilogue load (a "
                "data-dependent index) — the fragment epilogue cannot thread it; drop the atom "
                "token to use the scalar tier."
            )
        defs.update(s.defines())
    return None


# ---- operand staging --------------------------------------------------------------------------- #


def _tma_operand_rank(index: tuple, tile_name: str, k_name: str) -> bool:
    """Whether TMA's box can encode this operand's gmem index. The data plane is the TRAILING 2
    dims; extra LEADING dims ride as extent-1 box dims whose origin is evaluated once per fill, so
    those exprs must not move with the tile or the K loop."""
    if not 2 <= len(index) <= 4:
        return False
    return all(not ({tile_name, k_name} & e.free_vars()) for e in index[:-2])


def _warp_vector_copy(k_axis: Axis, tile_n: int, bk_elems: int, mask_n: bool, b_trans: bool) -> bool:
    """Vector-copy staging: a STATIC, tile-divisible K, an unmasked N, and an even inner slab dim."""
    if mask_n or not k_axis.extent.is_static:
        return False
    if k_axis.extent.as_static() % bk_elems:
        return False
    return bk_elems % _CP_ASYNC_MIN_ELEMS == 0 and (b_trans or tile_n % _CP_ASYNC_MIN_ELEMS == 0)


def _warp_tma(k_axis: Axis, n_axis: Axis, tile_n: int, bk_elems: int, a_bytes: int, b_bytes: int, mask_n: bool, b_trans: bool) -> bool:
    """TMA staging: STATIC tile-divisible K and N, and 16 B-aligned inner dims — each operand's
    box inner dim and gmem inner stride at its OWN element width (a byte-staged fp8 B sizes at
    1 B). A transposed B boxes N-major, so N drops out of the alignment gate."""
    if mask_n or not (k_axis.extent.is_static and n_axis.extent.is_static):
        return False
    k, n = k_axis.extent.as_static(), n_axis.extent.as_static()
    if k % bk_elems:
        return False
    inner = (bk_elems * a_bytes, k * a_bytes) + ((bk_elems * b_bytes, k * b_bytes) if b_trans else (tile_n * b_bytes, n * b_bytes))
    return all(x % _TMA_ALIGN == 0 for x in inner)


def resolve_warp_stage(c: Fold, tile: TilePlan, stage: Stage, budget: int, inputs=None) -> Stage | None:
    """Resolve an operand ``Stage`` against the warp (mma) contraction ``c`` — synchronous copy,
    cp.async, TMA, or gmem-direct (``None``). The resolved stage carries ``bk_elems``, ``depth`` clamped so the ring's
    slots fit ``budget``, and ``reg_depth`` clamped to ``bk``. A tile whose single depth-1 slot
    already exceeds ``budget`` DECLINES — unlike the scalar resolver it cannot shrink the slab.

    A resolver rather than a predicate because the legal answer is a SIZE, not a yes/no: this is the
    one enforcement point for the smem budget, and returning the largest legal stage is what keeps
    an over-budget row out of the fork instead of failing at materialization.

    ``inputs`` (the per-buffer tensors) gates operand dtypes: the copy transports byte-copy into
    slabs sized at each operand's element width, so a slab is byte-copied verbatim and the drain
    reads exactly the bytes the fill deposited. Two dtype forms resolve: an operand traced AT the
    atom's operand dtype (the 16-bit family — ldmatrix drain), and a 1-byte (fp8-stored) operand —
    a B under a 16-bit atom stages as a RAW BYTE slab drained by the cooperative convert gather
    (W8A16), and the fp8 (k32) atoms stage both operands as byte slabs drained by the byte repack.
    Any other mismatch DECLINES and keeps the warp tier gmem-direct, whose fragment load converts
    per element (the same rule flash's ``stageable`` flag states). A byte slab's fill runs 16 B
    chunks and its cp.async row pad is 16 B (``_addr.BYTE_SLAB_PAD``), so its inner span — and,
    canonical-B, the gmem row stride N — must be 16-divisible."""
    if stage.split and stage_split_groups(c) is not None:
        return None  # one multiply consumes both edges — there is one transport group, nothing to cut
    atom = tile.atom
    sync_copy = stage.transport == "sync" and atom.sync_copy_staging
    if atom.materialized_edges_only and not sync_copy:
        return None
    bk_elems = tile.bk * atom.atom_k
    m, n = tile.m, tile.n
    a_nbytes, b_nbytes = atom.operand_dtype("a").nbytes, atom.operand_dtype("b").nbytes
    if inputs:
        for edge, role in ((c.a, "a"), *((ch.b, "b") for ch in c.channels)):
            t = inputs.get(edge.input) if isinstance(edge, Load) else None
            if t is None or t.dtype == atom.operand_dtype(role):
                continue
            if sync_copy:
                return None  # the Volta shared gather consumes f16 slabs; synchronous copies do not convert
            if role == "b" and t.dtype.nbytes == 1 and b_nbytes == 2:
                b_nbytes = 1  # fp8-B under a 16-bit atom: byte slab, convert at the drain
                continue
            return None
    for eb, inner, row_axis in (
        (a_nbytes, bk_elems, None),
        (b_nbytes, bk_elems if c.b_trans else n.tile, None if c.b_trans else n.axis),
    ):
        if eb != 1:
            continue
        if inner % 16:
            return None  # byte slab: 16 B chunks + the 16 B row pad need a 16-divisible inner span
        if row_axis is not None and (not row_axis.extent.is_static or row_axis.extent.as_static() % 16):
            return None  # canonical byte B: the 16 B gmem chunks stride rows of N bytes
    rank_ok = isinstance(c.a, Load) and all(isinstance(ch.b, Load) for ch in c.channels)
    rank_ok = (
        rank_ok
        and _tma_operand_rank(c.a.index, m.axis.name, c.axis.name)
        and all(_tma_operand_rank(ch.b.index, n.axis.name, c.axis.name) for ch in c.channels)
    )
    box_ok = max(m.tile, n.tile, bk_elems) <= _TMA_MAX_BOX
    tma_ok = (
        stage.transport == "tma"
        and rank_ok
        and box_ok
        and _warp_tma(c.axis, n.axis, n.tile, bk_elems, a_nbytes, b_nbytes, n.mask, c.b_trans)
    )
    vector_copy_ok = _warp_vector_copy(c.axis, n.tile, bk_elems, n.mask, c.b_trans)
    cp_ok = stage.transport == "cp.async" and vector_copy_ok
    sync_ok = sync_copy and vector_copy_ok
    if not (tma_ok or cp_ok or sync_ok):
        return None
    pad_a, pad_b = (BYTE_SLAB_PAD if eb == 1 and cp_ok else 0 for eb in (a_nbytes, b_nbytes))
    b_rows, b_cols = (n.tile, bk_elems + pad_b) if c.b_trans else (bk_elems, n.tile + pad_b)
    slot_bytes = m.tile * (bk_elems + pad_a) * a_nbytes + len(c.channels) * b_rows * b_cols * b_nbytes
    if slot_bytes > budget:
        return None
    depth = clamp_depth(stage, slot_bytes, budget)
    return replace(stage, depth=depth, reg_depth=min(stage.reg_depth, tile.bk), bk_elems=bk_elems)


def clamp_depth(stage: Stage, slot_bytes: int, budget: int) -> int:
    """The deepest ring the smem ``budget`` affords at ``slot_bytes`` per slot, never deeper than
    the stage asked for. The ONE clamp every stage resolver ends in — a budget stated once, so a
    tier cannot drift into affording a slot the others refuse."""
    return min(stage.depth, budget // slot_bytes)


def warp_operand_dtype(c: Fold, tile: TilePlan, a_dtype) -> str | None:
    """A MATERIALIZED ``a`` edge must already carry the atom's own operand dtype: the transports that
    fill its fragment — gmem-direct ``ldmatrix``, cp.async, TMA — move raw BYTES and cannot convert.
    Only the sync compute-fill converts (on the slab store), which is why a mixed pair reaches the
    warp tier through the mixed-A promotion's cone instead. A COMPUTED edge is exempt for exactly
    that reason."""
    if not isinstance(c.a, Load) or a_dtype == tile.atom.operand_dtype("a"):
        return None
    return (
        f"warp TILE: atom {tile.atom.name} takes a {tile.atom.operand_dtype('a')} A operand but this "
        f"contraction's A is {a_dtype} — a copy transport cannot convert it. The mma tier is reachable "
        f"through the converting sync compute-fill (the mixed-A cone), or drop the atom for the scalar tier."
    )


def computed_operand_cover(c: Fold, tile: TilePlan) -> str | None:
    """Geometry required by a sync compute-filled contraction operand.

    Every compute-filled edge needs static K because the staged driver unrolls fixed K chunks. A
    computed A leaves B on the async-copy path, whose contiguous N-vector copy cannot clamp a
    partial inner row element-by-element, so N must be exact. A computed B leaves materialized A
    as the async operand; M is its *outer* slab row and can be safely clamped as a whole. Computed
    B's own per-cell fill clamps N before evaluating the generic producer cone.
    """
    if not c.axis.extent.is_static:
        return "a computed contraction operand needs a static K — the sync compute-fill has no K mask"
    materialized_b = [isinstance(ch.b, Load) for ch in c.channels]
    if any(materialized_b) and not all(materialized_b):
        return "sync compute-fill requires homogeneous B channels; mixed computed/materialized B layouts stay on the demoted reading"
    if tile.n.mask and any(isinstance(ch.b, Load) for ch in c.channels):
        return (
            f"a sync compute-fill with a materialized B needs a TILE whose N width exactly covers "
            f"the static output columns (N={tile.n.axis.extent}; copied inner-row chunks cannot "
            f"clamp individual N cells); pick a dividing tile."
        )
    return None


def computed_operand_copy_dtype(c: Fold, tile: TilePlan, inputs) -> str | None:
    """A materialized edge beside a compute-filled operand must already have the atom dtype.

    The ``sync`` stage evaluates computed cones into their typed shared-memory slabs, but it
    *copies* every materialized peer byte-for-byte.  A copied f32 edge therefore cannot feed an
    f16 ``ldmatrix`` fragment merely because another edge is computed; it must take the demoted
    scalar reading (or an explicitly converting producer cone).  Computed edges are exempt because
    their slab store performs the normal typed conversion.
    """
    for edge, role in ((c.a, "a"), *((ch.b, "b") for ch in c.channels)):
        if not isinstance(edge, Load):
            continue
        tensor = inputs.get(edge.input) if inputs else None
        # Structural scheduler fixtures (and a few pre-stamp inventory callers) intentionally do
        # not carry Tensor metadata.  Absence is not evidence of an unsafe byte copy; the concrete
        # lowering path always supplies inputs and is where a known mismatch must be rejected.
        if tensor is None:
            continue
        want = tile.atom.operand_dtype(role)
        if tensor.dtype == want:
            continue
        got = tensor.dtype
        return (
            f"sync compute-fill: materialized {role.upper()} edge {edge.input!r} is {got}, but "
            f"atom {tile.atom.name} copies it into a {want} slab without conversion; use the "
            "demoted scalar reading or an explicitly converting producer cone"
        )
    return None


def computed_a_cover(c: Fold, tile: TilePlan) -> str | None:
    """Compatibility alias for callers/tests phrased around the original computed-A lane."""
    return computed_operand_cover(c, tile)


def resolve_sync_stage(c: Fold, tile: TilePlan, budget: int, want_depth: int = 1) -> Stage | None:
    """The ``sync`` compute-fill :class:`Stage` for a computed-operand warp contraction under ``tile``
    — MANDATORY for this form (the gmem-direct mma leaf refuses a computed A, and cp.async / TMA are
    copy transports that cannot evaluate a producer cone), so it has no gmem-direct ``""`` sibling
    and a ``STAGE`` pin can only choose its DEPTH. ``None`` when the slabs exceed ``budget``: one A
    slab, one B slab per channel, and one fp32 row per bridged statistic (``ops.cone_seam``'s
    ``stats`` — the same node boundary the materializer fills through).

    ``want_depth >= 2`` is the **asymmetric B-only prefetch ring**: only the B cp.async slabs ring
    (their copies for chunk ``i+d-1`` fly under chunk ``i``'s compute fill and drain), while the
    compute-filled A slab and the stat rows stay single-buffer — ringing a compute fill buys no
    overlap, it runs on the drain's own threads. Measured on the gemma gate_up edge at M=512 (5090)
    the ring LOSES (897 vs 665 µs) — the extra B slot alone crosses the smem occupancy quantization
    — but at decode M (``tile_m ≤ 32``) the A slab and stat rows are tiny and the tradeoff inverts,
    so both depths are enumerated as fork siblings and measured per shape."""
    atom = tile.atom
    if atom.materialized_edges_only:
        return None
    if atom.operand_dtype("a").nbytes < 2:
        return None  # fp8 atoms: the compute fill's slab store + ldmatrix drain are 16-bit-only
    bk_elems = tile.bk * atom.atom_k
    a_nbytes = atom.operand_dtype("a").nbytes
    b_nbytes = atom.operand_dtype("b").nbytes
    _, _, stats = cone_seam(c.a) if not isinstance(c.a, Load) else ((), (), ())
    a_bytes = tile.m.tile * bk_elems * a_nbytes
    stat_bytes = len(stats) * tile.m.tile * 4
    sync_bytes = stat_bytes
    async_bytes = 0
    if isinstance(c.a, Load):
        async_bytes += a_bytes
    else:
        sync_bytes += a_bytes
    for ch in c.channels:
        if isinstance(ch.b, Load):
            async_bytes += tile.n.tile * bk_elems * b_nbytes
        else:
            sync_bytes += tile.n.tile * bk_elems * b_nbytes
    if sync_bytes + async_bytes > budget:
        return None
    depth = want_depth if want_depth >= 2 and async_bytes and sync_bytes + want_depth * async_bytes <= budget else 1
    computed = [] if isinstance(c.a, Load) else [operand_name(c.a)]
    computed.extend(operand_name(ch.b) for ch in c.channels if not isinstance(ch.b, Load))
    return Stage(depth=depth, transport="sync", smem=tuple(computed), bk_elems=bk_elems)


# ---- the twisted streaming pair ---------------------------------------------------------------- #


def _reaches(stmt, edge) -> bool:
    """Whether ``edge`` occurs anywhere under ``stmt``. The walk crosses operand EDGES as well as
    nested bodies: an edge is not a body dependency (``Fold.nested`` withholds a contraction's
    edges by design), and where an edge is consumed is exactly the question here."""
    if stmt is edge:
        return True
    if isinstance(stmt, Fold) and any(_reaches(e, edge) for e in stmt.operands):
        return True
    return any(_reaches(s, edge) for b in stmt.nested() for s in b)


def stage_split_groups(node: Fold) -> str | None:
    """Whether ``split`` — the transport GROUP GRANULARITY — is structurally available on ``node``.
    It cuts the fold's staged edges into one transport group EACH, which needs ≥ 2 staged operand
    edges consumed at DISTINCT positions of the derived evaluation: the streaming pair's score edge
    reads K at the head of the step, its synthesized PV reads V further down, so each group refills
    in the phase that no longer reads it. A contraction consumes both edges in ONE multiply, so
    there is a single group and nothing to cut — which is why the matmul resolvers decline the
    value, now for a stated reason rather than by name.

    STRUCTURAL, not geometric: the answer is a property of how the term is BUILT, so a refused pin
    is a user error rather than a budget."""
    if is_contraction(node):
        return "STAGE split: a contraction's one multiply consumes both operand edges — one transport group, nothing to cut"
    steps = list(node.step_stmts())
    at = {i for e in node.operands if (i := next((n for n, s in enumerate(steps) if _reaches(s, e)), None)) is not None}
    if len(at) >= 2:
        return None
    return (
        f"STAGE split cuts a fold's staged edges into one transport group each; this fold consumes its "
        f"{len(node.operands)} operand edge(s) at {len(at)} position(s) of its derived evaluation."
    )


def twisted_warp_columns(work) -> str | None:
    """Constraint 15 — a TWISTED site is m-ONLY. The materializer has no cross-warp merge of the
    twisted ``(m, l, O)`` carrier (both flash sites are realized at ``units=(um, 1)``), so a warp
    inventory spending a second warp COLUMN names a stream that cannot be materialized at all."""
    if work is None or work.kind != "warp" or work.units[1] == 1:
        return None
    return (
        f"the twisted streaming pair is m-only (WN = 1) — there is no cross-warp merge of the (m, l, O) "
        f"carrier; WORK {work.spell()} spends {work.units[1]} warp columns."
    )


def twisted_atom(atom, d_v: int) -> str | None:
    """The mma atom's own demands on a streaming pair: the C→A register repack that feeds the P@V
    fragments from the score's accumulator (constraint 11), and a value dim the PV fragment tiles
    exactly (``WN·FN·atom_n = d`` at ``WN = 1`` — constraint 2)."""
    if not atom.c_to_a_repack:
        return f"atom {atom.name} has no C→A register repack — the P@V fragment feed has no tier without it"
    if d_v % atom.atom_n:
        return f"the value dim {d_v} is not a multiple of the P@V atom's n width {atom.atom_n}"
    return None


def twisted_block(stream: Fold, qk_tile: TilePlan) -> str | None:
    """The streaming key block and the query-row block must COVER their extents exactly when those
    are STATIC: a static ragged tail has no fragment mask (the symbolic path masks at the fragment
    and guards the gmem reads, so a symbolic extent is fine). ``qk_tile`` is the PLACED score
    slice, so its ``m`` side is the query axis and its ``n`` tile is the streamed key block."""
    kv, m = stream.axis.extent, qk_tile.m.axis.extent
    if kv.is_static and kv.as_static() % qk_tile.tile_n:
        return f"the {qk_tile.tile_n}-key streaming block does not divide the static KV extent {kv.as_static()}"
    if m.is_static and m.as_static() % qk_tile.tile_m:
        return f"the {qk_tile.tile_m} query rows per CTA do not divide the static M extent {m.as_static()}"
    return None


def twisted_sites_agree(qk_tile: TilePlan, pv_tile: TilePlan) -> str | None:
    """The QK / PV SIBLING EQUALITY — constraints 3 and 8. The two sites are enumerated
    independently (they are two sites, and making them agree is what the recursion is for), so the
    pair reconciles here: both schedule the mma tier or neither, the PV rows ARE the score's rows
    (one register query chain each), and the PV K-chunk is the streamed key block the score just
    produced. The block must also be a whole number of P@V atom-K steps — the C→A repack pairs
    k-adjacent score fragments, so an odd one leaves a partial expect fold."""
    if qk_tile.is_warp != pv_tile.is_warp:
        return "the twisted pair schedules both sites on the mma tier or neither"
    if not qk_tile.is_warp:
        return None
    if pv_tile.reg_m != qk_tile.reg_m:
        return f"the P@V register query tiles ({pv_tile.reg_m}) must match the score's ({qk_tile.reg_m}) — one chain each"
    atom_k = pv_tile.atom.atom_k
    if qk_tile.tile_n % atom_k:
        return f"the {qk_tile.tile_n}-key streaming block is not a multiple of the P@V atom K-step {atom_k}"
    if pv_tile.bk != max(1, qk_tile.tile_n // atom_k):
        return (
            f"the P@V K-chunk k{pv_tile.bk} does not cover the {qk_tile.tile_n}-key streaming block "
            f"(k{max(1, qk_tile.tile_n // atom_k)} at atom_k={atom_k})"
        )
    return None


def splitkv_slice(stream: Fold, qk_tile: TilePlan, plan: ReducePlan) -> str | None:
    """What a flash split-KV partition (``REDUCE=g<n>k``) demands. The FINALIZE is the first half
    and it is categorical: the twisted ``(m, l, O)`` state folds across partitions through an
    exp-family LSE combine, which no atomic spells — ``030_split_reduce``'s deferred-kernel arm is
    the only realization. The second is geometric: the slices must be BLOCK-WHOLE, since the staged
    chunking and the fragment column masks assume a whole number of streaming key blocks. A SYMBOLIC
    kv always splits — ``030_split_reduce`` builds the bn-aligned runtime slice width and the
    absolute bound the realizer stops and masks against."""
    if plan.finalize != "kernel":
        return "the twisted (m, l, O) state cannot finalize atomically — its e^{Δm} rescale is not an atomic; pin REDUCE=g<n>k"
    ext = stream.axis.extent
    if not ext.is_static:
        return None
    n = ext.as_static()
    if n % plan.cta:
        return f"split-KV width {plan.cta} does not divide the KV extent {n}"
    if (n // plan.cta) % qk_tile.tile_n:
        return f"the split-KV slice {n // plan.cta} is not a whole number of {qk_tile.tile_n}-key streaming blocks"
    return None


def resolve_twisted_stage(stream: Fold, qk: Fold, qk_tile: TilePlan, pv_tile: TilePlan, stage: Stage, budget: int) -> Stage | None:
    """Resolve an operand ``Stage`` against a warp-flash streaming pair — the K/V slabs of ONE
    ``bn``-key streaming step (K ``bn × head_dim`` in its native N-major layout, V ``bn × d_v``
    K-major; both verbatim row copies, so staging stays bit-identical to gmem-direct). ``None`` when
    the transport declines or the slabs exceed ``budget``.

    Two transports resolve: **cp.async** (cooperative fills, +16 elem padded rows for the
    bank-conflict break) and **TMA** (the batched K/V encode as rank-N boxes with leading extent-1
    dims, so the slabs stay dense and take the hardware swizzle instead of padding; box dims cap at
    the 256 hardware limit). ``sync`` has nothing to overlap. A SYMBOLIC kv stages under both — TMA
    zero-fills the box overhang, cp.async clamp-reads the tail's key rows to the last valid key, and
    the drain's tail masks zero their P columns either way; a STATIC, non-block-divisible kv has no
    tail mask, so it stays gmem-direct.

    ``split`` is the ALTERNATING single-slab pipeline: one K slab + one V slab, each on its own
    mbarrier, refilled in the phase that no longer reads it (K under softmax + P·V, V under the next
    step's Q·K) — the FA-2 choreography that overlaps a WIDE streaming block's copies in HALF the
    paired ring's smem. It stages the QUERY tile through smem too (a padded row-major slab filled
    once, its A fragments ldmatrix'd per atom-K chunk), which frees the resident Q registers that
    made the wide block spill. Its structural eligibility is :func:`stage_split_groups`; the sizing
    is here.

    A resolver rather than a predicate, for the same reason the matmul ones are: the legal answer is
    a SIZE (the ring depth the budget affords), and the row carries the RESOLVED spelling."""
    if stage.transport not in ("cp.async", "tma"):
        return None  # sync has nothing to overlap on a stream — there is no compute fill here
    kv, bn = stream.axis.extent, qk_tile.tile_n
    if kv.is_static and kv.as_static() % bn:
        return None  # a static ragged tail has no mask on either transport — stay gmem-direct
    head_dim, d_v = qk.axis.extent.as_static(), pv_tile.tile_n
    elem_bytes = qk_tile.atom.operand_dtype("b").nbytes
    if stage.transport == "tma":
        if max(bn, head_dim, d_v) > _TMA_MAX_BOX:
            return None
        if (head_dim * elem_bytes) % _TMA_ALIGN or (d_v * elem_bytes) % _TMA_ALIGN:
            return None
        pad = 0  # TMA deposits dense — the hardware swizzle replaces the row pad
    else:
        pad = _STREAM_PAD_ELEMS
    slot_bytes = bn * (head_dim + d_v + pad) * elem_bytes
    if stage.split:
        q_bytes = qk_tile.tile_m * (head_dim + _STREAM_Q_PAD_ELEMS) * elem_bytes
        if q_bytes + slot_bytes > budget:
            return None
        # Single-slab by construction (``Stage`` rejects a deeper split) — the overlap comes from
        # the phase split, and the ldmatrix ping-pong has no slot to alternate over.
        return replace(stage, reg_depth=1, bk_elems=bn)
    if slot_bytes > budget:
        return None
    # ``reg_depth`` <= 2: the streaming drains support the two-slot ldmatrix ping-pong (the next
    # atom-K step's B fragments load while the current step's mmas consume, breaking the WAR hazard
    # on the fragment registers); deeper ping-pongs cost registers the fm chains do not have.
    return replace(stage, depth=clamp_depth(stage, slot_bytes, budget), reg_depth=min(stage.reg_depth, 2), bk_elems=bn)


def resolve_scalar_stage(c: Fold, tile: TilePlan, stage: Stage, inputs, budget: int) -> Stage | None:
    """Resolve an operand ``Stage`` against the scalar register-tile contraction ``c``, or ``None``
    (gmem-direct). The slab K-chunk ``bk_elems`` is DERIVED to fit ``depth`` operand slots in the
    smem ``budget`` (the largest offered chunk dividing K) — not codec-spelled, so no schema change;
    when no chunk fits at the requested depth the depth steps down, single-buffer last."""
    if stage.transport not in ("tma", "cp.async") or not c.axis.extent.is_static:
        return None
    if stage.split and stage_split_groups(c) is not None:
        return None
    # A masked-N B-slab fill would clamp a chunk-start column into a row-crossing gmem address and
    # hang on the misaligned copy; a transposed B has no scalar drain variant (the warp tier stages
    # it into an N-major slab).
    if tile.n.mask or c.b_trans:
        return None
    if not inputs or not isinstance(c.a, Load) or not isinstance(c.b, Load) or c.a.input not in inputs:
        return None
    # 1-byte (fp8) elements decline: the fill's chunk-width and alignment math below is written
    # for the 2/4-byte dtypes and is unaudited at nbytes == 1 — refusing keeps the tier
    # gmem-direct (correct, converts per element) instead of risking a mis-sized slab.
    if any(t is not None and t.dtype.nbytes < 2 for t in (inputs.get(c.a.input), inputs.get(c.b.input))):
        return None
    if stage.transport == "tma" and not (
        _tma_operand_rank(c.a.index, tile.m.axis.name, c.axis.name) and _tma_operand_rank(c.b.index, tile.n.axis.name, c.axis.name)
    ):
        return None
    # Staging needs the CTA to BE one (tile_m x tile_n) output tile (the cooperative fill / drain
    # contract). A register-only tile launches the scalar default block over unrelated cells.
    if tile.launch_threads is None:
        return None
    if stage.transport == "tma" and max(tile.m.tile, tile.n.tile) > _TMA_MAX_BOX:
        return None
    k = c.axis.extent.as_static()
    elem_bytes = inputs[c.a.input].dtype.nbytes
    # Every staged transport needs 16 B-aligned inner global strides — A's is K, B's is N.
    n_ext = tile.n.axis.extent
    if not n_ext.is_static or (k * elem_bytes) % _TMA_ALIGN or (n_ext.as_static() * elem_bytes) % _TMA_ALIGN:
        return None
    b_bytes = inputs[c.b.input].dtype.nbytes if c.b.input in inputs else elem_bytes
    depth, bk_elems = max(1, stage.depth), 0
    while depth >= 1:
        cap = budget // (depth * max(1, tile.m.tile * elem_bytes + tile.n.tile * b_bytes))
        bk_elems = next((v for v in (128, 64, 32, 16, 8, 4) if v <= cap and k % v == 0), 0)
        if bk_elems >= 4:
            break
        depth -= 1
    if bk_elems < 4:
        return None
    return replace(stage, depth=depth, reg_depth=1, bk_elems=bk_elems)


__all__ = [
    "computed_a_cover",
    "computed_operand_cover",
    "computed_operand_copy_dtype",
    "enforce",
    "fragment_epilogue",
    "coop_band_layout",
    "producer_band",
    "producer_transport",
    "resolve_scalar_stage",
    "resolve_sync_stage",
    "resolve_twisted_stage",
    "resolve_warp_stage",
    "scalar_block_threads",
    "splitk_materialized_b",
    "splitk_slice_k_step",
    "splitk_width",
    "splitkv_slice",
    "stage_split_groups",
    "stage_target",
    "strip_width",
    "twisted_atom",
    "twisted_block",
    "twisted_sites_agree",
    "twisted_warp_columns",
    "warp_k_step",
    "warp_atom_edges",
    "warp_atom_stage",
    "warp_atom_target",
    "warp_a_columns",
    "warp_operand_dtype",
]
