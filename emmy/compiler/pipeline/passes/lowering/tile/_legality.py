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
pick, the atom ladder and the stage catalog are the schedule's; this module only ever refuses.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.schedule import ReducePlan, Stage, TilePlan, WarpSpec
from emmy.compiler.ir.stmt import Body, Load, Loop
from emmy.compiler.ir.stmt.passes import has_contraction_tail
from emmy.compiler.ir.tile import Fold
from emmy.compiler.ir.tile.ir import operand_name
from emmy.compiler.ir.tile.ops import cone_seam
from emmy.compiler.pipeline.search.space import MAX_BLOCK_THREADS, WARP_LANES

# TMA hardware: every box dim must fall in 1..256, and the swizzle-split box caps the operand rank
# at 4 so it stays within the 5-dim limit.
_TMA_MAX_BOX = 256
_TMA_ALIGN = 16  # the NONE-swizzle box-copy rule: 16 B-aligned inner dim and inner global stride

# cp.async needs a >= 4 B contiguous chunk, so a 2 B/elem slab's inner dim must be even.
_CP_ASYNC_MIN_ELEMS = 2


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


def producer_band(spec: WarpSpec, block_threads: int | None) -> str | None:
    """A dedicated producer band adds ``32·aux`` threads ON TOP of the compute warps. Two budgets:
    the total fits the CTA limit, and the band does not outnumber the compute half.

    The only rule in this module that is NOT a bound — it mixes a sum with a product, which no
    coordinate change linearizes — so it stays an ordinary arithmetic predicate.
    """
    aux = WARP_LANES * spec.aux_warps
    if block_threads is None:
        return "a producer band needs a launch-thread count; this tile has none (register-only)"
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


def warp_k_step(node: Fold, plan: TilePlan) -> str | None:
    """The inner mma K-step ``atom_k·bk`` must tile a STATIC contraction K: the warp K-loop has no
    static-K tail masking, so a partial final step reads past the operand and silently corrupts the
    result. A SYMBOLIC K reaches the masked tier and is fine."""
    ext = node.axis.extent
    if not ext.is_static:
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
    epilogue stmt defined (an embedding gather) cannot be threaded through it."""
    defs: set[str] = set()
    for s in epilogue:
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


def _warp_cp_async(k_axis: Axis, tile_n: int, bk_elems: int, mask_n: bool, b_trans: bool) -> bool:
    """cp.async staging: a STATIC, tile-divisible K, an unmasked N, and an even inner slab dim."""
    if mask_n or not k_axis.extent.is_static:
        return False
    if k_axis.extent.as_static() % bk_elems:
        return False
    return bk_elems % _CP_ASYNC_MIN_ELEMS == 0 and (b_trans or tile_n % _CP_ASYNC_MIN_ELEMS == 0)


def _warp_tma(k_axis: Axis, n_axis: Axis, tile_n: int, bk_elems: int, elem_bytes: int, mask_n: bool, b_trans: bool) -> bool:
    """TMA staging: STATIC tile-divisible K and N, and 16 B-aligned inner dims. A transposed B boxes
    N-major, so N drops out of the alignment gate."""
    if mask_n or not (k_axis.extent.is_static and n_axis.extent.is_static):
        return False
    k, n = k_axis.extent.as_static(), n_axis.extent.as_static()
    if k % bk_elems:
        return False
    inner = (bk_elems, k) if b_trans else (bk_elems, tile_n, k, n)
    return all((x * elem_bytes) % _TMA_ALIGN == 0 for x in inner)


def resolve_warp_stage(c: Fold, tile: TilePlan, stage: Stage, budget: int) -> Stage | None:
    """Resolve an operand ``Stage`` against the warp (mma) contraction ``c`` — TMA > cp.async >
    gmem-direct (``None``). The resolved stage carries ``bk_elems``, ``depth`` clamped so the ring's
    slots fit ``budget``, and ``reg_depth`` clamped to ``bk``. A tile whose single depth-1 slot
    already exceeds ``budget`` DECLINES — unlike the scalar resolver it cannot shrink the slab.

    A resolver rather than a predicate because the legal answer is a SIZE, not a yes/no: this is the
    one enforcement point for the smem budget, and returning the largest legal stage is what keeps
    an over-budget row out of the fork instead of failing at materialization.
    """
    if stage.split:
        return None  # one multiply consumes both edges — there is one transport group, nothing to cut
    atom = tile.atom
    a_nbytes = atom.operand_dtype("a").nbytes
    bk_elems = tile.bk * atom.atom_k
    m, n = tile.m, tile.n
    rank_ok = (
        isinstance(c.a, Load)
        and isinstance(c.b, Load)  # a descriptor needs a gmem address on BOTH edges
        and _tma_operand_rank(c.a.index, m.axis.name, c.axis.name)
        and _tma_operand_rank(c.b.index, n.axis.name, c.axis.name)
    )
    box_ok = max(m.tile, n.tile, bk_elems) <= _TMA_MAX_BOX
    tma_ok = stage.transport == "tma" and rank_ok and box_ok and _warp_tma(c.axis, n.axis, n.tile, bk_elems, a_nbytes, n.mask, c.b_trans)
    cp_ok = stage.transport == "cp.async" and _warp_cp_async(c.axis, n.tile, bk_elems, n.mask, c.b_trans)
    if not (tma_ok or cp_ok):
        return None
    slot_bytes = (m.tile + n.tile) * bk_elems * a_nbytes
    if slot_bytes > budget:
        return None
    depth = min(stage.depth, budget // slot_bytes)
    return replace(stage, depth=depth, reg_depth=min(stage.reg_depth, tile.bk), bk_elems=bk_elems)


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


def computed_a_cover(c: Fold, tile: TilePlan) -> str | None:
    """The geometry a COMPUTED ``a`` edge's compute fill demands: a STATIC contraction K (the
    ``_staged`` driver reads one) and an N width that EXACTLY covers the output columns (the sync B
    fill has no N clamp). A masked / symbolic M is fine — the fill σ clamps the overhanging rows and
    the ``RegStore`` guard discards their store."""
    if not c.axis.extent.is_static:
        return "a computed A edge needs a static contraction K — the sync compute-fill has no K mask"
    if tile.n.mask:
        return (
            f"a computed A edge needs a TILE whose N width exactly covers the static output columns "
            f"(N={tile.n.axis.extent}, no N mask in the sync compute-fill); pick a dividing tile."
        )
    return None


def resolve_sync_stage(c: Fold, tile: TilePlan, budget: int, want_depth: int = 1) -> Stage | None:
    """The ``sync`` compute-fill :class:`Stage` for a **computed-A** warp contraction under ``tile``
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
    bk_elems = tile.bk * atom.atom_k
    a_nbytes = atom.operand_dtype("a").nbytes
    _, _, stats = cone_seam(c.a)
    a_bytes = tile.m.tile * bk_elems * a_nbytes
    b_bytes = len(c.channels) * tile.n.tile * bk_elems * a_nbytes
    stat_bytes = len(stats) * tile.m.tile * 4
    if a_bytes + b_bytes + stat_bytes > budget:
        return None
    depth = want_depth if want_depth >= 2 and a_bytes + stat_bytes + want_depth * b_bytes <= budget else 1
    return Stage(depth=depth, transport="sync", smem=(operand_name(c.a),), bk_elems=bk_elems)


def resolve_scalar_stage(c: Fold, tile: TilePlan, stage: Stage, inputs, budget: int) -> Stage | None:
    """Resolve an operand ``Stage`` against the scalar register-tile contraction ``c``, or ``None``
    (gmem-direct). The slab K-chunk ``bk_elems`` is DERIVED to fit ``depth`` operand slots in the
    smem ``budget`` (the largest offered chunk dividing K) — not codec-spelled, so no schema change;
    when no chunk fits at the requested depth the depth steps down, single-buffer last."""
    if stage.split or stage.transport not in ("tma", "cp.async") or not c.axis.extent.is_static:
        return None
    # A masked-N B-slab fill would clamp a chunk-start column into a row-crossing gmem address and
    # hang on the misaligned copy; a transposed B has no scalar drain variant (the warp tier stages
    # it into an N-major slab).
    if tile.n.mask or c.b_trans:
        return None
    if not inputs or not isinstance(c.a, Load) or not isinstance(c.b, Load) or c.a.input not in inputs:
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
    "enforce",
    "fragment_epilogue",
    "coop_band_layout",
    "producer_band",
    "producer_transport",
    "resolve_scalar_stage",
    "resolve_sync_stage",
    "resolve_warp_stage",
    "scalar_block_threads",
    "splitk_materialized_b",
    "splitk_slice_k_step",
    "splitk_width",
    "strip_width",
    "warp_k_step",
    "warp_operand_dtype",
]
