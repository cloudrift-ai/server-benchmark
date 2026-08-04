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
from emmy.compiler.ir.schedule import Stage, TilePlan, WarpSpec
from emmy.compiler.ir.stmt import Body, Load
from emmy.compiler.ir.tile import Fold
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
    slots fit ``budget`` (dropping ``ring`` when the clamp leaves nothing to cycle) and ``reg_depth``
    clamped to ``bk``. A tile whose single depth-1 slot already exceeds ``budget`` DECLINES — unlike
    the scalar resolver it cannot shrink the slab.

    A resolver rather than a predicate because the legal answer is a SIZE, not a yes/no: this is the
    one enforcement point for the smem budget, and returning the largest legal stage is what keeps
    an over-budget row out of the fork instead of failing at materialization.
    """
    if stage.alt:
        return None  # the alternating single-slab pipeline is the warp-flash stream's
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
    return replace(stage, depth=depth, ring=stage.ring and depth >= 2, reg_depth=min(stage.reg_depth, tile.bk), bk_elems=bk_elems)


def resolve_scalar_stage(c: Fold, tile: TilePlan, stage: Stage, inputs, budget: int) -> Stage | None:
    """Resolve an operand ``Stage`` against the scalar register-tile contraction ``c``, or ``None``
    (gmem-direct). The slab K-chunk ``bk_elems`` is DERIVED to fit ``depth`` operand slots in the
    smem ``budget`` (the largest offered chunk dividing K) — not codec-spelled, so no schema change;
    when no chunk fits at the requested depth the depth steps down, single-buffer last."""
    if stage.alt or stage.transport not in ("tma", "cp.async") or not c.axis.extent.is_static:
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
    return replace(stage, depth=depth, ring=stage.ring and depth >= 2, reg_depth=1, bk_elems=bk_elems)


__all__ = [
    "enforce",
    "fragment_epilogue",
    "producer_band",
    "resolve_scalar_stage",
    "resolve_warp_stage",
    "scalar_block_threads",
    "splitk_materialized_b",
    "splitk_slice_k_step",
    "splitk_width",
    "strip_width",
    "warp_k_step",
]
