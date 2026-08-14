"""Geometry-only cold leads for scalar reduction schedule rows.

The deploy evidence hierarchy lives in :mod:`greedy`; helpers here only rank a set of
already-legal, complete schedule rows when that caller has established there is no golden,
measured, or calibrated-online decision.
"""

from __future__ import annotations

import math


def cold_grid_reduction_lead(rows: list[dict], ctx) -> int | None:
    """Choose the cold transposed reduction partition from effective output geometry.

    A one-row contraction whose B reading cannot form a target atom falls to the scalar
    reduction tier.  ``coop-t`` maps each warp across contiguous output columns and the
    remaining thread groups across K; for a long K, one CTA still has a large serial tail.
    A deferred grid split shortens that tail without atomics and combines the partials in a
    compact output-sized kernel.  Conversely, a symbolic outer dimension can make the static
    histogram look like that one-row case while its runtime hint already supplies many waves of
    output CTAs.  In that regime an unsplit transposed row lets placement finish recomposing a
    materialized product/reduction pair into its staged contraction, instead of prematurely
    freezing the scalar split and adding a synthetic grid dimension.

    The lead is derived from the offered rows and continuous geometry only:

    * hint-resolved free geometry when available, with one reduce axis;
    * reduction work at least as wide as the output, and enough output values to amortize one
      cooperative CTA;
    * no target-legal warp/MMA row in the same offer;
    * a thread ``WORK`` + transposed cooperative reduce;
    * for a one-effective-dimension matvec, a kernel-finalized grid split with at least one K
      element per cooperative thread after the split;
    * for a multi-dimensional output, no grid split once its output groups fill the
      thread-limited resident CTA inventory.

    Among qualifying rows, maximize useful K parallelism up to that last bound, then minimize
    split-workspace multiplicity.  There is no GPU, model, axis-name, or fixed shape gate.
    """
    from emmy.compiler.ir.schedule import ReducePlan, Workers, resolve_site_tile  # noqa: PLC0415
    from emmy.compiler.pipeline.knob import canonical_row_key, family_value  # noqa: PLC0415

    if not rows or float(rows[0].get("H_opt", 3.0) or 3.0) != 3.0:
        return None
    base = rows[0]
    if int(float(base.get("S_ext_n_reduce_axis", 0.0) or 0.0)) != 1:
        return None
    free_prod = float(base.get("S_hint_free_prod", base.get("S_ext_free_prod", 0.0)) or 0.0)
    free_max = float(base.get("S_hint_free_max", base.get("S_ext_free_max", 0.0)) or 0.0)
    reduce_k = float(base.get("S_ext_reduce_max", 0.0) or 0.0)
    sm_count = float(base.get("H_sm_count", 0.0) or 0.0)
    if free_prod <= 0 or free_max <= 0 or sm_count <= 0:
        return None
    one_effective_free = free_prod == free_max
    if one_effective_free and reduce_k < free_prod:
        return None

    parsed: list[tuple[int, dict, Workers | None, ReducePlan]] = []
    for i, row in enumerate(rows):
        try:
            work = Workers.parse(str(row.get("WORK") or ""))
            reduce_spec = family_value(row, "REDUCE")
            reduce = ReducePlan.parse(str(reduce_spec or ""), work)
            tile_spec = family_value(row, "TILE")
            tile = resolve_site_tile(str(tile_spec or ""), work, reduce.coop)
        except ValueError:
            continue
        if tile.is_warp and tile.atom.available_on(ctx):
            return None
        parsed.append((i, row, work, reduce))

    ranked: list[tuple[tuple, int]] = []
    for i, row, work, reduce in parsed:
        if work is None or work.kind != "thread" or work.count > ctx.max_threads_per_cta:
            continue
        if work.count < ctx.warp_size or work.count % ctx.warp_size:
            continue
        if free_prod < work.count:
            continue  # the compact finalize cannot amortize one cooperative worker inventory
        if not reduce.coop_transposed or reduce.coop != work.count:
            continue
        if family_value(row, "STAGE") or family_value(row, "RASTER"):
            continue

        output_groups = math.ceil(free_prod / ctx.warp_size)
        if not one_effective_free:
            # Thread count bounds how many such CTAs can reside concurrently even before
            # registers/smem narrow occupancy.  Once the natural output covers that upper bound,
            # a cold split cannot unlock K parallelism the card was missing; retain the compact
            # unsplit row.  Goldens, measured rows, and pins never call this helper.
            resident_ctas = sm_count * max(1, ctx.max_threads_per_cta // work.count)
            if output_groups < resident_ctas or reduce.needs_split or reduce.reg > 1:
                continue

            def l2_ratio(a: float, b: float) -> float:
                return abs(math.log2(max(a, 1.0) / max(b, 1.0)))

            rank = (
                l2_ratio(work.count, min(ctx.max_threads_per_cta, reduce_k)),
                canonical_row_key(row),
            )
            ranked.append((rank, i))
            continue

        if reduce.cta <= 1 or reduce.finalize != "kernel" or reduce.reg > 1:
            continue

        k_groups = reduce.coop / ctx.warp_size
        useful_parallel = k_groups * reduce.cta
        target_parallel = reduce_k / ctx.warp_size
        if useful_parallel > target_parallel:
            continue  # fewer than one K item per cooperative thread
        launch_ctas = output_groups * reduce.cta

        def l2_ratio(a: float, b: float) -> float:
            return abs(math.log2(max(a, 1.0) / max(b, 1.0)))

        rank = (
            0 if launch_ctas >= sm_count else 1,
            l2_ratio(useful_parallel, target_parallel),
            reduce.cta,  # equal useful parallelism: the smaller workspace wins
            l2_ratio(work.count, min(ctx.max_threads_per_cta, reduce_k)),
            canonical_row_key(row),
        )
        ranked.append((rank, i))
    return min(ranked)[1] if ranked else None


__all__ = ["cold_grid_reduction_lead"]
