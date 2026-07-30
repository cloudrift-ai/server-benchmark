"""The online-prior featurizers — every knob-dict → feature-vector encoding, in one file.

:func:`knob_features` is the single featurizer over a whole knob dict (the ``D_*`` engineered
geometry / occupancy family, the ``MMA_*`` atom expansion, the ``S_*`` / ``H_*`` pass-throughs);
:func:`tile_signature` is the schema-agnostic structural identity used to join golden YAML rows
against enumerated candidates. Lives in the same package as :mod:`.space` so the whole search space
(dimensions × values × encoding) is analyzable in one place; the ``Knob`` descriptor / registry /
env plumbing stays in :mod:`~emmy.compiler.pipeline.knob`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from emmy.compiler.pipeline.knob import (
    _AXIS_FAMILIES,
    CTX_PREFIX,
    STRUCT_PREFIX,
    KnobType,
    axis_of,
    family_of,
    family_value,
    get,
)

# Version of the knob vocabulary + feature encoding this module reads. Rows recorded under a
# different version featurize to garbage (the 2026-07 tile-IR rebuild replaced ``BM/BN/FM/FN/…``
# with the ``TILE``/``STAGE``/``REDUCE`` codecs; a pre-rebuild reservoir scored by this featurizer
# collapses to constant predictions — worse than random). Every persisted training artifact (the
# prior checkpoint's reservoir, the autotune DB's ``node`` rows) is stamped with this version, and
# readers drop rows from another version. Version 1 is the retired pre-rebuild vocabulary (old
# artifacts carry no stamp and default to it). Version 2 is the retired blind encoding of the
# codec vocabulary: the warp ``TilePlan.bk`` never reached the features, ``_free_slots`` sorted
# the warp grid wide-is-n (transposed siblings collapsed; ``tile_m``/``tile_n`` mislabelled),
# warp rows dropped the split-K finalize letter, and the ``STAGE`` ``alt`` / ``REDUCE`` ``b<n>t``
# letters were unfeaturized — same raw knobs, different emitted VALUES, so artifacts fit on v2
# are semantically stale.
# Bump on any incompatible knob-spelling or feature-encoding change.
FEATURIZER_VERSION = 3


def masked_axis_features(*, m: bool = False, n: bool = False, k: bool = False) -> dict[str, float]:
    """The per-role boundary-masked structural features (``S_masked_m/n/k``).

    A tile boundary-masks an output / reduce axis when the extent is symbolic or a
    static non-divisor of the chosen tile — a *consequence* of the shape/tile
    pairing, not a tunable choice — so it belongs with the ``S_`` structural
    identity, not a tuning knob. Masking is only known once the tile geometry is
    chosen, so the producers stamp it at materialize / enumeration time; the
    feature definition lives here, beside the featurizer that reads it
    (``_geom_feats`` → ``D_neg_masked_*``).

    Split per role so the prior can learn that K-masking (SYNC-pinned, ring-declined) prices
    differently from M / N output masking. Only the masked roles are emitted — an unmasked kernel
    carries none, so its structural identity is unchanged and the featurizer defaults a missing flag
    to ``0.0``. ``S_masked_*`` pass through :func:`knob_features` as raw floats via the
    ``STRUCT_PREFIX`` branch automatically."""
    feats: dict[str, float] = {}
    if m:
        feats[f"{STRUCT_PREFIX}masked_m"] = 1.0
    if n:
        feats[f"{STRUCT_PREFIX}masked_n"] = 1.0
    if k:
        feats[f"{STRUCT_PREFIX}masked_k"] = 1.0
    return feats


def mma_atom(knobs: dict) -> str | None:
    """The concrete tensor-core atom-kind name carried by ``knobs``, or ``None`` for the scalar
    tier (no warp fragment).

    The atom is named by the warp form of the unified ``TILE`` codec (``a:<atom>/…``): a ``TILE``
    value carrying an ``a:<atom>`` token is the warp fragment, and its parsed :class:`TilePlan`'s
    ``atom`` names it. A scalar ``TILE`` (``n../f..`` or empty) names no atom → ``None``."""
    from emmy.compiler.ir.schedule import TilePlan, is_warp_codec  # noqa: PLC0415

    spec = family_value(knobs, "TILE")
    if not is_warp_codec(spec):
        return None
    try:
        return TilePlan.parse(spec).atom.name
    except ValueError:
        return None


def is_warp(knobs: dict) -> bool:
    """True if ``knobs`` is a warp-tier (tensor-core MMA) variant — i.e. it
    names a concrete atom kind. The single tier discriminator shared by the tile
    passes, the scorer, and the featurizer."""
    return mma_atom(knobs) is not None


def _stage_features(knobs: dict) -> dict[str, float]:
    """Engineered ``D_*`` features for the operand-staging decision (the ``STAGE`` codec
    ``d<depth>/sync|cp|tma[/ring]``). The prior prices the smem pipeline: a deeper / async
    transport trades smem footprint + a fill prologue for K-loop overlap. Absent / empty
    ``STAGE`` (the gmem-direct baseline) contributes nothing (``{}``); a present codec emits
    the pipeline depth and a small transport one-hot so the model separates the synchronous
    smem copy from cp.async from TMA. Read schema-agnostically off the raw codec, exactly as
    ``_reduce_decomp`` reads ``REDUCE`` — so a ``d2/cp`` stage featurizes identically on a
    scalar (``TILE``) and a warp (``WARP``) contraction (the cross-kind feature transfer)."""
    spec = family_value(knobs, "STAGE")
    if not spec:
        return {}
    from emmy.compiler.ir.schedule import Stage  # noqa: PLC0415

    try:
        st = Stage.parse(spec)
    except ValueError:
        return {}
    return {
        "D_stage_depth": float(st.depth),
        "D_stage_async": 1.0 if st.is_async else 0.0,
        "D_stage_tma": 1.0 if st.transport == "tma" else 0.0,
        "D_stage_ring": 1.0 if st.ring else 0.0,
        "D_stage_reg_depth": float(st.reg_depth),  # smem→register double-buffer (p<n>)
        # The alternating single-slab pipeline (``/alt`` — the flash stream's FA-2 choreography,
        # which also stages Q): enumerated as a sibling of the paired ring on flash rows, so
        # without this flag ``d1/tma/alt`` featurized byte-identically to plain ``d1/tma``.
        "D_stage_alt": 1.0 if st.alt else 0.0,
    }


# Per-node structural features the featurizer reads per axis-group (``S_ext_reduce_prod@<axis>`` etc):
# the reduce/free extents + masking that a node's geometry featurizer needs. On a **one-node** kernel
# these are stamped bare (one reduce axis → one ``S_ext_*``); a multi-node kernel (flash) stamps them
# addressed so each node reads its own extents. The slice builder reads ``@<axis>`` first, falling back
# to the bare key — so a one-node kernel (bare stamp) featurizes byte-identically.
_NODE_STRUCT_BASES = (
    "S_ext_reduce_prod",
    "S_ext_reduce_max",
    "S_ext_free_prod",
    "S_masked_m",
    "S_masked_n",
    "S_masked_k",
)


def _node_axes(knobs: dict) -> list[str | None]:
    """The schedule-bearing nodes' axes, in first-seen order — one per distinct ``@<axis>`` element
    across the per-node schedule families (``TILE`` / ``REDUCE`` / ``STAGE``). ``[None]`` (one bare
    node) when the schedule families are ALL bare (goldens / single-node canonical rows); a MIXED
    row — the phase-3 canonical flash spelling: ``TILE@dd`` / ``TILE@pj`` beside bare ``REDUCE`` /
    ``STAGE`` for the primary stream — appends the ``""`` bare-remainder group so the primary
    node's slices keep contributing to the sum-pool (byte-identical to the retired ``@kv``
    spelling's own group). ``[]`` when the kernel carries no schedule codec at all (a pure
    pointwise ``Map``)."""
    axes: list[str] = []
    seen: set[str] = set()
    has_bare = False
    for k in knobs:
        if family_of(k) not in _AXIS_FAMILIES:
            continue
        ax = axis_of(k)
        if ax is None:
            has_bare = True
        elif ax not in seen:
            seen.add(ax)
            axes.append(ax)
    if axes:
        return [*axes, ""] if has_bare else list(axes)
    return [None] if has_bare else []


def _node_slice(knobs: dict, axis: str | None) -> dict:
    """The single-node ``knobs`` sub-dict the geometry featurizers see for the node keyed ``axis``:
    that node's ``FAMILY@<axis>`` schedule codecs plus the shared ``S_*`` / ``H_*`` context,
    with any addressed per-node structural feature (``S_ext_reduce_prod@<axis>``) substituted in bare so
    ``_geom_feats`` reads the node's own extents. ``axis is None`` (the bare single node) returns
    ``knobs`` unchanged — the whole dict is that one node (byte-identical to the pre-loop
    featurizer); ``axis == ""`` is the mixed row's bare-remainder group — the BARE schedule
    families (the phase-3 primary node) plus the context."""
    if axis is None:
        return knobs
    # The shared structural / regime context (bare ``S_*`` / ``H_*``).
    sub: dict = {k: v for k, v in knobs.items() if k.startswith((STRUCT_PREFIX, CTX_PREFIX)) and "@" not in k}
    if axis == "":
        for fam in _AXIS_FAMILIES:
            if fam in knobs:
                sub[fam] = knobs[fam]
        return sub
    for fam in _AXIS_FAMILIES:
        key = f"{fam}@{axis}"
        if key in knobs:
            sub[key] = knobs[key]
    for base in _NODE_STRUCT_BASES:  # addressed per-node override; bare fallback already copied above
        addressed = knobs.get(f"{base}@{axis}")
        if addressed is not None:
            sub[base] = addressed
    return sub


def _schedule_node_features(node_knobs: dict) -> dict[str, float]:
    """The per-node schedule-geometry ``D_*`` / ``MMA_*`` feature block for ONE node's ``knobs`` slice
    (its ``TILE`` / ``REDUCE`` / ``STAGE`` codecs + the node's structural context). Reads every codec
    via :func:`family_value`, so a bare and a suffixed key featurize identically. ``MMA_tier`` is left
    unset for a scalar node (the caller defaults it to ``0.0`` once, after pooling)."""
    from emmy.compiler.ir.schedule import TilePlan, is_warp_codec  # noqa: PLC0415

    feats: dict[str, float] = {}
    # Atom (tensor-core cell) features. The warp fragment names its atom on the ``TILE`` codec
    # (``a:<atom>``); expand its physical cell / dtype properties into the ``MMA_*`` family the priors
    # rank on. A scalar ``TILE`` names no atom → no ``MMA_tier`` here (the caller's default fills it).
    tile_spec = family_value(node_knobs, "TILE")
    if is_warp_codec(tile_spec):
        try:
            feats.update(_atom_features(TilePlan.parse(tile_spec).atom))
        except ValueError:
            pass
    tile_feats = _tile_features(node_knobs)
    feats.update(tile_feats)
    # Warp-tier occupancy: the scalar ``_tile_features`` above models the thread tile (``BN·BM``) and
    # skips warp rows, so compute the SAME ``D_*`` family from the warp tile geometry instead — using
    # the atom cell dims read off the parsed warp ``TILE`` codec. Shared ``D_*`` names across tiers let
    # the prior learn occupancy / CTA-count uniformly.
    if is_warp(node_knobs):
        feats.update(_warp_tile_features(node_knobs))
    elif not tile_feats:
        # TILE-less row (a pure reduce kernel, or a contraction's REDUCE fork before the tile is
        # decided): the REDUCE codec must still featurize, or the row is indistinguishable from its
        # siblings — see ``_reduce_features``.
        feats.update(_reduce_features(node_knobs))
    feats.update(_stage_features(node_knobs))  # operand-staging pipeline (STAGE codec); {} when gmem-direct
    # TMA-conditioned tile pricing: TMA staging only enumerates where the hardware offers it
    # (Hopper/Blackwell), so a geometry term gated on ``D_stage_tma`` is where one weight set
    # prices those cards' tiles separately — no per-arch split needed. The 2026-07-09 5090
    # sweep showed the golden TMA tiles want narrower/squarer warp grids and wider splits than
    # the shared weights choose (TILE match 0/17); these give the fit that axis. Emitted only
    # on a TMA-staged row (skip-if-missing 0.0 elsewhere).
    if feats.get("D_stage_tma"):
        for src, dst in (
            ("D_aspect", "D_tma_aspect"),
            ("D_log2_area", "D_tma_log2_area"),
            ("D_w_grid_m", "D_tma_grid_m"),
            ("D_w_grid_n", "D_tma_grid_n"),
        ):
            if src in feats:
                feats[dst] = feats[src]
        if "D_splitk" in feats:
            feats["D_tma_l2_splitk"] = math.log2(max(feats["D_splitk"], 1.0))
    return feats


def knob_features(knobs: dict) -> dict[str, float]:
    """Convert a knob dict into a flat numeric feature vector for the planner priors — the single
    featurizer over the whole dict.

    - ``STRUCT_PREFIX`` (``S_``) structural-feature knobs and ``CTX_PREFIX``
      (``H_``) host/hardware-regime knobs pass through as floats: they already
      are the kernel's structural / regime feature set.
    - Registered tuning ``Knob``s are encoded by type: ``INT`` → float, ``BOOL``
      → 0/1, ``BINMASK`` (binary string) → ``{<name>_popcount, _width, _frac}``.
    - A ``Knob`` with a custom ``features`` callable dispatches through it — no
      per-knob special-casing here.
    - Unregistered, non-structural knobs are best-effort float-coerced (skipped
      when non-numeric); other ``STR`` knobs have no generic encoding.

    The schedule-geometry block (``D_*`` / ``MMA_*``) is featurized **per node** and **sum-pooled**: a
    multi-node kernel (flash) groups its ``FAMILY@<axis>`` codecs by axis (:func:`_node_axes`), slices
    each node's schedule + own structural extents (:func:`_node_slice`), featurizes it
    (:func:`_schedule_node_features`), and sums the blocks into the fixed-width vector. A single-node
    kernel has one group, so the sum is that one node's block — **byte-identical** to the pre-loop
    singleton featurizer (the migration is invisible until a kernel actually has two nodes). Per-node
    attribution / transfer is the gated per-node-predict follow-up; pool is the smallest change."""
    feats: dict[str, float] = {}
    for name, val in knobs.items():
        if name.startswith(STRUCT_PREFIX) or name.startswith(CTX_PREFIX):
            feats[name] = float(val)
            continue
        knob = get(name)
        if knob is not None and knob.unfeatured:
            continue  # unfeatured knob (cosmetic re-spell / umbrella gate): never a ranking feature
        if knob is not None and knob.features is not None:
            feats.update(knob.features(val))
            continue
        if knob is None:
            num = _coerce_float(val)
            if num is not None:
                feats[name] = num
            continue
        if knob.type is KnobType.INT:
            feats[name] = float(val)
        elif knob.type is KnobType.BOOL:
            feats[name] = 1.0 if _as_bool(val) else 0.0
        elif knob.type is KnobType.BINMASK:
            s = str(val)
            pop = float(s.count("1"))
            feats[f"{name}_popcount"] = pop
            feats[f"{name}_width"] = float(len(s))
            feats[f"{name}_frac"] = pop / len(s) if s else 0.0
        # STR knobs with no custom featurizer: no generic numeric encoding.
    # Per-node schedule geometry: featurize each schedule-bearing node's slice and sum-pool the blocks.
    for axis in _node_axes(knobs):
        for name, val in _schedule_node_features(_node_slice(knobs, axis)).items():
            feats[name] = feats.get(name, 0.0) + val
    feats.setdefault("MMA_tier", 0.0)  # scalar tier / no schedule node = no warp atom
    return feats


def _free_slots(knobs: dict) -> tuple[int, int, int, int] | None:
    """The ``(par_n, reg_n, par_m, reg_m)`` slot widths for the (≤2) tiled free axes.

    Both fragments source the free split from the single ``TILE`` codec. The **warp** fragment
    (``a:<atom>/w<WM>x<WN>/f<FM>x<FN>``) keeps the TRUE codec axes — ``(WN, FN)`` / ``(WM, FM)``
    verbatim: the mma atom is physically asymmetric (``atom_m ≠ atom_n``) and the enumeration
    offers both orientations (``w4x2`` and ``w2x4`` are different tiles), so any re-ordering here
    collapses transposed siblings into one feature vector / signature and mislabels ``tile_m`` /
    ``tile_n`` on every row whose wide slot is M. The **scalar** fragment
    (``n<N>[x<M>]/f<fn>[x<fm>]``) canonicalizes wide-is-``n`` (the coalesced slot) — a no-op for
    every enumerated row (the scalar grid spells ``par_n ≥ par_m``), kept so recorded scalar rows'
    historical encoding does not shift. A single free axis fills the ``n`` slot with a degenerate
    ``(1, 1)`` ``m`` slot. Returns ``None`` for a non-tiled scalar kernel (per-cell ``TILE``)."""
    from emmy.compiler.ir.schedule import TilePlan  # noqa: PLC0415

    spec = family_value(knobs, "TILE")
    try:
        tile = TilePlan.parse(spec)  # one parse for both fragments — the atom discriminates
    except ValueError:
        return None
    if not tile.is_tiled:
        return None
    if tile.is_warp:
        return tile.units_n, tile.reg_n, tile.units_m, tile.reg_m
    pairs = [(tile.units_n, tile.reg_n), (tile.units_m, tile.reg_m)]
    pairs.sort(key=lambda pr: (pr[0], pr[1]), reverse=True)  # wider par = the n slot
    (par_n, reg_n) = pairs[0]
    (par_m, reg_m) = pairs[1] if len(pairs) >= 2 else (1, 1)
    return par_n, reg_n, par_m, reg_m


@dataclass(frozen=True)
class _Decomp:
    """The reduce-axis decomposition factors the featurizer reads (``fold``/``cta``/``coop``)
    plus the cross-CTA ``finalize`` codec letter. The per-thread serial remainder is derived by
    the materializer (``ceil(extent / parallel)``), never spelled by the ``REDUCE`` codec, so it
    is NOT a field here — a ``serial`` field defaulting to 1 is what silently fed the warp
    K-chunk features for a year (the K-chunk lives on the ``TILE`` codec, ``TilePlan.bk``)."""

    fold: int = 1
    cta: int = 1
    coop: int = 1
    finalize: str = "atomic"
    # The ``b<n>t`` transposed cooperative band (k-major matvec lane mapping) — a different
    # kernel from the interleaved ``b<n>`` at the same width, so it must reach both the features
    # and the ``tile_signature`` identity (``b<n>t`` goldens are recorded in the per-GPU YAMLs).
    coop_transposed: bool = False


def _reduce_decomp(knobs: dict) -> _Decomp:
    """The primary reduce axis's ``(cta, coop, reg)`` partition factors, decoded from the
    single ``REDUCE`` codec knob (``g<n>`` cta / ``b<n>`` coop / ``r<n>`` reg — the reduce
    tier's one decomposition knob, decided in the ``_schedule`` helper). The ``serial``
    remainder is derived from the schedule (``ceil(extent / parallel)``), not a knob, so it
    stays the ``_Decomp`` default."""
    from emmy.compiler.ir.schedule import ReducePlan  # noqa: PLC0415

    plan = ReducePlan.parse(family_value(knobs, "REDUCE"))
    # ``finalize`` must be forwarded here AND by every ``_geom_feats`` caller: dropping it leaves
    # a default "atomic" in place, so ``D_finalize_kernel`` goes dead (0.0) on the affected rows
    # and the offline prior's atomic-free split interaction never fires (found scalar-side by the
    # 2026-07-07 reduce-featurization tests; the warp tier repeated the same drop until 2026-07-28).
    return _Decomp(fold=plan.reg, cta=plan.cta, coop=plan.coop, finalize=plan.finalize, coop_transposed=plan.coop_transposed)


def tile_signature(knobs: dict) -> tuple:
    """Schema-agnostic structural identity of a tile config: the free-axis slots, the slab
    K-chunk, the primary reduce decomposition, and the atom kind — read from the native codec
    knobs (``TILE`` / ``REDUCE`` / ``STAGE``, bare or ``@<axis>``-suffixed alike). Two configs
    with equal signatures are the same kernel variant whichever key form spelled them, so this
    is the bridge for matching a recorded golden YAML row against the native enumeration's
    candidate rows (``scripts/golden_knob_heuristics.py`` / ``search/golden_eval.evaluate_golden``).
    The K-chunk (``TilePlan.bk``) is part of the identity — without it every ``k<n>`` sibling in
    a warp pool joined ambiguously (a golden recorded at ``k4`` matched the ``k1`` candidate).
    Operand staging (the ``STAGE`` codec) is part of the identity — a staged and a gmem-direct
    config are different variants — but defaults to ``None`` when absent, so a golden recorded
    without a ``STAGE`` still matches a native unstaged candidate (both ``None``)."""
    return (_free_slots(knobs), _tile_bk(knobs), _reduce_decomp(knobs), mma_atom(knobs), _stage_sig(knobs))


def _tile_bk(knobs: dict) -> int:
    """The warp tile's slab K-chunk (``TilePlan.bk``, atom_k multiples) for ``tile_signature``;
    1 on the scalar tier / per-cell / unparseable rows (the scalar codec spells no K token)."""
    from emmy.compiler.ir.schedule import TilePlan  # noqa: PLC0415

    try:
        return TilePlan.parse(family_value(knobs, "TILE")).bk
    except ValueError:
        return 1


def _stage_sig(knobs: dict) -> tuple | None:
    """The structural staging identity ``(depth, transport, ring, alt)`` for ``tile_signature``,
    or ``None`` when ``STAGE`` is absent / empty (the gmem-direct baseline). ``alt`` (the flash
    stream's alternating single-slab pipeline) is a different variant from the paired ring, so it
    is part of the identity."""
    spec = family_value(knobs, "STAGE")
    if not spec:
        return None
    from emmy.compiler.ir.schedule import Stage  # noqa: PLC0415

    try:
        st = Stage.parse(spec)
    except ValueError:
        return None
    return (st.depth, st.transport, st.ring, st.alt)


def _geom_feats(
    knobs: dict,
    *,
    threads: int,
    cells: int,
    tile_m: int,
    tile_n: int,
    splitk: int,
    bn: int,
    bm: int,
    bk: int,
    br: int,
    free_prod,
    sm: float,
    warp: bool,
    finalize: str = "atomic",
) -> dict[str, float]:
    """The engineered ``D_*`` tile-geometry / occupancy feature family — the
    single featurization the priors rank on. It folds in everything the old
    hand-coded matmul heuristic scored (occupancy waves, tile-area / thread /
    aspect targets, the geometry "bands", K-chunk depth), so a fixed linear model
    over these features (:class:`~emmy.compiler.pipeline.search.prior.OfflinePrior`)
    reproduces that heuristic and the ``OnlinePrior`` sees the same
    derived signal a tree can't cheaply reconstruct from raw knobs + the *coarse*
    ``S_ext_*`` extents.

    Tier-aware: the "ideal" tile / thread targets differ between the scalar thread
    tile (256 threads, 8192-elem area) and the warp tile (128 threads = 4 warps,
    64×64 = 4096 area), selected by ``warp``. ``free_prod`` is the output free-dim
    product (``S_ext_free_prod``); when present the occupancy terms are added —
    ``#CTAs ≈ M·N / tile_area · SPLITK`` (ceil-free, needs only the product the
    ``S_*`` features carry, not the per-axis split). The ``BN``/``BM`` band
    features are the OFF sentinel ``0`` on a warp row (so they don't fire there);
    the K-chunk ``bk`` is a live knob on the warp tier only today (the ``TILE``
    codec's ``k<n>``, atom_k multiples — the scalar codec spells no K token, so
    the scalar ``D_*_bk`` bands stay 0), riding tier-split features (``D_*_bk``
    scalar vs ``D_w_*_bk`` warp) because the tiers pull opposite ways. The rest
    of the warp tier's signal rides the geometry / occupancy terms via the
    tier-aware targets."""

    def l2(x: float) -> float:
        return math.log2(max(float(x), 1.0))

    area = max(tile_m * tile_n, 1)
    reuse = area / (tile_m + tile_n) if (tile_m + tile_n) else 0.0
    aspect = l2(tile_m) - l2(tile_n)
    thr_target = 7.0 if warp else 8.0  # log2 threads: 128 (4-warp) vs 256
    area_target = 12.0 if warp else 13.0  # log2 area: 64×64=4096 vs 8192
    masked_m = float(knobs.get("S_masked_m", 0.0) or 0.0)
    masked_n = float(knobs.get("S_masked_n", 0.0) or 0.0)
    masked_k = float(knobs.get("S_masked_k", 0.0) or 0.0)
    k_ext = float(knobs.get("S_ext_reduce_prod") or 0.0)
    kchunks = max((k_ext / br) / bk, 1.0) if k_ext > 0 else 1.0
    out = {
        # core geometry
        "D_threads": float(threads),
        "D_cells": float(cells),
        "D_tile_m": float(tile_m),
        "D_tile_n": float(tile_n),
        "D_log2_area": l2(area),
        "D_reuse": reuse,
        "D_aspect": aspect,
        # offline (ex-heuristic) terms — tier-aware targets
        "D_l2_threads": l2(threads),
        "D_near_threads": -abs(l2(threads) - thr_target),
        "D_pow2_threads": 1.0 if threads > 0 and (threads & (threads - 1)) == 0 else 0.0,
        "D_cells_cap": min(float(cells), 128.0),
        "D_near_cells": -abs(float(cells) - 16.0),
        "D_near_area": -abs(l2(area) - area_target),
        "D_square": -abs(aspect),
        "D_l2_reuse": l2(reuse),
        "D_near_intensity": -abs(l2(reuse) - 5.0),
        "D_near_kchunks": -abs(l2(kchunks) - 5.0),
        # Per-role masked-tile penalties: split M / N / K so the prior can weight K-masking
        # distinctly. Negative = penalty.
        "D_neg_masked_m": -masked_m,
        "D_neg_masked_n": -masked_n,
        "D_neg_masked_k": -masked_k,
        # thread-tier geometry bands (raw BN/BM/BK/SPLITK; 0 on a warp row)
        "D_l2_bn": l2(bn),
        "D_l2_bm": l2(bm),
        "D_bn_ge_bm": 1.0 if bn > 0 and bn >= bm else 0.0,
        "D_bn_band": 1.0 if 16 <= bn <= 64 else 0.0,
        "D_bm_band": 1.0 if 8 <= bm <= 16 else 0.0,
        # BK bands are tier-specific: the scalar tile wants deep K-chunks (BK≥32)
        # while the warp / TMA tile wants a shallow pipelined BK≈2 — opposite
        # directions, so they ride separate features (one weight can't serve both).
        "D_l2_bk": 0.0 if warp else l2(bk),
        "D_bk_ge32": 0.0 if warp else (1.0 if bk >= 32 else 0.0),
        "D_w_l2_bk": l2(bk) if warp else 0.0,
        "D_w_near_bk": (-abs(l2(bk) - 1.0)) if warp else 0.0,
        "D_splitk": float(splitk),
        "D_splitk_le2": 1.0 if splitk <= 2 else 0.0,
        # Cross-CTA finalize fold (the REDUCE codec ``c`` field's letter): 1.0 = deferred
        # KERNEL combine (``c<cta>k``), 0.0 = in-place ATOMIC (``c<cta>a`` / bare). The
        # offline prior's split-K gate reads it.
        "D_finalize_kernel": 1.0 if (splitk > 1 and finalize == "kernel") else 0.0,
        "D_tilen_clean": 1.0 if tile_n in (32, 64, 128) else 0.0,
        "D_near_tilen": -abs(l2(tile_n) - 6.0),
        # A scalar tile on a warp-ELIGIBLE contraction (16-bit operands, atoms offered — the
        # scheduler's ``S_warp_eligible`` kernel stamp) competes against tensor cores: the
        # roofline bar none of the flat geometry terms can see. 0 on warp rows and on kernels
        # with no warp offer, so fp32 / non-contraction ranking is untouched.
        "D_scalar_on_warp_eligible": 1.0 if (not warp and float(knobs.get("S_warp_eligible", 0.0) or 0.0) > 0) else 0.0,
    }
    if free_prod:
        ctas = float(free_prod) / area * splitk
        waves = math.log2(max(ctas / sm, 1e-3))
        out["D_log2_ctas"] = l2(ctas)
        out["D_log2_waves"] = waves  # CTAs relative to SM count
        out["D_near_waves"] = -abs(waves - 1.0)  # target ~2 waves
        out["D_ctas_ge_sm"] = 1.0 if ctas >= sm else 0.0
        # Split-K beyond what occupancy needs is pure atomic/combine waste. The free
        # axes alone give ``free_ctas = free_prod/area`` CTAs; split-K is justified
        # only to lift that toward the ~2·SM ``D_near_waves`` target. The terms above
        # fold ``splitk`` straight into ``ctas``, so they CANNOT tell "≈2 waves via a
        # small tile" (golden, free) from "≈2 waves via heavy split-K on a big tile"
        # (atomic-bound) — both score the same waves / ctas≥sm. This credits split-K
        # up to the need and penalizes the excess, the engineered signal the online
        # prior needs to separate the SPLITK=1/2 goldens from the SPLITK=8/16 tiles
        # the -O1 sweep over-ranks (the offline prior already gets it via D_splitk_le2).
        free_ctas = float(free_prod) / area
        # Split-K is justified to (a) lift occupancy toward ~2 waves AND (b) hide the K-streaming
        # latency of a K-HEAVY GEMM — a long reduction per output tile parallelizes across the split
        # CTAs even when the free dims already fill the machine. The occupancy-only ``needed`` mispriced
        # exactly (b): mlp_down (M=512, N=4096, K=14336) has ``free_ctas`` alone saturating the SMs, so
        # it credited ZERO split and ranked the winning ``g8k`` golden ~9000-deep. ``k_ext /
        # sqrt(free_prod)`` is the K-per-output-linear-dim heaviness (config-independent; ≈1 for a
        # square / balanced GEMM, large only when K ≫ √(M·N)) — a second floor on the split the shape
        # justifies.
        occ_need = 2.0 * sm / max(free_ctas, 1.0)
        kheavy_need = k_ext / math.sqrt(max(float(free_prod), 1.0)) if k_ext > 0 else 1.0
        needed = max(occ_need, kheavy_need, 1.0)
        out["D_splitk_excess"] = math.log2(max(splitk / needed, 1.0))
        # The deficit side: UNDER-splitting a shape that justifies a wide split (``splitk < needed``)
        # is the K-heavy miss — the penalty that lifts the ``g<w>k`` golden over the ``g1`` tile the
        # occupancy terms alone rank as safe. Zero once ``splitk ≥ needed`` (and for every shape with
        # ``needed == 1``, so the well-tuned non-split geometries are untouched). Was absent: split-K
        # had only penalties (``D_splitk_le2`` / ``D_splitk_excess``), never a reward when justified.
        out["D_splitk_deficit"] = math.log2(max(needed / max(float(splitk), 1.0), 1.0))
        # The deferred split-K finalize (``g<w>k``) writes + re-reads a full free-size partial
        # workspace and launches the combine kernel — a round-trip volume the in-place atomic
        # ``g<w>a`` finalize does not pay.
        out["D_splitk_roundtrip"] = l2(free_prod) if out["D_finalize_kernel"] else 0.0
        # Register-tile intensity × occupancy interaction: a wide per-thread
        # register tile (big FM·FN) is a win only while the grid still covers
        # the SMs — the flat D_cells* terms can't express that, so the big-FM
        # goldens (square.2048's FM=26) rank deep under any sign the fit gives
        # them (2026-06-12 golden-sweep finding 2).
        out["D_l2_cells_occ"] = l2(cells) if ctas >= sm else 0.0
    return out


# The reduce-partition slice of the shared ``_geom_feats`` block — the keys a TILE-less row keeps.
# The tile-geometry keys (area / bands / aspect / cells) would be fabricated constants on a row
# with no tile; the thread / split-K / finalize / occupancy keys carry the real partition signal
# (``area=1`` makes ``#CTAs = free_prod·splitk`` — exactly the per-output-cell reduce grid).
_REDUCE_FEATURE_KEYS = (
    "D_threads",
    "D_l2_threads",
    "D_pow2_threads",
    "D_near_threads",
    "D_splitk",
    "D_splitk_le2",
    "D_splitk_excess",
    "D_finalize_kernel",
    "D_splitk_roundtrip",
    # The scalar-on-warp-eligible guard MUST travel with the bonus features: a per-cell contraction
    # leaf (TILE decided-OFF, REDUCE coop fold) is exactly a scalar row competing against tensor
    # cores, and granting it the thread/occupancy bonuses without this penalty made greedy deploy a
    # 1157us per-cell b256 kernel over the 3.5us mma golden (square.512.fp16, 2026-07-07 5090 gate).
    "D_scalar_on_warp_eligible",
    "D_log2_ctas",
    "D_log2_waves",
    "D_near_waves",
    "D_ctas_ge_sm",
)


def _reduce_features(knobs: dict) -> dict[str, float]:
    """Reduce-partition ``D_*`` features for a **TILE-less** row — a pure reduce kernel's rows
    (leaves included) and the partial rows at a contraction's REDUCE fork, where no tile is decided
    yet. Without this block the ``REDUCE`` codec featurized ONLY inside :func:`_tile_features` /
    :func:`_warp_tile_features`, so a TILE-less row's siblings (serial vs ``b64`` coop vs ``g2k``
    cross-CTA — 759 µs vs 17 µs subtrees) produced byte-identical vectors and NO prior could rank
    them — the 2026-07-07 cold-baseline REDUCE-regret finding (30–68x median across three cards).
    Runs the same :func:`_geom_feats` formulas with the tile degenerated to one output cell and
    keeps the :data:`_REDUCE_FEATURE_KEYS` slice; the ILP register fold (``r<n>``) rides its own
    ``D_reduce_ilp`` — serial and ``r4`` differ in neither threads nor split-K. Empty when the row
    carries no ``REDUCE`` family key, so pointwise rows stay feature-free as before. Additive
    encoding: raw knob dicts re-featurize at read time, so existing node rows / reservoirs gain
    these keys on the next fit — no ``FEATURIZER_VERSION`` bump."""
    if not any(family_of(k) == "REDUCE" for k in knobs):
        return {}
    d = _reduce_decomp(knobs)
    g = _geom_feats(
        knobs,
        threads=d.coop,
        cells=1,
        tile_m=1,
        tile_n=1,
        splitk=d.cta,
        bn=0,
        bm=0,
        bk=1,  # a TILE-less row has no K-chunk knob (the serial remainder is derived, not spelled)
        br=d.coop,
        free_prod=knobs.get("S_ext_free_prod"),
        sm=float(knobs.get("H_sm_count") or 170.0),
        warp=False,
        finalize=d.finalize,
    )
    out = {k: g[k] for k in _REDUCE_FEATURE_KEYS if k in g}
    out["D_reduce_ilp"] = math.log2(max(float(d.fold), 1.0))
    # The ``b<n>t`` transposed band: same thread count as its interleaved twin, entirely different
    # kernel (k-major lane sweep, smem-tree combine) — without this flag the two featurize
    # byte-identically. Coop moves enumerate on the TILE-less / per-cell tier only, so this block
    # is the one place the letter needs to surface.
    out["D_reduce_transposed"] = 1.0 if d.coop_transposed else 0.0
    return out


def _tile_features(knobs: dict) -> dict[str, float]:
    """Scalar thread-tile ``D_*`` features (``BN·BM`` threads, ``BM·FM × BN·FN``
    output). Empty unless the core tile knobs (``BN/BM/FM/FN``) are present, so
    pointwise / non-tiled kernels are unaffected (a TILE-less row with a
    ``REDUCE`` codec gets the :func:`_reduce_features` block instead). Warp-tier
    (tensor-core) rows are skipped here — :func:`knob_features` computes their
    occupancy via :func:`_warp_tile_features` (the warp tile is ``WM·WN·32``
    threads, ``WM·FM·atom_m × WN·FN·atom_n`` output), so the warp ``BM=BN=0`` OFF
    sentinels don't feed a meaningless scalar tile."""
    if is_warp(knobs):
        return {}
    slots = _free_slots(knobs)
    if slots is None:
        return {}
    par_n, reg_n, par_m, reg_m = slots  # (BN, FN, BM, FM)
    d = _reduce_decomp(knobs)
    # The scalar ``TILE`` codec spells no K-chunk (the smem slab's K granularity is derived
    # fit-to-smem at stage resolution, never a knob), so ``bk`` is structurally 1 here and the
    # scalar ``D_l2_bk`` / ``D_bk_ge32`` bands stay 0 until the codec grows a K token.
    bn, bm, fm, fn, br, bk, splitk = par_n, par_m, reg_m, reg_n, d.coop, 1, d.cta
    return _geom_feats(
        knobs,
        threads=bn * bm * br,
        cells=fm * fn,
        tile_m=bm * fm,
        tile_n=bn * fn,
        splitk=splitk,
        bn=bn,
        bm=bm,
        bk=bk,
        br=br,
        free_prod=knobs.get("S_ext_free_prod"),
        sm=float(knobs.get("H_sm_count") or 170.0),
        warp=False,
        finalize=d.finalize,
    )


def _warp_tile_features(knobs: dict) -> dict[str, float]:
    """Warp-tier (tensor-core MMA) tile ``D_*`` features — the warp analogue of
    :func:`_tile_features`. The CTA runs ``WM·WN`` warps (``·32`` lanes) over a
    ``WM·FM·atom_m × WN·FN·atom_n`` output tile, where ``atom_m/atom_n`` are the MMA cell dims read
    from the parsed warp ``TILE`` codec's atom. Empty if the ``TILE`` value isn't a warp codec or
    doesn't parse (so a malformed row degrades gracefully)."""
    from emmy.compiler.ir.schedule import TilePlan, is_warp_codec  # noqa: PLC0415

    spec = family_value(knobs, "TILE")
    if not is_warp_codec(spec):
        return {}
    try:
        plan = TilePlan.parse(spec)
        am, an = plan.atom.atom_m, plan.atom.atom_n
    except ValueError:
        return {}
    slots = _free_slots(knobs)
    if slots is None:
        return {}
    wn, fn, wm, fm = slots  # (WN, FN, WM, FM) — the true codec axes (units_n/reg_n/units_m/reg_m)
    if wm <= 0 or wn <= 0:
        return {}
    d = _reduce_decomp(knobs)
    out = _geom_feats(
        knobs,
        threads=wm * wn * 32,
        cells=fm * fn,
        tile_m=wm * fm * am,
        tile_n=wn * fn * an,
        splitk=d.cta,
        bn=0,  # OFF sentinels: the BN/BM bands don't fire on a warp row
        bm=0,
        # The slab K-chunk is the TILE codec's ``k<n>`` token (``TilePlan.bk``, atom_k multiples —
        # the codec's native unit, matching the shallow ``D_w_near_bk`` ≈2 target). It used to read
        # the never-set ``_Decomp.serial`` (always 1), so every ``D_w_*_bk`` was constant and
        # k-chunk siblings featurized byte-identically.
        bk=plan.bk,
        br=d.coop,
        free_prod=knobs.get("S_ext_free_prod"),
        sm=float(knobs.get("H_sm_count") or 170.0),
        warp=True,
        # Forward the split-K finalize letter: without it the ``_geom_feats`` default ("atomic")
        # made a warp ``g<n>k`` row featurize as its ``g<n>a`` twin — the deferred-combine choice
        # was invisible exactly on the tensor-core tier where wide split-K matters most.
        finalize=d.finalize,
    )
    # The warp-grid arrangement: how the CTA's warps split across the two canonical free slots
    # (``_free_slots``' wide/narrow ordering, same convention as the scalar ``D_l2_bn``/``D_l2_bm``
    # pair). The CTA tile dims fold ``w·f·atom``, so no tile-level feature recovers the grid
    # itself — yet it decides per-warp fragment reuse vs cross-warp parallelism. The 2026-07-09
    # 5090 sweep's TILE misses were exactly wide-vs-narrow warp grids over the same pool (golden
    # w2x4/w4x2 vs picked w4x1/w8x2). Absent on scalar rows (skip-if-missing 0.0), like the
    # tier-split ``D_w_*_bk`` pair.
    out["D_w_grid_m"] = math.log2(max(wm, 1))
    out["D_w_grid_n"] = math.log2(max(wn, 1))
    out["D_w_grid_aspect"] = math.log2(max(wm, 1)) - math.log2(max(wn, 1))
    return out


def _atom_features(atom) -> dict[str, float]:
    """Physical-property expansion of a tensor-core :class:`AtomKind` (the warp ``TILE`` codec's
    ``a:<atom>``) into the ``MMA_*`` feature family the priors rank on: the tier flag, the cell
    ``(m, n, k)`` dims, and the multiplicand / accumulator bit-widths."""
    m, n, k = atom.shape
    return {
        "MMA_tier": 1.0,
        "MMA_atom_m": float(m),
        "MMA_atom_n": float(n),
        "MMA_atom_k": float(k),
        "MMA_a_bits": float(atom.operand_dtype("a").nbytes * 8),
        "MMA_acc_bits": float(atom.operand_dtype("c").nbytes * 8),
    }


def _as_bool(v: object) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in {"1", "true", "yes", "on"}
    return bool(v)


def _coerce_float(v: object) -> float | None:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
