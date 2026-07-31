"""The offline prior's tensor-core preference gates.

A warp-eligible f16 contraction enumerates scalar AND mma rows; the deploy pick must not
land on a scalar tile when tensor cores are on offer. Historically the DYNAMIC (masked-tier)
weight set ranked scalar split-K rows first — the qwen3-emb / gemma-4-e2b layer-0 projection
deploys landed scalar at 5-20x the -O3 cost of their enumerated mma siblings — because no
feature told the prior the alternative existed. The ``S_warp_eligible`` kernel stamp
(``_schedule._tile_rows``) + ``D_scalar_on_warp_eligible`` / ``D_splitk_roundtrip``
(``features._geom_feats``) + the hard-coded ``OfflinePrior`` gates close that. No GPU.
"""

from __future__ import annotations

import pytest

from emmy.compiler.context import Context
from emmy.compiler.pipeline.search.features import knob_features
from emmy.compiler.pipeline.search.golden_eval import _enumerate
from emmy.compiler.pipeline.search.prior import OfflinePrior

CTX = Context.from_target((8, 9))


def _tile_of(row: dict) -> str:
    return row[next(k for k in row if k.startswith("TILE"))]


def _is_warp(row: dict) -> bool:
    # F1 site grammar: the tier discriminator IS the worker kind (the row's ONE WORK entry).
    return str(row.get("WORK", "")).startswith("w")


def _base(M: int, N: int, K: int, *, dynamic: bool) -> dict:
    free = float(N) if dynamic else float(M * N)
    b = {**CTX.features(), "S_ext_free_prod": free, "S_ext_reduce_prod": float(K), "S_ext_reduce_max": float(K)}
    if dynamic:
        b["S_ext_n_symbolic_axis"] = 1.0
    return b


def test_warp_eligible_stamp_fp16_present_fp32_absent():
    """Every row of a warp-eligible f16 contraction (scalar rows included) carries the
    ``S_warp_eligible`` kernel stamp; an fp32 contraction (no atoms) carries none."""
    rows16, _ = _enumerate(512, 1024, 1024, "fp16", CTX)
    assert rows16, "f16 matmul must enumerate"
    assert all(r.get("S_warp_eligible") == 1.0 for r in rows16)
    assert any(_is_warp(r) for r in rows16), "warp rows must be offered"
    rows32, _ = _enumerate(512, 1024, 1024, "fp32", CTX)
    assert rows32 and all("S_warp_eligible" not in r for r in rows32)


def test_scalar_on_warp_eligible_feature_fires_on_scalar_rows_only():
    rows, _ = _enumerate(512, 1024, 1024, "fp16", CTX)
    base = _base(512, 1024, 1024, dynamic=False)
    for r in rows[:200]:
        tile = _tile_of(r)
        if not tile:
            continue  # the per-cell serial row has no tile geometry — no D_* features at all
        feat = knob_features({**base, **r}).get("D_scalar_on_warp_eligible", 0.0)
        assert feat == (0.0 if _is_warp(r) else 1.0), tile


@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
def test_offline_ranks_mma_above_every_scalar_split(dynamic):
    """The deploy-critical property: some mma row outranks EVERY scalar row (g?a / g?k
    splits included) on a warp-eligible f16 projection shape — in BOTH weight regimes.
    The dynamic (symbolic-M) regime is the one that historically ranked all-scalar."""
    rows, _ = _enumerate(512, 1024, 1024, "fp16", CTX)
    ap = OfflinePrior()
    base = _base(512, 1024, 1024, dynamic=dynamic)
    scored = sorted(rows, key=lambda r: ap.score({**base, **r}))
    assert _is_warp(scored[0]), f"top pick must be a warp row, got {_tile_of(scored[0])!r}"
    best_scalar = next((i for i, r in enumerate(scored) if not _is_warp(r)), None)
    assert best_scalar is None or best_scalar > 0, "a scalar row must not outrank every mma row"
    # Stronger: no scalar row inside the tuner's first-page patience window.
    assert all(_is_warp(r) for r in scored[:25]), "scalar rows must not crowd the top-25"
