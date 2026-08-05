"""The DiT-XL-2 deploy winners are decided by the SHIPPED goldens, not by a table in the scheduler.

`tile/_schedule.py` used to carry three SKU-exact deploy overrides for this workload —
`_RTX4080_DIT_CONTRACTION_DEFAULTS` (a `(a_computed, b_trans, M, N, K) -> (TILE, STAGE, REDUCE)`
dict), `_rtx4080_dit_flash_deploy` (the SDPA winner) and a `b128` REDUCE narrowing for the
LayerNorm. What could be expressed as a golden now lives in `goldens/rtx4080_sm89.yaml`
(`dit_xl_2.*`), where the deploy evidence tier already reads recorded winners.

This pins the move end to end: resolve the real shape through the real pipeline against the REAL
shipped corpus (no synthetic golden), and assert the golden tier — not the model — picks exactly the
schedule the table used to force. A stub prior preferring emission order stands in for the model, so
a pass can only come from the golden tier. If the enumeration ever stops offering the recorded row,
`_golden_pick` logs drift and falls through, and this fails.

The table's two COMPUTED-A entries were DELETED rather than mis-filed: the DiT prologue is an
AdaLayerNorm-Zero LayerNorm, and every fused golden kind is RMSNorm by construction (its `snippet()`
builds `F.rms_norm`), so recording them as `norm_linear` would deploy right and re-tune wrong. Those
two shapes now deploy off the prior until a LayerNorm-cone golden kind exists.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from emmy.compiler.context import Context
from emmy.compiler.pipeline.search.golden import GOLDEN_CONFIGS, AttentionGoldenConfig, MatmulGoldenConfig, matmul_snippet

CARD = "NVIDIA GeForce RTX 4080"
CAP = (8, 9)

#: (name, M, N, K) — the shapes the retired table covered with a plain gmem A.
_DIT_MATMULS = [("dit_xl_2.attn_out_proj.s256", 256, 1152, 1152), ("dit_xl_2.ff_out_proj.s256", 256, 1152, 4608)]


class _EmissionOrderPrior:
    """Prefers the first offered row — any golden pick must override it, so a pass proves the
    evidence tier decided rather than the model."""

    def mean_scores(self, rows):
        return list(range(len(rows)))


def _shipped(name: str):
    got = [g for g in GOLDEN_CONFIGS if g.name == name and g.gpu_name == CARD and tuple(g.compute_cap) == CAP]
    assert len(got) == 1, f"expected exactly one shipped {name} entry for {CARD}, got {len(got)}"
    return got[0]


def _resolve_picks(snippet: str, monkeypatch) -> list[dict]:
    """The rows the GOLDEN tier picked during one full greedy resolve of the snippet.

    ``compile_flags=""`` forces the deployable (-O3) regime: ``make test`` runs the -Xcicc -O1
    correctness lane, where the golden tier is correctly inert, so without the override this would
    exercise nothing. The resolve stops at the tile dialect — no nvcc, no GPU.
    """
    from emmy.commands.trace import graph_from_code
    from emmy.compiler.pipeline import TILE_PASSES, Pipeline
    from emmy.compiler.pipeline.pipeline import Run
    from emmy.compiler.pipeline.search.policy import greedy as greedy_mod

    picks: list[dict] = []
    orig = greedy_mod._golden_pick

    def spy(index, rows, node_id):
        got = orig(index, rows, node_id)
        if got is not None:
            picks.append(rows[got[0]])  # (candidate_index, recorded_us)
        return got

    monkeypatch.setattr(greedy_mod, "_golden_pick", spy)
    graph, _, _ = graph_from_code(snippet)
    ctx = replace(Context.from_target(CAP, gpu_name=CARD), compile_flags="")
    Run(pipeline=Pipeline.build(TILE_PASSES), ctx=ctx).resolve(
        graph, greedy_mod.greedy_decide(prior=_EmissionOrderPrior(), price_structural=False)
    )
    return picks


@pytest.mark.parametrize(("name", "M", "N", "K"), _DIT_MATMULS)
def test_the_shipped_dit_golden_is_a_transposed_fp16_matmul(name, M, N, K) -> None:
    g = _shipped(name)
    assert isinstance(g, MatmulGoldenConfig)
    # NOT the shape: ``_DIT_MATMULS`` was transcribed from the YAML, so asserting it back is a
    # mapping compared to itself. The deploy test below drives the real pipeline at that shape and
    # fails if either moved.
    assert g.dtype == "fp16" and g.trans_b, "the DiT projections are F.linear — B is (N, K)"


@pytest.mark.parametrize(("name", "M", "N", "K"), _DIT_MATMULS)
def test_the_shipped_dit_golden_decides_the_live_deploy(name, M, N, K, monkeypatch) -> None:
    want = _shipped(name).knobs
    picks = _resolve_picks(matmul_snippet(M, N, K, "fp16", True), monkeypatch)
    assert picks, f"{name}: the golden tier never decided a fork — the recorded winner is unreachable"
    got = [{k: v for k, v in row.items() if k in want} for row in picks]
    assert want in got, f"{name}: golden knobs {want} not among the picked rows {got}"


def test_the_shipped_dit_flash_golden_is_non_causal_sdpa() -> None:
    g = _shipped("dit_xl_2.attn.s256")
    assert isinstance(g, AttentionGoldenConfig)
    assert (g.n_heads, g.seq, g.head_dim) == (16, 256, 72), "DiT-XL-2: hidden 1152 = 16 heads x 72"
    assert g.dtype == "fp16" and not g.causal, "the DiT block attends over all patches — no causal mask"


def test_the_shipped_dit_flash_golden_decides_the_live_deploy(monkeypatch) -> None:
    """The flash arm forks eagerly-built ``TileOp``\\ s rather than dict rows, so this covers the
    OTHER of the scheduler's two fork currencies through the same evidence tier."""
    g = _shipped("dit_xl_2.attn.s256")
    picks = _resolve_picks(g.snippet(), monkeypatch)
    assert picks, "the golden tier never decided the flash fork — the recorded winner is unreachable"
    got = [{k: v for k, v in row.items() if k in g.knobs} for row in picks]
    assert g.knobs in got, f"flash golden knobs {g.knobs} not among the picked rows {got}"
