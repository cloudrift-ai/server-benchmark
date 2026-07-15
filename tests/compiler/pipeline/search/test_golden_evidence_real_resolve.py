"""The golden evidence tier exercised through the REAL greedy resolve — trace the golden
kind's snippet, run the full ``TILE_PASSES`` resolve with ``greedy_decide``, and assert the
golden actually decided the fork. The unit tests join against hand-built stamp dicts; this
suite exists because those passed while a live deploy missed: the flash fork's root op is
the tile pass's RESTRUCTURED twisted op whose knobs carry re-derived extents only (no
histogram, no ``S_loop_depth``), so a key classified off final-op stamps never matched at
the fork (the 4090 evidence-tier verification: attention goldens silently skipped, 3.7-4.2x
deploys). Fixture knobs are read off the live offer itself, so enumeration drift moves the
fixture instead of false-failing the join."""

from __future__ import annotations

import pytest

from emmy.compiler.context import Context
from emmy.compiler.pipeline import TILE_PASSES, Pipeline
from emmy.compiler.pipeline.pipeline import Run
from emmy.compiler.pipeline.search import golden as golden_mod
from emmy.compiler.pipeline.search.golden import AttentionGoldenConfig, RmsNormGoldenConfig
from emmy.compiler.pipeline.search.policy import greedy as greedy_mod

CARD = "NVIDIA GeForce RTX 4090"
CAP = (8, 9)

_SDPA = (
    "F.scaled_dot_product_attention(torch.randn(1,16,512,256,dtype=torch.float16), "
    "torch.randn(1,16,512,256,dtype=torch.float16), torch.randn(1,16,512,256,dtype=torch.float16), is_causal=True)"
)
_SDPA_DYN = ["seq_len@x0:2", "seq_len@x1:2", "seq_len@x2:2"]
_RMS = "torch.nn.RMSNorm(3840)(torch.randn(512,3840))"


class _StubPrior:
    """Bare-``mean_scores`` prior preferring emission order — any golden pick must
    override it, so a pass proves the tier (not the model) decided."""

    def mean_scores(self, rows):
        return list(range(len(rows)))


def _resolve(snippet: str, dyn: list[str]):
    """One full greedy resolve of the traced snippet under the DEPLOYABLE regime — the
    ctx's compile flags are forced empty (-O3): ``make test`` runs the -Xcicc -O1
    correctness lane, where the golden tier is correctly inert (the regime guard), so
    without this override the suite would exercise nothing. No nvcc runs here (the
    resolve stops at the tile dialect), so the override is purely a feature stamp."""
    from dataclasses import replace

    from emmy.commands.trace import graph_from_code
    from emmy.compiler.trace.dynamic import build_torch_dynamic_shapes, parse_position_specs

    ds = build_torch_dynamic_shapes(parse_position_specs(dyn)) if dyn else None
    graph, _, _ = graph_from_code(snippet, dynamic_shapes=ds)
    ctx = replace(Context.from_target(CAP, gpu_name=CARD), compile_flags="")
    Run(pipeline=Pipeline.build(TILE_PASSES), ctx=ctx).resolve(graph, greedy_mod.greedy_decide(prior=_StubPrior(), price_structural=False))
    return graph


def _spying(monkeypatch, rows_sink: list, picks_sink: list):
    orig = greedy_mod._golden_pick

    def spy(index, rows, node_id):
        got = orig(index, rows, node_id)
        rows_sink.append(rows)
        picks_sink.append((node_id, got))
        return got

    monkeypatch.setattr(greedy_mod, "_golden_pick", spy)


def _offered_flash_form(rows: list[dict]) -> dict:
    """A non-degenerate offered flash form (skip the w1x1 emission-order head) to
    record as the test golden — read off the live enumeration, never hardcoded."""
    for r in rows:
        dd = r.get("TILE@dd", "")
        if "/w1x1/" not in dd and dd:
            return r
    return rows[0]


def _decoy():
    """An off-shape golden keeping the evidence index non-empty during the probe
    resolve — the tier (and so the spy) only runs when the card has recorded goldens."""
    return AttentionGoldenConfig(
        name="attention.hd64.decoy",
        n_heads=2,
        seq=512,
        head_dim=64,
        dtype="fp16",
        knobs={"TILE@dd": "a:mma_m16n8k16_f16_f32/w1x1/f1x2/k16"},
        emmy_us=1.0,
        gpu_name=CARD,
        compute_cap=CAP,
    )


@pytest.mark.parametrize("dyn", [[], _SDPA_DYN], ids=["static", "dynM"])
def test_attention_golden_decides_the_live_flash_fork(monkeypatch, dyn):
    # Probe resolve: capture the flash fork's offered rows (decoy keeps the tier live).
    monkeypatch.setattr(golden_mod, "GOLDEN_CONFIGS", [_decoy()])
    rows_sink: list = []
    picks_sink: list = []
    _spying(monkeypatch, rows_sink, picks_sink)
    _resolve(_SDPA, dyn)
    flash_rows = next(r for r in rows_sink if any("TILE@dd" in row for row in r))
    form = _offered_flash_form(flash_rows)
    # Record that form as the golden: static keeps the axis-keyed all-or-nothing pins;
    # the dynamic twin is schema-required to record ONE bare TILE (the dd plan).
    knobs = (
        {"TILE": form["TILE@dd"], "STAGE": form.get("STAGE@kv", "")}
        if dyn
        else {"TILE@dd": form["TILE@dd"], "TILE@pj": form["TILE@pj"], "STAGE": form.get("STAGE@kv", "")}
    )
    gold = AttentionGoldenConfig(
        name="attention.hd256.test",
        n_heads=16,
        seq=512,
        head_dim=256,
        dtype="fp16",
        dynamic=bool(dyn),
        knobs=knobs,
        emmy_us=44.3,
        gpu_name=CARD,
        compute_cap=CAP,
    )
    monkeypatch.setattr(golden_mod, "GOLDEN_CONFIGS", [gold])
    rows_sink.clear()
    picks_sink.clear()
    graph = _resolve(_SDPA, dyn)
    hits = [(n, got) for n, got in picks_sink if got is not None]
    assert hits and hits[0][1][1] == 44.3, f"attention golden never decided a fork: {picks_sink}"
    # The resolved graph realized the golden's dd plan on its flash kernel.
    realized = [
        (getattr(n.op, "knobs", {}) or {}).get("TILE@dd") for n in graph.nodes.values() if "TILE@dd" in (getattr(n.op, "knobs", {}) or {})
    ]
    assert form["TILE@dd"] in realized


def test_rms_norm_golden_decides_the_live_reduce_fork(monkeypatch):
    monkeypatch.setattr(golden_mod, "GOLDEN_CONFIGS", [_decoy()])
    rows_sink: list = []
    picks_sink: list = []
    _spying(monkeypatch, rows_sink, picks_sink)
    _resolve(_RMS, [])
    reduce_rows = next((r for r in rows_sink if any(k.startswith("REDUCE@") for row in r for k in row)), None)
    if reduce_rows is None:
        pytest.skip("the rms_norm lowering offers no REDUCE fork on this pipeline")
    form = reduce_rows[-1]  # any non-head row: the stub prior prefers emission order
    fam_key = next(k for k in form if k.startswith("REDUCE@"))
    gold = RmsNormGoldenConfig(
        name="rms_norm.k3840.test",
        M=512,
        K=3840,
        knobs={"REDUCE": form[fam_key]},
        emmy_us=6.3,
        gpu_name=CARD,
        compute_cap=CAP,
    )
    monkeypatch.setattr(golden_mod, "GOLDEN_CONFIGS", [gold])
    picks_sink.clear()
    _resolve(_RMS, [])
    assert any(got is not None and got[1] == 6.3 for _, got in picks_sink), f"rms_norm golden never decided: {picks_sink}"
