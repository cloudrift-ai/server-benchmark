"""The ``OfflinePrior`` — the cold-start scorer: its weights artifact, its quality proxy, and the
hard-coded tensor-core gates. No GPU.

Three sections, one subject:

- **The weights artifact** — the repo-checked default's schema, the ``EMMY_OFFLINE_FILE`` override,
  and the hard-error load semantics (a missing or version-mismatched artifact never silently falls
  back — an A/B that quietly reverts to other weights measures nothing).
- **Proxy desaturation** — the retired ``±80`` QUALITY clip inside the exp squash sat in the middle
  of the live range (at scale 0.1 thousands of good warp tiles score past 80): every such row
  collapsed onto one ``exp(-8)`` plateau, greedy's argmin fell through to emission order (the
  ``w1x1`` gemma misdeploys, 12-29x off the golden on a real 4090), and the evaluator's
  strictly-greater rank reported 0 for every tied row — the "prior finds the goldens" illusion. The
  clip now guards only the exp argument's float-safety bound, and the rank counts earlier-emitted
  ties against the golden.
- **Tier preference** — a warp-eligible f16 contraction enumerates scalar AND mma rows; the deploy
  pick must not land on a scalar tile when tensor cores are on offer. Historically the DYNAMIC
  (masked-tier) weight set ranked scalar split-K rows first — the qwen3-emb / gemma-4-e2b layer-0
  projection deploys landed scalar at 5-20x the -O3 cost of their enumerated mma siblings — because
  no feature told the prior the alternative existed. The ``S_warp_eligible`` kernel stamp
  (the contraction row product) + ``D_scalar_on_warp_eligible`` / ``D_splitk_roundtrip``
  (``features._geom_feats``) + the hard-coded ``OfflinePrior`` gates close that.
"""

from __future__ import annotations

import json
import math

import pytest

from emmy.compiler import dtype as _dt
from emmy.compiler.context import Context
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import MatmulOp
from emmy.compiler.pipeline.search import golden_eval
from emmy.compiler.pipeline.search.features import FEATURIZER_VERSION, knob_features
from emmy.compiler.pipeline.search.golden_eval import enumerate_graph
from emmy.compiler.pipeline.search.prior import OfflinePrior
from emmy.compiler.pipeline.search.prior.fallback import FallbackPrior
from emmy.compiler.pipeline.search.prior.offline import _DEFAULT_FILE, _PARAM_KEYS

# ---------------------------------------------------------------------------
# The weights artifact
# ---------------------------------------------------------------------------


def test_shipped_artifact_is_current_and_complete():
    """The repo-checked default: current featurizer vocabulary, both weight sets,
    all scoring params — the schema gate that keeps `OfflinePrior()` loadable on
    a fresh clone."""
    obj = json.loads(_DEFAULT_FILE.read_text())
    assert obj["feat_ver"] == FEATURIZER_VERSION
    assert obj["kind"] == "linear"
    assert obj["weights"] and obj["weights_dynamic"]
    for w_set in (obj["weights"], obj["weights_dynamic"]):
        assert all(isinstance(k, str) and isinstance(v, (int, float)) for k, v in w_set.items())
    assert set(_PARAM_KEYS) <= set(obj["params"])


def _artifact(**over) -> dict:
    base = {
        "feat_ver": FEATURIZER_VERSION,
        "kind": "linear",
        "weights": {"D_square": 3.0},
        "weights_dynamic": {"D_square": -3.0},
        "params": {"scale": 0.1, "atomic_free_split_threshold": 4.0, "atomic_free_weight": 0.0},
    }
    base.update(over)
    return base


def test_env_override_scores_through_the_file(tmp_path, monkeypatch):
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(_artifact()))
    monkeypatch.setenv("EMMY_OFFLINE_FILE", str(path))
    p = OfflinePrior()
    assert math.isclose(p.mean_score_features({"D_square": 1.0}), math.exp(-0.1 * 3.0))
    # The dynamic set rides the same file (opposite planted sign).
    assert math.isclose(p.mean_score_features({"D_square": 1.0, "S_ext_n_symbolic_axis": 1.0}), math.exp(0.1 * 3.0))


def test_params_come_from_the_file(tmp_path, monkeypatch):
    art = _artifact()
    art["params"]["scale"] = 0.5
    art["params"]["atomic_free_weight"] = 7.0
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(art))
    monkeypatch.setenv("EMMY_OFFLINE_FILE", str(path))
    p = OfflinePrior()
    assert p.model.scale == 0.5
    assert p.model.atomic_free_weight == 7.0


def test_explicit_kwargs_win_over_the_file(tmp_path, monkeypatch):
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(_artifact()))
    monkeypatch.setenv("EMMY_OFFLINE_FILE", str(path))
    p = OfflinePrior(weights={"D_square": 1.0}, scale=1.0)
    assert p.model.weights == {"D_square": 1.0}  # kwarg, not the file's 3.0
    assert p.model.scale == 1.0
    assert p.model.weights_dynamic == {"D_square": -3.0}  # unpassed fields still resolve from the file


def test_model_and_field_overrides_do_not_mix():
    """Passing both a ready model and a per-field override is rejected, not silently resolved one
    way. Ignoring the override would hand an A/B the unmodified model and let it report a difference
    of zero as a real result."""
    from emmy.compiler.pipeline.search.prior.linear_model import LinearModel  # noqa: PLC0415

    model = LinearModel(weights={"D_square": 1.0}, weights_dynamic={}, scale=0.1, atomic_free_weight=0.0, atomic_free_split_threshold=4.0)
    assert OfflinePrior(model=model).model is model
    with pytest.raises(ValueError, match="not both"):
        OfflinePrior(model=model, scale=2.0)


def test_missing_file_is_a_hard_error(tmp_path, monkeypatch):
    monkeypatch.setenv("EMMY_OFFLINE_FILE", str(tmp_path / "nope.json"))
    with pytest.raises(RuntimeError, match="missing or unreadable"):
        OfflinePrior()


def test_feat_ver_mismatch_is_a_hard_error(tmp_path, monkeypatch):
    path = tmp_path / "stale.json"
    path.write_text(json.dumps(_artifact(feat_ver=FEATURIZER_VERSION - 1)))
    monkeypatch.setenv("EMMY_OFFLINE_FILE", str(path))
    with pytest.raises(RuntimeError, match="feat_ver"):
        OfflinePrior()


def test_incomplete_artifact_is_a_hard_error(tmp_path, monkeypatch):
    art = _artifact()
    del art["params"]["scale"]
    path = tmp_path / "partial.json"
    path.write_text(json.dumps(art))
    monkeypatch.setenv("EMMY_OFFLINE_FILE", str(path))
    with pytest.raises(RuntimeError, match="params.scale"):
        OfflinePrior()


# ---------------------------------------------------------------------------
# Proxy desaturation + the golden rank
# ---------------------------------------------------------------------------

_GATES_OFF = {"atomic_free_split_threshold": 0.0, "atomic_free_weight": 0.0}


def _prior(scale: float = 0.1) -> OfflinePrior:
    """A single-weight prior: quality == the row's ``D_q`` value."""
    return OfflinePrior(weights={"D_q": 1.0}, weights_dynamic={"D_q": 1.0}, scale=scale, **_GATES_OFF)


def _matmul_graph(M: int, N: int, K: int, dtype: str) -> Graph:
    graph = Graph()
    dt = _dt.get({"fp16": "f16", "fp32": "f32"}.get(dtype, dtype))
    graph.add_node(InputOp(), [], Tensor("a", (M, K), dt), node_id="a")
    graph.add_node(InputOp(), [], Tensor("b", (K, N), dt), node_id="b")
    graph.add_node(MatmulOp(), ["a", "b"], Tensor("o", (M, N), dt), node_id="o")
    graph.inputs, graph.outputs = ["a", "b"], ["o"]
    return graph


def _enumerate(M: int, N: int, K: int, dtype: str, ctx: Context) -> list[dict]:
    return enumerate_graph(_matmul_graph(M, N, K, dtype), ctx, family="TILE")


def test_qualities_past_the_old_clip_stay_strictly_ordered():
    """Qualities 85 and 120 both sat past the retired ±80 clip and scored the identical
    exp(-8); they must now rank strictly (higher quality → lower latency proxy)."""
    ap = _prior()
    ok, better = ap.mean_score_features({"D_q": 85.0}), ap.mean_score_features({"D_q": 120.0})
    assert better < ok, "quality 120 must outrank quality 85 — the old clip tied them"
    assert ap.mean_score_features({"D_q": 90.0}) == ap.mean_score_features({"D_q": 90.0})


def test_exp_argument_clips_at_the_float_safety_bound_only():
    """Absurd qualities stay finite (the clip's only remaining job) and the bound sits
    outside anything a real pool produces."""
    ap = _prior()
    assert ap.mean_score_features({"D_q": 1e9}) == math.exp(-700.0)
    assert ap.mean_score_features({"D_q": -1e9}) == math.exp(700.0)
    # just inside the bound: still strictly ordered
    assert ap.mean_score_features({"D_q": 6900.0}) < ap.mean_score_features({"D_q": 6800.0})


def test_fallback_tilt_multiplier_stays_bounded(monkeypatch):
    """The trusted-online blend must not feed the unsaturated proxy (up to e**±700)
    straight into the µs product — the multiplier clamps to e**±8."""
    from emmy import config

    class _Online:
        trustworthy = True

        def score(self, knobs):  # noqa: ARG002 — fixed µs anchor
            return 100.0

    class _Offline:
        def __init__(self, proxy):
            self.proxy = proxy

        def score(self, knobs):  # noqa: ARG002 — an extreme unsaturated proxy
            return self.proxy

    monkeypatch.setattr(config, "offline_tilt", lambda: 1.0)
    lo = FallbackPrior(_Online(), _Offline(math.exp(-700.0))).score({})
    hi = FallbackPrior(_Online(), _Offline(math.exp(700.0))).score({})
    assert lo == 100.0 * math.exp(-8.0), "multiplier must clamp at e**-8, not vanish to ~0"
    assert hi == 100.0 * math.exp(8.0), "multiplier must clamp at e**+8, not blow up the µs anchor"


def test_evaluate_record_rank_is_tie_pessimistic(monkeypatch):
    """A golden tied with earlier-emitted rows loses the greedy argmin to them — the
    rank must count those ties, so a tie plateau can never report rank 0."""
    rows = [
        {"TILE@a0": "f2x4", "WORK": "t16x8"},
        {"TILE@a0": "f4x8", "WORK": "t32x8"},
        {"TILE@a0": "f4x8", "WORK": "t32x16"},  # the golden, emitted third
        {"TILE@a0": "f4x8", "WORK": "t64x16"},
    ]
    monkeypatch.setattr(golden_eval, "enumerate_graph", lambda graph, ctx: rows)
    golden = {"TILE": "f4x8", "WORK": "t32x16"}
    from types import SimpleNamespace

    record = SimpleNamespace(knobs=golden, target_program=Graph(), pin_map={"FAST_MATH": False})

    _, rank_tied, pool, rank_opt = golden_eval.evaluate_record(record, ctx=None, scorer=lambda r: 1.0)
    assert pool == 4
    assert rank_tied == 2, "two earlier-emitted ties must count against the golden"
    assert rank_opt == 0, "the optimistic count forgives the whole tie plateau"
    assert rank_tied - rank_opt == 2, "the pessimistic-optimistic gap IS the earlier-tie plateau width"

    def favors_golden(r):
        return 2.0 if r is rows[2] else 1.0

    _, rank_best, _, rank_best_opt = golden_eval.evaluate_record(record, ctx=None, scorer=favors_golden)
    assert rank_best == 0, "a strict winner still ranks 0"
    assert rank_best_opt == 0, "no plateau, no gap"


# ---------------------------------------------------------------------------
# The tensor-core tier-preference gates
# ---------------------------------------------------------------------------

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
    rows16 = _enumerate(512, 1024, 1024, "fp16", CTX)
    assert rows16, "f16 matmul must enumerate"
    assert all(r.get("S_warp_eligible") == 1.0 for r in rows16)
    assert any(_is_warp(r) for r in rows16), "warp rows must be offered"
    rows32 = _enumerate(512, 1024, 1024, "fp32", CTX)
    assert rows32 and all("S_warp_eligible" not in r for r in rows32)


def test_scalar_on_warp_eligible_feature_fires_on_scalar_rows_only():
    rows = _enumerate(512, 1024, 1024, "fp16", CTX)
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
    rows = _enumerate(512, 1024, 1024, "fp16", CTX)
    ap = OfflinePrior()
    base = _base(512, 1024, 1024, dynamic=dynamic)
    scored = sorted(rows, key=lambda r: ap.score({**base, **r}))
    assert _is_warp(scored[0]), f"top pick must be a warp row, got {_tile_of(scored[0])!r}"
    best_scalar = next((i for i, r in enumerate(scored) if not _is_warp(r)), None)
    assert best_scalar is None or best_scalar > 0, "a scalar row must not outrank every mma row"
    # Stronger: no scalar row inside the tuner's first-page patience window.
    assert all(_is_warp(r) for r in scored[:25]), "scalar rows must not crowd the top-25"
