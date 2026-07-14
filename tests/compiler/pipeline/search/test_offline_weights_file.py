"""The OfflinePrior weights artifact: the repo-checked default's schema, the
``EMMY_OFFLINE_FILE`` override, and the hard-error load semantics (a missing or
version-mismatched artifact never silently falls back — an A/B that quietly
reverts to other weights measures nothing)."""

import json
import math

import pytest

from emmy.compiler.pipeline.search.features import FEATURIZER_VERSION
from emmy.compiler.pipeline.search.prior.offline import _DEFAULT_FILE, _PARAM_KEYS, OfflinePrior


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
        "params": {
            "scale": 0.1,
            "atomic_free_split_threshold": 4.0,
            "atomic_free_weight": 0.0,
            "scalar_on_warp_weight": 40.0,
            "splitk_roundtrip_weight": 0.25,
        },
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
    art["params"]["scalar_on_warp_weight"] = 7.0
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(art))
    monkeypatch.setenv("EMMY_OFFLINE_FILE", str(path))
    p = OfflinePrior()
    assert p._scale == 0.5
    assert p._scalar_on_warp_weight == 7.0


def test_explicit_kwargs_win_over_the_file(tmp_path, monkeypatch):
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(_artifact()))
    monkeypatch.setenv("EMMY_OFFLINE_FILE", str(path))
    p = OfflinePrior(weights={"D_square": 1.0}, scale=1.0)
    assert p._w == {"D_square": 1.0}  # kwarg, not the file's 3.0
    assert p._scale == 1.0
    assert p._w_dyn == {"D_square": -3.0}  # unpassed fields still resolve from the file


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
