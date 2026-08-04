"""Tests for scripts/run_lmeval_gate.py — pure config/extraction helpers (no lm-eval install needed).

The canned results dict mirrors lm-eval 0.4.x `simple_evaluate` output (verified against 0.4.12):
per-task rows keyed "<metric>,<filter>" / "<metric>_stderr,<filter>" plus a presentation-only "alias".
"""

import sys
from pathlib import Path

import pytest


# Add scripts/ to sys.path so we can import the module directly
@pytest.fixture(autouse=True)
def _add_scripts_to_path():
    scripts_dir = str(Path(__file__).resolve().parents[2] / "scripts")
    sys.path.insert(0, scripts_dir)
    yield
    sys.path.remove(scripts_dir)


CANNED_RESULTS = {
    "results": {
        "gsm8k": {
            "alias": "gsm8k",
            "exact_match,strict-match": 0.57,
            "exact_match_stderr,strict-match": 0.01,
            "exact_match,flexible-extract": 0.585,
            "exact_match_stderr,flexible-extract": 0.011,
        }
    },
    "versions": {"gsm8k": 3},
    "n-shot": {"gsm8k": 5},
    "config": {"model": "local-completions"},
}


# --- completions_url / build_model_args ---


def test_completions_url_appends_path():
    from run_lmeval_gate import completions_url

    assert completions_url("http://localhost:8000/v1") == "http://localhost:8000/v1/completions"
    assert completions_url("http://localhost:8000/v1/") == "http://localhost:8000/v1/completions"


def test_completions_url_keeps_full_path():
    from run_lmeval_gate import completions_url

    assert completions_url("http://host:9000/v1/completions") == "http://host:9000/v1/completions"


def test_build_model_args():
    from run_lmeval_gate import build_model_args

    args = build_model_args("http://localhost:8000/v1", "google/gemma-4-12B-it")
    assert args == {
        "base_url": "http://localhost:8000/v1/completions",
        "model": "google/gemma-4-12B-it",
        "num_concurrent": 8,
        "tokenized_requests": False,
        # Overrides lm-eval's 300 s default: the emmy lane's slower request tails timed out
        # under it, and the retry storm that followed died on a closed session, sinking a
        # whole gate with no score (2026-08-01).
        "timeout": 900,
    }


# --- extract_metrics ---


def test_extract_metrics_drops_alias():
    from run_lmeval_gate import extract_metrics

    metrics = extract_metrics(CANNED_RESULTS, "gsm8k")
    assert "alias" not in metrics
    assert metrics["exact_match,strict-match"] == 0.57
    assert metrics["exact_match_stderr,strict-match"] == 0.01
    assert metrics["exact_match,flexible-extract"] == 0.585
    assert metrics["exact_match_stderr,flexible-extract"] == 0.011


# --- build_report ---


def test_build_report_shape():
    from run_lmeval_gate import build_report, extract_metrics

    metrics = extract_metrics(CANNED_RESULTS, "gsm8k")
    report = build_report("emmy-fastmath", "google/gemma-4-12B-it", "gsm8k", 200, 0, metrics, "0.4.12")
    assert report == {
        "label": "emmy-fastmath",
        "model": "google/gemma-4-12B-it",
        "task": "gsm8k",
        "limit": 200,
        "seed": 0,
        "metrics": metrics,
        "lm_eval_version": "0.4.12",
    }


def test_module_imports_without_lm_eval():
    """lm_eval is imported lazily inside main(); importing the module must not require it."""
    import run_lmeval_gate  # noqa: F401

    assert "lm_eval" not in sys.modules
