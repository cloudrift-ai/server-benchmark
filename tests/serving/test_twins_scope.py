from pathlib import Path

import pytest

from emmy.serving.twins import _serving_twin_buckets, validate_static_only_release_config


def _release_config(path: Path, **overrides: str) -> Path:
    values = {
        "SERVE_STATIC_ONLY": "1",
        "SERVE_MAX_NUM_BATCHED_TOKENS": "1",
        "SERVE_DECODE_BUCKET": "1",
        "SERVE_PREFILL_CAPACITY": "1",
        "SERVE_PREFILL_BUCKET": "0",
        "SERVE_M1_TIER": "1",
        "SERVE_CAPTURE_SIZES": "[1]",
    }
    values.update(overrides)
    path.write_text("".join(f"{key}={value}\n" for key, value in values.items()))
    return path


def test_standard_serving_twin_scope_keeps_all_widths_and_symbolic():
    assert _serving_twin_buckets(32, 256, (), symbolic=True, static_only=False) == [
        ("1", 1),
        ("8", 8),
        ("32", 32),
        ("64", 64),
        ("192", 192),
        ("256", 256),
        ("2048", 2048),
        ("4096", 4096),
        ("-sym", None),
    ]


def test_static_only_serving_twin_scope_is_exactly_m1():
    assert _serving_twin_buckets(1, 0, (), symbolic=True, static_only=True) == [("1", 1)]


@pytest.mark.parametrize(
    ("decode", "prefill", "extra"),
    [(2, 0, ()), (1, 1, ()), (1, 0, (8,))],
)
def test_static_only_serving_twin_scope_rejects_wider_inputs(decode, prefill, extra):
    with pytest.raises(ValueError, match="static-only serving capture requires"):
        _serving_twin_buckets(decode, prefill, extra, symbolic=True, static_only=True)


def test_static_only_release_config_accepts_exact_and_inherited_m1_warm_shapes(tmp_path):
    path = _release_config(tmp_path / "model.env", SERVE_WARM_SHAPES=":: 1:0:1")
    validate_static_only_release_config(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("SERVE_STATIC_ONLY", "0"),
        ("SERVE_MAX_NUM_BATCHED_TOKENS", "2"),
        ("SERVE_DECODE_BUCKET", "2"),
        ("SERVE_PREFILL_CAPACITY", "2"),
        ("SERVE_PREFILL_BUCKET", "1"),
        ("SERVE_M1_TIER", "0"),
        ("SERVE_CAPTURE_SIZES", "[1, 2]"),
    ],
)
def test_static_only_release_config_rejects_unsafe_pinned_values(tmp_path, field, value):
    path = _release_config(tmp_path / "model.env", **{field: value})
    with pytest.raises(ValueError, match=field):
        validate_static_only_release_config(path)


@pytest.mark.parametrize("warm", ["2::", ":1:", "::2", "1:0:1:fm"])
def test_static_only_release_config_rejects_unsafe_warm_overrides(tmp_path, warm):
    path = _release_config(tmp_path / "model.env", SERVE_WARM_SHAPES=warm)
    with pytest.raises(ValueError, match="static-only release"):
        validate_static_only_release_config(path)
