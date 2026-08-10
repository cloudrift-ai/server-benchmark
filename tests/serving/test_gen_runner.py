"""Fast CPU tests for ``gen_runner`` helpers (no GPU/model). The decode-bucket compile +
correctness are covered on GPU by ``test_gen_runner_gpu.py`` / ``test_vllm_plugin_gen_gpu.py``."""

import numpy as np
import pytest

from emmy.serving.gen_runner import _pad_rows


def test_pad_rows_pads_with_zeros_and_preserves_real_rows():
    a = np.arange(6, dtype=np.float16).reshape(3, 2)
    out = _pad_rows(a, 5)
    assert out.shape == (5, 2)
    assert out.dtype == np.float16
    np.testing.assert_array_equal(out[:3], a)  # real rows intact
    assert (out[3:] == 0).all()  # padding is zeros (computed then sliced away)


def test_pad_rows_is_passthrough_when_already_at_bucket():
    a = np.ones((4, 8), dtype=np.float16)
    assert _pad_rows(a, 4) is a  # no copy when t == bucket


@pytest.mark.parametrize(("quant_method", "coded_trunk"), [("exl3", True), ("fp8", False)])
def test_create_keeps_only_exl3_trunk_coded(tmp_path, monkeypatch, quant_method, coded_trunk):
    """EXL3 stays checkpoint-coded; FP8 preserves its decoded trunk lane."""
    import json

    from emmy.compiler.loader import safetensors
    from emmy.compiler.trace import huggingface
    from emmy.serving.gen_runner import EmmyGenRunner

    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": quant_method}}))
    seen = {}
    fake_model = object()
    fake_store = {"fmt": quant_method}

    monkeypatch.setattr(safetensors, "warn_if_unpinned", lambda _model_id: None)
    monkeypatch.setattr(huggingface, "quantized_checkpoint_dir", lambda _model_id: tmp_path)

    def fake_load(path, dtype, *, compress_trunk=False):
        seen.update(path=path, dtype=dtype, compress_trunk=compress_trunk)
        return fake_model, fake_store

    monkeypatch.setattr(huggingface, "load_quantized_split", fake_load)

    def fake_from_model(cls, model, **kwargs):
        seen.update(cls=cls, model=model, kwargs=kwargs)
        return "runner"

    monkeypatch.setattr(EmmyGenRunner, "from_model", classmethod(fake_from_model))

    assert EmmyGenRunner.create(str(tmp_path), dtype_str="float16") == "runner"
    assert seen["path"] == tmp_path
    assert seen["compress_trunk"] is coded_trunk
    assert seen["model"] is fake_model
    assert seen["kwargs"]["expert_store"] is fake_store
