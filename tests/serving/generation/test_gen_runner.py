"""Fast CPU tests for ``gen_runner`` helpers (no GPU/model). The decode-bucket compile +
correctness are covered on GPU by ``test_gen_runner_gpu.py`` / ``test_vllm_plugin_gen_gpu.py``."""

import numpy as np
import pytest

from emmy.serving.gen_runner import EmmyGenRunner, _pad_rows, _program_config_sha, _static_decode_covers_capacity


class _Config:
    def __init__(self, data):
        self.data = data

    def to_dict(self):
        import copy

        return copy.deepcopy(self.data)


def test_program_config_identity_ignores_only_generation_eos_policy():
    base = {
        "model_type": "gemma4_unified_text",
        "text_config": {"hidden_size": 3840, "intermediate_size": 15360},
        "eos_token_id": 1,
    }
    instruction = {**base, "eos_token_id": [1, 106]}
    assert _program_config_sha(_Config(base)) == _program_config_sha(_Config(instruction))

    different_architecture = {**instruction, "model_type": "gemma4_other_text"}
    assert _program_config_sha(_Config(base)) != _program_config_sha(_Config(different_architecture))

    different_geometry = {
        **instruction,
        "text_config": {**instruction["text_config"], "hidden_size": 4096},
    }
    assert _program_config_sha(_Config(base)) != _program_config_sha(_Config(different_geometry))


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


@pytest.mark.parametrize(
    ("max_tokens", "decode_bucket", "prefill_bucket", "expected"),
    [
        (None, 16, 0, False),
        (1, 1, 0, True),
        (1, 16, 0, True),
        (16, 16, 0, True),
        (17, 16, 0, False),
        (1, 0, 0, False),
        (1, 16, 32, False),
    ],
)
def test_static_decode_capacity_proof(max_tokens, decode_bucket, prefill_bucket, expected):
    assert _static_decode_covers_capacity(max_tokens, decode_bucket, prefill_bucket) is expected


def test_static_only_runner_counts_layers_without_symbolic_programs():
    runner = EmmyGenRunner(
        embed_weight=np.empty((1, 1), dtype=np.float16),
        norm=None,
        pre=[],
        post=[],
        attn_meta=[(1, 1, 1, 1.0), (1, 1, 1, 1.0)],
        np_dtype=np.dtype("float16"),
        pre_decode=[object(), object()],
        post_decode=[object(), object()],
        decode_bucket=1,
        prefill_capacity=1,
    )
    assert runner.num_layers == 2
    assert runner.global_layer_id(0) == 0
    assert runner.global_layer_id(1) == 1
    assert runner.prefill_capacity == 1
    assert runner.has_device_decode
    with pytest.raises(RuntimeError, match="token width 2 exceeds static-only capacity 1"):
        runner.forward_layer_pre(0, np.zeros((2, 1), dtype=np.float16))
    with pytest.raises(RuntimeError, match="token width 2 exceeds static-only capacity 1"):
        runner.forward_layer_post(
            0,
            np.zeros((2, 1), dtype=np.float16),
            np.zeros((2, 1), dtype=np.float16),
        )


def test_pipeline_runner_tracks_absolute_layers_and_boundary_ownership():
    runner = EmmyGenRunner(
        embed_weight=None,
        norm=None,
        hidden_size=8,
        layer_ids=[7, 8],
        pre=[],
        post=[],
        attn_meta=[(2, 4, 1, 0.5), (2, 4, 1, 0.5)],
        np_dtype=np.dtype("float16"),
    )

    assert runner.num_layers == 2
    assert runner.global_layer_id(0) == 7
    assert runner.global_layer_id(1) == 8
    with pytest.raises(RuntimeError, match="does not own the token embedding"):
        runner.embed([0])
    with pytest.raises(RuntimeError, match="does not own the final norm"):
        runner.final_norm(np.zeros((1, 8), dtype=np.float16))


@pytest.mark.parametrize(("quant_method", "coded_trunk"), [("exl3", True), ("awq", True), ("fp8", False)])
def test_create_keeps_storage_coded_trunks_packed(tmp_path, monkeypatch, quant_method, coded_trunk):
    """EXL3/AWQ stay checkpoint-coded; FP8 preserves its decoded trunk lane."""
    import json

    from emmy.compiler.loader import safetensors
    from emmy.compiler.trace import huggingface
    from emmy.serving.gen_runner import EmmyGenRunner

    quant_config = {"quant_method": quant_method}
    if quant_method == "awq":
        quant_config.update(bits=4, version="gemm", zero_point=True)
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": quant_config}))
    seen = {}
    fake_model = object()
    fake_store = {"fmt": quant_method}

    monkeypatch.setattr(safetensors, "warn_if_unpinned", lambda _model_id: None)
    monkeypatch.setattr(huggingface, "quantized_checkpoint_dir", lambda _model_id: tmp_path)

    def fake_load(path, dtype, *, compress_trunk=False, layer_range=None, include_embed=True, include_norm=True):
        seen.update(
            path=path,
            dtype=dtype,
            compress_trunk=compress_trunk,
            layer_range=layer_range,
            include_embed=include_embed,
            include_norm=include_norm,
        )
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
