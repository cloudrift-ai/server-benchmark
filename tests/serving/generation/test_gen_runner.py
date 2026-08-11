"""Fast CPU tests for ``gen_runner`` helpers (no GPU/model). The decode-bucket compile +
correctness are covered on GPU by ``test_gen_runner_gpu.py`` / ``test_vllm_plugin_gen_gpu.py``."""

import numpy as np
import pytest

from emmy.serving.gen_runner import EmmyGenRunner, _pad_rows, _static_decode_covers_capacity


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


def _native_exl3_inputs(*, down_bits=2, codes_dtype=None):
    import torch

    codes_dtype = getattr(torch, codes_dtype) if codes_dtype else torch.int16
    inputs = {}
    for projection, bits in (("gate", 2), ("up", 2), ("down", down_bits)):
        rows, cols = (16, 8) if projection == "down" else (8, 16)
        inputs[f"w_{projection}"] = torch.empty(4, rows, cols, bits * 16, dtype=codes_dtype)
        inputs[f"w_{projection}_suh"] = torch.empty(4, 256 if projection == "down" else 128, dtype=torch.float16)
        inputs[f"w_{projection}_svh"] = torch.empty(4, 128 if projection == "down" else 256, dtype=torch.float16)
    return inputs


def test_native_exl3_moe_spec_accepts_uniform_compressed_silu():
    import torch

    from emmy.serving.exl3_moe import Exl3MoeSpec, fused_m1_spec

    got = fused_m1_spec(
        _native_exl3_inputs(),
        {"w_gate": 2, "w_up": 2, "w_down": 2},
        hidden_size=128,
        intermediate_size=256,
        top_k=2,
        activation=torch.nn.SiLU(),
    )
    assert got == Exl3MoeSpec(2, 2, 128, 256, 4, 2, 128)


@pytest.mark.parametrize(
    ("inputs", "codebooks", "activation"),
    [
        (_native_exl3_inputs(down_bits=3), {"w_gate": 2, "w_up": 2, "w_down": 2}, "silu"),
        (_native_exl3_inputs(), {"w_gate": 2, "w_up": 1, "w_down": 2}, "silu"),
        (_native_exl3_inputs(), {"w_gate": 2, "w_up": 2, "w_down": 2}, "relu"),
        (_native_exl3_inputs(codes_dtype="int32"), {"w_gate": 2, "w_up": 2, "w_down": 2}, "silu"),
    ],
)
def test_native_exl3_moe_spec_rejects_nonuniform_or_wrong_activation(inputs, codebooks, activation):
    import torch

    from emmy.serving.exl3_moe import fused_m1_spec

    act = torch.nn.SiLU() if activation == "silu" else torch.nn.ReLU()
    assert (
        fused_m1_spec(
            inputs,
            codebooks,
            hidden_size=128,
            intermediate_size=256,
            top_k=2,
            activation=act,
        )
        is None
    )


def test_native_exl3_source_is_pinned_and_exports_route_kernel():
    from importlib.resources import files

    from emmy.serving.native.exl3 import source, symbol

    package = files("emmy.serving.native.exl3")
    rendered = source(2, 256, 2)
    assert "void exl3_moe_kernel(EXL3_MOE_KERNEL_ARGS)" in rendered
    assert 'extern "C" __global__ void emmy_exl3_moe_route_m1' in rendered
    assert "const half* scores" in rendered
    assert "791c83073f7f90c44f765a0ceeab7a05fa15b96b" in package.joinpath("README.md").read_text()
    assert package.joinpath("LICENSE.exllamav3").read_text().startswith("MIT License")
    assert symbol(2, 256, 2).startswith("_Z15exl3_moe_kernelILi2ELi256ELi2EE")


def test_native_exl3_gemv_source_selects_volta_narrow_path_and_typed_symbol():
    from emmy.serving.native.exl3 import gemv_source, gemv_symbol

    rendered = gemv_source(5, 2, c_fp32=True, residual=True, compute_capability=(7, 0))
    assert "#if __CUDA_ARCH__ >= 800" in rendered
    assert "gemv_int8_unit_narrow<bits, M, residual, false>" in rendered
    assert "Emmy target sm_70, K5, cb2, c_fp32=true, residual=true" in rendered
    assert gemv_symbol(5, c_fp32=True, residual=True).startswith("_Z24exl3_gemv_int8_sq_kernelILi5ELi1ELb1ELb1EE")

    ampere = gemv_source(5, 2, c_fp32=True, residual=True, compute_capability=(8, 0))
    plain = gemv_source(5, 2, c_fp32=True, residual=False, compute_capability=(7, 0))
    assert rendered != ampere != plain
    assert "Emmy target sm_80" in ampere and "residual=false" in plain

    with pytest.raises(ValueError, match="qualified only"):
        gemv_source(2, 2, c_fp32=True, residual=True, compute_capability=(7, 0))
