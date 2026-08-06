"""FP8 checkpoint ingestion (``loader.quant`` + the safetensors loader's dequant-on-load):
LUT decode against torch's float8 ground truth, block-derived scale application,
quant-spec stamping from a synthetic quantized checkpoint, and end-to-end binding."""

from __future__ import annotations

import json

import numpy as np
import pytest

from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import ConstantOp, QuantSpec
from emmy.compiler.ir.frontend.ir import TransposeOp
from emmy.compiler.loader.quant import decode_f8, dequantize, stamp_quant_specs
from emmy.compiler.loader.safetensors import load_constants_from_safetensors

from ..conftest import requires_cuda

torch = pytest.importorskip("torch")

rng = np.random.default_rng(11)

_TORCH_F8 = {"f8e4m3": torch.float8_e4m3fn, "f8e5m2": torch.float8_e5m2}

# ===================================================================
# LUT decode vs torch float8 ground truth
# ===================================================================


@pytest.mark.parametrize("fmt", ["f8e4m3", "f8e5m2"])
def test_decode_f8_matches_torch_all_codes(fmt):
    """All 256 bit patterns decode exactly as torch's float8 view — including the
    e4m3fn NaN codes (0x7f / 0xff; no infinities) and e5m2's ±inf / NaN band.
    ``assert_array_equal`` treats NaN positions as equal, so NaN placement is checked."""
    codes = np.arange(256, dtype=np.uint8)
    ref = torch.tensor(range(256), dtype=torch.uint8).view(_TORCH_F8[fmt]).float().numpy()
    np.testing.assert_array_equal(decode_f8(codes, fmt), ref)


def test_decode_f8_e4m3_nan_and_no_inf():
    out = decode_f8(np.arange(256, dtype=np.uint8), "f8e4m3")
    assert np.isnan(out[0x7F]) and np.isnan(out[0xFF])
    assert not np.isinf(out).any()  # "fn": finite + NaN, no infinities
    assert out[0x7E] == 448.0 and out[0xFE] == -448.0


def test_decode_f8_e5m2_inf():
    out = decode_f8(np.arange(256, dtype=np.uint8), "f8e5m2")
    assert out[0x7C] == np.inf and out[0xFC] == -np.inf
    assert np.isnan(out[0x7D:0x80]).all() and np.isnan(out[0xFD:]).all()


def test_decode_f8_preserves_shape():
    bits = rng.integers(0, 256, (4, 6)).astype(np.uint8)
    assert decode_f8(bits, "f8e4m3").shape == (4, 6)


# ===================================================================
# dequantize: granularity derived from the shapes
# ===================================================================


def _vals(shape):
    """Decoded-weight stand-in: finite f32 values."""
    return (rng.standard_normal(shape) * 4).astype(np.float32)


def test_dequantize_per_tensor_scalar_scale():
    w = _vals((8, 16))
    for scale in (np.float32(0.5), np.array([0.5], dtype=np.float32), np.array([[0.5]], dtype=np.float32)):
        np.testing.assert_array_equal(dequantize(w, scale), w * np.float32(0.5))


def test_dequantize_per_out_channel():
    w = _vals((8, 16))
    s = (rng.standard_normal((8, 1)) + 2).astype(np.float32)
    np.testing.assert_array_equal(dequantize(w, s), w * s)


def test_dequantize_2d_block():
    w = _vals((8, 16))
    s = (rng.standard_normal((2, 4)) + 2).astype(np.float32)  # derived block (4, 4)
    ref = w * np.repeat(np.repeat(s, 4, axis=0), 4, axis=1)
    np.testing.assert_array_equal(dequantize(w, s), ref)


def test_dequantize_inverse_divides():
    w = _vals((4, 8))
    s = np.full((4, 1), 2.0, dtype=np.float32)
    np.testing.assert_array_equal(dequantize(w, s, inverse=True), w / s)


def test_dequantize_rejects_non_divisible_scale():
    with pytest.raises(ValueError, match="does not evenly divide"):
        dequantize(_vals((8, 16)), np.ones((3, 1), dtype=np.float32))


def test_dequantize_rejects_rank_mismatch():
    with pytest.raises(ValueError, match="rank"):
        dequantize(_vals((8, 16)), np.ones((2, 2, 2), dtype=np.float32))


# ===================================================================
# Synthetic quantized checkpoint fixture
# ===================================================================

_FP8_QC = {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic", "modules_to_not_convert": []}


def _write_checkpoint(dirpath, tensors, quant_config=None):
    """Single-shard safetensors dir + config.json (with optional quantization_config)."""
    from safetensors.torch import save_file

    save_file({k: v.clone() for k, v in tensors.items()}, str(dirpath / "model.safetensors"))
    cfg = {"model_type": "test"}
    if quant_config is not None:
        cfg["quantization_config"] = quant_config
    (dirpath / "config.json").write_text(json.dumps(cfg))


def _fp8_tensor(bits: np.ndarray, fmt="f8e4m3"):
    return torch.from_numpy(np.ascontiguousarray(bits)).view(_TORCH_F8[fmt])


def _weight_graph(shape=(8, 16), dtype="f32", source_path="layer.weight", load_ops=(), out_shape=None):
    g = Graph()
    g.add_node(
        op=ConstantOp(name="p_w", source_path=source_path, source_shape=shape, source_dtype=dtype, load_ops=tuple(load_ops)),
        inputs=[],
        output=Tensor("p_w", out_shape or shape, dtype),
        node_id="p_w",
    )
    return g


def _finite_bits(shape):
    """fp8 bit patterns that avoid the e4m3 NaN codes, so scale references stay finite."""
    bits = rng.integers(0, 256, shape).astype(np.uint8)
    bits[bits == 0x7F] = 0x00
    bits[bits == 0xFF] = 0x80
    return bits


# ===================================================================
# Stamping: quantization_config + index pairing → QuantSpec
# ===================================================================


def test_stamp_pairs_weight_with_scale(tmp_path):
    bits = _finite_bits((8, 16))
    scale = torch.full((8, 1), 0.25, dtype=torch.float32)
    _write_checkpoint(tmp_path, {"layer.weight": _fp8_tensor(bits), "layer.weight_scale": scale}, _FP8_QC)
    g = _weight_graph()
    assert stamp_quant_specs(g, str(tmp_path)) == 1
    spec = g.nodes["p_w"].op.quant
    assert spec == QuantSpec(scale_path="layer.weight_scale", scale_shape=(8, 1), scale_dtype="f32", inverse=False)


def test_stamp_records_e5m2_fmt(tmp_path):
    """``QuantSpec.fmt`` carries the checkpoint's storage format (e4m3 is the
    default; an e5m2-stored weight must stamp its own token — the M2 expansion
    types the fp8 constant from it)."""
    bits = _finite_bits((8, 16))
    scale = torch.full((8, 1), 0.25, dtype=torch.float32)
    _write_checkpoint(tmp_path, {"layer.weight": _fp8_tensor(bits, "f8e5m2"), "layer.weight_scale": scale}, _FP8_QC)
    g = _weight_graph()
    assert stamp_quant_specs(g, str(tmp_path)) == 1
    assert g.nodes["p_w"].op.quant.fmt == "f8e5m2"


def test_stamp_weight_scale_inv_sets_inverse(tmp_path):
    bits = _finite_bits((8, 16))
    scale = torch.full((1,), 2.0, dtype=torch.float32)
    _write_checkpoint(tmp_path, {"layer.weight": _fp8_tensor(bits), "layer.weight_scale_inv": scale}, _FP8_QC)
    g = _weight_graph()
    assert stamp_quant_specs(g, str(tmp_path)) == 1
    assert g.nodes["p_w"].op.quant.inverse is True


def test_stamp_honors_modules_to_not_convert(tmp_path):
    bits = _finite_bits((8, 16))
    qc = dict(_FP8_QC, modules_to_not_convert=["layer"])
    _write_checkpoint(tmp_path, {"layer.weight": _fp8_tensor(bits), "layer.weight_scale": torch.ones(1)}, qc)
    g = _weight_graph()
    assert stamp_quant_specs(g, str(tmp_path)) == 0
    assert g.nodes["p_w"].op.quant is None


def test_stamp_skips_weight_without_scale(tmp_path):
    _write_checkpoint(tmp_path, {"layer.weight": _fp8_tensor(_finite_bits((8, 16)))}, _FP8_QC)
    g = _weight_graph()
    assert stamp_quant_specs(g, str(tmp_path)) == 0


def test_stamp_skips_non_fp8_weight(tmp_path):
    # quantization_config present, but this weight is stored at full precision
    # (a modules-kept-in-bf16 member) — no spec even with a stray scale tensor.
    _write_checkpoint(tmp_path, {"layer.weight": torch.ones(8, 16), "layer.weight_scale": torch.ones(1)}, _FP8_QC)
    assert stamp_quant_specs(_weight_graph(), str(tmp_path)) == 0


def test_stamp_noop_on_unquantized_checkpoint(tmp_path):
    _write_checkpoint(tmp_path, {"layer.weight": torch.ones(8, 16)})
    g = _weight_graph()
    assert stamp_quant_specs(g, str(tmp_path)) == 0
    assert g.nodes["p_w"].op.quant is None


def test_stamp_compressed_tensors_fp8_scheme(tmp_path):
    qc = {
        "quant_method": "compressed-tensors",
        "config_groups": {"group_0": {"weights": {"type": "float", "num_bits": 8, "strategy": "channel"}}},
        "ignore": ["lm_head"],
    }
    bits = _finite_bits((8, 16))
    tensors = {
        "layer.weight": _fp8_tensor(bits),
        "layer.weight_scale": torch.full((8, 1), 0.5, dtype=torch.float32),
        "lm_head.weight": _fp8_tensor(_finite_bits((4, 16))),
        "lm_head.weight_scale": torch.ones(1),
    }
    _write_checkpoint(tmp_path, tensors, qc)
    g = _weight_graph()
    g.add_node(
        op=ConstantOp(name="p_h", source_path="lm_head.weight", source_shape=(4, 16), source_dtype="f32"),
        inputs=[],
        output=Tensor("p_h", (4, 16), "f32"),
        node_id="p_h",
    )
    assert stamp_quant_specs(g, str(tmp_path)) == 1  # lm_head ignored
    assert g.nodes["p_w"].op.quant is not None
    assert g.nodes["p_h"].op.quant is None


def test_stamp_compressed_tensors_int_scheme_is_noop(tmp_path):
    qc = {
        "quant_method": "compressed-tensors",
        "config_groups": {"group_0": {"weights": {"type": "int", "num_bits": 8}}},
    }
    _write_checkpoint(tmp_path, {"layer.weight": _fp8_tensor(_finite_bits((8, 16))), "layer.weight_scale": torch.ones(1)}, qc)
    assert stamp_quant_specs(_weight_graph(), str(tmp_path)) == 0


def test_stamp_honors_regex_ignore(tmp_path):
    qc = {
        "quant_method": "compressed-tensors",
        "config_groups": {"group_0": {"weights": {"type": "float", "num_bits": 8}}},
        "ignore": ["re:.*lay.*"],
    }
    _write_checkpoint(tmp_path, {"layer.weight": _fp8_tensor(_finite_bits((8, 16))), "layer.weight_scale": torch.ones(1)}, qc)
    assert stamp_quant_specs(_weight_graph(), str(tmp_path)) == 0


# ===================================================================
# Loader: dequant-on-load binding vs the torch dequant reference
# ===================================================================


def _torch_ref(bits, scale_np, *, fmt="f8e4m3", inverse=False):
    """Reference dequant: torch float8 decode, then the same broadcast in plain numpy."""
    vals = torch.from_numpy(np.ascontiguousarray(bits)).view(_TORCH_F8[fmt]).float().numpy()
    s = np.asarray(scale_np, dtype=np.float32)
    if s.size == 1:
        s = s.reshape(())
    else:
        s = np.repeat(np.repeat(s, vals.shape[0] // s.shape[0], axis=0), vals.shape[1] // s.shape[1], axis=1)
    return vals / s if inverse else vals * s


@pytest.mark.parametrize(
    "scale_shape",
    [(), (8, 1), (2, 4)],
    ids=["per-tensor", "per-channel", "block"],
)
def test_loader_dequantizes_stamped_weight(tmp_path, scale_shape):
    bits = _finite_bits((8, 16))
    scale_np = np.asarray(np.abs(rng.standard_normal(scale_shape)) + 0.5, dtype=np.float32)  # () stays a 0-d ndarray
    _write_checkpoint(
        tmp_path,
        {"layer.weight": _fp8_tensor(bits), "layer.weight_scale": torch.from_numpy(scale_np)},
        _FP8_QC,
    )
    g = _weight_graph()
    assert stamp_quant_specs(g, str(tmp_path)) == 1
    out = load_constants_from_safetensors(g, str(tmp_path))
    np.testing.assert_array_equal(out["p_w"], _torch_ref(bits, scale_np))
    assert out["p_w"].dtype == np.float32


def test_loader_dequantizes_inverse_scale(tmp_path):
    bits = _finite_bits((8, 16))
    scale_np = np.full((8, 1), 2.0, dtype=np.float32)
    _write_checkpoint(
        tmp_path,
        {"layer.weight": _fp8_tensor(bits), "layer.weight_scale_inv": torch.from_numpy(scale_np)},
        _FP8_QC,
    )
    g = _weight_graph()
    assert stamp_quant_specs(g, str(tmp_path)) == 1
    out = load_constants_from_safetensors(g, str(tmp_path))
    np.testing.assert_array_equal(out["p_w"], _torch_ref(bits, scale_np, inverse=True))


def test_loader_casts_to_traced_compute_dtype(tmp_path):
    bits = _finite_bits((8, 16))
    scale_np = np.array([[0.75]], dtype=np.float32)
    _write_checkpoint(
        tmp_path,
        {"layer.weight": _fp8_tensor(bits), "layer.weight_scale": torch.from_numpy(scale_np)},
        _FP8_QC,
    )
    g = _weight_graph(dtype="f16")
    stamp_quant_specs(g, str(tmp_path))
    out = load_constants_from_safetensors(g, str(tmp_path))
    assert out["p_w"].dtype == np.float16
    np.testing.assert_array_equal(out["p_w"], _torch_ref(bits, scale_np).astype(np.float16))


def test_loader_dequantizes_before_load_ops(tmp_path):
    """Dequant happens when the SOURCE is read, before the ``load_ops`` chain: a
    per-channel scale applied after the fold's transpose would mis-broadcast."""
    bits = _finite_bits((8, 16))
    scale_np = (np.abs(rng.standard_normal((8, 1))) + 0.5).astype(np.float32)
    _write_checkpoint(
        tmp_path,
        {"layer.weight": _fp8_tensor(bits), "layer.weight_scale": torch.from_numpy(scale_np)},
        _FP8_QC,
    )
    g = _weight_graph(load_ops=(TransposeOp(axes=(1, 0)),), out_shape=(16, 8))
    stamp_quant_specs(g, str(tmp_path))
    out = load_constants_from_safetensors(g, str(tmp_path))
    np.testing.assert_array_equal(out["p_w"], _torch_ref(bits, scale_np).T)


@pytest.mark.parametrize("fmt", ["f8e4m3", "f8e5m2"])
def test_loader_reads_specless_fp8_as_plain_values(tmp_path, fmt):
    """A spec-less fp8 tensor bound at a NON-f8 graph dtype decodes to values (no
    scale). Since M2a the decode is dtype-directed: decode-to-values applies only
    when the graph wants a non-f8 dtype; an f8-dtype constant binds raw bits
    instead (the expanded form — see the raw-bits test below). This graph traces
    the weight at f32, so the M1 behavior is unchanged here."""
    bits = _finite_bits((8, 16))
    _write_checkpoint(tmp_path, {"layer.weight": _fp8_tensor(bits, fmt)})
    out = load_constants_from_safetensors(_weight_graph(), str(tmp_path))
    ref = torch.from_numpy(bits).view(_TORCH_F8[fmt]).float().numpy()
    np.testing.assert_array_equal(out["p_w"], ref)


@pytest.mark.parametrize("fmt", ["f8e4m3", "f8e5m2"])
def test_loader_binds_f8_dtype_constant_as_raw_bits(tmp_path, fmt):
    """A spec-less constant whose GRAPH dtype is an f8 dtype binds the raw uint8
    bit pattern — no LUT decode, no scale. This is the M2 expanded form: the
    graph's own dequant cone owns the value semantics, so handing over decoded
    values would double-decode."""
    bits = _finite_bits((8, 16))
    _write_checkpoint(tmp_path, {"layer.weight": _fp8_tensor(bits, fmt)})
    out = load_constants_from_safetensors(_weight_graph(dtype=fmt), str(tmp_path))
    assert out["p_w"].dtype == np.uint8
    np.testing.assert_array_equal(out["p_w"], bits)


def test_loader_unquantized_checkpoint_unchanged(tmp_path):
    """Plain f32 checkpoint through the same path — zero behavior change."""
    w = rng.standard_normal((8, 16)).astype(np.float32)
    _write_checkpoint(tmp_path, {"layer.weight": torch.from_numpy(w)})
    g = _weight_graph()
    assert stamp_quant_specs(g, str(tmp_path)) == 0
    np.testing.assert_array_equal(load_constants_from_safetensors(g, str(tmp_path))["p_w"], w)


# ===================================================================
# QuantSpec serialization round-trip
# ===================================================================


def test_quant_spec_survives_graph_json_roundtrip():
    g = _weight_graph()
    spec = QuantSpec(scale_path="layer.weight_scale", scale_shape=(2, 4), scale_dtype="f32", inverse=True)
    g.nodes["p_w"].op.quant = spec
    g2 = Graph.from_dict(json.loads(json.dumps(g.to_dict())))
    assert g2.nodes["p_w"].op.quant == spec


# ===================================================================
# Dequantized state dict + quantized-checkpoint detection
# ===================================================================


def test_load_dequantized_state_dict(tmp_path):
    from emmy.compiler.loader.quant import load_dequantized_state_dict

    bits = _finite_bits((8, 16))
    scale_np = np.full((8, 1), 0.25, dtype=np.float32)
    tensors = {
        "layer.weight": _fp8_tensor(bits),
        "layer.weight_scale": torch.from_numpy(scale_np),
        "norm.weight": torch.ones(16, dtype=torch.bfloat16) * 2,  # bf16 storage → f32 values
        "other.weight": torch.full((4, 16), 3.0),
    }
    _write_checkpoint(tmp_path, tensors, _FP8_QC)
    sd = load_dequantized_state_dict(tmp_path)
    np.testing.assert_array_equal(sd["layer.weight"], _torch_ref(bits, scale_np))
    np.testing.assert_array_equal(sd["norm.weight"], np.full(16, 2.0, dtype=np.float32))
    np.testing.assert_array_equal(sd["other.weight"], np.full((4, 16), 3.0, dtype=np.float32))
    assert "layer.weight_scale" not in sd  # consumed by the pairing


def test_load_dequantized_state_dict_unquantized_passthrough(tmp_path):
    from emmy.compiler.loader.quant import load_dequantized_state_dict

    w = rng.standard_normal((4, 8)).astype(np.float32)
    _write_checkpoint(tmp_path, {"layer.weight": torch.from_numpy(w)})
    np.testing.assert_array_equal(load_dequantized_state_dict(tmp_path)["layer.weight"], w)


def test_quantized_checkpoint_dir_detection(tmp_path):
    from emmy.compiler.trace.huggingface import quantized_checkpoint_dir

    quantized = tmp_path / "fp8"
    plain = tmp_path / "plain"
    quantized.mkdir()
    plain.mkdir()
    _write_checkpoint(quantized, {"layer.weight": _fp8_tensor(_finite_bits((4, 8)))}, _FP8_QC)
    _write_checkpoint(plain, {"layer.weight": torch.ones(4, 8)})
    assert quantized_checkpoint_dir(str(quantized)) == quantized
    assert quantized_checkpoint_dir(str(plain)) is None


# ===================================================================
# emmy compile / run wiring: whole-model trace of a quantized checkpoint
# (the ``_trace_model`` seam both commands share via ``load_or_trace``)
# ===================================================================


def _tiny_fp8_checkpoint(dirpath):
    """Tiny Llama-architecture checkpoint with every decoder-layer projection weight
    quantized to fp8 per-out-channel. Returns ``(config, ref_sd)`` where ``ref_sd``
    is the dequantized torch f32 state dict (the accuracy reference)."""
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(config).float().eval()
    tensors: dict = {}
    ref_sd: dict = {}
    for name, t in model.state_dict().items():
        t = t.detach().cpu()
        if name.endswith(".weight") and t.ndim == 2 and ".layers." in name:  # the linear projections
            scale = (t.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 448.0).float()
            q = (t / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            tensors[name] = q
            tensors[name[: -len(".weight")] + ".weight_scale"] = scale
            ref_sd[name] = q.float() * scale
        else:
            tensors[name] = t
            ref_sd[name] = t
    from safetensors.torch import save_file

    save_file({k: v.clone() for k, v in tensors.items()}, str(dirpath / "model.safetensors"))
    cfg = config.to_dict()
    cfg["quantization_config"] = dict(_FP8_QC, modules_to_not_convert=["lm_head"])
    (dirpath / "config.json").write_text(json.dumps(cfg))
    return config, ref_sd


def test_trace_model_unquantized_checkpoint_takes_existing_path(tmp_path):
    """The same seam on an UNQUANTIZED checkpoint: detection returns None, the model
    loads through the pre-existing ``from_pretrained`` branch, and zero specs stamp."""
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(config).float().eval()
    model.save_pretrained(str(tmp_path))
    from emmy.commands.compile import _trace_model

    graph, (wrapper, _args, _kws) = _trace_model(str(tmp_path), None, 16)
    assert all(op.quant is None for _nid, op in graph.loadable_constants())
    # from_pretrained bound the real checkpoint weights, as before the wiring.
    got = dict(wrapper.named_parameters())["model.model.embed_tokens.weight"]
    torch.testing.assert_close(got, model.model.embed_tokens.weight, rtol=0, atol=0)


def test_trace_model_stamps_specs_and_binds_dequantized_twin(tmp_path):
    """The compile/run seam on a quantized checkpoint: the traced twin carries the
    DEQUANTIZED real weights (not from_config's random init), and quant specs land on
    exactly the fp8-stored projection weights before any pass runs."""
    _config, ref_sd = _tiny_fp8_checkpoint(tmp_path)
    from emmy.commands.compile import _trace_model

    graph, (wrapper, _args, _kws) = _trace_model(str(tmp_path), None, 16)

    quants = {op.source_path: op.quant for _nid, op in graph.loadable_constants()}
    stamped = {p for p, q in quants.items() if q is not None}
    assert stamped, "no quant specs stamped"
    assert all(".layers." in p and p.endswith(".weight") for p in stamped)
    assert any("q_proj" in p for p in stamped) and any("down_proj" in p for p in stamped)
    for path, q in quants.items():
        if "embed_tokens" in path or "lm_head" in path or "norm" in path:
            assert q is None, f"unexpected spec on {path}"

    params = dict(wrapper.named_parameters())
    for name, ref in ref_sd.items():
        got = params["model." + name]  # the trace wrapper nests the CausalLM under .model
        torch.testing.assert_close(got, ref, rtol=0, atol=0)


@requires_cuda
def test_quantized_checkpoint_e2e_cuda_expanded(tmp_path, monkeypatch):
    """The SAME tiny quantized model with ``EMMY_FP8_EXPAND=1`` — the M2b correctness anchor: the
    dequant cone rides the graph into the kernels (fp8 bits in device memory, decode + mul-hoisted
    scale realized in-kernel), same accuracy gate as the M1 bind-time-dequant test. Constants bind
    from the CHECKPOINT (raw fp8 bits + scale tensors — the wrapper's parameters carry dequantized
    values, which the expanded form must not see); computed buffers (mask / rotary) still come
    from the traced wrapper."""
    monkeypatch.setenv("EMMY_FP8_EXPAND", "1")
    transformers = pytest.importorskip("transformers")
    config, ref_sd = _tiny_fp8_checkpoint(tmp_path)
    from emmy.commands.compile import _trace_model
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.loader.binder import bind_constants

    graph, (wrapper, _args, _kws) = _trace_model(str(tmp_path), None, 16)
    backend = CudaBackend()
    compiled = backend.compile(graph)

    fp8_nodes = [nid for nid, node in compiled.nodes.items() if getattr(node.output.dtype, "name", "") == "f8e4m3"]
    assert fp8_nodes, "no fp8-dtype constant survived to the compiled graph — the expansion did not fire"

    buf_sources: dict[str, np.ndarray] = {}
    for path, t in wrapper.named_buffers(remove_duplicate=False):
        buf_sources[path] = t.detach().cpu().numpy().astype(np.float32, copy=False)
    input_data: dict = dict(bind_constants(compiled, buf_sources))
    input_data.update(load_constants_from_safetensors(compiled, str(tmp_path)))

    ids = torch.randint(0, config.vocab_size, (1, 16), generator=torch.Generator().manual_seed(3))
    input_data[compiled.inputs[0]] = ids.numpy()
    result, _ = backend.run(compiled, input_data=input_data)
    emmy_logits = result.outputs[compiled.outputs[0]].reshape(1, 16, config.vocab_size)

    ref_model = transformers.AutoModelForCausalLM.from_config(config).float().eval()
    ref_model.load_state_dict(ref_sd)
    with torch.no_grad():
        ref_logits = ref_model(input_ids=ids).logits.numpy()

    assert not np.isnan(emmy_logits).any()
    assert np.abs(ref_logits).max() > 0.05, "reference logits suspiciously small; tolerance would be trivial"
    max_diff = np.abs(emmy_logits - ref_logits).max()
    assert max_diff < 5e-3, f"max_diff={max_diff} vs eager dequant reference (EMMY_FP8_EXPAND=1)"


@requires_cuda
def test_quantized_checkpoint_e2e_cuda(tmp_path):
    """Whole tiny quantized model through the same seam ``emmy compile`` / ``emmy run``
    use, compiled on the CUDA backend and compared against the dequantized eager
    reference — the M1 deliverable ("any FP8 checkpoint compiles and runs")."""
    transformers = pytest.importorskip("transformers")
    config, ref_sd = _tiny_fp8_checkpoint(tmp_path)
    from emmy.commands.compile import _trace_model
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.loader.binder import bind_constants

    graph, (wrapper, _args, _kws) = _trace_model(str(tmp_path), None, 16)
    backend = CudaBackend()
    compiled = backend.compile(graph)

    # Bind the way ``emmy run`` binds a whole model: sources from the traced
    # wrapper (its parameters carry the dequantized real weights; its buffers
    # carry the precomputed mask / rotary / position_ids).
    sources: dict[str, np.ndarray] = {}
    for path, t in wrapper.named_parameters(remove_duplicate=False):
        sources[path] = t.detach().cpu().numpy().astype(np.float32, copy=False)
    for path, t in wrapper.named_buffers(remove_duplicate=False):
        sources[path] = t.detach().cpu().numpy().astype(np.float32, copy=False)

    ids = torch.randint(0, config.vocab_size, (1, 16), generator=torch.Generator().manual_seed(3))
    input_data = {compiled.inputs[0]: ids.numpy()}
    input_data.update(bind_constants(compiled, sources))
    result, _ = backend.run(compiled, input_data=input_data)
    emmy_logits = result.outputs[compiled.outputs[0]].reshape(1, 16, config.vocab_size)

    # Independent eager reference: fresh model from config, dequantized weights
    # computed in this test (not through emmy's loader).
    ref_model = transformers.AutoModelForCausalLM.from_config(config).float().eval()
    ref_model.load_state_dict(ref_sd)
    with torch.no_grad():
        ref_logits = ref_model(input_ids=ids).logits.numpy()

    assert not np.isnan(emmy_logits).any()
    assert np.abs(ref_logits).max() > 0.05, "reference logits suspiciously small; tolerance would be trivial"
    max_diff = np.abs(emmy_logits - ref_logits).max()
    assert max_diff < 5e-3, f"max_diff={max_diff} vs eager dequant reference"


# ===================================================================
# Containment: quant metadata must not leak past the frontend band
# ===================================================================

# The design rule: ``ConstantOp.quant``
# is frontend-band scaffolding — stamped after trace, consulted by the loader and exactly two
# decomposition rules, CONSUMED by ``180_expand_quantized_constant``. Everything past the
# frontend (lowering, backends, search) must stay graph-structure-driven: a quantized weight
# past the band is just constants + algebra. This gate makes the rule mechanical: a new
# reader of the metadata must be frontend/loader-band code and must join the allowlist here,
# with that justification — anything else is the leak this test exists to stop.
_QUANT_ALLOWLIST = {
    "emmy/commands/compile.py",  # stamping call site (post-trace, pre-pipeline)
    "emmy/compiler/graph.py",  # QuantSpec constructor-repr serialization
    "emmy/compiler/ir/base.py",  # the definition
    "emmy/compiler/loader/binder.py",  # bind-time raw-bits routing
    "emmy/compiler/loader/__init__.py",  # re-export
    "emmy/compiler/loader/quant.py",  # decode / dequant / stamping
    "emmy/compiler/loader/safetensors.py",  # dequant-on-load
    "emmy/compiler/pipeline/passes/frontend/decomposition/035_merge_sibling_linears.py",  # pristine guard
    "emmy/compiler/pipeline/passes/frontend/decomposition/180_expand_quantized_constant.py",  # the consumer
    "emmy/compiler/trace/huggingface.py",  # quantized-twin construction
}


def test_quant_metadata_stays_in_the_frontend_band():
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    pat = re.compile(r"QuantSpec|\.quant\b|\bquant=")
    offenders = {str(p.relative_to(root)) for p in (root / "emmy").rglob("*.py") if pat.search(p.read_text())} - _QUANT_ALLOWLIST
    assert not offenders, (
        f"ConstantOp.quant / QuantSpec referenced outside the frontend/loader band: {sorted(offenders)}. "
        "The kernel path is graph-structure-driven by design — do not consume quant metadata there "
        "(see the allowlist comment above)."
    )
