"""FP8 checkpoint ingestion: LUT decode, scale algebra, birth-time spelling, and binding."""

from __future__ import annotations

import json

import numpy as np
import pytest

from emmy.compiler.dtype import F8E4M3, decode_f4x2
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.frontend.ir import LinearOp, ReshapeOp
from emmy.compiler.ir.tensor.ir import ElementwiseOp, ReduceOp
from emmy.compiler.loader.quant import (
    _fp4_quant_config,
    _fp8_quant_config,
    _static_fp4_activation_declared,
    decode_f8,
    dequantize,
    dequantize_awq4,
    dequantize_nvfp4,
    encode_f4x2,
    encode_f8,
    fuse_nvfp4_scales,
    quantize_nvfp4,
    spell_dynamic_fp8_activations,
    spell_quantized_constants,
    spell_quantized_inputs,
    spell_static_fp4_activations,
    unpack_awq4,
)
from emmy.compiler.loader.safetensors import load_constants_from_safetensors
from tests.compiler.helpers import requires_cuda

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
# NVFP4 two-level scale math
# ===================================================================


def _e2m1(*, sign: int, exp: int, man: int) -> int:
    """e2m1 code from its fields: ``[s | e e | m]`` (value ``(-1)^s * (1 + m/2) * 2^(e-1)``, subnormal e=0 → m/2)."""
    assert 0 <= sign < 2 and 0 <= exp < 4 and 0 <= man < 2, (sign, exp, man)
    return sign << 3 | exp << 1 | man


def _e4m3(*, sign: int, exp: int, man: int) -> int:
    """e4m3 code from its fields: ``[s | e e e e | m m m]`` (value ``(-1)^s * (1 + m/8) * 2^(e-7)``)."""
    assert 0 <= sign < 2 and 0 <= exp < 16 and 0 <= man < 8, (sign, exp, man)
    return sign << 7 | exp << 3 | man


def _pack(*, even: int, odd: int) -> int:
    """One byte from two e2m1 codes: ``b = v_even + 16 * v_odd``."""
    assert 0 <= even < 16 and 0 <= odd < 16, (even, odd)
    return even | odd << 4


def test_fuse_nvfp4_scales_e4m3_exact_in_f16():
    # With scale_2 = 1, fusing is the e4m3 decode itself — every finite code must
    # survive the f16 round-trip exactly (3 mantissa bits, range ±448 ⊂ f16).
    codes = np.arange(256, dtype=np.uint8)
    fused = fuse_nvfp4_scales(codes, np.array(1.0, dtype=np.float32))
    ref = decode_f8(codes, F8E4M3.name)
    finite = np.isfinite(ref)
    np.testing.assert_array_equal(fused.astype(np.float32)[finite], ref[finite])


def test_dequantize_nvfp4_hand_example():
    # One row, K=32, two blocks: values 1.0 then 6.0, block scales 2.0 then 1.0, scale_2 = 2.
    one = _e2m1(sign=0, exp=1, man=0)
    six = _e2m1(sign=0, exp=3, man=1)
    packed = np.array([[_pack(even=one, odd=one)] * 8 + [_pack(even=six, odd=six)] * 8], dtype=np.uint8)
    scale_bits = np.array([[_e4m3(sign=0, exp=8, man=0), _e4m3(sign=0, exp=7, man=0)]], dtype=np.uint8)
    out = dequantize_nvfp4(packed, scale_bits, np.array(2.0, dtype=np.float32))
    np.testing.assert_array_equal(out, np.array([[1.0 * 2 * 2] * 16 + [6.0 * 1 * 2] * 16], dtype=np.float32))


def test_dequantize_nvfp4_matches_unfused_reference():
    # Random codes vs an independently-broadcast reference built from the same fused
    # scale — equality is exact (see the docstring's significand-budget argument).
    packed = rng.integers(0, 256, (4, 16)).astype(np.uint8)  # logical K = 32
    scale_bits = rng.integers(0, 0x7F, (4, 2)).astype(np.uint8)  # finite e4m3 codes
    scale_2 = np.array(0.03125, dtype=np.float32)
    out = dequantize_nvfp4(packed, scale_bits, scale_2)
    fused = (decode_f8(scale_bits, F8E4M3.name) * scale_2).astype(np.float16).astype(np.float32)
    np.testing.assert_array_equal(out, decode_f4x2(packed) * np.repeat(fused, 16, axis=1))


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
_AWQ_QC = {"quant_method": "awq", "bits": 4, "group_size": 4, "zero_point": True, "version": "gemm"}

# The two NVFP4 dialects, pruned to the fields the recognizer reads plus ``ignore``
# (which the spellers consume). Field shapes mirror nvidia/Qwen3-8B-NVFP4 (modelopt)
# and RedHatAI/Qwen3-8B-NVFP4 (compressed-tensors).
_FP4_MODELOPT_QC = {"quant_method": "modelopt", "quant_algo": "NVFP4", "ignore": ["lm_head"]}
_FP4_CT_QC = {
    "quant_method": "compressed-tensors",
    "format": "nvfp4-pack-quantized",
    "config_groups": {"group_0": {"weights": {"type": "float", "num_bits": 4, "group_size": 16}, "targets": ["Linear"]}},
    "ignore": ["lm_head"],
}


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


def _weight_graph(shape=(8, 16), dtype="f32", source_path="layer.weight"):
    g = Graph()
    g.add_node(
        op=ConstantOp(name="p_w", source_path=source_path, source_shape=shape, source_dtype=dtype),
        inputs=[],
        output=Tensor("p_w", shape, dtype),
        node_id="p_w",
    )
    g.inputs, g.outputs = [], ["p_w"]
    return g


def _finite_bits(shape):
    """fp8 bit patterns that avoid the e4m3 NaN codes, so scale references stay finite."""
    bits = rng.integers(0, 256, shape).astype(np.uint8)
    bits[bits == 0x7F] = 0x00
    bits[bits == 0xFF] = 0x80
    return bits


def _pack_awq4(values):
    """AutoAWQ GEMM's output-channel packing, for compact synthetic fixtures."""
    values = np.asarray(values, dtype=np.int32)
    assert values.ndim == 2 and values.shape[1] % 8 == 0
    packed = np.zeros((values.shape[0], values.shape[1] // 8), dtype=np.uint32)
    order = (0, 2, 4, 6, 1, 3, 5, 7)
    for word in range(packed.shape[1]):
        for slot, logical in enumerate(order):
            packed[:, word] |= values[:, word * 8 + logical].astype(np.uint32) << np.uint32(slot * 4)
    return packed.view(np.int32)


def _spelled(tmp_path, *, scale_shape=(8, 1), fmt="f8e4m3", inverse=False, dtype="f32", qc=_FP8_QC):
    """Write a one-weight fp8 checkpoint, spell the graph, return ``(graph, bits, scale)``."""
    bits = _finite_bits((8, 16))
    scale = (np.abs(rng.standard_normal(scale_shape)) + 0.5).astype(np.float32) if scale_shape else np.float32(0.25).reshape(())
    key = "layer.weight_scale_inv" if inverse else "layer.weight_scale"
    _write_checkpoint(tmp_path, {"layer.weight": _fp8_tensor(bits, fmt), key: torch.from_numpy(np.asarray(scale))}, qc)
    g = _weight_graph(dtype=dtype)
    assert spell_quantized_constants(g, str(tmp_path)) == 1
    return g, bits, scale


def _constants(graph: Graph) -> dict[str, ConstantOp]:
    return {nid: n.op for nid, n in graph.nodes.items() if isinstance(n.op, ConstantOp)}


def _ops_by_type(graph: Graph, op_type) -> list:
    return [n for n in graph.nodes.values() if isinstance(n.op, op_type)]


# ===================================================================
# Birth-time spelling: quantization_config + index pairing → in-graph algebra
# ===================================================================


def test_spell_per_tensor_minimal_broadcast_form(tmp_path):
    g, _bits, _scale = _spelled(tmp_path, scale_shape=(1, 1))
    consts = _constants(g)
    w = next(op for op in consts.values() if op.source_path == "layer.weight")
    assert w.source_dtype == "f8e4m3"  # the bits constant carries the STORAGE dtype
    scale = next(op for op in consts.values() if op.source_path == "layer.weight_scale")
    assert scale.source_shape == (1, 1)
    casts = [n for n in _ops_by_type(g, ElementwiseOp) if n.op.op.name == "from_f8e4m3"]
    assert len(casts) == 1
    # Degenerate block: no reshape pair — the scale broadcasts straight onto the weight.
    assert not _ops_by_type(g, ReshapeOp)
    assert any(n.op.op.name == "multiply" for n in _ops_by_type(g, ElementwiseOp))


def test_spell_per_channel_no_reshape_pair(tmp_path):
    g, _bits, _scale = _spelled(tmp_path, scale_shape=(8, 1))
    assert not _ops_by_type(g, ReshapeOp)
    scale_node = next(n for n in g.nodes.values() if isinstance(n.op, ConstantOp) and n.op.source_path == "layer.weight_scale")
    assert tuple(d.as_static() for d in scale_node.output.shape) == (8, 1)
    assert not scale_node.op.load_ops  # stored shape == declared shape


def test_spell_2d_block_interleaved_reshape_pair(tmp_path):
    g, _bits, _scale = _spelled(tmp_path, scale_shape=(2, 4))  # derived block (4, 4)
    shapes = sorted(tuple(int(d) for d in n.op.shape) for n in _ops_by_type(g, ReshapeOp))
    assert shapes == [(2, 4, 4, 4), (8, 16)]  # blocked view + the reshape back
    scale_node = next(n for n in g.nodes.values() if isinstance(n.op, ConstantOp) and n.op.source_path == "layer.weight_scale")
    # The scale binds at the interleaved (grid, 1) layout via its own load_ops reshape.
    assert tuple(d.as_static() for d in scale_node.output.shape) == (2, 1, 4, 1)
    assert any(isinstance(lop, ReshapeOp) for lop in scale_node.op.load_ops)


def test_spell_scale_inv_multiplies(tmp_path):
    """DeepSeek's ``weight_scale_inv`` names the dequant MULTIPLIER (the inverse of the
    quantization scale), not a reciprocal to divide by — ``q * s``, as its own ``weight_dequant``
    and vLLM's block-fp8 path compute it."""
    g, _bits, _scale = _spelled(tmp_path, scale_shape=(8, 1), inverse=True)
    names = [n.op.op.name for n in _ops_by_type(g, ElementwiseOp)]
    assert "multiply" in names and "divide" not in names


def test_spell_e5m2_selects_its_cast(tmp_path):
    g, _bits, _scale = _spelled(tmp_path, scale_shape=(1, 1), fmt="f8e5m2")
    w = next(op for op in _constants(g).values() if op.source_path == "layer.weight")
    assert w.source_dtype == "f8e5m2"
    assert any(n.op.op.name == "from_f8e5m2" for n in _ops_by_type(g, ElementwiseOp))


def test_spell_cone_keeps_the_promised_interface(tmp_path):
    """Interface invariance: whatever the cone spells inside, the graph output tensor is
    exactly what the trace promised — dtype, shape, and name unchanged."""
    for i, (scale_shape, dtype) in enumerate([((1, 1), "f32"), ((8, 1), "f16"), ((2, 4), "f32")]):
        d = tmp_path / str(i)
        d.mkdir()
        g, _bits, _scale = _spelled(d, scale_shape=scale_shape, dtype=dtype)
        out = g.nodes[g.outputs[0]].output
        assert out.dtype.name == dtype and tuple(d_.as_static() for d_ in out.shape) == (8, 16)
        assert g.outputs == ["p_w"]  # consumers keep addressing the original node id


def test_spell_bits_constant_carries_f8_graph_dtype(tmp_path):
    g, _bits, _scale = _spelled(tmp_path, scale_shape=(8, 1))
    w_node = next(n for n in g.nodes.values() if isinstance(n.op, ConstantOp) and n.op.source_path == "layer.weight")
    assert w_node.output.dtype.name == "f8e4m3"
    assert w_node.op.source_shape == (8, 16)


def test_spell_is_idempotent(tmp_path):
    """A second speller run must not re-spell the bits constant it created (its
    ``source_dtype`` is the storage token — the idempotency guard)."""
    g, _bits, _scale = _spelled(tmp_path, scale_shape=(8, 1))
    nodes_after_first = set(g.nodes)
    assert spell_quantized_constants(g, str(tmp_path)) == 0
    assert set(g.nodes) == nodes_after_first


def test_spell_dynamic_activations_reuses_one_shared_w8a8_value(tmp_path):
    bits = _finite_bits((8, 16))
    scale = np.ones((8, 1), dtype=np.float32)
    _write_checkpoint(
        tmp_path,
        {
            "layer.a.weight": _fp8_tensor(bits),
            "layer.a.weight_scale": torch.from_numpy(scale),
            "layer.b.weight": _fp8_tensor(bits),
            "layer.b.weight_scale": torch.from_numpy(scale),
        },
        _FP8_QC,
    )
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 16), "f16"), node_id="x")
    for name in ("a", "b"):
        weight = f"p_{name}"
        g.add_node(
            ConstantOp(name=weight, source_path=f"layer.{name}.weight", source_shape=(8, 16), source_dtype="f16"),
            [],
            Tensor(weight, (8, 16), "f16"),
            node_id=weight,
        )
        g.add_node(LinearOp(), ["x", weight], Tensor(f"y_{name}", (4, 8), "f16"), node_id=f"y_{name}")
    g.inputs, g.outputs = ["x"], ["y_a", "y_b"]

    assert spell_quantized_constants(g, str(tmp_path)) == 2
    assert spell_dynamic_fp8_activations(g, str(tmp_path)) == 2

    activation_inputs = {g.nodes[name].inputs[0] for name in ("y_a", "y_b")}
    assert len(activation_inputs) == 1
    assert len([node for node in _ops_by_type(g, ElementwiseOp) if node.op.op.name == "to_f8e4m3"]) == 1
    assert len([node for node in _ops_by_type(g, ReduceOp) if node.op.op.name == "maximum"]) == 1
    materialized = {node.output.dtype.name for node in g.nodes.values() if node.hints.get("trace.materialize")}
    assert materialized == {"f32", "f8e4m3"}
    assert spell_dynamic_fp8_activations(g, str(tmp_path)) == 0


def test_spell_dynamic_activations_requires_checkpoint_declaration(tmp_path):
    qc = {**_FP8_QC, "activation_scheme": "static"}
    g, _bits, _scale = _spelled(tmp_path, qc=qc)
    assert spell_dynamic_fp8_activations(g, str(tmp_path)) == 0


def test_spell_dynamic_activations_accepts_compressed_tensors_token_scheme(tmp_path):
    qc = {
        "quant_method": "compressed-tensors",
        "format": "float-quantized",
        "ignore": [],
        "config_groups": {
            "group_0": {
                "weights": {"num_bits": 8, "type": "float", "strategy": "channel", "dynamic": False},
                "input_activations": {"num_bits": 8, "type": "float", "strategy": "token", "dynamic": True},
            }
        },
    }
    bits = _finite_bits((8, 16))
    _write_checkpoint(
        tmp_path,
        {"layer.weight": _fp8_tensor(bits), "layer.weight_scale": torch.ones((8, 1), dtype=torch.float32)},
        qc,
    )
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 16), "f16"), node_id="x")
    g.add_node(
        ConstantOp(name="p_w", source_path="layer.weight", source_shape=(8, 16), source_dtype="f16"),
        [],
        Tensor("p_w", (8, 16), "f16"),
        node_id="p_w",
    )
    g.add_node(LinearOp(), ["x", "p_w"], Tensor("y", (4, 8), "f16"), node_id="y")
    g.inputs, g.outputs = ["x"], ["y"]

    assert spell_quantized_constants(g, str(tmp_path)) == 1
    assert spell_dynamic_fp8_activations(g, str(tmp_path)) == 1
    assert any(node.op.op.name == "to_f8e4m3" for node in _ops_by_type(g, ElementwiseOp))


def test_spell_dynamic_activations_rejects_mixed_compressed_tensor_groups(tmp_path):
    qc = {
        "quant_method": "compressed-tensors",
        "config_groups": {
            "dynamic": {
                "weights": {"num_bits": 8, "type": "float"},
                "input_activations": {"num_bits": 8, "type": "float", "strategy": "token", "dynamic": True},
            },
            "weight_only": {
                "weights": {"num_bits": 8, "type": "float"},
                "input_activations": None,
            },
        },
    }
    graph, _bits, _scale = _spelled(tmp_path, qc=qc)
    assert spell_dynamic_fp8_activations(graph, str(tmp_path)) == 0


def test_spell_honors_modules_to_not_convert(tmp_path):
    bits = _finite_bits((8, 16))
    qc = dict(_FP8_QC, modules_to_not_convert=["layer"])
    _write_checkpoint(tmp_path, {"layer.weight": _fp8_tensor(bits), "layer.weight_scale": torch.ones(1)}, qc)
    g = _weight_graph()
    assert spell_quantized_constants(g, str(tmp_path)) == 0
    assert set(g.nodes) == {"p_w"}  # constant left alone


def test_spell_skips_weight_without_scale(tmp_path):
    _write_checkpoint(tmp_path, {"layer.weight": _fp8_tensor(_finite_bits((8, 16)))}, _FP8_QC)
    g = _weight_graph()
    assert spell_quantized_constants(g, str(tmp_path)) == 0
    assert set(g.nodes) == {"p_w"}


def test_spell_skips_non_fp8_weight(tmp_path):
    # quantization_config present, but this weight is stored at full precision
    # (a modules-kept-in-bf16 member) — no rewrite even with a stray scale tensor.
    _write_checkpoint(tmp_path, {"layer.weight": torch.ones(8, 16), "layer.weight_scale": torch.ones(1)}, _FP8_QC)
    g = _weight_graph()
    assert spell_quantized_constants(g, str(tmp_path)) == 0
    assert set(g.nodes) == {"p_w"}


def test_spell_noop_on_unquantized_checkpoint(tmp_path):
    _write_checkpoint(tmp_path, {"layer.weight": torch.ones(8, 16)})
    g = _weight_graph()
    assert spell_quantized_constants(g, str(tmp_path)) == 0
    assert set(g.nodes) == {"p_w"}


def test_spell_non_dividing_scale_leaves_constant_alone(tmp_path):
    _write_checkpoint(tmp_path, {"layer.weight": _fp8_tensor(_finite_bits((8, 16))), "layer.weight_scale": torch.ones(3, 1)}, _FP8_QC)
    g = _weight_graph()
    assert spell_quantized_constants(g, str(tmp_path)) == 0
    assert set(g.nodes) == {"p_w"}


def test_spell_compressed_tensors_fp8_scheme(tmp_path):
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
    g.outputs = ["p_w", "p_h"]
    assert spell_quantized_constants(g, str(tmp_path)) == 1  # lm_head ignored
    consts = _constants(g)
    assert any(op.source_dtype == "f8e4m3" for op in consts.values())
    assert consts["p_h"].source_dtype == "f32" and "p_h" in g.nodes


def test_spell_compressed_tensors_int_scheme_is_noop(tmp_path):
    qc = {
        "quant_method": "compressed-tensors",
        "config_groups": {"group_0": {"weights": {"type": "int", "num_bits": 8}}},
    }
    _write_checkpoint(tmp_path, {"layer.weight": _fp8_tensor(_finite_bits((8, 16))), "layer.weight_scale": torch.ones(1)}, qc)
    assert spell_quantized_constants(_weight_graph(), str(tmp_path)) == 0


def test_spell_honors_regex_ignore(tmp_path):
    qc = {
        "quant_method": "compressed-tensors",
        "config_groups": {"group_0": {"weights": {"type": "float", "num_bits": 8}}},
        "ignore": ["re:.*lay.*"],
    }
    _write_checkpoint(tmp_path, {"layer.weight": _fp8_tensor(_finite_bits((8, 16))), "layer.weight_scale": torch.ones(1)}, qc)
    assert spell_quantized_constants(_weight_graph(), str(tmp_path)) == 0


def test_spell_pairs_non_weight_leaf_scale(tmp_path):
    """A 3-D expert parameter can pair with a sibling ``<key>_scale`` tensor even though the
    parameter name does not end in ``.weight``."""
    qc = {
        "quant_method": "compressed-tensors",
        "config_groups": {"group_0": {"weights": {"type": "float", "num_bits": 8, "strategy": "channel"}}},
        "ignore": ["re:.*router"],
    }
    key = "model.layers.0.mlp.experts.gate_up_proj"
    bits = _finite_bits((2, 4, 6))  # (E, in, out)
    scale = torch.rand(2, 1, 6, dtype=torch.float32) + 0.5  # per-out-channel, per expert
    _write_checkpoint(tmp_path, {key: _fp8_tensor(bits), key + "_scale": scale}, qc)
    g = _weight_graph(shape=(2, 4, 6), source_path=key)
    assert spell_quantized_constants(g, str(tmp_path)) == 1
    consts = _constants(g)
    w = next(op for op in consts.values() if op.source_path == key)
    assert w.source_dtype == "f8e4m3"
    s = next(op for op in consts.values() if op.source_path == key + "_scale")
    assert s.source_shape == (2, 1, 6)


# ===================================================================
# Loader: bind the expanded generic algebra and evaluate it against torch.
# ===================================================================


def _torch_ref(bits, scale_np, *, fmt="f8e4m3", inverse=False):
    """Reference dequant: torch float8 decode, then the same broadcast in plain numpy."""
    vals = torch.from_numpy(np.ascontiguousarray(bits)).view(_TORCH_F8[fmt]).float().numpy()
    s = np.asarray(scale_np, dtype=np.float32)
    if s.size == 1:
        s = s.reshape(())
    else:
        s = np.repeat(np.repeat(s, vals.shape[0] // s.shape[0], axis=0), vals.shape[1] // s.shape[1], axis=1)
    # ``inverse`` selects the ``weight_scale_inv`` key; the stored value is the multiplier either way.
    del inverse
    return vals * s


def _run_spelled(graph: Graph, model_dir: str) -> np.ndarray:
    from emmy.compiler.backend.numpy import NumpyBackend

    data = load_constants_from_safetensors(graph, model_dir)
    result, _ = NumpyBackend().run(graph, input_data=data)
    return result.outputs[graph.outputs[0]]


@pytest.mark.parametrize(
    ("scale_shape", "inverse"),
    [((), False), ((8, 1), False), ((2, 4), False), ((8, 1), True)],
    ids=["per-tensor", "per-channel", "block", "inverse"],
)
def test_loader_binds_expanded_weight_bit_identical_to_torch(tmp_path, scale_shape, inverse):
    g, bits, scale = _spelled(tmp_path, scale_shape=scale_shape, inverse=inverse)
    assert _f8_constants(g)
    assert all(op.source_graph is None for _nid, op in g.loadable_constants())
    out = _run_spelled(g, str(tmp_path))
    np.testing.assert_array_equal(out, _torch_ref(bits, scale, inverse=inverse))
    assert out.dtype == np.float32


def test_loader_casts_to_traced_compute_dtype(tmp_path):
    g, bits, scale = _spelled(tmp_path, scale_shape=(1, 1), dtype="f16")
    out = _run_spelled(g, str(tmp_path))
    assert out.dtype == np.float16
    np.testing.assert_array_equal(out, _torch_ref(bits, scale).astype(np.float16))


@pytest.mark.parametrize("fmt", ["f8e4m3", "f8e5m2"])
def test_loader_reads_fp8_at_value_dtype_as_plain_values(tmp_path, fmt):
    """An fp8-stored tensor bound at a NON-f8 graph dtype decodes to values (no scale) —
    the decode is dtype-directed. This graph traces the weight at f32 with no
    quantization_config, so nothing is spelled and the tensor reads as values."""
    bits = _finite_bits((8, 16))
    _write_checkpoint(tmp_path, {"layer.weight": _fp8_tensor(bits, fmt)})
    out = load_constants_from_safetensors(_weight_graph(), str(tmp_path))
    ref = torch.from_numpy(bits).view(_TORCH_F8[fmt]).float().numpy()
    np.testing.assert_array_equal(out["p_w"], ref)


@pytest.mark.parametrize("fmt", ["f8e4m3", "f8e5m2"])
def test_loader_binds_f8_dtype_constant_as_raw_bits(tmp_path, fmt):
    """A constant whose GRAPH dtype is an f8 dtype binds the raw uint8 bit pattern — no LUT
    decode, no scale. The graph's own algebra owns the value semantics, so handing over
    decoded values would double-decode."""
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
    assert spell_quantized_constants(g, str(tmp_path)) == 0
    np.testing.assert_array_equal(load_constants_from_safetensors(g, str(tmp_path))["p_w"], w)


# ===================================================================
# AWQ GEMM int4: packed checkpoint algebra and eager-reference decode
# ===================================================================


def _awq_fixture(tmp_path, *, k=8, n=16, group_size=4):
    integers = (np.arange(k * n, dtype=np.int32).reshape(k, n) * 7 + 3) % 16
    zeros = (np.arange((k // group_size) * n, dtype=np.int32).reshape(k // group_size, n) * 5 + 1) % 16
    scales = (np.arange((k // group_size) * n, dtype=np.float32).reshape(k // group_size, n) + 1) / 128
    qc = dict(_AWQ_QC, group_size=group_size)
    _write_checkpoint(
        tmp_path,
        {
            "layer.qweight": torch.from_numpy(_pack_awq4(integers)),
            "layer.qzeros": torch.from_numpy(_pack_awq4(zeros)),
            "layer.scales": torch.from_numpy(scales).half(),
        },
        qc,
    )
    stored_scales = scales.astype(np.float16).astype(np.float32)
    ref = (integers - np.repeat(zeros, group_size, axis=0)) * np.repeat(stored_scales, group_size, axis=0)
    return integers, zeros, scales, ref


def test_unpack_awq4_reverses_gemm_pack_order():
    values = np.arange(32, dtype=np.int32).reshape(2, 16) % 16
    np.testing.assert_array_equal(unpack_awq4(_pack_awq4(values)), values)


def test_dequantize_awq4_matches_group_formula(tmp_path):
    integers, zeros, scales, ref = _awq_fixture(tmp_path)
    got = dequantize_awq4(_pack_awq4(integers), _pack_awq4(zeros), scales.astype(np.float16), 4)
    np.testing.assert_array_equal(got, ref)


def test_load_dequantized_state_dict_awq_emits_hf_weight(tmp_path):
    from emmy.compiler.loader.quant import load_dequantized_state_dict

    _integers, _zeros, _scales, ref = _awq_fixture(tmp_path)
    state = load_dequantized_state_dict(tmp_path)
    assert set(state) == {"layer.weight"}
    np.testing.assert_array_equal(state["layer.weight"], ref.T)


def test_spell_awq4_constants_preserves_packed_sources_and_values(tmp_path):
    _integers, _zeros, _scales, ref = _awq_fixture(tmp_path)
    graph = _weight_graph(shape=(16, 8), dtype="f32")
    assert spell_quantized_constants(graph, str(tmp_path)) == 1
    graph.validate()
    sources = {op.source_path for op in _constants(graph).values() if op.source_path is not None}
    assert sources == {"layer.qweight", "layer.qzeros", "layer.scales"}
    assert any(n.op.op.name == "right_shift" for n in _ops_by_type(graph, ElementwiseOp))
    assert any(n.op.op.name == "bitwise_and" for n in _ops_by_type(graph, ElementwiseOp))

    from emmy.compiler.backend.numpy.backend import NumpyBackend

    backend = NumpyBackend()
    compiled = backend.compile(graph)
    result, _ = backend.run(compiled, input_data=load_constants_from_safetensors(compiled, str(tmp_path)))
    np.testing.assert_array_equal(result.outputs[compiled.outputs[0]], ref.T)


def test_quantized_checkpoint_dir_detects_awq(tmp_path):
    from emmy.compiler.trace.huggingface import quantized_checkpoint_dir

    _awq_fixture(tmp_path)
    assert quantized_checkpoint_dir(str(tmp_path)) == tmp_path


# ===================================================================
# Input-sourced fp8: spell_quantized_inputs (the MoE serving seam's expert
# programs — weights are forward-argument InputOps, not constants)
# ===================================================================


def _input_graph(w_shape=(8, 16), dtype="f32"):
    """``y = x * w`` over two graph inputs — the smallest consumer of a weight input."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", w_shape, dtype), node_id="x")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("w", w_shape, dtype), node_id="w")
    g.add_node(op=ElementwiseOp(op="multiply"), inputs=["x", "w"], output=Tensor("y", w_shape, dtype), node_id="y")
    g.inputs, g.outputs = ["x", "w"], ["y"]
    return g


def test_spell_inputs_rewrites_input_to_bits_plus_scale(tmp_path):
    g = _input_graph()
    m = spell_quantized_inputs(g, {"w": ("f8e4m3", (1, 16), "bf16")})
    assert m == {"w": "w_scale"}
    g.validate()
    # The bits input keeps its id and its graph.inputs slot; the scale input is appended.
    assert g.inputs == ["x", "w", "w_scale"]
    assert g.nodes["w"].output.dtype.name == "f8e4m3"
    assert isinstance(g.nodes["w"].op, InputOp) and isinstance(g.nodes["w_scale"].op, InputOp)
    # bf16-stored scale declares f32 graph values (the loader convention).
    assert g.nodes["w_scale"].output.dtype.name == "f32"
    assert tuple(d.as_static() for d in g.nodes["w_scale"].output.shape) == (1, 16)
    # The decode cast is the bits input's only consumer; the original consumer reads the cone.
    casts = [n for n in _ops_by_type(g, ElementwiseOp) if n.op.op.name == "from_f8e4m3"]
    assert len(casts) == 1
    assert g.users("w") == {casts[0].id}
    assert "w" not in g.nodes["y"].inputs


def _run_numpy(g, input_data):
    from emmy.compiler.backend.numpy.backend import NumpyBackend

    backend = NumpyBackend()
    compiled = backend.compile(g)
    res, _ = backend.run(compiled, input_data=input_data)
    return res.outputs[compiled.outputs[0]]


def test_spell_inputs_values_per_channel():
    g = _input_graph()
    bits = _finite_bits((8, 16))
    scale = (np.abs(rng.standard_normal((1, 16))) + 0.5).astype(np.float32)
    x = rng.standard_normal((8, 16)).astype(np.float32)
    spell_quantized_inputs(g, {"w": ("f8e4m3", (1, 16), "f32")})
    out = _run_numpy(g, {"x": x, "w": bits, "w_scale": scale})
    np.testing.assert_allclose(out, x * (decode_f8(bits, "f8e4m3") * scale), rtol=1e-6)


def test_spell_inputs_values_2d_block():
    """A genuine 2-D block scale spells the interleaved reshape pair; the scale input binds at
    the interleaved ``(grid, 1)`` layout, so the feed reshapes the stored ``(2, 4)`` to it."""
    g = _input_graph()
    bits = _finite_bits((8, 16))
    scale = (np.abs(rng.standard_normal((2, 4))) + 0.5).astype(np.float32)  # derived block (4, 4)
    x = rng.standard_normal((8, 16)).astype(np.float32)
    spell_quantized_inputs(g, {"w": ("f8e4m3", (2, 4), "f32")})
    assert tuple(d.as_static() for d in g.nodes["w_scale"].output.shape) == (2, 1, 4, 1)
    out = _run_numpy(g, {"x": x, "w": bits, "w_scale": scale.reshape(2, 1, 4, 1)})
    ref = x * (decode_f8(bits, "f8e4m3") * np.repeat(np.repeat(scale, 4, axis=0), 4, axis=1))
    np.testing.assert_allclose(out, ref, rtol=1e-6)


def test_spell_inputs_inverse_divides():
    g = _input_graph()
    bits = _finite_bits((8, 16))
    scale = np.full((1, 16), 2.0, dtype=np.float32)
    x = rng.standard_normal((8, 16)).astype(np.float32)
    spell_quantized_inputs(g, {"w": ("f8e4m3", (1, 16), "f32")}, inverse=True)
    out = _run_numpy(g, {"x": x, "w": bits, "w_scale": scale})
    np.testing.assert_allclose(out, x * (decode_f8(bits, "f8e4m3") / scale), rtol=1e-6)


def test_spell_inputs_rejects_bad_specs():
    with pytest.raises(ValueError, match="not a graph input"):
        spell_quantized_inputs(_input_graph(), {"y": ("f8e4m3", (1, 16), "f32")})
    with pytest.raises(ValueError, match="does not tile"):
        spell_quantized_inputs(_input_graph(), {"w": ("f8e4m3", (3, 1), "f32")})
    with pytest.raises(ValueError, match="unknown fp8 storage format"):
        spell_quantized_inputs(_input_graph(), {"w": ("int4", (1, 16), "f32")})
    g = _input_graph()
    spell_quantized_inputs(g, {"w": ("f8e4m3", (1, 16), "f32")})
    with pytest.raises(ValueError, match="already carries the f8 storage dtype"):
        spell_quantized_inputs(g, {"w": ("f8e4m3", (1, 16), "f32")})


# ===================================================================
# Dequantized state dict + quantized-checkpoint detection (the eager twin)
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


def test_load_dequantized_state_dict_expert_param_pairing(tmp_path):
    """A 3-D expert parameter and per-output-channel scale dequantize together, consume the
    scale entry, and leave the sibling bias as ordinary values."""
    from emmy.compiler.loader.quant import load_dequantized_state_dict

    qc = {
        "quant_method": "compressed-tensors",
        "config_groups": {"group_0": {"weights": {"type": "float", "num_bits": 8, "strategy": "channel"}}},
        "ignore": ["re:.*router"],
    }
    key = "model.layers.0.mlp.experts.gate_up_proj"
    bits = _finite_bits((2, 4, 6))
    scale_np = (rng.random((2, 1, 6)).astype(np.float32) + 0.5).astype(np.float32)
    tensors = {
        key: _fp8_tensor(bits),
        key + "_scale": torch.from_numpy(scale_np).to(torch.bfloat16),
        key + "_bias": torch.full((2, 6), 3.0, dtype=torch.bfloat16),
        "model.layers.0.mlp.router.weight": _fp8_tensor(_finite_bits((4, 4))),  # ignored: loads as plain values
    }
    _write_checkpoint(tmp_path, tensors, qc)
    sd = load_dequantized_state_dict(tmp_path)
    scale_f32 = torch.from_numpy(scale_np).to(torch.bfloat16).float().numpy()  # the bf16-stored values
    ref = torch.from_numpy(bits).view(_TORCH_F8["f8e4m3"]).float().numpy() * scale_f32
    np.testing.assert_array_equal(sd[key], ref)
    assert key + "_scale" not in sd
    np.testing.assert_array_equal(sd[key + "_bias"], np.full((2, 6), 3.0, dtype=np.float32))
    assert sd["model.layers.0.mlp.router.weight"].shape == (4, 4)  # ignored module: loads as plain values


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


def _rung(dirpath, bits, allocation=None):
    """A checkpoint declaring an EXL3 rung: ``config.json`` plus, optionally, the allocation
    sidecar. Architecture fields are identical across rungs — only the rate differs."""
    dirpath.mkdir()
    (dirpath / "config.json").write_text(
        json.dumps(
            {"model_type": "llama", "hidden_size": 64, "quantization_config": {"quant_method": "exl3", "bits": bits, "head_bits": 6}}
        )
    )
    if allocation is not None:
        (dirpath / "quantization_config.json").write_text(json.dumps({"tensor_storage": allocation}))
    return dirpath


def test_checkpoint_quant_digest_separates_rungs_a_config_hash_cannot(tmp_path):
    """The serving pack key hashes the twin's config, and the twin is built with
    ``quantization_config`` STRIPPED — so two EXL3 rungs of one repo hash identically and shared a
    pack, whose stored plans carry the other rung's coded extents. The digest is what separates
    them, and it prefers the per-module allocation over the declared average rate."""
    from emmy.compiler.loader.quant import checkpoint_quant_digest

    a = _rung(tmp_path / "r200", 2.0)
    b = _rung(tmp_path / "r225", 2.26)
    # Precondition: strip quantization_config (what ``load_quantized_split`` does) and the two
    # configs are byte-identical — the aliasing this key entry exists to break.
    stripped = [{k: v for k, v in json.loads((d / "config.json").read_text()).items() if k != "quantization_config"} for d in (a, b)]
    assert stripped[0] == stripped[1]
    assert checkpoint_quant_digest(a) != checkpoint_quant_digest(b)

    # The sidecar wins when present: same declared rate, different per-module allocation.
    c = _rung(tmp_path / "s1", 2.26, {"model.layers.0.self_attn.q_proj": {"bits_per_weight": 4.0}})
    d = _rung(tmp_path / "s2", 2.26, {"model.layers.0.self_attn.q_proj": {"bits_per_weight": 3.0}})
    assert checkpoint_quant_digest(c) != checkpoint_quant_digest(d)

    # An unquantized checkpoint contributes nothing, so its pack key is unchanged.
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "config.json").write_text(json.dumps({"model_type": "llama", "hidden_size": 64}))
    assert checkpoint_quant_digest(plain) is None
    assert checkpoint_quant_digest(tmp_path / "absent") is None

    # The boot log's summary carries the rate, so which rung a running server opened is readable
    # from the log alone — the observable both revision controls are checked against.
    from emmy.compiler.loader.quant import checkpoint_quant_summary

    assert checkpoint_quant_summary(a) == "exl3 2.0 bpw (head 6)"
    assert checkpoint_quant_summary(b) == "exl3 2.26 bpw (head 6)"
    assert checkpoint_quant_summary(plain) == "unquantized"


def _config_dir(tmp_path, name, qc):
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "test", "quantization_config": qc}))
    return d


def test_fp4_quant_config_recognizes_both_dialects(tmp_path):
    for name, qc in (("modelopt", _FP4_MODELOPT_QC), ("ct", _FP4_CT_QC)):
        assert _fp4_quant_config(_config_dir(tmp_path, name, qc)) == qc


def test_fp4_quant_config_rejects_other_schemes(tmp_path):
    mxfp4 = {
        "quant_method": "compressed-tensors",
        "config_groups": {"group_0": {"weights": {"type": "float", "num_bits": 4, "group_size": 32}}},
    }
    modelopt_fp8 = {"quant_method": "modelopt", "quant_algo": "FP8"}
    for name, qc in (("mxfp4", mxfp4), ("modelopt-fp8", modelopt_fp8), ("fp8", _FP8_QC)):
        assert _fp4_quant_config(_config_dir(tmp_path, name, qc)) is None
    assert _fp4_quant_config(tmp_path / "absent") is None


def test_fp4_and_fp8_recognizers_do_not_cross_match(tmp_path):
    """A family system fails at the seams: each recognizer must decline the other's PURE
    checkpoints — one whose every quantized leaf is the other scheme. That is the whole contract;
    a checkpoint that really does hold both is the MIXED case below, where both must answer."""
    fp4_dir = _config_dir(tmp_path, "fp4", _FP4_CT_QC)
    fp8_dir = _config_dir(tmp_path, "fp8", _FP8_QC)
    assert _fp8_quant_config(fp4_dir) is None
    assert _fp4_quant_config(fp8_dir) is None
    assert _fp4_quant_config(_config_dir(tmp_path, "modelopt-nvfp4", _FP4_MODELOPT_QC)) is not None
    assert _fp8_quant_config(_config_dir(tmp_path, "modelopt-nvfp4-b", _FP4_MODELOPT_QC)) is None


# ===================================================================
# MIXED_PRECISION: one checkpoint, two schemes, sorted per leaf
# ===================================================================

# modelopt's form for a checkpoint whose schemes differ by leaf (the nvidia/Qwen3.6-27B-NVFP4
# shape): the algo names no single scheme, and one config group per scheme carries the widths
# instead. The ignore entry here exercises the mechanism; the real 27B ignores mtp*, not lm_head
# (its lm_head is NVFP4).
_MIXED_QC = {
    "quant_method": "modelopt",
    "quant_algo": "MIXED_PRECISION",
    "ignore": ["lm_head"],
    "config_groups": {
        "group_0": {"weights": {"type": "float", "num_bits": 8}, "input_activations": {"type": "float", "num_bits": 8}, "targets": ["a"]},
        "group_1": {"weights": {"type": "float", "num_bits": 4, "group_size": 16}, "targets": ["b"]},
    },
}


def _mixed_checkpoint(dirpath):
    """One checkpoint holding a leaf of each scheme plus an unquantized one — the three sibling
    signatures the per-leaf sort has to tell apart. Returns the oracle values."""
    dirpath.mkdir(exist_ok=True)
    f8_bits = _finite_bits((8, 16))
    f8_scale = np.float32(0.25).reshape(())
    packed = rng.integers(0, 256, (8, 16)).astype(np.uint8)  # 8 x 32 logical
    blk = rng.integers(0, 0x7F, (8, 2)).astype(np.uint8)
    s2 = np.array(0.5, dtype=np.float32)
    plain = rng.standard_normal((8, 16)).astype(np.float16)
    _write_checkpoint(
        dirpath,
        {
            "f8.weight": _fp8_tensor(f8_bits),
            "f8.weight_scale": torch.from_numpy(np.asarray(f8_scale)),
            "f4.weight": torch.from_numpy(packed),
            "f4.weight_scale": _fp8_tensor(blk),
            "f4.weight_scale_2": torch.tensor(float(s2), dtype=torch.float32),
            "plain.weight": torch.from_numpy(plain),
        },
        quant_config=_MIXED_QC,
    )
    return {
        "f8": decode_f8(f8_bits, "f8e4m3").astype(np.float32) * f8_scale,
        "f4": dequantize_nvfp4(packed, blk, s2),
        "plain": plain,
    }


def test_mixed_precision_answers_both_recognizers(tmp_path):
    """The declaration names no single scheme, so each recognizer reads the config groups for its
    own width — and both are present, which is what MIXED means."""
    d = _config_dir(tmp_path, "mixed", _MIXED_QC)
    assert _fp8_quant_config(d) is not None
    assert _fp4_quant_config(d) is not None


def test_mixed_precision_summary_names_both_schemes(tmp_path):
    """A boot log naming one scheme would misreport half the model."""
    from emmy.compiler.loader.quant import checkpoint_quant_summary

    d = _config_dir(tmp_path, "mixed", _MIXED_QC)
    assert checkpoint_quant_summary(d) == "mixed fp8+nvfp4 modelopt"


def test_mixed_precision_declines_when_only_one_scheme_is_declared(tmp_path):
    """The arm reads the GROUPS, not the algo string: a MIXED declaration carrying only 8-bit
    groups is an fp8 checkpoint and must not pull in the NVFP4 speller."""
    only8 = {**_MIXED_QC, "config_groups": {"g": {"weights": {"type": "float", "num_bits": 8}}}}
    only4 = {**_MIXED_QC, "config_groups": {"g": {"weights": {"type": "float", "num_bits": 4, "group_size": 16}}}}
    mx = {**_MIXED_QC, "config_groups": {"g": {"weights": {"type": "float", "num_bits": 4, "group_size": 32}}}}
    assert _fp4_quant_config(_config_dir(tmp_path, "only8", only8)) is None
    assert _fp8_quant_config(_config_dir(tmp_path, "only8b", only8)) is not None
    assert _fp8_quant_config(_config_dir(tmp_path, "only4", only4)) is None
    assert _fp4_quant_config(_config_dir(tmp_path, "only4b", only4)) is not None
    assert _fp4_quant_config(_config_dir(tmp_path, "mx", mx)) is None, "32-element blocks are MXFP4, not NVFP4"


def test_mixed_precision_sorts_leaves_by_their_stored_siblings(tmp_path):
    """The load-bearing claim: widening the recognizers is ENOUGH, because which speller takes a
    leaf was never a config question. The fp8 pair, the NVFP4 trio and the bare tensor sort
    themselves, and each decodes to its own oracle."""
    from emmy.compiler.loader.quant import load_dequantized_state_dict

    d = tmp_path / "mixed"
    want = _mixed_checkpoint(d)
    sd = load_dequantized_state_dict(d)
    np.testing.assert_allclose(sd["f8.weight"], want["f8"], rtol=1e-6)
    np.testing.assert_array_equal(sd["f4.weight"], want["f4"])
    np.testing.assert_array_equal(sd["plain.weight"], want["plain"])
    for gone in ("f8.weight_scale", "f4.weight_scale", "f4.weight_scale_2"):
        assert gone not in sd, f"{gone} survived as its own entry"


def test_mixed_precision_spells_each_leaf_into_its_own_cone(tmp_path):
    """The graph side of the same sort: the fp8 leaf spells the fp8 decode cone, the packed leaf
    the NVFP4 one, and the plain leaf is left alone — one pass over one checkpoint."""
    d = tmp_path / "mixed"
    want = _mixed_checkpoint(d)
    g = Graph()
    leaves = (("w8", "f8.weight", (8, 16), "f32"), ("w4", "f4.weight", (8, 32), "f32"), ("wp", "plain.weight", (8, 16), "f16"))
    for name, path, shape, dt in leaves:
        op = ConstantOp(name=name, source_path=path, source_shape=shape, source_dtype=dt)
        g.add_node(op=op, inputs=[], output=Tensor(name, shape, dt), node_id=name)
    g.inputs, g.outputs = [], ["w8", "w4", "wp"]
    assert spell_quantized_constants(g, str(d)) == 2, "both quantized leaves spell; the plain one does not"

    from emmy.compiler.backend.numpy import NumpyBackend

    data = load_constants_from_safetensors(g, str(d))
    result, _ = NumpyBackend().run(g, input_data=data)
    np.testing.assert_allclose(result.outputs["w8"], want["f8"], rtol=1e-6)
    np.testing.assert_array_equal(result.outputs["w4"], want["f4"])
    np.testing.assert_allclose(result.outputs["wp"], want["plain"].astype(np.float16))


def _nvfp4_checkpoint(dirpath, *, prefix="layer", ignore=("lm_head",)):
    """Single NVFP4-quantized linear (K=32) + config; returns (packed, scale_bits, s2)."""
    packed = rng.integers(0, 256, (8, 16)).astype(np.uint8)
    scale_bits = rng.integers(0, 0x7F, (8, 2)).astype(np.uint8)  # finite e4m3 codes
    dirpath.mkdir(exist_ok=True)
    _write_checkpoint(
        dirpath,
        {
            f"{prefix}.weight": torch.from_numpy(packed),
            f"{prefix}.weight_scale": _fp8_tensor(scale_bits),
            f"{prefix}.weight_scale_2": torch.tensor(0.25, dtype=torch.float32),
        },
        quant_config={**_FP4_MODELOPT_QC, "ignore": list(ignore)},
    )
    return packed, scale_bits, np.array(0.25, dtype=np.float32)


def test_spell_nvfp4_matches_oracle(tmp_path):
    packed, scale_bits, s2 = _nvfp4_checkpoint(tmp_path)
    g = _weight_graph(shape=(8, 32), dtype="f32", source_path="layer.weight")
    assert spell_quantized_constants(g, str(tmp_path)) == 1
    np.testing.assert_array_equal(_run_spelled(g, str(tmp_path)), dequantize_nvfp4(packed, scale_bits, s2))


def test_spell_nvfp4_idempotent(tmp_path):
    _nvfp4_checkpoint(tmp_path)
    g = _weight_graph(shape=(8, 32), dtype="f32", source_path="layer.weight")
    assert spell_quantized_constants(g, str(tmp_path)) == 1
    assert spell_quantized_constants(g, str(tmp_path)) == 0


def test_spell_nvfp4_respects_ignore(tmp_path):
    _nvfp4_checkpoint(tmp_path, prefix="lm_head")
    g = _weight_graph(shape=(8, 32), dtype="f32", source_path="lm_head.weight")
    assert spell_quantized_constants(g, str(tmp_path)) == 0


def test_spell_nvfp4_casts_only_where_the_dtype_changes(tmp_path):
    """Every ``copy`` the cone spells must change its dtype. The cone's two cast arms (the
    pair table's cast off f32, the fused scale's cast off f16) fire only when the promised
    dtype differs; an arm firing without a dtype change is a redundant identity node the
    packed-B reading then has to look through. (The arms once compared a ``DataType``
    against a string — never equal — so both always fired.)"""

    def all_graphs(root):
        yield root
        for _nid, op in root.loadable_constants():
            if op.source_graph is not None:
                yield from all_graphs(op.source_graph)

    for dtype in ("f32", "f16"):
        d = tmp_path / dtype
        d.mkdir()
        _nvfp4_checkpoint(d)
        g = _weight_graph(shape=(8, 32), dtype=dtype, source_path="layer.weight")
        assert spell_quantized_constants(g, str(d)) == 1
        for sub in all_graphs(g):
            for node in sub.nodes.values():
                if isinstance(node.op, ElementwiseOp) and node.op.name == "copy":
                    (src,) = node.inputs
                    if node.output.dtype == sub.nodes[src].output.dtype:
                        raise AssertionError(f"identity copy {node.output.name!r} in the {dtype} cone")


def _nvfp4_matmul_graph(tmp_path):
    """x [4, 32] @ dequant(w).T — the spelled cone feeding a matmul consumer."""
    from emmy.compiler.ir.base import InputOp
    from emmy.compiler.ir.frontend.ir import MatmulOp, TransposeOp

    packed, scale_bits, s2 = _nvfp4_checkpoint(tmp_path)
    g = Graph()
    x_in = g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4, 32), "f32"), node_id="x")
    w = g.add_node(
        op=ConstantOp(name="w", source_path="layer.weight", source_shape=(8, 32), source_dtype="f32"),
        inputs=[],
        output=Tensor("w", (8, 32), "f32"),
        node_id="w",
    )
    wt = g.add_node(op=TransposeOp(axes=(1, 0)), inputs=[w], output=Tensor("wt", (32, 8), "f32"))
    y = g.add_node(op=MatmulOp(), inputs=[x_in, wt], output=Tensor("y", (4, 8), "f32"))
    g.outputs = [y]
    g.inputs = [x_in]
    assert spell_quantized_constants(g, str(tmp_path)) == 1
    return g, y, (packed, scale_bits, s2)


# ===================================================================
# Static 4-bit activations — the W4A4 declared program's activation half
# ===================================================================

#: The 8B's own declaration shape: config_groups marking input_activations 4-bit static float.
_W4A4_MODELOPT_QC = {
    **_FP4_MODELOPT_QC,
    "config_groups": {
        "group_0": {
            "input_activations": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
            "weights": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
            "targets": ["Linear"],
        }
    },
}


def _w4a4_checkpoint(dirpath, modules, *, k=32, qc=_W4A4_MODELOPT_QC):
    """NVFP4 weight trio + static ``input_scale`` per module — the 8B's marking at toy shape.

    ``modules`` maps a module name to ``(n, input_scale)``; ``input_scale=None`` writes no
    activation marker for that module. Returns each module's ``(packed, scale_bits)``."""
    tensors, data = {}, {}
    for name, (n, iscale) in modules.items():
        packed = rng.integers(0, 256, (n, k // 2)).astype(np.uint8)
        scale_bits = rng.integers(0, 0x7F, (n, k // 16)).astype(np.uint8)
        tensors[f"{name}.weight"] = torch.from_numpy(packed)
        tensors[f"{name}.weight_scale"] = _fp8_tensor(scale_bits)
        tensors[f"{name}.weight_scale_2"] = torch.tensor(0.25, dtype=torch.float32)
        if iscale is not None:
            tensors[f"{name}.input_scale"] = torch.tensor(iscale, dtype=torch.float32)
        data[name] = (packed, scale_bits)
    dirpath.mkdir(exist_ok=True)
    _write_checkpoint(dirpath, tensors, quant_config=qc)
    return data


def _w4a4_graph(modules, *, k=32, dtype="f32"):
    """One shared activation feeding a ``LinearOp`` per module — q/k/v's shape at toy size."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4, k), dtype), node_id="x")
    outs = []
    for name, n in modules.items():
        w = g.add_node(
            op=ConstantOp(name=name, source_path=f"{name}.weight", source_shape=(n, k), source_dtype=dtype),
            inputs=[],
            output=Tensor(f"{name}_w", (n, k), dtype),
            node_id=f"{name}_w",
        )
        outs.append(g.add_node(op=LinearOp(), inputs=["x", w], output=Tensor(f"{name}_y", (4, n), dtype), node_id=f"{name}_y"))
    g.inputs, g.outputs = ["x"], outs
    return g


def _static_quantize_ref(x, s2):
    """The spelled quantize's numpy twin: :func:`quantize_nvfp4` with the static ``s2``, the
    divisor floored the way the graph floors it (a zero fused scale divides by the floor; the
    decode multiplies those blocks back to zero either way)."""
    blocks = x.astype(np.float32).reshape(*x.shape[:-1], -1, 16)
    scale_bits = encode_f8(np.abs(blocks).max(axis=-1) / 6.0 / s2, "f8e4m3")
    fused = fuse_nvfp4_scales(scale_bits, np.float32(s2).reshape(1)).astype(np.float32)
    quot = blocks / np.maximum(fused[..., None], 1e-12)
    return dequantize_nvfp4(encode_f4x2(quot.reshape(x.shape)), scale_bits, np.float32(s2).reshape(1))


def _count_encodes(g):
    from emmy.compiler.ir.tensor.ir import ElementwiseOp as EW

    return sum(1 for n in g.nodes.values() if isinstance(n.op, EW) and n.op.name == "to_f4e2m1")


def test_spell_static_fp4_matches_the_quantize_oracle(tmp_path):
    """numpy on the spelled W4A4 graph equals the declared round trip built from the repo's own
    encode primitives: x̂ @ ŵᵀ with x̂ the static-scale quantize of x. A zeroed block rides along
    to pin the zero-guard (its codes multiply back to zero, never NaN)."""
    from emmy.compiler.backend.numpy import NumpyBackend

    s2 = 0.03
    (packed, scale_bits) = _w4a4_checkpoint(tmp_path, {"layer": (8, s2)})["layer"]
    g = _w4a4_graph({"layer": 8})
    assert spell_quantized_constants(g, str(tmp_path)) == 1
    assert spell_static_fp4_activations(g, str(tmp_path)) == 1

    x = (rng.standard_normal((4, 32)) * 0.5).astype(np.float32)
    x[1, :16] = 0.0
    data = load_constants_from_safetensors(g, str(tmp_path))
    result, _ = NumpyBackend().run(g, input_data={**data, "x": x})
    got = result.outputs["layer_y"]
    ref = _static_quantize_ref(x, s2) @ dequantize_nvfp4(packed, scale_bits, np.array(0.25, dtype=np.float32)).T
    assert not np.any(np.isnan(got))
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-6)


def test_spell_static_fp4_shares_equal_scales_and_splits_unequal(tmp_path):
    """q/k share one calibrated scale value and must share ONE quantize chain; v carries its own
    value and quantizes separately — two encodes for three rewired linears."""
    _w4a4_checkpoint(tmp_path, {"q": (8, 0.03), "k": (8, 0.03), "v": (8, 0.07)})
    g = _w4a4_graph({"q": 8, "k": 8, "v": 8})
    assert spell_quantized_constants(g, str(tmp_path)) == 3
    assert spell_static_fp4_activations(g, str(tmp_path)) == 3
    assert _count_encodes(g) == 2


def test_spell_static_fp4_requires_the_per_linear_marker(tmp_path):
    """A module without its ``input_scale`` keeps its 16-bit activation even when the config
    declares the scheme — the per-linear tensor is the marker, and eligibility is per linear."""
    _w4a4_checkpoint(tmp_path, {"a": (8, 0.03), "b": (8, None)})
    g = _w4a4_graph({"a": 8, "b": 8})
    assert spell_quantized_constants(g, str(tmp_path)) == 2
    assert spell_static_fp4_activations(g, str(tmp_path)) == 1
    assert _count_encodes(g) == 1


def test_spell_static_fp4_is_idempotent(tmp_path):
    _w4a4_checkpoint(tmp_path, {"layer": (8, 0.03)})
    g = _w4a4_graph({"layer": 8})
    spell_quantized_constants(g, str(tmp_path))
    assert spell_static_fp4_activations(g, str(tmp_path)) == 1
    assert spell_static_fp4_activations(g, str(tmp_path)) == 0


def test_static_fp4_declaration_reader():
    """The config-level reader: 4-bit static input_activations declare; dynamic or absent
    activation groups decline; a bare modelopt NVFP4 config (no ``config_groups``) defers to the
    per-linear marker and declares."""
    assert _static_fp4_activation_declared(_W4A4_MODELOPT_QC)
    assert _static_fp4_activation_declared(_FP4_MODELOPT_QC), "bare modelopt: input_scale siblings are the marker"
    assert not _static_fp4_activation_declared(_FP4_CT_QC), "weights-only groups do not mark activations"
    dyn = {**_W4A4_MODELOPT_QC, "config_groups": {"g": {"input_activations": {"dynamic": True, "num_bits": 4, "type": "float"}}}}
    assert not _static_fp4_activation_declared(dyn), "a dynamic declaration is not the static path"


def test_both_loaders_bind_an_f8_bits_constant_identically(tmp_path):
    """The DOUBLE-DECODE guard: the graph-keyed and plan-keyed loaders must agree bit-for-bit.

    An NVFP4 block scale is stored e4m3 and consumed as an ``f8e4m3`` graph constant, which its own
    ``from_f8e4m3`` cone decodes. The graph-keyed loader keys that raw-bits rule on the node's
    dtype. The plan-keyed loader reaches the checkpoint without a graph, so its caller supplies the
    same rule from ``WeightSpec.graph_dtype``. Before that field existed the plan-keyed read
    decoded the scales to f32, the cone decoded them AGAIN, and serving computed silently wrong
    numbers while every kernel compiled and ran.

    The third assertion is the point: without the flag the two loaders genuinely disagree, so the
    flag is what makes them agree rather than the checkpoint happening to be read the same way.
    """
    from emmy.compiler.loader.quant import F8_SAFETENSORS_DTYPES
    from emmy.compiler.loader.safetensors import load_constants_from_safetensors, load_sources_by_path

    g, _y, _ = _nvfp4_matmul_graph(tmp_path)
    f8 = set(F8_SAFETENSORS_DTYPES.values())
    bits = {
        nid: op.source_path for nid, op in g.loadable_constants() if op.source_path is not None and g.nodes[nid].output.dtype.name in f8
    }
    assert bits, "the spelled NVFP4 graph must carry an f8-dtype constant for this to test anything"

    by_graph = load_constants_from_safetensors(g, str(tmp_path))
    by_path = load_sources_by_path(str(tmp_path), set(bits.values()), bits_paths=frozenset(bits.values()))
    decoded = load_sources_by_path(str(tmp_path), set(bits.values()))

    for nid, path in bits.items():
        want, got = np.asarray(by_graph[nid]), np.asarray(by_path[path])
        assert want.dtype == np.uint8, f"{nid}: an f8-dtype constant binds RAW BITS, got {want.dtype}"
        assert want.dtype == got.dtype, f"{nid}: {want.dtype} through the graph, {got.dtype} through the plan"
        np.testing.assert_array_equal(want, got, err_msg=f"{nid} binds different bytes through the two loaders")
        assert np.asarray(decoded[path]).dtype != np.uint8, "without the flag the plan-keyed read decodes — the bug this guards"


def test_spelled_nvfp4_matmul_matches_oracle(tmp_path):
    """The spelled graph's numpy execution: matmul through the decode cone equals the
    oracle. Runs on graph-op forwards (pre-lowering), so it needs no Cling."""
    from emmy.compiler.backend.numpy import NumpyBackend

    g, y, (packed, scale_bits, s2) = _nvfp4_matmul_graph(tmp_path)
    x = rng.random((4, 32)).astype(np.float32)
    data = load_constants_from_safetensors(g, str(tmp_path))
    result, _ = NumpyBackend().run(g, input_data={**data, "x": x})
    ref = x @ dequantize_nvfp4(packed, scale_bits, s2).T
    np.testing.assert_allclose(result.outputs[y], ref, rtol=1e-6)


def test_spelled_nvfp4_matmul_fuses_into_one_loop(tmp_path):
    """Baseline of the pre-kernel-phase lowering: the cone survives constant folding
    and loop fusion folds the decode into the consumer's single LoopOp — no
    materialized decoded weight between kernels. Structure only; LoopOp execution
    goes through the Cling JIT and is covered by the CUDA/CI lanes."""
    from emmy.compiler.pipeline import LOOP_PASSES, Pipeline

    g, _y, _ = _nvfp4_matmul_graph(tmp_path)
    lowered = Pipeline.build(LOOP_PASSES).run(g)
    loops = [nid for nid, n in lowered.nodes.items() if type(n.op).__name__ == "LoopOp"]
    assert len(loops) == 1, f"decode did not fuse into the matmul loop nest: {loops}"
    consts = {nid for nid, n in lowered.nodes.items() if isinstance(n.op, ConstantOp)}
    assert consts == {"w_bits", "w_f4_pairs", "w_scale_bits", "w_scale_2"}


def test_is_quantized_dir_recognizes_nvfp4(tmp_path):
    # The trace-side quantized-checkpoint check: an NVFP4 checkpoint must take the
    # quantized-twin path. Plain from_pretrained skips the unknown modelopt scheme
    # with a warning, loads raw tensors, and dies on the packed [N, K/2] shape
    # mismatches.
    from emmy.compiler.trace.huggingface import _is_quantized_dir

    _nvfp4_checkpoint(tmp_path)
    assert _is_quantized_dir(tmp_path)


def test_load_dequantized_state_dict_nvfp4(tmp_path):
    """The twin read of a synthetic NVFP4 checkpoint: the packed trio dequantizes to the
    oracle's exact values, consumed scales drop, activation-quant metadata and bf16
    modules pass through, and the boot summary names the dialect."""
    from emmy.compiler.loader.quant import checkpoint_quant_summary, load_dequantized_state_dict

    packed = rng.integers(0, 256, (2, 16)).astype(np.uint8)  # logical K = 32
    scale_bits = np.array([[_e4m3(sign=0, exp=8, man=0), _e4m3(sign=0, exp=7, man=4)]] * 2, dtype=np.uint8)
    d = tmp_path / "nvfp4"
    d.mkdir()
    _write_checkpoint(
        d,
        {
            "layer.weight": torch.from_numpy(packed),
            "layer.weight_scale": _fp8_tensor(scale_bits),
            "layer.weight_scale_2": torch.tensor(0.25, dtype=torch.float32),
            "layer.input_scale": torch.tensor(1.5, dtype=torch.float32),
            "emb.weight": torch.randn(4, 8, dtype=torch.bfloat16),
        },
        quant_config=_FP4_MODELOPT_QC,
    )

    sd = load_dequantized_state_dict(d)
    np.testing.assert_array_equal(sd["layer.weight"], dequantize_nvfp4(packed, scale_bits, np.array(0.25, dtype=np.float32)))
    assert "layer.weight_scale" not in sd and "layer.weight_scale_2" not in sd
    assert sd["layer.input_scale"] == np.float32(1.5)
    assert sd["emb.weight"].shape == (4, 8)
    assert checkpoint_quant_summary(d) == "nvfp4 modelopt"


def test_is_exl3_checkpoint_is_the_narrow_coded_trunk_decision(tmp_path):
    from emmy.compiler.loader.quant import is_exl3_checkpoint

    exl3 = _rung(tmp_path / "exl3", 1.98)
    fp8 = tmp_path / "fp8"
    plain = tmp_path / "plain-format"
    fp8.mkdir()
    plain.mkdir()
    (fp8 / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "fp8"}}))
    (plain / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "gptq"}}))

    assert is_exl3_checkpoint(exl3)
    assert not is_exl3_checkpoint(fp8)
    assert not is_exl3_checkpoint(plain)
    assert not is_exl3_checkpoint(tmp_path / "absent")


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


def _f8_constants(graph: Graph) -> dict[str, ConstantOp]:
    return {
        nid: n.op
        for nid, n in graph.nodes.items()
        if isinstance(n.op, ConstantOp) and getattr(n.output.dtype, "name", "") in ("f8e4m3", "f8e5m2")
    }


def test_trace_model_unquantized_checkpoint_takes_existing_path(tmp_path):
    """The same seam on an UNQUANTIZED checkpoint: detection returns None, the model
    loads through the pre-existing ``from_pretrained`` branch, and nothing is spelled."""
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
    assert not _f8_constants(graph)
    assert all(op.source_graph is None for _nid, op in graph.loadable_constants())
    # from_pretrained bound the real checkpoint weights, as before the wiring.
    got = dict(wrapper.named_parameters())["model.model.embed_tokens.weight"]
    torch.testing.assert_close(got, model.model.embed_tokens.weight, rtol=0, atol=0)


def test_trace_model_spells_cones_and_binds_dequantized_twin(tmp_path):
    """The compile/run seam on a quantized checkpoint: the traced twin carries the
    DEQUANTIZED real weights (not from_config's random init), and the dequant algebra is
    spelled on exactly the fp8-stored projection weights before any pass runs."""
    _config, ref_sd = _tiny_fp8_checkpoint(tmp_path)
    from emmy.commands.compile import _trace_model

    graph, (wrapper, _args, _kws) = _trace_model(str(tmp_path), None, 16)

    spelled = {op.source_path for op in _f8_constants(graph).values()}
    assert spelled, "no fp8 bits constants spelled"
    assert all(".layers." in p and p.endswith(".weight") for p in spelled)
    assert any("q_proj" in p for p in spelled) and any("down_proj" in p for p in spelled)
    for _nid, op in graph.loadable_constants():
        if op.source_path and ("embed_tokens" in op.source_path or "lm_head" in op.source_path):
            assert op.source_dtype != "f8e4m3", f"unexpected spelling on {op.source_path}"
    # Each bits constant and every dynamically encoded activation are consumed
    # by their matching decode cast in-graph.
    decode_ops = [n for n in graph.nodes.values() if isinstance(n.op, ElementwiseOp) and n.op.op.decodes is not None]
    weight_decodes = [
        node
        for node in decode_ops
        if isinstance(graph.producer(node.inputs[0]).op, ConstantOp) and graph.producer(node.inputs[0]).output.dtype.name == "f8e4m3"
    ]
    activation_encodes = [node for node in graph.nodes.values() if isinstance(node.op, ElementwiseOp) and node.op.op.name == "to_f8e4m3"]
    assert len(weight_decodes) == len(spelled)
    assert len(decode_ops) == len(spelled) + len(activation_encodes)
    assert activation_encodes

    params = dict(wrapper.named_parameters())
    for name, ref in ref_sd.items():
        got = params["model." + name]  # the trace wrapper nests the CausalLM under .model
        torch.testing.assert_close(got, ref, rtol=0, atol=0)


def _dynamic_fp8_activation_hook(_module, args):
    """Eager mirror of the spelled dynamic fp8 activation algebra
    (``quant.py::_spell_dynamic_activation``): per-row amax with a 1e-12 floor, scale to the
    e4m3 finite max, round-trip through fp8 bits."""
    (x,) = args
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 448.0
    return ((x / scale).to(torch.float8_e4m3fn).float() * scale,)


def _run_e2e(tmp_path, config, ref_sd):
    """Compile the traced quantized model on the CUDA backend, bind computed buffers from the
    wrapper and every checkpoint tensor through the safetensors loader (bits + scales for the
    in-graph cone and plain reads for the rest), and return
    ``(emmy_logits, ref_logits, compiled)``. When the checkpoint declares dynamic fp8
    activations, the eager reference applies the same activation quantization on every
    coded projection, matching the algebra the trace spells in-graph — the pure dequantized
    model would differ by the inherent activation rounding (~1e-2 on these logits), which is
    not a backend error."""
    import transformers

    from emmy.commands.compile import _trace_model
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.loader.binder import bind_constants
    from emmy.compiler.loader.quant import _dynamic_activation_declaration, _fp8_quant_config

    graph, (wrapper, _args, _kws) = _trace_model(str(tmp_path), None, 16)
    backend = CudaBackend()
    compiled = backend.compile(graph)

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
    enabled, fmt = _dynamic_activation_declaration(_fp8_quant_config(tmp_path))
    if enabled:
        assert fmt == "f8e4m3", f"the hook mirrors e4m3 only, checkpoint declares {fmt}"
        for name, module in ref_model.named_modules():
            if isinstance(module, torch.nn.Linear) and ".layers." in name:  # the fp8-stored projections
                module.register_forward_pre_hook(_dynamic_fp8_activation_hook)
    with torch.no_grad():
        ref_logits = ref_model(input_ids=ids).logits.numpy()
    return emmy_logits, ref_logits, compiled


def _assert_e2e_gate(emmy_logits, ref_logits, label):
    assert not np.isnan(emmy_logits).any()
    assert np.abs(ref_logits).max() > 0.05, "reference logits suspiciously small; tolerance would be trivial"
    max_diff = np.abs(emmy_logits - ref_logits).max()
    assert max_diff < 5e-3, f"max_diff={max_diff} vs eager dequant reference ({label})"


@requires_cuda
def test_quantized_checkpoint_e2e_cuda(tmp_path):
    """Whole tiny quantized model through the same seam ``emmy compile`` / ``emmy run`` use,
    compiled on the CUDA backend with the dequant cone unconditionally in-graph: fp8 bits stay
    compressed in device memory, and the output matches the eager reference running the same
    dequant + dynamic-activation algebra."""
    config, ref_sd = _tiny_fp8_checkpoint(tmp_path)
    emmy_logits, ref_logits, compiled = _run_e2e(tmp_path, config, ref_sd)
    assert _f8_constants(compiled), "no fp8-dtype constant survived to the compiled graph — the cone did not stay in-graph"
    _assert_e2e_gate(emmy_logits, ref_logits, "expanded fp8 algebra")


# ===================================================================
# Invariant gate: quantization is not a concept past the decomposition band
# ===================================================================

# The invariant (see compiler/ARCHITECTURE.md, "Quantized checkpoints"): downstream layers —
# lowering, backends, search — may know canonical dtypes (f8e4m3), decode-trait elementwise ops
# (``ElementwiseImpl.decodes``), and graph algebra; NEVER checkpoint formats, scheme names,
# scale pairing, or quantization metadata. The frontend band below (the birth-time speller +
# the checkpoint readers + the twin construction and its post-trace call site) is the only
# place quantization-as-a-concept exists. ``from_f8*`` / ``decodes`` / f8 dtype tokens are
# sanctioned everywhere and deliberately NOT in the pattern. A new match must be
# frontend/loader-band code and must join the allowlist with that justification — anything
# else is the leak this gate exists to stop.
_QUANT_CONCEPT_PATTERN = r"QuantSpec|quantization_config|quant_method|weight_scale|modules_to_not_convert|dequant"

_FRONTEND_BAND_ALLOWLIST = {
    "emmy/commands/compile.py",  # the post-trace spelling call site (twin trace + speller)
    "emmy/compiler/loader/exl3.py",  # EXL3 format reader: the trellis decode + the weight-free allocation sidecar
    "emmy/compiler/loader/quant.py",  # the speller + scheme detection + dequant math
    "emmy/compiler/loader/safetensors.py",  # checkpoint reads (fp8 bits, scale tensors)
    "emmy/compiler/trace/huggingface.py",  # quantized-twin construction + detection
}


def test_quantization_concepts_stay_in_the_frontend_band():
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    pat = re.compile(_QUANT_CONCEPT_PATTERN)
    offenders = {str(p.relative_to(root)) for p in (root / "emmy").rglob("*.py") if pat.search(p.read_text())} - _FRONTEND_BAND_ALLOWLIST
    assert not offenders, (
        f"quantization concepts referenced outside the frontend/loader band: {sorted(offenders)}. "
        "Past the decomposition band a quantized weight is just constants + algebra — downstream "
        "code may key on dtypes and the decode trait, never on checkpoint formats or scale "
        "pairing (see the invariant comment above)."
    )


def test_quantize_nvfp4_returns_the_checkpoint_carriers():
    # The trio must be storable as-is: packed pairs halve the last axis, one e4m3 scale per 16
    # elements, one f32 for the tensor.
    x = np.random.default_rng(0).standard_normal((3, 64)).astype(np.float32)
    packed, scale_bits, scale_2 = quantize_nvfp4(x)
    assert (packed.dtype, packed.shape) == (np.uint8, (3, 32))
    assert (scale_bits.dtype, scale_bits.shape) == (np.uint8, (3, 4))
    assert (scale_2.dtype, scale_2.shape) == (np.float32, (1,))
    # Non-negative block scales, so the same bits read as the unsigned ue4m3 the hardware wants.
    assert not (scale_bits >> 7).any()


def test_quantize_nvfp4_stays_within_the_format_over_a_wide_range():
    # Rows three decades apart, which is the case a single scale level would clip. The bound is
    # the format's own: within a block the largest magnitude maps to 6.0, where the grid steps by
    # 2, so nearest-rounding cannot miss by more than 1 in 6 of that block's largest value.
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((3, 64)) * np.array([[0.01], [1.0], [80.0]])).astype(np.float32)
    back = dequantize_nvfp4(*quantize_nvfp4(x))
    block_amax = np.abs(x).reshape(3, 4, 16).max(axis=-1).repeat(16, axis=-1).reshape(3, 64)
    assert (np.abs(back - x) <= block_amax / 6 + 1e-6).all()


def test_quantize_nvfp4_is_idempotent_through_a_dequantize():
    # Values already ON the grid must quantize to themselves, which is what makes the pair a
    # round trip rather than a lossy pass that keeps drifting.
    x = np.random.default_rng(1).standard_normal((2, 32)).astype(np.float32)
    once = dequantize_nvfp4(*quantize_nvfp4(x))
    np.testing.assert_array_equal(dequantize_nvfp4(*quantize_nvfp4(once)), once)


def test_quantize_nvfp4_keeps_a_zero_block_zero():
    packed, scale_bits, scale_2 = quantize_nvfp4(np.zeros((1, 16), dtype=np.float32))
    assert not dequantize_nvfp4(packed, scale_bits, scale_2).any()


def test_quantize_nvfp4_rejects_a_ragged_last_axis():
    with pytest.raises(AssertionError):
        quantize_nvfp4(np.zeros((1, 20), dtype=np.float32))


@pytest.mark.parametrize("magnitude", [0.0, 1e-42, 1e-30])
def test_quantize_nvfp4_survives_a_vanishing_tensor(magnitude):
    # Without a floor on the tensor amax, scale_2 underflows to zero and the block-scale division
    # blows up. The result stays finite and stays zero.
    x = np.full((1, 16), magnitude, dtype=np.float32)
    packed, scale_bits, scale_2 = quantize_nvfp4(x)
    assert scale_2[0] > 0
    back = dequantize_nvfp4(packed, scale_bits, scale_2)
    assert np.isfinite(back).all()


def test_quantize_nvfp4_rejects_a_tensor_too_large_for_an_f16_fused_scale():
    # fuse_nvfp4_scales hands back f16, so a tensor whose block scale overflows f16 has no
    # representable fused scale — an error beats a silent inf.
    with pytest.raises(AssertionError, match="f16 fused scale"):
        quantize_nvfp4(np.full((1, 16), 3e38, dtype=np.float32))
