"""``180_expand_quantized_constant`` — the quantized-constant dequant-cone expansion (M2 of the
FP8 plan): structure per granularity, interface invariance, the ``EMMY_FP8_EXPAND`` gate and the
layout-commute fallbacks, exact numerics vs the M1 bind-time dequant, and the transposed
``load_ops`` end-to-end path through the safetensors loader (raw fp8 bits + block-image scale
chain)."""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler.backend.numpy import NumpyBackend
from emmy.compiler.dtype import decode_f8
from emmy.compiler.graph import Graph
from emmy.compiler.ir.base import ConstantOp, QuantSpec
from emmy.compiler.ir.frontend.ir import ReshapeOp, TransposeOp
from emmy.compiler.ir.tensor.ir import ElementwiseOp
from emmy.compiler.loader.quant import dequantize
from emmy.compiler.pipeline import Pipeline
from emmy.compiler.tensor import Tensor

rng = np.random.default_rng(7)
_backend = NumpyBackend()

_RULE = "180_expand_quantized_constant"


@pytest.fixture
def fp8_expand_on(monkeypatch):
    monkeypatch.setenv("EMMY_FP8_EXPAND", "1")


def _finite_bits(shape):
    """fp8 bit patterns avoiding the e4m3 NaN codes, so exact comparisons stay finite."""
    bits = rng.integers(0, 256, shape).astype(np.uint8)
    bits[bits == 0x7F] = 0x00
    bits[bits == 0xFF] = 0x80
    return bits


def _qgraph(w_shape=(8, 16), scale_shape=(8, 1), *, dtype="f32", inverse=False, fmt="f8e4m3", load_ops=(), out_shape=None):
    """A graph of one quantized weight constant (its output is the graph output)."""
    g = Graph()
    spec = QuantSpec(scale_path="layer.weight_scale", scale_shape=scale_shape, scale_dtype="f32", inverse=inverse, fmt=fmt)
    g.add_node(
        op=ConstantOp(
            name="p_w",
            source_path="layer.weight",
            source_shape=w_shape,
            source_dtype=dtype,
            quant=spec,
            load_ops=tuple(load_ops),
        ),
        inputs=[],
        output=Tensor("p_w", out_shape or w_shape, dtype),
        node_id="p_w",
    )
    g.inputs, g.outputs = [], ["p_w"]
    return g


def _apply(graph: Graph) -> Graph:
    return Pipeline.build(["frontend/decomposition"], select=[_RULE]).run(graph)


def _ops_by_type(graph: Graph, op_type) -> list:
    return [n for n in graph.nodes.values() if isinstance(n.op, op_type)]


def _constants(graph: Graph) -> dict[str, ConstantOp]:
    return {nid: n.op for nid, n in graph.nodes.items() if isinstance(n.op, ConstantOp)}


# ===================================================================
# Structure per granularity
# ===================================================================


def test_per_tensor_expands_to_minimal_broadcast_form(fp8_expand_on):
    result = _apply(_qgraph(scale_shape=(1, 1)))
    consts = _constants(result)
    w = next(op for op in consts.values() if op.source_path == "layer.weight")
    assert w.quant is None  # spec consumed by the expansion
    scale = next(op for op in consts.values() if op.source_path == "layer.weight_scale")
    assert scale.source_shape == (1, 1) and scale.quant is None
    casts = [n for n in _ops_by_type(result, ElementwiseOp) if n.op.name == "from_f8e4m3"]
    assert len(casts) == 1
    # Degenerate block: no reshape pair — the scale broadcasts straight onto the weight.
    assert not _ops_by_type(result, ReshapeOp)
    assert any(n.op.name == "multiply" for n in _ops_by_type(result, ElementwiseOp))


def test_per_channel_expands_without_reshape_pair(fp8_expand_on):
    result = _apply(_qgraph(scale_shape=(8, 1)))
    assert not _ops_by_type(result, ReshapeOp)
    scale_node = next(n for n in result.nodes.values() if isinstance(n.op, ConstantOp) and n.op.source_path == "layer.weight_scale")
    assert tuple(d.as_static() for d in scale_node.output.shape) == (8, 1)


def test_2d_block_expands_to_interleaved_reshape_pair(fp8_expand_on):
    result = _apply(_qgraph(scale_shape=(2, 4)))  # derived block (4, 4)
    reshapes = _ops_by_type(result, ReshapeOp)
    shapes = sorted(tuple(int(d) for d in n.op.shape) for n in reshapes)
    assert shapes == [(2, 4, 4, 4), (8, 16)]  # blocked view + the reshape back
    scale_node = next(n for n in result.nodes.values() if isinstance(n.op, ConstantOp) and n.op.source_path == "layer.weight_scale")
    # The scale binds at the interleaved (grid, 1) layout via its own load_ops reshape.
    assert tuple(d.as_static() for d in scale_node.output.shape) == (2, 1, 4, 1)
    assert any(isinstance(lop, ReshapeOp) for lop in scale_node.op.load_ops)


def test_inverse_scale_expands_to_divide(fp8_expand_on):
    result = _apply(_qgraph(scale_shape=(8, 1), inverse=True))
    names = [n.op.name for n in _ops_by_type(result, ElementwiseOp)]
    assert "divide" in names and "multiply" not in names


def test_e5m2_fmt_selects_its_cast(fp8_expand_on):
    result = _apply(_qgraph(scale_shape=(1, 1), fmt="f8e5m2"))
    w = next(op for op in _constants(result).values() if op.source_path == "layer.weight")
    assert w.source_dtype == "f8e5m2"
    assert any(n.op.name == "from_f8e5m2" for n in _ops_by_type(result, ElementwiseOp))


def test_cone_output_keeps_the_promised_dtype_and_shape(fp8_expand_on):
    """Interface invariance: whatever the cone spells inside, its output tensor is
    exactly what the trace promised — dtype and shape unchanged."""
    for scale_shape, dtype in [((1, 1), "f32"), ((8, 1), "f16"), ((2, 4), "f32")]:
        g = _qgraph(scale_shape=scale_shape, dtype=dtype)
        orig = g.nodes[g.outputs[0]].output
        result = _apply(g)
        out = result.nodes[result.outputs[0]].output
        assert out.dtype == orig.dtype and tuple(out.shape) == tuple(orig.shape)


def test_w_constant_keeps_fp8_dtype_and_source_fields(fp8_expand_on):
    result = _apply(_qgraph(scale_shape=(8, 1)))
    w_node = next(n for n in result.nodes.values() if isinstance(n.op, ConstantOp) and n.op.source_path == "layer.weight")
    assert w_node.output.dtype.name == "f8e4m3"
    assert w_node.op.source_dtype == "f8e4m3"
    assert w_node.op.source_shape == (8, 16)


def test_no_quant_spec_survives_expansion(fp8_expand_on):
    result = _apply(_qgraph(scale_shape=(2, 4)))
    assert all(op.quant is None for op in _constants(result).values())


def test_idempotent(fp8_expand_on):
    once = _apply(_qgraph(scale_shape=(8, 1)))
    twice = _apply(once)
    assert len(twice.nodes) == len(once.nodes)


# ===================================================================
# Gate + layout-commute fallbacks (the M1 path stays intact)
# ===================================================================


def test_flag_off_leaves_the_constant_untouched():
    g = _qgraph(scale_shape=(8, 1))
    result = _apply(g)
    assert set(result.nodes) == {"p_w"}
    assert result.nodes["p_w"].op.quant is not None  # still the M1 bind-time path


def test_non_commuting_reshape_chain_falls_back_to_m1(fp8_expand_on):
    """A reshape under a non-scalar scale can cross quant-block boundaries, so the
    expansion declines and the constant keeps its spec (bind-time dequant)."""
    g = _qgraph(scale_shape=(8, 1), load_ops=(ReshapeOp(shape=(16, 8)),), out_shape=(16, 8))
    result = _apply(g)
    assert set(result.nodes) == {"p_w"}
    assert result.nodes["p_w"].op.quant is not None


def test_reshape_chain_commutes_under_per_tensor_scale(fp8_expand_on):
    g = _qgraph(scale_shape=(1, 1), load_ops=(ReshapeOp(shape=(16, 8)),), out_shape=(16, 8))
    result = _apply(g)
    w = next(op for op in _constants(result).values() if op.source_path == "layer.weight")
    assert w.quant is None and len(w.load_ops) == 1  # expanded; W keeps the chain


# ===================================================================
# Numerics: expanded cone on the interpreter == the M1 bind-time dequant.
# e4m3 → f16/f32 is exact and both paths multiply in f32 before the one
# rounding into the compute dtype, so equality is EXACT, not approximate.
# ===================================================================


@pytest.mark.parametrize(
    ("scale_shape", "inverse", "dtype"),
    [
        ((1, 1), False, "f32"),
        ((8, 1), False, "f32"),
        ((2, 4), False, "f32"),
        ((8, 1), True, "f32"),
        ((2, 4), False, "f16"),
    ],
    ids=["per-tensor", "per-channel", "block", "inverse", "f16-compute"],
)
def test_expanded_cone_matches_m1_dequant(fp8_expand_on, scale_shape, inverse, dtype):
    g = _apply(_qgraph(scale_shape=scale_shape, inverse=inverse, dtype=dtype))
    bits = _finite_bits((8, 16))
    scale = (np.abs(rng.standard_normal(scale_shape)) + 0.5).astype(np.float32)
    w_id = next(nid for nid, op in _constants(g).items() if op.source_path == "layer.weight")
    s_id = next(nid for nid, op in _constants(g).items() if op.source_path == "layer.weight_scale")
    s_shape = tuple(d.as_static() for d in g.nodes[s_id].output.shape)
    out = _backend.run(_backend.compile(g), input_data={w_id: bits, s_id: scale.reshape(s_shape)})[0].outputs
    ref = dequantize(decode_f8(bits, "f8e4m3"), scale, inverse=inverse)
    if dtype == "f16":
        ref = ref.astype(np.float16)
    np.testing.assert_array_equal(out["p_w"], ref)


# ===================================================================
# Transposed load_ops chain, end to end through the safetensors loader:
# W keeps the chain over the raw bits, the scale gets the block-image
# chain, and the bound + evaluated cone matches the M1 flag-off binding.
# ===================================================================

torch = pytest.importorskip("torch")


def _write_checkpoint(dirpath, bits, scale_np):
    import json

    from safetensors.torch import save_file

    tensors = {
        "layer.weight": torch.from_numpy(np.ascontiguousarray(bits)).view(torch.float8_e4m3fn),
        "layer.weight_scale": torch.from_numpy(scale_np),
    }
    save_file(tensors, str(dirpath / "model.safetensors"))
    cfg = {"model_type": "test", "quantization_config": {"quant_method": "fp8", "fmt": "e4m3", "modules_to_not_convert": []}}
    (dirpath / "config.json").write_text(json.dumps(cfg))


def test_transposed_chain_e2e_matches_m1_loader_binding(fp8_expand_on, tmp_path):
    from emmy.compiler.loader.quant import stamp_quant_specs
    from emmy.compiler.loader.safetensors import load_constants_from_safetensors

    bits = _finite_bits((8, 16))
    scale_np = (np.abs(rng.standard_normal((8, 1))) + 0.5).astype(np.float32)
    _write_checkpoint(tmp_path, bits, scale_np)

    def _graph():
        g = _qgraph(w_shape=(8, 16), scale_shape=(8, 1), load_ops=(TransposeOp(axes=(1, 0)),), out_shape=(16, 8))
        g.nodes["p_w"].op = ConstantOp(  # re-stamp from the checkpoint, not the fixture spec
            name="p_w",
            source_path="layer.weight",
            source_shape=(8, 16),
            source_dtype="f32",
            load_ops=(TransposeOp(axes=(1, 0)),),
        )
        assert stamp_quant_specs(g, str(tmp_path)) == 1
        return g

    # M1 reference: flag-off binding of the un-expanded graph.
    m1 = load_constants_from_safetensors(_graph(), str(tmp_path))["p_w"]

    expanded = _apply(_graph())
    w_node = next(n for n in expanded.nodes.values() if isinstance(n.op, ConstantOp) and n.op.source_path == "layer.weight")
    s_node = next(n for n in expanded.nodes.values() if isinstance(n.op, ConstantOp) and n.op.source_path == "layer.weight_scale")
    # W keeps the layout chain (bits are laid out post-chain); the scale carries its block image.
    assert [type(lop).__name__ for lop in w_node.op.load_ops] == ["TransposeOp"]
    assert [type(lop).__name__ for lop in s_node.op.load_ops] == ["TransposeOp"]
    assert tuple(d.as_static() for d in s_node.output.shape) == (1, 8)

    bound = load_constants_from_safetensors(expanded, str(tmp_path))
    assert bound[w_node.id].dtype == np.uint8  # raw bits, post-chain
    np.testing.assert_array_equal(bound[w_node.id], bits.T)
    out = _backend.run(_backend.compile(expanded), input_data=bound)[0].outputs
    np.testing.assert_array_equal(out["p_w"], m1)
