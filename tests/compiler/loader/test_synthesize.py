"""``--quantize``: a traced module's weights written as a real checkpoint, then read back.

The point of the design under test is that there is only ONE producer of quantized graphs. This
helper does not synthesize the decode algebra; it writes the checkpoint the spellers already read
and then reads it, so an inline expression compiles to exactly the program the same checkpoint
would give. These tests hold that: the written trio round-trips through the format's own
dequantize, the graph the spellers produce is the marked W4A4 one, and the activation scale is a
number on disk rather than an assumption.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from emmy.compiler.loader.quant import dequantize_nvfp4, spell_quantized_constants, spell_static_fp4_activations
from emmy.compiler.loader.synthesize import SCHEMES, write_quantized_checkpoint

pytest.importorskip("torch")


def _traced(code: str):
    from emmy.commands.trace import graph_from_code

    graph, _slug, bundle = graph_from_code(code)
    return graph, bundle


def test_a_traced_linear_becomes_a_checkpoint_the_spellers_read(tmp_path):
    """The whole contract in one pass: write, spell, and land on the declared W4A4 program."""
    graph, bundle = _traced("nn.Linear(256, 128, bias=False)(torch.randn(64, 256))")
    ckpt = write_quantized_checkpoint(graph, bundle, tmp_path / "ckpt")

    cfg = json.loads((ckpt / "config.json").read_text())["quantization_config"]
    assert cfg["quant_algo"] == "NVFP4" and cfg["quant_method"] == "modelopt"
    assert cfg["config_groups"]["group_0"]["input_activations"]["num_bits"] == 4, "the static activation declaration is the marker"

    assert spell_quantized_constants(graph, str(ckpt)) == 1
    assert spell_static_fp4_activations(graph, str(ckpt)) == 1
    packed = [n for n in graph.nodes.values() if n.output.dtype.name == "f4e2m1x2"]
    assert len(packed) == 2, "one packed weight and one packed activation"
    assert any(type(n.op).__name__ == "ElementwiseOp" and n.op.name == "to_f4e2m1" for n in graph.nodes.values())


def test_the_written_weight_round_trips_through_the_formats_own_dequantize(tmp_path):
    """What was written IS the weight, to one NVFP4 step. The checkpoint has to mean something on
    its own — it is the artifact a reader inspects when a dump looks wrong."""
    import torch
    from safetensors import safe_open

    torch.manual_seed(4)
    graph, bundle = _traced("nn.Linear(256, 64, bias=False)(torch.randn(32, 256))")
    original = bundle[0].state_dict()["weight"].detach().to(torch.float32).numpy()
    ckpt = write_quantized_checkpoint(graph, bundle, tmp_path / "ckpt")

    # The e4m3 scales have no numpy dtype, so they come back through torch as raw bits — the
    # same carrier ``dequantize_nvfp4`` takes.
    with safe_open(str(ckpt / "model.safetensors"), framework="pt") as f:
        back = dequantize_nvfp4(
            f.get_tensor("l0.weight").numpy(),
            f.get_tensor("l0.weight_scale").view(torch.uint8).numpy(),
            f.get_tensor("l0.weight_scale_2").numpy(),
        )
    rel = np.abs(back - original).max() / max(float(np.abs(original).max()), 1e-9)
    assert rel < 0.2, "past one NVFP4 quantization step of the original weight"


def test_the_activation_scale_is_calibrated_from_the_example_input(tmp_path):
    """``input_scale`` is modelopt's ``amax / (6 · 448)`` over the trace's one example input.
    Calibration on one sample is a real calibration and a poor one; the number is on disk so a
    reader can see which it is rather than inferring it."""
    from safetensors import safe_open

    graph, bundle = _traced("nn.Linear(256, 64, bias=False)(torch.full((32, 256), 3.0))")
    ckpt = write_quantized_checkpoint(graph, bundle, tmp_path / "ckpt")
    with safe_open(str(ckpt / "model.safetensors"), framework="pt") as f:
        got = float(np.asarray(f.get_tensor("l0.input_scale"), dtype=np.float32).reshape(-1)[0])
    assert got == pytest.approx(3.0 / (6.0 * 448.0), rel=1e-6)


def test_an_unknown_scheme_and_a_shapeless_graph_both_refuse(tmp_path):
    graph, bundle = _traced("nn.Linear(256, 64, bias=False)(torch.randn(32, 256))")
    with pytest.raises(ValueError, match="unknown quantization scheme"):
        write_quantized_checkpoint(graph, bundle, tmp_path / "a", scheme="int4")
    assert SCHEMES == ("nvfp4",)

    bare, bundle2 = _traced("torch.exp(torch.randn(8, 8))")
    with pytest.raises(ValueError, match="no linear"):
        write_quantized_checkpoint(bare, bundle2, tmp_path / "b")
