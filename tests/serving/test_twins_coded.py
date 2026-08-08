"""The serving twins' CODED arm (VQ Phase 4): a weight-free twin of an EXL3 checkpoint must spell
the same coded contractions the deployed program does.

Serving retargets each traced constant to its checkpoint key and runs
``spell_trellis_constants(…, expand=True)``, so a deployed trunk linear is the activation-side
basis restore around a hat-basis coded contraction — not an f16 matmul. A twin that skipped that
would key every trunk projection as its uncompressed twin and report GAP on every coded golden,
which is why the first coded golden file shipped untagged. The arm's weight-free source is the
checkpoint's ``quantization_config.json``.

These tests drive the spelling stage directly off a synthetic storage listing (no network, no
checkpoint, no transformers): the tracing stage is already covered by the drift gate, and what is
new here is the pairing — which traced module gets which checkpoint entry, and how the per-tensor
rate allocation multiplies the twins.
"""

from __future__ import annotations

import pytest

from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.frontend.ir import LinearOp, TrellisDecodeOp
from emmy.compiler.loader.safetensors import split_revision
from emmy.serving.twins import _spell_coded_twins


def _entry(base: str, n: int, k: int, bits: int) -> dict:
    """One ``tensor_storage`` entry, shaped exactly as exllamav3 writes it."""
    n_pad, k_pad = -(-n // 128) * 128, -(-k // 128) * 128
    return {
        "quant_format": "exl3",
        "bits_per_weight": bits,
        "stored_tensors": {
            f"{base}.trellis": {"shape": [k_pad // 16, n_pad // 16, 16 * bits]},
            f"{base}.suh": {"shape": [k_pad]},
            f"{base}.svh": {"shape": [n_pad]},
        },
    }


def _twin(mods: dict[str, tuple[int, int]], m: int = 1) -> Graph:
    """A twin-shaped graph: one activation input and one ``F.linear`` per traced module, with the
    wrapper-relative constant paths a split-wrapper trace produces (``q_proj.weight``)."""
    g = Graph()
    hidden = next(iter(mods.values()))[1]
    g.add_node(InputOp(), [], Tensor("x", (m, hidden), "f16"), node_id="x")
    for mod, (n, k) in mods.items():
        nid = mod.replace(".", "_")
        g.add_node(
            ConstantOp(name=nid, source_path=f"{mod}.weight", source_shape=(n, k), source_dtype="f16"), [], Tensor(nid, (n, k), "f16")
        )
        g.add_node(LinearOp(), ["x", nid], Tensor(f"y_{nid}", (m, n), "f16"), node_id=f"y_{nid}")
    g.inputs, g.outputs = ["x"], [f"y_{mod.replace('.', '_')}" for mod in mods]
    return g


def _decodes(g: Graph) -> dict[str, int]:
    """``{codes source path: k_bits}`` for every coded contraction the graph spells."""
    out = {}
    for nd in g.nodes.values():
        if isinstance(nd.op, TrellisDecodeOp):
            codes = g.nodes[nd.inputs[0]]
            out[codes.op.source_path] = int(codes.output.shape[2].as_static()) // 16
    return out


def test_coded_twin_spells_the_deployed_contraction():
    """Each traced weight is matched to its checkpoint module by dotted suffix within a layer
    (a twin traces ``q_proj.weight`` where the checkpoint says ``…self_attn.q_proj``) and spelled
    by the DEPLOYED speller. The kernel-path hint must ride along: without it ``032`` folds the
    hat-basis cone into a bind-time constant a weight-free twin cannot evaluate."""
    storage = {
        "model.layers.0.self_attn.q_proj": _entry("model.layers.0.self_attn.q_proj", 12288, 4096, 4),
        "model.layers.0.mlp.gate_proj": _entry("model.layers.0.mlp.gate_proj", 10944, 4096, 2),
    }
    twins = _spell_coded_twins({"pre1": _twin({"q_proj": (12288, 4096), "mlp.gate_proj": (10944, 4096)})}, storage)
    assert list(twins) == ["pre1@b2-4"]
    assert _decodes(twins["pre1@b2-4"]) == {
        "model.layers.0.self_attn.q_proj.trellis": 4,
        "model.layers.0.mlp.gate_proj.trellis": 2,
    }
    assert twins["pre1@b2-4"].hints.get("trellis.expand", False) is True


def test_one_twin_per_rate_profile():
    """An "optimized" rung allocates bits per tensor, and the rate is part of the ShapeKey, so one
    traced layer does not represent the trunk: every distinct allocation is emitted. (The pinned
    GLM-4.5-Air 2.25 checkpoint really does ship q/k/v at 4 bits on 42 layers and 3 on 4.)"""
    storage = {}
    for layer, bits in ((0, 4), (1, 4), (2, 3)):
        storage[f"model.layers.{layer}.self_attn.q_proj"] = _entry(f"model.layers.{layer}.self_attn.q_proj", 12288, 4096, bits)
    twins = _spell_coded_twins({"pre1": _twin({"q_proj": (12288, 4096)})}, storage)
    assert sorted(twins) == ["pre1@b3", "pre1@b4"]  # layer 1 repeats layer 0's profile and is dropped
    assert _decodes(twins["pre1@b4"]) == {"model.layers.0.self_attn.q_proj.trellis": 4}
    assert _decodes(twins["pre1@b3"]) == {"model.layers.2.self_attn.q_proj.trellis": 3}


def test_uncoded_twin_passes_through_untouched():
    """A twin holding no coded weight keeps its original name and graph — the coded arm must not
    rename or rewrite the models that make up every other golden file."""
    graph = _twin({"q_proj": (12288, 4096)})
    twins = _spell_coded_twins({"pre1": graph}, {"model.layers.0.mlp.gate_proj": _entry("model.layers.0.mlp.gate_proj", 10944, 4096, 2)})
    assert twins == {"pre1": graph}


def test_an_ambiguous_suffix_names_no_module():
    """``mlp.gate_proj`` must hit the dense MLP and never an expert's ``gate_proj``. Where the
    suffix is genuinely ambiguous the module is left uncoded — guessing spells the wrong rate,
    and an uncoded fork shows up as a GAP rather than as a wrong MATCH."""
    storage = {
        "model.layers.0.mlp.gate_proj": _entry("model.layers.0.mlp.gate_proj", 10944, 4096, 2),
        "model.layers.0.mlp.experts.0.gate_proj": _entry("model.layers.0.mlp.experts.0.gate_proj", 1408, 4096, 2),
        "model.layers.0.mlp.experts.1.gate_proj": _entry("model.layers.0.mlp.experts.1.gate_proj", 1408, 4096, 2),
    }
    dense = _spell_coded_twins({"post1": _twin({"mlp.gate_proj": (10944, 4096)})}, storage)
    assert _decodes(dense["post1@b2"]) == {"model.layers.0.mlp.gate_proj.trellis": 2}
    # A bare ``gate_proj`` matches all three names, so nothing is spelled.
    assert _spell_coded_twins({"post1": (g := _twin({"gate_proj": (1408, 4096)}))}, storage) == {"post1": g}


@pytest.mark.parametrize(
    ("spec", "want"),
    [
        ("turboderp/GLM-4.5-Air-exl3@2.25bpw", ("turboderp/GLM-4.5-Air-exl3", "2.25bpw")),
        ("google/gemma-3-12b-it", ("google/gemma-3-12b-it", None)),
    ],
)
def test_model_tag_may_pin_the_rung(spec, want):
    """A coded checkpoint's rung lives on a branch, and the rungs differ in exactly the bit
    allocation the keys carry — so the ``model:`` tag may pin one."""
    assert split_revision(spec) == want
