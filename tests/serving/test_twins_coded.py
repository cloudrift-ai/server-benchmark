"""The serving twins' coded arm spells checkpoint siblings through generic tensor algebra.

These tests drive the spelling stage directly off a synthetic storage listing (no network, no
checkpoint, no transformers): the tracing stage is already covered by the drift gate, and what is
new here is the pairing — which traced module gets which checkpoint entry, and how the per-tensor
rate allocation multiplies the twins.
"""

from __future__ import annotations

import pytest

from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.frontend.ir import LinearOp
from emmy.compiler.loader.safetensors import split_revision
from emmy.serving.twins import _profile_layers, _spell_coded_twins, _spell_expert_twins


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
        if isinstance(nd.op, ConstantOp) and nd.op.source_path and nd.op.source_path.endswith(".trellis"):
            out[nd.op.source_path] = int(nd.output.shape[2].as_static()) // 16
    return out


def test_coded_twin_spells_the_deployed_contraction():
    """Each traced weight is matched to its checkpoint module by dotted suffix within a layer
    (a twin traces ``q_proj.weight`` where the checkpoint says ``…self_attn.q_proj``) and spelled
    by the same generic speller used by the loader."""
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


def test_laguna_singular_shared_expert_matches_plural_transformers_path():
    storage = {"model.layers.1.mlp.shared_expert.gate_proj": _entry("model.layers.1.mlp.shared_expert.gate_proj", 1024, 3072, 4)}
    twins = _spell_coded_twins({"post1": _twin({"mlp.shared_experts.gate_proj": (1024, 3072)})}, storage)
    assert _decodes(twins["post1@b4"]) == {"model.layers.1.mlp.shared_expert.gate_proj.trellis": 4}


def test_laguna_selects_dense_full_sparse_sliding_and_sparse_full_profiles():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    cfg = transformers.LagunaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts=4,
        num_experts_per_tok=2,
        layer_types=["full_attention", "sliding_attention", "full_attention"],
        mlp_layer_types=["dense", "sparse", "sparse"],
        num_attention_heads_per_layer=[4, 6, 4],
        sliding_window=16,
    )
    with torch.device("meta"):
        trunk = transformers.AutoModel.from_config(cfg, dtype=torch.float16)
    profiles = _profile_layers(trunk, cfg)
    assert [(i, suffix) for i, _block, suffix in profiles] == [
        (0, "-dense-full"),
        (1, "-sparse-sliding"),
        (2, "-sparse-full"),
    ]


def test_laguna_coded_expert_inputs_are_spelled_per_allocation_profile():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    from emmy.serving.gen_runner import trace_split

    class Expert(nn.Module):
        def forward(self, x, w_gate, w_up, w_down):
            return nn.functional.linear(nn.functional.silu(nn.functional.linear(x, w_gate)) * nn.functional.linear(x, w_up), w_down)

    # The deployed Laguna shape. Shape-only compilation allocates no checkpoint tensors and
    # catches primitives which tiny padded examples can accidentally fold away.
    h, inter = 3072, 1024
    graph = trace_split(
        Expert(),
        (
            torch.zeros(1, h, dtype=torch.float16),
            torch.zeros(inter, h, dtype=torch.float16),
            torch.zeros(inter, h, dtype=torch.float16),
            torch.zeros(h, inter, dtype=torch.float16),
        ),
        None,
    )
    storage = {}
    for proj, n, k in (("gate_proj", inter, h), ("up_proj", inter, h), ("down_proj", h, inter)):
        base = f"model.layers.1.mlp.experts.0.{proj}"
        storage[base] = _entry(base, n, k, 2)
    twins = _spell_expert_twins("expert1-sparse-sliding", graph, storage)
    assert list(twins) == ["expert1-sparse-sliding@b2"]
    spelled = twins["expert1-sparse-sliding@b2"]
    assert spelled.inputs[:4] == ["x", "w_gate", "w_up", "w_down"]
    assert {spelled.nodes[name].output.dtype.name for name in ("w_gate", "w_up", "w_down")} == {"i16"}
    assert all(type(node.op).__name__ not in {"TrellisDecodeOp", "HadamardOp"} for node in spelled.nodes.values())

    from emmy.compiler.backend.plan import plan_from_graph
    from emmy.compiler.context import Context
    from emmy.compiler.ir.base import ConstantOp, InputOp
    from emmy.compiler.ir.cuda import CudaOp
    from emmy.compiler.pipeline import CUDA_PASSES, Pipeline

    lowered = Pipeline.build(CUDA_PASSES).run(spelled, ctx=Context(compute_capability=(12, 0)))
    cuda = [node.op for node in lowered.nodes.values() if isinstance(node.op, CudaOp)]
    assert cuda and all(op.kernel_source for op in cuda)
    assert all(isinstance(node.op, (InputOp, ConstantOp, CudaOp)) for node in lowered.nodes.values())
    plan = plan_from_graph(lowered)
    assert plan.launches and plan.weights
    assert all(weight.generated is not None and weight.load_ops == () for weight in plan.weights.values())
    assert {f"{weight}_decoded_tile_step" for weight in ("w_gate", "w_up", "w_down")} <= set(plan.weights)
    factors = [spec.generated for spec in plan.weights.values() if spec.generated is not None and spec.generated[1] == (128, 128)]
    assert len(factors) == 3 and {factor[0] for factor in factors} == {"<f4"}
    from emmy.serving.gen_runner import _bind_plan_constants

    assert set(_bind_plan_constants(plan, {}, cache=None)) == set(plan.weights)
    active_ir = "\n".join(f"{nid} {type(node.op).__module__} {type(node.op).__name__}" for nid, node in lowered.nodes.items())
    assert "trellis" not in active_ir.lower() and "exl3" not in active_ir.lower()


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
