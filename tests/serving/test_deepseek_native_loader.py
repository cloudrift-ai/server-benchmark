"""The DeepSeek V4 native-checkpoint serving load (no GPU, no network).

The published checkpoint is not in Hugging Face naming and does not store its routed experts the
way the gpt-oss MXFP4 lineage does: modules are ``layers.N.attn.wq_a`` / ``layers.N.ffn.experts.E.w1``,
the trunk is fp8-e4m3 with ``.scale`` siblings holding E8M0 block exponents, and each routed expert
projection is native MXFP4 (``I8 [out, in/2]`` nibble pairs plus ``F8_E8M0 [out, in/32]`` exponents).
These tests build that dialect synthetically and pin what the loader must produce: a twin whose dense
trunk carries real values, and an expert store whose routed weights stay COMPRESSED (blocks + scales
as program inputs), optionally narrowed to one tensor-parallel rank's expert shard.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

_rng = np.random.default_rng(11)


def _tiny_native_config(hidden: int, inter: int, experts: int) -> dict:
    """A DeepSeek V4 config as published: fp8 trunk declaration plus ``expert_dtype: fp4``."""
    return {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "vocab_size": 64,
        "hidden_size": hidden,
        "moe_intermediate_size": inter,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "head_dim": 16,
        "qk_rope_head_dim": 8,
        "q_lora_rank": 16,
        "n_routed_experts": experts,
        "num_experts_per_tok": 2,
        "n_shared_experts": 1,
        "o_groups": 1,
        "o_lora_rank": 16,
        "index_n_heads": 2,
        "index_head_dim": 4,
        "index_topk": 2,
        "hc_mult": 2,
        "hc_sinkhorn_iters": 2,
        "layer_types": ["sliding_attention"],
        "mlp_layer_types": ["moe"],
        "compress_rates": {"compressed_sparse_attention": 4, "heavily_compressed_attention": 4},
        "sliding_window": 4,
        "swiglu_limit": 10.0,
        "max_position_embeddings": 64,
        "expert_dtype": "fp4",
        "quantization_config": {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128],
        },
    }


def _mxfp4_projection(torch, out: int, in_features: int):
    """One native MXFP4 expert projection: ``(I8 blocks, E8M0 scales, dense reference)``."""
    packed = _rng.integers(-128, 128, size=(out, in_features // 2), dtype=np.int8)
    exps = _rng.integers(120, 132, size=(out, in_features // 32), dtype=np.uint8)
    table = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
    view = packed.view(np.uint8)
    nibbles = np.stack((view & 0x0F, view >> 4), axis=-1).reshape(out, in_features)
    reference = table[nibbles] * np.repeat(np.ldexp(1.0, exps.astype(np.int32) - 127), 32, axis=-1)
    return torch.from_numpy(packed), torch.from_numpy(exps).view(torch.float8_e8m0fnu), reference


def _native_checkpoint(tmp_path, torch, hidden: int = 64, inter: int = 32, experts: int = 4):
    """Write a one-layer checkpoint in the published dialect; return the expert references."""
    from safetensors.torch import save_file

    config = _tiny_native_config(hidden, inter, experts)
    (tmp_path / "config.json").write_text(json.dumps(config))

    tensors: dict = {}
    references: dict = {}
    for e in range(experts):
        for proj, out, in_features in (("w1", inter, hidden), ("w3", inter, hidden), ("w2", hidden, inter)):
            blocks, scales, reference = _mxfp4_projection(torch, out, in_features)
            tensors[f"layers.0.ffn.experts.{e}.{proj}.weight"] = blocks
            tensors[f"layers.0.ffn.experts.{e}.{proj}.scale"] = scales
            references[(e, proj)] = reference
    # Enough of the dense trunk to prove the native names reach the twin's own parameters.
    # One fp8 trunk linear with its published ``.scale`` block sibling (E8M0, one per 128x128 tile).
    fp8 = (torch.randn(16, hidden) / 8).to(torch.float8_e4m3fn)  # q_a_proj is [q_lora_rank, hidden]
    block_scale = torch.tensor([[130]], dtype=torch.uint8).view(torch.float8_e8m0fnu)  # 2^3
    tensors["layers.0.attn.wq_a.weight"] = fp8
    tensors["layers.0.attn.wq_a.scale"] = block_scale
    references["q_a_proj"] = fp8.float() * 8.0
    tensors["layers.0.attn_norm.weight"] = torch.randn(hidden, dtype=torch.bfloat16)
    # The hyper-connection blocks carry a LEARNED parameter whose name also ends in "scale".
    tensors["layers.0.hc_attn_scale"] = torch.randn(3)
    tensors["layers.0.hc_ffn_scale"] = torch.randn(3)
    tensors["layers.0.ffn_norm.weight"] = torch.randn(hidden, dtype=torch.bfloat16)
    tensors["layers.0.ffn.gate.weight"] = torch.randn(experts, hidden, dtype=torch.bfloat16)
    tensors["layers.0.ffn.gate.bias"] = torch.randn(experts, dtype=torch.float32)
    # The speculative-decoding head ships in the same checkpoint and belongs to no twin.
    tensors["mtp.0.attn_norm.weight"] = torch.randn(hidden, dtype=torch.bfloat16)
    for e in range(experts):
        blocks, scales, _ = _mxfp4_projection(torch, inter, hidden)
        tensors[f"mtp.0.ffn.experts.{e}.w1.weight"] = blocks
        tensors[f"mtp.0.ffn.experts.{e}.w1.scale"] = scales
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return config, references


def test_native_routed_experts_stay_compressed_and_stacked(tmp_path):
    """Routed experts load as MXFP4 program inputs — never as a dense expert table."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from emmy.compiler.loader.quant import decode_mxfp4
    from emmy.compiler.trace.huggingface import load_quantized_split

    hidden, inter, experts = 64, 32, 4
    _config, references = _native_checkpoint(tmp_path, torch, hidden, inter, experts)
    model, store = load_quantized_split(tmp_path, torch.float16)

    assert store["fmt"] == "mxfp4"
    layer = store["layers"][0]
    # gate and up concatenate on the OUT axis, exactly as the merged expert wrapper consumes them.
    assert tuple(layer["w_gate_up"].shape) == (experts, 2 * inter, hidden // 32, 16)
    assert tuple(layer["w_gate_up_scale"].shape) == (experts, 2 * inter, hidden // 32)
    assert tuple(layer["w_down"].shape) == (experts, hidden, inter // 32, 16)
    assert tuple(layer["w_down_scale"].shape) == (experts, hidden, inter // 32)
    assert all(layer[name].dtype == torch.uint8 for name in ("w_gate_up", "w_gate_up_scale", "w_down", "w_down_scale"))
    # The twin's dense expert table is never materialized.
    assert model.model.layers[0].mlp.experts.gate_up_proj.is_meta

    # The stored bytes decode to the published values, per expert and per projection.
    for e in range(experts):
        gate_up = decode_mxfp4(layer["w_gate_up"][e].numpy(), layer["w_gate_up_scale"][e].numpy()).T
        np.testing.assert_array_equal(gate_up[:inter], references[(e, "w1")])
        np.testing.assert_array_equal(gate_up[inter:], references[(e, "w3")])
        down = decode_mxfp4(layer["w_down"][e].numpy(), layer["w_down_scale"][e].numpy()).T
        np.testing.assert_array_equal(down, references[(e, "w2")])


def test_native_trunk_names_reach_the_twin(tmp_path):
    """``attn_norm`` / ``ffn_norm`` / ``ffn.gate`` are the published spellings of the twin's own
    ``input_layernorm`` / ``post_attention_layernorm`` / ``mlp.gate`` parameters."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from emmy.compiler.trace.huggingface import load_quantized_split

    _config, _references = _native_checkpoint(tmp_path, torch)
    model, _store = load_quantized_split(tmp_path, torch.float16)

    state = model.state_dict()
    for key in (
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.mlp.gate.e_score_correction_bias",
    ):
        assert key in state and not state[key].is_meta, f"{key} did not load from the native checkpoint"


def test_expert_range_narrows_the_load_to_one_shard(tmp_path):
    """A tensor-parallel rank reads only its own experts: the others are never touched."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from emmy.compiler.loader.quant import decode_mxfp4
    from emmy.compiler.trace.huggingface import load_quantized_split

    hidden, inter, experts = 64, 32, 4
    _config, references = _native_checkpoint(tmp_path, torch, hidden, inter, experts)
    _model, store = load_quantized_split(tmp_path, torch.float16, expert_range=(2, 4))

    layer = store["layers"][0]
    assert tuple(layer["w_gate_up"].shape) == (2, 2 * inter, hidden // 32, 16)
    # Shard-local index 0 is global expert 2 — the shard keeps the checkpoint's own order.
    gate_up = decode_mxfp4(layer["w_gate_up"][0].numpy(), layer["w_gate_up_scale"][0].numpy()).T
    np.testing.assert_array_equal(gate_up[:inter], references[(2, "w1")])


def test_multi_token_prediction_head_is_never_read(tmp_path):
    """The MTP head is a separate speculative model — 4,608 of the real checkpoint's tensors. No
    twin instantiates it, so the loader must not read it into the trunk or the expert store."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from emmy.compiler.trace.huggingface import load_quantized_split

    _config, _references = _native_checkpoint(tmp_path, torch)
    model, store = load_quantized_split(tmp_path, torch.float16)

    assert set(store["layers"]) == {0}, "the MTP head leaked into the expert store"
    assert not any("mtp" in key for key in model.state_dict()), "the MTP head leaked onto the twin"


def test_native_fp8_trunk_weight_loads_scaled(tmp_path):
    """The published block scale is spelled ``.scale``, not ``<weight>_scale``. Pairing on the raw
    name silently drops it and loads every fp8 trunk weight off by its block scale."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from emmy.compiler.trace.huggingface import load_quantized_split

    _config, references = _native_checkpoint(tmp_path, torch, hidden=64, inter=32, experts=2)
    model, _store = load_quantized_split(tmp_path, torch.float32)

    loaded = model.state_dict()["model.layers.0.self_attn.q_a_proj.weight"]
    assert not loaded.is_meta
    torch.testing.assert_close(loaded, references["q_a_proj"], rtol=0, atol=0)


def test_learned_hyper_connection_scale_is_not_mistaken_for_a_block_scale(tmp_path):
    """``hc_attn_scale`` is an mHC parameter, not a quantization sibling. Treating every ``.scale``
    leaf as a block scale renames it out of the twin and leaves ``attn_hc.scale`` on meta — random
    stream-mixing weights in every layer, with nothing failing loudly."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from emmy.compiler.trace.huggingface import load_quantized_split

    _config, _references = _native_checkpoint(tmp_path, torch, hidden=64, inter=32, experts=2)
    model, _store = load_quantized_split(tmp_path, torch.float32)

    state = model.state_dict()
    for key in ("model.layers.0.attn_hc.scale", "model.layers.0.ffn_hc.scale"):
        assert key in state and not state[key].is_meta, f"{key} did not load from the native checkpoint"


def test_host_process_config_shadow_does_not_reach_the_twin(tmp_path):
    """A hosting vLLM process re-registers ``deepseek_v4`` onto its own rope-only config class
    (``AutoConfig.register(..., exist_ok=True)`` in its config parser), and from then on EVERY
    ``AutoConfig.from_pretrained`` in that process returns that class — none of the fields the real
    ``__init__`` derives (``layer_types`` from ``compress_ratios``), so the twin cannot be built.
    The loader must resolve the architecture's OWN config even inside such a process."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers import AutoConfig, PretrainedConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    from emmy.compiler.trace.huggingface import load_quantized_split

    _config, _references = _native_checkpoint(tmp_path, torch)

    # The host's class as vLLM ships it: SAME name as the native class (that sameness is what the
    # loader's recovery keys on), raw kwargs, no derivation.
    class DeepseekV4Config(PretrainedConfig):
        model_type = "deepseek_v4"

    AutoConfig.register("deepseek_v4", DeepseekV4Config, exist_ok=True)
    try:
        model, _store = load_quantized_split(tmp_path, torch.float16)
    finally:
        CONFIG_MAPPING._extra_content.pop("deepseek_v4", None)

    assert isinstance(model.config, transformers.DeepseekV4Config)
    assert model.config.layer_types == ["sliding_attention"]


def test_eager_layer_twin_materializes_experts_in_the_modules_own_orientation(tmp_path):
    """The eager reference decodes the same MXFP4 bytes the serving lane binds, but into a MODULE:
    ``decode_mxfp4`` hands back the ``(in, out)`` matrix gpt-oss applies as ``x @ W``, while these
    experts are ``F.linear`` parameters storing ``(out, in)``. ``w2`` catches a missing transpose on
    shape alone; ``w1``/``w3`` are square here, exactly the case that would load silently wrong."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from emmy.compiler.trace.huggingface import load_quantized_layer_twin

    hidden, inter, experts = 64, 32, 4
    _config, references = _native_checkpoint(tmp_path, torch, hidden, inter, experts)
    model = load_quantized_layer_twin(tmp_path, torch.float16, 0)

    module = model.model.layers[0].mlp.experts
    assert tuple(module.gate_up_proj.shape) == (experts, 2 * inter, hidden)
    assert tuple(module.down_proj.shape) == (experts, hidden, inter)
    for e in range(experts):
        gate_up = module.gate_up_proj[e].detach().float().numpy()
        np.testing.assert_array_equal(gate_up[:inter], references[(e, "w1")])
        np.testing.assert_array_equal(gate_up[inter:], references[(e, "w3")])
        np.testing.assert_array_equal(module.down_proj[e].detach().float().numpy(), references[(e, "w2")])
