"""`EmmyGenModel` hosting the 1Cat fork's attention sublayer (DeepSeek V4).

The graded gates of the plugin work that run in one process on one card: the model class must
CONSTRUCT under a real vLLM config — one engine-owned attention sublayer per local decoder layer,
across all three attention layer types, each registering its own paged caches under a distinct
absolute-layer prefix so vLLM can build KV-cache specs for them — the pipeline boundary must
transport the seam's own carrier rather than one stream's worth of it, and the checkpoint's
attention weights must LOAD into the real fork modules through the ownership table.

Runs only where the fork is importable and the card is the one it was built for (the pinned 1Cat
image on sm_70); skipped everywhere else.
"""

from __future__ import annotations

import json
import math

import pytest

from tests.serving.generation.test_gen_fork_attention_load import CSA_KEYS, EXPECTED_DEST, HCA_KEYS, SLIDING_KEYS

pytestmark = [pytest.mark.perf, pytest.mark.xdist_group("cuda")]

# One layer of each attention type the published checkpoint mixes: two sliding, one
# compressed-sparse (with its lightning indexer), one heavily-compressed.
RATIOS = (0, 0, 4, 128)
KEYS_BY_TYPE = {
    "sliding_attention": SLIDING_KEYS,
    "compressed_sparse_attention": CSA_KEYS,
    "heavily_compressed_attention": HCA_KEYS,
}


def _tiny_deepseek(path, transformers, torch):
    """A DeepSeek V4 checkpoint small enough to construct in-process, saved in HF naming."""
    config = transformers.DeepseekV4Config(
        vocab_size=1024,
        hidden_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=len(RATIOS),
        num_attention_heads=4,
        # The three head dims stay REAL-sized: the fork's fused quant+cache kernels support only
        # the published geometry (compressor head_dim 512 with rope 64 -> 448 = 7x64 nope blocks,
        # indexer head 128). Everything else shrinks.
        head_dim=512,
        qk_rope_head_dim=64,
        q_lora_rank=128,  # every fp8 projection's group outputs must 128-align (SM70 grouped FP8)
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        o_groups=2,
        o_lora_rank=128,
        index_n_heads=2,
        index_head_dim=128,
        index_topk=8,
        hc_mult=2,
        hc_sinkhorn_iters=3,
        # Spelled the way the checkpoint publishes it: per-layer structure is DERIVED from
        # ``compress_ratios`` / ``num_hash_layers``. Passing explicit ``layer_types`` lists instead
        # marks them per-layer-ambiguous on a save/load round-trip, and transformers' own router
        # then cannot read ``num_local_experts`` off the config at all.
        compress_ratios=list(RATIOS),
        num_hash_layers=1,  # the first MoE layer routes by token id, as the published model's do
        # NO ``compress_rates``: the published config does not carry it, and the twin and the fork
        # derive their compressor geometry consistently only under the published spelling.
        sliding_window=32,
        swiglu_limit=10.0,
        max_position_embeddings=256,
    )
    torch.manual_seed(0)
    model = transformers.DeepseekV4ForCausalLM(config).eval().to(torch.float16)
    for layer in model.model.layers:
        # The hash layer's frozen token-id → expert table initializes to zeros (every token to
        # expert 0); randomize it so serving actually exercises the routing.
        if getattr(layer.mlp, "is_hash", False):
            layer.mlp.gate.tid2eid.copy_(torch.randint(0, config.n_routed_experts, layer.mlp.gate.tid2eid.shape))
    model.save_pretrained(path)
    # ``save_pretrained`` writes back the DERIVED per-layer lists, which the published checkpoint
    # does not carry; loading a config that has them marks those attributes ambiguous, and
    # transformers' own router can then no longer read ``num_local_experts`` at all. Save it the way
    # the checkpoint is actually published, so the test exercises the real load path.
    config_path = path / "config.json"
    saved = json.loads(config_path.read_text())
    for derived in ("layer_types", "mlp_layer_types"):
        saved.pop(derived, None)
    # ``compress_ratios`` / ``num_hash_layers`` are consumed as legacy kwargs and never serialized,
    # so write them back exactly as the checkpoint publishes them — they are what the config derives
    # its per-layer structure FROM.
    saved["compress_ratios"] = list(RATIOS)
    saved["num_hash_layers"] = 1
    # The fork's attention sublayer consumes the rope declaration at config-parse time
    # (``rope_parameters["rope_type"]``), so it must sit in the file, spelled the legacy way the
    # checkpoint publishes it. Scaled to this fixture: 64 original positions x factor 4 = 256 max.
    saved["rope_scaling"] = {"type": "yarn", "factor": 4, "beta_fast": 32, "beta_slow": 1, "original_max_position_embeddings": 64}
    config_path.write_text(json.dumps(saved))
    return config


# ``quantization_config`` rides in as an override rather than in the file: the fork's attention
# requires the published fp8 declaration (``scale_fmt``), but writing it into config.json would
# flip the RUNNER's load onto the quantized lane — and this fixture's weights are plain fp16. The
# real checkpoint declares it in the file and both sides read it, EXACTLY as published: a smaller
# block size sends vLLM's fp8 off the fork's block-quant path onto per-tensor kernels the card
# does not have.
HF_OVERRIDES = {
    "architectures": ["EmmyGenModel"],
    "quantization_config": {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "scale_fmt": "ue8m0",
        "weight_block_size": [128, 128],
    },
}


def _vllm_config(vllm, model_dir):
    from vllm.config import VllmConfig
    from vllm.engine.arg_utils import EngineArgs

    args = EngineArgs(
        model=str(model_dir),
        runner="generate",
        dtype="float16",
        max_model_len=128,
        max_num_batched_tokens=256,
        enforce_eager=True,
        kv_cache_dtype="fp8",  # the fork's paged attention supports no other cache format
        hf_overrides=HF_OVERRIDES,
    )
    config = args.create_engine_config()
    assert isinstance(config, VllmConfig)
    return config


def _build_model(tmp_path, torch, transformers, vllm):
    """The full in-process build a worker would do: engine config, single-rank parallel groups,
    model construction under the target device (vLLM's model loader constructs models there; the
    fork's rotary cache build relies on it)."""
    import socket

    from vllm.config import set_current_vllm_config
    from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment

    from emmy.serving.vllm_model_gen import EmmyGenModel

    config = _tiny_deepseek(tmp_path, transformers, torch)
    vllm_config = _vllm_config(vllm, tmp_path)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with set_current_vllm_config(vllm_config):
        # Idempotent forms: the second test in this process reuses the groups the first set up.
        init_distributed_environment(world_size=1, rank=0, distributed_init_method=f"tcp://127.0.0.1:{port}", local_rank=0)
        ensure_model_parallel_initialized(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        with torch.device("cuda"):
            model = EmmyGenModel(vllm_config=vllm_config, prefix="model")
    return config, vllm_config, model


def _requires_fork(torch):
    pytest.importorskip("transformers")
    pytest.importorskip("vllm")
    pytest.importorskip("cupy")
    pytest.importorskip("vllm.models.deepseek_v4.nvidia.model")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")


def test_fork_attention_registers_one_cache_set_per_absolute_layer(tmp_path):
    torch = pytest.importorskip("torch")
    _requires_fork(torch)
    import transformers
    import vllm

    config, vllm_config, model = _build_model(tmp_path, torch, transformers, vllm)

    # One engine-owned sublayer per local layer, and NO vLLM Attention / external RoPE: this
    # architecture has no q/k/v seam to place them in.
    assert model.fork_attn is not None and len(model.fork_attn) == config.num_hidden_layers
    assert len(model.attn) == 0 and len(model.rotary_emb) == 0

    # Every paged cache the fork registers must carry a distinct name, or vLLM's KV-cache spec
    # discovery collapses two layers onto one allocation.
    registered = sorted(vllm_config.compilation_config.static_forward_context)
    assert len(registered) == len(set(registered)), f"duplicate cache prefixes: {registered}"
    for layer in range(config.num_hidden_layers):
        assert any(f".layers.{layer}.attn" in name for name in registered), f"layer {layer} registered no attention cache"

    # The pipeline boundary transports the seam's carrier, not one stream's worth of it.
    carrier = model.runner.carrier_size
    assert carrier == config.hc_mult * config.hidden_size
    empty = model.make_empty_intermediate_tensors(batch_size=3, dtype=torch.float16, device=torch.device("cuda"))
    assert tuple(empty["hidden_states"].shape) == (3, carrier)


def _native_stream(torch, config, model, fills):
    """The checkpoint's attention keys for every layer, shaped off the REAL fork modules' own
    geometry (fused shards split by the merged projection's ``output_sizes``), plus the head."""
    block = HF_OVERRIDES["quantization_config"]["weight_block_size"][0]
    items = [("head.weight", torch.zeros(config.vocab_size, config.hidden_size, dtype=torch.float16))]
    for layer, layer_type in enumerate(config.layer_types):
        params = dict(model.fork_attn[layer].named_parameters())
        for key in KEYS_BY_TYPE[layer_type]:
            dest, shard = EXPECTED_DEST[key]
            param = params[dest]
            fill = fills(key)
            if dest == "attn_sink":
                tensor = torch.full((config.num_attention_heads,), fill, dtype=param.dtype)
            elif shard is None:
                shape = param.shape
                tensor = torch.full(shape, fill, dtype=torch.float32).to(param.dtype)
            else:
                module = model.fork_attn[layer].get_submodule(dest.rsplit(".", 1)[0])
                rows = module.output_sizes[shard]
                if dest.endswith(".weight_scale_inv"):
                    shape = (math.ceil(rows / block), param.shape[1])
                else:
                    shape = (rows, param.shape[1])
                tensor = torch.full(shape, fill, dtype=torch.float32).to(param.dtype)
            if key.endswith(".scale"):
                # The checkpoint stores fp8 block scales as E8M0 exponent bytes; 127 decodes to 1.0.
                tensor = torch.full(tensor.shape, 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
            items.append((f"layers.{layer}.attn.{key}", tensor))
    return items


def test_checkpoint_attention_weights_load_into_the_fork_sublayer(tmp_path):
    """The ownership table against the REAL fork modules: every published attention key loads,
    the fused shards land in order, and the loud completeness check passes on a full stream."""
    torch = pytest.importorskip("torch")
    _requires_fork(torch)
    import transformers
    import vllm

    config, _vllm_cfg, model = _build_model(tmp_path, torch, transformers, vllm)
    fills = lambda key: {"wq_a.weight": 0.5, "wkv.weight": 1.5}.get(key, 1.0)  # noqa: E731

    loaded = model.load_weights(iter(_native_stream(torch, config, model, fills)))

    # Every parameter of every fork attention module was claimed — the completeness check inside
    # ``load_weights`` already enforces it; this pins the returned names vLLM's tracker sees.
    for layer, module in enumerate(model.fork_attn):
        for name, _ in module.named_parameters():
            assert f"fork_attn.{layer}.{name}" in loaded, f"layer {layer} parameter {name} unclaimed"
    assert "lm_head.weight" in loaded

    # The fused projection keeps the checkpoint's order: wq_a rows first, wkv rows after.
    fused = model.fork_attn[0].fused_wqa_wkv
    rows = fused.output_sizes[0]
    assert torch.all(fused.weight[:rows].float() == 0.5)
    assert torch.all(fused.weight[rows:].float() == 1.5)
    # The E8M0 block scales decoded to real multipliers, and the sink logits copied through.
    assert torch.all(fused.weight_scale_inv.float() == 1.0)
    # The fork sizes ``attn_sink`` for its padded head count and keeps the padding at -inf;
    # the checkpoint's values land in the real-head prefix, exactly as the fork's own loader copies.
    sink = dict(model.fork_attn[0].named_parameters())["attn_sink"]
    assert torch.all(sink.float()[: config.num_attention_heads] == 1.0)
