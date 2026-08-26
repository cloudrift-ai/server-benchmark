"""`EmmyGenModel` hosting the 1Cat fork's attention sublayer (DeepSeek V4).

The first graded gate of the plugin work: the model class must CONSTRUCT under a real vLLM config —
one engine-owned attention sublayer per local decoder layer, each registering its own paged caches
under a distinct absolute-layer prefix, so vLLM can build KV-cache specs for them — and the pipeline
boundary must transport the seam's own carrier rather than one stream's worth of it.

Runs only where the fork is importable and the card is the one it was built for (the pinned 1Cat
image on sm_70); skipped everywhere else.
"""

from __future__ import annotations

import json

import pytest

pytestmark = [pytest.mark.perf, pytest.mark.xdist_group("cuda")]


def _tiny_deepseek(path, transformers, torch):
    """A DeepSeek V4 checkpoint small enough to construct in-process, saved in HF naming."""
    config = transformers.DeepseekV4Config(
        vocab_size=1024,
        hidden_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=32,
        qk_rope_head_dim=16,
        q_lora_rank=32,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        o_groups=2,
        o_lora_rank=32,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=8,
        hc_mult=2,
        hc_sinkhorn_iters=3,
        # Spelled the way the checkpoint publishes it: per-layer structure is DERIVED from
        # ``compress_ratios`` / ``num_hash_layers``. Passing explicit ``layer_types`` lists instead
        # marks them per-layer-ambiguous on a save/load round-trip, and transformers' own router
        # then cannot read ``num_local_experts`` off the config at all.
        compress_ratios=[0, 0],
        num_hash_layers=0,
        compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 4},
        sliding_window=32,
        swiglu_limit=10.0,
        max_position_embeddings=256,
    )
    torch.manual_seed(0)
    model = transformers.DeepseekV4ForCausalLM(config).eval().to(torch.float16)
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
    saved["compress_ratios"] = [0, 0]
    saved["num_hash_layers"] = 0
    # The fork's attention sublayer consumes the rope declaration at config-parse time
    # (``rope_parameters["rope_type"]``), so it must sit in the file, spelled the legacy way the
    # checkpoint publishes it. Scaled to this fixture: 64 original positions x factor 4 = 256 max.
    saved["rope_scaling"] = {"type": "yarn", "factor": 4, "beta_fast": 32, "beta_slow": 1, "original_max_position_embeddings": 64}
    config_path.write_text(json.dumps(saved))
    return config


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
        # ``quantization_config`` rides in as an override rather than in the file: the fork's
        # attention requires the published fp8 declaration (``scale_fmt``), but writing it into
        # config.json would flip the RUNNER's load onto the quantized lane — and this fixture's
        # weights are plain fp16. The real checkpoint declares it in the file and both sides read it.
        hf_overrides={
            "architectures": ["EmmyGenModel"],
            "quantization_config": {
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "quant_method": "fp8",
                "scale_fmt": "ue8m0",
                "weight_block_size": [128, 128],
            },
        },
    )
    config = args.create_engine_config()
    assert isinstance(config, VllmConfig)
    return config


def test_fork_attention_registers_one_cache_set_per_absolute_layer(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    vllm = pytest.importorskip("vllm")
    pytest.importorskip("cupy")
    pytest.importorskip("vllm.models.deepseek_v4.nvidia.model")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    import socket

    from vllm.config import set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    from emmy.serving.vllm_model_gen import EmmyGenModel

    config = _tiny_deepseek(tmp_path, transformers, torch)
    vllm_config = _vllm_config(vllm, tmp_path)

    # The model reads the parallel groups at construction (pipeline interval, expert-shard reduction),
    # so a bare in-process build needs the single-rank groups a worker would already have set up.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(world_size=1, rank=0, distributed_init_method=f"tcp://127.0.0.1:{port}", local_rank=0)
        initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        # vLLM's model loader constructs models under the target device; the fork's rotary cache
        # build relies on that (it mixes default-device tensors with explicitly-cuda ones).
        with torch.device("cuda"):
            model = EmmyGenModel(vllm_config=vllm_config, prefix="model")

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
