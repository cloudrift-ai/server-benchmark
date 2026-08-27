"""Hermetic equivalence tests for the DeepSeek V4 hyper-connection seam (no GPU/vLLM).

Proves the carve is correct on a tiny random DeepSeek V4 layer: ``pre`` (hyper-connection collapse
+ input norm) → the block's own attention sublayer standing in for the 1Cat fork's paged MLA
attention → ``post`` (attention-site stream mixing, feed-forward collapse + norm, shared expert,
feed-forward stream mixing) plus the runner-style torch half (HF router → per-expert
``expert(x, w_gate_up, w_down)`` → weighted ``index_add_`` → per-stream placement) reproduces the
eager ``DeepseekV4DecoderLayer`` output over the flattened ``[num_tokens, hc_mult * hidden]`` carrier.
Pure eager, CPU, fp32 — no compile.
"""

import pytest


def _tiny_config(transformers):
    return transformers.DeepseekV4Config(
        vocab_size=64,
        hidden_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=3,
        num_attention_heads=4,
        head_dim=8,
        qk_rope_head_dim=4,
        q_lora_rank=16,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        o_groups=2,
        o_lora_rank=8,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        hc_mult=2,
        hc_sinkhorn_iters=3,
        layer_types=["sliding_attention", "heavily_compressed_attention", "compressed_sparse_attention"],
        mlp_layer_types=["hash_moe", "moe", "moe"],
        compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 4},
        sliding_window=16,
        swiglu_limit=10.0,
        routed_scaling_factor=1.5,
        max_position_embeddings=64,
    )


def _random_model(transformers, torch):
    torch.manual_seed(0)
    cfg = _tiny_config(transformers)
    model = transformers.DeepseekV4ForCausalLM(cfg).eval()
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.2)
    for layer in model.model.layers:
        # Stream-mixing scales far from their ``ones`` init keep the test sensitive to every term.
        torch.nn.init.uniform_(layer.attn_hc.scale, 0.5, 2.0)
        torch.nn.init.uniform_(layer.ffn_hc.scale, 0.5, 2.0)
        layer.mlp.gate.weight.data.mul_(5.0)  # spread the router so every token picks distinct experts
        if layer.mlp.is_hash:
            layer.mlp.gate.tid2eid.copy_(torch.randint(0, cfg.n_routed_experts, layer.mlp.gate.tid2eid.shape))
    return cfg, model


@pytest.mark.parametrize("layer_idx", [0, 1, 2])
def test_hyper_connection_split_matches_eager_layer(layer_idx):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from emmy.compiler.trace.huggingface import build_moe_split_wrapper, hyper_connection_seam, moe_block_parts, place_routed_streams
    from emmy.serving.gen_runner import combine_routed_experts

    cfg, model = _random_model(transformers, torch)
    block = model.model.layers[layer_idx]
    t, h, hc = 8, cfg.hidden_size, cfg.hc_mult
    assert hyper_connection_seam(block) == (hc * h, h)

    pre, post, expert = build_moe_split_wrapper(block)
    gate, experts = moe_block_parts(block.mlp)

    ids = torch.randint(0, cfg.vocab_size, (1, t))
    position_ids = torch.arange(t).unsqueeze(0)
    embeds = model.model.embed_tokens(ids)
    position_embeddings = {
        kind: model.model.rotary_emb(embeds, position_ids=position_ids, layer_type=kind) for kind in ("main", "compress")
    }
    mask = torch.triu(torch.full((t, t), float("-inf")), diagonal=1)[None, None]
    streams = torch.randn(1, t, hc, h)
    attn_kwargs = dict(position_embeddings=position_embeddings, position_ids=position_ids, attention_mask=mask)

    with torch.no_grad():
        ref = block(streams, input_ids=ids, **attn_kwargs).reshape(t, hc * h)

        carrier = streams.reshape(t, hc * h)
        x = pre(carrier)
        assert x.shape == (t, h)
        attn_out, _ = block.self_attn(x.unsqueeze(0), **attn_kwargs)  # the fork's paged MLA attention stands here
        mixed, xn, mix = post(attn_out.squeeze(0), carrier)
        assert mixed.shape == (t, hc * h) and xn.shape == (t, h) and mix.shape == (t, hc)
        gated = gate(xn, ids.reshape(-1)) if block.mlp.is_hash else gate(xn)
        routed = combine_routed_experts(xn, gated, lambda e, rows: expert(rows, experts.gate_up_proj[e], experts.down_proj[e]))
        got = place_routed_streams(mixed, routed, mix)
    assert torch.allclose(ref, got, atol=1e-5)


def test_expert_wrapper_clamps_both_swiglu_branches():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from emmy.compiler.trace.huggingface import build_moe_split_wrapper

    _cfg, model = _random_model(transformers, torch)
    block = model.model.layers[1]
    _pre, _post, expert = build_moe_split_wrapper(block)
    x = torch.ones(1, 2)
    w_gate_up = torch.tensor([[50.0, 0.0], [-50.0, 0.0], [50.0, 0.0], [-50.0, 0.0]])
    w_down = torch.eye(2)
    want = torch.nn.functional.silu(torch.tensor([[10.0, -50.0]])) * torch.tensor([[10.0, -10.0]])
    with torch.no_grad():
        torch.testing.assert_close(expert(x, w_gate_up, w_down), want)


def test_classic_carve_rejects_hyper_connection_block():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from emmy.compiler.trace.huggingface import build_attention_split_wrapper

    _cfg, model = _random_model(transformers, torch)
    with pytest.raises(NotImplementedError, match="hyper-connection"):
        build_attention_split_wrapper(model.model.layers[0])
