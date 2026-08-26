"""End-to-end DeepSeek V4 seam on a real GPU: compiled programs must reproduce the eager layer.

Everything below the fork's attention runs through Emmy here — the hyper-connection stream collapse
and input norm (`pre`), the stream mixing, feed-forward collapse, norm and shared expert (`post`),
and the routed experts as compiled programs fed per-expert weight slices — with the block's own
attention sublayer standing in for the paged MLA attention the fork owns in production. The eager
`DeepseekV4DecoderLayer` is the oracle.

Also pins the two invariants the plugin depends on: the carrier crossing the seam is the FLATTENED
stream stack (`hc_mult * hidden`, not `hidden`), and a tensor-parallel expert shard's partials sum
to the unsharded result.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.xdist_group("cuda")]


def _tiny_config(transformers, mlp="moe"):
    return transformers.DeepseekV4Config(
        vocab_size=64,
        hidden_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=1,
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
        index_topk=2,
        hc_mult=2,
        hc_sinkhorn_iters=3,
        layer_types=["sliding_attention"],
        mlp_layer_types=[mlp],
        compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 4},
        sliding_window=16,
        swiglu_limit=10.0,
        routed_scaling_factor=1.5,
        max_position_embeddings=64,
    )


def _model(transformers, torch, mlp="moe"):
    torch.manual_seed(0)
    config = _tiny_config(transformers, mlp)
    model = transformers.DeepseekV4ForCausalLM(config).to(torch.float16).eval()
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.02)
    layer = model.model.layers[0]
    torch.nn.init.uniform_(layer.attn_hc.scale, 0.5, 2.0)
    torch.nn.init.uniform_(layer.ffn_hc.scale, 0.5, 2.0)
    layer.mlp.gate.weight.data.mul_(20.0)  # spread the router so the tokens reach distinct experts
    if getattr(layer.mlp, "is_hash", False):
        layer.mlp.gate.tid2eid.copy_(torch.randint(0, config.n_routed_experts, layer.mlp.gate.tid2eid.shape))
    return config, model


def _eager_reference(torch, model, streams, ids, attn_kwargs):
    with torch.no_grad():
        return model.model.layers[0](streams, input_ids=ids, **attn_kwargs)


def test_deepseek_seam_matches_eager_on_gpu():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("cupy")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from emmy.serving.gen_runner import EmmyGenRunner

    config, model = _model(transformers, torch)
    tokens, hidden, hc = 8, config.hidden_size, config.hc_mult
    runner = EmmyGenRunner.from_model(model, dtype_str="float16", decode_bucket=tokens, max_tokens=None)

    assert runner.carrier_size == hc * hidden, "the seam must carry the flattened stream stack"
    assert runner.hc_mult == hc

    ids = torch.randint(0, config.vocab_size, (1, tokens))
    position_ids = torch.arange(tokens).unsqueeze(0)
    embeds = model.model.embed_tokens(ids)
    attn_kwargs = {
        "position_embeddings": {
            kind: model.model.rotary_emb(embeds, position_ids=position_ids, layer_type=kind) for kind in ("main", "compress")
        },
        "position_ids": position_ids,
        "attention_mask": torch.triu(torch.full((tokens, tokens), float("-inf"), dtype=torch.float16), diagonal=1)[None, None],
    }
    streams = (torch.randn(1, tokens, hc, hidden) * 0.1).to(torch.float16)
    reference = _eager_reference(torch, model, streams, ids, attn_kwargs).reshape(tokens, hc * hidden)

    carrier = streams.reshape(tokens, hc * hidden).cuda()
    attention = model.model.layers[0].self_attn.cuda()
    with torch.no_grad():
        x = runner.forward_layer_pre_device(0, carrier)
        x = x[0] if isinstance(x, tuple) else x
        attn_out, _ = attention(x.unsqueeze(0), **_cuda_kwargs(torch, attn_kwargs))
        got = runner.forward_layer_post_device(0, attn_out.squeeze(0), carrier)

    np.testing.assert_allclose(got.float().cpu().numpy(), reference.float().numpy(), rtol=2e-2, atol=2e-2)


def test_sharded_expert_partials_sum_to_the_unsharded_combine_on_gpu():
    """The compiled expert programs, run per shard, sum to the single-rank result the all-reduce
    is standing in for — the invariant tensor-parallel serving rests on."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("cupy")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from emmy.serving.gen_runner import EmmyGenRunner

    config, model = _model(transformers, torch)
    runner = EmmyGenRunner.from_model(model, dtype_str="float16", decode_bucket=4, max_tokens=None)
    moe = next(m for m in runner._moe if m is not None)

    xn = (torch.randn(4, config.hidden_size) * 0.1).to(torch.float16).cuda()
    whole = runner._moe_combine(moe, xn)

    experts = config.n_routed_experts
    total = torch.zeros_like(whole)
    for lo in range(0, experts, experts // 2):
        shard = dict(moe, expert_range=(lo, lo + experts // 2))
        shard["inputs"] = {
            name: (tensor[lo : lo + experts // 2].contiguous() if name.startswith(("w_", "b_")) else tensor)
            for name, tensor in moe["inputs"].items()
        }
        total += runner._moe_combine(shard, xn)

    torch.testing.assert_close(total.float(), whole.float(), rtol=2e-2, atol=2e-2)


def _cuda_kwargs(torch, attn_kwargs):
    moved = dict(attn_kwargs)
    moved["position_embeddings"] = {k: tuple(t.cuda() for t in v) for k, v in attn_kwargs["position_embeddings"].items()}
    moved["position_ids"] = attn_kwargs["position_ids"].cuda()
    moved["attention_mask"] = attn_kwargs["attention_mask"].cuda()
    return moved


def test_embed_opens_and_final_norm_closes_the_stream_carrier():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("cupy")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from emmy.serving.gen_runner import EmmyGenRunner

    config, model = _model(transformers, torch)
    hidden, hc = config.hidden_size, config.hc_mult
    runner = EmmyGenRunner.from_model(model, dtype_str="float16", decode_bucket=4, max_tokens=None)

    ids = torch.randint(0, config.vocab_size, (5,)).cuda()
    opened = runner.embed_device(ids)
    assert tuple(opened.shape) == (5, hc * hidden), "the embedding must open into every residual stream"
    # Each stream starts as a copy of the same row, which is what the model's own expand does.
    per_stream = opened.view(5, hc, hidden)
    for stream in range(1, hc):
        torch.testing.assert_close(per_stream[:, 0], per_stream[:, stream])

    closed = runner.final_norm_device(opened)
    assert tuple(closed.shape) == (5, hidden), "the final norm must see the collapsed hidden width"
    with torch.no_grad():
        # The reference runs the model's OWN collapse and norm; move them beside the device tensor
        # (the runner holds its own deep copies, so this cannot mask a missing move on its side).
        expected = model.model.norm.cuda()(model.model.hc_head.cuda()(per_stream.unsqueeze(0)).squeeze(0))
    torch.testing.assert_close(closed.float(), expected.float(), rtol=2e-2, atol=2e-2)


def test_hash_routed_layer_selects_experts_by_token_id_on_gpu():
    """The serving path of a hash-MoE layer: the routed combine receives the step's token ids
    (the frozen ``tid2eid`` table selects the experts; the learned gate only weights them) and
    reproduces the eager layer — and refuses to route at all when the ids are missing."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("cupy")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from emmy.serving.gen_runner import EmmyGenRunner

    config, model = _model(transformers, torch, mlp="hash_moe")
    tokens, hidden, hc = 8, config.hidden_size, config.hc_mult
    runner = EmmyGenRunner.from_model(model, dtype_str="float16", decode_bucket=tokens, max_tokens=None)
    assert runner._moe[0]["hash"], "the hash router was not recognized off its tid2eid table"

    ids = torch.randint(0, config.vocab_size, (1, tokens))
    position_ids = torch.arange(tokens).unsqueeze(0)
    embeds = model.model.embed_tokens(ids)
    attn_kwargs = {
        "position_embeddings": {
            kind: model.model.rotary_emb(embeds, position_ids=position_ids, layer_type=kind) for kind in ("main", "compress")
        },
        "position_ids": position_ids,
        "attention_mask": torch.triu(torch.full((tokens, tokens), float("-inf"), dtype=torch.float16), diagonal=1)[None, None],
    }
    streams = (torch.randn(1, tokens, hc, hidden) * 0.1).to(torch.float16)
    reference = _eager_reference(torch, model, streams, ids, attn_kwargs).reshape(tokens, hc * hidden)

    carrier = streams.reshape(tokens, hc * hidden).cuda()
    attention = model.model.layers[0].self_attn.cuda()
    with torch.no_grad():
        x = runner.forward_layer_pre_device(0, carrier)
        x = x[0] if isinstance(x, tuple) else x
        attn_out, _ = attention(x.unsqueeze(0), **_cuda_kwargs(torch, attn_kwargs))
        with pytest.raises(RuntimeError, match="hash-routed"):
            runner.forward_layer_post_device(0, attn_out.squeeze(0), carrier)
        got = runner.forward_layer_post_device(0, attn_out.squeeze(0), carrier, token_ids=ids.reshape(-1).cuda())

    np.testing.assert_allclose(got.float().cpu().numpy(), reference.float().numpy(), rtol=2e-2, atol=2e-2)
