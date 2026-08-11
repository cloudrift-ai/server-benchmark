"""Hermetic equivalence test for the Phase-1 attention-split carve (no GPU/vLLM).

Proves the carve is correct: for one decoder layer, running ``pre`` → reconstruct RoPE →
external causal GQA torch SDPA → ``post`` reproduces the eager ``block(x)`` over the
flattened ``[num_tokens, H]`` layout. Exercises GQA (num_kv_heads < num_heads), Qwen3's
per-head q/k norm, and Gemma-3/4's 4-norm decoder layer (a global layer, so the plain-causal
reference matches). Pure eager, CPU, fp32 — no compile.
"""

import pytest


def _repeat_kv(x, n_rep):
    """[1, Hkv, T, D] -> [1, Hkv*n_rep, T, D] (GQA head expansion)."""
    b, h, t, d = x.shape
    if n_rep == 1:
        return x
    return x[:, :, None, :, :].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


def _split_path_output(pre, post, attn, hidden2d, cos, sin, mask, apply_rotary):
    """Reference reconstruction: pre → rope → causal GQA SDPA → post. The [1,H,T,D] layout
    lives ONLY here (the carve's seam ABI is 2-D), per the plan."""
    import torch.nn.functional as F

    head_dim = attn.head_dim
    num_heads = attn.q_proj.out_features // head_dim
    num_kv = attn.k_proj.out_features // head_dim
    t = hidden2d.shape[0]

    q2d, k2d, v2d = pre(hidden2d)  # [T, Hq*D], [T, Hkv*D]
    # 2-D seam -> [1, n_heads, T, D] (no HF-style transpose hazard; explicit here).
    q = q2d.view(t, num_heads, head_dim).transpose(0, 1).unsqueeze(0)
    k = k2d.view(t, num_kv, head_dim).transpose(0, 1).unsqueeze(0)
    v = v2d.view(t, num_kv, head_dim).transpose(0, 1).unsqueeze(0)

    q, k = apply_rotary(q, k, cos, sin)  # reconstruct the RoPE the eager layer applies
    k = _repeat_kv(k, num_heads // num_kv)
    v = _repeat_kv(v, num_heads // num_kv)
    attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=attn.scaling)  # [1, Hq, T, D]
    attn_out = attn_out.transpose(1, 2).reshape(t, num_heads * head_dim)  # [T, Hq*D]
    return post(attn_out, hidden2d)


@pytest.mark.parametrize("arch", ["qwen3", "llama", "gemma3", "laguna"])
def test_split_matches_eager_block(arch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from emmy.compiler.trace.huggingface import build_attention_split_wrapper, build_causal_mask

    if arch == "qwen3":
        config = transformers.Qwen3Config(
            vocab_size=64,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=64,
            use_sliding_window=False,
        )
        model = transformers.Qwen3ForCausalLM(config)
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    elif arch == "llama":
        config = transformers.LlamaConfig(
            vocab_size=64,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
        model = transformers.LlamaForCausalLM(config)
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    elif arch == "gemma3":  # 4-norm layer + per-head q/k norm (no v_norm). sliding_window_pattern=1 forces
        # layer 0 to full_attention (global) so the plain-causal reference SDPA matches.
        config = transformers.Gemma3TextConfig(
            vocab_size=64,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=64,
            sliding_window=16,
            sliding_window_pattern=1,
        )
        model = transformers.Gemma3ForCausalLM(config)
        from transformers.models.gemma3.modeling_gemma3 import apply_rotary_pos_emb
    else:  # Laguna: per-head q/k norm plus a softplus per-head attention output gate.
        pytest.importorskip("transformers.models.laguna")
        config = transformers.LagunaConfig(
            vocab_size=64,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=64,
            sliding_window=16,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=32,
            shared_expert_intermediate_size=32,
            layer_types=["full_attention"],
            mlp_layer_types=["dense"],
            num_attention_heads_per_layer=[4],
            gating="per-head",
        )
        model = transformers.LagunaForCausalLM(config)
        from transformers.models.laguna.modeling_laguna import apply_rotary_pos_emb

    torch.manual_seed(0)
    model = model.eval()
    trunk = model.model
    block = trunk.layers[0]
    attn = block.self_attn

    t = 6
    hidden3d = torch.randn(1, t, config.hidden_size)
    position_ids = torch.arange(t).unsqueeze(0)
    # Gemma and Laguna rotary embeddings are keyed per layer-type; layer 0 here is global.
    rotary_kwargs = {"layer_type": "full_attention"} if arch in ("gemma3", "laguna") else {}
    cos, sin = trunk.rotary_emb(hidden3d, position_ids, **rotary_kwargs)  # [1, T, D]
    mask = build_causal_mask(t, torch.float32)  # [1, 1, T, T] additive

    with torch.no_grad():
        eager = block(hidden3d, position_embeddings=(cos, sin), attention_mask=mask)
        eager = eager[0] if isinstance(eager, tuple) else eager  # [1, T, H]

        pre, post = build_attention_split_wrapper(block)
        out = _split_path_output(pre, post, attn, hidden3d.squeeze(0), cos, sin, mask, apply_rotary_pos_emb)

    assert tuple(out.shape) == (t, config.hidden_size)
    torch.testing.assert_close(out, eager.squeeze(0), rtol=1e-4, atol=1e-4)


def test_split_matches_eager_gemma4():
    """Gemma-4 (the release target, ``Gemma4TextConfig``): 4-norm layer + per-head q/k/**v** norm
    + partial/proportional RoPE. ``hidden_size_per_layer_input=0`` gives the dense (12B-style) layer
    with no per-layer-input block; 1 layer is forced global, so a plain-causal reference matches.
    Exercises the gemma-4-specific ``v_norm`` carve path and gemma-4's per-tensor ``apply_rotary``."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("transformers.models.gemma4")

    import torch.nn.functional as F
    from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb

    from emmy.compiler.trace.huggingface import build_attention_split_wrapper, build_causal_mask

    config = transformers.Gemma4TextConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        sliding_window=16,
        hidden_size_per_layer_input=0,
    )
    torch.manual_seed(0)
    trunk = transformers.Gemma4ForCausalLM(config).eval().model
    block = trunk.layers[0]
    # Real checkpoints carry layer_scalar far from 1 (12B: 0.005–0.92); fresh models hold 1.0,
    # which silently masks a carve that drops the multiply — pin it off 1 to stay sensitive.
    block.layer_scalar.fill_(0.7)
    attn = block.self_attn

    t = 6
    hidden3d = torch.randn(1, t, config.hidden_size)
    position_ids = torch.arange(t).unsqueeze(0)
    cos, sin = trunk.rotary_emb(hidden3d, position_ids, "full_attention")  # layer 0 forced global
    mask = build_causal_mask(t, torch.float32)
    hd, nh, nkv = attn.head_dim, config.num_attention_heads, config.num_key_value_heads

    with torch.no_grad():
        eager = block(hidden3d, position_embeddings=(cos, sin), attention_mask=mask, shared_kv_states={})
        eager = eager[0] if isinstance(eager, tuple) else eager

        pre, post = build_attention_split_wrapper(block)
        q2, k2, v2 = pre(hidden3d.squeeze(0))
        q = q2.view(t, nh, hd).transpose(0, 1).unsqueeze(0)
        k = k2.view(t, nkv, hd).transpose(0, 1).unsqueeze(0)
        v = v2.view(t, nkv, hd).transpose(0, 1).unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, cos, sin), apply_rotary_pos_emb(k, cos, sin)  # per-tensor (gemma-4)
        k, v = _repeat_kv(k, nh // nkv), _repeat_kv(v, nh // nkv)
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=attn.scaling)
        attn_out = attn_out.transpose(1, 2).reshape(t, nh * hd)
        out = post(attn_out, hidden3d.squeeze(0))

    assert tuple(out.shape) == (t, config.hidden_size)
    torch.testing.assert_close(out, eager.squeeze(0), rtol=1e-4, atol=1e-4)


def test_pre_emits_2d_seam_shapes():
    """The pre wrapper's seam ABI: q[T, Hq*D], k/v[T, Hkv*D] (2-D, no transpose)."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from emmy.compiler.trace.huggingface import build_attention_split_wrapper

    config = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        use_sliding_window=False,
    )
    torch.manual_seed(0)
    block = transformers.Qwen3ForCausalLM(config).eval().model.layers[0]
    pre, _ = build_attention_split_wrapper(block)
    t = 5
    q, k, v = pre(torch.randn(t, config.hidden_size))
    assert tuple(q.shape) == (t, 4 * 16)  # [T, Hq*D]
    assert tuple(k.shape) == (t, 2 * 16)  # [T, Hkv*D]
    assert tuple(v.shape) == (t, 2 * 16)


def test_laguna_post_applies_softplus_attention_gate_before_o_proj():
    """Laguna gates every attention head from the normalized layer input.  The split ABI only
    carries the residual into ``post``, so ``post`` must reconstruct that normalized input and
    apply the softplus gate before the output projection."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers.models.laguna.modeling_laguna import LagunaDecoderLayer

    from emmy.compiler.trace.huggingface import build_attention_split_wrapper

    cfg = transformers.LagunaConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_attention_heads_per_layer=[4],
        num_key_value_heads=2,
        head_dim=16,
        layer_types=["full_attention"],
        mlp_layer_types=["dense"],
        max_position_embeddings=64,
        gating=True,
    )
    torch.manual_seed(0)
    block = LagunaDecoderLayer(cfg, layer_idx=0).eval()
    for parameter in block.parameters():
        torch.nn.init.normal_(parameter, std=0.2)
    # Transformers 5.12 did not expose this declaration. The checkpoint still
    # spells the layout unambiguously as one gate projection output per head.
    del block.self_attn.gate_per_head
    _, post = build_attention_split_wrapper(block)

    residual = torch.randn(5, cfg.hidden_size)
    attn_out = torch.randn(5, cfg.num_attention_heads * cfg.head_dim)
    with torch.no_grad():
        got = post(attn_out, residual)
        normalized = block.input_layernorm(residual)
        gate = torch.nn.functional.softplus(block.self_attn.g_proj(normalized).float()).to(attn_out.dtype)
        gated = (attn_out.view(5, cfg.num_attention_heads, cfg.head_dim) * gate.unsqueeze(-1)).view(5, -1)
        h = residual + block.self_attn.o_proj(gated)
        ref = h + block.mlp(block.post_attention_layernorm(h))

    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)
