"""Hermetic equivalence tests for the MoE third-seam carve (no GPU/vLLM).

Proves the carve is correct on a tiny random OLMoE layer: ``post_attn`` (o_proj + residual +
post-attention norm, both ``h`` and ``xn`` out) plus the runner-style torch half (HF router
module → per-expert ``expert(x, w_gate_up, w_down)`` launches → weighted ``index_add_``)
reproduces the eager block tail exactly. Pure eager, CPU, fp32 — no compile.
"""

import pytest


def _tiny_olmoe_config(transformers):
    return transformers.OlmoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=8,
        num_experts_per_tok=2,
        norm_topk_prob=False,
        max_position_embeddings=64,
    )


def _combine(gate, experts, expert, xn):
    """The runner's torch half with the eager expert wrapper in place of the compiled program —
    the routing math itself is the SHARED ``combine_routed_experts`` serving runs."""
    from emmy.serving.gen_runner import combine_routed_experts

    return combine_routed_experts(xn, gate(xn), lambda e, rows: expert(rows, experts.gate_up_proj[e], experts.down_proj[e]))


def test_moe_split_matches_eager_block_tail():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers.models.olmoe.modeling_olmoe import OlmoeDecoderLayer

    from emmy.compiler.trace.huggingface import build_moe_split_wrapper, moe_block_parts

    torch.manual_seed(0)
    cfg = _tiny_olmoe_config(transformers)
    block = OlmoeDecoderLayer(cfg, layer_idx=0).eval()
    for p in block.parameters():
        torch.nn.init.normal_(p, std=0.2)

    _, post_attn, expert = build_moe_split_wrapper(block)
    gate, experts = moe_block_parts(block.mlp)

    t = 5
    attn_width = cfg.num_attention_heads * (cfg.hidden_size // cfg.num_attention_heads)
    attn_out = torch.randn(t, attn_width)
    residual = torch.randn(t, cfg.hidden_size)
    with torch.no_grad():
        h, xn = post_attn(attn_out, residual)
        got = h + _combine(gate, experts, expert, xn)
        h_ref = residual + block.self_attn.o_proj(attn_out)
        ref = h_ref + block.mlp(block.post_attention_layernorm(h_ref).unsqueeze(0)).squeeze(0)
    assert torch.allclose(ref, got, atol=1e-5)


def test_moe_expert_shares_one_shape_across_experts():
    """Every expert must be servable by ONE wrapper: the expert forward takes the weights as
    forward ARGUMENTS (they trace as inputs), and dim-0 slices of the 3-D expert tensors all
    have the same shape."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers.models.olmoe.modeling_olmoe import OlmoeDecoderLayer

    from emmy.compiler.trace.huggingface import build_moe_split_wrapper, moe_block_parts

    torch.manual_seed(0)
    cfg = _tiny_olmoe_config(transformers)
    block = OlmoeDecoderLayer(cfg, layer_idx=0).eval()
    for p in block.parameters():  # bare-layer params are torch.empty — init or the assert below is nondeterministic
        torch.nn.init.normal_(p, std=0.2)
    _, _, expert = build_moe_split_wrapper(block)
    _, experts = moe_block_parts(block.mlp)

    assert len(list(expert.parameters())) == 0, "expert wrapper must carry NO parameters (weights are inputs)"
    x = torch.randn(3, cfg.hidden_size)
    with torch.no_grad():
        y0 = expert(x, experts.gate_up_proj[0], experts.down_proj[0])
        y1 = expert(x, experts.gate_up_proj[1], experts.down_proj[1])
    assert y0.shape == y1.shape == x.shape
    assert not torch.allclose(y0, y1), "different experts' weights must produce different outputs"


def test_moe_split_folds_shared_experts_into_post_attn():
    """DeepSeek/GLM lineage blocks carry an always-on ``shared_experts`` MLP beside the routed
    ones; the carve folds it into ``post_attn``'s returned ``h`` (over the same normed ``xn``),
    so the runner's route-and-combine half needs no seam change."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers.models.olmoe.modeling_olmoe import OlmoeDecoderLayer

    from emmy.compiler.trace.huggingface import build_moe_split_wrapper

    torch.manual_seed(0)
    block = OlmoeDecoderLayer(_tiny_olmoe_config(transformers), layer_idx=0).eval()
    block.mlp.shared_experts = torch.nn.Linear(64, 64)
    _, post_attn, _ = build_moe_split_wrapper(block)
    attn_out = torch.randn(5, 64)
    residual = torch.randn(5, 64)
    with torch.no_grad():
        h, xn = post_attn(attn_out, residual)
        h_ref = residual + block.self_attn.o_proj(attn_out)
        xn_ref = block.post_attention_layernorm(h_ref)
    assert torch.allclose(xn, xn_ref, atol=1e-6)
    assert torch.allclose(h, h_ref + block.mlp.shared_experts(xn_ref), atol=1e-6)


def test_moe_split_rejects_gated_shared_expert():
    """Qwen-MoE's shared expert is GATED (``shared_expert_gate``); the ungated fold would
    silently drop the gate, so it must reject LOUDLY."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers.models.olmoe.modeling_olmoe import OlmoeDecoderLayer

    from emmy.compiler.trace.huggingface import build_moe_split_wrapper

    torch.manual_seed(0)
    block = OlmoeDecoderLayer(_tiny_olmoe_config(transformers), layer_idx=0).eval()
    block.mlp.shared_expert = torch.nn.Linear(64, 64)
    block.mlp.shared_expert_gate = torch.nn.Linear(64, 1)
    with pytest.raises(NotImplementedError, match="shared_expert_gate"):
        build_moe_split_wrapper(block)


def _tiny_glm4_moe_config(transformers):
    return transformers.Glm4MoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        first_k_dense_replace=1,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        use_qk_norm=False,
        partial_rotary_factor=0.5,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
    )


def test_glm4_moe_split_matches_eager_block_tail():
    """GLM-4.5-Air's block layout (Glm4Moe: 3-tuple router, packed v5 experts, always-on
    shared expert) through the carve: ``post_attn`` (shared experts folded in) + the runner's
    route-and-combine reproduces the eager MoE block tail."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeDecoderLayer

    from emmy.compiler.trace.huggingface import build_moe_split_wrapper, moe_block_parts

    torch.manual_seed(0)
    cfg = _tiny_glm4_moe_config(transformers)
    block = Glm4MoeDecoderLayer(cfg, layer_idx=1).eval()  # layer 1 = the MoE form
    for p in block.parameters():
        torch.nn.init.normal_(p, std=0.2)

    _, post_attn, expert = build_moe_split_wrapper(block)
    gate, experts = moe_block_parts(block.mlp)
    assert gate is block.mlp.gate  # the 3-tuple Glm4MoeTopkRouter; combine reads its LAST two entries

    t = 5
    attn_width = cfg.num_attention_heads * cfg.head_dim
    attn_out = torch.randn(t, attn_width)
    residual = torch.randn(t, cfg.hidden_size)
    with torch.no_grad():
        h, xn = post_attn(attn_out, residual)
        got = h + _combine(gate, experts, expert, xn)
        h_ref = residual + block.self_attn.o_proj(attn_out)
        ref = h_ref + block.mlp(block.post_attention_layernorm(h_ref).unsqueeze(0)).squeeze(0)
    assert torch.allclose(ref, got, atol=1e-5)


def test_combine_casts_fp32_router_scores():
    """Mixtral-family routers return fp32 scores; the combine must cast them to the activation
    dtype or ``index_add_`` crashes under the forced-fp16 serving lane."""
    torch = pytest.importorskip("torch")

    from emmy.serving.gen_runner import combine_routed_experts

    torch.manual_seed(0)
    xn = torch.randn(5, 8, dtype=torch.float16)
    scores = torch.rand(5, 2, dtype=torch.float32)
    indices = torch.randint(0, 4, (5, 2))
    out = combine_routed_experts(xn, (None, scores, indices), lambda e, rows: rows * (e + 1))
    assert out.dtype == torch.float16
    ref = torch.zeros_like(xn)
    for t in range(5):
        for j in range(2):
            ref[t] += (xn[t] * (indices[t, j].item() + 1)) * scores[t, j].to(torch.float16)
    assert torch.allclose(out, ref, atol=1e-2)


def test_moe_block_parts_rejects_dense_mlp():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from emmy.compiler.trace.huggingface import moe_block_parts

    config = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
    )
    torch.manual_seed(0)
    model = transformers.Qwen3ForCausalLM(config).eval()
    assert moe_block_parts(model.model.layers[0].mlp) is None


def test_flat_qk_norm_pre_matches_eager_projection():
    """OLMoE normalizes the FLAT q/k projections BEFORE the head reshape (norm width == the
    projection width, not head_dim) — the pre carve must reproduce that placement."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers.models.olmoe.modeling_olmoe import OlmoeDecoderLayer

    from emmy.compiler.trace.huggingface import build_moe_split_wrapper

    torch.manual_seed(0)
    cfg = _tiny_olmoe_config(transformers)
    block = OlmoeDecoderLayer(cfg, layer_idx=0).eval()
    for p in block.parameters():
        torch.nn.init.normal_(p, std=0.2)
    pre, _, _ = build_moe_split_wrapper(block)

    t, hd = 5, block.self_attn.head_dim
    nh = block.self_attn.q_proj.out_features // hd
    hidden = torch.randn(t, cfg.hidden_size)
    with torch.no_grad():
        q2, k2, _ = pre(hidden)
        hn = block.input_layernorm(hidden)
        q_ref = block.self_attn.q_norm(block.self_attn.q_proj(hn))
        k_ref = block.self_attn.k_norm(block.self_attn.k_proj(hn))
    assert torch.allclose(q2, q_ref, atol=1e-6)
    assert torch.allclose(k2, k_ref, atol=1e-6)
    assert q2.shape == (t, nh * hd)


def _tiny_gptoss_block(transformers, torch):
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssDecoderLayer

    cfg = transformers.GptOssConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        sliding_window=8,
    )
    torch.manual_seed(0)
    block = GptOssDecoderLayer(cfg, layer_idx=0).eval()
    for p in block.parameters():
        torch.nn.init.normal_(p, std=0.2)
    return cfg, block


def test_gptoss_expert_layout_detected_from_interface_not_shapes():
    """gpt-oss layout comes off the transformers experts-interface attributes (is_transposed /
    is_concatenated / has_bias), never shape sniffing — its down_proj is square, so shapes
    cannot disambiguate the orientation."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from emmy.compiler.trace.huggingface import moe_block_parts, moe_expert_layout

    cfg, block = _tiny_gptoss_block(transformers, torch)
    parts = moe_block_parts(block.mlp)
    assert parts is not None, "the mlp.router (gpt-oss) spelling must be accepted beside mlp.gate"
    router, experts = parts
    assert router is block.mlp.router
    assert moe_expert_layout(experts) == (True, True, True)  # (transposed, interleaved, has_bias)


def test_gptoss_moe_split_matches_eager_block_tail():
    """The gpt-oss carve — clamped-SwiGLU + interleaved gate/up (de-interleaved at load) +
    per-expert biases + the ``router`` module — reproduces the eager MoE block tail."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from emmy.compiler.trace.huggingface import build_moe_split_wrapper, deinterleave_gate_up, moe_block_parts
    from emmy.serving.gen_runner import combine_routed_experts

    cfg, block = _tiny_gptoss_block(transformers, torch)
    pre, post_attn, expert = build_moe_split_wrapper(block)
    router, experts = moe_block_parts(block.mlp)
    assert len(list(expert.parameters())) == 0, "expert wrapper must carry NO parameters (weights are inputs)"

    # The load-time de-interleave: gate/up columns to chunk halves (weights AND biases).
    w_gu = deinterleave_gate_up(experts.gate_up_proj.detach())
    b_gu = deinterleave_gate_up(experts.gate_up_proj_bias.detach())

    t = 5
    attn_width = cfg.num_attention_heads * cfg.head_dim
    attn_out = torch.randn(t, attn_width)
    residual = torch.randn(t, cfg.hidden_size)
    with torch.no_grad():
        h, xn = post_attn(attn_out, residual)
        combined = combine_routed_experts(
            xn,
            router(xn),
            lambda e, rows: expert(rows, w_gu[e], experts.down_proj[e], b_gu[e], experts.down_proj_bias[e]),
        )
        got = h + combined
        h_ref = residual + block.self_attn.o_proj(attn_out)
        ref = h_ref + block.mlp(block.post_attention_layernorm(h_ref).unsqueeze(0))[0].squeeze(0)
    assert torch.allclose(ref, got, atol=1e-5), f"max diff {(ref - got).abs().max()}"


def test_gptoss_pre_carries_attention_biases():
    """gpt-oss attention projections are biased (attention_bias=True); the pre carve must
    reproduce the biased projections at head_dim 64-style explicit dims."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from emmy.compiler.trace.huggingface import build_moe_split_wrapper

    cfg, block = _tiny_gptoss_block(transformers, torch)
    assert block.self_attn.q_proj.bias is not None
    pre, _, _ = build_moe_split_wrapper(block)

    t = 5
    hidden = torch.randn(t, cfg.hidden_size)
    with torch.no_grad():
        q, k, v = pre(hidden)
        hn = block.input_layernorm(hidden)
        q_ref = block.self_attn.q_proj(hn)
        k_ref = block.self_attn.k_proj(hn)
        v_ref = block.self_attn.v_proj(hn)
    assert torch.allclose(q, q_ref, atol=1e-6)
    assert torch.allclose(k, k_ref, atol=1e-6)
    assert torch.allclose(v, v_ref, atol=1e-6)
    assert q.shape == (t, cfg.num_attention_heads * cfg.head_dim)
