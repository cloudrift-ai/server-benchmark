"""Hermetic equivalence test for the linear-attention split carve (no GPU, no vLLM).

Qwen3.5 alternates two token mixers per layer: ordinary attention, and a gated delta net whose
recurrence stays in torch. This proves the carve around that recurrence is exact — ``pre`` → the
block's own core → ``post`` reproduces the eager layer over the flattened ``[num_tokens, H]``
layout — and that the full-attention layer of the same architecture is REFUSED rather than
mis-carved, because its query projection carries an output gate the attention carve has no seam
for.

Pure eager, CPU. The tracer/compile half lives in ``tests/compiler/trace``.
"""

from __future__ import annotations

import pytest

# One linear_attention layer and one full_attention layer, sized so every architecture constraint
# holds at toy width: value heads a multiple of key heads (the delta rule repeat-interleaves keys
# and queries up to the value head count), and head_dim dividing the projections.
TINY = dict(
    vocab_size=64,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_conv_kernel_dim=4,
    max_position_embeddings=64,
    layer_types=["linear_attention", "full_attention"],
)


def _tiny_model(dtype):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("transformers.models.qwen3_5")
    import torch
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    torch.manual_seed(0)
    model = Qwen3_5TextModel(Qwen3_5TextConfig(**TINY)).to(dtype).eval()
    # A_log/dt_bias arrive at their init values; the decay path is what the core owns, so leave
    # them alone and only make sure they are not degenerate.
    return model, transformers


def test_linear_attention_split_matches_eager_block():
    """``pre`` → the block's own delta-net core → ``post`` equals the eager layer.

    The core here is the eager mixer re-driven from the carve's outputs, which is the point: the
    carve must not move any arithmetic across the seam, only stop and restart around it.
    """
    torch = pytest.importorskip("torch")

    from emmy.compiler.trace.huggingface import build_linear_attention_split_wrapper

    model, _ = _tiny_model(torch.float32)
    block = model.layers[0]
    assert block.block_type == "linear_attention"
    mixer = block.linear_attn

    t = 6
    hidden3d = torch.randn(1, t, TINY["hidden_size"])

    with torch.no_grad():
        eager = block(hidden3d, position_embeddings=None, attention_mask=None)
        eager = eager[0] if isinstance(eager, tuple) else eager

        pre, post = build_linear_attention_split_wrapper(block)
        mixed_qkv, z, b, a = pre(hidden3d.squeeze(0))
        core_out = _delta_core(mixer, mixed_qkv, z, b, a, batch=1, seq=t)
        out = post(core_out, hidden3d.squeeze(0))

    assert tuple(out.shape) == (t, TINY["hidden_size"])
    torch.testing.assert_close(out, eager.squeeze(0), rtol=1e-4, atol=1e-4)


def _delta_core(mixer, mixed_qkv, z, b, a, *, batch, seq):
    """The torch-side core the carve leaves alone: causal conv, the gated delta rule, gated norm.

    Written against the module's own parameters and the seam's flat tensors — reshaping into
    ``[batch, seq, ·]`` here is exactly the work the serving core would do, and doing it in the
    test is what proves the flat seam carries enough.
    """
    import torch
    import torch.nn.functional as F
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

    key_dim, value_dim = mixer.key_dim, mixer.value_dim
    qkv = mixed_qkv.view(batch, seq, -1).transpose(1, 2)  # [B, conv_dim, S]
    qkv = F.silu(F.conv1d(qkv, mixer.conv1d.weight, mixer.conv1d.bias, groups=mixer.conv_dim, padding=mixer.conv_kernel_size - 1))
    qkv = qkv[..., :seq].transpose(1, 2)
    query, key, value = torch.split(qkv, [key_dim, key_dim, value_dim], dim=-1)
    query = query.reshape(batch, seq, -1, mixer.head_k_dim)
    key = key.reshape(batch, seq, -1, mixer.head_k_dim)
    value = value.reshape(batch, seq, -1, mixer.head_v_dim)

    beta = b.view(batch, seq, -1).sigmoid()
    g = -mixer.A_log.float().exp() * F.softplus(a.view(batch, seq, -1).float() + mixer.dt_bias)
    repeat = mixer.num_v_heads // mixer.num_k_heads
    if repeat > 1:
        query = query.repeat_interleave(repeat, dim=2)
        key = key.repeat_interleave(repeat, dim=2)

    core_out, _ = torch_chunk_gated_delta_rule(
        query, key, value, g=g, beta=beta, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True
    )
    core_out = mixer.norm(core_out.reshape(-1, mixer.head_v_dim), z.reshape(-1, mixer.head_v_dim))
    return core_out.reshape(batch * seq, -1)


def test_pre_returns_the_four_projections_at_their_declared_widths():
    """The seam ABI: four tensors, each at the width the block's own dimensions name."""
    torch = pytest.importorskip("torch")

    from emmy.compiler.trace.huggingface import build_linear_attention_split_wrapper

    model, _ = _tiny_model(torch.bfloat16)
    block = model.layers[0]
    mixer = block.linear_attn
    pre, post = build_linear_attention_split_wrapper(block)

    t = 5
    with torch.no_grad():
        mixed_qkv, z, b, a = pre(torch.randn(t, TINY["hidden_size"], dtype=torch.bfloat16))
        out = post(torch.randn(t, mixer.value_dim, dtype=torch.bfloat16), torch.randn(t, TINY["hidden_size"], dtype=torch.bfloat16))

    assert tuple(mixed_qkv.shape) == (t, mixer.conv_dim) == (t, mixer.key_dim * 2 + mixer.value_dim)
    assert tuple(z.shape) == (t, mixer.value_dim)
    assert tuple(b.shape) == tuple(a.shape) == (t, mixer.num_v_heads)
    assert tuple(out.shape) == (t, TINY["hidden_size"])


def test_the_carve_refuses_a_block_with_a_biased_input_projection():
    """The padding-mask multiply the carve leaves to the core commutes with the projections only
    while they are bias-free. A bias would make the two orders differ on padded rows."""
    torch = pytest.importorskip("torch")

    from emmy.compiler.trace.huggingface import build_linear_attention_split_wrapper

    model, _ = _tiny_model(torch.float32)
    block = model.layers[0]
    block.linear_attn.in_proj_b.bias = torch.nn.Parameter(torch.zeros(block.linear_attn.num_v_heads))
    with pytest.raises(NotImplementedError, match="bias"):
        build_linear_attention_split_wrapper(block)


def test_the_carve_refuses_a_full_attention_block():
    """The two mixers are not interchangeable, and the linear carve says so rather than reading
    ``self_attn`` attributes that are not there."""
    torch = pytest.importorskip("torch")

    from emmy.compiler.trace.huggingface import build_linear_attention_split_wrapper

    model, _ = _tiny_model(torch.float32)
    with pytest.raises(NotImplementedError, match="gated-delta-net"):
        build_linear_attention_split_wrapper(model.layers[1])


def _repeat_kv(x, n_rep):
    """[1, Hkv, T, D] -> [1, Hkv*n_rep, T, D] (GQA head expansion)."""
    b, h, t, d = x.shape
    return x if n_rep == 1 else x[:, :, None, :, :].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


def test_attention_split_carves_the_fused_query_gate_projection():
    """Qwen3.5's full-attention layer fuses its output gate into ``q_proj``: the projection is
    twice its query width and the forward chunks it per head. The carve splits the same way, hands
    the gate across the seam as a fourth ``pre`` output, and ``post`` applies it before ``o_proj``.

    The reference reconstructs what the seam leaves out — RoPE, then causal GQA attention — so a
    match proves the carve moved no arithmetic across it.
    """
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

    from emmy.compiler.trace.huggingface import build_attention_split_wrapper, build_causal_mask

    model, _ = _tiny_model(torch.float32)
    block = model.layers[1]
    assert block.block_type == "full_attention"
    attn = block.self_attn
    head_dim, n_heads = attn.head_dim, TINY["num_attention_heads"]
    n_kv = TINY["num_key_value_heads"]
    assert attn.q_proj.out_features == 2 * n_heads * head_dim, "the fused layout under test"

    t = 6
    hidden3d = torch.randn(1, t, TINY["hidden_size"])
    cos, sin = model.rotary_emb(hidden3d, torch.arange(t).unsqueeze(0))
    mask = build_causal_mask(t, torch.float32)

    with torch.no_grad():
        eager = block(hidden3d, position_embeddings=(cos, sin), attention_mask=mask)
        eager = eager[0] if isinstance(eager, tuple) else eager

        pre, post = build_attention_split_wrapper(block)
        assert pre.emits_gate, "the fused layout must widen the seam"
        q2d, k2d, v2d, gate = pre(hidden3d.squeeze(0))
        assert tuple(gate.shape) == (t, n_heads * head_dim)

        q = q2d.view(t, n_heads, head_dim).transpose(0, 1).unsqueeze(0)
        k = k2d.view(t, n_kv, head_dim).transpose(0, 1).unsqueeze(0)
        v = v2d.view(t, n_kv, head_dim).transpose(0, 1).unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        attn_out = F.scaled_dot_product_attention(
            q, _repeat_kv(k, n_heads // n_kv), _repeat_kv(v, n_heads // n_kv), attn_mask=mask, scale=attn.scaling
        )
        attn_out = attn_out.transpose(1, 2).reshape(t, n_heads * head_dim)
        out = post(attn_out, hidden3d.squeeze(0), gate)

    assert tuple(out.shape) == (t, TINY["hidden_size"])
    torch.testing.assert_close(out, eager.squeeze(0), rtol=1e-4, atol=1e-4)


def test_the_fused_gate_actually_changes_the_answer():
    """Guard against a vacuous parity test: dropping the gate must break it. Otherwise a carve that
    silently ignored the gate — the bug this whole shape exists to prevent — would still pass."""
    torch = pytest.importorskip("torch")

    from emmy.compiler.trace.huggingface import build_attention_split_wrapper

    model, _ = _tiny_model(torch.float32)
    block = model.layers[1]
    pre, post = build_attention_split_wrapper(block)
    t = 6
    hidden = torch.randn(t, TINY["hidden_size"])
    attn_out = torch.randn(t, TINY["num_attention_heads"] * block.self_attn.head_dim)
    with torch.no_grad():
        *_, gate = pre(hidden)
        gated, ungated = post(attn_out, hidden, gate), post(attn_out, hidden)
    assert not torch.allclose(gated, ungated, rtol=1e-3, atol=1e-3), "the gate must move the output"


def test_attention_split_refuses_a_query_projection_wider_for_another_reason():
    """The carve admits exactly two query-projection layouts: plain, and the fused gate's exact
    doubling. Anything else still refuses rather than mis-shaping q."""
    torch = pytest.importorskip("torch")

    from emmy.compiler.trace.huggingface import build_attention_split_wrapper

    model, _ = _tiny_model(torch.float32)
    block = model.layers[1]
    head_dim = block.self_attn.head_dim
    # 3x the query width: neither the plain layout nor the fused one.
    block.self_attn.q_proj = torch.nn.Linear(TINY["hidden_size"], 3 * TINY["num_attention_heads"] * head_dim, bias=False)
    with pytest.raises(NotImplementedError, match="besides queries"):
        build_attention_split_wrapper(block)


def test_attention_split_still_accepts_an_unfused_query_projection():
    """The guard must not narrow the archs that already carve: a plain Qwen3 layer still builds."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from emmy.compiler.trace.huggingface import build_attention_split_wrapper

    config = transformers.Qwen3Config(
        vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=64,
    )  # fmt: skip
    torch.manual_seed(0)
    pre, post = build_attention_split_wrapper(transformers.Qwen3ForCausalLM(config).eval().model.layers[0])
    assert pre is not None and post is not None
