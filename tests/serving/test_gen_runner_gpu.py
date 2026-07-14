"""Phase-2 multi-layer host-stitch test for ``EmmyGenRunner`` (no vLLM).

``perf``-marked: needs CUDA + cupy. Builds a tiny multi-layer Qwen3, then runs a whole-model
Python stitch — ``embed`` → per layer (emmy ``pre`` kernels → reconstruct RoPE →
reference causal GQA torch SDPA → emmy ``post`` kernels) → ``final_norm`` → lm_head —
and checks the stitched logits against eager. This is the dress rehearsal for the vLLM
forward (Phase 3) without vLLM's runner, isolating the emmy↔attention interleave.
fp32 (carve correctness is dtype-independent; the fp16 path is covered by the Phase-0 oracle).
"""

import numpy as np
import pytest

pytestmark = [pytest.mark.perf, pytest.mark.xdist_group("cuda")]


def _repeat_kv(x, n_rep):
    b, h, t, d = x.shape
    if n_rep == 1:
        return x
    return x[:, :, None, :, :].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


def _reference_attention(runner, q_np, k_np, v_np, cos, sin, mask, apply_rotary):
    """The carve's attention seam, reconstructed for the host stitch: 2-D q/k/v → [1,H,T,D]
    → RoPE → causal GQA SDPA → 2-D attn_out. (Phase 3 replaces this with vLLM paged attention.)"""
    import torch
    import torch.nn.functional as F

    d, hq, hkv = runner.head_dim, runner.num_heads, runner.num_kv_heads
    t = q_np.shape[0]
    q = torch.from_numpy(np.ascontiguousarray(q_np)).view(t, hq, d).transpose(0, 1).unsqueeze(0)
    k = torch.from_numpy(np.ascontiguousarray(k_np)).view(t, hkv, d).transpose(0, 1).unsqueeze(0)
    v = torch.from_numpy(np.ascontiguousarray(v_np)).view(t, hkv, d).transpose(0, 1).unsqueeze(0)
    q, k = apply_rotary(q, k, cos, sin)
    k, v = _repeat_kv(k, hq // hkv), _repeat_kv(v, hq // hkv)
    attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=runner.scaling)  # [1, Hq, T, D]
    return attn.transpose(1, 2).reshape(t, hq * d).numpy()


def test_gen_runner_stitch_matches_eager():
    pytest.importorskip("cupy")
    import torch
    import transformers

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    from emmy.compiler.trace.huggingface import build_causal_mask
    from emmy.serving.gen_runner import EmmyGenRunner

    config = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        use_sliding_window=False,
    )
    torch.manual_seed(0)
    model = transformers.Qwen3ForCausalLM(config).eval()  # fp32; build_attention_split_wrapper does NOT mutate it

    runner = EmmyGenRunner.from_model(model, dtype_str="float32")
    assert runner.num_layers == config.num_hidden_layers

    t = 7
    input_ids = list(range(1, t + 1))
    position_ids = torch.arange(t).unsqueeze(0)
    mask = build_causal_mask(t, torch.float32)  # [1, 1, T, T]
    cos, sin = model.model.rotary_emb(torch.zeros(1, t, config.hidden_size), position_ids)

    # --- emmy host stitch ---
    hidden = runner.embed(input_ids)  # np [T, H]
    for layer in range(runner.num_layers):
        residual = hidden
        q, k, v = runner.forward_layer_pre(layer, hidden, position_ids)
        attn_out = _reference_attention(runner, q, k, v, cos, sin, mask, apply_rotary_pos_emb)
        hidden = runner.forward_layer_post(layer, attn_out, residual)
    hidden = runner.final_norm(hidden)
    with torch.no_grad():
        logits_dep = model.lm_head(torch.from_numpy(np.ascontiguousarray(hidden))).numpy()  # [T, vocab]

        # --- eager reference ---
        eager = model(torch.tensor([input_ids], dtype=torch.long)).logits[0].numpy()  # [T, vocab]

    assert logits_dep.shape == eager.shape
    np.testing.assert_allclose(logits_dep, eager, rtol=2e-3, atol=2e-3)
    # next-token greedy agrees too
    assert int(np.argmax(logits_dep[-1])) == int(np.argmax(eager[-1]))


def test_gen_runner_gemma4_heterogeneous_stitch():
    """Gemma-4 (gap #9): global (``full_attention``) layers use a LARGER head_dim (``global_head_dim``)
    and ``attention_k_eq_v`` (no ``v_proj`` → V reuses K's projection) than the sliding layers, so the
    runner must carry PER-LAYER attention metadata — a single (num_heads, head_dim) misshapes the global
    layers' ``o_proj``. Stitches the whole tiny gemma-4 trunk with per-layer dims + per-layer-type RoPE
    (full-causal since T < sliding_window) and checks it against the HF trunk's own hidden states."""
    pytest.importorskip("cupy")
    pytest.importorskip("transformers.models.gemma4")
    import torch
    import torch.nn.functional as F
    import transformers

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb

    from emmy.serving.gen_runner import EmmyGenRunner

    config = transformers.Gemma4TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,  # 3 sliding + a forced-global last layer
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        global_head_dim=32,  # global layers: larger head_dim than sliding (16)
        attention_k_eq_v=True,  # global layers: no v_proj (V reuses K)
        num_global_key_value_heads=2,  # required when attention_k_eq_v
        max_position_embeddings=128,
        sliding_window=64,
        hidden_size_per_layer_input=0,  # dense: no PLE
    )
    torch.manual_seed(0)
    model = transformers.Gemma4ForCausalLM(config).to(torch.float32).eval()
    trunk = model.model
    # The heterogeneity must actually be present, else the test proves nothing.
    assert trunk.layers[0].self_attn.head_dim != trunk.layers[-1].self_attn.head_dim
    assert trunk.layers[-1].self_attn.v_proj is None  # global attention_k_eq_v

    runner = EmmyGenRunner.from_model(model, dtype_str="float32", decode_bucket=16)
    assert runner.num_layers == config.num_hidden_layers

    t = 24  # > decode_bucket, < sliding_window (→ every layer is full-causal)
    ids = torch.randint(0, config.vocab_size, (1, t))
    pos = torch.arange(t).unsqueeze(0)
    with torch.no_grad():
        ref = trunk(input_ids=ids).last_hidden_state.squeeze(0)  # [T, H]
    cos_sin = {lt: trunk.rotary_emb(ref[None], pos, lt) for lt in set(config.layer_types)}
    causal = torch.triu(torch.full((t, t), float("-inf")), diagonal=1)[None, None]

    hidden = runner.embed(ids.squeeze(0).numpy())
    for layer in range(runner.num_layers):
        hd, nh, nkv, sc = runner.layer_meta(layer)  # PER-LAYER dims
        residual = hidden
        q2, k2, v2 = runner.forward_layer_pre(layer, hidden)
        q = torch.from_numpy(np.ascontiguousarray(q2)).view(t, nh, hd).transpose(0, 1)[None]
        k = torch.from_numpy(np.ascontiguousarray(k2)).view(t, nkv, hd).transpose(0, 1)[None]
        v = torch.from_numpy(np.ascontiguousarray(v2)).view(t, nkv, hd).transpose(0, 1)[None]
        cos, sin = cos_sin[config.layer_types[layer]]  # per-layer-type rope (its own head_dim)
        q, k = apply_rotary_pos_emb(q, cos, sin), apply_rotary_pos_emb(k, cos, sin)  # per-tensor (gemma-4)
        k, v = _repeat_kv(k, nh // nkv), _repeat_kv(v, nh // nkv)
        ao = F.scaled_dot_product_attention(q, k, v, attn_mask=causal, scale=sc)
        ao = ao.transpose(1, 2).reshape(t, nh * hd).contiguous().numpy()
        hidden = runner.forward_layer_post(layer, ao, residual)
    got = runner.final_norm(hidden)

    np.testing.assert_allclose(got, ref.numpy(), rtol=2e-3, atol=2e-3)


def test_gen_runner_device_path_matches_host():
    """The device-resident decode path (``run_device`` / ``*_device``) must match the host numpy
    path for the real ``T`` rows (``T <= decode_bucket``) — stale prefix padding never leaks
    because pre/post are per-token-independent. Regression guard for Phase A."""
    pytest.importorskip("cupy")
    import torch
    import transformers

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from emmy.serving.gen_runner import EmmyGenRunner

    config = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        use_sliding_window=False,
    )
    torch.manual_seed(0)
    model = transformers.Qwen3ForCausalLM(config).eval()
    runner = EmmyGenRunner.from_model(model, dtype_str="float32", decode_bucket=16)
    if not runner.has_device_decode:
        pytest.skip("decode-bucket programs unavailable for this shape")

    t = 5  # <= decode_bucket
    ids = list(range(1, t + 1))
    ids_t = torch.tensor(ids, dtype=torch.long, device="cuda")
    attn_width = runner.num_heads * runner.head_dim

    # embed / pre / post run the SAME GPU kernels on both paths → bit-identical for the real rows.
    h_np = runner.embed(ids)
    h_t = runner.embed_device(ids_t)
    np.testing.assert_array_equal(h_np, h_t.cpu().numpy())

    q_np, k_np, v_np = runner.forward_layer_pre(0, h_np)
    q, k, v = runner.forward_layer_pre_device(0, h_t)
    np.testing.assert_array_equal(q_np, q.cpu().numpy())
    np.testing.assert_array_equal(k_np, k.cpu().numpy())
    np.testing.assert_array_equal(v_np, v.cpu().numpy())

    attn = np.random.RandomState(0).randn(t, attn_width).astype(runner._np_dtype)
    out_np = runner.forward_layer_post(0, attn, h_np)
    out_t = runner.forward_layer_post_device(0, torch.from_numpy(attn).cuda(), h_t)
    np.testing.assert_array_equal(out_np, out_t.cpu().numpy())

    # final_norm runs a torch module CPU (host) vs the deep-copied CUDA module (device) — fp32 ULPs.
    fn_np = runner.final_norm(h_np)
    fn_t = runner.final_norm_device(h_t)
    np.testing.assert_allclose(fn_np, fn_t.cpu().numpy(), rtol=1e-4, atol=1e-4)
    # the host final_norm must still work AFTER the device path moved nothing in place (deepcopy):
    np.testing.assert_array_equal(fn_np, runner.final_norm(h_np))
