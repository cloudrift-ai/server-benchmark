"""Phase-2 multi-layer host-stitch test for ``EmmyGenRunner`` (no vLLM).

Needs CUDA + cupy (skips itself otherwise). Builds a tiny multi-layer Qwen3, then runs a whole-model
Python stitch — ``embed`` → per layer (emmy ``pre`` kernels → reconstruct RoPE →
reference causal GQA torch SDPA → emmy ``post`` kernels) → ``final_norm`` → lm_head —
and checks the stitched logits against eager. This is the dress rehearsal for the vLLM
forward (Phase 3) without vLLM's runner, isolating the emmy↔attention interleave.
fp32 (carve correctness is dtype-independent; the fp16 path is covered by the Phase-0 oracle).
"""

import numpy as np
import pytest

# NOT perf-marked: these are correctness pins (the only regression guards for the serving
# runner's GPU paths), and the ``perf`` gating in the root ``tests/conftest.py`` skips every
# perf-marked item suite-wide under plain ``pytest tests/`` — a perf mark here would silently
# drop these from ``make test`` on GPU machines.
pytestmark = [pytest.mark.xdist_group("cuda")]


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
    # Pin layer_scalar off 1 per layer (fresh models hold 1.0, masking a dropped multiply — the
    # 12B checkpoint ranges 0.005–0.92 and dropping it overflows fp16 by mid-network).
    for i, blk in enumerate(trunk.layers):
        blk.layer_scalar.fill_(0.6 + 0.1 * i)

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


@pytest.mark.xfail(
    strict=False,
    reason="main #438's offline-weights refit steers this tiny fp32 shape onto a TMA-staged pick with "
    "run-to-run last-ulp instability (reproducible within one runner on pristine origin/main) — the "
    "bit-parity contract holds again once the staging race is fixed; see the step-7 merge notes",
)
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


def test_gen_runner_moe_stitch_matches_eager():
    """MoE third seam (OLMoE): the device stitch — emmy ``pre`` → reference SDPA → emmy
    ``post_attn`` + torch router + shared expert program + weighted combine — must match eager
    logits. Also pins the program-count contract: ONE expert program serves every layer
    (weights are inputs), so program count does not scale with num_experts."""
    pytest.importorskip("cupy")
    import torch
    import torch.nn.functional as F
    import transformers

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from transformers.models.olmoe.modeling_olmoe import OlmoeForCausalLM, apply_rotary_pos_emb

    from emmy.serving.gen_runner import EmmyGenRunner

    config = transformers.OlmoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=8,
        num_experts_per_tok=2,
        norm_topk_prob=False,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    model = OlmoeForCausalLM(config).eval()
    runner = EmmyGenRunner.from_model(model, dtype_str="float32", decode_bucket=16, max_tokens=64)
    assert runner._moe is not None and runner._expert_sym is not None
    assert runner._expert_sym.input_names == ["x", "w_gate_up", "w_down"]

    t = 7
    ids = torch.arange(1, t + 1, dtype=torch.long)
    pos = torch.arange(t).unsqueeze(0)
    cos, sin = model.model.rotary_emb(torch.zeros(1, t, config.hidden_size), pos)
    cos_d, sin_d = cos.cuda(), sin.cuda()
    mask = torch.triu(torch.full((t, t), float("-inf")), diagonal=1)[None, None].cuda()
    hd, nh, nkv = runner.head_dim, runner.num_heads, runner.num_kv_heads

    hidden = runner.embed_device(ids.cuda())
    for layer in range(runner.num_layers):
        residual = hidden
        q2, k2, v2 = runner.forward_layer_pre_device(layer, hidden)
        q = q2.view(t, nh, hd).transpose(0, 1)[None]
        k = k2.view(t, nkv, hd).transpose(0, 1)[None]
        v = v2.view(t, nkv, hd).transpose(0, 1)[None]
        q, k = apply_rotary_pos_emb(q, k, cos_d, sin_d)
        ao = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=runner.scaling)
        ao = ao.transpose(1, 2).reshape(t, nh * hd).contiguous()
        hidden = runner.forward_layer_post_device(layer, ao, residual)
    got = runner.final_norm_device(hidden)
    with torch.no_grad():
        logits = model.lm_head(got.cpu()).numpy()
        eager = model(ids.unsqueeze(0)).logits[0].numpy()
    np.testing.assert_allclose(logits, eager, rtol=2e-3, atol=2e-3)
    assert int(np.argmax(logits[-1])) == int(np.argmax(eager[-1]))


def test_moe_fixed_slot_combine_matches_routed_oracle():
    """The fixed-slot combine (``_moe_combine_slots`` — selector write → k indirect slot
    replays → score matmul) must match ``combine_routed_experts`` (the routed parity oracle)
    on random routings: random ``xn`` rows route to different top-k expert sets, so the
    selector write, the in-kernel table reads, and the score combine are all exercised across
    routings. fp32 allclose — the two paths sum the k partials in different orders."""
    pytest.importorskip("cupy")
    import torch
    import transformers

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from transformers.models.olmoe.modeling_olmoe import OlmoeForCausalLM

    from emmy.serving.gen_runner import EmmyGenRunner

    config = transformers.OlmoeConfig(
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
    torch.manual_seed(0)
    model = OlmoeForCausalLM(config).eval()
    runner = EmmyGenRunner.from_model(model, dtype_str="float32", decode_bucket=16, max_tokens=64)
    assert runner.has_moe_fixed_slot, "the fixed-slot tier must build for the plain OLMoE expert shape"
    assert len(runner._expert_slots) == config.num_experts_per_tok
    slots = runner._expert_slots
    assert len({slot.program.slab_plan.slab.data.ptr for slot in slots}) == 1, "fixed slots must share scratch"
    assert len({slot.program.arrays["x"].data.ptr for slot in slots}) == 1, "fixed slots must share their ordered input"
    assert len({slot.program.arrays[slot.output_names[0]].data.ptr for slot in slots}) == len(slots), (
        "each fixed slot must retain its own partial output"
    )

    torch.manual_seed(1)
    for layer in (0, 1):
        moe = runner._moe[layer]
        for _ in range(4):
            xn = torch.randn(1, config.hidden_size, device="cuda")
            got = runner._moe_combine_slots(moe, xn)
            ref = runner._moe_combine(moe, xn)
            assert got.shape == ref.shape == xn.shape
            torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-5)


def test_moe_indirect_slot_matches_direct_expert_bit_exact():
    """The indirect expert program (weight base pointers resolved in-kernel from the device
    tables — ``table[sel[slot]]``) must match the DIRECT M=1 expert program BIT-EXACT on every
    expert of every MoE layer: the indirection is ABI-level, so both programs run the same
    schedule and the same kernels modulo where the base pointer comes from. Runs each expert's
    weights through both paths on the same row and compares raw bytes."""
    pytest.importorskip("cupy")
    import cupy as cp
    import torch
    import transformers

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from transformers.models.olmoe.modeling_olmoe import OlmoeForCausalLM

    from emmy.compiler.backend.gpu_lock import gpu_lock
    from emmy.serving.gen_runner import EmmyGenRunner

    config = transformers.OlmoeConfig(
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
    torch.manual_seed(0)
    model = OlmoeForCausalLM(config).eval()
    runner = EmmyGenRunner.from_model(model, dtype_str="float32", decode_bucket=16, max_tokens=64)
    assert runner.has_moe_fixed_slot
    runner._ensure_device()

    slot0 = runner._expert_slots[0]
    # The slot plan must actually carry the indirect encoding for both weights.
    marked = {a for lc in slot0.program.compiled.launches for a, _, _, _ in lc.indirect_args}
    assert marked == {"w_gate_up", "w_down"}

    torch.manual_seed(2)
    for moe in runner._moe:
        for e in range(config.num_experts):
            x = torch.randn(1, config.hidden_size, device="cuda")
            with gpu_lock(), cp.cuda.Stream.from_external(torch.cuda.current_stream()):
                direct = runner._launch_expert(moe, e, x).clone()  # the direct M=1 twin (pointer swap)
                runner._slot_sel[0] = moe["sel_off"] + e  # steer slot 0's table read at this expert
                p = slot0.program
                p.upload_prefix_device({"x": cp.from_dlpack(x.detach().contiguous())})
                p.run_once()
            torch.cuda.synchronize()
            got = runner._slot_partials[0:1]
            assert got.cpu().numpy().tobytes() == direct.cpu().numpy().tobytes(), f"expert {e}: indirect != direct bytes"


def test_moe_expert_shape_groups_compile_and_dispatch_per_layer():
    """Layers whose per-expert weight SHAPE differs need one expert program set each.

    EXL3's mixed bit allocation is the live case (K varies per layer, so the codes shape does),
    and a single shared program used to raise on it. Reproduced here without a quantized
    checkpoint by re-minting layer 1's experts at a wider intermediate size: the runner must
    build two shape groups, keep every tier per group, and route each layer's launches — routed
    and fixed-slot alike — through its OWN group's programs."""
    pytest.importorskip("cupy")
    import torch
    import transformers

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from transformers.models.olmoe.modeling_olmoe import OlmoeForCausalLM

    from emmy.serving.gen_runner import EmmyGenRunner

    config = transformers.OlmoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=False,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    model = OlmoeForCausalLM(config).eval()
    wide, e, h = 48, config.num_experts, config.hidden_size
    experts1 = model.model.layers[1].mlp.experts
    experts1.gate_up_proj = torch.nn.Parameter(torch.randn(e, 2 * wide, h) * 0.05)
    experts1.down_proj = torch.nn.Parameter(torch.randn(e, h, wide) * 0.05)

    runner = EmmyGenRunner.from_model(model, dtype_str="float32", decode_bucket=8, max_tokens=64)
    assert len(runner._expert_tiers) == 2, "two distinct expert shapes must build two program sets"
    assert [m["group"] for m in runner._moe] == [0, 1]
    for g, tiers in enumerate(runner._expert_tiers):
        assert tiers["sym"] is not None and tiers["one"] is not None, f"group {g} lost a tier"
    assert runner.has_moe_fixed_slot, "the fixed-slot tier must build for every group"

    torch.manual_seed(1)
    for layer, inter in ((0, config.intermediate_size), (1, wide)):
        moe = runner._moe[layer]
        gu, dn = moe["inputs"]["w_gate_up"], moe["inputs"]["w_down"]
        assert gu.shape[1] == 2 * inter, "the group's store must carry that layer's own shape"
        for t in (1, 8):
            x = torch.randn(t, h, device="cuda")
            expert = 2
            got = runner._launch_expert(moe, expert, x)
            gate, up = torch.nn.functional.linear(x, gu[expert].cuda()).chunk(2, dim=-1)
            ref = torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, dn[expert].cuda())
            torch.testing.assert_close(got, ref, rtol=2e-3, atol=2e-3)
        xn = torch.randn(1, h, device="cuda")
        torch.testing.assert_close(runner._moe_combine_slots(moe, xn), runner._moe_combine(moe, xn), rtol=2e-3, atol=2e-3)


def test_moe_expert_m256_twin_matches_eager_across_experts():
    """The static M=256 prefill expert twin (captured whole-program replay, weights by
    UPLOAD) must match the eager expert wrapper on an over-bucket row set — and stay
    correct ACROSS experts: the captured graph bakes the twin's own buffer pointers, so a
    pointer-swap regression would freeze the first expert's weights into every later
    replay. ``_launch_expert`` routes row sets in (decode_bucket, 256] here."""
    pytest.importorskip("cupy")
    import cupy as cp
    import torch
    import transformers

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from transformers.models.olmoe.modeling_olmoe import OlmoeForCausalLM

    from emmy.compiler.backend.gpu_lock import gpu_lock
    from emmy.serving.gen_runner import EmmyGenRunner

    config = transformers.OlmoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=8,
        num_experts_per_tok=2,
        norm_topk_prob=False,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    model = OlmoeForCausalLM(config).eval()
    runner = EmmyGenRunner.from_model(model, dtype_str="float32", decode_bucket=4, max_tokens=64)
    assert runner._expert_m256 is not None, "the M=256 prefill twin must build for the plain OLMoE expert shape"
    runner._ensure_device()

    moe = runner._moe[0]
    torch.manual_seed(3)
    for t in (10, 5):  # both over-bucket widths must reuse the one cached graph (prefix upload pads)
        for e in (0, 3, 7):
            x = torch.randn(t, config.hidden_size, device="cuda")
            with gpu_lock(), cp.cuda.Stream.from_external(torch.cuda.current_stream()):
                got = runner._launch_expert(moe, e, x).clone()
            torch.cuda.synchronize()
            gate, up = torch.nn.functional.linear(x, moe["inputs"]["w_gate_up"][e]).chunk(2, dim=-1)
            ref = torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, moe["inputs"]["w_down"][e])
            assert got.shape == (t, config.hidden_size)
            torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-5)
    assert runner._expert_m256.program._e2e_graph is not None, "the m256 tier must serve via the captured program graph"


def test_decode_twin_shares_weight_buffers():
    """The static decode-bucket twin must NOT hold a second on-GPU copy of the layer's
    weights: every non-trivial constant buffer in a decode program shares its device
    allocation with the symbolic sibling (one upload per weight — the plan's
    duplicate-weights memory artifact, ~2× the trunk when regressed)."""
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
        num_hidden_layers=1,
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

    def const_ptrs(prog):
        """{base device pointer: nbytes} of every constant buffer in a built program."""
        p = prog.program
        return {p.arrays[b.name].data.ptr: p.arrays[b.name].nbytes for b in p.compiled.bufs if b.role == "constant"}

    for sym, dec in ((runner._pre[0], runner._pre_decode[0]), (runner._post[0], runner._post_decode[0])):
        sym_ptrs = const_ptrs(sym)
        shared = unshared = 0
        for ptr, nbytes in const_ptrs(dec).items():
            if nbytes < 1024:  # scalars / tiny synthetics may legitimately differ per build
                continue
            if ptr in sym_ptrs:
                shared += nbytes
            else:
                unshared += nbytes
        assert shared > 0, "expected the decode twin to share weight buffers with the symbolic program"
        assert unshared == 0, f"decode twin holds {unshared} bytes of duplicated weight buffers"


def test_device_residents_allocated_eagerly():
    """Every device-resident buffer must exist right after ``from_model`` — vLLM sizes its
    KV cache from a profiling pass that runs after model construction, so a lazy allocation
    at the first small-batch decode (the ~1.9 GiB gemma-4-12B embed table) lands on memory
    the KV cache already claimed and OOMs at high --gpu-memory-utilization. Found on a real
    RTX 5090; a concurrent bench masks it (batched decode rides the symbolic path) until
    concurrency drains."""
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
        num_hidden_layers=1,
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

    assert getattr(runner, "_dev_ready", False), "device residents must be allocated at construction, not first decode"
    assert runner._embed_weight_dev.is_cuda
    assert next(runner._norm_dev.parameters()).is_cuda


def test_adopted_raw_embed_table_matches_folded_gather():
    """``adopt_embed_table`` (the tied lm_head↔embed sharing that returns ~2 GiB to the KV
    budget on gemma) must gather identically to the runner's own folded upload: the shared
    table is RAW and the embed scale re-applies at gather in fp32 — the same
    fold-at-fp32-then-cast numerics as the host table."""
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
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        use_sliding_window=False,
    )
    torch.manual_seed(0)
    model = transformers.Qwen3ForCausalLM(config).eval()
    runner = EmmyGenRunner.from_model(model, dtype_str="float32", decode_bucket=16)
    ids = torch.tensor([1, 3, 5], dtype=torch.long, device="cuda")
    folded = runner.embed_device(ids)

    scale = 4.0  # pretend embed_scale (Qwen has none — exercise the re-apply path explicitly)
    raw = (model.model.embed_tokens.weight.detach().to(torch.float32) / 1.0).cuda()
    runner._embed_weight = (runner._embed_weight * scale).astype(runner._embed_weight.dtype)  # refold host at the new scale
    runner.adopt_embed_table(raw, scale=scale)
    adopted = runner.embed_device(ids)
    np.testing.assert_allclose(adopted.cpu().numpy(), folded.cpu().numpy() * scale, rtol=1e-6, atol=1e-6)
    assert runner._embed_weight_dev.data_ptr() == raw.data_ptr(), "the adopted table must be shared, not copied"


def test_layers_share_activation_arena():
    """Per-layer programs must NOT each hold their own capacity-sized activation
    buffers: with the runner's shared :class:`BufferArena`, same-kind programs of
    different layers view the SAME device memory for every input/output buffer and
    the scratch slab (layers run sequentially — the ~350 MB × num_layers artifact).
    Layer 0's builds grow the arena to its stable sizes, so the identity is asserted
    between layers 1 and 2. Numerics under pooling are covered by the multi-layer
    stitch test above, which runs all layers through the shared buffers."""
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
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        use_sliding_window=False,
    )
    torch.manual_seed(0)
    model = transformers.Qwen3ForCausalLM(config).eval()
    runner = EmmyGenRunner.from_model(model, dtype_str="float32", decode_bucket=16)

    def io_ptrs(prog):
        """{(role, name): base device pointer} for the input/output buffers."""
        p = prog.program
        return {(b.role, b.name): p.arrays[b.name].data.ptr for b in p.compiled.bufs if b.role in ("input", "output")}

    for kind in ("_pre", "_post", "_pre_decode", "_post_decode"):
        programs = getattr(runner, kind)
        if programs is None:
            continue
        a, b = programs[1], programs[2]
        assert a.program.slab_plan.slab.data.ptr == b.program.slab_plan.slab.data.ptr, f"{kind}: scratch slabs not shared"
        pa, pb = io_ptrs(a), io_ptrs(b)
        assert pa.keys() == pb.keys(), f"{kind}: layer programs disagree on buffer names"
        for key in pa:
            assert pa[key] == pb[key], f"{kind}: activation buffer {key} not shared across layers"


def test_expert_program_fp8_inputs_match_reference():
    """Input-sourced fp8 (MoE M3): the expert program compiled with its weight INPUTS spelled
    as fp8 bits + scale inputs (``spell_quantized_inputs``) must match the eager expert on the
    dequantized weights, and stay within the fp8 quantization bound of the eager expert on the
    ORIGINAL fp16 weights. Weights quantize per-out-channel and apply as ``x @ W`` in the
    transposed expert layout. Runs the M=16 and M=1 static twins; the build goes
    through ``_compile_split``'s plan path, so the per-input feed binding (fp8 bits on the
    uint8 carrier from a torch fp8 tensor, f32 scales) is exercised too."""
    pytest.importorskip("cupy")
    import torch
    import torch.nn as nn

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.backend.plan import plan_from_graph
    from emmy.compiler.loader.quant import spell_quantized_inputs
    from emmy.serving.gen_runner import _compile_split, trace_split

    class Expert(nn.Module):
        def forward(self, x, w_gate_up, w_down):
            gate, up = (x @ w_gate_up).chunk(2, dim=-1)
            return (nn.functional.silu(gate) * up) @ w_down

    h, inter = 64, 32
    torch.manual_seed(4)
    w_gu = torch.randn(h, 2 * inter, dtype=torch.float32) * 0.05
    w_dn = torch.randn(inter, h, dtype=torch.float32) * 0.05

    def quantize_per_out_channel(w):
        scale = (w.abs().amax(dim=0, keepdim=True).clamp(min=1e-8) / 448.0).float()
        bits = (w / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        return bits, scale

    gu_bits, gu_scale = quantize_per_out_channel(w_gu)
    dn_bits, dn_scale = quantize_per_out_channel(w_dn)
    gu_dq = gu_bits.float() * gu_scale
    dn_dq = dn_bits.float() * dn_scale

    for m in (16, 1):
        x = torch.randn(m, h, dtype=torch.float16)
        graph = trace_split(
            Expert(),
            (x, w_gu.to(torch.float16), w_dn.to(torch.float16)),
            None,
        )
        spell_quantized_inputs(graph, {"w_gate_up": ("f8e4m3", (1, 2 * inter), "f32"), "w_down": ("f8e4m3", (1, h), "f32")})
        graph.validate()
        plan = plan_from_graph(CudaBackend(tune_db="auto").compile(graph))
        assert plan.inputs == ["x", "w_gate_up", "w_down", "w_gate_up_scale", "w_down_scale"]
        prog, _ = _compile_split(
            Expert(),
            [x, gu_bits, dn_bits, gu_scale, dn_scale],  # torch fp8 tensors: the feed must view bits
            None,
            np.dtype("float16"),
            plan=plan,
        )
        (out,) = prog.run(
            [
                x.numpy(),
                gu_bits.view(torch.uint8).numpy(),
                dn_bits.view(torch.uint8).numpy(),
                gu_scale.numpy(),
                dn_scale.numpy(),
            ]
        )
        with torch.no_grad():
            ref_dq = Expert()(x.float(), gu_dq, dn_dq).numpy()
            ref_orig = Expert()(x.float(), w_gu, w_dn).numpy()
        got = np.asarray(out, dtype=np.float32).reshape(ref_dq.shape)
        rms = float(np.sqrt((ref_dq**2).mean()))
        assert rms > 1e-3, "reference suspiciously small; tolerance would be trivial"
        # vs the dequantized reference: only f16 kernel arithmetic separates the two.
        assert float(np.abs(got - ref_dq).max()) / rms < 2e-2, f"M={m}: kernel path deviates from the dequantized reference"
        # vs the original weights: the fp8 quantization error bound — RMS-relative, since the
        # per-element error of two chained quantized matmuls compounds past the single-cone
        # ~5% max while the aggregate stays a few percent.
        rms_orig = float(np.sqrt((ref_orig**2).mean()))
        assert float(np.sqrt(((got - ref_orig) ** 2).mean())) / max(rms_orig, 1e-6) < 5e-2, (
            f"M={m}: fp8 error exceeds the quantization bound"
        )


def test_expert_program_fp8_indirect_compose():
    """Indirect operands (the M2 fixed-slot dispatch) compose with input-sourced fp8: BOTH the
    bits input and its scale input compile as indirect operands, and the kernel resolves each
    expert's bits + scale slices from the device tables (``table[sel[slot]]``) — per-expert
    outputs match the direct dequantized matmul."""
    pytest.importorskip("cupy")
    import cupy as cp
    import torch
    import torch.nn as nn

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from emmy.compiler.backend.gpu_lock import gpu_lock
    from emmy.compiler.loader.quant import spell_quantized_inputs
    from emmy.serving.gen_runner import _compile_split, _indirect_covered, trace_split

    class XW(nn.Module):
        def forward(self, x, w):
            return x @ w

    h, n, n_experts = 64, 32, 3
    torch.manual_seed(5)
    graph = trace_split(XW(), (torch.zeros(1, h, dtype=torch.float16), torch.zeros(h, n, dtype=torch.float16)), None)
    spell_quantized_inputs(graph, {"w": ("f8e4m3", (1, n), "f32")})
    graph.hints.set("cuda.indirect_inputs", ("w", "w_scale"))

    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.backend.plan import plan_from_graph

    plan = plan_from_graph(CudaBackend(tune_db="auto").compile(graph))
    assert _indirect_covered(plan, ("w", "w_scale")), "an fp8 bits/scale input arg lacks the indirect encoding"

    x = torch.randn(1, h, dtype=torch.float16)
    w_e = [torch.randn(h, n, dtype=torch.float32) * 0.05 for _ in range(n_experts)]
    bits_e, scale_e = [], []
    for w in w_e:
        scale = (w.abs().amax(dim=0, keepdim=True).clamp(min=1e-8) / 448.0).float()
        bits_e.append((w / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).cuda())
        scale_e.append(scale.cuda())

    prog, _ = _compile_split(XW(), [x, bits_e[0], scale_e[0]], None, np.dtype("float16"), plan=plan)
    p = prog.program
    table_w = torch.tensor([t.data_ptr() for t in bits_e], dtype=torch.int64, device="cuda")
    table_s = torch.tensor([t.data_ptr() for t in scale_e], dtype=torch.int64, device="cuda")
    sel = torch.zeros(1, dtype=torch.int32, device="cuda")
    with gpu_lock(), cp.cuda.Stream.from_external(torch.cuda.current_stream()):
        p.arrays["w__table"] = cp.from_dlpack(table_w)
        p.arrays["w__sel"] = cp.from_dlpack(sel)
        p.arrays["w_scale__table"] = cp.from_dlpack(table_s)
        p.arrays["w_scale__sel"] = cp.from_dlpack(sel)
    for e in range(n_experts):
        with gpu_lock(), cp.cuda.Stream.from_external(torch.cuda.current_stream()):
            sel[0] = e
            p.upload_prefix_device({"x": cp.from_dlpack(x.cuda())})
            p.run_once()
            outs = p.output_prefix_device()
            got = torch.from_dlpack(outs[prog.output_names[0]]).clone()
        torch.cuda.synchronize()
        ref = (x.float() @ (bits_e[e].float().cpu() * scale_e[e].cpu())).to(torch.float16)
        torch.testing.assert_close(got.cpu().reshape(ref.shape), ref, rtol=2e-2, atol=2e-2)


def test_host_mapped_embed_table_gathers_identically_and_costs_no_vram(monkeypatch):
    """``EMMY_GEN_EMBED_HOST`` moves the vocab table out of VRAM without changing a single
    gathered row.

    The knob exists to hand a vocab-sized fp16 table (over a GiB on a 150k vocab) back to the
    KV cache on a card the weights already fill. Three things have to hold: the gather still
    lands on the same values, the table's device pointer is NOT device memory (otherwise the
    bytes were never returned), and the numpy table the oracle path reads is a VIEW of the same
    allocation rather than a second host copy."""
    pytest.importorskip("cupy")
    import cupy as cp
    import torch
    import transformers

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from emmy.serving.gen_runner import EmmyGenRunner

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
    model = transformers.Qwen3ForCausalLM(config).eval()
    ids = torch.tensor([1, 3, 5, 63], dtype=torch.long, device="cuda")

    device_runner = EmmyGenRunner.from_model(model, dtype_str="float32", decode_bucket=16)
    if not device_runner.has_device_decode:
        pytest.skip("decode-bucket programs unavailable for this shape")
    expected = device_runner.embed_device(ids).cpu().numpy()

    monkeypatch.setenv("EMMY_GEN_EMBED_HOST", "1")
    runner = EmmyGenRunner.from_model(model, dtype_str="float32", decode_bucket=16)
    np.testing.assert_array_equal(runner.embed_device(ids).cpu().numpy(), expected)
    np.testing.assert_array_equal(runner.embed(ids.cpu().numpy()), expected)

    table = runner._embed_weight_dev
    assert table.is_cuda, "the gather must stay a device-side op (host round trips break capture)"
    attrs = cp.cuda.runtime.pointerGetAttributes(table.data_ptr())
    assert int(attrs.hostPointer) == table.data_ptr(), "the table must be host memory mapped into the device space"
    assert runner._embed_weight.ctypes.data == table.data_ptr(), "the numpy view must alias the mapped buffer, not copy it"
