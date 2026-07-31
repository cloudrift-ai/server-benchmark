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
# runner's GPU paths), and the ``perf`` gating in ``tests/perf/conftest.py`` skips every
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
