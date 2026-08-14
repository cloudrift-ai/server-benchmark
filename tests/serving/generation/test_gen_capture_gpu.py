"""``_Program.run_device`` under an OUTER torch CUDA-graph capture (the vLLM whole-step
decode-capture path). Needs CUDA + cupy (skips itself off-GPU).

Builds one tiny static program (the decode-bucket twin shape class), then captures a
``run_device`` call inside ``torch.cuda.graph`` — the capture-aware branch must issue the
raw launch sequence (nested stream capture / graph launch would abort the capture) — and
checks the captured graph REPLAYS correctly: new input values written into the same input
tensor produce the matching output, i.e. the whole-step graph vLLM records is live, not a
baked snapshot of the capture-time values.
"""

import numpy as np
import pytest

# NOT perf-marked (correctness pin, must run under ``make test``; see tests/ARCHITECTURE.md).
pytestmark = [pytest.mark.xdist_group("cuda")]


def test_run_device_inside_outer_capture_replays_live():
    pytest.importorskip("cupy")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    from emmy.serving.gen_runner import _compile_split

    torch.manual_seed(0)
    wrapper = torch.nn.Linear(16, 16, bias=False).to(torch.float16).eval()
    prog, _ = _compile_split(wrapper, [torch.zeros(4, 16, dtype=torch.float16)], None, np.dtype("float16"))

    x = torch.randn(4, 16, dtype=torch.float16, device="cuda")
    ref0 = prog.run_device([x])[0].clone()  # uncaptured baseline (also warms the program)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = prog.run_device([x])[0]
    g.replay()
    torch.cuda.synchronize()
    assert torch.allclose(out, ref0, rtol=1e-3, atol=1e-3)

    # Replay must be LIVE: new values in the same input tensor flow through the graph.
    x2 = torch.randn(4, 16, dtype=torch.float16, device="cuda")
    ref2 = wrapper.cuda()(x2)
    x.copy_(x2)
    g.replay()
    torch.cuda.synchronize()
    assert torch.allclose(out, ref2, rtol=1e-2, atol=1e-2)


def test_run_device_sym_inside_outer_capture_replays_live():
    """``_Program.run_device_sym`` (the SYMBOLIC any-width path) under an outer torch
    CUDA-graph capture — the over-bucket decode sizes vLLM captures when
    ``cudagraph_capture_sizes`` follows ``max_num_seqs`` past the decode bucket (WS3.3).
    Protocol mirrors vLLM's: each size warms up uncaptured FIRST (which also populates
    the per-sym-key TMA descriptor overlay — descriptor encoding is an H2D copy and must
    never happen inside a capture), then captures at its exact width. Two sizes get two
    independent graphs over the same capacity buffers; each must replay LIVE."""
    pytest.importorskip("cupy")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    from emmy.serving.gen_runner import _compile_split

    torch.manual_seed(0)
    wrapper = torch.nn.Linear(16, 16, bias=False).to(torch.float16).eval()
    prog, _ = _compile_split(wrapper, [torch.zeros(4, 16, dtype=torch.float16)], ["input"], np.dtype("float16"), capacity=64)

    ref_mod = wrapper.cuda()
    graphs = {}
    ins = {}
    outs = {}
    out_backing_ptr = prog.program.arrays[prog.output_names[0]].data.ptr
    for t in (24, 40):
        x = torch.randn(t, 16, dtype=torch.float16, device="cuda")
        warm = prog.run_device_sym([x])[0]  # uncaptured warmup at this exact width
        assert warm.data_ptr() != out_backing_ptr  # uncaptured path must CLONE (buffer is reused)
        assert torch.allclose(warm, ref_mod(x), rtol=1e-2, atol=1e-2)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = prog.run_device_sym([x])[0]
        assert out.data_ptr() == out_backing_ptr  # captured path must be a VIEW (A1: no clone nodes)
        graphs[t], ins[t], outs[t] = g, x, out

    for t in (24, 40):
        x2 = torch.randn(t, 16, dtype=torch.float16, device="cuda")
        ins[t].copy_(x2)
        graphs[t].replay()
        torch.cuda.synchronize()
        assert outs[t].shape[0] == t
        assert torch.allclose(outs[t], ref_mod(x2), rtol=1e-2, atol=1e-2), f"size {t} replay diverged"


def test_run_device_aliased_input_backing_replays_live():
    """The EMMY_GEN_ALIAS_ATTN path: the caller writes INTO the program's own input backing (a
    ``torch.from_dlpack`` view), so ``upload_prefix_device`` self-copy-skips — and the captured
    graph must still replay LIVE (new values written into the backing flow through)."""
    pytest.importorskip("cupy")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    from emmy.serving.gen_runner import _compile_split

    torch.manual_seed(0)
    wrapper = torch.nn.Linear(16, 16, bias=False).to(torch.float16).eval()
    prog, _ = _compile_split(wrapper, [torch.zeros(4, 16, dtype=torch.float16)], None, np.dtype("float16"))

    # The alias: a torch view of the program's OWN input buffer (the post twin's attn_out class).
    x = torch.from_dlpack(prog.program.arrays[prog.input_names[0]])
    assert x.shape[0] == 4
    x.copy_(torch.randn(4, 16, dtype=torch.float16, device="cuda"))
    ref0 = prog.run_device([x])[0].clone()  # upload self-copy-skips (pointer equality)
    assert torch.allclose(ref0, wrapper.cuda()(x), rtol=1e-2, atol=1e-2)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = prog.run_device([x])[0]
    g.replay()
    torch.cuda.synchronize()
    assert torch.allclose(out, ref0, rtol=1e-3, atol=1e-3)

    # Replay must be LIVE through the aliased backing: writing new values into the SAME view
    # (what vLLM's attention does each step) flows through the graph with no upload copy.
    x2 = torch.randn(4, 16, dtype=torch.float16, device="cuda")
    ref2 = wrapper.cuda()(x2)
    x.copy_(x2)
    g.replay()
    torch.cuda.synchronize()
    assert torch.allclose(out, ref2, rtol=1e-2, atol=1e-2)


def test_rider_split_inside_outer_capture_replays_live():
    """The chunk+decode twin row SPLIT (a rider-width step) under an outer torch CUDA-graph
    capture — the rider-top rung of whole-step chunk capture. Both halves run
    ``run_device(out=...)``, whose copies into the persistent shared joint destinations must
    RECORD into the graph (the destinations were minted on the uncaptured warmup, exactly as
    vLLM's per-size warmup guarantees), and the replay must be LIVE: new values in the same
    input tensors flow through both halves. Checked against the SYMBOLIC program at the same
    width (independent kernels — hence a tolerance, not bit equality)."""
    pytest.importorskip("cupy")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    from transformers import LlamaConfig, LlamaForCausalLM

    from emmy.serving.gen_runner import EmmyGenRunner

    config = LlamaConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config).eval()
    runner = EmmyGenRunner.from_model(model, dtype_str="float32", decode_bucket=16, max_tokens=64, prefill_bucket=32)
    assert runner.prefill_bucket == 32 and runner.rider_width == 16
    t = 48  # the rider top: chunk 32 + decode 16
    attn_width = config.num_attention_heads * 16  # head_dim = hidden / heads
    hidden = (torch.randn(t, config.hidden_size, device="cuda") * 0.3).contiguous()
    attn = (torch.randn(t, attn_width, device="cuda") * 0.3).contiguous()
    residual = (torch.randn(t, config.hidden_size, device="cuda") * 0.3).contiguous()

    def sym_pre(h):
        return [o.clone() for o in runner._pre[0].run_device_sym([h])]

    def sym_post(a, r):
        return [o.clone() for o in runner._post[0].run_device_sym([a, r])]

    close = lambda a, b: torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)  # noqa: E731

    # Uncaptured warmup: mints the rider destinations and warms both twins at this width,
    # and pins the split against the symbolic program on the same values.
    warm_pre = runner.forward_layer_pre_device(0, hidden)
    pre_ptrs = [w.data_ptr() for w in warm_pre]
    for w, r in zip(warm_pre, sym_pre(hidden), strict=True):
        close(w, r)
    warm_post = runner._route_post_device(0, attn, residual)
    for w, r in zip(warm_post, sym_post(attn, residual), strict=True):
        close(w, r)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        pre_out = runner.forward_layer_pre_device(0, hidden)
        post_out = runner._route_post_device(0, attn, residual)
    # The captured outputs are the SAME persistent destination slices the warmup minted —
    # nothing was allocated inside the capture.
    assert [o.data_ptr() for o in pre_out] == pre_ptrs

    # Replay must be LIVE through both halves of both programs.
    torch.manual_seed(1)
    h2 = (torch.randn_like(hidden) * 0.3).contiguous()
    a2 = (torch.randn_like(attn) * 0.3).contiguous()
    r2 = (torch.randn_like(residual) * 0.3).contiguous()
    ref_pre = sym_pre(h2)
    ref_post = sym_post(a2, r2)
    hidden.copy_(h2)
    attn.copy_(a2)
    residual.copy_(r2)
    g.replay()
    torch.cuda.synchronize()
    for o, r in zip(pre_out, ref_pre, strict=True):
        assert o.shape[0] == t
        close(o, r)
    for o, r in zip(post_out, ref_post, strict=True):
        close(o, r)


def test_moe_fixed_slot_decode_step_inside_outer_capture_replays_live():
    """The fixed-slot MoE decode step (post_attn twin → router → index_select staging → k slot
    launches → score matmul) under an outer torch CUDA-graph capture — what vLLM's whole-step
    FULL_DECODE_ONLY capture at size 1 records for an MoE model. Captured once, replayed twice:
    new values in the same input tensors must flow through the ROUTING too (the top-k indices
    and the staged expert weights are data-dependent VALUES inside the graph), checked against
    the eager routed path on fresh inputs."""
    pytest.importorskip("cupy")
    import torch
    import transformers

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

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
    assert runner.has_moe_fixed_slot

    attn_width = runner.num_heads * runner.head_dim
    torch.manual_seed(1)
    attn = torch.randn(1, attn_width, device="cuda")
    residual = torch.randn(1, config.hidden_size, device="cuda")

    def eager_ref(a, r):
        """The routed path on the same twin outputs — the parity oracle for the captured step."""
        h, xn = runner._route_post_device(0, a, r)
        return (h + runner._moe_combine(runner._moe[0], xn)).clone()

    # Warmup (mints the twin + slot program graphs and the staging pair), then the baseline.
    warm = runner.forward_layer_post_device(0, attn, residual)
    ref0 = eager_ref(attn, residual)
    torch.testing.assert_close(warm, ref0, rtol=1e-4, atol=1e-5)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = runner.forward_layer_post_device(0, attn, residual)
    g.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, ref0, rtol=1e-4, atol=1e-5)

    # Replay must be LIVE through the routing: new inputs select a different expert set.
    for seed in (2, 3):
        torch.manual_seed(seed)
        a2 = torch.randn(1, attn_width, device="cuda")
        r2 = torch.randn(1, config.hidden_size, device="cuda")
        ref2 = eager_ref(a2, r2)
        attn.copy_(a2)
        residual.copy_(r2)
        g.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, ref2, rtol=1e-4, atol=1e-5)


def test_run_device_sym_aliased_input_backing_replays_live():
    """The A2 chained-seam primitive on the SYMBOLIC path: the caller writes INTO the sym
    program's own input backing (what the previous layer's chained post output is, after A2),
    so ``upload_prefix_device`` self-copy-skips — and a captured graph over ``run_device_sym``
    must still replay LIVE through the aliased backing."""
    pytest.importorskip("cupy")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    from emmy.serving.gen_runner import _compile_split

    torch.manual_seed(0)
    wrapper = torch.nn.Linear(16, 16, bias=False).to(torch.float16).eval()
    prog, _ = _compile_split(wrapper, [torch.zeros(4, 16, dtype=torch.float16)], ["input"], np.dtype("float16"), capacity=64)

    ref_mod = wrapper.cuda()
    t = 24
    # The alias: a torch prefix view of the sym program's OWN capacity input buffer.
    x = torch.from_dlpack(prog.program.arrays[prog.input_names[0]])[:t]
    x.copy_(torch.randn(t, 16, dtype=torch.float16, device="cuda"))
    warm = prog.run_device_sym([x])[0]  # uncaptured warmup at this width (self-copy-skips)
    assert torch.allclose(warm, ref_mod(x), rtol=1e-2, atol=1e-2)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = prog.run_device_sym([x])[0]
    x2 = torch.randn(t, 16, dtype=torch.float16, device="cuda")
    ref2 = ref_mod(x2)
    x.copy_(x2)
    g.replay()
    torch.cuda.synchronize()
    assert torch.allclose(out, ref2, rtol=1e-2, atol=1e-2)
