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
    for t in (24, 40):
        x = torch.randn(t, 16, dtype=torch.float16, device="cuda")
        warm = prog.run_device_sym([x])[0].clone()  # uncaptured warmup at this exact width
        assert torch.allclose(warm, ref_mod(x), rtol=1e-2, atol=1e-2)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = prog.run_device_sym([x])[0]
        graphs[t], ins[t], outs[t] = g, x, out

    for t in (24, 40):
        x2 = torch.randn(t, 16, dtype=torch.float16, device="cuda")
        ins[t].copy_(x2)
        graphs[t].replay()
        torch.cuda.synchronize()
        assert outs[t].shape[0] == t
        assert torch.allclose(outs[t], ref_mod(x2), rtol=1e-2, atol=1e-2), f"size {t} replay diverged"
