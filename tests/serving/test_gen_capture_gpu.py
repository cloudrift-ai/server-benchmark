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
    prog = _compile_split(wrapper, [torch.zeros(4, 16, dtype=torch.float16)], None, np.dtype("float16"))

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
