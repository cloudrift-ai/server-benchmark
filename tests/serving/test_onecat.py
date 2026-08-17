from types import SimpleNamespace

import torch

from emmy.compiler.backend.cuda.backend import CudaBackend
from emmy.compiler.backend.plan import plan_from_graph
from emmy.compiler.pipeline.search.pins import pinned_knobs
from emmy.compiler.target import set_target
from emmy.compiler.trace.torch import trace_module
from emmy.serving.onecat import _RmsNormAdapter, _RmsNormModule


def test_rms_norm_program_matches_reference_and_lowers_tuned_sm70_schedule():
    x = torch.randn((1, 4096), dtype=torch.float16)
    weight = torch.randn((4096,), dtype=torch.float16)
    actual = _RmsNormModule()(x, weight)

    x_fp32 = x.float()
    expected = (x_fp32 * torch.rsqrt((x_fp32 * x_fp32).mean(dim=-1, keepdim=True) + 1e-6)).half() * weight
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    graph = trace_module(_RmsNormModule(), (x, weight))
    try:
        set_target((7, 0))
        with pinned_knobs({"WORK": "t256", "REDUCE": "coop"}):
            plan = plan_from_graph(CudaBackend().compile(graph))
    finally:
        set_target(None)
    assert len(plan.launches) == 1
    assert plan.inputs == ["x", "weight"]
    assert len(plan.outputs) == 1
    assert plan.launches[0].block == ((256,), (1,), (1,))


def test_rms_norm_adapter_routes_unsupported_calls_to_original():
    calls = []

    def original(*args):
        calls.append(args)
        return "original"

    adapter = _RmsNormAdapter(original)
    layer = SimpleNamespace(
        weight=SimpleNamespace(data=torch.empty((4096,), dtype=torch.float16)),
        variance_epsilon=1e-6,
        variance_size_override=None,
    )
    x = torch.empty((1, 4096), dtype=torch.float16)

    assert adapter(layer, x) == "original"
    assert calls == [(layer, x, None)]
