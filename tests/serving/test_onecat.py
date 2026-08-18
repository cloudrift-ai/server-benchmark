from types import SimpleNamespace

import torch

from emmy.compiler.backend.cuda.backend import CudaBackend
from emmy.compiler.backend.plan import plan_from_graph
from emmy.compiler.dim import Dim
from emmy.compiler.pipeline.search.pins import pinned_knobs
from emmy.compiler.target import set_target
from emmy.compiler.trace.torch import trace_module
from emmy.serving.onecat import _MAX_ROWS, _build_rms_norm_program, _RmsNormAdapter, _RmsNormModule, _RmsNormProgram


def test_deepseek_v4_adapter_flag_is_live_and_off_by_default(monkeypatch):
    from emmy import config

    monkeypatch.delenv(config.ONECAT_DEEPSEEK_V4, raising=False)
    assert not config.onecat_deepseek_v4()
    monkeypatch.setenv(config.ONECAT_DEEPSEEK_V4, "1")
    assert config.onecat_deepseek_v4()


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


def test_rms_norm_external_program_is_one_bounded_dynamic_profile(monkeypatch):
    seen = {}

    class Runtime:
        pass

    class Plan:
        launches = [object()]
        inputs = ["x", "weight"]
        outputs = ["output"]

    def build(graph, **kwargs):
        seen["graph"] = graph
        seen["kwargs"] = kwargs
        return Runtime(), Plan()

    import emmy.serving.external as external
    from emmy import config

    monkeypatch.setenv(config.ONECAT_DEEPSEEK_V4, "1")
    monkeypatch.setattr(external, "load_external_program", build)
    program = _build_rms_norm_program()

    assert isinstance(program, _RmsNormProgram)
    assert program.inputs == ("x", "weight")
    assert program.output == "output"
    assert seen["kwargs"] == {
        "pins": {"WORK": "t256", "REDUCE": "coop"},
        "symbolic_values": {"num_tokens": _MAX_ROWS},
    }
    assert seen["graph"].nodes["x"].output.shape == (Dim("num_tokens"), 4096)


def test_standalone_rms_norm_experiment_retains_compile_on_miss(monkeypatch):
    import emmy.serving.external as external
    from emmy import config

    calls = []
    plan = SimpleNamespace(launches=[object()], inputs=["x", "weight"], outputs=["output"])

    monkeypatch.delenv(config.ONECAT_DEEPSEEK_V4, raising=False)
    monkeypatch.setattr(external, "build_external_program", lambda *_args, **_kwargs: (object(), plan))
    monkeypatch.setattr(external, "load_external_program", lambda *_args, **_kwargs: calls.append(True))

    assert isinstance(_build_rms_norm_program(), _RmsNormProgram)
    assert calls == []


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


def test_rms_norm_adapter_rebinds_each_width_and_requires_parity_before_capture(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_args: (7, 0))
    capturing = False
    builds = []
    launches = []
    references = []
    runtime = object()

    def original(layer, x, residual=None):
        references.append(x.shape[0])
        return x + layer.weight.data

    def build():
        builds.append(True)
        return _RmsNormProgram(runtime, ("x", "weight"), "output")

    def run(program, layer, x, output, rows):
        assert program.runtime is runtime
        launches.append(rows)
        output.copy_(x + layer.weight.data)

    adapter = _RmsNormAdapter(original, build_program=build, run_program=run, is_capturing=lambda: capturing)
    adapter._supported = lambda _layer, x, _residual: 0 < x.shape[0] <= _MAX_ROWS
    layer = SimpleNamespace(
        weight=SimpleNamespace(data=torch.ones((4096,), dtype=torch.float16)),
        variance_epsilon=1e-6,
        variance_size_override=None,
    )

    first = torch.zeros((1, 4096), dtype=torch.float16)
    assert torch.equal(adapter(layer, first), torch.ones_like(first))
    capturing = True
    assert torch.equal(adapter(layer, first), torch.ones_like(first))

    second = torch.zeros((17, 4096), dtype=torch.float16)
    assert torch.equal(adapter(layer, second), torch.ones_like(second))
    capturing = False
    assert torch.equal(adapter(layer, second), torch.ones_like(second))
    capturing = True
    assert torch.equal(adapter(layer, second), torch.ones_like(second))

    assert builds == [True]
    assert launches == [1, 1, 17, 17]
    assert references == [1, 17, 17]


def test_rms_norm_parity_mismatch_disables_shared_capacity_program(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_args: (7, 0))
    builds = []
    launches = []
    references = []

    def original(_layer, x, _residual=None):
        references.append(x.shape[0])
        return torch.ones_like(x)

    def build():
        builds.append(True)
        return _RmsNormProgram(object(), ("x", "weight"), "output")

    def run(_program, _layer, _x, output, rows):
        launches.append(rows)
        output.zero_()

    adapter = _RmsNormAdapter(original, build_program=build, run_program=run, is_capturing=lambda: False)
    adapter._supported = lambda _layer, x, _residual: 0 < x.shape[0] <= _MAX_ROWS
    layer = SimpleNamespace(
        weight=SimpleNamespace(data=torch.ones((4096,), dtype=torch.float16)),
        variance_epsilon=1e-6,
        variance_size_override=None,
    )
    x = torch.zeros((1, 4096), dtype=torch.float16)

    assert torch.equal(adapter(layer, x), torch.ones_like(x))
    assert torch.equal(adapter(layer, x), torch.ones_like(x))
    assert builds == [True]
    assert launches == [1]
    assert references == [1, 1]
