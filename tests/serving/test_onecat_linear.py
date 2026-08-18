import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest
import torch

from emmy.compiler.dtype import F16, F32
from emmy.compiler.ir.frontend.ir import LinearOp, MatmulOp
from emmy.serving.onecat_linear import (
    _linear_graph,
    _LinearAdapter,
    _LinearProfile,
    _ProgramEntry,
    _run_external,
    register_onecat_linear_kernels,
)

_PROFILES = (
    _LinearProfile(64, False),  # indexer.weights_proj
    _LinearProfile(256, True),  # replicated GateLinear
    _LinearProfile(512, True),  # indexer compressor
    _LinearProfile(1024, True),  # C128 outer compressor
    _LinearProfile(2048, True),  # C4 outer compressor
)


def _modules(*, eligible=True):
    gemv = ModuleType("test_gemv")
    attention = ModuleType("test_attention")
    calls = []

    def maybe_sm70_dsv4_fp16_gemv(x, weight, output_dtype):
        calls.append((x, weight, output_dtype))
        return x.new_full((x.shape[0], weight.shape[0]), 3, dtype=output_dtype)

    def can_use_sm70_dsv4_fp16_gemv(x, weight, output_dtype):
        return eligible

    gemv.maybe_sm70_dsv4_fp16_gemv = maybe_sm70_dsv4_fp16_gemv
    gemv.can_use_sm70_dsv4_fp16_gemv = can_use_sm70_dsv4_fp16_gemv
    attention.maybe_sm70_dsv4_fp16_gemv = maybe_sm70_dsv4_fp16_gemv
    return gemv, attention, calls


def _tensors(profile: _LinearProfile, *, device="cpu"):
    x = torch.empty((1, 4096), dtype=torch.float16, device=device)
    weight = torch.empty((profile.width, 4096), dtype=torch.float16, device=device)
    output_dtype = torch.float32 if profile.output_fp32 else torch.float16
    return x, weight, output_dtype


def _adapter(*, build_program=None, run_program=None, capturing=None, reference_value=1):
    calls = []

    def original(x, weight, output_dtype):
        calls.append((x, weight, output_dtype))
        return x.new_full((x.shape[0], weight.shape[0]), reference_value, dtype=output_dtype)

    def default_builder(profile):
        return _ProgramEntry(object(), ("x", "weight"), "output", profile)

    def default_runner(entry, x, weight, output):
        output.fill_(reference_value)

    adapter = _LinearAdapter(
        original,
        build_program=build_program or default_builder,
        run_program=run_program or default_runner,
        platform_supported=lambda x, weight: True,
        is_capturing=capturing or (lambda: False),
    )
    return adapter, calls


def test_linear_graphs_preserve_the_five_exact_m1_dtype_contracts():
    for profile in _PROFILES:
        graph = _linear_graph(profile)
        assert graph.inputs == ["x", "weight"]
        assert len(graph.outputs) == 1
        assert graph.nodes["x"].output.shape == (1, 4096)
        assert graph.nodes["x"].output.dtype == F16
        assert graph.nodes["weight"].output.shape == (profile.width, 4096)
        assert graph.nodes["weight"].output.dtype == F16
        output = graph.nodes[graph.outputs[0]]
        assert isinstance(output.op, MatmulOp if profile.output_fp32 else LinearOp)
        assert output.output.shape == (1, profile.width)
        assert output.output.dtype == (F32 if profile.output_fp32 else F16)


@pytest.mark.parametrize("profile", _PROFILES)
def test_adapter_accepts_only_the_pinned_contiguous_profiles(profile):
    adapter, _ = _adapter()
    assert adapter._profile(*_tensors(profile, device="meta")) == profile

    x, weight, output_dtype = _tensors(profile, device="meta")
    assert adapter._profile(x.expand(2, -1), weight, output_dtype) is None
    assert adapter._profile(x, weight[:, ::2], output_dtype) is None
    wrong_dtype = torch.float16 if profile.output_fp32 else torch.float32
    assert adapter._profile(x, weight, wrong_dtype) is None


def test_first_use_parity_latches_and_reuses_caller_owned_output():
    runs = []

    def runner(entry, x, weight, output):
        runs.append((entry, x, weight, output))
        output.fill_(1)

    adapter, calls = _adapter(run_program=runner)
    x, weight, output_dtype = _tensors(_PROFILES[0])

    output = adapter.dispatch(x, weight, output_dtype)
    assert output.shape == (1, 64)
    assert output.dtype == torch.float16
    assert torch.count_nonzero(output != 1) == 0
    assert len(calls) == 1
    assert runs[0][3] is output
    assert adapter._programs[_PROFILES[0]].verified

    adapter.dispatch(x, weight, output_dtype)
    assert len(calls) == 1
    assert len(runs) == 2


def test_external_views_are_created_inside_the_torch_stream_context(monkeypatch):
    events = []
    active = False
    torch_stream = object()

    class ExternalStream:
        def __enter__(self):
            nonlocal active
            active = True
            events.append("stream_enter")

        def __exit__(self, exc_type, exc, traceback):
            nonlocal active
            active = False
            events.append("stream_exit")

    class Stream:
        @staticmethod
        def from_external(stream):
            assert stream is torch_stream
            events.append("from_external")
            return ExternalStream()

    cupy = ModuleType("cupy")
    cupy.cuda = SimpleNamespace(Stream=Stream)

    def from_dlpack(tensor):
        assert active
        events.append(f"dlpack:{tensor.name}")
        return tensor.name

    cupy.from_dlpack = from_dlpack
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: events.append("current_stream") or torch_stream)

    from emmy.compiler.backend import gpu_lock as gpu_lock_module

    monkeypatch.setattr(gpu_lock_module, "gpu_lock", nullcontext)

    class Runtime:
        def run_once_external(self, bindings):
            assert active
            assert bindings == {"x": "x", "weight": "weight", "output": "output"}
            events.append("run")

    profile = _LinearProfile(64, False)
    entry = _ProgramEntry(Runtime(), ("x", "weight"), "output", profile)
    device = object()
    _run_external(
        entry,
        SimpleNamespace(name="x", device=device),
        SimpleNamespace(name="weight", device=device),
        SimpleNamespace(name="output", device=device),
    )

    assert events == [
        "current_stream",
        "from_external",
        "stream_enter",
        "dlpack:x",
        "dlpack:weight",
        "dlpack:output",
        "run",
        "stream_exit",
    ]


def test_build_failure_and_mismatch_permanently_fall_back_per_profile():
    build_calls = 0

    def failing_builder(profile):
        nonlocal build_calls
        build_calls += 1
        raise RuntimeError("compile failed")

    adapter, calls = _adapter(build_program=failing_builder)
    args = _tensors(_PROFILES[0])
    adapter.dispatch(*args)
    adapter.dispatch(*args)
    assert build_calls == 1
    assert len(calls) == 2
    assert _PROFILES[0] in adapter._disabled

    run_calls = 0

    def mismatching_runner(entry, x, weight, output):
        nonlocal run_calls
        run_calls += 1
        output.zero_()

    adapter, calls = _adapter(run_program=mismatching_runner)
    adapter.dispatch(*args)
    adapter.dispatch(*args)
    assert run_calls == 1
    assert len(calls) == 2
    assert _PROFILES[0] in adapter._disabled


def test_capture_uses_only_a_preverified_program():
    built = []
    runs = []

    def builder(profile):
        built.append(profile)
        return _ProgramEntry(object(), ("x", "weight"), "output", profile)

    def runner(entry, x, weight, output):
        runs.append(entry)
        output.fill_(1)

    adapter, calls = _adapter(build_program=builder, run_program=runner, capturing=lambda: True)
    args = _tensors(_PROFILES[0])
    adapter.dispatch(*args)
    assert built == [] and runs == [] and len(calls) == 1

    adapter._programs[_PROFILES[0]] = _ProgramEntry(object(), ("x", "weight"), "output", _PROFILES[0])
    adapter.dispatch(*args)
    assert runs == [] and len(calls) == 2

    adapter._programs[_PROFILES[0]].verified = True
    output = adapter.dispatch(*args)
    assert output.shape == (1, 64)
    assert len(runs) == 1 and len(calls) == 2


def test_registration_patches_source_and_direct_consumer_all_or_none():
    gemv, attention, calls = _modules()
    original = gemv.maybe_sm70_dsv4_fp16_gemv

    assert register_onecat_linear_kernels(gemv, attention)
    replacement = gemv.maybe_sm70_dsv4_fp16_gemv
    assert replacement is attention.maybe_sm70_dsv4_fp16_gemv
    assert replacement is not original
    assert replacement._emmy_onecat_linear_original is original
    assert register_onecat_linear_kernels(gemv, attention)
    assert gemv.maybe_sm70_dsv4_fp16_gemv is replacement

    x, weight, output_dtype = _tensors(_PROFILES[0])
    output = replacement(x, weight, output_dtype)
    assert output.shape == (1, 64)
    assert torch.count_nonzero(output != 3) == 0
    assert len(calls) == 1

    incompatible_gemv, incompatible_attention, _ = _modules()
    incompatible_attention.maybe_sm70_dsv4_fp16_gemv = lambda x, weight, output_dtype: None
    before = (
        incompatible_gemv.maybe_sm70_dsv4_fp16_gemv,
        incompatible_attention.maybe_sm70_dsv4_fp16_gemv,
    )
    assert not register_onecat_linear_kernels(incompatible_gemv, incompatible_attention)
    assert (
        incompatible_gemv.maybe_sm70_dsv4_fp16_gemv,
        incompatible_attention.maybe_sm70_dsv4_fp16_gemv,
    ) == before
