import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest
import torch

from emmy.compiler.dtype import F16, F32
from emmy.compiler.ir.frontend.ir import LinearOp, MatmulOp
from emmy.compiler.ir.tensor.ir import IndexMapOp
from emmy.serving.onecat_linear import (
    PROFILE_ROWS,
    _linear_graph,
    _LinearAdapter,
    _LinearProfile,
    _ProgramEntry,
    _run_external,
    _wrapper,
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


def _tensors(profile: _LinearProfile, *, rows=None, device="cpu"):
    rows = profile.rows if rows is None else rows
    x = torch.empty((rows, 4096), dtype=torch.float16, device=device)
    weight = torch.empty((profile.width, 4096), dtype=torch.float16, device=device)
    output_dtype = torch.float32 if profile.output_fp32 else torch.float16
    return x, weight, output_dtype


def _adapter(*, build_program=None, run_program=None, capturing=None, reference_value=1, original_returns_none=False):
    calls = []

    def original(x, weight, output_dtype):
        calls.append((x, weight, output_dtype))
        if original_returns_none:
            return None
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


@pytest.mark.parametrize("rows", (2, 4, 8, 16, 128, 1024, 4096))
def test_static_graphs_preserve_wide_projection_accumulation_contracts(rows):
    fp32_profile = _LinearProfile(512, True, rows)
    fp32_graph = _linear_graph(fp32_profile)
    fp32_output = fp32_graph.nodes[fp32_graph.outputs[0]]
    assert isinstance(fp32_output.op, MatmulOp)
    assert fp32_output.output.shape == (rows, 512)
    assert fp32_output.output.dtype == F32

    fp16_profile = _LinearProfile(64, False, rows)
    fp16_graph = _linear_graph(fp16_profile)
    fp16_output = fp16_graph.nodes[fp16_graph.outputs[0]]
    assert isinstance(fp16_output.op, LinearOp)
    assert fp16_output.output.shape == (rows, 64)
    assert fp16_output.output.dtype == F16


@pytest.mark.parametrize(
    ("profile", "symbolic"),
    (
        (_LinearProfile(256, True), False),
        (_LinearProfile(256, True, 2), False),
        (_LinearProfile(256, True, 4096, symbolic=True), True),
    ),
)
def test_router_graph_preserves_m1_fp32_accumulation_and_wide_fp16_accumulation(profile, symbolic):
    graph = _linear_graph(profile)
    output = graph.nodes[graph.outputs[0]]
    assert output.output.dtype == F32
    if profile.rows == 1:
        assert isinstance(output.op, MatmulOp)
    else:
        assert any(isinstance(node.op, LinearOp) and node.output.dtype == F16 for node in graph.nodes.values())
        assert isinstance(output.op, IndexMapOp)
    expected_rows = "num_tokens" if symbolic else profile.rows
    assert str(output.output.shape[0]) == str(expected_rows)


@pytest.mark.parametrize("profile", _PROFILES)
def test_adapter_accepts_only_the_pinned_contiguous_profiles(profile):
    adapter, _ = _adapter()
    for rows in PROFILE_ROWS:
        expected = _LinearProfile(profile.width, profile.output_fp32, rows)
        assert adapter._profile(*_tensors(profile, rows=rows, device="meta")) == expected

    symbolic = _LinearProfile(profile.width, profile.output_fp32, 4096, symbolic=True)
    for rows in (3, 5, 7, 9, 15, 17, 63, 127, 129, 257, 4095):
        assert adapter._profile(*_tensors(profile, rows=rows, device="meta")) == symbolic

    x, weight, output_dtype = _tensors(profile, device="meta")
    assert adapter._profile(x.expand(2, -1), weight, output_dtype) is None
    assert adapter._profile(x, weight[:, ::2], output_dtype) is None
    wrong_dtype = torch.float16 if profile.output_fp32 else torch.float32
    assert adapter._profile(x, weight, wrong_dtype) is None
    for rows in (0, 4097):
        assert adapter._profile(*_tensors(profile, rows=rows, device="meta")) is None


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


def test_wide_router_parity_uses_fp16_linear_then_fp32_cast_when_helper_returns_none():
    profile = _LinearProfile(256, True, 2)
    adapter, calls = _adapter(reference_value=0, original_returns_none=True)
    x, weight, output_dtype = _tensors(profile)
    x.zero_()
    weight.zero_()

    output = adapter.dispatch(x, weight, output_dtype)

    assert output.shape == (2, 256)
    assert output.dtype == torch.float32
    assert torch.count_nonzero(output) == 0
    assert len(calls) == 1
    assert adapter._programs[profile].verified


def test_arbitrary_prefill_uses_one_verified_symbolic_capacity_program():
    built = []

    def builder(profile):
        built.append(profile)
        return _ProgramEntry(object(), ("x", "weight"), "output", profile)

    adapter, calls = _adapter(build_program=builder)
    base = _PROFILES[0]
    first = _tensors(base, rows=17)
    second = _tensors(base, rows=257)

    assert adapter.dispatch(*first).shape == (17, 64)
    assert adapter.dispatch(*second).shape == (257, 64)

    symbolic = _LinearProfile(64, False, 4096, symbolic=True)
    assert built == [symbolic]
    assert adapter._programs[symbolic].verified
    assert len(calls) == 1


def test_profiled_prefill_also_realizes_and_verifies_symbolic_sibling():
    built = []
    runs = []

    def builder(profile):
        built.append(profile)
        return _ProgramEntry(object(), ("x", "weight"), "output", profile)

    def runner(entry, x, weight, output):
        runs.append(entry.profile)
        output.fill_(1)

    adapter, calls = _adapter(build_program=builder, run_program=runner)
    profile = _LinearProfile(64, False, 128)
    output = adapter.dispatch(*_tensors(profile))

    symbolic = _LinearProfile(64, False, 4096, symbolic=True)
    assert output.shape == (128, 64)
    assert built == [profile, symbolic]
    assert runs == [profile, symbolic]
    assert adapter._programs[profile].verified
    assert adapter._programs[symbolic].verified
    assert len(calls) == 1


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


def test_symbolic_runtime_width_is_set_inside_the_torch_stream_context(monkeypatch):
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
            return ExternalStream()

    cupy = ModuleType("cupy")
    cupy.cuda = SimpleNamespace(Stream=Stream)
    cupy.from_dlpack = lambda tensor: tensor.name
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: torch_stream)

    from emmy.compiler.backend import gpu_lock as gpu_lock_module

    monkeypatch.setattr(gpu_lock_module, "gpu_lock", nullcontext)

    class Runtime:
        def set_sym_values(self, values):
            assert active
            assert values == {"num_tokens": 17}
            events.append("set_sym_values")

        def run_once_external(self, bindings):
            assert active
            events.append("run")

    profile = _LinearProfile(64, False, 4096, symbolic=True)
    entry = _ProgramEntry(Runtime(), ("x", "weight"), "output", profile)
    device = object()
    _run_external(
        entry,
        SimpleNamespace(name="x", device=device, shape=(17, 4096)),
        SimpleNamespace(name="weight", device=device),
        SimpleNamespace(name="output", device=device),
    )

    assert events == ["stream_enter", "set_sym_values", "run", "stream_exit"]


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


def test_wrapper_bypasses_the_custom_op_for_a_cold_capture():
    adapter, calls = _adapter(capturing=lambda: True, original_returns_none=True)
    op_calls = []

    def op(x, weight, output_fp32):
        op_calls.append((x, weight, output_fp32))
        return x.new_empty((x.shape[0], weight.shape[0]), dtype=torch.float32 if output_fp32 else torch.float16)

    replacement = _wrapper(adapter, op)
    profile = _LinearProfile(256, True, 2)
    args = _tensors(profile)

    assert replacement(*args) is None
    assert len(calls) == 1
    assert op_calls == []

    adapter._programs[profile] = _ProgramEntry(object(), ("x", "weight"), "output", profile, verified=True)
    output = replacement(*args)
    assert output.shape == (2, 256)
    assert len(calls) == 1
    assert len(op_calls) == 1


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
