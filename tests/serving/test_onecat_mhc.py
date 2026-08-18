import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest
import torch

from emmy.serving.onecat_mhc import (
    _PLAN_INPUTS,
    _SYMBOLS,
    _MhcFamilyAdapter,
    _ProgramEntry,
    _ProgramProfile,
    _run_external,
    register_onecat_mhc_kernels,
)


def _broadcast(
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    n_splits=1,
    norm_weight=None,
    norm_eps=1e-6,
    fn_broadcast=None,
):
    return "broadcast"


def _pre(
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    n_splits=1,
    norm_weight=None,
    norm_eps=1e-6,
):
    return "pre"


def _fused(
    x,
    residual,
    post_layer_mix,
    comb_res_mix,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    n_splits=1,
    tile_n=1,
    norm_weight=None,
    norm_eps=1e-6,
):
    return "fused"


def _post(x, residual, post_layer_mix, comb_res_mix):
    return "post"


def _head(hs_flat, fn, hc_scale, hc_base, rms_eps, hc_eps):
    return "head"


def _model_module() -> ModuleType:
    module = ModuleType("test_onecat_model")
    for kind, function in {
        "broadcast": _broadcast,
        "pre": _pre,
        "fused": _fused,
        "post": _post,
        "head": _head,
    }.items():
        setattr(module, _SYMBOLS[kind], function)
    return module


def _args(kind: str, rows: int = 1) -> tuple:
    x = torch.empty((rows, 4096), dtype=torch.float16)
    residual = torch.empty((rows, 4, 4096), dtype=torch.float16)
    post = torch.empty((rows, 4, 1), dtype=torch.float32)
    comb = torch.empty((rows, 4, 4), dtype=torch.float32)
    fn = torch.empty((24, 16384), dtype=torch.float32)
    scale = torch.empty((3,), dtype=torch.float32)
    base = torch.empty((24,), dtype=torch.float32)
    norm = torch.empty((4096,), dtype=torch.float16)
    if kind == "broadcast":
        fn_broadcast = torch.empty((24, 4096), dtype=torch.float32)
        return (x, fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 20, 1, norm, 1e-6, fn_broadcast)
    if kind == "pre":
        return (residual, fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 20, 1, norm, 1e-6)
    if kind == "fused":
        return (x, residual, post, comb, fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 20, 1, 1, norm, 1e-6)
    if kind == "post":
        return x, residual, post, comb
    head_fn = torch.empty((4, 16384), dtype=torch.float32)
    head_scale = torch.empty((1,), dtype=torch.float32)
    head_base = torch.empty((4,), dtype=torch.float32)
    return residual, head_fn, head_scale, head_base, 1e-6, 1e-6


def _filled_outputs(kind: str, args: tuple, *, offset: float = 1.0):
    outputs = _MhcFamilyAdapter._outputs(kind, args)
    for index, output in enumerate(outputs):
        output.fill_(offset + index)
    return outputs if len(outputs) != 1 else outputs[0]


def _adapter(*, build_program=None, run_program=None, is_capturing=None):
    calls = {kind: 0 for kind in _SYMBOLS}

    def original(kind, args):
        calls[kind] += 1
        return _filled_outputs(kind, args)

    originals = {kind: lambda *args, kind=kind: original(kind, args) for kind in _SYMBOLS}

    def default_builder(profile):
        output_count = 4 if profile.kind in ("broadcast", "fused") else 3 if profile.kind == "pre" else 1
        return _ProgramEntry(object(), _PLAN_INPUTS[profile.kind], tuple(f"output_{index}" for index in range(output_count)), profile)

    def default_runner(entry, inputs, outputs):
        for index, output in enumerate(outputs):
            output.fill_(index + 1)

    adapter = _MhcFamilyAdapter(
        originals,
        build_program=build_program or default_builder,
        run_program=run_program or default_runner,
        platform_supported=lambda tensors: True,
        is_capturing=is_capturing or (lambda: False),
    )
    return adapter, calls


def test_external_launch_creates_dlpack_views_inside_the_torch_stream(monkeypatch):
    stream = object()
    seen = []
    active = False

    class ExternalStream:
        def __enter__(self):
            nonlocal active
            active = True

        def __exit__(self, *_args):
            nonlocal active
            active = False

    def from_dlpack(tensor):
        assert active
        return tensor

    fake_cupy = SimpleNamespace(
        from_dlpack=from_dlpack,
        cuda=SimpleNamespace(Stream=SimpleNamespace(from_external=lambda value: seen.append(value) or ExternalStream())),
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    import emmy.compiler.backend.gpu_lock as gpu_lock_module

    monkeypatch.setattr(gpu_lock_module, "gpu_lock", nullcontext)
    runtime = SimpleNamespace(run_once_external=lambda arrays: seen.append(arrays))
    entry = _ProgramEntry(runtime, ("input",), ("output",), _ProgramProfile("post", 1))

    _run_external(entry, ("x",), ("y",))

    assert seen == [stream, {"input": "x", "output": "y"}]


def test_register_patches_every_direct_alias_all_or_none_and_is_idempotent():
    module = _model_module()
    originals = {kind: getattr(module, symbol) for kind, symbol in _SYMBOLS.items()}

    assert register_onecat_mhc_kernels(module)
    replacements = {kind: getattr(module, symbol) for kind, symbol in _SYMBOLS.items()}
    assert all(replacements[kind] is not originals[kind] for kind in _SYMBOLS)
    assert all(replacements[kind]._emmy_onecat_mhc_original is originals[kind] for kind in _SYMBOLS)

    assert register_onecat_mhc_kernels(module)
    assert {kind: getattr(module, symbol) for kind, symbol in _SYMBOLS.items()} == replacements

    incompatible = _model_module()
    incompatible.mhc_pre_tilelang = lambda residual: residual
    before = {symbol: getattr(incompatible, symbol) for symbol in _SYMBOLS.values()}
    assert not register_onecat_mhc_kernels(incompatible)
    assert {symbol: getattr(incompatible, symbol) for symbol in _SYMBOLS.values()} == before


@pytest.mark.parametrize("kind", tuple(_SYMBOLS))
def test_first_use_verifies_every_output_in_live_order(kind):
    seen = []

    def runner(entry, inputs, outputs):
        seen.append((entry.profile, inputs, outputs))
        for index, output in enumerate(outputs):
            output.fill_(index + 1)

    adapter, calls = _adapter(run_program=runner)
    args = _args(kind)
    actual = adapter.dispatch(kind, *args)
    expected = _MhcFamilyAdapter._tuple(_filled_outputs(kind, args))
    actual = _MhcFamilyAdapter._tuple(actual)

    assert calls[kind] == 1
    assert len(actual) == len(expected)
    for index, (output, reference) in enumerate(zip(actual, expected, strict=True), start=1):
        assert output.shape == reference.shape
        assert output.dtype == reference.dtype
        assert torch.count_nonzero(output != index) == 0
    assert seen[0][0] == _ProgramProfile(kind, 1)
    assert seen[0][1] == _MhcFamilyAdapter._program_inputs(kind, args)

    adapter.dispatch(kind, *args)
    assert calls[kind] == 1


def test_exact_scalar_and_tensor_contract_failures_call_the_original():
    built = []

    def builder(profile):
        built.append(profile)
        raise AssertionError("unsupported calls must not build")

    adapter, calls = _adapter(build_program=builder)
    args = list(_args("fused"))
    args[13] = 2
    assert adapter.dispatch("fused", *args)[0].shape == (1, 4, 4096)
    args[13] = 1
    args[11] = object()
    assert adapter.dispatch("fused", *args)[0].shape == (1, 4, 4096)
    args = list(_args("head"))
    args[1] = args[1].half()
    assert adapter.dispatch("head", *args).shape == (1, 4096)

    assert calls["fused"] == 2
    assert calls["head"] == 1
    assert built == []


def test_build_failure_and_parity_mismatch_permanently_fall_back():
    build_calls = 0

    def failing_builder(profile):
        nonlocal build_calls
        build_calls += 1
        raise RuntimeError("compile failed")

    adapter, calls = _adapter(build_program=failing_builder)
    args = _args("post")
    adapter.dispatch("post", *args)
    adapter.dispatch("post", *args)
    assert build_calls == 1
    assert calls["post"] == 2
    assert adapter._disabled

    run_calls = 0

    def mismatching_runner(entry, inputs, outputs):
        nonlocal run_calls
        run_calls += 1
        for output in outputs:
            output.zero_()

    adapter, calls = _adapter(run_program=mismatching_runner)
    adapter.dispatch("broadcast", *_args("broadcast"))
    adapter.dispatch("broadcast", *_args("broadcast"))
    assert run_calls == 1
    assert calls["broadcast"] == 2
    assert adapter._disabled


def test_capture_uses_only_a_preverified_program():
    built = []
    runs = []
    capture = True

    def builder(profile):
        built.append(profile)
        return _ProgramEntry(object(), _PLAN_INPUTS[profile.kind], ("output",), profile)

    def runner(entry, inputs, outputs):
        runs.append(entry)
        outputs[0].fill_(1)

    adapter, calls = _adapter(build_program=builder, run_program=runner, is_capturing=lambda: capture)
    args = _args("post")
    adapter.dispatch("post", *args)
    assert built == [] and runs == [] and calls["post"] == 1

    profile = _ProgramProfile("post", 1)
    adapter._programs[profile] = _ProgramEntry(object(), _PLAN_INPUTS["post"], ("output",), profile)
    adapter.dispatch("post", *args)
    assert runs == [] and calls["post"] == 2

    adapter._programs[profile].verified = True
    output = adapter.dispatch("post", *args)
    assert output.shape == (1, 4, 4096)
    assert len(runs) == 1 and calls["post"] == 2


def test_prefill_profiles_include_exact_rows_and_a_shared_symbolic_capacity():
    built = []

    def builder(profile):
        built.append(profile)
        output_count = 4 if profile.kind == "fused" else 1
        return _ProgramEntry(object(), _PLAN_INPUTS[profile.kind], tuple(f"output_{index}" for index in range(output_count)), profile)

    adapter, _ = _adapter(build_program=builder)
    adapter.dispatch("head", *_args("head", rows=17))
    assert built == [_ProgramProfile("head", 4096, symbolic=True)]
    assert adapter._programs[built[0]].verified

    built.clear()
    adapter, _ = _adapter(build_program=builder)
    adapter.dispatch("fused", *_args("fused", rows=128))
    assert built == [_ProgramProfile("fused", 128), _ProgramProfile("fused", 4096, symbolic=True)]
    assert all(adapter._programs[profile].verified for profile in built)
