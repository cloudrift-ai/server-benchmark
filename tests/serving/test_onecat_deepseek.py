import sys
from contextlib import nullcontext
from types import SimpleNamespace

import torch

import emmy.serving.onecat_deepseek as onecat_deepseek
from emmy.serving.onecat_deepseek import (
    _build_inverse_rope_program,
    _build_qkv_program,
    _ExternalProgram,
    _FusedQKvRmsNormAdapter,
    _InverseRopeAdapter,
    _run_external,
    _symbolic_profile,
    register_onecat_deepseek_kernels,
)


def _qkv_inputs(rows=1):
    fused = torch.randn((rows, 1536), dtype=torch.float16)
    qr, kv = fused.split((1024, 512), dim=-1)
    return qr, kv, torch.ones((1024,), dtype=torch.float16), torch.ones((512,), dtype=torch.float16)


def _qkv_program(_rows, prepare_rows=None):
    return _ExternalProgram(
        SimpleNamespace(),
        ("fused_q_kv", "q_weight", "kv_weight"),
        ("q_output", "kv_output"),
        prepare_rows,
    )


def test_external_launch_passes_the_torch_stream_object_to_cupy(monkeypatch):
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
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: stream)
    import emmy.compiler.backend.gpu_lock as gpu_lock_module

    monkeypatch.setattr(gpu_lock_module, "gpu_lock", nullcontext)
    runtime = SimpleNamespace(run_once_external=lambda arrays: seen.append(arrays))
    program = _ExternalProgram(runtime, ("input",), ("output",))

    _run_external(program, (("input", "x"), ("output", "y")), "cuda:0")

    assert seen == [stream, {"input": "x", "output": "y"}]


def test_register_patches_both_qkv_aliases_and_inverse_idempotently():
    def qkv_original(qr, kv, q_weight, kv_weight, eps):
        return qr, kv

    def inverse_original(x, positions, cache, rope_dim):
        return x

    ops = SimpleNamespace(fused_q_kv_rmsnorm=qkv_original)
    attention = SimpleNamespace(fused_q_kv_rmsnorm=qkv_original)
    projection = SimpleNamespace(sm70_inverse_rope=inverse_original)

    assert register_onecat_deepseek_kernels(ops, attention, projection)
    qkv_wrapper = ops.fused_q_kv_rmsnorm
    inverse_wrapper = projection.sm70_inverse_rope
    assert attention.fused_q_kv_rmsnorm is qkv_wrapper
    assert qkv_wrapper.__wrapped__ is qkv_original
    assert qkv_wrapper._emmy_original is qkv_original
    assert inverse_wrapper.__wrapped__ is inverse_original
    assert inverse_wrapper._emmy_original is inverse_original

    assert register_onecat_deepseek_kernels(ops, attention, projection)
    assert ops.fused_q_kv_rmsnorm is qkv_wrapper
    assert projection.sm70_inverse_rope is inverse_wrapper

    late_attention = SimpleNamespace(fused_q_kv_rmsnorm=qkv_original)
    assert register_onecat_deepseek_kernels(ops, late_attention, projection)
    assert late_attention.fused_q_kv_rmsnorm is qkv_wrapper


def test_qkv_rejects_separate_storage_before_build(monkeypatch):
    monkeypatch.setattr(onecat_deepseek, "_is_exact_sm70", lambda _tensor: True)
    monkeypatch.setattr(onecat_deepseek, "_is_capturing", lambda: False)
    calls = []

    def original(*args):
        calls.append(args)
        return args[0], args[1]

    def unexpected_build(_rows):
        raise AssertionError("storage rejection must happen before build")

    adapter = _FusedQKvRmsNormAdapter(original, program_builder=unexpected_build)
    q_backing = torch.randn((1, 1536), dtype=torch.float16)
    kv_backing = torch.randn((1, 1536), dtype=torch.float16)
    qr = q_backing[:, :1024]
    kv = kv_backing[:, 1024:]
    q_weight = torch.ones((1024,), dtype=torch.float16)
    kv_weight = torch.ones((512,), dtype=torch.float16)

    assert adapter(qr, kv, q_weight, kv_weight, 1e-6) == (qr, kv)
    assert len(calls) == 1


def test_qkv_binds_both_outputs_in_return_order_and_shares_the_capacity_program(monkeypatch):
    monkeypatch.setattr(onecat_deepseek, "_is_exact_sm70", lambda _tensor: True)
    monkeypatch.setattr(onecat_deepseek, "_is_capturing", lambda: False)
    builds = []
    references = []
    seen_names = []
    prepared_rows = []

    def builder(rows):
        builds.append(rows)
        return _qkv_program(rows, prepared_rows.append)

    def original(qr, kv, _q_weight, _kv_weight, _eps):
        references.append(qr.shape[0])
        return qr.contiguous(), kv.contiguous()

    def runner(_program, bindings, _device):
        seen_names.append(tuple(name for name, _tensor in bindings))
        values = dict(bindings)
        fused = values["fused_q_kv"]
        values["q_output"].copy_(fused[:, :1024])
        values["kv_output"].copy_(fused[:, 1024:])
        assert values["q_output"].is_contiguous()
        assert values["kv_output"].is_contiguous()

    adapter = _FusedQKvRmsNormAdapter(original, program_builder=builder, runner=runner)
    for rows in (1, 2):
        qr, kv, q_weight, kv_weight = _qkv_inputs(rows)
        actual_q, actual_kv = adapter(qr, kv, q_weight, kv_weight, 1e-6)
        torch.testing.assert_close(actual_q, qr, rtol=0, atol=0)
        torch.testing.assert_close(actual_kv, kv, rtol=0, atol=0)

    assert builds == [1]
    assert references == [1, 2]
    assert prepared_rows == [1, 2]
    assert seen_names == [
        ("fused_q_kv", "q_weight", "kv_weight", "q_output", "kv_output"),
        ("fused_q_kv", "q_weight", "kv_weight", "q_output", "kv_output"),
    ]

    qr, kv, q_weight, kv_weight = _qkv_inputs(1)
    adapter(qr, kv, q_weight, kv_weight, 1e-6)
    assert builds == [1]
    assert references == [1, 2]
    assert prepared_rows == [1, 2, 1]


def test_symbolic_profile_covers_every_positive_serving_width_and_rejects_over_capacity():
    assert _symbolic_profile(0) is None
    assert _symbolic_profile(1) == _symbolic_profile(17) == _symbolic_profile(4096) == 4096
    assert _symbolic_profile(4097) is None


def test_symbolic_builders_trace_one_capacity_graph_and_rebind_runtime_rows(monkeypatch):
    import emmy.serving.deepseek as deepseek
    import emmy.serving.external as external

    qkv_graph = object()
    inverse_graph = object()
    runtime_rows = {"qkv": [], "inverse": []}
    qkv_runtime = SimpleNamespace(set_sym_values=lambda values: runtime_rows["qkv"].append(values))
    inverse_runtime = SimpleNamespace(set_sym_values=lambda values: runtime_rows["inverse"].append(values))
    traces = []
    builds = []

    def trace_qkv(**kwargs):
        traces.append(("qkv", kwargs))
        return qkv_graph

    def trace_inverse(**kwargs):
        traces.append(("inverse", kwargs))
        return inverse_graph

    def build(graph, *, symbolic_values):
        builds.append((graph, symbolic_values))
        if graph is qkv_graph:
            return qkv_runtime, SimpleNamespace(inputs=("fused_q_kv", "q_weight", "kv_weight"), outputs=("q", "kv"))
        return inverse_runtime, SimpleNamespace(inputs=("x", "positions", "cos_sin_cache"), outputs=("output",))

    monkeypatch.setattr(deepseek, "trace_fused_q_kv_rmsnorm", trace_qkv)
    monkeypatch.setattr(deepseek, "trace_inverse_rope", trace_inverse)
    monkeypatch.setattr(external, "load_external_program", build)

    qkv = _build_qkv_program(1)
    inverse = _build_inverse_rope_program(128)
    assert traces == [
        ("qkv", {"rows": 4096, "dynamic": True}),
        ("inverse", {"rows": 4096, "dynamic": True}),
    ]
    assert builds == [(qkv_graph, {"num_tokens": 4096}), (inverse_graph, {"num_tokens": 4096})]

    assert qkv.prepare_rows is not None and inverse.prepare_rows is not None
    qkv.prepare_rows(17)
    inverse.prepare_rows(4096)
    assert runtime_rows == {"qkv": [{"num_tokens": 17}], "inverse": [{"num_tokens": 4096}]}


def test_qkv_cold_or_built_but_unverified_capture_falls_back(monkeypatch):
    monkeypatch.setattr(onecat_deepseek, "_is_exact_sm70", lambda _tensor: True)
    capturing = True
    monkeypatch.setattr(onecat_deepseek, "_is_capturing", lambda: capturing)
    builds = []
    launches = []
    references = []

    def builder(rows):
        builds.append(rows)
        return _qkv_program(rows)

    def original(qr, kv, _q_weight, _kv_weight, _eps):
        references.append(qr.shape[0])
        return qr, kv

    adapter = _FusedQKvRmsNormAdapter(
        original,
        program_builder=builder,
        runner=lambda *_args: launches.append(True),
    )
    args = _qkv_inputs()

    adapter(*args, 1e-6)
    assert builds == [] and launches == [] and references == [1]

    capturing = False
    assert adapter.cache.get(1, capturing=False) is not None
    capturing = True
    adapter(*args, 1e-6)
    assert builds == [1] and launches == [] and references == [1, 1]


def test_qkv_capture_requires_first_use_parity_for_the_exact_runtime_width(monkeypatch):
    monkeypatch.setattr(onecat_deepseek, "_is_exact_sm70", lambda _tensor: True)
    capturing = False
    monkeypatch.setattr(onecat_deepseek, "_is_capturing", lambda: capturing)
    launches = []
    references = []

    def original(qr, kv, _q_weight, _kv_weight, _eps):
        references.append(qr.shape[0])
        return qr.contiguous(), kv.contiguous()

    def runner(_program, bindings, _device):
        values = dict(bindings)
        launches.append(values["fused_q_kv"].shape[0])
        values["q_output"].copy_(values["fused_q_kv"][:, :1024])
        values["kv_output"].copy_(values["fused_q_kv"][:, 1024:])

    adapter = _FusedQKvRmsNormAdapter(original, program_builder=_qkv_program, runner=runner)
    adapter(*_qkv_inputs(1), 1e-6)
    capturing = True
    adapter(*_qkv_inputs(1), 1e-6)
    adapter(*_qkv_inputs(2), 1e-6)

    assert launches == [1, 1]
    assert references == [1, 2]


def test_qkv_parity_mismatch_permanently_latches_fallback(monkeypatch):
    monkeypatch.setattr(onecat_deepseek, "_is_exact_sm70", lambda _tensor: True)
    monkeypatch.setattr(onecat_deepseek, "_is_capturing", lambda: False)
    builds = []
    launches = []
    references = []

    def builder(rows):
        builds.append(rows)
        return _qkv_program(rows)

    def original(qr, kv, _q_weight, _kv_weight, _eps):
        references.append(qr.shape[0])
        return torch.ones_like(qr).contiguous(), torch.ones_like(kv).contiguous()

    def runner(_program, bindings, _device):
        launches.append(True)
        values = dict(bindings)
        values["q_output"].zero_()
        values["kv_output"].zero_()

    adapter = _FusedQKvRmsNormAdapter(original, program_builder=builder, runner=runner)
    args = _qkv_inputs()
    for _ in range(2):
        q_out, kv_out = adapter(*args, 1e-6)
        assert torch.equal(q_out, torch.ones_like(q_out))
        assert torch.equal(kv_out, torch.ones_like(kv_out))

    assert builds == [1]
    assert launches == [True]
    assert references == [1, 1]


def test_inverse_rope_binds_one_external_output_and_activates_after_parity(monkeypatch):
    monkeypatch.setattr(onecat_deepseek, "_inverse_rope_supported", lambda *_args: True)
    monkeypatch.setattr(onecat_deepseek, "_is_capturing", lambda: False)
    references = []
    seen_names = []

    def original(x, _positions, _cache, _rope_dim):
        references.append(True)
        return x + 1

    def builder(_rows):
        return _ExternalProgram(SimpleNamespace(), ("x", "positions", "cache"), ("output",))

    def runner(_program, bindings, _device):
        seen_names.append(tuple(name for name, _tensor in bindings))
        values = dict(bindings)
        values["output"].copy_(values["x"] + 1)

    adapter = _InverseRopeAdapter(original, program_builder=builder, runner=runner)
    x = torch.randn((1, 2, 4), dtype=torch.float16)
    positions = torch.zeros((1,), dtype=torch.int64)
    cache = torch.zeros((1, 4), dtype=torch.float32)

    first = adapter(x, positions, cache, 4)
    second = adapter(x, positions, cache, 4)
    torch.testing.assert_close(first, x + 1, rtol=0, atol=0)
    torch.testing.assert_close(second, x + 1, rtol=0, atol=0)
    assert references == [True]
    assert seen_names == [
        ("x", "positions", "cache", "output"),
        ("x", "positions", "cache", "output"),
    ]
