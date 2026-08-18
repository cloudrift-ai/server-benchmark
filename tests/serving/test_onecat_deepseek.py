from types import SimpleNamespace

import torch

import emmy.serving.onecat_deepseek as onecat_deepseek
from emmy.serving.onecat_deepseek import (
    _ExternalProgram,
    _FusedQKvRmsNormAdapter,
    _InverseRopeAdapter,
    register_onecat_deepseek_kernels,
)


def _qkv_inputs(rows=1):
    fused = torch.randn((rows, 1536), dtype=torch.float16)
    qr, kv = fused.split((1024, 512), dim=-1)
    return qr, kv, torch.ones((1024,), dtype=torch.float16), torch.ones((512,), dtype=torch.float16)


def _qkv_program(_rows):
    return _ExternalProgram(
        SimpleNamespace(),
        ("fused_q_kv", "q_weight", "kv_weight"),
        ("q_output", "kv_output"),
    )


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


def test_qkv_binds_both_outputs_in_return_order_and_caches_each_width(monkeypatch):
    monkeypatch.setattr(onecat_deepseek, "_is_exact_sm70", lambda _tensor: True)
    monkeypatch.setattr(onecat_deepseek, "_is_capturing", lambda: False)
    builds = []
    references = []
    seen_names = []

    def builder(rows):
        builds.append(rows)
        return _qkv_program(rows)

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

    assert builds == [1, 2]
    assert references == [1, 2]
    assert seen_names == [
        ("fused_q_kv", "q_weight", "kv_weight", "q_output", "kv_output"),
        ("fused_q_kv", "q_weight", "kv_weight", "q_output", "kv_output"),
    ]

    qr, kv, q_weight, kv_weight = _qkv_inputs(1)
    adapter(qr, kv, q_weight, kv_weight, 1e-6)
    assert builds == [1, 2]
    assert references == [1, 2]


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
