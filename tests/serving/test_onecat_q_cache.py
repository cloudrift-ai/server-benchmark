from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import emmy.serving.onecat_deepseek as onecat_deepseek
from emmy.serving.onecat_deepseek import (
    _build_qnorm_rope_program,
    _ExternalProgram,
    register_onecat_q_cache_kernel,
)


def _program(_rows):
    return _ExternalProgram(
        SimpleNamespace(),
        ("q", "positions", "cos_sin_cache"),
        ("q_output",),
    )


def _inputs(rows=1):
    return (
        torch.zeros((rows, 8, 512), dtype=torch.float16),
        torch.zeros((rows, 512), dtype=torch.float16),
        torch.zeros((1, 1), dtype=torch.uint8),
        torch.zeros((rows,), dtype=torch.int64),
        torch.zeros((rows,), dtype=torch.int64),
        torch.zeros((1, 64), dtype=torch.float32),
        1e-6,
        256,
    )


def _modules(calls, *, original_result=None, kv_raises=False):
    def original(q, kv, swa_kv_cache, slot_mapping, positions, cos_sin_cache, eps, block_size):  # noqa: ARG001
        calls.append("original")
        return q if original_result is None else original_result

    def kv_insert(kv, swa_kv_cache, slot_mapping, positions, cos_sin_cache, block_size):  # noqa: ARG001
        calls.append("kv")
        if kv_raises:
            raise RuntimeError("KV failure")

    return (
        SimpleNamespace(sm70_qnorm_rope_kv_fp8_insert=original),
        SimpleNamespace(sm70_kv_rope_fp8_insert=kv_insert),
    )


def _runner(calls):
    def run(_program, bindings, _device):
        calls.append("q")
        values = dict(bindings)
        values["q_output"].copy_(values["q"])

    return run


def test_q_cache_hot_path_computes_q_once_and_runs_only_the_kv_shim(monkeypatch):
    monkeypatch.setattr(onecat_deepseek, "_qnorm_rope_supported", lambda *_args: True)
    monkeypatch.setattr(onecat_deepseek, "_is_capturing", lambda: False)
    calls = []
    fused, kv = _modules(calls)

    assert register_onecat_q_cache_kernel(fused, kv, program_builder=_program, runner=_runner(calls))
    replacement = fused.sm70_qnorm_rope_kv_fp8_insert
    first = replacement(*_inputs())
    second = replacement(*_inputs())

    assert torch.count_nonzero(first) == torch.count_nonzero(second) == 0
    assert calls == ["q", "kv", "q", "kv"]
    assert replacement._emmy_onecat_q_cache_original is replacement.__wrapped__
    assert replacement._emmy_onecat_q_cache_kv_insert is kv.sm70_kv_rope_fp8_insert


def test_q_cache_cold_or_unverified_capture_uses_the_complete_original(monkeypatch):
    monkeypatch.setattr(onecat_deepseek, "_qnorm_rope_supported", lambda *_args: True)
    capturing = True
    monkeypatch.setattr(onecat_deepseek, "_is_capturing", lambda: capturing)
    calls = []
    builds = []
    fused, kv = _modules(calls)

    def build(rows):
        builds.append(rows)
        return _program(rows)

    assert register_onecat_q_cache_kernel(fused, kv, program_builder=build, runner=_runner(calls))
    replacement = fused.sm70_qnorm_rope_kv_fp8_insert
    replacement(*_inputs(1))
    assert builds == [] and calls == ["original"]

    capturing = False
    replacement(*_inputs(1))
    capturing = True
    replacement(*_inputs(1))
    replacement(*_inputs(2))
    assert builds == [1]
    assert calls == ["original", "q", "kv", "q", "kv", "original"]


def test_q_cache_kv_failure_never_replays_the_stateful_original(monkeypatch):
    monkeypatch.setattr(onecat_deepseek, "_qnorm_rope_supported", lambda *_args: True)
    monkeypatch.setattr(onecat_deepseek, "_is_capturing", lambda: False)
    calls = []
    fused, kv = _modules(calls, kv_raises=True)
    assert register_onecat_q_cache_kernel(fused, kv, program_builder=_program, runner=_runner(calls))

    replacement = fused.sm70_qnorm_rope_kv_fp8_insert
    with pytest.raises(RuntimeError, match="after cache mutation began"):
        replacement(*_inputs())
    replacement(*_inputs())
    assert calls == ["q", "kv", "original"]


def test_q_cache_invalid_kv_contract_falls_back_before_q_or_cache_work(monkeypatch):
    monkeypatch.setattr(onecat_deepseek, "_qnorm_rope_supported", lambda *_args: True)
    monkeypatch.setattr(onecat_deepseek, "_is_capturing", lambda: False)
    calls = []
    fused, kv = _modules(calls)
    assert register_onecat_q_cache_kernel(fused, kv, program_builder=_program, runner=_runner(calls))
    inputs = list(_inputs(rows=2))
    inputs[1] = torch.zeros((1, 512), dtype=torch.float16)

    fused.sm70_qnorm_rope_kv_fp8_insert(*inputs)

    assert calls == ["original"]


def test_q_cache_registration_is_idempotent_and_rejects_signature_drift():
    calls = []
    fused, kv = _modules(calls)
    assert register_onecat_q_cache_kernel(fused, kv, program_builder=_program)
    replacement = fused.sm70_qnorm_rope_kv_fp8_insert
    assert register_onecat_q_cache_kernel(fused, kv, program_builder=_program)
    assert fused.sm70_qnorm_rope_kv_fp8_insert is replacement

    def incompatible(q, kv):
        return q

    bad = SimpleNamespace(sm70_qnorm_rope_kv_fp8_insert=incompatible)
    assert not register_onecat_q_cache_kernel(bad, kv, program_builder=_program)
    assert bad.sm70_qnorm_rope_kv_fp8_insert is incompatible


def test_qnorm_rope_builder_strictly_loads_one_symbolic_capacity_pack(monkeypatch):
    import emmy.serving.deepseek as deepseek
    import emmy.serving.external as external

    graph = object()
    rows = []
    runtime = SimpleNamespace(set_sym_values=lambda values: rows.append(values))
    seen = []
    monkeypatch.setattr(deepseek, "trace_qnorm_rope", lambda **kwargs: seen.append(("trace", kwargs)) or graph)
    monkeypatch.setattr(
        external,
        "load_external_program",
        lambda value, *, pins, symbolic_values: (
            seen.append(("load", value, pins, symbolic_values))
            or (
                runtime,
                SimpleNamespace(inputs=("q", "positions", "cos_sin_cache"), outputs=("q_out",), launches=(object(),)),
            )
        ),
    )

    program = _build_qnorm_rope_program(1)
    assert seen == [
        ("trace", {"rows": 4096, "dynamic": True}),
        ("load", graph, {"WORK": "t128", "REDUCE": "coop"}, {"num_tokens": 4096}),
    ]
    assert program.prepare_rows is not None
    program.prepare_rows(17)
    assert rows == [{"num_tokens": 17}]


def test_qnorm_rope_builder_rejects_pack_with_more_than_one_launch(monkeypatch):
    import emmy.serving.deepseek as deepseek
    import emmy.serving.external as external

    monkeypatch.setattr(deepseek, "trace_qnorm_rope", lambda **_kwargs: object())
    monkeypatch.setattr(
        external,
        "load_external_program",
        lambda *_args, **_kwargs: (
            object(),
            SimpleNamespace(
                inputs=("q", "positions", "cos_sin_cache"),
                outputs=("q_out",),
                launches=(object(), object()),
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="expected one launch, got 2"):
        _build_qnorm_rope_program(1)


def test_pinned_q_cache_shim_keeps_private_launch_inside_the_vllm_package(project_root):
    project_root = Path(project_root)
    dockerfile = (project_root / "docker/1cat-vllm-sm70/Dockerfile.emmy").read_text()
    shim = (project_root / "docker/1cat-vllm-sm70/shims/kv_rope_fp8_insert.py").read_text()

    assert "4c4031ff88c227cda4fa7e2a6b4e5c95585ba5ade2194aa26584b3aa0b49c853" in dockerfile
    assert '"$emmy_onecat_sm70_dir/kv_rope_fp8_insert.py"' in dockerfile
    assert "num_heads=0" in shim
    assert "quantize_and_insert_k_cache(" in shim
