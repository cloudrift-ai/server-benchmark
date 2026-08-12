"""Fail-closed serving runtime-log evidence gates."""

from types import SimpleNamespace

import pytest

from emmy.benchmark.execution import _capture_server_log_gate, validate_server_log_patterns


def test_server_log_patterns_require_positive_and_forbid_fallback() -> None:
    verdict = validate_server_log_patterns(
        "Using flashinfer-cutlass for NVFP4 GEMM",
        [r"(?i)using .*nvfp4 gemm"],
        [r"(?i)marlin|fallback"],
    )
    assert verdict["status"] == "pass"

    failed = validate_server_log_patterns(
        "GPU does not have native support; using Marlin fallback",
        [r"(?i)using .*nvfp4 gemm"],
        [r"(?i)marlin|fallback"],
    )
    assert failed["status"] == "fail"
    assert len(failed["errors"]) == 2


def test_native_nvfp4_moe_gate_ignores_unselected_candidate_names() -> None:
    required = [r"(?i)Using '(FLASHINFER_TRTLLM|VLLM_CUTLASS)' NvFp4 MoE backend"]
    forbidden = [r"(?i)Using '(MARLIN|EMULATION)' NvFp4 MoE backend"]
    native = (
        "Using 'FLASHINFER_TRTLLM' NvFp4 MoE backend out of potential backends: "
        "['FLASHINFER_TRTLLM', 'VLLM_CUTLASS', 'MARLIN', 'EMULATION']."
    )
    assert validate_server_log_patterns(native, required, forbidden)["status"] == "pass"
    for selected in ("MARLIN", "EMULATION"):
        logs = f"Using '{selected}' NvFp4 MoE backend out of potential backends: ['FLASHINFER_TRTLLM', '{selected}']."
        assert validate_server_log_patterns(logs, required, forbidden)["status"] == "fail"


@pytest.mark.asyncio
async def test_server_log_capture_failure_is_authoritative(tmp_path) -> None:
    async def run_cmd(_command, **_kwargs):
        return 1, "", "compose unavailable"

    recipe = SimpleNamespace(
        benchmark=SimpleNamespace(
            required_server_log_patterns=["NVFP4"],
            forbidden_server_log_patterns=["Marlin"],
        )
    )
    path = tmp_path / "server.log"
    verdict, success = await _capture_server_log_gate(run_cmd, recipe, path, dry_run=False)

    assert success is False
    assert verdict["status"] == "fail"
    assert "exited 1" in verdict["errors"][0]
    assert path.read_text() == "compose unavailable\n"


@pytest.mark.asyncio
async def test_server_log_gate_dry_run_does_not_require_a_host(tmp_path) -> None:
    async def unreachable(*_args, **_kwargs):
        raise AssertionError("dry run must not contact a server")

    recipe = SimpleNamespace(
        benchmark=SimpleNamespace(
            required_server_log_patterns=["NVFP4"],
            forbidden_server_log_patterns=[],
        )
    )
    verdict, success = await _capture_server_log_gate(unreachable, recipe, tmp_path / "server.log", dry_run=True)

    assert success is True
    assert verdict["status"] == "dry-run"
