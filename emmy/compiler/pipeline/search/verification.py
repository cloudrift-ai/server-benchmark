"""Fresh-process verification for every target in a working golden."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from emmy.compiler.pipeline.knob import format_tuning_knobs
from emmy.compiler.pipeline.search.golden import load_golden_file


class WorkingGoldenVerificationError(ValueError):
    """Raised when working-golden evidence is incomplete or invalid."""


@dataclass(frozen=True)
class Winner:
    """One unambiguous directly searched winner from a working golden."""

    name: str
    knobs: str


@dataclass(frozen=True)
class VerificationAttempt:
    """One fresh-process exact-winner verification result."""

    name: str
    repeat: int
    json_path: str
    returncode: int
    status: str
    winner_total_us: float | None = None
    deploy_emmy_us: float | None = None
    backend_latencies_us: dict[str, float] | None = None
    missing_optional_backends: list[str] | None = None
    error: str | None = None


@dataclass(frozen=True)
class GreedyAttempt:
    """One fresh-process cold local-evidence result."""

    name: str
    repeat: int
    json_path: str
    returncode: int
    status: str
    greedy_emmy_us: float | None = None
    error: str | None = None


RunCommand = Callable[..., subprocess.CompletedProcess]


def searched_winners(document: dict) -> list[Winner]:
    """Return exactly one directly searched winner per working-golden target."""
    if not document.get("configs"):
        raise WorkingGoldenVerificationError("working golden must contain at least one target")
    winners: list[Winner] = []
    for config_index, config in enumerate(document["configs"]):
        matches = []
        for realization in config["realizations"]:
            ranking = realization.get("ranking") or {}
            if ranking.get("tune_winner") is True and ranking.get("source") == "tune" and ranking.get("status") == "ok":
                matches.append((realization, ranking))
        if len(matches) != 1:
            raise WorkingGoldenVerificationError(
                f"configs[{config_index}] must contain exactly one successful direct tune winner, got {len(matches)}"
            )
        realization, ranking = matches[0]
        measured_knobs = ranking.get("measured_knobs")
        if not isinstance(measured_knobs, dict) or not measured_knobs:
            raise WorkingGoldenVerificationError(f"configs[{config_index}] winner has no measured_knobs mapping")
        if realization.get("knobs") != measured_knobs:
            raise WorkingGoldenVerificationError(f"configs[{config_index}] winner knobs differ from ranking.measured_knobs")
        knobs = format_tuning_knobs(measured_knobs)
        if knobs == "-":
            raise WorkingGoldenVerificationError(f"configs[{config_index}] winner has no replayable tuning knobs")
        winners.append(Winner(name=realization["name"], knobs=knobs.replace(", ", ",")))
    names = [winner.name for winner in winners]
    if len(names) != len(set(names)):
        raise WorkingGoldenVerificationError("winner names must be unique for exact --golden resolution")
    return winners


def inventory_names(document: dict) -> list[str]:
    """Return every unique realization name from an untuned inventory."""
    names = [realization["name"] for config in document["configs"] for realization in config["realizations"]]
    if not names or len(names) != len(set(names)):
        raise WorkingGoldenVerificationError("working-golden inventory must contain unique realization names")
    return names


def _backend_name(backend: str) -> str:
    return {
        "eager": "Eager PyTorch",
        "tcompile": "torch.compile",
        "emmy": "Emmy",
        "hidet": "Hidet",
    }.get(backend, backend)


def _require_strict_correctness(proof, *, label: str) -> None:
    if not isinstance(proof, dict):
        raise WorkingGoldenVerificationError(f"{label} has no direct eager correctness proof")
    required = {"status": "pass", "reference": "eager", "rtol": 1e-3, "atol": 1e-3}
    if any(proof.get(key) != value for key, value in required.items()):
        raise WorkingGoldenVerificationError(f"{label} lacks the required strict eager correctness verdict")
    for metric in ("max_abs_error", "mean_abs_error", "max_rel_error"):
        value = proof.get(metric)
        if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
            raise WorkingGoldenVerificationError(f"{label} has no valid {metric}")


def _validate_backend(record: dict, backend: str, *, required: bool) -> tuple[float | None, bool]:
    display_name = _backend_name(backend)
    result = record.get("backends", {}).get(display_name)
    latency = result.get("latency_us") if isinstance(result, dict) else None
    if isinstance(latency, bool) or not isinstance(latency, int | float) or latency <= 0:
        if required:
            raise WorkingGoldenVerificationError(f"has no positive {display_name} latency")
        return None, True
    if result.get("captured") is not True or result.get("timing_semantics") != "captured_whole_forward":
        raise WorkingGoldenVerificationError(f"{display_name} does not use like-for-like captured forward timing")
    if backend in {"tcompile", "hidet"} and result.get("correctness") != {
        "status": "pass",
        "rtol": 1e-3,
        "atol": 1e-3,
        "fullgraph": True,
    }:
        raise WorkingGoldenVerificationError(f"{display_name} lacks the required full-graph correctness proof")
    if backend == "emmy":
        _require_strict_correctness(result.get("correctness"), label="deploy Emmy row")
    return float(latency), False


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkingGoldenVerificationError(f"cannot load A/B JSON {path}: {exc}") from exc


def _validate_ab_json(
    path: Path,
    *,
    required_backends: tuple[str, ...],
    optional_backends: tuple[str, ...],
) -> tuple[float, dict[str, float], list[str]]:
    record = _load_json(path)
    pinned = [row for row in record.get("pinned", []) if row.get("kind") == "ab"]
    if len(pinned) != 1:
        raise WorkingGoldenVerificationError(f"{path} must contain exactly one explicit A/B row")
    row = pinned[0]
    if row.get("status") != "ok" or row.get("flags"):
        raise WorkingGoldenVerificationError(
            f"{path} exact-pinned row failed integrity checks: status={row.get('status')}, flags={row.get('flags')}"
        )
    total_us = row.get("total_us")
    if isinstance(total_us, bool) or not isinstance(total_us, int | float) or total_us <= 0:
        raise WorkingGoldenVerificationError(f"{path} exact-pinned row has no positive total_us")
    if row.get("captured") is not True:
        raise WorkingGoldenVerificationError(f"{path} exact-pinned row did not use CUDA graph capture")
    num_launches = row.get("num_launches")
    if isinstance(num_launches, bool) or not isinstance(num_launches, int) or num_launches <= 0:
        raise WorkingGoldenVerificationError(f"{path} exact-pinned row has no positive num_launches")
    expected_semantics = "whole_program_e2e" if num_launches > 1 else "single_launch"
    if row.get("timing_semantics") != expected_semantics:
        raise WorkingGoldenVerificationError(f"{path} exact-pinned row must use {expected_semantics} timing")
    _require_strict_correctness(row.get("correctness"), label=f"{path} exact-pinned row")

    latencies: dict[str, float] = {}
    for backend in required_backends:
        latency, _ = _validate_backend(record, backend, required=True)
        assert latency is not None
        latencies[_backend_name(backend)] = latency
    missing_optional = []
    for backend in optional_backends:
        latency, missing = _validate_backend(record, backend, required=False)
        if missing:
            missing_optional.append(_backend_name(backend))
        else:
            assert latency is not None
            latencies[_backend_name(backend)] = latency
    return float(total_us), latencies, missing_optional


def _validate_greedy_json(path: Path) -> float:
    record = _load_json(path)
    if record.get("pinned"):
        raise WorkingGoldenVerificationError(f"{path} unexpectedly contains pinned/search rows")
    latency, _ = _validate_backend(record, "emmy", required=True)
    assert latency is not None
    return latency


def _base_environment(cuda_visible_devices: str | None) -> dict[str, str]:
    environment = os.environ.copy()
    environment["EMMY_NVCC_FLAGS"] = ""
    if cuda_visible_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    return environment


def verify_tune_winners(
    golden_file: Path,
    output_dir: Path,
    *,
    emmy: str,
    repeats: int,
    warmup: int,
    iters: int,
    cuda_visible_devices: str | None,
    bench_backends: tuple[str, ...],
    optional_backends: tuple[str, ...] = (),
    run: RunCommand = subprocess.run,
) -> list[VerificationAttempt]:
    """Replay every directly searched winner in fresh O3 processes."""
    if repeats < 1:
        raise WorkingGoldenVerificationError("process repeats must be at least 1")
    winners = searched_winners(load_golden_file(golden_file))
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = _base_environment(cuda_visible_devices)
    environment["TORCHINDUCTOR_MAX_AUTOTUNE"] = "1"
    environment["TORCHINDUCTOR_COORDINATE_DESCENT_TUNING"] = "1"
    environment["TORCHINDUCTOR_CUDAGRAPHS"] = "1"
    requested = tuple(dict.fromkeys((*bench_backends, *optional_backends, "emmy")))
    required = tuple(backend for backend in requested if backend not in optional_backends)

    attempts: list[VerificationAttempt] = []
    for winner_index, winner in enumerate(winners):
        for repeat in range(repeats):
            json_path = output_dir / f"winner-{winner_index:03d}-repeat-{repeat:02d}.json"
            command = [
                emmy,
                "run",
                "--golden-file",
                str(golden_file),
                "--golden",
                winner.name,
                "--bench",
                "--strict-correctness",
                "--bench-backends",
                ",".join(requested),
                "--warmup",
                str(warmup),
                "--iters",
                str(iters),
                "--seed",
                "0",
                "--ab",
                winner.knobs,
                "--json",
                str(json_path),
            ]
            result = run(command, env=environment, check=False)
            status = "ok"
            error = None
            winner_total_us = None
            deploy_emmy_us = None
            backend_latencies_us = None
            missing_optional_backends = None
            if result.returncode != 0:
                status = "command_failed"
                error = f"emmy run exited {result.returncode}"
            else:
                try:
                    winner_total_us, backend_latencies_us, missing_optional_backends = _validate_ab_json(
                        json_path,
                        required_backends=required,
                        optional_backends=optional_backends,
                    )
                    deploy_emmy_us = backend_latencies_us["Emmy"]
                except WorkingGoldenVerificationError as exc:
                    status = "integrity_failed"
                    error = str(exc)
            attempts.append(
                VerificationAttempt(
                    name=winner.name,
                    repeat=repeat,
                    json_path=json_path.name,
                    returncode=result.returncode,
                    status=status,
                    winner_total_us=winner_total_us,
                    deploy_emmy_us=deploy_emmy_us,
                    backend_latencies_us=backend_latencies_us,
                    missing_optional_backends=missing_optional_backends,
                    error=error,
                )
            )
            _write_manifest(output_dir, golden_file, repeats, "searched_winner", attempts)
    failures = [attempt for attempt in attempts if attempt.status != "ok"]
    if failures:
        raise WorkingGoldenVerificationError(f"{len(failures)} of {len(attempts)} exact O3 verification attempts failed")
    return attempts


def verify_cold_greedy(
    golden_file: Path,
    output_dir: Path,
    *,
    emmy: str,
    repeats: int,
    warmup: int,
    iters: int,
    cuda_visible_devices: str | None,
    run: RunCommand = subprocess.run,
) -> list[GreedyAttempt]:
    """Benchmark every inventory target with fresh empty local evidence."""
    if repeats < 1:
        raise WorkingGoldenVerificationError("process repeats must be at least 1")
    names = inventory_names(load_golden_file(golden_file))
    output_dir.mkdir(parents=True, exist_ok=True)
    attempts: list[GreedyAttempt] = []
    for name_index, name in enumerate(names):
        for repeat in range(repeats):
            repeat_dir = output_dir / f"target-{name_index:03d}-repeat-{repeat:02d}"
            repeat_dir.mkdir()
            db_path = repeat_dir / "empty-autotune.db"
            online_path = repeat_dir / "empty-online.json"
            cubin_path = repeat_dir / "empty-cubins"
            json_path = repeat_dir / "greedy.json"
            environment = _base_environment(cuda_visible_devices)
            environment.update(
                {
                    "EMMY_TUNE_DB": str(db_path),
                    "EMMY_ONLINE_FILE": str(online_path),
                    "EMMY_CUBIN_CACHE": str(cubin_path),
                }
            )
            command = [
                emmy,
                "run",
                "--golden-file",
                str(golden_file),
                "--golden",
                name,
                "--bench",
                "--strict-correctness",
                "--bench-backends",
                "emmy",
                "--warmup",
                str(warmup),
                "--iters",
                str(iters),
                "--seed",
                "0",
                "--json",
                str(json_path),
            ]
            result = run(command, env=environment, check=False)
            status = "ok"
            error = None
            latency = None
            if result.returncode != 0:
                status = "command_failed"
                error = f"emmy run exited {result.returncode}"
            else:
                try:
                    latency = _validate_greedy_json(json_path)
                except WorkingGoldenVerificationError as exc:
                    status = "integrity_failed"
                    error = str(exc)
            attempts.append(
                GreedyAttempt(
                    name=name,
                    repeat=repeat,
                    json_path=str(json_path.relative_to(output_dir)),
                    returncode=result.returncode,
                    status=status,
                    greedy_emmy_us=latency,
                    error=error,
                )
            )
            _write_manifest(output_dir, golden_file, repeats, "cold_local_evidence", attempts)
    failures = [attempt for attempt in attempts if attempt.status != "ok"]
    if failures:
        raise WorkingGoldenVerificationError(f"{len(failures)} of {len(attempts)} cold greedy attempts failed")
    return attempts


def _write_manifest(output_dir: Path, golden_file: Path, repeats: int, mode: str, attempts: list) -> None:
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "golden_file": str(golden_file),
                "mode": mode,
                "process_repeats": repeats,
                "attempts": [asdict(item) for item in attempts],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
