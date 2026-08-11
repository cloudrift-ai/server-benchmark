#!/usr/bin/env python3
"""Replay every directly searched working-golden winner as an exact O3 A/B."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from emmy.compiler.pipeline.knob import format_tuning_knobs
from emmy.compiler.pipeline.search.golden import load_golden_file
from emmy.logging_setup import setup_cli_logging

logger = logging.getLogger(__name__)


class VerificationError(ValueError):
    """Raised when search output cannot be verified as exact deployable evidence."""


@dataclass(frozen=True)
class Winner:
    """One unambiguous directly searched winner from a working golden."""

    name: str
    knobs: str


@dataclass(frozen=True)
class Attempt:
    """One fresh-process verification result."""

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


def searched_winners(document: dict) -> list[Winner]:
    """Return exactly one directly searched winner per working-golden target."""
    winners: list[Winner] = []
    for config_index, config in enumerate(document["configs"]):
        matches = []
        for realization in config["realizations"]:
            ranking = realization.get("ranking") or {}
            if ranking.get("tune_winner") is True and ranking.get("source") == "tune" and ranking.get("status") == "ok":
                matches.append((realization, ranking))
        if len(matches) != 1:
            raise VerificationError(f"configs[{config_index}] must contain exactly one successful direct tune winner, got {len(matches)}")
        realization, ranking = matches[0]
        measured_knobs = ranking.get("measured_knobs")
        if not isinstance(measured_knobs, dict) or not measured_knobs:
            raise VerificationError(f"configs[{config_index}] winner has no measured_knobs mapping")
        if realization.get("knobs") != measured_knobs:
            raise VerificationError(f"configs[{config_index}] winner knobs differ from ranking.measured_knobs")
        knobs = format_tuning_knobs(measured_knobs)
        if knobs == "-":
            raise VerificationError(f"configs[{config_index}] winner has no replayable tuning knobs")
        winners.append(Winner(name=realization["name"], knobs=knobs.replace(", ", ",")))
    names = [winner.name for winner in winners]
    if len(names) != len(set(names)):
        raise VerificationError("winner names must be unique for exact --golden resolution")
    return winners


def _backend_name(backend: str) -> str:
    return {
        "eager": "Eager PyTorch",
        "tcompile": "torch.compile",
        "emmy": "Emmy",
        "hidet": "Hidet",
    }.get(backend, backend)


def _require_strict_correctness(proof, *, label: str) -> None:
    if not isinstance(proof, dict):
        raise VerificationError(f"{label} has no direct eager correctness proof")
    required = {"status": "pass", "reference": "eager", "rtol": 1e-3, "atol": 1e-3}
    if any(proof.get(key) != value for key, value in required.items()):
        raise VerificationError(f"{label} lacks the required strict eager correctness verdict")
    for metric in ("max_abs_error", "mean_abs_error", "max_rel_error"):
        value = proof.get(metric)
        if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
            raise VerificationError(f"{label} has no valid {metric}")


def _validate_ab_json(
    path: Path, *, required_backends: tuple[str, ...], optional_backends: tuple[str, ...]
) -> tuple[float, dict[str, float], list[str]]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot load A/B JSON {path}: {exc}") from exc
    pinned = [row for row in record.get("pinned", []) if row.get("kind") == "ab"]
    if len(pinned) != 1:
        raise VerificationError(f"{path} must contain exactly one explicit A/B row")
    row = pinned[0]
    if row.get("status") != "ok" or row.get("flags"):
        raise VerificationError(f"{path} exact-pinned row failed integrity checks: status={row.get('status')}, flags={row.get('flags')}")
    total_us = row.get("total_us")
    if isinstance(total_us, bool) or not isinstance(total_us, int | float) or total_us <= 0:
        raise VerificationError(f"{path} exact-pinned row has no positive total_us")
    if row.get("captured") is not True:
        raise VerificationError(f"{path} exact-pinned row did not use CUDA graph capture")
    num_launches = row.get("num_launches")
    if isinstance(num_launches, bool) or not isinstance(num_launches, int) or num_launches <= 0:
        raise VerificationError(f"{path} exact-pinned row has no positive num_launches")
    semantics = row.get("timing_semantics")
    expected_semantics = "whole_program_e2e" if num_launches > 1 else "single_launch"
    if semantics != expected_semantics:
        raise VerificationError(f"{path} exact-pinned row must use {expected_semantics} timing, got {semantics!r}")
    _require_strict_correctness(row.get("correctness"), label=f"{path} exact-pinned row")
    backends = record.get("backends")
    if not isinstance(backends, dict):
        raise VerificationError(f"{path} has no backend comparison")
    latencies: dict[str, float] = {}
    for backend in required_backends:
        display_name = _backend_name(backend)
        result = backends.get(display_name)
        latency = result.get("latency_us") if isinstance(result, dict) else None
        if isinstance(latency, bool) or not isinstance(latency, int | float) or latency <= 0:
            raise VerificationError(f"{path} has no positive {display_name} latency")
        if result.get("captured") is not True or result.get("timing_semantics") != "captured_whole_forward":
            raise VerificationError(f"{path} {display_name} does not use like-for-like captured forward timing")
        if backend == "tcompile" and result.get("correctness") != {
            "status": "pass",
            "rtol": 1e-3,
            "atol": 1e-3,
            "fullgraph": True,
        }:
            raise VerificationError(f"{path} torch.compile lacks the required full-graph correctness proof")
        if backend == "emmy":
            _require_strict_correctness(result.get("correctness"), label=f"{path} deploy Emmy row")
        latencies[display_name] = float(latency)
    missing_optional = []
    for backend in optional_backends:
        display_name = _backend_name(backend)
        result = backends.get(display_name)
        latency = result.get("latency_us") if isinstance(result, dict) else None
        if isinstance(latency, bool) or not isinstance(latency, int | float) or latency <= 0:
            missing_optional.append(display_name)
        else:
            if result.get("captured") is not True or result.get("timing_semantics") != "captured_whole_forward":
                raise VerificationError(f"{path} {display_name} does not use like-for-like captured forward timing")
            if backend == "hidet" and result.get("correctness") != {
                "status": "pass",
                "rtol": 1e-3,
                "atol": 1e-3,
                "fullgraph": True,
            }:
                raise VerificationError(f"{path} Hidet lacks the required full-graph correctness proof")
            latencies[display_name] = float(latency)
    return float(total_us), latencies, missing_optional


def verify(
    golden_file: Path,
    output_dir: Path,
    *,
    emmy: str,
    repeats: int,
    warmup: int,
    iters: int,
    cuda_visible_devices: str | None,
    bench_backends: str = "eager,tcompile,emmy",
    optional_backends: tuple[str, ...] = (),
    run=subprocess.run,
) -> list[Attempt]:
    """Run every winner in a fresh process and preserve a machine-readable attempt manifest."""
    if repeats < 2:
        raise VerificationError("repeats must be at least 2")
    winners = searched_winners(load_golden_file(golden_file))
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["EMMY_NVCC_FLAGS"] = ""
    environment["TORCHINDUCTOR_MAX_AUTOTUNE"] = "1"
    environment["TORCHINDUCTOR_COORDINATE_DESCENT_TUNING"] = "1"
    environment["TORCHINDUCTOR_CUDAGRAPHS"] = "1"
    if cuda_visible_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    attempts: list[Attempt] = []
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
                bench_backends,
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
            logger.info("Verifying %s, repeat %d/%d", winner.name, repeat + 1, repeats)
            result = run(command, env=environment, check=False)
            error = None
            status = "ok"
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
                        required_backends=("eager", "tcompile", "emmy"),
                        optional_backends=optional_backends,
                    )
                    deploy_emmy_us = backend_latencies_us["Emmy"]
                except VerificationError as exc:
                    status = "integrity_failed"
                    error = str(exc)
            attempts.append(
                Attempt(
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
            (output_dir / "manifest.json").write_text(
                json.dumps({"golden_file": str(golden_file), "repeats": repeats, "attempts": [asdict(item) for item in attempts]}, indent=2)
                + "\n",
                encoding="utf-8",
            )
    failures = [attempt for attempt in attempts if attempt.status != "ok"]
    if failures:
        raise VerificationError(f"{len(failures)} of {len(attempts)} exact O3 verification attempts failed")
    return attempts


def main() -> int:
    """Run the command-line verifier."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("golden_file", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--emmy", default="emmy")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--bench-backends", default="eager,tcompile,emmy")
    parser.add_argument("--optional-backend", action="append", default=[])
    args = parser.parse_args()
    setup_cli_logging()
    try:
        verify(
            args.golden_file,
            args.output_dir,
            emmy=args.emmy,
            repeats=args.repeats,
            warmup=args.warmup,
            iters=args.iters,
            cuda_visible_devices=args.cuda_visible_devices,
            bench_backends=args.bench_backends,
            optional_backends=tuple(args.optional_backend),
        )
    except VerificationError as exc:
        logger.error("Winner verification failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
