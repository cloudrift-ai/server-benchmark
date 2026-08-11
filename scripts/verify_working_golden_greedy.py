#!/usr/bin/env python3
"""Benchmark a working-golden inventory before tuning with empty local evidence."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from emmy.compiler.pipeline.search.golden import load_golden_file
from emmy.logging_setup import setup_cli_logging

logger = logging.getLogger(__name__)


class GreedyVerificationError(ValueError):
    """Raised when a cold greedy attempt is contaminated or fails."""


@dataclass(frozen=True)
class GreedyAttempt:
    """One fresh-process cold greedy result."""

    name: str
    repeat: int
    json_path: str
    returncode: int
    status: str
    greedy_emmy_us: float | None = None
    error: str | None = None


def inventory_names(document: dict) -> list[str]:
    """Return every unique realization name from an untuned inventory."""
    names = [realization["name"] for config in document["configs"] for realization in config["realizations"]]
    if not names or len(names) != len(set(names)):
        raise GreedyVerificationError("working-golden inventory must contain unique realization names")
    return names


def _greedy_latency(path: Path) -> float:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GreedyVerificationError(f"cannot load greedy JSON {path}: {exc}") from exc
    result = record.get("backends", {}).get("Emmy")
    latency = result.get("latency_us") if isinstance(result, dict) else None
    if isinstance(latency, bool) or not isinstance(latency, int | float) or latency <= 0:
        raise GreedyVerificationError(f"{path} has no positive cold greedy Emmy latency")
    if result.get("captured") is not True or result.get("timing_semantics") != "captured_whole_forward":
        raise GreedyVerificationError(f"{path} cold greedy Emmy latency is not a captured whole forward")
    if record.get("pinned"):
        raise GreedyVerificationError(f"{path} unexpectedly contains pinned/search rows")
    return float(latency)


def verify(
    golden_file: Path,
    output_dir: Path,
    *,
    emmy: str,
    repeats: int,
    warmup: int,
    iters: int,
    cuda_visible_devices: str | None,
    run=subprocess.run,
) -> list[GreedyAttempt]:
    """Run every inventory target before tuning with isolated empty local state."""
    if repeats < 2:
        raise GreedyVerificationError("repeats must be at least 2")
    names = inventory_names(load_golden_file(golden_file))
    output_dir.mkdir(parents=True, exist_ok=True)
    attempts = []
    for name_index, name in enumerate(names):
        for repeat in range(repeats):
            repeat_dir = output_dir / f"target-{name_index:03d}-repeat-{repeat:02d}"
            repeat_dir.mkdir()
            db_path = repeat_dir / "empty-autotune.db"
            online_path = repeat_dir / "empty-online.json"
            cubin_path = repeat_dir / "empty-cubins"
            if db_path.exists() or online_path.exists() or cubin_path.exists():
                raise GreedyVerificationError(f"cold greedy state is not empty for {name} repeat {repeat}")
            json_path = repeat_dir / "greedy.json"
            environment = os.environ.copy()
            environment.update(
                {
                    "EMMY_NVCC_FLAGS": "",
                    "EMMY_TUNE_DB": str(db_path),
                    "EMMY_ONLINE_FILE": str(online_path),
                    "EMMY_CUBIN_CACHE": str(cubin_path),
                }
            )
            if cuda_visible_devices is not None:
                environment["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
            command = [
                emmy,
                "run",
                "--golden-file",
                str(golden_file),
                "--golden",
                name,
                "--bench",
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
                    latency = _greedy_latency(json_path)
                except GreedyVerificationError as exc:
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
            (output_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "golden_file": str(golden_file),
                        "evidence_state": "empty_local_db_online_cubin_before_tuning",
                        "repeats": repeats,
                        "attempts": [asdict(item) for item in attempts],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    failures = [attempt for attempt in attempts if attempt.status != "ok"]
    if failures:
        raise GreedyVerificationError(f"{len(failures)} of {len(attempts)} cold greedy attempts failed")
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
        )
    except GreedyVerificationError as exc:
        logger.error("Cold greedy verification failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
