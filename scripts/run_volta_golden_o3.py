#!/usr/bin/env python3
"""Run pinned Volta O3 bootstrap-vs-candidate A/B lanes across 16 V100s."""

from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from emmy.logging_setup import setup_cli_logging

logger = logging.getLogger(__name__)

WARMUP = 5
ITERS = 15
EXPECTED_GPU = "NVIDIA Tesla V100 SXM3 32GB"

BOOTSTRAP = {
    "WORK": "w1x1",
    "TILE": "mma_m8n8k4_f16_f32/f1x1",
    "REDUCE": "",
    "STAGE": "",
    "LOOPIFY": "0",
    "RASTER": "",
}
_CHILD_MARKER = "VOLTA_O3_LONG_BENCH_CHILD"


def _run_emmy_child() -> None:
    """Run the normal CLI with enough accumulated GPU-time budget for the vocabulary projection."""
    from emmy.compiler.backend.cuda.backend import CudaBackend

    original_init = CudaBackend.__init__

    def long_bench_init(self, *args, **kwargs):
        kwargs.setdefault("bench_run_timeout_s", 60.0)
        original_init(self, *args, **kwargs)

    CudaBackend.__init__ = long_bench_init
    from emmy.emmy import main as emmy_main

    emmy_main()


def _run_one(gpu: int, out_dir: Path, row: dict, lane: str, repeat: int) -> dict:
    knobs = BOOTSTRAP if lane == "bootstrap" else row["knobs"]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env[_CHILD_MARKER] = "1"
    env["EMMY_CUBIN_CACHE"] = str(out_dir / "cubins" / f"gpu-{gpu}")
    env["EMMY_GPU_LOCK"] = str(out_dir / "locks" / f"gpu-{gpu}.lock")
    for key, value in knobs.items():
        env[f"EMMY_{key}"] = str(value)
    safe_name = row["name"].replace("/", "_")
    record_path = out_dir / "records" / f"gpu-{gpu}-{safe_name}-{lane}-{repeat}.json"
    record_path.parent.mkdir(exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "run",
        "--golden",
        row["name"],
        "--target",
        "sm_70",
        "--bench",
        "--bench-backends",
        "eager,emmy",
        "--warmup",
        str(WARMUP),
        "--iters",
        str(ITERS),
        "--json",
        str(record_path),
    ]
    proc = subprocess.run(command, env=env, capture_output=True, text=True, timeout=900)
    output = proc.stdout + proc.stderr
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    greedy = record.get("greedy", {})
    isolated = greedy.get("isolated", {})
    kernels = isolated.get("kernels", [])
    realized = kernels[0].get("record_knobs", {}) if len(kernels) == 1 else {}
    integrity_errors = []
    if proc.returncode != 0:
        integrity_errors.append(f"command returned {proc.returncode}")
    for key, expected in (
        ("golden", row["name"]),
        ("gpu", EXPECTED_GPU),
        ("warmup", WARMUP),
        ("iters", ITERS),
    ):
        if record.get(key) != expected:
            integrity_errors.append(f"{key}={record.get(key)!r}, expected {expected!r}")
    if greedy.get("lane") != "std":
        integrity_errors.append(f"greedy lane={greedy.get('lane')!r}, expected 'std'")
    if greedy.get("status") != "ok":
        integrity_errors.append(f"greedy status={greedy.get('status')!r}")
    if isolated.get("status") != "ok":
        integrity_errors.append(f"isolated status={isolated.get('status')!r}")
    if isolated.get("flags"):
        integrity_errors.append(f"isolated flags={isolated['flags']!r}")
    if len(kernels) != 1:
        integrity_errors.append(f"isolated kernel count={len(kernels)}, expected 1")
    for key, expected in knobs.items():
        if realized.get(key) != str(expected):
            integrity_errors.append(f"realized {key}={realized.get(key)!r}, expected {str(expected)!r}")
    eager_us = record.get("backends", {}).get("Eager PyTorch", {}).get("latency_us")
    isolated_us = isolated.get("total_us")
    interleaved_us = greedy.get("total_us")
    if not isinstance(eager_us, (int, float)) or eager_us <= 0:
        integrity_errors.append(f"invalid eager latency {eager_us!r}")
    if not isinstance(isolated_us, (int, float)) or isolated_us <= 0:
        integrity_errors.append(f"invalid isolated latency {isolated_us!r}")
    if not isinstance(interleaved_us, (int, float)) or interleaved_us <= 0:
        integrity_errors.append(f"invalid interleaved latency {interleaved_us!r}")
    log_path = out_dir / "logs" / f"gpu-{gpu}.log"
    with log_path.open("a") as log:
        log.write(f"\n===== {row['name']} {lane} repeat={repeat} rc={proc.returncode} =====\n")
        log.write(output)
        log.write("\n")
    return {
        "name": row["name"],
        "gpu": gpu,
        "lane": lane,
        "repeat": repeat,
        "returncode": proc.returncode,
        "valid": not integrity_errors,
        "integrity_errors": integrity_errors,
        "eager_us": eager_us,
        "emmy_us": isolated_us,
        "interleaved_emmy_us": interleaved_us,
        "knobs": knobs,
        "record_knobs": realized,
        "record": str(record_path),
        "schedule": ",".join(f"{key}={value}" for key, value in knobs.items()),
    }


def _worker(gpu: int, rows: list[dict], out_dir: Path) -> list[dict]:
    results: list[dict] = []
    for row in rows:
        for repeat in range(3):
            results.append(_run_one(gpu, out_dir, row, "bootstrap", repeat))
            results.append(_run_one(gpu, out_dir, row, "candidate", repeat))
    shard_path = out_dir / f"gpu-{gpu}.json"
    shard_path.write_text(json.dumps(results, indent=2) + "\n")
    return results


def main() -> None:
    setup_cli_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--filter", help="Optional golden-name glob for a bounded retry")
    parser.add_argument("--gpu-offset", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    candidates = json.loads(args.candidates.read_text())
    if len(candidates) != 44:
        raise ValueError(f"expected 44 candidates, found {len(candidates)}")
    if args.filter:
        candidates = [row for row in candidates if fnmatch.fnmatchcase(row["name"], args.filter)]
        if not candidates:
            raise ValueError(f"candidate filter matched no rows: {args.filter}")
    worker_count = min(16, len(candidates))
    if args.gpu_offset < 0 or args.gpu_offset + worker_count > 16:
        raise ValueError("GPU offset and worker count must fit the 16-GPU host")
    candidate_names = [row["name"] for row in candidates]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "logs").mkdir(exist_ok=True)
    (args.out_dir / "records").mkdir(exist_ok=True)
    (args.out_dir / "cubins").mkdir(exist_ok=True)
    (args.out_dir / "locks").mkdir(exist_ok=True)
    chunks = [candidates[worker::worker_count] for worker in range(worker_count)]
    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        futures = [pool.submit(_worker, worker + args.gpu_offset, chunk, args.out_dir) for worker, chunk in enumerate(chunks)]
        results = [item for future in futures for item in future.result()]
    results.sort(key=lambda row: (row["name"], row["lane"], row["repeat"]))
    usable = [row for row in results if row["valid"]]
    usable_counts = {
        (name, lane): sum(row["name"] == name and row["lane"] == lane for row in usable)
        for name in candidate_names
        for lane in ("bootstrap", "candidate")
    }
    hard_failures = [name for name in candidate_names if usable_counts[name, "bootstrap"] != 3 or usable_counts[name, "candidate"] != 3]
    command_failures = sum(row["returncode"] != 0 for row in results)
    (args.out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    logger.info(
        "Wrote %d A/B measurements with %d command failures and %d shapes lacking one complete lane",
        len(results),
        command_failures,
        len(hard_failures),
    )
    if hard_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    if os.environ.pop(_CHILD_MARKER, None):
        _run_emmy_child()
    else:
        main()
