#!/usr/bin/env python3
"""Run ONE serving-benchmark point several times against a live OpenAI-compatible server.

A thin driver around ``vllm bench serve``: the benchmark arguments are passed through
verbatim after ``--``, so a recipe can carry the exact client invocation a baseline was
measured with instead of a paraphrase of it. On top of that it adds the three things a
cross-engine comparison needs and the bare client does not:

1. **A hardened SSE parser.** ``vllm bench serve`` splits server-sent events on ``\\n\\n``
   and additionally strips every socket read, so it survives a CRLF-framed stream only in
   the "one complete event per read" regime, via its single-JSON fallback. A coalesced read
   under concurrency, or an SSE keepalive comment (``: ping - ...``), wedges the parser for
   the rest of the request: the client then reports empty generations and fabricated TPOT
   rather than an error. Measured against tabbyAPI at concurrency 16: 51 of 64 requests
   failed silently. This driver normalizes CRLF and drops SSE comment lines before the
   split, which fixes it for any server, and then refuses to report a point whose requests
   did not all complete.
2. **Repeats**, each with its own seed, plus a discarded warmup run.
3. **1 Hz ``nvidia-smi`` polling** per run, so a point reports power draw and peak VRAM
   beside its latency numbers.

Output per run in ``--out-dir``: ``<point>_r<N>.txt`` (the result stanza, the layout
``scripts/aggregate_serving_lanes.py`` reads), ``<point>_r<N>.json`` (the client's own
``--save-result`` record), ``<point>_r<N>.smi.csv`` (the raw poll) and
``<point>_r<N>.power.json`` (its summary). The warmup writes ``<point>_warmup.*``, which
the aggregator ignores by construction.

    python scripts/bench_serve_sweep.py --out-dir results/emmy --point c1_in512 --reps 3 -- \\
        --backend openai --endpoint /v1/completions --base-url http://127.0.0.1:8000 \\
        --model turboderp/GLM-4.5-Air-exl3 --dataset-name random \\
        --random-input-len 512 --random-output-len 128 --ignore-eos \\
        --num-prompts 8 --max-concurrency 1
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _harden_sse() -> None:
    """Make ``vllm bench serve``'s SSE reader tolerate CRLF framing and keepalive comments.

    Patches ``StreamedResponseHandler.add_chunk`` in place. Transport framing only — it
    changes which bytes are recognized as an event boundary, never a timestamp."""
    from vllm.benchmarks.lib import endpoint_request_func as erf

    def add_chunk(self, chunk_bytes: bytes) -> list[str]:
        self.buffer += self._decoder.decode(chunk_bytes).replace("\r\n", "\n")
        messages = []
        while "\n\n" in self.buffer:
            message, self.buffer = self.buffer.split("\n\n", 1)
            message = message.strip()
            # A line opening with ':' is an SSE comment (sse-starlette's keepalive ping).
            # The caller json.loads() whatever it is handed, so it must never see one.
            if message and not message.startswith(":"):
                messages.append(message)
        # The tail may already be a whole event whose separator has not arrived (or was
        # eaten by the caller's per-read strip) — the client's own fallback, kept as is.
        if self.buffer.startswith("data: "):
            content = self.buffer.removeprefix("data: ").strip()
            if content == "[DONE]":
                messages.append(self.buffer.strip())
                self.buffer = ""
            elif content:
                try:
                    json.loads(content)
                except json.JSONDecodeError:
                    pass  # incomplete JSON: wait for more bytes
                else:
                    messages.append(self.buffer.strip())
                    self.buffer = ""
        return messages

    erf.StreamedResponseHandler.add_chunk = add_chunk


def _start_poll(csv_path: Path, gpu_index: str, interval: float) -> subprocess.Popen | None:
    """Start a 1 Hz nvidia-smi poll writing power.draw / memory.used to ``csv_path``."""
    cmd = [
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-gpu=timestamp,power.draw,memory.used",
        "--format=csv,noheader,nounits",
        f"--loop-ms={int(interval * 1000)}",
    ]
    try:
        return subprocess.Popen(cmd, stdout=csv_path.open("w"), stderr=subprocess.DEVNULL)  # noqa: S603
    except FileNotFoundError:
        logger.warning("nvidia-smi not on PATH — no power / VRAM capture for this run")
        return None


def _summarize_poll(csv_path: Path) -> dict[str, float]:
    """Mean/max power (W) and peak memory (MiB) from a poll CSV; empty dict when there is none."""
    watts, mib = [], []
    for line in csv_path.read_text(errors="replace").splitlines() if csv_path.is_file() else []:
        fields = [f.strip() for f in line.split(",")]
        if len(fields) != 3:
            continue
        try:
            watts.append(float(fields[1]))
            mib.append(float(fields[2]))
        except ValueError:
            continue
    if not mib:
        return {}
    return {
        "samples": len(mib),
        "power_mean_w": sum(watts) / len(watts),
        "power_max_w": max(watts),
        "vram_peak_mib": max(mib),
    }


def _run_once(bench_args: list[str], out_dir: Path, stem: str, seed: int, gpu_index: str, interval: float) -> dict:
    """One client run: poll, bench, persist the stanza + the poll summary. Returns the metrics."""
    from vllm.benchmarks.serve import add_cli_args, main

    parser = argparse.ArgumentParser()
    add_cli_args(parser)
    ns = parser.parse_args(
        [*bench_args, "--seed", str(seed), "--save-result", "--result-dir", str(out_dir), "--result-filename", f"{stem}.json"]
    )

    csv_path = out_dir / f"{stem}.smi.csv"
    poll = _start_poll(csv_path, gpu_index, interval)
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            metrics = main(ns)
    finally:
        if poll is not None:
            poll.terminate()
            poll.wait(timeout=10)
            time.sleep(0.2)  # let the redirected file flush before it is read back

    (out_dir / f"{stem}.txt").write_text(captured.getvalue())
    power = _summarize_poll(csv_path)
    (out_dir / f"{stem}.power.json").write_text(json.dumps(power, indent=1))
    logger.info("%s: seed %d, %s", stem, seed, ", ".join(f"{k}={v:.1f}" for k, v in power.items()) or "no GPU poll")
    return metrics


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="bench_serve_sweep.py",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Everything after `--` is passed to `vllm bench serve` unchanged.",
    )
    parser.add_argument("--out-dir", required=True, help="directory for the stanzas, result JSON and poll CSVs")
    parser.add_argument("--point", required=True, help="point name, e.g. c1_in512 — the stanza file stem")
    parser.add_argument("--reps", type=int, default=3, help="recorded runs (default 3)")
    parser.add_argument("--warmup", type=int, default=1, help="discarded runs before the recorded ones (default 1)")
    parser.add_argument("--seed-base", type=int, default=42, help="run N uses --seed <base+N>, warmup uses <base> (default 42)")
    parser.add_argument("--gpu-index", default="0", help="nvidia-smi --id for the poll (default 0)")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="poll period in seconds (default 1.0)")
    parser.add_argument("bench_args", nargs=argparse.REMAINDER, help="-- <vllm bench serve args>")
    args = parser.parse_args(argv)
    if args.bench_args and args.bench_args[0] == "--":
        args.bench_args = args.bench_args[1:]
    if not args.bench_args:
        parser.error("no `vllm bench serve` arguments given (pass them after `--`)")
    return args


def main(argv: list[str] | None = None) -> int:
    from emmy.logging_setup import setup_cli_logging

    setup_cli_logging()
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _harden_sse()

    for i in range(args.warmup):
        stem = f"{args.point}_warmup" if args.warmup == 1 else f"{args.point}_warmup{i + 1}"
        _run_once(args.bench_args, out_dir, stem, args.seed_base, args.gpu_index, args.poll_interval)

    failed = []
    for rep in range(1, args.reps + 1):
        metrics = _run_once(args.bench_args, out_dir, f"{args.point}_r{rep}", args.seed_base + rep, args.gpu_index, args.poll_interval)
        # The silent-failure guard. A wedged stream, a refused request or a server-side
        # abort leaves `completed` below the request count while every latency field still
        # prints a plausible number; a point that lost requests is not a measurement.
        completed, requested = metrics.get("completed"), metrics.get("num_prompts")
        if completed is not None and requested is not None and completed != requested:
            logger.error("%s rep %d: only %s of %s requests completed — the point is not usable", args.point, rep, completed, requested)
            failed.append(rep)

    if failed:
        logger.error("%s: %d of %d recorded runs lost requests (reps %s)", args.point, len(failed), args.reps, failed)
        return 1
    logger.info("%s: %d recorded runs in %s", args.point, args.reps, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
