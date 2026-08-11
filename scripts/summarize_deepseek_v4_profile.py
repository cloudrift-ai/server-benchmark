#!/usr/bin/env python3
"""Summarize DeepSeek V4 serving Torch traces without loading them into memory."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

EVENT_RE = re.compile(r'"ph": "X", "cat": "([^"]+)", "name": "(.*)", "pid":')
TIME_RE = re.compile(r'"ts": ([0-9.]+), "dur": ([0-9.]+)')
RANK_RE = re.compile(r"dp\d+_pp(\d+)_tp(\d+)_.*_rank(\d+)\.")
ITERATION_RE = re.compile(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")


def kernel_category(name: str) -> str:
    lower = name.lower()
    if any(term in lower for term in ("sendrecv", "send_recv", "ncclsend", "ncclrecv")):
        return "pp_collectives"
    if any(
        term in lower
        for term in (
            "nccl",
            "allreduce",
            "all_reduce",
            "allgather",
            "all_gather",
            "reduce_scatter",
            "cross_device_reduce",
            "broadcast",
        )
    ):
        return "tp_collectives"
    if any(
        term in lower
        for term in (
            "mxfp4",
            "moe",
            "expert",
            "grouped_gemm",
            "grouped_bmm",
            "groupedgemm",
            "fp4_e2m1",
            "finalizemoerouting",
            "expandinputrows",
            "topkgating",
        )
    ):
        return "moe"
    if any(
        term in lower
        for term in (
            "kv",
            "slot_mapping",
            "quantize_and_insert",
            "gather_k",
            "index_k",
            "indexer",
            "rope_insert",
        )
    ):
        return "kv"
    if any(term in lower for term in ("attention", "attn", "sparse_paged", "sparse_gathered", "flash")):
        return "attention"
    return "other"


def summarize_trace(path: Path) -> dict[str, object]:
    match = RANK_RE.search(path.name)
    if match is None:
        raise ValueError(f"cannot read PP/TP rank from {path.name}")
    pp_rank, tp_rank, global_rank = map(int, match.groups())

    category_us: dict[str, float] = defaultdict(float)
    kernel_us: dict[str, float] = defaultdict(float)
    iterations: list[dict[str, float | int | str]] = []
    pending: tuple[str, str] | None = None

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as trace:
        for line in trace:
            event_match = EVENT_RE.search(line)
            if event_match is not None:
                pending = event_match.groups()
                continue
            if pending is None:
                continue
            time_match = TIME_RE.search(line)
            if time_match is None:
                continue
            category, escaped_name = pending
            pending = None
            timestamp_us, duration_us = map(float, time_match.groups())
            name = json.loads(f'"{escaped_name}"')
            if category == "kernel":
                kernel_us[name] += duration_us
                category_us[kernel_category(name)] += duration_us
            elif category == "user_annotation":
                iteration_match = ITERATION_RE.fullmatch(name)
                if iteration_match is not None:
                    context_requests, context_tokens, generation_requests, generation_tokens = map(int, iteration_match.groups())
                    iterations.append(
                        {
                            "name": name,
                            "timestamp_us": timestamp_us,
                            "duration_us": duration_us,
                            "context_requests": context_requests,
                            "context_tokens": context_tokens,
                            "generation_requests": generation_requests,
                            "generation_tokens": generation_tokens,
                        }
                    )

    if not iterations:
        raise ValueError(f"no execute_context annotations in {path}")
    active_us = sum(float(item["duration_us"]) for item in iterations)
    span_us = max(float(item["timestamp_us"]) + float(item["duration_us"]) for item in iterations) - min(
        float(item["timestamp_us"]) for item in iterations
    )
    top_kernels = sorted(kernel_us.items(), key=lambda item: item[1], reverse=True)[:30]
    segments: list[list[dict[str, float | int | str]]] = []
    for iteration in iterations:
        if int(iteration["context_requests"]) > 0:
            segments.append([])
        if segments:
            segments[-1].append(iteration)
    microbatches: list[dict[str, float | int | bool]] = []
    for segment in segments:
        segment_active_us = sum(float(item["duration_us"]) for item in segment)
        segment_span_us = max(float(item["timestamp_us"]) + float(item["duration_us"]) for item in segment) - min(
            float(item["timestamp_us"]) for item in segment
        )
        prefill = segment[0]
        microbatches.append(
            {
                "concurrency": int(prefill["context_requests"]),
                "context_tokens": int(prefill["context_tokens"]),
                "iteration_count": len(segment),
                "active_ms": segment_active_us / 1000,
                "span_ms": segment_span_us / 1000,
                "pp_utilization": segment_active_us / segment_span_us,
                "pp_bubble_ms": (segment_span_us - segment_active_us) / 1000,
                "pp_bubble_fraction": (segment_span_us - segment_active_us) / segment_span_us,
            }
        )
    return {
        "trace": path.name,
        "pp_rank": pp_rank,
        "tp_rank": tp_rank,
        "global_rank": global_rank,
        "iteration_count": len(iterations),
        "active_ms": active_us / 1000,
        "span_ms": span_us / 1000,
        "pp_utilization": active_us / span_us,
        "pp_bubble_ms": (span_us - active_us) / 1000,
        "pp_bubble_fraction": (span_us - active_us) / span_us,
        "kernel_ms": {key: value / 1000 for key, value in sorted(category_us.items())},
        "top_kernels_ms": {key: value / 1000 for key, value in top_kernels},
        "microbatches": microbatches,
        "iterations": iterations,
    }


def aggregate(traces: list[dict[str, object]]) -> dict[str, object]:
    by_stage: dict[int, list[dict[str, object]]] = defaultdict(list)
    for trace in traces:
        by_stage[int(trace["pp_rank"])].append(trace)

    stages: dict[str, object] = {}
    for pp_rank, stage_traces in sorted(by_stage.items()):
        category_names = sorted({name for trace in stage_traces for name in dict(trace["kernel_ms"]).keys()})
        stage_kernel_ms = {
            name: statistics.mean(float(dict(trace["kernel_ms"]).get(name, 0.0)) for trace in stage_traces) for name in category_names
        }
        stages[str(pp_rank)] = {
            "trace_count": len(stage_traces),
            "mean_pp_utilization": statistics.mean(float(trace["pp_utilization"]) for trace in stage_traces),
            "mean_pp_bubble_fraction": statistics.mean(float(trace["pp_bubble_fraction"]) for trace in stage_traces),
            "mean_pp_bubble_ms": statistics.mean(float(trace["pp_bubble_ms"]) for trace in stage_traces),
            "mean_active_ms": statistics.mean(float(trace["active_ms"]) for trace in stage_traces),
            "mean_kernel_ms": stage_kernel_ms,
        }
    return {"stages": stages, "traces": traces}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--all-tp", action="store_true", help="parse all TP ranks instead of TP rank 0 only")
    args = parser.parse_args()

    paths = sorted(args.trace_dir.glob("dp*_pp*_tp*_rank*.pt.trace.json.gz"))
    if not args.all_tp:
        paths = [path for path in paths if int(RANK_RE.search(path.name).group(2)) == 0]  # type: ignore[union-attr]
    if not paths:
        raise SystemExit(f"no worker traces found in {args.trace_dir}")
    print(json.dumps(aggregate([summarize_trace(path) for path in paths]), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
