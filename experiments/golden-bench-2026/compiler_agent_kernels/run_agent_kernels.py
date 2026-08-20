#!/usr/bin/env python3
"""Measure committed agent-generated kernels against eager PyTorch and Inductor, one JSON per target.

Each ``<agent>/<target>/`` directory holds ``reference.py`` (``Model``, ``get_inputs``,
``get_init_inputs``) and ``kernel.py`` (``ModelNew``) in the KernelBench module convention. The
agent kernel must match ``Model`` on the same inputs within the suite's tolerance before it is timed;
a mismatch is recorded as a failure, never dropped. Timings replay a captured CUDA graph when every
backend captures, and fall back together to direct launches otherwise, so the three numbers in one
record always share a method.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import traceback
from pathlib import Path

import torch


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem + "_" + path.parent.name.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _to_cuda(values):
    return [v.cuda() if isinstance(v, torch.Tensor) else v for v in values]


def _capture(fn):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side), torch.no_grad():
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(graph):
        fn()
    return graph, graph.replay


def _measure(functions: dict, warmup: int, iters: int) -> tuple[dict, bool]:
    graphs, replays = [], {}
    try:
        for name, fn in functions.items():
            graph, replay = _capture(fn)
            graphs.append(graph)
            replays[name] = replay
    except Exception:  # noqa: BLE001 - all backends fall back together to keep the timing method shared
        captured = False
    else:
        functions, captured = replays, True
    samples = {name: [] for name in functions}
    for iteration in range(warmup + iters):
        for name, fn in functions.items():
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            with torch.no_grad():
                start.record()
                fn()
                stop.record()
            stop.synchronize()
            if iteration >= warmup:
                samples[name].append(start.elapsed_time(stop) * 1000.0)
    return {name: statistics.median(values) for name, values in samples.items()}, captured


def _run_target(target_dir: Path, args) -> dict:
    record = {"target": target_dir.name, "status": "ok"}
    reference = _load(target_dir / "reference.py")
    kernel = _load(target_dir / "kernel.py")
    torch.manual_seed(0)
    init_inputs = _to_cuda(reference.get_init_inputs())
    inputs = _to_cuda(reference.get_inputs())
    model = reference.Model(*init_inputs).cuda().eval()
    agent = kernel.ModelNew(*init_inputs).cuda().eval()
    with torch.no_grad():
        expected = model(*inputs)
        actual = agent(*inputs)
    record["max_abs_error"] = float((actual.float() - expected.float()).abs().max())
    if not torch.allclose(actual.float(), expected.float(), rtol=args.rtol, atol=args.atol):
        record["status"] = "agent-incorrect"
        return record
    compiled = torch.compile(model, mode="max-autotune-no-cudagraphs", fullgraph=True)
    with torch.no_grad():
        compiled_out = compiled(*inputs)
    record["tcompile_matches_eager"] = bool(torch.allclose(compiled_out.float(), expected.float(), rtol=args.rtol, atol=args.atol))
    functions = {
        "eager_us": lambda: model(*inputs),
        "tcompile_us": lambda: compiled(*inputs),
        "agent_us": lambda: agent(*inputs),
    }
    timings, captured = _measure(functions, args.warmup, args.iters)
    record.update(timings)
    record["cuda_graph"] = captured
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("agent_dir", type=Path)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    targets = sorted(p for p in args.agent_dir.iterdir() if (p / "kernel.py").exists())
    if not targets:
        print(f"no <target>/kernel.py under {args.agent_dir}", file=sys.stderr)
        return 2
    measured = 0
    for target_dir in targets:
        try:
            record = _run_target(target_dir, args)
        except Exception:  # noqa: BLE001 - a failing target is a recorded result, not an aborted sweep
            record = {"target": target_dir.name, "status": "failed", "traceback": traceback.format_exc()}
        (args.results_dir / f"{target_dir.name}.json").write_text(json.dumps(record, indent=2))
        print(f"{record['target']}\t{record['status']}")
        measured += record["status"] == "ok"
    return 0 if measured else 1


if __name__ == "__main__":
    sys.exit(main())
