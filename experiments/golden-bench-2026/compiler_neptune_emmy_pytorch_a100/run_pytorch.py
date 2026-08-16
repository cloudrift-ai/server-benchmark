#!/usr/bin/env python3
"""Measure current eager PyTorch and Inductor when an untuned Emmy setup fails."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def _inputs(operator: str, sequence_length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_heads = 64 if operator.endswith("gqa") else 32
    kv_heads = 8 if operator.endswith("gqa") else 32
    q_length = 1 if operator.startswith("decode") else sequence_length
    torch.manual_seed(0)
    return tuple(
        torch.randn(shape, device="cuda", dtype=torch.float16)
        for shape in (
            (1, q_heads, q_length, 128),
            (1, kv_heads, sequence_length, 128),
            (1, kv_heads, sequence_length, 128),
        )
    )


def _attention(operator: str, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if operator == "decode_gqa":
        return F.scaled_dot_product_attention(
            q.reshape(1, 8, 8, 1, 128),
            k.reshape(1, 8, 1, k.shape[-2], 128),
            v.reshape(1, 8, 1, v.shape[-2], 128),
            is_causal=False,
        ).reshape(1, 64, 1, 128)
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=operator in {"prefill_causal", "prefill_gqa"},
        enable_gqa=operator == "prefill_gqa",
    )


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


def _measure(functions: dict[str, object], warmup: int, iters: int) -> tuple[dict[str, float], bool]:
    graphs = []
    captured_functions = {}
    try:
        for name, fn in functions.items():
            graph, replay = _capture(fn)
            graphs.append(graph)
            captured_functions[name] = replay
    except Exception:  # noqa: BLE001 - both backends fall back together to preserve timing parity
        captured = False
    else:
        functions = captured_functions
        captured = True

    events = {name: [] for name in functions}
    for iteration in range(warmup + iters):
        for name, fn in functions.items():
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            with torch.no_grad():
                start.record()
                fn()
                stop.record()
            if iteration >= warmup:
                events[name].append((start, stop))
    torch.cuda.synchronize()
    return ({name: min(start.elapsed_time(stop) * 1000 for start, stop in rows) for name, rows in events.items()}, captured)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("operator", choices=("prefill_global", "prefill_causal", "prefill_gqa", "decode_causal", "decode_gqa"))
    parser.add_argument("sequence_length", type=int)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--iters", type=int, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()

    q, k, v = _inputs(args.operator, args.sequence_length)

    def attention(q, k, v):
        return _attention(args.operator, q, k, v)

    torch._dynamo.reset()
    compiled_attention = torch.compile(attention, fullgraph=True, mode="max-autotune-no-cudagraphs")

    def eager():
        return attention(q, k, v)

    def compiled():
        return compiled_attention(q, k, v)

    with torch.no_grad():
        for _ in range(args.warmup + 5):
            compiled()
        torch.testing.assert_close(compiled(), eager(), rtol=1e-3, atol=1e-3)

    latencies, captured = _measure({"Eager PyTorch": eager, "torch.compile": compiled}, args.warmup, args.iters)
    semantics = "captured_whole_forward" if captured else "uncaptured_forward"
    payload = {
        "operator": args.operator,
        "sequence_length": args.sequence_length,
        "gpu": torch.cuda.get_device_name(0),
        "warmup": args.warmup,
        "iters": args.iters,
        "backends": {
            name: {
                "latency_us": latency,
                "captured": captured,
                "timing_semantics": semantics,
                **({"correctness": {"status": "pass", "rtol": 1e-3, "atol": 1e-3, "fullgraph": True}} if name == "torch.compile" else {}),
            }
            for name, latency in latencies.items()
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Eager PyTorch: {latencies['Eager PyTorch']:.3f} us")
    print(f"torch.compile: {latencies['torch.compile']:.3f} us")


if __name__ == "__main__":
    main()
