"""Aggregate `vllm bench serve` stanza files from an engine-lane A/B into the article tables.

Reads a results tree of ``<root>/<lane>/<point>[_rN].txt`` stanzas (the layout the serving
lane drivers and the ``experiments/gemma-4-12B`` recipes produce), extracts the throughput and
latency fields, folds repeats into mean ± stddev, and writes two markdown tables — output
token throughput and median TTFT / TPOT — one column per lane, one row per workload point,
plus a flat ``lanes.json`` with every parsed field for downstream tooling.

    python scripts/aggregate_serving_lanes.py _tune/artbench --lanes stock,emmy,fastmath,llamacpp
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

FIELDS = {
    "out_tok_s": r"Output token throughput \(tok/s\):\s+([\d.]+)",
    "req_s": r"Request throughput \(req/s\):\s+([\d.]+)",
    "ttft_mean": r"Mean TTFT \(ms\):\s+([\d.]+)",
    "ttft_median": r"Median TTFT \(ms\):\s+([\d.]+)",
    "tpot_mean": r"Mean TPOT \(ms\):\s+([\d.]+)",
    "tpot_median": r"Median TPOT \(ms\):\s+([\d.]+)",
    "successful": r"Successful requests:\s+(\d+)",
    "num_prompts": r"--num-prompts (\d+)",
}

POINT_ORDER = ["small_c1", "small_c64", "head_c1", "head_c4", "head_c8", "rag_c4"]
POINT_LABEL = {
    "small_c1": "256 / 256 / c=1",
    "small_c64": "256 / 256 / c=64",
    "head_c1": "4096 / 4096 / c=1",
    "head_c4": "4096 / 4096 / c=4",
    "head_c8": "4096 / 4096 / c=8",
    "rag_c4": "8192 / 256 / c=4",
}


def parse_stanza(path: Path) -> dict[str, float] | None:
    text = path.read_text(errors="replace")
    out = {}
    for k, pat in FIELDS.items():
        m = re.search(pat, text)
        if m:
            out[k] = float(m.group(1))
    return out or None


def fold(runs: list[dict[str, float]]) -> dict[str, tuple[float, float | None]]:
    """Per-field (mean, stddev-or-None) across repeat runs."""
    keys = set().union(*(r.keys() for r in runs))
    folded = {}
    for k in keys:
        vals = [r[k] for r in runs if k in r]
        folded[k] = (statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else None)
    return folded


def cell(folded: dict, key: str, digits: int = 1) -> str:
    if key not in folded:
        return "—"
    mean, sd = folded[key]
    base = f"{mean:.{digits}f}"
    return f"{base} ± {sd:.{digits}f}" if sd is not None else base


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", help="results root: <root>/<lane>/<point>[_rN].txt")
    ap.add_argument("--lanes", default="stock,emmy,fastmath,llamacpp")
    ap.add_argument("--out", default=None, help="output .md path (default <root>/lanes.md)")
    args = ap.parse_args()

    root = Path(args.root)
    lanes = args.lanes.split(",")
    data: dict[str, dict[str, dict]] = {}
    for lane in lanes:
        for f in sorted((root / lane).glob("*.txt")):
            m = re.match(r"([a-z0-9_]+?)(?:_r(\d+))?$", f.stem)
            if not m or m.group(1) not in POINT_ORDER:
                continue
            s = parse_stanza(f)
            if s:
                data.setdefault(lane, {}).setdefault(m.group(1), []).append(s)

    folded = {lane: {pt: fold(runs) for pt, runs in pts.items()} for lane, pts in data.items()}

    def table(title: str, key: str, digits: int) -> list[str]:
        lines = [f"### {title}", "", "| in / out / conc | " + " | ".join(lanes) + " |"]
        lines.append("|---|" + "---:|" * len(lanes))
        for pt in POINT_ORDER:
            row = [POINT_LABEL[pt]]
            for lane in lanes:
                fd = folded.get(lane, {}).get(pt)
                # a point whose requests didn't all succeed doesn't earn the cell
                if fd and "successful" in fd and "num_prompts" in fd and fd["successful"][0] < fd["num_prompts"][0]:
                    row.append(f"FAILED ({fd['successful'][0]:.0f}/{fd['num_prompts'][0]:.0f})")
                else:
                    row.append(cell(fd, key, digits) if fd else "—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        return lines

    lines = []
    lines += table("Output token throughput (tok/s)", "out_tok_s", 1)
    lines += table("Median TTFT (ms)", "ttft_median", 0)
    lines += table("Median TPOT (ms)", "tpot_median", 1)

    out = Path(args.out) if args.out else root / "lanes.md"
    out.write_text("\n".join(lines) + "\n")
    (root / "lanes.json").write_text(
        json.dumps(
            {ln: {pt: {k: v for k, v in fd.items()} for pt, fd in pts.items()} for ln, pts in folded.items()},
            indent=1,
        )
    )
    print("\n".join(lines))
    print(f"tables → {out}\n  json → {root / 'lanes.json'}")


if __name__ == "__main__":
    main()
