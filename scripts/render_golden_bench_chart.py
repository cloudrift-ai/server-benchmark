"""Render the gemma blog's per-kernel chart from ``emmy run --bench --json`` records.

Reads the directory of per-target JSON records a multi-target golden run writes and produces a
self-contained echarts HTML — one horizontal bar row per case, emmy and torch.compile speedups
vs the eager baseline, sorted winners-first by the emmy ratio — plus a ``.csv`` of the plotted
values. This is the scripted generator for the per-kernel figures in the "Optimizing Gemma4 12B
for RTX GPUs" article.

The input is the record ``--json`` already emits; there is no separate benchmark harness to run
first. Produce it with::

    ./venv/bin/emmy run --golden-file <working.yaml> --bench --bench-backends eager,tcompile,emmy \\
        --json _tune/golden-bench/records

then::

    ./venv/bin/python scripts/render_golden_bench_chart.py \\
        _tune/golden-bench/records \\
        --title "gemma-4-12B per-kernel speedup vs PyTorch eager — RTX 5090" \\
        --out /path/to/blog/public/gemma4_12b_rtx5090_per_kernel.html
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# repo-root on sys.path so this script runs without ``pip install -e``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emmy.visualize.bar_chart import Bar, BarChart, _option  # noqa: E402
from emmy.visualize.page import render_html  # noqa: E402

EMMY_COLOR = "#3ddc84"
FM_COLOR = "#ffb454"
TCOMPILE_COLOR = "#7dd3fc"


def _read_records(directory: Path) -> dict[str, dict[str, float]]:
    """Collapse a run's per-target records into ``{target: {backend: latency_us}}``.

    One record shape serves every consumer: ``--json`` was built to retire the stdout parsing this
    chart's predecessor did, and reading it here is what let that predecessor go away.
    """
    results: dict[str, dict[str, float]] = {}
    for record_path in sorted(directory.glob("*.json")):
        record = json.loads(record_path.read_text())
        name = record.get("golden") or record_path.stem
        row = {backend: float(values["latency_us"]) for backend, values in record.get("backends", {}).items() if values.get("latency_us")}
        if row:
            results[name] = row
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json_path", help="directory of `emmy run --bench --json` records (standard lane)")
    ap.add_argument("--fm-json", default=None, help="optional fast-math-lane record directory (adds a third bar)")
    ap.add_argument("--title", default="per-kernel speedup vs PyTorch eager")
    ap.add_argument("--subtitle", default="ratio > 1 = faster than eager; dashed line = eager parity")
    ap.add_argument("--out", required=True, help="output .html path (a sibling .csv is written too)")
    args = ap.parse_args()

    results = _read_records(Path(args.json_path))
    fm = _read_records(Path(args.fm_json)) if args.fm_json else {}
    rows = []
    for name, r in results.items():
        if not isinstance(r, dict) or "Emmy" not in r:
            continue  # failed case — belongs in the run log, not the chart
        eager = r.get("Eager PyTorch")
        if not eager:
            continue
        tc = r.get("torch.compile")
        f = fm.get(name)
        fm_ratio = eager / f["Emmy"] if isinstance(f, dict) and f.get("Emmy") else None
        rows.append((name, eager / r["Emmy"], fm_ratio, eager / tc if tc else None))
    rows.sort(key=lambda t: t[1], reverse=True)  # winners first (fastest emmy ratio at the top)

    bars = [Bar(name="Emmy (greedy deploy pick)", values=[e for _, e, _, _ in rows], color=EMMY_COLOR)]
    if fm:
        bars.append(Bar(name="Emmy FAST_MATH", values=[f for _, _, f, _ in rows], color=FM_COLOR))
    bars.append(Bar(name="torch.compile", values=[t for _, _, _, t in rows], color=TCOMPILE_COLOR))
    chart = BarChart(
        categories=[n for n, _, _, _ in rows],
        bars=bars,
        value_name="speedup vs eager",
        title=args.title,
        subtitle=args.subtitle,
        baseline=1.0,
        baseline_label="eager",
        orientation="horizontal",
    )

    option = _option(chart, theme_name="dark")
    payload = {
        "option": option,
        "rowHeight": chart.row_height,
        "n": len(chart.categories),
        "padTop": option["grid"]["top"],
        "padBot": option["grid"]["bottom"],
    }
    body_html = '<div id="chart" style="width:100%;"></div>\n'
    scripts_js = (
        "const PAYLOAD = " + json.dumps(payload) + ";\n"
        "const el = document.getElementById('chart');\n"
        "el.style.height = (PAYLOAD.n * PAYLOAD.rowHeight + PAYLOAD.padTop + PAYLOAD.padBot) + 'px';\n"
        "const chart = echarts.init(el, null, { renderer: 'canvas' });\n"
        "chart.setOption(PAYLOAD.option);\n"
        "window.addEventListener('resize', () => chart.resize());\n"
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(body_html=body_html, scripts_js=scripts_js, theme="dark", title=args.title, transparent=True))
    with (out.with_suffix(".csv")).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "emmy_vs_eager", "emmy_fm_vs_eager", "tcompile_vs_eager"])
        w.writerows(rows)
    print(f"chart → {out}\n  csv → {out.with_suffix('.csv')}  ({len(rows)} cases)")


if __name__ == "__main__":
    main()
