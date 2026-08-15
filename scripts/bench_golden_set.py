"""Per-golden-case bench table: emmy's greedy deploy pick vs eager PyTorch and ``torch.compile``.

For every golden entry whose name matches ``--filter``, run ``emmy run --golden NAME --bench``
(greedy pick = what a default compile deploys) against the requested torch backends, parse the
backend latency table, and write a markdown table + JSON next to each other. This is the
"per-kernel vs torch" view of a model's golden set on the current GPU — the source of the
gemma-4-12B RTX 4090 table in the part-3 blog follow-up.

    # the gemma-4-12B set, all three backends:
    python scripts/bench_golden_set.py --filter gemma4_12b

    # fast-math lane (consumer dies): same table under the gate
    EMMY_FAST_MATH=1 python scripts/bench_golden_set.py --filter gemma4_12b

Only names recorded for the live GPU are benched (``--golden`` resolution is card-scoped);
a filter that matches no card-local name lists which cards do carry it. Rows sort
losers-first by the eager ratio.

Each case comes from the *golden dataset*: a curated, card-scoped program with
a recorded ground-truth config benched live beside the greedy pick, so the table is stable while
the prior/deploy path is in flux.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# repo-root on sys.path so this script runs without ``pip install -e``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emmy.compiler.pipeline.search.golden import _GOLDENS_DIR, load_golden_file, load_golden_records  # noqa: E402

_BIN = Path(sys.executable).parent
_ROW = re.compile(r"^(Eager PyTorch|torch\.compile|Emmy)\s+([\d.]+)")


def _bench(path: Path, name: str, backends: str, timeout: int) -> dict[str, float] | str:
    """One ``run --bench --golden PATH --target NAME`` invocation → ``{backend: us}``, or an error string."""
    proc = subprocess.run(
        [str(_BIN / "emmy"), "run", "--bench", "--golden", str(path), "--target", name, "--bench-backends", backends],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    rows: dict[str, float] = {}
    for line in proc.stdout.splitlines():
        m = _ROW.match(line)
        if m:
            rows[m.group(1)] = float(m.group(2))
    if "Emmy" not in rows:
        tail = (proc.stdout + proc.stderr)[-200:].replace("\n", " ")
        return f"no Emmy row (rc={proc.returncode}): {tail}"
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--filter", default="", help="substring of the golden names to bench (default: all)")
    ap.add_argument("--backends", default="eager,tcompile,emmy")
    ap.add_argument("--out-dir", default="_tune/golden-bench", help="where the .md/.json land")
    ap.add_argument("--timeout", type=int, default=900, help="per-case timeout (s)")
    args = ap.parse_args()

    import torch

    from emmy.gpu import by_name

    # canonicalize like the deploy-time golden join: the torch name differs from the recorded
    # gpu_name on datacenter cards (e.g. "Tesla V100-SXM3-32GB" vs "NVIDIA Tesla V100 SXM3 32GB").
    torch_name = torch.cuda.get_device_name(0)
    known = by_name(torch_name)
    gpu = known.name if known is not None else torch_name
    # ``run --golden`` takes a file path (name resolution is per-file via --target), so map each
    # card-scoped matching name to the canonical file that records it (first file wins on a dup).
    sources: dict[str, Path] = {}
    cards: set[str] = set()
    for path in sorted(_GOLDENS_DIR.rglob("*.yaml")):
        for record in load_golden_records(load_golden_file(path)):
            if args.filter in record.name:
                cards.add(record.gpu_name)
                if record.gpu_name == gpu:
                    sources.setdefault(record.name, path)
    names = sorted(sources)
    if not names:
        print(f"no golden names match {args.filter!r} on {gpu!r} (matching cards: {sorted(cards)})", file=sys.stderr)
        sys.exit(1)

    results: dict[str, dict[str, float] | str] = {}
    for i, name in enumerate(names, 1):
        print(f"[{i}/{len(names)}] {name} …", flush=True)
        try:
            results[name] = _bench(sources[name], name, args.backends, args.timeout)
        except subprocess.TimeoutExpired:
            results[name] = f"timeout after {args.timeout}s"
        print(f"    {results[name]}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "golden_bench.json").write_text(json.dumps(results, indent=1))

    def ratio(rows: dict[str, float]) -> float:
        eager = rows.get("Eager PyTorch")
        return eager / rows["Emmy"] if eager else float("inf")

    ok = sorted(((n, r) for n, r in results.items() if isinstance(r, dict)), key=lambda t: ratio(t[1]))
    lines = [
        f"# Golden-set bench — filter {args.filter!r}",
        "",
        "| kernel | eager µs | torch.compile µs | emmy µs | vs eager | vs torch.compile |",
        "|---|--:|--:|--:|--:|--:|",
    ]
    for name, rows in ok:
        eager, tc, em = rows.get("Eager PyTorch"), rows.get("torch.compile"), rows["Emmy"]
        fmt = lambda v: f"{v:.0f}" if v is not None else "—"  # noqa: E731
        r_e = f"{eager / em:.2f}x" if eager else "—"
        r_tc = f"{tc / em:.2f}x" if tc else "—"
        lines.append(f"| {name} | {fmt(eager)} | {fmt(tc)} | {em:.1f} | {r_e} | {r_tc} |")
    for name, err in results.items():
        if not isinstance(err, dict):
            lines.append(f"| {name} | — | — | FAILED | — | — |")
    md = out_dir / "golden_bench.md"
    md.write_text("\n".join(lines) + "\n")
    print(f"table → {md}\n json → {out_dir / 'golden_bench.json'}")


if __name__ == "__main__":
    main()
