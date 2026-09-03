"""Conftest for ``tests/perf/``: ``bench_pair`` fixture, session-end
summary table, and JSON dump.

Tests in this directory carry ``pytestmark = [pytest.mark.perf,
requires_cuda]``. The ``perf`` marker is **deselected by default** —
the root ``tests/conftest.py`` hook skips perf-marked items for any
``tests/`` collection unless ``-m perf`` is passed, so ``make test``
stays fast. Run explicitly with ``pytest -m perf`` (or ``make
bench-kernels``).

The ``bench_pair`` fixture drives ``emmy run --bench --json`` (and
``--profile`` when ``EMMY_BENCH_NCU=1``) as a subprocess per
realization-corpus case, pinned to the schedule that case authors, and
reads the record to build a ``PerfRow``. Reusing the CLI keeps the
torch / torch.compile / emmy comparison and the ncu metrics on the same
code path users invoke directly, and reusing the corpus keeps one case
inventory in the tree instead of two.

The same measurement also answers the regression question: a case
carrying a stored latency for this card is compared against it, best of
a few runs, and a slower one REPORTS rather than fails.

After the session, ``pytest_terminal_summary`` prints a table sorted
by ratio (worst losses first) and writes the same data to
``tests/perf/.results/<utc-timestamp>.json``. With ncu enabled, extra
columns (occupancy, bank conflicts, SM/DRAM/FMA throughput, regs) are
appended.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.compiler.helpers import requires_cuda  # noqa: F401  (re-exported)
from tests.compiler.realization import helpers
from tests.compiler.realization.helpers import Case

_RESULTS_DIR = Path(__file__).resolve().parent / ".results"

# Cross-process advisory lock around the GPU iter loop in the
# subprocess-driven bencher. Set on conftest import so every spawned
# ``emmy run --bench`` (across xdist workers and inside each
# worker's per-case subprocess) coordinates on the same path. Trace,
# compile, dump-write all run unlocked — only the kernel-launch phase
# serializes. Override with ``EMMY_GPU_LOCK`` before invoking
# ``make bench-kernels`` if a different path is desired.
# Per-uid path: on a multi-user runner the first user's lock file (mode 0644, sticky /tmp)
# is unopenable by everyone else, so a shared path fails the run with PermissionError
# instead of serializing (CI run 32339655489). Cross-user serialization was never real.
os.environ.setdefault("EMMY_GPU_LOCK", f"/tmp/emmy-gpu-{os.getuid()}.lock")


# ---------------------------------------------------------------------------
# PerfRow + session collector
# ---------------------------------------------------------------------------


@dataclass
class PerfRow:
    name: str
    op: str
    shape: str
    dtype: str
    torch_us: float
    emmy_us: float
    ratio: float  # torch_us / emmy_us — >1 means emmy wins
    launches: int
    tags: tuple[str, ...]
    iters: int = 100
    torch_compile_us: float | None = None
    # Per-kernel ncu metrics keyed by kernel name. Populated only when
    # the optional ncu pass runs (``EMMY_BENCH_NCU=1``). Each entry
    # maps metric-name → numeric value (units are baked into the metric
    # name; see ``emmy.commands.run._NCU_METRICS``). ``None`` when
    # not collected.
    ncu: dict[str, dict[str, float]] | None = None
    #: This card's stored latency for the case, when it has one. ``None`` means the corpus has no
    #: baseline for this card yet — reported at session end so coverage can grow, never a failure.
    recorded_us: float | None = None
    #: Set when every run came in above the band. A finding, not a failure.
    regressed: bool = False


def _collector(config) -> list[PerfRow]:
    if not hasattr(config, "_perf_rows"):
        config._perf_rows = []
    return config._perf_rows


# ---------------------------------------------------------------------------
# bench_pair fixture
# ---------------------------------------------------------------------------


def _ncu_enabled() -> bool:
    if os.environ.get("EMMY_NCU_CHILD"):
        return False
    if os.environ.get("EMMY_BENCH_NCU", "") not in ("1", "true", "True"):
        return False
    return shutil.which("ncu") is not None


@pytest.fixture
def bench_pair(request):
    """Return a callable ``run(case) -> PerfRow`` for one realization-corpus case.

    Each call spawns ``emmy run --golden <case> --realization <name> --bench --json``, at
    deployable optimization, and reads the record. The case replays the schedule it authors rather
    than being left to greedy, because a corpus case names a schedule and a timing for a different
    kernel would answer a different question.

    Does not assert on the ratio — the lane tracks performance, it does not gate on it.
    """

    def _run(case: Case) -> PerfRow | None:
        if _tune_enabled():
            # Tune-only path: populate the autotune DB, measure nothing. Run
            # ``make bench-kernels-tuned`` afterwards to measure with the tuned knobs.
            _tune_via_subprocess(case)
            return None
        row = _bench_corpus_case(case, profile=_ncu_enabled())
        _collector(request.config).append(row)
        return row

    return _run


def _tune_enabled() -> bool:
    return os.environ.get("EMMY_TUNE", "") in ("1", "true", "True")


def _tune_via_subprocess(case: Case) -> None:
    """Search this case's kernel and record the winners into the autotune DB."""
    subprocess.run(
        [sys.executable, "-m", "emmy.emmy", "tune", "--golden", str(case.path), "--realization", case.record.name],
        check=False,
        env={**os.environ, "EMMY_TUNE": "1"},
        timeout=3600,
    )


def _bench_corpus_case(case: Case, *, profile: bool) -> PerfRow:
    """Measure one case and build its row, including the regression verdict.

    Samples are taken lazily against the stored baseline: a run inside the band settles the case,
    because the minimum can only fall. Only a case that looks slow pays for the extra runs — which
    is exactly the case where interference is the question.
    """
    facts = helpers.describe(case)
    recorded = helpers.recorded_latency(case, helpers.live_hardware_id())
    stored = float(recorded["emmy_us"]) if recorded else None
    with tempfile.TemporaryDirectory(prefix=f"emmy_perf_{case.path.stem}_") as tmp:
        record = None
        samples: list[float] = []
        for repeat in range(helpers.LATENCY_REPEATS if stored else 1):
            output = Path(tmp, f"{repeat}.json")
            command = helpers.bench_command(case, output)
            if profile:
                command.append("--profile")
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                env={**os.environ, "EMMY_NVCC_FLAGS": "", "EMMY_DUMP_DIR": tmp},
                timeout=1800,
            )
            if result.returncode != 0 or not output.exists():
                raise AssertionError(f"{case.id}: bench failed (exit {result.returncode})\n{result.stderr[-2000:]}")
            record = json.loads(output.read_text())
            pinned = [row for row in record.get("pinned", []) if row.get("status") == "ok" and row.get("total_us")]
            if not pinned:
                raise AssertionError(f"{case.id}: the pinned row measured nothing — {record.get('pinned')}")
            samples.append(float(pinned[0]["total_us"]))
            if stored is None or samples[-1] <= stored * (1 + helpers.LATENCY_BAND):
                break

        backends = record["backends"]
        emmy_us = min(samples)
        torch_us = float(backends.get("Eager PyTorch", {}).get("latency_us") or 0.0)
        tcompile = backends.get("torch.compile", {}).get("latency_us")
        launches = len([row for row in record.get("pinned", []) if row.get("status") == "ok"][0].get("kernels", []))
        ncu_path = Path(tmp, "61_ncu_metrics.json")
        return PerfRow(
            name=case.id,
            op=facts["op"],
            shape=facts["shape"],
            dtype=facts["dtype"],
            torch_us=torch_us,
            emmy_us=emmy_us,
            ratio=(torch_us / emmy_us) if emmy_us > 0 else 0.0,
            launches=launches,
            tags=(facts["family"], facts["op"]),
            iters=int(record.get("iters", 0)),
            torch_compile_us=float(tcompile) if tcompile is not None else None,
            ncu=json.loads(ncu_path.read_text()) if ncu_path.exists() else None,
            recorded_us=stored,
            regressed=bool(stored and emmy_us > stored * (1 + helpers.LATENCY_BAND)),
        )


def _aggregate_ncu(ncu: dict[str, dict[str, float]] | None) -> dict[str, float]:
    """Per-row aggregate of the per-kernel ncu metrics for the summary
    table. Sum durations and conflicts; time-weight the percentage
    metrics so a fast minor kernel doesn't drag the average."""
    if not ncu:
        return {}
    total_ns = sum(m.get("gpu__time_duration.sum", 0.0) for m in ncu.values())
    total_conflicts_ld = sum(m.get("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum", 0.0) for m in ncu.values())
    total_conflicts_st = sum(m.get("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum", 0.0) for m in ncu.values())
    total_lsu_inst = sum(m.get("smsp__inst_executed_pipe_lsu.sum", 0.0) for m in ncu.values())

    def _wavg(metric: str) -> float:
        if total_ns <= 0:
            return 0.0
        num = sum(m.get(metric, 0.0) * m.get("gpu__time_duration.sum", 0.0) for m in ncu.values())
        return num / total_ns

    return {
        "ncu_us": total_ns / 1000.0,
        "occ_pct": _wavg("sm__warps_active.avg.pct_of_peak_sustained_active"),
        "sm_pct": _wavg("sm__throughput.avg.pct_of_peak_sustained_elapsed"),
        "fma_pct": _wavg("sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active"),
        "dram_pct": _wavg("dram__throughput.avg.pct_of_peak_sustained_elapsed"),
        "conflicts": total_conflicts_ld + total_conflicts_st,
        "lsu_inst": total_lsu_inst,
        "regs": max((m.get("launch__registers_per_thread", 0.0) for m in ncu.values()), default=0.0),
    }


# ---------------------------------------------------------------------------
# Session summary
# ---------------------------------------------------------------------------


def _format_table(rows: list[PerfRow]) -> str:
    if not rows:
        return ""
    rows = sorted(rows, key=lambda r: r.ratio)  # losses first
    has_ncu = any(r.ncu for r in rows)
    has_compile = any(r.torch_compile_us is not None for r in rows)

    base_headers = ["case", "shape", "torch_us"]
    if has_compile:
        base_headers.append("tcomp_us")
    base_headers += ["depl_us", "ratio", "launches", "iters"]
    ncu_headers = ["occ%", "sm%", "fma%", "dram%", "conflicts", "regs"] if has_ncu else []
    headers = tuple(base_headers + ncu_headers)

    aggregates = [_aggregate_ncu(r.ncu) for r in rows]

    def _row_cells(r: PerfRow, agg: dict[str, float]) -> tuple[str, ...]:
        base = [r.name, r.shape, f"{r.torch_us:>8.1f}"]
        if has_compile:
            base.append(f"{r.torch_compile_us:>8.1f}" if r.torch_compile_us is not None else "—")
        base += [
            f"{r.emmy_us:>7.1f}",
            f"{r.ratio:>5.2f}x",
            f"{r.launches:>8d}",
            f"{r.iters:>5d}",
        ]
        if not has_ncu:
            return tuple(base)
        if not agg:
            return tuple(base + ["—"] * len(ncu_headers))
        return tuple(
            base
            + [
                f"{agg['occ_pct']:>4.0f}",
                f"{agg['sm_pct']:>4.0f}",
                f"{agg['fma_pct']:>4.0f}",
                f"{agg['dram_pct']:>5.0f}",
                f"{int(agg['conflicts']):>9,d}",
                f"{int(agg['regs']):>4d}",
            ]
        )

    body = [_row_cells(r, a) for r, a in zip(rows, aggregates, strict=True)]
    cells = [headers] + body
    widths = [max(len(c) for c in col) for col in zip(*cells, strict=True)]

    def _fmt(row_cells: tuple[str, ...]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(row_cells, widths, strict=True))

    lines = [_fmt(headers), _fmt(tuple("-" * w for w in widths))]
    for r_cells in body:
        lines.append(_fmt(r_cells))
    return "\n".join(lines)


def pytest_sessionfinish(session, exitstatus):
    """Worker-side hand-off for ``pytest-xdist`` runs.

    Each xdist worker has its own ``config._perf_rows``; without this
    hook only the controller's (empty) list reaches the terminal
    summary. ``workeroutput`` is xdist's built-in dict for
    worker→controller payloads — the controller picks it back up in
    ``pytest_testnodedown``.
    """
    rows = getattr(session.config, "_perf_rows", [])
    workeroutput = getattr(session.config, "workeroutput", None)
    if workeroutput is not None and rows:
        # Each row is a dataclass; serialize via asdict so the
        # controller can rehydrate without sharing the class import.
        workeroutput["perf_rows"] = [{**asdict(r), "tags": list(r.tags)} for r in rows]


def pytest_testnodedown(node, error):
    """Controller-side: drain a finished xdist worker's rows into the
    controller's collector so ``pytest_terminal_summary`` sees all
    cases (not just whichever ones happened to land on the controller).
    """
    payload = getattr(node, "workeroutput", {}).get("perf_rows", [])
    if not payload:
        return
    rows = _collector(node.config)
    for r in payload:
        rows.append(PerfRow(**{**r, "tags": tuple(r.get("tags", ()))}))


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    rows: list[PerfRow] = getattr(config, "_perf_rows", [])
    if not rows:
        return
    tw = terminalreporter

    tw.write_sep("=", "perf summary (sorted by ratio; >1.00x means emmy wins)")
    tw.write_line(_format_table(rows))

    # A regression is a FINDING, not a failure: the timing-refresh workflow turns it into a
    # labelled pull request a human accepts or declines. A lane that goes red because one
    # legitimate correctness fix cost latency is a lane nobody reads.
    slower = [r for r in rows if r.regressed]
    if slower:
        tw.write_sep("-", f"{len(slower)} case(s) slower than their recorded latency (band {100 * helpers.LATENCY_BAND:.0f}%)")
        for r in sorted(slower, key=lambda r: -(r.emmy_us / (r.recorded_us or 1))):
            tw.write_line(f"  {r.name:56s} {r.emmy_us:9.2f} us against a recorded {r.recorded_us:9.2f} us")
        tw.write_line("Accept a new baseline with `emmy run --golden <case> --realization <name> --bench --record`.")

    # Coverage grows by being asked once, on a card that can answer, and never anywhere else.
    missing = [r.name for r in rows if r.recorded_us is None]
    if missing:
        card = helpers.live_hardware_id()
        tw.write_sep("-", f"{len(missing)} case(s) have no {card} latency recorded")
        for name in sorted(missing):
            tw.write_line(f"  {name}")
        tw.write_line("Record them with the same command, or let the corpus-timings workflow fill them.")

    # Persist JSON for cross-run diffing. ncu metrics (when collected)
    # are nested under each row's ``ncu`` field; aggregated convenience
    # values are also written so downstream tooling doesn't need to
    # duplicate the time-weighted-average reduction.
    _RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%SZ")
    out = _RESULTS_DIR / f"{stamp}.json"
    payload = {
        "timestamp_utc": stamp,
        "git_rev": os.environ.get("EMMY_GIT_REV", ""),
        "rows": [
            {
                **asdict(r),
                "tags": list(r.tags),
                "ncu_aggregate": _aggregate_ncu(r.ncu),
            }
            for r in rows
        ],
    }
    out.write_text(json.dumps(payload, indent=2))
    tw.write_line(f"perf results saved to {out}")

    plot = _RESULTS_DIR / f"{stamp}.html"
    html = _render_plot(rows, stamp)
    plot.write_text(html)
    tw.write_line(f"perf plot saved to {plot}")

    png = plot.with_suffix(".png")
    try:
        from emmy.visualize.image import render as render_image

        render_image(html, png, transparent=True)
        tw.write_line(f"perf plot image saved to {png}")
    except ImportError:
        tw.write_line("perf plot PNG skipped (install '.[visualize]' + 'playwright install chromium')")
    except Exception as e:  # noqa: BLE001
        tw.write_line(f"perf plot PNG skipped: {e}")


# ---------------------------------------------------------------------------
# ECharts plot
# ---------------------------------------------------------------------------


def _render_plot(rows: list[PerfRow], stamp: str) -> str:
    """Render a self-contained HTML page comparing per-case speedup of
    Emmy and ``torch.compile`` against the PyTorch-eager baseline.
    Cases are sorted by Emmy ratio (wins at top). The shared
    ``emmy.visualize.bar_chart`` picks orientation based on case
    count — typical perf runs land in horizontal mode."""
    from emmy.visualize.bar_chart import Bar, BarChart, render_bar_chart

    rows_sorted = sorted(rows, key=lambda r: r.ratio, reverse=True)

    def _tcomp_ratio(r: PerfRow) -> float | None:
        if r.torch_compile_us is None or r.torch_compile_us <= 0:
            return None
        return round(r.torch_us / r.torch_compile_us, 3)

    tcomp_color = "#ffd166"
    depl_color = "#4dabf7"

    tooltip_rows: list[str] = []
    for r in rows_sorted:
        tr = _tcomp_ratio(r)
        t_us = round(r.torch_compile_us, 1) if r.torch_compile_us is not None else None
        tooltip_rows.append(
            "<br>".join(
                [
                    f"<b>{r.name}</b>",
                    f'<span style="color:#888">{r.shape}</span>',
                    "",
                    f'<span style="color:#999">■</span> eager: {round(r.torch_us, 1)} µs (1.00×)',
                    (f'<span style="color:{tcomp_color}">■</span> torch.compile: ' + ("—" if t_us is None else f"{t_us} µs ({tr:.2f}×)")),
                    (f'<span style="color:{depl_color}">■</span> emmy: {round(r.emmy_us, 1)} µs ({round(r.ratio, 2):.2f}×)'),
                ]
            )
        )

    chart = BarChart(
        categories=[r.name for r in rows_sorted],
        bars=[
            Bar(
                name="emmy / eager",
                values=[round(r.ratio, 3) for r in rows_sorted],
                color=depl_color,
            ),
            Bar(
                name="torch.compile / eager",
                values=[_tcomp_ratio(r) for r in rows_sorted],
                color=tcomp_color,
            ),
        ],
        value_name="speedup vs eager (×)",
        title=f"Per-kernel speedup vs PyTorch eager — {stamp}",
        subtitle=(
            "FP32 on RTX 5090. Ratio = eager_us / backend_us (higher is faster). "
            "Eager is the baseline at 1.0 (dashed line). Sorted by emmy ratio: "
            "wins at top, losses at bottom."
        ),
        baseline=1.0,
        baseline_label="1.0× (eager)",
        tooltip_rows=tooltip_rows,
        orientation="horizontal",
    )
    return render_bar_chart(chart, theme="dark", transparent=True)
