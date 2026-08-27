"""Session-end coverage report for the corpus's latency lane.

The two gates the corpus carries have deliberately different reach, and that asymmetry is the
point. The derived half is GPU-free, so its check fires everywhere and its fix works everywhere. A
timing is not: only a machine holding the card can produce one. If both fired everywhere, an agent
on a CPU box would face a failure it has no way to clear — so the timing gate is scoped to the
machine that can actually answer it, and it reports rather than fails.
"""

from __future__ import annotations


def pytest_terminal_summary(terminalreporter) -> None:
    """Name every closed case this card could measure but has no recorded latency for.

    Once, at session end, with the command that records them — the shape the root `conftest.py`
    already uses for the durations baseline, rather than N separate skips nobody reads.
    """
    missing: dict[str, list[str]] = {}
    for report in terminalreporter.stats.get("skipped", []):
        for name, value in getattr(report, "user_properties", []):
            if name == "missing_latency":
                missing.setdefault(value, []).append(report.nodeid.rpartition("[")[2].rstrip("]"))
    for hardware_id, cases in sorted(missing.items()):
        terminalreporter.write_sep("-", f"realization corpus: {len(cases)} case(s) have no {hardware_id} latency")
        for case in sorted(cases):
            terminalreporter.write_line(f"  {case}")
        terminalreporter.write_line(
            "Record them with `emmy run --golden-file <case> --golden <name> --bench "
            "--bench-backends eager,tcompile,emmy --record`, or let the corpus-timings workflow fill them."
        )
