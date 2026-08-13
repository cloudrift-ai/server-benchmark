#!/usr/bin/env python3
"""Audit which recorded goldens actually deploy against a set of graphs.

Compiles each graph with the GOLDEN TIER AS THE ONLY EVIDENCE (`tune_db=None`, and point
`EMMY_ONLINE_FILE` at a nonexistent path) and reports, per fork, one of:

  MATCH  — a golden keyed to this fork's ShapeKey was realized by an offered candidate
  DRIFT  — a golden matched the shape but NO offered candidate realizes it (the golden no
           longer realizes under the current enumeration; deploy falls through). This is
           the signal that a graph change has invalidated a recorded golden.
  GAP    — no golden recorded for this fork's ShapeKey

Use it to check that the golden YAMLs are still relevant after any tracer / recognizer /
enumeration change: DRIFT means a recorded config went stale, GAP means an uncovered shape.

For the self-capturing form — no graph files needed, the serving twins are re-traced
weight-free from the pinned serving config — use ``emmy eval golden GOLDEN_YAML --serving-config PATH``
(same verdicts, machine-independent, non-zero exit on DRIFT; also the CI gate's path).

Usage:
    python scripts/diagnostics/audit_golden_match.py _tune/twins-refresh/*.json
    python scripts/diagnostics/audit_golden_match.py --json out.json _tune/twins-refresh/pre32.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

# Add project root to path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def audit_graph(path: Path) -> list[dict]:
    """Compile one graph JSON against the golden tier alone; return a per-fork verdict list.

    Rides the supported ``search/audit.audit_graph`` seam (the library the
    ``emmy eval golden GOLDEN_YAML --serving-config PATH`` audit and the CI gate share) with the live device's
    context — this script's job is auditing explicit graph JSONs on a real box."""
    from emmy.compiler.graph import Graph
    from emmy.compiler.pipeline.search.audit import audit_graph as _audit_graph

    graph = Graph.from_dict(json.loads(path.read_text()))
    records = _audit_graph(graph)
    for r in records:
        r["key"] = str(r["key"])  # ShapeKey → str for the --json dump
    return records


def main():
    parser = argparse.ArgumentParser(description="Audit golden-tier coverage of a set of graphs")
    parser.add_argument("graphs", nargs="+", help="Graph IR JSON files to audit")
    parser.add_argument("--json", help="Write the full per-fork records here")
    parser.add_argument("--verbose", action="store_true", help="Print every fork, not just the summary")
    args = parser.parse_args()

    all_records: dict[str, list[dict]] = {}
    totals: Counter = Counter()
    for raw in args.graphs:
        path = Path(raw)
        logger.info("=== %s ===", path.name)
        try:
            records = audit_graph(path)
        except Exception as ex:  # noqa: BLE001 — one bad graph must not sink the sweep
            logger.error("  COMPILE FAILED: %s", ex)
            totals["compile_fail"] += 1
            continue
        all_records[path.name] = records
        counts = Counter(r["verdict"] for r in records)
        totals.update(counts)
        logger.info("  MATCH %d  DRIFT %d  GAP %d", counts["MATCH"], counts["DRIFT"], counts["GAP"])
        for r in records:
            if args.verbose or r["verdict"] == "DRIFT":
                us = f"{r['us']:.1f}us" if r["us"] else "-"
                logger.info("    %-5s %-28s %s  golden=%s %s", r["verdict"], r["node"], r["key"], r["golden"], us)

    logger.info(
        "\n=== TOTAL: MATCH %d  DRIFT %d  GAP %d  compile_fail %d ===",
        totals["MATCH"],
        totals["DRIFT"],
        totals["GAP"],
        totals["compile_fail"],
    )
    if args.json:
        Path(args.json).write_text(json.dumps(all_records, indent=2))
        logger.info("wrote %s", args.json)


if __name__ == "__main__":
    main()
