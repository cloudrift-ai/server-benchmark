#!/usr/bin/env python
"""Purge physically-impossible rows from an autotune DB's ``node`` table.

One-time repair for stores written before :meth:`SearchDB.record_nodes` grew its
write-time plausibility gate: the 2026-07-08/09 golden sweeps ran pre-#330, so split-K
variants whose over-budget staged main kernel was rejected at materialize benched
combine-kernel-only and landed as impossibly-fast ``ok`` leaves (a ~9 µs "matmul" —
thousands of TFLOP/s), min-propagated up their branch ancestries. This wraps the tested
core :meth:`SearchDB.purge_implausible`: flagged leaves are deleted; a flagged branch
with surviving honest leaves below it is *repaired* (its value-of-position bound
recomputed) so the fork-tree structure the diagnostics group on survives.

    ./venv/bin/python scripts/purge_node_store.py --dry-run     # count, change nothing
    ./venv/bin/python scripts/purge_node_store.py               # backup + purge the local store
    ./venv/bin/python scripts/purge_node_store.py --db /path/autotune.db

A timestamped ``.bak-pre-purge-*`` copy of the DB is written next to it before any
change (skipped for ``--dry-run``).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

from emmy.commands.compile import resolve_tune_db
from emmy.compiler.pipeline.search.db import SearchDB


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", help="Autotune DB to purge (default: EMMY_TUNE_DB or ~/.cache/emmy/autotune.db).")
    p.add_argument("--dry-run", action="store_true", help="Report what would change without touching the DB.")
    args = p.parse_args()

    db_path = Path(args.db).expanduser() if args.db else resolve_tune_db()
    if not db_path.exists():
        p.error(f"no autotune DB at {db_path}")

    if not args.dry_run:
        backup = db_path.with_name(f"{db_path.name}.bak-pre-purge-{datetime.now(UTC):%Y%m%d_%H%M%S}")
        shutil.copy2(db_path, backup)
        print(f"[purge_node_store] backup -> {backup}")

    db = SearchDB(db_path) if not args.dry_run else SearchDB.open_readonly(db_path)
    try:
        receipt = db.purge_implausible(dry_run=args.dry_run)
    finally:
        db.close()
    verb = "would delete" if args.dry_run else "deleted"
    print(
        f"[purge_node_store] {verb} {receipt['deleted_leaves']} leaf row(s) + "
        f"{receipt['deleted_branches']} branch row(s); "
        f"{'would repair' if args.dry_run else 'repaired'} {receipt['repaired_branches']} branch bound(s)  ({db_path})"
    )


if __name__ == "__main__":
    sys.exit(main())
