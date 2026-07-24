#!/usr/bin/env python
"""Freeze the autotune DB's ``node`` table into a digest-pinned per-GPU YAML directory.

The node DB is a live store (tunes and merges write into it), so a prior fit from it is
not reproducible. This wraps the tested core
:func:`emmy.compiler.pipeline.search.data.freeze.write_freeze`: read the DB read-only,
keep every identity-carrying leaf in the current featurizer vocabulary that passes the
plausibility predicates (``bench_fail`` leaves kept as negatives; branch rows and
identity-less legacy rows never freeze), and write a freeze DIRECTORY mirroring the
``goldens/`` layout — one golden-spelled YAML per (gpu, compute_cap) plus a
``manifest.json`` whose sha256s pin exactly which measurements a fit saw. Freezing the
same DB twice yields the same digests.

    ./venv/bin/python scripts/freeze_node_store.py                    # freeze the local store
    ./venv/bin/python scripts/freeze_node_store.py --db /path/autotune.db --out /path/freeze/
    ./venv/bin/python scripts/freeze_node_store.py --note "three-slice sweep, 4090+5090"

The source DB is never modified. Evals accept the freeze wherever they accept the DB:
``emmy eval online --dataset nodes --db <freeze-dir>``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from emmy.commands.compile import resolve_tune_db
from emmy.compiler.pipeline.search.data.freeze import write_freeze


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")  # surface the drop-reason tally
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", help="Autotune DB to freeze (default: EMMY_TUNE_DB or ~/.cache/emmy/autotune.db).")
    p.add_argument("--out", help="Freeze directory to write (default: <db-stem>-freeze/ next to the DB).")
    p.add_argument("--note", default="", help="Freeform collection-policy note stamped into the manifest.")
    args = p.parse_args()

    db_path = Path(args.db).expanduser() if args.db else resolve_tune_db()
    if not db_path.exists():
        p.error(f"no autotune DB at {db_path}")
    out = Path(args.out).expanduser() if args.out else db_path.with_name(f"{db_path.stem}-freeze")

    manifest = write_freeze(db_path, out, note=args.note)
    counts = manifest["counts"]
    per_gpu = ", ".join(f"{gpu or '(unknown card)'}: {n}" for gpu, n in counts["per_gpu"].items())
    print(f"[freeze_node_store] froze {counts['rows']} leaf row(s) ({counts['ok']} ok + {counts['bench_fail']} bench_fail) from {db_path}")
    print(f"[freeze_node_store]   per card: {per_gpu}")
    print(f"[freeze_node_store]   commit {manifest['repo_commit']}, {len(manifest['run_ids'])} run id(s)")
    print(f"[freeze_node_store]   sha256 {manifest['sha256']} over {len(manifest['files'])} per-GPU file(s)")
    print(f"[freeze_node_store]   -> {out}/")


if __name__ == "__main__":
    sys.exit(main())
