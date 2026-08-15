"""The corpus decode tripwire: every repository golden record decodes STRICTLY.

A record's persisted program must select exactly one kernel under the current compiler, and its
knobs must decode fail-closed — a routing record's ``PLACE`` keys resolve to legal cut seams on
the recognized tree; a schedule record's spelled row equals EXACTLY ONE enumerated leaf under the
record's own pins (``decode_record``). A structural change that invalidates a stored row fails
HERE, loudly, with the reason — the row is then re-recorded or removed by hand, never silently
carried.

``_MIGRATED`` lists the golden sets already strict-replayed onto the current codec; it grows one
file per corpus-migration commit and the listing (with this comment) is DELETED once every set is
migrated — from then on the gate is unconditional over the whole corpus."""

from __future__ import annotations

import glob
import os

from emmy.compiler.pipeline.search.golden import decode_record, load_golden_file, load_golden_records

_GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "../../../../emmy/compiler/pipeline/search/goldens")

_MIGRATED: set[str] = set()


def test_every_migrated_record_decodes_strictly():
    files = sorted(glob.glob(os.path.join(_GOLDEN_DIR, "*.yaml")))
    assert files, f"no golden YAMLs under {_GOLDEN_DIR}"
    failures: list[str] = []
    checked = 0
    for path in files:
        fname = os.path.basename(path)
        if _MIGRATED and fname not in _MIGRATED:
            continue
        if not _MIGRATED:
            continue  # no set migrated yet — the gate arms with the first migration commit
        for record in load_golden_records(load_golden_file(path)):
            reason = decode_record(record)
            checked += 1
            if reason is not None:
                failures.append(f"{fname}:{record.name}: {reason}")
    assert not failures, "records that no longer decode:\n" + "\n".join(failures)
    if _MIGRATED:
        assert checked > 0, "migrated sets listed but no record was checked — did the loader change?"
