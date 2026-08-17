"""The corpus decode tripwire: every repository golden record decodes STRICTLY.

A record's persisted program must select exactly one kernel under the current compiler, and its
knobs must decode fail-closed — a routing record's ``PLACE`` keys resolve to legal cut seams on
the recognized tree; a schedule record's spelled row equals EXACTLY ONE enumerated leaf under the
record's own pins (``decode_record``). A structural change that invalidates a stored row fails
HERE, loudly, with the reason — the row is then re-recorded or removed by hand, never silently
carried.

The gate covers the whole corpus, but it is **deselected by default** (``corpus`` marker) and runs
via ``make test-goldens``. Proving a row means enumerating its entire schedule fork, so the cost is
record count TIMES fork size — minutes per file on a cold cache, and cold is every CI run, since the
memo below lives outside the repo. Run it before landing a golden edit or a change to the codec,
recognition, or legality, which is when a stored row can actually be invalidated; a change that
touches neither cannot move a verdict.

Verdicts memo to the fingerprinted identity store (``~/.cache/emmy/golden_identity.json``): an
unchanged compiler tree revalidates in seconds; any compiler edit invalidates the memo and the next
run re-pays the full decode — exactly when the corpus must actually be re-proven."""

from __future__ import annotations

import glob
import os

import pytest

from emmy.compiler.pipeline.search.golden import decode_record, flush_identity_store, load_golden_file, load_golden_records

_GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "../../../../emmy/compiler/pipeline/search/goldens")

_FILES = sorted(glob.glob(os.path.join(_GOLDEN_DIR, "*.yaml")))
assert _FILES, f"no golden YAMLs under {_GOLDEN_DIR}"


@pytest.mark.corpus
@pytest.mark.parametrize("path", _FILES, ids=[os.path.basename(f) for f in _FILES])
def test_every_migrated_record_decodes_strictly(path):
    fname = os.path.basename(path)
    failures = []
    records = load_golden_records(load_golden_file(path))
    assert records, f"{fname}: no records — an empty migrated set should have been deleted"
    for record in records:
        reason = decode_record(record)
        if reason is not None:
            failures.append(f"{record.name}: {reason}")
    flush_identity_store()
    assert not failures, f"{fname}: records that no longer decode:\n" + "\n".join(failures)
