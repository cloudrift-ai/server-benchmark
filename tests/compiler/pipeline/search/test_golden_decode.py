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

import pytest

from emmy.compiler.pipeline.search.golden import decode_record, load_golden_file, load_golden_records

_GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "../../../../emmy/compiler/pipeline/search/goldens")

_MIGRATED: set[str] = {
    "rtx5090_sm120.yaml",
    "rtx5090_sm120_gemma4_base.yaml",
    "rtxpro6000_sm120.yaml",
    "rtx5090_sm120_olmoe.yaml",
    "rtx4080_sm89.yaml",
}

_FILES = sorted(glob.glob(os.path.join(_GOLDEN_DIR, "*.yaml")))
assert _FILES, f"no golden YAMLs under {_GOLDEN_DIR}"


@pytest.mark.parametrize("path", _FILES, ids=[os.path.basename(f) for f in _FILES])
def test_every_migrated_record_decodes_strictly(path):
    fname = os.path.basename(path)
    if fname not in _MIGRATED:
        pytest.skip(f"{fname}: not yet strict-replayed onto the current codec (see _MIGRATED)")
    failures = []
    records = load_golden_records(load_golden_file(path))
    assert records, f"{fname}: no records — an empty migrated set should have been deleted"
    for record in records:
        reason = decode_record(record)
        if reason is not None:
            failures.append(f"{record.name}: {reason}")
    assert not failures, f"{fname}: records that no longer decode:\n" + "\n".join(failures)
