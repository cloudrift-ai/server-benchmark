"""Strict decode of the checked-in golden corpus.

Off the default lane behind the ``goldens`` marker: a full pass re-derives every recorded row's
enumeration, which costs minutes per file. Run it with ``make test-goldens`` after a tuning round
has re-recorded a card's rows, to see which files the compiler can still replay.
"""

from pathlib import Path

import pytest

from emmy.compiler.pipeline.search.golden import (
    _records_of,
    _repository_golden_paths,
    decode_record,
    flush_identity_store,
    siblings_of,
)


def _paths() -> list[Path]:
    with _repository_golden_paths() as paths:
        return list(paths)


def _label(path: Path) -> str:
    """The card's file for a hardware golden, ``<model>/<card>.yaml`` for a recipe-local one."""
    return f"{path.parent.parent.name}/{path.name}" if path.parent.name == "golden" else path.name


@pytest.mark.goldens
@pytest.mark.parametrize("path", _paths(), ids=_label)
def test_every_recorded_row_still_decodes(path: Path) -> None:
    """Every row a golden file records must still equal an enumerated leaf of its own target.

    A row that does not is no evidence a deploy can use: the compile that reads it either picks a
    schedule the enumeration no longer offers, or falls through to the prior. Decoding replays the
    persisted program at the record's DECLARED capability, so this holds on any machine — a card is
    needed to re-record a stale row, not to detect one.
    """
    records = _records_of(path)
    failures = []
    for record in records:
        try:
            reason = decode_record(record, siblings_of(record, records))
        except Exception as exc:  # noqa: BLE001 — the reason IS the product here
            reason = f"{type(exc).__name__}: {exc}"
        if reason is not None:
            failures.append(f"  {record.name}: {' '.join(reason.split())[:120]}")
    # Persist what this pass derived: the memo is keyed by compiler fingerprint and record content,
    # so re-running after a tuning round only re-derives the rows that were actually re-recorded.
    flush_identity_store()
    listed = "\n".join(failures[:20])
    more = f"\n  ... and {len(failures) - 20} more" if len(failures) > 20 else ""
    assert not failures, f"{len(failures)}/{len(records)} recorded rows equal no enumerated leaf:\n{listed}{more}"
