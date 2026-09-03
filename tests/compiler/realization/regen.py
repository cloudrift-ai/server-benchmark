"""Apply the fix the corpus's staleness test detects — ``make test-corpus-regen``.

Detection lives in ``test_realization.py``; this only writes. The split is the repository's
existing shape: ``ruff format --check`` detects while ``make format`` fixes.

Two refusals keep it from laundering anything. A case whose *verdict* changed is not restamped —
if one commit moves an identity and breaks realization, fixing the first must not let the second
ride along under a mechanical command. And a knob spelling the codec no longer accepts is an
error, not something to canonicalize to itself.
"""

from __future__ import annotations

import argparse
import sys

from tests.compiler.realization import helpers


def verdict(case: helpers.Case) -> str:
    """The GPU-free half of the case's outcome, as a comparable string."""
    return f"offered={helpers.offered(case) is None} realized={helpers.realized(case) is None}"


def regenerate_all() -> int:
    stale: list[str] = []
    refused: list[str] = []
    for path in helpers.case_files():
        try:
            case = helpers.load_case(path)
            fresh = helpers.regenerate(case.document)
        except helpers.CaseError as exc:
            refused.append(f"{path.name}: {exc}")
            continue
        if fresh == case.document:
            continue
        before = verdict(case)
        after = verdict(helpers.Case(path=path, document=fresh, records=_records(fresh), xfail_stage=case.xfail_stage))
        if before != after:
            refused.append(f"{path.name}: the verdict changed with the derived half ({before} -> {after})")
            continue
        stale.append(path.name)
        helpers.write_case(path, fresh)

    for message in refused:
        print(f"refused: {message}", file=sys.stderr)
    if stale:
        print("restamped: " + ", ".join(sorted(stale)))
    elif not refused:
        print("corpus is current")
    return 1 if refused else 0


def _records(document: dict) -> tuple:
    """The regenerated document's entries as records, the way ``load_case`` reads them."""
    entry = document["configs"][0]
    return tuple(helpers.golden_record_from_entry(document, entry, realization) for realization in entry["realizations"])


def complete_all() -> int:
    """Add an entry for every kernel of a case's set that no entry decides yet
    (``helpers.complete``), then restamp — authoring, so it runs only when asked
    (``make test-corpus-regen COMPLETE=1``)."""
    for path in helpers.case_files():
        case = helpers.load_case(path)
        before = len(case.document["configs"][0]["realizations"])
        document = helpers.complete(case.document)
        if len(document["configs"][0]["realizations"]) != before:
            helpers.write_case(path, helpers.regenerate(document))
            added = len(document["configs"][0]["realizations"]) - before
            print(f"completed {path.relative_to(helpers.CASES_DIR).as_posix()}: +{added} entries")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--complete", action="store_true", help="also add an entry for every kernel a case's set leaves undescribed")
    args = parser.parse_args(argv)
    if args.complete and (code := complete_all()):
        return code
    return regenerate_all()


if __name__ == "__main__":
    raise SystemExit(main())
