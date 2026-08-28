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
        after = verdict(helpers.Case(path=path, document=fresh, record=_record(fresh), xfail_stage=case.xfail_stage))
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


def _record(document: dict):
    from emmy.compiler.pipeline.search.golden import golden_record_from_entry  # noqa: PLC0415

    entry = document["configs"][0]
    return golden_record_from_entry(document, entry, entry["realizations"][0])


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    return regenerate_all()


if __name__ == "__main__":
    raise SystemExit(main())
