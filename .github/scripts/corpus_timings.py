"""Bound what a realization-corpus timing run may select and may commit.

Stage 5 only bites where a timing exists, and a timing can only be produced on the card it
describes. This is the enforcement half of the workflow that fills them: which cases a given card
may measure, and the rule that a measurement run changes nothing but measurements.

The judgment half — which of the resulting rows is worth a compiler change — is not here. Emitting
the table is mechanical; reading it is not.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

CASES_DIR = Path("tests/compiler/realization/cases")


def compute_cap_of(path: Path) -> tuple[int, int] | None:
    """The capability a case declares, read without loading the compiler."""
    import yaml  # noqa: PLC0415

    try:
        document = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError):
        return None
    cap = (document or {}).get("compute_cap")
    if isinstance(cap, list) and len(cap) == 2 and all(isinstance(part, int) for part in cap):
        return (cap[0], cap[1])
    return None


def selectable(workspace: Path, cap: tuple[int, int]) -> list[Path]:
    """The closed cases this card may measure.

    Two bounds, both deliberate. The capability must match exactly, the same rule the build and
    accuracy stages use — a pinned schedule is a claim about one capability, not about a merely
    newer card. And open cases are excluded, because their schedule never runs and demanding a
    latency for one would be a false attribution.
    """
    return sorted(path for path in (workspace / CASES_DIR).rglob("*.yaml") if "_xfail" not in path.stem and compute_cap_of(path) == cap)


def choose_gpu(available: list[str], seed: int) -> str | None:
    """One GPU name, uniformly at random from what is currently rentable, reproducibly from a seed.

    Chosen by NAME rather than by instance type so the rental can go through `emmy vm create gpu`,
    which is the form that persists a lease and prints connection details. The name-to-type map is
    `hardware.GPU_INSTANCE_TYPES`, the same one the recipe query's availability filter reads, at a
    single GPU: this measures kernels, and there is no reason to hold an eight-GPU node to do it.

    Selection deliberately does NOT try to pick a card matching some case's capability. Nothing in
    the tree maps a card name to a compute capability, and adding a table would be a second thing
    to maintain and get wrong. Discover the capability on the host instead, then select the cases
    that match — which also makes an unseen card a graceful no-op rather than a failure.
    """
    from emmy.hardware import GPU_INSTANCE_TYPES, resolve_instance_type  # noqa: PLC0415

    offered = set(available)
    rentable = sorted(
        name
        for name, candidates in GPU_INSTANCE_TYPES.items()
        if any(provider == "cloudrift" and resolve_instance_type(provider, base, 1) in offered for provider, base in candidates)
    )
    if not rentable:
        return None
    return random.Random(seed).choice(rentable)


def _without_latency(document: object) -> object:
    """A case document with every realization's measured block removed."""
    if not isinstance(document, dict):
        return document
    stripped = {key: value for key, value in document.items() if key != "configs"}
    stripped["configs"] = [
        {
            **{key: value for key, value in entry.items() if key != "realizations"},
            "realizations": [{key: value for key, value in row.items() if key != "latency"} for row in entry.get("realizations", [])],
        }
        for entry in document.get("configs", [])
    ]
    return stripped


def _at_head(workspace: Path, path: str) -> object:
    import yaml  # noqa: PLC0415

    result = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=workspace, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return yaml.safe_load(result.stdout)


def validate_diff(workspace: Path) -> list[str]:
    """Every changed case whose non-measured half also moved.

    Compared structurally rather than by diff lines: strip each realization's ``latency`` block
    from both revisions and require the rest to be equal. A textual check would have to model the
    dump's indentation, and would pass or fail on formatting rather than on meaning.
    """
    import yaml  # noqa: PLC0415

    offenders: list[str] = []
    for path in changed_files(workspace):
        before = _at_head(workspace, path)
        if before is None:
            offenders.append(f"{path}: a timing run may not add a case")
            continue
        try:
            after = yaml.safe_load((workspace / path).read_text())
        except (OSError, yaml.YAMLError) as exc:
            offenders.append(f"{path}: unreadable after the run ({exc})")
            continue
        if _without_latency(before) != _without_latency(after):
            offenders.append(f"{path}: the derived half changed, so the corpus it measured was stale")
    return offenders


def changed_files(workspace: Path) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "HEAD", "--name-only", "--", str(CASES_DIR)],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def _untracked(workspace: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser("select", help="list the closed cases this capability may measure")
    select.add_argument("--compute-cap", required=True, help="live capability as MAJOR.MINOR, e.g. 9.0")

    choose = subparsers.add_parser("choose", help="pick one rentable GPU name reproducibly from a run id")
    choose.add_argument("--seed", type=int, required=True)

    subparsers.add_parser("validate", help="refuse a diff that changes anything but a latency entry")

    args = parser.parse_args(argv)

    if args.command == "select":
        major, _, minor = args.compute_cap.partition(".")
        for path in selectable(args.workspace, (int(major), int(minor))):
            print(path.relative_to(args.workspace).as_posix())
        return 0

    if args.command == "choose":
        available = json.load(sys.stdin)
        chosen = choose_gpu(available, args.seed)
        if chosen is None:
            print("no known GPU is currently rentable at a single-GPU instance type", file=sys.stderr)
            return 1
        print(chosen)
        return 0

    offenders = validate_diff(args.workspace)
    stray = _untracked(args.workspace)
    for line in offenders:
        print(f"a timing run may only change a latency entry: {line}", file=sys.stderr)
    for path in stray:
        print(f"a timing run may not create files: {path}", file=sys.stderr)
    if offenders or stray:
        return 1
    print("\n".join(changed_files(args.workspace)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
