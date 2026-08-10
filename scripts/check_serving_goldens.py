#!/usr/bin/env python
"""Golden coverage gate for a serving-image release: does this (model, card) pair have
recorded goldens to seed the fork picks with?

Emmy's greedy compile resolves every fork through the deploy evidence hierarchy, and the
**recorded goldens for the live card are its top tier** — the verified schedules the warm
run's kernels get picked from. Release with no goldens for the model's shapes and the warm
bakes whatever cold greedy happened to choose, which on unseeded projection shapes is not a
slightly-worse kernel but a catastrophic one (a scalar tile ~770x off cuBLAS, and on some
shapes a pick that hangs outright). Those bad picks then get frozen into the shipped cubins
and the execution-plan pack, where nothing downstream will ever revisit them.

So this is a gate, not a report: run it before warming, and treat "no goldens" as a decision
point for a human, never as something to proceed through quietly.

    ./venv/bin/python scripts/check_serving_goldens.py --model google/gemma-4-12B-it
    ./venv/bin/python scripts/check_serving_goldens.py --model <id> --gpu "NVIDIA GeForce RTX 5090"
    ./venv/bin/python scripts/check_serving_goldens.py --model <id> --revision <sha>   # a tagged golden set

With ``--strict-major-gaps`` the gate also captures the release checkpoint's serving twins
weight-free and fails on drift, compile failure, or any uncovered warp-contraction fork.
``--release-config`` adds every decode/prefill width named by the pinned config and its warm
shape matrix to that audit.
``--static-only-release`` instead audits only M=1, but is accepted only when that release
config proves runner capacity 1, decode bucket 1, disabled prefill, scheduler maximum 1, and
no wider warm override. The default standard-width and symbolic audit remains unchanged.
``--checkpoint PATH`` traces an exact local/preseeded artifact while golden selection and
reporting remain bound to ``--model`` plus ``--revision``; omit it for normal Hub capture.

Exit codes: 0 = goldens found for this (model, card), and strict coverage passed when requested;
1 = none found, coverage cannot be evaluated (see the revision rule below), the card has no
golden file, or strict coverage failed; 2 = bad usage.
The message names what IS recorded, so the caller can tell "this card has nothing" from "this
card is tuned, but for other models" from "this card is tuned for this model, at another
revision".

Matching has two halves, the REPO and the REVISION.

**Repo** — by **model slug**, the same schema the image name uses (model_slug.sh), with a
prefix rule on `-` boundaries so a golden recorded against a base checkpoint covers its
instruction-tuned sibling: goldens tagged `google/gemma-4-12B` satisfy a release of
`google/gemma-4-12B-it`, because a fine-tune shares its base's layer geometry and therefore
its kernel shapes. A quantized or resized variant does NOT share them — those slugs differ
past the boundary and correctly miss.

**Revision** — a golden's `model:` provenance may be spelled `<repo>@<revision>`, because one
repo publishes several checkpoints that do NOT share kernel shapes: an EXL3 repo carries one
branch per bit rate, and the rungs differ in exactly the per-tensor bit allocation the shape
keys carry. The slug deliberately does not encode the revision (two rungs share one image
name), so the revision is compared separately, against `--revision` (the release config's
`SERVE_REVISION`; `make serve-goldens` forwards it):

  * an UNTAGGED golden makes no revision claim and covers every revision of its repo — that is
    the pre-existing behaviour, and every non-EXL3 golden file today is untagged;
  * a golden tagged `@R` covers a release of `R` and no other revision;
  * a golden tagged `@R` when the release named NO revision is **unevaluable**, not covered:
    the gate fails and says so, rather than reporting zero coverage for a card that plainly
    has some. Pass `--revision`.

Revisions compare as exact strings, with one allowance: an abbreviated hex commit sha matches
the full one it prefixes (`git rev-parse --short` is a legitimate spelling of the same commit).
A branch name and a commit sha never match each other — nothing here can resolve one to the
other offline, and guessing is what this gate exists to prevent.
"""

from __future__ import annotations

import argparse
import functools
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

_SLUG_SCRIPT = PROJECT_ROOT / "docker" / "vllm-emmy-serve" / "model_slug.sh"


@functools.cache
def model_slug(model: str) -> str:
    """The image-naming slug for an HF model id.

    Shells out to `model_slug.sh` rather than reimplementing it: the slug decides both the
    image repo and which `models/<slug>.env` the warm and the bake read, so two
    implementations that disagree would let those two steps load different configs — the
    cache-key parity failure the release contract is built to prevent. Memoized because the
    matchers below call it once per golden and a card carries hundreds.
    """
    if not _SLUG_SCRIPT.is_file():  # a source checkout always has it; a stripped one won't
        return re.sub(r"[^a-z0-9._-]+", "-", model.rsplit("/", 1)[-1].lower()).strip("._-")
    out = subprocess.run([str(_SLUG_SCRIPT), model], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def split_revision(model: str) -> tuple[str, str | None]:
    """``<repo>@<revision>`` → ``(repo, revision)``; a bare id → ``(id, None)``.

    HF ids carry no ``@``, so the separator is unambiguous. The revision half must never reach
    :func:`model_slug` — ``turboderp/GLM-4.5-Air-exl3@2.25bpw`` sanitizes to the slug
    ``glm-4.5-air-exl3-2.25bpw``, which matches no release and is what made this gate report
    zero coverage for a card that had some.
    """
    repo, sep, rev = model.rpartition("@")
    return (repo, rev.strip() or None) if sep and repo else (model, None)


def revision_matches(golden_rev: str | None, target_rev: str | None) -> bool:
    """Does a golden's revision claim admit the revision being released?

    ``None`` on the golden side is "no claim" — it covers every revision. ``None`` on the
    target side against a golden that DOES claim one is unevaluable, which is not coverage:
    the caller reports it as its own failure mode rather than folding it into "no goldens".
    """
    if golden_rev is None:
        return True
    if not target_rev:
        return False
    g, t = golden_rev.strip(), target_rev.strip()
    if g == t:
        return True
    # An abbreviated commit sha names the same commit as the full one it prefixes (the release
    # tag itself is built from `git rev-parse --short`). Hex-only and >= 7 chars, so a short
    # branch name can never prefix-match a sha.
    hexish = all(re.fullmatch(r"[0-9a-f]{7,40}", s) for s in (g, t))
    return hexish and (g.startswith(t) or t.startswith(g))


def covers_repo(golden_model: str | None, target_slug: str) -> bool:
    """Does ``golden_model``'s REPO half name the same checkpoint family as ``target_slug``?

    Exact slug match, or the golden's slug being a `-`-boundary prefix of the target's (the
    base-checkpoint rule above). Untagged goldens (``model: None``) never count — a shape
    with no recorded provenance may belong to any model, and guessing here would report
    coverage that does not exist.
    """
    if not golden_model:
        return False
    g = model_slug(split_revision(golden_model)[0])
    return g == target_slug or target_slug.startswith(f"{g}-")


def covers(golden_model: str | None, target_slug: str, target_revision: str | None = None) -> bool:
    """Does a golden recorded against ``golden_model`` cover a release of this (slug, revision)?

    Both halves must admit it — see the module docstring for the repo and revision rules.
    """
    return covers_repo(golden_model, target_slug) and revision_matches(split_revision(golden_model or "")[1], target_revision)


def select_goldens(goldens, target_slug: str, target_revision: str | None) -> tuple[list, list]:
    """Partition ``goldens`` into ``(matched, wrong_revision)`` for this release.

    ``wrong_revision`` holds the goldens whose repo matches but whose revision claim does not
    (including "does not, because the release named none"). Keeping them separate is the whole
    point: they are the difference between "this card is not tuned for your model" and "it is,
    but for another checkpoint of it", and the second must not be reported as the first.
    """
    matched, wrong_rev = [], []
    for g in goldens:
        if not covers_repo(g.model, target_slug):
            continue
        (matched if revision_matches(split_revision(g.model)[1], target_revision) else wrong_rev).append(g)
    return matched, wrong_rev


def _revisions(goldens) -> str:
    """The distinct revision tags of ``goldens``, for a message."""
    return ", ".join(sorted({split_revision(g.model)[1] or "(untagged)" for g in goldens}))


def _provenance(golden_model: str) -> str:
    """A golden's provenance as ``slug`` / ``slug@revision`` — org-collapsed like the target it
    is listed beside, but with the revision kept OUT of the slug (running the whole tagged id
    through ``model_slug`` yields ``glm-4.5-air-exl3-2.25bpw``, a slug no release ever has)."""
    repo, rev = split_revision(golden_model)
    return f"{model_slug(repo)}@{rev}" if rev else model_slug(repo)


def release_widths(path: str | Path) -> tuple[int, ...]:
    """Decode/prefill twin widths configured by one pinned serving env file."""
    source = Path(path)
    values = {}
    for raw in source.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')

    widths: set[int] = set()

    def add(
        raw: str,
        *,
        field: str,
        allow_zero: bool = False,
        allow_default: bool = False,
        record: bool = True,
    ) -> int | None:
        if not raw:
            return None
        try:
            width = int(raw)
        except ValueError as exc:
            raise ValueError(f"{source}: {field} must be an integer, got {raw!r}") from exc
        if width == 0 and allow_zero:
            return width
        if width == -1 and allow_default:
            return width
        if width <= 0:
            raise ValueError(f"{source}: {field} must be positive, got {width}")
        if record:
            widths.add(width)
        return width

    add(values.get("SERVE_DECODE_BUCKET", ""), field="SERVE_DECODE_BUCKET", allow_zero=True)
    capacity = add(
        values.get("SERVE_PREFILL_CAPACITY", ""),
        field="SERVE_PREFILL_CAPACITY",
        allow_default=True,
        record=False,
    )
    bucket = add(
        values.get("SERVE_PREFILL_BUCKET", ""),
        field="SERVE_PREFILL_BUCKET",
        allow_zero=True,
        allow_default=True,
    )
    # With no explicit static bucket (or -1), serving defaults it to the configured
    # capacity; an explicit 0 disables the static twin and leaves capacity symbolic-only.
    if bucket in {None, -1} and capacity not in {None, -1}:
        widths.add(capacity)
    for spec in values.get("SERVE_WARM_SHAPES", "").split():
        fields = spec.split(":")
        if len(fields) not in {3, 4}:
            raise ValueError(f"{source}: invalid SERVE_WARM_SHAPES entry {spec!r}")
        add(fields[0], field=f"{spec} decode bucket", allow_zero=True)
        add(fields[1], field=f"{spec} prefill bucket", allow_zero=True)
    return tuple(sorted(widths))


def release_capture_source(repo: str, revision: str | None, checkpoint: str | None) -> tuple[str, str]:
    """Return ``(trace source, reported provenance)`` without conflating the two."""
    provenance = f"{repo}@{revision}" if revision else repo
    return checkpoint or provenance, provenance


def audit_release_twins(
    capture_model: str,
    gpu_name: str,
    caps: list[tuple[int, int]],
    widths: tuple[int, ...],
    *,
    provenance: str | None = None,
    static_only: bool = False,
) -> bool:
    """Return whether a release model has no drift, compile failure, or major gap."""
    from emmy.compiler.pipeline.search.audit import COMPILE_FAIL, audit_card, major_gap_keys, summarize
    from emmy.serving.twins import capture_twin_graphs

    release = provenance or capture_model
    source = f" from local checkpoint {capture_model!r}" if provenance and provenance != capture_model else ""
    scope = "static-only M=1" if static_only else f"standard plus extra widths {list(widths) or 'none'}"
    print(f"STRICT: tracing serving twins for {release!r}{source} (weight-free; {scope}).")
    if static_only:
        graphs = capture_twin_graphs(
            capture_model,
            decode_bucket=1,
            prefill_bucket=0,
            extra_widths=(),
            static_only=True,
        )
    else:
        graphs = capture_twin_graphs(capture_model, extra_widths=widths)
    passed = True
    for cap in caps:
        results = audit_card(graphs, gpu_name, cap)
        counts = summarize(results)
        majors = major_gap_keys(results)
        print(
            f"  sm_{cap[0]}{cap[1]}: MATCH {counts['MATCH']}  DRIFT {counts['DRIFT']}  "
            f"GAP {counts['GAP']}  compile_fail {counts[COMPILE_FAIL]}  major_gap {len(majors)}"
        )
        for key in sorted(majors, key=str):
            print(f"    MAJOR GAP: {key}")
        if counts["DRIFT"] or counts[COMPILE_FAIL] or majors:
            passed = False
    if not passed:
        print("FAIL: strict serving coverage found drift, compile failure, or an uncovered warp-contraction fork.")
        print("  Capture/tune/promote the exact release checkpoint's serving-twin inventory before warming the image.")
    return passed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF model id being released (a `<repo>@<revision>` spelling is accepted)")
    ap.add_argument(
        "--revision",
        default=None,
        help="checkpoint revision being released (the config's SERVE_REVISION) — required when this repo's goldens are revision-tagged",
    )
    ap.add_argument("--gpu", default=None, help="GPU name to check (default: the live card)")
    ap.add_argument(
        "--strict-major-gaps",
        action="store_true",
        help="Capture serving twins and fail on DRIFT, compile failure, or any major warp-contraction GAP.",
    )
    ap.add_argument(
        "--release-config",
        default=None,
        help="Pinned models/<slug>.env whose decode/prefill warm widths strict coverage must include.",
    )
    ap.add_argument(
        "--serving-width",
        action="append",
        type=int,
        default=None,
        metavar="N",
        help="Additional release width for strict serving-twin coverage. Repeatable.",
    )
    ap.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Local/preseeded quantized checkpoint to trace for strict coverage. Golden selection and provenance "
            "still use --model/--revision; default capture source is that Hub snapshot."
        ),
    )
    ap.add_argument(
        "--static-only-release",
        action="store_true",
        help=(
            "Audit only the static M=1 twins. Requires --strict-major-gaps and --release-config; "
            "the config must prove the exact static-only runner and scheduler envelope."
        ),
    )
    args = ap.parse_args()
    if args.release_config and not args.strict_major_gaps:
        ap.error("--release-config requires --strict-major-gaps")
    if args.serving_width and not args.strict_major_gaps:
        ap.error("--serving-width requires --strict-major-gaps")
    if args.checkpoint and not args.strict_major_gaps:
        ap.error("--checkpoint requires --strict-major-gaps")
    if args.static_only_release and not args.strict_major_gaps:
        ap.error("--static-only-release requires --strict-major-gaps")
    if args.static_only_release and not args.release_config:
        ap.error("--static-only-release requires --release-config")
    if args.static_only_release and args.serving_width:
        ap.error("--static-only-release cannot include additional --serving-width values")
    if args.serving_width and any(width <= 0 for width in args.serving_width):
        ap.error(f"--serving-width values must be positive, got {args.serving_width}")

    from emmy.compiler.pipeline.search.golden import GOLDEN_RECORDS, live_recorded_goldens

    repo, tagged = split_revision(args.model)
    revision = (args.revision or "").strip() or tagged
    target = model_slug(repo)

    if args.gpu:
        card = args.gpu
        on_card = [g for g in GOLDEN_RECORDS if g.gpu_name == card]
        if not on_card:
            known = sorted({g.gpu_name for g in GOLDEN_RECORDS if g.gpu_name})
            print(f"FAIL: no goldens recorded for {card!r}.")
            print(f"  cards with goldens: {', '.join(known) or '(none)'}")
            return 1
    else:
        on_card = live_recorded_goldens()
        if on_card is None:
            print("FAIL: no CUDA device visible — run this on the target card, or pass --gpu NAME.")
            return 1
        card = on_card[0].gpu_name if on_card else "(the live card)"
        if not on_card:
            print(f"FAIL: the live card has no recorded goldens at all ({card}).")
            print("  Nothing seeds the fork picks; the warm would bake cold-greedy kernels.")
            return 1

    matched, wrong_rev = select_goldens(on_card, target, revision)
    if not matched and wrong_rev and revision is None:
        # UNEVALUABLE, and the one case that must never read as "no goldens": the card is tuned
        # for this repo, but every entry claims a revision and the release named none.
        print(f"FAIL: {card} has {len(wrong_rev)} golden(s) for {repo!r}, but they are revision-tagged and this release named none.")
        print(f"  coverage CANNOT be evaluated. recorded revision(s): {_revisions(wrong_rev)}")
        print("  Pass --revision <sha> (the config's SERVE_REVISION, which `make serve-goldens` forwards).")
        return 1
    if not matched and wrong_rev:
        print(f"FAIL: {card} has {len(wrong_rev)} golden(s) for {repo!r}, but recorded against {_revisions(wrong_rev)}, not {revision!r}.")
        print("  A repo's revisions do not share kernel shapes — an EXL3 rung differs in exactly the per-tensor")
        print("  bit allocation the shape keys carry — so another revision's goldens are not coverage for this one.")
        print("  Seed goldens for this revision, or re-tag the golden file's `model:` header if the two spellings")
        print("  really name one checkpoint (a branch name and a commit sha are not resolvable to each other here).")
        return 1
    if not matched:
        others = Counter(_provenance(g.model) for g in on_card if g.model)
        print(f"FAIL: {card} has goldens, but none recorded for {args.model!r} (slug {target!r}).")
        if others:
            listed = ", ".join(f"{m} ({n})" for m, n in sorted(others.items()))
            print(f"  models this card IS tuned for: {listed}")
        untagged = sum(1 for g in on_card if not g.model)
        if untagged:
            print(f"  plus {untagged} golden(s) with no model provenance — not counted, see --help")
        print("  Releasing anyway bakes cold-greedy fork picks into the image's cubins and pack.")
        return 1

    kinds = Counter("+".join(g.origin_ops) for g in matched)
    at = f" at revision {revision!r}" if revision else ""
    print(f"OK: {len(matched)} golden(s) on {card} cover {repo!r} (slug {target!r}){at}.")
    print(f"  kinds: {', '.join(f'{k}={n}' for k, n in sorted(kinds.items()))}")
    provenance = sorted({g.model for g in matched if g.model})
    print(f"  recorded against: {', '.join(provenance)}")
    if wrong_rev:
        # Not a failure — something else covered the release — but never silent: these entries
        # WILL be consulted by the fork resolution (the `model:` tag is provenance, not a join
        # key), and they are not evidence for this checkpoint.
        print(f"  NOTE: {len(wrong_rev)} further golden(s) for this repo were NOT counted — recorded against {_revisions(wrong_rev)}.")
    if args.strict_major_gaps:
        widths = set(args.serving_width or ())
        if args.static_only_release:
            from emmy.serving.twins import validate_static_only_release_config

            try:
                validate_static_only_release_config(args.release_config)
            except (OSError, ValueError) as exc:
                print(f"ERROR: static-only release scope is unsafe: {exc}")
                return 2
        elif args.release_config:
            try:
                widths.update(release_widths(args.release_config))
            except (OSError, ValueError) as exc:
                print(f"ERROR: cannot read release widths: {exc}")
                return 2
        capture_source, model_ref = release_capture_source(repo, revision, args.checkpoint)
        caps = sorted({tuple(g.compute_cap) for g in matched})
        if not audit_release_twins(
            capture_source,
            card,
            caps,
            tuple(sorted(widths)),
            provenance=model_ref,
            static_only=args.static_only_release,
        ):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
