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

Exit codes: 0 = goldens found for this (model, card); 1 = none found (or the card has no
golden file at all); 2 = bad usage. The message names what IS recorded, so the caller can
tell "this card has nothing" from "this card is tuned, but for other models".

Matching is by **model slug**, the same schema the image name uses (model_slug.sh), with a
prefix rule on `-` boundaries so a golden recorded against a base checkpoint covers its
instruction-tuned sibling: goldens tagged `google/gemma-4-12B` satisfy a release of
`google/gemma-4-12B-it`, because a fine-tune shares its base's layer geometry and therefore
its kernel shapes. A quantized or resized variant does NOT share them — those slugs differ
past the boundary and correctly miss.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

_SLUG_SCRIPT = PROJECT_ROOT / "docker" / "vllm-emmy-serve" / "model_slug.sh"


def model_slug(model: str) -> str:
    """The image-naming slug for an HF model id.

    Shells out to `model_slug.sh` rather than reimplementing it: the slug decides both the
    image repo and which `models/<slug>.env` the warm and the bake read, so two
    implementations that disagree would let those two steps load different configs — the
    cache-key parity failure the release contract is built to prevent.
    """
    if not _SLUG_SCRIPT.is_file():  # a source checkout always has it; a stripped one won't
        return re.sub(r"[^a-z0-9._-]+", "-", model.rsplit("/", 1)[-1].lower()).strip("._-")
    out = subprocess.run([str(_SLUG_SCRIPT), model], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def covers(golden_model: str | None, target_slug: str) -> bool:
    """Does a golden recorded against ``golden_model`` cover a release of ``target_slug``?

    Exact slug match, or the golden's slug being a `-`-boundary prefix of the target's (the
    base-checkpoint rule above). Untagged goldens (``model: None``) never count — a shape
    with no recorded provenance may belong to any model, and guessing here would report
    coverage that does not exist.
    """
    if not golden_model:
        return False
    g = model_slug(golden_model)
    return g == target_slug or target_slug.startswith(f"{g}-")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF model id being released")
    ap.add_argument("--gpu", default=None, help="GPU name to check (default: the live card)")
    args = ap.parse_args()

    from emmy.compiler.pipeline.search.golden import GOLDEN_RECORDS, live_recorded_goldens

    target = model_slug(args.model)

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

    matched = [g for g in on_card if covers(g.model, target)]
    if not matched:
        others = Counter(model_slug(g.model) for g in on_card if g.model)
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
    print(f"OK: {len(matched)} golden(s) on {card} cover {args.model!r} (slug {target!r}).")
    print(f"  kinds: {', '.join(f'{k}={n}' for k, n in sorted(kinds.items()))}")
    provenance = sorted({g.model for g in matched if g.model})
    print(f"  recorded against: {', '.join(provenance)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
