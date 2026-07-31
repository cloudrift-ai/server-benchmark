#!/usr/bin/env python
"""Offline fit of the :class:`OfflinePrior` weights artifact over ``knob_features``.

Motivation
----------
The two-level autotuner explores the post-fusion kernel's knob space with an inner
MCTS that stops on **patience** (N consecutive measured terminals with no new
best). Cold (no online data) that search is ranked by the :class:`OfflinePrior` —
a fixed linear model over the engineered ``D_*`` geometry / occupancy features
:func:`features.knob_features` produces. If the golden config sits at rank 800 of 2400,
patience never reaches it. This script fits the linear weights so the golden lands
near the top.

It treats the problem as offline learning-to-rank, over the SINGLE featurization
(``features.knob_features`` — no parallel feature set), covering **every kernel
regime**: fp32-scalar + fp16/bf16-warp matmul, cooperative reduce, and pointwise.

This script is the legacy CLI wrapper around shared pieces: golden case building
lives in ``emmy/commands/fit.py`` (``build_cases`` — the command layer owns the
snippet tracer ``pipeline/`` must not import, and ``emmy fit`` is the harness that
also cross-validates the fit), and the fit / rank-eval / artifact-assembly core in
``emmy/compiler/pipeline/search/prior/fit/``. What this wrapper does:

  1. Build the golden cases (``build_golden_groups`` — each golden's candidate
     enumeration reconstructed under its own card's context, the golden's row pinned).
  2. Run the incumbent two-stage fit (``fit_two_stage``: static tiers seeded from
     the incumbent artifact, then the dynamic set seeded from the static result).
  3. Write the winning weights (both sets + the carried-over scoring params +
     provenance) to the ``OfflinePrior`` weights artifact (``--out``, default the
     repo-checked ``offline_weights.json``), and print them.

Prefer ``emmy fit`` for anything beyond regenerating the artifact — it runs the same
trainer AND writes the cross-validated metrics file.

Run:  ./venv/bin/python scripts/golden_knob_heuristics.py
      ./venv/bin/python scripts/golden_knob_heuristics.py --samples 40000 --out /tmp/candidate.json
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import numpy as np

from emmy import config, storage
from emmy.commands.fit import build_golden_groups
from emmy.compiler.pipeline.search.prior.fit import build_artifact, fit_two_stage, topk_table
from emmy.logging_setup import setup_cli_logging


def _print_weights(var: str, raw_w: dict[str, float]) -> None:
    print(f"\n== {var} ==")
    for name, wv in sorted(raw_w.items(), key=lambda t: -abs(t[1])):
        print(f"    {name!r}: {wv},")


def main() -> None:
    from emmy.compiler.pipeline.search.prior.offline import _DEFAULT_FILE  # noqa: PLC0415

    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=20000, help="random weight vectors to try")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out",
        default=str(_DEFAULT_FILE),
        help="Weights artifact to write (default: the repo-checked offline_weights.json).",
    )
    args = ap.parse_args()

    setup_cli_logging()  # the fit core logs its progress; render it print()-identical

    print("Building golden dataset (each golden under its own card's context) ...")
    cases, _skipped = build_golden_groups()
    names = sorted({n for c in cases for n in c.feat_names})
    n_dyn = sum(1 for c in cases if c.tier == "dyn")
    print(f"  {len(cases) - n_dyn} static + {n_dyn} dynamic golden cases, {len(names)} D_* features")

    # Seed: the incumbent artifact's static weights so each search starts from
    # today's ranking and can only improve on it (the dyn fit seeds from the STATIC
    # result — the masked tier shares most of the geometry priors and diverges where
    # the boundary guard / occupancy differences demand). Lenient read, no feat_ver
    # gate: a refit after a featurizer change is exactly when versions mismatch, and
    # a stale key simply seeds 0.0. The scoring params carry through unchanged —
    # this script fits only the linear weights.
    incumbent = storage.read_json(config.offline_path() or _DEFAULT_FILE)
    if not isinstance(incumbent, dict) or "params" not in incumbent:
        raise SystemExit(f"no incumbent weights artifact to seed from at {config.offline_path() or _DEFAULT_FILE}")

    res = fit_two_stage(cases, names, seed_weights=incumbent.get("weights", {}), rng=np.random.default_rng(args.seed), samples=args.samples)
    _print_weights("weights (static)", res.static_raw)
    dyn_raw, dyn_note = incumbent.get("weights_dynamic", res.static_raw), "carried from incumbent (no dynamic cases)"
    if res.dyn_ranks is not None:
        dyn_raw, dyn_note = res.dyn_raw, f"dynamic {topk_table(res.dyn_ranks)}"
        _print_weights("weights_dynamic", dyn_raw)

    artifact = build_artifact(
        weights=res.static_raw,
        weights_dynamic=dyn_raw,
        params=incumbent["params"],
        provenance={
            "fitted": datetime.date.today().isoformat(),
            "script": "scripts/golden_knob_heuristics.py",
            "args": {"samples": args.samples, "seed": args.seed},
            "cases": {"static": len(cases) - n_dyn, "dynamic": n_dyn},
            "notes": f"static {topk_table(res.static_ranks)}; {dyn_note}",
        },
    )
    storage.write_json(Path(args.out), artifact, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
