"""``emmy fit`` — fit an offline-prior artifact and cross-validate it, writing a per-run
metrics file.

The fitter entry point: one pipeline, two orthogonal switches — ``--trainer``
(model class: the incumbent ``linear`` weights or a ``catboost`` ranker) × ``--data`` (training data).
Only ``--data golden`` exists today; the freeze cells arrive with the measurement-freeze training work
and until then are rejected loudly. Both trainers write the same artifact shape, distinguished by its
``kind`` field, so either can be pointed at with ``EMMY_OFFLINE_FILE`` and A/B'd against the other.

A run writes ``<out>/metrics.json`` — the deterministic, diff-able record two fits are
compared by (same header inputs → identical content; the run dir name, not the file,
carries the timestamp) — and ``<out>/weights.json``, the full-train artifact in the
shipped ``offline_weights.json`` format. The metrics layout (``full_train`` +
a ``cv.shape`` holdout/train/gap block) is documented on
:mod:`emmy.compiler.pipeline.search.prior.fit.cv`, which owns all the fold machinery;
the run itself is :func:`~emmy.compiler.pipeline.search.prior.fit.run.run_fit`. This
module owns what ``pipeline/`` must not import: the snippet-tracing golden case builder
(:func:`build_golden_groups`) plus the CLI, the trainer wiring, and the file writing.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import replace
from pathlib import Path

from emmy import config, storage
from emmy.compiler.context import Context
from emmy.compiler.pipeline.search import features
from emmy.compiler.pipeline.search.golden import GOLDEN_RECORDS
from emmy.compiler.pipeline.search.golden_eval import enumerate_graph
from emmy.compiler.pipeline.search.prior.fit import Group
from emmy.compiler.pipeline.search.prior.fit import catboost as fit_catboost
from emmy.compiler.pipeline.search.prior.fit import cv as fit_cv
from emmy.compiler.pipeline.search.prior.fit import linear as fit_linear
from emmy.compiler.pipeline.search.prior.fit.group import DEFAULT_FEATURES, TREE_FEATURES, feature_view
from emmy.compiler.pipeline.search.prior.fit.run import run_fit
from emmy.compiler.pipeline.search.prior.linear_model import LinearModel

logger = logging.getLogger(__name__)


def register_fit_command(subparsers) -> None:
    parser = subparsers.add_parser(
        "fit",
        help="Fit the offline prior and cross-validate it (linear trainer x golden dataset), writing a metrics file",
    )
    parser.add_argument("--trainer", choices=("linear", "catboost"), default="linear")
    parser.add_argument("--data", default="golden", help="Training data: 'golden' or 'freeze:<path>' (freeze not yet supported).")
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="linear only: random weight vectors before coordinate descent (default 0: descent-from-seed, the incumbent practice).",
    )
    parser.add_argument(
        "--l2",
        type=float,
        default=fit_linear.DEFAULT_L2,
        help="linear only: raw-space L2 penalty strength in the fit loss (default: the declared tie-breaker strength; 0 disables).",
    )
    parser.add_argument("--iterations", type=int, default=500, help="catboost only: boosting iterations.")
    parser.add_argument(
        "--negatives",
        type=int,
        default=fit_catboost.DEFAULT_NEGATIVES,
        help="catboost only: sampled negatives per pool per round (the golden is the group's single positive).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=fit_catboost.DEFAULT_ROUNDS,
        help="catboost only: fit rounds — the first draws negatives uniformly, each further one mines hard negatives.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--folds",
        type=int,
        default=fit_cv.DEFAULT_FOLDS,
        help="Cross-validation folds, grouped by shape so goldens sharing a candidate pool are held out together "
        f"(default {fit_cv.DEFAULT_FOLDS}; 0 skips cross-validation).",
    )
    parser.add_argument(
        "--features",
        default=None,
        help="Feature view: comma-separated names, trailing '*' = prefix glob, leading '-' excludes (recorded in "
        "metrics + provenance). Default: the trainer's own view — the full D_* set for 'linear', and for "
        "'catboost' that set minus the features a tree re-derives from the columns it keeps.",
    )
    parser.add_argument(
        "--artifact",
        nargs="?",
        const="",
        default=None,
        help="Also write the fitted weights artifact to this path (no value: the repo-checked offline_weights.json).",
    )
    parser.add_argument("--out", default=None, help="Run dir (default: _tune/fits/<timestamp>-<trainer>-<data>/).")
    parser.set_defaults(func=handle_fit)


def _shape_group(g) -> str:
    """The golden's cross-validation fold group: its ``ShapeKey`` with the two fields normalized away that
    separate goldens sharing a candidate pool.

    ``is_dyn`` because a ``.dynM`` golden *enumerates its static counterpart's pool* (see
    :func:`build_golden_groups`) — split them across folds and the fold model is scored on rows it trained on.
    ``is_warp`` because the fp16/fp32 twins of one geometry are the same physical shape; their pools are
    disjoint, so this one is conservatism rather than necessity, and it costs 21 groups out of 435.

    ``kind`` and the extents stay: ``flash`` / ``softmax`` / ``rms_norm`` / ``fused`` are different kernels, not
    variants of one shape. The result is a string so it lands in the metrics file as-is."""
    return str(replace(g.shape_key, is_dyn=False, is_warp=False))


def build_golden_groups(features_spec: str = DEFAULT_FEATURES) -> tuple[list[Group], list[tuple[str, str, str]]]:
    """Enumerate each embedded golden program, pin its recorded row, and
    featurize every row, as :class:`Group` records (name, tier, card, golden index,
    per-row features filtered through the ``features_spec`` view; ``key`` is ``"<gpu>/<name>"``,
    parity duplicates suffixed ``#2``, ``#3``, … in dataset order). The second return is
    the goldens that did NOT become cases, as ``(gpu, name, reason)`` — enumeration
    failures plus the kinds this fitter has no case builder for
    (:data:`fit_cv.OUT_OF_SCOPE`) — so metrics can count every recorded golden.

    Matmul goldens enumerate via ``golden_eval._enumerate`` — the SAME gate-narrowed
    pool ``eval offline`` and the greedy deploy rank over (fp32 → thread tier,
    fp16/bf16 → warp tier; the block-DAG rework moved the scalar↔warp choice to a
    structural fork, so a real fp16 matmul ranks within the warp tier alone, no
    scalar rows in the pool). A dynamic (``.dynM``) golden enumerates the pool of its
    static counterpart at the hint size and featurizes with the symbolic-axis stamp
    (its own weight set). Reduce / pointwise goldens trace their snippet and capture the restored
    schedule fork's rows (``_snippet_rows``); a regime the live tree doesn't fork
    (pointwise) reports un-enumerable rather than reconstructing a search space that no longer exists.

    Each golden is reconstructed under its OWN card's context
    (``Context.from_target(cap, gpu_name=…)``, mirroring ``Sample.from_golden``):
    the multi-GPU golden set spans cards that differ in compute capability AND in
    SM count at the same cap (RTX 5090 = 170 vs RTX PRO 6000 = 188 vs RTX 4090 =
    128 SMs, the latter at sm_89), so both the candidate enumeration (cp.async /
    TMA tiers gate on cap) and the ``H_*`` / ``D_*`` occupancy features must use the
    recording card's regime — not one global cap — for the rank objective to match
    the deployed per-card featurization."""
    keep = feature_view(features_spec)
    cases: list[Group] = []
    skipped: list[tuple[str, str, str]] = []
    key_counts: dict[str, int] = {}
    # ONE Context per card: the per-card facts are identical across its goldens, and sharing the
    # instance shares its schedule pool cache — the std / parity / fm siblings of one shape
    # re-enumerate byte-identical pools (490 matmul goldens collapse to ~313 distinct), so the
    # dataset build pays each pool once. The fm-pinned enumeration keys apart on its own: the
    # precision gate rides the pool key's pin fingerprint.
    ctxs: dict[tuple, Context] = {}
    for g in GOLDEN_RECORDS:
        card = (tuple(g.compute_cap), g.gpu_name)
        ctx = ctxs.get(card)
        if ctx is None:
            ctx = ctxs[card] = Context.from_target(tuple(g.compute_cap), gpu_name=g.gpu_name)
        base = {**ctx.features(), **g.structural_features}
        from emmy.compiler.pipeline.search.pins import pinned_knobs  # noqa: PLC0415

        with pinned_knobs(g.pin_map):
            rows = enumerate_graph(g.target_program.copy(), ctx)
        tier = "dyn" if g.dynamic else (g.shape_key.kind or ("warp" if g.shape_key.is_warp else "thread"))
        if not rows:
            logger.info("  !! %s: nothing enumerated — skipping", g.name)
            skipped.append((g.gpu_name, g.name, "nothing enumerated"))
            continue
        # Match the legacy-recorded golden against the native candidate rows by
        # schema-agnostic structural signature (free-axis slots + reduce decomp +
        # atom) — the candidate rows use the native ``MOVE@element`` keys while the
        # golden YAML records legacy GEMM-letter keys, so comparing key-value tuples
        # directly never matches.
        want = features.tile_signature(g.knobs)
        gidx = next((i for i, r in enumerate(rows) if features.tile_signature(r) == want), None)
        if gidx is None:
            logger.info("  !! %s: golden not in %d candidates — skipping", g.name, len(rows))
            skipped.append((g.gpu_name, g.name, f"golden not in {len(rows)} candidates"))
            continue
        # The feature view (default ``DEFAULT_FEATURES``: ``D_*`` geometry/occupancy plus
        # ``MMA_tier`` — see its rationale in ``prior/fit/group.py``) filters here, before
        # the pool is packed, so the trained-under view is exactly what the Group stores.
        # ``feature_view`` keeps the routing features whatever the spec says, so a narrower
        # ``--features`` cannot silently misroute a symbolic-axis pool.
        feats = [{k: v for k, v in features.knob_features({**base, **r}).items() if keep(k)} for r in rows]
        key = f"{g.gpu_name}/{g.name}"
        key_counts[key] = key_counts.get(key, 0) + 1
        if key_counts[key] > 1:
            key = f"{key}#{key_counts[key]}"
        cases.append(Group.from_dicts(key, g.name, tier, g.gpu_name, _shape_group(g), gidx, feats))
    return cases, skipped


def _repo_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True, timeout=10)
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 — a fit outside a git checkout still gets a metrics file
        return "unknown"


def _linear_trainers(args, names: list[str]):
    """The linear cell's trainer pair and the hyperparameters its metrics header records.

    Full-train seeds from the incumbent artifact's weights; fold models seed from ZEROS
    (``warm_start=False``) — the incumbent's weights were fit on every golden, so warm-starting a fold from
    them would leak each held-out golden into its own holdout model. The scalar params seed from the incumbent
    either way: two numbers a fold fit re-derives, not a per-golden memory."""
    from emmy.compiler.pipeline.search.prior.offline import _DEFAULT_FILE  # noqa: PLC0415

    raw = storage.read_json(config.offline_path() or _DEFAULT_FILE)
    if not isinstance(raw, dict) or "scale" not in (raw.get("params") or {}):
        raise SystemExit(
            f"no usable incumbent weights artifact to seed from at {config.offline_path() or _DEFAULT_FILE} "
            f"(needs a 'params' block carrying 'scale')"
        )
    # Lenient read (``LinearModel.from_artifact`` does not version-gate): a refit after a featurizer
    # change is exactly when versions mismatch, and a stale key simply seeds 0.0. A pre-2026-08-05
    # artifact whose params block still lists the retired gate weights simply loses them here — they
    # are linear terms now. ``scale`` rides along on the model, rank-neutral and never fitted.
    incumbent = LinearModel.from_artifact(raw)
    trainer = fit_linear.LinearTrainer(feature_names=tuple(names), init=incumbent, samples=args.samples, l2=args.l2, random_state=args.seed)
    fold_trainer = replace(trainer, warm_start=False)
    params = {
        "samples": args.samples,
        "l2": args.l2,
        "objective": getattr(trainer.objective, "__name__", repr(trainer.objective)),
        "full_train_seed_weights": "incumbent" if trainer.warm_start else "zeros",
        "fold_seed_weights": "incumbent" if fold_trainer.warm_start else "zeros",
    }
    return trainer, fold_trainer, params, incumbent


def _catboost_trainers(args, names: list[str]):
    """The tree cell's trainer and its recorded hyperparameters. ONE trainer serves both the shippable model and
    every fold: a tree ensemble has no warm start, so there is no seeding policy to differ on and no way for a
    fold model to inherit anything from the held-out golden."""
    trainer = fit_catboost.CatBoostTrainer(
        feature_names=tuple(names),
        iterations=args.iterations,
        negatives=args.negatives,
        rounds=args.rounds,
        random_state=args.seed,
    )
    params = {
        "iterations": args.iterations,
        "negatives": args.negatives,
        "rounds": args.rounds,
        "depth": trainer.depth,
        "learning_rate": trainer.learning_rate,
        "objective": "QuerySoftMax",
    }
    return trainer, trainer, params, None


# Each trainer's factory and its default feature view. The views differ because the model classes do: the
# linear one needs the engineered step / fold / interaction features, having no way to form them, and the
# tree re-derives every one of them from the columns :data:`TREE_FEATURES` keeps. ``--features`` overrides
# either, which is how the two views are compared on one model class.
TRAINERS = {
    "linear": (_linear_trainers, DEFAULT_FEATURES),
    "catboost": (_catboost_trainers, TREE_FEATURES),
}


def handle_fit(args) -> None:
    from emmy.compiler.pipeline.search.prior.offline import _DEFAULT_FILE  # noqa: PLC0415

    if args.data != "golden":
        raise SystemExit(
            f"--data {args.data} is not yet supported — only 'golden' exists (the freeze cells land with the training-data work)"
        )

    out_dir = Path(args.out) if args.out else Path("_tune/fits") / f"{time.strftime('%Y%m%d-%H%M%S')}-{args.trainer}-{args.data}"
    out_dir.mkdir(parents=True, exist_ok=True)

    make_trainers, default_view = TRAINERS[args.trainer]
    view = args.features or default_view

    logger.info("Building golden dataset (each golden under its own card's context) ...")
    cases, skipped = build_golden_groups(view)
    names = sorted({n for c in cases for n in c.feat_names})
    n_dyn = sum(1 for c in cases if c.dynamic)
    logger.info("  %d static + %d dynamic golden cases, %d D_* features, %d skipped", len(cases) - n_dyn, n_dyn, len(names), len(skipped))

    trainer, fold_trainer, trainer_params, incumbent = make_trainers(args, names)
    header = {
        "trainer": args.trainer,
        "data": args.data,
        "seed": args.seed,
        "feat_ver": features.FEATURIZER_VERSION,
        "features": view,
        "folds": args.folds,
        "repo_commit": _repo_commit(),
        "trainer_params": trainer_params,
    }
    import datetime  # noqa: PLC0415

    metrics, fit = run_fit(cases, skipped, trainer=trainer, fold_trainer=fold_trainer, folds=args.folds, header=header)

    model, notes = fit.model, fit.notes
    # Shipping policy, and the reason ``run_fit`` hands back a fit rather than an artifact: a LINEAR fit with no
    # dynamic cases would otherwise ship with no dynamic weight set at all, so carry the incumbent's forward —
    # loudly, in the provenance notes, never silently. The tree model has no second weight set to be missing.
    if isinstance(model, LinearModel) and model.weights_dynamic is None:
        # ``is not None``, not truthiness: an incumbent that legitimately pruned every dynamic
        # coordinate carries an EMPTY set, and that is still its answer, not a missing one.
        carried = incumbent.weights_dynamic if incumbent.weights_dynamic is not None else model.weights
        source = "incumbent" if incumbent.weights_dynamic is not None else "the static fit"
        model = replace(model, weights_dynamic=carried)
        notes = f"{notes}; dynamic set carried from {source}"
    artifact = model.to_artifact(
        provenance={
            "fitted": datetime.date.today().isoformat(),
            "script": "emmy fit",
            "args": {"trainer": args.trainer, "seed": args.seed, **trainer_params},
            "features": view,
            "cases": {"static": len(cases) - n_dyn, "dynamic": n_dyn},
            "notes": notes,
        }
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    storage.write_json(out_dir / "weights.json", artifact, indent=2)
    if args.artifact is not None:
        artifact_path = Path(args.artifact) if args.artifact else _DEFAULT_FILE
        storage.write_json(artifact_path, artifact, indent=2)
        logger.info("wrote %s", artifact_path)

    for gpu, card in metrics["full_train"]["per_card"].items():
        logger.info(
            "full_train  %-34s n=%-3d median=%s (optimistic %s) unranked=%d out_of_scope=%d",
            gpu,
            card["n"],
            card["median"],
            card["median_optimistic"],
            card["unranked"],
            card["out_of_scope"],
        )
    for axis, block in metrics["cv"].items():
        for gpu, card in block["holdout"]["per_card"].items():
            logger.info(
                "cv.%-9s %-34s holdout median=%s (optimistic %s) train=%s gap=%s",
                axis,
                gpu,
                card["median"],
                card["median_optimistic"],
                block["train"]["per_card"][gpu]["median"],
                block["gap"].get(gpu),
            )
        for f, why in block["fold_detail"]["excluded"].items():
            logger.info("cv.%-9s fold %s EXCLUDED: %s", axis, f, why)
    logger.info("wrote %s", out_dir / "metrics.json")
