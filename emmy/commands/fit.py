"""``emmy fit`` — fit an offline-prior artifact and cross-validate it, writing a per-run
metrics file.

The fitter entry point: one pipeline, two orthogonal switches — ``--trainer``
(model class) × ``--data`` (training data). Only the ``linear`` × ``golden`` combination exists
today (the incumbent trainer on the golden dataset); the other combinations arrive with the
measurement-freeze training work and until then are rejected loudly.

A run writes ``<out>/metrics.json`` — the deterministic, diff-able record two fits are
compared by (same header inputs → identical content; the run dir name, not the file,
carries the timestamp) — and ``<out>/weights.json``, the full-train artifact in the
shipped ``offline_weights.json`` format. The metrics layout (``full_train`` +
``cv.<axis>`` holdout/train/gap blocks) is documented on
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
from pathlib import Path

from emmy import config, storage
from emmy.compiler.context import Context
from emmy.compiler.pipeline.search import features
from emmy.compiler.pipeline.search.golden import GOLDEN_RECORDS
from emmy.compiler.pipeline.search.golden_eval import enumerate_graph
from emmy.compiler.pipeline.search.prior.fit import Group, fit_two_stage, topk_table
from emmy.compiler.pipeline.search.prior.fit import linear as fit_linear
from emmy.compiler.pipeline.search.prior.fit.group import DEFAULT_FEATURES, feature_view
from emmy.compiler.pipeline.search.prior.fit.run import run_fit
from emmy.compiler.pipeline.search.prior.linear_model import ROUTING_FEATURES

logger = logging.getLogger(__name__)

FOLD_AXES = ("op_family", "gpu")


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
        help="Random weight vectors before coordinate descent (default 0: descent-from-seed, the incumbent practice).",
    )
    parser.add_argument(
        "--l2",
        type=float,
        default=fit_linear.DEFAULT_L2,
        help="Raw-space L2 penalty strength in the fit loss (default: the declared tie-breaker strength; 0 disables).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--folds", choices=(*FOLD_AXES, "both", "none"), default="both", help="Cross-validation fold axes (default: both).")
    parser.add_argument(
        "--features",
        default=DEFAULT_FEATURES,
        help="Feature view: comma-separated names, trailing '*' = prefix glob (recorded in metrics + provenance).",
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
        # ROUTING_FEATURES survive any view: they pick the weight set rather than contribute a
        # term, and ``Group.from_dicts`` lifts them straight out of the matrix. Keeping them
        # unconditionally means a narrower ``--features`` spec cannot silently misroute a
        # symbolic-axis pool, and the recorded spec stays comparable with earlier fits.
        feats = [{k: v for k, v in features.knob_features({**base, **r}).items() if keep(k) or k in ROUTING_FEATURES} for r in rows]
        key = f"{g.gpu_name}/{g.name}"
        key_counts[key] = key_counts.get(key, 0) + 1
        if key_counts[key] > 1:
            key = f"{key}#{key_counts[key]}"
        cases.append(Group.from_dicts(key, g.name, tier, g.gpu_name, gidx, feats))
    return cases, skipped


def _repo_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True, timeout=10)
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 — a fit outside a git checkout still gets a metrics file
        return "unknown"


def handle_fit(args) -> None:
    from emmy.compiler.pipeline.search.prior.offline import _DEFAULT_FILE, _PARAM_KEYS  # noqa: PLC0415

    if args.trainer != "linear" or args.data != "golden":
        raise SystemExit(
            f"--trainer {args.trainer} x --data {args.data} is not yet supported — only 'linear' x 'golden' exists "
            "(the freeze/catboost cells land with the training-data work)"
        )
    axes = list(FOLD_AXES) if args.folds == "both" else [] if args.folds == "none" else [args.folds]

    out_dir = Path(args.out) if args.out else Path("_tune/fits") / f"{time.strftime('%Y%m%d-%H%M%S')}-{args.trainer}-{args.data}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building golden dataset (each golden under its own card's context) ...")
    cases, skipped = build_golden_groups(args.features)
    names = sorted({n for c in cases for n in c.feat_names})
    n_dyn = sum(1 for c in cases if c.dynamic)
    logger.info("  %d static + %d dynamic golden cases, %d D_* features, %d skipped", len(cases) - n_dyn, n_dyn, len(names), len(skipped))

    incumbent = storage.read_json(config.offline_path() or _DEFAULT_FILE)
    if not isinstance(incumbent, dict) or "params" not in incumbent:
        raise SystemExit(f"no incumbent weights artifact to seed from at {config.offline_path() or _DEFAULT_FILE}")
    # The fitted scalar params seed from the incumbent; the rest of its params block (``scale``,
    # rank-neutral) carries through to the artifact unchanged. A pre-2026-08-05 artifact whose block
    # still lists the retired gate weights simply loses them here — they are linear terms now.
    seed_params = {n: float(incumbent["params"].get(n, 0.0)) for n in fit_linear.PARAM_NAMES}
    carried = {k: v for k, v in incumbent["params"].items() if k not in fit_linear.PARAM_NAMES and k in _PARAM_KEYS}

    # Full-train: the incumbent process exactly — seeded from the shipped artifact's
    # weights (lenient read: a refit after a featurizer change is exactly when versions
    # mismatch, and a stale key simply seeds 0.0). A fit with no dynamic cases carries
    # the incumbent's dynamic set forward into the shippable model.
    def full_train_fit(groups, rng):
        full = fit_two_stage(
            groups, names, seed_weights=incumbent.get("weights", {}), seed_params=seed_params, rng=rng, samples=args.samples, l2=args.l2
        )
        dyn_raw = full.dyn_raw if full.dyn_raw is not None else incumbent.get("weights_dynamic", full.static_raw)
        dyn_note = f"dynamic {topk_table(full.dyn_ranks)}" if full.dyn_ranks is not None else "carried from incumbent (no dynamic cases)"
        shipped = fit_linear.TwoStageFit(full.static_raw, full.static_ranks, dyn_raw, full.dyn_ranks, full.params)
        params_note = ", ".join(f"{k}={v:g}" for k, v in sorted(full.params.items()))
        return shipped, f"static {topk_table(full.static_ranks)}; {dyn_note}; params {params_note}"

    # Fold-seeding policy lives here (recorded in the header): fold models seed from
    # ZEROS — the incumbent's weights were fit on every golden, so seeding folds from
    # them would leak each held-out golden into its own holdout model. The scoring params
    # seed from the incumbent in both cases: they are two scalars the fold fits re-derive,
    # not a per-golden memory.
    def fit_model(groups, rng):
        return fit_two_stage(groups, names, seed_weights={}, seed_params=seed_params, rng=rng, samples=args.samples, l2=args.l2)

    header = {
        "trainer": args.trainer,
        "data": args.data,
        "seed": args.seed,
        "feat_ver": features.FEATURIZER_VERSION,
        "features": args.features,
        "fold_axes": axes,
        "repo_commit": _repo_commit(),
        "trainer_params": {"samples": args.samples, "l2": args.l2, "full_train_seed_weights": "incumbent", "fold_seed_weights": "zeros"},
    }
    import datetime  # noqa: PLC0415

    metrics, artifact = run_fit(
        cases,
        skipped,
        full_train_fit=full_train_fit,
        fit_model=fit_model,
        axes=axes,
        seed=args.seed,
        header=header,
        params=carried,
        provenance={
            "fitted": datetime.date.today().isoformat(),
            "script": "emmy fit",
            "args": {"samples": args.samples, "l2": args.l2, "seed": args.seed},
            "features": args.features,
        },
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
