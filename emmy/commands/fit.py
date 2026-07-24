"""``emmy fit`` — fit an offline-prior artifact and cross-validate it, writing a per-run
metrics file.

The Phase-4 fitter entry point: one pipeline, two orthogonal switches — ``--trainer``
(model class) × ``--data`` (training data). Only the ``linear`` × ``golden`` cell exists
today (the incumbent trainer on the golden dataset); the other cells arrive with the
measurement-freeze training work and until then are rejected loudly.

A run writes ``<out>/metrics.json`` — the deterministic, diff-able record two fits are
compared by (same header inputs → identical content; the run dir name, not the file,
carries the timestamp) — and ``<out>/weights.json``, the full-train artifact in the
shipped ``offline_weights.json`` format. The metrics layout (``full_train`` +
``cv.<axis>`` holdout/train/gap blocks) is documented on
:mod:`emmy.compiler.pipeline.search.prior.fit.cv`, which owns all the fold machinery;
this module owns what ``pipeline/`` must not import: the snippet-tracing golden case
builder (:func:`build_cases`, shared with ``scripts/golden_knob_heuristics.py``) and
the CLI/IO glue.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path

import numpy as np

from emmy import config, storage
from emmy.compiler.context import Context
from emmy.compiler.pipeline.search import features
from emmy.compiler.pipeline.search.golden import (
    GOLDEN_CONFIGS,
    MatmulGoldenConfig,
    PointwiseGoldenConfig,
    ReduceGoldenConfig,
)
from emmy.compiler.pipeline.search.golden_eval import _enumerate, enumerate_graph
from emmy.compiler.pipeline.search.prior.fit import build_artifact, fit_two_stage, topk_table
from emmy.compiler.pipeline.search.prior.fit import cv as fit_cv
from emmy.compiler.pipeline.search.prior.fit import linear as fit_linear

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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--folds", choices=(*FOLD_AXES, "both", "none"), default="both", help="Cross-validation fold axes (default: both).")
    parser.add_argument("--out", default=None, help="Run dir (default: _tune/fits/<timestamp>-<trainer>-<data>/).")
    parser.set_defaults(func=handle_fit)


def _base(ctx: Context, M: int, N: int, K: int, *, dynamic: bool = False) -> dict:
    """The shape / regime features the planner stamps and the prior featurizes —
    merged into each candidate row before ``knob_features`` so the occupancy terms
    (``#CTAs``, waves) and tier-aware BK targets fire. A dynamic (symbolic-M)
    golden mirrors the 992 stamp: the symbolic axis is EXCLUDED from the free-dim
    product and counted in ``S_ext_n_symbolic_axis`` — the same features the
    deployed featurization computes, and the flag the ``OfflinePrior`` selects
    its dynamic weight set on."""
    free = float(N) if dynamic else float(M * N)
    base = {**ctx.features(), "S_ext_free_prod": free, "S_ext_reduce_prod": float(K), "S_ext_reduce_max": float(K)}
    if dynamic:
        base["S_ext_n_symbolic_axis"] = 1.0
    return base


def _snippet_rows(snippet: str, ctx: Context) -> list[dict]:
    """A non-matmul golden's candidate rows — trace the golden's torch snippet
    (``torch.sum`` / ``torch.relu`` have no dedicated frontend op, so the graph comes from the real
    trace) and capture the RESTORED schedule fork's leaf rows via the same live-fork capture the
    matmul path uses (``golden_eval.enumerate_graph``). A reduce offers its ``REDUCE@<axis>``
    partitions; a pointwise kernel forks nothing today, so its pool is empty and the golden is
    reported un-enumerable (an honest read of the live space, not a reconstruction of the
    demolished one)."""
    from emmy.commands.trace import graph_from_code  # noqa: PLC0415

    graph = graph_from_code(snippet)[0]
    return enumerate_graph(graph, ctx)


def build_cases() -> tuple[list[fit_cv.GoldenCase], list[tuple[str, str, str]]]:
    """Reconstruct each golden's candidate enumeration, pin the golden's index, and
    featurize every row, as :class:`fit_cv.GoldenCase` records (name, tier, card, golden
    index, per-row ``D_*`` (+ ``MMA_tier``) feature dicts; ``key`` is ``"<gpu>/<name>"``,
    parity duplicates suffixed ``#2``, ``#3``, … in dataset order). The second return is
    the goldens that did NOT become cases, as ``(gpu, name, reason)`` — enumeration
    failures plus the kinds this fitter has no case builder for
    (:data:`fit_cv.OUT_OF_SCOPE`) — so metrics can count every recorded golden.

    Matmul goldens enumerate via ``golden_eval._enumerate`` — the SAME gate-narrowed
    pool ``eval offline`` and the greedy deploy rank over (fp32 → thread tier,
    fp16/bf16 → warp tier; the block-DAG rework moved the scalar↔warp choice to a
    structural fork, so a real fp16 matmul ranks within the warp tier alone, no
    scalar rows in the pool). A dynamic (``.dynM``) golden enumerates its hint-sized
    static twin's pool and featurizes with the symbolic-axis stamp (its own weight
    set). Reduce / pointwise goldens trace their snippet and capture the restored
    schedule fork's rows (``_snippet_rows``); a regime the live tree doesn't fork
    (pointwise) reports un-enumerable rather than reconstructing the demolished space.

    Each golden is reconstructed under its OWN card's context
    (``Context.from_target(cap, gpu_name=…)``, mirroring ``Sample.from_golden``):
    the multi-GPU golden set spans cards that differ in compute capability AND in
    SM count at the same cap (RTX 5090 = 170 vs RTX PRO 6000 = 188 vs RTX 4090 =
    128 SMs, the latter at sm_89), so both the candidate enumeration (cp.async /
    TMA tiers gate on cap) and the ``H_*`` / ``D_*`` occupancy features must use the
    recording card's regime — not one global cap — for the rank objective to match
    the deployed per-card featurization."""
    cases: list[fit_cv.GoldenCase] = []
    skipped: list[tuple[str, str, str]] = []
    key_counts: dict[str, int] = {}
    for g in GOLDEN_CONFIGS:
        ctx = Context.from_target(tuple(g.compute_cap), gpu_name=g.gpu_name)
        if isinstance(g, ReduceGoldenConfig):
            # The reduce's free axis (the ``M`` rows) maps to the planner's N axis
            # (the tuned ``FN`` register tile sweeps it): trace E_M=1, E_N=M, E_K=K.
            base = _base(ctx, 1, g.M, g.K)
            rows, tier = _snippet_rows(g.snippet(), ctx), "reduce"
        elif isinstance(g, PointwiseGoldenConfig):
            base = _base(ctx, g.M, g.N, 1)
            rows, tier = _snippet_rows(g.snippet(), ctx), "pointwise"
        elif isinstance(g, MatmulGoldenConfig):
            dyn = bool(g.dynamic)
            base = _base(ctx, g.M, g.N, g.K, dynamic=dyn)
            # A fast-math golden's row only exists in the gate-on enumeration (the same
            # self-gating seam as ``golden_eval.evaluate_golden``) — pin the gate for the
            # reconstruction so the f16-accumulate rows are present and the fit anchors
            # their ``MMA_acc_bits`` discriminator.
            if g.fast_math:
                from emmy.compiler.pipeline.search.space import F16_MMA_F32_ACC  # noqa: PLC0415

                with F16_MMA_F32_ACC.pinned("1"):
                    rows, _ = _enumerate(g.M, g.N, g.K, g.dtype, ctx)
            else:
                rows, _ = _enumerate(g.M, g.N, g.K, g.dtype, ctx)
            tier = "dyn" if dyn else ("thread" if g.dtype == "fp32" else "warp")
        else:
            skipped.append((g.gpu_name, g.name, fit_cv.OUT_OF_SCOPE))
            continue
        if not rows:
            logger.info("  !! %s: nothing enumerated — skipping", g.name)
            skipped.append((g.gpu_name, g.name, "nothing enumerated"))
            continue
        # Match the legacy-recorded golden against the native candidate rows by
        # schema-agnostic structural signature (free-axis slots + reduce decomp +
        # atom) — the candidates speak native ``MOVE@element``, the golden YAML legacy
        # GEMM-letters, so a per-key tuple compare never lines up.
        want = features.tile_signature(g.knobs)
        gidx = next((i for i, r in enumerate(rows) if features.tile_signature(r) == want), None)
        if gidx is None:
            logger.info("  !! %s: golden not in %d candidates — skipping", g.name, len(rows))
            skipped.append((g.gpu_name, g.name, f"golden not in {len(rows)} candidates"))
            continue
        # Keep ``D_*`` geometry/occupancy plus ``MMA_tier`` (the warp/scalar tier
        # discriminator, where the featurization still emits it) — the ``S_*`` /
        # ``H_*`` shape/regime features are constant within a shape, so they drop out
        # of a within-shape ranking.
        feats = [{k: v for k, v in features.knob_features({**base, **r}).items() if k.startswith("D_") or k == "MMA_tier"} for r in rows]
        key = f"{g.gpu_name}/{g.name}"
        key_counts[key] = key_counts.get(key, 0) + 1
        if key_counts[key] > 1:
            key = f"{key}#{key_counts[key]}"
        cases.append(fit_cv.GoldenCase(key, g.name, tier, g.gpu_name, gidx, feats))
    return cases, skipped


def _repo_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True, timeout=10)
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 — a fit outside a git checkout still gets a metrics file
        return "unknown"


def handle_fit(args) -> None:
    from emmy.compiler.pipeline.search.prior.offline import _DEFAULT_FILE  # noqa: PLC0415

    if args.trainer != "linear" or args.data != "golden":
        raise SystemExit(
            f"--trainer {args.trainer} x --data {args.data} is not yet supported — only 'linear' x 'golden' exists "
            "(the freeze/catboost cells land with the training-data work)"
        )
    axes = list(FOLD_AXES) if args.folds == "both" else [] if args.folds == "none" else [args.folds]

    out_dir = Path(args.out) if args.out else Path("_tune/fits") / f"{time.strftime('%Y%m%d-%H%M%S')}-{args.trainer}-{args.data}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building golden dataset (each golden under its own card's context) ...")
    cases, skipped = build_cases()
    names = sorted({k for c in cases for f in c.feats for k in f})
    n_dyn = sum(1 for c in cases if c.tier == "dyn")
    logger.info("  %d static + %d dynamic golden cases, %d D_* features, %d skipped", len(cases) - n_dyn, n_dyn, len(names), len(skipped))

    # Full-train: the incumbent process exactly — seeded from the shipped artifact's
    # weights (lenient read: a refit after a featurizer change is exactly when versions
    # mismatch, and a stale key simply seeds 0.0), scoring params carried through.
    incumbent = storage.read_json(config.offline_path() or _DEFAULT_FILE)
    if not isinstance(incumbent, dict) or "params" not in incumbent:
        raise SystemExit(f"no incumbent weights artifact to seed from at {config.offline_path() or _DEFAULT_FILE}")
    full = fit_two_stage(
        [c.fit_case for c in cases],
        names,
        seed_weights=incumbent.get("weights", {}),
        rng=np.random.default_rng(args.seed),
        samples=args.samples,
    )
    dyn_raw = full.dyn_raw if full.dyn_raw is not None else incumbent.get("weights_dynamic", full.static_raw)
    dyn_note = f"dynamic {topk_table(full.dyn_ranks)}" if full.dyn_ranks is not None else "carried from incumbent (no dynamic cases)"

    # CV folds re-fit ~2x per axis per fold — silence the fit core's per-case rank
    # logging for that stage (the full-train fit above already reported it).
    fit_logger = logging.getLogger(fit_linear.__name__)
    level = fit_logger.level
    fit_logger.setLevel(logging.WARNING)
    try:
        cv = {axis: fit_cv.run_axis(cases, names, axis, samples=args.samples, seed=args.seed) for axis in axes}
    finally:
        fit_logger.setLevel(level)

    header = {
        "trainer": args.trainer,
        "data": args.data,
        "seed": args.seed,
        "feat_ver": features.FEATURIZER_VERSION,
        "fold_axes": axes,
        "repo_commit": _repo_commit(),
        "trainer_params": {"samples": args.samples, "full_train_seed_weights": "incumbent", "fold_seed_weights": "zeros"},
    }
    shipped = fit_linear.TwoStageFit(full.static_raw, full.static_ranks, dyn_raw, full.dyn_ranks)
    metrics = fit_cv.build_metrics(header, cases, skipped, shipped, cv)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    import datetime  # noqa: PLC0415

    artifact = build_artifact(
        weights=full.static_raw,
        weights_dynamic=dyn_raw,
        params=incumbent["params"],
        provenance={
            "fitted": datetime.date.today().isoformat(),
            "script": "emmy fit",
            "args": {"samples": args.samples, "seed": args.seed},
            "cases": {"static": len(cases) - n_dyn, "dynamic": n_dyn},
            "notes": f"static {topk_table(full.static_ranks)}; {dyn_note}",
        },
    )
    storage.write_json(out_dir / "weights.json", artifact, indent=2)

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
    for axis, block in cv.items():
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
