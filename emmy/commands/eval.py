"""``emmy eval <knobs|prior|golden|variants|failures>`` — evaluate the tuning machinery.

Five subcommands:

- ``eval knobs``     — print the registered knob schema, then (with a tune DB)
  per-knob **regret** + a knob-interaction matrix (the analysis below).
- ``eval prior``     — how well the prior RANKS: one report over benched pools
  (``--dataset nodes``: Spearman + regret, what a wrong pick costs) or over the golden
  corpus (``--dataset golden``: the golden-rank screen, plus the greedy pipeline pick vs
  golden). BOTH prior halves are reported, labelled — they fail for different reasons.
  The summaries are assembled by ``search/prior/report.py`` and rendered here; ``emmy fit``
  writes the same summaries into its ``metrics.json``, so a fit and an eval state the golden
  screen with one implementation rather than two that agree by coincidence.
- ``eval golden``    — validate one canonical golden YAML against the pinned serving
  configuration and live GPU, then reproduce its rows and audit the exact serving matrix.
- ``eval variants``  — per-kernel leaderboard of the tune DB's measured variants
  (fastest first) and the config the prior deploys marked + ranked.
- ``eval failures``  — the tune DB's ``bench_fail`` rows clustered by
  ``(kernel, error)`` with the knob values shared by every failing row.

The ``eval knobs`` regret analysis: for each kernel (grouped by the kernel C
identifier extracted from ``cuda_op.pretty``), compute per-knob regret:

    regret[K] = max(best_us | K=v) / min(best_us | K=v)

where ``best_us | K=v`` is the minimum measured latency over variants
pinning ``K=v`` (marginalizing the other knobs by taking min). Aggregate
across kernels with median / p90 / geometric mean and print a sorted
table.

The intended use is to decide knob ordering for a hierarchical Fork tree
in the planner: high-regret knobs go at the root of the tree (commit
first), low-regret knobs go at the leaves. A second table shows
pairwise knob interaction so coupled knobs (where the optimal value of
K2 depends on K1) can be kept in the same Fork rather than split across
levels.

Grouping caveat: the analysis groups variants by kernel C identifier
only — different shapes of the same kernel collapse into one group.
Same-kernel-different-shape variants are comparable in *relative* knob
impact even when absolute latencies differ, so the rank order of knobs
is the load-bearing output here.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from emmy import storage
from emmy.commands.compile import resolve_tune_db
from emmy.commands.dataset_args import add_dataset_args, require_source, resolve_offline_arg, resolve_online_arg
from emmy.commands.table import GREEN as _GREEN
from emmy.commands.table import RED as _RED
from emmy.commands.table import YELLOW as _YELLOW
from emmy.commands.table import Col, col_widths, knob_columns, render_table
from emmy.compiler.pipeline.search.data import Dataset
from emmy.compiler.pipeline.search.pool import DEFAULT_SAMPLE
from emmy.compiler.pipeline.search.prior import report as report_mod

logger = logging.getLogger(__name__)


def _format_pins(pins) -> str:
    def spell(value) -> str:
        return str(value).lower() if isinstance(value, bool) else str(value)

    return ", ".join(f"{name}={spell(value)}" for name, value in pins) or "unpinned"


def _realization_label(name: str, pins) -> str:
    return name if dict(pins) == {"FAST_MATH": False} else f"{name} [{_format_pins(pins)}]"


def register_eval_command(subparsers) -> None:
    """``emmy eval <knobs|prior|golden|variants|failures>`` — evaluate the tuning knobs or
    the prior's ranking."""
    parser = subparsers.add_parser(
        "eval",
        help="Evaluate the tuning knobs / how well the prior ranks candidate pools",
    )
    sub = parser.add_subparsers(dest="eval_target", required=True)

    pk = sub.add_parser("knobs", help="Print the registered knob schema + (with a tune DB) per-knob regret + interactions")
    add_dataset_args(pk, default="db", with_min_variants=True)
    pk.set_defaults(func=handle_eval_knobs)

    pp = sub.add_parser(
        "prior",
        help="Report how well each prior half ranks — over benched pools (--dataset nodes) or the golden corpus",
    )
    pp.add_argument(
        "--online-file",
        "--prior",  # pre-rename spelling
        dest="online_file",
        help="Path to the online-prior JSON to load. Default: EMMY_ONLINE_FILE or ~/.cache/emmy/online.json. "
        "(`emmy tune` writes this file; it is NOT the tune DB.)",
    )
    pp.add_argument(
        "--offline-file",
        "--analytic-file",  # pre-rename spelling
        dest="offline_file",
        help="Offline weights artifact (JSON) to score the offline half with, for A/Bing candidate fits. "
        "Default: EMMY_OFFLINE_FILE or the repo-checked offline_weights.json.",
    )
    add_dataset_args(pp, default="golden")
    pp.add_argument(
        "--pool-sample",
        type=int,
        default=DEFAULT_SAMPLE,
        help=f"--dataset golden: candidates drawn per pool during enumeration (default {DEFAULT_SAMPLE}; 0 enumerates "
        "every row). Recorded in the report header — a rank is only comparable against a fit that drew the same way.",
    )
    pp.add_argument("--json", dest="json_out", metavar="PATH", help="Also write the report as JSON, for diffing two runs.")
    pp.add_argument(
        "--features",
        action="store_true",
        help="--dataset golden: also print the exact feature vector the prior regresses on per golden config (features.knob_features).",
    )
    pp.set_defaults(func=handle_eval_prior)

    pg = sub.add_parser(
        "golden",
        help="Validate one golden YAML against its pinned serving configuration and the live target GPU",
    )
    pg.add_argument("--golden", required=True, metavar="PATH", help="The exact canonical golden YAML to validate.")
    pg.add_argument(
        "--serving-config",
        required=True,
        metavar="PATH",
        help="Pinned release env that names the model, GPU, golden file, and reachable realization matrix.",
    )
    pg.set_defaults(func=handle_eval_golden)

    pv = sub.add_parser(
        "variants",
        help="Per-kernel leaderboard of the tune DB's measured variants, with the prior's deployed pick marked and ranked",
    )
    pv.add_argument(
        "--online-file",
        "--prior",  # pre-rename spelling
        dest="online_file",
        help="Online-prior JSON to load (default: EMMY_ONLINE_FILE or ~/.cache/emmy/online.json).",
    )
    add_dataset_args(pv, default="db")
    pv.add_argument(
        "--top",
        type=int,
        default=20,
        help="Variants shown per kernel, fastest first (0 = all; the pick row always shows). Default: 20.",
    )
    pv.set_defaults(func=handle_eval_variants)

    pf = sub.add_parser(
        "failures",
        help="Cluster the tune DB's bench_fail rows by kernel + error, with the knob values shared by every failing row",
    )
    add_dataset_args(pf, default="db")
    pf.set_defaults(func=handle_eval_failures)


def handle_eval_knobs(args) -> None:
    """``eval knobs`` — the registered knob schema, then (with a tune DB) per-knob
    regret + the knob-interaction matrix."""
    require_source(args, {"db"}, "eval knobs regret needs DB rows — use --dataset db (golden configs carry no kernel identity).")
    _emit_registry()

    db_path = Path(args.db) if args.db else resolve_tune_db()
    if not db_path.exists():
        logger.info("")
        logger.info("No tune DB at %s — skipping the measured per-knob regret analysis.", db_path)
        return
    logger.info("")
    logger.info("Reading: %s", db_path)

    all_kernels = Dataset.from_db(db_path, kernel=args.kernel).group_by_kernel_name()
    kernels = {
        name: [(s.all_knobs(), s.latency_us) for s in samples] for name, samples in all_kernels.items() if len(samples) >= args.min_variants
    }
    logger.info(
        "Kernels with ≥%d measured variants: %d (of %d total)",
        args.min_variants,
        len(kernels),
        len(all_kernels),
    )
    if not kernels:
        return

    rows = _compute_knob_regret(kernels)
    if not rows:
        logger.info("No knob varied across ≥2 values in any kernel — nothing to rank.")
        return
    _emit_regret_table(rows)

    interactions = _compute_interactions(kernels, [r.knob for r in rows])
    _emit_interaction_matrix([r.knob for r in rows], interactions)


def _check_offline_artifact() -> None:
    """Fail the command up front on an unloadable offline weights artifact
    (missing / feat_ver-mismatched override) — the per-shape eval harness catches
    exceptions into ERR rows, which would let a broken A/B exit 0."""
    from emmy.compiler.pipeline.search.prior import OfflinePrior  # noqa: PLC0415

    OfflinePrior()


def _prior_halves():
    """The two halves the report labels, in the order a failure is diagnosed in: the offline half decides what a
    cold sweep measures at all, so its ranking is upstream of everything the online half ever sees.

    An unfitted online half is dropped with a line saying so rather than reported. It would score every row the
    same constant, which reads as a model with no ranking ability — indistinguishable in a table from a trained
    model that collapsed, which is a real and different failure."""
    from emmy import config  # noqa: PLC0415
    from emmy.compiler.pipeline.search.prior import OfflinePrior, OnlinePrior  # noqa: PLC0415

    halves = [("offline", OfflinePrior())]
    online = OnlinePrior.load()
    if online.fitted:
        halves.append(("online", online))
    else:
        logger.info("No fitted online prior at %s — reporting the offline half only (run `emmy tune`).", config.online_path())
    return halves


def _freeze_provenance(path: Path) -> dict:
    """A freeze's ``sha256`` and the versions its rows are spelled in, for the report header.

    Empty for anything that is not a freeze directory (a live tune DB), so the header shows what
    a reader can act on rather than a placeholder. Read straight from the manifest — ``load_freeze``
    has already verified the digest by the time a report is built, so this does not re-verify."""
    manifest = path / "manifest.json"
    if not manifest.is_file():
        return {}
    try:
        m = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return {k: m[k] for k in ("sha256", "freeze_ver", "feat_ver", "knob_ver", "encoding_ver") if k in m}


def _measured_report(args, halves):
    """``eval prior --dataset nodes`` — the report over benched pools.

    ``--db`` takes a live tune DB or a measurement-freeze directory; ``load_node_rows`` sniffs which. The
    grouping, its key and every admission rule are :func:`group_measured`'s, so this reads the same pools the
    training-data work will.

    ``--kernel`` matches the op LABEL, since the store's own op identity is a digest with nothing readable in
    it. The label is a function of the row's ``S_*`` features, which every node of one op shares, so a filter
    keeps or drops a whole op atomically — a pool is never split against its own siblings."""
    from emmy import config  # noqa: PLC0415
    from emmy.compiler.pipeline.search.data import load_node_rows, op_label  # noqa: PLC0415
    from emmy.compiler.pipeline.search.data.group import group_measured  # noqa: PLC0415
    from emmy.compiler.pipeline.search.prior.report import EvalReport, measured_summaries  # noqa: PLC0415

    db_path = Path(args.db) if args.db else config.freeze_path()
    if not db_path.exists():
        logger.error("no measurement freeze or tune DB at %s — pass --db to point at one.", db_path)
        sys.exit(2)
    rows = load_node_rows(db_path)
    if args.kernel:
        rows = [r for r in rows if args.kernel in op_label(r.features)]
    groups, dropped = group_measured(rows)
    header = {
        "dataset": "nodes",
        "source": str(db_path),
        # A freeze's identity travels with the numbers: two reports are comparable only when
        # they were computed over the same rows, and the digest is what says so. Absent for a
        # live DB, which has no such identity — that is the point of preferring a freeze.
        **_freeze_provenance(db_path),
        "kernel": args.kernel,
        "rows": len(rows),
        "groups": len(groups),
        "dropped": dropped,
    }
    return EvalReport(header, [c for half, prior in halves for c in measured_summaries(half, groups, prior.score_rows)])


def _golden_report(args, halves):
    """``eval prior --dataset golden`` — the report over the recorded golden corpus.

    Built by ``emmy fit``'s own case builder over the FULL featurization, not the fit's ``D_*`` view. The view
    is a property of the model being fitted, and this command scores two model classes: the linear half reads
    only its own weight names, so its ranks are identical either way, while the online half regresses on the
    ``S_*`` / ``H_*`` columns a narrow view drops and would otherwise be asked about a kernel with no shape."""
    from emmy.commands.fit import build_golden_groups  # noqa: PLC0415
    from emmy.compiler.pipeline.search.prior.report import EvalReport, golden_summaries  # noqa: PLC0415

    logger.info("Building golden pools (each golden under its own card's context) ...")
    groups, skipped = build_golden_groups("*", sample=args.pool_sample, kernel=args.kernel)
    header = {
        "dataset": "golden",
        "source": "recorded golden corpus",
        "kernel": args.kernel,
        "pool_sample": args.pool_sample,
        "groups": len(groups),
        "positives": sum(len(g.golden_ids) for g in groups),
        "skipped": len(skipped),
    }
    return EvalReport(header, [c for half, prior in halves for c in golden_summaries(half, groups, prior.score_rows)])


def handle_eval_prior(args) -> None:
    """``eval prior`` — how well each prior half ranks a candidate pool.

    Two datasets, two different questions, one report schema (see ``search/prior/report.py``): benched pools
    say what a wrong pick COST, golden pools only say where the known-good row landed. ``--dataset golden``
    additionally runs the deploy-faithful check the ranks are a screen for — the greedy pipeline pick vs the
    recorded golden, with the deployable -O3 latency of the prior's pick beside it."""
    resolve_online_arg(args)
    resolve_offline_arg(args)
    _check_offline_artifact()
    require_source(
        args,
        {"golden", "nodes"},
        "eval prior ranks candidate pools: use --dataset golden (recorded goldens) or --dataset nodes (a tune DB "
        "or a measurement freeze). --dataset db reads only fully-decided leaf rows, with no op identity or compile "
        "regime to group them by — pass the same DB with --dataset nodes.",
    )
    halves = _prior_halves()
    golden = args.dataset == "golden"
    report = _golden_report(args, halves) if golden else _measured_report(args, halves)
    _emit_report(report)
    if args.json_out:
        storage.write_json(Path(args.json_out), report.to_json(), indent=2)
        logger.info("wrote %s", args.json_out)
    if golden:
        _emit_golden_deploy_check(args)


def _metric(block: dict, key: str, fmt: str) -> str:
    """One metric value with the pools it was computed over, or ``—`` when nothing in the summary qualified.

    The count is appended only where the block carries one, which is where the metric has a size minimum and so
    can cover fewer pools than the summary holds."""
    value = block.get(key)
    if value is None:
        return "—"
    return f"{fmt.format(value)} ({block['groups']})" if "groups" in block else fmt.format(value)


# Per dataset: the axis columns, then ``(header, render(summary))`` for each metric column. The axes are the ones the
# report keyed its summaries on — the renderer names them rather than discovering them, so a column order is a
# decision made here and not a side effect of dict insertion.
_REPORT_TABLES = {
    "nodes": (
        ["half", "gpu", "H_opt"],
        [
            ("rho", lambda c: _metric(c.metrics["spearman"], "median", "{:+.2f}")),
            ("regret@1", lambda c: _metric(c.metrics["regret1"], "median", "{:.2f}x")),
            ("worst@1", lambda c: _metric(c.metrics["regret1"], "worst", "{:.2f}x")),
            (f"regret@{report_mod.TOPK}", lambda c: _metric(c.metrics[f"regret{report_mod.TOPK}"], "median", "{:.2f}x")),
        ],
    ),
    "golden": (
        ["half", "gpu", "tier", "pool"],
        [
            ("rank", lambda c: _metric(c.metrics["rank"], "median", "{:g}")),
            ("rank(opt)", lambda c: _metric(c.metrics["rank"], "median_optimistic", "{:g}")),
            *((f"top{k}", lambda c, k=k: f"{c.metrics[f'top{k}']['count']}/{c.groups - c.unscored}") for k in (1, 10, 50)),
        ],
    ),
}

_REPORT_CAPTIONS = {
    "nodes": [
        "ranking quality over benched pools (rho: +1 = the model orders them as the hardware does;",
        "regret: 1.00x = the pick IS the measured best). Each number's (n) is the pools it covers.",
    ],
    "golden": [
        "golden rank — a SCREEN, not a gate: it says where a verified config landed, never what",
        "missing it costs. Only regret over benched pools (--dataset nodes) measures that.",
    ],
}


def _emit_report(report) -> None:
    """Print an :class:`EvalReport` — the provenance header, then one table of summaries.

    Which columns appear follows the report's dataset, since that is what decided which metrics the summaries carry.
    The ``pools`` column is the summary's own total, annotated when the model could not score some of them: an
    unscored pool is not a small one, and a report that dropped it silently would show a healthy corpus with no
    sign that part of the deploy surface is unmeasured."""
    head = report.header
    logger.info("")
    logger.info("[prior] %s dataset — %s", head.get("dataset", "?"), head.get("source", ""))
    # Every remaining header key, whatever the dataset put there. Printed generically so a builder that starts
    # recording a new provenance field does not also have to teach this about it — a count nobody prints is a
    # count nobody checks.
    provenance = ", ".join(f"{k}={head[k]}" for k in head if k not in ("dataset", "source", "dropped") and head[k] is not None)
    if provenance:
        logger.info("  %s", provenance)
    if dropped := head.get("dropped"):
        logger.info("  rows dropped before grouping: %s", ", ".join(f"{n} {why}" for why, n in sorted(dropped.items())))
    if not report.summaries:
        logger.info("  no candidate pools to score")
        return

    axes, metrics = _REPORT_TABLES[head["dataset"]]
    columns = [Col(a) for a in (*axes, "pools")] + [Col(name) for name, _ in metrics]
    rows = [
        [summary.axes.get(a, "") for a in axes]
        + [str(summary.groups) + (f" ({summary.unscored} unscored)" if summary.unscored else "")]
        + [render(summary) for _, render in metrics]
        for summary in report.summaries
    ]
    logger.info("")
    for line in _REPORT_CAPTIONS[head["dataset"]]:
        logger.info("  %s", line)
    for line in render_table(columns, rows, rule=True, indent="  "):
        logger.info("%s", line)


def _emit_golden_deploy_check(args) -> None:
    """The deploy-faithful half of ``eval prior --dataset golden``: the greedy tile-pipeline pick vs the
    recorded golden, per shape, with the deployable (-O3) latency of the prior's pick read from the online
    reservoir where one exists. This is what the golden RANK is only a screen for — a rank says where the
    verified row sat in the enumeration, this says what actually gets compiled."""
    from emmy.compiler.pipeline.search.prior import OnlinePrior, diagnostics  # noqa: PLC0415

    if args.features:
        _emit_golden_features(args.kernel)
    prior = OnlinePrior.load()
    # Deployable (-O3) perf of the prior's pick vs golden, read from the reservoir (no
    # re-bench); empty when there's no tuned -O3 data (column shows '—').
    perf = diagnostics.golden_deploy_perf(prior, args.kernel) if prior.fitted else {}
    _emit_prior_golden_check(_golden_configs(args.kernel), perf=perf)


def handle_eval_golden(args) -> None:
    """Validate one file-scoped golden corpus against the pinned serving envelope."""
    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.compiler.pipeline import CUDA_PASSES, Pipeline  # noqa: PLC0415
    from emmy.compiler.pipeline.search.golden import (
        # noqa: PLC0415,
        GoldenFileValidation,
        load_golden_file,
        load_golden_records,
        sole_evidence,
    )
    from emmy.compiler.pipeline.search.pins import pinned_knobs  # noqa: PLC0415
    from emmy.serving.release import load_serving_config, model_matches  # noqa: PLC0415
    from emmy.serving.twins import capture_twin_graphs  # noqa: PLC0415

    try:
        serving = load_serving_config(args.serving_config)
        golden_path = Path(args.golden).resolve()
        if golden_path != serving.golden_file:
            raise ValueError(f"serving config names {serving.golden_file}, not {golden_path}")
        document = load_golden_file(golden_path, validation=GoldenFileValidation.REPOSITORY)
        records = load_golden_records(document)
        ctx = Context.probe()
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error("golden evaluation setup failed: %s", exc)
        sys.exit(2)

    cap = tuple(document["compute_cap"])
    if document.get("gpu_name") != serving.gpu_name:
        logger.error("golden GPU %r does not match serving config GPU %r", document.get("gpu_name"), serving.gpu_name)
        sys.exit(1)
    if ctx.gpu_name != serving.gpu_name or tuple(ctx.compute_capability) != cap:
        logger.error(
            "live GPU %r sm_%d%d does not match golden/config target %r sm_%d%d",
            ctx.gpu_name,
            *ctx.compute_capability,
            serving.gpu_name,
            *cap,
        )
        sys.exit(1)
    if not records or any(not model_matches(record.model, serving) for record in records):
        recorded = sorted({record.model or "(missing)" for record in records})
        logger.error("golden model provenance %s does not cover %s", ", ".join(recorded) or "(none)", serving.model_provenance)
        sys.exit(1)

    expected = {(row.bindings, row.pins) for row in serving.realizations}
    missing = []
    for config_index, config in enumerate(document["configs"]):
        actual = {
            (tuple(sorted(realization["bindings"].items())), tuple(sorted(realization["pins"].items())))
            for realization in config["realizations"]
        }
        for bindings, pins in sorted(expected - actual, key=lambda item: (item[1], item[0])):
            missing.append((config_index, dict(bindings), pins))
    if missing:
        for config_index, bindings, pins in missing[:20]:
            logger.error("configs[%d] missing realization bindings=%s pins=%s", config_index, bindings, dict(pins))
        if len(missing) > 20:
            logger.error("... and %d more missing target realizations", len(missing) - 20)
        sys.exit(1)

    logger.info("OK: %d verified realizations cover %s on %s.", len(records), serving.model_provenance, serving.gpu_name)
    _emit_prior_golden_check(records, title=False)
    if _emit_offer_audit(records):
        sys.exit(1)

    source = serving.model_provenance
    try:
        if serving.static_only:
            graphs = capture_twin_graphs(source, decode_bucket=1, prefill_bucket=0, symbolic=False, static_only=True)
        else:
            graphs = capture_twin_graphs(
                source,
                decode_bucket=0,
                prefill_bucket=0,
                extra_widths=serving.static_widths,
                symbolic=True,
            )
    except (NotImplementedError, ValueError) as exc:
        logger.error("in-model audit cannot represent %s: %s", source, exc)
        sys.exit(1)

    # The serving-matrix half of the gate: each lane's twins compiled with that lane's rows as
    # the only evidence, strictly, on the live card the golden names — a fork no golden row
    # decides is an EvidenceError naming the kernel, never a prediction the prior makes.
    failed = False
    for pins in sorted({row.pins for row in serving.realizations}, key=repr):
        lane = _format_pins(pins)
        broken = 0
        with pinned_knobs(dict(pins)), sole_evidence([record for record in records if record.pins == pins]):
            for name, graph in graphs.items():
                try:
                    Pipeline.build(CUDA_PASSES).run(graph, ctx=ctx)
                except Exception as exc:  # noqa: BLE001 — one twin's failure is that twin's verdict
                    broken += 1
                    logger.error("%s: %s: %s", lane, name, " ".join(f"{type(exc).__name__}: {exc}".split()))
        logger.info("%s: %d twin(s) deploy from the golden rows alone, %d do not", lane, len(graphs) - broken, broken)
        failed |= bool(broken)
    if failed:
        logger.error("serving audit failed: every fork of every reachable kernel must be decided by a golden row")
        sys.exit(1)


def handle_eval_variants(args) -> None:
    """``eval variants`` — per-kernel leaderboard of the tune DB's measured
    variants (fastest first, knob columns aligned), with the config the prior would
    deploy marked + ranked. The per-kernel "did the
    search/prior reach the best measured config, and which knobs distinguish
    it?" drill-down view."""
    require_source(args, {"db"}, "eval variants lists measured tune-DB rows — --dataset golden has no per-variant measurements.")
    resolve_online_arg(args)
    from emmy import config  # noqa: PLC0415
    from emmy.compiler.pipeline.search.prior import load_prior  # noqa: PLC0415

    db_path = Path(args.db) if args.db else resolve_tune_db()
    if not db_path.exists():
        logger.error("no tune DB at %s — pass --db or run `emmy tune` first.", db_path)
        return
    groups = Dataset.from_db(db_path, kernel=args.kernel).group_by_kernel_name()
    if not groups:
        logger.info("No measured variants%s in %s.", f" matching --kernel '{args.kernel}'" if args.kernel else "", db_path)
        return
    fails = Counter(s.name for s in Dataset.from_db(db_path, kernel=args.kernel, status="bench_fail") if s.name)
    # FallbackPrior: the online CatBoost when fitted, else the cold OfflinePrior — the same ranking compile/run use.
    prior = load_prior()
    if not prior.fitted:
        logger.info("No fitted prior at %s — the pick is the cold OfflinePrior's (the ranking compile/run use).", config.online_path())
    for name in sorted(groups):
        _emit_variant_table(name, groups[name], prior, n_fail=fails.get(name, 0), top=args.top)


def handle_eval_failures(args) -> None:
    """``eval failures`` — the tune DB's ``bench_fail`` rows clustered by
    ``(kernel, error)``, each cluster with its count and the tunable knob
    assignments shared by EVERY failing row (the "all 28 rows have ``TMA=1``"
    signal). Replaces grepping the tune log against hand-written SQL; rows from
    pre-error-column DBs cluster under ``(no error recorded)``."""
    require_source(args, {"db"}, "eval failures reads tune-DB bench_fail rows — --dataset golden records no failures.")
    db_path = Path(args.db) if args.db else resolve_tune_db()
    if not db_path.exists():
        logger.error("no tune DB at %s — pass --db or run `emmy tune` first.", db_path)
        return
    fails = [s for s in Dataset.from_db(db_path, kernel=args.kernel, status="bench_fail") if s.name]
    n_ok = len(Dataset.from_db(db_path, kernel=args.kernel))
    if not fails:
        logger.info("No bench_fail rows%s in %s (%d ok rows).", f" matching --kernel '{args.kernel}'" if args.kernel else "", db_path, n_ok)
        return
    clusters: dict[tuple, list] = defaultdict(list)
    for s in fails:
        clusters[(s.name, s.error or "(no error recorded)")].append(s)
    logger.info("%d bench_fail rows (beside %d ok) in %s:", len(fails), n_ok, db_path)
    for (name, error), grp in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        shared = dict(grp[0].knobs)
        for s in grp[1:]:
            shared = {k: v for k, v in shared.items() if s.knobs.get(k) == v}
        knob_txt = ", ".join(f"{k}={v}" for k, v in sorted(shared.items())) or "(no shared knobs)"
        logger.info("")
        logger.info("  %s — %d row(s)", name, len(grp))
        logger.info("    error: %s", error)
        logger.info("    shared knobs: %s", knob_txt)


def _emit_variant_table(name: str, samples: list, prior, *, n_fail: int, top: int) -> None:
    """One kernel's leaderboard: measured leaf configs sorted by latency, the prior's pick marked,
    knobs in the canonical aligned columns (``tuning_knob_items`` — the same filtered view the
    ``run --bench`` kernel table renders). Non-leaf rows (partial-knob fork nodes) are dropped —
    a partial config is not a variant.

    The ``us`` column is the measured latency as stored. A sweep measures in the deployable
    regime, so on a store written since that became true every row is a deploy latency — but
    ``Dataset.from_db`` reads every ``context_key`` and ``PerfSample`` carries none, so a store
    holding rows from the era of a separate ranking lane still pools both here."""
    from emmy.compiler.pipeline.knob import tuning_knob_items  # noqa: PLC0415

    kmax = max(len(s.knobs) for s in samples)
    leaves = sorted((s for s in samples if len(s.knobs) == kmax), key=lambda s: s.latency_us)
    # Score through ``Prior.pick`` — measured evidence first, model argmin otherwise — so the
    # marker shows the config greedy ``compile`` / ``run`` would actually deploy, not just the
    # model's favourite.
    best_i, _ = prior.pick([s.all_knobs() for s in leaves])
    pick = leaves[best_i]
    rank = best_i + 1

    n_prefix = len(leaves) if not top else min(top, len(leaves))
    shown = list(enumerate(leaves[:n_prefix], start=1))
    if rank > n_prefix:
        shown.append((rank, pick))
    hidden = len(leaves) - n_prefix - (1 if rank > n_prefix else 0)

    logger.info("")
    logger.info("%s — %d measured configs%s", name, len(leaves), f", {n_fail} bench_fail" if n_fail else "")
    kcols, kcells = knob_columns([{k: (v, False) for k, v in tuning_knob_items(s.knobs)} for _, s in shown])
    columns = [Col("rank", "r"), Col("us", "r"), Col("pick"), *kcols]
    data = []
    for (r, s), kc in zip(shown, kcells, strict=True):
        data.append([str(r), f"{s.latency_us:.1f}", ("◄", _GREEN) if s is pick else "", *kc])
    for line in render_table(columns, data, indent="  "):
        logger.info(line)
    if hidden > 0:
        logger.info("  … %d more (--top 0 shows all)", hidden)
    if len(leaves) >= 2:
        ratio = pick.latency_us / leaves[0].latency_us
        flag = "  <-- misses best" if ratio > 1.2 else ""
        logger.info("  pick: rank %d/%d, %.2fx of best (measured latency)%s", rank, len(leaves), ratio, flag)


def _emit_registry() -> None:
    """List every registered :class:`~emmy.compiler.pipeline.knob.Knob` — the
    canonical tuning schema (name, type, candidate hints, help) collected
    by ``knob.registry`` from all loaded passes, regardless of any DB."""
    from emmy.compiler.pipeline import CUDA_PASSES, Pipeline, knob  # noqa: PLC0415

    # ``registry`` only sees passes already imported into ``sys.modules``; build
    # the full pipeline once so every Knob-bearing rule module is loaded first.
    Pipeline.build(CUDA_PASSES)
    import textwrap  # noqa: PLC0415

    reg = knob.registry()
    names = sorted(reg)
    kw = max((len(n) for n in names), default=4)  # knob-name column width
    tw = max((len(reg[n].type.value) for n in names), default=4)  # type column width
    hw = 33  # hints column width (truncated past this)
    help_w = 64  # help wraps to this width; continuation lines indent under it
    indent = " " * (kw + 2 + tw + 2 + hw + 2)

    logger.info("Registered tuning knobs (%d) — the canonical schema:", len(reg))
    logger.info(f"{'knob':<{kw}}  {'type':<{tw}}  {'candidates':<{hw}}  help")
    logger.info("-" * (kw + 2 + tw + 2 + hw + 2 + help_w))
    for name in names:
        k = reg[name]
        hints = ", ".join(str(h) for h in k.hints) if k.hints else "-"
        if len(hints) > hw:
            hints = hints[: hw - 1] + "…"
        help_txt = " ".join((k.help or "").split())  # collapse whitespace/newlines
        lines = textwrap.wrap(help_txt, width=help_w) or [""]
        logger.info(f"{name:<{kw}}  {k.type.value:<{tw}}  {hints:<{hw}}  {lines[0]}")
        for cont in lines[1:]:
            logger.info(indent + cont)


def _ratio_color(matched: int, total: int) -> str:
    """Green (all match) / yellow (>80%) / red (otherwise)."""
    frac = matched / total if total else 1.0
    return _GREEN if matched == total else (_YELLOW if frac > 0.8 else _RED)


def _knob_cells(entry: tuple) -> dict[str, tuple[str, bool]]:
    """``{knob: (value_text, red?)}`` for one renderable entry (no ``NAME=`` prefix —
    :func:`~emmy.commands.table.knob_columns` puts the name in the column header).
    A ``("row", lead, gold, got)`` entry renders ``found/golden`` per knob, red where the
    two differ (``knob.values_equal`` — so a legacy golden spelling compares equal to the
    site-form pick it realizes as); a ``("total", lead, summaries)`` entry carries its summaries
    pre-built."""
    if entry[0] == "total":
        return entry[2]
    _, _, gold, got = entry
    return {k: (f"{got.get(k, '-')}/{gold[k]}", not _knob_eq(k, gold[k], got)) for k in gold}


def _knob_eq(k: str, gv, got: dict) -> bool:
    """Whether the picked knob dict ``got`` reproduces the golden's ``k=gv`` — value equality
    through the registry-canonical :func:`~emmy.compiler.pipeline.knob.values_equal`, so the
    legacy golden corpus keeps matching the site-form picks during the step-7 window."""
    from emmy.compiler.pipeline.knob import values_equal  # noqa: PLC0415

    return k in got and values_equal(k, gv, got[k])


def _emit_golden_table(lead_cols: list[Col], entries: list[tuple], caption: str) -> None:
    """Stream a golden table via ``logger``: ``lead_cols`` (kernel, m/t, …) plus the aligned
    ``found/golden`` knob columns (knob name in the header, value-only summaries). ``entries``
    preserves config order — each is ``("row", lead_cells, gold, got)``,
    ``("total", lead_cells, knob_cells)`` (a pre-built aggregate row), or
    ``("err", kernel_name, message)``; an error row prints its kernel name (aligned to the
    kernel column) then the raw message in place. ``caption`` is printed above the table."""
    body = [e for e in entries if e[0] != "err"]
    kcols, kcells = knob_columns([_knob_cells(e) for e in body])
    columns = lead_cols + kcols
    data = [e[1] + kc for e, kc in zip(body, kcells, strict=True)]
    # Floor the kernel column to the widest error-row name so error rows align with the table.
    floor = [max((len(e[1]) for e in entries if e[0] == "err"), default=0)] + [0] * (len(columns) - 1)
    kernel_w = col_widths(columns, data, floor)[0]
    lines = iter(render_table(columns, data, indent="  ", min_widths=floor))
    logger.info("  " + caption)
    logger.info(next(lines))  # header row (column names, knobs included)
    for e in entries:
        logger.info("  " + e[1].ljust(kernel_w) + "  ERR  " + e[2] if e[0] == "err" else next(lines))


def _emit_golden_features(kernel_filter: str | None) -> None:
    """Print, per golden config, the exact feature vector the online
    :class:`OnlinePrior` regresses on — ``features.knob_features(merged)`` where
    ``merged`` is the ``H_*`` host/regime features + the ``S_*`` structural/shape
    features (obtained by compiling the shape to the loop dialect, where
    the IdentityStrategy stamps at the loop terminal) + the golden tuning knobs. This is
    the model's *input* for that shape+config — note the shape enters only as the
    coarse ``S_ext_*`` extent products/maxes; the occupancy / CTA-count / reuse
    terms that drive matmul perf (the engineered ``D_*`` features) are NOT here."""
    import logging as _logging  # noqa: PLC0415

    from emmy.compiler.pipeline.knob import CTX_PREFIX, STRUCT_PREFIX  # noqa: PLC0415
    from emmy.compiler.pipeline.search.data import Sample  # noqa: PLC0415
    from emmy.compiler.pipeline.search.golden import goldens_for_live_gpu  # noqa: PLC0415

    configs = [g for g in goldens_for_live_gpu() if g.is_matmul]
    if kernel_filter:
        configs = [g for g in configs if kernel_filter in g.name]

    logger.info("")
    logger.info("Online-prior feature vector (features.knob_features) — the CatBoost regressor's input per golden config:")
    quiet = [_logging.getLogger(n) for n in ("emmy.compiler", "emmy.commands.trace")]
    prev = [lg.level for lg in quiet]
    for lg in quiet:
        lg.setLevel(_logging.WARNING)
    try:
        for g in configs:
            try:
                # compile_s_feats=True derives the full S_* histogram (the CatBoost input), as eval did inline.
                feats = Sample.from_golden(g, compile_s_feats=True).features()
            except Exception as e:  # noqa: BLE001 — one shape's error shouldn't abort the report
                logger.info("  %-26s  ERR  %s", g.name, " ".join(f"{type(e).__name__}: {e}".split())[:100])
                continue
            logger.info("  %s  (%d features):", g.name, len(feats))
            tuning = {k: v for k, v in feats.items() if not k.startswith((STRUCT_PREFIX, CTX_PREFIX))}
            for label, sel in (
                ("S_", {k: v for k, v in feats.items() if k.startswith(STRUCT_PREFIX)}),
                ("H_", {k: v for k, v in feats.items() if k.startswith(CTX_PREFIX)}),
                ("knob", tuning),
            ):
                if sel:
                    logger.info("    %-5s %s", label, " ".join(f"{k}={v:g}" for k, v in sorted(sel.items())))
    finally:
        for lg, lv in zip(quiet, prev, strict=True):
            lg.setLevel(lv)


def _golden_configs(kernel_filter: str | None):
    """The matmul golden configs for the **live** card, optionally filtered by name
    substring. Scoping to the live GPU (:func:`goldens_for_live_gpu`) keeps the eval
    views about the card in hand when a multi-GPU goldens dir is checked in — a name
    recurs once per card and the GPU-blind ``ShapeKey`` join would otherwise mix
    cards (5090 / PRO 6000 even share ``compute_cap``)."""
    from emmy.compiler.pipeline.search.golden import goldens_for_live_gpu  # noqa: PLC0415

    configs = [g for g in goldens_for_live_gpu() if g.is_matmul]
    if kernel_filter:
        configs = [g for g in configs if kernel_filter in g.name]
    return configs


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _perf_color(ratio: float) -> str:
    """``vs gold`` colour: green = pick beats golden by >3%, **default (no colour)**
    within 3% (the expected outcome — shouldn't stand out), yellow = up to 20% slower,
    red = worse."""
    if ratio < 0.97:
        return _GREEN
    if ratio <= 1.03:
        return ""
    return _YELLOW if ratio <= 1.2 else _RED


def _perf_cell(perf: dict | None, name: str) -> tuple[str, str] | None:
    """The ``vs gold`` lead summary for one shape: ``pick_us/golden_us`` as ``N.NNx``
    (green >3% faster, white within 3%, yellow/red slower), ``—`` when the shape has no
    -O3 measurement. ``None`` when ``perf`` wasn't supplied (column absent — e.g.
    ``eval golden``)."""
    if perf is None:
        return None
    ratio = perf.get(name)
    if ratio is None:
        return ("—", "")
    return (f"{ratio:.2f}x", _perf_color(ratio))


def _bare_families(knobs: dict) -> dict:
    """Aggregate exact-site tuning knobs by family for the single-site summary table.

    The table compares one resolved choice per family rather than schedule identities. First key
    wins on a family collision; a multi-node kernel has no single family value to summarize.
    """
    from emmy.compiler.pipeline.knob import family_of  # noqa: PLC0415

    out: dict = {}
    for k, v in knobs.items():
        out.setdefault(family_of(k), v)
    return out


def _emit_prior_golden_check(configs: list, *, title: bool = True, perf: dict | None = None) -> None:
    """Greedy fork pick through the tile pipeline vs recorded golden. The pick reads
    the online-prior JSON (``config.online_path()``: ``EMMY_ONLINE_FILE`` /
    ``--prior``); option-0 with no fitted prior. Stops at the tile dialect (every
    knob fork resolves there: no codegen / nvcc). One row per shape (configs sharing a
    name share a snippet → one greedy pick): the pick is scored against the shape's
    *closest* recorded golden (most knobs reproduced), so multiple goldens for a shape
    don't duplicate rows. A trailing ``TOTAL`` row carries per-knob match counts over the
    deduped rows + the exactly-reproduced row count. Rows print with column-aligned
    ``found/golden`` knobs (canonical order). ``title`` prints the
    ``Golden reproduction — … prior: <path>`` banner (``eval prior``); ``eval golden``
    passes ``title=False`` for just the table."""
    import logging as _logging  # noqa: PLC0415

    from emmy import config  # noqa: PLC0415
    from emmy.compiler.pipeline import TILE_PASSES, Pipeline  # noqa: PLC0415

    def tunable(knobs: dict) -> dict:
        return {k: v for k, v in knobs.items() if not k.startswith(("S_", "H_"))}

    def picked(graph) -> dict:
        compiled = Pipeline.build(TILE_PASSES).run(graph)  # tile dialect only — no codegen/nvcc
        knobs: dict = {}
        for node in compiled.nodes.values():
            k = getattr(node.op, "knobs", None)
            if k:
                knobs.update(k)
        return _bare_families(tunable(knobs))

    if title:
        online_path = config.online_path()
        logger.info("")
        logger.info(
            "Golden reproduction — greedy pipeline pick vs recorded golden; prior: %s (%s):",
            online_path,
            "loaded" if online_path.exists() else "MISSING → option-0",
        )
    # Silence the trace/compile chatter (different logger subtrees) so this
    # function's own ``logger`` can stream one clean result line per config.
    quiet = [_logging.getLogger(n) for n in ("emmy.compiler", "emmy.commands.trace")]
    prev = [lg.level for lg in quiet]
    for lg in quiet:
        lg.setLevel(_logging.WARNING)
    # Group configs sharing a shape (same name → same snippet → same greedy pick) so the
    # table carries one row per shape, not one per recorded golden. Each shape's pick is
    # compared against its *closest* golden (the config it reproduces the most knobs of).
    # Realizations with different input pins form separate rows: each greedy pick runs under
    # the exact pins that produced it and never mixes with another enumeration regime.
    from emmy.compiler.pipeline.search.pins import pinned_knobs  # noqa: PLC0415

    groups: dict[str, list] = {}
    for g in configs:
        groups.setdefault(g.name, []).append(g)
    n_match = n_rows = 0
    knob_match: dict[str, int] = {}  # deduped rows where the pick matched this knob
    knob_total: dict[str, int] = {}  # deduped rows whose golden carries this knob
    entries: list[tuple] = []  # ("row", lead_cells, gold, got) | ("err", name, message)
    try:
        for name, group in groups.items():
            by_pins: dict[tuple, list] = {}
            for config in group:
                by_pins.setdefault(config.pins, []).append(config)
            for pins, sub in sorted(by_pins.items(), key=lambda item: repr(item[0])):
                label = _realization_label(name, pins)
                try:
                    with pinned_knobs(dict(pins)):
                        got = picked(sub[0].target_program.copy())
                except Exception as e:  # noqa: BLE001 — one shape's error shouldn't abort the report
                    entries.append(("err", label, " ".join(f"{type(e).__name__}: {e}".split())[:100]))
                    continue
                # Closest golden: most knobs reproduced (registry-canonical values_equal, via
                # _knob_eq — the legacy corpus vs the site-form pick), tie-broken by match fraction.
                golds = [tunable(c.knobs) for c in sub]
                scored = [(sum(1 for k in gd if _knob_eq(k, gd[k], got)), gd) for gd in golds]
                matched, gold = max(scored, key=lambda t: (t[0], t[0] / len(t[1]) if t[1] else 1.0))
                n_match += matched == len(gold)
                n_rows += 1
                for k in gold:
                    knob_total[k] = knob_total.get(k, 0) + 1
                    knob_match[k] = knob_match.get(k, 0) + _knob_eq(k, gold[k], got)
                lead = [label, (f"{matched}/{len(gold)}", _ratio_color(matched, len(gold)))]
                pc = _perf_cell(perf, name)
                if pc is not None:
                    lead.append(pc)
                entries.append(("row", lead, gold, got))
    finally:
        for lg, lv in zip(quiet, prev, strict=True):
            lg.setLevel(lv)
    # Totals row (replaces a trailing summary line): per-knob match counts over the deduped
    # rows, plus the exactly-reproduced row count in the m/t column.
    total_cells = {k: (f"{knob_match[k]}/{knob_total[k]}", knob_match[k] != knob_total[k]) for k in knob_total}
    total_lead = ["TOTAL", (f"{n_match}/{n_rows}", _ratio_color(n_match, n_rows))]
    if perf is not None:
        vals = list(perf.values())
        if vals:
            import statistics  # noqa: PLC0415

            geo = statistics.geometric_mean(vals)
            total_lead.append((f"{geo:.2f}x", _perf_color(geo)))
        else:
            total_lead.append(("—", ""))
    entries.append(("total", total_lead, total_cells))
    lead_cols = [Col("kernel"), Col("m/t")] + ([Col("vs gold", "r")] if perf is not None else [])
    _emit_golden_table(lead_cols, entries, "knobs (found/golden)")


def _emit_offer_audit(configs: list) -> bool:
    """The offer audit — does each recorded row still equal an enumerated leaf of its own target?

    The strict decode (``golden.decode_record``) asks it per entry: the persisted program replayed
    under the entry's own input pins, with the target's other entries walking the same path, and
    the spelled row compared with that kernel's enumerated leaves by exact schedule-row identity.
    An entry whose row equals no leaf is ``UNREALIZED``: it is no evidence a deploy can use, so
    the gate fails — re-record an offered row in this input regime, or close the enumeration gap.
    This is the OWN-SNIPPET view; the serving-matrix compile in :func:`handle_eval_golden` closes
    the other side, whether the fused serving graphs are decided by these rows. Returns True when any entry
    is unrealized (``eval golden`` exits 1)."""
    from emmy.compiler.pipeline.search.golden import decode_record, siblings_of  # noqa: PLC0415

    def kstr(g) -> str:  # the entry's distinguishing knobs, empty families dropped
        return ",".join(f"{k}={v}" for k, v in g.knobs.items() if v not in ("", None))

    logger.info("")
    logger.info("Offer audit — does each recorded row still equal an enumerated leaf (own snippet, deployable regime)?")
    unrealized = 0
    for g in configs:
        try:
            reason = decode_record(g, siblings_of(g, configs))
        except Exception as exc:  # noqa: BLE001 — one entry's error is that entry's verdict
            reason = f"{type(exc).__name__}: {exc}"
        if reason is None:
            continue
        unrealized += 1
        why = " ".join(reason.split())[:120]
        logger.info("  %-44s  UNREALIZED  %.1fus  %s  (%s)", _realization_label(g.name, g.pins), g.emmy_us, kstr(g), why)
    if unrealized:
        logger.error("  offer audit: %d of %d entries equal no enumerated leaf in their input regime", unrealized, len(configs))
        return True
    logger.info("  offer audit: all %d entries equal an enumerated leaf", len(configs))
    return False


@dataclass
class KnobRow:
    knob: str
    n_kernels: int
    median_values: int
    median_regret: float
    p90_regret: float
    geomean_regret: float


def _compute_knob_regret(kernels: dict[str, list[tuple[dict, float]]]) -> list[KnobRow]:
    per_knob_regret: dict[str, list[float]] = defaultdict(list)
    per_knob_n_values: dict[str, list[int]] = defaultdict(list)
    for variants in kernels.values():
        all_knobs: set[str] = set()
        for knobs, _ in variants:
            all_knobs.update(knobs.keys())
        for K in all_knobs:
            best_by_value: dict = {}
            for knobs, us in variants:
                v = knobs.get(K)
                if v is None:
                    continue
                if v not in best_by_value or us < best_by_value[v]:
                    best_by_value[v] = us
            if len(best_by_value) < 2:
                # Knob took only one distinct value across this kernel's
                # variants — no choice to evaluate.
                continue
            latencies = list(best_by_value.values())
            per_knob_regret[K].append(max(latencies) / min(latencies))
            per_knob_n_values[K].append(len(best_by_value))

    rows = [
        KnobRow(
            knob=K,
            n_kernels=len(per_knob_regret[K]),
            median_values=int(median(per_knob_n_values[K])),
            median_regret=median(per_knob_regret[K]),
            p90_regret=_percentile(per_knob_regret[K], 0.90),
            geomean_regret=_geomean(per_knob_regret[K]),
        )
        for K in per_knob_regret
    ]
    rows.sort(key=lambda r: -r.geomean_regret)
    return rows


def _compute_interactions(
    kernels: dict[str, list[tuple[dict, float]]],
    knobs: list[str],
) -> dict[tuple[str, str], float | None]:
    """For each ordered pair (K1, K2): fraction of kernels where the
    argmin K2 value changes across different K1 values."""
    out: dict[tuple[str, str], float | None] = {}
    for K1 in knobs:
        for K2 in knobs:
            if K1 == K2:
                continue
            n_changes = 0
            n_total = 0
            for variants in kernels.values():
                argmin_by_v1: dict = {}
                for knobs_dict, us in variants:
                    v1 = knobs_dict.get(K1)
                    v2 = knobs_dict.get(K2)
                    if v1 is None or v2 is None:
                        continue
                    v1, v2 = v1, v2
                    prev = argmin_by_v1.get(v1)
                    if prev is None or us < prev[1]:
                        argmin_by_v1[v1] = (v2, us)
                if len(argmin_by_v1) < 2:
                    continue
                n_total += 1
                if len({entry[0] for entry in argmin_by_v1.values()}) > 1:
                    n_changes += 1
            out[(K1, K2)] = (n_changes / n_total) if n_total else None
    return out


def _emit_regret_table(rows: list[KnobRow]) -> None:
    cols = [
        Col("knob"),
        Col("n_kernels", "r"),
        Col("median_n_vals", "r"),
        Col("median_regret", "r"),
        Col("p90_regret", "r"),
        Col("geomean_regret", "r"),
    ]
    data = [
        [r.knob, str(r.n_kernels), str(r.median_values), f"{r.median_regret:.2f}x", f"{r.p90_regret:.2f}x", f"{r.geomean_regret:.2f}x"]
        for r in rows
    ]
    for line in render_table(cols, data, rule=True):
        logger.info(line)


def _emit_interaction_matrix(knobs: list[str], interactions: dict[tuple[str, str], float | None]) -> None:
    logger.info("")
    logger.info("knob interaction — frac of kernels where argmin(K2) changes across K1 values")
    logger.info("(high value = knobs are coupled; can't commit to K1 then search K2 independently)")
    cols = [Col("K1\\K2"), *(Col(k, "r") for k in knobs)]
    data = []
    for K1 in knobs:
        row = [K1]
        for K2 in knobs:
            v = None if K1 == K2 else interactions.get((K1, K2))
            row.append(f"{v:.2f}" if v is not None else "-")
        data.append(row)
    for line in render_table(cols, data):
        logger.info(line)


def _percentile(xs: list[float], p: float) -> float:
    s = sorted(xs)
    return s[int(round((len(s) - 1) * p))]


def _geomean(xs: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs))
