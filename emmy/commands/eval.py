"""``emmy eval <knobs|offline|online|golden|variants|failures>`` — evaluate the tuning machinery.

Six subcommands:

- ``eval knobs``     — print the registered knob schema, then (with a tune DB)
  per-knob **regret** + a knob-interaction matrix (the analysis below).
- ``eval offline``  — evaluate the cold-start ``OfflinePrior`` (``search/golden_eval``
  is the golden-eval glue around it) on the golden configs: the golden's **rank**
  under the prior over the enumeration (the position the tuner's patience must reach).
- ``eval online``     — evaluate the online ``OnlinePrior`` on the golden
  configs: the greedy pipeline pick vs golden (per-knob ``found/golden``), the
  golden's rank under the prior, and (``--features``) the regressor input vector.
- ``eval golden``    — validate one canonical golden YAML against the pinned serving
  configuration and live GPU, then reproduce its rows and audit the exact serving matrix.
- ``eval variants``  — per-kernel leaderboard of the tune DB's measured variants
  (fastest first), the config the prior deploys marked + ranked, and the -O3
  re-bench latency from the prior reservoir where one was recorded.
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

import logging
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from emmy.commands.compile import resolve_tune_db
from emmy.commands.dataset_args import add_dataset_args, require_source, resolve_offline_arg, resolve_online_arg
from emmy.commands.table import GREEN as _GREEN
from emmy.commands.table import RED as _RED
from emmy.commands.table import YELLOW as _YELLOW
from emmy.commands.table import Col, col_widths, knob_columns, render_table
from emmy.compiler.pipeline.search.data import Dataset

logger = logging.getLogger(__name__)


def _format_pins(pins) -> str:
    def spell(value) -> str:
        return str(value).lower() if isinstance(value, bool) else str(value)

    return ", ".join(f"{name}={spell(value)}" for name, value in pins) or "unpinned"


def _realization_label(name: str, pins) -> str:
    return name if dict(pins) == {"FAST_MATH": False} else f"{name} [{_format_pins(pins)}]"


def register_eval_command(subparsers) -> None:
    """``emmy eval <knobs|online|offline>`` — evaluate the tuning knobs, the
    online prior, or the cold-start OfflinePrior against the golden configs."""
    parser = subparsers.add_parser(
        "eval",
        help="Evaluate tuning knobs / the online prior / the offline prior against golden configs",
    )
    sub = parser.add_subparsers(dest="eval_target", required=True)

    pk = sub.add_parser("knobs", help="Print the registered knob schema + (with a tune DB) per-knob regret + interactions")
    add_dataset_args(pk, default="db", with_min_variants=True)
    pk.set_defaults(func=handle_eval_knobs)

    ph = sub.add_parser(
        "offline",
        aliases=["analytic"],  # pre-rename spelling
        help="Evaluate the cold-start OfflinePrior on the golden configs (golden's rank under the prior over the enumeration)",
    )
    ph.add_argument(
        "--offline-file",
        "--analytic-file",  # pre-rename spelling
        dest="offline_file",
        help="Offline weights artifact (JSON) to score with, for A/Bing candidate fits. "
        "Default: EMMY_OFFLINE_FILE or the repo-checked offline_weights.json.",
    )
    add_dataset_args(ph, default="golden")
    ph.set_defaults(func=handle_eval_offline)

    pp = sub.add_parser(
        "online",
        aliases=["prior"],  # pre-rename spelling
        help="Evaluate the online prior on the golden configs (greedy pick vs golden + golden's rank under the prior)",
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
        help="Offline weights artifact (JSON) for the offline blocks of this eval. "
        "Default: EMMY_OFFLINE_FILE or the repo-checked offline_weights.json.",
    )
    add_dataset_args(pp, default="golden")
    pp.add_argument(
        "--features",
        action="store_true",
        help="Also print the exact feature vector the prior (CatBoost) regresses on per golden config (features.knob_features).",
    )
    pp.add_argument(
        "--blame",
        action="store_true",
        help="With --dataset nodes: per-feature blame table — which features' terms pushed each missed fork's wrong pick, "
        "regret-weighted per fork family (diagnostic, not a gate metric).",
    )
    pp.add_argument(
        "--ablate",
        action="store_true",
        help="With --dataset nodes: ablation Δ table — each family's median regret change with one feature masked "
        "(<0 = actively misleading), with per-feature fork support.",
    )
    pp.set_defaults(func=handle_eval_online)

    pg = sub.add_parser(
        "golden",
        help="Validate one golden YAML against its pinned serving configuration and the live target GPU",
    )
    pg.add_argument("golden_file", metavar="GOLDEN_YAML", help="The exact canonical golden YAML to validate.")
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


def handle_eval_offline(args) -> None:
    """``eval offline`` — the cold-start OfflinePrior's rank of each golden."""
    require_source(args, {"golden"}, "eval offline ranks recorded golden configs — --dataset db has no golden to rank.")
    resolve_offline_arg(args)
    _check_offline_artifact()
    _emit_offline_eval(args.kernel)


def handle_eval_online(args) -> None:
    """``eval online`` — the online prior on the golden configs: the greedy pick vs
    golden, the golden's rank under the prior, and (with ``--features``) the
    regressor input vector. With ``--dataset db`` instead reports the prior's pick
    reachability over the tune DB's *measured* variants (the orthogonal view); with
    ``--dataset nodes`` reports fork sibling regret + leaf reachability over the tune
    DB's search-tree node store (the search-faithful, partial-config view)."""
    resolve_online_arg(args)
    resolve_offline_arg(args)
    _check_offline_artifact()
    if (args.blame or args.ablate) and args.dataset != "nodes":
        logger.error("--blame/--ablate attribute fork records — they need --dataset nodes.")
        sys.exit(2)
    if args.dataset == "db":
        _emit_online_db_reachability(args)
        return
    if args.dataset == "nodes":
        _emit_online_nodes(args)
        return
    if args.features:
        _emit_golden_features(args.kernel)
    _emit_online_eval(args.kernel)


def handle_eval_golden(args) -> None:
    """Validate one file-scoped golden corpus against the pinned serving envelope."""
    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.compiler.pipeline.search.audit import COMPILE_FAIL, audit_card, gap_keys, summarize  # noqa: PLC0415
    from emmy.compiler.pipeline.search.golden import (  # noqa: PLC0415
        GoldenFileValidation,
        load_golden_file,
        load_golden_records,
    )
    from emmy.compiler.pipeline.search.pins import pinned_knobs  # noqa: PLC0415
    from emmy.serving.release import load_serving_config, model_matches  # noqa: PLC0415
    from emmy.serving.twins import capture_in_model_graphs, capture_twin_graphs  # noqa: PLC0415

    try:
        serving = load_serving_config(args.serving_config)
        golden_path = Path(args.golden_file).resolve()
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
            try:
                graphs = capture_twin_graphs(
                    source,
                    decode_bucket=0,
                    prefill_bucket=0,
                    extra_widths=serving.static_widths,
                    symbolic=True,
                )
            except NotImplementedError:
                graphs = capture_in_model_graphs(source)
    except (NotImplementedError, ValueError) as exc:
        logger.error("in-model audit cannot represent %s: %s", source, exc)
        sys.exit(1)

    failed = False
    for pins in sorted({row.pins for row in serving.realizations}, key=repr):
        lane_records = [record for record in records if record.pins == pins]
        with pinned_knobs(dict(pins)):
            results = audit_card(graphs, serving.gpu_name, cap, goldens=lane_records)
        counts = summarize(results)
        gaps = gap_keys(results)
        lane = _format_pins(pins)
        logger.info(
            "%s: MATCH %d  DRIFT %d  GAP %d  compile_fail %d",
            lane,
            counts["MATCH"],
            counts["DRIFT"],
            counts["GAP"],
            counts[COMPILE_FAIL],
        )
        if counts["DRIFT"] or counts[COMPILE_FAIL] or gaps:
            failed = True
    if failed:
        logger.error("serving audit failed: every reachable kernel must match a verified realization")
        sys.exit(1)


def handle_eval_variants(args) -> None:
    """``eval variants`` — per-kernel leaderboard of the tune DB's measured
    variants (fastest first, knob columns aligned), the config the prior would
    deploy marked + ranked, and the deployable -O3 re-bench latency (from the
    prior's reservoir) where one was recorded. The per-kernel "did the
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
    o3 = _o3_reservoir_index(prior)
    for name in sorted(groups):
        _emit_variant_table(name, groups[name], prior, n_fail=fails.get(name, 0), o3=o3, top=args.top)


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


def _variant_key(s) -> tuple:
    """Hashable identity of one measured config: the full ``S_*`` signature plus
    the tunable-knob dict (sorted items) — the join key between a DB row and its
    -O3 reservoir sibling."""
    return (
        tuple(sorted((k, v) for k, v in s.s_features().items())),
        tuple(sorted((k, v) for k, v in s.knobs.items())),
    )


def _o3_reservoir_index(prior) -> dict[tuple, float]:
    """Deployable (-O3) latencies from the prior's reservoir, keyed by
    :func:`_variant_key`. Tuning re-benches every config within ``EMMY_O3_TOL``
    of the running -O1 best at ``-Xcicc -O3`` and feeds it to the prior as an
    ``H_opt=3`` row WITHOUT writing a ``perf`` row — so the reservoir, not the DB,
    is the only -O3 source (the same reasoning as
    :func:`diagnostics.golden_deploy_perf`)."""
    from emmy.compiler.pipeline.search.data import Sample  # noqa: PLC0415

    out: dict[tuple, float] = {}
    for knobs, us in getattr(prior, "_dataset", None) or []:
        s = Sample.from_prior_row(knobs, us)
        if int(s.all_knobs().get("H_opt", 0)) != 3:
            continue
        key = _variant_key(s)
        if key not in out or us < out[key]:
            out[key] = us
    return out


def _emit_variant_table(name: str, samples: list, prior, *, n_fail: int, o3: dict, top: int) -> None:
    """One kernel's leaderboard: measured leaf configs sorted by tune-ranking
    latency, the prior's pick marked, knobs in the canonical aligned columns
    (``tuning_knob_items`` — the same filtered view the ``run --bench`` kernel
    table renders). Non-leaf rows (partial-knob fork nodes) are dropped,
    mirroring ``diagnostics.reachability``; the ``-O3 us`` column appears only
    when the reservoir holds any -O3 row at all."""
    from emmy.compiler.pipeline.knob import tuning_knob_items  # noqa: PLC0415

    kmax = max(len(s.knobs) for s in samples)
    leaves = sorted((s for s in samples if len(s.knobs) == kmax), key=lambda s: s.latency_us)
    # Score in the deploy regime (``H_opt=3``) through ``Prior.pick`` — measured
    # -O3 evidence first, model argmin otherwise — so the marker shows the config
    # greedy ``compile`` / ``run`` would actually deploy, not just the model's
    # favourite (the DB rows themselves carry the tune's ``H_opt=1`` stamp).
    best_i, _ = prior.pick([{**s.all_knobs(), "H_opt": 3.0} for s in leaves])
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
    columns = [Col("rank", "r"), Col("us", "r")] + ([Col("-O3 us", "r")] if o3 else []) + [Col("pick"), *kcols]
    data = []
    for (r, s), kc in zip(shown, kcells, strict=True):
        row = [str(r), f"{s.latency_us:.1f}"]
        if o3:
            o3_us = o3.get(_variant_key(s))
            row.append(f"{o3_us:.1f}" if o3_us is not None else "—")
        data.append([*row, ("◄", _GREEN) if s is pick else "", *kc])
    for line in render_table(columns, data, indent="  "):
        logger.info(line)
    if hidden > 0:
        logger.info("  … %d more (--top 0 shows all)", hidden)
    if len(leaves) >= 2:
        # Judge the pick in the DEPLOY regime when -O3 re-benches cover it: the
        # -O1 ranking lane inverts against -O3 by up to 8× on the fm big-tile
        # family, so an -O1 ratio flags a near-best deployable pick as "misses
        # best". Falls back to the -O1 lane when the reservoir has no -O3 row
        # for the pick or no second row to compare against.
        pick_o3 = o3.get(_variant_key(pick)) if o3 else None
        o3_measured = [us for s in leaves if (us := o3.get(_variant_key(s))) is not None] if o3 else []
        if pick_o3 is not None and len(o3_measured) >= 2:
            ratio, lane = pick_o3 / min(o3_measured), "-O3 deploy latency"
        else:
            ratio, lane = pick.latency_us / leaves[0].latency_us, "tune-ranking latency"
        flag = "  <-- misses best" if ratio > 1.2 else ""
        logger.info("  pick: rank %d/%d, %.2fx of best (%s)%s", rank, len(leaves), ratio, lane, flag)


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
    site-form pick it realizes as); a ``("total", lead, cells)`` entry carries its cells
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
    ``found/golden`` knob columns (knob name in the header, value-only cells). ``entries``
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
    ``992_stamp_structural_features`` runs) + the golden tuning knobs. This is
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


def _emit_offline_eval(kernel_filter: str | None) -> None:
    """``eval offline`` body: the cold-start ``OfflinePrior`` (``search/golden_eval``
    is the golden-eval glue around it) — no online data, no GPU, no measurements.
    One streamed line per golden config with the golden's **rank** under the prior
    over the enumeration (the position the tuner's patience must reach) + per-knob
    ``found/golden`` (mismatches red), summarized as median + top-k coverage."""
    from statistics import median  # noqa: PLC0415

    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.compiler.pipeline.knob import tuning_knob_items  # noqa: PLC0415
    from emmy.compiler.pipeline.search.golden_eval import evaluate_record  # noqa: PLC0415

    configs = _golden_configs(kernel_filter)
    ranks: list[int] = []
    entries: list[tuple] = []  # ("row", lead_cells, gold, got) | ("err", name, message)
    for g in configs:
        gold = dict(tuning_knob_items(g.knobs))
        try:
            ctx = Context.from_target(g.compute_cap, gpu_name=g.gpu_name)  # the golden's own card, not the live host's
            got, rank, pool, _ = evaluate_record(g, ctx)
        except Exception as e:  # noqa: BLE001 — one shape's error shouldn't abort the report
            entries.append(("err", g.name, " ".join(f"{type(e).__name__}: {e}".split())[:100]))
            continue
        matched = sum(1 for k in gold if _knob_eq(k, gold[k], got))
        lead = [g.name, (f"{matched}/{len(gold)}", _ratio_color(matched, len(gold))), str(rank) if rank is not None else "?", str(pool)]
        entries.append(("row", lead, gold, got))
        if rank is not None:
            ranks.append(rank)
    cols = [Col("kernel"), Col("m/t"), Col("rank"), Col("pool")]
    _emit_golden_table(cols, entries, "knobs (found/golden; red = mismatch)")
    if ranks:
        n = len(ranks)
        cov = "  ".join(f"top{k}={sum(r < k for r in ranks)}/{n}" for k in (1, 10, 25, 50, 100))
        logger.info("")
        logger.info("  offline golden rank — median=%d  %s", int(median(ranks)), cov)


def _emit_online_eval(kernel_filter: str | None) -> None:
    """``eval online`` body: the online ``OnlinePrior`` on the golden configs —
    the golden's rank under the prior over the full enumeration (offline) followed
    by the greedy tile-pipeline pick vs golden (the real selection). Reads the
    prior JSON (``EMMY_ONLINE_FILE`` / ``--prior``; option-0 when none loaded)."""
    from emmy import config  # noqa: PLC0415
    from emmy.compiler.pipeline.search.prior import OnlinePrior, diagnostics  # noqa: PLC0415

    prior = OnlinePrior.load()
    logger.info("")
    if prior.fitted:
        logger.info(diagnostics.golden_prior_eval(prior, kernel_filter))
    else:
        logger.info("No fitted prior at %s — greedy falls to option-0 (run `emmy tune`).", config.online_path())

    configs = _golden_configs(kernel_filter)
    # Deployable (-O3) perf of the prior's pick vs golden, read from the reservoir (no
    # re-bench); empty when there's no tuned -O3 data (column shows '—').
    perf = diagnostics.golden_deploy_perf(prior, kernel_filter) if prior.fitted else {}
    _emit_prior_golden_check(configs, perf=perf)


def _emit_online_db_reachability(args) -> None:
    """``eval online --dataset db`` body: the prior's pick **reachability** over the
    tune DB's *measured* variants — per op structure, does the online prior's
    predicted-fastest config recover the measured-best leaf? The orthogonal counter
    to the golden views: it scores the same prior over the DB rows instead of the
    curated goldens. Reuses the diagnostics machinery (the prior's ``_dataset`` is
    irrelevant here — the groups come from the DB)."""
    from emmy import config  # noqa: PLC0415
    from emmy.compiler.pipeline.search.prior import diagnostics, load_prior  # noqa: PLC0415

    db_path = Path(args.db) if args.db else resolve_tune_db()
    if not db_path.exists():
        logger.error("no tune DB at %s — pass --db or run `emmy tune` first.", db_path)
        return
    # FallbackPrior: the online CatBoost when fitted, else the cold OfflinePrior — the same ranking compile/run use.
    prior = load_prior()
    # Group DB variants by their full S_* signature — the (sig → Samples) mapping
    # diagnostics.reachability scores.
    groups = Dataset.from_db(db_path, kernel=args.kernel).group_by_op()
    logger.info("")
    if not prior.fitted:
        logger.info("No fitted prior at %s — run `emmy tune`; the cold OfflinePrior ranks by D_* geometry only.", config.online_path())
    rr = diagnostics.reachability(prior, groups)
    if not rr:
        logger.info("No op structure has ≥2 measured leaf configs in the DB — nothing to score.")
        return
    ratios = [r[3] for r in rr]
    logger.info("[prior] pick reachability over DB variants — does the prior recover each op's measured best?")
    logger.info("  mean %.2fx  median %.2fx  worst %.2fx   (1.00 = optimum)", _mean(ratios), median(ratios), max(ratios))
    for label, best_us, pick_us, ratio, n in sorted(rr, key=lambda r: -r[3]):
        flag = "  <-- misses best" if ratio > 1.2 else ""
        logger.info("    %-26s  best %8.2fus  pick %8.2fus  (%.2fx, %d configs)%s", label, best_us, pick_us, ratio, n, flag)


def _emit_online_nodes(args) -> None:
    """``eval online --dataset nodes`` body: fork sibling regret (what following the
    prior's per-fork pick costs vs each fork's best-reachable latency, bucketed by
    the knob family the fork decides) plus leaf reachability / calibration over the
    tune DB's search-tree ``node`` store. The search-faithful counterpart to
    ``--dataset db`` (which only scores fully-decided leaf variants).

    Reported ONCE PER PRIOR HALF, explicitly labeled — the composite ``FallbackPrior``
    would answer with whichever half is active, mixing two distinct failure modes: the
    cold ``OfflinePrior`` decides what a cold sweep measures at all (its regret ⇒ fix
    the offline weights / features), while the online ``OnlinePrior`` owns deploys
    once trustworthy (its regret ⇒ a training-data / featurization problem). Splitting
    the blocks makes the diagnostic say WHERE the problem is."""
    from emmy import config  # noqa: PLC0415
    from emmy.compiler.pipeline.search.data.freeze import load_node_rows  # noqa: PLC0415
    from emmy.compiler.pipeline.search.prior import OfflinePrior, OnlinePrior, diagnostics  # noqa: PLC0415

    db_path = Path(args.db) if args.db else resolve_tune_db()
    if not db_path.exists():
        logger.error("no tune DB at %s — pass --db or run `emmy tune` first.", db_path)
        return
    # ``--db`` takes the live sqlite DB or a measurement freeze — load_node_rows sniffs.
    nodes = load_node_rows(db_path)
    online = OnlinePrior.load()
    calib = f"calibration={online.calibration:+.2f}" if online.calibration is not None else "calibration=n/a"
    halves: list[tuple[str, object]] = [
        ("=== offline prior (cold-start ranking — decides what a cold sweep measures) ===", OfflinePrior()),
        (f"=== online prior (CatBoost, {calib} — owns deploys once trustworthy) ===", online),
    ]
    for header, prior in halves:
        logger.info("")
        logger.info("%s", header)
        if not prior.fitted:
            logger.info("  not fitted (no checkpoint at %s) — run `emmy tune` to train it.", config.online_path())
            continue
        logger.info("%s", diagnostics.node_report(prior, nodes, kernel_filter=args.kernel))
        if args.blame or args.ablate:
            logger.info("")
            logger.info("%s", diagnostics.attribution_report(prior, nodes, kernel_filter=args.kernel, blame=args.blame, ablate=args.ablate))


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
    """The ``vs gold`` lead cell for one shape: ``pick_us/golden_us`` as ``N.NNx``
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
    """Canonicalize a resolved op's tuning knobs to bare family spelling (``TILE@a2`` →
    ``TILE``) so they compare/render against a golden YAML row, whose knobs are recorded
    bare. The schedule passes stamp the codec knobs axis-suffixed since the tile-IR
    rebuild — compared raw, ``got.get("TILE")`` never matched and the whole golden
    reproduction table showed ``-``/0-matches over perfectly good picks. First key wins
    on a family collision (a matmul golden's pick is single-node; a multi-node kernel
    has no single bare row to compare against a golden anyway)."""
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
    ``Golden reproduction — … prior: <path>`` banner (``eval online``); ``eval golden``
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
    """The offer audit — does each recorded golden realize in its recorded input pin regime?
    Re-compiles every golden's own snippet greedily under the golden-audit seam
    (``search/audit.audit_card``: deployable regime, the golden file's own card, no local tune
    evidence — the enumeration is static given shape+context, so no GPU bench is needed) and
    reads per-entry realizability off the verdict records:

      PIN-ONLY      the entry's knobs realize only when pinned exactly (``EMMY_KNOBS`` /
                    working-file proposal measurement), not from the realization's input regime alone.
                    Legal as a documented lever while an OFFERED sibling floors the shape.
      FALL-THROUGH  NO entry of the shape realizes: a deploy logs "no offered candidate
                    realizes any of them" and falls past the golden tier (the 4090
                    ``attention.hd512.s4096`` pathology: a 111 ms 0.03x kernel NaN-poisoning
                    the downstream accuracy check) — the defect this audit catches at record
                    time. Fix: record an offered deploy-floor sibling (re-tune in this input regime) or
                    close the enumeration gap.

    Each realization audits under its recorded input pins, so every regime's rows judge against
    their own enumeration. This is the OWN-SNIPPET view — an
    entry can realize here yet still drift inside a served model's fused graph (the 5090
    ``mlp_down.m4096`` split-K row on the epilogue-fused twin); the serving-matrix audit
    closes that side. Returns True when any shape falls through (``eval golden`` exits 1)."""
    import logging as _logging  # noqa: PLC0415

    from emmy.compiler.pipeline.search.audit import COMPILE_FAIL, audit_card  # noqa: PLC0415
    from emmy.compiler.pipeline.search.pins import pinned_knobs  # noqa: PLC0415

    def kstr(g) -> str:  # the entry's distinguishing knobs, empty families dropped
        return ",".join(f"{k}={v}" for k, v in g.knobs.items() if v not in ("", None))

    logger.info("")
    logger.info("Offer audit — do the recorded knobs realize in each input pin regime (own snippet, deployable regime)?")
    cards: dict[tuple, list] = {}
    for g in configs:
        cards.setdefault((g.gpu_name, tuple(g.compute_cap)), []).append(g)
    n_shapes = n_entries = n_pin = 0
    fell: list[str] = []
    # Silence the trace/compile chatter — at ERROR, not WARNING: the greedy tier's per-fork
    # drift warning is this audit's MEASUREMENT (re-reported as FALL-THROUGH below), not news.
    quiet = [_logging.getLogger(n) for n in ("emmy.compiler", "emmy.commands.trace")]
    prev = [lg.level for lg in quiet]
    for lg in quiet:
        lg.setLevel(_logging.ERROR)
    try:
        for (gpu_name, cap), card_cfgs in sorted(cards.items()):
            if len(cards) > 1:
                logger.info("  --- %s (sm_%d%d) ---", gpu_name, cap[0], cap[1])
            pin_sets = sorted({record.pins for record in card_cfgs}, key=repr)
            for pins in pin_sets:
                groups: dict[str, list] = {}
                for g in card_cfgs:
                    if g.pins == pins:
                        groups.setdefault(g.name, []).append(g)
                graphs: dict[str, object] = {}
                for name, sub in groups.items():
                    try:
                        graphs[name] = sub[0].target_program.copy()
                    except Exception as e:  # noqa: BLE001 — one shape's error shouldn't abort the audit
                        logger.info("  %-44s  ERR  %s", _realization_label(name, pins), " ".join(f"{type(e).__name__}: {e}".split())[:100])
                if not graphs:
                    continue
                with pinned_knobs(dict(pins)):
                    res = audit_card(graphs, gpu_name, cap, goldens=[record for record in card_cfgs if record.pins == pins])
                for name, sub in groups.items():
                    if name not in graphs:
                        continue  # trace error, already reported
                    recs = res.get(name, [])
                    fail = next((r for r in recs if r["verdict"] == COMPILE_FAIL), None)
                    if fail is not None:
                        logger.info("  %-44s  ERR  %s", _realization_label(name, pins), " ".join(str(fail.get("error", "")).split())[:100])
                        continue
                    n_shapes += 1
                    n_entries += len(sub)
                    key = sub[0].shape_key
                    hits = [r for r in recs if r["key"] == key]
                    if not hits:
                        logger.warning(
                            "  %-44s  NO-FORK  no fork of its own snippet keys %s — the golden tier was never "
                            "consulted at this shape (key drift; the serving-matrix audit is the deploy-side authority)",
                            _realization_label(name, pins),
                            key,
                        )
                        continue
                    floor = next((r for r in hits if r["verdict"] == "MATCH"), None)
                    for g in sub:
                        if any(g not in (r["unrealized"] or ()) for r in hits):
                            continue  # offered somewhere in its own snippet without additional winner pins
                        n_pin += 1
                        via = f"deploy floor: {floor['golden']} @ {floor['us']:g}us" if floor else "NO offered sibling"
                        logger.info(
                            "  %-44s  PIN-ONLY  %.1fus  %s  (%s)",
                            _realization_label(name, pins),
                            g.emmy_us,
                            kstr(g),
                            via,
                        )
                    if all(r["verdict"] == "DRIFT" for r in hits):
                        fell.append(_realization_label(name, pins))
                        logger.error(
                            "  %-44s  FALL-THROUGH  none of the shape's %d recorded entr%s realizes in its input regime — a deploy "
                            'logs "no offered candidate realizes any of them" and falls past the golden tier; record an '
                            "offered deploy-floor sibling or fix the enumeration",
                            _realization_label(name, pins),
                            len(sub),
                            "y" if len(sub) == 1 else "ies",
                        )
    finally:
        for lg, lv in zip(quiet, prev, strict=True):
            lg.setLevel(lv)
    logger.info("")
    if fell:
        logger.info(
            "  offer audit: %d shape(s) FALL THROUGH the golden floor (%s); %d/%d entries pin-only",
            len(fell),
            ", ".join(fell),
            n_pin,
            n_entries,
        )
    elif n_pin:
        logger.info(
            "  offer audit: %d/%d entries pin-only across %d shapes — every shape keeps an offered deploy floor", n_pin, n_entries, n_shapes
        )
    else:
        logger.info("  offer audit: all %d entries realize in their input regimes across %d shapes", n_entries, n_shapes)
    return bool(fell)


@dataclass(frozen=True)
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
