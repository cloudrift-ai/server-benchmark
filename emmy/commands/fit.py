"""``emmy fit`` — fit an offline-prior artifact and cross-validate it, writing a per-run
metrics file.

The fitter entry point: one pipeline, two orthogonal switches — ``--trainer``
(model class: the incumbent ``linear`` weights or a ``catboost`` ranker) × ``--data`` (training data).
Only ``--data golden`` exists today; the freeze summaries arrive with the measurement-freeze training work
and until then are rejected loudly. Both trainers write the same artifact shape, distinguished by its
``kind`` field, so either can be pointed at with ``EMMY_OFFLINE_FILE`` and A/B'd against the other.

A run writes ``<out>/metrics.json`` — the deterministic, diff-able record two fits are
compared by (same header inputs → identical content; the run dir name, not the file,
carries the timestamp) — and ``<out>/weights.json``, the full-train artifact in the
shipped ``offline_weights.json`` format. The metrics layout (``full_train`` +
a ``cv`` holdout/train/gap block, both carrying ``prior/report.py`` summaries) is documented on
:mod:`emmy.compiler.pipeline.search.prior.fit.cv`, which owns all the fold machinery;
the run itself is :func:`~emmy.compiler.pipeline.search.prior.fit.run.run_fit`. This
module owns what ``pipeline/`` must not import: the snippet-tracing golden group builder
(:func:`build_golden_groups`) plus the CLI, the trainer wiring, and the file writing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from emmy import config, storage
from emmy.compiler.context import Context
from emmy.compiler.pipeline.search import features
from emmy.compiler.pipeline.search.data.group import DEFAULT_FEATURES, GoldenGroup, feature_view, pack_features
from emmy.compiler.pipeline.search.golden import GOLDEN_RECORDS, GoldenRecord
from emmy.compiler.pipeline.search.golden_eval import enumerate_graph
from emmy.compiler.pipeline.search.pool import DEFAULT_SAMPLE, PoolSample
from emmy.compiler.pipeline.search.prior.fit import catboost as fit_catboost
from emmy.compiler.pipeline.search.prior.fit import cv as fit_cv
from emmy.compiler.pipeline.search.prior.fit import linear as fit_linear
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
        help="catboost only: sampled negatives per pool per round (every golden matched into the pool is a positive).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=fit_catboost.DEFAULT_ROUNDS,
        help="catboost only: fit rounds — the first draws negatives uniformly, each further one mines hard negatives.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pool-sample",
        type=int,
        default=DEFAULT_SAMPLE,
        help=f"Candidates drawn per pool during enumeration (default {DEFAULT_SAMPLE}; 0 enumerates every row). "
        "Recorded in the metrics header and the artifact provenance - two fits are comparable only when it matches.",
    )
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


@dataclass
class _Pool:
    """One packed candidate pool and every golden found in it, before it becomes a :class:`Group`.

    ``namer`` is the first golden the builder found in this pool — it supplies the group's key, name,
    card and fold group. Deliberately not "the first record considered": a record can be grouped onto this
    pool and then fail to match a row in it, and naming a group after a golden the same run reports as
    skipped would put a rank in ``metrics.json`` under a name nothing pinned."""

    namer: GoldenRecord
    tier: str
    packed: tuple[tuple[str, ...], np.ndarray, bool]
    total: int
    goldens: list[int]


def _pool_identity(g, tier: str, packed: tuple[tuple[str, ...], np.ndarray, bool]) -> tuple:
    """A candidate pool's identity: everything about it except which golden pinned which row.

    Two enumerations belong in one group when this matches — the featurized pool is then byte-identical, so
    one enumeration's row index addresses the same row the other's does, and the goldens behind them are
    several verified answers to one question. Deciding it from the packed pool rather than from a key is what
    catches the same program recorded twice, which :attr:`~..compiler.pipeline.search.golden.GoldenRecord.pool_group`
    cannot see.

    The identity fields ride along with the matrix digest because they decide things the matrix does not: the
    weight set (``dynamic``), the fold group (``shape``) and the report axes. Requiring them to agree can only
    hold two pools apart, never fuse two that differ."""
    names, matrix, dynamic = packed
    return (g.gpu_name, tier, _shape_group(g), dynamic, names, hashlib.blake2b(matrix, digest_size=16).digest())


def _pool_bucket(g) -> tuple:
    """The bucket a golden's candidate pool can only be shared inside: its card and its pins.

    A SUPERSET of pool identity by construction - both fields are inputs to the scheduler's pool
    key - so two goldens that could share a pool always land in one bucket. That direction is the
    one that matters: an over-broad bucket costs a few extra retained rows, while a too-narrow one
    would hand two goldens over one pool different keep-sets, retain different rows, and stop them
    merging into a single training group."""
    return (g.gpu_name, tuple(g.compute_cap), g.pin_key)


def _keep_sets(records) -> dict[tuple, frozenset]:
    """Each bucket's union of recorded ``tile_signature`` values - the rows a sample may not drop.

    Membership must survive sampling EXACTLY: the builder locates a golden by scanning its pool for
    that signature and drops the golden when it misses, and ``eval golden`` reads the same miss as a
    pin or dtype mismatch. A draw that could lose the row would turn a real defect signal into
    noise."""
    out: dict[tuple, set] = {}
    for g in records:
        if g.knobs:
            out.setdefault(_pool_bucket(g), set()).add(features.tile_signature(g.knobs))
    return {k: frozenset(v) for k, v in out.items()}


def build_golden_groups(
    features_spec: str = DEFAULT_FEATURES, *, sample: int = 0, seed: int = 0, kernel: str | None = None
) -> tuple[list[GoldenGroup], list[tuple[str, str, str]]]:
    """Enumerate each embedded golden program, pin its recorded row, and
    featurize every row, as :class:`Group` records (name, tier, card, pinned rows,
    per-row features filtered through the ``features_spec`` view; ``key`` is ``"<gpu>/<name>"``,
    suffixed ``#2``, ``#3``, … when one name opens several distinct pools). The second return is
    the goldens that did NOT become groups, as ``(gpu, name, reason)`` — enumeration
    failures plus the kinds this fitter has no group builder for
    (:data:`fit_cv.OUT_OF_SCOPE`) — so metrics can count every recorded golden.

    **A group is a candidate pool, not a golden.** Several goldens can land on one pool — the same shape
    recorded under two names, or a name recorded twice — and they then share ONE group, each contributing a
    row to its golden set. Which goldens share a pool is read off the records (``GoldenRecord.pool_group``)
    BEFORE anything is enumerated, so each pool is enumerated, featurized and packed once, then folded by
    :func:`_pool_identity`; every group is built knowing all of its goldens. This logs how many goldens merged, and the caller records
    groups against positives in the metrics header.

    Matmul goldens enumerate via ``golden_eval.enumerate_graph`` — the SAME gate-narrowed
    pool ``eval prior --dataset golden`` and the greedy deploy rank over (fp32 → thread tier,
    fp16/bf16 → warp tier; the block-DAG rework moved the scalar↔warp choice to a
    structural fork, so a real fp16 matmul ranks within the warp tier alone, no
    scalar rows in the pool). A dynamic (``.dynM``) golden enumerates the pool of its
    static counterpart at the hint size and featurizes with the symbolic-axis stamp
    (its own weight set). Reduce / pointwise goldens trace their snippet and capture the restored
    schedule fork's rows (``_snippet_rows``); a regime the live tree doesn't fork
    (pointwise) reports un-enumerable rather than reconstructing a search space that no longer exists.

    ``kernel`` keeps only goldens whose name contains it — a narrowing VIEW of the corpus, for iterating on
    one kernel without paying for the rest. Each retained golden's rank is unchanged by it: the keep-sets are
    still computed over the whole corpus, so a pool retains the same rows under the same draw. What does change
    is which goldens MERGE — a dropped sibling is one fewer verified row in the pool it shared — so a filtered
    run's group and positive counts are its own, and only an unfiltered run compares against a fit.

    ``sample`` draws that many candidates per pool DURING enumeration (0 enumerates every row).
    The draw is a reservoir over the schedule walk's leaf stream — a pure function of that stream
    and ``(sample, seed)`` that never reads a row — so two goldens over one pool retain identical
    rows and still merge into one group; every recorded config survives it whatever the draw picks
    (:func:`_keep_sets`), so a golden that misses its pool still means what it always meant - a pin
    or dtype mismatch.

    Each golden is reconstructed under its OWN card's context
    (``Context.from_target(cap, gpu_name=…)``, mirroring ``Sample.from_golden``):
    the multi-GPU golden set spans cards that differ in compute capability AND in
    SM count at the same cap (RTX 5090 = 170 vs RTX PRO 6000 = 188 vs RTX 4090 =
    128 SMs, the latter at sm_89), so both the candidate enumeration (cp.async /
    TMA tiers gate on cap) and the ``H_*`` / ``D_*`` occupancy features must use the
    recording card's regime — not one global cap — for the rank objective to match
    the deployed per-card featurization."""
    keep = feature_view(features_spec)
    groups: list[GoldenGroup] = []
    skipped: list[tuple[str, str, str]] = []
    key_counts: dict[str, int] = {}
    matched = 0
    pools: dict[tuple, _Pool] = {}  # packed-pool identity -> the pool and every golden found in it
    # ONE Context per card: the per-card facts are identical across its goldens. Cross-record
    # enumeration dedup is the GROUPING's job — ``pool_group`` composes the target kernels'
    # identity keys, so the std / parity siblings of one shape (and cross-session re-recordings)
    # land in one group and the dataset build pays each pool once; the fm-pinned enumeration
    # keys apart on its own, since the record's pin regime is a group-key term.
    ctxs: dict[tuple, Context] = {}
    # The keep-sets are precomputed BEFORE the loop because a bucket's obligation spans the whole
    # corpus: the pool a golden opens may also carry a later golden's recorded row.
    keeps = _keep_sets(GOLDEN_RECORDS) if sample > 0 else {}
    # Group the records by the pool each will enumerate (:attr:`GoldenRecord.pool_group`) before touching
    # the scheduler, so each enumeration is paid once. Insertion order is corpus order, so the groups
    # come out in the order they always did.
    by_pool: dict[tuple, list] = defaultdict(list)
    for g in GOLDEN_RECORDS:
        if kernel is None or kernel in g.name:
            by_pool[g.pool_group].append(g)

    for members in by_pool.values():
        g = members[0]  # the pool is a property of the KEY; every member spells it identically
        card = (tuple(g.compute_cap), g.gpu_name)
        ctx = ctxs.get(card)
        if ctx is None:
            ctx = ctxs[card] = Context.from_target(tuple(g.compute_cap), gpu_name=g.gpu_name)
        base = {**ctx.features(), **g.structural_features}
        from emmy.compiler.pipeline.search.pins import pinned_knobs  # noqa: PLC0415

        # The sample rides a REPLACED Context; the pool stamp keys on the sample too, so a sampled
        # enumeration can never be mistaken for a live one
        # can never be served to a live compile.
        keep_set = keeps.get(_pool_bucket(g), frozenset())
        enum_ctx = ctx if sample <= 0 else replace(ctx, pool_sample=PoolSample(sample, seed, keep_set))
        with pinned_knobs(g.pin_map):
            candidates = enumerate_graph(g.target_program.copy(), enum_ctx)
        rows = candidates.rows
        tier = "dyn" if g.dynamic else (g.shape_key.kind or ("warp" if g.shape_key.is_warp else "thread"))
        if not rows:
            logger.info("  !! %s: nothing enumerated — skipping", g.name)
            skipped.extend((m.gpu_name, m.name, "nothing enumerated") for m in members)
            continue
        # Each member locates its OWN recorded config in the shared pool, by schema-agnostic structural
        # signature (free-axis slots + reduce decomp + atom) — the candidate rows use the native
        # ``MOVE@element`` keys while the golden YAML records legacy GEMM-letter keys, so comparing
        # key-value tuples directly never matches. A member that misses is dropped on its own; the pool
        # still stands for the members that hit.
        goldens, namer = [], None
        for m in members:
            want = features.tile_signature(m.knobs)
            gidx = next((i for i, r in enumerate(rows) if features.tile_signature(r) == want), None)
            if gidx is None:
                logger.info("  !! %s: golden not in %d candidates — skipping", m.name, len(rows))
                skipped.append((m.gpu_name, m.name, f"golden not in {len(rows)} candidates"))
            else:
                goldens.append(gidx)
                namer = namer or m  # the pool is named after a golden that is actually IN it
        if not goldens:
            continue
        matched += len(goldens)
        # The feature view (default ``DEFAULT_FEATURES``: ``D_*`` geometry/occupancy plus
        # ``MMA_tier`` — see its rationale in ``search/data/group.py``) filters here, before
        # the pool is packed, so the trained-under view is exactly what the Group stores.
        # ``feature_view`` keeps the routing features whatever the spec says, so a narrower
        # ``--features`` cannot silently misroute a symbolic-axis pool.
        feats = [{k: v for k, v in features.knob_features({**base, **r}).items() if keep(k)} for r in rows]
        packed = pack_features(feats)
        # Second stage: two enumerations can still land on one pool — the same program recorded in different
        # sessions keys apart above but packs identically here. Fold those together, so a pool is one group
        # however many times it was recorded.
        pool = pools.get(identity := _pool_identity(namer, tier, packed))
        if pool is None:
            pools[identity] = _Pool(namer, tier, packed, candidates.total, goldens)
        else:
            pool.goldens.extend(goldens)

    # Every pool now knows every golden in it, so each becomes ONE group whose labels are final at
    # construction. The ``#N`` suffix keeps ``Group.key`` unique (``cv.run_folds`` keys its train accumulator
    # on it) and is spent per POOL in first-appearance order, so goldens that merged never claim one.
    for pool in pools.values():
        g = pool.namer
        key_str = f"{g.gpu_name}/{g.name}"
        key_counts[key_str] = n = key_counts.get(key_str, 0) + 1
        groups.append(
            GoldenGroup.over(
                key_str if n == 1 else f"{key_str}#{n}",
                g.name,
                pool.tier,
                g.gpu_name,
                _shape_group(g),
                pool.packed,
                pool.goldens,
                pool.total,
            )
        )
    logger.info(
        "  %d matched goldens over %d candidate pools (%d beyond one golden per pool)",
        matched,
        len(groups),
        matched - len(groups),
    )
    return groups, skipped


def _write_artifact(path: Path, model, provenance: dict) -> None:
    """Write one weights artifact — the JSON, plus the model's binary sidecar when it has one.

    The sidecar is named after the JSON (``weights.json`` → ``weights.cbm``) and recorded RELATIVE in the JSON,
    so the pair travels together: copied into a run directory, rsynced to a tuning box, or checked in beside the
    shipped weights. Naming it after its own JSON is what lets two artifacts share a directory without one
    silently overwriting the other's model.

    Which classes have a sidecar is the MODEL's business, not this function's: it asks for ``model_file`` in the
    artifact and writes ``blob`` only if the model put the key there. A linear artifact is self-contained and
    simply does not."""
    artifact = model.to_artifact(provenance=provenance, model_file=f"{path.stem}.cbm")
    storage.write_json(path, artifact, indent=2)
    if "model_file" in artifact:
        (path.parent / artifact["model_file"]).write_bytes(model.blob)


def _repo_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True, timeout=10)
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 — a fit outside a git checkout still gets a metrics file
        return "unknown"


def _linear_trainers(args, names: list[str]):
    """The linear summary's trainer pair and the hyperparameters its metrics header records.

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
    """The tree summary's trainer and its recorded hyperparameters. ONE trainer serves both the shippable model and
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
# tree re-derives every one of them from the columns ``fit_catboost.TREE_FEATURES`` keeps. ``--features`` overrides
# either, which is how the two views are compared on one model class.
TRAINERS = {
    "linear": (_linear_trainers, DEFAULT_FEATURES),
    "catboost": (_catboost_trainers, fit_catboost.TREE_FEATURES),
}


def _log_cells(metrics: dict) -> None:
    """The run's summaries as one line per card per split — the same rows ``metrics.json`` carries, so what
    scrolls past and what is written down cannot disagree. Indexes rather than defends: every key read here
    is one the same process wrote a few lines earlier, so a shape mismatch should raise rather than render
    a line full of ``None``."""
    full, cv = metrics["full_train"], metrics["cv"]
    train = {c["axes"]["gpu"]: c["metrics"]["rank"]["median"] for c in cv.get("summaries", []) if c["axes"]["cv_split"] == "train"}
    gap = cv.get("gap", {})
    for summary in full["summaries"] + [c for c in cv.get("summaries", []) if c["axes"]["cv_split"] == "holdout"]:
        cv_split, gpu, rank = summary["axes"]["cv_split"], summary["axes"]["gpu"], summary["metrics"]["rank"]
        line = f"{cv_split:<11} {gpu:<34} n={summary['groups']:<3} median={rank['median']} (optimistic {rank['median_optimistic']})"
        if cv_split == "full_train":
            skipped = full["skipped"][gpu]
            line += f" unranked={skipped['unranked']} out_of_scope={skipped['out_of_scope']}"
        else:
            line += f" train={train.get(gpu)} gap={gap.get(gpu)}"
        logger.info("%s", line)
    # A card every one of whose goldens was skipped has no summary at all — say so rather than let it vanish.
    for gpu, skipped in full["skipped"].items():
        if gpu not in {c["axes"]["gpu"] for c in full["summaries"]}:
            logger.info("%-11s %-34s no ranked groups  unranked=%d out_of_scope=%d", "full_train", gpu, *skipped.values())
    for f, why in cv.get("fold_detail", {}).get("excluded", {}).items():
        logger.info("cv fold %s EXCLUDED: %s", f, why)


def handle_fit(args) -> None:
    from emmy.compiler.pipeline.search.prior.offline import _DEFAULT_FILE  # noqa: PLC0415

    if args.data != "golden":
        raise SystemExit(
            f"--data {args.data} is not yet supported — only 'golden' exists (the freeze summaries land with the training-data work)"
        )

    out_dir = Path(args.out) if args.out else Path("_tune/fits") / f"{time.strftime('%Y%m%d-%H%M%S')}-{args.trainer}-{args.data}"
    out_dir.mkdir(parents=True, exist_ok=True)

    make_trainers, default_view = TRAINERS[args.trainer]
    view = args.features or default_view

    logger.info("Building golden dataset (each golden under its own card's context) ...")
    groups, skipped = build_golden_groups(view, sample=args.pool_sample, seed=args.seed)
    names = sorted({n for c in groups for n in c.feat_names})
    n_dyn = sum(1 for c in groups if c.dynamic)
    # A group is a candidate pool and may carry several verified rows, so the group count alone no longer says
    # how much supervision the fit saw — both numbers travel together, into the header and the provenance.
    # ``merged`` is the positives beyond one per pool; the builder logs the record-level count, which also
    # counts a golden recorded twice at one config (it pins a row already pinned, so it adds no positive).
    positives = sum(len(c.golden_ids) for c in groups)
    logger.info(
        "  %d static + %d dynamic golden groups (%d positives, %d merged), %d D_* features, %d skipped",
        len(groups) - n_dyn,
        n_dyn,
        positives,
        positives - len(groups),
        len(names),
        len(skipped),
    )

    trainer, fold_trainer, trainer_params, incumbent = make_trainers(args, names)
    header = {
        "trainer": args.trainer,
        "data": args.data,
        "seed": args.seed,
        "feat_ver": features.FEATURIZER_VERSION,
        "features": view,
        "folds": args.folds,
        # Two fits are comparable only when they drew the same way: a sampled fit's ranks are RAW
        # ranks within the draw, and ``per_golden`` prints the true pool size beside them.
        "pool_sample": args.pool_sample,
        # A group IS a candidate pool; positives are the verified rows marked in it, and a pool can hold more
        # than one. Recorded so a metrics file whose group count dropped against an earlier fit says why,
        # instead of looking like lost data.
        "groups": {"total": len(groups), "positives": positives, "merged": positives - len(groups)},
        "repo_commit": _repo_commit(),
        "trainer_params": trainer_params,
    }
    import datetime  # noqa: PLC0415

    metrics, fit = run_fit(groups, skipped, trainer=trainer, fold_trainer=fold_trainer, folds=args.folds, header=header)

    model, notes = fit.model, fit.notes
    # Shipping policy, and the reason ``run_fit`` hands back a fit rather than an artifact: a LINEAR fit with no
    # dynamic groups would otherwise ship with no dynamic weight set at all, so carry the incumbent's forward —
    # loudly, in the provenance notes, never silently. The tree model has no second weight set to be missing.
    if isinstance(model, LinearModel) and model.weights_dynamic is None:
        # ``is not None``, not truthiness: an incumbent that legitimately pruned every dynamic
        # coordinate carries an EMPTY set, and that is still its answer, not a missing one.
        carried = incumbent.weights_dynamic if incumbent.weights_dynamic is not None else model.weights
        source = "incumbent" if incumbent.weights_dynamic is not None else "the static fit"
        model = replace(model, weights_dynamic=carried)
        notes = f"{notes}; dynamic set carried from {source}"
    provenance = {
        "fitted": datetime.date.today().isoformat(),
        "script": "emmy fit",
        "args": {"trainer": args.trainer, "seed": args.seed, **trainer_params},
        "features": view,
        "pool_sample": args.pool_sample,
        "groups": {"static": len(groups) - n_dyn, "dynamic": n_dyn},
        "positives": positives,
        "notes": notes,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    _write_artifact(out_dir / "weights.json", model, provenance)
    if args.artifact is not None:
        artifact_path = Path(args.artifact) if args.artifact else _DEFAULT_FILE
        _write_artifact(artifact_path, model, provenance)
        logger.info("wrote %s", artifact_path)

    _log_cells(metrics)
    logger.info("wrote %s", out_dir / "metrics.json")
