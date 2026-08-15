"""Offline diagnostics for the online tuning prior — "can the prior actually
reach the best configs?", with no GPU.

``emmy tune`` with no model / ``--code`` refits the prior on its persisted
reservoir dataset and prints :func:`report`: a dataset summary, per-op **pick
reachability** (does the prior's predicted-fastest config over an op's *measured*
configs recover the measured-best leaf?), an in-sample ranking calibration, and
coverage of the golden matmul configs. Reachability is the optimistic upper bound
on greedy quality — if the prior can't even rank an op's own benched leaves to the
top, greedy (which descends fork-by-fork) can't either.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from emmy.compiler.pipeline.search.data import Dataset, ShapeKey


def _n_tunable(knobs: dict) -> int:
    return sum(1 for k in knobs if not k.startswith(("S_", "H_")))


def _matmul_sig(d: dict) -> bool:
    """Histogram heuristic for "this op group is a matmul": a product feeding a
    reduce-add over ≥2 distinct inputs. ``S_n_mma`` is NOT usable as the marker:
    the stamp pass runs at fusion end, before the tile tier emits ``Mma`` stmts,
    so it is 0.0 on every stamped row — gating on it made golden coverage
    permanently empty and dropped every fp16 golden from the rank/deploy joins
    (see ``ShapeKey.from_s_features``)."""
    return bool(d.get("S_reduce_add", 0) and d.get("S_pw_multiply", 0) and d.get("S_n_distinct_input", 0) >= 2)


def _op_label(sig: tuple) -> str:
    d = dict(sig)
    if _matmul_sig(d):
        kind = "matmul"
    elif d.get("S_reduce_add", 0) or d.get("S_reduce_max", 0):
        kind = "reduce"
    else:
        kind = "pointwise"
    free, red = int(d.get("S_ext_free_max", 0)), int(d.get("S_ext_reduce_max", 0))
    return f"{kind:9} free={free}" + (f" red={red}" if red else "")


def reachability(prior, groups: dict) -> list[tuple]:
    """Per op: ``(label, best_us, pick_us, ratio, n_leaf_configs)`` — the config the
    prior predicts fastest (``mean_score`` argmin) over the op's measured *leaf*
    configs vs the measured best (ratio = the picked config's latency / the best's;
    1.0 = recovers the optimum). ``groups`` maps an ``S_*`` signature to its
    :class:`Sample`s (from :meth:`Dataset.group_by_op`)."""
    out = []
    for sig, grp in groups.items():
        kmax = max(_n_tunable(s.all_knobs()) for s in grp)
        leaves = [s for s in grp if _n_tunable(s.all_knobs()) == kmax]
        if len(leaves) < 2:
            continue
        best_us = min(s.latency_us for s in leaves)  # labels are latency µs — lower is better
        pick_us = min(leaves, key=lambda s: prior.mean_score(s.all_knobs())).latency_us
        out.append((_op_label(sig), best_us, pick_us, pick_us / best_us, len(leaves)))
    return out


def _calibration(prior, groups: dict) -> float | None:
    """Median per-op Spearman ρ between the prior's predicted latency and the
    measured latency over each op's leaf configs (both µs → ρ near ``+1`` when
    well-calibrated)."""
    from scipy.stats import spearmanr  # noqa: PLC0415

    rhos = []
    for grp in groups.values():
        kmax = max(_n_tunable(s.all_knobs()) for s in grp)
        leaves = [s for s in grp if _n_tunable(s.all_knobs()) == kmax]
        if len(leaves) < 3:
            continue
        rho = spearmanr([prior.mean_score(s.all_knobs()) for s in leaves], [s.latency_us for s in leaves]).statistic
        if not math.isnan(rho):
            rhos.append(rho)
    return statistics.median(rhos) if rhos else None


def _golden_coverage(groups: dict) -> tuple[int, int]:
    """How many golden matmul **shapes** have measured data in the dataset, matched by
    :class:`ShapeKey` (free-dim product, reduce extent, dtype flag — so an fp32
    square and its ``.fp16`` twin are counted separately). Counts *distinct shape
    keys*, not per-config rows, so multiple knob sets for one shape — and the same
    shape recurring across per-GPU golden files (``ShapeKey`` is GPU-blind) — count
    once."""
    from emmy.compiler.pipeline.search.golden import GOLDEN_RECORDS  # noqa: PLC0415

    have = set()
    for sig in groups:
        d = dict(sig)
        if _matmul_sig(d):
            have.add(ShapeKey.from_s_features(d))
    golden_keys = {g.shape_key for g in GOLDEN_RECORDS if g.is_matmul}
    covered = sum(1 for k in golden_keys if k in have)
    return covered, len(golden_keys)


def golden_prior_eval(prior, kernel_filter: str | None = None) -> str:
    """Per golden matmul, the golden's **rank under the prior** over the shape's
    full (gated) enumeration — the realistic "would greedy-with-prior pick the
    golden?" test (greedy picks the prior's predicted-fastest config across the
    enumeration, not just the benched leaves that :func:`reachability` scores).
    The ``S_*`` shape features
    come from the dataset's matching op group (so only shapes with tuned data are
    scored); ``H_*`` is the deployable compile regime (``Context.features``,
    ``H_opt=3``) the greedy ``compile`` / ``run`` actually queries with.
    ``kernel_filter`` restricts to golden configs whose name contains it.

    The rank is a **model diagnostic, not a deploy prediction**: the enumeration
    rows carry only the planner's tunables, while the rows the model trained on
    (and the leaves greedy scores late in the descent) also carry decision /
    transport stamps (``TMA``, ``PIPELINE_STAGES``, …) — absent keys are NaN
    ("not decided") to CatBoost, so the two surfaces can disagree (the 2026-06-12
    sweep's finding 6). Deploy reality is ``Prior.pick`` (measured -O3 evidence
    first); the faithful deploy check is ``eval golden``'s real greedy compile."""
    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.compiler.pipeline.search import golden_eval  # noqa: PLC0415
    from emmy.compiler.pipeline.search.golden import goldens_for_live_gpu  # noqa: PLC0415

    GOLDEN_RECORDS = goldens_for_live_gpu()  # live card only — see golden_deploy_perf

    # Index the matmul op groups by ShapeKey (free-dim product, reduce extent,
    # dtype flag — both sides built by the ShapeKey constructors, so the fp32/fp16
    # twins never merge) so each golden shape maps to the S_* signature it was
    # tuned under.
    index: dict[ShapeKey, dict] = {}
    for sig in Dataset.from_prior(prior).group_by_op():
        d = dict(sig)
        if not _matmul_sig(d):
            continue
        index.setdefault(ShapeKey.from_s_features(d), {k: v for k, v in d.items() if k.startswith("S_")})

    rows, skipped = [], []
    lines = ["[prior] golden selection — golden's rank under the prior over the full gated enumeration:"]
    for g in GOLDEN_RECORDS:
        if not g.is_matmul:
            continue
        if kernel_filter and kernel_filter not in g.name:
            continue
        s_feats = index.get(g.shape_key)
        if s_feats is None:
            # A silent skip here hid the fp16 lockout in the 2026-06-12 sweep —
            # every unjoinable shape gets a per-shape line instead.
            skipped.append((g.name, "no tuned rows for this shape in the prior dataset"))
            continue
        ctx = Context.from_target(g.compute_cap, gpu_name=g.gpu_name)  # the golden's own card, not the live host's
        base = {**ctx.features(), **s_feats}
        # ``evaluate_record`` ranks by descending score; the prior predicts latency
        # (lower = better), so negate to rank the predicted-fastest config first.
        _, rank, pool, _ = golden_eval.evaluate_record(g, ctx, scorer=lambda r, b=base: -prior.mean_score({**b, **r}))
        if rank is None:
            skipped.append((g.name, f"recorded knobs not in the enumeration ({pool} rows) — pin/dtype mismatch?"))
            continue
        rows.append((g.name, rank, pool))
    for name, rank, pool in sorted(rows, key=lambda t: -t[1]):
        lines.append(f"    {name:26}  rank {rank:5}/{pool}")
    for name, why in skipped:
        lines.append(f"    {name:26}  SKIPPED: {why}")
    if rows:
        ranks = [r for _, r, _ in rows]
        n = len(ranks)
        cov = "  ".join(f"top{k}={sum(r < k for r in ranks)}/{n}" for k in (1, 10, 25, 50))
        lines.append(f"  median rank={sorted(ranks)[n // 2]}  {cov}  (over {n} golden shapes with tuned data)")
    elif not skipped:
        lines.append("  no golden shapes have tuned data in the dataset yet")
    return "\n".join(lines)


def golden_deploy_perf(prior, kernel_filter: str | None = None) -> dict[str, float]:
    """Per golden shape, ``pick_us / golden_us`` — the deployable (-O3) latency of the
    prior's predicted-best **measured** config over the golden's recorded latency, read
    from the prior's reservoir with **no re-bench**.

    Tuning re-benches every winner at -O3 (``H_opt=3``) and feeds it to the prior, so
    each tuned shape's best config has a deployable row in the reservoir. For each
    golden shape we take the op group's ``H_opt=3`` rows, pick the one ``Prior.pick``
    deploys (measured -O3 evidence first, model argmin otherwise — the same selection
    greedy ``compile`` / ``run`` make), and divide its measured latency by the golden's
    recorded ``emmy_us`` (also -O3 → same regime, so the ratio is a real
    deployable speed comparison; <1.0 = the prior's pick is faster than golden). Shapes
    with no -O3 reservoir row are omitted (the caller renders ``—``). The reservoir is
    used rather than the raw ``perf`` table because only it carries the ``H_*`` regime
    columns needed to isolate the deployable measurements.

    Goldens are scoped to the live card (:func:`goldens_for_live_gpu`) so a multi-GPU
    goldens dir doesn't make a name's per-card entries collide on the GPU-blind
    ``ShapeKey`` (e.g. RTX 5090 / RTX PRO 6000 both ``(12, 0)``)."""
    from emmy.compiler.pipeline.search.golden import (  # noqa: PLC0415
        fast_math_knobs,
        goldens_for_live_gpu,
        precision_trading_pins,
    )

    GOLDEN_RECORDS = goldens_for_live_gpu()

    # Deployable (-O3) measured rows per matmul op group, indexed by ShapeKey.
    # An fp32 square and its ``.fp16`` twin share (free_prod, reduce), so the key's
    # dtype flag is what keeps them apart — ``ShapeKey.from_s_features`` derives it
    # from ``S_dtype_f32`` (see its docstring for why ``S_n_mma`` can't be the key).
    index: dict[ShapeKey, list] = {}
    for sig, samples in Dataset.from_prior(prior).group_by_op().items():
        d = dict(sig)
        if not _matmul_sig(d):
            continue
        o3 = [s for s in samples if int(s.all_knobs().get("H_opt", 0)) == 3]
        if not o3:
            continue
        index.setdefault(ShapeKey.from_s_features(d), []).extend(o3)

    out: dict[str, float] = {}
    for g in GOLDEN_RECORDS:
        if not g.is_matmul or not g.emmy_us:
            continue
        if kernel_filter and kernel_filter not in g.name:
            continue
        leaves = index.get(g.shape_key)
        if not leaves:
            continue
        best_i, _ = prior.pick([s.all_knobs() for s in leaves])
        # Within-regime comparison (the golden.py convention: a shape's fast-math entry sits
        # BESIDE its standard one, and each regime is judged against its own): a gate-off pick
        # must not be measured against an [fm] golden it cannot reach — and vice versa. Skip
        # cross-regime pairs; the pick's regime derives from its knobs like the golden's.
        if fast_math_knobs(leaves[best_i].knobs) != precision_trading_pins(g.pin_map):
            continue
        ratio = leaves[best_i].latency_us / g.emmy_us
        # A shape may record several parity entries under one name — compare against the BEST
        # (fastest) recorded golden of the pick's regime, not whichever entry iterates last
        # (max ratio = min emmy_us).
        out[g.name] = max(out.get(g.name, ratio), ratio)
    return out


def report(prior) -> str:
    """The full offline diagnostics block for a (re)fit prior."""
    dataset = prior._dataset
    groups = Dataset.from_prior(prior).group_by_op()
    lines = [f"[prior] dataset: {len(dataset)} rows, {len(groups)} op-structures, fitted={prior.fitted}"]
    if not prior.fitted:
        lines.append("  no model — dataset below min_rows; run `emmy tune <model>` to gather more")
        return "\n".join(lines)

    rr = reachability(prior, groups)
    if rr:
        ratios = [r[3] for r in rr]
        lines.append("[prior] pick reachability — does the prior's predicted-fastest config recover each op's measured best?")
        agg = f"mean {statistics.mean(ratios):.2f}x  median {statistics.median(ratios):.2f}x  worst {max(ratios):.2f}x"
        lines.append(f"  {agg}   (1.00 = optimum)")
        for label, best_us, pick_us, ratio, n in sorted(rr, key=lambda r: -r[3]):
            flag = "  <-- misses best" if ratio > 1.2 else ""
            lines.append(f"    {label:26}  best {best_us:8.2f}us  pick {pick_us:8.2f}us  ({ratio:.2f}x, {n} configs){flag}")
    calib = _calibration(prior, groups)
    if calib is not None:
        lines.append(f"[prior] ranking calibration (median per-op Spearman): {calib:+.2f}")

    covered, total = _golden_coverage(groups)
    lines.append(f"[prior] golden coverage: {covered}/{total} golden matmul shapes have data in the dataset")
    if covered == 0:
        lines.append("  none yet — tune a working golden file (`emmy tune --golden-file PATH`) to validate against them")
    return "\n".join(lines)


# Sentinel for "key absent" in the fork-family delta — distinct from every real knob value.
_ABSENT = object()


def _fork_family(parent, sibs) -> str:
    """The knob FAMILY a fork decides — the stable semantic notion of tree level (raw
    ``depth`` is rule-step distance from the sentinel root: it renumbers whenever a
    pass is added, and the live store shows forks only at depths 4–8/11–15). Primary:
    the tunable keys the children carry beyond (or differing from) the ``parent``
    row's; fallback when the parent row isn't in the store or the delta is empty
    (cross-session gaps): the keys whose values differ across the siblings. Keys map
    through ``knob.family_of`` (``TILE@d`` → ``TILE``); a fork deciding several
    families at once joins them sorted (``FM+FN``); an undeterminable delta labels
    ``?`` rather than dropping the fork."""
    from emmy.compiler.pipeline.knob import family_of  # noqa: PLC0415

    def tunable(feats: dict) -> dict:
        return {k: v for k, v in feats.items() if not k.startswith(("S_", "H_"))}

    keys: set[str] = set()
    if parent is not None:
        pk = tunable(parent.features)
        for s in sibs:
            keys |= {k for k, v in tunable(s.features).items() if pk.get(k, _ABSENT) != v}
    if not keys:
        rows = [tunable(s.features) for s in sibs]
        keys = {k for k in set().union(*rows) if len({r.get(k, _ABSENT) for r in rows}) > 1} if rows else set()
    fams = sorted({family_of(k) for k in keys})
    return "+".join(fams) if fams else "?"


@dataclass
class ForkRecord:
    """One multi-child fork priced against its measured truth — the shared unit the
    regret metric, the blame decomposition, and the ablation Δ all consume, so the
    three views agree on pick semantics by construction. ``fvecs`` is each sibling's
    ``knob_features`` row (the representation the prior actually scored — attribution
    masks / diffs these), ``scores`` the prior's predictions over them. ``picked_i``
    resolves predicted-score ties pessimistically (the WORST ``value_us`` among tied
    children): greedy breaks score ties by emission order, so a tie is a miss, not a
    win — the ``rank_of_golden`` ``>=`` rule. ``best_i`` is the measured-best child."""

    op_label: str
    family: str
    regret: float
    siblings: list
    fvecs: list[dict]
    scores: list[float]
    picked_i: int
    best_i: int


def fork_records(prior, nodes) -> list[ForkRecord]:
    """Build one :class:`ForkRecord` per multi-child fork of the ``node`` store:
    group nodes by ``parent_key`` (each group = one fork's children, the partial
    configs the prior actually ranks during ``_select``), featurize each sibling
    once, score the group through the prior's features seam
    (``mean_scores_features`` — identical to ``mean_score`` on the raw knob dict,
    but reusable for masked re-scoring), and price the pick:
    ``regret = value_us(predicted-best child) / min(value_us over siblings)``.
    The ratio is in latency units — 1.00 means the prior steers the search into the
    best-reachable subtree, 1.15 reads as "following the prior costs 15% at this
    fork". ``family`` is the knob family the fork decides (:func:`_fork_family`),
    the bucket the gate thresholds by. Forks whose best ``value_us`` is non-positive
    are skipped (a sentinel, not a measurement). Unlike :func:`reachability` this
    does NOT leaf-filter — siblings are already one fork level (branches and leaves
    alike), scored on their full feature dicts, exactly as ``mcts._select`` queries
    them."""
    from collections import defaultdict  # noqa: PLC0415

    from emmy.compiler.pipeline.search.features import knob_features  # noqa: PLC0415

    by_key = {n.node_key: n for n in nodes}
    by_parent: dict = defaultdict(list)
    for n in nodes:
        if n.parent_key is not None:
            by_parent[n.parent_key].append(n)

    records = []
    for parent_key, sibs in by_parent.items():
        if len(sibs) < 2:
            continue
        true_best = min(s.value_us for s in sibs)
        if true_best <= 0:
            continue
        fvecs = [knob_features(s.features) for s in sibs]
        scores = prior.mean_scores_features(fvecs)
        picked_i, best_i = _pessimistic_pick(scores, sibs), min(range(len(sibs)), key=lambda i: sibs[i].value_us)
        family = _fork_family(by_key.get(parent_key), sibs)
        records.append(
            ForkRecord(_node_op_label(sibs[0].features), family, sibs[picked_i].value_us / true_best, sibs, fvecs, scores, picked_i, best_i)
        )
    return records


def _pessimistic_pick(scores: list[float], sibs: list) -> int:
    """The child index the prior's argmin picks, ties resolved to the WORST measured
    value among tied children (a tie is a miss — see :class:`ForkRecord`)."""
    best_pred = min(scores)
    return max((i for i, p in enumerate(scores) if p == best_pred), key=lambda i: sibs[i].value_us)


def sibling_regret(prior, nodes) -> list[tuple[str, str, float, int]]:
    """Fork-level **value regret** over the autotune ``node`` store — the search-faithful
    gate metric, as ``(op_label, family, regret, n_children)`` tuples (data, not text —
    aggregation and rendering live with the caller). A projection of
    :func:`fork_records`, which holds the semantics."""
    return [(r.op_label, r.family, r.regret, len(r.siblings)) for r in fork_records(prior, nodes)]


def _regret_lines(records: list[tuple[str, str, float, int]]) -> list[str]:
    """Render :func:`sibling_regret` records as the per-kernel × per-family table plus
    the per-family aggregate line — the aggregate IS the gate number; the per-kernel
    rows are the diagnostic view that says *which* kernels the prior misprices."""
    fams = sorted({fam for _, fam, _, _ in records})
    by_op: dict[str, dict[str, list[float]]] = {}
    for op, fam, regret, _n in records:
        by_op.setdefault(op, {}).setdefault(fam, []).append(regret)
    w = max(26, *(len(op) for op in by_op))
    cw = max(7, *(len(f) for f in fams))

    def cell(vals: list[float] | None) -> str:
        return (f"{statistics.median(vals):.2f}x" if vals else "-").rjust(cw)

    lines = [f"      {'kernel':{w}}  " + "  ".join(f.rjust(cw) for f in fams) + "  forks"]
    for op in sorted(by_op):
        row = by_op[op]
        lines.append(f"      {op:{w}}  " + "  ".join(cell(row.get(f)) for f in fams) + f"  {sum(len(v) for v in row.values())}")
    agg = {f: [r for _, fam, r, _ in records if fam == f] for f in fams}
    lines.append(f"      {'ALL (median)':{w}}  " + "  ".join(cell(agg[f]) for f in fams) + f"  {len(records)}")
    return lines


def _node_op_label(features: dict) -> str:
    """The op label for a single ``node`` row — derived from its ``S_*`` features, the
    same labels :func:`reachability` rows carry, so ``--kernel`` filters the node store
    by op kind/shape (e.g. ``matmul`` / ``reduce`` / ``free=512``). All nodes of one op
    share these features, so filtering keeps/drops a whole op atomically — parent and
    children never split."""
    return _op_label(tuple(sorted((k, v) for k, v in features.items() if k.startswith("S_"))))


def _node_gpu_block(prior, gpu: str, gnodes: list) -> list[str]:
    """Per-card metrics: fork sibling regret + leaf reachability/calibration over ONE
    GPU's nodes. Grouping by card is what keeps same-die SKUs (H100 vs H200) from being
    compared against each other — their latencies differ but their ``S_*`` op signature
    (the leaf-reachability group key) is identical."""
    lines = [f"  [{gpu}] {len(gnodes)} nodes"]
    recs = sibling_regret(prior, gnodes)
    if not recs:
        lines.append("    fork sibling regret: no multi-child forks")
    else:
        lines.append("    fork sibling regret (predicted-best child's reachable µs / fork best; 1.00x = optimal, ties pessimistic):")
        lines.extend(_regret_lines(recs))
    groups = Dataset.from_node_rows(gnodes).group_by_op()
    rr = reachability(prior, groups)
    if rr:
        ratios = [r[3] for r in rr]
        agg = f"mean {statistics.mean(ratios):.2f}x  median {statistics.median(ratios):.2f}x  worst {max(ratios):.2f}x"
        lines.append(f"    leaf reachability: {agg}  ({len(rr)} ops, 1.00 = optimum)")
        for label, best_us, pick_us, ratio, n in sorted(rr, key=lambda r: -r[3]):
            if ratio > 1.2:
                lines.append(
                    f"      {label:26}  best {best_us:8.2f}us  pick {pick_us:8.2f}us  ({ratio:.2f}x, {n} configs)  <-- misses best"
                )
    calib = _calibration(prior, groups)
    if calib is not None:
        lines.append(f"    leaf calibration (median per-op Spearman): {calib:+.2f}")
    return lines


def _usable_nodes(nodes, kernel_filter: str | None) -> tuple[list, int, int]:
    """The node-store rows every node metric may read — ``bench_fail`` rows excluded
    (their ``value_us`` is the watchdog sentinel, not a measurement; pre-enrichment
    rows default to ``ok``), rows from another ``FEATURIZER_VERSION`` excluded (their
    ``features`` are spelled in a knob vocabulary the prior can't read — scoring them
    yields constant predictions and a worse-than-random report about a healthy model,
    the 2026-07 tile-IR-rebuild failure class), then the ``--kernel`` op-label filter.
    Returns ``(usable, n_stale, n_before_kernel_filter)`` so callers can report the
    exclusions."""
    from emmy.compiler.pipeline.search.features import FEATURIZER_VERSION  # noqa: PLC0415

    stale = sum(1 for n in nodes if n.feat_ver != FEATURIZER_VERSION)
    nodes = [n for n in nodes if n.status == "ok" and n.feat_ver == FEATURIZER_VERSION]
    total = len(nodes)
    if kernel_filter:
        nodes = [n for n in nodes if kernel_filter in _node_op_label(n.features)]
    return nodes, stale, total


def node_report(prior, nodes, *, kernel_filter: str | None = None) -> str:
    """The ``eval online --dataset nodes`` block. Groups the node store **by card**
    (``NodeRow.gpu``) and, per card, reports the fork sibling regret (the metric unique
    to this dataset — :func:`sibling_regret`) plus leaf reachability / calibration. Per-card grouping is what
    makes a cross-hardware dataset (H100, H200, …) evaluate correctly — same-die SKUs
    share an ``S_*`` op signature but not their latencies, so mixing them would corrupt
    both metrics. A card whose rows span several **compile regimes** (``H_opt`` — the
    -O1 tune rows vs the deployable -O3 re-bench rows the tune also records) splits
    into one sub-block per regime for the same reason: the latencies aren't comparable
    across opt levels. ``nodes`` is a list of :class:`db.NodeRow` from
    :meth:`SearchDB.iter_nodes`; ``kernel_filter`` keeps only nodes whose op label
    contains the substring (``--kernel``). ``bench_fail`` rows are excluded up front —
    their ``value_us`` is the watchdog sentinel, not a measurement, and would corrupt
    the ranking/reachability metrics (pre-enrichment rows default to ``ok``). Rows from
    another ``FEATURIZER_VERSION`` are excluded too, with a printed count: their
    ``features`` are spelled in a knob vocabulary the prior can't read, so scoring them
    yields constant predictions and a worse-than-random report about a healthy model
    (the 2026-07 tile-IR-rebuild failure class). Each card block ends with the
    **golden-anchored descent** section (:func:`_golden_anchor_block`): the regret view
    conditions on forks the search measured, so a golden in a subtree the search never
    built — or a shape with no node data at all — was previously silence that read as
    health (how the 2026-07 prior-saturation bug hid); the anchored rows make that
    absence loud, per golden."""
    from collections import defaultdict  # noqa: PLC0415

    from emmy.compiler.pipeline.search.features import FEATURIZER_VERSION  # noqa: PLC0415

    nodes, stale, total = _usable_nodes(nodes, kernel_filter)
    by_gpu: dict[str, list] = defaultdict(list)
    for n in nodes:
        by_gpu[n.gpu or "(unknown card)"].append(n)
    suffix = f" matching --kernel {kernel_filter!r}" if kernel_filter else ""
    lines = [f"[prior] node store: {len(nodes)} nodes{suffix} across {len(by_gpu)} card(s), fitted={prior.fitted}"]
    if stale:
        lines.append(
            f"  skipped {stale} row(s) from another featurizer vocabulary (feat_ver != {FEATURIZER_VERSION}) — "
            "unreadable by this prior; re-collect with `collect-node-data`"
        )
    if not nodes:
        if kernel_filter and total:
            lines.append(f"  no nodes match --kernel {kernel_filter!r} ({total} in store) — try an op label like 'matmul' / 'reduce'")
        else:
            lines.append("  empty — run `emmy tune <model>` to populate the node table")
        return "\n".join(lines)
    if not prior.fitted:
        lines.append("  no fitted prior — the cold OfflinePrior ranks by D_* geometry only; run `emmy tune`")
    for gpu in sorted(by_gpu):
        gnodes = by_gpu[gpu]
        regimes = sorted({n.features.get("H_opt") for n in gnodes}, key=lambda v: (v is None, v))
        if len(regimes) <= 1:  # single regime → the header stays the bare card name
            lines.extend(_node_gpu_block(prior, gpu, gnodes))
        else:
            for opt in regimes:
                sub = [n for n in gnodes if n.features.get("H_opt") == opt]
                label = f"{gpu} @ -O{int(opt)}" if opt is not None else f"{gpu} @ -O?"
                lines.extend(_node_gpu_block(prior, label, sub))
        lines.extend(_golden_anchor_block(prior, gpu, gnodes, kernel_filter))
    unanchored = _golden_cards_without_rows(set(by_gpu), kernel_filter)
    if unanchored:
        lines.append("  golden-anchored: card(s) with recorded goldens but NO node rows: " + ", ".join(unanchored))
    return "\n".join(lines)


def _golden_cards_without_rows(cards_in_store: set[str], kernel_filter: str | None) -> list[str]:
    """Cards that have recorded matmul goldens but zero rows in the node store — the
    loudest form of the anchored view's absence rule (a whole card's goldens are
    unanchorable, e.g. goldens seeded on a box whose node data was never collected)."""
    from collections import Counter  # noqa: PLC0415

    from emmy.compiler.pipeline.search.golden import GOLDEN_RECORDS  # noqa: PLC0415

    counts = Counter(
        g.gpu_name
        for g in GOLDEN_RECORDS
        if g.is_matmul and g.gpu_name not in cards_in_store and (not kernel_filter or kernel_filter in g.name)
    )
    return [f"{gpu} ({n} golden(s))" for gpu, n in sorted(counts.items())]


# ---------------------------------------------------------------------------
# golden-anchored descent: tie the regret view to the goldens the search may
# never have reached (diagnostic — absence must be loud, never a gate)
# ---------------------------------------------------------------------------


@dataclass
class _AnchorStep:
    """One fork group along a golden's matched path: the knob family it decides,
    whether the prior's (tie-pessimistic) pick keeps the golden's subtree alive, and —
    when lost and both sides are measured — the same-regime value ratio
    ``picked child / best golden-consistent child`` (>1 = the pick steers measurably
    worse than the golden's own branch; <1 = a near-equal-or-better sibling, fine)."""

    family: str
    n_sibs: int
    kept: bool
    gap: float | None


def _golden_prefix_consistent(feats: dict, gold: dict) -> bool:
    """Whether a node's decided tunables agree with the golden's recorded knobs —
    the branch-matching rule of the anchored walk. Family-aware and registry-canonical
    exactly like the A/B pin gate: a bare golden ``TILE`` matches an axis-stamped
    ``TILE@a2`` key (``knob.pin_key_matches``) and values compare through the
    registered knob's canonical parse (``knob.values_equal`` — alias spellings never
    false-mismatch). A family the golden doesn't record, or an axis-keyed golden pin
    that matches no node key, is no constraint (prefix semantics: only *decided,
    comparable* knobs can disagree)."""
    from emmy.compiler.pipeline.knob import family_of, pin_key_matches, values_equal  # noqa: PLC0415

    for k, v in feats.items():
        if k.startswith(("S_", "H_")):
            continue
        fam = family_of(k)
        for gk, gv in gold.items():
            if family_of(gk) == fam and pin_key_matches(gk, k):
                if not values_equal(gk, gv, v):
                    return False
                break
    return True


def _anchor_walk(prior, tree_nodes: list, gold: dict) -> tuple[int, list[_AnchorStep], str]:
    """Walk one op's parent-linked fork groups along the golden's prefix path:
    ``(matched_levels, steps, descriptor)`` for the deepest golden-consistent chain,
    the descriptor a rendered coverage phrase ("followed d/D fork levels …"). A full
    match's denominator is exact (the leaf ends the path); a partial match's is an
    ESTIMATE from the deepest sibling chain below the stop (rendered ``~D``) — the
    golden's own branch topology was never materialized, so the siblings' depth is
    the only witness of how much tree remains. ``tree_nodes`` must be one card's
    -O1-lane rows for the op (the -O3 regime rows are parentless re-measurements,
    not tree members). Each group with ≥2 children gets a descent check
    (:class:`_AnchorStep`); groups are walked per ``context_key`` (trees never span
    compile regimes) and the deepest result wins."""
    from collections import defaultdict  # noqa: PLC0415

    from emmy.compiler.pipeline.search.features import knob_features  # noqa: PLC0415

    # Rows stripped of tree schema (a measurement freeze: parentless with the loader's
    # ``depth=0`` stamp — live rows are always depth >= 1) carry no fork structure;
    # walking them would fabricate one root mega-fork over the op's whole leaf set.
    # Excluded up front, and their absence stays loud via the no-tree descriptor (an
    # empty input keeps the "no explored forks" spelling — the O3-only-op case).
    treeful = [n for n in tree_nodes if n.depth > 0 or n.parent_key is not None]
    if tree_nodes and not treeful:
        return (0, [], "no fork-tree data (leaf-only rows — a measurement freeze?)")
    tree_nodes = treeful

    # Sentinel depth -1 so the first real walk result always replaces it — a level-0
    # "branch never built" must not lose to the sentinel text at equal depth 0.
    best: tuple[int, list[_AnchorStep], str] = (-1, [], "no explored forks")

    def consider(depth: int, steps: list, stop: str) -> None:
        nonlocal best
        if depth > best[0]:
            best = (depth, steps, stop)

    by_ctx: dict = defaultdict(list)
    for n in tree_nodes:
        by_ctx[n.context_key].append(n)
    for op_nodes in by_ctx.values():
        by_key = {n.node_key: n for n in op_nodes}
        children: dict = defaultdict(list)
        for n in op_nodes:
            children[n.parent_key if n.parent_key in by_key else None].append(n)
        depth_memo: dict = {}

        def subtree_depth(key, children=children, depth_memo=depth_memo) -> int:
            """Deepest chain length strictly below ``key`` (0 = no children)."""
            if key in depth_memo:
                return depth_memo[key]
            depth_memo[key] = 0  # cycle guard; parent links form a tree in practice
            kids = children.get(key, [])
            depth_memo[key] = 1 + max(subtree_depth(c.node_key) for c in kids) if kids else 0
            return depth_memo[key]

        def descend(key, node, depth: int, steps: list, children=children, by_key=by_key) -> None:
            group = children.get(key, [])
            if not group:
                if node is not None and bool(node.is_leaf):
                    consider(depth, steps, f"followed {depth}/{depth} fork levels to a measured leaf")
                    return
                # Exploration just stopped here — estimate what remains from the stop
                # node's own siblings' subtrees (the golden's branch may go deeper).
                sib_group = children.get(node.parent_key if node.parent_key in by_key else None, []) if node is not None else []
                est = depth + max((subtree_depth(s.node_key) for s in sib_group if s is not node), default=0)
                cov = f"followed {depth} of ~{est} fork levels" if est > depth else f"followed {depth}/≥{depth} fork levels"
                consider(depth, steps, f"{cov} — subtree unexplored past the match")
                return
            matching = [c for c in group if _golden_prefix_consistent(c.features, gold)]
            if not matching:
                # The divergence group's children sit at depth+1; their deepest chain
                # estimates the levels the golden's unbuilt branch would still have.
                est = depth + 1 + max(subtree_depth(c.node_key) for c in group)
                fam = _fork_family(node, group)
                consider(
                    depth,
                    steps,
                    f"followed {depth} of ~{est} fork levels — golden's branch never built below @{fam} ({len(group)} sibling(s) explored)",
                )
                return
            step = None
            if len(group) >= 2:
                scores = prior.mean_scores_features([knob_features(s.features) for s in group])
                picked = group[_pessimistic_pick(scores, group)]
                kept = any(picked is m for m in matching)
                gap = None
                if not kept:
                    branch_best = min(c.value_us for c in matching)
                    gap = picked.value_us / branch_best if branch_best > 0 else None
                step = _AnchorStep(_fork_family(node, group), len(group), kept, gap)
            for m in matching:
                descend(m.node_key, m, depth + 1, steps + ([step] if step else []), children, by_key)

        descend(None, None, 0, [])
    return best if best[0] >= 0 else (0, [], "no explored forks")  # op had only -O3 regime rows → nothing to walk


def _golden_anchor_block(prior, gpu: str, gnodes: list, kernel_filter: str | None, goldens=None) -> list[str]:
    """The per-card golden-anchored section: one row per recorded golden of THIS card,
    anchored against the card's node rows only. Absence is loud — a golden whose shape
    has no tree data gets a row saying so, and the block headline counts them. The
    golden's recorded µs never touches the -O1 walk or its gaps (the regimes invert);
    it enters only the -O3 endpoint: the prior's pick over the op's ``H_opt=3`` regime
    rows vs ``emmy_us``, same-regime by construction (and skipped on a fast-math
    regime mismatch, the ``golden_deploy_perf`` convention)."""
    from emmy.compiler.pipeline.search.golden import (  # noqa: PLC0415
        GOLDEN_RECORDS,
        fast_math_knobs,
        precision_trading_pins,
    )

    if goldens is None:
        goldens = [g for g in GOLDEN_RECORDS if g.is_matmul and g.gpu_name == gpu]
    if kernel_filter:
        goldens = [g for g in goldens if kernel_filter in g.name]
    if not goldens:
        return []

    def is_o3(n) -> bool:
        return int(n.features.get("H_opt", 0)) == 3

    lines = ["    golden-anchored descent (does the explored tree contain each golden's branch, and does the prior follow it?):"]
    no_data = 0
    w = max(30, *(len(g.name) + 5 for g in goldens))
    for g in goldens:
        pin_suffix = "" if g.pin_map == {"FAST_MATH": False} else f" [{dict(g.pins)}]"
        name = g.name + pin_suffix
        op_nodes = [n for n in gnodes if _matmul_sig(n.features) and ShapeKey.from_s_features(n.features) == g.shape_key]
        if not op_nodes:
            no_data += 1
            lines.append(f"      {name:{w}}  NO TREE DATA for this shape on this card")
            continue
        gold = dict(g.knobs)
        _matched, steps, descriptor = _anchor_walk(prior, [n for n in op_nodes if not is_o3(n)], gold)
        kept = sum(1 for s in steps if s.kept)
        parts = [descriptor]
        if steps:
            detail = "".join(
                f", lost @{s.family}" + (f" {s.gap:.2f}x vs golden branch" if s.gap is not None else "") for s in steps if not s.kept
            )
            parts.append(f"descent kept {kept}/{len(steps)}{detail}")
        o3 = [n for n in op_nodes if is_o3(n) and n.value_us > 0]
        if o3 and g.emmy_us:
            rows = [n.features for n in o3]
            picker = getattr(prior, "pick", None)
            best_i = picker(rows)[0] if picker is not None else min(range(len(o3)), key=lambda i: prior.mean_score(rows[i]))
            tunables = {k: v for k, v in rows[best_i].items() if not k.startswith(("S_", "H_"))}
            if fast_math_knobs(tunables) == precision_trading_pins(g.pin_map):
                parts.append(f"-O3 pick/golden {o3[best_i].value_us / g.emmy_us:.2f}x")
        lines.append(f"      {name:{w}}  " + "; ".join(parts))
    lines.append(f"      summary: {no_data}/{len(goldens)} golden(s) have no tree data on this card")
    return lines


# ---------------------------------------------------------------------------
# per-feature regret attribution: blame decomposition + ablation Δ
# (diagnostic views over the fork records — NOT gate metrics: attribution among
# correlated features is non-unique, so never threshold on these numbers)
# ---------------------------------------------------------------------------

# A term/feature difference below this is float residue, not a model opinion.
_BLAME_EPS = 1e-9


def _blame_lines(prior, records: list[ForkRecord], *, top: int = 8) -> list[str]:
    """The blame table: per fork family, the features whose terms pushed the prior's
    pick above the measured-best sibling, regret-weighted. Per missed fork,
    ``blame_k = term_k(picked) − term_k(best)`` over :meth:`Prior.explain_features`
    — exact for the linear offline prior (the diffs sum to the model's preference
    gap by construction) — weighted by the fork's excess cost ``regret − 1`` (a
    correct pick weighs zero, a 3× miss counts 20× a 1.1× one) and summed per
    family. Positive = the feature argued FOR the wrong child; negative = it argued
    for the right one and was overruled. A missed fork where NO term separates pick
    from best is **blind** — the prior structurally cannot distinguish the siblings
    (the identical-vector failure class) — counted per family and overall."""
    from collections import defaultdict  # noqa: PLC0415

    fams: dict[str, dict] = {}
    for r in records:
        agg = fams.setdefault(r.family, {"n": 0, "missed": 0, "blind": 0, "weight": 0.0, "terms": defaultdict(float)})
        agg["n"] += 1
        if r.regret <= 1.0:
            continue
        tp, tb = prior.explain_features(r.fvecs[r.picked_i]), prior.explain_features(r.fvecs[r.best_i])
        if tp is None or tb is None:
            return ["    (this prior has no per-term decomposition — blame unavailable)"]
        diffs = {k: d for k in set(tp) | set(tb) if abs(d := tp.get(k, 0.0) - tb.get(k, 0.0)) > _BLAME_EPS}
        weight = r.regret - 1.0
        agg["missed"] += 1
        agg["weight"] += weight
        if not diffs:
            agg["blind"] += 1
        for k, v in diffs.items():
            agg["terms"][k] += weight * v

    lines = ["    per-feature blame — regret-weighted Σ(term(picked) − term(best)) over missed forks; + pushed the wrong pick:"]
    for fam in sorted(fams):
        a = fams[fam]
        lines.append(f"      [{fam}]  {a['n']} forks · {a['missed']} missed · regret-weight {a['weight']:.2f}")
        ranked = sorted(a["terms"].items(), key=lambda kv: -abs(kv[1]))[:top]
        for k, v in sorted(ranked, key=lambda kv: -kv[1]):
            lines.append(f"        {k:30} {v:+10.2f}")
        if a["blind"]:
            lines.append(f"        ({a['blind']} missed fork(s) BLIND — no term separates pick from best)")
    blind = sum(a["blind"] for a in fams.values())
    lines.append(f"      blind forks overall: {blind}")
    return lines


def _repick_regret(prior, r: ForkRecord, key: str) -> float:
    """The fork's regret with one feature masked out of every sibling before
    re-scoring — a deleted key carries each model's own absent semantics (``0.0``
    term for the linear prior = exact term removal; ``NaN`` routing for CatBoost)."""
    masked = [{k: v for k, v in f.items() if k != key} for f in r.fvecs]
    picked = _pessimistic_pick(prior.mean_scores_features(masked), r.siblings)
    return r.siblings[picked].value_us / r.siblings[r.best_i].value_us


def _ablation_lines(prior, records: list[ForkRecord], *, min_support: int = 5) -> list[str]:
    """The ablation table: per feature, the change in each family's median regret
    when the feature is masked before scoring (``Δ = regret_masked − regret_full``).
    Negative Δ = the feature is ACTIVELY MISLEADING (picks improve without it);
    positive = load-bearing. Only forks where the feature varies across siblings can
    change their pick, so each row carries its support (that fork count) — a Δ over
    thin support (< ``min_support``, flagged ``~``) is one-fork noise, and features
    that never vary are untestable (summarized, not tabled: they need targeted
    collection, not a Δ=0 row). Features whose Δ rounds to zero everywhere collapse
    into one line. Absent-vs-0.0 is not a variance for this view (the linear prior
    treats them identically; revisit with the SHAP mode)."""
    fams = sorted({r.family for r in records})
    fam_idx: dict[str, list[int]] = {f: [i for i, r in enumerate(records) if r.family == f] for f in fams}
    full = [r.regret for r in records]
    full_med = {f: statistics.median([full[i] for i in idx]) for f, idx in fam_idx.items()}

    varying: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        for k in set().union(*r.fvecs):
            if len({round(f.get(k, 0.0), 9) for f in r.fvecs}) > 1:
                varying.setdefault(k, []).append(i)

    rows = []  # (feature, support_total, {fam: (delta, support)})
    for k, var_idx in varying.items():
        masked = {i: _repick_regret(prior, records[i], k) for i in var_idx}
        cells = {}
        for f, idx in fam_idx.items():
            support = sum(1 for i in var_idx if records[i].family == f)
            delta = statistics.median([masked.get(i, full[i]) for i in idx]) - full_med[f] if support else 0.0
            cells[f] = (delta, support)
        rows.append((k, len(var_idx), cells))

    flat = sorted(k for k, _, cells in rows if all(abs(d) < 5e-3 for d, _ in cells.values()))
    rows = [r for r in rows if r[0] not in flat]
    rows.sort(key=lambda r: (min(d for d, _ in r[2].values()), -r[1]))  # most misleading first, then support

    w = max(28, *(len(k) for k, _, _ in rows)) if rows else 28
    cw = max(7, *(len(f) for f in fams))
    lines = [
        "    ablation Δ — median regret with one feature masked minus full; <0 = actively misleading, >0 = load-bearing:",
        f"      {'feature':{w}}  forks  " + "  ".join(f.rjust(cw) for f in fams),
    ]
    for k, support, cells in rows:
        mark = "~" if support < min_support else " "
        c = ["·".rjust(cw) if not sup else f"{delta:+.2f}x".rjust(cw) for delta, sup in (cells[f] for f in fams)]
        lines.append(f"      {k:{w}}  {support:4}{mark}  " + "  ".join(c))
    if flat:
        lines.append(f"      Δ = 0.00x everywhere ({len(flat)} feature(s)): {', '.join(flat)}")
    lines.append(f"      support legend: forks where the feature varies across siblings; ~ = support < {min_support} (one-fork noise)")
    return lines


def attribution_report(prior, nodes, *, kernel_filter: str | None = None, blame: bool = True, ablate: bool = False) -> str:
    """The ``eval online --dataset nodes --blame / --ablate`` block: per-feature
    regret attribution over the node store's fork records. Fork records are built
    per card (fork identity never crosses cards) but the tables POOL them: regret is
    a within-fork ratio, so cards and compile regimes pool safely — unlike
    reachability/calibration, which :func:`node_report` must split. Diagnostic
    views, not gate metrics (attribution among correlated features is non-unique)."""
    from collections import defaultdict  # noqa: PLC0415

    nodes, _stale, total = _usable_nodes(nodes, kernel_filter)
    suffix = f" matching --kernel {kernel_filter!r}" if kernel_filter else ""
    if not nodes:
        return f"[prior] attribution: no usable nodes{suffix}" + (f" ({total} in store)" if total else "")
    by_gpu: dict[str, list] = defaultdict(list)
    for n in nodes:
        by_gpu[n.gpu or "(unknown card)"].append(n)
    records = [r for gpu in sorted(by_gpu) for r in fork_records(prior, by_gpu[gpu])]
    lines = [f"[prior] per-feature regret attribution: {len(records)} multi-child forks{suffix}, pooled across {len(by_gpu)} card(s)"]
    if not records:
        lines.append("  no multi-child forks — nothing to attribute")
        return "\n".join(lines)
    if blame:
        lines.extend(_blame_lines(prior, records))
    if ablate:
        if not _mask_exact(prior):
            lines.append(
                "    CAVEAT: masked queries are out-of-distribution for the online model (no feature-dropout training yet) — "
                "Δ is indicative only"
            )
        lines.extend(_ablation_lines(prior, records))
    return "\n".join(lines)


def _mask_exact(prior) -> bool:
    """Whether masking a feature is exact for the prior that owns decisions — the
    prior's own answer (``Prior.masking_exact``), which the composite forwards from
    whichever half its blend has answering. True only for the linear offline model,
    where a deleted key is exact term removal; a tree re-routes its splits instead,
    and an online model never trained on masked rows at all."""
    return bool(getattr(prior, "masking_exact", False))
