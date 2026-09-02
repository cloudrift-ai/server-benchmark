"""What ``emmy tune`` prints about its own prior after an offline refit, and the golden joins
behind it — coverage and per-golden rank, with no GPU.

``emmy tune`` with no model / ``--code`` refits the prior on its persisted reservoir dataset and
prints :func:`report`: how many rows and op structures it holds, and how many golden matmul shapes
it has any data for. Counting, not judging.

**Ranking quality is deliberately not computed here.** It was, until 2026-08: a per-op pick ratio
and a median Spearman over the reservoir, grouped by :meth:`Dataset.group_by_op`. That key is the
``S_*`` signature alone, so one group pooled measurements taken under different opt levels — a
sweep of that era wrote both into one reservoir — and pooled cards. The regimes invert, so the
ratio compared measurements taken under different compilers. Both
statistics live in ``prior/report.py`` now, over a grouping keyed on card and regime as well as
kernel, and are reached through ``emmy eval prior``.

The node-tree diagnostics that also lived here — fork-sibling regret, the golden-anchored descent,
per-feature blame and ablation Δ — were retired at the same time; see Part 8 of
``pipeline/ARCHITECTURE.md`` for what they measured and why nothing replaced them in kind.
"""

from __future__ import annotations

from emmy.compiler.pipeline.search.data import Dataset, ShapeKey, is_matmul
from emmy.compiler.pipeline.search.prior.report import TOP_KS


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
        if is_matmul(d):
            have.add(ShapeKey.from_s_features(d))
    golden_keys = {g.shape_key for g in GOLDEN_RECORDS if g.is_matmul}
    covered = sum(1 for k in golden_keys if k in have)
    return covered, len(golden_keys)


def golden_prior_eval(prior, kernel_filter: str | None = None) -> str:
    """Per golden matmul, the golden's **rank under the prior** over the shape's
    full (gated) enumeration — the realistic "would greedy-with-prior pick the
    golden?" test (greedy picks the prior's predicted-fastest config across the
    enumeration, not just the benched leaves a measured pool holds).
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
        if not is_matmul(d):
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
        ranked = golden_eval.evaluate_record(g, ctx, scorer=lambda r, b=base: -prior.mean_score({**b, **r}))
        rank, pool = ranked.rank, ranked.pool
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
        cov = "  ".join(f"top{k}={sum(r < k for r in ranks)}/{n}" for k in TOP_KS)
        lines.append(f"  median rank={sorted(ranks)[n // 2]}  {cov}  (over {n} golden shapes with tuned data)")
    elif not skipped:
        lines.append("  no golden shapes have tuned data in the dataset yet")
    return "\n".join(lines)


def golden_deploy_perf(prior, kernel_filter: str | None = None) -> dict[str, float]:
    """Per golden shape, ``pick_us / golden_us`` — the deployable (-O3) latency of the
    prior's predicted-best **measured** config over the golden's recorded latency, read
    from the prior's reservoir with **no re-bench**.

    Tuning measures in the deployable regime and feeds every row to the prior, so each tuned
    shape's best config has a deployable row in the reservoir. For each
    golden shape we take the op group's ``H_opt=3`` rows (the filter still earns its place: a
    legacy checkpoint can hold rows from the era of a separate ranking lane), pick the one
    ``Prior.pick`` deploys (measured evidence first, model argmin otherwise — the same selection
    greedy ``compile`` / ``run`` make), and divide its measured latency by the golden's
    recorded ``emmy_us`` (also -O3 → same regime, so the ratio is a real
    deployable speed comparison; <1.0 = the prior's pick is faster than golden). Shapes
    with no -O3 reservoir row are omitted (the caller renders ``—``). The reservoir is
    used rather than the raw ``perf`` table because only it carries the ``H_*`` regime
    columns needed to isolate the deployable measurements.

    Goldens are scoped to the live card (:func:`goldens_for_live_gpu`) so a multi-GPU
    goldens dir doesn't make a name's per-card entries collide on the GPU-blind
    ``ShapeKey`` (e.g. RTX 5090 / RTX PRO 6000 both ``(12, 0)``)."""
    from emmy.compiler.pipeline.search.golden import (
    # noqa: PLC0415,
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
        if not is_matmul(d):
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

    # Ranking quality is NOT reported here. It was, as a per-op pick ratio and a median Spearman over
    # these same groups, and both were wrong: ``group_by_op`` keys on the ``S_*`` signature alone, so one
    # group pooled rows measured under different opt levels (a sweep of that era wrote both into one
    # reservoir) and pooled cards. The two regimes invert, so the ratio was a comparison between
    # measurements taken under different compilers. ``emmy eval prior`` computes the same two statistics
    # over a correctly-keyed grouping; this block reports only what the reservoir can say by counting.
    covered, total = _golden_coverage(groups)
    lines.append(f"[prior] golden coverage: {covered}/{total} golden matmul shapes have data in the dataset")
    if covered == 0:
        lines.append("  none yet — tune a working golden file (`emmy tune --golden-file PATH`) to validate against them")
    lines.append("[prior] ranking quality: run `emmy eval prior --dataset nodes` (this block counts coverage only)")
    return "\n".join(lines)
