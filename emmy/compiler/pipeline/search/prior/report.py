"""How well does a prior RANK — as a table of summaries, one serializable schema, two questions.

Hand a model a candidate pool (:class:`~..data.group.Group`), take the scores, and report what the ordering
was worth. This module is where that becomes a number. It computes nothing itself — :mod:`..metrics` owns every
metric's definition — it decides nothing about what to score, and it prints nothing: it takes groups and a
scoring callable and assembles :class:`EvalReport`, which serializes. Turning that into text is the CLI's job
(``emmy/commands/eval.py``), where the console-table renderer already lives.

Both ``emmy eval prior`` and ``emmy fit`` build from here — the eval through :func:`golden_summaries`, which scores a
pool and reads the ranks off the scores, and the fit through :func:`rank_metrics` directly, because a
cross-validated summary spans SEVERAL models and there is no one scorer to hand it. The two commands' numbers are
therefore comparable by construction rather than by two paths agreeing.

**Two questions, and they are not interchangeable.**

- A MEASURED pool (:class:`~..data.group.MeasuredGroup`: freeze or tune-DB rows, every candidate benched) can
  answer what a wrong pick COST. That is :func:`measured_summaries` — Spearman over the pool and regret at the top
  of the ranking — and it is the question that tracks deployed speed.
- A GOLDEN pool (:class:`~..data.group.GoldenGroup`: an enumeration with a verified-optimum row marked) can only
  answer WHERE the known-good row landed. That is :func:`golden_summaries`, and it is reported as a SCREEN: a rank
  is blind to the latency gap behind it, so ranks 1 and 555 of a 164k-row pool may be 0.5% apart or 3x apart,
  and the corpus's rank aggregate is dominated by pools small enough to rank by accident (44% under 100
  candidates). Hence the pool-size axis — a top-1 rate is only readable against the pool it was scored over.

**Cells carry the axes they were keyed on, as a dict.** Measured pools key on ``gpu`` x ``H_opt``; golden pools
on ``gpu`` x ``tier`` x pool-size bucket; both carry ``half``, because the composite prior's two halves fail for
different reasons — the offline half decides what a cold sweep measures at all, the online half owns deploys
once trustworthy — and one unlabelled number would hide which. A fixed axis tuple would force the golden side's
pool buckets onto measured summaries and guarantee empty rows, so each builder declares its own.

**Every summary publishes what it was computed over.** ``groups`` is how many pools keyed into it and ``unscored``
how many of those the model could not score at all. The measured metrics each add their OWN group count, because
they have different minimums: regret needs a pool of at least two rows (a one-row pool is trivially perfect) and
Spearman at least :data:`MIN_SPEARMAN_ROWS`. On the v3 freeze's 336 pools that is 297 and 216 — so an aggregate
that quietly averaged the excluded pools in would be reporting mostly arithmetic. ``regret@10`` is the strictest:
it needs eleven rows, which 90 pools have, so at the freeze's median pool size of seven it still excludes most of
the corpus. The rank metrics have no minimum, so they carry no count of their own.

Polarity is handled once, here. ``score`` returns ranking QUALITY (higher = predicted faster, the
:meth:`Prior.score_rows` contract); the regret and correlation families take a cost (lower = faster), so this
module negates. Both are read as ORDER only, so the negation is exact for a model fitted on ranks as much as
for one that regresses microseconds.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from emmy.compiler.pipeline.search.data.group import GoldenGroup, Group, MeasuredGroup
from emmy.compiler.pipeline.search.metrics import best_dual_rank, spearman, topk_regret

# Minimum pool size a Spearman is reported over. Two rows make ``rho`` computable and meaningless: it is +1 or
# -1 with nothing in between, so a summary of small pools would report a correlation built out of coin flips.
#
# Deliberately below the deploy gate's :data:`~.base._CALIBRATION_MIN_GROUP` (8), which is the same statistic on
# a different job. That one decides whether a model may own deploys, over the model's OWN reservoir, where more
# rows are always available and a false pass is expensive. This one describes an external corpus of fixed size:
# raising it to 8 would drop the freeze well below its 216 measurable pools and report a card as unmeasured
# rather than measured noisily.
MIN_SPEARMAN_ROWS = 5

# The tuning question's frontier width: bench the model's top ``TOPK`` predictions and keep the measured best.
# The deploy question is the same metric at k=1, which is why both are reported side by side — a model can be
# fine to tune with and wrong to deploy from.
TOPK = 10

# Golden top-k coverage. One ladder, imported by the fold harness (``fit/cv.py``) too, so a fit's metrics file
# and an eval report cannot report coverage at different cut-offs and be compared as though they agreed.
TOP_KS = (1, 10, 25, 50, 100)

# Pool-size buckets for the golden axis: ``(exclusive upper bound, label)``, last bound ``None`` = everything
# above. A rank is only readable against its pool size, and the corpus spans three orders of magnitude.
POOL_BUCKETS: tuple[tuple[int | None, str], ...] = ((100, "<100"), (1_000, "<1k"), (10_000, "<10k"), (None, ">=10k"))

# ``score(group) -> quality per row``, or ``None`` when this model cannot score the pool at all.
Scorer = Callable[[Group], "np.ndarray | None"]


def pool_bucket(total: int) -> str:
    """The pool-size bucket label for a pool of ``total`` candidates."""
    return next(label for bound, label in POOL_BUCKETS if bound is None or total < bound)


@dataclass(frozen=True)
class Summary:
    """One row of the report: the axes it was keyed on, what it covered, and its metrics.

    ``metrics`` maps a metric name to that metric's own block. A block carries ``groups`` — the pools it was
    actually computed over — only where that metric has a size minimum and so can differ from ``groups -
    unscored``; a value of ``None`` was computable for nothing in the summary."""

    axes: dict[str, str]
    groups: int
    unscored: int
    metrics: dict[str, dict]

    def to_json(self) -> dict:
        return {"axes": dict(self.axes), "groups": self.groups, "unscored": self.unscored, "metrics": self.metrics}


@dataclass(frozen=True)
class EvalReport:
    """A whole run: the provenance that makes two reports comparable, plus the summaries.

    ``header`` records what was scored and how — the dataset, its source, the enumeration draw, the drop counts
    the builder published. Two reports are only comparable when those match, which is exactly why they ship
    inside the artifact rather than in the prose around it."""

    header: dict
    summaries: list[Summary]

    def to_json(self) -> dict:
        return {"header": self.header, "summaries": [c.to_json() for c in self.summaries]}


def _round(x: float | None, digits: int) -> float | None:
    """Round for a DIFF: a metrics file is compared with ``diff``, and float noise in the last places would
    make two runs of the same fit look like a change."""
    return None if x is None else round(float(x), digits)


def _median(vals: list[float], digits: int) -> float | None:
    return _round(statistics.median(vals), digits) if vals else None


def _summaries(groups: Sequence[Group], score: Scorer, axes_of, metrics_of, *, half: str) -> list[Summary]:
    """Bucket ``groups`` by ``axes_of``, score each, and hand the survivors to ``metrics_of``.

    The scoring pass is here rather than in each metric builder because ``None`` — this model cannot score this
    pool — is a report fact, not a metric fact: a linear model that fitted no dynamic weight set answers
    ``None`` for every symbolic-axis pool, and a summary that dropped those silently would show a healthy static
    corpus and no sign that half the deploy surface is unscored."""
    buckets: dict[tuple, list] = {}
    unscored: dict[tuple, int] = {}
    for g in groups:
        axes = axes_of(g)
        key = tuple(axes.items())
        buckets.setdefault(key, [])
        unscored.setdefault(key, 0)
        scores = score(g)
        if scores is None:
            unscored[key] += 1
        else:
            buckets[key].append((g, np.asarray(scores, dtype=float)))
    out = []
    for key in sorted(buckets):
        entries = buckets[key]
        out.append(
            Summary(
                axes={"half": half, **dict(key)},
                groups=len(entries) + unscored[key],
                unscored=unscored[key],
                metrics=metrics_of(entries),
            )
        )
    return out


# --- measured pools: what the ordering cost ------------------------------------------------------------


def _measured_metrics(entries: list[tuple[MeasuredGroup, np.ndarray]]) -> dict:
    """Spearman and regret over benched pools — the quality vector is negated into a cost so it and the measured
    microseconds read the same direction (see the module docstring)."""
    rhos: list[float] = []
    regrets: dict[int, list[float]] = {1: [], TOPK: []}
    for g, quality in entries:
        cost, measured = (-quality).tolist(), g.latency_us.tolist()
        for k, acc in regrets.items():
            if (r := topk_regret(cost, measured, k)) is not None:
                acc.append(r)
        if len(measured) >= MIN_SPEARMAN_ROWS and (rho := spearman(cost, measured)) is not None:
            rhos.append(rho)
    out = {"spearman": {"median": _median(rhos, 3), "groups": len(rhos)}}
    for k, acc in regrets.items():
        # The WORST group travels with the median because the distribution is one-sided: regret floors at 1.00,
        # so a median near 1.00 with a 4x tail is a model that is usually right and occasionally catastrophic —
        # which is what a deploy actually feels, and what a median alone cannot say.
        out[f"regret{k}"] = {"median": _median(acc, 3), "worst": _round(max(acc), 3) if acc else None, "groups": len(acc)}
    return out


def measured_summaries(half: str, groups: Sequence[MeasuredGroup], score: Scorer) -> list[Summary]:
    """Cells over benched pools, keyed ``gpu`` x ``H_opt`` — the axes :func:`~..data.group.group_measured`
    already grouped on, minus the op identity that separates the pools within a summary."""
    return _summaries(groups, score, lambda g: {"gpu": g.gpu, "H_opt": f"O{g.h_opt:g}"}, _measured_metrics, half=half)


# --- golden pools: where the verified row landed --------------------------------------------------------


def rank_metrics(ranks: Sequence[tuple[int, int]]) -> dict:
    """The golden rank screen from already-computed ``(rank, rank_optimistic)`` pairs.

    Split from :func:`golden_summaries`' scoring pass because a cross-validated summary has no single scorer to pass
    it. ``eval prior`` scores one pool with one model and reads the rank off those scores. A ``holdout`` summary is
    also one rank per case from one model — but a DIFFERENT model per case, each case ranked by the fold model
    that never trained on it — so the summary spans as many models as there are folds and :func:`golden_summaries`'
    single ``score`` callable cannot express it. (The ``train`` block is looser still: it is deliberately
    in-sample, each case ranked by the k-1 models that DID train on it and those ranks medianed, and it exists
    only as the baseline the holdout is subtracted from.) Sharing the assembly rather than the scoring is what
    lets both commands report the same statistic without one of them faking a score vector.

    Ranks are medians, not means: a rank distribution has a long tail, and one 300k-row pool would otherwise
    decide the number for a whole card."""
    pessimistic, optimistic = [r for r, _ in ranks], [o for _, o in ranks]
    # No per-metric ``groups`` here, unlike the measured side: no rank metric has a size minimum, so every
    # block's count would be the summary's own scored total and the summary already publishes that. The top-k
    # denominator is ``groups - unscored``.
    out = {"rank": {"median": _median(pessimistic, 1), "median_optimistic": _median(optimistic, 1)}}
    for k in TOP_KS:
        out[f"top{k}"] = {"count": sum(r < k for r in pessimistic)}
    return out


def _golden_metrics(entries: list[tuple[GoldenGroup, np.ndarray]]) -> dict:
    """:func:`rank_metrics` over a scored pool. A pool with several verified rows is scored on the best of
    them (:func:`~..metrics.best_dual_rank`): deploy ships one config, so any acceptable one ranked first is
    the same win."""
    return rank_metrics([best_dual_rank(quality, g.golden_ids) for g, quality in entries])


def golden_summaries(half: str, groups: Sequence[GoldenGroup], score: Scorer) -> list[Summary]:
    """Cells over golden pools, keyed ``gpu`` x ``tier`` x pool-size bucket.

    The pool bucket is what makes the rank readable: without it a corpus whose small pools rank perfectly and
    whose large ones rank at chance reports one flattering middle number."""
    return _summaries(
        groups,
        score,
        lambda g: {"gpu": g.gpu, "tier": g.tier or "-", "pool": pool_bucket(g.total)},
        _golden_metrics,
        half=half,
    )
