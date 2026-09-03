"""The fitted estimate of a kernel's best achievable latency, and the artifact that ships it.

:mod:`~.kernel_cost` says what a kernel looks like and what its roofline is; this predicts how far
above that roofline the kernel will actually land, which turns the roofline into a **microsecond
estimate a structural fork can sum**.

**It is not a** :class:`~.prior.base.Prior`, deliberately. A prior ranks the candidates inside one
kernel's pool and its score is ordinal — meaningful only against its siblings. This answers a
different question, about a whole pool rather than a member of one, and answers it in absolute
microseconds. Nothing in ``Prior``'s surface fits: there is no pool to score rows of, no sibling
set to normalize within, and no reservoir to accumulate.

**What it regresses: ``log(measured_us)``, with the roofline as a FEATURE rather than a divisor.**

The design began the other way — regress ``log(measured / roofline)``, on the argument that a
ratio to a physical floor is card-independent and so transfers to hardware the corpus never saw.
That argument is measurably false, and the measurement is worth keeping because it is not obvious:

Taking the 104 kernels this corpus records on more than one card, and asking how much ONE kernel
moves between cards —

| quantity | median spread across cards | p90 |
| --- | ---: | ---: |
| raw ``log`` latency | 0.26 | 1.08 |
| ratio to the roofline | **0.38** | **1.39** |

Dividing makes a kernel LESS comparable across cards, not more. The floor scales by the ratio of
the cards' peak specs, and real kernels do not: a faster card achieves a smaller fraction of its
own peak, so the division pushes the fast card's ratio down and the slow card's up and injects a
per-card offset where raw latency had a smaller one. Measured on the 101 kernels shared by the
RTX 4090 and 5090, the same kernel is 0.19 slower on the 4090 while its ratio reads 0.28 lower —
the correction has the wrong sign and roughly twice the magnitude it should.

So the floor is kept as ``R_log_roofline_us``, a strong feature the fit can weigh, and nothing is
divided. Dropping the division left in-corpus accuracy unchanged (0.33 against 0.31 mean absolute
log error) and improved every held-out-card figure, which is what the diagnosis predicts.

**MEASURED** over the 982 golden rows, five-fold cross-validated by kernel. Every figure is a
MEDIAN — the error distribution has a heavy tail even in log space (1% of rows carry 12% of total
absolute error, and mean absolute error reads 0.33 against a median of 0.18), so a mean would let
a handful of catastrophic rows set a number meant to describe the typical one:

| predictor | median abs error | p90 |
| --- | ---: | ---: |
| the roofline times one constant — the baseline every cell is scored against | 0.51 | 1.91 |
| this | **0.17** | 0.69 |

A typical held-out kernel is priced within 1.2x, against 1.7x for a scaled floor.

**The limit that matters is not the typical case.** A structural fork compares one fused kernel
against a sum of cut pieces, so what flips a decision is bias that differs BETWEEN families, not
error that is even across them. Median signed error by kind (negative = priced faster than it
runs):

| kind | n | this | baseline |
| --- | ---: | ---: | ---: |
| not a sweep kind | 699 | +0.01 | +0.08 |
| ``rms_norm`` | 185 | +0.01 | +0.14 |
| ``flash`` | 34 | -0.09 | -0.45 |
| ``softmax`` | 12 | +0.09 | +1.08 |
| **``fused``** | **52** | **-0.11** | -0.86 |

The two large cells are stable across splits; ``fused``, ``flash`` and ``softmax`` are 52, 34 and
12 rows and their figures move with the fold count — see the band note below.

**What that costs a decision — and the cell is too thin to state it precisely.** A fused kernel is
priced below its true cost while its pieces are priced about right, so a fuse-or-cut comparison
prefers fusing while the fused kernel is somewhat slower than cutting it. How much somewhat is,
the corpus will not say: across six fold counts the ``fused`` median bias lands anywhere from
-0.03 to -0.34 (median -0.21), because that cell is 52 rows. So the band is **roughly 1.2x, and
anywhere from 1.03x to 1.4x depending on the split** — against about 2.4x for the baseline.

Quote it as a range. Three earlier revisions of this docstring each quoted a single split's figure
to two decimals (-0.56, -0.25, -0.11); every one of them was a draw from that spread, and the
apparent movement between them was not a real change in the model.

**Ordering is the question a fork actually asks, and it is scored separately.** Absolute error
asks whether a microsecond figure is right; a fork asks only which of two arms is smaller, so a
model uniformly 2x off would decide every fork correctly while scoring badly on error. Fraction of
CLOSE same-card pairs ordered as the hardware does — 0.5 is a coin flip:

| card | close pairs | this | baseline |
| --- | ---: | ---: | ---: |
| RTX 5090 | 18 449 | **0.74** | 0.59 |
| RTX 4090 | 4 453 | 0.67 | 0.62 |
| **V100** | 585 | **0.61** | **0.60** |

Two things this shows that the error cells do not. Ordering ability tracks how much data a card
has, and **on the V100 there is none at all** — the model orders close pairs no better than a
predictor that never looks at the kernel. And banding is what makes the statistic mean anything:
over ALL pairs the model scores 0.95 and the baseline 0.86, because most pairs are a 3 us kernel
against a 100 ms one and ordering those takes no skill. A fork is only decided by the model when
its arms are close.

**It does not transfer to an unseen card.** Cross-validating by kernel still lets a fit learn a
card from that card's other kernels. Holding a whole card out:

| held out | this | baseline | this p90 |
| --- | ---: | ---: | ---: |
| RTX 4090, 301 rows | +0.41 | +0.18 | 1.32 |
| RTX 5090, 539 rows | +0.33 | +0.50 | 0.93 |
| **V100, 125 rows** | **-0.68** | -1.05 | **5.02** |

Better than the mean reported (which read -1.39 on the V100, dragged by the tail) and better than
the baseline on two of three, but a typical unseen-V100 kernel is still mispriced 1.9x and a tenth
of them by more than 100x. **And it is not overfitting:** cutting capacity from 400 trees at depth
6 to 100 at depth 3 barely moves these while making ordinary cross-validated error clearly worse.
The features do not carry what distinguishes one card from another; regularization has nothing to
remove. That points at needing goldens from the card in hand, and away from another fit.

So: usable on cards the corpus covers, NOT on a new one.
"""

from __future__ import annotations

import itertools
import math
import statistics
from dataclasses import dataclass

from emmy.compiler.pipeline.search.features import FEATURIZER_VERSION
from emmy.compiler.pipeline.search.prior.fit.cv import DEFAULT_FOLDS, assign_folds

#: Fit settings. Small and shallow on purpose: ~1000 rows against a physical denominator is a
#: low-capacity correction, not a problem wanting depth. These are the online prior's numbers,
#: which were chosen for a comparable dataset size.
ITERATIONS = 400
DEPTH = 6
LEARNING_RATE = 0.05


def _regressor():
    from catboost import CatBoostRegressor  # noqa: PLC0415 — heavy, and only a fit or a load needs it

    # ``MAE`` rather than CatBoost's default ``RMSE``: it fits the conditional MEDIAN, which is
    # what the report measures and what a fork decision needs — a typical kernel, not a tail one.
    # Chosen on a paired comparison over seven splits (MAE better on median error in 5 of 7, by
    # 1.9 standard errors); the mechanism is the reason, the margin only agrees. Huber was clearly
    # worse at every delta tried. Note the metric had to be settled FIRST: compared on MEAN error
    # the three losses look tied, because that metric is the one RMSE optimizes.
    #
    # Small and shallow on purpose: ~1000 rows is a low-capacity problem, and the capacity sweep
    # in the module docstring shows depth buys nothing here that it does not cost in transfer.
    # ``nan_mode`` states the invariant ``catboost_model`` documents rather than relying on it
    # being CatBoost's default: an absent feature is its own bucket, not a zero.
    return CatBoostRegressor(
        iterations=400,
        depth=6,
        learning_rate=0.05,
        loss_function="MAE",
        nan_mode="Min",
        verbose=False,
        allow_writing_files=False,
    )


@dataclass(frozen=True)
class CostModel:
    """A fitted estimate, plus the feature vocabulary it was fitted over.

    ``cols`` is positional and load-bearing: the booster knows column ORDER, not names, so a row
    is packed against this list at both ends. A feature the fit never saw is absent from it and a
    caller offering one is ignored; a feature it did see but a caller omits fills ``nan``, which is
    what CatBoost's native missing-value handling wants — distinct from a real zero."""

    booster: object
    cols: tuple[str, ...]

    def _matrix(self, rows) -> list[list[float]]:
        from emmy.compiler.pipeline.search.prior.catboost_model import ABSENT  # noqa: PLC0415

        return [[float(r.features.get(c, ABSENT)) for c in self.cols] for r in rows]

    def predict_us(self, rows) -> list[float]:
        """Each row's predicted best achievable latency, in microseconds.

        Microseconds rather than the regressed ratio because that is the whole point: a structural
        fork sums these against measured microseconds from the evidence tiers, and a ratio would
        not compose. The roofline each row carries is the scale the ratio is against."""
        return [math.exp(float(v)) for v in self.booster.predict(self._matrix(rows))]

    def to_artifact(self, *, provenance: dict) -> dict:
        """The shipping form: one self-contained JSON object.

        The booster rides base64-inlined rather than as a sidecar file, so the artifact is one
        file to copy, name and load.

        **``feat_ver`` is a partial guard and ``cols`` is the real record.** It stamps the
        featurizer's knob vocabulary, which governs the ``S_*`` half of a row — but not the ``R_*``
        roofline terms defined in :mod:`~.kernel_cost`, nor the card facts read off
        :mod:`emmy.gpu`. Rename an ``R_*`` column or change a bandwidth figure and the stamp does
        not move. A loader that wants a real check should compare ``cols`` against what
        ``kernel_row`` emits today; that check belongs with the loader, and there is not one
        yet."""
        from emmy.compiler.pipeline.search.prior.catboost_model import to_b64  # noqa: PLC0415

        return {
            "feat_ver": FEATURIZER_VERSION,
            "kind": "cost",
            "cols": list(self.cols),
            "model": to_b64(self.booster),
            "provenance": provenance,
        }

    @classmethod
    def from_artifact(cls, obj: dict) -> CostModel:
        """Rebuild from :meth:`to_artifact`'s object. Hard errors, never a silent fallback: a
        caller that quietly scored with the wrong model would measure nothing."""
        from emmy.compiler.pipeline.search.prior.catboost_model import from_b64  # noqa: PLC0415

        if obj.get("kind") != "cost":
            raise RuntimeError(f"not a cost-model artifact: kind={obj.get('kind')!r}")
        if obj.get("feat_ver") != FEATURIZER_VERSION:
            raise RuntimeError(
                f"cost model was fitted under feat_ver={obj.get('feat_ver')!r}, this build is {FEATURIZER_VERSION} — refit it"
            )
        return cls(booster=from_b64(obj["model"], _regressor()), cols=tuple(obj["cols"]))


def target(row) -> float:
    """The regressed quantity for one row: ``log(measured_us)``.

    One spelling, so the fit and every evaluation of it cannot disagree about what was predicted."""
    return math.log(row.best_us)


#: A column has to appear on this fraction of rows to be fitted on. The ``S_*`` stamps are a
#: histogram, so a corpus-wide union carries a long tail of near-empty columns — four of them
#: appear on ONE row of 982 (``S_pw_bitwise_not``, ``S_pw_divide``, and friends). Filtering at 1%
#: takes the vocabulary from 68 columns to 42 and improves median held-out error at every fold
#: count tried.
#:
#: By COVERAGE, deliberately, not by fitted importance: selecting columns on how well they predict
#: the label would be choosing features from the whole corpus and then cross-validating on it,
#: which leaks. How often a column is populated is a property of the data alone.
MIN_COVERAGE = 0.01


def fit(rows) -> CostModel:
    """Fit over every row, on the columns that :data:`MIN_COVERAGE` admits."""
    seen: dict[str, int] = {}
    for r in rows:
        for k in r.features:
            seen[k] = seen.get(k, 0) + 1
    cols = tuple(sorted(k for k, n in seen.items() if n >= len(rows) * MIN_COVERAGE))
    model = CostModel(booster=_regressor(), cols=cols)
    model.booster.fit(model._matrix(rows), [target(r) for r in rows])
    return model


def cross_val_constant(rows, *, folds: int = DEFAULT_FOLDS) -> list[float]:
    """The baseline every cell is measured against: the roofline scaled by ONE constant, the best
    a predictor can do without looking at a kernel at all.

    Not the bare roofline. That carries a -0.65 systematic offset — real kernels sit above their
    floor — and any deployment could remove it by multiplying, so scoring against it would credit
    the model for a correction a single number already makes. Measured over this corpus the bare
    floor reads 1.06 mean absolute error against this baseline's 0.89, and its bias is 0.65
    against this one's zero. Comparing to the weaker of the two would flatter the fit exactly
    where the report is supposed to be sceptical.

    Cross-validated like the model, so the constant comes from the training folds and never from
    the rows it is scored on."""
    assigned = assign_folds([r.fold for r in rows], folds)
    out = [0.0] * len(rows)
    for f in range(folds):
        train = [target(r) - r.features["R_log_roofline_us"] for r in rows if assigned[r.fold] != f]
        if not train:
            continue
        shift = sum(train) / len(train)
        for i, r in enumerate(rows):
            if assigned[r.fold] == f:
                out[i] = math.exp(shift + r.features["R_log_roofline_us"])
    return out


def cross_val_predict(rows, *, folds: int = DEFAULT_FOLDS) -> list[float]:
    """Each row's prediction from a model that never saw its FOLD — the only honest way to read
    this corpus, since a kernel recurs across cards and several goldens can share a pool.

    Folds come from :func:`~.prior.fit.cv.assign_folds`, keyed on ``CostRow.fold`` — the
    card-blind kernel key, so a kernel is held out on EVERY card at once. Splitting any finer lets
    a model see a kernel on one card and be scored on it on another, and two cards carry 86% of
    this corpus."""
    assigned = assign_folds([r.fold for r in rows], folds)
    out = [0.0] * len(rows)
    for f in range(folds):
        train = [r for r in rows if assigned[r.fold] != f]
        held = [(i, r) for i, r in enumerate(rows) if assigned[r.fold] == f]
        if not train or not held:
            continue
        model = fit(train)
        for (i, _), us in zip(held, model.predict_us([r for _, r in held]), strict=True):
            out[i] = us
    return out


def _bias(errors: list[float]) -> dict:
    """One cell: which DIRECTION it is wrong in, how wrong typically, and how wrong at the tail.

    ``bias`` is the signed log error, so a negative cell means the model calls those kernels faster
    than they run — the number that flips a fork, since bias even across families cancels from a
    comparison and bias that differs does not.

    **All three are MEDIAN or quantile, never a mean.** The error distribution has a heavy tail
    even in log space: 1% of rows carry 12% of total absolute error, and mean absolute error reads
    0.33 against a median of 0.18. A mean would let a handful of catastrophic rows set a number
    that is supposed to describe the typical one — and it overstated the ``fused`` bias by more
    than twice, which is the figure a fuse-or-cut decision is read off. ``p90`` is beside them so
    the tail is reported rather than hidden by the robustness.

    The row count is not repeated here: :attr:`~..prior.report.Summary.groups` already carries it,
    and ``report.py`` reserves a metric-level count for metrics with their own size minimum."""
    absolute = sorted(abs(e) for e in errors)
    return {
        "bias": round(statistics.median(errors), 3),
        "err": round(statistics.median(absolute), 3),
        "p90": round(absolute[int(0.9 * len(absolute))], 3),
    }


#: How close two kernels' true latencies must be for their ordering to count as a real question.
#: Over all pairs the model scores 0.95 and the baseline 0.86, which says nothing: most pairs are
#: a 3 us kernel against a 100 ms one, and getting those right takes no skill. A fork is only
#: decided by the model when its arms are close, so that is the population to score on.
CLOSE_RATIO = 1.5


def _concordance(rows, scores, indices) -> tuple[float | None, int]:
    """The fraction of CLOSE same-card pairs the scores order as the hardware does — 0.5 is a coin
    flip, 1.0 is perfect.

    The question a structural fork actually asks. Absolute error asks whether a microsecond figure
    is right; a fork asks only which of two arms is smaller, so a model uniformly 2x off would
    still decide every fork correctly while scoring terribly on error. This measures the thing the
    decision needs, and it is the same statistic ``eval prior`` reports for the ranking prior.

    Pairs never cross cards: two kernels on different cards are not alternatives to each other."""
    ok = total = 0
    for a, b in itertools.combinations(indices, 2):
        ta, tb = rows[a].best_us, rows[b].best_us
        # A tie has no right answer, so scoring it would credit any predictor for free — including
        # one that orders every pair backwards. Excluded like a far-apart pair is.
        if ta == tb or max(ta, tb) / min(ta, tb) > CLOSE_RATIO:
            continue
        total += 1
        ok += (scores[a] < scores[b]) == (ta < tb)
    return (ok / total if total else None), total


def evaluate(rows, *, folds: int = DEFAULT_FOLDS) -> object:
    """Cross-validated report: is this estimate good enough to decide a structural fork?

    Every cell carries :func:`cross_val_constant`'s figure beside the model's — the roofline scaled
    by one constant, which is the strongest predictor that never looks at a kernel. Without that a
    bias number is uninterpretable, and against the BARE floor it would be flattering: see that
    function for why the weaker comparator was rejected.

    Grouped by kernel kind, because a fork is decided by comparing one family against a sum of
    others: bias that is even across families cancels from that comparison and bias that differs
    does not.

    Also split on how many goldens shared a pool. Every label is the best FOUND, never the best
    possible, and goldens cannot bound that gap — but a pool several goldens landed in was
    explored more than a singleton, so a bias difference between the two is the one read on
    censoring this corpus can give.

    There is deliberately no per-card cell under ordinary cross-validation. It would sit beside the
    held-out-card rows below saying something 20x different about the same card, distinguished only
    by a parenthetical: cross-validating by kernel still lets a fit learn a card from that card's
    other kernels, so an in-corpus per-card figure flatters and the held-out one is the answer."""
    from emmy.compiler.pipeline.search.prior.report import EvalReport, Summary  # noqa: PLC0415

    pred = cross_val_predict(rows, folds=folds)
    flat_us = cross_val_constant(rows, folds=folds)
    flat = flat_us
    err = [math.log(p) - math.log(r.best_us) for p, r in zip(pred, rows, strict=True)]
    base = [math.log(p) - math.log(r.best_us) for p, r in zip(flat, rows, strict=True)]

    summaries = [
        Summary(axes={"axis": "", "cell": "all"}, groups=len(rows), unscored=0, metrics={"model": _bias(err), "baseline": _bias(base)})
    ]
    for axis, keyfn in (("kind", lambda r: r.kind or "not a sweep kind"), ("goldens", lambda r: "one" if r.members == 1 else "several")):
        cells: dict[str, list[int]] = {}
        for i, r in enumerate(rows):
            cells.setdefault(keyfn(r), []).append(i)
        for key, idx in sorted(cells.items(), key=lambda kv: -len(kv[1])):
            summaries.append(
                Summary(
                    axes={"axis": axis, "cell": key},
                    groups=len(idx),
                    unscored=0,
                    metrics={"model": _bias([err[i] for i in idx]), "baseline": _bias([base[i] for i in idx])},
                )
            )

    # Ordering, per card. Reported separately from the error cells because it answers a different
    # question about the same predictions — not "is the microsecond figure right" but "would a
    # comparison between two kernels come out right", which is what a fork is.
    cards: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        cards.setdefault(r.gpu, []).append(i)
    for card, idx in sorted(cards.items(), key=lambda kv: -len(kv[1])):
        got, n_pairs = _concordance(rows, pred, idx)
        flat, _ = _concordance(rows, flat_us, idx)
        if got is None or n_pairs < 30:
            continue
        summaries.append(
            Summary(
                axes={"axis": "ordering", "cell": card},
                groups=n_pairs,
                unscored=0,
                metrics={"model": {"concordance": round(got, 3)}, "baseline": {"concordance": round(flat, 3)}},
            )
        )

    # Leave-one-card-out: the stricter question the ordinary folds cannot ask. Cross-validating by
    # kernel still lets a model learn a card from its other kernels; holding a whole card out asks
    # whether the card FEATURES transfer or are being memorized. A diagnostic, not a gate — two of
    # the five cards carry 9 rows between them, where the answer means nothing either way.
    for card in sorted({r.gpu for r in rows}):
        train = [r for r in rows if r.gpu != card]
        held = [r for r in rows if r.gpu == card]
        if len(held) < 20 or not train:
            continue
        out = fit(train).predict_us(held)
        loco = [math.log(p) - math.log(r.best_us) for p, r in zip(out, held, strict=True)]
        shift = sum(target(r) - r.features["R_log_roofline_us"] for r in train) / len(train)
        flat_loco = [shift + r.features["R_log_roofline_us"] - target(r) for r in held]
        summaries.append(
            Summary(
                axes={"axis": "held_out_card", "cell": card},
                groups=len(held),
                unscored=0,
                metrics={"model": _bias(loco), "baseline": _bias(flat_loco)},
            )
        )
    return EvalReport(
        header={
            "dataset": "cost",
            "source": "golden corpus",
            "rows": len(rows),
            "folds": folds,
            "feat_ver": FEATURIZER_VERSION,
            "target": "log(measured_us)",
        },
        summaries=summaries,
    )
