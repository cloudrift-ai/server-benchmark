"""The CatBoost trainer — the offline learning-to-rank fit of a :class:`CatBoostModel`.

:class:`CatBoostTrainer` holds the hyperparameters and :meth:`~CatBoostTrainer.fit` turns a
:class:`~.group.Group` list into a :class:`CatBoostFit`. Same contract as :class:`~.linear.LinearTrainer` — the
trainer is immutable and one instance serves every cross-validation fold — with one contract the linear trainer
keeps and this one does not: **a fit is not byte-reproducible**. CatBoost's histogram build is threaded, so two
runs on identical inputs give models that differ in the last bits. Deliberate: the alternative is
``thread_count=1``, and the metrics file plus the rank tables are what a fit is compared by, not a checksum.

This module owns what is specific to the model class; everything a different model class would also need lives
elsewhere (the scoring function in :mod:`..catboost_model`, the dataset in :mod:`.group`, the rank metrics in
:mod:`.rank`, the fold harness in :mod:`.cv`).

**The objective.** ``QuerySoftMax`` over one group per pool: the golden is the single positive (label 1.0) and
the sampled negatives are 0.0. That is the loss shape the golden dataset actually has — one *verified* row per
pool, no graded labels — and it is what the deployed ranking is: a softmax over one fork's candidates. The
deployed :func:`~..base.normalize_policy` is that same softmax numerator, so the fitted objective and the
deployed policy are the same function of the model's output.

**Why negatives are sampled.** The linear fit scores whole pools every descent step because a pool is one matrix
multiply. Here the pools are the training set: all of them at once is ~38 M rows / ~18 GB before CatBoost's own
quantized copy. So each round draws ``negatives`` rows per pool, uniformly. The full pool is still what
:meth:`CatBoostFit.score_rows` ranks the golden within, so the *metric* never sees the sampling.

A further round instead mines **hard negatives** — the rows the current model ranks nearest the golden. That is
implemented and reachable (``--rounds 2``) but OFF by default, because the one measurement of it says it hurts;
:data:`DEFAULT_ROUNDS` carries the numbers and the likely reason.

The routing feature (``S_ext_n_symbolic_axis``) is an ordinary column here, re-added from ``Group.dynamic``,
which :meth:`~.group.Group.from_dicts` holds out of the matrix so it can never become a linear descent
coordinate. For a tree it is simply a feature to split on — one model prices both regimes, where the linear model
needs a second weight set.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from emmy.compiler.pipeline.search.prior.catboost_model import ABSENT, DEFAULT_SCALE, CatBoostModel, new_ranker
from emmy.compiler.pipeline.search.prior.fit.group import Group
from emmy.compiler.pipeline.search.prior.fit.rank import rank_of_golden, topk_table
from emmy.compiler.pipeline.search.prior.linear_model import ROUTING_FEATURES

logger = logging.getLogger(__name__)

# Sampled negatives per pool per round. ~500 against one positive puts the softmax denominator in the range the
# loss was designed for, and 490 golden pools × 500 rows is a dataset CatBoost fits in seconds.
DEFAULT_NEGATIVES = 500
# Mining rounds. Round 0 is the uniform draw; each further round adds the rows the current model ranks near the
# golden. ONE by default — mining is implemented and reachable (``--rounds 2``), but it is not on, because the
# only measurement of it says it hurts: over the 1278-case golden dataset, round 1 moved top-1 from 545 to 517
# and the mean log2 rank from 2.08 to 2.35, i.e. worse on the fit's own objective, scored over full pools.
#
# The likely reason is that our negatives are UNLABELED, not known-bad. Mining selects the rows the model ranks
# closest to the golden, which in a ~78k-row pool are exactly the rows most likely to be genuinely faster than
# it; labelling those 0.0 teaches the model to bury good configs. That is the contamination the debiased
# contrastive-learning bound predicts — it scales with the expected NUMBER of better-but-unlabeled rows in the
# contrast set, and mining maximizes that number by construction. Standard hard-negative mining assumes the
# negatives are known bad; here only the positive is verified.
#
# The evidence is in-sample (no cross-validation run yet), so this is a default, not a verdict: re-measure
# ``--rounds 2`` held-out before concluding mining is useless.
DEFAULT_ROUNDS = 1


@dataclass(frozen=True)
class CatBoostTrainer:
    """The tree trainer: hyperparameters in, a fitted :class:`CatBoostFit` out.

    Immutable, and :meth:`fit` never touches ``self``, so ONE instance serves every cross-validation fold and a
    fit is a function of ``(groups, hyperparameters)`` alone — modulo CatBoost's threading, which is why this
    trainer promises reproducible *metrics* rather than a reproducible artifact.

    No ``init`` and no warm start: a tree ensemble does not chain from a previous fit the way a weight vector
    does, so every fit here starts from nothing. That also removes the fold-leakage question the linear trainer
    answers with ``warm_start=False`` — a fold model here cannot inherit anything from a model trained on the
    held-out golden, because there is nothing to inherit."""

    feature_names: tuple[str, ...]
    iterations: int = 500
    depth: int = 6
    learning_rate: float = 0.05
    negatives: int = DEFAULT_NEGATIVES
    rounds: int = DEFAULT_ROUNDS
    random_state: int = 0
    scale: float = DEFAULT_SCALE

    @property
    def cols(self) -> tuple[str, ...]:
        """The model's column order: the feature view plus the routing stamp the dataset holds out of its
        matrix. Spelled once, so the training matrix and the deployed model's ``cols`` cannot disagree."""
        return (*self.feature_names, *ROUTING_FEATURES)

    def fit(self, groups: list[Group]) -> CatBoostFit:
        """Fit over ``groups``, mining hard negatives across :attr:`rounds`.

        Each round rebuilds the training set (golden + the negatives sampled so far) and fits a FRESH model
        rather than continuing the previous one: continuing would weight the early uniform rows by however many
        rounds they have survived, and the point of mining is to shift the distribution, not to accumulate
        emphasis on the first draw."""
        rng = np.random.default_rng(self.random_state)
        cols = list(self.cols)
        pools = [g.matrix(cols, fill=ABSENT) for g in groups]
        # Sampled negative indices per pool, grown each round. The golden is added at assembly time, so it can
        # never be sampled in as a negative against itself.
        sampled = [self._uniform(len(m), g.pinned_idx, rng) for m, g in zip(pools, groups, strict=True)]

        model = self._fit_once(pools, groups, sampled)
        ranks = self._ranks(model, pools, groups)
        logger.info("  round 0 uniform (%d rows): %s", self._n_rows(sampled), topk_table(ranks))
        for r in range(1, max(1, self.rounds)):
            sampled = [
                np.union1d(s, self._hard(model.quality_rows(m), g.pinned_idx)) for s, m, g in zip(sampled, pools, groups, strict=True)
            ]
            model = self._fit_once(pools, groups, sampled)
            ranks = self._ranks(model, pools, groups)
            logger.info("  round %d mined (%d rows): %s", r, self._n_rows(sampled), topk_table(ranks))

        return CatBoostFit(model, ranks, rows=self._n_rows(sampled))

    @staticmethod
    def _n_rows(sampled) -> int:
        return sum(len(s) + 1 for s in sampled)  # +1 for each pool's golden

    @staticmethod
    def _ranks(model: CatBoostModel, pools, groups: list[Group]) -> list[int]:
        """Every golden's rank in its FULL pool under ``model`` — the fit-objective tie convention
        (:func:`~.rank.rank_of_golden`), matching what the linear fit reports per round."""
        return [rank_of_golden(model.quality_rows(m), g.pinned_idx) for m, g in zip(pools, groups, strict=True)]

    def _uniform(self, n: int, golden: int, rng) -> np.ndarray:
        """A uniform draw of negative row indices from one pool, the golden excluded. Small pools contribute
        every row they have rather than being padded with duplicates."""
        others = np.delete(np.arange(n), golden)
        if len(others) <= self.negatives:
            return others
        return rng.choice(others, size=self.negatives, replace=False)

    def _hard(self, scores: np.ndarray, golden: int) -> np.ndarray:
        """The current model's top-ranked non-golden rows — the ones it would deploy INSTEAD of the golden, so
        exactly the rows the loss needs to push down. A uniform draw over a 78k-row pool almost never lands
        here, which is why one uniform round is not enough."""
        order = np.argsort(-scores, kind="stable")
        return np.array([i for i in order[: self.negatives + 1] if i != golden][: self.negatives])

    def _fit_once(self, pools, groups: list[Group], sampled) -> CatBoostModel:
        """One CatBoost fit over the assembled groups — the golden first in each, then its sampled negatives."""
        from catboost import Pool  # noqa: PLC0415

        x, y, gid = [], [], []
        for i, (mat, g, neg) in enumerate(zip(pools, groups, sampled, strict=True)):
            rows = np.concatenate(([g.pinned_idx], np.asarray(neg, dtype=int)))
            x.append(mat[rows])
            y.append(np.concatenate(([1.0], np.zeros(len(rows) - 1))))
            gid.append(np.full(len(rows), i))
        booster = new_ranker(
            loss_function="QuerySoftMax",
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            random_seed=self.random_state,
            thread_count=-1,
            nan_mode="Min",  # absent (NaN) features get their own split-off bucket — see catboost_model
        )
        booster.fit(Pool(np.concatenate(x), np.concatenate(y), group_id=np.concatenate(gid)))
        return CatBoostModel(booster=booster, cols=self.cols, scale=self.scale)


@dataclass(frozen=True)
class CatBoostFit:
    """One :meth:`CatBoostTrainer.fit` result: the fitted model plus the golden ranks it reached.

    The linear fit's ``(static_ranks, dyn_ranks)`` pair has no counterpart — one model ranks every pool, so there
    is one rank list, and :meth:`score_rows` never returns ``None`` (the "this fold could not fit the weight set
    your holdout needs" case cannot arise)."""

    model: CatBoostModel
    ranks: list[int]
    rows: int

    def score_rows(self, group: Group) -> np.ndarray:
        """The group's per-row quality (higher = predicted faster) over its FULL pool, scored exactly as the
        shipped prior ranks. Training sampled negatives; the metric never does."""
        return self.model.quality_rows(group.matrix(list(self.model.cols), fill=ABSENT))

    @property
    def notes(self) -> str:
        """The one-line provenance summary the artifact records."""
        return f"{topk_table(self.ranks)}; trained on {self.rows} sampled rows; not byte-reproducible (threaded fit)"
