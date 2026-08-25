"""Base class for the online tuning prior + shared dataset / diagnostics.

A :class:`Prior` is **one global model** consulted by the MCTS to rank candidates
via PUCT (``tune``) and by the greedy driver to pick knobs (``compile`` / ``run``).
Unlike the earlier online-refit-every-few-benches design, it is trained in
**batches**: every tuned op's training rows are streamed into a bounded
dataset (:meth:`add_rows`), and the model is refit (:meth:`maybe_refit`) on a
**dataset-size-tiered cadence** (:data:`REFIT_SCHEDULE`) — frequently while the
model is data-poor, then progressively less often as returns diminish: every
100 rows up to 1k, every 1k up to 10k, every 10k from there on. The dataset is
capped at :data:`MAX_ROWS` by **reservoir sampling** (Algorithm R) — a uniform
random sample of every row ever seen, across runs — so a long-lived global prior
neither grows without bound nor over-weights recent ops. The model is therefore
*fixed during a single op's search*, and exploration comes from PUCT's own term,
not per-bench refitting.

Training labels credit a partial config with its best completion: real benches
exist only at leaves, but the prior ranks partial-knob siblings at every fork
level, so the label for any node is the best (min) latency over its benched
descendants (``1/best_reward``).
The search hands :meth:`add_rows` ``(knobs, median_latency_µs)`` rows for leaves
*and* branches — the prior regresses on latency, and the reward conversion lives
in the MCTS selection loop. :meth:`mean_score` returns ``0`` until the first fit,
so a cold prior gives a uniform PUCT policy (exploration via the PUCT term).

Two scoring surfaces, and only two: :meth:`mean_score` / :meth:`mean_scores` (a
latency-like prediction, lower is better — the greedy argmin and the calibration)
and :meth:`policy` (PUCT's ``P``, normalized within one fork's sibling set). The
retired third one, ``score``, was a per-candidate selection signal from the
Thompson-draw era; every model was deterministic, so it had collapsed into
``mean_score`` everywhere except the composite prior, where it carried a magnitude
blend that measurement showed to be inert.
"""

from __future__ import annotations

import logging
import math
import statistics
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from emmy.compiler.pipeline.search.metrics import spearman

if TYPE_CHECKING:
    from emmy.compiler.pipeline.search.data.group import Group

logger = logging.getLogger(__name__)

# Dataset-size-tiered refit cadence: ``(size_threshold, interval)`` bands, ordered
# by threshold. The refit interval is the first band whose threshold the current
# dataset size is still below, else the last band's interval — so refits coarsen
# as data accumulates: every 100 rows up to 1k, every 1k up to 10k, every 10k
# from 10k on (and beyond the cap, as the reservoir churns).
REFIT_SCHEDULE = ((1_000, 100), (10_000, 1_000), (100_000, 10_000))
# Hard cap on the training set — reservoir-sampled down to a uniform sample of all
# rows ever seen. Bounds memory + fit time for a long-lived global prior.
MAX_ROWS = 100_000

# ``H_opt`` stamp of the deployable -O3 re-bench rows (``tune`` re-benches every
# config within ``EMMY_O3_TOL`` of the -O1 best at -O3 and feeds the row in
# with this tag) — the measured ground truth :meth:`Prior.evidence_pick` ranks by.
_O3_OPT = 3.0

# The calibration gate: minimum median per-op Spearman (predicted vs measured latency over the
# model's own reservoir, :meth:`Prior._reservoir_calibration`) below which :attr:`Prior.trustworthy`
# is False and ``FallbackPrior`` keeps deploys / PUCT on the offline prior. In-sample Spearman is a
# deliberately lenient tripwire: a genuinely trained model scores ~+0.85, while the failure class it
# guards (model and rows built with different featurizer versions' feature names — e.g. a checkpoint
# that crossed a ``FEATURIZER_VERSION`` bump, RTX 5090 sweep-6 finding 2) collapses to ~0. A model that can't rank
# even its own training data must not own decisions.
CALIBRATION_MIN = 0.5
# Minimum rows an op group needs to contribute to the calibration median (smaller groups are noise).
_CALIBRATION_MIN_GROUP = 8


# The exponent bound for :func:`latency_proxy`. ``exp`` overflows a float64 just past 709.8, so this is a
# float-representation boundary rather than a modelling choice, and it is one the shipped artifact cannot
# approach: over all 1377 recorded goldens its quality spans 0 to 277, an exponent of at most 27.7.
#
# It is reachable only by a weight vector nothing bounds. The rank objective is indifferent to that vector's
# MAGNITUDE — scaling it preserves every ordering — so the raw-space L2 is the only thing that pins it, which
# is why ``--l2``'s help calls it a tie-breaker rather than a shrinkage term. ``--l2 0`` removes it.
PROXY_CLIP = 700.0

# Warn once per process. If this fires at all the run is already producing garbage rankings, and a pool holds
# ~78k rows — a line each would bury the finding under itself.
_clip_warned = False


def latency_proxy(quality: float, scale: float) -> float:
    """``exp(-scale · quality)`` — the latency proxy BOTH model classes return from ``mean_score_features``,
    so the two cannot drift on the transform that turns a ranking quality into a deployed score. Lower is
    better, matching the online prior's predicted µs, which is what lets one greedy argmin and one policy
    normalization consume either model.

    Clipping is the last resort and it is LOUD, because a clipped exponent silently destroys a ranking: every
    row past the bound lands on the same float, and a greedy argmin over a plateau of equal scores falls
    through to enumeration order. That is not hypothetical — it is the 2026-07 incident, caused by a ±80 clip
    on the QUALITY, which sat inside the live range and collapsed the whole good region onto one ``exp(-8)``
    value. Moving the bound onto the exponent put it two orders of magnitude outside anything reachable; the
    warning is what makes a return to that regime visible instead of silent.

    Consumers needing a BOUNDED value (the ``FallbackPrior`` tilt multiplier) clamp on their side."""
    global _clip_warned
    arg = -scale * quality
    if not -PROXY_CLIP <= arg <= PROXY_CLIP:
        if not _clip_warned:
            _clip_warned = True
            logger.warning(
                "[prior] latency proxy exponent %.1f is outside +/-%.0f and was clipped — every row past the "
                "bound now scores identically, so this ranking is decided by enumeration order, not by the "
                "model. The shipped artifact peaks near 28; a weight vector this large means a fit with no "
                "effective L2 (--l2 0) or a hand-edited artifact.",
                arg,
                PROXY_CLIP,
            )
        arg = max(min(arg, PROXY_CLIP), -PROXY_CLIP)
    return math.exp(arg)


def normalize_policy(scores: list[float]) -> list[float]:
    """One sibling set's latency-like scores (lower = better) as PUCT weights (higher = better),
    scaled so the BEST sibling is exactly ``1.0`` and a sibling the model prices 10× slower is
    ``0.1``. A non-positive score — a cold model's ``0.0`` — weighs a uniform ``1.0``, so an
    unmeasured sibling stays optimistic and the exploration term still drives breadth.

    Normalizing by the best sibling rather than by the sum is deliberate. PUCT compares
    ``c · P · √(N+1)/(1+n)`` against ``Q ∈ [0, 1]``, so ``P`` has to keep a scale comparable to
    ``Q``; dividing by the sum would shrink every ``P`` by ``1/n_siblings`` and quietly collapse
    exploration on wide forks, where breadth matters most. Dividing by the best also IS the
    documented intent — "a confidently-bad sibling gets a small ``P`` → tiny exploration term →
    deprioritized rather than force-visited".

    For either offline model this is exactly a Boltzmann weight: ``min(s)/s_i =
    exp(scale·(quality_i − quality_max))``, the softmax numerator over the sibling set — which is
    also the form the tree model is trained under (CatBoost ``QuerySoftMax``)."""
    positive = [s for s in scores if s > 0]
    if not positive:
        return [1.0] * len(scores)
    best = min(positive)
    return [best / s if s > 0 else 1.0 for s in scores]


class Prior(ABC):
    """Abstract global tuning prior over a bounded, reservoir-sampled dataset.

    ``max_rows`` / ``min_rows`` are instance params (defaulting to the module
    constants); ``refit_every`` overrides the tiered :data:`REFIT_SCHEDULE` with a
    single fixed interval (used by tests)."""

    def __init__(self, *, seed: int = 0, min_rows: int = 50, refit_every: int | None = None, max_rows: int = MAX_ROWS) -> None:
        self._seed = seed
        self.min_rows = min_rows
        self.refit_every = refit_every  # None → tiered REFIT_SCHEDULE
        self.max_rows = max_rows
        self._rng = np.random.default_rng(seed)
        # The bounded training set + reservoir bookkeeping. ``_seen`` is the total
        # rows ever offered (for Algorithm R); ``_since_fit`` is rows added since
        # the last fit (the refit trigger). Both persist in the checkpoint.
        self._dataset: list[tuple[dict, float]] = []
        self._seen = 0
        self._since_fit = 0
        # Bench trajectory for the end-of-run stats: (knobs, median, status) in
        # bench order. ``_first_fit_idx`` marks the warmup/post boundary.
        self.trajectory: list[tuple[dict, float, str]] = []
        self._first_fit_idx: int | None = None
        # Median per-op in-sample Spearman, refreshed by :meth:`maybe_refit` after
        # each fit and persisted in the checkpoint — the :attr:`trustworthy` gate's
        # input. ``None`` = not yet measured (ungated, so direct-``fit()`` callers
        # and the offline prior keep working).
        self.calibration: float | None = None
        # Checkpoint binding — set by ``load``; lets :meth:`checkpoint` persist
        # without the caller threading a path.
        self._path = None
        # Lazily-built -O3 evidence index (see :meth:`_o3_evidence`); the
        # fingerprint tracks reservoir churn so the index rebuilds on change.
        self._ev_index: dict[frozenset, list[tuple[dict, float]]] = {}
        self._ev_fp: tuple[int, int] | None = None

    @property
    @abstractmethod
    def fitted(self) -> bool: ...

    @property
    def trustworthy(self) -> bool:
        """Whether this model may OWN decisions (deploys, PUCT, structural pricing):
        fitted AND not demonstrably mis-calibrated (:data:`CALIBRATION_MIN` over the
        reservoir). ``fitted`` alone let a garbage model ship silently — the 2026-07
        RTX 5090 sweeps' finding class. An unmeasured calibration (``None``) passes:
        the gate is a tripwire for measured failure, not a proof-of-quality demand."""
        return self.fitted and (self.calibration is None or self.calibration >= CALIBRATION_MIN)

    @property
    def masking_exact(self) -> bool:
        """Whether deleting a key from a featurized row removes that feature's contribution
        EXACTLY, which the ablation diagnostics need to know before quoting a Δ. False by
        default — true only for the linear offline model, where a deleted key is exact term
        removal. A tree re-routes the splits reading that column, and a model trained without
        feature dropout has never seen a masked row at all."""
        return False

    @abstractmethod
    def fit(self) -> None:
        """Refit the model on the current :attr:`_dataset`."""

    @abstractmethod
    def mean_score(self, knobs: dict) -> float:
        """Prediction for ranking a candidate — the greedy argmin, the calibration, and the
        input :meth:`policy` normalizes. ``0.0`` until the first fit (cold prior → uniform
        PUCT policy)."""

    def mean_scores(self, knobs_list: list[dict]) -> list[float]:
        """Batched :meth:`mean_score` — the greedy driver flattens a kernel's whole
        candidate set into one scoring pass, so a model with a vectorized predict
        (``OnlinePrior``, ``OfflinePrior`` on a tree model) overrides this to score
        the lot in a single call. The default maps element-wise."""
        return [self.mean_score(k) for k in knobs_list]

    # --- the PUCT policy ----------------------------------------------------

    def policy(self, knobs_list: list[dict]) -> list[float]:
        """PUCT's ``P`` over ONE fork's sibling set: how much this model prefers each sibling,
        relative to the best of them.

        Normalized WITHIN the set (:func:`normalize_policy`), which is what makes it composable:
        the online model predicts calibrated µs and the offline one an ordinal proxy whose
        magnitude spans ``e**±700``, and only after normalization are the two the same kind of
        quantity. It also removes the failure the raw magnitudes caused — an offline proxy fed
        to PUCT as though it were µs.

        Batched by construction: a sibling set is scored in one call, so a vectorized model pays
        its per-call overhead once per fork rather than once per candidate."""
        return normalize_policy(self.mean_scores(knobs_list))

    # --- scoring already-featurized rows (attribution / ablation / offline fitting) ----------

    def mean_score_features(self, feats: dict) -> float:
        """:meth:`mean_score` on an ALREADY-featurized row (``features.knob_features``
        output). The entry point the attribution diagnostics and the offline fitter
        score through: they featurize once, then mask / perturb individual features —
        which no knob value maps back to — before scoring. Contract:
        ``mean_score_features(knob_features(knobs)) == mean_score(knobs)``, and a
        DELETED key carries each model's own absent-feature semantics (``0.0`` term
        for the linear offline prior, ``NaN`` routing for CatBoost)."""
        raise NotImplementedError

    def mean_scores_features(self, feats_list: list[dict]) -> list[float]:
        """Batched :meth:`mean_score_features`; override for a vectorized predict."""
        return [self.mean_score_features(f) for f in feats_list]

    @abstractmethod
    def score_rows(self, group: Group) -> np.ndarray | None:
        """This model's ranking QUALITY for every row of a packed candidate pool — higher = predicted faster.
        ``None`` when this model cannot score the pool at all (the linear model asked for a dynamic weight set
        it never fit); a model with nothing fitted yet still answers, with a constant.

        The pool-shaped scoring surface, and the third one this class has after :meth:`mean_scores` (knob dicts)
        and :meth:`mean_scores_features` (feature dicts). It exists because the dict surfaces cannot be used on
        the pools these questions are actually asked over: a matmul enumeration runs to ~10^5 rows, and the
        per-row dict of ~63 floats that made the fit OOM would make an evaluation OOM for the same reason. The
        packed matrix is that representation done once, and each implementation here projects it onto its OWN
        column list with its OWN absent-value fill (see :meth:`Group.matrix`) — which is the whole reason this
        cannot be one shared function over ``feat_names``.

        Polarity is deliberately the opposite of :meth:`mean_score`'s. This returns quality because that is what
        the fitted model classes compute and what the rank metrics take; the deployed score is the monotone
        ``exp(-scale·quality)`` of it, and a caller wanting the cost family (:func:`~..metrics.topk_regret`,
        :func:`~..metrics.spearman`) negates. Both orders are read as ORDER only, so the negation is exact.

        The pool must carry every column the model reads. A group packed under a narrow feature view — the fit's
        ``D_*`` view, say — answers the linear model correctly, because its weight names are inside that view,
        while the online model's ``S_*`` / ``H_*`` columns would all fill absent and its predictions would be
        about a kernel with no shape. Which columns a group carries is the BUILDER's decision, so an evaluation
        that scores both halves builds its pools over the full featurization."""

    def explain_features(self, feats: dict) -> dict[str, float] | None:
        """Signed per-term decomposition of this model's opinion on a featurized row,
        in the model's own ranking-quality units (HIGHER = predicted faster) — the
        attribution report diffs two rows' terms to attribute a misranking to features.
        ``None`` when the model has no decomposition (the default; the offline
        prior returns an exact one, a tree model may return SHAP values). Exactness
        contract, when implemented: the terms sum to the model's full quality score
        for the row, so ``Σ (term(a) − term(b))`` IS the model's preference gap."""
        return None

    # --- deployable-evidence pick ------------------------------------------

    @staticmethod
    def sig_groups(index: dict[frozenset, list], sig: frozenset) -> list[list]:
        """The index groups compatible with a candidate's ``S_*`` signature: the
        exact hit when present, else every group agreeing on all *shared* keys.
        A key present on one side only means the featurizer gained or lost a
        feature between recordings, not a shape difference — the deploy candidate's base can carry scheduler
        stamps neither the reservoir rows nor persisted perf rows have (#311's
        ``S_warp_eligible`` appears in no reservoir/perf row), and a
        strict-equality join lets one added feature silently disable an entire
        evidence tier. Shapes always share the extent keys, so shared-key
        agreement still separates them; an empty shared set matches nothing.
        Shared by this reservoir tier and the deploy-side DB tier
        (``policy/greedy._db_measured_pick``)."""
        if sig in index:
            return [index[sig]]
        cand = dict(sig)
        groups = []
        for row_sig, measured in index.items():
            row = dict(row_sig)
            shared = cand.keys() & row.keys()
            if shared and all(cand[k] == row[k] for k in shared):
                groups.append(measured)
        return groups

    def _o3_evidence(self) -> dict[frozenset, list[tuple[dict, float]]]:
        """Reservoir rows tagged ``H_opt=3`` (real -O3 measurements), indexed by
        their ``S_*`` structural signature. Rebuilt lazily whenever the reservoir
        changed (``(_seen, len)`` fingerprint — Algorithm R replacement bumps
        ``_seen`` even at constant length)."""
        fp = (self._seen, len(self._dataset))
        if self._ev_fp != fp:
            index: dict[frozenset, list[tuple[dict, float]]] = {}
            for knobs, us in self._dataset:
                if float(knobs.get("H_opt", 0.0)) != _O3_OPT or us <= 0:
                    continue
                sig = frozenset((k, v) for k, v in knobs.items() if k.startswith("S_"))
                tun = {k: v for k, v in knobs.items() if not k.startswith(("S_", "H_"))}
                index.setdefault(sig, []).append((tun, float(us)))
            self._ev_index = index
            self._ev_fp = fp
        return self._ev_index

    def evidence_pick(self, rows: list[dict], *, exact_families: frozenset[str] = frozenset()) -> tuple[int, float] | None:
        """Measured-evidence argmin over candidate knob rows: the candidate whose
        knob prefix is consistent with the **fastest -O3 reservoir row** of the
        same op (``S_*`` signature). A candidate is consistent with a measured row
        when every tunable knob the candidate specifies matches the row (knobs the
        candidate hasn't decided yet are free, so a partial fork prefix inherits
        the best measured outcome among its completions). Families named by
        ``exact_families`` instead require exact subset equality; placement forks
        use that to distinguish a fused leaf from each cut fragment.

        Returns ``(index, measured_µs)`` or ``None`` when no candidate has
        evidence. This is what keeps the greedy deploy from preferring an
        unmeasured extrapolation over a config the tune already proved fastest at
        -O3 (the 2026-06-12 golden-sweep findings 2–6 failure class). The model's
        ranking surface (:meth:`mean_scores`) is untouched — evidence only decides
        the argmin."""
        if not rows or float(rows[0].get("H_opt", _O3_OPT)) != _O3_OPT:
            return None  # deploying a non--O3 regime — -O3 evidence doesn't apply
        index = self._o3_evidence()
        if not index:
            return None
        from emmy.compiler.pipeline.knob import canonical_row_key, evidence_row_vouches  # noqa: PLC0415

        best: tuple[int, float] | None = None
        for i, cand in enumerate(rows):
            sig = frozenset((k, v) for k, v in cand.items() if k.startswith("S_"))
            cand_tun = {k: v for k, v in cand.items() if not k.startswith(("S_", "H_"))}
            for measured in self.sig_groups(index, sig):
                for row_tun, us in measured:
                    # A row counts as evidence when it matches every knob the candidate
                    # has decided; undecided knobs are free (``evidence_row_vouches``).
                    if not evidence_row_vouches(cand_tun, row_tun, exact_families=exact_families):
                        continue
                    # Tie on µs (one measured row matching several candidates) breaks by
                    # the candidates' canonical content, never their enumeration order.
                    if best is None or us < best[1] or (us == best[1] and canonical_row_key(cand) < canonical_row_key(rows[best[0]])):
                        best = (i, us)
        return best

    def pick(self, rows: list[dict]) -> tuple[int, float]:
        """The deploy argmin: measured -O3 evidence first (:meth:`evidence_pick`),
        the model's :meth:`mean_scores` argmin when no candidate has evidence.
        Score ties break by :func:`~emmy.compiler.pipeline.knob.canonical_row_key`
        (candidate content, never enumeration order — same-featurized siblings score
        identically, and an order-broken tie flips the deployed kernel per boot).
        Returns ``(index, µs)`` — measured when evidence decided, predicted
        otherwise."""
        ev = self.evidence_pick(rows)
        if ev is not None:
            return ev
        from emmy.compiler.pipeline.knob import canonical_row_key  # noqa: PLC0415

        scores = self.mean_scores(rows)
        best_i = min(range(len(scores)), key=lambda i: (scores[i], canonical_row_key(rows[i])))
        return best_i, scores[best_i]

    # --- dataset + batched refit ------------------------------------------

    def add_rows(self, rows: list[tuple[dict, float]]) -> None:
        """Stream training rows into the bounded dataset via reservoir
        sampling (Algorithm R): under the cap they append; at the cap each new row
        replaces a uniformly-random existing one with the correct probability, so
        ``_dataset`` stays a uniform sample of every row ever seen."""
        for k, v in rows:
            self._seen += 1
            self._since_fit += 1
            row = (dict(k), float(v))
            if len(self._dataset) < self.max_rows:
                self._dataset.append(row)
            else:
                j = int(self._rng.integers(self._seen))
                if j < self.max_rows:
                    self._dataset[j] = row

    def _refit_interval(self) -> int:
        """Rows that must accumulate before the next refit — the fixed
        ``refit_every`` override if set, else the :data:`REFIT_SCHEDULE` band for
        the current dataset size (coarsening as data grows)."""
        if self.refit_every is not None:
            return self.refit_every
        n = len(self._dataset)
        for threshold, interval in REFIT_SCHEDULE:
            if n < threshold:
                return interval
        return REFIT_SCHEDULE[-1][1]

    def maybe_refit(self, *, force: bool = False) -> bool:
        """Refit (and stamp the warmup boundary) iff enough rows have streamed in
        since the last fit (:meth:`_refit_interval`) and the dataset clears
        ``min_rows``. ``force`` (used at end-of-run) bypasses the interval so a
        small tune that never crossed a tier threshold still gets a fitted model —
        but only when there's new data or no model yet, never a redundant re-fit.
        Returns whether a fit happened (caller checkpoints on ``True``)."""
        if len(self._dataset) < self.min_rows:
            return False
        if force:
            if self.fitted and self._since_fit == 0:
                return False  # nothing new since the last fit
        elif self._since_fit < self._refit_interval():
            return False
        self.fit()
        self._since_fit = 0
        self._note_fit()
        # Calibration gate input: can the fresh model rank its own reservoir? A
        # measured collapse (feature/label vocabulary mismatch) flips
        # :attr:`trustworthy` off, and ``FallbackPrior`` quarantines the model
        # instead of shipping it to deploys. Logged so a quarantined model is
        # visible rather than silently demoted.
        self.calibration = self._reservoir_calibration()
        if self.calibration is not None:
            verdict = "owns deploys" if self.trustworthy else f"QUARANTINED (< {CALIBRATION_MIN}) — deploys stay offline"
            logger.info("[prior] reservoir calibration %+.2f → online model %s", self.calibration, verdict)
        return True

    def _reservoir_calibration(self) -> float | None:
        """Median per-op in-sample Spearman ρ between the model's predicted latency
        and the reservoir labels — ops grouped by their ``S_*`` signature, groups
        under :data:`_CALIBRATION_MIN_GROUP` rows skipped. ``None`` when nothing is
        measurable (no fitted model, no big-enough group, scipy absent). In-sample by
        design: it costs one vectorized predict over rows already in memory, never
        blocks a genuinely trained model, and catches exactly the collapse where
        the model and its rows no longer share feature names (constant
        predictions → ρ ≈ 0)."""
        if not self.fitted or not self._dataset:
            return None
        groups: dict[tuple, list[tuple[dict, float]]] = defaultdict(list)
        for knobs, label in self._dataset:
            if label <= 0:
                continue
            sig = tuple(sorted((k, v) for k, v in knobs.items() if k.startswith("S_")))
            groups[sig].append((knobs, label))
        rhos = []
        for rows in groups.values():
            if len(rows) < _CALIBRATION_MIN_GROUP:
                continue
            rho = spearman(self.mean_scores([k for k, _ in rows]), [v for _, v in rows])
            if rho is not None:
                rhos.append(rho)
        return statistics.median(rhos) if rhos else None

    def record_bench(self, knobs: dict, median: float, status: str) -> None:
        """Append a benched leaf to the trajectory (for the summary stats)."""
        self.trajectory.append((dict(knobs), median, status))

    def checkpoint(self) -> None:
        """Persist this prior to its bound JSON file (a no-op if unbound or there's
        nothing to save). The binding is set by ``load``."""
        if self._path is None:
            return
        obj = self.to_json()
        if obj is None:
            return
        from emmy import storage  # noqa: PLC0415

        storage.write_json(self._path, obj)

    def to_json(self) -> dict | None:
        """Serialize the model + dataset for persistence (``None`` when there's
        nothing yet). Override in subclasses; see :meth:`OnlinePrior.to_json`."""
        return None

    def _note_fit(self) -> None:
        if self._first_fit_idx is None:
            self._first_fit_idx = len(self.trajectory)

    # --- end-of-run sanity stats ------------------------------------------

    def summary(self, label: str) -> str:
        """A compact stats block judging the prior by *which picks* it made — how
        often it picked a config at least 2x slower than the best, before vs after
        the model's first fit, and the rank correlation between predicted and
        measured latency on the post-fit picks."""
        oks = [(k, us) for k, us, st in self.trajectory if st == "ok" and us > 0]
        n = len(self.trajectory)
        if not oks:
            return f"[prior] {label} — {n} benches, no ok measurements"
        best = min(us for _, us in oks)
        best_idx = next(i for i, (_, us, st) in enumerate(self.trajectory) if st == "ok" and us == best)
        warm = self._first_fit_idx if self._first_fit_idx is not None else n
        lines = [
            f"[prior] {label} — {n} benches (warmup {warm} / post {n - warm}), dataset {len(self._dataset)}/{self.max_rows}",
            f"  best:        {best:.3f} us @ bench #{best_idx}",
            f"  silly (>=2x best):  {self._silly(0, warm, 2 * best)}   {self._silly(warm, n, 2 * best, post=True)}",
        ]
        calib = self._calibration(warm, n)
        if calib is not None:
            lines.append(f"  calibration (post Spearman pred vs latency): {calib:+.2f}")
        return "\n".join(lines)

    def _silly(self, lo: int, hi: int, thresh: float, *, post: bool = False) -> str:
        tag = "post" if post else "warmup"
        oks = [us for _, us, st in self.trajectory[lo:hi] if st == "ok" and us > 0]
        if not oks:
            return f"{tag} 0/0"
        silly = sum(1 for us in oks if us >= thresh)
        return f"{tag} {silly}/{len(oks)} ({100 * silly / len(oks):.0f}%)"

    def _calibration(self, lo: int, hi: int) -> float | None:
        """Spearman ρ between the prior's predicted latency and the measured
        latency over post-warmup ok picks (``scipy.stats.spearmanr``). Both are
        µs, so a well-calibrated prior gives ρ near ``+1``."""
        if not self.fitted:
            return None
        pred, latency = [], []
        for knobs, us, st in self.trajectory[lo:hi]:
            if st != "ok" or us <= 0:
                continue
            pred.append(self.mean_score(knobs))
            latency.append(us)
        return spearman(pred, latency) if len(pred) >= 3 else None
