"""The fitted linear scoring function as a value object — the ONE definition the fitter optimizes and the
deploy path ranks by.

Two access shapes over one arithmetic: :meth:`LinearModel.quality` takes a feature dict (how ``OfflinePrior``
scores a live candidate) and :meth:`LinearModel.quality_rows` takes a packed pool matrix (what ``emmy fit``
descends on — one fp16 golden enumerates ~78k rows, so the per-dict path is not an option there). Before this
module the two were separate transcriptions of the same formula kept in step by a parity test; drift between them
would mean the fitter optimizing something other than what deploys. :meth:`LinearModel.score_rows` is the front
door onto the matrix shape rather than a third shape: it takes a whole candidate pool and decides, once and here,
which columns this model class wants out of it — through :meth:`LinearModel.cols_for`.

**What a model reads is declared by its artifact.** This is the canonical statement of it; everything else
points here. :attr:`LinearModel.columns` is the answer a pool BUILDER packs for, and it covers all three things
this arithmetic touches: both weight sets, the interaction's two inputs (:data:`GATE_FEATURES`), and the routing
stamp. The weight names are already in the artifact as dict keys, so only the rest is stored — the
``unweighted_cols`` field — and :attr:`~LinearModel.columns` unions the two.

The routing stamp is why the declaration could not be left implicit. It selects the weight set rather than
entering the sum, so it is NOT among the columns packed into the matrix (:meth:`LinearModel.cols_for`) and it may
never carry a weight — which means nothing derived from the weight keys can ever name it. A builder trusting such
a derivation packs a pool with no stamp, and every candidate is then priced by the static weight set with no
symptom at all, a pool without a stamp being exactly what a genuinely static pool looks like.
:meth:`LinearModel.score_rows` refuses a pool that carries no stamp for that reason.

The public scoring methods borrow ``Prior``'s own featurized surface — ``mean_score_features`` and
``mean_scores_features`` (``quality`` / ``quality_rows`` / ``score_rows`` are this module's own vocabulary, not
``Prior``'s). That is deliberate: ``FallbackPrior`` composes priors on exactly those, so a holder can delegate to
any model through them without a second vocabulary. :mod:`.catboost_model` is the other model class answering the
same surface. ``explain_features`` is NOT part of it — it is this model's internal per-term breakdown, and
:meth:`LinearModel.quality` is its only caller.
:meth:`~LinearModel.quality` / :meth:`~LinearModel.quality_rows` are linear-only — the pre-transform ranking
quantity a derivative-free descent walks, which a tree model has no additive equivalent of (its matrix entry
point is ``CatBoostModel.quality_rows``, a booster call rather than a dot product).

Weight-set routing is ONE fact read from ONE place: the ``S_ext_n_symbolic_axis`` stamp
(:data:`~..features.ROUTING_FEATURES`, spelled by the featurizer and read by
:func:`~..features.is_dynamic_row`; what the stamp MEANS is this module's business). A symbolic-axis
(masked-tile) kernel prices differently from its static counterpart — boundary-guard tax on small tiles,
staged prologues locked out, occupancy over a free-dim product that excludes the symbolic axis — so it ranks
under ``weights_dynamic``. That stamp must never ALSO be a fitted weight: it is constant across a candidate
pool, so a linear term on it adds the same constant to every candidate and cancels exactly out of the
within-pool ranking. The rank objective cannot see it, and whatever value a descent lands on
there is decided by the regularizer and noise. It routes; it is never a coordinate — but it is a column this
model reads, so it is declared, and a feature view that feeds this model names it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from emmy.compiler.pipeline.search.features import FEATURIZER_VERSION, ROUTING_FEATURES, is_dynamic_row
from emmy.compiler.pipeline.search.prior.base import latency_proxy

if TYPE_CHECKING:
    # Annotation only: importing ``search.data`` for real would pull it (and, through ``freeze.py``, yaml and
    # subprocess) onto the deploy path, which loads none of it today.
    from emmy.compiler.pipeline.search.data.group import Group


def descent_cols(names) -> tuple[str, ...]:
    """``names`` minus the routing stamps — the coordinates a linear descent may walk (why: the module
    docstring's identifiability argument).

    The narrowing lives with the model class, not with the dataset: the dataset packs every column it is
    given, and a tree splits on this one. A dataset that withheld it to protect THIS model class would be
    deciding for both — and did, leaving the tree a column of NaN.

    Filtering preserves relative order, so ``descent_cols(sorted(union | routing))`` is
    ``sorted(union - routing)`` exactly, which is why packing the column left every fitted artifact
    byte-identical."""
    return tuple(n for n in names if n not in ROUTING_FEATURES)


# The scalar params the FIT searches, in descent-coordinate order — the atomic-free interaction, the one term
# the deployed quality cannot express as a linear weight. ``fit/linear.py`` walks exactly these coordinates.
FITTED_PARAMS = ("atomic_free_weight", "atomic_free_split_threshold")

# The artifact's ``params`` block: the complete key set, in the order it is written (``storage.write_json``
# preserves insertion order). ``scale`` is carried rather than fitted, so it leads and the fitted pair follows.
# This is also the set ``offline._load_artifact`` requires an artifact to carry and the set the fitter's
# coordinates are drawn from, so writer, validator and descent cannot drift — add a scalar param once, here.
#
# The order matches what the fitter has emitted since the scalar params became fitted coordinates; it is NOT the
# order in the currently shipped ``offline_weights.json``, which predates that, so the next refit rewrites those
# two lines once. Reordering to match the shipped file would instead change what a refit emits.
PARAM_ORDER = ("scale", *FITTED_PARAMS)

# The two features :func:`atomic_free_term` reads, in the order it takes them — the INTERACTION's input list,
# not a column set. Which column is the finalize flag and which is the split count is the formula's business, and
# the fitted pair in ``params`` prices exactly this formula. Spelled once so the dict path (:func:`gate_values`),
# the matrix path (:func:`gate_columns`) and :meth:`LinearModel.cols_for` cannot disagree.
#
# ONE absent value, 0.0, on both access shapes — what :meth:`Group.matrix` fills a missing column with, so an
# absent finalize flag zeroes the term because the formula says so at ``finalize == 0``. Each name used to carry
# a default of its own instead (finalize 0.0, split count 1.0 — a count no featurization produces), applied when
# the NAME was missing from the packed columns, which made "the view dropped this column" indistinguishable from
# "the pool never stamped it". Neither reader can tell those apart, so the view case is checked where both are
# visible: ``LinearTrainer.fit``, which sees the view and what the pools stamp.
GATE_FEATURES = ("D_finalize_kernel", "D_splitk")


def atomic_free_term(finalize_kernel, splitk, *, weight: float, threshold: float):
    """The split-K finalize interaction, the one part of the quality that is NOT a linear weight: above
    ``threshold`` splits REWARD the deferred combine kernel, below it PENALIZE so a narrow split keeps the cheap
    atomicAdd fast path. The atomic finalize scores zero either way (``finalize_kernel`` is 0), keeping its
    geometry-driven rank.

    Written elementwise so ONE definition serves both access shapes: the dict path passes python floats, the
    matrix path whole feature columns. That shared definition is what makes the fit's objective the deployed
    score rather than a proxy — a hand-set constant here would be a constant the fit optimizes around."""
    return weight * finalize_kernel * ((splitk >= threshold) * 2.0 - 1.0)


def gate_values(feats: dict) -> tuple[float, ...]:
    """The interaction's inputs read off one feature dict — the per-row twin of :func:`gate_columns`, absent
    key = 0.0 (:data:`GATE_FEATURES`)."""
    return tuple(feats.get(name, 0.0) for name in GATE_FEATURES)


def gate_columns(mat: np.ndarray, names) -> tuple[np.ndarray, ...]:
    """The interaction's inputs read off a packed pool, one column each — the matrix twin of
    :func:`gate_values`, a name ``names`` does not carry reading as the same 0.0 (:data:`GATE_FEATURES`).

    Copies, never column views: the fitter z-scores its pools IN PLACE, and these values must stay in raw units —
    the interaction compares a split COUNT against its threshold."""
    idx = {n: j for j, n in enumerate(names)}
    return tuple(mat[:, idx[name]].copy() if name in idx else np.zeros(len(mat)) for name in GATE_FEATURES)


def unweighted_cols(weights: dict[str, float], weights_dynamic: dict[str, float] | None) -> tuple[str, ...]:
    """The columns a model with these weights reads WITHOUT weighting them: the routing stamp, plus any
    interaction input a fit pruned out of both weight sets — :attr:`LinearModel.unweighted_cols`.

    The WRITER's expression, called where a fit decides what its model reads (and again where the ship path
    carries a dynamic set forward). A reader unions two artifact fields instead, and needs neither
    :data:`GATE_FEATURES` nor ``ROUTING_FEATURES`` to do it."""
    return tuple(sorted((set(GATE_FEATURES) | set(ROUTING_FEATURES)) - set(weights) - set(weights_dynamic or {})))


def quality_columns(mat: np.ndarray, w: np.ndarray, gates: tuple[np.ndarray, np.ndarray], *, weight: float, threshold: float):
    """A pool's per-row quality (higher = predicted faster) from already-prepared columns — the linear part plus
    the atomic-free interaction.

    Takes the weight vector and gate columns rather than a :class:`LinearModel` because the fitter calls it
    mid-descent, where the weights are a raw array in the pool's z-space and no model exists yet.
    :meth:`LinearModel.quality_rows` is the same computation entered from a fitted model."""
    return mat @ w + atomic_free_term(gates[0], gates[1], weight=weight, threshold=threshold)


@dataclass(frozen=True)
class LinearModel:
    """A fitted linear ranker over ``features.knob_features`` — the columns it reads, two weight sets, the scalar
    scoring params, and nothing else. Immutable and comparable, so a fit result is a value the caller can diff,
    swap and serialize.

    ``weights_dynamic`` is ``None`` only inside an incomplete fit (a training slice with no symbolic-axis cases).
    A deploy artifact always carries both sets — :meth:`to_artifact` refuses to write a model that does not, and
    the caller substitutes its fallback set first."""

    # The declared columns that are not weight keys — the routing stamp, and any interaction input pruned out
    # of both weight sets. The other half of :attr:`columns`; :func:`unweighted_cols` is where a fit forms it.
    unweighted_cols: tuple[str, ...]
    weights: dict[str, float]
    weights_dynamic: dict[str, float] | None
    # exp() argument scale — keeps the deployed proxy in a finite, sane range. Rank-neutral (a monotone
    # transform of the quality), so it is carried between fits rather than fitted.
    scale: float
    # The atomic-free interaction's fitted pair (see :func:`atomic_free_term`). Both ARE fitted — they are the
    # one term the fit cannot express as a linear weight, so ``emmy fit`` searches them as descent coordinates
    # alongside the weights. A constant the fit cannot see is a constant the fit optimizes around.
    atomic_free_weight: float
    atomic_free_split_threshold: float

    def __post_init__(self) -> None:
        """A routing feature may never carry a fitted weight.

        Withholding the column from the matrix used to make this structural. It is a convention now that
        every column is packed and each model class narrows for itself (:func:`descent_cols`), so it gains
        a guard at construction — the one place a hand-edited artifact is also caught.

        Worth being precise about the harm, since the module docstring above says such a term "cancels
        exactly": within one candidate pool it does, and it cancels out of the greedy argmin, out of
        ``normalize_policy`` and out of ``TiltBlend`` for the same reason. The exception is
        ``policy/greedy._resolved_price``, which SUMS per-kernel scores to compare whole kernel sets — a
        routing weight there scales every symbolic-axis kernel's price and biases the fusion comparison."""
        for name, w_set in (("weights", self.weights), ("weights_dynamic", self.weights_dynamic)):
            if bad := set(w_set or {}) & set(ROUTING_FEATURES):
                raise ValueError(
                    f"{name} carries a fitted weight for routing feature(s) {sorted(bad)} — a routing stamp is "
                    f"constant within a candidate pool, so it selects a weight set and never contributes a term"
                )

    # --- the shared model surface (Prior's own names) ---------------------------------------------------

    def mean_score_features(self, feats: dict) -> float:
        """Latency proxy (``exp(-scale · quality)``), lower is better. A row the weights have no opinion on
        scores the neutral ``1.0``, so ties fall to enumeration order.

        The transform, its exponent bound and the warning that bound fires live in
        :func:`~.base.latency_proxy` — one definition, so this model class and the tree cannot drift on what a
        deployed score means."""
        return latency_proxy(self.quality(feats), self.scale)

    def mean_scores_features(self, feats_list: list[dict]) -> list[float]:
        """Batched :meth:`mean_score_features`, element-wise — this model has no vectorized per-dict path, and
        the arithmetic is a handful of dict lookups per row. The method exists so both model classes answer the
        same batched call; the tree model's version is where batching actually buys something."""
        return [self.mean_score_features(f) for f in feats_list]

    def _terms(self, feats: dict):
        """The score's ``(name, term)`` pairs, in the ONE scoring order — each nonzero linear term by its
        feature name, plus the atomic-free interaction as a ``gate:*`` pseudo-term. :meth:`quality` sums the
        terms and :meth:`explain_features` names them, off this one generator, so the decomposition and the
        number it decomposes cannot drift. Terms that would be ``±0.0`` (an absent or zero feature, or the
        gate on a row with no finalize kernel) are dropped: they are neutral in the total and would only pad
        the explanation."""
        w_set = self.weight_set(is_dynamic_row(feats))
        for k, w in w_set.items():
            if feats.get(k, 0.0):
                yield k, w * feats[k]
        finalize, splitk = gate_values(feats)  # splitk is the structural split-K count
        if finalize:
            yield (
                "gate:atomic_free",
                atomic_free_term(finalize, splitk, weight=self.atomic_free_weight, threshold=self.atomic_free_split_threshold),
            )

    def explain_features(self, feats: dict) -> dict[str, float]:
        """EXACT per-term decomposition of :meth:`quality` (higher = predicted faster) — see :meth:`_terms`.
        A two-row term diff is therefore the model's exact preference gap. (The float-safety clip is ignored
        here — it exists for finiteness, never inside the live range.)"""
        return dict(self._terms(feats))

    # --- linear-only: the quantity the fit optimizes -----------------------------------------------------

    def quality(self, feats: dict) -> float:
        """The ranking quantity itself (higher = predicted faster), before the monotone ``exp(-scale·)`` wrapper:
        the linear weights over ``feats`` plus the atomic-free interaction. Routes itself on the row's stamp —
        a live candidate always carries the full featurization.

        The total of :meth:`_terms` (the one definition of the score's terms — :meth:`explain_features` names
        the same pairs). The terms are summed with ``sum`` in term order, never re-associated: ``sum`` over
        floats is compensated (Neumaier) and re-grouping would change where the compensation lands. Two
        candidates an ULP apart are a tie either way, and the fit's own descent scores through
        :func:`quality_columns`, not this path."""
        return sum(t for _, t in self._terms(feats))

    def quality_rows(self, mat: np.ndarray, names, *, dynamic: bool) -> np.ndarray:
        """:meth:`quality` over a whole packed pool — column ``j`` of ``mat`` is feature ``names[j]``.

        The CALLER routes here rather than the row doing it itself: a feature view may legitimately omit the
        stamp column (it is constant within the pool and carries no weight), so the matrix cannot be relied on to
        answer the question. The dataset carries the answer instead."""
        w_set = self.weight_set(dynamic)
        vec = np.array([w_set.get(n, 0.0) for n in names])
        return quality_columns(
            mat, vec, gate_columns(mat, names), weight=self.atomic_free_weight, threshold=self.atomic_free_split_threshold
        )

    def score_rows(self, group: Group) -> np.ndarray | None:
        """:meth:`quality_rows` over a whole packed pool — the pool-shaped entry point, and the ONE place the
        linear model's column choice is made. :class:`~..catboost_model.CatBoostModel` answers the same
        shape, which is what lets the trainer and the fold harness hand either model class a group and rank it
        without knowing which one they hold.

        Two things a caller must not re-derive, which is why they live here rather than at each call site.
        The COLUMNS come from :meth:`cols_for`. The WEIGHT SET comes from :meth:`weight_set`, so the
        static-vs-dynamic choice is made once, here, and a second copy of that routing cannot drift from it.

        The pool must CARRY the routing stamp: ``Group.dynamic`` is read off it when the rows are packed, so a
        pool that lost it arrives labelled static and would be priced entirely by the static weight set, whatever
        its regime and with nothing in the result to say so (module docstring). Refusing makes that a failure
        rather than a silence, for every builder — a feature view that stopped naming the stamp, a golden pool,
        or measured rows recorded before the featurizer stamped it.

        Nothing else about the pool is required. A column this model weights but the pool never stamped is the
        ordinary absent case and scores 0.0 (:meth:`Group.matrix`'s fill for this model class), which is what a
        reduce or pointwise pool legitimately looks like against an artifact fitted over every regime.

        ``None`` when the group needs the dynamic set and this model has none — the unfittable
        cross-validation fold. Asking :meth:`weight_set` instead would raise, and a fold harness wants an
        answer it can skip on, not an exception."""
        if unstamped := [c for c in ROUTING_FEATURES if c not in group.feat_names]:
            raise ValueError(
                f"pool {group.key!r} carries no {unstamped} column: this model reads the routing stamp to pick a "
                f"weight set, and without it every candidate would be priced as static whatever its regime"
            )
        if group.dynamic and self.weights_dynamic is None:
            return None
        names = self.cols_for(group.dynamic)
        return self.quality_rows(group.matrix(names), names, dynamic=group.dynamic)

    def cols_for(self, dynamic: bool) -> list[str]:
        """The columns to PACK for one weight set's score: that set's names UNION :data:`GATE_FEATURES`. The
        interaction's two inputs have to be present whether or not the weight dict names them, since a pruned
        zero weight drops the key.

        A subset of :attr:`columns`, without the routing stamp: the stamp selects this list rather than
        entering the arithmetic (module docstring), and widening the matmul by a weightless column is exactly
        what the next paragraph says not to do.

        Per weight set rather than one list for both, which the declaration covers jointly. Scoring the static
        set over the union of both would add a column carrying weight ``0.0`` — arithmetically nothing, but
        ``mat @ w`` blocks differently at 60 columns than at 59 and the result moves in the last bits (measured:
        up to 5.7e-14 on qualities of order 10^2, over the shipped artifact). That is not absorbable noise here:
        :func:`~..metrics.dual_rank` counts EXACT float equality to build the pessimistic rank, so a
        perturbation that splits a genuine score plateau moves a reported rank by the whole plateau's width, and
        a rank is an integer no rounding can smooth."""
        return sorted(set(self.weight_set(dynamic)) | set(GATE_FEATURES))

    @property
    def columns(self) -> tuple[str, ...]:
        """Every column this model reads, from the artifact's two halves: the weight keys, plus
        :attr:`unweighted_cols` for what carries no weight. Pack a pool over this and it is a pool this model can
        score, without knowing anything else about it — see the module docstring for why that includes the
        routing stamp."""
        return tuple(sorted(set(self.weights) | set(self.weights_dynamic or {}) | set(self.unweighted_cols)))

    def weight_set(self, dynamic: bool) -> dict[str, float]:
        """The weight dict a row of this kind scores under — the ONE place the two sets are chosen between.
        Raises when the dynamic set is asked for and this model has none; callers that must tolerate that
        (an unfittable CV fold) check :attr:`weights_dynamic` first."""
        w_set = self.weights_dynamic if dynamic else self.weights
        if w_set is None:
            raise ValueError("this model has no dynamic weight set — it came from a fit with no symbolic-axis cases")
        return w_set

    # --- artifact round-trip -----------------------------------------------------------------------------

    @classmethod
    def from_artifact(cls, obj: dict, *, base_dir=None) -> LinearModel:  # noqa: ARG003 — see below
        """Construct from an artifact dict. Deliberately does NOT version-gate: the strict ``feat_ver`` check is
        deploy policy and lives in ``offline._load_artifact``, while the fitter's incumbent-seed read must
        tolerate a mismatch — a refit after a featurizer change is exactly the case a refit exists to fix, and a
        stale key simply seeds ``0.0``.

        ``base_dir`` is accepted and ignored: the loader calls both model classes the same way, and only the tree
        needs a directory to resolve its ``model_file`` sidecar against. A linear artifact is self-contained."""
        params = obj.get("params", {})
        dyn = obj.get("weights_dynamic")
        return cls(
            unweighted_cols=tuple(obj["unweighted_cols"]),
            weights=dict(obj.get("weights", {})),
            weights_dynamic=None if dyn is None else dict(dyn),
            scale=float(params["scale"]),
            atomic_free_weight=float(params.get("atomic_free_weight", 0.0)),
            atomic_free_split_threshold=float(params.get("atomic_free_split_threshold", 0.0)),
        )

    def to_artifact(self, *, provenance: dict, model_file: str | None = None) -> dict:  # noqa: ARG002 — see below
        """This model as the weights artifact dict, in its checked-in shape. ``provenance`` is caller-supplied
        whole (fitted date, script, args, case counts, notes) so the assembly stays pure and deterministic.

        ``model_file`` is accepted and ignored for the same reason ``from_artifact`` ignores ``base_dir``: the
        writer calls both classes identically, and only the tree has bytes that need a sidecar. Everything a
        linear model knows is already text in this dict."""
        if self.weights_dynamic is None:
            raise ValueError("no dynamic weight set — substitute a fallback set before assembling the artifact")
        return {
            "feat_ver": FEATURIZER_VERSION,
            "kind": "linear",
            # Leads the weights: a reader meets the columns that carry no weight before the ones that do, and
            # ``Prior.columns`` is this field unioned with the weight keys.
            "unweighted_cols": list(self.unweighted_cols),
            "weights": self.weights,
            "weights_dynamic": self.weights_dynamic,
            "params": {name: float(getattr(self, name)) for name in PARAM_ORDER},
            "provenance": provenance,
        }
