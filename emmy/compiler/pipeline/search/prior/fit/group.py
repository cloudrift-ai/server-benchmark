"""``Group`` — the fit pipeline's dataset representation: one candidate pool plus its labels.

A group is one shape's featurized candidate pool on one card, with whatever supervision exists for it. Today
that supervision is a single pinned verified-optimum row (``pinned_idx`` — the golden's index in the pool);
the planned measurement-freeze datasets add per-row measured labels when they land, so builders other than the
golden case builder can populate groups from measurement sources. Groups are built by plain functions (the golden builder
lives in ``emmy/commands/fit.py`` — case building needs the snippet tracer, which ``pipeline/`` must not
import) and consumed by trainers and the CV harness through this one shape; there is no iterator/batching
layer — the whole dataset is a small in-memory list.

Rows are ndarray-backed, not dict-backed: ``feats`` is one float64 matrix (rows × ``feat_names``), packed
once by :meth:`Group.from_dicts` from the builder's transient per-row feature dicts. A per-row dict of ~63
floats costs ~4 KB; the full golden dataset is ~2.5 M rows, and the dict representation (~10 GB) OOM-killed whole
fit runs — the matrix representation is ~20× smaller.

An absent feature is stored as ``NaN`` and :meth:`Group.matrix` projects it to whatever the caller's model class
means by "absent": ``0.0`` for the linear model (reproducing ``feats.get(k, 0.0)`` bit-identically) or ``NaN``
for a tree, which can branch on not-decided as a state of its own. The dataset stores the more informative of
the two so neither model constrains the other.

``key`` is ``"<gpu>/<name>"``, disambiguated by the builder when one name records several parity entries
(``#2``, ``#3``, … in dataset order). ``tier`` is the fit's case tier (``thread`` / ``warp`` / ``dyn`` /
``reduce`` / ``pointwise``) and is a REPORT LABEL only — it decides nothing. The weight set a group scores
under is ``dynamic``, read off the routing stamp exactly as the deployed prior reads it. ``shape`` is the
cross-validation fold group, and is the one identity here that decides something structural: two goldens
sharing it enumerate the same candidates, so a fold that separated them would train on the answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from emmy.compiler.pipeline.search.prior.linear_model import ROUTING_FEATURES, LinearModel

# The default feature view: the ``D_*`` geometry/occupancy features plus the two ``MMA_*`` atom features that
# vary between a pool's candidates — ``MMA_tier`` (the warp/scalar tier discriminator) and ``MMA_acc_bits``
# (32 for the f32-accumulate atom, 16 for the f16-accumulate one). The ``S_*`` / ``H_*`` shape/regime features
# are constant within a shape, so they drop out of a within-shape ranking — a weight on one is invisible to the
# rank objective (it shifts every candidate in the pool by the same amount) and therefore unidentifiable. The
# routing stamp is the same kind of quantity, which is why it is held out of the matrix entirely rather than
# merely excluded from a view: see :data:`~..linear_model.ROUTING_FEATURES`.
#
# ``MMA_acc_bits`` is load-bearing: the f16-accumulate fork is spelled in the TILE codec's atom token, so a row
# taking it is identical under every ``D_*`` feature to its f32-accumulate sibling. While the view dropped it,
# 93% of a fast-math pool's rows sat in a tied pair no weight vector could separate, and 125 of the 280 RTX 5090
# matmul goldens had a tied candidate ahead of them in emission order — unrankable at top-1 by construction.
# The remaining ``MMA_*`` (``MMA_atom_m/n/k``, ``MMA_a_bits``) measured exactly neutral, so they stay out.
DEFAULT_FEATURES = "D_*,MMA_tier,MMA_acc_bits"

# The tree view: the default view MINUS every feature that exists only because an additive model cannot
# form it. A tree forms these itself from columns the view keeps, so carrying them spends split budget on
# a fact the model can already express — and the hand-set constant inside each one (a target, a threshold)
# is a constant the fit cannot revise. Each exclusion below is derivable by axis-aligned splits on kept
# columns, which is exactly what a tree does; nothing here is a judgement about usefulness.
#
# - MONOTONE DUPLICATES of a kept column. A tree only ever compares a feature to a threshold, so any
#   order-preserving transform of a column it already has is the same column: ``D_l2_threads`` =
#   log2(``D_threads``), ``D_l2_reuse`` = log2(``D_reuse``), ``D_cells_cap`` = clipped ``D_cells``.
# - FOLDS, ``-|x - target|`` around a hand-set target: ``D_near_threads``, ``D_near_area``,
#   ``D_near_cells``, ``D_near_intensity``, ``D_near_tilen``, ``D_near_waves``, ``D_w_near_bk``, and
#   ``D_square`` = ``-|D_aspect|``. A linear model cannot represent a peak, so the peak was precomputed;
#   two splits on the kept column reproduce it, around a threshold the fit chooses rather than inherits.
#   (The tier-aware targets in ``D_near_threads`` / ``D_near_area`` come back as a split on ``MMA_tier``.)
# - THRESHOLDS on a kept column, i.e. one split each: ``D_stage_prefetch`` (``D_stage_depth`` >= 2),
#   ``D_bk_ge32``, ``D_splitk_le2``, ``D_ctas_ge_sm`` (``D_log2_waves`` >= 0), ``D_bn_band``,
#   ``D_bm_band``, ``D_tilen_clean``.
# - MASKED INTERACTIONS of two kept columns — a copy of one feature gated on another being nonzero,
#   which is a split on the gate followed by a split on the feature: the six ``D_tma_*`` mirrors (gated on
#   ``D_stage_tma``) and ``D_l2_cells_occ`` (``D_cells`` gated on ``D_ctas_ge_sm``).
#
# What deliberately STAYS, because axis-aligned splits cannot reach it: ``D_pow2_threads`` (a periodic
# predicate, not an interval), ``D_bn_ge_bm`` (a relation BETWEEN two columns), ``D_w_grid_aspect`` (a
# difference), ``D_log2_area`` (a product), and the whole knob × state block — ``D_splitk_excess`` /
# ``D_splitk_deficit`` / ``D_splitk_roundtrip`` / ``D_near_kchunks`` / ``D_scalar_on_warp_eligible`` —
# whose state operand (the needed split count, the reduce extent, the warp-eligibility stamp) is not a
# candidate column at all, so no split on the pool can recover it.
TREE_FEATURES = (
    "D_*,MMA_tier,MMA_acc_bits,"
    "-D_l2_threads,-D_l2_reuse,-D_cells_cap,"
    "-D_near_threads,-D_near_area,-D_near_cells,-D_near_intensity,-D_near_tilen,-D_near_waves,-D_w_near_bk,-D_square,"
    "-D_stage_prefetch,-D_bk_ge32,-D_splitk_le2,-D_ctas_ge_sm,-D_bn_band,-D_bm_band,-D_tilen_clean,"
    "-D_tma_*,-D_l2_cells_occ"
)

# The matmul view: every feature that can actually move a matmul candidate's rank, and no other. An
# ordinary spec for :func:`feature_view` — pass it as ``--features``; nothing else filters.
#
# Two classes are excluded, both provably free to drop rather than judged uninteresting. Measured over
# the RTX 5090 matmul goldens (286 pools, 36.6 M candidates):
#
# CONSTANT WITHIN EVERY POOL — the feature never varies between a pool's candidates, so a linear model
# adds the same contribution to all of them and the term cancels out of the ranking exactly. Dropped:
# ``D_bk_ge32``, ``D_l2_bk``, ``D_neg_masked_k/m/n``, ``D_pow2_threads``, ``D_raster_gn``,
# ``D_stage_split`` (nothing enumerates it yet), and the four ``S_ext_*`` shape stamps (constant within
# a shape by construction). ``D_pow2_threads`` is the striking one: it carries the shipped artifact's
# LARGEST weight (+136.5, the cold-deploy incident weight) and cannot change a matmul ranking at all.
#
# GLOBALLY AFFINE DUPLICATES — a scaled copy of a kept feature (verified exact, residual at float
# epsilon, over 282 022 rows), so keeping both merely splits one weight across two coordinates:
# ``D_bn_band`` / ``D_bn_ge_bm`` = ``D_bm_band``; ``D_cells_cap`` = ``D_cells``;
# ``MMA_atom_k/m/n`` / ``MMA_tier`` = a multiple of ``MMA_a_bits``. Note this retires ``MMA_tier``,
# which the default view keeps — within matmul it is 0.0625·``MMA_a_bits``. ``MMA_acc_bits`` stays: it
# has three levels and is the only feature separating the f16-accumulate atom from its f32 sibling.
#
# Both exclusions are expressiveness-neutral by construction — the model can express exactly the same
# ranking functions with these 53 coordinates as with all 72 — so this buys a smaller, faster,
# better-identified fit, not a different model. (Naming no ``S_ext_*`` stamp costs the view nothing on the
# weight-set choice either: the routing stamp never reaches the matrix in the first place, and ``Group.dynamic``
# carries the answer.)
#
# ``D_stage_prefetch`` is the one feature here that is a step rather than a measurement — see its
# definition in ``search/features.py`` for why the linear model cannot form it from ``D_stage_depth``.
MATMUL_FEATURES = (
    "D_aspect,D_bm_band,D_cells,D_ctas_ge_sm,D_finalize_kernel,D_l2_bm,D_l2_bn,D_l2_cells_occ,D_l2_reuse,"
    "D_l2_threads,D_log2_area,D_log2_ctas,D_log2_waves,D_near_area,D_near_cells,D_near_intensity,"
    "D_near_kchunks,D_near_threads,D_near_tilen,D_near_waves,D_raster_group,D_reduce_ilp,"
    "D_reduce_transposed,D_reuse,D_scalar_on_warp_eligible,D_splitk,D_splitk_deficit,D_splitk_excess,"
    "D_splitk_le2,D_splitk_roundtrip,D_square,D_stage_async,D_stage_depth,D_stage_prefetch,"
    "D_stage_reg_depth,D_stage_tma,"
    "D_threads,D_tile_m,D_tile_n,D_tilen_clean,D_tma_aspect,D_tma_grid_m,D_tma_grid_n,D_tma_l2_splitk,"
    "D_tma_log2_area,D_w_grid_aspect,D_w_grid_m,D_w_grid_n,D_w_l2_bk,D_w_near_bk,D_wspec_warps,"
    "MMA_a_bits,MMA_acc_bits"
)


def _matcher(pats: list[str]):
    """``name -> bool`` over a pattern list: exact names, or a trailing ``*`` making a prefix glob."""
    prefixes = tuple(p[:-1] for p in pats if p.endswith("*"))
    exact = frozenset(p for p in pats if not p.endswith("*"))
    return lambda name: name in exact or (bool(prefixes) and name.startswith(prefixes))


def feature_view(spec: str):
    """A feature-view spec — comma-separated feature names, a trailing ``*`` making a prefix glob, and a
    leading ``-`` excluding what a later pattern would otherwise have kept (``"D_*,-D_near_*"``) — parsed
    into a ``keep(name) -> bool`` predicate. The view a fit trained under is recorded in its metrics header
    and artifact provenance, so two fits are only comparable when the recorded specs match.

    Exclusions exist so a view can be written as "everything, minus what this model class has no use for"
    (:data:`TREE_FEATURES`). Written as an include list instead, such a view would silently go stale the
    moment the featurizer gained a feature — the new column would be dropped without anyone deciding to
    drop it. Excluding is the safe direction: an unforeseen feature arrives in the view, where at worst the
    model ignores it.

    :data:`~..linear_model.ROUTING_FEATURES` are kept by EVERY view, named or not, and cannot be excluded.
    They select a weight set rather than contribute a term, and :meth:`Group.from_dicts` lifts them out of
    the matrix afterwards, so keeping them costs a view nothing. A view that could drop them would instead
    route every pool to the static weight set and report a fit with zero dynamic cases — silently, since
    nothing downstream can tell an unfittable dynamic set from a genuinely static dataset."""
    pats = [p.strip() for p in spec.split(",") if p.strip()]
    keep = _matcher([p for p in pats if not p.startswith("-")])
    drop = _matcher([p[1:] for p in pats if p.startswith("-")])
    return lambda name: name in ROUTING_FEATURES or (keep(name) and not drop(name))


def feature_matrix(feats: list[dict[str, float]], names: list[str], *, fill: float = 0.0) -> np.ndarray:
    """Feature-dict rows as a dense float64 matrix over ``names`` — absent key = ``fill``."""
    return np.array([[f.get(n, fill) for n in names] for f in feats], dtype=float)


@dataclass(frozen=True)
class Group:
    """One golden's featurized candidate pool, plus the identity the fold axes and metrics keys need."""

    key: str
    name: str
    tier: str
    gpu: str
    # The cross-validation fold group: this pool's extent identity, spelled by the builder from the source
    # record's ``ShapeKey``. Goldens sharing it compete over the same candidates, so they must be held out
    # TOGETHER — see :func:`~.cv.assign_folds`. Unlike ``gpu`` (a report axis) this decides folds; unlike
    # ``tier`` (a label) it is load-bearing.
    shape: str
    dynamic: bool
    pinned_idx: int
    feat_names: tuple[str, ...]
    feats: np.ndarray = field(repr=False)
    # The last ``matrix()`` projection, ``((names, fill), array)`` — a cache, not part of the group's value, so
    # it stays out of ``__eq__`` / ``repr``. See :meth:`matrix`.
    _cache: tuple | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_dicts(cls, key: str, name: str, tier: str, gpu: str, shape: str, pinned_idx: int, feats: list[dict[str, float]]) -> Group:
        """Pack per-row feature dicts into the matrix representation: ``feat_names`` is the sorted
        union of the pool's keys, the matrix a column per name (absent key = 0.0). Callers
        drop the dicts after this — the matrix is the stored representation.

        The routing features (:data:`~..linear_model.ROUTING_FEATURES`) are read off the pool into
        :attr:`dynamic` and then LEFT OUT of ``feat_names``, so a weight-set selector can never
        become a descent coordinate. Reading row 0 is exact: the stamp comes from the shape, which
        every candidate in a pool shares.

        The stamp must agree with ``tier``, and disagreeing is a hard error. The two reach here by
        different routes — the stamp through the featurizer, the tier from the source record's own
        flag — and they are the same fact, so a mismatch means one of them is wrong and this pool
        would otherwise train and be scored under the wrong weight set with nothing reporting it.
        This is what keeps ``tier``, a label that decides nothing, honest."""
        dynamic = bool(feats) and LinearModel.is_dynamic_row(feats[0])
        if dynamic != (tier == "dyn"):
            raise ValueError(
                f"{key}: the routing stamp says dynamic={dynamic} but the case tier says {tier!r} — "
                f"the source record's flag and its featurized rows disagree about the weight set"
            )
        names = tuple(sorted({k for f in feats for k in f} - set(ROUTING_FEATURES)))
        return cls(key, name, tier, gpu, shape, dynamic, pinned_idx, names, feature_matrix(feats, list(names), fill=np.nan))

    def matrix(self, names: list[str], *, fill: float = 0.0) -> np.ndarray:
        """The pool projected onto ``names`` — column ``j`` is the stored ``names[j]`` column, or ``fill``
        where the value is absent, which happens two ways: the pool never stamped that feature at all, or a
        row inside the pool lacks a key its siblings carry.

        The CALLER declares the absent semantics because the two model classes disagree about them, and each
        must be asked with the fill its own training rows were packed with. ``fill=0.0`` (the default) is the
        linear contract — exactly ``feats.get(k, 0.0)`` per row, so scoring/fitting against any feature-name
        list matches the per-dict representation bit for bit. ``fill=np.nan`` is the tree contract, where
        "not decided / not stamped" is a state a split can branch on, distinct from a knob that is present and
        legitimately zero.

        Which is why the STORED matrix holds ``NaN``: it is strictly the more informative of the two, so the
        0.0 view is derivable from it and the reverse is not. Packing 0.0 would have destroyed the
        distinction at :meth:`from_dicts` time, before any model got a say.

        **The result is memoized and READ-ONLY.** Building it is a strided column copy of the whole pool, and a
        cross-validated run asks for the same projection over and over — once per fold's fit and again per
        fold's scoring pass. Measured on the golden dataset, those copies were 85% of a fit's wall time. One
        entry is enough: a fit uses one column list for its whole run, so a different ``(names, fill)`` simply
        evicts the previous one instead of growing without bound.

        Read-only is what makes sharing safe, and it keeps the dataset free of any per-model special case: a
        caller that needs to mutate — the linear fitter z-scores in place — copies inside itself, and one that
        does not pays nothing. Mutating the shared array instead raises."""
        want = (tuple(names), fill)
        if self._cache is None or self._cache[0] != want:
            idx = {n: j for j, n in enumerate(self.feat_names)}
            out = np.full((len(self.feats), len(names)), fill, dtype=float)
            for j, n in enumerate(names):
                if n in idx:
                    out[:, j] = self.feats[:, idx[n]]
            if not np.isnan(fill):
                np.nan_to_num(out, copy=False, nan=fill)
            out.flags.writeable = False
            object.__setattr__(self, "_cache", (want, out))  # frozen dataclass; the cache is not part of its value
        return self._cache[1]
