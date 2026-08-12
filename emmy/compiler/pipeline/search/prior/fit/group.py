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
fit runs — the matrix representation is ~20× smaller and :meth:`matrix` reproduces ``feats.get(k, 0.0)`` semantics
bit-identically (an absent feature is a zero column).

``key`` is ``"<gpu>/<name>"``, disambiguated by the builder when one name records several parity entries
(``#2``, ``#3``, … in dataset order). ``tier`` is the fit's case tier (``thread`` / ``warp`` / ``dyn`` /
``reduce`` / ``pointwise``) and is a REPORT LABEL only — it decides nothing. The weight set a group scores
under is ``dynamic``, read off the routing stamp exactly as the deployed prior reads it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from emmy.compiler.pipeline.search.prior.linear_model import ROUTING_FEATURES, LinearModel

# A golden name's trailing size/dtype/variant segments (``512``, ``fp16``, ``dynM``, ``h4096``, ``s2048``,
# ``n16384``, ``k8192``, ``hd128``) — stripped by :func:`op_family`.
_VARIANT_SEG = re.compile(r"fp16|dynM|(?:hd|[hsnk])?\d+")

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


def feature_view(spec: str):
    """A feature-view spec — comma-separated feature names, a trailing ``*`` making a prefix glob
    (``"D_*,MMA_tier"``) — parsed into a ``keep(name) -> bool`` predicate. The view a fit trained
    under is recorded in its metrics header and artifact provenance, so two fits are only comparable
    when the recorded specs match."""
    pats = [p.strip() for p in spec.split(",") if p.strip()]
    prefixes = tuple(p[:-1] for p in pats if p.endswith("*"))
    exact = frozenset(p for p in pats if not p.endswith("*"))
    return lambda name: name in exact or name.startswith(prefixes)


def feature_matrix(feats: list[dict[str, float]], names: list[str]) -> np.ndarray:
    """Feature-dict rows as a dense float64 matrix over ``names`` — absent key = 0.0."""
    return np.array([[f.get(n, 0.0) for n in names] for f in feats], dtype=float)


@dataclass(frozen=True)
class Group:
    """One golden's featurized candidate pool, plus the identity the fold axes and metrics keys need."""

    key: str
    name: str
    tier: str
    gpu: str
    dynamic: bool
    pinned_idx: int
    feat_names: tuple[str, ...]
    feats: np.ndarray = field(repr=False)

    @classmethod
    def from_dicts(cls, key: str, name: str, tier: str, gpu: str, pinned_idx: int, feats: list[dict[str, float]]) -> Group:
        """Pack per-row feature dicts into the matrix representation: ``feat_names`` is the sorted
        union of the pool's keys, the matrix a column per name (absent key = 0.0). Callers
        drop the dicts after this — the matrix is the stored representation.

        The routing features (:data:`~..linear_model.ROUTING_FEATURES`) are read off the pool into
        :attr:`dynamic` and then LEFT OUT of ``feat_names``, so a weight-set selector can never
        become a descent coordinate. Reading row 0 is exact: the stamp comes from the shape, which
        every candidate in a pool shares."""
        dynamic = bool(feats) and LinearModel.is_dynamic_row(feats[0])
        names = tuple(sorted({k for f in feats for k in f} - set(ROUTING_FEATURES)))
        return cls(key, name, tier, gpu, dynamic, pinned_idx, names, feature_matrix(feats, list(names)))

    def matrix(self, names: list[str]) -> np.ndarray:
        """The pool projected onto ``names`` — column ``j`` is the stored ``names[j]`` column,
        or zeros when the pool never saw that feature: exactly ``feats.get(k, 0.0)`` per row,
        so scoring/fitting against any feature-name list matches the per-dict representation bit for bit."""
        idx = {n: j for j, n in enumerate(self.feat_names)}
        out = np.zeros((len(self.feats), len(names)))
        for j, n in enumerate(names):
            if n in idx:
                out[:, j] = self.feats[:, idx[n]]
        return out

    @property
    def family(self) -> str:
        return op_family(self.name)


def op_family(name: str) -> str:
    """The golden's op family — its dot-name with trailing size/dtype/variant segments stripped:
    ``matmul.square.512.fp16`` → ``matmul.square``, ``gemma4_12b.q_proj.s2048`` → ``gemma4_12b.q_proj``,
    ``reduce.k2048.dynM`` → ``reduce``. The leave-one-family-out axis holds out every size/dtype/dynamic
    variant of one op shape together, so the holdout fold measures generalization to an unseen shape family,
    not interpolation between its own sizes. (Model-prefixed names keep the model tag: ``gemma4_12b.mlp_down``
    and ``matmul.mlp_down`` are distinct families — different shape geometry.)"""
    segs = name.split(".")
    while len(segs) > 1 and _VARIANT_SEG.fullmatch(segs[-1]):
        segs.pop()
    return ".".join(segs)
