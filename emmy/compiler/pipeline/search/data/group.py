"""``Group`` — one candidate pool plus its labels: the candidate pools every ranking question is asked over.

A group is one shape's featurized candidate pool on one card, with whatever supervision exists for it. The
base carries the pool and its identity and nothing about supervision, because the two kinds of it are not
one thing: :class:`GoldenGroup` holds the row INDICES goldens verified, so it alone can answer which rows are
the answer, and :class:`MeasuredGroup` holds a benched latency PER ROW, so it alone can answer what a wrong
pick cost. Nothing generic reads either — the metrics take plain sequences and never see a group — so a
single label column on the base would be a field with no consumer and two mutually exclusive meanings.

Groups are built by plain functions — :func:`group_measured` below for benched rows, and the golden one in
``emmy/commands/fit.py``, because case building needs the snippet tracer that ``pipeline/`` must not import —
and are consumed by the trainers and the fold harness through this one shape; there is no iterator/batching
layer, the whole dataset is a small in-memory list.

It lives with the other data types rather than under ``prior/fit/`` because a candidate pool is data, not a
fitter detail — the fit is only its first consumer. The planned evaluation reports rank over the same pools, and
a second pool representation for them would be a second chance to disagree about what a candidate set IS.
Nothing here imports a model: a group carries the columns and the labels, and each model class decides for
itself which columns it wants (see :meth:`Group.matrix`).

Rows are ndarray-backed, not dict-backed: ``feats`` is one float64 matrix (rows × ``feat_names``), packed
once by :func:`pack_features` from the builder's transient per-row feature dicts. A per-row dict of ~63
floats costs ~4 KB; the full golden dataset is ~2.5 M rows, and the dict representation (~10 GB) OOM-killed whole
fit runs — the matrix representation is ~20× smaller.

An absent feature is stored as ``NaN`` and :meth:`Group.matrix` projects it to whatever the caller's model class
means by "absent": ``0.0`` for the linear model (reproducing ``feats.get(k, 0.0)`` bit-identically) or ``NaN``
for a tree, which can branch on not-decided as a state of its own. The dataset stores the more informative of
the two so neither model constrains the other.

``key`` is ``"<gpu>/<name>"`` of the first golden the builder found in this pool, disambiguated when one
name opens several distinct pools (``#2``, ``#3``, … in dataset order); the other goldens over that pool are
further entries in its label set rather than groups of their own — the builder resolves that before it
constructs anything, so a group is built once, already knowing every verified row it holds. ``tier`` is the
fit's case tier (``thread`` / ``warp`` / ``dyn`` / ``reduce`` / ``pointwise``) and is a REPORT LABEL only — it
decides nothing.
The weight set a group scores under is ``dynamic``, read off the routing stamp exactly as the deployed prior
reads it. ``shape`` is the cross-validation fold group, and is the one identity here that decides something
structural: two goldens sharing it enumerate the same candidates, so a fold that separated them would train on
the answer.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from emmy.compiler.pipeline.search.data.freeze import freeze_reason
from emmy.compiler.pipeline.search.features import ROUTING_FEATURES, is_dynamic_row, knob_features
from emmy.compiler.structural import digest

# The default feature view: the ``D_*`` geometry/occupancy features plus the two ``MMA_*`` atom features that
# vary between a pool's candidates — ``MMA_tier`` (the warp/scalar tier discriminator) and ``MMA_acc_bits``
# (32 for the f32-accumulate atom, 16 for the f16-accumulate one). The ``S_*`` / ``H_*`` shape/regime features
# are constant within a shape, so they drop out of a within-shape ranking — a weight on one is invisible to the
# rank objective (it shifts every candidate in the pool by the same amount) and therefore unidentifiable. The
# routing stamp is the same kind of quantity, which is why the LINEAR model narrows it out of its own
# coordinates (:func:`~..prior.linear_model.descent_cols`) while the matrix still carries it for a tree to split on.
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
# weight-set choice either: :attr:`Group.dynamic` carries the answer, and the linear fit narrows the stamp
# out of its coordinates regardless of the view.)
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


def pack_features(feats: list[dict[str, float]]) -> tuple[tuple[str, ...], np.ndarray, bool]:
    """Per-row feature dicts as ``(feat_names, matrix, dynamic)`` — the stored representation, without the
    labels. Separate from :class:`Group` because a builder needs it BEFORE it can say what a pool's labels
    are: two goldens share a pool when their packed rows are identical, so the packing is what decides
    which groups exist, and only then is it known how many verified rows each one has.

    Callers drop the dicts after this. That is the whole point of packing — a per-row dict of ~63 floats
    costs ~4 KB, and holding the corpus in that form OOM-killed whole fit runs."""
    names = tuple(sorted({k for f in feats for k in f}))
    return names, feature_matrix(feats, list(names), fill=np.nan), bool(feats) and is_dynamic_row(feats[0])


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

    :data:`~..features.ROUTING_FEATURES` are kept by EVERY view, named or not, and cannot be excluded.
    They select a weight set rather than contribute a term, and the packed column is what a tree splits the
    two regimes on, so keeping them costs a view nothing. A view that could drop them would instead route
    every pool to the static weight set and report a fit with zero dynamic cases — silently, since nothing
    downstream can tell an unfittable dynamic set from a genuinely static dataset. What keeps the stamp out
    of a LINEAR descent is the model class narrowing its own coordinates
    (:func:`~..prior.linear_model.descent_cols`), not the dataset withholding the column."""
    pats = [p.strip() for p in spec.split(",") if p.strip()]
    keep = _matcher([p for p in pats if not p.startswith("-")])
    drop = _matcher([p[1:] for p in pats if p.startswith("-")])
    return lambda name: name in ROUTING_FEATURES or (keep(name) and not drop(name))


def feature_matrix(feats: list[dict[str, float]], names: list[str], *, fill: float = 0.0) -> np.ndarray:
    """Feature-dict rows as a dense float64 matrix over ``names`` — absent key = ``fill``."""
    return np.array([[f.get(n, fill) for n in names] for f in feats], dtype=float)


@dataclass(frozen=True)
class Group:
    """One featurized candidate pool, plus the identity the fold axes and metrics keys need."""

    key: str
    name: str
    tier: str
    gpu: str
    # The cross-validation fold group: this pool's extent identity — spelled by the golden builder from the
    # source record's ``ShapeKey``, and by :func:`group_measured` as the op's own signature. Goldens sharing it
    # compete over the same candidates, so they must be held out TOGETHER by the fold harness. Unlike ``gpu``
    # (a report axis) this decides folds; unlike ``tier`` (a label) it is load-bearing.
    shape: str
    dynamic: bool
    feat_names: tuple[str, ...]
    feats: np.ndarray = field(repr=False)
    # The size of the candidate pool ``feats`` was drawn from, equal to ``len(feats)`` unless the
    # builder sampled. Two things need it and neither can recover it from the matrix: a report,
    # which must print the raw sample rank BESIDE the true total rather than scaling it, and the
    # linear trainer, whose z-scored moments are over the whole pool and are ESTIMATED from a
    # sample - so each group's rows carry weight ``total / len(feats)`` in those two passes.
    # Without it a 5-row pool and a 325k one would weigh the same under fixed-size sampling, which
    # silently changes the standardization and with it the raw-space L2 the artifact ships.
    total: int
    # The last ``matrix()`` projection, ``((names, fill), array)`` — a cache, not part of the group's value, so
    # it stays out of ``__eq__`` / ``repr``. See :meth:`matrix`.
    _cache: tuple | None = field(default=None, repr=False, compare=False)

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
        distinction at :func:`pack_features` time, before any model got a say.

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


@dataclass(frozen=True)
class MeasuredGroup(Group):
    """A pool whose labels ARE the measurement — one benched latency in microseconds per row, lower = faster.

    The counterpart of :class:`GoldenGroup`: there the labels mark which rows are the answer, here they say how
    fast each row ran. What the two kinds have in common is all the base needs, which is why a ranking metric
    takes the base and this class adds only the one fact the base has nowhere to put.

    :attr:`h_opt` is the compile regime every row in the pool was measured under. It is part of the grouping key
    (:func:`group_measured`) rather than a description of it: ``-O1`` and ``-O3`` reorder the same candidates
    often enough that a pool spanning both would measure neither, so the regime is what makes these rows
    comparable at all. A report reads it as an axis, and it is a FIELD rather than a lookup into the packed
    ``H_opt`` column for a mechanical reason: :meth:`Group.matrix` memoizes exactly one projection, so asking
    for the single-column one right before a model asks for its own would evict and rebuild the model's
    projection for every group — two full strided copies each, on the path whose measurements made that cache
    exist in the first place."""

    # One benched latency in MICROSECONDS per row, lower = faster. Lives here rather than on the base because
    # nothing generic reads it: the metrics take plain sequences and never see a group, so a label on the base
    # would be a column with no consumer and two mutually exclusive meanings.
    latency_us: np.ndarray = field(kw_only=True, repr=False)
    h_opt: float = field(kw_only=True)

    @classmethod
    def from_measured(
        cls, key: str, gpu: str, op_sig: str, h_opt: float, latency_us: Sequence[float], feats: list[dict[str, float]]
    ) -> MeasuredGroup:
        """A pool where every row was benched, labelled with its own measured µs.

        ``op_sig`` serves as both ``name`` and ``shape`` — a measured pool IS one op, so there is no second
        identity to spell — and ``tier`` is empty, because the golden tiers are a closed report vocabulary
        about how a CASE was built and a sixth value would mean nothing to the two places that read it.

        No ``total`` either: a measured pool is not a sample of a larger enumeration, it is exactly the
        configs someone ran, and a rank against anything else would be against candidates never measured."""
        names, matrix, dynamic = pack_features(feats)
        return cls(
            key,
            op_sig,
            "",
            gpu,
            op_sig,
            dynamic,
            names,
            matrix,
            len(matrix),
            latency_us=np.asarray(latency_us, dtype=float),
            h_opt=h_opt,
        )


@dataclass(frozen=True)
class GoldenGroup(Group):
    """A pool whose labels mark the rows a GOLDEN verified — the fit's supervision.

    A subclass rather than a flag on :class:`Group` because the two kinds differ in what can be ASKED of
    them, not merely in what their numbers mean. Only here is "which rows are the answer" a question with an
    answer, so only here is :attr:`golden_ids` defined, and the rank metrics that take it say what they need
    by taking this type. A discriminator field would have moved that check to runtime and to every caller.
    """

    # The rows a golden verified, ascending and deduplicated — the index set the rank metrics take. Several
    # goldens can land on one pool (the same shape recorded twice, or under two names), so this is a SET.
    # Deploy ships one config, so any of them ranked first is a win, which is why the fit's per-group term is
    # the best rank over it and the reported one the matching dual rank. At one golden both collapse to the
    # single-golden functions exactly. Stored as the indices themselves: the metrics take indices, so a
    # per-row marker column would only be an encoding to decode back on every read.
    golden_ids: tuple[int, ...] = field(kw_only=True)

    @classmethod
    def from_dicts(
        cls,
        key: str,
        name: str,
        tier: str,
        gpu: str,
        shape: str,
        goldens: int | Sequence[int],
        feats: list[dict[str, float]],
        total: int | None = None,
    ) -> GoldenGroup:
        """Pack per-row feature dicts and mark the rows the goldens verified. ``goldens`` is one row index or
        several; a bare int is accepted because most callers have exactly one and wrapping it would say
        nothing. ``total`` is the size of the pool ``feats`` was drawn from, defaulting to "nothing was
        sampled" — see :attr:`Group.total`."""
        return cls.over(key, name, tier, gpu, shape, pack_features(feats), goldens, total)

    @classmethod
    def over(
        cls,
        key: str,
        name: str,
        tier: str,
        gpu: str,
        shape: str,
        packed: tuple[tuple[str, ...], np.ndarray, bool],
        goldens: int | Sequence[int],
        total: int | None = None,
    ) -> GoldenGroup:
        """The same over an already-packed pool — the entry point for a builder that had to pack before it
        could know which goldens landed in it, which is every builder that deduplicates pools. ``goldens`` is
        one row index or several, in any order, with duplicates; what is stored is the sorted set.

        ``tier`` must agree with the routing stamp the rows carry, and disagreeing is a hard error. The two
        reach here by different routes — the stamp through the featurizer, the tier from the source record's
        own flag — and they are the same fact, so a mismatch means one of them is wrong and this pool would
        otherwise train and be scored under the wrong weight set with nothing reporting it. That is what
        keeps ``tier``, a label that decides nothing, honest. A measured pool has no second route to check
        against and no tier to check, which is why the check lives here and not on the base."""
        _, matrix, dynamic = packed
        if dynamic != (tier == "dyn"):
            raise ValueError(
                f"{key}: the routing stamp says dynamic={dynamic} but the case tier says {tier!r} — "
                f"the source record's flag and its featurized rows disagree about the weight set"
            )
        rows = (goldens,) if isinstance(goldens, int) else goldens
        ids = tuple(sorted({int(i) for i in rows}))
        return cls(key, name, tier, gpu, shape, dynamic, packed[0], matrix, len(matrix) if total is None else total, golden_ids=ids)


def kernel_sig(feats: dict) -> str:
    """The op signature of the kernel a row actually measured, digested from its own ``S_*`` stamps.

    Exactly what :meth:`~...passes.identity.Identity.op_sig` computes for an op, applied to the row's
    recorded stamps instead — so this is the SAME identity, asked of the kernel that ran rather than of the
    site it came from. On the RTX 5090 freeze the two agree for 83% of rows; the rest are where the
    distinction matters."""
    return digest(*sorted((k, float(v)) for k, v in feats.items() if k.startswith("S_")))


def group_measured(rows) -> tuple[list[MeasuredGroup], dict[str, int]]:
    """Benched :class:`~..db.NodeRow`s as groups labelled with measured µs, keyed
    ``(gpu, kernel_sig, H_opt)`` — one group per set of configs that genuinely competed, plus a count of
    what was dropped and why.

    Takes ``NodeRow``s rather than :class:`Sample`s because three of the four decisions below read
    columns a ``Sample`` does not carry. ``load_node_rows`` accepts a live tune DB or a measurement
    freeze, so both sources reach this the same way.

    Each part of the key is load-bearing, and each has a plausible wrong answer:

    - **The KERNEL's own signature, not the offer site's** (:func:`kernel_sig`, the digest the row's
      recorded ``op_sig`` column holds when the two agree, which is 83% of the RTX 5090 freeze). Two kernels
      of the same structure on the same card are ONE tuning problem whatever produced them — which is
      already how the deploy path joins evidence (``Prior.evidence_pick`` and
      ``policy/greedy._db_measured_pick`` both index on the ``S_*`` signature), so this makes the
      candidate pools agree with the tier that consumes them.

      Keying on the recorded ``op_sig`` instead gets it wrong in both directions, and the freeze shows
      both. It **over-merges**: ``op_sig`` digests the PRE-DESCENT offer op, and a site realized as several
      kernels can leave one piece filed as a rival of the whole — nine pools paired a fused
      ``rms_norm``->linear megakernel with a row for a single kernel of the same op's unfused realization,
      a 5.9 µs norm kernel against a 131 ms whole-op row. And it **fragments**: 73 structures were searched
      in two separate pools, the losing pool's best landing a median 1.46x behind the winning pool's
      (p90 3.89x, worst 14x) — the same kernel tuned twice, once badly, because the two arrived from
      different sites.

      Measured against ``op_sig``: 336 pools rather than 401, but MORE rows sitting beside a rival
      (3778 of 3817, against 3760) and a median pool of 7 rather than 5. Pool count falls because merging
      is the point; what a metric needs is rivals per row.
    - **``H_opt`` from the features, never ``context_key``.** (Present and numeric by then: the
      admission filter has already rejected a row missing its ``H_*`` stamps.) A live DB spells the regime as
      ``digest(Context, cap, flags)`` and a freeze spells the same one ``capX.Y-OZ``, so grouping on
      ``context_key`` would key one regime two ways depending on which side of the loader the rows
      came from. It looks like the first-class column; that is the trap. The regimes must not pool
      either way — ``-O1`` and ``-O3`` invert often enough that a merged group measures neither.
    - **``gpu``.** Cards never pool.

    ONE rule is this function's own: **``bench_fail`` excluded.** ``freeze_reason`` deliberately keeps
    failures, as durable negative examples; a pool being ranked by measured latency wants them gone,
    because the watchdog sentinel is a huge positive that any model gets right for free and that
    inflates every correlation over the pool.

    The counts are returned rather than logged because a report has to publish them: "Spearman 0.6 over
    340 groups" means something different when 143 rows were dropped than when none were. Reasons are
    keyed by their leading clause, the same normalization ``write_freeze`` counts its own drops by, so
    the two publish one vocabulary."""
    dropped: dict[str, int] = defaultdict(int)
    buckets: dict[tuple, list] = defaultdict(list)
    for r in rows:
        reason = freeze_reason(r)
        if reason is not None:
            dropped[reason.split(":")[0]] += 1
        elif r.status != "ok":
            dropped[r.status] += 1
        elif not r.op_sig or not r.gpu:
            dropped["unkeyed"] += 1
        else:
            buckets[(r.gpu, kernel_sig(r.features), r.features["H_opt"])].append(r)

    groups = []
    for (gpu, sig, h_opt), grp in sorted(buckets.items()):
        feats = [knob_features(r.features) for r in grp]
        groups.append(MeasuredGroup.from_measured(f"{gpu}/{sig}@O{h_opt:g}", gpu, sig, h_opt, [r.value_us for r in grp], feats))
    return groups, dict(dropped)
