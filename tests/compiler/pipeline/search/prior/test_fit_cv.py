"""The ``emmy fit`` cross-validation harness (``search/prior/fit/cv``) and run harness
(``search/prior/fit/run``): shape-grouped fold partitioning, the fittability-exclusion
guard, dual-rank tie semantics, several positives per candidate pool, metrics-file
determinism, the golden group builder's pool merge, and the trainer-callable seam
— all on synthetic groups, no tracing, no GPU."""

import argparse
import json
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from emmy.commands import fit as fit_cmd
from emmy.commands.fit import TRAINERS, build_golden_groups, register_fit_command
from emmy.compiler.pipeline.search import features
from emmy.compiler.pipeline.search.data.group import DEFAULT_FEATURES, MATMUL_FEATURES, GoldenGroup, feature_view
from emmy.compiler.pipeline.search.pool import Candidates
from emmy.compiler.pipeline.search.prior.fit import LinearFit, LinearTrainer
from emmy.compiler.pipeline.search.prior.fit import cv as fit_cv
from emmy.compiler.pipeline.search.prior.fit.catboost import TREE_FEATURES
from emmy.compiler.pipeline.search.prior.fit.run import run_fit
from emmy.compiler.pipeline.search.prior.linear_model import LinearModel, descent_cols

# --- feature view ------------------------------------------------------------------


def test_default_feature_view_keeps_the_geometry_and_atom_features():
    """The default spec keeps the ``D_``-prefixed geometry features plus the two atom features
    that vary within a candidate pool — ``MMA_tier`` and ``MMA_acc_bits``, the f16-vs-f32
    accumulate discriminator whose absence made every fast-math candidate a feature twin of its
    f32 sibling. Everything else (the shape/hardware pass-throughs) is constant within a pool."""
    keep = feature_view(DEFAULT_FEATURES)
    sample = {"D_waves": 1.0, "D_bk_gap": 2.0, "MMA_tier": 3.0, "MMA_acc_bits": 4.0, "MMA_atom_m": 5.0, "S_ext_free_prod": 6.0, "D_": 7.0}
    assert {k for k in sample if keep(k)} == {"D_waves", "D_bk_gap", "D_", "MMA_tier", "MMA_acc_bits"}


def test_matmul_view_is_exact_names_through_the_one_parser():
    """The matmul view is an ordinary :func:`feature_view` spec — no separate filtering path — and it
    is spelled in EXACT names, which is what lets it exclude a feature whose name extends a kept one.
    Its exclusions are the two provably-free classes: features constant within every matmul pool (a
    term that cancels out of a within-pool ranking) and globally affine duplicates of a kept feature
    (a scaled copy that only splits one weight in two)."""
    names = MATMUL_FEATURES.split(",")
    assert len(names) == len(set(names)) and names == sorted(names)
    keep = feature_view(MATMUL_FEATURES)
    assert keep("D_cells") and not keep("D_cells_cap")  # exact, not prefix: the reason globs won't do
    for dropped in ("D_pow2_threads", "D_raster_gn", "D_stage_split", "S_ext_free_prod"):
        assert not keep(dropped), f"{dropped} cannot move a matmul ranking and must stay out"
    for dupe in ("D_bn_band", "D_bn_ge_bm", "D_cells_cap", "MMA_tier", "MMA_atom_m"):
        assert not keep(dupe), f"{dupe} is an affine copy of a kept feature"
    assert keep("MMA_acc_bits") and keep("MMA_a_bits")  # the atom's only non-redundant levels


def test_feature_view_globs_and_names():
    keep = feature_view("MMA_*, D_waves")
    assert keep("MMA_tier") and keep("MMA_acc_bits") and keep("D_waves")
    assert not keep("D_bk_gap") and not keep("S_ext_free_prod") and not keep("MMA")


def test_a_merged_case_reports_its_positive_count():
    """A group is a candidate pool, so a merged one is one row in ``per_golden`` where two goldens used to be
    two. ``positives`` is what makes that visible: without it a metrics file with fewer groups looks like lost
    data rather than a pool that gained a second verified answer."""
    groups = [_case("m.512", "warp", "gpuA", pinned=1), _case("m.1024", "warp", "gpuA", pinned=(0, 2))]
    model = LinearFit(LinearModel(weights={"D_a": -1.0}, weights_dynamic=None, scale=0.1, **SEED_PARAMS), [0], None)
    rows = fit_cv.evaluate_full_train(groups, model)["per_golden"]
    assert rows["gpuA/m.512"]["positives"] == 1
    assert rows["gpuA/m.1024"]["positives"] == 2
    # The merged group scores on its BEST positive: D_a descends, so row 0 wins and the pool ranks top-1.
    assert rows["gpuA/m.1024"]["rank"] == 0 and rows["gpuA/m.512"]["rank"] == 1
    out = fit_cv.run_folds(groups, trainer=FOLD_TRAINER, k=2)
    assert out["holdout_per_golden"]["gpuA/m.1024"]["positives"] == 2
    assert out["train_per_golden"]["gpuA/m.1024"]["positives"] == 2


# --- the golden group builder's pool merge ------------------------------------------


class _StubRecord:
    """The slice of a ``GoldenRecord`` the group builder reads, with the candidate pool handed in directly
    instead of traced. ``knobs`` / the pool rows are single-token dicts the stub signature below reads.

    The builder groups on ``GoldenRecord.pool_key``, so a stub that hands its pool in must answer that
    question consistently with the pool it hands in, or it is describing a world that cannot happen.
    ``program`` therefore defaults to the pool's own content, and is passed explicitly only to build the two
    shapes that matter: the FAST_MATH one (one program, two pin sets, two enumerations) and the re-recorded
    one (two programs, one pool, which only the packed matrix can see)."""

    def __init__(self, name, knobs, rows, *, pins=(), program=None):
        self.name, self.knobs, self.target_program = name, knobs, rows
        self.gpu_name, self.compute_cap = "gpuA", (12, 0)
        self.pool_key = (repr(rows if program is None else program), tuple(pins))
        self.pin_key, self.pin_map = tuple(pins), dict(pins)
        self.structural_features, self.dynamic = {}, False
        self.shape_key = SimpleNamespace(kind="", is_warp=True)


@dataclass(frozen=True)
class _StubContext:
    """A dataclass because the builder ``replace``s it to attach a sample - the same shape the real
    Context has, so the plumbing under test is the real plumbing."""

    pool_sample: object = None

    @classmethod
    def from_target(cls, cap, gpu_name=None):  # noqa: ARG003 — the stub needs no card facts
        return cls()

    def features(self):
        return {}


def _build(records, monkeypatch, **kwargs):
    """``build_golden_groups`` over stub records: the pool arrives as the record's ``target_program``, a row
    is ``{"TILE": tag}``, and its one feature is the tag's position in the alphabet — enough for two pools to
    featurize differently whenever their rows differ. The stub enumerator honours ``ctx.pool_sample``: a list
    IS a space as far as the draw is concerned (it asks only for a length and a member)."""

    def enumerate_stub(rows, ctx):
        sample = ctx.pool_sample
        return Candidates(rows if sample is None else sample.take(rows), len(rows))

    monkeypatch.setattr(fit_cmd, "GOLDEN_RECORDS", records)
    monkeypatch.setattr(fit_cmd, "Context", _StubContext)
    monkeypatch.setattr(fit_cmd, "enumerate_graph", enumerate_stub)
    monkeypatch.setattr(fit_cmd, "_shape_group", lambda g: "S(free=512,red=512)")  # noqa: ARG005
    monkeypatch.setattr(fit_cmd.features, "tile_signature", lambda knobs: knobs["TILE"])
    monkeypatch.setattr(fit_cmd.features, "knob_features", lambda row: {"D_a": float(ord(row["TILE"][0]))})
    return build_golden_groups(**kwargs)


def test_builder_merges_goldens_that_share_a_recorded_program_and_only_those(monkeypatch):
    """Membership is decided by what the RECORD says the enumeration will read — the program, the target,
    the bindings and the pins — never by the name and never by a derived summary of the shape.

    The name is the obvious wrong answer: most same-name duplicates are ``FAST_MATH`` siblings, one program
    under two pin sets, and the fast-math enumeration offers an atom the standard one never emits — so
    merging on the name would pin row indices that do not exist in the other pool. Here ``m.512`` is recorded
    twice against ONE program with different pins, and only the two goldens whose provenance really does
    agree merge."""
    std = [{"TILE": "a"}, {"TILE": "b"}, {"TILE": "c"}]
    fastmath = [*std, {"TILE": "f"}]  # the extra atom row the standard enumeration never emits
    groups, skipped = _build(
        [
            _StubRecord("m.512", {"TILE": "b"}, std),
            _StubRecord("m.512.alias", {"TILE": "c"}, std),  # a different name, same provenance: merges
            # Same name AND same program, but pinned differently — so a different enumeration, its own group.
            _StubRecord("m.512", {"TILE": "f"}, fastmath, program=std, pins=(("FAST_MATH", True),)),
            _StubRecord("m.512.absent", {"TILE": "zz"}, std),  # matches no row: skipped, as before
        ],
        monkeypatch,
    )
    assert [(c.key, c.golden_ids, len(c.feats)) for c in groups] == [("gpuA/m.512", (1, 2), 3), ("gpuA/m.512#2", (3,), 4)]
    assert skipped == [("gpuA", "m.512.absent", "golden not in 3 candidates")]
    # The ``#N`` suffix is spent only where a key would otherwise collide, so the merged group keeps the plain
    # key that ``cv.run_folds`` accumulates train ranks under.
    assert len({c.key for c in groups}) == len(groups)


def test_two_goldens_on_one_pool_still_merge_under_sampling(monkeypatch):
    """Sampling must not break the merge, and the merge is what makes two verified answers to one
    question one training group instead of two.

    The two are one group before the draw happens — they share a program, so they share an enumeration and
    the builder runs it once. What sampling must not break is the PINS: both recorded rows have to survive
    the draw, which they do because it is a pure function of ``(pool size, sample, seed)`` and both goldens
    bucket onto the same keep-set. Neither was picked by the draw itself: the pool's first and last rows are
    exactly the positions a 4-of-26 draw is least likely to reach."""
    pool = [{"TILE": chr(ord("a") + i)} for i in range(26)]
    groups, skipped = _build(
        [
            _StubRecord("m.512", {"TILE": "a"}, list(pool)),  # the FIRST row
            _StubRecord("m.512.alias", {"TILE": "z"}, list(pool)),  # and the LAST
        ],
        monkeypatch,
        sample=4,
    )
    assert skipped == []
    assert len(groups) == 1, "one pool, one group - the draw must not fracture it into two"
    (group,) = groups
    assert len(group.golden_ids) == 2, "both recorded rows survive the draw and land in the group"
    assert group.total == 26 and len(group.feats) < 26, "the true pool size travels beside the sample"
    kept = {chr(int(v)) for v in group.feats[:, 0]}
    assert {"a", "z"} <= kept


def test_builder_folds_away_a_golden_recorded_twice_at_one_config(monkeypatch):
    """Two recordings of the same config over the same pool are ONE fact. They verify the same row, so the
    label set does not grow — where the ``#2`` group they used to become counted that one fact twice in every
    metric."""
    std = [{"TILE": "a"}, {"TILE": "b"}]
    groups, _ = _build([_StubRecord("m.512", {"TILE": "b"}, std), _StubRecord("m.512", {"TILE": "b"}, std)], monkeypatch)
    assert [(c.key, c.golden_ids) for c in groups] == [("gpuA/m.512", (1,))]


def test_builder_folds_two_recordings_of_one_program_that_key_apart(monkeypatch):
    """The reason the packed pool decides membership and the recorded key only groups the work.

    A program recorded in two sessions carries two different wires — different node ids, same graph — so it
    reads as two enumerations and is enumerated twice. The pools they produce are identical, and the second
    stage folds them. Without it the V100 corpus, where the same shapes are recorded many times over, reports
    306 groups where there are 108 pools, each one counting a single fact on its own."""
    std = [{"TILE": "a"}, {"TILE": "b"}, {"TILE": "c"}]
    groups, _ = _build(
        [
            _StubRecord("m.512", {"TILE": "b"}, std),
            # Same pool, but recorded as a different program — the recorded key cannot see they are one.
            _StubRecord("m.512.resession", {"TILE": "c"}, std, program=[{"same-graph": "other-ids"}]),
        ],
        monkeypatch,
    )
    assert [(c.key, c.golden_ids) for c in groups] == [("gpuA/m.512", (1, 2))]


def test_a_pool_is_named_after_a_golden_that_is_actually_in_it(monkeypatch):
    """A record can be grouped onto a pool and then fail to find its own row in it. Naming the group after
    that record would put a rank in ``metrics.json`` under a name the same run reports as skipped — and the
    golden whose row IS pinned would appear nowhere."""
    std = [{"TILE": "a"}, {"TILE": "b"}]
    groups, skipped = _build(
        [_StubRecord("m.512.absent", {"TILE": "zz"}, std), _StubRecord("m.512", {"TILE": "b"}, std)],
        monkeypatch,
    )
    assert [(c.key, c.name) for c in groups] == [("gpuA/m.512", "m.512")]
    assert [s[1] for s in skipped] == ["m.512.absent"]


# --- synthetic groups ---------------------------------------------------------------


def _case(name, tier, gpu, pinned=1, n_rows=6, key=None, shape=None):
    """A tiny group whose rows carry a monotone D_a so a samples=0 descent has signal. EVERY group
    carries the routing stamp on every row, as the featurizer writes it (``passes/identity._extents``
    emits the key unconditionally, 0.0 when no axis is symbolic) — that stamp's VALUE, not the tier
    label, is what puts the group on the dynamic weight set.

    ``pinned`` is the pool's positive row, or several of them (a pool the builder matched more than
    one golden into). ``shape`` is the fold group; it defaults to the name with any ``.dynM`` suffix
    stripped, which mirrors what the builder does for real: a dynamic golden enumerates its static
    twin's pool, so the twins share a group."""
    stamp = {"S_ext_n_symbolic_axis": 1.0 if tier == "dyn" else 0.0}
    feats = [{"D_a": float(i), "D_b": float((i * 7) % 3), **stamp} for i in range(n_rows)]
    shape = shape or name.removesuffix(".dynM")
    return GoldenGroup.from_dicts(key or f"{gpu}/{name}", name, tier, gpu, shape, pinned, feats)


def _cases():
    return [
        _case("matmul.square.512", "thread", "gpuA"),
        _case("matmul.square.1024", "thread", "gpuA", pinned=2),
        _case("matmul.qkv.h4096", "warp", "gpuA", pinned=0),
        _case("matmul.qkv.h4096.dynM", "dyn", "gpuA", pinned=3),
        _case("matmul.square.512", "thread", "gpuB"),
        _case("matmul.square.512.dynM", "dyn", "gpuB", pinned=2),
        _case("matmul.qkv.h4096", "warp", "gpuB", pinned=4),
        _case("reduce.k2048", "reduce", "gpuB", pinned=0),
    ]


# --- weight-set routing ------------------------------------------------------------


def test_routing_stamp_selects_the_weight_set_and_never_becomes_a_coordinate():
    """``S_ext_n_symbolic_axis`` picks the weight set, is PACKED like any other column, and is narrowed
    out by the linear model class rather than by the dataset.

    A weight on it would be noise — the stamp is constant across a pool, so a term on it shifts every
    candidate equally and cancels out of the within-shape ranking, leaving its value to the regularizer.
    That is a fact about additive models, though, not about the data: a tree splits on the regime
    happily. So the dataset packs the column and :func:`descent_cols` withholds it from the descent,
    which is what keeps the two model classes from constraining each other."""
    static, dyn = _case("m.512", "warp", "gpuA"), _case("m.512.dynM", "dyn", "gpuA")
    assert (static.dynamic, dyn.dynamic) == (False, True)
    # Packed, and carrying the real stamp — not a name the matrix cannot fill. A NaN column here rather
    # than 1.0 is the train/serve skew this replaced: live candidates always carry the value.
    assert "S_ext_n_symbolic_axis" in dyn.feat_names
    assert (dyn.matrix(["S_ext_n_symbolic_axis"], fill=np.nan) == 1.0).all()
    for group in (static, dyn):
        assert group.feats.shape[1] == len(group.feat_names)
        assert "S_ext_n_symbolic_axis" not in descent_cols(group.feat_names)
    # The tier decides nothing, but it still has to AGREE: it and the stamp are the same fact arriving
    # by two routes, so a disagreement means one of them is wrong and the pool would otherwise train
    # under the wrong weight set with nothing reporting it.
    with pytest.raises(ValueError, match="disagree about the weight set"):
        GoldenGroup.from_dicts("gpuA/x", "x", "warp", "gpuA", "x", 0, [{"D_a": 1.0, "S_ext_n_symbolic_axis": 1.0}])
    with pytest.raises(ValueError, match="disagree about the weight set"):
        GoldenGroup.from_dicts("gpuA/x", "x", "dyn", "gpuA", "x", 0, [{"D_a": 1.0}])


def test_a_routing_feature_may_never_carry_a_fitted_weight():
    """The guard that replaced the structural one. While the dataset withheld the routing column, no descent
    could reach it; now that every column is packed, the rule is enforced where a violation would do harm —
    a routing weight in a shipped artifact becomes an additive term on every live candidate. Checking at
    construction catches every route in, a hand-edited artifact included."""
    for field in ("weights", "weights_dynamic"):
        sets = {"weights": {"D_a": 1.0}, "weights_dynamic": {"D_a": 1.0}}
        sets[field] = {**sets[field], "S_ext_n_symbolic_axis": 0.5}
        with pytest.raises(ValueError, match="never contributes a term"):
            LinearModel(scale=0.1, atomic_free_weight=0.0, atomic_free_split_threshold=4.0, **sets)


def test_matrix_fill_declares_the_absent_semantics():
    """``GoldenGroup`` stores absent features as ``NaN`` and each model class projects them to what IT means by
    absent. Both kinds of absence are covered: a name the pool never stamped, and a key missing from one
    row whose siblings carry it.

    The ``fill=0.0`` default must stay bit-identical to ``feats.get(k, 0.0)`` — that is the linear model's
    contract, and the whole dataset is packed once for both trainers."""
    g = GoldenGroup.from_dicts("gpuA/x", "x", "warp", "gpuA", "x", 0, [{"D_a": 1.0, "D_b": 2.0}, {"D_a": 3.0}])
    zeros = g.matrix(["D_a", "D_b", "D_never"])
    assert zeros.tolist() == [[1.0, 2.0, 0.0], [3.0, 0.0, 0.0]]

    nans = g.matrix(["D_a", "D_b", "D_never"], fill=np.nan)
    assert nans[0].tolist() == [1.0, 2.0] + [pytest.approx(np.nan, nan_ok=True)]
    assert nans[1][0] == 3.0 and np.isnan(nans[1][1]) and np.isnan(nans[1][2])
    # A genuine 0.0 is never confused for an absent value under either fill.
    z = GoldenGroup.from_dicts("gpuA/z", "z", "warp", "gpuA", "z", 0, [{"D_a": 0.0}])
    assert z.matrix(["D_a"], fill=np.nan).tolist() == [[0.0]]


def test_matrix_is_memoized_and_read_only():
    """Building the projection is a strided copy of the whole pool, and cross-validation asks for the
    same one over and over — measured, those copies were 85% of a fit's wall time. So it is built once
    and shared, which is only safe if it cannot be mutated underneath another holder: a caller that
    needs to write (the linear fitter z-scores in place) copies inside itself.

    One entry is enough — a fit uses one column list for its whole run — so a different request evicts
    rather than accumulating."""
    g = GoldenGroup.from_dicts("gpuA/x", "x", "warp", "gpuA", "x", 0, [{"D_a": 1.0, "D_b": 2.0}, {"D_a": 3.0}])
    first = g.matrix(["D_a", "D_b"])
    assert g.matrix(["D_a", "D_b"]) is first, "a repeat request must not rebuild the projection"
    assert not first.flags.writeable
    with pytest.raises(ValueError):
        first[0, 0] = 99.0
    # ... and copying is what a mutating caller does, exactly as fit_weights now does.
    mine = g.matrix(["D_a", "D_b"]).copy()
    mine[0, 0] = 99.0
    assert g.matrix(["D_a", "D_b"])[0, 0] == 1.0, "the shared projection survived a caller's private copy"
    # A different (names, fill) evicts, and the evicted request rebuilds correctly rather than
    # returning the wrong shape.
    other = g.matrix(["D_a"], fill=np.nan)
    assert other.shape == (2, 1) and g.matrix(["D_a", "D_b"]).shape == (2, 2)


def test_no_feature_view_can_drop_the_routing_stamp():
    """Routing is not a view choice. A spec that names neither the stamp nor a prefix covering it still
    keeps it — otherwise every pool would route to the static weight set and the run would report a
    dataset with zero dynamic groups rather than an error."""
    for spec in (DEFAULT_FEATURES, MATMUL_FEATURES, "D_waves"):
        assert feature_view(spec)("S_ext_n_symbolic_axis"), spec
    assert not feature_view("D_waves")("S_ext_free_prod")  # only the routing features are exempt


def test_tree_view_drops_only_what_a_tree_re_derives():
    """The tree view keeps the raw columns and the terms axis-aligned splits cannot reach, and drops the
    engineered ones a tree forms for itself. Exclusions never touch the default view."""
    keep, default = feature_view(TREE_FEATURES), feature_view(DEFAULT_FEATURES)
    derivable = ("D_l2_threads", "D_near_threads", "D_square", "D_stage_prefetch", "D_splitk_le2", "D_tma_aspect", "D_l2_cells_occ")
    for name in derivable:
        assert not keep(name), f"{name} is derivable from a kept column by splits"
        assert default(name), f"{name} must stay in the linear view, which cannot form it"
    # Kept: the raw columns, and the terms no split on a kept column can reach — a periodic predicate,
    # a relation BETWEEN two columns, and the knob x state block whose state operand is not a column.
    for name in ("D_threads", "D_cells", "D_aspect", "D_stage_depth", "D_splitk", "D_pow2_threads", "D_bn_ge_bm", "D_splitk_excess"):
        assert keep(name), name
    assert keep("S_ext_n_symbolic_axis"), "the routing stamp is exempt from every view, exclusions included"


def test_feature_view_exclusions_apply_to_names_and_globs():
    keep = feature_view("D_*,MMA_tier,-D_near_*,-MMA_tier")
    assert keep("D_threads") and not keep("D_near_area") and not keep("MMA_tier")


def test_featurizer_preserves_the_routing_stamp():
    """The stamp survives ``knob_features`` — the one step between a golden's structural features and
    the row the fit packs. Without it the whole dynamic weight set would silently stop being reached."""
    assert features.knob_features({"S_ext_n_symbolic_axis": 1.0, "S_ext_free_prod": 4096.0})["S_ext_n_symbolic_axis"] == 1.0
    assert "S_ext_n_symbolic_axis" not in features.knob_features({"S_ext_free_prod": 4096.0})


NAMES = ["D_a", "D_b"]
# The fitted scalar params seed OFF at the shipped threshold (the atomic-free interaction).
SEED_PARAMS = {"atomic_free_weight": 0.0, "atomic_free_split_threshold": 4.0}


# The fold trainer the command layer wires up: zero-seeded weights, samples=0. One instance serves
# every fold — the trainer is immutable and its fit is pure.
FOLD_TRAINER = LinearTrainer(
    feature_names=tuple(NAMES),
    init=LinearModel(weights={}, weights_dynamic=None, scale=0.1, **SEED_PARAMS),
    samples=0,
    warm_start=False,
)


# --- fold partitioning + pooling ---------------------------------------------------


def test_folds_hold_out_every_golden_exactly_once():
    out = fit_cv.run_folds(_cases(), trainer=FOLD_TRAINER, k=3)
    assert set(out["holdout_per_golden"]) == {c.key for c in _cases()}
    for row in out["holdout_per_golden"].values():
        assert isinstance(row["rank"], int) and isinstance(row["rank_optimistic"], int)
        assert row["rank"] >= row["rank_optimistic"]  # pessimistic can only add ties
    # Train side: same keys (every group was in the other folds' training slices).
    assert set(out["train_per_golden"]) == {c.key for c in _cases()}
    # Aggregates are per card only — a REPORT axis, independent of what the folds group by — and the
    # gap is their arithmetic difference.
    for gpu in ("gpuA", "gpuB"):
        summaries = {(c["axes"]["cv_split"], c["axes"]["gpu"]): c for c in out["summaries"]}
        assert ("holdout", gpu) in summaries and ("train", gpu) in summaries
        assert out["gap"][gpu] == round(
            summaries[("holdout", gpu)]["metrics"]["rank"]["median"] - summaries[("train", gpu)]["metrics"]["rank"]["median"], 2
        )
    assert out["fold_detail"]["excluded"] == {}


def test_a_shape_group_is_never_split_across_folds():
    """THE leakage guard, and the reason this fitter folds by shape at all.

    Goldens sharing an extent identity enumerate the same candidate pool. Hold one out while training
    on another and the fold model has already been shown the answer, so its "holdout" rank is not a
    holdout rank. The retired ``op_family`` axis keyed on the golden's NAME and missed exactly this:
    on the real corpus, 178 shape groups spanned more than one family, covering 695 of 1385 goldens.

    The groups below are that situation in miniature — one shape under three unrelated names, on two
    different cards, which must still land in ONE fold."""
    groups = [
        _case("rms_norm.k2048", "reduce", "gpuA", shape="S(free=2048,red=2048)"),
        _case("gemma4_12b.rms_norm", "reduce", "gpuA", shape="S(free=2048,red=2048)"),
        _case("olmoe_1b7b.rms_norm.k2048.m1", "reduce", "gpuB", shape="S(free=2048,red=2048)"),
        _case("matmul.square.512", "thread", "gpuA", shape="S(free=512,red=512)"),
        _case("matmul.square.1024", "thread", "gpuB", shape="S(free=1024,red=1024)"),
    ]
    by_shape = fit_cv.assign_folds(groups, 3)
    folds = {c.key: by_shape[c.shape] for c in groups}
    shared = [k for k in folds if "rms_norm" in k]
    assert len({folds[k] for k in shared}) == 1, "one shape, one fold — on every card"

    out = fit_cv.run_folds(groups, trainer=FOLD_TRAINER, k=3)
    held = out["holdout_per_golden"]
    assert len({held[k]["fold"] for k in shared}) == 1


def test_folds_are_balanced_and_deterministic():
    """Groups are very uneven on the real corpus (largest 122 groups, 162 singletons), so assignment is
    largest-first onto the currently-smallest fold; a thin fold's median would be noise. Assignment is
    a pure function of the group list, so a re-run reproduces it."""
    groups = [_case(f"m.{i}", "thread", "gpuA", shape=f"s{i // 2}") for i in range(12)]  # 6 groups of 2
    a = fit_cv.assign_folds(groups, 3)
    assert a == fit_cv.assign_folds(list(reversed(groups)), 3)
    counts = {f: sum(1 for c in groups if a[c.shape] == f) for f in set(a.values())}
    assert set(counts) == {0, 1, 2} and max(counts.values()) - min(counts.values()) <= 1


def test_more_folds_than_groups_leaves_no_empty_fold_scored():
    """k above the group count is not an error — the surplus folds simply hold nothing and are
    skipped, rather than being scored as empty holdouts."""
    groups = [_case("m.1", "thread", "gpuA", shape="s1"), _case("m.2", "thread", "gpuA", shape="s2")]
    out = fit_cv.run_folds(groups, trainer=FOLD_TRAINER, k=5)
    assert set(out["holdout_per_golden"]) == {c.key for c in groups}
    assert all(d["n"] > 0 for d in out["fold_detail"]["holdout_medians"].values())


def test_fittability_guard_excludes_fold_loudly():
    """A fold whose holdout needs the dynamic set while its training slice has no dyn
    groups is excluded with a reason — never scored with a stale/empty vector — and its
    goldens are visibly missing from the pooled holdout."""
    groups = [
        _case("matmul.square.512", "thread", "gpuA"),
        _case("matmul.qkv.h4096", "warp", "gpuA", pinned=0),
        _case("matmul.mlp_down.h4096.dynM", "dyn", "gpuA", pinned=3),  # the ONLY dyn group
        _case("matmul.square.512", "thread", "gpuB"),
    ]
    out = fit_cv.run_folds(groups, trainer=FOLD_TRAINER, k=4)
    assert list(out["fold_detail"]["excluded"].values()) == ["dynamic weight set unfittable (0 dyn groups in training)"]
    assert "gpuA/matmul.mlp_down.h4096.dynM" not in out["holdout_per_golden"]
    # The healthy folds still pool.
    assert "gpuA/matmul.square.512" in out["holdout_per_golden"]


# --- metrics assembly: determinism + skip accounting -------------------------------


def _metrics():
    groups = _cases()
    model = LinearFit(LinearModel(weights={"D_a": -1.0, "D_b": 0.25}, weights_dynamic={"D_a": -0.5}, scale=0.1, **SEED_PARAMS), [0], [0])
    cv = fit_cv.run_folds(groups, trainer=FOLD_TRAINER, k=3)
    skipped = [
        ("gpuA", "attention.hd128", fit_cv.OUT_OF_SCOPE),
        ("gpuA", "matmul.o_proj.h4096", "golden not in 12 candidates"),
        ("gpuC", "softmax.k2048", fit_cv.OUT_OF_SCOPE),  # a card with no ranked group at all
    ]
    header = {"trainer": "linear", "data": "golden", "seed": 0}
    return fit_cv.build_metrics(header, groups, skipped, model, cv)


def test_metrics_json_is_deterministic():
    a = json.dumps(_metrics(), indent=2, sort_keys=True)
    b = json.dumps(_metrics(), indent=2, sort_keys=True)
    assert a == b


def test_metrics_counts_every_skipped_golden():
    m = _metrics()
    assert all(c["axes"]["cv_split"] == "full_train" for c in m["full_train"]["summaries"])
    # Skipped goldens sit BESIDE the summaries: they have no pool, so they are a fact about the corpus rather
    # than about a scored card — and a fit summary stays the same four-key shape an eval report emits.
    assert set(m["full_train"]["summaries"][0]) == {"axes", "groups", "unscored", "metrics"}
    assert m["full_train"]["skipped"]["gpuA"] == {"unranked": 1, "out_of_scope": 1}
    assert m["full_train"]["skipped"]["gpuB"] == {"unranked": 0, "out_of_scope": 0}
    # A card whose every golden was skipped still appears — as a skipped entry with no summary.
    assert m["full_train"]["skipped"]["gpuC"]["out_of_scope"] == 1
    assert "gpuC" not in {c["axes"]["gpu"] for c in m["full_train"]["summaries"]}
    # full_train per-golden rows carry the pool size; ranks respect the tie ordering.
    row = m["full_train"]["per_golden"]["gpuA/matmul.square.512"]
    assert row["pool"] == 6 and row["rank"] >= row["rank_optimistic"]


# --- run harness: the trainer-callable seam ----------------------------------------


class _StubModel:
    """A minimal fitted model — enough to prove ``run_fit`` needs nothing from the linear model
    class beyond ``score_rows``."""

    def __init__(self, w):
        self.w = w

    def score_rows(self, group):
        return group.matrix(NAMES) @ np.array([self.w, 0.0])


class _StubTrainer:
    """The trainer seam: one object with ``fit(groups) -> model``, reused across every fold."""

    def __init__(self, w):
        self.w = w

    def fit(self, groups):  # noqa: ARG002 — a stub ignores the data
        return _StubModel(self.w)


def _run_stub_fit():
    return run_fit(
        _cases(),
        [("gpuC", "softmax.k2048", fit_cv.OUT_OF_SCOPE)],
        trainer=_StubTrainer(-1.0),
        fold_trainer=_StubTrainer(-0.5),
        folds=3,
        header={"trainer": "stub", "seed": 0},
    )


def test_run_fit_stub_trainer_deterministic():
    """``run_fit`` is a pure function of its inputs and knows nothing about the linear model: a stub
    trainer whose models implement only ``score_rows`` yields the full metrics shape, identically on
    every call. It returns the FIT, not an artifact — assembling one is the caller's shipping policy."""
    metrics, fit = _run_stub_fit()
    assert set(metrics) == {"header", "full_train", "cv"}
    assert metrics["header"] == {"trainer": "stub", "seed": 0}
    assert metrics["full_train"]["skipped"]["gpuC"]["out_of_scope"] == 1
    assert set(metrics["cv"]["holdout_per_golden"]) == {c.key for c in _cases()}
    assert fit.w == -1.0  # the shippable model came from the full-train trainer, not a fold one
    a, b = _run_stub_fit(), _run_stub_fit()
    assert json.dumps(a[0], sort_keys=True) == json.dumps(b[0], sort_keys=True)


# --- CLI surface -------------------------------------------------------------------


def test_fit_command_defaults_and_unsupported_cells():
    parser = argparse.ArgumentParser()
    register_fit_command(parser.add_subparsers())
    args = parser.parse_args(["fit"])
    assert (args.trainer, args.data, args.samples, args.seed, args.folds) == ("linear", "golden", 0, 0, 5)
    # --artifact: absent = no extra write, bare = "" (the shipped offline_weights.json), a value = that path.
    assert args.artifact is None
    assert parser.parse_args(["fit", "--artifact"]).artifact == ""
    assert parser.parse_args(["fit", "--artifact", "/tmp/cand.json"]).artifact == "/tmp/cand.json"

    # --features defaults to the TRAINER's view, resolved in the handler rather than by argparse:
    # the linear model needs the engineered step / fold / interaction features, the tree re-derives them.
    assert args.features is None
    assert TRAINERS["linear"][1] == DEFAULT_FEATURES
    assert TRAINERS["catboost"][1] == TREE_FEATURES

    bad = parser.parse_args(["fit", "--data", "freeze:/tmp/x.jsonl"])
    with pytest.raises(SystemExit, match="not yet supported"):
        bad.func(bad)


def test_handle_fit_writes_metrics_and_a_loadable_artifact(tmp_path, monkeypatch):
    """The command layer end to end on a synthetic dataset — trainer wiring, artifact assembly and
    both output files — without tracing a single golden. ``--artifact`` is absent, so the run writes
    only into its own directory and never touches the shipped weights."""
    monkeypatch.setattr("emmy.commands.fit.build_golden_groups", lambda spec, **_kw: (_cases(), []))  # noqa: ARG005
    parser = argparse.ArgumentParser()
    register_fit_command(parser.add_subparsers())
    args = parser.parse_args(["fit", "--folds", "3", "--out", str(tmp_path / "run")])
    args.func(args)

    metrics = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert metrics["header"]["trainer_params"]["objective"] == "mean_log_rank"
    # The two seeding policies stay recorded: without them two fits are not comparable.
    assert metrics["header"]["trainer_params"]["fold_seed_weights"] == "zeros"
    assert metrics["cv"]["summaries"]
    # Cases are candidate pools now, so the header carries how many verified rows they hold between them —
    # otherwise a group count that fell because two goldens shared a pool reads as lost data.
    assert metrics["header"]["groups"] == {"total": 8, "positives": 8, "merged": 0}

    artifact = json.loads((tmp_path / "run" / "weights.json").read_text())
    assert artifact["kind"] == "linear" and artifact["weights"] and artifact["weights_dynamic"]
    assert artifact["provenance"]["groups"] == {"static": 6, "dynamic": 2} and artifact["provenance"]["positives"] == 8
    assert "static top1=" in artifact["provenance"]["notes"]
