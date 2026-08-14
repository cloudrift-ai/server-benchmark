"""The ``emmy fit`` cross-validation harness (``search/prior/fit/cv``) and run harness
(``search/prior/fit/run``): fold partitioning on both axes, the fittability-exclusion
guard, dual-rank tie semantics, metrics-file determinism, and the trainer-callable seam
— all on synthetic cases, no tracing, no GPU."""

import argparse
import json

import numpy as np
import pytest

from emmy.commands.fit import TRAINERS, register_fit_command
from emmy.compiler.pipeline.search import features
from emmy.compiler.pipeline.search.prior.fit import (
    DEFAULT_FEATURES,
    MATMUL_FEATURES,
    TREE_FEATURES,
    Group,
    LinearFit,
    LinearTrainer,
    dual_rank,
    feature_view,
    op_family,
)
from emmy.compiler.pipeline.search.prior.fit import cv as fit_cv
from emmy.compiler.pipeline.search.prior.fit.run import run_fit
from emmy.compiler.pipeline.search.prior.linear_model import LinearModel

# --- op-family derivation ----------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "family"),
    [
        ("matmul.square.512", "matmul.square"),
        ("matmul.square.512.fp16", "matmul.square"),
        ("matmul.mlp_down.h4096.dynM", "matmul.mlp_down"),
        ("gemma4_12b.q_proj_global.s2048", "gemma4_12b.q_proj_global"),
        ("gemma4_12b.o_proj", "gemma4_12b.o_proj"),
        ("reduce.k2048.dynM", "reduce"),
        ("pointwise.n16384", "pointwise"),
        ("rms_norm.k3840", "rms_norm"),
        ("attention.hd128.dynM", "attention"),
    ],
)
def test_op_family_strips_variant_segments(name, family):
    assert op_family(name) == family


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


# --- dual-rank tie semantics -------------------------------------------------------


def test_dual_rank_tie_plateau():
    """Pessimistic counts strictly-greater rows plus EARLIER ties (the deploy tiebreak);
    optimistic counts strictly-greater only — they diverge by exactly the earlier-tie
    plateau width."""
    scores = [5.0, 3.0, 5.0, 5.0, 7.0]  # golden at 2: one strict winner (7.0), one earlier tie (idx 0)
    rank, opt = dual_rank(scores, 2)
    assert (rank, opt) == (2, 1)
    assert rank - opt == 1  # the earlier-tie plateau width, not the full tie count (idx 3 is later)

    assert dual_rank(scores, 4) == (0, 0)  # the strict winner: no plateau, no gap
    assert dual_rank([1.0, 1.0, 1.0], 0) == (0, 0)  # golden emitted first wins its plateau
    assert dual_rank([1.0, 1.0, 1.0], 2) == (2, 0)  # emitted last, the whole plateau deploys first


# --- synthetic cases ---------------------------------------------------------------


def _case(name, tier, gpu, gidx=1, n_rows=6, key=None):
    """A tiny case whose rows carry a monotone D_a so a samples=0 descent has signal. A ``dyn``
    case carries the routing stamp on every row, exactly as the golden case builder writes it —
    that stamp, not the tier label, is what puts the group on the dynamic weight set."""
    stamp = {"S_ext_n_symbolic_axis": 1.0} if tier == "dyn" else {}
    feats = [{"D_a": float(i), "D_b": float((i * 7) % 3), **stamp} for i in range(n_rows)]
    return Group.from_dicts(key or f"{gpu}/{name}", name, tier, gpu, gidx, feats)


def _cases():
    return [
        _case("matmul.square.512", "thread", "gpuA"),
        _case("matmul.square.1024", "thread", "gpuA", gidx=2),
        _case("matmul.qkv.h4096", "warp", "gpuA", gidx=0),
        _case("matmul.qkv.h4096.dynM", "dyn", "gpuA", gidx=3),
        _case("matmul.square.512", "thread", "gpuB"),
        _case("matmul.square.512.dynM", "dyn", "gpuB", gidx=2),
        _case("matmul.qkv.h4096", "warp", "gpuB", gidx=4),
        _case("reduce.k2048", "reduce", "gpuB", gidx=0),
    ]


# --- weight-set routing ------------------------------------------------------------


def test_routing_stamp_selects_the_weight_set_and_never_becomes_a_coordinate():
    """``S_ext_n_symbolic_axis`` picks the weight set and is then held OUT of the fitted matrix.

    Holding it out is structural, not a feature-view choice: the stamp is constant across a pool, so
    a weight on it shifts every candidate equally and cancels out of the within-shape ranking. The
    rank objective cannot see it, which makes any value a descent lands on there noise — so it must
    not be reachable as a coordinate at all."""
    static, dyn = _case("m.512", "warp", "gpuA"), _case("m.512.dynM", "dyn", "gpuA")
    assert (static.dynamic, dyn.dynamic) == (False, True)
    for case in (static, dyn):
        assert "S_ext_n_symbolic_axis" not in case.feat_names
        assert case.feats.shape[1] == len(case.feat_names)
    # The tier decides nothing, but it still has to AGREE: it and the stamp are the same fact arriving
    # by two routes, so a disagreement means one of them is wrong and the pool would otherwise train
    # under the wrong weight set with nothing reporting it.
    with pytest.raises(ValueError, match="disagree about the weight set"):
        Group.from_dicts("gpuA/x", "x", "warp", "gpuA", 0, [{"D_a": 1.0, "S_ext_n_symbolic_axis": 1.0}])
    with pytest.raises(ValueError, match="disagree about the weight set"):
        Group.from_dicts("gpuA/x", "x", "dyn", "gpuA", 0, [{"D_a": 1.0}])


def test_matrix_fill_declares_the_absent_semantics():
    """``Group`` stores absent features as ``NaN`` and each model class projects them to what IT means by
    absent. Both kinds of absence are covered: a name the pool never stamped, and a key missing from one
    row whose siblings carry it.

    The ``fill=0.0`` default must stay bit-identical to ``feats.get(k, 0.0)`` — that is the linear model's
    contract, and the whole dataset is packed once for both trainers."""
    g = Group.from_dicts("gpuA/x", "x", "warp", "gpuA", 0, [{"D_a": 1.0, "D_b": 2.0}, {"D_a": 3.0}])
    zeros = g.matrix(["D_a", "D_b", "D_never"])
    assert zeros.tolist() == [[1.0, 2.0, 0.0], [3.0, 0.0, 0.0]]

    nans = g.matrix(["D_a", "D_b", "D_never"], fill=np.nan)
    assert nans[0].tolist() == [1.0, 2.0] + [pytest.approx(np.nan, nan_ok=True)]
    assert nans[1][0] == 3.0 and np.isnan(nans[1][1]) and np.isnan(nans[1][2])
    # A genuine 0.0 is never confused for an absent value under either fill.
    z = Group.from_dicts("gpuA/z", "z", "warp", "gpuA", 0, [{"D_a": 0.0}])
    assert z.matrix(["D_a"], fill=np.nan).tolist() == [[0.0]]


def test_no_feature_view_can_drop_the_routing_stamp():
    """Routing is not a view choice. A spec that names neither the stamp nor a prefix covering it still
    keeps it — otherwise every pool would route to the static weight set and the run would report a
    dataset with zero dynamic cases rather than an error."""
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


def test_run_axis_gpu_pools_every_golden_exactly_once():
    out = fit_cv.run_axis(_cases(), "gpu", trainer=FOLD_TRAINER)
    # Every case held out exactly once, tagged with the fold (= its own card) that never trained on it.
    assert set(out["holdout"]["per_golden"]) == {c.key for c in _cases()}
    for key, row in out["holdout"]["per_golden"].items():
        assert row["fold"] == key.split("/")[0]
        assert isinstance(row["rank"], int) and isinstance(row["rank_optimistic"], int)
        assert row["rank"] >= row["rank_optimistic"]  # pessimistic can only add ties
    # Train side: same keys (every case was in the other folds' training slices).
    assert set(out["train"]["per_golden"]) == {c.key for c in _cases()}
    # Aggregates are per card only, and the gap is their arithmetic difference.
    for gpu in ("gpuA", "gpuB"):
        assert gpu in out["holdout"]["per_card"] and gpu in out["train"]["per_card"]
        assert out["gap"][gpu] == round(out["holdout"]["per_card"][gpu]["median"] - out["train"]["per_card"][gpu]["median"], 2)
    assert set(out["fold_detail"]["holdout_medians"]) == {"gpuA", "gpuB"}
    assert out["fold_detail"]["excluded"] == {}


def test_run_axis_op_family_folds_group_variants():
    out = fit_cv.run_axis(_cases(), "op_family", trainer=FOLD_TRAINER)
    # matmul.qkv.h4096 and its .dynM variant share one fold: held out together.
    qkv = [row for key, row in out["holdout"]["per_golden"].items() if "qkv" in key]
    assert len(qkv) == 3 and {r["fold"] for r in qkv} == {"matmul.qkv"}
    assert set(out["fold_detail"]["holdout_medians"]) == {"matmul.square", "matmul.qkv", "reduce"}


def test_fittability_guard_excludes_fold_loudly():
    """A fold whose holdout needs the dynamic set while its training slice has no dyn
    cases is excluded with a reason — never scored with a stale/empty vector — and its
    goldens are visibly missing from the pooled holdout."""
    cases = [
        _case("matmul.square.512", "thread", "gpuA"),
        _case("matmul.qkv.h4096", "warp", "gpuA", gidx=0),
        _case("matmul.mlp_down.h4096.dynM", "dyn", "gpuA", gidx=3),  # the ONLY dyn case
        _case("matmul.square.512", "thread", "gpuB"),
    ]
    out = fit_cv.run_axis(cases, "op_family", trainer=FOLD_TRAINER)
    assert out["fold_detail"]["excluded"] == {"matmul.mlp_down": "dynamic weight set unfittable (0 dyn cases in training)"}
    assert "gpuA/matmul.mlp_down.h4096.dynM" not in out["holdout"]["per_golden"]
    assert "matmul.mlp_down" not in out["fold_detail"]["holdout_medians"]
    # The healthy folds still pool.
    assert "gpuA/matmul.square.512" in out["holdout"]["per_golden"]


# --- metrics assembly: determinism + skip accounting -------------------------------


def _metrics():
    cases = _cases()
    model = LinearFit(LinearModel(weights={"D_a": -1.0, "D_b": 0.25}, weights_dynamic={"D_a": -0.5}, scale=0.1, **SEED_PARAMS), [0], [0])
    cv = {"gpu": fit_cv.run_axis(cases, "gpu", trainer=FOLD_TRAINER)}
    skipped = [
        ("gpuA", "attention.hd128", fit_cv.OUT_OF_SCOPE),
        ("gpuA", "matmul.o_proj.h4096", "golden not in 12 candidates"),
        ("gpuC", "softmax.k2048", fit_cv.OUT_OF_SCOPE),  # a card with no ranked case at all
    ]
    header = {"trainer": "linear", "data": "golden", "seed": 0}
    return fit_cv.build_metrics(header, cases, skipped, model, cv)


def test_metrics_json_is_deterministic():
    a = json.dumps(_metrics(), indent=2, sort_keys=True)
    b = json.dumps(_metrics(), indent=2, sort_keys=True)
    assert a == b


def test_metrics_counts_every_skipped_golden():
    m = _metrics()
    cards = m["full_train"]["per_card"]
    assert cards["gpuA"]["unranked"] == 1 and cards["gpuA"]["out_of_scope"] == 1
    assert cards["gpuB"]["unranked"] == 0 and cards["gpuB"]["out_of_scope"] == 0
    # A card whose every golden was skipped still appears — absence is loud, not silent.
    assert cards["gpuC"]["n"] == 0 and cards["gpuC"]["out_of_scope"] == 1
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
        axes=["gpu"],
        header={"trainer": "stub", "seed": 0},
    )


def test_run_fit_stub_trainer_deterministic():
    """``run_fit`` is a pure function of its inputs and knows nothing about the linear model: a stub
    trainer whose models implement only ``score_rows`` yields the full metrics shape, identically on
    every call. It returns the FIT, not an artifact — assembling one is the caller's shipping policy."""
    metrics, fit = _run_stub_fit()
    assert set(metrics) == {"header", "full_train", "cv"} and set(metrics["cv"]) == {"gpu"}
    assert metrics["header"] == {"trainer": "stub", "seed": 0}
    assert metrics["full_train"]["per_card"]["gpuC"]["out_of_scope"] == 1
    assert set(metrics["cv"]["gpu"]["holdout"]["per_golden"]) == {c.key for c in _cases()}
    assert fit.w == -1.0  # the shippable model came from the full-train trainer, not a fold one
    a, b = _run_stub_fit(), _run_stub_fit()
    assert json.dumps(a[0], sort_keys=True) == json.dumps(b[0], sort_keys=True)


# --- CLI surface -------------------------------------------------------------------


def test_fit_command_defaults_and_unsupported_cells():
    parser = argparse.ArgumentParser()
    register_fit_command(parser.add_subparsers())
    args = parser.parse_args(["fit"])
    assert (args.trainer, args.data, args.samples, args.seed, args.folds) == ("linear", "golden", 0, 0, "both")
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
    monkeypatch.setattr("emmy.commands.fit.build_golden_groups", lambda spec: (_cases(), []))  # noqa: ARG005
    parser = argparse.ArgumentParser()
    register_fit_command(parser.add_subparsers())
    args = parser.parse_args(["fit", "--folds", "gpu", "--out", str(tmp_path / "run")])
    args.func(args)

    metrics = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert metrics["header"]["trainer_params"]["objective"] == "mean_log_rank"
    # The two seeding policies stay recorded: without them two fits are not comparable.
    assert metrics["header"]["trainer_params"]["fold_seed_weights"] == "zeros"
    assert set(metrics["cv"]) == {"gpu"}

    artifact = json.loads((tmp_path / "run" / "weights.json").read_text())
    assert artifact["kind"] == "linear" and artifact["weights"] and artifact["weights_dynamic"]
    assert artifact["provenance"]["cases"] == {"static": 6, "dynamic": 2}
    assert "static top1=" in artifact["provenance"]["notes"]
