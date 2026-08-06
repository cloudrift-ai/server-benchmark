"""The ``emmy fit`` cross-validation harness (``search/prior/fit/cv``) and run harness
(``search/prior/fit/run``): fold partitioning on both axes, the fittability-exclusion
guard, dual-rank tie semantics, metrics-file determinism, and the trainer-callable seam
— all on synthetic cases, no tracing, no GPU."""

import argparse
import json

import numpy as np
import pytest

from emmy.commands.fit import register_fit_command
from emmy.compiler.pipeline.search.prior.fit import (
    DEFAULT_FEATURES,
    MATMUL_FEATURES,
    Group,
    TwoStageFit,
    dual_rank,
    feature_view,
    fit_two_stage,
    op_family,
)
from emmy.compiler.pipeline.search.prior.fit import cv as fit_cv
from emmy.compiler.pipeline.search.prior.fit.run import run_fit

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
    """A tiny case whose rows carry a monotone D_a so a samples=0 descent has signal."""
    feats = [{"D_a": float(i), "D_b": float((i * 7) % 3)} for i in range(n_rows)]
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


NAMES = ["D_a", "D_b"]
# The fitted scalar params seed OFF at the shipped threshold (the atomic-free interaction).
SEED_PARAMS = {"atomic_free_weight": 0.0, "atomic_free_split_threshold": 4.0}


def _fit_model(groups, rng):
    """The zero-seeded fold trainer the command layer wires up, at samples=0."""
    return fit_two_stage(groups, NAMES, seed_weights={}, seed_params=SEED_PARAMS, rng=rng, samples=0)


# --- fold partitioning + pooling ---------------------------------------------------


def test_run_axis_gpu_pools_every_golden_exactly_once():
    out = fit_cv.run_axis(_cases(), "gpu", fit_model=_fit_model, seed=0)
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
    out = fit_cv.run_axis(_cases(), "op_family", fit_model=_fit_model, seed=0)
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
    out = fit_cv.run_axis(cases, "op_family", fit_model=_fit_model, seed=0)
    assert out["fold_detail"]["excluded"] == {"matmul.mlp_down": "dynamic weight set unfittable (0 dyn cases in training)"}
    assert "gpuA/matmul.mlp_down.h4096.dynM" not in out["holdout"]["per_golden"]
    assert "matmul.mlp_down" not in out["fold_detail"]["holdout_medians"]
    # The healthy folds still pool.
    assert "gpuA/matmul.square.512" in out["holdout"]["per_golden"]


# --- metrics assembly: determinism + skip accounting -------------------------------


def _metrics():
    cases = _cases()
    model = TwoStageFit({"D_a": -1.0, "D_b": 0.25}, [0], {"D_a": -0.5}, [0], SEED_PARAMS)
    cv = {"gpu": fit_cv.run_axis(cases, "gpu", fit_model=_fit_model, seed=0)}
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
    """A minimal trainer-protocol model — enough to prove ``run_fit`` never needs the
    linear model class: ``score_rows`` + ``to_artifact`` and nothing else."""

    def __init__(self, w):
        self.w = w

    def score_rows(self, group):
        return group.matrix(NAMES) @ np.array([self.w, 0.0])

    def to_artifact(self, *, params, provenance):
        return {"kind": "stub", "params": params, "provenance": provenance}


def _run_stub_fit():
    def full_train_fit(groups, rng):
        return _StubModel(-1.0), "stub notes"

    def fit_model(groups, rng):
        return _StubModel(float(rng.standard_normal()))

    return run_fit(
        _cases(),
        [("gpuC", "softmax.k2048", fit_cv.OUT_OF_SCOPE)],
        full_train_fit=full_train_fit,
        fit_model=fit_model,
        axes=["gpu"],
        seed=0,
        header={"trainer": "stub", "seed": 0},
        params={"scale": 0.1},
        provenance={"fitted": "2026-01-01", "script": "test"},
    )


def test_run_fit_stub_trainer_deterministic():
    """``run_fit`` is a pure function of its inputs: a stub trainer (no linear-model
    machinery) yields the full metrics shape and an artifact whose caller-supplied
    provenance is completed with the case counts and the trainer's notes — identically
    on every call."""
    metrics, artifact = _run_stub_fit()
    assert set(metrics) == {"header", "full_train", "cv"} and set(metrics["cv"]) == {"gpu"}
    assert metrics["header"] == {"trainer": "stub", "seed": 0}
    assert metrics["full_train"]["per_card"]["gpuC"]["out_of_scope"] == 1
    assert set(metrics["cv"]["gpu"]["holdout"]["per_golden"]) == {c.key for c in _cases()}
    assert artifact == {
        "kind": "stub",
        "params": {"scale": 0.1},
        "provenance": {
            "fitted": "2026-01-01",
            "script": "test",
            "cases": {"static": 6, "dynamic": 2},
            "notes": "stub notes",
        },
    }
    a, b = _run_stub_fit(), _run_stub_fit()
    assert json.dumps(a[0], sort_keys=True) == json.dumps(b[0], sort_keys=True) and a[1] == b[1]


# --- CLI surface -------------------------------------------------------------------


def test_fit_command_defaults_and_unsupported_cells():
    parser = argparse.ArgumentParser()
    register_fit_command(parser.add_subparsers())
    args = parser.parse_args(["fit"])
    assert (args.trainer, args.data, args.samples, args.seed, args.folds) == ("linear", "golden", 0, 0, "both")
    assert args.features == DEFAULT_FEATURES
    # --artifact: absent = no extra write, bare = "" (the shipped offline_weights.json), a value = that path.
    assert args.artifact is None
    assert parser.parse_args(["fit", "--artifact"]).artifact == ""
    assert parser.parse_args(["fit", "--artifact", "/tmp/cand.json"]).artifact == "/tmp/cand.json"

    for bad in (
        parser.parse_args(["fit", "--trainer", "catboost"]),
        parser.parse_args(["fit", "--data", "freeze:/tmp/x.jsonl"]),
    ):
        with pytest.raises(SystemExit, match="not yet supported"):
            bad.func(bad)
