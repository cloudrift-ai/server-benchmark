"""The node-store physical-plausibility gate + the one-time purge repair.

The 2026-07-08/09 golden sweeps (pre-#330) recorded split-K variants whose over-budget
staged main kernel was rejected at materialize: the bench saw only the tiny combine
kernel and stored its cached ~9 µs as the whole matmul — thousands of TFLOP/s, poisoning
every value-of-position minimum above it. ``record_nodes`` now refuses any ``ok`` row
whose latency beats the card's recorded peak for its stamped shape;
``purge_implausible`` repairs stores written before the gate."""

from __future__ import annotations

import json

from emmy.compiler.pipeline.search.db import NodeRow, SearchDB, implausible_value_reason, impossible_kernel_reason
from emmy.compiler.pipeline.search.features import FEATURIZER_VERSION

_GPU = "NVIDIA GeForce RTX 5090"  # registry records fp32/fp16 peaks -> the gate is active

# The poisoned class's shape: symbolic-M mlp_down (free excludes the symbolic axis,
# benched at the dynamic hint), fp16 operands. 2*4096*14336*512 FLOPs. The stamps
# certify every loop multiplies the iteration space (depth 3 = 1 free + 1 reduce +
# 1 symbolic — the dynM matmul spelling), which is what licenses the free x red work
# formula.
_F16_FEATS = {
    "S_ext_free_prod": 4096.0,
    "S_ext_reduce_max": 14336.0,
    "S_ext_n_free_axis": 1.0,
    "S_ext_n_reduce_axis": 1.0,
    "S_loop_depth": 3.0,
    "S_ext_n_symbolic_axis": 1.0,
    "S_dtype_f16": 2.0,
    "TILE@a2": "a:mma_m16n8k16_f16/w1x8/f2x8/k8",
    "REDUCE@a2": "g2k",
}


def _row(key: str, *, value_us: float, features: dict | None = None, gpu: str = _GPU, **over) -> NodeRow:
    kw = dict(
        node_key=key,
        parent_key=None,
        context_key="ctx",
        op_sig="op",
        features=dict(_F16_FEATS if features is None else features),
        value_us=value_us,
        depth=5,
        gpu=gpu,
        visits=1,
        is_leaf=True,
        status="ok",
        run_id="run",
        measured_at="2026-07-09T00:00:00+00:00",
    )
    kw.update(over)
    return NodeRow(**kw)


def _insert_raw(db: SearchDB, row: NodeRow) -> None:
    """Seed a row bypassing the ``record_nodes`` gate — how pre-gate stores got poisoned."""
    db._conn.execute(
        "INSERT INTO node (node_key, parent_key, context_key, op_sig, gpu, features, value_us, depth, n_updates, "
        "updated_at, visits, is_leaf, variance, n_samples, status, run_id, measured_at, feat_ver) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            row.node_key,
            row.parent_key,
            row.context_key,
            row.op_sig,
            row.gpu,
            json.dumps(row.features),
            row.value_us,
            row.depth,
            row.measured_at,
            row.visits,
            None if row.is_leaf is None else int(row.is_leaf),
            row.variance,
            row.n_samples,
            row.status,
            row.run_id,
            row.measured_at,
            row.feat_ver,
        ),
    )


def _keys(db: SearchDB) -> set[str]:
    return {r.node_key for r in db.iter_nodes()}


# ---------------------------------------------------------------------------
# the predicate
# ---------------------------------------------------------------------------


def test_impossible_leaf_is_flagged() -> None:
    assert implausible_value_reason(_row("k", value_us=9.17)) is not None  # ~6500 TFLOP/s


def test_plausible_leaf_passes() -> None:
    assert implausible_value_reason(_row("k", value_us=500.0)) is None  # ~120 TFLOP/s < 419 fp16 peak


def test_peak_follows_stamped_dtype() -> None:
    # Without a 16-bit operand stamp the fp32 peak applies: ~200 TFLOP/s trips it,
    # the same latency passes under the fp16 ceiling.
    f32 = {k: v for k, v in _F16_FEATS.items() if k != "S_dtype_f16"}
    assert implausible_value_reason(_row("k", value_us=300.0, features=f32)) is not None
    assert implausible_value_reason(_row("k", value_us=300.0)) is None


def test_ungateable_rows_pass() -> None:
    assert implausible_value_reason(_row("k", value_us=9.17, gpu="")) is None  # unknown card
    assert implausible_value_reason(_row("k", value_us=9.17, status="bench_fail")) is None  # sentinel, not a measurement
    assert implausible_value_reason(_row("k", value_us=9.17, features={"TILE@a2": "n16x8/f2x4"})) is None  # no shape
    assert implausible_value_reason(_row("k", value_us=9.17, feat_ver=FEATURIZER_VERSION - 1)) is None  # retired vocabulary


def test_branch_rows_gated_by_same_physics() -> None:
    # A branch whose value-of-position min came from a poisoned leaf trips the predicate
    # itself — that's what keeps an in-batch poisoned chain from landing at all.
    assert implausible_value_reason(_row("k", value_us=9.17, is_leaf=False)) is not None


def test_overlapping_reduce_kinds_stay_ungated() -> None:
    # A softmax/norm's reduced axis is PART of its full-size output, so free x red
    # overcounts its work by the reduce extent — the loop-depth != free+reduce axis-count
    # inequality marks it, and a blazing (honest, memory-bound) latency must NOT be
    # flagged. This was a real false-positive class: 280 of 324 rows on the first
    # DB-wide dry run were softmax/rms_norm leaves.
    softmax = {
        "S_ext_free_prod": 4194304.0,
        "S_ext_reduce_max": 8192.0,
        "S_ext_n_free_axis": 2.0,
        "S_ext_n_reduce_axis": 2.0,
        "S_loop_depth": 2.0,  # 2 != 2 + 2: the reduce axes overlap the output
        "S_dtype_f32": 3.0,
        "REDUCE@a1": "b64",
    }
    assert implausible_value_reason(_row("k", value_us=13.0, features=softmax)) is None
    # rms_norm.dynM spelling (d=2, nf=1, nr=1, sym=1 -> 2 != 3): a b32 cooperative fold
    # at ~12us IS honest (a coop norm legitimately runs ~100x its serial sibling).
    norm_dyn = {
        "S_ext_free_prod": 2048.0,
        "S_ext_reduce_max": 2048.0,
        "S_ext_n_free_axis": 1.0,
        "S_ext_n_reduce_axis": 1.0,
        "S_ext_n_symbolic_axis": 1.0,
        "S_loop_depth": 2.0,
        "S_dtype_f32": 2.0,
        "REDUCE@a1": "b32",
    }
    assert implausible_value_reason(_row("k", value_us=11.8, features=norm_dyn)) is None


def test_over_budget_staged_kernel_is_flagged_at_any_shape() -> None:
    # The square.512.dynM residue: a cp.async-staged warp tile whose slab (139 KB for
    # w1x8/f2x8/k8) exceeds the ~99 KB dynamic-smem cap could never launch — but the
    # combine-only 2 µs it left behind implies a LEGAL 133 TFLOP/s on that small shape,
    # so only the validity check catches it. The same config unstaged is a real kernel.
    small = {
        **{k: v for k, v in _F16_FEATS.items() if not k.startswith(("S_ext_free", "S_ext_reduce"))},
        "S_ext_free_prod": 512.0,
        "S_ext_reduce_max": 512.0,
        "STAGE@a2": "d1/cp",
    }
    assert impossible_kernel_reason(_row("k", value_us=2.02, features=small)) is not None
    assert implausible_value_reason(_row("k", value_us=2.02, features=small)) is None  # latency floor is blind here
    unstaged = {**small, "STAGE@a2": ""}
    assert impossible_kernel_reason(_row("k", value_us=2.02, features=unstaged)) is None
    db = SearchDB()
    db.record_nodes([_row("bad", value_us=2.02, features=small), _row("ok", value_us=6.0, features=unstaged)])
    assert _keys(db) == {"ok"}


# ---------------------------------------------------------------------------
# the write-time gate
# ---------------------------------------------------------------------------


def test_record_nodes_drops_implausible_rows() -> None:
    db = SearchDB()
    db.record_nodes(
        [
            _row("bad-leaf", value_us=9.17),
            _row("bad-branch", value_us=9.17, is_leaf=False),
            _row("good", value_us=500.0),
        ]
    )
    assert _keys(db) == {"good"}


def test_merge_inherits_the_gate(tmp_path) -> None:
    # A pre-gate remote store merged in via merge_nodes goes through record_nodes,
    # so its poisoned rows are dropped at the boundary.
    src_path = tmp_path / "remote.db"
    src = SearchDB(src_path)
    _insert_raw(src, _row("bad", value_us=9.17))
    _insert_raw(src, _row("good", value_us=500.0))
    src.close()
    dst = SearchDB()
    assert dst.merge_nodes(src_path) == 2
    assert _keys(dst) == {"good"}


# ---------------------------------------------------------------------------
# the one-time purge repair
# ---------------------------------------------------------------------------


def _seed_poisoned_store(db: SearchDB) -> None:
    # branch B: poisoned bound (min over bad leaf L1 + honest leaf L2)
    _insert_raw(db, _row("B", value_us=9.17, is_leaf=False, depth=3))
    _insert_raw(db, _row("L1", value_us=9.17, parent_key="B", depth=4))
    _insert_raw(db, _row("L2", value_us=500.0, parent_key="B", depth=4))
    # branch B2: only a poisoned leaf below -> both go
    _insert_raw(db, _row("B2", value_us=9.19, is_leaf=False, depth=3))
    _insert_raw(db, _row("L3", value_us=9.19, parent_key="B2", depth=4))
    # untouched honest chain
    _insert_raw(db, _row("B3", value_us=480.0, is_leaf=False, depth=3))
    _insert_raw(db, _row("L4", value_us=480.0, parent_key="B3", depth=4))


def test_purge_deletes_leaves_and_repairs_branches() -> None:
    db = SearchDB()
    _seed_poisoned_store(db)
    receipt = db.purge_implausible()
    assert receipt == {"deleted_leaves": 2, "deleted_branches": 1, "repaired_branches": 1}
    rows = {r.node_key: r for r in db.iter_nodes()}
    assert set(rows) == {"B", "L2", "B3", "L4"}
    assert rows["B"].value_us == 500.0  # bound recomputed over the surviving honest leaf


def test_purge_dry_run_changes_nothing() -> None:
    db = SearchDB()
    _seed_poisoned_store(db)
    receipt = db.purge_implausible(dry_run=True)
    assert receipt == {"deleted_leaves": 2, "deleted_branches": 1, "repaired_branches": 1}
    assert len(_keys(db)) == 7
