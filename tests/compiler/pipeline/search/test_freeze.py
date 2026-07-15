"""The measurement freeze — leaf-only sanity filter, freeze-twice determinism, the
loader's hard-error contract, and the DB/freeze interchange seam (``load_node_rows``).

The freeze is Phase 3 of the offline-prior rework: a fit must be a pure function of
(repo, pinned data), so one command snapshots the live node DB into a digest-pinned
JSONL file. These tests never touch a GPU — the plausibility physics itself is covered
by ``test_node_gate.py``; here the fixtures only need one plausible and one implausible
latency on a registry-known card."""

from __future__ import annotations

import dataclasses
import json

import pytest

from emmy.compiler.pipeline.search.data import Dataset
from emmy.compiler.pipeline.search.data.freeze import FREEZE_KIND, FREEZE_VER, freeze_reason, load_freeze, load_node_rows, write_freeze
from emmy.compiler.pipeline.search.db import NodeRow, SearchDB
from emmy.compiler.pipeline.search.features import FEATURIZER_VERSION

_GPU = "NVIDIA GeForce RTX 5090"  # registry records fp32/fp16 peaks -> the plausibility gate is active
_GPU2 = "NVIDIA GeForce RTX 4090"

# The dynM f16 matmul spelling from test_node_gate.py: stamps certify free x red work,
# so 9.17 us implies ~6500 TFLOP/s (implausible) while 500 us is honest.
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


# ---------------------------------------------------------------------------
# freeze_reason — the sanity filter
# ---------------------------------------------------------------------------


def test_reason_keeps_plausible_ok_leaf() -> None:
    assert freeze_reason(_row("k", value_us=500.0)) is None


def test_reason_keeps_bench_fail_leaf_as_negative() -> None:
    # A fail's value_us is the watchdog sentinel — absurd as a latency, but the row is
    # a durable "doesn't build/launch here" negative and must freeze.
    assert freeze_reason(_row("k", value_us=9.17, status="bench_fail")) is None


def test_reason_drops_non_leaves() -> None:
    assert freeze_reason(_row("k", value_us=500.0, is_leaf=False)) is not None
    assert freeze_reason(_row("k", value_us=500.0, is_leaf=None)) is not None  # pre-enrichment unknown


def test_reason_drops_stale_feat_ver() -> None:
    assert freeze_reason(_row("k", value_us=500.0, feat_ver=FEATURIZER_VERSION - 1)) is not None
    # ... including fail rows: a negative spelled in a retired vocabulary is unreadable too.
    assert freeze_reason(_row("k", value_us=9.17, status="bench_fail", feat_ver=FEATURIZER_VERSION - 1)) is not None


def test_reason_drops_implausible_value() -> None:
    assert "implausible value" in freeze_reason(_row("k", value_us=9.17))  # ~6500 TFLOP/s


def test_reason_drops_impossible_kernel() -> None:
    # The square.512 residue: over-cap cp.async slab -> legal-looking latency, invalid kernel.
    small = {
        **{k: v for k, v in _F16_FEATS.items() if not k.startswith(("S_ext_free", "S_ext_reduce"))},
        "S_ext_free_prod": 512.0,
        "S_ext_reduce_max": 512.0,
        "STAGE@a2": "d1/cp",
    }
    assert "impossible kernel" in freeze_reason(_row("k", value_us=2.02, features=small))


# ---------------------------------------------------------------------------
# write_freeze / load_freeze round trip
# ---------------------------------------------------------------------------


def _seed_db(path, rows) -> None:
    db = SearchDB(path)
    db.record_nodes(rows)
    db.close()


_SEED = [
    _row("leaf-a", value_us=500.0, op_sig="mm1", run_id="run-a"),
    _row("leaf-b", value_us=480.0, op_sig="mm1", gpu=_GPU2, run_id="run-b"),
    _row("leaf-fail", value_us=60000.0, op_sig="mm2", status="bench_fail", run_id="run-a"),
    _row("branch", value_us=480.0, op_sig="mm1", is_leaf=False),
    _row("leaf-stale", value_us=500.0, op_sig="mm1", feat_ver=FEATURIZER_VERSION - 1),
]
_KEPT_KEYS = {"leaf-a", "leaf-b", "leaf-fail"}


def test_write_freeze_round_trip(tmp_path) -> None:
    db_path = tmp_path / "autotune.db"
    _seed_db(db_path, _SEED)
    out = tmp_path / "freeze.jsonl"
    header = write_freeze(db_path, out, note="unit-test policy")

    assert header["kind"] == FREEZE_KIND
    assert header["freeze_ver"] == FREEZE_VER
    assert header["feat_ver"] == header["knob_ver"] == header["encoding_ver"] == FEATURIZER_VERSION
    assert header["counts"] == {"rows": 3, "ok": 2, "bench_fail": 1, "per_gpu": {_GPU2: 1, _GPU: 2}}
    assert header["run_ids"] == ["run-a", "run-b"]
    assert header["policy_note"] == "unit-test policy"
    assert header["source_db"] == str(db_path.resolve())

    loaded_header, rows = load_freeze(out)
    assert loaded_header == header
    assert {r.node_key for r in rows} == _KEPT_KEYS
    # Field-for-field identical to the DB's filter-passing leaves, minus the tree schema.
    db = SearchDB.open_readonly(db_path)
    try:
        db_rows = {r.node_key: r for r in db.iter_nodes() if freeze_reason(r) is None}
    finally:
        db.close()
    for r in rows:
        assert r.parent_key is None and r.depth == 0 and r.visits == 0 and r.is_leaf is True
        expected = dataclasses.replace(db_rows[r.node_key], parent_key=None, depth=0, visits=0)
        assert r == expected
    # The existing consumers work unchanged on freeze rows.
    assert len(Dataset.from_node_rows(rows)) == 3
    folds = Dataset.fold_node_rows(rows, by="gpu")
    assert {g: len(rs) for g, rs in folds.items()} == {_GPU: 2, _GPU2: 1}


def test_freeze_twice_same_digest(tmp_path) -> None:
    db_path = tmp_path / "autotune.db"
    _seed_db(db_path, _SEED)
    h1 = write_freeze(db_path, tmp_path / "f1.jsonl")
    h2 = write_freeze(db_path, tmp_path / "f2.jsonl")
    assert h1["sha256"] == h2["sha256"]
    tail1 = (tmp_path / "f1.jsonl").read_bytes().partition(b"\n")[2]
    tail2 = (tmp_path / "f2.jsonl").read_bytes().partition(b"\n")[2]
    assert tail1 == tail2  # only the header (created_at) may differ


def test_freeze_digest_insertion_order_independent(tmp_path) -> None:
    _seed_db(tmp_path / "fwd.db", _SEED)
    _seed_db(tmp_path / "rev.db", list(reversed(_SEED)))
    h_fwd = write_freeze(tmp_path / "fwd.db", tmp_path / "fwd.jsonl")
    h_rev = write_freeze(tmp_path / "rev.db", tmp_path / "rev.jsonl")
    assert h_fwd["sha256"] == h_rev["sha256"]


# ---------------------------------------------------------------------------
# the loader's hard-error contract
# ---------------------------------------------------------------------------


def _frozen(tmp_path):
    db_path = tmp_path / "autotune.db"
    _seed_db(db_path, _SEED)
    out = tmp_path / "freeze.jsonl"
    write_freeze(db_path, out)
    return out


def test_load_freeze_digest_mismatch_hard_error(tmp_path) -> None:
    out = _frozen(tmp_path)
    head, sep, tail = out.read_bytes().partition(b"\n")
    out.write_bytes(head + sep + tail.replace(b'"status":"ok"', b'"status":"OK"', 1))
    with pytest.raises(RuntimeError, match="corrupt"):
        load_freeze(out)


def test_load_freeze_feat_ver_mismatch_hard_error(tmp_path) -> None:
    # Rewriting the header alone doesn't trip the digest (it covers only the rows) —
    # the version gate must catch it on its own.
    out = _frozen(tmp_path)
    head, sep, tail = out.read_bytes().partition(b"\n")
    header = json.loads(head)
    header["feat_ver"] = FEATURIZER_VERSION - 1
    out.write_bytes(json.dumps(header, sort_keys=True).encode() + sep + tail)
    with pytest.raises(RuntimeError, match="feat_ver"):
        load_freeze(out)


def test_load_freeze_foreign_file_hard_error(tmp_path) -> None:
    garbage = tmp_path / "notes.txt"
    garbage.write_text("hello\nworld\n")
    with pytest.raises(RuntimeError, match="not a measurement freeze"):
        load_freeze(garbage)


def test_write_freeze_nothing_survives_hard_error(tmp_path) -> None:
    db_path = tmp_path / "autotune.db"
    _seed_db(db_path, [_row("branch-only", value_us=480.0, is_leaf=False)])
    with pytest.raises(RuntimeError, match="no freezable leaf rows"):
        write_freeze(db_path, tmp_path / "freeze.jsonl")


# ---------------------------------------------------------------------------
# load_node_rows — the DB/freeze interchange seam
# ---------------------------------------------------------------------------


def test_load_node_rows_sniffs_db_and_freeze(tmp_path) -> None:
    db_path = tmp_path / "autotune.db"
    _seed_db(db_path, _SEED)
    out = tmp_path / "freeze.jsonl"
    write_freeze(db_path, out)
    from_db = load_node_rows(db_path)
    from_freeze = load_node_rows(out)
    assert {r.node_key for r in from_db} == _KEPT_KEYS | {"branch", "leaf-stale"}  # raw iter_nodes view
    assert {r.node_key for r in from_freeze} == _KEPT_KEYS
    with pytest.raises(RuntimeError, match="not a measurement freeze"):
        garbage = tmp_path / "notes.txt"
        garbage.write_text("hello\n")
        load_node_rows(garbage)
