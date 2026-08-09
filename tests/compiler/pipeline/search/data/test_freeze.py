"""The measurement freeze (v3) — the leaf sanity filter, freeze-twice
determinism over the per-GPU YAML directory, load-time feature re-derivation, the
loader's hard-error contract, and the DB/freeze interchange seam (``load_node_rows``).

The freeze is Phase 3 of the offline-prior rework: a fit must be a pure function of
(repo, pinned data), so one command snapshots the live node DB into a digest-pinned
directory of golden-spelled per-GPU YAML files. These tests never touch a GPU — the
plausibility physics itself is covered by ``test_node_gate.py``; the specs here are
small fp32 matmuls so the loader's snippet re-trace stays cheap (and lru-cached)."""

from __future__ import annotations

import hashlib
import json

import pytest
import yaml

from emmy.compiler.pipeline.search.data import Dataset
from emmy.compiler.pipeline.search.data.freeze import (
    FREEZE_KIND,
    FREEZE_VER,
    MANIFEST_NAME,
    freeze_reason,
    load_freeze,
    load_node_rows,
    write_freeze,
)
from emmy.compiler.pipeline.search.data.shape import ShapeKey
from emmy.compiler.pipeline.search.db import SearchDB
from emmy.compiler.pipeline.search.features import FEATURIZER_VERSION
from tests.compiler.pipeline.search.conftest import GPU_5090 as _GPU
from tests.compiler.pipeline.search.conftest import impossible_staged_feats
from tests.compiler.pipeline.search.conftest import node_row as _row

_GPU2 = "NVIDIA GeForce RTX 4090"

# Small declarative identities — the loader re-traces each DISTINCT shape once (cached).
_SPEC_A = {"kernel": "matmul", "M": 64, "N": 64, "K": 64, "dtype": "fp32", "trans_b": False, "dynamic": False}
_SPEC_B = {"kernel": "matmul", "M": 64, "N": 64, "K": 128, "dtype": "fp32", "trans_b": False, "dynamic": False}


def _feats(*, opt: float = 3.0, cc: float = 120.0, **knobs) -> dict:
    """Write-time features consistent with ``_SPEC_A``-sized shapes: the regime stamps the
    freeze needs (``H_cc`` / ``H_opt``), plausible extents, and the tunable knobs that
    must survive the round trip verbatim."""
    return {
        "H_cc": cc,
        "H_opt": opt,
        "S_ext_free_prod": 4096.0,
        "S_ext_free_max": 64.0,
        "S_ext_reduce_max": 64.0,
        "S_ext_n_free_axis": 2.0,
        "S_ext_n_reduce_axis": 1.0,
        "S_loop_depth": 3.0,
        "S_dtype_f32": 3.0,
        "TILE": "f2x2",
        "WORK": "t16x16",
        **knobs,
    }


# ---------------------------------------------------------------------------
# freeze_reason — the sanity filter
# ---------------------------------------------------------------------------


def test_reason_keeps_ok_leaf() -> None:
    assert freeze_reason(_row("k", value_us=500.0, features=_feats())) is None


def test_reason_keeps_bench_fail_leaf_as_negative() -> None:
    # A fail's value_us is the watchdog sentinel — absurd as a latency, but the row is
    # a durable "doesn't build/launch here" negative and must freeze.
    assert freeze_reason(_row("k", value_us=9.17, features=_feats(), status="bench_fail")) is None


def test_reason_keeps_ordinary_tune_rows() -> None:
    assert freeze_reason(_row("k", value_us=500.0, features=_feats())) is None


def test_reason_drops_rows_without_regime_stamps() -> None:
    feats = {k: v for k, v in _feats().items() if k != "H_cc"}
    assert "missing H_" in freeze_reason(_row("k", value_us=500.0, features=feats))


def test_reason_drops_non_leaves() -> None:
    assert freeze_reason(_row("k", value_us=500.0, features=_feats(), is_leaf=False)) is not None
    assert freeze_reason(_row("k", value_us=500.0, features=_feats(), is_leaf=None)) is not None


def test_reason_drops_stale_feat_ver() -> None:
    stale = FEATURIZER_VERSION - 1
    assert freeze_reason(_row("k", value_us=500.0, features=_feats(), feat_ver=stale)) is not None
    # ... including fail rows: a negative spelled in a retired vocabulary is unreadable too.
    assert freeze_reason(_row("k", value_us=9.17, features=_feats(), status="bench_fail", feat_ver=stale)) is not None


def test_reason_drops_implausible_value() -> None:
    # The conftest f16 mlp_down extents at 9.17 µs imply ~6500 TFLOP/s.
    from tests.compiler.pipeline.search.conftest import F16_MATMUL_FEATS

    feats = {"H_cc": 120.0, "H_opt": 3.0, **F16_MATMUL_FEATS}
    assert "implausible value" in freeze_reason(_row("k", value_us=9.17, features=feats))


def test_reason_drops_impossible_kernel() -> None:
    # The square.512 residue: over-cap cp.async slab -> legal-looking latency, invalid kernel.
    feats = {"H_cc": 120.0, "H_opt": 3.0, **impossible_staged_feats()}
    assert "impossible kernel" in freeze_reason(_row("k", value_us=2.02, features=feats))


# ---------------------------------------------------------------------------
# write_freeze / load_freeze round trip
# ---------------------------------------------------------------------------


def _seed_db(path, rows) -> None:
    db = SearchDB(path)
    db.record_nodes(rows)
    db.close()


_SEED = [
    # _SPEC_A's -O3/-O1 twins on the 5090 — same declarative op, two lanes.
    _row("leaf-a3", value_us=500.0, op_sig="mm1", features=_feats(), run_id="run-a"),
    _row("leaf-a1", value_us=900.0, op_sig="mm1", features=_feats(opt=1.0), run_id="run-a"),
    _row("leaf-b", value_us=480.0, op_sig="mm2", gpu=_GPU2, features=_feats(cc=89.0), run_id="run-b"),
    _row("leaf-fail", value_us=60000.0, op_sig="mm2", features=_feats(), status="bench_fail", run_id="run-a"),
    _row("branch", value_us=480.0, op_sig="mm1", features=_feats(), is_leaf=False),
    _row("leaf-stale", value_us=500.0, op_sig="mm1", features=_feats(), feat_ver=FEATURIZER_VERSION - 1),
    _row("leaf-legacy", value_us=500.0, op_sig="mm1", features=_feats()),  # no identity — never freezes
]
_N_KEPT = 5


def test_write_freeze_round_trip(tmp_path) -> None:
    db_path = tmp_path / "autotune.db"
    _seed_db(db_path, _SEED)
    out = tmp_path / "freeze"
    manifest = write_freeze(db_path, out, note="unit-test policy")

    assert manifest["kind"] == FREEZE_KIND
    assert manifest["freeze_ver"] == FREEZE_VER
    assert manifest["feat_ver"] == manifest["knob_ver"] == manifest["encoding_ver"] == FEATURIZER_VERSION
    assert manifest["counts"] == {"rows": _N_KEPT, "ok": 4, "bench_fail": 1, "per_gpu": {_GPU2: 1, _GPU: 4}}
    assert manifest["run_ids"] == ["run", "run-a", "run-b"]
    assert manifest["policy_note"] == "unit-test policy"
    assert manifest["source_db"] == str(db_path.resolve())
    # One YAML per (gpu, cap); device context is derived while measured structural features persist.
    assert set(manifest["files"]) == {"nvidia_geforce_rtx_5090_sm120.yaml", "nvidia_geforce_rtx_4090_sm89.yaml"}
    doc = yaml.safe_load((out / "nvidia_geforce_rtx_5090_sm120.yaml").read_text())
    assert doc["gpu_name"] == _GPU and doc["compute_cap"] == [12, 0]
    assert all(set(c) & {"H_cc", "H_opt"} == set() for c in doc["configs"])
    assert all(not any(k.startswith(("S_", "H_")) for k in c["knobs"]) for c in doc["configs"])
    assert all(c["structural_features"] for c in doc["configs"])

    loaded_manifest, rows = load_freeze(out)
    assert loaded_manifest == manifest
    assert len(rows) == _N_KEPT
    by_run_value = {(r.run_id, r.value_us): r for r in rows}
    a3 = by_run_value[("run-a", 500.0)]
    a1 = by_run_value[("run-a", 900.0)]
    fail = by_run_value[("run-a", 60000.0)]
    assert a3.status == "ok" and a3.measured_at == "2026-07-09T00:00:00+00:00"
    assert fail.status == "bench_fail"
    # Features are RE-DERIVED: card-faithful H_* (H_cc from the file cap), the opt lane
    # from the row's own field, full traced S_* keying to the spec, tunables verbatim.
    assert a3.features["H_cc"] == 120.0 and a3.features["H_opt"] == 3.0
    assert a1.features["H_opt"] == 1.0
    assert a3.features["TILE"] == "f2x2"
    assert ShapeKey.from_s_features(a3.features).joins(ShapeKey.from_matmul(_SPEC_A["M"], _SPEC_A["N"], _SPEC_A["K"], _SPEC_A["dtype"]))
    assert a3.feat_ver == FEATURIZER_VERSION
    # Treeless contract + stored DB identity: -O1/-O3 twins share one op_sig.
    assert all(r.parent_key is None and r.depth == 0 and r.visits == 0 and r.is_leaf is True for r in rows)
    assert a3.op_sig == a1.op_sig == "mm1"
    assert a3.op_sig != fail.op_sig and a3.node_key != a1.node_key
    # The existing consumers work unchanged on freeze rows.
    assert len(Dataset.from_node_rows(rows)) == _N_KEPT
    assert {g: len(rs) for g, rs in Dataset.fold_node_rows(rows, by="gpu").items()} == {_GPU: 4, _GPU2: 1}
    assert {sig: len(rs) for sig, rs in Dataset.fold_node_rows(rows, by="op").items()} == {a3.op_sig: 3, fail.op_sig: 2}


def test_freeze_twice_same_digest(tmp_path) -> None:
    db_path = tmp_path / "autotune.db"
    _seed_db(db_path, _SEED)
    m1 = write_freeze(db_path, tmp_path / "f1")
    m2 = write_freeze(db_path, tmp_path / "f2")
    assert m1["sha256"] == m2["sha256"]
    assert m1["files"] == m2["files"]
    for name in m1["files"]:
        assert (tmp_path / "f1" / name).read_bytes() == (tmp_path / "f2" / name).read_bytes()


def test_freeze_digest_insertion_order_independent(tmp_path) -> None:
    _seed_db(tmp_path / "fwd.db", _SEED)
    _seed_db(tmp_path / "rev.db", list(reversed(_SEED)))
    m_fwd = write_freeze(tmp_path / "fwd.db", tmp_path / "fwd")
    m_rev = write_freeze(tmp_path / "rev.db", tmp_path / "rev")
    assert m_fwd["sha256"] == m_rev["sha256"]


def test_refreezing_a_freeze_is_stable(tmp_path) -> None:
    # write_freeze reads through load_node_rows, so a freeze re-freezes: the loaded rows
    # carry the stored structural measurements, and the digests must not drift.
    db_path = tmp_path / "autotune.db"
    _seed_db(db_path, _SEED)
    m1 = write_freeze(db_path, tmp_path / "f1")
    m2 = write_freeze(tmp_path / "f1", tmp_path / "f2")
    assert m1["sha256"] == m2["sha256"]


# ---------------------------------------------------------------------------
# the loader's hard-error contract
# ---------------------------------------------------------------------------


def _frozen(tmp_path):
    db_path = tmp_path / "autotune.db"
    _seed_db(db_path, _SEED)
    out = tmp_path / "freeze"
    write_freeze(db_path, out)
    return out


def test_load_freeze_digest_mismatch_hard_error(tmp_path) -> None:
    out = _frozen(tmp_path)
    f = out / "nvidia_geforce_rtx_4090_sm89.yaml"
    f.write_text(f.read_text().replace("value_us: 480.0", "value_us: 4.0"))
    with pytest.raises(RuntimeError, match="corrupt"):
        load_freeze(out)


def test_load_freeze_missing_listed_file_hard_error(tmp_path) -> None:
    out = _frozen(tmp_path)
    (out / "nvidia_geforce_rtx_4090_sm89.yaml").unlink()
    with pytest.raises(RuntimeError, match="missing"):
        load_freeze(out)


def test_load_freeze_ver_mismatch_hard_error(tmp_path) -> None:
    out = _frozen(tmp_path)
    manifest = json.loads((out / MANIFEST_NAME).read_text())
    manifest["freeze_ver"] = FREEZE_VER + 1
    (out / MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="freeze_ver"):
        load_freeze(out)


def test_load_freeze_foreign_dir_hard_error(tmp_path) -> None:
    plain = tmp_path / "not-a-freeze"
    plain.mkdir()
    with pytest.raises(RuntimeError, match="not a measurement freeze"):
        load_freeze(plain)


def test_load_freeze_malformed_structural_features_hard_error(tmp_path) -> None:
    out = _frozen(tmp_path)
    name = "nvidia_geforce_rtx_5090_sm120.yaml"
    doc = yaml.safe_load((out / name).read_text())
    for c in doc["configs"]:
        c["structural_features"] = "not-a-mapping"
    (out / name).write_text(yaml.safe_dump(doc, sort_keys=True, width=120))
    manifest = json.loads((out / MANIFEST_NAME).read_text())
    # Keep the digest honest so schema validation (not integrity) is what trips.
    from emmy.compiler.pipeline.search.data.freeze import _row_line

    digest = hashlib.sha256()
    for c in doc["configs"]:
        digest.update(_row_line(c))
    manifest["files"][name]["sha256"] = digest.hexdigest()
    (out / MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="lacks op_sig/structural_features"):
        load_freeze(out)


def test_write_freeze_nothing_survives_hard_error(tmp_path) -> None:
    # A store with no leaves has no measurement rows to freeze.
    db_path = tmp_path / "autotune.db"
    _seed_db(db_path, [_row("branch", value_us=500.0, features=_feats(), is_leaf=False)])
    with pytest.raises(RuntimeError, match="no freezable leaf rows"):
        write_freeze(db_path, tmp_path / "freeze")


def test_write_freeze_refuses_to_replace_non_freeze_dir(tmp_path) -> None:
    db_path = tmp_path / "autotune.db"
    _seed_db(db_path, _SEED)
    target = tmp_path / "precious"
    target.mkdir()
    (target / "notes.txt").write_text("do not delete\n")
    with pytest.raises(RuntimeError, match="refusing to replace"):
        write_freeze(db_path, target)
    assert (target / "notes.txt").exists()


# ---------------------------------------------------------------------------
# load_node_rows — the DB/freeze interchange seam
# ---------------------------------------------------------------------------


def test_load_node_rows_zero_byte_file_is_empty_db(tmp_path) -> None:
    # sqlite creates the DB file empty before the first write (an aborted tune), and a
    # 0-byte file IS a valid empty sqlite DB — it must yield no rows, not a freeze error.
    empty = tmp_path / "autotune.db"
    empty.touch()
    assert load_node_rows(empty) == []


def test_load_node_rows_refuses_v1_jsonl_freeze(tmp_path) -> None:
    v1 = tmp_path / "freeze.jsonl"
    header = {"kind": FREEZE_KIND, "freeze_ver": 1, "sha256": "deadbeef"}
    v1.write_text(json.dumps(header) + "\n" + '{"node_key": "k"}\n')
    with pytest.raises(RuntimeError, match="superseded"):
        load_node_rows(v1)


def test_open_readonly_rejects_v1_freeze_with_pointer(tmp_path) -> None:
    # A v1-style JSON file handed to a perf-table consumer (eval variants/knobs/failures
    # --db) must fail at open with a named reason, not a bare sqlite DatabaseError.
    v1 = tmp_path / "freeze.jsonl"
    v1.write_text(json.dumps({"kind": FREEZE_KIND, "freeze_ver": 1}) + "\n")
    with pytest.raises(RuntimeError, match="nodes-dataset"):
        SearchDB.open_readonly(v1)
    garbage = tmp_path / "notes.txt"
    garbage.write_text("hello\n")
    with pytest.raises(RuntimeError, match="not a sqlite database"):
        SearchDB.open_readonly(garbage)


def test_anchor_walk_treats_freeze_rows_as_treeless(tmp_path) -> None:
    # Freeze rows (parentless, the loader's depth=0 stamp) carry no fork structure: the
    # golden-anchored descent must render its loud no-tree absence, never fabricate a
    # root mega-fork from the op's whole leaf set. Live parentless TOP-fork rows
    # (depth >= 1) keep walking.
    from emmy.compiler.pipeline.search.prior import diagnostics

    class _Flat:
        fitted = True

        def mean_scores_features(self, fvecs):
            return [0.0] * len(fvecs)

    out = _frozen(tmp_path)
    _manifest, rows = load_freeze(out)
    matched, steps, descriptor = diagnostics._anchor_walk(_Flat(), rows, {})
    assert (matched, steps) == (0, [])
    assert "no fork-tree data" in descriptor
    live_top_fork = [_row("c1", value_us=500.0, depth=1), _row("c2", value_us=600.0, depth=1)]
    matched_live, _steps, descriptor_live = diagnostics._anchor_walk(_Flat(), live_top_fork, {})
    assert matched_live == 1 and "followed" in descriptor_live


def test_load_node_rows_sniffs_db_and_freeze_dir(tmp_path) -> None:
    db_path = tmp_path / "autotune.db"
    _seed_db(db_path, _SEED)
    out = tmp_path / "freeze"
    write_freeze(db_path, out)
    from_db = load_node_rows(db_path)
    from_freeze = load_node_rows(out)
    assert len(from_db) == len(_SEED)  # raw iter_nodes view — everything
    assert len(from_freeze) == _N_KEPT
    garbage = tmp_path / "notes.txt"
    garbage.write_text("hello\n")
    with pytest.raises(RuntimeError, match="neither a sqlite node DB nor"):
        load_node_rows(garbage)
