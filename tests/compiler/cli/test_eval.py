"""Tests for ``emmy eval knobs`` / ``eval variants`` — the tune-DB analysis CLIs.

Each test builds a synthetic tune-DB inline (just the two tables the
commands read: ``cuda_op`` and ``perf``), so the suite stays hermetic
and does not depend on a real autotune cache or GPU. The ``variants``
CLI tests pin ``--prior`` to a nonexistent file so the pick comes from
the cold ``OfflinePrior`` regardless of any prior checkpoint on the host.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

from tests.compiler.pipeline.search.helpers import GPU_5090 as _GPU


def _make_tune_db(path: Path, variants: list[tuple[str, str, dict, float]]) -> None:
    """Write a minimal tune DB to ``path``.

    ``variants`` is a list of ``(op_key, kernel_name, knobs, latency_us)``
    rows; one ``cuda_op`` + one ``perf`` row is written per entry. Other
    real-DB columns (kernel_source, arg_order, grid, block, smem_bytes)
    are filled with dummy values — ``knobs`` only reads ``cuda_op.pretty``
    and ``perf.knobs``/``perf.latency_us_median``.
    """
    con = sqlite3.connect(str(path))
    con.executescript(
        """
        CREATE TABLE cuda_op (
            key           TEXT PRIMARY KEY,
            kernel_source TEXT NOT NULL,
            arg_order     TEXT NOT NULL,
            grid          TEXT NOT NULL,
            block         TEXT NOT NULL,
            smem_bytes    INTEGER NOT NULL,
            pretty        TEXT NOT NULL
        );
        CREATE TABLE perf (
            context_key          TEXT NOT NULL,
            op_key               TEXT NOT NULL,
            backend              TEXT NOT NULL,
            status               TEXT NOT NULL,
            latency_us_median    REAL NOT NULL,
            latency_us_min       REAL NOT NULL,
            latency_us_max       REAL NOT NULL,
            latency_us_mean      REAL NOT NULL,
            latency_us_variance  REAL NOT NULL,
            n_samples            INTEGER NOT NULL,
            measured_at          TEXT NOT NULL,
            knobs                TEXT NOT NULL DEFAULT '{}',
            PRIMARY KEY (context_key, op_key, backend)
        );
        """
    )
    for op_key, kernel_name, knobs, us in variants:
        pretty = f'extern "C" __global__\n__launch_bounds__(256) void {kernel_name}(const float* x) {{ }}\n'
        con.execute(
            "INSERT INTO cuda_op (key, kernel_source, arg_order, grid, block, smem_bytes, pretty) "
            "VALUES (?, '', '[]', '[1,1,1]', '[1,1,1]', 0, ?)",
            (op_key, pretty),
        )
        con.execute(
            "INSERT INTO perf (context_key, op_key, backend, status, latency_us_median, latency_us_min, latency_us_max, "
            "latency_us_mean, latency_us_variance, n_samples, measured_at, knobs) "
            "VALUES ('ctx', ?, 'cuda', 'ok', ?, 0, 0, 0, 0, 1, '2026-05-24', ?)",
            (op_key, us, json.dumps(knobs)),
        )
    con.commit()
    con.close()


def test_eval_golden_requires_exact_file_and_serving_config(run_cli, tmp_path):
    rc, stdout, stderr = run_cli("eval", "golden")
    assert rc == 2
    assert "GOLDEN_YAML" in stdout + stderr

    golden = tmp_path / "given.yaml"
    configured = tmp_path / "configured.yaml"
    config = tmp_path / "release.env"
    config.write_text(
        f"SERVE_MODEL=org/model\nSERVE_GPU=NVIDIA-Test\nSERVE_GOLDEN_FILE={configured}\n"
        "SERVE_MAX_NUM_BATCHED_TOKENS=32\nSERVE_DECODE_BUCKET=8\n"
    )
    rc, stdout, stderr = run_cli("eval", "golden", str(golden), "--serving-config", str(config))
    assert rc == 2
    assert "serving config names" in stdout + stderr


def test_serving_config_derives_standard_and_fast_math_realizations(tmp_path):
    from emmy.serving.release import load_serving_config

    golden = tmp_path / "golden.yaml"
    config = tmp_path / "release.env"
    config.write_text(
        f"SERVE_MODEL=org/model\nSERVE_GPU=NVIDIA-Test\nSERVE_GOLDEN_FILE={golden}\n"
        "SERVE_MAX_NUM_BATCHED_TOKENS=72\nSERVE_DECODE_BUCKET=8\nSERVE_PREFILL_CAPACITY=64\n"
        'SERVE_WARM_SHAPES="32:64:96:fm"\n'
    )

    serving = load_serving_config(config)

    got = {(dict(row.bindings).get("num_tokens"), row.pins) for row in serving.realizations}
    assert got == {
        *((width, (("FAST_MATH", False),)) for width in (None, 1, 8, 64)),
        *((width, (("FAST_MATH", True),)) for width in (None, 1, 32, 64)),
    }


def _write_release_golden(path: Path, realizations: list[dict]) -> None:
    from emmy.commands.trace import trace_inline_code
    from emmy.compiler.pipeline.search.golden import GoldenFileValidation, dump_golden_file
    from emmy.compiler.torch_wire import graph_to_wire

    graph = trace_inline_code("torch.relu(torch.randn(8))")["graph"]
    terminal = graph.producer(graph.outputs[0])
    dump_golden_file(
        {
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "compute_cap": [8, 9],
            "model": "org/model",
            "programs": [graph_to_wire(graph)],
            "configs": [{"program": 0, "target": {"origins": [terminal.id]}, "realizations": realizations}],
        },
        path,
        validation=GoldenFileValidation.REPOSITORY,
    )


def test_eval_golden_audits_file_scoped_static_release(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import emmy.commands.eval as eval_cmd
    import emmy.compiler.pipeline.search.audit as audit
    import emmy.serving.twins as twins
    from emmy.compiler.context import Context

    golden = tmp_path / "golden.yaml"
    _write_release_golden(
        golden,
        [
            {
                "name": "relu.m1",
                "bindings": {"num_tokens": 1},
                "pins": {"FAST_MATH": False},
                "knobs": {},
                "measurements": {"emmy_us": 1.0, "reference_us": 1.0, "reference_backend": "torch"},
            }
        ],
    )
    config = tmp_path / "release.env"
    config.write_text(
        f'SERVE_MODEL=org/model\nSERVE_GPU="NVIDIA GeForce RTX 4090"\nSERVE_GOLDEN_FILE={golden}\n'
        "SERVE_STATIC_ONLY=1\nSERVE_MAX_NUM_BATCHED_TOKENS=1\nSERVE_DECODE_BUCKET=1\n"
        "SERVE_PREFILL_CAPACITY=1\nSERVE_PREFILL_BUCKET=0\nSERVE_M1_TIER=1\nSERVE_CAPTURE_SIZES=[1]\n"
    )
    ctx = Context.from_target((8, 9), gpu_name="NVIDIA GeForce RTX 4090")
    monkeypatch.setattr(Context, "probe", staticmethod(lambda: ctx))
    monkeypatch.setattr(eval_cmd, "_emit_prior_golden_check", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(eval_cmd, "_emit_offer_audit", lambda _records: False)
    captured = {}

    def fake_capture(source, **kwargs):
        captured["capture"] = (source, kwargs)
        return {"pre1": object()}

    def fake_audit(graphs, gpu_name, compute_cap, *, goldens):
        captured["audit"] = (graphs, gpu_name, compute_cap, goldens)
        return {"pre1": [{"verdict": "MATCH"}]}

    monkeypatch.setattr(twins, "capture_twin_graphs", fake_capture)
    monkeypatch.setattr(audit, "audit_card", fake_audit)
    monkeypatch.setattr(audit, "summarize", lambda _results: {"MATCH": 1, "DRIFT": 0, "GAP": 0, audit.COMPILE_FAIL: 0})
    monkeypatch.setattr(audit, "gap_keys", lambda _results: set())

    eval_cmd.handle_eval_golden(SimpleNamespace(golden_file=str(golden), serving_config=str(config), update_consult_baseline=False))

    assert captured["capture"] == ("org/model", {"decode_bucket": 1, "prefill_bucket": 0, "symbolic": False, "static_only": True})
    _, gpu_name, compute_cap, records = captured["audit"]
    assert (gpu_name, compute_cap) == ("NVIDIA GeForce RTX 4090", (8, 9))
    assert {(record.bindings, record.pins) for record in records} == {((("num_tokens", 1),), (("FAST_MATH", False),))}


def test_eval_golden_rejects_a_missing_config_realization(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import pytest

    import emmy.commands.eval as eval_cmd
    from emmy.compiler.context import Context

    golden = tmp_path / "golden.yaml"
    _write_release_golden(
        golden,
        [
            {
                "name": "relu.m8",
                "bindings": {"num_tokens": 8},
                "pins": {"FAST_MATH": False},
                "knobs": {},
                "measurements": {"emmy_us": 1.0, "reference_us": 1.0, "reference_backend": "torch"},
            }
        ],
    )
    config = tmp_path / "release.env"
    config.write_text(
        f'SERVE_MODEL=org/model\nSERVE_GPU="NVIDIA GeForce RTX 4090"\nSERVE_GOLDEN_FILE={golden}\n'
        "SERVE_MAX_NUM_BATCHED_TOKENS=32\nSERVE_DECODE_BUCKET=8\n"
        "SERVE_PREFILL_CAPACITY=32\nSERVE_PREFILL_BUCKET=0\n"
    )
    ctx = Context.from_target((8, 9), gpu_name="NVIDIA GeForce RTX 4090")
    monkeypatch.setattr(Context, "probe", staticmethod(lambda: ctx))

    with pytest.raises(SystemExit) as exc:
        eval_cmd.handle_eval_golden(SimpleNamespace(golden_file=str(golden), serving_config=str(config), update_consult_baseline=False))
    assert exc.value.code == 1


def test_serving_config_resolves_consult_baseline(tmp_path):
    from emmy.serving.release import load_serving_config

    golden = tmp_path / "golden.yaml"
    baseline = tmp_path / "consult.json"
    config = tmp_path / "release.env"
    base = (
        f"SERVE_MODEL=org/model\nSERVE_GPU=NVIDIA-Test\nSERVE_GOLDEN_FILE={golden}\n"
        "SERVE_MAX_NUM_BATCHED_TOKENS=32\nSERVE_DECODE_BUCKET=8\n"
    )
    config.write_text(base)
    assert load_serving_config(config).consult_baseline is None
    config.write_text(base + f"SERVE_CONSULT_BASELINE={baseline}\n")
    assert load_serving_config(config).consult_baseline == baseline


def _static_release_audit(monkeypatch, tmp_path, *, records: list[dict], extra_env: str = ""):
    """The static-release eval-golden harness: one twin (``pre1``) whose audit yields
    ``records``. Returns ``(eval_cmd, args_namespace)`` ready for ``handle_eval_golden``."""
    from types import SimpleNamespace

    import emmy.commands.eval as eval_cmd
    import emmy.compiler.pipeline.search.audit as audit
    import emmy.serving.twins as twins
    from emmy.compiler.context import Context

    golden = tmp_path / "golden.yaml"
    _write_release_golden(
        golden,
        [
            {
                "name": "relu.m1",
                "bindings": {"num_tokens": 1},
                "pins": {"FAST_MATH": False},
                "knobs": {},
                "measurements": {"emmy_us": 1.0, "reference_us": 1.0, "reference_backend": "torch"},
            }
        ],
    )
    config = tmp_path / "release.env"
    config.write_text(
        f'SERVE_MODEL=org/model\nSERVE_GPU="NVIDIA GeForce RTX 4090"\nSERVE_GOLDEN_FILE={golden}\n'
        "SERVE_STATIC_ONLY=1\nSERVE_MAX_NUM_BATCHED_TOKENS=1\nSERVE_DECODE_BUCKET=1\n"
        "SERVE_PREFILL_CAPACITY=1\nSERVE_PREFILL_BUCKET=0\nSERVE_M1_TIER=1\nSERVE_CAPTURE_SIZES=[1]\n" + extra_env
    )
    ctx = Context.from_target((8, 9), gpu_name="NVIDIA GeForce RTX 4090")
    monkeypatch.setattr(Context, "probe", staticmethod(lambda: ctx))
    monkeypatch.setattr(eval_cmd, "_emit_prior_golden_check", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(eval_cmd, "_emit_offer_audit", lambda _records: False)
    monkeypatch.setattr(twins, "capture_twin_graphs", lambda source, **kwargs: {"pre1": object()})
    monkeypatch.setattr(audit, "audit_card", lambda graphs, gpu_name, compute_cap, *, goldens: {"pre1": list(records)})
    return eval_cmd, SimpleNamespace(golden_file=str(golden), serving_config=str(config), update_consult_baseline=False)


# Two verified-tier consultations on the audited twin; key=None keeps the GAP out of the
# gap-ratchet failure set so only the consultation count is under test.
_TWO_CONSULTATIONS = [{"verdict": "MATCH", "key": None}, {"verdict": "GAP", "key": None}]


def test_eval_golden_consultation_drop_fails_naming_the_twin(monkeypatch, tmp_path, caplog):
    """A twin whose verified-tier consultation count falls below the checked-in baseline — or
    that vanishes from the audit entirely — is a gate failure naming the twin, even with zero
    DRIFT: a kernel that stops forking consults nothing, so its MATCHes silently vanish (the
    regression class the verdicts cannot see)."""
    import logging

    import pytest

    baseline = tmp_path / "consult.json"
    baseline.write_text(json.dumps({"FAST_MATH=false": {"pre1": 3, "gone": 1}}))
    eval_cmd, args = _static_release_audit(
        monkeypatch, tmp_path, records=_TWO_CONSULTATIONS, extra_env=f"SERVE_CONSULT_BASELINE={baseline}\n"
    )

    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as exc:
        eval_cmd.handle_eval_golden(args)
    assert exc.value.code == 1
    assert any("pre1" in r.message and "dropped 3 -> 2" in r.message for r in caplog.records)
    assert any("gone" in r.message and "vanished" in r.message for r in caplog.records)


def test_eval_golden_consultation_baseline_holds_and_flags_staleness(monkeypatch, tmp_path, caplog):
    """Counts at (or above) baseline pass; a grown count only marks the baseline stale."""
    import logging

    baseline = tmp_path / "consult.json"
    baseline.write_text(json.dumps({"FAST_MATH=false": {"pre1": 1}}))
    eval_cmd, args = _static_release_audit(
        monkeypatch, tmp_path, records=_TWO_CONSULTATIONS, extra_env=f"SERVE_CONSULT_BASELINE={baseline}\n"
    )

    with caplog.at_level(logging.INFO):
        eval_cmd.handle_eval_golden(args)
    assert any("stale" in r.message for r in caplog.records)


def test_eval_golden_missing_consult_baseline_fails(monkeypatch, tmp_path, caplog):
    """SERVE_CONSULT_BASELINE naming a nonexistent file is a misconfigured gate, not a skip."""
    import logging

    import pytest

    eval_cmd, args = _static_release_audit(
        monkeypatch, tmp_path, records=_TWO_CONSULTATIONS, extra_env=f"SERVE_CONSULT_BASELINE={tmp_path / 'consult.json'}\n"
    )

    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as exc:
        eval_cmd.handle_eval_golden(args)
    assert exc.value.code == 1
    assert any("--update-consult-baseline" in r.message for r in caplog.records)


def test_eval_golden_update_records_consult_baseline(monkeypatch, tmp_path):
    """``--update-consult-baseline`` writes the observed per-lane per-twin counts."""
    baseline = tmp_path / "consult.json"
    eval_cmd, args = _static_release_audit(
        monkeypatch, tmp_path, records=_TWO_CONSULTATIONS, extra_env=f"SERVE_CONSULT_BASELINE={baseline}\n"
    )
    args.update_consult_baseline = True

    eval_cmd.handle_eval_golden(args)

    assert json.loads(baseline.read_text()) == {"FAST_MATH=false": {"pre1": 2}}


def test_knobs_missing_db(run_cli, tmp_path):
    """A non-existent DB path is not an error: the registry schema still prints and
    the regret analysis is skipped cleanly (exit 0, no traceback)."""
    rc, stdout, stderr = run_cli("eval", "knobs", "--db", str(tmp_path / "does_not_exist.db"))
    combined = (stdout + stderr).lower()
    assert rc == 0, f"stderr: {stderr}"
    assert "no tune db" in combined and "skipping" in combined, f"expected graceful skip:\nstdout={stdout!r}"
    assert "traceback" not in combined


def test_knobs_empty_db(run_cli, tmp_path):
    """Empty DB → command exits 0 and reports zero kernels."""
    db = tmp_path / "empty.db"
    _make_tune_db(db, variants=[])
    rc, stdout, stderr = run_cli("eval", "knobs", "--db", str(db))
    assert rc == 0, f"stderr: {stderr}"
    assert "0 (of 0 total)" in stdout


def test_knobs_ranks_higher_impact_knob_first(run_cli, tmp_path):
    """Two knobs with very different regret: ``BIG`` swings 100x, ``SMALL``
    swings 1.1x. ``BIG`` must appear first in the ranked table."""
    db = tmp_path / "ranked.db"
    rows = []
    # 8 variants of one kernel: 4 BIG values × 2 SMALL values.
    # Latency = 1.0 × big_factor[BIG] × small_factor[SMALL].
    big_factor = {16: 1.0, 64: 5.0, 128: 50.0, 256: 100.0}
    small_factor = {1: 1.0, 2: 1.1}
    i = 0
    for big, bf in big_factor.items():
        for small, sf in small_factor.items():
            rows.append((f"k{i}", "k_test_kernel", {"BIG": big, "SMALL": small}, bf * sf))
            i += 1
    _make_tune_db(db, rows)

    rc, stdout, stderr = run_cli("eval", "knobs", "--db", str(db))
    assert rc == 0, f"stderr: {stderr}"
    # Both knobs appear, BIG first.
    assert "BIG" in stdout and "SMALL" in stdout
    big_pos = stdout.index("\nBIG ")
    small_pos = stdout.index("\nSMALL ")
    assert big_pos < small_pos, f"BIG should rank above SMALL:\n{stdout}"
    # Regret math: BIG = 100/1 = 100; SMALL = 1.1.
    assert "100.00x" in stdout
    assert "1.10x" in stdout


def test_knobs_min_variants_filter(run_cli, tmp_path):
    """Kernels below ``--min-variants`` are excluded from the table."""
    db = tmp_path / "filter.db"
    # k_small: 3 variants (filtered out at --min-variants 5)
    # k_big: 6 variants (kept)
    rows = []
    for i in range(3):
        rows.append((f"s{i}", "k_small", {"A": i}, 10.0 + i))
    for i in range(6):
        rows.append((f"b{i}", "k_big", {"A": i}, 1.0 + i))
    _make_tune_db(db, rows)

    rc, stdout, stderr = run_cli("eval", "knobs", "--db", str(db), "--min-variants", "5")
    assert rc == 0, f"stderr: {stderr}"
    assert "1 (of 2 total)" in stdout


def test_knobs_kernel_filter(run_cli, tmp_path):
    """``--kernel`` substring restricts the kernel set."""
    db = tmp_path / "kfilter.db"
    rows = []
    for i in range(8):
        rows.append((f"m{i}", "k_matmul", {"BM": 16 if i < 4 else 64}, 10.0 + i))
        rows.append((f"r{i}", "k_rmsnorm", {"BM": 16 if i < 4 else 64}, 1.0 + i))
    _make_tune_db(db, rows)

    rc, stdout, stderr = run_cli("eval", "knobs", "--db", str(db), "--kernel", "matmul")
    assert rc == 0, f"stderr: {stderr}"
    assert "1 (of 1 total)" in stdout


def test_knobs_interaction_matrix_present(run_cli, tmp_path):
    """Output includes the K1\\K2 interaction matrix."""
    db = tmp_path / "interact.db"
    rows = []
    # 8 variants with two knobs that vary; the matrix needs ≥2 values per knob.
    i = 0
    for a in (1, 2):
        for b in (10, 20, 30, 40):
            rows.append((f"x{i}", "k_test", {"A": a, "B": b}, float(a * b)))
            i += 1
    _make_tune_db(db, rows)

    rc, stdout, stderr = run_cli("eval", "knobs", "--db", str(db))
    assert rc == 0, f"stderr: {stderr}"
    assert "knob interaction" in stdout
    assert "K1\\K2" in stdout


def _add_perf_row(path: Path, op_key: str, kernel_name: str, knobs: dict, us: float, *, status: str) -> None:
    """Append one ``cuda_op`` + ``perf`` row with an explicit status (the shared
    ``_make_tune_db`` writes only ``ok`` rows)."""
    con = sqlite3.connect(str(path))
    pretty = f'extern "C" __global__\n__launch_bounds__(256) void {kernel_name}(const float* x) {{ }}\n'
    con.execute(
        "INSERT INTO cuda_op (key, kernel_source, arg_order, grid, block, smem_bytes, pretty) "
        "VALUES (?, '', '[]', '[1,1,1]', '[1,1,1]', 0, ?)",
        (op_key, pretty),
    )
    con.execute(
        "INSERT INTO perf (context_key, op_key, backend, status, latency_us_median, latency_us_min, latency_us_max, "
        "latency_us_mean, latency_us_variance, n_samples, measured_at, knobs) "
        "VALUES ('ctx', ?, 'cuda', ?, ?, 0, 0, 0, 0, 1, '2026-06-10', ?)",
        (op_key, status, us, json.dumps(knobs)),
    )
    con.commit()
    con.close()


# --- eval variants ----------------------------------------------------------


def test_variants_missing_db(run_cli, tmp_path):
    """A non-existent DB path exits cleanly with a pointer, no traceback."""
    rc, stdout, stderr = run_cli("eval", "variants", "--db", str(tmp_path / "does_not_exist.db"), "--online-file", str(tmp_path / "p.json"))
    combined = (stdout + stderr).lower()
    assert rc == 0, f"stderr: {stderr}"
    assert "no tune db" in combined
    assert "traceback" not in combined


def test_variants_golden_dataset_rejected(run_cli, tmp_path):
    """``--dataset golden`` is a degenerate source — fail fast with a message."""
    rc, stdout, stderr = run_cli("eval", "variants", "--dataset", "golden")
    assert rc == 2
    assert "no per-variant measurements" in (stdout + stderr)


def test_variants_leaderboard_sorted_with_ranked_pick(run_cli, tmp_path):
    """The leaderboard sorts by latency (fastest first), marks exactly one pick
    row, and prints its rank summary. No prior checkpoint → cold OfflinePrior,
    no reservoir → no ``-O3 us`` column."""
    db = tmp_path / "v.db"
    _make_tune_db(
        db,
        [
            ("a", "k_matmul", {"BM": 8, "BN": 16}, 30.0),
            ("b", "k_matmul", {"BM": 16, "BN": 16}, 10.0),
            ("c", "k_matmul", {"BM": 64, "BN": 16}, 20.0),
        ],
    )
    rc, stdout, stderr = run_cli("eval", "variants", "--db", str(db), "--online-file", str(tmp_path / "missing.json"))
    assert rc == 0, f"stderr: {stderr}"
    assert "k_matmul — 3 measured configs" in stdout
    assert stdout.index("10.0") < stdout.index("20.0") < stdout.index("30.0")  # fastest first
    assert stdout.count("◄") == 1
    assert "pick: rank" in stdout and "/3," in stdout
    assert "-O3 us" not in stdout  # empty reservoir → column omitted


def test_variants_top_truncation_keeps_pick_row(run_cli, tmp_path):
    """``--top N`` truncates to the N fastest but always shows the pick row, and
    reports how many rows were hidden."""
    db = tmp_path / "t.db"
    _make_tune_db(db, [(f"v{i}", "k_matmul", {"BM": 2**i, "BN": 16}, 10.0 + i) for i in range(6)])
    rc, stdout, stderr = run_cli("eval", "variants", "--db", str(db), "--top", "2", "--online-file", str(tmp_path / "missing.json"))
    assert rc == 0, f"stderr: {stderr}"
    assert "--top 0 shows all" in stdout  # truncation note printed
    assert stdout.count("◄") == 1  # pick row survives the cut


def test_variants_counts_bench_fail_rows(run_cli, tmp_path):
    """``bench_fail`` rows are counted in the kernel header, not listed as variants."""
    db = tmp_path / "f.db"
    _make_tune_db(db, [("a", "k_matmul", {"BM": 8}, 10.0), ("b", "k_matmul", {"BM": 16}, 20.0)])
    _add_perf_row(db, "x", "k_matmul", {"BM": 64, "TMA": True}, 2_000_000.0, status="bench_fail")
    rc, stdout, stderr = run_cli("eval", "variants", "--db", str(db), "--online-file", str(tmp_path / "missing.json"))
    assert rc == 0, f"stderr: {stderr}"
    assert "k_matmul — 2 measured configs, 1 bench_fail" in stdout


def test_failures_clusters_by_kernel_and_error(run_cli, tmp_path):
    """``eval failures`` clusters bench_fail rows by (kernel, error) and reports
    the knob values shared by every failing row in a cluster (the 'all rows have
    TMA=1' forensics signal). Built with the real ``SearchDB`` write path so the
    error column round-trips end to end."""
    from emmy.compiler.pipeline.search.db import PerfStats, SearchDB

    def stats(median):
        return PerfStats(median=median, min=median, max=median, mean=median, variance=0.0, n_samples=1)

    db_path = tmp_path / "f.db"
    db = SearchDB(db_path)
    pretty = 'extern "C" __global__\n__launch_bounds__(256) void k_mm(const float* x) { }\n'
    for i, bm in enumerate((8, 16)):
        db.record_cuda_op(f"f{i}", kernel_source="", arg_order=[], grid=[1, 1, 1], block=[1, 1, 1], smem_bytes=0, pretty=pretty)
        db.record_perf(
            "ctx",
            f"f{i}",
            backend="cuda",
            status="bench_fail",
            stats=stats(1.0),
            knobs={"BM": bm, "TMA": True},
            error="cuTensorMapEncodeTiled failed",
        )
    db.record_cuda_op("ok1", kernel_source="", arg_order=[], grid=[1, 1, 1], block=[1, 1, 1], smem_bytes=0, pretty=pretty)
    db.record_perf("ctx", "ok1", backend="cuda", status="ok", stats=stats(10.0), knobs={"BM": 8, "TMA": False})
    db.close()

    rc, stdout, stderr = run_cli("eval", "failures", "--db", str(db_path))
    assert rc == 0, f"stderr: {stderr}"
    assert "2 bench_fail rows (beside 1 ok)" in stdout
    assert "k_mm — 2 row(s)" in stdout
    assert "cuTensorMapEncodeTiled failed" in stdout
    assert "TMA=True" in stdout and "BM=" not in stdout  # shared knob only — BM differs across the rows


def test_failures_old_db_without_error_column(run_cli, tmp_path):
    """bench_fail rows from a pre-error-column DB cluster under a placeholder
    instead of failing the read."""
    db = tmp_path / "old.db"
    _make_tune_db(db, [("a", "k_mm", {"BM": 8}, 10.0)])
    _add_perf_row(db, "x", "k_mm", {"BM": 64, "TMA": True}, 2_000_000.0, status="bench_fail")
    rc, stdout, stderr = run_cli("eval", "failures", "--db", str(db))
    assert rc == 0, f"stderr: {stderr}"
    assert "(no error recorded)" in stdout


# --- eval prior ----------------------------------------------------------------


def test_prior_rejects_the_leaf_only_db_dataset(run_cli, tmp_path):
    """``--dataset db`` reads fully-decided leaf rows with no op identity or compile regime on them, so there is
    nothing to group a comparison set by. The same DB read as ``--dataset nodes`` has both, and the error says
    so rather than emitting an empty table."""
    db = tmp_path / "r.db"
    _make_tune_db(db, [("a", "k_matmul", {"BM": 8}, 30.0)])
    rc, stdout, stderr = run_cli("eval", "prior", "--dataset", "db", "--db", str(db))
    assert rc == 2
    assert "--dataset nodes" in stdout + stderr


def test_prior_reports_both_halves_over_benched_pools(run_cli, tmp_path):
    """``eval prior --dataset nodes`` renders the shared report over the node store: one cell per card and
    compile regime, labelled by prior half. With no fitted checkpoint the online half is named as absent
    rather than answered for — a cold model scores every row the same constant, which in a table is
    indistinguishable from a trained model that collapsed."""
    from emmy.compiler.pipeline.search.db import NodeRow, SearchDB

    db_path = tmp_path / "n.db"
    db = SearchDB(db_path)
    s = {"S_ext_free_prod": 1024.0, "S_ext_reduce_max": 512.0, "S_loop_depth": 2.0, "H_cc": 120.0, "H_opt": 3.0}
    db.record_nodes(
        [
            NodeRow("P", None, "ctx", "mm", s, 1.0, 1, gpu=_GPU, is_leaf=False),
            NodeRow("c8", "P", "ctx", "mm", {**s, "BN": 32, "BM": 8}, 3.0, 2, gpu=_GPU, is_leaf=True),
            NodeRow("c64", "P", "ctx", "mm", {**s, "BN": 32, "BM": 64}, 1.0, 2, gpu=_GPU, is_leaf=True),
        ]
    )
    db.close()
    rc, stdout, stderr = run_cli("eval", "prior", "--dataset", "nodes", "--db", str(db_path), "--online-file", str(tmp_path / "no.json"))
    assert rc == 0, f"stderr: {stderr}"
    assert "No fitted online prior" in stdout  # the online half is named absent, not reported as flat
    assert "ranking quality over benched pools" in stdout
    assert "offline" in stdout and "O3" in stdout
    # The per-fork view still runs beside the cells: it answers what a pool of benched leaves cannot, and it is
    # retired with the rest of the fork machinery rather than by the report landing.
    assert "per-fork view" in stdout and "node store: 3 nodes" in stdout
    assert "traceback" not in (stdout + stderr).lower()


def test_report_renders_both_dataset_kinds_with_their_own_columns():
    """The renderer picks its columns from the report's dataset, since that is what decided which metrics the
    cells carry. Both branches are exercised here rather than through a corpus build, which costs minutes."""
    import emmy.commands.eval as eval_cmd
    from emmy.compiler.pipeline.search.data.group import GoldenGroup, MeasuredGroup
    from emmy.compiler.pipeline.search.prior.report import EvalReport, golden_cells, measured_cells

    feats = [{"D_a": float(i)} for i in range(3)]
    by_d_a = lambda g: g.matrix(["D_a"])[:, 0]  # noqa: E731 — a one-line scorer stub
    measured = MeasuredGroup.from_measured("g/op@O3", "g", "op", 3.0, [30.0, 20.0, 10.0], feats)
    golden = GoldenGroup.from_dicts("g/k", "k", "warp", "g", "k", 0, feats, 50_000)

    lines: list[str] = []
    with patch.object(eval_cmd.logger, "info", lambda fmt, *a: lines.append(str(fmt) % a if a else str(fmt))):
        eval_cmd._emit_report(EvalReport({"dataset": "nodes", "source": "freeze"}, measured_cells("offline", [measured], by_d_a)))
        eval_cmd._emit_report(EvalReport({"dataset": "golden", "source": "corpus"}, golden_cells("online", [golden], by_d_a)))
    text = "\n".join(lines)

    assert "rho" in text and "regret@1" in text and "1.00x (1)" in text
    assert "SCREEN" in text  # the golden block says what a rank is not
    assert "top1" in text and "0/1" in text  # the golden ranks last under this scorer
    assert ">=10k" in text  # and its pool bucket, without which the rank is unreadable


def test_an_unmeasurable_metric_renders_a_dash_rather_than_a_number():
    """Nothing in the cell qualified — which must not read as a value of zero."""
    import emmy.commands.eval as eval_cmd
    from emmy.compiler.pipeline.search.data.group import MeasuredGroup
    from emmy.compiler.pipeline.search.prior.report import EvalReport, measured_cells

    one_row = MeasuredGroup.from_measured("g/op@O3", "g", "op", 3.0, [10.0], [{"D_a": 1.0}])
    cells = measured_cells("offline", [one_row], lambda g: g.matrix(["D_a"])[:, 0])

    lines: list[str] = []
    with patch.object(eval_cmd.logger, "info", lambda fmt, *a: lines.append(str(fmt) % a if a else str(fmt))):
        eval_cmd._emit_report(EvalReport({"dataset": "nodes", "source": "freeze"}, cells))

    assert cells[0].metrics["regret1"]["median"] is None
    assert "—" in "\n".join(lines)


def test_prior_writes_the_report_as_json(run_cli, tmp_path):
    """``--json`` writes the cells, so two runs are compared with ``diff`` instead of by eye.

    Also pins the pre-rename ``--prior`` spelling of ``--online-file``: it lives in shell profiles and scripts."""
    from emmy.compiler.pipeline.search.db import NodeRow, SearchDB

    db_path = tmp_path / "n.db"
    db = SearchDB(db_path)
    s = {"S_ext_free_prod": 1024.0, "S_ext_reduce_max": 512.0, "S_loop_depth": 2.0, "H_cc": 120.0, "H_opt": 3.0}
    db.record_nodes(
        [
            NodeRow("c8", None, "ctx", "mm", {**s, "BM": 8}, 3.0, 2, gpu=_GPU, is_leaf=True),
            NodeRow("c64", None, "ctx", "mm", {**s, "BM": 64}, 1.0, 2, gpu=_GPU, is_leaf=True),
        ]
    )
    db.close()
    out = tmp_path / "report.json"
    rc, _stdout, stderr = run_cli(
        "eval", "prior", "--dataset", "nodes", "--db", str(db_path), "--json", str(out), "--prior", str(tmp_path / "no.json")
    )
    assert rc == 0, f"stderr: {stderr}"
    obj = json.loads(out.read_text())
    assert obj["header"]["dataset"] == "nodes" and obj["header"]["groups"] == 1
    (cell,) = obj["cells"]
    assert cell["axes"] == {"half": "offline", "gpu": _GPU, "H_opt": "O3"}
    assert set(cell["metrics"]) == {"spearman", "regret1", "regret10"}
    assert cell["metrics"]["regret1"]["groups"] == 1


def test_nodes_dataset_rejected_by_db_only_subcommand(run_cli, tmp_path):
    """Widening the ``--dataset`` vocabulary with ``nodes`` must not leak into the
    db-only subcommands — ``eval variants`` still exits 2 on it."""
    db = tmp_path / "n.db"
    _make_tune_db(db, [("a", "k_mm", {"BM": 8}, 10.0)])
    rc, _stdout, _stderr = run_cli("eval", "variants", "--dataset", "nodes", "--db", str(db))
    assert rc == 2


def test_failures_none_recorded(run_cli, tmp_path):
    db = tmp_path / "clean.db"
    _make_tune_db(db, [("a", "k_mm", {"BM": 8}, 10.0)])
    rc, stdout, stderr = run_cli("eval", "failures", "--db", str(db))
    assert rc == 0, f"stderr: {stderr}"
    assert "No bench_fail rows" in stdout


def test_o3_reservoir_index_joins_db_rows_by_sig_and_knobs():
    """``_o3_reservoir_index`` keeps only ``H_opt=3`` reservoir rows and keys them
    so a DB sample with the same ``S_*`` signature + tunable knobs joins (the -O3
    re-bench never writes a ``perf`` row, so the reservoir is the only -O3 source)."""
    from emmy.commands.eval import _o3_reservoir_index, _variant_key
    from emmy.compiler.pipeline.search.data import Sample
    from emmy.compiler.pipeline.search.db import PerfSample

    stamped = {"BM": 8, "BN": 16, "S_ext_free_prod": 1024.0}

    class _P:
        _dataset = [
            ({**stamped, "H_opt": 3.0}, 9.0),
            ({**stamped, "H_opt": 1.0}, 12.0),  # -O1 sweep row — excluded
            ({"BM": 64, "BN": 16, "S_ext_free_prod": 1024.0, "H_opt": 3.0}, 7.0),
        ]

    o3 = _o3_reservoir_index(_P())
    assert len(o3) == 2
    db_sample = Sample.from_perf_sample(PerfSample(pretty="__global__ void k_m(float*)", knobs=stamped, latency_us=12.5))
    assert o3[_variant_key(db_sample)] == 9.0


def test_emit_variant_table_o3_column_and_deterministic_pick(caplog):
    """With a fake prior the pick is deterministic; the ``-O3 us`` column shows the
    reservoir latency for the matching config and ``—`` elsewhere."""
    import logging

    from emmy.commands.eval import _emit_variant_table, _variant_key
    from emmy.compiler.pipeline.search.data import Sample
    from emmy.compiler.pipeline.search.db import PerfSample

    samples = [
        Sample.from_perf_sample(PerfSample(pretty="__global__ void k_m(float*)", knobs={"BM": 8, "BN": 16}, latency_us=30.0)),
        Sample.from_perf_sample(PerfSample(pretty="__global__ void k_m(float*)", knobs={"BM": 16, "BN": 16}, latency_us=10.0)),
    ]

    class _P:  # predicts the slow BM=8 config fastest → pick is rank 2
        def mean_score(self, knobs):
            return knobs["BM"]

        def pick(self, rows):  # the real Prior.pick model-argmin fallback (no evidence here)
            scores = [self.mean_score(r) for r in rows]
            best_i = min(range(len(scores)), key=scores.__getitem__)
            return best_i, scores[best_i]

    o3 = {_variant_key(samples[0]): 9.5}
    with caplog.at_level(logging.INFO, logger="emmy.commands.eval"):
        _emit_variant_table("k_m", samples, _P(), n_fail=0, o3=o3, top=0)
    out = "\n".join(caplog.messages)
    assert "-O3 us" in out
    assert "9.5" in out and "—" in out
    assert "pick: rank 2/2, 3.00x of best" in out
    assert "<-- misses best" in out


def test_knob_columns_names_in_header_values_in_cells():
    """``knob_columns`` puts the knob name in the column header (canonical knob_sort_key
    order) and value-only cells (no ``NAME=`` prefix), blank where a row lacks a knob;
    ``render_table`` aligns the columns to the widest of header + cells."""
    from emmy.commands.table import knob_columns, render_table

    cols, cells = knob_columns(
        [
            {"TILE": ("n16", False), "REDUCE": ("b32", False)},
            {"TILE": ("n32", False), "STAGE": ("d2/smem-async", False)},
        ]
    )
    assert [c.name for c in cols] == ["TILE", "REDUCE", "STAGE"]  # canonical KNOB_ORDER
    lines = render_table(cols, cells)
    assert lines[0].split() == ["TILE", "REDUCE", "STAGE"]  # header row carries the names
    assert lines[1].split() == ["n16", "b32"]  # values only, no "TILE=" prefix; trailing STAGE blank stripped
    assert "TILE=" not in lines[1]
    assert lines[2].split() == ["n32", "d2/smem-async"]  # REDUCE column blank between TILE and STAGE


def test_render_table_ansi_aware_width():
    """A coloured cell is padded by its *visible* length, so embedded ANSI codes never
    throw off column alignment (right- and left-aligned columns both line up)."""
    import re  # noqa: PLC0415

    from emmy.commands.table import Col, render_table

    lines = render_table([Col("a", "r"), Col("b")], [["1", "x"], [("22", "\033[31m"), "y"]])
    plain = [re.sub(r"\033\[[0-9]+m", "", line) for line in lines]
    assert plain[1] == " 1  x"  # "1" right-aligned in a width-2 column
    assert plain[2] == "22  y"  # coloured "22" fills the column; "y" still aligns under "x"


def test_bare_families_canonicalizes_axis_suffixed_knobs():
    """The golden-reproduction table compares the greedy pick against golden YAML rows,
    whose knobs are spelled bare — but resolved ops carry axis-suffixed codec keys
    (``TILE@a2``). ``_bare_families`` bridges the two; compared raw, every found cell
    rendered ``-`` over perfectly good picks (the post-rebuild 0/29 table)."""
    from emmy.commands.eval import _bare_families

    got = _bare_families({"TILE@a2": "f2x4", "WORK": "t16x8", "STAGE@a2": "d3/smem-tma", "REDUCE@a2": "g2a"})
    assert got == {"TILE": "f2x4", "WORK": "t16x8", "STAGE": "d3/smem-tma", "REDUCE": "g2a"}
    # bare keys pass through; first key wins a family collision (single-node picks in practice)
    assert _bare_families({"TILE": "a", "TILE@x": "b"}) == {"TILE": "a"}


def test_offer_audit_flags_unrealized_entries_and_fall_through(monkeypatch, caplog):
    """``eval golden``'s offer audit, over the strict-identity tier: an entry whose spelled row
    equals no enumerated leaf is UNREALIZED (tolerable while an offered sibling floors the
    target); a target whose entries are ALL unrealized FALLS THROUGH the verified tier and the
    audit returns True (the command exits 1) — the 4090 ``attention.hd512.s4096`` class, caught
    at record time instead of in production benches. The guaranteed-unrealizable row here is a
    fabricated ``TILE`` fragment: no enumeration offers it, so no leaf can equal the row."""
    import logging

    import pytest

    pytest.importorskip("torch")
    import emmy.commands.eval as eval_cmd
    from emmy import config
    from emmy.commands.trace import trace_inline_code
    from emmy.compiler.context import Context
    from emmy.compiler.ir.base import InputOp
    from emmy.compiler.pipeline.knob import stamp_schedule_families
    from emmy.compiler.pipeline.search.golden import load_golden_records
    from emmy.compiler.pipeline.search.golden_eval import enumerate_graph
    from emmy.compiler.torch_wire import graph_to_wire

    gpu, cap = "NVIDIA GeForce RTX 5090", (12, 0)
    with config.nvcc_flags_override(""):  # the deployable -O3 regime the tier is gated on
        ctx = Context.from_target(cap, gpu_name=gpu)

    def enumerated_row(graph):
        rows = enumerate_graph(graph.copy(), ctx).rows
        return stamp_schedule_families(next(r for r in rows if str(r.get("WORK", "")).startswith("w")))

    def records(graph, name, entries):
        origins = [nid for nid, node in graph.nodes.items() if not isinstance(node.op, InputOp)]
        return load_golden_records(
            {
                "gpu_name": gpu,
                "compute_cap": list(cap),
                "model": "org/model",
                "programs": [graph_to_wire(graph)],
                "configs": [
                    {
                        "program": 0,
                        "target": {"origins": origins},
                        "realizations": [
                            {
                                "name": name,
                                "bindings": {},
                                "pins": {"FAST_MATH": False},
                                "knobs": knobs,
                                "measurements": {"emmy_us": us, "reference_us": 30.0, "reference_backend": "cublas"},
                            }
                            for knobs, us in entries
                        ],
                    }
                ],
            }
        )

    def matmul(m):
        code = f"torch.matmul(torch.randn({m},128, dtype=torch.float16), torch.randn(128,{m}, dtype=torch.float16))"
        return trace_inline_code(code)["graph"]

    # Two DIFFERENT extents, so the two targets carry different structural identities and the
    # floored target's offered row cannot decide the orphan's fork.
    small, big = matmul(64), matmul(256)
    drifted = {**enumerated_row(small), "TILE": "mma_m16n8k16_f16_f32/f9x9"}  # a fragment nothing offers
    floored = records(small, "audit.floored", [(drifted, 10.0), (enumerated_row(small), 20.0)])
    orphan = records(big, "audit.orphan", [({**enumerated_row(big), "TILE": "mma_m16n8k16_f16_f32/f9x9"}, 10.0)])

    with caplog.at_level(logging.INFO, logger="emmy.commands.eval"):
        fell = eval_cmd._emit_offer_audit(floored + orphan)

    assert fell is True
    msgs = [r.getMessage() for r in caplog.records]
    assert any("UNREALIZED" in m and "audit.floored" in m and "deploy floor" in m for m in msgs)
    assert any("UNREALIZED" in m and "audit.orphan" in m and "NO offered sibling" in m for m in msgs)
    assert any("FALL-THROUGH" in m and "audit.orphan" in m for m in msgs)
    assert not any("FALL-THROUGH" in m and "audit.floored" in m for m in msgs)

    # A set whose every entry equals an enumerated leaf is clean.
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="emmy.commands.eval"):
        fell = eval_cmd._emit_offer_audit([floored[1]])
    assert fell is False
    msgs = [r.getMessage() for r in caplog.records]
    assert any("equal an enumerated leaf" in m for m in msgs)
    assert not any("UNREALIZED" in m or "FALL-THROUGH" in m for m in msgs)
