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
    assert "--golden" in stdout + stderr

    golden = tmp_path / "given.yaml"
    configured = tmp_path / "configured.yaml"
    config = tmp_path / "release.env"
    config.write_text(
        f"SERVE_MODEL=org/model\nSERVE_GPU=NVIDIA-Test\nSERVE_GOLDEN_FILE={configured}\n"
        "SERVE_MAX_NUM_BATCHED_TOKENS=32\nSERVE_DECODE_BUCKET=8\n"
    )
    rc, stdout, stderr = run_cli("eval", "golden", "--golden", str(golden), "--serving-config", str(config))
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
        return {"pre1": None}

    monkeypatch.setattr(twins, "capture_twin_graphs", fake_capture)
    monkeypatch.setattr(audit, "audit_card", fake_audit)

    eval_cmd.handle_eval_golden(SimpleNamespace(golden=str(golden), serving_config=str(config)))

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
        eval_cmd.handle_eval_golden(SimpleNamespace(golden=str(golden), serving_config=str(config)))
    assert exc.value.code == 1


def test_eval_golden_fails_when_a_twin_is_not_decided_by_the_golden_rows(monkeypatch, tmp_path, caplog):
    """The serving-matrix compile runs under strict evidence: a twin with a fork no golden row
    decides is a gate failure naming the twin and the kernel, however many others deploy."""
    import logging
    from types import SimpleNamespace

    import pytest

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
    monkeypatch.setattr(twins, "capture_twin_graphs", lambda source, **kwargs: {"pre1": object(), "pre2": object()})
    verdict = {"pre1": None, "pre2": "EvidenceError: strict evidence: kernel 'k_linear' (node 'n') has no measured evidence"}
    monkeypatch.setattr(audit, "audit_card", lambda graphs, gpu_name, compute_cap, *, goldens: verdict)

    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as exc:
        eval_cmd.handle_eval_golden(SimpleNamespace(golden=str(golden), serving_config=str(config)))
    assert exc.value.code == 1
    assert any("pre2" in r.message and "k_linear" in r.message for r in caplog.records)


def test_offer_audit_flags_unrealized_entries(monkeypatch, caplog):
    """``eval golden``'s offer audit is the strict decode per entry: an entry whose spelled row
    equals no enumerated leaf of its own target is UNREALIZED and fails the gate — it is no
    evidence a deploy can use (the 4090 ``attention.hd512.s4096`` class, caught at record time
    instead of in production benches) — while a set of entries that all decode passes. The
    guaranteed-unrealizable row here is a fabricated ``TILE`` fragment: no enumeration offers it,
    so no leaf can equal the row."""
    import logging

    import pytest

    pytest.importorskip("torch")
    import emmy.commands.eval as eval_cmd
    from emmy import config
    from emmy.commands.trace import trace_inline_code
    from emmy.compiler.context import Context
    from emmy.compiler.ir.base import InputOp
    from emmy.compiler.ir.schedule import Tile, Work
    from emmy.compiler.ir.schedule import classic_projection as classic
    from emmy.compiler.pipeline.knob import complete_kernel_row
    from emmy.compiler.pipeline.search.golden import load_golden_records
    from emmy.compiler.pipeline.search.golden_eval import enumerate_graph
    from emmy.compiler.torch_wire import graph_to_wire

    gpu, cap = "NVIDIA GeForce RTX 5090", (12, 0)
    with config.nvcc_flags_override(""):  # the deployable -O3 regime the tier is gated on
        ctx = Context.from_target(cap, gpu_name=gpu)

    # This test owns the audit verdicts, not catalog breadth. Keep Algorithm 1 intact over a
    # bounded independent product so every audit compile remains a fast unit test.
    warp = Tile.parse("mma_m16n8k16_f16_f32/f2x2/k2", Work.parse("w2x1"))
    monkeypatch.setattr(classic, "scalar_tile_moves", lambda: [Tile()])
    monkeypatch.setattr(classic, "warp_tile_moves", lambda atoms: [warp] if warp.atom.name in atoms else [])
    monkeypatch.setattr(classic, "coop_reduce_moves", lambda: [])
    monkeypatch.setattr(classic, "stage_moves", lambda *, warp, ctx=None: [])
    monkeypatch.setattr(classic, "raster_moves", lambda: [""])

    def enumerated_row(graph):
        rows = enumerate_graph(graph.copy(), ctx).rows
        return complete_kernel_row(next(r for r in rows if str(r.get("WORK", "")).startswith("w")))

    def drifted_tile(row):
        row = dict(row)
        tile_key = next(key for key in row if key.split("@", 1)[0] == "TILE")
        row[tile_key] = "mma_m16n8k16_f16_f32/f9x9"
        return row

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
    small_row, big_row = enumerated_row(small), enumerated_row(big)
    drifted = drifted_tile(small_row)  # a fragment nothing offers
    floored = records(small, "audit.floored", [(drifted, 10.0), (small_row, 20.0)])
    orphan = records(big, "audit.orphan", [(drifted_tile(big_row), 10.0)])

    with caplog.at_level(logging.INFO, logger="emmy.commands.eval"):
        failed = eval_cmd._emit_offer_audit(floored + orphan)

    assert failed is True
    msgs = [r.getMessage() for r in caplog.records]
    assert any("UNREALIZED" in m and "audit.floored" in m for m in msgs)
    assert any("UNREALIZED" in m and "audit.orphan" in m for m in msgs)
    assert sum("UNREALIZED" in m for m in msgs) == 2, "the offered sibling of the floored target decodes"
    assert eval_cmd._emit_offer_audit(records(small, "audit.good", [(small_row, 20.0)])) is False

    # A set whose every entry equals an enumerated leaf is clean.
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="emmy.commands.eval"):
        fell = eval_cmd._emit_offer_audit([floored[1]])
    assert fell is False
    msgs = [r.getMessage() for r in caplog.records]
    assert any("equal an enumerated leaf" in m for m in msgs)
    assert not any("UNREALIZED" in m or "FALL-THROUGH" in m for m in msgs)
