"""Bench-to-node recording — offer-site pool keying, whole-variant leaf semantics, the
run --bench row filter, and the node store's quality-aware leaf replacement.

The compile fixture lowers a real f16 matmul to CUDA dialect on the CPU (codegen is
text; no GPU is touched), so the offer-site recovery and knob stamps under test are the
genuine pipeline artifacts, not hand-built fakes. Bench results ARE faked — a
``BenchmarkResult`` duck with per-launch samples."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from emmy.compiler.context import Context
from emmy.compiler.pipeline.search.bench_record import (
    FAIL_SENTINEL_US,
    bench_leaves,
    meets_quality_bar,
    mint_bench_run_id,
    record_bench_leaves,
)
from emmy.compiler.pipeline.search.data.freeze import freeze_reason
from emmy.compiler.pipeline.search.db import SearchDB
from tests.compiler.pipeline.search.conftest import node_row

_GPU = "NVIDIA GeForce RTX 5090"


@pytest.fixture(scope="module")
def compiled_matmul():
    from emmy.commands.trace import graph_from_code
    from emmy.compiler.pipeline import CUDA_PASSES, Pipeline

    code = "torch.matmul(torch.randn(512,512,dtype=torch.float16), torch.randn(512,512,dtype=torch.float16))"
    graph, _, _ = graph_from_code(code)
    return Pipeline.build(CUDA_PASSES).run(graph)


def _fake_bench(n_launches: int = 1, time_ms: float = 0.5) -> SimpleNamespace:
    per_launch = [
        SimpleNamespace(idx=i, kernel_name=f"k{i}", time_ms=time_ms, samples=(time_ms, time_ms * 1.04, time_ms * 0.96))
        for i in range(n_launches)
    ]
    return SimpleNamespace(time_ms=time_ms * n_launches, per_launch=per_launch)


def _n_cuda_kernels(compiled) -> int:
    from emmy.compiler.ir.cuda.ir import CudaOp

    return sum(1 for n in compiled.nodes.values() if isinstance(n.op, CudaOp))


# ---------------------------------------------------------------------------
# bench_leaves — offer-site keying + whole-variant values
# ---------------------------------------------------------------------------


def test_bench_leaves_keys_by_offer_site(compiled_matmul) -> None:
    from emmy.compiler.ir.cuda.ir import CudaOp
    from emmy.compiler.pipeline.search.bench_record import _offer_site
    from emmy.compiler.structural import digest

    n_kernels = _n_cuda_kernels(compiled_matmul)
    leaves = bench_leaves(compiled_matmul, _fake_bench(n_kernels))
    assert leaves, "a compiled matmul must yield at least one recordable leaf"
    ops = [n.op for n in compiled_matmul.nodes.values() if isinstance(n.op, CudaOp)]
    site = _offer_site(ops[0])
    expected_sig = digest(*sorted((k, v) for k, v in site.knobs.items() if k.startswith("S_")))
    assert leaves[0].op_sig == expected_sig
    # The pool key comes from the PRE-DESCENT site: descent stamps extra S_* deltas,
    # so the terminal op's own S_* digest must NOT be the pool key.
    own_sig = digest(*sorted((k, v) for k, v in ops[0].knobs.items() if k.startswith("S_")))
    assert leaves[0].op_sig != own_sig
    # Realized tunables ride the leaf's knob dict (S_* stamps included).
    assert any(not k.startswith(("S_", "H_")) for k in leaves[0].knobs)
    assert any(k.startswith("S_") for k in leaves[0].knobs)


def test_bench_leaves_values_and_stats(compiled_matmul) -> None:
    n_kernels = _n_cuda_kernels(compiled_matmul)
    leaves = bench_leaves(compiled_matmul, _fake_bench(n_kernels, time_ms=0.5))
    # All kernels of the variant share one offer site -> ONE leaf at the summed time.
    assert len(leaves) == 1
    leaf = leaves[0]
    assert leaf.status == "ok"
    assert leaf.value_us == pytest.approx(500.0 * n_kernels)
    if n_kernels == 1:
        assert leaf.n_samples == 3 and leaf.variance > 0.0
    else:  # multi-kernel groups can't sum per-window samples honestly
        assert leaf.n_samples is None and leaf.variance is None


def test_bench_leaves_fail_sentinel(compiled_matmul) -> None:
    leaves = bench_leaves(compiled_matmul, None, status="bench_fail")
    assert len(leaves) == 1
    assert leaves[0].status == "bench_fail"
    assert leaves[0].value_us == FAIL_SENTINEL_US
    assert leaves[0].n_samples is None and leaves[0].variance is None


def _compile_pinned(monkeypatch, pins: dict) -> object:
    from emmy import config
    from emmy.commands.trace import graph_from_code
    from emmy.compiler.pipeline import CUDA_PASSES, Pipeline

    for knob, value in pins.items():
        monkeypatch.setenv(config.knob_var(knob), value)
    graph, _, _ = graph_from_code("torch.matmul(torch.randn(512,512), torch.randn(512,512))")
    ctx = Context.from_target((8, 9), gpu_name="NVIDIA GeForce RTX 4090")
    return Pipeline.build(CUDA_PASSES).run(graph, ctx=ctx)


def test_mma_path_records_and_joins_the_scalar_pool(monkeypatch) -> None:
    """The 2026-07-16 4090 regression: the tensor-core (mma) tile-lowering preserves no
    LoopOp in ``.source``, so the loop-only offer-site predicate silently dropped every
    mma-path kernel — a golden sweep recorded ZERO rows, exactly the fast variants the
    feature exists to capture. The tile-dialect fallback must (a) recover a site, (b)
    group the split-K main+combine pair into ONE whole-variant leaf, and (c) produce
    the same ``op_sig`` as the scalar/loop path for the same shape (one pool)."""
    mma = _compile_pinned(monkeypatch, {"TILE": "n16x8/f4x8", "REDUCE": "g2k", "STAGE": "d2/cp/ring", "RASTER": "", "WSPEC": ""})
    n_mma_kernels = _n_cuda_kernels(mma)
    assert n_mma_kernels == 2, "the pinned split-K config must compile to a main + combine pair"
    mma_leaves = bench_leaves(mma, _fake_bench(n_mma_kernels, time_ms=0.5))
    assert len(mma_leaves) == 1, "mma-path kernels must be recordable (split-K pair -> one leaf)"
    # The combine kernel has NO S_* provenance anywhere in its chain — it must attribute
    # to its producer through the graph edge, so the leaf is the WHOLE-variant value
    # (partial-only values are fast-biased against the tune's whole-slice leaves).
    assert mma_leaves[0].value_us == pytest.approx(1000.0)
    assert mma_leaves[0].n_samples is None and mma_leaves[0].variance is None  # multi-kernel group
    scalar = _compile_pinned(monkeypatch, {"TILE": "n16x8/f2x4", "REDUCE": "", "STAGE": "d2/cp/ring", "RASTER": "", "WSPEC": ""})
    scalar_leaves = bench_leaves(scalar, _fake_bench(_n_cuda_kernels(scalar)))
    assert len(scalar_leaves) == 1
    assert mma_leaves[0].op_sig == scalar_leaves[0].op_sig  # same shape, same pool, either lowering path


# ---------------------------------------------------------------------------
# record_bench_leaves — tune-recipe keying, freeze pickup
# ---------------------------------------------------------------------------


def test_record_round_trip_into_node_store(compiled_matmul, tmp_path) -> None:
    ctx = Context.from_target((12, 0), gpu_name=_GPU)
    leaves = bench_leaves(compiled_matmul, _fake_bench(_n_cuda_kernels(compiled_matmul), time_ms=0.5))
    db_path = tmp_path / "tune.db"
    n = record_bench_leaves(db_path, ctx, leaves)
    assert n == len(leaves)
    db = SearchDB.open_readonly(db_path)
    try:
        rows = list(db.iter_nodes())
    finally:
        db.close()
    assert len(rows) == len(leaves)
    row = rows[0]
    assert row.is_leaf is True and row.parent_key is None and row.depth == 0
    assert row.status == "ok" and row.run_id.startswith("bench-")
    assert row.op_sig == leaves[0].op_sig
    assert row.gpu == _GPU
    assert row.context_key == ctx.structural_key()
    assert any(k.startswith("H_") for k in row.features)  # regime features stamped
    assert freeze_reason(row) is None  # the measurement freeze picks it up unchanged


def test_record_fail_leaf_kept_as_negative(compiled_matmul, tmp_path) -> None:
    ctx = Context.from_target((12, 0), gpu_name=_GPU)
    leaves = bench_leaves(compiled_matmul, None, status="bench_fail")
    record_bench_leaves(tmp_path / "tune.db", ctx, leaves)
    db = SearchDB.open_readonly(tmp_path / "tune.db")
    try:
        rows = list(db.iter_nodes())
    finally:
        db.close()
    assert len(rows) == 1 and rows[0].status == "bench_fail"
    assert freeze_reason(rows[0]) is None  # negatives freeze too


def test_quality_bar_and_run_id() -> None:
    assert meets_quality_bar(5, 20) and meets_quality_bar(10, 100)
    assert not meets_quality_bar(4, 100) and not meets_quality_bar(10, 19)
    assert mint_bench_run_id().startswith("bench-")


# ---------------------------------------------------------------------------
# the run --bench row filter (_recordable_bench_leaves)
# ---------------------------------------------------------------------------


def test_recordable_rows_filter(compiled_matmul) -> None:
    from emmy.commands.run import _recordable_bench_leaves

    bench = _fake_bench(_n_cuda_kernels(compiled_matmul))
    ok = SimpleNamespace(graph=compiled_matmul, bench=bench, flags=[], status="ok")
    flagged = SimpleNamespace(graph=compiled_matmul, bench=bench, flags=["wrong-answer: rel err 0.2"], status="ok")
    unmatched = SimpleNamespace(graph=compiled_matmul, bench=None, flags=["unreproducible pin: ..."], status="pin_unmatched")
    no_graph = SimpleNamespace(graph=None, bench=None, flags=["compile failed: boom"], status="bench_fail")
    failed = SimpleNamespace(graph=compiled_matmul, bench=None, flags=["bench_fail: watchdog"], status="bench_fail")
    iso = SimpleNamespace(graph=compiled_matmul, bench=bench, flags=[], status="ok")

    leaves = _recordable_bench_leaves([ok, flagged, unmatched, no_graph, failed], iso)
    # iso + ok record as ok; failed records as a negative; flagged/unmatched/no-graph never.
    assert [leaf.status for leaf in leaves] == ["ok", "ok", "bench_fail"]


# ---------------------------------------------------------------------------
# record_nodes quality-aware leaf replacement
# ---------------------------------------------------------------------------


def _quality_row(value_us: float, *, measured_at: str, variance, n_samples):
    return node_row("k-quality", value_us=value_us, measured_at=measured_at, variance=variance, n_samples=n_samples)


def _value(db: SearchDB) -> float:
    (row,) = list(db.iter_nodes())
    return row.value_us


def test_leaf_quality_guard_blocks_worse_newer_measurement() -> None:
    db = SearchDB()
    db.record_nodes([_quality_row(500.0, measured_at="2026-07-01T00:00:00+00:00", variance=1.0, n_samples=50)])
    # Newer but unambiguously worse (fewer samples AND higher variance): kept out.
    db.record_nodes([_quality_row(700.0, measured_at="2026-07-02T00:00:00+00:00", variance=9.0, n_samples=3)])
    assert _value(db) == 500.0
    # Newer and comparable quality: plain newest-wins.
    db.record_nodes([_quality_row(510.0, measured_at="2026-07-03T00:00:00+00:00", variance=0.8, n_samples=50)])
    assert _value(db) == 510.0
    # Newer with UNKNOWN quality (no stats — e.g. an -O3 re-bench): newest-wins, so
    # honest re-measurement still heals stale rows.
    db.record_nodes([_quality_row(520.0, measured_at="2026-07-04T00:00:00+00:00", variance=None, n_samples=None)])
    assert _value(db) == 520.0
