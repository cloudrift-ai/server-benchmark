"""``run --bench`` records one row per benched KERNEL.

The recorder used to group a variant's kernels under the offer site they lowered from and
record ONE leaf holding their summed launch time. For a variant a structural fork made
several kernels of — a cross-CTA split's main + combine — that row described a cost no
kernel ran at, and it filed both pieces under the identity of the op the split replaced, a
kernel that did not run at all.

Now every kernel is its own row, keyed by the stamp it was born with, which is the rule the
tune walk records by (``policy/mcts._measured_kernel_rows``). That is what lets a kernel
benched here and the same kernel tuned by the search meet on one ``node_key``.
"""

from __future__ import annotations

from types import SimpleNamespace

from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.cuda.ir import CudaOp
from emmy.compiler.pipeline.passes.identity import kernel_sig
from emmy.compiler.pipeline.search.bench_record import FAIL_SENTINEL_US, bench_leaves

# What a cross-CTA split leaves behind: two kernels, each stamped at birth with its own
# shape, both descended from the op the split consumed.
PRESPLIT = {"S_n_load": 2.0, "S_n_accum": 1.0, "S_n_loop": 3.0}
MAIN = {"S_n_load": 2.0, "S_n_accum": 1.0, "S_n_loop": 4.0}
COMBINE = {"S_n_load": 1.0, "S_n_accum": 1.0, "S_n_loop": 1.0}
# Tile materialization merges this onto the op it builds — a stamp the kernel was not born
# with, and the reason a row's identity cannot come from the realized op's own knobs.
DESCENT = {"S_warp_eligible": 1.0}


def _ancestors(*stamped: dict):
    """A rewrite chain, nearest first: one tile-dialect op then one loop-dialect op per entry."""
    chain = None
    for stamps in reversed(stamped):
        chain = SimpleNamespace(dialect="loop", knobs=dict(stamps), source=chain)
    return chain


def _kernel(name: str, knobs: dict, *stamped: dict) -> CudaOp:
    """A compiled kernel carrying ``knobs``, descended from ``stamped`` (nearest first).

    The rendered source spells the kernel's own knobs, because ``CudaOp.cache_key`` digests
    the source and the launch geometry with the kernel NAME normalized out — two stubs
    differing only by name genuinely share a key, exactly as two renamings of one kernel
    should, so a per-kernel ``perf`` row needs bodies that actually differ."""
    body = "; ".join(f"{key}={value}" for key, value in sorted(knobs.items()))
    return CudaOp(
        kernel_name=name,
        kernel_source=f"__global__ void {name}() {{ /* {body} */ }}",
        knobs=knobs,
        source=_ancestors(*stamped),
    )


def _graph(*ops: CudaOp) -> Graph:
    graph = Graph()
    for idx, op in enumerate(ops):
        graph.add_node(op=op, inputs=[], output=Tensor(f"o{idx}", (1,)), node_id=f"k{idx}")
    return graph


def _bench(*launches: tuple[float, list[float] | None], captured: bool = True):
    return SimpleNamespace(
        captured=captured,
        per_launch=[SimpleNamespace(time_ms=ms, samples=samples) for ms, samples in launches],
    )


def test_a_splits_kernels_are_two_rows_with_their_own_latencies() -> None:
    """The whole defect in one case: two kernels, two rows, neither holding the 12 µs total."""
    graph = _graph(
        _kernel("main", {**MAIN, **DESCENT, "WORK": "w4x2"}, MAIN, PRESPLIT),
        _kernel("combine", {**COMBINE, **DESCENT, "WORK": ""}, COMBINE, PRESPLIT),
    )

    leaves = bench_leaves(graph, _bench((0.010, None), (0.002, None)))

    assert [leaf.stats.median for leaf in leaves] == [10.0, 2.0]
    assert [leaf.op_sig for leaf in leaves] == [kernel_sig(MAIN), kernel_sig(COMBINE)]
    assert [leaf.knobs["WORK"] for leaf in leaves] == ["w4x2", ""], "each row carries its own kernel's knobs"


def test_a_piece_is_keyed_by_its_own_stamp_not_the_op_the_split_replaced() -> None:
    graph = _graph(_kernel("main", {**MAIN, **DESCENT}, MAIN, PRESPLIT))

    [leaf] = bench_leaves(graph, _bench((0.010, None)))

    assert leaf.op_sig == kernel_sig(MAIN)
    assert leaf.op_sig != kernel_sig(PRESPLIT), "the op a split consumed is a different kernel, and it did not run"


def test_identity_excludes_the_stamp_descent_added() -> None:
    """A realized op's own knobs carry ``S_warp_eligible``; the kernel's identity does not."""
    graph = _graph(_kernel("main", {**MAIN, **DESCENT}, MAIN))

    [leaf] = bench_leaves(graph, _bench((0.010, None)))

    assert leaf.op_sig == kernel_sig(MAIN)
    assert leaf.op_sig != kernel_sig(leaf.knobs)


def test_each_kernel_keeps_its_own_bench_statistics() -> None:
    """Stats used to carry only for single-kernel groups, because a summed variance would be
    fiction. Nothing is summed now, so every kernel keeps its own."""
    graph = _graph(_kernel("main", MAIN, MAIN), _kernel("combine", COMBINE, COMBINE))

    leaves = bench_leaves(graph, _bench((0.010, [0.010, 0.012]), (0.002, [0.002, 0.002])))

    assert [leaf.stats.n_samples for leaf in leaves] == [2, 2]
    assert leaves[0].stats.variance > 0 and leaves[1].stats.variance == 0


def test_a_kernel_with_no_stamp_in_its_chain_is_skipped() -> None:
    """No stamp means no identity to file the row under — and nothing to attribute it to."""
    graph = _graph(_kernel("main", MAIN, MAIN), _kernel("helper", {}))

    leaves = bench_leaves(graph, _bench((0.010, None), (0.002, None)))

    assert [leaf.op_sig for leaf in leaves] == [kernel_sig(MAIN)]


def test_a_graph_with_no_stamps_at_all_warns_rather_than_recording_silence(caplog) -> None:
    graph = _graph(_kernel("main", {}), _kernel("helper", {}))

    with caplog.at_level("WARNING"):
        assert bench_leaves(graph, _bench((0.010, None), (0.002, None))) == []
    assert "not preserving op provenance" in caplog.text


def test_a_single_kernel_failure_records_its_negative() -> None:
    graph = _graph(_kernel("main", {**MAIN, **DESCENT}, MAIN))

    [leaf] = bench_leaves(graph, None, status="bench_fail")

    assert leaf.status == "bench_fail"
    assert leaf.stats.median == FAIL_SENTINEL_US
    assert leaf.op_sig == kernel_sig(MAIN)


def test_a_multi_kernel_failure_records_nothing() -> None:
    """One sentinel spread across several kernels would file a number none of them measured;
    the failure belongs to the variant."""
    graph = _graph(_kernel("main", MAIN, MAIN), _kernel("combine", COMBINE, COMBINE))

    assert bench_leaves(graph, None, status="bench_fail") == []


def test_a_kernel_with_no_launch_timing_is_skipped_without_taking_the_others_down() -> None:
    graph = _graph(_kernel("main", MAIN, MAIN), _kernel("combine", COMBINE, COMBINE))

    leaves = bench_leaves(graph, _bench((0.010, None)))

    assert [leaf.op_sig for leaf in leaves] == [kernel_sig(MAIN)]


def test_a_leaf_carries_the_kernels_perf_key_and_capture_flag() -> None:
    """The perf key is the kernel's ``cache_key`` — the same identity the tune writes its own
    measured row under, which is what lets the two meet on one row instead of racing."""
    graph = _graph(_kernel("main", MAIN, MAIN), _kernel("combine", COMBINE, COMBINE))

    leaves = bench_leaves(graph, _bench((0.010, None), (0.002, None)))

    assert [leaf.op_key for leaf in leaves] == [node.op.cache_key() for node in graph.nodes.values()]
    assert leaves[0].op_key != leaves[1].op_key
    assert all(leaf.captured for leaf in leaves)
    assert not bench_leaves(graph, _bench((0.010, None), (0.002, None), captured=False))[0].captured


# --- what reaches which table ---------------------------------------------------------------
#
# ``node`` is training data and takes every leaf; ``perf`` is the measured evidence an ordinary
# compile reads to decide a fork, so it takes only a bench that succeeded under graph capture.


def _ctx():
    from emmy.compiler.context import Context

    return Context((8, 9))


def _record(tmp_path, *leaves):
    from emmy.compiler.pipeline.search.bench_record import record_bench_leaves
    from emmy.compiler.pipeline.search.db import SearchDB

    path = tmp_path / "autotune.db"
    counts = record_bench_leaves(path, _ctx(), list(leaves))
    return counts, SearchDB.open_readonly(path)


def _tuned(name: str, work: str, us: float):
    """One benched kernel of a two-way fork, distinguished by its ``WORK`` row."""
    graph = _graph(_kernel(name, {**MAIN, "WORK": work}, MAIN))
    [leaf] = bench_leaves(graph, _bench((us / 1000.0, [us / 1000.0] * 4)))
    return leaf


def test_a_launch_without_samples_keeps_its_median_and_no_sample_count(tmp_path) -> None:
    """``time_ms`` already IS the median of the measured iters, so a launch that reported no
    per-iter list loses nothing but the spread — which must read as unknown, not as zero.

    The stored side is the one that matters: ``record_nodes``' quality guard only holds when
    BOTH sides are known, so a row claiming zero samples at zero variance would read as "not
    worse" than every stored leaf and displace tune-grade data unconditionally."""
    graph = _graph(_kernel("main", MAIN, MAIN))

    [leaf] = bench_leaves(graph, _bench((0.010, None)))

    assert leaf.stats.median == leaf.stats.min == leaf.stats.mean == 10.0
    assert leaf.stats.n_samples == 0

    _, db = _record(tmp_path, leaf)
    [row] = db.iter_nodes()
    assert row.value_us == 10.0
    assert row.n_samples is None and row.variance is None, "an unknown spread must not store as zero"
    db.close()


def test_a_recorded_bench_row_becomes_evidence_a_later_compile_reads(tmp_path) -> None:
    """The whole point: a sweep's measurement has to be able to decide a fork, and only the
    ``perf`` table is consulted while compiling."""
    from emmy.compiler.pipeline.search.policy.greedy import _db_measured_index, _db_measured_pick

    (nodes, measured), db = _record(tmp_path, _tuned("slow", "w1x1", 40.0), _tuned("fast", "w4x2", 10.0))

    assert (nodes, measured) == (2, 2)
    candidates = [{**MAIN, "WORK": "w1x1"}, {**MAIN, "WORK": "w4x2"}]
    assert _db_measured_pick(_db_measured_index(db, _ctx()), candidates) == (1, 10.0)
    db.close()


def test_an_uncaptured_measurement_is_training_data_but_not_evidence(tmp_path) -> None:
    """Capture can hold for one pinned row and fall back for the next inside one sweep, and the
    evidence table compares medians across configs."""
    graph = _graph(_kernel("main", MAIN, MAIN))
    [leaf] = bench_leaves(graph, _bench((0.010, None), captured=False))

    (nodes, measured), db = _record(tmp_path, leaf)

    assert (nodes, measured) == (1, 0)
    assert len(list(db.iter_nodes())) == 1
    assert list(db.iter_perf(_ctx().structural_key(), backend="cuda")) == []
    db.close()


def test_a_failed_bench_is_training_data_but_not_evidence(tmp_path) -> None:
    """A stored failure would also make a later tune report the terminal failed without
    benching it: the tune's bench cache hits on any row it finds."""
    graph = _graph(_kernel("main", MAIN, MAIN))
    [leaf] = bench_leaves(graph, None, status="bench_fail")

    (nodes, measured), db = _record(tmp_path, leaf)

    assert (nodes, measured) == (1, 0)
    assert [row.status for row in db.iter_nodes()] == ["bench_fail"]
    assert db.lookup_perf(_ctx().structural_key(), leaf.op_key, backend="cuda") is None
    db.close()


def test_a_bench_row_and_a_tune_row_for_one_kernel_meet_on_one_measured_row(tmp_path) -> None:
    """Both writers key by the kernel's ``cache_key``, so they arbitrate instead of racing:
    keep-best-``ok`` holds whichever measured it faster."""
    from emmy.compiler.pipeline.search.bench_record import record_bench_leaves
    from emmy.compiler.pipeline.search.db import PerfStats, SearchDB

    leaf = _tuned("main", "w4x2", 10.0)
    path = tmp_path / "autotune.db"
    db = SearchDB(path)
    db.record_perf(_ctx().structural_key(), leaf.op_key, backend="cuda", status="ok", stats=PerfStats.point(25.0), captured=True)
    db.close()

    record_bench_leaves(path, _ctx(), [leaf])
    db = SearchDB.open_readonly(path)
    assert db.lookup_perf(_ctx().structural_key(), leaf.op_key, backend="cuda").stats.median == 10.0
    db.close()

    db = SearchDB(path)
    db.record_perf(_ctx().structural_key(), leaf.op_key, backend="cuda", status="ok", stats=PerfStats.point(90.0), captured=True)
    db.close()
    db = SearchDB.open_readonly(path)
    assert db.lookup_perf(_ctx().structural_key(), leaf.op_key, backend="cuda").stats.median == 10.0, "a slower re-measure loses"
    db.close()
