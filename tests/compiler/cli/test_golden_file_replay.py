"""Explicit working-golden replay for ``compile`` / ``run``."""

import asyncio
import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F16
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.frontend.ir import MatmulOp, RmsNormOp
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.tensor.ir import ElementwiseOp
from emmy.compiler.loop_wire import loop_graph_from_wire, loop_graph_to_wire
from emmy.compiler.pipeline.search.golden import dump_golden_file, load_golden_file
from emmy.compiler.pipeline.search.working_golden import write_trace_inventory


def _working_loop(path, *, state="inventory", pins=None):
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (16,)), node_id="x")
    graph.add_node(ElementwiseOp("relu"), ["x"], Tensor("y", (16,)), node_id="y")
    graph.inputs, graph.outputs = ["x"], ["y"]
    write_trace_inventory(
        graph,
        path,
        ctx=Context.from_target((8, 9)),
        force_loop_targets=True,
    )
    document = load_golden_file(path)
    entry = document["configs"][0]
    realization = entry["realizations"][0]
    realization["name"] = "working.relu"
    if pins is not None:
        realization["pins"] = pins
    if state in {"proposal", "tuned", "verified"}:
        realization["knobs"] = {"WORK": "w1x1"}
    if state == "tuned":
        realization["ranking"] = {
            "source": "tune",
            "status": "ok",
            "tune_winner": True,
            "measured_knobs": {"WORK": "w1x1"},
        }
    if state == "verified":
        realization["measurements"] = {
            "emmy_us": 1.0,
            "reference_us": 2.0,
            "reference_backend": "torch",
        }
    loop = loop_graph_from_wire(document["loops"][entry["target"]["loop"]])
    loop.nodes["y"].op = replace(loop.nodes["y"].op, name="working_exact_loop")
    document["loops"][entry["target"]["loop"]] = loop_graph_to_wire(loop)
    dump_golden_file(document, path, overwrite=True)
    return document


def _working_placement_route(path):
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (Dim(1), Dim(2), Dim(16)), dtype=F16), node_id="x")
    graph.add_node(InputOp(), [], Tensor("wn", (Dim(16),), dtype=F16), node_id="wn")
    graph.add_node(InputOp(), [], Tensor("w", (Dim(16), Dim(16)), dtype=F16), node_id="w")
    graph.add_node(
        RmsNormOp(),
        ["x", "wn"],
        Tensor("xn", (Dim(1), Dim(2), Dim(16)), dtype=F16),
        node_id="xn",
    )
    graph.add_node(
        MatmulOp(),
        ["xn", "w"],
        Tensor("y", (Dim(1), Dim(2), Dim(16)), dtype=F16),
        node_id="y",
    )
    graph.inputs, graph.outputs = ["x", "wn", "w"], ["y"]
    write_trace_inventory(graph, path, ctx=Context.from_target((8, 9)), force_loop_targets=True)
    document = load_golden_file(path)
    realization = document["configs"][0]["realizations"][0]
    realization["name"] = "working.route"
    realization["pins"] = {"FAST_MATH": False}
    realization["knobs"] = {"PLACE@inner.1/map": "cut"}  # a routing row: the measured price of that kernel set
    realization["measurements"] = {"emmy_us": 1.0, "reference_us": 2.0, "reference_backend": "torch"}
    dump_golden_file(document, path, overwrite=True)
    return document


def _args(path, **overrides):
    values = {
        "realization": "working.relu",
        "golden": str(path),
        "code": None,
        "input": None,
        "ir": "cuda",
        "dynamic": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_compile_working_file_uses_exact_loop_target(run_cli, tmp_path):
    path = tmp_path / "working.yaml"
    _working_loop(path)

    rc, stdout, stderr = run_cli(
        "compile",
        "--golden",
        str(path),
        "--realization",
        "working.relu",
        "--target",
        "sm_89",
        "--ir",
        "cuda",
    )

    assert rc == 0, stderr
    assert "working_exact_loop" in stdout


def test_working_file_requires_name_and_reports_its_own_available_rows(run_cli, tmp_path):
    path = tmp_path / "working.yaml"
    _working_loop(path)

    rc, stdout, stderr = run_cli("compile", "--golden", str(path), "--ir", "cuda")
    assert rc == 2
    assert "--golden PATH requires --realization NAME" in stdout + stderr

    rc, stdout, stderr = run_cli("compile", "--golden", str(path), "--realization", "missing", "--ir", "cuda")
    assert rc == 2
    assert "unknown golden config" in stdout + stderr
    assert "working.relu" in stdout + stderr


def test_frontend_target_features_follow_the_replay_slice_after_maximal_fusion() -> None:
    from emmy.compiler.pipeline.search.golden import load_golden_records
    from emmy.compiler.torch_wire import graph_to_wire

    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (16,)), node_id="x")
    graph.add_node(ElementwiseOp("relu"), ["x"], Tensor("hidden", (16,)), node_id="hidden")
    graph.add_node(ElementwiseOp("relu"), ["hidden"], Tensor("out", (16,)), node_id="out")
    graph.inputs, graph.outputs = ["x"], ["out"]
    (record,) = load_golden_records(
        {
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "compute_cap": [8, 9],
            "model": "org/model",
            "programs": [graph_to_wire(graph)],
            "configs": [
                {
                    "program": 0,
                    "target": {"origins": ["hidden"]},
                    "realizations": [{"name": "working.hidden", "bindings": {}, "pins": {"FAST_MATH": False}}],
                }
            ],
        }
    )

    assert record.structural_features


def test_working_file_golden_conflicts_with_direct_input(run_cli, tmp_path):
    path = tmp_path / "working.yaml"
    _working_loop(path)

    rc, stdout, stderr = run_cli(
        "compile",
        "--golden",
        str(path),
        "--realization",
        "working.relu",
        "--code",
        "torch.zeros(4)",
    )
    assert rc == 2
    assert "mutually exclusive" in stdout + stderr


def test_duplicate_name_requires_target_scoped_working_file(tmp_path, caplog):
    """A repeated shape name must not silently choose between distinct embedded targets."""
    from emmy.commands.compile import resolve_golden_arg

    path = tmp_path / "working.yaml"
    document = _working_loop(path)
    second = copy.deepcopy(document["configs"][0])
    loop = loop_graph_from_wire(document["loops"][second["target"]["loop"]])
    loop.nodes["y"].op = replace(loop.nodes["y"].op, name="working_second_loop")
    second["target"]["loop"] = len(document["loops"])
    document["loops"].append(loop_graph_to_wire(loop))
    document["configs"].append(second)
    dump_golden_file(document, path, overwrite=True)

    with pytest.raises(SystemExit) as exc:
        resolve_golden_arg(_args(path))
    assert exc.value.code == 2
    assert "resolves to 2 different embedded program targets" in caplog.text

    scoped = copy.deepcopy(document)
    scoped["configs"] = [scoped["configs"][1]]
    scoped_path = tmp_path / "working-scoped.yaml"
    dump_golden_file(scoped, scoped_path, overwrite=True)
    args = _args(scoped_path)
    resolve_golden_arg(args)
    assert args._golden_graph.nodes["y"].op.name == "working_second_loop"


def test_named_proposal_is_pinned_and_a_file_walk_leaves_it_to_the_tuner(tmp_path):
    """Naming a realization asks for that row: it benches as a pinned row whatever its measurement
    state (the corpus and perf lanes replay unmeasured cases this way). A bare ``--golden PATH``
    walk names nothing, so a proposal there stays the tuner's and only verified rows bench."""
    from emmy.commands.compile import resolve_golden_arg
    from emmy.commands.run import _pinned_samples_for_ir

    path = tmp_path / "working.yaml"
    _working_loop(path, state="proposal")
    args = _args(path)

    resolve_golden_arg(args)

    assert isinstance(args._golden_graph.nodes["y"].op, LoopOp)
    assert args._golden_graph.nodes["y"].op.name == "working_exact_loop"
    (row,) = args.golden_configs
    assert row.name == "working.relu" and row.knobs == {"WORK": "w1x1"} and row.record.name == "working.relu"
    assert [record.name for record in args._golden_records] == ["working.relu"]
    args.ab = ["WORK=w2x2"]
    pinned, manual = _pinned_samples_for_ir(args, args._golden_graph)
    assert pinned is row
    assert manual.name == "ab WORK=w2x2" and manual.knobs == {"WORK": "w2x2"} and not hasattr(manual, "record")

    walked = _args(path, _explicit_realization=False)
    resolve_golden_arg(walked)
    assert walked.golden_configs == []


@pytest.mark.parametrize("explicit", [False, True], ids=["ordinary", "explicit"])
def test_recorded_route_cuts_the_selected_compile_target(run_cli, tmp_path, monkeypatch, explicit):
    """A measured routing row is the measured price of the kernel set its route spells. Selecting
    the file makes it evidence, so at the placement fork it outranks the fused arm nothing measured
    and the pass's own cut arm is taken: the compile splits into the placed producer plus its
    consumers. A hand pin of the same route through ``EMMY_KNOBS`` lands identically."""
    path = tmp_path / "working-route.yaml"
    _working_placement_route(path)
    monkeypatch.delenv("EMMY_KNOBS", raising=False)
    monkeypatch.setenv("EMMY_NVCC_FLAGS", "")
    monkeypatch.setenv("EMMY_READABLE", "1")
    monkeypatch.setenv("EMMY_TUNE_DB", str(tmp_path / "tune.db"))
    if explicit:
        monkeypatch.setenv("EMMY_KNOBS", "PLACE@inner.1/map=cut")

    rc, stdout, stderr = run_cli(
        "compile",
        "--golden",
        str(path),
        "--realization",
        "working.route",
        "--target",
        "sm_89",
        "--ir",
        "tile",
    )

    # Kernel names carry volatile identity digests, so assert the kernel set's shape, not the
    # spelled names.
    assert rc == 0, stderr
    headers = [line for line in stdout.splitlines() if line.startswith("=== ")]
    assert len(headers) == 3, stdout
    assert sum("__place_" in line for line in headers) == 1, stdout


def test_selected_records_scope_the_tier_and_a_split_regime_publishes_nothing(monkeypatch, tmp_path):
    """The selected realization's records are the compile's whole golden scope; the input regime
    (the precision pins) reaches the environment only when every record agrees on it."""
    from emmy.commands.compile import resolve_golden_arg
    from emmy.compiler.pipeline.search import golden
    from emmy.compiler.pipeline.search.golden import shared_regime_pins

    path = tmp_path / "working.yaml"
    document = _working_loop(path, pins={"FAST_MATH": False, "PLACE@inner.1/map": "cut"})
    second = copy.deepcopy(document["configs"][0]["realizations"][0])
    second["pins"]["FAST_MATH"] = True
    document["configs"][0]["realizations"].append(second)
    dump_golden_file(document, path, overwrite=True)
    args = _args(path)

    resolve_golden_arg(args)

    assert [record.route for record in args._golden_records] == [{"PLACE@inner.1/map": "cut"}] * 2
    assert [dict(record.pins) for record in args._golden_records] == [
        {"FAST_MATH": False, "PLACE@inner.1/map": "cut"},
        {"FAST_MATH": True, "PLACE@inner.1/map": "cut"},
    ]
    assert shared_regime_pins(args._golden_records) == {}
    assert shared_regime_pins(args._golden_records[:1]) == {"FAST_MATH": False}

    # Without --golden PATH the live card's repository corpus is searched, and its matches scope the tier the same way.
    records = golden.load_golden_records(document)
    monkeypatch.setattr(golden, "goldens_for_live_gpu", lambda: records)
    monkeypatch.setattr(golden, "GOLDEN_RECORDS", records)
    canonical = _args(path, golden=None)
    resolve_golden_arg(canonical)
    assert [record.name for record in canonical._golden_records] == ["working.relu", "working.relu"]


def test_working_verified_row_is_automatically_pinned(tmp_path):
    from emmy.commands.compile import resolve_golden_arg
    from emmy.commands.run import _sample_replay_knobs

    path = tmp_path / "working.yaml"
    _working_loop(path, state="verified")
    args = _args(path)

    resolve_golden_arg(args)

    assert len(args.golden_configs) == 1
    assert args.golden_configs[0].knobs == {"WORK": "w1x1"}
    assert args.golden_configs[0].pins == {"FAST_MATH": False}
    assert _sample_replay_knobs(args.golden_configs[0]) == {"FAST_MATH": False, "WORK": "w1x1"}


def test_working_direct_tune_winner_is_automatically_pinned(tmp_path):
    from emmy.commands.compile import resolve_golden_arg

    path = tmp_path / "working.yaml"
    _working_loop(path, state="tuned")
    args = _args(path)

    resolve_golden_arg(args)

    assert len(args.golden_configs) == 1
    assert args.golden_configs[0].knobs == {"WORK": "w1x1"}


def test_working_invalid_direct_tune_winner_is_rejected(tmp_path):
    from emmy.commands.compile import resolve_golden_arg

    path = tmp_path / "working.yaml"
    document = _working_loop(path, state="tuned")
    document["configs"][0]["realizations"][0]["ranking"]["measured_knobs"] = {"WORK": "w2x2"}
    dump_golden_file(document, path, overwrite=True)

    with pytest.raises(SystemExit, match="2"):
        resolve_golden_arg(_args(path))


def test_run_replays_embedded_loop_golden_through_structural_stamps(tmp_path):
    from emmy.commands.compile import resolve_golden_arg
    from emmy.commands.run import _passes_after_stage, _replay_stage_and_passes
    from emmy.compiler.pipeline import CUDA_PASSES

    path = tmp_path / "working.yaml"
    _working_loop(path)
    args = _args(path)
    resolve_golden_arg(args)

    stage, passes = _replay_stage_and_passes(args._golden_graph, embedded_golden=True)
    assert stage == "golden Loop"
    assert passes == CUDA_PASSES

    stage, passes = _replay_stage_and_passes(args._golden_graph, embedded_golden=False)
    assert stage == "loop"
    assert passes == _passes_after_stage("loop")
    assert passes != CUDA_PASSES


def test_emmy_only_benchmark_returns_same_input_reference():
    """Embedded Loop replay can return its greedy inputs/outputs without a Torch twin."""
    import numpy as np

    from emmy.commands.run import bench_lowered_vs_torch

    graph = Graph()
    graph.add_node(ConstantOp(name="y", value=2.0), [], Tensor("y", (1,)), node_id="y")
    graph.outputs = ["y"]
    outputs = {"y": np.array([2.0], dtype=np.float32)}

    class FakeBackend:
        def run(self, _graph, *, input_data):
            return SimpleNamespace(outputs=outputs), None

        async def benchmark_async(self, *_args, **_kwargs):
            return SimpleNamespace(time_ms=0.001, captured=True)

    refs = []
    asyncio.run(
        bench_lowered_vs_torch(
            None,
            graph,
            FakeBackend(),
            seed=0,
            do_bench=True,
            warmup=1,
            iters=1,
            bench_backends="emmy",
            ref_out=refs,
        )
    )
    assert len(refs) == 1
    assert refs[0][0] == {"y": [2.0]}
    assert refs[0][1] is outputs


def test_emmy_only_benchmark_does_not_duplicate_inputs_on_torch(monkeypatch):
    """A reference-free Loop target owns one device input allocation, not a redundant Torch copy."""
    import numpy as np

    from emmy.commands import run as run_module

    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (8,), "f16"), node_id="x")
    graph.outputs = ["x"]

    class FakeBackend:
        def run(self, _graph, *, input_data):
            assert input_data["x"].shape == (8,)
            return SimpleNamespace(outputs={"x": np.ones(8, dtype=np.float16)}, time_ms=0.001), None

        async def benchmark_async(self, *_args, **_kwargs):
            return SimpleNamespace(time_ms=0.001, captured=True)

    monkeypatch.setattr(
        run_module,
        "_to_cuda_tensor",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("reference-free replay must not make a Torch copy")),
    )
    asyncio.run(
        run_module.bench_lowered_vs_torch(
            None,
            graph,
            FakeBackend(),
            seed=0,
            do_bench=True,
            warmup=1,
            iters=1,
            bench_backends="emmy",
        )
    )


def test_embedded_loop_pins_receive_greedy_output_reference(monkeypatch, tmp_path, caplog):
    """Exact Loop targets have no Torch twin, so pinned replay must compare against the greedy Loop execution."""
    from emmy.commands import run as run_module
    from emmy.commands.compile import resolve_golden_arg
    from emmy.compiler.pipeline import Pipeline

    path = tmp_path / "working.yaml"
    _working_loop(path, state="verified")
    args = _args(
        path,
        ir=None,
        bench=True,
        ab=None,
        debug=False,
        dump_dir=None,
        bench_backends="emmy",
        warmup=5,
        iters=20,
        seed=0,
        json=None,
        profile=False,
    )
    resolve_golden_arg(args)

    reference = ({"x": object()}, {"y": object()})
    returned = {"reference": reference, "greedy_error": None, "reference_run_us": None, "accuracy_error": None}
    seen = {}

    class FakePipeline:
        def run(self, graph, **_kwargs):
            return graph

    class FakeBackend:
        name = "cuda"
        tune_db = None
        bench_compile_timeout_s = 1.0
        bench_run_timeout_s = 1.0

        def __init__(self, **_kwargs):
            pass

        async def benchmark_compare_async(self, _graph, **kwargs):
            seen["want_ref"] = kwargs["want_ref"]
            return {
                "results": {},
                "result": None,
                "captured": False,
                "torch_available": False,
                "accuracy_error": returned["accuracy_error"],
                "run_io": returned["reference"],
                "greedy_error": returned["greedy_error"],
                "reference_run_us": returned["reference_run_us"],
            }

        async def aclose_async_worker(self):
            pass

    class FakeDump:
        @staticmethod
        def resolve(_path):
            return None

    async def fake_isolated(*_args, **_kwargs):
        return None

    async def fake_pinned(_backend, _source, _pins, **kwargs):
        seen["ref"] = kwargs["ref"]
        if kwargs["strict_correctness"]:
            seen["strict_reference"] = kwargs["strict_reference"]
        return []

    monkeypatch.setattr(Pipeline, "build", lambda _passes: FakePipeline())
    monkeypatch.setattr(run_module, "_bench_greedy_isolated", fake_isolated)
    monkeypatch.setattr(run_module, "_bench_golden_variants", fake_pinned)
    monkeypatch.setattr(run_module, "_print_kernel_stats", lambda *_args, **_kwargs: None)

    run_module._handle_run_ir(args, FakeBackend, FakeDump)

    assert seen == {"want_ref": True, "ref": reference}

    args.strict_correctness = True
    returned["accuracy_error"] = "strict eager correctness unavailable: frontend IR is not runnable"
    returned["reference"] = ({"x": [1.0]}, {"y": [1.0]})
    seen.clear()
    with pytest.raises(SystemExit) as exc:
        run_module._handle_run_ir(args, FakeBackend, FakeDump)
    assert exc.value.code == 1
    assert seen == {
        "want_ref": True,
        "ref": returned["reference"],
        "strict_reference": "same-input-greedy",
    }

    args.strict_correctness = False
    returned["accuracy_error"] = None
    returned["reference"] = reference

    async def fail_if_isolated(*_args, **_kwargs):
        raise AssertionError("a failed greedy timing must not be re-benched or made eligible")

    seen.clear()
    returned["greedy_error"] = "HungKernelError: repeated timing crossed the watchdog"
    returned["reference_run_us"] = 4_000_000.0
    monkeypatch.setattr(run_module, "_bench_greedy_isolated", fail_if_isolated)
    with pytest.raises(SystemExit) as exc:
        run_module._handle_run_ir(args, FakeBackend, FakeDump)
    assert exc.value.code == 1
    assert seen == {"want_ref": True, "ref": reference}
    assert "untimed greedy is ineligible; pinned rows still bench" in caplog.text

    seen.clear()
    returned["greedy_error"] = None
    returned["reference_run_us"] = None
    returned["reference"] = None
    monkeypatch.setattr(run_module, "_bench_greedy_isolated", fake_isolated)
    with pytest.raises(SystemExit) as exc:
        run_module._handle_run_ir(args, FakeBackend, FakeDump)
    assert exc.value.code == 1
    assert seen == {"want_ref": True}
    assert "requires same-input greedy outputs" in caplog.text


def test_replay_keys_its_cache_by_the_entry_identity(tmp_path):
    """Two entries of one set can spell the same row and pins on different kernels — a seam
    spelling recurs on a residual as earlier cuts renumber its tree — and each replays its own
    fork: the entry naming the kernel a fork is offered on reports the arm it spelled there, and a
    same-spelled sibling naming another kernel reports none."""
    from emmy.compiler.pipeline.search.golden import _replay, golden_record_from_entry, kernel_identity

    path = tmp_path / "working-route.yaml"
    document = _working_placement_route(path)
    entry = document["configs"][0]
    routing = golden_record_from_entry(document, entry, entry["realizations"][0])
    owner = replace(routing, identity=kernel_identity(routing))
    other = replace(owner, name="working.other", identity="f" * 64)

    assert len(_replay(owner, siblings=(other,), lead=owner).arms) == 1
    assert _replay(other, siblings=(owner,), lead=owner).arms == ()


def test_recorded_greedy_pick_is_picked_again_under_strict_evidence(tmp_path):
    """The kernel set a compile picked, recorded as measured rows — one routing row per kernel-set
    decision it took and one child-identity schedule receipt per kernel — is evidence enough: those
    rows alone yield the same kernels with the same rows under strict evidence, with no prior and
    no tune DB. A receipt carries the input regime and no route: seam spellings are
    kernel-local, so a cut key copied onto every receipt would re-cut any piece that offers a
    same-spelled seam."""
    from emmy import config
    from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
    from emmy.compiler.pipeline.search.golden import (
        GoldenEntryState,
        golden_entry_state,
        golden_record_from_entry,
        records_override,
        sole_evidence,
    )
    from emmy.compiler.pipeline.search.pins import pinned_knobs
    from emmy.compiler.pipeline.search.working_golden import KernelSetDecisions, greedy_pick_rows, record_greedy_pick

    path = tmp_path / "working-route.yaml"
    document = _working_placement_route(path)
    entry = document["configs"][0]
    seed = golden_record_from_entry(document, entry, entry["realizations"][0])
    ctx = Context.from_target((8, 9))
    taken = KernelSetDecisions()
    # The pick to record: the routing row decides the cut, the prior decides the pieces' schedules.
    with records_override([seed]), pinned_knobs({"FAST_MATH": False}):
        picked = Pipeline.build(CUDA_PASSES).with_strategies(taken).run(seed.target_program.copy(), ctx=ctx, db=None)
    rows = greedy_pick_rows(picked)
    # The routing row's cut first; the pieces may take further kernel-set decisions of their own
    # (a cross-CTA split of the residual), each recorded as a routing row of its own kernel.
    assert len(rows) >= 2 and taken.decisions[0][1] == {"PLACE@inner.1/map": "cut"}

    written = record_greedy_pick(
        path,
        document,
        "working.route",
        decisions=[(identity, knobs, 5.0, 6.0) for identity, knobs in taken.decisions],
        kernels=[(identity, row, 1.0, 2.0) for identity, row in rows],
        reference_backend="same-input-greedy",
    )

    reloaded = load_golden_file(path)
    added = [row for row in reloaded["configs"][0]["realizations"] if row["name"] in written]
    assert len(added) == len(rows) + len(taken.decisions)
    assert all(golden_entry_state(row) is GoldenEntryState.VERIFIED and row["identity"] for row in added)
    assert all(row["pins"] == {"FAST_MATH": False} for row in added)
    records = [golden_record_from_entry(reloaded, reloaded["configs"][0], row) for row in added]
    with sole_evidence(records), pinned_knobs({"FAST_MATH": False}), config.strict_evidence_override(True):
        again = Pipeline.build(CUDA_PASSES).run(seed.target_program.copy(), ctx=ctx, db=None)
    assert greedy_pick_rows(again) == rows


def test_run_records_the_greedy_pick_of_an_embedded_golden(monkeypatch, tmp_path):
    """``run --golden PATH --realization NAME --bench --record-greedy``: the greedy row compiles with
    the file's rows as its golden evidence (here the routing row, so the cut is taken), and after
    the bench the kernel set it picked is written back as measured rows — a routing row per
    kernel-set decision with the isolated whole-graph timing, a receipt per kernel with its
    isolated launch timing, the greedy comparison row as every reference — while the per-kernel
    perf rows and node leaves every embedded-golden bench records by default are recorded too."""
    from emmy.commands import run as run_module
    from emmy.commands.compile import resolve_golden_arg
    from emmy.compiler import target as target_mod
    from emmy.compiler.pipeline.search.policy.greedy import _is_route_row

    path = tmp_path / "working-route.yaml"
    _working_placement_route(path)
    args = _args(
        path,
        realization="working.route",
        ir=None,
        bench=True,
        ab=None,
        debug=False,
        dump_dir=None,
        bench_backends="emmy",
        warmup=5,
        iters=20,
        seed=0,
        json=None,
        profile=False,
        record=False,
        record_greedy=True,
        strict_correctness=False,
    )
    resolve_golden_arg(args)

    def launches(graph, ms_per_launch):
        n = len(run_module._launch_order_cuda_nodes(graph))
        per_launch = [SimpleNamespace(idx=i, time_ms=ms_per_launch * (i + 1), samples=[ms_per_launch * (i + 1)]) for i in range(n)]
        total = sum(launch.time_ms for launch in per_launch)
        return SimpleNamespace(min_ms=total, time_ms=total, e2e_min_ms=None, captured=True, num_launches=n, per_launch=per_launch)

    class FakeBackend:
        name = "cuda"
        tune_db = None
        bench_compile_timeout_s = 1.0
        bench_run_timeout_s = 1.0

        def __init__(self, **_kwargs):
            pass

        async def benchmark_compare_async(self, graph, **_kwargs):
            return {
                "results": {"Emmy": 1.0},
                "result": launches(graph, 0.002),
                "captured": True,
                "torch_available": False,
                "accuracy_error": None,
                "run_io": ({"x": object()}, {"y": object()}),
                "greedy_error": None,
                "reference_run_us": None,
            }

        async def aclose_async_worker(self):
            pass

    class FakeDump:
        @staticmethod
        def resolve(_path):
            return None

    async def fake_isolated(_backend, compiled, *, warmup, iters):
        sample = SimpleNamespace(name="greedy (isolated)", knobs={}, shape=None, dynamic=None)
        return run_module._GoldenBench(sample, compiled, launches(compiled, 0.001), [], "ok")

    async def fake_pinned(_backend, _source, _pins, **_kwargs):
        return []

    recorded = {}
    monkeypatch.setattr(run_module, "_bench_greedy_isolated", fake_isolated)
    monkeypatch.setattr(run_module, "_bench_golden_variants", fake_pinned)
    monkeypatch.setattr(run_module, "_print_kernel_stats", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_module, "_record_bench_evidence", lambda _args, benches, iso: recorded.update(benches=benches, iso=iso))
    target_mod.set_target((8, 9))
    try:
        run_module._handle_run_ir(args, FakeBackend, FakeDump)
    finally:
        target_mod.set_target(None)

    assert recorded["benches"] == [] and recorded["iso"].status == "ok"
    added = load_golden_file(path)["configs"][0]["realizations"][1:]
    routing = [row for row in added if _is_route_row(row["knobs"])]
    receipts = [row for row in added if not _is_route_row(row["knobs"])]
    assert routing[0]["knobs"] == {"PLACE@inner.1/map": "cut"} and len(receipts) >= 2
    total = sum(range(1, len(receipts) + 1))
    assert [row["measurements"] for row in routing] == [
        {"emmy_us": pytest.approx(total * 1.0), "reference_us": pytest.approx(total * 2.0), "reference_backend": "same-input-greedy"}
    ] * len(routing)
    assert [row["measurements"] for row in receipts] == [
        {"emmy_us": pytest.approx((i + 1) * 1.0), "reference_us": pytest.approx((i + 1) * 2.0), "reference_backend": "same-input-greedy"}
        for i in range(len(receipts))
    ]


def test_run_files_a_hung_greedy_kernel_as_bench_fail_evidence(monkeypatch, tmp_path):
    """A greedy pick that hangs the watchdog is evidence too: the kernel the watchdog NAMED earns
    a ``bench_fail`` perf row in the tune DB and no other kernel does, the same narrowing the
    tuner's terminal bench applies — so the next compile's evidence pick disqualifies that arm
    instead of electing the identical route and hanging again. Before, a hung greedy on an
    embedded golden recorded nothing (the run exited on the missing same-input reference before
    any recording ran), so ``run --bench`` could never advance an election on its own."""
    from emmy.commands import run as run_module
    from emmy.commands.compile import resolve_golden_arg
    from emmy.compiler import target as target_mod
    from emmy.compiler.backend.cuda.program import BenchWorkerJobError
    from emmy.compiler.pipeline.search.db import SearchDB

    db_path = tmp_path / "autotune.db"
    monkeypatch.setenv("EMMY_TUNE_DB", str(db_path))
    monkeypatch.setenv("EMMY_NVCC_FLAGS", "")  # the deployable regime: perf evidence is recorded only there
    path = tmp_path / "working-route.yaml"
    _working_placement_route(path)
    args = _args(
        path,
        realization="working.route",
        ir=None,
        bench=True,
        ab=None,
        debug=False,
        dump_dir=None,
        bench_backends="emmy",
        warmup=5,
        iters=20,
        seed=0,
        json=None,
        profile=False,
        record=False,
        record_greedy=True,
        strict_correctness=False,
    )
    resolve_golden_arg(args)
    seen = {}

    class FakeBackend:
        name = "cuda"
        tune_db = None
        bench_compile_timeout_s = 1.0
        bench_run_timeout_s = 2.0

        def __init__(self, **_kwargs):
            pass

        async def benchmark_compare_async(self, graph, **_kwargs):
            seen["nodes"] = run_module._launch_order_cuda_nodes(graph)
            culprit = seen["nodes"][-1].op.kernel_name
            hang = f"kernel '{culprit} (iter 0)' did not complete within 60000 ms — variant marked bench_fail"
            raise BenchWorkerJobError(f'bench worker error: HungKernelError("{hang}")')

        async def aclose_async_worker(self):
            pass

    class FakeDump:
        @staticmethod
        def resolve(_path):
            return None

    monkeypatch.setattr(run_module, "_print_kernel_stats", lambda *_args, **_kwargs: None)
    target_mod.set_target((8, 9))
    try:
        with pytest.raises(SystemExit):
            run_module._handle_run_ir(args, FakeBackend, FakeDump)
        context_key = Context.probe().structural_key()
    finally:
        target_mod.set_target(None)

    nodes = seen["nodes"]
    assert len(nodes) >= 2, "the route must hold an innocent kernel beside the culprit"
    db = SearchDB(db_path)
    try:
        keys = {n.op.kernel_name: n.op.identity_key(with_io=True, with_knobs=True) for n in nodes}
        rows = {name: db.lookup_perf(context_key, key, backend="cuda") for name, key in keys.items()}
    finally:
        db.close()
    filed = {name: row.status for name, row in rows.items() if row is not None}
    assert filed == {nodes[-1].op.kernel_name: "bench_fail"}, "only the kernel the watchdog named is evidence"
    assert rows[nodes[-1].op.kernel_name].stats.median == pytest.approx(2.0e6), "priced at the run budget's fail sentinel"
