"""Explicit working-golden replay for ``compile`` / ``run``."""

import asyncio
import copy
from types import SimpleNamespace

import pytest

from emmy.compiler.context import Context
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.tensor.ir import ElementwiseOp
from emmy.compiler.loop_wire import loop_graph_from_wire, loop_graph_to_wire
from emmy.compiler.pipeline.search.golden import dump_golden_file, load_golden_file
from emmy.compiler.pipeline.search.working_golden import write_trace_inventory


def _working_loop(path, *, state="inventory"):
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
    loop.nodes["y"].op.name = "working_exact_loop"
    document["loops"][entry["target"]["loop"]] = loop_graph_to_wire(loop)
    dump_golden_file(document, path, overwrite=True)
    return document


def _args(path, **overrides):
    values = {
        "golden": "working.relu",
        "golden_file": str(path),
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
        "--golden-file",
        str(path),
        "--golden",
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

    rc, stdout, stderr = run_cli("compile", "--golden-file", str(path), "--ir", "cuda")
    assert rc == 2
    assert "--golden-file requires --golden NAME" in stdout + stderr

    rc, stdout, stderr = run_cli("compile", "--golden-file", str(path), "--golden", "missing", "--ir", "cuda")
    assert rc == 2
    assert "unknown golden config" in stdout + stderr
    assert "working.relu" in stdout + stderr


def test_working_file_golden_conflicts_with_direct_input(run_cli, tmp_path):
    path = tmp_path / "working.yaml"
    _working_loop(path)

    rc, stdout, stderr = run_cli(
        "compile",
        "--golden-file",
        str(path),
        "--golden",
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
    loop.nodes["y"].op.name = "working_second_loop"
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


def test_working_proposal_supplies_graph_but_is_not_automatically_pinned(tmp_path):
    from emmy.commands.compile import resolve_golden_arg
    from emmy.commands.run import _pinned_samples_for_ir

    path = tmp_path / "working.yaml"
    _working_loop(path, state="proposal")
    args = _args(path)

    resolve_golden_arg(args)

    assert isinstance(args._golden_graph.nodes["y"].op, LoopOp)
    assert args._golden_graph.nodes["y"].op.name == "working_exact_loop"
    assert args.golden_configs == []
    args.ab = ["WORK=w2x2"]
    (manual,) = _pinned_samples_for_ir(args, args._golden_graph)
    assert manual.name == "ab WORK=w2x2"
    assert manual.knobs == {"WORK": "w2x2"}


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
    assert args.expected_golden_pins == 1


def test_working_kernel_set_winner_is_automatically_pinned(tmp_path):
    from emmy.commands.compile import resolve_golden_arg

    path = tmp_path / "working.yaml"
    document = _working_loop(path)
    realization = document["configs"][0]["realizations"][0]
    realization["ranking"] = {
        "source": "tune",
        "status": "ok",
        "tune_winner": True,
        "compile_flags": "-O1",
        "latency_us": 7.0,
        "kernel_set": {
            "placement": {},
            "kernels": [
                {
                    "op_key": "working-op",
                    "multiplicity": 1,
                    "pins": {"WORK": "w1x1"},
                    "latency_us": 7.0,
                    "cuda_record_knobs": [{"WORK": "w1x1"}],
                }
            ],
        },
    }
    dump_golden_file(document, path, overwrite=True)
    args = _args(path)

    resolve_golden_arg(args)

    assert len(args.golden_configs) == 1
    sample = args.golden_configs[0]
    assert sample.knobs == {}
    assert sample.latency_us == 7.0
    assert sample.kernel_set == realization["ranking"]["kernel_set"]
    assert args.expected_golden_pins == 1


def test_working_kernel_set_winner_rejects_extra_component_field(tmp_path):
    from emmy.commands.compile import resolve_golden_arg

    path = tmp_path / "working.yaml"
    document = _working_loop(path)
    document["configs"][0]["realizations"][0]["ranking"] = {
        "source": "tune",
        "status": "ok",
        "tune_winner": True,
        "latency_us": 7.0,
        "kernel_set": {
            "placement": {},
            "kernels": [
                {
                    "op_key": "working-op",
                    "multiplicity": 1,
                    "pins": {"WORK": "w1x1"},
                    "latency_us": 7.0,
                    "cuda_record_knobs": [{"WORK": "w1x1"}],
                    "node_id": "y",
                }
            ],
        },
    }
    dump_golden_file(document, path, overwrite=True)

    with pytest.raises(SystemExit, match="2"):
        resolve_golden_arg(_args(path))


def test_working_unreplayable_tune_result_requires_an_exact_row(tmp_path):
    from emmy.commands.compile import resolve_golden_arg

    path = tmp_path / "working.yaml"
    document = _working_loop(path)
    document["configs"][0]["realizations"][0]["ranking"] = {
        "source": "tune",
        "status": "no_exact_pin",
        "compile_flags": "-O1",
        "latency_us": None,
        "measured_knobs": None,
        "error": "tune produced no exact replayable winner",
    }
    dump_golden_file(document, path, overwrite=True)
    args = _args(path)

    resolve_golden_arg(args)

    assert args.golden_configs == []
    assert args.expected_golden_pins == 1


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
    returned = {"reference": reference, "greedy_error": None, "reference_run_us": None}
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
                "accuracy_error": None,
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
        return []

    monkeypatch.setattr(Pipeline, "build", lambda _passes: FakePipeline())
    monkeypatch.setattr(run_module, "_bench_greedy_isolated", fake_isolated)
    monkeypatch.setattr(run_module, "_bench_golden_variants", fake_pinned)
    monkeypatch.setattr(run_module, "_print_kernel_stats", lambda *_args, **_kwargs: None)

    run_module._handle_run_ir(args, FakeBackend, FakeDump)

    assert seen == {"want_ref": True, "ref": reference}

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
