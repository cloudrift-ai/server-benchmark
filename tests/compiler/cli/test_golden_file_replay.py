"""Explicit working-golden replay for ``compile`` / ``run``."""

from types import SimpleNamespace

import pytest

from emmy.compiler.context import Context
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
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
