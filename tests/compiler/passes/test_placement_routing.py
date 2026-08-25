"""Placement-routing acceptance tests over complete Fold trees."""

from __future__ import annotations

from dataclasses import replace

import pytest

from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F16
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import MatmulOp, RmsNormOp
from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
from emmy.compiler.pipeline.fork import flatten_leaves
from emmy.compiler.pipeline.pipeline import Run
from emmy.compiler.pipeline.search.golden import GoldenRecord


def _compile(graph, knobs_env: str | None, monkeypatch, ctx: Context | None = None):
    if knobs_env is None:
        monkeypatch.delenv("EMMY_KNOBS", raising=False)
        monkeypatch.delenv("EMMY_PLACE", raising=False)
    else:
        monkeypatch.setenv("EMMY_KNOBS", knobs_env)
    ctx = ctx or Context.from_target((12, 0))
    out, _ = Run(pipeline=Pipeline.build(CUDA_PASSES), ctx=ctx).resolve(graph, lambda fp: flatten_leaves(fp.options)[0])
    return out


def _kernel_ids(out) -> list[str]:
    return sorted(node_id for node_id, node in out.nodes.items() if getattr(node.op, "kernel_source", None))


def _inp(graph: Graph, name: str, shape: tuple) -> None:
    graph.add_node(op=InputOp(), inputs=[], output=Tensor(name, tuple(Dim(size) for size in shape), dtype=F16), node_id=name)


def _rms_graph(rows: int = 64, width: int = 4096) -> Graph:
    graph = Graph()
    _inp(graph, "x", (rows, width))
    _inp(graph, "w", (width,))
    graph.add_node(RmsNormOp(), ["x", "w"], Tensor("y", (Dim(rows), Dim(width)), dtype=F16), node_id="y")
    graph.inputs, graph.outputs = ["x", "w"], ["y"]
    return graph


def _norm_linear_graph(rows: int = 32, width: int = 1024, intermediate: int = 3072) -> Graph:
    graph = Graph()
    _inp(graph, "x", (1, rows, width))
    _inp(graph, "wn", (width,))
    _inp(graph, "w", (width, intermediate))
    graph.add_node(RmsNormOp(), ["x", "wn"], Tensor("xn", (1, Dim(rows), Dim(width)), dtype=F16), node_id="xn")
    graph.add_node(
        MatmulOp(),
        ["xn", "w"],
        Tensor("y", (1, Dim(rows), Dim(intermediate)), dtype=F16),
        node_id="y",
    )
    graph.inputs, graph.outputs = ["x", "wn", "w"], ["y"]
    return graph


def test_is_routing_reads_place_only_knob_dicts() -> None:
    base = GoldenRecord(
        name="test",
        gpu_name="TESTGPU",
        compute_cap=(12, 0),
        model=None,
        program_index=0,
        program_wire={"inputs": [], "outputs": [], "nodes": []},
        origins=(),
        bindings=(),
        pins=(("FAST_MATH", False),),
        knobs={},
        measurements={"emmy_us": 1.0, "reference_us": 1.0, "reference_backend": "test"},
        ranking=None,
    )
    assert replace(base, knobs={"PLACE@a": "cut"}).is_routing
    assert not replace(base, knobs={"TILE": "f4x8", "WORK": "t16x8"}).is_routing
    assert not replace(base, knobs={}).is_routing


def test_rms_norm_deploys_unchanged_under_default_fuse(monkeypatch) -> None:
    assert len(_kernel_ids(_compile(_rms_graph(), None, monkeypatch))) == 1


def test_rms_norm_place_cut_splits_stat_and_scale(monkeypatch) -> None:
    out = _compile(_rms_graph(), "PLACE=cut", monkeypatch)
    kernels = _kernel_ids(out)
    assert any("__cut_" in kernel for kernel in kernels)
    assert len(kernels) >= 2


def test_norm_linear_cone_cut_recurses_to_the_full_cascade(monkeypatch) -> None:
    kernels = _kernel_ids(_compile(_norm_linear_graph(), "PLACE=cut", monkeypatch))
    assert len([kernel for kernel in kernels if "__cut_" in kernel]) >= 2
    assert "y" in kernels


def test_cut_preserves_every_multi_output_parent_port(monkeypatch) -> None:
    graph = _norm_linear_graph(rows=1, width=16, intermediate=16)
    graph.outputs = ["y", "xn"]
    out = _compile(graph, "PLACE=cut", monkeypatch)
    assert out.outputs == ["y", "xn"]
    assert all(out.buffer(buffer) is not None for buffer in out.outputs)
    assert any("__cut_" in kernel for kernel in _kernel_ids(out))


def test_scoped_place_pin_from_replay_context_cuts_the_cone(monkeypatch) -> None:
    from emmy.compiler.pipeline.search.pins import pinned_knobs

    monkeypatch.delenv("EMMY_KNOBS", raising=False)
    with pinned_knobs({"PLACE@a": "cut"}):
        out, _ = Run(pipeline=Pipeline.build(CUDA_PASSES), ctx=Context.from_target((12, 0))).resolve(
            _norm_linear_graph(rows=1, width=16, intermediate=16),
            lambda fp: flatten_leaves(fp.options)[0],
        )
    assert any("__cut_" in kernel for kernel in _kernel_ids(out))


def test_explicit_fuse_pin_suppresses_cutting(monkeypatch) -> None:
    assert len(_kernel_ids(_compile(_rms_graph(), "PLACE=fuse", monkeypatch))) == 1


def test_pin_naming_no_seam_is_skipped(monkeypatch) -> None:
    assert len(_kernel_ids(_compile(_rms_graph(), "PLACE@b=cut", monkeypatch))) == 1


def test_a_cut_taken_at_a_fork_mid_batch_still_reaches_the_stamp(monkeypatch) -> None:
    monkeypatch.delenv("EMMY_KNOBS", raising=False)
    monkeypatch.delenv("EMMY_PLACE", raising=False)
    graph = Graph()
    for tag in ("a", "b"):
        _inp(graph, f"x{tag}", (1, 2, 16))
        _inp(graph, f"wn{tag}", (16,))
        _inp(graph, f"w{tag}", (16, 16))
        graph.add_node(
            RmsNormOp(),
            [f"x{tag}", f"wn{tag}"],
            Tensor(f"xn{tag}", (Dim(1), Dim(2), Dim(16)), dtype=F16),
            node_id=f"xn{tag}",
        )
        graph.add_node(
            MatmulOp(),
            [f"xn{tag}", f"w{tag}"],
            Tensor(f"y{tag}", (Dim(1), Dim(2), Dim(16)), dtype=F16),
            node_id=f"y{tag}",
        )
    graph.inputs = [f"{name}{tag}" for tag in ("a", "b") for name in ("x", "wn", "w")]
    graph.outputs = ["ya", "yb"]

    taken: list[str] = []

    def cut_once(fork):
        leaves = flatten_leaves(fork.options)
        cut = next((option for option in leaves if isinstance(option, Graph)), None) if not taken else None
        if cut is None:
            return leaves[0]
        taken.append("cut")
        return cut

    out, _ = Run(pipeline=Pipeline.build(CUDA_PASSES), ctx=Context.from_target((12, 0))).resolve(graph, cut_once)
    kernels = _kernel_ids(out)
    assert taken
    assert any("__cut_" in kernel for kernel in kernels)
    assert {"ya", "yb"} <= set(kernels)


def test_pinned_transposed_coop_band_refuses_without_a_free_axis_to_sweep(monkeypatch) -> None:
    monkeypatch.setenv("EMMY_WORK", "t256")
    monkeypatch.setenv("EMMY_REDUCE", "coop-t")
    with pytest.raises(ValueError, match="innermost free axis"):
        _compile(_rms_graph(rows=1), None, monkeypatch)
