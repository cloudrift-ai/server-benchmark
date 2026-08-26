"""Placement-routing acceptance tests over complete Fold trees."""

from __future__ import annotations

from dataclasses import replace

import pytest

from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F16
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.ir.frontend.ir import MatmulOp, RmsNormOp
from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
from emmy.compiler.pipeline.search.golden import GoldenRecord
from emmy.compiler.pipeline.search.pins import pinned_knobs

_CTX = Context.from_target((12, 0))
_SCALAR = {"WORK": "", "TILE": "", "STAGE": "", "REDUCE": "", "RASTER": ""}


def _compile(graph: Graph, placement: dict[str, str]) -> Graph:
    with pinned_knobs({**placement, **_SCALAR}):
        return Pipeline.build(CUDA_PASSES).run(graph, ctx=_CTX)


def _kernels(graph: Graph):
    return [node for node in graph.nodes.values() if isinstance(node.op, CudaOp)]


def _input(graph: Graph, name: str, shape: tuple[int, ...]) -> None:
    graph.add_node(InputOp(), [], Tensor(name, tuple(Dim(size) for size in shape), dtype=F16), node_id=name)


def _rms_graph(rows: int = 4, width: int = 32) -> Graph:
    graph = Graph()
    _input(graph, "x", (rows, width))
    _input(graph, "w", (width,))
    graph.add_node(RmsNormOp(), ["x", "w"], Tensor("y", (Dim(rows), Dim(width)), dtype=F16), node_id="y")
    graph.inputs, graph.outputs = ["x", "w"], ["y"]
    return graph


def _norm_linear_graph(*, keep_norm: bool = False) -> Graph:
    rows, width, intermediate = 2, 16, 16
    graph = Graph()
    _input(graph, "x", (1, rows, width))
    _input(graph, "wn", (width,))
    _input(graph, "w", (width, intermediate))
    graph.add_node(
        RmsNormOp(),
        ["x", "wn"],
        Tensor("xn", (Dim(1), Dim(rows), Dim(width)), dtype=F16),
        node_id="xn",
    )
    graph.add_node(
        MatmulOp(),
        ["xn", "w"],
        Tensor("y", (Dim(1), Dim(rows), Dim(intermediate)), dtype=F16),
        node_id="y",
    )
    graph.inputs = ["x", "wn", "w"]
    graph.outputs = ["y", "xn"] if keep_norm else ["y"]
    return graph


def test_place_only_golden_rows_are_routing_rows() -> None:
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


def test_rms_norm_fused_pin_lowers_one_kernel() -> None:
    assert len(_kernels(_compile(_rms_graph(), {"PLACE": "fuse"}))) == 1


def test_rms_norm_cut_pin_splits_statistic_and_scale() -> None:
    kernels = _kernels(_compile(_rms_graph(), {"PLACE": "cut"}))
    assert len(kernels) == 2
    assert any("__place_" in node.id for node in kernels)


@pytest.mark.parametrize("spelling", ("PLACE", "PLACE@map", "PLACE@a1"))
def test_norm_linear_each_closed_cone_pin_lowers(spelling: str) -> None:
    kernels = _kernels(_compile(_norm_linear_graph(), {spelling: "cut"}))
    assert len(kernels) == 2
    assert any("__place_" in node.id for node in kernels)
    assert any(node.id == "y" for node in kernels)


def test_scoped_cut_preserves_every_multi_output_parent_port() -> None:
    lowered = _compile(_norm_linear_graph(keep_norm=True), {"PLACE@a": "cut"})
    assert lowered.outputs == ["y", "xn"]
    assert all(lowered.buffer(name) is not None for name in lowered.outputs)
    assert len(_kernels(lowered)) == 2
    assert any("__place_" in node.id for node in _kernels(lowered))


def test_pinned_transposed_coop_band_still_refuses_without_a_free_axis() -> None:
    with pytest.raises(ValueError, match="innermost free axis"):
        with pinned_knobs({"PLACE": "fuse", "WORK": "t256", "REDUCE": "coop-t"}):
            Pipeline.build(CUDA_PASSES).run(_rms_graph(rows=1), ctx=_CTX)
