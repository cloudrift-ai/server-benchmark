"""Kernel-placement forks over closed stored Fold edges."""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler.backend.cuda.backend import CudaBackend
from emmy.compiler.context import Context
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.frontend.ir import SdpaOp, SoftmaxOp
from emmy.compiler.ir.pure.fold import Channel, Fold
from emmy.compiler.ir.stmt import Assign, Body, Load, Write
from emmy.compiler.ir.tile import Placement, Store, TileOp
from emmy.compiler.pipeline import CUDA_PASSES, TILE_PASSES, Pipeline
from emmy.compiler.pipeline.fork import Fork
from emmy.compiler.pipeline.pipeline import Run, _is_structural_option
from emmy.compiler.pipeline.search.golden import GoldenRecord, decode_record
from emmy.compiler.pipeline.search.pins import pinned_knobs
from emmy.compiler.torch_wire import graph_to_wire
from tests.compiler.helpers import requires_cuda

_CTX = Context.from_target((12, 0))
_OFF = {"WORK": "", "TILE": "", "REDUCE": "", "STAGE": "", "RASTER": ""}


def _input(graph: Graph, name: str, shape, dtype="f16") -> None:
    graph.add_node(InputOp(), [], Tensor(name, shape, dtype), node_id=name)


def _computed_operand_graph(side: str) -> Graph:
    m, n, k = Axis("m", 8), Axis("n", 8), Axis("k", 16)
    computed = Fold.projection(
        body=Body(
            (
                Load(name="raw", input="computed", index=(Var("m" if side == "a" else "n"), Var("k"))),
                Assign(name="scaled", op="multiply", args=("raw", "raw")),
            )
        )
    )
    direct = Load(
        name="direct",
        input="direct",
        index=(Var("k"), Var("n")) if side == "a" else (Var("m"), Var("k")),
    )
    a, b = (computed, direct) if side == "a" else (direct, computed)
    contraction = Fold.contraction(k_axis=k, a=a, channels=(Channel(b=b, acc="acc"),))
    tile = TileOp(op=contraction, name="out", place=Placement(free=(m, n)))
    graph = Graph()
    _input(graph, "computed", (8, 16))
    _input(graph, "direct", (16, 8) if side == "a" else (8, 16))
    graph.add_node(tile, ["computed", "direct"], Tensor("out", (8, 8), "f16"), node_id="out")
    graph.inputs, graph.outputs = ["computed", "direct"], ["out"]
    return graph


def _mimo_graph() -> Graph:
    m, n, k = Axis("m", 8), Axis("n", 8), Axis("k", 16)

    def contraction(a: str, b: str, acc: str) -> Fold:
        return Fold.contraction(
            k_axis=k,
            a=Load(name=f"{a}_v", input=a, index=(Var("m"), Var("k"))),
            channels=(Channel(b=Load(name=f"{b}_v", input=b, index=(Var("k"), Var("n"))), acc=acc),),
        )

    first, second = contraction("a", "b", "first"), contraction("c", "d", "second")
    op = Fold.projection(body=Body(), operands=(first, second))
    tile = TileOp(
        op=op,
        name="out0",
        place=Placement(free=(m, n)),
        stores=(
            Store(Write(output="out0", index=(Var("m"), Var("n")), value="first")),
            Store(Write(output="out1", index=(Var("m"), Var("n")), value="second")),
        ),
    )
    graph = Graph()
    for name in ("a", "c"):
        _input(graph, name, (8, 16))
    for name in ("b", "d"):
        _input(graph, name, (16, 8))
    graph.add_node(
        tile,
        ["a", "b", "c", "d"],
        outputs=(Tensor("out0", (8, 8), "f16"), Tensor("out1", (8, 8), "f16")),
        node_id="out0",
    )
    graph.inputs, graph.outputs = ["a", "b", "c", "d"], ["out0", "out1"]
    return graph


def _sdpa_graph(causal: bool) -> Graph:
    graph = Graph()
    for name in ("q", "k", "v"):
        _input(graph, name, (1, 2, 8, 16))
    graph.add_node(SdpaOp(is_causal=causal), ["q", "k", "v"], Tensor("out", (1, 2, 8, 16), "f16"), node_id="out")
    graph.inputs, graph.outputs = ["q", "k", "v"], ["out"]
    return graph


def _softmax_graph() -> Graph:
    graph = Graph()
    _input(graph, "x", (4, 32))
    graph.add_node(SoftmaxOp(axis=-1), ["x"], Tensor("out", (4, 32), "f16"), node_id="out")
    graph.inputs, graph.outputs = ["x"], ["out"]
    return graph


def _offered(graph: Graph, *, frontend: bool = False) -> list[dict]:
    offered: list[dict] = []
    passes = TILE_PASSES if frontend else ["lowering/tile"]
    select = None if frontend else {"cut"}

    def decide(fork):
        place = [option for option in fork.options if any(str(key).startswith("PLACE") for key in option.knobs)]
        if place:
            offered.extend(dict(option.knobs) for option in place)
            return next(option for option in place if not _is_structural_option(option))
        option = fork.options[0]
        while isinstance(option, Fork) and not option.is_leaf:
            option = option.expand()[0]
        return option

    Run(Pipeline.build(passes, select=select), _CTX).resolve(graph, decide)
    return offered


def _lower_cut(graph: Graph, spelling: str) -> Graph:
    with pinned_knobs({spelling: "cut", **_OFF}):
        lowered = Pipeline.build(CUDA_PASSES).run(graph, ctx=_CTX)
    lowered.validate()
    return lowered


def test_pinned_fusion_lowers_one_computed_operand_kernel() -> None:
    with pinned_knobs({"PLACE": "fuse", **_OFF}):
        lowered = Pipeline.build(CUDA_PASSES).run(_computed_operand_graph("a"), ctx=_CTX)
    assert sum(type(node.op).__name__ == "CudaOp" for node in lowered.nodes.values()) == 1


@pytest.mark.parametrize("side", ("a", "b"))
def test_computed_operand_offers_fused_and_cut_and_pinned_cut_lowers(side: str) -> None:
    offered = _offered(_computed_operand_graph(side))
    assert {frozenset(row.items()) for row in offered} == {
        frozenset({("PLACE", "fuse")}),
        frozenset({("PLACE", "cut")}),
    }
    lowered = _lower_cut(_computed_operand_graph(side), "PLACE")
    cuda = [node for node in lowered.nodes.values() if type(node.op).__name__ == "CudaOp"]
    assert len(cuda) == 2
    assert len(cuda[1].inputs) == 2 and any("__place_" in name for name in cuda[1].inputs)


@pytest.mark.parametrize("causal", (False, True))
def test_sdpa_score_cut_is_offered_and_pinned_cut_lowers(causal: bool) -> None:
    offered = _offered(_sdpa_graph(causal), frontend=True)
    assert {"PLACE": "fuse"} in offered
    assert {"PLACE@a3": "cut"} in offered
    lowered = _lower_cut(_sdpa_graph(causal), "PLACE@a3")
    cuda = [node for node in lowered.nodes.values() if type(node.op).__name__ == "CudaOp"]
    assert len(cuda) == 2
    workspace = next(node.output for node in cuda if "__place_" in node.id)
    assert workspace.dtype.name == "f32"


def test_recorded_sdpa_cut_decodes_exactly_and_stale_path_fails_loudly() -> None:
    wire = graph_to_wire(_sdpa_graph(False))
    fields = {
        "name": "sdpa.route",
        "gpu_name": "",
        "compute_cap": (12, 0),
        "model": None,
        "program_index": 0,
        "program_wire": wire,
        "origins": ("out",),
        "bindings": (),
        "pins": (),
        "measurements": None,
        "ranking": None,
    }
    assert decode_record(GoldenRecord(knobs={"PLACE@a3": "cut"}, **fields)) is None
    reason = decode_record(GoldenRecord(knobs={"PLACE@missing": "cut"}, **fields))
    assert reason is not None and "does not resolve" in reason


@requires_cuda
def test_softmax_state_cut_is_offered_and_pinned_cut_lowers() -> None:
    offered = _offered(_softmax_graph(), frontend=True)
    assert {"PLACE": "fuse"} in offered and {"PLACE": "cut"} in offered
    lowered = _lower_cut(_softmax_graph(), "PLACE")
    cuda = [node for node in lowered.nodes.values() if type(node.op).__name__ == "CudaOp"]
    assert len(cuda) == 2
    assert len(next(node for node in cuda if "__place_" in node.id).outputs) == 2  # maximum + denominator state
    values = np.linspace(-3, 3, 128, dtype=np.float16).reshape(4, 32)
    got = CudaBackend().run(lowered, input_data={"x": values})[0].outputs["out"]
    shifted = values.astype(np.float32) - values.max(axis=-1, keepdims=True).astype(np.float32)
    expected = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(got, expected, rtol=2e-3, atol=2e-3)


def test_mimo_cut_preserves_both_outputs_and_lowers_both_pieces() -> None:
    offered = _offered(_mimo_graph())
    cuts = [next(iter(row)) for row in offered if next(iter(row.values())) == "cut"]
    assert len(cuts) == 2
    lowered = _lower_cut(_mimo_graph(), cuts[0])
    assert lowered.outputs == ["out0", "out1"]
    assert sum(type(node.op).__name__ == "CudaOp" for node in lowered.nodes.values()) == 2
