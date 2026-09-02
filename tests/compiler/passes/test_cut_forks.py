"""Kernel-placement forks over closed stored Fold edges."""

from __future__ import annotations

from dataclasses import replace
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

from emmy.compiler.backend.cuda.backend import CudaBackend
from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F16, F32
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis, Window
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.frontend.ir import SdpaOp, SoftmaxOp
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.pure.carrier import exp_combine_states
from emmy.compiler.ir.pure.fold import Channel, Fold, Lambda, is_contraction, loaded_buffers
from emmy.compiler.ir.schedule.classic_projection import project_classic
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
from emmy.compiler.ir.tile import OutputSpec, Placement, ProjectionRegion, TileOp
from emmy.compiler.ir.tile.path import sites
from emmy.compiler.loop_wire import loop_graph_to_wire
from emmy.compiler.pipeline import CUDA_PASSES, LOOP_PASSES, TILE_PASSES, Match, Pipeline, Rule
from emmy.compiler.pipeline.fork import DeferredFork, Fork
from emmy.compiler.pipeline.passes.lowering.tile._cut import (
    CutSite,
    _environments,
    _external_reads,
    _producer_order,
    _workspace_axes,
    cuttable_seams,
    output_map,
    realize,
)
from emmy.compiler.pipeline.passes.lowering.tile._pieces import projection_region_pieces, realize_projection_regions
from emmy.compiler.pipeline.pipeline import Run, _is_structural_option
from emmy.compiler.pipeline.search.golden import (
    GoldenRecord,
    _candidate_rows,
    _lifted_target,
    decode_record,
    kernel_identity,
    load_golden_file,
    load_golden_records,
    validate_golden_file,
)
from emmy.compiler.pipeline.search.pins import pinned_knobs
from emmy.compiler.torch_wire import graph_to_wire
from tests.compiler.helpers import direct_classic_leaf, requires_cuda

_CTX = Context.from_target((12, 0))
_CUT = import_module("emmy.compiler.pipeline.passes.lowering.tile.030_cut")


def _input(graph: Graph, name: str, shape, dtype="f16") -> None:
    graph.add_node(InputOp(), [], Tensor(name, shape, dtype), node_id=name)


def test_cut_and_assignment_passes_share_the_generic_schedule_driver() -> None:
    from emmy.compiler.ir.schedule import schedule

    assert _CUT.schedule is schedule
    assert import_module("emmy.compiler.pipeline.fork").schedule is schedule


def _split_reduction_graph(op=None) -> Graph:
    """The fresh 761/cc22 producer shape: N is spelled by the clean ``(a1, a25)`` pair."""
    if op is None:
        composite = BinaryExpr("+", BinaryExpr("*", Var("a1"), Literal(128, "int")), Var("a25"))
        reduction = Loop(
            axis=Axis("a26", 1024),
            body=Body(
                (
                    Load(name="weight", input="linear_wt", index=(Var("a26"), composite)),
                    Load(name="norm", input="norm", index=(Var("a26"),)),
                    Load(name="value", input="to", index=(Literal(0, "int"), Var("a0"), Var("a26"))),
                    Load(name="scale", input="scale", index=(Var("a0"),)),
                    Assign(name="scaled", op="multiply", args=("scale", "value")),
                    Assign(name="converted", op="copy", args=("scaled",), dtype=F16),
                    Assign(name="normalized", op="multiply", args=("norm", "converted")),
                    Assign(name="product", op="multiply", args=("normalized", "weight")),
                    Accum(name="acc", value="product", op="add"),
                )
            ),
        )
        cell = Body((reduction, Write(output="out", index=(Var("a0"), Var("a1"), Var("a25")), value="acc")))
        body = Body(
            (
                Loop(
                    axis=Axis("a0", 512),
                    body=Body((Loop(axis=Axis("a1", 16), body=Body((Loop(axis=Axis("a25", 128), body=cell),))),)),
                ),
            )
        )
        op = LoopOp(body=body)

    graph = Graph()
    for name, shape in (
        ("linear_wt", (1024, 2048)),
        ("norm", (1024,)),
        ("to", (1, 512, 1024)),
        ("scale", (512,)),
    ):
        _input(graph, name, shape)
    graph.add_node(op, ["linear_wt", "norm", "to", "scale"], Tensor("out", (512, 16, 128), F16), node_id="out")
    graph.inputs, graph.outputs = ["linear_wt", "norm", "to", "scale"], ["out"]
    return graph


def test_fresh_cut_piece_fuses_newly_clean_split_axes_before_identity_stamp() -> None:
    """A structural cut can remove the access that kept a reshape's output axes distinct. Its
    fresh piece must re-enter the one Loop canonicalization before identity is stamped, so the
    split-axis spelling converges with an initially canonical kernel and exposes the contraction."""
    original = _split_reduction_graph()
    raw_tile = Pipeline.build(["lowering/tile"], select=["lift"]).run(original).nodes["out"].op
    fresh = _split_reduction_graph(replace(raw_tile, name="mul_3__place_761458f514_0", placement_decided=True))
    before = fresh.nodes["out"].op
    assert [axis.extent for axis in before.place.free] == [Dim(512), Dim(16), Dim(128)]
    assert not any(is_contraction(site.node) for site in sites(before.op))

    choice = DeferredFork(lambda: fresh, {"PLACE@root": "cut"}, structural=True)
    (fragment,) = _CUT._canonicalized(choice).expand()
    actual = fragment.nodes["out"].op
    assert actual.name == before.name and actual.placement_decided
    assert [axis.extent for axis in actual.place.free] == [Dim(512), Dim(2048)]
    assert any(is_contraction(site.node) for site in sites(actual.op))
    domains = project_classic(actual, _CTX)
    assert any(choice.tile.is_warp for choices in domains.nodes.values() for choice in choices)

    canonical_loop = Pipeline.build(["loop/canonicalize"]).run(_split_reduction_graph())
    canonical = Pipeline.build(["lowering/tile"], select=["lift"]).run(canonical_loop).nodes["out"].op
    assert actual.loop_body == canonical.loop_body
    assert actual.identity_key() == canonical.identity_key()

    again = fragment.nodes["out"].op
    (same_fragment,) = _CUT._canonicalized(DeferredFork(lambda: fragment, structural=True)).expand()
    assert same_fragment is fragment
    assert fragment.nodes["out"].op is again


def test_placement_cut_preserves_a_cross_cta_split_receipt() -> None:
    """A split piece can re-enter placement; cutting it must not make REDUCE pending again."""
    from emmy.compiler.pipeline.passes.lowering.tile._split import split_pending

    graph = _computed_operand_graph("a")
    tile = graph.nodes["out"].op
    partitioned = replace(tile.op, axis=replace(tile.op.axis, window=Window(parent=tile.op.axis, partition=True)))
    graph.nodes["out"].op = replace(tile, op=partitioned)
    pipeline = Pipeline.build(["lowering/tile"], select={"cut"})
    match = pipeline.match(graph, pipeline.passes[0].rules[0])[0]
    seams = cuttable_seams(match.root.op)
    renamed = output_map(match.root)

    fragment = realize(match, match.root, (seams[0],), renamed)

    pieces = [node.op for node in fragment.nodes.values() if isinstance(node.op, TileOp)]
    assert pieces and all(piece.split_consumed for piece in pieces)
    assert not any(split_pending(piece) for piece in pieces)


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
        output_specs=(
            OutputSpec(Write(output="out0", index=(Var("m"), Var("n")), value="first")),
            OutputSpec(Write(output="out1", index=(Var("m"), Var("n")), value="second")),
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


def _projection_region_graph() -> Graph:
    """A scalar Tile root over two independently owned contraction regions."""
    k = Axis("k", 16)

    def region(row: Axis, column: Axis, a: str, b: str, acc: str) -> ProjectionRegion:
        product = Fold.contraction(
            k_axis=k,
            a=Load(name=f"{a}_v", input=a, index=(Var(row.name), Var("k"))),
            channels=(Channel(b=Load(name=f"{b}_v", input=b, index=(Var("k"), Var(column.name))), acc=acc),),
        )
        inner = ProjectionRegion(
            axis=column,
            lift=Lambda(params=(column.name, row.name), body=Body((product,)), results=(acc,)),
        )
        return ProjectionRegion(axis=row, lift=Lambda(params=(row.name,), body=Body((inner,)), results=()))

    m, n = Axis("m", 4), Axis("n", 8)
    p, q = Axis("p", 8), Axis("q", 4)
    tile = TileOp(
        op=Fold.projection(body=Body((region(m, n, "a", "b", "first"), region(p, q, "c", "d", "second")))),
        output_specs=(
            OutputSpec(Write(output="out0", index=(Var("m"), Var("n")), value="first")),
            OutputSpec(Write(output="out1", index=(Var("p"), Var("q")), value="second")),
        ),
    )
    graph = Graph()
    for name, shape in (("a", (4, 16)), ("b", (16, 8)), ("c", (8, 16)), ("d", (16, 4))):
        _input(graph, name, shape)
    graph.add_node(
        tile,
        ["a", "b", "c", "d"],
        outputs=(Tensor("out0", (4, 8), "f16"), Tensor("out1", (8, 4), "f16")),
        node_id="out0",
    )
    graph.inputs, graph.outputs = ["a", "b", "c", "d"], ["out0", "out1"]
    return graph


def _single_projection_region_graph() -> Graph:
    """A cut consumer with one output region behind a prefix and materialized operand."""
    batch, row, column = Axis("a0", 2), Axis("a1", 4), Axis("a25", 8)
    inner = ProjectionRegion(
        axis=column,
        lift=Lambda(
            params=("a25", "a0", "a1", "scale"),
            body=Body(
                (
                    Load(name="value", input="x", index=(Var("a0"), Var("a1"), Var("a25"))),
                    Assign(name="result", op="multiply", args=("value", "scale")),
                )
            ),
            results=("result",),
        ),
    )
    region = ProjectionRegion(
        axis=row,
        lift=Lambda(params=("a1", "a0", "scale"), body=Body((inner,)), results=()),
    )
    tile = TileOp(
        op=Fold.projection(
            operands=(Load(name="stat", input="stat_workspace", index=(Var("a0"),)),),
            body=Body((Assign(name="scale", op="reciprocal", args=("stat",)), region)),
        ),
        place=Placement(free=(batch,)),
        output_specs=(OutputSpec(Write(output="out", index=(Var("a0"), Var("a1"), Var("a25")), value="result")),),
    )
    graph = Graph()
    _input(graph, "stat_workspace", (2,), dtype="f32")
    _input(graph, "x", (2, 4, 8), dtype="f32")
    graph.add_node(tile, ["stat_workspace", "x"], Tensor("out", (2, 4, 8), F32), node_id="out")
    graph.inputs, graph.outputs = ["stat_workspace", "x"], ["out"]
    return graph


def _shared_provider_region_graph() -> Graph:
    """Two output regions, one reading a scalar provider from the shared prefix."""
    m, n, p = Axis("m", 4), Axis("n", 4), Axis("p", 4)
    contraction_axis, reduction_axis = Axis("k", 8), Axis("h", 8)
    scaled = Fold.projection(
        body=Body(
            (
                Load(name="qv", input="q", index=(Var("m"), Var("k"))),
                Assign(name="av", op="multiply", args=("qv", "scale")),
            )
        ),
        results=("av",),
    )
    contraction = Fold.contraction(
        k_axis=contraction_axis,
        a=scaled,
        channels=(Channel(b=Load(name="wv", input="w", index=(Var("k"), Var("n"))), acc="acc"),),
    )
    reduction = Fold(
        axis=reduction_axis,
        lift=Lambda(
            params=("h",),
            body=Body((Load(name="rv", input="r", index=(Var("n"), Var("h"))),)),
            results=("rv",),
        ),
        init=(0.0,),
        combine=Lambda(
            params=("red", "red__o"),
            body=Body((Assign(name="red", op="add", args=("red", "red__o")),)),
            results=("red",),
        ),
    )
    inner = ProjectionRegion(
        axis=n,
        lift=Lambda(
            params=("n", "m", "scale"),
            body=Body(
                (
                    contraction,
                    reduction,
                    Assign(name="sum", op="add", args=("acc", "red")),
                    Assign(name="result", op="add", args=("sum", "scale")),
                )
            ),
            results=("result",),
        ),
    )
    first = ProjectionRegion(axis=m, lift=Lambda(params=("m", "scale"), body=Body((inner,)), results=()))
    second = ProjectionRegion(
        axis=p,
        lift=Lambda(
            params=("p",),
            body=Body((Load(name="other", input="z", index=(Var("p"),)),)),
            results=("other",),
        ),
    )
    tile = TileOp(
        op=Fold.projection(
            body=Body(
                (
                    Load(name="epsilon", input="epsilon", index=()),
                    Assign(name="scale", op="reciprocal", args=("epsilon",)),
                    first,
                    second,
                )
            )
        ),
        output_specs=(
            OutputSpec(Write(output="out", index=(Var("m"), Var("n")), value="result")),
            OutputSpec(Write(output="other_out", index=(Var("p"),), value="other")),
        ),
    )
    graph = Graph()
    for name, shape in (("epsilon", ()), ("q", (4, 8)), ("w", (8, 4)), ("r", (4, 8)), ("z", (4,))):
        _input(graph, name, shape, dtype="f32")
    graph.add_node(
        tile,
        ["epsilon", "q", "w", "r", "z"],
        outputs=(Tensor("out", (4, 4), F32), Tensor("other_out", (4,), F32)),
        node_id="out",
    )
    graph.inputs, graph.outputs = ["epsilon", "q", "w", "r", "z"], ["out", "other_out"]
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


def _computed_value_expectation_tile() -> TileOp:
    """A twisted expectation whose value is a stored contraction until placement cuts it."""
    query, column, key, inner = Axis("q", 4), Axis("n", 16), Axis("j", 8), Axis("k", 16)
    value = Fold.contraction(
        k_axis=inner,
        a=Load(name="xv", input="x", index=(Var("j"), Var("k"))),
        channels=(Channel(b=Load(name="wv", input="w", index=(Var("n"), Var("k"))), acc="vacc"),),
    )
    states = ("maximum", "denominator", "expectation")
    other = tuple(f"{name}__o" for name in states)
    carrier = Fold(
        axis=key,
        operands=(value,),
        lift=Lambda(
            params=("j", "vacc"),
            body=Body((Load(name="score", input="scores", index=(Var("q"), Var("j"))),)),
            results=("score", 1.0, "vacc"),
        ),
        init=(float("-inf"), 0.0, 0.0),
        combine=Lambda(params=states + other, body=Body(exp_combine_states(states, other)), results=states),
    )
    root = Fold.projection(
        operands=(carrier,),
        body=Body(
            (
                Assign(name="inverse", op="reciprocal", args=("denominator",)),
                Assign(name="out", op="multiply", args=("expectation", "inverse")),
            )
        ),
        results=("out",),
    )
    tile = TileOp(
        op=root,
        name="out",
        place=Placement(free=(query, column)),
        output_specs=(OutputSpec(Write(output="out", index=(Var("q"), Var("n")), value="out")),),
    )
    return replace(
        tile,
        inputs={
            "x": Tensor("x", (8, 16), F16),
            "w": Tensor("w", (16, 16), F16),
            "scores": Tensor("scores", (4, 8), F16),
        },
        outputs={"out": Tensor("out", (4, 16), F16)},
    )


def _typed_value_projection_graph() -> Graph:
    """A typed value contraction paired with a query-dependent result."""
    query, column, key, inner = Axis("query", 4), Axis("column", 8), Axis("key", 16), Axis("v", 32)
    value = Fold.contraction(
        k_axis=inner,
        a=Load(name="x_value", input="x", index=(Var("key"), Var("v"))),
        channels=(Channel(b=Load(name="weight", input="weight", index=(Var("v"), Var("column"))), acc="value"),),
    )
    pair = Fold.projection(
        operands=(value,),
        body=Body(
            (
                Load(name="score", input="score", index=(Var("query"), Var("key"))),
                Assign(name="probability", op="copy", args=("score",), dtype=F32),
                Assign(name="rounded_value", op="copy", args=("value",), dtype=F16),
            )
        ),
        results=("probability", "rounded_value"),
    )
    attention = Fold(
        axis=key,
        operands=(pair,),
        lift=Lambda(
            params=("key", "probability", "rounded_value"),
            body=Body((Assign(name="weighted", op="multiply", args=("probability", "rounded_value")),)),
            results=("weighted",),
        ),
        init=(0.0,),
        combine=Lambda(
            params=("out", "out__o"),
            body=Body((Assign(name="out", op="add", args=("out", "out__o")),)),
            results=("out",),
        ),
    )
    tile = TileOp(
        op=attention,
        name="out",
        place=Placement(free=(query, column)),
        output_specs=(OutputSpec(Write(output="out", index=(Var("query"), Var("column")), value="out")),),
    )
    graph = Graph()
    _input(graph, "score", (4, 16))
    _input(graph, "x", (16, 32))
    _input(graph, "weight", (32, 8))
    graph.add_node(tile, ["score", "x", "weight"], Tensor("out", (4, 8), "f16"), node_id="out")
    graph.inputs, graph.outputs = ["score", "x", "weight"], ["out"]
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


def _lower(graph: Graph, placement: dict[str, str]) -> Graph:
    with pinned_knobs(placement):
        lowered, _ = Run(Pipeline.build(CUDA_PASSES), _CTX).resolve(graph, direct_classic_leaf)
    lowered.validate()
    return lowered


def _lower_cut(graph: Graph, spelling: str) -> Graph:
    return _lower(graph, {spelling: "cut"})


def _nested_attention_cut(pins: dict[str, str]) -> Graph:
    case = Path(__file__).parents[1] / "realization/cases/attention/rmsnorm-gqa-b-cut.yaml"
    (record,) = load_golden_records(load_golden_file(case))
    tile = _lifted_target(record)
    graph = Graph()
    for name, tensor in tile.inputs.items():
        graph.add_node(InputOp(), [], tensor, node_id=name)
    graph.add_node(tile, list(tile.inputs), next(iter(tile.outputs.values())), node_id=tile.name)
    match = Match(graph=graph, root_node_id=tile.name, rule=Rule(name="test", pattern=[]))
    with pinned_knobs(pins):
        result = _CUT.rewrite(match, graph.nodes[tile.name])
    options = result if isinstance(result, list) else [result]
    cut = next(option for option in options if "cut" in option.knobs.values())
    return cut.expand()[0]


def _piece_with_seam(fragment: Graph):
    return next(node for node in fragment.nodes.values() if isinstance(node.op, TileOp) and cuttable_seams(node.op))


def test_cut_workspace_retains_static_unit_axes() -> None:
    """A unit seam axis remains workspace geometry even when the produced value is invariant in it."""
    unit, unused, column = Axis("batch", 1), Axis("unused", 8), Axis("n", 64)
    produced = Fold.projection(
        body=Body((Load(name="value", input="x", index=(Var("n"),)),)),
        results=("value",),
    )
    seam = CutSite(
        node=produced,
        spelling="PLACE",
        axes=(unit, unused, column),
        dtypes=(F16,),
    )

    assert _workspace_axes(seam, produced) == (unit, column)


def test_composed_cut_topologically_orders_equal_degree_workspace_chain() -> None:
    """Counting direct workspace reads cannot order A->C->B when A and C each read one."""

    def piece(name: str, source: str | None):
        body = Body((Load(name=f"{name}_value", input=source or "input", index=()),))
        produced = Fold.projection(body=body, results=(f"{name}_value",))
        return (None, produced, (), (), name, (f"{name}_value",), (name,))

    pieces = [piece("a", "c"), piece("c", "b"), piece("b", None)]

    assert [buffers[0] for *_, buffers in _producer_order(pieces)] == ["b", "c", "a"]


def test_pinned_fusion_lowers_one_computed_operand_kernel() -> None:
    lowered = _lower(_computed_operand_graph("a"), {"PLACE": "fuse"})
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


def test_twisted_expectation_value_cut_uses_the_public_store_dtype() -> None:
    """Cutting a computed expectation value turns it into the derived contraction's B slab."""
    seams = {tuple(seam.node.defines()): seam for seam in cuttable_seams(_computed_value_expectation_tile())}

    assert seams[("vacc",)].dtypes == (F16,)
    assert seams[("maximum", "denominator", "expectation")].dtypes == (F32, F32, F32)


def test_selected_result_cut_uses_its_own_dtype_and_axes() -> None:
    """A selected result retains its sibling without inheriting the sibling's query axis."""
    graph = _typed_value_projection_graph()
    tile = graph.nodes["out"].op
    value = next(seam for seam in cuttable_seams(tile) if seam.selected == "rounded_value")

    assert value.spelling == "PLACE@result.2"
    assert value.dtypes == (F16,)
    match = Match(graph=graph, root_node_id="out", rule=Rule(name="test", pattern=[]))
    with pinned_knobs({value.spelling: "cut"}):
        fork = _CUT.rewrite(match, graph.nodes["out"])
    assert fork.knobs == {value.spelling: "cut"} and _is_structural_option(fork)
    (fragment,) = fork.expand()
    producer = next(node for node in fragment.nodes.values() if "__place_" in node.id)
    consumer = next(node for node in fragment.nodes.values() if isinstance(node.op, TileOp) and node is not producer)

    assert tuple(dim.as_static() for dim in producer.output.shape) == (8, 16)
    assert producer.output.dtype == F16
    assert [axis.name for axis in producer.op.place.free] == ["column", "key"]
    lowered = Body(consumer.op.op.lower())
    assert any(isinstance(stmt, Load) and stmt.name == "rounded_value" for stmt in lowered.iter())
    assert any("probability" in stmt.defines() for stmt in lowered.iter())


def test_a_result_needed_by_its_sibling_can_only_be_cut_as_a_whole() -> None:
    axis = Axis("k", 4)
    child = Fold.projection(
        body=Body(
            (
                Load(name="raw", input="x", index=(Var("k"),)),
                Assign(name="value", op="copy", args=("raw",), dtype=F32),
                Assign(name="dependent", op="multiply", args=("value", "value")),
            )
        ),
        results=("value", "dependent"),
    )
    root = Fold(
        axis=axis,
        operands=(child,),
        lift=Lambda(
            params=("k", "value", "dependent"),
            body=Body((Assign(name="product", op="multiply", args=("value", "dependent")),)),
            results=("product",),
        ),
        init=(0.0,),
        combine=Lambda(
            params=("out", "out__o"),
            body=Body((Assign(name="out", op="add", args=("out", "out__o")),)),
            results=("out",),
        ),
    )
    tile = replace(TileOp(op=root), inputs={"x": Tensor("x", (4,), F32)}, outputs={"out": Tensor("out", (), F32)})

    spellings = {seam.spelling for seam in cuttable_seams(tile)}

    assert "PLACE@result.1" not in spellings
    assert "PLACE@result.2" in spellings


def test_mimo_cut_preserves_both_outputs_and_lowers_both_pieces() -> None:
    offered = _offered(_mimo_graph())
    cuts = [next(iter(row)) for row in offered if next(iter(row.values())) == "cut"]
    assert len(cuts) == 2
    lowered = _lower_cut(_mimo_graph(), cuts[0])
    assert lowered.outputs == ["out0", "out1"]
    assert sum(type(node.op).__name__ == "CudaOp" for node in lowered.nodes.values()) == 2


def test_root_projection_region_cut_lifts_each_piece_placement() -> None:
    offered = _offered(_projection_region_graph())
    assert {"PLACE": "fuse"} in offered and {"PLACE@root": "cut"} in offered

    graph = _projection_region_graph()
    match = Match(graph=graph, root_node_id="out0", rule=Rule(name="test", pattern=[]))
    with pinned_knobs({"PLACE@root": "cut"}):
        choice = _CUT.rewrite(match, graph.nodes["out0"])
    choice = choice[0] if isinstance(choice, list) else choice
    fragment = choice.expand()[0]
    pieces = [node.op for node in fragment.nodes.values() if isinstance(node.op, TileOp)]

    assert fragment.outputs == ["out0__split", "out1__split"]
    assert [[axis.extent.as_static() for axis in piece.place.free] for piece in pieces] == [[4, 8], [8, 4]]
    assert all(not piece.placement_decided for piece in pieces)
    assert [{spec.write.output for spec in piece.output_specs} for piece in pieces] == [
        {"out0__split"},
        {"out1__split"},
    ]


def test_root_projection_region_cut_lifts_a_single_remaining_region() -> None:
    """A prior cut may leave one region whose axes still need a root placement choice."""
    graph = _single_projection_region_graph()
    offered = _offered(graph)
    assert {"PLACE": "fuse"} in offered
    assert {"PLACE@root": "cut"} in offered

    graph = _single_projection_region_graph()
    root = graph.nodes["out"]
    match = Match(graph=graph, root_node_id=root.id, rule=Rule(name="test", pattern=[]))
    with pinned_knobs({"PLACE@root": "fuse"}):
        fused = _CUT.rewrite(match, root).expand()[0]
    assert isinstance(fused, TileOp)
    assert [axis.name for axis in fused.place.free] == ["a0"]
    assert any(isinstance(member, ProjectionRegion) for member in fused.op.body.iter())

    graph = _single_projection_region_graph()
    root = graph.nodes["out"]
    match = Match(graph=graph, root_node_id=root.id, rule=Rule(name="test", pattern=[]))
    with pinned_knobs({"PLACE@root": "cut"}):
        choice = _CUT.rewrite(match, root)
    choice = choice[0] if isinstance(choice, list) else choice
    fragment = choice.expand()[0]
    pieces = [node.op for node in fragment.nodes.values() if isinstance(node.op, TileOp)]

    assert len(pieces) == 1
    piece = pieces[0]
    assert [axis.name for axis in piece.place.free] == ["a0", "a1", "a25"]
    assert not any(isinstance(member, ProjectionRegion) for member in piece.op.body.iter())
    assert loaded_buffers(piece.op) == {"stat_workspace", "x"}
    assert fragment.outputs == ["out__split"]


def test_root_projection_region_pin_lowers_two_independently_scheduled_kernels() -> None:
    lowered = _lower(_projection_region_graph(), {"PLACE@root": "cut"})

    assert lowered.outputs == ["out0", "out1"]
    assert sum(type(node.op).__name__ == "CudaOp" for node in lowered.nodes.values()) == 2
    assert not any(isinstance(node.op, TileOp) for node in lowered.nodes.values())


def test_composed_cut_keeps_a_shared_prefix_provider_on_an_output_piece() -> None:
    """Replacing one nested Fold must not move the output piece's shared provider exclusively
    under a sibling contraction and leave the piece's remaining consumer open."""
    graph = _shared_provider_region_graph()
    root = graph.nodes["out"]
    match = Match(graph=graph, root_node_id=root.id, rule=Rule(name="test", pattern=[]))
    fragment = realize_projection_regions(match, root, projection_region_pieces(root.op))
    piece = next(
        node
        for node in fragment.nodes.values()
        if isinstance(node.op, TileOp) and any(spec.write.output == "out__split" for spec in node.op.output_specs)
    )
    assert set(piece.op.op.deps()) <= {axis.name for axis in piece.op.place.free}

    seams = tuple(seam for seam in cuttable_seams(piece.op) if seam.node.axis is not None and seam.node.axis.name in {"k", "h"})
    assert {seam.node.axis.name for seam in seams} == {"k", "h"}
    composed = realize(
        Match(graph=fragment, root_node_id=piece.id, rule=Rule(name="test", pattern=[])),
        piece,
        seams,
        output_map(piece),
        placement_decided=True,
    )

    for child in (node.op for node in composed.nodes.values() if isinstance(node.op, TileOp)):
        assert set(child.op.deps()) <= {axis.name for axis in child.place.free}


def test_root_region_cut_precedes_scoped_child_site_resolution() -> None:
    """A root cut changes the kernel set before graph-scoped child paths become meaningful."""
    graph = _shared_provider_region_graph()
    tile = graph.nodes["out"].op

    with pinned_knobs({"PLACE@root": "cut", "PLACE@a": "cut"}):
        restriction = _CUT._placement_restriction(tile, cuttable_seams(tile), projection_region_pieces(tile))

    assert restriction == (("PLACE@root",), "regions", {})


def test_scoped_place_cut_is_consumed_once_by_both_pieces() -> None:
    fragment = _nested_attention_cut({"PLACE@map.fold.a.fold.b1": "cut"})
    pieces = [node for node in fragment.nodes.values() if isinstance(node.op, TileOp)]

    assert pieces and all(node.op.placement_decided for node in pieces)
    node = _piece_with_seam(fragment)
    match = Match(graph=fragment, root_node_id=node.id, rule=Rule(name="test", pattern=[]))
    with pinned_knobs({"PLACE@map.fold.a.fold.b1": "cut"}):
        result = _CUT.rewrite(match, node)
    options = result if isinstance(result, list) else [result]
    assert options and all(not any(name.startswith("PLACE") for name in option.knobs) for option in options)


def test_bare_place_cut_is_consumed_once_by_both_pieces() -> None:
    fragment = _nested_attention_cut({"PLACE": "cut"})
    node = _piece_with_seam(fragment)
    match = Match(graph=fragment, root_node_id=node.id, rule=Rule(name="test", pattern=[]))

    assert node.op.placement_decided
    with pinned_knobs({"PLACE": "cut"}):
        result = _CUT.rewrite(match, node)
    options = result if isinstance(result, list) else [result]
    assert options and all(not any(name.startswith("PLACE") for name in option.knobs) for option in options)


def test_unpinned_place_keeps_offering_fuse_and_recursive_cuts() -> None:
    fragment = _nested_attention_cut({})
    node = _piece_with_seam(fragment)
    match = Match(graph=fragment, root_node_id=node.id, rule=Rule(name="test", pattern=[]))

    assert not node.op.placement_decided
    options = _CUT.rewrite(match, node)
    assert {"fuse", "cut"} <= {value for option in options for value in option.knobs.values()}


def test_composed_scoped_place_pins_cut_together_and_foreign_pins_are_skipped() -> None:
    """Every scoped PLACE pin that resolves on one kernel joins ONE realization — a producer per
    seam and one consumer, with a producer reading another seam's workspace when its value nests
    inside it — while a pin whose site path exists on no kernel here is another kernel's and is
    skipped, never an error."""
    case = Path(__file__).parents[1] / "realization/cases/attention/rmsnorm-qk-sdpa-composed-cut.yaml"
    (record,) = load_golden_records(load_golden_file(case))
    tile = _lifted_target(record)
    graph = Graph()
    for name, tensor in tile.inputs.items():
        graph.add_node(InputOp(), [], tensor, node_id=name)
    graph.add_node(tile, list(tile.inputs), next(iter(tile.outputs.values())), node_id=tile.name)
    match = Match(graph=graph, root_node_id=tile.name, rule=Rule(name="test", pattern=[]))
    pins = {
        "PLACE@map.fold.a21": "cut",
        "PLACE@map.fold.a.map.fold.fold.b1": "cut",
        "PLACE@map.fold.a.map.fold.fold.a1": "cut",
        "PLACE@map.map.map.map1": "cut",  # no such site here — another kernel's pin
    }
    with pinned_knobs(pins):
        fork = _CUT.rewrite(match, graph.nodes[tile.name])
    assert set(fork.knobs) == {"PLACE@map.fold.a21", "PLACE@map.fold.a.map.fold.fold.b1", "PLACE@map.fold.a.map.fold.fold.a1"}
    (fragment,) = fork.expand()
    pieces = [node for node in fragment.nodes.values() if isinstance(node.op, TileOp)]
    producers = [node for node in pieces if "__place_" in node.id]
    assert len(producers) == 3 and len(pieces) == 4
    assert all(node.op.placement_decided for node in pieces)
    workspaces = {node.id for node in producers}
    assert any(set(node.inputs) & workspaces for node in producers), "the nested value's producer must read a sibling workspace"


def test_bare_and_scoped_place_cuts_compose_in_one_decision() -> None:
    match, graph = _composed_case_match()
    pins = {"PLACE": "cut", "PLACE@map.fold.a.map.fold.fold.b1": "cut"}

    with pinned_knobs(pins):
        fork = _CUT.rewrite(match, graph.nodes[match.root_node_id])

    assert len(fork.knobs) == 2 and set(fork.knobs.values()) == {"cut"}
    (fragment,) = fork.expand()
    pieces = [node for node in fragment.nodes.values() if isinstance(node.op, TileOp)]
    assert len(pieces) == 3 and all(node.op.placement_decided for node in pieces)


def test_selected_result_replacement_applies_a_nested_cut_to_its_retained_slice() -> None:
    """A retained result slice remains a consumer of every nested seam cut beside it."""
    k = Axis("k", 8)
    nested = Fold(
        axis=k,
        lift=Lambda(
            params=("k",),
            body=Body((Load(name="value", input="values", index=(Var("k"),)),)),
            results=("value",),
        ),
        init=(0.0,),
        combine=Lambda(
            params=("total", "total__o"),
            body=Body((Assign(name="total", op="add", args=("total", "total__o")),)),
            results=("total",),
        ),
    )
    pair = Fold.projection(
        operands=(nested, Load(name="other", input="other", index=())),
        body=Body(
            (
                Assign(name="kept", op="copy", args=("total",), dtype=F32),
                Assign(name="selected", op="copy", args=("other",), dtype=F32),
            )
        ),
        results=("kept", "selected"),
    )
    root_fold = Fold.projection(
        operands=(pair,),
        body=Body((Assign(name="out", op="add", args=("kept", "selected"), dtype=F32),)),
        results=("out",),
    )
    tile = TileOp(op=root_fold, name="out")
    graph = Graph()
    _input(graph, "values", (8,), dtype="f32")
    _input(graph, "other", (), dtype="f32")
    graph.add_node(tile, ["values", "other"], Tensor("out", (), F32), node_id="out")
    graph.inputs, graph.outputs = ["values", "other"], ["out"]
    node = graph.nodes["out"]
    seams = cuttable_seams(tile)
    selected = next(seam for seam in seams if seam.node is pair and seam.selected == "selected")
    nested_seam = next(seam for seam in seams if seam.node is nested and seam.selected is None)

    fragment = realize(
        Match(graph=graph, root_node_id=node.id, rule=Rule(name="test", pattern=[])),
        node,
        (selected, nested_seam),
        output_map(node),
        placement_decided=True,
    )

    consumer = next(piece for piece in fragment.nodes.values() if isinstance(piece.op, TileOp) and "__place_" not in piece.id)
    producer = next(piece for piece in fragment.nodes.values() if isinstance(piece.op, TileOp) and "values" in piece.inputs)
    assert producer.id in loaded_buffers(consumer.op.op)
    assert "values" not in loaded_buffers(consumer.op.op)
    assert all(site.node.axis is None or site.node.axis.name != "k" for site in sites(consumer.op.op))


def _composed_case_match() -> tuple[Match, Graph]:
    case = Path(__file__).parents[1] / "realization/cases/attention/rmsnorm-qk-sdpa-composed-cut.yaml"
    (record,) = load_golden_records(load_golden_file(case))
    tile = _lifted_target(record)
    graph = Graph()
    for name, tensor in tile.inputs.items():
        graph.add_node(InputOp(), [], tensor, node_id=name)
    graph.add_node(tile, list(tile.inputs), next(iter(tile.outputs.values())), node_id=tile.name)
    return Match(graph=graph, root_node_id=tile.name, rule=Rule(name="test", pattern=[])), graph


_STAT_PINS = {
    "PLACE@map.fold.a21": "cut",
    "PLACE@map.fold.a.map.fold.fold.b1": "cut",
    "PLACE@map.fold.a.map.fold.fold.a1": "cut",
    "PLACE@map.fold.a1": "cut",
    "PLACE@map.fold.a.map.fold.a31": "cut",
    "PLACE@map.fold.a.map.fold.a32": "cut",
}


def test_statistics_seams_close_via_providers_and_declare_requirements() -> None:
    """Provider closure records every fold-produced capture as a requirement."""
    match, graph = _composed_case_match()
    tile = next(node.op for node in graph.nodes.values() if isinstance(node.op, TileOp))
    seams = {seam.spelling: seam for seam in cuttable_seams(tile)}
    norm = seams["PLACE@map.fold.a21"]
    first = seams["PLACE@map.fold.a.map.fold.a31"]
    second = seams["PLACE@map.fold.a.map.fold.a32"]
    assert first.providers and [producer for _, producer in first.requires] == [norm.node]
    assert second.providers and [producer for _, producer in second.requires] == [norm.node, first.node]


def test_dependent_seam_pins_pull_their_producer_into_the_composed_cut() -> None:
    """Pinning only the second statistics pass cuts the first beside it — the requirement is
    structural, so the pin cannot decline it."""
    match, graph = _composed_case_match()
    node = next(node for node in graph.nodes.values() if isinstance(node.op, TileOp))
    with pinned_knobs({"PLACE@map.fold.a.map.fold.a32": "cut"}):
        fork = _CUT.rewrite(match, node)
    assert set(fork.knobs) == {
        "PLACE@map.fold.a.map.fold.a32",
        "PLACE@map.fold.a.map.fold.a31",
        "PLACE@map.fold.a21",
    }


def test_statistics_route_shares_the_row_reduction_across_output_keys() -> None:
    """The composed statistics route computes each row statistic once per query: no piece repeats
    a key-extent reduce beneath its output-key axis, and the softmax-weight piece keeps only the
    per-element score contraction (the recompute PR #679 measured at three orders of magnitude)."""
    match, graph = _composed_case_match()
    node = next(node for node in graph.nodes.values() if isinstance(node.op, TileOp))
    key_extent = 8

    def reduce_extents(op) -> list[int]:
        out = []
        stack = [op]
        while stack:
            current = stack.pop()
            if isinstance(current, Fold):
                if current.axis is not None:
                    out.append(current.axis.extent.as_static())
                stack.extend(current.operands)
                stack.extend(current.lift.body)
        return out

    with pinned_knobs(_STAT_PINS):
        fragment = _CUT.rewrite(match, node).expand()[0]
    pieces = [piece for piece in fragment.nodes.values() if isinstance(piece.op, TileOp)]
    assert len(pieces) == 7  # six workspaces plus the consumer
    workspaces = {piece.id for piece in pieces if "__place_" in piece.id}
    # The two statistics pieces each run the key-extent scan ONCE, into a per-query workspace
    # (batch·head × query — no output-key axis).
    statistics = [piece for piece in pieces if "__place_" in piece.id and reduce_extents(piece.op.op).count(key_extent) == 1]
    assert len(statistics) == 2
    assert all(len(piece.op.place.free) == 2 for piece in statistics)
    # The softmax-weight piece sweeps the output-key axis, reads those workspaces back, and keeps
    # only the per-element score contraction — the key-extent scan does not reappear beneath its
    # output-key axis, and neither does it in the consumer beyond the softmax·V contraction itself.
    weight = next(piece for piece in pieces if "__place_" in piece.id and len(piece.op.place.free) == 3 and set(piece.inputs) & workspaces)
    assert key_extent not in reduce_extents(weight.op.op)
    consumer = next(piece for piece in pieces if "__place_" not in piece.id)
    assert reduce_extents(consumer.op.op).count(key_extent) == 1


def _receipt_fields() -> dict:
    return {
        "name": "sdpa.child",
        "gpu_name": "",
        "compute_cap": (12, 0),
        "model": None,
        "program_index": 0,
        "program_wire": graph_to_wire(_sdpa_graph(False)),
        "origins": ("out",),
        "bindings": (),
        "pins": (("PLACE@a3", "cut"),),
        "measurements": None,
        "ranking": None,
    }


def test_child_identity_receipts_decode_per_child_and_join_by_stored_identity() -> None:
    """Conflicting per-child schedules behind one pinned cut persist as sibling receipts: each
    stored child identity selects its own kernel's rows, a sibling child's row does not vouch for
    it, and the verified-tier join reads the stored identity instead of the pre-cut lift."""
    fields = _receipt_fields()
    parent = GoldenRecord(knobs={}, **fields)
    lift_identity = _lifted_target(parent).identity_key(with_io=True)
    children = {i: rows for i, rows in _candidate_rows(parent).items() if i is not None and i != lift_identity}
    assert len(children) == 2, "the pinned cut must resolve to two distinctly identified child kernels"
    (id_a, rows_a), (id_b, rows_b) = sorted(children.items())
    row_a = next(iter(rows_a - rows_b), None)
    assert row_a is not None, "the children must offer at least one distinguishing schedule row"

    receipt = GoldenRecord(knobs=dict(row_a), identity=id_a, **fields)
    assert decode_record(receipt) is None
    assert kernel_identity(receipt) == id_a

    sibling = GoldenRecord(knobs=dict(row_a), identity=id_b, **fields)
    reason = decode_record(sibling)
    assert reason is not None and "no enumerated row of the identified kernel" in reason

    stale = GoldenRecord(knobs=dict(row_a), identity="0" * 64, **fields)
    reason = decode_record(stale)
    assert reason is not None and "equals none" in reason


def test_child_identity_receipt_selects_one_kernel_from_multi_kernel_loop_target() -> None:
    """A stored child identity is the selector when a regenerated target now lowers to several
    kernels; strict decoding must consult that identity's rows before requiring a one-kernel lift."""
    graph = _sdpa_graph(False)
    _input(graph, "x", (4, 32))
    graph.add_node(SoftmaxOp(axis=-1), ["x"], Tensor("softmax", (4, 32), "f16"), node_id="softmax")
    graph.inputs.append("x")
    graph.outputs.append("softmax")
    loop = Pipeline.build(LOOP_PASSES).run(graph.copy(), ctx=_CTX)
    fields = {
        **_receipt_fields(),
        "program_wire": graph_to_wire(graph),
        "origins": (),
        "loop_index": 0,
        "loop_wire": loop_graph_to_wire(loop),
    }
    parent = GoldenRecord(knobs={}, **fields)
    with pytest.raises(ValueError, match="target lowers to 2 kernels"):
        _lifted_target(parent)
    identity, rows = next((identity, rows) for identity, rows in _candidate_rows(parent).items() if identity is not None)
    receipt = GoldenRecord(knobs=dict(next(iter(rows))), identity=identity, **fields)
    assert decode_record(receipt) is None


def test_receipt_validation_requires_child_identity_and_place_pins_stay_live() -> None:
    from emmy.compiler.pipeline.search.policy.greedy import _pins_live

    fields = _receipt_fields()
    document = {
        "compute_cap": [12, 0],
        "programs": [fields["program_wire"]],
        "configs": [
            {
                "program": 0,
                "target": {"origins": ["out"]},
                "realizations": [{"name": "sdpa.child", "bindings": {}, "pins": {"PLACE@a3": "cut"}, "knobs": {"WORK": "w4x2"}}],
            }
        ],
    }
    with pytest.raises(ValueError, match="child-identity schedule receipt"):
        validate_golden_file(document)
    document["configs"][0]["realizations"][0]["identity"] = "0" * 64
    validate_golden_file(document)
    assert _pins_live({"PLACE@a3": "cut"}), "a receipt's routing pins are replay context, never a dead env regime"


def test_pool_group_fuses_node_id_respellings_and_keys_on_pins() -> None:
    """``pool_group`` composes the target kernels' identity keys, so two recordings of ONE
    program whose node ids differ (separate recording sessions) fuse into one enumeration
    group — the wire digest this replaced split them — while a different pin regime still
    keys apart."""
    fields = _receipt_fields()
    respelled = _sdpa_graph(False)
    for nid in [n for n in respelled.nodes if n not in respelled.inputs]:
        respelled.rename_node(nid, f"session2_{nid}")
    twin_fields = {
        **fields,
        "program_wire": graph_to_wire(respelled),
        "origins": tuple(f"session2_{o}" for o in fields["origins"]),
    }
    a = GoldenRecord(knobs={}, **fields)
    b = GoldenRecord(knobs={}, **twin_fields)
    assert a.pool_group == b.pool_group, "node-id spelling must not split an enumeration group"

    unpinned = GoldenRecord(knobs={}, **{**fields, "pins": ()})
    assert unpinned.pool_group != a.pool_group, "the pin regime is a group-key term"


def test_lowered_captures_resolve_in_program_order() -> None:
    """A definition covers only the reads that FOLLOW it.

    A stored tree may hold one value object at two positions — a projection member and, canonically
    shared, deep inside a contraction-operand cone. A seam that keeps only the deeper position
    lowers the definition inside the reduce loop, AFTER a shallower sibling read of the same name:
    the name is still a capture the piece must receive as a provider. Order-blind resolution let
    the later definition mask the read, offered the seam as closed, and cut a piece whose reader
    had no definition — nvcc's undefined identifier on DeepSeek-V4 post4096's composed-cut
    ``mean_reduce`` piece."""
    scalar = Load(name="x", input="eps", index=())
    cone = Fold.projection(body=Body((scalar, Assign(name="v", op="rsqrt", args=("x",)))), results=("v",))
    weight = Load(name="w_e", input="w", index=(Var("k"),))
    b_edge = Fold.projection(operands=(cone,), body=Body((weight, Assign(name="b", op="multiply", args=("w_e", "v")))), results=("b",))
    inner = Fold.contraction(
        k_axis=Axis("k", 8),
        a=Load(name="a_e", input="a", index=(Var("k"),)),
        channels=(Channel(b=b_edge, acc="acc"),),
    )
    node = Fold.projection(
        body=Body((Assign(name="r", op="rsqrt", args=("x",)), inner, Assign(name="out", op="multiply", args=("r", "acc")))),
        results=("out",),
    )
    assert "x" in node.deps(), "the stored tree knows the shallow read is a capture"
    assert "x" in _external_reads(node), "the lowered accounting must agree: a later, deeper definition covers nothing"


def test_masked_capture_threads_into_the_piece_and_the_guardrail_fires_without_it() -> None:
    """End to end at the cut: the recovered capture becomes a provider, and its absence is loud.

    The seam is a reducing fold whose lift reads ``x`` ahead of a contraction holding the ``x``
    cone on its b edge (the deeper position lowers inside the reduce loop). Provider closure must
    NEED ``x``, resolve it in the host body, and prepend the ``Load`` to the produced piece; a
    seam stripped of that provider must trip realization's read-before-definition assert instead
    of reaching nvcc."""
    j, k = Axis("j", 4), Axis("k", 16)
    shared = Load(name="x", input="eps", index=())
    cone = Fold.projection(body=Body((shared, Assign(name="v", op="rsqrt", args=("x",)))), results=("v",))
    weight = Load(name="w_v", input="w", index=(Var("k"),))
    b_edge = Fold.projection(
        operands=(cone,),
        body=Body((weight, Assign(name="bv", op="multiply", args=("w_v", "v")), Assign(name="bs", op="multiply", args=("bv", "s")))),
        results=("bs",),
    )
    inner = Fold.contraction(
        k_axis=k,
        a=Load(name="a_v", input="a", index=(Var("j"), Var("k"))),
        channels=(Channel(b=b_edge, acc="acc"),),
    )
    reducing = Fold(
        axis=j,
        lift=Lambda(
            params=("j",),
            body=Body(
                (
                    Assign(name="r", op="rsqrt", args=("x",)),
                    Assign(name="s", op="rsqrt", args=("r",)),
                    inner,
                    Assign(name="step", op="multiply", args=("acc", "r")),
                )
            ),
            results=("step",),
        ),
        init=(0.0,),
        combine=Lambda(params=("acc6", "other"), body=Body((Assign(name="acc6", op="add", args=("acc6", "other")),)), results=("acc6",)),
    )
    host = Fold.projection(body=Body((shared, reducing, Assign(name="out", op="rsqrt", args=("acc6",)))), results=("out",))
    tile = TileOp(
        op=host,
        name="host_kernel",
        place=Placement(free=()),
        output_specs=(OutputSpec(Write(output="out_buf", index=(), value="out")),),
    )
    assert tile.op is host, "the shape must survive construction normalization unchanged"

    graph = Graph()
    for name, shape in (("eps", (1,)), ("a", (4, 16)), ("w", (16,))):
        _input(graph, name, shape, dtype="f32")
    graph.add_node(tile, ["eps", "a", "w"], outputs=(Tensor("out_buf", (), "f32"),), node_id="host_kernel")
    match = Match(graph=graph, root_node_id="host_kernel", rule=Rule(name="test", pattern=[]))
    root = graph.nodes["host_kernel"]

    seam = next(s for s in cuttable_seams(tile) if s.node is reducing)
    assert [tuple(p.defines()) for p in seam.providers] == [("x",)], "the masked capture resolves to its host Load"

    fragment = realize(match, root, (seam,), output_map(root))
    piece = next(node.op for node in fragment.nodes.values() if isinstance(node.op, TileOp) and "__place_" in node.op.name)
    assert piece.op.lift.body[0] is seam.providers[0], "the provider Load opens the piece"
    assert not _external_reads(piece.op), "the produced piece reads nothing before its definition"

    stripped = CutSite(node=seam.node, spelling=seam.spelling, axes=seam.axes, dtypes=seam.dtypes)
    with pytest.raises(AssertionError, match=r"reads \['x'\] before any definition"):
        realize(match, root, (stripped,), output_map(root))


def test_a_fold_held_by_a_plain_statement_still_gets_an_environment() -> None:
    """A plain statement binds axes, not SSA definitions — but it can HOLD a stored fold.

    ``ProjectionRegion`` keeps its cones as terms, and a fold reached only that way had no lexical
    environment at all, so provider closure could not resolve its captures and silently dropped its
    seam. The canonical tree walk alternates node-wise and statement-wise for the same reason."""
    from emmy.compiler.ir.pure import Lambda
    from emmy.compiler.ir.tile.ir import ProjectionRegion

    cone = Fold.projection(body=Body((Assign(name="c", op="relu", args=("x",)),)), results=("c",))
    region = ProjectionRegion(axis=Axis("j", 4), lift=Lambda(params=("j",), body=Body((cone,)), results=("c",)))
    root = Fold.projection(
        body=Body((Load(name="x", input="a", index=(Var("j"),)), region, Assign(name="out", op="copy", args=("c",)))),
        results=("out",),
    )

    assert id(cone) in _environments(root), "a cone a region holds must still resolve its captures"
    assert _environments(root)[id(cone)] == [(root,)]


def _cone_seam(providers: tuple = (), requires: tuple = ()) -> CutSite:
    """A bare seam record standing in for a clustered operand cone."""
    node = Fold.projection(body=Body((Load(name="w", input="w", index=(Var("n"), Var("k"))),)), results=("w",))
    return CutSite(node=node, spelling="PLACE@b", axes=(Axis("n", 8), Axis("k", 8)), dtypes=(F16,), providers=providers, requires=requires)


def test_two_cones_that_close_over_different_sources_are_not_one_value() -> None:
    """Clustering merges cones that are alpha-equivalent — but a capture is a FREE name.

    Two B cones can spell ``w[n,k] * x`` identically while one host defines ``x = sum(a)`` and the
    other ``x = sum(b)``; normalization refuses to sink either reduce, so both cones keep the same
    free name. Merging them materializes one and lets the other read it, which silently hands the
    second contraction the first's value. The closure is part of the value."""
    from emmy.compiler.pipeline.passes.lowering.tile._cut import _cluster_value_seams

    first_source = Fold.projection(body=Body((Load(name="x", input="a", index=(Var("k"),)),)), results=("x",))
    second_source = Fold.projection(body=Body((Load(name="x", input="b", index=(Var("k"),)),)), results=("x",))
    consumer = object()

    same = [_cone_seam(requires=(("x", first_source),)), _cone_seam(requires=(("x", first_source),))]
    differing = [_cone_seam(requires=(("x", first_source),)), _cone_seam(requires=(("x", second_source),))]

    clustered = _cluster_value_seams(same, {id(seam.node): consumer for seam in same})
    assert len(clustered) == 1 and len(clustered[0].siblings) == 1

    kept = _cluster_value_seams(differing, {id(seam.node): consumer for seam in differing})
    assert len(kept) == 2 and not any(seam.siblings for seam in kept)


def test_a_required_producer_keeps_its_own_seam() -> None:
    """A dependent reads its producer's workspace by the name that producer BINDS.

    Clustering re-points a value at its representative, whose result names are its own, so folding
    a required producer into somebody else's cluster leaves the requirement naming a seam that no
    longer exists — or, worse, one that binds a different name."""
    from emmy.compiler.pipeline.passes.lowering.tile._cut import _cluster_value_seams

    # The producer is deliberately NOT first: the cluster representative is whoever leads, so a
    # required producer that trails would be folded away as a sibling.
    twin, producer = _cone_seam(), _cone_seam()
    dependent = _cone_seam(requires=(("w", producer.node),))
    consumer = object()
    seams = [twin, producer, dependent]

    kept = _cluster_value_seams(seams, {id(seam.node): consumer for seam in seams})

    assert any(seam.node is producer.node for seam in kept), "the required producer must survive as its own seam"
    assert not any(sibling is producer.node for seam in kept for sibling, _ in seam.siblings)


def test_a_dependent_seam_is_an_unpinned_composed_arm() -> None:
    """The unpinned fork offers a dependent seam WITH its transitive producer closure — one arm,
    composed exactly as the pin path composes it. The plain-only ballot could never elect the one
    placement measured to work on DeepSeek-V4 post4096 (a dependent seam's closure), however the
    evidence ranked: the arm was not offered."""
    match, graph = _composed_case_match()
    node = next(node for node in graph.nodes.values() if isinstance(node.op, TileOp))
    options = _CUT.rewrite(match, node)
    arms = [dict(option.knobs) for option in options if "cut" in option.knobs.values()]
    dependent = {
        "PLACE@map.fold.a.map.fold.a32": "cut",
        "PLACE@map.fold.a.map.fold.a31": "cut",
        "PLACE@map.fold.a21": "cut",
    }
    assert dependent in arms, f"the dependent seam's closure must be one composed arm, got {arms}"
    seams = cuttable_seams(node.op)
    offered = {spelling for arm in arms for spelling in arm}
    missing = {seam.spelling for seam in seams} - offered
    assert not missing, f"every seam must appear on the ballot through some closure: {missing}"
