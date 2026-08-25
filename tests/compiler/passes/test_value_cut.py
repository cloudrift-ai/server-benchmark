"""Closed fused-value materialization into ordinary multi-output graph fragments."""

from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F8E4M3, F16, F32
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import Match, Pipeline, Rule
from emmy.compiler.pipeline.fork import flatten_leaves
from emmy.compiler.pipeline.passes.lowering.tile._value_cut import (
    realize_value_cut,
    route_value_cut,
    spell_value_cut,
    value_cut_sites,
)
from emmy.compiler.pipeline.pipeline import Run
from emmy.compiler.pipeline.search.pins import pinned_knobs
from tests.compiler.passes.test_value_demand import _x_f8_like


def _match(graph: Graph, root: str) -> Match:
    return Match(graph=graph, root_node_id=root, rule=Rule("test", []), consumed={root})


def _live_output_graph() -> Graph:
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (Dim(4), Dim(8)), F16), node_id="x")
    graph.add_node(InputOp(), [], Tensor("w", (Dim(8), Dim(8)), F16), node_id="w")
    graph.add_node(
        _x_f8_like(),
        ["x", "w"],
        outputs=(Tensor("x_f8", (Dim(4), Dim(8)), F8E4M3), Tensor("y", (Dim(4), Dim(8)), F32)),
        node_id="x_f8",
    )
    graph.inputs, graph.outputs = ["x", "w"], ["y", "x_f8"]
    return graph


def _workspace_graph() -> Graph:
    m = Axis("m", Dim(4))
    n = Axis("n", Dim(8))
    k = Axis("k", Dim(8))
    reduce = Loop(
        axis=k,
        body=Body(
            (
                Load(name="xv", input="x", index=(Var("m"), Var("k"))),
                Assign(name="q", op="to_f8e4m3", args=("xv",), dtype=F8E4M3),
                Assign(name="decoded", op="from_f8e4m3", args=("q",), dtype=F16),
                Load(name="wv", input="w", index=(Var("n"), Var("k"))),
                Assign(name="product", op="multiply", args=("decoded", "wv")),
                Accum(name="acc", value="product"),
            )
        ),
    )
    loop = LoopOp(
        body=Body(
            (
                Loop(
                    axis=m,
                    body=Body(
                        (
                            Loop(
                                axis=n,
                                body=Body((reduce, Write(output="y", index=(Var("m"), Var("n")), value="acc"))),
                            ),
                        )
                    ),
                ),
            )
        )
    )
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (Dim(4), Dim(8)), F16), node_id="x")
    graph.add_node(InputOp(), [], Tensor("w", (Dim(8), Dim(8)), F16), node_id="w")
    graph.add_node(loop, ["x", "w"], Tensor("y", (Dim(4), Dim(8)), F32), node_id="y")
    graph.inputs, graph.outputs = ["x", "w"], ["y"]
    return graph


def _dependent_live_output_graph() -> Graph:
    """A live quantized value whose closed dependency is itself another live output."""
    m = Axis("m", Dim(4))
    r = Axis("r", Dim(8))
    k = Axis("k", Dim(8))
    loop = LoopOp(
        body=Body(
            (
                Load(name="factor", input="factor", index=(Literal(0), Literal(0))),
                Loop(
                    axis=m,
                    body=Body(
                        (
                            Loop(
                                axis=r,
                                body=Body(
                                    (
                                        Load(name="xr", input="x", index=(Var("m"), Var("r"))),
                                        Accum(name="total", value="xr"),
                                    )
                                ),
                            ),
                            Assign(name="scale", op="multiply", args=("total", "factor"), dtype=F32),
                            Assign(name="inverse", op="reciprocal", args=("scale",), dtype=F32),
                            Assign(name="scale_output", op="multiply", args=("total", "factor"), dtype=F32),
                            Loop(
                                axis=k,
                                body=Body(
                                    (
                                        Load(name="xk", input="x", index=(Var("m"), Var("k"))),
                                        Assign(name="normalized", op="multiply", args=("xk", "inverse"), dtype=F32),
                                        Assign(name="quantized", op="to_f8e4m3", args=("normalized",), dtype=F8E4M3),
                                        Assign(name="decoded", op="from_f8e4m3", args=("quantized",), dtype=F16),
                                        Assign(name="restored", op="multiply", args=("decoded", "scale"), dtype=F32),
                                        Write(output="x_f8", index=(Var("m"), Var("k")), value="quantized"),
                                        Write(output="y", index=(Var("m"), Var("k")), value="restored"),
                                    )
                                ),
                            ),
                            Write(output="x_s", index=(Var("m"), Literal(0)), value="scale_output"),
                        )
                    ),
                ),
            )
        )
    )
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (Dim(4), Dim(8)), F16), node_id="x")
    graph.add_node(InputOp(), [], Tensor("factor", (Dim(1), Dim(1)), F32), node_id="factor")
    graph.add_node(
        loop,
        ["x", "factor"],
        outputs=(
            Tensor("x_s", (Dim(4), Dim(1)), F32),
            Tensor("x_f8", (Dim(4), Dim(8)), F8E4M3),
            Tensor("y", (Dim(4), Dim(8)), F32),
        ),
        node_id="x_s",
    )
    graph.inputs, graph.outputs = ["x", "factor"], ["y", "x_f8", "x_s"]
    return graph


def _apply_value_cut(graph: Graph, output: str | None = None) -> Graph:
    root = next(node for node in graph.nodes.values() if isinstance(node.op, LoopOp))
    sites = value_cut_sites(root.op)
    site = next(site for site in sites if (output is None and not site.outputs) or output in site.outputs)
    match = _match(graph, root.id)
    fragment = realize_value_cut(match, root, site)
    graph.splice(fragment, consumed=match.consumed, output=match.output)
    return graph


def _ops(loop: LoopOp, name: str) -> int:
    return sum(isinstance(stmt, Assign) and stmt.op.name == name for stmt in loop.body.iter())


def test_live_output_cut_moves_the_port_and_loads_it_in_the_parent() -> None:
    graph = _live_output_graph()
    graph.nodes["x_f8"].hints.set("test.provenance", "root")
    graph = _apply_value_cut(graph, "x_f8")
    assert graph.outputs == ["y", "x_f8"]
    assert graph.producer("x_f8") is not graph.producer("y")
    child = graph.producer("x_f8")
    parent = graph.producer("y")
    assert isinstance(child.op, LoopOp) and isinstance(parent.op, LoopOp)
    assert child.output.shape == (Dim(4), Dim(8)) and child.output.dtype == F8E4M3
    assert parent.output.shape == (Dim(4), Dim(8)) and parent.output.dtype == F32
    assert child.hints.get("test.provenance") == parent.hints.get("test.provenance") == "root"
    assert _ops(child.op, "to_f8e4m3") == 1
    assert _ops(parent.op, "to_f8e4m3") == 0
    assert "x_f8" in parent.inputs
    assert any(load.input == "x_f8" for load in parent.op.body.loads)
    graph.validate()


def test_repeated_closed_value_cut_creates_a_typed_workspace() -> None:
    graph = _apply_value_cut(_workspace_graph())
    workspace = next(node for node in graph.nodes.values() if "__cut_" in node.id)
    parent = graph.producer("y")
    assert workspace.output.shape == (Dim(4), Dim(8))
    assert workspace.output.dtype == F8E4M3
    assert workspace.id in parent.inputs
    assert _ops(workspace.op, "to_f8e4m3") == 1
    assert _ops(parent.op, "to_f8e4m3") == 0
    graph.validate()


def test_live_value_cut_transfers_live_outputs_computed_by_its_dependency() -> None:
    graph = _dependent_live_output_graph()
    root = graph.nodes["x_s"]
    root.hints.set("test.provenance", "root")
    site = next(site for site in value_cut_sites(root.op) if "x_f8" in site.value.live_outputs)
    assert site.outputs == ("x_f8", "x_s")

    match = _match(graph, root.id)
    fragment = realize_value_cut(match, root, site)
    graph.splice(fragment, consumed=match.consumed, output=match.output)

    child = graph.producer("x_f8")
    parent = graph.producer("y")
    assert child is graph.producer("x_s")
    assert child is not parent
    assert isinstance(child.op, LoopOp) and isinstance(parent.op, LoopOp)
    assert graph.outputs == ["y", "x_f8", "x_s"]
    assert child.hints.get("test.provenance") == parent.hints.get("test.provenance") == "root"
    assert (graph.buffer("x_s").shape, graph.buffer("x_s").dtype) == ((Dim(4), Dim(1)), F32)
    assert (graph.buffer("x_f8").shape, graph.buffer("x_f8").dtype) == ((Dim(4), Dim(8)), F8E4M3)
    assert (graph.buffer("y").shape, graph.buffer("y").dtype) == ((Dim(4), Dim(8)), F32)
    assert {write.output for write in child.op.body.writes} == {"x_f8", "x_s"}
    assert {write.output for write in parent.op.body.writes} == {"y"}
    assert {load.input for load in parent.op.body.loads} >= {"x_f8", "x_s"}
    assert sum(isinstance(stmt, Accum) for stmt in child.op.body.iter()) == 1
    assert not any(isinstance(stmt, Accum) for stmt in parent.op.body.iter())
    graph.validate()


def test_duplicate_live_writes_fail_closed() -> None:
    graph = _live_output_graph()
    root = graph.nodes["x_f8"]
    write = next(write for write in root.op.body.writes if write.output == "x_f8")
    root.op = LoopOp(body=root.op.body.map(lambda stmt: (stmt, stmt) if stmt is write else stmt))
    assert not any("x_f8" in site.outputs for site in value_cut_sites(root.op))


def test_recognition_offers_live_value_materialization_beside_inline() -> None:
    graph = _live_output_graph()
    captured: list = []

    def inline(fork):
        leaves = flatten_leaves(fork.options)
        captured.extend(leaves)
        return next(option for option in leaves if isinstance(option, TileOp))

    Run(
        pipeline=Pipeline.build(["lowering/tile"], select=["recognize"]),
        ctx=Context.from_target((12, 0)),
    ).resolve(graph, inline)
    materialized = [
        option
        for option in captured
        if isinstance(option, Graph) and any(op.knobs.get("PLACE@=x_f8") == "cut" for op in (node.op for node in option.nodes.values()))
    ]
    assert any(isinstance(option, TileOp) for option in captured)
    assert len(materialized) == 1


def test_value_name_pin_routes_the_exact_materialization() -> None:
    loop = _live_output_graph().nodes["x_f8"].op
    sites = value_cut_sites(loop)
    wanted = next(site for site in sites if "x_f8" in site.outputs)
    assert spell_value_cut(wanted) == "PLACE@=x_f8"
    with pinned_knobs({"PLACE@=x_f8": "cut"}):
        assert route_value_cut(sites) == ("cut", wanted)
