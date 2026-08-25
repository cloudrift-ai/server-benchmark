"""Closed fused-value materialization into ordinary multi-output graph fragments."""

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F8E4M3, F16, F32
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
from emmy.compiler.pipeline import Match, Rule
from emmy.compiler.pipeline.passes.lowering.tile._value_cut import realize_value_cut, value_cut_sites
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
    workspace = next(node for node in graph.nodes.values() if "__materialized_" in node.id and "parent" not in node.id)
    parent = graph.producer("y")
    assert workspace.output.shape == (Dim(4), Dim(8))
    assert workspace.output.dtype == F8E4M3
    assert workspace.id in parent.inputs
    assert _ops(workspace.op, "to_f8e4m3") == 1
    assert _ops(parent.op, "to_f8e4m3") == 0
    graph.validate()


def test_duplicate_live_writes_fail_closed() -> None:
    graph = _live_output_graph()
    root = graph.nodes["x_f8"]
    write = next(write for write in root.op.body.writes if write.output == "x_f8")
    root.op = LoopOp(body=root.op.body.map(lambda stmt: (stmt, stmt) if stmt is write else stmt))
    assert not any("x_f8" in site.outputs for site in value_cut_sites(root.op))
