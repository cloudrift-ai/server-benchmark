"""Maximal loop fusion is one schedule-blind fixpoint."""

import numpy as np

from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import LinearOp
from emmy.compiler.ir.loop import Loop, LoopOp
from emmy.compiler.pipeline import LOOP_PASSES, Pipeline


def _nests_reduce(loop_op: LoopOp) -> bool:
    reduce_names = loop_op.reduce_axis_names

    def walk(body, inside_reduce: bool) -> bool:
        for stmt in body:
            if isinstance(stmt, Loop):
                is_reduce = stmt.axis.name in reduce_names
                if is_reduce and inside_reduce:
                    return True
                if walk(stmt.body, inside_reduce or is_reduce):
                    return True
        return False

    return walk(loop_op.body, False)


def _chained_matmuls(m=8, k0=4, k1=6, n=5) -> Graph:
    """``(x @ w0) @ w1`` — the smallest graph with nested contractions after fusion."""
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (m, k0)), node_id="x")
    graph.add_node(InputOp(), [], Tensor("w0", (k1, k0)), node_id="w0")
    graph.add_node(InputOp(), [], Tensor("w1", (n, k1)), node_id="w1")
    graph.add_node(LinearOp(), ["x", "w0"], Tensor("h", (m, k1)), node_id="h")
    graph.add_node(LinearOp(), ["h", "w1"], Tensor("y", (m, n)), node_id="y")
    graph.inputs, graph.outputs = ["x", "w0", "w1"], ["y"]
    return graph


def test_chained_matmuls_fuse_into_one_nested_reduction() -> None:
    result = Pipeline.build(LOOP_PASSES).run(_chained_matmuls())
    kernels = [node for node in result.nodes.values() if isinstance(node.op, LoopOp)]
    assert [node.id for node in kernels] == ["y"]
    assert _nests_reduce(kernels[0].op)


def test_nested_reduction_fusion_preserves_numerics() -> None:
    from emmy.compiler.backend.numpy import NumpyBackend

    rng = np.random.default_rng(0)
    inputs = {
        "x": rng.standard_normal((8, 4)).astype(np.float32),
        "w0": rng.standard_normal((6, 4)).astype(np.float32),
        "w1": rng.standard_normal((5, 6)).astype(np.float32),
    }
    backend = NumpyBackend()
    fused = Pipeline.build(LOOP_PASSES).run(_chained_matmuls())
    got = next(iter(backend.run(backend.compile(fused), input_data=inputs)[0].outputs.values()))
    want = (inputs["x"] @ inputs["w0"].T) @ inputs["w1"].T
    np.testing.assert_allclose(got.reshape(want.shape), want, rtol=1e-5, atol=1e-5)


def test_an_unfusable_chain_does_not_shatter_the_rest_of_the_region():
    """The maximal region contains a recurrence-shaped chain no budget can construct. Fusion
    must decline the CHAIN — dropping the splicer-named origin and its downstream closure — and
    still merge everything else, rather than abandoning the whole region. Abandoning it is what
    shattered DeepSeek-V4's post block into 433 kernels where pre-maximal fusion produced 92.
    """
    from importlib import import_module

    from emmy.compiler.ir.expr import Var
    from emmy.compiler.ir.loop import Assign, Axis, Load, Write
    from emmy.compiler.pipeline import Match, Rule
    from tests.compiler.ir.loop.test_splicer import _affine_recurrence_chain

    fusion = import_module("emmy.compiler.pipeline.passes.loop.fusion.010_merge_loop_ops")

    axis = Axis("i", 8)
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("b0", (9,)), node_id="b0")
    # The shared producer both feeds the doomed chain and a plain elementwise consumer.
    producer = LoopOp(
        body=(
            Loop(
                axis=axis,
                body=(
                    Load(name="p", input="b0", index=(Var("i"),)),
                    Assign(name="pv", op="relu", args=("p",)),
                    Write(output="root", index=(Var("i"),), value="pv"),
                ),
            ),
        ),
    )
    graph.add_node(producer, ["b0"], Tensor("root", (8,)), node_id="root")
    chain_loops, _edges, _roots = _affine_recurrence_chain(12)
    upstream = "root"
    for tag, op in chain_loops.items():
        rebound = op.rename_buffers({"b0": upstream}) if upstream != "b0" else op
        graph.add_node(rebound, [upstream], Tensor(tag, (8,)), node_id=tag)
        upstream = tag
    sibling = LoopOp(
        body=(
            Loop(
                axis=axis,
                body=(
                    Load(name="q", input="root", index=(Var("i"),)),
                    Assign(name="qv", op="negative", args=("q",)),
                    Write(output="easy", index=(Var("i"),), value="qv"),
                ),
            ),
        ),
    )
    graph.add_node(sibling, ["root"], Tensor("easy", (8,)), node_id="easy")
    graph.inputs, graph.outputs = ["b0"], [upstream, "easy"]

    match = Match(graph=graph, root_node_id="root", rule=Rule(name="test", pattern=[]))
    fragment = fusion.rewrite(match, graph.nodes["root"])

    assert fragment is not None, "the fusable half of the region must still merge"
    fused = set(match.consumed)
    assert {"root", "easy"} <= fused, "the plain sibling consumer fused with the producer"
    assert not any(nid.startswith("s") and nid[1:].isdigit() for nid in fused), "no stage of the doomed chain was pulled into the merge"
