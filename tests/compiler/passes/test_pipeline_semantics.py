"""Tests for the compile pipeline: decomposition → optimization → fusion → extract.

After the pipeline, every primitive op is inside a LoopOp. Reductions
live as ``Accum + Accum`` on the LoopOp; elementwise ops as
``Assign``; final output as a ``Write``.

Structural fixtures also have a matching ``*_correctness`` test that runs
the pre-pipeline graph and the post-pipeline graph through ``NumpyBackend``
(now that ``LoopOp.forward`` can execute fused kernels) and compares the
outputs — this validates the full decomposition+optimization+fusion chain
preserves semantics without needing a GPU.
"""

import numpy as np
import pytest

from emmy.compiler.backend.numpy import NumpyBackend
from emmy.compiler.context import Context
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.ir.loop import Accum, Assign, LoopOp, Write
from emmy.compiler.ir.tensor.ir import ElementwiseOp, ReduceOp, ScanOp
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import CUDA_PASSES, LOOP_PASSES, TILE_PASSES, Pipeline

_backend = NumpyBackend()
rng = np.random.default_rng(0)


def _compile(graph: Graph) -> Graph:
    return Pipeline.build(LOOP_PASSES).run(graph)


def _fully_rewrite(graph: Graph) -> Graph:
    """Apply the full pass chain (decomposition → optimization → fusion)."""
    return _compile(graph)


def _run(graph: Graph, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return _backend.run(_backend.compile(graph), input_data=inputs)[0].outputs


def _assert_pipeline_preserves_semantics(make_graph, inputs, *, rtol=1e-5, atol=1e-5):
    """Numpy-execute the original graph and the rewritten graph; outputs must match."""
    before = _run(make_graph(), inputs)
    after = _run(_fully_rewrite(make_graph()), inputs)
    bvals, avals = list(before.values()), list(after.values())
    assert len(bvals) == len(avals)
    for i, (b, a) in enumerate(zip(bvals, avals, strict=True)):
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=f"output[{i}]")


def _input(g: Graph, name: str, shape: tuple) -> str:
    return g.add_node(op=InputOp(), inputs=[], output=Tensor(name, shape), node_id=name)


def _elementwise_fns(body) -> list[str]:
    return [s.op.name for s in body.iter() if isinstance(s, Assign)]


def _has_update(body) -> bool:
    return any(isinstance(s, Accum) for s in body.iter())


def _has_write(body) -> bool:
    return any(isinstance(s, Write) for s in body.iter())


def _loop_nodes(graph: Graph) -> list:
    return [n for n in graph.nodes.values() if isinstance(n.op, LoopOp)]


def test_pointwise_add():
    g = Graph()
    _input(g, "x", (4,))
    _input(g, "y", (4,))
    g.add_node(op=ElementwiseOp(op="add"), inputs=["x", "y"], output=Tensor("z", (4,)), node_id="z")
    g.inputs = ["x", "y"]
    g.outputs = ["z"]

    result = _compile(g)
    launches = _loop_nodes(result)
    assert len(launches) == 1
    assert "add" in _elementwise_fns(launches[0].op.body)
    assert _has_write(launches[0].op.body)


def test_chained_pointwise_fuses_into_one():
    g = Graph()
    _input(g, "x", (4,))
    g.add_node(op=ElementwiseOp("exp"), inputs=["x"], output=Tensor("e", (4,)), node_id="e")
    g.add_node(op=ElementwiseOp("negative"), inputs=["e"], output=Tensor("n", (4,)), node_id="n")
    g.inputs = ["x"]
    g.outputs = ["n"]

    result = _compile(g)
    launches = _loop_nodes(result)
    assert len(launches) == 1
    fns = _elementwise_fns(launches[0].op.body)
    assert "exp" in fns and "negative" in fns


def test_reduce_sum():
    g = Graph()
    _input(g, "x", (4, 8))
    g.add_node(op=ReduceOp(op="sum", axis=-1), inputs=["x"], output=Tensor("r", (4, 1)), node_id="r")
    g.inputs = ["x"]
    g.outputs = ["r"]

    result = _compile(g)
    launches = _loop_nodes(result)
    assert len(launches) == 1
    loop = launches[0].op
    assert _has_update(loop.body)
    assert any(lb.op.name == "add" for lb in loop.body.accums)


def test_scan_sum_lifts_and_preserves_prefix_values():
    def make_graph():
        graph = Graph()
        _input(graph, "x", (2, 4))
        graph.add_node(op=ScanOp(op="sum", axis=-1), inputs=["x"], output=Tensor("out", (2, 4)), node_id="out")
        graph.inputs = ["x"]
        graph.outputs = ["out"]
        return graph

    result = _compile(make_graph())
    launches = _loop_nodes(result)
    assert len(launches) == 1
    assert any(accum.op.name == "add" for accum in launches[0].op.body.accums)
    _assert_pipeline_preserves_semantics(make_graph, {"x": np.arange(8, dtype=np.float32).reshape(2, 4)}, rtol=0, atol=0)

    lowered = Pipeline.build(CUDA_PASSES).run(make_graph(), ctx=Context.from_target((8, 9)))
    cuda_ops = [node.op for node in lowered.nodes.values() if isinstance(node.op, CudaOp)]
    assert len(cuda_ops) == 1
    source = cuda_ops[0].kernel_source
    assert source.index("+=") < source.index("out[")


def test_scan_after_pointwise_keeps_the_write_inside_its_reduce_loop():
    """Fusion must not rebuild an ordered prefix as one full reduce plus an output sweep."""

    def make_graph():
        graph = Graph()
        _input(graph, "x", (2, 4))
        graph.add_node(op=ElementwiseOp("negative"), inputs=["x"], output=Tensor("neg", (2, 4)), node_id="neg")
        graph.add_node(op=ScanOp(op="sum", axis=-1), inputs=["neg"], output=Tensor("out", (2, 4)), node_id="out")
        graph.inputs = ["x"]
        graph.outputs = ["out"]
        return graph

    result = _compile(make_graph())
    launches = _loop_nodes(result)
    assert len(launches) == 2, "the pointwise producer stays materialized across the ordered scan boundary"
    scan = next(node.op for node in launches if node.id == "out")
    reduce_loop = next(loop for loop in scan.body.iter() if getattr(loop, "axis", None) and loop.axis.name in scan.reduce_axis_names)
    assert any(isinstance(stmt, Accum) for stmt in reduce_loop.body)
    assert any(isinstance(stmt, Write) for stmt in reduce_loop.body)
    _assert_pipeline_preserves_semantics(make_graph, {"x": np.arange(8, dtype=np.float32).reshape(2, 4)}, rtol=0, atol=0)

    tiled = Pipeline.build(TILE_PASSES).run(make_graph(), ctx=Context.from_target((8, 9)))
    scan_tile = next(node.op for node in tiled.nodes.values() if isinstance(node.op, TileOp) and node.id == "out")
    assert scan_tile.schedule == {}
    assert scan_tile.knobs["REDUCE"] == "" and scan_tile.knobs["WORK"] == ""

    from emmy.compiler.pipeline.search.space import REDUCE, WORK

    with WORK.pinned("t4"), REDUCE.pinned("coop"):
        pinned = Pipeline.build(TILE_PASSES).run(make_graph(), ctx=Context.from_target((8, 9)))
    pinned_scan = next(node.op for node in pinned.nodes.values() if isinstance(node.op, TileOp) and node.id == "out")
    assert pinned_scan.schedule == {}
    assert pinned_scan.knobs["REDUCE"] == "" and pinned_scan.knobs["WORK"] == ""

    lowered = Pipeline.build(CUDA_PASSES).run(make_graph(), ctx=Context.from_target((8, 9)))
    source = next(
        node.op.kernel_source for node in lowered.nodes.values() if isinstance(node.op, CudaOp) and "out[" in node.op.kernel_source
    )
    lines = source.splitlines()
    update = next(i for i, line in enumerate(lines) if "acc0 +=" in line)
    # The stored value is the observer's fresh name (``acc0__obs``), never the raw accumulator —
    # the boundary distinguishes a streamed store from a post-fold store by exactly that name.
    write = next(i for i, line in enumerate(lines) if "out[a0 * 4 + a1] = acc0__obs;" in line)
    loop_open = max(i for i in range(update) if lines[i].lstrip().startswith("for ("))
    loop_indent = len(lines[loop_open]) - len(lines[loop_open].lstrip())
    loop_close = next(
        i for i in range(update + 1, len(lines)) if lines[i].strip() == "}" and len(lines[i]) - len(lines[i].lstrip()) == loop_indent
    )
    assert update < write < loop_close, "the prefix store must execute after each accumulator update"


def test_matmul():
    from emmy.compiler.ir.frontend.ir import MatmulOp

    g = Graph()
    _input(g, "a", (4, 8))
    _input(g, "b", (8, 4))
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (4, 4)), node_id="o")
    g.inputs = ["a", "b"]
    g.outputs = ["o"]

    result = _compile(g)
    launches = _loop_nodes(result)
    has_mul = any("multiply" in _elementwise_fns(k.op.body) for k in launches)
    has_sum = any(any(lb.op.name == "add" for lb in k.op.body.accums) for k in launches)
    assert has_mul
    assert has_sum


def test_mul_fanout_fuses_into_one_multi_output_loop():
    g = Graph()
    _input(g, "a", (4, 8))
    _input(g, "b", (4, 8))
    g.add_node(op=ElementwiseOp("multiply"), inputs=["a", "b"], output=Tensor("m", (4, 8)), node_id="m")
    g.add_node(op=ReduceOp(op="sum", axis=-1), inputs=["m"], output=Tensor("d", (4, 1)), node_id="d")
    g.add_node(op=ElementwiseOp("negative"), inputs=["m"], output=Tensor("n", (4, 8)), node_id="n")
    g.inputs = ["a", "b"]
    g.outputs = ["d", "n"]

    result = _compile(g)
    launches = _loop_nodes(result)
    assert len(launches) == 1
    assert {output.name for output in launches[0].outputs} == {"d", "n"}
    assert _has_update(launches[0].op.body)
    assert "negative" in _elementwise_fns(launches[0].op.body)


def test_matmul_op_decomposes_and_fuses():
    from emmy.compiler.ir.frontend.ir import MatmulOp

    g = Graph()
    _input(g, "a", (4, 8))
    _input(g, "b", (8, 4))
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("m", (4, 4)), node_id="m")
    g.inputs = ["a", "b"]
    g.outputs = ["m"]

    result = _compile(g)
    launches = _loop_nodes(result)
    has_reduce = any(_has_update(k.op.body) for k in launches)
    assert has_reduce


def test_compile_produces_kernel_ops():
    g = Graph()
    _input(g, "x", (4,))
    g.add_node(op=ElementwiseOp("exp"), inputs=["x"], output=Tensor("e", (4,)), node_id="e")
    g.inputs = ["x"]
    g.outputs = ["e"]

    result = _compile(g)
    launches = _loop_nodes(result)
    assert len(launches) == 1


# ===================================================================
# Correctness: full rewriter (decomp + opt + fusion) preserves semantics.
# ===================================================================


def test_pointwise_add_correctness():
    def _make():
        g = Graph()
        _input(g, "x", (4,))
        _input(g, "y", (4,))
        g.add_node(op=ElementwiseOp("add"), inputs=["x", "y"], output=Tensor("z", (4,)), node_id="z")
        g.inputs, g.outputs = ["x", "y"], ["z"]
        return g

    x = rng.standard_normal(4).astype(np.float32)
    y = rng.standard_normal(4).astype(np.float32)
    _assert_pipeline_preserves_semantics(_make, {"x": x, "y": y})


def test_chained_pointwise_correctness():
    def _make():
        g = Graph()
        _input(g, "x", (4,))
        g.add_node(op=ElementwiseOp("exp"), inputs=["x"], output=Tensor("e", (4,)), node_id="e")
        g.add_node(op=ElementwiseOp("negative"), inputs=["e"], output=Tensor("n", (4,)), node_id="n")
        g.inputs, g.outputs = ["x"], ["n"]
        return g

    x = rng.standard_normal(4).astype(np.float32)
    _assert_pipeline_preserves_semantics(_make, {"x": x})


def test_reduce_sum_correctness():
    def _make():
        g = Graph()
        _input(g, "x", (4, 8))
        g.add_node(op=ReduceOp(op="sum", axis=-1), inputs=["x"], output=Tensor("r", (4, 1)), node_id="r")
        g.inputs, g.outputs = ["x"], ["r"]
        return g

    x = rng.standard_normal((4, 8)).astype(np.float32)
    _assert_pipeline_preserves_semantics(_make, {"x": x})


def test_matmul_correctness():
    from emmy.compiler.ir.frontend.ir import MatmulOp

    def _make():
        g = Graph()
        _input(g, "a", (4, 8))
        _input(g, "b", (8, 4))
        g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (4, 4)), node_id="o")
        g.inputs, g.outputs = ["a", "b"], ["o"]
        return g

    a = rng.standard_normal((4, 8)).astype(np.float32)
    b = rng.standard_normal((8, 4)).astype(np.float32)
    _assert_pipeline_preserves_semantics(_make, {"a": a, "b": b}, rtol=1e-4)


def test_mul_fan_out_correctness():
    def _make():
        g = Graph()
        _input(g, "a", (4, 8))
        _input(g, "b", (4, 8))
        g.add_node(op=ElementwiseOp("multiply"), inputs=["a", "b"], output=Tensor("m", (4, 8)), node_id="m")
        g.add_node(op=ReduceOp(op="sum", axis=-1), inputs=["m"], output=Tensor("d", (4, 1)), node_id="d")
        g.add_node(op=ElementwiseOp("negative"), inputs=["m"], output=Tensor("n", (4, 8)), node_id="n")
        g.inputs, g.outputs = ["a", "b"], ["d", "n"]
        return g

    a = rng.standard_normal((4, 8)).astype(np.float32)
    b = rng.standard_normal((4, 8)).astype(np.float32)
    _assert_pipeline_preserves_semantics(_make, {"a": a, "b": b})
