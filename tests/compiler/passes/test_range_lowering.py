"""Static ranges lower through the generic CUDA boundary and remain plan-reproducible."""

from emmy.compiler.backend.plan import plan_from_graph
from emmy.compiler.context import Context
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.ir.tensor.ir import BitcastOp, CastOp, RangeOp
from emmy.compiler.pipeline import CUDA_PASSES, Pipeline


def test_static_integer_range_lowers_to_a_plannable_cuda_fill():
    graph = Graph()
    graph.add_node(RangeOp(start=9, stop=-4, step=-3, dtype="i32"), [], Tensor("r", (5,), "i32"), node_id="r")
    graph.outputs = ["r"]

    lowered = Pipeline.build(["lowering/cuda"]).run(graph, ctx=Context(compute_capability=(12, 0)))
    assert all(isinstance(node.op, (InputOp, ConstantOp, CudaOp)) for node in lowered.nodes.values())
    op = lowered.nodes["r"].op
    assert isinstance(op, CudaOp)
    assert "long long value = 9LL + (long long)i * -3LL" in op.kernel_source
    assert "static_cast<int>(value)" in op.kernel_source

    plan = plan_from_graph(lowered)
    assert len(plan.launches) == 1 and plan.launches[0].node_id == "r"
    assert not plan.weights


def test_static_runtime_cast_and_bitcast_are_plannable_cuda_copies():
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("words", (8,), "i16"), node_id="words")
    graph.add_node(BitcastOp(dtype="u16"), ["words"], Tensor("bits", (8,), "u16"), node_id="bits")
    graph.add_node(CastOp(dtype="u64"), ["bits"], Tensor("wide", (8,), "u64"), node_id="wide")
    graph.inputs, graph.outputs = ["words"], ["wide"]

    lowered = Pipeline.build(CUDA_PASSES).run(graph, ctx=Context(compute_capability=(12, 0)))
    assert all(isinstance(node.op, (InputOp, ConstantOp, CudaOp)) for node in lowered.nodes.values())
    source = "\n".join(node.op.kernel_source for node in lowered.nodes.values() if isinstance(node.op, CudaOp))
    assert "emmy_bitcast<unsigned short>(in0)" in source
    assert "unsigned long long v1 = v0" in source
    assert len(plan_from_graph(lowered).launches) == 1
