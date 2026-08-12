"""Tests for the fusion pass (lift-then-splice).

The fusion pass lifts each tensor op into a trivial ``LoopOp`` and then
splices adjacent pairs and closed reconvergent ``LoopOp`` DAGs through
the SSA-preserving N-way splicer. Tests verify post-fixpoint structural
properties (kernel count, graph composition, expected ops in SSA bodies)
*and* numeric correctness — each fixture is executed via ``NumpyBackend``
both pre- and post-fusion, and the outputs must match. ``LoopOp.forward``
makes the post-fusion run possible without a GPU.
"""

import numpy as np
import pytest

from emmy.compiler.backend.numpy import NumpyBackend
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.expr import Literal, placeholder
from emmy.compiler.ir.frontend.ir import LinearOp, RmsNormOp
from emmy.compiler.ir.loop import Accum, Assign, Load, LoopOp, Write
from emmy.compiler.ir.tensor.ir import ElementwiseOp, GatherOp, IndexMapOp, IndexSource, ReduceOp
from emmy.compiler.pipeline import Pipeline

rng = np.random.default_rng(0)
_backend = NumpyBackend()


def _fuse(graph: Graph) -> Graph:
    return Pipeline.build(["loop/lifting", "loop/fusion"]).run(graph)


def _run(graph: Graph, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return _backend.run(_backend.compile(graph), input_data=inputs)[0].outputs


def _assert_close(before: dict, after: dict, *, rtol=1e-5, atol=1e-5):
    bvals = list(before.values())
    avals = list(after.values())
    assert len(bvals) == len(avals), f"output count mismatch: {len(bvals)} vs {len(avals)}"
    for i, (b, a) in enumerate(zip(bvals, avals, strict=True)):
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=f"output[{i}]")


def _assert_correctness(make_graph, inputs):
    """Run the pre- and post-fusion graph through NumpyBackend; assert outputs match.

    Exercises fusion rules for *semantic* equivalence on top of the
    structural checks in this file — ``LoopOp.forward`` executes the
    lifted+merged kernels numerically against the original tensor-IR
    evaluation.
    """
    g_before = make_graph()
    g_after = _fuse(make_graph())
    before = _run(g_before, inputs)
    after = _run(g_after, inputs)
    _assert_close(before, after)


def _kernel_nodes(graph: Graph) -> list:
    return [n for n in graph.nodes.values() if isinstance(n.op, LoopOp)]


def _assign_fns(body) -> list[str]:
    return [s.op.name for s in body.iter() if isinstance(s, Assign)]


def _count_copies(body) -> int:
    """Count identity ``Assign(op=copy)`` statements in a LoopOp body."""
    return sum(1 for s in body.iter() if isinstance(s, Assign) and s.op.name == "copy")


def _has_update(body) -> bool:
    return any(isinstance(s, Accum) for s in body.iter())


def _local_combine_fns(locals_) -> set[str]:
    return {lb.op.name for lb in locals_ if lb.op is not None}


# ===================================================================
# Pointwise chain: neg → exp → single kernel
# ===================================================================


def _make_pointwise_chain():
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (8,)), node_id="x")
    g.add_node(ElementwiseOp("negative"), ["x"], Tensor("n", (8,)), node_id="n")
    g.add_node(ElementwiseOp("exp"), ["n"], Tensor("y", (8,)), node_id="y")
    g.inputs, g.outputs = ["x"], ["y"]
    return g


def test_pointwise_chain_fuses_to_one_kernel():
    result = _fuse(_make_pointwise_chain())
    kernels = _kernel_nodes(result)
    assert len(kernels) == 1


def test_pointwise_chain_only_kernel_input_constant():
    result = _fuse(_make_pointwise_chain())
    for n in result.nodes.values():
        assert isinstance(n.op, (LoopOp, InputOp, ConstantOp))


def test_pointwise_chain_body_ops():
    result = _fuse(_make_pointwise_chain())
    kernel = _kernel_nodes(result)[0]
    body_ops = _assign_fns(kernel.op.body)
    assert "negative" in body_ops
    assert "exp" in body_ops


def test_pointwise_chain_inputs_are_loads():
    from emmy.compiler.ir.loop import Load

    result = _fuse(_make_pointwise_chain())
    kernel = _kernel_nodes(result)[0]
    loads = kernel.op.body.loads
    assert len(loads) >= 1
    assert all(isinstance(ld, Load) for ld in loads)


def test_pointwise_chain_has_write():
    result = _fuse(_make_pointwise_chain())
    kernel = _kernel_nodes(result)[0]
    assert any(isinstance(s, Write) for s in kernel.op)


def test_pointwise_chain_no_residual_copies():
    """After fusion + copy-elimination, no identity ``copy`` Assigns remain."""
    result = _fuse(_make_pointwise_chain())
    kernel = _kernel_nodes(result)[0]
    assert _count_copies(kernel.op.body) == 0


# ===================================================================
# Reconvergent producer DAG: one SSA definition, multiple consumers.
# ===================================================================


def _make_pointwise_diamond():
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (8,)), node_id="x")
    g.add_node(InputOp(), [], Tensor("a", (8,)), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (8,)), node_id="b")
    g.add_node(ElementwiseOp("negative"), ["x"], Tensor("shared", (8,)), node_id="shared")
    g.add_node(ElementwiseOp("add"), ["shared", "a"], Tensor("left", (8,)), node_id="left")
    g.add_node(ElementwiseOp("multiply"), ["shared", "b"], Tensor("right", (8,)), node_id="right")
    g.add_node(ElementwiseOp("add"), ["left", "right"], Tensor("out", (8,)), node_id="out")
    g.inputs, g.outputs = ["x", "a", "b"], ["out"]
    return g


def test_reconvergent_dag_fuses_as_one_ssa_region():
    result = _fuse(_make_pointwise_diamond())
    kernels = _kernel_nodes(result)
    assert len(kernels) == 1
    fns = _assign_fns(kernels[0].op.body)
    assert fns.count("negative") == 1
    assert fns.count("add") == 2
    assert fns.count("multiply") == 1


def test_reconvergent_dag_fusion_is_numerically_exact():
    inputs = {name: rng.standard_normal(8).astype(np.float32) for name in ("x", "a", "b")}
    before = _run(_make_pointwise_diamond(), inputs)
    after = _run(_fuse(_make_pointwise_diamond()), inputs)
    _assert_close(before, after)


def test_non_reconvergent_fanout_stays_materialized():
    graph = _make_pointwise_diamond()
    graph.add_node(ElementwiseOp("exp"), ["shared"], Tensor("escape", (8,)), node_id="escape")
    graph.outputs.append("escape")
    result = _fuse(graph)
    shared = result.nodes.get("shared")
    assert shared is not None and isinstance(shared.op, LoopOp)
    assert len(result.outputs) == 2


def _make_indexmap_diamond():
    from emmy.compiler.pipeline.passes.frontend.decomposition._helpers import const_bc

    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (2, 4)), node_id="x")
    shared = const_bc(g, name="one", value=1.0, target_shape=(2, 4), dtype="f32")
    g.add_node(ElementwiseOp("add"), ["x", shared], Tensor("left", (2, 4)), node_id="left")
    g.add_node(ElementwiseOp("multiply"), ["x", shared], Tensor("right", (2, 4)), node_id="right")
    g.add_node(ElementwiseOp("add"), ["left", "right"], Tensor("out", (2, 4)), node_id="out")
    g.inputs, g.outputs = ["x"], ["out"]
    return g


def test_reconvergent_indexmap_is_owned_by_shared_region_merge():
    lifted = Pipeline.build(["loop/lifting"]).run(_make_indexmap_diamond())
    split_only = Pipeline.build(["loop/fusion"], select={"split_shared_indexmap"}).run(lifted)
    assert "one_bc" in split_only.nodes

    result = Pipeline.build(["loop/fusion"]).run(split_only)
    (kernel,) = _kernel_nodes(result)
    assert sum(isinstance(stmt, Load) and stmt.input == "one" for stmt in kernel.op.body.iter()) == 1


def test_typed_copy_fanout_fuses_after_contraction_halves_join():
    """A fan-out retains its precision copy inside the complete contraction."""
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (2, 4), "f32"), node_id="x")
    g.add_node(ElementwiseOp("copy"), ["x"], Tensor("narrow", (2, 4), "f16"), node_id="narrow")
    g.add_node(ElementwiseOp("negative"), ["narrow"], Tensor("left", (2, 4), "f16"), node_id="left")
    g.add_node(ElementwiseOp("multiply"), ["narrow", "left"], Tensor("product", (2, 4), "f16"), node_id="product")
    g.add_node(ReduceOp("sum", -1), ["product"], Tensor("out", (2, 1), "f16"), node_id="out")
    g.inputs, g.outputs = ["x"], ["out"]

    result = _fuse(g)
    (kernel,) = _kernel_nodes(result)
    assert "copy" in _assign_fns(kernel.op.body)
    assert any(isinstance(stmt, Accum) for stmt in kernel.op.body.iter())


# ===================================================================
# Expensive pointwise producer → contraction
# ===================================================================


def _make_activation_linear(activation: str):
    """``activation(x[M,K]) @ w[N,K].T`` with enough N to expose duplication."""
    m, k, n = 2, 16, 16
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (m, k)), node_id="x")
    g.add_node(InputOp(), [], Tensor("w", (n, k)), node_id="w")
    g.add_node(ElementwiseOp(activation), ["x"], Tensor("activated", (m, k)), node_id="activated")
    g.add_node(LinearOp(), ["activated", "w"], Tensor("out", (m, n)), node_id="out")
    g.inputs, g.outputs = ["x", "w"], ["out"]
    return g


def _decompose_and_fuse(graph: Graph) -> Graph:
    return Pipeline.build(["frontend/decomposition", "frontend/optimization", "loop/lifting", "loop/fusion"]).run(graph)


@pytest.mark.xfail(
    strict=True,
    reason="gate-free merge-loop experiment duplicates the transcendental across contraction columns",
)
def test_transcendental_is_not_duplicated_across_contraction_columns():
    """Materialize exp once per (M,K), instead of recomputing it N times."""
    result = _decompose_and_fuse(_make_activation_linear("exp"))
    kernels = _kernel_nodes(result)
    assert len(kernels) == 2
    exp_kernels = [node for node in kernels if "exp" in _assign_fns(node.op.body)]
    assert len(exp_kernels) == 1
    assert exp_kernels[0].output.shape == (2, 16)


def test_cheap_activation_still_fuses_into_contraction():
    """The narrow transcendental guard does not become a blanket fusion barrier."""
    result = _decompose_and_fuse(_make_activation_linear("negative"))
    assert len(_kernel_nodes(result)) == 1


def test_multisource_gated_activation_still_fuses_into_contraction():
    """Gated activations remain evidence-controlled serving kernels."""
    g = _make_activation_linear("exp")
    g.add_node(InputOp(), [], Tensor("gate", (2, 16)), node_id="gate")
    g.add_node(ElementwiseOp("multiply"), ["activated", "gate"], Tensor("gated", (2, 16)), node_id="gated")
    g.nodes["out"].inputs[0] = "gated"
    g.inputs.append("gate")

    result = _decompose_and_fuse(g)
    assert len(_kernel_nodes(result)) == 1


def test_packed_gated_activation_still_fuses_into_contraction():
    """SSA fusion must retain the two lanes of a packed gate/up projection."""
    m, k, n = 2, 16, 16
    g = Graph()
    g.add_node(InputOp(), [], Tensor("packed", (m, 2 * k)), node_id="packed")
    g.add_node(InputOp(), [], Tensor("w", (n, k)), node_id="w")
    g.add_node(
        IndexMapOp(out_shape=(m, k), sources=(IndexSource(input_idx=0, coord_map=(placeholder(0), placeholder(1))),)),
        ["packed"],
        Tensor("gate", (m, k)),
        node_id="gate",
    )
    g.add_node(
        IndexMapOp(
            out_shape=(m, k),
            sources=(IndexSource(input_idx=0, coord_map=(placeholder(0), placeholder(1) + Literal(k, "int"))),),
        ),
        ["packed"],
        Tensor("up", (m, k)),
        node_id="up",
    )
    g.add_node(ElementwiseOp("tanh"), ["gate"], Tensor("activated", (m, k)), node_id="activated")
    g.add_node(ElementwiseOp("multiply"), ["activated", "up"], Tensor("gated", (m, k)), node_id="gated")
    g.add_node(LinearOp(), ["gated", "w"], Tensor("out", (m, n)), node_id="out")
    g.inputs, g.outputs = ["packed", "w"], ["out"]

    inputs = {
        "packed": rng.standard_normal((m, 2 * k)).astype(np.float32),
        "w": rng.standard_normal((n, k)).astype(np.float32),
    }
    before = _run(g, inputs)
    result = _decompose_and_fuse(g)
    assert len(_kernel_nodes(result)) == 1
    _assert_close(before, _run(result, inputs))


# ===================================================================
# Copy elimination: transitive alias chain + port-ref rewriting
# ===================================================================


def _make_rms_norm_like():
    """A 4-op fused graph (mul → reduce_sum → div → mul-by-weight) that
    produces many bridge copies during merge — the realistic target of the
    copy-elimination pass."""
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 8)), node_id="x")
    g.add_node(ConstantOp(name="w"), [], Tensor("w", (4, 8)), node_id="w")
    g.add_node(ElementwiseOp("multiply"), ["x", "x"], Tensor("sq", (4, 8)), node_id="sq")
    g.add_node(ReduceOp("sum", -1), ["sq"], Tensor("red", (4, 1)), node_id="red")
    g.add_node(ElementwiseOp("multiply"), ["red", "red"], Tensor("sqr", (4, 1)), node_id="sqr")
    g.inputs, g.outputs = ["x", "w"], ["sqr"]
    return g


def test_rms_norm_like_no_residual_copies():
    result = _fuse(_make_rms_norm_like())
    kernel = _kernel_nodes(result)[0]
    assert _count_copies(kernel.op.body) == 0, "copy-elimination must clear all bridge copies"


def test_rms_norm_like_correctness():
    x = rng.standard_normal((4, 8)).astype(np.float32)
    w = rng.standard_normal((4, 8)).astype(np.float32)
    _assert_correctness(_make_rms_norm_like, {"x": x, "w": w})


def test_rms_norm_like_ssa_names_are_canonical():
    """After rename pass, every SSA name in the body is v0, v1, v2, ... in order."""
    from emmy.compiler.ir.loop import Select

    result = _fuse(_make_rms_norm_like())
    kernel = _kernel_nodes(result)[0]
    ssa_names = [s.name for s in kernel.op if isinstance(s, (Assign, Select))]
    assert ssa_names == [f"v{i}" for i in range(len(ssa_names))], f"unexpected SSA names: {ssa_names}"


# ===================================================================
# Reduce chain: mul → reduce_sum (contraction)
# ===================================================================


def _make_contraction():
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (4, 8)), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (4, 8)), node_id="b")
    g.add_node(ElementwiseOp("multiply"), ["a", "b"], Tensor("m", (4, 8)), node_id="m")
    g.add_node(ReduceOp("sum", -1), ["m"], Tensor("y", (4, 1)), node_id="y")
    g.inputs, g.outputs = ["a", "b"], ["y"]
    return g


def test_contraction_fuses_to_one_kernel():
    result = _fuse(_make_contraction())
    kernels = _kernel_nodes(result)
    assert len(kernels) == 1


def test_contraction_body_has_mul_and_sum():
    result = _fuse(_make_contraction())
    kernel = _kernel_nodes(result)[0]
    assert "multiply" in _assign_fns(kernel.op.body)
    assert _has_update(kernel.op.body)
    assert "add" in _local_combine_fns(kernel.op.body.accums)


# ===================================================================
# bilinear fold + epilogue: mul → sum → add(bias)
# ===================================================================


def _make_contraction_with_epilogue():
    from emmy.compiler.pipeline.passes.frontend.decomposition._broadcast import broadcast_to

    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (4, 8)), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (4, 8)), node_id="b")
    g.add_node(InputOp(), [], Tensor("bias", (4, 1)), node_id="bias")
    g.add_node(ElementwiseOp("multiply"), ["a", "b"], Tensor("m", (4, 8)), node_id="m")
    g.add_node(ReduceOp("sum", -1), ["m"], Tensor("s", (4, 1)), node_id="s")
    g.add_node(ElementwiseOp("add"), ["s", broadcast_to(g, "bias", (4, 1))], Tensor("y", (4, 1)), node_id="y")
    g.inputs, g.outputs = ["a", "b", "bias"], ["y"]
    return g


def test_contraction_epilogue_fuses_to_one_kernel():
    result = _fuse(_make_contraction_with_epilogue())
    kernels = _kernel_nodes(result)
    assert len(kernels) == 1


def test_contraction_epilogue_body_has_add():
    result = _fuse(_make_contraction_with_epilogue())
    kernel = _kernel_nodes(result)[0]
    assert "add" in _assign_fns(kernel.op.body)


# ===================================================================
# Softmax: reduce_max → sub → exp → reduce_sum → div
# ===================================================================


def _make_softmax():
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 8)), node_id="x")
    g.add_node(ReduceOp("maximum", -1), ["x"], Tensor("mx", (4, 1)), node_id="mx")
    g.add_node(ElementwiseOp("subtract"), ["x", "mx"], Tensor("subtract", (4, 8)), node_id="subtract")
    g.add_node(ElementwiseOp("exp"), ["subtract"], Tensor("exp", (4, 8)), node_id="exp")
    g.add_node(ReduceOp("sum", -1), ["exp"], Tensor("sm", (4, 1)), node_id="sm")
    g.add_node(ElementwiseOp("divide"), ["exp", "sm"], Tensor("out", (4, 8)), node_id="out")
    g.inputs, g.outputs = ["x"], ["out"]
    return g


def test_softmax_only_kernel_input_constant():
    result = _fuse(_make_softmax())
    for n in result.nodes.values():
        assert isinstance(n.op, (LoopOp, InputOp, ConstantOp))


def test_softmax_body_covers_all_ops():
    result = _fuse(_make_softmax())
    all_fns = set()
    for k in _kernel_nodes(result):
        all_fns |= set(_assign_fns(k.op.body))
        all_fns |= _local_combine_fns(k.op.body.accums)
    # Expect elementwise sub/exp and reduce combine add/max from the
    # max and sum accumulators. ``divide(x, acc_sum)`` is split by
    # ``split_invariant_divides`` (in ``ir/stmt/normalize.py``) into
    # ``reciprocal(acc_sum) + multiply(x, recip)`` so the rcp can hoist
    # out of the inner reduce — divide no longer appears as a body op.
    assert {"subtract", "exp", "reciprocal", "multiply"} <= all_fns
    assert {"add", "maximum"} <= all_fns


# ===================================================================
# Single elementwise: identity case
# ===================================================================


def test_single_elementwise_fuses():
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (8,)), node_id="x")
    g.add_node(ElementwiseOp("negative"), ["x"], Tensor("y", (8,)), node_id="y")
    g.inputs, g.outputs = ["x"], ["y"]
    result = _fuse(g)
    kernels = _kernel_nodes(result)
    assert len(kernels) == 1


# ===================================================================
# SSA invariants: unique names, defined-before-use
# ===================================================================


def test_ssa_invariants_hold():
    """LoopOp.__post_init__ validates SSA; this just confirms no crash."""
    result = _fuse(_make_softmax())
    for k in _kernel_nodes(result):
        # Re-validate explicitly
        defined = set()
        for decl in k.op.body.accums:
            defined.add(decl.name)
        from emmy.compiler.ir.loop import Accum, Load

        for s in k.op:
            if isinstance(s, Assign):
                for arg in s.args:
                    assert arg in defined, f"arg {arg!r} not defined before use in {s.name}"
                defined.add(s.name)
            elif isinstance(s, (Load, Accum)):
                defined.add(s.name)
            elif isinstance(s, Write):
                assert s.value in defined


# ===================================================================
# Correctness: pre-fusion vs post-fusion numeric equivalence via NumpyBackend.
# ===================================================================


def test_pointwise_chain_correctness():
    x = rng.standard_normal(8).astype(np.float32)
    _assert_correctness(_make_pointwise_chain, {"x": x})


def test_contraction_correctness():
    a = rng.standard_normal((4, 8)).astype(np.float32)
    b = rng.standard_normal((4, 8)).astype(np.float32)
    _assert_correctness(_make_contraction, {"a": a, "b": b})


def test_contraction_epilogue_correctness():
    a = rng.standard_normal((4, 8)).astype(np.float32)
    b = rng.standard_normal((4, 8)).astype(np.float32)
    bias = rng.standard_normal((4, 1)).astype(np.float32)
    _assert_correctness(_make_contraction_with_epilogue, {"a": a, "b": b, "bias": bias})


def test_softmax_correctness():
    x = rng.standard_normal((4, 8)).astype(np.float32)
    _assert_correctness(_make_softmax, {"x": x})


def test_single_elementwise_correctness():
    def _make():
        g = Graph()
        g.add_node(InputOp(), [], Tensor("x", (8,)), node_id="x")
        g.add_node(ElementwiseOp("negative"), ["x"], Tensor("y", (8,)), node_id="y")
        g.inputs, g.outputs = ["x"], ["y"]
        return g

    x = rng.standard_normal(8).astype(np.float32)
    _assert_correctness(_make, {"x": x})


# ===================================================================
# Sibling reductions: axis aliasing collapses them into one kernel.
# ===================================================================


def _make_sibling_reductions():
    """``s = sum(x, -1); m = max(x, -1); out = s + m`` — two reduces over x
    feeding one elementwise. With reduce-axis aliasing, fuses into one kernel
    with two accumulators sharing one reduce axis."""
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 8)), node_id="x")
    g.add_node(ReduceOp("sum", -1), ["x"], Tensor("s", (4, 1)), node_id="s")
    g.add_node(ReduceOp("maximum", -1), ["x"], Tensor("m", (4, 1)), node_id="m")
    g.add_node(ElementwiseOp("add"), ["s", "m"], Tensor("out", (4, 1)), node_id="out")
    g.inputs, g.outputs = ["x"], ["out"]
    return g


def test_sibling_reductions_fuse_to_one_kernel():
    result = _fuse(_make_sibling_reductions())
    kernels = _kernel_nodes(result)
    assert len(kernels) == 1, f"expected 1 kernel, got {len(kernels)}"


def test_sibling_reductions_share_reduce_axis():
    result = _fuse(_make_sibling_reductions())
    kernel = _kernel_nodes(result)[0]
    reduce_axes = [a for a in kernel.op.axes if a.name in kernel.op.reduce_axis_names]
    assert len(reduce_axes) == 1, f"expected 1 reduce axis, got {[a.name for a in reduce_axes]}"


def test_sibling_reductions_have_both_accumulators():
    result = _fuse(_make_sibling_reductions())
    kernel = _kernel_nodes(result)[0]
    combine_fns = _local_combine_fns(kernel.op.body.accums)
    assert {"add", "maximum"} <= combine_fns


def test_sibling_reductions_correctness():
    x = rng.standard_normal((4, 8)).astype(np.float32)
    _assert_correctness(_make_sibling_reductions, {"x": x})


# ===================================================================
# Softmax: multi-port producer consumption + reduce-axis aliasing.
# ===================================================================


def test_softmax_fuses_to_one_kernel():
    """Softmax's two-reduce pattern (max sweep → sub → exp → sum sweep → div)
    fuses into a single kernel with two accumulators sharing one reduce axis."""
    result = _fuse(_make_softmax())
    kernels = _kernel_nodes(result)
    assert len(kernels) == 1, f"expected 1 kernel, got {len(kernels)}"


def test_softmax_single_reduce_axis():
    result = _fuse(_make_softmax())
    kernel = _kernel_nodes(result)[0]
    reduce_axes = [a for a in kernel.op.axes if a.name in kernel.op.reduce_axis_names]
    assert len(reduce_axes) == 1


def test_softmax_has_both_accumulators():
    result = _fuse(_make_softmax())
    kernel = _kernel_nodes(result)[0]
    combine_fns = _local_combine_fns(kernel.op.body.accums)
    assert {"add", "maximum"} <= combine_fns, f"missing accumulators: {combine_fns}"


# ===================================================================
# Split non-reconvergent shared index-map fan-out (005).
#
# Closed reconvergent fan-out is absorbed as one SSA region by
# ``merge_loop_ops``. When consumers remain separate, a pure index map can be
# cheaply copied into each consumer; ``005_split_shared_indexmap`` performs
# that multi-output rewrite and leaves no copy kernel behind.
# ===================================================================


def _is_pure_indexmap(op: LoopOp) -> bool:
    return not any(isinstance(s, (Assign, Accum)) for s in op.body.iter())


def _pure_indexmap_kernels(graph: Graph) -> list:
    return [n for n in _kernel_nodes(graph) if _is_pure_indexmap(n.op)]


def _loads_from(op: LoopOp) -> set[str]:
    return {ld.input for ld in op.body.loads}


def _split_only(graph: Graph) -> Graph:
    """Run lifting + the split rule alone (no merge) — the in-isolation view."""
    return Pipeline.build(
        ["loop/lifting", "loop/fusion"],
        select={"lift_elementwise", "lift_reduce", "lift_indexmap", "lift_gather", "split_shared_indexmap"},
    ).run(graph)


def _make_shared_const_broadcast():
    """A scalar ``const_bc(1.0)`` broadcast feeding two elementwise consumers —
    the headline case: torch.export folds attention-mask / RoPE scaffolding to
    scalar broadcasts that fan out (Qwen3 GQA query + key paths)."""
    from emmy.compiler.pipeline.passes.frontend.decomposition._helpers import const_bc

    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 8)), node_id="x")
    g.add_node(InputOp(), [], Tensor("y", (4, 8)), node_id="y")
    bc = const_bc(g, name="one", value=1.0, target_shape=(4, 8), dtype="f32")
    g.add_node(ElementwiseOp("multiply"), ["x", bc.id], Tensor("c1", (4, 8)), node_id="c1")
    g.add_node(ElementwiseOp("add"), ["y", bc.id], Tensor("c2", (4, 8)), node_id="c2")
    g.inputs, g.outputs = ["x", "y"], ["c1", "c2"]
    return g


def test_shared_const_broadcast_no_pure_indexmap_remains():
    """After fusion the shared constant broadcast is gone — no pure-indexmap copy kernel."""
    result = _fuse(_make_shared_const_broadcast())
    assert _pure_indexmap_kernels(result) == []


def test_shared_const_broadcast_consumers_load_constant_directly():
    """Both consumers Load the ``ConstantOp`` (``one``) directly — the broadcast
    folded in, so the cuda literal-inline path can stamp ``float x = 1.0f;``."""
    result = _fuse(_make_shared_const_broadcast())
    kernels = _kernel_nodes(result)
    assert len(kernels) == 2, f"expected 2 consumer kernels, got {len(kernels)}"
    for k in kernels:
        assert "one" in _loads_from(k.op), f"{k.id} does not load the constant directly: {_loads_from(k.op)}"


def test_shared_const_broadcast_correctness():
    x = rng.standard_normal((4, 8)).astype(np.float32)
    y = rng.standard_normal((4, 8)).astype(np.float32)
    _assert_correctness(_make_shared_const_broadcast, {"x": x, "y": y})


def test_shared_const_broadcast_split_in_isolation():
    """Split rule alone (no merge): the producer is fused into *all* consumers in
    one shot — no broadcast kernel survives and each consumer reads the constant
    directly, without ``merge_loop_ops`` running at all."""
    result = _split_only(_make_shared_const_broadcast())
    assert _pure_indexmap_kernels(result) == [], "the shared broadcast should be fully dissolved by the split rule alone"
    kernels = _kernel_nodes(result)
    assert len(kernels) == 2, f"expected 2 fused consumer kernels, got {len(kernels)}"
    for k in kernels:
        assert "one" in _loads_from(k.op), f"{k.id} does not load the constant directly: {_loads_from(k.op)}"


def _make_shared_transpose():
    """A transpose (general layout op) feeding two elementwise consumers —
    covers the non-constant pure-indexmap path."""
    from emmy.compiler.ir.expr import placeholder
    from emmy.compiler.pipeline.passes.frontend.decomposition._helpers import single_indexmap

    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 8)), node_id="x")
    # out[i, j] = x[j, i] — transpose (4, 8) -> (8, 4).
    t = single_indexmap(g, "x", out_shape=(8, 4), coord_map=(placeholder(1), placeholder(0)), name="xt")
    g.add_node(ElementwiseOp("exp"), [t.id], Tensor("c1", (8, 4)), node_id="c1")
    g.add_node(ElementwiseOp("negative"), [t.id], Tensor("c2", (8, 4)), node_id="c2")
    g.inputs, g.outputs = ["x"], ["c1", "c2"]
    return g


def test_shared_transpose_no_pure_indexmap_remains():
    result = _fuse(_make_shared_transpose())
    assert _pure_indexmap_kernels(result) == []


def test_shared_transpose_consumers_index_source_directly():
    """Both consumers read the transpose's source (``x``) directly — the layout
    op folded into each as lazy per-consumer indexing."""
    result = _fuse(_make_shared_transpose())
    kernels = _kernel_nodes(result)
    assert len(kernels) == 2, f"expected 2 consumer kernels, got {len(kernels)}"
    for k in kernels:
        assert "x" in _loads_from(k.op), f"{k.id} does not load the source directly: {_loads_from(k.op)}"


def test_shared_transpose_correctness():
    x = rng.standard_normal((4, 8)).astype(np.float32)
    _assert_correctness(_make_shared_transpose, {"x": x})


# ===================================================================
# Data-dependent (gather) producer fused into a multi-read reduce.
#
# A gather lowers to ``in1 = load w[(int)in0, h]`` where ``in0 = load idx`` — a
# Load whose *index* reads SSA ``in0``. When the gather output feeds an
# RMSNorm-like consumer that reads it at two scopes (the variance reduce + the
# normalize), each read inlines the gather and its ``load idx``. Deduping the
# duplicate ``load idx`` must rewire the surviving gather's *index* reference,
# or the second read's index dangles and the canonical renamer collapses it into
# a self-referential Load (``in4 = load w[(int)in4]``) — silently reading a
# constant row instead of ``idx[pos]``. ``Load.deps()`` reporting index SSA +
# ``dedup_loads`` rewiring kept Loads' indices is what makes this correct.
# ===================================================================


def _make_gather_into_reduce():
    from emmy.compiler.pipeline.passes.frontend.decomposition._broadcast import broadcast_to

    V, H, S = 16, 8, 4
    g = Graph()
    g.add_node(InputOp(), [], Tensor("w", (V, H)), node_id="w")
    g.add_node(InputOp(), [], Tensor("idx", (1, S)), node_id="idx")
    g.add_node(GatherOp(axis=0), ["w", "idx"], Tensor("emb", (1, S, H)), node_id="emb")
    # RMSNorm-like: variance reduce reads emb, normalize reads emb again.
    g.add_node(ElementwiseOp("multiply"), ["emb", "emb"], Tensor("sq", (1, S, H)), node_id="sq")
    g.add_node(ReduceOp("sum", -1), ["sq"], Tensor("red", (1, S, 1)), node_id="red")
    g.add_node(ElementwiseOp("multiply"), ["emb", broadcast_to(g, "red", (1, S, H))], Tensor("out", (1, S, H)), node_id="out")
    g.inputs, g.outputs = ["w", "idx"], ["out"]
    return g


def test_gather_into_reduce_no_pure_indexmap():
    """The gather fuses into its consumers — no standalone copy kernel survives
    (it would, if the data-dependent index couldn't be carried across the fuse)."""
    result = _fuse(_make_gather_into_reduce())
    assert _pure_indexmap_kernels(result) == []


def test_gather_into_reduce_correctness():
    w = rng.standard_normal((16, 8)).astype(np.float32)
    idx = rng.integers(0, 16, size=(1, 4)).astype(np.float32)
    _assert_correctness(_make_gather_into_reduce, {"w": w, "idx": idx})


def _make_repeated_projection_read():
    """A reduction producer consumed twice by one pointwise node."""
    m, k, n = 2, 8, 6
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (m, k)), node_id="x")
    g.add_node(InputOp(), [], Tensor("w", (n, k)), node_id="w")
    g.add_node(LinearOp(), ["x", "w"], Tensor("projection", (m, n)), node_id="projection")
    g.add_node(ElementwiseOp("multiply"), ["projection", "projection"], Tensor("square", (m, n)), node_id="square")
    g.inputs, g.outputs = ["x", "w"], ["square"]
    return g


def test_equal_coordinate_projection_reads_share_one_reduction():
    """The splicer shares identical demands instead of treating every repeated read as duplication."""
    result = _decompose_and_fuse(_make_repeated_projection_read())
    kernels = _kernel_nodes(result)
    assert len(kernels) == 1
    assert sum(isinstance(stmt, Accum) for stmt in kernels[0].op.body.iter()) == 1


def test_repeated_projection_read_fusion_is_correct():
    x = rng.standard_normal((2, 8)).astype(np.float32)
    w = rng.standard_normal((6, 8)).astype(np.float32)
    _assert_correctness(_make_repeated_projection_read, {"x": x, "w": w})


def _make_norm_linear():
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (2, 8)), node_id="x")
    g.add_node(InputOp(), [], Tensor("nw", (8,)), node_id="nw")
    g.add_node(InputOp(), [], Tensor("w", (6, 8)), node_id="w")
    g.add_node(RmsNormOp(eps=1e-6), ["x", "nw"], Tensor("normalized", (2, 8)), node_id="normalized")
    g.add_node(LinearOp(), ["normalized", "w"], Tensor("out", (2, 6)), node_id="out")
    g.inputs, g.outputs = ["x", "nw", "w"], ["out"]
    return g


def test_single_read_norm_linear_still_fuses():
    """The repeated-reduce guard must not block the intended computed-A norm→linear cone."""
    result = _decompose_and_fuse(_make_norm_linear())
    assert len(_kernel_nodes(result)) == 1


def _make_projection_norm():
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (2, 8)), node_id="x")
    g.add_node(InputOp(), [], Tensor("w", (6, 8)), node_id="w")
    g.add_node(InputOp(), [], Tensor("nw", (6,)), node_id="nw")
    g.add_node(LinearOp(), ["x", "w"], Tensor("projection", (2, 6)), node_id="projection")
    g.add_node(RmsNormOp(eps=1e-6), ["projection", "nw"], Tensor("out", (2, 6)), node_id="out")
    g.inputs, g.outputs = ["x", "w", "nw"], ["out"]
    return g


def test_projection_feeding_rms_value_and_statistic_stays_materialized():
    """RMSNorm reads its input for both x² and x; do not execute an upstream projection twice."""
    result = _decompose_and_fuse(_make_projection_norm())
    kernels = _kernel_nodes(result)
    assert len(kernels) == 2
    projection = next(node for node in kernels if set(node.inputs) == {"w", "x"})
    norm = next(node for node in kernels if node is not projection)
    assert sum(load.input == projection.id for load in norm.op.loads) == 2


def test_projection_norm_materialization_is_correct():
    inputs = {
        "x": rng.standard_normal((2, 8)).astype(np.float32),
        "w": rng.standard_normal((6, 8)).astype(np.float32),
        "nw": rng.standard_normal((6,)).astype(np.float32),
    }
    _assert_correctness(_make_projection_norm, inputs)


def _make_shared_broadcast_chain():
    """A scalar broadcast (shared) feeding two *further* broadcasts that each feed
    a downstream multiply — mirrors the full-model RoPE/mask shape:
    ``scalar -> [1,1,4,8] (shared) -> {[1,3,4,8], [1,2,4,8]} -> multiply``.

    The intermediate broadcasts are the regression trigger: lifting gives each a
    node id (``lift_<id>``) that differs from its output Tensor name, so when the
    split rule peels one onto a private copy it MUST rename the new node's
    ``Write.output`` to its new id. Forgetting that leaves the node writing the
    old buf — ``splice_graph`` (which assumes Write.output == node id) then can't
    fold it into the downstream multiply, and a pure-indexmap copy survives."""
    from emmy.compiler.pipeline.passes.frontend.decomposition._broadcast import broadcast_to
    from emmy.compiler.pipeline.passes.frontend.decomposition._helpers import const_bc

    g = Graph()
    g.add_node(InputOp(), [], Tensor("mq", (1, 3, 4, 8)), node_id="mq")
    g.add_node(InputOp(), [], Tensor("mk", (1, 2, 4, 8)), node_id="mk")
    shared = const_bc(g, name="zero", value=0.0, target_shape=(1, 1, 4, 8), dtype="f32")
    bq = broadcast_to(g, shared.id, (1, 3, 4, 8))
    bk = broadcast_to(g, shared.id, (1, 2, 4, 8))
    g.add_node(ElementwiseOp("multiply"), ["mq", bq.id], Tensor("fq", (1, 3, 4, 8)), node_id="fq")
    g.add_node(ElementwiseOp("multiply"), ["mk", bk.id], Tensor("fk", (1, 2, 4, 8)), node_id="fk")
    g.inputs, g.outputs = ["mq", "mk"], ["fq", "fk"]
    return g


def test_shared_broadcast_chain_no_pure_indexmap_remains():
    """Regression: the intermediate broadcasts (node id != output name) must fully
    fold — no pure-indexmap copy left writing a stale buf."""
    result = _fuse(_make_shared_broadcast_chain())
    assert _pure_indexmap_kernels(result) == []


def test_shared_broadcast_chain_correctness():
    mq = rng.standard_normal((1, 3, 4, 8)).astype(np.float32)
    mk = rng.standard_normal((1, 2, 4, 8)).astype(np.float32)
    _assert_correctness(_make_shared_broadcast_chain, {"mq": mq, "mk": mk})


def test_output_reshape_folds_into_reduce_producer():
    """``030_fold_output_reshape``: a graph-output memcpy-identity flatten of a reduce-bearing
    producer folds by retargeting the producer's ``Write`` to the output buffer at the same flat
    address (clean affine per-dim index — no div/mod), dropping the copy kernel. The normal merge
    rule can't take this pair (inlining the producer at the consumer's load would re-run the
    reduce per element; the flatten's div/mod reader σ defeats the splicer anyway) — the gemma-4
    decode pre twin's q/k/v head-layout flattens after the per-head qk norms."""
    from emmy.compiler.dim import Dim
    from emmy.compiler.ir.axis import Axis
    from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
    from emmy.compiler.ir.loop import Load, Loop
    from emmy.compiler.ir.stmt import Body

    H, D = 3, 8

    def make_graph():
        # Producer: per-head scale-by-row-sum (a reduce + sweep per (row, head) — reduce-heavy).
        a0, a1 = Var("a0"), Var("a1")
        red = Loop(
            axis=Axis(name="a2", extent=Dim(D)),
            body=Body(
                (
                    Load(name="in0", input="x", index=(a0, a1, Var("a2"))),
                    Accum(name="acc", value="in0", op="add", axes=("a2",)),
                )
            ),
        )
        sweep = Loop(
            axis=Axis(name="a3", extent=Dim(D)),
            body=Body(
                (
                    Load(name="in1", input="x", index=(a0, a1, Var("a3"))),
                    Assign(name="v0", op="multiply", args=("in1", "acc")),
                    Write(output="y", index=(a0, a1, Var("a3")), value="v0"),
                )
            ),
        )
        producer = LoopOp(
            body=Body(
                (
                    Loop(
                        axis=Axis(name="a0", extent=Dim(4)),
                        body=Body((Loop(axis=Axis(name="a1", extent=Dim(H)), body=Body((red, sweep))),)),
                    ),
                )
            )
        )
        # Consumer: the traced flatten (4, H, D) -> (4, H*D) — a flat-memory identity copy.
        b0, b1 = Var("b0"), Var("b1")
        copy = LoopOp(
            body=Body(
                (
                    Loop(
                        axis=Axis(name="b0", extent=Dim(4)),
                        body=Body(
                            (
                                Loop(
                                    axis=Axis(name="b1", extent=Dim(H * D)),
                                    body=Body(
                                        (
                                            Load(
                                                name="c0",
                                                input="y",
                                                index=(b0, BinaryExpr("/", b1, Literal(D, "int")), BinaryExpr("%", b1, Literal(D, "int"))),
                                            ),
                                            Write(output="out", index=(b0, b1), value="c0"),
                                        )
                                    ),
                                ),
                            )
                        ),
                    ),
                )
            )
        )
        g = Graph()
        g.add_node(InputOp(), [], Tensor("x", (4, H, D)), node_id="x")
        g.add_node(producer, ["x"], Tensor("y", (4, H, D)), node_id="y")
        g.add_node(copy, ["y"], Tensor("out", (4, H * D)), node_id="out")
        g.inputs, g.outputs = ["x"], ["out"]
        return g

    fused = Pipeline.build(["loop/fusion"]).run(make_graph())
    kernels = _kernel_nodes(fused)
    assert len(kernels) == 1, f"the flatten copy must fold into its producer: {[n.id for n in kernels]}"
    writes = [s for s in kernels[0].op.body.iter() if isinstance(s, Write)]
    assert len(writes) == 1 and writes[0].output == "out"
    idx = writes[0].index
    assert len(idx) == 2, "the retargeted Write indexes the flat output shape"
    assert "/" not in idx[1].pretty() and "%" not in idx[1].pretty(), f"clean affine index expected: {idx[1].pretty()}"

    x = rng.standard_normal((4, H, D)).astype(np.float32)
    before = _run(make_graph(), {"x": x})
    after = _run(fused, {"x": x})
    _assert_close(before, after)


# ===================================================================
# Flash P@V deferral vs residual epilogue: the contraction halves must
# reunite past an exp-bearing residual stream
# ===================================================================


def _loops2(inner, ax0, ext0, ax1, ext1):
    from emmy.compiler.dim import Dim
    from emmy.compiler.ir.axis import Axis
    from emmy.compiler.ir.loop import Loop
    from emmy.compiler.ir.stmt import Body

    return Body((Loop(axis=Axis(name=ax0, extent=Dim(ext0)), body=Body((Loop(axis=Axis(name=ax1, extent=Dim(ext1)), body=inner),))),))


def _make_pending_product_into_exp_residual_add():
    """The mid-fusion state the engine reaches on every transformer layer past the first: the
    decomposed o_proj's sum-reduce already merged into the residual add (which also reads the
    residual stream ``res``), the bare product ``ew`` still materialized, and ``res`` one
    producer hop from an exp (the previous layer's silu sigmoid / softmax). The product must
    still merge into the add's reduce — the flash P@V deferral must not mistake the post-reduce
    residual operand for a V operand (it kept every o_proj product materialized: a (b, s, k, n)
    gmem scratch per layer, 17 GB at S=4096 on the Qwen3 embed trunk)."""
    from emmy.compiler.dim import Dim
    from emmy.compiler.ir.axis import Axis
    from emmy.compiler.ir.expr import Var
    from emmy.compiler.ir.loop import Load, Loop
    from emmy.compiler.ir.stmt import Body

    S, K, N = 4, 8, 6
    a0, a1, a2 = Var("a0"), Var("a1"), Var("a2")

    ew_cell = Body(
        (
            Loop(
                axis=Axis(name="a2", extent=Dim(N)),
                body=Body(
                    (
                        Load(name="i0", input="act", index=(a0, a1)),
                        Load(name="i1", input="w", index=(a2, a1)),
                        Assign(name="v0", op="multiply", args=("i0", "i1")),
                        Write(output="ew", index=(a0, a1, a2), value="v0"),
                    )
                ),
            ),
        )
    )
    ew = LoopOp(body=_loops2(ew_cell, "a0", S, "a1", K))

    exp_cell = Body(
        (
            Load(name="c0", input="r", index=(a0, a1)),
            Assign(name="c1", op="exp", args=("c0",)),
            Write(output="e", index=(a0, a1), value="c1"),
        )
    )
    e = LoopOp(body=_loops2(exp_cell, "a0", S, "a1", N))
    res_cell = Body(
        (
            Load(name="d0", input="r", index=(a0, a1)),
            Load(name="d1", input="e", index=(a0, a1)),
            Assign(name="d2", op="add", args=("d0", "d1")),
            Write(output="res", index=(a0, a1), value="d2"),
        )
    )
    res = LoopOp(body=_loops2(res_cell, "a0", S, "a1", N))

    add_cell = Body(
        (
            Loop(
                axis=Axis(name="a2", extent=Dim(K)),
                body=Body(
                    (
                        Load(name="in0", input="ew", index=(a0, a2, a1)),
                        Accum(name="acc0", value="in0", op="add", axes=("a2",)),
                    )
                ),
            ),
            Load(name="in1", input="res", index=(a0, a1)),
            Assign(name="v1", op="add", args=("acc0", "in1")),
            Write(output="y", index=(a0, a1), value="v1"),
        )
    )
    add13 = LoopOp(body=_loops2(add_cell, "a0", S, "a1", N))

    g = Graph()
    g.add_node(InputOp(), [], Tensor("act", (S, K)), node_id="act")
    g.add_node(InputOp(), [], Tensor("w", (N, K)), node_id="w")
    g.add_node(InputOp(), [], Tensor("r", (S, N)), node_id="r")
    g.add_node(ew, ["act", "w"], Tensor("ew", (S, K, N)), node_id="ew")
    g.add_node(e, ["r"], Tensor("e", (S, N)), node_id="e")
    g.add_node(res, ["r", "e"], Tensor("res", (S, N)), node_id="res")
    g.add_node(add13, ["ew", "res"], Tensor("y", (S, N)), node_id="y")
    # ``res`` doubles as a graph output — the residual stream has other readers in the real
    # model, so the exp-bearing chain can never be absorbed into the consumer and the guard
    # decides the product's fate alone.
    g.inputs, g.outputs = ["act", "w", "r"], ["y", "res"]
    return g


def test_pending_product_reunites_past_exp_residual():
    fused = Pipeline.build(["loop/fusion"]).run(_make_pending_product_into_exp_residual_add())
    bare = [n.id for n in _kernel_nodes(fused) if not _has_update(n.op.body) and len(n.output.shape) > 2]
    assert bare == [], f"bare product left materialized: {bare}"


def test_pending_product_exp_residual_correctness():
    act = rng.standard_normal((4, 8)).astype(np.float32)
    w = rng.standard_normal((6, 8)).astype(np.float32)
    r = rng.standard_normal((4, 6)).astype(np.float32)
    _assert_correctness(_make_pending_product_into_exp_residual_add, {"act": act, "w": w, "r": r})
