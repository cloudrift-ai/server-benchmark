"""Tests for ``035_merge_sibling_linears``: sibling ``LinearOp``s sharing one activation merge
into a single linear over an N-concat (``source_parts``) weight, with ``SliceOp`` views
re-deriving the original outputs. Structural checks, numeric parity (numpy backend before ==
after, weights bound through ``bind_constants`` so the concat loader path is exercised), and
every guard."""

import numpy as np

from emmy.compiler.backend.numpy import NumpyBackend
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.frontend.ir import LinearOp, ReshapeOp, SliceOp, TransposeOp
from emmy.compiler.ir.tensor.ir import ElementwiseOp
from emmy.compiler.loader.binder import bind_constants
from emmy.compiler.pipeline import Pipeline

_DECOMP_PASS = "frontend/decomposition"
_RULE = "035_merge_sibling_linears"

rng = np.random.default_rng(7)
_backend = NumpyBackend()


def _apply(graph: Graph, select=(_RULE,)) -> Graph:
    return Pipeline.build([_DECOMP_PASS], select=list(select)).run(graph)


def _run(graph: Graph, x: np.ndarray, sources: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    input_data = {"x": x, **bind_constants(graph, sources)}
    return _backend.run(_backend.compile(graph), input_data=input_data)[0].outputs


def _sibling_graph(ns=(32, 16), k=64, *, direct_outputs=False, bias=False, shared_weight=False, weight_load_ops=()):
    """``ns`` sibling linears over one input ``x[4, k]``; each output squared through an
    elementwise consumer (so the linears aren't graph outputs) unless ``direct_outputs``."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4, k), "f32"), node_id="x")
    outs = []
    for i, n in enumerate(ns):
        wid = "w0" if shared_weight else f"w{i}"
        if wid not in g.nodes:
            g.add_node(
                op=ConstantOp(name=wid, source_path=f"m.{wid}", source_shape=(n, k), source_dtype="f32", load_ops=tuple(weight_load_ops)),
                inputs=[],
                output=Tensor(wid, (n, k), "f32"),
                node_id=wid,
            )
        lid = f"lin{i}"
        inputs = ["x", wid]
        if bias:
            bid = f"b{i}"
            g.add_node(
                op=ConstantOp(name=bid, source_path=f"m.{bid}", source_shape=(n,), source_dtype="f32"),
                inputs=[],
                output=Tensor(bid, (n,), "f32"),
                node_id=bid,
            )
            inputs.append(bid)
        g.add_node(op=LinearOp(has_bias=bias), inputs=inputs, output=Tensor(lid, (4, n), "f32"), node_id=lid)
        if direct_outputs:
            outs.append(lid)
        else:
            g.add_node(op=ElementwiseOp(op="multiply"), inputs=[lid, lid], output=Tensor(f"sq{i}", (4, n), "f32"), node_id=f"sq{i}")
            outs.append(f"sq{i}")
    g.inputs, g.outputs = ["x"], outs
    return g


def _sources(g: Graph) -> dict[str, np.ndarray]:
    return {
        op.source_path: rng.standard_normal(op.source_shape).astype(np.float32)
        for _nid, op in g.constant_ops()
        if op.source_path is not None
    }


def _count(g: Graph, op_type) -> int:
    return sum(1 for n in g.nodes.values() if isinstance(n.op, op_type))


# ===================================================================
# The merge: structure + numerics
# ===================================================================


def test_merge_produces_one_linear_with_concat_weight():
    result = _apply(_sibling_graph())
    assert _count(result, LinearOp) == 1
    (w_op,) = [n.op for n in result.nodes.values() if isinstance(n.op, ConstantOp) and n.op.source_parts]
    assert w_op.source_parts == (("m.w0", (32, 64)), ("m.w1", (16, 64)))
    assert w_op.source_path is None
    slices = sorted((n.op for n in result.nodes.values() if isinstance(n.op, SliceOp)), key=lambda s: s.start)
    assert [(s.dim, s.start) for s in slices] == [(-1, 0), (-1, 32)]


def test_merge_preserves_output_names_and_shapes():
    g = _sibling_graph()
    orig = {oid: g.nodes[oid].output.shape for oid in g.outputs}
    result = _apply(g)
    assert set(result.outputs) == set(orig)
    for oid, shape in orig.items():
        assert result.nodes[oid].output.shape == shape


def test_merge_numeric_parity():
    g = _sibling_graph()
    sources = _sources(g)
    x = rng.standard_normal((4, 64)).astype(np.float32)
    before = _run(g, x, sources)
    after = _run(_apply(g), x, sources)
    for oid in g.outputs:
        np.testing.assert_allclose(after[oid], before[oid], rtol=1e-5, atol=1e-6)


def test_three_siblings_fold_to_flat_parts():
    g = _sibling_graph(ns=(32, 16, 8))
    sources = _sources(g)
    result = _apply(g)
    assert _count(result, LinearOp) == 1
    (w_op,) = [n.op for n in result.nodes.values() if isinstance(n.op, ConstantOp) and n.op.source_parts]
    assert w_op.source_parts == (("m.w0", (32, 64)), ("m.w1", (16, 64)), ("m.w2", (8, 64)))
    x = rng.standard_normal((4, 64)).astype(np.float32)
    before = _run(g, x, sources)
    after = _run(result, x, sources)
    for oid in g.outputs:
        np.testing.assert_allclose(after[oid], before[oid], rtol=1e-5, atol=1e-6)


def test_merge_idempotent():
    once = _apply(_sibling_graph())
    twice = _apply(once)
    assert len(twice.nodes) == len(once.nodes)


def test_full_decomposition_after_merge_numeric_parity():
    """The merged linear + slices decompose through the FULL pass (040_linear, 140_slice, …)
    to the same numbers as the unmerged graph."""
    g = _sibling_graph()
    sources = _sources(g)
    x = rng.standard_normal((4, 64)).astype(np.float32)
    before = _run(g, x, sources)
    after = _run(Pipeline.build([_DECOMP_PASS]).run(g), x, sources)
    for oid in g.outputs:
        np.testing.assert_allclose(after[oid], before[oid], rtol=1e-4, atol=1e-5)


# ===================================================================
# Guards
# ===================================================================


def test_no_merge_when_linears_are_graph_outputs():
    result = _apply(_sibling_graph(direct_outputs=True))
    assert _count(result, LinearOp) == 2


def test_no_merge_through_view_only_output_path():
    """A linear whose output reaches a graph output through layout ops only stays unmerged —
    the slice view would demote to a materialized copy kernel at the capture ABI."""
    g = _sibling_graph()
    g.add_node(op=ReshapeOp(shape=(2, 2, 32)), inputs=["lin0"], output=Tensor("v0", (2, 2, 32), "f32"), node_id="v0")
    g.outputs = ["v0", "sq1"]
    result = _apply(g)
    assert _count(result, LinearOp) == 2


def test_biased_siblings_merge_weights_and_biases():
    result = _apply(_sibling_graph(bias=True))
    assert _count(result, LinearOp) == 1
    params = [n.op for n in result.nodes.values() if isinstance(n.op, ConstantOp) and n.op.source_parts]
    assert len(params) == 2
    weight = next(op for op in params if len(op.source_shape) == 2)
    bias = next(op for op in params if len(op.source_shape) == 1)
    assert weight.source_parts == (("m.w0", (32, 64)), ("m.w1", (16, 64)))
    assert bias.source_parts == (("m.b0", (32,)), ("m.b1", (16,)))


def test_biased_sibling_merge_numeric_parity():
    g = _sibling_graph(bias=True)
    sources = _sources(g)
    x = rng.standard_normal((4, 64)).astype(np.float32)
    before = _run(g, x, sources)
    after = _run(_apply(g), x, sources)
    for oid in g.outputs:
        np.testing.assert_allclose(after[oid], before[oid], rtol=1e-5, atol=1e-6)


def test_no_merge_with_shared_weight():
    result = _apply(_sibling_graph(ns=(32, 32), shared_weight=True))
    assert _count(result, LinearOp) == 2


def test_no_merge_with_folded_load_ops():
    result = _apply(_sibling_graph(weight_load_ops=(TransposeOp(axes=(1, 0)),)))
    assert _count(result, LinearOp) == 2


def test_no_merge_with_folded_constant_subgraphs():
    """A generic ``source_graph`` bind record is not pristine: its evaluation has no
    concat-of-paths spelling, so the siblings stay unmerged."""
    from dataclasses import replace

    g = _sibling_graph()
    for wid in ("w0", "w1"):
        record = Graph()
        op = g.nodes[wid].op
        record.add_node(
            op=ConstantOp(name=f"{wid}_bits", source_path=op.source_path, source_shape=op.source_shape, source_dtype="f8e4m3"),
            inputs=[],
            output=Tensor(f"{wid}_bits", op.source_shape, "f8e4m3"),
            node_id=f"{wid}_bits",
        )
        record.outputs = [f"{wid}_bits"]
        g.nodes[wid].op = replace(op, source_path=None, source_graph=record)
    result = _apply(g)
    assert _count(result, LinearOp) == 2


def test_no_merge_across_different_activations():
    g = Graph()
    for i, xid in enumerate(("x", "y")):
        g.add_node(op=InputOp(), inputs=[], output=Tensor(xid, (4, 64), "f32"), node_id=xid)
        g.add_node(
            op=ConstantOp(name=f"w{i}", source_path=f"m.w{i}", source_shape=(16, 64), source_dtype="f32"),
            inputs=[],
            output=Tensor(f"w{i}", (16, 64), "f32"),
            node_id=f"w{i}",
        )
        g.add_node(op=LinearOp(), inputs=[xid, f"w{i}"], output=Tensor(f"lin{i}", (4, 16), "f32"), node_id=f"lin{i}")
        sq = Tensor(f"sq{i}", (4, 16), "f32")
        g.add_node(op=ElementwiseOp(op="multiply"), inputs=[f"lin{i}", f"lin{i}"], output=sq, node_id=f"sq{i}")
    g.inputs, g.outputs = ["x", "y"], ["sq0", "sq1"]
    result = _apply(g)
    assert _count(result, LinearOp) == 2
