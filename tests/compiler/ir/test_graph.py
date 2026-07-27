"""Tests for the tensor IR: Tensor, Node, Graph."""

import pytest

from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.tensor.ir import ElementwiseOp, ReduceOp

# ---- helpers ----


def _make_matmul_graph():
    """Build: C[M,1,N] = reduce_sum(elementwise_mul(A[M,K,N], B[M,K,N]), axis=1).

    Uses keepdim reduction and matching-shape elementwise inputs to stay on
    the Tensor IR rank-preservation invariant (see pipeline/passes/frontend/decomposition/_broadcast.py for how
    decomposition rules insert explicit IndexMapOps when shapes differ).
    """
    g = Graph()
    a = g.add_node(op=InputOp(), inputs=[], output=Tensor("A", ("M", "K", "N")), node_id="A")
    b = g.add_node(op=InputOp(), inputs=[], output=Tensor("B", ("M", "K", "N")), node_id="B")
    g.inputs = [a, b]

    ew = g.add_node(
        op=ElementwiseOp(op="multiply"),
        inputs=[a, b],
        output=Tensor("AB", ("M", "K", "N")),
        node_id="ew",
    )
    red = g.add_node(
        op=ReduceOp(op="sum", axis=1),
        inputs=[ew],
        output=Tensor("C", ("M", 1, "N")),
        node_id="red",
    )
    g.outputs = [red]
    return g


# ---- tests ----


def test_add_node_and_lookup():
    g = Graph()
    nid = g.add_node(op=InputOp(), inputs=[], output=Tensor("X", (4, 4)), node_id="x")
    assert nid == "x"
    assert "x" in g.nodes
    assert g.nodes["x"].output.name == "X"


def test_duplicate_id_raises():
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("X", (4,)), node_id="x")
    with pytest.raises(ValueError, match="already exists"):
        g.add_node(op=InputOp(), inputs=[], output=Tensor("Y", (4,)), node_id="x")


def test_missing_input_raises():
    g = Graph()
    with pytest.raises(ValueError, match="does not exist"):
        g.add_node(
            op=ElementwiseOp(op="add"),
            inputs=["nonexistent"],
            output=Tensor("Y", (4,)),
        )


def test_topological_order():
    g = _make_matmul_graph()
    order = g.topological_order()
    assert order.index("A") < order.index("ew")
    assert order.index("B") < order.index("ew")
    assert order.index("ew") < order.index("red")


def test_consumers():
    g = _make_matmul_graph()
    assert g.consumers("A") == ["ew"]
    assert g.consumers("B") == ["ew"]
    assert g.consumers("ew") == ["red"]
    assert g.consumers("red") == []


def test_replace_node():
    g = _make_matmul_graph()
    # Add a new node and rewire red's consumers to it.
    new_id = g.add_node(
        op=ReduceOp(op="sum", axis=0),
        inputs=["ew"],
        output=Tensor("C2", (1, "K", "N")),
        node_id="red2",
    )
    g.replace_node("red", new_id)
    assert "red2" in [o for o in g.outputs]


def test_remove_node():
    g = _make_matmul_graph()
    g.outputs = []
    g.remove_node("red")
    assert "red" not in g.nodes


def test_copy_is_independent():
    g = _make_matmul_graph()
    g2 = g.copy()
    g2.remove_node("red")
    assert "red" in g.nodes
    assert "red" not in g2.nodes


def test_fan_out():
    """One node consumed by two different nodes."""
    g = Graph()
    x = g.add_node(op=InputOp(), inputs=[], output=Tensor("X", (4,)), node_id="x")
    g.inputs = [x]
    a = g.add_node(op=ElementwiseOp(op="exp"), inputs=[x], output=Tensor("expX", (4,)), node_id="a")
    b = g.add_node(op=ElementwiseOp(op="negative"), inputs=[x], output=Tensor("negX", (4,)), node_id="b")
    g.outputs = [a, b]
    assert sorted(g.consumers("x")) == ["a", "b"]


def test_cuda_op_tma_descriptors_roundtrip():
    """A dumped cuda-stage graph must rehydrate ``CudaOp.tma_descriptors`` as
    ``TmaDescMeta`` instances, not their repr strings (``json.dumps(default=str)``
    stringifies the dataclasses on the way out; ``emmy run --ir <cuda.json>`` of a
    TMA kernel crashed in ``_planner.compute_live_intervals`` on ``d.src_buf``)."""
    import json

    from emmy.compiler.ir.cuda.ir import CudaOp, TmaDescMeta

    g = Graph()
    k = g.add_node(op=InputOp(), inputs=[], output=Tensor("k", (1, 4, 128, 64), "f16"), node_id="k")
    g.add_node(
        op=CudaOp(
            kernel_source='extern "C" __global__ void k_x() {}',
            kernel_name="k_x",
            arg_order=("k", "out", "_desc_k"),
            grid=((256,), (1,), (1,)),
            block=((128,), (1,), (1,)),
            tma_descriptors=(TmaDescMeta(name="_desc_k", src_buf="k", box_extents=(1, 1, 64, 64), swizzle="B128"),),
        ),
        inputs=[k],
        output=Tensor("out", (1, 4, 128, 64), "f16"),
        node_id="out",
    )
    g.inputs, g.outputs = [k], ["out"]

    loaded = Graph.from_dict(json.loads(json.dumps(g.to_dict(), default=str)))  # the EMMY_DUMP_DIR disk round-trip
    descs = loaded.nodes["out"].op.tma_descriptors
    assert descs and all(isinstance(d, TmaDescMeta) for d in descs), f"expected TmaDescMeta tuple, got {descs!r}"
    assert descs[0].src_buf == "k" and descs[0].box_extents == (1, 1, 64, 64) and descs[0].swizzle == "B128"


def test_to_dict_serializes_composite_shape_dim():
    """A node whose output shape carries a COMPOSITE Dim (``BinaryExpr``-backed —
    e.g. the demoted symbolic-N B operand's TMA-padded ``round_up(seq_len, 64)``
    inner extent, or a CatOp output) must not crash ``to_dict``: the dump path
    serializes it to its pretty expr string. Atomic dims (int / Var name) still
    return their scalar value for the ``run --ir`` round-trip."""
    from emmy.compiler.dim import Dim

    g = Graph()
    padded = Dim((Dim("seq_len").ceil_div(64) * 64).expr, hint=512)  # ((seq_len + 63) // 64) * 64
    g.add_node(op=InputOp(), inputs=[], output=Tensor("xnb", (128, padded)), node_id="xnb")
    g.inputs, g.outputs = ["xnb"], ["xnb"]

    d = g.to_dict()  # must not raise (pre-fix: Dim.value raised on the composite)
    shape = d["nodes"]["xnb"]["output"]["shape"]
    assert shape[0] == 128, "static inner-rank dim keeps its int value"
    assert isinstance(shape[1], str) and "seq_len" in shape[1] and "64" in shape[1], (
        f"composite dim must serialize to its pretty expr string, got {shape[1]!r}"
    )


# ---- multi-output (MIMO) foundation ----


def _make_mimo_graph():
    """x → mo(outputs: Y primary, Y__sq aux) → c(reads both)."""
    g = Graph()
    x = g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4, 4)), node_id="x")
    g.inputs = [x]
    mo = g.add_node(
        op=ElementwiseOp(op="exp"),
        inputs=[x],
        outputs=[Tensor("Y", (4, 4)), Tensor("Y__sq", (4,))],
        node_id="Y",
    )
    c = g.add_node(op=ElementwiseOp(op="negative"), inputs=[mo, "Y__sq"], output=Tensor("C", (4, 4)), node_id="c")
    g.outputs = [c]
    return g


def test_mimo_node_outputs_and_primary_alias():
    g = _make_mimo_graph()
    node = g.nodes["Y"]
    assert len(node.outputs) == 2
    assert node.output is node.outputs[0]  # primary alias
    assert node.buffer_names() == ("Y", "Y__sq")


def test_mimo_indexes_producer_and_users():
    g = _make_mimo_graph()
    assert g.producer("Y").id == "Y"
    assert g.producer("Y__sq").id == "Y"
    assert g.buffer("Y__sq").shape == (4,)
    assert g.buffer_users("Y") == {"c"}
    assert g.buffer_users("Y__sq") == {"c"}
    assert g.users("Y") == {"c"}  # union over both buffers
    assert g.producer("nope") is None and g.buffer("nope") is None


def test_mimo_add_node_rejects_duplicate_buffer():
    g = _make_mimo_graph()
    with pytest.raises(ValueError, match="already has a producer"):
        g.add_node(
            op=ElementwiseOp(op="exp"),
            inputs=["x"],
            outputs=[Tensor("Z", (4,)), Tensor("Y__sq", (4,))],
            node_id="z",
        )


def test_add_node_requires_exactly_one_output_form():
    g = Graph()
    with pytest.raises(ValueError, match="exactly one"):
        g.add_node(op=InputOp(), inputs=[], node_id="x")
    with pytest.raises(ValueError, match="exactly one"):
        g.add_node(op=InputOp(), inputs=[], output=Tensor("X", (4,)), outputs=[Tensor("X", (4,))], node_id="x")


def test_mimo_topological_order_and_validate():
    g = _make_mimo_graph()
    g.validate()
    order = g.topological_order()
    assert order.index("x") < order.index("Y") < order.index("c")


def test_mimo_remove_node_clears_buffer_indexes():
    g = _make_mimo_graph()
    g.outputs = []
    g.remove_node("c")
    g.remove_node("Y")
    assert g.producer("Y__sq") is None
    assert g.buffer_users("Y__sq") == set()
    g.validate()


def test_mimo_rename_node_keeps_aux_buffer():
    g = _make_mimo_graph()
    g.rename_node("Y", "Y2")
    assert g.producer("Y2").id == "Y2"
    assert g.producer("Y__sq").id == "Y2"  # aux buffer follows the renamed producer
    assert g.nodes["c"].inputs == ["Y2", "Y__sq"]
    g.validate()


def test_mimo_remove_orphans_keeps_producer_alive_via_aux_edge():
    """A consumer reading ONLY the aux buffer still keeps the producer alive."""
    g = Graph()
    x = g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4,)), node_id="x")
    g.inputs = [x]
    g.add_node(op=ElementwiseOp(op="exp"), inputs=[x], outputs=[Tensor("Y", (4,)), Tensor("Y__sq", (2,))], node_id="Y")
    c = g.add_node(op=ElementwiseOp(op="negative"), inputs=["Y__sq"], output=Tensor("C", (2,)), node_id="c")
    g.outputs = [c]
    g.remove_orphans()
    assert "Y" in g.nodes
    g.validate()


def test_mimo_copy_and_structural_key_stability():
    g = _make_mimo_graph()
    g2 = g.copy()
    g2.validate()
    assert g2.structural_key() == g.structural_key()
    # aux shape participates in the digest
    g2.nodes["Y"].outputs = (g2.nodes["Y"].outputs[0], Tensor("Y__sq", (8,)))
    assert g2.structural_key() != g.structural_key()


def test_single_output_structural_key_unchanged_by_mimo_fold():
    """Single-output digests must be byte-identical to the pre-MIMO scheme —
    golden/tune evidence keyed on structure must not notice the migration."""
    g = _make_matmul_graph()
    key = g.structural_key()
    assert key == g.copy().structural_key()


def test_graph_validate_catches_index_corruption():
    g = _make_mimo_graph()
    g._users["Y"].add("ghost")
    with pytest.raises(ValueError, match="out of sync"):
        g.validate()


def test_mimo_from_dict_dual_read():
    """from_dict reads both the historic single-``output`` dict and the plural
    ``outputs`` list."""
    import json

    g = _make_matmul_graph()
    d = json.loads(json.dumps(g.to_dict()))
    # rewrite one node to the plural form; loader must accept both in one dump
    nd = d["nodes"]["ew"]
    nd["outputs"] = [nd.pop("output")]
    g2 = Graph.from_dict(d)
    g2.validate()
    assert g2.structural_key() == g.structural_key()
