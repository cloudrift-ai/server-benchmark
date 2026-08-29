"""The deploy join key — ``Op.deploy_identity`` over the schedule-free lowered Loop-IR body.

What these pin: (a) the key hashes the canonical lowered body, not the term — kernels whose
epilogues differ only within one compute-unit op cluster (``relu`` vs ``tanh``) share the key
while their term hashes differ, because their schedule evidence transfers; (b) DISCRIMINATION —
a cross-cluster op change moves the key, and so does an extent change (the extent fingerprint,
folded in beside the body digest)."""

from __future__ import annotations

from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
from emmy.compiler.pipeline import Pipeline


def _tile(body: Body):
    graph = Graph()
    graph.add_node(LoopOp(body=body), [], Tensor("out", (1,)), node_id="out")
    graph.outputs = ["out"]
    return Pipeline.build(["lowering/tile"], select=["lift"]).run(graph).nodes["out"].op


def _matmul_tile(epilogue_op: str | None = None, k_extent: int = 128):
    m, n, k = Axis("m", Dim(32)), Axis("n", Dim(64)), Axis("k", Dim(k_extent))
    inner = Body(
        (
            Load(name="xv", input="x", index=(Var("m"), Var("k"))),
            Load(name="wv", input="w", index=(Var("n"), Var("k"))),
            Assign(name="prod", op=ElementwiseImpl("multiply"), args=("xv", "wv")),
            Accum(name="acc", value="prod", op=ElementwiseImpl("add"), axes=("k",)),
        )
    )
    result = "acc"
    epilogue = ()
    if epilogue_op is not None:
        epilogue = (Assign(name="outv", op=ElementwiseImpl(epilogue_op), args=("acc",)),)
        result = "outv"
    cell = (Loop(axis=k, body=inner), *epilogue, Write(output="out", index=(Var("m"), Var("n")), value=result))
    return _tile(Body((Loop(axis=m, body=Body((Loop(axis=n, body=Body(cell)),))),)))


def test_cluster_sibling_epilogues_share_the_structural_key_but_not_the_exact_one() -> None:
    relu, tanh = _matmul_tile("relu"), _matmul_tile("tanh")
    assert relu.body_identity(structural=False) != tanh.body_identity(structural=False)
    assert relu.deploy_identity() == tanh.deploy_identity(), "schedule evidence transfers within a cluster"
    assert relu.deploy_identity(structural=False) != tanh.deploy_identity(structural=False), "the exact kernels differ"


def test_cross_cluster_epilogues_key_apart() -> None:
    assert _matmul_tile("relu").deploy_identity() != _matmul_tile("abs").deploy_identity()


def test_extent_moves_the_key() -> None:
    assert _matmul_tile(k_extent=128).deploy_identity() != _matmul_tile(k_extent=64).deploy_identity()
