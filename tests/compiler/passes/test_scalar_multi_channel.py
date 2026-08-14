"""Scalar register tiling of product-monoid contractions.

The scalar backend is the correctness fallback for a fused sibling group when a hardware MMA
geometry is not selected.  It must therefore preserve every B/accumulator channel rather than
assuming the contraction has arity one.
"""

from __future__ import annotations

from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F16
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
from emmy.compiler.pipeline import CUDA_PASSES, Pipeline

VOLTA = "mma_m8n8k4_f16_f32"
AMPERE = "mma_m16n8k16_f16_f32"


def _two_channel_graph() -> Graph:
    m, n, k = (Axis(name, Dim(extent)) for name, extent in (("m", 16), ("n", 16), ("k", 16)))
    step = Body(
        (
            Load(name="a", input="x", index=(Var("m"), Var("k"))),
            Load(name="b0", input="wg", index=(Var("n"), Var("k"))),
            Assign(name="p0", op="multiply", args=("a", "b0")),
            Accum(name="acc0", value="p0", op="add", axes=("k",)),
            Load(name="b1", input="wu", index=(Var("n"), Var("k"))),
            Assign(name="p1", op="multiply", args=("a", "b1")),
            Accum(name="acc1", value="p1", op="add", axes=("k",)),
        )
    )
    cell = Body(
        (
            Loop(axis=k, body=step, role=AxisRole.CONTRACTION),
            Assign(name="out", op="multiply", args=("acc0", "acc1")),
            Write(output="y", index=(Var("m"), Var("n")), value="out"),
        )
    )
    op = LoopOp(body=Body((Loop(axis=m, body=Body((Loop(axis=n, body=cell),))),)))

    graph = Graph()
    for name, shape in (("x", (16, 16)), ("wg", (16, 16)), ("wu", (16, 16))):
        graph.add_node(InputOp(), [], Tensor(name, shape, F16), node_id=name)
    graph.add_node(op, ["x", "wg", "wu"], Tensor("y", (16, 16), F16), node_id="y")
    graph.inputs, graph.outputs = ["x", "wg", "wu"], ["y"]
    return graph


def test_scalar_tile_emits_every_product_channel(monkeypatch) -> None:
    """The f4 fallback declares, folds, and consumes both gate/up accumulators."""
    monkeypatch.delenv("EMMY_KNOBS", raising=False)
    monkeypatch.setenv("EMMY_TILE", "f4")
    monkeypatch.setenv("EMMY_WORK", "")
    monkeypatch.setenv("EMMY_STAGE", "")
    monkeypatch.setenv("EMMY_REDUCE", "")

    out = Pipeline.build(CUDA_PASSES).run(_two_channel_graph(), ctx=Context(compute_capability=(7, 0)))
    kernel = out.nodes["y"].op
    src = kernel.kernel_source
    assert kernel.knobs["TILE"] == "f4"
    for cell in range(4):
        assert f"float acc0__c0_{cell} = 0.0f;" in src
        assert f"float acc1__c0_{cell} = 0.0f;" in src
        assert f"acc0__c0_{cell} +=" in src
        assert f"acc1__c0_{cell} +=" in src
        assert f"acc0__c0_{cell} * acc1__c0_{cell}" in src
    assert src.count("= x[") == 1, "the shared A row load remains reused across all channels and columns"


def _warp_source(monkeypatch, *, cc: tuple[int, int], atom: str, stage: str) -> str:
    monkeypatch.delenv("EMMY_KNOBS", raising=False)
    monkeypatch.setenv("EMMY_TILE", f"{atom}/f1x1/k1")
    monkeypatch.setenv("EMMY_WORK", "w1x1")
    monkeypatch.setenv("EMMY_STAGE", stage)
    monkeypatch.setenv("EMMY_REDUCE", "")
    out = Pipeline.build(CUDA_PASSES).run(_two_channel_graph(), ctx=Context(compute_capability=cc))
    assert out.nodes["y"].op.knobs["STAGE"] == stage
    return out.nodes["y"].op.kernel_source


def test_multichannel_gmem_direct_mma_reuses_a_fragment(monkeypatch) -> None:
    """The fragment register file supports several B/C chains without forcing a transport."""
    src = _warp_source(monkeypatch, cc=(7, 0), atom=VOLTA, stage="")
    assert "emmy_mma884_load_a_gmem(_a0" in src
    assert "emmy_mma884_load_b_gmem_trans(_b0" in src
    assert "emmy_mma884_load_b_gmem_trans(_b0_x1" in src
    assert "emmy_mma_m8n8k4_f16_f32(_c0_0" in src
    assert "emmy_mma_m8n8k4_f16_f32(_c0_0_x1" in src


def test_multichannel_volta_sync_copy_stages_every_b(monkeypatch) -> None:
    src = _warp_source(monkeypatch, cc=(7, 0), atom=VOLTA, stage="d1/sync")
    assert "_a_smem" in src and "_b_smem" in src and "_b_x1_smem" in src
    assert "emmy_mma884_load_b_smem_trans(_b0_x1" in src
    assert "cp.async" not in src


def test_multichannel_cp_async_stages_every_b(monkeypatch) -> None:
    src = _warp_source(monkeypatch, cc=(8, 0), atom=AMPERE, stage="d1/cp")
    assert "_a_smem" in src and "_b_smem" in src and "_b_x1_smem" in src
    assert "cp.async.ca.shared.global" in src
    assert "emmy_mma_m16n8k16_f16_f32(_c0_0_x1" in src


def test_multichannel_tma_stages_every_b(monkeypatch) -> None:
    src = _warp_source(monkeypatch, cc=(9, 0), atom=AMPERE, stage="d1/tma")
    assert "_a_smem" in src and "_b_smem" in src and "_b_x1_smem" in src
    assert "cp.async.bulk.tensor" in src
    assert "emmy_mma_m16n8k16_f16_f32(_c0_0_x1" in src
