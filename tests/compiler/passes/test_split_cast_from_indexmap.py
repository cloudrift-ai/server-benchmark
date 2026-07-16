"""Tests for ``frontend/optimization/005_split_cast_from_indexmap`` and the TMA
operand-rank rule it unlocks staging through.

A traced view that also narrows dtype (gemma's f32 ``mul`` → f16 ``transpose`` feeding
SDPA) must split into a source-shaped elementwise ``copy`` (the cast — real compute, so
the flash-operand materialization guards keep it a buffer) plus a dtype-preserving
IndexMapOp (plumbing that keeps composing/splicing). Without the split, the cast rides
the view into the consumer kernel and the operand-staging gates see the wide source
buffer (the 2026-07 gemma layer-0 flash lockout).
"""

import numpy as np

from emmy.compiler.backend.numpy import NumpyBackend
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var, placeholder
from emmy.compiler.ir.tensor.ir import ElementwiseOp, IndexMapOp, IndexSource
from emmy.compiler.pipeline import Pipeline
from emmy.compiler.pipeline.passes.lowering.tile._schedule import _tma_operand_rank_ok

rng = np.random.default_rng(7)
_backend = NumpyBackend()


def _transpose_graph(out_dtype: str) -> Graph:
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (2, 3), "f32"), node_id="x")
    g.add_node(
        IndexMapOp(out_shape=(3, 2), sources=(IndexSource(input_idx=0, coord_map=(placeholder(1), placeholder(0))),)),
        ["x"],
        Tensor("xt", (3, 2), out_dtype),
        node_id="xt",
    )
    g.inputs = ["x"]
    g.outputs = ["xt"]
    return g


def _optimize(graph: Graph) -> Graph:
    return Pipeline.build(["frontend/optimization"]).run(graph)


def test_dtype_preserving_indexmap_is_untouched():
    g = _optimize(_transpose_graph("f32"))
    assert not any("_cast" in nid for nid in g.nodes)


def test_cast_splits_out_at_source_shape():
    g = _optimize(_transpose_graph("f16"))
    casts = [n for n in g.nodes.values() if isinstance(n.op, ElementwiseOp) and n.op.op.name == "copy"]
    assert len(casts) == 1, f"expected one split-out cast copy, got {[n.id for n in casts]}"
    cast = casts[0]
    assert tuple(cast.output.shape) == (2, 3), "cast materializes at the SOURCE shape, not the view's"
    assert cast.output.dtype.name == "f16"
    # The residual view is dtype-preserving plumbing again.
    xt = g.nodes["xt"]
    assert isinstance(xt.op, IndexMapOp)
    assert g.nodes[xt.inputs[0]].id == cast.id
    assert xt.output.dtype.name == "f16"


def test_split_preserves_values():
    x = rng.standard_normal((2, 3), dtype=np.float32)
    before = _backend.run(_backend.compile(_transpose_graph("f16")), input_data={"x": x})[0].outputs
    after = _backend.run(_backend.compile(_optimize(_transpose_graph("f16"))), input_data={"x": x})[0].outputs
    np.testing.assert_allclose(list(after.values())[0], list(before.values())[0], rtol=0, atol=0)


# ===================================================================
# _tma_operand_rank_ok — the matmul-tier TMA box rank rule
# ===================================================================

_M, _K = Var("m"), Var("k")


def test_tma_rank_plain_2d_ok():
    assert _tma_operand_rank_ok((_M, _K), "m", "k")


def test_tma_rank_leading_literal_ok():
    assert _tma_operand_rank_ok((Literal(0, "int"), _M, _K), "m", "k")


def test_tma_rank_leading_batch_var_ok():
    assert _tma_operand_rank_ok((Var("b"), _M, _K), "m", "k")


def test_tma_rank_leading_dim_moving_with_k_declines():
    lead = BinaryExpr("+", Var("b"), Var("k"))
    assert not _tma_operand_rank_ok((lead, _M, _K), "m", "k")


def test_tma_rank_leading_dim_moving_with_tile_declines():
    assert not _tma_operand_rank_ok((Var("m"), _M, _K), "m", "k")


def test_tma_rank_1d_and_5d_decline():
    assert not _tma_operand_rank_ok((_K,), "m", "k")
    assert not _tma_operand_rank_ok((Var("a"), Var("b"), Var("c"), _M, _K), "m", "k")
