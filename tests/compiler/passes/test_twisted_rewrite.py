"""The Tile rewrite recognizes the whole exp-family through one algebraic path."""

from __future__ import annotations

import pytest

from emmy.commands.trace import graph_from_code
from emmy.compiler.context import Context
from emmy.compiler.ir.axis import AxisRole
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.ir.pure import Fold
from emmy.compiler.ir.stmt import Load, Select
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import CUDA_PASSES, LOOP_PASSES, Pipeline


def _folds(root: Fold):
    yield root
    for edge in root.operands:
        if isinstance(edge, Fold):
            yield from _folds(edge)
    for stmt in root.lift.body:
        if isinstance(stmt, Fold):
            yield from _folds(stmt)


def _twisted(code: str) -> Fold:
    graph, _, _ = graph_from_code(code)
    graph = Pipeline.build(LOOP_PASSES).run(graph)
    graph = Pipeline.build(["lowering/tile"], select=["lift", "twisted"]).run(graph)
    tile = next(node.op for node in graph.nodes.values() if isinstance(node.op, TileOp))
    matches = [fold for fold in _folds(tile.op) if fold.role is AxisRole.TWISTED]
    assert len(matches) == 1
    return matches[0]


def test_softmax_rewrites_to_twisted_pair() -> None:
    fold = _twisted("torch.softmax(torch.randn(4, 8, dtype=torch.float16), dim=-1)")

    assert len(fold.init) == 2
    assert not fold.operands


def test_sdpa_rewrites_to_twisted_expectation() -> None:
    fold = _twisted(
        "F.scaled_dot_product_attention("
        "torch.randn(1, 1, 4, 2, dtype=torch.float16), "
        "torch.randn(1, 1, 4, 2, dtype=torch.float16), "
        "torch.randn(1, 1, 4, 2, dtype=torch.float16))"
    )

    assert len(fold.init) == 3
    assert sum(isinstance(edge, Load) for edge in fold.operands) == 1
    assert sum(isinstance(stmt, Fold) and stmt.role is AxisRole.CONTRACTION for stmt in fold.step_stmts()) == 2


def test_causal_sdpa_uses_the_same_twisted_rewrite() -> None:
    fold = _twisted(
        "F.scaled_dot_product_attention("
        "torch.randn(1, 1, 4, 2, dtype=torch.float16), "
        "torch.randn(1, 1, 4, 2, dtype=torch.float16), "
        "torch.randn(1, 1, 4, 2, dtype=torch.float16), is_causal=True)"
    )

    assert len(fold.init) == 3
    assert any(isinstance(stmt, Select) for stmt in fold.lift.body)
    assert sum(isinstance(stmt, Fold) and stmt.role is AxisRole.CONTRACTION for stmt in fold.step_stmts()) == 2


@pytest.mark.parametrize("causal", [False, True])
def test_sdpa_fold_tree_reaches_both_mma_sites(monkeypatch, causal: bool) -> None:
    suffix = ", is_causal=True" if causal else ""
    graph, _, _ = graph_from_code(
        "F.scaled_dot_product_attention("
        "torch.randn(1, 1, 32, 16, dtype=torch.float16), "
        "torch.randn(1, 1, 32, 16, dtype=torch.float16), "
        f"torch.randn(1, 1, 32, 16, dtype=torch.float16){suffix})"
    )
    monkeypatch.setenv("EMMY_WORK", "w1x1")
    monkeypatch.setenv("EMMY_TILE", "mma_m16n8k16_f16_f32/f1x2")
    monkeypatch.setenv("EMMY_REDUCE", "")

    lowered = Pipeline.build(CUDA_PASSES).run(graph, ctx=Context.from_target((8, 0)))
    (source,) = (node.op.kernel_source for node in lowered.nodes.values() if isinstance(node.op, CudaOp))

    assert source.count("emmy_mma_m16n8k16_f16_f32(") >= 3  # helper plus both contraction sites
    assert "__shfl_xor_sync" in source
    if causal:
        assert "?" in source
