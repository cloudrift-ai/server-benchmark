"""The Tile rewrite recognizes the whole exp-family through one algebraic path."""

from __future__ import annotations

from emmy.commands.trace import graph_from_code
from emmy.compiler.ir.axis import AxisRole
from emmy.compiler.ir.pure import Fold
from emmy.compiler.ir.stmt import Load, Select
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import LOOP_PASSES, Pipeline


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
    assert len(fold.operands) == 1 and isinstance(fold.operands[0], Load)
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
