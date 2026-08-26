"""The Tile rewrite recognizes the whole exp-family through one algebraic path.

The negative case and the two algebraic helpers below were deleted with
``tests/compiler/passes/test_online_softmax_channels.py`` and are RESTORED here, which is the
successor home. The positives above only prove the rewrite FIRES; nothing proved it stays put. An
exp-family rewrite that fires on a plain row-max plus an unrelated sum is a miscompile, not a slow
kernel, and no numerics assert downstream would attribute the wrong answer to this pass.
"""

from __future__ import annotations

import pytest

from emmy.commands.trace import graph_from_code
from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.pure import Fold
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Select
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.ir.tile.ops import split_invariant_factors
from emmy.compiler.pipeline import CUDA_PASSES, LOOP_PASSES, Pipeline
from emmy.compiler.pipeline.passes.lowering.tile._fromloop import _stamp_axes, fold_from_loop
from emmy.compiler.pipeline.passes.lowering.tile._twist import rewrite_twisted


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
    monkeypatch.setenv("EMMY_PLACE", "fuse")

    lowered = Pipeline.build(CUDA_PASSES).run(graph, ctx=Context.from_target((8, 0)))
    (source,) = (node.op.kernel_source for node in lowered.nodes.values() if isinstance(node.op, CudaOp))

    assert source.count("emmy_mma_m16n8k16_f16_f32(") >= 3  # helper plus both contraction sites
    assert "__shfl_xor_sync" in source
    if causal:
        assert "?" in source


# ===================================================================
# The rewrite must STAY PUT — restored negatives and algebraic helpers
# ===================================================================


def _reduce_loop(*stmts, axis: str = "a1", extent: int = 128) -> Loop:
    return Loop(axis=Axis(name=axis, extent=Dim(extent)), body=Body(tuple(stmts)))


def _row_max() -> Loop:
    idx = (Var("a0"), Var("a1"))
    return _reduce_loop(
        Load(name="in0", input="x", index=idx),
        Accum(name="acc0", value="in0", op=ElementwiseImpl("maximum")),
    )


def _sum_exp_shifted() -> Loop:
    idx = (Var("a0"), Var("a1"))
    return _reduce_loop(
        Load(name="in1", input="x", index=idx),
        Assign(name="v0", op="subtract", args=("in1", "acc0")),
        Assign(name="v1", op="exp", args=("v0",)),
        Accum(name="acc1", value="v1", op=ElementwiseImpl("add")),
    )


def _plain_sum() -> Loop:
    idx = (Var("a0"), Var("a1"))
    return _reduce_loop(
        Load(name="in1", input="x", index=idx),
        Accum(name="acc1", value="in1", op=ElementwiseImpl("add")),
    )


def _rewrite(*loops: Loop) -> Fold:
    folds = tuple(fold_from_loop(_stamp_axes(loop)) for loop in loops)
    assert all(fold is not None for fold in folds)
    return rewrite_twisted(Fold.projection(body=Body(folds)), ("a0",))


@pytest.mark.parametrize(("kind", "should_pair"), [("softmax_pair", True), ("unrelated_pair", False)])
def test_rewrite_pairs_only_the_online_softmax_pair(kind: str, should_pair: bool) -> None:
    """The rewrite collapses the decomposed two-pass softmax (row-max + ``sum exp(x - max)``) into
    ONE twisted fold, and is a NO-OP on an unrelated row-max + plain-sum pair.

    RESTORED: the second half is the one that matters. Pairing two folds that are not an
    exp-family pair rewrites the program into a different program."""
    second = _sum_exp_shifted() if should_pair else _plain_sum()
    root = _rewrite(_row_max(), second)
    twisted = [fold for fold in _folds(root) if fold.role is AxisRole.TWISTED]
    assert bool(twisted) == should_pair
    if should_pair:
        (pair,) = twisted
        assert set(pair.combine.results) == {"acc0", "acc1"}, "the carrier keeps the original acc names"


def test_split_invariant_factors_reads_the_product() -> None:
    """``sum_k c*x_k = c*sum_k x_k``: the loop-invariant leaves split off the multiply spine, the
    loop-varying ones stay. RESTORED because the helper is exported and currently has no other
    caller or test in the tree — an unexercised algebraic license is one refactor from silently
    returning ``None`` and costing every fold that depends on it."""
    body = [
        Load(name="xk", input="x", index=(Var("k"),)),
        Assign(name="p", op="multiply", args=("c", "xk")),
    ]
    assert split_invariant_factors(body, "p", "k") == (("c",), ("xk",))

    # A bare leaf is the degenerate product.
    assert split_invariant_factors([Load(name="xk", input="x", index=(Var("k"),))], "xk", "k") == ((), ("xk",))

    # The loop axis itself is loop-varying, never an invariant factor.
    axis_body = [Assign(name="p", op="multiply", args=("c", "k"))]
    assert split_invariant_factors(axis_body, "p", "k") == (("c",), ("k",))

    # A spine temp read by another statement is not private to the product — decline, do not split.
    shared = [
        Load(name="xk", input="x", index=(Var("k"),)),
        Assign(name="inner", op="multiply", args=("c", "xk")),
        Assign(name="p", op="multiply", args=("inner", "d")),
        Assign(name="other", op="add", args=("inner", "xk")),
    ]
    assert split_invariant_factors(shared, "p", "k") is None


def test_twisted_folds_derived_loop_is_well_formed() -> None:
    """The rewritten carrier's DERIVED loop defines every name before it is read.

    The restored version of this test asserted the stronger closure — that ``Fold.loop`` re-lifts
    to the same node — because cut and split pieces once round-tripped through the loop dialect.
    They no longer do: pieces are minted structurally in Tile IR, and measured against the current
    tree NO fold round-trips (a twisted carrier's derived loop is not canonical Loop IR at all, and
    even a plain contraction's re-lifts to a different structural key). Asserting the old closure
    would pin a retired guarantee, so what is pinned is what materialization actually consumes."""
    root = _rewrite(_row_max(), _sum_exp_shifted())
    (pair,) = [fold for fold in _folds(root) if fold.role is AxisRole.TWISTED]
    loop = pair.loop
    defined = {loop.axis.name, "a0", *pair.defines(), *pair.combine.results}  # axes and carried state
    for stmt in loop.body:
        reads = set() if isinstance(stmt, Fold) else set(stmt.deps())
        assert reads <= defined, f"{stmt} reads {sorted(reads - defined)} before definition"
        defined |= set(stmt.defines())
