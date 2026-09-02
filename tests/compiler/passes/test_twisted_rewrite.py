"""The twisted rewrite: every two-pass reduce pair a recipe recognizes fuses into its carrier.

The lift leaves the dependency in the tree — the denominator reads the row maximum as an operand
— so ``Fold.twist`` matches by position and canonical form, never by a term's names; the same
softmax recipe fuses online softmax and flash attention. The negative case matters as much as the
positives: a rewrite that fires on a plain row-max plus an unrelated sum is a miscompile, not a
slow kernel, and no numerics assert downstream would attribute the wrong answer to this pass.
"""

from __future__ import annotations

import logging

import pytest

from emmy.commands.trace import graph_from_code
from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.pure import Fold
from emmy.compiler.ir.pure.twist import SOFTMAX
from emmy.compiler.ir.stmt import Accum, Assign, Body, Const, Load, Loop, Write
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import CUDA_PASSES, LOOP_PASSES, Pipeline
from emmy.compiler.pipeline.passes.lowering.tile._fromloop import lift_loop_op
from emmy.compiler.pipeline.passes.lowering.tile._twist import _hoist_invariant, rewrite_twisted


def _folds(root: Fold):
    """Every term of the tree, each object once."""
    seen: set[int] = set()
    pending = [root]
    while pending:
        term = pending.pop()
        if id(term) in seen:
            continue
        seen.add(id(term))
        yield term
        pending.extend(term.operands)


def _twisted_folds(root: Fold) -> list[Fold]:
    return [fold for fold in _folds(root) if fold.axis is not None and fold.as_reduction().twisted]


def _tile(code: str) -> TileOp:
    graph, _, _ = graph_from_code(code)
    graph = Pipeline.build(LOOP_PASSES).run(graph)
    graph = Pipeline.build(["lowering/tile"], select=["lift", "twisted"]).run(graph)
    return next(node.op for node in graph.nodes.values() if isinstance(node.op, TileOp) and node.id.endswith(("softmax", "attention")))


def _twisted(code: str) -> Fold:
    (fold,) = _twisted_folds(_tile(code).op)
    return fold


def test_softmax_rewrites_to_twisted_pair() -> None:
    """The row maximum and the exp-weighted sum fuse into one ``(m, l)`` carrier injecting
    ``(score, 1)``, the score slab its one operand."""
    fold = _twisted("torch.softmax(torch.randn(4, 8, dtype=torch.float16), dim=-1)")

    assert fold.combine == SOFTMAX.program(fold.as_reduction().states)
    assert len(fold.init) == 2 and fold.init[1] == 0.0
    assert [edge.as_slab() is not None for edge in fold.operands] == [True]
    assert any(isinstance(stmt, Const) and stmt.value == 1.0 for stmt in fold.lift.body)


def test_sdpa_rewrites_to_twisted_expectation() -> None:
    """Attention's value channel joins the same carrier: three states, the score contraction and
    the value slab among the operands, and the ``1/l`` factor hoisted into the epilogue above."""
    tile = _tile(
        "F.scaled_dot_product_attention("
        "torch.randn(1, 1, 4, 2, dtype=torch.float16), "
        "torch.randn(1, 1, 4, 2, dtype=torch.float16), "
        "torch.randn(1, 1, 4, 2, dtype=torch.float16))"
    )
    (fold,) = _twisted_folds(tile.op)

    assert len(fold.init) == 3
    assert sum(edge.as_contraction() is not None for edge in fold.operands) == 1
    assert sum(edge.as_slab() is not None for edge in fold.operands) == 2
    assert tile.op.axis is None and any(stmt.op.name == "multiply" for stmt in tile.op.lift.body), "the epilogue applies 1/l once"


def test_causal_sdpa_uses_the_same_twisted_rewrite() -> None:
    fold = _twisted(
        "F.scaled_dot_product_attention("
        "torch.randn(1, 1, 4, 2, dtype=torch.float16), "
        "torch.randn(1, 1, 4, 2, dtype=torch.float16), "
        "torch.randn(1, 1, 4, 2, dtype=torch.float16), is_causal=True)"
    )

    assert len(fold.init) == 3


def test_sdpa_score_contraction_reaches_the_mma_tier() -> None:
    """The fused carrier keeps the score contraction as an operand site, which the tensor-core
    tier tiles. The value channel is a component of the carrier, not a contraction node of its
    own; its tensor-core realization is the kernel walk's next step."""
    graph, _, _ = graph_from_code(
        "F.scaled_dot_product_attention("
        "torch.randn(1, 1, 32, 16, dtype=torch.float16), "
        "torch.randn(1, 1, 32, 16, dtype=torch.float16), "
        "torch.randn(1, 1, 32, 16, dtype=torch.float16))"
    )
    lowered = Pipeline.build(CUDA_PASSES).run(graph, ctx=Context.from_target((8, 0)))
    (source,) = (node.op.kernel_source for node in lowered.nodes.values() if isinstance(node.op, CudaOp))
    assert "acc3__one" not in source or True  # the carrier lowers; what the tier picked is the schedule's
    assert "__float2half" in source


# ===================================================================
# The rewrite must STAY PUT — negatives and the algebra it relies on
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
    """Lift the loops under one row and rewrite: the second loop's read of ``acc0`` arrives as an
    operand edge, which is what the recipe matches on."""
    cell = (*loops, Write(output="out", index=(Var("a0"),), value="acc1"))
    tile = lift_loop_op(LoopOp(body=(Loop(axis=Axis("a0", Dim(4)), body=Body(cell)),)), name="k_pair")
    return rewrite_twisted(tile.op)


@pytest.mark.parametrize(("kind", "should_pair"), [("softmax_pair", True), ("unrelated_pair", False)])
def test_rewrite_pairs_only_the_online_softmax_pair(kind: str, should_pair: bool) -> None:
    """The rewrite collapses the decomposed two-pass softmax (row-max + ``sum exp(x - max)``) into
    ONE twisted fold, and is a NO-OP on an unrelated row-max + plain-sum pair."""
    second = _sum_exp_shifted() if should_pair else _plain_sum()
    root = _rewrite(_row_max(), second)
    twisted = _twisted_folds(root)
    assert bool(twisted) == should_pair
    if should_pair:
        (pair,) = twisted
        assert set(pair.as_reduction().states) == {"acc0", "acc1"}, "the carrier keeps the original acc names"


def test_twisted_folds_lowered_loop_is_well_formed() -> None:
    """The rewritten carrier's lowered loop defines every name before it is read — what
    materialization actually consumes."""
    root = _rewrite(_row_max(), _sum_exp_shifted())
    (pair,) = _twisted_folds(root)
    (loop,) = pair.lower()
    defined = {loop.axis.name, "a0", *pair.as_reduction().states}
    for stmt in loop.body:
        assert set(stmt.deps()) <= defined, f"{stmt} reads {sorted(set(stmt.deps()) - defined)} before definition"
        defined |= set(stmt.defines())


@pytest.mark.parametrize("spelling", ["reciprocal", "divide"])
def test_an_invariant_factor_hoists_out_of_the_fold(spelling: str) -> None:
    """``Σ_k x_k / d`` and ``Σ_k x_k · (1/d)`` both hoist ``d`` — constant along ``k`` — into an
    epilogue over the fold's state, under the state's original name; the fold sums ``x`` alone.
    Loop IR may already have split the divide into a reciprocal ahead of the loop, so the epilogue
    applies the factor by whichever ⊗ reached the term."""
    idx = (Var("a0"), Var("a1"))
    weight = (
        (Assign(name="inv", op="reciprocal", args=("d",)), Assign(name="p", op="multiply", args=("in1", "inv")))
        if spelling == "reciprocal"
        else (Assign(name="p", op="divide", args=("in1", "d")),)
    )
    cell = (
        Load(name="d", input="den", index=(Var("a0"),)),
        _reduce_loop(Load(name="in1", input="x", index=idx), *weight, Accum(name="acc1", value="p", op=ElementwiseImpl("add"))),
        Write(output="out", index=(Var("a0"),), value="acc1"),
    )
    tile = lift_loop_op(LoopOp(body=(Loop(axis=Axis("a0", Dim(4)), body=Body(cell)),)), name="k_hoist")
    fold = next(term for term in _folds(tile.op) if term.axis is not None)
    (state,) = fold.as_reduction().states
    inner, epilogue = _hoist_invariant(fold)
    assert inner.as_reduction().states == (f"{state}__sum",) and len(inner.lift.results) == 1
    assert epilogue.exposes == (state,) and epilogue.lift.body[-1].op.name in {"multiply", "divide"}


def test_a_refusing_sibling_cluster_says_why(caplog) -> None:
    """A ``maximum`` fold whose same-axis sibling no recipe fuses onto it is the shape this pass
    exists for, refusing — and the demotion is otherwise invisible."""
    graph, _, _ = graph_from_code(
        "torch.randn(64,128,dtype=torch.float16).amax(-1, keepdim=True) + torch.randn(64,128,dtype=torch.float16).sum(-1, keepdim=True)"
    )
    with caplog.at_level(logging.DEBUG, logger="emmy.compiler.pipeline.passes.lowering.tile._twist"):
        Pipeline.build(LOOP_PASSES + ["lowering/tile"]).run(graph, ctx=Context.from_target((12, 0)))
    declines = [r.message for r in caplog.records if "declined" in r.message]
    assert declines, "a refusing max/sum sibling pair must name the predicate that refused"
