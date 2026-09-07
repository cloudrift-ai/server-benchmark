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
from emmy.compiler.ir.pure.twist import SOFTMAX, WELFORD
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
    """The row maximum and the exp-weighted sum fuse into one ``(m, l)`` carrier over the BASE
    monoid: the lift contributes ``(score, exp(score))`` off the score slab, and ψ takes that
    singleton to ``(score, 1)`` for the step to fold."""
    fold = _twisted("torch.softmax(torch.randn(4, 8, dtype=torch.float16), dim=-1)")

    assert fold.twist.recipe is SOFTMAX and fold.combine == SOFTMAX.program(fold.as_reduction().states)
    assert len(fold.init) == 2 and fold.init[1] == 0.0
    assert [edge.as_slab() is not None for edge in fold.operands] == [True], "the score slab is its one operand"
    assert [stmt.op.name for stmt in fold.lift.body] == ["exp"], "the base contribution is (score, exp score)"
    assert any(isinstance(stmt, Const) and stmt.value == 1.0 for stmt in fold.injected.body), "psi injects 1"


def test_sdpa_rewrites_to_twisted_expectation() -> None:
    """Attention's value channel joins the same carrier, and the carrier comes out A × B: the
    weight cone leads, the value slab is the streamed operand, and the score contraction sits under
    the cone — one node, not one per binder. The ``1/l`` factor hoists into the epilogue above."""
    tile = _tile(
        "F.scaled_dot_product_attention("
        "torch.randn(1, 1, 4, 2, dtype=torch.float16), "
        "torch.randn(1, 1, 4, 2, dtype=torch.float16), "
        "torch.randn(1, 1, 4, 2, dtype=torch.float16))"
    )
    (fold,) = _twisted_folds(tile.op)

    assert len(fold.init) == 3
    cone, streamed = fold.operands
    assert cone.axis is None and streamed.as_slab() is not None
    assert sum(edge.as_contraction() is not None for edge in cone.operands) == 1, "the one score node"
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
    sources = [node.op.kernel_source for node in lowered.nodes.values() if isinstance(node.op, CudaOp)]
    # The kernel that writes the f16 output converts at the boundary. Asked of the FINALIZE, not of
    # the set: a cross-CTA split's partial keeps the carrier in an f32 workspace and converts nothing.
    assert sources and "__float2half" in sources[-1]


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


def _sum_x() -> Loop:
    idx = (Var("a0"), Var("a1"))
    return _reduce_loop(Load(name="in0", input="x", index=idx), Accum(name="acc0", value="in0", op=ElementwiseImpl("add")))


def _mean_of_sum() -> tuple:
    # ``1/N`` is a scalar input, as the frontend spells a constant in Loop IR: a zero-index load.
    return (Load(name="inv_n", input="inv_n", index=()), Assign(name="mean", op="multiply", args=("acc0", "inv_n")))


def _sum_sq_dev() -> Loop:
    idx = (Var("a0"), Var("a1"))
    return _reduce_loop(
        Load(name="in1", input="x", index=idx),
        Assign(name="v0", op="subtract", args=("in1", "mean")),
        Assign(name="v1", op="multiply", args=("v0", "v0")),
        Accum(name="acc1", value="v1", op=ElementwiseImpl("add")),
    )


def test_welford_variance_pair_fuses_into_one_carrier() -> None:
    """The two-pass variance — a sum, its mean, then the sum of squared deviations from it — fuses
    by the Welford recipe into ONE fold carrying ``(sum, count, mean, M2)``: the pivot sits behind
    the mean's projection, the ``1/N`` inside the deviation is a scalar operand the pattern binds
    as an extra, and the count and running mean are states the two-pass form never had."""
    root, axes = _rewrite(_sum_x(), *_mean_of_sum(), _sum_sq_dev())
    (fold,) = _twisted_folds(root)
    view = fold.as_reduction()
    assert view.states == ("acc0", "acc1__n", "acc1__mean", "acc1")
    assert fold.combine.alpha_eq(WELFORD.program(view.states))
    assert fold.init == (0.0, 0.0, 0.0, 0.0)
    score, one, mean, square = fold.lift.results
    consts = {stmt.name: stmt.value for stmt in fold.lift.body if isinstance(stmt, Const)}
    products = {stmt.name: stmt.args for stmt in fold.lift.body if isinstance(stmt, Assign) and stmt.op.name == "multiply"}
    assert mean == score and consts[one] == 1.0, "the base contribution is (x, 1, x, x*x)"
    assert products[square] == (score, score), "channel 3 squares one edge, so it is no contraction"
    injected = {stmt.name: stmt.value for stmt in fold.injected.body if isinstance(stmt, Const)}
    assert injected[fold.injected.results[3]] == 0.0, "psi takes it to 0 — a lone element deviates from its own mean by nothing"
    lowered = fold.lower(axes=axes)
    (loop,) = [stmt for stmt in lowered if isinstance(stmt, Loop)]  # ``1/N`` is hoisted ahead of it
    defined = {loop.axis.name, "a0", *view.states, *(name for stmt in lowered for name in stmt.defines())}
    for stmt in loop.body:
        assert set(stmt.deps()) <= defined, f"{stmt} reads a name not yet defined"
        defined |= set(stmt.defines())


def test_welford_declines_a_deviation_that_is_not_squared() -> None:
    """A sum followed by ``Σ (x − mean)`` is not the variance's second pass: no channel of the
    recipe matches, and the pair stays two folds."""
    idx = (Var("a0"), Var("a1"))
    linear = _reduce_loop(
        Load(name="in1", input="x", index=idx),
        Assign(name="v0", op="subtract", args=("in1", "mean")),
        Accum(name="acc1", value="v0", op=ElementwiseImpl("add")),
    )
    root, _ = _rewrite(_sum_x(), *_mean_of_sum(), linear)
    assert not _twisted_folds(root)


def _plain_sum() -> Loop:
    idx = (Var("a0"), Var("a1"))
    return _reduce_loop(
        Load(name="in1", input="x", index=idx),
        Accum(name="acc1", value="in1", op=ElementwiseImpl("add")),
    )


def _rewrite(*stmts) -> tuple[Fold, tuple]:
    """Lift the statements under one row and rewrite: the second loop's read of ``acc0`` arrives as
    an operand edge, which is what the recipe matches on. Returns the tree and the kernel's axis table."""
    cell = (*stmts, Write(output="out", index=(Var("a0"),), value="acc1"))
    tile = lift_loop_op(LoopOp(body=(Loop(axis=Axis("a0", Dim(4)), body=Body(cell)),)), name="k_pair")
    return rewrite_twisted(tile.op, tile.axes), tile.axes


@pytest.mark.parametrize(("kind", "should_pair"), [("softmax_pair", True), ("unrelated_pair", False)])
def test_rewrite_pairs_only_the_online_softmax_pair(kind: str, should_pair: bool) -> None:
    """The rewrite collapses the decomposed two-pass softmax (row-max + ``sum exp(x - max)``) into
    ONE twisted fold, and is a NO-OP on an unrelated row-max + plain-sum pair."""
    second = _sum_exp_shifted() if should_pair else _plain_sum()
    root, _ = _rewrite(_row_max(), second)
    twisted = _twisted_folds(root)
    assert bool(twisted) == should_pair
    if should_pair:
        (pair,) = twisted
        assert set(pair.as_reduction().states) == {"acc0", "acc1"}, "the carrier keeps the original acc names"


def test_twisted_folds_lowered_loop_is_well_formed() -> None:
    """The rewritten carrier's lowered loop defines every name before it is read — what
    materialization actually consumes."""
    root, axes = _rewrite(_row_max(), _sum_exp_shifted())
    (pair,) = _twisted_folds(root)
    (loop,) = pair.lower(axes=axes)
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
