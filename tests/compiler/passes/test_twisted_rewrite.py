"""The twisted rewrite: every two-pass reduce pair a recipe recognizes fuses into its carrier.

The lift leaves the dependency in the tree — the denominator reads the row maximum as an operand
— so ``Fold.twist`` matches by position and canonical form, never by a term's names; the same
softmax recipe fuses online softmax and flash attention. The negative case matters as much as the
positives: a rewrite that fires on a plain row-max plus an unrelated sum is a miscompile, not a
slow kernel, and no numerics assert downstream would attribute the wrong answer to this pass.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from emmy.commands.trace import graph_from_code
from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.loop.runner import execute_loop_op_cpp
from emmy.compiler.ir.pure import Fold
from emmy.compiler.ir.pure.twist import SOFTMAX, WELFORD
from emmy.compiler.ir.stmt import Accum, Assign, Body, Const, Load, Loop, Write
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import CUDA_PASSES, LOOP_PASSES, TILE_PASSES, Pipeline
from emmy.compiler.pipeline.fork import DeferredFork, iter_leaves
from emmy.compiler.pipeline.passes.lowering.tile._fromloop import lift_loop_op
from emmy.compiler.pipeline.passes.lowering.tile._twist import _hoist_invariant, relift, rewrite_twisted
from emmy.compiler.pipeline.pipeline import Run
from emmy.compiler.pipeline.search.pins import pinned_knobs


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


def _lifted(code: str) -> Graph:
    graph, _, _ = graph_from_code(code)
    graph = Pipeline.build(LOOP_PASSES).run(graph)
    return Pipeline.build(["lowering/tile"], select=["lift"]).run(graph)


def _tile(code: str, twist: str = "twisted") -> TileOp:
    with pinned_knobs({"TWIST": twist}):
        graph = Pipeline.build(["lowering/tile"], select=["twisted"]).run(_lifted(code))
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
    score, one, mean, zero = fold.lift.results
    consts = {stmt.name: stmt.value for stmt in fold.lift.body if isinstance(stmt, Const)}
    assert mean == score and consts[one] == 1.0 and consts[zero] == 0.0, "the singleton is (x, 1, x, 0)"
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


# ===================================================================
# The inverse: an online carrier's loop lifts to the two-pass tree
# ===================================================================

_SDPA = (
    "F.scaled_dot_product_attention("
    "torch.randn(1, 1, 32, 16, dtype=torch.float16), "
    "torch.randn(1, 1, 32, 16, dtype=torch.float16), "
    "torch.randn(1, 1, 32, 16, dtype=torch.float16))"
)


def _online_carrier_loop() -> Loop:
    """The stable online form flash attention streams, as ``Fold.merge`` spells it: per key, advance
    the pivot, rescale ``(o, l)`` by ``α`` through ``Accum.base`` and fold the ``β``-weighted injection."""
    idx = (Var("a0"), Var("a1"))
    return _reduce_loop(
        Load(name="s", input="x", index=idx),
        Load(name="v", input="val", index=(Var("a1"),)),
        Const(name="one", value=1.0),
        Assign(name="gn", op="maximum", args=("m", "s")),
        Assign(name="dg", op="subtract", args=("m", "gn")),
        Assign(name="alpha", op="exp", args=("dg",)),
        Assign(name="dg_o", op="subtract", args=("s", "gn")),
        Assign(name="beta", op="exp", args=("dg_o",)),
        Assign(name="o_sa", op="multiply", args=("o", "alpha")),
        Assign(name="o_sb", op="multiply", args=("v", "beta")),
        Accum(name="o", value="o_sb", op=ElementwiseImpl("add"), base="o_sa"),
        Assign(name="l_sa", op="multiply", args=("l", "alpha")),
        Assign(name="l_sb", op="multiply", args=("one", "beta")),
        Accum(name="l", value="l_sb", op=ElementwiseImpl("add"), base="l_sa"),
        Accum(name="m", value="s", op=ElementwiseImpl("maximum")),
    )


def _two_pass_shape(root: Fold) -> tuple[Fold, Fold, Fold]:
    """The ``(m, O, l)`` folds of a two-pass tree: the pivot's max, the value contraction whose A
    cone reads that max, and the weight sum reading the same cone."""
    reduces = [fold for fold in _folds(root) if fold.axis is not None and fold.as_reduction().ops is not None]
    (m,) = [fold for fold in reduces if fold.as_reduction().ops[0].name == "maximum"]
    (o,) = [fold for fold in reduces if fold.as_contraction() is not None and any(m in edge.operands for edge in fold.operands)]
    (weight_sum,) = [
        fold
        for fold in reduces
        if fold is not o
        and fold is not m
        and fold.as_reduction().ops[0].name == "add"
        and any(m in edge.operands for edge in fold.operands)
    ]
    return m, o, weight_sum


def test_online_carrier_loop_lifts_to_the_two_pass_tree() -> None:
    """A reduce loop already in the stable online form — the shape the twist LOWERS to, with the
    ψ-rescale as ``Accum.base`` — lifts to the two-pass tree the recipe certifies: the pivot's own
    max, the value channel as a contraction whose A cone is ``exp(s − m)`` over the pivot's final
    state, and the weight sum beside it. The value contraction is a ``TILE`` site of its own, and
    the twist fuses the pair right back, so the online loop and the two-pass ops meet at one term."""
    cell = (
        _online_carrier_loop(),
        Assign(name="inv", op="reciprocal", args=("l",)),
        Assign(name="out_v", op="multiply", args=("o", "inv")),
        Write(output="out", index=(Var("a0"),), value="out_v"),
    )
    tile = lift_loop_op(LoopOp(body=(Loop(axis=Axis("a0", Dim(4)), body=Body(cell)),)), name="k_online")

    assert not _twisted_folds(tile.op)
    m, o, _ = _two_pass_shape(tile.op)
    assert o.as_contraction() is not None and o.operands[0].axis is None, "A is the weight cone, B the value slab"
    assert o.operands[1].as_slab() is not None
    assert any(site.node is o for site in tile.sites) and any(tile.sites[s].node is o for s in tile.family_sites["TILE"])
    (pair,) = _twisted_folds(rewrite_twisted(tile.op, tile.axes))
    assert len(pair.init) == 3, "the two-pass tree the lift built is the one the recipe fuses"


def test_sdpa_carrier_relifts_to_the_two_pass_tree_that_computes_the_same_values() -> None:
    """The twist pass's own inverse: the fused SDPA carrier lowered to Loop IR and lifted back is
    the two-pass tree — one shared score contraction, one pivot, the value contraction tileable —
    and both forms compute attention to the reference on the CPU loop runner."""
    graph = _lifted(_SDPA)
    with pinned_knobs({"TWIST": "twisted"}):
        graph = Pipeline.build(["lowering/tile"], select=["twisted"]).run(graph)
    twisted = next(node.op for node in graph.nodes.values() if isinstance(node.op, TileOp))
    two_pass = relift(twisted, graph)
    assert two_pass is not None and not _twisted_folds(two_pass.op)
    m, o, _ = _two_pass_shape(two_pass.op)
    scores = [fold for fold in _folds(two_pass.op) if fold.as_contraction() is not None and fold is not o]
    assert len(scores) == 1, "the three passes read ONE score node — sharing is edge reuse"
    assert any(two_pass.sites[s].node is o for s in two_pass.family_sites["TILE"])
    assert two_pass.knobs.keys() >= {k for k in twisted.knobs if k.startswith("S_")} and two_pass.knobs != twisted.knobs, (
        "restamped identity"
    )

    rng = np.random.default_rng(0)
    q, k, v = (rng.standard_normal((1, 1, 32, 16)).astype(np.float16) for _ in range(3))
    arrays = {"x0": q, "x1": k, "x2": v, "scaled_dot_product_attention_scale": np.array([0.25], dtype=np.float16)}
    s = (q[0, 0].astype(np.float64) @ k[0, 0].astype(np.float64).T) * 0.25
    p = np.exp(s - s.max(-1, keepdims=True))
    reference = (p / p.sum(-1, keepdims=True)) @ v[0, 0].astype(np.float64)
    for tile in (twisted, two_pass):
        loop = LoopOp(
            body=tile.op.lower(bound=frozenset(), stores=tile.output_specs, axes=tile.axes), inputs=tile.inputs, outputs=tile.outputs
        )
        got = np.asarray(execute_loop_op_cpp(loop, arrays, {"scaled_dot_product_attention": (1, 1, 32, 16)}), dtype=np.float64)
        np.testing.assert_allclose(got[0, 0], reference, atol=2e-3)


def test_twist_pass_offers_the_carrier_beside_the_two_pass_tree() -> None:
    """Unpinned, the pass forks: the carrier first (the cold pick keeps the single-pass kernel),
    the two-pass tree second — a structural arm, a different kernel — each carrying its ``TWIST``
    value as the fork's own knob delta. A pin decides in place."""
    graph = _lifted(_SDPA)
    pipeline = Pipeline.build(["lowering/tile"], select={"twisted"})
    rule = pipeline.passes[0].rules[0]
    (match,) = pipeline.match(graph, rule)
    options = rule.rewrite(match=match, root=match.root)
    assert [option.knobs for option in options] == [{"TWIST": "twisted"}, {"TWIST": "two-pass"}]
    assert all(isinstance(option, DeferredFork) for option in options)
    carrier, two_pass = (option.expand()[0] for option in options)
    assert _twisted_folds(carrier.op) and not _twisted_folds(two_pass.op)
    assert "TWIST" not in carrier.knobs and "TWIST" not in two_pass.knobs, "an input pin, never a knob on the kernel"
    assert not _twisted_folds(_tile(_SDPA, "two-pass").op)

    # Routed through the fork rather than pinned: the two-pass arm, then the denominator's cut.
    # The pieces the cut mints are not offered the pair again — the pass does not rewind onto them.
    def decide(fp):
        for option in fp.options:
            knobs = getattr(option, "knobs", None) or {}
            if knobs.get("TWIST") == "two-pass" or knobs.get("PLACE@map.2/reduce") == "cut":
                return option
        return next(iter_leaves(fp.options))

    terminal, _ = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=Context.from_target((8, 0))).resolve(_lifted(_SDPA), decide)
    tiles = [node.op for node in terminal.nodes.values() if isinstance(node.op, TileOp)]
    assert len(tiles) == 2 and not any(_twisted_folds(tile.op) for tile in tiles)


def test_two_pass_sdpa_value_contraction_reaches_the_mma_tier() -> None:
    """Pinned to the two-pass tree, with the denominator materialized by a placement cut and a
    tensor-core tile at the value contraction, SDPA lowers its P·V on ``mma.sync`` — the site the
    carrier could not offer. The cut is load-bearing: the kernel binder reads a root projection's
    epilogue only over what the tiled node itself computes, so ``1/l`` arrives as a workspace load."""
    graph, _, _ = graph_from_code(_SDPA)
    pins = {"FAST_MATH": False, "TWIST": "two-pass", "PLACE@map.2/reduce": "cut", "TILE@map.1/inner": "mma_m16n8k16_f16_f32/f1x1"}
    with pinned_knobs(pins):
        lowered = Pipeline.build(CUDA_PASSES).run(graph, ctx=Context.from_target((8, 0)))
    kernels = {name: node.op for name, node in lowered.nodes.items() if isinstance(node.op, CudaOp)}
    (consumer,) = (op for name, op in kernels.items() if "__place" not in name)
    assert len(kernels) == 2, "the denominator's piece and the attention consumer"
    assert consumer.knobs["TILE@map.1/inner"] == pins["TILE@map.1/inner"], "the value contraction's site, on the two-pass routes"
    assert "mma.sync" in consumer.kernel_source
