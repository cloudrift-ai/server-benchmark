"""Loud-error guardrail for a lowering that leaves a node un-lowered.

A deterministic (greedy) compile of a pipeline that runs to the final
lowering pass promises a graph of ``CudaOp``. A node that reaches the
terminal still holding a ``TileOp`` breaks that promise, and used to do
so quietly: the compile reported success and the graph died downstream
in ``CudaBackend`` / ``plan_from_graph`` with a cryptic ``non-CudaOp``
``TypeError``. Two rewrites strand a node that way, and this file covers
both.

**The filtered option.** The produced ``KernelOp`` fails
``validate(ctx)`` — e.g. the chosen tile shape's materialized smem
exceeds ``ctx.max_dynamic_smem`` — and ``Candidate.try_rewrite`` drops
the only option, recording it in ``Run.rejections``. This is the SDPA
"silent TileOp leak": a scoring change nudges the planner into an
over-budget QK^T / P@V tile.

**The silent decline.** The rule declines the offered row with an
ordinary ``RuleSkipped``, or no rule matches the node at all. Nothing is
recorded, so every mechanism keyed on the rejection sink looked straight
past it.

``GreedyStrategy`` reads the settled terminal itself, so both reach the
same fallbacks — blocklist the row and re-resolve, then re-resolve once
with the prior dropped — and, if nothing lowers the node, the same loud
:class:`LoweringError` naming it. The tuning path
(``Pipeline.tune_async`` / ``TuningSearch``) never raises: a dropped
branch is a graceful dead end there, sibling branches carry other tile
shapes, and an un-lowered terminal is priced a ``bench_fail``.
"""

from __future__ import annotations

import inspect

import pytest

from emmy.compiler.context import Context
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.kernel.ir import KernelOp, Smem
from emmy.compiler.ir.tile.ir import TileOp
from emmy.compiler.pipeline import FINAL_LOWERING_PASS, LoweringError
from emmy.compiler.pipeline.pipeline import Pass, Pattern, Pipeline, Rule, RuleSkipped
from emmy.compiler.pipeline.search.strategy.greedy import _raise_on_unlowered
from tests.compiler.helpers import drain_tune


def _small_smem_ctx() -> Context:
    """A ctx whose dynamic-smem cap (2 KiB) is far below the test
    kernel's 16 KiB slab, so ``KernelOp.validate`` rejects it."""
    return Context(compute_capability=(9, 0), max_dynamic_smem=2048)


def _graph_with_tile() -> Graph:
    """``x -> y`` where ``y`` holds a (placeholder) ``TileOp``. The rule's
    rewrite ignores the body, so an empty one is fine."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4,), "f32"), node_id="x")
    g.add_node(op=TileOp(name="k_test"), inputs=["x"], output=Tensor("y", (4,), "f32"), node_id="y")
    g.inputs = ["x"]
    g.outputs = ["y"]
    return g


def _over_budget_kernel() -> KernelOp:
    """A ``KernelOp`` whose 16 KiB smem slab exceeds the 2 KiB test cap."""
    return KernelOp(body=[Smem(name="buf", extents=(4096,), dtype="float")], name="k_test")


def _build_pipeline() -> Pipeline:
    """One pass, one rule matching the ``TileOp`` at ``y`` whose rewrite
    always returns the over-budget kernel (so its only option is filtered
    by ``validate(ctx)``)."""

    def rewrite(root):  # noqa: ARG001 — the over-budget kernel is fixed
        return _over_budget_kernel()

    rule = Rule(
        name="__over_budget_lower__",
        pattern=[Pattern(name="root", op_type=TileOp)],
        rewrite=rewrite,
        param_names=tuple(inspect.signature(rewrite).parameters.keys()),
    )
    pass_ = Pass(name="__test_lower__", rules=[rule], index=0)
    rule.pass_ = pass_
    return Pipeline(passes=[pass_])


# ---------------------------------------------------------------------------
# Unit: _raise_on_unlowered
# ---------------------------------------------------------------------------


def test_raise_on_unlowered_fires_for_stuck_tileop():
    g = _graph_with_tile()
    rejections = [("y", "k:100_materialize_tile", "smem 104960 > max_dynamic_smem 101376")]
    with pytest.raises(LoweringError) as exc:
        _raise_on_unlowered(g, rejections, lowers_to_cuda=False)
    msg = str(exc.value)
    assert "'y'" in msg
    assert "k:100_materialize_tile" in msg
    assert "smem 104960 > max_dynamic_smem 101376" in msg


def test_no_raise_when_no_rejections_in_a_truncated_pipeline():
    # Empty rejection list under a pipeline that stops before the final lowering pass:
    # an un-lowered TileOp is the intended terminal there (``--ir tile``), so stay silent.
    _raise_on_unlowered(_graph_with_tile(), [], lowers_to_cuda=False)


def test_raise_when_no_rejections_but_the_pipeline_lowers_to_cuda():
    # The silent strand: no rule recorded a decline (a materializer's ordinary
    # ``RuleSkipped``, or no rule matched at all) yet the pipeline promised a
    # Graph[CudaOp]. The node, not the rejection sink, is what makes it un-lowered.
    with pytest.raises(LoweringError) as exc:
        _raise_on_unlowered(_graph_with_tile(), [], lowers_to_cuda=True)
    msg = str(exc.value)
    assert "'y'" in msg
    assert "no lowering rule produced a kernel" in msg
    assert "TileOp" in msg


def test_no_raise_when_node_lowered_despite_rejection():
    # A rejection was recorded, but a later rule lowered the node anyway
    # (its terminal op is no longer a TileOp/LoopOp) → stay silent.
    g = _graph_with_tile()
    g.nodes["y"].op = _over_budget_kernel()  # now a KernelOp, i.e. lowered
    _raise_on_unlowered(g, [("y", "k:100_materialize_tile", "smem ...")], lowers_to_cuda=True)


def test_no_raise_when_rejection_node_absent():
    _raise_on_unlowered(_graph_with_tile(), [("ghost", "k:x", "smem ...")], lowers_to_cuda=False)


# ---------------------------------------------------------------------------
# Integration: greedy raises, tuning prunes gracefully
# ---------------------------------------------------------------------------


def test_greedy_run_raises_lowering_error():
    pipeline = _build_pipeline()
    with pytest.raises(LoweringError) as exc:
        pipeline.run(_graph_with_tile(), ctx=_small_smem_ctx())
    msg = str(exc.value)
    assert "'y'" in msg
    # The reason is derived from KernelOp.validate via _validate_reason.
    assert "smem 16384 > max_dynamic_smem 2048" in msg


def test_tuning_does_not_raise_and_prunes_branch():
    from emmy.compiler.pipeline import TuningSearch
    from emmy.compiler.pipeline.search.db import SearchDB

    pipeline = _build_pipeline()
    terminals = drain_tune(pipeline, _graph_with_tile(), search=TuningSearch(patience=10**6), ctx=_small_smem_ctx(), db=SearchDB())
    # Tuning yields the (dead) terminal without raising; the node stays a
    # TileOp because its only lowering option was validate-filtered.
    assert terminals, "tuning should still yield the dead terminal"
    assert isinstance(terminals[0].graph.nodes["y"].op, TileOp)


# ---------------------------------------------------------------------------
# Tune-side bench guard: a terminal still carrying an un-lowered node is a
# bench_fail decided in ``_TerminalBench.prelude`` — never an ``ok``. Without
# the guard the un-lowered node is invisible to the bench (it sums CudaOps
# only): a no-CudaOp terminal reported ok @ 0 µs, and a cached residual kernel
# (a split's finalize) reported its own µs as the terminal's — the issue-#327
# "impossibly fast" ok rows that poisoned the tune DB.
# ---------------------------------------------------------------------------


def _terminal_bench(graph, *, backend, db):
    from types import SimpleNamespace

    from emmy.compiler.pipeline.search.policy.terminal_bench import TerminalBench as _TerminalBench

    return _TerminalBench(SimpleNamespace(graph=graph, ctx=_small_smem_ctx()), backend=backend, db=db)


def test_unlowered_terminal_is_bench_fail_without_cuda_nodes():
    from emmy.compiler.pipeline.search.db import SearchDB

    b = _terminal_bench(_graph_with_tile(), backend=None, db=SearchDB())
    assert b.unlowered == ["y"]
    kind, (_stats, status) = b.prelude()
    assert kind == "done"
    assert status == "bench_fail"


def test_unlowered_terminal_is_bench_fail_despite_cached_residual_kernel():
    from emmy.compiler.ir.cuda.ir import CudaOp
    from emmy.compiler.pipeline.search.db import SearchDB

    class _StubBackend:
        name = "cuda"
        bench_run_timeout_s = 1.0

    # ``x -> y (TileOp) -> z (CudaOp)`` — the split shape: an un-lowered partial
    # feeding a lowered finalize whose perf row is already cached.
    g = _graph_with_tile()
    cuda = CudaOp(kernel_source="__global__ void k_fin() {}", kernel_name="k_fin")
    g.add_node(op=cuda, inputs=["y"], output=Tensor("z", (4,), "f32"), node_id="z")
    g.outputs = ["z"]
    db = SearchDB()
    b = _terminal_bench(g, backend=_StubBackend(), db=db)
    db.record_perf(
        b.context_key, cuda.identity_key(with_io=True, with_knobs=True), backend="cuda", status="ok", stats=b._point_stats(104.0)
    )
    kind, (stats, status) = b.prelude()
    assert kind == "done"
    assert status == "bench_fail"
    assert stats.median > 104.0  # the pinned fail latency, not the residual kernel's cached µs


# ---------------------------------------------------------------------------
# Emission-order fallback: a prior that over-extrapolates large (over-budget)
# tiles onto a small shape must not abort the greedy compile. The retry
# blocklist exhausts on the prior-ranked over-budget tiles, then a resolve with
# the prior dropped recovers the in-budget tile (these fixture rules emit it
# first — a property of the fixture, not a promise the enumeration makes; the
# fallback is a VALIDITY mechanism and claims nothing about speed). This is the
# tune-time golden-sweep crash: a prior trained on big square matmuls picked a
# >smem-cap tile for the tiny ``qwen3_06b.q_proj`` projection and the assemble
# raised instead of falling back.
# ---------------------------------------------------------------------------


class _BiggestBNFirstPrior:
    """Stub global prior that ranks leaves by ``BN`` descending — i.e. always
    prefers the largest (over-budget) tile, the way a prior trained on big
    square matmuls extrapolates onto a tiny shape. ``pick`` returns the
    argmax-BN row, so greedy keeps choosing over-budget tiles until the
    blocklist retry budget is exhausted."""

    fitted = True

    def pick(self, rows: list[dict]) -> tuple[int, float]:
        best_i = max(range(len(rows)), key=lambda i: rows[i].get("BN", 0))
        return best_i, 0.0


def _two_pass_tile_pipeline(n_over_budget: int, *, decline: bool = False) -> Pipeline:
    """Mirror the real lowering shape: pass 0 (partition → tile ``Fork``) emits
    an in-budget option-0 (``BN=8``, emitted first) followed by
    ``n_over_budget`` over-budget tile leaves (``BN=16, 24, …``); pass 1
    (``100_materialize_tile``, carrying the final lowering pass's name so the
    pipeline promises a ``Graph`` of kernels) materializes the chosen tile into a
    ``KernelOp`` and lets ``validate(ctx)`` filter it. Over-budget tiles only fail at the
    materialize pass (like the real planner emitting tile leaves that pass
    through ``Candidate.try_rewrite``'s validate filter unchecked), so the
    prior can rank them top and the blocklist retry engages per tile identity
    (``BN``). Pass 0 ``RuleSkipped``-guards on the BN marker so it never
    re-fires on its own (already-tiled) output.

    With ``decline`` those same leaves are declined by an ordinary ``RuleSkipped``
    instead — the row is refused with nothing recorded, the second way a node ends up
    stranded."""
    from emmy.compiler.pipeline.fork import OptionFork

    def _tile_leaf(bn: int) -> TileOp:
        return TileOp(name="k_test", knobs={"BN": bn})

    def emit_tiles(root):
        if "BN" in root.op.knobs:  # already tiled (our own output) → don't re-fork
            raise RuleSkipped("already tiled")
        leaves = [OptionFork(option=_tile_leaf(8), knobs={"BN": 8})]
        leaves += [OptionFork(option=_tile_leaf(16 + 8 * i), knobs={"BN": 16 + 8 * i}) for i in range(n_over_budget)]
        return leaves

    def materialize(root):
        bn = root.op.knobs.get("BN", 0)
        if decline and bn > 8:
            # The SILENT decline: an ordinary ``RuleSkipped`` (``reject=False``) records
            # nothing in the rejection sink, so only the surviving TileOp says the row failed.
            raise RuleSkipped(f"nothing lowers BN={bn}")
        extents = (64,) if bn <= 8 else (4096,)  # 256 B fits the 2 KiB cap; 16 KiB overflows
        return KernelOp(body=[Smem(name="buf", extents=extents, dtype="float")], name="k_test", knobs={"BN": bn})

    emit_rule = Rule(
        name="__emit_tiles__",
        pattern=[Pattern(name="root", op_type=TileOp)],
        rewrite=emit_tiles,
        param_names=tuple(inspect.signature(emit_tiles).parameters.keys()),
    )
    mat_rule = Rule(
        name="100_materialize_tile",
        pattern=[Pattern(name="root", op_type=TileOp)],
        rewrite=materialize,
        param_names=tuple(inspect.signature(materialize).parameters.keys()),
    )
    p0 = Pass(name="__partition__", rules=[emit_rule], index=0)
    p1 = Pass(name=FINAL_LOWERING_PASS, rules=[mat_rule], index=1)
    emit_rule.pass_ = p0
    mat_rule.pass_ = p1
    return Pipeline(passes=[p0, p1])


def test_greedy_run_falls_back_to_option0_when_prior_overflows(monkeypatch):
    # The prior ranks 12 over-budget tiles above the lone in-budget tile, so
    # the blocklist retry can never reach it within its budget. Before the
    # prior-less fallback this raised LoweringError; now ``Pipeline.run``
    # re-resolves without the prior and this fixture's first leaf is in budget.
    import emmy.compiler.pipeline.search.policy.greedy as greedy_mod

    monkeypatch.setattr(greedy_mod, "_load_prior_safe", lambda: _BiggestBNFirstPrior())
    terminal = _two_pass_tile_pipeline(n_over_budget=12).run(_graph_with_tile(), ctx=_small_smem_ctx())
    op = terminal.nodes["y"].op
    assert isinstance(op, KernelOp), "the in-budget first-emitted tile must lower (no LoweringError)"
    assert op.knobs.get("BN") == 8, "the recovered tile is the budget-safe emission-order leaf"


def test_greedy_run_still_raises_when_no_in_budget_option(monkeypatch):
    # The fallback must not paper over a genuinely un-lowerable shape: when
    # EVERY tile is over-budget, option-0 overflows too and the loud
    # LoweringError still fires (no in-budget leaf exists to recover).
    import emmy.compiler.pipeline.search.policy.greedy as greedy_mod

    monkeypatch.setattr(greedy_mod, "_load_prior_safe", lambda: _BiggestBNFirstPrior())
    # Drop the in-budget option-0: shift all leaves over budget by tuning the
    # materializer to overflow for every BN (handled by the all-over-budget
    # single-option pipeline already covered by ``_build_pipeline``).
    with pytest.raises(LoweringError):
        _build_pipeline().run(_graph_with_tile(), ctx=_small_smem_ctx())


# ---------------------------------------------------------------------------
# The SILENT strand: a materializer that declines a row with an ordinary
# ``RuleSkipped`` (``reject=False``) records nothing, so for a node in that state the
# rejection sink is empty and every mechanism keyed on it — the blocklist retry, the
# prior-off re-resolve, the loud ``LoweringError`` — used to look straight past it. The
# compile returned a terminal still holding the ``TileOp`` and reported success; the graph
# died later in ``plan_from_graph`` ("non-CudaOp 'TileOp'"), at engine init on the deploy
# and mid-campaign under ``run --bench``. The terminal itself is now the evidence: under a
# pipeline that runs to the final lowering pass, a surviving tile is a stranded node.
# ---------------------------------------------------------------------------


def _silently_declining_pipeline(pass_name: str) -> Pipeline:
    """One pass, named ``pass_name``, whose only rule declines the ``TileOp`` at ``y``
    with an ordinary ``RuleSkipped``. Nothing reaches the rejection sink and the node
    survives the pass unchanged."""

    def rewrite(root):  # noqa: ARG001 — the decline is fixed
        raise RuleSkipped("this row's projection is not one this lowering owns")

    rule = Rule(
        name="010_materialize",
        pattern=[Pattern(name="root", op_type=TileOp)],
        rewrite=rewrite,
        param_names=tuple(inspect.signature(rewrite).parameters.keys()),
    )
    pass_ = Pass(name=pass_name, rules=[rule], index=0)
    rule.pass_ = pass_
    return Pipeline(passes=[pass_])


def test_greedy_run_raises_when_the_only_lowering_declines_silently():
    with pytest.raises(LoweringError) as exc:
        _silently_declining_pipeline(FINAL_LOWERING_PASS).run(_graph_with_tile(), ctx=_small_smem_ctx())
    msg = str(exc.value)
    assert "'y'" in msg
    assert "no lowering rule produced a kernel" in msg


def test_greedy_run_keeps_the_tile_terminal_of_a_truncated_pipeline():
    # ``emmy compile --ir tile`` and the loop backend stop before the final lowering pass,
    # where a surviving TileOp is the requested answer — the check must not fire there.
    terminal = _silently_declining_pipeline("lowering/tile").run(_graph_with_tile(), ctx=_small_smem_ctx())
    assert isinstance(terminal.nodes["y"].op, TileOp)


def test_greedy_run_retries_past_a_silently_declined_row(monkeypatch):
    # The fallback the strand used to escape: the prior ranks two rows this lowering
    # declines above the one it accepts, and the blocklist retry walks past both (keyed on
    # the surviving tile's BN) instead of returning the half-lowered graph as a success.
    import emmy.compiler.pipeline.search.policy.greedy as greedy_mod

    monkeypatch.setattr(greedy_mod, "_load_prior_safe", lambda: _BiggestBNFirstPrior())
    terminal = _two_pass_tile_pipeline(n_over_budget=2, decline=True).run(_graph_with_tile(), ctx=_small_smem_ctx())
    op = terminal.nodes["y"].op
    assert isinstance(op, KernelOp), "the accepted row must lower (no half-lowered terminal)"
    assert op.knobs.get("BN") == 8


def test_tuning_prunes_a_silently_declined_row_without_raising():
    # Under tune the same decline stays a graceful dead end: the terminal is yielded with
    # the node still a TileOp and ``TerminalBench`` prices it a bench_fail — never a raise
    # that would abort the whole session over one row.
    from emmy.compiler.pipeline import TuningSearch
    from emmy.compiler.pipeline.search.db import SearchDB

    pipeline = _silently_declining_pipeline(FINAL_LOWERING_PASS)
    terminals = drain_tune(pipeline, _graph_with_tile(), search=TuningSearch(patience=10**6), ctx=_small_smem_ctx(), db=SearchDB())
    assert terminals, "tuning should still yield the dead terminal"
    assert isinstance(terminals[0].graph.nodes["y"].op, TileOp)


def test_run_leaves_no_state_on_pipeline():
    # The rejection sink is Run-scoped (``Run.rejections``), never stashed on
    # the shared frozen Pipeline — a subsequent tune on the same Pipeline sees
    # no sink (silent fork-pruning preserved), and concurrent runs can't
    # clobber each other.
    pipeline = _build_pipeline()
    with pytest.raises(LoweringError):
        pipeline.run(_graph_with_tile(), ctx=_small_smem_ctx())
    assert not hasattr(pipeline, "_lowering_rejections")


# ---------------------------------------------------------------------------
# Per-variant containment: a lowering pass that *raises* (not a validate
# filter) aborts a greedy compile loudly, but under tune is a dropped dead
# end so one un-lowerable search fork can't abort the whole tune. This is the
# stacked defect the static tune-findings report flagged: an un-handled
# fused-cell slab shape (compute_phase_info LoweringError) / an orphan AtomTile
# at render would crash mid-tune with no per-variant containment.
# ---------------------------------------------------------------------------


def _build_raising_pipeline() -> Pipeline:
    """One pass, one rule matching the ``TileOp`` at ``y`` whose rewrite
    *raises* a ``LoweringError`` (an un-lowerable shape a deterministic pass
    chokes on), rather than returning a validate-filtered option."""

    def rewrite(root):  # noqa: ARG001
        raise LoweringError("synthetic un-lowerable shape")

    rule = Rule(
        name="__raising_lower__",
        pattern=[Pattern(name="root", op_type=TileOp)],
        rewrite=rewrite,
        param_names=tuple(inspect.signature(rewrite).parameters.keys()),
    )
    pass_ = Pass(name="__test_raise__", rules=[rule], index=0)
    rule.pass_ = pass_
    return Pipeline(passes=[pass_])


def test_greedy_run_propagates_lowering_exception():
    # Greedy uses ``Run.resolve`` (no containment) — a raising lowering pass
    # propagates loudly, exactly as before.
    pipeline = _build_raising_pipeline()
    with pytest.raises(LoweringError, match="synthetic un-lowerable shape"):
        pipeline.run(_graph_with_tile(), ctx=_small_smem_ctx())


def test_tuning_contains_raising_lowering_pass(caplog):
    # Under tune, ``Run.drive`` catches the lowering exception, drops the
    # candidate's subtree, logs a warning, and finishes without raising —
    # so a single un-lowerable fork can't abort the whole tune.
    import logging

    from emmy.compiler.pipeline import TuningSearch
    from emmy.compiler.pipeline.search.db import SearchDB

    pipeline = _build_raising_pipeline()
    with caplog.at_level(logging.WARNING, logger="emmy.compiler.pipeline"):
        terminals = drain_tune(pipeline, _graph_with_tile(), search=TuningSearch(patience=10**6), ctx=_small_smem_ctx(), db=SearchDB())
    # The only lowering option raised, so no terminal is benchable — the search
    # ends cleanly with zero terminals instead of crashing.
    assert terminals == []
    assert any("dropped un-lowerable candidate" in r.message for r in caplog.records)


def test_tuning_contains_raising_fork_thunk(caplog):
    # Same containment, one step earlier in the loop: a rule can defer the
    # un-lowerable shape behind a ``DeferredFork`` thunk, so the raise lands in
    # ``LazyCandidate.resolve`` -> ``Fork.expand`` -> the rule's materializer
    # rather than in the next ``_step``. Observed in the wild on a DeepSeek-V4
    # tune (a realize-time cut guardrail firing under ``030_cut``'s realize
    # lambda), where it killed the whole session. The sibling option must still
    # reach a terminal.
    import logging

    from emmy.compiler.pipeline import TuningSearch
    from emmy.compiler.pipeline.fork import DeferredFork
    from emmy.compiler.pipeline.search.db import SearchDB

    def raise_at_expand():
        raise LoweringError("synthetic un-lowerable shape")

    def rewrite(root):  # noqa: ARG001 — both options are fixed
        return [
            DeferredFork(materialize=raise_at_expand, knobs={"BN": 8}),
            KernelOp(body=[Smem(name="buf", extents=(64,), dtype="float")], name="k_test", knobs={"BN": 16}),
        ]

    rule = Rule(
        name="__deferred_raising_lower__",
        pattern=[Pattern(name="root", op_type=TileOp)],
        rewrite=rewrite,
        param_names=tuple(inspect.signature(rewrite).parameters.keys()),
    )
    pass_ = Pass(name="__test_deferred_raise__", rules=[rule], index=0)
    rule.pass_ = pass_
    pipeline = Pipeline(passes=[pass_])

    with caplog.at_level(logging.WARNING, logger="emmy.compiler.pipeline"):
        terminals = drain_tune(pipeline, _graph_with_tile(), search=TuningSearch(patience=10**6), ctx=_small_smem_ctx(), db=SearchDB())
    assert [t.graph.nodes["y"].op.knobs.get("BN") for t in terminals] == [16], "the surviving sibling must still terminate"
    assert any("dropped un-lowerable candidate" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# The materializer's OTHER decline: a projection tail whose chain reads a name
# the sliced node does not compute (``_atom._warp_epilogue``). It is the same
# kind of answer as the ``UnbindableProjection`` decline — this row's binding is
# refused — so it must reach the rejection sink. While it did not, a compile that
# took a structural cut whose remainder is mis-sliced returned a terminal still
# holding the ``TileOp``, reported success, and died downstream in
# ``plan_from_graph`` ("non-CudaOp 'TileOp'") — an engine-init crash on a serve
# boot whose measured evidence priced that cut cheapest.
# ---------------------------------------------------------------------------


def _mis_sliced_tail():
    """A projection tail whose chain op reads ``acc2`` — a sibling channel's
    accumulator, which a node sliced to one channel never computes."""
    from emmy.compiler.ir.elementwise import ElementwiseImpl
    from emmy.compiler.ir.expr import Var
    from emmy.compiler.ir.stmt import Assign, Write

    return [
        Assign(name="result", op=ElementwiseImpl("copy"), args=("acc2",)),
        Write(output="out", index=(Var("m"), Var("n")), value="result"),
    ]


def test_mis_sliced_projection_tail_declines_as_a_rejection():
    from emmy.compiler.ir.sigma import Sigma
    from emmy.compiler.pipeline import RuleSkipped
    from emmy.compiler.pipeline.passes.lowering.kernel._atom import _warp_epilogue

    with pytest.raises(RuleSkipped) as exc:
        _warp_epilogue(_mis_sliced_tail(), "acc", "m", "n", Sigma.IDENTITY)
    assert "acc2" in exc.value.reason
    assert exc.value.reject, "the decline must record as a rejection or the greedy retry never sees it"


def test_greedy_run_raises_when_the_projection_tail_is_mis_sliced():
    # End to end through the engine: the only lowering declines this row, so the
    # node stays a TileOp — the compile must say so, not hand a half-lowered
    # graph to the backend.
    from emmy.compiler.ir.sigma import Sigma
    from emmy.compiler.pipeline.passes.lowering.kernel._atom import _warp_epilogue

    def rewrite(root):  # noqa: ARG001 — the decline is fixed
        return _warp_epilogue(_mis_sliced_tail(), "acc", "m", "n", Sigma.IDENTITY)

    rule = Rule(
        name="010_materialize",
        pattern=[Pattern(name="root", op_type=TileOp)],
        rewrite=rewrite,
        param_names=tuple(inspect.signature(rewrite).parameters.keys()),
    )
    pass_ = Pass(name="lowering/kernel", rules=[rule], index=0)
    rule.pass_ = pass_
    with pytest.raises(LoweringError) as exc:
        Pipeline(passes=[pass_]).run(_graph_with_tile(), ctx=_small_smem_ctx())
    assert "'y'" in str(exc.value)
    assert "acc2" in str(exc.value)
