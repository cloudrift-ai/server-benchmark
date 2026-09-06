"""Loud-error guardrail for the silent ``validate(ctx)`` lowering drop.

When a deterministic (greedy) compile reaches the final tile→kernel
lowering and the produced ``KernelOp`` fails ``validate(ctx)`` (e.g. the
chosen tile shape's materialized smem exceeds ``ctx.max_dynamic_smem``),
``Candidate.try_rewrite`` filters the only option away. Historically this
was silent (DEBUG-only) and the un-lowered ``TileOp`` leaked all the way
to ``CudaBackend``, which raised a cryptic ``non-CudaOp`` ``TypeError``.

``Pipeline.run`` now installs a rejection sink and, after the single
terminal settles, raises a loud :class:`LoweringError` naming the node,
the pass that declined it, and the validate reason. The tuning path
(``Pipeline.tune_async`` / ``TuningSearch``) installs no sink, so the
fork-pruning drop stays silent and a dropped branch is a graceful dead
end — sibling branches carry other tile shapes.

This is the SDPA "silent TileOp leak" failure mode: a scoring change can
nudge the planner into an over-budget QK^T / P@V tile, and without this
guardrail the only symptom was the downstream ``CudaBackend`` mystery.
"""

from __future__ import annotations

import inspect

import pytest

from emmy.compiler.context import Context
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.kernel.ir import KernelOp, Smem
from emmy.compiler.ir.tile.ir import TileOp
from emmy.compiler.pipeline import LoweringError
from emmy.compiler.pipeline.pipeline import Pass, Pattern, Pipeline, Rule
from emmy.compiler.pipeline.search.policy.terminal_bench import point_stats
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
        _raise_on_unlowered(g, rejections, _small_smem_ctx())
    msg = str(exc.value)
    assert "'y'" in msg
    assert "k:100_materialize_tile" in msg
    assert "smem 104960 > max_dynamic_smem 101376" in msg


def test_no_raise_when_no_rejections():
    # Empty rejection list — even with an un-lowered TileOp present, the
    # guardrail only fires for a *recorded* validate drop.
    _raise_on_unlowered(_graph_with_tile(), [], _small_smem_ctx())


def test_no_raise_when_node_lowered_despite_rejection():
    # A rejection was recorded, but a later rule lowered the node anyway
    # (its terminal op is no longer a TileOp/LoopOp) → stay silent.
    g = _graph_with_tile()
    g.nodes["y"].op = _over_budget_kernel()  # now a KernelOp, i.e. lowered
    _raise_on_unlowered(g, [("y", "k:100_materialize_tile", "smem ...")], _small_smem_ctx())


def test_no_raise_when_rejection_node_absent():
    _raise_on_unlowered(_graph_with_tile(), [("ghost", "k:x", "smem ...")], _small_smem_ctx())


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
    db.record_perf(b.context_key, cuda.identity_key(with_io=True, with_knobs=True), backend="cuda", status="ok", stats=point_stats(104.0))
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


def _two_pass_tile_pipeline(n_over_budget: int) -> Pipeline:
    """Mirror the real lowering shape: pass 0 (partition → tile ``Fork``) emits
    an in-budget option-0 (``BN=8``, emitted first) followed by
    ``n_over_budget`` over-budget tile leaves (``BN=16, 24, …``); pass 1
    (``100_materialize_tile``) materializes the chosen tile into a ``KernelOp``
    and lets ``validate(ctx)`` filter it. Over-budget tiles only fail at the
    materialize pass (like the real planner emitting tile leaves that pass
    through ``Candidate.try_rewrite``'s validate filter unchecked), so the
    prior can rank them top and the blocklist retry engages per tile identity
    (``BN``). Pass 0 ``RuleSkipped``-guards on the BN marker so it never
    re-fires on its own (already-tiled) output."""
    from emmy.compiler.pipeline.fork import OptionFork
    from emmy.compiler.pipeline.pipeline import RuleSkipped

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
    p1 = Pass(name="__materialize__", rules=[mat_rule], index=1)
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
# A refused row on ONE piece of a composed route re-ranks that piece inside the
# same structural route, across as many retries as the piece has rows; once no
# row of the piece binds, the one structural pick retired is the cut that minted
# the piece, so every other kernel-set decision stays priced by evidence.
# Observed on the DeepSeek-V4 ``post4096`` twin: the greedy elected the composed
# placement route, a consumer piece's top-ranked row was refused by the kernel
# binder, and the retry could never re-rank it — the blocklist was keyed on the
# terminal node's knob row, which a kernel-stage pass had stamped with a policy
# knob (``LOOPIFY``) no schedule leaf spells — so the second attempt re-picked
# the same row and the strategy retired the structural picks wholesale, onto the
# fused root the disqualification index already priced ``inf``.
# ---------------------------------------------------------------------------


def _composed_fragment(input_id: str, suffix: str) -> Graph:
    """The composed route ``input -> y_ws (k_ws) -> y__cut (k_residual)`` replacing the kernel it is
    offered on; the splice hands that kernel's id to the residual root, so the terminal tells the
    routes apart by op name."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor(input_id, (4,), "f32"), node_id=input_id)
    ws, cut = f"y_ws{suffix}", f"y__cut{suffix}"
    g.add_node(op=TileOp(name=f"k_ws{suffix}"), inputs=[input_id], output=Tensor(ws, (4,), "f32"), node_id=ws)
    g.add_node(op=TileOp(name=f"k_residual{suffix}"), inputs=[ws], output=Tensor(cut, (4,), "f32"), node_id=cut)
    g.outputs = [cut]
    return g


def _composed_route_pipeline(refused: dict[str, set[int]], *, rows: tuple[int, ...] = (8, 16), cut_residual: bool = False) -> Pipeline:
    """Pass 0 is the kernel-set fork at ``k_test`` (and, with ``cut_residual``, again at the residual
    ``k_residual`` — the piece's own cut): the fused ``rows`` beside the composed route. Pass 1
    schedules each piece over ``rows``. Pass 2 materializes every row except the ``refused`` ones
    per kernel name, which it declines with a rejecting skip — the kernel binder's
    projection-ownership refusal. Pass 3 is the kernel-stage policy stamp: a decided knob lands on
    every node still un-lowered, exactly as ``LOOPIFY`` does on the production terminal."""
    from dataclasses import replace

    from emmy.compiler.pipeline.fork import OptionFork
    from emmy.compiler.pipeline.pipeline import RuleSkipped

    def cut(root):
        if "BN" in root.op.knobs:
            raise RuleSkipped("already scheduled")
        if root.op.name == "k_test":
            fragment, seam = _composed_fragment("x", ""), "PLACE@seam"
        elif root.op.name == "k_residual" and cut_residual:
            fragment, seam = _composed_fragment("y_ws", "2"), "PLACE@seam.1"
        else:
            raise RuleSkipped("not a kernel-set fork")
        fused = [OptionFork(option=TileOp(name=root.op.name, knobs={"BN": bn}), knobs={"BN": bn}) for bn in rows]
        return [*fused, OptionFork(option=fragment, knobs={seam: "cut"})]

    def schedule(root):
        if "BN" in root.op.knobs:
            raise RuleSkipped("already scheduled")
        return [OptionFork(option=replace(root.op, knobs={"BN": bn}), knobs={"BN": bn}) for bn in rows]

    def materialize(root):
        bn = root.op.knobs["BN"]
        if bn in refused.get(root.op.name, ()):
            raise RuleSkipped("kernel binder refuses this row's projection ownership", reject=True)
        return KernelOp(body=[Smem(name="buf", extents=(64,), dtype="float")], name=root.op.name, knobs={"BN": bn})

    def stamp(root):
        if "LOOPIFY" in root.op.knobs:
            raise RuleSkipped("already stamped")
        return replace(root.op, knobs={**root.op.knobs, "LOOPIFY": 0})

    passes = []
    for i, (name, fn) in enumerate((("__cut__", cut), ("__schedule__", schedule), ("__materialize__", materialize), ("__stamp__", stamp))):
        rule = Rule(name=name, pattern=[Pattern(name="root", op_type=TileOp)], rewrite=fn, param_names=("root",))
        rule.pass_ = Pass(name=name, rules=[rule], index=i)
        passes.append(rule.pass_)
    return Pipeline(passes=passes)


def _elect_composed_route(monkeypatch) -> None:
    """The prior ranks the widest ``BN`` first; every composed route prices below its fused side,
    and the fused root is disqualified (``inf`` — every measured variant of it failed)."""
    import emmy.compiler.pipeline.search.policy.greedy as greedy_mod

    monkeypatch.setattr(greedy_mod, "_load_prior_safe", lambda: _BiggestBNFirstPrior())
    monkeypatch.setattr(greedy_mod, "_price_graph", lambda *_: 1.0)
    monkeypatch.setattr(greedy_mod, "_price_op_leaf", lambda fp, *_: float("inf") if fp.root_op.name == "k_test" else 10.0)


def _kernels(terminal: Graph) -> dict[str, tuple[str, int]]:
    return {nid: (n.op.name, n.op.knobs["BN"]) for nid, n in terminal.nodes.items() if isinstance(n.op, KernelOp)}


def test_refused_piece_rows_rerank_within_the_composed_route(monkeypatch):
    # The consumer's first two rows are refused; the third binds. Two retries, both re-ranking the
    # same piece, and the route never surrenders to the fused root.
    _elect_composed_route(monkeypatch)
    pipeline = _composed_route_pipeline({"k_residual": {24, 16}}, rows=(8, 16, 24))
    terminal = pipeline.run(_graph_with_tile(), ctx=_small_smem_ctx())
    assert _kernels(terminal) == {"y_ws": ("k_ws", 24), "y": ("k_residual", 8)}, "the refused piece re-ranks onto its third row"


def test_exhausted_piece_retires_only_its_own_cut(monkeypatch, caplog):
    # Every row of the residual's own residual piece is refused: the cut that minted it is retired
    # at its fork and re-priced, while the root's cut — an evidence decision — stays. Retirement
    # is logged loudly with the rejection reason.
    import logging

    _elect_composed_route(monkeypatch)
    pipeline = _composed_route_pipeline({"k_residual2": {8, 16}}, cut_residual=True)
    with caplog.at_level(logging.WARNING, logger="emmy.compiler.pipeline"):
        terminal = pipeline.run(_graph_with_tile(), ctx=_small_smem_ctx())
    assert _kernels(terminal) == {"y_ws": ("k_ws", 16), "y": ("k_residual", 16)}, "the residual keeps fused; the root's cut survives"
    assert any("retir" in r.message and "projection ownership" in r.message for r in caplog.records if r.levelno == logging.WARNING)


def test_structural_pick_is_revisited_only_when_no_piece_row_binds(monkeypatch):
    # With no row of the residual binding and no cut of its own, the cut that minted it is the
    # root's — the fused root is the one resolution left, disqualified or not.
    _elect_composed_route(monkeypatch)
    terminal = _composed_route_pipeline({"k_residual": {8, 16}}).run(_graph_with_tile(), ctx=_small_smem_ctx())
    assert _kernels(terminal) == {"y": ("k_test", 16)}, "every row refused → the fused route is the fallback"
