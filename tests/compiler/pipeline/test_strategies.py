"""The strategy system: discovery, engine events, and the two discovered strategies.

The engine is IR-agnostic — it emits events (``RunStartEvent`` / ``SpliceEvent`` /
``SplicedEvent`` / ``PassEndEvent``) and every cross-cutting concern is a strategy class
discovered from the ``passes/`` top level (``pipeline.strategy.discovered_strategies``). These
tests pin the discovery contract, the event dispatch, and the two concerns' observable
behavior: op provenance (mint at decomposition, aggregate after, absent without the strategy)
and structural identity (every kernel stamped at birth — fusion end or mint splice — and one
read-API spelling).
"""

from __future__ import annotations

from emmy.compiler import provenance
from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F16
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import MatmulOp, RmsNormOp
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.pipeline import CUDA_PASSES, LOOP_PASSES, Pipeline
from emmy.compiler.pipeline.fork import flatten_leaves
from emmy.compiler.pipeline.knob import STRUCT_PREFIX
from emmy.compiler.pipeline.passes.identity import IdentityStrategy, structure_features
from emmy.compiler.pipeline.passes.provenance import ProvenanceStrategy
from emmy.compiler.pipeline.pipeline import Run
from emmy.compiler.pipeline.strategy import PipelineStrategy, discovered_strategies

_CTX = Context.from_target((12, 0))


def _matmul(m: int = 64, k: int = 64, n: int = 64) -> Graph:
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (Dim(m), Dim(k)), dtype=F16), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (Dim(k), Dim(n)), dtype=F16), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("o", (Dim(m), Dim(n)), dtype=F16), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]
    return g


def _norm_linear(m: int = 2, h: int = 16) -> Graph:
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (Dim(1), Dim(m), Dim(h)), dtype=F16), node_id="x")
    g.add_node(InputOp(), [], Tensor("wn", (Dim(h),), dtype=F16), node_id="wn")
    g.add_node(InputOp(), [], Tensor("w", (Dim(h), Dim(h)), dtype=F16), node_id="w")
    g.add_node(RmsNormOp(), ["x", "wn"], Tensor("xn", (Dim(1), Dim(m), Dim(h)), dtype=F16), node_id="xn")
    g.add_node(MatmulOp(), ["xn", "w"], Tensor("y", (Dim(1), Dim(m), Dim(h)), dtype=F16), node_id="y")
    g.inputs, g.outputs = ["x", "wn", "w"], ["y"]
    return g


def _resolve(passes, graph):
    """Option-0 resolution over a freshly built pipeline (strategies discovered)."""
    return Run(pipeline=Pipeline.build(passes), ctx=_CTX).resolve(graph, lambda fp: flatten_leaves(fp.options)[0])


# --- discovery --------------------------------------------------------------------------------


def test_discovery_finds_the_two_strategies_once() -> None:
    """Every ``PipelineStrategy`` subclass defined in a ``passes/`` top-level module is discovered,
    instantiated once (shared instances), in deterministic class-name order."""
    found = discovered_strategies()
    assert [type(s).__name__ for s in found] == ["IdentityStrategy", "ProvenanceStrategy"]
    assert found is discovered_strategies(), "instances are cached and shared"
    assert all(isinstance(s, PipelineStrategy) for s in found)


def test_built_pipelines_carry_the_discovered_strategies() -> None:
    assert Pipeline.build(LOOP_PASSES).strategies == discovered_strategies()
    assert Pipeline.from_pattern([]).strategies == (), "test shims carry no strategies"


# --- provenance -------------------------------------------------------------------------------


def test_decomposition_mints_fresh_pieces_and_fusion_aggregates() -> None:
    """The matmul's decomposition pieces each become a distinct piece of the 'o' origin
    (mint); fusion then aggregates them back so the fused kernel covers the origin."""
    out, _ = _resolve(LOOP_PASSES, _matmul())
    kernels = [(nid, n) for nid, n in out.nodes.items() if isinstance(n.op, LoopOp)]
    assert kernels, "the matmul must fuse into at least one loop kernel"
    totals = provenance.totals(out)
    assert "o" in totals, "the original op is an origin"
    covered = set()
    for _nid, node in kernels:
        prov = provenance.get(node)
        assert prov, "every kernel carries provenance"
        covered |= set(prov.get("o", {}).get("pieces", []))
    assert covered == totals["o"], "the kernels together cover every piece of the origin"


def test_a_pipeline_without_the_provenance_strategy_has_no_provenance() -> None:
    """PipelineStrategy-scoped concern: strip ProvenanceStrategy from the pipeline and NO node carries
    provenance — the graph and engine hold none of it. (Identity is kept so kernels still stamp.)"""
    pipeline = Pipeline.build(LOOP_PASSES)
    stripped = Pipeline(
        passes=pipeline.passes,
        strategies=tuple(s for s in pipeline.strategies if not isinstance(s, ProvenanceStrategy)),
    )
    out, _ = Run(pipeline=stripped, ctx=_CTX).resolve(_matmul(), lambda fp: flatten_leaves(fp.options)[0])
    assert all(not provenance.get(n) for n in out.nodes.values()), "no strategy → no provenance anywhere"


# --- identity ---------------------------------------------------------------------------------


def _identity() -> IdentityStrategy:
    return next(s for s in discovered_strategies() if isinstance(s, IdentityStrategy))


def test_every_fusion_born_kernel_is_stamped_at_the_loop_terminal() -> None:
    out, _ = _resolve(LOOP_PASSES, _matmul())
    for nid, node in out.nodes.items():
        if isinstance(node.op, LoopOp):
            stamped = {k: v for k, v in node.op.knobs.items() if k.startswith(STRUCT_PREFIX)}
            assert stamped, f"{nid} must carry its structural identity at the loop terminal"
            assert stamped == structure_features(node.op.body, out), "the stamp IS structure_features of the final body"


def test_minted_pieces_are_stamped_with_their_own_identity(monkeypatch) -> None:
    """A cut minted during lowering is stamped at the splice event — each piece with features of
    its OWN body — and attributed to the kernel it decomposed (``Op.source``)."""
    monkeypatch.setenv("EMMY_PLACE", "cut")
    out, _ = _resolve(CUDA_PASSES, _norm_linear())
    kernels = {nid: n for nid, n in out.nodes.items() if getattr(n.op, "kernel_source", None)}
    assert any("__cut_" in nid for nid in kernels), f"the pin must cut: {list(kernels)}"
    sigs = set()
    identity = _identity()
    for nid, node in kernels.items():
        loop_ancestor = next(op for op in node.op.source_chain() if isinstance(op, LoopOp))
        sig = identity.signature(loop_ancestor)
        assert sig, f"{nid} has no structural identity"
        sigs.add(sig)
    assert len(sigs) == len(kernels), "each piece featurizes as itself, not as its parent"


def test_read_api_is_knobs_first_and_compute_equal() -> None:
    """``signature`` serves the stamped row when present and computes the same values when not —
    the one spelling of identity, with no ordering dependence."""
    out, _ = _resolve(LOOP_PASSES, _matmul())
    identity = _identity()
    for node in out.nodes.values():
        if not isinstance(node.op, LoopOp):
            continue
        stamped_sig = identity.signature(node.op)
        bare = type(node.op)(body=node.op.body)  # an unstamped twin of the same body
        assert identity.signature(bare, out) == stamped_sig
        assert identity.op_sig(bare, out) == identity.op_sig(node.op)


# --- events -----------------------------------------------------------------------------------


def test_events_fire_in_loop_order() -> None:
    """A run-scoped observer sees run start, then splices (with receipts), then pass ends —
    the engine's own moments, one protocol."""

    class Recorder(PipelineStrategy):
        def __init__(self) -> None:
            self.events: list[str] = []

        def on_run_start(self, e) -> None:
            self.events.append("run_start")

        def on_splice(self, e) -> None:
            self.events.append(f"splice:{e.pass_name}")

        def on_spliced(self, e) -> None:
            assert e.receipt.redirected, "the receipt names what was redirected"
            self.events.append(f"spliced:{e.pass_name}")

        def on_pass_end(self, e) -> None:
            self.events.append(f"pass_end:{e.pass_name}")

    rec = Recorder()
    run = Run(pipeline=Pipeline.build(LOOP_PASSES).with_strategies(rec), ctx=_CTX)
    run.resolve(_matmul(), lambda fp: flatten_leaves(fp.options)[0])
    assert rec.events[0] == "run_start"
    assert any(ev.startswith("splice:frontend/decomposition") for ev in rec.events)
    # Every pre-splice event has its post-splice receipt event.
    assert sum(ev.startswith("splice:") for ev in rec.events) == sum(ev.startswith("spliced:") for ev in rec.events)
    pass_ends = [ev.removeprefix("pass_end:") for ev in rec.events if ev.startswith("pass_end:")]
    assert pass_ends == LOOP_PASSES, "one pass-end per pass, in pipeline order"
