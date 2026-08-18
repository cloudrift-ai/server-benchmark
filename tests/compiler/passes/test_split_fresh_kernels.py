"""A cross-CTA split mints BRAND-NEW kernels — the invariant `030_split_reduce` realizes.

The split is an ordinary ``REDUCE`` value decided at the ordinary schedule fork; what makes it
structural is that the rewrite returns a different set of nodes. Those nodes are new kernels:

- they carry NO knob, NO placement and NO schedule slice of the kernel they replace;
- each reaches ``020_schedule`` and decides its own row, exactly like a freshly recognized term;
- each carries structural features re-derived from its OWN body, so it is separately identifiable
  to the evidence store;
- the split is CONSUMED by the kernel that realizes it — the sliced axis is a window of its
  parent, so nothing partitions it again and a pin's remaining row is what reaches the pieces.

Off-GPU except where noted: the pieces compile through the full CUDA pass list, so kernel sets and
knob rows are asserted without a device. Numerics for every carrier and both arms live in the
``e2e`` reduce-coverage matrix.
"""

from __future__ import annotations

import pytest

from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F16, F32
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import MatmulOp
from emmy.compiler.ir.tile.ir import TileOp
from emmy.compiler.pipeline import CUDA_PASSES, TILE_PASSES, Pipeline
from emmy.compiler.pipeline.fork import flatten_leaves
from emmy.compiler.pipeline.knob import SCHEDULE_FAMILIES, STRUCT_PREFIX, decision_view, family_of
from emmy.compiler.pipeline.pipeline import Run

_CTX = Context.from_target((12, 0))


def _matmul(m: int = 128, k: int = 512, n: int = 128) -> Graph:
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (Dim(m), Dim(k)), dtype=F16), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (Dim(k), Dim(n)), dtype=F16), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("o", (Dim(m), Dim(n)), dtype=F16), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]
    return g


def _resolve(passes, graph=None):
    """Option-0 resolution — the no-evidence emission-order pick, so the assertions are about what
    the pipeline BUILDS rather than about what a prior happens to rank first."""
    return Run(pipeline=Pipeline.build(passes), ctx=_CTX).resolve(graph or _matmul(), lambda fp: flatten_leaves(fp.options)[0])


def _kernels(out) -> dict[str, dict]:
    return {nid: dict(n.op.knobs) for nid, n in out.nodes.items() if getattr(n.op, "kernel_source", None)}


def test_the_split_returns_two_kernels(monkeypatch) -> None:
    """The deferred arm splices a partial + finalize; the atomic arm one kernel. Either way the
    rewrite returns a GRAPH — including the one-node atomic arm, which replaces the kernel rather
    than deciding it further. An op rebind would merge the replaced kernel's knobs forward and
    would not restart the pass scan, so the piece would never reach its own schedule fork."""
    monkeypatch.setenv("EMMY_REDUCE", "g2k")
    assert set(_kernels(_resolve(CUDA_PASSES)[0])) == {"o", "o__partial"}
    monkeypatch.setenv("EMMY_REDUCE", "g2a")
    assert set(_kernels(_resolve(CUDA_PASSES)[0])) == {"o"}


def test_no_piece_inherits_the_kernel_it_replaces(monkeypatch) -> None:
    """The pieces leave the rewrite UNSCHEDULED: no placement, no schedule slice, no decided knob,
    and a structural stamp of their own. (The partial used to arrive wearing the pre-split kernel's
    whole row — 21 ``S_*`` features describing a body it no longer had — and the finalize
    already-placed with no knobs at all: no fork, no identity, untunable.)"""
    monkeypatch.setenv("EMMY_REDUCE", "g2k")
    out, _ = _resolve(["frontend/decomposition", "frontend/optimization", "loop/lifting", "loop/prefusion", "loop/fusion", "loop/stamp"])
    # Drive the tile pass by hand so the pieces are caught between the splice and the schedule.
    pieces: list[TileOp] = []

    def catch(fp):
        leaf = flatten_leaves(fp.options)[0]
        return leaf

    graph, _ = Run(pipeline=Pipeline.build(["lowering/tile"]), ctx=_CTX).resolve(out, catch)
    pieces = [n.op for n in graph.nodes.values() if isinstance(n.op, TileOp)]
    assert len(pieces) == 2, "the split must have produced two kernels"
    for piece in pieces:
        # Each SCHEDULED itself — so what it carries is its own row, keyed against its own tree.
        assert piece.place.is_mapped, "020_schedule must pick each piece up"
        assert not any(str(v).startswith("g") for k, v in piece.knobs.items() if family_of(k) == "REDUCE"), (
            f"a piece must not carry the split it came from: {decision_view(piece.knobs)}"
        )
        assert {k for k in piece.knobs if k.startswith(STRUCT_PREFIX)}, "…and its own structural stamp"


def test_each_piece_decides_its_own_row(monkeypatch) -> None:
    """The pieces reach the schedule fork independently — each gets its own decision in the trace
    and leaves with a full schedule row. (Before, the partial arrived pre-decided and the finalize
    arrived with no row at all; neither was ever offered a fork.)"""
    monkeypatch.setenv("EMMY_REDUCE", "g2k")
    out, trace = _resolve(CUDA_PASSES)
    kernels = _kernels(out)
    assert set(kernels) == {"o", "o__partial"}
    for row in kernels.values():
        assert set(SCHEDULE_FAMILIES) <= {family_of(k) for k in row}, row
    scheduled = {d.node_id for d in trace if "schedule" in d.rule_name}
    assert scheduled >= {"o", "o__partial"}, f"each piece must be offered its own schedule fork, saw {scheduled}"


def test_each_piece_carries_its_own_structural_identity(monkeypatch) -> None:
    """A piece featurizes as ITSELF. Without this the partial joined the pre-split kernel's
    evidence — the same signature for a kernel doing half the reduction. Nothing in the rule stamps
    it: ``005_stamp`` picks up any op with no ``S_*`` on the pass-scan restart."""
    monkeypatch.setenv("EMMY_REDUCE", "g2k")
    stamps = [{k: v for k, v in row.items() if k.startswith(STRUCT_PREFIX)} for row in _kernels(_resolve(CUDA_PASSES)[0]).values()]
    assert all(stamps), "every piece must carry a structural stamp"
    assert stamps[0] != stamps[1], "the pieces are structurally different kernels"


def test_a_pieces_features_are_read_off_its_reconstituted_body(monkeypatch) -> None:
    """A piece is minted as a loop-dialect kernel, so ``_piece`` has to BUILD the body: the term's
    lowered per-cell nest with the boundary stores put back, re-nested under its free axes. Both
    halves of that matter and both were once wrong to assume:

    - the STORES must come back, or a piece reports ``S_n_write = 0`` while every kernel that
      reached the stamp through the loop dialect reports its writes;
    - the FREE AXES must be re-nested, since recognition peels them onto the placement — a piece
      stamped off the bare lowered body reports ``S_ext_n_free_axis = 0`` and every extent feature
      the occupancy and wave models are built on collapses to 1.

    And the finalize's cross-partition fold must READ as a reduce: ``Loop.is_reduce``'s structural
    fallback wants an ``Accum`` / ``Mma`` carrier and this one's is a ``StateMerge``, so the loop
    carries an explicit ``PLANAR`` role — without it the finalize featurizes as a 3-deep parallel
    nest that reduces nothing. (Annotating it is emission-neutral: the kernel source digests
    identically across both arms and both carrier kinds.)

    Read against the pieces' known geometry: the partial's frees are ``(ksplit=2, m=128, n=128)``
    and the finalize's the grid ``(m=128, n=128)`` over a 2-wide fold."""
    monkeypatch.setenv("EMMY_REDUCE", "g2k")
    kernels = _kernels(_resolve(CUDA_PASSES)[0])
    partial, finalize = kernels["o__partial"], kernels["o"]
    for name, row in (("partial", partial), ("finalize", finalize)):
        assert row.get("S_n_write") == 1.0, f"{name}: the boundary store must come back as a Write — {row.get('S_n_write')}"
    assert (partial["S_ext_n_free_axis"], partial["S_ext_free_prod"]) == (3.0, 2.0 * 128 * 128), partial
    assert (finalize["S_ext_n_free_axis"], finalize["S_ext_free_prod"]) == (2.0, 128.0 * 128), finalize
    assert finalize["S_ext_reduce_prod"] == 2.0, f"the cross-partition fold must read as a reduce — {finalize}"


@pytest.mark.parametrize("pin", ["g2k", "g4k", "g2a"])
def test_the_split_is_consumed_by_the_kernel_that_realizes_it(monkeypatch, pin) -> None:
    """One split per axis. The pieces are indistinguishable from fresh kernels, so a pin that
    applies to fresh kernels applies to them too — and the sliced axis is the only thing that
    records the partition already happened. Without that reading the partial re-splits its own
    slice on every sweep: K=512 → 256 → … → 1, ending in a raise."""
    monkeypatch.setenv("EMMY_REDUCE", pin)
    kernels = _kernels(_resolve(CUDA_PASSES)[0])
    assert len(kernels) <= 2, f"{pin}: the split must not cascade — {sorted(kernels)}"
    assert not [v for row in kernels.values() for k, v in row.items() if family_of(k) == "REDUCE" and str(v).startswith("g")]


def test_a_pin_hands_its_remaining_row_to_the_pieces(monkeypatch) -> None:
    """Only the cross-CTA half is consumed. ``g2k/coop`` splits the kernel AND asks each piece for
    the cooperative fold — which is what a pin means: a statement about how kernels run, minus the
    part that has already been realized."""
    monkeypatch.setenv("EMMY_REDUCE", "g2k/coop")
    monkeypatch.setenv("EMMY_WORK", "t64")
    from emmy.compiler.ir.tensor.ir import ReduceOp

    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (Dim(4), Dim(1024)), dtype=F32), node_id="x")
    g.add_node(ReduceOp(axis=1), ["x"], Tensor("s", (Dim(4), Dim(1)), dtype=F32), node_id="s")
    g.inputs, g.outputs = ["x"], ["s"]
    kernels = _kernels(_resolve(CUDA_PASSES, g)[0])
    assert len(kernels) == 2
    partial = next(row for nid, row in kernels.items() if nid.endswith("__partial"))
    assert any(str(v) == "coop" for k, v in partial.items() if family_of(k) == "REDUCE"), partial


def test_the_split_node_is_priced_as_the_sum_of_its_pieces(monkeypatch) -> None:
    """A kernel that splits has no latency of its own — it does not run. Its estimate is the Σ over
    the kernels the resolution ends with, which is what lets the split row be compared against the
    rows that keep one kernel."""
    from emmy.compiler.pipeline.search.policy import greedy

    monkeypatch.setenv("EMMY_REDUCE", "g2k")
    terminal, trace = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=_CTX).resolve(_matmul(), greedy.greedy_decide(prior=None))
    kernels = [nid for nid, n in terminal.nodes.items() if isinstance(n.op, TileOp)]
    assert len(kernels) == 2, "the pinned split must produce two kernels to price"

    class _Flat:
        def mean_scores(self, rows):
            return [7.0] * len(rows)

    scored = {d.node_id: d.score for d in trace}
    total = greedy._resolved_price(terminal, trace, _CTX, _Flat())
    assert total == pytest.approx(sum(scored.get(nid) if scored.get(nid) is not None else 7.0 for nid in kernels))
