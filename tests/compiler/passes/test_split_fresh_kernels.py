"""A cross-CTA split mints BRAND-NEW kernels — the invariant ``035_split_reduce`` realizes.

The split is a STRUCTURAL fork in the cut phase after ``030_cut``, decided BEFORE scheduling (a ``REDUCE`` pin's
``g<n>[a|k]`` half is authoritative over it); its rewrite returns a different set of nodes, and
those nodes are new kernels:

- they carry NO knob, NO placement and NO schedule slice of the kernel they replace;
- each reaches ``040_schedule`` and decides its own row, exactly like a newly lifted Fold tree;
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
from emmy.compiler.dtype import BF16, F16, F32
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import MatmulOp
from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.stmt import Loop
from emmy.compiler.ir.tensor.ir import ElementwiseOp, ReduceOp
from emmy.compiler.ir.tile.ir import TileOp
from emmy.compiler.pipeline import CUDA_PASSES, TILE_PASSES, Pipeline
from emmy.compiler.pipeline.fork import iter_leaves
from emmy.compiler.pipeline.knob import STRUCT_PREFIX, decision_view, family_of
from emmy.compiler.pipeline.pipeline import Run

_CTX = Context.from_target((12, 0))


def _matmul(m: int = 128, k: int = 512, n: int = 128, *, out_dtype=F16) -> Graph:
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (Dim(m), Dim(k)), dtype=F16), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (Dim(k), Dim(n)), dtype=F16), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("o", (Dim(m), Dim(n)), dtype=out_dtype), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]
    return g


def _sum(*, dtype=F16) -> Graph:
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (Dim(4), Dim(512)), dtype=dtype), node_id="x")
    g.add_node(ReduceOp(axis=1), ["x"], Tensor("s", (Dim(4), Dim(1)), dtype=dtype), node_id="s")
    g.inputs, g.outputs = ["x"], ["s"]
    return g


def _multi_output_matmul() -> Graph:
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (Dim(128), Dim(512)), dtype=F16), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (Dim(512), Dim(512)), dtype=F16), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("o", (Dim(128), Dim(512)), dtype=F16), node_id="o")
    g.add_node(ElementwiseOp("negative"), ["o"], Tensor("o_neg", (Dim(128), Dim(512)), dtype=F16), node_id="o_neg")
    g.inputs, g.outputs = ["a", "b"], ["o", "o_neg"]
    return g


def _resolve(passes, graph=None):
    """Option-0 resolution — the no-evidence emission-order pick, so the assertions are about what
    the pipeline BUILDS rather than about what a prior happens to rank first."""
    return Run(pipeline=Pipeline.build(passes), ctx=_CTX).resolve(graph or _matmul(), lambda fp: next(iter_leaves(fp.options)))


def _kernels(out) -> dict[str, dict]:
    return {nid: dict(n.op.knobs) for nid, n in out.nodes.items() if getattr(n.op, "kernel_source", None)}


def _tile_pieces(graph=None) -> list[TileOp]:
    loop, _ = _resolve(
        ["frontend/decomposition", "frontend/optimization", "loop/lifting", "loop/fusion", "loop/stamp"],
        graph,
    )
    tiled, _ = Run(pipeline=Pipeline.build(["lowering/tile"]), ctx=_CTX).resolve(
        loop,
        lambda fp: next(iter_leaves(fp.options)),
    )
    return [node.op for node in tiled.nodes.values() if isinstance(node.op, TileOp)]


def _contains_raw_loop(value) -> bool:
    if isinstance(value, Loop):
        return True
    if isinstance(value, Fold):
        return any(_contains_raw_loop(edge) for edge in value.operands) or any(_contains_raw_loop(stmt) for stmt in value.lift.body)
    return any(_contains_raw_loop(stmt) for body in value.nested() for stmt in body)


def test_the_split_returns_two_kernels(monkeypatch) -> None:
    """The deferred arm splices a partial + finalize; the atomic arm one kernel. Either way the
    rewrite returns a GRAPH — including the one-node atomic arm, which replaces the kernel rather
    than deciding it further. An op rebind would merge the replaced kernel's knobs forward and
    would not restart the pass scan, so the piece would never reach its own schedule fork."""
    monkeypatch.setenv("EMMY_REDUCE", "g2k")
    assert set(_kernels(_resolve(CUDA_PASSES)[0])) == {"o", "o__partial"}
    monkeypatch.setenv("EMMY_REDUCE", "g2a")
    assert set(_kernels(_resolve(CUDA_PASSES, _matmul(out_dtype=F32))[0])) == {"o"}


def test_split_preserves_every_fused_output(monkeypatch) -> None:
    """The finalize kernel retains all ports of the fused kernel, not only its primary output."""
    monkeypatch.setenv("EMMY_REDUCE", "g2k")
    out, _ = _resolve(TILE_PASSES, _multi_output_matmul())
    assert out.outputs == ["o", "o_neg"]
    owner = out.producer("o")
    assert owner is not None and owner is out.producer("o_neg")
    assert set(owner.buffer_names()) == {"o", "o_neg"}
    assert f"{owner.id}__partial" in out.nodes
    out.validate()


@pytest.mark.parametrize("dtype", [F16, BF16])
def test_low_precision_output_refuses_direct_atomic_split(monkeypatch, dtype) -> None:
    """F16/BF16 outputs would round every CTA partial; contraction and plain reduce fail closed."""
    monkeypatch.setenv("EMMY_REDUCE", "g2a")
    for graph in (_matmul(out_dtype=dtype), _sum(dtype=dtype)):
        with pytest.raises(ValueError, match="direct atomic REDUCE.*output storage"):
            _resolve(TILE_PASSES, graph)


def test_finalize_keeps_projection_input_edges(monkeypatch) -> None:
    """A deferred finalize keeps every external buffer its projection reads. The CUDA op's
    argument order cannot name a buffer absent from the graph node's inputs."""
    graph = _matmul()
    graph.add_node(InputOp(), [], Tensor("bias", (Dim(128),), dtype=F16), node_id="bias")
    graph.add_node(ElementwiseOp("add"), ["o", "bias"], Tensor("biased", (Dim(128), Dim(128)), dtype=F16), node_id="biased")
    graph.inputs, graph.outputs = ["a", "b", "bias"], ["biased"]
    monkeypatch.setenv("EMMY_REDUCE", "g2k")

    out, _ = _resolve(CUDA_PASSES, graph)
    finalize = out.nodes["biased"]
    assert finalize.inputs == ["biased__partial", "bias"]
    assert finalize.op.arg_order == ("biased__partial", "bias", "biased")


def test_no_piece_inherits_the_kernel_it_replaces(monkeypatch) -> None:
    """The pieces leave the rewrite UNSCHEDULED: no placement, no schedule slice, no decided knob,
    and a structural stamp of their own. (The partial used to arrive wearing the pre-split kernel's
    whole row — 21 ``S_*`` features describing a body it no longer had — and the finalize
    already-placed with no knobs at all: no fork, no identity, untunable.)"""
    monkeypatch.setenv("EMMY_REDUCE", "g2k")
    pieces = _tile_pieces()
    assert len(pieces) == 2, "the split must have produced two kernels"
    for piece in pieces:
        # Each SCHEDULED itself — so what it carries is its own row, keyed against its own tree.
        assert piece.place.is_mapped, "040_schedule must pick each piece up"
        assert not any(str(v).startswith("g") for k, v in piece.knobs.items() if family_of(k) == "REDUCE"), (
            f"a piece must not carry the split it came from: {decision_view(piece.knobs)}"
        )
        assert {k for k in piece.knobs if k.startswith(STRUCT_PREFIX)}, "…and its own structural stamp"


@pytest.mark.parametrize("graph", [_matmul(), _sum()])
def test_split_reductions_remain_fold_trees(monkeypatch, graph) -> None:
    """Both split-K and plain-reduce pieces preserve Fold trees without embedded Loop IR."""
    monkeypatch.setenv("EMMY_REDUCE", "g2k")
    pieces = _tile_pieces(graph)
    assert len(pieces) == 2
    assert not any(_contains_raw_loop(piece.op) for piece in pieces)


def test_each_piece_decides_its_own_row(monkeypatch) -> None:
    """The pieces reach the schedule fork independently — each gets its own decision in the trace
    and leaves with a full schedule row. (Before, the partial arrived pre-decided and the finalize
    arrived with no row at all; neither was ever offered a fork.)"""
    monkeypatch.setenv("EMMY_REDUCE", "g2k")
    out, trace = _resolve(CUDA_PASSES)
    kernels = _kernels(out)
    assert set(kernels) == {"o", "o__partial"}
    for row in kernels.values():
        families = {family_of(k) for k in row}
        assert {"WORK", "RASTER", "REDUCE"} <= families, row
    scheduled = {d.node_id for d in trace if "schedule" in d.rule_name}
    assert scheduled >= {"o", "o__partial"}, f"each piece must be offered its own schedule fork, saw {scheduled}"


def test_each_piece_carries_its_own_structural_identity(monkeypatch) -> None:
    """A piece featurizes as ITSELF. Without this the partial joined the pre-split kernel's
    evidence — the same signature for a kernel doing half the reduction. The identity strategy
    stamps each fragment at the splice boundary."""
    monkeypatch.setenv("EMMY_REDUCE", "g2k")
    stamps = [{k: v for k, v in row.items() if k.startswith(STRUCT_PREFIX)} for row in _kernels(_resolve(CUDA_PASSES)[0]).values()]
    assert all(stamps), "every piece must carry a structural stamp"
    assert stamps[0] != stamps[1], "the pieces are structurally different kernels"


def test_a_pieces_features_are_read_off_its_reconstituted_body(monkeypatch) -> None:
    """A piece stays in Tile IR, so identity temporarily lowers its Fold with the output specifications
    restored and the free axes nested around it. Both halves matter:

    - the STORES must come back, or a piece reports ``S_n_write = 0`` while every kernel that
      reached the stamp through Loop IR reports its writes;
    - the FREE AXES must be re-nested, since recognition peels them onto the placement — a piece
      stamped off the bare lowered body reports ``S_ext_n_free_axis = 0`` and every extent feature
      the occupancy and wave models are built on collapses to 1.

    And the finalize's cross-partition fold must READ as a reduce: the loop carries an explicit
    ``PLANAR`` role rather than leaning on ``Loop.is_reduce``'s structural ``Accum`` fallback —
    without a stated role the finalize would featurize as a 3-deep parallel nest that reduces
    nothing whenever the fallback missed. (Annotating it is emission-neutral: the kernel source
    digests identically across both arms and both carrier kinds.)

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
    graph = _matmul(out_dtype=F32) if pin.endswith("a") else _matmul()
    kernels = _kernels(_resolve(CUDA_PASSES, graph)[0])
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


def _softmax_scale_chain() -> Graph:
    """``softmax(x · c)`` with a broadcast scalar — the CHAIN form: the head fold is a BODY member
    of its projection wrapper (``head``'s sweep case is the same family), and its lift CAPTURES
    ``in0``, the scalar the projection loads once per cell."""
    from emmy.compiler.ir.frontend.ir import SoftmaxOp

    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (Dim(4), Dim(512)), dtype=F16), node_id="x")
    g.add_node(InputOp(), [], Tensor("c", (Dim(1),), dtype=F16), node_id="c")
    g.add_node(ElementwiseOp("multiply"), ["x", "c"], Tensor("xs", (Dim(4), Dim(512)), dtype=F16), node_id="xs")
    g.add_node(SoftmaxOp(axis=-1), ["xs"], Tensor("y", (Dim(4), Dim(512)), dtype=F16), node_id="y")
    g.inputs, g.outputs = ["x", "c"], ["y"]
    return g


def test_chain_split_carries_the_captured_prologue_and_strips_the_finalize(monkeypatch) -> None:
    """The body-resident (chain-form) head fold splits WHOLE: the partial carries the prologue
    cone its slice still captures (the ``c`` load — a bare sliced fold would leave ``in0``
    dangling, the ``k_softmax__partial`` nvcc miscompile), and the finalize's epilogue DROPS the
    original fold (keeping it would re-run the whole reduction per cell and shadow the workspace
    states — its only remaining axis fold is the ``_state_fold`` over the two partitions)."""
    from emmy.compiler.ir.stmt import Body
    from emmy.compiler.ir.tile.path import sites

    monkeypatch.setenv("EMMY_REDUCE", "g2k")
    out, _ = _resolve(TILE_PASSES, _softmax_scale_chain())
    tiles = {nid: n.op for nid, n in out.nodes.items() if isinstance(n.op, TileOp)}
    partial = next(op for nid, op in tiles.items() if nid.endswith("__partial"))
    finalize = next(op for nid, op in tiles.items() if not nid.endswith("__partial"))
    assert "c" in {load.input for load in Body.coerce(tuple(partial.op.lower())).loads}, (
        "the partial must carry the captured prologue's defining load"
    )
    extents = {
        s.node.axis.extent.as_static()
        for s in sites(finalize.op)
        if isinstance(s.node, Fold) and s.node.axis is not None and s.node.axis.extent.is_static
    }
    assert extents == {2}, f"the finalize may fold only the two partitions, got axis extents {sorted(extents)}"


def test_sweep_resident_head_fold_refuses_the_split(monkeypatch) -> None:
    """A head fold that READS the boundary store's sweep axis lands inside the sweep ``Loop``
    ``apply_output_specs`` wraps — neither an operand nor a top-level projection member — so the
    realization cannot strip it from the epilogue and the split declines at the offer (a pin
    raises the recorded refusal; the catalog arm keeps the unsplit tree)."""
    from types import SimpleNamespace

    from emmy.compiler.ir.axis import Axis
    from emmy.compiler.ir.elementwise import ElementwiseImpl
    from emmy.compiler.ir.expr import Var
    from emmy.compiler.ir.pure import Lambda, M
    from emmy.compiler.ir.stmt import Assign, Body, Load, Write
    from emmy.compiler.ir.tile import OutputSpec, Placement
    from emmy.compiler.ir.tile.ops import head
    from emmy.compiler.pipeline.passes.lowering.tile._split import _projection_refusal, split_forks

    init, combine = M(ElementwiseImpl("add"), names=("acc",))
    # The fold reads ``in0``, a name the projection body defines, so normalization keeps it a BODY
    # member (a free-standing fold would be hoisted to an operand edge) — and that prologue reads
    # the sweep axis ``j``, so the whole chain lands inside the sweep ``Loop``.
    fold = Fold(
        axis=Axis("k", Dim(16)),
        lift=Lambda(
            params=("k",),
            body=Body(
                (
                    Load(name="v", input="x", index=(Var("k"), Var("j"))),
                    Assign(name="p", op="multiply", args=("v", "in0")),
                )
            ),
            results=("p",),
        ),
        init=init,
        combine=combine,
    )
    wrapper = Fold.projection(
        body=Body(
            (
                Load(name="in0", input="c", index=(Var("j"),)),
                fold,
                Assign(name="y", op="multiply", args=("acc", "in0")),
            )
        )
    )
    tile = TileOp(
        op=wrapper,
        place=Placement(free=(Axis("i", Dim(4)),)),
        output_specs=(OutputSpec(write=Write(output="o", index=(Var("i"), Var("j")), value="y"), sweep=Axis("j", Dim(8))),),
    )
    node = head(tile.op)
    assert node is not None and node.axis is not None
    why = _projection_refusal(tile, node)
    assert why is not None and "cannot strip" in why
    root = SimpleNamespace(op=tile, id="o")
    for var in ("EMMY_REDUCE", "EMMY_WORK"):
        monkeypatch.delenv(var, raising=False)
    options = split_forks(None, root)
    assert options is not None and len(options) == 1, "only the unsplit arm may survive the residence refusal"
    monkeypatch.setenv("EMMY_REDUCE", "g2k")
    with pytest.raises(ValueError, match="cannot strip"):
        split_forks(None, root)
