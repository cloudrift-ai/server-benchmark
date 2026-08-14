"""Phase 4 — placement routing + the cut realizer.

ROUTING entries (PLACE-only golden knobs) and authoritative ``PLACE`` pins resolve a cut BEFORE
any schedule fork exists; the realizer splits the recognized tree at the seam into un-mapped
``LoopOp`` pieces that re-recognize as fresh roots (recursive — a piece's entry may itself cut).
Fuse is the evidence-free default by absence when generic capability and workspace invariants
admit the fused schedule. These tests run off-GPU: the pieces compile through the full
CUDA pass list with deterministic option-0 resolution, so kernel SETS and buffer wiring are
asserted without a device (GPU accuracy is covered by the e2e smoke on the 5090 host).
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F16
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.frontend.ir import MatmulOp, ReshapeOp, RmsNormOp
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
from emmy.compiler.ir.tensor.ir import ElementwiseOp
from emmy.compiler.pipeline import CUDA_PASSES, LOOP_PASSES, TILE_PASSES, Pipeline
from emmy.compiler.pipeline.fork import flatten_leaves
from emmy.compiler.pipeline.pipeline import Run
from emmy.compiler.pipeline.search.data.shape import ShapeKey
from emmy.compiler.pipeline.search.golden import GoldenRecord


def _compile(graph, knobs_env: str | None, monkeypatch, ctx: Context | None = None):
    if knobs_env is None:
        monkeypatch.delenv("EMMY_KNOBS", raising=False)
        monkeypatch.delenv("EMMY_PLACE", raising=False)
    else:
        monkeypatch.setenv("EMMY_KNOBS", knobs_env)
    ctx = ctx or Context.from_target((12, 0))
    out, _ = Run(pipeline=Pipeline.build(CUDA_PASSES), ctx=ctx).resolve(graph, lambda fp: flatten_leaves(fp.options)[0])
    return out


def _kernel_ids(out) -> list[str]:
    return sorted(nid for nid, n in out.nodes.items() if getattr(n.op, "kernel_source", None))


def _inp(g: Graph, name: str, shape: tuple) -> None:
    g.add_node(op=InputOp(), inputs=[], output=Tensor(name, tuple(Dim(s) for s in shape), dtype=F16), node_id=name)


def _rms_graph(S: int = 64, H: int = 4096) -> Graph:
    g = Graph()
    _inp(g, "x", (S, H))
    _inp(g, "w", (H,))
    g.add_node(RmsNormOp(), ["x", "w"], Tensor("y", (Dim(S), Dim(H)), dtype=F16), node_id="y")
    g.inputs, g.outputs = ["x", "w"], ["y"]
    return g


def _norm_linear_graph(S: int = 32, H: int = 1024, inter: int = 3072) -> Graph:
    g = Graph()
    _inp(g, "x", (1, S, H))
    _inp(g, "wn", (H,))
    _inp(g, "w", (H, inter))
    g.add_node(RmsNormOp(), ["x", "wn"], Tensor("xn", (1, Dim(S), Dim(H)), dtype=F16), node_id="xn")
    g.add_node(MatmulOp(), ["xn", "w"], Tensor("y", (1, Dim(S), Dim(inter)), dtype=F16), node_id="y")
    g.inputs, g.outputs = ["x", "wn", "w"], ["y"]
    return g


def _activation_linear_graph(S: int = 2, H: int = 16, inter: int = 32) -> Graph:
    g = Graph()
    _inp(g, "x", (S, H))
    _inp(g, "w", (H, inter))
    g.add_node(ElementwiseOp("silu"), ["x"], Tensor("xa", (Dim(S), Dim(H)), dtype=F16), node_id="xa")
    g.add_node(MatmulOp(), ["xa", "w"], Tensor("y", (Dim(S), Dim(inter)), dtype=F16), node_id="y")
    g.inputs, g.outputs = ["x", "w"], ["y"]
    return g


def _sequential_linears_graph() -> Graph:
    """Two contractions fused maximally into a raw nested-reduction cell."""
    g = Graph()
    _inp(g, "x", (1, 1, 16))
    _inp(g, "w1", (16, 32))
    _inp(g, "w2", (32, 8))
    g.add_node(MatmulOp(), ["x", "w1"], Tensor("hidden", (1, 1, 32), dtype=F16), node_id="hidden")
    g.add_node(MatmulOp(), ["hidden", "w2"], Tensor("y", (1, 1, 8), dtype=F16), node_id="y")
    g.inputs, g.outputs = ["x", "w1", "w2"], ["y"]
    return g


def _norm_linear_rejoined_reshape_graph() -> Graph:
    """Norm→projection with the projection's ``[M,H,D]`` view flattened again."""
    g = Graph()
    _inp(g, "x", (2, 16))
    _inp(g, "wn", (16,))
    _inp(g, "w", (16, 16))
    g.add_node(RmsNormOp(), ["x", "wn"], Tensor("xn", (Dim(2), Dim(16)), dtype=F16), node_id="xn")
    g.add_node(MatmulOp(), ["xn", "w"], Tensor("linear", (Dim(2), Dim(16)), dtype=F16), node_id="linear")
    g.add_node(
        ReshapeOp((Dim(2), Dim(4), Dim(4))),
        ["linear"],
        Tensor("view", (Dim(2), Dim(4), Dim(4)), dtype=F16),
        node_id="view",
    )
    g.add_node(
        ReshapeOp((Dim(2), Dim(16))),
        ["view"],
        Tensor("y", (Dim(2), Dim(16)), dtype=F16),
        node_id="y",
    )
    g.inputs, g.outputs = ["x", "wn", "w"], ["y"]
    return g


def _norm_gated_mlp_graph(M: int = 1) -> Graph:
    """A closed stat prologue feeding two projections, SwiGLU, then a down projection."""
    g = Graph()
    _inp(g, "x", (M, 16))
    _inp(g, "wn", (16,))
    _inp(g, "wg", (16, 32))
    _inp(g, "wu", (16, 32))
    _inp(g, "wd", (32, 16))
    g.add_node(RmsNormOp(), ["x", "wn"], Tensor("xn", (Dim(M), Dim(16)), dtype=F16), node_id="xn")
    g.add_node(MatmulOp(), ["xn", "wg"], Tensor("gate", (Dim(M), Dim(32)), dtype=F16), node_id="gate")
    g.add_node(MatmulOp(), ["xn", "wu"], Tensor("up", (Dim(M), Dim(32)), dtype=F16), node_id="up")
    g.add_node(ElementwiseOp("silu"), ["gate"], Tensor("sg", (Dim(M), Dim(32)), dtype=F16), node_id="sg")
    g.add_node(ElementwiseOp("multiply"), ["sg", "up"], Tensor("act", (Dim(M), Dim(32)), dtype=F16), node_id="act")
    g.add_node(MatmulOp(), ["act", "wd"], Tensor("y", (Dim(M), Dim(16)), dtype=F16), node_id="y")
    g.add_node(ElementwiseOp("add"), ["x", "y"], Tensor("z", (Dim(M), Dim(16)), dtype=F16), node_id="z")
    g.inputs, g.outputs = ["x", "wn", "wg", "wu", "wd"], ["z"]
    return g


def _expanded_reduce_graph(M: int = 2, H: int = 8, K: int = 4, N: int = 16) -> Graph:
    """A compact ``[M,H]`` reduction broadcast to ``[M,H,N]`` before matmul."""
    g = Graph()
    _inp(g, "p", (M, H, K))
    _inp(g, "w", (N, H))
    producer = LoopOp(
        body=Body(
            (
                Loop(
                    axis=Axis("m", Dim(M)),
                    body=Body(
                        (
                            Loop(
                                axis=Axis("h", Dim(H)),
                                body=Body(
                                    (
                                        Loop(
                                            axis=Axis("r", Dim(K)),
                                            body=Body(
                                                (
                                                    Load(name="p_e", input="p", index=(Var("m"), Var("h"), Var("r"))),
                                                    Accum(name="p_sum", value="p_e", op="add", axes=("r",)),
                                                )
                                            ),
                                        ),
                                        Loop(
                                            axis=Axis("broadcast", Dim(N)),
                                            body=Body(
                                                (
                                                    Write(
                                                        output="expanded",
                                                        index=(Var("m"), Var("h"), Var("broadcast")),
                                                        value="p_sum",
                                                    ),
                                                )
                                            ),
                                        ),
                                    )
                                ),
                            ),
                        )
                    ),
                ),
            )
        )
    )
    g.add_node(producer, ["p"], Tensor("expanded", (Dim(M), Dim(H), Dim(N)), F16), node_id="expanded")
    consumer = LoopOp(
        body=Body(
            (
                Loop(
                    axis=Axis("cm", Dim(M)),
                    body=Body(
                        (
                            Loop(
                                axis=Axis("cn", Dim(N)),
                                body=Body(
                                    (
                                        Loop(
                                            axis=Axis("ch", Dim(H)),
                                            body=Body(
                                                (
                                                    Load(name="a", input="expanded", index=(Var("cm"), Var("ch"), Var("cn"))),
                                                    Load(name="b", input="w", index=(Var("cn"), Var("ch"))),
                                                    Assign(name="ab", op="multiply", args=("a", "b")),
                                                    Accum(name="out", value="ab", op="add", axes=("ch",)),
                                                )
                                            ),
                                        ),
                                        Write(output="y", index=(Var("cm"), Var("cn")), value="out"),
                                    )
                                ),
                            ),
                        )
                    ),
                ),
            )
        )
    )
    g.add_node(consumer, ["expanded", "w"], Tensor("y", (Dim(M), Dim(N)), F16), node_id="y")
    g.inputs, g.outputs = ["p", "w"], ["y"]
    return g


def _placed_product_reduction_graph(K: int = 32, N: int = 16, M: int | None = None) -> Graph:
    """A placement-produced product workspace already followed by a scheduled reduction."""
    g = Graph()
    lead = (M,) if M is not None else ()
    lead_index = (Var("m"),) if M is not None else ()
    _inp(g, "activation__cut_v", (*lead, K))
    _inp(g, "w", (K, N))
    _inp(g, "residual", (*lead, N))
    product_cell = Body(
        (
            Load(name="a", input="activation__cut_v", index=(*lead_index, Var("k"))),
            Load(name="b", input="w", index=(Var("k"), Var("n"))),
            Assign(name="p", args=("a", "b"), op="multiply"),
            Write(output="product", index=(*lead_index, Var("k"), Var("n")), value="p"),
        )
    )
    product_body = Body(
        (
            Loop(
                axis=Axis("k", Dim(K)),
                body=Body((Loop(axis=Axis("n", Dim(N)), body=product_cell),)),
            ),
        )
    )
    if M is not None:
        product_body = Body((Loop(axis=Axis("m", Dim(M)), body=product_body),))
    product = LoopOp(body=product_body)
    reduction = LoopOp(
        body=Body(
            (
                Loop(
                    axis=Axis("n", Dim(N)),
                    body=Body(
                        (
                            Loop(
                                axis=Axis("k", Dim(K)),
                                body=Body(
                                    (
                                        Load(
                                            name="p",
                                            input="product",
                                            index=(*lead_index, Var("k"), Var("n")),
                                        ),
                                        Accum(name="sum", value="p", op="add", axes=("k",)),
                                    )
                                ),
                            ),
                            Load(name="r", input="residual", index=(*lead_index, Var("n"))),
                            Assign(name="out", args=("sum", "r"), op="add"),
                            Write(output="y", index=(*lead_index, Var("n")), value="out"),
                        )
                    ),
                ),
            )
        )
    )
    if M is not None:
        reduction = LoopOp(body=Body((Loop(axis=Axis("m", Dim(M)), body=reduction.body),)))
    product_args = (
        product,
        ["activation__cut_v", "w"],
        Tensor("product", tuple(Dim(s) for s in (*lead, K, N)), dtype=F16),
        "product",
    )
    reduction_args = (
        reduction,
        ["product", "residual"],
        Tensor("y", tuple(Dim(s) for s in (*lead, N)), dtype=F16),
        "y",
    )
    for op, inputs, output, node_id in (product_args, reduction_args):
        g.add_node(op, inputs, output, node_id=node_id)
    g.inputs, g.outputs = ["activation__cut_v", "w", "residual"], ["y"]
    return g


# --- the loader split (routing vs schedule entries) ----------------------------------------------


def test_is_routing_reads_place_only_knob_dicts() -> None:
    base = GoldenRecord(
        name="test",
        gpu_name="TESTGPU",
        compute_cap=(12, 0),
        model=None,
        program_index=0,
        program_wire={"inputs": [], "outputs": [], "nodes": []},
        origins=(),
        bindings=(),
        pins=(("FAST_MATH", False),),
        knobs={},
        measurements={"emmy_us": 1.0, "reference_us": 1.0, "reference_backend": "test"},
        ranking=None,
    )
    routing = replace(base, knobs={"PLACE@a": "cut"})
    sched = replace(base, knobs={"TILE": "f4x8", "WORK": "t16x8"})
    assert routing.is_routing and not sched.is_routing
    assert not replace(base, knobs={}).is_routing


def _routing_record(*, gpu_name: str, emmy_us: float = 3.8):
    return SimpleNamespace(
        name="rms.routing",
        knobs={"PLACE": "cut"},
        is_routing=True,
        gpu_name=gpu_name,
        compute_cap=(12, 0),
        emmy_us=emmy_us,
        shape_key=ShapeKey(free_prod=64 * 4096, reduce_max=4096, is_warp=False, kind="rms_norm"),
    )


def test_routing_entries_never_join_the_schedule_golden_tier(monkeypatch) -> None:
    """A PLACE-only entry has no schedule keys, so ``_golden_matches_row`` would read every
    family as FREE and 'match' any row at the routing total — the schedule-tier index must skip
    routing entries entirely."""
    from emmy.compiler.pipeline.search import golden as golden_mod
    from emmy.compiler.pipeline.search.policy import greedy

    entry = _routing_record(gpu_name="NVIDIA GeForce RTX 5090", emmy_us=1.0)
    monkeypatch.setattr(golden_mod, "GOLDEN_RECORDS", [entry])
    ctx = Context.from_target((12, 0), gpu_name=entry.gpu_name)
    assert greedy._golden_evidence_index(ctx) == {}


# --- the PLACE path family ------------------------------------------------------------------------


def test_place_sites_are_the_non_root_nodes() -> None:
    from emmy.compiler.ir.tile.path import family_sites, resolve, sites, spell
    from emmy.compiler.pipeline.passes.lowering.tile._atomize import bind_prologue_contraction

    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from test_recognize_boundary_rules import _prologue_shape

    node, free = _prologue_shape(b_layouts=(False, False))
    c_map, _n_ax, _stores = bind_prologue_contraction(node, free)
    all_sites = sites(c_map)
    seams = family_sites("PLACE", all_sites)
    assert seams and all(s.depth > 1 for s in seams)
    # The cone edge spells through the view-role label — the plan's worked spelling.
    cone = c_map.operands[0].a  # noqa: F841 — the a edge carries its role label on the stored node
    labels = {spell(c_map, "PLACE", s.node, all_sites=all_sites) for s in seams}
    assert "PLACE@a" in labels
    assert resolve(c_map, "PLACE@a", all_sites=all_sites).node is next(
        s.node for s in seams if spell(c_map, "PLACE", s.node, all_sites=all_sites) == "PLACE@a"
    )


# --- the realizer: pin-driven cuts, fuse-default, recursion ---------------------------------------


def test_rms_norm_deploys_unchanged_under_default_fuse(monkeypatch) -> None:
    out = _compile(_rms_graph(), None, monkeypatch)
    assert len(_kernel_ids(out)) == 1, "no routing entry and no pin = fuse = the recognized form"


def test_rms_norm_place_cut_splits_stat_and_scale(monkeypatch) -> None:
    out = _compile(_rms_graph(), "PLACE=cut", monkeypatch)
    kernels = _kernel_ids(out)
    assert any("__cut_" in k for k in kernels), kernels
    assert len(kernels) >= 2  # the stat piece (possibly g-split into partial+finalize) + the scale
    ws = next(k for k in kernels if "__cut_" in k and "__partial" not in k)
    # The workspace producer feeds the residue, which writes the original output.
    consumer = out.nodes["y"]
    assert ws in consumer.inputs or any(ws in out.nodes[i].inputs for i in consumer.inputs)


def test_norm_linear_cone_cut_recurses_to_the_full_cascade(monkeypatch) -> None:
    """Bare ``PLACE=cut`` cuts the cone edge (the fold seam is the pure-copy degenerate and is
    not cuttable), the cone piece re-recognizes as the rms_norm shape and cuts again — the
    plan's worked cascade: statistic + scale + plain matmul, all from EXISTING kinds."""
    out = _compile(_norm_linear_graph(), "PLACE=cut", monkeypatch)
    kernels = _kernel_ids(out)
    cuts = [k for k in kernels if "__cut_" in k]
    assert len(cuts) >= 2, f"expected the recursive cascade, got {kernels}"
    assert "y" in kernels, "the residue matmul keeps the original output"


def test_materialized_only_atom_cuts_computed_a_without_routing_evidence(monkeypatch) -> None:
    """Any target whose only dtype-legal atom requires materialized edges cuts computed A."""
    out = _compile(_norm_linear_graph(S=1), None, monkeypatch, ctx=Context.from_target((7, 0)))
    kernels = _kernel_ids(out)
    assert any("__cut_" in k for k in kernels), kernels
    assert "y" in kernels, "the materialized residue keeps the original linear output"
    assert len(kernels) == 2, f"the target-legality cut must not recursively split RMSNorm: {kernels}"


def test_computed_capable_atom_preserves_fusion_without_routing_evidence(monkeypatch) -> None:
    """A target with any computed-edge-capable atom keeps the recognized fused form."""
    out = _compile(_norm_linear_graph(S=1), None, monkeypatch, ctx=Context.from_target((8, 0)))
    assert not any("__cut_" in k for k in _kernel_ids(out))


def test_explicit_fuse_overrides_required_computed_a_materialization(monkeypatch) -> None:
    """PLACE pins remain authoritative over the capability-derived placement default."""
    out = _compile(_norm_linear_graph(S=1), "PLACE=fuse", monkeypatch, ctx=Context.from_target((7, 0)))
    assert not any("__cut_" in k for k in _kernel_ids(out))


def test_rejoined_projection_reshape_recognizes_separable_contraction(monkeypatch) -> None:
    """A flatten→head-view→flatten boundary must not make weight N look M-dependent."""
    from emmy.compiler.ir.tile.ir import is_contraction
    from emmy.compiler.ir.tile.path import sites

    monkeypatch.delenv("EMMY_KNOBS", raising=False)
    monkeypatch.delenv("EMMY_PLACE", raising=False)
    out, _ = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=Context.from_target((7, 0))).resolve(
        _norm_linear_rejoined_reshape_graph(), lambda fp: flatten_leaves(fp.options)[0]
    )
    projection = out.nodes["y"].op
    contractions = [site.node for site in sites(projection.op) if is_contraction(site.node)]
    assert len(contractions) == 1
    weight = next(load for load in contractions[0].lower() for load in Body((load,)).loads if load.input == "w")
    assert weight.index == (Var("a2"), Var("a1"))


def test_nested_contraction_materializes_when_hardware_atom_cannot_compose_it(monkeypatch) -> None:
    """A raw two-reduction cell recovers independently schedulable contraction kernels."""
    out = _compile(_sequential_linears_graph(), None, monkeypatch, ctx=Context.from_target((7, 0)))
    kernels = _kernel_ids(out)
    assert any("__cut_" in kernel for kernel in kernels), kernels
    assert "y" in kernels
    assert len(kernels) == 2


def test_nested_contraction_stays_fused_without_a_hardware_atom(monkeypatch) -> None:
    """Materialization is not a platform policy when it cannot recover a hardware tier."""
    out = _compile(_sequential_linears_graph(), None, monkeypatch, ctx=Context.from_target((6, 0)))
    assert _kernel_ids(out) == ["y"]


def test_explicit_fuse_overrides_nested_contraction_materialization(monkeypatch) -> None:
    """The placement pin remains authoritative over the structural legality default."""
    out = _compile(_sequential_linears_graph(), "PLACE=fuse", monkeypatch, ctx=Context.from_target((7, 0)))
    assert _kernel_ids(out) == ["y"]


def test_nested_contraction_lifts_closed_enclosing_statistic_prologue(monkeypatch) -> None:
    """An enclosing RMS statistic closes the compact child instead of blocking placement."""
    out = _compile(_norm_gated_mlp_graph(), None, monkeypatch, ctx=Context.from_target((7, 0)))
    kernels = _kernel_ids(out)
    assert len(kernels) == 4, kernels
    cuts = [node for node in out.nodes.values() if "__cut_" in node.output.name]
    assert {tuple(dim.as_static() for dim in node.output.shape) for node in cuts} == {(16,), (32,)}
    activation = next(node for node in cuts if tuple(dim.as_static() for dim in node.output.shape) == (32,))
    assert activation.output.dtype is F16
    expanded = next(node for node in out.nodes.values() if tuple(dim.as_static() for dim in node.output.shape) == (1, 32, 16))
    assert expanded.inputs == [activation.output.name]


def test_nested_cut_retains_only_parent_live_prologue_slice() -> None:
    """A lifted child may share invariant definitions with later parent reductions."""
    from emmy.compiler.pipeline.passes.lowering.tile._cut import (
        _NestedReductionCut,
        _replace_nested_region,
    )

    source = Load(name="source", input="x", index=())
    shared = Assign(name="shared", op="add", args=("source", "source"))
    child_only = Load(name="child_only", input="w", index=())
    lifted = Assign(name="lifted", op="multiply", args=("shared", "child_only"))
    survivor = Assign(name="survivor", op="add", args=("shared", "shared"))
    container = Body((lifted, survivor))
    parent = Body((source, shared, child_only, Loop(axis=Axis("n", Dim(4)), body=container)))
    cut = _NestedReductionCut(
        container=container,
        prologue=(source, shared, child_only),
        members=(lifted,),
        result="lifted",
        axes=(),
    )

    rewritten = _replace_nested_region(parent, cut, Load(name="lifted", input="workspace", index=()))

    assert source in rewritten and shared in rewritten
    assert child_only not in rewritten
    residue = next(stmt for stmt in rewritten if isinstance(stmt, Loop)).body
    assert isinstance(residue[0], Load) and residue[0].input == "workspace"
    assert residue[1] is survivor


def test_materialized_multichannel_contraction_preserves_every_fold(monkeypatch) -> None:
    """Cutting shared computed A must bind both gate/up product-monoid channels."""
    from emmy.compiler.ir.tile.ir import TileOp, is_contraction
    from emmy.compiler.ir.tile.path import sites

    monkeypatch.delenv("EMMY_KNOBS", raising=False)
    monkeypatch.delenv("EMMY_PLACE", raising=False)
    out, _ = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=Context.from_target((7, 0))).resolve(
        _norm_gated_mlp_graph(M=2), lambda fp: flatten_leaves(fp.options)[0]
    )
    activation = next(
        node
        for node in out.nodes.values()
        if isinstance(node.op, TileOp) and tuple(dim.as_static() for dim in node.output.shape) == (2, 32)
    )
    contractions = [site.node for site in sites(activation.op.op) if is_contraction(site.node)]
    assert len(contractions) == 1
    contraction = contractions[0]
    assert len(contraction.channels) == 2
    assert tuple(activation.op.op.lift.params) == tuple(channel.acc for channel in contraction.channels)


@pytest.mark.parametrize("consumer_first", [False, True])
def test_placement_recomposes_product_with_its_additive_reduction(monkeypatch, consumer_first: bool) -> None:
    """A placement residue must not materialize a full K×N product used by one sum."""
    from emmy.compiler.pipeline.passes.lowering.tile._cut import _product_reduction_producer

    monkeypatch.delenv("EMMY_KNOBS", raising=False)
    monkeypatch.delenv("EMMY_PLACE", raising=False)
    graph = _placed_product_reduction_graph()
    producer = graph.nodes["product"]
    consumer = graph.nodes["y"]
    endpoints = (consumer, producer) if consumer_first else (producer, consumer)
    assert [_product_reduction_producer(graph, endpoint) for endpoint in endpoints] == [producer, producer]

    lowering = CUDA_PASSES[len(LOOP_PASSES) :]
    out, _ = Run(pipeline=Pipeline.build(lowering), ctx=Context.from_target((7, 0))).resolve(
        graph, lambda fp: flatten_leaves(fp.options)[0]
    )
    assert _kernel_ids(out) == ["y"]
    assert out.buffer("product") is None
    assert out.nodes["y"].inputs == ["residual", "w", "activation__cut_v"]


def test_placement_recomposition_recognizes_m512_multicell_contraction(monkeypatch) -> None:
    """The same structural repair recovers the contraction tier for a many-row output."""
    from emmy.compiler.ir.tile.ir import TileOp, is_contraction
    from emmy.compiler.ir.tile.path import sites

    monkeypatch.delenv("EMMY_KNOBS", raising=False)
    monkeypatch.delenv("EMMY_PLACE", raising=False)
    lowering = TILE_PASSES[len(LOOP_PASSES) :]
    out, _ = Run(pipeline=Pipeline.build(lowering), ctx=Context.from_target((7, 0))).resolve(
        _placed_product_reduction_graph(M=512), lambda fp: flatten_leaves(fp.options)[0]
    )
    assert out.buffer("product") is None
    tile = out.nodes["y"].op
    assert isinstance(tile, TileOp)
    assert any(is_contraction(site.node) for site in sites(tile.op))


def test_place_fuse_pin_preserves_product_reduction_boundary(monkeypatch) -> None:
    """The explicit placement override keeps even an inefficient diagnostic boundary."""
    monkeypatch.setenv("EMMY_PLACE", "fuse")
    lowering = CUDA_PASSES[len(LOOP_PASSES) :]
    out, _ = Run(pipeline=Pipeline.build(lowering), ctx=Context.from_target((7, 0))).resolve(
        _placed_product_reduction_graph(), lambda fp: flatten_leaves(fp.options)[0]
    )
    assert _kernel_ids(out) == ["product", "y"]
    assert out.buffer("product") is not None
    assert "product" in out.nodes["y"].inputs


def test_broadcast_expansion_materializes_the_compact_producer_domain(monkeypatch) -> None:
    """PLACE never allocates a store-sweep dimension the producer value does not read."""
    monkeypatch.delenv("EMMY_KNOBS", raising=False)
    monkeypatch.delenv("EMMY_PLACE", raising=False)
    lowering = CUDA_PASSES[len(LOOP_PASSES) :]
    out, _ = Run(pipeline=Pipeline.build(lowering), ctx=Context.from_target((12, 0))).resolve(
        _expanded_reduce_graph(), lambda fp: flatten_leaves(fp.options)[0]
    )
    compact = out.buffer("expanded__cut_compact")
    assert compact is not None
    assert sorted(dim.as_static() for dim in compact.shape) == [2, 8]
    assert all(tuple(dim.as_static() for dim in node.output.shape) != (2, 8, 16) for node in out.nodes.values())
    assert "expanded__cut_compact" in out.nodes["y"].inputs


def test_place_fuse_pin_preserves_expanded_boundary_for_diagnostics(monkeypatch) -> None:
    """The existing authoritative fuse pin remains an escape hatch for placement A/Bs."""
    monkeypatch.setenv("EMMY_PLACE", "fuse")
    lowering = CUDA_PASSES[len(LOOP_PASSES) :]
    out, _ = Run(pipeline=Pipeline.build(lowering), ctx=Context.from_target((12, 0))).resolve(
        _expanded_reduce_graph(), lambda fp: flatten_leaves(fp.options)[0]
    )
    assert out.buffer("expanded__cut_compact") is None
    expanded = out.buffer("expanded")
    assert expanded is not None
    assert tuple(dim.as_static() for dim in expanded.shape) == (2, 8, 16)


def test_scoped_place_pin_from_replay_context_cuts_the_cone(monkeypatch) -> None:
    """The replay context publishes ``PLACE@a`` through the aggregate consumed by routing."""
    from emmy.compiler.pipeline.search.pins import pinned_knobs

    monkeypatch.delenv("EMMY_KNOBS", raising=False)
    monkeypatch.delenv("EMMY_PLACE@A", raising=False)
    with pinned_knobs({"PLACE@a": "cut"}):
        out, _ = Run(pipeline=Pipeline.build(CUDA_PASSES), ctx=Context.from_target((12, 0))).resolve(
            _norm_linear_graph(S=1, H=16, inter=16), lambda fp: flatten_leaves(fp.options)[0]
        )
    kernels = _kernel_ids(out)
    assert any("__cut_" in k for k in kernels), kernels
    assert "y" in kernels, "the residue matmul keeps the original output"


def test_explicit_fuse_pin_suppresses_cutting(monkeypatch) -> None:
    out = _compile(_rms_graph(), "PLACE=fuse", monkeypatch)
    assert len(_kernel_ids(out)) == 1


def test_pin_naming_no_seam_is_skipped(monkeypatch) -> None:
    """A whole-model pin targets one kernel shape — trees without that seam compile fused."""
    out = _compile(_rms_graph(), "PLACE@b=cut", monkeypatch)
    assert len(_kernel_ids(out)) == 1


# --- routing entries drive the same realizer -------------------------------------------------------


def test_routing_golden_cuts_without_a_pin(monkeypatch) -> None:
    from emmy import config
    from emmy.compiler.pipeline.search import golden as golden_mod

    entry = _routing_record(gpu_name="NVIDIA GeForce RTX 5090")
    monkeypatch.setattr(golden_mod, "GOLDEN_RECORDS", [entry])
    # Goldens are -O3 truth: under make test's -Xcicc -O1 lane the routing consult (like the
    # schedule golden tier) is silent — force the deployable regime, the audit's own move.
    with config.nvcc_flags_override(""):
        ctx = Context.from_target((12, 0), gpu_name=entry.gpu_name)
        out = _compile(_rms_graph(), None, monkeypatch, ctx=ctx)
    assert any("__cut_" in k for k in _kernel_ids(out)), "the routing entry cuts with no pin present"


def test_stat_free_computed_a_routing_golden_uses_its_fused_key(monkeypatch) -> None:
    """A stat-free activation→linear record deploys through the canonical computed-A key."""
    from emmy import config
    from emmy.compiler.pipeline.search import golden as golden_mod

    gpu_name = "NVIDIA GeForce RTX 5090"
    entry = SimpleNamespace(
        name="activation-linear.routing",
        knobs={"PLACE@a": "cut"},
        is_routing=True,
        gpu_name=gpu_name,
        compute_cap=(12, 0),
        emmy_us=1.0,
        shape_key=replace(ShapeKey.from_matmul(2, 32, 16, "fp16"), kind="fused"),
    )
    monkeypatch.setattr(golden_mod, "GOLDEN_RECORDS", [entry])
    with config.nvcc_flags_override(""):
        out = _compile(_activation_linear_graph(), None, monkeypatch, ctx=Context.from_target((12, 0), gpu_name=gpu_name))
    kernels = _kernel_ids(out)
    assert any("__cut_" in k for k in kernels), kernels
    assert "y" in kernels, "the residue matmul keeps the original output"


def test_schedule_pin_suppresses_the_routing_entry(monkeypatch) -> None:
    """Any live schedule-family pin marks a pinned re-record / ``--ab`` compile, and pins are
    authoritative over every golden tier — the recorded routing entry must not reroute it. This
    is the 2026-07-31 fused re-record dead end: with a same-shape ``.cut`` row recorded, every
    pinned fused golden replay silently compiled the cut's pieces and gated ``realized (off)``."""
    from emmy import config
    from emmy.compiler.pipeline.search import golden as golden_mod

    entry = _routing_record(gpu_name="NVIDIA GeForce RTX 5090")
    monkeypatch.setattr(golden_mod, "GOLDEN_RECORDS", [entry])
    monkeypatch.setenv("EMMY_STAGE", "")  # the OFF spelling — any schedule-family pin, PLACE excluded
    with config.nvcc_flags_override(""):
        ctx = Context.from_target((12, 0), gpu_name=entry.gpu_name)
        out = _compile(_rms_graph(), None, monkeypatch, ctx=ctx)
    assert len(_kernel_ids(out)) == 1, "a live schedule pin is authoritative — the routing entry must not cut"


def test_routing_golden_ignored_off_its_card(monkeypatch) -> None:
    from emmy.compiler.pipeline.search import golden as golden_mod

    entry = _routing_record(gpu_name="NVIDIA GeForce RTX 4090")
    monkeypatch.setattr(golden_mod, "GOLDEN_RECORDS", [entry])
    ctx = Context.from_target((12, 0), gpu_name="NVIDIA GeForce RTX 5090")
    out = _compile(_rms_graph(), None, monkeypatch, ctx=ctx)
    assert len(_kernel_ids(out)) == 1
