"""Phase 4 — placement routing + the cut realizer.

ROUTING entries (PLACE-only golden knobs) and authoritative ``PLACE`` pins resolve a cut BEFORE
any schedule fork exists; the realizer splits the recognized tree at the seam into un-mapped
``LoopOp`` pieces that re-recognize as fresh roots (recursive — a piece's entry may itself cut).
Fuse is the default by ABSENCE: with no pin and no entry, recognition is byte-untouched (the
digest harness holds separately). These tests run off-GPU: the pieces compile through the full
CUDA pass list with deterministic option-0 resolution, so kernel SETS and buffer wiring are
asserted without a device (GPU accuracy is covered by the e2e smoke on the 5090 host).
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F16
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.frontend.ir import MatmulOp, ReshapeOp, RmsNormOp, SdpaOp, TransposeOp
from emmy.compiler.ir.tensor.ir import ElementwiseOp, ReduceOp
from emmy.compiler.pipeline import CUDA_PASSES, LOOP_PASSES, TILE_PASSES, Pipeline
from emmy.compiler.pipeline.fork import flatten_leaves
from emmy.compiler.pipeline.pipeline import Run
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


def _has_grouped_cut(loop) -> bool:
    from emmy.compiler.pipeline.passes.lowering.tile._classify import fused_view
    from emmy.compiler.pipeline.passes.lowering.tile._cut import cuttable_seams
    from emmy.compiler.pipeline.passes.lowering.tile._lift import recognized_tile

    tile = recognized_tile(loop, name=loop.name)
    view = fused_view(tile)
    if view is None:
        return False
    tree, free, stores = view[0], (*tile.place.free, *view[1]), view[2]
    return any(len(cut.members) == 2 for cut in cuttable_seams(tree, stores, free))


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


def _gqa_sdpa_flatten_graph() -> Graph:
    """A small grouped-query attention cell followed by its output-layout flatten."""
    batch, query_heads, kv_heads, seq, head_dim = 1, 6, 2, 8, 16
    g = Graph()
    _inp(g, "q", (batch, query_heads, seq, head_dim))
    _inp(g, "k", (batch, kv_heads, seq, head_dim))
    _inp(g, "v", (batch, kv_heads, seq, head_dim))
    g.add_node(
        SdpaOp(is_causal=True, scale=head_dim**-0.5),
        ["q", "k", "v"],
        Tensor("attention", (batch, query_heads, seq, head_dim), F16),
        node_id="attention",
    )
    g.add_node(
        TransposeOp(axes=(1, 2)),
        ["attention"],
        Tensor("attention_t", (batch, seq, query_heads, head_dim), F16),
        node_id="attention_t",
    )
    g.add_node(
        ReshapeOp(shape=(batch, seq, query_heads * head_dim)),
        ["attention_t"],
        Tensor("flat", (batch, seq, query_heads * head_dim), F16),
        node_id="flat",
    )
    g.inputs, g.outputs = ["q", "k", "v"], ["flat"]
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


# --- the PLACE path family ------------------------------------------------------------------------


def test_place_sites_are_the_non_root_nodes() -> None:
    from emmy.compiler.ir.loop import LoopOp
    from emmy.compiler.ir.tile.path import family_sites, resolve, sites, spell
    from emmy.compiler.pipeline import LOOP_PASSES
    from emmy.compiler.pipeline.passes.lowering.tile._classify import fused_view
    from emmy.compiler.pipeline.passes.lowering.tile._lift import recognized_tile

    # The fused computed-A reading, derived exactly as the pass derives it: lift the lowered
    # norm→linear kernel to its Fold tree, then take the fused view — the reference tree whose
    # seams a ``PLACE`` key spells.
    lowered = Pipeline.build(LOOP_PASSES).run(_norm_linear_graph())
    node = next(n for n in lowered.nodes.values() if isinstance(n.op, LoopOp))
    node.op.populate_io(lowered, node)
    tile = recognized_tile(node.op, name=node.op.name)
    pro = fused_view(tile)
    assert pro is not None, "the norm→linear tile must derive its fused computed-A view"
    c_map, _added_axes, _stores = pro
    all_sites = sites(c_map)
    seams = family_sites("PLACE", all_sites)
    assert seams and all(s.depth > 1 for s in seams)
    # With the empty projection elided, the A cone is the root contraction's shallowest seam, so
    # the shortest canonical spelling is bare. The explicit view-role spelling remains accepted for
    # scoped pins and recorded evidence.
    cone = c_map.a
    assert spell(c_map, "PLACE", cone, all_sites=all_sites) == "PLACE"
    assert resolve(c_map, "PLACE@a", all_sites=all_sites).node is cone


def _attention_cuts(root):
    from emmy.compiler.dim import Dim
    from emmy.compiler.ir.axis import Axis
    from emmy.compiler.pipeline.passes.lowering.tile._cut import cuttable_seams

    return cuttable_seams(root, free=(Axis("m", Dim(8)), Axis("n", Dim(16))))


def _rebuild_attention_root(root, cone):
    from emmy.compiler.ir.pure.fold import Channel, Fold

    pv = root.operands[0]
    channel = pv.channels[0]
    rebuilt = Fold.contraction(k_axis=pv.axis, a=cone, channels=(Channel(b=channel.b, acc=channel.acc),))
    return Fold.projection(operands=(rebuilt,))


@pytest.mark.parametrize("difference", ["input", "op"])
def test_grouped_cut_rejects_non_equivalent_computed_edges(difference) -> None:
    """Same geometry is insufficient: exact operations and external buffers remain semantic."""
    from dataclasses import replace

    from emmy.compiler.ir.pure.fold import Channel, Fold, operand_name
    from emmy.compiler.ir.stmt import Assign, Body
    from tests.compiler.passes.test_recognize_boundary_rules import _attention_cone_term

    root, cone = _attention_cone_term()
    score = cone.operands[1]
    channel = score.channels[0]
    if difference == "input":
        changed_b = replace(channel.b, input="other_k")
        changed = Fold.contraction(k_axis=score.axis, a=score.a, channels=(Channel(b=changed_b, acc=channel.acc),))
    else:
        a_name = operand_name(score.a)
        changed_a = Fold.projection(
            operands=(score.a,),
            body=Body((Assign(name=f"{a_name}__neg", op="negative", args=(a_name,)),)),
        )
        changed = Fold.contraction(k_axis=score.axis, a=changed_a, channels=(Channel(b=channel.b, acc=channel.acc),))
    changed_cone = replace(cone, operands=(cone.operands[0], changed))

    cuts = _attention_cuts(_rebuild_attention_root(root, changed_cone))
    assert not [cut for cut in cuts if len(cut.members) > 1]


def test_grouped_cut_rejects_more_than_two_equivalent_uses() -> None:
    """The bounded inverse is exact-two; an additional equal use leaves every seam independent."""
    from emmy.compiler.dim import Dim
    from emmy.compiler.ir.axis import Axis
    from emmy.compiler.ir.expr import Var
    from emmy.compiler.ir.pure.fold import Channel, Fold
    from emmy.compiler.ir.stmt import Assign, Body, Load
    from tests.compiler.passes.test_recognize_boundary_rules import _attention_cone_term

    root, cone = _attention_cone_term()
    score3 = Fold.contraction(
        k_axis=Axis("ddc", Dim(16)),
        a=Load(names=("s3__q",), input="q", index=(Var("m"), Var("ddc"))),
        channels=(Channel(b=Load(names=("s3__k",), input="k", index=(Var("kvb"), Var("ddc"))), acc="s3"),),
    )
    cone3 = Fold.projection(
        operands=(*cone.operands, score3),
        body=Body((*cone.body, Assign(name="pw3", op="add", args=(cone.out, "s3")))),
    )

    cuts = _attention_cuts(_rebuild_attention_root(root, cone3))
    assert sum(1 for cut in cuts for member in cut.members if isinstance(member.node, Fold) and member.node.axis is not None) >= 3
    assert not [cut for cut in cuts if len(cut.members) > 1]


def test_sdpa_offers_fused_and_one_shared_score_cut_without_overrides() -> None:
    """The fused cell and its two-kernel inverse are structural siblings.

    The inverse materializes one score producer. Its two contextual uses retain distinct
    column-axis coordinates while loading the same workspace.
    """
    from emmy.commands.trace import trace_inline_code
    from emmy.compiler.ir.loop import LoopOp
    from emmy.compiler.ir.tile import TileOp
    from emmy.compiler.pipeline import LOOP_PASSES

    graph = trace_inline_code(
        "F.scaled_dot_product_attention(torch.randn(1,2,8,128), torch.randn(1,2,8,128), torch.randn(1,2,8,128), is_causal=True)"
    )["graph"]
    fused = Pipeline.build(LOOP_PASSES).run(graph)
    loop_nodes = [node for node in fused.nodes.values() if isinstance(node.op, LoopOp)]
    assert len(loop_nodes) == 1, "maximal fusion must produce one Loop-IR cell"

    captured: list = []

    def decide(fp):
        leaves = flatten_leaves(fp.options)
        captured.extend(leaves)
        return leaves[0]

    Run(pipeline=Pipeline.build(["lowering/tile"], select=["recognize"]), ctx=Context.from_target((8, 0))).resolve(fused, decide)
    assert any(isinstance(option, TileOp) for option in captured), "the fused TileOp sibling is missing"
    grouped = [option for option in captured if isinstance(option, Graph) and any("__cut_acc" in node_id for node_id in option.nodes)]
    assert len(grouped) == 1, "the two equivalent score uses must collapse to one materialized sibling"

    (split,) = grouped
    ws = next(node_id for node_id in split.nodes if "__cut_acc" in node_id)
    parent = split.nodes["scaled_dot_product_attention"].op
    loads = [load for load in parent.body.loads if load.input == ws]
    assert len(loads) == 2, "both reconstructed uses must read the one workspace"
    assert loads[0].index != loads[1].index, "each use must keep its contextual free-axis mapping"


@pytest.mark.parametrize(
    "source",
    [
        (
            "torch.manual_seed(0);"
            "q=torch.randn(1,2,1,128,dtype=torch.float16);"
            "k=torch.randn(1,2,8,128,dtype=torch.float16);"
            "v=torch.randn(1,2,8,128,dtype=torch.float16);"
            "F.scaled_dot_product_attention(q,k,v,is_causal=False)"
        ),
        (
            "torch.manual_seed(0);"
            "q=torch.randn(1,4,1,128,dtype=torch.float16);"
            "k=torch.randn(1,2,8,128,dtype=torch.float16);"
            "v=torch.randn(1,2,8,128,dtype=torch.float16);"
            "F.scaled_dot_product_attention("
            "q.reshape(1,2,2,1,128),k.reshape(1,2,1,8,128),v.reshape(1,2,1,8,128),"
            "is_causal=False).reshape(1,4,1,128)"
        ),
    ],
    ids=["mha", "gqa"],
)
def test_decode_sdpa_offers_grouped_fused_target(source) -> None:
    """A unit-elided query row keeps the same fused/grouped placement alternatives."""
    from emmy.commands.trace import trace_inline_code
    from emmy.compiler.ir.loop import LoopOp
    from emmy.compiler.ir.tile import TileOp
    from emmy.compiler.pipeline import LOOP_PASSES
    from emmy.compiler.pipeline.passes.lowering.tile._classify import fused_view
    from emmy.compiler.pipeline.passes.lowering.tile._lift import recognized_tile

    graph = trace_inline_code(source)["graph"]
    fused = Pipeline.build(LOOP_PASSES).run(graph)
    (node,) = [node for node in fused.nodes.values() if isinstance(node.op, LoopOp)]
    node.op.populate_io(fused, node)
    view = fused_view(recognized_tile(node.op, name=node.id))
    assert view is not None
    assert [(axis.name, axis.extent) for axis in view[1][:-1]] == [("_um", Dim(1))]

    captured: list = []

    def decide(fork):
        leaves = flatten_leaves(fork.options)
        captured.extend(leaves)
        return leaves[0]

    Run(pipeline=Pipeline.build(["lowering/tile"], select=["recognize"]), ctx=Context.from_target((8, 0))).resolve(fused, decide)
    assert any(isinstance(option, TileOp) for option in captured)
    grouped = [option for option in captured if isinstance(option, Graph) and any("__cut_acc" in node_id for node_id in option.nodes)]
    assert len(grouped) == 1


@pytest.mark.parametrize("index", [(Var("m"), Var("n")), (Var("n"),)], ids=["nonunit", "missing-row"])
def test_synthetic_decode_row_requires_a_literal_unit_coordinate(index) -> None:
    """A real or absent row coordinate cannot opt into the M=1 contraction reading."""
    from emmy.compiler.ir.stmt import Write
    from emmy.compiler.pipeline.passes.lowering.tile._classify import _unit_output_row

    write = Write(output="out", index=index, value="acc")
    assert _unit_output_row((write,), "n") is None


def test_maximal_sdpa_with_fused_producer_recovers_grouped_inverse() -> None:
    """Placement must recover the attention cut after an upstream producer joins the cell."""
    from emmy.commands.trace import trace_inline_code
    from emmy.compiler.ir.loop import LoopOp
    from emmy.compiler.pipeline import LOOP_PASSES

    graph = trace_inline_code(
        "F.scaled_dot_product_attention(torch.randn(1,2,8,128), torch.randn(1,2,8,128), torch.randn(1,2,8,128), is_causal=True)"
    )["graph"]
    graph = Pipeline.build(LOOP_PASSES).run(graph, ctx=Context.from_target((8, 0)))
    attention = graph.nodes["scaled_dot_product_attention"]
    assert isinstance(attention.op, LoopOp) and _has_grouped_cut(attention.op)

    source = graph.nodes["x0"].output
    graph.rename_node("x0", "q_src")
    graph.add_node(ElementwiseOp("negative"), ["q_src"], replace(source, name="x0"), node_id="x0")
    graph.replace_input(attention.id, "q_src", "x0")

    fused = Pipeline.build(["loop/lifting", "loop/fusion"]).run(graph, ctx=Context.from_target((8, 0)))
    loops = [node for node in fused.nodes.values() if isinstance(node.op, LoopOp)]
    assert len(loops) == 1, "maximal fusion must consume the upstream producer"
    assert _has_grouped_cut(loops[0].op), "placement must recover the grouped attention cut"
    attention = fused.nodes["scaled_dot_product_attention"]
    assert _has_grouped_cut(attention.op)


def test_output_flatten_preserves_the_gqa_sdpa_grouped_inverse() -> None:
    """Pure output layout must preserve attention's grouped placement inverse."""
    from emmy.compiler.ir.loop import LoopOp

    lowered = Pipeline.build(LOOP_PASSES).run(_gqa_sdpa_flatten_graph(), ctx=Context.from_target((8, 9)))
    loops = [node for node in lowered.nodes.values() if isinstance(node.op, LoopOp)]
    assert {node.id for node in loops} == {"flat"}
    attention = lowered.nodes["flat"]
    assert _has_grouped_cut(attention.op)


def test_output_flatten_keeps_an_ordinary_reduce_on_the_normal_fusion_path() -> None:
    """An equal-size layout after a plain fold does not mint grouped placement evidence."""
    from emmy.compiler.ir.loop import LoopOp

    g = Graph()
    _inp(g, "x", (2, 3, 8))
    g.add_node(ReduceOp(op="sum", axis=-1), ["x"], Tensor("sum", (2, 3, 1), F16), node_id="sum")
    g.add_node(ReshapeOp(shape=(2, 3)), ["sum"], Tensor("flat", (2, 3), F16), node_id="flat")
    g.inputs, g.outputs = ["x"], ["flat"]

    lowered = Pipeline.build(LOOP_PASSES).run(g, ctx=Context.from_target((8, 9)))
    loops = [node for node in lowered.nodes.values() if isinstance(node.op, LoopOp)]
    assert len(loops) == 1
    assert loops[0].id == "flat"
    assert not _has_grouped_cut(loops[0].op)


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


def test_cut_preserves_every_multi_output_parent_port(monkeypatch) -> None:
    graph = _norm_linear_graph(S=1, H=16, inter=16)
    graph.outputs = ["y", "xn"]

    out = _compile(graph, "PLACE=cut", monkeypatch)

    assert out.outputs == ["y", "xn"]
    assert all(out.buffer(buf) is not None for buf in out.outputs)
    assert any("__cut_" in kernel for kernel in _kernel_ids(out))


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


def test_a_cut_taken_at_a_fork_mid_batch_still_reaches_the_stamp(monkeypatch) -> None:
    """An UNPINNED cut is a fork option, and a fork's option is applied by the CALLER — which
    advances the rule cursor only when the applied match closed its batch. With a second kernel
    still pending, the cursor stays on ``010_recognize``, so the pieces it just minted are
    re-matched on the very next step, BEFORE the scan wraps back to ``005_stamp_structural_features``.
    Recognizing them there lifts kernels with no ``S_*`` and ``020_schedule`` asserts. Two chains,
    the first one cut, is the smallest graph that puts a cut match ahead of another kernel's."""
    monkeypatch.delenv("EMMY_KNOBS", raising=False)
    monkeypatch.delenv("EMMY_PLACE", raising=False)
    g = Graph()
    for tag in ("a", "b"):
        _inp(g, f"x{tag}", (1, 2, 16))
        _inp(g, f"wn{tag}", (16,))
        _inp(g, f"w{tag}", (16, 16))
        g.add_node(RmsNormOp(), [f"x{tag}", f"wn{tag}"], Tensor(f"xn{tag}", (Dim(1), Dim(2), Dim(16)), dtype=F16), node_id=f"xn{tag}")
        g.add_node(MatmulOp(), [f"xn{tag}", f"w{tag}"], Tensor(f"y{tag}", (Dim(1), Dim(2), Dim(16)), dtype=F16), node_id=f"y{tag}")
    g.inputs = [f"{n}{tag}" for tag in ("a", "b") for n in ("x", "wn", "w")]
    g.outputs = ["ya", "yb"]

    taken: list[str] = []

    def cut_once(fp):
        leaves = flatten_leaves(fp.options)
        cut = next((o for o in leaves if isinstance(o, Graph)), None) if not taken else None
        if cut is None:
            return leaves[0]
        taken.append("cut")
        return cut

    out, _ = Run(pipeline=Pipeline.build(CUDA_PASSES), ctx=Context.from_target((12, 0))).resolve(g, cut_once)
    kernels = _kernel_ids(out)
    assert taken, "no placement fork offered a cut — the graph no longer exercises the ordering"
    assert any("__cut_" in k for k in kernels), kernels
    assert {"ya", "yb"} <= set(kernels), kernels


# --- routing entries drive the same realizer -------------------------------------------------------


def test_contraction_operand_seam_takes_the_output_dtype(monkeypatch) -> None:
    """A seam standing in for a contraction OPERAND holds what the fused slab stored — the atom's
    16-bit element — not the f32 its cone computed in: typed f32 (an f32-computing norm over f16
    keys), the materialized B could feed no warp atom (only ``a`` has a converting fill)."""
    from emmy.compiler.ir.loop import LoopOp
    from emmy.compiler.ir.tile import TileOp
    from emmy.compiler.pipeline.search.pins import pinned_knobs
    from tests.compiler.passes.test_recognize_boundary_rules import _normed_sdpa_graph

    g = _normed_sdpa_graph()
    monkeypatch.delenv("EMMY_KNOBS", raising=False)
    with pinned_knobs({"PLACE@b": "cut"}):
        out, _ = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=Context.from_target((12, 0))).resolve(
            g, lambda fp: flatten_leaves(fp.options)[0]
        )
    ws = [n for n in out.nodes.values() if "__cut_" in n.id and isinstance(n.op, (LoopOp, TileOp))]
    assert ws, [n.id for n in out.nodes.values()]
    assert any(n.output.dtype == F16 for n in ws), [(n.id, str(n.output.dtype)) for n in ws]


def test_pinned_transposed_coop_band_refuses_without_a_free_axis_to_sweep(monkeypatch) -> None:
    """A REDUCE pin meets the transposed band's legality as a refusal, never a crash: at one row
    the rms statistic has no innermost free axis for the band's 32 lanes to sweep (unpinned,
    the catalog simply omits the band)."""
    monkeypatch.setenv("EMMY_WORK", "t256")
    monkeypatch.setenv("EMMY_REDUCE", "coop-t")
    with pytest.raises(ValueError, match="innermost free axis"):
        _compile(_rms_graph(S=1), None, monkeypatch)
