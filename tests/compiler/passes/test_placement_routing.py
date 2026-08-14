"""Phase 4 — placement routing + the cut realizer.

Placement offers the maximal fused region plus every realizable single-seam cut before schedule
enumeration. ROUTING entries (PLACE-only golden knobs) and authoritative ``PLACE`` pins collapse
that fork; otherwise option-0 is fused and search may measure the fragments. Cut pieces re-recognize
and schedule as fresh roots, recursively. These tests run off-GPU through the full CUDA pass list,
asserting fork identity, kernel sets, and buffer wiring.
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
from emmy.compiler.ir.frontend.ir import LinearOp, MatmulOp, ReshapeOp, RmsNormOp, SdpaOp, TransposeOp
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


def _sdpa_projection_graph() -> Graph:
    """Symbolic causal attention followed by flatten-heads and output projection.

    The general non-flash attention cell contains two reductions and therefore keeps
    its trailing head-width expansion as raw Loop IR rather than an extracted Store.
    """
    B, H, D, N = 1, 4, 64, 96
    seq = Dim("seq_len", hint=64)
    g = Graph()
    for name in ("q", "k", "v"):
        g.add_node(InputOp(), [], Tensor(name, (B, H, seq, D), F16), node_id=name)
    g.add_node(InputOp(), [], Tensor("wo", (N, H * D), F16), node_id="wo")
    g.add_node(SdpaOp(is_causal=True), ["q", "k", "v"], Tensor("att", (B, H, seq, D), F16), node_id="att")
    g.add_node(TransposeOp(axes=(1, 2)), ["att"], Tensor("attt", (B, seq, H, D), F16), node_id="attt")
    g.add_node(ReshapeOp(shape=(B, seq, H * D)), ["attt"], Tensor("attr", (B, seq, H * D), F16), node_id="attr")
    g.add_node(LinearOp(has_bias=False), ["attr", "wo"], Tensor("o", (B, seq, N), F16), node_id="o")
    g.inputs, g.outputs = ["q", "k", "v", "wo"], ["o"]
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


def _norm_gate_up_graph(S: int = 2, H: int = 16, inter: int = 32) -> Graph:
    g = Graph()
    _inp(g, "x", (1, S, H))
    _inp(g, "wn", (H,))
    _inp(g, "wg", (H, inter))
    _inp(g, "wu", (H, inter))
    g.add_node(RmsNormOp(), ["x", "wn"], Tensor("xn", (1, Dim(S), Dim(H)), dtype=F16), node_id="xn")
    g.add_node(MatmulOp(), ["xn", "wg"], Tensor("gate", (1, Dim(S), Dim(inter)), dtype=F16), node_id="gate")
    g.add_node(MatmulOp(), ["xn", "wu"], Tensor("up", (1, Dim(S), Dim(inter)), dtype=F16), node_id="up")
    g.add_node(ElementwiseOp("silu"), ["gate"], Tensor("sg", (1, Dim(S), Dim(inter)), dtype=F16), node_id="sg")
    g.add_node(ElementwiseOp("multiply"), ["sg", "up"], Tensor("o", (1, Dim(S), Dim(inter)), dtype=F16), node_id="o")
    g.inputs, g.outputs = ["x", "wn", "wg", "wu"], ["o"]
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
    assert len(_kernel_ids(out)) == 1, "cold option-0 keeps the recognized form fused"


def _placement_rows(graph: Graph, ctx: Context, passes=TILE_PASSES) -> list[dict]:
    """Capture the first placement fork and keep its fused option."""
    from emmy.compiler.pipeline.knob import family_of

    rows: list[dict] = []

    def decide(fp):
        leaves = flatten_leaves(fp.options)
        offered = [
            {key: value for key, value in (getattr(leaf, "knobs", {}) or {}).items() if family_of(key) == "PLACE"} for leaf in leaves
        ]
        if not rows and offered and all(offered):
            rows.extend(offered)
        return leaves[0]

    Run(pipeline=Pipeline.build(passes), ctx=ctx).resolve(graph, decide)
    return rows


def test_unpinned_placement_offers_fused_and_every_legal_cut(monkeypatch) -> None:
    """Placement is a structural fork, not an edit to the maximal fused region."""
    monkeypatch.delenv("EMMY_KNOBS", raising=False)
    monkeypatch.delenv("EMMY_PLACE", raising=False)
    rows = _placement_rows(_norm_linear_graph(S=2, H=16, inter=32), Context.from_target((9, 0)))
    assert len(rows) >= 2
    assert all(value == "fuse" for value in rows[0].values())
    assert all(set(row) == set(rows[0]) for row in rows), "every option must spell the same seam keys"
    assert all(sum(value == "cut" for value in row.values()) == 1 for row in rows[1:])


def test_placement_space_is_capability_independent(monkeypatch) -> None:
    """Recognition and placement enumerate the same algebraic seams on every target."""
    monkeypatch.delenv("EMMY_KNOBS", raising=False)
    monkeypatch.delenv("EMMY_PLACE", raising=False)
    rows = [
        _placement_rows(_norm_linear_graph(S=2, H=16, inter=32), Context.from_target(target))
        for target in ((6, 0), (7, 0), (8, 0), (9, 0), (10, 0), (12, 0))
    ]
    assert rows and all(row == rows[0] for row in rows[1:])
    assert rows[0][0]["PLACE@a"] == "fuse"
    assert any(row["PLACE@a"] == "cut" for row in rows[0][1:])


def test_selecting_place_cut_schedules_each_resulting_kernel(monkeypatch) -> None:
    """A cut is followed by independent recognition and schedule enumeration for both pieces."""
    from emmy.compiler.pipeline.knob import SCHEDULE_FAMILIES, family_of

    monkeypatch.delenv("EMMY_KNOBS", raising=False)
    monkeypatch.delenv("EMMY_PLACE", raising=False)
    selected = False

    def decide(fp):
        nonlocal selected
        leaves = flatten_leaves(fp.options)
        if not selected:
            cut = next(
                (leaf for leaf in leaves if any(family_of(key) == "PLACE" and value == "cut" for key, value in leaf.knobs.items())),
                None,
            )
            if cut is not None:
                selected = True
                return cut
        return leaves[0]

    out, trace = Run(pipeline=Pipeline.build(CUDA_PASSES), ctx=Context.from_target((9, 0))).resolve(
        _norm_linear_graph(S=2, H=16, inter=32), decide
    )
    kernels = [node.op for node in out.nodes.values() if getattr(node.op, "kernel_source", None)]
    assert selected and len(kernels) >= 2
    assert any(decision.chosen_kind == "graph" for decision in trace)
    assert all(any(family_of(key) in SCHEDULE_FAMILIES for key in kernel.knobs) for kernel in kernels)
    assert all(not any(family_of(key) == "PLACE" for key in kernel.knobs) for kernel in kernels)
    assert all(any(family_of(key) == "PLACE" for key in kernel.decision_knobs) for kernel in kernels)
    assert all(all(family_of(key) == "PLACE" for key in kernel.decision_knobs) for kernel in kernels)


def test_shared_a_product_cut_retains_every_contraction_channel(monkeypatch) -> None:
    """A cut materializes the shared A edge; recursive recognition must preserve the whole product."""
    from emmy.compiler.ir.tile import Fold, TileOp
    from emmy.compiler.ir.tile.ir import is_contraction
    from emmy.compiler.ir.tile.ops import axis_names
    from emmy.compiler.pipeline.search.two_level import outer_pipeline

    monkeypatch.setenv("EMMY_KNOBS", "PLACE@a=cut")
    graph, _ = Run(pipeline=outer_pipeline(), ctx=Context.from_target((9, 0))).resolve(
        _norm_gate_up_graph(), lambda fp: flatten_leaves(fp.options)[0]
    )
    products = []
    for node in graph.nodes.values():
        if not isinstance(node.op, TileOp):
            continue
        root = node.op.op
        contraction = root.operands[0] if isinstance(root, Fold) and root.axis is None and root.operands else root
        if is_contraction(contraction) and len(contraction.channels) > 1:
            products.append((node.op, root, contraction))
    assert len(products) == 1
    tile, root, product = products[0]
    assert len(product.channels) == 2
    assert not (root.lift.free_names() - axis_names(root) - {axis.name for axis in tile.place.free})

    lowered = _compile(_norm_gate_up_graph(), "PLACE@a=cut", monkeypatch)
    assert all(not isinstance(node.op, TileOp) for node in lowered.nodes.values())


def test_nested_placement_delta_excludes_the_inherited_parent_choice() -> None:
    """Recursive placement replay compares both metadata channels on the recognized root."""
    from emmy.compiler.pipeline.pipeline import _option_decision

    fragment = Graph()
    fragment.add_node(
        InputOp(decision_knobs={"PLACE@a": "cut", "PLACE@b": "cut"}),
        [],
        Tensor("out", (), dtype=F16),
        node_id="out",
    )
    assert _option_decision(fragment, {"PLACE@a": "cut"}) == {"PLACE@b": "cut"}


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


def test_computed_a_cut_is_replayable_on_materialized_only_target(monkeypatch) -> None:
    """Capability constrains schedules, not whether the computed-A placement exists."""
    out = _compile(_norm_linear_graph(S=1), "PLACE@a=cut", monkeypatch, ctx=Context.from_target((7, 0)))
    kernels = _kernel_ids(out)
    assert any("__cut_" in k for k in kernels), kernels
    assert "y" in kernels, "the materialized residue keeps the original linear output"
    fused = _compile(_norm_linear_graph(S=1), "PLACE@a=fuse", monkeypatch, ctx=Context.from_target((7, 0)))
    assert not any("__cut_" in k for k in _kernel_ids(fused))


def test_rejoined_projection_reshape_recognizes_separable_contraction(monkeypatch) -> None:
    """A flatten→head-view→flatten boundary must not make weight N look M-dependent."""
    from emmy.compiler.ir.tile.ir import is_contraction
    from emmy.compiler.ir.tile.path import sites

    monkeypatch.setenv("EMMY_KNOBS", "PLACE@a=cut")
    monkeypatch.delenv("EMMY_PLACE", raising=False)
    out, _ = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=Context.from_target((7, 0))).resolve(
        _norm_linear_rejoined_reshape_graph(), lambda fp: flatten_leaves(fp.options)[0]
    )
    projection = out.nodes["y"].op
    contractions = [site.node for site in sites(projection.op) if is_contraction(site.node)]
    assert len(contractions) == 1
    weight = next(load for load in contractions[0].lower() for load in Body((load,)).loads if load.input == "w")
    assert weight.index == (Var("a2"), Var("a1"))


def test_nested_contraction_cut_is_replayable_and_target_independent(monkeypatch) -> None:
    """A raw nested contraction offers the same fused/cut rows on every target."""
    monkeypatch.delenv("EMMY_KNOBS", raising=False)
    monkeypatch.delenv("EMMY_PLACE", raising=False)
    rows = [_placement_rows(_sequential_linears_graph(), Context.from_target(target)) for target in ((6, 0), (7, 0), (9, 0))]
    assert rows[0] == rows[1] == rows[2]
    # The nested-reduction lift is a work repair, so it leads the cold option order; the raw
    # fused reading stays offered as the functional fallback and an explicit pin can select it.
    assert rows[0][0] == {"PLACE@nested": "cut"}
    assert rows[0][1] == {"PLACE@nested": "fuse"}

    out = _compile(_sequential_linears_graph(), "PLACE@nested=cut", monkeypatch, ctx=Context.from_target((7, 0)))
    kernels = _kernel_ids(out)
    assert any("__cut_" in kernel for kernel in kernels), kernels
    assert "y" in kernels
    assert len(kernels) == 2


def test_explicit_fuse_keeps_nested_contraction_maximal(monkeypatch) -> None:
    out = _compile(_sequential_linears_graph(), "PLACE@nested=fuse", monkeypatch, ctx=Context.from_target((7, 0)))
    assert _kernel_ids(out) == ["y"]


def test_nested_contraction_lifts_closed_enclosing_statistic_prologue(monkeypatch) -> None:
    """An enclosing RMS statistic closes the compact child instead of blocking placement."""
    out = _compile(_norm_gated_mlp_graph(), "PLACE@nested=cut", monkeypatch, ctx=Context.from_target((7, 0)))
    kernels = _kernel_ids(out)
    assert len(kernels) == 3, kernels
    cuts = [node for node in out.nodes.values() if "__cut_" in node.output.name]
    assert {tuple(dim.as_static() for dim in node.output.shape) for node in cuts} == {(32,)}
    activation = cuts[0]
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


def test_nested_cut_preserves_multichannel_activation_closure(monkeypatch) -> None:
    """The lifted gate/up activation is closed and retains its typed compact boundary."""
    from emmy.compiler.ir.tile.ir import TileOp
    from emmy.compiler.ir.tile.ops import axis_names

    monkeypatch.setenv("EMMY_KNOBS", "PLACE@nested=cut")
    out, _ = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=Context.from_target((7, 0))).resolve(
        _norm_gated_mlp_graph(M=2), lambda fp: flatten_leaves(fp.options)[0]
    )
    activation = next(
        node
        for node in out.nodes.values()
        if isinstance(node.op, TileOp) and tuple(dim.as_static() for dim in node.output.shape) == (2, 32)
    )
    root = activation.op.op
    assert activation.output.dtype is F16
    assert activation.op.decision_knobs == {"PLACE@nested": "cut"}
    assert not (root.lift.free_names() - axis_names(root) - {axis.name for axis in activation.op.place.free})


@pytest.mark.parametrize("consumer_first", [False, True])
def test_placement_recomposes_product_with_its_additive_reduction(monkeypatch, consumer_first: bool) -> None:
    """Product recomposition is maximal option 0 from either recognized endpoint."""
    from emmy.compiler.pipeline.passes.lowering.tile._cut import _product_reduction_producer

    monkeypatch.delenv("EMMY_KNOBS", raising=False)
    monkeypatch.delenv("EMMY_PLACE", raising=False)
    graph = _placed_product_reduction_graph()
    producer = graph.nodes["product"]
    consumer = graph.nodes["y"]
    endpoints = (consumer, producer) if consumer_first else (producer, consumer)
    assert [_product_reduction_producer(graph, endpoint) for endpoint in endpoints] == [producer, producer]
    lowering = TILE_PASSES[len(LOOP_PASSES) :]
    assert _placement_rows(graph.copy(), Context.from_target((7, 0)), passes=lowering) == [
        {"PLACE@product": "fuse"},
        {"PLACE@product": "cut"},
    ]

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


def test_product_recomposition_preserves_a_real_narrow_tensor_boundary() -> None:
    from emmy.compiler.pipeline.passes.lowering.tile._cut import _product_reduction_producer

    graph = _placed_product_reduction_graph()
    graph.rename_node("activation__cut_v", "activation")
    assert _product_reduction_producer(graph, graph.nodes["product"]) is None
    assert _product_reduction_producer(graph, graph.nodes["y"]) is None


def test_cold_greedy_uses_graph_first_maximal_recomposition(monkeypatch) -> None:
    """Option 0 is the structural default even when the maximal leaf is a Graph splice."""
    from emmy.compiler.pipeline.search.policy.greedy import greedy_decide
    from emmy.compiler.pipeline.search.two_level import outer_pipeline

    monkeypatch.delenv("EMMY_KNOBS", raising=False)
    monkeypatch.delenv("EMMY_PLACE", raising=False)
    out, _ = Run(pipeline=outer_pipeline(), ctx=Context.from_target((7, 0))).resolve(
        _placed_product_reduction_graph(M=512), greedy_decide(prior=object())
    )
    assert out.buffer("product") is None


def test_product_cut_pin_preserves_product_reduction_boundary(monkeypatch) -> None:
    """The explicit cut reading retains the already-materialized product pair."""
    monkeypatch.setenv("EMMY_KNOBS", "PLACE@product=cut")
    lowering = CUDA_PASSES[len(LOOP_PASSES) :]
    out, _ = Run(pipeline=Pipeline.build(lowering), ctx=Context.from_target((7, 0))).resolve(
        _placed_product_reduction_graph(), lambda fp: flatten_leaves(fp.options)[0]
    )
    assert _kernel_ids(out) == ["product", "y"]
    assert out.buffer("product") is not None
    assert "product" in out.nodes["y"].inputs


def test_product_fuse_pin_replays_recomposition(monkeypatch) -> None:
    monkeypatch.setenv("EMMY_KNOBS", "PLACE@product=fuse")
    lowering = CUDA_PASSES[len(LOOP_PASSES) :]
    out, _ = Run(pipeline=Pipeline.build(lowering), ctx=Context.from_target((7, 0))).resolve(
        _placed_product_reduction_graph(), lambda fp: flatten_leaves(fp.options)[0]
    )
    assert _kernel_ids(out) == ["y"]
    assert out.buffer("product") is None


def test_schedule_pin_keeps_product_residue_for_exact_rerecord(monkeypatch) -> None:
    """A schedule pin applies to the current kernel and suppresses structural placement."""
    monkeypatch.delenv("EMMY_KNOBS", raising=False)
    monkeypatch.setenv("EMMY_REDUCE", "")
    lowering = CUDA_PASSES[len(LOOP_PASSES) :]
    out, _ = Run(pipeline=Pipeline.build(lowering), ctx=Context.from_target((7, 0))).resolve(
        _placed_product_reduction_graph(), lambda fp: flatten_leaves(fp.options)[0]
    )
    assert _kernel_ids(out) == ["product", "y"]
    assert out.buffer("product") is not None


def test_broadcast_expansion_materializes_the_compact_producer_domain(monkeypatch) -> None:
    """Selecting compact placement never allocates the virtual broadcast dimension."""
    monkeypatch.delenv("EMMY_KNOBS", raising=False)
    monkeypatch.delenv("EMMY_PLACE", raising=False)
    lowering = CUDA_PASSES[len(LOOP_PASSES) :]

    def choose_compact(fp):
        leaves = flatten_leaves(fp.options)
        return next((leaf for leaf in leaves if leaf.knobs.get("PLACE@broadcast") == "cut"), leaves[0])

    out, _ = Run(pipeline=Pipeline.build(lowering), ctx=Context.from_target((12, 0))).resolve(_expanded_reduce_graph(), choose_compact)
    compact = out.buffer("expanded__cut_compact")
    assert compact is not None
    assert sorted(dim.as_static() for dim in compact.shape) == [2, 8]
    assert all(tuple(dim.as_static() for dim in node.output.shape) != (2, 8, 16) for node in out.nodes.values())
    assert "expanded__cut_compact" in out.nodes["y"].inputs


def test_broadcast_rows_are_replayable_and_default_to_maximal(monkeypatch) -> None:
    monkeypatch.delenv("EMMY_KNOBS", raising=False)
    monkeypatch.delenv("EMMY_PLACE", raising=False)
    lowering = TILE_PASSES[len(LOOP_PASSES) :]
    rows = _placement_rows(_expanded_reduce_graph(), Context.from_target((12, 0)), passes=lowering)
    assert rows[0] == {"PLACE": "fuse", "PLACE@broadcast": "fuse"}
    assert all(set(row) == set(rows[0]) for row in rows)
    assert all(sum(value == "cut" for value in row.values()) == 1 for row in rows[1:])
    assert any(row["PLACE@broadcast"] == "cut" for row in rows[1:])
    cuda_lowering = CUDA_PASSES[len(LOOP_PASSES) :]
    out, _ = Run(pipeline=Pipeline.build(cuda_lowering), ctx=Context.from_target((12, 0))).resolve(
        _expanded_reduce_graph(), lambda fp: flatten_leaves(fp.options)[0]
    )
    assert out.buffer("expanded__cut_compact") is None
    expanded = out.buffer("expanded")
    assert expanded is not None
    assert tuple(dim.as_static() for dim in expanded.shape) == (2, 8, 16)

    # The canonical ordinary seam may spell bare ``PLACE``.  It is still one
    # scoped row member when a specialized boundary is selected, not a command
    # that short-circuits the remaining keys during replay.
    monkeypatch.setenv("EMMY_KNOBS", "PLACE=fuse,PLACE@broadcast=cut")
    replayed, _ = Run(pipeline=Pipeline.build(cuda_lowering), ctx=Context.from_target((12, 0))).resolve(
        _expanded_reduce_graph(), lambda fp: flatten_leaves(fp.options)[0]
    )
    assert replayed.buffer("expanded__cut_compact") is not None
    assert replayed.buffer("expanded") is None


def test_raw_multireduction_broadcast_is_a_maximal_first_fork(monkeypatch) -> None:
    """The SDPA output expansion is visible to placement even when it remains raw Loop IR."""
    monkeypatch.delenv("EMMY_KNOBS", raising=False)
    monkeypatch.delenv("EMMY_PLACE", raising=False)
    offered: list[dict] = []

    def choose_maximal(fp):
        leaves = flatten_leaves(fp.options)
        rows = [{key: value for key, value in leaf.knobs.items() if key.startswith("PLACE")} for leaf in leaves]
        if any("PLACE@broadcast" in row for row in rows):
            offered.extend(rows)
        return leaves[0]

    out, _ = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=Context.from_target((12, 0))).resolve(_sdpa_projection_graph(), choose_maximal)
    # The nested-reduction lift (a work repair) leads the cold order; the maximal fused row and
    # the broadcast cut stay offered so evidence or a pin can select either.
    assert offered
    assert any(all(value == "fuse" for value in row.values()) for row in offered)
    assert any(row["PLACE@broadcast"] == "cut" for row in offered)
    assert out.buffer("o_a_unsq_bc__cut_compact") is None
    expanded = out.buffer("o_a_unsq_bc")
    assert expanded is not None and tuple(str(dim) for dim in expanded.shape) == ("1", "seq_len", "256", "96")


def test_raw_multireduction_broadcast_exact_replay_materializes_compact(monkeypatch) -> None:
    """The scoped cut pin reproduces the compact attention boundary and projection wiring."""
    monkeypatch.setenv("EMMY_KNOBS", "PLACE@broadcast=cut")
    monkeypatch.delenv("EMMY_PLACE", raising=False)
    out, _ = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=Context.from_target((12, 0))).resolve(
        _sdpa_projection_graph(), lambda fp: flatten_leaves(fp.options)[0]
    )
    compact = out.buffer("o_a_unsq_bc__cut_compact")
    assert compact is not None and tuple(str(dim) for dim in compact.shape) == ("seq_len", "256")
    assert out.buffer("o_a_unsq_bc") is None
    assert out.nodes["o"].inputs == ["o_a_unsq_bc__cut_compact", "wo"]
    assert out.nodes["o"].op.decision_knobs == {"PLACE@broadcast": "cut"}


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
