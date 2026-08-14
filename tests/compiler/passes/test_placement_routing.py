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

from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F16
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import MatmulOp, RmsNormOp
from emmy.compiler.ir.tensor.ir import ElementwiseOp
from emmy.compiler.pipeline import CUDA_PASSES, TILE_PASSES, Pipeline
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


def _placement_rows(graph: Graph, ctx: Context) -> list[dict]:
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

    Run(pipeline=Pipeline.build(TILE_PASSES), ctx=ctx).resolve(graph, decide)
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
        for target in ((8, 0), (9, 0), (10, 0), (12, 0))
    ]
    assert rows and all(row == rows[0] for row in rows[1:])


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
