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

from emmy.compiler.context import Context
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F16
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import MatmulOp, RmsNormOp
from emmy.compiler.ir.tensor.ir import ElementwiseOp
from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
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
    c_map, _n_ax, _stores = pro
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
