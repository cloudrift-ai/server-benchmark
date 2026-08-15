"""The VERIFIED golden tier — strict structural-identity decode.

A recorded golden joins a live fork by exact structural identity (the recognized term's algebra
digest + dtype fingerprint, derived record-side from the record's own persisted program through
the shared recognition core) and decodes by exact spelled-row equality. No classified shape, no
prefix matching: a record that matches the identity but equals no enumerated row is DRIFT and
decides nothing (fail-closed)."""

from __future__ import annotations

import logging

import pytest

from emmy import config
from emmy.compiler.context import Context
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import TILE_PASSES, Pipeline
from emmy.compiler.pipeline.knob import schedule_row_key, stamp_schedule_families
from emmy.compiler.pipeline.pipeline import Run
from emmy.compiler.pipeline.search import golden as golden_mod
from emmy.compiler.pipeline.search.golden import decode_record, kernel_identity, load_golden_records
from emmy.compiler.pipeline.search.golden_eval import enumerate_graph
from emmy.compiler.pipeline.search.policy.greedy import greedy_decide

_GPU = "NVIDIA GeForce RTX 5090"
_CAP = (12, 0)


def _row_key(knobs: dict):
    """Both sides of every comparison normalize through the ONE schedule-row identity — the same
    rule the tier's decode uses."""
    return schedule_row_key(knobs)


def _matmul_graph():
    from emmy.commands.trace import trace_inline_code

    return trace_inline_code("torch.matmul(torch.randn(64,128, dtype=torch.float16), torch.randn(128,64, dtype=torch.float16))")["graph"]


def _document(graph, knobs: dict, *, pins: dict | None = None, name: str = "probe.m64", us: float = 7.5) -> dict:
    from emmy.compiler.ir.base import InputOp
    from emmy.compiler.torch_wire import graph_to_wire

    # The target selector is the kernel's FULL frontend provenance — every non-input node fuses
    # into the one kernel for these fixtures (a partial selector would slice the program and name
    # a different kernel: the linear alone instead of the fused norm→linear).
    origins = [nid for nid, node in graph.nodes.items() if not isinstance(node.op, InputOp)]
    return {
        "gpu_name": _GPU,
        "compute_cap": list(_CAP),
        "model": "org/model",
        "programs": [graph_to_wire(graph)],
        "configs": [
            {
                "program": 0,
                "target": {"origins": origins},
                "realizations": [
                    {
                        "name": name,
                        "bindings": {},
                        "pins": {"FAST_MATH": False} if pins is None else pins,
                        "knobs": knobs,
                        "measurements": {"emmy_us": us, "reference_us": 9.0, "reference_backend": "cublas"},
                    }
                ],
            }
        ],
    }


def _ctx() -> Context:
    with config.nvcc_flags_override(""):  # the deployable -O3 regime the tier is gated on
        return Context.from_target(_CAP, gpu_name=_GPU)


def _recorded_row(graph) -> dict:
    rows = enumerate_graph(graph.copy(), _ctx())
    row = next(r for r in rows if str(r.get("WORK", "")).startswith("w") and r.get("STAGE") == "d2/smem-async")
    return stamp_schedule_families(row)


def _deployed_row(graph, prior=None) -> dict:
    out, _ = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=_ctx()).resolve(graph.copy(), greedy_decide(prior=prior))
    (tile,) = [n.op for n in out.nodes.values() if isinstance(n.op, TileOp)]
    return {k: v for k, v in tile.knobs.items() if not str(k).startswith(("S_", "H_"))}


def test_recorded_row_deploys_by_identity_without_a_prior(monkeypatch):
    """The tier needs no prior: a recorded non-option-0 row decides its fork on a prior-less
    resolve, joined by structural identity and decoded by exact row equality."""
    graph = _matmul_graph()
    knobs = _recorded_row(graph)
    records = load_golden_records(_document(graph, knobs))
    assert kernel_identity(records[0]) is not None
    assert decode_record(records[0]) is None
    monkeypatch.setattr(golden_mod, "records_for_card", lambda gpu, cap: [r for r in records if r.gpu_name == gpu])
    deployed = _deployed_row(graph)
    assert _row_key(deployed) == _row_key(knobs)


def test_drift_is_fail_closed(monkeypatch, caplog):
    """A record whose spelled row equals NO enumerated leaf decides nothing: a loud drift
    warning, then the ordinary fall-through (option-0 on a prior-less resolve)."""
    graph = _matmul_graph()
    knobs = _recorded_row(graph)
    drifted = {**knobs, "TILE": "mma_m16n8k16_f16_f32/f9x9"}  # a fragment this enumeration never offers
    records = load_golden_records(_document(graph, drifted))
    reason = decode_record(records[0])
    assert reason is not None and "no enumerated row equals" in reason
    monkeypatch.setattr(golden_mod, "records_for_card", lambda gpu, cap: [r for r in records if r.gpu_name == gpu])
    with caplog.at_level(logging.WARNING):
        deployed = _deployed_row(graph)
    assert _row_key(deployed) != _row_key(drifted)
    assert any("drift" in r.message for r in caplog.records)


def test_foreign_pin_regime_is_ignored(monkeypatch):
    """A record measured under FAST_MATH decides nothing on a standard deploy — pin scoping is
    exact, never a matching heuristic."""
    graph = _matmul_graph()
    knobs = _recorded_row(graph)
    records = load_golden_records(_document(graph, knobs, pins={"FAST_MATH": True}))
    monkeypatch.setattr(golden_mod, "records_for_card", lambda gpu, cap: [r for r in records if r.gpu_name == gpu])
    deployed = _deployed_row(graph)
    assert _row_key(deployed) != _row_key(knobs)


def test_ranking_regime_never_consults(monkeypatch):
    """A recorded µs is -O3 truth: under the -O1 ranking flags the tier is not consulted."""
    graph = _matmul_graph()
    knobs = _recorded_row(graph)
    records = load_golden_records(_document(graph, knobs))
    monkeypatch.setattr(golden_mod, "records_for_card", lambda gpu, cap: records)
    with config.nvcc_flags_override("-Xcicc -O1"):
        ctx = Context.from_target(_CAP, gpu_name=_GPU)
    out, _ = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=ctx).resolve(graph.copy(), greedy_decide(prior=None))
    (tile,) = [n.op for n in out.nodes.values() if isinstance(n.op, TileOp)]
    row = {k: v for k, v in tile.knobs.items() if not str(k).startswith(("S_", "H_"))}
    assert _row_key(row) != _row_key(knobs)


def _norm_linear_graph():
    """The weighted RMSNorm→linear fixture, built directly in frontend IR — the fused computed-A
    edge whose cone seam the placement fork can cut."""
    from emmy.compiler.dim import Dim
    from emmy.compiler.dtype import F16
    from emmy.compiler.graph import Graph, Tensor
    from emmy.compiler.ir.base import InputOp
    from emmy.compiler.ir.frontend.ir import LinearOp, RmsNormOp

    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (1, Dim(32), Dim(1024)), dtype=F16), node_id="x")
    g.add_node(InputOp(), [], Tensor("wn", (Dim(1024),), dtype=F16), node_id="wn")
    g.add_node(InputOp(), [], Tensor("w", (Dim(3072), Dim(1024)), dtype=F16), node_id="w")
    g.add_node(RmsNormOp(), ["x", "wn"], Tensor("xn", (1, Dim(32), Dim(1024)), dtype=F16), node_id="xn")
    g.add_node(LinearOp(), ["xn", "w"], Tensor("y", (1, Dim(32), Dim(3072)), dtype=F16), node_id="y")
    g.inputs, g.outputs = ["x", "wn", "w"], ["y"]
    return g


def _shallowest_seam_key(graph) -> str:
    """The spelled ``PLACE`` key of the recognized tree's shallowest legal cut seam — read through
    the same helpers placement uses, so the test records what the compiler can actually cut."""
    from emmy.compiler.ir.tile.path import sites, spell
    from emmy.compiler.pipeline.passes.lowering.tile._atomize import bind_prologue_contraction
    from emmy.compiler.pipeline.passes.lowering.tile._cut import cuttable_seams
    from emmy.compiler.pipeline.search.golden import _recognized_target, load_golden_records

    records = load_golden_records(_document(graph, {"WORK": ""}, name="seam.probe"))
    tile = _recognized_target(records[0])
    pro = bind_prologue_contraction(tile.op, tuple(tile.place.free))
    route_tree, route_free, route_stores = (
        (pro[0], (*tile.place.free, pro[1]), pro[2]) if pro is not None else (tile.op, tile.place.free, tile.stores)
    )
    seams = cuttable_seams(route_tree, route_stores, route_free)
    assert seams, "the fixture must have a cuttable seam"
    return spell(route_tree, "PLACE", seams[0].node, all_sites=sites(route_tree))


def test_routing_record_decides_the_placement_fork(monkeypatch):
    """A ROUTING record (PLACE-only knobs) picks the cut fragment whose parent piece stamps the
    record's seam — the recorded-cut deploy, restored on strict identity."""
    graph = _norm_linear_graph()
    seam_key = _shallowest_seam_key(graph)
    routed = _document(graph, {seam_key: "cut"}, name="probe.cut", us=16.0)
    records = load_golden_records(routed)
    assert records[0].is_routing
    assert decode_record(records[0]) is None, decode_record(records[0])
    monkeypatch.setattr(golden_mod, "records_for_card", lambda gpu, cap: [r for r in records if r.gpu_name == gpu])
    out, _ = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=_ctx()).resolve(graph.copy(), greedy_decide(prior=None))
    tiles = {nid: n.op for nid, n in out.nodes.items() if isinstance(n.op, TileOp)}
    assert len(tiles) == 2, f"the routing record must cut the kernel: {sorted(tiles)}"
    assert any((op.knobs or {}).get(seam_key) == "cut" for op in tiles.values())


def test_fused_record_holds_the_placement_fork(monkeypatch):
    """A SCHEDULE record for the fused identity keeps the fused side of the placement fork — a
    verified fused µs must not lose the kernel set to a prediction."""
    graph = _norm_linear_graph()
    rows = enumerate_graph(graph.copy(), _ctx())
    fused_row = stamp_schedule_families(next(r for r in rows if str(r.get("WORK", "")).startswith("w")))
    records = load_golden_records(_document(graph, fused_row, name="probe.fused"))
    assert decode_record(records[0]) is None, decode_record(records[0])
    monkeypatch.setattr(golden_mod, "records_for_card", lambda gpu, cap: [r for r in records if r.gpu_name == gpu])
    out, _ = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=_ctx()).resolve(graph.copy(), greedy_decide(prior=None))
    tiles = [n.op for n in out.nodes.values() if isinstance(n.op, TileOp)]
    assert len(tiles) == 1, "the fused record must keep one kernel"
    row = {k: v for k, v in tiles[0].knobs.items() if not str(k).startswith(("S_", "H_"))}
    assert _row_key(row) == _row_key(fused_row)


@pytest.fixture(autouse=True)
def _no_local_evidence(monkeypatch, tmp_path):
    """Point the online prior at nothing so the resolves above are golden-tier only."""
    monkeypatch.setenv("EMMY_ONLINE_FILE", str(tmp_path / "online.json"))
