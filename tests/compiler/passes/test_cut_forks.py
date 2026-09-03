"""Kernel-placement forks over closed stored Fold edges."""

from __future__ import annotations

from dataclasses import replace
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

from emmy.compiler.backend.cuda.backend import CudaBackend
from emmy.compiler.context import Context
from emmy.compiler.dtype import F16
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis, Window
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.frontend.ir import SdpaOp, SoftmaxOp
from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.stmt import Assign, Load, Write
from emmy.compiler.ir.tile import OutputSpec, Placement, TileOp
from emmy.compiler.loop_wire import loop_graph_to_wire
from emmy.compiler.pipeline import CUDA_PASSES, LOOP_PASSES, TILE_PASSES, Match, Pipeline, Rule
from emmy.compiler.pipeline.fork import Fork
from emmy.compiler.pipeline.passes.lowering.tile._cut import (
    CutSite,
    _producer_order,
    _workspace_axes,
    cuttable_seams,
    output_map,
    realize,
)
from emmy.compiler.pipeline.pipeline import RuleSkipped, Run, _is_structural_option
from emmy.compiler.pipeline.search.golden import (
    GoldenRecord,
    _lifted_target,
    _replay,
    decode_record,
    kernel_identity,
    load_golden_file,
    load_golden_records,
    validate_golden_file,
)
from emmy.compiler.pipeline.search.pins import pinned_knobs
from emmy.compiler.torch_wire import graph_to_wire
from tests.compiler.helpers import direct_classic_leaf, requires_cuda
from tests.compiler.terms import contraction, projection

_CTX = Context.from_target((12, 0))
_CUT = import_module("emmy.compiler.pipeline.passes.lowering.tile.030_cut")


def _input(graph: Graph, name: str, shape, dtype="f16") -> None:
    graph.add_node(InputOp(), [], Tensor(name, shape, dtype), node_id=name)


def test_cut_and_assignment_passes_share_the_generic_schedule_driver() -> None:
    from emmy.compiler.ir.schedule import schedule

    assert _CUT.schedule is schedule
    assert import_module("emmy.compiler.pipeline.fork").schedule is schedule


def test_placement_cut_preserves_a_cross_cta_split_receipt() -> None:
    """A split piece can re-enter placement; cutting it must not make REDUCE pending again."""
    from emmy.compiler.pipeline.passes.lowering.tile._split import split_pending

    graph = _computed_operand_graph("a")
    tile = graph.nodes["out"].op
    # The partition receipt is the reduce axis's window in the kernel's axis table; the term names it only.
    axes = tuple(replace(axis, window=Window(parent=axis, partition=True)) if axis.name == tile.op.axis else axis for axis in tile.axes)
    graph.nodes["out"].op = replace(tile, axes=axes)
    pipeline = Pipeline.build(["lowering/tile"], select={"cut"})
    match = pipeline.match(graph, pipeline.passes[0].rules[0])[0]
    seams = cuttable_seams(match.root.op)
    renamed = output_map(match.root)

    fragment = realize(match, match.root, (seams[0],), renamed)

    pieces = [node.op for node in fragment.nodes.values() if isinstance(node.op, TileOp)]
    assert pieces and all(piece.split_consumed for piece in pieces)
    assert not any(split_pending(piece) for piece in pieces)


def _computed_operand_graph(side: str) -> Graph:
    m, n, k = Axis("m", 8), Axis("n", 8), Axis("k", 16)
    computed = projection(
        (),
        (
            Load(name="raw", input="computed", index=(Var("m" if side == "a" else "n"), Var("k"))),
            Assign(name="scaled", op="multiply", args=("raw", "raw")),
        ),
    )
    direct = Load(
        name="direct",
        input="direct",
        index=(Var("k"), Var("n")) if side == "a" else (Var("m"), Var("k")),
    )
    a, b = (computed, direct) if side == "a" else (direct, computed)
    tile = TileOp(op=contraction(k, a, (b, "acc")), name="out", place=Placement(free=(m, n)), axes=(m, n, k))
    graph = Graph()
    _input(graph, "computed", (8, 16))
    _input(graph, "direct", (16, 8) if side == "a" else (8, 16))
    graph.add_node(tile, ["computed", "direct"], Tensor("out", (8, 8), "f16"), node_id="out")
    graph.inputs, graph.outputs = ["computed", "direct"], ["out"]
    return graph


def _mimo_graph() -> Graph:
    m, n, k = Axis("m", 8), Axis("n", 8), Axis("k", 16)

    def matmul(a: str, b: str, acc: str) -> Fold:
        return contraction(
            k,
            Load(name=f"{a}_v", input=a, index=(Var("m"), Var("k"))),
            (Load(name=f"{b}_v", input=b, index=(Var("k"), Var("n"))), acc),
        )

    first, second = matmul("a", "b", "first"), matmul("c", "d", "second")
    tile = TileOp(
        op=projection((first, second), results=("first", "second")),
        name="out0",
        place=Placement(free=(m, n)),
        axes=(m, n, k),
        output_specs=(
            OutputSpec(Write(output="out0", index=(Var("m"), Var("n")), value="first")),
            OutputSpec(Write(output="out1", index=(Var("m"), Var("n")), value="second")),
        ),
    )
    graph = Graph()
    for name in ("a", "c"):
        _input(graph, name, (8, 16))
    for name in ("b", "d"):
        _input(graph, name, (16, 8))
    graph.add_node(
        tile,
        ["a", "b", "c", "d"],
        outputs=(Tensor("out0", (8, 8), "f16"), Tensor("out1", (8, 8), "f16")),
        node_id="out0",
    )
    graph.inputs, graph.outputs = ["a", "b", "c", "d"], ["out0", "out1"]
    return graph


def _sdpa_graph(causal: bool) -> Graph:
    graph = Graph()
    for name in ("q", "k", "v"):
        _input(graph, name, (1, 2, 8, 16))
    graph.add_node(SdpaOp(is_causal=causal), ["q", "k", "v"], Tensor("out", (1, 2, 8, 16), "f16"), node_id="out")
    graph.inputs, graph.outputs = ["q", "k", "v"], ["out"]
    return graph


def _softmax_graph() -> Graph:
    graph = Graph()
    _input(graph, "x", (4, 32))
    graph.add_node(SoftmaxOp(axis=-1), ["x"], Tensor("out", (4, 32), "f16"), node_id="out")
    graph.inputs, graph.outputs = ["x"], ["out"]
    return graph


def _offered(graph: Graph, *, frontend: bool = False) -> list[dict]:
    offered: list[dict] = []
    passes = TILE_PASSES if frontend else ["lowering/tile"]
    select = None if frontend else {"cut"}

    def decide(fork):
        place = [option for option in fork.options if any(str(key).startswith("PLACE") for key in option.knobs)]
        if place:
            offered.extend(dict(option.knobs) for option in place)
            return next(option for option in place if not _is_structural_option(option))
        option = fork.options[0]
        while isinstance(option, Fork) and not option.is_leaf:
            option = option.expand()[0]
        return option

    Run(Pipeline.build(passes, select=select), _CTX).resolve(graph, decide)
    return offered


def _lower(graph: Graph, placement: dict[str, str]) -> Graph:
    with pinned_knobs(placement):
        lowered, _ = Run(Pipeline.build(CUDA_PASSES), _CTX).resolve(graph, direct_classic_leaf)
    lowered.validate()
    return lowered


def _lower_cut(graph: Graph, spelling: str) -> Graph:
    return _lower(graph, {spelling: "cut"})


def _nested_attention_cut(pins: dict[str, str]) -> Graph:
    case = Path(__file__).parents[1] / "realization/cases/attention/rmsnorm-gqa-b-cut.yaml"
    (record,) = load_golden_records(load_golden_file(case))
    tile = _lifted_target(record)
    graph = Graph()
    for name, tensor in tile.inputs.items():
        graph.add_node(InputOp(), [], tensor, node_id=name)
    graph.add_node(tile, list(tile.inputs), next(iter(tile.outputs.values())), node_id=tile.name)
    match = Match(graph=graph, root_node_id=tile.name, rule=Rule(name="test", pattern=[]))
    with pinned_knobs(pins):
        result = _CUT.rewrite(match, graph.nodes[tile.name])
    options = result if isinstance(result, list) else [result]
    cut = next(option for option in options if "cut" in option.knobs.values())
    return cut.expand()[0]


def _piece_with_seam(fragment: Graph):
    return next(node for node in fragment.nodes.values() if isinstance(node.op, TileOp) and cuttable_seams(node.op))


def test_cut_workspace_retains_static_unit_axes() -> None:
    """A unit seam axis remains workspace geometry even when the produced value is invariant in it."""
    unit, unused, column = Axis("batch", 1), Axis("unused", 8), Axis("n", 64)
    produced = projection((), (Load(name="value", input="x", index=(Var("n"),)),), results=("value",))
    seam = CutSite(
        node=produced,
        spelling="PLACE",
        axes=(unit, unused, column),
        dtypes=(F16,),
    )

    assert _workspace_axes(seam, produced) == (unit, column)


def test_composed_cut_topologically_orders_equal_degree_workspace_chain() -> None:
    """Counting direct workspace reads cannot order A->C->B when A and C each read one."""

    def piece(name: str, source: str | None):
        produced = projection((), (Load(name=f"{name}_value", input=source or "input", index=()),), results=(f"{name}_value",))
        return (None, produced, (), (), name, (f"{name}_value",), (name,))

    pieces = [piece("a", "c"), piece("c", "b"), piece("b", None)]

    assert [buffers[0] for *_, buffers in _producer_order(pieces)] == ["b", "c", "a"]


def test_pinned_fusion_lowers_one_computed_operand_kernel() -> None:
    lowered = _lower(_computed_operand_graph("a"), {"PLACE": "fuse"})
    assert sum(type(node.op).__name__ == "CudaOp" for node in lowered.nodes.values()) == 1


@pytest.mark.parametrize("side", ("a", "b"))
def test_computed_operand_offers_fused_and_cut_and_pinned_cut_lowers(side: str) -> None:
    offered = _offered(_computed_operand_graph(side))
    assert {frozenset(row.items()) for row in offered} == {
        frozenset({("PLACE", "fuse")}),
        frozenset({("PLACE", "cut")}),
    }
    lowered = _lower_cut(_computed_operand_graph(side), "PLACE")
    cuda = [node for node in lowered.nodes.values() if type(node.op).__name__ == "CudaOp"]
    assert len(cuda) == 2
    assert len(cuda[1].inputs) == 2 and any("__place_" in name for name in cuda[1].inputs)


@pytest.mark.parametrize(
    "causal",
    (
        False,
        pytest.param(
            True, marks=pytest.mark.xfail(strict=True, reason="fused value channel on tensor cores: not on this tree yet (PR #699)")
        ),
    ),
)
def test_sdpa_score_cut_is_offered_and_pinned_cut_lowers(causal: bool) -> None:
    offered = _offered(_sdpa_graph(causal), frontend=True)
    assert {"PLACE": "fuse"} in offered
    assert {"PLACE@map.1/twist.1/inner": "cut"} in offered
    lowered = _lower_cut(_sdpa_graph(causal), "PLACE@map.1/twist.1/inner")
    cuda = [node for node in lowered.nodes.values() if type(node.op).__name__ == "CudaOp"]
    assert len(cuda) == 2 + causal  # the two pieces of the cut; the causal mask is its own pointwise kernel
    workspace = next(node.output for node in cuda if "__place_" in node.id)
    assert workspace.dtype.name == "f32"


def test_recorded_sdpa_cut_decodes_exactly_and_stale_path_fails_loudly() -> None:
    wire = graph_to_wire(_sdpa_graph(False))
    fields = {
        "name": "sdpa.route",
        "gpu_name": "",
        "compute_cap": (12, 0),
        "model": None,
        "program_index": 0,
        "program_wire": wire,
        "origins": ("out",),
        "bindings": (),
        "pins": (),
        "measurements": None,
        "ranking": None,
    }
    assert decode_record(GoldenRecord(knobs={"PLACE@map.1/twist.1/inner": "cut"}, **fields)) is None
    reason = decode_record(GoldenRecord(knobs={"PLACE@missing": "cut"}, **fields))
    assert reason is not None and "does not resolve" in reason


@requires_cuda
def test_softmax_state_cut_is_offered_and_pinned_cut_lowers() -> None:
    offered = _offered(_softmax_graph(), frontend=True)
    # The carrier is the one cuttable seam, spelled by its route beside the epilogue map sites.
    assert {"PLACE": "fuse"} in offered and {"PLACE@map.1/map.1/twist": "cut"} in offered
    lowered = _lower_cut(_softmax_graph(), "PLACE")
    cuda = [node for node in lowered.nodes.values() if type(node.op).__name__ == "CudaOp"]
    assert len(cuda) == 2
    assert len(next(node for node in cuda if "__place_" in node.id).outputs) == 2  # maximum + denominator state
    values = np.linspace(-3, 3, 128, dtype=np.float16).reshape(4, 32)
    got = CudaBackend().run(lowered, input_data={"x": values})[0].outputs["out"]
    shifted = values.astype(np.float32) - values.max(axis=-1, keepdims=True).astype(np.float32)
    expected = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(got, expected, rtol=2e-3, atol=2e-3)


def test_mimo_cut_preserves_both_outputs_and_lowers_both_pieces() -> None:
    offered = _offered(_mimo_graph())
    cuts = [next(iter(row)) for row in offered if next(iter(row.values())) == "cut"]
    assert len(cuts) == 2
    lowered = _lower_cut(_mimo_graph(), cuts[0])
    assert lowered.outputs == ["out0", "out1"]
    assert sum(type(node.op).__name__ == "CudaOp" for node in lowered.nodes.values()) == 2


def test_scoped_place_cut_is_consumed_once_by_both_pieces() -> None:
    fragment = _nested_attention_cut({"PLACE@map.1/twist.1/inner.2/map": "cut"})
    pieces = [node for node in fragment.nodes.values() if isinstance(node.op, TileOp)]

    assert pieces and all(node.op.placement_decided for node in pieces)
    node = _piece_with_seam(fragment)
    match = Match(graph=fragment, root_node_id=node.id, rule=Rule(name="test", pattern=[]))
    # The pin is consumed: the rule offers nothing under PLACE again — its remaining domain (split
    # forks) or, with none pending, its skip.
    with pinned_knobs({"PLACE@map.1/twist.1/inner.2/map": "cut"}):
        try:
            result = _CUT.rewrite(match, node)
        except RuleSkipped as skipped:
            assert "no pending kernel-set cut" in str(skipped)
            return
    options = result if isinstance(result, list) else [result]
    assert options and all(not any(name.startswith("PLACE") for name in option.knobs) for option in options)


def test_bare_place_cut_is_consumed_once_by_both_pieces() -> None:
    fragment = _nested_attention_cut({"PLACE": "cut"})
    node = _piece_with_seam(fragment)
    match = Match(graph=fragment, root_node_id=node.id, rule=Rule(name="test", pattern=[]))

    assert node.op.placement_decided
    with pinned_knobs({"PLACE": "cut"}):
        result = _CUT.rewrite(match, node)
    options = result if isinstance(result, list) else [result]
    assert options and all(not any(name.startswith("PLACE") for name in option.knobs) for option in options)


def test_unpinned_place_keeps_offering_fuse_and_recursive_cuts() -> None:
    fragment = _nested_attention_cut({})
    node = _piece_with_seam(fragment)
    match = Match(graph=fragment, root_node_id=node.id, rule=Rule(name="test", pattern=[]))

    assert not node.op.placement_decided
    options = _CUT.rewrite(match, node)
    assert {"fuse", "cut"} <= {value for option in options for value in option.knobs.values()}


def test_composed_scoped_place_pins_cut_together_and_foreign_pins_are_skipped() -> None:
    """Every scoped PLACE pin that resolves on one kernel joins ONE realization — a producer per
    seam and one consumer, with a producer reading another seam's workspace when its value nests
    inside it — while a pin whose site path exists on no kernel here is another kernel's and is
    skipped, never an error."""
    case = Path(__file__).parents[1] / "realization/cases/attention/rmsnorm-qk-sdpa-composed-cut_xfail_realized.yaml"
    (record,) = load_golden_records(load_golden_file(case))
    tile = _lifted_target(record)
    graph = Graph()
    for name, tensor in tile.inputs.items():
        graph.add_node(InputOp(), [], tensor, node_id=name)
    graph.add_node(tile, list(tile.inputs), next(iter(tile.outputs.values())), node_id=tile.name)
    match = Match(graph=graph, root_node_id=tile.name, rule=Rule(name="test", pattern=[]))
    pins = {
        "PLACE@map.1/twist.1/inner.1/map": "cut",  # the normalized-Q cone
        "PLACE@map.1/twist.1/inner.2/map": "cut",  # the normalized-K cone
        "PLACE@map.1/twist.1/inner.2/map.3/map.1/reduce": "cut",  # the K statistic nested inside it
        "PLACE@map.9/map": "cut",  # no such site here — another kernel's pin
    }
    with pinned_knobs(pins):
        fork = _CUT.rewrite(match, graph.nodes[tile.name])
    assert set(fork.knobs) == {
        "PLACE@map.1/twist.1/inner.1/map",
        "PLACE@map.1/twist.1/inner.2/map",
        "PLACE@map.1/twist.1/inner.2/map.3/map.1/reduce",
    }
    (fragment,) = fork.expand()
    pieces = [node for node in fragment.nodes.values() if isinstance(node.op, TileOp)]
    producers = [node for node in pieces if "__place_" in node.id]
    assert len(producers) == 3 and len(pieces) == 4
    assert all(node.op.placement_decided for node in pieces)
    workspaces = {node.id for node in producers}
    assert any(set(node.inputs) & workspaces for node in producers), "the nested value's producer must read a sibling workspace"


def test_bare_and_scoped_place_cuts_compose_in_one_decision() -> None:
    match, graph = _composed_case_match()
    pins = {"PLACE": "cut", "PLACE@map.1/twist.1/inner.2/map": "cut"}

    with pinned_knobs(pins):
        fork = _CUT.rewrite(match, graph.nodes[match.root_node_id])

    assert len(fork.knobs) == 2 and set(fork.knobs.values()) == {"cut"}
    (fragment,) = fork.expand()
    pieces = [node for node in fragment.nodes.values() if isinstance(node.op, TileOp)]
    assert len(pieces) == 3 and all(node.op.placement_decided for node in pieces)


def _composed_case_match() -> tuple[Match, Graph]:
    case = Path(__file__).parents[1] / "realization/cases/attention/rmsnorm-qk-sdpa-composed-cut_xfail_realized.yaml"
    (record,) = load_golden_records(load_golden_file(case))
    tile = _lifted_target(record)
    graph = Graph()
    for name, tensor in tile.inputs.items():
        graph.add_node(InputOp(), [], tensor, node_id=name)
    graph.add_node(tile, list(tile.inputs), next(iter(tile.outputs.values())), node_id=tile.name)
    return Match(graph=graph, root_node_id=tile.name, rule=Rule(name="test", pattern=[])), graph


def _receipt_fields() -> dict:
    return {
        "name": "sdpa.child",
        "gpu_name": "",
        "compute_cap": (12, 0),
        "model": None,
        "program_index": 0,
        "program_wire": graph_to_wire(_sdpa_graph(False)),
        "origins": ("out",),
        "bindings": (),
        "pins": (("PLACE@map.1/twist.1/inner", "cut"),),
        "measurements": None,
        "ranking": None,
    }


def test_child_identity_receipts_decode_per_child_and_join_by_stored_identity() -> None:
    """Conflicting per-child schedules behind one pinned cut persist as sibling receipts: each
    stored child identity selects its own kernel's rows, a sibling child's row does not vouch for
    it, and the strict decode joins by the stored identity instead of the pre-cut lift."""
    fields = _receipt_fields()
    parent = GoldenRecord(knobs={}, **fields)
    lift_identity = _lifted_target(parent).identity_key(with_io=True)
    children = {i: rows for i, rows in _replay(parent, exhaustive=True).rows.items() if i is not None and i != lift_identity}
    assert len(children) == 2, "the pinned cut must resolve to two distinctly identified child kernels"
    (id_a, rows_a), (id_b, rows_b) = sorted(children.items())
    row_a = next(iter(rows_a - rows_b), None)
    assert row_a is not None, "the children must offer at least one distinguishing schedule row"

    receipt = GoldenRecord(knobs=dict(row_a), identity=id_a, **fields)
    assert decode_record(receipt) is None
    assert kernel_identity(receipt) == id_a

    sibling = GoldenRecord(knobs=dict(row_a), identity=id_b, **fields)
    reason = decode_record(sibling)
    assert reason is not None and "no enumerated row of the identified kernel" in reason

    stale = GoldenRecord(knobs=dict(row_a), identity="0" * 64, **fields)
    reason = decode_record(stale)
    assert reason is not None and "equals none" in reason


def test_child_identity_receipt_selects_one_kernel_from_multi_kernel_loop_target() -> None:
    """A stored child identity is the selector when a regenerated target now lowers to several
    kernels; strict decoding must consult that identity's rows before requiring a one-kernel lift."""
    graph = _sdpa_graph(False)
    _input(graph, "x", (4, 32))
    graph.add_node(SoftmaxOp(axis=-1), ["x"], Tensor("softmax", (4, 32), "f16"), node_id="softmax")
    graph.inputs.append("x")
    graph.outputs.append("softmax")
    loop = Pipeline.build(LOOP_PASSES).run(graph.copy(), ctx=_CTX)
    fields = {
        **_receipt_fields(),
        "program_wire": graph_to_wire(graph),
        "origins": (),
        "loop_index": 0,
        "loop_wire": loop_graph_to_wire(loop),
    }
    parent = GoldenRecord(knobs={}, **fields)
    with pytest.raises(ValueError, match="target lowers to 2 kernels"):
        _lifted_target(parent)
    identity, rows = next((identity, rows) for identity, rows in _replay(parent, exhaustive=True).rows.items() if identity is not None)
    receipt = GoldenRecord(knobs=dict(next(iter(rows))), identity=identity, **fields)
    assert decode_record(receipt) is None


def test_evidence_rows_key_each_row_by_the_kernel_it_decides() -> None:
    """Golden evidence is per kernel. A routing record is a route row under the signature of the
    kernel its cut was offered on; a child-identity receipt's ``PLACE`` pin is the same route row,
    and its schedule row is keyed under its child's signature — a piece inherits nothing from the
    kernel it replaced."""
    from emmy.compiler.pipeline.search.golden import evidence_rows, records_override

    fields = {**_receipt_fields(), "measurements": {"emmy_us": 1.0, "reference_us": 2.0, "reference_backend": "torch"}}
    route = {"PLACE@map.1/twist.1/inner": "cut"}
    routing = GoldenRecord(knobs=route, **{**fields, "pins": ()})
    parent = GoldenRecord(knobs={}, **fields)
    lift_identity = _lifted_target(parent).identity_key(with_io=True)
    replay = _replay(parent, exhaustive=True)
    child, rows = next((identity, rows) for identity, rows in replay.rows.items() if identity is not None and identity != lift_identity)
    receipt = GoldenRecord(knobs=dict(next(iter(rows))), identity=child, **fields)
    parent_signature = frozenset((key, str(value)) for key, value in parent.structural_features.items())

    assert _replay(routing).arms == ((parent_signature, route),)
    with records_override([routing, receipt]):
        got = evidence_rows("", (12, 0))
    assert got == [
        (parent_signature, route, 1.0, routing.name),
        (parent_signature, route, 1.0, receipt.name),
        (replay.signatures[child], receipt.schedule_row, 1.0, receipt.name),
    ]


def test_receipt_validation_requires_child_identity_and_place_pins_stay_live() -> None:
    from types import SimpleNamespace

    from emmy.compiler.pipeline.search.golden import regime_live

    fields = _receipt_fields()
    document = {
        "compute_cap": [12, 0],
        "programs": [fields["program_wire"]],
        "configs": [
            {
                "program": 0,
                "target": {"origins": ["out"]},
                "realizations": [
                    {"name": "sdpa.child", "bindings": {}, "pins": {"PLACE@map.1/twist.1/inner": "cut"}, "knobs": {"WORK": "w4x2"}}
                ],
            }
        ],
    }
    with pytest.raises(ValueError, match="child-identity schedule receipt"):
        validate_golden_file(document)
    document["configs"][0]["realizations"][0]["identity"] = "0" * 64
    validate_golden_file(document)
    receipt = SimpleNamespace(pin_map={"PLACE@map.1/twist.1/inner": "cut"})
    assert regime_live(receipt), "a receipt's routing pins are its route, never a dead env regime"


def test_pool_group_fuses_node_id_respellings_and_keys_on_pins() -> None:
    """``pool_group`` composes the target kernels' identity keys, so two recordings of ONE
    program whose node ids differ (separate recording sessions) fuse into one enumeration
    group — the wire digest this replaced split them — while a different pin regime still
    keys apart."""
    fields = _receipt_fields()
    respelled = _sdpa_graph(False)
    for nid in [n for n in respelled.nodes if n not in respelled.inputs]:
        respelled.rename_node(nid, f"session2_{nid}")
    twin_fields = {
        **fields,
        "program_wire": graph_to_wire(respelled),
        "origins": tuple(f"session2_{o}" for o in fields["origins"]),
    }
    a = GoldenRecord(knobs={}, **fields)
    b = GoldenRecord(knobs={}, **twin_fields)
    assert a.pool_group == b.pool_group, "node-id spelling must not split an enumeration group"

    unpinned = GoldenRecord(knobs={}, **{**fields, "pins": ()})
    assert unpinned.pool_group != a.pool_group, "the pin regime is a group-key term"


def _cone_seam() -> CutSite:
    """A bare seam record standing in for a clustered operand cone."""
    node = projection((), (Load(name="w", input="w", index=(Var("n"), Var("k"))),), results=("w",))
    return CutSite(node=node, spelling="PLACE@map.1/twist.1/inner.2/map", axes=(Axis("n", 8), Axis("k", 8)), dtypes=(F16,))


def test_alpha_equivalent_operand_cones_cluster_into_one_seam() -> None:
    """Two operand cones spelling the same value are ONE placement decision: the representative
    carries the other as a sibling with its capture correspondence."""
    from emmy.compiler.pipeline.passes.lowering.tile._cut import _cluster_value_seams

    consumer = object()
    same = [_cone_seam(), _cone_seam()]

    clustered = _cluster_value_seams(same, {id(seam.node): consumer for seam in same})
    assert len(clustered) == 1 and len(clustered[0].siblings) == 1


def test_every_seam_is_an_unpinned_arm() -> None:
    """The unpinned fork offers every cuttable seam as its own structural arm, spelled by the same
    key the pin path resolves."""
    match, graph = _composed_case_match()
    node = next(node for node in graph.nodes.values() if isinstance(node.op, TileOp))
    options = _CUT.rewrite(match, node)
    arms = [dict(option.knobs) for option in options if "cut" in option.knobs.values()]
    seams = cuttable_seams(node.op)
    assert [set(arm) for arm in arms] == [{seam.spelling} for seam in seams]
