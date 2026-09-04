"""Kernel-placement forks over closed stored Fold edges."""

from __future__ import annotations

from dataclasses import replace
from importlib import import_module

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
from emmy.compiler.ir.tensor.ir import ElementwiseOp
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
    _target_kernel_nodes,
    decode_record,
    kernel_identity,
    validate_golden_file,
)
from emmy.compiler.pipeline.search.pins import pinned_knobs
from emmy.compiler.torch_wire import graph_to_wire
from tests.compiler.helpers import case_target_tile, direct_classic_leaf, requires_cuda
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


def _case_match(case: str) -> tuple[Match, Graph]:
    """A one-node match on a corpus case's lifted target — the fork point the cut pass rewrites."""
    tile = case_target_tile(case)
    graph = Graph()
    for name, tensor in tile.inputs.items():
        graph.add_node(InputOp(), [], tensor, node_id=name)
    graph.add_node(tile, list(tile.inputs), next(iter(tile.outputs.values())), node_id=tile.name)
    return Match(graph=graph, root_node_id=tile.name, rule=Rule(name="test", pattern=[])), graph


def _nested_attention_cut(pins: dict[str, str]) -> Graph:
    match, graph = _case_match("attention/rmsnorm-gqa-b-cut.yaml")
    with pinned_knobs(pins):
        result = _CUT.rewrite(match, graph.nodes[match.root_node_id])
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
    match, graph = _case_match("attention/rmsnorm-qk-sdpa-composed-cut_xfail_realized.yaml")
    pins = {
        "PLACE@map.1/twist.1/inner.1/map": "cut",  # the normalized-Q cone
        "PLACE@map.1/twist.1/inner.2/map": "cut",  # the normalized-K cone
        "PLACE@map.1/twist.1/inner.2/map.3/map.1/reduce": "cut",  # the K statistic nested inside it
        "PLACE@map.9/map": "cut",  # no such site here — another kernel's pin
    }
    with pinned_knobs(pins):
        fork = _CUT.rewrite(match, graph.nodes[match.root_node_id])
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
    match, graph = _case_match("attention/rmsnorm-qk-sdpa-composed-cut_xfail_realized.yaml")
    pins = {"PLACE": "cut", "PLACE@map.1/twist.1/inner.2/map": "cut"}

    with pinned_knobs(pins):
        fork = _CUT.rewrite(match, graph.nodes[match.root_node_id])

    assert len(fork.knobs) == 2 and set(fork.knobs.values()) == {"cut"}
    (fragment,) = fork.expand()
    pieces = [node for node in fragment.nodes.values() if isinstance(node.op, TileOp)]
    assert len(pieces) == 3 and all(node.op.placement_decided for node in pieces)


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
    """Golden evidence is per kernel. A target's entries walk one path: the leading entry (the
    routing record here) decides the parent's placement fork and is its route row under the
    signature of the kernel the cut was offered on; the child-identity receipt decides only the
    forks of the kernel it names, and its schedule row is keyed under that child's signature — a
    piece inherits nothing from the kernel it replaced."""
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
        (replay.signatures[child], receipt.schedule_row, 1.0, receipt.name),
    ]


def test_multi_output_kernel_record_derives_the_identity_its_live_fork_carries() -> None:
    """A record whose one target kernel writes SEVERAL output buffers must derive the identity its
    live fork carries. Every evidence row a golden contributes is keyed by that identity, so a
    derivation that kept only output slot 0 keys the record's rows off a fingerprint no fork can
    produce and the deploy reads none of them. The derivation lifts the persisted kernel and the
    fork root op is whatever the matcher's ``with_io`` produced — a map holding every output slot,
    which is why the lift goes through the same call."""
    graph = Graph()
    _input(graph, "x", (8,))
    graph.add_node(ElementwiseOp("relu"), ["x"], Tensor("hot", (8,), "f16"), node_id="hot")
    graph.add_node(ElementwiseOp("negative"), ["hot"], Tensor("cold", (8,), "f16"), node_id="cold")
    graph.inputs, graph.outputs = ["x"], ["hot", "cold"]
    loop = Pipeline.build(LOOP_PASSES).run(graph.copy(), ctx=_CTX)
    fields = {
        **_receipt_fields(),
        "name": "fused.multi_output",
        "pins": (),
        "program_wire": graph_to_wire(graph),
        "origins": (),
        "loop_index": 0,
        "loop_wire": loop_graph_to_wire(loop),
    }
    record = GoldenRecord(knobs={}, **fields)
    _lowered, nodes = _target_kernel_nodes(record)
    assert len(nodes) == 1 and len(nodes[0].outputs) == 2, "the fused target must be ONE kernel writing two buffers"

    identity = kernel_identity(record)
    rows = _replay(record, exhaustive=True).rows
    assert identity in rows, "the derived identity names no kernel the live resolve offers"
    # The join is the subject; spelling one of that kernel's own rows shows the record decodes
    # strictly through it too.
    assert decode_record(GoldenRecord(knobs=dict(next(iter(rows[identity]))), **fields)) is None


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


# ---------------------------------------------------------------------------
# The routing lane: a recorded ROUTING row decides a placement fork.
# ---------------------------------------------------------------------------

#: A card in the ``emmy.gpu`` registry, so the record's context reconstructs without a live device.
_ROUTING_CARD = "NVIDIA GeForce RTX 5090"


def _sdpa_kernel_identity() -> str:
    """The deploy identity carried by the sdpa program's PLACEMENT fork — the PRE-CUT kernel, which
    is the kernel a routing row names: the route it records is the one decision taken on that
    kernel, before any piece of it exists. Probed off a resolve rather than restated here, so these
    tests pin the routing lane and not a second copy of the identity derivation."""
    from emmy.compiler.pipeline.fork import flatten_leaves

    ctx = Context.from_target((12, 0), gpu_name=_ROUTING_CARD)
    lowered = Pipeline.build(LOOP_PASSES).run(_sdpa_graph(False), ctx=ctx)
    seen: list[str] = []

    def decide(fp):
        if not seen and isinstance(fp.root_op, TileOp) and fp.match.rule.name == _CUT.__name__.rsplit(".", 1)[-1]:
            seen.append(fp.root_op.identity_key(with_io=True))
        return flatten_leaves(fp.options)[0]

    Run(pipeline=Pipeline.build(TILE_PASSES), ctx=ctx).resolve(lowered, decide)
    assert seen, "the sdpa program must offer a placement fork"
    return seen[0]


def _routing_record(knobs: dict, *, name: str = "sdpa.route") -> GoldenRecord:
    """A verified ROUTING row over the sdpa kernel — nothing but ``PLACE`` keys, which is what
    makes it a recorded placement rather than a recorded schedule. The identity is stored so this
    exercises the routing LANE and not the record-side identity derivation, which has its own
    tests."""
    return GoldenRecord(
        name=name,
        gpu_name=_ROUTING_CARD,
        compute_cap=(12, 0),
        model=None,
        program_index=0,
        program_wire=graph_to_wire(_sdpa_graph(False)),
        origins=("out",),
        bindings=(),
        pins=(),
        knobs=knobs,
        identity=_sdpa_kernel_identity(),
        measurements={"emmy_us": 1.0, "reference_us": 2.0, "reference_backend": "torch"},
        ranking=None,
    )


@pytest.fixture
def deployable_flags(monkeypatch):
    """The deployable nvcc regime, which the verified tier is scoped to.

    ``greedy_decide`` consults the tier only where ``H_opt`` is the deployable level: a recorded µs
    is a deployable-regime measurement and must not arbitrate a compile pinned to another
    optimization level. That gate is one tier-wide rule — it withholds recorded SCHEDULE rows just
    as it withholds routing rows — so a test of either lane has to state the regime it deploys in
    rather than inherit the suite's (``make test`` runs the correctness lane at ``-Xcicc -O1``).
    Same spelling the golden replay lane's tests and the realization harness use."""
    monkeypatch.setenv("EMMY_NVCC_FLAGS", "")


def _deploy_kernels(records: list) -> list[str]:
    """Resolve the sdpa program through the deploy policy with ``records`` as the card's corpus,
    and return the resolved kernel set. ``prior=None`` pins the non-recorded forks to emission
    order, so the recorded evidence is the only thing that can move the answer. Callers hold
    :func:`deployable_flags`, without which the verified tier declines to answer at all."""
    from emmy.compiler.pipeline.search.golden import records_override
    from emmy.compiler.pipeline.search.policy.greedy import greedy_decide

    ctx = Context.from_target((12, 0), gpu_name=_ROUTING_CARD)
    lowered = Pipeline.build(LOOP_PASSES).run(_sdpa_graph(False), ctx=ctx)
    with records_override(records):
        terminal, _trace = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=ctx).resolve(lowered, greedy_decide(prior=None))
    return sorted(node.id for node in terminal.nodes.values() if isinstance(node.op, TileOp))


#: The sdpa kernel's two offered seams, as the route codec spells them: the twist that carries the
#: softmax statistics, and the score contraction inside it.
_SDPA_ROUTES = ("PLACE@map.1/twist", "PLACE@map.1/twist.1/inner")

#: A well-formed route that stands on no site of THIS tree — one hop past the score contraction,
#: where the operand is a gmem slab and takes no hop of its own. What a recorded route looks like
#: after the tree it was spelled on moved.
_SDPA_STALE_ROUTE = "PLACE@map.1/twist.1/inner.1/map"


def test_a_recorded_routing_row_deploys_its_multi_seam_cut(deployable_flags) -> None:
    """The sdpa kernel offers its seams ONE at a time, so no arm of the placement fork spells a
    two-seam route. The recorded row is applied the way the ``--golden`` replay lane applies
    placement pins — published as pins, the rule re-asked — so the cut machinery composes both
    seams into the one realization it was measured on."""
    fused = _deploy_kernels([])
    assert len(fused) == 1, f"with no recorded route the fork falls to emission order (fuse): {fused}"

    routed = _deploy_kernels([_routing_record(dict.fromkeys(_SDPA_ROUTES, "cut"))])
    assert len(routed) > len(fused), f"the recorded route must deploy its fragments: {routed}"
    assert sum(1 for name in routed if "__place_" in name) >= 2, f"both recorded seams must be cut: {routed}"


def test_a_routing_row_naming_a_stale_seam_decides_nothing(caplog, deployable_flags) -> None:
    """Fail-closed. The cut machinery reads a pin that addresses no site on this kernel as naming
    ANOTHER kernel of the graph and drops it, which would silently deploy a shorter route than the
    one measured. The tier checks the composed decision back against the row, so a stale route
    warns and leaves the fork to pricing."""
    import logging

    from emmy.compiler.pipeline.search.policy import greedy

    row = {_SDPA_ROUTES[1]: "cut", _SDPA_STALE_ROUTE: "cut"}
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=greedy.logger.name):
        kernels = _deploy_kernels([_routing_record(row, name="sdpa.stale")])

    assert kernels == _deploy_kernels([]), "a partially realizable route must decide nothing at all"
    assert "sdpa.stale" in caplog.text and _SDPA_STALE_ROUTE in caplog.text, caplog.text
    assert "drift" in caplog.text


def test_a_recorded_schedule_row_never_routes(deployable_flags) -> None:
    """The lanes do not cross: a schedule row carries no ``PLACE`` key, so it is not a route and
    the placement fork stays with pricing even though the row joins the same kernel identity."""
    schedule_row = _routing_record({"WORK": "w4x1", "TILE": ""}, name="sdpa.schedule")
    assert not schedule_row.is_routing
    assert _deploy_kernels([schedule_row]) == _deploy_kernels([])


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
    match, graph = _case_match("attention/rmsnorm-qk-sdpa-composed-cut_xfail_realized.yaml")
    node = next(node for node in graph.nodes.values() if isinstance(node.op, TileOp))
    options = _CUT.rewrite(match, node)
    arms = [dict(option.knobs) for option in options if "cut" in option.knobs.values()]
    seams = cuttable_seams(node.op)
    assert [set(arm) for arm in arms] == [{seam.spelling} for seam in seams]
