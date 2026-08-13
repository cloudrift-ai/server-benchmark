"""Binding-neutral execution-plan template reuse (CPU-only)."""

from __future__ import annotations

import dataclasses
import json

import pytest

from emmy.compiler.backend.plan import WeightSpec, plan_from_graph, plan_to_dict
from emmy.compiler.backend.plan_cache import PlanTemplateBindingError, PlanTemplateCache
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.cuda import CudaOp


def _graph(
    source_path: str | None = "model.layers.0.w",
    *,
    source_parts: tuple[tuple[str, tuple[int, ...]], ...] = (),
    source_shape: tuple[int, ...] = (4, 4),
    scalar: float = 1.0,
) -> Graph:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (2, 4)), node_id="x")
    g.add_node(
        op=ConstantOp(
            name="w",
            source_path=source_path,
            source_parts=source_parts,
            source_shape=source_shape,
            source_dtype="f32",
        ),
        inputs=[],
        output=Tensor("w", (4, 4)),
        node_id="w",
    )
    g.add_node(op=ConstantOp(name="scale", value=scalar), inputs=[], output=Tensor("scale", (1,)), node_id="scale")
    g.add_node(
        op=CudaOp(
            kernel_source='extern "C" __global__ void k_cached() {}',
            kernel_name="k_cached",
            arg_order=("x", "w", "scale", "y"),
            grid=((1,), (1,), (1,)),
            block=((32,), (1,), (1,)),
        ),
        inputs=["x", "w", "scale"],
        output=Tensor("y", (2, 4)),
        node_id="y",
    )
    g.inputs = ["x"]
    g.outputs = ["y"]
    return g


def _counting_compiler(calls):
    def compile_plan(graph):
        calls.append(graph)
        return plan_from_graph(graph)

    return compile_plan


def test_address_only_change_hits_and_restores_actual_provenance():
    cache = PlanTemplateCache()
    calls = []
    compile_plan = _counting_compiler(calls)

    plan0 = cache.resolve(_graph("model.layers.0.w"), compile_plan)
    plan1 = cache.resolve(_graph("model.layers.1.w"), compile_plan)

    assert len(calls) == 1
    assert (cache.hits, cache.misses, len(cache)) == (1, 1, 1)
    assert plan0 is not plan1
    assert plan0.weights["w"].source_path == "model.layers.0.w"
    assert plan1.weights["w"].source_path == "model.layers.1.w"
    assert "__emmy_plan_source_slot_" not in json.dumps(plan_to_dict(plan1))
    # A later instantiation cannot mutate the already returned plan or the cached template.
    assert plan0.weights["w"].source_path == "model.layers.0.w"


def test_source_parts_order_and_aliasing_are_preserved_and_keyed():
    cache = PlanTemplateCache()
    calls = []
    compile_plan = _counting_compiler(calls)
    shape = (4, 4)

    first = (("l0.q", shape), ("l0.q", shape), ("l0.k", shape))
    same_pattern = (("l1.q", shape), ("l1.q", shape), ("l1.k", shape))
    different_pattern = (("l2.q", shape), ("l2.k", shape), ("l2.q", shape))
    cache.resolve(_graph(None, source_parts=first, source_shape=(12, 4)), compile_plan)
    rebound = cache.resolve(_graph(None, source_parts=same_pattern, source_shape=(12, 4)), compile_plan)
    cache.resolve(_graph(None, source_parts=different_pattern, source_shape=(12, 4)), compile_plan)

    assert len(calls) == 2
    assert rebound.weights["w"].source_parts == same_pattern
    assert (cache.hits, cache.misses, len(cache)) == (1, 2, 2)


def _nested_graph(path: str) -> Graph:
    source = Graph()
    source.add_node(
        op=ConstantOp(name="leaf", source_path=path, source_shape=(4, 4), source_dtype="f32"),
        inputs=[],
        output=Tensor("leaf", (4, 4)),
        node_id="leaf",
    )
    source.outputs = ["leaf"]
    graph = _graph(None)
    graph.nodes["w"].op = ConstantOp(name="w", source_graph=source, source_shape=(4, 4), source_dtype="f32")
    return graph


def test_nested_source_graph_rebinds_and_pack_wire_contains_only_real_hit_path():
    cache = PlanTemplateCache()
    calls = []

    def compile_nested(graph):
        calls.append(graph)
        source = graph.nodes["w"].op.source_graph
        leaf_path = source.nodes["leaf"].op.source_path
        # Model generic decomposition surfacing the nested external leaf as the plan's
        # bindable constant. The cache must know that path came from the nested record.
        return plan_from_graph(_graph(leaf_path))

    first = cache.resolve(_nested_graph("model.layers.0.w.storage"), compile_nested)
    hit = cache.resolve(_nested_graph("model.layers.1.w.storage"), compile_nested)

    assert len(calls) == 1
    assert first.weights["w"].source_path == "model.layers.0.w.storage"
    assert hit.weights["w"].source_path == "model.layers.1.w.storage"
    pack_wire = json.dumps(plan_to_dict(hit))
    assert "model.layers.1.w.storage" in pack_wire
    assert "model.layers.0.w.storage" not in pack_wire
    assert "__emmy_plan_source_slot_" not in pack_wire


def test_codegen_fields_and_hints_split_profiles():
    cache = PlanTemplateCache()
    calls = []
    compile_plan = _counting_compiler(calls)

    cache.resolve(_graph("l0.w"), compile_plan)
    hinted = _graph("l1.w")
    hinted.hints.set("cuda.indirect_inputs", ("w",))
    cache.resolve(hinted, compile_plan)
    cache.resolve(_graph("l2.w", source_shape=(2, 8)), compile_plan)
    cache.resolve(_graph("l3.w", scalar=2.0), compile_plan)

    assert len(calls) == 4
    assert (cache.hits, cache.misses, len(cache)) == (0, 4, 4)


def test_unknown_compiled_source_fails_closed_and_is_not_cached():
    cache = PlanTemplateCache()
    graph = _graph("known.w")

    def bad_compile(g):
        plan = plan_from_graph(g)
        weights = dict(plan.weights)
        weights["w"] = dataclasses.replace(weights["w"], source_path="invented.w")
        return dataclasses.replace(plan, weights=weights)

    with pytest.raises(PlanTemplateBindingError, match="not present in the input graph"):
        cache.resolve(graph, bad_compile)

    assert len(cache) == 0
    assert (cache.hits, cache.misses) == (0, 1)


def test_unresolved_template_slot_fails_closed():
    from emmy.compiler.backend.plan_cache import _instantiate_plan_template

    template = plan_from_graph(_graph("known.w"))
    template = dataclasses.replace(template, weights={"w": WeightSpec(source_path="missing-slot")})
    with pytest.raises(PlanTemplateBindingError, match="unresolved source slot"):
        _instantiate_plan_template(template, {})
