"""Session-scoped structural reuse for binding-bearing execution plans.

An :class:`~emmy.compiler.backend.plan.ExecutionPlan` separates immutable compiled
structure from checkpoint bindings, but its ``WeightSpec`` records still name the
concrete source tensors.  This cache erases only those source addresses from its key
and stored template, then restores the current graph's addresses on every lookup.

The cache is deliberately process-local and caller-owned.  A session therefore freezes
one compiler/config/evidence view, like an in-memory pack, without inventing a second
persistent validity contract.  It also keys the exact graph wire form (including names
and hints), rather than the alpha-invariant ``Graph.structural_key``: execution plans
refer to concrete buffer names, and graph hints can change the generated ABI.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from dataclasses import dataclass

from emmy.compiler.backend.plan import ExecutionPlan, WeightSpec
from emmy.compiler.graph import Graph

_SLOT_PREFIX = "__emmy_plan_source_slot_"


class PlanTemplateBindingError(ValueError):
    """A compiled plan cannot be represented by the graph's binding slots."""


@dataclass(frozen=True)
class _Bindings:
    payload: str
    actual_by_slot: dict[str, str]
    slot_by_actual: dict[str, str]


def _binding_neutral_graph(graph: Graph) -> _Bindings:
    """Return an exact graph payload with external source addresses replaced by slots.

    Slot allocation is per distinct path, preserving aliasing.  ``source_parts`` order
    and shapes stay in the payload, as do every other op field, graph/node hints, and
    interface name.  Nested constant source graphs are handled recursively.
    """
    wire = graph.to_dict()
    actual_by_slot: dict[str, str] = {}
    slot_by_actual: dict[str, str] = {}

    def slot(path: str) -> str:
        found = slot_by_actual.get(path)
        if found is not None:
            return found
        found = f"{_SLOT_PREFIX}{len(slot_by_actual):04d}__"
        slot_by_actual[path] = found
        actual_by_slot[found] = path
        return found

    def visit_graph(graph_wire: dict) -> None:
        # Sort by the exact node id so slot numbering is independent of insertion order.
        # Node ids and their insertion order remain in the payload and therefore in cache
        # identity: independent-node order can influence deterministic rewrite traversal.
        nodes = graph_wire.get("nodes", {})
        graph_wire["__node_order__"] = list(nodes)
        for nid in sorted(nodes):
            node = nodes[nid]
            if node.get("op") != "ConstantOp":
                continue
            fields = node.get("op_fields", {})
            source_path = fields.get("source_path")
            if source_path is not None:
                fields["source_path"] = slot(source_path)
            source_parts = fields.get("source_parts") or []
            canonical_parts = []
            for part in source_parts:
                if not isinstance(part, (list, tuple)) or len(part) != 2 or not isinstance(part[0], str):
                    raise PlanTemplateBindingError(f"malformed ConstantOp.source_parts entry at {nid!r}: {part!r}")
                canonical_parts.append((slot(part[0]), part[1]))
            if source_parts:
                fields["source_parts"] = tuple(canonical_parts)
            source_graph = fields.get("source_graph")
            if isinstance(source_graph, dict) and isinstance(source_graph.get("__graph__"), dict):
                visit_graph(source_graph["__graph__"])

    visit_graph(wire)
    # The full canonical payload is the dict key: profile counts are small, and avoiding a
    # digest means a collision can never hand one graph another graph's executable plan.
    payload = json.dumps(wire, sort_keys=True, separators=(",", ":"))
    return _Bindings(payload=payload, actual_by_slot=actual_by_slot, slot_by_actual=slot_by_actual)


def _weight_to_template(nid: str, weight: WeightSpec, slot_by_actual: dict[str, str]) -> WeightSpec:
    def template_path(path: str) -> str:
        try:
            return slot_by_actual[path]
        except KeyError as exc:
            raise PlanTemplateBindingError(
                f"compiled plan weight {nid!r} names source {path!r}, which was not present in the input graph"
            ) from exc

    return dataclasses.replace(
        weight,
        source_path=template_path(weight.source_path) if weight.source_path is not None else None,
        source_parts=tuple((template_path(path), shape) for path, shape in weight.source_parts),
    )


def _plan_to_template(plan: ExecutionPlan, slot_by_actual: dict[str, str]) -> ExecutionPlan:
    weights = {nid: _weight_to_template(nid, weight, slot_by_actual) for nid, weight in plan.weights.items()}
    return dataclasses.replace(plan, weights=weights)


def _instantiate_plan_template(template: ExecutionPlan, actual_by_slot: dict[str, str]) -> ExecutionPlan:
    """Restore one graph instance's concrete source paths, failing closed on any gap."""

    def actual_path(nid: str, path: str) -> str:
        try:
            return actual_by_slot[path]
        except KeyError as exc:
            raise PlanTemplateBindingError(f"plan template weight {nid!r} has unresolved source slot {path!r}") from exc

    weights = {
        nid: dataclasses.replace(
            weight,
            source_path=actual_path(nid, weight.source_path) if weight.source_path is not None else None,
            source_parts=tuple((actual_path(nid, path), shape) for path, shape in weight.source_parts),
        )
        for nid, weight in template.weights.items()
    }
    # Return a fresh plan even when it has no weights.  Runtime construction is already
    # per-call; the fresh shell also prevents a caller from mutating the cached template.
    return dataclasses.replace(template, weights=weights)


class PlanTemplateCache:
    """Binding-neutral execution-plan templates for one immutable compile session."""

    def __init__(self) -> None:
        self._entries: dict[str, ExecutionPlan] = {}
        self.hits = 0
        self.misses = 0

    def resolve(self, graph: Graph, compile_plan: Callable[[Graph], ExecutionPlan]) -> ExecutionPlan:
        """Compile ``graph`` once per exact binding-neutral profile and rebind its plan.

        ``compile_plan`` receives the original graph on a miss; source spelling remains a
        loader/compiler concern.  Only the resulting plan's binding addresses are templated.
        Failed compilation or templating is never cached.
        """
        bindings = _binding_neutral_graph(graph)
        template = self._entries.get(bindings.payload)
        if template is None:
            self.misses += 1
            plan = compile_plan(graph)
            template = _plan_to_template(plan, bindings.slot_by_actual)
            instantiated = _instantiate_plan_template(template, bindings.actual_by_slot)
            self._entries[bindings.payload] = template
            return instantiated
        else:
            self.hits += 1
        return _instantiate_plan_template(template, bindings.actual_by_slot)

    def __len__(self) -> int:
        return len(self._entries)
