"""Placement cuts at closed fused Loop-IR values.

A value cut is the inverse of schedule-blind fusion at an SSA boundary.  The child evaluates one
closed pure value over its producer coordinates and writes either an existing live output or a
new workspace.  The parent replaces every alpha-equivalent inline spelling with a Load, removes
live outputs transferred to the child, and drops the now-dead producer cones.  Both pieces are
ordinary LoopOps and retain no schedule from the kernel they replace.

Construction is deliberately fail-closed: only exact value classes from ``_demand`` qualify;
output writes must be unique, scalar, non-atomic, and indexed only by the producer coordinates;
the parent must retain at least one original output; and a new workspace requires an explicit
value dtype.  No effect or unresolved capture moves into the child.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import Assign, Body, Load, Loop, Select, Write
from emmy.compiler.ir.stmt.normalize import rename_ssa_sequential
from emmy.compiler.pipeline.knob import consume_kernel_row
from emmy.compiler.pipeline.passes.lowering.tile._cut import _nest
from emmy.compiler.pipeline.passes.lowering.tile._demand import ValueClass, ValueOccurrence, ValueUse, value_demands
from emmy.compiler.pipeline.pipeline import Match, RuleSkipped


@dataclass(frozen=True)
class ValueCut:
    """One closed value class and the occurrence evaluated by its child piece."""

    value: ValueClass
    producer: ValueOccurrence

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.value.live_outputs


def _coordinate_axes(occurrence: ValueOccurrence):
    by_name = {axis.name: axis for axis in occurrence.axes}
    try:
        return tuple(by_name[name] for name in occurrence.coordinate_axes)
    except KeyError as exc:
        raise RuleSkipped(f"value coordinate {exc.args[0]!r} is not bound at its definition") from exc


def _write_uses(value: ValueClass) -> dict[str, tuple[ValueOccurrence, ValueUse, Write]]:
    """The unique direct Write of each live port in this value class."""
    found: dict[str, list[tuple[ValueOccurrence, ValueUse, Write]]] = {output: [] for output in value.live_outputs}
    for occurrence in value.occurrences:
        for use in occurrence.uses:
            if use.output in found and isinstance(use.consumer, Write):
                found[use.output].append((occurrence, use, use.consumer))
    if any(len(items) != 1 for items in found.values()):
        raise RuleSkipped("a materialized live value needs exactly one direct Write per output")
    return {output: items[0] for output, items in found.items()}


def _legal_write(write: Write, occurrence: ValueOccurrence, use: ValueUse) -> bool:
    axes = set(occurrence.coordinate_axes)
    demand = {axis.name for axis in use.axes}
    return (
        write.is_scalar
        and not write.atomic
        and write.swizzle == "NONE"
        and demand == axes
        and all(expr.free_vars() <= axes for expr in write.index)
    )


def value_cut_sites(loop: LoopOp) -> tuple[ValueCut, ...]:
    """Every exact closed value materialization the fused LoopOp can represent."""
    sites: list[ValueCut] = []
    for value in value_demands(loop):
        if not value.repeated and not value.live_outputs:
            continue
        if value.live_outputs and set(value.live_outputs) >= set(loop.outputs):
            continue
        computed = [occurrence for occurrence in value.occurrences if isinstance(occurrence.definition, (Assign, Select))]
        if not computed:
            continue
        if value.live_outputs:
            try:
                writes = _write_uses(value)
            except RuleSkipped:
                continue
            if any(not _legal_write(write, occurrence, use) for occurrence, use, write in writes.values()):
                continue
            all_writes = loop.body.writes
            if any(sum(write.output == output for write in all_writes) != 1 for output in writes):
                continue
            live_names = {occurrence.name for occurrence, _, _ in writes.values()}
            producer = min(
                (occurrence for occurrence in computed if occurrence.name in live_names),
                key=lambda occurrence: (len(occurrence.repeated_axes), len(occurrence.axes)),
                default=None,
            )
            if producer is None:
                continue
        else:
            producer = min(computed, key=lambda occurrence: (len(occurrence.repeated_axes), len(occurrence.axes)))
            if getattr(producer.definition, "dtype", None) is None:
                continue
        try:
            _coordinate_axes(producer)
        except RuleSkipped:
            continue
        sites.append(ValueCut(value=value, producer=producer))
    return tuple(sites)


def _substitute_coordinates(exprs, source: ValueOccurrence, target: ValueOccurrence):
    if len(source.coordinate_axes) != len(target.coordinate_axes):
        raise RuleSkipped("alpha-equivalent values disagree on coordinate rank")
    sigma = {old: Var(new) for old, new in zip(source.coordinate_axes, target.coordinate_axes, strict=True)}
    return tuple(expr.substitute(sigma) for expr in exprs)


def _live_indexes(site: ValueCut, writes: dict[str, tuple[ValueOccurrence, ValueUse, Write]]) -> dict[str, tuple]:
    indexes = {}
    for output, (occurrence, _, write) in writes.items():
        indexes[output] = _substitute_coordinates(write.index, occurrence, site.producer)
    return indexes


def _replacement_indexes(site: ValueCut, materialized: str, writes: dict[str, tuple[ValueOccurrence, ValueUse, Write]]) -> dict[str, Load]:
    if writes:
        source, _, template = writes[site.outputs[0]]
        return {
            occurrence.name: Load(
                name=occurrence.name,
                input=materialized,
                index=_substitute_coordinates(template.index, source, occurrence),
                dtype=getattr(occurrence.definition, "dtype", None),
            )
            for occurrence in site.value.occurrences
        }
    return {
        occurrence.name: Load(
            name=occurrence.name,
            input=materialized,
            index=tuple(Var(name) for name in occurrence.coordinate_axes),
            dtype=getattr(occurrence.definition, "dtype", None),
        )
        for occurrence in site.value.occurrences
    }


def _dce(body: Body) -> Body:
    """Keep the transitive definitions of the remaining effects, preserving loop structure."""
    definitions = body.definitions
    required: set[int] = set()
    pending: list[str] = []
    for stmt in body.iter():
        if isinstance(stmt, Write) or stmt.has_side_effects():
            required.add(id(stmt))
            pending.extend(stmt.deps())
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        stmt = definitions.get(name)
        if stmt is None:
            continue
        required.add(id(stmt))
        pending.extend(stmt.deps())

    def prune(stmts: Body) -> Body:
        kept = []
        for stmt in stmts:
            if isinstance(stmt, Loop):
                child = prune(stmt.body)
                if child:
                    kept.append(replace(stmt, body=child))
            elif id(stmt) in required:
                kept.append(stmt)
        return Body(kept)

    return prune(body)


def _pieces(loop: LoopOp, site: ValueCut, materialized: str, child_outputs: dict[str, str]) -> tuple[LoopOp, LoopOp]:
    writes = _write_uses(site.value) if site.outputs else {}
    axes = _coordinate_axes(site.producer)
    child_stmts = list(dict.fromkeys(site.producer.dependencies))
    if writes:
        indexes = _live_indexes(site, writes)
        child_stmts.extend(Write(output=child_outputs[output], index=indexes[output], value=site.producer.name) for output in site.outputs)
    else:
        child_stmts.append(Write(output=materialized, index=tuple(Var(axis.name) for axis in axes), value=site.producer.name))
    child = LoopOp(body=rename_ssa_sequential(Body(tuple(_nest(child_stmts, list(axes))))))

    replacements = _replacement_indexes(site, materialized, writes)
    removed_outputs = set(site.outputs)

    def replace_value(stmt):
        if isinstance(stmt, Write) and stmt.output in removed_outputs:
            return None
        defined = stmt.defines()
        if len(defined) == 1 and defined[0] in replacements:
            return replacements[defined[0]]
        return stmt

    parent_body = _dce(loop.body.map(replace_value))
    if not parent_body.writes:
        raise RuleSkipped("value materialization would leave no parent output")
    parent = LoopOp(body=parent_body, name=loop.name)
    return child, parent


def realize_value_cut(match: Match, root: Node, site: ValueCut) -> Graph:
    """Build a two-kernel fragment for one closed fused value."""
    old_outputs = root.buffer_names()
    selected = set(site.outputs)
    if not selected <= set(old_outputs):
        raise RuleSkipped("a materialized live value must own graph output ports")
    remaining = tuple(output for output in old_outputs if output not in selected)
    if site.outputs and not remaining:
        raise RuleSkipped("materializing every output would not split the kernel")

    writes = _write_uses(site.value) if site.outputs else {}
    for output, (_, _, write) in writes.items():
        tensor = match.graph.buffer(output)
        if tensor is None or len(write.index) != len(tensor.shape):
            raise RuleSkipped(f"output {output!r} does not have a representable value coordinate")

    token = site.outputs[0] if site.outputs else site.producer.name
    port_names = (root.output.name, *(f"{root.output.name}__placed_out{i}" for i in range(1, len(old_outputs))))
    output_buffers = dict(zip(old_outputs, port_names, strict=True))
    child_buffers = {output: output_buffers[output] for output in site.outputs}
    materialized = child_buffers[site.outputs[0]] if site.outputs else f"{root.id}__materialized_{token}"
    parent_buffers = {output: output_buffers[output] for output in remaining}
    child_op, parent_op = _pieces(root.op, site, materialized, child_buffers)

    frag = Graph()
    for input_name in root.inputs:
        frag.add_node(InputOp(), [], output=match.graph.buffer(input_name), node_id=input_name)
    child_reads = {load.input for load in child_op.body.loads}
    parent_reads = {load.input for load in parent_op.body.loads}

    if site.outputs:
        child_tensors = tuple(replace(match.graph.buffer(output), name=child_buffers[output]) for output in site.outputs)
    else:
        dtype = getattr(site.producer.definition, "dtype", None)
        if dtype is None:
            raise RuleSkipped("a non-output value materialization needs an explicit dtype")
        child_tensors = (Tensor(materialized, tuple(axis.extent for axis in _coordinate_axes(site.producer)), dtype),)
    frag.add_node(
        child_op,
        [input_name for input_name in root.inputs if input_name in child_reads],
        outputs=child_tensors,
        node_id=materialized,
    )

    output_rename = {output: parent_buffers[output] for output in remaining}

    def retarget(stmt):
        if isinstance(stmt, Write) and stmt.output in output_rename:
            return replace(stmt, output=output_rename[stmt.output])
        return stmt

    parent_op = replace(parent_op, body=parent_op.body.map(retarget))
    parent_tensors = tuple(replace(match.graph.buffer(output), name=parent_buffers[output]) for output in remaining)
    parent_inputs = [*(input_name for input_name in root.inputs if input_name in parent_reads), materialized]
    parent_id = parent_buffers[remaining[0]]
    frag.add_node(parent_op, parent_inputs, outputs=parent_tensors, node_id=parent_id)

    output_map = {output: output_buffers[output] for output in old_outputs}
    frag.outputs = [output_map[output] for output in old_outputs]
    match.output = output_map
    for node_id in (materialized, parent_id):
        frag.nodes[node_id].op.knobs = consume_kernel_row(frag.nodes[node_id].op.knobs)
    return frag


__all__ = ["ValueCut", "realize_value_cut", "value_cut_sites"]
