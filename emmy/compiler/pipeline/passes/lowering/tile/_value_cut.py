"""Placement cuts at closed fused Loop-IR values.

A value cut is the inverse of schedule-blind fusion at an SSA boundary.  The child evaluates one
closed pure value over its producer coordinates and writes either an existing live output or a
new workspace. When the dependency closure already computes another live output, the child owns
that port too. The parent replaces every alpha-equivalent inline spelling with a Load, removes
live outputs transferred to the child, and drops the now-dead producer cones. Both pieces are
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
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Select, Write
from emmy.compiler.ir.stmt.normalize import rename_ssa_sequential
from emmy.compiler.pipeline.knob import consume_kernel_row
from emmy.compiler.pipeline.passes.lowering.tile._cut import _nest, placement_pins
from emmy.compiler.pipeline.passes.lowering.tile._demand import ValueClass, ValueOccurrence, ValueUse, value_demands
from emmy.compiler.pipeline.pipeline import Match, RuleSkipped

_CUT = "cut"
_VALUE_PREFIX = "PLACE@="


@dataclass(frozen=True)
class _ValuePort:
    """A live output represented by an occurrence already available in the child."""

    output: str
    value: ValueClass
    producer: ValueOccurrence
    occurrence: ValueOccurrence
    use: ValueUse
    write: Write


@dataclass(frozen=True)
class ValueCut:
    """One closed value class, its child occurrence, and live ports computed there."""

    value: ValueClass
    producer: ValueOccurrence
    ports: tuple[_ValuePort, ...] = ()

    @property
    def outputs(self) -> tuple[str, ...]:
        return tuple(port.output for port in self.ports)


def spell_value_cut(site: ValueCut) -> str:
    """The reserved graph-value placement spelling for ``site``."""
    token = site.value.live_outputs[0] if site.value.live_outputs else site.producer.name
    return f"{_VALUE_PREFIX}{token}"


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


def _ports_for(value: ValueClass, producer: ValueOccurrence) -> tuple[_ValuePort, ...]:
    return tuple(
        _ValuePort(
            output=output,
            value=value,
            producer=producer,
            occurrence=occurrence,
            use=use,
            write=write,
        )
        for output, (occurrence, use, write) in _write_uses(value).items()
    )


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


def _producer_reaches_port(port: _ValuePort, order: dict[int, int], scopes: dict[int, tuple]) -> bool:
    """Whether ``port.producer`` is live and coordinate-compatible at its output Write."""
    producer_scope = scopes[id(port.producer.definition)]
    write_scope = scopes[id(port.write)]
    return (
        write_scope[: len(producer_scope)] == producer_scope
        and len(port.producer.coordinate_axes) == len(port.occurrence.coordinate_axes)
        and order[id(port.producer.definition)] < order[id(port.write)]
    )


def _statement_locations(body: Body) -> tuple[dict[int, int], dict[int, tuple]]:
    order: dict[int, int] = {}
    scopes: dict[int, tuple] = {}

    def walk(stmts: Body, scope: tuple) -> None:
        for stmt in stmts:
            order[id(stmt)] = len(order)
            scopes[id(stmt)] = scope
            for index, nested in enumerate(stmt.nested()):
                walk(nested, (*scope, (id(stmt), index)))

    walk(body, ())
    return order, scopes


def value_cut_sites(loop: LoopOp) -> tuple[ValueCut, ...]:
    """Every exact closed value materialization the fused LoopOp can represent."""
    sites: list[ValueCut] = []
    values = value_demands(loop)
    order, scopes = _statement_locations(loop.body)
    for value in values:
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
            primary_ports = _ports_for(value, producer)
            if any(not _producer_reaches_port(port, order, scopes) for port in primary_ports):
                continue
        else:
            producer = min(computed, key=lambda occurrence: (len(occurrence.repeated_axes), len(occurrence.axes)))
            if getattr(producer.definition, "dtype", None) is None:
                continue
            primary_ports = ()
        try:
            _coordinate_axes(producer)
        except RuleSkipped:
            continue
        ports = list(primary_ports)
        dependency_definitions = {id(stmt) for stmt in producer.dependencies}
        if primary_ports:
            for dependency in values:
                if dependency is value or not dependency.live_outputs:
                    continue
                sources = [occurrence for occurrence in dependency.occurrences if id(occurrence.definition) in dependency_definitions]
                for source in sorted(sources, key=lambda occurrence: (len(occurrence.axes), len(occurrence.repeated_axes))):
                    try:
                        dependency_ports = _ports_for(dependency, source)
                    except RuleSkipped:
                        continue
                    if all(
                        _legal_write(port.write, port.occurrence, port.use) and _producer_reaches_port(port, order, scopes)
                        for port in dependency_ports
                    ):
                        ports.extend(dependency_ports)
                        break
        if set(port.output for port in ports) >= set(loop.outputs):
            continue
        sites.append(ValueCut(value=value, producer=producer, ports=tuple(ports)))
    return tuple(sites)


def route_value_cut(candidates: tuple[ValueCut, ...]) -> tuple[str | None, ValueCut | None]:
    """Resolve a ``PLACE@=<value>`` pin, or a bare pin when no tree seam handled it."""
    if not candidates:
        return None, None
    by_key = {spell_value_cut(site): site for site in candidates}
    for key, value in placement_pins().items():
        if key == "PLACE":
            return (_CUT, candidates[0]) if value == _CUT else ("fuse", None)
        site = by_key.get(key)
        if site is not None:
            return (_CUT, site) if value == _CUT else ("fuse", None)
    return None, None


def _substitute_coordinates(exprs, source: ValueOccurrence, target: ValueOccurrence):
    if len(source.coordinate_axes) != len(target.coordinate_axes):
        raise RuleSkipped("alpha-equivalent values disagree on coordinate rank")
    sigma = {old: Var(new) for old, new in zip(source.coordinate_axes, target.coordinate_axes, strict=True)}
    return tuple(expr.substitute(sigma) for expr in exprs)


def _replacement_indexes(site: ValueCut, materialized: str, child_outputs: dict[str, str]) -> dict[str, Load]:
    if site.ports:
        replacements = {}
        seen: set[int] = set()
        for port in site.ports:
            marker = id(port.value)
            if marker in seen:
                continue
            seen.add(marker)
            for occurrence in port.value.occurrences:
                replacements[occurrence.name] = Load(
                    name=occurrence.name,
                    input=child_outputs[port.output],
                    index=_substitute_coordinates(port.write.index, port.occurrence, occurrence),
                    dtype=getattr(occurrence.definition, "dtype", None),
                )
        return replacements
    return {
        occurrence.name: Load(
            name=occurrence.name,
            input=materialized,
            index=tuple(Var(name) for name in occurrence.coordinate_axes),
            dtype=getattr(occurrence.definition, "dtype", None),
        )
        for occurrence in site.value.occurrences
    }


def _dce(body: Body, *, preserve_effects: bool = True) -> Body:
    """Keep the transitive definitions of the remaining effects, preserving loop structure."""
    definitions = body.definitions
    required: set[int] = set()
    pending: list[str] = []
    for stmt in body.iter():
        if isinstance(stmt, Write) or (preserve_effects and stmt.has_side_effects()):
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
    axes = _coordinate_axes(site.producer)
    reduction_derived = any(isinstance(stmt, Accum) for stmt in site.producer.dependencies)
    if site.ports and (reduction_derived or set(site.outputs) != set(site.value.live_outputs)):
        selected = set(site.outputs)
        ports = {port.output: port for port in site.ports}

        def select_outputs(stmt):
            if isinstance(stmt, Write) and stmt.output not in selected:
                return None
            if isinstance(stmt, Write):
                port = ports[stmt.output]
                return Write(
                    output=child_outputs[stmt.output],
                    index=_substitute_coordinates(stmt.index, port.occurrence, port.producer),
                    value=port.producer.name,
                    value_dtype=stmt.value_dtype,
                    atomic=stmt.atomic,
                    swizzle=stmt.swizzle,
                )
            return stmt

        child_body = _dce(loop.body.map(select_outputs), preserve_effects=False)
        if any(not isinstance(stmt, (Loop, Write)) and stmt.has_side_effects() for stmt in child_body.iter()):
            raise RuleSkipped("a reduction-derived value cut cannot move an effect into its child")
        child = LoopOp(body=rename_ssa_sequential(child_body))
    else:
        child_stmts = list(site.producer.dependencies)
        if site.ports:
            child_stmts.extend(
                Write(
                    output=child_outputs[port.output],
                    index=_substitute_coordinates(port.write.index, port.occurrence, port.producer),
                    value=port.producer.name,
                )
                for port in site.ports
            )
        else:
            child_stmts.append(Write(output=materialized, index=tuple(Var(axis.name) for axis in axes), value=site.producer.name))
        child = LoopOp(body=rename_ssa_sequential(Body(tuple(_nest(child_stmts, list(axes))))))

    replacements = _replacement_indexes(site, materialized, child_outputs)
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

    for port in site.ports:
        output, write = port.output, port.write
        tensor = match.graph.buffer(output)
        if tensor is None or len(write.index) != len(tensor.shape):
            raise RuleSkipped(f"output {output!r} does not have a representable value coordinate")

    token = site.value.live_outputs[0] if site.value.live_outputs else site.producer.name
    port_names = (root.output.name, *(f"{root.output.name}__placed_out{i}" for i in range(1, len(old_outputs))))
    output_buffers = dict(zip(old_outputs, port_names, strict=True))
    child_buffers = {output: output_buffers[output] for output in site.outputs}
    materialized = child_buffers[site.value.live_outputs[0]] if site.value.live_outputs else f"{root.id}__cut_{token}"
    parent_buffers = {output: output_buffers[output] for output in remaining}
    child_op, parent_op = _pieces(root.op, site, materialized, child_buffers)

    frag = Graph()
    for input_name in root.inputs:
        frag.add_node(InputOp(), [], output=match.graph.buffer(input_name), node_id=input_name)
    child_reads = {load.input for load in child_op.body.loads}
    parent_reads = {load.input for load in parent_op.body.loads}

    if site.outputs:
        child_output_order = tuple(output for output in old_outputs if output in selected)
        child_tensors = tuple(replace(match.graph.buffer(output), name=child_buffers[output]) for output in child_output_order)
        child_id = child_buffers[child_output_order[0]]
    else:
        dtype = getattr(site.producer.definition, "dtype", None)
        if dtype is None:
            raise RuleSkipped("a non-output value materialization needs an explicit dtype")
        child_tensors = (Tensor(materialized, tuple(axis.extent for axis in _coordinate_axes(site.producer)), dtype),)
        child_id = materialized
    frag.add_node(
        child_op,
        [input_name for input_name in root.inputs if input_name in child_reads],
        outputs=child_tensors,
        node_id=child_id,
    )

    output_rename = {output: parent_buffers[output] for output in remaining}

    def retarget(stmt):
        if isinstance(stmt, Write) and stmt.output in output_rename:
            return replace(stmt, output=output_rename[stmt.output])
        return stmt

    # Reconstruct so BodyOp re-seeds its I/O names from the retargeted writes; dataclass
    # ``replace`` would retain the pre-retarget output map and fail on the next matcher visit.
    parent_op = LoopOp(body=parent_op.body.map(retarget), name=parent_op.name)
    parent_tensors = tuple(replace(match.graph.buffer(output), name=parent_buffers[output]) for output in remaining)
    produced = (child_id, *(tensor.name for tensor in child_tensors[1:]))
    parent_inputs = [
        *(input_name for input_name in root.inputs if input_name in parent_reads),
        *(buffer for buffer in produced if buffer in parent_reads),
    ]
    parent_id = parent_buffers[remaining[0]]
    frag.add_node(parent_op, parent_inputs, outputs=parent_tensors, node_id=parent_id)

    output_map = {output: output_buffers[output] for output in old_outputs}
    frag.outputs = [output_map[output] for output in old_outputs]
    match.output = output_map
    for node_id in (child_id, parent_id):
        frag.nodes[node_id].op.knobs = consume_kernel_row(frag.nodes[node_id].op.knobs)
    parent = frag.nodes[parent_id].op
    parent.knobs = {**(parent.knobs or {}), spell_value_cut(site): _CUT}
    return frag


__all__ = ["ValueCut", "realize_value_cut", "route_value_cut", "spell_value_cut", "value_cut_sites"]
