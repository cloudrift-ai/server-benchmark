"""Materialize a stored Fold edge as a workspace kernel boundary.

The cut is structural: the child Fold keeps its algebra, writes every state component to a
workspace, and the parent reads those components through ordinary ``Load`` edges.  Both pieces
are fresh unmapped ``TileOp`` objects and therefore re-enter the normal scheduling pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from emmy.compiler.dtype import F32
from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.pure.fold import Fold, _operand_result_names, deep_defines, deep_reads, refs_axis
from emmy.compiler.ir.stmt import Body, Load, Write
from emmy.compiler.ir.tile import Placement, Store, TileOp
from emmy.compiler.ir.tile.ops import edge_dtypes
from emmy.compiler.ir.tile.path import family_sites, sites, spell
from emmy.compiler.pipeline import Match
from emmy.compiler.pipeline.knob import consume_kernel_row
from emmy.compiler.structural import digest
from emmy.compiler.tensor import Tensor


@dataclass(frozen=True)
class CutSite:
    """All stored occurrences of one canonically shared child Fold."""

    node: Fold
    spelling: str
    axes: tuple


def _child_folds(member):
    if isinstance(member, Fold):
        yield member
        return
    for body in member.nested():
        for child in body:
            yield from _child_folds(child)


def _occurrences(node: Fold, available: tuple):
    """Stored child occurrences and the axes in scope at each incoming edge."""
    inner = available + (() if node.axis is None else (node.axis,))
    children = [edge for edge in node.operands if isinstance(edge, Fold)]
    children.extend(child for stmt in node.lift.body for child in _child_folds(stmt))
    for child in children:
        yield child, inner
        yield from _occurrences(child, inner)


def _closed_at(node: Fold, axes: tuple) -> bool:
    """Whether ``node`` has no capture other than axes available at its incoming edge."""
    lowered = tuple(node.lower())
    defined = set().union(*(deep_defines(stmt) for stmt in lowered)) if lowered else set()
    available = {axis.name for axis in axes}
    available.update(site.node.axis.name for site in sites(node) if isinstance(site.node, Fold) and site.node.axis is not None)
    return deep_reads(list(lowered)) <= defined | available


def cuttable_seams(tile: TileOp) -> tuple[CutSite, ...]:
    """Every semantically closed stored Fold edge, grouped only by object sharing."""
    all_sites = sites(tile.op)
    outer = (*tile.place.free, *(store.sweep for store in tile.stores if store.sweep is not None))
    occurrence_axes: dict[int, list[tuple]] = {}
    for node, available in _occurrences(tile.op, outer):
        occurrence_axes.setdefault(id(node), []).append(available)
    out: list[CutSite] = []
    seen: set[int] = set()
    for site in family_sites("PLACE", all_sites):
        node = site.node
        scopes = occurrence_axes.get(id(node), ())
        if not isinstance(node, Fold) or id(node) in seen or not scopes or not all(_closed_at(node, scope) for scope in scopes):
            continue
        seen.add(id(node))
        axes = tuple({axis.name: axis for scope in scopes for axis in scope}.values())
        out.append(CutSite(node=node, spelling=spell(tile.op, "PLACE", node, all_sites=all_sites), axes=axes))
    return tuple(out)


def _replace_member(member, target: Fold, loads: tuple[Load, ...]):
    if member is target:
        return loads
    if isinstance(member, Fold):
        return (_replace_fold(member, target, loads),)
    nested = member.nested()
    if not nested:
        return (member,)
    bodies = []
    for body in nested:
        replaced = tuple(piece for child in body for piece in _replace_member(child, target, loads))
        bodies.append(Body(replaced))
    return (member.with_bodies(tuple(bodies)),)


def _replace_fold(node: Fold, target: Fold, loads: tuple[Load, ...]) -> Fold:
    operands = tuple(piece for edge in node.operands for piece in _replace_member(edge, target, loads))
    body = tuple(piece for stmt in node.lift.body for piece in _replace_member(stmt, target, loads))
    return replace(node, operands=operands, lift=replace(node.lift, body=Body(body)))


def _workspace_axes(seam: CutSite) -> tuple:
    child = seam.node
    bound = {site.node.axis.name for site in sites(child) if isinstance(site.node, Fold) and site.node.axis is not None}
    lowered = tuple(child.lower())
    return tuple(axis for axis in seam.axes if axis.name not in bound and any(refs_axis(stmt, axis.name) for stmt in lowered))


def _piece_inputs(root: Node, fold: Fold, first: tuple[str, ...] = ()) -> list[str]:
    reads = {load.input for load in Body.coerce(fold.lower()).loads}
    return [*first, *(name for name in root.inputs if name in reads)]


def _input_fragment(match: Match, root: Node) -> Graph:
    fragment = Graph()
    for name in root.inputs:
        fragment.add_node(op=InputOp(), inputs=[], output=match.graph.buffer(name), node_id=name)
    return fragment


def output_map(root: Node) -> dict[str, str]:
    """Stable temporary output names used by every cut sibling of ``root``."""
    return {name: f"{name}__placed" for name in root.buffer_names()}


def realize(match: Match, root: Node, seam: CutSite, renamed_outputs: dict[str, str]) -> Graph:
    """Build the two-kernel fragment for ``seam``."""
    tile: TileOp = root.op
    child = seam.node
    names = _operand_result_names(child)
    axes = _workspace_axes(seam)
    shape = tuple(axis.extent for axis in axes)
    index = tuple(Var(axis.name) for axis in axes)
    token = digest(tile.structural_key(), seam.spelling)[:10]
    buffers = tuple(f"{root.id}__place_{token}_{i}" for i in range(len(names)))

    inferred = (F32,) * len(names) if child.axis is not None else edge_dtypes(child, tile.inputs)
    if len(inferred) != len(names):
        inferred = (F32,) * len(names)
    dtypes = tuple(dtype or F32 for dtype in inferred)
    loads = tuple(Load(name=name, input=buffer, index=index) for name, buffer in zip(names, buffers, strict=True))
    parent_fold = _replace_fold(tile.op, child, loads)

    producer = TileOp(
        op=child,
        name=f"{tile.name}__place_producer",
        place=Placement(free=axes),
        stores=tuple(Store(Write(output=buffer, index=index, value=name)) for name, buffer in zip(names, buffers, strict=True)),
    )
    producer.knobs = consume_kernel_row(producer.knobs)
    parent_stores = tuple(
        replace(store, write=replace(store.write, output=renamed_outputs.get(store.write.output, store.write.output)))
        for store in tile.stores
    )
    consumer = TileOp(op=parent_fold, name=tile.name, place=tile.place, stores=parent_stores)
    consumer.knobs = consume_kernel_row(consumer.knobs)

    fragment = _input_fragment(match, root)
    workspace_tensors = tuple(Tensor(name=buffer, shape=shape, dtype=dtype) for buffer, dtype in zip(buffers, dtypes, strict=True))
    fragment.add_node(
        op=producer,
        inputs=_piece_inputs(root, child),
        outputs=workspace_tensors,
        node_id=buffers[0],
    )
    output_tensors = (
        replace(root.outputs[0], name=root.outputs[0].name),
        *(replace(tensor, name=renamed_outputs[name]) for name, tensor in zip(root.buffer_names()[1:], root.outputs[1:], strict=True)),
    )
    fragment.add_node(
        op=consumer,
        inputs=_piece_inputs(root, parent_fold, buffers),
        outputs=output_tensors,
        node_id=renamed_outputs[root.id],
    )
    fragment.outputs.extend(renamed_outputs.values())
    return fragment


__all__ = ["CutSite", "cuttable_seams", "output_map", "realize"]
