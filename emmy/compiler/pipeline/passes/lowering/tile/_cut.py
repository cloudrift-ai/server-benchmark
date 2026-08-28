"""Materialize a stored Fold edge as a workspace kernel boundary.

The cut is structural: the child Fold keeps its algebra, writes every state component to a
workspace, and the parent reads those components through ordinary ``Load`` edges.  Both pieces
are fresh unmapped ``TileOp`` objects and therefore re-enter the normal scheduling pipeline.
A contraction-operand seam whose cone passes through a storage waypoint cuts THERE instead
(:func:`storage_frontier`): the workspace holds the raw storage bits and the consumer keeps the
decode-plus-factors residue.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import islice

from emmy.compiler.dtype import F32
from emmy.compiler.dtype import get as get_dtype
from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.pure.fold import Fold, _operand_result_names, deep_defines, deep_reads, is_contraction, refs_axis
from emmy.compiler.ir.stmt import Assign, Body, Load, Write
from emmy.compiler.ir.tile import OutputSpec, Placement, TileOp
from emmy.compiler.ir.tile.ops import edge_dtypes
from emmy.compiler.ir.tile.path import family_sites, sites, spell
from emmy.compiler.pipeline import Match
from emmy.compiler.pipeline.knob import consume_kernel_row
from emmy.compiler.pipeline.passes.lowering.tile._tree import walk
from emmy.compiler.structural import digest
from emmy.compiler.tensor import Tensor


@dataclass(frozen=True)
class CutSite:
    """All stored occurrences of one canonically shared child Fold. ``dtypes`` is the workspace's
    per-component materialization, decided at offer time so the realization stores exactly what
    was offered. ``frontier`` (contraction-operand seams only) moves the cut to the cone's storage
    waypoint — see :class:`Frontier`."""

    node: Fold
    spelling: str
    axes: tuple
    dtypes: tuple
    frontier: Frontier | None = None


@dataclass(frozen=True)
class Frontier:
    """A contraction-operand cone's STORAGE waypoint: a decode (the ``ElementwiseImpl.decodes``
    trait) of a value the cone itself computes. The seam materializes there instead of at the
    cone's result — the workspace holds the raw storage bits (exact, the element the graph's own
    quantize produced), the producer piece computes ``producer`` (the encode prefix), and the
    consumer keeps ``residue`` (the decode plus the factor chain), which the normalize-time
    decode hoist then absorbs into a raw storage-dtype load with the factors on the accumulator
    epilogue — the same ``sum_k a*(s*w) = s*sum_k a*w`` reassociation as the materialized case."""

    name: str  # the encoded value the workspace holds
    producer: tuple  # the prefix stmts computing ``name`` (spliceable operand bodies inlined)
    residue: tuple  # the decode + factor stmts the consumer keeps
    dtype: object  # the storage DataType the decode op names


def _spliceable(edge) -> tuple | None:
    """A zero-axis operand's flat stmt list, or ``None`` when it cannot splice inline (an
    iterating fold, nested operands, or non-scalar members)."""
    if not isinstance(edge, Fold) or edge.axis is not None or edge.operands:
        return None
    members = tuple(edge.lift.body)
    return members if all(isinstance(stmt, (Load, Assign)) for stmt in members) else None


def storage_frontier(node: Fold) -> Frontier | None:
    """``node``'s storage frontier, or ``None`` when it has none the cut can separate.

    The shape is semantic, not an op list: exactly one decode of a value DEFINED by the cone's own
    body (a decode of a materialized load was already absorbed by normalization), whose backward
    cone separates cleanly — only the decode reads a prefix-computed name, so the residue's value
    is a pure function of the stored bits and its own leaves. Every operand must splice inline
    (each side takes the operand bodies it reads), keeping both pieces free of nested edges."""
    if not isinstance(node, Fold) or node.axis is not None or len(node.lift.results) != 1:
        return None
    body = node.lift.body
    if any(not isinstance(stmt, (Load, Assign)) for stmt in body):
        return None
    computed = {name for stmt in body if isinstance(stmt, Assign) for name in stmt.defines()}
    decodes = [
        stmt
        for stmt in body
        if isinstance(stmt, Assign) and stmt.op.decodes is not None and len(stmt.args) == 1 and stmt.args[0] in computed
    ]
    if len(decodes) != 1:
        return None
    decode = decodes[0]
    frontier = decode.args[0]
    spliced = [_spliceable(edge) for edge in node.operands]
    if any(members is None for members in spliced):
        return None
    prefix = tuple(body.backward_cone((frontier,)).members)
    prefix_ids = {id(stmt) for stmt in prefix}
    prefix_defs = {name for stmt in prefix for name in stmt.defines()}
    residue = tuple(stmt for stmt in body if id(stmt) not in prefix_ids)
    for stmt in residue:
        crossing = deep_reads([stmt]) & prefix_defs
        if crossing and (stmt is not decode or crossing != {frontier}):
            return None  # a residue stmt reads past the frontier — the waypoint does not separate
    if node.lift.results[0] in prefix_defs:
        return None
    result = get_dtype(decode.op.decodes)

    def side(stmts: tuple) -> tuple:
        reads = deep_reads(list(stmts))
        inlined: list = []
        for edge, members in zip(node.operands, spliced, strict=True):
            needed = set(_operand_result_names(edge)) & reads
            if needed:  # only the cone the side reads — a dead spliced def would decline the decode hoist
                inlined.extend(Body(members).backward_cone(tuple(sorted(needed))).members)
        return (*inlined, *stmts)

    return Frontier(name=frontier, producer=side(prefix), residue=side(residue), dtype=result)


def _closed_at(node: Fold, axes: tuple) -> bool:
    """Whether ``node`` has no capture other than axes available at its incoming edge."""
    lowered = tuple(node.lower())
    defined = set().union(*(deep_defines(stmt) for stmt in lowered)) if lowered else set()
    available = {axis.name for axis in axes}
    available.update(site.node.axis.name for site in sites(node) if isinstance(site.node, Fold) and site.node.axis is not None)
    return deep_reads(list(lowered)) <= defined | available


def _fed_store_dtype(tile: TileOp, consumer: Fold):
    """The dtype ``consumer`` stores its result at: the output its accumulators transitively feed
    (a forward closure over the root's lowered stmts covers any epilogue between the two), or
    ``None`` when the fed dtypes are not a singleton. A multi-output kernel can store siblings at
    other dtypes (w8a8's fp8 encode beside the f16 linear), so only the contraction's own stores
    speak for its slabs — and when it feeds outputs at SEVERAL dtypes no one of them does, so the
    seam stays undetermined and unoffered rather than resolved by list order."""
    if not tile.output_specs:  # the default store: the root's result to the kernel's one output
        tensor = next(iter(tile.outputs.values()), None)
        return None if tensor is None else tensor.dtype
    dependent = set(_operand_result_names(consumer))
    stmts = tuple(tile.op.lower())
    for _ in stmts:
        grown = False
        for stmt in stmts:
            defines = deep_defines(stmt)
            if not defines <= dependent and deep_reads([stmt]) & dependent:
                dependent |= defines
                grown = True
        if not grown:
            break
    fed = {
        tensor.dtype
        for store in tile.output_specs
        if store.write.value in dependent
        if (tensor := tile.outputs.get(store.write.output)) is not None
    }
    return fed.pop() if len(fed) == 1 else None


def _workspace_dtypes(node: Fold, tile: TileOp, consumer: Fold | None) -> tuple | None:
    """The cut workspace's per-component dtypes, or ``None`` when they cannot be determined.
    Reduction carrier precision is a Kernel IR policy — every Fold state is f32 until lowering
    stamps the concrete Accum/Init pair; a zero-axis value has no carrier and is inferred from its
    typed pure program instead. A seam standing in for a contraction OPERAND (``consumer`` is the
    consuming contraction) is the exception: it materializes explicitly at the dtype that
    contraction's output is stored at — the element the fused slab would have stored — never the
    carrier its cone computed in (only the ``a`` edge has a converting fill, so an f32 workspace on
    a ``b`` edge could feed no warp atom). A seam whose dtypes stay undetermined is not offered:
    the offer and the realization must agree, and a raise past the offer would kill the compile."""
    names = _operand_result_names(node)
    if consumer is not None:
        dtype = _fed_store_dtype(tile, consumer)
        return None if dtype is None else (dtype,) * len(names)
    dtypes = (F32,) * len(names) if node.axis is not None else edge_dtypes(node, tile.inputs)
    if len(dtypes) != len(names) or any(dtype is None for dtype in dtypes):
        return None
    return dtypes


def cuttable_seams(tile: TileOp) -> tuple[CutSite, ...]:
    """Every semantically closed stored Fold edge whose workspace dtypes are determined, grouped
    only by object sharing. A contraction's operand edges are seams too — cutting one materializes
    the cone feeding the operand into its own kernel and the contraction reads it back as an
    ordinary load — and they take the explicit contraction-operand dtype rule (`_workspace_dtypes`)."""
    all_sites = sites(tile.op)
    contraction_operands = {
        id(edge): site.node
        for site in all_sites
        if is_contraction(site.node)
        for edge in (site.node.a, *(channel.b for channel in site.node.channels))
        if isinstance(edge, Fold)
    }
    outer = (*tile.place.free, *(store.sweep for store in tile.output_specs if store.sweep is not None))
    occurrence_axes: dict[int, list[tuple]] = {}
    for node, available in islice(walk(tile.op, outer), 1, None):
        occurrence_axes.setdefault(id(node), []).append(available)
    out: list[CutSite] = []
    seen: set[int] = set()
    for site in family_sites("PLACE", all_sites):
        node = site.node
        scopes = occurrence_axes.get(id(node), ())
        if not isinstance(node, Fold) or id(node) in seen or not scopes or not all(_closed_at(node, scope) for scope in scopes):
            continue
        if node.observed:
            # An observed fold's per-step results exist only inside its stream — a cut would
            # separate the scan from its streamed boundary store, which no piece can then spell.
            continue
        consumer = contraction_operands.get(id(node))
        # A frontier REPLACES the fed-store realization at this seam rather than joining the
        # offer: the raw bits dominate the fed-store workspace on both precision (exact vs
        # re-rounded) and footprint (storage width vs store width), so there is no trade for the
        # evidence to decide — one site stays one decision.
        frontier = storage_frontier(node) if consumer is not None else None
        dtypes = (frontier.dtype,) if frontier is not None else _workspace_dtypes(node, tile, consumer)
        if dtypes is None:
            continue
        seen.add(id(node))
        axes = tuple({axis.name: axis for scope in scopes for axis in scope}.values())
        out.append(
            CutSite(node=node, spelling=spell(tile.op, "PLACE", node, all_sites=all_sites), axes=axes, dtypes=dtypes, frontier=frontier)
        )
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


def _workspace_axes(seam: CutSite, produced: Fold) -> tuple:
    """The seam axes the PRODUCED piece actually sweeps — its workspace dimensions. ``produced``
    is the seam node, or the frontier prefix when the seam materializes at a storage waypoint."""
    bound = {site.node.axis.name for site in sites(produced) if isinstance(site.node, Fold) and site.node.axis is not None}
    lowered = tuple(produced.lower())
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
    """Build the two-kernel fragment for ``seam``. A frontier seam cuts at the cone's storage
    waypoint: the producer computes the encode prefix, the workspace holds the raw bits, and the
    consumer keeps the decode + factor residue as its operand cone (which normalization then binds
    as a raw storage-dtype load with the factors hoisted onto the accumulator epilogue)."""
    tile: TileOp = root.op
    child = seam.node
    front = seam.frontier
    if front is not None:
        names = (front.name,)
        produced = Fold.projection(body=Body(front.producer), results=names)
    else:
        names = _operand_result_names(child)
        produced = child
    axes = _workspace_axes(seam, produced)
    shape = tuple(axis.extent for axis in axes)
    index = tuple(Var(axis.name) for axis in axes)
    token = digest(tile.structural_key(), seam.spelling)[:10]
    buffers = tuple(f"{root.id}__place_{token}_{i}" for i in range(len(names)))

    loads = tuple(Load(name=name, input=buffer, index=index) for name, buffer in zip(names, buffers, strict=True))
    if front is not None:
        raw = replace(loads[0], dtype=front.dtype)
        loads = (Fold.projection(body=Body((raw, *front.residue)), results=child.lift.results),)
    parent_fold = _replace_fold(tile.op, child, loads)

    producer = TileOp(
        op=produced,
        # The seam token keeps recursive pieces' kernel names distinct — the one-name-one-source
        # launch rule stated beside ``nvcc.load_cubin_function``: two same-named producers from
        # different cut levels would launch one kernel twice.
        name=f"{tile.name}__place_{token}",
        place=Placement(free=axes),
        output_specs=tuple(OutputSpec(Write(output=buffer, index=index, value=name)) for name, buffer in zip(names, buffers, strict=True)),
    )
    producer.knobs = consume_kernel_row(producer.knobs)
    parent_stores = tuple(
        replace(store, write=replace(store.write, output=renamed_outputs.get(store.write.output, store.write.output)))
        for store in tile.output_specs
    )
    consumer = TileOp(op=parent_fold, name=tile.name, place=tile.place, output_specs=parent_stores)
    consumer.knobs = consume_kernel_row(consumer.knobs)

    fragment = _input_fragment(match, root)
    workspace_tensors = tuple(Tensor(name=buffer, shape=shape, dtype=dtype) for buffer, dtype in zip(buffers, seam.dtypes, strict=True))
    fragment.add_node(
        op=producer,
        inputs=_piece_inputs(root, produced),
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


__all__ = ["CutSite", "Frontier", "cuttable_seams", "output_map", "realize", "storage_frontier"]
