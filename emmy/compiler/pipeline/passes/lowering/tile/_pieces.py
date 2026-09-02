"""Fresh output-owning Tile pieces shared by structural kernel-set rewrites."""

from dataclasses import replace

from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.pure.fold import Fold, _operand_result_names, loaded_buffers
from emmy.compiler.ir.schedule import Placement
from emmy.compiler.ir.stmt import Assign, Body, Init, Load, Select
from emmy.compiler.ir.tile import OutputSpec, ProjectionRegion, TileOp
from emmy.compiler.ir.tile.ir import projection_results
from emmy.compiler.ir.tile.ops import carries_partition
from emmy.compiler.pipeline import Match
from emmy.compiler.pipeline.knob import consume_kernel_row


def input_fragment(match: Match, root: Node) -> Graph:
    """A replacement fragment seeded with the replaced node's inputs."""
    fragment = Graph()
    for name in root.inputs:
        fragment.add_node(InputOp(), [], match.graph.buffer(name), node_id=name)
    return fragment


def piece_inputs(root: Node, body, *first: str) -> list[str]:
    """Fragment buffers followed by external inputs read by one piece."""
    if isinstance(body, TileOp):
        reads = loaded_buffers(body.op)
    elif isinstance(body, Fold):
        reads = loaded_buffers(body)
    else:
        reads = {load.input for load in Body.coerce(body).loads}
    return [*first, *(name for name in root.inputs if name in reads)]


def output_root(root: Node, outputs: set[str]) -> Node:
    """A graph-node view containing only the ports one output piece owns."""
    by_name = dict(zip(root.buffer_names(), root.outputs, strict=True))
    ordered = tuple(name for name in root.buffer_names() if name in outputs)
    if outputs != set(ordered):
        raise ValueError(f"projection stores target unknown output buffers: {sorted(outputs - set(ordered))}")
    return replace(root, id=ordered[0], outputs=tuple(replace(by_name[name], name=name) for name in ordered))


def tile_piece(body, free, *, output_specs=()) -> TileOp:
    """One fresh unscheduled Tile kernel preserving its Fold algebra."""
    op = body if isinstance(body, Fold) else Fold.projection(body=Body.coerce(body))
    piece = TileOp(op=op, place=Placement(free=tuple(free)), output_specs=tuple(output_specs))
    return replace(piece, knobs=consume_kernel_row(piece.knobs))


def _peel_region(region: ProjectionRegion) -> tuple[tuple, Body]:
    """Lift one region's leading parallel chain into its piece's placement."""
    axes = [region.axis]
    prefix = []
    current = list(region.body)
    while True:
        index = 0
        while index < len(current) and isinstance(current[index], (Load, Assign, Init, Select)):
            index += 1
        head, rest = current[:index], current[index:]
        if len(rest) != 1 or not isinstance(rest[0], ProjectionRegion):
            return tuple(axes), Body((*prefix, *current))
        prefix.extend(head)
        axes.append(rest[0].axis)
        current = list(rest[0].body)


def projection_region_pieces(tile: TileOp) -> tuple[tuple[Body, tuple, tuple[OutputSpec, ...]], ...]:
    """Closed pieces for a root prefix followed by one or more output regions.

    One TileOp has one root-global placement. This offer therefore separates sibling regions so
    each can lift its own leading axes into an ordinary placement. A single region can remain after
    earlier cuts; it uses the same offer because lifting its axes still trades shared-prefix reuse
    for parallel placement. It declines unless the body is exactly a pure prefix followed by
    regions, every output has one owner, and every capture closes over the prefix and root operands.
    """
    root = tile.op
    if not isinstance(root, Fold) or root.axis is not None:
        return ()
    first = next((index for index, member in enumerate(root.body) if isinstance(member, ProjectionRegion)), len(root.body))
    prefix, regions = root.body[:first], root.body[first:]
    if not regions or any(not isinstance(member, ProjectionRegion) for member in regions):
        return ()

    grouped = [[] for _ in regions]
    result_sets = tuple(projection_results((region,)) for region in regions)
    for spec in tile.output_specs:
        owners = [index for index, results in enumerate(result_sets) if set(spec.write.values) <= results]
        if len(owners) != 1:
            return ()
        grouped[owners[0]].append(spec)
    if any(not stores for stores in grouped):
        return ()
    output_groups = [{spec.write.output for spec in stores} for stores in grouped]
    if any(left & right for index, left in enumerate(output_groups) for right in output_groups[index + 1 :]):
        return ()
    if tile.outputs and set().union(*output_groups) != set(tile.outputs):
        return ()

    outer_axes = {axis.name for axis in tile.place.free}
    operand_names = {name for edge in root.operands for name in _operand_result_names(edge)}
    pieces = []
    for region, stores in zip(regions, grouped, strict=True):
        provider = prefix.backward_cone(region.deps())
        needed_operands = provider.external_reads - outer_axes
        if needed_operands - operand_names:
            return ()
        operands = tuple(edge for edge in root.operands if set(_operand_result_names(edge)) & needed_operands)
        axes, body = _peel_region(region)
        free = (*tile.place.free, *axes)
        if len({axis.name for axis in free}) != len(free):
            return ()
        pieces.append((Body((*operands, *provider.members, *body)), free, tuple(stores)))
    return tuple(pieces)


def add_output_piece(match: Match, fragment: Graph, root: Node, piece: TileOp, inputs: list[str]) -> Graph:
    """Add one piece and extend the graph-splice output mapping with its owned ports."""
    buffers = root.buffer_names()
    renamed = {name: f"{name}__split" for name in buffers}
    piece = replace(
        piece,
        output_specs=tuple(
            replace(spec, write=replace(spec.write, output=renamed.get(spec.write.output, spec.write.output)))
            for spec in piece.output_specs
        ),
    )
    tensors = (
        replace(root.outputs[0], name=buffers[0]),
        *(replace(tensor, name=renamed[name]) for name, tensor in zip(buffers[1:], root.outputs[1:], strict=True)),
    )
    fragment.add_node(piece, inputs, outputs=tensors, node_id=renamed[buffers[0]])
    fragment.outputs.extend(renamed.values())
    output = dict(match.output) if isinstance(match.output, dict) else {}
    output.update(renamed)
    match.output = output
    return fragment


def realize_projection_regions(match: Match, root: Node, pieces) -> Graph:
    """Replace one output-region root with fresh output-owning TileOps."""
    tile: TileOp = root.op
    fragment = input_fragment(match, root)
    receipt = tile.split_consumed or carries_partition(tile.op)
    for body, free, stores in pieces:
        owned = output_root(root, {store.write.output for store in stores})
        piece = replace(tile_piece(body, free, output_specs=stores), split_consumed=receipt)
        add_output_piece(match, fragment, owned, piece, piece_inputs(owned, piece))
    return fragment


__all__ = [
    "add_output_piece",
    "input_fragment",
    "output_root",
    "piece_inputs",
    "projection_region_pieces",
    "realize_projection_regions",
    "tile_piece",
]
