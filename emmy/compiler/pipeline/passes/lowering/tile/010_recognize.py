"""Lift a ``LoopOp``'s loop nest to its UNMAPPED ``TileOp`` — the structural half of the
Loop-IR → Tile-IR boundary, rebuilt as ONE total algorithm (:func:`~._lift.recognized_tile`):
peel the free axes, lift every parseable reduce ``Loop`` to a typed :class:`Fold` in place,
split the boundary effects to :class:`Store`\\ s, then classify the tree (:mod:`._classify` —
online-softmax pairing, contraction binding, legalize). Nothing here dispatches on the algebra.

After the lift, PLACEMENT resolves — before any schedule fork exists, because a ``PLACE`` pin
must cut the recognized tree into a fragment of un-mapped ``LoopOp``\\ s (each piece
re-recognizing as a fresh root on the pass-scan restart, recursive). The fused
(:func:`~._classify.fused_view`) reading is the reference tree when it binds — its seams (the
``a`` cone edge) are the ones a ``PLACE`` key spells. UNPINNED, placement is an enumerated
STRUCTURAL fork: the fused form beside one cut fragment per legal seam and per closed fused value,
so tune DISCOVERS cuts and a deploy prices them like any kernel-set choice. A live schedule pin
may retain the placement siblings whose fully-spliced direct schedule rows realize it; this probe
never ranks, and retains the whole fork when no sibling realizes the pin. Each fragment's parent
piece is stamped with the exact ``PLACE`` key that names it; the splice then consumes the stamp
with everything else.

After this rule nothing downstream traffics in ``LoopOp``. Every ``LoopOp`` arrives already
carrying its ``S_*`` structural identity (the ``IdentityStrategy`` stamps fusion-born kernels at
the loop dialect's end and minted pieces at the splice event), so the lift never orders itself
against a stamp."""

from __future__ import annotations

from copy import deepcopy

from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import Match, Pattern
from emmy.compiler.pipeline.fork import Fork, flatten_leaves
from emmy.compiler.pipeline.knob import active_schedule_pins, family_of, pin_key_matches, values_equal
from emmy.compiler.pipeline.passes.lowering.tile._classify import fused_view
from emmy.compiler.pipeline.passes.lowering.tile._cut import cuttable_seams, realize_cut, route_cut
from emmy.compiler.pipeline.passes.lowering.tile._lift import recognized_tile
from emmy.compiler.pipeline.passes.lowering.tile._schedule import schedule
from emmy.compiler.pipeline.passes.lowering.tile._value_cut import realize_value_cut, route_value_cut, value_cut_sites
from emmy.compiler.pipeline.pipeline import RuleSkipped
from emmy.compiler.pipeline.search.slice import single_node_graph

PATTERN = [Pattern("root", LoopOp)]


def _detached_graph(graph: Graph) -> Graph:
    """Copy graph structure and ops without advancing its id allocator."""
    copied = Graph()
    copied.hints = deepcopy(graph.hints)
    for node_id in graph.topological_order():
        node = graph.nodes[node_id]
        copied.add_node(
            deepcopy(node.op),
            list(node.inputs),
            outputs=node.outputs,
            node_id=node_id,
        )
        copied.nodes[node_id].hints = deepcopy(node.hints)
    copied.inputs = list(graph.inputs)
    copied.outputs = list(graph.outputs)
    return copied


def _scheduled_rows(tile: TileOp, ctx) -> list[dict]:
    """Enumerate rows under the live pins without choosing or ranking one."""
    try:
        scheduled = schedule(tile, tile.name, tile.knobs, ctx)
    except ValueError:
        # This probe runs only for an unpinned placement fork. An explicit PLACE choice routes
        # before this function and reaches scheduling normally, preserving its refusal.
        return []
    options = scheduled if isinstance(scheduled, list) else [scheduled]
    leaves = flatten_leaves(options)
    return [dict(leaf.knobs if isinstance(leaf, Fork) else leaf.knobs or {}) for leaf in leaves]


def _pin_hits(option: TileOp | Graph, pins: dict[str, str], match: Match, ctx) -> set[str]:
    """Schedule pins at least one direct kernel row in ``option`` realizes."""
    tiles = []
    if isinstance(option, TileOp):
        tiles.append(option)
    else:
        # A fragment's primary buffer names are temporary until splice restores the replaced
        # kernel's identities. Probe that exact graph surgery on independent copies; otherwise
        # populate_io would reject a valid fragment whose body already spells the final output.
        graph = _detached_graph(single_node_graph(match.graph, match.root_node_id))
        receipt = graph.splice(
            _detached_graph(option),
            consumed=match.consumed,
            output=match.output or match.root_node_id,
        )
        for node_id in receipt.new_compute_ids:
            node = graph.nodes.get(node_id)
            if node is None or not isinstance(node.op, LoopOp):
                continue
            node.op.populate_io(graph, node)
            tile = recognized_tile(node.op, name=node.id)
            tile.outputs = dict(zip(node.buffer_names(), node.outputs, strict=True))
            tiles.append(tile)
    rows = [row for tile in tiles for row in _scheduled_rows(tile, ctx)]
    return {
        pinned
        for pinned, want in pins.items()
        if any(
            family_of(realized) == family_of(pinned) and pin_key_matches(pinned, realized) and values_equal(pinned, want, got)
            for row in rows
            for realized, got in row.items()
        )
    }


def _honor_schedule_pins(options: list[TileOp | Graph], match: Match, ctx) -> list[TileOp | Graph]:
    """Keep pin-realizing placement siblings when the fork contains such a sibling.

    A pin absent from every sibling belongs to another kernel, so it constrains nothing here. This
    is feasibility only: schedule rows are enumerated but never ranked.
    """
    pins = active_schedule_pins()
    if not pins:
        return options
    hits = [_pin_hits(option, pins, match, ctx) for option in options]
    covered = set().union(*hits)
    realizing = [option for option, got in zip(options, hits, strict=True) if got >= covered]
    return realizing or options


def rewrite(match: Match, root: Node, ctx=None) -> TileOp | Graph | list | None:
    loop: LoopOp = root.op
    map_tile = recognized_tile(loop, name=loop.name)
    # The matcher re-populates io when a later pass matches the op; seeding the output here makes
    # the UNMAPPED tile self-describing before any match has run (``deploy_identity`` folds the
    # output dtype).
    map_tile.outputs = dict(zip(root.buffer_names(), root.outputs, strict=True))
    # PLACEMENT — resolved FIRST, before any schedule fork exists. The fused (computed-A) view is
    # the reference tree when it binds; its seams are the ones a ``PLACE`` key spells.
    pro = fused_view(map_tile)
    route_tree, route_free, route_stores = (
        (pro[0], (*map_tile.place.free, *pro[1]), pro[2]) if pro is not None else (map_tile.op, map_tile.place.free, map_tile.stores)
    )
    verdict, seam = route_cut(route_tree, route_stores, route_free)
    if verdict == "cut":
        return realize_cut(match, root, route_tree, route_free, route_stores, seam)
    value_sites = value_cut_sites(loop)
    value_verdict, value_site = route_value_cut(loop, value_sites)
    if value_verdict == "cut":
        return realize_value_cut(match, root, value_site)
    if verdict == "fuse" or value_verdict == "fuse":
        return map_tile
    if verdict is None:
        seams = cuttable_seams(route_tree, route_stores, route_free)
        cut_options = []
        for s_ in seams:
            try:
                cut_options.append(realize_cut(match, root, route_tree, route_free, route_stores, s_))
            except RuleSkipped:
                continue  # the seam's workspace already exists — a piece of an applied cut
            except ValueError:
                # The realizer cannot BUILD this seam's fragment (a piece body that fails Loop IR
                # validation) — the enumeration drops it, exactly the unpinned half of the
                # ``legal.enforce`` convention; a PLACE pin naming the same seam still raises
                # loudly through the ``route_cut`` arm above.
                continue
        value_options = []
        for site in value_sites:
            try:
                value_options.append(realize_value_cut(match, root, site))
            except (RuleSkipped, ValueError):
                continue
        if cut_options or value_options:
            return _honor_schedule_pins([map_tile, *cut_options, *value_options], match, ctx)
    return map_tile
