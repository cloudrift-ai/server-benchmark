"""Give a twisted carrier's channels a semiring to live in, before scheduling sees the tree."""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.lowering.tile._block import block_tree

PATTERN = [Pattern("root", TileOp)]


def rewrite(match: Match, root: Node, ctx=None) -> Graph:
    """Block every twisted carrier, then hand the kernel BACK to the loop dialect.

    The blocked tree is SPLICED as a ``LoopOp`` rather than rebound in place, for two effects
    that only the loop dialect has. ``LoopOp.__post_init__`` runs ``normalize_body``, whose
    sibling-reduce merge is the only thing that collapses the per-channel block loops into one
    (a ``TileOp`` normalizes its TERM instead, and the materializer builds straight from
    ``Fold.lower``, so the merge never reaches the emitted kernel). And a ``Graph`` splice is
    what restarts this pass's rule scan (``Cursor.advance``), so ``010_lift`` re-derives the
    Fold tree from the merged nest.
    """
    del ctx
    tile: TileOp = root.op
    if tile.op is None or tile.place.is_mapped or tile.schedule is not None:
        raise RuleSkipped("TileOp already scheduled / nothing to rewrite")
    blocked = block_tree(tile.op, tile.axes)
    if blocked is None:
        raise RuleSkipped("no blockable twisted carrier")
    op, axes = blocked
    loop = LoopOp(body=replace(tile, op=op, axes=axes).loop_body, name=tile.name)

    fragment = Graph()
    for name in root.inputs:
        fragment.add_node(op=InputOp(), inputs=[], output=match.graph.buffer(name), node_id=name)
    renamed = {name: f"{name}__blocked" for name in root.buffer_names()}
    match.output = renamed
    fragment.add_node(
        op=loop,
        inputs=list(root.inputs),
        outputs=[replace(tensor, name=renamed[name]) for name, tensor in zip(root.buffer_names(), root.outputs, strict=True)],
        node_id=renamed[root.id],
    )
    fragment.outputs.extend(renamed.values())
    return fragment
