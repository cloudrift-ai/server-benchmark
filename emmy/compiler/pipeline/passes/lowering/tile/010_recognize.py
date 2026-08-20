"""Lift a ``LoopOp``'s loop nest to its UNMAPPED ``TileOp`` — the structural half of the
Loop-IR → Tile-IR boundary, rebuilt as ONE total algorithm (:func:`~._lift.recognized_tile`):
peel the free axes, lift every parseable reduce ``Loop`` to a typed :class:`Fold` in place,
split the boundary effects to :class:`Store`\\ s. Nothing here dispatches on the algebra —
recognition (contraction binding, online-softmax pairing, the monoid-producer composition) is
CLASSIFICATION of the lifted tree and runs downstream, stated on ``Fold`` fields.

After this rule nothing downstream traffics in ``LoopOp``. Every ``LoopOp`` arrives already
carrying its ``S_*`` structural identity (the ``IdentityStrategy`` stamps fusion-born kernels at
the loop dialect's end and minted pieces at the splice event), so the lift never orders itself
against a stamp."""

from __future__ import annotations

from emmy.compiler.graph import Node
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import Match, Pattern
from emmy.compiler.pipeline.passes.lowering.tile._lift import recognized_tile

PATTERN = [Pattern("root", LoopOp)]


def rewrite(match: Match, root: Node, ctx=None) -> TileOp:
    loop: LoopOp = root.op
    map_tile = recognized_tile(loop, root.output.name, name=loop.name)
    # The matcher re-populates io when a later pass matches the op; seeding the output here makes
    # the UNMAPPED tile self-describing before any match has run (``deploy_identity`` folds the
    # output dtype).
    map_tile.outputs = {root.output.name: root.output}
    return map_tile
