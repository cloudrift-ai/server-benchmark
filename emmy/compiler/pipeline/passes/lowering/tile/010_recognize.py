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
STRUCTURAL fork: the fused form beside one cut fragment per legal seam, so tune DISCOVERS cuts
and a deploy prices them like any kernel-set choice. Each fragment's parent piece is stamped
``PLACE@<seam>: cut`` so a recorded routing golden can match the option by the seam it names;
the splice then consumes the stamp with everything else.

After this rule nothing downstream traffics in ``LoopOp``. Every ``LoopOp`` arrives already
carrying its ``S_*`` structural identity (the ``IdentityStrategy`` stamps fusion-born kernels at
the loop dialect's end and minted pieces at the splice event), so the lift never orders itself
against a stamp."""

from __future__ import annotations

from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import Match, Pattern
from emmy.compiler.pipeline.passes.lowering.tile._classify import fused_view
from emmy.compiler.pipeline.passes.lowering.tile._cut import cuttable_seams, realize_cut, route_cut
from emmy.compiler.pipeline.passes.lowering.tile._lift import recognized_tile
from emmy.compiler.pipeline.pipeline import RuleSkipped

PATTERN = [Pattern("root", LoopOp)]


def rewrite(match: Match, root: Node, ctx=None) -> TileOp | Graph | list | None:
    loop: LoopOp = root.op
    map_tile = recognized_tile(loop, name=loop.name)
    # The matcher re-populates io when a later pass matches the op; seeding the output here makes
    # the UNMAPPED tile self-describing before any match has run (``deploy_identity`` folds the
    # output dtype).
    map_tile.outputs = {root.output.name: root.output}
    # PLACEMENT — resolved FIRST, before any schedule fork exists. The fused (computed-A) view is
    # the reference tree when it binds; its seams are the ones a ``PLACE`` key spells.
    pro = fused_view(map_tile)
    route_tree, route_free, route_stores = (
        (pro[0], (*map_tile.place.free, pro[1]), pro[2]) if pro is not None else (map_tile.op, map_tile.place.free, map_tile.stores)
    )
    verdict, seam = route_cut(ctx, dict(loop.knobs or {}), route_tree, route_stores, route_free)
    if verdict == "cut":
        return realize_cut(match, root, route_tree, route_free, route_stores, seam)
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
        if cut_options:
            return [map_tile, *cut_options]
    return map_tile
