"""Algebra recognition and structural placement: ``LoopOp`` → resolved kernel set.

1. **Recognize** (``010_recognize``) — certify the maximal algebraic region and lift it to an
   unmapped ``TileOp``. Flash, online-softmax, and contraction recognition are algebra-only; no
   target capability or profitability signal enters this rule.
2. **Place** (``015_place``) — keep that maximal region or materialize one closed child seam. Cut
   pieces remain recognized ``TileOp`` trees and re-enter this pass until every kernel's placement
   is resolved.

The following ``lowering/schedule`` pass maps the fixed kernel set. The schedule implementation
remains in ``_schedule`` because it consumes the same Tile-IR predicates, but placement always
reaches a graph-level fixpoint before any schedule row is enumerated.
"""
