"""Loop IR → Tile IR.

``010_lift`` peels the outer parallel axes and mechanically converts every remaining reduction loop
to a ``Fold``. ``020_twisted`` fuses every reduce that reads a reduce into the twisted carrier a
recipe recognizes (``ir/pure/twist.py``) and offers it beside the two-pass tree the lift reconstructs from that
carrier's own loop (``_twist.relift`` through ``_untwist``) — the ``TWIST`` fork. The single
``030_cut`` pass runs to a fixpoint: it first offers the maximal fused tree beside every
closed stored Fold-edge cut, then the unsplit tree beside every cross-CTA reduce split the head Fold
admits. A structural choice replaces the kernel with fresh unmapped pieces that re-enter this pass.
``040_schedule`` schedules the stored tree only after the cut rule is quiescent.

``030_cut`` reads the structural tree through ``ir.tile.path.sites``. ``040_schedule`` adapts the classic
schedule model's accepted assignments to lazy pipeline forks; schedule membership remains independent of traversal order.
"""
