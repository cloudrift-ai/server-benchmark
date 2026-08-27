"""Loop IR → Tile IR.

``010_lift`` peels the outer parallel axes and mechanically converts every remaining reduction loop
to a ``Fold``. ``020_twisted`` rewrites equivalent exp-family siblings into one twisted Fold,
``030_cut`` offers the maximal fused tree beside every closed stored Fold-edge cut, and
``040_schedule`` schedules the stored tree.

``030_cut`` and ``040_schedule`` read the tree through the ONE walk in ``_tree``, and differ only in
what they take from it — the cut forks and the schedule forks respectively.
"""
