"""Loop IR → Tile IR.

``010_lift`` peels the outer parallel axes and mechanically converts every remaining reduction loop
to a ``Fold``. ``020_twisted`` rewrites equivalent exp-family siblings into one twisted Fold,
``030_cut`` offers the maximal fused tree beside every closed stored Fold-edge cut,
``035_split_reduce`` offers the unsplit tree beside every cross-CTA reduce split the head fold
admits (both structural: a chosen cut or split replaces the kernel with fresh unmapped pieces that
re-enter this pass), and ``040_schedule`` schedules the stored tree.

``030_cut`` reads the structural tree through ``_tree``. The rebuilt ``040_schedule`` may reuse
that traversal mechanically, but schedule membership must remain independent of traversal order.
"""
