"""Loop IR → Tile IR.

``010_lift`` peels the outer parallel axes and mechanically converts every remaining reduction loop
to a ``Fold``. ``020_twisted`` rewrites equivalent exp-family siblings into one twisted Fold. The
single ``030_cut`` pass runs to a fixpoint: it first offers the maximal fused tree beside every
closed stored Fold-edge cut, then the unsplit tree beside every cross-CTA reduce split the head Fold
admits. A structural choice replaces the kernel with fresh unmapped pieces. Before the splice stamps
their identities, each piece reuses Loop's split-free-axis canonicalization because the cut may have
removed the access that kept a reshape's output axes distinct; the canonical pieces then re-enter this
pass. ``040_schedule`` schedules the stored tree only after the cut rule is quiescent.

``030_cut`` reads the structural tree through ``ir.pure.tree``. ``040_schedule`` adapts the classic
schedule model's accepted assignments to lazy pipeline forks; schedule membership remains independent of traversal
order.
"""
