"""Loop IR → Tile IR.

``010_lift`` peels the outer parallel axes and mechanically converts every remaining
reduction loop to a ``Fold``. ``015_twisted`` rewrites equivalent exp-family siblings into one
twisted Fold, ``020_schedule`` schedules the stored tree, and ``030_split_reduce`` realizes
cross-CTA reduction plans.
"""
