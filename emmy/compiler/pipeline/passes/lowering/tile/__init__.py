"""Loop IR → Tile IR.

``010_lift`` peels the outer parallel axes and mechanically converts every remaining
reduction loop to a ``Fold``. ``020_schedule`` schedules the stored tree, and
``030_split_reduce`` realizes cross-CTA reduction plans.
"""
