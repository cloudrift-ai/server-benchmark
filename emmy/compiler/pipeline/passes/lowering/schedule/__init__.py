"""Schedule a resolved kernel set and realize graph-level schedule decisions.

``020_schedule`` maps each recognized ``TileOp`` independently and enumerates its typed schedule
slices. ``030_split_reduce`` then realizes a selected cross-CTA reduction as partial/finalize graph
nodes. Structural placement has already reached a graph-level fixpoint in ``lowering/tile``.
"""
