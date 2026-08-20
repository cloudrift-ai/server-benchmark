"""Search strategies — the search SHAPES over the engine's loop (see :mod:`.base`).

Mirrors ``search/policy/``: ``policy`` answers the questions ONE loop asks (frontier ranking,
terminal valuation); ``strategy`` composes loops (how many, over which passes, with which policy
inside, and what the results mean together).
"""

from emmy.compiler.pipeline.search.strategy.base import SearchStrategy
from emmy.compiler.pipeline.search.strategy.greedy import GreedyStrategy
from emmy.compiler.pipeline.search.strategy.two_level import TwoLevelResult, TwoLevelStrategy

__all__ = ["GreedyStrategy", "SearchStrategy", "TwoLevelResult", "TwoLevelStrategy"]
