"""Sampling a candidate pool DURING enumeration — the size, the draw, and what must survive it.

Building the offline-prior dataset enumerates every golden's candidate pool and featurizes every
row. The corpus is millions of rows and tens of gigabytes, paid again on every experiment, and one
golden's pool alone exceeds the enumerator's materialization budget — so the fit does not finish.
The fix is to draw the sample from the schedule walk's leaf STREAM as it is produced — reservoir
sampling — rather than to build everything and throw most of it away: the walk stays lazy, each
candidate dict exists only for the moment it passes the reservoir, and nothing proportional to the
pool is ever retained. What is bounded is MEMORY, not time: the draw is one pass over every leaf,
so a fit still pays O(pool) walk time per golden — the 19.4M-row EXL3 coded-linear pool included —
where the deleted product space could address ``size`` members without visiting the rest. That
old guarantee is gone by design; the walk has no index to address by.

**The draw is a pure function of the stream and** ``(size, seed)``. The walk's leaf order is
deterministic and the reservoir never reads a row, so two byte-identical pools draw byte-identical
samples — which is what keeps the fit reproducible and keeps two goldens over one pool mergeable
into one training group.

**Membership survives the draw exactly.** ``keep`` is the set of :func:`~.features.tile_signature`
values that must be retained whatever the draw picks. The reservoir visits every candidate by
construction, and a row whose signature is in the set is retained beside the draw: a golden that is
genuinely absent from its pool still reads as absent, which is a real defect class (a pin or dtype
mismatch) the fit and ``eval golden`` both detect by exactly that miss.

**Reported rank is the RAW sample rank with the exact total beside it, never scaled**
(:class:`Candidates`). A sample's rank resolution floor is ``n / size``; pretending otherwise would
report a precision the draw does not have. The exact total is the reservoir's candidate count —
known the moment the stream ends, with no product space to pre-sum.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from emmy.compiler.pipeline.search import features
from emmy.compiler.structural import digest

#: Candidates drawn per pool when ``emmy fit`` samples. Measured against the alternative: at this
#: size the linear trainer's z-scored moments and its rank objective both track the full-pool fit,
#: while the corpus fits in memory and builds in minutes rather than an hour.
DEFAULT_SAMPLE = 2000


@dataclass(frozen=True)
class Candidates:
    """One graph's enumerated candidate rows beside the size of the pool they came from.

    ``total`` equals ``len(rows)`` when nothing was sampled, and is the exact pool size otherwise —
    the two travel together because a rank is only interpretable next to what it was ranked among.
    """

    rows: list[dict]
    total: int


@dataclass(frozen=True)
class PoolSample:
    """How many candidates a pool contributes, which of them must survive, and where each pool
    reports its exact size.

    Carried on :class:`~emmy.compiler.context.Context` and folded into the schedule pool's cache
    key. That key part is not decoration: ``dataclasses.replace`` SHARES a Context's session cache,
    so a sampled compile and a live one sit on one memo, and a flag that did not key the cache would
    serve a sampled pool to a live compile. ``None`` on the Context means live, and live never
    samples.
    """

    #: Candidates to draw. ``0`` (or a pool no larger than it) means the whole pool.
    rows: int
    seed: int = 0
    #: The ``tile_signature`` values the draw may not drop.
    keep: frozenset = frozenset()
    #: Where each drawn pool reports its EXACT size, keyed by that pool's cache key. The sampled
    #: rows cannot carry it and the fork tree has no channel for it, so the enumerator writes here
    #: and the caller that asked for the sample reads it back. Keyed rather than appended so a
    #: re-entered pool overwrites instead of double-counting. EXCLUDED from the value
    #: (``compare=False``): a sink is not part of a sample's identity, and the pool cache keys on
    #: that identity.
    totals: dict[str, int] = field(default_factory=dict, compare=False, repr=False)

    @property
    def key(self) -> str:
        """This sample's cache identity. The keep-set is SORTED into it: a ``frozenset``'s
        iteration order follows string hashing, which is randomized per process, so its ``repr``
        would key the same sample differently on two runs."""
        return digest(self.rows, self.seed, sorted(str(s) for s in self.keep))

    def take(self, rows: Iterable[dict]) -> Candidates:
        """Reservoir-sample the candidate stream ``rows``, returning the drawn rows in stream
        order beside the exact count behind them.

        One pass, O(size) retained: the first ``size`` candidates fill the reservoir and each later
        one displaces a uniformly chosen slot, so the draw is uniform without knowing the count up
        front — the property that lets a lazy walk be sampled at all. The kept-signature rows ride
        beside the reservoir (the keep-set ADDS to the draw, it never displaces it), and the whole
        stream is returned when ``size`` is 0 or the stream is no larger."""
        stream = iter(rows)
        if self.rows <= 0:
            out = list(stream)
            return Candidates(out, len(out))
        rng = np.random.default_rng(self.seed)
        reservoir: list[tuple[int, dict]] = []
        kept: dict[int, dict] = {}
        count = 0
        for index, row in enumerate(stream):
            count = index + 1
            if self.keep and features.tile_signature(row) in self.keep:
                kept[index] = row
            if index < self.rows:
                reservoir.append((index, row))
            else:
                slot = int(rng.integers(0, index + 1))
                if slot < self.rows:
                    reservoir[slot] = (index, row)
        chosen = dict(reservoir)
        chosen.update(kept)
        return Candidates([chosen[index] for index in sorted(chosen)], count)


__all__ = ["DEFAULT_SAMPLE", "Candidates", "PoolSample"]
