"""Sampling a candidate pool DURING enumeration — the size, the draw, and what must survive it.

Building the offline-prior dataset enumerates every golden's candidate pool and featurizes every
row. The corpus is millions of rows and tens of gigabytes, paid again on every experiment, and one
golden's pool alone exceeds the enumerator's materialization budget — so the fit does not finish.
The fix is to draw the sample where the pool is still a SPACE
(:mod:`~emmy.compiler.pipeline.passes.lowering.tile._pool`), before a candidate dict exists, rather
than to build everything and throw most of it away.

**The index set is a pure function of** ``(n, size, seed)``. It never looks at a row, so two
byte-identical pools draw byte-identical samples — which is what keeps the fit reproducible and
keeps two goldens over one pool mergeable into one training case.

**Membership survives the draw exactly.** ``keep`` is the set of :func:`~.features.tile_signature`
values that must be retained whatever the draw picks. With one, :meth:`PoolSample.take` VISITS every
candidate and retains the drawn ones plus every signature in the set: a golden that is genuinely
absent from its pool still reads as absent, which is a real defect class (a pin or dtype mismatch)
the fit and ``eval golden`` both detect by exactly that miss.

**Reported rank is the RAW sample rank with the exact total beside it, never scaled**
(:class:`Candidates`). A sample's rank resolution floor is ``n / size``; pretending otherwise would
report a precision the draw does not have.
"""

from __future__ import annotations

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

    def take(self, space) -> list[dict]:
        """The drawn candidates of ``space``, in index order.

        Two paths, and the difference is what ``keep`` costs: with no keep-set the draw addresses
        ``size`` members and builds nothing else; with one it iterates the space so membership stays
        exact, building and dropping one dict per candidate. That is the same order of work the
        full-pool scan it replaces already did, so it is not the thing to optimize first."""
        n = len(space)
        if self.rows <= 0 or n <= self.rows:
            return list(space)  # nothing to draw — the pool IS the sample
        picked = _indices(n, self.rows, self.seed)
        if not self.keep:
            return [space[i] for i in picked]
        want = set(picked.tolist())
        return [row for i, row in enumerate(space) if i in want or features.tile_signature(row) in self.keep]


def _indices(n: int, size: int, seed: int) -> np.ndarray:
    """``size`` distinct indices below ``n``, ascending — a PURE function of ``(n, size, seed)``.

    It never reads a candidate, which is the whole determinism argument: the draw cannot depend on
    anything a rebuild might reorder, so a refit of the same corpus is byte-identical."""
    return np.sort(np.random.default_rng(seed).choice(n, size=size, replace=False))


__all__ = ["DEFAULT_SAMPLE", "Candidates", "PoolSample"]
