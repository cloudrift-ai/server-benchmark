"""The enumerated schedule pool as an addressable SPACE — one catalog, two traversals.

There are not two enumerations. There is ONE structure — the product of the site catalogs under a
chosen inventory — and two ways to read it: iterate every member, or address member *i*. Both go
through the same mixed-radix arithmetic and the same :func:`spell`, so they cannot drift apart.

The structure has three levels, each a plain rectangle:

- a :class:`Block` is one rectangle of a SITE's value space — everything the site decides except
  ``STAGE``, crossed with the ``STAGE`` slices legal for that assignment. The cooperative band
  rides ``REDUCE``, which a block fixes, so the row-level inventory validation runs once per block
  rather than once per candidate;
- a :class:`Segment` is one ``(WORK, view)`` slice: the partly-decided rows the view's site product
  left legal under that inventory, crossed with the view's ``RASTER`` values. A row spans
  ``width x len(rasters)`` CONTIGUOUS candidates, so a prefix sum is all it takes to turn a
  segment-local index back into a ``(row, stage, raster)`` triple;
- a :class:`PoolSpace` is the segments end to end, addressed by a second prefix sum.

**Nothing here knows a catalog, a legality rule or a schedule family.** A row is opaque: the space
reads ``knobs`` (the spelled families the row has already decided), ``stages`` (the ``{key:
spelling}`` stamps its one still-open ``STAGE`` axis offers, empty when it has none) and the
derived ``width``. That is what keeps this module free of any ``_schedule`` import, and with it any
import cycle — the walk builds the blocks, this addresses them.

**Why the size falls out for free.** Every rectangle's extent is known before a single candidate
dict exists, so :meth:`PoolSpace.__len__` is a prefix-sum lookup that builds no row. That is what
lets the row budget be checked before 400k dicts are built, and what makes an exact indexed sample
possible with no rejection loop.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from itertools import accumulate

#: The stamp a row with no open ``STAGE`` axis contributes — one candidate, nothing extra spelled.
_CLOSED: dict = {}


@dataclass(frozen=True)
class Block:
    """One rectangle of a site's value space: the ``{family -> slice}`` assignment every candidate
    in it shares, crossed with the ``STAGE`` slices legal for that assignment.

    ``stages`` is never empty — a site that decides no transport offers the single decided empty
    ``(None,)``, so the rectangle always has an extent and the block IS the site's unit of
    legality. A ``(TILE, REDUCE)`` pair whose every stage was refused is not a one-stage block, it
    is no block at all: the walk drops it."""

    values: dict
    stages: tuple


@dataclass(frozen=True)
class Segment:
    """One ``(WORK, view)`` slice of the space — the legal partly-decided rows, the kernel-global
    stamps that close them, and the prefix sums that address them.

    ``knobs`` is the segment's own stamp (its ``WORK`` spelling): kernel-global means the whole
    slice shares it. ``rasters`` is one stamp per launch order, the FASTEST axis of the space, so
    that a row's candidates stay contiguous."""

    rows: tuple
    knobs: dict
    rasters: tuple
    #: Candidate offsets, one per row plus the total — ``offsets[i]`` is where row ``i`` starts.
    offsets: tuple[int, ...]

    @classmethod
    def build(cls, rows, knobs: dict, rasters) -> Segment:
        rows, rasters = tuple(rows), tuple(rasters)
        return cls(rows, knobs, rasters, (0, *accumulate(r.width * len(rasters) for r in rows)))

    def __len__(self) -> int:
        return self.offsets[-1]


@dataclass(frozen=True)
class PoolSpace:
    """One term's whole candidate space — the fork's site keys, the decided-empty base every
    candidate is spelled over, and the segments end to end.

    ``base`` carries the union's keys at their decided empty plus the structural stamps that are a
    fact about the KERNEL rather than a site's choice, and it is already free of the keys no row
    decides — so a candidate is spelled ONCE, never built and rebuilt."""

    keys: tuple[str, ...]
    base: dict
    segments: tuple[Segment, ...]
    #: Candidate offsets, one per segment plus the total.
    offsets: tuple[int, ...]

    @classmethod
    def build(cls, keys, base: dict, segments) -> PoolSpace:
        segments = tuple(s for s in segments if len(s))
        return cls(tuple(keys), base, segments, (0, *accumulate(len(s) for s in segments)))

    def __len__(self) -> int:
        return self.offsets[-1]

    def __getitem__(self, i: int) -> dict:
        """Candidate ``i`` — two bisects and a divmod over the same radices :meth:`__iter__`
        loops, so addressing and iteration answer identically BY CONSTRUCTION."""
        n = len(self)
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError(f"candidate {i} is out of range for a {n}-candidate pool")
        s = bisect_right(self.offsets, i, hi=len(self.segments)) - 1
        seg = self.segments[s]
        at = i - self.offsets[s]
        r = bisect_right(seg.offsets, at, hi=len(seg.rows)) - 1
        stage, raster = divmod(at - seg.offsets[r], len(seg.rasters))
        return spell(self, seg, seg.rows[r], stage, raster)

    def __iter__(self):
        for seg in self.segments:
            for row in seg.rows:
                for stage in range(row.width):
                    for raster in range(len(seg.rasters)):
                        yield spell(self, seg, row, stage, raster)


def spell(space: PoolSpace, seg: Segment, row, stage: int, raster: int) -> dict:
    """THE candidate dict — the only place one is built, which is what makes "address member *i*"
    and "iterate every member" the same traversal rather than two that must be kept in step."""
    return {**space.base, **row.knobs, **(row.stages[stage] if row.stages else _CLOSED), **seg.knobs, **seg.rasters[raster]}


__all__ = ["Block", "PoolSpace", "Segment", "spell"]
