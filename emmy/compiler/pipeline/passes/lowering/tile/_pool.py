"""The enumerated schedule pool as an addressable SPACE — one catalog, two traversals.

There are not two enumerations. There is one addressable product of the site catalogs under a
chosen inventory, read either by iteration or by index. Candidate dictionaries are created only at
that read boundary.

The structure has three levels, each a plain rectangle:

- a :class:`Block` is one rectangle of a site's value space;
- a :class:`Segment` is one ``WORK`` slice: its recursive site-row sequence crossed with the
  kernel's ``RASTER`` values;
- a :class:`PoolSpace` is the segments end to end, addressed by a second prefix sum.

**Nothing here knows a catalog, a legality rule or a schedule family.** A row is opaque: the space
reads ``knobs`` (the spelled families the row has already decided), ``stages`` (the ``{key:
spelling}`` stamps its one still-open ``STAGE`` axis offers, empty when it has none) and the
derived ``width``. That is what keeps this module free of any ``_schedule`` import, and with it any
import cycle — the walk builds the blocks, this addresses them.

Every rectangle's extent is known before a candidate dict exists, so :meth:`PoolSpace.__len__` is a
prefix-sum lookup and an indexed sample needs no rejection loop.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate
from types import MappingProxyType

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
    stamps that close them.

    ``knobs`` is the segment's own stamp (its ``WORK`` spelling): kernel-global means the whole
    slice shares it. ``rasters`` is one stamp per launch order, the FASTEST axis of the space, so
    that a row's candidates stay contiguous."""

    rows: Sequence
    knobs: dict
    rasters: tuple
    offsets: tuple[int, ...] | None = None

    @classmethod
    def build(cls, rows, knobs: dict, rasters) -> Segment:
        offsets = None if getattr(rows, "closed", False) else (0, *accumulate(row.width for row in rows))
        return cls(rows, knobs, tuple(rasters), offsets)

    def __len__(self) -> int:
        rows = len(self.rows) if self.offsets is None else self.offsets[-1]
        return rows * len(self.rasters)


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
        logical, raster = divmod(at, len(seg.rasters))
        if seg.offsets is None:
            row, stage = logical, None
        else:
            row = bisect_right(seg.offsets, logical, hi=len(seg.rows)) - 1
            stage = logical - seg.offsets[row] if seg.rows[row].stages else None
        return spell(self, seg, seg.rows[row], raster, stage)

    def __iter__(self):
        for seg in self.segments:
            if seg.offsets is None:
                for row in seg.rows:
                    for raster in range(len(seg.rasters)):
                        yield spell(self, seg, row, raster)
                continue
            for row in seg.rows:
                for stage in range(row.width):
                    for raster in range(len(seg.rasters)):
                        yield spell(self, seg, row, raster, stage if row.stages else None)

    def partition(self, key: str):
        """Partition this space by one schedule key without visiting candidate rows."""
        if any(key in segment.knobs for segment in self.segments):
            grouped: dict[str, list[Segment]] = {}
            for segment in self.segments:
                grouped.setdefault(segment.knobs.get(key, ""), []).append(segment)
            return tuple((value, PoolSpace.build(self.keys, self.base, segments)) for value, segments in grouped.items())

        if any(key in raster for segment in self.segments for raster in segment.rasters):
            grouped: dict[str, list[Segment]] = {}
            for segment in self.segments:
                by_value: dict[str, list[dict]] = {}
                for raster in segment.rasters:
                    by_value.setdefault(raster.get(key, ""), []).append(raster)
                for value, rasters in by_value.items():
                    grouped.setdefault(value, []).append(Segment.build(segment.rows, segment.knobs, rasters))
            return tuple((value, PoolSpace.build(self.keys, self.base, segments)) for value, segments in grouped.items())

        grouped: dict[str, list[Segment]] = {}
        for segment in self.segments:
            structural = getattr(segment.rows, "partition", None)
            if structural is None:
                raise ValueError("a live schedule PoolSpace must carry structurally partitionable rows")
            for value, rows in structural(key):
                grouped.setdefault(value, []).append(Segment.build(rows, segment.knobs, segment.rasters))
        return tuple((value, PoolSpace.build(self.keys, self.base, segments)) for value, segments in grouped.items())


def spell(space: PoolSpace, seg: Segment, row, raster: int, stage: int | None = None):
    """THE candidate dict — the only place one is built, which is what makes "address member *i*"
    and "iterate every member" the same traversal rather than two that must be kept in step."""
    return MappingProxyType(
        {
            **space.base,
            **row.knobs,
            **(row.stages[stage] if stage is not None else _CLOSED),
            **seg.knobs,
            **seg.rasters[raster],
        }
    )


__all__ = ["Block", "PoolSpace", "Segment", "spell"]
