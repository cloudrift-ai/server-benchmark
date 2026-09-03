"""Generic schedule assignments and compatible enumeration.

The interface deliberately exposes no domain catalog, site order, restriction object, or schedule
family. A context chooses the smallest useful frontier, and :func:`schedule` lazily composes that
frontier. For example, the classic context groups one node with its incident edges so it can reject
mixed transport and fragment-seam combinations before they create subtrees.

Three invariants make those different granularities one enumeration:

* ``assignment`` is an immutable kernel × node × edge :class:`Schedule`; a non-``None`` kernel
  marks a complete leaf.
* ``extensions`` yields a lazy, context-aware frontier. It may omit picks already proved
  incompatible, but must retain a route to every accepted complete assignment.
* ``extend`` is the authority. It accepts a frontier pick or a complete assignment supplied by a
  caller, returns a new context, and raises :class:`ScheduleRefused` without mutating the prefix.

The generic driver knows only those operations. Repeatedly calling it on the returned contexts is
the lazy enumeration; no schedule-family visitor or product materialization exists beside it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Self

from frozendict import frozendict

from .views import EdgeSite, NodeId


@dataclass(frozen=True)
class Schedule[KernelT, NodeT, EdgeT]:
    """One immutable kernel × node × edge assignment, possibly still incomplete."""

    kernel: KernelT | None
    nodes: Mapping[NodeId, NodeT]
    edges: Mapping[EdgeSite, EdgeT]

    def __post_init__(self) -> None:
        if not isinstance(self.nodes, Mapping) or not isinstance(self.edges, Mapping):
            raise TypeError("schedule node and edge assignments must be mappings")
        if any(type(site) is not int or site < 0 for site in self.nodes):
            raise TypeError("schedule node assignments must use non-negative integer sites")
        if any(
            not isinstance(edge, tuple)
            or len(edge) != 2
            or type(edge[0]) is not int
            or edge[0] < 0
            or type(edge[1]) is not int
            or edge[1] < 0
            for edge in self.edges
        ):
            raise TypeError("schedule edge assignments must use (consumer, operand) sites")
        object.__setattr__(self, "nodes", frozendict(self.nodes))
        object.__setattr__(self, "edges", frozendict(self.edges))


class ScheduleRefused(ValueError):
    """A pick cannot compose with the immutable schedule context."""


@dataclass(frozen=True)
class KernelPins:
    """Knob pins one kernel carries as its own state.

    ``values`` maps a knob key (``WORK``, ``REDUCE@<route>``, ``PLACE@<route>``, …) to the pinned
    spelling; ``source`` names the measured evidence row the deploy installed them from (a golden
    realization, or a tune-DB row), or ``None`` when a caller pinned the kernel by hand. The cut,
    split and schedule passes read these beside the ambient ``EMMY_<KNOB>`` pins, and every piece
    a cut or split mints inherits them minus the family that decision consumed — so one measured
    row reaches every kernel the route it spells creates."""

    values: frozendict = frozendict()
    source: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.values, frozendict):
            object.__setattr__(self, "values", frozendict(self.values))

    def __repr__(self) -> str:
        # A constructor repr over a plain dict: the graph JSON round trip re-evaluates it.
        return f"KernelPins({dict(self.values)!r}, {self.source!r})"

    def __bool__(self) -> bool:
        return bool(self.values)

    def without(self, *families: str) -> KernelPins:
        """The pins left for a piece once ``families`` were consumed by the decision that minted it."""
        kept = frozendict({key: value for key, value in self.values.items() if key.split("@", 1)[0] not in families})
        return KernelPins(kept, self.source)


class ScheduleContext[KernelT, NodeT, EdgeT](ABC):
    """One immutable prefix of a compatible enumeration.

    Implementations own frontier granularity, compatibility, restrictions, and validation. They
    may therefore emit one site at a time, a node together with related edges, or one complete
    schedule when the restriction already identifies it.
    """

    @property
    @abstractmethod
    def assignment(self) -> Schedule[KernelT, NodeT, EdgeT]:
        """The immutable kernel × node × edge assignment prefix decided so far."""

    @abstractmethod
    def extensions(self) -> Iterator[Schedule[KernelT, NodeT, EdgeT]]:
        """Yield the next lazy frontier without losing any accepted completion."""

    @abstractmethod
    def extend(self, pick: Schedule[KernelT, NodeT, EdgeT]) -> Self:
        """Compose a partial or complete pick, or raise when it is incompatible."""


def schedule[KernelT, NodeT, EdgeT](
    context: ScheduleContext[KernelT, NodeT, EdgeT],
    *,
    recursive: bool = True,
) -> Iterator[ScheduleContext[KernelT, NodeT, EdgeT] | Schedule[KernelT, NodeT, EdgeT]]:
    """Lazily enumerate complete assignments, or one frontier for a generic tree adapter."""
    for pick in context.extensions():
        try:
            child = context.extend(pick)
        except ScheduleRefused:
            continue
        if child.assignment.kernel is not None:
            yield child.assignment
        elif recursive:
            yield from schedule(child)
        else:
            yield child


__all__ = ["Schedule", "ScheduleContext", "ScheduleRefused", "schedule"]
