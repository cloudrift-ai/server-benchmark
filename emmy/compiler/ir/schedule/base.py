"""Generic schedule assignments and compatible enumeration.

The interface deliberately exposes no domain catalog, site order, restriction object, or schedule
family. A context chooses the smallest useful frontier, and :func:`schedule` lazily composes that
frontier. For example, the classic context groups one node with its incident edges so it can reject
mixed transport and fragment-seam combinations before they create subtrees. Structural passes use
the first-class :meth:`ScheduleContext.only_cuts` entry point for a domain that has no assignment
composition.

Three invariants make those different granularities one enumeration:

* ``assignment`` is immutable and ``complete`` says whether it is a leaf. The assignment type is
  opaque to the driver: classic enumeration uses :class:`Schedule`, while structural cut
  enumeration uses the cut choice itself.
* ``extensions`` yields a lazy, context-aware frontier. It may omit picks already proved
  incompatible, but must retain a route to every accepted complete assignment.
* ``extend`` is the authority. It accepts a frontier pick or a complete assignment supplied by a
  caller, returns a new context, and raises :class:`ScheduleRefused` without mutating the prefix.

The generic driver knows only those operations. Repeatedly calling it on the returned contexts is
the lazy enumeration; no schedule-family visitor or product materialization exists beside it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
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


class ScheduleContext[AssignmentT](ABC):
    """One immutable prefix of a compatible enumeration.

    Implementations own frontier granularity, compatibility, restrictions, and validation. They
    may therefore emit one site at a time, a node together with related edges, or one complete
    schedule when the restriction already identifies it.
    """

    @property
    @abstractmethod
    def assignment(self) -> AssignmentT:
        """The immutable assignment prefix decided so far."""

    @property
    @abstractmethod
    def complete(self) -> bool:
        """Whether :attr:`assignment` is a complete leaf."""

    @abstractmethod
    def extensions(self) -> Iterator[AssignmentT]:
        """Yield the next lazy frontier without losing any accepted completion."""

    @abstractmethod
    def extend(self, pick: AssignmentT) -> Self:
        """Compose a partial or complete pick, or raise when it is incompatible."""

    @staticmethod
    def only_cuts[CutT](options: Iterable[CutT]) -> ScheduleContext[CutT | None]:
        """Return a context whose complete assignments are exactly the supplied structural cuts."""
        return _ChoiceContext(tuple(options))


@dataclass(frozen=True)
class _ChoiceContext[ChoiceT](ScheduleContext[ChoiceT | None]):
    """One independent choice factor, used by :meth:`ScheduleContext.only_cuts`."""

    options: tuple[ChoiceT, ...]
    _assignment: ChoiceT | None = None

    @property
    def assignment(self) -> ChoiceT | None:
        return self._assignment

    @property
    def complete(self) -> bool:
        return self.assignment is not None

    def extensions(self) -> Iterator[ChoiceT | None]:
        return iter(()) if self.complete else iter(self.options)

    def extend(self, pick: ChoiceT | None) -> _ChoiceContext[ChoiceT]:
        if self.complete or pick is None or not any(pick is option for option in self.options):
            raise ScheduleRefused("pick is outside the structural cut domain")
        return _ChoiceContext(self.options, pick)


def schedule[AssignmentT](
    context: ScheduleContext[AssignmentT],
    *,
    recursive: bool = True,
) -> Iterator[ScheduleContext[AssignmentT] | AssignmentT]:
    """Lazily enumerate complete assignments, or one frontier for a generic tree adapter."""
    for pick in context.extensions():
        try:
            child = context.extend(pick)
        except ScheduleRefused:
            continue
        if child.complete:
            yield child.assignment
        elif recursive:
            yield from schedule(child)
        else:
            yield child


__all__ = ["Schedule", "ScheduleContext", "ScheduleRefused", "schedule"]
