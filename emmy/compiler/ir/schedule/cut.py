"""The structural cut phase's generic schedule context."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, replace

from .base import Schedule, ScheduleContext, ScheduleRefused


@dataclass(frozen=True)
class CutScheduleContext[CutT](ScheduleContext[CutT, object, object]):
    """One independent structural choice domain in the ordered cut phase."""

    options: tuple[CutT, ...]
    _assignment: Schedule[CutT, object, object] = field(default_factory=lambda: Schedule(None, {}, {}), repr=False)

    @property
    def assignment(self) -> Schedule[CutT, object, object]:
        return self._assignment

    def extensions(self) -> Iterator[Schedule[CutT, object, object]]:
        if self.assignment.kernel is None:
            for option in self.options:
                yield Schedule(option, {}, {})

    def extend(self, pick: Schedule[CutT, object, object]) -> CutScheduleContext[CutT]:
        if (
            self.assignment.kernel is not None
            or pick.kernel is None
            or pick.nodes
            or pick.edges
            or not any(pick.kernel is option for option in self.options)
        ):
            raise ScheduleRefused("pick is outside the structural cut domain")
        return replace(self, _assignment=pick)


__all__ = ["CutScheduleContext"]
