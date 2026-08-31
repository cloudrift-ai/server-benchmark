"""Interfaces shared by schedule implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Self


class Schedule(ABC):
    """An immutable hardware-execution plan.

    Implementations expose replacement rather than in-place mutation so a schedule can remain a
    stable value while it is enumerated, validated, serialized, and measured.
    """

    @abstractmethod
    def replace(self, **changes: object) -> Self:
        """Return a new schedule with ``changes`` applied."""


class ScheduleMaterialization(ABC):
    """Facts derived from an accepted schedule for lowering."""

    @abstractmethod
    def validate(self, schedule: Schedule, root: object, *, place: object, workers: object) -> None:
        """Raise when these facts do not derive from ``schedule`` for ``root``."""


class ScheduleContext[ScheduleT: Schedule, AcceptanceT](ABC):
    """Problem and target facts that decide schedule compatibility."""

    @abstractmethod
    def accepts(self, schedule: ScheduleT) -> AcceptanceT:
        """Return the compatibility verdict for one complete schedule."""


class ScheduleCodec[ScheduleT: Schedule](ABC):
    """Strict canonical wire boundary for one schedule type."""

    @abstractmethod
    def encode(self, schedule: ScheduleT) -> dict[str, str]:
        """Encode one accepted schedule in canonical key order."""

    @abstractmethod
    def decode(self, row: Mapping[str, str]) -> ScheduleT:
        """Decode and validate one complete canonical row."""

    @abstractmethod
    def keys(self) -> tuple[str, ...]:
        """Return the accepted keys in canonical encoding order."""


__all__ = ["Schedule", "ScheduleCodec", "ScheduleContext", "ScheduleMaterialization"]
