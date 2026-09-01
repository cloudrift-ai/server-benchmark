"""Adapt a semantic schedule enumeration to the pipeline's lazy fork interface."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from frozendict import frozendict

from emmy.compiler.ir.schedule import Schedule, ScheduleContext
from emmy.compiler.pipeline.fork import Fork, iter_leaves, schedule_forks


@dataclass(frozen=True)
class ScheduleLeaf(Fork):
    """One accepted typed schedule, materialized only if search selects it."""

    schedule: Schedule
    row: Mapping
    inherited_knobs: Mapping
    materialize: Callable[[Schedule, dict], object]
    pool_id: str
    is_leaf = True

    @property
    def knobs(self) -> dict:
        return {**self.inherited_knobs, **self.row}

    def expand(self) -> list:
        return [self.materialize(self.schedule, self.knobs)]


class _SampleRow(dict):
    __slots__ = ("schedule",)

    def __init__(self, schedule: Schedule, row: Mapping) -> None:
        super().__init__(row)
        self.schedule = schedule


def fork_schedule(
    context: ScheduleContext,
    *,
    codec,
    inherited_knobs: Mapping,
    row_prefix: Mapping,
    materialize: Callable[[Schedule, dict], object],
    pool_id: str,
    pool_bound: int,
    pool_descent_bound: int,
    sample=None,
) -> list[Fork]:
    """Build and optionally sample a lazy fork tree from a semantic schedule context."""

    def leaf(assignment: Schedule) -> ScheduleLeaf:
        row = frozendict({**row_prefix, **codec._encode(assignment)})
        return ScheduleLeaf(assignment, row, dict(inherited_knobs), materialize, pool_id)

    roots = schedule_forks(
        context,
        branch_knobs={**inherited_knobs, **row_prefix},
        row_delta=codec.delta,
        leaf=leaf,
        pool_id=pool_id,
        pool_bound=pool_bound,
        pool_descent_bound=pool_descent_bound,
    )
    if sample is None:
        return roots
    drawn = sample.take(_SampleRow(option.schedule, option.row) for option in iter_leaves(roots) if isinstance(option, ScheduleLeaf))
    sample.totals[pool_id] = drawn.total
    return [ScheduleLeaf(row.schedule, frozendict(row), dict(inherited_knobs), materialize, pool_id) for row in drawn.rows]


__all__ = ["ScheduleLeaf", "fork_schedule"]
