"""Symbolic evaluation-demand analysis for fused Loop IR values.

Fusion may place two alpha-equivalent computations at different demand scopes.  This module
expands each pure SSA value to its exact external-load expression, alpha-renames the coordinate
axes, and groups equal values without enumerating tensor elements.  Each occurrence then compares
its execution scope with the axes its expression actually depends on.  An enclosing axis absent
from the coordinate map proves repeated evaluation; for static affine maps,
``expr.index_set_size`` additionally proves repetition when the execution domain is larger than
the distinct coordinate set.

This is analysis only.  It exposes no fusion boundary and chooses no placement.  The placement
realizer consumes its closed value classes when constructing materialization alternatives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, is_dataclass

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Expr, SimplifyCtx, Var, affine_form, index_set_size
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import Assign, Body, Load, Loop, Select, Write
from emmy.compiler.ir.stmt.base import Stmt


@dataclass(frozen=True)
class ValueUse:
    """One direct consumer of a pure value at its enclosing execution scope."""

    consumer: Stmt
    axes: tuple[Axis, ...]
    output: str | None = None


@dataclass(frozen=True)
class ValueOccurrence:
    """One SSA spelling of a canonical pure value."""

    name: str
    definition: Stmt
    axes: tuple[Axis, ...]
    coordinate_axes: tuple[str, ...]
    repeated_axes: tuple[Axis, ...]
    evaluations: int | None
    coordinate_upper_bound: int | None
    uses: tuple[ValueUse, ...]
    dependencies: tuple[Stmt, ...]

    @property
    def replication_lower_bound(self) -> int | None:
        """Minimum replication proven by the static affine coordinate upper bound."""
        if self.evaluations is None or self.coordinate_upper_bound in (None, 0):
            return None
        return math.ceil(self.evaluations / self.coordinate_upper_bound)

    @property
    def repeated(self) -> bool:
        """Whether this spelling provably evaluates a coordinate more than once."""
        return bool(self.repeated_axes) or (
            self.evaluations is not None and self.coordinate_upper_bound is not None and self.evaluations > self.coordinate_upper_bound
        )


@dataclass(frozen=True)
class ValueDemand:
    """Alpha-equivalent occurrences with one canonical execution-demand map."""

    signature: tuple
    occurrences: tuple[ValueOccurrence, ...]


@dataclass(frozen=True)
class ValueClass:
    """All alpha-equivalent spellings of one exact pure computation."""

    key: tuple
    demands: tuple[ValueDemand, ...]
    live_outputs: tuple[str, ...]

    @property
    def occurrences(self) -> tuple[ValueOccurrence, ...]:
        return tuple(o for demand in self.demands for o in demand.occurrences)

    @property
    def repeated(self) -> bool:
        """Repetition from an extra demand axis, affine ratio, or duplicate spelling."""
        occurrences = self.occurrences
        return any(o.repeated for o in occurrences) or len(occurrences) > 1


@dataclass(frozen=True)
class _Definition:
    name: str
    stmt: Stmt
    axes: tuple[Axis, ...]
    lane: int = 0


@dataclass
class _Expansion:
    axes: dict[str, int]
    coordinate_exprs: list[Expr]
    loads: set[tuple[int, int]]
    values: set[str]


class _Unsupported(Exception):
    """A value is not an exact closed pure expression in the supported grammar."""


def _dtype_key(dtype) -> str | None:
    return getattr(dtype, "name", None)


def _axis_extent(axis: Axis) -> str:
    return repr(axis.extent)


class _Analysis:
    def __init__(self, loop: LoopOp):
        self.loop = loop
        self.definitions: dict[str, _Definition] = {}
        self.uses: dict[str, list[ValueUse]] = {}
        self.order: dict[str, int] = {}
        self._collect(loop.body, ())

    def _collect(self, body: Body, axes: tuple[Axis, ...]) -> None:
        for stmt in body:
            if isinstance(stmt, Loop):
                self._collect(stmt.body, (*axes, stmt.axis))
                continue
            for lane, name in enumerate(stmt.defines()):
                self.order.setdefault(name, len(self.order))
                self.definitions[name] = _Definition(name=name, stmt=stmt, axes=axes, lane=lane)
            output = stmt.output if isinstance(stmt, Write) else None
            for name in stmt.deps():
                self.uses.setdefault(name, []).append(ValueUse(consumer=stmt, axes=axes, output=output))
            for nested in stmt.nested():
                self._collect(nested, axes)

    @staticmethod
    def _axis(scope: tuple[Axis, ...], name: str) -> Axis | None:
        return next((axis for axis in reversed(scope) if axis.name == name), None)

    def _axis_token(self, name: str, scope: tuple[Axis, ...], expansion: _Expansion) -> tuple:
        axis = self._axis(scope, name)
        if axis is None:
            return ("free", name)
        ordinal = expansion.axes.setdefault(name, len(expansion.axes))
        return ("axis", ordinal, _axis_extent(axis))

    def _expr_key(self, expr: Expr, scope: tuple[Axis, ...], expansion: _Expansion, pending: set[str]) -> tuple:
        scope_names = {axis.name for axis in scope}
        form = affine_form(expr, scope_names)
        if form is not None and form[1]:
            anchor, coeffs = form
            anchor = anchor.simplify(SimplifyCtx.empty())
            return (
                "affine",
                self._expr_key(anchor, scope, expansion, pending),
                tuple((self._axis_token(name, scope, expansion), coeff) for name, coeff in coeffs.items()),
            )
        if isinstance(expr, Var):
            if self._axis(scope, expr.name) is not None:
                return self._axis_token(expr.name, scope, expansion)
            if expr.name in self.definitions:
                value = self._value_key(expr.name, expansion, pending)
                if value is None:
                    raise _Unsupported
                return ("value", value)
            return ("free", expr.name)
        if not is_dataclass(expr):
            return (type(expr).__name__, repr(expr))

        def freeze(value):
            if isinstance(value, Expr):
                return self._expr_key(value, scope, expansion, pending)
            if isinstance(value, (tuple, list)):
                return tuple(freeze(item) for item in value)
            if isinstance(value, dict):
                return tuple(sorted((freeze(k), freeze(v)) for k, v in value.items()))
            return value

        return (type(expr).__name__, tuple((field.name, freeze(getattr(expr, field.name))) for field in fields(expr)))

    def _value_key(self, name: str, expansion: _Expansion, pending: set[str]) -> tuple | None:
        definition = self.definitions.get(name)
        if definition is None or name in pending:
            return None
        stmt = definition.stmt
        expansion.values.add(name)
        pending.add(name)
        try:
            if isinstance(stmt, Load):
                marker = (id(stmt), definition.lane)
                if marker not in expansion.loads:
                    expansion.loads.add(marker)
                    expansion.coordinate_exprs.extend(stmt.index)
                index = tuple(self._expr_key(expr, definition.axes, expansion, pending) for expr in stmt.index)
                return ("load", stmt.input, index, _dtype_key(stmt.dtype), definition.lane)
            if isinstance(stmt, Assign):
                args = tuple(self._value_key(arg, expansion, pending) for arg in stmt.args)
                if any(arg is None for arg in args):
                    return None
                return ("assign", stmt.op.name, args, _dtype_key(stmt.dtype))
            if isinstance(stmt, Select):
                branches = []
                for branch in stmt.branches:
                    value = self._value_key(branch.value, expansion, pending)
                    if value is None:
                        return None
                    branches.append((value, self._expr_key(branch.select, definition.axes, expansion, pending)))
                return ("select", tuple(branches))
            return None
        finally:
            pending.remove(name)

    def _occurrence(self, definition: _Definition) -> tuple[tuple, tuple, ValueOccurrence] | None:
        # A load is already materialized data, not a computed value placement may extract.
        if isinstance(definition.stmt, Load):
            return None
        expansion = _Expansion(axes={}, coordinate_exprs=[], loads=set(), values=set())
        try:
            key = self._value_key(definition.name, expansion, set())
        except _Unsupported:
            return None
        if key is None:
            return None
        coordinate_axes = tuple(expansion.axes)
        repeated_axes = tuple(axis for axis in definition.axes if axis.name not in expansion.axes)
        evaluations = None
        distinct = None
        static = all(axis.extent.is_static for axis in definition.axes)
        if static:
            extents = {axis.name: axis.extent.as_static() for axis in definition.axes}
            evaluations = math.prod(extents.values())
            if all(expr.free_vars() <= set(extents) for expr in expansion.coordinate_exprs):
                distinct = index_set_size(tuple(expansion.coordinate_exprs), extents) if expansion.coordinate_exprs else 1
        signature = tuple(
            ("coord", expansion.axes[axis.name], _axis_extent(axis)) if axis.name in expansion.axes else ("demand", _axis_extent(axis))
            for axis in definition.axes
        )
        occurrence = ValueOccurrence(
            name=definition.name,
            definition=definition.stmt,
            axes=definition.axes,
            coordinate_axes=coordinate_axes,
            repeated_axes=repeated_axes,
            evaluations=evaluations,
            coordinate_upper_bound=distinct,
            uses=tuple(self.uses.get(definition.name, ())),
            dependencies=tuple(
                item.stmt
                for item in sorted(self.definitions.values(), key=lambda item: self.order[item.name])
                if item.name in expansion.values
            ),
        )
        return key, signature, occurrence

    def run(self) -> tuple[ValueClass, ...]:
        grouped: dict[tuple, dict[tuple, list[ValueOccurrence]]] = {}
        live: dict[tuple, list[str]] = {}
        first: dict[tuple, int] = {}
        for definition in sorted(self.definitions.values(), key=lambda item: self.order[item.name]):
            got = self._occurrence(definition)
            if got is None:
                continue
            key, signature, occurrence = got
            first.setdefault(key, self.order[definition.name])
            grouped.setdefault(key, {}).setdefault(signature, []).append(occurrence)
            for use in occurrence.uses:
                if use.output is not None:
                    live.setdefault(key, []).append(use.output)
        classes = []
        for key in sorted(grouped, key=first.__getitem__):
            demands = tuple(ValueDemand(signature=sig, occurrences=tuple(items)) for sig, items in grouped[key].items())
            classes.append(ValueClass(key=key, demands=demands, live_outputs=tuple(dict.fromkeys(live.get(key, ())))))
        return tuple(classes)


def value_demands(loop: LoopOp) -> tuple[ValueClass, ...]:
    """Return exact pure-value equivalence classes and their symbolic execution demands."""
    return _Analysis(loop).run()


__all__ = ["ValueClass", "ValueDemand", "ValueOccurrence", "ValueUse", "value_demands"]
