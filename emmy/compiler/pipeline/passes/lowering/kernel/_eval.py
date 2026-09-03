"""Residence-aware evaluation of pure Lambdas over tensor-core fragments.

The evaluator is deliberately algebra-blind. A value is uniform for the CTA cell, distributed
one scalar per fragment row, or distributed elementwise in C fragments.  ``Assign`` and ``Select``
broadcast over those residences, and a structural child is delegated to the caller that owns its
schedule.  The same machinery evaluates a Fold's stored carrier ``combine`` Lambda; no operation
family is named here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal, TernaryExpr, Var
from emmy.compiler.ir.kernel.ir import (
    FRAG,
    FRAG_COL,
    FRAG_ROW,
    ROW,
    UNIFORM,
    FragmentApply,
    FragmentLoad,
    FragmentMask,
    FragmentSelect,
    Reassign,
)
from emmy.compiler.ir.pure import Fold, Lambda
from emmy.compiler.ir.stmt import Assign, Const, Load, Select, SelectBranch, Stmt


@dataclass(frozen=True)
class Value:
    """One pure value and its physical residence.

    ``data`` is a scalar name/literal for :data:`UNIFORM`, ``rows × lanes`` scalar names for
    :data:`ROW`, and ``register rows × register columns`` fragment names for :data:`FRAG`.
    """

    kind: str
    data: object

    @classmethod
    def uniform(cls, value: object) -> Value:
        return cls(UNIFORM, value)

    @classmethod
    def row(cls, rows) -> Value:
        return cls(ROW, tuple(tuple(group) for group in rows))

    @classmethod
    def frag(cls, frags) -> Value:
        return cls(FRAG, tuple(tuple(group) for group in frags))


_RANK = {UNIFORM: 0, ROW: 1, FRAG: 2}


def _residence(values) -> str:
    return max((value.kind for value in values), key=_RANK.__getitem__, default=UNIFORM)


def _scalar(value: Value, i: int = 0, r: int = 0):
    if value.kind == UNIFORM:
        return value.data
    if value.kind == ROW:
        return value.data[i][r]
    raise ValueError("a fragment value has no scalar spelling")


def _frag_arg(value: Value, i: int, j: int):
    if value.kind == FRAG:
        return value.data[i][j]
    if value.kind == ROW:
        return value.data[i]
    return value.data


def _target(name: str, kind: str, shape: tuple[int, int], bound: Value | None) -> Value:
    if bound is not None:
        if bound.kind != kind:
            raise ValueError(f"Lambda result {name!r} changes residence from {bound.kind} to {kind}")
        return bound
    m, n = shape
    if kind == FRAG:
        return Value.frag(tuple(tuple(f"{name}__f{i}_{j}" for j in range(n)) for i in range(m)))
    if kind == ROW:
        return Value.row(tuple(tuple(f"{name}__r{i}_{r}" for r in range(n)) for i in range(m)))
    return Value.uniform(name)


def _assign(stmt: Assign, args: tuple[Value, ...], target: Value | None, shape: tuple[int, int]) -> tuple[list[Stmt], Value]:
    kind = _residence(args)
    out = _target(stmt.name, kind, shape, target)
    if kind == FRAG:
        like = next(value for value in args if value.kind == FRAG)
        rows, cols = len(like.data), len(like.data[0])
        if target is None:
            out = _target(stmt.name, kind, (rows, cols), None)
        body = [
            FragmentApply(
                out=out.data[i][j],
                op=stmt.op,
                args=tuple(_frag_arg(value, i, j) for value in args),
                kinds=tuple(value.kind for value in args),
                in_place=target is not None,
            )
            for i in range(rows)
            for j in range(cols)
        ]
        return body, out
    if kind == ROW:
        like = next(value for value in args if value.kind == ROW)
        rows, lanes = len(like.data), len(like.data[0])
        if target is None:
            out = _target(stmt.name, kind, (rows, lanes), None)
        body: list[Stmt] = []
        for i in range(rows):
            for r in range(lanes):
                name = out.data[i][r]
                values = tuple(_scalar(value, i, r) for value in args)
                if target is None:
                    body.append(Assign(name=name, op=stmt.op, args=values, dtype=stmt.dtype))
                else:
                    tmp = f"{name}__next"
                    body.extend((Assign(name=tmp, op=stmt.op, args=values, dtype=stmt.dtype), Reassign(name=name, value=tmp)))
        return body, out
    if target is None:
        return [stmt], out
    tmp = f"{target.data}__next"
    return [Assign(name=tmp, op=stmt.op, args=tuple(value.data for value in args), dtype=stmt.dtype), Reassign(target.data, tmp)], out


def _select(
    stmt: Select,
    env: dict[str, Value],
    target: Value | None,
    bases: tuple[tuple[tuple[Expr, Expr], ...], ...] | None,
) -> tuple[list[Stmt], Value]:
    values = tuple(env.get(branch.value, Value.uniform(branch.value)) for branch in stmt.branches)
    coordinate = any({FRAG_ROW, FRAG_COL} & branch.select.free_vars() for branch in stmt.branches)
    kind = FRAG if coordinate else _residence(values)
    if kind == FRAG:
        if bases is None or any(value.kind != UNIFORM for value in values):
            raise ValueError("fragment Select needs coordinate bases and uniform branch values")
        rows, cols = len(bases), len(bases[0])
        out = _target(stmt.name, FRAG, (rows, cols), target)
        selected = out if target is None else _target(f"{stmt.name}__next", FRAG, (rows, cols), None)
        body = [
            FragmentSelect(
                out=selected.data[i][j],
                branches=tuple(SelectBranch(value.data, branch.select) for value, branch in zip(values, stmt.branches, strict=True)),
                row_base=bases[i][j][0],
                col_base=bases[i][j][1],
            )
            for i in range(rows)
            for j in range(cols)
        ]
        if target is not None:
            body.extend(
                FragmentApply(
                    out=out.data[i][j],
                    op=ElementwiseImpl("copy"),
                    args=(selected.data[i][j],),
                    kinds=(FRAG,),
                    in_place=True,
                )
                for i in range(rows)
                for j in range(cols)
            )
        return body, out
    if kind == ROW:
        like = next(value for value in values if value.kind == ROW)
        rows, lanes = len(like.data), len(like.data[0])
        out = _target(stmt.name, ROW, (rows, lanes), target)
        body: list[Stmt] = []
        for i in range(rows):
            for r in range(lanes):
                name = out.data[i][r]
                selected = name if target is None else f"{name}__next"
                scalar = Select(
                    name=selected,
                    branches=tuple(
                        SelectBranch(_scalar(value, i, r), branch.select) for value, branch in zip(values, stmt.branches, strict=True)
                    ),
                )
                body.append(scalar)
                if target is not None:
                    body.append(Reassign(name, selected))
        return body, out
    out = _target(stmt.name, UNIFORM, (1, 1), target)
    if target is None:
        return [stmt], out
    selected = f"{out.data}__next"
    return [Select(name=selected, branches=stmt.branches), Reassign(out.data, selected)], out


Child = Callable[[Fold, dict[str, Value]], tuple[list[Stmt], Value | tuple[Value, ...]]]


def evaluate(
    lam: Lambda,
    bindings: dict[str, Value],
    *,
    child: Child | None = None,
    targets: dict[str, Value] | None = None,
    bases: tuple[tuple[tuple[Expr, Expr], ...], ...] | None = None,
    axes: tuple[str, str] | None = None,
    bounds: tuple[tuple[str, Expr, float | None], ...] = (),
) -> tuple[list[Stmt], tuple[Value, ...], dict[str, Value]]:
    """Evaluate ``lam`` with residence-aware bindings.

    ``targets`` pre-binds carried result names, turning their final writes into reassignment.
    ``child`` is the only structural dispatch: it receives a scheduled Fold plus the live value
    environment and returns its statements and result residence.

    ``bounds`` carries one ``(axis, extent, fill)`` entry per runtime-bounded coordinate axis:
    every coordinate-dependent Load clamps that axis's coordinate in-bounds (the overhang reads a
    duplicate of the last valid element), and an entry with a non-``None`` fill — a reduced axis,
    whose overhang the fold consumes — additionally masks the loaded fragment to that identity.
    A bound is applied per Load, to the axes that Load actually reads; an axis outside the
    fragment's own ``axes`` pair has no fragment coordinate and its bound is silently ignored.
    """

    env = dict(bindings)
    targets = targets or {}
    body: list[Stmt] = []
    coordinate_values = set(axes or ())
    for stmt in lam.body:
        if isinstance(stmt, Fold):
            if child is None:
                raise ValueError("Lambda contains a Fold but no scheduled child evaluator was supplied")
            emitted, value = child(stmt, env)
            body.extend(emitted)
            values = value if isinstance(value, tuple) else (value,)
            if len(values) != len(stmt.defines()):
                raise ValueError("scheduled child result arity does not match its Fold definitions")
            env.update(zip(stmt.defines(), values, strict=True))
            continue
        if isinstance(stmt, Load):
            coordinate = set(axes or ()) & set(stmt.deps())
            if not coordinate:
                body.append(stmt)
                for name in stmt.names:
                    env[name] = Value.uniform(name)
                continue
            if bases is None or axes is None or not stmt.is_scalar:
                raise ValueError("fragment Lambda needs cell bases for a coordinate-dependent scalar Load")
            sub = {axes[0]: Var(FRAG_ROW), axes[1]: Var(FRAG_COL)}
            fills: list[tuple[Var, Expr, float]] = []
            for axis, ext, fill in bounds:
                if axis not in coordinate:
                    continue
                coordinate_var = Var(FRAG_ROW if axis == axes[0] else FRAG_COL)
                sub[axis] = TernaryExpr(
                    BinaryExpr("<", coordinate_var, ext),
                    coordinate_var,
                    BinaryExpr("-", ext, Literal(1, "int")),
                )
                if fill is not None:
                    fills.append((coordinate_var, ext, fill))
            rows, cols = len(bases), len(bases[0])
            value = _target(stmt.name, FRAG, (rows, cols), None)
            body.extend(
                FragmentLoad(
                    out=value.data[i][j],
                    input=stmt.input,
                    index=tuple(expr.substitute(sub) for expr in stmt.index),
                    row_base=bases[i][j][0],
                    col_base=bases[i][j][1],
                    dtype=stmt.dtype,
                )
                for i in range(rows)
                for j in range(cols)
            )
            body.extend(
                FragmentMask(
                    frag=value.data[i][j],
                    mask_when=BinaryExpr(">=", coordinate_var, ext),
                    row_base=bases[i][j][0],
                    col_base=bases[i][j][1],
                    fill=fill,
                )
                for coordinate_var, ext, fill in fills
                for i in range(rows)
                for j in range(cols)
            )
            env[stmt.name] = value
            continue
        if isinstance(stmt, Assign):
            if set(stmt.args) & coordinate_values:
                raise ValueError("fragment Lambda cannot broadcast a coordinate-dependent value")
            emitted, value = _assign(
                stmt,
                tuple(env.get(arg, Value.uniform(arg)) for arg in stmt.args),
                targets.get(stmt.name),
                (1, 1),
            )
        elif isinstance(stmt, Const):
            emitted, value = [stmt], Value.uniform(stmt.name)
        elif isinstance(stmt, Select):
            if axes and any(set(axes) & branch.select.free_vars() for branch in stmt.branches):
                stmt = coordinate_select(stmt, axes)
            emitted, value = _select(stmt, env, targets.get(stmt.name), bases)
        else:
            raise ValueError(f"fragment Lambda cannot evaluate {type(stmt).__name__}")
        body.extend(emitted)
        env[stmt.defines()[0]] = value
    results = tuple(env[result] for result in lam.results)
    return body, results, env


def coordinate_select(stmt: Select, axes: tuple[str, str]) -> Select:
    """Rewrite a scalar coordinate Select over ``axes`` to fragment coordinate placeholders."""
    sub = {axes[0]: Var(FRAG_ROW), axes[1]: Var(FRAG_COL)}
    return Select(
        name=stmt.name,
        branches=tuple(SelectBranch(branch.value, branch.select.substitute(sub)) for branch in stmt.branches),
    )


__all__ = ["Value", "coordinate_select", "evaluate"]
