"""Total structural lift from Loop IR to a Fold tree."""

from __future__ import annotations

from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.stmt import Assign, Body, Init, Load, Loop, Select
from emmy.compiler.ir.stmt.base import Stmt
from emmy.compiler.ir.tile import Placement, TileOp, split_effects
from emmy.compiler.pipeline.passes.lowering.tile._fromloop import _lift_body


def _peel(body: Body) -> tuple[list, list[Stmt]]:
    """Peel the outer parallel loop chain into placement axes."""
    axes = []
    prefix: list[Stmt] = []
    current = list(body)
    while True:
        index = 0
        while index < len(current) and isinstance(current[index], (Load, Assign, Init, Select)):
            index += 1
        head, rest = current[:index], current[index:]
        if len(rest) != 1 or not isinstance(rest[0], Loop) or rest[0].is_reduce:
            return axes, prefix + current
        prefix.extend(head)
        axes.append(rest[0].axis)
        current = list(rest[0].body)


def _raw_loops(body) -> list[Loop]:
    out = []
    for stmt in Body.coerce(body):
        if isinstance(stmt, Loop):
            out.append(stmt)
            out.extend(_raw_loops(stmt.body))
        elif isinstance(stmt, Fold):
            out.extend(_raw_loops(stmt.body))
    return out


def lift_tile(op: LoopOp, name: str = "") -> TileOp:
    """Peel free axes and lift the complete remaining nest as one Fold tree."""
    free, cell = _peel(Body(tuple(op.body)))
    lifted = _lift_body(cell)
    split = split_effects(lifted)
    if split is None:
        raise ValueError("Loop IR effects are not a kernel-boundary store")
    body, stores = split
    raw = _raw_loops(body)
    if raw:
        axes = ", ".join(loop.axis.name for loop in raw)
        raise ValueError(f"total lift left raw inner loops: {axes}")
    return TileOp(
        op=Fold.projection(body=Body(body)),
        name=name,
        place=Placement(free=tuple(free)),
        inputs=dict(op.inputs),
        stores=stores,
    )
