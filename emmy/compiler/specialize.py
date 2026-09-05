"""Bind symbolic dimensions in persisted compiler programs."""

from __future__ import annotations

from collections.abc import Mapping

from emmy.compiler.ir.expr import Interval, Literal, SimplifyCtx
from emmy.compiler.loop_wire import loop_graph_from_wire, loop_graph_to_wire
from emmy.compiler.torch_wire import expr_from_wire, expr_to_wire, graph_from_wire, graph_to_wire

_EXPR_TAGS = {"var", "literal", "binary", "builtin", "call", "ternary", "cast"}
_NAMED_SHAPE_OPS = {"torch.reshape", "torch.slice"}


def _specialize_expr(value: Mapping, bindings: Mapping[str, int], *, extent: bool = False) -> dict:
    """Bind the named dimensions inside one wire expression and simplify what that fixes.

    ``extent`` says the expression IS a dimension, so every name still free in it is a
    tensor extent and simplification may use the one fact an extent carries: it is at
    least 1. Every other expression indexes a tensor rather than sizing one, and its free
    names are output coordinates and loop variables that start at 0 — reading those as
    extents folds a real predicate away (an IndexMap's ``out_coord_1 < 1`` becomes false,
    silently dropping that source), so they simplify with no range at all.
    """
    expr = expr_from_wire(dict(value))
    replacements = {name: Literal(size, "int") for name, size in bindings.items()}
    specialized = expr.substitute(replacements)
    ranges = {name: Interval(1, 1 << 30) for name in specialized.free_vars()} if extent else {}
    return expr_to_wire(specialized.simplify(SimplifyCtx(ranges)))


def _specialize_dim(value, bindings: Mapping[str, int]):
    if isinstance(value, int):
        return value
    if not isinstance(value, Mapping):
        return value
    if "sym" in value and set(value) <= {"sym", "hint"}:
        return bindings.get(value["sym"], dict(value))
    if "expr" in value and set(value) <= {"expr", "hint"}:
        expr = _specialize_expr(value["expr"], bindings, extent=True)
        if set(expr) == {"literal"} and expr["literal"].get("dtype") == "int":
            return int(expr["literal"]["value"])
        result = {"expr": expr}
        if "hint" in value:
            result["hint"] = value["hint"]
        return result
    return {key: _specialize_wire(item, bindings) for key, item in value.items()}


def _specialize_named_shape(value, bindings: Mapping[str, int]):
    if isinstance(value, str):
        return bindings.get(value, value)
    if isinstance(value, list):
        return [_specialize_named_shape(item, bindings) for item in value]
    if isinstance(value, Mapping) and set(value) == {"__tuple__"}:
        return {"__tuple__": _specialize_named_shape(value["__tuple__"], bindings)}
    return value


def _specialize_wire(value, bindings: Mapping[str, int]):
    if isinstance(value, list):
        return [_specialize_wire(item, bindings) for item in value]
    if not isinstance(value, Mapping):
        return value

    keys = set(value)
    if len(value) == 1 and keys <= _EXPR_TAGS:
        return _specialize_expr(value, bindings)
    if keys <= {"sym", "hint"} and "sym" in value:
        return _specialize_dim(value, bindings)
    if keys <= {"expr", "hint"} and "expr" in value and "hint" in value:
        return _specialize_dim(value, bindings)
    if keys == {"__dim__"}:
        return {"__dim__": _specialize_dim(value["__dim__"], bindings)}
    if keys == {"dim"}:
        return {"dim": _specialize_dim(value["dim"], bindings)}
    if keys in ({"__expr__"}, {"expr"}):
        key = next(iter(keys))
        return {key: _specialize_expr(value[key], bindings)}
    specialized = {key: _specialize_wire(item, bindings) for key, item in value.items()}
    attrs = specialized.get("attrs")
    tag = specialized.get("op")
    if isinstance(tag, str) and tag in _NAMED_SHAPE_OPS and isinstance(attrs, Mapping) and "shape" in attrs:
        specialized["attrs"] = dict(attrs)
        specialized["attrs"]["shape"] = _specialize_named_shape(attrs["shape"], bindings)
    return specialized


def specialize_program(graph, bindings: Mapping[str, int], *, loop: bool = False):
    """Return a copy of ``graph`` with the named symbolic dimensions bound."""
    if not bindings:
        return graph.copy()
    invalid = {
        name: value for name, value in bindings.items() if not isinstance(name, str) or not name or type(value) is not int or value <= 0
    }
    if invalid:
        raise ValueError(f"dimension bindings must map non-empty names to positive integers: {invalid!r}")
    if loop:
        return loop_graph_from_wire(_specialize_wire(loop_graph_to_wire(graph), bindings))
    return graph_from_wire(_specialize_wire(graph_to_wire(graph), bindings))


__all__ = ["specialize_program"]
