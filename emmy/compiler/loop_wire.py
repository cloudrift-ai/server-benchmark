"""Repository-local wire payload for post-fusion Loop IR fallbacks.

Torch provenance remains the preferred golden target because frontend IR is
the stable persistence boundary.  A traced kernel that has no frontend origin
is instead stored as its standalone Loop IR slice so inventory generation is
complete rather than lossy.  Golden files in this repository are regenerated
when this implementation-level Loop IR representation changes.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import DataType
from emmy.compiler.dtype import get as get_dtype
from emmy.compiler.graph import Graph
from emmy.compiler.ir.axis import Axis, AxisRole, Window
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Builtin, CastExpr, FuncCallExpr, Literal, TernaryExpr, Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import (
    Accum,
    Assign,
    Body,
    Cond,
    Init,
    Lambda,
    Load,
    Loop,
    Mma,
    Pack,
    Select,
    SelectBranch,
    StateMerge,
    StridedLoop,
    Unpack,
    Write,
    ZeroPrologue,
)
from emmy.compiler.torch_wire import (
    dim_from_wire,
    dim_to_wire,
    expr_from_wire,
    expr_to_wire,
    op_from_wire,
    op_to_wire,
    tensor_from_wire,
    tensor_to_wire,
)

_EXPR_TYPES = (Var, Literal, BinaryExpr, Builtin, FuncCallExpr, TernaryExpr, CastExpr)
_DATA_CLASSES = (
    Axis,
    Window,
    Accum,
    Assign,
    Cond,
    Init,
    Lambda,
    Load,
    Loop,
    Mma,
    Pack,
    Select,
    SelectBranch,
    StateMerge,
    StridedLoop,
    Unpack,
    Write,
    ZeroPrologue,
)
_CLASS_BY_NAME = {cls.__name__: cls for cls in _DATA_CLASSES}


def _value_to_wire(value: Any) -> Any:
    if isinstance(value, Dim):
        return {"dim": dim_to_wire(value)}
    if isinstance(value, _EXPR_TYPES):
        return {"expr": expr_to_wire(value)}
    if isinstance(value, DataType):
        return {"dtype": value.name}
    if isinstance(value, ElementwiseImpl):
        return {"elementwise": value.name}
    if isinstance(value, AxisRole):
        return {"axis_role": value.value}
    if isinstance(value, Body):
        return {"body": [_value_to_wire(item) for item in value]}
    if isinstance(value, tuple):
        return {"tuple": [_value_to_wire(item) for item in value]}
    if isinstance(value, frozenset):
        return {"frozenset": [_value_to_wire(item) for item in sorted(value, key=repr)]}
    if isinstance(value, list):
        return [_value_to_wire(item) for item in value]
    if isinstance(value, dict):
        return {"mapping": [[_value_to_wire(key), _value_to_wire(item)] for key, item in value.items()]}
    if is_dataclass(value) and type(value) in _DATA_CLASSES:
        return {
            "class": type(value).__name__,
            "fields": {field.name: _value_to_wire(getattr(value, field.name)) for field in fields(value)},
        }
    if isinstance(value, Enum):
        raise TypeError(f"Loop IR wire does not support enum {type(value).__name__}")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Loop IR wire does not support {type(value).__name__}")


def _value_from_wire(value: Any) -> Any:
    if isinstance(value, list):
        return [_value_from_wire(item) for item in value]
    if not isinstance(value, dict) or (len(value) != 1 and set(value) != {"class", "fields"}):
        if isinstance(value, dict):
            raise ValueError("Loop IR value must be a tagged mapping")
        return value
    if set(value) == {"dim"}:
        return dim_from_wire(value["dim"])
    if set(value) == {"expr"}:
        return expr_from_wire(value["expr"])
    if set(value) == {"dtype"}:
        return get_dtype(value["dtype"])
    if set(value) == {"elementwise"}:
        return ElementwiseImpl(str(value["elementwise"]))
    if set(value) == {"axis_role"}:
        return AxisRole(value["axis_role"])
    if set(value) == {"body"}:
        payload = value["body"]
        if not isinstance(payload, list):
            raise ValueError("Loop IR body must be a list")
        return Body(_value_from_wire(item) for item in payload)
    if set(value) == {"tuple"}:
        payload = value["tuple"]
        if not isinstance(payload, list):
            raise ValueError("Loop IR tuple must be a list")
        return tuple(_value_from_wire(item) for item in payload)
    if set(value) == {"frozenset"}:
        payload = value["frozenset"]
        if not isinstance(payload, list):
            raise ValueError("Loop IR frozenset must be a list")
        return frozenset(_value_from_wire(item) for item in payload)
    if set(value) == {"mapping"}:
        payload = value["mapping"]
        if not isinstance(payload, list) or any(not isinstance(pair, list) or len(pair) != 2 for pair in payload):
            raise ValueError("Loop IR mapping must be a list of key/value pairs")
        return {_value_from_wire(key): _value_from_wire(item) for key, item in payload}
    if set(value) == {"class", "fields"}:
        class_name = value["class"]
        if not isinstance(class_name, str):
            raise ValueError("Loop IR class name must be a string")
        cls = _CLASS_BY_NAME.get(class_name)
        if cls is None:
            raise ValueError(f"Loop IR value has unknown class {class_name!r}")
        payload = value["fields"]
        if not isinstance(payload, dict):
            raise ValueError(f"Loop IR {class_name} fields must be a mapping")
        expected = {field.name for field in fields(cls)}
        unknown = set(payload) - expected
        if unknown:
            raise ValueError(f"Loop IR {class_name} has unknown fields: {', '.join(sorted(unknown))}")
        try:
            return cls(**{name: _value_from_wire(item) for name, item in payload.items()})
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Loop IR {class_name} is invalid: {exc}") from exc
    raise ValueError("Loop IR value has an unknown tag")


def loop_graph_to_wire(graph: Graph) -> dict:
    """Serialize a standalone Loop IR graph to YAML-safe data."""
    nodes = []
    for node_id in graph.topological_order():
        node = graph.nodes[node_id]
        if isinstance(node.op, LoopOp):
            op = "loop"
            attrs = {"body": _value_to_wire(node.op.body)}
            if node.op.name:
                attrs["name"] = node.op.name
        elif isinstance(node.op, (InputOp, ConstantOp)):
            encoded = op_to_wire(node.op)
            op, attrs = encoded["op"], encoded["attrs"]
        else:
            raise TypeError(f"Loop IR graph contains unsupported compute op {type(node.op).__name__}")
        item = {"id": node_id, "op": op}
        if attrs:
            item["attrs"] = attrs
        if node.inputs:
            item["inputs"] = list(node.inputs)
        item["outputs"] = [tensor_to_wire(tensor) for tensor in node.outputs]
        nodes.append(item)
    return {"inputs": list(graph.inputs), "outputs": list(graph.outputs), "nodes": nodes}


def loop_graph_from_wire(value: object) -> Graph:
    """Decode and validate a standalone post-fusion kernel slice."""
    if not isinstance(value, dict):
        raise ValueError("Loop IR program must be a mapping")
    unknown = set(value) - {"inputs", "outputs", "nodes"}
    if unknown:
        raise ValueError(f"Loop IR program has unknown fields: {', '.join(sorted(unknown))}")
    nodes = value.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("Loop IR program nodes must be a list")
    graph = Graph()
    for index, item in enumerate(nodes):
        if not isinstance(item, dict):
            raise ValueError(f"Loop IR node {index} must be a mapping")
        unknown = set(item) - {"id", "op", "attrs", "inputs", "outputs"}
        if unknown:
            raise ValueError(f"Loop IR node {index} has unknown fields: {', '.join(sorted(unknown))}")
        node_id = item.get("id")
        inputs = item.get("inputs", [])
        outputs = item.get("outputs")
        attrs = item.get("attrs", {})
        if not isinstance(node_id, str) or not node_id:
            raise ValueError(f"Loop IR node {index} requires a non-empty id")
        if not isinstance(inputs, list) or not all(isinstance(name, str) for name in inputs):
            raise ValueError(f"Loop IR node {node_id!r} inputs must be string names")
        if not isinstance(outputs, list) or not outputs:
            raise ValueError(f"Loop IR node {node_id!r} outputs must be a non-empty list")
        if not isinstance(attrs, dict):
            raise ValueError(f"Loop IR node {node_id!r} attrs must be a mapping")
        if item.get("op") == "loop":
            unknown_attrs = set(attrs) - {"body", "name"}
            if unknown_attrs or "body" not in attrs:
                raise ValueError(f"Loop IR node {node_id!r} has invalid loop attrs")
            try:
                body = _value_from_wire(attrs["body"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Loop IR node {node_id!r} body is invalid: {exc}") from exc
            if not isinstance(body, Body):
                raise ValueError(f"Loop IR node {node_id!r} body did not decode to Body")
            op = LoopOp(body=body, name=str(attrs.get("name", "")))
        else:
            op = op_from_wire({"op": item.get("op"), "attrs": attrs})
            if not isinstance(op, (InputOp, ConstantOp)):
                raise ValueError(f"Loop IR node {node_id!r} boundary must be input or constant")
        try:
            graph.add_node(op, inputs, outputs=tuple(tensor_from_wire(output) for output in outputs), node_id=node_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Loop IR node {node_id!r} is invalid: {exc}") from exc
    graph_inputs = value.get("inputs")
    graph_outputs = value.get("outputs")
    if not isinstance(graph_inputs, list) or not all(isinstance(name, str) for name in graph_inputs):
        raise ValueError("Loop IR program inputs must be string names")
    if not isinstance(graph_outputs, list) or not all(isinstance(name, str) for name in graph_outputs):
        raise ValueError("Loop IR program outputs must be string names")
    graph.inputs = list(graph_inputs)
    graph.outputs = list(graph_outputs)
    compute = [node for node in graph.nodes.values() if not isinstance(node.op, (InputOp, ConstantOp))]
    if not compute or any(not isinstance(node.op, LoopOp) for node in compute):
        raise ValueError("Loop IR program must contain only LoopOp compute nodes")
    output_producers = [graph.producer(name) for name in graph.outputs]
    if not output_producers or any(node is None or not isinstance(node.op, LoopOp) for node in output_producers):
        raise ValueError("Loop IR program outputs must be produced by LoopOp nodes")
    for name in (*graph.inputs, *graph.outputs):
        if graph.buffer(name) is None:
            raise ValueError(f"Loop IR program references unknown boundary buffer {name!r}")
    graph.topological_order()
    return graph


def intern_loop_program(programs: list[dict], graph: Graph) -> int:
    payload = loop_graph_to_wire(graph)
    for index, current in enumerate(programs):
        if current == payload:
            return index
    programs.append(payload)
    return len(programs) - 1


def validate_loop_program_pool(programs: object) -> list[dict]:
    if programs is None:
        return []
    if not isinstance(programs, list):
        raise ValueError("golden loops must be a list")
    out: list[dict] = []
    for index, payload in enumerate(programs):
        if not isinstance(payload, dict):
            raise ValueError(f"golden Loop IR program {index} must be a mapping")
        try:
            loop_graph_from_wire(payload)
        except ValueError as exc:
            raise ValueError(f"golden Loop IR program {index}: {exc}") from exc
        out.append(payload)
    return out
