"""Stable wire format for trace-stage Torch IR programs.

This is the persistence boundary used by golden YAML.  It is intentionally
separate from :meth:`Graph.to_dict`, which remains an implementation/debug dump
for every compiler dialect.  The wire format admits only boundary, frontend,
and tensor operations; uses stable external operation tags; and represents
expressions and symbolic dimensions as tagged data rather than Python reprs.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph
from emmy.compiler.ir.base import ConstantOp, InputOp, Op
from emmy.compiler.ir.expr import BinaryExpr, Builtin, CastExpr, FuncCallExpr, Literal, TernaryExpr, Var
from emmy.compiler.ir.frontend.ir import (
    CatOp,
    LayerNormOp,
    LinearOp,
    MatmulOp,
    MeanOp,
    ReshapeOp,
    RmsNormOp,
    SdpaOp,
    SliceOp,
    SoftmaxOp,
    TransposeOp,
    UnsqueezeOp,
)
from emmy.compiler.ir.tensor.ir import (
    BitcastOp,
    CastOp,
    ElementwiseOp,
    GatherOp,
    IndexMapOp,
    IndexSource,
    RangeOp,
    ReduceOp,
    ScanOp,
    ScatterOp,
)
from emmy.compiler.tensor import Tensor

IR_VERSION = 1
PROGRAM_PREFIX = "sha256:"


# Stable wire tag -> (runtime class, semantic dataclass fields).  Runtime-only
# Op fields (source/knobs/populated IO) never appear here.
_OP_SPECS: dict[str, tuple[type[Op], tuple[str, ...]]] = {
    "input": (InputOp, ()),
    "constant": (
        ConstantOp,
        (
            "name",
            "value",
            "context_value",
            "load_ops",
            "source_path",
            "source_parts",
            "source_shape",
            "source_dtype",
            "source_graph",
        ),
    ),
    "torch.transpose": (TransposeOp, ("axes",)),
    "torch.reshape": (ReshapeOp, ("shape",)),
    "torch.slice": (SliceOp, ("shape", "dim", "start")),
    "torch.cat": (CatOp, ()),
    "torch.unsqueeze": (UnsqueezeOp, ("dim",)),
    "torch.linear": (LinearOp, ("has_bias",)),
    "torch.matmul": (MatmulOp, ("has_bias",)),
    "torch.sdpa": (SdpaOp, ("is_causal", "sliding_window", "scale")),
    "torch.mean": (MeanOp, ("axis",)),
    "torch.rms_norm": (RmsNormOp, ("eps",)),
    "torch.layer_norm": (LayerNormOp, ("eps",)),
    "torch.softmax": (SoftmaxOp, ("axis",)),
    "tensor.range": (RangeOp, ("start", "stop", "step", "dtype")),
    "tensor.cast": (CastOp, ("dtype",)),
    "tensor.bitcast": (BitcastOp, ("dtype",)),
    "tensor.elementwise": (ElementwiseOp, ("op",)),
    "tensor.reduce": (ReduceOp, ("op", "axis")),
    "tensor.scan": (ScanOp, ("op", "axis")),
    "tensor.gather": (GatherOp, ("axis",)),
    "tensor.scatter": (ScatterOp, ("axis", "reduce_fn")),
    "tensor.index_map": (IndexMapOp, ("out_shape", "sources")),
}
_TAG_BY_CLASS = {cls: tag for tag, (cls, _fields) in _OP_SPECS.items()}


def _keys(value: dict, expected: set[str], where: str) -> None:
    unknown = set(value) - expected
    if unknown:
        raise ValueError(f"{where}: unknown field(s): {', '.join(sorted(unknown))}")


def expr_to_wire(expr) -> dict:
    if isinstance(expr, Var):
        return {"var": expr.name}
    if isinstance(expr, Literal):
        return {"literal": {"value": expr.value, "dtype": expr.dtype}}
    if isinstance(expr, BinaryExpr):
        return {"binary": {"op": expr.op, "left": expr_to_wire(expr.left), "right": expr_to_wire(expr.right)}}
    if isinstance(expr, Builtin):
        return {"builtin": expr.name}
    if isinstance(expr, FuncCallExpr):
        return {"call": {"name": expr.name, "args": [expr_to_wire(arg) for arg in expr.args]}}
    if isinstance(expr, TernaryExpr):
        return {
            "ternary": {
                "cond": expr_to_wire(expr.cond),
                "if_true": expr_to_wire(expr.if_true),
                "if_false": expr_to_wire(expr.if_false),
            }
        }
    if isinstance(expr, CastExpr):
        return {"cast": {"dtype": expr.dtype, "expr": expr_to_wire(expr.expr)}}
    raise TypeError(f"Torch IR wire: unsupported expression {type(expr).__name__}")


def expr_from_wire(value: object):
    if not isinstance(value, dict) or len(value) != 1:
        raise ValueError("Torch IR expression must be a one-key mapping")
    tag, payload = next(iter(value.items()))
    if tag == "var":
        if not isinstance(payload, str):
            raise ValueError("Torch IR var must be a string")
        return Var(payload)
    if tag == "literal":
        if not isinstance(payload, dict):
            raise ValueError("Torch IR literal must be a mapping")
        _keys(payload, {"value", "dtype"}, "Torch IR literal")
        return Literal(payload["value"], payload.get("dtype", "float"))
    if tag == "binary":
        if not isinstance(payload, dict):
            raise ValueError("Torch IR binary expression must be a mapping")
        _keys(payload, {"op", "left", "right"}, "Torch IR binary expression")
        return BinaryExpr(str(payload["op"]), expr_from_wire(payload["left"]), expr_from_wire(payload["right"]))
    if tag == "builtin":
        if not isinstance(payload, str):
            raise ValueError("Torch IR builtin must be a string")
        return Builtin(payload)
    if tag == "call":
        if not isinstance(payload, dict):
            raise ValueError("Torch IR call must be a mapping")
        _keys(payload, {"name", "args"}, "Torch IR call")
        return FuncCallExpr(str(payload["name"]), tuple(expr_from_wire(arg) for arg in payload.get("args", [])))
    if tag == "ternary":
        if not isinstance(payload, dict):
            raise ValueError("Torch IR ternary must be a mapping")
        _keys(payload, {"cond", "if_true", "if_false"}, "Torch IR ternary")
        return TernaryExpr(
            expr_from_wire(payload["cond"]),
            expr_from_wire(payload["if_true"]),
            expr_from_wire(payload["if_false"]),
        )
    if tag == "cast":
        if not isinstance(payload, dict):
            raise ValueError("Torch IR cast must be a mapping")
        _keys(payload, {"dtype", "expr"}, "Torch IR cast")
        return CastExpr(str(payload["dtype"]), expr_from_wire(payload["expr"]))
    raise ValueError(f"Torch IR expression has unknown tag {tag!r}")


def dim_to_wire(dim: Dim) -> dict:
    payload = expr_to_wire(dim.expr)
    if isinstance(dim.expr, Literal) and dim.expr.dtype == "int":
        return {"const": int(dim.expr.value)}
    if isinstance(dim.expr, Var):
        out: dict[str, object] = {"sym": dim.expr.name}
        if dim.hint is not None:
            out["hint"] = dim.hint
        return out
    out = {"expr": payload}
    if dim.hint is not None:
        out["hint"] = dim.hint
    return out


def dim_from_wire(value: object) -> Dim:
    if not isinstance(value, dict):
        raise ValueError("Torch IR dimension must be a mapping")
    if "const" in value:
        _keys(value, {"const"}, "Torch IR static dimension")
        return Dim(int(value["const"]))
    if "sym" in value:
        _keys(value, {"sym", "hint"}, "Torch IR symbolic dimension")
        return Dim(str(value["sym"]), hint=int(value["hint"]) if value.get("hint") is not None else None)
    if "expr" in value:
        _keys(value, {"expr", "hint"}, "Torch IR composite dimension")
        return Dim(expr_from_wire(value["expr"]), hint=int(value["hint"]) if value.get("hint") is not None else None)
    raise ValueError("Torch IR dimension requires const, sym, or expr")


def _value_to_wire(value: Any) -> Any:
    from emmy.compiler.ir.elementwise import ElementwiseImpl

    if isinstance(value, Dim):
        return {"__dim__": dim_to_wire(value)}
    if isinstance(value, (Var, Literal, BinaryExpr, Builtin, FuncCallExpr, TernaryExpr, CastExpr)):
        return {"__expr__": expr_to_wire(value)}
    if isinstance(value, ElementwiseImpl):
        return {"__elementwise__": value.name}
    if isinstance(value, IndexSource):
        return {
            "__index_source__": {
                "input_idx": value.input_idx,
                "coord_map": [expr_to_wire(expr) for expr in value.coord_map],
                "select": expr_to_wire(value.select) if value.select is not None else None,
            }
        }
    if isinstance(value, Op):
        return {"__op__": op_to_wire(value)}
    if isinstance(value, Graph):
        return {"__program__": graph_to_wire(value)}
    if isinstance(value, tuple):
        return {"__tuple__": [_value_to_wire(item) for item in value]}
    if isinstance(value, list):
        return [_value_to_wire(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _value_to_wire(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Torch IR wire: unsupported value {type(value).__name__}")


def _value_from_wire(value: Any) -> Any:
    from emmy.compiler.ir.elementwise import ElementwiseImpl

    if isinstance(value, list):
        return [_value_from_wire(item) for item in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {"__dim__"}:
        return dim_from_wire(value["__dim__"])
    if set(value) == {"__expr__"}:
        return expr_from_wire(value["__expr__"])
    if set(value) == {"__elementwise__"}:
        return ElementwiseImpl(str(value["__elementwise__"]))
    if set(value) == {"__index_source__"}:
        payload = value["__index_source__"]
        if not isinstance(payload, dict):
            raise ValueError("Torch IR index source must be a mapping")
        _keys(payload, {"input_idx", "coord_map", "select"}, "Torch IR index source")
        return IndexSource(
            input_idx=int(payload["input_idx"]),
            coord_map=tuple(expr_from_wire(expr) for expr in payload.get("coord_map", [])),
            select=expr_from_wire(payload["select"]) if payload.get("select") is not None else None,
        )
    if set(value) == {"__op__"}:
        return op_from_wire(value["__op__"])
    if set(value) == {"__program__"}:
        return graph_from_wire(value["__program__"])
    if set(value) == {"__tuple__"}:
        return tuple(_value_from_wire(item) for item in value["__tuple__"])
    return {key: _value_from_wire(item) for key, item in value.items()}


def op_to_wire(op: Op) -> dict:
    tag = _TAG_BY_CLASS.get(type(op))
    if tag is None:
        raise ValueError(f"Torch IR wire does not support {type(op).__name__}")
    _cls, semantic_fields = _OP_SPECS[tag]
    attrs = {name: _value_to_wire(getattr(op, name)) for name in semantic_fields}
    return {"op": tag, "attrs": attrs}


def op_from_wire(value: object) -> Op:
    if not isinstance(value, dict):
        raise ValueError("Torch IR operation must be a mapping")
    _keys(value, {"op", "attrs"}, "Torch IR operation")
    tag = value.get("op")
    if tag not in _OP_SPECS:
        raise ValueError(f"Torch IR operation has unknown op {tag!r}")
    cls, semantic_fields = _OP_SPECS[tag]
    attrs = value.get("attrs", {})
    if not isinstance(attrs, dict):
        raise ValueError(f"Torch IR operation {tag!r} attrs must be a mapping")
    _keys(attrs, set(semantic_fields), f"Torch IR operation {tag!r}")
    decoded = {name: _value_from_wire(item) for name, item in attrs.items()}
    try:
        return cls(**decoded) if decoded else cls()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Torch IR operation {tag!r} is invalid: {exc}") from exc


def _tensor_to_wire(tensor: Tensor) -> dict:
    return {
        "name": tensor.name,
        "dtype": tensor.dtype.name,
        "shape": [dim_to_wire(dim) for dim in tensor.shape],
    }


def _tensor_from_wire(value: object) -> Tensor:
    if not isinstance(value, dict):
        raise ValueError("Torch IR tensor must be a mapping")
    _keys(value, {"name", "dtype", "shape"}, "Torch IR tensor")
    return Tensor(
        name=str(value["name"]),
        dtype=str(value["dtype"]),
        shape=tuple(dim_from_wire(dim) for dim in value.get("shape", [])),
    )


def graph_to_wire(graph: Graph) -> dict:
    nodes = []
    for node_id in graph.topological_order():
        node = graph.nodes[node_id]
        encoded = op_to_wire(node.op)
        nodes.append(
            {
                "id": node_id,
                "op": encoded["op"],
                "attrs": encoded["attrs"],
                "inputs": list(node.inputs),
                "outputs": [_tensor_to_wire(tensor) for tensor in node.outputs],
            }
        )
    return {
        "ir_version": IR_VERSION,
        "inputs": list(graph.inputs),
        "outputs": list(graph.outputs),
        "nodes": nodes,
    }


def graph_from_wire(value: object) -> Graph:
    if not isinstance(value, dict):
        raise ValueError("Torch IR program must be a mapping")
    _keys(value, {"ir_version", "inputs", "outputs", "nodes"}, "Torch IR program")
    if value.get("ir_version") != IR_VERSION:
        raise ValueError(f"unsupported Torch IR version {value.get('ir_version')!r}; expected {IR_VERSION}")
    nodes = value.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("Torch IR program nodes must be a list")
    graph = Graph()
    for index, item in enumerate(nodes):
        if not isinstance(item, dict):
            raise ValueError(f"Torch IR node {index} must be a mapping")
        _keys(item, {"id", "op", "attrs", "inputs", "outputs"}, f"Torch IR node {index}")
        node_id = item.get("id")
        if not isinstance(node_id, str) or not node_id:
            raise ValueError(f"Torch IR node {index} requires a non-empty id")
        inputs = item.get("inputs")
        outputs = item.get("outputs")
        if not isinstance(inputs, list) or not all(isinstance(name, str) for name in inputs):
            raise ValueError(f"Torch IR node {node_id!r} inputs must be string names")
        if not isinstance(outputs, list) or not outputs:
            raise ValueError(f"Torch IR node {node_id!r} outputs must be a non-empty list")
        op = op_from_wire({"op": item.get("op"), "attrs": item.get("attrs", {})})
        tensors = tuple(_tensor_from_wire(output) for output in outputs)
        try:
            graph.add_node(op, list(inputs), outputs=tensors, node_id=node_id)
        except ValueError as exc:
            raise ValueError(f"Torch IR node {node_id!r} is invalid: {exc}") from exc
    graph.inputs = list(value.get("inputs", []))
    graph.outputs = list(value.get("outputs", []))
    for name in (*graph.inputs, *graph.outputs):
        if graph.buffer(name) is None:
            raise ValueError(f"Torch IR program references unknown boundary buffer {name!r}")
    graph.topological_order()  # validate acyclicity
    return graph


def canonical_program_bytes(program: dict) -> bytes:
    """Canonical semantic bytes used for content addressing."""
    graph_from_wire(program)  # reject malformed/non-Torch payloads before hashing
    return json.dumps(program, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def program_id(program: dict) -> str:
    return PROGRAM_PREFIX + hashlib.sha256(canonical_program_bytes(program)).hexdigest()


def intern_program(programs: dict[str, dict], graph: Graph) -> str:
    payload = graph_to_wire(graph)
    key = program_id(payload)
    current = programs.setdefault(key, payload)
    if canonical_program_bytes(current) != canonical_program_bytes(payload):  # pragma: no cover - SHA collision guard
        raise ValueError(f"Torch IR program digest collision for {key}")
    return key


def validate_program_pool(programs: object) -> dict[str, dict]:
    if not isinstance(programs, dict):
        raise ValueError("golden programs must be a mapping")
    out: dict[str, dict] = {}
    for key, payload in programs.items():
        if not isinstance(key, str) or not key.startswith(PROGRAM_PREFIX):
            raise ValueError(f"golden program id must start with {PROGRAM_PREFIX!r}, got {key!r}")
        if not isinstance(payload, dict):
            raise ValueError(f"golden program {key} must be a mapping")
        actual = program_id(payload)
        if actual != key:
            raise ValueError(f"golden program digest mismatch: key {key}, payload {actual}")
        out[key] = payload
    return out


def supported_op_tags() -> tuple[str, ...]:
    return tuple(_OP_SPECS)
