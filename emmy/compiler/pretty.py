"""Render a tensor graph as Rust-like pseudocode.

The graph stages (``--ir torch`` / ``--ir tensor``) print as one program: the
constants become module-level bindings, the graph's inputs become the
parameters of ``pub fn main``, every compute node becomes a ``let``, and the
graph's outputs become the tail expression. Types carry the shape
(``f16[1,512,4096]``), so a reader checks a line by reading its type rather
than a trailing shape column.

Names that emmy owns carry an ``emmy::`` prefix; names that torch or numpy
already mean print bare. See ``IR-PSEUDOCODE-TORCH.md`` for the semantics.
"""

from __future__ import annotations

from dataclasses import fields as dc_fields

from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.elementwise import _NAME_TO_FN
from emmy.compiler.ir.expr import PLACEHOLDER_PREFIX, Var
from emmy.compiler.ir.tensor.ir import BitcastOp, CastOp, ElementwiseOp, GatherOp, IndexMapOp, ReduceOp, ScanOp

_DIM_NAMES = ("i", "j", "k", "l", "m", "n")
# Elementwise names torch or numpy already means. Everything else in
# ``elementwise.py``'s table is one of emmy's own intrinsics and prints under
# ``emmy::intrinsics::``, so a new entry there needs no edit here.
_BORROWED_ELEMENTWISE = frozenset({"copy", "where", "arange", "rsqrt", "relu", "sigmoid", "silu", "softplus", "erf", "gelu"})
# Attribution metadata every op carries; never part of what an op computes.
_SKIP_FIELDS = frozenset({"source", "knobs", "inputs", "outputs"})
_WRAP = 118
# Class names whose lowercased form is not the torch spelling.
_OP_SPELLING = {"RmsNormOp": "rms_norm", "LayerNormOp": "layer_norm"}


def fmt_type(shape: tuple, dtype) -> str:
    """``f16[1,512,4096]``; a rank-0 tensor is just its dtype."""
    return str(dtype) if not shape else f"{dtype}[{','.join(str(d) for d in shape)}]"


def _fields(op) -> list[str]:
    return [f"{f.name}={getattr(op, f.name)}" for f in dc_fields(op) if f.name not in _SKIP_FIELDS and f.name != "name"]


def _binders(op: IndexMapOp) -> list[str]:
    """One name per output axis; unused axes take Rust's ``_`` prefix."""
    used: set[str] = set()
    for src in op.sources:
        for e in (*src.coord_map, *(() if src.select is None else (src.select,))):
            used |= e.free_vars()
    return [(n if f"{PLACEHOLDER_PREFIX}{d}" in used else f"_{n}") for d, n in enumerate(_DIM_NAMES[: len(op.out_shape)])]


def _index_expr(op: IndexMapOp, names: list[str], arg_names: list[str]) -> str:
    src = op.sources[0]
    rename = {f"{PLACEHOLDER_PREFIX}{d}": Var(n.lstrip("_")) for d, n in enumerate(names)}
    coords = ", ".join(e.substitute(rename).pretty() for e in src.coord_map)
    return f"{arg_names[src.input_idx]}[{coords}]"


def fmt_expr(node, graph, names: dict[str, str]) -> str:
    """The right-hand side of one ``let``."""
    op = node.op
    args = [names.get(i, i) for i in node.inputs]
    out = node.output
    src_t = graph.buffer(node.inputs[0]) if node.inputs else None
    same_dtype = src_t is not None and str(src_t.dtype) == str(out.dtype)

    if isinstance(op, ElementwiseOp) and op.name == "copy" and len(args) == 1:
        # The identity. Elementwise preserves shape (``Graph.validate`` enforces it),
        # so a copy either changes the declared dtype — a conversion — or nothing.
        return args[0] if same_dtype else f"emmy::cast({args[0]})"
    if isinstance(op, ElementwiseOp):
        prefix = "emmy::intrinsics::" if op.name in _NAME_TO_FN and op.name not in _BORROWED_ELEMENTWISE else ""
        return f"{prefix}{op.name}({', '.join(args)})"
    if isinstance(op, (ReduceOp, ScanOp)):
        name = op.name if isinstance(op, ReduceOp) else f"scan_{op.name}"
        return f"{name}({', '.join(args)}, axis={op.axis})"
    if isinstance(op, CastOp):
        return f"emmy::cast({args[0]})"
    if isinstance(op, BitcastOp):
        return f"emmy::bitcast({args[0]})"
    if isinstance(op, IndexMapOp):
        if src_t is not None and op.is_identity(tuple(src_t.shape)):
            return args[0] if same_dtype else f"emmy::cast({args[0]})"
        if len(op.sources) == 1 and op.sources[0].select is None and len(op.out_shape) <= len(_DIM_NAMES):
            names = _binders(op)
            return f"emmy::tensor_from_fn(|{', '.join(names)}| {_index_expr(op, names, args)})"
        return f"emmy::index_map({', '.join(args)})"
    if isinstance(op, GatherOp):
        # One op class, two operations. The operand shapes decide which, by the same
        # test the evaluator applies. ``gather`` picks one element per output
        # position, which is what torch means by the name. ``emmy::gather_by_axis``
        # looks up whole slices by index — torch spells that ``index_select`` or
        # ``embedding``, neither of which covers both, so the name is emmy's.
        data, idx = (graph.buffer(i) for i in node.inputs[:2])
        axis = op.axis
        if data is not None and idx is not None:
            if isinstance(axis, int) and axis < 0:
                axis += len(data.shape)
            per_element = len(idx.shape) == len(data.shape) and all(
                e == d for k, (e, d) in enumerate(zip(idx.shape, data.shape, strict=True)) if k != axis
            )
            name = "gather" if per_element else "emmy::gather_by_axis"
            return f"{name}({', '.join(args)}, axis={axis})"
        return f"emmy::gather({', '.join(args)}, axis={op.axis})"

    name = _OP_SPELLING.get(type(op).__name__, type(op).__name__.removesuffix("Op").lower())
    if (body := op.pretty_body()) is not None:
        # Loop / tile / kernel ops: their fields are whole statement trees, so the
        # dialect's own body rendering reads where a dataclass repr does not.
        nested = "\n".join(f"    {line}" for line in body.splitlines())
        return f"{name}({', '.join(args)}) {{\n{nested}\n    }}"
    return f"{name}({', '.join([*args, *_fields(op)])})"


def fmt_constant(node) -> str:
    """Where a constant's value comes from: a literal, the launch context, or the checkpoint."""
    op: ConstantOp = node.op
    if op.value is not None:
        base = repr(op.value)
    elif getattr(op, "context_value", None) is not None:
        base = f"context({op.context_value.pretty()})"
    elif op.source_path:
        base = f'load("{op.source_path}")'
    elif getattr(op, "source_parts", ()):
        base = "load(" + ", ".join(f'"{p}"' for p, _ in op.source_parts) + ")"
    elif getattr(op, "source_graph", None) is not None:
        base = "emmy::const_eval()"
    else:
        base = "emmy::input_data()"
    for load_op in getattr(op, "load_ops", ()):
        base += f".{type(load_op).__name__.removesuffix('Op').lower()}({', '.join(_fields(load_op))})"
    return base


def _let(name: str, ty: str, rhs: str, indent: str, tensor_name: str = "") -> list[str]:
    """One binding. The name is the node id; a tensor name that differs from it
    still carries information (``broadcast_to`` names its result after its
    source), so it rides along in a trailing comment."""
    note = f"  // aka tensor name `{tensor_name}`" if tensor_name and tensor_name != name else ""
    line = f"{indent}let {name}: {ty} = {rhs};"
    if len(line) <= _WRAP:
        return [line + note]
    return [f"{indent}let {name}: {ty}", f"{indent}    = {rhs};{note}"]


def render_graph(graph) -> str:
    order = graph.topological_order()
    # A binding name is the node id — the graph's own edge key, unique by
    # construction. Two nodes can share one tensor name (``broadcast_to`` names
    # its result after its source, so one table broadcast to two head counts
    # yields the name twice), so a differing tensor name rides in a comment.
    names = {nid: nid for nid in order}
    outs = [(names.get(b, b), graph.buffer(b)) for b in graph.outputs]

    lines = [f"// {len(graph.nodes)} nodes, {len(graph.inputs)} inputs, {len(graph.outputs)} outputs", ""]

    consts = [n for n in order if isinstance(graph.nodes[n].op, ConstantOp)]
    for nid in consts:
        t = graph.nodes[nid].output
        lines += _let(names[nid], fmt_type(t.shape, t.dtype), fmt_constant(graph.nodes[nid]), "", t.name)
    if consts:
        lines.append("")

    ret = "Outputs" if len(outs) > 1 else fmt_type(outs[0][1].shape, outs[0][1].dtype) if outs else "()"

    lines.append("pub fn main(")
    for buf in graph.inputs:
        t = graph.buffer(buf)
        lines.append(f"    {names.get(buf, buf)}: {fmt_type(t.shape, t.dtype)},")
    lines.append(f") -> {ret} {{")

    for nid in (n for n in order if not isinstance(graph.nodes[n].op, (InputOp, ConstantOp))):
        node = graph.nodes[nid]
        t = node.output
        lines += _let(names[nid], fmt_type(t.shape, t.dtype), fmt_expr(node, graph, names), "    ", t.name)

    lines.append("")
    tail = "Outputs { " + ", ".join(n for n, _ in outs) + " }" if len(outs) > 1 else outs[0][0] if outs else "()"
    lines += [f"    {tail}", "}"]
    if len(outs) > 1:
        # Declared after the function, next to its only use.
        lines += ["", "struct Outputs {", *[f"    {n}: {fmt_type(t.shape, t.dtype)}," for n, t in outs], "}"]
    return "\n".join(lines)
