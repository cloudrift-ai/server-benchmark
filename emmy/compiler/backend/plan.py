"""Execution plan — the serializable runtime projection of a compiled graph.

The plan is everything the runtime (``backend/cuda/program.py``) needs to allocate buffers and
launch kernels, with no reference to the compiler IR that produced it: buffer specs, scalar /
runtime constants, the launch list, symbolic-axis plumbing, kernel refs (source and/or a
content-addressed cubin-cache key), and per-weight checkpoint bindings. ``plan_from_graph``
projects a lowered ``Graph[CudaOp]`` into a plan; the JSON form (``plan_to_dict`` /
``plan_from_dict``) is the on-disk pack format (``backend/pack.py``).

Stack independence: the JSON schema carries only plain data plus a tiny expression grammar —
``int`` literal, ``"name"`` var, ``[op, lhs, rhs]`` with ``op ∈ {+, -, *, /, //, %}`` — so a
stored plan survives compiler changes; only runtime-contract changes (arg passing, buffer
semantics) bump ``PLAN_FORMAT_VERSION``. CUDA-specific launch fields (TMA descriptors) nest
under a ``"cuda"`` key so another backend can add its own namespace.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from emmy.compiler import dtype as dtypes
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import DataType
from emmy.compiler.ir.cuda import TmaDescMeta
from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal, Var

if TYPE_CHECKING:
    from emmy.compiler.graph import Graph

logger = logging.getLogger(__name__)

# Version of the runtime contract a stored plan assumes (arg-passing convention, zero_outputs
# semantics, bf16-bits buffer encoding, TMA descriptor construction). Compiler-side changes do
# NOT bump this — a stored plan keeps serving its frozen snapshot; bump only when the runtime's
# interpretation of the plan changes.
PLAN_FORMAT_VERSION = 1

# Binary ops the on-disk expression grammar admits. Everything a ``Dim`` shape, a ceil-div grid
# factor, or a runtime-constant expr can contain; anything else fails serialization loudly.
_EXPR_OPS = ("+", "-", "*", "/", "//", "%")


@dataclass
class BufferSpec:
    """One named device buffer: shape (``Dim``-valued — static ``Literal``, atomic symbolic
    ``Var``, or composite ``BinaryExpr``), dtype, and planning role."""

    name: str
    shape: tuple[Dim, ...]
    dtype: DataType
    role: str  # "input" | "constant" | "output" | "scratch"

    @property
    def is_symbolic(self) -> bool:
        return any(not d.is_static for d in self.shape)

    def resolve_shape(self, sym_values: dict[str, int]) -> tuple[int, ...]:
        return tuple(int(d.expr.eval(sym_values)) for d in self.shape)


@dataclass
class LaunchSpec:
    """One kernel launch site, in program order. ``grid`` / ``block`` are per-dim factor tuples
    whose entries are ``int`` (static), ``str`` (symbolic axis name resolved from sym_values),
    or ``Expr`` (composite extent, e.g. ceil-div) — see ``ir.cuda.resolve_dim``."""

    node_id: str
    kernel_name: str
    arg_names: tuple[str, ...]
    grid: tuple
    block: tuple
    smem_bytes: int
    zero_outputs: tuple[str, ...]
    tma_descriptors: tuple[TmaDescMeta, ...] = ()
    runtime_args: tuple[str, ...] = ()


@dataclass
class KernelSpec:
    """How to obtain one kernel's device code: ``source`` (compile via the cubin cache — the
    from-graph path) and/or ``binary_key`` (a content-addressed cubin-cache key — the pack
    path). A pack stores only the key; a missing cubin falls back to full recompilation."""

    source: str | None = None
    binary_key: str | None = None
    uses_tma: bool = False


@dataclass
class WeightSpec:
    """Checkpoint binding for one constant buffer: the ``state_dict`` key it loads from and the
    encoded load-op chain (the plan's own vocabulary, applied with pure numpy — see
    ``apply_weight_loads``). ``load_ops`` is ``None`` when the graph carried a load op the
    grammar can't express — such a plan still runs, but can't rebind weights from a pack."""

    source_path: str
    load_ops: tuple[tuple, ...] | None = ()


@dataclass
class ExecutionPlan:
    """The pure-data half of a compiled program. ``_Compiled`` (the live runtime object) is this
    plus loaded kernel modules; everything else the runtime builds (buffers, slab layout, TMA
    descriptors, captured graphs) derives from these fields plus the live GPU."""

    backend: str
    inputs: list[str]
    outputs: list[str]
    buffers: list[BufferSpec]
    constants: dict[str, float]
    runtime_constants: dict[str, Expr]
    launches: list[LaunchSpec]
    kernels: dict[str, KernelSpec]
    weights: dict[str, WeightSpec] = field(default_factory=dict)
    symbolic_bindings: dict[str, tuple[str, int]] = field(default_factory=dict)
    symbolic_hints: dict[str, int] = field(default_factory=dict)
    symbolic_caps: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Graph → plan projection
# ---------------------------------------------------------------------------


def plan_from_graph(graph: Graph) -> ExecutionPlan:
    """Project a lowered ``Graph[CudaOp]`` into an :class:`ExecutionPlan`.

    This is the seam the whole runtime builds from: after this call nothing reads the graph
    again. Kernel specs carry the rendered source (the cubin-cache key input); grid/block specs
    are normalized to factor tuples (a legacy bare-int spec becomes a single-factor tuple) so a
    plan round-trips the JSON form exactly."""
    from emmy.compiler.ir.base import ConstantOp, InputOp  # noqa: PLC0415
    from emmy.compiler.ir.cuda import CudaOp  # noqa: PLC0415

    buffers = [
        BufferSpec(name=nid, shape=tuple(node.output.shape), dtype=node.output.dtype, role=graph.node_role(nid))
        for nid, node in graph.nodes.items()
    ]
    constants = {nid: float(op.value) for nid, op in graph.constant_ops() if op.value is not None}
    runtime_constants = {nid: op.context_value for nid, op in graph.constant_ops() if op.context_value is not None}

    kernels: dict[str, KernelSpec] = {}
    launches: list[LaunchSpec] = []
    for nid in graph.topological_order():
        node = graph.nodes[nid]
        op = node.op
        if isinstance(op, (InputOp, ConstantOp)):
            continue
        if not isinstance(op, CudaOp):
            raise TypeError(f"plan_from_graph: node {nid!r} has non-CudaOp {type(op).__name__!r}; lowering must produce Graph[CudaOp].")
        spec = kernels.get(op.kernel_name)
        if spec is None:
            kernels[op.kernel_name] = KernelSpec(source=op.kernel_source, uses_tma=bool(op.tma_descriptors))
        elif spec.source != op.kernel_source:
            raise ValueError(f"kernel name {op.kernel_name!r} used by two distinct sources")
        launches.append(
            LaunchSpec(
                node_id=nid,
                kernel_name=op.kernel_name,
                arg_names=tuple(op.arg_order),
                grid=tuple(_normalize_spec(s) for s in op.grid),
                block=tuple(_normalize_spec(s) for s in op.block),
                smem_bytes=op.smem_bytes,
                zero_outputs=tuple(op.zero_outputs),
                tma_descriptors=tuple(op.tma_descriptors),
                runtime_args=tuple(getattr(op, "runtime_args", ())),
            )
        )

    weights: dict[str, WeightSpec] = {}
    for nid, op in graph.loadable_constants():
        weights[nid] = WeightSpec(source_path=op.source_path, load_ops=_encode_load_ops(op.load_ops))

    return ExecutionPlan(
        backend="cuda",
        inputs=list(graph.inputs),
        outputs=list(graph.outputs),
        buffers=buffers,
        constants=constants,
        runtime_constants=runtime_constants,
        launches=launches,
        kernels=kernels,
        weights=weights,
        symbolic_bindings=graph.symbolic_bindings(),
        symbolic_hints=graph.symbolic_hints(),
        symbolic_caps={},
    )


def _normalize_spec(spec) -> tuple:
    """A grid/block dim spec as a factor tuple: legacy bare-int specs become one-factor tuples,
    and scalar ``Expr`` factors collapse to their plain form (``Literal`` int → ``int``,
    ``Var`` → ``str``) — identical at resolve time, and it keeps the plan's JSON round-trip
    exact (the grammar reserves the list form for genuinely composite factors)."""
    if isinstance(spec, int):
        return (spec,)
    out = []
    for f in spec:
        if isinstance(f, Literal) and isinstance(f.value, int):
            out.append(int(f.value))
        elif isinstance(f, Var):
            out.append(f.name)
        else:
            out.append(f)
    return tuple(out)


# ---------------------------------------------------------------------------
# Weight load-op vocabulary (pure numpy — no IR classes at apply time)
# ---------------------------------------------------------------------------


def _encode_load_ops(load_ops: tuple) -> tuple[tuple, ...] | None:
    """Encode a ``ConstantOp.load_ops`` chain into the plan's vocabulary
    (``("transpose", axes)`` / ``("reshape", shape)``), or ``None`` when a chain member isn't
    expressible — the plan still runs (binding came from the live module), it just can't rebind
    that weight from a pack."""
    from emmy.compiler.ir.frontend.ir import ReshapeOp, TransposeOp  # noqa: PLC0415

    out: list[tuple] = []
    for op in load_ops:
        if isinstance(op, TransposeOp):
            out.append(("transpose", tuple(int(a) for a in op.axes)))
        elif isinstance(op, ReshapeOp) and all(isinstance(d, int) for d in op.shape):
            out.append(("reshape", tuple(op.shape)))
        else:
            logger.warning("plan: load op %r not expressible in the pack vocabulary; weight will not rebind from a pack", op)
            return None
    return tuple(out)


def apply_weight_loads(source: np.ndarray, load_ops: tuple[tuple, ...]) -> np.ndarray:
    """Apply an encoded load-op chain to a raw checkpoint tensor — the pack-side counterpart of
    ``loader.binder.apply_load_ops``, matching its semantics (a 2-tuple transpose is a swap, à
    la ``aten.transpose``; longer tuples are full permutations)."""
    a = np.ascontiguousarray(source)
    for kind, arg in load_ops:
        if kind == "transpose":
            a = np.swapaxes(a, arg[0] % a.ndim, arg[1] % a.ndim) if len(arg) == 2 else np.transpose(a, arg)
        elif kind == "reshape":
            a = a.reshape(arg)
        else:
            raise ValueError(f"apply_weight_loads: unknown load op kind {kind!r}")
    return np.ascontiguousarray(a)


# ---------------------------------------------------------------------------
# Expression grammar — the only non-plain data in the JSON schema
# ---------------------------------------------------------------------------


def _expr_to_json(e):
    """``Expr`` → JSON prefix form: ``int``/``float`` literal, ``"name"`` var, ``[op, l, r]``."""
    if isinstance(e, Literal):
        if isinstance(e.value, int) and e.dtype == "int":
            return int(e.value)
        return float(e.value)
    if isinstance(e, Var):
        return e.name
    if isinstance(e, BinaryExpr) and e.op in _EXPR_OPS:
        return [e.op, _expr_to_json(e.left), _expr_to_json(e.right)]
    raise ValueError(f"plan: expression {e!r} not expressible in the plan grammar (ops {_EXPR_OPS})")


def _expr_from_json(v) -> Expr:
    if isinstance(v, bool):
        raise ValueError(f"plan: invalid expression literal {v!r}")
    if isinstance(v, int):
        return Literal(v, "int")
    if isinstance(v, float):
        return Literal(v)
    if isinstance(v, str):
        return Var(v)
    if isinstance(v, list) and len(v) == 3 and v[0] in _EXPR_OPS:
        return BinaryExpr(v[0], _expr_from_json(v[1]), _expr_from_json(v[2]))
    raise ValueError(f"plan: malformed expression {v!r}")


def _factor_to_json(f):
    """Grid/block factor → JSON: ``int`` and ``str`` (symbolic name) pass through, ``Expr``
    becomes the grammar's list form."""
    if isinstance(f, (int, str)):
        return f
    return [*_expr_to_json_as_list(f)]


def _expr_to_json_as_list(e) -> list:
    v = _expr_to_json(e)
    # A factor that is an Expr must serialize as a list so deserialization can tell it apart
    # from the plain int/str factor forms (a bare Var/Literal factor never reaches here — the
    # lowering emits those as str/int directly).
    if not isinstance(v, list):
        raise ValueError(f"plan: grid factor {e!r} reduces to a scalar — expected int/str factor instead")
    return v


def _factor_from_json(v):
    if isinstance(v, (int, str)):
        return v
    return _expr_from_json(v)


def _dim_to_json(d: Dim):
    return _expr_to_json(d.expr)


def _dim_from_json(v) -> Dim:
    return Dim(_expr_from_json(v))


# ---------------------------------------------------------------------------
# Plan ⇄ dict (the JSON pack format)
# ---------------------------------------------------------------------------


def plan_to_dict(plan: ExecutionPlan) -> dict:
    return {
        "format": PLAN_FORMAT_VERSION,
        "backend": plan.backend,
        "inputs": list(plan.inputs),
        "outputs": list(plan.outputs),
        "buffers": [
            {"name": b.name, "shape": [_dim_to_json(d) for d in b.shape], "dtype": b.dtype.name, "role": b.role} for b in plan.buffers
        ],
        "constants": dict(plan.constants),
        "runtime_constants": {nid: _expr_to_json(e) for nid, e in plan.runtime_constants.items()},
        "launches": [
            {
                "node_id": lc.node_id,
                "kernel": lc.kernel_name,
                "args": list(lc.arg_names),
                "grid": [[_factor_to_json(f) for f in spec] for spec in lc.grid],
                "block": [[_factor_to_json(f) for f in spec] for spec in lc.block],
                "smem": lc.smem_bytes,
                "zero_outputs": list(lc.zero_outputs),
                "runtime_args": list(lc.runtime_args),
                "cuda": {
                    "tma": [
                        {"name": t.name, "src_buf": t.src_buf, "box_extents": list(t.box_extents), "swizzle": t.swizzle}
                        for t in lc.tma_descriptors
                    ]
                },
            }
            for lc in plan.launches
        ],
        "kernels": {
            name: {
                **({"source": spec.source} if spec.source is not None else {}),
                **({"binary_key": spec.binary_key} if spec.binary_key is not None else {}),
                "uses_tma": spec.uses_tma,
            }
            for name, spec in plan.kernels.items()
        },
        "weights": {
            nid: {"path": w.source_path, **({"ops": [[k, list(a)] for k, a in w.load_ops]} if w.load_ops is not None else {})}
            for nid, w in plan.weights.items()
        },
        "symbols": {
            "bindings": {name: [buf, idx] for name, (buf, idx) in plan.symbolic_bindings.items()},
            "hints": dict(plan.symbolic_hints),
            "caps": dict(plan.symbolic_caps),
        },
    }


def plan_from_dict(d: dict) -> ExecutionPlan:
    fmt = d.get("format")
    if fmt != PLAN_FORMAT_VERSION:
        raise ValueError(f"plan format {fmt!r} unsupported (runtime speaks {PLAN_FORMAT_VERSION})")
    symbols = d.get("symbols", {})
    return ExecutionPlan(
        backend=d["backend"],
        inputs=list(d["inputs"]),
        outputs=list(d["outputs"]),
        buffers=[
            BufferSpec(
                name=b["name"],
                shape=tuple(_dim_from_json(v) for v in b["shape"]),
                dtype=dtypes.get(b["dtype"]),
                role=b["role"],
            )
            for b in d["buffers"]
        ],
        constants={nid: float(v) for nid, v in d.get("constants", {}).items()},
        runtime_constants={nid: _expr_from_json(v) for nid, v in d.get("runtime_constants", {}).items()},
        launches=[
            LaunchSpec(
                node_id=lc["node_id"],
                kernel_name=lc["kernel"],
                arg_names=tuple(lc["args"]),
                grid=tuple(tuple(_factor_from_json(f) for f in spec) for spec in lc["grid"]),
                block=tuple(tuple(_factor_from_json(f) for f in spec) for spec in lc["block"]),
                smem_bytes=int(lc["smem"]),
                zero_outputs=tuple(lc.get("zero_outputs", ())),
                tma_descriptors=tuple(
                    TmaDescMeta(name=t["name"], src_buf=t["src_buf"], box_extents=tuple(t["box_extents"]), swizzle=t["swizzle"])
                    for t in lc.get("cuda", {}).get("tma", ())
                ),
                runtime_args=tuple(lc.get("runtime_args", ())),
            )
            for lc in d["launches"]
        ],
        kernels={
            name: KernelSpec(source=spec.get("source"), binary_key=spec.get("binary_key"), uses_tma=bool(spec.get("uses_tma", False)))
            for name, spec in d.get("kernels", {}).items()
        },
        weights={
            nid: WeightSpec(
                source_path=w["path"],
                load_ops=tuple((k, tuple(a)) for k, a in w["ops"]) if "ops" in w else None,
            )
            for nid, w in d.get("weights", {}).items()
        },
        symbolic_bindings={name: (buf, int(idx)) for name, (buf, idx) in symbols.get("bindings", {}).items()},
        symbolic_hints={name: int(v) for name, v in symbols.get("hints", {}).items()},
        symbolic_caps={name: int(v) for name, v in symbols.get("caps", {}).items()},
    )
