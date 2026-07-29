"""Graph container + ``Hints`` metadata.

The graph is a directed acyclic dataflow graph (Kahn-style) over tensor
values. Each ``Node`` wraps one primitive ``Op`` subclass, so nodes are
parameterized by their op type — a ``Node[ElementwiseOp]`` is statically
distinguishable from a ``Node[ReduceOp]``. The structural compiler IR
(``LoopOp`` trees in ``ir/block.py``) relies on that distinction to carry
typed chains like ``tuple[Node[ElementwiseOp], ...]``.

The same ``Graph`` hosts nodes from every IR level as the rewriter runs:
frontend ops during tracing → tensor/minimal ops after decomposition →
``LoopOp`` nodes after fusion. Shapes, inputs, outputs, and hints live
on nodes; ops themselves are shape-free and shared-free by convention.

``Hints`` is the advisory metadata bag attached to individual nodes and
to the graph as a whole. It lives here because ``Node`` and ``Graph`` are
its only users.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from emmy.compiler.ir.base import ConstantOp, InputOp, Op
from emmy.compiler.tensor import Tensor

# ---------------------------------------------------------------------------
# Hints
# ---------------------------------------------------------------------------


@dataclass
class Hints:
    """Advisory metadata bag keyed by dotted namespace strings.

    Hints do not affect computation semantics — backends may ignore unknown
    hints. Keys use dotted namespaces (e.g. ``cuda.matmul.strategy``).
    """

    _data: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for *key*, or *default* if absent."""
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a hint value."""
        self._data[key] = value

    def has(self, key: str) -> bool:
        """Return whether *key* is present."""
        return key in self._data

    def remove(self, key: str) -> None:
        """Remove *key* if present."""
        self._data.pop(key, None)

    def merge(self, other: Hints) -> None:
        """Merge *other* into self.  Other's values win on conflict."""
        self._data.update(other._data)

    def prefix(self, ns: str) -> dict[str, Any]:
        """Return all hints under *ns* as a flat dict with the prefix stripped.

        >>> h = Hints(); h.set("cuda.matmul.strategy", "naive")
        >>> h.prefix("cuda.matmul")
        {'strategy': 'naive'}
        """
        dot = ns if ns.endswith(".") else ns + "."
        return {k[len(dot) :]: v for k, v in self._data.items() if k.startswith(dot)}

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return dict(self._data)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Hints:
        """Deserialize from a dict."""
        return Hints(_data=dict(data))

    def __bool__(self) -> bool:
        return bool(self._data)

    def __repr__(self) -> str:
        return f"Hints({self._data!r})"


def resolve_hints(graph: Graph, node_id: str) -> Hints:
    """Merge graph-level hints with node-level hints (node wins)."""
    merged = Hints()
    merged.merge(graph.hints)
    merged.merge(graph.nodes[node_id].hints)
    return merged


# ---------------------------------------------------------------------------
# Tensor + Node + Graph
# ---------------------------------------------------------------------------


@dataclass
class Node[T_Op: Op]:
    """A single operation in the compute graph.

    Generic on the op type (PEP 695) so consumers can narrow to
    ``Node[ElementwiseOp]`` or ``Node[ReduceOp]`` where the surrounding
    structure demands it (e.g. a body-chain slot that admits only
    elementwise ops, or a ``ReduceStage`` whose ``reduce`` field must be
    a reduction). The parameter is erased at runtime; owning dataclasses
    enforce the invariant via ``__post_init__``.

    ``outputs`` is the ordered, non-empty tuple of buffers this node's
    kernel writes. Slot 0 is the **primary** output — its buffer travels
    under the node's id (``node.id == outputs[0].name`` in steady state),
    which is what :attr:`output` returns, so single-output call sites
    read exactly what they always did. Non-primary buffers (slot ≥ 1)
    are addressed by their own tensor names (``<primary>__sq`` style).
    """

    id: str
    op: T_Op
    inputs: list[str]  # buffer names (the producing node's id for primary outputs)
    outputs: tuple[Tensor, ...]
    hints: Hints = field(default_factory=Hints)

    @property
    def output(self) -> Tensor:
        """The primary output (slot 0). Single-output nodes — the vast
        majority — read this exactly as before the MIMO migration."""
        return self.outputs[0]

    def buffer_names(self) -> tuple[str, ...]:
        """The edge keys this node's outputs travel under: the node id
        for slot 0, each tensor's own name for slots ≥ 1."""
        return (self.id, *(t.name for t in self.outputs[1:]))


def _lookup_op_class(name: str) -> type[Op] | None:
    """Find an Op subclass by name across all IR dialect modules.

    Used by ``Graph.from_dict`` to reconstruct nodes. Lazy-imports each
    module to avoid pulling them all in at graph import time.
    """
    from emmy.compiler.ir import base as _base
    from emmy.compiler.ir.cuda import ir as _cuda
    from emmy.compiler.ir.frontend import ir as _frontend
    from emmy.compiler.ir.kernel import ir as _kernel
    from emmy.compiler.ir.loop import ir as _loop
    from emmy.compiler.ir.tensor import ir as _tensor
    from emmy.compiler.ir.tile import ir as _tile

    for module in (_base, _tensor, _frontend, _loop, _kernel, _cuda, _tile):
        cls = getattr(module, name, None)
        if isinstance(cls, type) and issubclass(cls, Op):
            return cls


def _serialize_field(v):
    """Flatten non-JSON-friendly op field values to a JSON-compatible form.

    Special cases:

    - ``ir.elementwise.ElementwiseImpl`` → its ``name`` string.
    - ``ir.stmt.Body`` → the underlying ``stmts`` tuple. Body is an
      in-memory wrapper; on disk we keep the tuple-of-stmts form so
      ``_deserialize_field``'s list-of-Stmt-reprs path keeps working
      unchanged. ``LoopOp.__post_init__`` / ``TileOp.__post_init__``
      coerce the tuple back to Body on load.
    - Nested ``Op`` instances (e.g. ``ConstantOp.load_ops``) →
      ``{"__op__": ClassName, "fields": {...}}`` dicts; tuples/lists of
      Ops recurse element-wise.

    Everything else passes through. ``_deserialize_field`` reverses
    this on load.
    """
    from emmy.compiler.dim import Dim
    from emmy.compiler.ir.elementwise import ElementwiseImpl
    from emmy.compiler.ir.stmt import Body
    from emmy.compiler.ir.tensor.ir import IndexSource

    # ``IndexSource`` (IndexMapOp.sources) is a plain dataclass holding ``Expr``
    # objects — serialize as eval-able repr strings, the same round-trip the
    # body-Stmt path uses, so it survives JSON instead of being stringified by
    # ``json.dumps(default=str)`` (which ``_deserialize_field`` couldn't reverse).
    if isinstance(v, IndexSource):
        return repr(v)
    if isinstance(v, (list, tuple)) and v and all(isinstance(x, IndexSource) for x in v):
        return [repr(x) for x in v]
    if isinstance(v, ElementwiseImpl):
        return v.name
    if isinstance(v, Dim):
        return v.value
    if isinstance(v, Body):
        # Body is a ``tuple`` subclass; downcast to plain tuple so
        # JSON encodes element-by-element rather than via ``__repr__``
        # (which would produce a single ``"Body((...))"`` string).
        return tuple(v)
    if isinstance(v, Op):
        return {
            "__op__": type(v).__name__,
            "fields": {k: _serialize_field(x) for k, x in v.__dict__.items() if not k.startswith("_")},
        }
    if isinstance(v, (list, tuple)) and v and all(isinstance(x, Op) for x in v):
        return [_serialize_field(x) for x in v]
    if isinstance(v, (list, tuple)) and v and any(isinstance(x, Dim) for x in v):
        return [_serialize_field(x) for x in v]
    return v


def _deserialize_field(k, v):
    """Reverse of ``_serialize_field``: rehydrate the ``op`` field name
    back to ``ElementwiseImpl``, and eval Stmt-repr strings (produced by
    ``json.dumps(..., default=str)`` against dataclass Stmts) back into
    Stmt instances. Nested-op dicts (``{"__op__": ..., "fields": ...}``)
    are reconstructed via ``_lookup_op_class``. The eval scope mirrors
    the IR's ``__all__`` exports — same classes the Stmt reprs reference."""
    from emmy.compiler.ir.elementwise import ElementwiseImpl

    if k == "op" and isinstance(v, str):
        # A bare name (``"add"``) is an ``ElementwiseImpl``; a constructor repr
        # (``"Map(...)"`` — the ``TileOp.op`` node tree) is eval'd back like a Stmt repr.
        return _eval_stmt(v) if "(" in v else ElementwiseImpl(v)
    # ``TileOp``'s schedule descriptors serialize as constructor-repr strings
    # (``Placement(...)`` / ``TilePlan(...)`` / ``ReducePlan(...)`` / ``Stage(...)`` /
    # ``WarpSpec(...)``); ``None`` fields round-trip as JSON null. Eval a string only when it
    # opens with a constructor whose name is a known IR class — so a plain string field
    # (a name / path / C source) that merely looks like ``Name(...)`` is never eval'd.
    if isinstance(v, str):
        return _maybe_eval_ctor(v)
    if k == "body" and isinstance(v, list) and v and all(isinstance(e, str) for e in v):
        return tuple(_eval_stmt(e) for e in v)
    if k == "sources" and isinstance(v, list) and v and all(isinstance(e, str) and e.startswith("IndexSource(") for e in v):
        return tuple(_eval_stmt(e) for e in v)
    if isinstance(v, dict) and "__op__" in v:
        op_cls = _lookup_op_class(v["__op__"])
        if op_cls is None:
            raise ValueError(f"Unknown nested op class: {v['__op__']}")
        fields = {fk: _deserialize_field(fk, fv) for fk, fv in v.get("fields", {}).items()}
        return op_cls(**fields) if fields else op_cls()
    if isinstance(v, list) and v and all(isinstance(e, dict) and "__op__" in e for e in v):
        return tuple(_deserialize_field(k, e) for e in v)
    if isinstance(v, list):
        # Constructor-repr string *elements* rehydrate under the same known-class guard —
        # e.g. ``CudaOp.tma_descriptors``'s ``TmaDescMeta(...)`` reprs (stringified by the
        # dump's ``json.dumps(default=str)``), which a plain ``tuple(v)`` left as strings.
        return tuple(_maybe_eval_ctor(e) if isinstance(e, str) else e for e in v)
    return v


def _maybe_eval_ctor(s: str):
    """Eval ``s`` back to an IR object iff it opens with a constructor whose name is a known
    IR class — so a plain string field (a name / path / C source) that merely looks like
    ``Name(...)`` is never eval'd. Returns ``s`` unchanged otherwise."""
    import re as _re

    m = _re.match(r"([A-Z]\w*)\(", s)
    return _eval_stmt(s) if m and m.group(1) in _stmt_eval_scope() else s


def _eval_stmt(s: str):
    scope = _stmt_eval_scope()
    # ``repr(enum_value)`` produces ``<SwizzleMode.NONE: 'NONE'>`` which
    # isn't eval-able. Rewrite to the dotted form before eval.
    if "<" in s and ":" in s:
        import re as _re

        s = _re.sub(r"<([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*): [^>]+>", r"\1.\2", s)
    return eval(s, scope)


_STMT_EVAL_SCOPE: dict | None = None


def _lenient_lambda(params=(), body=(), results=()):
    """Rehydrate a dumped ``Lambda`` repr through the interim ``effectful_lambda`` path (see the
    scope entry below — retires at 1q)."""
    from emmy.compiler.ir.stmt.body import Body, effectful_lambda  # noqa: PLC0415

    return effectful_lambda(params, Body.coerce(body), results)


def _stmt_eval_scope() -> dict:
    """Lazy-built eval scope for Stmt-repr strings."""
    global _STMT_EVAL_SCOPE
    if _STMT_EVAL_SCOPE is not None:
        return _STMT_EVAL_SCOPE
    import numpy as _np

    from emmy.compiler.dim import Dim
    from emmy.compiler.dtype import DataType
    from emmy.compiler.ir.axis import Axis, AxisRole
    from emmy.compiler.ir.elementwise import ElementwiseImpl
    from emmy.compiler.ir.expr import (
        BinaryExpr,
        Builtin,
        CastExpr,
        FuncCallExpr,
        Literal,
        TernaryExpr,
        Var,
    )
    from emmy.compiler.ir.kernel.ir import Smem, Sync, TreeHalve, WarpShuffle
    from emmy.compiler.ir.stmt import (
        Accum,
        Assign,
        Carrier,
        Cond,
        Init,
        Load,
        Loop,
        Pack,
        Select,
        SelectBranch,
        State,
        StateMerge,
        StridedLoop,
        Twist,
        Unpack,
        Write,
    )
    from emmy.compiler.ir.stmt.carrier import Channel
    from emmy.compiler.ir.tensor.ir import IndexSource

    _STMT_EVAL_SCOPE = {
        "Dim": Dim,
        "Axis": Axis,
        "Var": Var,
        "Literal": Literal,
        "BinaryExpr": BinaryExpr,
        "Builtin": Builtin,
        "FuncCallExpr": FuncCallExpr,
        "TernaryExpr": TernaryExpr,
        "CastExpr": CastExpr,
        "Load": Load,
        "Pack": Pack,
        "Unpack": Unpack,
        "Assign": Assign,
        "Accum": Accum,
        "Init": Init,
        "Write": Write,
        "Select": Select,
        "SelectBranch": SelectBranch,
        "Loop": Loop,
        "StridedLoop": StridedLoop,
        "Cond": Cond,
        "AxisRole": AxisRole,
        "Carrier": Carrier,
        "State": State,
        "Twist": Twist,
        "StateMerge": StateMerge,
        "Channel": Channel,
        "Smem": Smem,
        "Sync": Sync,
        "TreeHalve": TreeHalve,
        "WarpShuffle": WarpShuffle,
        "ElementwiseImpl": ElementwiseImpl,
        "IndexSource": IndexSource,
        "DataType": DataType,
        # ``repr(np.dtype('float32'))`` is ``dtype('float32')`` — eval needs
        # ``dtype`` in scope to round-trip ``DataType.np``.
        "dtype": _np.dtype,
        # INTERIM (retires at 1q): a dumped ``Map.fn`` lambda may still carry the projection
        # ``Write``s / a split partial's ``Loop`` (the effects that move to the kernel boundary
        # at 1q), so the round-trip rebuilds through the same lenient path ``Map`` construction
        # uses — strict ``Lambda`` formation once the body is pure.
        "Lambda": _lenient_lambda,
        "__builtins__": {},
    }
    # The tile-IR structural nodes (``Map`` / ``Fold`` / ``ContractionView``) and the
    # schedule descriptors (``Placement`` / ``TilePlan`` / ``ReducePlan`` / ``Stage`` /
    # ``WarpSpec`` + their component dataclasses / enums) round-trip through ``TileOp``'s
    # repr-string fields (``op`` / ``place`` / ``reduce`` / ``tier`` / ``stage`` /
    # ``workers``), and the cuda stage's ``CudaOp.tma_descriptors`` round-trips its
    # ``TmaDescMeta`` reprs the same way, so ``emmy run --ir <stage.json>`` can eval them
    # back. Auto-populate every public class from these modules (``setdefault`` so the
    # explicit stmt/expr entries above win on any name clash) — a new node/knob field
    # needs no edit here.
    import emmy.compiler.ir.cuda.ir as _cuda_mod  # noqa: PLC0415
    import emmy.compiler.ir.kernel.ir as _kernel_mod  # noqa: PLC0415
    import emmy.compiler.ir.schedule as _sched_mod  # noqa: PLC0415
    import emmy.compiler.ir.tile.ir as _tile_mod  # noqa: PLC0415

    for _mod in (_sched_mod, _tile_mod, _kernel_mod, _cuda_mod):
        for _nm in dir(_mod):
            _obj = getattr(_mod, _nm)
            if isinstance(_obj, type):
                _STMT_EVAL_SCOPE.setdefault(_nm, _obj)
    return _STMT_EVAL_SCOPE


# Op dataclass fields excluded from ``Graph.structural_key`` for ops that
# don't carry a ``Body``. ``name`` is an instance label; ``source`` is the
# rewrite-chain predecessor on the base ``Op`` (attribution metadata only,
# stamped automatically by the engine — see ``Op.source``).
_STRUCTURAL_SKIP_FIELDS = frozenset({"name", "source", "meta"})

# Op dataclass fields excluded from JSON serialization in :meth:`Graph.to_dict`:
# pure runtime state (``source`` / ``knobs`` chain metadata, ``inputs`` /
# ``outputs`` snapped by the matcher) — none of it belongs in the persisted
# IR.
_SERIALIZE_SKIP_FIELDS = frozenset({"source", "knobs", "inputs", "outputs", "meta"})


class Graph:
    """Directed acyclic compute graph of tensor operations.

    Stores both directions of each edge: every ``Node`` carries its
    backward edges as ``node.inputs`` (producer ids), and the graph
    maintains a forward-edge index ``_users`` (consumer id set per node).
    Mutation methods keep both sides consistent, so forward walks
    (``users`` / ``consumers``) are O(1) per hop.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.inputs: list[str] = []
        self.outputs: list[str] = []
        self.hints: Hints = Hints()
        self._id_counter = itertools.count()
        # Both indexes are keyed by BUFFER name (== node id for primary
        # outputs): ``_users`` maps buffer → consumer node ids, ``_producers``
        # maps buffer → (producing node id, output slot). Mutation methods
        # keep them in lockstep with ``node.inputs`` / ``node.outputs``.
        self._users: dict[str, set[str]] = {}
        self._producers: dict[str, tuple[str, int]] = {}

    def _next_id(self) -> str:
        while True:
            nid = f"n{next(self._id_counter)}"
            if nid not in self.nodes:
                return nid

    def add_node(
        self,
        op: Op,
        inputs: list[Node | str],
        output: Tensor | None = None,
        *,
        outputs: Iterable[Tensor] | None = None,
        node_id: str | None = None,
    ) -> str:
        """Add a node to the graph. Returns the node id.

        ``inputs`` accepts a mix of buffer names and ``Node`` objects
        (Nodes get their ``id`` extracted) — convenient for decomposition
        rules that thread ``Node`` values through pipelines of helpers.

        Exactly one of ``output`` (the single-output convenience form) or
        ``outputs`` (the ordered multi-output form, slot 0 primary) is
        given. Non-primary buffer names must be free — every buffer name
        has exactly one producer graph-wide (SSA per buffer).

        When ``node_id`` is omitted, defaults to the primary output's
        ``name`` if that name is non-empty and not already taken;
        otherwise falls back to an auto-generated ``n<i>`` id. This keeps
        semantic names visible in kernel buf refs (Load.input /
        Write.output) without forcing every caller to repeat
        ``node_id=name``.
        """
        if (output is None) == (outputs is None):
            raise ValueError("add_node: pass exactly one of output= / outputs=")
        outs = (output,) if output is not None else tuple(outputs)
        if not outs:
            raise ValueError("add_node: outputs must be non-empty")
        if node_id is None:
            if outs[0].name and outs[0].name not in self.nodes:
                nid = outs[0].name
            else:
                nid = self._next_id()
        else:
            nid = node_id
        if nid in self.nodes:
            raise ValueError(f"Node id {nid!r} already exists")
        for t in outs[1:]:
            if not t.name:
                raise ValueError(f"add_node: non-primary output of {nid!r} needs a name")
            if t.name in self._producers or t.name in self.nodes:
                raise ValueError(f"Buffer name {t.name!r} already has a producer")
        input_ids = [inp.id if isinstance(inp, Node) else inp for inp in inputs]
        for inp in input_ids:
            if inp not in self._producers and inp not in self.nodes:
                raise ValueError(f"Input buffer {inp!r} does not exist")
        node = Node(id=nid, op=op, inputs=input_ids, outputs=outs)
        self.nodes[nid] = node
        for slot, buf in enumerate(node.buffer_names()):
            self._producers[buf] = (nid, slot)
            self._users.setdefault(buf, set())
        for inp in input_ids:
            self._users[inp].add(nid)
        return nid

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the graph."""
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id!r} not found")
        node = self.nodes[node_id]
        for inp in node.inputs:
            users = self._users.get(inp)
            if users is not None:
                users.discard(node_id)
        bufs = set(node.buffer_names())
        self.inputs = [i for i in self.inputs if i not in bufs]
        self.outputs = [o for o in self.outputs if o not in bufs]
        del self.nodes[node_id]
        for buf in bufs:
            self._producers.pop(buf, None)
            self._users.pop(buf, None)

    def rename_node(self, old_id: str, new_id: str) -> None:
        """Change a node's id in place, updating every reference.

        Updates ``self.nodes`` keying, ``self.inputs`` / ``self.outputs``
        membership, the ``_users`` index, every consumer's ``node.inputs``
        list, and every consumer LoopOp's internal buf refs
        (``Load.source`` / ``Write.output`` track the same identity as
        graph node ids).
        """
        if old_id == new_id:
            return
        if old_id not in self.nodes:
            raise KeyError(f"Node {old_id!r} not found")
        if new_id in self.nodes or new_id in self._producers:
            raise ValueError(f"Node id {new_id!r} already exists")
        node = self.nodes.pop(old_id)
        node.id = new_id
        self.nodes[new_id] = node
        consumers = self._users.pop(old_id, set())
        self._users[new_id] = consumers
        self._producers.pop(old_id, None)
        self._producers[new_id] = (new_id, 0)
        # Non-primary buffers keep their own names; only their producer id moves.
        for buf in node.buffer_names()[1:]:
            self._producers[buf] = (new_id, self._producers[buf][1])
        for consumer_id in consumers:
            consumer = self.nodes[consumer_id]
            consumer.inputs = [new_id if i == old_id else i for i in consumer.inputs]
            consumer.op = _rename_buf_in_op(consumer.op, old_id, new_id)
        for inp in node.inputs:
            users = self._users.get(inp)
            if users is not None and old_id in users:
                users.discard(old_id)
                users.add(new_id)
        self.inputs = [new_id if i == old_id else i for i in self.inputs]
        self.outputs = [new_id if o == old_id else o for o in self.outputs]
        node.op = _rename_buf_in_op(node.op, old_id, new_id)

    def replace_node(self, old_id: str, new_id: str) -> None:
        """Rewire all references from buffer ``old_id`` to buffer ``new_id``.

        Updates graph-level edges (consumer ``node.inputs`` and
        ``graph.inputs/outputs``) AND any LoopOp's internal buf references
        (``Load.source`` / ``Write.output`` are buf names tracking the same
        identity as the graph node ids). Both arguments are buffer names —
        a node id for primary outputs, the tensor's own name for
        non-primary ones.
        """
        if new_id not in self._producers and new_id not in self.nodes:
            raise KeyError(f"Replacement buffer {new_id!r} not found")
        old_users = self._users.get(old_id, set())
        for consumer_id in list(old_users):
            node = self.nodes[consumer_id]
            node.inputs = [new_id if i == old_id else i for i in node.inputs]
            node.op = _rename_buf_in_op(node.op, old_id, new_id)
            self._users[new_id].add(consumer_id)
        if old_id in self._users:
            self._users[old_id] = set()
        self.inputs = [new_id if i == old_id else i for i in self.inputs]
        self.outputs = [new_id if o == old_id else o for o in self.outputs]

    def splice(
        self,
        fragment: Graph,
        *,
        consumed: Iterable[str],
        output: str | dict[str, str],
        mint_pieces: bool = False,
    ) -> str | dict[str, str]:
        """Splice ``fragment`` into this graph in place of ``consumed``.

        Fragment ``InputOp`` nodes alias existing graph nodes by id (no
        copy); every other fragment node is added with its original id
        when it doesn't collide, otherwise a fresh id.

        ``consumed`` is the set of node ids the rewriter declares the match
        owns. ``output`` selects which graph node(s) get their consumers
        redirected onto fragment output(s):

        - **single** (``str``) — the one node whose consumers redirect to
          the fragment's sole output (``fragment.outputs[0]``). Returns the
          new output's id (post-rename). Hints from every consumed node
          merge onto that output.
        - **multi** (``dict[str, str]``) — a ``{graph_node_id:
          fragment_output_id}`` map redirecting several graph nodes to
          distinct fragment outputs in one splice. Used to inline one
          producer into *all* its consumers at once (``005_split_shared_indexmap``):
          each consumer is replaced by its own fused fragment node. Returns
          ``{old_id: new_id}`` (post-rename). Each redirected node's hints
          merge onto its own new output; other consumed nodes' hints (e.g.
          the dissolved shared producer) are dropped.

        ``mint_pieces`` selects how op provenance threads through (see
        :mod:`emmy.compiler.provenance`): ``True`` for decomposition (each
        new fragment node is a fresh piece of the consumed origins), ``False``
        for fusion / lifting / folds (fragment outputs aggregate the consumed
        pieces). The prov hint is overwritten after the generic hint merge so
        union semantics win over last-writer."""
        from emmy.compiler import provenance  # noqa: PLC0415

        consumed = list(consumed)
        consumed_prov = {nid: provenance.get(self.nodes[nid]) for nid in consumed if nid in self.nodes}
        id_map: dict[str, str] = {}
        buf_map: dict[str, str] = {}  # fragment buffer name → post-add buffer name
        for frag_id in fragment.topological_order():
            frag_node = fragment.nodes[frag_id]
            if isinstance(frag_node.op, InputOp):
                id_map[frag_id] = frag_id  # references existing graph node
                buf_map[frag_id] = frag_id
                continue
            mapped_inputs = [buf_map.get(inp, inp) for inp in frag_node.inputs]
            # Preserve fragment ids when they don't collide. Lifting / fusion
            # use stable names (``lift_<nid>``, ``merged_<nid>``) that don't
            # clash because the original is already consumed. Stable ids keep
            # buf names inside LoopOp bodies (``Load.source``/``Write.output``,
            # both buf names) consistent with the surrounding graph.
            preferred_id = frag_id if frag_id not in self.nodes else None
            new_id = self.add_node(
                op=frag_node.op,
                inputs=mapped_inputs,
                outputs=tuple(Tensor(t.name, t.shape, t.dtype) for t in frag_node.outputs),
                node_id=preferred_id,
            )
            if frag_node.hints:
                self.nodes[new_id].hints = frag_node.hints
            id_map[frag_id] = new_id
            buf_map[frag_id] = new_id
            for t in frag_node.outputs[1:]:
                buf_map[t.name] = t.name  # non-primary buffers keep their names on add

        single = isinstance(output, str)
        output_map = {output: fragment.outputs[0]} if single else dict(output)

        # Redirect each old buffer's consumers onto its fragment buffer.
        new_by_old: dict[str, str] = {}
        for old_id, frag_out in output_map.items():
            new_out = buf_map[frag_out]
            self.replace_node(old_id, new_out)
            new_by_old[old_id] = new_out

        # Merge consumed-node hints, then remove consumed. Single-output keeps
        # the legacy behavior (every consumed node's hints land on the one
        # output); multi-output routes each redirected node's hints to its own
        # new output and drops the rest (the shared producer is dissolved).
        for nid in consumed:
            orig = self.nodes.get(nid)
            if orig is None:
                continue
            if orig.hints:
                if single:
                    self.producer(new_by_old[output]).hints.merge(orig.hints)
                elif nid in new_by_old:
                    self.producer(new_by_old[nid]).hints.merge(orig.hints)
            if nid not in output_map:
                self.remove_node(nid)
        for old_id in output_map:
            if old_id in self.nodes:
                self.remove_node(old_id)

        # Thread op provenance onto the new fragment nodes (mint fresh pieces
        # for decomposition, aggregate consumed pieces otherwise). Runs after
        # the generic hint merge so its union semantics win for the prov key.
        new_compute_ids = [id_map[fid] for fid, fn in fragment.nodes.items() if not provenance.is_boundary(fn.op)]
        provenance.propagate(
            self,
            consumed_prov=consumed_prov,
            new_compute_ids=new_compute_ids,
            # Prov hints live on nodes — resolve each redirected buffer to its
            # producing node id (identity for primary outputs).
            new_by_old={old: self._producers[new][0] for old, new in new_by_old.items()},
            output_map=output_map,
            mint_pieces=mint_pieces,
        )

        # Promote each new node's id to its friendly output.name once consumed
        # nodes are gone — keeps kernel buf names (which embed the node id)
        # readable. Falls back silently if the friendly name is taken. A
        # non-primary redirected buffer keeps its own name — nothing to promote.
        result: dict[str, str] = {}
        for old_id, new_out in new_by_old.items():
            node = self.nodes.get(new_out)
            if node is not None:
                desired = node.output.name
                if desired and desired != new_out and desired not in self.nodes and desired not in self._producers:
                    self.rename_node(new_out, desired)
                    new_out = desired
            result[old_id] = new_out

        self.remove_orphans()
        return result[output] if single else result

    def remove_orphans(self) -> None:
        """Remove nodes with zero consumers that aren't graph inputs/outputs."""
        output_set = set(self.outputs)
        input_set = set(self.inputs)

        def _is_protected(nid: str) -> bool:
            if nid in output_set or nid in input_set:
                return True
            node = self.nodes.get(nid)
            return node is not None and isinstance(node.op, InputOp)

        def _dep_ids(node: Node) -> set[str]:
            """Producing node ids of ``node``'s input buffers (an edge naming
            ANY of a producer's buffers keeps that producer alive)."""
            return {self._producers[inp][0] for inp in node.inputs if inp in self._producers}

        consumer_count: dict[str, int] = dict.fromkeys(self.nodes, 0)
        for node in self.nodes.values():
            for dep in _dep_ids(node):
                if dep in consumer_count:
                    consumer_count[dep] += 1

        queue: list[str] = [nid for nid, c in consumer_count.items() if c == 0 and not _is_protected(nid)]
        while queue:
            nid = queue.pop()
            if nid not in self.nodes:
                continue
            node = self.nodes[nid]
            deps = _dep_ids(node)
            self.remove_node(nid)
            for dep in deps:
                if dep in consumer_count:
                    consumer_count[dep] -= 1
                    if consumer_count[dep] == 0 and not _is_protected(dep):
                        queue.append(dep)

    def users(self, node_id: str) -> set[str]:
        """Return the set of node ids that consume ANY output buffer of
        ``node_id`` (the union over its buffers — identical to the historic
        semantics for single-output nodes). O(1) for those; rules that care
        which specific buffer an edge reads use :meth:`buffer_users`."""
        node = self.nodes.get(node_id)
        if node is None or len(node.outputs) == 1:
            return self._users.get(node_id, set())
        out: set[str] = set()
        for buf in node.buffer_names():
            out |= self._users.get(buf, set())
        return out

    def consumers(self, node_id: str) -> list[str]:
        """List form of :meth:`users`, preserved for existing callers."""
        return list(self.users(node_id))

    def producer(self, buf: str) -> Node | None:
        """The node producing buffer ``buf`` (any output slot), or ``None``
        when no such buffer exists. The buffer-name counterpart of
        ``graph.nodes.get(nid)`` — identical for primary outputs."""
        entry = self._producers.get(buf)
        return self.nodes[entry[0]] if entry is not None else None

    def buffer(self, buf: str) -> Tensor | None:
        """The :class:`Tensor` traveling under buffer name ``buf``, or
        ``None`` when no such buffer exists."""
        entry = self._producers.get(buf)
        if entry is None:
            return None
        nid, slot = entry
        return self.nodes[nid].outputs[slot]

    def buffer_users(self, buf: str) -> set[str]:
        """The node ids reading buffer ``buf`` specifically (one output
        slot — not the producing node's other buffers). O(1)."""
        return self._users.get(buf, set())

    def validate(self) -> None:
        """Check the graph's structural invariants; raise ``ValueError`` on the
        first violation. Covers: non-empty ``node.outputs``; SSA per buffer
        (``_producers`` maps every buffer to exactly its producing node/slot,
        no orphan or missing entries); ``_users`` consistency with every
        ``node.inputs`` edge; ``graph.inputs`` / ``graph.outputs`` naming known
        buffers. Runs in tests always and behind ``EMMY_VALIDATE_GRAPH`` at
        pass boundaries — never in production compile paths."""
        expected_producers: dict[str, tuple[str, int]] = {}
        for nid, node in self.nodes.items():
            if node.id != nid:
                raise ValueError(f"node keyed {nid!r} carries id {node.id!r}")
            if not node.outputs:
                raise ValueError(f"node {nid!r} has no outputs")
            for slot, buf in enumerate(node.buffer_names()):
                if buf in expected_producers:
                    raise ValueError(f"buffer {buf!r} has two producers: {expected_producers[buf]} and {(nid, slot)}")
                expected_producers[buf] = (nid, slot)
        if expected_producers != self._producers:
            missing = expected_producers.keys() - self._producers.keys()
            stale = self._producers.keys() - expected_producers.keys()
            wrong = {b for b in expected_producers.keys() & self._producers.keys() if expected_producers[b] != self._producers[b]}
            raise ValueError(f"_producers index out of sync: missing={sorted(missing)} stale={sorted(stale)} wrong={sorted(wrong)}")
        expected_users: dict[str, set[str]] = {buf: set() for buf in expected_producers}
        for nid, node in self.nodes.items():
            for inp in node.inputs:
                if inp not in expected_producers:
                    raise ValueError(f"node {nid!r} reads unknown buffer {inp!r}")
                expected_users[inp].add(nid)
        if expected_users != self._users:
            diff = {b for b in expected_users.keys() | self._users.keys() if expected_users.get(b) != self._users.get(b)}
            raise ValueError(f"_users index out of sync for buffers {sorted(diff)}")
        for label, refs in (("inputs", self.inputs), ("outputs", self.outputs)):
            for buf in refs:
                if buf not in expected_producers:
                    raise ValueError(f"graph.{label} names unknown buffer {buf!r}")

    # ------------------------------------------------------------------
    # Boundary / symbolic-dim queries
    # ------------------------------------------------------------------

    def constant_ops(self) -> Iterator[tuple[str, ConstantOp]]:
        """Yield ``(node_id, op)`` for every ``ConstantOp`` node, in insertion order."""
        for nid, node in self.nodes.items():
            if isinstance(node.op, ConstantOp):
                yield nid, node.op

    def loadable_constants(self) -> Iterator[tuple[str, ConstantOp]]:
        """Yield ``(node_id, op)`` for the constants an executor must *load* — non-static
        (``value is None``) constants that name an external ``source_path`` (or a
        ``source_parts`` concat of several). Scalar, ``context_value``, and synthetic
        (source-less) constants are skipped: the backend materializes those itself. The
        single entry point shared by the safetensors and live-module loaders."""
        for nid, op in self.constant_ops():
            if op.value is None and (op.source_path is not None or op.source_parts):
                yield nid, op

    def node_role(self, node_id: str) -> str:
        """Classify a node for buffer planning: ``'input'`` / ``'constant'`` /
        ``'output'`` / ``'scratch'``. Inputs take precedence over outputs (a passthrough
        node that is both binds as an input)."""
        if node_id in self.inputs:
            return "input"
        node = self.nodes.get(node_id)
        if node is not None and isinstance(node.op, ConstantOp):
            return "constant"
        if node_id in self.outputs:
            return "output"
        return "scratch"

    def buffer_role(self, buf: str) -> str:
        """Per-BUFFER planning role — :meth:`node_role` for primary outputs; a
        non-primary buffer classifies by its own graph membership (a MIMO node
        can produce one graph-output buffer and one scratch buffer)."""
        nid, slot = self._producers[buf]
        if slot == 0:
            return self.node_role(nid)
        if buf in self.inputs:
            return "input"
        if buf in self.outputs:
            return "output"
        return "scratch"

    def symbolic_env(self, input_data: dict[str, Any] | None) -> dict[str, int]:
        """Resolve each atomic symbolic input dim to its runtime size from the supplied
        input arrays. A symbolic name is recovered from an atomic ``Var`` axis
        (``shape[d] == Var(name)``); first-seen position wins. Inputs absent from
        ``input_data`` — or with fewer runtime dims than declared — are skipped.
        Downstream (possibly composite) shapes then resolve via ``dim.expr.eval(env)``."""
        import numpy as np  # noqa: PLC0415

        from emmy.compiler.ir.expr import Var  # noqa: PLC0415

        input_data = input_data or {}
        env: dict[str, int] = {}
        for nid in self.inputs:
            t = self.buffer(nid)
            if t is None or nid not in input_data:
                continue
            arr_shape = np.asarray(input_data[nid]).shape
            for i, d in enumerate(t.shape):
                if isinstance(d.expr, Var) and i < len(arr_shape):
                    env.setdefault(d.expr.name, int(arr_shape[i]))
        return env

    def symbolic_bindings(self) -> dict[str, tuple[str, int]]:
        """Map every symbolic dim name to its source ``(input_buf, dim_index)`` — the launch
        resolver reads the runtime value from ``input_arrays[buf].shape[dim_index]``.
        First-seen position wins on conflicts so each name resolves deterministically.

        A symbolic name is recovered from an **atomic** ``Var`` axis (``shape[d] ==
        Var(name)``). A **composite** symbolic input dim — e.g. a sliced masked-K producer
        slab's padded extent ``((seq_len + 63) // 64) * 64`` reaching a standalone-benched
        consumer as a synthetic input — can't be inverted to its free var directly, so it
        does NOT bind; it is fine as long as every free var it carries is bound from an
        atomic axis somewhere (the slab's own ``seq_q`` axis, the sibling P operand, …).
        Only a name that appears *exclusively* in composite dims is unrecoverable — that
        raises."""
        from emmy.compiler.ir.expr import Var  # noqa: PLC0415

        bindings: dict[str, tuple[str, int]] = {}
        composite_names: set[str] = set()
        for nid in self.inputs:
            for d, dim in enumerate(self.buffer(nid).shape):
                if dim.is_static:
                    continue
                if isinstance(dim.expr, Var):
                    bindings.setdefault(dim.expr.name, (nid, d))
                else:
                    composite_names |= dim.expr.free_vars()
        unrecoverable = composite_names - bindings.keys()
        if unrecoverable:
            raise ValueError(
                f"symbolic name(s) {sorted(unrecoverable)} appear only in composite input dims; "
                "no atomic Var-backed axis to recover them from at launch"
            )
        return bindings

    def symbolic_hints(self) -> dict[str, int]:
        """Map every symbolic input-dim name to its ``Dim`` hint (default expected size),
        read straight off the input ``Dim``. Used as the fallback bench size when no
        ``input_data`` is supplied (the autotuner case)."""
        from emmy.compiler.ir.expr import Var  # noqa: PLC0415

        hints: dict[str, int] = {}
        for nid in self.inputs:
            for dim in self.buffer(nid).shape:
                if isinstance(dim.expr, Var) and dim.hint is not None:
                    hints.setdefault(dim.expr.name, dim.hint)
        return hints

    def structural_key(self) -> str:
        """Implements :class:`emmy.compiler.structural.Structural`.

        Merkle-style structural digest of the graph. Two graphs that
        compute the same dataflow — same op kinds, same canonicalized op
        bodies, same Tensor shapes / dtypes, same input wiring — produce
        the same digest. Two graphs that differ in any of those produce
        different digests.

        Per node the key combines:

        - op class name,
        - op body's :meth:`Body.structural_key` (for body-bearing ops:
          ``LoopOp`` / ``TileOp`` / ``KernelOp``) — already canonicalizes
          SSA / axis / commutative-arg / buffer names,
        - other dataclass fields of the op rendered via ``repr`` — skipping
          ``name`` (instance identifier),
        - ``Tensor.shape`` and ``Tensor.dtype`` of the output (skipping
          ``Tensor.name`` — graph-internal label),
        - the ordered tuple of input nodes' digests (recursive).

        Top-level digest folds in the graph's :attr:`inputs` /
        :attr:`outputs` sequences. ``Hints`` (advisory) and graph-internal
        node ids are deliberately excluded.

        Not cached: ``Graph`` is mutable. Callers that dedup many
        candidates should snapshot the digest into a dict / set themselves.
        """
        from dataclasses import fields as dc_fields  # noqa: PLC0415

        from emmy.compiler.dim import Dim  # noqa: PLC0415
        from emmy.compiler.ir.stmt.body import Body  # noqa: PLC0415
        from emmy.compiler.structural import digest  # noqa: PLC0415

        def _unwrap_dims(v: object) -> object:
            """Unwrap ``Dim`` to its ``int | str`` value (recursively into
            tuples). Keeps digests stable across the static-int → ``Dim``
            migration; symbolic dims hash by name, not by current binding."""
            if isinstance(v, Dim):
                return v.value
            if isinstance(v, tuple):
                return tuple(_unwrap_dims(x) for x in v)
            return v

        keys: dict[str, str] = {}

        def node_key(nid: str) -> str:
            cached = keys.get(nid)
            if cached is not None:
                return cached
            node = self.nodes[nid]
            op = node.op
            body = getattr(op, "body", None)
            if isinstance(body, Body):
                op_payload: tuple = ("body", body.structural_key())
            else:
                attrs = tuple(
                    (f.name, repr(_unwrap_dims(getattr(op, f.name)))) for f in dc_fields(op) if f.name not in _STRUCTURAL_SKIP_FIELDS
                )
                op_payload = ("attrs", attrs)
            out = node.output
            out_payload = (_unwrap_dims(tuple(out.shape)), out.dtype)
            if len(node.outputs) > 1:
                # Fold non-primary outputs in slot order. Appended (rather than
                # always tupling every output) so single-output digests are
                # byte-identical to the pre-MIMO scheme.
                out_payload = (out_payload, tuple((_unwrap_dims(tuple(t.shape)), t.dtype) for t in node.outputs[1:]))
            input_payload = tuple(edge_key(i) for i in node.inputs)
            d = digest(type(op).__name__, op_payload, out_payload, input_payload)
            keys[nid] = d
            return d

        def edge_key(buf: str) -> str:
            """Digest of one edge: the producing node's key, with the output
            slot folded in for non-primary buffers (slot 0 stays the bare node
            key — pre-MIMO digests unchanged)."""
            entry = self._producers.get(buf)
            if entry is None:
                return node_key(buf)  # dangling ref — keyerror as before
            nid, slot = entry
            return node_key(nid) if slot == 0 else digest("slot", node_key(nid), slot)

        # Hash from outputs back so disconnected dead nodes don't perturb the
        # key, and from inputs forward so an input-order swap is observable.
        out_keys = tuple(edge_key(o) for o in self.outputs)
        in_keys = tuple(edge_key(i) for i in self.inputs)
        return digest("graph", in_keys, out_keys)

    def topological_order(self) -> list[str]:
        """Return node ids in topological order (inputs before consumers).

        Kahn's algorithm in O(N+E) using the maintained ``_users`` index.
        """
        in_degree: dict[str, int] = {nid: 0 for nid in self.nodes}
        for node in self.nodes.values():
            deps = {self._producers[inp][0] for inp in node.inputs if inp in self._producers}
            in_degree[node.id] += len(deps - {node.id})

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        result: list[str] = []
        while queue:
            nid = queue.pop(0)
            result.append(nid)
            for consumer_id in self.users(nid):
                in_degree[consumer_id] -= 1
                if in_degree[consumer_id] == 0:
                    queue.append(consumer_id)

        if len(result) != len(self.nodes):
            raise ValueError("Graph has a cycle")
        return result

    def copy(self) -> Graph:
        """Return a deep copy of the graph."""
        g = Graph()
        g._id_counter = itertools.count(next(self._id_counter))
        g.hints = Hints.from_dict(self.hints.to_dict())
        for nid, node in self.nodes.items():
            g.nodes[nid] = Node(
                id=node.id,
                op=node.op,
                inputs=list(node.inputs),
                outputs=tuple(Tensor(name=t.name, shape=t.shape, dtype=t.dtype) for t in node.outputs),
                hints=Hints.from_dict(node.hints.to_dict()),
            )
        g._users = {buf: set(users) for buf, users in self._users.items()}
        g._producers = dict(self._producers)
        g.inputs = list(self.inputs)
        g.outputs = list(self.outputs)
        return g

    @staticmethod
    def from_dict(data: dict) -> Graph:
        """Deserialize a graph from a JSON-compatible dict (inverse of to_dict)."""
        g = Graph()
        g.hints = Hints.from_dict(data.get("hints", {}))
        # First pass: create all nodes (inputs first, then in order).
        for nid, ndata in data["nodes"].items():
            op_cls_name = ndata["op"]
            op_cls = _lookup_op_class(op_cls_name)
            if op_cls is None:
                raise ValueError(f"Unknown op class: {op_cls_name}")

            fields = {k: _deserialize_field(k, v) for k, v in ndata.get("op_fields", {}).items()}
            op = op_cls(**fields) if fields else op_cls()
            # Dual-read: the historic single-``output`` dict and the plural
            # ``outputs`` list (slot order). Old dumps stay loadable.
            outs_data = ndata["outputs"] if "outputs" in ndata else [ndata["output"]]
            tensors = tuple(Tensor(name=out["name"], shape=tuple(out["shape"]), dtype=out.get("dtype", "f32")) for out in outs_data)
            node_hints = Hints.from_dict(ndata.get("hints", {}))
            # Add directly to bypass input validation (nodes may reference later nodes).
            g.nodes[nid] = Node(id=nid, op=op, inputs=list(ndata["inputs"]), outputs=tensors, hints=node_hints)

        # Rebuild the producer / forward-edge indexes now that every node is present.
        for nid, node in g.nodes.items():
            for slot, buf in enumerate(node.buffer_names()):
                g._producers[buf] = (nid, slot)
                g._users.setdefault(buf, set())
        for nid, node in g.nodes.items():
            for inp in node.inputs:
                if inp in g._users:
                    g._users[inp].add(nid)

        g.inputs = list(data["inputs"])
        g.outputs = list(data["outputs"])
        return g

    def pretty_print(self) -> str:
        """Render the graph as readable text (topological order + sections)."""
        from emmy.compiler.ir.base import ConstantOp, InputOp

        order = self.topological_order()
        lines: list[str] = [
            f"# Graph: {len(self.nodes)} nodes, {len(self.inputs)} inputs, {len(self.outputs)} outputs",
            "",
        ]

        if self.inputs:
            lines.append("inputs:")
            for nid in self.inputs:
                lines.append(f"  {_fmt_tensor(self.buffer(nid))}")
            lines.append("")

        const_ids = [nid for nid in order if isinstance(self.nodes[nid].op, ConstantOp)]
        if const_ids:
            lines.append("constants:")
            for nid in const_ids:
                n = self.nodes[nid]
                val = getattr(n.op, "value", None)
                suffix = f" = {val}" if val is not None else ""
                lines.append(f"  {_fmt_tensor(n.output)}{suffix}")
            lines.append("")

        compute_ids = [nid for nid in order if not isinstance(self.nodes[nid].op, (InputOp, ConstantOp))]
        if compute_ids:
            name_w = max(len(self.nodes[nid].output.name) for nid in compute_ids)
            op_w = max(len(_fmt_op(self.nodes[nid], self)) for nid in compute_ids)
            for nid in compute_ids:
                n = self.nodes[nid]
                op_str = _fmt_op(n, self)
                shape_str = _fmt_shape(n.output.shape)
                lines.append(f"{n.output.name:<{name_w}}  =  {op_str:<{op_w}}  -> {shape_str} {n.output.dtype}")
            lines.append("")

        if self.outputs:
            lines.append("outputs:")
            for nid in self.outputs:
                lines.append(f"  {_fmt_tensor(self.buffer(nid))}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize graph to a JSON-compatible dict."""
        result: dict = {
            "inputs": self.inputs,
            "outputs": self.outputs,
            "nodes": {},
        }
        if self.hints:
            result["hints"] = self.hints.to_dict()
        from dataclasses import fields as dc_fields  # noqa: PLC0415

        for nid, node in self.nodes.items():
            entry: dict = {
                "op": type(node.op).__name__,
                "op_fields": {
                    f.name: _serialize_field(getattr(node.op, f.name)) for f in dc_fields(node.op) if f.name not in _SERIALIZE_SKIP_FIELDS
                },
                "inputs": node.inputs,
                # Serialized in the plural ``outputs`` form (slot order);
                # ``from_dict`` dual-reads this and the historic single-
                # ``output`` dict, so old dumps stay loadable.
                "outputs": [
                    {
                        "name": t.name,
                        # JSON dump preserves atomic Dims (int / Var name) so the
                        # round-trip ``emmy run --ir <json>`` flow reconstructs
                        # them via ``Dim(value)``. Composite Dims (BinaryExpr-backed,
                        # e.g. a CatOp output or the demoted symbolic-N B operand's
                        # ``round_up(seq_len, 64)`` TMA-padded inner extent) serialize
                        # to their pretty expr string so a ``EMMY_DUMP_DIR`` dump
                        # of a dynamic-attention graph doesn't crash; they don't
                        # round-trip back through ``run --ir`` (the string isn't
                        # re-parsed) — a debug-dump artifact only, matching the prior
                        # composite-shape limitation.
                        "shape": [_dim_to_json(d) for d in t.shape],
                        "dtype": t.dtype.name,
                    }
                    for t in node.outputs
                ],
            }
            if node.hints:
                entry["hints"] = node.hints.to_dict()
            result["nodes"][nid] = entry
        return result


# ---------------------------------------------------------------------------
# Op buf-rename helper — keeps LoopOp's internal buf refs aligned with
# graph node ids when ``replace_node`` rewires consumers.
# ---------------------------------------------------------------------------


def _dim_to_json(d):
    """Serialize a ``Dim`` for the JSON dump. Atomic dims (static int / symbolic
    ``Var`` name) return their scalar ``value`` and round-trip via ``Dim(value)``;
    a composite ``Dim`` (``BinaryExpr``-backed) has no scalar value, so it falls
    back to the pretty expr string (a debug-dump artifact — see ``to_dict``)."""
    try:
        return d.value
    except TypeError:
        return d.expr.pretty()


def _rename_buf_in_op(op, old: str, new: str):
    """Rewrite ``Load.source`` / ``Write.output`` references inside a
    ``LoopOp`` body from ``old`` to ``new`` (recursively into nested Loops).
    Pass-through for op types without internal buf refs. Preserves the op's
    ``name`` / ``knobs`` / ``source`` identity — a rename after
    ``991_stamp_loop_names`` / ``992_stamp_structural_features`` (e.g. the
    splice id-promotion of a lowering-phase fragment like the demoted-matmul
    split) must not strip the stamped kernel name, the ``S_*`` features, or
    the decomposition attribution link (``Candidate.apply`` stamps the
    pre-split op as each fragment kernel's ``source``; the two-level tuner's
    composed Σ rows group by it)."""
    from emmy.compiler.ir.loop import Load, LoopOp, Write

    if not isinstance(op, LoopOp):
        return op

    def fn(s):
        if isinstance(s, Load) and s.input == old:
            return Load(name=s.name, input=new, index=s.index)
        if isinstance(s, Write) and s.output == old:
            return Write(output=new, index=s.index, value=s.value)
        return s

    renamed = LoopOp(body=op.body.map(fn))
    renamed.name = op.name
    renamed.knobs = dict(op.knobs)
    renamed.source = op.source
    return renamed


# ---------------------------------------------------------------------------
# Pretty-print helpers (used by Graph.pretty_print)
# ---------------------------------------------------------------------------


def _fmt_shape(shape: tuple) -> str:
    inside = ", ".join(str(d) for d in shape)
    if len(shape) == 1:
        inside += ","
    return f"({inside})"


def _fmt_tensor(t: Tensor) -> str:
    return f"{t.name}: {_fmt_shape(t.shape)} {t.dtype}"


_DIM_NAMES = ("i", "j", "k", "l", "m", "n")


def _fmt_op(node: Node, graph: Graph) -> str:
    """Render `node` as a function-call-like string using input tensor names."""
    op = node.op
    cls = type(op).__name__
    arg_names = [t.name if (t := graph.buffer(inp)) is not None else inp for inp in node.inputs]

    if cls == "ElementwiseOp":
        return f"{op.name}({', '.join(arg_names)})"
    if cls == "ReduceOp":
        return f"{op.name}({', '.join(arg_names)}, axis={op.axis})"
    if cls == "ScanOp":
        return f"scan_{op.name}({', '.join(arg_names)}, axis={op.axis})"
    if cls == "GatherOp":
        return f"gather({', '.join(arg_names)}, axis={op.axis})"
    if cls == "ScatterOp":
        red = f", reduce={op.reduce_fn}" if getattr(op, "reduce_fn", None) else ""
        return f"scatter({', '.join(arg_names)}, axis={op.axis}{red})"
    if cls == "IndexMapOp":
        return _fmt_indexmap(node, graph)

    # Generic fallback (frontend ops like MeanOp, LinearOp, SdpaOp, or LoopOp).
    fields = {k: v for k, v in op.__dict__.items() if not k.startswith("_") and k != "name"}
    label = cls.removesuffix("Op").lower() or cls
    parts = list(arg_names) + [f"{k}={v}" for k, v in fields.items()]
    return f"{label}({', '.join(parts)})"


def _fmt_indexmap(node: Node, graph: Graph) -> str:
    """Render IndexMapOp concisely when its coord expression is simple.

    Single-source, no select, ≤6 output dims → ``src[coord_0, coord_1, ...]``
    with ``out_coord_N`` placeholders replaced by ``i``/``j``/``k``/``l``/``m``/``n``.
    A ``Literal(0)`` at a position where the input's corresponding dim has
    extent 1 is rendered as ``na`` (numpy-style newaxis), making broadcasts
    visually distinct from scalar slices. Everything else (multi-source,
    selected reads, many dims) falls back to ``indexmap(src0, src1, ...)``.
    """
    from emmy.compiler.ir.expr import PLACEHOLDER_PREFIX, Literal, Var

    op = node.op
    sources = op.sources
    out_shape = op.out_shape
    arg_names = [t.name if (t := graph.buffer(inp)) is not None else inp for inp in node.inputs]

    if len(sources) != 1 or sources[0].select is not None or len(out_shape) > len(_DIM_NAMES):
        return f"indexmap({', '.join(arg_names)})"

    src = sources[0]
    input_name = arg_names[src.input_idx] if src.input_idx < len(arg_names) else f"${src.input_idx}"
    input_id = node.inputs[src.input_idx] if src.input_idx < len(node.inputs) else None
    input_t = graph.buffer(input_id) if input_id else None
    input_shape = input_t.shape if input_t is not None else ()

    placeholder_rename = {f"{PLACEHOLDER_PREFIX}{n}": Var(name) for n, name in enumerate(_DIM_NAMES)}

    coord_strs: list[str] = []
    for dim, e in enumerate(src.coord_map):
        # Literal(0) at an extent-1 input dim is a broadcast — render as `na`.
        if isinstance(e, Literal) and e.value == 0 and dim < len(input_shape) and input_shape[dim] == 1:
            coord_strs.append("na")
        else:
            coord_strs.append(e.substitute(placeholder_rename).pretty())
    return f"{input_name}[{', '.join(coord_strs)}]"
