"""Collapse a maximal static computation cone into ONE bind-time constant.

The cone becomes a ``ConstantOp`` whose ``source_graph`` is the constant-only mini-graph itself.
The loader binds its source leaves and evaluates the record through the reference NumPy backend.
Everything downstream therefore sees one ordinary constant, while the record stays a transparent,
serializable account of how that value is reconstructed.

Matching rule — the cone rooted at a node ``r``:

- every transitive input of ``r`` bottoms out in a source-backed or scalar ``ConstantOp``;
- every interior op is a NumPy-evaluable generic tensor/layout op with static shapes;
- the cone contains real computation, not only layout changes;
- ``r`` is MAXIMAL: no consumer of ``r`` would itself extend the constant-only cone (otherwise
  the fold waits and fires at that consumer, so one constant absorbs the whole cone);
- interior nodes are owned by the cone (no external consumer reads a mid-cone value).

Storage-decode cones deliberately stay expanded. Folding one would replace compressed device
storage with a larger compute-dtype buffer, so ``ElementwiseImpl.decodes`` is the generic policy
boundary: dtype decode + ordinary algebra remains available to lowering without a format flag.
Pure layout cones also stay for the dedicated ``050``/``060`` policy, which decides whether a
physical load-time layout change is profitable for the target.

Numbered 032 runs after arithmetic normalization and before the sibling merge and layout folds.
"""

from __future__ import annotations

import copy

from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.base import ConstantOp
from emmy.compiler.ir.frontend.ir import MatmulOp, ReshapeOp, SliceOp, TransposeOp
from emmy.compiler.ir.tensor.ir import BitcastOp, CastOp, ElementwiseOp, GatherOp, IndexMapOp, RangeOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped

_FOLDABLE = (
    ElementwiseOp,
    MatmulOp,
    ReshapeOp,
    SliceOp,
    TransposeOp,
    GatherOp,
    IndexMapOp,
    CastOp,
    BitcastOp,
    RangeOp,
)
_COMPUTE = (ElementwiseOp, MatmulOp, GatherOp, CastOp, BitcastOp, RangeOp)
PATTERN = [Pattern("root", _FOLDABLE)]


def _is_static_leaf(op: ConstantOp) -> bool:
    """A source-backed or literal scalar leaf that bind-time evaluation can supply."""
    return op.value is not None or op.source_path is not None or bool(op.source_parts) or op.source_graph is not None


def _collect_cone(graph: Graph, root_id: str) -> tuple[set[str], bool, bool] | None:
    """Return ``(node ids, has_compute, has_storage_decode)`` for a static cone."""
    ids: set[str] = set()
    has_compute = False
    has_storage_decode = False
    stack = [root_id]
    while stack:
        nid = stack.pop()
        if nid in ids:
            continue
        node = graph.nodes.get(nid)
        if node is None:
            return None
        op = node.op
        if isinstance(op, ConstantOp):
            if not _is_static_leaf(op):
                return None
            ids.add(nid)
            continue
        if not isinstance(op, _FOLDABLE) or any(not d.is_static for d in node.output.shape):
            return None
        has_compute |= isinstance(op, _COMPUTE)
        has_storage_decode |= isinstance(op, ElementwiseOp) and op.op.decodes is not None
        ids.add(nid)
        for inp in node.inputs:
            producer = graph.producer(inp)
            if producer is None:
                return None
            stack.append(producer.id)
    return ids, has_compute, has_storage_decode


def _extends_cone(graph: Graph, consumer_id: str, cone: set[str]) -> bool:
    """Whether ``consumer_id`` would itself belong to a (bigger) constant-only cone — i.e. it is
    a foldable op every input of which is either inside ``cone`` or a constant-only cone of its
    own. If so, the current match is not maximal."""
    node = graph.nodes.get(consumer_id)
    if node is None or not isinstance(node.op, _FOLDABLE) or any(not d.is_static for d in node.output.shape):
        return False
    for inp in node.inputs:
        producer = graph.producer(inp)
        if producer is None:
            return False
        if producer.id in cone:
            continue
        if isinstance(producer.op, ConstantOp) and _is_static_leaf(producer.op):
            continue
        if _collect_cone(graph, producer.id) is None:
            return False
    return True


def rewrite(match: Match, root: Node, out: Tensor) -> Graph:
    collected = _collect_cone(match.graph, root.id)
    if collected is None:
        raise RuleSkipped("not a constant-only cone")
    cone, has_compute, has_storage_decode = collected
    if not has_compute:
        raise RuleSkipped("constant cone is layout-only — leave it to the target layout-fold policy")
    if has_storage_decode:
        raise RuleSkipped("storage-decode cone stays expanded to preserve compressed device storage")
    for nid in cone:
        if nid != root.id and (nid in match.graph.outputs or not match.graph.users(nid) <= cone):
            raise RuleSkipped(f"cone-interior node {nid!r} has consumers outside the cone")
    for cid in match.graph.users(root.id):
        if _extends_cone(match.graph, cid, cone):
            raise RuleSkipped(f"consumer {cid!r} extends the constant cone — fold fires at the maximal root")

    # The bind record IS the cone, copied verbatim (same ids, same op fields) into a mini-graph
    # whose single output is the root — the loader binds the leaf constants and evaluates the
    # rest. Ops are shallow-copied with their runtime state (matcher-snapped ``inputs`` /
    # ``outputs``, ``knobs``, the rewrite-chain ``source``) cleared, so the record's structural
    # key is stable across a serialization round-trip — persisted-IR form, not live-engine form.
    # Topological order guarantees every input exists by the time its consumer is added.
    record = Graph()
    for nid in match.graph.topological_order():
        if nid not in cone:
            continue
        node = match.graph.nodes[nid]
        op = copy.copy(node.op)
        op.inputs, op.outputs, op.knobs, op.source = {}, {}, {}, None
        record.add_node(
            op=op,
            inputs=list(node.inputs),
            outputs=tuple(Tensor(t.name, t.shape, t.dtype) for t in node.outputs),
            node_id=nid,
        )
    record.outputs = [root.id]

    shape = tuple(d.as_static() for d in out.shape)
    frag = Graph()
    folded = frag.add_node(
        op=ConstantOp(name=out.name, source_graph=record, source_shape=shape, source_dtype=out.dtype.name),
        inputs=[],
        output=Tensor(out.name, out.shape, out.dtype),
    )
    frag.outputs = [folded]
    match.consumed = set(cone)
    return frag
