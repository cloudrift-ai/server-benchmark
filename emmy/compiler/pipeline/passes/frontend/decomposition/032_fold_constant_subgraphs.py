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

Storage-expanding cones deliberately stay expanded. Folding one would replace compact source
storage with a larger computed buffer, so the policy compares the physical bytes of the generic
constant leaves with the root tensor. ``ElementwiseImpl.decodes`` remains the explicit policy
boundary for equal-size dtype decodes; neither rule needs to know a checkpoint format.
Pure layout cones also stay for the dedicated ``050``/``060`` policy, which decides whether a
physical load-time layout change is profitable for the target.

Numbered 032 runs after arithmetic normalization and before the sibling merge and layout folds.
"""

from __future__ import annotations

import copy
from math import prod

from emmy.compiler.dtype import get as get_dtype
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


def _shape_nbytes(shape, dtype) -> int:
    return prod(int(d) for d in shape) * get_dtype(dtype).nbytes


def _leaf_storage_nbytes(graph: Graph, nid: str) -> int:
    """External source bytes represented by one static constant leaf.

    ``source_shape``/``source_dtype`` describe checkpoint storage before any load-time
    transformations.  A nested bind record recursively contributes its external leaves.
    Synthetic scalar/range algebra contributes zero: it is compile-time computation, not a
    compact checkpoint representation that should be preserved on device.
    """
    node = graph.nodes[nid]
    op: ConstantOp = node.op
    if op.source_graph is not None:
        return sum(
            _leaf_storage_nbytes(op.source_graph, leaf_id)
            for leaf_id, leaf in op.source_graph.nodes.items()
            if isinstance(leaf.op, ConstantOp) and _is_static_leaf(leaf.op)
        )
    if op.source_path is None and not op.source_parts:
        return 0
    dtype = op.source_dtype or node.output.dtype
    if op.source_parts:
        return sum(_shape_nbytes(shape, dtype) for _path, shape in op.source_parts)
    shape = op.source_shape or tuple(d.as_static() for d in node.output.shape)
    return _shape_nbytes(shape, dtype)


def _cone_storage_nbytes(graph: Graph, cone: set[str]) -> int:
    return sum(
        _leaf_storage_nbytes(graph, nid)
        for nid in cone
        if isinstance(graph.nodes[nid].op, ConstantOp) and _is_static_leaf(graph.nodes[nid].op)
    )


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
    collected = _collect_cone(graph, consumer_id)
    if collected is None:
        return False
    if _cone_storage_nbytes(graph, cone):
        return True
    larger, _has_compute, _has_storage_decode = collected
    # A source-free static factor inside a larger checkpoint reconstruction must fold even
    # though its consumer is technically another constant cone: the larger cone will be kept
    # expanded by the compact-storage policy, so waiting for that root would leave the factor
    # as runtime work forever.
    source_nbytes = _cone_storage_nbytes(graph, larger)
    # Stop a deterministic, source-free cone at the first checkpoint-backed consumer.  The
    # storage cone has its own expansion/decode policy below; absorbing its scalar tables and
    # ranges merely turns one-time generated constants back into per-launch plumbing.
    return not source_nbytes


def rewrite(match: Match, root: Node, out: Tensor) -> Graph:
    collected = _collect_cone(match.graph, root.id)
    if collected is None:
        raise RuleSkipped("not a constant-only cone")
    cone, has_compute, has_storage_decode = collected
    if not has_compute:
        raise RuleSkipped("constant cone is layout-only — leave it to the target layout-fold policy")
    if isinstance(root.op, IndexMapOp) and any(
        (producer := match.graph.producer(inp)) is not None
        and all(d.is_static for d in producer.output.shape)
        and prod(d.as_static() for d in producer.output.shape) < prod(d.as_static() for d in out.shape)
        for inp in root.inputs
    ):
        # A computed scalar followed by a broadcast is still one scalar computation, not a
        # reason to allocate a full constant tensor.  Keep the broadcast lazy so it can inline
        # at the eventual consumer.  Genuinely per-element generated tables (range/index math)
        # do not end in an expanding IndexMap and still fold once.
        raise RuleSkipped("constant cone ends in a broadcast expansion — keep the scalar computation lazy")
    if has_storage_decode:
        raise RuleSkipped("storage-decode cone stays expanded to preserve compressed device storage")
    output_nbytes = _shape_nbytes(tuple(d.as_static() for d in out.shape), out.dtype)
    source_nbytes = _cone_storage_nbytes(match.graph, cone)
    if source_nbytes and output_nbytes > source_nbytes:
        raise RuleSkipped(
            f"constant cone expands {source_nbytes} external source bytes to {output_nbytes} output bytes — preserve compact storage"
        )
    retained_leaves: set[str] = set()
    for nid in cone:
        if nid == root.id:
            continue
        external = nid in match.graph.outputs or not match.graph.users(nid) <= cone
        if not external:
            continue
        node = match.graph.nodes[nid]
        if isinstance(node.op, ConstantOp) and _leaf_storage_nbytes(match.graph, nid) == 0:
            # Literal/generated leaves are immutable and cheap to reference from both the bind
            # record and another live cone.  Keep the original node for its external users while
            # copying it into this record; rejecting the whole fold would strand shared scalar
            # tables and ranges as runtime kernels.
            retained_leaves.add(nid)
            continue
        raise RuleSkipped(f"cone-interior node {nid!r} has consumers outside the cone")
    consumers = match.graph.users(root.id)
    if len(consumers) == 1:
        cid = next(iter(consumers))
        if _extends_cone(match.graph, cid, cone):
            raise RuleSkipped(f"consumer {cid!r} extends the constant cone — fold fires at the maximal root")
    elif not source_nbytes:
        # A shared deterministic value is already a maximal *owned* cone.  Waiting independently
        # for two or more consumers makes each larger cone reject the shared interior, leaving the
        # computation at runtime.  Fold the branch point once and let every consumer bind it.
        pass
    else:
        for cid in consumers:
            if _extends_cone(match.graph, cid, cone):
                raise RuleSkipped(f"consumer {cid!r} extends the source-backed cone — keep one bindable record")

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
    match.consumed = set(cone) - retained_leaves
    return frag
