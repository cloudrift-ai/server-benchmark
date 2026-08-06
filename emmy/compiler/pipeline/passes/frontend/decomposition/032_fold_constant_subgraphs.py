"""Collapse a maximal constant-only cone containing a storage-decode op into ONE bind-time constant.

A quantized checkpoint enters the graph as pure algebra from birth
(``loader.quant.spell_quantized_constants``): bits constant + scale constant + the decode-cast /
broadcast-multiply cone. This rule dissolves that algebra as early as possible — the cone
collapses into a single ``ConstantOp`` carrying the evaluation as a ``source_graph`` bind record
(the constant-only mini-graph itself), which the loader binds leaf-by-leaf and evaluates through
the reference NumPy backend at bind time. Everything downstream then sees a plain constant:
``035``'s sibling merge and the ``050``/``060`` layout folds keep pattern-matching ordinary
constants (a later fold appends its transpose to THIS constant's ``load_ops``, applied after the
record evaluates), and no pass past this point can tell a quantized checkpoint from a plain one.

Matching rule — the cone rooted at a node ``r``:

- every transitive input of ``r`` bottoms out in a source-backed ``ConstantOp`` (a checkpoint
  ``source_path`` / ``source_parts`` / nested ``source_graph`` leaf — scalar and synthetic
  constants decline the fold);
- every interior op is a numpy-evaluable layout/elementwise op with static shapes;
- the cone contains at least one storage-decode op (``ElementwiseImpl.decodes`` non-``None``);
- ``r`` is MAXIMAL: no consumer of ``r`` would itself extend the constant-only cone (otherwise
  the fold waits and fires at that consumer, so one constant absorbs the whole cone);
- interior nodes are owned by the cone (no external consumer reads a mid-cone value).

The decode-trait requirement is the digest-safety SCOPE, not a structural necessity: this rule
is the conservative first instantiation of a general constant-subgraph fold, and folding
anything else (e.g. an existing model's constant mask-math cone) is deliberately out of scope —
widening the scope is gated on kernel-source digest evidence (``scripts/digest_kernels.py``),
because those cones' kernels exist today and must not change bytes silently.

Gate: the fold is DEFAULT ON. ``EMMY_FP8_EXPAND`` (``config.fp8_expand``) SKIPS it — the decode
cone then stays in-graph and rides the operand cone into the kernel (fp8 bits in device memory,
the decode absorbed by the storage dtype at the warp tier — see ``lowering/tile/_atomize``).

Numbered 032 — after the arithmetic normalizations (``010``–``030``), BEFORE the sibling merge
(``035``) and the layout folds (``050``/``060``), so those passes only ever see the collapsed
plain constant and compose on top of it.
"""

from __future__ import annotations

import copy

from emmy import config
from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.base import ConstantOp
from emmy.compiler.ir.frontend.ir import ReshapeOp, TransposeOp
from emmy.compiler.ir.tensor.ir import ElementwiseOp, IndexMapOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped

# A cone root is always a compute node; ReshapeOp covers the 2-D-block form's reshape-back root.
PATTERN = [Pattern("root", (ElementwiseOp, ReshapeOp))]

# Interior ops the fold evaluates at bind time — the vocabulary the birth-time speller emits
# (decode cast, broadcast, scale multiply, block reshapes) plus the plain layout ops. All have
# working numpy ``forward``s; anything else declines the fold.
_FOLDABLE = (ElementwiseOp, ReshapeOp, TransposeOp, IndexMapOp)


def _is_source_leaf(op: ConstantOp) -> bool:
    """A source-backed constant leaf the loader can bind (never a scalar / synthetic one)."""
    return op.value is None and (op.source_path is not None or bool(op.source_parts) or op.source_graph is not None)


def _collect_cone(graph: Graph, root_id: str) -> tuple[set[str], bool] | None:
    """The constant-only cone rooted at ``root_id`` as ``(node ids, has_decode)``, or ``None``
    when any transitive input disqualifies it (non-foldable op, symbolic shape, non-source
    constant)."""
    ids: set[str] = set()
    has_decode = False
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
            if not _is_source_leaf(op):
                return None
            ids.add(nid)
            continue
        if not isinstance(op, _FOLDABLE) or any(not d.is_static for d in node.output.shape):
            return None
        if isinstance(op, ElementwiseOp) and op.op.decodes is not None:
            has_decode = True
        ids.add(nid)
        for inp in node.inputs:
            producer = graph.producer(inp)
            if producer is None:
                return None
            stack.append(producer.id)
    return ids, has_decode


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
        if isinstance(producer.op, ConstantOp) and _is_source_leaf(producer.op):
            continue
        if _collect_cone(graph, producer.id) is None:
            return False
    return True


def rewrite(match: Match, root: Node, out: Tensor) -> Graph:
    if config.fp8_expand():
        raise RuleSkipped("EMMY_FP8_EXPAND on — the decode cone stays in-graph for the kernel path")
    collected = _collect_cone(match.graph, root.id)
    if collected is None:
        raise RuleSkipped("not a constant-only cone")
    cone, has_decode = collected
    if not has_decode:
        # Digest-safety scope: only cones carrying a storage decode fold — see module docstring.
        raise RuleSkipped("constant cone carries no storage-decode op — outside the fold's scope")
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
