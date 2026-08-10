"""Treeify a compact, storage-expanding ``MatmulOp`` B producer.

Loop fusion can inline a pure producer tree into its sole contraction consumer, but a generic
decode/reindex DAG naturally shares address arithmetic and intermediate values.  Materializing
every shared node turns that DAG into dozens of launch-sized buffers.  For a B operand whose
ordinary tensor algebra expands compact boundary storage into a larger matrix, clone shared
interior nodes into an expression tree instead.  Later generic fusion can then stream each B
element at its point of use.

The policy is structural and format-neutral:

* B is a static two-dimensional matmul operand;
* its producer cone contains only maps, casts, bit operations, gathers, ranges, and layouts;
* every interior value is owned by this contraction;
* materialized B bytes exceed the non-scalar boundary-input bytes; and
* at least one interior node is actually shared.

No annotation survives the rewrite.  The result is the same ordinary tensor algebra with
single-consumer ownership, trading compile-time graph size and scalar recomputation for removal
of the expanded global-memory intermediates.
"""

from __future__ import annotations

import copy
from math import prod

from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.frontend.ir import MatmulOp, ReshapeOp, TransposeOp
from emmy.compiler.ir.tensor.ir import BitcastOp, CastOp, ElementwiseOp, GatherOp, IndexMapOp, RangeOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.frontend.decomposition._helpers import open_fragment

PATTERN = [Pattern("matmul", MatmulOp)]

_MAP_OPS = (BitcastOp, CastOp, ElementwiseOp, GatherOp, IndexMapOp, ReshapeOp, TransposeOp)
_MATERIALIZED_BOUNDARIES = (RangeOp,)


def _shape(node: Node) -> tuple[int, ...] | None:
    if any(not d.is_static for d in node.output.shape):
        return None
    return tuple(d.as_static() for d in node.output.shape)


def _nbytes(node: Node) -> int:
    shape = _shape(node)
    assert shape is not None
    return prod(shape) * node.output.dtype.nbytes


def _collect(graph: Graph, root: Node) -> tuple[set[str], set[str]] | None:
    interior: set[str] = set()
    boundaries: set[str] = set()
    pending = [root.id]
    while pending:
        nid = pending.pop()
        if nid in interior or nid in boundaries:
            continue
        node = graph.nodes.get(nid)
        if node is None or _shape(node) is None or len(node.outputs) != 1:
            return None
        if isinstance(node.op, (InputOp, ConstantOp, *_MATERIALIZED_BOUNDARIES)):
            boundaries.add(nid)
            continue
        if not isinstance(node.op, _MAP_OPS):
            return None
        interior.add(nid)
        pending.extend(node.inputs)
    return interior, boundaries


def _source_bytes(graph: Graph, boundaries: set[str]) -> tuple[int, bool]:
    """Return boundary bytes and whether a real non-scalar storage input exists."""
    total = 0
    has_storage = False

    def externally_sourced(op: ConstantOp) -> bool:
        if op.source_path is not None or op.source_parts:
            return True
        if op.source_graph is None:
            return False
        return any(isinstance(node.op, ConstantOp) and externally_sourced(node.op) for node in op.source_graph.nodes.values())

    for nid in boundaries:
        node = graph.nodes[nid]
        size = _nbytes(node)
        op = node.op
        external = isinstance(op, InputOp) or (isinstance(op, ConstantOp) and externally_sourced(op))
        if external:
            total += size
            has_storage |= prod(_shape(node)) > 1
        elif isinstance(op, ConstantOp) and op.value is not None:
            total += size
    return total, has_storage


def rewrite(match: Match, matmul: Node, inp_a: Node, inp_b: Node, inp_bias: Node | None, out: Tensor) -> Graph:  # noqa: ARG001
    b_shape = _shape(inp_b)
    if b_shape is None or len(b_shape) != 2:
        raise RuleSkipped("matmul B is not a static matrix")
    collected = _collect(match.graph, inp_b)
    if collected is None:
        raise RuleSkipped("matmul B producer is not a pure generic map cone")
    interior, boundaries = collected
    if not interior:
        raise RuleSkipped("matmul B is already materialized")

    owned = interior | {matmul.id}
    for nid in interior:
        if nid in match.graph.outputs or not match.graph.users(nid) <= owned:
            raise RuleSkipped(f"B-cone interior node {nid!r} escapes this contraction")
    shared = {nid for nid in interior if len(match.graph.users(nid) & interior) > 1}
    if not shared:
        raise RuleSkipped("matmul B producer is already a single-consumer tree")

    source_bytes, has_storage = _source_bytes(match.graph, boundaries)
    if not has_storage or _nbytes(inp_b) <= source_bytes:
        raise RuleSkipped(f"matmul B does not expand compact storage ({source_bytes} -> {_nbytes(inp_b)} bytes)")

    frag = open_fragment(match.graph, [match.graph.nodes[nid] for nid in boundaries])
    serial = 0

    def clone(nid: str) -> str:
        nonlocal serial
        if nid in boundaries:
            return nid
        node = match.graph.nodes[nid]
        cloned_inputs = [clone(buf) for buf in node.inputs]
        serial += 1
        # A numerical CastOp is exactly an assignment into the declared destination dtype.
        # Spell it as the generic elementwise copy while treeifying so loop fusion may keep the
        # conversion at its scalar point of use. A same-width BitcastOp stays explicit and its
        # generic loop lifting likewise keeps it at the scalar use site.
        op = ElementwiseOp(op="copy") if isinstance(node.op, CastOp) else copy.copy(node.op)
        op.inputs, op.outputs = {}, {}
        cloned_id = frag.add_node(
            op=op,
            inputs=cloned_inputs,
            output=Tensor(f"{node.output.name}__tree{serial}", node.output.shape, node.output.dtype),
        )
        return cloned_id

    root = clone(inp_b.id)
    frag.outputs = [root]
    match.consumed = interior
    match.output = inp_b.id
    return frag
