"""Merge sibling ``LinearOp``s sharing one activation into a single linear over an N-concat weight.

Stock engines concatenate q/k/v (and gate/up) weights and biases at load and run ONE gemm per family; emmy
traces the HF module structure and deploys each projection separately — with split-K that is a
partial+finalize pair (plus a memset) per sibling where one launch would do. This rule is the
general form of that fusion: two ``LinearOp``s consuming the SAME activation node, whose
weights and optional biases are pristine checkpoint constants (a source, no ``load_ops``, no
other consumer), merge into one ``LinearOp`` over load-time concatenated parameters
(``ConstantOp.source_parts`` — the loader concatenates the parts along axis 0, the ``(N, K)``
out-features axis for weights and the ``N`` axis for biases, before running any ``load_ops``
chain, so the concat costs nothing at run time). Each original output is re-derived as a
``SliceOp`` N-window view of the merged output; the slice decomposes to a pure indexmap that
downstream fusion folds into the real consumers as offset loads of the one workspace.

The merged edge is a PLAIN matmul — a new shape, nothing else: recognition, split-K, the weight
transports and the golden machinery all apply unchanged. One rewrite merges ALL eligible
siblings at once (multi-output splice), with the concat in graph-insertion order — the outcome
is one canonical concat regardless of which sibling anchored the match or how the engine
enumerated it (``topological_order`` tie-breaks are not stable across processes, and the concat
order is buffer-layout ABI: goldens and packs key on it).

Guards, each a ``RuleSkipped``:

- **graph-output / view-only-output sibling** — a merged buffer cannot BE two outputs, and an
  output reached from the linear through layout ops only (the norm-free q/k/v capture ABI:
  ``proj → view → reshape → output``) would demote the free view to a materialized copy kernel,
  trading the saved launches back. Models whose projections feed real compute (gemma/Qwen's
  per-head norms, any in-graph SDPA) merge freely.
- **shared parameter** (tied embeddings) / **mixed bias presence** / **non-pristine parameter**
  (a ``load_ops`` chain already folded in, or a ``source_graph`` bind record — a folded constant
  subgraph has no concat-of-paths spelling) — the concat semantics need a bare ``(N, K)``
  checkpoint weight and, when present, a bare ``(N,)`` checkpoint bias per part.
- **mismatched K / dtype** — not the same contraction.

Concat order is the graph insertion order of the two linears (deterministic), and the earlier
sibling anchors the merge, so one canonical pair fires regardless of match enumeration order.
"""

from __future__ import annotations

from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.base import ConstantOp
from emmy.compiler.ir.frontend.ir import CatOp, LinearOp, ReshapeOp, SliceOp, TransposeOp, UnsqueezeOp
from emmy.compiler.ir.tensor.ir import IndexMapOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.frontend.decomposition._helpers import open_fragment

PATTERN = [Pattern("lin", LinearOp)]

_LAYOUT_OPS = (TransposeOp, ReshapeOp, SliceOp, UnsqueezeOp, CatOp, IndexMapOp)


def _parameter(graph: Graph, lin: Node, index: int, rank: int) -> Node | None:
    """A pristine, exclusively-owned checkpoint parameter of ``rank`` — else ``None``."""
    expected_inputs = 3 if lin.op.has_bias else 2
    if len(lin.inputs) != expected_inputs:
        return None
    param = graph.nodes.get(lin.inputs[index])
    if param is None or not isinstance(param.op, ConstantOp):
        return None
    op = param.op
    if op.value is not None or op.context_value is not None or op.load_ops:
        return None
    if op.source_path is None and not op.source_parts:
        # Covers ``source_graph`` bind records too (a folded constant subgraph is not a bare
        # checkpoint tensor — its evaluation has no concat-of-paths spelling).
        return None
    if graph.users(param.id) != {lin.id}:
        return None
    shape = tuple(param.output.shape)
    if len(shape) != rank or not all(d.is_static for d in shape):
        return None
    return param


def _weight(graph: Graph, lin: Node) -> Node | None:
    return _parameter(graph, lin, 1, 2)


def _bias(graph: Graph, lin: Node) -> Node | None:
    return _parameter(graph, lin, 2, 1) if lin.op.has_bias else None


def _view_only_path_to_output(graph: Graph, nid: str) -> bool:
    """``nid`` reaches a graph output through layout ops only (or IS one) — the buffer's
    identity is the capture ABI, so a slice view of a merged workspace would have to
    materialize as a copy kernel there."""
    outputs = set(graph.outputs)
    seen: set[str] = set()
    stack = [nid]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        if cur in outputs:
            return True
        for uid in graph.users(cur):
            u = graph.nodes.get(uid)
            if u is not None and isinstance(u.op, _LAYOUT_OPS):
                stack.append(uid)
    return False


def _mergeable(graph: Graph, lin: Node, a_id: str) -> Node | None:
    """The linear's weight node when ``lin`` is eligible to merge over activation ``a_id``."""
    if not isinstance(lin.op, LinearOp):
        return None
    if not lin.inputs or lin.inputs[0] != a_id:
        return None
    if lin.id in graph.outputs or _view_only_path_to_output(graph, lin.id):
        return None
    weight = _weight(graph, lin)
    bias = _bias(graph, lin) if lin.op.has_bias else None
    if weight is None or (lin.op.has_bias and bias is None):
        return None
    if bias is not None and bias.output.shape[0] != weight.output.shape[0]:
        return None
    return weight


def _parts(w: Node) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Flatten a weight constant to its ``(path, pre-chain shape)`` concat parts — an already
    merged weight contributes its own parts, so repeated merges stay a flat concat."""
    op = w.op
    if op.source_parts:
        return op.source_parts
    shape = tuple(op.source_shape) if op.source_shape else tuple(int(d.as_static()) for d in w.output.shape)
    return ((op.source_path, shape),)


def rewrite(match: Match, lin: Node) -> Graph | None:
    graph = match.graph
    a_id = lin.inputs[0] if lin.inputs else None
    if a_id is None or _mergeable(graph, lin, a_id) is None:
        raise RuleSkipped(f"{lin.id!r} is not a mergeable sibling linear (bias / weight / output constraints)")

    order = {nid: i for i, nid in enumerate(graph.nodes)}
    group = sorted(
        (graph.nodes[cid] for cid in graph.users(a_id) if _mergeable(graph, graph.nodes[cid], a_id) is not None),
        key=lambda n: order[n.id],
    )
    lead_w = _weight(graph, group[0])
    lead_b = _bias(graph, group[0]) if group[0].op.has_bias else None
    group = [
        n
        for n in group
        if (w := _weight(graph, n)).output.shape[-1] == lead_w.output.shape[-1]  # same K
        and n.op.has_bias == group[0].op.has_bias
        and w.output.dtype.name == lead_w.output.dtype.name  # concat must not erase a dtype boundary
        and n.output.dtype.name == group[0].output.dtype.name
        and w.op.source_dtype == lead_w.op.source_dtype
        and (
            lead_b is None
            or (
                (b := _bias(graph, n)) is not None
                and b.output.dtype.name == lead_b.output.dtype.name
                and b.op.source_dtype == lead_b.op.source_dtype
            )
        )
    ]
    if len(group) < 2:
        raise RuleSkipped(f"{lin.id!r} has no mergeable sibling linear on {a_id!r}")

    weights = [_weight(graph, n) for n in group]
    biases = [_bias(graph, n) for n in group] if group[0].op.has_bias else []
    ns = [int(w.output.shape[0].as_static()) for w in weights]
    x = graph.nodes[a_id]

    frag = open_fragment(graph, [x])
    lin_name = "__cat__".join(n.output.name for n in group)
    w_cat = frag.add_node(
        op=ConstantOp(
            name="__cat__".join(w.op.name for w in weights),
            source_parts=tuple(p for w in weights for p in _parts(w)),
            source_shape=(sum(ns), int(lead_w.output.shape[-1].as_static())),
            source_dtype=lead_w.op.source_dtype,
        ),
        inputs=[],
        output=Tensor(f"{lin_name}_w", (sum(ns),) + tuple(lead_w.output.shape[1:]), lead_w.output.dtype),
    )
    merged_inputs = [x.id, w_cat]
    if biases:
        lead_b = biases[0]
        b_cat = frag.add_node(
            op=ConstantOp(
                name="__cat__".join(b.op.name for b in biases),
                source_parts=tuple(p for b in biases for p in _parts(b)),
                source_shape=(sum(ns),),
                source_dtype=lead_b.op.source_dtype,
            ),
            inputs=[],
            output=Tensor(f"{lin_name}_b", (sum(ns),), lead_b.output.dtype),
        )
        merged_inputs.append(b_cat)
    merged = frag.add_node(
        op=LinearOp(has_bias=bool(biases)),
        inputs=merged_inputs,
        output=Tensor(lin_name, tuple(group[0].output.shape[:-1]) + (sum(ns),), group[0].output.dtype),
    )
    output_map: dict[str, str] = {}
    offset = 0
    for n, width in zip(group, ns, strict=True):
        # Static ints in the op's ``shape`` field (SliceOp.forward keys the slice end off an
        # int extent); the fragment Tensor keeps the graph's Dim shapes.
        sl = frag.add_node(
            op=SliceOp(shape=tuple(int(d.as_static()) if d.is_static else d for d in n.output.shape), dim=-1, start=offset),
            inputs=[merged],
            output=Tensor(n.output.name, n.output.shape, n.output.dtype),
        )
        output_map[n.id] = sl
        offset += width
    frag.outputs = list(output_map.values())

    match.consumed = {n.id for n in group} | {w.id for w in weights} | {b.id for b in biases}
    match.output = output_map
    return frag
