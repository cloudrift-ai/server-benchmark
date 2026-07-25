"""Inline a deferred split-reduce finalize into its consumers — realize ``PLACE@fin=fuse``.

The ``g<w>k`` deferred-kernel finalize (``030_split_reduce``) is a tiny cross-partition fold over a
``ws[cta, *cell]`` ``__partial`` workspace — one extra launch and one in-stream sync point per split
matvec. In the M=1 decode twins that chain link costs ~2-3 µs of serialized wall per matvec while the
fold itself is a handful of f32 adds per output cell, so a CONSUMER can open with the fold instead:
every consumer ``Load`` of the finalize output is replaced by the finalize's own body — the seeds, the
``_ksplit`` fold loop over the workspace, the original projection — σ-substituted from the finalize's
output-cell vars to that ``Load``'s index and SSA/axis-renamed per site (``__fin<n>``). The redundant
per-read-site recompute is the redundant-statistic pattern; no barrier is needed because each thread
folds exactly the cells it reads. Numerics: the same seed → in-order partition fold → projection, on
the f32 state instead of its rounded re-load — the coop-reduce class of reassociation, accuracy gates
judge.

Anchored on the CONSUMER (the rewrite that changes is the consumer's body); the finalize node itself
is never edited — when the LAST consumer is rewired onto the workspace, the splice's
``remove_orphans`` deletes the finalize and its dead output buffer. Each firing handles one consumer;
the pass-scan restart finds the next.

Gates (all structural):

- **decision** — the ``PLACE@fin`` stamp on the finalize tile (threaded from the split node's knob
  row by ``030``) or the ``PLACE@fin`` pin; the built-in default is ``cut`` (keep the kernel), so
  this realizer only fires on ``fuse``. Evidence-only: an unseeded site never pays.
- **no graph-boundary crossing** — a finalize output that IS a graph output stays materialized.
- **every consumer inlinable** — otherwise the finalize kernel survives for the un-inlinable reader
  and the inlined folds would be pure overhead. Consumer shapes handled: a flat ``Map`` (pointwise),
  a ``Map`` over a plain ``Reduction`` source (norm stat + sweep), a bare ``Reduction``. A
  ``Contraction`` consumer bails (its operand loads live inside factorized tiers — v2).
- **single-component workspace** (``ws[cta, *cell]``) — the multi-component (flash ``(m, l, O)``)
  finalize's per-cell fold is not a few adds; it stays a kernel.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Body, Load, Stmt, Write
from emmy.compiler.ir.tile import Map, Reduction, TileOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.search.space import place_decision

PATTERN = [Pattern("consumer", TileOp)]

_SPLIT = "_ksplit"  # 030_split_reduce's cross-CTA split axis


def _is_finalize(node: Node) -> bool:
    """A deferred split-reduce finalize: a flat-``Map`` TileOp whose sole input is a ``__partial``
    workspace."""
    op = node.op
    return (
        isinstance(op, TileOp)
        and isinstance(op.op, Map)
        and op.op.source is None
        and len(node.inputs) == 1
        and node.inputs[0].endswith("__partial")
    )


def _op_bodies(op: TileOp):
    """The stmt bodies of an inlinable-shaped TileOp (``Map`` body + a plain ``Reduction``
    source's partial), for read scans."""
    inner = op.op
    if isinstance(inner, Map):
        yield inner.body
        if isinstance(inner.source, Reduction):
            yield inner.source.partial
    elif isinstance(inner, Reduction):
        yield inner.partial


def _reads(node: Node, buf: str) -> bool:
    """``node``'s op reads ``buf`` — through a graph edge OR a body-level ``Load`` (a
    ``030_split_reduce`` finalize's projection loads its sweep operands without an edge)."""
    if buf in node.inputs:
        return True
    if isinstance(node.op, TileOp) and node.op.op is not None:
        return any(isinstance(s, Load) and s.input == buf for b in _op_bodies(node.op) for s in b.iter())
    return False


def _readers(graph: Graph, buf: str) -> list[Node]:
    """Every node reading ``buf`` (edge consumers ∪ body-level readers)."""
    return [n for nid, n in graph.nodes.items() if nid != buf and _reads(n, buf)]


def _inlinable(op) -> bool:
    """Consumer op shapes whose reads are plain ``Load`` stmts this rule can rewrite."""
    if not isinstance(op, TileOp) or op.op is None:
        return False
    inner = op.op
    if isinstance(inner, Map):
        src = inner.source
        if src is None:
            return True
        return isinstance(src, Reduction) and src.source is None and all(isinstance(x, Stmt) for x in src.partial)
    if isinstance(inner, Reduction):
        return inner.source is None and all(isinstance(x, Stmt) for x in inner.partial)
    return False  # Contraction: operand loads live inside factorized tiers


def _template(fin: Map, out_name: str, ws_rank: int):
    """The finalize body split into (fold stmts, final ``Write``, deep-defined SSA names) — or
    ``None`` when the body is not the single-scalar-store single-component shape."""
    writes = [s for s in fin.body if isinstance(s, Write)]
    if len(writes) != 1 or writes[0].output != out_name or writes[0].atomic or not writes[0].is_scalar:
        return None
    w = writes[0]
    if len(w.index) + 1 != ws_rank:
        return None  # comp lead axis — multi-component carrier state
    stmts = tuple(s for s in fin.body if s is not w)
    defs: set[str] = set()
    for s in Body(stmts).iter():
        defs.update(s.defines())
    return stmts, w, defs


def _inline_at(stmts, w: Write, defs: set[str], load: Load, suffix: str):
    """The finalize fold re-instantiated at one consumer read site: cell vars σ-substituted to the
    ``Load``'s index, every SSA name and the split axis renamed per site, the final store replaced
    by binding the projected value to the ``Load``'s own SSA name. ``None`` when the indices can't
    be matched structurally."""
    sigma: dict = {}
    for cell_e, idx_e in zip(w.index, load.index, strict=True):
        if isinstance(cell_e, Var):
            sigma[cell_e.name] = idx_e
        elif isinstance(cell_e, Literal):
            if not (isinstance(idx_e, Literal) and idx_e.value == cell_e.value):
                return None
        else:
            return None
    split_new = f"{_SPLIT}{suffix}"
    sigma[_SPLIT] = Var(split_new)
    rename = {d: f"{d}{suffix}" for d in defs}
    rename[w.value] = load.name

    def ren(n: str) -> str:
        return rename.get(n, n)

    def ax(a):
        return replace(a, name=split_new) if a.name == _SPLIT else a

    sg = Sigma(sigma)
    return tuple(s.rewrite(ren, sg, ax) for s in stmts)


def rewrite(match: Match, root: Node) -> Graph | None:
    graph = match.graph
    cons: TileOp = root.op
    if not _inlinable(cons):
        raise RuleSkipped("consumer shape is not inlinable")

    candidates = list(dict.fromkeys(root.inputs))
    candidates += [s.input for b in _op_bodies(cons) for s in b.iter() if isinstance(s, Load) and s.input not in candidates]
    producer = None
    for inp in candidates:
        p = graph.nodes.get(inp)
        if p is None or not _is_finalize(p):
            continue
        stamped = p.op.knobs.get("PLACE@fin")
        if (stamped or place_decision("fin")) != "fuse":
            continue
        producer = p
        break
    if producer is None:
        raise RuleSkipped("no PLACE@fin=fuse finalize among the consumer's producers")
    if producer.id in graph.outputs:
        raise RuleSkipped("finalize output is a graph output — must stay materialized")
    if not all(_inlinable(r.op) for r in _readers(graph, producer.id)):
        raise RuleSkipped("a sibling reader can't inline — the finalize kernel would survive anyway")

    ws_id = producer.inputs[0]
    ws_node = graph.nodes.get(ws_id)
    if ws_node is None:
        raise RuleSkipped("workspace node missing")
    tpl = _template(producer.op.op, producer.id, len(ws_node.output.shape))
    if tpl is None:
        raise RuleSkipped("finalize body is not the single-component single-store shape")
    stmts, w, defs = tpl
    sites = 0

    def _rewrite_op(tile: TileOp) -> TileOp | None:
        """``tile`` with every ``Load`` of the finalize output replaced by the inlined fold —
        ``None`` when any read site can't be matched (nothing partially rewritten)."""
        nonlocal sites
        matched = 0
        failed = False

        def fn(s: Stmt):
            nonlocal sites, matched, failed
            if isinstance(s, Load) and s.input == producer.id:
                if failed or not s.is_scalar or len(s.index) != len(w.index):
                    failed = True
                    return s
                sites += 1
                matched += 1
                new = _inline_at(stmts, w, defs, s, f"__fin{sites}")
                if new is None:
                    failed = True
                    return s
                return new
            return s

        inner = tile.op
        if isinstance(inner, Map):
            new_body = inner.body.map(fn)
            src = inner.source
            if isinstance(src, Reduction):
                src = replace(src, partial=src.partial.map(fn))
            new_inner = replace(inner, body=new_body, source=src)
        else:  # bare Reduction (gated by _inlinable)
            new_inner = replace(inner, partial=inner.partial.map(fn))
        if failed or matched == 0:
            return None
        # The consumer realizes the placement — stamp it so realized-knob checks (the ``--ab``
        # pin reproducibility gate, the audit) see ``PLACE@fin=fuse`` on the surviving kernels.
        new = replace(tile, op=new_inner, name=f"{tile.name}__fin" if tile.name else "")
        new.knobs = {**tile.knobs, "PLACE@fin": "fuse"}
        return new

    new_root = _rewrite_op(cons)
    if new_root is None:
        raise RuleSkipped("a read of the finalize output can't be matched to its cell index")

    # The splice's ``remove_orphans`` deletes the finalize as soon as its last EDGE consumer is
    # rewired — but body-level readers hold no edge, so the firing that removes that last edge
    # must inline every remaining body-only reader IN THE SAME rewrite (in-place op swap, the
    # 005_delegate_zero_init precedent) or they'd be left loading a dead buffer. Compute every
    # sibling rewrite BEFORE committing anything, so a single unmatchable site aborts whole.
    edge_consumers = set(graph.consumers(producer.id))
    last_edge = edge_consumers <= {root.id}
    sibling_swaps: list[tuple[Node, TileOp]] = []
    if last_edge:
        for r in _readers(graph, producer.id):
            if r.id == root.id:
                continue
            new_op = _rewrite_op(r.op)
            if new_op is None:
                raise RuleSkipped("a sibling body-reader can't inline — aborting to keep the finalize alive")
            sibling_swaps.append((r, new_op))
    for r, new_op in sibling_swaps:
        new_op.inputs = dict(r.op.inputs)
        new_op.outputs = dict(r.op.outputs)
        r.op = new_op

    new_inputs = [ws_id if i == producer.id else i for i in root.inputs]
    if ws_id not in new_inputs:
        new_inputs.append(ws_id)  # a body-only reader gains the workspace edge it now loads from
    frag = Graph()
    for ext in dict.fromkeys(new_inputs):
        n = graph.nodes[ext]
        frag.add_node(op=InputOp(), inputs=[], output=n.output, node_id=ext)
    new_id = f"{root.id}__fin"
    frag.add_node(
        op=new_root,
        inputs=list(dict.fromkeys(new_inputs)),
        output=Tensor(root.output.name, root.output.shape, root.output.dtype),
        node_id=new_id,
    )
    frag.outputs = [new_id]
    return frag
