"""Realize ``PLACE@stat=sink`` — migrate a norm kernel's row statistic into its producer.

``020_cut_edge``'s sibling for the PLACE codec's ``stat`` element, and like it a pure REALIZER:
the decision lives in the schedule (the ``PLACE@stat`` stamp the recognizer threads onto the
offered row), the graph carries the kernel shape. The matched kernel B is the fused norm form —
``Map(body=[π…, sweep Loop], source=Reduction(PLANAR))`` reading tensor ``T[row, n]`` (the
``Σx²`` stat + scale sweep: post-attn/post-ff rms norms, the per-head qknorm sweeps) — a
grid-per-row latency-bound kernel whose reduce re-reads a tensor its producer A just computed.
The rewrite:

- **producer epilogue** — A (a settled ``TileOp`` writing each ``T`` cell exactly ONCE with a
  plain top-level ``Write`` — a split-K FINALIZE or a pointwise sweep; NOT an atomic partial,
  whose per-partition values are incomplete, and not an mma ``RegStore`` epilogue — that arm is
  v2) gains B's reduce cell re-evaluated on the just-stored value (``v·v`` on the ``Write``'s
  SSA value) plus a :class:`RowAccum` accumulating it into a fresh f32 row buffer
  ``T__sq[row]``, ``row = flat(cell) / n`` (zero-init'd per launch via the ordinary
  ``zero_outputs`` memset machinery — ``RowAccum`` is detected exactly as ``RegStore.atomic``).
- **aux node** — ``T__sq`` becomes an :class:`AuxOutputOp` graph node whose sole input is A:
  no launch of its own, but a real node so the buffer is planned/allocated and B' depends on A
  through ordinary edges.
- **consumer** — B re-emits as an UN-mapped ``LoopOp`` (the 020-cut re-entry idiom): its
  ``Reduction`` drops to a ``Load`` of ``T__sq[row]`` and the projection rides verbatim inside
  the free row loops — the restarted scan hands it back to ``010_recognize``, which peels BOTH
  the row and sweep axes free and schedules the wide 2-D pointwise grid the fused form never
  had.

Numerics: the statistic is the SAME f32 sum reordered (a warp/atomic tree instead of one
thread's serial fold, over the pre-store f32 values instead of their rounded re-loads) — the
coop-reduce class of reassociation; accuracy gates judge.

Waiting is structural: a producer still un-recognized (``LoopOp``) or carrying an unconsumed
GRID split waits for ``010``/``030`` via ``RuleSkipped`` (the scan restarts on every applied
rewrite, so this rule re-fires once the producer settles). A producer whose settled form is
ineligible (atomic / mma epilogue / nested store) skips permanently and B simply deploys the
row's own (coop) schedule — the sink is evidence-only, so an unseeded or unrealizable site
never pays for the miss."""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.base import AuxOutputOp, InputOp
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import Assign, Body, Load, Loop, RowAccum, Write
from emmy.compiler.ir.tile import Map, TileOp
from emmy.compiler.ir.tile.ops import reduce_plan
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.loop.stamp._stamp import restamp_structural_features
from emmy.compiler.pipeline.passes.lowering.tile._sink import bind_sinkable_stat, flat_index_expr
from emmy.compiler.pipeline.search.space import place_decision

PATTERN = [Pattern("root", TileOp)]


def _producer_write(pop: TileOp, buf: str) -> Write | None:
    """The producer's single eligible store of ``buf`` — a plain scalar top-level ``Write`` in a
    flat ``Map`` body that is the buffer's ONLY store (gated in the negative: an atomic write, a
    nested/loop store, an mma epilogue, or a multi-store body all return ``None``)."""
    pmap = pop.op
    if not isinstance(pmap, Map) or pmap.source is not None:
        return None
    all_writes = [s for s in pmap.body.iter() if isinstance(s, Write) and s.output == buf]
    top_writes = [s for s in pmap.body if isinstance(s, Write) and s.output == buf]
    if len(all_writes) != 1 or len(top_writes) != 1:
        return None
    w = top_writes[0]
    if w.atomic or not w.is_scalar:
        return None
    return w


def rewrite(match: Match, root: Node) -> Graph | None:
    tile: TileOp = root.op
    stamped = tile.knobs.get("PLACE@stat")
    decision = stamped if stamped in ("fuse", "sink") else place_decision("stat")
    if decision != "sink":
        raise RuleSkipped("PLACE@stat is fuse — the row statistic stays local")
    if tile.op is None:
        raise RuleSkipped("kernel-less TileOp — nothing to sink")
    binding = bind_sinkable_stat(tile.op, tile.place.free, tile.inputs)
    if binding is None:
        raise RuleSkipped("not a sinkable row-stat form")
    graph = match.graph
    src = binding.src
    sq = f"{src}__sq"
    if sq in graph.nodes:
        raise RuleSkipped(f"stat already sunk — {sq} exists")
    producer = graph.nodes.get(src)
    if producer is None or src in graph.inputs:
        raise RuleSkipped("reduced tensor is a graph input — no in-graph producer to sink into")
    pop = producer.op
    if isinstance(pop, LoopOp):
        raise RuleSkipped("producer not yet recognized — waiting for 010")
    if not isinstance(pop, TileOp) or pop.op is None:
        raise RuleSkipped("producer is not a lowerable tile kernel")
    pplan = reduce_plan(pop)
    if pplan is not None and pplan.needs_split:
        raise RuleSkipped("producer's cross-CTA split pending — waiting for 030")
    write = _producer_write(pop, src)
    if write is None:
        raise RuleSkipped("producer has no single plain complete-value store of the reduced tensor (atomic partial / mma epilogue)")
    t = tile.inputs.get(src)
    if t is None or len(write.index) != len(t.shape) or not all(d.is_static for d in t.shape):
        raise RuleSkipped("producer store rank/shape doesn't cover the reduced tensor statically")

    # --- producer side: B's reduce cell re-evaluated on the just-stored value + the RowAccum.
    # The chain's defs are suffixed so they can't collide with producer SSA names; the cell's
    # Load of T is substituted with the Write's (pre-round f32) value.
    sub = {binding.load.name: write.value}
    contrib: list[Assign] = []
    for a in binding.chain:
        renamed = Assign(name=f"{a.name}__snk", op=a.op, args=tuple(sub.get(x, x) for x in a.args), dtype=a.dtype)
        sub[a.name] = renamed.name
        contrib.append(renamed)
    value = sub.get(binding.value, binding.value)
    accum = RowAccum(dst=sq, flat=flat_index_expr(write.index, t.shape), n=binding.n, value=value)
    pmap: Map = pop.op
    body = list(pmap.body)
    at = body.index(write) + 1
    new_pop = replace(pop, op=replace(pmap, body=Body((*body[:at], *contrib, accum, *body[at:]))))

    # --- consumer side: the Reduction drops to a Load of the row stat; the projection rides
    # verbatim inside the free row loops. Un-mapped, so 010 re-recognizes it wide.
    acc_load = Load(name=binding.acc, input=sq, index=(binding.row_expr(),))
    inner = Body((acc_load, *tile.op.body))
    for ax in reversed(tile.place.free):
        inner = Body((Loop(axis=ax, body=inner),))
    consumer = LoopOp(body=inner)

    out = root.output
    frag = Graph()
    ext_inputs = dict.fromkeys((*producer.inputs, *(i for i in root.inputs if i != src)))
    for inp in ext_inputs:
        n = graph.nodes[inp]
        frag.add_node(op=InputOp(), inputs=[], output=n.output, node_id=inp)
    frag.add_node(op=new_pop, inputs=list(producer.inputs), output=producer.output, node_id=src)
    frag.add_node(op=AuxOutputOp(), inputs=[src], output=Tensor(sq, (Dim(binding.rows),), F32), node_id=sq)
    cons_inputs = [*(i for i in root.inputs if i != src and i in consumer.inputs), src, sq]
    frag.add_node(op=consumer, inputs=cons_inputs, output=out, node_id=out.name)
    for node in frag.nodes.values():
        if isinstance(node.op, LoopOp):
            restamp_structural_features(node.op, frag)
    frag.outputs = [out.name]
    # This rewrite consumes TWO nodes (B and its producer) and re-emits both — declare the
    # extra consumption and the two-output redirect map (splice's dict form). The identity
    # snapshot keeps ``Match.is_alive`` meaningful for the added node (a remapped/deferred
    # match would otherwise read a missing snapshot as "dead").
    match.consumed.add(src)
    match._identities[src] = id(producer)  # noqa: SLF001 — the match owns the snapshot; no public setter
    match.output = {src: src, match.root_node_id: out.name}
    return frag
