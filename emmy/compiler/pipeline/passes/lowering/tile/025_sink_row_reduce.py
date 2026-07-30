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
  ``T__sq`` is output slot 1 of the producer node itself — planned/allocated per buffer, with
  B' reading it through an ordinary buffer edge.
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
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import Assign, Body, Load, Loop, RowAccum, Write
from emmy.compiler.ir.tile import Map, TileOp
from emmy.compiler.ir.tile.ops import reduce_plan
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.loop.stamp._stamp import restamp_structural_features
from emmy.compiler.pipeline.passes.lowering.tile._sink import bind_sinkable_stat, flat_index_expr
from emmy.compiler.pipeline.search.space import place_decision

PATTERN = [Pattern("root", TileOp)]


def _producer_writes(pop: TileOp, buf: str) -> list[Write] | None:
    """The producer's eligible stores of ``buf`` — plain scalar TOP-LEVEL ``Write``\\ s in a flat
    ``Map`` body covering every store of the buffer (gated in the negative: an atomic write, a
    nested/loop store, or an mma epilogue all return ``None``). One write is the per-cell
    finalize; several are a register-tiled sweep's per-cell stores (e.g. the ``f4`` pointwise
    tile — four writes per thread), grouped by :func:`_same_row_base`."""
    pmap = pop.op
    if not isinstance(pmap, Map) or pmap.source is not None:
        return None
    all_writes = [s for s in pmap.body.iter() if isinstance(s, Write) and s.output == buf]
    top_writes = [s for s in pmap.body if isinstance(s, Write) and s.output == buf]
    if not top_writes or len(all_writes) != len(top_writes):
        return None
    if any(w.atomic or not w.is_scalar for w in top_writes):
        return None
    return top_writes


def _same_row_base(flats: list, n: int):
    """Prove a write group lands in ONE row of the ``n``-wide reduce and return the base flat
    address, or ``None``. The proof: the flat addresses share affine coefficients and their
    anchors form the dense run ``base .. base+W-1`` with ``base ≡ 0 (mod W)`` (every coefficient
    and the base anchor divisible by ``W``) and ``W | n`` — then ``base % n ≤ n − W`` (a multiple
    of W below n), so ``base+k`` stays in ``base``'s row for every ``k < W``."""
    from emmy.compiler.ir.expr import SimplifyCtx, affine_form  # noqa: PLC0415

    if len(flats) == 1:
        return flats[0]
    w = len(flats)
    if n % w != 0:
        return None
    forms = []
    for f in flats:
        vars_ = set(f.free_vars())
        form = affine_form(f, vars_)
        if form is None:
            return None
        anchor = form[0].simplify(SimplifyCtx.empty())
        from emmy.compiler.ir.expr import Literal  # noqa: PLC0415

        if not isinstance(anchor, Literal):
            return None
        forms.append((int(anchor.value), form[1]))
    coeffs0 = forms[0][1]
    if any(c != coeffs0 for _, c in forms[1:]):
        return None
    anchors = sorted(a for a, _ in forms)
    base_anchor = anchors[0]
    if anchors != list(range(base_anchor, base_anchor + w)):
        return None
    if base_anchor % w != 0 or any(c % w != 0 for c in coeffs0.values()):
        return None
    return flats[[a for a, _ in forms].index(base_anchor)]


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
    if graph.buffer(sq) is not None:
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
    writes = _producer_writes(pop, src)
    if writes is None:
        raise RuleSkipped("producer has no plain complete-value stores of the reduced tensor (atomic partial / mma epilogue)")
    t = tile.inputs.get(src)
    if t is None or any(len(w.index) != len(t.shape) for w in writes) or not all(d.is_static for d in t.shape):
        raise RuleSkipped("producer store rank/shape doesn't cover the reduced tensor statically")
    flats = [flat_index_expr(w.index, t.shape) for w in writes]
    base_flat = _same_row_base(flats, binding.n)
    if base_flat is None:
        raise RuleSkipped("producer's write group isn't provably one row — per-write RowAccums not emitted (v2)")

    # --- producer side: B's reduce cell re-evaluated on each just-stored value, the per-write
    # terms summed per thread, then ONE RowAccum. The chains' defs are suffixed so they can't
    # collide with producer SSA names; the cell's Load of T is substituted with each Write's
    # (pre-round f32) value.
    from emmy.compiler.ir.elementwise import ElementwiseImpl  # noqa: PLC0415

    contrib: list[Assign] = []
    terms: list[str] = []
    for i, w in enumerate(writes):
        sub = {binding.load.name: w.value}
        for a in binding.chain:
            renamed = Assign(name=f"{a.name}__snk{i}", op=a.op, args=tuple(sub.get(x, x) for x in a.args), dtype=a.dtype)
            sub[a.name] = renamed.name
            contrib.append(renamed)
        terms.append(sub.get(binding.value, binding.value))
    value = terms[0]
    for i, term in enumerate(terms[1:]):
        summed = Assign(name=f"{binding.value}__snks{i}", op=ElementwiseImpl("add"), args=(value, term))
        contrib.append(summed)
        value = summed.name
    accum = RowAccum(dst=sq, flat=base_flat, n=binding.n, value=value)
    pmap: Map = pop.op
    body = list(pmap.body)
    at = max(body.index(w) for w in writes) + 1
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
    frag.add_node(
        op=new_pop,
        inputs=list(producer.inputs),
        outputs=[producer.output, Tensor(sq, (Dim(binding.rows),), F32)],
        node_id=src,
    )
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
