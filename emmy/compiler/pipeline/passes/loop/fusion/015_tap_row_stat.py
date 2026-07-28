"""Fission a fused norm kernel at loop level: its row statistic becomes a TAP in the producer.

The inverse resting state of the old ``025_sink_row_reduce`` realizer: instead of reconstructing
the write→row correspondence AFTER the producer's schedule settles (flat-address bijection proofs,
``RuleSkipped`` waiting, a bespoke stored fold stmt), the statistic fuses into the producer HERE —
at ``merge_loop_ops`` time, while the row index is still a live loop variable — as an ordinary
atomic-accumulate ``Write`` (``ir/loop/tap.py``, the stored form). The escape (``PLACE@stat=fuse``,
option-0) is realized by cutting the tap back out at tile lowering (``015_cut_stat_tap``): sinking
IN needs eligibility gates, cutting OUT is always legal, so the fused state is safe to make
canonical — the ``PLACE@stat`` fork, its evidence-only status, and the measured wins all live at
the tapped producer's fork now.

Matches a producer→consumer ``LoopOp`` pair where the consumer ``B`` is the fully-assembled fused
norm form over the producer ``A``'s output ``T``:

    for rows…: [prefix…, for n: (Load T, term chain, additive Accum), π…, sweep Loop reading T]

Because rule ``010_merge_loop_ops`` precedes this one in scan order, ``B`` only reaches here once
every merge has exhausted — a norm whose sole consumer absorbs it (the norm→linear computed-A
edge) fissions never, and ``B`` is maximal (residual adds already in its sweep). The rewrite is
**tap-only** (the sweep needs the completed stat, so it keeps its own kernel):

- **A′** — ``A`` with the consumer's per-cell term chain (``__tap``-minted names, evaluated on the
  pre-store value) + the atomic tap ``Write T__sq[rows…] += term`` spliced after its store of
  ``T``; ``T__sq`` joins as output slot 1 (a true multi-output node). Fan-out on ``T`` does not
  block the tap — the tap consumes nothing, ``T`` stays ``A``'s output.
- **S** — ``B`` with its statistic ``Reduction`` dropped to a ``Load`` of ``T__sq``: the wide
  pointwise sweep, a separate node. ``010_recognize`` DEFERS it until the producer's ``PLACE@stat``
  decision settles, so the ``fuse`` cut-out re-welds the VERBATIM loop — bit-parity with the
  never-fissioned form by construction.

The index correspondence comes from the same σ-solve ``splice_graph`` does, anchored at ``A``'s
``Write`` instead of a ``Load``: pairing the write index against the statistic's read index
classifies each position as row / reduce / mixed-radix (the per-head norm over a flattened
``heads·W`` axis — its row coordinate is ``index // W``, no flat-address algebra). Gates, all in
the negative:

- ``A`` must be an in-graph ``LoopOp`` with ONE output and ONE store of ``T`` (an input-norm has
  no in-graph producer and never matches — the refusal carries over by construction); ``A`` (or
  ``B``) being a graph output is FINE (``T`` stays materialized, ``S`` inherits ``B``'s id).
- ``A`` must not carry a rowmax (``maximum`` ``Accum``): a softmax / flash offer site's operands
  and certification are owned by ``try_flash`` — a tap would de-certify the form.
- every extent this rule reasons about is STATIC, and the statistic's read must cover ``T``
  exactly (each row folds exactly the reduce extent, every cell in exactly one row). The stored
  form expresses symbolic rows for free, but offering them would create tapped kernels on the
  un-lifted dynamic tier, where no ``PLACE@stat`` fork exists to cut them back out.
- a DEPENDENT stat chain (softmax's ``Σ exp(x − max)``) is algebraically un-tappable — the
  per-cell term reads another statistic's final value, so the single-``Load``-pure-chain gate
  refuses it structurally (impossibility, not a TODO).

The tap adds O(numel) work to ``A`` (one fma-class chain + one accumulate per cell) — far under
``_BLOWUP_FACTOR``; no metric check is needed. ``010_merge_loop_ops`` carries the mirror brakes
(a multi-output producer / any ``__sq`` reader stays out of later merges), so the fission
artifacts are never re-fused and every OTHER pending merge sees the graph it would have seen.
"""

from __future__ import annotations

from math import prod

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal, SimplifyCtx, Var, affine_form
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.loop.splicer import solve_sigma
from emmy.compiler.ir.loop.tap import TAP_BUF_SUFFIX
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Stmt, Write
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped

PATTERN = [
    Pattern("producer", LoopOp),
    Pattern("consumer", LoopOp),
]


def _cell(body: Body) -> tuple[list[Stmt], Loop, list[Stmt], Loop] | None:
    """Descend ``body``'s free-loop chain to the norm cell and split it as
    ``(prefix, stat_loop, mid, sweep_loop)`` — ``[pure…, one reduce Loop, pure…, one trailing
    non-reduce Loop]``. Any other shape → ``None``."""
    cur = list(body)
    while True:
        loops = [i for i, s in enumerate(cur) if isinstance(s, Loop)]
        if not loops:
            return None
        first = cur[loops[0]]
        if len(loops) == 1 and not first.is_reduce:
            # A free-loop chain level: [pure prefix…, one trailing non-reduce Loop] — descend.
            if loops[0] != len(cur) - 1 or not all(isinstance(s, (Load, Assign)) for s in cur[: loops[0]]):
                return None
            cur = list(first.body)
            continue
        # Cell level: expect [prefix…, reduce Loop, mid…, sweep Loop] with the sweep LAST.
        if len(loops) != 2:
            return None
        i_stat, i_sweep = loops
        stat, sweep = cur[i_stat], cur[i_sweep]
        if not stat.is_reduce or sweep.is_reduce or i_sweep != len(cur) - 1:
            return None
        prefix, mid = cur[:i_stat], cur[i_stat + 1 : i_sweep]
        if not all(isinstance(s, (Load, Assign)) for s in (*prefix, *mid)):
            return None
        return prefix, stat, mid, sweep


def _bind_stat(stat: Loop, src: str) -> tuple[Load, tuple[Assign, ...], Accum] | None:
    """Bind the statistic reduce loop's body as (single scalar ``Load`` of ``src``, pure ``Assign``
    chain over it, one additive ``Accum`` folding the chain's value) — the ``bind_sinkable_stat``
    cell gates, re-anchored on the loop dialect."""
    stmts = list(stat.body)
    loads = [s for s in stmts if isinstance(s, Load)]
    accums = [s for s in stmts if isinstance(s, Accum)]
    if len(loads) != 1 or not loads[0].is_scalar or loads[0].input != src or len(accums) != 1:
        return None
    load, accum = loads[0], accums[0]
    if accum.op.reduce_canon != "add" or accum.base is not None:
        return None
    chain: list[Assign] = []
    allowed = {load.name}
    for s in stmts:
        if s is load or s is accum:
            continue
        if not isinstance(s, Assign) or any(a not in allowed for a in s.args):
            return None
        allowed.add(s.name)
        chain.append(s)
    if accum.value not in allowed:
        return None
    return load, tuple(chain), accum


def _classify_positions(
    load: Load, stat_axis, consumer: LoopOp, t: Tensor
) -> tuple[tuple[tuple[int, int | None, int | None], ...], list[Expr]] | None:
    """Classify each position of the statistic's read index of ``T`` as row / reduce / mixed and
    return ``(row_slots, b_frame_dst)`` — the position→destination map plus the destination index
    in the CONSUMER's frame. The read must cover ``T`` exactly: the reduce axis appears in exactly
    one position, plainly (the whole extent) or mixed-radix (``W·h + n`` with ``n``'s extent
    ``W`` and ``h``'s extent ``·W`` spanning the dim) — so every cell lands in exactly one row and
    every row folds exactly the reduce extent."""
    n_name = stat_axis.name
    if not stat_axis.extent.is_static:
        return None
    n_ext = stat_axis.extent.as_static()
    axes = {ax.name: ax for ax in consumer.axes}
    row_slots: list[tuple[int, int | None, int | None]] = []
    b_frame_dst: list[Expr] = []
    slot = 0
    seen_reduce = False
    for p, r_expr in enumerate(load.index):
        dim = t.shape[p]
        if not dim.is_static:
            return None
        ext = dim.as_static()
        fv = set(r_expr.free_vars())
        if n_name not in fv:
            row_slots.append((p, slot, None))
            b_frame_dst.append(r_expr)
            slot += 1
            continue
        if seen_reduce:
            return None  # the reduce axis split across positions — no single row width
        seen_reduce = True
        if isinstance(r_expr, Var):
            if ext != n_ext:
                return None  # a slice of the dim — the tap would overcount the uncovered cells
            row_slots.append((p, None, None))
            continue
        # Mixed-radix: ``W·h + n`` — the per-head norm over a flattened axis.
        form = affine_form(r_expr, fv)
        if form is None:
            return None
        anchor, coeffs = form
        anchor_s = anchor.simplify(SimplifyCtx.empty())
        if not (isinstance(anchor_s, Literal) and anchor_s.value == 0):
            return None
        if coeffs.get(n_name) != 1 or len(coeffs) != 2:
            return None
        (h_name, w) = next((k, v) for k, v in coeffs.items() if k != n_name)
        h_ax = axes.get(h_name)
        if w != n_ext or h_ax is None or not h_ax.extent.is_static or h_ax.extent.as_static() * w != ext:
            return None
        row_slots.append((p, slot, w))
        b_frame_dst.append(Var(h_name))
        slot += 1
    if not seen_reduce:
        return None
    return tuple(row_slots), b_frame_dst


def _insert_after(stmts: Body, target: Stmt, new: tuple[Stmt, ...]) -> Body | None:
    """``stmts`` with ``new`` spliced right after ``target`` (identity match, deep through
    ``Loop`` bodies); ``None`` when ``target`` isn't found."""
    out: list[Stmt] = []
    hit = False
    for s in stmts:
        if s is target:
            out.extend((s, *new))
            hit = True
            continue
        if isinstance(s, Loop) and not hit:
            inner = _insert_after(s.body, target, new)
            if inner is not None:
                s = Loop(axis=s.axis, body=inner, unroll=s.unroll, role=s.role, carrier=s.carrier)
                hit = True
        out.append(s)
    return Body(tuple(out)) if hit else None


def _replace_stat(stmts: Body, target: Loop, replacement: Stmt) -> Body | None:
    """``stmts`` with the statistic ``Loop`` swapped for ``replacement`` (deep)."""
    out: list[Stmt] = []
    hit = False
    for s in stmts:
        if s is target:
            out.append(replacement)
            hit = True
            continue
        if isinstance(s, Loop) and not hit:
            inner = _replace_stat(s.body, target, replacement)
            if inner is not None:
                s = Loop(axis=s.axis, body=inner, unroll=s.unroll, role=s.role, carrier=s.carrier)
                hit = True
        out.append(s)
    return Body(tuple(out)) if hit else None


def rewrite(match: Match, producer: Node, consumer: Node) -> Graph | None:
    graph = match.graph
    if not isinstance(producer.op, LoopOp) or not isinstance(consumer.op, LoopOp):
        raise RuleSkipped("producer or consumer is no longer a LoopOp")
    if producer.id not in consumer.inputs:
        raise RuleSkipped(f"producer {producer.id!r} is not an input of consumer {consumer.id!r}")
    if len(producer.outputs) != 1:
        raise RuleSkipped("multi-output producer — already tapped, or not this rule's shape")
    sq = f"{producer.id}{TAP_BUF_SUFFIX}"
    if graph.buffer(sq) is not None:
        raise RuleSkipped(f"stat already tapped — {sq} exists")
    if any(isinstance(s, Accum) and s.op.reduce_canon == "maximum" for s in producer.op.body.iter()):
        raise RuleSkipped("rowmax-bearing producer (softmax / flash offer site) — its form is owned by try_flash")

    # --- bind B as the fused norm form over A's output -----------------------------------
    cell = _cell(consumer.op.body)
    if cell is None:
        raise RuleSkipped("consumer is not the assembled norm form ([…, stat reduce, π…, sweep])")
    _prefix, stat, _mid, sweep = cell
    bound = _bind_stat(stat, producer.id)
    if bound is None:
        raise RuleSkipped("statistic reduce is not a single-Load pure-chain additive fold of the producer's output")
    load, chain, accum = bound
    if not any(isinstance(s, Load) and s.input == producer.id for s in sweep.body.iter()):
        raise RuleSkipped("the trailing sweep does not re-read the producer's output — not the norm pair")
    t = producer.output
    if len(load.index) != len(t.shape):
        raise RuleSkipped("statistic read rank does not match the produced tensor")
    classified = _classify_positions(load, stat.axis, consumer.op, t)
    if classified is None:
        raise RuleSkipped("statistic read does not cover the produced tensor exactly (symbolic / sliced / non-affine)")
    row_slots, b_frame_dst = classified

    # --- the σ anchor: A's single store of T --------------------------------------------
    a_writes = [s for s in producer.op.body.iter() if isinstance(s, Write) and s.output == producer.id]
    if len(a_writes) != 1 or not a_writes[0].is_scalar or a_writes[0].atomic:
        raise RuleSkipped("producer does not store its output through one plain scalar Write")
    aw = a_writes[0]
    sigma = solve_sigma(aw.index, load.index, {ax.name for ax in producer.op.axes})
    if sigma is None:
        raise RuleSkipped("σ-solve failed pairing the producer's write index against the statistic's read index")

    # The tap's A-frame destination index: for each destination slot, the producer's own write
    # expr at that position (integer-divided by W on a mixed-radix position). ``solve_sigma``
    # guaranteed each write index entry is a Var or Literal.
    a_dst: list[Expr] = []
    dst_shape: list[Dim] = []
    for p, slot, w in row_slots:
        if slot is None:
            continue
        e: Expr = aw.index[p]
        ext = t.shape[p].as_static()
        if w is not None:
            e = BinaryExpr("/", e, Literal(w, "int"))
            ext //= w
        a_dst.append(e)
        dst_shape.append(Dim(ext))

    # --- A′: term chain (on the pre-store value) + atomic tap after the store ------------
    used = {nm for s in producer.op.body.iter() for nm in s.defines()}
    sub = {load.name: aw.value}
    tap_stmts: list[Stmt] = []
    for a in chain:
        renamed = f"{a.name}__tap"
        while renamed in used:
            renamed += "_"
        used.add(renamed)
        tap_stmts.append(Assign(name=renamed, op=a.op, args=tuple(sub.get(x, x) for x in a.args), dtype=a.dtype))
        sub[a.name] = renamed
    tap_value = sub.get(accum.value, accum.value)
    tap_stmts.append(Write(output=sq, index=tuple(a_dst), value=tap_value, atomic=True))
    new_body = _insert_after(producer.op.body, aw, tuple(tap_stmts))
    if new_body is None:
        raise RuleSkipped("producer store site vanished mid-rewrite")
    tapped = LoopOp(body=new_body)
    tapped.knobs = dict(producer.op.knobs)

    # --- S: the sweep node — the statistic Reduction drops to a Load of T__sq ------------
    acc_load = Load(name=accum.name, input=sq, index=tuple(b_frame_dst))
    s_body = _replace_stat(consumer.op.body, stat, acc_load)
    if s_body is None:
        raise RuleSkipped("consumer statistic loop vanished mid-rewrite")
    sweep_op = LoopOp(body=s_body)
    sweep_op.knobs = dict(consumer.op.knobs)

    # --- fragment: A′ (T + T__sq) + S, edges preserved -----------------------------------
    out = consumer.output
    frag = Graph()
    ext_inputs = dict.fromkeys((*producer.inputs, *(i for i in consumer.inputs if i != producer.id)))
    for inp in ext_inputs:
        frag.add_node(op=InputOp(), inputs=[], output=graph.buffer(inp), node_id=inp)
    frag.add_node(
        op=tapped,
        inputs=list(producer.inputs),
        outputs=[t, Tensor(sq, tuple(dst_shape), F32)],
        node_id=producer.id,
    )
    # S's inputs in first-use (body-Load) order — the interpreter zips ``node.inputs`` against
    # ``input_bufs`` positionally, and dropping the stat loop moved the first read of ``T``.
    s_inputs = list(dict.fromkeys(ld.input for ld in sweep_op.body.loads if ld.input in {*consumer.inputs, sq}))
    s_inputs += [i for i in (*consumer.inputs, sq) if i not in s_inputs]
    frag.add_node(op=sweep_op, inputs=s_inputs, output=out, node_id=consumer.id)
    frag.outputs = [consumer.id]
    assert prod(d.as_static() for d in t.shape) == prod(d.as_static() for d in dst_shape) * stat.axis.extent.as_static()
    match.consumed = {producer.id, consumer.id}
    match.output = {producer.id: producer.id, consumer.id: consumer.id}
    return frag
