"""Tile-side tap machinery: the recognition PEEL and the shared ``PLACE@stat=fuse`` cut-out.

The loop dialect stores a fused row statistic as ordinary stmts in the producer's body
(``ir/loop/tap.py``); this module is the tile boundary's counterpart:

- :func:`peel_taps` strips the tap stmts off a tapped ``LoopOp`` body **before** classification
  and lifts the peeled facts into :class:`~emmy.compiler.ir.tile.ir.Tap` decorations, so the host
  recognizes EXACTLY as its untapped self — same structural nodes, same fork keys, same golden
  identity. A pure pre-pass: strip → classify → stamp back (the taps ride ``TileOp.taps``).
- :func:`cut_out_taps` realizes ``PLACE@stat=fuse`` (option-0) — and the sink degrade — by
  cutting the tap back out: the producer re-emits with its picked schedule intact, minus the
  decoration and the aux output; each deferred sweep node re-welds its statistic (tap term chain +
  a reconstructed reduce ``Loop`` in place of the ``T__sq`` ``Load``) and re-enters ``010`` as an
  un-mapped ``LoopOp``, landing on today's local-stat (coop) norm schedule. The sweep is
  guaranteed to still be a ``LoopOp`` — ``010_recognize`` defers any ``__sq`` reader while its
  producer carries taps — so the re-weld reproduces the never-fissioned form structurally
  (round-trip parity is a unit-tested contract).

Both realizers (``015_cut_stat_tap`` — the decision's realizer — and ``034_attach_taps``'s
ineligible-host degrade) share :func:`cut_out_taps`, so "sink but unattachable" lands on exactly
the fuse deployment.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal, Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.loop.tap import TAP_BUF_SUFFIX, is_tap_write, strip_taps, tap_chains
from emmy.compiler.ir.stmt import Assign, Body, Load, Loop, Stmt, Write
from emmy.compiler.ir.tile import Tap, TileOp
from emmy.compiler.pipeline.passes.loop.stamp._stamp import name_for_loop, restamp_structural_features
from emmy.compiler.pipeline.pipeline import LoweringError


def _deep_stmts(stmts) -> list[Stmt]:
    out: list[Stmt] = []
    for s in stmts:
        out.append(s)
        for b in s.nested():
            out.extend(_deep_stmts(b))
    return out


def peel_taps(body: Body, out) -> tuple[tuple[Stmt, ...], tuple[Tap, ...]]:
    """Strip the tap stmts off ``body`` and lift each into a :class:`Tap`. ``out`` is the host
    node's primary output :class:`Tensor` (the tapped buffer — the tap fusion rule only ever taps
    the producer's own output). Raises :class:`LoweringError` on a malformed stored form — the
    naming contract is minted by ``015_tap_row_stat`` alone, so a shape this can't re-derive is a
    broken invariant, never a decline."""
    deep = _deep_stmts(body)
    tap_writes = [s for s in deep if is_tap_write(s)]
    chains = tap_chains(body)
    taps: list[Tap] = []
    for tw in tap_writes:
        src = tw.output[: -len(TAP_BUF_SUFFIX)]
        if src != out.name:
            raise LoweringError(f"tap {tw.output!r} does not tap the host's own output {out.name!r}")
        hosts = [s for s in deep if isinstance(s, Write) and s.output == src and not is_tap_write(s)]
        if len(hosts) != 1 or not hosts[0].is_scalar:
            raise LoweringError(f"tapped host must store {src!r} through one plain scalar Write (found {len(hosts)})")
        aw = hosts[0]
        # The term chain: the exclusive backward Assign-cone of the accumulated value (the shared
        # structural contract — ``tap_chains``); its one external read is the host store's value.
        chain = list(chains.get(id(tw), ()))
        names = {a.name for a in chain}
        external = {a for c in chain for a in c.args if a not in names} | ({tw.value} if tw.value not in names else set())
        if len(external) != 1:
            raise LoweringError(f"tap chain for {tw.output!r} must read exactly the host store's value (reads {sorted(external)})")
        anchor = next(iter(external))
        if anchor != aw.value:
            raise LoweringError(f"tap chain anchor {anchor!r} is not the host store's value {aw.value!r}")
        # Position map: pair each destination coord against the host store's index — plain
        # (``index[p]``) or mixed-radix (``index[p] // W``); unpaired positions are reduce-only.
        paired: dict[int, tuple[int, int | None]] = {}  # src pos -> (dst slot, W)
        for j, d in enumerate(tw.index):
            hit = None
            for p, e in enumerate(aw.index):
                if p in paired:
                    continue
                if d == e:
                    hit = (p, None)
                    break
                if isinstance(d, BinaryExpr) and d.op == "/" and d.left == e and isinstance(d.right, Literal):
                    hit = (p, int(d.right.value))
                    break
            if hit is None:
                raise LoweringError(f"tap {tw.output!r}: destination coord {d.pretty()} pairs with no host store position")
            paired[hit[0]] = (j, hit[1])
        row_slots = tuple((p, *(paired.get(p) or (None, None))) for p in range(len(aw.index)))
        width = 1
        for p, slot, w in row_slots:
            if slot is None:
                width *= out.shape[p].as_static()
            elif w is not None:
                width *= w
        taps.append(
            Tap(
                dst=tw.output,
                src=src,
                anchor=anchor,
                chain=tuple(chain),
                value=tw.value,
                src_index=tuple(aw.index),
                row_slots=row_slots,
                width=width,
            )
        )
    return strip_taps(body), tuple(taps)


# --------------------------------------------------------------------------- #
# The fuse cut-out — shared by ``015_cut_stat_tap`` and ``034_attach_taps``'s degrade.
# --------------------------------------------------------------------------- #


def _replace_stmt(stmts, target: Stmt, replacement: Stmt):
    out: list[Stmt] = []
    hit = False
    for s in stmts:
        if s is target:
            out.append(replacement)
            hit = True
            continue
        if isinstance(s, Loop) and not hit:
            inner = _replace_stmt(s.body, target, replacement)
            if inner is not None:
                s = Loop(axis=s.axis, body=inner, unroll=s.unroll, role=s.role, carrier=s.carrier)
                hit = True
        out.append(s)
    return Body(tuple(out)) if hit else None


def _reweld(body: Body, tap: Tap, seq: int) -> Body:
    """``body`` (the sweep node's) with its ``Load`` of ``tap.dst`` swapped for the reconstructed
    local statistic: ``for n: (Load T, term chain, additive Accum)`` — the loop the tap fission
    dropped, rebuilt from the tap's facts + the sweep's own destination index exprs. The ``Accum``
    binds the Load's name, so the projection reads it unchanged."""
    loads = [s for s in _deep_stmts(body) if isinstance(s, Load) and s.input == tap.dst]
    if len(loads) != 1 or not loads[0].is_scalar:
        raise LoweringError(f"sweep must read {tap.dst!r} through one scalar Load (found {len(loads)})")
    sq_load = loads[0]
    deep = _deep_stmts(body)
    used = {nm for s in deep for nm in s.defines()} | {s.axis.name for s in deep if isinstance(s, Loop)}

    def fresh(base: str) -> str:
        nm = base
        while nm in used:
            nm += "_"
        used.add(nm)
        return nm

    n_name = fresh(f"k__stat{seq}")
    # The T read index: destination coords come from the sweep's OWN dst-load exprs; reduce-only
    # positions take the fresh reduce var; a mixed-radix position recombines ``W·coord + n``.
    index: list[Expr] = []
    for p, slot, w in tap.row_slots:
        if slot is None:
            index.append(Var(n_name))
        elif w is None:
            index.append(sq_load.index[slot])
        else:
            index.append(BinaryExpr("+", BinaryExpr("*", sq_load.index[slot], Literal(w, "int")), Var(n_name)))
    x_name = fresh(f"x__stat{seq}")
    x_load = Load(name=x_name, input=tap.src, index=tuple(index))
    sub = {tap.anchor: x_name}
    chain: list[Stmt] = []
    for a in tap.chain:
        nm = fresh(f"{a.name}__st")
        chain.append(Assign(name=nm, op=a.op, args=tuple(sub.get(x, x) for x in a.args), dtype=a.dtype))
        sub[a.name] = nm
    from emmy.compiler.ir.stmt import Accum  # noqa: PLC0415 — leaf import keeps module deps flat

    acc = Accum(name=sq_load.name, value=sub.get(tap.value, tap.value), op=ElementwiseImpl(tap.op), axes=(n_name,))
    stat_loop = Loop(axis=Axis(name=n_name, extent=Dim(tap.width)), body=Body((x_load, *chain, acc)))
    out = _replace_stmt(body, sq_load, stat_loop)
    if out is None:
        raise LoweringError(f"sweep's Load of {tap.dst!r} vanished mid-rewrite")
    return out


def cut_out_taps(match, root: Node) -> Graph:
    """Cut every tap back out of the settled host ``root`` (a taps-bearing ``TileOp``): re-emit
    the producer untapped (schedule intact, aux outputs dropped) and each sweep consumer with its
    statistic re-welded, un-mapped, for ``010`` to re-recognize. See module docstring."""
    graph = match.graph
    tile: TileOp = root.op
    by_consumer: dict[str, list[Tap]] = {}
    for tap in tile.taps:
        readers = [n for n in graph.nodes.values() if tap.dst in n.inputs]
        if len(readers) != 1:
            raise LoweringError(f"tap aux {tap.dst!r} must have exactly one reader (found {len(readers)})")
        (s_node,) = readers
        if not isinstance(s_node.op, LoopOp):
            raise LoweringError(f"tap sweep {s_node.id!r} was recognized before the stat decision settled — the deferral contract broke")
        by_consumer.setdefault(s_node.id, []).append(tap)

    dst_bufs = {tap.dst for tap in tile.taps}
    untapped = replace(tile, taps=())

    frag = Graph()
    ext: dict[str, None] = dict.fromkeys(root.inputs)
    for sid in by_consumer:
        for i in graph.nodes[sid].inputs:
            if i != root.id and i not in dst_bufs:
                ext.setdefault(i)
    for inp in ext:
        frag.add_node(op=InputOp(), inputs=[], output=graph.buffer(inp), node_id=inp)
    frag.add_node(op=untapped, inputs=list(root.inputs), outputs=[root.output], node_id=root.id)
    for seq, (sid, taps) in enumerate(by_consumer.items()):
        s_node = graph.nodes[sid]
        body = s_node.op.body
        for tap in taps:
            body = _reweld(body, tap, seq)
        welded = LoopOp(body=body)
        welded.knobs = dict(s_node.op.knobs)
        # Re-stamp the kernel label from the sweep node's provenance (its hints carry the whole
        # original norm's op pieces — the fission's splice merged them forward): the re-welded
        # body is structurally identical to the never-fissioned form, so the provenance label +
        # structural-body hash reproduce the exact kernel name ``010_stamp_loop_names`` would
        # have minted — no dump/kname churn.
        welded.name = name_for_loop(welded, s_node, graph)
        s_inputs = [i for i in s_node.inputs if i not in dst_bufs]
        frag.add_node(op=welded, inputs=s_inputs, output=s_node.output, node_id=sid)
        restamp_structural_features(welded, frag)
    frag.outputs = list(by_consumer)

    match.consumed = {root.id, *by_consumer}
    for sid in by_consumer:
        match._identities[sid] = id(graph.nodes[sid])  # noqa: SLF001 — the match owns the snapshot; no public setter
    match.output = {root.id: root.id, **{sid: sid for sid in by_consumer}}
    return frag


__all__ = ["cut_out_taps", "peel_taps"]
