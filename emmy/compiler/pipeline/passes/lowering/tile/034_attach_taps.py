"""Materialize retained taps (``PLACE@stat=sink``) into the settled host's body — tier-derived.

The last tap-bearing pass: after ``015_cut_stat_tap`` consumed every fuse-stamped tap and
``030_split_reduce`` relocated split hosts' taps onto their finalize, whatever still carries
``TileOp.taps`` is a sink-stamped host whose schedule has settled. This rule turns the decoration
into body stmts — the per-cell term chain re-evaluated on each just-stored value plus a row fold —
deriving the FOLD TIER from what the schedule did to the tapped axis (the derive-never-store rule
cooperative reduces follow; nothing about the fold is stored in the IR):

1. **in-thread** — a store group covering the WHOLE row (the register tile spans the reduce
   width): serial register sum + one plain ``Write`` per row. No atomics and no zero-init — the
   memset floor that ate the qknorm / m64 sites simply never exists on this tier.
2. **CTA-local smem fold** — not yet derived (the structural seam is the group classification
   below); such groups fall to tier 3, which is always correct.
3. **row spans CTAs** (the realistic matmul grid) — the hierarchical :class:`RowAccum` fold
   (warp shfl → smem stage → ~1 atomic per block), zero-init'd per launch through the ordinary
   ``zero_outputs`` machinery.

Eligibility is the old sink realizer's, gated in the negative: the host must store the tapped
buffer through plain top-level scalar ``Write``\\ s in a flat ``Map`` body — a split-K FINALIZE or
a pointwise sweep. An atomic partial (``g<w>a`` — no complete value anywhere) or an mma
``RegStore`` epilogue host DEGRADES to the always-legal cut-out instead (``_tap.cut_out_taps``),
deploying exactly the fuse kernels — which is what makes the sink row safe to mirror onto every
schedule row. Each store's destination row index derives positionally from the store's OWN index
exprs through ``Tap.row_slots`` (a σ-solve against the recorded fission-time store index), so
register-tiled groups need no flat-address algebra: same-row stores group by their derived
destination exprs and contribute one summed term per group."""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal
from emmy.compiler.ir.loop.splicer import solve_sigma
from emmy.compiler.ir.stmt import Assign, Body, RowAccum, Stmt, Write
from emmy.compiler.ir.tile import Map, Tap, TileOp
from emmy.compiler.ir.tile.ops import reduce_plan
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.lowering.tile._tap import cut_out_taps

PATTERN = [Pattern("root", TileOp)]


def _host_writes(op, buf: str) -> list[Write] | None:
    """The host's eligible stores of ``buf`` — plain scalar TOP-LEVEL ``Write``\\ s in a flat
    ``Map`` body covering every store of the buffer (an atomic write, a nested/loop store, or an
    mma epilogue all return ``None`` — the degrade signal)."""
    if not isinstance(op, Map) or op.source is not None:
        return None
    all_writes = [s for s in op.body.iter() if isinstance(s, Write) and s.output == buf]
    top_writes = [s for s in op.body if isinstance(s, Write) and s.output == buf]
    if not top_writes or len(all_writes) != len(top_writes):
        return None
    if any(w.atomic or not w.is_scalar for w in top_writes):
        return None
    return top_writes


def _dst_index(tap: Tap, w: Write) -> tuple[Expr, ...] | None:
    """The destination row index for one settled store ``w`` — the tap's recorded fission-time
    store index σ-solved against ``w``'s own index, applied slot-wise (``// W`` on a mixed-radix
    position). ``None`` when the store's index no longer pairs (the degrade signal)."""
    axes = {v for e in tap.src_index for v in e.free_vars()}
    sigma = solve_sigma(tap.src_index, w.index, axes)
    if sigma is None:
        return None
    out: list[Expr] = []
    for p, _slot, div in sorted((r for r in tap.row_slots if r[1] is not None), key=lambda r: r[1]):
        e = sigma.apply(tap.src_index[p])
        if div is not None:
            e = BinaryExpr("/", e, Literal(div, "int"))
        out.append(e)
    return tuple(out)


def _attach(body: list[Stmt], tap: Tap, writes: list[Write], used: set[str]) -> list[Stmt] | None:
    """``body`` with ``tap``'s term chains + row folds spliced after its stores, or ``None`` when
    any store's destination can't be derived (the degrade signal). Same-destination stores group;
    a group covering the whole row width takes the in-thread tier (plain store, no atomics)."""
    groups: dict[tuple, list[Write]] = {}
    for w in writes:
        dst = _dst_index(tap, w)
        if dst is None:
            return None
        groups.setdefault(dst, []).append(w)
    tier1 = all(len(g) == tap.width for g in groups.values())

    def fresh(base: str) -> str:
        nm = base
        while nm in used:
            nm += "_"
        used.add(nm)
        return nm

    inserts: dict[int, list[Stmt]] = {}  # body index -> stmts to splice after
    for dst, group in groups.items():
        stmts: list[Stmt] = []
        terms: list[str] = []
        for w in group:
            sub = {tap.anchor: w.value}
            for a in tap.chain:
                nm = fresh(f"{a.name}_at")
                stmts.append(Assign(name=nm, op=a.op, args=tuple(sub.get(x, x) for x in a.args), dtype=a.dtype))
                sub[a.name] = nm
            terms.append(sub.get(tap.value, tap.value))
        value = terms[0]
        for term in terms[1:]:
            nm = fresh(f"{value}_s")
            stmts.append(Assign(name=nm, op=ElementwiseImpl(tap.op), args=(value, term)))
            value = nm
        if tier1:
            stmts.append(Write(output=tap.dst, index=dst, value=value))
        else:
            stmts.append(RowAccum(dst=tap.dst, index=dst, value=value))
        # Identity search, not ``.index`` — two value-equal sibling Writes must not collide.
        at = max(next(i for i, s in enumerate(body) if s is w) for w in group)
        inserts.setdefault(at, []).extend(stmts)
    out: list[Stmt] = []
    for i, s in enumerate(body):
        out.append(s)
        out.extend(inserts.get(i, ()))
    return out


def rewrite(match: Match, root: Node) -> TileOp | Graph | None:
    tile: TileOp = root.op
    if not tile.taps:
        raise RuleSkipped("untapped kernel — nothing to attach")
    plan = reduce_plan(tile) if tile.op is not None else None
    if plan is not None and plan.needs_split:
        raise RuleSkipped("cross-CTA split pending — 030 relocates the taps onto the finalize first")
    used = {nm for s in (tile.op.body.iter() if isinstance(tile.op, Map) else ()) for nm in s.defines()}
    body: list[Stmt] | None = list(tile.op.body) if isinstance(tile.op, Map) and tile.op.source is None else None
    for tap in tile.taps:
        writes = _host_writes(tile.op, tap.src) if body is not None else None
        if writes is None or (body := _attach(body, tap, writes, used)) is None:
            # No plain complete-value store site on this schedule (atomic partial / mma epilogue /
            # coop form) — degrade to the always-legal cut-out: exactly the fuse deployment.
            return cut_out_taps(match, root)
    attached = replace(tile, op=replace(tile.op, body=Body(tuple(body))), taps=())
    # A single-node FRAGMENT, not an op rebind: only a Graph splice counts as a functional rewrite
    # (``Cursor.n_applied``), and the deferred sweep needs the restarted scan to re-enter ``010``
    # once the taps are cleared here.
    frag = Graph()
    for inp in root.inputs:
        frag.add_node(op=InputOp(), inputs=[], output=match.graph.buffer(inp), node_id=inp)
    frag.add_node(op=attached, inputs=list(root.inputs), outputs=list(root.outputs), node_id=root.id)
    frag.outputs = [root.id]
    return frag
