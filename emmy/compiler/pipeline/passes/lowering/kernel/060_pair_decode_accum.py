"""Pair the decode band's fold into f16 (``F16_REDUCE_F32_ACC`` — pin-only, off by default).

After ``055_fuse_trellis_runs`` a decode-band body is one tile column per step: a run
:class:`TrellisLoad` binding 16 decoded weights, then 16 copies of ``load x; multiply; accumulate``,
one per k row. Every one of those 16 triples spends two fp16→fp32 widenings (the decode's own
return and the activation's), one ``FFMA``, and one 16-bit ``LDG`` — five instructions of the ~11.5
the band spends per 2-bit weight, against a DRAM floor it is nowhere near (NCU on the past-L2
square: SM throughput 69 %, DRAM 35 %).

This pass rewrites that set into the f16-pair form exllamav3's gemv uses:

- the run decodes PACKED (``TrellisLoad.packed``): ``names[p]`` is the ``__half2`` of k rows ``2p``
  and ``2p+1``, straight out of the codebook's own fp16 add, never widened;
- the activations ride as width-2 vector ``Load``\\ s, one 32-bit read per pair instead of two
  16-bit ones, repacked into a ``__half2``;
- the products are ``__hmul2``, summed over the tile column by an fp16 TREE, and the one surviving
  pair is promoted into the f32 accumulators once per tile step.

**The promote cadence is one tile step** — the band's own quantum, and deliberately not a knob. The
fp16 tree is 3 deep, so the error is the fp16 PRODUCT's (~4e-4 rel, the floor of any scheme that
multiplies in fp16) rather than an accumulation's; longer cadences measure within 1 % on the same
kernel and lose accuracy monotonically (a chain over the whole K slice reaches 1.0-1.7e-3 and grows
with the slice). That is the same "chunk IS the cadence" answer the mma tier's f16-accumulate atom
gives — see ``_atom._F16ACC_STEPS`` — with the chunk here being one step.

Precision-trading, so never silently on: ``EMMY_F16_REDUCE_F32_ACC=1`` or the ``EMMY_FAST_MATH``
umbrella (:func:`precision_pin`). The knob is stamped either way, so the realized config records the
policy and idempotence rides the stamp.

Only the 16 k rows of ONE tile column pair, and only when the whole set is present and shaped as the
band emits it — the pass reads the structure it needs off the body and declines otherwise, so a
per-element decode, a partial column, or an accumulator the tail also reads keeps the f32 form.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.dtype import F16, F32, F16x2
from emmy.compiler.graph import Node
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Literal, SimplifyCtx, affine_form
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Pack, StateMerge, Stmt, TrellisLoad, Unpack
from emmy.compiler.ir.stmt.leaves import TRELLIS_TILE
from emmy.compiler.pipeline import Pattern, RuleSkipped
from emmy.compiler.pipeline.search.space import F16_REDUCE_F32_ACC, precision_pin

PATTERN = [Pattern("root", KernelOp)]

_ADD = ElementwiseImpl("add")
_MUL = ElementwiseImpl("multiply")


def _offset(lo, hi) -> int | None:
    """``hi - lo`` when the two index Exprs are affine in the same free vars with the same
    coefficients and differ only by a literal constant, else ``None``."""
    forms = []
    for e in (lo, hi):
        form = affine_form(e, set(e.free_vars()))
        if form is None:
            return None
        const = form[0].simplify(SimplifyCtx.empty())
        if not (isinstance(const, Literal) and isinstance(const.value, int)):
            return None
        forms.append((int(const.value), form[1]))
    return forms[1][0] - forms[0][0] if forms[0][1] == forms[1][1] else None


def _column_triples(stmts: list[Stmt], run: TrellisLoad) -> list[tuple[int, Load, Assign, Accum]] | None:
    """The ``(load x, multiply, accumulate)`` triple for each of ``run``'s 16 decoded names, in k
    order, or ``None`` when the body is not the shape the decode band emits. Each triple must be
    exactly: a SCALAR ``Load`` of one activation, an ``Assign`` multiplying that load by the run's
    element, and an additive ``Accum`` folding the product — with the multiply and the accumulate
    read nowhere else, so replacing the three is a local rewrite."""
    by_name = {nm: i for i, nm in enumerate(run.names)}
    found: dict[int, tuple[int, Load, Assign, Accum]] = {}
    reads: dict[str, int] = {}
    for s in stmts:
        for d in s.deps():
            reads[d] = reads.get(d, 0) + 1
    for pos, s in enumerate(stmts):
        if not isinstance(s, Assign) or s.op.name != _MUL.name or len(s.args) != 2:
            continue
        weight = next((a for a in s.args if a in by_name), None)
        if weight is None:
            continue
        act = s.args[0] if s.args[1] == weight else s.args[1]
        ld = next((t for t in stmts if isinstance(t, Load) and t.is_scalar and t.names[0] == act), None)
        acc = next((t for t in stmts if isinstance(t, Accum) and t.value == s.name), None)
        if ld is None or acc is None or type(ld) is not Load or acc.op.name != _ADD.name or acc.base is not None:
            return None
        if reads.get(s.name, 0) != 1 or reads.get(act, 0) != 1 or acc.name in reads:
            return None  # the product / activation / accumulator is read elsewhere in this scope
        k = by_name[weight]
        if k in found:
            return None
        found[k] = (pos, ld, s, acc)
    if len(found) != TRELLIS_TILE:
        return None
    return [found[k] for k in range(TRELLIS_TILE)]


def _pair_loads(triples: list[tuple[int, Load, Assign, Accum]]) -> list[Load] | None:
    """One width-2 vector ``Load`` per k-row pair, or ``None`` when a pair's two activation reads
    are not the two consecutive elements of one buffer row (which is what makes them one 32-bit
    read). The band's activation index is the reduce coordinate, so ``2p`` and ``2p+1`` differ by
    exactly one in the last position and agree everywhere else."""
    out: list[Load] = []
    for p in range(TRELLIS_TILE // 2):
        lo, hi = triples[2 * p][1], triples[2 * p + 1][1]
        if lo.input != hi.input or lo.dtype != hi.dtype or lo.dtype is None or lo.dtype != F16:
            return None
        if len(lo.index) != len(hi.index) or lo.index[:-1] != hi.index[:-1]:
            return None
        if _offset(lo.index[-1], hi.index[-1]) != 1:
            return None
        out.append(Load(names=(lo.names[0], hi.names[0]), input=lo.input, index=lo.index, dtype=lo.dtype))
    return out


def _rewrite_column(stmts: list[Stmt], at: int, run: TrellisLoad) -> list[Stmt] | None:
    """Replace one tile column's 16 scalar triples with the paired form, or ``None`` to decline."""
    triples = _column_triples(stmts, run)
    if triples is None:
        return None
    loads = _pair_loads(triples)
    if loads is None:
        return None
    npair = TRELLIS_TILE // 2
    tag = run.names[0]
    pair_names = tuple(f"{tag}__p{p}" for p in range(npair))
    body: list[Stmt] = [replace(run, names=pair_names, dtype=F16x2, packed=True)]
    prod: list[str] = []
    for p in range(npair):
        lo, hi = triples[2 * p][1].names[0], triples[2 * p + 1][1].names[0]
        body.append(loads[p])
        body.append(Pack(name=f"{tag}__x{p}", low=lo, high=hi, dtype=F16x2))
        body.append(Assign(name=f"{tag}__m{p}", op=_MUL, args=(pair_names[p], f"{tag}__x{p}"), dtype=F16x2))
        prod.append(f"{tag}__m{p}")
    # The fp16 TREE over the tile column: depth 3, so the pair that survives carries the whole
    # column's product sum at the fp16 product's own accuracy.
    level = 0
    while len(prod) > 1:
        level += 1
        nxt = []
        for i in range(len(prod) // 2):
            nm = f"{tag}__t{level}_{i}"
            body.append(Assign(name=nm, op=_ADD, args=(prod[i], prod[i + len(prod) // 2]), dtype=F16x2))
            nxt.append(nm)
        prod = nxt
    # The promote: the surviving pair's lanes are k rows 0 and 1 of the column, so they fold into
    # exactly those two accumulators and the other 14 chains are gone.
    body.append(Unpack(low_name=f"{tag}__lo", high_name=f"{tag}__hi", value=prod[0], lane_dtype=F16))
    keep = (triples[0][3].name, triples[1][3].name)
    body.append(replace(triples[0][3], value=f"{tag}__lo", dtype=F32))
    body.append(replace(triples[1][3], value=f"{tag}__hi", dtype=F32))
    dropped = {t[3].name for t in triples} - set(keep)
    consumed = {id(run)} | {id(t[1]) for t in triples} | {id(t[2]) for t in triples} | {id(t[3]) for t in triples}
    return [
        *(s for i, s in enumerate(stmts) if i < at and id(s) not in consumed),
        *body,
        *(s for i, s in enumerate(stmts) if i > at and id(s) not in consumed),
    ], dropped


def _prune_merges(stmts: list[Stmt], dropped: set[str]) -> list[Stmt]:
    """Drop the REG-tree :class:`StateMerge` of every accumulator the rewrite removed. The tree
    merges the band's ``reg`` register copies into copy 0; the paired fold already summed them over
    the tile column, so those partials no longer exist."""
    return [s for s in stmts if not (isinstance(s, StateMerge) and set(s.state_b) & dropped)]


def _walk(body: Body) -> tuple[Body, set[str]]:
    """Rewrite every tile column in ``body`` and its nested scopes, returning the new body and the
    accumulator names the rewrite removed. Those names propagate OUTWARD: the fold lives in the
    band's reduce loop while the REG-tree merges that read its per-copy partials sit in the
    enclosing tile body, so the scope that holds the merges is the one that prunes them."""
    stmts: list[Stmt] = []
    dropped: set[str] = set()
    for s in body:
        nested = s.nested()
        if not nested:
            stmts.append(s)
            continue
        rebuilt = []
        for b in nested:
            nb, drop = _walk(b)
            rebuilt.append(nb)
            dropped |= drop
        stmts.append(s.with_bodies(tuple(rebuilt)))
    for run in [s for s in stmts if isinstance(s, TrellisLoad) and not s.is_scalar and not s.packed]:
        at = next((i for i, s in enumerate(stmts) if s is run), None)
        if at is None or len(run.names) != TRELLIS_TILE:
            continue
        done = _rewrite_column(stmts, at, run)
        if done is not None:
            stmts, drop = done
            dropped |= drop
    return Body(tuple(_prune_merges(stmts, dropped) if dropped else stmts)), dropped


def rewrite(root: Node) -> KernelOp | None:
    op: KernelOp = root.op
    if F16_REDUCE_F32_ACC.name in op.knobs:
        raise RuleSkipped("F16_REDUCE_F32_ACC already decided (idempotence via knob)")
    if not any(isinstance(s, TrellisLoad) for s in op.body.iter()):
        raise RuleSkipped("no trellis decode in this kernel")
    if not precision_pin(F16_REDUCE_F32_ACC):
        return KernelOp(body=op.body, name=op.name, knobs={**op.knobs, F16_REDUCE_F32_ACC.name: False})
    new_body, dropped = _walk(op.body)
    paired = bool(dropped)
    return KernelOp(body=new_body if paired else op.body, name=op.name, knobs={**op.knobs, F16_REDUCE_F32_ACC.name: paired})
