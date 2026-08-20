"""The TOTAL lift — any loop nest → its Fold tree, no algebra dispatch.

This is the structural half of the Loop-IR → Tile-IR boundary rebuilt as ONE algorithm: peel the
free (parallel) axes, then lift every reduce ``Loop`` in the cell — bottom-up, wherever it sits —
through the one loop→term parser (:func:`~._fromloop.fold_from_loop`, byte-identity gated). A loop
the parser declines stays a verbatim ``Loop`` stmt IN PLACE: the raw-loop escape is a subtree now,
never the whole cell. Recognition (which algebra a fold IS — contraction binding, online-softmax
pairing, the monoid-producer composition) happens AFTERWARD, as classification passes over the
lifted tree, stated on ``Fold`` fields; nothing here dispatches on the algebra.

The lift is total: its worst case is the identity transform on the cell. What it guarantees is
scope, not meaning — a fold's λ may read names of earlier siblings (an operand's carried state,
an outer scalar), and root formation restores any fold whose reads would cross the
operands-before-body lowering order back to its verbatim loop (the gate makes that restoration
byte-exact).

Assumption inherited from the old recognizer: a top-level ``Init`` seeding a lifted fold's
accumulator is the canonical dissolved spelling (the identity seed) and is dropped — the fold
re-seeds from its op identity."""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.pure.fold import Fold, deep_defines, deep_reads
from emmy.compiler.ir.stmt import Accum, Assign, Body, Init, Load, Loop, Select, Write
from emmy.compiler.ir.stmt.base import Stmt
from emmy.compiler.ir.tile import Placement, TileOp, split_effects
from emmy.compiler.pipeline.passes.lowering.tile._classify import classify
from emmy.compiler.pipeline.passes.lowering.tile._fromloop import fold_from_loop

# --------------------------------------------------------------------------- #
# Peel — the outer free-loop chain becomes the kernel's parallel axes.
# --------------------------------------------------------------------------- #


def _peel(body: Body) -> tuple[list, list[Stmt]]:
    """Split a body into ``(free_axes, per_cell_stmts)``.

    The outer chain of **free** loops becomes the parallel axes. At every level of the
    chain a leading run of pure stmts (``Load`` / ``Assign`` / ``Init`` — loop-invariant
    loads hoisted above or between the free loops, e.g. a broadcast row scale ``rs[m]``
    read once per ``m``) is sunk into the per-cell body, re-evaluated per cell. The chain
    stops at the first reduce loop, branch, or non-pure stmt — everything from there down
    is the per-cell body."""
    axes: list = []
    prefix: list[Stmt] = []
    cur = list(body)
    while True:
        i = 0
        while i < len(cur) and isinstance(cur[i], (Load, Assign, Init)):
            i += 1
        head, rest = cur[:i], cur[i:]
        if len(rest) != 1 or not isinstance(rest[0], Loop) or rest[0].is_reduce:
            return axes, prefix + cur
        prefix += head
        axes.append(rest[0].axis)
        cur = list(rest[0].body)


# --------------------------------------------------------------------------- #
# The total lift — every reduce loop through the one parser, in place.
# --------------------------------------------------------------------------- #


def _route_prologue(pending: list, loop: Loop, suffix_reads: set[str]) -> tuple[Loop, list]:
    """Sink the reduce-feeding pure prologue into ``loop``'s body — SPECULATIVE (the caller
    reverts to the originals when the lift declines). A pure stmt whose defs the loop body reads
    moves inside (the fold's λ must define what it reads per step); one ALSO read after the loop
    (``suffix_reads`` — the epilogue, a later fold) stays put unless it is a dependency-free
    scalar ``Load``, which sinks as a renamed copy (``__stat``) while the original serves the
    suffix. A stmt that stays put and still feeds the loop makes the λ read an undefined name, so
    the extraction declines and the loop keeps its raw spelling — routing failure degrades
    per-loop, never per-cell."""
    need = set(deep_reads(list(loop.body)))
    take: set[int] = set()
    dup: dict[int, dict[str, str]] = {}
    for i in range(len(pending) - 1, -1, -1):
        s = pending[i]
        if not isinstance(s, (Load, Assign, Select)):
            continue  # a fold / raw loop / Init never sinks; its results read as free names
        defs = set(s.defines())
        if not defs & need:
            continue
        if defs & suffix_reads:
            if isinstance(s, Load) and not s.deps() and not any(e.free_vars() for e in s.index or ()):
                dup[i] = {nm: f"{nm}__stat" for nm in defs}
                take.add(i)
                continue
            continue  # feeds both sides and cannot be duplicated — the λ read stays free
        take.add(i)
        need |= set(s.deps())
    if not take:
        return loop, list(pending)
    renames = {old: new for m in dup.values() for old, new in m.items()}
    rename = lambda n: renames.get(n, n)  # noqa: E731
    sunk = []
    for i in sorted(take):
        s = pending[i]
        sunk.append(_rewrite_deep(s, rename) if i in dup else s)
    body = tuple(_rewrite_deep(s, rename) for s in loop.body) if renames else tuple(loop.body)
    kept = [s for i, s in enumerate(pending) if i not in take or i in dup]
    return replace(loop, body=Body((*sunk, *body))), kept


def _rewrite_deep(stmt: Stmt, rename) -> Stmt:
    """Apply an SSA rename through a stmt, recursing into ``Loop`` bodies."""
    if isinstance(stmt, Loop):
        return replace(stmt, body=Body(tuple(_rewrite_deep(s, rename) for s in stmt.body)))
    return stmt.rewrite(rename)


def _stamp_axes(loop: Loop) -> Loop:
    """Stamp each axes-less ``Accum`` with its OWN loop's axis, deep — the canonical dissolved
    spelling the parser's gate re-derives. Pipeline-lowered IR arrives stamped; hand-lowered IR
    (and some minted pieces) omits it, and the omission is spelling, not semantics: an ``Accum``
    folds over exactly the loop it sits in."""
    body = tuple(
        replace(s, axes=(loop.axis.name,)) if isinstance(s, Accum) and not s.axes else (_stamp_axes(s) if isinstance(s, Loop) else s)
        for s in loop.body
    )
    return replace(loop, body=Body(body))


def _lift_cell(cell: list[Stmt]) -> list:
    """The bottom-up total lift of the per-cell stmts: every reduce ``Loop`` — at cell level or
    inside a free sweep — goes through :func:`fold_from_loop` and, on success, is replaced by its
    typed :class:`Fold` IN PLACE (a term is a legal body member); on decline the raw loop stands
    verbatim. Reduce loops NESTED in a reduce are the parser's own composed-step reading (operand
    edges), so only free-loop bodies recurse here. Seeding ``Init``\\ s of a lifted fold's
    accumulators are dropped (the fold re-seeds)."""
    out: list = []
    for i, s in enumerate(cell):
        if isinstance(s, Loop) and s.is_reduce:
            suffix_reads = deep_reads(cell[i + 1 :])
            routed, kept = _route_prologue(out, _stamp_axes(s), suffix_reads)
            fold = fold_from_loop(routed)
            if fold is None:
                out.append(s)
                continue
            seeds = set(fold.combine.results)
            out = [p for p in kept if not (isinstance(p, Init) and p.name in seeds)]
            out.append(fold)
            continue
        if isinstance(s, Loop):
            out.append(replace(s, body=Body(tuple(_lift_cell(list(s.body))))))
            continue
        out.append(s)
    return out


# --------------------------------------------------------------------------- #
# Root formation — folds become the projection's operands, scope permitting.
# --------------------------------------------------------------------------- #


def _form_root(cell: list):
    """Shape the lifted cell into the stored root: top-level folds hoist to the projection's
    OPERANDS (in body order — an operand's λ may read an earlier operand's carried state), the
    rest is the projection body. Operands lower BEFORE the body, so a fold reading a name the
    body defines cannot hoist — it is restored to its verbatim loop (byte-exact by the parser's
    gate) and stays a body member, and the check re-runs until it settles (a restored loop's defs
    may strand another fold). A single bare fold with nothing else is the root node itself."""
    kept = list(cell)
    while True:
        body_defs = {n for s in kept if not isinstance(s, Fold) for n in deep_defines(s)}
        demoted = False
        for i, s in enumerate(kept):
            if isinstance(s, Fold) and deep_reads(s.lower()) & body_defs:
                kept[i] = s.loop
                demoted = True
        if not demoted:
            break
    folds = tuple(s for s in kept if isinstance(s, Fold))
    body = tuple(s for s in kept if not isinstance(s, Fold))
    if len(folds) == 1 and not body:
        return folds[0]
    return Fold.projection(body=Body(body), operands=folds)


def _order_free_by_output(node, free: list, stores: tuple = ()) -> tuple:
    """Order the free (grid) axes to match the **output Write's index order**, so the innermost
    grid axis is the output's *contiguous* dim. The root ``Write`` is read from the body (a
    raw-loop spelling) or the boundary ``stores`` (a converted projection — its top-level store
    only). A free axis the store does NOT index is a PARTITION: it cannot be output-ordered, and
    it sorts OUTERMOST (in peel order among partitions) so the trailing pair stays the output's
    ``(m, n)``. A node with no explicit output ``Write`` is left as-is."""
    body = node.lower() if isinstance(node, Fold) else getattr(node, "body", ())
    write = next((s for s in body if isinstance(s, Write)), None)
    if write is None:
        write = next((st.write for st in stores if st.sweep is None), None)
    if write is None:
        return tuple(free)
    pos = {e.name: i for i, e in enumerate(write.index) if isinstance(e, Var)}
    order = list(free)
    return tuple(sorted(free, key=lambda ax: (1, pos[ax.name]) if ax.name in pos else (0, order.index(ax))))


def recognized_tile(op: LoopOp, output_name: str, name: str = "") -> TileOp:
    """The total lift: ``op``'s body → the UNMAPPED lifted ``TileOp`` (free axes peeled and
    output-ordered, every parseable reduce a typed :class:`Fold`, boundary effects split to
    :class:`Store`\\ s). This is the ONE entry point the live compile and the strict golden
    decode share — a record's kernel identity derives by calling exactly this on the record's
    own lowered target. Classification of the lifted tree (which algebra each fold realizes)
    is the tile schedule's concern, downstream."""
    free, cell = _peel(Body(tuple(op.body)))
    cell = _lift_cell(list(cell))
    split = split_effects(tuple(cell))
    cell, stores = (list(split[0]), split[1]) if split is not None else (cell, ())
    node = _form_root(cell)
    if any(st.sweep is not None for st in stores) and isinstance(node, Fold) and node.axis is not None:
        # A sweep store rides a projecting zero-axis root (the materializer's flat-root arm
        # asserts ``sweep is None``) — wrap the bare fold in its empty projection.
        node = Fold.projection(body=Body(()), operands=(node,))
    ordered = _order_free_by_output(node, free, stores)
    node = classify(node, ordered)
    return TileOp(op=node, name=name, place=Placement(free=ordered), inputs=dict(op.inputs), stores=stores)
