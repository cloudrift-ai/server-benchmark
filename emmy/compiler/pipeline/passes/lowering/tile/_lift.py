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

from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.pure.fold import Fold, deep_defines, deep_reads
from emmy.compiler.ir.stmt import Assign, Body, Init, Load, Loop, Select, Write
from emmy.compiler.ir.stmt.base import Stmt
from emmy.compiler.ir.tile import Placement, TileOp, split_effects
from emmy.compiler.pipeline.passes.lowering.tile._classify import classify, pair_softmax
from emmy.compiler.pipeline.passes.lowering.tile._fromloop import _stamp_axes, fold_from_loop

# --------------------------------------------------------------------------- #
# Peel — the outer free-loop chain becomes the kernel's parallel axes.
# --------------------------------------------------------------------------- #


def _peel(body: Body) -> tuple[list, list[Stmt]]:
    """Split a body into ``(free_axes, per_cell_stmts)``.

    The outer chain of **free** loops becomes the parallel axes. At every level of the
    chain a leading run of pure stmts (``Load`` / ``Assign`` / ``Init`` / ``Select`` — loop-invariant
    loads hoisted above or between the free loops, e.g. a broadcast row scale ``rs[m]``
    read once per ``m``) is sunk into the per-cell body, re-evaluated per cell. The chain
    stops at the first reduce loop, branch, or non-pure stmt — everything from there down
    is the per-cell body."""
    axes: list = []
    prefix: list[Stmt] = []
    cur = list(body)
    while True:
        i = 0
        while i < len(cur) and isinstance(cur[i], (Load, Assign, Init, Select)):
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
    # Reads by stmts that stay AHEAD of the loop: the suffix, plus — walking backwards — every
    # pending stmt that does not sink (a raw loop, a fold, a kept stmt). A value one of those
    # reads cannot move past it into the λ, or it would read an undefined name.
    outside = set(suffix_reads)
    for i in range(len(pending) - 1, -1, -1):
        s = pending[i]
        defs = set(s.defines()) if isinstance(s, (Load, Assign, Select)) else set()
        if not defs & need:
            outside |= set(deep_reads([s]))  # a fold / raw loop / Init never sinks
            continue
        if defs & outside:
            if isinstance(s, Load) and not s.deps() and not any(e.free_vars() for e in s.index or ()):
                dup[i] = {nm: f"{nm}__stat" for nm in defs}
                take.add(i)
                continue
            outside |= set(s.deps())
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


def _chain(cell: list) -> list:
    """Close every fold over the values it reads from the cell: a fold whose λ reads a name a
    PURE cell stmt defines (a normalized row's scale, a sibling statistic's projected result)
    takes that value through a zero-axis projection EDGE — the producing pure stmts as the
    projection's body, the folds they read as its operands, any fold state the consumer also
    reads passed through as a result — bound positionally to a new lift param. The chain is the
    stored form (no fold is demoted for reading the body); a fold consumed only through edges
    leaves the cell. Reads of a sibling fold's carried state stay direct: an operand's λ may read
    an earlier operand's state (operands lower first), and the online-softmax pairing reads the
    max's state off the denominator's lift by name."""
    from emmy.compiler.ir.pure import Lambda  # noqa: PLC0415
    from emmy.compiler.ir.stmt import Assign, Load, Select  # noqa: PLC0415

    def _reads_of(t) -> set[str]:
        if isinstance(t, Fold):
            return set(deep_reads(list(t.lift.body))) - set(t.lift.params) - {n for x in t.lift.body for n in x.defines()}
        return set(deep_reads([t]))

    def names_of(s) -> set[str]:
        """What ``s`` makes available to later stmts: a fold its carried STATE (the combine's
        results — a planar fold's accumulators are not lift-body defs), a stmt its defs."""
        if isinstance(s, Fold):
            return deep_defines(s) | (set(s.combine.results) if s.combine is not None else set(s.lift.results if s.axis is None else ()))
        return set(s.defines())

    out = list(cell)
    for i, f in enumerate(out):
        if not isinstance(f, Fold) or f.axis is None:
            continue
        bound = {f.axis.name, *f.lift.params}
        free = set(deep_reads([*f.lift.body])) - bound - {n for s in f.lift.body for n in s.defines()}
        pure_defs = {n: j for j, s in enumerate(out[:i]) if isinstance(s, (Assign, Load, Select)) for n in s.defines()}
        # A sibling fold's carried state read directly also chains — through the edge, as a
        # pass-through result — when nothing else in the cell reads that sibling (a state the
        # tail or another fold still reads stays a direct operand read: operands lower first).
        needed = [n for n in free if n in pure_defs]
        # The backward cone of the needed names over the earlier cell stmts: pure stmts join the
        # projection body, folds become its operands (their state reads are then bound params).
        members: set[int] = set()

        _grow(out, i, members, list(needed), names_of)
        # A sibling fold's carried state read directly also chains — through the edge, as a
        # pass-through result — when every OTHER reader of that sibling is already in the cone
        # (a state the tail or an outside fold still reads stays a direct operand read: operands
        # lower first). Pulling one in can make another's readers all-in-cone; iterate.
        while True:
            added = False
            for j, t in enumerate(out[:i]):
                if j in members or not (isinstance(t, Fold) and t.axis is not None) or not (free & names_of(t)):
                    continue
                others = [k2 for k2, u in enumerate(out) if k2 != j and u is not f and _reads_of(u) & names_of(t)]
                if all(k2 in members for k2 in others):
                    members.add(j)
                    needed.extend(sorted(free & names_of(t)))
                    added = True
            if not added:
                break
        if not needed:
            continue
        operands = tuple(out[j] for j in sorted(members) if isinstance(out[j], Fold))
        body = tuple(out[j] for j in sorted(members) if not isinstance(out[j], Fold))
        edge_states = {n for e in operands for n in names_of(e)}
        passthrough = sorted((free & edge_states) - set(needed))
        # A pure cone member some OTHER cell stmt also reads stays in the cell too; the edge's
        # copy is α-renamed so the two spellings never define one name twice when both lower.
        shared = {
            n
            for j in members
            if not isinstance(out[j], Fold)
            for n in out[j].defines()
            if any(k2 not in members and k2 != i and n in _reads_of(u) for k2, u in enumerate(out))
        }
        # Renaming one member re-spells the whole pure cone: the kept cell copies read each
        # other, so every pure member stays live in the cell once any does. The FOLD members'
        # names — their carried state and internal defs — rename with it: a shared pure member
        # keeps its reader-fold in the cell too, so the edge's fold copy would otherwise define
        # the cell fold's accumulator a second time in the same rendered scope.
        rename = (
            {n: f"{n}__e{i}" for j in members for n in (names_of(out[j]) if isinstance(out[j], Fold) else set(out[j].defines()))}
            if shared
            else {}
        )
        ren = _renamer(rename)
        results = tuple(ren(n) for n in (*sorted(needed), *passthrough))
        if rename:
            operands = tuple(o.rewrite(ren) for o in operands)
            body = tuple(s.rewrite(ren) for s in body)
            lift = Lambda(
                params=(*f.lift.params, *results),
                body=Body(tuple(s.rewrite(ren) for s in f.lift.body)),
                results=tuple(ren(r) if isinstance(r, str) else r for r in f.lift.results),
            )
        else:
            lift = Lambda(params=(*f.lift.params, *results), body=f.lift.body, results=f.lift.results)
        edge = Fold.projection(operands=operands, body=Body(body), results=results)
        out[i] = replace(f, operands=(*f.operands, edge), lift=lift)
    # A stmt every reader reaches through an edge lives there only: the edge's body / operands
    # re-spell it, so a copy left in the cell would define the name twice when both lower. A
    # fold's OUTSIDE reads are its λ's free names — what its params (the edges) do not bind.
    in_edge = {
        id(m)
        for t in out
        if isinstance(t, Fold)
        for e in t.operands
        if isinstance(e, Fold) and e.axis is None
        for m in (*e.operands, *e.body)
    }

    def outside_reads(t) -> set[str]:
        if isinstance(t, Fold):
            return set(deep_reads(list(t.lift.body))) - set(t.lift.params) - {n for s in t.lift.body for n in s.defines()}
        return set(deep_reads([t]))

    # Liveness is a fixpoint: an in-edge stmt stays when something outside the edges reads it, or
    # when a stmt that stays reads it.
    live_ids = {id(s) for s in out if id(s) not in in_edge}
    while True:
        grew = False
        for s in out:
            if id(s) in live_ids:
                continue
            names = names_of(s)
            if any(id(t) in live_ids and t is not s and outside_reads(t) & names for t in out):
                live_ids.add(id(s))
                grew = True
        if not grew:
            break
    return [s for s in out if id(s) in live_ids]


def _form_root(cell: list, *, chain: bool = False, sweeps: frozenset = frozenset()):
    """Shape the cell into the stored root: top-level folds hoist to the projection's OPERANDS
    (in body order — an operand's λ may read an earlier operand's carried state), the rest is
    the projection body. Operands lower BEFORE the body, so a fold reading a name the body
    defines cannot hoist: it stays a BODY member — a term is a legal body stmt — until
    :func:`_chain` closes it over those values through a projection edge (``chain=True``, run
    after the online-softmax pairing has consumed its sibling pair), after which it hoists. A
    single bare fold with nothing else is the root node itself."""
    kept = _chain(list(cell)) if chain else list(cell)
    body_defs = {n for s in kept if not isinstance(s, Fold) for n in deep_defines(s)}
    # A fold reading a boundary store's SWEEP axis runs once per swept element — it is a body
    # member (a stored node in place), never an operand (operands lower ahead of the sweep).
    folds = tuple(s for s in kept if isinstance(s, Fold) and not ((deep_reads(s.lower()) | stmt_axis_reads(s)) & (body_defs | sweeps)))
    body = tuple(s for s in kept if not any(s is f for f in folds))
    if len(folds) == 1 and not body:
        return folds[0]
    return Fold.projection(body=Body(body), operands=folds)


def stmt_axis_reads(s) -> set[str]:
    """Every axis / value name a node's lowering reads (its coordinate exprs included)."""
    return {v for st in s.lower() for e in _deep_exprs(st) for v in e.free_vars()}


def _deep_exprs(st):
    yield from st.exprs()
    for b in st.nested():
        for child in b:
            yield from _deep_exprs(child)


def _renamer(mapping: dict):
    def ren(n: str) -> str:
        return mapping.get(n, n)

    return ren


def _grow(out: list, i: int, members: set[int], frontier: list, names_of) -> None:
    """Extend ``members`` with the backward cone of ``frontier`` over ``out[:i]``: the latest
    definer of each name joins; a pure stmt's own reads join the frontier, a fold is a leaf."""
    while frontier:
        n = frontier.pop()
        for j in range(i - 1, -1, -1):
            s = out[j]
            if j in members or n not in names_of(s):
                continue
            members.add(j)
            if not isinstance(s, Fold):
                frontier.extend(s.deps())
            break


def _chain_root(node, sweeps: frozenset = frozenset()):
    """Re-form a projection root with its cell CLOSED (:func:`_chain`): every fold left in the
    body for reading a body-defined value takes that value through a projection edge and hoists.
    Runs after the pairing, which reads the ``(max, den)`` siblings by name."""
    if not (isinstance(node, Fold) and node.axis is None and any(isinstance(s, Fold) for s in node.body)):
        return node
    return _form_root([*node.operands, *node.body], chain=True, sweeps=sweeps)


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
    # Position by index-expr membership, not bare-Var identity, at the INNERMOST dim carrying the
    # axis: a re-fused axis reaches the store as the split pair ``f/Q`` … ``f%Q``, and the
    # remainder dim is the one its unit step moves — under a transposed output
    # (``[…, f/Q, s, f%Q]``) the quotient dim sits outside another axis entirely.
    pos: dict[str, int] = {}
    for i, e in enumerate(write.index):
        for v in e.free_vars():
            pos[v] = i
    order = list(free)
    return tuple(sorted(free, key=lambda ax: (1, pos[ax.name]) if ax.name in pos else (0, order.index(ax))))


def recognized_tile(op: LoopOp, name: str = "") -> TileOp:
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
    sweeps = frozenset(st.sweep.name for st in stores if st.sweep is not None)
    node = _chain_root(pair_softmax(_form_root(cell, sweeps=sweeps)), sweeps)
    if any(st.sweep is not None for st in stores) and isinstance(node, Fold) and node.axis is not None:
        # A sweep store rides a projecting zero-axis root (the materializer's flat-root arm
        # asserts ``sweep is None``) — wrap the bare fold in its empty projection.
        node = Fold.projection(body=Body(()), operands=(node,))
    ordered = _order_free_by_output(node, free, stores)
    node = classify(node, ordered)
    return TileOp(op=node, name=name, place=Placement(free=ordered), inputs=dict(op.inputs), stores=stores)
