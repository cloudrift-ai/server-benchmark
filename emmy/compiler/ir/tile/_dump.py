"""The structural dump — the STORED term as a tree, and nothing derived.

Split out of ``ops``: this is a debugging VIEW, read by ``emmy compile --ir tile`` and the
``EMMY_DUMP_DIR`` artifacts, and no lowering path consults it. Keeping it beside the compute
reads made ``ops`` half presentation."""

from __future__ import annotations

from emmy.compiler.ir.stmt import Body, Load
from emmy.compiler.ir.stmt.base import Stmt, pretty_body
from emmy.compiler.ir.tile.ir import Fold
from emmy.compiler.ir.tile.ops import axis_names, sched_of


def unplaced_slices(tile) -> list[tuple[str, object]]:
    """The kernel's schedule entries that NO stored node carries — the keys addressing DERIVED
    material (flash's synthesized PV, ``TILE@pj``), sorted by key.

    Every other slice reaches its reader by annotating the node it keys against. These cannot: the
    node they address is a consequence of the stored params, not one of them. They are a schedule
    fact either way, so the dump prints them in its schedule region rather than reconstructing the
    derived node inside the term to hang them on. Empty for a kernel whose sites are all stored —
    which, measured over the frontend's kernels, is all of them but causal flash."""
    if tile.op is None or not tile.schedule:
        return []
    from emmy.compiler.ir.tile.path import sites  # noqa: PLC0415 — path imports ir; keep ops light

    sched = sched_of(tile)
    stored = [s.node for s in sites(tile.op) if not s.derived]
    claimed = {k for nd in stored for f in ("TILE", "REDUCE", "STAGE") if (k := sched.key(f, nd)) is not None}
    return sorted((k, v) for k, v in tile.schedule.items() if k not in claimed)


# --------------------------------------------------------------------------- #
# The structural dump — the STORED term as a tree, and NOTHING derived.
#
# The tile term is a tree of three node kinds over operand EDGES, and every fact a pass
# dispatches on is a stored param on a node (this module's whole premise). The dump renders
# exactly that: each node's own header, its stored params as labelled branches, and each operand
# edge recursed into — so an inline COMPUTED edge is visibly a subtree and a MATERIALIZED one
# visibly a leaf ``Load``.
#
# It renders NO derived material. The structure is already complete in the stored tree — the
# operand edges and their nesting — and a derived evaluation (the per-cell step, the synthesized
# nodes inside it, the loop nest a node lowers to) is a CONSEQUENCE of the stored params, exactly
# as re-derivable as ``Fold.lower()``'s output. Printing it beside storage is the inversion this module
# exists to prevent, and it is bulk: measured over eight kernels the step branch restated
# ``lift`` + ``combine`` and contributed no schedule site on seven of them. ``--ir loop`` is where
# a reader goes for a body.
#
# Schedule slices are not on the term at all — they annotate a node from the owning
# ``TileOp.schedule`` when one is supplied. The one slice that can address DERIVED material
# (flash's synthesized PV, ``TILE@pj``) has no stored node to annotate, so it prints in the
# ``schedule`` region beside the term rather than dragging the derived node into it.
# --------------------------------------------------------------------------- #

_TEE, _ELBOW, _PIPE, _GAP = "├─ ", "└─ ", "│  ", "   "


class _Ctx:
    """The dump's read-only context — the owning ``TileOp``'s schedule view (so each STORED node
    can be annotated with the slices keyed against it) and the iteration space a λ's capture set
    is measured against. ``None`` everywhere when a bare term is printed without its op."""

    def __init__(self, tile, root=None) -> None:
        self.sched = sched_of(tile) if tile is not None and tile.op is not None else None
        # The ITERATION SPACE a capture set is measured against. Only the OWNING ``TileOp`` knows
        # it in full: the term's own axes (:func:`axis_names`), the placement's free/grid axes, and
        # a boundary store's sweep axis (off-term since 1q) — the same three the cut's closure
        # check unions. Without a tile the placement is unknown, so ``captures`` declines to
        # answer rather than report grid coordinates as captured values.
        self.axes = None if tile is None else axis_names(root) if root is not None else set()
        if tile is not None:
            self.axes |= {a.name for a in (*tile.place.free, *tile.place.grid)}
            self.axes |= {st.sweep.name for st in tile.stores if st.sweep is not None}

    def captures(self, lam) -> tuple[str, ...]:
        """The VALUE names ``lam``'s body reads but neither binds nor takes from the iteration
        space — the same reading the cut's closure predicate applies
        (``_cut._captured_values``). Non-empty means the λ is NOT closed, which is exactly what
        makes a subtree unhoistable to an operand edge: flash's ``P = exp(s − m)`` reads the
        online-softmax carrier's running max, so it can never be an edge and its seam is not
        cuttable. Empty when the iteration space is unknown (no owning ``TileOp``) — an unanswered
        question prints as no annotation, never as "closed"."""
        return () if self.axes is None else tuple(sorted(lam.free_names() - self.axes))

    def note(self, node) -> str:
        """The schedule annotation for ``node`` — every slice the kernel keys against it, spelled
        by the codec (``''`` = the family's decided-empty)."""
        if self.sched is None:
            return ""
        bits = []
        for family in ("TILE", "REDUCE", "STAGE"):
            slice_ = self.sched.get(family, node)
            if slice_ is not None:
                bits.append(f"{family}={slice_.spell() or '·'}")
        return f"   ⟨{' '.join(bits)}⟩" if bits else ""


def _lam_sig(lam, ctx: _Ctx | None = None) -> str:
    """A lambda's one-line signature. A float result is the ι literal injected in the lift
    (softmax's singleton ``(x, 1)``), which has no def to name.

    A non-empty CAPTURE set is spelled between the params and the results — without it a λ that
    reads an enclosing value would print as though it were closed, which is the one property the
    reader most needs (an unclosed subtree can never become an operand edge)."""
    rs = ", ".join(r if isinstance(r, str) else format(r, "g") for r in lam.results)
    cap = ctx.captures(lam) if ctx is not None else ()
    free = f" [captures {', '.join(cap)}]" if cap else ""
    return f"λ({', '.join(lam.params)}){free} -> ({rs})"


def _axis_span(axis) -> str:
    win = getattr(axis, "window", None)
    parent = f" ⊂ {win.parent.name}" if win is not None and win.parent is not None else ""
    return f"{axis.name} in 0..{axis.extent}{parent}"


def _kind(edge) -> str:
    """An operand edge's inhabitant — the two things an input can be."""
    return "materialized" if isinstance(edge, Load) else "computed"


def _head(node, ctx: _Ctx) -> str:
    """One node's header line — its kind and the stored params that fit on a line. A λ-valued
    field is NOT one of them: its signature belongs on its own branch, next to the body it binds
    (``lift:`` / ``combine:`` / ``fn:``), not one screenful above it."""
    if not isinstance(node, Fold):
        return str(node)
    if node.axis is None:
        text = "Fold  free" + ("" if node.operands else "  ‹pointwise›")
    else:
        text = f"Fold[{_axis_span(node.axis)}] {node.role.name.lower()}" + (" unroll" if node.unroll else "")
    return text + ctx.note(node)


def _stmts(stmts, ctx: _Ctx):
    """Render a λ body (a ``lift`` / ``combine`` / the zero-axis fold's ``lift``) — indented two under the signature
    line that binds it, so the program reads as the binder's body rather than as a sibling of the
    branch labels. A structural NODE may occupy a statement position here (a demoted cone's inline
    node); it expands in place, since a lift body is storage like any other."""

    def render(cont: str) -> list[str]:
        out: list[str] = []
        for s in stmts:
            if isinstance(s, Fold):
                out.append(f"{cont}  {_head(s, ctx)}")
                out.extend(_branch(_items(s, ctx), cont + "  "))
            else:
                out.extend(pretty_body(Body((s,)), cont + "  "))
        return out

    return render


def _subtree(node, ctx: _Ctx):
    return lambda cont: _branch(_items(node, ctx), cont)


def _edge(label: str, edge, ctx: _Ctx) -> tuple[str, object]:
    """One operand edge as a tree item — a ``Load`` is a leaf spelled inline, a computed edge
    recurses into the node stored on it."""
    if isinstance(edge, Load):
        return f"{label}: {edge.pretty()[0].strip()}   ‹materialized›", lambda cont: []
    return f"{label}: {_head(edge, ctx)}   ‹computed›", _subtree(edge, ctx)


def _items(node, ctx: _Ctx) -> list[tuple[str, object]]:
    """A node's STORED children, each a labelled branch. Nothing derived: the step, the
    synthesized nodes inside it and the lowered nest are all consequences of these params."""
    items: list[tuple[str, object]] = []
    if not isinstance(node, Fold):
        return items
    con = node._contraction
    if con is not None:
        # The bilinear reading labels its edges ``a`` / ``b`` — the same labels the path codec
        # spells, so a reader can match a dump line to a ``PLACE@a`` key by eye.
        a, chans = con
        items.append(_edge("operand[a]", a, ctx))
        one = len(chans) == 1
        items += [_edge("operand[b]" if one else f"operand[b{i}] -> {ch.acc}", ch.b, ctx) for i, ch in enumerate(chans)]
    else:
        items += [_edge(f"operand[{i}]", e, ctx) for i, e in enumerate(node.operands)]
    if node.axis is not None:
        init = ", ".join(x if isinstance(x, str) else format(x, "g") for x in node.init)
        items.append((f"init: ({init})", lambda cont: []))
    # Always emitted, even for an empty body: the branch carries the SIGNATURE, and a node's
    # binder is storage whether or not it computes anything (an identity projection binds too).
    items.append((f"lift: {_lam_sig(node.lift, ctx)}", _stmts(node.lift.body, ctx)))
    if node.combine is not None:
        items.append((f"combine: {_lam_sig(node.combine, ctx)}", _stmts(node.combine.body, ctx)))
    return items


def _branch(items: list[tuple[str, object]], cont: str) -> list[str]:
    out: list[str] = []
    for i, (head, sub) in enumerate(items):
        last = i == len(items) - 1
        out.append(f"{cont}{_ELBOW if last else _TEE}{head}")
        out.extend(sub(cont + (_GAP if last else _PIPE)))
    return out


def pretty(op, indent: str = "", *, tile=None) -> list[str]:
    """Structurally pretty-print a kernel op (for dumps) as the STORED tree and nothing else —
    each node's kind and params, its operand edges recursed into. No derived material: the
    per-cell step, the nodes synthesized inside it and the lowered nest all follow from these
    params (``--ir loop`` is where a body lives). Pass ``tile`` — the owning ``TileOp`` — to
    annotate each node with the schedule slices keyed against it. A bare stmt falls back to its
    own pretty."""
    ctx = _Ctx(tile, root=op)
    if isinstance(op, Fold):
        return [f"{indent}{_head(op, ctx)}", *_branch(_items(op, ctx), indent)]
    if isinstance(op, Stmt):
        return list(op.pretty(indent))
    return [f"{indent}{op!r}"]


__all__ = ["pretty", "tile_body", "unplaced_slices"]


def _pretty_region(title: str, rows: list[str]) -> list[str]:
    """One titled region below the term, its rows as tree branches. Empty rows print nothing —
    an absent region IS the decided-empty, same as an absent schedule key."""
    if not rows:
        return []
    return [f"    {title}", *(f"    {'└─' if i == len(rows) - 1 else '├─'} {r}" for i, r in enumerate(rows))]


def _pretty_place(tile) -> list[str]:
    """The geometry lines above the term — the placement (free axes and their grid binding,
    or ``unmapped`` before the schedule decides one) and the kernel's one worker
    inventory. Omitted entirely when nothing has been decided yet."""
    out = []
    axes = lambda a: ", ".join(x.name for x in a)  # noqa: E731
    if tile.place.free or tile.place.grid:
        grid = f"grid=({axes(tile.place.grid)})" if tile.place.is_mapped else "unmapped"
        out.append(f"place  free=({axes(tile.place.free)})  {grid}")
    if tile.work is not None:
        out.append(f"work   {tile.work.spell()}")
    if tile.workers is not None:
        out.append(f"wspec  {tile.workers.spell()}")
    return out


def tile_body(tile) -> str:
    """Render the kernel structurally (the dump view) — no lowering and nothing derived: the
    caller facts that live BESIDE the term (``place`` / ``workers`` / ``work``), then the STORED
    ``op`` tree with each node's schedule slices annotated from ``schedule``, then any schedule
    entry no stored node carries (:func:`unplaced_slices` — a slice keyed against DERIVED
    material, which the term deliberately does not show), then the kernel-boundary ``stores``
    (the root ``Write``\\s live there, so a dump without them would hide where the kernel's
    output lands). The regions are the owners — geometry, algebra, schedule, boundary — kept
    visibly apart."""
    if tile.op is None:
        return ""
    lines = [f"    {line}" for line in _pretty_place(tile)]
    lines += pretty(tile.op, "    ", tile=tile)
    lines += _pretty_region("schedule", [f"{k} = {v.spell() or '·'}" for k, v in unplaced_slices(tile)])
    stores = [f"{f'sweep({st.sweep.name}) ' if st.sweep else ''}{ln.strip()}" for st in tile.stores for ln in st.write.pretty()]
    lines += _pretty_region("stores", stores)
    return "\n".join(lines)
