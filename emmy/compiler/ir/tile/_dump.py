"""The structural dump — the STORED term as a tree, and nothing derived.

Split out of ``ops``: this is a debugging VIEW, read by ``emmy compile --ir tile`` and the
``EMMY_DUMP_DIR`` artifacts, and no lowering path consults it. Keeping it beside the compute
reads made ``ops`` half presentation."""

from __future__ import annotations

from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.schedule.classic import CLASSIC_FAMILIES
from emmy.compiler.ir.stmt import Body, Load
from emmy.compiler.ir.stmt.base import Stmt, pretty_body
from emmy.compiler.ir.tile.ir import ProjectionRegion
from emmy.compiler.ir.tile.ops import axis_names, sched_of

# --------------------------------------------------------------------------- #
# The structural dump — the STORED term as a tree, and NOTHING derived.
#
# The tile term is a tree of one node kind over operand EDGES, and every fact a pass
# dispatches on is a stored param on a node (this module's whole premise). The dump renders
# exactly that: each node's own header, its stored params as labelled branches, and each operand
# edge's positional lambda binding before recursing into it — so an inline COMPUTED edge is
# visibly a subtree and a MATERIALIZED one visibly a leaf ``Load``.
#
# It renders NO derived material. The structure is already complete in the stored tree — the
# operand edges and their nesting — and a derived evaluation (the per-cell step, the synthesized
# nodes inside it, the loop nest a node lowers to) is a CONSEQUENCE of the stored params, exactly
# as re-derivable as ``Fold.lower()``'s output. Printing it beside storage is the inversion this module
# exists to prevent, and it is bulk: measured over eight kernels the step branch restated
# ``lift`` + ``combine`` and contributed no schedule site on seven of them. ``--ir loop`` is where
# a reader goes for a body.
#
# Schedule choices are not on the term at all. The owning ``TileOp`` supplies one complete
# generic ``Schedule`` whose node choices annotate their canonical sites.
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
        # an output specification's sweep axis (off-term) — the same three the cut's closure
        # check unions. Without a tile the placement is unknown, so ``captures`` declines to
        # answer rather than report grid coordinates as captured values.
        self.axes = None if tile is None else axis_names(root) if root is not None else set()
        if tile is not None:
            self.axes |= {a.name for a in (*tile.place.free, *tile.place.grid)}
            self.axes |= {st.sweep.name for st in tile.output_specs if st.sweep is not None}

    def captures(self, lam) -> tuple[str, ...]:
        """The VALUE names ``lam``'s body reads but neither binds nor takes from the iteration
        space. Non-empty means the λ is NOT closed, which is exactly what
        makes a subtree unhoistable to an operand edge; no stored term prints one (the computed-A
        cone binds the statistic's ``m`` positionally — ``ops.make_cone``), so an annotation marks
        a hand-built tree. Empty when the iteration space is unknown (no owning ``TileOp``) — an
        unanswered question prints as no annotation, never as "closed"."""
        return ()  # nothing captures: a Lambda binds everything it reads (Lambda.__post_init__)

    def note(self, node) -> str:
        """The schedule annotation for ``node`` — every slice the kernel keys against it, spelled
        by the codec (``''`` = the family's decided-empty)."""
        if self.sched is None:
            return ""
        bits = []
        for family in CLASSIC_FAMILIES:
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
            elif isinstance(s, ProjectionRegion):
                out.append(f"{cont}  project[{_axis_span(s.axis)}]{' unroll' if s.unroll else ''}")
                out.append(f"{cont}    {_lam_sig(s.lift, ctx)}")
                out.extend(_stmts(s.body, ctx)(cont + "    "))
            else:
                out.extend(pretty_body(Body((s,)), cont + "  "))
        return out

    return render


def _subtree(node, ctx: _Ctx):
    return lambda cont: _branch(_items(node, ctx), cont)


def _edge(edge, ctx: _Ctx, result: str | None = None) -> tuple[str, object]:
    """One operand edge and the lift params it binds — a ``Load`` is a leaf spelled inline, a
    computed edge recurses into the node stored on it. The binding is explicit because the
    bilinear view prints edges in A/B role order, which can differ from the lift's positional
    parameter order."""
    names = edge.exposes
    head = f"operand[{', '.join(names)}]" + (f" -> {result}" if result is not None else "")
    if isinstance(edge, Load):
        load = edge.pretty()[0].strip().removeprefix(f"{edge.name} = ")
        return f"{head}: {load}   ‹materialized›", lambda cont: []
    return f"{head}: {_head(edge, ctx)}   ‹computed›", _subtree(edge, ctx)


def _items(node, ctx: _Ctx) -> list[tuple[str, object]]:
    """A node's STORED children, each a labelled branch with operand bindings explicit. Nothing
    derived: the step, the synthesized nodes inside it and the lowered nest are all consequences
    of these params."""
    items: list[tuple[str, object]] = []
    if not isinstance(node, Fold):
        return items
    con = node._contraction
    if con is not None:
        # The bilinear reading presents A before its B channels even though its stored operand
        # order is ``(b₀, a, b₁…)``. Each edge's bracket names the positional lift param, so
        # the presentation order cannot be mistaken for the binding order.
        a, chans = con
        items.append(_edge(a, ctx))
        one = len(chans) == 1
        items += [_edge(ch.b, ctx, None if one else ch.acc) for ch in chans]
    else:
        items += [_edge(e, ctx) for e in node.operands]
    if node.axis is not None:
        init = ", ".join(x if isinstance(x, str) else format(x, "g") for x in node.init)
        items.append((f"init: ({init})", lambda cont: []))
    # Always emitted, even for an empty body: the branch carries the SIGNATURE, and a node's
    # binder is storage whether or not it computes anything (an identity projection binds too).
    items.append((f"lift: {_lam_sig(node.lift, ctx)}", _stmts(node.lift.body, ctx)))
    if node.combine is not None:
        items.append((f"combine: {_lam_sig(node.combine, ctx)}", _stmts(node.combine.body, ctx)))
    if node.observe is not None:
        items.append((f"observe: {_lam_sig(node.observe, ctx)}", _stmts(node.observe.body, ctx)))
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
    each node's kind and params, and which lift param each recursed operand edge binds. No derived
    material: the per-cell step, the nodes synthesized inside it and the lowered nest all follow
    from these params (``--ir loop`` is where a body lives). Pass ``tile`` — the owning
    ``TileOp`` — to annotate each node with its accepted schedule choices. A bare stmt
    falls back to its own pretty."""
    ctx = _Ctx(tile, root=op)
    if isinstance(op, Fold):
        return [f"{indent}{_head(op, ctx)}", *_branch(_items(op, ctx), indent)]
    if isinstance(op, Stmt):
        return list(op.pretty(indent))
    return [f"{indent}{op!r}"]


__all__ = ["pretty", "tile_body"]


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
    if tile.schedule is not None and tile.schedule.kernel.work.spell():
        out.append(f"work   {tile.schedule.kernel.work.spell()}")
    if tile.workers is not None:
        out.append(f"band   {tile.workers.spell()}")
    return out


def tile_body(tile) -> str:
    """Render the kernel structurally (the dump view) — no lowering and nothing derived: the
    caller facts that live beside the term (``place`` / ``workers`` / ``classic.kernel.work``), then the stored
    ``op`` tree with each node's accepted choices annotated, then the kernel-boundary ``stores``
    (the root ``Write``\\s live there, so a dump without them would hide where the kernel's
    output lands). The regions are the owners — geometry, algebra, schedule, boundary — kept
    visibly apart."""
    if tile.op is None:
        return ""
    lines = [f"    {line}" for line in _pretty_place(tile)]
    lines += pretty(tile.op, "    ", tile=tile)
    outputs = [
        f"{f'sweep({spec.sweep.name}) ' if spec.sweep else ''}{line.strip()}" for spec in tile.output_specs for line in spec.write.pretty()
    ]
    lines += _pretty_region("outputs", outputs)
    return "\n".join(lines)
