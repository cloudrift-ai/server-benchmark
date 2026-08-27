r"""Schedule a lifted ``TileOp`` by walking its stored Fold tree.

One recursive generator IS the enumeration. A Fold offers its own options; each option extends the
:class:`Ctx` of what the kernel has already agreed, and the subtree below is walked under that
extended context. Siblings thread left to right, so a choice anywhere restricts everything
enumerated after it::

    S(node, ctx) = for each option o of node under ctx:  o x S(children(node), ctx + o)

There is no product over a flat site list and no join afterwards. The three reasons two sites are
not one kernel — one worker inventory, agreeing tile geometry on a shared physical axis, and one
decision per Fold however many paths reach it — are stated once, in :meth:`Ctx.extend`, and applied
while descending, so an illegal combination is never built. Traversal order is the fork order:
``WORK`` leads because the root owns the free axes it is read off, and the site keys follow as the
walk decides them.

**PROTOTYPE.** The smallest thing that walks the tree and lowers a kernel. It offers the serial and
cooperative reduce partitions and the scalar contraction tile — no tensor-core tile, staging,
cross-CTA split, register strip, producer band, fragment seam, launch-order swizzle, derived site or
env pin — and it enumerates eagerly into a list.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from emmy.compiler.ir.pure.fold import Fold, is_contraction
from emmy.compiler.ir.schedule import ReducePlan, TilePlan, Workers, derive_inventory, plan_workers, resolve_site_tile
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.ir.tile.identity import hint_extent
from emmy.compiler.ir.tile.ops import Sched, scheduled
from emmy.compiler.pipeline.fork import Fork
from emmy.compiler.pipeline.passes.lowering.tile._tree import children, walk
from emmy.compiler.pipeline.search.space import WORK, coop_reduce_moves, scalar_tile_moves


def _kids(node) -> tuple:
    """``node``'s stored Fold children — the shared walk's answer, without the axes the cut pass
    needs and the schedule does not."""
    return tuple(child for child, _axes in children(node))


def _nodes(node) -> Iterator:
    """The stored tree, preorder — the shared walk again, so a Fold the cut pass can see is one
    this pass can schedule."""
    return (node for node, _axes in walk(node))


@dataclass(frozen=True)
class _Option:
    """One site's local choice: what it spells, the worker inventory that claims (``None`` claims
    nothing and composes with any), and the placed tile the rest of the kernel must agree with."""

    knobs: dict
    work: Workers | None = None
    tile: TilePlan | None = None


def _options(sched: Sched, node) -> list[_Option]:
    """Everything this ONE Fold can spell, ignoring every other site. Dispatch is the two stored
    params of the node — never the operation family.

    **This is also where legality lives, and it is not a separate idea.** A candidate this node
    cannot realize is one this function does not return; there is no second pass that offers a row
    and then withdraws it. The catalogs carry the constraints that are a function of the MOVE (the
    scalar space is generated under the CTA thread budget, so no member can exceed it) and the
    guards below carry the ones that are a function of the NODE. Nothing here may narrow for
    SPEED — a slow candidate is a fork the evidence decides, never a row withheld."""
    if not isinstance(node, Fold) or node.axis is None:
        return [_Option({})]  # a per-cell projection decides nothing, but its children do
    if is_contraction(node):
        if len(node.channels) != 1:
            # A multi-channel product needs one accumulator family per channel, which is a warp
            # compute-fill form; the scalar atom carries a single fold. Nothing on offer realizes
            # it, so the node offers nothing and the term stays unmapped — the guardrail contract,
            # not a crash at materialization.
            return []
        key = sched.key("TILE", node)
        placed = [(p, sched.placed(node, p)) for p in scalar_tile_moves()]
        return [
            _Option({key: p.spell()} if key else {}, plan_workers(p), q if p.is_tiled else None)
            for p, q in placed
            # A tile the grid cannot bind to an (m, n) pair has no geometry to realize.
            if not p.is_tiled or q.axes is not None
        ]
    key = sched.key("REDUCE", node)
    extent = hint_extent(node.axis)
    # A band wider than the axis has work for cannot fill its workers; the split and transposed
    # families are not offered at all, so their own conditions never arise.
    bands = [p for p in coop_reduce_moves() if not p.needs_split and not p.coop_transposed and p.coop <= extent and p.reg <= extent]
    return [_Option({key: p.spell()} if key else {}, derive_inventory((), coop=p.coop)) for p in (ReducePlan(), *bands)]


@dataclass(frozen=True)
class Ctx:
    """What the walk has already decided for the WHOLE kernel, carried down and across siblings.

    ``work`` — a kernel has ONE worker inventory. ``axes`` — two sites sharing a physical grid axis
    must give it the same tile and units. ``decided`` — one Fold reached by several paths is ONE
    decision, so a later path can only re-spell what the first chose."""

    work: Workers | None = None
    axes: dict = field(default_factory=dict)
    decided: dict = field(default_factory=dict)

    def extend(self, option: _Option) -> Ctx | None:
        """This context with ``option`` folded in, or ``None`` when the option contradicts it."""
        if any(self.decided.get(k, v) != v for k, v in option.knobs.items()):
            return None
        work = self.work
        if option.work is not None:
            if work not in (None, option.work):
                return None
            work = option.work
        axes = dict(self.axes)
        for side in option.tile.mn if option.tile is not None else ():
            if axes.setdefault(side.axis.name, (side.tile, side.units)) != (side.tile, side.units):
                return None
        return Ctx(work, axes, {**self.decided, **option.knobs})


# ---- the walk, reified as the fork tree ---------------------------------------------------------- #


@dataclass(frozen=True)
class _State:
    """The per-kernel constants every node of the fork tree shares."""

    tile: TileOp
    sched: Sched
    name: str
    knobs: dict


def _spelled(knobs: dict, option: _Option, ctx: Ctx) -> dict:
    """The row prefix one decision leaves behind: what the option spells, plus the inventory as
    soon as any option claims it — :meth:`Ctx.extend` refuses a second one, so a prefix that
    carries ``WORK`` already carries its final value."""
    out = {**knobs, **option.knobs}
    if ctx.work is not None:
        out[WORK.name] = ctx.work.spell()
    return out


def _step(state: _State, stack: tuple, ctx: Ctx, knobs: dict) -> list[Fork]:
    """One level of the walk: descend past every FORCED decision, then return the siblings standing
    at the first real choice — or the leaf, when the stack runs out.

    ``stack`` is the walk's own work list. Popping a node and pushing its children is what makes
    this the same depth-first order a recursive generator would take, with the difference that the
    remainder is DATA, so a sibling can be resumed later instead of having to be produced now."""
    while stack:
        node, rest = stack[0], stack[1:]
        offers = [(o, below) for o in _options(state.sched, node) if (below := ctx.extend(o)) is not None]
        if not offers:
            return []  # nothing schedules under here
        children = _kids(node) + rest
        if len(offers) == 1:
            option, ctx = offers[0]
            knobs, stack = _spelled(knobs, option, ctx), children
            continue  # a level with one option is no choice at all — collapse it
        return [_Branch(state, children, below, _spelled(knobs, o, below)) for o, below in offers]
    return [_Leaf(state, {**knobs, WORK.name: ctx.work.spell() if ctx.work is not None else ""})]


@dataclass(frozen=True)
class _Branch(Fork):
    """A partly-walked schedule: the nodes still to decide, the context they must honour, and the
    row prefix decided so far. The subtree does not exist until ``expand`` walks one level more."""

    state: _State
    stack: tuple
    ctx: Ctx
    knobs: dict
    is_leaf = False

    def expand(self) -> list[Fork]:
        return _step(self.state, self.stack, self.ctx, self.knobs)


@dataclass(frozen=True)
class _Leaf(Fork):
    """A complete walk: ``knobs`` is the kernel's whole identity, materialized on demand."""

    state: _State
    knobs: dict
    is_leaf = True

    def expand(self) -> list[TileOp]:
        return [_materialize(self.state, self.knobs)]


def _materialize(state: _State, row: dict) -> TileOp:
    """One row -> its ``TileOp``, every slice RE-RESOLVED from the row's own spellings over the same
    ``_nodes`` order the walk decided in. The row is the kernel's complete identity, so
    decode-by-spelling is what makes it replayable."""
    sched, tile = state.sched, state.tile
    work = Workers.parse(row.get(WORK.name) or None)
    slices = []
    for node in _nodes(tile.op):
        if not isinstance(node, Fold) or node.axis is None:
            continue
        red = ReducePlan.parse(row.get(sched.key("REDUCE", node) or "") or None, work)
        if not is_contraction(node):
            slices.append(("REDUCE", node, red if red.stages else None))
            continue
        plan = resolve_site_tile(row.get(sched.key("TILE", node) or "") or None, work, red.coop)
        if plan.is_tiled:
            slices.append(("TILE", node, plan))
    return scheduled(
        tile.op, name=state.name, place=sched.place, knobs={**state.knobs, **row}, output_specs=tile.output_specs, slices=slices
    )


def schedule(tile: TileOp, name: str, knobs: dict, ctx) -> list[Fork] | TileOp:
    """Map a newly lifted, unmapped ``tile`` onto the grid and offer its scheduling fork.

    Returns the siblings at the walk's first real choice — each one lazy, holding a work list and a
    context rather than any row — a single ``TileOp`` when the whole walk is forced, or ``[]`` when
    nothing schedules, which is the guardrail contract that leaves the term unmapped."""
    del ctx  # the walk reads only the stored term
    state = _State(tile, Sched(tile.op, {}, place=tile.place.on_grid()), name, knobs)
    # A node that offers nothing offers it under EVERY context — options are a function of the node
    # alone — so one pass over the tree says whether the term has any schedule at all. It is also
    # what keeps a lazy branch honest: past this check every node still has an option that composes
    # with anything (the per-cell tile, the serial fold), so no branch can expand to nothing and
    # promise leaves it does not have.
    if any(not _options(state.sched, node) for node in _nodes(tile.op)):
        return []
    options = _step(state, (tile.op,), Ctx(), {})
    if len(options) == 1 and options[0].is_leaf:
        return options[0].expand()[0]
    return options


__all__ = ["Ctx", "schedule"]
