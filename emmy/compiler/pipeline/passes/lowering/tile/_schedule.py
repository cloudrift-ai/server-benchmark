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

**PROTOTYPE.** The smallest thing that walks the tree and lowers a kernel. It offers the whole
reduce-partition catalog and both contraction tiers — the scalar output tile and the tensor-core
warp tile — but no operand staging, cross-CTA split of a CONTRACTION, pointwise register strip,
producer band, fragment seam, launch-order swizzle or derived site, and it enumerates eagerly into
a list. Two warp guards below read as narrowing and are not: a computed operand realizes only
through the shared-memory compute fill, which is a ``STAGE`` and not on offer, and the fp8
gmem-direct tier is simply not restored yet.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

from emmy.compiler.ir.atom import ATOM_REGISTRY, AtomKind, atoms_for
from emmy.compiler.ir.pure.fold import Fold, is_contraction
from emmy.compiler.ir.schedule import ReducePlan, TilePlan, Workers, derive_inventory, plan_workers, resolve_site_tile
from emmy.compiler.ir.stmt import Load, Loop, Write
from emmy.compiler.ir.stmt.passes import has_contraction_tail
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.ir.tile.identity import hint_extent
from emmy.compiler.ir.tile.ops import Sched, edge_dtypes, projection_tail, scheduled
from emmy.compiler.ir.tile.path import SLICE_FAMILIES
from emmy.compiler.pipeline.fork import Fork
from emmy.compiler.pipeline.knob import axis_of
from emmy.compiler.pipeline.passes.lowering._addr import gmem_axis_step, split_addressable
from emmy.compiler.pipeline.passes.lowering.tile._tree import children, walk
from emmy.compiler.pipeline.search.space import (
    F16_MMA_F32_ACC,
    MAX_BLOCK_THREADS,
    REDUCE,
    TILE,
    WARP_LANES,
    WORK,
    coop_reduce_moves,
    precision_pin,
    scalar_tile_moves,
    warp_tile_moves,
)

logger = logging.getLogger(__name__)


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


# ---- what one Fold can spell ---------------------------------------------------------------------- #


def _pin(knob, key: str | None) -> str | None:
    """The live env pin addressing ``key`` — ``EMMY_KNOBS``'s ``FAMILY@<element>`` entry, falling
    back to the bare ``EMMY_<FAMILY>``. Unset reads ``None``, which is the distinction the
    enumeration keys on: an unset family offers its catalog, a set one is authoritative."""
    if key is None:
        return None
    element = axis_of(key)
    return knob.narrow_at(element) if element else knob.raw()


def _is_warp_site(node) -> bool:
    """A single-channel contraction Fold — the one node shape the warp tier can address. A
    multi-channel product needs one accumulator family per channel, which is a warp compute-fill
    form; the scalar atom carries a single fold."""
    return isinstance(node, Fold) and node.axis is not None and is_contraction(node) and len(node.channels) == 1


def _options(state: _State, node) -> list[_Option]:
    """Everything this ONE Fold can spell, ignoring every other site. Dispatch is the two stored
    params of the node — never the operation family.

    **This is also where legality lives, and it is not a separate idea.** A candidate this node
    cannot realize is one this function does not return; there is no second pass that offers a row
    and then withdraws it. The catalogs carry the constraints that are a function of the MOVE (the
    scalar space is generated under the CTA thread budget, so no member can exceed it) and the
    guards below carry the ones that are a function of the NODE. Nothing here may narrow for
    SPEED — a slow candidate is a fork the evidence decides, never a row withheld. A pin narrows
    the same way — an option the pin does not name is not offered — and its refusals are
    two-layered: it DROPS where the node's algebra and operand dtypes select no warp tier (a
    graph-wide pin fans out to siblings it cannot mean), and RAISES where a tier was selected and
    the named plan cannot realize on it."""
    sched = state.sched
    if not isinstance(node, Fold) or node.axis is None:
        return [_Option({})]  # a per-cell projection decides nothing, but its children do
    if is_contraction(node):
        if not _is_warp_site(node):
            # A multi-channel product: nothing on offer realizes it (see :func:`_is_warp_site`),
            # so the node offers nothing and the term stays unmapped — the guardrail contract,
            # not a crash at materialization.
            return []
        key = sched.key("TILE", node)
        placed = [(p, sched.placed(node, p)) for p in _tile_moves(state, node, key)]
        opts = [
            _Option({key: p.spell()} if key else {}, plan_workers(p), q if p.is_tiled else None)
            for p, q in placed
            # A tile the grid cannot bind to an (m, n) pair has no geometry to realize.
            if not p.is_tiled or q.axes is not None
        ]
        return _claimable(state, opts)
    key = sched.key("REDUCE", node)
    opts = [_Option({key: p.spell()} if key else {}, derive_inventory((), coop=p.coop)) for p in _reduce_moves(state, node, key)]
    return _claimable(state, opts)


def _claimable(state: _State, opts: list[_Option]) -> list[_Option]:
    """``opts`` minus the ones whose inventory claim a live ``WORK`` pin refuses. Kernel-global, so
    it is a FACT on :class:`_State` rather than a site decision: an option that claims nothing
    composes with any pin (the leaf asks again once the walk knows what did claim it), and a claim
    the pin does not name can never reach a leaf that honors it."""
    return [o for o in opts if o.work is None or state.honors_work_pin(o.work)]


def _tile_moves(state: _State, node, key: str | None) -> list[TilePlan]:
    """The output tiles this contraction offers: the scalar and warp catalogs, or — under a ``TILE``
    pin — the plan that pin NAMES at each inventory the site can spell it against.

    A pin is authoritative over the VALUE and not over the inventory: the unit widths are read OFF
    ``WORK``, so one pin names a different plan under each one, and it may well name a plan no
    catalog generates — fixing widths no ladder predicts is what a pin is for. It is also
    authoritative over enumeration POLICY (the f16-accumulate precision gate narrows the catalog
    arm only, never a pin). What a pin cannot do is name a plan the node cannot REALIZE, and the
    refusal is two-layered: it DROPS where the node's algebra and operand dtypes select no warp
    tier (a graph-wide pin fans out to siblings it cannot mean; the drop is explained at debug
    level), and RAISES where a tier was selected and the named plan cannot realize on it — an atom
    these fragments cannot bind, an inventory over the CTA thread budget."""
    entry = state.atoms.get(id(node))  # None = the node's algebra / dtypes select no warp tier
    base, f16acc = entry or ((), ())
    pin = _pin(TILE, key)
    if pin is None:
        # The f16-accumulate family is a precision-trading POLICY, off by default — resolved once
        # per kernel on ``_State`` (the precise ``F16_MMA_F32_ACC`` pin is authoritative, else the
        # ``FAST_MATH`` umbrella offers the family everywhere it is legal and the evidence ranks
        # it per shape and card).
        offered = (*base, *(f16acc if state.f16acc else ()))
        return [*scalar_tile_moves(), *(warp_tile_moves(offered) if offered else [])]
    if entry is None and _names_warp_atom(pin):
        # The choice-layer drop: no warp tier here, whatever the pin says. Explicable, not silent.
        if logger.isEnabledFor(logging.DEBUG):
            frag = _fragment_epilogue_ok(projection_tail(state.tile))
            logger.debug("TILE pin %r at %s dropped: %s", pin, key or "TILE", _node_refusal(state.tile, state.ctx, node, frag))
        return []
    if state.work_pinned:
        works = [state.work_pin]
    else:
        catalog = [*scalar_tile_moves(), *(warp_tile_moves((*base, *f16acc)) if entry else [])]
        works = list(dict.fromkeys(plan_workers(p) for p in catalog))
    reduce_pin = _pin(REDUCE, state.sched.key("REDUCE", node))
    out: list[TilePlan] = []
    refused: list[str] = []
    for work in works:
        try:
            # The empty-TILE-beside-a-thread-inventory ambiguity resolves against the REDUCE pin's
            # cooperative width, exactly as a stamped row's spelling does.
            coop = ReducePlan.parse(reduce_pin, work).coop if reduce_pin else 1
            plan = resolve_site_tile(pin, work, coop)
        except ValueError as e:
            refused.append(str(e))
            continue
        if plan in out:
            continue
        why = _plan_refusal(state, node, plan)
        if why is not None:
            refused.append(why)
            continue
        out.append(plan)
    if not out:
        detail = refused[-1] if refused else "no inventory this site can spell resolves it"
        raise ValueError(f"TILE pin {pin!r} at {key or 'TILE'} names no schedule this site can realize: {detail}")
    return out


def _names_warp_atom(pin: str) -> bool:
    """Whether a ``TILE`` pin names a tensor-core atom — probed against a unit warp inventory, so
    the TIER the pin means is known before any inventory question. A malformed pin answers
    ``False`` and stays on the loud path."""
    try:
        return resolve_site_tile(pin, Workers(kind="warp", units=(1, 1))).is_warp
    except ValueError:
        return False


def _plan_refusal(state: _State, node, plan: TilePlan) -> str | None:
    """Why a PINNED plan cannot realize on ``node`` (``None`` when it can) — the same node facts the
    catalogs are generated under, re-asked of the one plan a pin names, since a pin bypasses the
    generators that carry them. The CTA thread budget binds BOTH tiers — the catalogs are
    generated under it, so only a pin can exceed it."""
    if plan.block_threads > MAX_BLOCK_THREADS:
        work = plan_workers(plan)
        return (
            f"inventory {work.spell() if work is not None else plan.spell()} spends {plan.block_threads} threads, which "
            f"exceeds the {MAX_BLOCK_THREADS}-thread/CTA limit; shrink the worker widths or move work to the f register sub-tile."
        )
    if plan.is_warp:
        # The tier was selected here (`_tile_moves` dropped the choice layer), so an unbindable
        # atom is a per-atom refusal with a message.
        base, f16acc = state.atoms.get(id(node)) or ((), ())
        return None if plan.atom.name in (*base, *f16acc) else _warp_refusal(state, node, plan.atom)
    return None


def _reduce_moves(state: _State, node, key: str | None) -> list[ReducePlan]:
    """The reduce partitions this fold offers: the serial fold plus every :func:`coop_reduce_moves`
    band the node admits, or — under a ``REDUCE`` pin — the ONE partition that pin names, read
    against the kernel's pinned inventory (the ``coop`` token's width lives in ``WORK``). A pin is
    authoritative over the value; it cannot make a band this node has no geometry for legal, and
    one that names no legal partition raises the refusal instead of silently emptying the
    enumeration."""
    extent = hint_extent(node.axis)
    pin = _pin(REDUCE, key)
    if pin is None:
        return [ReducePlan(), *(p for p in coop_reduce_moves() if _band_refusal(node, p, extent, state.transposed_ok) is None)]
    try:
        plan = ReducePlan.parse(pin, state.work_pin)
    except ValueError as e:
        raise ValueError(f"REDUCE pin {pin!r} at {key or 'REDUCE'} does not resolve: {e}") from None
    why = _band_refusal(node, plan, extent, state.transposed_ok)
    if why is not None:
        raise ValueError(f"REDUCE pin {pin!r} at {key or 'REDUCE'} names no partition this fold can realize: {why}")
    return [plan]


# ---- the reduce partition: which bands this fold can carry ---------------------------------------- #


def _band_refusal(node, plan: ReducePlan, extent: int, transposed_ok: bool) -> str | None:
    """Why ``node`` cannot realize one reduce-partition candidate (``None`` when it can). Three
    facts about the node, and nothing about speed: a band wider than the axis has work for cannot
    fill its workers; a cross-CTA composite needs an axis it may split; and the TRANSPOSED band
    swaps the lane mapping so 32 lanes sweep the innermost FREE axis while each lane walks K
    serially — whole warps, an axis to sweep, and a per-cell epilogue for the swapped map to run
    (``transposed_ok``, the per-kernel half, precomputed once on :class:`_State`)."""
    if plan.coop > extent or plan.reg > extent:
        return f"the band is wider than the {extent}-element reduce axis has work for"
    if plan.needs_split and not _splittable(node, plan.cta):
        return f"the axis cannot carry a cross-CTA split of {plan.cta} (needs a static extent it divides, not already a slice)"
    if not plan.coop_transposed:
        return None
    if plan.coop % WARP_LANES:
        return f"the transposed coop band sweeps whole warps — coop={plan.coop} is not a multiple of {WARP_LANES}"
    if not transposed_ok:
        return "the transposed coop band needs an innermost free axis to sweep and a per-cell epilogue"
    return None


def _splittable(node, width: int) -> bool:
    """Whether ``node``'s axis can carry a cross-CTA split of ``width``: the σ-reindex reconstructs
    an absolute k from ``ksplit·(K/w) + kslice``, a bijection only over a STATIC extent the width
    divides — and an axis that is ITSELF a partition slice already spent its split, so a second one
    would halve the same stream again."""
    ext = node.axis.extent
    if not ext.is_static or ext.as_static() % width:
        return False
    return node.axis.window is None or not node.axis.window.partition


def _inner_free(tile: TileOp):
    """The innermost NON-UNIT free axis — a synthesized unit axis can sit innermost, and it is not
    the axis the transposed emitter sweeps."""
    free = tile.place.free
    return next((a for a in reversed(free) if not (a.extent.is_static and a.extent.as_static() == 1)), None)


# ---- the warp (tensor-core) tile: which atoms this contraction's fragments can bind ---------------- #


def _bindable_atoms(tile: TileOp, ctx, node, tail: list) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The tensor-core atoms ``node``'s fragments can BIND, as the ``(f32-accumulate,
    f16-accumulate)`` family pair — legality only, named by how each family is built rather than
    re-derived from operand widths (whether the f16-accumulate half is OFFERED is
    :func:`_tile_moves`' policy; a pin bypasses policy, never legality). Computed once per
    warp-applicable node by :func:`schedule`'s prescan (:func:`_node_refusal` has already passed):
    every condition is a fact about the node — the gmem addressing its fragment loaders and its
    fragment store must read — so it answers the same for every point of the tile space an atom
    opens."""
    dtype = edge_dtypes(node.a, tile.inputs)[0]
    a_step = gmem_axis_step(node.a, node.axis.name, tile.inputs) if isinstance(node.a, Load) else None
    shapes = {**tile.inputs, **tile.outputs}

    def bindable(names: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            n for n in names if _atom_refusal(ATOM_REGISTRY[n], a_step, isinstance(node.a, Load), tail, tile.place.free, shapes) is None
        )

    return bindable(atoms_for(dtype, ctx=ctx)), bindable(atoms_for(dtype, acc=dtype, ctx=ctx))


def _node_refusal(tile: TileOp, ctx, node, frag_ok: bool) -> str | None:
    """Why the node's algebra and operand dtypes select NO warp tier, atom-independent (``None``
    when a tier is selected) — the CHOICE layer, which is also the layer a pin drops on."""
    ring = node.semiring
    # The mma atom realizes ONLY the (·, +) semiring instance — the bilinear reading is
    # semiring-generic, so any other registered instance takes the scalar / reduce tiers rather
    # than silently reaching a tensor core that sums products.
    if ring is None or tuple(o.name for o in ring) != ("multiply", "add"):
        return "the mma atom realizes only the (multiply, add) semiring instance"
    if not tile.inputs:
        return "no typed inputs to read the operand dtypes from"
    # An (m, n) pair is what a fragment tiles; a rank-<2 grid supplies none at any site depth.
    if len(tile.place.free) < 2:
        return "the grid supplies no (m, n) output pair for a fragment to tile"
    if not frag_ok:
        return "the projection epilogue is not a per-fragment straight-line program"
    # A Fold operand edge — a computed cone (axis None) or a nested scheduling site — has no value
    # a byte-moving fragment loader can read: it realizes only through the shared-memory compute
    # fill, which is a ``STAGE`` and not on offer.
    if any(isinstance(e, Fold) for e in (node.a, *(ch.b for ch in node.channels))):
        return "a Fold operand edge realizes only through the shared-memory compute fill, which is not on offer"
    dtype = edge_dtypes(node.a, tile.inputs)[0]
    if dtype is None:
        return "the A operand's dtype cannot be read"
    if dtype.nbytes == 1:
        return "the fp8 gmem-direct tier is not restored yet"
    if not (atoms_for(dtype, ctx=ctx) or atoms_for(dtype, acc=dtype, ctx=ctx)):
        return f"no tensor-core atom takes a {dtype} multiplicand on this target"
    return None


def _atom_refusal(atom: AtomKind, a_step, a_is_load: bool, tail: list, free: tuple, shapes: dict) -> str | None:
    """Why ``atom`` cannot bind an otherwise warp-applicable contraction (``None`` when it can) —
    the per-atom half of the legality: the fragment loaders' contiguous-K addressing and the
    fragment store's split-pair addressability, both functions of the atom's own cell shape."""
    if a_is_load and (a_step is None or a_step[0] != 1 or (a_step[1] != 0 and a_step[1] % atom.atom_k)):
        # The loader advances one element at a time within an atom_k-wide row; a blocked index is
        # still representable when every fragment stays inside one contiguous run.
        motion = "unknown" if a_step is None else f"{a_step[0]} elements per column"
        return (
            f"warp TILE: A fragment loaders read {atom.atom_k} contraction columns CONTIGUOUSLY, but this "
            f"operand's gmem index moves {motion}; drop the atom token to use the scalar tier."
        )
    return _split_store_refusal(tail, free, atom.shape, shapes)


def _split_store_refusal(tail: list, free: tuple, atom_shape: tuple, shapes: dict) -> str | None:
    """Why an mma FRAGMENT store cannot address some buffer ``tail`` reads or writes (``None`` when
    it can address them all). The fragment epilogue addresses the output (and each epilogue load)
    per ATOM: the cell base is evaluated once at the atom origin and the lanes add ``col`` /
    ``row · ldm``. A re-fused split axis spells its coordinate across two buffer dims
    (``[…, f/Q, …, f%Q]``), addressable only under :func:`~…lowering._addr.split_addressable`;
    otherwise the scalar tiers, which evaluate every element's index, are the kernel's tiers."""
    roles = [(free[-1].name, atom_shape[1], "n", True)]
    if len(free) >= 2:
        roles.append((free[-2].name, atom_shape[0], "m", False))
    for s in tail:
        if not isinstance(s, (Write, Load)):
            continue
        buf = s.output if isinstance(s, Write) else s.input
        shape = getattr(shapes.get(buf), "shape", None)
        for name, ext, role, trailing in roles:
            if not split_addressable(s.index, shape, name, ext, trailing):
                return f"warp TILE: the {role} axis reaches {buf} as a split dim pair the fragment store cannot address"
    return None


def _fragment_epilogue_ok(tail: list) -> bool:
    """Whether the kernel's projection epilogue is a per-fragment straight-line program. The mma
    store folds the projection into a fragment epilogue whose leaf ``Load``\\ s are evaluated
    independently per fragment element, so a load whose INDEX reads a name an earlier epilogue stmt
    defined (an embedding gather) cannot be threaded through it, and neither can an output sweep."""
    defs: set[str] = set()
    for s in tail:
        if isinstance(s, Loop):
            return False
        if isinstance(s, Load) and {v for e in s.index for v in e.free_vars()} & defs:
            return False
        defs.update(s.defines())
    return True


def _warp_refusal(state: _State, node, atom: AtomKind) -> str:
    """Why a PINNED ``atom`` is not bindable on ``node`` — the same chain :func:`_bindable_atoms`
    filtered under, re-asked for its message. Failure path only, and only where the node's tier
    was SELECTED (:func:`_tile_moves` drops the choice layer first), so recomputing the per-kernel
    facts here is free and :func:`_node_refusal` need not be re-asked."""
    tile, ctx = state.tile, state.ctx
    tail = projection_tail(tile)
    if not atom.available_on(ctx):
        cc = ctx.compute_capability
        return f"atom {atom.name} requires target feature {atom.target_feature}, which is unavailable on sm_{cc[0]}{cc[1]}"
    dtype = edge_dtypes(node.a, tile.inputs)[0]
    if atom.operand_dtype("a") != dtype:
        return f"atom {atom.name} takes a {atom.operand_dtype('a')} A operand but this contraction's A is {dtype}"
    a_step = gmem_axis_step(node.a, node.axis.name, tile.inputs) if isinstance(node.a, Load) else None
    why = _atom_refusal(atom, a_step, isinstance(node.a, Load), tail, tile.place.free, {**tile.inputs, **tile.outputs})
    return why if why is not None else f"atom {atom.name} does not bind this contraction's fragments"


# ---- what the whole kernel has agreed -------------------------------------------------------------- #


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
    """The per-kernel constants every node of the fork tree shares — the term, its keying
    structure, and the per-kernel FACTS the prescan computed once (a branch expansion re-asks
    ``_options`` per node, so anything constant across the walk lives here, not in the guards)."""

    tile: TileOp
    sched: Sched
    ctx: object  # the compile Context — which mma instruction families the target has
    name: str
    knobs: dict
    off: dict  # every slice key the tree spells, at the codec families' declared OFF (the empty spelling)
    atoms: dict  # id(node) -> the (f32-accumulate, f16-accumulate) bindable atom family pair
    transposed_ok: bool  # the transposed coop band's per-kernel half (an axis to sweep, a per-cell epilogue)
    f16acc: bool = False  # whether the f16-accumulate family is OFFERED (precision policy, resolved once)
    work_pin: Workers | None = None  # the parsed EMMY_WORK pin — a FACT, read once, compared as Workers
    work_pinned: bool = False

    def honors_work_pin(self, work: Workers | None) -> bool:
        """Whether ``work`` is the inventory the live ``WORK`` pin named (vacuously true unpinned).
        Compared as parsed :class:`Workers`, never as spellings — ``t16x1`` and ``t16`` are one
        inventory."""
        return not self.work_pinned or work == self.work_pin


def _off(sched: Sched, root) -> dict:
    """Every slice key the stored tree spells, at the codec families' declared OFF — the empty
    spelling (the per-cell tile / serial fold / no intermediate). A row is the kernel's WHOLE
    identity, so a family the walk decided nowhere is spelled decided-empty rather than left
    absent — otherwise two rows of one kernel would carry different family vocabularies. A family
    with no keyed site keeps its bare key, for the same reason."""
    out: dict[str, str] = {}
    for family in SLICE_FAMILIES:
        keys = [key for node in _nodes(root) if (key := sched.key(family, node)) is not None]
        out.update(dict.fromkeys(keys or [family], ""))
    return out


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
        offers = [(o, below) for o in _options(state, node) if (below := ctx.extend(o)) is not None]
        if not offers:
            return []  # nothing schedules under here
        children = _kids(node) + rest
        if len(offers) == 1:
            option, ctx = offers[0]
            knobs, stack = _spelled(knobs, option, ctx), children
            continue  # a level with one option is no choice at all — collapse it
        return [_Branch(state, children, below, _spelled(knobs, o, below)) for o, below in offers]
    if not state.honors_work_pin(ctx.work):
        return []  # the walk finished without ever claiming the pinned inventory
    return [_Leaf(state, {**state.off, **knobs, WORK.name: ctx.work.spell() if ctx.work is not None else ""})]


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
    nothing schedules, which is the guardrail contract that leaves the term unmapped. A live SITE
    pin that names nothing raises out of the prescan instead."""
    sched = Sched(tile.op, {}, place=tile.place.on_grid())
    # The per-kernel FACTS, computed once: the projection tail and what it permits (the fragment
    # epilogue, the transposed band's sweep + per-cell conditions), the bindable atoms per
    # contraction node, and the parsed WORK pin — the ONE read of that env var.
    tail = projection_tail(tile)
    frag_ok = _fragment_epilogue_ok(tail)
    transposed_ok = _inner_free(tile) is not None and not any(isinstance(s, Loop) for s in tail) and not has_contraction_tail(tail)
    # ``atoms`` carries one entry per node whose algebra and operand dtypes SELECT a warp tier —
    # each a (f32-accumulate, f16-accumulate) family pair, possibly both empty when a tier was
    # selected but addressing refuses every atom (a strided A): the entry's presence is the choice
    # layer and its members the per-atom one, which is exactly the two layers a pin's refusal
    # splits on.
    atoms = {
        id(node): _bindable_atoms(tile, ctx, node, tail)
        for node in _nodes(tile.op)
        if _is_warp_site(node) and _node_refusal(tile, ctx, node, frag_ok) is None
    }
    raw = WORK.raw()
    work_pin = Workers.parse(raw) if raw is not None else None
    if work_pin is not None and work_pin.producer:
        raise ValueError(f"WORK pin {raw!r}: the +p producer band is not on offer in this prototype")
    state = _State(
        tile,
        sched,
        ctx,
        name,
        knobs,
        _off(sched, tile.op),
        atoms,
        transposed_ok,
        f16acc=bool(precision_pin(F16_MMA_F32_ACC)),
        work_pin=work_pin,
        work_pinned=raw is not None,
    )
    # A node that offers nothing offers it under EVERY context — options are a function of the node
    # and the pins alone — so one pass over the tree says whether the term has any schedule at all.
    # It is also what keeps a lazy branch honest: past this check every node still has an option
    # that composes with anything (the per-cell tile, the serial fold), so no branch can expand to
    # nothing and promise leaves it does not have. The one exception is a ``WORK`` pin, which is
    # kernel-global and can only be answered at the leaf.
    if any(not _options(state, node) for node in _nodes(tile.op)):
        return []
    # ``S_``-prefixed — not a schedule family, so tile identity and prefix-consistency are
    # untouched; it prices "a scalar tile where tensor cores were on offer". It is read off the
    # sites' own bindable atoms, never off the rows — a pin naming the scalar tier cannot erase it —
    # and rides the row PREFIX so fork rows and the materialized op (what ``realized_knobs`` reads)
    # carry the one signature.
    warp = any(any(pair) for pair in atoms.values())
    options = _step(state, (tile.op,), Ctx(), {"S_warp_eligible": 1.0} if warp else {})
    if len(options) == 1 and options[0].is_leaf:
        return options[0].expand()[0]
    return options


__all__ = ["Ctx", "schedule"]
