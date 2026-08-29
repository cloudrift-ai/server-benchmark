r"""Schedule a lifted ``TileOp`` by walking its stored Fold tree.

One recursive generator IS the enumeration. A Fold offers its own options; each option extends the
:class:`Ctx` of what the kernel has already agreed, and the subtree below is walked under that
extended context. Siblings thread left to right, so a choice anywhere restricts everything
enumerated after it::

    S(node, ctx) = for each option o of node under ctx:  o x S(children(node), ctx + o)

There is no product over a flat site list and no join afterwards. The reasons two sites are not
one kernel — one worker inventory, agreeing tile geometry on a shared physical axis, one decision
per Fold however many paths reach it, and a compatible fragment seam across a producer/consumer
edge — are stated once, in :meth:`Ctx.extend`, and applied while descending, so an illegal
combination is never built. Traversal order is the fork order:
``WORK`` leads because the root owns the free axes it is read off, and the site keys follow as the
walk decides them.

It offers the whole reduce-partition catalog — on plain folds AND on the per-cell contraction
tier, whose K folds cooperatively / across ILP register chains through the same moves — both
contraction tiers (the scalar output tile and the tensor-core warp tile, the fp8 (k32) family
included), the whole ``STAGE`` transport family (the smem compute fill, the synchronous copy,
cp.async, TMA, and the ``+p`` producer band riding a resolved TMA stage), the pointwise register
strip (a ``TILE`` value on the root map cell, materialized as a term variant), and the
kernel-global ``RASTER`` launch-order swizzle (decided once per kernel, like ``WORK``), and the
walk reaches DERIVED sites (flash's synthesized PV contraction). The cross-CTA ``GRID`` split is
NOT a row here: it changes the kernel SET, so it is the structural ``035_split_reduce`` fork's —
the walk only CONSUMES a pin's ``g<n>[a|k]`` half on a kernel that already realized its split.

Every :func:`schedule` call enumerates its own prescan — the per-node option lists are a pure
function of the term and the live pins, and nothing is memoized across kernels or trajectories
(the retired session pool memo cached them keyed by hints, pins, samples, split receipts and
the spelled vocabulary — a growing side-channel of every fact identity rightly excludes, retired
as a bug surface rather than re-guarded). Under ``ctx.pool_sample`` (``emmy fit``'s offline
dataset build) the walk's leaf stream is reservoir-sampled instead of returned lazy.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from functools import lru_cache
from types import MappingProxyType

from emmy.compiler.dim import DEFAULT_SEQ_HINT, Dim
from emmy.compiler.ir.atom import ATOM_REGISTRY, AtomKind, atoms_for
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.pure.fold import Fold, deep_reads, edge_refs_axis, is_contraction
from emmy.compiler.ir.schedule import (
    Level,
    Raster,
    ReducePlan,
    Stage,
    TilePlan,
    WarpSpec,
    Workers,
    derive_inventory,
    plan_workers,
    resolve_site_tile,
)
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Stmt, Write
from emmy.compiler.ir.stmt.passes import has_contraction_tail
from emmy.compiler.ir.tile import OutputSpec, Placement, TileOp
from emmy.compiler.ir.tile.ops import Sched, carries_partition, cone_seam, edge_dtypes, projection_tail, scheduled
from emmy.compiler.ir.tile.path import SLICE_FAMILIES, sites
from emmy.compiler.pipeline.fork import Fork, iter_leaves
from emmy.compiler.pipeline.knob import axis_of, schedule_pin_fingerprint
from emmy.compiler.pipeline.passes.lowering._addr import gmem_axis_step, split_addressable
from emmy.compiler.pipeline.passes.lowering.tile import _staging as staging
from emmy.compiler.pipeline.passes.lowering.tile._tree import children, walk
from emmy.compiler.pipeline.search.space import (
    F16_MMA_F32_ACC,
    FP8_MMA,
    MAX_BLOCK_THREADS,
    MAX_REGISTERS_PER_CTA,
    MAX_REGISTERS_PER_THREAD,
    RASTER,
    REDUCE,
    STAGE,
    TILE,
    WARP_LANES,
    WORK,
    coop_reduce_moves,
    map_tile_moves,
    precision_pin,
    raster_moves,
    scalar_tile_moves,
    stage_moves,
    warp_tile_moves,
)
from emmy.compiler.structural import digest

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
    nothing and composes with any), the placed tile the rest of the kernel must agree with, and the
    fragment-seam entries it stakes (``(role, edge key, value)`` triples — see :class:`Ctx`).

    Fully immutable — the knob dict is sealed at construction — because option lists are shared
    by every branch of one kernel's walk: a walk reads options, it never writes one."""

    knobs: Mapping
    work: Workers | None = None
    tile: TilePlan | None = None
    seam: tuple = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "knobs", MappingProxyType(dict(self.knobs)))


# ---- what one Fold can spell ---------------------------------------------------------------------- #


class PinRefused(ValueError):
    """A live pin names nothing THIS kernel can realize — a different kernel set may. The class
    is the distinction ``040_schedule``'s defer keys on: a kernel-dependent refusal defers to the
    placement fork while a cuttable seam remains, whereas a malformed / nowhere-realizable pin (a
    codec parse failure, an ``m``-strip no catalog spells) stays a plain ``ValueError`` and raises
    immediately — deferring it would waste a cut and repeat an already-correct message per piece."""


def _pin(knob, key: str | None) -> str | None:
    """The live env pin addressing ``key`` — ``EMMY_KNOBS``'s ``FAMILY@<element>`` entry, falling
    back to the bare ``EMMY_<FAMILY>``. Unset reads ``None``, which is the distinction the
    enumeration keys on: an unset family offers its catalog, a set one is authoritative."""
    if key is None:
        return None
    element = axis_of(key)
    return knob.narrow_at(element) if element else knob.raw()


def _supports_scalar(node) -> bool:
    """Whether the scalar atom can carry this contraction: inline operand cones are evaluated
    directly by the scalar register tile, but a multi-channel product needs one accumulator family
    per channel, which is a warp compute-fill form."""
    return len(node.channels) == 1


def _computed_edge(node) -> bool:
    """Whether either operand role is an inline zero-axis cone — the smem compute fill's operand
    form. A nonzero-axis Fold edge is a nested scheduling site, not a scalar producer evaluated at
    each contraction cell (the choice layer refuses it, :func:`_node_refusal`)."""
    return any(isinstance(e, Fold) and e.axis is None for e in (node.a, *(ch.b for ch in node.channels)))


def _needs_fill(state: _State, node, plan: TilePlan) -> bool:
    """Whether this warp candidate's operands take the MANDATORY smem compute fill — a computed
    edge, a multi-channel product, or a materialized ``a`` the fill must convert. The ONE predicate
    every fill dispatch reads (the stage options, the pin raise, re-materialization), so they
    cannot drift."""
    return _computed_edge(node) or len(node.channels) > 1 or (plan.is_warp and staging.converting_a(node, plan.atom, state.tile.inputs))


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
    the named plan cannot realize on it.

    A pure function of the node and the live pins, so :func:`schedule`'s prescan computes it ONCE
    per node onto ``_State.options`` (a per-kernel FACT) and the walk reads the memo — a branch
    expansion re-asks per node, and re-resolving every stage there multiplied the cost."""
    sched = state.sched
    if not isinstance(node, Fold) or node.axis is None:
        return _strip_options(state, node)  # the root map cell's register strip; else no decision
    if is_contraction(node):
        return _claimable(state, _contraction_options(state, node))
    key = sched.key("REDUCE", node)
    opts = [_Option({key: p.spell()} if key else {}, derive_inventory((), coop=p.coop)) for p in _reduce_moves(state, node, key)]
    return _claimable(state, opts)


def _contraction_options(state: _State, node) -> list[_Option]:
    """The contraction's options: the tile × stage × reduce legal product, each with the
    producer-band inventory variants the resolved stage can drive (the reduce partition rides the
    per-cell tier — :func:`_contraction_reduces`). The transport is RESOLVED here, at option
    construction — the smem budget is per-site (a slab either fits ``ctx.max_dynamic_smem`` or the
    option is not offered), so an option carries its sized :class:`Stage` and materialization can
    only re-derive the same one. A ``STAGE`` pin that resolves on no tile the site offers raises
    the recorded refusal; one whose tier the plan does not select (the per-cell tile, the scalar
    tier's inline cone) drops that plan instead of silently spelling gmem-direct."""
    sched = state.sched
    facts = state.facts[id(node)]
    key = sched.key("TILE", node)
    stage_key = sched.key("STAGE", node)
    red_key = sched.key("REDUCE", node)
    stage_pin = _pin(STAGE, stage_key)
    # Whether the STAGE pin addressed THIS site (``STAGE@<element>``) or is the bare graph-wide
    # fallback — the distinction the fill's transport arm drops or raises on.
    stage_scoped = stage_pin is not None and (element := axis_of(stage_key)) is not None and STAGE.pin_at(element) is not None
    tile_pin = _pin(TILE, key)
    red_pin = _pin(REDUCE, red_key)
    opts: list[_Option] = []
    refused: list[str] = []
    tile_refused: list[str] = []
    red_refused: list[str] = []
    for plan in _tile_moves(state, node, key):
        placed = sched.placed(node, plan)  # bound ONCE per plan — every per-plan check below reads this binding
        if plan.is_tiled and (placed is None or placed.axes is None):
            continue  # a tile the grid cannot bind to an (m, n) pair has no geometry to realize
        why = _plan_node_refusal(state, node, plan, placed)
        if why is not None:
            tile_refused.append(why)
            continue
        reduces = _contraction_reduces(state, node, red_key, plan.is_tiled)
        if not reduces:
            # A pinned cooperative / ILP partition is the per-cell tier's; this tiled plan offers
            # nothing under it. Recorded like the tile/stage refusals: REDUCE has no choice of
            # tier, so a pin every plan drops must RAISE at the empty-offer site, never leave the
            # term silently unmapped (a multi-channel product has no per-cell tile at all).
            red_refused.append(
                f"REDUCE pin {red_pin!r} at {red_key or 'REDUCE'} names a cooperative / ILP partition, which only the "
                f"per-cell tile realizes — the tiled {plan.spell() or 'scalar'} tile contracts K serially per register cell"
            )
            continue
        for stage in _stage_options(state, node, plan, placed, stage_pin, stage_scoped, refused):
            # The ADDITIVE producer/consumer bound: a compute fill keeps the consuming fragments
            # live while it builds one scheduled producer block, so the pair's registers sum.
            why = _paired_budget_refusal(node, facts.producer, placed, stage)
            if why is not None:
                refused.append(why)
                if tile_pin is not None:
                    # PinRefused: the bound exists because the PAIR's registers sum — cutting the
                    # producer into its own kernel removes the pairing, so another set may realize.
                    raise PinRefused(why)
                continue
            knobs = {}
            if key is not None:
                knobs[key] = plan.spell()
            if stage_key is not None:
                knobs[stage_key] = stage.spell() if stage is not None else ""
            tile = placed if plan.is_tiled else None
            seam = _seam_entries(state, node, key, plan, placed, stage)
            for red in reduces:
                # The K partition claims the kernel's inventory exactly like a plain fold's band —
                # reconciled through the ONE rule (a coop band IS the t<coop> thread inventory,
                # only ever offered on the untiled tier, so nothing here can disagree).
                work = derive_inventory((plan,), coop=red.coop)
                red_knobs = {**knobs, red_key: red.spell()} if red_key is not None else knobs
                opts.append(_Option(red_knobs, work, tile, seam))
                opts.extend(
                    _Option(red_knobs, replace(work, producer=band), tile, seam)
                    for band in _producer_bands(work, stage, plan.block_threads)
                )
    if not opts:
        if tile_pin is not None and tile_refused:
            raise PinRefused(f"TILE pin {tile_pin!r} at {key or 'TILE'} names no schedule this site can realize: {tile_refused[-1]}")
        if red_pin and red_refused:
            # PinRefused: the coop/ILP partition rides the per-cell tier, which turns on the plans
            # THIS node offers — a cut that leaves a single-channel contraction restores it.
            raise PinRefused(red_refused[-1])
        if stage_pin and refused:
            key_name = stage_key or "STAGE"
            raise PinRefused(f"STAGE pin {stage_pin!r} at {key_name} names no stage this contraction can realize: {refused[-1]}")
    return opts


def _resolve_stage(state: _State, node, plan: TilePlan, placed: TilePlan, want: Stage, why: list[str] | None = None) -> Stage | None:
    """The ONE resolve dispatch — enumeration and the leaf's re-resolution (:func:`_stage_of`) both
    take it, chosen by the same predicate (:func:`_needs_fill`), so the materialized slice always
    reproduces the one the row identity was built from. The fill branch reads only ``want.depth``
    (the fill's transport is fixed); the warp / scalar branches resolve the full spelling."""
    budget = state.ctx.max_dynamic_smem
    if plan.is_warp and _needs_fill(state, node, plan):
        facts = state.facts[id(node)]
        seam, k_axis, producer = facts.seam, facts.k_axis, facts.producer
        return staging.resolve_fill_stage(
            node, placed, budget, want.depth, inputs=state.tile.inputs, why=why, seam=seam, k_axis=k_axis, producer=producer
        )
    if plan.is_warp:
        return staging.resolve_warp_stage(node, placed, want, budget, state.tile.inputs)
    return staging.resolve_scalar_stage(node, placed, want, state.tile.inputs, budget)


def _stage_options(
    state: _State, node, plan: TilePlan, placed: TilePlan, pin: str | None, scoped: bool, refused: list[str]
) -> list[Stage | None]:
    """The RESOLVED operand stages one tile candidate offers — gmem-direct ``None`` first, then
    every catalog move that resolves against the node under this plan (deduped on the resolved
    spelling: a depth that clamps under the smem budget spells identically to its shallower
    sibling and must yield ONE row). A fill-needing warp plan takes :func:`_fill_options` instead
    — the fill is mandatory and has no gmem-direct sibling. A pinned ``STAGE`` is authoritative:
    it resolves exactly, or the plan is dropped with the refusal recorded (the caller raises when
    no plan realizes the pin); a pinned EMPTY spelling is gmem-direct, also authoritative."""
    if not plan.is_tiled or (not plan.is_warp and _computed_edge(node)):
        # No stage TIER exists here — the per-cell tile has no operand slab, and the scalar atom
        # evaluates inline cones directly in its register row/column reads (the warp-only compute
        # fill is unnecessary, and byte transports cannot evaluate a cone). A live pin fans out to
        # a tier these plans cannot mean, so it DROPS here (the choice layer) and the plan keeps
        # its no-intermediate form.
        if pin:
            logger.debug("STAGE pin %r dropped: this plan has no operand slab to stage", pin)
        return [None]
    if plan.is_warp and _needs_fill(state, node, plan):
        return _fill_options(state, node, plan, placed, pin, scoped, refused)
    if pin is not None:
        if not pin:
            return [None]  # pinned gmem-direct
        want = Stage.parse(pin)  # a malformed pin RAISES here, loudly
        why = staging.stage_target(want, state.ctx)
        if why is not None:
            refused.append(why)
            return []
        r = _resolve_stage(state, node, plan, placed, want)
        if r is None:
            refused.append(f"pinned STAGE {pin!r} does not resolve for this contraction")
            return []
        return [r]
    out: list[Stage | None] = [None]
    spelled = {""}
    for move in stage_moves(warp=plan.is_warp, ctx=state.ctx):  # target-filtered in the catalog (a pin RAISES instead)
        r = _resolve_stage(state, node, plan, placed, move)
        if r is not None and r.spell() not in spelled:
            spelled.add(r.spell())
            out.append(r)
    return out


def _fill_options(
    state: _State, node, plan: TilePlan, placed: TilePlan, pin: str | None, scoped: bool, refused: list[str]
) -> list[Stage | None]:
    """The RESOLVED smem compute-fill stages a computed operand, multi-channel product or
    converting materialized ``a`` offers — its depths, and nothing else: the fill is MANDATORY
    (no gmem-direct sibling, no byte transport can evaluate a cone or carry several B/C channels),
    so a ``STAGE`` pin can only choose the depth. ``d1`` and the asynchronous-peer prefetch ring
    ``d2`` are fork siblings, measured per shape; a ``d2`` that clamps back under the smem budget
    spells identically and dedupes to one row. A SCOPED pin naming a byte transport RAISES — the
    fill's tier is selected here by construction, so an addressed refusal is never silent — while
    a BARE byte-transport pin DROPS (the choice layer: it fans out to every STAGE site, and this
    one has no byte-transport tier for it to mean). A pinned depth the budget refuses on THIS
    plan's slabs is recorded like the family's other per-plan refusals; the caller raises when no
    plan honors the pin."""
    depths = [1, 2]
    if pin:
        # A pinned spelling names a kernel, so its TRANSPORT cannot be quietly dropped and read as
        # depth alone: the fill's own asynchronous B-slab prefetch ring is the depth-2 ``smem``
        # row, not ``smem-async``.
        want = Stage.parse(pin)
        if want.transport != "smem":
            reason = (
                f"the smem compute fill has no {want.transport} sibling: a computed operand cannot ride a byte "
                f"transport (nothing but the fill can evaluate a producer cone). Its own asynchronous B-slab "
                f"prefetch ring is spelled d2/smem."
            )
            if scoped:
                raise ValueError(reason)
            # The choice-layer drop: a bare pin fans out to every STAGE site of the graph, and the
            # mandatory fill is a tier a byte-transport spelling cannot mean — refusing here would
            # fail whole-graph sweeps on any kernel that hosts a fill. The fill keeps its own
            # catalog; a pin that means THIS site spells it scoped and still raises above.
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("STAGE pin %r dropped at the compute fill: %s", pin, reason)
            pin = None
        else:
            depths = [want.depth]
    out: list[Stage | None] = []
    spelled: set[str] = set()
    for depth in depths:
        why: list[str] = []
        r = _resolve_stage(state, node, plan, placed, Stage(depth=depth), why=why)
        if r is None:  # per DECLINED depth, so a pin that fits no depth names the gate it hit
            reason = f"the smem compute fill does not resolve at depth {depth}: " + (
                why[-1] if why else f"its slabs must fit the {state.ctx.max_dynamic_smem} B smem budget"
            )
            if pin:
                # Plan-scoped: the slab is sized by THIS tile's geometry, so the pinned depth may
                # still resolve on a sibling plan — record the refusal and raise only when no plan
                # honors the pin (the caller's rule, shared with the TILE / transport refusals).
                refused.append(reason)
            else:
                logger.debug("%s", reason)
            continue
        if r.spell() not in spelled:
            spelled.add(r.spell())
            out.append(r)
    return out


# ---- the producer band: the +p inventory a resolved stage can drive ------------------------------ #


def _band_transport_refusal(stage: Stage | None) -> str | None:
    """What a producer band can actually drive: a RESOLVED TMA stage (the band arms the box-copy
    mbarrier ring — cp.async's wait-group is issuing-thread-scoped and a smem compute fill has no
    async load half)."""
    if stage is None or stage.transport != "smem-tma":
        return "a producer band drives a resolved TMA stage; this row has none"
    return None


def _band_budget_refusal(band: int, block_threads: int) -> str | None:
    """A dedicated producer band adds ``32·p`` threads ON TOP of the compute warps. Two budgets:
    the total fits the CTA limit, and the band does not outnumber the compute half."""
    aux = WARP_LANES * band
    if aux > block_threads:
        return f"producer band {aux} threads outnumbers the {block_threads} compute threads"
    if block_threads + aux > MAX_BLOCK_THREADS:
        return f"producer band {aux} + {block_threads} compute exceeds the {MAX_BLOCK_THREADS}-thread/CTA limit"
    return None


def _producer_bands(work: Workers | None, stage: Stage | None, block_threads: int) -> tuple[int, ...]:
    """The producer-band widths an option ALSO claims as inventory variants. The band is
    kernel-global, but every condition on it is a fact about the OPTION: it drives a resolved TMA
    stage and needs a warp inventory wide enough to spare it. Claiming it here is what makes the
    old "no band beside a synchronous compute fill" gate fall out: a fill stage is not TMA, so a
    fill option claims no band and :meth:`Ctx.extend` finds no partner — which also makes an
    unclaimable ``+p`` WORK pin a leaf-level refusal, so the drops are explained at debug level
    like the family's other choice-layer drops."""
    if work is None or work.kind != "warp":
        return ()
    why = _band_transport_refusal(stage)
    if why is not None:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("producer band not offered: %s", why)
        return ()
    out = []
    for band in (1, 2):
        why = _band_budget_refusal(band, block_threads)
        if why is None:
            out.append(band)
        elif logger.isEnabledFor(logging.DEBUG):
            logger.debug("producer band +p%d not offered: %s", band, why)
    return tuple(out)


# ---- the fragment seam: the producer/consumer cross-site rule ------------------------------------ #


def _seam_entries(state: _State, node, key: str | None, plan: TilePlan, placed: TilePlan, stage: Stage | None) -> tuple:
    """The fragment-seam stakes this option carries — an OFFER when the node produces a fragment
    operand for another contraction, a NEED when it consumes one — as ``(role, edge key, value)``
    triples :meth:`Ctx.extend` reconciles. Both are spelled off the option alone; the cross-site
    check lives in the context, whichever endpoint the walk decides first."""
    out = []
    if key is not None and key in state.frag_producers:
        if not plan.is_tiled:
            offer = ("free",)  # an untiled producer is evaluated elementwise into the sync slab
        elif plan.is_warp:
            offer = ("warp", plan.atom.shape, plan.atom.fragment_layout, placed.n.units, placed.n.tile)
        else:
            offer = ("scalar",)
        out.append(("offer", key, offer))
    facts = state.facts[id(node)]
    if facts.need is not None:
        if plan.is_tiled and plan.is_warp and stage is not None and stage.transport == "smem":
            kind = "step" if facts.need_step else "warp"
            need = (kind, plan.atom.shape, plan.atom.fragment_layout, stage.bk_elems)
        else:
            need = ("free",)
        out.append(("need", facts.need, need))
    return tuple(out)


def _frag_regs(atom: AtomKind, role: str) -> int:
    """The exact per-lane register count of one emitted mma fragment."""
    explicit = atom.fragment_nregs(role)
    if explicit is not None:
        return explicit
    m, n, k = atom.ptx_shape
    dtype = atom.operand_dtype(role)
    if role == "a":
        return m * k * dtype.nbytes // 128
    if role == "b":
        return n * k * dtype.nbytes // 128
    return m * n // (64 if dtype.nbytes == 2 else 32)


def _paired_fragment_registers(node, producer, tile: TilePlan, stage: Stage | None) -> tuple[int, int] | None:
    """``(required, available)`` peak registers/lane for two composed contractions.

    A computed fill keeps the consuming fragments live while it builds one scheduled producer
    block through the same mma atom. Count the exact ``RegFragment`` families emitted by
    ``_MmaOps.state`` for both contractions. This is a lower bound: scalar carrier state and
    address temporaries are intentionally absent, so the check rejects only rows whose fragments
    alone cannot fit the CTA register file."""
    if not (tile.is_warp and stage is not None and producer is not None):
        return None
    atom = tile.atom
    if stage.bk_elems % atom.atom_n:
        return None  # the producer fragment block does not realize this geometry
    a_regs, b_regs, c_regs = _frag_regs(atom, "a"), _frag_regs(atom, "b"), _frag_regs(atom, "c")
    # The f16-accumulate atom keeps an additional f32 shadow C family.
    if atom.operand_dtype("c").nbytes == 2:
        c_regs += atom.atom_m * atom.atom_n // 32
    depth = max(1, stage.reg_depth)
    channels = len(node.channels)
    outer_c = channels * tile.reg_m * tile.reg_n * c_regs
    outer = tile.reg_m * depth * a_regs + channels * (tile.reg_n * depth * b_regs + tile.reg_m * tile.reg_n * c_regs)
    producer_n = stage.bk_elems // atom.atom_n
    producer_regs = tile.reg_m * a_regs + len(producer.channels) * (producer_n * b_regs + tile.reg_m * producer_n * c_regs)
    available = min(MAX_REGISTERS_PER_THREAD, MAX_REGISTERS_PER_CTA // tile.block_threads)
    # Consumer A/B are first loaded in the drain after the producer block. Only the initialized
    # consumer C fragments span both regions; the two A/B families may reuse registers.
    return max(outer, outer_c + producer_regs), available


def _paired_budget_refusal(node, producer, tile: TilePlan, stage: Stage | None) -> str | None:
    """Why coexisting producer/consumer mma fragments exceed the CTA register-file envelope
    (``None`` when they fit) — the fragment seam's ADDITIVE bound. Not cross-site: the producer's
    fragment block is a function of the consumer's own stage (``bk_elems``), so the option builder
    checks it where the option is built."""
    counts = _paired_fragment_registers(node, producer, tile, stage)
    if counts is None or counts[0] <= counts[1]:
        return None
    required, available = counts
    return (
        f"paired contractions require at least {required} live fragment registers/thread, over the "
        f"{available}-register envelope at {tile.block_threads} threads/CTA"
    )


def _seam_ok(need: tuple, offer: tuple) -> bool:
    """Whether a consumer's fragment NEED composes with a producer's OFFER across one fragment
    edge. An untiled producer composes with a nested-cone need (the compute fill re-evaluates the
    cone elementwise into its slab) but NOT with a sibling-step one (``"step"``): a sibling's
    per-step result exists only in the enclosing carrier's stream, and the fragment fill has no
    way to re-evaluate it — its fragments must come from a warp-scheduled producer. A TILED
    producer produces fragments, so it composes only with a warp consumer over an smem compute
    fill whose atom family matches and whose slab chunk the producer's single-unit N tile fills
    exactly."""
    if offer[0] == "free":
        return need[0] != "step"
    if need[0] not in ("warp", "step") or offer[0] != "warp":
        return False
    _, shape, layout, bk = need
    _, o_shape, o_layout, o_units_n, o_tile_n = offer
    return shape == o_shape and layout == o_layout and o_units_n == 1 and o_tile_n == bk


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
    facts = state.facts[id(node)]
    scalar = scalar_tile_moves() if _supports_scalar(node) else []
    pin = _pin(TILE, key)
    if pin is None:
        # ``facts.offered`` already carries the precision POLICY (the f16-accumulate and fp8
        # families are off by default — the precise pin is authoritative, else the ``FAST_MATH``
        # umbrella offers each family everywhere it is legal and the evidence ranks it per shape).
        # Per-plan NODE refusals (the fp8 K-step, the fill's cover) are the option builder's —
        # it binds the placed geometry once per plan and drops or raises there.
        return [*scalar, *(warp_tile_moves(facts.offered) if facts.offered else [])]
    if facts.warp_refusal is not None and _names_warp_atom(pin):
        # The choice-layer drop: no warp tier here, whatever the pin says. Explicable, not silent.
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("TILE pin %r at %s dropped: %s", pin, key or "TILE", facts.warp_refusal)
        return []
    if state.work_pinned:
        works = [state.work_pin]
    else:
        catalog = [*scalar, *(warp_tile_moves((*facts.offered, *facts.pin_only)) if facts.warp_refusal is None else [])]
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
        raise PinRefused(f"TILE pin {pin!r} at {key or 'TILE'} names no schedule this site can realize: {detail}")
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
        facts = state.facts[id(node)]
        if plan.atom.name not in (*facts.offered, *facts.pin_only):
            return _warp_refusal(state, node, plan.atom)
    elif plan.is_tiled and not _supports_scalar(node):
        return "the scalar emitter carries one accumulator channel; a multi-channel product is a warp compute-fill form"
    return None


def _plan_node_refusal(state: _State, node, plan: TilePlan, placed: TilePlan) -> str | None:
    """Why ``node`` cannot realize one (node × plan) candidate (``None`` when it can) — the facts
    that need the PLAN in hand and so cannot ride the per-atom prescan: the fp8 byte-gather K-step
    and the compute fill's cover / copy-dtype geometry. The option builder is the ONE caller — it
    binds ``placed`` once per plan, drops a catalog plan on the refusal and raises it for a pinned
    one, so the two arms share one statement of each rule."""
    if not plan.is_warp:
        return None
    facts = state.facts[id(node)]
    why = _kstep_refusal(facts.k_axis, plan)
    if why is not None:
        return why
    if not _needs_fill(state, node, plan):
        return None
    conv = staging.converting_a(node, plan.atom, state.tile.inputs)
    return staging.computed_operand_cover(node, placed, converting=conv, k_axis=facts.k_axis) or staging.computed_operand_copy_dtype(
        node, placed, state.tile.inputs, converting=conv
    )


def _kstep_refusal(k_axis: Axis, plan: TilePlan) -> str | None:
    """Whether this atom's fragment loaders can reach the contraction K. The warp K-loop steps by
    ``atom_k`` and zero-fills the overhanging half of its final fragment, so a K the step does not
    divide — static or symbolic — is masked and correct; a STAGED row's K-chunk divisibility is
    the stage resolvers' own rule, stated where the chunk width is. The fp8 atoms are the
    exception on both counts: their byte-gather fragment loaders have no masked-K zero-fill
    family, so they take an exact K — static, and tiled by the full K-step."""
    if not (plan.is_warp and plan.atom.operand_dtype("a").nbytes == 1):
        return None
    ext = k_axis.extent
    if not ext.is_static:
        return f"atom {plan.atom.name}: the fp8 byte-gather loaders have no masked-K zero-fill — a symbolic K stays off the fp8 tier"
    k, step = ext.as_static(), plan.atom.atom_k * plan.bk
    if k % step == 0:
        return None
    return (
        f"warp TILE K-step {step} (atom_k={plan.atom.atom_k}*bk={plan.bk}) does not divide the static "
        f"contraction K={k}, and atom {plan.atom.name}'s byte-gather loaders have no masked-K zero-fill; "
        f"pin a K that is a multiple of {step}, or drop the fp8 atom token."
    )


def _reduce_catalog(state: _State, extent: int) -> list[ReducePlan]:
    """The serial fold plus every cooperative / ILP band ``extent`` admits — the ONE catalog arm,
    shared by the plain fold and the contraction's per-cell tier (a contraction is a monoid with
    a ⊗ lift, so its K partitions through the same moves and the same filter)."""
    return [ReducePlan(), *(p for p in coop_reduce_moves() if _band_refusal(p, extent, state.transposed_ok) is None)]


def _reduce_moves(state: _State, node, key: str | None) -> list[ReducePlan]:
    """The reduce partitions this fold offers: the serial fold plus every :func:`coop_reduce_moves`
    band the node admits, or — under a ``REDUCE`` pin — the ONE partition that pin names, read
    against the kernel's pinned inventory (the ``coop`` token's width lives in ``WORK``). A pin is
    authoritative over the value; it cannot make a band this node has no geometry for legal, and
    one that names no legal partition raises the refusal instead of silently emptying the
    enumeration. Two pin exemptions, both restatements of where a decision lives:

    - The cross-CTA ``g<n>[a|k]`` half is the structural ``035_split_reduce`` fork's decision. It
      was realized by REPLACING the kernel it addressed, and the receipt is the sliced axis's
      partition window — kernel-scoped, so ONE pinned split means one split however many folds the
      pieces still carry. What reaches every fold of a piece is the rest of the row (``g2k/coop``
      on a split kernel is ``coop``); a ``g`` half on a kernel that realized no split raises.
    - The catalog's width filter (a band wider than the axis has work for) does not bind a pin: a
      pinned over-wide band idles its extra lanes and still realizes — the split's finalize takes
      the kernel's pinned inventory over a fold as narrow as the split width."""
    extent = node.axis.hint_extent
    pin = _pin(REDUCE, key)
    if node.observed:
        # An observer makes the stream order-visible: every partitioned combine — cooperative
        # band, ILP register partials, the cross-CTA split — changes which prefixes exist, so a
        # scan offers exactly the serial fold. A pin naming a partition DROPS to serial (the same
        # adaptation as the consumed ``g``-half strip: an ambient pin fans out to every kernel,
        # and refusing here would fail whole-model sweeps on any scan); a pinned replay still
        # fails loudly at the offered oracle, whose membership check sees the pin unsatisfied.
        if pin is not None and ReducePlan.parse(pin, state.work_pin).stages and logger.isEnabledFor(logging.DEBUG):
            logger.debug("REDUCE pin %r names a partition; an observed fold (a scan) realizes the serial fold only", pin)
        return [ReducePlan()]
    if pin is None:
        return _reduce_catalog(state, extent)
    return [_parsed_reduce_pin(state, pin, key)]


def _parsed_reduce_pin(state: _State, pin: str, key: str | None) -> ReducePlan:
    """The ONE partition a live ``REDUCE`` pin names, resolved against the kernel's pinned
    inventory (the ``coop`` token's width lives in ``WORK``) — shared by the plain-fold and
    contraction pin arms, so the split-receipt consumption and the transposed-band legality are
    stated once. A malformed pin, a ``g`` half on a kernel that realized no split, and a
    transposed band this kernel has no geometry for all RAISE the recorded refusal."""
    try:
        plan = ReducePlan.parse(pin, state.work_pin)
    except ValueError as e:
        # plain: a malformed spelling is wrong everywhere — no cut can change what the codec reads
        raise ValueError(f"REDUCE pin {pin!r} at {key or 'REDUCE'} does not resolve: {e}") from None
    if plan.needs_split:
        if not state.carries_partition:
            # plain: only a SPLIT mints the receipt this pin names; a cut never does
            raise ValueError(
                f"REDUCE pin {pin!r} at {key or 'REDUCE'} names a cross-CTA split, which only the structural "
                f"035_split_reduce fork realizes on a kernel's head fold — this kernel realized none"
            )
        plan = ReducePlan(tuple(st for st in plan.stages if st.level is not Level.GRID))
    why = _transposed_refusal(plan, state.transposed_ok)
    if why is not None:
        raise PinRefused(f"REDUCE pin {pin!r} at {key or 'REDUCE'} names no partition this fold can realize: {why}")
    return plan


def _contraction_reduces(state: _State, node, key: str | None, tiled: bool) -> list[ReducePlan]:
    """The reduce partitions ONE contraction tile candidate offers — the serial fold, plus (on the
    PER-CELL tier only) every cooperative / ILP band the static K admits: the coop reduce spec's
    contract is the non-output-tiled contraction (a tiled output contracts K serially per register
    cell), and its K partitions through the SAME :func:`coop_reduce_moves` catalog and
    :func:`_band_refusal` filter as a plain monoid fold — a contraction is a monoid with a ⊗ lift.
    K stays STATIC here (unlike the plain fold's hint-extent bound): the scalar contraction
    emitters carry no masked-K band. Under a ``REDUCE`` pin the shared pin arm resolves it
    (:func:`_parsed_reduce_pin`); a cooperative / ILP pin then reaches only the per-cell tier — a
    tiled plan offers nothing under it, and the option builder records that per-plan refusal and
    RAISES when no plan honors the pin (``REDUCE`` has no choice of tier, so there is no drop
    layer) — while a serial pin keeps every plan on the serial fold."""
    pin = _pin(REDUCE, key)
    if node.observed:
        # Same stream-order gate as the plain fold's (:func:`_reduce_moves`) — defensive here:
        # nothing builds an observed contraction yet.
        return [ReducePlan()]
    if pin is not None:
        pinned = _parsed_reduce_pin(state, pin, key)
        if pinned.coop > 1 or pinned.reg > 1:
            return [] if tiled else [pinned]
        return [ReducePlan()]
    ext = node.axis.extent
    if tiled or not ext.is_static:
        return [ReducePlan()]
    return _reduce_catalog(state, ext.as_static())


# ---- the reduce partition: which bands this fold can carry ---------------------------------------- #


def _band_refusal(plan: ReducePlan, extent: int, transposed_ok: bool) -> str | None:
    """Why one CATALOG reduce-partition candidate is not offered (``None`` when it is). Two
    different kinds of filter, named apart: the TRANSPOSED band's geometry is LEGALITY
    (:func:`_transposed_refusal` — the swapped lane map does not exist without it), while the
    width check is a BOUND ON THE ENUMERATED SPACE, not a legality — an over-wide band idles its
    extra lanes and still realizes, which is why the pin path exempts it and only this catalog arm
    applies it (a short axis would otherwise enumerate every band in the catalog to no effect).
    The catalog carries no cross-CTA stage — the ``GRID`` split changes the kernel set, so it is
    the structural ``035_split_reduce`` fork's catalog, not a row of this walk."""
    if plan.coop > extent or plan.reg > extent:
        return f"the band is wider than the {extent}-element reduce axis has work for"
    return _transposed_refusal(plan, transposed_ok)


def _transposed_refusal(plan: ReducePlan, transposed_ok: bool) -> str | None:
    """Why the TRANSPOSED band cannot realize here (``None`` for any non-transposed plan): it
    swaps the lane mapping so 32 lanes sweep the innermost FREE axis while each lane walks K
    serially — whole warps, an axis to sweep, and a per-cell epilogue for the swapped map to run
    (``transposed_ok``, the per-kernel half, precomputed once on :class:`_State`)."""
    if not plan.coop_transposed:
        return None
    if plan.coop % WARP_LANES:
        return f"the transposed coop band sweeps whole warps — coop={plan.coop} is not a multiple of {WARP_LANES}"
    if not transposed_ok:
        return "the transposed coop band needs an innermost free axis to sweep and a per-cell epilogue"
    return None


def _inner_free(tile: TileOp):
    """The innermost NON-UNIT free axis — a synthesized unit axis can sit innermost, and it is not
    the axis the transposed emitter sweeps."""
    free = tile.place.free
    return next((a for a in reversed(free) if not (a.extent.is_static and a.extent.as_static() == 1)), None)


# ---- the pointwise cell: the register strip ------------------------------------------------------ #


def _strip_extent(tile: TileOp) -> int:
    """The static inner free extent the pointwise register strip tiles — ``0`` when the cell does
    not admit the strip: a pure zero-axis root fold with no operands whose body is FLAT elementwise
    (per-cell ``Load`` / ``Assign`` + boundary root stores, no nested ``Loop`` / carried state),
    over a static innermost free axis."""
    op, place = tile.op, tile.place
    if not (isinstance(op, Fold) and op.axis is None and not op.operands) or not place.free:
        return 0
    if not place.free[-1].extent.is_static:
        return 0
    if not all(isinstance(s, (Load, Assign, Write)) for s in op.body) or any(st.sweep is not None for st in tile.output_specs):
        return 0
    return place.free[-1].extent.as_static()


def _strip_width(plan: TilePlan) -> int:
    """The strip ratio ``r`` a strip row's ``TILE`` names — the inner register width. A warp codec
    names none (there is no fragment on a pointwise cell), so it reads ``0`` and is dropped. An
    ``m`` half RAISES: :func:`~…search.space.map_tile_moves` never spells one, so only a pin can
    carry it, and silently reading ``f<n>x<m>`` as ``f<n>`` would honor a plan nobody offered."""
    if plan.is_warp:
        return 0
    if plan.reg_m > 1:
        raise ValueError(f"TILE {plan.spell()!r}: a pointwise cell has no m strip (the grid already parallelizes it); spell f<n>")
    return plan.reg_n


def _strip_refusal(extent: int, width: int) -> str | None:
    """Why one strip width cannot realize on the cell (``None`` when it can): the strip hands each
    thread ``width`` CONTIGUOUS inner-axis elements, so the width must tile the inner free extent.

    Not an unimplemented mask — MEASURED. The one form that masks the overhang without breaking the
    strip's flat shape slides the last cell back onto the final full run (``min(cell·width,
    extent − width)``, idempotent because the cell is a pure map), and that slid base is no longer a
    provably aligned affine form, so ``050_vectorize_loads`` / ``080_vectorize_stores`` decline —
    which is the only thing the strip exists to buy. On a V100, gelu over 65536×255 (no width tiles
    255): the flat per-cell map runs 158.5 µs while the slid ``f2`` / ``f4`` / ``f8`` strips run
    199.5 / 220.2 / 390.8 µs. The refused rows are strictly worse than the row that remains."""
    if width <= 1 or (extent and extent % width == 0):
        return None
    return f"register strip width {width} does not divide the inner free extent {extent}"


def _strip_options(state: _State, node) -> list[_Option]:
    """The register-strip options of a zero-axis fold — the flat per-cell tile and every ladder
    width the cell can carry, offered only where the codec keys ``TILE`` on it (the pure pointwise
    ROOT cell; any other per-cell projection decides nothing, but its children do). ``r`` IS the
    spelled ``TILE=f<r>`` — the strip is a TERM VARIANT applied at materialization, a function of
    the ROW. No option claims an inventory: the strip stays on the derived per-cell launch
    geometry. A ``TILE`` pin follows the family's two-layer rule: it DROPS where the cell has no
    strip tier for a graph-wide pin to mean (a warp atom, or a cell the strip does not admit at
    any width — a symbolic / swept / stateful inner), and RAISES where the tier applies and the
    named plan cannot realize (an indivisible width, an ``m`` half no catalog spells)."""
    key = state.sched.key("TILE", node) if isinstance(node, Fold) else None
    if key is None:
        return [_Option({})]
    ext = _strip_extent(state.tile)
    pin = _pin(TILE, key)
    if pin is None:
        opts = [_Option({key: ""})]  # the flat per-cell map, 1 elem/thread
        opts.extend(_Option({key: p.spell()}) for p in map_tile_moves() if _strip_refusal(ext, _strip_width(p)) is None)
        return opts
    if _names_warp_atom(pin):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("TILE pin %r at %s dropped: a pointwise cell has no warp tier", pin, key)
        return [_Option({key: ""})]
    if ext == 0:
        # The choice-layer drop: this cell admits no strip at any width, so the pin fans out to a
        # tier that does not exist here — the flat per-cell map is the one plan the cell has.
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("TILE pin %r at %s dropped: this cell admits no register strip (no static flat inner axis)", pin, key)
        return [_Option({key: ""})]
    plan = resolve_site_tile(pin, None)
    why = _strip_refusal(ext, _strip_width(plan))
    if why is not None:
        raise ValueError(f"TILE pin {pin!r} at {key} names no register strip this cell can realize: {why}")
    return [_Option({key: plan.spell()})]


def _strip_variant(state: _State, plan: TilePlan, row: dict) -> TileOp:
    """The pointwise register-STRIP term variant: hand each thread ``r`` CONTIGUOUS inner-axis
    elements. The inner free axis shrinks to ``extent/r`` (the grid walks it) and the cell body is
    unrolled ``r`` times — copy ``i`` reads/writes ``inner·r + i`` with its SSA names suffixed —
    then regrouped as ``r`` loads · ``r`` computes · ``r`` writes so the unit-stride runs feed
    ``050_vectorize_loads`` / ``080_vectorize_stores``. A different term, hence a different
    ``structural_key`` and ``identity_key(with_io=True, with_knobs=True)`` — which is why it is applied HERE and not at
    recognition."""
    tile = state.tile
    inner = tile.place.free[-1]
    r = plan.reg_n
    op = tile.op
    ssa: set[str] = set()
    for s in op.body:
        ssa.update(s.defines())
    loads: list[Stmt] = []
    computes: list[Stmt] = []
    stores: list[OutputSpec] = []
    for i in range(r):

        def rename(n: str, i: int = i) -> str:  # suffix only the body's SSA names; axis vars stay
            return f"{n}__u{i}" if n in ssa else n

        sigma = Sigma({inner.name: BinaryExpr("+", BinaryExpr("*", Var(inner.name), Literal(r, "int")), Literal(i, "int"))})
        for s in op.body:
            s2 = s.rewrite(rename, sigma)
            (loads if isinstance(s2, Load) else computes).append(s2)
        stores.extend(OutputSpec(write=st.write.rewrite(rename, sigma)) for st in tile.output_specs)
    new_inner = replace(inner, extent=Dim(inner.extent.as_static() // r))
    new_free = (*tile.place.free[:-1], new_inner)
    new_place = Placement(free=new_free, grid=new_free)
    return scheduled(
        Fold.projection(body=Body((*loads, *computes))),
        name=state.name,
        place=new_place,
        knobs={**state.knobs, **row},
        output_specs=tuple(stores),
    )


# ---- the warp (tensor-core) tile: which atoms this contraction's fragments can bind ---------------- #


def _channel_dtype(tile: TileOp, node, ctx):
    """The unambiguous tensor-core dtype supplied by the B channels, if any — the fallback the
    demoting / converting smem compute fill reads its atom family off when the A edge's own dtype
    selects none (a computed cone's f32 leaf, an erased ``.float()`` cast, flash's register P)."""
    dts = {edge_dtypes(ch.b, tile.inputs)[0] for ch in node.channels}
    if len(dts) == 1:
        return next(iter(dts))
    eligible = {dtype for dtype in dts if dtype is not None and atoms_for(dtype, ctx=ctx)}
    return next(iter(eligible)) if len(eligible) == 1 else None


def _atom_families(tile: TileOp, ctx, node, tail: list) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The tensor-core atoms ``node``'s fragments can BIND, split ``(offered, pin-only)``: the
    catalog enumerates ``offered``; ``pin-only`` holds the precision-POLICY-gated remainder a pin
    may still name (pins bypass policy, never legality). Two policies, each off by default and
    resolved through its own precise pin with the ``FAST_MATH`` umbrella behind it: the
    f16-accumulate siblings (``F16_MMA_F32_ACC``) and the native fp8 (k32) family (``FP8_MMA`` —
    its sm_89 hardware floor is absolute and lives in ``atoms_for``'s target filter, which no pin
    overrides). Computed once per warp-applicable node by :func:`schedule`'s prescan
    (:func:`_node_refusal` has already passed): every condition is a fact about the node, so it
    answers the same for every point of the tile space an atom opens."""
    dtype = edge_dtypes(node.a, tile.inputs)[0]
    a_is_load = isinstance(node.a, Load)
    a_step = gmem_axis_step(node.a, node.axis.name, tile.inputs) if a_is_load else None
    shapes = {**tile.inputs, **tile.outputs}

    def bindable(names: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(n for n in names if _atom_refusal(ATOM_REGISTRY[n], dtype, a_step, a_is_load, tail, tile.place.free, shapes) is None)

    if dtype is not None and dtype.nbytes == 1:
        atoms = bindable(atoms_for(dtype, ctx=ctx))
        return (atoms, ()) if precision_pin(FP8_MMA) else ((), atoms)
    ab = dtype if atoms_for(dtype, ctx=ctx) else _channel_dtype(tile, node, ctx)
    base = bindable(atoms_for(ab, ctx=ctx))
    f16acc = bindable(atoms_for(ab, acc=ab, ctx=ctx))
    return ((*base, *f16acc), ()) if precision_pin(F16_MMA_F32_ACC) else (base, f16acc)


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
    # A ZERO-axis Fold edge is a cone the smem compute fill evaluates into its slab. A
    # nonzero-axis edge is a nested scheduling site, which no operand TRANSPORT realizes — but the
    # MANDATORY multi-channel fill evaluates every non-materialized B channel into its slab, nested
    # reduce included (the streamed computed-B decode cone), so only the forms with no mandatory
    # fill to ride refuse: a nested A, and a nested B on a single-channel node.
    if isinstance(node.a, Fold) and node.a.axis is not None:
        return "a nested scheduling site inhabits the A edge; only a zero-axis cone rides the smem compute fill"
    if any(isinstance(ch.b, Fold) and ch.b.axis is not None for ch in node.channels) and len(node.channels) == 1:
        return "a nested scheduling site inhabits the B edge and no multi-channel fill is mandated to evaluate it"
    dtype = edge_dtypes(node.a, tile.inputs)[0]
    if dtype is not None and dtype.nbytes == 1:
        # The native fp8 (k32) family's STRUCTURAL requirements, which hold under any pin: the
        # byte-gather loaders move raw bits, so A must be a MATERIALIZED f8 load and every channel
        # must carry the SAME f8 dtype (a mismatched operand would be read at the wrong width);
        # the K-step rule is per-plan (:func:`_kstep_refusal`). Outside that, an f8 A has no warp
        # tier at all: the compute fill would DEMOTE the cone's value on a 1-byte slab store.
        if not isinstance(node.a, Load):
            return "the fp8 byte-gather loaders read a MATERIALIZED f8 A; the compute fill cannot store a cone at 1 byte"
        if _channel_dtype(tile, node, ctx) != dtype:
            return "the fp8 atoms read raw bytes at one width — every channel must carry the same f8 dtype as A"
        if not atoms_for(dtype, ctx=ctx):
            return f"no tensor-core atom takes a {dtype} multiplicand on this target"
        return None
    ab = dtype
    if not atoms_for(ab, ctx=ctx):
        # The demoting / converting compute fill: an A the atoms cannot bind (a computed cone's
        # f32 leaf, a plain materialized f32 load) still rides the CHANNELS' 16-bit atom — the
        # fill converts on the slab store, and stage resolution mandates the fill for these edges.
        ab = _channel_dtype(tile, node, ctx)
        if ab is not None and ab.nbytes == 1:
            return "an f8 channel under a demoting fill stays off the warp tier (the fill would demote to f8)"
    if ab is None:
        return "no operand dtype selects a tensor-core atom family"
    if not (atoms_for(ab, ctx=ctx) or atoms_for(ab, acc=ab, ctx=ctx)):
        return f"no tensor-core atom takes a {ab} multiplicand on this target"
    return None


def _atom_refusal(atom: AtomKind, a_dtype, a_step, a_is_load: bool, tail: list, free: tuple, shapes: dict) -> str | None:
    """Why ``atom`` cannot bind an otherwise warp-applicable contraction (``None`` when it can) —
    the per-atom half of the legality: the fragment loaders' contiguous-K addressing and the
    fragment store's split-pair addressability, both functions of the atom's own cell shape. A
    CONVERTING materialized A (a 2-byte-or-wider dtype the atom cannot bind) is exempt from the
    contiguity rule — its fill reads A per element through its own σ, never a fragment loader."""
    converting = a_is_load and a_dtype is not None and a_dtype.nbytes >= 2 and a_dtype != atom.operand_dtype("a")
    if a_is_load and not converting and (a_step is None or a_step[0] != 1 or (a_step[1] != 0 and a_step[1] % atom.atom_k)):
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


def _fold_states(op) -> frozenset[str]:
    """Every state name a Fold of the root term binds into the projection tail: the root's own
    results when it iterates, else the results of its Fold operands and body members."""
    if not isinstance(op, Fold):
        return frozenset()
    if op.axis is not None:
        return frozenset(op.defines())
    return frozenset(name for edge in (*op.operands, *op.body) if isinstance(edge, Fold) for name in edge.defines())


def _fragment_epilogue_ok(tail: list, states: frozenset[str]) -> bool:
    """Whether the kernel's projection epilogue is a per-fragment straight-line program. The mma
    store folds the projection into a fragment epilogue whose leaf ``Load``\\ s are evaluated
    independently per fragment element, so a load whose INDEX reads a name an earlier epilogue stmt
    defined (an embedding gather) cannot be threaded through it, and neither can an output sweep.
    Every ``Write`` must also read some fold state (``states``): the store rides accumulator
    fragments, so the fragment materializer has no per-cell store loop for an output whose backward
    cone reads none and fails to lower such a row — the choice layer refuses first, so every
    offered warp row realizes."""
    defs: set[str] = set()
    for s in tail:
        if isinstance(s, Loop):
            return False
        if isinstance(s, Load) and {v for e in s.index for v in e.free_vars()} & defs:
            return False
        defs.update(s.defines())
    body = Body(tail)
    return all(body.backward_cone(s.values).external_reads & states for s in tail if isinstance(s, Write))


def _warp_refusal(state: _State, node, atom: AtomKind) -> str:
    """Why a PINNED ``atom`` is not bindable on ``node`` — the same chain :func:`_atom_families`
    filtered under, re-asked for its message. Failure path only, and only where the node's tier
    was SELECTED (:func:`_tile_moves` drops the choice layer first), so recomputing the per-kernel
    facts here is free and :func:`_node_refusal` need not be re-asked."""
    tile, ctx = state.tile, state.ctx
    tail = projection_tail(tile)
    if not atom.available_on(ctx):
        cc = ctx.compute_capability
        return f"atom {atom.name} requires target feature {atom.target_feature}, which is unavailable on sm_{cc[0]}{cc[1]}"
    dtype = edge_dtypes(node.a, tile.inputs)[0]
    if dtype is not None and dtype != atom.operand_dtype("a") and (dtype.nbytes == 1 or atom.operand_dtype("a").nbytes == 1):
        # Byte movers cannot convert; a 2-byte-or-wider mismatch instead rides the converting fill.
        return f"atom {atom.name} takes a {atom.operand_dtype('a')} A operand but this contraction's A is {dtype}"
    a_step = gmem_axis_step(node.a, node.axis.name, tile.inputs) if isinstance(node.a, Load) else None
    why = _atom_refusal(atom, dtype, a_step, isinstance(node.a, Load), tail, tile.place.free, {**tile.inputs, **tile.outputs})
    return why if why is not None else f"atom {atom.name} does not bind this contraction's fragments"


# ---- the per-contraction FACTS the prescan computes once ----------------------------------------- #


@dataclass(frozen=True)
class _SiteFacts:
    """What one contraction node IS, read once by :func:`schedule`'s prescan — every field a fact
    about the stored node (and its live precision-policy pins), never a decision."""

    warp_refusal: str | None  # why the node selects no warp tier (``None`` when it does) — the choice layer
    offered: tuple[str, ...]  # tensor-core atoms the catalog enumerates (precision policy applied)
    pin_only: tuple[str, ...]  # bindable atoms only a pin names (the policy-gated remainder)
    k_axis: Axis  # the reduction domain — a derived unit marker inherits its enclosing fold's axis
    seam: tuple | None  # the computed-A stat-row seam, or a derived marker's carried-state seam
    producer: Fold | None  # the single contraction nested in the computed A edge (the paired budget)
    need: str | None  # the TILE key of this consumer's fragment producer (the cross-site seam)
    #: whether the needed fragment is a SIBLING step's value rather than a nested cone's: a nested
    #: cone the compute fill can re-evaluate elementwise, but a sibling's per-step result exists
    #: only in the enclosing carrier's stream, so its fragments must come from a warp-scheduled
    #: producer (:func:`_seam_ok` keys the free-offer rule on this).
    need_step: bool = False


def _site_facts(tile: TileOp, ctx, sched: Sched, tail: list, frag_ok: bool) -> dict:
    """One :class:`_SiteFacts` per contraction node of the stored tree.

    A DERIVED contraction with a unit marker axis (flash's synthesized PV) carries that axis merely
    to bind its result: its reduction domain is the ENCLOSING fold's axis, and its seam bridges the
    states the enclosing carrier streams (the running max / denominator) rather than a cone
    prologue — a parent/child interface fact, read here so no guard rewrites the stored tree.

    A fragment edge joins a consumer contraction to the ONE contraction that produces its computed
    fragment operand — nested in its A cone and varying with its K, or a sibling in the enclosing
    fold's derived step whose result the consumer's computed edges read (the same generic dataflow
    relation the fragment evaluator follows, no operation-family recognition)."""
    parents: dict[int, object] = {}
    for node in _nodes(tile.op):
        for child in _kids(node):
            parents.setdefault(id(child), node)
    derived_ids = {id(s.node) for s in sites(tile.op) if s.derived}
    sibling = _sibling_fragment_edges(tile.op, sched)
    out: dict[int, _SiteFacts] = {}
    for node in _nodes(tile.op):
        if not (isinstance(node, Fold) and node.axis is not None and is_contraction(node)):
            continue
        parent = parents.get(id(node))
        # The seam is the fill's stat-row interface and a function of the node, not of a tile
        # plan. Read it once for a warp-eligible computed A instead of lowering the entire cone
        # again for every plan. The derived marker is the one override: its states are the
        # ENCLOSING carrier's, which no cone read can see.
        k_axis, seam = node.axis, None
        refusal = _node_refusal(tile, ctx, node, frag_ok)
        if (
            id(node) in derived_ids
            and node.axis.extent.is_static
            and node.axis.extent.as_static() == 1
            and isinstance(parent, Fold)
            and parent.axis is not None
        ):
            k_axis = parent.axis
            seam = ((), (), tuple(parent.combine.results[: -len(node.combine.results)]))
        elif refusal is None and isinstance(node.a, Fold):
            seam = cone_seam(node.a, k_axis.name)
        producer = None
        if isinstance(node.a, Fold):
            nested = tuple(s.node for s in sites(node.a) if is_contraction(s.node) and edge_refs_axis(s.node, k_axis.name))
            producer = nested[0] if len(nested) == 1 else None
        need = sibling.get(id(node))
        step = need is not None
        if need is None and producer is not None:
            need = sched.key("TILE", producer)
        offered, pin_only = _atom_families(tile, ctx, node, tail) if refusal is None else ((), ())
        out[id(node)] = _SiteFacts(refusal, offered, pin_only, k_axis, seam, producer, need, step)
    return out


def _sibling_fragment_edges(root, sched: Sched) -> dict[int, str]:
    """``id(consumer) -> producer TILE key`` for the SIBLING fragment-edge form: a fold may compute
    a fragment operand through a sibling contraction in its derived step (flash's PV reading the
    score). The dependency is the backward cone of the consumer's computed edges — generic
    dataflow, and only an edge whose consumer's result the enclosing carrier accumulates."""
    out: dict[int, str] = {}
    for node in _nodes(root):
        if not (isinstance(node, Fold) and node.axis is not None) or is_contraction(node) or node.combine is None:
            continue
        steps = node.step_stmts()
        states = set(node.combine.results)
        for index, consumer in ((i, s) for i, s in enumerate(steps) if is_contraction(s)):
            accumulated = any(
                isinstance(stmt, Accum) and stmt.name in states and stmt.value in consumer.defines() for stmt in steps[index + 1 :]
            )
            reads = {name for edge in consumer.operands if isinstance(edge, Fold) for name in deep_reads(edge.lower())}
            if not accumulated or not reads:
                continue
            cone = Body(tuple(steps[:index])).backward_cone(reads)
            producers = tuple(stmt for stmt in cone.members if is_contraction(stmt))
            if len(producers) == 1 and (key := sched.key("TILE", producers[0])) is not None:
                out[id(consumer)] = key
    return out


# ---- what the whole kernel has agreed -------------------------------------------------------------- #


@dataclass(frozen=True)
class Ctx:
    """What the walk has already decided for the WHOLE kernel, carried down and across siblings.

    ``work`` — a kernel has ONE worker inventory. ``axes`` — two sites sharing a physical grid axis
    must give it the same tile and units. ``decided`` — one Fold reached by several paths is ONE
    decision, so a later path can only re-spell what the first chose. ``seam`` — a fragment edge
    joins two contractions the walk decides at different steps, so each endpoint's option records
    its stake under ``(role, producer key)`` — the producer an OFFER (its placed fragment
    interface), the consumer a NEED (what its fill's slab chunk requires) — and whichever side
    arrives second is reconciled against the first (:func:`_seam_ok`); a re-record must equal the
    first, the same one-decision rule ``decided`` states for spellings."""

    work: Workers | None = None
    axes: dict = field(default_factory=dict)
    decided: dict = field(default_factory=dict)
    seam: dict = field(default_factory=dict)

    def extend(self, option: _Option) -> Ctx | None:
        """This context with ``option`` folded in, or ``None`` when the option contradicts it.

        The hot inner loop of the walk (one call per option per branch level), so the untouched
        halves share the parent's maps instead of copying: ``decided`` / ``axes`` / ``seam`` are
        read only here and every mutation below is on a fresh copy, so sharing is safe."""
        decided = self.decided
        for k, v in option.knobs.items():
            if decided.get(k, v) != v:
                return None
        work = self.work
        if option.work is not None:
            if work is not None and work != option.work:
                return None
            work = option.work
        axes = self.axes
        if option.tile is not None:
            axes = dict(axes)
            for side in option.tile.mn:
                if axes.setdefault(side.axis.name, (side.tile, side.units)) != (side.tile, side.units):
                    return None
        seam = self.seam
        if option.seam:
            seam = dict(seam)
            for role, edge, value in option.seam:
                if seam.setdefault((role, edge), value) != value:
                    return None
                other = seam.get(("need" if role == "offer" else "offer", edge))
                if other is not None:
                    need, offer = (value, other) if role == "need" else (other, value)
                    if not _seam_ok(need, offer):
                        return None
        return Ctx(work, axes, {**decided, **option.knobs} if option.knobs else decided, seam)


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
    facts: dict  # id(node) -> :class:`_SiteFacts`, one per contraction node
    frag_producers: frozenset  # TILE keys of fragment-edge producers (the seam's offer side)
    transposed_ok: bool  # the transposed coop band's per-kernel half (an axis to sweep, a per-cell epilogue)
    #: whether this kernel already realized a cross-CTA split — the sliced axis's partition-window
    #: receipt, KERNEL-scoped. A ``REDUCE`` pin's ``g<n>[a|k]`` half is consumed against it: one
    #: pinned split means one split, however many folds the pieces still carry.
    carries_partition: bool = False
    work_pin: Workers | None = None  # the parsed EMMY_WORK pin — a FACT, read once, compared as Workers
    work_pinned: bool = False
    #: id(node) -> its option tuple, computed ONCE by :func:`schedule`'s prescan. Options are a
    #: pure function of the node and the live pins, so this is a per-kernel FACT the walk reads —
    #: a branch expansion re-asks per node, and re-resolving every stage there is pure waste.
    options: dict = field(default_factory=dict)
    #: The pool's minted STAMP — the variant key + hints + pins + the sample identity, carried by
    #: every Fork of the tree (:attr:`Fork.pool_id`). Not a cache key (nothing stores pools any
    #: more): the greedy decision memo keys picks on it and the budgeted descent seeds its
    #: deterministic draw from it, and both fail safe on any drift the stamp cannot see (a row
    #: that no longer decodes re-decides).
    pool_id: str = ""

    @property
    def pool_bound(self) -> int:
        """Upper bound on the pool's leaf count — Π over the per-node option tuples × the RASTER
        fan-out, before ``Ctx`` legality prunes (legality only shrinks). Derived from the prescan
        rather than stored, so it cannot go stale against ``options``; carried by every Fork of
        the tree (:attr:`Fork.pool_bound`) so a consumer can ask "roughly how large is this pool"
        without walking it — the greedy cold-pool budget trigger."""
        bound = len(_raster_values(self))
        for opts in self.options.values():
            bound *= len(opts)
        return bound

    @property
    def pool_descent_bound(self) -> int:
        """Upper bound on ``Ctx.extend`` calls in one complete random descent. ``_step`` checks
        every sibling at a level before choosing one, so the bound is the sum of option counts."""
        return sum(len(opts) for opts in self.options.values())

    def honors_work_pin(self, work: Workers | None) -> bool:
        """Whether ``work`` is the inventory the live ``WORK`` pin named (vacuously true unpinned).
        Compared as parsed :class:`Workers`, never as spellings — ``t16x1`` and ``t16`` are one
        inventory."""
        return not self.work_pinned or work == self.work_pin

    @property
    def observed(self) -> bool:
        """Whether this kernel's tree holds an observed fold — a scan. Its one row is the serial
        fold, so an ambient ``WORK``/``REDUCE`` pin fanning out over the graph cannot mean it: the
        leaf keeps the decided-empty row instead of dropping the kernel to unmapped."""
        return any(isinstance(s.node, Fold) and s.node.axis is not None and s.node.observed for s in self.sched._all_sites())


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
    # Kernel-global like WORK. The walk decides it once per kernel (:func:`_raster_values` — the
    # flat order on every kernel the swizzle cannot mean), and the OFF entry keeps the family in
    # every row's vocabulary for the same reason as the site families.
    out["RASTER"] = ""
    return out


def _raster_values(state: _State) -> tuple[str, ...]:
    """The ``RASTER`` candidates — kernel-global like ``WORK``, so they are decided ONCE per
    kernel as the walk's LEADING fork level (no ``Ctx`` reconciliation: nothing else can claim
    the launch order; a one-value level collapses like any other), and CONTRACTION-scoped: only
    a 2-D-tiled contraction grid decodes the swizzle (the ``grid_tile`` seal applies it where
    both ``(m, n)`` block axes exist). A symbolic-axis
    (masked-tile) grid renders through the dynamic decode path, which does not carry it, so
    offering ``gm8`` there would stamp a launch order the kernel doesn't realize — the flat
    ``""`` is the one honest value, and a live pin DROPS with the other choice-layer drops."""
    tile = state.tile
    eligible = any(isinstance(n, Fold) and is_contraction(n) for n in _nodes(tile.op)) and all(
        ax.extent.is_static for ax in tile.place.free
    )
    if eligible:
        values = tuple(RASTER.narrow(raster_moves()))
        for value in values:
            Raster.parse(value)  # a malformed pin RAISES here, loudly — narrow is authoritative, not a parser
        return values
    pin = RASTER.raw()
    if pin and logger.isEnabledFor(logging.DEBUG):
        logger.debug("RASTER pin %r dropped: no static 2-D contraction grid decodes the swizzle here", pin)
    return ("",)


@lru_cache(maxsize=1024)
def _work_spelling(work: Workers) -> str:
    """``Workers.spell`` memoized per (frozen, hashable) inventory — the walk re-spells the same
    few inventories once per branch level otherwise."""
    return work.spell()


def _spelled(knobs: dict, option: _Option, ctx: Ctx) -> dict:
    """The row prefix one decision leaves behind: what the option spells, plus the inventory as
    soon as any option claims it — :meth:`Ctx.extend` refuses a second one, so a prefix that
    carries ``WORK`` already carries its final value."""
    out = {**knobs, **option.knobs}
    if ctx.work is not None:
        out[WORK.name] = _work_spelling(ctx.work)
    return out


def _step(state: _State, stack: tuple, ctx: Ctx, knobs: dict) -> list[Fork]:
    """One level of the walk: descend past every FORCED decision, then return the siblings standing
    at the first real choice — or the leaf, when the stack runs out.

    ``stack`` is the walk's own work list. Popping a node and pushing its children is what makes
    this the same depth-first order a recursive generator would take, with the difference that the
    remainder is DATA, so a sibling can be resumed later instead of having to be produced now.

    The leaf's row ALWAYS spells the kernel-global ``WORK`` — empty when no option claimed an
    inventory — and that unconditional write is a stated invariant, not an accident: a complete
    schedule row carries ``WORK`` and a structural arm's knob delta (a cut, the cross-CTA split's
    ``g``-half or its unsplit receipt) never does, which is the one marker consumers use to tell
    the two apart (``search/golden_eval``)."""
    while stack:
        node, rest = stack[0], stack[1:]
        offers = [(o, below) for o in state.options[id(node)] if (below := ctx.extend(o)) is not None]
        if not offers:
            return []  # nothing schedules under here
        children = _kids(node) + rest
        if len(offers) == 1:
            option, ctx = offers[0]
            knobs, stack = _spelled(knobs, option, ctx), children
            continue  # a level with one option is no choice at all — collapse it
        return [_Branch(state, children, below, _spelled(knobs, o, below)) for o, below in offers]
    if not state.honors_work_pin(ctx.work) and not state.observed:
        return []  # the walk finished without ever claiming the pinned inventory
    return [_Leaf(state, {**state.off, **knobs, WORK.name: _work_spelling(ctx.work) if ctx.work is not None else ""})]


@dataclass(frozen=True)
class _WalkFork(Fork):
    """Every node of one schedule tree carries the pool's minted identity and size bounds — read
    off the shared :class:`_State` rather than stored per node."""

    state: _State

    @property
    def pool_id(self) -> str:
        return self.state.pool_id

    @property
    def pool_bound(self) -> int:
        return self.state.pool_bound

    @property
    def pool_descent_bound(self) -> int:
        return self.state.pool_descent_bound


@dataclass(frozen=True)
class _Branch(_WalkFork):
    """A partly-walked schedule: the nodes still to decide, the context they must honour, and the
    row prefix decided so far. The subtree does not exist until ``expand`` walks one level more."""

    stack: tuple
    ctx: Ctx
    knobs: dict
    is_leaf = False

    def expand(self) -> list[Fork]:
        return _step(self.state, self.stack, self.ctx, self.knobs)


@dataclass(frozen=True)
class _Leaf(_WalkFork):
    """A complete walk: ``knobs`` is the kernel's whole identity, materialized on demand."""

    knobs: dict
    is_leaf = True

    def expand(self) -> list[TileOp]:
        return [_materialize(self.state, self.knobs)]


def _stage_of(state: _State, node, plan: TilePlan, spec: str) -> Stage | None:
    """The row's ``STAGE`` re-resolved against the node, through the enumeration's own dispatch
    (:func:`_resolve_stage`), so this reproduces the slice the leaf identity was built from."""
    return _resolve_stage(state, node, plan, state.sched.placed(node, plan), Stage.parse(spec))


def _materialize(state: _State, row: dict) -> TileOp:
    """One row -> its ``TileOp``, every slice RE-RESOLVED from the row's own spellings over the same
    ``_nodes`` order the walk decided in. The row is the kernel's complete identity, so
    decode-by-spelling is what makes it replayable."""
    sched, tile = state.sched, state.tile
    work = Workers.parse(row.get(WORK.name) or None)
    root = tile.op
    if isinstance(root, Fold) and root.axis is None and not root.operands:
        # The register strip is a TERM VARIANT: a row whose root ``TILE`` names a width unrolls
        # the cell rather than decorating it with a slice.
        plan = resolve_site_tile(row.get(sched.key("TILE", root) or "") or None, work)
        if _strip_width(plan) > 1:
            return _strip_variant(state, plan, row)
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
            spec = row.get(sched.key("STAGE", node) or "") or ""
            if spec:
                slices.append(("STAGE", node, _stage_of(state, node, plan, spec)))
        else:
            # The per-cell tier's cooperative / ILP K partition rides a REDUCE slice, exactly as
            # a plain fold's does (a decided-empty spelling resolves to no slice).
            slices.append(("REDUCE", node, red if red.stages else None))
    workers = WarpSpec(work.producer) if work is not None and work.producer else None
    return scheduled(
        tile.op,
        name=state.name,
        place=sched.place,
        knobs={**state.knobs, **row},
        output_specs=tile.output_specs,
        slices=slices,
        workers=workers,
    )


def schedule(tile: TileOp, name: str, knobs: dict, ctx) -> list[Fork]:
    """Map a newly lifted, unmapped ``tile`` onto the grid and offer its scheduling fork.

    Returns the siblings at the walk's first real choice — each one lazy, holding a work list and a
    context rather than any row — a single leaf when the whole walk is forced (still a FORK: the
    engine records a one-option fork as a decision, which is what keys a fully pinned kernel's row
    into the trace and the evidence), or ``[]`` when nothing schedules, which is the guardrail
    contract that leaves the term unmapped. A live SITE pin that names nothing raises out of the
    prescan instead.

    Every call enumerates its own prescan — options are a pure function of the node and the
    live pins, and nothing is cached across kernels (see the module docstring for why the
    session pool memo was retired).

    Under ``ctx.pool_sample`` (``emmy fit``, never a deploy) the lazy fork is NOT returned:
    the walk's leaf stream is reservoir-sampled (:meth:`~…search.pool.PoolSample.take`), the pool's
    exact size is reported through ``sample.totals``, and the drawn rows come back as leaf forks."""
    sched = Sched(tile.op, {}, place=tile.place.on_grid())
    # The per-kernel FACTS, computed once: the projection tail and what it permits (the fragment
    # epilogue, the transposed band's sweep + per-cell conditions), the per-contraction facts
    # (atom families, reduction domain, seam, producer, fragment edges), and the parsed WORK pin —
    # the ONE read of that env var.
    tail = projection_tail(tile)
    frag_ok = _fragment_epilogue_ok(tail, _fold_states(tile.op))
    transposed_ok = _inner_free(tile) is not None and not any(isinstance(s, Loop) for s in tail) and not has_contraction_tail(tail)
    facts = _site_facts(tile, ctx, sched, tail, frag_ok)
    raw = WORK.raw()
    off = _off(sched, tile.op)
    # The IR receipt (the sliced axis's partition Window), or the flag receipt a piece with no
    # sliced axis carries (a realized split's independent projection sibling — ``split_consumed``):
    # both mean the kernel-set decision was consumed, so a ``REDUCE`` pin's ``g`` half strips.
    partition = carries_partition(tile.op) or tile.split_consumed
    sample = getattr(ctx, "pool_sample", None)
    # The pool STAMP (``_State.pool_id`` — see its comment): the variant key plus the situational
    # facts a replayed PICK must not cross (hints size the space; pins prune it; a sampled draw
    # must never seed a live one). Not a cache key — nothing stores pools.
    io = (*tile.inputs.values(), *tile.outputs.values())
    key = digest(
        tile.identity_key(with_io=True, with_knobs=True),
        tuple(d.hint or DEFAULT_SEQ_HINT for t in io for d in t.shape if not d.is_static),
        schedule_pin_fingerprint(),
        sample.key if sample is not None else "",
    )
    state = _State(
        tile,
        sched,
        ctx,
        name,
        knobs,
        off,
        facts,
        frozenset(f.need for f in facts.values() if f.need is not None),
        transposed_ok,
        carries_partition=partition,
        work_pin=Workers.parse(raw) if raw is not None else None,
        work_pinned=raw is not None,
        pool_id=key,
    )
    nodes = tuple(_nodes(tile.op))
    # A node that offers nothing offers it under EVERY context — options are a function of the
    # node and the pins alone — so one pass over the tree says whether the term has any
    # schedule at all, and that same pass IS the option table the walk reads (a site pin that
    # names nothing raises here, out of the prescan). It is also what keeps a lazy branch
    # honest: past this check every node still has an option that composes with anything (the
    # per-cell tile, the serial fold), so no branch can expand to nothing and promise leaves it
    # does not have. The exceptions are kernel-global: a ``WORK`` pin is answered at the leaf,
    # and a fragment seam can empty a sibling's offer mid-walk.
    state.options.update((id(node), tuple(_options(state, node))) for node in nodes)
    if any(not opts for opts in state.options.values()):
        return []
    # ``S_``-prefixed — not a schedule family, so tile identity and prefix-consistency are
    # untouched; it prices "a scalar tile where tensor cores were on offer". It is read off the
    # sites' own offered atoms, never off the rows — a pin naming the scalar tier cannot erase it —
    # and rides the row PREFIX so fork rows and the materialized op (what ``realized_knobs`` reads)
    # carry the one signature.
    warp = any(f.offered for f in facts.values())
    prefix = {"S_warp_eligible": 1.0} if warp else {}
    # The kernel-global RASTER LEADS the walk as its own fork level: one decision per kernel, no
    # cross-site agreement to thread through Ctx, so each candidate seeds the row prefix and the
    # whole site walk is one branch beneath it. A single-value level is collapsed (the walk runs
    # directly, like any other one-option level). The walk's aliveness is value-independent (RASTER
    # rides only the prefix, never the Ctx), so ONE probe under the first value states the
    # "[] when nothing schedules" guardrail for the whole fan-out: the walk must yield a first
    # branch AND that tree must hold a leaf. The per-node offer check above cannot promise the
    # latter — its own kernel-global exceptions (a ``WORK`` pin answered at the leaf, a fragment
    # seam emptying a sibling's offer mid-walk) can kill every leaf, and a fork with no leaf breaks
    # the guardrail for every consumer. A live probe costs one extra leftmost-spine expansion; a
    # dead one drains the whole tree before answering — the accepted price of the guardrail.
    values = _raster_values(state)
    forks = _step(state, (tile.op,), Ctx(), {**prefix, "RASTER": values[0]})
    if forks and next(iter_leaves(forks), None) is None:
        return []
    if forks and len(values) > 1:
        forks = [_Branch(state, (tile.op,), Ctx(), {**prefix, "RASTER": value}) for value in values]
    if sample is None:
        return forks
    drawn = sample.take(dict(leaf.knobs) for leaf in iter_leaves(forks))
    sample.totals[key] = drawn.total
    return [_Leaf(state, dict(row)) for row in drawn.rows]


__all__ = ["Ctx", "schedule"]
