"""Schedule a lifted (UNMAPPED) ``TileOp``: map its free axes onto the grid and offer the
scheduling fork — the second half of the Loop-IR → Tile-IR boundary.

``010_lift`` is purely structural: it reads the algebra off a ``LoopOp`` and emits an UNMAPPED
:class:`~emmy.compiler.ir.tile.ir.TileOp` (its ``op`` set, ``place`` carrying just the free axes).
THIS rule picks that up and decides the schedule — the free-axis → grid mapping plus the per-node
``TILE`` / ``REDUCE`` / ``STAGE`` / ``WORK`` / ``RASTER`` families through the classic model.

The fixed candidate-space contract is Algorithm 1(c, p, t): the schedule restriction, problem, and target form one
immutable context over independently projected kernel, node, and edge domains. The generic traversal never unpacks
that context. Its composition may reject a prefix only when the combined state proves there is no completion, and
traversal order cannot change membership.

Splitting the two halves is what makes the fork ONE thing: a kernel reaches scheduling by
several routes — the ordinary lift and a cross-CTA split's partial and finalize — and all converge here. The engine restarts its
rule scan after every functional rewrite, so a ``TileOp`` this pass's ``010`` just emitted is
matched here on the next sweep, and so is every unmapped ``TileOp`` a structural rewrite minted.
That is exactly why none of them needs a special case: each arrives as a kernel with no schedule,
like any other, and this rule cannot tell them apart.

Empty enumeration remains a skip rather than a guessed schedule.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.schedule.classic import ClassicScheduleCodec, ClassicScheduleContext
from emmy.compiler.ir.schedule.classic_projection import (
    ClassicProjectionError,
    materialize_classic,
    project_classic,
)
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.ir.tile.ops import carries_partition
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.fork import Fork

# NOTE: no ``Knob`` objects (``TILE`` / ``REDUCE`` / ``STAGE``) may be imported here — ``Pass.load``
# scans rule modules for ``Knob`` attrs and OFF-fills any it finds bare onto every variant of the
# pass. Pin reads / knob-key spelling ride the enumerator's helpers instead; the family NAMES below
# are plain strings and a function, which that scan does not see.
from emmy.compiler.pipeline.knob import STRUCT_PREFIX, family_pins, schedule_pin_fingerprint
from emmy.compiler.pipeline.schedule import fork_schedule
from emmy.compiler.structural import digest

PATTERN = [Pattern("root", TileOp)]


def block_problems(tile: TileOp, ctx) -> list[TileOp]:
    """One enumerable problem per candidate block form of this kernel.

    Blocking splits a reduce axis into ``k_o × k_i`` and re-associates the fold over the two levels.
    It does not change what the kernel computes, which is why it lives HERE rather than in a pass of
    its own: the kernel the pipeline stamps, identifies, pools and prices is the unblocked one, and
    each block form is another shape of the same problem to enumerate over — like a tile size, not
    like a cut.

    Only a TWISTED carrier is offered a block, because it is the only carrier a block gives
    anything. A contraction's block is already spelled — ``bk`` says how many atom K-steps one inner
    step consumes and the materializer chunks K by it — and a plain reduction's partition is
    ``REDUCE``'s, with the cross-CTA split already factoring the axis. A twisted carrier's ⊕ is a
    rescaling program, so no site inside it is bilinear at all; blocking separates the two monoids
    and CREATES the site, and the row's ``TILE`` there spells the width, so no codec family is added.

    The unblocked form leads, so a cold walk descends what the kernel already was.
    """
    from emmy.compiler.ir.schedule.classic_projection import block_widths  # noqa: PLC0415
    from emmy.compiler.pipeline.passes.lowering.tile._block import block_tree, blockable_streams, is_blocked  # noqa: PLC0415

    if is_blocked(tile.axes):
        return [tile]
    streams = blockable_streams(tile.op, tile.axes)
    rows: list[dict[int, int]] = [{}]
    for node, axis in streams:
        widths = block_widths(tile, ctx, node, axis)
        rows = [{**row, **extra} for row in rows for extra in ({}, *({id(node): width} for width in widths))]
    problems = [tile]
    for row in rows:
        if not row:
            continue
        op, axes = block_tree(tile.op, tile.axes, row)
        problems.append(replace(tile, op=op, axes=axes))
    return problems


def classic_forks(tile: TileOp, name: str, knobs: dict, ctx, identity: str | None = None) -> list[Fork]:
    """Adapt the classic semantic enumeration to the pipeline's lazy search tree.

    ``identity`` is the KERNEL's identity key when the problem is one block form of it — every form
    computes the same thing, so they share one pool and one recorded identity, and only the row
    tells them apart."""
    from emmy.compiler.pipeline.search.space import F16_MMA_F32_ACC, FP8_MMA, precision_pin  # noqa: PLC0415

    try:
        domains = project_classic(tile, ctx)
    except ClassicProjectionError:
        return []
    context = ClassicScheduleContext(tile, ctx, domains).restrict(
        {family: family_pins(family) for family in ("WORK", "TILE", "REDUCE", "STAGE", "RASTER")},
        split_consumed=carries_partition(tile) or tile.split_consumed,
        allow_f16_accumulate=precision_pin(F16_MMA_F32_ACC) is True,
        allow_fp8=precision_pin(FP8_MMA) is True,
        validate_pins=ctx.validate_pins,
    )
    codec = ClassicScheduleCodec(context)
    pool_id = digest(
        identity if identity is not None else (tile.identity_key(with_io=True) or ""),
        ctx.structural_key(),
        tuple((axis.name, repr(axis.extent)) for axis in tile.place.free),
        codec.keys(),
        schedule_pin_fingerprint(),
        tile.split_consumed,
    )
    prefix = {"S_warp_eligible": 1.0} if any(choice.tile.is_warp for choices in domains.nodes.values() for choice in choices) else {}
    descent_bound = len(domains.kernel) + sum(
        len(choices) * max((len(domains.edges[edge]) for edge in domains.edges if edge[0] == site), default=1)
        for site, choices in domains.nodes.items()
    )
    return fork_schedule(
        context,
        codec=codec,
        inherited_knobs=knobs,
        row_prefix=prefix,
        materialize=lambda assignment, row: materialize_classic(
            tile,
            name=name,
            knobs=row,
            target=ctx,
            assignment=assignment,
        ),
        pool_id=pool_id,
        pool_bound=domains.product_size,
        pool_descent_bound=descent_bound,
        sample=getattr(ctx, "pool_sample", None),
    )


def rewrite(match: Match, root: Node, ctx=None) -> Fork | list[TileOp] | TileOp:
    del match  # the scheduled op replaces the matched node in place — no graph surgery here
    tile: TileOp = root.op
    if tile.op is None or tile.place.is_mapped:
        raise RuleSkipped("TileOp already scheduled / nothing to map")
    # This pass DECIDES, so it requires the kernel's identity. Every row it enumerates carries the
    # ``S_*`` stamp forward, and that is what the prior ranks on, what a recorded golden matches by,
    # and what the measurement is later filed under — decide without it and the fork's pick is made
    # against an empty signature that matches every kernel and identifies none. the ``IdentityStrategy`` stamps at birth
    # ahead of this rule for exactly that reason, so an unstamped kernel here is a pass-order
    # break, not a case to handle.
    assert any(k.startswith(STRUCT_PREFIX) for k in tile.knobs), (
        f"{tile.name!r}: scheduling a kernel with no structural identity — the IdentityStrategy stamps at birth"
    )
    identity = tile.identity_key(with_io=True) or ""
    options = [
        fork for problem in block_problems(tile, ctx) for fork in classic_forks(problem, tile.name, tile.knobs, ctx, identity)
    ]
    if not options:
        raise RuleSkipped("no enumerable schedule row for this term — leave it unmapped")
    return options if len(options) > 1 else options[0]
