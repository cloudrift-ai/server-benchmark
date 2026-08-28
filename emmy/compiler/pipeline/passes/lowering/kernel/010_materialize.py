"""Materialize a ``TileOp``'s schedule into a ``KernelOp``.

Binds the schedule's grid axes to GPU threads and realizes the reduce partition through the **one**
node-kind dispatcher, ``_factor.factorize`` — every ``TileOp`` root (a tiled contraction, a
cooperative / ILP reduce, or a pointwise / scalar cell) lowers through that single emitter, which
reads the node kind + role + reduce plan off ``tile.op`` and picks the tier:

- **Tiled ``CONTRACTION``** (warp / register tile) — ``factorize`` synthesizes its bare
  grid-``Write`` (needs ``root.output``, so it can't ride the node) and expands it (mma → the
  ``RegFragment`` / ``LdmatrixLoad`` / ``MmaSyncPtx`` / ``RegStore`` fragment soup; scalar → the
  per-thread register cell tile) through the shared four-level tiling layer (in ``_factor``).
- **Reduce tier** — a ``PLANAR`` / ``TWISTED`` reduce (or a non-output-tiled ``CONTRACTION``) whose
  ``ReducePlan`` carries a BLOCK ``coop`` and/or a REG ``reg`` stage: the reduce axis is partitioned
  ``coop`` ways across the CTA's threads (cooperation) and ``reg`` ways across per-thread register
  accumulators (ILP), then a REG-tree fold, the cross-thread combine, and the projection.
- **Scalar tier** — one thread per output cell (``op.lower()`` + an output-store glue).

The op tree + ``lower`` are shared across kinds; only the schedule's partition changes — the
article's "schedule separate from combine" thesis. The tier machinery all lives in ``_factor``.
"""

from __future__ import annotations

from emmy.compiler.graph import Node
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.stmt import Body
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.ir.tile.ops import UnbindableProjection, reduce_plan
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.lowering.kernel._factor import factorize

PATTERN = [Pattern("root", TileOp)]


def rewrite(match: Match, root: Node) -> KernelOp | None:
    tile: TileOp = root.op
    # By the kernel pass no schedule slice can carry a cross-CTA ``GRID`` stage: the split is the
    # structural ``tile/035_split_reduce`` fork's, decided BEFORE scheduling (the walk's catalog
    # offers no ``g`` row, and its pin path strips the consumed ``g`` half). A surviving split
    # request is a bug — the materializer only lowers single-launch kernels.
    rplan = reduce_plan(tile) if tile.op is not None else None
    assert rplan is None or not rplan.needs_split, "materialize: a GRID split stage reached the kernel pass past 035_split_reduce"
    try:
        return KernelOp(body=Body((factorize(tile, root),)), name=tile.name)
    except UnbindableProjection as exc:
        # The offered row has no multi-root binding (e.g. it tiles two contraction operands of a
        # projection whose outputs do not partition by root). The row stays OFFERED — the
        # realization corpus pins that — and the compile declines it here: the skip is recorded,
        # the node stays a TileOp, and the greedy blocklist retry resolves onto the next row.
        raise RuleSkipped(f"kernel binder refuses this row's projection ownership: {exc}", reject=True) from exc
