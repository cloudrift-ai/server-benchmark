"""Schedule a recognized (UNMAPPED) ``TileOp``: map its free axes onto the grid and offer the
scheduling forks — the second half of the Loop-IR → Tile-IR boundary.

``010_recognize`` is purely structural: it reads the algebra off a ``LoopOp`` body and emits an
UNMAPPED :class:`~emmy.compiler.ir.tile.ir.TileOp` (its ``op`` set, ``place`` carrying just the free
axes) into the graph. THIS rule picks that up and decides the schedule — the free-axis → grid
mapping plus the per-node ``TILE`` / ``REDUCE`` / ``STAGE`` / ``WORK`` / ``RASTER`` forks — through
the ``_schedule`` helper, dispatched on the axes' ``AxisRole`` (never on a kernel kind).

Splitting the two halves is what makes the fork ONE thing: a kernel reaches scheduling by three
routes — the ordinary lift, flash's graph rewrite (``try_flash`` fuses a softmax-then-P@V pair into
a fresh ``TileOp`` node; a graph fragment cannot embed a scheduling fork), and a placement cut's
re-recognized pieces — and all three now converge here instead of the first two riding a re-entry
arm inside recognition. The engine restarts its rule scan after every functional rewrite
(``Cursor.advance``), so a ``TileOp`` this pass's ``010`` just emitted is matched here on the next
sweep, exactly as ``030_split_reduce`` already matches the ``TileOp``\\ s this rule produces.

Two things beyond the plain ``schedule()`` call live here, because both are decisions ABOUT the
schedule rather than about the structure:

1. **The MONOID-producer merge** — a lifted ``Map(source=Fold)`` whose body is a statistic epilogue
   plus ⊗-folds over one shared A value (the fused norm→linear edge ``rmsnorm(x)·nw @ w``; its
   N-channel form the gate/up MLP edge) ALSO nodifies to a computed-A ``Contraction``
   (``_atomize.bind_prologue_contraction``). Both readings of the SAME loop are scheduled and their
   candidates merged into ONE fork: the reduce-``Map`` rows first (option-0 stays the conservative
   coop pick, lowerable everywhere), then the ``Contraction`` form's warp (mma) rows over the
   ``sync`` compute-fill stage. Each form carries the OTHER's family keys as decided-empty stamps
   (:func:`prologue_knob_bases`) so every leaf row spells the same key set — the evidence pick's
   prefix-consistency. A warp ``TILE`` pin is authoritative and keeps the ``Contraction`` rows
   alone: offering the reduce sibling would let cold greedy pick past the pin.
2. **The PLANAR demote fallback** — a computed-A ``Contraction`` can have NO legal row on any tier
   (fp32, no dtype-eligible atom, bad geometry), and an empty option list is a SILENT no-op to the
   engine, so the node would leak as a ``LoopOp`` all the way to ``plan_from_graph``. Demote it back
   to its PLANAR reduce (``_demote_planar``) so the kernel still compiles.

``bind_prologue_contraction`` is structure-only and pure, so ``010`` calling it too (it needs the
fused reading as the placement router's reference tree, resolved BEFORE any schedule fork exists)
costs a rebuild, not a divergence.
"""

from __future__ import annotations

from emmy.compiler.graph import Node
from emmy.compiler.ir.stmt import Body
from emmy.compiler.ir.tile import Map, Placement, TileOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.fork import Fork
from emmy.compiler.pipeline.passes.lowering.tile._atomize import bind_prologue_contraction

# NOTE: no ``Knob`` objects (``TILE`` / ``REDUCE`` / ``STAGE``) may be imported here — ``Pass.load``
# scans rule modules for ``Knob`` attrs and OFF-fills any it finds bare onto every variant of the
# pass. Pin reads / knob-key spelling ride the ``_schedule`` helpers instead.
from emmy.compiler.pipeline.passes.lowering.tile._schedule import prologue_knob_bases, schedule, warp_tile_pinned

PATTERN = [Pattern("root", TileOp)]


def _as_list(scheduled) -> list:
    """Normalize a ``schedule()`` result (a single ``TileOp``, a branch ``Fork``, or a candidate
    list — possibly empty) into a flat options list for the MONOID-producer merge."""
    return scheduled if isinstance(scheduled, list) else [scheduled]


def _demote_planar(node):
    """The PLANAR sibling of a computed-A :class:`~emmy.compiler.ir.tile.ir.Contraction` whose
    contraction form yielded no legal schedule row (fp32 / no atoms / bad geometry — the
    scheduler's never-a-raising-row guardrail returns ``[]``): the operand hoist is UNDONE — every
    edge moves INLINE into the lift body (the cone as a structural NODE — a term is a value, legal
    in a pure ``Lambda``; ``_flatten_nodes`` flattens it at lowering, so the loop is byte-identical
    to the old flatten-and-refold) — and with no operand edges the bilinear parse declines, so the
    fold **derives** ``PLANAR`` (``Fold.role``), the same route ``_nodify_contraction`` leaves an
    unbindable cell on, applied post-nodification."""
    src = node.sources[0] if isinstance(node, Map) else node
    red = src.demoted()
    projection = Body(tuple(node.body) if isinstance(node, Map) else ())
    return Map(body=projection, sources=(red,)) if len(projection) else red


def rewrite(match: Match, root: Node, ctx=None) -> Fork | list[TileOp] | TileOp:
    del match  # the scheduled op replaces the matched node in place — no graph surgery here
    tile: TileOp = root.op
    if tile.op is None or tile.place.is_mapped:
        raise RuleSkipped("TileOp already scheduled / nothing to map")
    node, free = tile.op, tile.place.free
    pro = bind_prologue_contraction(node, free)
    if pro is None:
        rows = _as_list(schedule(tile, tile.name, tile.knobs, ctx))
        if not rows:
            # No legal row on any tier — demote to the PLANAR reduce so the kernel still compiles
            # (see the module docstring: an empty option list is a silent no-op to the engine).
            fallback = TileOp(
                op=_demote_planar(node), name=tile.name, place=Placement(free=free), inputs=dict(tile.inputs), stores=tile.stores
            )
            return schedule(fallback, tile.name, tile.knobs, ctx)
        return rows if len(rows) > 1 else rows[0]
    # The MONOID-producer composition — both readings of the one loop, merged into one fork.
    c_map, n_ax, con_stores = pro
    src = c_map.sources[0]  # the ONE bilinear fold — its components share one k axis / cone
    # The cone's source node IS the row-invariant prologue; ITS source is the statistic reduce.
    # Both forms' keys spell against the CONTRACTION tree (the merged fork's one reference tree);
    # the map form keys its own reduce spec on the stat fold's explicit spelling.
    con_base, map_base, stat_key = prologue_knob_bases(c_map, src.a.sources[0].sources[0])
    con_tile = TileOp(op=c_map, name=tile.name, place=Placement(free=(*free, n_ax)), inputs=dict(tile.inputs), stores=con_stores)
    con = _as_list(schedule(con_tile, tile.name, {**tile.knobs, **con_base}, ctx))
    if con and warp_tile_pinned():
        return con if len(con) > 1 else con[0]
    maps = _as_list(schedule(tile, tile.name, {**tile.knobs, **map_base}, ctx, reduce_key=stat_key))
    merged = [*maps, *con]
    return merged if len(merged) > 1 else merged[0]
