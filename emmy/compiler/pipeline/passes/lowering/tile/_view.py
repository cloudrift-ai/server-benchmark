"""Construct the PLACED readings the tile schedule and the Kernel-IR lowering tiers consume.

:mod:`~emmy.compiler.pipeline.passes.lowering._placed` owns the view TYPE — a stored
:class:`~emmy.compiler.ir.tile.ir.Contraction` bound to its caller's placement + schedule facts.
This module owns the two CONSTRUCTORS that read those facts off a recognized tree: the tiled
contraction leaf (:func:`contraction_view`) and the flash streaming pair (:func:`twisted_pair`).
Both are pure readings: they take an op tree + a placement and return views, deciding no schedule
and stamping no ``TileOp`` — that stays with ``_schedule``, their one caller. Neither BUILDS a
node: every contraction is nodified recognize-side (``010_recognize._nodify_contraction`` /
``_atomize.bind_prologue_contraction``), so a tree that reaches here without one is not schedulable
as a contraction and ``contraction_view`` declines with ``LoweringError``.

A view is the TILED CELL's reading — the node, the ``(m, n)`` output axes and a ``TilePlan``. The
schedule's legality gates (the smem slot sizing, the N-mask / TMA-box refusals, the block-thread
limit) are functions of that cell alone, so a probe built here binds nothing else: the resolved
``Stage`` is the schedule's own RESULT (it lands in ``TileOp.schedule``, not in the probe that
judged it), and the kernel's leading grid axes are the GRID's fact, threaded at materialize by the
caller that holds one — ``_factor``, whose grid for a split partial is not the pre-split grid a
probe here ever saw (the ``ksplit`` axis is introduced by the split option).

They live apart from ``_schedule`` because the placement conventions they encode (a root kernel's
output cell is the trailing grid pair, the leading axes ride untiled; a flash consumer supplies the
query axis and the stream / value axis itself) are the SAME conventions the materializer
(``lowering/kernel/_factor``) and the flash realizer (``lowering/kernel/_twist``) rebuild their own
views under — a fact about the view, not about scheduling.
"""

from __future__ import annotations

from emmy.compiler.ir.axis import AxisRole
from emmy.compiler.ir.schedule import TilePlan
from emmy.compiler.ir.stmt import Body
from emmy.compiler.ir.tile import Contraction, Fold, Map
from emmy.compiler.pipeline.passes.lowering._placed import Placed
from emmy.compiler.pipeline.passes.lowering._placed import place as place_view
from emmy.compiler.pipeline.pipeline import LoweringError


def contraction_view(node, place, tile_plan: TilePlan) -> tuple[Placed, Body]:
    """The PLACED reading of a tiled ``CONTRACTION`` leaf — the recognize-side
    :class:`Contraction` node bound to the placement's trailing ``(m, n)`` grid axes and the
    schedule fork's ``tile_plan``. Recognition nodified every contraction
    (``010_recognize._nodify_contraction`` for the per-cell scalar,
    ``_atomize.bind_prologue_contraction`` for the fused cone), so this only reads: the operand→role
    binding is NOT redone here. Returns ``(view, projection)`` — the projection has ONE home, the
    wrapping ``Map``'s body, so the option builders re-wrap it and materialize peels it into the
    store tail (the synthesized grid-``Write`` for a bare contraction stays a materialize concern —
    it needs ``root.output``).

    Raises ``LoweringError`` when the tree holds no placeable contraction — in practice a
    ``Map(Contraction)`` on a 1-D grid (a matvec-shaped decode cell), whose ``(m, n)`` output pair
    does not exist. ``_schedule`` catches it into the ``probe is None`` route: the per-cell / reduce
    tiers, which need no view."""
    grid = list(place.grid)
    if isinstance(node, Map) and len(node.sources) == 1 and isinstance(node.sources[0], Contraction) and len(grid) >= 2:
        # The recognizer's ``Map(body=projection, sources=(Contraction,))`` spelling — the ONE
        # node under the wrapper is the schedulable unit, the projection its ``Map`` body. The
        # placed reading is STAMPED here, its output axes the placement's trailing grid.
        return place_view(node.sources[0], grid[-2], grid[-1], tile=tile_plan), node.body
    if isinstance(node, Contraction) and len(grid) >= 2:
        return place_view(node, grid[-2], grid[-1], tile=tile_plan), Body(())
    raise LoweringError(
        f"warp tier: no bindable contraction to place — need a (Map-wrapped) Contraction node over an (m, n) grid, "
        f"got {type(node).__name__} on a {len(grid)}-D grid"
    )


def twisted_pair(op, free) -> tuple[Fold, Contraction, Contraction, Placed, Placed] | None:
    """The flash-shaped ``TWISTED`` streaming contraction pair — ``(reduction, head_fold, pv_fold,
    head, pv)``: the STORED :class:`Contraction` nodes (the score at the partial's head, the single
    computed-A expect later in the sequence) plus their PLACED readings (output
    axes off the placement's trailing ``free`` — the query axis, and the stream axis for the score /
    the value axis for the expect). ``None`` when not a streaming pair (an online-softmax / RMSNorm
    ``TWISTED`` reduce takes the reduce-partition tiers). The one structural guard the warp / chain /
    scalar flash forms share; each form's own demands (a gmem-``Load`` A, the mma atom's dtype /
    divisibility, the chain's register budget) stay with its builder. Stamping targets the STORED
    nodes (`s is head_fold` in the partial); reads go through the placed copies."""
    src = op.sources[0] if isinstance(op, Map) and len(op.sources) == 1 else None
    red = src if isinstance(src, Fold) else (op if isinstance(op, Fold) else None)
    if red is None or red.role is not AxisRole.TWISTED:
        return None
    stmts = list(red.step_stmts())
    if not stmts:
        return None
    head_fold = stmts[0]
    if not isinstance(head_fold, Contraction) or len(free) < 2:
        return None
    tail_folds = [st for st in stmts[1:] if isinstance(st, Contraction)]
    if len(tail_folds) != 1:
        return None
    pv_fold = tail_folds[0]
    pv = place_view(pv_fold, free[-2], free[-1])
    if not pv.a_computed:
        return None
    return red, head_fold, pv_fold, place_view(head_fold, free[-2], red.axis), pv
