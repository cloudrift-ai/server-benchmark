"""Recognize a ``LoopOp``'s algebraic structure and lift it to an UNMAPPED ``TileOp`` — the
STRUCTURAL half of the Loop-IR → Tile-IR boundary.

After this rule nothing downstream traffics in ``LoopOp``. Recognition reads the algebra off the
body and lifts the per-cell compute into a :class:`~emmy.compiler.ir.tile.ir.Fold` whose body is the
**annotated loop nest** (the reduce ``Loop`` stamped with its
:class:`~emmy.compiler.ir.axis.AxisRole` — the only loop annotation; the algebra is the body),
wrapped in a :class:`~emmy.compiler.ir.tile.ir.TileOp` whose ``place`` carries just the free axes.
That op IS the rewrite's result. The schedule picks it up on the next rule sweep, maps the free
axes onto the grid and offers the scheduling forks; materialization back to loop IR happens in
``lowering/kernel``.

Nothing here reads a knob or a pin — recognition is structure, and every choice it makes is
unconditional. (The one exception is PLACEMENT, step 3.5: a ``PLACE`` pin
cuts the recognized tree into a fragment of un-mapped ``LoopOp``\\ s, and it must resolve BEFORE any
schedule fork exists, so it cannot wait for ``020``.)

All recognition lives in THIS one rule (no separate softmax pass), in order (each
step unconditional — no knobs):

1. **Online softmax** — an adjacent ``(rowmax, Σ exp)`` reduce pair over the same input fuses
   into one streaming online-softmax loop: a ``TWISTED`` reduce ``Loop`` carrying the
   exp-family merge dissolved in the body. The carrier is N-channel: a further sibling additive
   fold whose lifted value is the pair's weight × a value cone joins the same loop as an
   EXPECTATION channel (loop-invariant factors split off, multiplied back after the loop), and
   a pair whose channels sit inside a following free output sweep (fused softmax·V) sinks into
   that sweep first. The ``_softmax`` helper (``_fuse``).
3. **Lift** — peel the free (parallel) axes off the kernel and lift the per-cell compute into a
   zero-axis ``Fold`` whose body holds the annotated reduce ``Loop`` + projection: a pure pointwise body is a
   flat zero-axis fold; a single flat reduce is annotated in place — ``CONTRACTION`` (clean contraction)
   / ``PLANAR`` (plain ``sum`` / ``max`` / ``mean``) / pre-annotated ``TWISTED`` (online softmax) —
   with the projection after it. The free axes ride on
   the ``TileOp``'s schedule (the root's concern); ``_schedule`` maps them onto the grid. A cell
   the lift can't cleanly factor (no reduce, several reduces, or a nested non-flash reduce) stays a
   flat un-annotated zero-axis ``Fold`` (→ the scalar tier).
4. **The MONOID-producer composition** — a lifted ``Fold.projection(source=Fold)`` whose body is the
   statistic's scalar epilogue + a fresh free (column) ``Loop`` over one or more ⊗-folds of ONE
   shared A value reading the statistic (the fused norm→linear edge ``rmsnorm(x)·nw @ w``; its
   N-channel form the gate/up MLP edge ``swiglu(x̂@Wg, x̂@Wu)`` — a product-monoid fold) ALSO
   nodifies to ``Fold.projection(body=projection, operands=(<contraction>,))``: ONE computed-A product
   contraction with a :class:`Channel` per ⊗-fold, the A cone stored inline on its edge
   (a real node tree — the per-row statistic its ``Fold`` source)
   (``_atomize.bind_prologue_contraction``, structure-only), its column axis joining the grid.
   Recognition only *builds* that reading (for the routing router's reference tree, below);
   The schedule re-derives it and merges both forms' candidates into ONE fork, because which
   of the two a row realizes is a decision about the SCHEDULE.

Flash must precede online-softmax which must precede the lift: each later step consumes the
``Accum``\\ s an earlier one matches. A **symbolic** axis (dynamic ``seq_len``) is left
un-lifted (the scalar ``Tile`` decode needs static extents) — the ``LoopOp`` stays put for
the dynamic-shape tier.

Recognition reads through TWO shared algebra parsers and nothing else: the λ-fold reading
(:func:`~._fromloop.fold_from_loop` — every reduce ``Loop`` interpreted as a ``Fold``, gated by
byte-identity of the re-derived loop) and the ⊗-lift reading (``_atomize._bilinear_reads`` — every
bilinear fold's (B, A-value, accumulator) facts, shared by both contraction binders). The
online-softmax pairing states its condition on λ-fold results; contraction candidacy
(:func:`_bilinear_candidate`) is deliberately liberal and the ONE binder arbitrates operand
shapes; the monoid composition binds its per-channel folds through the same lift reading. What
remains case-by-case is the DISPATCH (which composition applies), not the parsing — no step holds
a private stmt-pattern reading of the algebra.
"""

from __future__ import annotations

from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import Match, Pattern
from emmy.compiler.pipeline.passes.lowering.tile._atomize import bind_prologue_contraction
from emmy.compiler.pipeline.passes.lowering.tile._cut import cuttable_seams, realize_cut, route_cut
from emmy.compiler.pipeline.passes.lowering.tile._lift import recognized_tile
from emmy.compiler.pipeline.pipeline import RuleSkipped

PATTERN = [Pattern("root", LoopOp)]


def rewrite(match: Match, root: Node, ctx=None) -> TileOp | Graph | None:
    loop: LoopOp = root.op
    # Steps (1)–(3) — softmax fusion, the free-axis peel, the cell lift / nodification — are the
    # pure recognition core (:func:`._lift.recognized_tile`), shared verbatim with the strict
    # golden decode's record-side identity derivation.
    map_tile = recognized_tile(loop, root.output.name, name=loop.name)
    # The matcher re-populates io when a later pass matches the op; seeding the output here makes
    # the UNMAPPED tile self-describing at the placement fork, where the verified tier reads its
    # identity (``deploy_identity`` folds the output dtype) before any match has run.
    map_tile.outputs = {root.output.name: root.output}
    node, free, stores = map_tile.op, map_tile.place.free, map_tile.stores
    # A symbolic FREE (parallel) axis rides a **symbolic grid**: the ``Tile`` decode sizes the
    # launch from the runtime extent (``_gid < ∏extents``, the ``Dim`` name threaded as an
    # ``int`` arg by the cuda lowering) — the dynamic-grid tier. A symbolic REDUCE /
    # output-sweep axis is likewise supported (the reduce loop strides to the runtime extent,
    # the ``< seq_len`` cap masking the tail). Register-tiled symbolic axes mask their tail
    # cell (clamp-read + guarded write) in ``lowering/kernel``.
    # Wrap the lifted node + its unmapped placement in an UNMAPPED ``TileOp`` — recognition's
    # OUTPUT. The schedule picks it up on the next rule sweep, maps the free axes onto the grid
    # and offers the per-axis scheduling forks (``REDUCE`` partition / ``TILE`` output tile).
    # ``inputs`` is seeded from the matched ``LoopOp`` (the matcher populated its real Tensors) so
    # the scheduler can read operand shapes (the shared-row stage detection); the matcher refreshes
    # it from the graph again when a later pass matches the scheduled op.
    pro = bind_prologue_contraction(node, free)
    # (3.5) PLACEMENT — resolved FIRST, before any schedule fork exists: an authoritative
    # PLACE pin cuts the recognized tree into a fragment of un-mapped LoopOps (or keeps it
    # fused); each piece re-recognizes as a fresh root on the pass-scan restart (recursive — a
    # deeper pin key may cut a piece again). The fused (computed-A) view is the reference tree
    # when it binds — its seams (the `a` cone edge) are the ones a ``PLACE`` key spells.
    #
    # UNPINNED, placement is an enumerated STRUCTURAL fork: the fused form beside one cut fragment
    # per legal seam, so tune DISCOVERS cuts and a deploy prices them like any kernel-set choice
    # (``greedy._priced_pick``). Nothing holds the fused side ahead of the cuts — this list is a
    # set of legal placements, not a ranking. Each fragment's parent piece is stamped
    # ``PLACE@<seam>: cut`` so a recorded routing golden can match the OPTION by the seam it names
    # (``greedy._verified_pick``); the splice then consumes the stamp with everything else, because
    # the resulting kernel set is the record of what was chosen.
    route_tree, route_free, route_stores = (pro[0], (*free, pro[1]), pro[2]) if pro is not None else (node, free, stores)
    verdict, seam = route_cut(ctx, dict(loop.knobs or {}), route_tree, route_stores, route_free)
    if verdict == "cut":
        return realize_cut(match, root, route_tree, route_free, route_stores, seam)
    if verdict is None:
        seams = cuttable_seams(route_tree, route_stores, route_free)
        cut_options = []
        for s_ in seams:
            try:
                cut_options.append(realize_cut(match, root, route_tree, route_free, route_stores, s_))
            except RuleSkipped:
                continue  # the seam's workspace already exists — a piece of an applied cut
            except ValueError:
                # The realizer cannot BUILD this seam's fragment (a piece body that fails Loop IR
                # validation) — the enumeration drops it, exactly the unpinned half of the
                # ``legal.enforce`` convention; a PLACE pin naming the same seam still raises
                # loudly through the ``route_cut`` arm above.
                continue
        if cut_options:
            return [map_tile, *cut_options]
    # Recognition ends here: the UNMAPPED tile is the rewrite's result. The MONOID-producer
    # composition (``pro``) is re-derived by the schedule — it is a decision about the SCHEDULE
    # (which of the two readings of this one loop each fork row realizes), not about the structure,
    # and it needs the schedule results to arbitrate (a warp ``TILE`` pin keeps the contraction
    # rows alone; a contraction form with no legal row demotes back to the PLANAR reduce).
    return map_tile
