# Schedule model

`Schedule` is the generic immutable kernel × node × edge assignment. Node sites are non-negative integers and edge
sites are `(consumer node id, operand position)` tuples; the assignment contains no problem, target, path spelling, or
lowering facts. A concrete schedule family may carry derived lowering facts in a separate materialization type.

`ScheduleContext` is the immutable compatibility-prefix interface shared by every enumeration slice. Its defining
operations are a lazy frontier and composition:

    for pick in context.extensions():
        next = context.extend(pick)

Each pick is itself a partial `Schedule`. A returned context contains the composed facts and leaves the original
unchanged; incompatibility raises `ScheduleRefused`. `extend` is also the validation boundary for a complete assignment
supplied directly by a pinned golden, even when that assignment was not emitted by `extensions`. The generic
`schedule(context)` recursively composes those lazy frontiers and yields only complete assignments. Recursion is the
generic Algorithm 1 traversal; consumers do not write a family-specific visitor or feed contexts back themselves. The
driver knows no concrete family, pipeline fork type, site order, restriction, or enumeration slice. The pipeline's
generic schedule-fork adapter preserves the same contexts as deferred search branches without adding compatibility
logic.

Classic assignment contexts additionally expose the independent kernel, node, and edge factors. Tests compare the
generic traversal with their literal Cartesian oracle:

    D(p, t) = K(p, t) × ∏ N(p, t, node) × ∏ E(p, t, edge)
    Algorithm 1(c, p, t) = {a ∈ D(p, t) | extend(c + p + t, a) succeeds}

`ClassicScheduleContext` contains all three inputs and evaluates the two acceptance terms together. Its frontier may
omit a pick when `c + p + t` proves that no completion exists, but repeated generic expansion must enumerate exactly
the accepted set in every node traversal order. The restriction `c` stays inside the context, never changes an
independent domain, and is not inspected by the generic traversal. Restriction-filtered views and compatibility
indexes are immutable caches over those domains, not alternate definitions of membership.

Reusable leaf choices such as `Work`, `Tile`, `Reduce`, `Stage`, and `Raster` contain neither sites nor target facts.
`views.py` owns the reusable target-independent `ScheduleInventory` over Fold views. `TileOp.nodes` and
`TileOp.node_edges` cache its stable enumeration order; identity indexes are rebuilt after pickling, and a concrete
codec alone translates integer and tuple sites to wire spellings.

## Classic schedule

`ClassicScheduleContext` is the immutable `c + p + t` prefix. `ClassicProblem` is constructed only from a complete
source TileOp and carries the target plus every derived contraction and refusal fact; there is no weaker root-only
problem. `ClassicDomains` carries only the literal independent factors. The context owns all classic compatibility and
restriction behavior:
worker inventory, physical-axis agreement, fragment seams, raster eligibility, resource limits, producer-band/TMA
agreement, target availability, pins, and precision restrictions. The independent domains hold choices only; local
compatibility evidence is derived lazily after the context has selected one node and its incident edge values. This
node-plus-incident-edges frontier is granular enough to reject mixed transport and fragment-seam combinations before
they create subtrees, without materializing the full node × edge product. `extensions` emits partial schedules at
that granularity; `extend` derives and composes their support. Kernel picks form the final frontier. The fragment-seam
relation has no pipeline-side copy.

The schedule package imports no pipeline implementation. Shared Fold-tree traversal, address algebra, and packed
operand recognition live in `ir/fold_tree.py`, `ir/address.py`, and `ir/packed.py`; `ClassicProblem.from_tile` derives
its complete facts there rather than calling upward into domain projection. Per-problem and per-domain compatibility
indexes are derived memo tables, excluded from pickles and rebuilt after transport.

`ClassicScheduleCodec` is the concrete strict wire boundary. It accepts a `ClassicScheduleContext`, validates complete
assignments through that context, and encodes canonical `WORK`, `TILE`, `REDUCE`, `STAGE`, and `RASTER` rows. There is
one syntax-only `parse` step for graph reconstruction: materialization is decoded from the typed schedule, construction
then validates the complete TileOp through the context, and canonical encoding is checked with that validated source.
There is no codec base class: a second schedule family should demonstrate any shared codec contract before one is
extracted.

The structural cut phase runs before assignment composition. `030_cut` decides stored-Fold-edge placement and
`035_split_reduce` then decides the cross-CTA reduction split; fresh pieces re-enter those ordinary passes. Both pass
their independent choice domain through `CutScheduleContext`; `040_schedule` passes the resulting classic assignment
context. All three use the generic schedule/context contract. The cut context stores its one structural factor in the
generic kernel field; it does not invent a second enumeration interface.
