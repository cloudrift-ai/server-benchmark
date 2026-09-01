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
`schedule(context)` function applies exactly one lazy frontier and returns either child contexts or complete
assignments. Consumers build a lazy tree by feeding each returned context back to the same function. The driver knows
no concrete family, pipeline fork type, site order, restriction, or enumeration slice.

Classic assignment contexts additionally expose the independent kernel, node, and edge factors.
`enumerate_classic_reference` is their literal Cartesian oracle:

    D(p, t) = K(p, t) × ∏ N(p, t, node) × ∏ E(p, t, edge)
    Algorithm 1(c, p, t) = {a ∈ D(p, t) | c.accepts(a) ∧ accepts(p, t, a)}

`ClassicScheduleContext` contains all three inputs and evaluates the two acceptance terms together. Its frontier may
omit a pick when `c + p + t` proves that no completion exists, but repeated generic expansion must enumerate exactly
the accepted set in every node traversal order. The restriction `c` stays inside the context, never changes an
independent domain, and is not inspected by the generic traversal. Restriction-filtered views and compatibility
indexes are immutable caches over those domains, not alternate definitions of membership.

Reusable leaf choices such as `Work`, `Tile`, `Reduce`, `Stage`, and `Raster` contain neither sites nor target facts.
`views.py` owns reusable target-independent Fold views. `TileOp.nodes` and `TileOp.node_edges` cache their stable
enumeration order; a concrete codec alone translates integer and tuple sites to wire spellings.

## Classic schedule

`ClassicScheduleContext` is the immutable `c + p + t` prefix. It projects the classic domains and owns all classic
compatibility and restriction behavior:
worker inventory, physical-axis agreement, fragment seams, raster eligibility, resource limits, producer-band/TMA
agreement, target availability, pins, and precision restrictions. The independent domains hold choices only; an
expensive `LocalSupport` is derived lazily after the context has selected one node and its incident edge values. This
node-plus-incident-edges frontier is granular enough to reject mixed transport and fragment-seam combinations before
they create subtrees, without materializing the full node × edge product. `extensions` emits partial schedules at
that granularity; `extend` derives and composes their support. Kernel picks form the final frontier. The fragment-seam
relation has no pipeline-side copy.

`ClassicScheduleCodec` is the concrete strict wire boundary. It accepts a `ClassicScheduleContext`, validates complete
assignments through that context, and encodes canonical `WORK`, `TILE`, `REDUCE`, `STAGE`, and `RASTER` rows. There is
no codec base class: a second schedule family should demonstrate any shared codec contract before one is extracted.

The structural cut phase runs before assignment composition. `030_cut` exhausts stored-Fold-edge placement and
`035_split_reduce` then decides the cross-CTA reduction split. Both pass their independent choice domain through
`CutScheduleContext`; `040_schedule` passes the resulting classic assignment context. All three invoke the same
generic `schedule(context)` function. The cut context stores its one structural factor in the generic kernel field; it
does not invent a second enumeration interface.
