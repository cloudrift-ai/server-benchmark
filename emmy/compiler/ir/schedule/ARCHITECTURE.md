# Schedule model

`Schedule` is the generic immutable kernel × node × edge assignment. Node sites are non-negative integers and edge
sites are `(consumer node id, operand position)` tuples; the assignment contains no problem, target, path spelling, or
lowering facts. A concrete schedule family may carry derived lowering facts in a separate materialization type.

`ScheduleContext` is the immutable compatibility-prefix interface shared by every enumeration slice. Its defining
operations are a lazy frontier and composition:

    for pick in context.extensions():
        next = context.extend(pick)

Every context assignment and extension is a `Schedule[KernelT, NodeT, EdgeT]`; a non-`None` kernel marks completion.
A returned context contains the composed facts and leaves the original unchanged; incompatibility raises
`ScheduleRefused`. `extend` is also the validation boundary for a complete classic assignment supplied directly by a
pinned golden, even when that assignment was not emitted by `extensions`. The generic `schedule(context)` recursively
composes those lazy frontiers and yields only complete assignments. Recursion is the generic Algorithm 1 traversal;
consumers do not write a family-specific visitor or feed contexts back themselves. The driver knows no concrete
family, pipeline fork type, site order, restriction, or enumeration slice. The pipeline's generic schedule-fork
adapter preserves the same contexts as deferred search branches without adding compatibility logic.

Classic assignment contexts additionally expose the independent kernel, node, and edge factors. Tests retain a
literal Cartesian oracle:

    D(p, t) = K(p, t) × ∏ N(p, t, node) × ∏ E(p, t, edge)
    Algorithm 1(c, p, t) = {a ∈ D(p, t) | extend(c + p + t, a) succeeds}

`ClassicScheduleContext` contains all three inputs and evaluates the two acceptance terms together. Its frontier may
omit a pick when `c + p + t` proves that no completion exists, but repeated generic expansion must enumerate exactly
the accepted set in every node traversal order. The restriction `c` stays inside the context, never changes an
independent domain, and is not inspected by the generic traversal. Restriction-filtered views and compatibility
indexes are immutable caches over those domains, not alternate definitions of membership.

Reusable leaf choices such as `Work`, `Tile`, `Reduce`, `Stage`, and `Raster` contain neither sites nor target facts.
The `TileOp` **is** the site index — there is no second object over the same term. Whether a bilinear site takes
`TILE` and `STAGE` at all (`contracts`) is the tile's question, since it needs the kernel's extents: a pair whose
role-less side shares a coordinate with the other side qualifies only while that coordinate partitions the reduction
(it composes with the reduction index) or the role-bearing side's reads are value-dead in it (a merged weight's
reshape residue); a B that changes with the row it is contracted against is no slab per tile. It derives stable
node ids,
operand-edge sites, each site's projection or reduction view, and each contraction's schedule-independent
`ContractionFacts` — its effective K axis, computed-A cone seam, nested producer, and fragment need.
`ir/schedule/views` supplies the vocabulary (`node_view`, `Projection`, `Reduction`, `Contraction`,
`ContractionFacts`) and the one derivation that is not a projection of the site table, `contraction_facts`; the tile
layer reads through them. The composition context publishes the schedule-facing API (`node`, `site`, `operand`,
`producer`, `incident_edges`, the key spellings) and does not re-export the kernel's structural members under second
names. A contraction view belongs to a node site and expresses its operand roles as edge positions, and is not
another Fold node. A concrete codec alone translates integer and tuple sites to wire spellings, and the route it spells is the site
record's own.

The node list is the one walk, `ir/tile/path.sites`, deduplicated by object identity. That walk yields a `Site` per
node — the term, the axes in scope, the segment path — so the schedule's integer ids, the tree-path codec's segments,
and the cut pass's scopes are all readings of one traversal and cannot drift. Operands are visited in stored order,
which formation orients (a contraction's A first), and a route spells the stored position taken at each departure;
an ambiguous node family spells its site by that route (`TILE@map.1/twist.1/inner`), the one grammar `PLACE` uses.

**Every derivation memoizes on the ROOT, not on the wrapper**: several `TileOp`s exist over one term across a
lowering, and a cache on the wrapper silently re-derives per wrapper. `schedule_nodes`, `schedule_views` and
`contraction_facts` all key their memo on the Fold root, and the `TileOp` properties are accessors over it.

## Classic schedule

`ClassicScheduleContext` is the immutable `c + p + t` prefix: the problem `p` is the unscheduled TileOp itself and `t`
its target, held as two fields, and `ClassicDomains` carries only the literal independent factors. There is no separate
problem object. Everything a schedule choice cannot change is derived from those two and memoized on the term it
derives from — a contraction's `ContractionFacts` on the Fold root (`TileOp.contractions`), the packed operand
readings and the placement on the TileOp, and the per-target support tables on the TileOp beside their target. The
context owns all classic compatibility and restriction behavior:
worker inventory, physical-axis agreement, fragment seams, raster eligibility, resource limits, producer-band/TMA
agreement, target availability, pins, and precision restrictions. The independent domains hold choices only; an
expensive local support record is derived lazily after the context has selected one node and its incident edge values.
This node-plus-incident-edges frontier is granular enough to reject mixed transport and fragment-seam combinations
before they create subtrees, without materializing the full node × edge product. `extensions` emits partial schedules
at that granularity; `extend` derives and composes their support. Kernel picks form the final frontier. The
fragment-seam relation has no pipeline-side copy.

Classic domain projection, move catalogs, packed-operand readings, staging resolution, materialization, and
compatibility all live in `ir/schedule`. Projection returns the independent `ClassicDomains` alone; pipeline search
neither defines nor filters those domains. `ir/schedule` may import other IR modules but never the
pipeline layer. The pipeline retains only knob/pin reads, pool identity, sampling, and the generic lazy-Fork adapter.

A reduction domain is projected from node and kernel facts alone, so the shapes the kernel factorizer cannot bind are
decided once, at the offer, and never dropped from a priced row later. The partition catalog is offered only on the
reduce nodes the binder builds the kernel around — the roots it peels from the root projection (`ops.kernel_roots`: a
tiled contraction's root, every one of them for a multi-output kernel, else the first operand); a reduce nested under
a root or beside it lowers serially inside its reader, so it carries the serial fold only, as does an observed node
and one whose reduce reads a boundary store's output sweep. The contraction per-cell tier reads that same
projection, so a contraction inherits those readings rather than restating them.

A contraction whose K is a BLOCK axis has a SYMBOLIC extent: blocking is a normalization that
leaves the width unbound, and `bk` is what binds it. The two are one quantity — both say how many
contraction columns one step of the enclosing loop consumes — so nothing narrows the other, the
domain enumerates `bk` freely, and `materialize_classic` reads the width back off the accepted row.
That is why blocking adds no schedule family: the row already spells the width, once, in the family
that owns the K.

A pin is a restriction on those projected domains, never a source of choices, so it narrows what a site may select and
cannot manufacture a value the projection withheld. A value scoped to a site that does not offer it empties that
site's restriction and the kernel enumerates no row — the loud direction. A bare family pin is applicable at a site
only when the value already belongs to that site's projected values, which is what lets one ambient pin sweep a whole
model; on a site that cannot carry it, it is silently inapplicable rather than refused.

`ClassicScheduleCodec` is the concrete strict wire boundary. Its public encode and decode operations validate through
one `ClassicScheduleContext`; private syntax-only parsing and encoding let graph reconstruction attach materialization
before the `TileOp` constructor performs that same validation once. It owns canonical complete-row and prefix-delta
encoding for `WORK`, `TILE`, `REDUCE`, `STAGE`, and `RASTER`. There is no codec base class: a second schedule family
should demonstrate any shared codec contract before one is extracted.

The structural cut phase runs before assignment composition. The single `030_cut` pass reaches a fixpoint over two
ordered domains: stored-Fold-edge placement first, then cross-CTA reduction splitting. Every successful choice and
fresh piece re-enters the same rule. `030_cut` presents its restricted structural frontier through a schedule context;
`040_schedule` supplies a `ClassicScheduleContext`. Both passes use the same generic `schedule` traversal.
