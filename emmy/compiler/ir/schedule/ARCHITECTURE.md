# Schedule model

`Schedule` is the generic immutable kernel × node × edge assignment. Node sites are non-negative integers and edge
sites are `(consumer node id, operand position)` tuples; the assignment contains no problem, target, path spelling, or
lowering facts. A concrete schedule family may carry derived lowering facts in a separate materialization type.

`ScheduleContext` is the immutable compatibility-prefix interface shared by every enumeration slice. Its defining
operations are a lazy frontier and composition:

    for pick in context.extensions():
        next = context.extend(pick)

Every context assignment and extension is a `Schedule[KernelT, NodeT, EdgeT]`; a non-`None` kernel marks completion.
Edge options implement the small `Edge` contract, whose `is_cut()` query lets `ScheduleContext.only_cuts()` expose
the cut-bearing part of the same frontier without manufacturing another context. A returned context contains the
composed facts and leaves the original unchanged; incompatibility raises `ScheduleRefused`. `extend` is also the
validation boundary for a complete classic assignment supplied directly by a pinned golden, even when that
assignment was not emitted by `extensions`. The generic `schedule(context)` recursively composes those lazy frontiers
and yields only complete assignments. Recursion is the generic Algorithm 1 traversal; consumers do not write a
family-specific visitor or feed contexts back themselves. The driver knows no concrete family, pipeline fork type,
site order, restriction, or enumeration slice. The pipeline's generic schedule-fork adapter preserves the same
contexts as deferred search branches without adding compatibility logic.

Classic assignment contexts additionally expose the independent kernel, node, and edge factors.
`enumerate_classic_reference` is their literal Cartesian oracle:

    D(p, t) = K(p, t) × ∏ N(p, t, node) × ∏ E(p, t, edge)
    Algorithm 1(c, p, t) = {a ∈ D(p, t) | extend(c + p + t, a) succeeds}

`ClassicScheduleContext` contains all three inputs and evaluates the two acceptance terms together. Its frontier may
omit a pick when `c + p + t` proves that no completion exists, but repeated generic expansion must enumerate exactly
the accepted set in every node traversal order. The restriction `c` stays inside the context, never changes an
independent domain, and is not inspected by the generic traversal. Restriction-filtered views and compatibility
indexes are immutable caches over those domains, not alternate definitions of membership.

Reusable leaf choices such as `Work`, `Tile`, `Reduce`, `Stage`, and `Raster` contain neither sites nor target facts.
`ClassicSites` is the formal target-independent reading of one Fold root: it derives stable node ids, operand-edge
sites, and each site's projection or reduction view while storing only the root. A contraction view belongs to a node
site and expresses its operand roles as edge positions; it is not another Fold node. A concrete codec alone translates
integer and tuple sites to wire spellings.

## Classic schedule

`ClassicScheduleContext` is the immutable `c + p + t` prefix. `ClassicProblem` carries the source TileOp, target, and
derived contraction facts; `ClassicDomains` carries only the literal independent factors. The context owns all classic
compatibility and restriction behavior:
worker inventory, physical-axis agreement, fragment seams, raster eligibility, resource limits, producer-band/TMA
agreement, target availability, pins, and precision restrictions. The independent domains hold choices only; an
expensive `LocalSupport` is derived lazily after the context has selected one node and its incident edge values. This
node-plus-incident-edges frontier is granular enough to reject mixed transport and fragment-seam combinations before
they create subtrees, without materializing the full node × edge product. `extensions` emits partial schedules at
that granularity; `extend` derives and composes their support. Kernel picks form the final frontier. The fragment-seam
relation has no pipeline-side copy.

Classic domain projection, move catalogs, packed-operand readings, staging resolution, materialization, and
compatibility all live in `ir/schedule`. Projection returns one `ClassicProblem` and its independent `ClassicDomains`;
pipeline search neither defines nor filters those domains. `ir/schedule` may import other IR modules but never the
pipeline layer. The pipeline retains only knob/pin reads, pool identity, sampling, and the generic lazy-Fork adapter.

`ClassicScheduleCodec` is the concrete strict wire boundary. With a `ClassicScheduleContext` it validates compatibility;
with `ClassicSites` it performs target-independent structural validation for graph serialization. It owns canonical
complete-row and prefix-delta encoding for `WORK`, `TILE`, `REDUCE`, `STAGE`, and `RASTER`. There is no codec base
class:
a second schedule family should demonstrate any shared codec contract before one is extracted.

The structural cut phase runs before assignment composition. The single `030_cut` pass reaches a fixpoint over two
ordered domains: stored-Fold-edge placement first, then cross-CTA reduction splitting. Every successful choice and
fresh piece re-enters the same rule. `030_cut` emits its pass-native structural forks directly; it does not manufacture
a schedule context for them. `040_schedule` supplies a `ClassicScheduleContext` to the generic schedule traversal.
