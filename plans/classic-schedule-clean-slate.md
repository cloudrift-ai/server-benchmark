# ClassicSchedule clean-slate reconstruction

Status: proposed. This plan has not been executed.

## Goal

Replace the current scheduler with one explicit `ClassicSchedule` model for the ordinary
grid/CTA/warp/thread/register execution model. It defines the scheduling problem, site classification, independent
kernel/node/edge domains, compatibility, enumeration, serialization, and materialization contract.

Future warp-specialized and megakernel schedules may define different problems and schedules. They do not inherit
`ClassicScheduleContext`. Extract shared interfaces only after a second schedule family demonstrates them.

This is a clean-slate reconstruction. Phase 0 deletes the old scheduling policy and places every intentionally broken
schedule-dependent test in one strict xfail registry. Later phases recover that registry by coherent design cluster.
The registry must be deleted before the work is complete.

## Core invariants

1. Sites are the only identities that receive schedules. A node site is scheduled once; every consumer operand is a
   distinct edge site even when several edges reach the same producer.
2. Classification reads a node site. It may inspect that site's Fold, but it binds contraction operand roles relative
   to the site.
3. Node and edge domains are independent. They depend on static problem facts, never on another selected schedule.
4. Compatibility is one pure relation over a complete kernel, node, and edge assignment.
5. Enumeration is the compatible subset of one Cartesian product. A faster traversal may prune that product but may
   not change it.
6. Structural choices that change computation or kernel boundaries happen before site construction.
7. Schedule values contain choices only. They do not cache shapes, resources, classifications, paths, or encodings.
8. Serialization and search consume the schedule model; they do not define it.
9. There is no legacy schedule representation, compatibility adapter, alias codec, or dual reader.

## Scheduling problem and sites

A classic scheduling problem consists only of the unscheduled Fold tree and target. Everything else is derived.

```python
@dataclass(frozen=True)
class ClassicProblem:
    root: Fold
    target: Target


@dataclass(frozen=True)
class NodeSite:
    id: NodeId


@dataclass(frozen=True)
class EdgeSite:
    consumer: NodeSite
    operand: int
```

An immutable site index resolves `NodeSite` to its Fold. `EdgeSite` needs no source field: its producer is recovered
from the consumer Fold and operand position. `NodeId` has one stable serialized spelling, but that spelling is not part
of the semantic type.

A shared producer has one `NodeSite`. Each use has a distinct `EdgeSite`. Recompute, fusion, cuts, and split-K create a
different `ClassicProblem`; an edge schedule cannot create, clone, suppress, or relocate computation.

## Site classification

Classification returns a reading for a `NodeSite`. The reading is stored under that site, so it does not repeat the
site as a field:

```python
@dataclass(frozen=True)
class Projection:
    pass


@dataclass(frozen=True)
class Contraction:
    a: int
    channels: tuple[int, ...]


@dataclass(frozen=True)
class Reduction:
    contraction: Contraction | None = None


NodeView = Projection | Reduction
```

`classify(site_index, site) -> NodeView` returns `Projection` for a zero-axis site and `Reduction` for a fold site.
`Contraction` is a capability of a reduction, not a competing top-level view. Its integers are consumer operand
positions; `EdgeSite(site, position)` binds them to schedulable edges. Channel order corresponds to carried-state
order, so no channel record is needed.

Classification cannot receive the target. This makes its algebraic independence an API property rather than a
convention.

Axis, carrier, observer, semiring, shapes, and data types remain on the Fold and are read through the problem. Do not
duplicate them in a view. Do not add scan, twisted, pointwise, or generic tagged view classes: add a field or type only
when classification must bind a stable identity that cannot be recovered from the problem and site.

Bare-Fold algebra helpers may support classification, but they are not scheduling views. Scheduling primitives receive
the problem, site, and already-derived view, never anonymous tuples, role enums, or classification booleans.

## Schedule algebra

Use sums to make invalid node assignments unrepresentable and explicit values for direct/off choices:

```python
@dataclass(frozen=True)
class KernelSchedule:
    work: Work
    raster: Raster


@dataclass(frozen=True)
class ProjectionSchedule:
    tile: Tile


@dataclass(frozen=True)
class ReductionSchedule:
    tile: Tile
    reduce: Reduce


NodeSchedule = ProjectionSchedule | ReductionSchedule


@dataclass(frozen=True)
class EdgeSchedule:
    stage: Stage


@dataclass(frozen=True)
class ClassicSchedule:
    kernel: KernelSchedule
    nodes: Mapping[NodeSite, NodeSchedule]
    edges: Mapping[EdgeSite, EdgeSchedule]
```

`Work`, `Raster`, `Tile`, `Reduce`, and `Stage` include their own direct/off values where those are valid choices.
There are no optional primitive fields.

Do not add producer-band state to the minimal model. If it proves to be an independent kernel choice, add it to
`KernelSchedule`; if it describes transport, add it to `Stage`. Its scope must be established semantically first.

`ClassicSchedule` is data and cannot validate its own coverage without a problem. Completeness, scope, and
compatibility are checked by `ClassicScheduleContext`.

## Independent domains

The semantic domains are pure functions of static problem facts:

```text
kernel_domain(problem) -> Iterable[KernelSchedule]
node_domain(problem, site, view) -> Iterable[NodeSchedule]
edge_domain(problem, edge) -> Iterable[EdgeSchedule]
```

Each domain owns local legality for its choice: target availability, static shape requirements, data-type support, and
single-scope resource bounds. A domain never receives or inspects another selected schedule.

This strict independence may generate combinations that are obviously incompatible. Removing them is the job of
compatibility and of an equivalent pruning implementation, not a reason to make one domain depend on another.

## Compatibility

`ClassicScheduleContext` contains the derived facts for one `ClassicProblem` and exposes one semantic operation:

```text
context.accepts(schedule) -> Result[None, Refusal]
```

It validates assignment coverage and owns every relation between choices, including:

- node schedule agreement with its classified view;
- worker and physical-axis agreement;
- producer/consumer tile agreement;
- edge transport agreement with both endpoints;
- fragment and layout agreement;
- combined thread, register, and shared-memory limits;
- raster eligibility.

The relation is pure and symmetric over the complete assignment. It does not enumerate, rank, encode, materialize,
rewrite the Fold tree, or expose incremental solver state.

An optimized enumerator may use a private constraint propagator to reject partial assignments. Set-equivalence tests
must prove that it implements the same relation as `context.accepts` over complete schedules. Pending constraints,
choice ordering, and incremental caches are enumerator details, not public context semantics.

## Enumeration

For problem `P`, node sites `V`, and edge sites `E`, define:

```text
G(P) = kernel_domain(P)
N(P, v) = node_domain(P, v, classify(site_index(P), v))
M(P, e) = edge_domain(P, e)

Classic(P) = {
    (g, n, m) in G(P) × product(N(P, v) for v in V) × product(M(P, e) for e in E)
    | ClassicScheduleContext(P).accepts(g, n, m)
}
```

The first implementation is the literal Cartesian reference enumerator. It is deliberately simple and need not be
fast. The production lazy enumerator comes later and must return exactly the same set.

Enumeration emits complete typed schedules and refusal diagnostics. It does not emit knob dictionaries, derive
features, rank candidates, apply performance heuristics, or prefer defaults.

Structural scheduling is an outer operation:

```text
P' in structural_choices(P) -> enumerate ClassicSchedule independently for every kernel problem in P'
```

Each resulting kernel starts unscheduled. Classic enumeration does not know whether it came from an ordinary lift, a
cut, or split-K.

## Serialization

Serialization is a separate boundary over the stable schedule algebra. `ClassicScheduleCodec` walks kernel, node, and
edge scopes in canonical site order. Leaf encoders live beside their choice types, but codec policy is not part of the
primitive or compatibility contract.

Strict decoding constructs a complete typed schedule and passes it through `ClassicScheduleContext`. It either returns
one accepted schedule or fails loudly. Pins use the same path; there is no permissive parse mode.

Old schedule rows have no compatibility guarantee. Once the new algebra is stable, regenerate tracked fixtures and
discard mutable evidence using the retired spelling. Do not add legacy readers.

## Phase 0 — demolition and xfail registry

Delete the old schedule model, domains, compatibility logic, enumeration, codec, and schedule-driven materialization
adapter. Retain mathematical IR, reusable classification mechanics, lowering emitters, structural rewrites, generic
search machinery, and tests.

Add the smallest importable skeleton. Any schedule behavior not yet rebuilt raises `ClassicScheduleUnavailable`.

Create one strict reconstruction registry. Each entry names an exact collected test node ID, recovery cluster, phase,
reason, and expected `ClassicScheduleUnavailable`. Wildcards and inline reconstruction xfails are forbidden. Audit
that every entry is collected once, no test belongs to two clusters, and every reconstruction failure is registered.

Phase 0 exits when the old scheduler is gone, the complete suite collects, every schedule-dependent failure is an
expected strict xfail, unrelated failures remain visible, and lint passes. Freeze the registry count; later phases may
only remove entries or split a cluster without increasing the total.

## Recovery phases

Each phase removes complete registry clusters. A phase is done only when its recovered tests pass unmarked and the
full suite has no unregistered regression.

### Phase 1 — reference model and one vertical path

Implement `ClassicProblem`, the site index, site classification, the schedule algebra, independent domains, pure
compatibility, and the Cartesian reference enumerator for direct projection and direct edges. Materialize that path.

Include shared-producer and multiple-consumer examples now, not as a later extension. Prove one node assignment and
distinct edge assignments, complete coverage validation, deterministic classification, and enumeration as an exact
filtered product.

### Phase 2 — reductions

Add reduction domains, compatibility, and materialization for serial, cooperative, segmented, and observed folds.
Recover reduction unit, end-to-end, and realization clusters through the same reference enumerator.

### Phase 3 — scalar contractions

Add contraction classification and scalar contraction choices. Recover scalar geometry, reduction, masking,
end-to-end, and realization clusters. A contraction remains a `ReductionSchedule` selected for a reduction with the
contraction capability.

### Phase 4 — tensor cores, edge transport, and kernel choices

Add tensor-core node choices together with direct, shared-memory, asynchronous, and fragment transport edge choices,
plus raster choices. Recover atom, staging, fragment, target-capability, and combined-resource clusters as one vertical
contract. An edge choice may transport an existing value but cannot change the computation graph.

### Phase 5 — multiple-node kernels

Recover composed contractions, attention, and other producer/consumer combinations. All behavior must emerge from site
classification, independent domains, and compatibility relations; do not add graph-shape-specific enumeration
branches.

### Phase 6 — structural choices

Reconnect cuts and split-K outside classic enumeration. Each resulting kernel receives a fresh `ClassicProblem` and
fresh sites. Recover structural and multi-kernel realization clusters.

### Phase 7 — production enumeration

Implement a lazy enumerator with private constraint propagation. Prove set equality with the Cartesian reference for
every recovered cluster, including node-first, edge-first, and randomized traversal. Candidate order has no semantics.

### Phase 8 — codec, search, evidence, and corpus

Implement the complete codec only after the schedule algebra is stable. Make search, features, pins, goldens, and
tuning evidence consume decoded typed schedules. Regenerate tracked rows once and recover all remaining corpus and CLI
clusters without a legacy parsing path.

## Completion

1. Delete the Cartesian enumerator only if exhaustive set-equivalence testing remains practical without it; otherwise
   retain it as the semantic oracle for small problems.
2. Delete the empty xfail registry, `ClassicScheduleUnavailable`, and all reconstruction-only facades.
3. Delete obsolete schedule representations, parsers, catalogs, and duplicate compatibility logic.
4. Verify every tracked schedule strictly decodes and every decoded schedule is accepted by the context.
5. Run the complete correctness, lint, realization, and exact-GPU verification lanes.
6. Update durable architecture documentation with the final model, then delete this plan.

The work is complete only when the registry is gone and there is exactly one semantic definition of classic schedule
membership, with any optimized enumerator proven equivalent to it.
