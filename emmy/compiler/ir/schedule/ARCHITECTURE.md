# Schedule model

The schedule package separates the shared contract from concrete schedule implementations. `Schedule` is an immutable
hardware-execution plan, `ScheduleContext` owns the problem and target facts that decide compatibility, and
`ScheduleCodec` is the strict canonical wire boundary. A future implementation, such as a warp schedule, supplies all
three without adding type branches to the classic model.

Schedule values never mutate. A schedule implementation returns a new value for every update, so enumeration,
compatibility checks, serialization, and measurements can safely retain the original assignment. A context receives
one complete schedule and returns its compatibility verdict. A codec encodes accepted values, decodes and validates
complete rows, and exposes one canonical key order.

Reusable leaf choices such as `Work`, `Tile`, `Reduce`, `Stage`, and `Raster` are independent of a concrete assignment
model. They carry neither problem sites nor target facts. A concrete schedule may compose them, but does not add a
second spelling or mutate them during validation.

## Classic schedule

`ClassicSchedule` is a frozen assignment of kernel, node, and edge choices. Its node and edge mappings are read-only;
`replace`, `with_kernel`, `with_node`, and `with_edge` return new `ClassicSchedule` values. `ClassicScheduleContext`
is the sole complete-assignment compatibility authority, and `ClassicScheduleCodec` is its strict wire boundary.

For immutable schedule restriction `c`, unscheduled Fold program `p`, and target `t`, classic enumeration follows
Algorithm 1 exactly:

    D(p, t) = K(p, t) × ∏ N(p, t, node) × ∏ E(p, t, edge)
    Algorithm 1(c, p, t) = {a ∈ D(p, t) | c.accepts(a) ∧ accepts(p, t, a)}

The cut pass explores structural cuts before this schedule enumeration begins. The enumerator carries `c` intact and
consults it only for a complete assignment. It never unpacks `c` or lets it alter an independent domain. Prefix
pruning may use only incompatibility facts derived from `p` and `t`, and every traversal order must retain the same
compatible subset of the Cartesian product.
