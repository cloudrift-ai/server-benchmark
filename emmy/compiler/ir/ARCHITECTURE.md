# IR Dialects

Per-dialect op definitions. A `Graph` (`compiler/graph.py`) hosts nodes
from every dialect; the population shifts as passes run. For the
top-level layer/pass picture see `compiler/ARCHITECTURE.md`.

## Dialects at a glance

| Dialect           | When populated                  | Ops                                                                                                   |
|-------------------|---------------------------------|-------------------------------------------------------------------------------------------------------|
| `base`            | always                          | `Op` (base), `InputOp`, `ConstantOp`                                                                  |
| `frontend/ir`     | after tracing / loader spelling | `LinearOp`, `MatmulOp`, `SdpaOp`, `MeanOp`, layout ops                             |
| `tensor/ir`       | after decomposition             | `ElementwiseOp`, `ReduceOp`, `ScanOp`, `GatherOp`, `ScatterOp`, `IndexMapOp`                          |
| `loop/ir`         | after fusion                    | `LoopOp` + body types (`Load`, `Assign`, `Accum`, `Write`, `Select`, `Loop`, `Axis`)                  |
| `tile/ir`         | after `lowering/tile`           | `TileOp` holding the structural root `op`, output specifications, placement, workers, knobs, a typed classic schedule, and its materialization |
| `kernel/ir`       | after `lowering/kernel`         | `KernelOp` + hardware stmts (`Tile`, `Smem`, `Sync`, `TreeHalve`)                                     |
| `cuda/ir`         | after `lowering/cuda`           | `CudaOp` (rendered `__global__` source)                                                               |

## Pure terms vs statements

Two vocabularies, and exactly one direction between them.

A **statement** (`ir/stmt/`) occupies a position in an instruction stream: it has an order, a
scope, and — for a carrier — a seed the enclosing scope has to declare. A **pure term**
(`ir/pure/`) denotes a value: it binds names, carries an algebra, substitutes and compares up to
α-renaming, and has no position at all. `Lambda`, the `Fold` term and the twist recipes (`ir/pure/twist.py` — a
twisted monoid as data, which `Fold.twist` fuses a reduce and the reduce it reads into) all live on the term side.

**A pure class is never a `Stmt` subclass and never occupies a statement position.** When a term
has to reach the instruction stream it is RENDERED into statements at the point of use — never
spliced in as one. `Fold.merge(other)` is the shape of that: the cross-partition state⊕state
combine IS the fold's stored `combine` applied with its second operand naming the partial being
merged, and it becomes `Assign` rescale temps plus one `Accum` per state component wherever the
lowering needs statements (the REG-tree merge, the cooperative tail, the cross-CTA finalize loop);
the serial step is the same derivation at the injected singleton. There is no `StateMerge` type: the term is the
`Lambda` the fold already stores,
and a rendering function is not a kind.

The invariant is what stops facts from acquiring a second home. A term that renders itself needs
no private spelling of anything the statements already carry:

- the neutral elements come from `Accum.op.identity` through the ONE identity placement
  (`Loop.render` / `StridedLoop.render`), so no `identities` field rides on the term and no
  `Init` seed is emitted beside it;
- the rescale temps arrive as ordinary `Assign`s, so the generic SSA rename, liveness and read
  counters see them with no special `deps()` channel to keep complete.

While that combine travelled as one opaque renderable stmt it needed both, and both were subtly
wrong: its temps were invisible to `rename_ssa_sequential` (patched by a uniquifying overlay in the
rewrite handler) and its seed was a second placement path that `_lift` stripped. The flash
cross-CTA finalize was numerically wrong as a result, and became correct when the combine started
arriving as ordinary statements.

**Derived reads memoize on the immutable term; pickling carries only the stored params.** A term's
expensive derived reads — the structural key, `Fold.deps`, `Lambda.free_names`, the synthesized
`loop`, the normalize fixpoint stamp, the codec's spelling tables — ride the instance
(`cached_property` entries and `structural.instance_memo` tables), which is sound because the term
never changes, and is what keeps a walk over a large fused tree linear instead of re-deriving every
subtree per ancestor. `__getstate__` strips them: every memo recomputes after transport, and an
id-keyed cache carried across processes could collide with a fresh object's id. A memo holds only
values derivable from the term — never decisions, never mutable policy.

**Tile IR stores terms, not statements.** `TileOp` holds the `Fold` term; the typed classic
schedule, materialization, output specifications, and knobs belong to `TileOp`, not the term. So `Fold` lives in
`ir/pure/fold.py` and is not a `Stmt`.

## Classic schedule model

The [schedule package](schedule/ARCHITECTURE.md) separates schedule-wide interfaces and reusable choices from concrete
implementations. `schedule/classic.py` owns the semantic model for the ordinary grid/CTA/warp/thread/register schedule.
The problem compatibility composes against is the unscheduled `TileOp` itself, paired with a target.
The `TileOp` assigns one stable integer node id to each Fold identity and one `(consumer, operand)` edge site to every
consumer operand position, so a shared producer is scheduled once while each use receives an independent transport
choice. It also derives each node site's projection or reduction view, and each contraction's schedule-independent
`ContractionFacts`, from the Fold alone; target facts cannot affect whether a site is a projection, reduction, or
contraction-capable reduction, nor what its K axis, cone seam, producer, or fragment need are. `ClassicScheduleContext` composes compatibility
over those sites and the separately projected node and edge domains.

`Schedule` is the immutable, generic assignment of kernel, node, and edge choices. Direct work, flat raster, untiled
nodes, serial reductions, and direct edges are explicit values rather than missing fields. Choice values never carry
site identities, paths, target facts, encodings, or materialization results: `Tile` is axis-free and `Stage` contains
no slab names or resolved K chunk. The wire codec is injective: empty `TILE` means only per-cell, while a parallel
unit-register thread tile spells `f1`; `WORK` never changes the meaning of an empty node choice.
`ClassicMaterialization` separately maps accepted sites to `PlacedTile`
geometry and `ResolvedStage` transport facts. Construction enforces the value types; completeness, site scope,
node-sum agreement, worker inventory and thread limits, producer-band/TMA agreement, raster eligibility, target
choice availability, and current per-contraction transport agreement need the problem and are enforced by
`ClassicScheduleContext`. Every enumeration and decode leaf crosses that complete-assignment boundary exactly
once before search or lowering can observe it. A validated leaf retains its canonical codec row; inspection and
materialization reuse that row and typed assignment instead of repeating the compatibility walk. Encoding an arbitrary
schedule remains a validating public boundary.

Graph reconstruction is the staged exception forced by the wire dependency: a private codec step parses typed schedule
values and checks their canonical spelling, those values identify the separately encoded materialization sites, and
constructing the complete `TileOp` then performs the one context validation. Parsing alone is never an acceptance
boundary.

Kernel, node, and edge domains are independent projections of static offers; none reads another selected choice.
Their Cartesian product is the definition of the candidate space. For immutable schedule restriction `c`, unscheduled
Fold program `p`, and target `t`, Algorithm 1 is exactly:

    D(p, t) = K(p, t) × ∏ N(p, t, node) × ∏ E(p, t, edge)
    Algorithm 1(c, p, t) = {a ∈ D(p, t) | extend(c + p + t, a) succeeds}

The one `ClassicScheduleContext` is the immutable `c + p + t` prefix. It owns restriction and compatibility state and
composes each node and its incident edges, followed by the kernel factor, through `extend`. The generic enumerator
never unpacks `c` or imports classic scheduling. The context retains derived physical-axis and fragment-seam facts outside the choice values. Domain membership uses immutable indexes; local support is derived
only after the context has selected one node and its incident edge values, so a precise restriction does not construct
the rest of the relation. Production may prune only prefixes whose `c + p + t` state proves they have no completion.
The literal reference enumerator remains the oracle. Bounded-product checks and traversal-order tests require every
traversal order to produce the same complete set; the lowering implementation must satisfy that product contract.

A composed step — flash's `Σ Q·K` ahead of its `Σ_j P·V`, split-K's sliced contraction — used to be
the argument for `Stmt`-hood: it has to appear at a POSITION in the emitted step stream. It does not
need to be a statement to get there. The tree already carries it: a composed node is an entry in
`operands`, and its position is produced by the derivation — `Fold.lower` places every term of the
tree at the shallowest scope binding its free coordinates, operands ahead of the term that reads
them. A loop-invariant scale `Load` ahead of attention's score contraction is that rule at work, not
a special case: it reads no coordinate the reduce loop binds, so it lands ahead of the loop even
though the cone that reads it rides inside. The only place terms become statements is `Fold.lower()`.

`Fold` does keep a small structural protocol whose names it shares with `Stmt` — `nested()` for its
children, `rewrite()` for α-renaming, and `defines()` for its result names.
These are term operations spelled the same way so one canonicalizer and one deep walk serve both
vocabularies; they are not statement behaviour, and `Fold` has no `render`. Impure computation stays
in Loop IR until total lift; there is no impure `Lambda` construction path.

**`nested()` is the STATEMENT protocol, and it deliberately does not reach a Fold's operand edges**
— it yields the lift body, and nothing at all for a contraction, whose algebra is meant to read as
edges rather than body deps. Every walk built on it (`Body.iter`, and so `Body.loads` /
`Body.writes`) therefore answers for a fully flattened stream only. The STORED tree is not one:
its operand edges are terms, and a statement walk over a lift body alone silently under-reports
everything beneath them.
Ask `loaded_buffers` instead whenever the answer must cover what a consumer of the STORED tree will
reach — the kernel materializer walks that tree, so anything deciding a node's graph inputs has to
see what it sees. Asking the lowered view there is what let a cut declare fewer inputs than the
kernel it produced went on to read.

## Invariants by stage

- **Frontend → tensor** (after `decomposition`): `LinearOp`, `MatmulOp`,
  `SdpaOp`, `MeanOp`, and the layout ops are gone. Only
  `ElementwiseOp`, `ReduceOp`, `IndexMapOp`, scan/gather/scatter, plus
  boundaries survive. (The broadcast-explicit invariant for
  `ElementwiseOp` inputs lives in `compiler/ARCHITECTURE.md`.)
- **Tensor → loop** (after `fusion`): only `LoopOp` + boundaries.
  Tensor-IR ops survive only *inside* `LoopOp.body` as `Assign.op` or
  `Accum.op` (`ElementwiseOp` only — `ReduceOp` is not a valid body
  op; reductions are `Accum` statements inside a reduce `Loop`). `LoopOp` construction orders a free-loop chain by
  the row-major coordinate depth in its boundary writes; axis spelling is only the fallback when output storage does
  not totally order the chain. The resulting geometry, rather than source names, reaches Tile IR placement.
- **Loop → tile** (after `lowering/tile`): `LoopOp` nodes are replaced by
  `TileOp` holding the structural-IR root `op` directly (`tile/ir` — one `Fold` kind), structural
  placement, one accepted site-indexed `Schedule`, and separate `ClassicMaterialization`
  facts. A kernel's structure is read from each node's derived classification, not a Python kernel
  type. `010_lift` lifts the `Fold` tree (the loop nest reconstructed on demand; a `Loop` carries no
  annotation, it folds iff its body carries an `Accum`; the algebra is the term's own
  `lift` / `(init, combine)`) with an UNMAPPED `Placement`;
  The tile schedule maps the free axes onto the grid and decides the reduce `Reduce` via the single
  `REDUCE` codec knob (`g<n>` cta / `coop` (its width in `WORK`) / `r<n>` reg; the
  decision hierarchy = env pin > the deploy evidence hierarchy, and nothing
  else — there is no default partition). The knob is ephemeral — resolved here
  into the typed assignment's `Reduce`; the combine stays the `Fold` node's stored program. Any static
  `PLANAR` / `TWISTED` reduce is cooperation-eligible (degenerate
  `sum`/`max`/`mean` AND twisted online-softmax, scalar AND
  full-row outputs), and the schedule enumerates the serial fold beside every
  band the reduce extent can feed, whatever the grid measures.
- **Tile → kernel** (after `lowering/kernel`): `TileOp` materialized to
  `KernelOp` whose body is a `Tile` (the thread-grid decode) over the
  lowered op tree. A cooperative `Reduce` lowers the reduce as a
  `StridedLoop` (lane-strided fold) + the derived algebra-generic
  cross-thread combine (`_factor.emit_combine`, reading the fold node's
  stored combine → `WarpShuffle` /
  `Smem`+`Sync`+`TreeHalve`, multi-component for a twisted fold) +
  the projection (a full-row output sweep distributed across the coop
  lanes, a scalar output guarded to lane 0); the `Tile` gains the coop
  lane axis and `block_threads = coop`. A **symbolic reduce axis**
  (dynamic `seq_len`) is supported — the `StridedLoop`'s `< seq_len`
  bound is the runtime-extent mask (idle lanes fold the identity; no
  ceil-div / clamp) and the `Dim` name is threaded as a runtime `int`
  arg. Cross-CTA reduction splits are structural choices in `030_cut`. A symbolic FREE axis
  (dynamic grid), strided rows, and the tensor-core `warp_tile` are reserved future tiers.
- **Kernel → CUDA** (after `lowering/cuda`): `KernelOp` replaced by
  `CudaOp` carrying rendered source.

Multi-output ABI order always comes from `Node.buffer_names()`: matcher population reorders a body-carrying op's
input/output maps to the graph ports after body normalization, Loop execution returns outputs in that order,
`010_lift` copies every port onto Tile IR, and Kernel/CUDA lowering renders every `OutputSpec` and output pointer.
Independent body placement may reorder sibling writes without changing the ABI.

`Op.source` is the rewrite-chain predecessor — the engine's
`_apply_one` stamps it on every 1:1 in-place rebind, so a fully
lowered `CudaOp` carries the full chain back to its originating
`LoopOp` (`cuda.source.source.source`) without any rule needing to
pass it explicitly. The base-class field is keyword-only and
`compare=False`, so subclass positional construction and equality
keep working unchanged. `source` is excluded from
`Graph.structural_key` and from the variant key (`identity_key(with_io=True, with_knobs=True)`) — kernels rendered
along different lowering paths still dedup in the tuning cache.

**Stmt subclasses are `@dataclass(frozen=True)`** — every concrete Loop-IR
/ Tile-IR / Kernel-IR statement (`Loop`, `Cond`, leaves, `Tile`, `Smem`, `Sync`,
`CpAsyncCopy`, `TmaDescriptor`, …) is immutable + hashable. `Body` is a `tuple[Stmt, ...]`
subclass, so a full body tree hashes structurally end-to-end. This makes
`Body.structural_key()` and any other bodies-as-cache-keys path work
without a try/except fallback for unhashable stmts. To "edit" a frozen
Stmt, return a fresh instance via `dataclasses.replace(stmt, field=value)`;
`__post_init__` coercions use `object.__setattr__`. Ops, by contrast,
are frozen and unhashable — rewrites replace the op and rebind its graph node. Op fields stored inside Stmts (e.g.
`Assign.op`) must be lightweight value objects (e.g. `ElementwiseImpl`,
not `ElementwiseOp`) so the surrounding Stmt's hashability isn't poisoned.

## `base.py`

Cross-cutting root. Imported by every dialect, imports nothing from
them.

| Symbol          | Role                                                                           |
|-----------------|--------------------------------------------------------------------------------|
| `Op`            | Base class. Subclasses implement `infer_output_shape` and `forward` (numpy).   |
| `InputOp`       | Sentinel: graph input tensor. Value supplied by the executor.                  |
| `ConstantOp`    | Sentinel: weights / scalar constants. Scalars carry `value`; tensors carry `source_path` / `source_shape` / `source_dtype` (the safetensors / `nn.Module` address) plus `load_ops` — a chain of frontend ops applied at bind time by the loader. `source_parts` is the multi-source alternative: `(path, shape)` pairs the loader reads and concatenates along axis 0 before running the chain. `source_graph` is the N-source bind record (`032_fold_constant_subgraphs`' collapsed static cone): a mini-graph whose external leaves name source paths and whose scalar leaves carry values — the loader binds/evaluates it through the NumPy backend, then runs the chain. Exactly one of `source_path` / `source_parts` / `source_graph` is set on a loadable constant. |
| `_keepdim_axis` | Shape helper shared by `ReduceOp` (tensor) and `MeanOp` (frontend).            |

## `expr.py`

Shared expression sublanguage used by every IR layer: `Load.index`,
`Write.index`, `SelectBranch.select`, `IndexMapOp.coord_map`,
`StridedLoop.start`/`step`, `Cond.cond`, etc. Imports nothing from
other IR files.

| Symbol                                                                       | Role                                                                     |
|------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| `Var`, `Literal`, `BinaryExpr`, `Builtin`, `FuncCallExpr`, `TernaryExpr`, `CastExpr` | Expression nodes. Each has `eval(env) → value/ndarray`, `pretty()`, `substitute(mapping)`, `free_vars()`. |
| `_ExprOps`                                                                   | Mixin: Python operator overloading for expression building; default `NotImplementedError` for `pretty`/`substitute`/`free_vars`. |
| `Expr`                                                                       | Union type alias.                                                        |
| `PLACEHOLDER_PREFIX`, `placeholder`, `is_placeholder`                        | Convention for output-coord placeholders in coord maps.                  |

## `frontend/ir.py`

Ops captured directly from PyTorch. Every one has a decomposition rule
under `pipeline/passes/frontend/decomposition/`; after that pass none of these
remain.

| Group         | Ops                                                                                                      |
|---------------|----------------------------------------------------------------------------------------------------------|
| Layout-only   | `TransposeOp`, `ReshapeOp`, `SliceOp`, `CatOp`, `UnsqueezeOp` — rewrite to `IndexMapOp`.                 |
| Compound math | `LinearOp`, `MatmulOp`, `SdpaOp`, normalization/reduction ops — rewrite to elementwise + reduce chains. |

## `tensor/ir.py`

Minimal IR fusion consumes. `IndexMapOp` is the unified layout-only op;
it replaces the frontend layout ops via `coord_map` expressions.

| Symbol                               | Role                                                           |
|--------------------------------------|----------------------------------------------------------------|
| `ElementwiseOp`                      | Per-element scalar function (`add`/`mul`/`where`/`exp`/`sin`/`cos`/…); unary `pad` is the frontend's exact zero-width identity. |
| `CastOp`, `BitcastOp`                | Numeric conversion and same-width bit reinterpretation.       |
| `RangeOp`                            | Static one-dimensional integer sequence.                       |
| `ReduceOp`                           | Collapse one axis via associative binary op.                   |
| `ScanOp`                             | Cumulative variant of reduce.                                  |
| `GatherOp`, `ScatterOp`              | Data-dependent reads / writes.                                 |
| `IndexMapOp` + `IndexSource`         | Unified layout-only op over `Expr`.                            |

`RangeOp`, `CastOp`, and `BitcastOp` are generic value operations, not checkpoint-format operations. Their current
consumer is static reconstruction algebra, so `032_fold_constant_subgraphs` removes them before Loop lifting. A
future runtime consumer may add ordinary lifting for the same semantics without changing checkpoint ingestion.

Op metadata (arity / `commutative` / `associative` / `identity` /
`has_identity` / `selecting` / `semiring_product`) lives on `ElementwiseImpl` in
`ir/elementwise.py` — the single source of truth shared across elementwise,
reduce, scan, and accumulator use sites. The algebraic traits are what
reassociation gates (split-K, cooperative tree-combine) query instead of matching
op names. The per-op trait *properties* (`op.semiring_product` — is this op a `⊗`
in some semiring) and the binary method `⊗.distributes_over(⊕)` (does this product
distribute over that reduce — the `_SEMIRING` table, only `(+, ×)` today) live on
`ElementwiseImpl`; the module's op-name-free **role queries** the planner /
atom-cell matchers ask round them out: `reduce_canon` (alias →
base combine, `sum` → `add` …) and the `_REDUCE_SPELLING` registry
(`reduce_spelling`) — the single op-keyed table
behind the four sites that used to switch on the reduce op name (`Accum.render`'s
`+=` / `*=` / `fmax` / `fmin`, `kernel/ir._binary_combine_expr`, and
`ReduceOp.forward` / `ScanOp.forward`'s numpy reductions). `op.selecting` (the
max/min family) drives the init-placement dtype choice. `op.decodes` names the storage
dtype an op is the decode cast for (the f8 family today) — the trait the tile binding
arm's factor hoist queries instead of matching op names.
Non-ufunc scalar functions whose arity cannot be read from NumPy declare it in the same module; ternary `where` is
the current example. Its condition and both value operands are explicitly broadcast before the elementwise node.

## `loop/`

One `LoopOp` = one GPU kernel described as an SSA program over named
iteration axes. Free vs reduce is inferred from body structure — a
`Loop` is a reduce Loop iff its body holds a carrier — `Accum` or its
tensor-core form `Mma`. `is_reduce` (and axis threading and the other
carrier-agnostic checks) test exactly that `isinstance(s, (Accum, Mma))`
tuple — there is no shared base class; the carriers are plain `Stmt`s
that happen to share the reduce-surface methods. `Accum` / `Mma` expose
`associative` / `commutative` / `has_identity` traits (`Accum` forwards to its scalar
`op`; `Mma` reports the additive-fold constants). A reduce `Loop` carries no annotation — it
folds iff its body carries an `Accum` — and NO algebra payload (the fold's ⊕ lives on the `Fold` node's
stored `combine`). Commutativity
is unused — split/reorder legality is a future cooperative-tier concern, recorded
structurally when it returns.

**The algebra is in the term, not a tag.** There is no stored / derived `AlgebraKind` and no op-tree node zoo. The
stored tile IR has exactly **ONE node kind**, `Fold` — `reduce(⊕) ∘ map(f)` in the λ-foldMap spelling:

- an OPTIONAL iteration `axis` (`None` = the zero-axis node) — a NAME, derived: the lift's first param when there is a
  combine. A term carries no extent at all: the coordinates it reads are the lift's trailing params, and the extent
  and window of every axis, bound or read, live in the kernel's AXIS TABLE, `TileOp.axes` (the free axes, each
  reduce axis, a split's slice and partition, a sweep). `Fold.lower` takes the table whole — a reduce loop reads its
  axis from it, and only the closed program opens the free coordinates' loops — and every kernel-side reader asks
  `Sched.axis_of` rather than the term. So a term is its function, whatever domain it is evaluated on; a sum over
  128 and one over 256 are one term under two tables, like a slab under two M.
- a pure `lift` `Lambda` `λ(k, v₁…vₙ) → S` — the element's SINGLETON state (ι is spelled in the lift;
  softmax's is `(x, 1)`);
- the monoid's flat `(init, combine)` fields — ONE program, whose results ARE the fold's accumulator names;
- a symmetric tuple of `operands` — the CLOSED inputs, each an edge, bound POSITIONALLY to the lift params. The
  params are the term's OWN names (`Fold.bindings` pairs each with its edge and component); nothing above an edge
  reads how the edge spells its results until the term is rendered — `Fold.applied` is the lift with the binding
  applied, and `step` / `lower` / the projection-region reader emit that form, so a lowered body reads producer
  names throughout while a rewrite that swaps an operand touches no name at all.

**`Map` and `Contraction` are DERIVED READINGS, not stored kinds.** Each is a reading of the stored params:
`axis is None` for the projection, `as_contraction()` (a `ContractionView`, or `None`) for the bilinear one, beside
`as_slab()` and `as_reduction()`. A reading cannot be constructed, subclassed or annotated, which is the point —
there is no type to dispatch on and no second place for a fact to live.

- A ZERO-AXIS fold is what `Map` was: no iteration and no monoid, its `lift` IS the per-cell projection. So
  softmax's normalize and RMSNorm's are one kind composed at two depths.
- The BILINEAR shape — operands `(b₀, a, b₁…)` under a `multiply` lift with a componentwise-additive
  combine — is what `Contraction` was, exposing `a` / `channels` / `b_trans` off `operands`. The `⊗` and the
  additive fold `Accum` appear in the DERIVED `Fold.loop`, never as stored loop syntax.
- Every ROLE derives from arity (`Fold.role`, never stored): `FREE` with no axis, `TWISTED` off the combine's
  claiming family, `CONTRACTION` off the bilinear reading alone, `PLANAR` otherwise. `ops.head` reaches the node
  through the projection wrapper. Scheduling reads these facts directly from the Fold tree; `Fold.lower()` is
  reserved for callers that consume Loop IR.
- A SCAN is a fold with a per-step `observe` — a pure `λ(axis, *state)` run after each combine whose fresh results
  only boundary output writes consume. Observation makes the
  stream order-visible, so an observed fold schedules as the serial fold only.

`Fold.lower(bound, stores)` flattens the term to the loop nest: a plain loop for every free coordinate the caller
left unbound, outermost the one the most terms share, the reduce loop of each term innermost, every term placed at
the shallowest scope binding its free coordinates, and each boundary store right after the term defining its value. Loops carry NO algebra and no annotation, so the derived nest depends
only on what is stored, which is what makes every identity of the term a digest of its lowered body — there is no
separate term hasher.
The `TileOp`'s body identity is the canonical digest of the nest `lower()` derives (the body is the
term's normal form); the variant key (`identity_key(with_io=True, with_knobs=True)`) folds the schedule-free body
identity with the knobs; and the deploy join key (the deploy identity (`identity_key(with_io=True)`), over
`TileOp.loop_body`) adds the io fingerprint, so term re-spellings and cluster-sibling ops that lower alike share
schedule evidence.
`Fold.deps()` exposes names captured outside the lift params, including captures reached recursively through operand
edges. A contraction deliberately hides its pure lift body from generic nested-body walks, so this direct dependency
surface is what keeps an operand's captured statistic ordered before the contraction that reads it. A read walk
STOPS at this rollup — the **`Stmt.deps_deep` trait** (a `Stmt`-protocol member beside `pure`, conservative `False`
default; `Fold` opts in) tells `_member_reads` that `deps()` already answers for the whole subtree, scope-correctly.
Re-walking the lift's flat namespace cannot see its params, so an operand-supplied name read inside the lift leaked
out of every enclosing lambda as a phantom capture: an operand-supplied name is not an enclosing capture.

A reduce is a contraction not by "two loads" but by the genuine algebra — the lift ⊗
**distributes over** the fold ⊕ (`multiply` over `add`; *not* `add` over `add`, a sum of two
operands) and exposes two distinct free-axis operand roles (`x[m, k]·x[m, k]` is a squared reduce,
not a contraction). Tile IR canonicalization constructs that form from a flat Fold when the
semiring and operand roles prove it; the mma atom tier reads the resulting Fold operands.

**The `Algebra` bundle is retired** — the stored term keeps exactly ONE spelling of ⊕, the
`Fold` node's flat `(init, combine)` pair, and everything else derives where it is consumed.
`Lambda.componentwise` builds a plain fold's combine and `Lambda.components` reads the shape back off any
stored combine (the componentwise op vector, or `None` for a twisted program — no family annotation); the `Fold`
rewrite handler renames the combine through `Lambda.rename` in lockstep with the body, and `Fold.canonical`
renumbers the combine's own names (its second operand, its temps) after the term's, so how a fold spelled its
accumulators never reaches the form. The state⊕state combine's one statement realization is the term's
own `Fold.merge(other)`, of which `Fold.step` is the instance at the injected singleton; the kernel
materializer reads the algebra through `Fold.as_reduction()` (the `ReductionView`: states, the
second operand's names, the terms, the componentwise op vector or `None` for a twisted combine) and
`merge` for cross-thread partitions. A *degenerate* fold is a plain `sum`/`max`/`mean` reduce; a
*twisted* one is online-softmax; a contraction's algebra is the degenerate algebra of its additive
fold.

The neutral element IS stored, as `Fold.init` — a monoid is `(S, ⊕, e)`, and a term that kept only
`combine` would be storing a semigroup while calling it a monoid. What is NOT stored is any
emitter's use of it: a degenerate fold dissolves into its `Accum`s and takes each fold's seed from
its `op.identity`, and a twisted fold's merge derives its own (`Fold.merge` spells the combine as
`Accum`s, never reading the stored `init`'s `−inf`). So `init` is algebra the term owes
its own definition, not a value the lowering path consults — which is why removing it would change every
`structural_key` and, with it, every variant key (`identity_key(with_io=True, with_knobs=True)`) used by tune DB
measurement replay and the cubin cache, in exchange for a field nothing reads.

**The twisted combine — a recipe, never hand-authored on a term.** Transport of structure: a monoid `(·, e)`
conjugated by a bijection ψ gives the twisted combine `x ⊕ y = ψ(ψ⁻¹(x) · ψ⁻¹(y))`, associative because the base
monoid is. A **recipe** (`ir/pure/twist.py`) states exactly that — the base's componentwise ⊕ per state, its
per-element lift, ψ and ψ⁻¹ — and beside the definition stores what conjugation does not give stably: one pattern per
channel (the per-element map a dependent reduce's lift must spell, over ROLES — `exp(s − g)` for a denominator,
`exp(s − g)·v` for an expectation, `(s − g·c)²` for Welford's deviation), what each state is at the singleton (`1`,
`v`, `0`), any state the two-pass form never had (Welford's count and running mean), and the fused ⊕ in its stable
spelling: two lambdas over roles for an open channel count (softmax's pivot advance and per-channel rescale, one
recipe for softmax and flash attention alike) or one lambda over every state pair (Welford's fixed carrier
`(sum, count, mean, M2)`). `Recipe.program(states)` instantiates either over a fold's state names by renaming, and
the definition certifies the data: the program is the conjugate of the base on random states, the seeds are the base
identities under ψ⁻¹, the injections are the lift seen through ψ. `Fold.twist(recipe)`
fuses a reduce onto the reduce it reads, found among its operands: the pivot's state is the lift param bound to it,
the score is the sub-cone of the lift alpha-equal to the pivot's own per-element map (operand for operand, through a
projection's
components), and what remains, in role order, must equal a channel's pattern by canonical form. A click gives the
role-to-name map and the recipe instantiates itself by renaming; no recipe names a term's variables. Online softmax
and flash attention are one recipe: the expectation channel joins by the same call, the pivot then being the fused
fold. **Example** — the online-softmax carrier: state `(m, d)`, partial `(score, 1)`, identity `(−inf, 0)`, merge
`m_new=max(m,s); d=d·exp(m−m_new)+exp(s−m_new); m=m_new`.

**The λ-foldMap primitives** (`ir/pure/lam.py`) — the finished algebra vocabulary the tile IR
stores against (see the tile-lowering ARCHITECTURE for the storage story). `Lambda(params, body, results)` is the ONE
binder kind over the reused stmt vocabulary — a `Body` of PURE stmts only (ANF ≙ a let-chain), validated in
`__post_init__` via the **`Stmt.pure` trait** (declared on the `Stmt` interface, conservative `False` default;
`Load`/`Assign`/`Select` and the structural `Fold` node opt in; `Accum`/`Write`/`Init`/`Loop`
never do — no
isinstance whitelist), with results-defined checked there too and α-invariance by canonical renumbering
(`Lambda.canonical` — free names never renumbered). A term is closed over its coordinates by construction — values
arrive through operand edges, and only the enclosing iteration axes are read from outside — so `Fold.canonical`
(and `Lambda.canonical` for a lambda) is the one cross-scope equivalence the Tile canonical forms and the lowering
passes (cone sharing, twisted-pair recognition, seam value clustering) all consult.
`Lambda.__post_init__` installs a
dependency-safe body order and commutative argument order, so these context-independent storage invariants do not
belong to `Fold`, `TileOp`, or the structural-key path. Contraction operand roles live on Fold edges, so sorting a
commutative product's arguments does not change them. Formation is strict: a kernel's writes ride
`TileOp.output_specs`, and synthesized split-reduce loops remain Loop IR until the new kernel re-enters total lift. A
result
may be a bare
`float` literal — ι is spelled in the lift (softmax's singleton
is `(x, 1)`). The TRUE monoid is the flat `(init, combine)` pair stored directly on the `Fold` (the `Monoid` wrapper
class dissolved at 1r) — ONE program, `combine : S × S → S` a pure `Lambda` whose
results carry the fold's REAL accumulator names; the serial streaming step is NEVER stored (it derives as combine
specialized at the singleton), so update-vs-combine consistency holds by construction. `Lambda.componentwise`
builds a plain fold's combine (DEGENERATE is the derived `Lambda.components()` shape reading, not a storage
arm). A `Fold` carries NO
precision: accumulator dtype is a KERNEL-IR fact, stamped on the lowered `Accum` by the Init-placement pass, and a
reduce `Loop` arriving with a typed `Accum` is not canonical input to total lift. A twisted monoid's combine is a
recipe's program, recognized by canonical form (`Fold.twist`), never by a stored family name;
`tests/compiler/ir/pure/test_twist.py` pins its associativity on random states.

### `loop/ir.py` — LoopOp types

| Symbol                       | Role                                                                                                              |
|------------------------------|-------------------------------------------------------------------------------------------------------------------|
| `Axis`                       | Named iteration variable (`name`, `extent`). Defined in `ir/axis.py`, re-exported here. Carries an optional `window` (`Window` — the `parent` axis this one is a slice of, plus a cross-CTA slice's absolute `base`/`bound`): the ONE windowing concept, read by the realizer and the mask machinery alike; `source_axis` is the derived compat read (`window.parent`). Excluded from equality / hashing. |
| `LoopOp`                     | One kernel. Stored field: `body` (nested `Loop` tree). Computed: `axes`, `loads`, `accums`.                       |
| `Load`                       | Body-form external read: `name = load(input)[index...]`. `input` matches the producing graph node's id.           |
| `Assign`                     | SSA body stmt: `name = op(args)` with `op: ElementwiseImpl`.                                                      |
| `Accum`                      | Reduce accumulator: `name = op(name, value)` inside a reduce `Loop`. Initialized to its op's identity. ``axes`` lists the reduction axis names — propagated through Sigma renames (including σ-splits via `Expr.free_vars()`); the escape-analysis helper derives cross-thread cooperativity from ``axes ∩ enclosing ThreadTile.axes``. |
| `Init`                       | Explicit `<dtype> name = identity;` seed at this scope (`name` + scalar `identity` + `dtype`). Used for a carried state's seed (one per component), emitted above the streaming `Loop`. Scope-bound (never hoisted); shadows a deeper same-named `Accum` init. |
| `Write`                      | Write an SSA value to output at `index`.                                                                          |
| `Select` + `SelectBranch`    | Coord-predicated binding (replaces the old Mux).                                                                  |
| `Loop`                       | Serial iteration block: `axis` + nested `body`.                                                                   |
| `StridedLoop`                | Strided iteration (`start`, `step`) — cooperative thread-stride loop reused by Tile/Kernel IR.                    |
| `Cond`                       | If/else block over an `Expr` predicate.                                                                           |
| `Stmt`                       | Base class — every body statement subclasses it. Leaves and control-flow nodes live in `ir/stmt/`.               |

Body walkers: `iter_body(body)` (pre-order; powers `for s in loop_op`),
`map_body(body, fn)` (transformer), `Stmt.rewrite(rename_ssa, sigma)`
(per-stmt copy with SSA rename + Expr substitution),
`Stmt.pretty(indent)` (rendered lines for kernel dumps; block stmts
recurse via `pretty_body`).

CUDA scalar rendering goes through `stmt.base.op_to_expr`. Boolean masks retain the historical f32 SSA convention,
so Torch's `bitwise_not` spelling renders as logical zero-test (`mask == 0`); explicitly bool-stamped values use the
same semantics. Integer complement is not inferred from that name and fails closed until it has a typed consumer.
The optional readable-source fold keeps a single-use `Assign` named when any argument's stamped dtype differs from
the result dtype, so the target-aware `Assign.render` path remains responsible for conversions such as
`__half2float`.

Dependence cones (`ir/stmt/body.py`): `Body.backward_cone(roots)` builds a `Cone` —
the subset of the body's immediate stmts closed under SSA dependence (a wrapper joins as a unit; internally-bound
axes excluded), plus `external_reads`, the names read from outside (axis vars and enclosing/sibling scopes alike).
Construction never fails: unresolved names are data, and chaining scope levels means seeding the next level's
`backward_cone` with the previous one's `external_reads`. `Body.defs_die_at(members, roots=…, allowed=…)` is the
matching escape check (may the cone be cut out, with only the designated consumers reading its roots?). This is
the shared substrate behind the rules that slice cones (the demoted-operand producer cut in
`lowering/tile/030_cut`) — eligibility judgments stay in the rules, per
`pipeline/passes/ARCHITECTURE.md`.

`rewrite` has two distinct rename channels that must stay disjoint:
`rename_ssa` carries **SSA-name** renames, `sigma` carries **axis**
substitutions. `Load`/`Write` index exprs apply *both*
(`_rename_ssa_vars_in_expr(sigma.apply(e), rename)`) so an indirect
(gather) index Var gets renamed exactly once. Putting the same name in
both maps renames it twice — and if the two passes form a chain (e.g.
`x → in5` and a pre-existing `in5 → in26`) the double application
collapses it transitively, silently wiring a gather to the wrong row.

`rewrite` is also **not scope-aware**: it descends into every nested body and maps a stmt's own bindings as well as
its reads, while `Assign` / `Load` / `Select` names bound inside a `Loop` / `Cond` body are scoped to that body. The
two are safe together only for a whole-subtree renumbering (`rename_ssa_sequential`). When the rename instead comes
from *dropping* a binding — load dedup, CSE — an inner scope that merely re-uses the dropped name's spelling is a
different variable, and renaming it both redeclares the survivor inside the scope and rewires the inner arithmetic to
the outer value. `passes.rename_free(stmt, alias)` is the hygienic form: it prunes the alias of whatever each child
scope re-binds before descending. `normalize.dedup_loads` applies the same rule while threading its own per-scope
environment. σ has the same hazard with axis names, which collide across a tree by design (a cone statistic's axis
may spell the same as the enclosing contraction's): `fold.subst_free(stmt, sigma)` is σ's hygienic form — it stops at
a `Loop` / reducing `Fold` binder that re-binds a substituted name, and is what the smem compute fill substitutes
cell coordinates through.

### `ir/stmt/normalize.py` — structural canonicalization

Pure `body → body` passes run from `LoopOp.__post_init__` so every
constructed `LoopOp` (including intermediate fusion results) is
canonicalized before validation:

- `topo_sort_siblings` — stable Kahn reorder so SSA defs precede their uses
  within each body (fixes splicer-produced use-before-def).
- `drop_size_one_free_axes` — inline extent-1 free Loops.
- `drop_size_one_reduce_axes` — collapse a canonical extent-1 reduction to its single monoid update. This includes
  decode-softmax values that fusion hoists into the enclosing scope; copy-alias elimination then removes the identity
  update before total reduction lifting.
- `canonicalize_free_axis_order` — sort outer free Loops by their row-major position in boundary writes, so output
  storage geometry rather than axis spelling decides the nest. When the writes cannot totally order the chain, axis
  names provide a deterministic fallback. A cross-CTA partition coordinate occupies the workspace's leading index,
  so the same rule keeps it outside the axes it partitions without a naming convention.

- `eliminate_copy_aliases` — drop `y = copy(x)` Assigns.
- `unify_sibling_reduce_axes` — rename sibling reduce Loops whose
  reduce-axis Load positions overlap on any `(source, dim)` pair so
  they share one canonical axis name (softmax's max + sum sweeps; the
  two matmul reductions in `silu(x@Wg) * (x@Wu)` that both index `x`
  at the same K slot). Union-find groups all transitively-overlapping
  Loops at one scope.
- `merge_sibling_reduce_loops` — concatenate sibling reduce Loops that
  share `axis.name` / `extent` into one Loop body. Gated on disjoint
  SSA defs across the two halves, the second body not reading any name
  the first body defines (blocks softmax-style sequential reduces
  where sum-exp reads `acc_max`), and no between-stmt def consumed by
  the second loop. Eliminates the duplicate K traversal in patterns
  like `silu(x@Wg) * (x@Wu)`; subsequent normalization collapses the
  duplicate `x` loads, and the lowering passes stage both weight tensors
  symmetrically.
- `split_invariant_divides` — rewrite `divide(x, y)` into
  `reciprocal(y) + multiply(x, recip)` when `y` is loop-invariant
  w.r.t. some axis `x` depends on, so the rcp can hoist out of the
  inner loop and the per-iter cost drops from XU divide to FMA
  multiply.
- `hoist_loop_invariants` — pull loop-invariant Assigns out of reduce
  Loops. Effect summaries are cached on immutable statements, and `Body.axis_dependencies` retains only the axes
  reachable from each definition. Long SSA chains therefore remain linear in definitions × loop depth instead of
  materializing the quadratic full SSA dependency closure.
- `dedup_loads` — after expression simplification, keep one `Load` for
  each identical `(input, index)` read in a scope and rewire its users.
  This is canonicalization for every Loop / Tile body, not a fusion
  profitability decision.
- `rename_ssa_sequential` — cosmetic: `Load` names become `in0, in1,
  …`, Assign/Select `v0, v1, …`, Accum `acc0, …`, in definition order.
  Records renames only in the SSA channel (`rename`), never the axis
  channel (`sigma`) — see the `rewrite` two-channel rule above; an SSA
  name leaking into `sigma` double-renames indirect (gather) indices.
- `canonicalize_buffer_names` — rename `Load.input` / `Write.output` to
  `b0, b1, …` in encounter order. Off by default (buffer names bind to
  graph nodes) — opt in via `normalize_body(..., canonical_buffers=True)`.
  Used by `Body.structural_key()` for dedup queries where buffer identity
  doesn't matter.
- `sort_commutative_args` — sort `Assign.args` for commutative ops
  (`add` / `multiply` / `maximum` / `minimum`) so two bodies that
  differ only by argument order land in the same canonical form.
  Runs last so the sort key is the post-rename canonical SSA / buffer
  names.

`Body.structural_key()` re-runs `normalize_body(self, hoist=False,
canonical_buffers=True)` and joins `pretty_body`'s line list — a
`cached_property` returning the canonical text rendering. Two bodies
that differ only by SSA / axis names, commutative-arg order, or
external-buffer names produce the same key. Use it as a dict key /
set member when deduping candidate bodies in a search.

### `ir/expr.py` — Expr simplification

`simplify` (called inside `normalize_body`). Generic bottom-up Expr rewriter:
constant folding, algebraic identities, range-based comparison folding
(`(k0 > 2047 ? 2047 : k0) < 0 ? 0 : k0` → `k0`). `SimplifyCtx`/`Interval`
track integer ranges from axis extents (`axis.extend_simplify_ctx` pushes
each loop axis into the ctx). `SimplifyCtx.bounds` additionally tracks a
*symbolic* exclusive upper bound per var (`i < seq_len`) so a modulo by a
non-literal divisor folds — `i % seq_len → i` when `i`'s loop extent is
`seq_len` (`_mod_below_divisor`). This collapses the delinearized seq
coordinate `((i*stride + feat) / stride) % seq_len` that compose-indexmaps
emits back to `i`, the symbolic-shape counterpart of the literal-divisor
`_div_mod_decompose` cleanup (a static `seq_len` already constant-folds it).
Symbolic-extent axes get `[0, sentinel]` ranges (non-negativity for the inner
`(i*c + …)//c → i` div fold) instead of being dropped.

`_div_mod_decompose` also sees through a division standing in its way (`A / d`
decomposes by `n` once `A` decomposes by `n·d`), and through a sum whose one
addend is a clean multiple of the divisor (the partner then owns the whole
remainder, so restating it as `n·(x/n) + x%n` suffices). Together those separate
a sub-byte-packed operand address: an NVFP4 weight spells `((row·K + k)/2) %
(K/2)`, holding the row axis inside a division, and the decomposition puts the
row on the quotient side where a consumer asking "does this index still mention
the row outside a div/mod" can see it. The `loop/canonicalize` axis re-fusion is
that consumer, and its answer decides whether a packed matmul binds a
contraction at all.

### `loop/splicer.py` — LoopOp merger

The machinery `pipeline/passes/loop/fusion/010_merge_loop_ops.py` calls to splice a DAG of `LoopOp` nodes. `Sigma`
(from `ir/sigma.py`) is the axis-substitution bookkeeping threaded through the merge.

`splice_graph` resolves each internal Load through `Graph.producer(buffer)`, so primary and secondary output buffers
use the same path. Every graph output supplies an explicit `(loop tag, Write.output)` root. Separate terminal loops
therefore seed one worklist; its one binding table shares equal upstream demands across output ports instead of
inlining a shared producer per consumer. The single-sink convenience form still derives the unique terminal loop and
selects all its Writes. Every `_NotSupported` carries a reason string, logged at DEBUG by `splice_loops` —
`compile -vv` shows which pattern a rejected edge hit.

Before dependency reconstruction, `splice_graph` finds output equivalence clusters: single-owner copy chains ending
at a terminal graph output, with the same dtype and element count and an exact symbolic proof that the source and
destination coordinates are related by a reshape and axis permutation. Equal element count alone is insufficient;
slices, broadcasts, and conversions remain ordinary edges. The proof compares each source coordinate with one
mixed-radix digit of the destination's dense flat address, then composes those inverse layouts across the chain. The
splicer retargets the computed source's `Write` through that inverse and removes the copy roots from reconstruction.
This preserves the producer's loop geometry through terminal reshape/transpose chains without enumerating the output
domain.

A `Write` that observes an `Accum` inside that accumulator's own reduce scope is an ordered prefix output. The
splicer refuses that shape whether it is the merged root or a producer edge: dependency reconstruction would freshen
the reduce loop and move the `Write` after it, changing every prefix value into the final reduction. Such an
effectful inner loop is not valid input to total lift.

Before splicing, `loop/lifting/090_spell_store_rounding` turns a public store that narrows an `Accum` into an ordinary
typed `copy` statement. A decomposition may route that accumulator through one transient, shape-only buffer before a
pass-through LoopOp writes the public buffer; that direct load retains the accumulator's implicit f32 dtype, so the
same rule spells its public conversion. An actual `Assign` computation over private reduction state remains untyped:
normalization and softmax therefore retain f32 internal state rather than narrowing it at an inferred projection edge.
`splice_graph` then preserves the explicit conversion through its ordinary statement path and reconstructs no dtype
boundary from source provenance or graph topology.

Construction is bounded per statement: the dedup table shares each `(stmt, emit scope, σ)` binding, and in
every legitimate splice no single statement takes more than a handful of distinct bindings. A recurrence-shaped
region — each stage re-demanded under compositions of σs, DeepSeek-V4's 20-iteration Sinkhorn chain being the live
case — multiplies bindings per stage instead of deduplicating, and such a merge cannot be constructed at any budget.
The first statement past the cap stops the splice. The doom is structured (`UnfusableStmt` names the offending
loop) and surfaced to the fusion pass on request, which drops that loop plus its downstream closure from the region
and retries — so one doomed chain costs only itself, not every other merge in its region. This is a termination
bound, not a fusion-quality gate: placement still owns every cut on a merge that CAN be built, and the refusal must
stay cheap because the greedy policy re-runs fusion on every candidate graph it prices.

Each splice memoizes `Expr.free_vars()` by expression identity while placing dependencies. Sigma expressions remain
live for the splice, and identity avoids both repeated coordinate-tree walks and the recursive structural hashing a
global cache would require; the memo is discarded with the splicer.

`Sigma` computes canonical expression text once for each initial substitution. Derived substitutions created by
`extend` and `restrict` retain the applicable canonical entries from their parent, so dependency placement neither
reformats deep coordinate trees nor retains duplicate canonical strings.

### `loop/runner.py` — C++ JIT executor

`execute_loop_op_cpp(loop, input_arrays, out_shapes)` renders the LoopOp body to a C++ source string and JIT-compiles
it in-process via cppyy / Cling (cached by the rendered source), then calls it with raw pointers to the input and every
output array. One output returns an array; multiple outputs return a tuple in the operation's graph-populated ABI
order. Each Write's own scope determines its output shape, so independent sibling nests may reuse axis names with
different extents. This powers `LoopOp.forward`, so post-fusion graphs run through the default `Backend.run` topo-walk
like any pre-fusion graph.

### `loop/builder.py` — fluent construction

`LoopBuilder` constructs merged `LoopOp` bodies for the fusion splicer without spelling out every `Loop(Axis(…))`
nest. Construction is mutable — descent is a dict lookup per scope level and a prepend is an append to a
reverse-ordered list — and the immutable body is materialized once by `finish()`. Rebuilding the tuple tree per
insert is quadratic in program size and re-runs each level's `Loop` construction normalization per insert, which is
invisible on small graphs and decisive on large ones. Fresh SSA names retain the lowest available deterministic
suffix while a per-hint monotonic cursor ensures each occupied suffix is tested at most once; the used-name set
remains authoritative when another hint claims a future suffix.

## `tile/`

Tile IR stores the complete inner loop nest as one tree of `Fold` terms. The Loop IR boundary peels the outer parallel
axes, converts every reduction from its explicit `Accum` statements, and leaves each nested reduction in the same
position inside its parent lambda. A root zero-axis Fold holds the per-cell statement sequence. An output loop's
per-cell projection becomes a zero-axis term evaluated over the sweep axis — a sibling operand of the root — and its
writes live in `TileOp.output_specs` as sweep specs.

A nonzero-axis Fold exposes its combine result names through `Fold.defines()`, so later sibling statements and outer
folds may consume its result without hoisting it to an operand edge. `Fold.loop` mechanically lowers the tree back to
the corresponding nested Loop IR.

The total-lift invariant is that no raw inner `Loop` survives. A bilinear term orients itself at formation, its
shared argument in the contraction's canonical A slot; `TileOp.__post_init__` then applies the tree-wide
canonicalization — an identity projection dissolves into its operand, and same-value cones become one shared object. Scoped lambda equivalence is an analysis over the
canonical Folds. A separate pre-scheduling rewrite fuses every reduce that reads a reduce into the twisted carrier a
recipe recognizes — `(maximum, denominator, expectations…)` for the exp family — hoisting the factors constant along
the axis (attention's `1/l`) out of the fold first; softmax and masked or unmasked SDPA are arity variants of one
recipe, not separate matchers. Placement and cross-CTA split are structural phases before site construction. Classic scheduling
classifies the resulting Fold tree, assigns each node once and every consumer operand edge independently, then stores
one complete typed assignment on `TileOp.schedule`. Unsupported shapes remain unmapped; scheduling never annotates or
rewrites the Fold tree.

See [`tile/ARCHITECTURE.md`](tile/ARCHITECTURE.md) for the exact storage and boundary contract.

## `kernel/`

### `kernel/ir.py` — fully-scheduled kernel form

Reuses `Tile` + leaf stmts from Tile IR; adds hardware primitives
materialized from scheduling decisions. `KernelOp` carries the body
directly (no separate AST class).

| Symbol             | Role                                                              |
|--------------------|-------------------------------------------------------------------|
| `KernelOp`         | Graph-op wrapper around a `Tile`-rooted body. One per kernel.     |
| `Smem`             | `__shared__` array allocation (name + dtype + extents + optional `align`). Swizzled TMA operand slabs align to their full swizzle atom (`8 × swizzle_width` B: B128→1024, B64→512, B32→256) — the coordinate-only `ldmatrix` XOR only reproduces the hardware's absolute-address swizzle when the base zeroes the swizzle's source-address bits; non-swizzled TMA keeps 128 B, fp16 16 B. `pack_smem` (the shared pool packer used by `smem_bytes` and the renderer) pads each buffer to `max(sizeof(dtype), align)` so the static-vs-dynamic gate and the launch-time dynamic-pool size agree. |
| `Sync`             | `__syncthreads()` barrier.                                        |
| `TreeHalve`        | Cross-thread tree reduction over a smem buffer.                   |
| `RegFragment`      | Per-thread `mma.sync` register array declaration, zero-initialized for C. The established m16n8k16 layout uses A/B/C counts 4/2/4 for f16/f16/f32; the Volta m8n8k4 layout carries explicit 2/2/8 counts because one instruction realizes four PTX cells arranged as one logical 16×16 tile. Carries instruction shape, dtype, and an optional explicit register count. The opaque `nvcuda::wmma` nodes remain retired. |
| `LdmatrixLoad`     | Load one operand into a `RegFragment`. The m16n8k16 layout can use `ldmatrix.sync.aligned.m8n8.x{4,trans}.b16` from shared memory or a global-memory-direct gather with the same lane map. SM70 has no `ldmatrix`, so the Volta m8n8k4 layout uses its cooperative gather for both address spaces: a global pointer for the direct path or a shared-slab pointer after synchronous-copy staging; its four computation groups duplicate the appropriate A or B quadrant. `b_trans=True` marks a `[N, K]` weight and selects the corresponding transposed gather. Guards clamp M/N lanes and zero masked K elements in both layouts. A 1-byte staged slab (`byte_slab=True`) has no `ldmatrix` below sm_100a and drains through the cooperative gather too; when it also carries a `scale_buffer` the slab holds PACKED PAIRS (an NVFP4 weight — one byte, two K elements, and one scale per k block in that companion slab), and the loader decodes both codes through the f16 value table and scales them as it fills the fragment. |
| `MmaSyncPtx`       | Inline PTX for either `mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32` on the Volta fragment layout or the established `mma.sync.aligned.m16n8k16.row.col.{f32,f16}.{f16,bf16}.{f16,bf16}.{f32,f16}` family. The renderer includes only the selected family's prelude, so SM70 never parses newer `ldmatrix` or m16n8k16 assembly. The BLOCK-SCALED fp4 form (`m16n8k64`, `kind::mxf4nvf4`) additionally carries `sfa_frag` / `sfb_frag`: both multiplicands are packed e2m1 pairs and the instruction applies one ue4m3 scale per 16 K elements itself, so the call passes those two scale registers where the others repeat the accumulator. Its data fragments reuse the fp8 byte loaders — the k64 4-bit lane map is the k32 8-bit one, over a row of K/2 bytes — leaving only the scale loaders new. It assembles only for the arch-suffixed consumer-Blackwell target, which the plan requests through `KernelSpec.arch_specific` (the flag TMA also sets). |
| `FragmentPromote`  | Fold a packed f16-accumulate C fragment into its f32 shadow fragment and rezero it (`emmy_mma_promote_f16acc`: PTX `cvt.f32.f16` + add per element) — the chunked-accumulation promote pairing the f16-acc `MmaSyncPtx`. The mma chain accumulates in f16 at full rate; each K chunk (the staged bk slab, every `_F16ACC_STEPS` gmem-direct atom steps) folds into the f32 shadow, bounding the f16 rounding to one chunk while the store/epilogue read f32. |
| `FragmentLoad`     | Load one scalar tensor element per C-fragment element using the shared fragment layout's absolute row/column coordinates. The residence evaluator uses it when a Fold Lambda reads a materialized source at fragment residence. |
| `FragmentSelect`   | Coordinate-predicated uniform values over one C fragment. It substitutes each fragment element's absolute row/column through the shared fragment layout, then uses scalar `Select` branch order and casts exactly; fragment-valued or per-cell branches fail closed at the lifting boundary. |
| `RegStore`         | Layout-aware per-lane epilogue store: four C elements for m16n8k16 or eight elements covering the four Volta output quadrants for m8n8k4. Stores f32 directly or downconverts to f16. Optional `RegEpilogue` loads and pointwise chains are evaluated at each element's own coordinates; guarded tails predicate every load and store. |
| Shared from `tile` | `Tile` (launch geometry); from `ir/stmt/`: `Loop`, `StridedLoop`, `Load`, `Assign`, `Accum`, `Write`, `Select`, `Cond`. |

## `cuda/ir.py`

| Symbol    | Role                                                                        |
|-----------|-----------------------------------------------------------------------------|
| `CudaOp`  | Graph-op carrying `kernel_source`, `kernel_name`, `arg_order`, `grid`, `block`, `smem_bytes`, `zero_outputs`, `comment`. Produced by `pipeline/passes/lowering/cuda` (renders the `KernelOp` body to a `__global__` source string). |

## Graph as the single program form

There is no separate program type. A `Graph` is the execution plan:
node ids are buffer names, `node.output.shape` is the buffer shape,
`graph.topological_order()` is the launch order, and
`graph.inputs` / `graph.outputs` / `ConstantOp` membership gives each
buffer its role (input / output / constant / scratch).
