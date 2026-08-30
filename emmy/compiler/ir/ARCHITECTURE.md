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
α-renaming, and has no position at all. `Lambda`, the `Fold` term, the monoid vocabulary (`M` / `component_ops` /
`rename_combine` / the foldMap spec oracle), the monoid-family registry (`family_of` — every stored combine is
claimed by a registered family, componentwise or twisted, by generator-output equality) and the exp-family combine
generators all live on the term side.

**A pure class is never a `Stmt` subclass and never occupies a statement position.** When a term
has to reach the instruction stream it is RENDERED into statements at the point of use — never
spliced in as one. `algebra.merge_stmts(combine, other)` is the shape of that: the cross-partition
state⊕state combine IS the fold's stored `combine` applied with its second operand naming the
partial being merged, and it becomes `Assign` rescale temps plus one `Accum` per state component
wherever the lowering needs statements (the REG-tree merge, the cooperative tail, the cross-CTA
finalize loop). There is no `StateMerge` type: the term is the `Lambda` the fold already stores,
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

**Tile IR stores terms, not statements.** `TileOp` holds the `Fold` term and pure projection regions; the schedule
slices, output specifications and knobs are the `TileOp`'s, not the term's. So
`Fold` lives in `ir/pure/fold.py` and is not a `Stmt`.

## Classic schedule model

`classic_schedule.py` owns the semantic model for the ordinary grid/CTA/warp/thread/register schedule. A
`ClassicProblem` contains only an unscheduled `Fold` tree and target. Its immutable `SiteIndex` assigns one stable
`NodeSite` to each Fold identity and one `EdgeSite` to every consumer operand position, so a shared producer is
scheduled once while each use receives an independent transport choice. Site classification reads only the Fold at a
node site; target facts cannot affect whether that site is a projection, reduction, or contraction-capable reduction.

`ClassicSchedule` is an immutable, typed assignment of kernel, node, and edge choices. Direct work, flat raster,
untiled nodes, serial reductions, and direct edges are explicit values rather than missing fields. Choice values never
carry site identities, paths, target facts, encodings, or materialization results: `Tile` is axis-free and `Stage`
contains no slab names or resolved K chunk. The wire codec is injective: empty `TILE` means only per-cell, while a
parallel unit-register thread tile spells `f1`; `WORK` never changes the meaning of an empty node choice.
`ClassicMaterialization` separately maps accepted sites to `PlacedTile`
geometry and `ResolvedStage` transport facts. Construction enforces the value types; completeness, site scope,
node-sum agreement, worker inventory and thread limits, producer-band/TMA agreement, raster eligibility, target
choice availability, and current per-contraction transport agreement need the problem and are enforced by
`ClassicScheduleContext.accepts`. Every enumeration and decode leaf crosses that complete-assignment boundary before
search or lowering can observe it.

Kernel, node, and edge domains are independent projections of static offers; none reads another selected choice.
Their Cartesian product is the definition of the candidate space. Static support retains derived physical-axis and
fragment-seam facts outside the choice values, and `ClassicScheduleContext.accepts` filters the complete product by
that one compatibility relation. The literal reference enumerator remains the oracle. The production Fold walk may
prune incompatible prefixes, but bounded-product checks and traversal-order tests require its complete leaf set to
equal the reference set exactly.

A composed step — flash's `Σ Q·K` ahead of its `Σ_j P·V`, split-K's sliced contraction — used to be
the argument for `Stmt`-hood: it has to appear at a POSITION in the emitted step stream. It does not
need to be a statement to get there. The tree already carries it: a composed node is an entry in
`operands`, and its position is produced by the derivation — `_twisted_derived_step` PLACES each
inline-node edge before the first stmt that reads its bound name (lift body or merge), and
`splice_operands` applies the same first-use rule to every other edge. Placement, not prepending, is
what lets a step whose pure prologue precedes its producer (a loop-invariant scale `Load` ahead of
attention's score contraction) re-derive to the program it was read from. `Fold.loop` passes that
mixed term/stmt sequence to `_flatten_nodes` as a plain tuple; the only place terms become statements
is `Fold.lower()`.

`Fold` does keep a small structural protocol whose names it shares with `Stmt` — `nested()` for its
children, `rewrite()` for α-renaming, and `defines()` for its result names.
These are term operations spelled the same way so one canonicalizer and one deep walk serve both
vocabularies; they are not statement behaviour, and `Fold` has no `render`. Impure computation stays
in Loop IR until total lift; there is no impure `Lambda` construction path.

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
  placement, one accepted site-indexed `ClassicSchedule`, and separate `ClassicMaterialization`
  facts. A kernel's structure is read from each node's derived classification, not a Python kernel
  type. `010_lift` lifts the `Fold` tree (the loop nest reconstructed on demand, each reduce `Loop`
  carrying its `AxisRole` — the ONLY loop annotation; the algebra is the term's own
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
  cross-thread combine (`_factor.emit_combine`, reading the fold node
  through the lowering-side `Reduction` view → `WarpShuffle` /
  `Smem`+`Sync`+`TreeHalve`, multi-component for a twisted fold) +
  the projection (a full-row output sweep distributed across the coop
  lanes, a scalar output guarded to lane 0); the `Tile` gains the coop
  lane axis and `block_threads = coop`. A **symbolic reduce axis**
  (dynamic `seq_len`) is supported — the `StridedLoop`'s `< seq_len`
  bound is the runtime-extent mask (idle lanes fold the identity; no
  ceil-div / clamp) and the `Dim` name is threaded as a runtime `int`
  arg. The cross-CTA split (`035_split_reduce`), `reg` fold, a symbolic FREE
  axis (dynamic grid), strided rows, and the tensor-core `warp_tile`
  are reserved future tiers.
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
`Graph.structural_key` and from `Op.cache_key` — kernels rendered
along different lowering paths still dedup in the tuning cache.

**Stmt subclasses are `@dataclass(frozen=True)`** — every concrete Loop-IR
/ Tile-IR / Kernel-IR statement (`Loop`, `Cond`, leaves, `Tile`, `Smem`, `Sync`,
`CpAsyncCopy`, `TmaDescriptor`, …) is immutable + hashable. `Body` is a `tuple[Stmt, ...]`
subclass, so a full body tree hashes structurally end-to-end. This makes
`Body.structural_key()` and any other bodies-as-cache-keys path work
without a try/except fallback for unhashable stmts. To "edit" a frozen
Stmt, return a fresh instance via `dataclasses.replace(stmt, field=value)`;
`__post_init__` coercions use `object.__setattr__`. Ops, by contrast,
are NOT frozen — the engine mutates `op.source` / `op.knobs` / `op.inputs` /
`op.outputs` post-construction. Op fields stored inside Stmts (e.g.
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
`op`; `Mma` reports the additive-fold constants). A reduce `Loop` also carries its
scheduling `AxisRole` (`loop.role`) — its ONLY annotation, derived when a Fold lowers; the loop holds NO algebra
payload (the fold's ⊕ lives on the `Fold` node's stored `combine`). Commutativity
is unused — split/reorder legality is a future cooperative-tier concern, recorded
structurally when it returns.

**The algebra is in the term, not a tag** (`ir/pure/algebra.py` — the consolidated algebraic
vocabulary). There is no stored / derived `AlgebraKind` and no op-tree node zoo. The stored tile IR has
exactly **ONE node kind**, `Fold` — `reduce(⊕) ∘ map(f)` in the λ-foldMap spelling:

- an OPTIONAL iteration `axis` (`None` = the zero-axis node);
- a pure `lift` `Lambda` `λ(k, v₁…vₙ) → S` — the element's SINGLETON state (ι is spelled in the lift;
  softmax's is `(x, 1)`);
- the monoid's flat `(init, combine)` fields — ONE program, whose results ARE the fold's accumulator names;
- a symmetric tuple of `operands` — the CLOSED inputs, each an edge, bound POSITIONALLY to the lift params.

**`Map` and `Contraction` are DERIVED READINGS, not stored kinds.** Each is a constructor returning a
`Fold` plus a PREDICATE answering the reading: `axis is None` for the projection, `is_contraction(x)` for the
bilinear one. A predicate cannot be constructed, subclassed or annotated, which is the point — there is no
type to dispatch on and no second place for a fact to live.

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
  only boundary output writes consume (`Fold.observed`, a structural probe like `composed`). Observation makes the
  stream order-visible, so an observed fold schedules as the serial fold only.

`Fold.lower()` flattens the term to the loop nest: `Fold.loop` reconstructs the annotated reduce `Loop`
from the stored params, splicing each operand's body before the first read of its bound param. Loops carry NO
algebra — a `Loop` holds only its `AxisRole` — so the derived nest depends only on what is stored, which is
what makes kernel identity the α-invariant TERM HASH (`Fold.structural_key`) rather than the lowered nest.
`Fold.deps()` exposes names captured outside the lift params, including captures reached recursively through operand
edges. A contraction deliberately hides its pure lift body from generic nested-body walks, so this direct dependency
surface is what keeps an operand's captured statistic ordered before the contraction that reads it.

A reduce is a contraction not by "two loads" but by the genuine algebra — the lift ⊗
**distributes over** the fold ⊕ (`multiply` over `add`; *not* `add` over `add`, a sum of two
operands) and exposes two distinct free-axis operand roles (`x[m, k]·x[m, k]` is a squared reduce,
not a contraction). Tile IR canonicalization constructs that form from a flat Fold when the
semiring and operand roles prove it; the mma atom tier reads the resulting Fold operands.

**The `Algebra` bundle is retired** — the stored term keeps exactly ONE spelling of ⊕, the
`Fold` node's flat `(init, combine)` pair, and everything else derives where it is consumed.
`ir/pure/algebra.py` is the IR core only: `M` (the componentwise free constructor),
`component_ops`/`degenerate` (the DEGENERATE-vs-TWISTED shape test on a stored combine — `None` ⇒
the exp family; no family annotation), `rename_combine` (the SSA-rename lockstep, applied by the
`Fold` rewrite handler — a twisted program regenerates over the renamed state), and the
denotational foldMap spec oracle, and `merge_stmts` — the state⊕state combine's one statement
realization, a function over the stored combine rather than a second term kind. Fold lowering uses
that realization when an identity lift receives complete states, including a cross-CTA finalize.
The kernel materializer reads the same algebra through
`pipeline/passes/lowering/_reduction.Reduction` (`names`, `state_b`, `combine_states`, and
`merge_stmts`) for cross-thread partitions. A *degenerate* fold is a plain
`sum`/`max`/`mean` reduce; a *twisted* one is online-softmax; a contraction's algebra is
the degenerate algebra of its additive fold.

The neutral element IS stored, as `Fold.init` — a monoid is `(S, ⊕, e)`, and a term that kept only
`combine` would be storing a semigroup while calling it a monoid. What is NOT stored is any
emitter's use of it: a degenerate fold dissolves into its `Accum`s and takes each fold's seed from
its `op.identity`, and a twisted fold's streaming merge regenerates its own (`_reduction` reads the
generated merge's `Accum`s, never the stored `init`'s `−inf`). So `init` is algebra the term owes
its own definition, not a value the lowering path consults — which is why removing it would change
every `structural_key`, and with it every `Op.cache_key` the tune DB's measurement replay and the cubin
cache are keyed on, in exchange for a field nothing reads.

**The twisted combine — generated, not hand-authored.** Transport of structure: a monoid `(·, e)`
conjugated by a bijection ψ gives the twisted combine `x ⊕ y = ψ(ψ⁻¹(x) · ψ⁻¹(y))`. Generation
(`ir/pure/carrier.py` — `exp_combine_states` / `exp_merge` over `(names, terms)`) builds the naive
`ψ∘base∘(ψ⁻¹×ψ⁻¹)` combine — associativity inherited from the base monoid for free — then a
per-family stabilizer rewrites it to the numerically-stable form (distribute the ψ-rescale, fuse
exponentials, fold identities, DCE/CSE) and a structural certificate asserts every surviving
`exp` has a `≤ 0` argument. Recognition calls the generators directly (`exp_merge` for the dissolved streaming body,
`exp_combine_states` for the stored combine) — a twisted `Fold`'s combine IS the generator's
program (the formation invariant `Fold.__post_init__` asserts), and the component ROLES are shape-derived off
the terms: component 0 the pivot (score), a literal-`1.0` term a denominator, a value term an
expectation (the online-softmax pairing builds an expectation channel per joined value fold — a fused
softmax·V region carries `(m, d, o…)`). **Example** — the
online-softmax carrier: state `(m, d)`, partial `(score, 1)`, identity `(−inf, 0)`, merge
`m_new=max(m,s); d=d·exp(m−m_new)+exp(s−m_new); m=m_new`.

**The λ-foldMap primitives** (`ir/pure/lam.py` / `ir/pure/algebra.py`) — the finished algebra vocabulary the tile IR
stores against (see the tile-lowering ARCHITECTURE for the storage story). `Lambda(params, body, results)` is the ONE
binder kind over the reused stmt vocabulary — a `Body` of PURE stmts only (ANF ≙ a let-chain), validated in
`__post_init__` via the **`Stmt.pure` trait** (declared on the `Stmt` interface, conservative `False` default;
`Load`/`Assign`/`Select` and the structural `Fold` and `ProjectionRegion` nodes opt in; `Accum`/`Write`/`Init`/`Loop`
never do — no
isinstance whitelist), with results-defined checked there too and α-invariance by canonical renumbering
(`Lambda.canonical` — free names never renumbered). `Lambda.__post_init__` invokes `ir/pure/normalize.py` to install a
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
specialized at the singleton), so update-vs-combine consistency holds by construction. `M(op…)` is the free
componentwise pair constructor (DEGENERATE is the derived `component_ops(combine)` shape predicate, not a storage
arm; `rename_combine` carries the rename lockstep incl. the twisted regeneration rule). A `Fold` carries NO
precision: accumulator dtype is a KERNEL-IR fact, stamped on the lowered `Accum` by the Init-placement pass, and a
reduce `Loop` arriving with a typed `Accum` is not canonical input to total lift. A twisted
monoid's
combine is the exp/LSE generator's program, selected structurally, never by a stored family name. The module also
ships the executable SPEC: `eval_lambda` / `foldmap_eval`, the ~20-line denotational evaluator the agreement
(`⟦tree⟧ == lowered loop`) and ASSOCIATIVITY property tests in `tests/compiler/ir/stmt/test_lambda_monoid.py` run
against.

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

Dependence cones (`ir/stmt/body.py`): `Body.backward_cone(roots)` / `Body.forward_cone(seeds)` build a `Cone` —
the subset of the body's immediate stmts closed under SSA dependence (a wrapper joins as a unit; internally-bound
axes excluded), plus `external_reads`, the names read from outside (axis vars and enclosing/sibling scopes alike).
Construction never fails: unresolved names are data, and chaining scope levels means seeding the next level's
`backward_cone` with the previous one's `external_reads`. `Body.defs_die_at(members, roots=…, allowed=…)` is the
matching escape check (may the cone be cut out, with only the designated consumers reading its roots?). This is
the shared substrate behind the rules that slice cones (the demoted-operand producer cut in
`lowering/tile/035_split_reduce`) — eligibility judgments stay in the rules, per `pipeline/passes/ARCHITECTURE.md`. The
`classify_fragment_epilogue` walk (`ir/pure/algebra.py`) deliberately does NOT use it: it is a single pass
interleaving reduce-scope flags with its negative-form blocker reporting, a different operator than the cone's
any-dep taint.

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

An `Accum` stores its value into the producing tensor before a distinct frontend operation loads that tensor. When the
declared tensor dtype differs from the accumulator dtype (implicitly f32 until Kernel IR), `splice_graph` keeps that
boundary as a typed `copy` alias. Nodes created by decomposing and rewriting one frontend operation share the ultimate
`Op.source` object and may reconstruct their private edge directly. A private output stays recognizable even when its
consumer fragment already mixes origins: it is absent from the ultimate frontend source's declared outputs. Missing
or unrelated source chains preserve the conversion. Equal-dtype reductions and non-`Accum` producers keep the
ordinary untyped alias, so fusion does not duplicate a conversion already carried by the defining statement.

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
position inside its parent lambda. A root zero-axis Fold holds the per-cell statement sequence. Pure sibling output
loops become `ProjectionRegion` terms, and their writes live in `TileOp.output_specs`.

A nonzero-axis Fold exposes its combine result names through `Fold.defines()`, so later sibling statements and outer
folds may consume its result without hoisting it to an operand edge. `Fold.loop` mechanically lowers the tree back to
the corresponding nested Loop IR.

The total-lift invariant is that no raw inner `Loop` survives. `TileOp.__post_init__` then applies general contextual
canonicalization, including maximal pure operand-cone factoring into semiring contractions and closed-child
extraction. Commutative products place their shared argument in the contraction's canonical shared operand slot;
overlapping producer cones become one multi-result operand edge. Scoped lambda equivalence is an analysis over the
canonical Folds. A separate pre-scheduling rewrite joins equivalent maximum and exp-weighted sibling Folds into the
general `(maximum, denominator, expectations…)` twisted carrier, including when contraction canonicalization has
nested the statistics inside a computed probability edge; softmax and masked or unmasked SDPA are arity variants, not
separate matchers. Placement and cross-CTA split are structural phases before site construction. Classic scheduling
classifies the resulting Fold tree, assigns each node once and every consumer operand edge independently, then stores
one complete typed assignment on `TileOp.classic`. Unsupported shapes remain unmapped; scheduling never annotates or
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
| `LdmatrixLoad`     | Load one operand into a `RegFragment`. The m16n8k16 layout can use `ldmatrix.sync.aligned.m8n8.x{4,trans}.b16` from shared memory or a global-memory-direct gather with the same lane map. SM70 has no `ldmatrix`, so the Volta m8n8k4 layout uses its cooperative gather for both address spaces: a global pointer for the direct path or a shared-slab pointer after synchronous-copy staging; its four computation groups duplicate the appropriate A or B quadrant. `b_trans=True` marks a `[N, K]` weight and selects the corresponding transposed gather. Guards clamp M/N lanes and zero masked K elements in both layouts. |
| `MmaSyncPtx`       | Inline PTX for either `mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32` on the Volta fragment layout or the established `mma.sync.aligned.m16n8k16.row.col.{f32,f16}.{f16,bf16}.{f16,bf16}.{f32,f16}` family. The renderer includes only the selected family's prelude, so SM70 never parses newer `ldmatrix` or m16n8k16 assembly. |
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
