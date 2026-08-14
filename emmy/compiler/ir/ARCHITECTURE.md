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
| `tile/ir`         | after `lowering/tile`           | `TileOp` holding the structural-IR root `op` (`tile/ir`: ONE `Fold` kind) + `place` / `work` / `workers` / `knobs` / `schedule` / `stores` — every per-node slice keyed into `schedule` by the tree-path codec |
| `kernel/ir`       | after `lowering/kernel`         | `KernelOp` + hardware stmts (`Tile`, `Smem`, `Sync`, `TreeHalve`)                                     |
| `cuda/ir`         | after `lowering/cuda`           | `CudaOp` (rendered `__global__` source)                                                               |

## Invariants by stage

- **Frontend → tensor** (after `decomposition`): `LinearOp`, `MatmulOp`,
  `SdpaOp`, `MeanOp`, and the layout ops are gone. Only
  `ElementwiseOp`, `ReduceOp`, `IndexMapOp`, scan/gather/scatter, plus
  boundaries survive. (The broadcast-explicit invariant for
  `ElementwiseOp` inputs lives in `compiler/ARCHITECTURE.md`.)
- **Tensor → loop** (after `fusion`): only `LoopOp` + boundaries.
  Tensor-IR ops survive only *inside* `LoopOp.body` as `Assign.op` or
  `Accum.op` (`ElementwiseOp` only — `ReduceOp` is not a valid body
  op; reductions are `Accum` statements inside a reduce `Loop`).
- **Loop → tile** (after `lowering/tile`): `LoopOp` nodes replaced by
  `TileOp` holding the structural-IR root `op` directly (`tile/ir` —
  one `Fold` kind) plus the root-global schedule fields
  (`place` / `work` / `workers`, with every per-node slice — `TilePlan` / `ReducePlan` / `Stage` — keyed into
  the `schedule` dict); a kernel's structure is read off the node's derived role, not a Python
  type. `010_recognize` lifts the `Fold` term (the loop nest reconstructed on demand, each reduce `Loop`
  carrying its `AxisRole` — the ONLY loop annotation; the algebra is the term's own
  `lift` / `(init, combine)`) with an UNMAPPED `Placement`;
  The tile schedule maps the free
  axes onto the grid and decides the reduce `ReducePlan` via the single
  `REDUCE` codec knob (`g<n>` cta / `coop` (its width in `WORK`) / `r<n>` reg; the
  decision hierarchy = env pin > search/prior fork > conservative
  default). The knob is ephemeral — resolved here into the schedule's
  `ReducePlan`; the combine stays the `Fold` node's stored program. Any static
  `PLANAR` / `TWISTED` reduce is cooperation-eligible (degenerate
  `sum`/`max`/`mean` AND twisted online-softmax / flash, scalar AND
  full-row outputs); the default cooperates a wide reduce feeding an
  under-occupied grid.
- **Tile → kernel** (after `lowering/kernel`): `TileOp` materialized to
  `KernelOp` whose body is a `Tile` (the thread-grid decode) over the
  lowered op tree. A cooperative `ReducePlan` lowers the reduce as a
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
  arg. The cross-CTA split (`030_split_reduce`), `reg` fold, a symbolic FREE
  axis (dynamic grid), strided rows, and the tensor-core `warp_tile`
  (incl. flash's warp tier) are reserved future tiers.
- **Kernel → CUDA** (after `lowering/cuda`): `KernelOp` replaced by
  `CudaOp` carrying rendered source.

`Op.source` is the rewrite-chain predecessor — the engine's
`_apply_one` stamps it on every 1:1 in-place rebind, so a fully
lowered `CudaOp` carries the full chain back to its originating
`LoopOp` (`cuda.source.source.source`) without any rule needing to
pass it explicitly. The base-class field is keyword-only and
`compare=False`, so subclass positional construction and equality
keep working unchanged. `source` is excluded from
`Graph.structural_key` and from `Op.cache_key` — kernels rendered
along different lowering paths still dedup in the tuning cache.

`Op.knobs` and `Op.decision_knobs` have distinct ownership. `knobs` is the realized configuration of this kernel and
participates in its tuning/cache identity. `decision_knobs` records a structural fork that changed which kernels
exist, such as `PLACE@a=cut`; the engine propagates it for decision replay, but graph serialization, structural keys,
kernel cache keys, and per-kernel schedule evidence exclude it. Search still carries the structural row on the fork
lineage, so an end-to-end measurement remains attributable to the placement decision.

**Stmt subclasses are `@dataclass(frozen=True)`** — every concrete Loop-IR
/ Tile-IR / Kernel-IR statement (`Loop`, `Cond`, leaves, `Tile`, `Smem`, `Sync`,
`CpAsyncCopy`, `TmaDescriptor`, …) is immutable + hashable. `Body` is a `tuple[Stmt, ...]`
subclass, so a full body tree hashes structurally end-to-end. This makes
`Body.structural_key()` and any other bodies-as-cache-keys path work
without a try/except fallback for unhashable stmts. To "edit" a frozen
Stmt, return a fresh instance via `dataclasses.replace(stmt, field=value)`;
`__post_init__` coercions use `object.__setattr__`. Ops, by contrast,
are NOT frozen — the engine mutates `op.source` / `op.knobs` / `op.decision_knobs` / `op.inputs` /
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
| `ConstantOp`    | Sentinel: weights / scalar constants. Scalars carry `value`; tensors carry `source_path` / `source_shape` / `source_dtype` (the safetensors / `nn.Module` address) plus `load_ops` — a chain of frontend ops applied at bind time by the loader. `source_parts` is the multi-source alternative (`merge_sibling_linears`' weight concat): `(path, shape)` pairs the loader reads and concatenates along axis 0 before running the chain. `source_graph` is the N-source bind record (`032_fold_constant_subgraphs`' collapsed static cone): a mini-graph whose external leaves name source paths and whose scalar leaves carry values — the loader binds/evaluates it through the NumPy backend, then runs the chain. Exactly one of `source_path` / `source_parts` / `source_graph` is set on a loadable constant. |
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
| `ElementwiseOp`                      | Per-element scalar function (`add`/`mul`/`where`/`exp`/`sin`/`cos`/…). |
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
atom-cell matchers / flash recognizer ask round them out: `reduce_canon` (alias →
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
scheduling `AxisRole` (`loop.role`) — its ONLY annotation, stamped by tile-lowering
recognition; the loop holds NO algebra payload (the fold's ⊕ lives on the `Fold` node's
stored `combine`, and `Fold.from_loop` reconstructs it from the body alone). Commutativity
is unused — split/reorder legality is a future cooperative-tier concern, recorded
structurally when it returns.

**The algebra is in the term, not a tag** (`ir/stmt/algebra.py` — the consolidated algebraic
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
  softmax's normalize, RMSNorm's, and flash's `divide(O, l)` are all one kind composed at two depths.
- The BILINEAR shape — operands `(b₀, a, b₁…)` under a `multiply` lift with a componentwise-additive
  combine — is what `Contraction` was, exposing `a` / `channels` / `b_trans` off `operands`. The `⊗` and the
  additive fold `Accum` appear in the DERIVED `Fold.loop`, never as stored loop syntax.
- Every ROLE derives from arity (`Fold.role`, never stored): `FREE` with no axis, `TWISTED` off the combine's
  twist family, `CONTRACTION` off the bilinear reading alone, `PLANAR` otherwise. `ops.head` reaches the node
  through the projection wrapper; `ops.reduce_loop` still returns the outermost annotated reduce `Loop`, but
  only for callers that consume a body — reading a node FACT off a synthesized nest is the inversion `ops`
  exists to prevent (`Fold.loop` splices every edge and flattens every nested node just to hand back the
  property it was given).

`Fold.lower()` flattens the term to the loop nest: `Fold.loop` reconstructs the annotated reduce `Loop`
from the stored params, splicing each operand's body before the first read of its bound param. Loops carry NO
algebra — a `Loop` holds only its `AxisRole` — so the derived nest depends only on what is stored, which is
what makes kernel identity the α-invariant TERM HASH (`Fold.structural_key`) rather than the lowered nest.

A reduce is a contraction not by "two loads" but by the genuine algebra — the lift ⊗
**distributes over** the fold ⊕ (`multiply` over `add`; *not* `add` over `add`, a sum of two
operands) and contracts ≥ 2 distinct operand buffers (`x·x` is a squared reduce, not a
contraction). Recognition stamps the `CONTRACTION` role on that form (keeping the matmul's
`Accum` a loose `Accum` rather than degenerate-folding it like a plain reduce);
The schedule gates flash structurally (a reduce loop nested inside a reduce loop); the mma
atom tier reads the operands off the annotated loop to pick the tensor-core cell.

**The `Algebra` bundle is retired** — the stored term keeps exactly ONE spelling of ⊕, the
`Fold` node's flat `(init, combine)` pair, and everything else derives where it is consumed.
`ir/stmt/algebra.py` is the IR core only: `M` (the componentwise free constructor),
`component_ops`/`degenerate` (the DEGENERATE-vs-TWISTED shape test on a stored combine — `None` ⇒
the exp family; no family annotation), `rename_combine` (the SSA-rename lockstep, applied by the
`Fold` rewrite handler — a twisted program regenerates over the renamed state), and the
denotational foldMap spec oracle; the renderable `StateMerge` stmt lives with the other stmt
leaves (`ir/stmt/leaves.py`, next to `Accum`). The lowering side reads the algebra
through ONE helper, `pipeline/passes/lowering/_reduction.Reduction` (wrap a `Fold`; `names` /
`state_b` / `twisted`, the `combine_states` re-emission, `state_merge(other)`, the finalize
`identities`, and `loop_state_head` — the loop-body read of the carried state's head), consumed
only by the kernel materializer and `030_split_reduce`. A *degenerate* fold is a plain
`sum`/`max`/`mean` reduce; a *twisted* one is online-softmax / flash; a contraction's algebra is
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
(`ir/stmt/carrier.py` — `exp_combine_states` / `exp_merge` over `(names, terms)`) builds the naive
`ψ∘base∘(ψ⁻¹×ψ⁻¹)` combine — associativity inherited from the base monoid for free — then a
per-family stabilizer rewrites it to the numerically-stable form (distribute the ψ-rescale, fuse
exponentials, fold identities, DCE/CSE) and a structural certificate asserts every surviving
`exp` has a `≤ 0` argument. Recognition calls the generators directly (`exp_merge` for the dissolved streaming body,
`exp_combine_states` for the stored combine) — a twisted `Fold`'s combine IS the generator's
program (the formation invariant `Fold.__post_init__` asserts), and the component ROLES are shape-derived off
the terms: component 0 the pivot (score), a literal-`1.0` term a denominator, a value term an
expectation — softmax is flash minus the expectation component. **Example** — flash attention's
online softmax: state `(m, l, O)`, partial `(score, value)`, identity `(−inf, 0, 0)`, merge
`m_new=max(m,s); alpha=exp(m−m_new); l=l·alpha+exp(s−m_new); O=O·alpha+exp(s−m_new)·v; m=m_new`.

**The λ-foldMap primitives** (`ir/stmt/body.py` / `ir/stmt/algebra.py`) — the finished algebra vocabulary the tile IR
stores against (see the tile-lowering ARCHITECTURE for the storage story). `Lambda(params, body, results)` is the ONE
binder kind over the reused stmt vocabulary — a `Body` of PURE stmts only (ANF ≙ a let-chain), validated in
`__post_init__` via the **`Stmt.pure` trait** (declared on the `Stmt` interface, conservative `False` default;
`Load`/`Assign`/`Select` and the structural `Fold` node opt in; `Accum`/`Write`/`Init`/`Loop` never do — no
isinstance whitelist), with results-defined checked there too and α-invariance by canonical renumbering
(`Lambda.canonical` — free names never renumbered). Formation is STRICT everywhere since 1q (the interim
`effectful_lambda` is deleted; a kernel's root stores ride `TileOp.stores`, and only the tile layer's raw-loop-IR
arm — `tile/ir._loop_ir_fn`, for the un-recognized escape / `030` finalize kernels — may hold an impure body). A result may be a bare `float` literal — ι is spelled in the lift (softmax's singleton
is `(x, 1)`). The TRUE monoid is the flat `(init, combine)` pair stored directly on the `Fold` (the `Monoid` wrapper
class dissolved at 1r) — ONE program, `combine : S × S → S` a pure `Lambda` whose
results carry the fold's REAL accumulator names; the serial streaming step is NEVER stored (it derives as combine
specialized at the singleton), so update-vs-combine consistency holds by construction. `M(op…)` is the free
componentwise pair constructor (DEGENERATE is the derived `component_ops(combine)` shape predicate, not a storage
arm; `rename_combine` carries the rename lockstep incl. the twisted regeneration rule). A `Fold` carries NO
precision: accumulator dtype is a KERNEL-IR fact, stamped on the lowered `Accum` by the Init-placement pass, and a
reduce `Loop` arriving with a typed `Accum` declines recognition (`_extract_lift`) rather than dropping it. A twisted
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
`lowering/tile/030_split_reduce`) — eligibility judgments stay in the rules, per `pipeline/passes/ARCHITECTURE.md`. The
scope-sensitive companion `lexical_free_values` preserves statement order, nested value scope, and accumulator
exports; placement and recognition use it when a global `reads - definitions` difference would incorrectly close an
open term. The
`classify_fragment_epilogue` walk (`ir/stmt/algebra.py`) deliberately does NOT use it: it is a single pass
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

### `ir/stmt/normalize.py` — structural canonicalization

Pure `body → body` passes run from `LoopOp.__post_init__` so every
constructed `LoopOp` (including intermediate fusion results) is
canonicalized before validation:

- `topo_sort_siblings` — stable Kahn reorder so SSA defs precede their uses
  within each body (fixes splicer-produced use-before-def).
- `drop_size_one_free_axes` — inline extent-1 free Loops.
- `canonicalize_free_axis_order` — sort outer free Loops by axis name.
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
  like `silu(x@Wg) * (x@Wu)`; downstream `dedup_loads` then collapses
  the duplicate `x` loads, and the lowering passes stage both weight
  tensors symmetrically.
- `split_invariant_divides` — rewrite `divide(x, y)` into
  `reciprocal(y) + multiply(x, recip)` when `y` is loop-invariant
  w.r.t. some axis `x` depends on, so the rcp can hoist out of the
  inner loop and the per-iter cost drops from XU divide to FMA
  multiply.
- `hoist_loop_invariants` — pull loop-invariant Assigns out of reduce
  Loops.
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

The machinery `pipeline/passes/loop/fusion/010_merge_loop_ops.py` calls to
splice adjacent `LoopOp` pairs. `Sigma` (from `ir/sigma.py`) is the
axis-substitution bookkeeping threaded through the merge.

`splice_graph` derives splice edges as `(node_id, node_id)` — it **assumes a
producer LoopOp's sole `Write.output` buf is its node id** (the buf-name ==
node-id invariant the whole graph maintains). A rule that emits a LoopOp whose
`Write.output` doesn't match its node id silently breaks every later fold of
that node: the edge points at a Write that doesn't exist, so the splicer raises
`_NotSupported` and the node survives as its own kernel. Rules that rename a
node must rename its body `Write.output` to match (`fusion/_helpers.py::rename_write_output`).
Every `_NotSupported` carries a reason string, logged at DEBUG by `splice_loops`
— `compile -vv` shows which pattern a rejected edge hit.

### `loop/runner.py` — C++ JIT executor

`execute_loop_op_cpp(loop, input_arrays, out_shape) → ndarray` renders the
LoopOp body to a C++ source string and JIT-compiles it in-process via cppyy /
Cling (cached by the rendered source), then calls it with raw pointers to the
input arrays. Powers `LoopOp.forward` — so post-fusion graphs run through the
default `Backend.run` topo-walk like any pre-fusion graph.

### `loop/builder.py` — fluent construction

`LoopBuilder` helper used by decomposition/fusion tests to construct
LoopOp bodies without spelling out every `Loop(Axis(…))` nest.

## `tile/`

**The layer has its own doc — [`tile/ARCHITECTURE.md`](tile/ARCHITECTURE.md)** — for the one stored kind, its
derived readings, the operand edge's two inhabitants, term identity and the tree-path codec. What follows is the
module map and the parts that touch the rest of `ir/`.

Tile IR keeps the stored term pure algebra and the schedule beside it. The layer is **one concern per module**:
`tile/ir.py` the term vocabulary (`Fold`, `Channel`, `Store`, `TileOp`), `tile/ops.py` the geometry-free compute reads
and the `Sched` accessor, `tile/path.py` the tree-path codec, `tile/_key.py` kernel identity (`structural_key`),
`tile/_dump.py` the structural dump. Loop IR → term is NOT here: `fold_from_loop` / `nodify_reduce` are a parser and
live with the passes that consume them (`passes/lowering/tile/_fromloop.py`). A `TileOp` holds the structural-IR root
`op` directly — a `Fold`, the ONE stored node kind (defined in `tile/ir.py`) — plus the root-global free→grid
`Placement` (`place`), the worker inventory (`work`) and warp split (`workers`); every per-node schedule slice
(`TilePlan` / `ReducePlan` / `Stage`) lives in the tree-path-keyed `TileOp.schedule` dict.

**One kind, three readings.** A `Fold` is an optional `axis`, the `operands`, a joint `lift` and an optional
`(init, combine)` monoid. `Map` is deleted outright — a zero-axis node (`Fold.projection(...)`) is the projection /
pointwise cell it used to name, and `.sources` / `.fn` went with it (`.operands` / `.lift`). `Contraction` is
deleted as well: `Fold.contraction(...)` is the BUILDER — it generates the bilinear lift and the additive
combine — and `is_contraction(x)` the READING, a predicate rather than a kind, with
`a` / `channels` / `b_trans` reading off `operands`. The A/B split
rides the stored operand ORDER `(b₀, a, b₁…)`, because node-locally the two are symmetric — `A[m,k]` and
`B[k,n]` each carry K plus one free axis — and telling M from N needs the placement, which lives on the `TileOp`.

**No node carries a schedule field at all**: a contraction reading is its axis + edges + generated algebra and
nothing more, so a node's `==` / `hash` / `Fold.structural_key` is its algebra — two kernels
differing only in tile key identically, and no emission path can leak a schedule into a stored term. The placement +
schedule a tier needs travels beside the node as its SCHEDULE SLICE — algebra and geometry as a `(node, tile)`
pair, never fused into one object. The **geometry is the slice's own**: a `TilePlan` carries the
`(m, n)` output axes it tiles (`axes`, bound by `.at(m, n)`) and derives the `Side` pair from them, so the tiled
CELL's reading is a function of the schedule slice alone. `axes` is `compare=False` — placement is not a search
dimension, so it never reaches `spell()`, a stamped knob row, a golden or a prior key. The `Kernel` / `TileSchedule` wrapper is gone. A kernel's structure is read
structurally off the node (`Fold.role` — every role DERIVES from arity: no axis is `FREE`, a twisted combine
`TWISTED`, the bilinear shape `CONTRACTION`, else `PLANAR`), not a bespoke Python type per schedule. The PLANAR
demotion is therefore a formation fact with no role rewrite: moving the edges inline is enough, and the same
derivation answers `PLANAR` by itself. That rewrite is `Fold.demoted()` — each edge placed before the first read of its
bound name (ties in operand order, the splice rule), a materialized `Load` verbatim and a computed edge as the
structural NODE, which `_flatten_nodes` flattens at lowering, so the derived loop is byte-identical to the hoisted
spelling's. Its one caller is the schedule's COLLAPSE term reading.

There is exactly ONE node walk over a stored term — `tile/path.py::sites` — shared by the key resolver, the
stampers, the seam enumerator and every plain "walk the nodes" reader (take `.node` off each site). `tile/ir.py`
keeps only the generic *stmt* walks the node kinds derive through (`deep_reads` / `deep_defines` /
`stmt_axis_names` / `refs_axis`); a helper used by exactly one pass lives with that pass instead (the cut's
closure predicate in `passes/lowering/tile/_cut.py`, the fragment-loader row step in `passes/lowering/_addr.py`).

The schedule type system lives at the ir root in `schedule.py` (used by both the tile IR and the kernel
materializer, so it sits beside `atom.py`, not under `tile/`) — the merge of the former
`tile/{schedule,codec,role}.py`: the schedule value types and the codec ser/de engine in one module. The role
*registry* is gone with the `WSPEC` knob it served (see the producer band below).

`ReducePlan` (`schedule.py`) is a list of `ReduceStage`s, one per hardware `Level` the reduce axis is
partitioned across, coarse→fine: `GRID` (split-K across CTAs), `BLOCK` (cooperative threads within a CTA), `REG`
(ILP register-fold), `SERIAL` (the per-thread remainder). The per-level combine `Fold` (`SHFL` lane butterfly /
`SMEM` block tree / `ATOMIC` cross-CTA finalize) is **derived** from the level (`ReduceStage.combine`), not stored
or tuned. The single `REDUCE` codec knob decides the plan schedule-side; the combine itself stays in the op
tree.

**Every codec parses and spells by hand**, and they all read the same way: one `/`-separated token string, order-free,
each field binding at most ONCE — a repeated token (`d2/cp/d3`, `sync/tma`) raises rather than letting the last one
win, since an order-free grammar gives a silent overwrite no reading the pin could have meant. No field is mandatory:
an absent token takes its default, which is what lets a codec add a field without invalidating values spelled before
it. Each parse error names its codec and offers the grammar, because the featurizers degrade on a `ValueError` and a
bad pin has to say which knob it came from. The one non-uniform value codec is the `REDUCE` `g<n>[a|k]` finalize
letter, kept inside the value so the round-trip stays byte-identical.

There was a schema engine here — `Schema` / `Field` / `decode` / `encode` — and it is gone. It served four codecs; the
site-local rewrite hand-wrote `TILE`, `REDUCE` and `WORK` (they parse *against* a `Workers`, which a generic decoder
cannot express), the `WSPEC` collapse took the fourth, and a general mechanism serving one caller is a worse
statement of that caller's grammar than the grammar itself. Since step
7 the WIRE forms are site-local: `Workers` is the kernel-global
inventory (`WORK` — `Workers.spell`/`parse`, the `+p<n>` producer band absorbing the retired per-row `WSPEC` key), and
`TilePlan.spell`/`parse` + `ReducePlan.spell`/`parse` are the worker-token-free site values the stamped rows and the
golden corpus carry — they parse **against** a `Workers`, which is why a generic decoder never fit them. There is no
second, self-contained reading: the retired embedded-worker spellings raise. The
`a:scalar` / `a:none` aliases stay pin-only vocabulary for the scalar tier.

The **producer band** is warp specialization, and it is one integer: `WarpSpec.producer_warps`, carried on an
**orthogonal** `workers: WarpSpec | None` field of the uniform schedule (`None` = uniform SIMT), **not** a union arm
— it adds a warp band over the fixed pipeline rather than replacing it. The COMPUTE consumer warps stay implicit,
sized by `TilePlan.units`. The band is DECIDED as inventory (`WORK`'s `+p<n>`, `Workers.producer`), and whether a row
may offer it at all is one predicate, `_legality.producer_transport`: a resolved **TMA** stage, un-split, on a kernel
not split across CTAs — the box copy is issued by one elected thread and lands on a slot mbarrier any thread can
parity-wait, so the fill moves warp bands freely, while cp.async's wait-group is issuing-thread-scoped and a `sync`
compute-fill has no async load half. A row the predicate refuses is never enumerated; a stamped one is materialized
by the staged K-loop (`lowering/kernel/_stage`). There is no role registry, no per-role param schema and no second
legality gate: the `WSPEC` codec that carried them went with its knob, and what a schedule can express is exactly
what an emitter reads.

Lowering has ONE spelling and it lives on the node: `tile/ir.py` `Fold.lower()`. There is no free `ops.lower`
wrapper duplicating it — which is also what keeps `tile/ir.py` free of any import back into `tile/ops.py`. Reading an
operand EDGE is likewise free-function, not per-role: `ir.operand_body(edge)` / `ir.operand_name(edge)` (an edge is an
edge, and which role it plays — A vs B — is the caller's reading of the operand ORDER, never a property of the edge,
so there is no `a_body` / `b_body` / `a_name` / `b_name` quartet on the node).

**A derived reading earns its place by deriving something.** `role`, `a` / `channels` / `b_trans`, `composed`, `loop`,
`step_stmts` all compute; a property that only renames a stored field or wraps one `isinstance` does not, and is spelled
at the call site instead — the contraction axis is `fold.axis` (there is no `k_axis` alias), a computed A is
`not isinstance(c.a, Load)` (the same spelling the B side already used), and a schedule slice is `sched.get("STAGE", n)`
(only the placement binding is a method — `Sched.placed(node, plan)`, the ONE `(m, n)` rule per site shape, with
`tile_of` its stored-slice reading and the enumeration passing candidate plans through the same door).

`Fold.lower()` returns the derived loop nest with its annotated reduce `Loop`s, the
carriers already dissolved into loose folds + the streaming `merge` at recognition. The tensor-core, cooperative-combine, staging (cp.async / TMA), and warp-specialization tiers are materialized
downstream in `lowering/kernel` against the op tree + schedule. The older tile-level `GridTile` / `ThreadTile` /
`Stage` structures were removed in the tile-IR rebuild and are being rebuilt there as the schedules return (see
`pipeline/passes/ARCHITECTURE.md`).

**The structural dump** (`tile/_dump.pretty` / `TileOp.pretty_body`, what `emmy compile --ir tile` and the `EMMY_DUMP_DIR`
`.txt` artifacts print) renders the STORED tree as a tree, never a lowered nest — the dump is where a reader meets the
term directly, so it has to show what the term IS. Each node prints its kind and stored params as labelled branches
(`operands` first, then `init` / `lift` / `combine`, with the bilinear reading labelling its edges `operand[a]` /
`operand[b<i>]` — the same `a` / `b` tokens the path codec spells, so a dump line matches a `PLACE@a` key by eye),
and every operand edge is recursed into and tagged with its inhabitant — `‹computed›`
for an inline node subtree, `‹materialized›` for a leaf gmem `Load`. **A λ-valued field labels its own branch with its
signature and nests its body two under it** — `lift:` / `combine:` / `fn:` all read the same way, so a binder is always
adjacent to what it binds; none of them ride the node header, where on a big fold the signature sat a screenful above
its stmts. The `fn:` branch is emitted even for an empty body, since the branch carries the signature and an identity
projection binds too. A λ signature carries its CAPTURE SET when
non-empty (`λ() [captures m_i__t5] -> (…)`) — the free names that are not iteration vars, the same reading the cut's
closure predicate applies (`axis_names`, relocated to `tile/ops.py` so the dump and `_cut._captured_values` share one
definition; the iteration space is the term's axes ∪ the placement's free/grid ∪ the boundary stores' sweep axes).
Without a capture set a λ reads as closed, and closure is precisely what decides whether a subtree can hoist to an
operand edge — flash's `P = exp(s − m)` captures the carrier's running max, which is why its seam is not cuttable. The
set is measured only when the owning `TileOp` is supplied; a bare term has no placement, so the annotation is omitted
rather than reporting grid coordinates as captures.

**Nothing DERIVED is printed** — not the per-cell step, not the nodes synthesized inside it, not the lowered nest.
The structure is already complete in the stored tree: the operand edges and their nesting say it, and a derived
evaluation follows from the same params, as re-derivable as `Fold.lower()`'s output. Printing one beside storage is the
inversion this layer exists to prevent, and it was the bulk of the output — measured over eight frontend kernels the
step branch restated `lift` + `combine` and contributed no schedule site on seven of them (flash 50 → 28 term lines,
softmax 31 → 21). `--ir loop` is where a reader goes for a body.

The caller facts that live BESIDE the term get their own regions — `place` / `work` / `wspec` above it, `schedule` and
`stores` below. Schedule slices annotate a node as `⟨TILE=… REDUCE=… STAGE=…⟩` only when the owning `TileOp` is
supplied (`pretty(op, tile=…)`), read through `Sched` and the path codec, so nothing on the term can carry one. The one
slice whose site is DERIVED (flash's synthesized PV, `TILE@pj`) has no stored node to annotate: `ops.unplaced_slices`
reports it and it prints in the `schedule` region, rather than reconstructing the derived node inside the term to hang
it on. That region is empty — and therefore absent — for every kernel whose sites are all stored.

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
| `FragmentPromote`  | Fold a packed f16-accumulate C fragment into its f32 shadow fragment and rezero it (`emmy_mma_promote_f16acc`: PTX `cvt.f32.f16` + add per element) — the chunked-accumulation promote pairing the f16-acc `MmaSyncPtx`. The mma chain accumulates in f16 at full rate; each K chunk (the staged bk slab, every `_F16ACC_STEPS` gmem-direct atom steps, or the flash streaming KV block) folds into the f32 shadow, bounding the f16 rounding to one chunk while the store/epilogue read f32. |
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
