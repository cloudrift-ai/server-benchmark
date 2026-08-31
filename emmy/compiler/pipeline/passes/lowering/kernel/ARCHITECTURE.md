# Kernel lowering — `TileOp` → `KernelOp`

This stage turns a scheduled `TileOp` into a `KernelOp` (a thread-bound CUDA-IR body), then runs a short chain of
Kernel-IR peepholes over it. The CUDA lowering (`lowering/cuda`) renders the `KernelOp` to a `__global__` source string
afterwards.

## `010_materialize` — bind the schedule to threads (and expand the contraction)

`010_materialize` is a thin wrapper: after the split-survivor assert it makes **one** call to
`_factor.factorize(tile, root)`, the entry to the recursive emitter, then passes the finished body through
`_drop_repeated_declarations` — the emitted body's one legality guard. Operand cones splice INDEPENDENTLY
(`Fold.spliced_step`), so two sibling cones reading one shared broadcast constant each carry their own copy of its
`buf[0]` `Load`, under the same SSA name and at the same address; flattened into one loop body that is two C
declarations of one name, which nvcc rejects (*already declared in the current scope*). The repeat binds nothing new,
so it is dropped with NO rewrite — which is what keeps the guard narrow enough to run over a WHOLE kernel body. The
renaming forms (`_atom._dedup_loads`, `stmt.dedup_loads`) collapse two DIFFERENT names at one address, which needs a
memory-effect reading neither has: a `Write` or an async fill between two identical loads of a staged buffer makes the
second a different value. A same-name repeat cannot hide such a reload, since a rebind in one C scope is already
illegal. A name re-bound to a DIFFERENT address is left alone: that is an SSA fault and must surface as one.

`factorize` builds the ambient `Ctx` and dispatches `tile.op` through `_factorize`, which peels projecting zero-axis
`Fold`s and binds each leaf via the ONE root-binding pipeline (`_factor._bind`) — its form is read off the node's
SCHEDULE (which axes are tiled), never a kernel kind, and
it seals through the one `grid_tile` finalizer (the article's "schedule separate from combine" thesis — the op tree +
`ir/tile` `Fold.lower` are shared across kinds; only the partition changes). Its arms are points of one
`(output-tiling) × (reduce-folding)` space:

- **OUTPUT-tiled** (a contraction — warp / register tile) — a `Fold` whose canonical algebra has a bilinear reading.
  Tile canonicalization establishes this shape before scheduling. When it is present,
  `_bind` only **synthesizes its bare grid-`Write`** (needs `root.output`, so it can't ride the node) and
  **expands** it through the shared tiling layer (below); the leaf type selects the codegen
  (mma / scalar). A contraction without an applicable output tile takes the ordinary Fold path. The schedule only
  places the algebra and declines with `LoweringError` when there is no `(m, n)` grid pair to place onto.
- **REDUCE-tiled** (`_tile_reduce_axis`, a `PLANAR` / `TWISTED` reduce — or a non-output-tiled `CONTRACTION` — whose
  `Reduce` cooperates / register-folds) — the reduce axis is tiled instead: `coop` lanes across the CTA's threads
  (its unit level) and `reg` ILP chains across per-thread accumulators (its register level), then a REG-tree fold, the
  cross-thread combine (`emit_combine`), and the projection. It reads the reduce straight off the `Fold` node (no
  `lower`-then-refind) and builds its per-cell body via the recursion (`_emit`, below); the output stays one cell per
  thread (the 1×1 `atomize`, the grid riding `lead_axes` untiled).
- **Degenerate** — nothing tiled: one thread per output cell (`_emit(op)` + an output-store glue).

### The recursive node walk (`_emit`) — one hierarchical emitter

Two recursions cooperate. The **root** recursion `_factorize(op, ctx, tail, out_val)` binds a node to the grid: a
zero-axis `Fold` recurses through its operand roots (projection → `tail`), and each leaf binds via the one `_bind`
pipeline. The
**body** recursion `_emit(op, ctx) -> Frag` builds the per-cell Loop IR over the Fold tree, threading a `Ctx` **down**
(the ambient cell environment: the grid axes, operand
`inputs`, `stage`, output buffer) and returning a `Frag` **up** (the per-cell `body` this node contributes, the produced
`Handle` wire). The reduce binder drives `_emit` off the `Fold` node to
build its per-cell reduce loop, so a **nested** contraction (a composed fold's inner contraction) is reached AS A
NODE. This is the
tile-IR-rebuild mandate's *one hierarchical emitter, no divergent codegen path*: `_emit(node).body` is byte-identical to
`node.lower()` for a scalar-nested (block=1) node today. `Handle` carries `name` + `residence` (a scalar
register value); the **tensor-core seam** is the view arm in `_bind` — an output-warp-tiled contraction (an mma
`Tile`) emits through the register-tile pipeline + the accumulator→operand fragment recast there, where the rebuild
extends `Handle` with the mma fragment descriptor `(mma_role, shape, dtype)` and `_emit`'s `Ctx` grows the warp binding +
the inbound `wires`.

A materialized Fold-edge cut reaches this walk as an ordinary `Load` operand. `_emit` preserves that load and returns
its bound value as the wire, so cut and fused trees use the same parent emission path.

The output-tiled arm travels as **`(node, tile)`** — the stored `Fold` (bilinear reading) and its PLACED `Tile`
slice. There is no fused view object in `_bind` / `_atom`: `_factor._bind` dispatches on "`is_contraction(op)` with a
TILE slice over a grid with an `(m, n)` pair" and threads the two on; the slice arrives ALREADY PLACED from
`Sched.tile_of`, which binds the caller's `(m, n)` through `Tile.at`. It is
binding-driven for both atoms, with **no per-atom subclass**, and cleanly
splits the **placement/schedule the slice owns** (its `axes` and the `Side`
geometry derived from them — the tiled CELL and nothing outside it, so the kernel's leading batch axes stay the
grid's fact and reach the per-cell rename from `_factor` as its own `lead`) from the **algebra the node owns** (what to
contract: the reduce `axis`, the shared `a` operand edge plus the product `channels` `(b_i, acc_i)` — every edge a gmem
`Load` (materialized) or the computed node itself, stored inline (the fused cone); a projection is NEVER a contraction
field, its one home is the wrapping zero-axis `Fold.lift`. The edges share ONE type. Canonicalization places the
argument shared across channels in `a`; `Sched.tile_of` then orients algebraic M toward the physical output axis that
edge references and N toward the other axis. Either side may be computed and use the synchronous compute fill)
from the **schedule** (the `Tile` slice carrying the leaf `atom` — a tensor-core `AtomKind` / the scalar
`ScalarAtom`, `ir/atom.py` — plus the unit/register widths + K-chunk). The per-CTA geometry (the `(m, n)` `Side` pair —
tile width / mask / block+unit var names — plus `launch_threads`) is **derived on the slice**, from its widths × its
own `axes` (`@property`). Keeping the schedule a single swappable
slice is what lets the same operand/`acc` params be tiled by a *different* `Tile`.

A symbolic / non-divisible tail is **clamp-to-identity** (the masked overhang folds a no-op or guards its store); the
dynamic-grid tier ceil-divides the launch and threads the runtime extent as an `int seq_len` arg.

### The one factorizer

`_factor.factorize(tile, root)` is the **entry** every `TileOp` root lowers through: it builds the ambient `Ctx` and
dispatches `tile.op` into the recursion `_factorize(op, ctx, tail, out_val)`. `_factorize` walks the node tree — a
zero-axis `Fold` with an operand recurses (its projection body is walked via `_emit_body` into the `tail`), and each
leaf binds to the grid via the **ONE** root-binding pipeline, `_bind` — a single pipeline that reads WHICH AXES the
schedule tiles off the node
and seals through the one `grid_tile` finalizer. A tiled contraction tiles its OUTPUT `(m, n)` axes (register / warp
cells; the reduce K serial per cell); a cooperating `Fold` tiles its REDUCE axis instead (`_tile_reduce_axis` —
BLOCK `coop` lanes at the unit level, REG `reg` ILP chains at the register level, the algebra merge — read off the
fold node's `Reduction` view — closing the fold),
its per-cell reduce loop built via `_emit` off the node; each ILP copy suffixes only its per-copy SSA temps (`__r{r}`)
— the shared iteration coordinates, **including any nested contraction's own reduce-axis var** (whose `for`
declaration `copy_cell` does not rename), stay shared, so each copy re-declares its own nested
loop under the one name; anything else tiles nothing and folds serially one thread per
output cell (the degenerate `_emit(op)` + `with_store`) — there is **no** separate "scalar tier" branch, and no
per-kind emitter: which axis is tiled is schedule data, not a kernel identity. The projection sink and the store value
(`out_val`, the root node's produced `Handle`) are threaded down the recursion, so `with_store` is node-agnostic. The
kernel-boundary `TileOp.output_specs` are reconstituted at their owning projection region or the zero-axis Fold peel
(a STREAMED store — one whose values are an observed fold's observer results — rides the recursion down to the leaf
instead and splices into the reduce loop after the observer stmts), so everything below the peel — the sinks,
cooperative loop distribution, and split realizers — consumes the identical statement stream that entered total
lift. An output sweep whose axis the peeled root's cone reads cannot wrap at the peel (the root binds outside the
projection, so no wrap position encloses it): the serial fold binds the projection UNPEELED so the sweep loop wraps
operand and projection together, and a cooperative / ILP row — whose lanes the sweep would be distributed across —
declines via `UnbindableProjection` (`RuleSkipped(reject=True)` at the pass boundary; the greedy retries the next
row). The
recursion, the binder, the reduce-axis tiling, and the shared-row staging apply live in `_factor.py`; the four tiling
levels every tier seals through are `_tiling.py`, which knows a `Side` pair, integer counts and three callables — no
node kinds, no algebra, no `Ctx`. That is the decide/realize seam: the tile schedule picks the plan, `_tiling` is
where a plan becomes bound `Axis` objects. **There is no
kind-specific path — no attention special case.** SDPA lowers as ordinary contraction-shaped `Fold`s plus the
online-softmax `TWISTED` reduce, each factorizing through this one recursion like any other contraction or monoid
fold — **never** a bespoke emitter, which would be a divergent codegen path the mandate forbids. A zero-axis projection
may contain several independent tiled roots. Their projection cones are disjoint, so each root uses `_bind` unchanged;
the resulting regions merge only when their physical grid axes and worker inventories agree. Fragment names carry a
root-local prefix, while identical shared-memory declarations are reused between the sequential regions.

When the scheduler tiles contractions directly inside an exp-family Fold, `_bind` applies only the distributive
codegen reading needed by the existing contraction factorization. With two children, the first supplies score
fragments and the normalized weight becomes the computed A edge of the second. With only the derived expectation
child, the enclosing Fold supplies the materialized score and its sweep axis becomes the contraction K. The stored
Fold tree is unchanged, and `reduce_codegen` plus the generic contraction or Fold sink remain the one MMA realization
path for ordinary contractions, SDPA, and softmax followed by a value contraction.

**The contraction factorization — two atoms.** `_bind`'s output-tiled arm is atom-generic — there is no per-atom
variant, and **no per-atom geometry object**. It expands any contraction-shaped `Fold` by tiling a **leaf atom**
four ways through
the tiling layer (**`_tiling.py`**):
`grid_tile(unit_tile(register_tile(atomize(...))))` — **GRID** block / **UNIT** / **REGISTER** / **ATOM**. The tiling
geometry (the `(m, n)` `Side` pair — `tile` / `mask` / `block` / `unit` per axis — plus `block_threads` / `lanes`) is
**derived on the contraction reading** (`@property`, from the `tile` schedule × the output axes); the two sides
thread through the tiling levels + the codegen callables as one `(m, n)` pair. `factorize` reads it straight off `c`
and hands
`grid_tile` the codegen in two halves: `_atom.reduce_codegen` — the reusable, **sink-agnostic** `(state_decls,
reduce_region)` (operand fragments + the K-loop) — and a per-cell **sink** `store`. The
default is the matmul `_atom.store_sink`; `factorize(c, store=…)` swaps it. The K-loop itself is **one driver** on the
strategy base (`_AtomOps.reduce`), deciding nothing: the **scheduler-resolved** `Stage` picks its form — gmem-direct
(`None`) through the shared `_contract_kloop` `read → ⊗ → fold` spine, or staged through the shared `_staged`
fill→drain skeleton — and the atom contributes only leaves, never a loop. Per-atom diff:

- **mma** (`_MmaOps`) — atom `(16, 8, 16)`, `lanes == 32`. The UNIT is a **warp**; its leaves emit `RegFragment` /
  `LdmatrixLoad` / `MmaSyncPtx` / `RegStore` and decode the atom-lane offset at render. `RegStore` derives both
  algebraic M/N strides from the output index: contiguous N keeps packed stores, while a reversed physical orientation
  uses scalar strided stores. A multi-channel root partitions its projection by output dependence and emits one store
  sink per output.
- **scalar** (`_ScalarOps`) — atom `(1, 1, 1)`, `lanes == 1`. The UNIT is a **single thread** (so there is no `_lane`
  axis); its leaves are plain `Load`s + an fma cell, the projection `tail` replicated per register cell with its
  operand loads deduped (the arithmetic-intensity reuse). One A read per register ROW and one B read per COLUMN is a
  property of the OPERAND, not of the tier: a computed edge whose cone reads the OTHER output axis (an A broadcast
  over n — the o_proj shape `out[m, n] = Σ_k B[n, k] · A[m, k, n]`) holds a different value in every cell of its row,
  so it is read once per register CELL instead, σ-bound to both coordinates. Sharing it would both fold the wrong
  value into every column past the first and leave the sibling coordinate free — the kernel binds only the split
  `_b` / `_u` vars, so the per-copy rename would emit an identifier nothing defines.

Each atom is a strategy class in **`_atom.py`** supplying `state` / `store` plus the descriptor reads the shared
`reduce` consumes — `gmem_leaves` (the four gmem-direct leaf constructors), `staged_drain` (the slab-reading leaf),
`slab_elem` (the slab element dtype) — with `_atom_ops` the dispatch + `reduce_codegen` / `store_sink` the seam
`_factor` calls: the new-atom seam. Staging eligibility + sizing are **not** an atom method: they resolved
scheduler-side into the stamped `Stage` (see Operand staging below).

The **unit** is the atom's parallel thread footprint (`atom.lanes`) — so the tensor-core warp tile and the scalar
parallel thread-tile are the *same* level, differing only in `lanes`; `block_threads = units · lanes`. `grid_tile` also
carries any leading (batch) grid axes and supports a 1-D (m-absent) output. (The store-glue helpers `with_store` /
`has_write`, shared by the constructor and the thread-binding tiers, live in `_factor.py`.)

## Operand staging — the warp-tier smem pipeline (`STAGE` codec → `Stage`)

The warp (mma) tier stages its reused gmem operands through an smem slab, driven off the node's `STAGE` codec →
`schedule.Stage`. Every staged path runs **one** liveness-scheduled K-loop skeleton, `pipelined_kloop` in
**`_stage.py`**: the loop body arrives as ordered segments tagged with the slab names each READS, every staged
operand-group is a `(transport, depth)` pair, and the fill / wait / barrier placement is DERIVED from each group's
live range (`[first reader, last reader]` over the segments) — wait before the first reader, a CTA barrier past the
last, `depth >= 2` prefetching chunk `i+ring-1` at the top of the body, whole-body `depth == 1` filling the current
chunk (the single-buffer degenerate), and a `depth == 1` group live in a PROPER sub-interval refilling chunk `i+1` at
its kill point so the copy overlaps every segment outside the live range. cp.async `wait_group(N)` counts are a
static pass over the placed schedule (the commits younger than a group's fill at its wait point); the prologue primes
exactly the fills the pre-loop iterations would have issued. `staged_kloop` is the whole-body single-group entry
(the matmul tier's classic `fill → commit → wait → drain → Sync` phases fall out of the derivation). Behind it, a
`Transport` strategy: `SyncTransport` (the `smem` fill — a producer cone evaluated per thread into its slab, its
materialized peers copied underneath it, closed by one CTA barrier; `copy_sync` swaps those peer copies from
`cp.async` to the blocking vector load/store on a target without it, which is also how a fully materialized term
stages on sm_70), `CpAsyncTransport`
(fill → commit → wait-group), and `TmaTransport` (an `arrive.expect_tx` + box copy gated by a **per-slot mbarrier
array**, so `depth` is a free knob for TMA too). The three producers —
structurally different primitives — sit behind one `fill`/`commit`/`wait` seam, and
**one atom-agnostic driver** (`_atom._staged`) builds the operand pair + the transport for either atom; the atom
supplies only the slab drain leaf via `_AtomOps.staged_drain` (the shared inner fragment drain
`_staged_inner_atom_loop` — `ldmatrix` on modern atoms, a cooperative shared gather on Volta — or the scalar
`_scalar_drain`). A fill's gmem-address σ binds **every** tiled output axis, not just the operand's own: the tile
axis at `tile_base + cell` (masked axes clamp in-bounds) and the SIBLING axis at its block base — a slab is
CTA-shared across the sibling, so a sibling var can only survive as a value-dead flat-index reshape residue (a
merged / reshaped weight row), and left unbound it would emit the unsplit axis name the kernel no longer defines. The staging **decision** does not live here at all: the
`ResolvedStage` in `ClassicMaterialization` arrives **already resolved** by the scheduler (transport eligibility, the
slab names, K-chunk `bk_elems`, and depth clamps). A direct edge has an explicit direct `Stage` choice and no resolved
materialization. The `state` builder (which slots the operand fragments) and shared `reduce` (which emits the loop)
apply the resolved facts verbatim. The `Stage` choice names the intermediate storage and its fill mechanism — `smem`
(the synchronous thread fill), `smem-async` (cp.async), `smem-tma` (TMA); an EMPTY `STAGE` is no intermediate at all
(gmem→register on a materialized operand, register-to-register on a computed one) — and spells two buffering levels:
`d<depth>` is the gmem→smem ring (blocking synchronous slot fill / cp.async commit group / TMA mbarrier-phased
prefetch over the K-slab loop),
`p<reg_depth>` is the smem→register double-buffer (the fragment-load ping-pong over the inner atom-K steps). Staging is a
**pure perf transform** — an ineligible kernel (masked N, or a symbolic / non-divisible K on a BYTE-COPIED
operand, whose chunk runs along K; a transposed B stages N-major on every transport since the serving-layout work)
silently falls back to gmem-direct, and a staged kernel is
**bit-identical** to its gmem-direct baseline. A synchronous `smem` ring uses the same slot rotation and barriers, but the
fill runs on the consumer threads and therefore cannot overlap the current drain; `/p<n>` remains the independent
smem→register fragment pipeline. The Volta m8n8k4 atom enables only the synchronous byte-copy fill for materialized
f16 A/B edges and keeps computed edges and newer instruction families disabled. The **`smem-tma`** transport
additionally requires **sm_90+**
(Hopper/Blackwell): below it (the schedule's TMA gate, mirroring the frontend TMA-fold gate) the `d*/smem-tma*` moves
are never offered and a `smem-tma` pin refuses — Ada/Ampere have no
`cp.async.bulk.tensor` and nvcc has no `sm_89a` target, so a TMA kernel there would fail to compile. Unpinned, the
schedule fork enumerates the resolver-gated stage grid (`search/space.stage_moves`) alongside the tile / reduce moves;
an `EMMY_STAGE` pin stays authoritative.

**Computed operands and nested Folds.** Every computed edge remains a schedule site. Scalar rows evaluate a pure
producer in registers. Warp rows place a producer either in a synchronous shared-memory slab or, when the child is a
scheduled contraction, directly in fragments before storing the slab consumed by `ldmatrix`. Materialized peers keep
using the ordinary vectorized copy transports. These are residence choices over the same Fold tree; the scheduler and
materializer do not recognize operation families.

The fragment Fold evaluator assigns each live value one of three residences: CTA-cell uniform, one scalar per fragment
row, or one C fragment. It interprets the stored `Lambda` directly. `Assign` broadcasts to the highest input residence,
`Select` substitutes the fragment layout's absolute coordinates, and coordinate-dependent `Load` becomes
`FragmentLoad`. Every runtime-bounded coordinate clamp-reads in-bounds — a masked M row exactly like the reduce
axis — and the reduce boundary additionally adds `FragmentMask` with the Fold identity (the overhanging M row is a
discarded duplicate instead, the copy-transport contract). A Lambda-evaluated producer's fragment column cells span
the whole `bk`-wide slab chunk the drain reads, independent of the output tile's register tiling. The same evaluator
applies the stored carrier `combine` Lambda to the running state, using in-place targets for carried values.

A scheduled child contraction is supplied through the evaluator's structural callback. The ordinary atom strategy
declares and drains its fragments; `FragmentRowReduce` derives row-resident partials; the parent Fold's `combine`
Lambda merges those partials and fragment-resident channels. `SyncOperand.producer` and `_stage.LeadSegment` merely
describe the resulting whole-slab producer and its live range to the common staged-loop scheduler. A materialized
source follows the same path through `FragmentLoad`. No separate blocked, chain, or attention emitter remains.

Fragment stores preserve semantic coordinates independently of their destination address, and retain every
`Assign.dtype`; rebasing a producer into a tile-local slab therefore cannot rebase a causal predicate or erase a
store/load conversion. The producer's `RegStore` carries the same slab swizzle the consumer's `ldmatrix` undoes.
Loop-invariant and chunk-varying child operands use the same ordinary staging transports and liveness rules as every
other contraction.

Where the edge is not a bindable contraction (or the atom has no modeled C layout) it stays per-cell: spliced into the
fill's cell and evaluated inline from lowered loop IR, a scalar dot per slab cell. Geometry: exact cover on N only. A
masked / symbolic **M** clamp-reads (the A / stat-prologue σ ride `_clamp_last`; the overhang store is discarded by
the `RegStore` guard). A symbolic **K** rides the fill's own **K MASK** — the same clamp-to-identity discipline on the
contraction axis: the cone's reads clamp in-bounds and every slab lane whose k index reaches past the runtime extent
stores the additive fold identity 0 (`_atom._k_masked`; the bilinear reading pins ⊕ = add, so a zero operand folds to
nothing and the drain still reads whole chunks), while a canonical materialized peer clamps its overhanging slab ROW
so its `cp.async` chunk stays contiguous. The K-MAJOR orientations keep the refusal, and the reason names why: a
materialized A and a transposed B both stage K as the slab's contiguous inner dim, so their copy chunk runs ALONG K
and clamping only its start still copies past the extent. A **multi-channel product node** (the gate/up MLP edge — N
`(b, acc)` channels over one shared A edge, either a computed cone or a materialized load; `_AtomOps.channels` reads
them off the node) fills one B slab per channel, drains N mma chains off the ONE ldmatrix'd A fragment into
per-channel C fragments (`_fold_frag`), and the projection (SwiGLU) combines the channels per element in the store's
`RegEpilogue` (`extra_accs`). Materialized A copies into the same single A slab; computed A evaluates into it. Both
forms use the synchronous compute fill because the gmem-direct and byte-copy MMA paths remain single-channel.

**Staged fp8 (1-byte) operand slabs.** A storage-dtype (fp8) operand stages as a RAW BYTE slab — each `Operand`
sized at its OWN element width (the mixed-dtype seam the scalar tier already had), the cp.async fill running 16 B
16-element chunks. ldmatrix is b16-only below sm_100a, so the drain is a **cooperative byte gather** instead
(`LdmatrixLoad(byte_slab=True)`): the gmem fragment loaders' lane→element map pointed at the slab — under a 16-bit
atom (W8A16, the fp8-B mul-hoist form) converting per element, with the transposed-B slab's contiguous (k, k+1)
pair loading as ONE fp8x2 and converting with one hardware `cvt.rn.f16x2.e4m3x2` (`emmy_mma_load_b_smem_trans_f8_f16`);
under the fp8 k32 atoms (W8A8) repacking raw bytes, contiguous-K lanes as single u32 loads (`_smem_b8v`). Staged is
BIT-identical to gmem-direct (same converts, same K order). Byte slabs are NONE-swizzle by construction (the
ldmatrix XOR is b16-indexed); the cp.async byte slab instead pads each row by `_stage.BYTE_SLAB_PAD` (16 B — keeps
every chunk 16 B-aligned and takes the drain from 4-way bank conflicts to ≤ 2-way per the lane→bank oracle), and a
TMA byte slab (the U8 `CUtensorMap`) deposits dense and eats the measured conflicts. Legality
(`resolve_warp_stage`'s byte arm): 16-divisible inner spans (and, canonical-B, a 16-divisible gmem row stride N);
the multi-channel sync compute-fill and the scalar resolver still decline 1-byte elements.

**Warp specialization (the producer band → `TileOp.workers`; rows spell it as `WORK`'s `+p<n>` suffix, which is also
how it is pinned — the `WSPEC` key is retired).** A resolved `WarpSpec` splits the SAME staged phases across two
warp bands instead of software-pipelining them in-warp (`_stage._wspec_kloop` — the workers arm of `staged_kloop`,
TMA transport only per the scheduler's legality): the **producer** band rides at the TAIL of the thread block
(`blockDim = block_threads + 32·aux_warps`; the `Tile` decode wraps `threadIdx.x % block_threads`, so the compute
warps' cell decode is untouched, an aux thread gets correct BLOCK coords for the tile origin, and the transport's
`linear_tid == 0` election lands on exactly the band's first thread — the TMA fill is reused verbatim). Its elected
thread primes the ring, then per chunk parity-waits the consumers' slot release (`_mbar_empty`, one u64 per slot,
count 1) and arms + box-copies the prefetch chunk. The **compute** band parity-waits the data mbarrier, drains
(ldmatrix + mma), closes on a named `bar.sync` (a CTA-wide `__syncthreads()` is UB on the divergent role branch) and
ONE elected thread releases the slot — `mbarrier_arrive`'s `fence.proxy.async` orders the band's GENERIC slab reads
before the producer's next ASYNC box copy into the slot (without it the refill overtakes in-flight reads; silent
corruption under scheduling pressure). `SetMaxNReg` redistributes registers between the bands when the raised total
fits the 64K regfile. Stores are guarded to the compute band (`grid_tile`), and the launch/`__launch_bounds__`
account for the aux band (`Tile.aux_threads`). Accuracy-gated, not bit-identical — the split changes scheduling.

The **scalar** contraction tier stages too, under the same `STAGE` codec, through the **same** `_staged` driver — the
scheduler's scalar stage resolver sizes the slab (the depth-aware fit-to-smem K-chunk `bk_elems`, not a codec field;
the depth steps down when no chunk fits) and its `staged_drain` is the plain-`Load` inner loop (`_scalar_drain`,
reading the ring slot via the same slot-row seam as the mma drain). `depth >= 2` is the scalar gmem→smem prefetch
ring — the identical `staged_kloop` cp.async / TMA-mbarrier phases the warp tier runs; only `p<n>` (the
smem→register double-buffer) stays warp-only (an `ldmatrix` transform). The nested outer-slab / inner-drain
accumulator lifetime is handled by seeding the per-cell accumulators once in `_ScalarOps.state` (outside the outer
loop) and marking the inner drain `Loop(seed=False)` so it folds without re-declaring. A masked **M** is supported
(the drain indexes the slab by LOCAL tile coords, so an overhanging row reads in-slab and its store is guarded); a
masked **N** or a transposed **B** declines staging (gmem-direct) — the B-slab fill would fault a row-crossing copy.
Unstaged is byte-identical gmem-direct.

**Split-K composes with staging.** A split partial is a fresh kernel whose own schedule fork resolves a `STAGE`
spec against the SLICED view (the `kslice` extent + the `ksplit`-offset operand indices), so the partial kernel's
K-loop stages its slice through the same pipeline (the TMA box origin is the operand's own index evaluated at the
tile base — an offset operand lands the box at absolute coordinates).

## Fold carriers lower at their scheduled residence

Scalar Fold carriers use the common reduce-axis tiling (`_tile_reduce_axis` — cooperative lanes, register ILP, or
serial). A Fold whose value-producing child is a scheduled contraction uses the residence evaluator above: child
fragments become row partials through `FragmentRowReduce`, and the Fold's stored `combine` Lambda merges them into
row- and fragment-resident state. This applies equally to planar and twisted monoids; the materializer reads only the
Fold algebra and schedule.

Boundary writes consume those residences directly. Fragment state stores in place; row and uniform state broadcasts
through the fragment layout before storing, so split partial kernels preserve every carrier component without
assuming that all components share the contraction accumulator's residence.

The Fold move is never re-decided during materialization. `ReduceStage.combine` is the placement-keyed selector:
within-warp uses `SHFL`, within-block uses a `SHFL` plus shared-memory tree, and cross-CTA uses `ATOMIC` or `KERNEL`
(a multi-component carrier is kernel-finalize only). Scalar materialization consumes it through `emit_combine`, while
the structural `tile/035_split_reduce` fork realizes the graph-level partition.

**Shared-row staging (`_tile_reduce_axis`) — the reduce tier's `sync` transport.** The fused norm→linear prologue is a
cooperative reduce: an input row folded by the cooperative reduce AND re-read per output column of a contraction tail (a
free-axis `Loop` over an inner reduce). Like the contraction tiers, it is **`Stage`-driven**: the scheduler
(the shared-row stage detection, narrow so a plain softmax sum or a bare reduction is untouched) detects that one row
and stamps a depth-1 `sync` `Stage` whose `smem` names it — a derived schedule field, never a knob.
`_tile_reduce_axis` only *applies* it: the row is filled cooperatively via `_stage.sync_row_fill` — the **same
`_stage.py` fill module** the warp tier's
cp.async / TMA fills live in, indexed off the same linear-tid / thread-count seam — and both readers are rewritten to
the slab (`_restage_loads`). So every staging decision rides a `Stage` on the schedule and every transport (`sync` 1-D
row · cp.async / TMA 2-D slab) lowers through one module. A contraction operand `Stage` never sets `smem`, which is how
the two apply paths stay distinct on a coop-K contraction.

## Kernel-IR peepholes

`030_stamp_types` resolves element dtypes. Integer algebra is always restamped from its typed operands, repairing a
stale float stamp that a structurally cloned, previously untyped body can carry. `050_vectorize_loads` /
`080_vectorize_stores` /
`095_interleave_loads` pack/reorder memory ops; `096_pair_ldmatrix_loads` fuses slab-adjacent staged `x2` B-fragment
`LdmatrixLoad`s into one `x4` (`pair_frag` — plain `x4` for an N-adjacent transposed-B pair, `x4.trans` for a
col-adjacent canonical pair; equal swizzle modes pair too — the per-lane address XOR commutes with the paired lane
map; halves the staged drains' LSU count, bit-identical; fires on the matmul tier's staged drains);
`110_drop_redundant_syncs` collapses the defensive `Sync`s the
cooperative / shared-row templates emit (body-level only — a slab `Smem` decl flags `smem_seen`, so a load-bearing
prologue `Sync` is correctly retained; `with_bodies` preserves the cooperative tile's `block_threads`).

Every codegen-policy peephole records its decision as an on-by-default BOOL policy knob on the `KernelOp`
(`VECTORIZE_LOADS` / `VECTORIZE_STORES` / `INTERLEAVE_LOADS` / `PAIR_LDMATRIX` — the `050` pattern: idempotence via
the recorded knob, `EMMY_<NAME>=0` pins it off, never a search dimension), so no rewrite that touches emitted code
is unconditional-and-unrecorded.

Two of these peepholes are **pin-only policy stamps** — off by default, byte-identical, decoupled from production
codegen (each records its knob on the `KernelOp` for idempotence, like `095`, and returns the body unchanged when off,
so the whole default pipeline is unaffected and there is no golden / snapshot churn): `085_fast_exp` (`EMMY_FAST_EXP=1`
lowers f32 `exp` through the SFU `__expf`, the one non-bit-exact policy) and `100_loopify` (`EMMY_LOOPIFY=N`, a generic
**loop re-roller** iterated to a fixpoint). Loopify folds a maximal run of ≥ `N` congruent per-fragment statements —
an mma body's per-fragment epilogue (`FragmentApply`), its load+mma pairs, the fragment `RegStore`s, the A-fragment
loads, a nested contraction's K-chunks × N-atoms — into
`#pragma unroll` `StridedLoop`s over `_r{depth}`. The matcher (`_reroll`) is node-type-agnostic: a recursive structural
walk over each candidate window's Stmt/Expr trees returns a template ONLY when every per-iteration difference is a
**contiguous fragment family** index (`O_i_f0 … O_i_f{N-1}` or an already-arrayed `fam[0] …`, arrayed into one
`count`-arrayed `RegFragment` decl and indexed `fam[_r]`) or an **affine address offset** — and it peels a trailing
`± Literal` off each index Expr so a codegen-**folded** `kv0` (really `kv0 + 0*8`) reconciles with a sibling's `kv0 + 8`.
A family that is not `RegFragment`-declared (a scalar carrier / an inline-declared fragment) bails the run. Iterating to
a fixpoint (fresh `_r{depth}` per pass; already-arrayed `fam[i]` refs re-parsed) turns nested runs into nested loops:
pass k re-rolls the inner N-atoms, pass k+1 sees the resulting sibling loops as a run and wraps them. Correctness is
structural: an unrolled congruent run executes in the SAME order as the original straight-line statements, so a template
that reproduces every window is byte-identical after nvcc unrolls — identical SASS.
A readability lever for `--ir cuda` inspection, `N=4` the recommended sweet spot (skips the 2-long runs).
Two orthogonal readability transforms ride alongside: `FragmentApply` **always** renders as one element
`#pragma unroll for (_e)` loop (the ROW operand as the row-split ternary `_e < 2 ? row0 : row1`), so a re-rolled family
nests as `for (_r) { for (_e) … }`; and pin-gated **chain fusion** (`_fuse_chains`, before the re-roll) folds a
`FragmentApply` immediately consumed by an in-place unary `FragmentApply` on the same fragment (`p <- s − m` then
`p *= exp(p)`) into one node carrying the tail on its `post` field, so the softmax renders `p[_e] = expf(s[_e] − m)`.
NOTE: the whole-body SSA rename this pass runs is why the shared `_rewrite` `Tile` handler must carry `block_threads` /
`aux_threads` through verbatim — dropping them silently over-launches a cooperative tile.
