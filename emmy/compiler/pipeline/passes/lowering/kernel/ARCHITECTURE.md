# Kernel lowering — `TileOp` → `KernelOp`

This stage turns a scheduled `TileOp` into a `KernelOp` (a thread-bound CUDA-IR body), then runs a short chain of
Kernel-IR peepholes over it. The CUDA lowering (`lowering/cuda`) renders the `KernelOp` to a `__global__` source string
afterwards.

## `010_materialize` — bind the schedule to threads (and expand the contraction)

`010_materialize` is a thin wrapper: after the split-survivor assert it makes **one** call to
`_factor.factorize(tile, root)`, the entry to the recursive emitter. `factorize` builds the ambient `Ctx` and dispatches
`tile.op` through `_factorize`, which peels the projecting `Map`s and binds the leaf via the ONE root binder
(`_factor._bind`) — a single pipeline whose form is read off the node's SCHEDULE (which axes are tiled), never a kernel
kind, sealed through the one `grid_tile` finalizer (the article's "schedule separate from combine" thesis — the op tree
+ `ir/tile` `Fold.lower` are shared across kinds; only the partition changes). Its arms are points of one
`(output-tiling) × (reduce-folding)` space:

- **OUTPUT-tiled** (a contraction — warp / register tile) — a `Fold` that reads as bilinear (`ir/tile/ir.py`; since
  the collapse `Contraction` is that READING, not a stored kind), its operand→role binding resolved recognize-side
  (`010_recognize._nodify_contraction` / `_atomize.bind_prologue_contraction` — the ONLY nodification sites; the
  schedule just places), so
  `_bind` only **synthesizes its bare grid-`Write`** (needs `root.output`, so it can't ride the node) and
  **expands** it through the shared tiling layer (below); the leaf type selects the codegen
  (mma / scalar). An unbindable contraction (a non-`Load` operand) keeps the `Map` form and falls through to the
  degenerate arm here. (This build was a separate `005_contract` pass, then folded into materialize, and now lives
  recognize-side so the node exists before scheduling; the schedule only PLACES it, and declines with
  `LoweringError` when there is no `(m, n)` grid pair to place onto.)
- **REDUCE-tiled** (`_tile_reduce_axis`, a `PLANAR` / `TWISTED` reduce — or a non-output-tiled `CONTRACTION` — whose
  `ReducePlan` cooperates / register-folds) — the reduce axis is tiled instead: `coop` lanes across the CTA's threads
  (its unit level) and `reg` ILP chains across per-thread accumulators (its register level), then a REG-tree fold, the
  cross-thread combine (`emit_combine`), and the projection. It reads the reduce straight off the `Fold` node (no
  `lower`-then-refind) and builds its per-cell body via the recursion (`_emit`, below); the output stays one cell per
  thread (the 1×1 `atomize`, the grid riding `lead_axes` untiled).
- **Degenerate** — nothing tiled: one thread per output cell (`_emit(op)` + an output-store glue).

### The recursive node walk (`_emit`) — one hierarchical emitter

Two recursions cooperate. The **root** recursion `_factorize(op, ctx, tail, out_val)` binds a node to the grid: a `Map`
with a `source` recurses (projection → `tail`), the leaf binds via the one `_bind` pipeline. The **body**
recursion `_emit(op, ctx) -> Frag` builds the per-cell loop-IR — over the `Map` / `Fold` tree,
through **`source` AND `partial`** — threading a `Ctx` **down** (the ambient cell environment: the grid axes, operand
`inputs`, `stage`, output buffer) and returning a `Frag` **up** (the per-cell `body` this node contributes, the produced
`Handle` wire). The reduce binder drives `_emit` off the `Fold` node to
build its per-cell reduce loop, so a **nested** contraction (a composed fold's inner contraction) is reached AS A
NODE. This is the
tile-IR-rebuild mandate's *one hierarchical emitter, no divergent codegen path*: `_emit(node).body` is byte-identical to
`node.lower()` for a scalar-nested (block=1) node today. `Handle` carries `name` + `residence` (a scalar
register value); the **tensor-core seam** is the view arm in `_bind` — an output-warp-tiled contraction (an mma
`TilePlan`) emits through the register-tile pipeline + the accumulator→operand fragment recast there, where the rebuild
extends `Handle` with the mma fragment descriptor `(mma_role, shape, dtype)` and `_emit`'s `Ctx` grows the warp binding +
the inbound `wires`.

The output-tiled arm travels as **`(node, tile)`** — the stored `Fold` (bilinear reading) and its PLACED `TilePlan`
slice. There is no fused view object in `_bind` / `_atom`: `_factor._bind` dispatches on "`is_contraction(op)` with a
TILE slice over a grid with an `(m, n)` pair" and threads the two on; the slice arrives ALREADY PLACED from
`Sched.tile_of`, which binds the caller's `(m, n)` through `TilePlan.at`. It is
binding-driven for both atoms, with **no per-atom subclass**, and cleanly
splits the **placement/schedule the slice owns** (its `axes` and the `Side`
geometry derived from them — the tiled CELL and nothing outside it, so the kernel's leading batch axes stay the
grid's fact and reach the per-cell rename from `_factor` as its own `lead`) from the **algebra the node owns** (what to
contract: the reduce `axis`, the shared `a` operand edge plus the product `channels` `(b_i, acc_i)` — every edge a gmem `Load` (materialized) or the
computed node itself, stored inline (the fused cone); a projection
is NEVER a node field, its one home is the wrapping `Map.body`. The edges share ONE type: the A/B asymmetry that is real
— A is M-resident and compute-fillable, B is the K×N operand the loop streams — is a SCHEDULE fact, so each staged /
mma tier states `isinstance(c.b, Load)` as an eligibility precondition and declines a computed B to gmem-direct)
from the **schedule** (the `TilePlan` slice carrying the leaf `atom` — a tensor-core `AtomKind` / the scalar
`ScalarAtom`, `ir/atom.py` — plus the unit/register widths + K-chunk). The per-CTA geometry (the `(m, n)` `Side` pair —
tile width / mask / block+unit var names — plus `launch_threads`) is **derived on the slice**, from its widths × its
own `axes` (`@property`). Keeping the schedule a single swappable
slice is what lets the same operand/`acc` params be tiled by a *different* `TilePlan`.

A symbolic / non-divisible tail is **clamp-to-identity** (the masked overhang folds a no-op or guards its store); the
dynamic-grid tier ceil-divides the launch and threads the runtime extent as an `int seq_len` arg.

### The one factorizer — the single binder + reduce-axis tiling (`_factor.py`), atom strategies (`_atom.py`), axis realization (`_tiling.py`)

`_factor.factorize(tile, root)` is the **entry** every `TileOp` root lowers through: it builds the ambient `Ctx` and
dispatches `tile.op` into the recursion `_factorize(op, ctx, tail, out_val)`. `_factorize` walks the node tree — a `Map`
with a `source` **recurses** (its projection `body` walked, via `_emit_body`, into the `tail`), and the leaf binds to
the grid via the **ONE** root binder, `_bind` — a single pipeline that reads WHICH AXES the schedule tiles off the node
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
kernel-boundary `TileOp.stores` (1q — the root `Write`s / output sweep that left the term) are reconstituted into the
projection `tail` at the `Map` peel (`effect_tail`; plain stores append at a flat/bare root), so everything below the
peel — the sinks, the sweep's coop `StridedLoop` distribution, the split realizers — consumes the identical stmt
stream the stored-`Write` era carried. The
recursion, the binder, the reduce-axis tiling, and the shared-row staging apply live in `_factor.py`; the four tiling
levels every tier seals through are `_tiling.py`, which knows a `Side` pair, integer counts and three callables — no
node kinds, no algebra, no `Ctx`. That is the decide/realize seam: the tile schedule picks the plan, `_tiling` is
where a plan becomes bound `Axis` objects. **There is no
kind-specific path — no attention special case.** SDPA lowers as ordinary contraction-shaped `Fold`s plus the
online-softmax `TWISTED` reduce, each factorizing through this one recursion like any other contraction or monoid
fold — **never** a bespoke emitter, which would be a divergent codegen path the mandate forbids.

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
  `LdmatrixLoad` / `MmaSyncPtx` / `RegStore` and decode the atom-lane offset at render.
- **scalar** (`_ScalarOps`) — atom `(1, 1, 1)`, `lanes == 1`. The UNIT is a **single thread** (so there is no `_lane`
  axis); its leaves are plain `Load`s + an fma cell, the projection `tail` replicated per register cell with its
  operand loads deduped (the arithmetic-intensity reuse).

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
`Transport` strategy: `SyncCopyTransport` (blocking vector load/store → CTA barrier), `CpAsyncTransport`
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
`Stage` on the `TileOp` arrives **already resolved** by the scheduler (transport eligibility, the slab K-chunk
`bk_elems`, the depth clamps — or `None`, gmem-direct), and `state` (which slots the operand fragments) and the
shared `reduce` (which emits the loop) apply it verbatim. The `Stage` spells two buffering levels:
`d<depth>` is the gmem→smem ring (blocking synchronous slot fill / cp.async commit group / TMA mbarrier-phased
prefetch over the K-slab loop),
`p<reg_depth>` is the smem→register double-buffer (the fragment-load ping-pong over the inner atom-K steps). Staging is a
**pure perf transform** — an ineligible kernel (masked N, symbolic / non-divisible K; a
transposed B stages N-major on every transport since the serving-layout work) silently falls back
to gmem-direct, and a staged kernel is
**bit-identical** to its gmem-direct baseline. A synchronous ring uses the same slot rotation and barriers, but the
copy runs on the consumer threads and therefore cannot overlap the current drain; `/p<n>` remains the independent
smem→register fragment pipeline. The Volta m8n8k4 atom enables only this copy transport for materialized f16 A/B
edges and keeps computed edges and newer instruction families disabled. The **TMA** transport additionally
requires **sm_90+**
(Hopper/Blackwell): below it (the schedule's TMA gate, mirroring the frontend TMA-fold gate) the `d*/tma*` moves are
never offered and a `tma` pin declines to cp.async / gmem-direct — Ada/Ampere have no `cp.async.bulk.tensor` and nvcc
has no `sm_89a` target, so a TMA kernel there would fail to compile. Unpinned, the schedule fork enumerates the
resolver-gated stage grid (`search/space.stage_moves`) alongside the tile / reduce moves; a `EMMY_STAGE` pin stays
authoritative.

**Inline operands — the mma tier's `sync` transport.** A matmul with a pure producer cone on either operand reaches
the warp tier through a COMPUTED edge: recognition stores the producer tree inline on that edge
(`_atomize.make_cone`), and the schedule offers a MANDATORY resolved `sync` `Stage` (there is no gmem-direct sibling —
a copy transport cannot evaluate a cone). `_staged` builds a `SyncTransport` whose computed A or B fill evaluates
ordinary scalar tensor algebra per shared-memory slab cell, feeding the unchanged `ldmatrix` drain. A is stored in
canonical `(tile_m × bk)` geometry and B in canonical `(bk × tile_n)` geometry. Materialized peer operands use the
same vectorized `cp.async` path as ordinary staged matmul, so a generic compact-storage B producer can be evaluated
directly into Tensor Core fragments without first constructing its expanded dense matrix. This facility is defined
entirely in generic tensor/loop IR; checkpoint formats are already dissolved before it is selected.

The compute fill assigns each thread a contiguous run of slab cells (the row/col derivation hoists out of the
per-cell code and the cone replicates with a `__c<j>` SSA suffix). A materialized canonical B uses the K-major
`(bk × tile_n)` slab; a transposed B (the serving `F.linear` layout) uses the N-major `(tile_n × bk)` slab in its own
gmem orientation (`Operand.trans`). When a two-slot ring also fits the smem budget, the stage resolves at `depth=2`
and copied peer chunks can stay in flight across the current chunk's drain. A
**reduce-bearing (MONOID) cone** — the fused norm→linear edge — is the schedule's fused term READING
(`_atomize.bind_prologue_contraction`; real fork rows unioned with the map form's, not a pin rescue): the A cone is an
inline node tree whose
SOURCE is the row-invariant prologue (the per-row statistic) and whose `body` is the per-cell normalize, so the K seam
IS the node boundary — read by `ops.cone_seam` in `_sync_operands` — and the prologue runs ONCE per tile row as the
transport prologue
(`_stage.sync_stat_fill` — one row per WARP: the 32 lanes stride the row's reduce coalesced and close the fold with
the stat fold's shuffle butterfly (`emit_combine` off the threaded `Reduction`), lane 0 writing the bridged stat into its smem row; one barrier);
the per-cell compute-fill reads the bridged values back from the stat rows. Geometry: exact cover on N/K only — a
masked / symbolic **M** clamp-reads (the A / stat-prologue σ ride `_clamp_last`; the overhang store is discarded by
the `RegStore` guard). A **multi-channel product node** (the gate/up MLP edge — N `(b, acc)` channels over the ONE
shared inline cone; `_AtomOps.channels` reads them off the node) fills one B slab per channel, drains N
mma chains off the ONE ldmatrix'd A fragment
into per-channel C fragments (`_fold_frag`), and the projection (SwiGLU) combines the channels per element in the
store's `RegEpilogue` (`extra_accs`).

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

**Split-K composes with staging.** The split-K option resolves a `STAGE` spec against the SLICED inner view
(the `kslice` extent + the `ksplit`-offset operand indices) and `030_split_reduce` threads the resolved `Stage` onto its
partial `TileOp`s, so the partial kernel's K-loop stages its slice through the same pipeline (the TMA box origin is
the operand's own index evaluated at the tile base — an offset operand lands the box at absolute coordinates).

## A TWISTED carrier lowers through the generic reduce tiers

There is no attention emitter and no fragment-residence realizer for a `TWISTED` reduce: the online-softmax fold
lowers at scalar residence through the same reduce-axis tiling every `PLANAR` fold takes (`_tile_reduce_axis` —
coop lanes / ILP chains / serial), its multi-component streaming merge regenerated off the node's `Reduction` view.
The fold MOVE itself is never re-decided per site: `ReduceStage.combine` (`ir/schedule.py`) is the ONE
placement-keyed selector — within-warp → `SHFL`, within-block → `SHFL`+`SMEM` tree, cross-CTA → `ATOMIC`/`KERNEL`
(a multi-component carrier is kernel-finalize only) — and every emitter consumes its output (`emit_combine` at
scalar residence, `030_split_reduce` as the graph rewrite).

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

`030_stamp_types` resolves element dtypes; `050_vectorize_loads` / `080_vectorize_stores` /
`095_interleave_loads` pack/reorder memory ops; `096_pair_ldmatrix_loads` fuses slab-adjacent staged `x2` B-fragment
`LdmatrixLoad`s into one `x4` (`pair_frag` — plain `x4` for an N-adjacent transposed-B pair, `x4.trans` for a
col-adjacent canonical pair; equal swizzle modes pair too — the per-lane address XOR commutes with the paired lane
map; halves the staged drains' LSU count, bit-identical; fires on the matmul tier's staged drains);
`110_drop_redundant_syncs` collapses the defensive `Sync`s the
cooperative / shared-row templates emit (body-level only — a slab `Smem` decl flags `smem_seen`, so a load-bearing
prologue `Sync` is correctly retained; `with_bodies` preserves the cooperative tile's `block_threads`).

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
