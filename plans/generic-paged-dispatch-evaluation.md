# Generic paged CPU/GPU dispatch evaluation

Status: proposed. This plan evaluates one mechanism; it does not propose a production serving path.

## Objective

Determine whether Emmy can manage dispatch-unit inputs and outputs as generic pages and choose CPU or GPU execution
from page residency, without the page manager recognizing attention, transformer blocks, experts, or other model
structure.

Use `allenai/OLMoE-1B-7B-0125-Instruct`. Its real checkpoint and routing provide repeated, non-uniform weight access,
while an artificial GPU page-cache capacity makes misses reproducible on the local 32 GB RTX 5090.

## Claim boundary

The page manager and dispatcher are model-independent. The workload adapter is not: Emmy's existing MoE third seam
turns a selected expert slice into a runtime input binding. A conservative analysis of the original E-stacked weight
alone would mark the entire tensor read and could not discover which expert was selected.

The experiment may consume generic buffer bindings produced by that adapter, including base offset, byte length,
content identity, and read-only status. It must not consume expert numbers, router scores, layer types, provenance
labels, or model module names. This evaluates whether memory management can be generic after sparse access has been
made explicit; it does not claim that Loop or Tile IR discovers sparse routing by itself.

This plan also does not evaluate explicit software address translation inside generated CUDA loads and stores. The
first experiment uses stable virtual addresses backed by CUDA virtual-memory mappings so existing kernels retain
ordinary pointers. Software translation, and end-to-end model integration, require separate plans if this mechanism
passes.

## Questions and success gates

The evaluation answers three questions in order:

1. Can Loop IR logical inputs/outputs plus the lowered execution plan yield a complete dispatch-unit memory contract?
2. Can one ordered runtime execute a unit on CPU or GPU while maintaining correct page versions and in-flight use?
3. Does a generic least-recently-used policy retain repeatedly accessed weight pages on a real routed-expert trace?

The mechanism passes when:

- Static validation accounts for every external input and output of each unit and every lowered internal side effect.
- CPU-only, GPU-only, and deliberately mixed schedules agree with one shared reference within existing tolerances.
- Every reader observes the latest completed write; no mapped page is evicted while a launch can still access it.
- Repeating a stationary routed trace raises dispatch-unit GPU eligibility, while a shifted trace replaces the old
  working set within a measurable number of dispatches.
- Results separate compute, transfer, policy, and dispatch time and report useful bytes versus bytes transferred.

Do not claim a useful serving design merely from a high page-hit rate. One missing page makes an entire dispatch unit
CPU-ineligible, so dispatch-unit eligibility and latency are the primary outcomes.

## Fixed scope

- Use programmatic dispatch only; disable CUDA graph capture in every measured arm.
- Use 2 MiB logical pages for the functional evaluation, subject to the CUDA allocation granularity reported by the
  driver. Page-size comparison is trace-only and happens after correctness.
- Page only a dispatch unit's external buffers. Its registers, shared memory, TMA descriptors, and lowering-created
  scratch buffers remain device-local under the existing CUDA allocator.
- Give every paged external buffer its own page-aligned virtual range. Do not pack small buffers in the first version;
  report internal fragmentation instead of introducing partial-page false sharing.
- Treat read-only status as a generic runtime binding property with a stable content identity and version. Do not
  infer it from `BufferSpec.role`: expert weights are graph inputs, not `ConstantOp` buffers.
- Keep read-only source pages in ordinary pageable or mapped CPU memory and use a bounded pinned staging pool for
  transfers. GPU copies are clean cache entries and need no write-back; do not pin the full float32 expert layer.
- Start with synchronous demand admission and least-recently-used eviction. Do not add decay, asynchronous promotion,
  prefetch prediction, or expert-specific pinning until the minimal mechanism is measured.

## Dispatch-unit construction

Take a copy of the graph after Loop fusion and stamping. For each finalized `LoopOp`:

1. Build a `single_node_graph` slice with its external inputs and outputs.
2. Retain that exact `LoopOp` and slice as the CPU realization.
3. Lower a copy of the isolated slice and retain its complete `ExecutionPlan` as the GPU realization.
4. Join the two realizations by graph SSA buffer names, shapes, and dtypes, not provenance or operation names.

Do not lower the whole graph and try to regroup CUDA launches afterward. Lowering may split work into new kernels and
delegate a zero-init to an earlier launch; those transformations do not preserve enough source identity for reliable
regrouping.

The two representations have different responsibilities:

- Loop IR defines the unit's logical external reads and writes.
- The lowered `ExecutionPlan` defines all CUDA launches, TMA sources, zero-init effects, and group-internal buffers.

Coherence applies only to external buffers. Lowering-created scratch lives for the GPU group and never enters the
cross-device page table. An unknown external access, alias, TMA source, zeroed output, or multi-output write aborts plan
construction.

Instantiate symbolic shapes before converting external buffer byte ranges to page IDs. Begin conservatively: every
page in each external input is read, and every page in each external output is written. Resolve a dynamically rebound
input to its logical content identity and byte range before dispatch without interpreting what selected it.

## Minimal virtual-memory allocator seam

The current `CompiledProgram` allocates standalone CuPy arrays and a liveness-packed scratch slab. The prototype needs
a narrow allocator seam rather than pretending those allocations can be paged in place:

- Reserve a stable, page-aligned GPU virtual range for each paged external buffer.
- Allocate physical GPU pages from one capacity-limited pool and map/unmap them into those ranges.
- Expose CuPy views over the reserved ranges for existing kernel argument binding.
- Let the existing allocator continue to own each isolated GPU group's internal scratch and descriptors.
- Track in-flight pin counts so physical pages cannot be unmapped until their CUDA completion event has fired.
- Reclaim mutable logical pages after their last dispatch-unit use using the static graph liveness information.

The prototype stays in memory and does not change the serialized execution-plan format. A short feasibility spike must
first prove reserve, map, launch through a CuPy view, unmap, and remap on the local driver. Stop if stable virtual
mapping cannot preserve the current kernel and descriptor contracts.

## Runtime page table and admission

Use a host-owned table indexed by logical page ID:

```text
logical page ID
  byte length
  mutable or read-only
  stable content identity + version
  CPU valid + CPU address
  GPU valid + GPU physical slot
  latest completed version
  pending completion event/future
  in-flight pin count
  last-access sequence
```

Admission is transactional for the union of a unit's read and write pages:

1. Wait for pending writers of every read page.
2. Pin all GPU-resident read pages.
3. Reserve and map every output page that the GPU launch can write.
4. Evict only clean, completed, unpinned pages if capacity is needed.
5. Launch on GPU only when all reads are valid on GPU and all writes have mapped output storage.
6. If the read-plus-write working set exceeds the entire pool, mark that unit CPU-only for the configuration.
7. On failure, roll back reservations and pins before choosing CPU.

For CPU dispatch, copy any newer mutable inputs from GPU to CPU and execute `LoopOp.forward`. The current runner
returns fresh NumPy arrays, so the prototype explicitly copies those results into page-owned CPU output storage and
charges that copy to CPU dispatch; a caller-provided output ABI is a later optimization. Mark those pages as the
latest version. The primary coherence arm then synchronously copies live CPU outputs to GPU using the same
transactional reservation and rollback rules as GPU admission, so one miss does not automatically force every
downstream unit onto CPU. An owner-only control omits that copy and measures the resulting activation-cascade length.
Both rules depend only on mutability and liveness.

After either device completes, increment written-page versions, invalidate older mutable replicas, release pins, and
record the completion event. Conservatively mark every possible external output page written.

## Initial GPU page-cache policy

Use synchronous demand admission with least-recently-used eviction:

1. A unit with all reads resident and output space reservable runs on GPU.
2. A unit with any missing read page runs on CPU.
3. After the CPU unit completes, admit its missing read-only pages into remaining GPU capacity.
4. Evict the least-recently-used clean, completed, unpinned read-only pages as needed.
5. Reserve live mutable output pages before admitting read-only pages.

This intentionally gives the policy a simple adaptation rule and makes transfer cost visible. Compare it with a
static first-pages control and, in trace replay only, offline static-frequency and Belady replacement oracles. Add a
more elaborate policy only if least-recently-used replacement leaves a clear, measured opportunity.

## Evaluation stages

### Stage 0: synthetic coherence and VMM proof

Build a small graph with read-only inputs, mutable intermediates, multiple outputs, a partial final page, zero-init,
and at least three dispatch units. Force CPU/GPU/CPU/GPU and GPU/CPU/GPU/CPU schedules.

Verify:

- VMM reserve/map/remap through ordinary CuPy kernel arguments.
- Transactional read and write admission, rollback, and capacity rejection.
- Page versions, event waits, invalidations, in-flight pins, and last-use reclamation.
- Multi-output writes, zeroed outputs, symbolic sizes, and the owner-only activation cascade.
- Failure-closed handling for an unaccounted access or lowered side effect.

Run pure page-table tests in the ordinary CPU suite and one small GPU test for mapping, event, and transfer ordering.

Exit: all-CPU, all-GPU, and every forced mixed schedule produce the same result, and sanitizers find no invalid access.

### Stage 1: one real OLMoE expert layer

Use the existing expert wrapper with real OLMoE checkpoint slices. Convert the wrapper and selected real weights to
float32 before tracing, then produce both CPU and CUDA realizations from that same float32 Loop graph. Converting arrays
after tracing is insufficient because it would not change buffer specs or generated kernel types.

The float32 restriction comes from the current Loop IR CPU runner. This stage evaluates correctness and page behavior,
not bf16 serving performance or the quality of an optimized CPU backend.

Replay three deterministic logical-buffer binding traces through the same expert program:

- Stationary: a skewed set repeated long enough to warm the cache.
- Uniform: every expert slice selected equally.
- Shifted: one skewed set followed by a disjoint hot set.

Use 2 MiB pages and synchronous least-recently-used replacement. Sweep approximately 128, 256, 512, 1024, and
2048 MiB because one layer's float32 expert weights are roughly 1.5 GiB; the full-model 1-8 GiB sweep would make most
one-layer configurations degenerate.

Exit: mixed execution matches the all-GPU result, stationary eligibility rises after warmup, uniform access does not
manufacture a hot set, and shifted access has a measurable adaptation interval.

### Stage 2: real routed-expert trace replay

Run the existing OLMoE eager/serving path on a fixed prompt corpus and record the ordered generic expert-program
bindings it presents: stable content identity, byte range, dtype, mutability, and dispatch-unit external buffer sizes.
The recorder may use the existing model-aware serving adapter; the page manager receives only the resulting bindings.

Replay page decisions without executing model math. This is a full routed-expert workload trace, not an end-to-end
model-memory result: attention, the router, and other non-expert allocations remain outside the measured page pool.

Use fixed seeds and prompt order. Include repeated prompts, unrelated prompts, and a distribution shift. Reset the
cache explicitly for cold runs and define the exact warmup boundary for warm runs.

Sweep:

- GPU page-cache capacity: 1, 2, 4, and 8 GiB.
- Page size: 256 KiB, 1 MiB, 2 MiB, and 4 MiB where CUDA backing permits it.
- Policy: static first-pages, least-recently-used, offline static-frequency, and offline Belady oracle.

This is not a full Cartesian functional matrix. Stage 2 screens capacities and page sizes cheaply; only the best and
worst non-oracle configurations are rerun through Stage 1 execution.

Exit: every binding is covered, the capacity curve shows whether generic reuse exists, and the gap to the offline
oracles quantifies whether policy sophistication is worth a follow-up.

## Measurements

Record for each configuration:

- Dispatch-unit GPU eligibility and CPU/GPU dispatch counts.
- Page hits by bytes and count, plus compulsory, capacity, and policy misses.
- Output-reservation failures and units whose read-plus-write set exceeds total capacity.
- Useful bytes, transferred bytes, page-size overfetch, and standalone-buffer fragmentation.
- H2D/D2H copy count, bytes, and time; evictions, admissions, waits, and in-flight peak.
- Activation-cascade length after each CPU dispatch.
- CPU compute, GPU compute, transfer, page-policy, and total wall time separately.
- Per-unit latency p50, p95, and p99, plus whole-run median and dispersion.
- Peak pageable GPU pool, other VRAM, pinned CPU memory, and page-table memory.
- Correctness error statistics and the first divergent dispatch unit.

Fix and record CPU thread count and affinity, pinned-host allocation, PCIe topology, copy-stream setup, compiler
regime, GPU clocks, checkpoint revision, prompts, seeds, and warmup/reset rules. Preserve the raw ordered event trace
so every aggregate can be reconstructed.

## Baselines

1. Unconstrained all-GPU execution of the same isolated float32 units.
2. All-CPU execution of the same finalized Loop IR units.
3. Capacity-limited GPU with synchronous promotion before every miss instead of CPU fallback. Mark this baseline
   unavailable for a configuration when any unit's read-plus-write working set exceeds the pool; never silently fall
   back to CPU.
4. Hybrid CPU/GPU dispatch with synchronous least-recently-used replacement.
5. Trace-only static-frequency and Belady oracles, clearly labeled as future-aware and computation-free.

The primary comparison is not vLLM. This evaluation isolates page residency, coherence, and dispatch.

## Expected repository seams

- `emmy/compiler/pipeline/search/slice.py`: reuse the existing single-node slicing mechanism.
- `emmy/compiler/backend/plan.py`: reuse buffer specs, launch arguments and writes, TMA sources, zero effects,
  symbolic shapes, and weight bindings without changing the persisted plan schema.
- `emmy/compiler/backend/cuda/program.py`: permit externally supplied arrays for a unit's paged external buffers and
  expose execution of one isolated launch group with a completion event.
- A small experimental CUDA VMM allocator beside `program.py`: own virtual ranges, physical-page slots, CuPy views,
  map/unmap operations, and in-flight pins.
- `emmy/compiler/ir/loop/runner.py`: provide the initial CPU realization; keep its float32 limitation visible.
- `tests/compiler/backend`: hold page-table, VMM, coherence, admission, and failure-closed tests.
- `experiments/olmoe-paged-dispatch`: only after correctness, preserve the routed-trace evaluation and raw results
  through the normal experiment workflow.

## Stop conditions

Stop before a larger integration if any of these holds:

- The local CUDA driver cannot preserve stable virtual addresses across map/unmap for current kernel arguments.
- Isolated Loop slices cannot produce structurally paired CPU and GPU realizations without operation names.
- External-buffer accounting misses a lowered side effect or mixed execution fails synthetic correctness.
- Whole-buffer footprints create enough overfetch that policies cannot be distinguished; add sub-buffer access-range
  analysis before repeating Stage 1.
- Stationary and repeated real traces show no useful dispatch-unit eligibility separation from uniform access.
- Page management and transfer time exceed the CPU work avoided in the isolated expert evaluation.

If these stages pass, write separate plans for end-to-end OLMoE integration and explicit software page-table
translation. Those efforts must cover the serving seams and CUDA atomics, vector accesses, `cp.async`, fragment loads,
and TMA rather than being folded into this minimal evaluation.
