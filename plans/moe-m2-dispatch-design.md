# MoE M2 — decode-dispatch recovery design

Decision doc for the M2 capture-recovery item in the MoE plan: which dispatch model replaces the per-expert Python
launch chain, measured against emmy's actual machinery. Baseline (M1, RTX 5090, OLMoE fp16): decode c=1 TPOT
17.9 ms vs stock vLLM 2.15 ms (8.3x), ~128 expert launches/step at ~117 us Python framing each, eager; the 12 MB
per-expert weight copy is ~15 us (not the wall); stock runs fused grouped-GEMM inside a FULL decode cudagraph at
the ~1.7 ms k-experts weight-bytes floor. Bring-up model OLMoE (E=64, k=8, 16 MoE layers, expert weights
12.6 MB/expert/layer fp16 -> 1.61 GB of expert reads per c=1 step); delivery target gpt-oss-20b fp8 (E=32, k=4,
24 layers -> 96 expert launches/step). RTX 5090 only.

## 1. Ground truth from the machinery (verified in-repo)

- **Whole-step capture today**: MoE serves eager only. `EmmyGenModel.__init__` rejects an MoE boot with capture
  enabled (`vllm_model_gen.py` ~line 230) and `emmy serve` auto-adds `--enforce-eager` for MoE
  (`commands/serve.py::_is_moe_model`). The one host-sync in the decode step is
  `combine_routed_experts`' `indices.unique().tolist()` (`gen_runner.py:159`) — everything else in
  `_moe_combine`/`_launch_expert` is device work on torch's stream. `torch.topk`, `index_add_`, `index_select`,
  and the score einsum are all fixed-shape, data-dependent-VALUES-only ops — capture-legal. The variable part is
  the LAUNCH SET (which experts, how many), not the math.
- **Captured programs bake pointers twice**: `capture_program_graph` freezes kernel arg pointers, and TMA
  descriptors bake the weight base address at encode (`_prebuild_descriptors` / `_descs_now`; the M1
  pointer-swap-under-descriptors wrong-results check confirmed it). The existing `_expert_swap_safe` gate
  (`gen_runner.py:325`) detects descriptor-free expert tiers via `"_desc" in launch.arg_names`.
- **cupy 14.1.1 exposes NO graph surgery**: `cupy.cuda.graph.Graph` has only
  `graph`/`graphExec`/`launch`/`upload`/`debug_dot_str`; the runtime binding has no `cudaGraphGetNodes`, no
  `cudaGraphExecKernelNodeSetParams`. BUT `cuda-python` / `cuda-bindings` 13.3.1 is already installed in
  venv-serving (vLLM dependency) and exposes the full set (`cudaGraphGetNodes`, `cudaGraphKernelNodeGetParams`,
  `cudaGraphExecKernelNodeSetParams`, plus the driver-API twins). `Graph.graph`/`Graph.graphExec` are plain
  handles, interoperable with those APIs. So option (a) is API-feasible without new dependencies — but see its
  verdict.
- **Kernel ABI is centralized and small**: `render_kernelop` (`ir/kernel/render.py:753`) derives the signature
  mechanically — `const T* <input>` params, outputs, then `const CUtensorMap* __restrict__ <desc>` (descriptors
  are already passed BY POINTER in device memory, not `__grid_constant__`), then `int` runtime args.
  `LaunchSpec.arg_names` + `program._launch` assemble args by name. An "operand indirection" change (swap one
  pointer param for table+index) is a narrow, well-contained ABI extension.
- **Gather machinery materializes**: the tracer maps `index_select`/`embedding`/`gather` to `GatherOp`, and
  `040_lift_gather` lowers a standalone `GatherOp` to a COPY kernel. There is no gather-fused operand load in the
  tile lowering; the mma tier's `fragment_epilogue` legality gate refuses data-dependent-index epilogue loads.
  So a traced bmm-over-gathered-weights runs, but pays a full materializing copy of the selected weights.
- **No hand-written serving kernels exist**: `RawKernel`/`RawModule` appear only in compiler internals
  (`nvcc.py`, `_tma.py`, `program.py`, `ir/kernel/ir.py`, `commands/run.py`). Everything serving runs is a
  compiled program with goldens, roofline audit, and pack coverage. A hand-written grouped-GEMM would be a first
  and would sit outside all three — strong precedent against.
- **Expert program shape**: `build_moe_split_wrapper`'s `expert(x, w_gate_up[2I,H], w_down[H,I])` — two matmuls
  plus the gated activation, i.e. ~2–3 kernels per program, with exactly 2 weight-pointer args. Three tiers exist
  (`moe.expert.one` / `moe.expert.bucket` / `moe.expert.sym`); M1 landed the M=1 tier as the c=1 hot path.

Overhead constants used in the cost models below: Python-framed expert launch ~117 us (measured M1); bare cupy
kernel dispatch ~5–15 us host; captured-graph kernel node ~1–2 us at replay; `cudaGraphExecKernelNodeSetParams`
~2–5 us host each; one host sync (router indices readback) ~15–30 us.

## 2. Option analysis

### (a) Fixed-sequence dispatch + graph-exec kernel-node param updates

Always replay k expert-program chains per layer; per step, rewrite each chain's weight-pointer args on the
instantiated graph-exec via `cuda-bindings`' `cudaGraphExecKernelNodeSetParams` (cupy exposes the handles, not
the API). TMA tiers: either pin the expert schedules descriptor-free (the `_expert_swap_safe` condition, made a
build-time guarantee via golden/knob pins) or prebuild E descriptors per weight ONCE at boot (64x2x128 B — the
descriptor arg is already a device pointer, so the update is the same SetParams call on that arg).

**The decisive flaw**: whole-step capture stays impossible. The pointers must be set from the router's indices
BEFORE launch, but layer L's routing is computed inside the step from layer L-1's output. (a) is therefore
per-layer by construction: sync router indices to host (~20 us), ~2k SetParams (~16 calls, ~50 us), one graph
launch per layer. This is also what exllamav3 actually does — its per-layer graphs are fixed chains with param
updates, not pattern caches.

- Cost model (c=1 OLMoE): 16 layers x (sync + 16 updates + launch) ≈ 1.6–2.4 ms host, serialized with the GPU at
  c=1, + ~1.5 ms GPU + the ~2.9 ms non-expert eager residual -> **TPOT ~4.5–6 ms**. 3–4x win, still 2–3x off
  stock, and the M2 exit's capture recovery is NOT achieved.
- Files: new graph-exec helper beside `program.py` (node->launch mapping must match by recorded arg POINTERS from
  `cudaGraphKernelNodeGetParams` — `cudaGraphGetNodes` order is not capture order, and all k slots share one
  kernel function); `gen_runner.py` per-layer chain capture + update path.
- Size ~300–450 lines; risk medium-high (driver-level arg repacking, cupy/cuda-bindings handle interop, CUDA 12
  cupy vs CUDA 13 bindings — use the driver-API entry points, which are version-agnostic against the installed
  driver).
- Verdict: **fallback only**. Right shape of host cost, wrong end-state.

### (b) Device-side indirection (argument indirection in the CUDA backend)

The kernel fetches its weight base pointer from a device table indexed by the router's indices tensor:
signature swaps `const half* w_gate_up` for `const half* const* w_table, const int* w_sel, int w_slot` and the
body prepends `const half* w_gate_up = w_table[w_sel[w_slot]];` — every downstream `Load` unchanged. The E-entry
pointer table is static (built once in `_ensure_device`); `w_sel` IS the router's `indices` device tensor;
`w_slot` is a static per-launch literal. TMA tiers get the identical trick on the descriptor arg
(`const CUtensorMap* const*` table of E prebuilt descriptors — legal because descriptors are already passed by
pointer, `render.py:819`). Everything is device-driven: launch exactly k slot chains per layer, fixed shapes,
zero host syncs -> **whole-step FULL_DECODE_ONLY capture works again** for MoE at capture size 1.

- Cost model (c=1 OLMoE): captured step = weight floor ~1.7 ms + ~400–600 graph nodes x 1–2 us (16 layers x
  8 slots x 2–3 kernels + dense) ≈ 0.5–1.2 ms -> **TPOT ~2.4–3.2 ms** (~1.1–1.5x stock). gpt-oss (96 slots,
  ~240 expert nodes) lands closer to the floor.
- At T>1 the fixed-slot layout becomes (t, j) pairs of M=1 chains — node count x T. Capture sizes stay capped
  small for MoE (ladder {1} first, {2,4} by measurement); larger decode batches stay eager.
- Files: `ir/kernel/render.py` (indirect-param rendering, ~30 lines), `backend/plan.py` (an indirect-operand
  field on `LaunchSpec`/the plan + pack encoding bump), `program.py::_launch` (arg assembly for table/index
  args), a compile-side option to mark the expert wrapper's weight inputs indirect (threaded through
  `CudaBackend` -> kernel lowering), `gen_runner.py` (fixed-slot decode combine + k program instances),
  `vllm_model_gen.py`/`commands/serve.py` (lift the MoE capture rejection for supported sizes).
- Size ~450–650 lines + GPU tests; risk medium. Kernel source changes -> new cubin keys; schedules and goldens
  are untouched (the indirection is ABI-level, invisible to the schedule search).
- Verdict: **the right end-state mechanism for decode**; also the seed of (d)'s per-block indirection.

### (c) Per-layer CUDA graphs keyed by routing pattern

Pattern space at c=1 is the k-subset of E: C(64,8) ≈ 4.4e9 (OLMoE), C(32,4) ≈ 36k (gpt-oss) per layer per step
— the cache essentially never hits on OLMoE, and re-capture costs a graph instantiate (~0.1–1 ms/layer/step),
worse than today's eager chain. exllamav3's actual mechanism is (a), not pattern caching. **Rejected.**

### (d) Sorted grouped-GEMM (one kernel per layer per projection)

vLLM `moe_align_block_size` style: sort token-expert pairs by expert (torch, fixed max shapes, capture-safe),
then one kernel whose M-blocks each read `expert_ids[block]` for the B base and gather their rows through
`sorted_token_ids`. Needs TWO new lowering mechanisms: per-M-tile operand indirection (a generalization of (b)
from a launch literal to a tile coordinate) and per-row gathered A loads inside a tile (breaks TMA/cp.async
staging legality; direct staging only). The mma tier's refusal is output-epilogue-side; the input side is simply
unschedulable today — this is real tile/loop-lowering work plus a new golden kind, multi-week. A hand-written
kernel instead has NO precedent in serving (verified — no RawKernel outside compiler internals) and would bypass
goldens, the roofline audit, and the pack; rejected in that form.

- Payoff when built: 2–3 launches/layer, expert weight reads deduped (the only option that wins c=8: fixed-slot
  at T=8 pays ~1.6x duplicate reads + ~2k nodes, stock's fused kernel pays neither), and it is THE prefill
  answer (jagged per-expert M in one pass).
- Verdict: **deferred to M3+**, promoted if M2 measurements show c>1 decode or prefill is the binding gap. Build
  it in the compiler on top of (b)'s indirection primitive, never hand-written.

### (e) Hybrids / cheaper options found

- **(e1) Gather-staged fixed slots — ZERO compiler changes, capture-safe today.** Per MoE layer, one
  `torch.index_select(W3d, 0, indices.flatten())` stages the k selected experts' weights into one persistent
  [k,2I,H]+[k,H,I] staging pair (101 MB total, shared across layers — layers are stream-ordered), and k
  pre-built instances of the M=1 expert program have their `w_gate_up`/`w_down` arrays REWIRED once at boot onto
  the staging slices (the post->pre chaining precedent: `program.arrays[name] = view`). Fixed k launches, fixed
  einsum combine, no `unique()`, no host sync -> whole-step capture works with today's kernels. Cost: staging
  triples expert weight traffic (gather read + write + kernel re-read; 101 MB/layer > the 5090's L2) ->
  +~1.6–2 ms GPU -> **TPOT ~4–5 ms** at c=1. The k instances build from the stored plan (no re-trace; buffers are
  KB-scale at M=1, arena-excluded so slots don't alias).
- **(e2) One bmm-style traced program with a 3-D gathered weight input**: traceable (GatherOp exists), but the
  gather lowers to a materializing copy kernel — identical traffic to (e1) with less control, and a
  gather-fused operand load does not exist in the tile lowering (that IS (b)/(d)). No advantage over (e1).
- **(e3) C-level launcher / slimmer Python framing only**: framing floor ~10–20 us/launch -> 128 launches ≈
  2–2.5 ms + syncs -> TPOT ~5–6 ms eager, capture still off. Dead end as a target, but the framing slimming
  falls out of (e1)/(b) anyway (prebuilt launch lists, no per-call dict/feed churn).

## 3. Recommendation — primary (b) reached through (e1), fallback (a)

**Stage 1 (land first): (e1) gather-staged fixed-slot decode.** No compiler changes; restores FULL_DECODE_ONLY
at capture size 1 (the c=1 TPOT headline) and builds ALL the serving/vLLM plumbing stage 2 needs (fixed-slot
combine, k instances, capture-enable, ladder capping, tests). Expected c=1 TPOT ~4–5 ms (17.9 -> ~4x).

**Stage 2 (the M2 exit): (b) argument indirection.** Replace the staging gather with the in-kernel pointer-table
read (weight tables + descriptor tables built in `_ensure_device`); slot programs read the E-tensor directly.
Removes the 3x staging traffic and the gather nodes. Expected c=1 TPOT ~2.4–3.2 ms (~1.1–1.5x stock; the
remaining gap is graph-node overhead, which shrinks on gpt-oss's 96 slots).

**Fallback if (b)'s compile-side threading slips:** (a) per-layer captured chains + `cuda-bindings` SetParams —
eager, ~4.5–6 ms, no capture recovery, but bounded and API-verified. **Deferred:** (d) grouped pass for c>1
decode and prefill, M3+, on top of (b)'s primitive.

Decode routing after M2: T=1 captured fixed-slot; 1<T<=bucket eager routed dispatch (today's path, slimmed);
prefill unchanged (below). `combine_routed_experts` stays the parity oracle for the routed path; the fixed-slot
combine gets its own parity test against it.

## 4. Prefill lane (M2 item 1, unchanged in mechanism)

At a 2048-token chunk, OLMoE hits ~all 64 experts/layer at mean per-expert M ≈ 256 (gpt-oss: 2048·4/32 = 256
exactly) -> ~1024 expert launches/chunk. Per-launch GPU work is ~35 us (3.2 GFLOP + 12.6 MB) vs ~117 us Python
framing — **prefill FFN is launch-bound too (~3x)**, so the symbolic path alone is NOT fine at prefill widths.
But no new dispatch mechanism is needed: a static expert twin at M=256 (pad up / row-split down, `.dynM`
symbolic fallback for stragglers, per the plan) served via per-program captured replay
(`capture_program_graph`/`replay_program_graph` — one host call, ~30 us framing) brings dispatch to parity with
the GPU work, where it hides. Seed the `.dynM` + m256 expert goldens here (the M1-skipped item). (d) later
removes the jag entirely. Measure chunk TTFT vs stock as the M2 exit datum.

## 5. Implementation steps

1. Fixed-slot decode path in `gen_runner.py`: k instances of `moe.expert.one` built from the stored plan
   (`_compile_split(plan=...)`, no arena), staging pair + boot rewire, capture-safe `_moe_combine_slots`
   (index_select -> k replays -> score einsum), routed path kept for T>1/prefill. Verify: parity vs
   `combine_routed_experts` on random routings (GPU test).
2. Capture enablement: lift the `EmmyGenModel.__init__` MoE rejection behind the fixed-slot capability; MoE
   capture ladder capped at size 1 in `commands/serve.py` (`_gen_graph_args` path) instead of forcing
   `--enforce-eager`. Verify: capture-validated live-replay test (the `test_gen_capture_gpu` pattern), greedy
   parity unchanged.
3. Measure stage 1 A/B vs stock (c=1 TPOT, c=8 aggregate) — the (d)-promotion datum.
4. Indirect-operand mechanism: `render_kernelop` param swap + preamble, plan/`LaunchSpec` field + pack encoding,
   `_launch` arg assembly, `CudaBackend` compile option on the expert wrapper, descriptor tables for TMA tiers.
   Verify: kernel-source digest for non-indirect kernels unchanged; indirect expert program matches the direct
   one bit-exact on all E experts.
5. Swap stage 1's staging for the tables; delete the staging pair. Re-measure; seed expert goldens (now
   dispatch-unbound); re-run the boot roofline audit and the KV-admission check (staging removal returns
   ~100 MB).
6. Prefill: m256 expert twin + captured per-expert replay + `.dynM` goldens; chunk TTFT A/B.

## 6. Risks

- vLLM capturing the torch router/combine alongside the runner's raw launches: the dense path already validates
  mixed torch+emmy capture; the new ingredients (topk, index_select) are standard capture-legal ops. Test first
  at stage 1 (cheap to back out).
- Graph-node overhead estimate (1–2 us/node) is assumed, not measured on the 5090 — stage 3's measurement gates
  stage 2's expected TPOT claim.
- Pack format bump for the indirect-operand field (the `_encode_load_ops`/pack-vocabulary risk pattern): version
  the field; old packs fall back to full compile.
- c=8 stays ~1.5–2x off stock even after stage 2 (duplicate reads + node count) — that is (d)'s brief, stated
  in the M2 exit as a known deferral, not a regression.
- The M=1 expert tier must stay compiling (its failure contract currently falls back to bucket/symbolic — the
  fixed-slot path needs a loud gate instead of a silent tier drop).

## 7. M2 exit criteria

- OLMoE c=1 decode TPOT <= 4.5 ms after stage 1, <= 3.2 ms after stage 2 (from 17.9), FULL_DECODE_ONLY restored
  at capture size 1; greedy parity + GSM8K-sanity unchanged.
- Prefill 2048-chunk TTFT within 1.5x stock with the m256 twin + captured replay; expert goldens seeded.
- Measured c=8 gap recorded with the (d) go/no-go decision for M3.

