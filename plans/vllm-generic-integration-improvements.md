# Generic Emmy / vLLM Integration Improvement Plan

## Scope

Improve Emmy's current generic vLLM integration without depending on vLLM's
model-specific implementations.

The existing attention-split architecture remains:

```text
Emmy pre-attention program
→ vLLM RoPE, KV cache, and paged attention
→ Emmy post-attention program
```

The primary change is to make the boundary between those components zero-copy,
Torch-native, and aligned with vLLM's scheduler and CUDA-graph buckets.

## Status (2026-08-03) — Milestone A executed and MEASURED; Milestone B re-scoped by the result

Milestone A landed as a PR stack: A1 capture-aware no-clone `run_device_sym` (#453, merged, 5090-validated);
A2 post→pre chaining for all program families (#456); A3 slice-bound rider outputs (#457); A4 attention-output
aliasing for every tier (#458). All four are pointer/view changes only — no compiled-graph changes, goldens and
packs stay valid.

**Corrected copy accounting (found during #456 review):** the upload self-copy-skip fires on pointer equality,
and the EAGER path's protective output clone breaks it. So A2's chaining pays only under an outer capture
(composing with A1's views); A4's attention aliasing is the one change that also wins on eager steps (it makes
the UPLOAD side pointer-equal). The plan's original "eager chunk step" savings claim for chaining was wrong.

**Milestone measurement (rented 5090, gemma-4-12B-it, main@e07ea37f vs the stack tip, box-local tuned twin DB
on both arms):** chunk p2048/c8 cell TPOT 54.34/54.18 → 53.93/54.00 ms; over-bucket c64 cell TPOT 37.54/37.78 →
37.21/37.83 ms; TTFT within noise; alias-off ~0.4 ms behind alias-on; greedy chat outputs content-identical.
No regression anywhere; the win is at the noise edge — the removed copies are µs-class against 37–54 ms steps.

**Consequence for the roadmap (the First-task gate question, answered):** integration D2D copies are NOT where
the remaining serving gap lives. Milestone B's `bind_io` loses its TPOT justification; its surviving motivation
is Phase 5 (Torch-owned weights/workspaces → KV capacity, startup, TP). The remaining TPOT/TTFT gap belongs to
glue-kernel time and eager host overhead (gpu_lock + DLPack dispatch per program call) — i.e. compiler fusion
and WHOLE-CHUNK-STEP CAPTURE, which this measurement promotes ahead of Milestone B/C.

Session caveats worth keeping: the 5090 golden set was orphaned for the current fused-kind forms on the
measurement base (fix in PR #449; the A/B ran on a box-local 12-twin tuned DB instead), and the over-bucket c64
cell's req/s is bimodal by boot order (8 vs 22 req/s, symmetric across arms) — compare arms on median TPOT.

## Status (2026-08-13) — whole-chunk-step capture IMPLEMENTED (the promoted item); 5090 bench pending

The promoted follow-up landed as `EMMY_GEN_CHUNK_CAPTURE` (default on): `emmy serve --generate` now asks vLLM for
`cudagraph_mode: FULL` with token-count capture sizes spanning the prefill widths — the exact chunk width rides
the chunk twin, the rider top rides the chunk+decode row split (whose `out=` copies now record under capture),
every other rung rides the capture-aware symbolic programs — and selects `--attention-backend TRITON_ATTN`,
because mixed-batch FULL capture needs `AttentionCGSupport.ALWAYS` and FA2 declares uniform-batch support only
(vLLM silently downgrades FULL to FULL_DECODE_ONLY there). This activates Milestone A2's post→pre chaining on
chunk steps (the eager protective clone was the pointer-breaker) and removes the per-program host framing and
per-step staging D2D from mixed steps. Off under speculative decoding; MoE keeps its capture-size-1 ladder.
Validation per the serving findings protocol (small_c1 TTFT, c64 TPOT vs stock, 3 runs, greedy parity) is the
next step; the decode-attention swap FA2→triton is the one regression risk the A/B must price.

## Target architecture

```text
Current:

Torch tensor
→ DLPack/CuPy view
→ copy into Emmy-owned capacity buffer
→ Emmy kernels
→ DLPack view or clone
→ Torch tensor


Target:

Torch-owned input/output buffers
→ Emmy kernels bound directly to those addresses
→ the same Torch buffers consumed by vLLM
```

This design remains generic across decoder models that can be partitioned
around their attention operations.

## Current integration costs

### Input staging

`CompiledProgram.upload_prefix_device()` copies every external tensor into a
compiler-owned capacity buffer:

```text
emmy/compiler/backend/cuda/program.py
```

The existing self-copy check eliminates a transfer only when the producer and
consumer have already been manually wired to the same pointer.

### Output wrapping and cloning

`_Program.run_device()` avoids output clones during an outer vLLM CUDA capture,
but `run_device_sym()` always clones its outputs. Decode widths above the static
bucket can therefore capture multiple D2D copies per layer.

### Incomplete buffer chaining

Post-to-pre output chaining currently covers the static decode and M=1 program
families. It does not cover:

- Symbolic programs
- Static prefill programs
- Rider head/tail combinations

### Limited attention-output aliasing

vLLM can write paged-attention output into a supplied buffer through
`unified_attention_with_output`, but Emmy exposes a compatible post-program
input backing only for the M=1 tier.

### Rider concatenation

The chunk-plus-rider path joins separately computed tensors with `torch.cat`.
This introduces allocations and copy nodes for Q, K, V, and the post-layer
hidden state.

### Independent shape policies

Emmy selects one decode bucket, an optional M=1 tier, and an optional prefill
bucket. vLLM captures several decode batch sizes. A request can consequently
run a padded Emmy program even though vLLM has an exact capture size for the
step.

### Separate runtime ownership

Emmy's constants, activations, and workspaces are primarily CuPy-owned. vLLM
and Torch cannot manage or reason about them as ordinary model parameters and
buffers.

### Embedding-specific packing

The embedding model copies `positions` to CPU to determine sequence spans,
then runs the compiled trunk per sequence or as a padded batch. This prevents
the compiler from consuming vLLM's packed token representation directly.

## Priorities

| Priority | Change | Main benefit |
| --- | --- | --- |
| P0 | Remove symbolic-path clones under outer capture | Fewer D2D nodes for over-bucket decode |
| P0 | Chain all program families | Fewer per-layer hidden-state copies |
| P0 | Remove rider `torch.cat` operations | Fewer allocations and graph nodes |
| P0 | Generalize attention-output aliasing | Removes an attention-to-post copy per layer |
| P1 | Bind programs directly to Torch buffers | Removes the remaining staging copies |
| P1 | Add a generic Torch custom-op interface | Better vLLM compile/capture integration |
| P1 | Match program tiers to vLLM capture sizes | Less padded computation |
| P2 | Move weights and workspaces under Torch ownership | Better startup, VRAM, and distributed support |
| P2 | Add a packed-sequence embedding ABI | Removes CPU splitting and sequential execution |

## Phase 1: Complete the existing zero-copy mechanisms

This phase works within the current `CompiledProgram` buffer model.

### 1.1 Make symbolic output handling capture-aware

Change `run_device_sym()` to mirror `run_device()`:

```python
if torch.cuda.is_current_stream_capturing():
    program.run_once()
    return cached_torch_output_views

program.run_once()
return cloned_outputs
```

The no-clone path is safe only when:

- The outer graph fixes producer/consumer ordering
- Every output is consumed before its backing is reused
- The backing pointer remains stable
- No uncaptured caller retains the returned tensor across another invocation

Cache the Torch views before capture. Creating a new DLPack wrapper during an
active capture can negotiate stream state and invalidate capture.

### 1.2 Chain symbolic and static-prefill programs

Extend post-output to next-pre-input pointer chaining to:

- Symbolic pre/post programs
- Static prefill twins
- Every static decode tier added later

Validate:

- Same dtype
- Same logical layout
- Compatible capacity
- Sequential execution
- No concurrent use of the shared arena

### 1.3 Replace rider concatenation with slice-bound destinations

Allocate or select one contiguous destination tensor for each combined result.
Bind:

```text
prefill output → destination[:prefill_rows]
decode output  → destination[prefill_rows:total_rows]
```

Do this for:

- Q
- K
- V
- Post-layer hidden state

The caller should receive the complete destination tensor without running
`torch.cat`.

### 1.4 Generalize attention-output aliasing

Expose the post program's `attn_out` input backing for:

- M=1
- Every static decode tier
- Static prefill
- Symbolic execution

Use vLLM's output-taking attention call to write directly into this view.

Keep the current fallbacks for attention configurations that cannot safely use
an external output tensor, including unsupported query quantization or KV-scale
paths.

As with other output views, construct and cache the Torch wrapper before CUDA
capture.

## Phase 2: Direct external-buffer binding

> **2026-08-03:** the Milestone A measurement killed this phase's TPOT case (copies are µs-class); execute it
> only if/when Phase 5's ownership goals need the binding machinery, and prioritize whole-chunk-step capture
> and glue-kernel fusion first (see Status above).

The first phase removes avoidable copies through pointer rewiring. The next
phase removes the compiler-owned input/output staging model itself.

### 2.1 Add a bound-program API

Introduce an API conceptually similar to:

```python
bound = program.bind_io(
    inputs={
        "hidden": hidden_input,
        "residual": residual_input,
    },
    outputs={
        "hidden_out": hidden_output,
    },
    sym_values={"num_tokens": token_count},
)
bound.run()
```

Requirements:

- Validate device, dtype, shape, stride, alignment, and capacity
- Hold strong references to all owning Torch tensors
- Avoid a D2D copy when a tensor is directly compatible
- Fall back to the existing staging path when it is not compatible
- Rebuild TMA descriptors when a bound address or logical shape changes
- Cache bindings and descriptors for stable capture-size buffers
- Never rebind pointers inside an active CUDA capture

### 2.2 Torch-owned ping-pong hidden buffers

Allocate two persistent hidden-state buffers for each capture tier:

```text
layer 0: read A, write B
layer 1: read B, write A
layer 2: read A, write B
...
```

The post program reads:

- Attention output
- The current hidden buffer as the residual

It writes the next hidden state to the alternate buffer. This eliminates the
protective residual upload while retaining the original value until the post
program has finished.

### 2.3 Direct Q/K/V outputs

Bind the pre-program outputs to persistent Torch tensors that vLLM consumes
directly for:

- RoPE
- KV-cache update
- Paged attention

No output clone or DLPack negotiation should occur on the steady path.

### 2.4 Direct attention output

Bind the post-program attention input to a Torch-owned tensor and pass the same
tensor to vLLM's output-taking attention API.

The resulting layer has this ownership:

```text
hidden A (Torch)
  → Emmy pre
q/k/v (Torch)
  → vLLM RoPE and attention
attention output (Torch, also Emmy post input)
  → Emmy post with residual A
hidden B (Torch)
```

### 2.5 Pointer and descriptor caching

vLLM CUDA graphs provide stable addresses for a capture size. Create a bound
program once during warm-up and reuse it during capture/replay.

Cache keys should include:

- Device
- Program identity
- Token/capture tier
- Input and output addresses
- Dtype
- Stride/layout
- Symbolic shape values

For uncaptured eager execution where addresses vary, use either:

- A short-lived bound program with cached descriptors when possible, or
- The existing copy-based path as a safe fallback

## Phase 3: Torch custom-operator integration

Register generic operators such as:

```text
torch.ops.emmy.run_program
torch.ops.emmy.run_program_out
```

Use vLLM's direct custom-op registration or a dedicated `torch.library`.

Each operator must:

- Have a fake/meta implementation
- Declare mutated arguments
- Support dynamic token dimensions
- Execute on Torch's current CUDA stream
- Be CUDA graph capture-safe
- Perform no allocation or pointer rebinding during capture
- Have a native/reference implementation for correctness testing

The custom operator should invoke the direct-bound program, not merely wrap the
old copy-based runner.

### Expected effect

This primarily improves:

- vLLM graph and compilation awareness
- Eager and symbolic-prefill host overhead
- Capture initialization reliability
- Maintainability of the Torch/CuPy boundary

It will not reduce the internal Emmy kernel count. The compiled program remains
opaque to Torch, and its internal fusion remains Emmy's responsibility.

## Phase 4: Scheduler-aligned program tiers

Build static Emmy programs for selected vLLM capture sizes rather than relying
on a single padded decode bucket.

A candidate tier set is:

```text
1, 2, 4, 8, 16, 32, 64
```

The exact set should come from the active vLLM capture configuration and the
target workload.

### Tier rules

- Route exact widths to exact programs where profitable
- Route other widths to the smallest profitable covering tier
- Use the symbolic program when padding cost exceeds its dynamic-grid cost
- Share constants across every tier
- Share activation/scratch arenas where sequential execution permits it
- Store compiled tiers in the existing pack format
- Disable tiers that fail compilation or lose end-to-end benchmarks

### Automatic selection

Store the winner by:

- GPU architecture
- Model configuration hash
- Dtype
- Program fragment
- Token width
- Prefill/decode mode

The fallback for unmeasured combinations remains the symbolic Emmy path or the
existing safe implementation.

## Phase 5: Torch-owned weights and workspaces

Decouple program construction from weight loading.

### Target lifecycle

```text
1. Build or load the Emmy execution plans
2. Let vLLM load model weights
3. Bind the loaded Torch parameters to Emmy program constants
4. Allocate Torch-owned activation and scratch storage
5. Warm and capture the bound programs
```

This avoids having the runner independently load a full Hugging Face model
during vLLM model construction.

Benefits include:

- One checkpoint-loading path
- Better startup time
- Torch-visible parameter and workspace memory
- Less interaction with the CuPy memory pool
- Easier tensor-parallel sharding
- Easier pipeline-parallel ownership
- Cleaner tied-weight sharing
- More accurate vLLM memory and KV-cache planning

This phase is not expected to close TPOT by itself, but it removes substantial
operational and scaling limitations.

## Phase 6: Packed-sequence embedding integration

The embedding runner should consume packed vLLM inputs directly:

```text
input_ids[T]
positions[T]
sequence_offsets[B + 1]
```

Possible implementations:

1. Compile a packed-token trunk with variable-length attention.
2. Split the model around attention and use vLLM's packed attention backend.
3. As an intermediate step, pack and unpack padded batches entirely on GPU.

The steady path must avoid:

- `positions.cpu().numpy()`
- Python sequence-span construction
- One compiled forward per sequence
- Host-visible sequence-length synchronization
- `torch.cat` of per-sequence results
- Padding to an unnecessarily large global batch/sequence shape

This change is model-independent: it requires an attention/sequence ABI, not a
native implementation for each transformer family.

## Implementation order

### Milestone A: Low-risk copy removal

1. Make symbolic outputs capture-aware.
2. Cache all internal Torch views before capture.
3. Chain symbolic and static-prefill post/pre buffers.
4. Replace rider concatenations with shared slice outputs.
5. Extend attention output aliasing to every program family.

### Milestone B: True zero-copy execution

1. Implement external input/output binding.
2. Add Torch-owned hidden-state ping-pong buffers.
3. Bind Q/K/V outputs directly.
4. Bind attention output directly.
5. Retain copy-based fallback for unsupported layouts.

### Milestone C: Native Torch boundary

1. Register the generic Emmy custom operator.
2. Add fake/reference implementations.
3. Validate eager, compiled, and CUDA-graph modes.
4. Switch the model runner to the custom-op path.

### Milestone D: Scheduler alignment

1. Read vLLM capture sizes.
2. Build candidate exact-width tiers.
3. Benchmark and persist winners.
4. Route requests through the selected tier automatically.

### Milestone E: Runtime ownership and embeddings

1. Move weights and workspaces under Torch ownership.
2. Add vLLM-compatible sharding.
3. Replace the embedding CPU split with a packed-sequence ABI.

## Validation

### Correctness

- Compare every modified fragment with the existing runner.
- Compare greedy generation tokens end to end.
- Test mixed prefill/decode batches.
- Test every static tier and symbolic fallback.
- Test rider widths.
- Test supported sliding/global attention mixtures.
- Test eager and CUDA-graph execution.
- Test pointer-rebinding fallback paths.

### Capture safety

- No DLPack wrapping during active capture
- No allocation during capture or replay
- No TMA descriptor construction during capture
- Stable pointers for every captured tier
- No nested CUDA graph capture
- Correct current-stream ordering

### Profiling

For each benchmark row, record:

- TTFT
- TPOT
- Output throughput
- CUDA graph node count
- Kernel count
- D2D copy count and bytes
- Allocations after warm-up
- GPU memory held by Torch
- GPU memory held by CuPy
- Available KV-cache capacity

Use Nsight Systems to verify that the removed copies actually disappear from
the captured decode graph.

## Performance gates

### Milestone A

- No regression in correctness
- Fewer symbolic-path D2D copies
- No rider `torch.cat` allocations
- Attention-to-post copy removed for all supported tiers

### Milestone B

- No hidden-state staging copy between decoder layers
- No residual protection copy
- Q/K/V consumed directly by vLLM
- Attention output consumed directly by Emmy
- Steady-state model execution performs no D2D integration copies, except
  documented fallbacks

### Milestone C

- Custom-op path works in eager, `torch.compile`, and CUDA graphs
- No regression relative to the direct-bound runner
- Lower or equal host overhead for symbolic prefill

### Final

- Common-concurrency TPOT reaches or beats stock vLLM
- c64 throughput reaches or beats stock vLLM
- TTFT remains at least as good as the current tuned Emmy path
- KV-cache capacity is no worse than the current integration
- Unsupported cases fall back safely

## Expected limitations

These changes remove integration overhead but do not automatically reduce
Emmy's internal kernel count. The current graph-node difference also includes
compiler-generated norm, scale, reduction, and pointwise kernels.

If TPOT remains behind after D2D copies and padding waste have been removed,
the remaining work belongs in compiler fusion and kernel scheduling rather
than the vLLM integration layer.

## First task — DONE (see Status: measurement 2026-08-03)

Implement and benchmark the low-risk Milestone A changes before introducing a
new ABI:

1. Capture-aware no-clone `run_device_sym`
2. Symbolic/prefill post-to-pre chaining
3. Slice-bound rider outputs
4. Attention-output aliasing for all tiers

Compare graph nodes, D2D copies, TPOT, and throughput against the current
runner. The results will quantify how much of the remaining gap is integration
overhead before undertaking direct external-buffer binding.
