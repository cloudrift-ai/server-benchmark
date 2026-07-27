# V1: DiT Block Optimization with PyTorch Comparison

## Summary

Add experimental support for compiling and tuning one transformer block from
[`facebook/DiT-XL-2-256`](https://huggingface.co/facebook/DiT-XL-2-256). V1
will compare the identical pretrained block across Eager PyTorch,
`torch.compile`, and Emmy using the existing correctness and interleaved
benchmarking machinery.

The fixed FP16 workload will be batch 1 with hidden-state shape
`[1, 256, 1152]`, derived from the checkpoint's 32x32 latent grid, patch size
2, and 16x72 attention width
([configuration](https://huggingface.co/facebook/DiT-XL-2-256/blob/main/transformer/config.json)).

## Public Interface

- Add `--adapter {causal-lm,dit}` to `emmy compile`, `emmy run`, and
  `emmy tune`; default to `causal-lm` so existing behavior is unchanged.
- Support this primary command:

  ```bash
  emmy run facebook/DiT-XL-2-256 \
    --adapter dit \
    --layer 0 \
    --bench \
    --bench-backends eager,tcompile,emmy \
    --warmup 10 \
    --iters 100 \
    --json dit-layer0.json
  ```

- Require `--layer` for the DiT adapter, validate it against the checkpoint's
  28 blocks, and reject `--dynamic` in v1.
- Accept layers 0-27, but document layer 0 as the initial validation target.
- Add an `image` dependency extra containing `diffusers>=0.39,<0.40`; missing
  dependencies produce an installation hint for `.[compile,image]`.
- Preserve the existing comparison table and JSON backend schema, including
  latency and speedup relative to eager PyTorch.

## Implementation Changes

- Add a DiT trace adapter that:
  - Loads only the checkpoint's `transformer` subfolder through Diffusers
    `AutoModel`, not the VAE or scheduler.
  - Converts the selected block to FP16, sets evaluation mode, and selects
    PyTorch SDPA as its attention processor.
  - Creates deterministic inputs using seed 0: FP16 hidden states
    `[1, 256, 1152]`, timestep `500`, and class label `207`.
  - Returns the existing `(graph, module, args, kwargs)` bundle so weight
    binding, eager execution, accuracy checking, profiling, and benchmarking
    need no parallel implementation.
- Thread the adapter name through `load_or_trace` and the isolated benchmark
  worker's `trace_args` payload so the worker reconstructs the same block and
  inputs.
- Extend the PyTorch tracer for `aten.chunk`:
  - Materialize each static chunk as a separate `SliceOp` using FX-provided
    output shapes and cumulative offsets.
  - Resolve subsequent `operator.getitem` nodes to the correct slice.
  - Preserve PyTorch's uneven final-chunk behavior and reject dynamic chunk
    counts, nonconstant dimensions, and invalid tuple indices clearly.
  - Do not introduce a general multi-output IR or unrelated split operators
    in v1.
- Add `aten.bmm` as a matmul alias only if the normalized Diffusers SDPA trace
  still emits it; otherwise keep the operator surface unchanged.
- Reuse the existing fatal eager-correctness gate before latency reporting.
  All three backends use the same module and example inputs; CUDA-graph
  capture remains all-or-nothing so the table never mixes timing semantics.
- Make the same adapter available to `emmy tune`, allowing DiT block schedules
  to populate the RTX 4080 tuning database and later be reused by `compile`
  and `run`.

## Test and Acceptance Plan

- Add tracer tests for divisible and uneven `chunk`, correct `getitem` routing,
  eager-equivalent values, and explicit unsupported dynamic cases.
- Add a tiny randomly initialized Diffusers DiT block test with no network
  download to cover AdaLayerNorm-Zero, chunking, attention, GELU MLP,
  residuals, and weight binding.
- Add CLI tests for adapter dispatch, required layer, layer bounds, dependency
  errors, unsupported dynamic shapes, and unchanged default CausalLM behavior.
- Add a CUDA integration test that checks:
  - Emmy output passes the existing FP16 eager parity gate.
  - The benchmark includes Eager PyTorch, `torch.compile`, and Emmy.
  - JSON contains all three latency rows and valid eager-relative speedups.
- Keep the real pretrained-checkpoint test opt-in/performance-marked to avoid
  multi-gigabyte downloads in normal CI.
- Manual RTX 4080 acceptance requires layer 0 to complete without OOM, pass
  accuracy, and produce stable three-way results over at least 10 warmups and
  100 measured iterations.

## Assumptions

- V1 is block-level, analogous to the existing LLM `--layer` workflow; it does
  not generate an image or compile patch projection, final projection, VAE, or
  scheduler.
- MusicGen and speech models are deferred until the DiT adapter and
  modality-neutral comparison seam are proven.
- FP16 and fixed batch/shape are deliberate v1 constraints; BF16, dynamic
  resolution, full-denoiser execution, serving, and end-to-end image latency
  are follow-ups.
- The checkpoint's CC-BY-NC-4.0 license is documented alongside the
  experimental example.

## Future Development Ideas

### Phase 2: Complete DiT denoiser

- Compile all 28 transformer blocks and the conditioning/final-projection
  paths as one reusable denoising program.
- Add convolution or an equivalent patchify lowering for the input projection,
  plus the reshape/einsum coverage needed for unpatchifying the output.
- Bind the real latent, timestep, and class-label inputs used by the Diffusers
  pipeline and verify every denoising step against PyTorch.
- Benchmark one complete denoising step in addition to per-block timings,
  reporting latency, throughput, peak VRAM, and speedup against eager and
  `torch.compile`.

### Phase 3: Hybrid and end-to-end image generation

- Add a hybrid runtime that keeps the scheduler and VAE in PyTorch while
  executing the denoiser through Emmy.
- Compare complete image-generation latency for identical initial noise,
  scheduler settings, class label, guidance scale, and number of steps.
- Add deterministic output checks on final latents and decoded images, followed
  by perceptual metrics for optimizations that intentionally relax numerical
  precision.
- Add warm execution-plan and cubin caching so repeated generations do not pay
  trace or compile costs.
- Expose image-generation workloads through benchmark recipes, including
  resolution, batch size, step count, precision, and GPU as matrix dimensions.

### Phase 4: Broader image-model coverage

- Add reusable compiler support for Conv2d, GroupNorm, interpolation,
  up/downsampling, and transposed convolution to cover VAEs and U-Net-based
  diffusion models.
- Support dynamic batch size and latent height/width, including masked boundary
  tiles and shape-aware tuning records.
- Add BF16, FP8, and quantized-weight experiments with accuracy or perceptual
  quality gates appropriate to each precision.
- Introduce adapter implementations for text-conditioned DiT families,
  including cross-attention and multiple conditioning inputs, after the
  class-conditioned path is stable.
- Generalize `--adapter dit` into a documented adapter protocol for third-party
  PyTorch model loaders, example-input factories, dynamic-shape declarations,
  reference execution, and result metadata.

### Phase 5: Audio and speech

- Start audio support with `facebook/musicgen-small`: compile its transformer
  decoder while leaving the text encoder and EnCodec implementation in
  PyTorch.
- Add multi-codebook generation, KV-cache-aware execution, and audio-specific
  measurements such as tokens per second, real-time factor, time to first audio,
  and peak VRAM.
- Add Conv1d, transposed convolution, resampling, and FFT/STFT coverage before
  attempting to compile neural codecs or vocoders.
- Progress from text-to-audio to streaming TTS and then speech-to-speech,
  retaining PyTorch fallbacks for unsupported stages so each subsystem can be
  adopted incrementally.
- Add speech quality gates such as waveform error for exact paths and
  intelligibility/perceptual metrics for approximate paths.

### Cross-cutting improvements

- Record modality, model family, graph role, input geometry, precision, and GPU
  characteristics in tuning data so priors can distinguish LLM, image, and
  audio workloads.
- Add cold-start versus tuned-result reports to show how much performance comes
  from compilation alone versus Emmy's search.
- Track compilation time, cache-hit rate, peak host memory, peak device memory,
  and runtime latency as separate metrics.
- Provide an unsupported-operator inventory command that traces a model,
  reports coverage by runtime cost, and recommends the next operators to
  implement.
- Add regression dashboards containing correctness, latency, memory, and
  quality results across supported GPUs and model adapters.
