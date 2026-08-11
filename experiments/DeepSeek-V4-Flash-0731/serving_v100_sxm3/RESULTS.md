# DeepSeek V4 Flash 0731 serving qualification on V100 SXM3

The exact checkpoint revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062` was qualified on sixteen V100 SXM3
32 GB GPUs with TP8, PP2, FP16 weights, FP8 KV cache, a 4096-token context, and eight concurrent requests. The
runtime is 1Cat revision `d76126608155c334df7c2fb9b75096f879624859`, with the original 22/21 transformer-layer
pipeline split. The qualified canonical local image is
`cloudriftai/1cat-vllm-deepseek-v4-flash-0731:1.2.3-d76126608` (image ID
`sha256:276240257b224097876b5b6db8f0d32484dff6a6f168d6b03d6df188e5c65bc1`).

## Cache bake and zero-JIT gate

The initial request compiled the 16 expected Triton kernel functions:

```text
_compressed_slot_mapping_kernel
_build_prefill_chunk_metadata_kernel
_build_c128a_topk_metadata_kernel
_compute_prefill_metadata_kernel
_sm70_qnorm_rope_kernel
quantize_and_insert_k_kernel
_dequantize_and_gather_k_kernel
_combine_topk_swa_indices_kernel
_sm70_sparse_gathered_kernel
_save_partial_states_kernel
_fused_kv_compress_norm_rope_insert_indexer_attn
_fill_short_context_topk_indices
_fused_kv_compress_norm_rope_insert_sparse_attn
_compute_swa_indices_and_lens_kernel
_sm70_sparse_paged_fp8_kernel
_compute_global_topk_indices_and_lens_kernel
```

Long prefill and decode additionally reached `_weighted_query_kernel`, `_dequant_contiguous_index_k_kernel`, and
`_dequant_paged_index_k_kernel`. The honest request-time inventory is therefore 19 functions, not 16. CUDA graph
qualification added one generated Inductor Triton specialization and another sparse-paged-attention specialization.
The later greedy-logprobs and sampling qualification reached `_topk_topp_kernel` and the generated
`triton_red_fused_ge_sum_0`, taking the complete qualified request inventory to 21 functions. It also added six
specializations of existing request functions. The final image contains 862 Triton cache files and 64 TileLang
cache files:

| Cache | Manifest SHA-256 |
| --- | --- |
| Triton | `1a9f55e12151dba0008df765e45adbda7fbbb0374084a426cf6fd844647fb489` |
| TileLang | `bf45091a4e4398fb6d4df9fc3bc24ee2dcbda3c0edf44b2d1f5acc0d78693805` |
| TileLang JIT | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

The deterministic [cache index](cache-index.json) maps all 19 matrix/tool request functions and both added quality
functions to sorted Triton cache keys and file counts, records graph-generated and launcher-only entries, and
enumerates all 16 TileLang keys. All 103 cached Triton PTX files target `sm_70`. The TileLang-JIT cache and its
normalized manifest are both empty; the rebuild replaced the earlier one-line `sha256sum` sentinel representation.

The cache was populated to a fixpoint with prompt/output lengths 32/8, 256/32, 1024/64, and 3072/128 at
concurrency 1, 2, 4, and 8, plus a deterministic structured `multiply(17, 19)` tool call. A preliminary diagonal
gate correctly failed when the unseen 32/8 concurrency-4 specialization invoked `ptxas`; the full cross-product
was then baked before the final image was built.

The original guarded gate passed those 16 cells and the tool call, but a later short-prompt greedy-logprobs probe
correctly found one unbaked `_dequantize_and_gather_k_kernel` specialization. An unguarded quality pass populated 66
files for ten cache keys: eight kernel specializations and two launchers. The active-expert experiment generated 25
additional files, but those were deliberately excluded because that route failed its quality gate.

The rebuilt active-expert-off image then started in a fresh container with fail-closed guards over both `ptxas`
locations, `nvcc`, and `ninja`. Startup, the complete 16-cell matrix, 3968/128 at concurrency 1 and 8, greedy
logprobs, two sampling probes, and the structured tool call all passed. The compiler log remained zero bytes and all
three normalized cache manifests stayed byte-identical. The runtime's monitor wraps `JITFunction.compile()` before
the disk-cache decision, so its warnings remain cache-load false positives; the compiler guard and manifests are
authoritative.

## Serving benchmark

Each cell submitted one simultaneous batch of exact random-token prompts with greedy sampling and EOS ignored. These
are one-batch qualification measurements, not confidence intervals. The eager baseline used the same image source,
model revision, and serving configuration with `--enforce-eager`; graph results are steady-state measurements after
startup capture and NCCL initialization.

| Prompt/output | C | Graph TTFT (ms) | Graph TPOT (ms) | Output tok/s | Total tok/s | TPOT vs eager |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32/8 | 1 | 417.23 | 97.75 | 7.26 | 36.28 | 3.34x |
| 32/8 | 2 | 566.89 | 121.34 | 11.29 | 56.44 | 3.43x |
| 32/8 | 4 | 1016.91 | 162.06 | 14.36 | 71.82 | 4.24x |
| 32/8 | 8 | 1429.69 | 207.78 | 21.78 | 108.90 | 3.66x |
| 256/32 | 1 | 874.94 | 103.83 | 7.81 | 70.33 | 3.57x |
| 256/32 | 2 | 1138.37 | 126.46 | 12.65 | 113.84 | 2.94x |
| 256/32 | 4 | 1806.81 | 158.92 | 18.79 | 169.11 | 2.38x |
| 256/32 | 8 | 2518.79 | 210.05 | 28.17 | 253.53 | 1.66x |
| 1024/64 | 1 | 1190.33 | 118.96 | 7.37 | 125.27 | 3.05x |
| 1024/64 | 2 | 1537.40 | 150.90 | 11.47 | 194.92 | 2.17x |
| 1024/64 | 4 | 3074.81 | 175.20 | 18.03 | 306.52 | 1.90x |
| 1024/64 | 8 | 3528.52 | 222.34 | 29.09 | 494.59 | 1.55x |
| 3072/128 | 1 | 2850.90 | 141.89 | 6.13 | 153.32 | 2.50x |
| 3072/128 | 2 | 2348.93 | 172.35 | 10.53 | 263.27 | 2.13x |
| 3072/128 | 4 | 4100.82 | 201.75 | 17.11 | 427.73 | 1.76x |
| 3072/128 | 8 | 5919.26 | 262.10 | 25.92 | 648.01 | 1.41x |

CUDA graphs improved TPOT in every cell by 1.41–4.24x. Graph TTFT was within 2% of eager at concurrency 1 and was
generally lower at higher concurrency. vLLM selected `FULL_AND_PIECEWISE`, captured decode graphs for batch sizes
1, 2, 4, 8, and 16, and bounded SM70 attention graphs to a 2048-token context bucket. The graph pool used about
0.60 GiB per PP0 rank and 0.58 GiB per PP1 rank. Removing `--enforce-eager` is therefore qualified.

## Profile and pipeline split

An eager Torch trace used 1024-token prompts, 16 output tokens, and request concurrency 1, 2, 4, and 8. The table
reports GPU-kernel time on the TP0 rank in each pipeline stage across the mixed trace. These are attribution totals,
can overlap in wall-clock terms, and are not a deployed graph-mode hotspot profile.

The original parser classified every NCCL broadcast as TP. Raw kernel arguments instead show 69 one-integer
`ncclDevKernel_Broadcast` calls over PP ranks 0 and 8: this is the sampled-token PP control path. Separating it gives:

| Pipeline stage | Execute annotation (ms) | TP collectives | PP token broadcast | PP activation | Attention | KV | MoE | Other compute |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PP0, 22 layers | 64602.07 | 19370.89 | 29963.13 | 1.21 | 2327.52 | 81.96 | 6614.82 | 1836.81 |
| PP1, 21 layers | 31654.70 | 15926.65 | 0.64 | 2175.25 | 2404.65 | 81.82 | 6437.70 | 1825.47 |

The PP0 broadcast duration is receiver arrival skew and synchronization wait for one integer, not 29.96 seconds of
link transfer. The custom all-reduce kernel is similarly barriered, so its duration also includes rank-arrival skew.

The raw execute-annotation occupancy before any split change was:

| Requested C | PP0 annotation/span (ms) | PP0 occupancy | PP1 annotation/span (ms) | PP1 occupancy |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 12615.3 / 17746.3 | 71.1% | 6363.4 / 14495.6 | 43.9% |
| 2 | 13384.3 / 13776.0 | 97.2% | 6520.3 / 12643.2 | 51.6% |
| 4 | 16511.2 / 16911.0 | 97.6% | 7821.8 / 15308.6 | 51.1% |
| 8 | 19504.8 / 19903.0 | 98.0% | 9223.7 / 17785.3 | 51.9% |

These annotations contain compute and collective waiting, including the PP control broadcast. They therefore do not
measure PP compute utilization or bubbles. Attention, KV, MoE, and other compute are nearly balanced, so there is no
evidence for moving a layer. The 22/21 split is retained provisionally; a split change remains gated on a graph-mode
measurement that separates stage compute, activation communication, token synchronization, and idle time.

### TP16/PP1 trial

TP16/PP1 cannot serve this checkpoint with the current model geometry. The config has eight output groups, and the
attention path computes `n_local_groups = o_groups // tensor_parallel_size`; TP16 therefore produces zero local
groups. The qualified TurboMind lane failed during weight preparation with `groups=0, weight=(512, 4096)` before
profiling or graph capture. Its compiler guard stayed empty.

A diagnostic automatic-backend fallback reached the profile run but failed at `reshape([4096, 0, -1])`. It also
generated a previously unseen `_sm70_inverse_rope_kernel`, so it would violate the baked-cache contract even if the
zero-group error were repaired. No TP16 benchmark or tool-call result is valid. TP8/PP2 and the 22/21 split remain
the qualified configuration.

### PP scheduling and collective trial

Source inspection confirmed that asynchronous scheduling is already enabled and the PP2 batch queue already allows
two concurrent batches. The remaining low-risk collective switch was `--disable-custom-all-reduce`, which changes
the TP path from custom all-reduce with PYNCCL fallback to PYNCCL only. A guarded fresh-container comparison produced:

| Prompt/output, C8 | Duration | Output throughput | TTFT | TPOT |
| --- | ---: | ---: | ---: | ---: |
| 1024/64 | +18.65% | -15.72% | +79.92% | +1.75% |
| 3072/128 | +6.23% | -5.87% | +92.10% | -9.15% |

The long-context decode TPOT gain did not offset the prefill and end-to-end regressions. The tool call still passed,
the compiler guard remained empty, and the baked Triton and TileLang caches did not change. NCCL-only is rejected;
the qualified custom-all-reduce path with PYNCCL fallback is retained.

### Active-expert B1 trial

The largest remaining kernel opportunity was the SM70 MXFP4 active-expert B1 route, which skips empty expert
launches for decode batches up to eight tokens. A matched active-off/active-on graph-mode trial used the same image,
TP8/PP2 split, and all other experimental flags disabled. Each side ran three repeats of eight cells; all 108
candidate requests completed with exact lengths and no failures.

| Prompt/output | C | Baseline TPOT (ms) | Active TPOT (ms) | TPOT delta | Output tok/s delta | TTFT delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32/8 | 1 | 97.50 | 41.27 | -57.67% | +47.03% | +6.73% |
| 32/8 | 8 | 210.47 | 98.54 | -53.18% | +42.63% | -0.85% |
| 1024/64 | 1 | 118.93 | 63.30 | -46.77% | +72.64% | +0.45% |
| 1024/64 | 8 | 229.16 | 113.27 | -50.57% | +76.89% | -1.86% |
| 3072/128 | 1 | 141.78 | 86.85 | -38.75% | +58.47% | +0.35% |
| 3072/128 | 8 | 233.89 | 128.69 | -44.98% | +70.80% | -1.82% |
| 3968/128 | 1 | 141.86 | 86.09 | -39.31% | +60.78% | -4.53% |
| 3968/128 | 8 | 226.47 | 125.17 | -44.73% | +71.33% | +0.22% |

Performance is accepted as a real route hit, but release qualification is withheld. In a matched temperature-zero
probe both routes chose first token ` Paris`, then the active route flipped token two from `.` to `.",`. The
same-context log probability for `.` moved by 0.6112, only one of 32 generated tokens remained position-aligned, and
the active sampling continuation was qualitatively weaker. Tool calling still matched exactly. Existing operator
evidence proves isolated W13/W2 and permute parity, but not a composed dense-versus-active B1 CUDA-graph replay.
`VLLM_SM70_MXFP4_MOE_ACTIVE_EXPERT_B1` therefore remains disabled, and its three compact-sort specializations plus
launcher were excluded from the final image.

## Artifacts and publication

Machine-readable evidence is in [qualification.json](qualification.json). Raw benchmark JSON, fresh-container logs,
cache manifests, compiler-guard logs, and 17 compressed Torch traces remain on the VM under
`/home/riftuser/onecat-dsv4-0731/optimization`; the rejected parallelism and collective trials are under
`tp16-pp1-trial/` and `pp2-nccl-trial/`; the canonical-image fresh-container gate is in
`quality-zero-jit/`, and the active-expert evidence is in `active-expert-trial/`. The prior canonical image remains
locally tagged `pre-quality-bake`; no image was deleted. The final image is retained locally on the V100 node and
has not been pushed. `emmy publish --dry-run` passed its local
naming and metadata checks, then stopped at the registry collision gate because Docker Hub returned
`insufficient_scope` for the absent or inaccessible repository. No publication approval or credentials were
requested.
