# DeepSeek V4 Flash 0731 serving qualification on V100 SXM3

The exact checkpoint revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062` was qualified on sixteen V100 SXM3
32 GB GPUs with TP8, PP2, FP16 weights, FP8 KV cache, a 4096-token context, and eight concurrent requests. The
runtime is 1Cat revision `d76126608155c334df7c2fb9b75096f879624859`, with the original 22/21 transformer-layer
pipeline split. The qualified local image is
`cloudriftai/onecat-vllm-deepseek-v4-flash-0731:sm70-d76126608-jitfree` (image ID
`sha256:970b8c418271d443337acadb5bc7f7b3a7d470d4bb033c79c0cd8e666a4b8a78`).

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
The final image contains 796 Triton cache files and 64 TileLang cache files:

| Cache | Manifest SHA-256 |
| --- | --- |
| Triton | `9b2d72f0a91dad55a0f924e2fae59446e4daef8e319c6b0bdf4856e1f446e992` |
| TileLang | `bf45091a4e4398fb6d4df9fc3bc24ee2dcbda3c0edf44b2d1f5acc0d78693805` |
| TileLang JIT | `abcfa6a9d4df344d1781bc2560b5e4cdcae08b39ed303063535e7e1e926a304a` |

The cache was populated to a fixpoint with prompt/output lengths 32/8, 256/32, 1024/64, and 3072/128 at
concurrency 1, 2, 4, and 8, plus a deterministic structured `multiply(17, 19)` tool call. A preliminary diagonal
gate correctly failed when the unseen 32/8 concurrency-4 specialization invoked `ptxas`; the full cross-product
was then baked before the final image was built.

The final zero-JIT gate started a fresh container from that image with fail-closed guards mounted over both `ptxas`
locations, `nvcc`, and `ninja`. Startup, all 16 benchmark cells, and the structured tool call completed with zero
compiler invocations, and the live cache manifests remained byte-identical to the image labels. The runtime's
Triton monitor hooks `JITFunction.compile()` before Triton decides between a disk-cache load and backend compilation,
so it still reports cache loads as “JIT compilation.” The empty compiler-guard log and unchanged manifests are the
authoritative zero-JIT evidence.

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

An eager Torch trace used 1024-token prompts, 16 output tokens, and concurrency 1, 2, 4, and 8. The table reports
GPU-kernel time on the TP0 rank in each pipeline stage across the mixed trace; categories are attribution totals and
can overlap concurrent execution in wall-clock terms.

| Pipeline stage | Active (ms) | TP collectives | PP collectives | Attention | KV | MoE | Other compute |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PP0, 22 layers | 64602.07 | 49334.01 | 1.21 | 2327.52 | 81.96 | 6614.82 | 1836.81 |
| PP1, 21 layers | 31654.70 | 15927.29 | 2175.25 | 2404.65 | 81.82 | 6437.70 | 1825.47 |

Measured execute-annotation occupancy before any split change was:

| C | PP0 active/span (ms) | PP0 utilization | PP0 bubble | PP1 active/span (ms) | PP1 utilization | PP1 bubble |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 12615.3 / 17746.3 | 71.1% | 28.9% | 6363.4 / 14495.6 | 43.9% | 56.1% |
| 2 | 13384.3 / 13776.0 | 97.2% | 2.8% | 6520.3 / 12643.2 | 51.6% | 48.4% |
| 4 | 16511.2 / 16911.0 | 97.6% | 2.4% | 7821.8 / 15308.6 | 51.1% | 48.9% |
| 8 | 19504.8 / 19903.0 | 98.0% | 2.0% | 9223.7 / 17785.3 | 51.9% | 48.1% |

Attention, KV, MoE, and other compute are already nearly balanced across the two stages. The large difference is in
collective behavior and PP1 waiting, which moving one transformer layer cannot correct. The 22/21 split is retained;
the next optimization target is pipeline communication and scheduling, not layer reassignment.

## Artifacts and publication

Machine-readable evidence is in [qualification.json](qualification.json). Raw benchmark JSON, fresh-container logs,
cache manifests, compiler-guard logs, and 17 compressed Torch traces remain on the VM under
`/home/riftuser/onecat-dsv4-0731/optimization`. The final image is retained locally on the V100 node. It has not been
pushed: the release workflow requires explicit human approval before publication, and the earlier source-image push
also showed that the current Docker credentials lack the `cloudriftai` namespace scope.
