# Laguna S 2.1 EXL3 on one RTX 5090

These experiments qualify the verified 1.98 bpw local checkpoint on one 32 GB RTX 5090.
The native ExLlamaV3 lane is the recommended runtime and the only performance result here.
The custom Emmy/vLLM lane is deliberately constrained functional evidence; it is not a
throughput result or a claim of general vLLM compatibility.

Checkpoint accuracy was evaluated separately on an A100; see
[`quality_a100/RESULTS.md`](quality_a100/RESULTS.md).

## Checkpoint identity

| Field | Value |
|---|---|
| Intended Hub repository | `cloudriftai/Laguna-S-2.1-exl3` |
| Intended bitrate branch | `1.98bpw` |
| Hub commit | Not available when these local experiments ran |
| Source | `poolside/Laguna-S-2.1@00af5a51782109b587a3b3bbf11875e566036fa7` |
| Configured body allocation | 1.98 bpw |
| Shards | 4, totaling 30,191,374,574 bytes |
| Indexed tensors | 146,068 |
| Laguna correction-bias tensors | 47 |
| `config.json` SHA-256 | `dedf94f07f5a92935e43d7ce6b7710314ae26b37f476e41e447dc49ff340de7a` |
| `model.safetensors.index.json` SHA-256 | `dfc19be2e337430fc0e0c012e71a2b178d281e5d0c5d54e5ec11a235e69eef5c` |
| `quantization_config.json` SHA-256 | `3d0a4769f3ead9f2cf82e0e60dd28af72e2e7ce652177392ddd7a5adb529e1f2` |

The local benchmark snapshot was placed in a directory named
`96ab72f4afbcb028fc880d3215812ffb494ebd7a`. That value is a local directory identifier,
not a Hugging Face commit and not a content digest. The three file hashes above identify the
tested artifact until publication supplies an immutable Hub revision. The raw native manifests
retain the historical fields `checkpoint_revision` and `checkpoint.revision`; in these files
both fields contain this local identifier and make no Hub-revision claim.

## Evidence lanes

| Directory | Runtime | Scope |
|---|---|---|
| [`serving_exllamav3_rtx5090`](serving_exllamav3_rtx5090) | ExLlamaV3 v1.4.1 native `Generator`/`Job` API | Recommended runtime and performance evidence |
| [`serving_emmy_rtx5090_functional_smoke`](serving_emmy_rtx5090_functional_smoke) | Custom Emmy model under vLLM 0.23.0 | Non-eager functional smoke only |
| [`compiler_rtx5090`](compiler_rtx5090) | Emmy compiler through the custom serving integration | Partial compiler evidence and explicit remaining gaps |

## Native ExLlamaV3 result

The native recipe pins ExLlamaV3 v1.4.1 at commit
`4f8ad0121f483ba66a5336244a4c3b6d7210385e`, its CPython 3.12 / CUDA 12.8 /
Torch 2.10 wheel, and Torch `2.10.0+cu128`. The completed run used a q2 KV cache with
6,144 admitted tokens on an otherwise-idle RTX 5090 with driver 580.173.02.

The runner uses exact 512-token deterministic prompts and greedy 128-token outputs at
concurrency 1, 4, and 8, with 8, 24, and 48 prompts per recorded run. All requests in a wave
are enqueued before the first `Generator.iterate()`. Each cell has one discarded exact-shape
warmup wave and three recorded runs. The aggregate table reports the median of the three
recorded-run throughput values, median request latency, and maximum sampled device use.

| Concurrency | Output tok/s | Median TTFT | Median TPOT | Peak sampled device use |
|---:|---:|---:|---:|---:|
| 1 | 90.7799 | 232.896 ms | 9.259 ms | 29.794 GiB |
| 4 | 170.3335 | 931.036 ms | 16.313 ms | 29.813 GiB |
| 8 | 212.6328 | 1,891.374 ms | 22.953 ms | 29.839 GiB |

[`native-results.json`](serving_exllamav3_rtx5090/native-results.json) contains every recorded
request, prompt/output hashes, queue/prefill/generation timing, and aggregate statistics.
[`native-versions.json`](serving_exllamav3_rtx5090/native-versions.json) records the checkpoint
manifest, runtime/package versions, native scheduler source hashes, GPU/driver, cache settings,
model-load time, and load-memory metrics. Both files are byte-for-byte copies of the final local
run; consequently, their embedded `snapshot` and `versions_manifest` paths record the original
ignored evidence directory.

The recipe is intentionally local until an immutable Hub commit exists. It records the exact
absolute snapshot path on the authoring host and accepts an equivalent path through
`LAGUNA_EXL3_LOCAL_SNAPSHOT` when that variable is present in the workload shell. The basename
and all three manifest hashes are checked before the GPU is touched:

```bash
emmy bench experiments/Laguna-S-2.1-EXL3/serving_exllamav3_rtx5090 --local
```

## Custom Emmy/vLLM functional smoke

The final custom lane loaded the same local artifact with `EmmyGenModel`, FlashAttention 2,
FP16 KV cache, and `enforce_eager=False`. It captured one `FULL_DECODE_ONLY` CUDA graph at
size 1 and served three HTTP 200 completion requests. Two independent captured-graph requests
produced identical two-token output and token log probabilities; a separate ten-token prompt
plus four decode tokens also completed.

This lane used max model length 128, one max batched token, one sequence, host embedding,
decode bucket 1, prefill bucket 0, and M=1 tier. Those restrictions make it a functional smoke,
not a useful performance configuration. Native ExLlamaV3 remains the recommended RTX 5090
runtime. Exact responses, server logs, limits, and memory observations are preserved under
[`serving_emmy_rtx5090_functional_smoke`](serving_emmy_rtx5090_functional_smoke).
