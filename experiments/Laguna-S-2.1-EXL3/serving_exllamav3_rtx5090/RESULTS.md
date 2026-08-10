# Laguna S 2.1 EXL3 native RTX 5090 result

Status: **PASS** using the direct ExLlamaV3 v1.4.1 `Generator`/`Job` API.

- Checkpoint: Laguna-S-2.1 EXL3, configured body target 1.98 bpw
- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB; driver 580.173.02
- Runtime: Torch 2.10.0+cu128; ExLlamaV3 1.4.1+cu128.torch2.10.0 at
  `4f8ad0121f483ba66a5336244a4c3b6d7210385e`
- Allocator: `PYTORCH_ALLOC_CONF=expandable_segments:True`
- Cache: q2 keys and values, 6,144 shared tokens
- Protocol: greedy 512-token prompts plus 128 output tokens at concurrency 1, 4, and 8
- Peak sampled device use: 29.839 GiB
- Peak Torch allocated/reserved: 28.984 / 28.994 GiB
- Model-load time: 10.630178 seconds

| Concurrency | Median output tok/s | Median TTFT | Median TPOT |
|---:|---:|---:|---:|
| 1 | 90.779871 | 232.8957 ms | 9.2591 ms |
| 4 | 170.333490 | 931.0355 ms | 16.3135 ms |
| 8 | 212.632830 | 1,891.3743 ms | 22.9533 ms |

Original evidence hashes:

- Native results: `5a6c53d5a971687472900c920c7f26c75119d3cab21c13ae5f7a7f5298b5bd22`
- Native version manifest: `449060fd6874d532f6a8f2a111b5c70553b84960cf429670c25a2197600089dc`

The initial run without the expandable-segments allocator setting OOMed. These measurements
apply only to the explicit configuration above. Raw per-request JSON and logs are intentionally
not stored in the repository.
