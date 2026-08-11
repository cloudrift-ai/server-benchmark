# Gemma 4 12B RTX 5090 article reproduction

Status: partial serving reproduction plus current-code kernel qualification. The article's highest-concurrency stock
cell reproduces within 1.1% throughput on the supplied workstation. The complete 18-cell serving grid was not run
because its unchanged memory envelope no longer boots beside the workstation's graphical session and the bounded
qualification window could not accommodate all three long-generation lanes.

## Scope

- Date: 2026-08-11
- Article: [Outperforming vLLM and Llama.cpp on Gemma4-12B](https://riftstack.ai/research/optimizing-gemma-4-12b-rtx)
- Performance checkpoint: `google/gemma-4-12B-it`
- Resolved checkpoint revision: `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`
- Hardware: 1× `NVIDIA GeForce RTX 5090`, compute capability 12.0, driver 580.173.02
- Stock image: `vllm/vllm-openai:v0.23.0`, digest
  `sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f`
- Workload: 256 input tokens, 256 output tokens, concurrency 64, 256 prompts, seed 0, greedy decoding, ignored EOS,
  prefix caching disabled

The article uses the instruction-tuned checkpoint. The separately requested `google/gemma-4-12B` base checkpoint is
qualified in [`serving_base_rtx5090`](../serving_base_rtx5090/RESULTS.md); its continuation endpoint and output
semantics are not interchangeable with this chat benchmark.

## Memory-envelope correction

The unchanged article recipe requests `gpu_memory_utilization: 0.96`. On this workstation, Xorg and the active desktop
retain about 941 MiB. vLLM then sees 29.86--29.91 GiB after imports, below the 30.07 GiB requested at 0.96. Another
attempt reached sampler warm-up but exhausted memory while constructing vLLM's default 256 dummy requests.

The measured lane changed only two capacity controls:

- `gpu_memory_utilization: 0.95`, the highest clean setting with the active desktop left intact
- `max_concurrent_requests: 64`, matching the protocol's actual maximum instead of warming 256 unused sampler slots

Prompt counts, token lengths, concurrency, model, image, seed, decoding, and cache policy are unchanged. The adjusted
recipe is preserved in [`reproduction_2026-08-11/recipe.yaml`](reproduction_2026-08-11/recipe.yaml).

## Serving result

| Metric | Published stock vLLM | Reproduction | Delta |
| --- | ---: | ---: | ---: |
| Output throughput | 1,435.9 tok/s | 1,420.7 tok/s | -1.06% |
| Median TTFT | 1,513 ms | 1,445 ms | -4.49% |
| Median TPOT | 27.7 ms | 28.82 ms | +4.04% |
| Successful requests | 256 | 256 | 0 |

The raw structured result is [`stock_c64.json`](reproduction_2026-08-11/stock_c64.json), with the full rendered Emmy
benchmark record in [`stock_c64.txt`](reproduction_2026-08-11/stock_c64.txt). A 256/256 concurrency-1 diagnostic also
passed three repeats at 60.73 ± 0.02 output tok/s, 57.09 ± 0.30 ms median TTFT, and 16.30 ± 0.01 ms median TPOT; the
article intentionally omits that synthetic single-stream point.

## Current-code kernel inventory

The repository's existing `kernels_rtx5090` command experiment stages the current worktree and measures every Gemma 4
golden realization at deployable O3 against eager PyTorch and `torch.compile`, in standard and fast-math lanes. This
is the branch's direct kernel-regression gate; serving above measures the released article image instead.

The current inventory contains 309 card-local realizations per lane, so the complete cold-cache run would exceed the
qualification window. A targeted replay used the same existing command utility and measured the four MLP realizations
that dominate the article's long-prompt lane:

| Realization | Lane | Eager (us) | `torch.compile` (us) | Emmy (us) | Emmy vs eager |
| --- | --- | ---: | ---: | ---: | ---: |
| `mlp_geglu.m32` | standard | 156 | 153 | 174 | 0.90x |
| `mlp_geglu.m32.lin` | standard | 162 | 160 | 173 | 0.94x |
| `mlp_geglu.m4096` | standard | 4,757 | 4,584 | 8,863 | 0.54x |
| `mlp_geglu.m4096.lin` | standard | 4,699 | 4,531 | 10,056 | 0.47x |
| `mlp_geglu.m32` | fast math | 156 | 153 | 169 | 0.92x |
| `mlp_geglu.m32.lin` | fast math | 163 | 160 | 171 | 0.95x |
| `mlp_geglu.m4096` | fast math | 4,746 | 4,567 | 8,910 | 0.53x |
| `mlp_geglu.m4096.lin` | fast math | 4,702 | 4,515 | 10,400 | 0.45x |

The structured results and rendered tables are preserved under
[`kernels_rtx5090/reproduction_2026-08-11`](../kernels_rtx5090/reproduction_2026-08-11). The M=32 cells are close to
eager, but the M=4096 fused MLP remains 1.9--2.2x slower. Placement-cut probes did not produce a legal fully lowered
split for the two-branch graph, so no speculative golden was promoted. The serving target above is met; this long-
prompt kernel performance gap remains open and is recorded rather than hidden behind an unverified route.

## Decision

The reproduced stock cell is within a narrow same-host performance band of the published target. It is valid only for
the adjusted workstation capacity envelope described above. The two interrupted unchanged runs are boot failures,
not benchmark regressions, and no missing article cell is inferred from the one completed performance point. The
targeted current-code kernel replay is a qualification result, not a claim that the entire published 18-cell grid was
reproduced.
