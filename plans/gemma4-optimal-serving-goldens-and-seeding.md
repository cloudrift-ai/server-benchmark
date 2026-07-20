# Plan: extend the gemma-4 golden set, seed the deploy evidence, and make serving deploy the optimal kernels

## RESULT (2026-07-20) — goldens/seeding CLOSED; e2e still behind stock, and the reason is now measured

Eleven commits on `feature/serve-boot-resolution-cache`, `make test` 2535 green throughout.

### Twin totals (5090, cold, repo goldens as the ONLY evidence)

| twin | session start | now | |
| --- | --: | --: | --: |
| pre32 | 91.0 | **34.2** | 2.7x |
| post32 | 18,202.2 | **268.0** | 68x |
| pre256 | 474.0 | **125.7** | 3.8x |
| post256 | 144,202.2 | **553.6** | 260x |
| pre32-global / post32-global | 92.3 / 325.7 | **34.3 / 272.5** | |
| pre256-global / post256-global | 463.3 / 710.8 | **155.5 / 589.3** | |
| pre-sym / post-sym | 369.9 / bench_fail | **204.6 / 1,750.1** | |

Golden audit across all 12 twins: **DRIFT 0**. Five entries the cast-sink obsoleted were pruned, including one
actively harmful (`mlp_geglu.m256` reported 226.5 us while deploying a 563.7 us kernel).

### E2e serving A/B — gemma-4-12B, both arms WARM (the only valid comparison)

| in-8 / out-64 | req/s | TTFT mean / med (ms) | TPOT mean / med (ms) |
| --- | --: | --: | --: |
| stock vLLM | **25.69** | 101.5 / 93.8 | **18.13 / 18.17** |
| emmy (goldens only) | 22.94 | **84.9 / 69.3** | 20.76 / 20.80 |

| in-256 / out-64 | req/s | TTFT mean / med (ms) | TPOT mean / med (ms) |
| --- | --: | --: | --: |
| stock vLLM | **14.28** | **459 / 218** | **25.85 / 27.81** |
| emmy (goldens only) | 8.33 | 1198 / 790 | 29.48 / 29.39 |

**emmy wins TTFT on the decode workload (69.3 vs 93.8 ms median, 1.35x) and loses req/s (0.89x) and TPOT (1.14x).
The mixed workload stays clearly behind.** Versus the previous report's emmy arm at identical config: req/s
20.86 -> 22.94, TPOT 22.7 -> 20.80, mixed req/s 7.11 -> 8.33. Real, but far short of what the twin numbers suggest.

**A/B METHOD BUG THAT NEARLY SHIPPED A FALSE WIN — the first arm of a script is not a baseline.** The initial
`stock-in8` measured 19.13 req/s / 478 ms TTFT, which read as emmy winning outright. Its TPOT (18.06) reproduced the
earlier session exactly while its TTFT was 5x worse; that split means warmup landed inside the measured window. The
re-run gave 25.69 / 93.8 / 18.17, reproducing the earlier session on all three. Both arms improve on a second run
(emmy 21.80 -> 22.94), so **every arm must be run warm, and a baseline that moves on one metric but not another is
not a baseline.**

### Why the 68x / 260x twin gains did NOT translate — the decode floor, measured

Decode weight traffic is 448 MB/layer (geglu 236 + down 118 + qkv 63 + o 31) = **21.5 GB/token**, which at the
5090's 1.79 TB/s is a **12.02 ms hard floor**. emmy's 14.5 ms of kernel time is 83% of that ceiling. Stock's
18.17 ms TPOT implies ~6.2 ms of non-kernel overhead; emmy's 20.80 implies ~6.3 ms.

So the decode gap is no longer kernel SELECTION: it is ~2.5 ms of kernel time against a bandwidth floor plus
comparable per-step overhead. **A perfect decode kernel set lands at ~12.0 + 6.3 = 18.3 ms, which only just reaches
stock's 18.17.** Beating stock on decode TPOT is therefore a runner/overhead problem and a bandwidth-efficiency
problem, not a golden problem — which is what this plan's "explicitly out of scope" section always said, now with
the number behind it.

Prefill is where the mixed-workload gap lives: 679 us/layer x 48 = 32.6 ms per 256-token chunk, and 32 concurrent
256-token prompts queue ~64 chunks deep, which is exactly the 790 ms median TTFT observed.

### The change that mattered, and it was not a golden

`optimization/007_sink_narrowing_cast`. Loop fusion was splicing the f16 cast into its CONSUMERS, so the RMSNorm
wrote `float* mul_3` and gate/up read f32. A mixed-dtype A has no copy transport, so `_demote_mixed_a` diverted them
onto the `sync` compute-fill with no weight-prefetch ring: 1.12 TB/s against the 1.61 TB/s the neighbouring
down_proj reached on an identical 118 MB weight. Sinking the cast is a pure retype (bit-identical; verified in the
emitted kernel that the statistic still squares in f32 via `__half2float`) and it dissolved the norm->qkv computed-A
cones entirely -- q/k/v became plain f16 matmuls matching goldens recorded long before this session.

### Compiler defects found and fixed (each was silent)

1. `bind_contraction` bound A as "the first (m, k)-indexed Load". With a fused cone that is a cone-INTERNAL load, so
   it emitted `gate @ W` with the gelu and up projection GONE -- a wrong kernel, not a slow one.
2. Refusing to bind a stat-free cone demoted the cell to a PLANAR scalar fold: 274 -> 18,202 us on post32.
3. A cut row carrying split-K was realized into a partial/finalize pair before `020_cut_edge` ran, so the cut
   silently never fired at M=256.
4. The stat-free cut did not terminate (pointwise producer re-fused, node re-cut) -- a hard failure under the pin.
5. `PLACE@cone` stamped on the TileOp instead of the fork ROWS, so no golden could ever select the cut.
6. `make test` (2535) caught NONE of these. No test asserts an edge stays on the mma tier.

### Remaining, in order

1. **Per-step runner overhead** (~6.3 ms of a 20.80 ms TPOT) -- the only remaining decode lever, and it is a serving
   runtime question, not a compiler one.
2. **Prefill throughput** -- 32.6 ms/chunk is ~81% of dense fp16 peak, so the win must come from a better prefill
   FORM (packed varlen, fewer chunks), not better configs.
3. `post-sym` at 1,750 us -- only reachable when a step exceeds the prefill bucket; unused in this A/B config.


## Superseded

Everything below the RESULT block in earlier revisions of this plan described the intermediate states this session
passed through (a 10-DRIFT audit, a "multi-fold cone" diagnosis that was inferred and wrong, two reverted fixes).
All of it is either landed in the commits above or contradicted by the final measurements, so it has been dropped
rather than left to mislead. The commit messages carry the per-change reasoning; `git log
feature/serve-boot-resolution-cache` is the record.
