# gemma-4-12B serving verification after the review-fix batch (RTX 5090, 2026-07-20)

The final verification step of the 2026-07-20 review-and-fix session (the ~16-commit batch merged to main:
split-K f32 workspace, cone-key offer signal + fused-golden schema anchor, stale-evidence cut guard, prefill-twin
exact-width routing, accuracy-gate tightening, fusion cut-brake, release-pipeline hardening). Run on the local
5090 box (`kenshin`), bare venv serving (vLLM 0.23.0, torch 2.13.0+cu130, CUDA 13), seeded 5090 goldens +
warm tune caches in place.

## GPU test gate

`tests/serving` + `tests/compiler/e2e/{test_fused_edge,test_attention_coverage}` + source determinism at
`-Xcicc -O1`: **211 passed / 1 failed** on the first run; the failure (banded+bias flash no longer fusing) was
root-caused to the session's softmax add-discount scoping and **reverted** (7fa90024) — the piecewise softmax
assembly applies masks in the shifted-store loop, which no rowmax/exp def-chain reaches, so precise scoping is
structurally impossible and the blanket all-adds discount stays (its over-merge imprecision on hypothetical
add-heavy rowmax producers is accepted). After the revert: **attention coverage 122/122 green**.

## Serving A/B (matched bench: 64 prompts, in 256 / out 128, concurrency 32, seed 0; mml 4096, mnbt 4096)

emmy: decode bucket 16 (whole-step FULL_DECODE_ONLY graphs), `EMMY_GEN_PREFILL_BUCKET=0` (see the blocker below),
util 0.97. stock: util 0.90 (0.97 OOMs stock's own sampler warmup — that headroom setting exists for emmy's
device residents; stock throughput is util-insensitive once the KV cache fits). All 64 requests succeeded in
both runs.

| metric                          | stock vLLM | emmy   | emmy 2026-07-16 (pre-fix, mnbt 512, bucket 0) |
|---------------------------------|-----------:|-------:|----------------------------------------------:|
| Request throughput (req/s)      |   **7.52** |   0.10 |                                          0.06 |
| Output token throughput (tok/s) |    **963** |   13.4 |                                           7.4 |
| Mean TTFT (ms)                  |  **1 024** | 30 005 |                                       107 248 |
| Mean TPOT (ms)                  |   **21.3** |  1 694 |                                         2 928 |
| Bench duration (s)              |        8.5 |    612 |                                         1 113 |

- Directionally better than the last recorded emmy run — ~1.8× output throughput, TPOT −42%, TTFT −72% — with
  the config differences noted in the header (this run: mnbt 4096 + decode graphs + device-resident paths + the
  exact-width prefill routing; the pre-fix row ran mnbt 512, bucket 0).
- The gap to stock (~72× output throughput) remains the known integration wall, not a kernel-quality story: the
  per-step host orchestration around the compiled kernels dominates. The fix batch was correctness/robustness,
  not a throughput lever, and the numbers confirm nothing regressed while the routing fix removed the worst
  mid-width over-compute.

## Blocker found: M=4096 prefill-chunk twin dies with cudaErrorIllegalAddress (pre-existing)

With the prefill twin ON at its default width (mnbt = 4096), the server dies in vLLM's profiling pass — the
twin's first captured-graph launch hits an illegal memory access (both pre and post twins, layer 0). Reproduced
standalone with a minimal runner driver at current main content AND at the pre-session branch tip `a18a8b5c`,
so it predates the review-fix batch; the branch's own serving benches all ran mnbt 512 and never exercised this
width on this box. Notable: the release-image validation passed at this exact config inside the docker
toolchain, so the fault may be toolchain/env-sensitive (bare venv: torch 2.13.0+cu130, CUDA 13.0). Workaround:
`EMMY_GEN_PREFILL_BUCKET=0`. Needs a dedicated debug session: dump the M=4096 twin program (`EMMY_DUMP_DIR`),
bound-check its kernels, compute-sanitizer the first replay.

## Workflow notes

- The session's vector-store temp-digest widening re-keys the source of nearly every vectorized-store kernel:
  one full cubin-cache rebuild (~22 min on the 5090) on the first post-merge boot. One-time; expected.
- Only nvcc is cached across boots — the CPU-side pipeline (trace → passes → resolve → codegen, 48 layers × up
  to 4 programs) runs every boot (~20 min for the 12B). The boot-resolution-cache work trims the resolve slice;
  the rest is the standing boot-cost problem.
- `emmy serve --bench` waiters: poll the bench log, not `pgrep` (a `bash -c` waiter matches its own pattern).
