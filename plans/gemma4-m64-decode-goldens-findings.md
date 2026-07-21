# gemma-4-12B M=64 over-bucket decode goldens + serving A/B — session findings (2026-07-21, RTX 5090)

Execution of `plans/gemma4-5090-optimization.md` WS1 (extend the m32 golden playbook to M=64) and WS4 (pack-dir
wiring + stock re-baseline), branch `feature/gemma4-m64-decode-goldens`. WS2/WS3 (computed-A pipeline, hd512
flash) remain research-class, untouched by design.

## WS1 — the M=64 decode golden set (seeded, 3x medians, spread <=0.8%)

Method: the established manual pinned `--ab` flow (NOT the tuner) — cold-bench the four captured M=64 twins
(`_tune/decode-m64/`), sweep laggards from each shape's m32 anchor, confirm 3x, seed
`rtx5090_sm120_gemma4.yaml`. Sweep env: scratch `EMMY_TUNE_DB`, empty online prior, `-O3`, `--iters 50`.

Cold-vs-seeded twin totals (µs, `emmy run --ir <twin> --bench` under the serving evidence env):

| twin | cold | seeded | vs cold |
| --- | --: | --: | --: |
| pre64 (sliding) | 96.7 | **50.1** | 1.9x |
| post64 (sliding) | 684.2 | **261.4** | 2.6x |
| pre64-global | 87.6 | **48.0** | 1.8x |
| post64-global | 717.8 | **274.2** | 2.6x |

Per c=64 decode step (40 sliding + 8 global layers + lm_head): ~37.6 ms of twin time cold → **~16.3 ms**
seeded. The seeded post64 lands BELOW the m32 post twin (261 vs 284) — the cut route pays for the doubled
rows outright.

Under `EMMY_FAST_MATH=1` the same twins drop further — 45.4 / 258.8 / 45.0 / 271.0 (fm goldens + the cut
deploy cleanly; every fm twin beats its std sibling). The fm SERVER cold boot however blows past the
`--bench` 30-min health cap (evidence resolution over the doubled fork space; the std cold boot was
already ~29 min at bucket 64) — boot the fm arm once without `--bench` to write its pack, then bench off
the pack. ⚠ A `--bench` health timeout leaves the vLLM EngineCore child alive and holding the GPU
(31.7 GiB) — kill it by pid before the next arm, or every subsequent boot dies at `init_device`.

Key structure findings:

- **The m16/m32 `w1x8/f2x2` family transfers to M=64 with two systematic shifts**: deep-K shapes move to
  `k4/g4k` (q_proj, mlp_down, gate-up split), and M-heavy kernels take a second M warp-unit (`w2x8`) —
  lm_head halves (2392 → 1215 = cuBLAS parity); the fused geglu's best tile is also w2x8.
- **M=64 CUTS the gate⊗up cone** (unlike M=32, which fuses): fused best 199.6 (w2x8/f2x2/k4 g4k d1/sync)
  loses 15% to `PLACE@cone=cut` over two golden-tiled split matmuls (~78.5 each) — 169.2 e2e = 0.99x eager.
  The m256 cut pattern arrives at decode M. Seeded as `mlp_geglu.m64.cut(.lin)` + `mlp_gate_up_split.m64(.lin)`.
- **lm_head.m64 was the biggest absolute hazard**: no entry existed, and the m32 tile does NOT transfer
  (w1x8 2391.5 vs w2x8 1215.1); cold greedy picked 2539.5. Every over-bucket decode step pays this kernel.
- **qknorm bucket-64 row counts** (k256 r1024, k512 r64/r1024) had no entries; seeded per the row-count rule
  (b128 many rows / b512 few). Cold picks were near-optimal this time, but this is the highest cold-rescue
  class in the file — seeded as misdeploy guards.
- [fm] f16acc wins or ties on kv/o_proj/o_proj_global/down/gate-up-split at M=64 (recorded beside std);
  loses on q_proj_global; k_proj_global stays std-only. kv [fm] 9.5/9.6, k_proj_global 5.2, o_proj 15.9-16.7,
  down 76.4-81.0 and lm_head 1216 land at or under cuBLAS.

## Verification

- In-model drift audit (`emmy eval golden --in-model --model google/gemma-4-12B`): **MATCH 101 / DRIFT 0 /
  GAP 0 / compile_fail 0 on BOTH cards** (4090 + 5090) with the m64 block in.
- No shadowing: the readiness-era m32 twin files bench IDENTICALLY with and without the m64 block
  (pre32 112.8 vs 112.7, pre32-global 94.9 vs 94.8, post32 274.7 vs 275.4 — stash A/B). NOTE these stale
  twin files run well above the 2026-07-20 fresh-capture numbers (pre 32/30 µs) under TODAY'S goldens
  either way — they no longer represent the serving graphs (two golden retunes landed since their
  capture); the serving-level c=32 regression check is the A/B below, not these files.

## Serving A/B (protocol: #407 invariants + EMMY_PACK_DIR)

Decode workload in-8/out-64, seed 0; c=64: 128 prompts, `--max-num-seqs 64`; c=32: 64 prompts, mns 32.

| arm | req/s | out tok/s | TTFT med (ms) | TPOT mean/med (ms) |
| --- | --: | --: | --: | --: |
| emmy c=64, bucket 64 (NEW) | **40.62** | **2599** | 128.8 | **22.26 / 22.49** |
| emmy c=64, bucket 64 + FAST_MATH | **42.03** | **2690** | 129.3 | **21.46 / 21.74** |
| emmy c=64, bucket 32 + symbolic (pre-session config) | 26.77 | 1713 | 120.6 | 35.28 / 35.53 |
| emmy c=32, bucket 32 (regression check) | 22.75 | 1456 | 95.6 | 20.7 / 20.66 |
| emmy c=32, bucket 32 + FAST_MATH | 23.82 | 1525 | 96.4 | 19.6 / 19.68 |
| stock c=32 (re-baseline) | 25.61 | 1639 | 97.2 | 18.1 / 18.14 |
| stock c=64 | 34.75 | 2224 | 544.3¹ | 20.12 / 20.14 |

¹ stock's c=64 TTFT is anomalous (544 ms median vs its own c=32 arm's 97) — the same transient class the
2026-07-20 session hit at c=32; TPOT is the clean decode signal.

**emmy now LEADS stock on c=64 decode req/s: 40.62 vs 34.75 (+17%)** (TPOT 22.49 vs 20.14 = 1.12x, the
same per-token residual class as c=32's 1.14x — the WS2 computed-A pipeline). Before this session the
c=64 emmy arm ran 26.78 req/s at TPOT 35.4 — bucket-64 goldens closed the whole over-bucket gap.
FAST_MATH adds a consistent ~3.5-5% on top wherever decode dominates (c=64 TPOT 21.74 = **1.08x stock**,
req/s 42.03 = +21% over stock; c=32 TPOT 20.66 → 19.68 = 1.085x stock), matching the per-twin fm deltas.

Stock re-baseline (WS4): the 2026-07-20 decode-arm TTFT anomaly is GONE (median 97.2 vs the anomalous 473;
TPOT 18.14 reproduces 18.2 exactly) — the anomaly was transient, not structural. Clean c=32 ratios: emmy
req/s 0.89x, TPOT 1.14x stock (decode residual unchanged, as expected — WS2 territory).

**WS1 exit gate met**: c=64 TPOT 35.4 → **22.49 ms** (−36%), req/s 26.78 → **40.62** (+52%). The
bucket-64 decode step now runs ~1.09x the c=32 per-token time instead of 1.71x.

## Long-context 4K-in/4K-out (mml 8448, mnbt 4096, c=8, 16 prompts, seed 0, bucket 32)

| arm | duration (s) | out tok/s | TTFT mean/med (ms) | TPOT mean/med (ms) |
| --- | --: | --: | --: | --: |
| stock (util 0.90) | **176.0** | **372.3** | 2416 / **2064** | **20.90 / 20.99** |
| emmy std (util 0.97) | 276.9 | 236.7 | 26126 / 3686 | 22.10 / 22.33 |
| emmy FAST_MATH | 360.1 | 182.0 | 43162 / 4836 | 21.78 / 21.89 |

The 2026-07-20 "emmy wins 4K/4K" result REVERSES against today's stock arm — stock's own long-workload
baseline jumped 204.5 → 372.3 tok/s between sessions (its TTFT profile collapsed from mean 38.2 s to
2.4 s — a vLLM/scheduler-profile shift, not an emmy regression; emmy's own number held 235.5 → 236.7).
The decode side is fine (TPOT 22.33 = 1.06x stock); the whole loss is prefill-side: emmy TTFT mean 26.1 s
vs 2.4 s — with c=8 and 4K chunked prefills, decode steps stall behind emmy's slower large-M prefill
(the documented WS3/computed-A pipeline residual, now the clear top priority for the long workload).

FAST_MATH on the long workload is a NET LOSS despite the better decode (TPOT 21.89 < 22.33): out tok/s
236.7 → 182.0, TTFT mean 26.1 → 43.2 s — the fm lane's large-M prefill picks run SLOWER than std (the fm
dynM evidence is anchored at M=512; at 4K-chunk M it extrapolates worse than the std lane). Run the long
workload on the std lane; fm pays only where decode dominates.

## WS4 — workflow

- `EMMY_PACK_DIR` wired into the serving A/B protocol (`_tune/decode-m64/serve-ab.sh`): per-lane pack dirs
  (`packs-std` / `packs-fm`) because the gen-runner pack key does NOT carry the fast-math lane — an fm boot
  loading a std pack would silently deploy std kernels. Measured: the c=32 arm booted off the bucket-32
  pack in **~2 min vs ~23 min cold** (>10x; the b64 cold boot with its larger twin set + capture ladder was
  ~29 min). One pack per (decode_bucket, max_len) config, ~3.2 MB each.
- Stock decode re-baseline: done (see the c=32 row — anomaly transient, TPOT reproduces exactly).
- Code changes riding along: `(2, 8)` added to `_WARP_UNITS` (the lm_head.m64 winner geometry must be
  enumerable — `test_golden_knobs_are_members_of_the_move_catalog` gates it); stale over-bucket capture
  paragraph in `serving/ARCHITECTURE.md` brought up to the #408 ladder behavior.
- Remote 4090 `make test` note: `test_bench_dry_run_tinyllama_block` fails on the rental box on an
  untouched code path (the dry-run's rendered command never reaches stdout there) — pre-existing
  environment behavior, passes locally.

## Remaining (unchanged, research-class — the successor plan's WS2/WS3)

- Decode TPOT residual (1.08-1.14x stock): the fused/split gate⊗up + down_proj computed-A pipeline at
  decode M — async multi-stage weight prefetch, own session (`plans/computed-a-pipeline-and-sdpa-oproj.md`).
- Long-context/prefill: emmy TTFT mean 26 s vs stock 2.4 s at 4K chunks — the large-M computed-A pipeline
  (same lever) + hd512 flash cold-unreachability. The 4K/4K throughput loss is entirely this.
- M=128 goldens: only if a real workload needs c>64 — same recipe as this session (expect k4/g4k + another
  M warp-unit shift; `(4, 8)` may need enumerating).
- fm large-M prefill picks (the fm long-workload regression): the fm dynM evidence anchors at M=512 —
  either seed fm prefill-chunk evidence at 4K-chunk M or gate fm off the prefill path.
