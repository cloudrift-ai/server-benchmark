# TTFT sweep + TPOT parity campaign (planned 2026-07-26)

Mandate (user, autonomous mode): **beat stock vLLM on TTFT in EVERY row and reach TPOT parity**, on the
equal-tuning protocol (util 0.96 all lanes, per-cell decode buckets, np=64 wave for the c=64 TTFT cell).
Premise worth honoring: emmy's kernels are faster in every mode, so a TTFT loss is a routing/step-shape
problem to fix, not physics.

## Scoreboard to close (fm vs stock @0.96, 2026-07-25/26 final pass)

| row | fm TTFT | stock | gap | fm TPOT | stock | gap |
| --- | --: | --: | --: | --: | --: | --: |
| small_c1 256/256 | 64.6 | 56.3 | −8.3 | 17.04 | 16.28 | +0.76 |
| head_c1 4K/4K | **474.6** | 565.9 | WIN | 18.10 | 17.35 | +0.75 |
| head_c4 4K/4K | 1363.5 | 1087.8 | −275.7 | 18.85 | 18.23 | +0.62 |
| head_c8 4K/4K | 1828.1 | 1100.1 | −728.0 | 20.56 | 20.56 | PAR |
| rag_c4 8192/256 | 2474.5 | 2428.7 | −45.8 | **24.54** | 26.24 | WIN |
| small_c64 wave | **1323.9** | 1467.7 | WIN | 28.53 | 28.02 | +0.51 |

## WS-A — STATUS: CLOSED (2026-07-26; collision-proof classifier + stock trace)

Instrument: `_tune/nsys/step_attribution.py` — exact `(kernel, grid, block)` match against the pack
plans' launch lists (symbolic grids fall back by name), steps segmented by the once-per-step lm_head
cutlass kernel, per-step T read off vLLM's rotary-kernel gridX. Traces: `c4_b8fm` (fm bucket-8,
np8/out256) vs a fresh `stock_c4` (same bench shape, util 0.96).

FINDINGS — the WS6 "two-pass mixed step / decode-dominated TTFT window" verdict is REFUTED (a
FIRST-WINS classifier collision artifact):

1. The fm c=4 TTFT window is ~97% PURE m4096-chunk-twin busy time; ZERO sym steps anywhere in the
   trace. Per-step T histogram: 8 × T=4096 (one per prompt), 506 × T=4 (steady decode), crumbs at
   T=7/3/2/1 (prompt tails).
2. The real structure: the emmy lane pinned `--max-num-batched-tokens` at 4096 = exactly the chunk
   width, so a chunk step had ZERO headroom — no decode riders (decode froze ~440 ms per queued
   chunk), and the +1 BOS token (prompts are 4097) pushed every prompt's last token into a deferred
   tail step, so first-token sampling waited for the whole serialized wave (first delimiter at
   1313 ms after THREE full chunk passes).
3. Per-kernel, emmy prefill is FASTER than stock: fm m4096 pass ≈ 341 ms busy / 435 ms wall ≈
   83 µs/token trunk-busy vs stock's ~101-124 µs/token prefill GEMMs (`128x64_64x3`/`64x256_32x4`).
   Stock's decode-step GEMM busy ≈ 13.6 ms vs emmy decode busy 15.0 (WS-D lead). Under identical
   traced conditions (np8/out256) stock median TTFT read 1385 vs fm ~1363 — the scoreboard gap is
   step-shape dynamics at deeper queues (np32/out4096), not kernel speed.
4. Stock rides mnbt=2496 (vLLM auto-raised from 2048 for the multimodal prefix-LM) — small chunks
   that always carry decode riders and sample nearly every step.
5. KV capacity at util 0.96: stock 38,292 tokens vs emmy 24,021 (emmy's cupy residents shrink the
   vLLM budget) — a ~60% admission advantage for stock at c=8 (4.5 vs 2.8 concurrent 8.4k
   contexts). Main c=8 compounding factor beyond the step shape.

## WS-A — honest mixed-step attribution (first; the WS6 conclusion is suspect)

The WS6 "two-pass mixed step" verdict came from `per_kernel_classified.py`, whose name→class map is
FIRST-WINS and known to collide (the sym cut consumers share kernel-name stems with static twins). The
runner's actual routing (gen_runner): decode T ≤ bucket → static decode twin; a chunk step that fills to
`--max-num-batched-tokens` exactly (queue deep ⇒ chunk + decode riders = 4096) → the static m4096 chunk
twin; ragged T (drain phases, shallow queue) → the symbolic program at hint 512. Redo the c4/c8 TTFT-window
attribution with a collision-proof classifier (map kernels via each pack plan's launch list per program,
disambiguate shared stems by grid signature), plus a step-shape histogram (T per step from scheduler
counters). Output: how many TTFT-window steps ride static-4096 vs sym, and each class's µs vs stock's
equivalent fused-varlen step.

## WS-B — STATUS (2026-07-26, in flight)

**Round 1 — rider split LANDED (code): chunk+decode twin row split.** `EmmyGenRunner.rider_width` +
split branches in `forward_layer_{pre,post}_device` (T in `(prefill_bucket, prefill_bucket +
decode_bucket]` splits row-wise: chunk twin on the first `prefill_bucket` rows, decode twin on the
riders — per-token-independent, request-agnostic), mnbt bound widened to `DYNAMIC_DIM_MAX + bucket`
(`vllm_model_gen` + `emmy serve` default; boot refuses over-cap mnbt when the twin families are
missing). Unit-tested (`test_gen_prefill_device_gpu` T=40 split regime + serve-cmd bounds).
A/B (fm, bucket 8, mnbt 4104, pack-fm-b8, util 0.96): /metrics proves every 4097-token prompt now
rides ONE (4096,8192]-bucket step (sampling immediate, riders carried). **head_c8 TTFT 1828 →
1424** (TPOT 20.70 vs 20.56, tok/s 380.0 vs 381.0 — held). head_c4 TTFT unchanged (1388 vs 1363,
TPOT 18.89, tok/s 208.2 — held); probe TPOT at np8/out256 elevated (20.4) — mixed-step decode
stall ~1.5-2 s per prefill event observed in scheduler windows, unexplained residual (split step
should cost ~500 ms; trace pending if it matters after round 2).

**Round 2 — chunk-QUANTUM theory (probe CONFIRMS structure; sym tier too slow).** Stock's flat c4
TTFT ≈ 1084 (mean ≈ median) is its mnbt=2496 chunk quantum: queued prompts pipeline in ~270 ms
quanta, spreading each wave; emmy's 4096 quantum serializes waves in ~500 ms steps. A/B
`--max-num-batched-tokens 2056` (sym prefill at T<=2056; capacity decoupled from mnbt so the pack
still hits — `vllm_model_gen` now pins runner capacity/prefill-bucket at DYNAMIC_DIM_MAX instead of
mnbt, or pack keys would recompile per scheduler knob): probe np8/out256 TTFT 1365 → **1146**
median (structure works) but TPOT 18.9-class → **22.67** — the sym pass at T≈2050 runs ~2.5x the
static twin's per-token cost (hint-512 geometry; the WS4 "same schedule family" claim does not hold
in-graph at T=2048). TWO structural bonuses discovered: (a) at mnbt<=~2056 the vLLM profiling dummy
peak shrinks and **emmy KV = 38,069 tokens ≈ stock parity** (was 24,021 at mnbt 4096/4104 — the c8
admission gap closes for free); (b) with mnbt = prefill_bucket + bucket, every FILLED step is
static (chunk twin + rider split), sub-bucket tails ride the decode twin, sym only at rare drain
steps.

**Round 3 — the m2048 STATIC chunk tier: goldens SEEDED (18 rows), serving A/B in flight.**
Sweeps `_tune/m2048/sweep{1,2}.sh` (30-iter medians, weights variable-bound). Winners (fm | std |
eager µs): qkv_cat 532.7 serial-w4x2/f4x8/k4 | 696.5 | 664; qk_global_cat 541.2 | 702.1 | 712;
gate_up_cat 1549.7 **gm8** | 3644.6 | 2302; gate_up_split 830.3 gm8 | 1252.7 | 1201; o_proj 222.1
**w4x2/f4x8/k4+g2k** (the greedy find, beats serial 283 and w2x2-g2k 265) | 336.2 | 305;
o_proj_global 419.5 g2k | 699.6 | 606; mlp_down 799.2 g2k | 1275.5 g4k | 1161; rms k3840 b128
13.1. Fused keys: CUT wins everywhere (qkv 578 vs fused 1432, qkg 547 vs 1655, gate_up 1766 vs
7302, down_fused 974 vs 3176) — recorded as `PLACE@cone: cut` rows. Note: in-cone consumers
resolve slightly off the .lin rows (qkv → g2k 568, gate_up → serial 1755, +206 µs ≈ 0.13% of a
pass — left alone). fm chunk-pass matmul budget ≈ 151 ms → ~190 ms quanta (< stock's 270). A/B:
fresh pack `_tune/m2048/pack-fm-b8-p2048`, EMMY_GEN_PREFILL_BUCKET=2048 + mnbt 2056, c4/c8.

**Round 3 VERDICT (2026-07-26 14:38): BOTH TTFT ROCKS BEAT STOCK.** fm p2048, article shapes:
head_c4 TTFT **1063** (was 1363; stock 1088) TPOT 18.99 (was 18.85) tok/s 207.96 (held); head_c8
TTFT **1014** (was 1828; stock 1100) TPOT 20.81 (was/stock 20.56) tok/s 379.96 (gate 381,
within noise); probe np8/out256 TTFT 1043. Twin widths list gained 2048 (`twins.py`) so the audit
covers the new rows. Residual: TPOT +0.14/+0.25 vs the pre-quantum fm numbers (two ~200 ms chunk
stalls per prompt instead of one ~500 ms), P99 c8 TTFT 3.9 s (admission tail). Ops incident worth
keeping: a fresh p2048 pack build takes ~35-60 min (48 layers x ~8 programs, resolve-bound) — the
first A/B attempt's 30-min health window expired, benches ran against the compiling server and the
teardown killed it mid-compile (pack unsaved); the harness now waits 100 min and HARD-FAILS if
health never arrives. rag_c4 on p2048 (same pack): TTFT **1632** (was 2474; stock 2429 — a −800 win) but TPOT 28.73
(was **24.54**, stock 26.24 — the chunk-host-overhead tax x4 chunks per prompt kills the TPOT
win). Deciding bench: rag on the RIDER config (m4096 + mnbt 4104, one stall per prompt,
sampling immediate) — expect TTFT ~stock with TPOT ~24.5 held. small_c1/head_c1/c64 configs
unchanged so far (they get the default-mnbt rider headroom only). Known residual mechanism (short-output
shapes): each eager chunk step carries ~90 ms of host overhead (48 layers x pre+post
run_device: gpu_lock + dlpack negotiation + output clones ≈ 0.9 ms/layer — measured as the
m4096 chunk's 435 wall vs 341 busy) — two 2048-chunks per prompt pay it twice, which is why
probe-shape (out=256) TPOT reads ~22 while the article shape dilutes to ~19.0. The structural
fix would be capturing the static chunk step as its own whole-step CUDA graph (the twin widths
are exact) — future work, not this campaign.

## WS-B — the c4/c8 TTFT gap (the big rocks, −276/−728 ms)

Hypotheses in test order:
1. **Ragged-sym steps at large T are slow** (hint-512 geometry at T≈4096; WS4's closure only checked the
   SCHEDULE family, not in-graph µs). If confirmed: per-hint symbolic tiers (a second sym program at hint
   4096 routed for T > 2048) or pad-to-static routing (pad T ∈ (4096−64, 4096] up to the m4096 twin —
   `_pad_rows` already exists for decode; prefill-side analog).
2. **Admission/step-shape**: with 4+ requests queued, emmy's slower decode-mixed steps delay later chunks
   (WS2 shrank this; quantify what remains). Scheduler knobs legal under the article's protocol
   (`--long-prefill-token-threshold`, `--max-num-partial-prefills`) count as per-lane tuning like the
   bucket knob — A/B them.
3. c8 specifically: stock TTFT ~equals its c4 (1100 vs 1088 — admission-limited, not work-limited); emmy's
   1828 grows with concurrency ⇒ emmy's mixed-step cost compounds per queued prompt. Fixing (1) should
   collapse both; verify c8 explicitly.

## WS-C — small_c1 TTFT (−8 ms)

256-token prefill: T=256 rides which program? (`prefill_bucket` default 0 ⇒ symbolic at hint 512, near its
tuned geometry — but stock does 56 ms wall vs emmy 65.) One nsys trace of the 256-prefill step; likely
levers: the m256 golden set (exists) not being reached (sym hint routing), or fixed per-boot overhead
(first-step warmup leaking into TTFT — check bench warmup discipline).

## WS-D — TPOT parity (+0.5..0.8 ms/step at c1/c4/c64)

The launch-count theory is refuted (3×); the whole decode step replays as one CUDA graph, so the residual
is inside the replay: kernel-time sum + inter-kernel gaps + the sampler stall. Diff emmy's captured-step
timeline against stock's at c1 (nsys, graph-node granularity): bucket the gap into (a) kernel µs we can
golden-tune, (b) gaps/serialization (chain structure — the m1 split-chain), (c) shared vLLM overhead
(sampler, host sync) that stock pays too. Only (a)+(b) are ours; if the honest emmy-attributable slice is
< the 0.7 ms gap, the remainder is measurement framing (e.g. stock's leaner sampler path) — then attack
THAT explicitly rather than declaring parity impossible.

## Method / verification

Per change: pinned `--ab` or code change + unit tests → targeted serving A/B on the affected cell (fresh
pack, empty online, util 0.96, the cell's bucket) → audits green (`eval golden --in-model` DRIFT 0,
ratchet) → `make test`. Full six-cell + wave re-bench only when the scoreboard claims all gates. Article
updates only after the campaign settles.

## Exit gates

- TTFT: fm ≤ stock on ALL six rows (small_c1, head_c1, head_c4, head_c8, rag_c4, c64 wave).
- TPOT: fm within ±0.15 ms of stock on every row (parity), no row regressing.
- No throughput regression: c64 np256 tok/s ≥ 1223, c8 ≥ 381.
