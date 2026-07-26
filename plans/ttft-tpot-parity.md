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
