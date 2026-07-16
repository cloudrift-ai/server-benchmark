# Gemma-4-12B post-gate-fix reproduction — golden audit, layer-0 re-tune, e2e serving A/B (RTX 5090)

- **Status**: complete. The serving A/B was blocked mid-session by desktop VRAM (~6.8 GiB held by the GUI
  session); after the user freed it, all three benches ran to completion — results below.
- **Context**: follow-up to `plans/gemma4-12b-layer0-fastmath-ab-tune-findings.md` (pre-fix run) after the three
  in-model gate fixes landed (PR #374: split traced casts, rank-N matmul TMA boxes, s2048 split-K cap). This run
  answers: do the goldens still reproduce and deploy, did the layer-0 in-model picture improve, and what does
  end-to-end *serving* look like vs stock vLLM — fast-math and standard.
- **Run commands** (2026-07-15, local RTX 5090, driver 580.159.03; per-run `EMMY_TUNE_DB` / `EMMY_ONLINE_FILE`
  under `_tune/tune-model-gemma4-12b-repro-e2e/`, never the user cache):

  ```bash
  # golden reproduction sweep (34 names × 2 regimes, pinned re-bench + greedy join)
  ./sweep.sh golden-std && ./sweep.sh golden-fm fm       # emmy run --golden NAME --bench --json per name

  # layer-0 tunes (whole-model tune BLOCKED — see finding 4)
  emmy tune google/gemma-4-12B --layer 0 --dynamic seq_len@x:1 --clean --bench --dump-dir <dir>/dump-l0      # std
  EMMY_FAST_MATH=1 emmy tune google/gemma-4-12B --layer 0 --dynamic seq_len@x:1 --clean --bench -v \
      --dump-dir <dir>/dump-l0-fm                                                                            # fm

  # serving A/B (vLLM 0.23.0, venv-serving; emmy runs: EMMY_GEN_DECODE_BUCKET=0)
  emmy serve google/gemma-4-12B --generate --bench [--stock] --max-model-len 512 --max-num-batched-tokens 512 \
      --num-prompts 64 --random-input-len 256 --random-output-len 128 --max-concurrency 32 --bench-seed 0
  ```

- **Run stats**: golden sweeps ≈ 6 min/regime (warm cubin cache). Layer-0 std: DB 559 ok / 20 `bench_fail`
  (run interrupted once by a host OOM kill and resumed; completing portion searched 2165.1 s). Layer-0 fm:
  3583.8 s search, 501 ok / 11 `bench_fail`.
- **Regime disclaimer**: all table numbers below are the `--bench` -O3 re-bench (deployable, CUDA-graph
  captured); tune-DB latencies quoted for ranking context are `-Xcicc -O1` and never comparable to them.
- **Dynamic run**: symbolic `seq_len`, benched at the 512 hint (`benched at seq_len=512 (symbolic hint; torch
  inputs tiled to match)`). Layer 0 is a `sliding_attention` layer (hd256, 16 Q / 8 KV heads).

## Golden reproduction — PASS both regimes; greedy join follows the goldens on 33/34 names

Every recorded entry of `goldens/rtx5090_sm120_gemma4.yaml` (68 entries / 34 names: std + fm siblings, static +
dynM + s2048) re-benched pinned via `run --golden NAME --bench --json`, in both regimes:

- **Pinned reproduction: 68/68 rows `ok` per regime (136 total), all within ~2% of the recorded `emmy_us`**
  (`rows failing or >5% slower than recorded: 0` in both summaries). The goldens are healthy as measurements.
- **Greedy-deploy join**: 33/34 names follow the golden in each regime. Under `EMMY_FAST_MATH=1` every matmul
  name deploys its fm entry (the umbrella works); `attention.hd256.dynM` deploys the std entry at 32.0 µs —
  correct, it is the fastest recorded config for that shape.
- **The previous session's golden re-records now deploy**: `kv_proj.s2048` fm (greedy 110.1 vs rec 109.6),
  `k_proj_global.s2048` fm (36.3 vs 36.4) and std (45.6 vs 45.4) — the three update candidates from the pre-fix
  report realize unpinned.
- **The one deviation, both regimes: `attention.hd512` (static)** — greedy 158/157.6 µs vs golden 113.0 µs.
  Known enumeration limitation (corrected verdict in the pre-fix report): flash stage candidates resolve on the
  PRE-split geometry where the hd512 slabs don't fit, so the (split-KV + staged) combination is never offered;
  the pinned recording itself reproduces at 113.1 µs (1.00×). Not a recording artifact; enumeration follow-up.

## Layer-0 bench — per-kernel is the signal; two fused-form losers carry the whole gap

Full layer (eager / torch.compile / emmy, -O3, seq 512 symbolic hint):

| regime | Eager | torch.compile | Emmy | Emmy vs eager |
| --- | --: | --: | --: | --: |
| std | 1395 | 1234 | 2171 | **0.64×** |
| FAST_MATH | 1405 | 1247 | 2807 | **0.50×** |

⚠ The fm e2e row is NOT consistent with its own per-kernel table (sum ≈ 2082 µs < 2807 µs) — the whole-program
capture anomaly the pre-fix report flagged, reproduced; see finding 3. Per-kernel reproducer rows below are the
stable signal.

Per-kernel (`--bench` tables, emmy-std-descending; layer-op labels from the dump's `.torch.txt` provenance):

| kernel | layer op | eager | tcompile | emmy std | emmy fm |
| --- | --- | --: | --: | --: | --: |
| k_linear_mean_reduce_241f7a | gate⊗up matmuls + GELU + pre-FF-norm mean (fused, computed-A) | — | — | 783 | 768 |
| k_mean_linear_reduce_cd923e | post-FF norm + layer_scalar + down-proj finalize (fused) | 486 | 306 | 303 | 225 |
| k_linear_reduce_dbf34f | down-proj main matmul | 294 | 295 | 292 | 216 |
| k_linear_sdpa_reduce_fb39d3 | SDPA tail + o-proj (fused, computed-A) | 129 | 127 | 189 | 199 |
| k_mean_linear_reduce_f2c987 | post-attn norm + o-proj prologue slice | — | — | 124 | 132 |
| k_linear_reduce_5dd170 | q-proj main matmul | — | — | 119 | 126 |
| k_mean_linear_reduce_586289 | input-norm + v-proj prologue slice | 250 | 106 | 89 | 72 |
| k_mean_linear_reduce_46ea14/9aa814 | q/k-norm + proj prologue slices | — | — | 73–74 | 73–75 |
| k_linear_pointwise_57365e | q-norm apply (pointwise) | — | — | 73 | 73 |
| k_linear_reduce_a17f46 | v-proj main matmul | — | — | 71 | 71 |
| k_scaled_dot_product_attention_reduce | flash SDPA (hd256 causal) | 31 | 31 | **32** | **32** |
| k_mean + rope slice/cat pointwise (×5) | means, rotary sin/cos prep | 39–145 | 2–12 | 2–7 | 2–7 |

**The PR #374 gate fixes held**: in-model flash SDPA 208 → 32 µs (std; golden parity), down-proj main matmul
345 → 292 µs std / 308 → **216 µs fm** (= its 214 µs golden — the rank-N TMA box fix delivering the fm big-tile
family in-model). Dominators now: the gate⊗up megakernel (783/768) + SDPA→o-proj (189/199) ≈ 45% of the emmy
total and effectively the entire gap to eager.

## Finding 1 — computed-A fused contractions still lock out async staging + split-K: gate⊗up megakernel 2.1× off its golden form (fm)

- **Symptom**: `k_linear_mean_reduce_241f7a` (the fused RMSNorm→gate⊗up→GELU megakernel) deploys 783 µs std /
  768 µs fm; the standalone `mlp_gate_up.dynM` golden runs 562 µs std / **362 µs fm** (+ ~7 µs for the absorbed
  mean). The fm gap is the larger one because the fm big-tile family (`w4x2/f4x8/k4 d2/tma/ring`) is exactly the
  staged configuration the fused form cannot reach.
- **Evidence**: `eval variants --kernel linear_mean` — **every** measured config (39 std / 113 fm) is
  `PLACE@cone=fuse` + `STAGE=d1/sync`; no staged transport, no split-K, in either regime. The fm leaderboard does
  enumerate the f16acc atoms (the umbrella works) but only on the sync compute-fill.
- **Root cause** (class 2, known/by-design gate — unchanged from pre-fix finding 2): a computed-A (fused-cone)
  contraction offers no split-K — "a producer-cone A cannot be sliced over K (its statistic prologue spans the
  row)" (`_reduce_candidates`, `lowering/tile/_schedule.py`, the `probe.a_computed` guard) — and the stage
  resolver extends A's forced sync compute-fill to the whole stage decision, so the B (weight) slab loses its
  async transport too.
- **Fix suggestion (P1, ~400 µs/layer fm at stake on this kernel alone)**: stage the RAW B slab async while A
  rides the compute-fill — B is a plain gmem operand even when A is computed. Alternative mitigation: record
  in-model-form goldens for the fused shapes so the join stops deploying cold.
- **Repro**: `emmy run --ir <dump-l0>/08_lowering_cuda.kernels/k_linear_mean_reduce_241f7a.torch.json --bench
  --ab "TILE=a:mma_m16n8k16_f16_f16/w4x2/f4x8/k4,STAGE=d2/tma/ring"` (expect the stage pin to degrade to sync).

## Finding 2 — SDPA→o-proj fused kernel REGRESSED 129 → 189/199 µs: the golden join now matches its shape but cannot realize it

- **Symptom**: `k_linear_sdpa_reduce_fb39d3` (o-proj main matmul consuming the attention output) was at eager
  parity pre-fix (129 µs); post-fix it deploys 189 µs std / 199 µs fm (0.65–0.69× vs eager), and each tune's
  -O3 bench prints the (new, intentionally loud) drift warning:
  `node 'linear_3' matches golden shape ShapeKey(free_prod=3840, reduce_max=4096, is_warp=True, is_dyn=True) …
  but no offered candidate realizes any of them — falling through to the normal evidence hierarchy.
  Investigate enumeration drift for: gemma4_12b.o_proj.dynM`.
- **Evidence**: the golden `o_proj.dynM` entries ride `STAGE d2/tma/ring` + `REDUCE g4k`/`g2k`; the fused
  kernel's leaderboard (27 configs) offers NO stage and NO split-K — A is the computed attention output, the
  same computed-A class as finding 1. The -O1-ranking pick lands rank 8/27; its -O3 re-bench col reads 154 µs
  while the kernel-table deploy measured 189 µs (run-to-run spread on this kernel is real, ±20%).
- **Root cause**: same computed-A gate as finding 1. The *regression* vs pre-fix is a join-behavior change: the
  gate fixes made the golden ShapeKey match this node (pre-fix it didn't match and the evidence hierarchy landed
  a lucky parity config); now the matched-but-unrealizable golden falls through loudly and the fallback pick is
  worse. The warning is working as designed (commit `8d631db6` made unreached goldens loud) — the enumeration
  gap it names is the thing to fix.
- **Fix (P1, ~70 µs/layer)**: same two levers as finding 1 (async B-slab staging on computed-A forms; or record
  an in-model-form golden for the fused SDPA→o-proj shape).
- **Repro**: `emmy run --ir <dump-l0>/08_lowering_cuda.kernels/k_linear_sdpa_reduce_fb39d3.torch.json --bench
  --ab "TILE=a:mma_m16n8k16_f16_f16/w4x2/f4x8/k4,REDUCE=g2k,STAGE=d2/tma/ring"`.

## Finding 3 — fm whole-layer capture anomaly: e2e row 2807 µs vs per-kernel sum ≈ 2082 µs

- **Symptom**: the fm full-layer e2e emmy row (2807 µs, 0.50×) exceeds the sum of its own per-kernel -O3 rows
  (≈ 2082 µs) by ~35%; the std row is consistent (2171 vs ≈ 2242 sum, overlap-plausible). The pre-fix report hit
  the same fm-only inflation (its fm e2e regressed while every fm per-kernel row improved) and flagged it as a
  whole-program capture anomaly; this run reproduces it.
- **Class**: measurement/harness (the whole-program CUDA-graph capture path), not kernel quality. Distinguishing
  diagnostic for a follow-up: capture the fm program once and profile the graph replay (`--profile` on the
  layer run) — if the inter-kernel gaps carry the missing ~700 µs, the capture stream/dependency chain is the
  cause; if a kernel runs slower captured than solo, it's a clock/occupancy interaction.
- **Priority P2** — it corrupts the headline e2e ratio every fm run; per-kernel tables stay trustworthy.

## Finding 4 — whole-model tune is blocked: `unknown dtype 'bool'` in the full-model trace

- **Symptom**: `emmy tune google/gemma-4-12B --dynamic seq_len@input_ids:1 …` (whole-model) dies in the trace:
  `ValueError: unknown dtype 'bool'` — so the skill's whole-model e2e table cannot be produced; layer-0 is the
  fallback scope (serving covers e2e instead).
- **Root cause**: scalar literals inherit the CONSUMING op's output dtype (`_resolve_inputs`,
  `emmy/compiler/trace/torch.py:343`) and the whole-model graph contains bool-output ops (the explicit-mask
  construction that the layer-0 wrapper avoids); `dtype.py:111` has no `bool` entry. Layer scope never hits it.
- **Fix (P2)**: add a `bool`/`i8` dtype (or make the scalar-const dtype inheritance fall back to the literal's
  own dtype when the consumer's output dtype is non-numeric). Compile-only repro, no GPU:
  `emmy compile google/gemma-4-12B --dynamic seq_len@input_ids:1 --dynamic seq_len@attention_mask:2 --dynamic
  seq_len@attention_mask:3 --dynamic seq_len@position_ids:1 --ir input`.

## Finding 5 — hung-variant bench failures cluster on the two computed-A kernels

- **Evidence** (`eval failures`): std 20 rows — 11 on `k_linear_mean_reduce_241f7a` (7 `HungKernelError` >1 s,
  2 >2 s GPU budget, 2 worker wall-budget SIGKILL) and 9 on `k_scaled_dot_product_attention_reduce`; fm 11 rows,
  same two kernels. All hung rows share empty `REDUCE@a3`/`STAGE@a3`/`TILE@a3` assignments (the scalar-tier /
  unstaged corner) — consistent with the known gemma-4 scalar-hang class (`gemma4-longseq` memory; the golden
  YAML's own header notes cold greedy on `mlp_gate_up.dynM` "picks kernels that hang outright").
- **Cost**: wasted search slots (each hang burns a 1–16 s timeout) and they seed the online prior with
  2 000 000 µs pins. P3 — the timeouts contain the damage; enumerating fewer hang-prone scalar corners on these
  shapes (or a cheap static hang predictor) would recover the slots.

## Finding 6 — the -O1 ranking lane misprices the fm big-tile family (again)

- **Evidence**: on the fm gate⊗up leaderboard the deployed pick ranks **37/113 by -O1 latency (2.22× of the
  -O1 best)** yet its -O3 re-bench (760 µs) is near-best — while several -O1-top rows re-bench 1.4–5.5× slower
  at -O3 (e.g. the rank-20 row: 5300 µs -O1 → 4227 µs -O3; the rank-1 family inverts). The aggregate
  reachability view (`eval online --dataset db`, -O1 lane) reads mean 1.26× / worst 3.58× fm — but the worst
  rows are exactly the big-tile shapes where the -O1 lane itself is the distortion (the known `-O1 lane
  censoring` follow-up from the big-tile f16acc work, PR #350 notes).
- **Per-half attribution**: not probative for this run's binding findings — findings 1–2 are class-2
  never-enumerated gaps (no sibling rows exist below the missing `@STAGE`/`@REDUCE` families, so neither prior
  half can price them), and the visible pick-vs-best deltas are -O1-lane artifacts per above. No class-1
  (search-shortfall) finding is claimed, so no offline/online blame table is presented.
- **Fix (P3, standing)**: the planned -O1→-O3 refit / censoring correction for the ranking lane; until then,
  treat `eval variants` rank lines on big-tile fm kernels as untrustworthy without the -O3 column.

## Serving A/B — stock vLLM vs emmy plugin (std and FAST_MATH)

First local serving A/B of the 12B on this box (vLLM 0.23.0 in `venv-serving`; the `serving` extra's
`vllm<0.23` pin cannot dispatch gemma-4's head-size-512 global attention — the pin bump is still-pending item 3
of `plans/gemma4-prebuilt-kernel-bench-image.md`). Matched config, all rows: fp16, `--max-model-len 512`,
`--max-num-batched-tokens 256`, 64 prompts × (256 in / 128 out), concurrency 32, seed 0. Stock needs
`--language-model-only` to boot at small mnbt (the MM encoder budget check; `--limit-mm-per-prompt
'{"image":0}'` parses but does NOT clear it). Emmy runs: `EMMY_GEN_DECODE_BUCKET=0`,
`--gpu-memory-utilization 0.95` — see the fit note below. All 64 requests succeeded in every run.

| metric | stock vLLM | emmy std | emmy FAST_MATH |
| --- | --: | --: | --: |
| Request throughput (req/s) | **6.13** | 0.06 | 0.06 |
| Output token throughput (tok/s) | **784** | 7.4 | 7.5 |
| Mean TTFT (ms) | **1 097** | 107 248 | 96 971 |
| Mean TPOT (ms) | **24.9** | 2 928 | 2 909 |
| Bench duration (s) | 10.5 | 1 113 | 1 096 |

(A stock control at mnbt 512 measured the same: 6.08 req/s / 778 tok/s / TPOT 26.0 — stock is insensitive to
the mnbt reduction at this workload.)

**Verdict: emmy serving of the 12B is ~100× behind stock, and the gap is structural, not kernel math** — fm vs
std is a wash (±1.5%), exactly as predicted: with the decode-bucket twin off, EVERY decode step runs the
symbolic hint-512 M-tile program at M=1 (the documented ~66×-too-slow path, `serving/ARCHITECTURE.md`), and
per-layer pre/post host stitching adds the rest. The per-kernel tuning wins (goldens, gate fixes) are invisible
behind this integration wall.

### Post-fix follow-up (same night): PR #375 verified — the twin fits; its static shapes deploy cold

`main` landed PR #375 (share weight constants across the program twins + `BufferArena` activation pooling) and
PR #372 (vLLM ≥ 0.23 pin) mid-session; the tree was fast-forwarded and the decode-bucket configs re-benched
(same matched params; `HF_HUB_OFFLINE=1` after a Hub outage — two cosmetic snapshot files placeholder-touched):

| metric | emmy bucket off | emmy bucket 16, util .95 | emmy bucket 32, util .95 | emmy bucket 32, util .90 |
| --- | --: | --: | --: | --: |
| Output tok/s | 7.4 | 9.2 | CRASH | 6.6 |
| Mean TPOT (ms) | 2 928 | 2 997 | — | 3 639 |
| Requests ok | 64/64 | 64/64 | 31/64 | 64/64 |
| GPU fit | ~31 GiB | **31.4 GiB — fits** (pre-fix: ~44 GB OOM) | torch OOM at 1st batch-32 decode step | fits |

- **The memory goal is met**: the decode twin binds zero new weight bytes (PR #375's pointer-identity test) and
  the 12B serves with the bucket ON — the configuration that OOM'd at ~44 GB before the fix.
- **Bucket 16 at concurrency 32 never engages** (decode batch 32 > bucket → symbolic fallback) — bucket size
  must cover the decode batch (`min(max_num_seqs, concurrency)`), or the twin is dead weight.
- **util 0.95 with the bucket is an overcommit trap**: vLLM sizes KV before the twins' captured-graph
  workspaces grow the cupy pool; the first full-batch decode step then fails a 1.88 GiB torch (attention
  workspace) allocation. util 0.90 works post-#375 (the pooled arena lowered the baseline that previously made
  0.90 fail KV sizing).
- **Bucket 32 runs but is SLOWER than symbolic (TPOT 3 639 vs 2 997 ms)**: the static M=32 shapes have no
  goldens and no tune evidence, so all ~96 per-layer program builds deploy cold greedy picks (~30 ms/replay ×
  96 calls/token ≈ the observed TPOT) — the known cold-greedy-on-unseeded-shapes hazard (the gemma4 golden
  YAML's own header documents it for `mlp_gate_up.dynM`), now on the decode-shape family.

**Next lever (new)**: seed/tune the decode-bucket shape family — the M=`decode_bucket` static twins of every
projection (gate⊗up, down, q/kv/o at M=16/32) — as a golden set (like the s2048 family), so the twin deploys
tuned kernels. A tuned M=32 projection is ~50–100 µs, ~2 orders under the cold picks; only then does the
decode bucket's designed speedup materialize in serving. After that: whole-step CUDA-graph capture (drop
`--enforce-eager`) to collapse the 96 host launches per token.

**Memory-fit note (the config window is one knob wide).** With the twin off, the runner's cupy pool holds
~26 GiB invisible to vLLM's planner: at `util 0.97, mnbt 512` the server boots and passes `/health` but the
engine dies on the FIRST scheduler step (cupy OOM on a 33 MB lazy allocation — captured-graph workspaces);
at `util 0.90` vLLM refuses at startup ("No available memory for the cache blocks" — its device-wide snapshot
already contains the cupy pool, leaving zero KV budget). `util 0.95` + `mnbt 256` (halving the per-layer
symbolic activation buffers) is the working point. cupy also had to be installed into `venv-serving` by hand
(the `serving` extra documents it as a separate install).

## Repro / artifacts

- Work dir: `_tune/tune-model-gemma4-12b-repro-e2e/` — `tune-l0.log` / `tune-l0-fm.log` (tunes + -O3 bench
  tables), `dump-l0/` / `dump-l0-fm/` (kernel reproducers under `08_lowering_cuda.kernels/`, `62_kernel_bench.json`,
  `kernels.html`), `golden-std/` / `golden-fm/` + `summary-{std,fm}.txt` (golden sweep, per-name JSON),
  `serve-*.log` (serving benches), `tune-wholemodel-fp16-bool-trace-fail.log` (finding 4 traceback).
- Compile-only repros (no GPU): finding 4's `emmy compile` line;
  `EMMY_KNOBS="STAGE=d2/tma/ring" emmy compile <dump-l0>/…/k_linear_mean_reduce_241f7a.torch.json --ir cuda`
  shows the stage decline on the computed-A cone.

## Workflow notes

- **Whole-model tune is unusable for gemma-4** (finding 4, bool-dtype trace failure) — the tune-model skill's
  whole-model scope dies before the tune starts. Fix the trace (small dtype-table change) or teach the skill the
  layer-scope fallback explicitly. The whole-model *trace attempt* also host-OOM'd once (emmy killed at ~45 GB
  RSS during `torch.export` of the 48-layer graph) — a `--layer`-scoped trace never does; worth a memory note in
  the command docs.
- **Serving preflight is too late and too quiet.** Three boot attempts (~3 min each) to discover: (a) vLLM
  0.22.1 can't dispatch hd512 (needed a manual venv upgrade against the pyproject pin), (b) the MM budget check
  kills multimodal checkpoints at small mnbt and only `--language-model-only` clears it, (c) free-VRAM vs model
  size fails only deep inside engine boot. `emmy serve` could preflight all three in seconds: pin
  `vllm>=0.23` in the `serving` extra, pass `--language-model-only` automatically for `--stock --generate` on a
  multimodal checkpoint, and compare free VRAM against the checkpoint size before exec'ing vLLM.
- **Golden verification is a hand-rolled loop.** The 34-name × 2-regime sweep is `sweep.sh` + a bespoke
  `summarize_sweep.py` over per-name JSON. A built-in `emmy run --golden ALL --bench` (or `emmy eval golden
  --verify`) emitting exactly the per-name reproduce/join table would remove ~150 lines of session scripting —
  this is the second session to hand-roll it.
- **`eval variants` rank lines are misleading on fm big-tile kernels** (finding 6): the -O1 lane inverts against
  -O3 by up to 8× on this family, so "pick: rank 37/113 — misses best" reads as a search failure when the pick
  is near-best deployable. The -O3 column rescues it, but the verdict line itself should be computed from the
  -O3 column when present.
- **Background tune resumability worked well**: the std tune was killed mid-run (session clear) and re-running
  the same command without `--clean` resumed from the DB/cubin caches — the completing portion cost 2165 s
  instead of a full re-search. Worth documenting as the blessed interrupt-recovery path.
- **Desktop VRAM is a standing hazard for 12B-class serving on this box** (~6.8 GiB held by the session even
  when idle — it blocked the A/B until the user freed it). The tune/bench flows are unaffected.
- **The 12B serving fit window is one knob wide and every miss costs a full boot** (~6–10 min compile each):
  `util 0.97/mnbt 512` dies AFTER `/health` on the first step (cupy lazy OOM), `util 0.90` dies at KV sizing,
  `util 0.95/mnbt 256` works. Two improvements: `emmy serve` could ship the known-good 12B preset (twin off →
  force `mnbt ≤ 256`, util 0.95) and, deeper, the runner could pre-reserve its lazy allocation headroom at boot
  so post-`/health` OOMs become boot-time failures. Also: `cupy` missing from `venv-serving` cost one more boot
  cycle — fold it into the `serving` extra instead of a doc note.
- **Serving A/B wall-clock was dominated by emmy's own slowness** (~19 min/run at 64×128 tokens vs stock's
  10 s). For future A/Bs at this integration stage, 16 prompts × 32 output tokens would give the same verdict
  in a quarter of the time; the TPOT/TTFT means converge long before 64 requests.
- Previous report's workflow notes: the fm-tune silent wedge did NOT recur with `-v` (fix held); the e2e
  capture anomaly recurred (now finding 3, escalated from a caveat to a finding).
