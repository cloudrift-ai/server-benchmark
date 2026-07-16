# Gemma-4-12B layer-0 tune findings — FAST_MATH A/B + golden reproduction audit (RTX 5090)

- **Status**: complete; **gate fixes LANDED same day** (branch `feature/gemma4-inmodel-golden-gates`) — see the
  "Fixes landed" section at the end. Golden reproduction is a clean pass (136/136 pinned rows within ~1% across
  both regimes); the layer-0 in-model deploy was the problem — emmy 0.80×/0.76× vs eager because the model's
  fused/viewed kernel forms were locked out of the transports and splits the goldens ride (three code-cited
  gates, findings 1–3).
- **Run commands** (2026-07-15, local RTX 5090, driver 580.159.03; per-run `EMMY_TUNE_DB` / `EMMY_ONLINE_FILE`
  under `_tune/tune-model-gemma4-12b-fm-ab/`, never the user cache):

  ```bash
  emmy tune google/gemma-4-12B --layer 0 --dynamic seq_len@x:1 --clean --bench --dump-dir <dir>/dump-std   # std
  EMMY_FAST_MATH=1 emmy tune google/gemma-4-12B --layer 0 --dynamic seq_len@x:1 --clean --bench \
      --dump-dir <dir>/dump-fm                                                                             # fm
  emmy run --golden <name> --bench --json <dir>/golden-{std,fm}/<name>.json   # ×34 names ×2 regimes
  ```

- **Run stats**: std search 1570 s, 446 ok / 20 `bench_fail` DB rows; fm search 3345 s (the fm gate roughly
  doubles the atom enumeration), 539 ok / 28 `bench_fail`. The FIRST fm attempt wedged silently for 35 min
  (workflow notes); the `-v` rerun completed. Golden sweeps: 34 names × 2 regimes ≈ 6 min each (warm cubin cache).
- **Regime disclaimer**: all table numbers below are the `--bench` -O3 re-bench (deployable, CUDA-graph captured);
  tune-DB latencies quoted for ranking context are `-Xcicc -O1` and never comparable to them.
- **Dynamic run**: symbolic `seq_len`, everything benched at the 512 hint (`benched at seq_len=512 (symbolic
  hint; torch inputs tiled to match)`). Layer 0 is a `sliding_attention` layer (hd256, 16 Q / 8 KV heads,
  window 1024 — at the 512 hint the trace carries plain `is_causal`, no explicit mask). Single-layer scope: no
  servable artifact, so no serving A/B (step 2b skipped by design).

## Golden reproduction — PASS standalone; the greedy join deviates on 4 (std) / 3 (fm) of 34 names

Every recorded entry of `goldens/rtx5090_sm120_gemma4.yaml` (68 entries / 34 names, std + fm siblings, static +
dynM + s2048) was re-benched pinned via `run --golden NAME --bench --json`, in both regimes:

- **Pinned reproduction: 136/136 rows `ok`, all within ~1% of the recorded `emmy_us`** (worst 1.01×; integrity
  flags empty). The goldens are healthy as measurements — no stale µs, no pin drift, on either regime.
- **Greedy-deploy join** (fresh DB, so pure goldens → cold prior): std follows the golden on 30/34 names, fm on
  31/34, and under `EMMY_FAST_MATH=1` every followed matmul name deploys its **fm** entry (the umbrella works).
  The seven deviations, each root-caused:

| name (regime) | greedy µs | golden µs | verdict |
| --- | --: | --: | --- |
| q_proj.s2048 (std) | 402.3 | 313.0 | drift warning: split-K goldens don't realize unpinned at M=2048 — `free/tile_area > _SPLITK_MAX_CTAS=512` (`lowering/tile/_schedule.py:344,500`); the pin path bypasses the guard, so the recording is reachable only pinned |
| mlp_down.s2048 (std) | 1204.5 | 1112.0 | same CTA-cap drift class as q_proj.s2048 |
| attention.hd512 (std+fm) | 158.0 | 113.0 | drift warning — CORRECTED verdict (07-15, post-fix analysis): the recorded config (`g2k` split-KV + `d1/cp/alt`) DOES realize pinned — the pin bench's kernel is staged (97.5 KB smem, `record_knobs` carry the stage, no integrity flag). The enumeration resolves flash stage candidates on the PRE-split geometry, where the hd512 slabs don't fit — so the (split + staged) combination is never OFFERED and the join can't reach it. Enumeration follow-up (resolve stages post-split), not a recording artifact |
| attention.hd512.dynM (both) | 134.2 | 116.6 | silent slower-sibling deploy: the faster entry adds `REDUCE: g2k` (split-KV), which realizes pinned but is never OFFERED on the dynamic flash fork, so the join falls to the shape's un-split second entry — no warning fires because *a* golden matched |
| k_proj_global.s2048 (both) | 36.1–45.6 | 37.2–47.2 | greedy silently BEATS the golden in both regimes (std `w4x2` vs recorded `w2x2`; fm `…/k4` vs recorded `…/k2`) — stale-golden update candidate |
| kv_proj.s2048 (fm) | 109.5 | 136.9 | greedy `f16_f16/w4x2/f4x8/k4` un-split big tile beats the recorded fm golden by 1.25× — strong update candidate |

Update candidates for a follow-up golden edit (per the manual-sweep convention: reproduce 3× before recording):
`kv_proj.s2048 [fm] w4x2/f4x8/k4 d2/tma/ring ≈ 109.5`, `k_proj_global.s2048 [fm] w2x4/f2x4/k4 g2k ≈ 36.1`,
`k_proj_global.s2048 [std] w4x2/f2x4/k2 g8k ≈ 45.6`. **RECORDED 2026-07-15** (3× reproduced at <0.5% spread,
verified deploying via `run --golden`); `attention.hd512` needs NO re-record — see the corrected verdict above.

## Bench results — layer-0 e2e and per-kernel (both -O3)

Full layer (eager / torch.compile / emmy, seq 512 symbolic hint):

| regime | Eager | torch.compile | Emmy | Emmy vs eager |
| --- | --: | --: | --: | --: |
| std | 1429 | 1248 | 1791 | **0.80×** |
| FAST_MATH | 1433 | 1249 | 1880 | **0.76×** |

(The fm e2e row regressing while every fm per-kernel row improves is a whole-program capture anomaly — the
per-kernel reproducer rows below are the stable signal; treat the e2e delta between regimes as noise pending a
re-run.)

Per-kernel (`--bench` table, emmy-descending; layer-op labels from the dump's `.torch.txt` provenance):

| kernel | layer op | eager | tcompile | emmy std | emmy fm |
| --- | --- | --: | --: | --: | --: |
| k_linear_mean_reduce | gate+up matmuls + gelu + next-norm mean (fused, computed-A) | — | — | 809 | 620 |
| k_mean_linear_reduce | norm mean + down-proj finalize (fused) | 487 | 308 | 365 | 320 |
| k_linear_reduce | down-proj main matmul | 295 | 290 | 345 | 308 |
| k_scaled_dot_product_attention_reduce | flash SDPA (hd256 causal) | 33 | 33 | **208** | **132** |
| k_linear_sdpa_reduce | o-proj main + SDPA epilogue edge | 129 | 127 | 129 | 130 |
| k_mean_linear_reduce (×4 more) | q/k/v-proj + norm-mean fusions | 253 | 105 | 103–124 | 73–124 |
| k_mean / rope pointwise (×5) | means, rotary sin/cos prep | 39–141 | 2–12 | 2–7 | 2–7 |

Dominators: the MLP fusions (809+365+345 ≈ 1519 µs std) plus SDPA (208) carry ~85% of the emmy total. The
pointwise/mean tail beats eager 14–25× and needs no work.

## Finding 1 — in-model flash SDPA is 6× off torch: fp32/strided operands block staging and force demotes

- **Symptom**: `k_scaled_dot_product_attention_reduce` 208 µs std / 132 µs fm vs torch SDPA 33 µs — while the
  SAME logical op standalone (golden `attention.hd256.dynM`) runs 32–35 µs and reproduces to 1%.
- **Evidence**: the emitted in-model signature is
  `k_…(const __half* transpose, const float* _flash_scale, const __half* transpose_1, const float* mul_10, …)` —
  K/V arrive as transpose VIEWS and Q as the **float32** rope product `mul_10`; the standalone golden form is
  `k_…(const __half* x1, const __half* x2, const __half* x0, …, const CUtensorMap* _desc_k, _desc_v)` — raw
  contiguous fp16 + realized TMA descriptors. Pinning the golden configs on the in-model reproducer declines
  loudly: `STAGE=d1/cp/alt realized (off)` / `STAGE=d2/tma/ring realized (off)` (rows correctly kept unbenched),
  and the tune DB's 21 measured configs all have empty `STAGE@kv` — the async transports were never offered
  in-model (`eval variants --kernel scaled_dot`; pick rank 1/21, so NOT a search shortfall).
- **Root cause** (class 2, tier/optimization lockout): the flash K/V slab transports move raw gmem bytes
  (`_resolve_twisted_stage`, `lowering/tile/_schedule.py:1660-1717`) and decline the model's viewed/fp32 operand
  forms; the fp32 Q additionally rides per-element demotes into every QK mma. No structural escape exists today:
  `PLACE@cone=cut` does not apply to this kernel (realized `PLACE@fold=fuse` only).
- **Fix suggestion (P1, ~175 µs/layer at stake — sliding layers ×40)**: materialize/demote the flash operands
  before recognition (fp16-contiguize rope'd Q/K and the V view — a few µs of pointwise work), which restores the
  standalone geometry, the staging offers, and golden reachability in one move.
- **Repro**: `emmy run --ir <dump-std>/08_lowering_cuda.kernels/k_scaled_dot_product_attention_reduce.torch.json
  --bench --ab "TILE=a:mma_m16n8k16_f16_f32/w4x1/f1x4/k16,STAGE=d2/tma/ring"` (decline is compile-time — no GPU
  needed with `emmy compile … --ir cuda`).

## Finding 2 — fused computed-A matmuls lock out split-K and async staging (by design), ~40% on the biggest kernel

- **Symptom**: the gate+up fusion (`k_linear_mean_reduce_241f7a`) deploys 809 µs std; the standalone
  `mlp_gate_up.dynM` golden is 562 µs (+ ~7 µs for the mean it absorbs).
- **Evidence**: all 45 measured configs are `STAGE d1/sync` (`eval variants`); pinning the golden
  `TILE=…w4x2/f4x8/k4,STAGE=d2/tma/ring` realizes the TILE but degrades the stage to `d2/sync`
  (pin_unmatched, unbenched). The deploy drift warnings for `mlp_down.dynM` / `o_proj.dynM` fired in-model during
  the tune's bench while the same goldens deploy fine standalone. `eval online --dataset nodes` (both runs)
  anchors it: the goldens' branches were "never built below @STAGE / @TILE".
- **Root cause** (class 2, correct-by-design gate): a computed-A (fused-cone) contraction offers no split-K —
  "a producer-cone A cannot be sliced over K" (`_reduce_candidates`, `lowering/tile/_schedule.py:500-505`) — and
  its A slab must ride the sync compute-fill, which the current resolver extends to the whole stage decision.
  `PLACE@cone=cut` is not a usable escape: on the reproducer it collapses the kernel to a 65 ms scalar form (the
  cut un-nodifies the contraction).
- **Fix suggestions (P1, ~240 µs on gate_up + ~120 µs spread over the q/kv/o fusions)**: (a) stage the RAW B
  (weight) slab async while A rides the compute-fill — B is a plain gmem operand even when A is computed; (b)
  record in-model-form goldens (the fused kernels' `record_knobs` from the tune bench JSON) so the deploy join
  stops falling to the cold prior on these shapes.
- **Repro**: `emmy run --ir <dump-std>/…/k_linear_mean_reduce_241f7a.torch.json --bench
  --ab "TILE=a:mma_m16n8k16_f16_f32/w4x2/f4x8/k4,STAGE=d2/tma/ring" --ab "PLACE@cone=cut"`.

## Finding 3 — rank-3 (unit-batch) operands decline TMA on every in-model matmul

- **Symptom**: the down-proj main matmul deploys 345 µs std vs the 286 µs standalone golden; its leaderboard
  offers `d1/cp`-family only. Pinning the golden `…k2,REDUCE=g4k,STAGE=d2/tma/ring` realizes TILE and g4k but
  `STAGE=d2/tma/ring realized (off)`; the same pin with cp transport (`d2/cp/ring`) realizes and runs.
- **Root cause** (class 2): the TMA descriptor's box is 2-D over the operand's own array —
  `tma_rank_ok = len(a.index) == 2 and len(b.index) == 2` (`_resolve_warp_stage`,
  `lowering/tile/_schedule.py`) — and the model's `[1, seq, K]` unit-batch views are rank-3, so every in-model
  matmul silently downgrades TMA → cp.async. The standalone golden snippets are rank-2, hence the split.
- **Fix suggestion (P2, ~60 µs/layer)**: squeeze literal-1 leading dims before/at recognition (or fold them into
  the descriptor's globalDim) so in-model operands match the standalone rank.
- **Repro**: `emmy run --ir <dump-std>/…/k_linear_reduce_dbf34f.torch.json --bench
  --ab "TILE=a:mma_m16n8k16_f16_f32/w4x2/f2x4/k2,REDUCE=g4k,STAGE=d2/tma/ring"`.

## Finding 4 — FAST_MATH A/B: the umbrella works end-to-end; fm halves the SDPA gap via FAST_EXP

- Under `EMMY_FAST_MATH=1`, every matmul-shape golden deploys its **fm** sibling (golden sweep, 31/34 names) and
  the layer tune's per-kernel rows improve across the board (gate_up 809→620, SDPA 208→132, down-proj 345→308).
  SDPA's fm win is `FAST_EXP=True` (`__expf` in the softmax; the fm DB rows carry it) — it cannot reach the
  f16acc PV atom in-model for the same staging/lockout reasons as finding 1.
- fm search cost: 3345 s vs 1570 s (2.1×) for the doubled atom enumeration — worth knowing when budgeting tunes.
- The fm e2e whole-program row (1880 vs std 1791) contradicts the per-kernel wins; treated as capture noise (see
  bench table note) — the reproducer rows are the before/after signal.

## Finding 5 — bench-failure clusters: the scalar masked-SDPA / scalar fused-linear hang class burns search slots

- std: 20 `bench_fail` rows — 15 on `k_scaled_dot_product_attention_reduce` (scalar-tier rows, empty `TILE@dd`,
  `HungKernelError` 1 s probe / 2 s run / 16 s wall SIGKILL), 5 on scalar `k_linear_mean_reduce`; fm: 28 rows,
  same clusters (10 SIGKILL rows share `FAST_EXP=True, PLACE@fold=fuse, STAGE@kv=`). Same hazard family the
  gemma4 golden seeding recorded (scalar b256 misdeploys / hangs); the workers contain them correctly, but each
  costs 2–16 s of wall — a scalar-tier extent guard for `free_prod ≥ ~4M` flash rows would give the slots back.

## Prior-half diagnostics (std run; fm mirrors it)

| metric | offline prior | online prior |
| --- | --- | --- |
| leaf reachability (19 ops) | mean 1.78× · median 1.55× · worst 3.54× | mean 1.20× · median 1.02× · worst 1.88× |
| leaf reachability (15 ops, 2nd block) | mean 1.83× · median 1.39× · worst 6.36× | mean 1.15× · median 1.00× · worst 1.74× |
| o_proj.dynM golden anchor | never built below @STAGE; lost @TILE 1.25×; -O3 pick/golden 1.62× | same shape: lost @TILE 1.16×; pick/golden 1.17× |
| mlp_down.dynM golden anchor | never built below @STAGE; lost @TILE 1.66×; pick/golden 2.35× | lost @TILE 1.11×; pick/golden 1.21× |

The anchors blame the ENUMERATION, not either prior half: the golden subtrees were never materialized (findings
1–3), so regret conditioned on measured forks stays clean while the deploy misses the goldens — exactly the blind
spot the golden-anchored descent view exists to make loud. No calibration action on either half from this run.

## Repro / artifacts

- Work dir: `_tune/tune-model-gemma4-12b-fm-ab/` — `tune-{std,fm}.log`, `dump-{std,fm}/` (incl.
  `62_kernel_bench.json`, `kernels.html`, per-kernel `.torch.json` reproducers), `golden-{std,fm}/<name>.{json,log}`
  (68 pinned A/B records per regime), `summary-{std,fm}.txt`, `drill-*.json` (the finding A/Bs), `sweep.sh` /
  `summarize_sweep.py` (the sweep driver).
- Compile-only repros (no GPU): every finding's pin decline reproduces via
  `EMMY_KNOBS="…" emmy compile <reproducer>.torch.json --ir cuda` — the realized-vs-pinned knobs and the emitted
  signature carry the whole diagnosis.
- NCU was not needed: all three gaps root-caused structurally (missing offers / realized-off pins), not at the
  counter level.

## Workflow notes

- **Silent fm-tune wedge (worst friction)**: the first `EMMY_FAST_MATH=1` tune froze after weight loading —
  35 min, 113 threads spinning (~24 cores), zero log bytes, zero GPU/compile activity; killed and the identical
  `-v` rerun completed. One occurrence, unreproduced, so cause unknown (the `-v`/`stdbuf` delta or plain flake).
  Proposal: a tune heartbeat line (phase + variant counter every ~60 s) so a wedge is distinguishable from a
  quiet compile phase without `ps` forensics; py-spy needs ptrace perms not available here.
- **`run --golden` has no batch mode**: sweeping 34 names × 2 regimes needed a hand-rolled shell loop + a JSON
  summarizer (`sweep.sh` / `summarize_sweep.py`). The per-name `--json` records are perfect inputs — an
  `emmy eval golden --bench` (or `run --golden all`) emitting the reproduction verdict table would collapse ~1 h
  of scripting into one command. The `pick-matches-golden` join in my summarizer also re-implements
  prefix-consistency approximately; a deploy-time "golden followed: <name>" log line on SUCCESS (there's only a
  failure warning today) would make the check trivial and exact.
- **Silent slower-sibling deploys**: when the join's fastest entry doesn't realize but a slower sibling does
  (attention.hd512.dynM), nothing logs. A one-line "golden <name>: fastest entry unrealizable, deployed sibling
  (+16%)" would have saved a manual JSON diff.
- **Per-kernel bench torch refs missing**: 10 of 17 kernels print `-` for eager/tcompile (fused reproducers whose
  torch closure bench didn't run), so `vs eager` is unavailable exactly on the most interesting (fused) kernels;
  the 809 µs dominator has no reference row. Worth a look at the reproducer-bench fallback.
- **First-run flakiness of drill-run greedy picks**: cold-prior greedy on a fresh drill DB picked 1451 µs
  (gate_up) / 287 µs (SDPA) forms vs the tune's 803 / 208 deploys — reproducer A/Bs should always pin both sides
  (`--ab` the tune's pick too) rather than trust the greedy row, which this report's numbers do.
- **Kernel chart PNG**: `kernels.html` renders but the PNG export needs `playwright install` (absent chromium);
  harmless, one log banner per tune.
- **What worked well**: `run --golden --json` (integrity flags + `record_knobs` made the audit mechanical),
  `eval variants`' STAGE column emptiness as the lockout fingerprint, the realized-vs-pinned decline messages
  (every root cause fell out of them), and the golden-anchored descent rows agreeing with the deploy warnings.

## Fixes landed (2026-07-15, branch `feature/gemma4-inmodel-golden-gates`)

Same-day follow-up: three of the gates above are fixed and verified against this report's drill reproducers.

- **Finding 3 (TMA rank-3)** — `_tma_operand_rank_ok` (`lowering/tile/_schedule.py`): TMA now boxes extra LEADING
  operand dims as extent-1 box dims when they are tile/K-invariant (the flash K/V rank-N convention, extended to
  both matmul tiers; `_slab_operands` emits the rank-N box). The down-proj reproducer's pinned golden
  (`…k2,g4k,d2/tma/ring`) realizes and benches 282.9 µs ≈ the standalone 286, and greedy itself deploys the golden
  in-model (288 µs, 1.02× vs eager — was 345 / 0.86×).
- **Finding 1 (flash operand lockout)** — two composing changes: `frontend/optimization/005_split_cast_from_indexmap`
  splits a traced dtype-changing view into a source-shaped elementwise `copy` + a pure map, and loop fusion's
  plumbing exemption now admits only dtype-PRESERVING copies (`_is_castfree_indexmap` + the cast-vs-plumbing guard
  in `merge_loop_ops`), so the cast stays a materialized f16 buffer at flash offer sites. The in-model SDPA
  reproducer: greedy deploys the golden `d1/cp/alt` config at 32.2 µs + a 2.6 µs cast kernel (e2e 0.89× vs eager —
  was 0.11–0.16×, ~6×); pinned `d2/tma/ring` / `d1/cp/alt` realize at 33.9 / 32.1 (recorded 34.4 / 32.0).
- **s2048 split-K cap** — `_SPLITK_MAX_CTAS` 512 → 1024: the recorded `q_proj.s2048` / `mlp_down.s2048` split-K
  goldens sit at 960–1024-CTA grids and now realize unpinned; greedy deploys them (315 µs was 402; 1102 was 1205).
- **Finding 2 (computed-A)** — resolved by analysis, no code change: the sync transport already cp.asyncs the
  canonical B slab under the compute-fill, and the split-K refusal on a computed A is correct by design (the cone
  cannot slice over K). The fused-form residual is codegen depth, not a gate.
- **Correctness**: full suite green (2434 passed); layer-0 accuracy at fixed seed PASSES with a slightly better
  mean error than baseline (0.277 vs 0.292 mean_diff; the one unseeded FAIL observed during verification was the
  fp16 outlier ceiling at 1.07× on an unlucky draw — baseline runs the same mean band). Post-fix focused golden
  sweep: 25/25 pinned rows ≤1.03× of recorded, zero regressions.
- **Golden YAML refreshed** (same day): the three update candidates above are recorded and verified deploying.
- **Still open** (documented, out of scope here): the flash fork resolves stage candidates on the PRE-split
  geometry, so `attention.hd512`'s recorded (split-KV + `d1/cp/alt`) config — which realizes pinned — is never
  offered at deploy (same family: `attention.hd512.dynM`'s faster split-KV entry is not in the dynamic flash
  offer); in-model
  `o_proj.dynM` drift remains (its A is the flash output through a K-splitting reshape — TMA correctly declines;
  greedy now deploys a faster `g2k d2/cp/ring` form, 118 µs was 129); the cold-prior misdeploys on split-partial
  forks (no goldens there) dominate the untuned layer total — a search/prior story, not an enumeration gate.
