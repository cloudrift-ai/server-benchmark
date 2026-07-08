# Golden sweep — RTX 5090 (sm_120), 2026-07-08 — first sweep under the refit analytic weights

Full cold re-tune + greedy-vs-golden A/B of all 36 recorded goldens on a freshly rented CloudRift RTX 5090
(instance `67ae9a7a-7a58-11f1-946b-cb9ca63c60b2`, billing-exempt), on branch `feature/reduce-featurizer-fix`
(495962b3 — the refit analytic prior + reduce featurization + evidence-join fix). Numbers are `-O3` deployable; A/B
rows are live re-benches from `emmy run --bench --golden NAME` on the box. `ratio` = `greedy / best-golden`
(>1 = greedy slower); `vs cuBLAS` = `greedy / recorded cublas_us` (>1 = emmy slower than PyTorch/cuBLAS).

- **Sweep command:** `emmy tune --dataset golden --clean` — ONE cold invocation, no prior seeding, default patience.
  It expanded to **39 tune targets, not 36** — the golden dataset unions shape names across ALL card YAMLs (see
  Finding 2, a real dtype-shadowing bug, not just a count quirk).
- **Wall time:** tune 12,885 s ≈ **3.6 h** summed per-shape (log `done: 39/39 shape(s)`; heaviest:
  `qkv.h4096` 1150 s, `square.4096.fp16` 1080 s, `attention.hd64` 1076 s). The 36-shape A/B pass was **7.5 min**
  total (~12–18 s/shape — the sweep's -O3 rebenches leave the cubin cache warm), plus 2×5 noise-floor re-runs and
  two pin probes ≈ 15 min. GPU-busy total ≈ 3.9 h (within the 4.5 h cap; box wall-clock was much longer due to two
  idle stalls between phases — monitor notifications never fired, see Workflow notes).
- **Category tally:** **4 replaced / 0 added / 27 unchanged / 5 worse.**
  - **Replaced (4):** `matmul.square.512.fp16` (w2x4/f2x2/k4·d4 → w1x8/f4x1/k4·d3, 4.5→4.2 µs, recorded 3.62 was
    stale-fast), `matmul.mlp_gate_up.h4096` (w8x2/f2x8/k4+g2k·d2/tma/ring → **unsplit** w2x2/f4x4/k4·d1/tma,
    659→605.5 µs), `matmul.o_proj.h4096.dynM` (w4x2/f2x4/k4 → w2x4/f2x2/k4, both g2k·d2/tma/ring, 107→97.7 µs),
    `attention.hd64.dynM` (bare TILE w4x1/f1x8/k4 → w4x1/f1x16/k4 [pj re-derives f1x8/k8], 9.3→8.6 µs).
  - **Worse (5), left untouched — the findings:** `matmul.mlp_down.h4096` (1.24×), `matmul.square.512.dynM`
    (1.22×), `attention.hd128` (1.21× vs recorded, golden pin silently no-ops), `matmul.o_proj.h4096` (1.18×),
    `matmul.mlp_down.h4096.dynM` (1.13×).
- **Fork sibling regret aggregate** (`emmy eval prior --dataset nodes`, RTX 5090 `-O1` block, 6944 nodes, 84 forks) —
  `ALL (median)`: PLACE+REDUCE+STAGE+TILE **1.06x** · PLACE+REDUCE+STAGE+TILE+WSPEC **1.03x** · REDUCE **1.00x** ·
  STAGE **1.02x** · **TILE 1.38x**. TILE is the steering gap: every matmul row sits at 1.34–1.76× (worst
  `free=4096 red=14336` = mlp_down at **1.76x**), while REDUCE/STAGE are exonerated at the fork level almost
  everywhere. Leaf reachability (-O1): mean 1.09×, median 1.02×, worst 1.56× (gate_up). Leaf calibration
  (median per-op Spearman): +0.90 (-O1), +0.80 (-O3).
- All four wins reproduced 3/3 in the noise-floor re-runs (ratios: gate_up 0.918/0.918/0.917, o_proj.dynM
  0.914/0.913/0.912, hd64.dynM 0.925/0.925/0.926, square.512.fp16 0.933/0.933/0.933). After recording, every new
  golden row re-benches at the greedy config and latency (4.2 / 619.2 / 99.8 / 8.7 µs — within noise of the
  greedy 4.2 / 606 / 97.5 / 8.6).

## The headline: the analytic refit worked — the misses moved from the heuristic to the search

Last sweep's core diagnosis ("the LINEAR `_W_A` misprices split-K; rank 9116 for the `mlp_down` golden") is
**fixed**: under the refit weights `eval analytic` now ranks the `mlp_down.h4096` golden **0/14805** (top-1 cold),
`o_proj.h4096` **0/14805**, `mlp_gate_up.h4096` **0/7569** (and dynM twins 2358 / 389 / 0). Yet the deployed greedy
still misses two of those three (and the third was won by a *different* config than the golden). The failure moved
downstream: the goldens' configs were **never measured** this sweep (`eval variants`: all 63+16 measured `mlp_down`
variants are `d4/cp/ring` with only `g2k`; all 37+6 `o_proj` variants likewise), and the *learned* prior — trained
within-sweep on the shapes tuned before them — buries what the analytic ranks #1 (`eval prior --dataset golden`:
mlp_down rank **8215**/14805, o_proj **4558**/14805). The fork-regret view says where: the TILE family (median
**1.38x**, mlp_down row 1.76x). The search descends the TILE/STAGE forks the learned prior likes (cp-staged, g2k)
and patience expires before the tma+g8k subtree is ever opened — a censoring loop between within-sweep learning and
fork-level steering, no longer an analytic-pricing problem.

## Full outcome table (all 36 shapes)

`greedy µs` = deployed pick, kernel-sum TOTAL from `run --bench` (split-K greedy picks also print a whole-program
e2e — noted per finding); `golden µs` = the live best golden-row sum from the same run; `cuBLAS µs` = recorded
`cublas_us` (live `Eager PyTorch` agreed within noise everywhere). Reference per kind: matmul fp16 → HGEMM (fp32
`square.512` → SGEMM), attention → torch SDPA, softmax/rms_norm → torch fused eager, reduce → unfused `torch.sum`
(**weak**), pointwise → torch `relu`.

| shape | S/D | greedy µs | golden µs | ratio g/gold | cuBLAS µs | vs cuBLAS | category |
|---|---|--:|--:|--:|--:|--:|---|
| matmul.square.512 (fp32) | sta | 8.2 | 8.6 | 0.95 | 12.28 | 0.67 | same (diff knobs, sub-noise; both beat recorded 10.24) |
| matmul.square.512.fp16 | sta | 4.2 | 4.5 | 0.93 | 6.14 | 0.68 | **replaced** (3/3; recorded 3.62 was stale) |
| matmul.square.1024 | sta | 15.8 | 16.2 | 0.98 | 14.51 | 1.09 | same |
| matmul.square.2048 | sta | 110.8 | 112.2 | 0.99 | 98.36 | 1.13 | same |
| matmul.square.4096 | sta | 654.7 | 640.9 | 1.02 | 640.93 | 1.02 | same |
| matmul.square.512.dynM | dyn | 5.5 | 4.5 | **1.22** | 6.14 | 0.90 | **worse — Finding 2** |
| matmul.qkv.h4096 | sta | 255.4 | 259.3 | 0.99 | 250.54 | 1.02 | same |
| matmul.o_proj.h4096 | sta | 123.3 | 104.2 | **1.18** | 95.22 | **1.29** | **worse — Finding 4** |
| matmul.mlp_gate_up.h4096 | sta | 605.5 | 659.4 | 0.92 | 557.94 | 1.09 | **replaced** (3/3; unsplit d1/tma beats g2k) |
| matmul.mlp_down.h4096 | sta | 386.0 | 311.2 | **1.24** | 296.95 | **1.30** | **worse — Finding 1** |
| matmul.qkv.h4096.dynM | dyn | 268.0 | 259.6 | 1.03 | 249.82 | 1.07 | same (borderline) |
| matmul.o_proj.h4096.dynM | dyn | 97.7 | 106.9 | 0.91 | 95.08 | 1.03 | **replaced** (3/3) |
| matmul.mlp_gate_up.h4096.dynM | dyn | 597.8 | 613.1 | 0.98 | 553.70 | 1.08 | same |
| matmul.mlp_down.h4096.dynM | dyn | 351.8 | 310.1 | **1.13** | 295.03 | **1.19** | **worse — Finding 1** |
| softmax.k2048 | sta | 3.8 | 3.8 | 1.00 | 4.1 | 0.93 | same (same knobs; prev Finding 3 resolved — greedy deploys b512) |
| reduce.k2048 | sta | 1.6 | 1.6 | 1.00 | 16.38 | 0.10 | same (same knobs) |
| softmax.k8192 | sta | 10.4 | 10.5 | 0.99 | 14.29 | 0.73 | same (same knobs) |
| reduce.k8192 | sta | 3.4 | 3.4 | 1.00 | 16.39 | 0.21 | same (same knobs) |
| rms_norm.k2048 | sta | 4.0 | 4.0 | 1.00 | 4.1 | 0.98 | same (same knobs) |
| rms_norm.k4096 | sta | 6.8 | 6.8 | 1.00 | 6.14 | 1.11 | same (same knobs) |
| rms_norm.k8192 | sta | 10.3 | 10.4 | 0.99 | 10.24 | 1.01 | same (same knobs) |
| pointwise.n4096 | sta | 3.4 | 3.4 | 1.00 | 4.1 | 0.83 | same (same knobs) |
| pointwise.n16384 | sta | 11.5 | 11.4 | 1.01 | 12.51 | 0.92 | same (same knobs) |
| softmax.k2048.dynM | dyn | 3.8 | 3.8 | 1.00 | 4.1 | 0.93 | same (same knobs) |
| reduce.k2048.dynM | dyn | 1.5 | 1.5 | 1.00 | 16.38 | 0.09 | same (same knobs) |
| softmax.k8192.dynM | dyn | 10.4 | 10.5 | 0.99 | 14.27 | 0.73 | same (same knobs) |
| reduce.k8192.dynM | dyn | 3.4 | 3.4 | 1.00 | 16.39 | 0.21 | same (same knobs) |
| rms_norm.k2048.dynM | dyn | 4.0 | 4.0 | 1.00 | 4.1 | 0.98 | same (same knobs) |
| rms_norm.k4096.dynM | dyn | 6.6 | 6.6 | 1.00 | 6.14 | 1.07 | same (same knobs) |
| rms_norm.k8192.dynM | dyn | 10.3 | 10.4 | 0.99 | 10.25 | 1.00 | same (same knobs) |
| pointwise.n4096.dynM | dyn | 3.4 | 3.4 | 1.00 | 4.1 | 0.83 | same (same knobs) |
| pointwise.n16384.dynM | dyn | 11.5 | 11.4 | 1.01 | 12.52 | 0.92 | same (same knobs) |
| attention.hd64 | sta | 8.4 | 8.5 | 0.99 | 10.24 | 0.82 | same (same knobs) |
| attention.hd128 | sta | 19.9 | 20.0* | **1.21 vs recorded 16.5** | 18.42 | **1.08** | **worse — Finding 3** (*golden row ≠ recorded config) |
| attention.hd64.dynM | dyn | 8.6 | 9.3 | 0.93 | 10.23 | 0.84 | **replaced** (3/3; bare TILE f1x16/k4) |
| attention.hd128.dynM | dyn | 16.5 | 16.7 | 0.99 | 18.4 | 0.90 | same (same knobs; PR #302-era staged form holds) |

## Finding 1 — `mlp_down` g8k split-K: analytic now prices it top-1, but the search never benches it

Static greedy 386.0 µs (e2e 397.5) vs golden 311.2 µs → **1.24×**; dynM 351.8 vs 310.1 → **1.13×**. Both greedy
picks are `g2k` splits; the goldens are `g8k`. Same shape as the #318-era Finding 1, but the evidence chain has
inverted:

- `eval golden --kernel mlp_down` — found/golden: TILE `w4x1/f4x8 / w4x4/f2x2/k4`, REDUCE `g2k/g8k`, STAGE
  `d4/cp/ring / d2/tma/ring` (static). Three families wrong at once: split factor, staging transport, warp tile.
- `eval analytic --kernel mlp_down` — golden rank **0/14805** (static, was 9116 pre-refit) and **2358/14805**
  (dynM, was ~1351–3833). **The refit analytic weights fixed the static mispricing** — the cold prior would now
  propose the golden first.
- `eval prior --dataset golden` — learned rank **8215/14805** (static), **7997** (dynM); `vs gold` 1.24× / 1.16×.
  The within-sweep learned prior *buries* the config the cold analytic ranks #1 — it was trained only on the
  cp-staged `g2k` variants the search actually benched (censoring).
- `eval variants` (`k_matmul_cd8b18`/`fae29e`) — **reachability zero**: 63 + 16 measured static variants, every
  single one `d4/cp/ring`, split ladder only `g2k`; the `g8k`/`d2/tma/ring` golden was never compiled or benched.
  Deployed pick ranks 8/63 by -O1 but is the best **-O3** among measured (397.4 µs) — so within what it measured,
  deploy was right; the miss is pure reachability. (The -O1→-O3 inversion here is large: the -O1-rank-1 config is
  585.5 µs at -O3 vs the pick's 397.4.)
- `eval prior --dataset nodes` (`free=4096 red=14336` row) — fork regret **TILE 1.76x** (2 forks), no REDUCE/STAGE
  fork rows survived to score: the subtree carrying the golden was pruned at the TILE fork.

**Recommendation (high):** this is now a **search/steering** lever, not a weights lever. Two concrete options:
(a) seed the fork expansion with the analytic top-k *leaves* (force-bench the cold-analytic #1 config per shape —
one guaranteed bench of the golden-class config breaks the censoring loop at the cost of one compile); (b) train
the CatBoost prior on the cross-card node store (this sweep's 8429 5090 nodes + the 4090 sibling sweep) so the
learned prior stops unlearning the analytic's split-K pricing. Patience alone is the wrong lever — the fork regret
shows the walk is *steered* away, not stopped early.

## Finding 2 — `square.512.dynM` was never tuned: cross-card name shadowing drops the 5090's fp16 dtype

Greedy 5.5 µs (e2e 7.3) vs golden 4.5 µs → **1.22×**. The greedy pick (`w2x4/f2x2` no-k4, `g4k`, bare `d1/tma`)
missed the golden (`w2x4/f2x2/k4`, `g2k`, `d4/tma/ring`) because **the sweep never tuned this shape at its real
dtype**:

- tune.log target 9/39: `matmul.square.512.dynM → torch.matmul(torch.randn(512,512), torch.randn(512,512))` —
  **fp32**, while the 5090 golden is `dtype: fp16`.
- Root cause: `emmy/commands/tune.py:291` builds the dataset as `Dataset.from_golden(kernel=args.kernel)` —
  **without `live_gpu=True`** — and line 292's `by_name.setdefault(...)` keeps the *first* config per name across
  ALL card YAMLs. `rtx4090_sm89.yaml` also records `matmul.square.512.dynM` with **no dtype** (fp32 default), and
  it wins the dedupe. The same union is why the sweep ran 39 targets for a 36-entry YAML.
- `eval prior --dataset golden` confirms: `matmul.square.512.dynM SKIPPED: no tuned rows for this shape in the
  prior dataset` — the fp16 masked shape has no data, so the A/B's greedy pick was a cold generalization.
- `eval analytic` ranks this golden 1198/19085 (masked tier) — reachable-in-principle had the shape been tuned.
- The measured fp32 twin (`k_matmul_244b50`): pick rank 1/55 on the main kernel — the search did fine on the shape
  it was actually given.

**Recommendation (highest, one-line fix):** pass `live_gpu=True` in `_tune_targets`
(`emmy/commands/tune.py:291`) — `Dataset.from_golden` already supports it (dataset.py:57) and its docstring calls
out exactly this cross-card collision; falling back to the full set only when the live card has no goldens. Then
re-tune this one shape (`emmy tune --kernel square.512.dynM`, no `--clean`) and re-A/B.

## Finding 3 — `attention.hd128` recorded golden is silently unreproducible; static flash lost its w2x1 form

Greedy 19.9 µs vs **recorded** 16.5 µs → 1.21× — but the live A/B shows ratio 1.00 because the `golden` row did
not run the recorded config: its realized knobs (`TILE@dd=TILE@pj=w4x1/f1x16/k8`, `d1/tma`, regs 255) are
identical to the greedy pick, not the YAML's (`w2x1/f1x8/k8` / `w2x1/f1x16/k4`, `d2/tma/ring`). A direct probe
(`run --bench --ab "PLACE=fuse,TILE@dd=…w2x1/f1x8/k8,TILE@pj=…w2x1/f1x16/k4,STAGE=d2/tma/ring"`) also realized
`w4x1/f1x16/k8 + d1/tma` — **the pin is silently dropped/re-derived**. Meanwhile `attention.hd64`'s per-axis pin
applies exactly (golden row = recorded knobs), so this is specific to the hd128 static form. Strongest evidence
it's a planner regression, not noise: the **dynamic** twin still maps `w2x1` and runs **16.5 µs — faster than the
19.9 µs static** on the same head count/seq, an inversion that should be impossible (the masked kernel does strictly
more work). The recorded 16.5 config predates the flash-form-fork / warp-move-grid rework (#300/#302); the w2x1
static hd128 form appears to have dropped out of the current warp move grid. `eval analytic`/`variants` still can't
enumerate attention, so the A/B + probe are the only instruments. Fork regret for the attention contractions
(`matmul free=128 red=128` row): 1.04x — the search steered fine within the forms it has.

**Recommendation (high):** (a) make an unmappable `--ab`/`--golden`/`EMMY_KNOBS` pin a hard error or at minimum a
loud `pin dropped: <knob>` warning — a silent no-op turns the golden A/B into greedy-vs-greedy and *hides*
regressions exactly like this one; (b) check whether the w2x1 static hd128 flash form was intentionally retired by
the warp move grid; if not, restore it (a ~17% deployable regression on this shape), and if yes, re-record the
golden to the reachable 19.9 µs so the YAML stops claiming 16.5. Left untouched this sweep pending (b).

## Finding 4 — `o_proj.h4096` static: same censoring signature as mlp_down (unsplit cp-staged pick vs k4+tma golden)

Greedy 123.3 µs vs golden 104.2 µs → **1.18×** (vs cuBLAS 1.29 — the worst matmul parity of the sweep next to
mlp_down). Greedy picked unsplit `w1x8/f4x2 + d4/cp/ring`; the golden is `w8x2/f2x4/k4 + d2/tma/ring`.

- `eval analytic --kernel o_proj` — golden rank **0/14805** (top-1 cold; dynM twin 389). Refit weights price it.
- `eval prior --dataset golden` — learned rank **4558/14805**, `vs gold` 1.24×. Buried again.
- `eval variants` (`k_matmul_e24882`) — all 37 main + 6 partial measured variants are `d4/cp/ring` (or bare
  `d1/cp`); the `d2/tma/ring` k4 golden was **never measured**. Deployed pick rank 2/37 by -O1, and its -O3
  (126.0) is best among measured — within its horizon the deploy was right; the tma+k4 subtree was never opened.
- Fork regret (`free=4096 red=4096` row): REDUCE 1.00x, STAGE 1.02x, **TILE 1.58x** (8 forks) — consistent with
  the tma-staged wide-warp tile living under a TILE branch the prior down-scores.
- Contrast with the twin that *worked*: `o_proj.h4096.dynM` greedy found `w2x4/f2x2/k4 + g2k + d2/tma/ring`
  (rank 7/55 -O1, best -O3 100.5 among measured) and **beat** its golden — the masked tier's `_W_A_DYN` steering
  reached tma+k4 where the static tier didn't.

**Recommendation (high, same lever as Finding 1):** the static-tier censoring fix (analytic-top-k force-bench or
node-store CatBoost retrain) covers this shape too; treat mlp_down + o_proj as the acceptance pair for it. A
`D_*` feature is *not* the ask this time — the analytic already ranks both goldens #1.

## Notes on the near-misses and other observations

- **`matmul.square.512` (fp32)** — greedy (`n16x8/f4x4`, unsplit) benches 8.2 vs the recorded-config row at 8.6
  (7.6 + 1.0 g2k) — 4.7%, sub-noise, left alone. But both are far below the recorded `emmy_us: 10.24` — codegen got
  ~16% faster on this shape since the last record; worth lowering on the next same-knobs win.
- **`matmul.qkv.h4096.dynM`** — greedy 268.0 vs golden 259.6 (1.03) — inside the noise band, left.
- **Memory-bound kinds are fully converged:** all 20 softmax/reduce/rms_norm/pointwise shapes (static + dynM)
  reproduce their goldens with identical knobs — the #317 fold-ladder wins and the prev sweep's reduce re-records
  all hold. Prev sweep's Finding 3 (softmax.k2048 b256-vs-b512) is **resolved**: greedy now deploys `b512` at
  parity (3.8 µs, 3/3).
- **`attention.hd128.dynM` holds its #302-era staged form** (16.5 µs, 0.90 vs SDPA) — the dynamic-flash staging
  work survived the reduce-featurizer changes.
- The 4 golden replaces re-verified post-edit on the box: `pytest tests/compiler/test_golden_configs.py` 20/20
  passed, and each new golden row re-benches at the greedy config/latency.

## Artifacts (local `_tune/golden-sweep-rtx5090-refit/`, verified)

- `tune.log` (39/39 shapes), 36 `ab-<shape>.log` + 10 `ab-*.rerun{2,3}.log` + 2 pin-probe logs + 4 `ab-*.post.log`
  re-verifications, `eval-golden.log`, `eval-prior-golden.log`, `eval-prior-nodes.log`, `eval-analytic-*.log`,
  `eval-variants-k_matmul.log`.
- **`autotune.snapshot.db`** — 676,413,440 bytes; `SELECT COUNT(*), MAX(feat_ver) FROM node` → **8429, 2**; all
  node rows GPU-keyed `NVIDIA GeForce RTX 5090`; perf 1704 rows, cuda_op 1665 rows.
- **`prior.json`** — 23,364,070 bytes; parses; `feat_ver` 2; CatBoost model blob present; **dataset reservoir
  26,483 rows** (`seen` 26483, `since_fit` 0); calibration 0.871. These two files seed the analytic-prior rework's
  Phase-3 snapshot.

## Workflow notes

Status of the #318-era report's notes, then this sweep's new friction:

- **`eval variants --kernel <shape-name>` still matches only DB kernel-hash names** (NOT fixed): `--kernel
  mlp_down` → "No measured variants matching"; I fell back to `--kernel k_matmul` and grepped the per-hash blocks
  by hand. Same one-key ask as last sweep: let `--kernel` accept the golden shape name across all eval views.
- **`eval analytic` / `eval variants` still matmul-only** (NOT fixed): Finding 3 (attention) again has no
  rank/reachability leg — the silent-pin regression had to be established with two hand-built `--ab` probes.
- **Golden e2e-vs-kernel-sum** (PARTIALLY fixed): `run --bench` now prints a `whole-program (e2e)` row for split-K
  greedy picks — good — but golden rows are still kernel-sum only, so the split-vs-unsplit judgment call remains
  (hit it on `square.512.fp16`: kernel-sum 4.2 vs e2e 5.2–6.1 across runs).
- **`--repeat N` on `run --bench --golden`** (NOT implemented): step 4 again cost 10 manual re-runs (5 shapes × 2).
  All five candidates were stable (±0.1 µs), so a `--repeat 3` printing min/median would have folded step 4 away.
- **`! golden` FLOP-sanity wording** (NOT fixed): the sub-µs `reduce.*.dynM` rows still print
  `impossible: implies 693 TFLOP/s > 105 device peak`.
- **`run --bench --json PATH` exists now** and its help text says it "retires ad-hoc table parsing in the
  golden-sweep workflow" — but this skill run still parsed tables with regexes because the skill text doesn't
  mention it. *Fix the skill:* use `--json` for step 2/3 categorization (it also carries the integrity flags).
- **NEW — silent knob-pin drop** (Finding 3): an unmappable pin re-derives to the greedy config with no
  diagnostic, making `--golden`/`--ab` A/Bs silently vacuous. This is the most dangerous tooling gap this sweep:
  it converts a regression into a "same, same knobs" row. Error or warn.
- **NEW — cross-card golden dataset union** (Finding 2): `tune --dataset golden` tunes other cards' shape variants
  (39 targets for a 36-entry YAML) and can shadow the live card's dtype. One-line fix (`live_gpu=True`).
- **NEW — schema vs greedy-pick asymmetry for dynamic attention:** the tuner's greedy pick for `hd64.dynM` is
  axis-keyed (`TILE@dd`/`TILE@pj`), but the golden schema (correctly) rejects axis-keyed knobs for dynamic
  attention; finding the bare-TILE equivalent required a probe (`--ab "TILE=…f1x16/k4"` → pj re-derives to the
  greedy's `f1x8/k8`). The A/B table (or `--json`) should print the canonical bare-pin form for masked-flash rows.
- **NEW — no per-shape timestamps in tune.log:** pace/ETA for a 3.6 h remote sweep had to be reconstructed by
  polling; a wall-clock prefix on the `[tune] === N/M` lines would make remote monitoring trivial.
- **Timing:** tune 3.6 h (vs the skill's "~2.5–3.5 h on this card" — the 3 extra cross-card fp16 squares from
  Finding 2 account for most of the overshoot: `square.4096.fp16` 1080 s + `square.1024.fp16` 819 s +
  `square.2048.fp16` ~500 s ≈ 40 min); A/Bs 7.5 min; evals ~4 min; re-runs + probes ~8 min.
