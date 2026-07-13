# Gemma-4-12B layer-0 tune findings — RTX 4090 (sm_89), dynamic

> ## CURRENT STATE — 2026-07-11 re-run on `main` `cbc9805d` (#347; post-#342 o_proj fix + #347 analytic-prior rework)
>
> The detailed analysis further below is the **pre-#342 (#338) state**; its headline o_proj scalar misdeploy (48%
> of the layer) is **FIXED by #342**. A fresh full `emmy tune google/gemma-4-12B --layer 0 --dynamic seq_len@x:1
> --clean --bench` on a rented 4090 (autotune 2 751 s, torn down after) gives:
>
> **E2E** (benched at the seq_len=512 symbolic hint): Eager **1 774 µs** / torch.compile 1 553 / **Emmy 4 782 µs =
> 0.37×** (2.7× behind). Down from **pre-#342's 8.07 ms** (the o_proj fix) *and* from the **#339 run's 5.41 ms** —
> #347's new analytic prior improved the non-tier-locked deploy picks by ~12% (the tier-locked megakernel is
> unchanged, so the gain is elsewhere).
>
> | Backend | µs | vs eager |
> | --- | --: | --: |
> | Eager PyTorch | 1 774 | 1.00× |
> | torch.compile | 1 553 | 1.14× |
> | **Emmy** | **4 782** | **0.37×** |
>
> **Per-kernel** (-O3 reproducer): `k_linear_mean_reduce` **2 283 µs ≈ 48%** (fused RMSNorm→gate_up MLP megakernel,
> no eager ref) · `k_linear_reduce` 844 (down_proj, 0.45×) · `k_mean_linear_reduce` 833 (0.76×) ·
> `k_linear_sdpa_reduce` 345 (0.44×) · `k_scaled_dot_product_attention_reduce` 60 (0.71×) · pointwise/mean/slice
> 2–6 µs (14–32× wins). bench_fails: 6 "exceeded 2.0s" + 5 hung + 1 compile-budget.
>
> **The bottleneck is now the two compiler TIER LOCKOUTS (architectural — unchanged by #342/#347, and identical to
> the 4080 gemma report's Findings 1–2):**
> - **Fused gate_up megakernel locked to `d1/sync`** — **64/65** measured configs are `d1/sync`; cp.async
>   pipelining is refused because the matmul's A-operand is the on-chip-computed normalized RMSNorm row. tune-DB
>   -O3 2 141 ≈ reproducer 2 283 (real). #347's prior can't touch this — it's a lowering gate, not a ranking.
> - **down_proj on a smem-LESS schedule** — `k_linear_reduce_f1b366` deploys `w2x2/f4x4`, **blank STAGE / 0 smem**,
>   -O3 1 192 µs (0.45× eager); no tiling → no cp.async. The q/k projections deploy `d1/sync`.
>
> **4080 vs 4090:** same tier lockouts (architectural, both sm_89), ~1.6× faster absolute (megakernel 2 283 vs the
> 4080's 7 582 µs — though the 4080 total was also bench-artifact-inflated). The **degenerate-fast tune-DB rows
> reproduce** here (`k_mean_linear_reduce` tune-DB **3–6 µs** vs reproducer **202–833 µs**) — systemic on both
> cards — but the layer-bench **inflation is milder** on the 4090 (reproducer `k_mean_linear_reduce` 833 ≈ eager
> 633, vs the 4080's 2 189 = 2× eager).
>
> **Fix priorities (unchanged, help both cards):** (1) per-operand cp.async transport for the fused megakernel —
> stage the plain-global weight B-operand even when the A-operand is a computed prologue; (2) the smem-tiled tier
> for the `k_linear_reduce` K-reduction projections. Data: `_tune/gemma-4090-347/logs/tune.log`.

- **Status: emmy is 4.4× behind eager on this layer (8.07 ms vs 1.83 ms), and 48% of that is ONE misdeploy** —
  the o_proj matmul ships a scalar `n32x8/f4x14` tile at 3 625 µs while the tune DB holds measured -O3 rows for
  the same kernel at **225–227 µs** (16×). The warp tier is *unreachable at deploy-time lowering* for this
  terminal (root-caused below, with spy transcripts); fixing it alone brings the layer to ~4.1 ms (≈2.2× eager,
  in line with the 5090's 2.13×). The #337 masked-flash work is validated on sm_89: the flash kernel runs at
  SDPA parity (48.6 µs vs eager 44).
- **Command** (on a rented billing-exempt CloudRift RTX 4090, repo @ `3087433a` = #338):
  `emmy tune google/gemma-4-12B --layer 0 --dynamic seq_len@x:1 --clean --bench --dump-dir <dir>/dump` with
  `EMMY_TUNE_DB` / `EMMY_PRIOR_FILE` / `EMMY_CUBIN_CACHE` pointed into the run dir (the qwen3-notes isolation
  pattern — worked cleanly again). Model is the ungated **base** `google/gemma-4-12B` (earlier campaigns used
  `-it`; identical layer architecture). Date: 2026-07-09.
- **Scope**: single layer 0, **dynamic** (symbolic `seq_len`, masked tiles — the deployable artifact), benched at
  the `DEFAULT_SEQ_HINT=512` hint; torch rows tiled to the same hint.
- **Run stats**: tune wall 4 013.6 s (~67 min) + the -O3 re-bench pass (~8 min); DB **477 ok / 16 bench_fail**;
  prior trained on 1 204 benches (18 warmup / 1 186 post), reservoir 9 732 rows, calibration Spearman **+0.76**,
  post-warmup silly rate 98%.
- Numbers below are the `--bench` **-O3** re-bench (deployable, CUDA-graph captured) or the `run --bench`
  launch-order table (also -O3). Tune-DB latencies quoted for ranking context are **-O1** and are marked as such.
  Run-to-run e2e variance was ~7% (tune-bench 8 069 µs vs a later `run --bench` 7 523 µs on the same artifacts).

## E2E — full layer forward (eager / torch.compile / emmy), µs

| Backend | µs | vs eager |
|---|--:|--:|
| Eager PyTorch | 1 833 | 1.00× |
| torch.compile | 1 581 | 1.16× |
| **Emmy** | **8 069** | **0.23× (4.40× behind)** |

## Per-kernel, in-layer launch-order table (ground truth)

From `emmy run <dump>/00_input.json --bench` — per-launch attribution (post-#324), e2e 7 523 µs, Σ 7 514 µs.
This is the table to trust; the tune `--bench` reproducer table below misattributes the split pair (finding 6).

| Kernel | Layer op | µs | % | deployed config |
|---|---|--:|--:|---|
| `k_linear_sdpa_reduce_ad75ba__partial` | **o_proj matmul** (linear_3 7/8 + sdpa 1/17 + reshape/transpose) | **3 625** | 48.2% | **scalar `n32x8/f4x14`, `g2k`** |
| `k_linear_mean_reduce_02ef41` | **pre-FFW norm → gate⊗up → GeGLU** (linear_4+5 8/8 full, gelu 13/13) | **2 323** | 30.9% | mma `w4x2/f4x4/k8`, `d1/sync`, cone=fuse, 97K smem |
| `k_linear_reduce_f1b366__partial` | down_proj matmul (linear_6 7/8) | 768 | 10.2% | mma `w2x2/f4x8/k2`, `g2k` |
| `k_linear_reduce_eb26a6` | q_proj matmul (linear 7/8) | 314 | 4.2% | mma `w2x4/f4x2/k2`, `d1/sync` |
| `k_linear_reduce_cdc109` ×2 | k_proj / v_proj matmul (linear_1 7/8, deduped kernel) | 186 ×2 | 5.0% | mma `w4x2/f2x4/k8`, `d1/sync` |
| `k_scaled_dot_product_attention_reduce` | **flash** (sdpa 16/17) | 48.6 | 0.6% | mma `@dd w4x1/f1x2/k16`, `d2/cp/ring` |
| `k_mean_*` / `k_mean_linear_reduce_*` ×5 | RMSNorm stats + projection epilogue slices | 3.5–6.7 | 0.4% | scalar `b32–b256` |
| RoPE / GeGLU pointwise ×5, 2 copy epilogues | slices, cat, mul; `__partial` copy-backs | 2.8–6.1 | 0.5% | flat / `f2` |

The two fused matmul kernels are 79% of the layer; with down_proj, 89%.

## Per-kernel -O3 reproducer table (tune `--bench`) — with the misattribution caveat

| Kernel | eager | tcompile | emmy | note |
|---|--:|--:|--:|---|
| `k_linear_sdpa_reduce` | 154 | 152 | 3 695 | reproducer re-lowers the full op — reproduces the misdeploy; in-layer the cost sits on the `__partial`, not this name |
| `k_linear_mean_reduce` | – | – | 2 493 | full coverage; ≈ in-layer 2 323 |
| `k_mean_linear_reduce` ×5 | 334–661 | 120–403 | 203–847 | **inflated**: these kernels run 3–6 µs in-layer (linear 1/8 slices; reproducer re-lowers the whole fused op) |
| `k_linear_reduce` ×3 | 394 | 394 | 200–844 | partial-coverage inflation (linear 7/8) |
| `k_scaled_dot_product_attention_reduce` | 44 | 43 | 47 | ≈ full coverage — trustworthy, at parity |
| `k_mean`, RoPE/pointwise ×5 | 50–199 | 3–13 | 2–6 | emmy wins 15–32× vs eager |

## Finding 1 — o_proj deploys a scalar tile 16× slower than its own measured best (≈3.4 ms at stake)

**Symptom**: `k_linear_sdpa_reduce_ad75ba__partial` (o_proj: seq×4096 @ 4096×3840, A = flash output) deploys
scalar `n32x8/f4x14`+`g2k` at **3 625 µs** (48% of the layer). The tune DB for this very kernel name holds 28
fused-form rows (-O3 best **227.1 µs**) and 6 split-form rows (-O3 best **225.2 µs**), all `a:mma_*` on the same
axis key (`TILE@a2`). `eval variants` marks the mma row as the pick (◄) — yet the deployed program runs scalar.

**Root cause (proven by instrumented compiles)**: at deploy-time lowering the warp tier is never *offered* for
this terminal:

- A `flatten_leaves` spy shows the terminal's fork = **370 leaves, all scalar** (`n16x16…n64x16` × `f*` ×
  `b*/g*/r*`), zero `a:mma` rows; the evidence join (`greedy.py::_db_measured_pick`, the #326 hierarchy) finds
  the kernel's signature group but **no measured row prefix-matches any scalar candidate** (all measured rows are
  mma) → returns None → the model extrapolates → scalar.
- `_warp_atoms` refuses because the contraction's **A `Load` reads f32** at enumeration
  (`emmy/compiler/pipeline/passes/lowering/tile/_schedule.py:330-338`); the mixed-dtype rescue `_demote_mixed_a`
  ALSO refuses because **the fold's B `Load` reads f32 too** (`_schedule.py:363-376`, spied
  `bdtypes=['f32']`) — demotion requires genuinely-16-bit folds.
- The torch-level graph is **entirely f16** (reproducer JSON: sdpa out f16, weight f16) — the f32 view is the
  flash tail's internal pre-demotion intermediate (the deployed program's realized boundary tensor IS `__half`,
  per the emitted kernel signature).
- The asymmetry: the tune measured 34 mma rows on this fork family, so tune-time enumeration DID offer warp.
  Leading hypothesis: the tune's inner per-kernel loop enumerates from the **realized kernel-boundary slice**
  (A stamped f16) while deploy re-lowers from the graph view (A = f32 intermediate). Distinguishing diagnostic:
  instrument the two_level inner enumeration vs a deploy compile for this terminal and diff `_tile_rows`' output.
- **Pins cannot reach it either**: an axis-keyed `--ab "TILE@a2=a:mma_…"` silently no-ops (the ab rows deploy the
  identical scalar config, no red diff) — the same silent-pin-degrade family as findings-5 F2 / #337's dynM STAGE
  drop. An un-keyed `TILE=` pin leaks into the flash kernel's form narrowing and degrades it (10 s bench cap).

**Fix directions** (priority 1): make the graph-view enumeration see the fold's post-demotion dtype (or extend
`_demoted_atoms` to accept an f32 B whose producer store is a 16-bit demote); and/or re-enumerate consumer forks
after the producer's output dtype resolves; independently, deploy should warn when a kernel's candidate set is
**disjoint** from all of its measured evidence (that condition is exactly "the tune measured a tier the deploy
cannot build"), and a pin that matches no offered row must fail loudly instead of no-opping.

**Repro**: `emmy run --ir <dump>/08_lowering_cuda.kernels/k_linear_sdpa_reduce_ad75ba.torch.json --bench`
(reproduces the scalar deploy, 3 680 µs e2e); spy scripts in `_tune/tune-model-gemma4-12b-l0-4090/spies/`.

## Finding 2 — the gate⊗up GeGLU fused edge is smem-bank-conflict-bound (2 323 µs, 31%)

The fused pre-FFW-norm → gate⊗up → GeGLU kernel (`k_linear_mean_reduce_02ef41`, both 15360×3840 linears as
⊗-fold channels + gelu, 120.8 GFLOP → 52 TFLOP/s) deploys its **measured-best** config (pick = -O3 best 2 342.9
in a 61-row leaderboard — the search did its job; -O1 rank 18 is the usual inversion). Root-NCU counters
(`sudo` needed on CloudRift, `ERR_NVGPUCTRPERM` otherwise):

| metric | value | read |
|---|--:|---|
| occupancy | 16.7% | 97K smem/CTA → 1 CTA/SM |
| SM throughput | 23.3% | stalled |
| **ld bank conflicts** | **206 M** | vs **92 M** LSU instructions — >2 conflicts/load |
| st bank conflicts | 48 M | same layout problem on the slab store |

Class 3 codegen: the `d1/sync` computed-A compute-fill's slab layout conflicts pathologically. There are no
`PAD_SMEM` / lane-permute knobs in the codebase to A/B — this is a codegen work item (pad or swizzle the A slab),
priority 2 (~1.5 ms upside if it reaches the eager gate+up share of ~740 µs).

## Finding 3 — down_proj at 1.6× its eager share (773 µs pair, 10%)

`k_linear_reduce_f1b366__partial` deploys ≈ its measured best (768 µs in-layer vs -O3 834.5 in `eval variants`;
the "misses best" flag is the -O1 inversion false-positive again — the -O1 rank-1's -O3 twin is 1 219 µs,
*worse*). Same class as the known "sm_89 mma schedule ~2× off cuBLAS" gap (golden-sweep findings). Priority 3.

## Finding 4 — flash at SDPA parity on sm_89 dynM (positive), but its search pool is dirty

Deployed flash (`TILE@dd w4x1/f1x2/k16`, `STAGE d2/cp/ring`) runs **48.6 µs vs eager/tcompile 44/43** — #337's
cp.async K/V staging works on this 16-head hd256 sliding-window layer. Two hygiene items on the same kernel:

- **13 of the run's 16 bench_fail rows** are its search variants: 6 hung-kernel (1 s cap), 3 "2 s GPU time",
  2 bench-worker 16 s SIGKILLs, 2 nvcc >12 s compile-budget (both `k16` full-K mma tiles). Wasted slots (class 4).
- The DB pool poisons the **raw** prior argmax: `eval prior --dataset db` reports pick 207 319 µs (938× of best)
  for this op — a hung-adjacent variant recorded `ok` at 0.2 s. The deployed pick dodges it (48.6 µs), but the
  two eval views disagree about "the pick" (`eval variants`: rank 37 @ -O1 511 µs; `eval prior`: the 207 ms row) —
  confusing during triage, and `bench_fail`-adjacent `ok` rows still have no purge path (recurring since
  findings-3).

## Finding 5 — everything else is healthy

q_proj rides the demoted computed-A cone at mma (314 µs, `d1/sync` — the #325 mixed-dtype path working for plain
projections); k/v dedup into one kernel launched twice (186 µs each); the RMSNorm stats, RoPE slices, GeGLU
pointwise and epilogue slices all run 2–7 µs in-layer, beating eager 15–32×. The five `k_mean_linear_reduce`
reproducer rows (203–847 µs) are re-lower inflation, not in-layer cost — see finding 6.

## Finding 6 — the tune `--bench` per-kernel table misattributes split pairs (tooling)

The reproducer table put 3 695 µs on `k_linear_sdpa_reduce_ad75ba` — in-layer that name is a **5 µs copy
epilogue**; the cost sits on `…__partial`. Reproducer rows for the 1/8-slice kernels are inflated 40–200×
(re-lowered full ops), and Σ(reproducer) ≈ 9.9 ms > e2e 8.07 ms. The `run --bench` launch-order table (post-#324)
is the antidote and should be printed by `tune --bench` too (or the reproducer table should carry the coverage
fraction + in-layer µs — the exact ask from the qwen3 findings' workflow notes, still open).

## Repro / artifacts

Local: `_tune/tune-model-gemma4-12b-l0-4090/` — `tune.log`, `dump/` (66 artifacts incl.
`62_kernel_bench.json`, `kernels.html`, per-kernel `.torch.json` reproducers), `autotune.db` (337 MB),
`prior.json` (8.7 MB). The rented 4090 is deleted.

```bash
# ground-truth in-layer table (re-lowers greedily from the run's prior + DB):
EMMY_TUNE_DB=_tune/tune-model-gemma4-12b-l0-4090/autotune.db \
EMMY_PRIOR_FILE=_tune/tune-model-gemma4-12b-l0-4090/prior.json \
  emmy run _tune/tune-model-gemma4-12b-l0-4090/dump/00_input.json --bench
# finding-1 leaderboard vs deployed reality:
  emmy eval variants --kernel linear_sdpa_reduce --db _tune/tune-model-gemma4-12b-l0-4090/autotune.db
```

## Follow-up (same day) — flash kernel optimization pass (local RTX 4080, same sm_89)

Asked to close the flash gap, a config sweep on the pulled reproducer settled it: **the deployed config is the
-O3 optimum of the reachable knob space.** Variants tried (all ≥ baseline ~75 µs vs SDPA ~52 on the 4080): STAGE
`d1/cp`/`d2/cp/ring`/`d3`/`±p2`, `FAST_EXP=True`, `INTERLEAVE_LOADS=False`, the 4090 DB's -O1-rank-1 geometry
(`f1x8/k16`+`f1x32/k4`: 66K smem, 8% occ, **103 µs** — the -O1 inversion again), `f1x4/k2` (82.5), and `w2x1`
64-thread forms (94–101; halving the CTA doubles K/V re-streaming). The residual gap (1.10× on the 4090, 1.45×
here — the 4080 also pays wave quantization: 128 CTAs over 76 SMs = 1.68 waves) is **codegen-level: 243
regs/thread → 17% occupancy** on a streaming kernel. That register-pressure item is the remaining engineering
lead; NCU stall attribution needs root (no passwordless sudo locally).

Actionable outcome — **`attention.hd256.dynM` golden seeded for the 4080** (`goldens/rtx4080_sm89.yaml`, its
first attention entry): cold greedy on this unseeded shape doesn't just misdeploy, it picks a **>10 s hang** (the
same hazard class as the run's 13 sdpa bench_fails, reproduced on a second card). Bare-TILE probe (the
findings-4 F3 manual flow): `TILE=a:mma_m16n8k16_f16/w4x1/f1x2/k16` + `STAGE=d2/cp/ring` reproduces the
axis-keyed optimum exactly (the pj contraction resolves its own `f1x32`); measured 68.4 µs vs torch SDPA 49.0
(3-pass stable), replay through the golden plumbing 68.3 µs with clean integrity flags. **The 4090 entry is
seeded too** (second rented box): 44.1 µs vs torch SDPA 42.0 — **0.94×, at parity** — 3-pass stable, `/p2` within
noise (44.3), replay validated 44.3 µs with clean flags. Both cards record the same knob spelling.

New tooling findings from this pass:

- **The flash dispatch consults only the bare `STAGE` knob** (`_schedule.py:1293` reads `STAGE.raw()`) — an
  axis-keyed `STAGE@kv=` pin is silently ignored (third silent-pin-degrade instance this report).
- **`run --golden` cannot validate a golden whose greedy pick hangs**: the greedy row benches first and its
  10 s bench_fail aborts the whole run — the exact shape a golden exists to fix blocks its own validation.
  The golden/ab bench should survive a greedy-row bench_fail and still report the pinned rows.
- The 4090 layer deployed `d2/cp/ring` while the DB pick spelled `d2/cp/ring/p2` — realized ≠ picked again
  (perf-neutral here: p2 measured within noise on the 4080), and the unstaged mma form NaN'd once in two
  otherwise-identical local runs — the non-staged dynM paths carry a latent correctness/latency hazard.
- The accuracy-check probes spam `STAGE pin does not resolve (static kv)` warnings on every pinned dynamic run
  (the static-shape probe re-resolves the pin and declines) — noise that buries real warnings.

## Workflow notes

Held up from previous reports' notes: the **cache-isolation pattern** (three `EMMY_*` vars into the run dir) and
the **launch-order `run --bench` table** (#324) — the latter overturned the headline attribution in one command
and was the single most valuable view this run. `eval variants`' **-O3 column** again decided every "misses best"
call (all four flags this run were -O1 inversions; the ask to rank on -O3 stands).

New friction, with proposals:

- **No fork-offer debug view.** Root-causing finding 1 took four hand-written monkeypatch spies (flatten_leaves,
  `_warp_atoms`, `_demote_mixed_a`, `_tile_rows`) run via a wrapper around `emmy.emmy.main`. Propose
  `emmy eval offer --kernel <substr>` (or an `EMMY_DEBUG` fork dump): per fork, the candidate TILE/REDUCE/STAGE
  sets, each refusal gate's verdict, and which evidence tier decided — that view would have made finding 1 a
  15-minute diagnosis instead of ~2 hours.
- **Pins are unverifiable in multi-kernel programs.** Un-keyed `--ab "TILE=…"` leaks onto every kernel (degraded
  the flash to a >10 s bench_fail); axis-keyed `TILE@a2=…` silently no-ops when the fork lacks the family. Both
  cost a GPU round-trip to discover. Propose: `--ab` verifies each pin bound to ≥1 kernel's fork and marks the
  row `pin ignored` (or fails) otherwise.
- **`eval prior --dataset db` and `eval variants` disagree on "the pick"** (207 ms vs 511 µs for the flash op) —
  one is the raw model argmax over the pool, the other the deploy-narrowed pick. Label them differently or make
  both print the deploy-narrowed one.
- **NCU on CloudRift boxes needs root** (`ERR_NVGPUCTRPERM` as riftuser): `sudo env …` worked; worth one line in
  the skill. Also `ncu compare`'s reference side is empty when the reproducer has no torch closure (the
  erased-cast `float != c10::Half` closures) — propose reproducers emit a cast-restored eager closure.
- **`eval prior --dataset nodes --kernel` filters by op label, not kernel name** (first query returned 0 nodes;
  the hint text saved it). Accepting kernel-name substrings there would remove a dead end.
- Minor: the tune progress bar is TTY-gated so a `nohup` log is silent for the whole tune phase (monitoring fell
  back to process/DB checks); `kernels.html` PNG export skipped on the box (no playwright chromium) — harmless.
- **No flakiness**: single tune run, exit 0; ~7% e2e run-to-run swing (8 069 → 7 523 µs) — the "re-run before
  reporting" guidance stands.
