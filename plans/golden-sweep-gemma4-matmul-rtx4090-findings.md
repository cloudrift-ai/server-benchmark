# Gemma-4 matmul golden sweep — RTX 4090 (sm_89), narrowed to the cuBLAS laggards

- **Date / box**: 2026-07-17, RTX 4090 24 GB (driver 580.65.06, CUDA 12.9), fresh `vm-perf-tuning` VM, repo at
  `f95e9783`.
- **Scope**: NOT the full golden set — per request, only Gemma-4 matmul shapes, and among those only the 7 whose
  recorded standard-lane `emmy_us / cublas_us` was worst (≥1.14×): `k_proj_global` {static, `.dynM`, `.s2048`},
  `kv_proj` {static, `.dynM`}, `q_proj_global.dynM`, `mlp_gate_up.s2048`. Everything else in
  `rtx4090_sm89_gemma4.yaml` was ≤1.10× and left untouched.
- **Sweep command**: sequential `EMMY_FAST_MATH=1 emmy tune --golden <name>` per shape (small → large), `--clean` on
  the first invocation only — one cold sweep with within-run prior/DB transfer, just narrowed. A per-shape runner was
  needed because `--kernel` takes a single name substring and the picked set shares none. Fast-math umbrella on
  (consumer die), so one enumeration measured both lanes.
- **Wall time**: tune 2 h 51 m (per shape 5–36 min); A/B pass (7 shapes × 2 lanes) ~55 min; confirmation re-runs
  ~10 min.
- **Tally (entries)**: **2 replaced** (both lanes of `k_proj_global.s2048`) / 0 added / 12 unchanged (greedy
  reproduced the recorded knobs exactly) / **0 worse**.

## Fork sibling regret

The two halves of `emmy eval online --dataset nodes` diagnose different failures: the **offline prior** is the
cold-start ranking that decides what a cold sweep measures at all (a big regret here = the sweep wastes benches
being steered wrong, and the online model never sees the censored region); the **online prior** is the CatBoost
this sweep trained (a big regret here with a clean offline = training/calibration problem). This card's `-O1`
block, 3863 nodes over the 7 shapes (only TILE and STAGE produced multi-child forks in this narrowed sweep):

| metric (-O1, 30 forks)                        | offline prior | online prior (CatBoost) |
| --- | --- | --- |
| TILE fork regret (median)                     | 4.88× | 1.46× |
| STAGE fork regret (median)                    | 1.00× | 1.00× |
| worst TILE fork — matmul free=2048 red=3840   | 6.29× | 1.61× |
| 2nd worst — matmul free=8192 red=3840         | 4.54× | 1.45× |
| 3rd worst — matmul free=512 red=3840          | 4.40× | 1.57× |
| leaf reachability (mean / median / worst)     | 1.47× / 1.43× / 2.02× | 1.29× / 1.35× / 1.47× |
| leaf calibration (median per-op Spearman)     | +0.46 | +0.90 |

(No roofline-impossible baselines: the worst-row "best" values, e.g. 81.75 µs for free=2048 red=3840, sit at
~97 TFLOP/s — well under the 4090's fp16 peak.)

Diagnosis: the miss is squarely in the **offline half's TILE weights** — a cold search on these small-N / split-K
Gemma shapes gets steered ~5× off at the TILE fork and only recovers by brute-force exploration (the sweep still
converged: it benched 73–182 configs per shape). The online half, trained on this sweep's fresh data, prices the
same forks at 1.46× with +0.90 calibration — so this is a cold-start weights/features problem
(`scripts/golden_knob_heuristics.py` refit), not a trained-model problem. Note the online numbers inherit the
offline prior's censoring: they only cover regions the cold ranking let the sweep visit.

## Per-shape outcomes

All emmy / golden numbers are pinned-comparable **-O3** rows from the same `run --bench --golden` run (greedy
`isolated` block vs pinned golden rows; split-K totals = partial + finalize). `cuBLAS µs` is the shape's recorded
`cublas_us` (live eager rows agreed within noise: 17 / 50 / 199 / 2907 on the four distinct shapes). Ratio >1.0 in
`vs cuBLAS` = emmy slower than PyTorch/cuBLAS.

### Standard lane (fp32-accumulate — what a default compile deploys)

| shape | greedy µs | best-golden µs | greedy/golden | cuBLAS µs | vs cuBLAS | category |
| --- | --: | --: | --: | --: | --: | --- |
| `k_proj_global`          | 20.0 | 20.1 | 1.00 | 16.6 | 1.20 | same knobs |
| `k_proj_global.dynM`     | 20.2 | 20.1 | 1.00 | 16.6 | 1.22 | same knobs |
| `k_proj_global.s2048`    | 58.0 | 67.2 | **0.86** | 48.7 | 1.19 | **replaced** (was 1.38 vs cuBLAS) |
| `kv_proj`                | 58.2 | 58.3 | 1.00 | 49.0 | 1.19 | same knobs |
| `kv_proj.dynM`           | 59.5 | 59.6 | 1.00 | 49.0 | 1.21 | same knobs |
| `q_proj_global.dynM`     | 226.1 | 226.5 | 1.00 | 195.4 | 1.16 | same knobs |
| `mlp_gate_up.s2048`      | 4346.9 | 4348.9 | 1.00 | 2945.0 | 1.48 | same knobs |

### Fast-math lane (f16-accumulate `TILE` atom; entries kept beside the standard ones, never replacing them)

| shape | greedy µs | best-golden µs | greedy/golden | cuBLAS µs | vs cuBLAS | category |
| --- | --: | --: | --: | --: | --: | --- |
| `k_proj_global`          | 18.3 | 18.3 | 1.00 | 16.6 | 1.10 | same knobs |
| `k_proj_global.dynM`     | 18.7 | 18.6 | 1.00 | 16.6 | 1.13 | same knobs |
| `k_proj_global.s2048`    | 48.1 | 52.1 | **0.92** | 48.7 | 0.99 | **replaced** |
| `kv_proj`                | 48.3 | 48.2 | 1.00 | 49.0 | 0.99 | same knobs |
| `kv_proj.dynM`           | 48.2 | 48.3 | 1.00 | 49.0 | 0.98 | same knobs |
| `q_proj_global.dynM`     | 147.0 | 148.2 | 1.00 | 195.4 | 0.75 | same knobs |
| `mlp_gate_up.s2048`      | 2342.9 | 2340.9 | 1.00 | 2945.0 | 0.80 | same knobs |

The two `k_proj_global.s2048` wins reproduced with <1% spread across three independent runs (A/B + 2 confirmation
runs; old-golden rebenches 67.0–67.2 and 52.0–52.2 µs) — clearly outside noise despite sitting below the 10–13%
band the full-set sweeps see on small shapes. Recorded from `--json` `record_knobs`:

- std: `TILE a:mma_m16n8k16_f16_f32/w2x2/f4x4/k2, REDUCE g2k` (was `…/w2x2/f2x4/k2, g8k`) — 66.2 → 57.9 µs.
- fm: `TILE a:mma_m16n8k16_f16_f16/w2x2/f4x8/k2, REDUCE g4k` (was `…/w2x4/f2x4/k2, g2k`) — 51.3 → 47.9 µs.

Both winners are exactly the `kv_proj` family's recorded configs — plausible, since `k_proj_global.s2048`
(M=2048, N=512, K=3840) is `kv_proj` (M=512, N=2048, K=3840) with the free axes swapped. The 2026-07-13 manual
seeding of this shape simply never tried that family; this was a seeding gap, not a search shortfall.

## Finding 1 — offline prior misprices TILE on the Gemma small-N / split-K region (5× steering regret)

The regret table above localizes it: offline TILE median 4.88× vs online 1.46×, STAGE clean in both halves. The
golden-anchored descent view agrees — for most of these shapes the recorded golden's branch was "never built below
@TILE" during a prior-guided descent, with offline `-O3 pick/golden` misses of 3.1–3.8× (fm rows) where the online
half shows 1.2–1.8×. The sweep converged anyway (brute force within patience), and deploys are protected by the
goldens-first evidence tier, so nothing misdeploys today — the cost is sweep efficiency and censored training data.

The `--blame` attribution (offline half; 6 TILE forks, all missed, regret-weight 22.4) names the culprits:
`D_w_grid_n` (+67.4) and `D_near_threads` (+33.4) push the wrong TILE pick, with `D_l2_cells_occ` / `D_w_grid_m`
secondary; `D_cells_cap` (−34.3) is the main term pulling toward the right sibling. Zero blind forks — the
featurizer does separate the siblings, so this is purely a weights problem.

**Recommendation**: refit the offline weights over the node store now that it holds this card's Gemma rows
(`scripts/golden_knob_heuristics.py`; A/B the candidate via `emmy eval offline --offline-file <file>` before
overwriting), and expect the refit to shrink the `D_w_grid_n` / `D_near_threads` contributions on the
small-N / split-K region.

## Finding 2 — goldens rank deep under the fresh online prior; the golden evidence tier is load-bearing

`eval online --dataset golden`: over the 14 golden shapes with tuned data this sweep, the recorded golden's rank
under the online prior has median 7198 (enumerations 5k–32k), 0/14 in the top-50. Yet every deployed greedy pick
reproduced (or beat) its golden. The deploy evidence hierarchy — live card's goldens first, then the sweep's own
measured DB evidence, then the prior — is what closes that gap; a prior-only deploy on these shapes would land the
1.2–1.8× (online) / 3.1–3.8× (offline) misses shown in the descent view. Nothing to fix in this sweep's scope, but
rank-deep goldens mean the prior cannot be trusted alone on Gemma projection shapes; keep the golden tier populated
for any new card before serving Gemma on it.

## Finding 3 — the standard-lane cuBLAS gap on these shapes is structural, not staleness

Six of seven shapes reproduced their goldens after a fresh 73–182-config search each, and the seventh improved only
14% — every standard-lane config still trails cuBLAS by 1.16–1.48× (worst: `mlp_gate_up.s2048`, 4347 µs vs
2945 µs, 0.67× e2e). Meanwhile the fast-math lane sits at 0.75–1.13× cuBLAS on the same shapes. The search is not
going to close the standard-lane gap by finding better knobs — it has now looked twice (seeding + this sweep).
On sm_89 the f16-accumulate tensor-core atom has 2× the fp32-accumulate throughput, and cuBLAS HGEMM (fp32
accumulate, torch default) still reaches near-peak — so the gap points at the fp32-accumulate codegen tier
(pipeline depth / tile shapes available to it), not at knob selection.

**Recommendation**: kernel-level analysis (the `tune-model` NCU flow) of the fp32-acc split-K family on sm_89,
starting with `mlp_gate_up.s2048` (biggest absolute gap, 1.4 ms/launch) and `k_proj_global` static (smallest —
easiest to profile). Treat as a codegen/tier investigation, not a tuning one.

## Workflow notes

- **Long-lived plain `ssh` calls died twice** (a poll loop and the first eval batch; the VM dropped the connection
  mid-run) — everything longer than ~1 min belongs in a remote tmux session (the tune / A/B / confirm / test runs
  all survived this way). The eval batch happened to finish before the disconnect, but only by luck.
- **`eval variants --kernel` filters on the C kernel identifier**, not the golden name — `--kernel k_proj_global`
  matched nothing and cost a re-derivation via other views. A name-based filter (or accepting both) would remove
  the trap; every other `eval` view accepts the golden-name substring.
- **`record_knobs` emits `LOOPIFY`, the YAML convention stores only the five schedule families** — the skill's
  "copy verbatim" rule is ambiguous the first time you hit it. Either stamp `LOOPIFY` into the YAML convention or
  drop it from `record_knobs`.
- **Narrowed set = per-shape runner script**: `tune --kernel` takes one substring and `--golden` one name, so a
  hand-picked shape list needs a shell loop (losing nothing measurable here, but a `--golden NAME,NAME,…` or
  repeatable `--kernel` flag would make ad-hoc narrowed sweeps one invocation).
- **Noise floor on these shapes was far below the documented 10–13%**: split-K pinned rows reproduced within <1%
  across three runs, so the confirm-twice rule resolved marginal calls (8% fm win) cleanly. The band in the skill
  text is calibrated on the small square shapes; per-kind bands would let smaller wins be recorded with confidence.
- **Mid-sweep scope change**: the initial full 75-shape sweep ran ~15 min before being retargeted to the 7-shape
  set (user narrowed the goal); the killed sweep's partial data was wiped by the narrowed sweep's `--clean`.
  Cheap lesson: for exploratory requests, confirm scope before launching the full-set sweep.
- **Follow-up on the previous 4090 report's notes** (`golden-sweep-rtx4090-anchored-findings.md`): the
  greedy-vs-pinned measurement gap is fixed and held — #376's `greedy (isolated)` row made every comparison here
  pinned-comparable, no false "better"s to catch; the bracket-`pgrep` convention avoided the self-match trap
  throughout; the golden-anchored descent rows again carried every finding directly (no manual `eval`
  cross-referencing). Still true: the tune dominates wall time.
- **Node rows not merged**: this sweep's node rows (4,377, this card) live in the VM's `~/.cache/emmy/autotune.db`
  only. Deliberate: the sweep ran deterministic eps-0 (deploy-faithful), so its sibling coverage is censored — the
  cross-hardware node dataset wants the `collect-node-data` flow (eps 0.25) instead. The offline-weights refit
  (Finding 1) can still use these rows; fetch the DB from the VM before tearing it down.
