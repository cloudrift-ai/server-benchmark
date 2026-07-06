# Golden sweep — RTX 4080 (sm_89), 2026-07-03

**GPU:** NVIDIA GeForce RTX 4080 (sm_89, 76 SMs), driver 595.71.05. **Code:** `main` @ `2c823cfb` (goldens seeded at
`272153d3`, #303 "Seed RTX 4080 golden matmul configs"). **Commits landed since the seed:** #305 (scalar staging
refuses masked-N / transposed-B), #306 (gate TMA staging to sm_90+), #307 (flash codegen).

**Sweep command:**
```
emmy tune --dataset golden --clean          # 34/34 shapes, 0 failures; trains the learned prior
emmy run --bench --golden NAME --json …     # per-shape greedy-vs-golden A/B at -O3 (23 recorded shapes)
```
**Wall time:** tune **53.4 min** (17:43:40 → 18:37:02 — 34 shapes incl. the heavy fp16 + large squares; longer than
the skill's ~30 min because the dataset is 34, not 23), A/B ~2 min, confirm re-benches ~5 min. Work dir
`_tune/golden-sweep-rtx4080/2026-07-03/` (tune.log, ab/*.json, confirm/*.json, eval_*.txt, NOTES.md).

**Category tally: 19 replaced · 1 added · 1 STAGE-replaced (parity) · 2 unchanged (worse).**
- **19 better** (greedy >3% faster than the golden row): 3 fp32 squares (512/1024/2048), square.4096.fp16, all 15
  qwen3_06b projections. Recorded greedy's config + `emmy_us`; deleted the superseded `d1/cp` / atomic-split-K entries.
- **square.4096** (parity, same TILE, only STAGE `d1/cp`→`d2/cp/ring`, greedy stable ≥ golden every run): replaced the
  STAGE + `emmy_us`.
- **square.1024.fp16** (parity, different knobs): **added** the deployed `d2/cp/ring/p2` config as a second entry
  alongside the recorded `d1/cp` one.
- **2 worse** (greedy slower — real prior shortfalls): **square.512.fp16**, **square.2048.fp16** — left untouched, see
  Findings 2–3.

## Per-shape outcome — slowest-vs-golden first

`greedy` / `golden` are both **live -O3 re-benches this run** (the A/B). `ratio` = greedy÷golden-row. `seed` = the config's
recorded `emmy_us` at seed time; `vsSeed` = greedy÷seed (the real improvement over what shipped). `cuBLAS` = recorded
`cublas_us` (config-independent; live Eager agreed within ~5%); `vsCB` = greedy÷cuBLAS (>1 = emmy slower than cuBLAS).

| shape | greedy µs | golden µs | ratio | seed µs | vsSeed | cuBLAS µs | vsCB | category |
|---|---|---|---|---|---|---|---|---|
| square.2048.fp16 | 238.1 | 208.5 | 1.14 | 207.1 | 1.15 | 197.0 | 1.21 | **worse** |
| square.512.fp16 | 9.0 | 8.3 | 1.08 | 8.3 | 1.08 | 6.0 | 1.60 | **worse** |
| square.4096 | 4636.7 | 4690.9 | 0.99 | 4854.8 | 0.96 | 3922.0 | 1.21 | same → STAGE replaced |
| square.1024.fp16 | 32.8 | 33.2 | 0.99 | 35.7 | 0.92 | 27.0 | 1.22 | same → added |
| square.4096.fp16 | 1399.8 | 1606.7 | 0.87 | 1580.0 | 0.89 | 1350.0 | 1.04 | better |
| qwen3_06b.down_proj.s32 | 16.9 | 20.7 | 0.81 | 26.1 | 0.65 | 15.0 | 1.13 | better |
| qwen3_06b.down_proj.s128 | 39.9 | 50.0 | 0.80 | 59.3 | 0.67 | 32.0 | 1.25 | better |
| qwen3_06b.o_proj.s32 | 11.5 | 14.8 | 0.78 | 21.9 | 0.53 | 11.0 | 1.05 | better |
| square.512 | 15.0 | 19.4 | 0.77 | 17.9 | 0.84 | 18.0 | **0.83** | better |
| qwen3_06b.o_proj.s128 | 26.0 | 35.7 | 0.73 | 40.7 | 0.64 | 24.0 | 1.08 | better |
| qwen3_06b.q_proj.s512 | 91.7 | 127.9 | 0.72 | 92.5 | 0.99 | 76.0 | 1.21 | better |
| qwen3_06b.gate_up_proj.s128 | 40.1 | 56.0 | 0.72 | 53.5 | 0.75 | 33.0 | 1.22 | better |
| qwen3_06b.q_proj.s128 | 28.1 | 40.1 | 0.70 | 32.8 | 0.86 | 28.0 | 1.00 | better |
| qwen3_06b.q_proj.s32 | 11.3 | 16.3 | 0.70 | 16.1 | 0.70 | 10.0 | 1.13 | better |
| qwen3_06b.down_proj.s512 | 138.4 | 203.6 | 0.68 | 157.9 | 0.88 | 120.0 | 1.15 | better |
| qwen3_06b.gate_up_proj.s512 | 129.5 | 191.1 | 0.68 | 146.8 | 0.88 | 123.0 | 1.05 | better |
| qwen3_06b.gate_up_proj.s32 | 16.7 | 24.8 | 0.67 | 21.7 | 0.77 | 15.0 | 1.11 | better |
| qwen3_06b.o_proj.s512 | 85.1 | 126.5 | 0.67 | 117.8 | 0.72 | 84.0 | 1.01 | better |
| square.1024 | 85.2 | 128.9 | 0.66 | 101.8 | 0.84 | 77.0 | 1.11 | better |
| square.2048 | 587.8 | 933.9 | 0.63 | 595.5 | 0.99 | 499.0 | 1.18 | better |
| qwen3_06b.kv_proj.s128 | 13.8 | 22.0 | 0.63 | 24.4 | 0.56 | 18.0 | **0.77** | better |
| qwen3_06b.kv_proj.s512 | 43.5 | 72.3 | 0.60 | 54.6 | 0.80 | 41.0 | 1.06 | better |
| qwen3_06b.kv_proj.s32 | 7.0 | 12.4 | 0.56 | 12.3 | 0.57 | 8.0 | **0.87** | better |

## Key result — the fresh prior deploys `d2/cp/ring` + `g2k` almost everywhere

Every one of the 23 shapes moved off the seed's regime. Two structural shifts drive all 19 wins, and both are the exact
levers the 2026-07-02 lowperf analysis (an untracked working note) flagged as left on the table:

1. **Pipeline: `d1/cp` (single-buffered) → `d2/cp/ring` (2-stage ring).** 20 of 24 recorded entries are now `d2/cp/ring`.
   The seed universally deployed the shallow `d1/cp` (`wait_group 0` right after `commit_group`, no load/compute
   overlap). The prior now double-buffers, and it wins on every fp32 square and projection.
2. **Tiny-M reduction: global-atomic split-K (`g4a`/`g8a`) → `g2k` (split-K through the reduce path).** The five s32
   shapes and three s128 down/o/kv shapes dropped the naive `atomicAdd`-to-global reduction for a `g2k` split that
   reduces through the kernel's own reduce lowering. `vsSeed` for these is 0.53–0.67 — the biggest **real** gains in the
   sweep, and they agree with the A/B ratio (so they're not an artifact of Finding 1).

Four shapes now **beat cuBLAS**: square.512 (0.83×), kv_proj.s128 (0.77×), kv_proj.s32 (0.87×), q_proj.s128 (1.00×). The
rest still trail cuBLAS 1.04–1.25× — the small/skinny-GEMM GPU-fill ceiling the lowperf plan described is unchanged; the
sweep closed the emmy-vs-emmy gap, not the emmy-vs-cuBLAS one.

## Finding 1 — the recorded `d1/cp` goldens regressed since the seed; the A/B ratios overstate the large-square wins

The `golden` column re-benches the *recorded config's knobs live this run*, and for the large fp32 squares it lands far
**above** the seed's recorded `emmy_us` — the pinned `d1/cp` config no longer reproduces its seed latency:

| shape | seed `emmy_us` (d1/cp) | golden row re-bench (3×, stable) | Δ |
|---|---|---|---|
| square.2048 | 595.5 | 933.9 / 937.8 / 937.0 | **+57%** |
| square.1024 | 101.8 | 128.9 | +27% |
| square.512 | 17.9 | 19.4 | +8% |
| square.4096 | 4854.8 | 4690.9 → 4900.7 (drifts) | ~0 |

This is **not** a bench-order / persisting-L2 artifact from benching greedy before the golden: pinning the square.2048
`d1/cp` config as the *primary* emmy kernel (`EMMY_KNOBS="TILE=n16x8/f4x8,STAGE=d1/cp" emmy run --bench -c "…2048³…"`,
benched first, nothing after) still measures **957 µs** (occupancy 8%, 72 regs, single-buffered). So the recorded 595.5
is genuinely unreachable by that config on `2c823cfb`.

Consequence: the A/B `ratio` for the large fp32 squares (square.2048 0.63, square.1024 0.66) is **inflated** — the honest
number is `vsSeed` (square.2048 **0.99**, square.1024 0.84): `d2/cp/ring` mostly *recovers* the seed-time speed that
`d1/cp` lost, rather than beating it outright. The recorded-config replacement is still correct (the new golden must
reflect the deployed `d2/cp/ring`, and the stale 595.5 advertised a latency the `d1/cp` config can no longer hit), but
the framing matters.

**Recommendation (HIGH — this is a codegen regression, not a golden-set issue).** Bisect `272153d3..2c823cfb` on the
single square.2048 `d1/cp` config (the isolation command above is the repro; look for a ~1.6× jump). Occupancy 8% on a
single-buffered kernel points at a register-pressure / launch-bound change. If `d1/cp` is meant to stay a live option in
the search space (`space.py:321` still offers it), it should not silently cost 1.6×. This is independent of the golden
YAML and worth its own issue.

## Finding 2 — square.2048.fp16: the prior over-prefers the deep pipeline on the large fp16 square (greedy 1.14×)

Greedy deploys `a:mma_m16n8k16_f16/w4x1/f2x4/k2 + d2/cp/ring/p2` at **238 µs**; the golden is
`a:mma_m16n8k16_f16/w2x1/f4x4/k2 + d1/cp` at **208 µs** — here the *shallow* `d1/cp` with a wider `f4x4` register tile
wins, the opposite of the fp32 story. Confirmed stable 3× (1.14 / 1.14 / 1.18). `eval prior --dataset golden` scores this
at **1.28× vs gold** and ranks the golden config **1177 / 5692** under the learned prior; the cold `AnalyticPrior` ranks
it **1262 / 5692**. Both priors bury the correct config: they learned "`d2/cp/ring` wins" from the fp32 CUDA-core shapes
and over-generalized it to the fp16 tensor-core path, where a large square (2048³, K=2048 → many K-steps but each
`mma`-heavy) prefers the wider tile + shallow stage.

**Recommendation (MEDIUM).** Refit the analytic tensor-core weights with `scripts/golden_knob_heuristics.py` after this
sweep — the two fp16 square goldens (this one + Finding 3) are now recorded, so the refit has ground truth that
distinguishes fp16 staging from fp32. If a single weight set can't separate "deep pipe good (fp32 / large-K)" from "wide
tile + shallow good (fp16 large square)", add an engineered feature keyed on `dtype==fp16 ∧ mma` × register-tile-width so
the prior can price the fp16 tensor-core staging separately. Deep analytic rank (1262) ⇒ the heuristic is the problem, not
search patience.

## Finding 3 — square.512.fp16: the fp16-optimal `d4/cp/ring` was never sampled (deep-ring exploration gap; greedy 1.08×)

Greedy deploys `a:mma_m16n8k16_f16/w4x2/f2x2/k2 + d2/cp/ring/p2` at **9.0 µs**; the golden is
`a:mma_m16n8k16_f16/w2x1/f1x4/k2 + d4/cp/ring` at **8.3 µs** — a **4-stage** ring. Confirmed dead-stable 3× (1.08 / 1.08 /
1.08); the delta is small (8%) and both configs sit ~1.5–1.6× cuBLAS HGEMM (6.0 µs), so this is a marginal shape, but the
*reason* the prior misses it is the most actionable finding in the sweep:

- The golden config ranks **7769 / 9884** under the learned prior (near-last) and **1921 / 9884** under the cold analytic
  — the learned prior made it *worse* than cold, having trained only on shallow-staged fp16 measurements.
- **`d3/cp/ring` and `d4/cp/ring` were benched 0 times across the entire 34-shape tune** (`grep -c "d[34]/cp/ring"
  tune.log` = 0). The deep-ring stages are in the search space (`emmy/compiler/pipeline/search/space.py:321`:
  `ring = ["", "d1/cp", "d2/cp/ring", "d3/cp/ring", "d4/cp/ring", …]`), but greedy — driven by the prior that ranks them
  ~7700th — never *selects* them within its ~112-bench patience for this shape. `eval variants` confirms it: the tune DB
  holds essentially no measured fp16 `mma` variants for the square kernels (only 8, all in one `__partial` kernel). It's
  a self-reinforcing blind spot: the prior buries deep-ring → greedy never measures it → the prior never learns it's good.

**Recommendation (HIGH — one flag closes the loop for all fp16 shapes).** Re-tune the fp16 shapes with ε-greedy
exploration — `emmy tune --dataset golden --kernel fp16 --explore-eps 0.25` (the same lever `collect-node-data` uses) — so
the search samples the deep-ring siblings the incumbent prior skips, measures `d3`/`d4/cp/ring`, and lets the learned
prior discover the fp16-optimal staging. Pair with the analytic refit from Finding 2. Without exploration, no amount of
patience reaches a rank-7769 config.

## Workflow notes

Retrospective on the loop itself (for whoever maintains the CLI + this skill). No prior RTX 4080 sweep report exists to
diff against (this is the first).

- **The tune dominates wall time (53 of ~60 min) and re-runs 34 shapes even though only 23 are recorded for this GPU.**
  The 11 extra (`.dynM` + pointwise + GPU-agnostic) train the prior but produce no 4080 golden to A/B. *Improvement:* a
  `tune --dataset golden --gpu-only` (or reuse `--kernel`) to tune just the live GPU's recorded shapes would cut the
  correctness-irrelevant fp16-partial / pointwise time. Also `--gpus N` (multi-GPU) is a no-op on a 1-GPU box — noted, not
  a gap.
- **`--json` + a tiny parser retired all table-scraping — this worked well.** `run --bench --golden --json` gave
  greedy/golden/eager/knobs/flags as one record; `parse_ab.py` categorized 23 shapes in one pass. Keep this; it's the
  right primitive. The one gap: the JSON has no `recorded_emmy_us` cross-check surfaced against the *live* golden row, so
  Finding 1 (the d1/cp regression) took a hand-written diff of seed-YAML vs the A/B JSONs. *Improvement:* have
  `--golden` emit `recorded_emmy_us` vs `golden_row_us` and flag a >20% gap — that regression would have been a printed
  warning, not a manual discovery.
- **`eval variants` can't be reached by golden name.** `--kernel square.512.fp16` returns "no measured variants"; the
  view keys by the DB kernel hash (`k_matmul_180e20`). Mapping a golden shape → its kernel hash took a full
  `eval variants --top 0` dump + grep-by-latency-magnitude — a multi-command detour for what should be one lookup.
  *Improvement:* accept a golden name in `eval variants --dataset golden --kernel NAME` and resolve it to the hash (it
  already errors "`--dataset golden has no per-variant measurements`", so the plumbing is half there).
- **The `d1/cp` regression was invisible until I diffed against the seed YAML by hand.** The whole sweep's A/B ratios
  read as huge wins (median ~0.70) that were partly a baseline regression. Only the `vsSeed` column (recovered from
  `git show HEAD:…yaml`) separated real gains (split-K shapes) from recovered-regressions (large squares). *Improvement:*
  the report/parse should always print `vsSeed` next to the A/B ratio — it's the honest "did emmy get faster" number and
  it's free from data already on disk.
- **Noise-floor re-runs (step 4) were cheap and decisive here.** 7 shapes × 2 re-benches confirmed every category
  (fp16 worses stable to ±0.04, square.2048 golden stable at ~937). No shape needed more than 2 re-runs; the wins were so
  large (≥19%) that only the 4 near-noise fp16/parity shapes actually needed confirming. The ~10–13% noise band held.
