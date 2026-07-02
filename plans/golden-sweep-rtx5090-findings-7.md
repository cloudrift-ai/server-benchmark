# Golden sweep findings — RTX 5090 (sm_120), 2026-07-02 (seventh sweep; the perf-gap-fixes sweep)

- **Branch under test:** `feature/golden-perf-gap-fixes` — phases 0–4 of the golden-perf-gap-fixes plan: the A/B
  integrity gates (intensity floor, wrong-answer check, `--json` record, shape-attached golden rows, reduce/pointwise
  paths), the deep-FM `_SCALAR_REG` widening + permanence test, the `_W_A`/`_W_A_DYN` refit, the s512/fp32-square
  re-sweep, and the fp16 staged-slab B64/B128 swizzle.
- **Sweep:** `emmy tune --golden NAME` over the 5 `.s512` statics + 3 fp32 squares, warm resume — **~8 min wall**
  total (the refit prior starts the inner search at the winners: best latency at bench #1 on most shapes; the sixth
  sweep needed ~2.5 h for 29 shapes). Deploy picks analytic-only.
- **A/B:** `run --bench --golden NAME --json`, run TWICE (confirm-twice as JSON fields, not table scraping); the
  intensity floor and wrong-answer gates were active on every pinned row. Zero flags fired; per-shape pass swing 0–1%
  (vs the sixth sweep's 21.4 ↔ 53.0 µs fp16 swings).
- **Tally (12 shapes judged):** 1 better → replaced / 3 same → added / 1 parity reproduction / 7 worse (all ≤ 1.21×,
  five of them ≤ 1.10×; the sixth sweep's worst was 1.49× with 14/29 worse).

## Per-shape outcomes (pass-max, -O3 live A/B, analytic deploys)

| shape | greedy µs | best-golden µs | ratio | eager/cuBLAS µs | vs cuBLAS | category |
|---|---|---|---|---|---|---|
| qwen3_06b.q_proj.s512 | 43.2 | 40.8 | 1.06 | 44.8 | 0.96 | worse (marginal) |
| qwen3_06b.kv_proj.s512 | 27.7 | 30.7 | 0.90 | 33.6 | 0.82 | **replaced** (`n32x8/f2x8 d4/tma/ring`) |
| qwen3_06b.o_proj.s512 | 54.2 | 55.7 | 0.97 | 51.1 | 1.06 | same → added |
| qwen3_06b.gate_up_proj.s512 | 55.0 | 53.5 | 1.03 | 68.7 | 0.80 | same → added |
| qwen3_06b.down_proj.s512 | 81.0 | 67.0 | 1.21 | 65.6 | 1.24 | worse |
| square.1024 | 43.2 | 40.7 | 1.06 | 44.7 | 0.97 | worse (marginal) |
| square.2048 | 365.1 | 242.9 | 1.50 | 257.3 | 1.42 | worse (the one remaining deploy gap) |
| square.4096 | 2183.1 | 2022.1 | 1.08 | 2081.8 | 1.05 | worse (marginal) |
| square.512.fp16 | 5.5 | 5.5 | 1.00 | 6.1 | 0.90 | parity (greedy = recorded knobs) |
| square.1024.fp16 | 17.8 | 16.3 | 1.10 | 14.4 | 1.24 | worse (marginal); legacy row re-spelled + refreshed |
| square.2048.fp16 | 111.2 | 95.9 | 1.16 | 96.9 | 1.15 | worse; winner refreshed to **95.9 µs / 0.99×** |
| square.4096.fp16 | 699.3 | 706.3 | 0.99 | 646.6 | 1.08 | same (greedy = recorded knobs, swizzled re-bench) |

Previous sixth-sweep state for the same shapes: the five `.s512` statics were SKIPPED-unreachable at 1.29–1.49×, the
fp32 squares 1.27–1.45×, the fp16 squares 1.15–1.56× vs cuBLAS at best-golden.

## Finding 1 — reachability + refit closed the s512 / deep-FM gap (plan phases 1–2)

`eval analytic`: ZERO `recorded knobs not in the enumeration` (was 9 SKIPPED), median golden rank **4** (top1 17/43,
top10 26/43) over pools 2–4× larger than the sixth sweep's. The deep-FM tiles both enumerate and deploy: greedy now
picks `f2x8`/`f2x14`/`f4x10`/`f4x12`/`f4x14` register tiles, and the tune's inner search starts at the winners (bench
#1 best on most shapes — the 8-shape sweep took ~8 minutes). The permanence test
(`tests/compiler/test_golden_configs.py::test_golden_knobs_are_members_of_the_move_catalog`) pins every recorded
golden's TILE/STAGE/REDUCE into the move grids so a future space edit cannot silently orphan a golden again.

The widening exposed two latent legality gaps, both fixed: the scalar TMA resolver never checked
`cuTensorMapEncodeTiled`'s 16 B global-stride rule (an odd-N shape crashed at encode once the new weights routed tiny
shapes to TMA), and `_splitk_option` skipped the 1024-thread/CTA budget (an over-limit pinned tile escaped through the
split arm).

## Finding 2 — the fp16 swizzle restored (and beat) the pre-rebuild bar (plan phase 4)

A TMA slab feeding an mma drain is now swizzled (derived per operand — `pick_swizzle_atom` over the slab inner span;
descriptor mode + box split, ldmatrix XOR drain, 8-row-atom slab alignment; cp.async/sync slabs stay row-major).
`square.2048.fp16`'s winning config re-benched **95.9 µs / 0.99× cuBLAS** vs the 106.7 µs / 1.06× pre-rebuild bar;
the fp16 squares' vs-cuBLAS column moved from (0.90, 1.56, 1.37, 1.15) to (0.90, 1.24, 1.15, 1.08) at greedy and
(0.90, 1.13, 0.99, 1.08) at best-golden. Bit-identity held everywhere (the staged-vs-gmem-direct suite is unchanged);
WSPEC p1/p2 composes with the swizzle (same descriptor + drain) — its dedicated re-A/B is follow-up work now that the
drain is no longer conflict-bound.

## Finding 3 — `tune` silently re-creates `prior.json` and flips deploys to the learned prior

The first A/B pass ran with **learned** deploys: the sweep's `tune` re-fit and wrote `prior.json` (the checkpoint
deleted last sweep for mis-calibration), and `FallbackPrior` promotes a fitted learned half automatically. fp16 picks
went catastrophic (square.4096.fp16 at 22.5 ms — 32× the golden; the learned model had never seen fp16 rows from this
sweep), while the freshly-tuned fp32 shapes looked great (evidence-backed picks). The pass was discarded and re-run
after deleting the checkpoint. Until the trainer investigation lands: **delete `prior.json` (or set
`EMMY_PRIOR_FILE` to a nonexistent path) before any analytic A/B judging**, and consider a calibration gate before
`FallbackPrior` promotes the learned half — mis-calibration currently ships to deploys with no check.

## Finding 4 — square.2048 (fp32) is the one remaining deploy gap (1.50×)

The winning config (`n32x8/f4x26 d2/tma/ring`, 242.9 µs — faster than cuBLAS 257.3) ranks 19 under the analytic
prior; greedy deploys `n32x8/f4x12 d4/tma/ring` (365 µs). The `f4x26` row is also -O1-mis-ranked in the tune DB
(rank 13 at -O1, best at -O3) — the -O1 ranking-lane inversion class. down_proj.s512 (1.21×) is the same shape of
miss (greedy `f2x8 d4` vs the `f2x14 g2a d2` winner). A linear model over the current `D_*` features trades these
two shapes against the rest of the set; a register-tile-geometry interaction term (FM×occupancy) is the next lever
if a future sweep wants them, or the learned prior once the trainer is trusted.

## Finding 5 — the reduce fork emits nothing through the live-fork capture

The reduce/pointwise tune + A/B paths landed (finding 6 of the fifth/sixth sweeps closed: `tune --dataset golden`
expands all 34 targets, `run --bench --golden reduce.2048x2048` A/Bs the pinned `b16` row at 3.4 µs), but
`analytic.enumerate_graph` captures ZERO rows for a pure reduce — so greedy deploys serial option-0 (180 µs vs
eager's 6 µs on reduce.2048x2048) and the fitter has no reduce cases (the `_W_A` fit is matmul-only in practice).
The reduce schedule fork either stopped emitting or stopped surfacing through `Run.resolve` after the rebuild —
needs its own investigation before the reduce goldens mean anything again.

## Workflow notes

- The `--json` A/B record + integrity gates worked as designed: no hand-written table parser this sweep (3 sweeps
  running), zero impossible/wrong-answer flags, and the confirm-twice rule is now two JSON files diffed by a script.
- Golden rows survived split deploys (the shape-attach fix) — no `~` approximations in the table above.
- The two `WARPSPEC: true` legacy rows benched with their pinned TILE but an **unpinned** stage (the knob is dead —
  the live codec is `WSPEC`); their honest modern spelling (`STAGE: d4/tma/ring`, which the analytic pick resolved)
  is what got recorded. No legacy-spelling rows remain in the 5090 YAML.
- `eval variants` shows the -O1 ranking lane still inverts against -O3 on big register tiles (square.4096's -O1
  rank-13 pick is the -O3 best) — carried; the two-lane story is unchanged.
