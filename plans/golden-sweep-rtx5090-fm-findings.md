# Golden sweep findings — RTX 5090, fast-math lane (2026-07-09)

- **GPU**: NVIDIA GeForce RTX 5090 (sm_120), branch `feature/f16-accumulate-mma`
- **Sweep**: `EMMY_FAST_MATH=1 EMMY_O3_TOL=0.10 emmy tune --dataset golden --clean` — one cold in-process invocation,
  36 shapes, GPU loop ≈ 3.6 h (the doubled gate-on warp enumeration pushed the fp16 matmuls to 9–18 min each; the
  memory-bound kinds stayed ~11 s). First sweep run under the `FAST_MATH` umbrella (PR #339): the gate-on enumeration
  is a superset, so this one sweep measured both regimes.
- **Tally (36 shapes)**: 3 replaced (standard lane) / 2 fast-math `[fm]` entries added / 17 same / 13 worse /
  1 pathological gate-off deploy (`matmul.qkv.h4096`, Finding 1).
- Logs and A/B data: `_tune/golden-sweep-rtx5090-fm/` (tune.log, per-shape `ab/*.log`, eval snapshots).

## Fast-math lane summary

The two `[fm]` entries recorded (side-by-side with the standard entries, never replacing them — a gate-off compile
cannot reach an f16-accumulate config; `GoldenConfig.fast_math` discriminates the regime from the knobs):

| shape | fm config | fm µs | std golden live µs | ratio | vs cuBLAS |
| --- | --- | --- | --- | --- | --- |
| matmul.qkv.h4096 | `a:mma_m16n8k16_f16_f16/w2x4/f2x4/k8 d1/tma` | 238.5 | 258.9 | 0.92 | **0.95** (beats HGEMM) |
| matmul.square.512.dynM | `a:mma_m16n8k16_f16_f16/w2x2/f2x2/k8 d2/tma/ring` | 3.6 | 4.2 | 0.86 | 0.60 |

Both reproduced 3× with <1% run variance. Numerics contract: torch-default fp16 (chunked f16-accumulate ≈ cuBLAS
HGEMM's own reduced-precision reduction; see the accuracy evaluation in the PR #339 thread). Gate-off replay of the
`[fm]` entry verified via its pin (`EMMY_KNOBS="TILE=a:...f16_f16/w2x4/f2x4/k8,STAGE=d1/tma"` → 237.5 µs, no env).

Where fast-math did NOT win: the tuner measured the f16acc siblings everywhere the gate offered them and kept
f32-accumulate on — all four attention shapes (the flash PV is latency-bound at these sizes, not mma-bound; the
gate-on greedy picked the f32-acc PV atom), the fp16 squares (the f16acc rows lost to the f32-acc incumbents at
equal schedules — the square.4096 win seen in the PR #339 hand A/B needs the `k8` deep-bk region the cold search
didn't reach this sweep), and `mlp_down.h4096.dynM` (fm greedy picked `f16_f16/w8x2/f1x4/k8 g2k` but at 327 µs it
lost to the standard golden's live 300). No forced records — the lane rule is within-regime wins only.

## Fork sibling regret

`emmy eval prior --dataset nodes`, this card's `-O1` block (snapshot: `_tune/golden-sweep-rtx5090-fm/eval_nodes.txt`).
The **analytic** half is the cold-start ranking that decided what this cold sweep measured at all; the **learned**
half is the CatBoost this sweep trained. Regret = predicted-best child's reachable µs / fork best (1.00x = optimal).

| metric (-O1, 51 forks) | analytic prior | learned prior (CatBoost) |
| --- | --- | --- |
| TILE fork regret (median) | 1.73x | 1.36x |
| REDUCE fork regret (median) | 1.01x | 1.00x |
| STAGE fork regret (median) | 1.00x | 1.00x |
| structural PLACE+R+S+T (median) | 2.71x | 1.02x |
| structural +WSPEC (median) | 3.55x | 1.04x |
| worst TILE fork | matmul free=2048 red=2048: 13.41x | matmul free=12288 red=4096: 3.88x |
| 2nd-worst TILE fork | matmul free=4096 red=14336: 11.37x | matmul free=2048 red=2048: 2.00x |
| 3rd-worst TILE fork | matmul free=12288 red=4096: 4.28x | matmul free=4096 red=4096: 1.52x |
| leaf reachability (mean / median / worst) | 1.86x / 1.21x / 12.73x | 1.38x / 1.00x / 8.42x |
| leaf calibration (median per-op Spearman) | +0.56 | +0.86 |

Diagnosis: the misses are overwhelmingly the **analytic TILE family** — the cold ranking steers the fp16 warp-TILE
search into the wrong region on the big-N/big-K GEMMs (13.4x on the 2048 square fork, 11.4x on mlp_down's K=14336),
which also censors what the learned model ever sees. The learned half is far cleaner per-fork (median TILE 1.36x,
calibration +0.86) but inherits the censoring — and its **golden rank** is inverted from the analytic's (Finding 2):
per-fork it ranks measured siblings well, yet over the full enumeration it buries the goldens. The structural-fork
medians (2.71x/3.55x analytic) sit on the small square.512 family where the analytic prefers scalar/wrong forms.
Fix priority: refit `_W_A` via `scripts/golden_knob_heuristics.py` with this sweep's recorded rows, TILE features
first (`D_near_waves`/`D_square` carry the blame weight on the worst forks — see Finding 1's blame table).

## Per-shape outcomes — standard lane

Live -O3 A/B (`emmy run --bench --golden NAME`, no env); ratio = greedy/best-golden-total (golden split rows summed);
`(*)` = the harness's physical-impossibility flag fired on a golden row (tiny memory-bound re-bench artifact —
category unreliable). vs cuBLAS = greedy_us / live eager row (>1 = emmy slower).

| shape | greedy µs | golden µs (live) | ratio | category | cuBLAS µs | vs cuBLAS |
| --- | --- | --- | --- | --- | --- | --- |
| matmul.qkv.h4096 | bench_fail | — | — | **pathological pick** (Finding 1) | 250 | ~1000x |
| matmul.square.512.fp16 | 11.5 | 4.4 | 2.61 | worse (Finding 3) | 6 | 1.92 |
| matmul.square.1024 | 28.0 | 21.1 | 1.33 | worse | 14 | 2.00 |
| reduce.k2048 | 2.0 | 1.5 | 1.33 | worse | 6 | 0.33 |
| attention.hd64 | 10.7 | 8.3 | 1.29 | worse (Finding 5) | 10 | 1.07 |
| matmul.square.512.dynM | 5.3 | 4.2 | 1.26 | worse | 6 | 0.88 |
| matmul.mlp_down.h4096.dynM | 367.4 | 301.2 | 1.22 | worse | 292 | 1.26 |
| matmul.square.2048 | 98.5 | 90.8 | 1.08 | worse | 98 | 1.01 |
| matmul.mlp_down.h4096 | 321.6 | 297.9 | 1.08 | worse | 292 | 1.10 |
| matmul.o_proj.h4096 | 96.4 | 90.5 | 1.07 | worse | 96 | 1.00 |
| matmul.square.512 | 9.0 | 8.5 | 1.06 | worse | 12 | 0.75 |
| rms_norm.k2048 | 4.1 | 3.9 | 1.05 | worse | 4 | 1.02 |
| matmul.square.4096 | 670.8 | 646.9 | 1.04 | worse | 654 | 1.03 |
| reduce.k8192 | 3.3 | 3.2 | 1.03 | worse (boundary) | 6 | 0.55 |
| matmul.mlp_gate_up.h4096 | 612.2 | 641.5 | 0.95 | **better → replaced** (Finding 4) | 556 | 1.10 |
| matmul.mlp_gate_up.h4096.dynM | 610.6 | 641.0 | 0.95 | **better → replaced** | 554 | 1.10 |
| matmul.qkv.h4096.dynM | 250.8 | 260.6 | 0.96 | **better → replaced** | 250 | 1.00 |
| attention.hd128 / hd128.dynM / hd64.dynM | 16.1 / 16.3 / 8.5 | 16.4 / 16.6 / 8.5 | 0.98–1.00 | same | 18 / 18 / 10 | 0.85–0.91 |
| matmul.o_proj.h4096.dynM | 94.6 | 96.1 | 0.98 | same | 96 | 0.99 |
| pointwise ×4 | 3.3–11.4 | 3.3–11.2 | 1.00–1.02 | same | 4–12 | 0.82–0.95 |
| rms_norm k4096/k8192 (+dynM) | 6.6–10.0 | 6.6–10.0 | 1.00 | same (dynM (*)) | 6–10 | 1.00–1.10 |
| softmax ×4 | 3.7–10.3 | 3.7–10.3 | 1.00–1.05 | same (k2048.dynM worse(*)) | 4–14 | 0.73–0.97 |
| reduce k2048.dynM / k8192.dynM | 2.1 / 3.3 | 1.5(*) / 3.3(*) | 1.40(*) / 1.00(*) | unreliable golden re-bench | 5 / 6 | 0.42 / 0.55 |

Replacement notes: all three replaces reproduced across 3–6 runs with <1% variance; the gate_up pair's recorded
`emmy_us` is HIGHER than the old entries' recorded values because the old recordings were never reproducible under
their own knobs (Finding 4, bisect-resolved) — the YAML entries carry inline comments explaining it.

## Finding 1 — gate-off greedy deploys a per-cell scalar kernel on qkv.h4096 (~1000×)

The worst miss of the sweep, and it is a **deploy-path** miss, not a search miss. Gate-off (`emmy run --bench
--golden matmul.qkv.h4096`, no env) the greedy pick compiles the per-cell scalar tier — `eval golden` shows the
pick's `TILE` **empty** vs the golden's `a:mma_m16n8k16_f16_f32/w2x4/f2x4` — and each launch runs ~260 ms
(512×12288×4096 fp16 GEMM on one thread per cell), blowing the A/B's 10 s GPU budget (`bench_fail`, 3× reproduced).
Gate-ON the greedy picks the f16acc row at 238 µs. The same sweep's data, two regimes, opposite outcomes.

Evidence: `eval prior --dataset nodes --blame --kernel "free=12288"` — analytic -O1 reachability 8.42x/5.72x
(`<-- misses best` on both qkv ops), analytic per-op calibration **−0.29** (anti-correlated!), TILE blame led by
`D_near_waves +23.9` / `D_square +14.7` / `D_near_tilen +10.0` (analytic half). The -O3 lane holds only 32 nodes
for this shape — the winner-focused -O3 rebench thinned the deployable evidence, and the gate-on winner
(the f16acc row) **does not exist in the gate-off enumeration**, so the deploy-time evidence pick falls through to
the model over rows it has thin/anti-correlated data for and lands on the scalar tier.

**FIXED (2026-07-09, post-sweep).** Root cause: the -O3 rebench band (`mcts.TuningSearch.observe`) measured every
row against the GLOBAL best -O1 latency; under the gate the f16acc row owned the best, and this sweep's
`EMMY_O3_TOL=0.10` (the skill's time lever) put the best standard row (~8% off) outside the band — so the standard
lane got zero -O3 evidence and the gate-off evidence-first deploy fell to the model argmin over anti-calibrated
scores. The fix makes the band **per-regime**: a standard row competes against the best standard -O1 latency
(`fast_math_knobs` discriminates; no gate → identical behavior), so the best standard row always reaches -O3.
Verified: after a gate-on re-tune of the shape, the gate-off greedy deploys `w4x4/f2x2/k4 d2/tma/ring` at 250.8 µs
(0.99x vs cuBLAS), and the same A/B now benches both golden rows — the `[fm]` entry replays at 240.1 µs gate-off
via its pin. Remaining half: the analytic TILE weights stay anti-calibrated on free=12288 (Spearman −0.29) —
include this sweep's rows in the next `golden_knob_heuristics.py` refit.

## Finding 2 — the freshly-trained learned prior buries the goldens (median rank 587) while the analytic ranks them #1

`eval analytic`: 9/14 matmul goldens rank **0** (top-1) under the cold analytic prior, median rank 0. `eval prior
--dataset golden`: the same goldens under the just-trained CatBoost rank at **median 587** (best 24, worst 6360;
top-10 = 0/14). The inversion means the learned model — despite +0.86 per-op calibration on measured siblings —
mis-extrapolates over the full enumeration: it was trained on the gate-on sweep's measured rows (winner-skewed,
analytic-censored) and its scores don't transfer to the unmeasured bulk where the goldens' exact rows sit. The
`vs gold` -O3 perf column corroborates: the deploys it grounds on measured evidence are fine (the A/B "same" rows),
the model-argmin ones are not (Finding 1).

Recommendation: this is the training-data/censoring problem, not a featurization gap (no blind forks were
reported). Two cheap levers: include the golden rows themselves as training anchors (they are measured -O3 rows —
`Dataset.from_golden` already builds the samples), and keep `--explore-eps` sweeps (the collect-node-data skill's
ε-greedy) feeding sibling coverage so the model sees non-winner regions.

## Finding 3 — square.512.fp16: greedy 2.61× off a split-K golden (repeat offender)

Golden: `w1x8/f4x1/k4 + g2k + d3/tma/ring` (live 4.4 µs total). Greedy: `w2x2/f2x2/k4`, no stage, no split
(11.5 µs). The analytic prior ranks the golden **#1** (rank 0/19038) — so the cold ranking is right here, and the
miss is the search/O-3 pipeline: `eval variants` shows the greedy's deployed pick ranked far from the measured best
at -O3 (analytic -O3 reachability worst entry: free=512 4.37x). The small-square family also carries the sweep's
worst structural regret (3.55x +WSPEC). This is the same early-stopped fp16-small-shape class the 2026-07-07 4090
A/B hit. Recommendation: patience is already 50 and the golden ranks #1 analytically — the gap is the -O1→-O3
inversion on tiny kernels (`EMMY_O3_TOL=0.10` trims contenders whose -O1 rank is >10% off); for the ≤512 shapes
specifically, widen the -O3 rebench band (or rebench top-N by count, not tolerance) in `emmy/commands/tune.py`.

## Finding 4 — the old gate_up recordings were never reproducible (recording fidelity, NOT codegen drift)

**Bisect-resolved (2026-07-09, post-sweep).** The replaced `mlp_gate_up.h4096` golden (`w2x2/f4x4/k4 d1/tma`,
recorded 605.5 µs at commit 8ab967cf) re-benches at **638 µs at its own recording commit** and 639 at HEAD
(worktree bisect, identical pin, cuBLAS steady at ~552-557 on both) — so no code change slowed it; the recorded
605.5 was not produced by the recorded knob set. The pre-sweep tune-DB backup predates that sweep, so the original
measurement row is unrecoverable; the mechanism is the same family as the dynM twin's wrinkle: its recorded knobs
never pinned `REDUCE`, and today's greedy fill adds `g2k` (613.6 + 27.4 finalize = 641.0) — the recorded knob set
under-specifies what actually ran. Recommendations: (a) when recording goldens, record every resolved schedule
family (even the empty `REDUCE`) — the enumeration's honest-stamping rule extended to the YAML; (b) the recorder
should close the loop: after editing the YAML, one `run --bench --golden NAME` and require the golden row within
~3% of the just-written `emmy_us` (the #335 pin-verification gate checks knob fidelity, not latency
reproducibility — this is the missing half).

## Finding 6 — the refit surfaced two latent masked-ILP bugs (both fixed)

Landing the recommended analytic refit (`golden_knob_heuristics.py` over this sweep's rows; dyn median rank
1063 → 97, the fm rows reconstructed under the gate) steered the cold pick for the dynamic scalar softmax / SDPA
kernels onto the ILP register fold (`_W_A_DYN`'s `D_reduce_ilp +25.5`) — a region the old weights never reached —
and exposed two pre-existing bugs in `_factor`'s masked ILP tail, both fixed with regression tests:

1. **The per-copy rename suffixed symbolic dims in buffer strides** (`seq_len` → `seq_len__r3`, an nvcc failure):
   the protected set covered the reduce axis's own extent but not a symbolic dim entering through a flattened
   4-D index's strides. Fix: protect every expression-read name the body doesn't define.
2. **Per-`Accum` masking corrupts a TWISTED carrier's tail** (softmax r4 at seq=33: max err 5.7e-3): the merge's
   shared intermediates (`t0 = max(m, s_raw)` feeding the `l·exp(m − t0)` rescale) read the raw wrapped duplicate,
   silently down-scaling the denominator whenever the duplicate beat the running max — sizes passed or failed by
   luck of the wrapped values. Fix: a twisted carrier's masked copy clamps the PIVOT TERM (the score) to the pivot
   fold identity, which the monoid absorbs whole-chain (`max(m, −inf) = m`, rescale 1, weight 0, lifted `0·V`) —
   the term, not the streamed loads, because a flash score is computed by a nested Q@K contraction whose input
   loads must stay raw. Old coverage never caught it: the only ILP×twisted×dynamic case ran at seq=700 ≡ 0 (mod 4)
   — no masked tail. New tests sweep off-stride sizes at 1e-5 tolerance.

The episode is the fork-regret table's diagnosis in action: the analytic prior was censoring an entire region
(ILP folds on dynamic reduce shapes), and the first weight set that priced it honestly immediately found broken
codegen there. Post-fix, the full suite is green with the new weights (2277 tests).

## Finding 5 — attention.hd64 static: greedy 1.29× off its golden; the dynM twin reproduces

Golden `w4x1/f1x16/k4` variants at 8.3 µs; static greedy lands 10.7 while the **dynM twin picks its golden exactly**
(8.5/8.5). The static/dynM asymmetry means the masked-tier weights are fine and the static flash ranking is the
miss. Flash forks don't appear in the nodes regret table (single-fork ops this sweep), so the evidence is thin;
`eval golden --kernel attention.hd64` shows the pick differing on the PV `TILE@pj` reg geometry. Recommendation:
low priority (2.4 µs absolute); fold the flash geometry rows into the next heuristics refit rather than a
dedicated fix.

## Workflow notes

- **The A/B harness cannot bench golden rows when the greedy pick is pathological**: `run --bench --golden` benches
  the greedy first, and qkv's 260 ms pick burned the 10 s budget before any golden row ran — three runs wasted
  before diagnosing via `eval golden` + a pinned `EMMY_KNOBS` run instead. Proposed: bench golden/ab rows before the
  greedy pick, or add `--golden-only`.
- **Golden split-K configs print as two rows with no total**: the partial + finalize rows must be hand-summed (my
  first parse took `min` and produced physically impossible "5.7 µs" GEMMs). Proposed: a `golden NAME (total)` row,
  or a stored per-config row-group id.
- **Memory-bound `.dynM` golden re-benches trip the impossibility gate** (`! ... implies 712 TFLOP/s`): the tiny
  reduce/softmax re-bench rows are unreliable this sweep (flagged `(*)` above) — pre-existing; the static twins
  stand in fine. Proposed: skip the dynM re-bench for the memory-bound kinds (the skill already allows skipping
  their tune).
- **Fast-math lane selection was manual**: the 17 gate-relevant shapes were hand-listed (fp16 dtype). Proposed:
  `eval golden --fast-math` filter or a `tune --dataset golden --fast-math-only` subset, driven by
  `GoldenConfig.dtype`/`fast_math`.
- **Wall time 3.6 h vs the 2.5 h budget**: the gate-on enumeration roughly doubles the warp-TILE rows, and the fp16
  matmuls dominate (9–18 min each). Routine fast-math sweeps could narrow to `--kernel` the fp16 shapes and reuse
  the standard sweep for the rest — the two lanes' winners are separable.
- **`eval analytic`'s found column showed `-` (absent) for every knob** while the rank column worked — the found
  side didn't stamp under this branch's build; didn't block the sweep (ranks + A/B carried the analysis) but worth
  a look before the next sweep.
- Previous sweep's notes (2026-07-06/07, deleted reports): the live-card golden scoping and pin-verification gate
  fixes (#335) held — no cross-card shadowing, and the one unreproducible-pin risk this sweep (gate_up.dynM's
  unpinned REDUCE) was caught by reading the golden row's knob columns, which #335's gate made visible.
