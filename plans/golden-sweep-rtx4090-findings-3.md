# Golden sweep — RTX 4090 (sm_89), matmul family, 2026-07-07

Third 4090 sweep report (siblings: `golden-sweep-rtx4090-findings-2.md`, 2026-07-06, which ran pre-#315 on the
per-model shapes). First sweep on the op-typed golden set: 17 matmul shapes = the 9 recorded 4090 entries + the 8
standardized `h4096` projection shapes recorded only for the RTX 5090 until now (the gaps findings-2 flagged).

**GPU:** NVIDIA GeForce RTX 4090 (sm_89), CUDA 12.9. **Code:** `main` @ `ba7590d6` + the compile-budget bump
(4 s → 12 s, wall 8 s → 16 s) that ships in this PR — see Finding 1 for why it was needed mid-sweep.

**Sweep commands:**
```
emmy tune --dataset golden --kernel matmul --clean     # 17 shapes, ~3 h (incl. one partial re-tune, see F1)
emmy tune --dataset golden --kernel mlp                # re-tune of the 4 MLP shapes after the budget fix, ~47 min
emmy run --bench --golden matmul.square.*              # 9 recorded shapes, live -O3 A/B, ×3 for marginal calls
emmy run --bench -c "<snippet>" [--dynamic seq_len@x0:0]   # seed benches for the 8 h4096 gap shapes
emmy run --bench -c "<gate_up snippet>" --ab "<knobs>"     # pinned re-bench of the tune's best -O3 gate_up config
```

**Category tally: 2 replaced · 2 added · 8 seeded · 2 unchanged (same knobs) · 1 left (in-band loss) · 2 worse.**

- **2 replaced** (confirmed in 3/3 A/B runs): `square.512` 14.4 → 13.4 µs, `square.1024` 60.1 → 55.4 µs. Both are
  the same move — a `n16x8` block tile displacing the recorded `n32x8` (reduce/stage unchanged), continuing the
  small-tile shift findings-2 started.
- **2 added** (parity, different knobs): `square.4096.fp16` (`w2x4/f4x4`, no k-split, 904.2 µs) and
  `square.512.dynM` (`n16x8/f4x8 + g2k + d4/cp/ring`, 11.2 µs — the greedy g2k program at parity with the
  recorded g4k config).
- **8 seeded**: all four `h4096` projections + `.dynM` twins — first 4090 entries. The two `mlp_gate_up` entries
  record the tune's best -O3 variant (`w4x1/f4x8/k2`, ~976 µs) benched pinned via `--ab`, NOT the greedy pick —
  greedy misdeploys both (Findings 1–2).
- **2 unchanged**: `square.2048` / `square.4096` — greedy reproduces the recorded knobs exactly.
- **1 left**: `square.2048.fp16` — greedy lost all 3 runs (1.3 / 1.9 / 5.5 %); median inside the 3 % band but a
  config that never wins is not a parity alternate.
- **2 worse**: `square.512.fp16` (1.09×), `square.1024.fp16` (1.13×) — Findings 3–4.

## Per-shape outcome

Live -O3 program totals (split-K = matmul + epilogue summed); medians over the confirmation runs where several
were made. `ratio` = greedy ÷ best-golden re-bench; `vs CB` = recorded ÷ `cublas_us` (>1 = emmy slower than
cuBLAS). For the two gate_up shapes the recorded µs is the pinned best, with the greedy misdeploy in parentheses.

| shape | recorded µs | golden µs | ratio | cuBLAS µs | vs CB | category |
|---|---|---|---|---|---|---|
| square.512 | 13.4 | 14.6 | 0.92 | 10.8 | 1.24 | replaced |
| square.1024 | 55.4 | 60.5 | 0.92 | 45.4 | 1.22 | replaced |
| square.2048 | 358.4 | 364.9 | 0.98 | 320.0 | 1.12 | unchanged (same knobs) |
| square.4096 | 2829.3 | 2806.6 | 1.01 | 2458.6 | 1.15 | unchanged (same knobs) |
| square.512.fp16 | 7.3 | 6.7 | **1.09** | 5.8 | 1.26 | **worse** (F4) |
| square.1024.fp16 | 26.1 | 23.0 | **1.13** | 18.1 | 1.44 | **worse** (F3) |
| square.2048.fp16 | 123.0 | 116.7 | 1.05 | 115.2 | 1.07 | left (in-band loss, F5) |
| square.4096.fp16 | 904.2 | 918.4 | 0.98 | 822.3 | 1.10 | added |
| square.512.dynM | 11.2 | 11.3 | 0.99 | 10.8 | 1.04 | added |
| qkv.h4096 | 389.9 | — | — | 328 | 1.19 | seeded |
| o_proj.h4096 | 160.8 | — | — | 119 | 1.35 | seeded |
| mlp_gate_up.h4096 | 976.9 (greedy 1271.4) | — | — | 721 | 1.35 (greedy 1.77) | seeded pinned (F2) |
| mlp_down.h4096 | 397.5 | — | — | 388 | 1.02 | seeded |
| qkv.h4096.dynM | 394.7 | — | — | 330 | 1.20 | seeded |
| o_proj.h4096.dynM | 153.9 | — | — | 112 | 1.37 | seeded |
| mlp_gate_up.h4096.dynM | 975.9 (greedy **5978.1**) | — | — | 742 | 1.32 (greedy **8.06**) | seeded pinned (F1) |
| mlp_down.h4096.dynM | 412.9 | — | — | 388 | 1.06 | seeded |

Emmy still trails cuBLAS everywhere in this family (1.02–1.44×); the closest shapes are the K-heavy `mlp_down`
pair and `square.512.dynM`. No shape beats cuBLAS — the wins of findings-2 were the memory-bound skinny
projections and reduces, which are not in the matmul slice.

## Finding 1 — `mlp_gate_up.h4096.dynM`: greedy deploys a scalar tile 6× slower than the known-best mma config

The sweep's worst miss. `run --bench` deploys `TILE=n16x8/f4x26` (scalar, no mma, 8 % occupancy, 68 K smem) at
**5978 µs** vs eager 742 µs — while the same tune's own best -O3 measurement for this kernel is **987.7 µs**
(`w4x1/f4x8/k2`, reproduced pinned at 975.9 µs). The per-kernel ranking is *not* the problem:

- `eval variants --kernel 462b55`: the prior's pick marker sits on `w4x1/f4x8/k2` — the -O3-best config — rank
  17/106 by -O1 latency. Within the mma group the prior is right.
- The failure is one level up: the **structural/tier decision** (scalar vs warp-mma) flipped to scalar at deploy.
- Prime suspect: this exact shape ate 9 of the 17 compile-budget `bench_fail`s (4 s budget vs 4.1–5.4 s cicc on
  the N=28672 warp-mma kernels; the fix is this PR's budget bump). Each fail streamed a 2 000 000 µs sentinel row
  into `prior.json`'s training reservoir — dense negative labels on precisely the big-mma region of this shape.
  The DB fail rows were purged and re-tuned cleanly after the fix, but the reservoir keeps its sentinels (no
  surgical removal exists), so the learned model still carries them.
- `eval prior` reports `vs gold 1.01x` for this shape — the view compares knob configs within the matched kernel
  group and cannot see a tier-level misdeploy (see Workflow notes).

**Recommendation:** re-run the full matmul sweep `--clean` under the 12 s budget so the reservoir is rebuilt
without sentinels, then re-check this shape's deploy. Independently: `eval prior` (and any future CI golden
check) should compare the *deployed program* against the golden — kernel set + tier, not just knobs on the
matched kernel — so an 8× structural miss cannot report as 1.01×. The seeded golden now makes
`run --bench --golden matmul.mlp_gate_up.h4096.dynM` the one-command regression check.

## Finding 2 — `mlp_gate_up.h4096` (static): greedy takes an unmeasured split-K structure, 1.30× the fused best

Greedy deploys the `g2k` split (`w4x2/f4x8` partial + epilogue, 1271 µs total) over the fused `w4x1/f4x8/k2`
(976.9 µs pinned). `eval variants --kernel a8ecf3` shows the fused group's 88 measured configs with the pick on
the -O3 best — but the deployed `a8ecf3__partial` split kernel has **no measured rows at all**: the structural
fork was priced purely by prior extrapolation and won. Same shape family as Finding 1, milder symptom (it at
least stayed on mma), same root: the structural pick is the weak link, and sentinel-poisoned training data for
this family makes its extrapolations untrustworthy.

**Recommendation:** covered by Finding 1's re-sweep; additionally consider requiring at least one measured row
(or a one-shot bench) before a structural fork can displace a measured incumbent at deploy time.

## Finding 3 — `square.1024.fp16`: shallow rank, wrong pick (1.13×)

The largest live loss among recorded shapes. Greedy picks `w2x2/f2x4` (26.1 µs) over the golden `w1x2/f4x4/k2`
(23.0 µs) — `eval golden` m/t 1/2 (TILE wrong, STAGE right). But the learned prior ranks the golden **42/8890**
(analytic: 155) — the ranking is basically fine; greedy just prefers a nearby-worse sibling. Notably findings-2's
sweep *found* this exact golden config (recorded it at 22.2 µs); this sweep's prior — trained on a dataset now
dominated by the big h4096 fp16 shapes — drifted off it. A rank-42 config is well within reach of modest
exploration.

**Recommendation:** ε-greedy on the small-fp16 family (`--explore-eps`, as `collect-node-data` uses) or a small
patience bump; no analytic refit needed for this shape (rank 155 cold is workable).

## Finding 4 — `square.512.fp16`: the perennial mispricing (1.09×)

Third sweep in a row (findings-2 Finding 2). Greedy: `w2x2/f2x2/k2 + g2k + d2/cp/ring/p2`; golden:
`w2x1/f1x4/k2 + d4/cp/ring` — m/t **0/2**, every knob wrong. Ranks: analytic **833/9884**, learned **1932/9884**.
The learned prior improved on findings-2's 5536 but the tiny-fp16-tile regime remains the one family where both
priors put the truth outside any reachable neighborhood.

**Recommendation:** unchanged from findings-2 and now overdue — refit the analytic weights over the recorded
goldens (`scripts/golden_knob_heuristics.py`, its own cross-GPU PR) so the cold rank isn't 800+, and pair with
ε-greedy. The golden YAML keeps the old config; nothing to record.

## Finding 5 — `square.2048.fp16`: consistent in-band loss, no action

Greedy (`w2x2/f4x8`) lost to the golden (`w1x4/f4x4`) in all three runs (1.3 / 1.9 / 5.5 %). Median is inside
the 3 % category band, so formally "same" — but a config that never wins was not recorded as an alternate. The
golden's rank under the learned prior is 99/5692; a patience/exploration fix for Findings 3–4 likely covers this
shape for free.

## Workflow notes

Fixes from findings-2's notes: none landed — the eager row is still integer-rounded (the seeded `cublas_us`
values 112/119/328/388/721/742 are integers), `--golden` still can't seed a shape recorded only for another card
(the 8 h4096 seeds went through `run --bench -c` + `--dynamic`, fine once known), and the golden program total
still has to be summed by hand from the split-K rows. The warm-cache confirmation pass stayed cheap (seconds per
repeat) and greedy picks stayed within ±1 % across runs, as before — only golden re-bench rows swung.

New this sweep:

- **`eval variants --kernel` matches the generated C identifier, `tune --kernel` matches golden names.** The
  obvious `eval variants --kernel mlp_gate_up` returns "no measured variants"; you must first learn the kernel is
  `k_matmul_462b55` from a bench table. Fix: let the variants view accept golden-name substrings too.
- **`eval prior`'s `vs gold` cannot see structural/tier misdeploys.** It reports 1.01× for the shape that
  deploys 8.06× slower (F1), and 1.00× for F2's split-vs-fused miss, because it compares knobs within the matched
  kernel group. Fix: compare deployed kernel *sets* (program totals), which is also what `run --bench --golden`
  already measures.
- **`bench_fail` rows have no retry path.** The perf cache serves fails on re-runs (the cache key ignores the
  budget), so recovering from the 4 s-budget lockout took hand SQL (`DELETE FROM perf/node WHERE
  status='bench_fail' ...` after a WAL checkpoint + backup) before the re-tune. Fix: a `tune --retry-fails` (or
  budget-aware cache invalidation). Also: the reservoir in `prior.json` keeps the sentinel rows forever — only a
  `--clean` re-sweep truly clears them.
- **The compile budget was hard-coded** (`_tune_backend`, 4 s) and its overshoots were marginal (4.06–5.41 s,
  all on N=28672 fp16 mma kernels). This PR raises it to 12 s (wall 16 s). A `--compile-budget` flag would have
  avoided a code edit mid-sweep.
- **`run --bench` writes nothing to the DB or prior** — A/B and seed numbers exist only in the tee'd logs. Right
  behavior (deploy path stays read-only), but worth knowing: the YAML is the only durable record, so tee
  everything.
- **`eval prior` on an unrecorded shape prints "no golden shapes have tuned data"** — seeding is what unlocks
  prior evaluation, so seed first, evaluate after. With the 8 shapes recorded, the matmul-family eval now covers
  19 golden entries (median rank 12, top-10 for 9/19).
- **Recorded `emmy_us` (run --bench program total) and the reservoir's isolated -O3 rows disagree** by up to
  ~25 % on the small seeds (`o_proj` shows `vs gold` 0.77×/0.81× with *identical* knobs) — different bench
  contexts. Harmless for categorization (both sides of every A/B come from the same context) but confusing in
  eval output; worth a note in the eval header.
