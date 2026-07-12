# Golden sweep findings — RTX 5090, fast-math lane, second run (2026-07-10)

- **GPU**: NVIDIA GeForce RTX 5090 (sm_120), branch `feature/refit-analytic-fm-golden-retune`
- **Sweep**: `EMMY_FAST_MATH=1 EMMY_O3_TOL=0.10 emmy tune --dataset golden --clean` — 36 shapes, ≈2.6 h (vs 3.6 h
  on 07-09; the refit ranking converges faster). First sweep with all three 07-09/#343 improvements active: the
  post-rebase analytic refit (this branch — 15/16 golden matmuls rank #1 cold), #339's per-regime -O3 band, and
  #343's node plausibility gate. Delta-report against `golden-sweep-rtx5090-fm-findings.md` (the 07-09 sweep).
- **Tally (36 shapes)**: 1 fast-math `[fm]` entry added (square.4096 — the big one) / 1 standard replace
  (o_proj.h4096.dynM) / 13 WSPEC-underspecification stamps / 22 same / ~11 worse / **0 bench failures** (07-09: 1
  pathological gate-off deploy — the per-regime band fix held in the wild).
- Logs: `_tune/golden-sweep-rtx5090-fm2/` (tune.log, `ab/*.log`, ab_table.txt).

## Headline — square.4096 reaches the deep-bk f16-accumulate region: 0.77× vs cuBLAS

The 07-09 sweep's cold ranking never reached the `k4+` deep-bk f16acc rows on the big square (the PR #339 hand A/B
had found them at 1.35×). The refit weights reached them cold: the fm-lane greedy picked
`a:mma_m16n8k16_f16_f16/w2x2/f4x8/k4 d1/tma` at **491.9 µs** — 0.73× vs the standard golden's live 678, **0.77× vs
cuBLAS HGEMM** (≈279 TFLOP/s; the f32-accumulate hardware ceiling is ≈210). Reproduced 3× at <0.5% spread; recorded
as the shape's `[fm]` entry beside the standard one. This is now the fastest kernel emmy has ever produced on this
card for any golden shape relative to cuBLAS.

Also recorded: **o_proj.h4096.dynM** replaced (`w1x4/f4x2/k4 g4k d1/tma`, 91.4 µs, 0.96× vs the old golden's live
95.2 — reproduced in both lanes). The qkv/square.512.dynM `[fm]` entries from 07-09 re-verified (see Finding 2).

## Finding 1 — the per-regime -O3 band fix held; one NEW gate-off deploy miss (square.4096 std → scalar)

The 07-09 sweep's pathological gate-off deploy (qkv → per-cell scalar, bench_fail) did **not** recur — qkv's
gate-off greedy deploys a warp row at 278.7 µs (1.03× vs its golden; every lane benched clean). But the standard
lane's greedy for **square.4096** deploys a SCALAR register tile (`n32x8/f4x10 d1/tma`, 2919 µs, 4.3×) while the
fm lane's greedy on the same shape picks the f16acc row at 491.9. Same mechanism family as 07-09's qkv miss —
the gate-off enumeration lacks the fm winner and the deploy falls to model/evidence over the remainder — but this
time the standard best DID get its -O3 rebench (the band fix); the miss is in the deploy-side pick, not evidence
starvation. `matmul.square.512.dynM`'s gate-off greedy similarly regressed to a stage-less warp row (17.8 µs vs
the 4.2 golden; it picked `w1x4/f2x2/k4` with no STAGE). Both are `eval variants` / deploy-hierarchy follow-ups
on main, not this branch: the recorded goldens are unaffected (they replay by pin), and both shapes' goldens
re-bench true.

## Finding 2 — WSPEC is the third unpinned-family drift; every warp+TMA golden now stamps it

The recorded qkv `[fm]` entry re-benched at 309 µs (was 238) in the A/B. Root cause: the recorded knob set never
pinned `WSPEC`, and the refit weights flipped the greedy fill for the unpinned family from uniform to `p1` (a
producer-warp split) — worth −30% on this kernel. Pin-replay with `WSPEC=` (uniform) restored 235.9 µs exactly.
This is the same under-specification class as Finding 4's dynM `REDUCE` (07-09) — the third occurrence
(`REDUCE`, then the fill-order drift on gate_up, now `WSPEC`) — so the fix went wide: **every warp-TILE + TMA-stage
golden in the 5090 file now records `WSPEC: ''` explicitly** (13 entries stamped; WSPEC is only offered on that
combination, so the stamp set is exactly the eligible set). Post-stamp, both qkv entries replay at their recorded
speeds (258.6 std / 240.0 fm) gate-off. The durable recommendation stands and is now urgent for the other card
files: the recorder should stamp every resolved schedule family, and a post-record `--golden` re-bench should
gate the YAML edit (the 07-09 workflow-notes item).

## Deltas vs the 07-09 sweep (standard lane, live A/B)

Unchanged-within-noise: attention.hd128/hd128.dynM/hd64.dynM (0.98–1.02), all pointwise/rms_norm/softmax/reduce
kinds (1.00–1.05, the k2048 reduce pair still shows the flagged impossible golden row → `(*)` unreliable), the
mlp family (1.03–1.16 worse — the greedy still trails its goldens there), square.512 fp32 (1.06) and
square.512.fp16 (2.6× — unchanged; its `g2k` split golden stays unreachable cold, same as 07-09 Finding 3).
attention.hd64 static improved 1.29 → 1.13–1.15 (the refit helped, still a miss). The fm lane picked f32-acc
everywhere except square.4096 — including attention (PV stays f32-acc, latency-bound) and the small squares —
i.e. the tuner continues to adopt f16acc only where it measures a win, which is the intended shape of the fork.

## Workflow notes

- The two-regime golden A/B parse needed per-config grouping (a knob-less golden row = the preceding config's
  split-K finalize; the `_f16_f16` atom token = the fm regime) — the `golden NAME (total)`-row proposal from the
  07-09 notes would have made this trivial; it stands.
- The fm-lane categorization now compares against the shape's own `[fm]` entry when one exists (`[vs-fm]` rows) —
  the within-regime rule works end-to-end with recorded fm entries for the first time.
- Sweep wall time 2.6 h under the refit (down from 3.6 h): better cold ranking = fewer wasted deep benches.
  The `EMMY_O3_TOL=0.10` lever is now safe under gates (per-regime band).
