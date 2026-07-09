# Deploy evidence tiers: root cause, fix, and verification — RTX 4090, 2026-07-08 (PR #326)

Standalone report of the deploy-correctness work that grew out of the ninth 4090 golden sweep (findings-3, since
pruned; its sweep half is superseded by `golden-sweep-rtx4090-findings-4.md`). Scope here: why `compile`/`run`
shipped configs the tune had measured slower, the four-commit fix, its live verification, and the golden updates
the fixed deploys earned. GitHub artifacts: PR #326; issues #327 (lowering bug, fixed by #330) and #328 (tune
numerics gate, open).

## The failure

The ninth sweep's seeded `mlp_gate_up.h4096` goldens exposed greedy misdeploys: **5978 µs (scalar tile, 8× eager)**
on `.dynM` and 1273 µs (split) static, while the tune's own best -O3 measurement for both was ~977 µs — a config
the prior itself ranked #1 within its kernel group. Both "prior shortfall" fp16-square findings of the sweep
(`square.512.fp16` 1.09×, `square.1024.fp16` 1.13×) turned out to be the same failure.

## Root cause — both measured-evidence tiers of the deploy hierarchy were silently dead

Found with a decide-level probe (spies on `_db_measured_pick` / `Prior.evidence_pick` over a live compile):

- **Vocabulary drift killed the joins.** #311 added the `S_warp_eligible` scheduler stamp to deploy-time candidate
  rows; no persisted perf row and no reservoir row carried it, and both evidence joins matched `S_*` signatures by
  strict frozenset equality — one added key disabled the reservoir -O3 tier AND the DB tier against every existing
  store. Every deploy was a pure model argmin.
- **The deployable DB lane didn't exist.** All perf rows live under the -O1 ranking-lane context key (the tune's
  -O3 re-benches reach only the node table and the reservoir). With drift fixed but the reservoir tier still dead,
  -O1 medians decided deploys and their -O1/-O3 inversions regressed qkv (385 → 466 µs) and mlp_down (391 → 456 µs)
  until the lane preference + reservoir fix landed.
- **Mechanism correction:** the `g2k` split is the `REDUCE` knob at the partition fork, not a structural Graph
  splice — `_pick_structural` (initial suspect) never fires for these shapes. Prediction-only structural pricing
  (#222/#223) is real but was not the trigger; it received the same evidence grounding as a byproduct.

Origins: #222/#223 (2026-06-10) made deploys prior-driven with the DB consult deliberately removed; #304 added the
knob-level evidence lane; #311 (2026-07-07) broke both tiers the day it merged — unnoticed because the join fails
silent (model fallback) and no shape could contradict the deploy until #320 seeded goldens with a large
fused-vs-split margin. #322 later fixed the stamping side (rows and candidates share a vocabulary going forward);
the tolerant join remains necessary for stores recorded before any given stamp existed.

## The fix (PR #326, four commits)

1. `2c94ec15` — structural pricing probes receive the deploy's `db` (evidence hierarchy inside `_pick_structural`'s
   nested resolves).
2. `74061b9f` — drift-tolerant `S_*` join for the DB tier: exact hit first, else agreement on all shared keys
   (one-sided keys are vocabulary drift, not shape identity; empty overlap matches nothing).
3. `e988654e` — the DB index spans the deploy/-O1/-O3 context twins; deployable-lane rows decide outright, -O1
   ranking rows only when nothing deployable matched.
4. `6820b381` — the same tolerant join for the reservoir tier, shared as `Prior.sig_groups`. The live reservoir
   tier preempts the -O1 fallback.

## Verification (live golden A/Bs, 2 runs per shape at each step)

| shape | pre-fix deploy | post-fix | recorded golden |
|---|---|---|---|
| mlp_gate_up.h4096.dynM | **5978** (scalar) | **922.6** (golden's config) | 923.6 |
| mlp_gate_up.h4096 | 1273 (split) | 922.6–964.6 (golden's config) | 964.6 |
| square.1024.fp16 | 26.1–30.7 | 22.1 (golden's config exactly) | 22.2 |
| square.512.fp16 | 7.3–7.8 | 6.1 (new config, beats the golden's live 6.7) | 6.1 |
| square.2048.fp16 | 118–129 | 116.0 (golden's config) | 116.0 |
| qkv.h4096 / .dynM | 385 / 395 (466/416 during the interim regression) | 374.1 / 370.0 | 374.1 / 370.0 |
| o_proj.h4096 / .dynM | 161 / 154 | 121.6 / 123.3 (−24 %) | 121.6 / 123.3 |
| mlp_down pair | at golden | at golden | unchanged |

Every shape the sweep classed "worse — prior shortfall" deploys its golden's config or better: they were deploy
bugs, not search misses. A post-fix ε-tune of the 512 family then found `square.512` → `n16x8/f4x8 + g2k +
d2/cp/ring` at **10.1 µs = 0.94× cuBLAS — the first fp32 square at/under cuBLAS on this card** (3/3 runs), and
`square.512.dynM` → 10.8 µs (independently reproduced at 10.6 by the findings-4 sweep; the YAML records 10.6).

## The fast-but-wrong incident (found by the same ε-tuning)

An ε-tune of the gate_up pair reached `TILE=a:mma_m16n8k16_f16/w4x4/f4x8/k8`, which lowers broken: `TypeError`
(unlowered `TileOp` partial) under `g4k`/`d1/cp`, and a **launchable kernel measured at ~104 µs** — >1
PFLOPS-equivalent, physically impossible — under `g2k`/`d1/cp`. The tune has no numerics gate, so the garbage
measurement became the shape's `ok` "best" (silly rates 93–99 % were the tell). Contained by architecture luck:
the bogus rows never reached the perf table and never got an -O3 re-bench, so the deployable tiers stayed clean
and no wrong deploy shipped. 11 bogus node rows purged; 36 -O1 reservoir rows remain as minor training noise.
Filed: #327 (lowering — since fixed by #330: over-budget warp stages declined, un-lowered terminals bench_fail)
and #328 (numerics gate — open; post-#326, measurement truth is a deploy-correctness assumption and nothing
enforces it).

## Follow-ups

- **#328**: verify output vs torch before a measurement becomes evidence (new per-shape bests + -O3 re-bench
  candidates — exactly what deploys consume).
- **Record the tune's -O3 re-benches as perf rows** — the deployable DB lane added by commit 3 is empty until
  `two_level` writes them; today the reservoir is the only deployable-truth store.
- **`emmy eval deploy --explain <shape>`** — productize the decide-level probe (which evidence tier fired, key
  diffs on a join miss); it answered in minutes what three blind A/B rounds could not.
- Merge note (this branch): the pruned-alternates policy was applied when combining with findings-4's additions —
  parity alternates kept (`square.512.fp16` g2k @6.3, `qkv` g2k @380.3), superseded-slower ones pruned
  (`square.512` @12.7, `square.4096` @2819.1, `square.1024.fp16` @25.7 — 3.3–26 % behind the confirmed entries),
  and the `square.512.dynM` twins collapsed to one entry at the lower independently-reproduced 10.6 µs.
