# Manual golden variant sweep — RTX 5090 (local), 2026-07-12

**Method:** NO tuner — per the session's direction (tune exploration untrusted after the deploy-evidence audit),
every shape was explored by hand: `emmy run --bench --golden NAME` plus batches of pinned `--ab` variants
(coordinate descent around the incumbents + targeted regions: big-tile f16-accumulate, split-K, staging depth,
WSPEC fills), all benched live against the recorded golden rows and the eager/cuBLAS reference in the same run.
6 waves + 2 confirmation passes per winner, ~120 pinned variant benches total, ~3 h wall. Repo @ `main`
`dfdc67a2`; logs and per-batch tables under `_tune/manual-golden-5090/`.

**Tally:** **12 fast-math `[fm]` entries added** (square.512.fp16, square.2048, square.4096 parity-add,
qkv.dynM, o_proj ±dynM, mlp_gate_up ±dynM, mlp_down ±dynM, attention.hd64, attention.hd128) / **2
standard-lane parity adds** (mlp_gate_up ±dynM) / **10 `REDUCE: ''` stamps** on under-specified entries / 0
replacements / all other incumbents confirmed best-known (square.512 fp32, square.1024, square.512.dynM,
qkv static, memory-bound kinds untouched).

## Headline — one tile family beats cuBLAS by 1.4–1.6× across the warp-tier set (FAST_MATH lane)

**`a:mma_m16n8k16_f16_f16/w4x2/f4x8/k4 · d2/tma/ring` (+`g2k` where the un-split grid underfills the 170 SMs)**
is the best-known config on 7 of the 10 fp16 matmul shapes. Mechanism: the f16 accumulator **halves register
pressure**, so the `w4x2/f4x8` register tile — 242 regs/thread, unbuildable in the f32-accumulate lane — fits,
and its 4× larger output tile amortizes everything else. This is a *joint* (atom × geometry) move: the earlier
fm sweeps only atom-swapped at fixed geometry (the fm2 `square.4096` deep-bk find was the lone exception), so
the region was never measured.

Per-shape outcome (µs are totals incl. split-K finalize; golden = live replay same run; all winners 3×
reproduced at <1% spread; `run` accuracy vs torch passes on the gate_up / mlp_down winners):

| shape | new best (fm lane) | µs | golden live | cuBLAS µs | vs cuBLAS |
| --- | --- | --- | --- | --- | --- |
| square.2048 | w4x2/f4x8/k4 | **61.3** | 90.7 | 98.36 | **0.62** |
| square.4096 | w4x2/f4x8/k4 | **481.1** | 494.8 (recorded [fm]) | 640.93 | **0.75** |
| o_proj.h4096 | w4x2/f4x8/k4 g2k | **64.8** | 90.5 | 95.22 | **0.68** |
| o_proj.h4096.dynM | w4x2/f4x8/k4 g2k | **64.9** | 91.9 | 95.08 | **0.68** |
| mlp_gate_up.h4096 | w4x2/f4x8/k4 | **369.7** | 616.1 | 557.94 | **0.66** |
| mlp_gate_up.h4096.dynM | w4x2/f4x8/k4 | **370.1** | 618.3 | 553.70 | **0.67** |
| mlp_down.h4096 | w4x2/f4x8/k4 g2k | **201.3** | 295.5 | 296.95 | **0.68** |
| mlp_down.h4096.dynM | w4x2/f4x8/k4 g2k | **201.8** | 297.4 | 295.03 | **0.68** |
| qkv.h4096.dynM | w2x4/f2x4/k8 d1/tma | **240.6** | 251.0 | 249.82 | 0.96 |
| qkv.h4096 | (recorded [fm] stands) | 240.1 | — | 250.54 | 0.96 |

(vs cuBLAS = emmy/cuBLAS, <1.0 = emmy faster.) The dynM twins pay **no masked-tile penalty** in this family
(369.6–370.4 dynamic vs 369.2–370.0 static). Where the family loses: square.1024 and smaller (the 256-wide
output tile underfills the card — 20.7 vs 15.6), and qkv-class wide-N shallow-K (parity with the k8 entry).
FAST_MATH's default-off stance is unchanged — these are `[fm]` entries beside the standard goldens, never
replacing them.

Standard lane: the same geometry with the f32-acc atom (`w4x2/f4x8/k4 d2/tma/ring`, 166 regs) is a parity-add
on mlp_gate_up ±dynM (594.9 / 595.6 vs recorded 610.6, ~2.5% — under the 3% replace gate, 4× reproduced).
Nothing else in the standard lane beat its incumbent beyond noise.

## Finding 1 — why every tuned sweep missed this region (and what to fix)

`emmy eval analytic` on the updated goldens ranks **every new entry at rank 0** (median 0, top-1 27/28) — the
post-#347 analytic prior prices the big-tile f16acc family correctly, so the cold ranking is NOT the blocker.
The culprit is the **`-Xcicc -O1` tune ranking lane, CONFIRMED by direct measurement**: replaying the recorded
entries under `EMMY_NVCC_FLAGS="-Xcicc -O1"`, the big-tile fm config is **5.1× slower than the standard golden
on mlp_down** (2008 vs 392 µs) and **4.7× slower on square.2048** (591 vs 126) — the same configs that are
~32% *faster* at -O3. cicc at -O1 doesn't schedule the big unrolled register tiles (the very blowup the -O1
test lane exists to dodge), so the family ranks dead last in the tune's -O1 ordering and the `EMMY_O3_TOL`
band never grants it a deployable -O3 rebench. A ~5× *systematic inversion*, not noise.
**Recommendation (high priority): the -O3 rebench band needs a family-aware floor — always rebench the top-K
per (atom, tile-size) bucket, not just the global -O1 top band** (or rank the widest register tiles on -O3
directly). The learned prior inherits the censoring either way; the node store has no measurements in the
region, and the goldens now seed it.

**Analytic refit: attempted and REJECTED.** Per the skill, `scripts/golden_knob_heuristics.py --out
<candidate>` was re-run over the updated goldens; the candidate A/B'd worse than the checked-in weights
(`eval analytic --analytic-file`: top-1 20/28 vs 27/28, `mlp_down.h4096.dynM` std dropping to rank 3449).
The current `analytic_weights.json` already ranks all new entries #0, so it stands unchanged.

## Finding 2 — `REDUCE` is the fourth unpinned-family drift; every matmul entry now stamps it

`matmul.mlp_gate_up.h4096` (static) recorded no `REDUCE` key; today's greedy fill adds `g2k`, so the golden
replayed at 652.6 (625.1 + 27.5 finalize) vs its recorded 610.6 — a 7% phantom regression, the same class as
the dynM-`REDUCE` / fill-order / `WSPEC` drifts (#345). With `REDUCE=` pinned the entry replays true (616.1).
Every 5090 matmul entry lacking the key is now stamped `REDUCE: ''` (verified value-preserving by this
session's replays); post-stamp the file replays clean end to end. The recorder-side fix (stamp every resolved
schedule family at record time) remains open and is now urgent for the other card files — this is the fourth
occurrence of the class.

## Finding 3 — `square.512.dynM [fm]` replay drift RESOLVED: another `REDUCE` fill, now stamped

The recorded fm entry (3.6 µs) replayed at 4.2 total because the unpinned `REDUCE` drifted into a `g2k` fill
(3.2 main + 1.0 finalize). With `REDUCE=` pinned off it benches **3.6–3.7 across 4 passes — exactly its
recorded value**: the fm2 sweep measured it un-split, and the fill drift is Finding 2's class again. The entry
is now stamped `REDUCE: ''` and replays true. (The std entry's recorded 6.14 vs live 4.5 remains a
small-shape stale-`emmy_us` oddity — the config is unchanged and still golden, left as is.)

Bonus small-shape coverage: `square.512.fp16` (static) gained its first `[fm]` entry — the atom-swap of its
standard golden (`f16_f16/w1x8/f4x1/k4 g2k d3/tma/ring`) at **3.9 total vs the standard's live 4.4** (0.89×,
1.57× vs cuBLAS), 3× reproduced at zero spread — mirroring the dynM twin's existing fm win.

## Finding 4 — attention: PV-only f16acc wins ~11–13% on the static shapes; the masked-flash path can't record it

Swapping only the P·V contraction (`TILE@pj`) to the f16-accumulate atom — QK^T stays f32-acc for the
online-softmax statistics — wins on both static attention goldens: **hd64 7.2 vs 8.3 (0.87×, 1.39× vs torch
SDPA)** and **hd128 14.6 vs 16.4 (0.89×, 1.26×)**, each 3× reproduced at zero spread, accuracy vs torch
passing. Both recorded as `[fm]` entries. The both-contractions swap is refused by the flash form narrowing
(the dd pin realizes f32-acc, integrity-flagged) — correct behavior, softmax stats need f32.

**The dynM twins cannot record the same win**: dynamic attention goldens are schema-required to record a
single bare `TILE` (the masked-flash pin doesn't resolve `TILE@<axis>`), and a bare f16acc `TILE` pin
degrades to the **scalar fallback** (18.5 ms / 9.7 ms — the flatten pathology). So the f16acc PV sibling is
unreachable on the masked-flash path both by pin and (presumably) by search. **Recommendation:** teach the
masked-flash form narrowing to accept the f16acc atom on its PV contraction (or make the bare-TILE pin
resolve per-contraction atoms), then mirror the two static wins to hd64.dynM / hd128.dynM — the deployable
artifacts models actually run.

## Finding 5 — deploy-side confirmations (audit follow-ups, not this task's scope)

Greedy deployed above the golden on nearly every shape this session (qkv 279.7 vs 259.1, qkv.dynM 273.8 vs
251.6, mlp_down 305–311 vs 295, gate_up 633–636 vs 616, downdyn 340 vs 297), with repeated disjoint-evidence
warnings (`square.1024`: "181 measured rows, none matches the 4 offered candidates"). Root causes were
established by this session's earlier audit: the local DB is -O1-only and fm2-skewed (regime starvation), the
-O3 reservoir covers only slow families for some shapes, and the learned prior extrapolates poorly (median
golden rank 3206 vs the analytic's 0). The newly recorded entries make the golden dataset the reliable
source of truth for these shapes regardless of the deploy path chosen later.

## Workflow notes

- **The `--ab` A/B harness is excellent for manual sweeps**: ~8–12 pinned variants per invocation, live golden
  + eager rows as controls in every batch, and `_unreproducible_pin_flag` caught every declined pin (d3/d4
  stage pins clamping to d2 on 96K-smem tiles; a stage pin realizing `(off)`). Zero silent degrades reached
  the results.
- **Pin `REDUCE=` (and `WSPEC=`) explicitly in every `--ab` row** — an unpinned family gets planner fill
  (wave 1 lost 3 rows to surprise `g2k` fills, +2.4–77 µs finalize noise in the comparison).
- The split-K finalize prints as a separate knob-less row after its config — totals must be summed by hand.
  The `golden NAME (total)` row proposal from the fm2 notes stands; it would also fix per-config grouping in
  these logs.
- `emmy run` accuracy checks are silent on success (exit 0, no output) — a one-line `accuracy: PASS (rel err
  …)` would save a foreground rerun to confirm the check actually ran.
- Confirmation economics: cubin caching made the 19 confirmation invocations ~1–2 min each; the expensive part
  was first-compile of each variant (~30–60 s for the h4096/4096² kernels). Total ~120 variants ≈ 2 h GPU.
- The disjoint-evidence deploy warning prints once per `--ab` row compile (11× per invocation on square.1024)
  — it should dedupe per (node, compile session).
