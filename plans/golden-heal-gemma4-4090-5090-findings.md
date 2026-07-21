# Gemma-4 golden heal: 4090 dynM drift + full-coverage seeding on both cards (2026-07-20)

**Cards:** RTX 4090 (rented CloudRift box, sm_89, CUDA 12.9) and RTX 5090 (kenshin, sm_120, CUDA 13.0).
**Method:** manual pinned sweeps only — `emmy run --bench -c <snippet> --ab <knobs>` batching a curated grid per shape
(std + fm atom pins in one run), harvest-then-pin for the greedy pick, one confirm re-pin per winner. `emmy tune` was
NOT used (broken per operator instruction). Driven by the in-model drift audit (`emmy eval golden --in-model`): the
work list is exactly the audit's DRIFT + GAP keys, and the exit criterion is DRIFT 0 / GAP 0 on both cards.
**Wall time:** ~4.5 h total GPU time (4090: 22-pin dynM sweeps ~70 min, 39-job coverage sweep ~3 h, confirm ~1 h;
5090: 19-job sweep + confirm, run in parallel).

## Finding 1 — the 9 dynM DRIFTs were a layout class, not stale configs

Every drifted 4090 entry (q/kv/o/mlp_down + global twins, std and fm lanes) records a staged `d2/cp[/ring][/p2]`
config. Two independent proofs of the root cause:

- **Canonical replay is perfect**: `run --bench --golden gemma4_12b.q_proj.dynM` on the box realizes every pin and
  greedy reproduces the std golden knob-for-knob (103.7 µs, recorded 101.9 — within noise). The entries are healthy
  on the layout they were tuned on.
- **The serving layout declines staging**: in every serving-layout (`F.linear`, trans_b) sweep, the two `d2/cp` probe
  pins came back `pin_unmatched` while all 20 gmem-direct pins realized — cp.async transports decline transposed B,
  exactly as `MatmulGoldenConfig.trans_b`'s docstring records. The canonical-tuned entries therefore can never
  realize on the forks serving actually compiles → DRIFT.

**Correction from the 5090 sweep: the decline is universal, not cp.async-specific.** On kenshin the `d2/tma/ring`
pins ALSO came back `pin_unmatched` on the trans_b forks (12 of 12 tile combinations on `q_proj_global.m256.lin`).
The 5090 never drifted only because its golden files already carried *realizable* siblings under the same ShapeKeys —
the fastest-realizable-first coexistence doing its job, while the staged entries silently never match. Heal:
`.lin.dynM` sibling entries (trans_b: true, gmem-direct) recorded beside the canonical twins (which stay, per the
schema's ordering-protection note — a `.lin` gmem config realizes on canonical forks too and must never be the
fastest entry there).

## Finding 2 — fast-math loses on the serving layout at large M, wins at decode M=32

On gmem-direct trans_b shapes the fm (f16-accumulate) winner lost to std everywhere at M≥256 (mlp_down.dynM: 1082 vs
784 µs; q_proj.dynM: 214 vs 197) — without a staged ring these kernels are B-bandwidth-bound and the atom's compute
rate is irrelevant. At M=32 (decode) several fm configs won (small tiles + deep splits). Recording follows the
skill's rule — fm entries only where they beat std — so the dynM heal records std only, and the m32 statics record
fm siblings where measured faster.

## Finding 3 — serving-layout matmuls on sm_89 run ~1.9–2× behind cuBLAS

Best-of-grid vs the same run's live eager (cuBLAS HGEMM), symbolic-M hint 512: q_proj 197.3/105.4, kv_proj
106.7/49.8, o_proj 207.0/107.9, mlp_down 784.1/394.6, q_proj_global 409.1/202.6, k_proj_global 39.4/17.6,
o_proj_global 419.0/206.0. This is the price of gmem-direct A/B on a bandwidth-bound shape. The goldens pin the best
available config; the actual perf lever is enumeration-side — **teach the cp.async transport transposed-B staging**
(or a B-transpose epilogue/pre-pass) so sm_89 serving forks regain a prefetch ring. Until then this gap is the honest
serving floor on 4090-class cards and the recorded `cublas_us` values make it visible per shape.

## Finding 4 — coverage was the bigger hole, and the audit's GAP view found it mechanically

Beyond the 9 drifts, the audit listed 39 uncovered fork keys on the 4090 and 19 on the 5090 — static-bucket
projections (M=32 decode / M=256 prefill), the fused GeGLU family, all rms_norm shapes (row and per-head), and the
f16 pointwise maps. Two schema gaps blocked closure and were fixed in this branch: `RmsNormGoldenConfig` needed a
`heads` axis (a dynamic per-head q/k-norm keeps heads static beside the symbolic token axis — a 2-D snippet's
dynamic key can never join), and `PointwiseGoldenConfig` was fp32-only (the model's pointwise forks are f16,
`is_warp=True`). Every manifest was generated FROM the golden dataclasses and key-diffed against the audit baseline
before sweeping (`gaps NOT covered: NONE`), so closure is by construction, not hope.

## Finding 5 — the GeGLU fused/cut decision differs per card and per M, and the cut opens new coverage

4090 (pinned-comparable): m32 keeps the FUSED form (329.7 vs cut 515.8 — small-M shares one A compute-fill across
both channels); m256 and dynM take the CUT (416.3 vs 819.2 fused; 1098.3 vs 1843.2). 5090: m32/m256 already
recorded (fused / cut respectively, prior sessions); dynM now takes the cut too — the fused sweep best (1084.2,
`d2/sync`) is within 8% of the crude pinned cut (1177.6), and cold greedy on the fused dyn snippet is broken
outright on sm_120 (Finding 6). A `.cut` golden splits the megakernel, so its FRAGMENTS (per-channel `(M, inter)`
matmuls, the `__stat` mean-square reduce, the fp32 rsqrt-scale and gelu-combine maps) become new forks — the audit
loop surfaced them as new GAP keys and a second fragment sweep closed them. `ReduceGoldenConfig` needed the
symbolic-axis exclusion in its dynamic key to join the dyn `__stat` fragment (`free_prod=M` could never match).
The layout class then bit the session itself: the first fragment recording used the canonical-snippet winners
(staged `d2/cp`), which promptly DRIFTed in the twins — the serving cut's channel matmuls are trans_b like every
other Linear. The re-audit caught it within minutes and gmem-direct `.lin` siblings fixed it; a good demonstration
that the audit loop, not care, is what makes recording safe.

## Finding 6 — cold greedy miscompiles the dynamic fused GeGLU on sm_120

Two independent cold compiles of the dyn GeGLU snippet on kenshin: one deployed a 41,257 µs kernel set (eager:
597 µs — the misdeploy hazard live), another failed the accuracy probe with `max_diff=64, mean_diff=nan` (the run
aborts, correctly, before benching). The NaN is intermittent — the same snippet passed accuracy in the sweep run
minutes earlier — pointing at uninitialized memory or a race in a fused computed-A dyn variant. The `.cut` golden
sidesteps the broken form entirely; the flake deserves its own investigation (not in this session's scope).

## Finding 7 — authoritative pins can win with configs the planner can't reach

The sweep grids included hand-guessed tiles; several WON their shape (e.g. `w2x2/f1x2/k2` on `mlp_down.m32.lin`) but
are not members of the enumerated move grids — an authoritative pin realizes them, but a golden recording one can
never MATCH an offer at deploy (permanent drift by construction). `test_golden_knobs_are_members_of_the_move_catalog`
caught every one at recording time; the extraction now filters winners to catalog members (mirroring the test's
per-kind scope). The rejected winners are also a free enumeration-gap signal: each is a measured-faster config the
move grids currently exclude.

## Per-shape outcomes

Recorded entries live in `rtx4090_sm89_gemma4.yaml` (48: 7 `.lin.dynM` heals, 14 static `.lin` matmuls, 13 rms_norm,
9 pointwise, fused m32 + m256.cut + dynM.cut, then the cut-fragment entries) and `rtx5090_sm120_gemma4.yaml` (19+
fragments: 2 `.lin` matmuls, 6 rms_norm, 10 pointwise, dynM.cut). Every winner re-benched once in a confirm pass;
zero drifted beyond the ~13% noise band. The sweep JSONs with every pin's µs are the session artifacts under
`_tune/heal-4090` on the rented box and `_tune/heal-5090` on kenshin. Exit state: `emmy eval golden --in-model`
reports DRIFT 0 / GAP 0 on both cards, and the CI gate enforces empty baselines thereafter.

## Workflow notes

- **The audit is the work list.** DRIFT/GAP keys → manifest → sweep → record → re-audit-to-zero closed the loop with
  no judgment calls about "what runs in the model"; the two schema extensions fell out as type errors of that loop
  (a gap key no golden kind could construct).
- **Harvest-then-pin makes the greedy pick a fair candidate**: round A unpinned, harvest `record_knobs` off the
  greedy kernel, round B re-pins it beside the curated grid — so the recorded winner is always pinned-comparable
  (the greedy row itself never is: it benches interleaved with the torch reference).
- **`--ab` batching is the right sweep engine**: 22–28 pins per invocation amortize trace/compile setup; the
  realized-vs-pinned gate turns enumeration questions into data (`pin_unmatched` on the staged probes was the
  root-cause proof, free of charge).
- **`--json` schema friction**: `backends` is a name→stats dict while `pinned` is a list; `record_knobs` lives per
  kernel, not per row; pin grammar rejects an empty `PLACE=` (enum) while empty codec pins (`STAGE=`) are meaningful
  OFF — each cost one iteration. A documented example record would have saved all three.
- **fm/std lanes in one run worked as documented** — atom pins bypass the gate, `lane` tags each row, no env needed.
- **The `--ab` pin gate cannot verify a `PLACE@cone=cut` pin**: post-split, no realized kernel carries the
  `PLACE@cone` stamp (the fused form stamps `fuse` fine), so the row reports `pin_unmatched` and never benches —
  while the same pin via `EMMY_KNOBS` realizes the cut perfectly. Cut totals were measured via env-pin + a benign
  `--ab` tile row to force the pinned bench path, or the `greedy (isolated)` twin. Either the cut should stamp its
  placement on the fragment kernels, or the pin gate should verify structural pins against the resolve trace's
  `Decision`s instead of kernel knobs.
- `emmy tune` unavailability (operator-reported breakage) was not a blocker for coverage-class tuning: curated grids
  of 20–30 pins found consistent winners (the same tile family won every large shape per card). It WOULD bite on
  shapes needing deep schedule exploration (flash, WSPEC, raster interactions) — none were in this session's scope.
