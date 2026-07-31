# Serving-config golden coverage audit + MTP m192 seeding — RTX 5090, 2026-07-31

Brief: verify every kernel the gemma-4-12B serving configurations deploy has a golden in
`rtx5090_sm120_gemma4.yaml` — with particular attention to the MTP lanes (`serving_mtp_rtx5090`) and the
context bump to 16384 — and seed whatever is unseeded. Method: the in-model audit (`search/audit.audit_card`
over weight-free twins), extended beyond its stock width list with the MTP c=64 verify bucket (192), then
manual pinned `--ab` seeding on the local 5090 (the manual golden method — no tuner sweep).

## The serving width inventory (what "all kernels from the serving configurations" means)

Emmy-side kernel shapes in generative serving are decided ONLY by the step width (`num_tokens`), never by
context length. The deployed widths across `serving_rtx5090`, `serving_mtp_rtx5090`, `gsm8k[_mtp]_rtx5090`
and the baked image (`models/gemma-4-12b-it.env`):

| width | source |
| --- | --- |
| m1 | `EMMY_GEN_M1_TIER` gemv twins (true single-token decode) |
| m8 | decode bucket, c=4/c=8 lanes; MTP c=1 lanes (verify widths 3/4/6 pad into it) |
| m32 | default decode bucket (baked image); MTP c=4/c=8 depth 2–3 (verify widths 12–32) |
| m64 | c=64 decode bucket |
| **m192** | **MTP c=64 depth-2 verify bucket** (64 × (2+1); pinned by the quality gate — 256 fails GSM8K) |
| m2048 | chunk quantum (`EMMY_GEN_PREFILL_BUCKET=2048`), 4k/4k batched lanes |
| m4096 | default chunked-prefill width (`--max-num-batched-tokens` chunk) |
| dynM | the symbolic programs (everything between rungs; over-bucket capture) |

**The 16K context bump changes none of these.** `context_length: 16384` moves vLLM-side artifacts only
(KV pool, paged attention, rope tables); the twins are `[num_tokens, H]`-parameterized and never see
`max_model_len`. What 16384 does change: the pack/cubin cache key (the baked image's `.env` is already
flagged UNVERIFIED at 16384 — the headroom sweep + re-warm remain open, out of scope here).

## Audit result (before edits)

`audit_card` over the full twin set + the 192-width twins, RTX 5090: **MATCH 138 / DRIFT 0 / GAP 77 /
compile_fail 0**. m32 and m256 audit clean (zero gaps). The gaps split three ways:

1. **MTP m192 majors (NEW, seeded below):** `o_proj` (M=192, N=3840, K=4096) and `o_proj_global`
   (K=8192) — the only warp-contraction hazards at the MTP c=64 bucket. The 2026-07-30 m192 hand sweep
   covered the merged QKV/gate⊗up matmuls and the fused cones but missed the attention output projection,
   and the stock audit never traced width 192, so nothing could see it.
2. **The #446 fused burn-down (known, closed below):** the PLACE-knob retirement commented out every
   fused computed-A golden with a `PLACE@` spelling, reopening the fused norm→linear / down forks at
   m1/m8/m64/m2048/m4096 (18 forks; already tracked in the drift-gate ratchet).
3. **Aux keys (left open, consistent with the ratchet):** per-head qk-norm rms sweeps, cut-stat rms rows,
   merged-cat dup-view glue — the greedy-near-optimal class the baseline already accepts at other widths.
   The m192 members joined `EXPECTED_GAPS`.

## What was recorded

| shape | config | emmy µs | eager µs | ratio |
| --- | --- | --- | --- | --- |
| o_proj.m192.lin (std) | `w2x4 f2x2/k2 g4a gm8 d2/tma/ring` | 41.8 | 47 | 1.12× |
| o_proj.m192.lin (fm) | `w2x4 f2x2/k4 g4a gm8 d2/tma/ring` | 37.9 | 47 | 1.24× |
| o_proj_global.m192.lin (std) | `w2x4 f2x2/k2 g8a gm8 d2/tma/ring` | 71.9 | 74 | 1.03× |
| o_proj_global.m192.lin (fm) | `w2x4 f2x2/k2 g8a gm8 d2/tma/ring` | 71.3 | 74 | 1.04× |
| 14 fused-fork cut routings | `PLACE: cut` | (per-row) | (per-row) | see YAML |

The o_proj rows follow the m192 tile rule from the 2026-07-30 sweep (tile-M-32/64 + a cross-CTA atomic
split sized to fill the card + `gm8`): the m64-family `w1x8/k2/g8k` two-kernel split benches 48.2 / 85.7
total against the winners' 41.8 / 71.9. Deployability confirmed unpinned — the greedy pick IS the
recorded config on both shapes (the g8a/gm8-at-one-tile "benches but cannot deploy" trap from the
2026-07-30 workflow notes did not bite here).

## Finding (CORRECTED 2026-07-31 evening) — the routing consult mis-keys stat-free computed-A cones

The morning's two findings ("the staged computed-A form no longer realizes in the isolated snippet"
and "the down cone's cut routing is in-model-inert") were **one defect, misdiagnosed twice**, found
properly during the article-repro serving A/B (emmy lanes −22…−32% vs published with an exact stock
control; the pack diff showed the published image cutting the m4096 down into cone + big-tile matmul
where the re-baked image deployed one fused computed-A kernel):

- `_cut.py::_routing_entry` (the #446 pre-fork routing consult) keyed the kernel with a raw
  `ShapeKey.from_s_features`, whose histogram classifier can only fire `kind="fused"` off a
  top-level `rsqrt`. A **stat-free** computed-A cone — the geglu→down edge — keys `kind=''` and
  never joins its `fused`-keyed `.cut` routing entries, so the down could not cut **anywhere**
  (in-model or snippet) and deployed the fused computed-A form at every width: the m4096/m2048
  chunk-prefill TTFT regression and the bucket-8/64 TPOT regression. `greedy._fork_shape_key`
  documents the fused-kind convention and rebuilds from the offer's sync-STAGE signal; the routing
  consult had no such rebuild. **Fixed**: `_routing_entry` now rebuilds the key off the routing
  tree's computed-A edge (`_has_computed_a` — the structural twin of the offer signal).
- The "snippet realization loss" was the same consult from the other side: for the **rms** cones the
  histogram does fire `fused` at routing time, so the `.cut` sibling routes **pre-fork in the golden
  replay too** — the fused form then never exists for a `STAGE=d1/sync` pin to realize against, and
  the replay reports `unreproducible pin … realized (off)`. The staged computed-A tier was never
  broken. Residual eval-side limitation, still open: a fused *schedule* row cannot be replayed by
  name while a `.cut` sibling exists (the replay would need an implicit `PLACE=fuse` pin for
  schedule-row goldens); until then fused-row µs can only be refreshed by temporarily removing the
  sibling.

Closure: the four down `.cut` rows re-recorded (snippet totals were valid), `pw.n15360` tiers added
at m64/m192/m2048/m4096 (the geglu-materialize piece's key; m8/m32/m256 were already covered), and
the drift-gate ratchet tightened — the down-fused keys are now enforced covered, which is the
regression test for the routing key: if it mis-keys again, the fused fork reappears and the gate
goes red.

## Fused-fork cut routings (the #446 closure that IS possible)

The 14 reopened norm→linear forks (norm_qkv / norm_qk_global / norm_gate_up at m1/m8/m64/m2048/m4096)
now carry `PLACE: 'cut'` routing rows; pieces resolve each width's seeded stat/scale/matmul tiers, the
cut demonstrably fires in-model (the m8 audit's gate_up node resolves as stat + plain merged matmul),
and the corresponding ratchet lines are deleted. Notable totals (pieces summed, same-run eager): cut
beats eager outright at the decode widths — m1 norm_qkv 15.2 vs 47 (3.1×), m8 norm_qkv 20.3 vs 21 —
and lands within noise at m64/m2048/m4096 except the std-lane wide gate_up (the FP32-accumulate
half-rate wall; the fm piece tier covers those widths). Full per-row numbers in the YAML.

## Gate/audit changes

- `twins.py`: width 192 joins the audit width list — the MTP bucket was a deployed width the audit
  could not see (the exact blindness class the 2026-07-24 width extension note warns about).
- `test_golden_drift_gate.py`: 5090 baseline — 15 fused lines deleted (closed by the cut routings +
  m192 seeding), the 4 down-fused keys kept with the inert-cut explanation, the 7 m192 aux keys
  added; 4090 baseline — the 13 m192 keys added (no m192 serving on 24 GB; mirror the 5090 seeding
  if a 4090 m192 tier is ever wanted).

## Workflow notes

- `--golden NAME` matches by prefix, so a name whose `.cut` sibling exists benches both — and one
  unrealizable pinned row left the OTHER row unbenched in the same run (the m8 cut row failed alongside
  the fused row's `realized (off)`, then benched clean solo). Per-name runs are the workaround; a
  pin-failure that poisons sibling rows is worth a look.
- The audit's GAP records carry the ShapeKey but not a human name; mapping keys back to
  `free_prod = M × N` by hand was the most repeated manual step. A `--names` view (nearest golden-name
  guess per gap key) would cut it.
- The fused-tier realization loss was invisible to every automated check: the drift gate MATCHes when
  ANY same-key row realizes, so a cut sibling masks a dead fused arm. A per-ROW realization audit
  (each golden row must still realize, not each key) would have caught it the day it landed.
