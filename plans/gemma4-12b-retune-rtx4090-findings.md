# gemma-4-12B layer-0 re-tune on RTX 4090 — post #323/#324/#325 findings

Date: 2026-07-08. GPU: rented CloudRift RTX 4090 (driver 580.65.06, CUDA 12.9), emmy @ `c3619392` (main with the
demoting sync compute-fill, the bench-attribution fix, and the first rtx4090 rms_norm goldens). Purpose: measure the
end-to-end layer win the three PRs promised, and seed the projection matmul goldens for the gemma shapes.

Method: no HF download — the layer-0 traced IR (`00_input.json`) and per-kernel reproducers from the 2026-07-07 dump
replay directly (`emmy run <ir> --bench`, `emmy tune <ir>`). Cold = fresh box, no tune DB / prior. All benches -O3.

## Cold baseline (greedy, no prior) — the deploy-without-tuning story

Whole-layer e2e: **26.2 ms**. Highlights of the cold greedy pick:

| Kernel | shape | cold pick | µs |
| --- | --- | --- | --- |
| k_linear_reduce_65a10e (q_proj) | 512×3840×4096 mixed | `a:mma_m16n8k16_f16/w2x4/f2x2 d1/sync` | 458.8 |
| k_linear_reduce_70fe8d (k/v_proj ×2) | 512×3840×2048 mixed | `a:mma_m16n8k16_f16/w2x2/f2x4 d1/sync` | 249 each |
| k_linear_mean_reduce_b87781 (gate+up fused) | 512×3840×15360 ×2ch mixed | mma `w1x1/f2x2` grid 15360 blk 32 | 12783.6 |
| k_linear_sdpa_reduce_a15de4 (sdpa+o fused) | — | scalar `n32x16/f4x8` | 3384.3 |
| k_linear_reduce_b62521 (down_proj) | 512×15360×3840 f16 | mma `w1x1/f2x2` | 2781.2 |
| k_mean_linear_reduce_33f743 / e4bb2c (fused norm→o / →down) | — | scalar, grid 2 | ~2150 each |
| k_mean_9b7435 (residual mean) | 512×3840 | scalar, grid 2 | 1675.0 |

Two immediate observations:

- **The #325 demotion works cold**: the plain mixed projections (q/k/v) deploy on mma with sensible micro-tiles
  straight from the analytic prior — no tune needed. The old behavior (pre-#325) was a scalar-tile prison at
  ~1.6 ms per projection; cold greedy now lands 249–459 µs.
- **The cold misses are concentrated in the fused forms**: gate+up (12.8 ms on a degenerate `w1x1/f2x2` pick with
  32-thread CTAs), the norm→linear fusions (scalar, grid 2 — the whole GPU idles on 2 CTAs), and the sdpa+o edge.
  The torch reference for the whole layer is unavailable (`float != c10::Half`) — the traced graph carries the
  erased-cast mixed matmul torch itself refuses to execute, so e2e has no eager twin; per-shape references come
  from HGEMM at the same shapes instead (see the golden table).

## Finding 1 — the layer tune REGRESSES the layer 6.5×: warm greedy deploys bare scalar on every big matmul

After `emmy tune 00_input.json` (3581 s, 1267 benches, reservoir 9307 rows, post-warmup calibration +0.96), the warm
`run --bench` deploys **bare scalar** on every plain projection the cold analytic prior had correctly put on mma:

| Kernel | cold pick / µs | warm pick / µs | DB best (-O3) |
| --- | --- | --- | --- |
| k_linear_reduce_65a10e (q_proj) | mma `w2x4/f2x2` / 458.8 | bare scalar / 24 856.6 | mma, ~630 (see phase-D) |
| k_linear_reduce_70fe8d (k/v ×2) | mma `w2x2/f2x4` / 249 | bare scalar / 12 432 each | mma |
| k_linear_reduce_b62521 (down) | mma `w1x1/f2x2` / 2781.2 | bare scalar / 94 569.7 | mma `w1x4/f4x4` / 1254.4 |
| k_linear_sdpa_reduce_a15de4 | scalar `n32x16/f4x8` / 3384.3 | worse scalar / 24 032.2 | — |
| whole-layer e2e | **26 196** | **170 474** | — |

The tune itself did its job — the small/fused kernels are all fixed (norm kernels 3–6 µs on coop `b32`–`b256`
picks, `k_mean_9b7435` 1675 → 6.5 µs, gate+up 12 783.6 → 2037.8 µs on `w4x2/f4x2/k4 d1/sync`, sdpa 86.3 → 46.8 µs
on `d2/cp/ring`), and the DB knows the truth for the matmuls (`eval variants linear_reduce_b62521`: 18 measured
configs, ALL mma, best 1254.4 µs -O3). The failure is **deploy-time ranking**: `eval variants` marks the prior's
pick among *measured DB leaves* only (all mma → picks mma), but greedy `run` enumerates the full fork set including
the never-measured bare-scalar row — and the learned CatBoostPrior, trained on a reservoir dominated by the layer's
many small coop-reduce kernels (98 % of post-warmup benches were ≥2× off best), extrapolates a better score for
bare scalar than for the measured-good mma rows. The cold AnalyticPrior does not make this mistake (its tier gates
rank the mma rows first — that is #311's analytic tier gating working as designed).

This is the same pathology seen 2026-07-07 on the isolated q_proj tune (bare scalar at 20.9 ms deployed after a
tiny single-kernel tune, calibration −0.47) — reproduced now at layer scale with GOOD global calibration (+0.96).
Calibration over the whole reservoir is not the right health metric: the ranking that matters is scalar-vs-mma *per
shape*, and a reservoir with no scalar rows for a shape lets the model rank an unmeasured scalar row anywhere.

Also observed during the tune: `k_linear_mean_reduce_b87781` (gate+up) hit repeated `HungKernelError` /
2 s-GPU-time bench_fails (variants pinned at 2 000 000 µs) — the tune recovered, but a chunk of its hour went to
timing out degenerate variants.

## Finding 2 — per-shape tune data does NOT rescue the deploy pick; the failure is at the structural fork

Tuning each projection shape in isolation (`emmy tune -c "<snippet>"`, same DB/prior) measures rich mma variant
sets — and greedy *still* deploys bare scalar on the very next `run --bench` of the same snippet:

| shape | tune benches | best -O3 µs (knobs) | greedy deploy µs | eager HGEMM µs |
| --- | --- | --- | --- | --- |
| q_proj (mixed) | 268 | 163.7 (`w1x4/f4x4/k2 d1/sync`) | 3 144.8 bare scalar | 122 |
| kv_proj (mixed) | 221 | 132.8 (`w1x4/f4x4/k2 d1/sync`) | 1 599 bare scalar | 57 |
| gate+up (mixed) | 135 | 665.6 (`w2x2/f4x8/k2 d1/sync`) | 13 441 bare scalar | 444 |
| o_proj (fp16) | 103 | 128.8 (`w2x4/f4x4/k2 g2k d1/cp`) | 3 147 bare scalar | 123 |
| down_proj (fp16) | 151 | 434.0 (`w1x2/f4x8 g2k d3/cp/ring`) | 13 183 bare scalar | 401 |

Two structural facts pin the failure to the fork walk, not the data:

- At the **leaf** level the prior is fine: `eval variants` (H_opt=3, measured -O3 evidence first) puts the ◄ pick on
  the true best -O3 row for every one of the five kernels.
- The pure-fp16 shapes fail identically to the mixed ones — this is not a #325 demotion artifact; even the
  classic warp tier loses to the unmeasured bare-scalar row at deploy time.

Greedy walks the fork tree option by option; at the structural fork (bare scalar loop vs mma subtree) the
candidates are partial-knob nodes with no measured-leaf identity, the model extrapolates, and scalar wins before
the walk ever reaches the measured mma leaves. `feature/greedy-structural-evidence` (in flight, same day) makes
DB evidence load-bearing at deploy — the cross-validation below tests exactly that.

## Finding 3 — cross-validation of `feature/greedy-structural-evidence` (ba4b8843): fp16 rescued, mixed still scalar

With the box's DB/prior intact, the same layer + snippet runs on the evidence branch:

| target | main deploy | evidence-branch deploy |
| --- | --- | --- |
| layer e2e | 170 474 µs | 77 162 µs |
| k_linear_reduce_b62521 (down, fp16) | bare scalar 94 570 µs | mma `w2x4/f4x4/k8 g2k` split 1095 µs ✅ |
| o_proj snippet (fp16) | bare scalar 3147 µs | mma `w2x4/f4x4/k2 g2k d1/cp` 130.9 µs — the exact DB winner ✅ |
| q_proj snippet (mixed) | bare scalar 3145 µs | bare scalar 3148 µs ❌ |
| q/k/v layer projections (mixed) | bare scalar 24.9 / 12.4 / 12.4 ms | unchanged ❌ |
| k_linear_sdpa_reduce_a15de4 (fused) | scalar 24 032 µs | unchanged ❌ |

The discriminator is clean: the deploy-time DB-evidence join rescues **pure-fp16** shapes (their `ShapeKey` is
warp-keyed) and misses the **mixed** ones — which key `is_warp=False` because the stamped `S_dtype_f32` counts the
f32 A operand (see the schema note below; the golden side must mirror the stamp, so the fix belongs in the evidence
join / lane keying, not the key). The fused sdpa+o kernel also stays scalar — possibly the same join gap via its
extra fused loads, worth checking separately. Recommendation for the branch: include the mixed signature
(f32-A × 16-bit-B, the #325 demotion class) in the deployable-evidence join — the -O3 evidence rows for these
kernels exist (that's how the goldens below were recorded); they are just not consulted at the structural fork.

## Golden seeding — five projection shapes recorded, `mixed` dtype added to the schema

New `mixed` matmul dtype in the golden schema (f32-A-erased × f16-B): torch cannot execute a mixed matmul, so the
snippet's A downcast rides `.to(torch.float16)` — which the tracer erases — leaving f32-A × f16-B in the traced
graph, the exact gemma norm→linear signature. Eager executes the cast, so the reference is cuBLAS HGEMM, matching
the model's real execution. The `ShapeKey` for `mixed` keys `is_warp=False`, mirroring the stamped op side
(`S_dtype_f32` ≠ 0 — a mixed contraction still loads an f32 operand) even though post-demotion it deploys on the
warp tier.

Recorded from pinned -O3 `run --bench` (accuracy-checked vs eager; same-run eager row = `cublas_us`, cross-checked
against a standalone HGEMM bench within noise). The mixed entries record no STAGE — the sync compute-fill is
mandatory (planner-derived) for the demoted form, and the permanence test correctly rejects `d1/sync` as a catalog
move:

| name | M×N×K | dtype | knobs | emmy µs | cuBLAS µs | ratio |
| --- | --- | --- | --- | --- | --- | --- |
| matmul.q_proj.h3840.mixed | 512×4096×3840 | mixed | `w1x4/f4x4/k2` | 164.4 | 109 | 0.66 |
| matmul.kv_proj.h3840.mixed | 512×2048×3840 | mixed | `w1x4/f4x4/k2` | 142.8 | 57 | 0.40 |
| matmul.mlp_gate_up.h3840.mixed | 512×15360×3840 | mixed | `w2x2/f4x8/k2` | 611.8 | 410 | 0.67 |
| matmul.o_proj.h3840 | 512×3840×4096 | fp16 | `w2x4/f4x4/k2 g2k d1/cp` | 131.1 | 113 | 0.87 |
| matmul.mlp_down.h3840 | 512×3840×15360 | fp16 | `w1x2/f4x8 g2k d3/cp/ring` | 426.4 | 385 | 0.90 |

None reach the ≥0.95 golden bar — the sm_89 mma schedule still trails HGEMM (0.4–0.9×), consistent with the known
~2× projection gap; the kv shape is the worst (0.40×, small-N tail). These entries are the ground truth the gap
work can be measured against. Summed over the layer's matmuls (q + k + v + gate_up + o + down ≈ 1.72 ms vs
HGEMM ≈ 1.26 ms), the *achievable* layer is ~30× faster than what greedy deploys today (Finding 1) — deploy
ranking, not kernel quality, is the whole ballgame on this card right now.

## Workflow notes

- The whole session ran off the saved `EMMY_DUMP_DIR` artifacts (17 reproducers + `00_input.json` from 2026-07-07)
  — no HF token, no 24 GB model download on the box. Replayability of dumps is a big deal for rented-GPU work;
  worth documenting as the default pattern for re-tunes.
- `emmy tune <layer.json>` (one invocation, all kernels) took 3581 s and self-recovered from repeated
  `HungKernelError` bench_fails on gate+up variants. The 2 s GPU-time cap and 1 s hung-kernel timeout did their
  job; no manual intervention needed.
- `run --bench` on a mixed-dtype IR prints `torch reference unavailable` and skips the vs-torch table — expected
  (torch can't execute the erased-cast graph), but it means layer-level A/Bs have no eager anchor. The per-shape
  snippet benches (whose eager side executes the cast) are the anchor; keep both in the loop.
- `eval variants` -O1 rank order inverts vs -O3 heavily here (q_proj -O1 rank 1 is -O3 rank ~8; the true -O3 best
  sat at -O1 rank 3, 11, even 30+ on some shapes). Winner selection MUST read the -O3 evidence column, never the
  -O1 ordering — reaffirms the skill's step-4 warning with the sharpest example yet.
- The box `git clone --depth 1` is single-branch: `git checkout origin/<branch>` fails with exit 128. Cross-branch
  validation needs `git fetch --depth 200 origin <branch>:<local>` first. Worth a line in the box-setup notes.
- Monitor cadence (ssh-poll queue.log every 90–120 s from the local box) made the multi-hour driver hands-off;
  the driver script + step log pattern (`=== step exit=N timestamp`) is cheap and diagnosable. Reuse.
