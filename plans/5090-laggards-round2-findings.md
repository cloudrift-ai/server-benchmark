# 5090 laggards round 2: computed-A async-B staging + fused `.lin` twins + tail retune (2026-07-20)

Session goal: work down `plans/staged-transposed-b-lin-goldens-findings.md`'s remaining-laggards plan — step 0
(live triage), step 1 (computed-A async-B staging, the class-2 lever), step 3 (canonical std tail), step 4
(rms_norm probe), plus the 4090 twins and a final e2e serving A/B vs stock vLLM.

## Step 0 — live re-bench triage (30 laggard names × 3 reps, local 5090)

A uniform **+10–19% absolute inflation** on BOTH the emmy and eager rows of the sustained-load reps (hot-card
clocks; ratios preserved), so conclusions are ratio-based; µs pairs recorded in this session come from same-run
medians. Ratio movers vs the recorded YAML:

- **No longer laggards** (purged from the worklist): `qknorm.k512` 0.94 → **1.10**, `o_proj.s2048` 0.97 →
  **1.00**, `k_proj_global.m256.lin` 0.95 → 0.99.
- **Improved but still behind**: `kv_proj.m256` 0.81 → 0.90, `mlp_ch.dynM` 0.81 → 0.86, `q_proj.m256` 0.91 →
  0.90 (flat).
- **Worse live**: `mlp_gate_up_split.m256` 0.81 → **0.79** (worst tail shape), `norm_q_proj_global.m32` 0.89 →
  0.83, `rms_norm.k3840[.dynM]` 0.97 → **0.91**.
- **Confirmed**: the fused norm_* decode laggards (0.49–0.83), `mlp_geglu.m32` 0.89, attention hd256 family
  0.86–0.95, hd512 0.86–0.92 pinned.
- `attention.hd512` static: pinned golden runs 0.92 live but **cold greedy misdeploys at 0.50×** — the standing
  cold-unreachable enumeration gap (memory: hd512 flash), unchanged by this session.
- Both `mlp_geglu.*.cut` goldens were **unbenchable** — see the pin bug below.

## The `PLACE@cone=cut` pin bug (fixed)

`run --golden mlp_geglu.{m256,dynM}.cut` failed with `unreproducible pin: PLACE@cone=cut realized (off)`. On the
MONOID (fused norm→gate⊗up) edge, a pinned cut took the `pro=None` path in `010_recognize`, failed
`_cuttable_cone` (multi-fold, stats), and scheduled the plain Map form with no `PLACE@cone` stamp — the pin
never realized, so the golden row was integrity-refused. Greedy (unpinned) still deployed the cut fine from
evidence. Fix: a pinned cut on a monoid-bindable node now schedules the Map form with `PLACE@cone=cut` threaded
onto every row (the same stamping as the stat-free pin path), which `020_cut_edge` realizes.

## Step 1 — computed-A async-B staging (the class-2 lever)

The sync compute-fill's transposed-B channels used to fill **per-cell** (strided gather on the drain's own
threads, no prefetch). `_atom._sync_operands` now builds EVERY B fold channel — canonical K-major or transposed
N-major (`tile_n × bk`, K stride-1, `Operand.trans`, swizzle from `bk_elems`) — as a vectorized `cp.async`
`Operand` on `async_operands`, flying under the compute fill; `_schedule._resolve_sync_stage` drops the
`not c.b_trans` term so the asymmetric B-only `d2` ring is enumerable on transposed-B fused edges. The staged
sync output stays bit-identical (same fill values, same drain); the fused-edge + matmul-coverage + knob-pinning
suites pass on sm_89 (4090, 106 passed) and sm_120.

The canonical fused goldens (`norm_*.m32`, `mlp_geglu.m32` — canonical `@ w` snippets) are UNAFFECTED by this
change (their B already rode cp.async); their 0.49–0.92 residual is a different, structural story (below). The
change targets the **serving layout**: the fused edges a served model actually deploys are `F.linear`
(`b_trans`), which had no golden twins at all.

## Fused `.lin` twins (new golden kind capability + seeding)

`NormLinearGoldenConfig` / `MlpGeGluGoldenConfig` gained `trans_b` (the `F.linear` snippet — the serving fused
edge), mirroring `MatmulGoldenConfig.trans_b`; same layout-blind ShapeKey, twins coexist and sort by µs.

**Cold greedy on the unseeded fused `.lin` shapes misdeploys catastrophically** (4090: `w1x1/f2x4` tiles at
0.08–0.28×; 5090 picks reasonable forms) — seeding these is a rescue class, not a polish.

Manual pinned `--ab`, 3 waves + a 3-rep confirmation per card. Learnings: the wide-N `w1x16` warp unit
transfers from the canonical fused winners onto the `.lin` twins; the **f16acc `[fm]` lane realizes on the fused
forms** (first time recorded) and wins or ties most shapes; `d1` vs `d2` sync is shape-dependent (kept as
measured); `k4 g8k` / `g16k` are unrepresentable at K=3840 (split-K slice not a multiple of the mma K-step).

**RTX 5090 recorded** (std / [fm], x = same-run torch-unfused eager ÷ emmy):

| shape | std µs (x) | fm µs (x) | vs canonical twin's recorded x |
|---|---|---|---|
| norm_q_proj.m32.lin | 26.5 (0.77) | 24.2 (0.84) | 0.68 |
| norm_kv_proj.m32.lin | 18.4 (0.78) | 18.2 (0.79) | 0.60 |
| norm_q_proj_global.m32.lin | 36.5 (0.85) | 30.6 (**1.01**) | 0.89 |
| norm_kv_proj_global.m32.lin | 13.3 (0.77) | 12.7 (0.81) | 0.50 |
| mlp_geglu.m32.lin | 178.6 (0.93) | 172.5 (**0.97**) | 0.92 |

**RTX 4090 recorded**: norm_q_proj.m32.lin std 26.4 (0.84; fm loses), norm_kv_proj.m32.lin 18.2/17.2
(0.90/**0.95**), norm_q_proj_global.m32.lin 40.7/36.3 (1.11/**1.25** — beats the torch unfused pair),
norm_kv_proj_global.m32.lin 12.7/12.6 (0.70/0.71), mlp_geglu.m32.lin std 285.0 (**0.99**; fm loses).

The 5090's canonical fused entries were re-recorded at the same-regime triage medians so the layout-blind
shared buckets compare like with like (stale cool-card µs let the canonical config shadow the fresh `.lin`
twin). Deploy verified on both cards: greedy resolves the recorded `.lin` configs from the tier.

Note the plan's step-1 exit (norm_* ≥0.9×) is met only on q_proj_global; the decode norm→q/kv fused forms
still lose 15–25% to torch's unfused pair on the 5090 (they lose the same on the canonical layout — a
structural computed-A decode residual, not a layout residue). What the goldens now anchor is the *serving*
fused-vs-cut decision with honest fused evidence — and the misdeploy guard (sm_89 cold greedy picked `w1x1`
tiles at 0.08–0.28× on these forks).

## Cut goldens re-recorded (`mlp_geglu.*.cut`)

Beyond the recognize-side pin fix, the pinned-row integrity gate needed a rule: a realized cut leaves NO
`PLACE@cone` stamp on its component kernels (the halves re-enter `010` fresh), so the pin read "realized
(off)" and refused the row. Every non-cut sibling of a cone fork stamps `PLACE@cone=fuse`, so off-only
evidence now passes the `cut` pin (a dropped pin that fell back to any fused/coop form still flags).
Re-records: 4090 `dynM.cut` 1098.3 → **890.8** (eager 910.2, 0.83 → 1.02×); 4090 `m256.cut` unchanged
(415 live ≈ 416.3 recorded); 5090 re-records below.

## Step 3 — canonical std tail (manual `--ab` neighborhoods, 3 reps, 5090)

Recorded (>3% std-lane wins; fm entries untouched — every one re-confirmed at 1.14–1.49× live):

- `k_proj_global.m256.lin` std: `w2x4/f2x4/k2 g8k d2/tma/ring` — 12.4 → **10.7 µs (1.15×**, was 0.99 live) —
  the doubled N-warp split (`w2x4`, the fm winner's geometry) transfers to the std atom.
- `mlp_ch.dynM` std: transport flip `d2/cp/ring → d2/tma/ring` on the same `w4x1/f4x8/k2` tile — 367.0 →
  **333.5 µs (0.95×**, was 0.86 live). The cp preference was tuned pre-#406-era; TMA now wins this shape too.
- `mlp_gate_up_split.m256` std: `w2x2/f4x8/k2 g4k` (halved bk, doubled split) — 191.5 → **178.8 µs (0.85×**,
  was 0.79 live). Still the worst std tail shape; the rest of its neighborhood loses.

No change (live golden already best-in-neighborhood): `q_proj.m256` (1.00× live), `o_proj.s2048` (1.00),
`o_proj_global.s2048` (0.96), `mlp_gate_up.s2048` (0.93; the w4x2/f2x4 neighbor is +2%, below threshold),
`kv_proj.m256` (0.89–0.90 ceiling), `k_proj_global[.dynM]` (0.90 ceiling), `mlp_down.dynM.lin` (0.92),
`o_proj_global.dynM.lin` (0.91). The std residual on these is the ordinary std-vs-fm atom gap — every shape's
fm sibling is at 1.14× or better, so the deployable lane is already ahead of cuBLAS.

## Step 2 — hd512 flash (timeboxed audit, no code)

Triage confirms the standing picture: the pinned hd512 goldens run 0.86–0.92 live, but cold greedy misdeploys
the static shape at 0.50× (the cold-unreachable enumeration gap; `eval golden --kernel attention.hd512` shows
zero matchable rows). The fix (porting the hd256 tile-skip/split-KV/alternating-staging levers + symbolic
split-KV for dynM) is new lowering, deferred to its own session per the original plan.

## Step 4 — rms_norm probes (5090)

The plan's cheap probe hit: `rms_norm.k3840.m32` REDUCE **b256 → b512 = 5.6 → 3.6 µs (0.73 → 1.13×)** — the
class-4 "launch-overhead outlier" was actually under-threaded (32 rows × 512 threads covers K=3840 with fewer
serial folds per thread). b128/b64 lose badly; `w32` is not a REDUCE token (vocab is g<n>/b<n>/r<n>). The
M=512 twins (`k3840`, `k3840.dynM`) stay b256 (b512 ties at 6.8–6.9 vs 6.7 — no change); their 0.91 live is
the bandwidth floor. Deploy verified (greedy resolves b512 from the tier).

## 5090 cut re-records (with the pin fixes)

- `mlp_geglu.dynM.cut`: 1177.6/597.3 (**0.51×**, the known crude stale pin) → **690.4/626.8 (0.91×)**.
- `mlp_geglu.m256.cut`: 350.2/326.0 (0.93×) → 372.7/329.6 (0.88× — same-regime refresh; pinned total matches
  greedy exactly, the cut is the deployed form).

## TODO(session): serving A/B, 4090 wrap
