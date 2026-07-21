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

(sweep results and recorded entries below)

## TODO(session): fill in sweep results, tail results, rms_norm, 4090, serving A/B
