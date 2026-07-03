# Golden sweep findings — RTX 5090 (sm_120), 2026-07-02 (eighth sweep; first learned-prior-owned sweep)

- **Branch under test:** `tuning/gemma-4-5090` (main + `b4695256` — the learned-prior calibration gate + cross-vocabulary
  guard; PRs #295–#302 in the base: fused-contraction recognize, sync compute-fill, WSPEC, perf-gap fixes, ML-ready node
  store with **fold partitioning**, flash-form fork, warp K/V staging).
- **Sweep:** `emmy tune --dataset golden --clean` — all 34 shapes, cold DB, **~66 min wall** (19:59–21:06). A/B:
  `run --bench --golden NAME --json`, all 34 shapes × 2 passes (confirm-twice as JSON diffs), ~4 min/pass off the warm
  cubin cache. Zero integrity flags (intensity floor / wrong-answer) on any pinned row.
- **Deploys:** the **learned prior owned every greedy pick** — `prior.json` was re-fit by the sweep, and the new
  calibration gate promoted it (reservoir Spearman **+0.89** ≥ 0.5 floor; `Prior.trustworthy=True`, verified in-process
  before judging). This is the first sweep judged against learned deploys — sweep 7 had to delete the checkpoint after
  fp16 picks went 32× wrong; this sweep the fp16 family is all wins or parity, so the gate + trainer rework (#299) held.
- **Tally (34 shapes):** **16 replaced / 6 added / 7 parity (5 of them µs-refreshed) / 5 worse** (3 = the structurally
  dead reduce fork, plus q_proj.s512.dynM 1.10× and square.1024 1.07×).

## Per-shape outcomes (pass-max over 2 A/B passes, -O3 live, learned-prior deploys)

`cuBLAS µs` is the live `Eager PyTorch` row from the same `run --bench` (min across passes); `vs cuBLAS` =
greedy / cuBLAS (>1 = emmy slower than PyTorch).

| shape | greedy µs | best-golden µs | ratio | cuBLAS µs | vs cuBLAS | category |
|---|---|---|---|---|---|---|
| square.512 | 8.7 | 9.9 | 0.88 | 12.3 | 0.71 | **replaced** (`n16x8/f4x8 g2k d3`) |
| square.1024 | 43.3 | 40.6 | 1.07 | 45.0 | 0.96 | worse (marginal); winner µs refreshed, stale sibling pruned |
| square.2048 | 243.1 | 243.6 | 1.00 | 252.7 | 0.96 | parity, same knobs — **the sweep-7 1.50× gap is closed** |
| square.4096 | 2088.6 | 2036.2 | 1.03 | 2056.1 | 1.02 | same → added (`f4x26 d2` beside refreshed `f4x10`) |
| square.512.fp16 | 3.8 | 5.5 | 0.68 | 6.1 | 0.62 | **replaced** (`w1x4/f2x2/k4 d2`) |
| square.1024.fp16 | 15.4 | 16.3 | 0.95 | 14.4 | 1.07 | **replaced** (`w4x2/f2x4/k4 d4`) |
| square.2048.fp16 | 91.8 | 95.5 | 0.96 | 97.5 | 0.94 | same → added (`w2x2/f2x4/k4 d2`) — first time under cuBLAS |
| square.4096.fp16 | 629.1 | 704.0 | 0.89 | 638.6 | 0.99 | **replaced** (`w2x4/f2x4/k2 d2`) |
| qwen3_06b.q_proj.s32 | 7.5 | 7.5 | 1.00 | 8.2 | 0.91 | parity (greedy = recorded knobs) |
| qwen3_06b.kv_proj.s32 | 5.4 | 6.0 | 0.90 | 6.1 | 0.89 | **replaced** (`n32x8/f4x4 g8k d4`) |
| qwen3_06b.o_proj.s32 | 8.8 | 9.1 | 0.96 | 8.2 | 1.07 | same → added |
| qwen3_06b.gate_up_proj.s32 | 9.9 | 10.0 | 0.98 | 12.3 | 0.80 | same → added (`g2k` beside the two `g2a` rows) |
| qwen3_06b.down_proj.s32 | 9.7 | 12.7 | 0.76 | 10.2 | 0.95 | **replaced** (`n32x8/f4x2 g8k d4`); duplicate old rows dropped |
| qwen3_06b.q_proj.s128 | 14.2 | 16.4 | 0.86 | 16.4 | 0.87 | **replaced** (`n16x8/f4x8 g2k d2`) |
| qwen3_06b.kv_proj.s128 | 9.4 | 10.2 | 0.92 | 10.2 | 0.92 | **replaced** (`n32x16/f4x4 g8k d4`) |
| qwen3_06b.o_proj.s128 | 13.7 | 17.7 | 0.77 | 15.3 | 0.90 | **replaced** (`n32x8/f4x8 g8k d2`) |
| qwen3_06b.gate_up_proj.s128 | 20.2 | 20.5 | 0.99 | 32.8 | 0.62 | same → added |
| qwen3_06b.down_proj.s128 | 18.3 | 28.1 | 0.65 | 84.6 | 0.22 | **replaced** (`n32x8/f4x8 g8k d2`) |
| qwen3_06b.q_proj.s512 | 39.3 | 40.6 | 0.97 | 44.8 | 0.88 | same → added (`d2` beside refreshed `d4`) |
| qwen3_06b.kv_proj.s512 | 22.8 | 28.0 | 0.81 | 34.0 | 0.67 | **replaced** (`n32x8/f4x8 g2k d2`) |
| qwen3_06b.o_proj.s512 | 42.6 | 54.6 | 0.78 | 51.3 | 0.83 | **replaced** (`n32x8/f4x8 g2k d2`) |
| qwen3_06b.gate_up_proj.s512 | 52.9 | 53.8 | 0.98 | 68.9 | 0.77 | parity (greedy = recorded knobs) |
| qwen3_06b.down_proj.s512 | 63.8 | 67.5 | 0.94 | 66.9 | 0.95 | **replaced** (`n16x16/f4x8 g2k d2`) |
| square.512.dynM | 10.1 | 10.8 | 0.93 | 12.3 | 0.82 | **replaced** (`n16x16/f4x4 g8k d4`) |
| qwen3_06b.q_proj.s512.dynM | 51.7 | 47.1 | 1.10 | 44.9 | 1.15 | worse — the one real deploy miss (Finding 2) |
| qwen3_06b.kv_proj.s512.dynM | 23.3 | 29.2 | 0.80 | 33.8 | 0.69 | **replaced** (`n16x8/f4x8 d2`, REDUCE dropped) |
| qwen3_06b.o_proj.s512.dynM | 46.1 | 44.7 | 1.03 | 51.3 | 0.90 | parity-marginal; golden µs refreshed 52.7→44.7 |
| qwen3_06b.gate_up_proj.s512.dynM | 59.9 | 71.0 | 0.84 | 69.3 | 0.86 | **replaced** (`n16x8/f4x10 d2`) |
| qwen3_06b.down_proj.s512.dynM | 66.2 | 63.4 | 1.04 | 66.2 | 1.00 | parity-marginal; golden µs refreshed 74.8→63.4 |
| reduce.2048x2048 | 180.5 | 3.4 | 52.9 | 6.0 | 30.1 | worse — dead fork (Finding 3) |
| reduce.1024x512 | 45.6 | 3.0 | 15.2 | 3.9 | 11.7 | worse — dead fork (Finding 3) |
| reduce.2048x128 | 12.0 | 3.1 | 3.9 | 2.0 | 6.0 | worse — dead fork (Finding 3) |
| pointwise.2048x2048 | 8.1 | 8.1 | 1.00 | 8.2 | 0.99 | parity |
| pointwise.512x4096 | 4.3 | 4.3 | 1.00 | 4.1 | 1.05 | parity; stale refs refreshed (emmy 6.15→4.3, ref 6.15→4.1) |

Sweep-7 state for comparison: 12 shapes judged, 1 replaced, worst deploy gap 1.50× (square.2048), fp16 judged only
after discarding a mis-calibrated learned pass. This sweep: 22 shapes improved or newly recorded, the worst
**matmul** deploy gap is 1.10×, and 19 of 29 matmul shapes now beat live cuBLAS at greedy.

## Finding 1 — the calibrated learned prior owns deploys and closed the historical gaps

The sweep re-fit `prior.json` and the new calibration gate (`b4695256`) promoted it: reservoir Spearman +0.89
(`CALIBRATION_MIN` 0.5), so `FallbackPrior` let the learned half own every deploy — the exact configuration that was
catastrophic in sweep 7 (fp16 32× misses) is now the best sweep on record. The two long-standing deploy gaps closed:

- **square.2048 (fp32)**: greedy now picks the `n32x8/f4x26 d2` golden itself (243 µs, 0.96× cuBLAS; sweep 7 deployed
  365 µs / 1.50×). The deep-FM tile the analytic prior priced at rank 19–33 is a top learned pick.
- **The fp16 squares**: every shape better or parity; square.2048.fp16 is under cuBLAS for the first time (0.94×), and
  square.512.fp16's new `k4` reg-chunk pick is 0.62× cuBLAS.

The win class is concentrated in the **new fold-partitioning REDUCE codecs** (`g8k` / `g2k`, PR #299) — 13 of the 16
replaced shapes carry one — plus the fp16 `k2`/`k4` reg-depth chunks. These moves post-date the recorded goldens, so
the sweep was partly a re-tune over a genuinely larger space, not purely a prior improvement. Recommendation: none —
this is the success case; keep the gate. The next `scripts/golden_knob_heuristics.py` refit should ingest the new
YAML so the cold analytic prior learns the `gNk` family too.

## Finding 2 — q_proj.s512.dynM (1.10×): the learned prior misprices the masked-tile winner

Greedy deploys `n32x8/f4x8 g8k d3` (51.7 µs); the recorded golden `n32x8/f4x14 g2a d3` re-benched 47.1 µs (also under
live cuBLAS 44.9 → the deploy is 1.15× behind PyTorch). `eval prior --dataset golden` puts the golden at rank
**1763/3740** — the single deepest golden rank in the whole set (next worst: 1279 on a square.2048.fp16 alternate).
The same `g8k`-over-`g2a` substitution pattern shows up in the other two dynM marginals (o_proj / down_proj .s512.dynM,
1.03–1.04): the learned prior generalizes "gNk wins" from the static family onto masked tiles, where the deeper-FM
`g2a` forms actually hold. square.1024 (1.07×, `f2x14` picked over the measured-best `f4x8`) is the same class:
**the deploy is a pure prior argmax — tune evidence never overrides it** (the old DB→fork replay was removed), so a
config the tune measured as best can still lose the deploy. Recommendation: (a) let the deploy consult the tune DB
when an exact ShapeKey match exists (evidence-over-prior for tuned shapes — cheap, surgical); (b) failing that, add a
masked-tile interaction feature (`S_ext_n_symbolic_axis` × REDUCE family) so the fitter can separate the dynM regime;
the `_W_A_DYN` analytic split already proved the regimes differ.

## Finding 3 — the reduce fork is still dead at deploy (carried from sweep 7, now quantified)

`tune --dataset golden` gives a pure reduce **3 benches total** (`reduce.2048x2048`: "3 benches, best 180.667 µs @
bench #1") — the fork enumerates nothing, so greedy falls to the serial option-0 schedule: **52.9× / 15.2× / 3.9×**
behind the pinned goldens on the three reduce shapes (the goldens themselves still lower and run fine when pinned —
3.0–3.4 µs, at or under eager). PR #300's chain/coop/serial prior-ranked siblings cover the flash-form (attention)
fork; the bare-`Reduction` schedule fork still emits zero rows through `analytic.enumerate_graph`, and `eval analytic`
prints no reduce rows at all. This stays the top structural gap: greedy-deployed reduces are unusable while pinned
configs prove the codegen is fine. Recommendation (major, own investigation): restore the bare-reduce schedule fork
through `Run.resolve` (chain/coop/block siblings for a root `Reduction` with no contraction), then re-tune the three
reduce goldens; until then reduce golden ranks are meaningless and `_W_A` stays matmul-only in practice.

## Finding 4 — warp-tier TMA staging missed the 256 box-extent gate (fixed this session)

Four bench-worker failures during `square.4096.fp16`: `TMA box dim 0 extent 512 outside the hardware range 1..256`
(boxes (512, 16) / (512, 32)). `_resolve_warp_stage` gated K-divisibility and 16 B strides (`_can_stage_warp_tma`) but
never the boxDim ≤ 256 hardware rule — an A slab of `(tile_m, bk)` with tile_m = 512 (w4 × f8 register tiles) encoded
and died; the scalar resolver has had the equivalent gate since sweep 7. Each failure burned a bench slot at the 2 s
`bench_fail` pin instead of enumerating a legal sibling. Fixed in `lowering/tile/_schedule.py` (`tma_box_ok`:
`max(m.tile, n.tile, bk·atom_k) ≤ 256` → decline TMA), stale `050_use_tma` comment in `backend/cuda/_tma.py`
re-pointed; regression test `test_warp_tma_declines_oversized_box` (verified red pre-fix at `_tma.py:123`, green
post-fix).

## Finding 5 — stale reference latencies in long-unswept rows

Live cross-checks caught two reference drifts: `pointwise.512x4096` recorded emmy/ref 6.15/6.15 µs vs live 4.3/4.1
(both refreshed in the YAML — the only cublas_us edit this sweep, per the "config-independent" rule the drift means
the recorded bench itself was stale, not the config); the reduce rows record 2.05–4.12 µs vs live pinned 3.0–3.4 and
eager 2.0–6.0 (left untouched — they need the Finding 3 re-tune anyway). The q_proj.s128 legacy `d3` row swung
24.3 ↔ 16.4 µs *between passes* (48%!, the widest confirm-twice spread this sweep) — superseded and pruned, but it
shows the per-row noise floor on small shapes can far exceed the documented 10–13%.

## Workflow notes

- The `--json` A/B record held up (4th sweep, zero table scraping), but the driver + categorizer + knob-diff are still
  three session-written scripts (`run_ab.sh`, `summarize.py`, `detail.py`). The whole steps 2–4 loop is mechanical:
  an `emmy eval sweep` (or `tune --dataset golden --ab`) that runs both passes, applies the ±3%/5% rules, and emits
  the category table + proposed YAML diff would collapse ~40 min of scripting to one command.
- Shape names for the A/B loop were grepped out of the YAML (and matched `gpu_name:` — a bogus "NVIDIA" shape ran and
  cleanly failed). A `emmy eval golden --names` listing (one name per line) is the missing primitive.
- `eval variants --kernel` matches the *kernel hash* (`k_matmul_00d612`), not the golden name — cross-referencing a
  golden shape to its DB variants is a manual join (this sweep: skipped; the prior-rank view carried the evidence).
  A `--golden NAME` filter on `eval variants` would close it.
- Confirm-twice caught nothing this sweep (pass-max ≈ pass-1 everywhere except the q_proj.s128 48% swinger), which is
  itself evidence the integrity gates + warm-cache A/B are stable; cost is only ~4 min/pass now, keep it.
- Cold `--clean` sweep: 66 min / 34 shapes (fp32 squares ≈ 8 min each up front; the s32/s128 qwen shapes ≈ 1 min).
  The warm-resume path (sweep 7: 8 shapes in 8 min) remains the right tool for targeted re-tunes; `--clean` was
  correct here because the fold-partitioning moves invalidated the old rankings wholesale.
- From sweep 7's notes: the `--json`/integrity-gate items stayed fixed; the "tune silently re-creates prior.json"
  hazard is **resolved** by the calibration gate (this sweep intentionally judged WITH the learned prior); the -O1/-O3
  ranking-lane inversion wasn't re-audited (deploys no longer route through the -O1 lane's argmax, lowering its
  stakes).

## Post-fix retune — ninth sweep (2026-07-03, same branch)

Findings 1–3 were fixed and the full sweep re-run (`tune --dataset golden --clean`, 34 shapes, ~72 min; two
confirm A/B passes at ~5 min each). Fixes, with their in-sweep verification:

- **Finding 3 (reduce fork) — FIXED.** `_reduce_specs` only ever emitted the single `_pick_coop` heuristic spec
  (`_FREE_CAP=256` collapsed every golden reduce to serial-only, no fork). It now forks option-0 (the unchanged
  conservative pick — cold-greedy deploys identical) plus the full legal `coop_reduce_moves()` catalog, gated on the
  hint extent like the contraction path. In-sweep: `reduce.2048x2048` tuned **14 benches** (was 3), and the A/B
  deploys are transformed — greedy 3.2 / 1.2 / 1.6 µs on the three reduce shapes (was 180.5 / 45.6 / 12.0), i.e.
  **1.3–3.4× faster than eager** where sweep 8 was 4–53× behind the pinned goldens. All three reduce goldens
  replaced (`b32` / `b16` / `b4`, 0.41–0.95× of the old records). Regression test:
  `test_bare_reduce_forks_the_coop_catalog`.
- **Finding 2 (evidence-over-prior) — IMPLEMENTED, and it re-frames the residual.** Greedy's deploy hierarchy is now
  -O3 reservoir evidence → tune-DB measured best on exact `S_*` signature (new: `_db_measured_pick`, querying the
  deploy context AND the `-Xcicc -O1` tune-lane twin, `evidence_pick` prefix-consistency) → model argmin
  (`policy/greedy.py`; tests in `test_db_evidence_deploy.py`). q_proj.s512.dynM still misses (1.09×), but the DB
  now proves why: its golden `n32x8/f4x14 g2a` was **never measured this tune** (124 variants benched for the
  shape, golden not among them) while the deploy ranks **4/124 within 1.2% of the measured -O1 best** — the deploy
  policy is honest; the residual is *search coverage* of the deep-FM `g2a` masked-tile region (an
  `--explore-eps` / patience re-tune or a `D_*` fold-family feature is the next lever, not deploy plumbing).
- **Finding 1 (analytic refit) — attempted; weights HELD BACK, one exposed bug fixed.** The
  `golden_knob_heuristics.py` refit over the sweep-8 YAML gained little (all-cards top1 1→6/79, median 45→43 — no
  `D_*` feature sees the fold-partition family, the durable cold-start gap), and shipping the new weights steered
  cold deploys onto TWO latent hang/misalign variants (`test_e2e_matmul`'s (8,3) fp32 B under a vectorized
  cp.async ring; a `k_sdpa_linear_reduce__partial` split). The first is a real gate hole and was FIXED — the
  scalar stage resolver's 16 B inner-stride rule now covers cp.async as well as TMA (an odd-stride operand stays
  gmem-direct; regression test `test_scalar_cpasync_declines_odd_stride`, and this likely also explains the Gemma
  `k_linear_reduce` bench_fail cluster's signature). The sdpa-split variant is unfixed, so the old weights stay:
  re-attempt the refit only after hardening the cold-deploy variants it reaches (tune survives them via
  bench_fail pins; greedy deploys cannot). Sweep-9 deploys were learned-prior-owned (+0.92), so its records are
  unaffected by the revert.
- **Finding 4's fix held in production: zero `bench_fail` rows in the entire ninth sweep** (sweep 8 had 37; the 4
  TMA box-extent crashes are gone and the misaligned-address cluster was Gemma-shape-specific).

### Per-shape outcomes — ninth sweep (pass-max over 2 A/B passes, -O3 live, learned-prior + evidence deploys)

`cuBLAS µs` is the live `Eager PyTorch` row from the same `run --bench` (min across passes); `vs cuBLAS` =
greedy / cuBLAS (>1 = emmy slower than PyTorch). Ratios are against the sweep-8-updated YAML records.

| shape | greedy µs | best-golden µs | ratio | cuBLAS µs | vs cuBLAS | category |
|---|---|---|---|---|---|---|
| square.512 | 8.7 | 8.7 | 1.00 | 12.3 | 0.71 | parity (greedy = recorded knobs) |
| square.1024 | 39.2 | 40.5 | 0.97 | 44.7 | 0.88 | same → added (`n32x8/f4x8 d2`) |
| square.2048 | 242.6 | 241.8 | 1.00 | 253.0 | 0.96 | parity, same knobs; µs refreshed 243.6→241.8 |
| square.4096 | 2134.7 | 1964.9 | 1.09 | 2091.7 | 1.02 | worse (marginal); both golden rows re-benched faster and µs-refreshed |
| square.512.fp16 | 3.8 | 3.8 | 1.00 | 6.1 | 0.62 | parity (greedy = recorded knobs) |
| square.1024.fp16 | 15.2 | 15.4 | 0.98 | 14.4 | 1.06 | same → added (`w2x4/f2x2/k4 d2`) |
| square.2048.fp16 | 91.6 | 91.7 | 1.00 | 96.6 | 0.95 | parity (greedy = recorded knobs) |
| square.4096.fp16 | 627.5 | 634.4 | 0.99 | 640.2 | 0.98 | parity (greedy = recorded knobs) |
| qwen3_06b.q_proj.s32 | 6.0 | 7.4 | 0.81 | 8.2 | 0.73 | **replaced** (`n32x8/f4x4 g8k d3`) |
| qwen3_06b.kv_proj.s32 | 4.6 | 5.4 | 0.85 | 6.1 | 0.74 | **replaced** (`n32x8/f2x4 g8k d3`) |
| qwen3_06b.o_proj.s32 | 6.3 | 8.8 | 0.72 | 8.2 | 0.76 | **replaced** (`n32x8/f2x4 g8k d4`); both old rows pruned |
| qwen3_06b.gate_up_proj.s32 | 9.0 | 9.6 | 0.93 | 12.3 | 0.73 | **replaced** (`n32x8/f4x4 g4k d4`); three old rows pruned |
| qwen3_06b.down_proj.s32 | 8.4 | 9.6 | 0.87 | 10.2 | 0.82 | **replaced** (`n32x8/f2x4 g8k d3`) |
| qwen3_06b.q_proj.s128 | 13.4 | 14.2 | 0.94 | 16.4 | 0.82 | **replaced** (`n32x8/f4x8 g4k d2`) |
| qwen3_06b.kv_proj.s128 | 8.3 | 9.3 | 0.90 | 10.2 | 0.81 | **replaced** (`n32x8/f4x8 g8k d3`) |
| qwen3_06b.o_proj.s128 | 13.3 | 13.5 | 0.98 | 15.2 | 0.88 | same → added (`d3` beside refreshed `d2`) |
| qwen3_06b.gate_up_proj.s128 | 20.2 | 20.1 | 1.00 | 32.8 | 0.62 | parity; µs refreshed |
| qwen3_06b.down_proj.s128 | 18.0 | 18.1 | 1.00 | 85.8 | 0.21 | same → added (`d3` beside refreshed `d2`) |
| qwen3_06b.q_proj.s512 | 39.1 | 39.1 | 1.00 | 44.6 | 0.88 | parity (greedy = recorded knobs) |
| qwen3_06b.kv_proj.s512 | 23.0 | 22.8 | 1.01 | 34.3 | 0.67 | parity (greedy = recorded knobs) |
| qwen3_06b.o_proj.s512 | 43.5 | 41.6 | 1.05 | 51.3 | 0.85 | worse (marginal); golden µs refreshed 42.6→41.6 |
| qwen3_06b.gate_up_proj.s512 | 53.2 | 53.5 | 0.99 | 69.1 | 0.77 | same → added (`d2`); stale `d4` row pruned |
| qwen3_06b.down_proj.s512 | 60.5 | 60.2 | 1.00 | 66.8 | 0.91 | parity, same knobs; µs refreshed 63.8→60.2 |
| square.512.dynM | 9.2 | 10.1 | 0.92 | 12.3 | 0.75 | **replaced** (`n16x16/f2x4 d4`) |
| qwen3_06b.q_proj.s512.dynM | 51.7 | 47.5 | 1.09 | 44.9 | 1.15 | worse — search-coverage miss (golden unmeasured in 124 variants) |
| qwen3_06b.kv_proj.s512.dynM | 23.7 | 33.0 | 0.72 | 34.0 | 0.70 | **replaced** (`n16x8/f4x8 d3`; sweep-8 record didn't reproduce) |
| qwen3_06b.o_proj.s512.dynM | 44.5 | 45.1 | 0.99 | 51.2 | 0.87 | parity (greedy = recorded knobs) |
| qwen3_06b.gate_up_proj.s512.dynM | 66.6 | 76.2 | 0.87 | 68.7 | 0.97 | **replaced** (`n16x16/f4x8 g4k d2`; sweep-8 record didn't reproduce) |
| qwen3_06b.down_proj.s512.dynM | 65.5 | 63.5 | 1.03 | 66.7 | 0.98 | worse (marginal) |
| reduce.2048x2048 | 3.2 | 3.4 | 0.95 | 6.1 | 0.53 | **replaced** (`b32`) — restored fork, beats eager 1.9× |
| reduce.1024x512 | 1.2 | 3.0 | 0.41 | 4.1 | 0.31 | **replaced** (`b16`) — restored fork, beats eager 3.4× |
| reduce.2048x128 | 1.6 | 3.1 | 0.52 | 2.0 | 0.78 | **replaced** (`b4`) — restored fork |
| pointwise.2048x2048 | 8.1 | 8.1 | 1.00 | 8.2 | 0.99 | parity |
| pointwise.512x4096 | 4.3 | 4.3 | 1.00 | 4.1 | 1.05 | parity |

Ninth-sweep tally (vs the sweep-8-updated YAML, pass-max over two passes, learned prior calibration +0.92):
**14 replaced / 4 added (+2 pruned stale alternates) / 12 knob-reproductions / 4 worse** — the worse set is
down_proj.s512.dynM 1.03, o_proj.s512 1.05, q_proj.s512.dynM 1.09 (all three the dynM/search-coverage class above)
and square.4096 1.09 (its two golden rows re-benched *faster* than recorded — 1964.9 / 2030.8 — and were refreshed;
the greedy pick's 2134.7 same-knob re-bench sits 5% above its own pinned twin, i.e. inside this 2 ms kernel's
run-to-run band). Every emmy golden µs in the YAML now beats its cuBLAS reference except square.1024.fp16 (1.06×)
and pointwise.512x4096 (1.05×, memory-bound parity). Notable instability caught by the live A/B: the two dynM
records from sweep 8 (gate_up 59.9, kv 23.3) re-benched at 76.2 / 33.0 this sweep — masked-tile shapes swing far
beyond the documented noise band across sessions; their entries were replaced with configs that reproduced.
