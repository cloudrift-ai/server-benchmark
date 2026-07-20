# Gemma-4-12B golden re-tune + verification + e2e serving A/B (RTX 5090)

- **Date**: 2026-07-18 · **GPU**: NVIDIA GeForce RTX 5090 (sm_120, driver 580.159.03, CUDA 13.0) · fp16
- **Sweep command** (fast-math superset, cold, gemma4 shapes only, dedicated DB so the 589 MB main
  `autotune.db` is untouched):

  ```
  EMMY_FAST_MATH=1 EMMY_O3_TOL=0.10 \
    EMMY_TUNE_DB=_tune/gemma4-golden-retune-5090/tune.db \
    EMMY_ONLINE_FILE=_tune/gemma4-golden-retune-5090/online.json \
    emmy tune --dataset golden --clean --kernel gemma4 --gpus 1
  ```

- **Wall time**: tune ≈ 11.5 h (57 shapes, one cold invocation, within-sweep transfer); focused -O3 A/B ≈ 30 min
  (26 shape-lanes); empty-DB verification ≈ 20 min (18 shapes); serving A/B ≈ 40 min.
- **Category tally**: **0 replaced / 0 added / 56 confirmed-unchanged / 1 worse** (`attention.hd512`, a
  cold-reachability gap the golden already protects). The full re-tune **confirms the gemma-4 5090 golden set is
  already at its optimum** — no greedy pick beat a recorded golden above the noise floor.

## Headline

This was a **confirmation sweep, not a rewrite**. The gemma-4 goldens were tuned within the prior few days
(PRs #392–#394) on this same card and compiler, so the cold re-tune reproduces them: `emmy eval golden` shows the
fresh greedy pick matches the recorded golden on **TILE 67/67, STAGE 67/67, RASTER 67/67** of the matmul goldens
(REDUCE 52/67 — the g-split fill is the only family that ever differs, and never enough to change the deployed
latency). The A/B and the empty-DB verification both land every shape but one back on its recorded golden.

## Per-shape A/B — greedy pick vs recorded golden (-O3 `run --bench`, both lanes)

Greedy µs is the isolated emmy-only bench of the deployed pick; golden µs is the best recorded config for that lane
benched live the same run; `rec` is the YAML's recorded `emmy_us`; eager is the live cuBLAS/torch reference.

| shape | lane | greedy µs | golden µs | rec µs | eager/cuBLAS µs | greedy/gold | outcome |
| --- | --- | --: | --: | --: | --: | --: | --- |
| attention.hd256 | std | 36.5 | 36.4 | 31.7 | 34.8 | 1.002 | confirm (golden selected) |
| attention.hd256.s2048 | std | 305.3 | 305.9 | 262.7 | 280.2 | 0.998 | confirm (golden selected) |
| attention.hd512 | std | 186.2 | 123.1 | 113.0 | 112.4 | 1.512 | WORSE 1.51x |
| lm_head.m32 | std | 1201.2 | 1203.0 | 1199.1 | 1204.1 | 0.999 | confirm (golden selected) |
| mlp_down.m32 | std | 74.5 | 74.5 | 74.1 | 76.5 | 1.000 | confirm (golden selected) |
| mlp_gate_up | fm | 418.3 | 417.9 | 362.4 | 618.5 | 1.001 | confirm (golden selected) |
| mlp_gate_up | std | 642.4 | 641.7 | 561.7 | 629.7 | 1.001 | confirm (golden selected) |
| mlp_gate_up.dynM | std | 634.8 | 643.1 | 562.4 | 636.1 | 0.987 | confirm (golden selected) |
| mlp_gate_up.m32 | std | 144.9 | 144.9 | 144.8 | 152.5 | 1.000 | confirm (golden selected) |
| mlp_gate_up.s2048 | fm | 1558.3 | 1557.9 | 1413.0 | 2303.4 | 1.000 | confirm (golden selected) |
| mlp_gate_up.s2048 | std | 2496.4 | 2500.3 | 2191.0 | 2315.1 | 0.998 | confirm (golden selected) |
| mlp_gate_up_split.m256 | fm | 131.4 | 133.6 | 125.6 | 158.3 | 0.984 | confirm (golden selected) |
| mlp_gate_up_split.m256 | std | 192.2 | 193.3 | 187.3 | 153.1 | 0.994 | confirm (golden selected) |
| mlp_geglu.m32 | std | 175.6 | 175.4 | 171.9 | 159.7 | 1.002 | confirm (golden selected) |
| norm_kv_proj.m32 | std | 22.9 | 22.9 | 20.8 | 14.3 | 1.000 | confirm (golden selected) |
| norm_q_proj.m32 | std | 26.6 | 26.6 | 24.1 | 20.5 | 0.998 | confirm (golden selected) |
| o_proj.s2048 | fm | 229.5 | 229.6 | 205.0 | 301.3 | 0.999 | confirm (golden selected) |
| o_proj.s2048 | std | 330.6 | 331.2 | 294.0 | 330.3 | 0.998 | confirm (golden selected) |
| o_proj_global.s2048 | std | 683.1 | 682.3 | 595.1 | 665.9 | 1.001 | confirm (golden selected) |
| q_proj | fm | 68.9 | 66.0 | 61.5 | 92.1 | 1.044 | WORSE 1.04x |
| q_proj | std | 90.5 | 91.2 | 84.2 | 100.3 | 0.992 | confirm (golden selected) |
| q_proj.m32 | std | 10.9 | 10.9 | 10.7 | 16.4 | 1.002 | confirm (golden selected) |
| q_proj_global.s2048 | std | 678.6 | 676.7 | 601.4 | 689.4 | 1.003 | confirm (golden selected) |
| qknorm.k256 | std | 3.7 | 3.7 | 3.6 | 6.1 | 1.001 | confirm (golden selected) |
| rms_norm.k3840 | std | 6.7 | 6.7 | 6.3 | 6.1 | 0.999 | confirm (golden selected) |

Notes:
- **Lane hygiene matters**: the `pinned` set carries both the std and `[fm]` goldens for a name; comparing a std
  greedy against the *fast-math* golden (f16-accumulate, ~1.3× faster but gate-locked) manufactures a phantom
  "regression". Compared within-lane, every shape but `attention.hd512` reproduces its golden.
- The golden rows re-bench ~8–14 % **slower** than their recorded `emmy_us` on the big shapes (e.g. `mlp_gate_up`
  642 vs rec 562). The direction is uniform and correlates with 11.5 h of sustained load ⇒ read as mild thermal
  drift within the skill's ~10–13 % golden-row noise band, **not** a codegen regression — the recorded values are
  the verified tier and stand.
- `q_proj [fm]` reads 1.04× (66.0 golden vs 68.9 greedy) at **identical knobs** — run noise, not a real miss.

## Cold-deploy verification — golden tier alone (empty tune DB + online file)

`emmy run --bench --golden <name> --no-record-nodes` with `EMMY_TUNE_DB`/`EMMY_ONLINE_FILE` pointed at fresh empty
paths (reset between shapes), so **only the repo golden tier drives deployment**. Representative shapes across every
kind:

**17 / 18 goldens deploy from the tier** (greedy realized knobs ≡ golden knobs, latency within 3 %). The one gap:

| shape | greedy (empty DB) | golden | verdict |
| --- | --: | --: | --- |
| attention.hd512 | 173.5 µs | 123.7 µs (1.40×) | golden **not** deployed even from the tier — see Finding 1 |

- **The finding-2 ShapeKey shadow is resolved.** `q_proj_global.m32` (32×8192) now deploys its own 17.6 µs golden,
  not the 512×512-square shadow (26.5 µs) the decode-seeding report flagged — PR #386's `free_max` discriminator
  holds.

## Finding 1 — `attention.hd512`'s flash golden is not cold-reachable (the only miss)

- **Symptom**: with the re-tune DB, greedy deploys `PLACE@fold=fuse / TILE@dd=w4x1/f1x4/k32 / TILE@pj=w4x1/f1x64/k2
  / STAGE=d1/cp` at **186 µs**; with an empty DB (golden tier) it still lands `w4x1/f1x4 d1/cp` at **173.5 µs** —
  never the golden's `w4x1/f1x2/k32 + TILE@pj=w4x1/f1x64 + REDUCE=g2k + STAGE=d1/cp/alt` at **123 µs** (1.40–1.51×).
- **Class**: cold search-reachability on a fused flash schedule (matches the standing `hd512 flash codegen-bound`
  note in memory). The golden records the good schedule but the greedy/prior can't reach it cold, and — unlike the
  matmul goldens — the golden tier does **not** pin it at deploy time, so the deployed hd512 kernel runs 1.4× its
  golden.
- **Serving impact: none.** The 12B's 8 global (`full_attention`, hd512) layers route attention through vLLM's
  paged-attention backend in `emmy serve`, not emmy's flash kernel — emmy owns the projections / MLP / norms there.
  hd512 is a `compile`/bench-path gap only.
- **Recommendation**: this is the known research-class hd512-flash item (symbolic split-KV not built); leave the
  golden (it is the better config and the honest target). If the deployed hd512 needs the golden without a DB,
  the fix is on the golden-tier match for the fused-flash `TILE@dd`/`TILE@pj`/`STAGE` fork set (the golden pins the
  schedule but the tier doesn't apply it), not a re-tune.

## Fork sibling regret (`emmy eval online --dataset nodes`, -O1)

The skill's card-level fork-regret table (`emmy eval online --dataset nodes`) does **not** scope to the gemma4 set:
the node store keys rows by op label (`matmul`/`reduce`/…), not golden name, so `--kernel gemma4` matches 0 of the
22 952 stored nodes in both prior halves. The steering question is therefore answered directly from the golden-scoped
reproduction instead of the node-store regret:

- **The prior reaches every gemma4 golden it was shown.** `emmy eval golden --dataset golden --kernel gemma4` (fresh
  re-tune DB) shows the greedy pick matches the recorded golden on **TILE 67/67, STAGE 67/67, RASTER 67/67, WSPEC
  67/67** and **REDUCE 52/67** (the g-split fill is the only family that ever differs, and the A/B confirms it never
  changes the deployed latency). Effective per-fork regret on the gemma4 matmul set is ≈0 — the online prior this
  sweep trained deploys the golden schedule on every shape.
- **The one steering gap is `attention.hd512`** (Finding 1): a fused-flash fork the cold search can't reach and the
  golden tier does not pin. That is a reachability/enumeration gap on the flash `TILE@dd`/`TILE@pj`/`STAGE` fork set,
  not a weight-mispricing the node-store regret table would surface — so refitting `scripts/golden_knob_heuristics.py`
  is not the lever; the flash stage-enumeration on the pre-split geometry is (the standing hd512 item).

Net: no offline-weight refit is warranted from this sweep — the cold ranking already reaches the recorded goldens.

## E2e serving A/B — gemma-4-12B, emmy vs stock vLLM (local 5090)

Both arms: vLLM 0.23.0, fp16, `--max-model-len 512 --max-num-batched-tokens 256 --gpu-memory-utilization 0.90`,
concurrency 32, 64 prompts, seed 0. emmy: repo goldens (confirmed above) + the in-model twin serving DB
(`_tune/decode-twin-readiness/twins.db` + `twins-online.json`), decode CUDA-graph bucket 32
(`FULL_DECODE_ONLY`). stock: native vLLM gemma-4 with `--language-model-only` (skip the vision encoder so it boots
at small mnbt and matches emmy's text-only path).

| workload | arm | req/s | output tok/s | TTFT mean / median (ms) | TPOT mean / median (ms) |
| --- | --- | --: | --: | --: | --: |
| in-8 / out-64 (decode-dominated) | stock | **25.70** | **1645** | 103 / 95 | **18.1 / 18.1** |
| in-8 / out-64 (decode-dominated) | emmy | 20.86 | 1335 | 99 / **81** | 22.7 / 22.8 |
| in-256 / out-64 (mixed) | stock | **14.25** | **912** | **460 / 217** | **26.0 / 28.0** |
| in-256 / out-64 (mixed) | emmy | 7.11 | 455 | 1814 / 2332 | 32.1 / 32.0 |

**Reading it honestly — stock vLLM currently leads on gemma-4-12B, a reversal from the prior report.**

- **Decode (in-8)**: stock TPOT **18.1 ms** vs emmy **22.7 ms** (emmy 1.26× slower). emmy wins **TTFT** (81 vs 95 ms
  median) and is within 20 % on throughput. emmy's 22.7 ms is only ~7 % off the prior tuned twins-online decode
  (21.3 ms), so the decode gap holds regardless of the online config — stock's native gemma kernels + torch.compile
  decode are simply faster here.
- **Mixed (in-256)**: stock wins across the board — req/s 14.25 vs 7.11 (emmy 0.50×), median TTFT 217 ms vs 2332 ms.
  This is the prefill wall, and it is **overstated by the emmy arm's config**: this run used the *empty* online
  prior (twins.db carries the decode-twin evidence but the tuned symbolic-prefill twins live only in the online
  prior), so emmy's prefill fell back to worse kernels → 2332 ms TTFT vs the prior tuned device-prefill twin's
  1579 ms. Even at that tuned 1579 ms / 7.06 req/s, stock's current 217 ms / 14.25 req/s leads.

**Config caveat (important).** The prior "emmy beats stock" (decode 21.3 vs 24.9 ms, req/s 0.70× stock) used the full
twins-online prior + the symbolic-prefill twin tune, and a stock baseline measured at 10.33 req/s. This session's
matched stock is both leaner (native text-only via `--language-model-only`, vLLM 0.23 default `VLLM_COMPILE` +
full cudagraphs) and faster (14.25 req/s), and the emmy arm ran with the empty online prior because the 56 MB
twins-online file made the serve boot re-parse it per fork (≈15 min, and the background job was killed before it
could bench — see the workflow note). The decode conclusion is robust to that config difference; the mixed-workload
gap is partly the config and partly a real prefill-compute wall (the research-class large-M computed-A pipeline and
packed-varlen prefill still open per `plans/computed-a-pipeline-and-sdpa-oproj.md`).

## Seeding the deploy prior from goldens — mechanism, validated + refined

Question raised mid-session: `evidence_pick` deploys the best measured kernel, so can *replaying* all goldens seed the
prior to reach them cold? Answer: **yes in principle, but the naive replay hits the wrong table, and it doesn't carry
serving.** Evidence (all on this 5090, golden TIER moved aside so only the prior/evidence decides):

- **`emmy eval offline --kernel gemma4`**: the cold `OfflinePrior` (model, no evidence) ranks the gemma4 goldens at
  **median rank 682** (top-10 only 9/67; many at rank 4k–13k). So the *model* never reaches them cold — deploy reaches
  55/57 purely via the golden tier + evidence, never the model surface.
- **Deploy tier order** (`greedy.decide`): golden → `evidence_pick` (online reservoir, `H_opt=3`) → `_db_measured_pick`
  (tune-DB `perf` rows) → model argmin. Evidence sits **above** the model; the model surface is never retrained on
  goldens by design.
- **The naive replay writes the wrong table.** `run --bench --golden --record-nodes` records into the **`node`** table
  (the *offline-prior training feed*), not the **`perf`** table that `_db_measured_index` reads. Isolation test (tier
  removed): the `--record-nodes` seed deployed **byte-identical to the model-only baseline** on all four probe shapes
  (mlp_gate_up.m16 stayed 1449 µs, o_proj.m32 32.7 µs) — no deploy effect.
- **The `tune`-populated `perf`/reservoir evidence DOES seed the deploy.** Same isolation with the 57-shape re-tune DB
  (perf rows from `tune --dataset golden`) as the only evidence: **mlp_gate_up.m16 1449→144.6 µs (10× rescue)**,
  **o_proj.m32 32.7→11.1 µs (exact golden knobs)**; q_proj / k_proj_global.s2048 rescued to near-golden (a measured
  `-O1`-lane row that inverts vs the `-O3` golden — exact only where the tune's `-O3` rebench covered the golden). This
  is the mechanism working, via the correct (perf/reservoir) path.
- **But it does NOT carry serving.** Serving the 12B with the golden-seeded re-tune DB: py-spy showed most in-model
  forks falling through to `mean_scores` — the isolated matmul/fused golden ShapeKeys don't match the in-model
  computed-A decode twins (norm→qkv cones, gate⊗up), so the decode-critical fused forms stay unseeded (matches the
  documented "bare goldens bought 1.9× (3639→1924 ms)"; fused forms remain the ~160× wall). The run also exposed a
  boot pathology: `_db_measured_index` rebuilds the full perf index **per program (~96×)** over the 57-shape DB (the DB
  analog of the online-prior reload), pushing the serve boot past 21 min (vs ~11 for `twins.db`) — aborted before
  capture. **Competitive serving needs the in-model *twin* tune (`twins.db`), not an isolated-golden replay.**

## Workflow notes

- **`--clean` wiped the shared cubin cache** (`~/.cache/emmy/cubin`), so the first `emmy serve` boot recompiled the
  whole 12B kernel set from scratch. *Symptom*: 13-min cold serve boot. *Improvement*: `tune --clean` could scope
  its cache purge to the shapes it tunes, or the serve boot could report compile progress.
- **The A/B parser must split the `pinned` set by lane.** `min()` over all recorded goldens picks the fast-math
  entry and manufactures a phantom "greedy 1.5× slower" against every std greedy. *Improvement*: `run --bench
  --golden` (and `eval golden`) should label each pinned row `[fm]`/std in `--json` so a comparison can't cross
  lanes by accident.
- **`emmy serve --stock` needs two env fixes that aren't discoverable.** (1) stock loads the full multimodal
  checkpoint and dies at small mnbt on the MM-encoder budget check — only `--language-model-only` clears it; (2) the
  inductor compile subprocess needs `ninja` on PATH, which running `./venv-serving/bin/emmy` directly does not
  provide (`FileNotFoundError: ninja`). *Improvement*: `emmy serve --stock --generate` could auto-add
  `--language-model-only` for known-MM checkpoints and prepend the venv bin dir to the child PATH.
- **The serve boot re-parses the online prior JSON per fork.** py-spy on a stalled boot showed the MainThread
  pinned in `json.loads` of the 56 MB `twins-online.json`, reached through `evidence_pick`→`sig_groups`
  (`prior/base.py`) inside `greedy.decide`, called once per fork across all ~96 in-model programs — an O(programs ×
  file-size) boot cost (a 67 MB fresh online file stalled the boot 15 min in pure resolution before any cubin
  compiled). *Improvement*: cache the parsed online prior once per process instead of re-loading it per
  `evidence_pick`; this is the single biggest serve-boot lever.
- **Slowest step by far is the tune itself (11.5 h).** The global projection giants dominate — `q_proj_global`
  (512×8192) alone took 1706 s. A cold full-set re-tune is the wrong tool when the goldens are recent: `eval golden`
  (knob reproduction) + a focused -O3 A/B on the fused/attention forms would have reached the same "confirmed, no
  changes" verdict in <1 h. Reserve the full cold sweep for a compiler/kernel change that actually invalidates the
  set.
