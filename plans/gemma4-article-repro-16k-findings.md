# Gemma-4-12B article reproduction at 16K context — RTX 5090, 2026-07-31 / 08-01

Brief: re-run the "Beating vLLM and Llama.cpp on Gemma4-12B" benchmarks, re-bake the serving image (goldens
changed, context raised to 16384), local 5090 only, update the article. Article:
`cloudriftai/cloudrift-landing` → `packages/blog/content/blog/optimizing-gemma-4-12b-rtx/index.md`.

**Result: the article reproduces.** Stock vLLM controls land within 1% at every point, Emmy serving within 1.8%,
per-kernel geomeans 1.15× (std) / 1.30× (fm) against 1.15× / 1.32× published, GSM8K all inside ±0.033. Two real
changes: the MTP saturated-batch cell improved **2.6×** (332.7 → 878.5 tok/s) from this session's fixes, and the
c=64 wave TTFT regressed for both Emmy lanes while their TPOT improved. Article updated on branch
`update/gemma4-16k-remeasure` (commit `0dc84dc`).

## Setup

| | |
|---|---|
| Card / host | RTX 5090 32 GB, driver 580.159.03, CUDA 13.0, 32 cores / 60 GB |
| Image | `cloudriftai/vllm-emmy-gemma-4-12b-it:0.23.0-d3df1f2a`, locally baked, **not pushed** |
| Context | 16384 (published run: 8448) — both engines moved together |
| Config | bucket 32, mnbt 4128, util 0.97; verified on-card (bench + no engine fatal) |
| Gates | goldens 386 ✓, preflight 306 OK / 0 FAIL ✓, warm 92 cubins / 385 plans ✓, verify PASS (zero recompiles, pack hit) ✓ |

Deploy-vs-command split: the serving/MTP suites are deploy recipes (measure the image); kernels, GSM8K, accum-error
and llama.cpp are command recipes (measure the working tree).

## Two defects found and fixed

### 1. The routing consult mis-keyed stat-free computed-A cones (the serving regression)

First full serving run on a fresh image measured Emmy **−22…−32%** against published, with the stock control
exact. Diffing the published image's baked execution-plan pack against the new one showed the m4096 `mlp_down`
deploying as one fused computed-A kernel where the published pack cut it into geglu-cone + big-tile matmul.

Root cause: #446's pre-fork routing consult keyed the kernel with a raw `ShapeKey.from_s_features`, whose
histogram classifier can only fire `kind="fused"` off a top-level `rsqrt`. The geglu→down cone has **no
statistic**, so it keyed `kind=''` and never joined its `fused`-keyed `.cut` routing entries — at any width, in or
out of model. Fixed in `_cut.py` (`_routing_entry` rebuilds the key off the tree's computed-A edge, mirroring the
offer-signal convention `greedy._fork_shape_key` documents). Commit `69c76383`; the drift-gate ratchet now
enforces the down-fused keys covered, which is the regression test.

**Corollary — a same-day finding retracted.** The 2026-07-31-morning report claimed the staged computed-A form
"no longer realizes in the isolated snippet". That was this defect seen from the rms side: those cones *do* key
`fused` at routing, so their `.cut` sibling routes pre-fork **in the golden replay too**, leaving no fused form
for a `STAGE=d1/sync` pin to realize against. The staged tier was never broken. Residual eval-side limitation,
still open: a fused *schedule* row cannot be replayed by name while a `.cut` sibling exists — the replay path
needs an implicit `PLACE=fuse` pin.

### 2. The execution-plan pack key ignored the precision gate

The fm and std lanes measured **digit-identical** at c=1 (both runs), while differing at batched widths. The
`FAST_MATH` family changes which kernel forks the compile enumerates, not the nvcc flags, so it appeared in
neither half of the pack validity key: an `EMMY_FAST_MATH=1` boot at the baked serving shape pack-**hit** the std
plans and silently served std kernels. Fixed in `pack.py` (the effective `FAST_EXP` / `F16_MMA_F32_ACC` pins join
the environment tag; older packs mismatch and fall back to a full compile). Commit `d3df1f2a`.

Confirmation: with the fix, fm 4k/4k c=1 TTFT = **471 ms** against published 475 (std 625/628) — the published
fast-math single-stream TTFT edge is real and reproduces; it had been invisible under the alias.

## Results

### Serving (tok/s | median TTFT ms | median TPOT ms), measured vs published

| point | lane | tok/s | pub | TTFT | pub | TPOT | pub |
|---|---|--:|--:|--:|--:|--:|--:|
| 4k/4k c=1 | stock | 57.2 | 57.2 | 565 | 566 | 17.3 | 17.4 |
| | emmy | 54.4 | 54.8 | 625 | 628 | 18.2 | 18.1 |
| | fm | 54.5 | 54.9 | **471** | 475 | 18.2 | 18.1 |
| 4k/4k c=4 | stock | 216.6 | 216.4 | 1086 | 1088 | 18.2 | 18.2 |
| | emmy | 203.9 | 206.4 | 1271 | 1266 | 19.3 | 19.1 |
| | fm | 205.2 | 207.7 | 1070 | 1066 | 19.2 | 19.0 |
| 4k/4k c=8 | stock | 383.8 | 383.6 | 1099 | 1100 | 20.6 | 20.6 |
| | emmy | 371.9 | 375.3 | 1224 | 1236 | 21.2 | 21.0 |
| | fm | 375.9 | 379.5 | 1007 | 1016 | 21.0 | 20.8 |
| 8k/256 c=4 | stock | 112.7 | 112.0 | **2027** | 2429 | 27.3 | 26.2 |
| | emmy | 99.9 | 101.7 | 2666 | 2655 | 29.4 | 29.2 |
| | fm | 112.5 | 113.7 | 2176 | 2173 | 26.6 | 27.3 |
| 256/256 c=64 | stock | 1435.9 | 1425.1 | 1513 ᵂ | 1468 | 27.7 | 28.0 |
| | emmy | 1139.0 | 1138.8 | **2277** ᵂ | 1772 | 29.3 | 30.0 |
| | fm | 1218.6 | 1219.5 | **1841** ᵂ | 1322 | 28.1 | 28.8 |

ᵂ single-wave protocol (`--num-prompts 64`), the article's footnote-2 measure; the recipe's own c=64 row is the
np=256 queue-drain and is not comparable to it.

**The two TTFT reversals.** At 8k/256 the *stock* lane improved (2429 → 2027) while fm held flat (2173 → 2176) —
the loss is stock getting better, not Emmy getting worse. At c=64 the Emmy lanes genuinely regressed (+28% / +39%)
while their TPOT improved 2%; at 64 streams on a 16K pool the KV cache is the binding constraint and Emmy's
footprint costs admission capacity exactly where prefill queues — the mechanism already documented in
`serving/ARCHITECTURE.md` under "Device footprint sets admission capacity", amplified by the doubled context.
Both are stated in the article rather than smoothed over.

### Speculative decoding (tok/s)

| lane | 256/256 c=1 | 4k/4k c=1 | 4k/4k c=4 | 4k/4k c=8 | 256/256 c=64 |
|---|--:|--:|--:|--:|--:|
| stock | 60.8 | 57.2 | 216.4 | 383.6 | 1433.7 |
| stock + MTP d2 | 106.1 | 111.5 | 349.2 | 595.1 | 1438.8 |
| stock + MTP d3 | 112.9 | 145.7 | 448.3 | 667.9 | — |
| stock + MTP d5 | 116.2 | 196.4 | — | — | — |
| emmy | 57.5 | 54.4 | 204.0 | 372.0 | 1137.7 |
| emmy + MTP d2 | 95.6 | 107.2 | 349.1 | 450.3 | **878.5** (pub 332.7) |
| emmy + MTP d3 | 104.7 | 135.7 | 425.3 | 548.0 | — |
| emmy + MTP d5 | 112.6 | 186.4 | — | — | — |

The c=64 d2 cell is the session's biggest win: **2.6× over published**, from the m192 golden tier (seeded in the
prior session, `dbc62e57`) plus this session's routing fix. It now clears Emmy's own speculation-off number
(1137.7) instead of losing to it 3×, and is deterministic where it used to be a 10.7–332.7 tok/s lottery.

### Per-kernel catalog (277 cases, `-O3`, both lanes)

| | torch.compile | Emmy std | Emmy fm |
|---|---|---|---|
| ≥ eager | 261/277 | 155/277 | **226/276** |
| geomean | 1.07× | 1.15× (pub 1.15×) | **1.30×** (pub 1.32×) |
| p90 / best | — | 2.0× / 6.0× | 2.0× / 6.0× |
| worst | — | 0.46× `mlp_geglu.m4096` | 0.46× `mlp_geglu.m4096` |

`attention.hd512` — the published worst case — has moved off the floor; the residual is now the m4096 fused GeGLU,
whose rescuing cut cannot lower (the #389 multichannel class).

### Quality and numerics

GSM8K (200 questions, strict exact-match, ±0.033): stock **0.685**, Emmy **0.670**, Emmy FAST_MATH **0.695**,
MTP-stock d2/d3 **0.675 / 0.685**, MTP-Emmy d2/d3 **0.680 / 0.690**. All within one standard error, matching the
published conclusion that hybrid accumulation costs no task quality. The accumulation-error sweep reproduces the
published table exactly (fp32 2.8e-7 at K=3840 → 8.1e-7 at K=32768; fp16 growing to 6.6e-3; hybrid flat 3.3e-4).

## Open leads (not fixed here)

1. **std-lane greedy picks trail their own recorded goldens on the wide projections.** Live catalog sweep:
   `q_proj` std 99 µs against a recorded golden of 84.2 (live eager 90 — *faster* than the recorded 97.6 cuBLAS,
   so this is not uniform box drift). Same shape in fm deploys at 65. Either the deploy pick is not reaching the
   golden config on those shapes, or the goldens need a refresh at this revision. Worth an `eval golden` /
   `run --bench --golden` pass on `q_proj` / `kv_proj` / `mlp_down`. This is why the article's TMA and
   hybrid-accumulation tables were left as pinned-configuration microbenchmarks and explicitly marked as such.
2. **`mlp_geglu.m4096` fp16 accuracy NaN in the fm lane** — second sighting (first was 2026-07-30, unreproduced
   then). Now recurring: `max_diff=64.0 mean_diff=nan`. A NaN in the fused geglu path deserves a look.
3. **A stale online prior silently wrecks bare-metal serving.** `~/.cache/emmy/prior.json` on this box is a 25 MB
   fitted CatBoost from 2026-07-10, predating the tile-IR re-key and featurizer v3. Bare-metal
   `emmy serve --generate` with `EMMY_FAST_MATH=1` served at **2.4–3.8 tok/s** (~18× slow) reading it; the same
   lane in docker (no prior) runs at full speed. Consider staleness-gating the prior on a feature/format version,
   or at minimum warning at boot when the checkpoint predates the current featurizer.

## Harness fixes landed alongside (`4a2048fb`)

- `serving_llamacpp` recipe: fail fast when `cmake` is absent (it was a `command not found` buried in a build log),
  and bound build parallelism by RAM — `-j$(nproc)` on 32 cores / 60 GB gets OOM-killed on the CUDA fattn
  template instances.
- `run_lmeval_gate.py`: explicit 900 s client timeout. lm-eval's 300 s default let the Emmy lane's slower request
  tails start a tenacity retry storm that died on a closed aiohttp session, sinking a gate with no score.
- `make serve-image`: the in-container reshard runs as the invoking user; as root it left the snapshot root-owned
  and the host-side `split_hf.sh` hard-links then failed under `protected_hardlinks` (no-op on root rentals).
- `docker/vllm-emmy-serve/ARCHITECTURE.md`: the preflight command mounts `scripts/`, not the single file — it
  imports its sibling `check_serving_goldens` and the documented invocation dies on `ModuleNotFoundError`.

## Not measured

- **llama.cpp lane** — deliberately skipped (nothing Emmy-side affects it; the stock control already establishes
  the box reproduces). Its article column carries the published values, now footnoted as not re-measured.
- **RTX 4090 per-kernel catalog** — out of scope by request (no 4090 in this session).
- **The TMA / hybrid / FlashAttention microbenchmark tables** — pinned-configuration replays, not part of the
  catalog sweep; see open lead 1.
