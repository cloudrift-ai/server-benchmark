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

## The 256/256 c=64 lane — what it is, and two hypotheses that died

The lane loses TTFT to stock and this session did not close it. What it did do is eliminate every
tunable explanation by measurement, which is worth more than another knob sweep.

**Ruled out — KV admission.** The leading theory (emmy's footprint starves the KV pool, so requests
queue) is false for this workload: it needs 64 x 512 = 32,768 KV tokens and emmy has **43,499**
without the chunk quantum, **65,232** with it. All 64 requests are resident either way, and
`peak_concurrent_requests` reads 64 on both engines.

**Ruled out — prefill kernel speed.** In the fast-math lane the deployed m4096 kernels aggregate
*faster* than eager (39,047 vs 44,275 us summed across all m4096 shapes). Emmy also WINS pure
prefill: one 4096-token prompt at c=1 gives TTFT 471 vs stock 565.

**Ruled out — chunk padding.** The hypothesis: emmy's prefill twin is a static fixed-width program,
so partial chunks pay for padding stock's varlen batch does not; 64 prompts of 256 tokens make
chunks partial. It predicted TTFT would keep improving as the quantum shrank toward the 256-token
prompt length (zero padding). The prediction was registered in the recipe before running. Measured,
fast-math lane, single-wave protocol:

| quantum | 4096 | 2048 | 1024 | 512 | 256 | stock |
|---|--:|--:|--:|--:|--:|--:|
| TTFT ms | 1834 | **1688** | 2253 | 1798 | 2104 | **1512** |
| tok/s | 1132 | 1122 | 1074 | 1106 | 1066 | 1198 |

Zero-padding (256) lands mid-pack, so **padding is not the mechanism**. The ordering is also
non-monotonic (512 beats 1024), and 1688 / 1798 / 1834 sit inside a ~9% band that single runs
cannot separate — treat 2048 / 512 / 4096 as indistinguishable and 1024 as an outlier worth a
repeat. 2048 stays the lane's setting (committed) because it is the best point measured and the
one with a baked pack.

**A confound found and removed mid-experiment.** The first pass at this sweep read 8707 ms at
quantum 1024 and 57,605 ms at 512 — apparently damning. Both cells showed ~1100-1200 s init, i.e.
they pack-missed and cold-compiled, and the golden file had merged matmul tiers at
m8/32/64/192/256/2048/4096 and **nothing at 512 or 1024**. The sweep was measuring missing coverage,
not the quantum. Seeding those two tiers (40 entries, commit `f84cc869`) moved the same cells to
2253 and 1798 — a 3.9x and 32x correction, and a clean demonstration of what an unseeded width
costs. Any future sweep over a new width must seed it first or it measures the wrong thing.

**What is left** is the integration seam: the plugin runs a mixed prefill+decode step as two passes
over disjoint rows where stock composes one fused varlen batch. That is consistent with the tail
shape (emmy mean 1.6x median, p99/median ~5; stock mean ~= median, p99/median 2.96) and with emmy
winning pure prefill while losing batched prefill. Closing it means owning attention and batch
composition — a fork of the serving stack, not a knob or a kernel.

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

## MTP: 13 of 15 blank table cells filled, and an assumption overturned

The article's three depth tables carried em-dashes where the grid never measured. Filling them
(all on baked shapes, so ~2 min boots) showed that one of the *reasons* cells were omitted is
false. The recipe header asserted "depth 5 already loses under batching"; it was never measured,
and it is wrong — depth 5 is the best depth for stock at c=4 and for emmy at c=8:

| point | d3 | d5 |
|---|--:|--:|
| 4k/4k c=4, stock | 443.7 | **586.5** |
| 4k/4k c=8, emmy | 496.0 | **632.7** |
| 4k/4k c=8, stock | 716.5 | 728.2 |
| 4k/4k c=4, emmy | **436.3** | 404.2 |

The 8192/256 RAG row (absent from the grid entirely) is now measured at every depth: speculation
helps stock (112.9 -> 129.3 at d5) and is roughly flat for emmy (102.5-113.9), consistent with a
prefill-dominated point where speculation has little to offer.

**Two cells stay blank deliberately**: 256/256 c=64 at depths 3 and 5 need decode buckets of 256
and 384, and bucket 256 is the configuration that FAILS the GSM8K quality gate (0.0, empty
completions). Publishing throughput for a config that emits garbage is worse than an em-dash.
Re-opening them means warming bucket 256 and re-running the quality gate first — now more
plausible than before, since the multi-shape bake can warm that width and the original failure was
attributed to cold-resolved kernels at an unwarmed width.

## Measurement-plumbing failures worth not repeating

Three runs were invalidated by the harness rather than the code under test, all mine:

1. **A leftover `vllm_0` container** from a previous suite held all 31.9 GB, so every bench worker
   in the m512/m1024 seeding pass died on allocation. `emmy bench` does not tear the container down
   when the run ends; chain scripts must `docker rm -f` between suites.
2. **`nvcc` absent from a systemd unit's PATH** — the next attempt compiled nothing
   (`RuntimeError: nvcc unavailable`). Transient units do not inherit an interactive PATH; export
   `/usr/local/cuda/bin` explicitly.
3. **A `head`-truncated log parsed as if complete**, which manufactured a 180x "win" on
   `norm_gate_up.m1024.lin.cut` (6.4 us against eager 1154). The real number was 1302.2 (0.89x).
   Parse the per-shape logs, never the console summary.

A fourth, non-harness: the desktop session's ~1.2 GiB of VRAM put a util-0.97 boot 216 MiB short
and killed a bake. At util 0.97 on a 32 GB card the desktop must stay under ~979 MiB.

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
