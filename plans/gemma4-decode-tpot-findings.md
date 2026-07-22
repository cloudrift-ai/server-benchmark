# Decode TPOT research session — findings (2026-07-21, RTX 5090)

Execution of `plans/gemma4-decode-tpot-research.md` (WS1 epilogue-fuse the tail, WS2 warp-spec computed-A,
WS3 sdpa→o_proj staging, WS4 harness hang), branch `feature/gemma4-fm-universal-4k-goldens`.

## Baseline (fresh bucket-32 twin captures, fm, serving evidence env)

Fresh captures (`_tune/tpot/{pre,post}32*.json`) reproduce the predecessor's decomposition. Per sliding
layer: pre 36.2 µs (mains ~20.4), post 251.9 µs (mains ~232); global 32.9 / 259.3. Tail per step
(×48 layers): split-K finalizes ~550 µs, norm-stat `k_mean`s ~540 µs, qknorms ~240 µs, cut glue
(stat/cone/combine) ~220 µs, reshapes ~150 µs — ≈1.7 ms, the plan's ~1.4 ms bound plus the glue it
counted separately.

## WS1.1 — single-kernel atomic split-K on the mma tier (LANDED)

The `g<w>a` atomic finalize now covers warp (mma) partials: `RegStore.atomic` renders each C-fragment
store as the packed f16x2/bf16x2 `atomicAdd` (per-element for f32 — the cut channels' f32 workspaces),
`030_split_reduce`'s ATOMIC arm accepts `AtomKind` partials (single-fold, distributive projection only —
the gate in `_reduce_candidates` judges the FULL tail incl. a computed-A `Map` wrapper body),
`splitk_moves` offers `g<w>a` on both tiers, and `zero_outputs` re-zeros via `memset_async` (a graph
MEMSET node, not a cupy fill kernel). Accuracy: one output-dtype rounding per partition (f16 out:
max_diff ~5e-4 on the coverage fixture; f32 ws: none) — twin run PASS at max_diff 3.1e-5.

Where it pays (manual pinned `--ab`, .lin serving layout, 50 iters): the BIG-N decode shapes —

| shape | deferred g\*k | atomic g\*a |
| --- | --: | --: |
| gate_up_split.m32.lin fm | 77.4 | **74.7** |
| gate_up_split.m64.lin fm | 79.1 | **69.1** (−12.6%) |
| gate_up_split.m64 can fm | 78.8 | **67.5** (beats eager 78.0) |
| mlp_down.m32.lin fm k4 | 75.6 | **74.7** |
| mlp_down.m64.lin fm | 78.1 | **76.6** |

o_proj / q_proj / kv_proj / o_proj_global: wash or slightly worse (8-way atomic contention at small N
eats the finalize saving) — NOT flipped. Seeded: 12 golden rows flipped `g4k→g4a` + 2 new
`mlp_down.m32(.lin)` fm rows (14 atomic rows total). Twin verify (deploy-from-tier): post32
251.9→246.3, post32-global 259.3→253.3, post64 261→**244.3**, post64-global →255.7; audit
MATCH 111 / DRIFT 0 / GAP 0.

## WS1.x — graph-output reshape fold (LANDED; beyond the plan's list)

The pre-twin's three ~1 µs `k_reshape` kernels were exact flat-memory-identity copies (the q/k/v
head-layout flattens after the per-head qk norms) that could never fuse: the splicer would re-run the
reduce-bearing qknorm per element and its σ-solve rejects the div/mod reader form anyway. New rule
`loop/fusion/030_fold_output_reshape.py`: a graph-output pure single-Load indexmap whose map is a
flat-memory identity (verified EXACTLY over the finite domain via `Expr.eval`) folds by retargeting the
producer's `Write` to the output buffer at the same flat address, re-decomposed onto the output strides
as clean per-dim affine indices. pre32 36.2→33.1, pre32-global 32.9→31.8; values bit-match; audit
DRIFT 0 / GAP 0 both cards. (The 8 static `pw.n{2048,4096,8192}` flat-copy anchor rows now match no
kernel — inert, kept; the dynM pw rows still anchor the symbolic path, which the rule skips.)

## Twin-level net

Per step (40 sliding + 8 global, fm): 13.86 → **13.46 ms** of twin kernel time (−0.4 ms, −2.9%) —
the WS1 twin exit gate (≤13.8 from 14.85 captured) cleared at the bench-TOTAL level. The decode
capture also shrinks by ~6 kernels/layer (2 ch finalizes + down finalize + 3 reshapes ≈ 290 nodes),
which attacks the ~2.7 ms vLLM graph/dispatch bucket beyond raw kernel time.

## Post-WS1.1 per-kernel breakdown (fm, current tree — the remaining tail work list)

pre32, TOTAL 33.1 µs (mains 20.4 = q/kv split partials at cuBLAS parity):

| kernel | µs | role | remaining lever |
| --- | --: | --- | --- |
| `k_mean_b3bbda` | 3.7 | input norm stat + scale sweep → `type_as_cast` | producer = twin INPUT (vLLM writes h) — no in-twin producer to sink into; cross-twin sink (post twin's `k_mean_90bae6` computes h and could emit the next pre's stat) crosses the program boundary → runner interface change |
| q/kv `__partial` ×3 | 20.4 | mains (g4k/g8k) | at per-kernel cuBLAS parity |
| q/kv finalize ×3 | 4.1 | split-K deferred sum | atomic measured WASH/worse at N≤4096 (8-way red contention) — needs the last-CTA fused finalize (semaphore + in-kernel fold, CUTLASS stream-k style) to remove without precision/contention cost |
| qknorm sweeps ×3 | 4.9 | per-head rms (stat + sweep), now writing the flat q/k/v outputs directly | per-head stat contributions from the q/kv FINALIZE epilogues (256-elem segments align with finalize CTAs) → sweeps become pure pointwise; same sink machinery as below |
| ~~reshape ×3~~ | 0 | FOLDED (this session) | — |

post32, TOTAL 246.3 µs (mains 232.6 = o_proj 9.4 + gate 74.9 + up 74.3 + down 74.0):

| kernel | µs | role | remaining lever |
| --- | --: | --- | --- |
| o_proj finalize | 1.6 | g8k deferred sum | atomic wash at N=3840/g8 — last-CTA finalize, or absorbed by the stat sink below (the finalize is the natural stat-contribution site) |
| `k_mean_dcd9ba` | 3.8 | post_attn norm: Σx² stat + scale/residual sweep → `add_1` | **stat-into-producer sink**: o_proj finalize block-reduces its cells' x² per row + atomicAdd → `__sq[m]`; this kernel drops its Reduction and becomes a wide-grid pointwise sweep (~1.5 µs, cf. `k_mul_4__cone` class) |
| `k_mul_4__stat` | 1.7 | pre_ff norm stat over `add_1` (cut bridged stat) | same sink: `k_mean_dcd9ba`'s sweep is row-per-CTA — block-reduce its own output's Σx² and write the stat directly (no atomics needed) |
| `k_mul_4__cone` | 1.2 | normed x̂ workspace | stays (the cut halves need the materialized A) |
| gate/up ch ×2 | 149.2 | mains, now single-kernel g4a | bandwidth (WS2) |
| `k_mul_4` combine | 1.7 | GeGLU | fold into the down matmul's A cone (stat-free computed-A + the PR#406 async-B staging) — blocked by the deliberate `_CUT_WS_RE` fusion brake; needs a measured fork sibling, not a brake removal. With g4a channels an up-half epilogue combine is impossible (non-distributive over partitions) |
| down | 74.0 | main, single-kernel g4a | bandwidth (WS2) |
| `k_mean_90bae6` | 3.8 | post_ff norm stat + sweep (+residual, +final scalar) → `mul_7` | same stat sink — but down is now an ATOMIC partial (no complete-value site); sinking here needs down back on g4k (+0.9) for a net −1.4, or the last-CTA finalize |

Remaining tail ≈ 26.5 µs/layer ≈ **1.27 ms/step** (was ~1.7 pre-session). The dominant item is now the
`k_mean`/qknorm family (~0.78 ms/step), all one mechanism:

**The stat-sink design (next bounded workstream).** One general rewrite: a PLANAR row-reduce kernel
whose input tensor's producer writes each cell exactly once (elementwise / finalize / sweep, NOT an
atomic partial) migrates its reduce into that producer as a per-cell contribution + segmented
block-reduce + `atomicAdd` into a `__sq[m]` accumulator (`zero_outputs` per launch); the reduce
kernel keeps only its projection sweep (re-entering recognition as a wide pointwise, the π chain
rsqrt inlined per element — hoisted per-thread by the serial n-sweep). Two integration costs found
this session: (1) the producer gains a SECOND output (`T` + `__sq`) — the graph is single-output-
per-node, so this needs either a real multi-output node or the 030-style fragment idiom extended;
(2) the pick must be evidence-selectable (wins at decode M, murky at prefill M where the per-element
rsqrt and atomic traffic scale up) — a `REDUCE`-adjacent knob row realized like `PLACE@cone=cut`,
with a `linear_norm` golden kind (matmul + trailing rms) anchoring the pair's total. ~0.5 ms/step
bound if it lands for k_mean + qknorm both.

## Serving A/B (in progress)

Protocol: twins.db + empty online + FRESH packs (`_tune/tpot/packs-fm`), seed 0, 4K/4K workloads.

## 4090 port (DONE — remote manual `--ab`, riftvm)

The atomic split transfers to sm_89 (cp.async ring lane) at least as strongly as on the 5090:

| shape (.lin, 4090) | g4k | g4a | eager |
| --- | --: | --: | --: |
| gate_up_split.m32 fm | 146.8 | **137.5** | 136.2 |
| gate_up_split.m32 f32 | 147.4 | **137.9** | 136.2 |
| gate_up_split.m64 fm | 160.3 | **141.0** | 144.9 |
| mlp_down.m32 f32 (f4x1) | 140.0 | **137.9** | 141.7 |
| mlp_down.m32 fm (f4x1) | — | **137.5** | 141.7 |

Seeded: the existing `mlp_down.m32.lin` row flipped `g4k→g4a` + fm sibling, and the m32/m64
gate/up split-half rows seeded fresh (NEW on this card — the cut channels previously deployed from
db/prior only). The gate half now sits at eager parity (m32) / beats eager (m64) on sm_89.

Twin verify (twins-refresh env, fm): down deploys the single-kernel g4a from tier (138.4+1.3 →
137.2). The 4090 post twin's geglu edge deploys the FUSED megakernel (285-288 µs) — the m32 cut
never had a golden row on this card, and the measured A/B says that is CORRECT for sm_89: cut
(stat+cone+2×g4a ch+combine) = 289.3 ≈ fused 287.7. The cut channels realize at 142 µs in-graph
(f32 ch-workspace scalar atomics) vs 137.5 standalone (packed f16 out) — sm_89's scalar-f32 red
overhead eats the margin the 5090's halves keep. Per-card divergence, correctly held by the
evidence tiers (no cut row → stays fused). Ring depths: d2/cp optimal on sm_89 (down d3 138.4 /
d4 142.3 vs d2 137.7; gate d3 −1% ≈ noise) — prefetch depth is NOT the sm_89 bandwidth lever.

## WS4 notes (pre-probe reading)

The eager framing (`iter_once`) and the plain run (`run_once`) share `_descs_now()` — descs are NOT
the differentiator. Deltas that remain: per-launch CUDA events + `_wait_for_event` polling, repeated
iterations, and the `on_iter` torch-peer interleave (plain run has none of the three). Probe when the
GPU frees: reproduce under `--iters 3` (expect the 1 s watchdog), then re-run with an extended
watchdog and attach `cuda-gdb` for warp PCs on the stuck kernel.
