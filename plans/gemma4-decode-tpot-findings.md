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

**The stat-sink: implementation sketch (next session's primary lever, ~0.5 ms/step bound).**

*The move.* One general rewrite, applied per matched pair, never per named shape: a kernel B whose
op is `Map(body=sweep, source=Reduction(PLANAR over axis n))` reading tensor `T[m, n]` — the fused
rms/norm form (`k_mean_dcd9ba`, `k_mean_90bae6`, the three qknorm sweeps) — migrates its reduce into
T's producer kernel A as an epilogue contribution, when A writes each T cell exactly once (a split
FINALIZE, a pointwise sweep, an UNSPLIT matmul store — NOT an atomic partial, whose per-partition
values are incomplete). B keeps only its projection sweep. Grid-32 latency-bound stat+sweep kernels
(3.7–3.8 µs) become wide pointwise sweeps (~1.5 µs) and the stat rides for ~0.2–0.4 µs of epilogue.

*Producer-side codegen — one new leaf stmt.* `RowAccum(dst, row_index, value)`: "accumulate `value`
into `dst[row]` from every thread holding a cell of that row." Injected right after A's `Write(T)`
site together with the contribution chain (the reduce CELL of B — e.g. `v*v` — re-evaluated on the
just-stored value, substituting B's `Load(T)` with the local SSA name; the ×1/N of `mean` moves to
the consumer, so contributions are raw Σv²). Render by placement, resolved at materialize:
- CTA-covers-row (A's grid axis IS the row — the `k_mean` sweeps): thread-local partial over the
  serial n-sweep (`Init` before the loop, `Accum` inside), block tree, ONE plain `Write` — no atomics.
- flat cell grid (the finalize kernels, one cell/thread): warp-uniform-row check (`base/N ==
  (base+31)/N`) → shfl butterfly + one `atomicAdd` per warp; per-lane `atomicAdd` on the rare
  boundary warps. ~480 atomics per o_proj-finalize launch — noise. The per-head qknorm case is the
  same code path: the "row" is `(m, head)` and the finalize CTAs' 256-cell spans align with the
  256-wide heads, so warps stay uniform.
- mma RegStore epilogue (unsplit matmul A): per-fragment-element contributions + the same warp fold
  — reuses the `RegEpilogue` leaf machinery; defer to v2, the decode sites are all covered by the
  two arms above.
`__sq` zero-init rides the existing `zero_outputs`/memset machinery (detect `RowAccum` in
`_atomic_outputs`, exactly as `RegStore.atomic` is detected today).

*Consumer-side rewrite.* B re-emits as an UN-mapped `LoopOp` (the 020-cut re-entry idiom, so
`010_recognize` gives the sweep its own wide 2-D schedule): `for m: for n: [Load __sq[m], π
(reciprocal·+eps·rsqrt — B's post-reduce prefix with `acc := Load __sq[m]`), sweep cell, Write]`.
The π chain is per-element but hoists per-thread over the serial n-run (f4-vectorized threads own
4+ same-row cells; at decode M the rsqrt count is trivial either way). A bare-stat B (`k_mul_4__stat`,
projection = `Write(stat[m])` only) is v2: killing it requires splicing π into ITS consumers.

*The multi-output problem (the one real plumbing cost).* A now produces `T` AND `__sq`, but
`graph.Node` is single-output. Decision needed up front, two candidates:
1. **Aux-workspace node (recommended):** keep A's node output = `T`; `__sq` becomes a graph node of
   its own whose producer is A — extend `Graph.add_node`/splice with an `aux_outputs` list that the
   backend allocates like any buffer and the CudaOp declares as an extra write arg. Bounded: buffer
   planning already handles multi-buffer kernels (`external_writes` is plural); only the NODE/edge
   bookkeeping (topo order, liveness for the slab planner, serialization) learns aux edges.
2. True multi-output nodes — cleaner long-term, touches far more (every pass that assumes
   `node.output`); not worth it for one feature.

*Selection/evidence (never geometric).* The sink is a fork the RECOGNIZER offers on B's rows —
spelled as a `PLACE@stat: sink` sibling (the `PLACE` codec's second element, realized by a
`040_sink_row_reduce` pass reading the stamp exactly as `020_cut_edge` reads `PLACE@cone`), and
**evidence-only** like the cut row (withheld from cold greedy: at prefill M the per-element π and
atomic traffic scale up and the win is unproven). Whole-cost convention: a new `linear_norm` golden
kind — snippet `F.linear(x, w)` + trailing `F.rms_norm` (+ residual for the post_attn form) —
records the PAIR's total (A+epilogue+B'), the same convention `PLACE@cone: cut` uses. Seed at
decode M=32/64 on both cards; the enumeration gate additionally requires the producer be
non-atomic (which re-opens the down-site question: down g4k+sink ≈ down g4a+unsunk −1.4 µs — an
A/B at seeding time, or the last-CTA fused finalize later).

*Expected per-step savings (5090 fm, from the measured table above):* post_attn site −2.3
(k_mean 3.8→sweep 1.5), post_ff site −1.4 net (needs down back on g4k), qknorm ×3 ≈ −2.0 (sweeps
already write flat outputs; stats fold into the q/kv finalizes), pre_ff stat −1.7 (v2, via the
now-pointwise `k_mean_dcd9ba` sweep's block reduce) → ≈ −7 µs/layer ≈ **−0.35 ms/step v1, −0.5 with
v2** → serving TPOT ~19.3–19.5.

*Verification:* per-site snippet A/B (3×) → golden seeding → twin re-bench (fresh captures) →
`eval golden --in-model` MATCH/DRIFT/GAP → serving A/B with fresh packs. Numerics: the stat is the
SAME f32 sum reordered (warp/block tree vs serial) — not bit-identical to the old kernel, same
class as the coop-reduce forms; accuracy gates judge.

## Pre-WS2 probe: ring depth is NOT the bandwidth lever (both cards)

The mains run ~1.58 TB/s vs the ~1.75 floor; the cheap hypothesis was prefetch distance. Refuted:
5090 `d3/d4/tma/ring` are flat-to-worse on down (74.7/74.3/74.8) and the gate half (74.7/74.6/75.7),
o_proj regresses (11.5→12.0/12.1); 4090 `d3/d4/cp` likewise (down 137.7/138.4/142.3). The d2 ring
already keeps the next chunk in flight — the residual is the per-chunk mbarrier/syncthreads cadence
and the single-thread TMA issue, i.e. the WS2 warp-spec structure, not depth. One real find fell
out: the gate/up half at **k8 with NO split** (serial, `d2/tma/ring`) — wider chunks halve the
barrier count and the N=15360 grid stands on its own occupancy. CONFIRMED and seeded, both cards:

- **5090 m32**: k8-serial 73.4/73.5 (±0.1 over 3×) vs g4a 74.7, both lanes, both layouts — the 4
  m32 gate rows flipped (atomics-free, and a serial form is epilogue-capable — relevant to the
  future combine-fusion fork). m64 stays g4a (69.1 ≪ serial 77.9 — the split still pays there).
  Twin verify: post32 246.3 → **244.6**, post32-global → 252.3.
- **4090 m32**: k8-serial **130.0** vs g4a 137.2 (−5.3%), beating eager 133.9 — and this flip
  TIPPED the fused-vs-cut verdict: the m32 GeGLU cut (stat 1.8 + cone 1.0 + 2×~132 + combine 1.8
  = 268.8) now beats the fused megakernel's 288.1 by ~7%, where it was a wash with g4a halves. The
  `mlp_geglu.m32.cut.lin` whole-cost row is seeded on sm_89 + the three glue anchors
  (`cut_cone_stat/scale/combine.m32` — without the stat anchor the twin's bridged statistic
  cold-misdeployed to a grid-1 scalar at 81.5 µs, a 48× rescue). 4090 post32 twin:
  447.5 (session start) → **431.7 µs** (−15.8/layer ≈ −0.63 ms/step at c=1); down g4a and the cut
  all deploy from tier. Audit after the round: **MATCH 103 / DRIFT 0 / GAP 0 on BOTH cards.**

## Serving A/B — single-batch (DONE; the session's e2e proof)

Protocol: twins.db + empty online + FRESH packs (`_tune/tpot/packs-fm`), seed 0, 4K-in/4K-out, c=1,
3 prompts. (Boot gotcha reconfirmed twice: a fresh-pack fm boot exceeds the 1800 s bench health cap —
the successful run needed the nvcc cache warmed by two prior attempts; kill zombie `VLLM::EngineCore`
after any timeout, it holds ~20 GB.)

| arm (c=1, 4K/4K) | out tok/s | TTFT mean/med (ms) | TPOT mean/med (ms) |
| --- | --: | --: | --: |
| stock (re-run, NO drift) | 56.97 | 563 / 548 | 17.42 / 17.41 |
| emmy fm pre-session | 48.93 | 458 / 450 | 20.33 / 20.32 |
| emmy fm post-session | **50.17** | 451 / 447 | **19.83 / 19.71** |

The twin-level work carried through end-to-end: TPOT −0.50 ms mean (−2.5%), throughput +2.5%,
TTFT unchanged (still ~20% ahead of stock). The decode gap to stock narrowed 2.91 → 2.41 ms
(~17% of it closed); the WS1 serving exit gate (≤19.5) is 0.2 ms short — exactly the stat-sink
family's territory (~0.5 ms/step bound, designed above, not yet built) plus WS2.

The c=8 LONG workload (16 prompts, mml 8448, mnbt 4096): out tok/s 345.55 → **354.34** (0.953×
stock's 371.65, was 0.930×), TPOT 22.66/22.72 → **22.09/22.23**, TTFT 1985/1789 → 1968/1769 —
emmy still beats stock TTFT (2449/2081) on the long workload too.

## WS2 — VERDICT: the weight-streaming mains are AT the memory ceiling; warp-spec is NOT worth it

The prototype was built and measured (harness at the session scratchpad `ws2/`, driver-API cubin
loader + `cuTensorMapEncodeTiled` mirroring `_tma.py`; baseline = the production
`mlp_down.m32.lin` fm g4a kernel, reproduced in-harness at 75.8 µs / 1.57 TB/s with a passing
CPU-reference check):

| variant | µs | TB/s |
| --- | --: | --: |
| production kernel (d2/tma/ring, per-chunk syncthreads) | 75.8 | 1.57 |
| warp-spec: 9th producer warp, DEPTH-4 ring, per-slot full/empty mbarriers, NO syncthreads | 75.6 | 1.58 |
| warp-spec + 8-way split (240 CTAs) | 75.8 | 1.57 |
| pure-read probe (same boxes, no compute) | 73.7 | 1.62 |
| pure-read, 256 B rows (BK=128, unswizzled) | 73.4 | 1.62 |
| plain LDG grid-stride over the same 118.5 MB, 240 CTAs | 74.1 | 1.60 |
| plain LDG, 2720 CTAs (full occupancy) | 71.7 | **1.66** |

Every structural lever lands on the same wall: barrier cadence (warp-spec −0.3%), ring depth
(d3/d4 flat both cards), DRAM stream count (g8 flat), burst width (256 B rows flat; a B128-swizzled
TMA box hardware-caps its inner row at 128 B anyway). The plan's "1.75 TB/s floor ≈ 67 µs" was
optimistic — this card streams a 118 MB working set at ~1.66 max (92% of the 1.79 spec), and only
at full occupancy the matmul grid can't reach. The production kernel sits at **97% of its own
access-pattern ceiling** (75.8 vs 73.7); the recoverable headroom is ~2%, far under the ~10%
generalization bar. CLOSED. Corollary: the remaining serving TPOT gap vs stock does NOT live in
the big streams (per-kernel parity is real) — it is the kernel-count tail (the stat-sink family)
plus attention/graph overhead, which re-ranks the stat-sink as the next session's primary lever.

## ⚠ Correctness bug found + fixed during WS2 prep: `RegStore.atomic` dropped on stmt rewrite

Dumping the down.m32.lin g4a kernel for the warp-spec prototype exposed racing PLAIN stores in the
f16acc array-fragment (rolled-store) form: `RegStore`'s registered `_rewrite` reconstructs the stmt
field-by-field and omitted the new `atomic` flag — any σ-rewritten atomic store silently degraded
to partition-clobbering assigns (numerically wrong, no loud failure). Exposure audit: the
golden/twin/serving deploys all realize the UNROLLED form (every accuracy gate passed all session,
twin max_diff 3.1e-5), and the true-atomic timing is identical to the seeded values (74.7 µs on the
repro shape) — so no seeded number or flip decision changes. Fixed (one line + comment); regression
tests: a hardware-free `rewrite`-preserves-`atomic` unit test and an e2e f16acc/staged/rolled `g2a`
accuracy test that fails loudly (~4×-low values) if the flag is ever dropped again.

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

## WS4 — the bench-harness "hang": RESOLVED as a watchdog artifact (kernels healthy)

Measured facts (post4096-global, fm, geglu cut deployed, `--iters 3`, default warmup):

- Under the 1 s per-launch watchdog: bench_fail **5/5**, always the SAME site — `HungKernelError` on
  kernel 0 (o_proj_global, 242 regs / 96 KB smem) at **iter 10**, the first iteration after the
  warmup-extension re-calibration (+re-capture) fires the `iters_run == warmup` equality a second
  time. Batch 1, grace 1 (the first-iter grace passed — iter 0 was FAST).
- Under ANY deadline ≥ 2 s (tested 2 / 4 / 15 / 600 s + 3× at the new default): **clean 9/9**,
  TOTAL 4.97 ms with o_proj at its steady 1.25 ms — and, decisively, **no event wait ever reached
  even the 0.2 s warning threshold**. So when the deadline is ≥2 s, the >1 s stall does not merely
  fit under the cap — it does not occur at all.
- Refuted along the way: kernel deadlock (memcheck was already clean; the full bench completes with
  identical numbers), a TMA-desc delta between framings (`run_once`/`iter_once` share `_descs_now`),
  the prior session's warmup-0-is-the-variable bisect (a default-warmup run completed clean), and
  the "earlier probe abort poisons the queue" model (nothing fails before iter 10 in the full log).

The deadline-correlation below the driver line remains UNEXPLAINED (suspect: the 1 ms
`cudaEventQuery` poll loop's abort path interacting with in-flight graph-exec state on this
9-kernel / 96 KB-smem program). Shipped mitigations: default `_KERNEL_TIMEOUT_MS` 1000 → **2000**
(past the empirical cliff; real hangs still evicted in 2 s), a 30× `_FIRST_ITER_GRACE` on iter 0
(cold-start lazy SASS load / carveout / first-touch stalls are not hangs), the
`EMMY_KERNEL_TIMEOUT_MS` env override, and iteration-tagged watchdog labels. The golden YAML's
`--warmup 0` caveat is removed; verified 3/3 clean at pure defaults.

## Not reached this session

WS3 (sdpa→o_proj staging — materialize the flash-output transpose or extend the A-fill closure to
strided cp.async) was not started: it affects the whole-model `emmy run` path only, not serving
TPOT, and the session's time went to the two unplanned wins (k8-serial + the 4090 cut flip), the
RegStore.atomic correctness fix, and closing WS2 with proof. It carries over unchanged from the
research plan (whose WS1/WS2/WS4 are now executed — plan pruned per the plans/ policy). The
stat-sink design (WS1 items 2-3's real form, ~0.5 ms/step bound) is this file's "next bounded
workstream" section and is now the top-ranked decode lever, per the WS2 corollary.
