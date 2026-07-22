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
the WS1 twin exit gate (≤13.8 from 14.85 captured) cleared at the bench-TOTAL level.

## Serving A/B (in progress)

Protocol: twins.db + empty online + FRESH packs (`_tune/tpot/packs-fm`), seed 0, 4K/4K workloads.
