# Qwen3-0.6B kernel tuning on RTX 4090 / RTX 5090 / V100: two wins, two compiler gaps

Status: complete (FP8 tune noted below). Date: 2026-08-19. Revision: `f184aa4d`. Workflow: the `tune-kernels`
skill over the model corpus named by `experiments/golden-bench-2026/kernels/recipe.yaml`, on three caller-supplied
single-GPU hosts: RTX 4090 (sm_89, CUDA 13.3), RTX 5090 (sm_120, CUDA 13.0), Tesla V100-SXM3-32GB (sm_70,
CUDA 12.9). Recipe budget throughout: `--max-candidates 12 --patience 4 --seed 0`. The tune ranking lane compiles
at `-Xcicc -O1`; every performance conclusion below is a fresh `-O3` measurement (3 repeats, spreads 0.1-3.4%).

## The question

The recipe's models are `Qwen/Qwen3-0.6B@c1899de2` (BF16) and `RedHatAI/Qwen3-0.6B-FP8-dynamic@068a9040`
(per the recipe's GPU matrices: BF16 on all three cards, FP8 on sm_89+ only — Volta has no FP8 mma). The ask:
trace the models to extract their golden kernel sets, tune them, and where emmy trails
`torch.compile(mode="max-autotune")`, either fix by tuning or identify the gap.

## What tracing produced

`emmy trace <model> --layer 0 --seq-len 512` embeds one post-fusion inventory per card. One transformer layer of
the BF16 model compiles to **9 fused kernels** (identical set on all three cards): the input RMSNorm, three
projection linears (q: 512x1024->2048, k/v: 512x1024->1024), a q/k-norm + RoPE fusion, two SDPA fusions, the fused
post-attention norm + gate/up (the computed-A path #545 restored), and down_proj + residual. Layer 0 is
representative by construction: Qwen3-0.6B's 28 layers are structurally identical and the tuner dedupes kernels
by structure, so tuning layer 0 covers every layer (embedding and lm_head sit outside the block and are not
covered). The FP8 checkpoint traces to **19 kernels** for the same layer: per-tensor dynamic quantization emits a
`*_dynamic_fp8_scale_reduce` + `*_dynamic_fp8_bits_pointwise` pair around each linear, fragmenting the block.

## Baseline: a cold machine deploys catastrophically, and the bar is torch.compile

Whole-layer totals at `-O3` on a fresh host (empty tune DB, no goldens for these targets — "cold greedy"):

| card | eager | torch.compile | emmy cold greedy |
| --- | ---: | ---: | ---: |
| RTX 4090 | 1056 | 223 | 49209 |
| RTX 5090 | 896 | 188 | 38801 |
| V100 SXM3 | 4315 | 876 | 57369 |
| RTX 5090, FP8 (19 kernels) | 1827 | 159 | 36971 |

(µs; the tcompile totals each exclude one kernel whose torch.compile lane failed to measure.) Two readings:
at these shapes `torch.compile` beats eager ~5x, so **eager parity is the wrong goal — 188-876 µs is the bar**;
and cold greedy misdeploys by up to 250x on individual kernels (q_proj: 10026 µs vs tcompile 18 on the 5090),
which is the strongest live demonstration yet that unseeded deploys need golden evidence.

## How the tuning worked, and what it found

Per card, two equal-budget searches from the same inventory: `mcts.yaml` (pure MCTS) and `hybrid.yaml` (MCTS plus
a handful of agent proposals seeded from canonical goldens at the same M=512 shapes, in each card's own knob
vocabulary — `d2/smem-async` staging on sm_89, `d2/smem-tma` on sm_120, `d*/sync` on sm_70). Identical DB, prior,
seed, budget, cubin caches per arm. Winners were then re-measured at deployable `-O3` with exact `--ab` pins,
3 repeats, against the greedy incumbent, eager, and torch.compile.

Of the 9 BF16 kernels per card, **5 produced tunable search results; 4 could not be tuned at all** (reasons in
the gaps section). Hybrid beat pure MCTS wherever the tensor-core tier was reachable. `-O3` verification:

| card | kernel | greedy | tuned | verdict |
| --- | --- | ---: | ---: | --- |
| 4090 | down_proj (`k_linear_6b4b5f`) | 170.8 | **62.5** | **promote (2.7x)** — still 0.44x of tcompile |
| 4090 | v_proj (`k_linear_reduce_06a42b`) | 31.3 | **19.9** | **promote (1.6x)** — still 0.52x of tcompile |
| 5090 | down_proj (TMA candidate) | 36.4 | 48.6 | reject — 1.34x WORSE at `-O3` |
| 5090 | v_proj (TMA candidate) | 12.9 | 12.9 | reject — tie |
| V100 | down_proj (scalar candidate) | 301.4 | 913.4 | reject — 3x WORSE at `-O3` |

So: **10 arm runs across 3 cards, 5 tunable targets each, 5 verification candidates, 2 promoted wins (both on
the 4090)**. The rejected rows are the `-O1`->`-O3` inversion at work — the TMA candidate ranked 5.8x ahead at
`-O1` and lost at `-O3`; the ranking lane orders search, it does not predict deployment. Every conclusion here
survived 3-repeat `-O3` measurement or was rejected by it.

## The gaps: why 4 of 9 kernels cannot be tuned (and why they hold ~90% of the loss)

**Gap 1 — split-N contraction demotion (q_proj, k_proj; ~15000 µs/card).** The traced kernels fuse the
reshape-to-heads and an f32 cast onto the projection linears. The reshape splits the output axis (N=2048 becomes
16x128), so the weight is indexed by the composite `(a1*128)+a2`. The recognizer still classifies the kernel as a
contraction — but the warp-tile placer needs a single (m, n) axis pair, finds no single n, offers zero warp rows,
and by design "a contraction form with no legal row demotes back to the PLANAR reduce"
(`lowering/tile/010_recognize.py`). Result: a scalar gmem loop, ~200x off. Dtypes are innocent (both operands
f16), and the control case is in the same layer: the identical-class linear WITHOUT the fused view (`linear_2`)
gets the full warp tier and tuned 11x. **Support to add, three candidate shapes:** (a) an axis-refusion
canonicalization — the split is contiguous (`a1*128+a2`, a2 extent 128 ≡ one 2048-wide axis), so merging is
mechanical; (b) stop fusing rank-changing views into contraction stores; (c) offer a placement cut at the view
edge. (a) is smallest and keeps the fusion.

**Gap 2 — SDPA fusions cannot enter the tuner (~23000-29000 µs/card).** Every candidate for the online-softmax
kernels dies at scheduling with `"scheduling a kernel with no structural identity — 005_stamp must run first"`.
This is a *documented* hole: `lowering/tile/005_stamp_structural_features.py` states that terms `010_recognize`
splices as a `Graph` — "the online-softmax form" — have "carried no structural identity since it was written — a
separate hole." Until those splices are stamped, the SDPA fusions are unreachable by search on every card.

**FP8 fragmentation (5090).** The dynamic-FP8 checkpoint's 19-kernel layer runs 36971 µs cold vs tcompile's
159 µs. The per-linear quantize pairs are pointwise/reduce kernels the search can order but not fuse; the
structural cost is set at trace/fusion time. An MCTS tune of this corpus (budget 12, seed 0) completed with
17/19 targets ranked: the quantize-pair kernels tune fine (cheap pointwise/reduce work, 2-57 µs), but **every
FP8 linear ended on a scalar tier or a knobless row** — the tensor-core tier was never reached on any of them
(worst: `k_linear_reduce_b17b94` at 30422 µs `-O1`), and the two SDPA targets are missing for the same
stamp-hole reason as BF16. So the FP8 corpus exhibits the same two compiler gaps plus its own fusion
fragmentation. No proposal lane was run — the hybrid-vs-MCTS question was already answered on the BF16 corpus,
and fp8-tier knob spellings were not evidenced enough to propose without wasting reserved slots.

**Watchdog rejections** (2 s bench cap on some fused computed-A candidates) are the tuner working as intended,
listed only for completeness.

## Answer to the original goal

Emmy trails torch.compile max-autotune on this corpus, and the split is now exact: **tuning recovers the
reachable minority** (two verified promotions; the other tunable kernels are already at or near their
offered-space best — e.g. 5090 down_proj greedy 36 µs vs tcompile 25 µs is fusion/launch structure, not tile
choice), while **~90% of the gap sits behind the two compiler gaps above**, which no search budget can cross.
Re-running the tune will not change this: same seed replays deterministically, and the lockouts are structural.
The two promoted 4090 configurations are preserved in the working goldens (`_tune/qwen3-4090/`); canonical
promotion is deferred to the model-onboarding flow since no `recipes/Qwen3-0.6B/` lane exists yet.

## Workflow notes

- Datacenter hosts needed dependency rescues before any GPU work: V100 requires torch cu126 (cu130 wheels ship
  no sm_70 kernels) and `cupy-cuda12x==13.6.0` + `fastrlock` (cupy 14.x bundles nvrtc 13.0, which dropped Volta —
  `cp.full()` on any constant buffer dies). A `cp.full((4,),1.5,dtype=float16)` canary before tuning catches both.
- `-O1` ranking magnitudes inverted against `-O3` on 3 of 5 verification candidates this run (and in previous
  sessions); never report an `-O1` gain as a result.
- Raw artifacts (traces, arm files with rankings, tune logs, `-O3` verification JSONs, baselines) are preserved
  locally under `_tune/qwen3-{4090,5090,v100}/`; hosts are caller-owned and disposable after this report.
