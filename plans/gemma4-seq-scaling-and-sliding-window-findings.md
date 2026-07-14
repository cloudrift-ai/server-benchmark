# Gemma-4-12B seq-scaling + sliding-window findings — RTX 5090, 2026-07-14

Two checks on the post-golden state: (1) does the real model's sliding-window attention reach the warp flash at
seq > window; (2) do the seq-512-tuned golden picks still hold at seq 2048. Method: real-layer compiles
(`emmy compile google/gemma-4-12B-it --layer N --seq-len S --dump-dir`), pinned `--ab` A/Bs on the golden snippet
forms at M=2048, winners 3× reproduced.

## 1. Attention at seq > 1024 never reaches the warp flash — and the deployed fallback hangs

- **Layer 0 (sliding_attention, window 1024) @ seq 512**: fused warp-tier flash (mma + TMA descriptors, no mask
  operand). The window ≥ seq, so the mask degenerates to plain causal and resolves structurally — this is the only
  regime the attention goldens exercise.
- **@ seq 2048**: the HF trace materializes an explicit ADDITIVE mask tensor (the kernel gains a `_flash_ninf`
  operand). The warp tier does not realize the explicit-mask form: greedy deploys the scalar reduce partition —
  an 8.4M-block per-cell kernel that **hangs the 1 s watchdog** (bench_fail, the known unseeded-shape hazard class,
  now structural rather than a pick miss). Force-pinning the warp TILE crashes the emitter
  (`KeyError: 'acc2'` in the fragment realizer's element-values env), so this is a RECOGNITION/REALIZATION gap —
  no golden row can fix it.
- **The full-attention layer (5) at seq 2048 behaves identically** (explicit mask tensor, scalar, no mma) — at real
  context lengths EVERY gemma attention layer is on the scalar tier. Emmy attention on the real model at seq > 1024
  is currently unusable, not merely slow.
- Fix direction: teach the warp flash the explicit additive-mask form (a per-block mask tile added to the score
  fragments ahead of the softmax merge — a `FragmentApply` add, structurally like the causal `Select` realization);
  the banded sliding structure would then also admit a tile-skip analogue (skip KV blocks wholly outside
  `[m − window, m]` — the same derived-bound machinery as the causal `k_end`, applied at both ends).

## 2. Seq-512 golden picks at seq 2048: attention family transfers, split-K inverts on the shallow-K projections

All static M=2048 A/Bs, canonical-matmul snippet form (NOTE the trap hit on the way: a `functional.linear`
snippet traces transposed-B, which DECLINES cp/TMA staging entirely and silently benches gmem-direct at ~0.28× —
bench golden shapes only through the golden snippet form).

- **Attention** (synthetic causal 16h × 2048 × hd256; torch SDPA 237.6 µs): ordering preserved from seq 512 —
  fm `d1/cp/alt` nt8 **245.8** (0.97×) < fm ring nt2 255.8 < std alt nt8 263.2 < std ring nt4 271.6. The 512
  family transfers with slight erosion (1.03× → 0.97×), no pick flip. Cold greedy on the unseeded 2048 shape
  picks a scalar b256 kernel that hangs.
- **q_proj** (2048×4096, K=3840; cuBLAS 319 µs): **split-K inverts.** The std seq-512 golden (`g8k`) drops to
  388.5 = **0.82×, losing to cuBLAS**; std g2k 312 / std serial 320 sit at parity; the fm seq-512 golden (`g2k`)
  runs 242, and **fm serial wins at 223.8 = 1.43×** (3× stable). At M=2048 the output grid no longer starves the
  SMs, so the split's finalize cost outweighs its occupancy gain.
- **mlp_down** (2048×3840, K=15360; cuBLAS 1077 µs): **split still wins** — fm g2k 681 vs fm serial 872 (1.58×
  vs cuBLAS). The inversion is grid-starvation-dependent: shallow-K/moderate-N shapes shed their split at large M,
  deep-K keeps it.

## What follows

1. The explicit-mask warp-flash form is the gating item — without it none of the attention tuning matters on the
   real model past seq 1024 (and 5 of every 6 gemma layers are sliding, so this is the model's dominant regime at
   long context).
2. The golden dataset needs seq-keyed rows (at least a 2048 tier) for attention and the split-K-carrying
   projections — or an M-aware split heuristic — since the 512 picks are provably wrong (q_proj std 0.82×) and
   cold greedy at 2048 both misdeploys and hangs.
3. The emitter crash under the forced warp pin (`KeyError: 'acc2'`) deserves a guard regardless: an ineligible
   mask form should decline at schedule time, never die in codegen.
