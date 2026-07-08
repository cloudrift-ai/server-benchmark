# Gemma-4-12B RMSNorm kernel tune findings — RTX 4090 (sm_89)

> **CORRECTION (2026-07-07, nsys-verified):** the per-kernel µs in this report are **cross-labeled** — `run --bench`
> paired `per_launch` times to kernels by graph **dict** order while the backend launches in **topo** order (fixed in
> `_launch_order_cuda_nodes`, run.py). The "626–1540 µs norm epilogues" are actually the **q/k projections running
> scalar tiles** (the norm's f32 output feeds the f16 weights — a mixed-dtype matmul the f16 MMA tier never takes);
> the norm kernels themselves are 4–5 µs and healthy. e4bb2c's "98 ms" is its reproducer's down-projection running
> scalar, not the norm and not a hang; its 18 µs -O1 measurement was the true norm cost, not a wrong-bench. Findings
> 2–3 below are superseded accordingly; the real open item is the **mixed-dtype scalar matmul fallback**.

- **Scope:** targeted tune of the 7 RMSNorm-carrying kernels of `google/gemma-4-12B` layer 0 on a rented CloudRift
  RTX 4090 (24 GB, driver 580.65.06, CUDA 12.9, torch 2.12.1+cu130). Static shapes at the seq_len=512 hint. Each
  kernel tuned from its dump reproducer (`emmy tune <k>.torch.json --bench`), then greedy-benched
  (`emmy run <k>.torch.json --bench`) off the freshly written prior. NOT a full-layer tune — companion to the
  sm_120 campaign in `gemma-4-12b-layer0-tune-findings.md`, which found these kernels mostly *winning* vs eager.
- **Kernel inventory** (layer 0 compiles to 19 kernels; 7 carry `rsqrt`): 1 standalone input RMSNorm (`k_mean`,
  512×3840), per-head Q/K-norms over head_dim 256 fused behind the q/k projections (`k_mean_linear_reduce
  323fd5/a58cd7/48c05d`), the pre-FFN norm fused into the gate/up two-channel contraction (`k_linear_mean_reduce
  b87781`, 512×15360×3840), one norm→linear MLP edge (`33f743`), and the post-FFN norm + residual + layer-scale
  pointwise (`e4bb2c`, 512×3840, no matmul despite the provenance name).

## Results — per-op -O3 bench, µs (seq 512, fp16)

`emmy` = greedy pick after tune (whole-op e2e where the standalone compile split the op). `–` = the slicer wired
no torch reference (partial-coverage reproducer; no single torch op for the fused chain).

| Kernel | Layer op | eager | tcompile | emmy | emmy/eager |
|---|---|--:|--:|--:|--:|
| `k_mean_9b7435` | input RMSNorm (512×3840) | 199 | 14 | **18** | **0.09×** |
| `k_mean_linear_reduce_33f743` | norm → linear MLP edge | 335 | 122 | 227 | 0.68× |
| `k_mean_linear_reduce_323fd5` | k_proj + per-head K-norm (8×256) | – | – | 640 | – |
| `k_mean_linear_reduce_a58cd7` | q-side per-head norm | – | – | 645 | – |
| `k_mean_linear_reduce_48c05d` | q_proj + per-head Q-norm (16×256) | – | – | 1542 | – |
| `k_linear_mean_reduce_b87781` | pre-FFN norm → gate/up (512×15360×3840) | – | – | 2550 | – |
| `k_mean_linear_reduce_e4bb2c` | post-FFN norm + residual + scale | 635 | 385 | **hangs** | ∞ |

In the split ops the matmul halves are healthy: q/k_proj land on MMA partial+finalize pairs at **12–13 µs total**
(`n16x8/f4x26`, `g8k`/`g2k`). The norm epilogues are the whole problem — 98–99 % of each op's time.

## Findings

### Finding 1 — simple RMSNorm is healthy; near the sm_120 relative numbers

`k_mean` tunes to 18 µs = **11× faster than eager** (199 µs), within 1.3× of torch.compile (14 µs). ~655 GB/s
effective on a ~1 TB/s card. Consistent with the 5090 campaign's `k_mean` 0.08–0.13× ratios. No action needed
beyond the golden seeding below.

### Finding 2 — per-head QK-norm epilogues run thread-serial: a recognition gap, NOT a prior/ranking miss

**(Root cause corrected after the golden-seeding session — see `golden-sweep-rtx4090-findings.md`.)** The greedy
picks for `323fd5`/`a58cd7`/`48c05d` norm kernels bench at 626–1540 µs for 2–4 MB of memory-bound work (~10–30 µs
at roofline). The generated CUDA shows why: **one thread per (row, head)** — `_gid < 4096` guards off 127/128
launched threads, each survivor serially reducing + sweeping 256 elements. The table's `REDUCE=b32` never
materializes as a cooperative reduce. Two proofs it's not the prior: (a) a standalone RMSNorm with the *same
geometry* (4096×256, seeded as golden `gemma4_12b.qknorm.k256`) picks the same-looking config and runs **4.5 µs —
139× faster, beating torch eager**; (b) after seeding rms_norm goldens + priors for exactly these shapes, the
fused kernel's greedy pick is byte-identical (625 µs). The difference: the standalone kernel recognizes as
RMSNorm (`Map(body=sweep, source=Reduction)`, reduce-tier coop enumeration — its variant table shows the serial
lane at rank 8/8, 1800 µs, correctly rejected), while the fused kernel's mean-count and epsilon arrive as
**materialized 1-element buffers** (`mean_2_count[0]`, `add_3_c1[0]`) instead of constants, the Reduction lift
never fires, and the coop-reduce fork is never offered to the search. Fix belongs in the lift/recognize pass
(loop dialect), not in goldens or prior weights.

### Finding 3 — `e4bb2c` (post-FFN norm+residual) greedy pick hangs: prior poisoned by a wrong-bench winner

During its tune the "best" measurement was **18.152 µs @ bench #252** — implausibly fast for an op eager does in
635 µs — with 99 % of post-warmup benches flagged silly (≥2× best). The -O3 re-bench of that winner measured
~98 ms, and the subsequent greedy `emmy run --bench` **timed out: "benchmark run stage exceeded 10.0s of GPU
time"**. Same `HungKernelError`/timeout class the sm_120 campaign logged as wasted search slots (its Finding 4),
but here the mis-measured variant *won*. **Follow-up (same day): an isolated `--clean --explore-eps 0.25` retune
reproduced it exactly** (best 18.15 µs at -O1, ~98 ms at -O3, greedy run over the 10 s bench budget = 100 iters
× 98 ms — a 98 ms kernel, not a hang), so it is a deterministic wrong-bench at -O1 ranking, not prior poisoning,
and `eval failures` is blind to it (the bogus rows are "ok" rows). Needs the golden A/B integrity gates
(wrong-answer + AI-floor) ported into tune's winner selection; the Finding-2 lift fix likely dissolves the bad
lane too. Full triage in `golden-sweep-rtx4090-findings.md` Finding 3.

### Finding 4 — norm→gate/up contraction works but leaves ~2–3× on the table

`b87781` picks a real MMA schedule (`mma_m16n8k16_f16/w2x2/f4x8/k2`, `d1/sync`, block 128, smem 24.5K) — the
fork-sibling compute-fill lane from the tile docs — at 2550 µs ≈ 47 TFLOP/s effective for the 2-channel
512×15360×3840 contraction. regs 255 / occ 17 % says register pressure is capping it. cuBLAS-class would be
~1 ms. Same "fused norm→linear reduce is the remaining deficit" conclusion as sm_120's Finding 2.

## Recommended next steps

1. **Seed `rms_norm` goldens for `rtx4090_sm89.yaml`** (tune-golden flow): at least `k3840` static + `.dynM`
   (gemma-4-12B hidden), ideally also a per-head `k256` shape — mirror the 5090 seeds. This is the cheapest
   fix for Finding 2 and directly serves greedy deploys on 4090 fleets.
2. `emmy eval failures` on the remote DB for the `e4bb2c` hang cluster; re-tune that shape `--clean
   --explore-eps 0.25` once the winner can't be a hang. Consider porting the golden A/B integrity gates
   (AI-floor + wrong-answer) into tune's winner selection.
3. NCU the scalar-lane QK-norm picks vs a hand-pinned coop-reduce variant (`emmy eval variants --kernel
   mean_linear_reduce`) to confirm the lockout is ranking (prior) and not a missing sm_89 moveset.
4. Re-run this sweep `--dynamic seq_len@x:1` for the deployable masked-tile artifact (this sweep was static).

## Repro / artifacts

- Remote box (billed until deleted): CloudRift `zonked-garden-4497`, id `311bd186-79df-11f1-95dd-135c01e2c5e4`,
  `ssh -p 57007 riftuser@211.21.50.85`. nvcc is NOT on PATH on this image — `export CUDA_HOME=/usr/local/cuda`.
- On the box: dumps `~/dumps/gemma4-l0/` (kernels + `.torch.json` reproducers), logs `~/tune-logs/*.log`,
  tune DB + prior `~/.cache/emmy/`.
- Local copies of the tuned artifacts: `~/.cache/emmy/rtx4090-gemma4/{autotune.db,prior.json}`.
- Reproduce any row: `emmy tune ~/dumps/gemma4-l0/08_lowering_cuda.kernels/<k>.torch.json --bench` /
  `emmy run <same>.torch.json --bench` with `CUDA_HOME=/usr/local/cuda`.
