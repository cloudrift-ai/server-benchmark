# Kernel-gap tuning: RTX 4090 (gemma-4-12B) and V100 SXM3 (qwen35 / laguna / deepseek)

Status: complete. Date: 2026-08-15. Revision: `c74bed48`. Hardware: CloudRift RTX 4090 (sm_89, driver 580.159.03)
and Tesla V100-SXM3-32GB (sm_70, driver 580.159.03), one GPU each, caller-supplied.

Scope: verify the recorded per-kernel gaps on both cards, tune the worst families, and classify every gap that
survives. Compile lanes are labelled throughout: `-Xcicc -O1` is the tune ranking lane, `-O3` (`EMMY_NVCC_FLAGS=`)
is the deployable lane. Only `-O3` numbers back a conclusion.

## Headline

Five configurations reproduced a deployable win across two repeated `-O3` runs and were promoted. The `-O1` search
ranking **systematically overstated** those wins — the largest ranked gain (3.45x) did not verify at all, and the
best verified gain is 1.89x. On both cards emmy still trails eager cuBLAS on every wide-N matmul tested.

| card | targets tuned | ranked gains >1.15x (`-O1`) | verified >1.05x (`-O3`) | promoted |
| --- | ---: | ---: | ---: | ---: |
| RTX 4090 | 16 | 12 | 4 | 4 |
| V100 SXM3 | 61 | 23 | 1 | 1 |

## RTX 4090 — the wide-N MLP block

### Baseline

`scripts/bench_golden_set.py --filter gemma4_12b`, backends `eager,tcompile,emmy`, `torch.compile` at
`mode="max-autotune"`: 150 cases, 146 measured, 4 failed.

- **vs `torch.compile` max-autotune: emmy wins 67 of 81** comparable cases. The 14 losses are led by
  `mlp_gate_up_split.m4096.lin` (0.44x) and the `attention.hd512` family (0.56–0.81x).
- **vs eager cuBLAS: 88 of 146 losing.** cuBLAS, not `torch.compile`, is the bar that matters on this card.

### What was tuned and what verified

Equal-budget arms (`--max-candidates 24 --seed 7`, separate DB / prior / cubin cache per arm) over 16 targets: the
three worst families plus six new realizations filling the m512 / m1024 / m2048 coverage hole (the recorded set
jumped from m256 straight to m4096).

The hybrid arm beat MCTS-only on 13 of 16 targets at `-O1`. Every hybrid winner carried `RASTER: gm8` — the
L2-friendly grouping that the one parity-reaching family (`mlp_gate_up`) already used and that every big loser
lacked — plus a wider `f2x8/k2` fragment and real staging (`d2/cp`).

At deployable `-O3`, over two repeated runs each (spread <= 3.9%):

| target | eager | golden | tuned | vs golden | vs eager | `-O1` had claimed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mlp_gate_up.s2048` | 2833.3 | 5684.7 | 3001.3 | **1.89x** | 0.94x | 3.38x |
| `mlp_gate_up_split.m4096.lin` | 2796.7 | 4170.7 | 3080.4 | **1.35x** | 0.91x | 3.07x |
| `gate_up_cat.m512.lin` | 726.5 | 990.2 | 850.0 | **1.16x** | 0.85x | 2.45x |
| `gate_up_cat.m2048.lin` | 2791.9 | 3467.9 | 3073.5 | **1.13x** | 0.91x | 1.31x |
| `gate_up_cat.m1024.lin` | 1467.7 | 1533.5 | 1510.0 | 1.02x | — | 1.36x |
| `gate_up_cat.m64.lin` | 287.7 | 274.9 | 276.6 | 0.99x | — | — |

The direction is right and the schedule family is now established; the magnitude was largely an `-O1` artifact.
**No tuned 4090 configuration reaches eager** — the best is 0.94x, so roughly half the recorded gap remains.

## V100 SXM3 — staging is offered, and mostly already found

The recorded V100 goldens all carry `STAGE: ''` (no shared-memory staging) on their losing rows, which suggested a
uniform search shortfall. A direct `-O3` A/B on `attn_kv.m512.lin` confirmed staging is legal and large on Volta:
527.9 us recorded config -> **217.9 us** with `STAGE=d2/sync` pinned, same other knobs (2.4x).

That result does **not** generalize. Across 61 tuned targets, 23 ranked >1.15x at `-O1` and exactly one survived
`-O3` verification:

| target | eager | golden | tuned | vs golden |
| --- | ---: | ---: | ---: | ---: |
| `qwen35_122b.attn_q.lin.dynM` | 665.6 | 2774.0 | 1850.4 | **1.50x** |
| 10 others verified | — | — | — | 0.99–1.03x |

Baseline (348 cases, 327 measured): where a torch reference exists at all, **39 of 40 lose to eager**, worst
`gdn_qkv.m512.lin` at 0.02x (41689 us vs 634 us). The promoted target still runs at 0.36x of eager. Volta's
synchronous-only copy path (no `cp.async` below sm_80) and the older `m8n8k4` atom are codegen-tier limits that a
knob search cannot remove; this is the classification for the V100 gap as a whole.

**Recorded-vs-measured drift**: `attn_q.lin.dynM` records `emmy_us: 7615.488` but its recorded configuration
measured 2774.0 us here — a 2.7x discrepancy on the same knobs. The other rows reproduced. Worth an audit pass;
the promoted row carries this session's measurement.

## Gaps that blocked promotion

1. **m4096 miscompile (correctness, not performance).** Four cases fail the accuracy check on `c74bed48`:
   `mlp_geglu.m4096` differs on **12.0%** of elements (greatest absolute difference 96.0), `mlp_geglu.m4096.lin` on
   0.5% (32.0); `gate_up_cat.m4096.lin` and `norm_gate_up.m4096.lin` fail pinned-Loop verification for want of a
   valid greedy baseline. This is the single largest recorded-loss family, so its gap cannot be closed until the
   miscompile is fixed. Repro logs: `_tune/4090-gaps/repro/`. **Classification: codegen correctness bug.**
2. **`pin_unmatched` on three `norm_gate_up` targets** (`m1024`, `m2048`, `lin.dynM`): neither the recorded golden
   nor the tuned candidate could be realized — the compiler no longer offers those knobs at these shapes. A golden
   the compiler cannot produce is worse than none, so these need an eligibility review before re-recording.
   **Classification: eligibility / optimization lockout.**
3. **`greedy (isolated)` watchdog failures.** On several wide-N shapes the greedy pick is catastrophic —
   `gate_up_cat.m1024.lin` deploys `w8x2/f4x8/k8` with no staging and no raster at **95687 us** against a 1467 us
   eager baseline (65x), tripping the 10 s bench watchdog. The golden tier masks this in normal deploys; it is
   visible only when a golden is absent or unmatched. **Classification: search shortfall, and the strongest
   argument that the fix belongs in the prior rather than in per-card goldens.**
4. **Laguna FP8 could not be tuned at all — environment, not compiler.** All three targets die in
   `cp.full()` (`emmy/compiler/backend/cuda/program.py`, the constant-buffer path) with
   `nvrtc: invalid value for --gpu-architecture`. Cause: `cupy-cuda12x 14.1.1` bundles **nvrtc 13.0, and CUDA 13
   dropped Volta**. Emmy's own kernels are unaffected (system `nvcc 12.9` still accepts `sm_70`), so only
   constant-buffer kernels fail. Fix: pin `cupy-cuda12x==13.6.0` on any sm_70 host.

## Tooling fixes made during the run

`scripts/bench_golden_set.py` was broken in two ways and produced no usable output before these:

- it passed a golden **name** to `emmy run --golden`, which takes a **path** plus `--target` — every case failed
  instantly with "invalid golden file";
- it compared `torch.cuda.get_device_name(0)` directly against the recorded `gpu_name`, which never matches on
  datacenter cards (`Tesla V100-SXM3-32GB` vs `NVIDIA Tesla V100 SXM3 32GB`), so the V100 corpus resolved to zero
  names. Now canonicalized through `emmy.gpu.by_name`, matching the deploy-time join.

## Recommendations

1. **Fix the m4096 miscompile first.** It blocks the largest gap on the 4090 and is a correctness defect.
2. **Treat the wide-N schedule as a prior problem.** `RASTER: gm8` + `f2x8/k2` + `d2/cp` won on every wide-N shape
   tuned, yet greedy deploys `w8x2/f4x8/k8` with neither staging nor raster and lands 65x off. Per-card goldens
   paper over this one shape at a time; the prior should learn it.
3. **Do not trust `-O1` ranking magnitudes.** Ranked gains ran 1.8–3x optimistic here, and one 3.45x ranked winner
   failed to realize. Budget `-O3` verification into every tuning session.
4. **Pin `cupy-cuda12x==13.6.0` for sm_70** in the V100 setup notes.
5. **V100 expectations should be reset.** Its goldens are already near the search's reach; the remaining 2–50x gap
   to cuBLAS is codegen-tier (no `cp.async`, `m8n8k4` atom), not something tuning will close.

## Artifacts

`_tune/4090-gaps/` and `_tune/v100-gaps/` (untracked): baselines, both arms per corpus, tune logs, `-O3`
verification JSON/logs, and the miscompile repro logs.
