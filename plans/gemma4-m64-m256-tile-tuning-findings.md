# Hand-tuning the gemma-4-12B decode widths on the RTX 5090 — the tile-M rule

Goal: raise emmy's throughput on the published gemma-4-12B / RTX 5090 table by tuning kernels **by hand** (the cost
prior and `emmy tune`'s sweep are not trusted at the moment, so every number here comes from an explicitly pinned
`emmy run --ab` A/B against live cuBLAS).

## Short version

- **One mistake explains every large kernel-level gap I found, and it is a single rule about one knob.** At a decode
  width M, a wide-N streaming edge needs an output tile whose **M extent is exactly M**. Every recorded golden at
  m64 and m256 violated it. Fixing it is worth **1.56x** on the biggest decode kernel at m64 (`gate_up_cat`
  228.3 → 146.7 us, from 1.48x cuBLAS to **0.95x**) and **1.63x** at m256 (475.1 → 291.8 us).
- **At m256 the fix reaches the form serving deploys, and it is the session's real result.** The fused
  `norm_gate_up.m256` edge deploys `PLACE@cone: cut`, so its matmul resolves against the plain `gate_up_cat.m256`
  shape — recording the fixed tile there takes that edge from **483.5 → 314.4 us (1.54x)**, verified by benching
  the fused shape after the edit. That is ~169 us per layer, ~8 ms per decode step over 48 layers, on the bucket
  the c=64 MTP-d2 cell runs at.
- **At m64 it does not reach serving**, because that edge deploys *uncut* (168.9 us, already 1.03x cuBLAS). The
  best config for it — cut + the fixed tile, 152.1 us, 1.11x — **cannot be recorded in the golden format**: an
  entry pinning `PLACE@cone` and `TILE` together does not realize, and an unrealizable entry there is not neutral
  but catastrophic (see "The golden format cannot express the fused winner"). So the m64 gain is measured but
  undeployable today.
- **The three `gate_up_cat.m64` goldens were dead.** The deploy logs `DRIFT` for that shape — "no offered candidate
  realizes any of them" — so the single biggest decode kernel had no golden floor at all and was resolved by the
  prior. This is a bug in its own right and it is the mechanism by which an untuned width becomes a lottery.
- **The prefill width is NOT the problem.** m2048 — the chunk width the 4k/4k c=4/c=8 cells ride — measures
  0.99–1.08x cuBLAS across every matmul edge, and the best pin buys 1.01–1.05x. So the 4k/4k TTFT gap (24–26 s
  against stock's 8.7 s) is not prefill matmul quality.
- **Several of the briefing's load-bearing claims are wrong.** In particular "emmy's advantage is gone by M=32" is
  a population artifact — on a same-family sweep emmy wins at *every* M, and the weakest width is **M=64**, which
  is exactly the width nobody had looked at. Details in "Corrections".

---

## The tile-M rule

`gate_up_cat` at m64 is `linear(A[64, 3840], B[30720, 3840])` in fp16. Its B slab is 30720·3840·2 = **236 MB**, so
at 1.79 TB/s of DRAM the floor is **132 us** and cuBLAS's 154 us is 86% of peak — the kernel is bandwidth-bound, and
the only thing that matters is reading B exactly once while keeping the SMs fed.

The `TILE` codec spells an output tile as `w<WM>x<WN>/f<FM>x<FN>/k<BK>`, so the tile's M extent is
`WM · FM · atom_m` (16 for `mma_m16n8k16`). Three regimes, all measured on the same shape, same box, same run:

| tile M | example geometry | grid | measured | vs cuBLAS | why |
|--:|---|--:|--:|--:|---|
| 32 | `w1x8/f2x2/k4` | 480 | 228.6 us | 1.48x | two M-blocks, each streams the whole B slab — B read **twice** |
| 128 | `w4x2/f2x4/k2` (the recorded golden) | 480 | 228.3 us | 1.48x | B read once, but **half** the mma is padding |
| 256 | `w4x2/f4x8/k4` | 240 | 174.8 us (fm) | 1.13x | B read once, three quarters of the mma wasted |
| **64** | `w1x8/f4x1/k8` | 480 | **146.7 us** | **0.95x** | B read once, no wasted mma → 1.61 TB/s = 90% of peak |

The tile-M=64 family is flat and wide — `w1x8/f4x1`, `w2x4/f2x2`, `w4x2/f1x4`, `w1x8/f4x4` all land 146.7–152.2 us,
so the win is attributable to the M extent and not to one lucky geometry. Two secondary knobs matter once the M
extent is right: `k >= 4` is required (the same winner at `k2` falls back to 222.9 us), and at m256 `RASTER=gm8`
(group-M rasterization for L2 reuse) is part of the winning row — no std-lane record at that width had it.

**Where the rule does not apply.** It is a bandwidth argument, so it only bites when B-streaming dominates. On the
narrow-N and long-K edges the tile-M=64 family is 3–6% *slower* and the existing `w1x8/f2x2` records stand: `q_proj`
(N=4096, its 31 MB slab is L2-resident on a 96 MB L2), `o_proj`, `o_proj_global`, `mlp_down` (K=15360, K-bound),
`kv_proj`, `k_proj_global`. That asymmetry is why a single tile family cannot be applied blindly across a width.

## What changed

Three entries in `emmy/compiler/pipeline/search/goldens/rtx5090_sm120_gemma4.yaml` were replaced (better → replace,
per the goldens' own convention), each 2x reproduced at <1% spread, all with clean integrity flags (no pin mismatch,
no wrong answer, no arithmetic-intensity floor):

| golden | knobs before | before | after | after / cuBLAS |
|---|---|--:|--:|--:|
| `gate_up_cat.m64.lin` | `w4x2/f2x4/k2` | 233.6 us | **146.7** (`w1x8/f4x1/k8`) | 0.95x |
| `qkv_cat.m64.lin` | `w1x8/f2x2/k2`, `g8k` | 34.0 us | **28.6** (`w2x4/f2x4/k4`, `g2a`) | 1.07x |
| `q_proj_global.m64.lin` | `w1x8/f2x2/k2`, `g8k` | 33.9 us | **28.6** (`w2x4/f2x4/k4`, `g2a`) | 1.07x |
| `gate_up_cat.m256.lin` | `w4x2/f2x4/k2`, `g4k` | 481.4 us | **291.8** (`w4x2/f2x4/k4`, `gm8`) | 1.07x |

Every one was verified to *deploy*, not just to pin — after the edit the default (greedy) resolve of each shape
lands on the new row and reproduces its latency, and the DRIFT warning is gone:

| shape | greedy before | greedy after | vs cuBLAS after |
|---|--:|--:|--:|
| `gate_up_cat.m64` | 228.3 us | **146.8** | 0.94x |
| `qkv_cat.m64` / `q_proj_global.m64` | 35.1 us | **29.2** | 1.05x |
| `gate_up_cat.m256` | 475.1 us | **283.6** | 1.04x |
| `norm_gate_up.m256` (fused, deploys cut → floored by the row above) | 483.5 us | **314.4** | 1.11x |

`qkv_cat.m64` and `q_proj_global.m64` are the same shape (N=8192, K=3840); they are recorded separately because the
golden index is name-keyed, and both are needed so either edge floors the deploy.

The replacements were checked for the failure mode that made the old ones useless: after the edit the deploy no
longer logs `DRIFT` for these shapes, i.e. the recorded knobs are realizable under the current enumeration.

## Negative results

These cost real time and are worth as much as the wins, because three of them close off directions the briefing
proposed.

- **The prefill width (m2048) is already at parity.** Sweeping tile M 128/256 × tile N 64/256 × `RASTER=gm8` over
  the six biggest m2048 edges: `gate_up_cat` greedy 2197.4 vs eager 2033.9 (1.08x), best pin 2127.4 (1.05x);
  `mlp_gate_up_split` 1108.8 → 1060.2 (1.01x); `mlp_down` 1113.0 → 1110.4; `o_proj` and `qkv_cat` — no pin beat
  greedy at all (0.99–1.02x already). **So the 4k/4k c=4/c=8 gap is not prefill matmul quality.** Note also that
  the YAML's recorded m2048 numbers are stale: it records `gate_up_cat.m2048` at 3644.6 us against cuBLAS 2302,
  i.e. 1.58x, and the live measurement today is 2197 against 2034 — 1.08x. Any target selection that reads ratios
  out of the YAML at that width is reading a two-generations-old compiler.
- **The fused serving form gains far less than the isolated golden.** `norm_gate_up.m64` — the form the goldens'
  own comment says "owns serving" — is only 1.03x cuBLAS to begin with (168.9 vs 164.6), so the best pin
  (`PLACE@cone=cut` + `w1x8/f4x1/k8` + `d2/tma/ring`, which splits the statistic prologue into a 1.4 us stat and a
  1.2 us cone kernel and lets the matmul ride the TMA ring) reaches 152.1 us — **1.11x**, 16.8 us per layer. Over
  48 layers that is ~0.8 ms per decode step. Tuning the isolated `gate_up_cat` golden looks 1.56x better than it
  is from the server's point of view.
- **`norm_qkv.m64` is not improvable and was never broken.** Its greedy resolve already picks `w2x4/f2x4/k4`+`g2a`
  — the same tile my sweep found for the unfused edge — landing at 33.8 us against eager 30.8. Best pin 32.3 us,
  inside noise.
- **Six of the nine m64 matmul edges have no win available**: `q_proj` (16.5 greedy / 15.5 eager, best pin 17.0),
  `o_proj` (16.7 / 16.4, best 17.3), `o_proj_global` (29.5 / 24.6, best 30.2), `mlp_down` (78.7 / 84.1 — already
  0.94x), `mlp_gate_up_split` (69.9 / 79.5 — already 0.88x), `kv_proj` and `k_proj_global` (already 0.92x / 0.62x).
- **Split-K and deeper pipelining do not help the fixed tile.** On the m64 winner, `REDUCE=g2k` costs 6% (155.8),
  `g2a` costs 9% (160.3), `d3/tma/ring` and `d2/cp/ring` are within noise of `d2/tma/ring`, and `RASTER=gm8` is
  exactly neutral (146.7 either way). The tile M extent is the whole effect at this width.
- **The m1/m8/m32 widths need nothing.** At m32 the recorded `gate_up_cat` golden is `w1x8/f2x2/k8` — tile M 32 for
  a 32-row step, i.e. the rule already satisfied — and it measures 0.97x cuBLAS. This is the structural reason the
  c=8 MTP cells (decode bucket 32) are not a decode-kernel tuning gap.

## Corrections to the briefing's claims

**Claim 2 — "emmy's advantage over cuBLAS is concentrated at narrow widths and gone by M=32" — WRONG.** The
article dataset's cross-M medians compare different kernel populations at each M. Restricting to families that
appear at more than one width (the goldens carry both `emmy_us` and `cublas_us` per entry, so this is a free,
apples-to-apples check), emmy wins at every width:

| M | 1 | 8 | 16 | 32 | 64 | 256 | 512 | 2048 | 4096 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| median emmy/cuBLAS | 0.32 | 0.72 | 0.43 | 0.68 | **0.96** | 0.77 | 0.76 | 0.70 | 0.75 |
| families winning | 9/9 | 20/21 | 8/8 | 24/27 | **15/22** | 24/28 | 19/24 | 19/20 | 15/19 |

There is no cliff at M=32 and no loss of advantage at large M. The **one** weak width is M=64, and the reason is
the tile-M rule above, not a regime change. This inverts the briefing's target selection: it argued the wide widths
were compute-bound territory where cuBLAS deserves to win; in fact emmy's recorded kernels beat cuBLAS by ~1.3x at
m2048/m4096 and the outlier is the narrow-ish m64. (Caveat: these are recorded numbers and the m2048 rows are
stale, as noted above — the live m2048 ratios are ~1.0x, not 0.70x. The claim being refuted, that emmy's advantage
*ends* at M=32, is unaffected either way.)

**Claim 3 — "the c=8 MTP deficit is not a tuning gap" — CONFIRMED, and for the reason claimed.** m32 satisfies the
tile-M rule already and measures 0.97x cuBLAS on the dominant edge. I found no m32 win.

**Claim 6 — "the published image freezes kernel choices, so tuning may have no effect through docker" — TRUE but
much narrower than stated, and the docker `ARCHITECTURE.md` is misleading on it.** The pack key is
`{kind, model_type, config_sha, dtype, decode_bucket, max_tokens, prefill_bucket}` and `load_pack` compares the
whole dict with `!=`, then hashes it into the pack *directory* name. So a bucket override is a **whole-pack** miss,
not a decode-programs miss: the boot re-runs trace + passes + fork resolution + codegen for **every** program, and
on a miss the wheel-shipped goldens **are** consulted (`_golden_evidence_index` → `_golden_pick`). Consequences for
the published table:

| lane | `decode_bucket` | `prefill_bucket` | vs the image's warm (32 / 4096) | goldens govern it? |
|---|--:|--:|---|---|
| c=1 (all depths at bucket 32) | 32 | 4096 | hit | **no** — picks frozen in the pack |
| MTP c=4, c=8 (d2 and d3) | 32 | **2048** | **miss** | **yes** |
| c=64 no-MTP | **64** | 4096 | **miss** | **yes** |
| MTP c=64 (d2) | **256** | 4096 | **miss** | **yes** |

So kernel tuning reaches every batched cell in the table through the docker path; only the c=1 cells are frozen.
The `EMMY_GEN_PREFILL_BUCKET=2048` knob on the 4k/4k lanes is what makes them pack misses even though their decode
bucket matches the warm.

**Claim 5 — "attention is entirely vLLM's; emmy owns embed, norm+QKV+RoPE, o_proj+MLP, final norm" — right about
attention, wrong about the boundary.** RoPE is **not** emmy's: the split wrapper returns un-rotated q/k and vLLM's
fused `rotary_embedding` CustomOp runs between the `pre` and `post` halves. The final norm is a plain torch module
on CUDA, and the embedding is a torch gather, not a compiled kernel. And the important half of the claim holds:
there is **no** query-length branch anywhere in the runner — a spec-decode verify step of `n_seqs·(depth+1)` tokens
is a byte-identical invocation to a plain decode step of the same total token count, because the twins are traced
over a flat `[num_tokens, H]` layout with no mask operand and no sequence structure. All tree/causal structure is in
vLLM's `attn_metadata`.

**Claim 7 — "a wide bucket corrupts output because the twin's padding is not neutral" — the mechanism is wrong,
and the replacement is a more serious bug.** Padding is *structurally* inert: every reduction in the twins is over
H, K or head_dim, never over the token axis; vLLM only ever receives `[:t]` slices, so no padded lane reaches a KV
write or the sampler. (It is worth knowing that the device path does not zero-pad at all — it uploads `t` rows into
the buffer prefix and leaves rows `[t:bucket)` holding the previous step's values, which under post→pre chaining is
a free-running 48-layer fp16 recurrence that reliably goes to ±inf. That is harmless *only* while every deployed
kernel is genuinely row-separable.) Two facts fit the bucket-256 failure better:

1. **Nothing in the deploy path checks numbers.** `_golden_pick` ranks candidates by recorded microseconds only;
   there is no reference comparison anywhere in `CudaBackend.compile` or `build_from_plan`. A legal-but-wrong knob
   row deploys silently. The goldens' own header already documents that cold greedy at unseeded widths reaches
   kernels that *hang*; a wrong answer has no such backstop.
2. **At bucket 256 the decode twin also does prefill.** The step classification in `EmmyGenModel.forward` is a pure
   width test — `decode_ok = 0 < t <= decode_bucket` — with no consultation of `attn_metadata`. So with
   `EMMY_GEN_DECODE_BUCKET=256`, *every prompt shorter than 257 tokens is prefilled through the m256 decode twin*.
   A defective m256 twin therefore corrupts prefill as well as decode, which fits a flat 0.0 far better than a
   partial degradation — and at bucket 32 those same prefills went through the well-covered symbolic programs.

My m256 measurement supports the "the m256 resolve is bad" half of this directly: the greedy pick for the dominant
m256 edge is **1.74x** off cuBLAS and **1.63x** off the best available config (475.1 vs 291.8 us). That width is
resolved cold in production and resolved badly.

**Claim 8 — "no well-covered width between m8 and m32; seeding the fused twins at m16 is the most promising lead"
— the coverage claim is correct, the lead is not worth taking.** m16 does carry only unfused projections. But no
lane in the table lands on m16: the MTP c=4 lanes moved to bucket 32 in #441, and the tile-M rule says a 16-row
step on a 32-row-tiled kernel wastes half the mma while a 24-token step padded to 32 wastes a quarter — a *smaller*
penalty than seeding a whole new tier would repay. The productive version of that lead is what I did instead: the
widths that *are* deployed (64 and 256) had records that violated the tile rule.

**Claim 1 (bucket, not step width, selects the kernel) and claim 4 (padding to the bucket is nearly free)** are
consistent with everything I saw and I did not re-measure them.

**Claim 10 — the recorded accuracy failures.** I did not reproduce them, but the greedy-pick scan I did run (10 of
the m64/m256 golden shapes, benched for integrity flags rather than latency) produced **no** flagged rows, so
whatever those four entries record is not reproducing as a deploy-time wrong answer on those shapes today. The
scan was cut short in favour of the tile sweep; the remaining 49 shapes are unchecked.

## The golden format cannot express the fused winner — and failing to express it costs 323x

The best configuration for the fused m64 gate/up edge is `PLACE@cone: cut` **plus** a pinned `TILE`. Recording both
keys in one entry does not realize: the deploy logs `DRIFT` for the shape and falls through. That fallthrough is
not a graceful degradation — it lands on a per-cell `fuse`/`b128` scalar config and the edge measures **54.6 ms**
instead of 168.9 us, a **323x** regression, reproduced on both the gate/up (54625 us) and the global qk
(14825 us) edges before the entries were reverted. Two things follow:

- **The 1.11x fused-m64 win is real but undeployable** through the goldens as they stand. Making it deployable needs
  either a golden format that can pin a cone placement and the tile it implies together, or a `PLACE` default at this
  shape that cuts. Both are code changes, out of scope here.
- **An unrealizable golden is a live hazard, not dead weight.** The recorded uncut entry at this shape is the only
  thing standing between the deploy and a 323x misdeploy. That reframes the DRIFT class below: it is not merely
  "the golden stopped helping", it is "the safety net silently came off".

The m256 edge escapes this because it already deploys cut, so its matmul resolves against the plain
`gate_up_cat.m256` shape and a plain-matmul record floors it — which is why the m256 change lands and the m64 one
cannot.

## The DRIFT bug — the biggest decode kernel had no golden floor

Every compile of the m64 gate/up edge logs:

```
deploy: node 'linear' matches golden shape ShapeKey(free_prod=1966080, reduce_max=3840, is_warp=True,
is_dyn=False, kind='', free_max=30720) (3 recorded entries), but no offered candidate realizes any of them
— the golden(s) no longer realize under the current enumeration; falling through to the normal evidence
hierarchy. Investigate enumeration drift for: gemma4_12b.gate_up_cat.m64.lin ×3
```

All three recorded entries for the single largest kernel in the decode step were unrealizable, so the shape was
resolved by the prior — the tier the briefing says is broken. The entries still *pin* fine (`--ab` reproduces their
latencies exactly), which is why the isolated golden-reproduction check passes and only the in-model audit catches
it. The same audit reports **GAP 86** against the 4090's golden file for these twins, so this is not a one-shape
accident.

This matters beyond the one shape: it is the concrete mechanism behind "an untuned width is a lottery". A width
whose goldens do not realize is indistinguishable, at deploy time, from a width with no goldens at all.

## What is not established

- **No end-to-end serving measurement.** I did not get a bare-metal `emmy serve` A/B of the c=64 no-MTP cell
  before/after the golden change. The honest expectation from the per-kernel numbers is small: the serving-form
  gain is ~17–19 us per layer at m64, ~0.9 ms per decode step, against a step time of roughly 50 ms inferred from
  the published 1109 tok/s at c=64 — so **1–2%**, not the 22% the cell is behind. Anyone continuing this should
  measure the step time directly rather than trusting that inference.
- **That gap therefore has to live somewhere else.** The arithmetic is worth stating because it is a strong
  constraint on where to look next: summing the measured m64 kernels gives roughly 350 us per layer, ~17 ms per
  step for all 48 layers, against an inferred ~50 ms step. Attention, sampling and scheduling are vLLM's and
  identical between the two engines, so a 12.7 ms per-step deficit against stock cannot be explained by kernels
  that are already at 0.92–1.07x cuBLAS. The next investigation at this cell should be the runner's per-step
  overhead (the `upload_prefix_device` copies, the per-program launch path), not the kernels.
- **The m256 fused form is unmeasured.** The isolated m256 edge improves 1.63x and `norm_gate_up.m256` is recorded
  at 1.55–1.67x cuBLAS, so a large win probably exists there — but the sweep of the fused m256 form did not finish,
  so no record was changed at that width.
- **Quality.** No GSM8K run. The change is a knob swap on three matmul shapes, numerics-preserving in the sense
  that the accumulate dtype is unchanged (`f16_f32` in, `f16_f32` out) and every recorded row passed the
  wrong-answer integrity check against torch. It does not touch the fast-math lane.

## Reproducing

Per-kernel A/B, one shape at a time (this is the whole method — no tuner is involved):

```bash
export CUDA_HOME=<the venv's nvidia/cu13>          # nvcc ships in the wheel on these boxes
emmy run --golden gemma4_12b.gate_up_cat.m64.lin --bench --bench-backends eager \
    --iters 200 --warmup 20 --no-record-nodes --json out.json \
    --ab 'TILE=a:mma_m16n8k16_f16_f32/w1x8/f4x1/k8,STAGE=d2/tma/ring' \
    --ab 'TILE=a:mma_m16n8k16_f16_f32/w4x2/f2x4/k2,STAGE=d2/tma/ring'
```

The `--json` file carries the per-row integrity flags (realized-vs-pinned knobs, wrong-answer, intensity floor);
a row with a non-empty `flags` list is not a measurement. The in-model drift audit that surfaces dead goldens is
`emmy eval golden --in-model --model google/gemma-4-12B` (needs `HF_HOME` and a token — it fetches `config.json`
for the base checkpoint, which is not the `-it` snapshot the boxes cache).

All numbers: RTX 5090, box A (30 cores) and box C (15 cores), bare-metal, `-O3` (the default for `run`), single
GPU idle, std lane only — the published emmy rows do not set `EMMY_FAST_MATH`, so the `f16_f16` fast-math siblings
are not deployable there and were excluded from every comparison.
