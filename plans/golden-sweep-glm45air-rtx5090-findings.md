# GLM-4.5-Air 2.25 bpw EXL3 golden sweep on the RTX 5090 — findings (2026-08-08)

VQ Phase 4. Card: NVIDIA GeForce RTX 5090 (sm_120). Checkpoint: `turboderp/GLM-4.5-Air-exl3` @
`6a309ed6d606fc0154e6e1aeb0912cd3c25534fe` (the 2.25-optimized rung). Measurement wall time ~5 h; total session
including the two 46-layer runner builds and the serving boots ~7 h.

**Method**: manual pinned `--ab` exploration, not the tuner — the repo's recorded practice, and the right one here
(the tune loop would have spent its whole budget on the ~3.8k-row warp pool per shape, and the one thing that
mattered turned out to be a single direction in that pool). Each entry is a 3-rep median from
`emmy run -c "<snippet>" --bench --bench-backends emmy --warmup 10 --iters 100 --ab "<row>" …`; every entry carries
its rep min/max in the YAML.

| category | count |
| --- | --- |
| added (new shape, greedy beaten by > 3 %) | 32 |
| added (new shape, greedy already at the optimum — recorded for cold-boot determinism) | 8 |
| replaced (existing entry, config unchanged, µs refreshed) | 2 |
| measured and **deliberately withheld** (isolated ranking inverts in-model) | 28 |
| unchanged (the synthetic past-L2 control, not re-measured) | 1 |

Headline: **batch-1 TPOT 104.2 → 57.90 ms (1.80×)** on the served model, spread 0.02 % over three runs. TTFT is
unchanged at 1.93 s, which is exactly what the withheld prefill entries predict.

## 1. The 17× roofline flag — what it actually was

The boot audit's one warning, `L0.post.decode.m32` at 17× its ~67 µs weight-streaming floor, is layer 0's
post-attention program: the uncoded fp16 `o_proj` (4096×12288, 100.7 MB — the quantizer leaves layer 0's alone) plus
the dense MLP's three 2-bit coded projections. Timed directly (`_Program.program.iter_once()`, per-launch CUDA
events) it was **1170.8 µs against a 64.7 µs floor, 18.1×**, and the per-kernel split named the cause immediately:

| launch | before | what it is |
| --- | --- | --- |
| `k_linear_trellis_decode_reduce_4a4538` | 553.2 µs | dense MLP `down`, 11008 → 4096, 2 bits |
| `k_linear_trellis_decode_reduce_e4a5e7` ×2 | 215.5 / 214.8 µs | dense MLP `gate` / `up`, 4096 → 11008, 2 bits |
| `k_linear_4022c9` | 151.7 µs | the uncoded fp16 `o_proj` |
| 10 smaller launches | 6–19 µs each | norms, the residual add, pointwise glue |

So the flag was **not** a lowering failure, a demoted tier or a hang. All three coded matmuls were on the intended
warp/mma tier with the intended `sync` compute fill; they were simply carrying the wrong OUTPUT TILE, and the same
mistake was on every coded projection in the model. On the isolated snippet, `mlp_gate_up` at M=32 measured 137.9 µs
with greedy's `w4x1` / `f1x8` / `k8` at **25 % occupancy** (113 registers, 32 KB smem, 172 CTAs); walking the warp
grid found `w2x2` / `f1x1` / `k8` at **58 %** and **63.4 µs**, a 2.17× win from nothing but a narrower fragment.

**Why the cold pick is wrong, stated generally.** A wide register fragment is the right answer for a MATERIALIZED B
— it buys operand reuse against a fixed staging cost. A DECODED B has no operand to reuse: the decode is per-element
ALU work (~29 warp-instructions per 2-bit weight on this tier), so the fragment buys nothing and spends registers,
and the register pressure is what caps occupancy. The correct answer is the narrowest tile the shape allows with as
many CTAs as the output axis will give. Split-K would be the other way to buy CTAs and it is refused outright on a
computed B ("split-K needs a materialized B on every channel — a computed B has no gmem index to σ-reindex"), so the
output axis is the only source of parallelism, which is why the miss is worst on the narrow-N shapes.

**Did it clear?** Partly, and honestly: `L0.post.decode.m32` is now **682.0 µs, 10.4× the floor** — 1.72× faster and
right at the `WARN_RATIO` threshold, so the boot still logs it. What it cannot log is the rest, which is the real
story: the audit times ONE layer per attention class and skips any program whose floor is under `MIN_FLOOR_US`
(20 µs). On this model that means layer 0 only — and layer 0 is the model's single DENSE layer, which clears the
floor at all only because of its uncoded fp16 `o_proj`. The **unreported** representative MoE layer was worse than
the flagged one:

| program | weights MB | floor µs | before | after | speedup | before/floor | after/floor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `L0.pre.decode.m32` | 29.5 | 14.2 | 590.7 | 349.2 | 1.69× | 42× | 24× |
| `L0.post.decode.m32` *(the flagged one)* | 134.6 | 64.7 | 1170.8 | 682.0 | 1.72× | **18×** | **10×** |
| `L1.pre.decode.m32` | 22.1 | 10.6 | 584.7 | 349.3 | 1.67× | 55× | 32× |
| `L1.post.decode.m32` | 25.5 | 12.2 | 936.5 | 406.3 | **2.30×** | **77×** | 33× |
| `moe.expert.g0.bucket.m32` | 6.7 | 3.2 | 460.4 | 180.2 | 2.56× | 143× | 55× |
| `moe.expert.g1.bucket.m32` | 8.9 | 4.3 | 460.7 | 184.2 | 2.50× | 108× | 42× |
| `moe.expert.g2.bucket.m32` | 4.6 | 2.2 | 466.1 | 180.9 | 2.58× | 213× | 81× |
| `moe.expert.g3.bucket.m32` | 5.3 | 2.5 | 453.2 | 192.9 | 2.35× | 179× | 75× |
| `moe.expert.g0.one.m1` | 6.7 | 3.2 | 59.2 | 60.4 | 1.00× | 18× | 18× |
| `moe.expert.g0.m256` | 6.7 | 3.2 | 480.6 | 479.2 | 1.00× | 149× | 146× |

Only the second row was ever printed at boot. `emmy/serving/ARCHITECTURE.md` now says so beside the audit's entry:
on a low-bit model a quiet roofline audit is no information.

Two things in that table are worth calling out. The four `moe.expert.*.bucket.m32` programs improved 2.4–2.6× with
**no expert golden of their own** — a routed expert's gate/up and down have the same extents and the same code rates
as the shared expert's, and the weights arriving as program inputs rather than constants does not change the
contraction, so the `shared_gate_up` / `shared_down` entries key them. That is the answer to the brief's
expert-coverage question: the expert programs are covered, by shared-expert entries, and verified by observed step
time rather than by the audit (which cannot build an expert twin). And `moe.expert.*.m256` did not move, because
M=256 was not swept — see §4.

## 2. Per-shape outcomes

Fourteen coded (shape, rate) pairs — every coded projection family in the dense trunk, at every rate the checkpoint's
mixed allocation uses. `golden µs` is the recorded 3-rep median; `f16 twin µs` is the same extents in uncompressed
f16 through torch eager, and is **L2-resident at all these sizes**, so a coded row reading slower than it says
nothing about the served model (the file's preamble carries this warning at length).

### M = 1 — the decode band (reduce tier)

| shape | N | K | bits | golden µs | spread | greedy µs | greedy/golden | golden | greedy | f16 twin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q_proj.b4 | 12288 | 4096 | 4 | 16.70 | 0.57 % | 19.17 | **1.15×** | `g32k` | `g8k` | 23.4 |
| q_proj.b3 | 12288 | 4096 | 3 | 16.79 | 0.42 % | 19.32 | **1.15×** | `g32k` | `g8k` | 23.4 |
| kv_proj.b4 | 1024 | 4096 | 4 | 3.98 | 0.23 % | 3.98 | 1.00× | `g32k` | `g32k` | 8.4 |
| kv_proj.b3 | 1024 | 4096 | 3 | 4.09 | 0.39 % | 4.09 | 1.00× | `g32k` | `g32k` | 8.4 |
| o_proj.b4 | 4096 | 12288 | 4 | 16.89 | 0.91 % | 21.85 | **1.29×** | `g96k` | `g16k` | 25.0 |
| o_proj.b3 | 4096 | 12288 | 3 | 17.44 | 0.28 % | 22.03 | **1.26×** | `g96k` | `g16k` | 25.0 |
| mlp_gate_up | 11008 | 4096 | 2 | 14.96 | 10.4 % | 14.96 | 1.00× | `g32k` | `g32k` | 21.1 |
| mlp_down | 4096 | 11008 | 2 | 15.25 | 0.58 % | 15.27 | 1.00× | `g86k` | `g86k` | 22.8 |
| shared_gate_up.b2/b3/b4 | 1408 | 4096 | 2/3/4 | 4.64 / 4.74 / 4.77 | ≤ 0.15 % | same | 1.00× | `g32k` | `g32k` | 10.4 |
| shared_down.b2/b3/b4 | 4096 | 1408 | 2/3/4 | 4.31 / 4.49 / 4.40 | ≤ 0.49 % | same | 1.00× | `g11k` | `g11k` | 8.4 |

The band is mostly right cold. The two corrections are both long-K shapes where the cold pick takes a split 4–6×
too narrow: `o_proj` (K=12288, 768 tiles) wants `g96k` = 8 tile steps per CTA and gets `g16k`; `q_proj` wants `g32k`
and gets `g8k`. The two pre-existing entries (`mlp_gate_up.m1`, `mlp_down.m1`) reproduced their recorded split
exactly and moved only in µs (15.0 → 14.96, 15.3 → 15.25), which is a clean re-measurement check on the
2026-08-07 round. `mlp_gate_up.m1`'s 10.4 % spread is one fast outlier in three (13.40 / 14.96 / 14.97); the median
is the stable value and it matches the recorded 15.0.

### M = 8 and M = 32 — the warp tier at decode width

M=32 is what the pinned serving config runs (`EMMY_GEN_DECODE_BUCKET=32`); M=8 is the documented alternative.

| shape | bits | m8 golden | m8 greedy | m8 ratio | m32 golden | m32 greedy | m32 ratio | m32 knobs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q_proj | 4 | 74.56 | 151.95 | 2.04× | 65.03 | 160.30 | **2.47×** | `w2x2/f1x1/k8` |
| q_proj | 3 | 72.77 | 152.92 | 2.10× | 63.27 | 153.37 | **2.42×** | `w2x2/f1x1/k8` |
| kv_proj | 4 | 36.73 | 124.96 | 3.40× | 26.05 | 124.36 | **4.77×** | `w2x1/f1x1/k8` |
| kv_proj | 3 | 37.29 | 126.74 | 3.40× | 25.84 | 126.38 | **4.89×** | `w2x1/f1x1/k8` |
| o_proj | 4 | 132.26 | 373.64 | 2.82× | 100.92 | 258.34 | **2.56×** | `w2x2/f1x1/k8` |
| o_proj | 3 | 133.21 | 377.81 | 2.84× | 100.68 | 261.73 | **2.60×** | `w2x2/f1x1/k8` |
| mlp_gate_up | 2 | 67.63 | 139.07 | 2.06× | 63.40 | 137.76 | **2.17×** | `w2x2/f1x1/k8` |
| mlp_down | 2 | 114.13 | 326.52 | 2.86× | 90.57 | 217.20 | **2.40×** | `w2x2/f1x1/k8` |
| shared_gate_up | 2 | 42.00 | 123.22 | 2.93× | 27.15 | 121.63 | **4.48×** | `w2x2/f1x1/k8` |
| shared_gate_up | 3 | 42.35 | 127.74 | 3.02× | 27.90 | 126.28 | **4.53×** | `w2x2/f1x1/k8` |
| shared_gate_up | 4 | 42.53 | 125.98 | 2.96× | 28.00 | 124.07 | **4.43×** | `w2x2/f1x1/k8` |
| shared_down | 2 | 16.95 | 44.09 | 2.60× | 12.80 | 28.68 | **2.24×** | `w2x2/f1x1/k8` |
| shared_down | 3 | 17.72 | 45.78 | 2.58× | 12.77 | 30.93 | **2.42×** | `w2x2/f1x1/k8` |
| shared_down | 4 | 17.54 | 45.25 | 2.58× | 12.87 | 30.54 | **2.37×** | `w2x2/f1x1/k8` |

Every spread is ≤ 0.6 % except `mlp_gate_up.m64` (1.4 %) and `q_proj.b4.m64` (1.9 %), neither of which shipped.
Twelve of fourteen M=32 winners are the identical knob row, `w2x2` / `f1x1` / `k8` / `d1/sync`; the two exceptions
are the narrowest output (`kv_proj`, N=1024), which prefers `w2x1` — half the CTA's N extent, twice the CTAs. At M=8
the M warp count is pure waste (one 16-row mma block already covers the rows) and the winner moves to `w1x4`.

The **rate never changes the winner**, only the absolute µs (3-bit and 4-bit twins of one shape land within 3 % of
each other and pick the same row). That is worth knowing for the next mixed-allocation checkpoint: the rate has to be
in the key, and it does have to be measured, but it does not need its own search.

The f16-accumulate atom (the `FAST_MATH` lane) measured within 0.1 % of the f32 one at the winning tile on
`mlp_gate_up.m32` (63.3 vs 63.3 µs), so no `[fm]` sibling entries were recorded; one schedule serves both lanes, as
it already did on the M=1 band.

## 3. Prior diagnosis — where the cold ranking puts the answer

`emmy eval offline --kernel glm45air`, over the shipped set:

| metric (offline / cold-start prior) | value |
| --- | --- |
| median rank of the recorded golden | 187 |
| top-1 / top-10 / top-25 | 0 / 35 · 7 / 35 · 14 / 35 |
| M = 1 band entries (pool 3–7 rows) | rank 2–4 |
| M = 8 warp entries (pool ~3 974 rows) | rank 20–254 |
| M = 32 warp entries (pool ~3 974 rows) | rank **143–1079** |

The split is the diagnosis. On the M=1 band the pool is a handful of split widths and the cold ranking is nearly
right (rank 2–4, and 8 of 14 shapes need no correction at all) — that catalog was built for this op and priced for
it. On the warp tier the pool is ~4 000 rows and the cold ranking puts the measured optimum at rank 143–1079: no
patience setting reaches that, which is why greedy shipped a 2–5× miss on every coded projection in the model and
why the whole set had to be seeded by hand.

The mispricing is one feature, not a family of them: the prior scores an output tile by the operand reuse it buys,
and a decoded B has no operand traffic to reuse — its cost is the decode's instruction count, which is invariant in
the fragment shape while the register pressure is not. **Recommendation (high priority):** an engineered feature
that carries the decoded-B instruction count (or simply the `dtype_class == "trellis"` × fragment-cell product) and
an `emmy fit` refit over these 42 rows; the alternative is hand-seeding every coded shape of every future checkpoint.
Six rows here rank shallow enough (≤ 25) that patience alone would find them, so the refit does not need to be
perfect to close most of the gap.

Two eval-side gaps found while gathering this: six entries (`kv_proj.m1`, `shared_gate_up.*.m1`, `shared_down.*.m1`)
report `pool 0` / rank `?` in `eval offline` — their enumeration is not reproducible in that view even though the
same shapes bench and deploy fine, so the median above is over 35 of 42 rows. Not chased.

## 4. What was deliberately left unswept, and one thing that was swept and withheld

**Withheld: the prefill width (M=512) and M=64 — 28 measured entries, not shipped.** Both swept clean in isolation
(M=512 at 1.03–1.76× over greedy, M=64 at 2.1–4.5×). The M=512 set was then checked in the real programs and it
**inverted**: layer 1's coded `o_proj` went 711 → 1182 µs and the whole `post` chunk twin 1334 → 1818 µs, a 1.36×
regression where the isolated bench promised 1.27× the other way.

The mechanism is the file's own L2 warning, biting the measurement instead of the reader. Both withheld widths' winners
cut the CTA's M extent below the cold pick's — at M=512, 64 rows against 128, so eight M blocks over the output
instead of four; at M=64, 32 against 128 — which re-reads the codes slab more times and narrows the DRAM stripe.
Replaying one kernel a hundred times over an 11–25 MB slab that fits in the 5090's 96 MB L2, that redundancy is free.
Against a 29 GB trunk it is not. The M = 1 / 8 / 32 entries do **not** have this property — every one of them keeps
the shape's single M block, verified mechanically — and their in-model gain reproduced at 1.7–2.3× per program. So
the shipped rule is: **an entry is kept only if its tile reads the codes slab no more often than the pick it
replaces, or it was verified in-model.** A prefill golden needs an in-model or past-L2 harness, which is a real
follow-up and not a small one.

Also left, with reasons:

- **The symbolic (`.dynM`) coded forks.** At `--max-num-batched-tokens 512` a 512-token prompt is one full chunk and
  takes the static twin, so the symbolic programs carry only chunk tails. They are also wide-M and would hit exactly
  the measurement problem above.
- **The warp grid beyond 9 sampled rows per width.** The pool is ~3.8k rows. The sample was built from one full
  exploration on `mlp_gate_up.m32` (30 pinned rows across WORK / TILE / STAGE / REDUCE) and the direction reproduced
  on 13 more shapes, so the direction is solid; the last few percent is not claimed.
- **The widths `twins.py` traces for other serving lanes** — M = 192 / 256 / 1024 / 2048 / 4096. This deployment runs
  none of them and they are all wide-M. This is why `moe.expert.*.m256` did not move.
- **The uncoded f16 forks of this model** — layer 0's fp16 `o_proj` (151.7 µs in-model at M=32, and 455 µs at M=512),
  the norms, the router, the fused and pointwise glue. Real cost, different lever, untouched.
- **The `pastl2_22016` synthetic control** was not re-measured; its 2026-08-07 row stands unchanged.

## 5. Verification

**In-model audit.** `emmy eval golden --in-model --model turboderp/GLM-4.5-Air-exl3@6a309ed6…`:
**MATCH 20 → 40, DRIFT 0, GAP 317 → 297, compile_fail 0.** Every remaining trellis GAP is a width this deployment
does not run (M = 192 / 256 / 1024 / 2048 / 4096) or a symbolic fork. Two structural limits bound what that number
can ever be here, both pre-existing: the coded twin pairs ONE traced layer's structure with each checkpoint layer's
rates, and GLM-4.5-Air's traced layer is layer 0 — the only dense one — so the shared-expert shapes are not in any
twin and 12 of the 42 entries are unauditable by construction; and there is still no expert twin at all.

**Deploy, the part the audit cannot show.** The two 46-layer runner builds above are the real check: greedy picks the
recorded config in the actual serving programs, and the programs got 1.7–2.6× faster. The shared-expert entries
deploying into the routed-expert programs (2.4–2.6×) is only visible this way.

**End to end, batch 1**, `scripts/bench_serve_sweep.py`, 512 in / 128 out, `--ignore-eos`, N=8, c=1, one discarded
warmup and three recorded runs (seeds 43/44/45):

| | before (2026-08-08 plan record) | after | delta |
| --- | --- | --- | --- |
| TPOT median | 104.2 ms | **57.90 ms** (57.90 / 57.90 / 57.91, spread 0.02 %) | **1.80× faster** |
| TTFT median | 1.96 s | 1.93 s (1.919 / 1.929 / 1.966) | unchanged |
| output throughput | 8.4 tok/s | **13.79 tok/s** | 1.64× |
| peak VRAM | 32 037 MiB | 31 965 MiB | — |

TTFT not moving is the expected result, not a disappointment: prefill runs the M=512 chunk twin and the symbolic
programs, and this sweep shipped nothing for either.

**Config deviations from the record**, both forced and both noted for Phase 5:

- `--max-num-seqs 32` (vLLM's default 256). At the recorded knobs the boot died in vLLM's post-KV sampler warmup
  ("CUDA out of memory … warming up sampler with 256 dummy requests"), and before that the 256-seq warmup budget
  pushed available KV down to 0.28 GiB, below what `--max-model-len 4096` needs. With 32 the pool is **9,184 tokens**
  against the record's 9,392 — within 2 %, so the serving shape is effectively the recorded one. Reproduces on a pack
  hit, so it is not a cold-compile transient; not root-caused here.
- The model is served from the local snapshot directory, not the repo id: `quantized_checkpoint_dir` reads
  `config.json` from the repo's DEFAULT branch, which on a branch-per-rung repo has none, so the runner falls through
  to `AutoModelForCausalLM.from_pretrained` and dies with "Unrecognized model … should have a `model_type` key".
  `--revision` reaches vLLM but not the runner's own checkpoint resolution. Worth fixing before the image bake, since
  the release pipeline pins `SERVE_REVISION`.

**Gates**: `make test` green (pytest summary 0 failed; the durations-gate non-zero exit is the known pre-existing
one), the golden drift gate green (13 passed), `make lint` clean, and `scripts/digest_kernels.py` byte-identical to
the checked-in `scripts/kernel_digests.txt` — the one compiler-side change in this commit is emission-neutral.

## 6. The compiler change that came with the sweep

Recording `STAGE: 'd1/sync'` on a matmul golden failed the permanence gate: the compute-fill depths lived as a
literal `[1, 2]` inside `_schedule._sync_values`, so a recorded `d1/sync` was not a member of any catalog the gate
could check. This is the same defect the decode band's split widths had before Phase 4 enablement lifted them into
`space.decode_band_moves`, and it is fixed the same way — `space.SYNC_STAGE_DEPTHS` / `space.sync_stage_moves()`,
consumed by the scheduler and by the gate (which asks the sync catalog for a `dtype: trellis` entry and the transport
grid otherwise). No behaviour change: same depths, same order, digests identical.

## 7. Remaining gap — per-kernel versus dispatch, honestly

Two quantities are measured rather than modelled, and they bracket the answer.

The trunk is 92 static programs per decode step (46 `pre` + 46 `post`). Summing the per-launch times above over the
real layer mix — one dense layer 0 plus 45 MoE layers — gives **70.2 ms before, 35.0 ms after**, a 35.2 ms saving.
Measured batch-1 TPOT fell by 46.3 ms (104.2 → 57.90). The trunk therefore accounts for about three quarters of the
win and the remaining ~11 ms came from the expert programs, which improved 2.4–2.6× on the same shared-expert
entries; the exact per-tier split at c=1 was not decomposed.

What is left is **~23 ms of a 57.90 ms step that is not trunk kernel time**, and this sweep cannot touch it:
`combine_routed_experts` issues one launch per distinct expert a step routes to — 360 at c=1 across 45 MoE layers —
and `plans/moe-m2-dispatch-design.md` measured ~117 µs of Python framing per launch, i.e. ~42 ms of framing that only
partly hides behind GPU work. So per-kernel work was ~2/3 of the step before and is roughly half of it now, and from
here the dispatch chain is the larger single item at batch 1, as it already was at concurrency.

The per-kernel side is not exhausted, but the remaining headroom is compiler work rather than tuning. The trunk decode
programs now sit at 24–33× their weight floor (down from 42–77×), and NCU on the Phase 3.2 round put that residual on
instruction issue, not bandwidth: the warp tier decodes **per element** at ~29 warp-instructions per 2-bit weight
where the M=1 band's run-fused column decode does it in 8.25. **Porting the run-fused decode to the warp tier's
compute fill is the next per-kernel lever** — on that instruction ratio it is worth roughly another 2× on every coded
projection at decode width, and no golden can reach it because the run form is not in the warp tier's vocabulary.

## Workflow notes

- **The in-model probe was the whole sweep's safety net and it is ad hoc.** Building the 46-layer runner and calling
  `_Program.program.iter_once()` per static twin is what found the 17× flag's cause in one shot, what proved the
  M = 1/8/32 entries deploy, and what caught the M=512 inversion before it shipped. It is 100 lines of scratch
  script. *Improvement*: fold it into the CLI — `emmy eval golden --in-model --measure`, or a `--roofline` flag on
  the serving boot that times every program (not one layer) and prints the per-launch split. The boot audit already
  has the machinery and throws the data away.
- **The roofline audit is nearly blind on a low-bit model** — one layer, and only above a 20 µs floor. On this model
  that is 1 of 92 trunk programs, and the one it reports is the least representative (layer 0 is the only dense
  layer). *Improvement*: audit one program per DISTINCT program shape rather than per attention class, and scale
  `MIN_FLOOR_US` by the model's bit rate — or drop the floor and rank instead of threshold.
- **`eval golden --in-model` cannot see the deployed prefill width.** `twins.py` traces 1/8/32/64/192/256/2048/4096;
  this deployment's static prefill twin is at 512 (`--max-num-batched-tokens`). Adding 512 to the list would have
  churned the gemma-4 and OLMoE drift-gate baselines for models nobody measured, so it was left alone.
  *Improvement*: make the traced width list configurable per model tag, so a model's golden file declares the widths
  its lane deploys.
- **A per-shape ladder pass is ~20 s**, so a 14-shape width sweeps in ~6 minutes — the sweep itself was never the
  bottleneck. The two 46-layer runner builds (~25 min each, cold) and the serving boots dominated the wall clock, and
  a golden edit invalidates the pack, so every verification round pays a cold build. *Improvement*: nothing obvious;
  it is the price of verifying on the real model, and it is worth paying.
- **Three serving boots were lost to configuration, not to the sweep**: the repo-id checkpoint resolution above, the
  `max_num_seqs` sampler warmup, and one utilization probe. Each costs 4–25 minutes. *Improvement*: `emmy serve`
  could refuse a branch-only checkpoint id up front with the message the release pipeline's `warm.sh` already has.
- The `--json` record from `emmy run --bench` carries `record_knobs` per row, which is what made a scripted sweep
  possible at all — no table parsing. That part of the workflow is in good shape.
