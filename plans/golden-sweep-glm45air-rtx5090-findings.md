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

> **Part 2 (§8-14) is a later round (2026-08-09) that went after TTFT** and revises two conclusions of this one: the
> boot audit's prefill flag is 0.15 % of a prefill step (not a lever), and prefill is GPU-bound rather than
> dispatch-bound, with 85 % of its kernel time in the routed-expert programs. Read §8 before acting on §4 or §7.

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
- The model was served from the local snapshot directory, not the repo id, because `quantized_checkpoint_dir` read
  `config.json` from the repo's DEFAULT branch, which on a branch-per-rung repo has none, so the runner fell through
  to `AutoModelForCausalLM.from_pretrained` and died with "Unrecognized model … should have a `model_type` key" —
  `--revision` reached vLLM but not the runner's own checkpoint resolution. **FIXED 2026-08-08**: the plugin hands the
  runner `<repo>@<revision>` and every resolver honors it, so `emmy serve --generate turboderp/GLM-4.5-Air-exl3
  --revision <sha>` boots the pinned rung directly.

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

# Part 2 — the prefill/TTFT round (2026-08-09)

Same card, same checkpoint, same pinned serving config. The brief: attack TTFT, starting from the boot audit's one
remaining flag (`L0.post.chunk.m512` at 46× its weight floor) and the open question of whether prefill is
host-dispatch-bound the way the decode step turned out to be. Measurement wall ~1 h; no cold runner build was needed
(the 2026-08-09 pack was reused and only the four expert `m256` programs were recompiled — see the workflow note).

## 8. What a 512-token prefill chunk actually spends its time on

One `T = 512` chunk through all 46 layers, at the runner level (`EmmyGenRunner` at `decode_bucket=32`,
`prefill_bucket=512`, `capacity=512`; fake attention output, real embedding). Wall **1791.9 ms**, five runs, spread
0.3 %.

GPU-side, from nsys with `--cuda-graph-trace=node`:

| | ms | % of GPU time |
| --- | --- | --- |
| routed experts, `m256` tier (1 141 launches) | 635.9 | 42.0 |
| routed experts, `bucket` M=32 tier (3 981 launches) | 596.7 | 39.4 |
| **trunk chunk twins** (46 `pre.prefill` + 46 `post.prefill`, M=512) | **121.7** | **8.0** |
| torch router / combine / glue (51 678 launches) | 98.2 | 6.5 |
| routed experts, symbolic tier (120 launches) | 50.1 | 3.3 |
| routed experts, M=1 tier (297 launches) | 10.4 | 0.7 |
| **total GPU kernel time** (120 034 kernel instances) | **1513.1** | 100 |

The step span under nsys is 2033.7 ms, so **GPU utilization is 74.4 %** instrumented and ≈ 84 % against the clean
1791.9 ms wall. **Prefill is kernel-bound, not dispatch-bound** — the opposite of the decode step, and the reason the
warp-tier fill port moved TTFT 1.15× while leaving TPOT at zero. Coded (trellis) matmuls are 1185.8 ms, **78.4 % of
all GPU time**; the routed experts together are **85.5 %**.

Two things this table settles.

**The 46× flag is a red herring for TTFT.** `L0.post.chunk.m512` measures 2740.7 µs against a 59.6 µs floor — it is
layer 0's post half, the model's only DENSE layer (`first_k_dense_replace=1`), carrying the uncoded fp16 `o_proj` plus
the dense MLP's three 2-bit projections. It is **0.15 % of the step**. The audit reports it not because it is
expensive but because it is the only program in the model whose weight floor clears `MIN_FLOOR_US`, so it is the only
one eligible to be flagged at all. Every other chunk twin is 1.3 MB–29 MB of weights and invisible to the audit.

**A prefill golden on the trunk is not worth its risk, and now there is a number for that.** All 92 chunk twins
together are 8 % of the step's GPU time. A 1.5× win on every one of them would be ~2 % of TTFT — against the
measured 1.36× *regression* the 2026-08-08 round got when it tried (§4). The prefill kernel time is in the routed
experts, and that is where this round went instead.

## 9. The routed-expert prefill tier — 42 % of the step, and it had never been swept

`_launch_expert` routes a hit expert's rows through `moe.expert.*.m256` when the row count lands in
(`decode_bucket`, 256]. At a 512-token chunk with `top_k=8` over `E=128` the mean per-expert row count is exactly 32,
so the distribution straddles the bucket boundary: 3 981 launches take the M=32 twin, **1 141 take the M=256 twin**,
297 take M=1 and 120 spill to the symbolic program. Every MoE layer launches ~123 of its 128 experts, i.e. **5 539
expert program launches per chunk** — each expert exactly once per layer, so there is no duplicated weight read to
reclaim.

The M=32 tier is golden-covered (the `shared_gate_up` / `shared_down` entries key it — §1). **The M=256 tier was
not**, and it was invisible: `twins.py` builds no expert twin, so `eval golden --in-model` cannot audit it, and the
file's own residual list had M=256 filed under "widths this deployment does not run". It does run it — on 42 % of the
step's GPU time.

Swept at M=256 over a 13-row ladder walking both of the CTA's output extents:

| shape | N | K | bits | golden µs | reps | cold pick µs | ratio | golden knobs | cold pick |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shared_gate_up | 1408 | 4096 | 2 | 45.87 | 45.75/46.01 (0.56 %) | 70.27 | **1.53×** | `w2x2/f2x1/k8` | `w4x1/f1x8/k8` |
| shared_gate_up | 1408 | 4096 | 3 | 47.24 | 47.24/47.25 (0.02 %) | 72.73 | **1.54×** | `w2x2/f2x1/k8` | `w4x1/f1x8/k8` |
| shared_gate_up | 1408 | 4096 | 4 | 46.03 | 45.84/46.28 (0.96 %) | 71.15 | **1.55×** | `w2x2/f2x1/k8` | `w4x1/f1x8/k8` |
| shared_down | 4096 | 1408 | 2 | — | — | 34.62 | 1.00× | *not recorded* | `w8x1/f1x8/k8` |

Same diagnosis as the decode band, one width up, and — checked rather than assumed — the same *correction*. The CTA
extents are `WORK=wAxB` warps by `fCxD` fragment cells over the m16n8k16 atom, so M extent = A·C·16 and
N extent = B·D·8:

| | knobs | M extent | M blocks | N extent | N tiles | CTAs |
| --- | --- | --- | --- | --- | --- | --- |
| `gate_up.m256` cold pick | `w4x1/f1x8` | 64 | 4 | 64 | 22 | **88** |
| `gate_up.m256` recorded | `w2x2/f2x1` | 64 | 4 | 16 | 88 | **352** |
| `gate_up.m256` tied row, withheld | `w2x2/f1x2` | 32 | **8** | 32 | 44 | 352 |

Both grids were confirmed against the observed launch grids in the stored plan (88 for the cold pick, and 88/128 for
the M=32 gate_up / down twins, which the same arithmetic reproduces). So the cold pick is **not** short of CTAs
because of the M axis — it spends its parallelism budget on a 64-wide N fragment, exactly the wide-fragment mistake §1
diagnosed at decode width. The recorded row narrows N back to 16 at an **unchanged four M blocks**: same codes
traffic, four times the CTAs, on a 170-SM card. Split-K is refused outright on a computed B, so the output axis is
the only source of CTAs either way.

That matters for whether the entry is shippable at all. The file's rule is that a kept entry must read the codes slab
no more often than the pick it replaces, and this one reads it **exactly** as often — so the M=512 inversion
mechanism does not apply, and no L2-residency argument is needed. The row that TIES it in the isolated bench,
`w2x2/f1x2` (45.86 against 45.87), reaches the same 352 CTAs by halving the M extent instead — eight M blocks, twice
the codes reads — and is therefore the one row in the ladder that carries the M=512 hazard. It is not recorded. That
was luck rather than judgement when the entry was written (b3/b4 simply preferred `f2x1` outright); the arithmetic
above is what turned it into a reason, and it is the check to run on any future wide-M entry **before** measuring
in-model rather than after.

`shared_down` at M=256 is left unrecorded on purpose: greedy is already within 0.4 % of the best pinned row, that
0.4 % is inside the rep spread, and the best row is one greedy does not reach on its own — recording it would risk
the cold-pick drop for no measured gain.

## 10. In-model verification — paired, with only the four expert programs rebuilt

The pack's validity key is model × GPU × serving shape, so a golden edit does **not** invalidate it and a stale pack
silently re-serves the old kernels. Exploited deliberately here to get a clean paired A/B: the "after" pack is a
byte-copy of the "before" pack with the four `*.m256` entries removed from `manifest.json`, so `load_pack` returns
384 of 388 plans and exactly those four programs recompile against the new goldens. Everything else in the runner is
the identical stored plan.

| | before | after | delta |
| --- | --- | --- | --- |
| `moe.expert.m256` (g0) per launch | 349.7 µs | 319.9 µs | −29.8 |
| `moe.expert.g1.m256` per launch | 347.6 µs | 317.3 µs | −30.3 |
| `moe.expert.g2.m256` per launch | 345.9 µs | 317.0 µs | −28.9 |
| `moe.expert.g3.m256` per launch | 345.6 µs | 316.8 µs | −28.8 |
| `moe.expert.*.bucket` / `.one` (controls, unchanged) | 129.1–129.3 / 34.9–36.9 µs | 129.1–129.3 / 34.9–36.9 µs | 0.0 |
| **T=512 prefill step wall** | **1791.9 ms** | **1767.9 ms** | **−24.0 (1.014×)** |

The direction is right and the controls did not move, which is what makes the 29 µs attributable. It is also **less
than the isolated bench promised**: 1.53× on the two gate/up matmuls of a 346 µs program predicts −49 µs and the
program delivers −29, so about 40 % of the isolated gain does not survive being put in a program. Not root-caused,
and the codes-traffic mechanism of §9 is ruled out (the recorded tile reads the slab exactly as often as the pick it
replaces). The plausible remainder is context — in the program these two matmuls sit inside the activation-side basis
chain, reading different buffers, with L2 in a different state — but that was not separated here. The operational
lesson is the safe one either way: **read an isolated coded-matmul ratio as a rank, never as a budget**, and take the
size of a win from the program.

**The greedy gate passes**: after the edit, greedy lands on `w2x2/f2x1/k8` unaided at all three rates (45.36 / 47.07 /
45.96 µs), and `shared_down.b2.m256` correctly stays on its own `w8x1/f1x8`. No shape was dropped to a cold pick.

## 11. In-situ versus isolated — a factor of 2–3, and it is not noise

Worth recording because every µs in this file is an isolated number. The same kernel, at the same recorded tile:

| kernel | isolated (`run --bench`, L2-resident) | in-situ (nsys, inside the real step) | ratio |
| --- | --- | --- | --- |
| `shared_gate_up.b2.m32` (expert `bucket` gate/up) | 20.16 µs | 58.7 µs | 2.9× |
| `shared_gate_up.b2.m256` cold pick (expert `m256` gate/up) | 70.27 µs | 132.3 µs | 1.9× |

The isolated bench replays one kernel ~100× over a slab that fits in L2, on an otherwise idle card. In the step the
same kernel runs inside a 46-layer sweep of a 29 GB trunk, with L2 thrashed and clocks under a sustained ALU load.
Neither number is wrong; they answer different questions. The file's existing L2 warning covers cross-shape and
cross-format comparisons — this adds that **absolute** in-model cost is 2–3× the recorded µs, so the recorded value
must not be used to budget a step either.

## 12. Where the prefill headroom actually is

After this round the step is still ~1.77 s for 512 tokens, and the shape of what is left is clear and is not tuning:

**The routed-expert matmuls are CTA-starved, and that is structural.** At M=32 the expert gate/up kernel runs at
**88 CTAs on 170 SMs** with 4 warps each, and takes 58.7 µs in-situ to stream 1.44 MB — **~59× its DRAM floor**. The
M=256 tier now reaches 352 CTAs, which is why it is 3× more efficient per row (3.8 µs/row against 11.5). The M=32
tier carries 24 % of the rows for 39 % of the GPU time for exactly this reason, and no golden can fix it: at 32 rows
the shape offers one M block, N=1408 offers 88 tiles, and split-K is refused on a computed B. **1.23 s of the 1.51 s
GPU budget is expert matmuls running at 40–60× their weight floor because the launches are too small to fill the
card.**

This promotes option (d) of `plans/moe-m2-dispatch-design.md` — the sorted grouped pass, one kernel per layer per
projection — from "deferred, promote if measurements show prefill is the binding gap" to measured-and-binding. That
document already calls it "THE prefill answer"; the numbers above are the go/no-go datum it asked for, and they are
unambiguous: prefill GPU time is 85 % routed-expert matmuls, and their inefficiency is per-launch CTA starvation,
which is exactly what a grouped pass removes. Note the argument that prefill FFN is *launch*-bound ~3× (§4 of that
doc, modelled on OLMoE) does **not** hold on this model — measured 74–84 % GPU-busy — so (d)'s value here is the
occupancy, not the dispatch saving.

Two smaller items, both measured and both real:

- **The activation-side basis chain is ~7 % of the step's GPU time** (the `k_linear_reduce_*` Hadamard/suh/svh
  kernels around the coded contraction: ~102 ms of the `m256` tier's 636 ms alone). The 2026-08-08 index-map fix made
  the flat↔128-block reshapes lowerable as index maps, which should let that chain collapse; unexploited.
- **The torch router/combine chain issues 51 678 of the step's 120 034 kernel launches** for 98.2 ms (6.5 %) of GPU
  work — `torch.where` + gather + `index_add_` per hit expert, ~1.9 µs a kernel. It is not the wall at prefill, but
  it is 43 % of the wall's *host* side and the whole wall at decode.

## 13. Serving A/B and the boot audit

Paired same-session boots (the operational rule from the decode round: host framing tracks machine load, so numbers
from different sessions are not comparable), pack hit both arms, identical config —
`VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE` 256 MiB, util 0.9641, `--max-model-len` 4096,
`--max-num-batched-tokens` 512, `--max-num-seqs` 32, decode bucket 32, prefill capacity 512,
`--kv-cache-dtype fp8_e4m3`, `--enforce-eager`, snapshot path (not `--revision`).

| arm (both pack-hit, same session, idle box) | TTFT median, 3 reps | median | TPOT median | out tok/s |
| --- | --- | --- | --- | --- |
| **A** — pack as shipped (`m256` at the cold pick) | 1665.5 / 1677.5 / 1708.7 | **1677.5 ms** | 31.00 / 31.01 / 31.01 | 22.84 / 22.78 / 22.68 |
| **B** — same pack, the four `*.m256` programs recompiled on the new goldens | 1641.5 / 1644.6 / 1688.1 | **1644.6 ms** | 30.99 / 31.01 / 31.01 | 22.95 / 22.89 / 22.76 |
| delta | every rep paired-lower, 3/3 | **1.020x (-32.9 ms)** | unchanged | +0.5 % |

Both arms reported the same 9 184-token KV pool, so the serving shape is identical. TPOT not moving is the expected
result and a useful control: the `m256` tier is prefill-only, so a change confined to it must leave the decode step
alone, and it does. The rep ranges overlap (A spans 43 ms, B spans 47 ms — machine-side jitter dominates a 33 ms
effect), so the resolving evidence is the paired ordering plus the runner-level step measurement in §10, not the
serving medians on their own. A 2 % TTFT change is near the limit of what this workload can resolve; anything smaller
should be measured at the runner level instead.

**Boot roofline audit, after**: unchanged, one flag, `L0.post.chunk.m512` at 46-47x its ~66 µs
floor (47x on the A boot, 46x on the B boot — the audit's own 3-rep timing jitter). Expected: this round touched no
trunk program. Nothing new appeared, and nothing that was flagged cleared. See §8 for why that flag is not the
prefill problem and `emmy/serving/ARCHITECTURE.md` for the durable version of the caveat.

## 14. Workflow notes from this round

- **`nsys` without `--cuda-graph-trace=node` silently hides every captured program.** The first profile of this step
  reported 754 ms of GPU work at 37.4 % utilization and led to the wrong conclusion (dispatch-bound) for twenty
  minutes. The trunk chunk twins and the `m256` expert tier both go through
  `capture_program_graph` / `replay_program_graph`, so their kernels are collapsed into graph-launch nodes and
  vanish from `cuda_gpu_kern_sum`; the `bucket` / `one` / `sym` tiers use `run_once` and stayed visible, which made
  the truncated trace look plausible rather than empty. It was caught by mapping every kernel name in the trace back
  to the pack's per-program kernel lists and noticing that no `pre.prefill` / `post.prefill` / `*.m256` program was
  represented at all. **Always pass `--cuda-graph-trace=node` when profiling emmy serving, and always check the
  kernel↔program mapping before believing a utilization number.**
- **Dropping entries from a pack's `manifest.json` is the cheap way to re-measure one program.** `load_pack` is
  all-or-nothing on a *missing file* but ignores a program simply absent from the manifest index, so deleting an
  entry recompiles exactly that program while every other plan is served from the pack. That turned this round's
  in-model A/B from two 18-minute cold builds into two 90-second ones, and it is what made a paired same-session
  comparison affordable at all.
- **The roofline audit's blindness now has a second face.** §7 noted it audits one layer and only above a 20 µs
  floor. It also builds no expert twin — and on this model the expert programs are 85 % of prefill GPU time and
  ~half of decode. The audit's one prefill flag is 0.15 % of the prefill step. *Improvement*: give the audit the
  expert tiers it already has handles for (`runner._expert_tiers`) and rank by measured cost rather than by
  ratio-over-floor, so the biggest item is reported rather than the one with the biggest floor.

---

# Part 3 — the head-to-head baseline on the 2.25 rung (2026-08-08, one session)

Phase 6 requires one comparison table both engines can be held to, and until this round none existed. Every
exllamav3 number in the campaign came off the **2.00** rung during Phase 0; the serving target moved to **2.25**
when 2.00 failed the quality gate. Part 1 §13 additionally established that host framing tracks machine load, so
numbers from different sessions do not compare — the 57.90 ms emmy TPOT recorded in Part 1 re-measured at 31.04 ms
from the identical pack on an idle box. **Everything below was therefore measured back to back in one session on
one idle card**, each server torn down and the card confirmed drained to its 185 MiB desktop baseline before the
next booted.

The short version: **emmy loses this comparison at every concurrency, by 2.8x at batch 1 and 4.4x at c = 16 on
output throughput**, and the deficit is not admission capacity — emmy serves a *larger* KV pool in tokens than the
exllamav3 lane it loses to. What emmy does match is the weights: teacher-forced perplexity agrees with
exllamav3's to 0.48 % on the identical checkpoint, which is the Phase 6 correctness bar and it passes.

## 15. What was measured, and with what

| | emmy | exllamav3 / tabbyAPI |
| --- | --- | --- |
| weights | the pinned 2.25 rung, decoded in-kernel | the identical snapshot, same bytes |
| engine | vLLM 0.23.0 + the emmy generative plugin | exllamav3 `1.4.0+cu128.torch2.10.0`, torch `2.10.0+cu128` |
| server | `emmy serve --generate`, snapshot path | tabbyAPI git `d844f705` |
| client | `vllm bench serve` 0.23.0 through `scripts/bench_serve_sweep.py` | the same client, driver and tokenizer snapshot |

Versions were re-recorded this session and had not drifted from Phase 0. The tabby clone carries **one** local
patch, the `logprobs: null` guard `scripts/patch_tabbyapi.py` applies; Phase 0's second patch (`sep="\n"` on the two
`EventSourceResponse` call sites) was **reverted before measuring** — the client-side CRLF/keepalive hardening in
`bench_serve_sweep.py` subsumes it, and a source patch to a contender's transport is a thing reviewers are right to
distrust. `sse_ping_interval: 0` stays in tabby's config, which is a supported option.

Protocol: the Phase 0 grid — random dataset, 512 in / 128 out, `--ignore-eos`, N = 8/24/48/64 at c = 1/4/8/16, one
discarded warmup, **two recorded reps**, 1 Hz `nvidia-smi` polling. The driver's completed-count guard earned its
keep twice this session (§16). The long-input point (2048 in) was not run.

**The emmy pack was rebuilt, not reused.** The pack on disk predated two commits that change what the compiler
emits — `8c43c3c9` (the run-fused column decode at the warp tier) and `eff405ac` (the routed-expert `m256` prefill
golden) — and a pack hit ignores compiler source and golden changes, so reusing it would have silently re-served
stale kernels. Cold rebuild: **1641 s, 416 programs compiled, 388 plans written**; the fresh pack has 388 plans
where the stale one had 296, which is the change made visible.

## 16. The exllamav3 cache ceiling on 2.25 — a boot is not a fit

The plan expected the 2.25 rung (~29.8 GiB) might leave no room for an fp16 cache at all. **It does leave room, and
more than Phase 0's probe suggested** — but the number that matters is not the one the boot reports.

| cache mode | boots? | survives a 512-token prefill at c = 16? |
| --- | --- | --- |
| FP16 8 192 | yes, 31 313 MiB | **yes — the served config** |
| FP16 10 240 | yes, 31 665 MiB | **no** — `torch.OutOfMemoryError` on the first batch |
| FP16 11 264 | yes, 31 857 MiB | **no** — OOM on the first 512-token prefill |
| FP16 12 288 | no — `Insufficient VRAM in split for model and cache` | — |
| Q4 16 384 | yes, 30 705 MiB | **yes — the served config** |
| Q4 32 768 | yes, 31 537 MiB | **no** — OOM under the c = 16 workload |
| Q4 36 864 | yes, 31 729 MiB | not attempted (32 768 already failed) |
| Q4 40 960 | no — `Insufficient VRAM in split` | — |

Phase 0's probe reported 4 096 fp16 / 8 192 Q4 for this rung; both are pessimistic, because that probe went through
the raw exllamav3 API with its own generator overhead rather than through tabby's loader. The honest ceiling is
neither the probe's nor the boot's: **the server loads happily at cache sizes it cannot then serve from**, and the
failure is a mid-request `torch.OutOfMemoryError` that kills every in-flight job, not a refused boot. A health check
proves nothing here; validate with the real workload. The Q4 usable edge was not bisected — 16 384 passed, 32 768
failed, and 16 384 was pinned because Phase 0 used it.

**KV arithmetic, and what it says about the model.** Across the fp16 boot ladder the marginal cost measures
**182.9 KiB/token**, which is 46 layers of K+V at 8 heads x 128 dims in fp16 (184.0 KiB/token) and not 47
(188.0) — exllamav3 does not instantiate the checkpoint's MTP layer, consistent with the Phase 2 note that the index
carries one `Glm4MoeForCausalLM` does not build. Extrapolating the ladder to a zero-sized cache puts weights + CUDA
context + workspace at **29 661 MiB (28.96 GiB)** of the card's 32 607.

## 17. The comparison table

512 in / 128 out. Mean of two recorded reps, ± the spread between them. `exl3-fp16` is the contender at its own
reference cache precision; `exl3-q4` is the same engine with its quantized cache, the nearest analogue of emmy's
fp8 KV.

| c | lane | out tok/s | req/s | mean TTFT ms | mean TPOT ms | p99 TPOT ms | power mean/max W | peak VRAM MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | **emmy** | 22.94±0.07 | 0.179±0.001 | 1643±14 | 31.00±0.02 | 31.0 | 220/234 | 31 979 |
| 1 | exl3-fp16 | **63.61**±0.77 | 0.508±0.016 | **488**±11 | **11.92**±0.18 | 12.2 | 294/335 | 31 787 |
| 1 | exl3-q4 | 62.90±1.63 | 0.530±0.053 | 491±4 | 11.15±1.49 | 12.1 | 295/352 | 31 309 |
| 4 | **emmy** | 16.16±0.14 | 0.126±0.001 | 4593±29 | 213.16±1.88 | 223.8 | 173/213 | 31 979 |
| 4 | exl3-fp16 | 98.41±0.82 | 0.787±0.005 | **1487**±23 | 27.86±0.15 | 32.8 | 317/430 | 31 813 |
| 4 | exl3-q4 | **106.25**±1.53 | 0.864±0.002 | 1561±153 | **24.29**±1.55 | 32.9 | 345/469 | 31 309 |
| 8 | **emmy** | 19.14±0.41 | 0.150±0.003 | 6374±44 | 370.58±8.63 | 401.6 | 171/217 | 31 979 |
| 8 | exl3-fp16 | 99.72±1.39 | 0.789±0.002 | 6023±58 | **29.13**±0.50 | 30.8 | 336/417 | 31 813 |
| 8 | exl3-q4 | **119.89**±3.27 | 0.974±0.012 | **2629**±3 | 43.56±0.51 | 55.6 | 364/505 | 31 309 |
| 16 | **emmy** | 22.48±1.14 | 0.176±0.009 | 11 172±617 | 594.78±32.26 | 875.9 | 195/306 | 31 980 |
| 16 | exl3-fp16 | 98.84±1.94 | 0.800±0.002 | 14 432±5 | **30.93**±1.20 | 65.1 | 340/424 | 31 813 |
| 16 | exl3-q4 | **121.12**±0.34 | 0.966±0.004 | **10 012**±114 | 41.70±0.74 | 52.6 | 365/504 | 31 309 |

Every point completed every request on every rep (the driver refuses to report otherwise).

**Reading it straight.**

- **Batch 1**: emmy TTFT 1643 ms against 488, TPOT 31.00 ms against 11.92. **3.4x behind on TTFT, 2.6x on TPOT,
  2.8x on throughput.**
- **Under load**: exllamav3 saturates near 99 tok/s (fp16) and 120 tok/s (q4) from c = 4 and stays there — the same
  flattening Phase 0 saw at 2.00. emmy does not flatten so much as **fail to rise**: 22.94 → 16.16 → 19.14 → 22.48
  as c goes 1 → 4 → 8 → 16, i.e. c = 16 throughput is *equal to* batch 1, with a dip in the middle. Both engines
  flatten; exllamav3 flattens at 4-5x emmy's level.
- **The one place emmy is ahead**: mean TTFT at c = 16, 11.2 s against the fp16 lane's 14.4 s. That is real, and it
  is the continuous-batching admission behaviour the plan predicted — but it is worth little while the same run
  takes 4.6x longer end to end, and the q4 lane (10.0 s) beats emmy on it anyway.
- **Power corroborates the mechanism**: emmy draws 171-220 W mean where exllamav3 draws 294-365 W. On a card this
  size that is the signature of a machine waiting on launches, not one doing arithmetic.
- **Phase 0's exllamav3 numbers transfer to 2.25 essentially unchanged** (c1 TTFT 488 vs 499 ms, TPOT 11.92 vs
  11.91, c16 98.8 vs 98.9 tok/s). That is now measured rather than assumed, and it is what the campaign needed: the
  extra 0.25 bpw costs the contender nothing at the same cache size.

## 18. The KV pools, which are part of the result and do not excuse it

| lane | pool | dtype | bytes/token | GiB |
| --- | --- | --- | --- | --- |
| emmy | **9 184 tokens** | fp8_e4m3 | 92.5 KiB | 0.81 |
| exl3-fp16 | 8 192 tokens | fp16 | 182.9 KiB | 1.47 |
| exl3-q4 | 16 384 tokens | Q4 | — | — |

The asymmetry runs **in emmy's favour against the lane it loses to**: emmy admits 12 % more tokens than the fp16
lane while spending 55 % of the bytes, because fp8 KV is exactly half the width. At 640 tokens per request
(512 + 128) that is 14.35 concurrent requests for emmy against 12.8 for exl3-fp16. So the throughput gap is not an
admission-capacity artifact and cannot be argued away as one. Only the q4 lane has a genuine capacity advantage
(16 384 tokens, 1.8x emmy's), and it is the lane that also happens to be fastest — worth stating plainly rather
than quoting the fp16 lane when it flatters.

**Where emmy's step time actually goes.** TPOT scales close to linearly with concurrency — 31 → 213 → 371 → 595 ms
at c = 1/4/8/16 — which is the diagnostic. Per concurrent row that is 31.0 / 53.3 / 46.3 / 37.2 ms: batching
amortizes weakly, and **no batched point beats batch 1 per row**. If the static decode bucket (32) were the
dominant cost, TPOT would be roughly *flat* across c = 4/8/16, since every step would pay for 32 rows regardless of
fill; it is not flat, it nearly doubles from c = 4 to c = 8. So the cost tracks the rows actually present, not the
bucket, and the mechanism is the routed-expert dispatch: one program per (layer, expert) hit, so more concurrent
rows touch more distinct experts and the launch count grows with the batch. This is the same wall §12 named for
prefill, and the same fix answers it — option (d) of `plans/moe-m2-dispatch-design.md`, the sorted grouped pass,
which is **unbuilt**. Separating the bucket term from the dispatch term properly needs a rebuild at
`EMMY_GEN_DECODE_BUCKET` 8 or 16, which is a pack-key change and therefore a fresh ~27-minute cold build; not done
here.

## 19. Quality on the served rung — emmy matches exllamav3 to 0.48 %

The plan requires our PPL/KL to match exllamav3's on the same checkpoint, since decode is exact reconstruction and
any gap is a bug. Measured teacher-forced, on **identical token ids** so no tokenizer difference can enter: 16 rows
of 256 tokens from wikitext-2-raw test, fed to emmy as an explicit token-id prompt with `prompt_logprobs`, and to
exllamav3 through the same `model.forward(..., {"attn_mode": "flash_attn_nc"})` call `eval/ppl.py` uses.

| arm | PPL over 16 x 255 scored tokens |
| --- | --- |
| exllamav3 1.4.0, no KV quantization | **7.3581** |
| emmy, served, fp8_e4m3 KV | **7.3934** (+0.0353, **+0.48 %**) |

emmy is higher on 10 of 16 rows — a small consistent bias, the size one expects from an fp8 paged cache against an
unquantized forward, and far too small to be a decode defect. **The correctness bar passes.** These are 256-token
contexts, so the absolute value is not comparable to Phase 0's 6.306 (measured at 2048); the paired comparison is
what the gate asks for and it is clean.

**Two things this cost, both worth recording.** First, the served configuration **cannot** produce prompt logprobs:
it has 22 MiB free, and logits for even one 512-token chunk over vocab 151 552 is ~310 MB, so the request OOMs and
takes EngineCore down with it. Lowering utilization does not help in the obvious way either — at 0.93 there is no
KV pool at all, because the emmy runner claims its residents before vLLM's profiler runs. The quality arm was
measured at utilization 0.95 (a 4 144-token pool), same pack, same kernels, same fp8 KV dtype; it is not a serving
measurement and is labelled as such in the manifest.

Second, **greedy agreement is not a usable quality metric here**, and the control is what shows it. Across 16
prompts at temperature 0:

| pair | exact-match sequences | median first divergence |
| --- | --- | --- |
| emmy vs exl3-fp16 | 0/16 | token 4.5 |
| emmy vs exl3-q4 | 0/16 | token 6.5 |
| **control: exl3-fp16 vs exl3-q4** (same engine, cache precision only) | **2/16** | token 8 |

Changing nothing but the KV cache precision *within exllamav3* already destroys agreement, so a cross-engine greedy
comparison measures floating-point path differences rather than model quality. Do not report it as a quality
number; the PPL pairing above is the datum.

## 20. Handicaps, named against the numbers they touch

None of these are excuses — the table above is the result — but a reader comparing engines should know which emmy
numbers carry a known, unremoved cost.

- **`--enforce-eager`** applies to **every** emmy row. Whole-step decode capture does not fit in the memory budget,
  so the decode step pays per-launch host overhead that the captured path would remove. On a step already suspected
  of being launch-bound this is the handicap most likely to matter, and it is not separable from the numbers here.
- **The unbuilt sorted grouped MoE dispatch pass** applies to the c = 4/8/16 rows, and §18 argues it is the
  dominant term in them.
- **The withheld prefill trunk goldens** (Part 1 §4) apply to every TTFT figure.
- **No prebuilt serving image** — Phase 5 is incomplete, so the emmy lane boots from source with a locally built
  pack rather than the released artifact the recipe names.
- **The M1 tier was ON** (`EMMY_GEN_M1_TIER` unset, default 1) and is what makes batch 1 the best per-row point;
  it routes only T = 1, so it contributes nothing at c > 1.
- **`--ignore-eos` asymmetry, favouring exllamav3 slightly**: tabby's backend does not honour the flag and produced
  93-99 % of the 128-token cap per request, so it did 1-7 % less decode work per request than emmy, which honours
  it exactly. Throughput and TPOT are per-token and unaffected; `req/s` and duration slightly favour exllamav3.

**One operational note.** The emmy sweep is split across two boots: the harness reaped the server process during
the c = 16 point, so c = 1/4/8 come from the cold-build boot and c = 16 from a pack-hit reboot minutes later. A
c = 1 control was re-run on the second boot and reproduces the first to 0.1 % (22.92 vs 22.94 tok/s, TPOT 31.03 vs
31.00, TTFT 1644 vs 1643 ms), both boots reporting the same 9 184-token pool — which is what makes the split
comparable rather than a defect. Launch long-lived servers detached (`setsid nohup`), not as a background job of
the agent harness.

**Raw manifests**: `experiments/GLM-4.5-Air-EXL3/serving_exllamav3_rtx5090/results_2026-08-08/{fp16,q4}/` and
`experiments/GLM-4.5-Air-EXL3/serving_rtx5090/results_2026-08-08/{emmy,emmy_boot2_c1_control,quality}/`, each
directory's `manifest.json` carrying the versions, the serving shape, the cache-ceiling probe and the boot facts.
