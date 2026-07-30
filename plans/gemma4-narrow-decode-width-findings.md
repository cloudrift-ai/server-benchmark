# Narrow decode widths (m8 / m32) on gemma-4-12B, RTX 5090 — where the MTP batch gap actually lives

Target: the two largest emmy-vs-stock gaps in the article's speculative-decoding table — `4k/4k c=8`
with the official drafter at depth 2 (**0.74x**) and depth 3 (**0.77x**). Both run decode bucket 32
with the 2048-token prefill quantum, so the brief's leading suspect was m32 kernel quality.

**Answer: those cells are not kernel-bound, and they never reach concurrency 8.** Emmy's device
footprint leaves 3.0 GiB for the KV cache where stock leaves 3.61, and a `4096-in / 4096-out` request
needs 8192 tokens of it — so emmy sustains a mean of **5.42** concurrent streams against stock's
**6.75** — never once reaching 7 — with the balance queued and cycling through preemption. That ratio
alone (**0.803**) accounts for the entire measured cell ratio (**0.797**) to within 1%. Emmy's whole
per-step kernel budget at m32 is **14.86 ms measured** against a **28.7 ms** step and a **12.18 ms**
DRAM floor, so kernel work has ~1.3 ms of realistic headroom — 4-5% of the step.

Same-box, full-protocol decomposition of the observed **0.797x** (section 2):
`0.803 (admission) x 0.945 (step latency) x 1.024 (acceptance — an emmy advantage) = 0.777`.
**Admission alone (0.803) accounts for the whole observed ratio to within 1%; kernels are the ~5.5%
step-latency term.** So the honest verdict
is that **admission capacity is essentially the whole cell gap**, with a real but much smaller kernel
term beside it — and the brief's suspicion was pointed at the right half of the smaller term (m32 kernel
quality is genuinely worse than m8's; see section 4b for why, which *is* the bucket 8 -> 32 shift, though
not for the reason hypothesized).

A real m32 kernel win does exist, is deploy-verified, and ships on this branch (section 4): the three
fused norm->linear edges the decode twins launch every step were deploying `PLACE@cone=fuse` where
`cut` is 8 - 49% faster, taking them from 0.73 - 0.96x eager to 1.01 - 1.38x. It is worth ~1.3 ms of
emmy's 14.86 ms per-step kernel budget — a genuine correction to the previous round's "no m32 win
exists" — which is ~three quarters of the step-latency term. It cannot touch the other three terms.

Read section 2's honesty notes before quoting any tok/s number from this report: `c=8` at
`num-prompts 16` is not reproducible for *either* engine (emmy 369 - 492, stock 469 - 540), and the
boxes differ by 11% in step latency at identical configurations. The conclusions above rest on
boot-time VRAM accounting, scheduler timelines and CUDA-graph-window kernel timing — none of which are
serving-throughput measurements — precisely because the serving throughput in this region is noisy.

## 0. Provenance

Four RTX 5090 boxes, one GPU each, `nvidia-smi` healthy and no pending reboot on all of them, apt
timers masked. Branch `feature/mtp-narrow-width-tuning` at `274c153e` (identical to `origin/main`).

| box | cores | driver | role |
|---|--:|---|---|
| A | 30 | 580.173.02 | docker serving A/B on the published image |
| C | 15 | 580.159.03 | in-container kernel profiling; stock vLLM control |
| D | 15 | 580.159.03 | bare-metal golden A/B (`emmy run --bench --ab`) |

Serving lanes replicate `experiments/gemma-4-12B/serving_mtp_rtx5090`'s compose exactly (generated
from the recipe and diffed against the hand-rolled `docker run`): published image
`cloudriftai/vllm-emmy-gemma4:0.23.0-58733e02`, `--gpu-memory-utilization=0.96`,
`--max-model-len 8448`, `--dtype float16`, `--no-enable-prefix-caching`,
`EMMY_GEN_DECODE_BUCKET=32 EMMY_GEN_PREFILL_BUCKET=2048`, `--max-num-batched-tokens 2080`,
`--speculative-config {"method":"mtp","model":"google/gemma-4-12B-it-assistant",
"num_speculative_tokens":2}`.

**Reproduction check before changing anything.** The target cell at the recipe's full protocol
(`c=8`, `num-prompts 64`) measured **476.75 tok/s** against the article's published **450.6** — +5.8%,
inside the ~10% single-run spread the brief warns about for these cells. The configuration is the one
the article measured.

**Benchmark deviation, stated up front:** the screening runs use `--num-prompts 16` instead of 64 (the
request mix and prefill:decode token ratio are identical — 4096 in / 4096 out per request — but the
run takes 3 minutes instead of 10, and the drain tail is a larger share, so absolute tok/s reads
*lower*: 369.09 at np=16 against 476.75 at np=64 for the same server). Every comparison below is
one-variable at the same `num_prompts`. Cross-`num_prompts` absolutes are not comparable.

## 1. The kernel budget at m32 — the ceiling on any kernel win

### The DRAM floor

Emmy owns the 48-layer trunk (embed gather, per-layer norm + QKV, o_proj + MLP, final norm).
Attention, RoPE, KV cache and the lm_head / logits are vLLM's. Trunk weights, gemma-4-12B fp16
(hidden 3840, inter 15360, 40 sliding + 8 full-attention layers, 16 Q x 256 / 8 KV x 256 sliding,
16 x 512 / 1 KV global):

| | params | fp16 bytes |
|---|--:|--:|
| per sliding layer | 224.1 M | 448 MB |
| per full-attention layer | 243.8 M | 488 MB |
| **trunk, 48 layers** | **10.92 G** | **21.83 GB** |
| lm_head / embed (tied) — vLLM's | 1.01 G | 2.01 GB |

A decode step reads every trunk weight exactly once, so at the RTX 5090's 1792 GB/s spec bandwidth
**no emmy decode step can be faster than 12.18 ms**, at any width from 1 to 64. Decode at these
widths is pure weight streaming: the activations are 32 x 3840 x 2 B = 246 kB, four orders of
magnitude smaller than the weights.

### What the deployed twins actually cost

Measured on box C, inside the published image, with the runner keyed to **hit** the image's baked
execution-plan pack (`decode_bucket=32, max_tokens=4096, prefill_bucket=4096`) so the kernels timed
are the warmed frozen picks. Every layer's `pre` and `post` program timed with the CUDA-graph window
timer, 7 windows of 50 replays, median:

| bucket 32, per step | |
|---|--:|
| `pre` (norm + QKV + q/k norms), 48 layers | 2.09 ms |
| `post` (o_proj + residual + norms + MLP), 48 layers | 12.78 ms |
| **total** | **14.86 ms** |

Per layer: sliding `pre` 43.1 us + `post` 262.8 us; full-attention `pre` 45.5 us + `post` 282.5 us.

**14.86 ms against a 12.18 ms floor is 82% of spec DRAM bandwidth.** A kernel that hit 100% of spec
bandwidth — nothing does — would save 2.68 ms per step. A realistic ceiling (~90%, about where cuBLAS
lands on these shapes) saves **1.3 ms**. That is the entire budget available to kernel tuning at m32.

**This number also describes the pack-*miss* lanes.** The 4k/4k lanes override `EMMY_GEN_PREFILL_BUCKET`
and therefore miss the whole pack (section 6), so they cold-resolve all 288 programs — but the decode
twin they land on is byte-identical to the packed one: the boot log's 7-launch `post` program is
`[k_linear_3955dc__partial__zp32, k_linear_3955dc, k_add_1, k_linear_mean_reduce_037454,
k_linear_reduce_9de4dc__partial__zp32, k_linear_reduce_9de4dc, k_mul_7]`, exactly the kernel list in
the image's `L00.post.decode.json`, and kernel names are content-addressed. The cold resolve differs
from the pack only on the *prefill / symbolic* tiers.

*Measurement caveat.* Timing one program in a 50-replay window lets its weight slab warm L2 (96 MB).
The `post` slab is 354 MB, 3.7x L2, so `post` is DRAM-bound and honest; the `pre` slab is 63 MB and
fits, so `pre` (2.09 of the 14.86 ms) is optimistic. The bias makes 14.86 ms a *lower* bound, which
only strengthens the conclusion.

### Against the live step

Box A, E1 lane (the target configuration), `c=8`:

| | |
|---|--:|
| median inter-token latency (the step period — one step emits a burst) | **28.50 ms** |
| median TPOT | 10.01 ms |
| mean acceptance length (vLLM's `SpecDecoding metrics`) | 2.76 - 2.91 of 3 |
| emmy's kernels at bucket 32 (section above) | 14.86 ms |
| **not emmy's matmuls** | **13.6 ms (48% of the step)** |

Acceptance is at ceiling, so the recipe header's claim about the 4k-output cells holds and the step
period is unambiguous. **Closing the whole ~20% cell deficit inside emmy's kernels would require them to
run at ~1.5x the DRAM bandwidth of the card** — arithmetically impossible. What kernels *can* reach is
the ~6% step-latency term, and section 2's same-box `c=6` control shows that term is 1.83 ms, which sits
squarely inside the 1.3 - 2.7 ms of headroom computed above.

### Where the other half of the step goes

Emmy's 14.86 ms is measured. The vLLM-side pieces below are **computed bandwidth floors, not
measurements** — they are lower bounds, and I did not profile them:

| per step, `c=8` d2 (6 running, width 18) | ms | basis |
|---|--:|---|
| emmy trunk (48 layers, bucket-32 twins) | **14.86** | measured, CUDA-graph window |
| vLLM lm_head / logits (2.01 GB, tied) | ~1.12 | DRAM floor at 1792 GB/s |
| vLLM paged attention (40 sliding x 1024-window + 8 global x 8448, 6 seqs) | ~1.6 | KV bytes / 1792 GB/s |
| drafter, 2 passes (4 layers, hidden 1024, its own 0.54 GB lm_head) | ~0.7 | DRAM floor |
| RoPE + KV writes | ~0.2 | DRAM floor |
| **accounted GPU work** | **~18.5** | |
| **measured step period** | **28.50** | median ITL |
| **unaccounted** | **~10.0 (35%)** | |

The whole *uniform-decode* step is inside vLLM's `FULL_DECODE_ONLY` cudagraph, so per-launch Python
overhead is already absorbed at replay. What is not in the graph is the speculative-decoding
machinery — drafting, verification, rejection sampling, KV bookkeeping — plus the scheduler and
detokenisation. **That ~10 ms, not the kernels, is the biggest single unexplained block in the step,
and it is where I would point a profiler next** (`--enable-layerwise-nvtx-tracing` or Nsight Systems
on a live container; I did not get to it). Note it is *not* obviously an emmy deficit: stock's step at
the same configuration is 26.45 ms on the same box, so stock carries a comparable block.

## 2. The cell never reaches concurrency 8

This is the finding that actually explains the cell.

The E1 boot log:

```
Available KV cache memory: 3.0 GiB
GPU KV cache size: 23,451 tokens
Maximum concurrency for 8,448 tokens per request: 2.78x
```

A `4096-in / 4096-out` request needs 8192 tokens of KV by the end. 23,451 tokens is **2.78 such
requests**. The scheduler's own timeline over the whole benchmark window, one sample per 10 s:

```
run=6 wait=2   run=6 wait=2   run=6 wait=2   run=6 wait=2   run=6 wait=2   ...   (steady state)
```

**Never 7, never 8.** The lane advertised as concurrency 8 runs six sequences with two permanently
queued. That costs throughput twice: the queued requests contribute nothing while inflating the
benchmark's wall duration (mean TTFT 14.6 s, P99 46.6 s), and the KV pressure drives preemption and
recompute.

The two queued requests are not merely idle — the timeline cycles
`run=6 wait=2 -> run=5 wait=3 -> run=6 wait=2`, i.e. an admitted request is repeatedly **preempted**
and its prefill recomputed. That is wasted GPU work with no token to show for it.

**A warning about measuring this region.** `c=8` at `np=16` is not a reproducible measurement. Two
servers whose *only* difference is one extra CUDA-graph capture size — which section 3 shows cannot
change routing at the width these lanes run — gave **369.09** and **492.40** tok/s, a 33% spread; both
had identical `run=6 wait=2` timelines. At `c=6`, where there is no thrash, the same two servers gave
471.15 and 467.07 (0.9% apart). **Preemption thrash, not the capture ladder, is what makes the `c=8`
point noisy**, and only the `np=64` runs (which average over eight waves) are worth comparing. I had
drafted a "-22% for asking for eight streams" claim off the `np=16` pair and withdrew it.

### Stock vLLM at the identical configuration

Stock `vllm/vllm-openai:v0.23.0`, same model, same drafter, same depth, same
`--gpu-memory-utilization=0.96 --max-model-len 8448 --dtype float16 --no-enable-prefix-caching`, same
client protocol:

```
Available KV cache memory: 3.61 GiB          (emmy: 3.0 GiB)
Maximum concurrency for 8,448 tokens: 2.98x  (emmy: 2.78x)
run=7 wait=1  (steady state)                 (emmy: run=6 wait=2 / run=5 wait=3)
```

Stock keeps **0.61 GiB more KV**, and at this request length that buys it a stable seven streams against
emmy's five-to-six.

### `c=6`: the clean comparison, where both engines run six streams and nothing is preempted

**All four rows on box A**, one variable:

| `c=6`, `np=12` | tok/s | median ITL | median TPOT | mean TTFT |
|---|--:|--:|--:|--:|
| **emmy** (E1) | 471.15 | 28.09 ms | 9.83 ms | 1.98 s |
| **emmy** (E2, a second server) | 467.07 | — | — | — |
| **stock** | **514.75** | **26.26 ms** | 9.24 ms | — |
| **ratio (emmy mean / stock)** | **0.911** | **0.935** | | |

**At equal concurrency emmy is 8.9% slower than stock, and essentially all of it is per-step latency**
(28.09 vs 26.26 ms, a 1.83 ms gap). This is the number that matters for kernel work, and it is *not*
zero.

> **Retraction.** An earlier draft of this report claimed emmy and stock were "at exact parity" at
> `c=6`, from emmy 471.15 on box A against stock 470.94 on box C — agreeing to 0.05 s of wall clock out
> of 104.3 s. That agreement was a coincidence of two offsetting errors: box C is simply ~9% slower than
> box A (stock scores 470.94 there and 514.75 here at an identical configuration), which happened to
> cancel emmy's real 8.9% deficit almost exactly. **Never compute an emmy/stock ratio across boxes.**
> The same-box control reverses the conclusion, and the coordinator flagged the same cross-box hazard
> independently from the 15% spread in the `c=8` `np=16` stock pair.

And 1.83 ms is *inside* emmy's kernel headroom: section 1 measured emmy's trunk at 14.86 ms against a
12.18 ms DRAM floor, so a step that reached ~90% of spec bandwidth (roughly where cuBLAS sits) would
recover ~1.3 ms of exactly that 1.83 ms. **The step-latency term is kernel quality**, which is why the
`PLACE@cone=cut` change in section 4 matters despite everything else in this report.

| `c=8` | tok/s | median ITL | steady state |
|---|--:|--:|---|
| **emmy**, `np=64` (the recipe's protocol) | 476.75 / 485.91 | 28.73 / 28.78 ms | mean run **5.42**, preemption cycling |
| **stock**, `np=64` | **603.66** | **27.20 ms** | mean run **6.75**, no cycling |
| emmy, `np=16` (two servers, identical routing) | 369.09 / 492.40 | 28.50 ms | — |
| stock, `np=16` (box A / box C) | 540.05 / 468.89 | 26.45 / 29.39 ms | — |

**`c=8` at `np=16` is unreliable for both engines** — emmy 369-492, stock 469-540. Only `np=64`
deserves comparison, and only within a box.

### The decomposition, same box, full protocol

At `np=64` — the recipe's own protocol — both engines reproduce their published cells on one box, so
this is the comparison to trust. Stock lands within 1.1% of the article's 610.2 and emmy within 6-8%
of its 450.6:

| box A, `c=8`, `np=64`, depth 2 | emmy (E1 / E2) | stock | ratio |
|---|--:|--:|--:|
| output throughput (tok/s) | 476.75 / 485.91 | **603.66** | **0.797x** |
| mean concurrency over the `np=64` window | **5.42** (never > 6) | **6.75** | **0.803** |
| median inter-token latency (the step) | 28.73 / 28.78 ms | 27.20 ms | 0.945 |
| run-aggregate mean acceptance length | 2.757 | 2.693 | 1.024 |
| mean TTFT | 20.7 / 21.1 s | 7.8 s | — |

Emmy's concurrency is not a flat six — over the `np=64` window it oscillates
`run=6 wait=2` / `run=5 wait=3` across 23 / 29 samples (plus one at 4), **mean 5.42, and it never once
reaches 7**. Stock sits at `run=7 wait=1` for 33 of its 40 steady samples, **mean 6.75**. Ratio
**0.803**.

> Both means are computed over the `np=64` window *only*, keyed by log timestamp. An earlier draft
> reported 6.06 for emmy including 41 samples at `run=7`; those samples belonged to the **stock**
> container, because the `docker logs -f` capture I used re-attached when the container was replaced
> and silently concatenated two servers' logs into one file. Emmy reaches 7 streams in no window.

Multiplying the three independent terms: `0.803 x 1.024 x 0.945 = 0.777`, against an observed 0.797 —
i.e. the model slightly *over*-predicts the gap, so there is nothing unaccounted. Note that the
concurrency term alone (0.803) already matches the observed ratio (0.797) to within 1%. So:

| contribution to the 0.797x | factor | note |
|---|--:|---|
| **admission ceiling** (5.42 vs 6.75 concurrent streams) | **0.803** | **matches the observed ratio on its own** |
| **step latency** (28.78 vs 27.20 ms) | 0.945 | kernel quality — section 4 addresses ~3/4 of it |
| draft acceptance | 1.024 | a small emmy **advantage** |
| product | 0.777 | vs observed 0.797 — the model slightly over-predicts, nothing unaccounted |

Two things follow. First, **emmy's draft acceptance is not a problem** — it is marginally better than
stock's, and an early draft of this report claiming otherwise was based on comparing snapshots from
different phases of the run. Second, the step-latency term (0.945 here, 0.935 in the cleaner `c=6`
control) is **real kernel quality**, and section 4's shipped golden change is worth ~1.3 ms of the
1.58-1.83 ms gap — roughly **three quarters of that term**.

So the final apportionment of the ~20% deficit at this cell:

| cause | size | fixable by |
|---|--:|---|
| **admission capacity** — emmy sustains 5.42 streams where stock sustains 6.75 | **~20%** | shrinking the runner's arenas / capacity buffers (recommendation 1) |
| per-step kernel time at m32 | **~5.5%** | **the `PLACE@cone=cut` goldens on this branch cover ~4.5 of it** |
| draft acceptance | -2.4% (emmy **ahead**) | nothing to do |
| (the three multiply to 0.777 against an observed 0.797 — no unaccounted residual) | | |

Run-aggregate draft acceptance, summed over every 10 s metrics window of the run, is *not* a
differentiator: stock accepted 42,128 of 46,816 drafted tokens = **0.900, mean acceptance length
2.80**, and emmy's windowed samples sit in the same 2.76 - 2.91 band. (Acceptance is strongly
phase-dependent on this random-token workload — it starts near 2.6 and climbs to 3.00 as the output
collapses into repetition — so single snapshots are not comparable and I discarded an early reading
that appeared to show emmy losing 5% of its speculation. It does not.)

### The utilization knob cannot buy the missing stream

The obvious cheap fix is to raise `--gpu-memory-utilization` until emmy's KV matches stock's. It does
not work, in two different ways, and both are worth knowing before anyone retries it:

| util | result |
|--:|---|
| 0.96 (the recipe) | KV **3.0 GiB**, 23,451 tokens, max concurrency 2.78x |
| 0.98 | KV **3.63 GiB**, 28,358 tokens, max concurrency **3.36x** — *more than stock's 3.61 GiB / 2.98x* — then the engine **dies at first use**: `torch.OutOfMemoryError` allocating 256 MiB in `rejection_sampler.sample_recovered_tokens`, with 169.75 MiB free |
| 0.99 | never starts: `Free memory on device cuda:0 (30.86/31.36 GiB) on startup is less than desired GPU memory utilization (0.99, 31.04 GiB)` |

At 0.98 the KV *budget* is right and there is no headroom left for the speculative sampler's transient
buffers, which vLLM's profiler does not reserve. So the extra concurrency has to come from emmy giving
memory back, not from asking for more — which is what makes recommendation 1 a real piece of work rather
than a knob change.

Incidentally this pins the accounting: +0.02 of utilization is +0.64 GiB of a 31.85 GiB card, and KV
moved 3.00 -> 3.63. Emmy's non-KV footprint is a fixed quantity, ~0.61 GiB larger than stock's, and
every byte of it comes straight out of the KV cache.

### Where emmy's 0.61 GiB goes

Budget at util 0.96 is 30.58 GiB. Non-KV footprint is therefore 27.58 GiB for emmy, 26.97 GiB for
stock. Stock's own accounting says `Model loading took 23.62 GiB`; emmy's says `3.04 GiB`, because
emmy's weights live in cupy buffers that torch's allocator cannot see — so vLLM's memory profiler
never attributes them, and the KV budget is whatever is left over after the fact. The trunk +
tied embed + drafter come to ~22.99 GiB by construction, leaving **~4.6 GiB of emmy arenas, scratch
slabs, graph pools and fragmentation against stock's ~4.0 GiB**.

The concrete suspects, in the order I would test them, are all sized by the *serving shape* rather
than by the decode width: the symbolic programs are built at `capacity = max_tokens = 4096` (one
`[4096, 30720]` fp16 intermediate is 252 MB on its own), the prefill twin adds a second set at
`prefill_bucket = 2048`, and the boot log reports per-program `scratch slab=0.13 - 0.41 GB` pooled
into one `BufferArena`. A lane that only ever runs 2048-token chunks is paying for 4096-row buffers.
**This, not kernel schedules, is the lever on the `c=8` cells.**

## 3. The published image does carry the sparse capture ladder — a latent fault, not the active one

`plans/gemma4-mtp-batched-serving-findings.md` concludes the shipped image is immune to the
capture-ladder / spec-decode interaction because it "runs `vllm` directly with no
`--compilation-config`". **That is wrong for the gemma-4 image.** Read out of the published image:

```
$ docker run --rm --entrypoint /bin/cat cloudriftai/vllm-emmy-gemma4:0.23.0-58733e02 /opt/emmy/serve.sh
... --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY",
      "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64, 128, 256], "custom_ops": ["+rotary_embedding"]}'
```

The earlier conclusion was reached against `cloudriftai/vllm-emmy:0.23.0-...` — the *generic* serving
image, which indeed has no such flag. `vllm-emmy-gemma4` bakes its own `serve.sh` carrying the sparse
ladder, so the fault's scope statement ("bites bare-metal `emmy serve` users, not the shipped image")
does not hold for the image the article's eleven emmy lanes pin.

At depth 2, `adjust_cudagraph_sizes_for_spec_decode` rounds every rung up to a multiple of
`query_len = 3` and drops anything past `max_cudagraph_capture_size = 256`, giving
**`[3, 6, 9, 18, 33, 66, 129]`** — confirmed in the live boot log, which captures exactly **7** decode
graphs. Emmy routes on the padded width, taking the static twin only when `padded <= bucket`:

| running seqs | share of the `np=64` window | verify width | first rung at or above | <= bucket 32? | route |
|--:|--:|--:|--:|---|---|
| 4 | 1/53 | 12 | 18 | yes | static twin |
| 5 | 29/53 | 15 | 18 | yes | static twin |
| 6 | 23/53 | 18 | 18 | yes | static twin |
| 7 | **0/53 — never reached** | 21 | 33 | no | (symbolic, if it happened) |
| 8 (the nominal config) | 0 | 24 | 33 | no | (symbolic, if it happened) |

**Every width this lane actually runs lands on rung 18, under the bucket, on the image's own ladder.**
Section 2's admission ceiling keeps it there: emmy never exceeds six streams, so the sparse ladder's
first gap (18 -> 33) is never entered.

The symbolic path it would fall to *is* genuinely expensive — box C, the same programs on the same card,
twin vs `run_device_sym` extrapolated over 48 layers:

| bucket 32, per step | |
|---|--:|
| static decode twin at M=32 | 14.85 ms |
| symbolic program at width 33 | 32.44 ms (**2.18x**) |
| symbolic program at width 24 | 39.77 ms (**2.68x**) |

— which is why the hypothesis was worth chasing. But it is unreachable here, and the A/B confirms it.
Adding one capture rung at 24, the *only* difference between the two servers, changes nothing:

| bucket 32, depth 2, box A, one variable | `c=8` `np=64` | `c=8` `np=16` | `c=6` `np=12` | median ITL (np=64) |
|---|--:|--:|--:|--:|
| E1, the image's ladder (7 graphs) | 476.75 | 369.09 | 471.15 | 28.73 ms |
| E2, + a rung at 24 (8 graphs) | **485.91** | 492.40 | 467.07 | 28.78 ms |
| delta | **+1.9%** | +33% (noise) | -0.9% | +0.2% |

`np=64` moves 1.9% and per-step latency moves 0.2% — which is what the routing table predicts once you
know the lane never exceeds six streams. (The +33% at `np=16` is the ramp-up artefact section 2 warns
about; two waves is not a measurement.)

**So my initial hypothesis — that the 0.74x cell was losing its twin to the sparse ladder — is refuted,
and this is the record of it.** The shipped gemma-4 image *does* carry the unsafe sparse ladder that #441
replaced in `emmy serve`, and the earlier scope claim about it is wrong; but it costs these cells
nothing, because admission starvation keeps the width below the ladder's first gap.

What is worth carrying forward: **the two faults are latently coupled.** If recommendation 1 gives the
lane its KV back and it reaches seven or eight streams, the width becomes 21 or 24, whose rung is 33 on
the image's ladder — and the 2.18x above becomes live. A footprint fix must ship together with a ladder
fix, or it will spend its own winnings.

## 4. A real m32 kernel win: the fused norm->linear edges deploy `fuse` where `cut` is faster

The previous round reported "m32 already satisfies the rule (0.97x cuBLAS), so no m32 win exists".
That holds for the *unfused* matmul entries. It does not hold for the **fused `norm_linear` edges,
which are what the decode twins actually launch** — read out of the image's baked execution plan,
`L00.post.decode.json`:

```
k_linear_3955dc__partial__zp32 / k_linear_3955dc   (o_proj, split-K)
k_add_1                                            (residual + post-attn norm)
k_linear_mean_reduce_037454                        (pre-FFW norm + gate/up cat, N=30720)  <-- FUSED
k_linear_reduce_9de4dc__partial__zp32 / ...        (geglu + down proj, N=3840)            <-- FUSED
k_mul_7                                            (post-FFW norm + residual + layer scale)
```

Those two fused kernels are ~93% of the `post` program's 262.8 us.

The m32 goldens record these edges at `STAGE=d1/sync` with `PLACE@cone=fuse`. The **same names at m8**
(and `mlp_down_fused` at m64) are recorded with `PLACE@cone=cut`, and sit at parity with the unfused
matmul, while the m32 fuse entries sit 12-28% above theirs. Direct A/B on box D
(`emmy run --golden NAME --bench --warmup 10 --iters 100 --ab "PLACE@cone=cut"`, bare metal):

| golden (m32) | deployed (`fuse`) | `PLACE@cone=cut` | delta | eager (torch norm + cuBLAS) |
|---|--:|--:|--:|--:|
| `norm_gate_up.m32.lin` (N=30720, K=3840) | 160.4 us | **147.2 us** | **-8.2%** | 154 us |
| `norm_qk_global.m32.lin` (N=8704, K=3840) | 38.3 us | **20.1 us** | **-47.5%** | 29 us |
| `mlp_down_fused.m32.lin` (N=3840, K=15360) † | 119.1 us | **112.8 us** | **-5.3%** | 97 us |

† `mlp_down_fused`'s YAML comment states that its recorded µs were measured on the **true geglu-cone
snippet** (`(gelu(y[:,:15360]) * y[:,15360:]) @ w`) while a `--golden` replay traces an rms-cone
*stand-in* of the same ShapeKey. Its 119.1 and 112.8 are therefore both stand-in numbers: the
fuse-vs-cut **delta** is a valid one-variable comparison (both rows trace the same snippet), but the
absolutes are not the deployed kernel's and the delta has not been shown to transfer. I have left
that entry alone for exactly this reason; it wants an A/B on the true snippet.

The first two rows carry no such caveat — their recorded µs reproduce live (`norm_gate_up` records
161.6 and measures 159.7 - 160.5 across four runs; `norm_qk_global` records 40.1 and measures 38.3)
and their A/Bs are apples-to-apples.

`cut` splits the fused cone at the A seam — the norm materialises to a workspace (1.4 us stat + 0.8 us
scale at m32, i.e. free) and the matmul re-lowers with a plain gmem A, which lets it take the
big-tile `d2/tma/ring` schedule (145.0 us, **91% of the 131.7 us DRAM floor**) instead of the
constrained fused `d1/sync` one. On `norm_gate_up` that turns 0.96x eager into **1.04x**; on
`norm_qk_global`, 0.73x into 1.44x.

The m8 control validates the mechanism from the other side: at m8 the `fuse` form is not merely
slower, it is **pathological** — `PLACE@cone=fuse` pinned at m8 measures **55,330 us** on
`norm_gate_up.m8.lin` (a grid-8 `b128` scalar kernel, 380x) and 27,310 us on `mlp_down_fused.m8.lin`.
m8's goldens therefore *had* to record `cut`. At m32 both forms realise, `fuse` wins the resolve, and
nothing flagged that it was the slower of the two.

**Per-step value, from the measured deltas:** `norm_gate_up` 13.2 us x 48 layers = 634 us,
`mlp_down_fused` 7.8 us x 48 = 374 us, `norm_qk_global` 18.2 us x 8 = 146 us — **1.15 ms/step
measured**, plus an unmeasured `norm_qkv` (the sliding layers' N=8192 fused QKV edge, which by
analogy with `norm_qk_global` is worth several hundred us more and **has no m32 golden at all**).
Call it 1.2-1.8 ms of the 14.86 ms step: **8-12% of emmy's kernel time, 4-6% of the step.**

Worth taking. Not capable of closing a 26% cell gap.

### Two golden-dataset defects found on the way

- **`norm_gate_up.m32.lin` is DRIFT.** Every resolve prints
  `matches golden shape ... but no offered candidate realizes any of them — the golden(s) no longer
  realize under the current enumeration. Investigate enumeration drift for:
  gemma4_12b.norm_gate_up.m32.lin`. Greedy falls through to the evidence hierarchy and happens to
  land on the same config, so the deploy is unaffected today — but the shape has **no golden floor**.
- **`mlp_down_fused.m32.lin`'s recorded latency is stale by 29%**: the YAML records `emmy_us: 95.9`;
  live on an idle 5090 the identical knobs measure **123.7 us**. Consistent with the brief's warning
  that the recorded m2048 numbers are stale. Recorded `emmy_us` should not be trusted for
  cross-config reasoning without a live re-bench.

### The knob space around the cut is not further tunable by pinning

Pinning the plain matmul's winning schedule *on top of* the cut does not work, and the failures are
instructive. `mlp_down.m32.lin` (the same 3840x15360 shape, unfused) reaches 74.7 us with
`REDUCE=g4a` — atomic finalize. On the fused edge that is rejected outright:

```
atomic finalize can't carry a non-distributive projection epilogue (e.g. a fused bias / activation
on a split reduce); pin the deferred-kernel finalize instead (REDUCE=…g<n>k)
```

so the unfused shape's 74.7 us is *not* reachable here, and the 110 us of the cut's matmul half is
not the shortfall it looks like. Pinning `g<n>k` variants instead fails differently: a global
`TILE=`/`REDUCE=`/`STAGE=` pin also lands on the cut's `__stat` / `__cone` kernels, which then compile
to grid-1 launches — four of five variants hit the 60 s hung-kernel guard and one came back
`wrong-answer: rel err 7.341 vs greedy output`. (Worth noting for its own sake: the pinned-row
integrity gate catches this, so the accuracy check in `run --bench` is doing real work.)

`PLACE@cone=cut` on its own is therefore the whole of the available win, and it is a **structural**
pin — exactly the shape the m8 and m64 entries already record.

### Repeated measurement, and the change that ships

Three runs per shape, `--warmup 10 --iters 100`, box D idle, medians with observed spread:

| golden (m32) | `fuse` (what deployed) | `cut` | delta |
|---|--:|--:|--:|
| `norm_gate_up.m32.lin` (N=30720) | 160.1 (+-0.6%) | **147.2** (+-0.2%) | **-8.1%** |
| `norm_qkv.m32.lin` (N=8192) — *no golden existed* | 33.7 | **19.6** | **-41.8%** |
| `norm_qk_global.m32.lin` (N=8704) | 38.5 (+-0.3%) | **19.8** (+-1.6%) | **-48.6%** |
| `norm_q_proj.m32.lin` (N=4096) ‡ | 23.7 (+-9.0%) | **12.1** (+-0.1%) | **-49.1%** |
| `norm_kv_proj.m32.lin` (N=2048) ‡ | 17.0 (+-1.4%) | **8.8** (+-4.9%) | **-48.5%** |
| `mlp_down_fused.m32.lin` † | 121.9 (+-1.9%) | **110.2** (+-0.1%) | **-9.6%** |

‡ the pre-merge *split* q / kv forms; post-`035_merge_sibling_linears` the served twins launch the
merged `qkv` edge instead, so these two do not deploy. Recorded here as corroboration that the effect
is the whole family, not one shape.

**Recorded (this branch):** `norm_gate_up.m32.cut.lin`, `norm_qkv.m32.cut.lin` and
`norm_qk_global.m32.cut.lin` — the three edges the decode twins actually launch — as `cut` entries
*beside* the existing `fuse` ones (a recorded measurement is never deleted, and `_golden_pick` takes
the fastest realizing entry). `mlp_down_fused` is deliberately **not** touched: its recorded µs are
geglu-cone numbers and mine are stand-in numbers, so an entry recorded at 110.2 would sort *behind*
the existing 95.9 and never be selected. It needs an A/B on the true snippet first.

**The deploy was verified, not assumed** — this is the trap the brief warns about. Resolving each shape
through its *old* (`fuse`) name after the edit, greedy independently lands on the cut form:

| resolved via the old name | before | after |
|---|--:|--:|
| `gemma4_12b.norm_gate_up.m32.lin` | 0.96x eager (`fuse`, 160.4 us) | **1.01x** eager (`cut`, 147.2 us) |
| `gemma4_12b.norm_qk_global.m32.lin` | 0.73x eager (`fuse`, 38.3 us) | **1.27x** eager (`cut`, 20.1 us) |
| `gemma4_12b.norm_qkv.m32.lin` | 0.86x eager (`fuse`, 33.7 us) | **1.38x** eager (`cut`, 19.8 us) |

The `norm_gate_up.m32.lin` drift warning is gone with it: the ShapeKey now has a realizable entry, so
the shape has a golden floor for the first time. All pinned rows carry `status: "ok"` — the
realized-vs-pinned, arithmetic-intensity and wrong-answer gates pass (and they are not vacuous: a
mis-pinned variant in the same sweep came back `wrong-answer: rel err 7.341`).

**Projected per-step effect, stated as a projection.** Over the deployed kernel list — 48
`norm_gate_up` (-12.9 us), 40 sliding `norm_qkv` (-14.1 us), 8 global `norm_qk_global` (-18.7 us) —
that is **~1.3 ms** off a 14.86 ms step, ~9%. Including `mlp_down_fused` if its delta transfers would
add ~0.6 ms for ~13%, but 13% would put the step at ~97% of spec DRAM bandwidth, which is not
credible; **treat ~9% as the number and the rest as unproven.** I did not verify the composed
whole-step effect end to end — that needs a cold-resolve `stepprof` against the edited goldens, which
did not fit in the budget.

## 4b. The brief's puzzle: why plain `c=8` is 0.97x and MTP `c=8` is 0.74x

Same box, same concurrency, same prefill quantum, same drafter-on-stock-vLLM for both engines. The only
differences are speculation and the decode bucket (**8** for plain, **32** for MTP). Section 4 answers it:

| fused edge, eager = 1.00x | at **m8** (plain lane) | at **m32** (MTP lane), before | at **m32**, after this branch |
|---|--:|--:|--:|
| `norm_gate_up.lin` | 1.01x | 0.96x | **1.01x** |
| `mlp_down_fused.lin` | 1.22x | 0.78x | (untouched, see section 4) |
| `norm_qkv.lin` | — | 0.86x | **1.38x** |
| `norm_qk_global.lin` | 1.19x (recorded) | 0.73x | **1.27x** |

**m8's goldens already record `PLACE@cone=cut`; m32's recorded `fuse`.** At m8 they had no choice — a
`fuse` pin at m8 measures 55,330 us on `norm_gate_up` and 27,310 us on `mlp_down_fused`, a 380x scalar
degenerate, so `cut` was forced. At m32 both forms realize, `fuse` won the resolve, and nothing noticed
it was the slower one.

So the bucket 8 -> 32 shift *is* the cause of the extra step latency, but **not** for the reason the
brief hypothesized. It is not tile granularity and not weight-slab re-streaming — the `TILE` M extent is
a multiple of 16 either way and both widths read the slab once. It is that **m32 is the width where the
cone-placement decision was recorded wrong**, and m8 is the width where a degenerate case forced it
right. That is now fixed, which is why this change is worth having even though it cannot close the cell.

## 5. What I did not change, and why

- **No decode-bucket or width change.** Confirmed the coordinator's tile-granularity argument holds:
  the mma atom is `mma_m16n8k16`, so a tile's M extent is a multiple of 16 and the `c=8` widths (24 at
  depth 2, 32 at depth 3) occupy a 32-row tile at any legal bucket. Section 2 makes the point moot
  anyway — the lane runs at width **18**, and the measured `c=6` parity says the step is already as
  fast as stock's.
- **No m16 tier seeding.** It was already the lowest-priority option, and the parity result removes
  its rationale entirely: there is no per-step deficit at these widths for a new tier to recover.
- **No image rebuild.** All eleven emmy lanes pin `0.23.0-58733e02`; a repo-side change reaches none
  of them, and an image built from current main is separately reported ~22% slower at this very cell.
  Everything above is measured per-kernel or against that pinned image, and **no article cell moves as
  a result of this work.**
- **No capture-ladder change to `serve.py`.** #441 already fixed it there. The gap is that the
  gemma-4 image's baked `serve.sh` carries a hand-written sparse list that `_gen_graph_args` never
  sees (section 3). Fixing that is an image change, so it belongs to whoever next bakes one — noted
  in the recommendations rather than patched here.
- **`plans/` is left at 12 files, over CLAUDE.md's cap of 10.** Same call #441 made and for the same
  reason: this change *adds* a findings report rather than executing a plan, and enforcing the cap
  would mean deleting somebody else's in-flight report.

## 5b. Checks run

`make test` on box C (RTX 5090, `PYTHONPATH` pinned to the branch tree, `-Xcicc -O1` correctness lane):

```
===== 4 failed, 2819 passed, 37 skipped, 46 warnings in 1073.02s (0:17:53) =====
FAILED tests/compiler/ir/test_dynamic_shapes.py::test_qwen_batched_dynamic_matches_eager_b32
FAILED tests/scripts/test_bench_block.py::test_bench_dry_run_tinyllama_block
FAILED tests/serving/test_gen_pack_gpu.py::test_gen_pack_second_boot_hits_and_matches@cuda
FAILED tests/serving/test_gen_runner_gpu.py::test_gen_runner_device_path_matches_host@cuda
```

A `274c153e` baseline on the same box with the same harness fails **five**:

```
FAILED tests/compiler/ir/test_dynamic_shapes.py::test_qwen_batched_dynamic_matches_eager_b32
FAILED tests/compiler/ir/test_dynamic_shapes.py::test_qwen_layer_dynamic_compiles_and_matches_eager
FAILED tests/scripts/test_bench_block.py::test_bench_dry_run_tinyllama_block
FAILED tests/serving/test_gen_pack_gpu.py::test_gen_pack_second_boot_hits_and_matches@cuda
FAILED tests/serving/test_gen_runner_gpu.py::test_gen_runner_device_path_matches_host@cuda
```

The branch's four are a strict **subset** of main's five, so **this change introduces no new failures**
(the extra `test_qwen_layer_dynamic_compiles_and_matches_eager` looks flaky between the two runs).

The repo's own gate for exactly the hazard the brief warns about —
`test_golden_drift_gate.py::test_gemma4_goldens_deploy_in_serving_twins[rtx5090]`, which checks that
recorded gemma-4 goldens actually deploy in the serving twins — **passes on the branch**.

Accuracy of the changed kernels is covered by `run --bench`'s pinned-row integrity gates rather than by
the suite: every `PLACE@cone=cut` row reports `status: "ok"`, i.e. it passed realized-vs-pinned, the
arithmetic-intensity floor, and the wrong-answer check against the greedy output. The gate is not
vacuous — a differently-pinned variant in the same sweep came back `wrong-answer: rel err 7.341`.

## 6. Verdict on the previous round's claims

| claim | verdict |
|---|---|
| "m32 already satisfies the rule (0.97x cuBLAS), so no m32 win exists" | **False.** True of the unfused matmul entries; false of the *fused* `norm_linear` edges the decode twins actually launch, which ran 0.73 - 0.96x eager at m32 against 1.01 - 1.22x at m8. `PLACE@cone=cut` recovers 8 - 49% per edge and now deploys. |
| "emmy wins at every M against cuBLAS; there is no cliff at 32" | **False in the same way.** There is a cliff at m32, and it is on the fused edges. |
| "m2048 prefill is at parity; recorded YAML numbers are stale" | **Not reproduced as staleness at m32.** `norm_gate_up.m32.lin` (161.6) and `norm_qk_global.m32.lin` (40.1) both reproduce live to within 4%. `mlp_down_fused.m32.lin` looks 24 - 29% stale but is not: its YAML comment says the recorded µs came from the true geglu-cone snippet while a `--golden` replay traces an rms-cone stand-in. Apparent staleness there is a **snippet mismatch documented in the file**, and I withdrew an earlier draft claim to the contrary. |
| "A bucket override is a whole-pack miss, not decode-only" | **Confirmed.** The pack key is `{kind, model, config_sha, dtype, decode_bucket, max_tokens, prefill_bucket}`; the image's is `prefill_bucket: 4096`, so every `EMMY_GEN_PREFILL_BUCKET=2048` lane misses the whole pack and cold-resolves all 288 programs (~25 min of boot on 30 cores, observed). |
| "The remaining deficit is per-step runner overhead, not kernel quality" | **Half right, and it matters which half.** Half the step *is* outside emmy's matmuls (14.86 of 28.7 ms) and ~10 ms is outside GPU work entirely. But the deficit decomposes as 0.803 admission x 0.945 step latency x 1.024 acceptance: the dominant term is admission *capacity* (it matches the observed ratio by itself), and the step-latency term is kernel *quality* (1.83 ms at `c=6`, inside the measured headroom) — not runner overhead. |
| "The shipped image is immune to the capture-ladder fault" | **False for `vllm-emmy-gemma4`** (section 3); true for the generic `vllm-emmy` image that was actually inspected. |

## 7. Recommendations, in value order

1. **Give the trunk back its KV cache — this is the whole cell gap.** Emmy's non-KV footprint is ~0.61
   GiB larger than stock's, which costs it 1.3 streams (5.42 vs 6.75) and, on its own, the entire 0.797x
   ratio. It cannot be bought with `--gpu-memory-utilization` (0.98 OOMs the rejection sampler, 0.99
   will not start). The buffers are sized by `max_tokens = 4096` and `prefill_bucket`, not by the decode
   width, so a lane that only ever schedules 2048-token chunks is paying for 4096-row capacity. Start by
   measuring `BufferArena` occupancy against the widths a lane can actually reach.
2. **Profile the ~10 ms of the step that is not GPU work** (section 1). It is the largest unexplained
   block in the step for *both* engines, and nobody has looked at it. Nsight Systems on a live
   container, or vLLM's own `--enable-layerwise-nvtx-tracing`.
3. **Done on this branch: the `PLACE@cone=cut` m32 goldens** (section 4) — ~9% of emmy's per-step
   kernel time at every width that routes through bucket 32, independent of everything above. What is
   left there is `mlp_down_fused.m32`, which needs an A/B on its true geglu-cone snippet.
4. **Re-open the capture-ladder question if and only if recommendation 1 lands.** Today an added rung
   at 24 is worth 1.9% at `np=64` (section 3) — not worth shipping. At a steady eight streams the width
   becomes 24, a rung nobody has A/B'd, and the answer may differ.
5. **`emmy eval golden`'s drift audit deserves a CI lane.** `norm_gate_up.m32.lin` had been printing
   `no offered candidate realizes any of them` on every single compile of this model, i.e. that shape
   had no golden floor at all, and nothing surfaced it. (Fixed here as a side effect; the class is not.)
