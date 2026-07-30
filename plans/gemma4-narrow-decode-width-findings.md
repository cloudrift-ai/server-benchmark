# Narrow decode widths (m8 / m32) on gemma-4-12B, RTX 5090 — where the MTP batch gap actually lives

Target: the two largest emmy-vs-stock gaps in the article's speculative-decoding table — `4k/4k c=8`
with the official drafter at depth 2 (**0.74x**) and depth 3 (**0.77x**). Both run decode bucket 32
with the 2048-token prefill quantum, so the brief's leading suspect was m32 kernel quality.

**Answer: those cells are not kernel-bound, and they are not even running at concurrency 8.** Emmy
leaves 3.0 GiB for the KV cache where the workload needs room for eight 8448-token sequences; vLLM
admits six and permanently queues two. Emmy's whole per-step kernel budget at m32 is 14.9 ms against
a measured 28.5 ms step, so *no* kernel change can move the cell by more than a few percent.

A real m32 kernel win does exist and is worth taking on its own terms (section 4): the fused
norm->linear edges at m32 deploy the `PLACE@cone=fuse` schedule where `cut` is 6-48% faster. It is
worth ~8-12% of emmy's per-step kernel time — a genuine correction to the previous round's "no m32
win exists" — but that is ~4-6% of the step, not 26%.

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

**Benchmark deviation, stated up front:** the client runs `--num-prompts 16` instead of the recipe's
64 (the request mix and the prefill:decode token ratio are identical — 4096 in / 4096 out per
request — but the run takes 3 minutes instead of 10, and the drain tail is a larger share, so
absolute tok/s reads *lower* than the article's). Every comparison below is same-box, one-variable,
same `num_prompts`. Absolute numbers are not comparable to the article's cells; ratios between my
own lanes are.

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
period is unambiguous. Emmy's entire matmul budget is half the step; the other half is vLLM's
attention + RoPE + lm_head + sampling, the drafter's two forward passes, and scheduling. **Closing
26% of this cell inside emmy's kernels would require them to run at 1.9x the DRAM bandwidth of the
card.** It is arithmetically impossible.

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

Direct evidence that the extra requests are *harmful*, not merely idle — same server, booted once,
only the client's concurrency varies:

| client concurrency | tok/s | median TPOT |
|--:|--:|--:|
| 6 | **471.15** | 9.83 ms |
| 8 | 369.09 | 10.01 ms |

Asking for eight streams makes the system **22% slower** than asking for six. Per-token latency is
flat (9.83 vs 10.01 ms), so the loss is not step cost — it is admission thrash.

### Stock vLLM at the identical configuration

Stock `vllm/vllm-openai:v0.23.0`, same model, same drafter, same depth, same
`--gpu-memory-utilization=0.96 --max-model-len 8448 --dtype float16 --no-enable-prefix-caching`, same
client protocol:

```
Available KV cache memory: 3.61 GiB          (emmy: 3.0 GiB)
Maximum concurrency for 8,448 tokens: 2.98x  (emmy: 2.78x)
run=7 wait=1  (steady state)                 (emmy: run=6 wait=2)
```

Stock keeps **0.61 GiB more KV** — which at this request length is worth exactly one more concurrent
sequence. Then:

| `c=6`, `np=12` | duration | tok/s | median ITL | median TPOT | mean TTFT |
|---|--:|--:|--:|--:|--:|
| **emmy** (box A) | 104.32 s | **471.15** | 28.09 ms | 9.83 ms | 1.98 s |
| **stock** (box C) | 104.37 s | **470.94** | 28.66 ms | 9.97 ms | 1.68 s |

| `c=8`, `np=16` | duration | tok/s | median ITL | median TPOT | steady state |
|---|--:|--:|--:|--:|---|
| **emmy** (box A) | 177.56 s | **369.09** | 28.50 ms | 10.01 ms | run=6 wait=2 |
| **stock** (box C) | 139.77 s | **468.89** | 29.39 ms | 10.51 ms | run=7 wait=1 |

**At the concurrency emmy can actually sustain, emmy and stock are at exact parity — 471.15 against
470.94, a 0.05% difference, with emmy's per-step latency 2% *lower*.** (The two lanes ran on different
boxes; agreeing to 0.05 s of wall clock over 104 s also retires the box as a confound.)

Going from six streams to eight, stock is flat (470.94 -> 468.89) while emmy loses 22%
(471.15 -> 369.09). **The entire `c=8` deficit is that stock holds seven streams and emmy holds six,
plus the thrash emmy pays for being asked for eight it cannot admit.**

Run-aggregate draft acceptance, summed over every 10 s metrics window of the run, is *not* a
differentiator: stock accepted 42,128 of 46,816 drafted tokens = **0.900, mean acceptance length
2.80**, and emmy's windowed samples sit in the same 2.76 - 2.91 band. (Acceptance is strongly
phase-dependent on this random-token workload — it starts near 2.6 and climbs to 3.00 as the output
collapses into repetition — so single snapshots are not comparable and I discarded an early reading
that appeared to show emmy losing 5% of its speculation. It does not.)

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

| running seqs | verify width | first rung at or above | <= bucket 32? | route |
|--:|--:|--:|---|---|
| 6 (**what actually runs**) | 18 | 18 | yes | **static twin** |
| 7 | 21 | 33 | no | symbolic |
| 8 (the nominal config) | 24 | 33 | no | symbolic |

So the ladder does **not** bite this cell — because section 2's KV starvation pins it at six, one of
the two widths that still lands on a rung under the bucket. My initial hypothesis (that the 0.74x
cell was running the symbolic program) is **refuted by measurement**, and I am recording it as such.

It matters anyway, for one reason: **the two faults are coupled.** Fixing the KV footprint so the
lane reaches seven or eight streams moves the width to 21 or 24, which lands on rung 33 and retires
the twin. Measured cost of that, box C, same programs, same card:

| bucket 32, extrapolated 48-layer step | |
|---|--:|
| static decode twin at M=32 | 14.85 ms |
| symbolic program at width 24 | 39.77 ms (**2.68x**) |
| symbolic program at width 33 | 32.44 ms (**2.18x**) |

Per layer, sliding: twin 305.8 us, `sym@33` 677.4 us. So any KV fix must ship together with a ladder
rung at or below the bucket for the widths it unlocks, or the concurrency win is spent twice over.

*Caveat:* the twin rows are CUDA-graph replays and the symbolic rows are timed around `run_once`
(the symbolic path declines graph capture, so this matches production), which charges the symbolic
rows for per-launch dispatch the twin rows do not pay. At ~7 launches per program that is under 10%
of the gap; the 2.18x is overwhelmingly GPU work.

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
| `mlp_down_fused.m32.lin` (N=3840, K=15360) | 123.7 us | **115.9 us** | **-6.3%** | — |
| `norm_qk_global.m32.lin` (N=8704, K=3840) | 38.3 us | **20.1 us** | **-47.5%** | 29 us |

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

## 6. Verdict on the previous round's claims

| claim | verdict |
|---|---|
| "m32 already satisfies the rule (0.97x cuBLAS), so no m32 win exists" | **False.** True of the unfused matmul entries; false of the *fused* `norm_linear` edges the decode twins actually launch, which run 0.73 - 0.96x eager at m32 against 1.01 - 1.22x at m8. `PLACE@cone=cut` recovers 6 - 47% per edge. |
| "emmy wins at every M against cuBLAS; there is no cliff at 32" | **False in the same way.** There is a cliff at m32, and it is on the fused edges. |
| "m2048 prefill is at parity; recorded YAML numbers are stale" | **Stale confirmed, independently.** `mlp_down_fused.m32.lin` records 95.9 us; live it is 119 - 124 us (+24 - 29%). Recorded `emmy_us` should not be used for cross-config reasoning. |
| "A bucket override is a whole-pack miss, not decode-only" | **Confirmed.** The pack key is `{kind, model, config_sha, dtype, decode_bucket, max_tokens, prefill_bucket}`; the image's is `prefill_bucket: 4096`, so every `EMMY_GEN_PREFILL_BUCKET=2048` lane misses the whole pack and cold-resolves all 288 programs (~25 min of boot on 30 cores, observed). |
| "The remaining deficit is per-step runner overhead, not kernel quality" | **Half right, and the wrong half matters.** Roughly half the step is outside emmy's matmuls (14.86 of 28.5 ms), which is real — but it is *not* a deficit: at equal concurrency emmy's step is 2% *faster* than stock's. The deficit is admission capacity, not per-step time. |
| "The shipped image is immune to the capture-ladder fault" | **False for `vllm-emmy-gemma4`** (section 3); true for the generic `vllm-emmy` image that was actually inspected. |

## 7. Recommendations, in value order

1. **Give the trunk back its KV cache.** One concurrent sequence at this request length costs 0.61
   GiB, and that single sequence is the whole `c=8` gap. The buffers are sized by `max_tokens = 4096`
   and `prefill_bucket`, not by the decode width, so a lane that only ever schedules 2048-token chunks
   is paying for 4096-row capacity. Start by measuring `BufferArena` occupancy against the widths a
   lane can actually reach.
2. **Put the capture ladder in the baked image.** `serve.sh` hard-codes
   `[1,2,4,8,16,32,64,128,256]`, which #441 replaced in `emmy serve` precisely because it is unsafe
   under speculation. It is inert today only because KV starvation pins the width at 18; recommendation
   1 will move the width to 21 or 24, both of which land on rung 33 and cost **2.18x** on the step.
   These two must ship together.
3. **Record the `PLACE@cone=cut` m32 goldens** (section 4) — worth 8 - 12% of emmy's per-step kernel
   time at every width that routes through bucket 32, independent of everything above.
4. **`norm_qkv` has no m32 golden at all**, though it is the sliding layers' fused QKV edge and
   therefore 40 of 48 `pre` programs. Its m8 and m64 siblings both exist and both record `cut`.
5. **Fix the `norm_gate_up.m32.lin` drift** — the resolve prints `no offered candidate realizes any of
   them` on every compile, so the shape has no golden floor. It lands on the right config today by
   luck of the evidence hierarchy.
