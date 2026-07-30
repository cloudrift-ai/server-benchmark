# Batched MTP serving on the RTX 5090 — what was wrong and what changed

Goal: bring emmy's speculative-decoding (MTP) serving throughput at batch up to stock vLLM's for
gemma-4-12B-it with the official drafter. Single-stream was already at parity; the batched cells sat
at 0.24x to 0.74x.

Short version. I found and fixed a real fault that silently switched emmy's tuned decode kernels off
under speculative decoding, worth **+30% to +34%** on decode-dominated load. But I also established
that this fault **does not explain the article's numbers**, because the published serving image never
runs the code path that has it. The article's batched gap has two other causes: the 4k/4k cells are
prefill-bound rather than decode-bound, and the c=64 cell runs on a kernel width with no tuning data
at all. Details, evidence, and the things that turned out to be wrong are all below.

---

## Fault 1 — the capture ladder collides with speculative decoding (`emmy serve` only)

Under speculative decoding vLLM rounds every CUDA-graph capture size **up** to a multiple of
`query_len = num_speculative_tokens + 1`
(`CompilationConfig.adjust_cudagraph_sizes_for_spec_decode`, vLLM 0.23, guarding its issue #28207).
A decode step is then padded to the first captured size at or above its width, and the model sees
that **padded** width.

Emmy handed vLLM a sparse power-of-two ladder (`emmy/commands/serve.py`, `_gen_graph_args`). At depth
2 `query_len` is 3, and no power of two is a multiple of 3, so every rung moved up:

    [1,2,4,8,16,32,64,128,256]  ->  [3,6,9,18,33,66,129]

The 16 and 32 rungs became 18 and 33 — one step **above** the decode bucket they existed to serve.
Emmy routes on the padded width (`gen_runner.py`, `t <= decode_bucket`), so every steady-state verify
step missed its static decode twin and ran the symbolic masked-tile program instead: runtime-sized
grids with a boundary guard, where the bucket exists to supply a goldens-tuned static kernel.

Two notes on what this is *not*. The cost is kernel quality, not launch overhead — vLLM still
captures the whole step in its own graph at the padded size, so nothing runs eager. That is
consistent with `plans/decode-parity-closers-findings.md` having refuted launch-count theories three
separate ways at m1 and m32. And the loss was invisible because the twins audit only covers widths it
is handed, so a rung one step above the bucket looks like ordinary symbolic traffic.

### Measured: the routing cliff

One server, booted once, never restarted — decode bucket 32, quantum 2048, mnbt 2080, depth 2, the
unfixed ladder. Its boot log reports seven decode graphs captured, exactly the predicted
`[3,6,9,18,33,66,129]`. Only client concurrency varies, so the routing boundary falls between 6 and 7:

| conc | width M | padded rung | <= bucket 32? | route | tok/s | TPOT (ms) |
|--:|--:|--:|---|---|--:|--:|
| 5 | 15 | 18 | yes | static twin | 357.40 | 11.49 |
| 6 | 18 | 18 | yes | static twin | 414.54 | **11.87** |
| 7 | 21 | 33 | no | symbolic | 366.53 | **16.56** |
| 8 | 24 | 33 | no | symbolic | 443.14 | 16.24 |

Per-token latency steps up ~38% precisely where the padded width crosses the bucket, and adding a
seventh concurrent request makes total throughput *fall below* what six achieved.

### SCOPE — this does not affect the published image

I initially believed this fault explained the article's batched cells. It does not, and the
correction matters more than the original claim.

The published serving image does not invoke `emmy serve`. Its entrypoint is `vllm` directly with
`--hf-overrides '{"architectures":["EmmyGenModel"]}'` and **no `--compilation-config`**, so
`_gen_graph_args` never runs and none of emmy's graph-capture choices apply. Verified on a live
container of `cloudriftai/vllm-emmy:0.23.0-90d010ef`:

- its full argument list contains no `--compilation-config`;
- its log reports `cudagraph_mode: FULL_AND_PIECEWISE` (vLLM's default), not emmy's `FULL_DECODE_ONLY`;
- its ladder is vLLM's own **dense** default, `[1,2,4,8,16,24,32,40,48,...,256,272,...,512]`.

A dense ladder is immune: floored to multiples of 3 it still contains 18, 24 and 30, so for any width
up to the bucket there is always a rung just above it that stays *under* the bucket. Only a sparse
ladder leaves a gap (18 to 33) wide enough to jump the bucket.

Confirmed by measurement on a second box: on that container at the same knobs the cliff **does not
reproduce** — concurrency 6 gives 536.57 tok/s at TPOT 9.62, concurrency 7 gives 602.30 at TPOT 9.85.
It keeps scaling.

So the fault is real and costly on the `emmy serve` path — used for kernel development, bare-metal
serving, and by anyone following the README — but it is not why the article's numbers look the way
they do. Two side findings worth the team's attention:

1. **The shipped image bypasses emmy's own serving configuration.** Whatever `emmy serve` decides
   about capture ladders and cudagraph mode, deployments do not get it. That divergence between the
   tuned path and the shipped path should be deliberate, not incidental.
2. **vLLM's default ladder is better than emmy's override for speculative decoding.** Emmy's sparse
   power-of-two list is strictly worse here; the dense default needs no fixing.

## Fault 2 — width 192 has no tuned kernels at all

The c=64 depth-2 cell routes correctly (its bucket of 192 is itself a multiple of 3, so the rung
matches and the static twin runs). The problem is that the twin is 192 wide and
`goldens/rtx5090_sm120_gemma4.yaml` has **no records at that width** — none of any kind. Recorded
widths are 1, 8, 16, 32, 64, 256, 512, 1024, 2048, 4096. With no record the fused twins fall to a
cold greedy search, which is the misdeploy class the golden file's own header describes (a scalar
b256 tile around 37 ms, ~770x off cuBLAS); one such kernel per layer over 48 layers is plenty.

A related gap that matters once routing is correct: width 16 carries only the *unfused* projection
shapes. Every fused serving twin — `norm_qkv.lin`, `norm_gate_up.lin`, `qkv_cat.lin`,
`gate_up_cat.lin`, `mlp_down_fused.lin`, `cut_cone_*`, `rms_norm.k3840`, all the `pw.n*` — exists at
m32 and is missing at m16. So a decode bucket of 16 is a much weaker place to land than 32 even when
routing is correct.

---

## What changed

### The ladder fix (`9006ddd7`, refined in `a9200a22`)

`serve.py` now derives `query_len` from `--speculative-config` and, when it is greater than 1, builds
the ladder from **dense** candidates (mirroring vLLM's stride-8 default) each **floored** to a
multiple of `query_len`. Flooring only moves a rung down, so the bucket's own rung stays reachable
and vLLM's round-up becomes a no-op; density then removes the leftover padding. Depth-3 and depth-5
ladders are unaffected, which is why only depth 2 ever broke.

Two kinds of overshoot are worth separating, because I conflated them at first:

- **past the bucket** — fatal, routes the step to the symbolic program (the 18-vs-16, 33-vs-32 case);
- **within the bucket** — merely wasteful padding, and cheap (see the rung scan below).

Verified by running emmy's real `_gen_graph_args` through vLLM's real adjustment function: a
24-token step now lands on 24 exactly, and a 192-token step on 192 exactly, instead of on 30 and 255.

`gen_runner.py` also gains `_warn_symbolic_decode`, which reports once per width when a decode-shaped
step misses the twin — the signal whose absence let this hide. Tests assert the *invariant* (for every
reachable verify width the first rung at or above it must still be at or below the bucket),
parametrised over depths 1/2/3/5, rather than a literal list.

### Recipe buckets (`a929d56a`)

The MTP lanes picked the tightest tuned width covering the verify step. That is the wrong rule: the
bucket must leave a rung *under* itself. Since overshoot inside the bucket is nearly free, the buckets
now pick the well-**covered** width instead of the tightest one — c=4 goes 16 to 32 (depths 2 and 3,
mnbt 2064 to 2080), c=64 goes 192 to 256, c=8 keeps 32. The dry-run gate's expected map follows.

---

## Measurements

Bare-metal on one box (`emmy serve`, no docker), RTX 5090, utilization 0.96, seed 0, greedy,
ignore-eos, 8448 context, single runs, vLLM 0.23.0 + torch 2.11.0, warm cubin cache. Deltas under
about 7% are treated as noise (the team's docker sweep shows +-6% run-to-run on spec cells).

### The fix on decode-dominated load — the win

1024-token prompts and outputs, `3 x conc` prompts, bucket 32, quantum 2048, mnbt 2080, depth 2.
Same box, same commit, the only difference being the ladder. Stock measured on the same box.

| conc | before (symbolic) | after (twin) | delta | stock | before/stock | after/stock |
|--:|--:|--:|---|--:|--:|--:|
| 7 | 366.53 | **477.32** | +30.2% | 547.27 | 0.67x | **0.87x** |
| 8 | 443.14 | **591.88** | +33.6% | 723.15 | 0.61x | **0.82x** |

Per-token at c=8: TPOT 16.24 -> 11.49 against stock's 9.91 — from 64% behind to 16% behind. The fix
closes well over half the gap but does not reach the 0.95x bar.

Routing was verified rather than assumed: the new warning fires only at rung 63 during graph capture
(a rung legitimately above the bucket) and never at rung 30, so the c=8 steps did take the twin.

### The article's 4k/4k cells — no change, and why

| cell | before | after | stock | after/stock |
|---|--:|--:|--:|--:|
| c=4 d2, 4096/4096, 32 prompts | 306.80 | 303.91 | 341.09 | 0.89x |
| c=8 d2, 4096/4096, 64 prompts | 392.42 | 396.07 | 529.05 | 0.75x |

Both deltas are inside noise. The bare-metal c=8 ratio of 0.75x reproduces the article's docker ratio
(450.6/610.2 = 0.74x) almost exactly, so the bare-metal lane is measuring the same phenomenon.

**These cells are prefill-bound, not decode-bound**: emmy's mean TTFT is 24-26 *seconds* against
stock's 8.7 s, because 64 prompts of 4096 tokens against an mnbt of 2080 keeps the queue deep, so most
of the wall clock is chunked prefill, which decode-twin routing does not touch. A 30% decode win
dilutes to nothing.

That relocates the remaining 4k/4k gap: it is in prefill and chunk scheduling, not the decode twin.
The parity campaign that landed as #429 targeted exactly that surface (rider split, chunk quantum,
mnbt headroom), consistent with it having moved these cells while leaving decode untouched.

### Padded width is nearly free — why the bucket choice works

All three rows route to the same static twin; only the padded rung differs.

| conc | M | rung | tok/s | TPOT (ms) |
|--:|--:|--:|--:|--:|
| 5 | 15 | 15 | 360.47 | 11.47 |
| 6 | 18 | 30 | 428.82 | 11.57 |
| 10 | 30 | 30 | 609.52 | 11.78 |

Doubling the padded rung costs 0.9% of TPOT; adding 67% more real work at a fixed rung costs 1.8%.
The weight-bandwidth prediction holds — a decode step reads the weights once regardless of width — so
**landing on a well-tuned wider width beats landing on an untuned narrower one**. That is what makes
bucket 32 for c=4 and bucket 256 for c=64 the right calls rather than seeding new m12/m24/m192 tiers.

### c=64 depth 2 — speculation is currently self-defeating

From the team's fresh docker sweep at the protocol knobs:

| c=64, 256/256 | tok/s |
|---|--:|
| emmy, no speculation | 1109.2 |
| emmy + MTP depth 2 (bucket 192) | **332.7** |
| stock, no speculation | 1423.1 |
| stock + MTP depth 2 | 1401.2 |

Turning speculation on costs emmy 3.3x — it is not merely behind stock, it is far behind its own
non-speculative self. And stock gains nothing from MTP here either (1401.2 against 1423.1, inside
noise): at a saturated batch the GPU is already busy, so speculation adds verification work and no
useful parallelism. Speculative decoding pays when the batch is too small to fill the machine; c=64 is
the opposite regime.

Two bars follow, in order: first beat speculation-off (~1109), then match stock+MTP (~1400).

Bare-metal on the stock lane, same protocol knobs: stock+MTP **1302.80** tok/s (TPOT 19.59) and stock
without speculation **1242.98** (TPOT 36.17). So bare-metal reproduces the docker conclusion — MTP is
roughly neutral for stock at this batch (here marginally positive, there marginally negative, both
inside noise). There is no speculative win available at c=64 for either engine.

The attempt is bucket 192 -> 256, moving the step from a width with no golden records to one with a
near-full fused set.

**An untuned width is a lottery, and that is the real lesson of this cell.** The "before" side at
bucket 192 measures wildly differently depending on how its kernels were resolved:

| c=64 d2, bucket 192 (no goldens at that width) | tok/s |
|---|--:|
| published docker image (kernels baked at image build) | 332.7 |
| bare-metal, cold-resolved on the box | **10.72** |

Same code, same knobs, same width — a 31x spread, because with no golden record the fused twins fall
to a cold greedy search and what it finds is not reproducible. The bare-metal run showed mean
acceptance length 2.36 of 3 (healthy) and TPOT of 1494 ms, so this is kernel cost, not a drafter
problem. Both numbers are far below the 1109 speculation-off bar. This is exactly the misdeploy hazard
the golden file exists to prevent, and it is the strongest argument for the bucket change: m256
resolves deterministically from records, with no search and no lottery.

<!-- C64 RESULT -->

**If bucket 256 cannot clear the first bar, the honest recommendation is to run this cell with
speculation off** rather than keep tuning it — a defensible engineering answer, not a concession,
because stock demonstrates there is no speculative win available at saturated batch. Note that vLLM
0.23's `SpeculativeConfig` has no `disable_by_batch_size` field, so this is a deployment-level choice
(route saturated-batch traffic to a non-speculative deployment) or a small upstream addition, not an
existing knob.

---

## Flagged for the team: a post-#429 regression on the docker path

Not mine to fix, but it showed up while establishing baselines and it is worth more than the cells I
was sent after.

At identical protocol knobs on the c=8 d2 cell, through the docker path:

| image | tok/s |
|---|--:|
| `cloudriftai/vllm-emmy-gemma4:0.23.0-58733e02` (published, pre-merge) | 450.6 |
| `cloudriftai/vllm-emmy:0.23.0-90d010ef` (merged main) | **353.22** |

That is a **22% regression** between the published image and merged main, measured on the box that
had both. It is not the ladder (neither image uses emmy's ladder) and not bare-metal-versus-docker.
Candidates, by what they touch: **#438 Featurizer v3** — it changes codec-field propagation, which is
exactly what drives kernel picks on shapes with *no* goldens, i.e. the symbolic path these cells lean
on — and **#433 MIMO multi-output nodes**, which changes graph structure. Owner: the team.

This also corrects an inference of mine: I had attributed the 392.4-versus-450.6 gap to bare-metal
being slower than docker. It is mostly code. Bare-metal main (392.4) is in fact *faster* than docker
main (353.2) at the same knobs.

---

## What I tried that did not work, and what I got wrong

- **The ladder fix does not move the article's 4k/4k cells.** I expected it to. Those cells are
  prefill-bound, so the decode win is invisible there. Reported above rather than buried.
- **My first reading of the routing cliff was confounded.** The c=6-to-c=7 comparison changes the
  padded rung (18 to 33) *and* the program (twin to symbolic) at once, so it did not isolate which
  mattered. The rung scan on the fixed server separated them and showed padded width is nearly free,
  so the program is the cause. I initially over-claimed from the confounded version.
- **"The rule reproduces the entire published table" was wrong.** It reproduces the `emmy serve`
  cells. The docker lanes never used emmy's ladder, so applying the rule to them was invalid — the
  agreement I saw was coincidence, and finding the container's real configuration is what caught it.
- **I claimed the symbolic path is not graph-captured.** Wrong: `run_device_sym` declines only its own
  per-T capture; vLLM still captures the whole step. The penalty is kernel quality, not launches.
- **Flooring alone left padding on the table.** The first fix stopped the fatal overshoot but still
  padded 24 to 30; the dense variant lands widths exactly. Both are committed, second supersedes.
- **An unexplained anomaly, load-bearing on nothing:** symbolic at rung 63 measured about the same as
  the twin, while symbolic at rung 33 cost ~40% more per step. The `3 x conc` scans are short and
  ramp-up contaminated; only the c=7/c=8 A/B is treated as evidence.
- **Not attempted:** seeding m192/m24/m12 golden tiers. The rung scan made it clear that pointing the
  bucket at an existing well-covered width is both cheaper and better, so new tiers were not the
  right spend. A direct bucket-16-versus-32 A/B at c=4 was also not run — the recommendation to
  prefer 32 rests on the coverage matrix plus the padding-is-free result, not on its own measurement.

## Reproducing

Bare-metal, per cell (this is the path the fix affects):

```
EMMY_GEN_DECODE_BUCKET=32 EMMY_GEN_PREFILL_BUCKET=2048 \
  emmy serve google/gemma-4-12B-it --generate --dtype float16 --max-model-len 8448 \
  --no-enable-prefix-caching --gpu-memory-utilization 0.96 --max-num-batched-tokens 2080 \
  --speculative-config '{"method":"mtp","model":"google/gemma-4-12B-it-assistant","num_speculative_tokens":2}'

vllm bench serve --model google/gemma-4-12B-it --dataset-name random \
  --random-input-len 1024 --random-output-len 1024 --max-concurrency 8 --num-prompts 24 \
  --ignore-eos --temperature 0 --seed 0 --base-url http://localhost:8000
```

Add `--stock` and drop the `EMMY_*` variables for the stock lane. Swap in
`--random-input-len 4096 --random-output-len 4096 --max-concurrency 8 --num-prompts 64` for the
article's c=8 cell, and `256/256 --max-concurrency 64 --num-prompts 256` for c=64 (with
`EMMY_GEN_DECODE_BUCKET=256`, no quantum, mnbt 4096).

The docker/recipe path is `emmy bench experiments/gemma-4-12B/serving_mtp_rtx5090 --local` with the
image fields pointed at the tag under test. Note that this path does **not** exercise the ladder fix,
for the reason given under Scope.

## Recommended next steps, in order

1. **Chase the 22% post-#429 docker regression** (#438/#433). It is larger than anything else here
   and it affects the shipped artifact.
2. **Decide whether the image should run `emmy serve`.** Today it bypasses emmy's own capture
   configuration. If the answer is no, then `emmy serve`'s ladder is dev-only and should say so.
3. **Attack prefill/chunk scheduling for the 4k/4k batched cells.** That is where their remaining gap
   lives; TTFT of 24-26 s against stock's sub-second is the signal.
4. **Turn speculation off at saturated batch** (c=64 class) unless bucket 256 clears ~1109.
5. Only then consider new golden tiers; the padding-is-free result says width coverage beats width
   precision.
