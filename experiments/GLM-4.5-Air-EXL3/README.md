# GLM-4.5-Air 2.25 bpw EXL3 on one RTX 5090 — the serving comparison

Four lanes serving a 106B-A12B mixture-of-experts model on a 32 GB consumer card, benchmarked with one client under
one workload grid. emmy decodes trellis-coded weights inside its kernels; exllamav3 reads the identical checkpoint;
llama.cpp reads a different quantization at a comparable bit rate; stock vLLM cannot read the files at all.

This directory is benchmark configuration. The deployment configuration it measures is
[`recipes/GLM-4.5-Air-EXL3`](../../recipes/GLM-4.5-Air-EXL3/recipe.yaml), which holds one variant and no workload
grid — see [`recipes/ARCHITECTURE.md`](../../recipes/ARCHITECTURE.md) for why the two are separate directories.

## The pinned checkpoint

| field | value |
| --- | --- |
| repo | `turboderp/GLM-4.5-Air-exl3` (the format author's own conversion, base model `zai-org/GLM-4.5-Air`, MIT) |
| branch | `2.25bpw` — the Hessian-allocated "optimized" rung, bits 2.26, head_bits 6 |
| commit | `6a309ed6d606fc0154e6e1aeb0912cd3c25534fe` |
| size | 29.37 GiB on disk; 28.888 GiB of it becomes emmy-owned resident memory |
| quality | KL(quant→orig) 0.272, PPL 6.306 vs bf16 5.098 (ΔPPL +1.21, +23.7 %), top-1 agreement 0.817 |

The rung is a **branch, not `main`**. Every lane passes `--revision <commit>`; without it the same repo id resolves to
the 2.00 bpw rung, which is a different model with visibly worse quality (KL 0.409, ΔPPL +1.97) and which failed the
plan's quality gate. EXL3 is a single-maintainer, actively developed format: a re-cut checkpoint under the same branch
name would silently move both the quality numbers and the per-tensor bit allocation that the seeded goldens are keyed
on. Pin by commit, always.

## The Phase 0 baselines are on the WRONG RUNG — re-measure before publishing anything

`plans/vq-phase0-findings.md` §5 carries a full exllamav3/tabbyAPI serving sweep: c = 1/4/8/16, batch-1 TTFT 499 ms /
TPOT 11.9 ms, throughput saturating near 100 tok/s from c = 4 onward, plus a q4-cache lane reaching ~120 tok/s.

**Those numbers were measured on the 2.00 bpw rung, and they are not the baseline for this comparison.** The 2.25 rung
is 2.9 GiB heavier, which changes the cache ceiling (Phase 0 probed 4096 FP16 / 8192 Q4 tokens at 2.25 against
8192 / 32768 at 2.00) and therefore changes admission capacity, queueing and every throughput number that depends on
it. The `serving_exllamav3_rtx5090` lane exists to re-measure them on 2.25 with the same client. Until it has run, no
table in this directory may quote a Phase 0 figure as the contender's number; the Phase 0 sweep is a
methodology reference and a 2.00-vs-2.25 comparison point, nothing more.

## The L2-residency caveat, for anyone reading kernel-level numbers

The trellis decode work carries microsecond-level A/B numbers per matmul shape (in `plans/vq-weight-compression.md`
Phase 3, and in the seeded goldens). **Those do not predict served step time, and two of the three shapes routinely
quoted are measured L2-resident.** `run --bench --golden` replays one kernel over one weight slab about a hundred
times; the 5090 has 96 MB of L2, so any slab under that is timed out of cache at ~4.9 TB/s — a bandwidth win from
2-bit weights cannot appear, and the f16 comparison point is flattered by an amount no 30 GB model ever sees.

Concretely, of the shapes in the record: the past-L2 square (N = K = 22016, codes 121 MB) is a real DRAM measurement
and reads 4.1–4.7× ahead of its f16 twin; the gate/up and down projections are L2-resident and are relative A/B only.
Serving numbers in this directory come from end-to-end request latency and throughput, never from those benches.

## Lanes

| directory | engine | weights | why it is here |
| --- | --- | --- | --- |
| `serving_rtx5090` (lane `emmy`) | vLLM + the emmy generative plugin | the pinned EXL3 rung, decoded in-kernel | the subject |
| `serving_rtx5090` (lane `stock`) | stock vLLM 0.23.0 | the pinned EXL3 rung | expected to fail to load — the headline |
| `serving_exllamav3_rtx5090` | exllamav3 1.4.0 / tabbyAPI | **the identical checkpoint** | the primary contender |
| `serving_llamacpp_rtx5090` | llama.cpp | a GGUF at matched bit rate | the secondary contender |

exllamav3 is primary because it reads the same bytes: same weights, same quality, so the only thing left to compare is
kernels plus serving stack. llama.cpp reads a different quantization, so its row is a market comparison — what the
community actually runs at this size on this card — and any quality difference there belongs to the format, which the
article has to say rather than leave to the reader.

The exllamav3 lane runs twice, once per cache precision. emmy serves with an fp8 KV cache because nothing else fits, so
tabby's Q4 lane — not its FP16 lane — is the like-for-like row; both are recorded so the difference is visible instead
of assumed.

The stock-vLLM lane is a load-failure record, not a measurement: vLLM has no EXL3 quantization method and aborts with
`Unknown quantization method: exl3` during config parsing, before it touches a weight. The lane captures that text
verbatim. It runs at the same context and utilization as the emmy lane so that nobody can attribute the failure to a
memory or context choice.

## Workload grid and client

512 in / 128 out, and 2048 in / 128 out, each at concurrency 1 / 4 / 8 / 16 with 8 / 24 / 48 / 64 prompts — the
`c = 1/4/8/16 → N = 8/24/48/64` pattern Phase 0 used, so the short-input cells are directly comparable to the
re-measured exllamav3 baseline. Each point runs one discarded warmup plus three recorded runs (seeds 42 / 43 / 44 / 45)
with `nvidia-smi` polled at 1 Hz for power draw and peak VRAM.

Every lane issues the same client invocation, `vllm bench serve --backend openai --endpoint /v1/completions
--dataset-name random --ignore-eos …` against a single shared tokenizer (the pinned EXL3 snapshot), driven by
[`scripts/bench_serve_sweep.py`](../../scripts/bench_serve_sweep.py). The benchmark arguments are passed to that
driver verbatim after `--`, so each recipe carries the literal invocation rather than a paraphrase of it.

### The two client-side defects Phase 0 found, and where each fix now lives

Both silently corrupt results rather than failing, so both have to stay fixed or the numbers are wrong in a way no
reviewer would catch.

1. **tabbyAPI crashes on the client's `"logprobs": null`.** The vLLM client always sends the field; tabby's exllamav3
   backend compares it against an int with no `None` guard. Server-side, no client workaround exists. Fixed by
   [`scripts/patch_tabbyapi.py`](../../scripts/patch_tabbyapi.py), which the exllamav3 lane runs when it clones tabby —
   a scripted, idempotent step rather than a note in a findings file, because a hand-patched clone is exactly what
   does not survive to the next box.
2. **`vllm bench serve` wedges on CRLF-framed events and on keepalive pings.** Its parser splits on `\n\n`, so a
   CRLF-separated stream survives only via a single-JSON fallback: a coalesced read under concurrency, or
   sse-starlette's 15 s keepalive comment, wedges it permanently and the client then reports empty generations with
   fabricated TPOT. At c = 16 against tabby it lost 51 of 64 requests without an error. Fixed **client-side, once**, in
   `bench_serve_sweep.py`, which normalizes CRLF and drops SSE comment lines before the split — the defect belongs to
   the client, not to any one server, so fixing it there protects every lane including llama.cpp, whose framing has
   **not** been verified and must be treated as suspect. Phase 0 additionally patched tabby's own `EventSourceResponse`
   to emit LF; that is not repeated here, because the client fix subsumes it and a source patch to a contender's
   transport is a thing reviewers are right to distrust. `sse_ping_interval: 0` stays in tabby's config, which is a
   supported option and removes the keepalive half at the source.

The backstop for both, and for anything like them: the driver refuses to report a run whose completed-request count is
below the requested count, and exits non-zero. A silently-lost request is now a failed task.

## Reproducing

Each lane is a separate `emmy bench` invocation on a machine with the card (`--local` skips cloud provisioning). The
lanes are separate because they need mutually incompatible environments — a docker image, a private torch build, a
CUDA source build — not because they are separate experiments.

```bash
# 1. the emmy lane + the stock-vLLM load-failure record  (9 variants)
emmy bench experiments/GLM-4.5-Air-EXL3/serving_rtx5090 --local

# 2. the primary contender, both cache precisions        (16 variants)
emmy bench experiments/GLM-4.5-Air-EXL3/serving_exllamav3_rtx5090 --local

# 3. the secondary contender                             (8 variants)
emmy bench experiments/GLM-4.5-Air-EXL3/serving_llamacpp_rtx5090 --local

# a subset, e.g. only the short-input cells:
emmy bench experiments/GLM-4.5-Air-EXL3/serving_rtx5090 --local --filter "benchmark.random_input_len=512"
```

Each run writes a timestamped directory beside its recipe. Results arrive flattened as
`<variant>_<lane>_<point>_r<N>.txt`; re-tree them per lane and fold the repeats into the comparison tables:

```bash
RUNS=experiments/GLM-4.5-Air-EXL3          # the three timestamped run directories live under here
TREE=$(mktemp -d)
for lane in emmy exllamav3 exllamav3q4 llamacpp; do
  mkdir -p "$TREE/$lane"
  for f in $RUNS/*/*_${lane}_*.txt; do [ -e "$f" ] && cp "$f" "$TREE/$lane/${f##*_${lane}_}"; done
done
python scripts/aggregate_serving_lanes.py "$TREE" \
  --lanes emmy,exllamav3,exllamav3q4,llamacpp \
  --points c1_in512=512/128/c=1,c4_in512=512/128/c=4,c8_in512=512/128/c=8,c16_in512=512/128/c=16,\
c1_in2048=2048/128/c=1,c4_in2048=2048/128/c=4,c8_in2048=2048/128/c=8,c16_in2048=2048/128/c=16
```

That prints and writes `lanes.md` (output throughput, median TTFT, median TPOT — mean ± stddev over the three recorded
runs) plus a flat `lanes.json`. Power and peak VRAM ride alongside each point as `<point>_r<N>.power.json`.

## What is not settled yet

- **Every serving-shape value is an estimate.** `--max-model-len`, `--max-num-batched-tokens`, `--kv-cache-dtype`,
  `--gpu-memory-utilization` and the decode bucket are marked `TODO(Phase 5)` in
  [`recipes/GLM-4.5-Air-EXL3/recipe.yaml`](../../recipes/GLM-4.5-Air-EXL3/recipe.yaml) with the arithmetic behind each
  guess. They come out of the release workflow's headroom sweep, and the emmy lane's flags must be changed together
  with the recipe's — a test asserts the two agree.
- **The prebuilt image does not exist**, and two gaps in the release pipeline block building it: `models/<slug>.env`
  has no revision field (so a warm would bake the default 2.00 branch), and `docker/vllm-emmy-serve/serve.sh` emits
  neither the EXL3 `quantization_config: null` override nor the MoE capture-ladder cap that `emmy serve --generate`
  already applies.
- **The long-input point is provisional at 2048** and should become 4096 if the headroom sweep reaches
  `--max-model-len 8192`.
- **The GGUF is not pinned.** The llama.cpp lane names a quant class and enforces a size band; the exact file and
  revision are Phase 6's to resolve and write down.
- **The quality gate has to be re-run on the served model**, with the fp8 KV cache on. The checkpoint's KL/PPL is not
  the served model's once the cache is quantized.
