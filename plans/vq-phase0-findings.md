# VQ weight compression — Phase 0 findings (recon + format)

Phase 0 deliverable of plans/vq-weight-compression.md. Two parts: checkpoint/quality/baseline recon, then the
EXL3 decode-format writeup. Measured 2026-08-07 on the local RTX 5090.

# Phase 0 recon — GLM-4.5-Air-exl3 2.0bpw on RTX 5090: checkpoint, quality gate, baseline serving

Date: 2026-08-07. Machine: single RTX 5090 (32607 MiB, sm_120), 32 cores, 60 GB RAM, driver 580.173.02,
CUDA 13.0 (nvcc V13.0.88). All raw artifacts under
`/tmp/claude-1000/-home-dikobraz-Projects-emmy/744c968d-a05a-4f02-9884-7dedde866d74/scratchpad/vq/phase0-recon/`.

## Verdict up front

**The 2.00 bpw rung FAILS the Phase 0 quality gate on both arms, decisively.** KL(quant→orig) = 0.409 (gate ≳ 0.35)
and ΔPPL = +1.97 (gate ≳ +1.0; 7.071 vs bf16 5.098 on wikitext-2 test, exllamav3 `model_diff` defaults). The
**sparsity hypothesis is refuted**: Air (~12B active / 106B total, ~4× denser per token than Qwen3-Coder-Next's
~3B / 80B) degrades essentially identically in KL (0.409 vs 0.411) and *worse* in relative PPL (+38.7% vs +14.8%).
Per-token density does not rescue 2.0 bpw trellis quantization. Do not build the plan on Air-2.00.

Measured against the plan's fallbacks (details below): **no fallback passes either.** Air 2.25bpw-opt passes the
KL arm (0.272) but fails ΔPPL (+1.21) and leaves almost no KV room. The REAP-82B fallback repo has **no 2.25bpw
rung** — its lowest is 2.5bpw_H6 at 25.3 GiB — and it fails three ways: its bf16 base already scores wiki2 PPL
12.30 (vs Air bf16 5.10 — the pruning cost on general text is enormous), the 2.5bpw quant fails the gate against
its own base (KL 0.394, ΔPPL +3.96), and generation **crashes exllamav3 1.4.0 outright** ("Graph update failed"
in the batched block-sparse MoE path — 96 experts vs Air's working 128; only the graph-free eval forward runs).
The plan needs a new target or a higher bit rate.

## 1. Pinned checkpoints

| repo | branch | commit sha | size (total / safetensors) |
| --- | --- | --- | --- |
| `turboderp/GLM-4.5-Air-exl3` | `2.0bpw` | `a1adde54568f29a04c4c369180be2c17286dbec6` | 28,485,775,443 B = 26.53 GiB / 26.49 GiB |
| `turboderp/GLM-4.5-Air-exl3` | `2.25bpw` | `6a309ed6d606fc0154e6e1aeb0912cd3c25534fe` | 29.37 GiB |
| `turboderp/GLM-4.5-Air-exl3` | `4.0bpw` | `dbad5e8c1e38838612dd2ed554a30d217d62a59e` | (backup reference; unused — bf16 worked) |
| `zai-org/GLM-4.5-Air` (bf16 reference) | `main` | `a24ceef6ce4f3536971efe9b778bdaa1bab18daa` | 205.8 GiB |
| `ArtusDev/cerebras_GLM-4.5-Air-REAP-82B-A12B-EXL3` | `2.5bpw_H6` | `b04e61c0664aa77351d6bd0ccf57b4cda74082bd` | 25.27 GiB |
| `cerebras/GLM-4.5-Air-REAP-82B-A12B` (bf16 ref for REAP) | `main` | `69356056842bbee3f4e9b05acf843c60799e62c9` | 159 GiB on disk |

All live in the standard HF hub cache (`~/.cache/huggingface/hub`) and persist across sessions.
2.0bpw quantization_config: exl3 v0.0.5, bits 2.0, **head_bits 6**, calibration 100×2048, out_scales auto.
2.25bpw: bits 2.26, head_bits 6. Model config: 46 layers + 1 MTP layer, hidden 4096, 96 Q / 8 KV heads,
head_dim 128, 128 routed experts top-8 + 1 shared, vocab 151552.

## 2. Software stack (fresh venv `~/venvs/exl3`, emmy untouched)

- Python 3.12.3; torch **2.10.0+cu128** (sm_120 supported)
- exllamav3 **1.4.0+cu128.torch2.10.0** — prebuilt wheel from GitHub release v1.4.0 (source commit
  `791c83073f7f90c44f765a0ceeab7a05fa15b96b`, 2026-08-06). No JIT build needed; kernels run on sm_120.
  Source clone at `~/venvs/exl3/exllamav3-src` (tag v1.4.0) for the eval scripts.
- tabbyAPI git `d844f705aa4f18b3425a4c611dc1ff7d59e8a256` (2026-08-07), clone at `~/venvs/exl3/tabbyAPI`, with
  **two 1-line local patches** (see §5.1) and `config.yml` (model dir/name = the 2.0bpw snapshot, max_seq_len 8192,
  cache_size 8192, cache_mode FP16, `sse_ping_interval: 0`, auth disabled).
- Bench client: `vllm bench serve` from vllm **0.23.0** (emmy venv binary, used read-only).
- datasets 5.0.1 (eval clone patched: `load_dataset("wikitext", ...)` → `"Salesforce/wikitext"` — same dataset,
  new canonical id required by datasets 5.x).

## 3. VRAM occupancy and max context (2.0bpw)

Measured with `probe_load.py` (exllamav3 API; Cache alloc'd post-load, real generation per trial):

- Model load: **26,886 MiB torch-allocated** (26.3 GiB), 27,703 MiB nvidia-smi (includes CUDA context; X server
  baseline 188 MiB).
- Marginal KV cost measured ≈ **325 KiB/token** (fp16 K+V, 47 paged cache layers incl. MTP; theoretical floor
  188 KiB/token — the rest is generator/workspace overhead at first alloc).
- **Max fp16 KV cache: 8192 tokens** (10240 misses by ~20 MiB; 12288 OOMs). Validated with a real 8064-token
  prefill + 64-token generation: OK in 5.6 s end-to-end, peak 29,801 MiB.
- **Max q4/q4 quantized KV cache: 32,768 tokens** (34,816 OOMs).
- KV headroom ≈ 32,607 − 27,703 ≈ 4.9 GB — above the plan's <1.5 GB abort line, but 8K fp16 tokens total
  (shared across concurrent jobs) is thin for the concurrency story; q4 KV is the realistic serving config.

Fits of the fallback checkpoints (same probe): **Air 2.25bpw** loads at 29,750 MiB torch-alloc (30,539 smi);
max cache **4096 fp16 / 8192 q4** tokens — very tight. **REAP-82B 2.5bpw_H6** loads at 25,310 MiB (26,101 smi) —
the roomiest fit — but generation crashes (see gate outcome).

## 4. Quality (the gate) — exllamav3 `eval/model_diff.py`, NeuroSenko methodology

Method: `model_diff.py -ma <quant> -mb <reference> -r 100 -l 2048` — wikitext-2-raw-v1 test split, 100 rows ×
2048 tokens, layer-by-layer module streaming (each module loaded to GPU, forwarded over all rows, unloaded), KL
computed both directions over full vocab at the logits layer. This is exactly the tooling behind NeuroSenko's
Qwen3-Coder-Next-exl3 table (their card documents exllamav3 v0.0.22; we ran v1.4.0). **The 205.8 GiB bf16
reference WAS evaluable locally** despite 32 GB VRAM / 60 GB RAM: module streaming peaks at one bf16 MoE layer
(~4.7 GB) per model on the GPU; the whole run takes ~5 min from page-cache/NVMe. No reference substitution was
needed — numbers are directly comparable to the sibling-model table.

| model (A) vs reference (B) | KL(A,B) | KL(B,A) | PPL A | PPL B | ΔPPL | top-1 agree |
| --- | --- | --- | --- | --- | --- | --- |
| **Air 2.00bpw vs Air bf16** | **0.4090** | **0.4802** | **7.0707** | 5.0983 | **+1.97 (+38.7%)** | 0.777 |
| Air 2.25bpw-opt vs Air bf16 | 0.2716 | 0.3127 | 6.3057 | 5.0983 | +1.21 (+23.7%) | 0.817 |
| REAP-82B 2.5bpw_H6 vs REAP bf16 | 0.3939 | 0.3847 | 16.2570 | 12.3013 | +3.96 (+32.2%) | 0.799 |

Anchor (NeuroSenko, Qwen3-Coder-Next 80B-A3B, exl3 v0.0.22): 2.0bpw KL 0.4111 / PPL 8.851 vs orig 7.708
(ΔPPL +1.14, +14.8%).

**Gate outcome:**

- **Air 2.00bpw: FAIL** — both arms (KL 0.409 ≳ 0.35; ΔPPL +1.97 ≳ +1.0). Per-token KL median is only 0.132
  (the mean is dominated by low-confidence tokens), but the gate metric is the mean and it fails cleanly.
- **Air 2.25bpw-opt: MARGINAL FAIL** — KL arm passes (0.272 < 0.35), ΔPPL fails (+1.21 > +1.0). The optimized
  (Hessian-allocated) rung recovers a lot (KL −33%, ΔPPL −0.76 vs 2.00) but weighs 29.4 GiB → ~2 GB total KV
  headroom, q4-KV-only serving.
- **REAP-82B 2.5bpw_H6: FAIL** — KL 0.394 ≳ 0.35, ΔPPL +3.96 vs its own (already degraded) bf16 base, and it
  cannot generate under exllamav3 1.4.0 at all (CUDA graph crash above). Not servable, not quality-viable.
- **Sparsity hypothesis: REFUTED.** At 2.0 bpw, KL is a dead match to the 4×-sparser sibling (0.409 vs 0.411)
  and relative PPL degradation is ~2.6× worse. Whatever governs 2-bit degradation here, per-token active-parameter
  density is not the lever the plan hoped for. The article cannot claim "denser MoE quantizes better at 2 bit".

## 5. Baseline serving (tabbyAPI + exllamav3, identical 2.0bpw checkpoint)

Server: tabbyAPI as in §2, FP16 KV, cache_size = max_seq_len = 8192, one 5090. Client (**Phase 6 must reuse
verbatim**, modulo `--max-concurrency/--num-prompts/--seed/--result-filename`):

```
/home/dikobraz/Projects/emmy/venv/bin/vllm bench serve --backend openai \
  --base-url http://127.0.0.1:5000 --endpoint /v1/completions \
  --model a1adde54568f29a04c4c369180be2c17286dbec6 \
  --tokenizer /home/dikobraz/.cache/huggingface/hub/models--turboderp--GLM-4.5-Air-exl3/snapshots/a1adde54568f29a04c4c369180be2c17286dbec6 \
  --dataset-name random --random-input-len 512 --random-output-len 128 --ignore-eos \
  --num-prompts <N> --max-concurrency <C> --seed <42+rep> \
  --save-result --result-dir <dir> --result-filename result_c<C>_r<rep>.json
```

N = 8/24/48/64 for C = 1/4/8/16; one discarded warmup run first; two recorded runs per point; nvidia-smi polled
at 1 Hz per point (power.draw, memory.used). Note: tabby's exllamav3 backend ignores `ignore_eos`
(logs "Ignoring sampler params ... ban_eos_token"); with random-token prompts outputs still hit the 128-token cap
(~127 generated) in practice, so the point is moot at these lengths.

| c | run | compl | dur s | req/s | out tok/s | mean TTFT ms | med TTFT ms | mean TPOT ms | p99 TPOT ms | power mean/max W | peak VRAM MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | r1 | 8/8 | 16.0 | 0.50 | 63.8 | 500 | 492 | 11.84 | 12.01 | 213/303 | 32037* |
| 1 | r2 | 8/8 | 15.3 | 0.52 | 62.7 | 497 | 493 | 11.91 | 12.04 | 238/348 | 29061 |
| 4 | r1 | 24/24 | 29.5 | 0.81 | 104.0 | 1291 | 660 | 27.58 | 30.56 | 281/391 | 29061 |
| 4 | r2 | 24/24 | 30.0 | 0.80 | 100.8 | 1341 | 805 | 28.17 | 33.96 | 286/399 | 29061 |
| 8 | r1 | 48/48 | 60.4 | 0.79 | 99.6 | 5993 | 7122 | 29.02 | 32.73 | 306/394 | 29061 |
| 8 | r2 | 48/48 | 60.8 | 0.79 | 98.5 | 5986 | 5939 | 29.46 | 34.89 | 305/394 | 29061 |
| 16 | r1 | 64/64 | 80.8 | 0.79 | 98.2 | 14562 | 15980 | 30.77 | 58.47 | 305/362 | 29061 |
| 16 | r2 | 64/64 | 81.7 | 0.78 | 99.7 | 14753 | 15396 | 29.13 | 30.56 | 320/404 | 29061 |

*32037 MiB includes the model-load spike in r1's polling window; steady-state peak is 29,061 MiB.

Means over the two runs: c1 TTFT 499 ms / TPOT 11.9 ms (≈84 tok/s decode); c4 TTFT 1.32 s / TPOT 27.9 ms;
c8 TTFT 5.99 s / TPOT 29.2 ms; c16 TTFT 14.7 s / TPOT 29.9 ms. Spread between runs is small (≤3% on TPOT,
≤8% on TTFT at c≥4).

**Reading**: output throughput saturates at ~100 tok/s already at c = 4 and stays flat through c = 16 — added
concurrency converts 1:1 into queue time (server logs show 14–17 s queue per request at c16; server-side prefill
runs ~1200 tok/s but is serialized against decode). exllamav3/tabbyAPI accepts c = 16 without failures, but its
effective parallelism is ~4–5 on this cache. This is the continuous-batching opening the plan predicts: TTFT under
load, admission capacity, and per-request isolation are the fight, not batch-1 TPOT (11.9 ms is strong).

### 5.0 q4-KV lane (bonus): cache_mode Q4, cache_size 16384

Same server and client otherwise (tabby refuses cache_size 32768/24576 q4 with "Insufficient VRAM in split" even
though raw allocation fits — its loader reserve is conservative; 16384 loads). c8/c16 only, 2 runs each:

| c | run | compl | dur s | out tok/s | mean TTFT ms | med TTFT ms | mean TPOT ms | p99 TPOT ms | power mean/max W | peak VRAM MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | r1 | 48/48 | 49.7 | 118.4 | 2781 | 2176 | 43.20 | 57.62 | 320/497 | 28903 |
| 8 | r2 | 48/48 | 49.7 | 121.2 | 2740 | 2174 | 42.15 | 48.10 | 321/477 | 28381 |
| 16 | r1 | 64/64 | 65.8 | 122.1 | 9910 | 9783 | 41.32 | 49.00 | 346/494 | 29035 |
| 16 | r2 | 64/64 | 65.1 | 120.2 | 9784 | 9495 | 41.36 | 50.15 | 342/494 | 28395 |

Doubling usable cache (16K q4 vs 8K fp16) lifts saturation throughput ~100 → ~120 tok/s and cuts c8 TTFT
6.0 s → 2.8 s (more admitted jobs), at higher per-stream TPOT (29 → 42 ms). c16 still queues ~10 s — the
ceiling is the engine's prefill/decode serialization, not just cache capacity.

### 5.1 Serving-stack bugs hit on the way (recorded; both patched in our clone)

1. **tabbyAPI + `"logprobs": null`**: the vllm client always sends `logprobs: null`; tabby's exllamav3 backend
   crashes (`NoneType > int`) at `backends/exllamav3/model.py:1371`. Patched to `if params.logprobs and ...`.
2. **SSE framing vs the vllm client**: tabby (sse-starlette) separates SSE events with `\r\n\r\n` while
   `vllm bench serve`'s parser splits on `\n\n` and additionally `strip()`s each socket read, so it survives only
   in the "one complete event per read" regime via a single-JSON fallback. Two consequences: (a) coalesced reads
   under load wedge the parser permanently — at c16 this silently failed 51/64 requests; (b) sse-starlette's
   15 s keepalive ping (`: ping - ...`) lands whenever queueing >15 s and wedges the fallback the same way —
   after which the client reports empty generations and garbage TPOT. Fixes: `sep="\n"` on tabby's OAI
   `EventSourceResponse`s (2 call sites) and `sse_ping_interval: 0` in config (a supported tabby option).
   Both are transport-formatting only — no effect on model execution or timing. The same client wedge would bite
   ANY server emitting CRLF SSE or pings; emmy's vLLM-based stack emits LF and no pings, so Phase 6 is safe, but
   this must be kept in mind when comparing against other engines (llama.cpp lane).
   The two broken sweep result sets are preserved (`serve-bench-broken-sse/`, `serve-bench-ping-wedge/`) as
   evidence; the good set is `serve-bench/`.

## 6. Artifacts

- `probe_load.py`, `probe_fp16.log`, `probe_q4.log`, `probe_reap25_*.log`, `probe_air225_*.log` — load/context probes
- `model_diff_2.0bpw_vs_bf16.log`, `model_diff_2.25bpw_vs_bf16.log`, `model_diff_reap25_vs_reapbf16.log` — quality
- `serve-bench-q4/` — q4-KV lane results; `sweep_q4.sh`, `tabby-q4-32768.log`
- `reap_canonical.py` — minimal REAP generation crash repro
- `sweep.sh`, `sweep.log`, `serve-bench/` (result_c{1,4,8,16}_r{1,2}.json + smi_*.csv + client_*.log) — serving
- `tabby-fp16-8192.log` — server log; `client_tap.py`, `aiohttp_c16_repro.py`, `repro*.py` — SSE-bug diagnosis
- `download-*.log`, `tree-2.0bpw.json` — checkpoint pinning

---

# EXL3 weight format — decode reference (Phase 0 writeup)

Sources pinned: `turboderp-org/exllamav3` @ `791c83073f7f90c44f765a0ceeab7a05fa15b96b` (= release tag **v1.4.0**),
cloned at `scratchpad/vq/exllamav3`. All file references below are relative to that tree. Checkpoint validated
against: `turboderp/GLM-4.5-Air-exl3`, branch **`2.0bpw`** → commit **`a1adde54568f29a04c4c369180be2c17286dbec6`**
(28,443,406,032 bytes = 26.49 GiB of safetensors across 4 shards; quantizer version `0.0.5` per
`quantization_config.json`). The hub cache at `~/.cache/huggingface/hub/models--turboderp--GLM-4.5-Air-exl3`
already holds the full snapshot — reuse it.

EXL3 is QTIP's bitshift ("tail-biting") trellis quantization with a procedural ("3INST") codebook and QuIP#-style
Hadamard incoherence processing, repackaged: per-tensor sibling safetensors, a fixed 16x16 weight tile equal to the
mma tile, K = integer bits/weight per tensor, and per-channel sign/scale vectors stored as plain fp16.

## 1. On-disk layout of one EXL3 linear

For a logical weight W of shape (in_features = k, out_features = n) — note exllamav3 stores/consumes weights in
**(in, out)** orientation, i.e. y = x @ W — a quantized module `<key>` ships these sibling tensors
(`LinearEXL3.get_tensors` / `Linear.load_exl3` in `exllamav3/modules/quant/exl3.py`, `exllamav3/modules/linear.py`):

| tensor | dtype | shape | meaning |
| --- | --- | --- | --- |
| `<key>.trellis` | int16 | `(k/16, n/16, 16*K)` | packed codes, one row per 16x16 tile; `K = shape[-1] // 16` bits/weight |
| `<key>.suh` | fp16 | `(k,)` | input-side per-channel multiplier (random signs x per-channel scale, see §4) |
| `<key>.svh` | fp16 | `(n,)` | output-side per-channel multiplier (random signs, x optional out-channel scales) |
| `<key>.bias` | fp16 | `(n,)` | optional plain bias (GLM-4.5-Air has it on q/k/v_proj) |
| `<key>.mcg` | int32 | scalar | optional marker: codebook cb=1; stored value = 0xCBAC1FED, never read — presence is the flag |
| `<key>.mul1` | int32 | scalar | optional marker: codebook cb=2; stored value = 0x83DCD12D, presence is the flag |
| `<key>.su`/`.sv` | int16 | `(k/16,)`/`(n/16,)` | legacy packed sign bitfields (old checkpoints only; unpacked to ±1 via `unpack_bf`) |

There is **no separate scale tensor** ("scale is no longer used", `LinearEXL3.__init__`): the global scale, the
codebook scale and all per-channel scales are folded into `suh`/`svh` at quant time. K is discovered purely from
`trellis.shape[-1] // 16`; nothing else parametrizes the decode except the cb marker tensors.

Both dims are always padded to multiples of 128 at quant time (`Linear.pad_to = 128`), so k % 128 == n % 128 == 0.

Tile grid: `trellis[kt, nt, :]` encodes the 16x16 block `W_hat[16*kt : 16*kt+16, 16*nt : 16*nt+16]` (rows = in
features, cols = out features) — see the output indexing of `reconstruct_kernel` in
`exllamav3/exllamav3_ext/quant/reconstruct.cu`.

`quantization_config.json` at checkpoint root duplicates all of this per module (`tensor_storage[key]`:
`quant_format: "exl3"`, `bits_per_weight`, `stored_tensors` with dtype/shape, `mcg_multiplier`/`mul1_multiplier`
when present) — written by `exllamav3/conversion/quant_config.py:create_quantization_config_json`. Top-level keys:
`quant_method: exl3`, `bits` (decoder target bpw), `head_bits`, `calibration {rows, cols}`, `out_scales: auto`.

## 2. Trellis bit-stream and windows (the decode core)

Reference: `exl3_dq.cuh` (`dq`/`dq2`/`dq4`/`dq8*`), `pack.cu` (`pack_trellis_kernel`, `unpack_trellis_kernel`).

Each tile is one self-contained tail-biting trellis walk over its 256 weights. The packed tile is a circular
bit-stream of `256*K` bits stored in `16*K` int16 words. **Stream order**: view consecutive int16 pairs as
little-endian uint32s (`u32[i] = u16[2i] | u16[2i+1] << 16` — note this means the odd int16 of each pair holds the
*earlier* bits, an artifact of the SWAP16 store in `pack_trellis_kernel`); within each uint32, bits are MSB-first.
So global stream bit j lives at bit `31 - (j % 32)` of `u32[j / 32]`.

The trellis is QTIP's bitshift trellis with L = 16 (state width 16 bits, i.e. 2^(16-K) states), step K bits:

- **window(t)** for weight step t in [0, 256): the 16 stream bits `[(t+1)*K - 16, (t+1)*K) mod 256*K`, MSB-first.
  Equivalently `window(t) = ((window(t-1) << K) | fresh_bits(t)) & 0xFFFF`, where `fresh_bits(t)` = stream bits
  `[t*K, (t+1)*K)`. The `mod` gives tail-biting: step 0's window borrows the *last* 16-K bits of the stream.
- **value(t)** = `decode_3inst<cb>(window(t))` — a deterministic fp16 (§3). No LUT is stored anywhere.

Structural invariant (used as a CPU self-check; holds on real checkpoint data): for all t, circular,
`window(t) >> K == window(t-1) mod 2^(16-K)`.

Packing (what the quantizer writes, `pack_trellis_kernel`): concatenate `window(t) & ((1<<K)-1)` for t = 0..255,
MSB-first, into the 256*K-bit stream; emit as uint16 words; store each uint32-aligned pair half-swapped (SWAP16).
The numpy prototype's `pack_windows` reproduces the stored bytes exactly (validated, §7).

## 3. The 3INST procedural codebook

Reference: `decode_3inst<cb>` in `exllamav3/exllamav3_ext/quant/codebook.cuh`. Input: 16-bit window x (as uint32).
All integer arithmetic mod 2^32. `lop3 ... 0x6a` is `(x & 0x8FFF8FFF) ^ 0x3B603B60`.

- **cb = 0** (default; the *only* codebook in the GLM-4.5-Air 2.0bpw rung):
  `x = x * 89226354 + 64248484; x = (x & 0x8FFF8FFF) ^ 0x3B603B60;`
  `value = fp16(x & 0xFFFF) + fp16(x >> 16)` — reinterpret each half as IEEE fp16, one fp16 add (RNE). The mask
  keeps sign + low mantissa bits of two pseudo-random halves, the XOR pins their exponents; the sum of the two is
  approximately Gaussian. Full-codebook stats over all 65536 windows: mean ~0, **std = 1.2437 = `codebook_scale`
  (1.24371088, `quantize.py` line 16)**, range ±3.97.
- **cb = 1** (`mcg` marker): `x = x * 0xCBAC1FED;` then the same mask/XOR/half-sum (no additive constant).
- **cb = 2** (`mul1` marker): `x = x * 0x83DCD12D;` `s = bytesum(x) + 0x6400` (dp4a with accumulator 0x6400);
  `h = fp16_bits(s & 0xFFFF)`; `value = fma_fp16(h, 0x1EEE, 0xC931)` (constants are fp16 bit patterns:
  ~1/147.7 and ~-10.39).

The codebook is exact integer + fp16 arithmetic — bit-exact reproducibility on CPU is trivial. K only changes how
many fresh bits feed the window; the value map is always 16-bit-window → fp16.

## 4. Tile element order (tensor-core layout)

Reference: `tensor_core_perm` in `exllamav3/modules/quant/exl3_lib/quantize.py` (the quantizer's flattening);
inverse shuffle in `reconstruct_kernel`. The 256 trellis steps within a tile are NOT row-major: they follow the
mma m16n8k16 B-fragment layout (32 lanes x 8 values). For step index `e = 8*lane + j` (lane in [0,32), j in [0,8)):

```
row = 2*(lane % 4) + (j & 1) + 8*((j >> 1) & 1)      # row within tile = in-feature offset
col = (lane // 4) + 8*(j >> 2)                        # col within tile = out-feature offset
```

(rows hit {0,1,8,9} + 2*(lane%4), cols hit lane//4 and lane//4 + 8.) A GPU decode consuming mma fragments needs no
shuffle at all — lane l decodes its own 8 steps `t = 8l..8l+7` and holds exactly its B-fragment registers; that is
the entire point of the ordering.

## 5. Hadamard / sign transforms, and what `W_hat` means

Reference: `regularize` + `quantize_exl3` in `quantize.py`, `LinearEXL3.get_weight_tensor`,
`reconstruct_had_kernel` in `reconstruct.cu`, `test_reconstruct_had.py`.

Let `W_hat` = the raw trellis decode above, shape (k, n). The original-basis weight is

```
W = diag(suh) . H128 . W_hat . H128 . diag(svh)
```

where `H128` is the **block-diagonal natural-order Sylvester Hadamard of size 128** applied per 128-block along
each dim, scaled `1/sqrt(128)` per side (`r_scale = 0.08838834764831845`). Both dims are 128-divisible by
construction, so this is always well-formed. There is no stored Hadamard matrix and no per-checkpoint sign masks
beyond `suh`/`svh` themselves; `had_k = had_n = 128` are compile-time constants.

`suh` content: random ±1 signs (seeded) x per-row RMS scale x `-1/codebook_scale` x `1/g_scale` (the negative
global sign is absorbed here — do not "fix" it). `svh`: random ±1 signs, optionally x out-channel scales
(`out_scales: auto`). Measured on the checkpoint: `suh` rms ~0.01-0.014 (carries the magnitude), `svh` rms ~1.0
(nearly pure signs). Everything needed for exact reconstruction is in the two vectors.

**At inference** the transform can sit on either side (mathematically identical):

- Weight-side (fold into decode): what `reconstruct_had_kernel` does — fully foldable, no activation work.
  This is the natural choice for emmy's bind-time (Phase 2) decode. Granularity caveat: the fold needs full
  128x128 blocks (8x8 tiles), not single 16x16 tiles.
- Activation-side (what the gemm/gemv kernels do): `x_had = H128(x * suh)` per 128-block of the input
  (`had_r_128` / `had_hf_r_128_inner`, fused as a pre-stage inside the same cooperative kernel launch), gemm
  against `W_hat`, then `y = H128(y_acc) * svh` per 128-block of the output (fused epilogue; in the MoE path the
  routing weight multiplies into the same epilogue scale). A/Bs in `LinearEXL3.reconstruct_hgemm` put the
  standalone had launches at ~14% of long-chunk prefill time — motivation for folding.

## 6. Per-tensor bpw, head/embeddings, MoE, kernels

**Bit allocation.** K is integer per tensor, stored implicitly in the trellis shape. Uniform rungs (like this
2.00 bpw checkpoint): every budgeted linear at `floor(bpw)` with leftover budget spent +1 bit at a time by group
priority and distance-to-ends (`conversion/allocation.py:create_q_strategy`); "optimized" rungs replace this with
a measured KLD-vs-cost solve (`conversion/optimize_model.py`). Either way the decoder needs nothing beyond
`trellis.shape`. GLM-4.5-Air 2.0bpw measured: **17601 tensors at K=2 (cb0), lm_head alone at K=6**, all cb0.

**lm_head** is EXL3-quantized at `head_bits` (default 6; here 6): `lm_head.{suh,svh,trellis}`, in=4096,
out=151552, trellis (256, 9472, 96). It is the single biggest tensor: 4096x151552 at 6 bpw ~ 466 MB.
**Embeddings are NOT quantized**: `model.embed_tokens.weight` fp16 (151552, 4096), ~1.24 GB. Also unquantized:
all norms, and each MoE **router** (`model.layers.L.mlp.gate.weight` fp16 (128, 4096) +
`gate.e_score_correction_bias` fp16 (128,)).

**MoE layout (GLM-4.5-Air)**: strictly per-expert sibling tensors, no fusion in the checkpoint —
`model.layers.L.mlp.experts.E.{up,gate,down}_proj.{suh,svh,trellis}` for E in 0..127, plus
`shared_experts.{up,gate,down}_proj.*` and the fp16 router (`architecture/glm4_moe.py`). Layers 0..(first_k_dense_replace-1)
= dense `mlp.{up,gate,down}_proj`. Every expert has its own suh/svh (own random signs) — expert tensors cannot
share an activation-side Hadamard input; the exllamav3 MoE kernel (`exl3_mgemm_kernel`, `exl3_gemm_kernel.cuh`)
re-runs the input-had per expert with that expert's `suh_list[mat_index]` into a per-expert `A_had` slot, then
applies `svh_list[mat_index]` x routing weight in the epilogue, and finally reduces across experts.

**GEMM kernel staging** (`exl3_gemm_kernel.cuh` + `exl3_gemm_inner.cuh`) — Marlin-style, informs emmy's
computed-B fill: persistent cooperative grid slicing the (tiles_k x tiles_n) space; per iteration `cp.async`
copies the A tile (fp16, XOR-swizzled) **and the packed B codes verbatim (uint16, no decode)** gmem→smem through
`SH_STAGES` ring buffers; at fragment-load time each lane reads two uint32s of its tile's code words from smem and
runs `dq_dispatch` (funnel-shift window extraction + 3INST) straight into B-fragment registers — i.e. **decode
happens at the smem→register drain, between cp.async and mma**, amortized once per 16x16 tile per lane (8 values);
then `ptx_mma_m16n8k16`. K-splits reduce via lock-guarded gmem atomics; TILESIZE_M is fixed at 16 and larger M
loops the whole inner kernel per 16-row slab. fp16 accumulate on sm_86 only (`EXL3_GEMM_H_ACC`).

**GEMV kernel** (`exl3_gemv_kernel.cuh`, m <= 8, K in {2,3,4}) — informs emmy's reduce-tier decode: warps split k
with **no block-level pipeline barriers**; B codes stream gmem→registers with `ld.global.cs` (evict-first) behind a
register prefetch ring (depth 4/2); window extraction fully in registers per lane (`dq8_regs_*`); one m16n8k16 mma
pair per tile with the A operand built from broadcast activation halves, fp16 accumulation folded to fp32 on a
fixed cadence; cross-warp k-reduction through smem; input/output Hadamard stages fused in the same cooperative
launch with one grid.sync each side. An int8 variant (`exl3_gemv_int8.cu`) quantizes activations on the fly.

`AUTO_RECONSTRUCT_THRESHOLD = 144` rows (`exl3.py`): above that, exllamav3 dequantizes the full W to fp16
(`reconstruct` / fused `reconstruct_had_slice`) and runs a plain hgemm — i.e. even the reference stack treats
prefill as "decode-then-gemm"; only decode-phase shapes ride the fused trellis kernels.

## 7. Recipe for a numpy decoder (validated prototype: `decode_prototype.py` alongside this file)

Inputs per linear: `trellis` int16 `(kt, nt, 16K)`, `suh` fp16 `(16*kt,)`, `svh` fp16 `(16*nt,)`; `cb` = 1 if an
`mcg` sibling exists, 2 if `mul1` exists, else 0.

```
K = trellis.shape[-1] // 16
for each tile (kt_i, nt_j):                                  # independent, parallel
    u16   = trellis[kt_i, nt_j].view(uint16)                 # 16K words
    u32[i] = u16[2i] | u16[2i+1] << 16                       # little-endian pair join, i in [0, 8K)
    bit[j] = (u32[j//32] >> (31 - j%32)) & 1                 # stream, j in [0, 256K)
    win[t] = bits [(t+1)K-16, (t+1)K) mod 256K, MSB-first    # t in [0, 256), tail-biting
    val[t] = decode_3inst_cb(win[t])                         # §3, exact uint32+fp16 arithmetic
    tile16x16[row(e), col(e)] = val[e]                       # §4 permutation, e = 8*lane + j
    W_hat[16*kt_i:+16, 16*nt_j:+16] = tile16x16
# original basis (fold once, or keep activation-side):
W = diag(suh) @ H128_blockdiag @ W_hat @ H128_blockdiag @ diag(svh)   # 1/sqrt(128) per side
y = x @ W  (+ bias)
```

**Validation status (CPU-only, per Phase 0 constraints):**

- Overlap invariant `win[t] >> K == win[t-1] mod 2^(16-K)` holds circularly on 64 random tiles per tensor across
  three real GLM-4.5-Air tensors (dense up_proj, expert down_proj, q_proj). This pins window alignment and
  endianness against real data.
- **Byte-exact repack roundtrip**: re-packing the extracted windows' fresh bits through the §2 pack algorithm
  reproduces the stored trellis bytes exactly (same 64x3 tiles). Together with the invariant this pins the
  placement of every bit.
- Distribution: decoded `W_hat` rms 1.14-1.29 ~ codebook_scale; full-codebook std exactly 1.2437 = the quantizer's
  `codebook_scale` constant; folded W rms ~0.013-0.016 (plausible fp16 weight scale); `suh` carries magnitude
  (rms ~0.01), `svh` ~pure signs (rms ~1.0).
- **Bit-exact comparison against `ext.reconstruct` is deferred to Phase 1** — that kernel is CUDA-only and the GPU
  is owned by a sibling agent. The bit-exact bar applies to `W_hat` (integer windows + exact fp16 codebook); the
  Hadamard/sign fold is mathematically exact but its fp16 rounding is implementation-defined (exllamav3's own
  fused-vs-reference test tolerates 2e-3 relative).

## 8. Notes for emmy integration (observations only, no design here)

- The B-side compute fill maps 1:1: codes ride the async prefetch ring untouched (uint16 slabs, 64 bytes/tile at
  K=2), decode sits at the ldmatrix-drain slot, and the §4 ordering means a lane's decode output IS its mma
  B fragment. The activation-side Hadamard (`x*suh` then per-128 butterfly) is the computed-A prologue; the
  output-side one is an epilogue on 128-column blocks; or both fold into bind-time dequant (Phase 2) at
  128x128-block granularity.
- Per-expert `suh` means the MoE input-had cannot be hoisted above routing — exllamav3 pays it per expert.
- lm_head at K=6 uses the same machinery (dq8 generic path, no aligned fast path); embeddings load as plain fp16.
- exllamav3's `unpack_trellis` kernel emits, per step, the *two* words `(win[t] , win[t] >> K)` — a handy GPU
  fixture for Phase 1 window-level (pre-codebook) A/B if needed.

## 9. Phase-2 corrections (2026-08-07, verified against the pinned snapshot)

- **Not every linear is trellis-coded.** §6's "17601 tensors at K=2, lm_head at K=6" undercounts the exceptions:
  the quantizer keeps sensitivity-selected linears at plain fp16 `.weight` — on this checkpoint exactly ONE such
  linear exists, `model.layers.0.self_attn.o_proj.weight` (fp16 (4096, 12288); every other layer's o_proj is
  trellis-coded). Ingestion must treat "has a `.trellis` sibling" as the per-module quantization test, never the
  config alone. (The fp16-kept set also includes the already-documented embeddings, norms, and MoE routers.)
- **Padding is real and observable.** §1's "both dims padded to multiples of 128" bites on this model:
  `intermediate_size` 10944 stores as 11008 (dense-layer gate/up out-dim and down in-dim; suh/svh/trellis all at
  the padded extent). The logical weight is the top-left submatrix of the padded decode — slicing is exactly
  exllamav3's math (zero-padded activations in, sliced outputs).
- **The index also carries an MTP layer** (`model.layers.46.*`, `num_nextn_predict_layers: 1`) that
  `Glm4MoeForCausalLM` does not instantiate — loaders must tolerate those unexpected keys.
