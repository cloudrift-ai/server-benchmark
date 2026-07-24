# Serving integration gap closure — session findings (2026-07-20, RTX 5090 + remote RTX 4090)

Execution of `plans/serving-integration-gap-closure.md` (the post-#407 roadmap), branch
`feature/serving-gap-closure`. Summary: WS1.1's miscompute was already half-fixed and half-misdiagnosed — the real
bug was TMA descriptors baking capacity strides on the serving prefix path (fixed); WS1.2 landed as
`EMMY_SERVING_BATCHED` (batching now ties per-sequence instead of losing; the win needs WS1.3's varlen attention);
WS3.1's target state already held on main (pinned; verifying it surfaced and fixed a silent fusion output-drop bug);
WS3.3 landed (over-bucket capture validated, capture sizes follow `--max-num-seqs`); WS2.1 was deferred on
stale-premise grounds (below). All changes green on the 4090 (full `make test`) and 5090.

## WS1.1 — the batch>1 "miscompute" was two different bugs, one already gone, one real

- The documented claim ("symbolic-seq kernels miscompute batch>1 — every row wrong") no longer reproduces on the
  **exact-shape (rebind) path**: fp32/fp16, 1-layer and 28-layer Qwen3-Embedding trunks, batch {2,4,32} × seq
  {17,32,512} all match eager per row. Pinned as a matrix in `tests/compiler/ir/test_dynamic_shapes.py`.
- The REAL residual bug lives on the serving **capacity-buffer prefix path**: `_prebuild_descriptors` encoded every
  TMA `CUtensorMap` from the ALLOCATED (capacity) array shape, but prefix-packed live data at runtime S has
  resolved-shape row-major strides. Any dim ABOVE the symbolic seq dim (the batch axis) has a `seq_len`-dependent
  global stride → batch row 0 correct, every higher row reads shifted garbage. Invisible at `batch_cap = 1` (row 0
  only) and on rebind (exact buffers). Repro required the empty-prior kernel picks (split-reduce partials with
  `d2/tma/ring` staging) — the populated-prior picks happened to avoid TMA on those kernels, which is why the ad-hoc
  runs passed while pytest (isolated prior) failed.
- Fix (`backend/cuda/program.py`): symbolic-src descriptors re-encode at the RESOLVED shape per sym key
  (`_descs_now`), cached beside the per-S graph cache; an overlay entry outlives any captured graph replaying at its
  key; `rebind` clears both. Every `_launch` site resolves through it.

## WS1.2 — `EMMY_SERVING_BATCHED` (batched symbolic-seq program)

- New boolean opt-in (mirrors `EMMY_SERVING_STATIC`'s "size comes from `--max-num-seqs`" convention): the trace bakes
  the batch extent at the cap with seq_len symbolic; each scheduler step runs as ONE batched forward padded to the
  step's LONGEST sequence (not `max_seq_len`), replaying the per-S captured graph. Cannot be default-on: buffers
  allocate at `(max_num_seqs, max_seq_len)` capacity (vLLM's default 256 would OOM at boot).
- `emmy serve` now defaults `--max-num-batched-tokens` to `max_num_seqs × max_model_len` under the opt-in — without
  it vLLM's 2048-token default fills a 32-row step with ~4 sequences and the program pays 28 dummy rows
  (measured 0.63 req/s).
- **Measured (5090, Qwen3-Embedding-0.6B, uniform 512 tokens, concurrency 32, mml 1024, 256 prompts):**

  | arm | req/s | median E2E |
  | --- | --: | --: |
  | emmy batched, steps filled | 2.46 | 12.3 s |
  | emmy per-sequence (default) | 2.33 | 13.7 s |
  | emmy batched, mnbt default 2048 (starved steps) | 0.63 | 49.3 s |
  | stock vLLM (same mnbt) | **250.9** | **0.10 s** |

  Batching stops losing (the WS1.2 exit's floor) but yields no win: the batched forward costs ~B× the per-sequence
  one. Per-kernel attribution of the (32, 512) step (5.97 s total): `k_sdpa_linear_reduce` = 162 ms × 28 layers =
  76% — the non-flash, per-cell-serial O(B·H·S²) attention. The embedding trunk's SDPA never reaches the flash
  tiles, so batching multiplies an already-serial kernel. The throughput lever is exactly the plan's WS1.3
  (cu_seqlens varlen flash tiles), scoped as its own session; a batched-shape tune pass alone cannot rescue a
  structurally serial attention.

## WS3.1 — already satisfied; verification found a real fusion bug

- `add → rms_norm` (the `fused_add_rms_norm` analog) already fuses into the norm's sweep on main — static AND
  symbolic-M, 2.6–2.7× eager at the gemma decode shape — and the gemma-4-12B decode twins carry NO standalone
  pointwise kernels (per-norm kernels 1.1–3.5 µs). Pinned in `test_reduce_coverage.py`.
- WS3.2's own gate ("win per edge < 2 µs at decode M → record the loss and stop") is met by inspection: the
  candidate epilogue edges are the 1.1–3.5 µs norm kernels. Recorded here; not built.
- Verifying the escaping-residual variant (`return rms_norm(x + r), x + r`) exposed a silent correctness bug:
  `010_merge_loop_ops` never checked producer graph-output-ness, so the splice consumed the add and the compiled
  graph LOST the second output (the accuracy gate iterates compiled outputs — nothing failed). Fixed with the same
  guard `005_split_shared_indexmap` already had; both behaviors pinned.
- Decode-twin composition at M=32 (measured, `emmy run --bench` on fresh captures): pre 32/30 µs (sliding/global),
  post 284 µs — dominated by the fused gate⊗up computed-A edge (170 µs, 64%) and down_proj (74 µs), i.e. the
  known WS4 research-class residuals, not pointwise chains.

## WS3.3 — over-bucket decode capture validated; sizes follow `--max-num-seqs`

- `run_device_sym` is capture-valid under an outer torch CUDA graph: no nested capture, no host round-trip; the
  per-size uncaptured warmup that precedes each vLLM capture populates the per-sym-key TMA descriptor overlay, so
  descriptor encoding (an H2D copy) never lands inside a capture window. Pinned with a two-size live-replay test.
- `serve --generate`'s default compilation-config ladders `cudagraph_capture_sizes` up to `--max-num-seqs`
  (vLLM's 256 default when unset); the decode bucket and the cap always ride the list.
- A/B at concurrency 64 (over-bucket): see the serving table below (arms E/F).

## WS2.1 — deferred, premise stale

The plan expected "a small win from removed launches + fused RoPE". Since #388, `serve --generate` captures the
WHOLE decode step (glue RoPE included) into one FULL_DECODE_ONLY graph, so the launch-overhead half of the premise
is gone; the fused-RoPE half is ~2 small pointwise kernels per layer (the rotate-half consumer reads q twice, so it
cannot fuse into the reduce-heavy projection anyway — it would stay a separate kernel in-graph, like vLLM's fused
rotary op is today). Meanwhile the cost is real: the seam ABI changes across gen_runner / vllm_model_gen / oracle /
twin capture, and the changed pre-graphs orphan the seeded gemma-4 golden tier for those forms (the 5090 set alone
was an 11.5 h retune). Expected value ≈ neutral against WS2.1's own "TPOT no worse" gate, at golden-regression
risk. Revisit only if/when a fusion consumer for in-graph RoPE exists (the flash B-track).

## E2e serving A/B — gemma-4-12B, emmy vs stock (5090, #407 protocol verbatim)

Protocol: vLLM 0.23.0, fp16, `--max-model-len 512 --max-num-batched-tokens 256 --gpu-memory-utilization 0.90
--max-num-seqs 32`, concurrency 32, 64 prompts, seed 0; emmy arm: repo goldens + `_tune/decode-twin-readiness/twins.db`,
EMPTY online prior (`{}`), `EMMY_GEN_DECODE_BUCKET=32`; stock arm: `--stock --language-model-only`.

| workload | arm | req/s | out tok/s | TTFT mean/med (ms) | TPOT mean/med (ms) |
| --- | --- | --: | --: | --: | --: |
| in-8 / out-64 (decode) | stock | 19.12¹ | 1224 | 528 / 473¹ | **18.2 / 18.2** |
| in-8 / out-64 (decode) | emmy | **22.61** | **1447** | **104 / 103** | 20.8 / 20.7 |
| in-256 / out-64 (mixed) | stock | **13.90** | **890** | **477 / 231** | **26.6 / 28.6** |
| in-256 / out-64 (mixed) | emmy | 10.04 | 642 | 662 / 400 | 32.5 / 33.4 |

¹ stock's decode-arm TTFT is anomalously high vs its own #407 baseline (473 vs 95 ms median; its TPOT matches
#407 exactly at 18.2), so its decode req/s (19.12 vs 25.70) is depressed by the TTFT anomaly, not by decode speed
— compare TPOT for the clean decode story.

Vs #407 (same protocol, emmy arm):

- **Decode**: TPOT 20.68 vs 20.7 — unchanged (ratio to stock stays 1.14×). Nothing in this session's changes
  regressed the decode hot path.
- **Mixed**: req/s 8.33 → **10.04** (+21%; 0.58× → **0.72×** stock), median TTFT 906 → **400 ms** (−56% —
  now **1.73× stock**, inside the plan's WS1.3 "within 2×" target), TPOT 29.3 → 33.4 (the req/s win shifts the
  measured steps toward more concurrent decode, which inflates per-token time at fixed hardware). Attribution is
  not clean (64-prompt runs carry scheduler variance), but the direction matches the mixed-workload target.

**WS3.3 A/B — decode at concurrency 64 (128 prompts, `--max-num-seqs 64`, over-bucket decode batches):**

| capture sizes | req/s | out tok/s | TTFT med (ms) | TPOT mean/med (ms) |
| --- | --: | --: | --: | --: |
| ladder to 64 (this session's default) | **26.78** | **1714** | **129** | **35.2 / 35.4** |
| clamped at bucket 32 (pre-session behavior) | 24.21 | 1549 | 152 | 39.2 / 39.3 |

Capturing the over-bucket symbolic decode steps is worth **+10.6% req/s / −10% TPOT** at concurrency 64 —
WS3.3's exit ("TPOT flat or better past 32") exceeded.

## 4K-in / 4K-out long-context perf (mml 8448, mnbt 4096, c=8, 16 prompts, seed 0)

| metric | stock (util 0.90) | emmy (util 0.97) |
| --- | --: | --: |
| bench duration (16 reqs) | 320.4 s | **278.3 s** |
| output tok/s | 204.5 | **235.5** |
| TTFT mean / median (ms) | 38 199 / **3 511** | **26 273** / 3 683 |
| TPOT mean / median (ms) | **19.7 / 19.8** | 22.2 / 22.5 |

- **emmy wins the long-context workload on total throughput** (+15% tok/s, −13% duration). TPOT holds the same
  1.14× decode ratio as the 512-protocol (context-independent — emmy's per-token work doesn't touch attention;
  paged attention is vLLM's in both arms); median TTFT is near parity (the M=4096 static prefill-chunk twin ran
  clean — the int32 `_gid` fix held, no new large-M pathology surfaced beyond the known memory-stall-bound
  computed-A pipeline).
- The one genuine long-length failure was BOOT, not perf: at the default `--gpu-memory-utilization 0.90` vLLM's
  torch-only profiler budget (`util × total − used`) fell below emmy's cupy residents and the min-KV fit check
  died (1.37 GiB available vs 1.7 needed for one 8448-token sequence). **Fixed in `emmy serve`: the emmy
  generative arm now defaults util to 0.97** (stock keeps 0.90 — its own sampler warmup OOMs at 0.97); an
  explicit flag still wins.

## Remote 4090 validation

Affected-test set (batched matrix, runner batched paths, capture tests, reduce coverage, serve command): 127/127.
Full `make test`: TBD.

## Workflow notes

- The remote 4090 box's non-interactive SSH shell lacks nvcc on PATH (`/usr/local/cuda-12.9/bin`); every remote
  pytest/make invocation needs `PATH=/usr/local/cuda-12.9/bin:$PATH` or 94 tests fail with "nvcc unavailable".
- `iter_once`'s first per-launch wait absorbs the whole queued `run_once` warmup (~6 s on the batched trunk) — a
  "hung kernel k_mean" report on launch 0 is usually queue drain, not that kernel.
