# Close the decode TPOT gap (20.1 → 17.4 ms): kernel-structure research (RTX 5090, successor session)

Successor to the 2026-07-21 4K-prefill session (`plans/gemma4-4k-prefill-goldens-findings.md`, PR #410).
Starting state: emmy fm BEATS stock TTFT on both 4K/4K workloads (450 vs 545 ms c=1, 1985 vs 2449 mean
c=8); the ONLY residual vs stock is the decode step — TPOT 20.1 vs 17.4 (c=1 fm), 22.7 vs 21.0 (c=8) —
worth 0.85x/0.93x throughput. Dataset levers are EXHAUSTED, measured: every decode-M matmul golden sits
at cuBLAS bandwidth parity per kernel (down 76 µs vs eager 79, gate/up halves 76-78 vs 76-78, kv 9.5),
every serving M now takes the gate⊗up CUT, and the m32 flip's twin-level 1.15 ms compressed to 0.23 in
serving. What remains is kernel STRUCTURE, in three bounded workstreams plus one bug.

## The measured budget (fm, bucket 32, per decode step)

| component | ms | source |
| --- | --: | --- |
| twin kernels (48 layers, pre+post) | 14.85 | `profile_gen_decode.py` captured windows |
| — of which weight-streaming mains | ~13.4 | at per-kernel cuBLAS parity (irreducible at current bandwidth) |
| — of which small-kernel tail | ~1.4 | norms 3.7-3.8 µs ×2/layer, qknorms 1.2-1.6, split-K finalizes 1.1-1.9, reshapes 0.8, cut glue ~3.9 |
| lm_head (vLLM torch, tied) | ~1.2 | 2.0 GiB fp16 read |
| vLLM paged attention + sampler + graph overhead | ~2.7 | remainder to TPOT 20.1 |

Stock's 17.4 = the same weight stream at slightly higher efficiency + the tail fused into epilogues.
Bench twins from FRESH captures only (the stale `decode-twin-readiness` files realize unstaged picks —
the 2026-07-21 session lost an hour to that mirage); bench the cut-deployed global twin with
`--warmup 0` (see the harness-hang note below).

## WS1 — epilogue-fuse the small-kernel tail (~1.4 ms bound; the bounded win)

Target structures, in value order:
1. **Split-K finalize into the consumer**: every g4k/g8k decode matmul launches a 1.1-1.9 µs finalize
   (grid 256-1920, pure sum-reduce). Either emit the finalize as the next kernel's prologue (the
   consumer already reads the output) or switch decode splits to an atomic/cooperative single-kernel
   reduce. ~8-10 finalizes/layer-pair — the largest single tail item.
2. **linear→norm epilogue** (o_proj + post_attn_norm, down + post_ff_norm): today they SPLIT into
   matmul + 3.8 µs `k_mean` per site (2/layer). The norm statistic is a per-row reduce over the
   matmul's own output — an epilogue candidate the m256 research already named (WS3b there). At
   decode M=32 the matmul output is tiny (32×3840) — the epilogue costs one warp reduction.
3. **qknorm into the q/kv projections**: the per-head rms (1.2-1.6 µs ×3/layer) reads the projection
   output — same epilogue class, but the per-head reshape makes the index math messier; do (1)+(2)
   first and re-measure before deciding.
4. **Cut glue** (stat 1.5 + cone 0.9 + combine 1.5): fold the GeGLU combine into the up-half's
   epilogue (drops the 245k-thread combine kernel — also shrinks the serving graph node count that
   ate the m32 flip's win: twin 1.15 ms → serving 0.23). The stat+cone stay (the halves need the
   materialized normed A).

Method: these are `tile/_schedule` / lowering changes, not knobs — each lands with a twin re-bench
(fresh capture!), the golden audit (producer-anchor rows may need updating as kernels merge), and a
serving A/B (fresh packs — the pack bakes kernel sets). Exit gate: twin kernel time 14.85 → ≤13.8,
serving TPOT ≤19.5, audit MATCH / zero DRIFT.

## WS2 — the bandwidth residual (research; the warp-specialized computed-A)

The mains run ~1.5-1.65 TB/s where stock's GEMV-class kernels run closer to peak. The proven-out
design (see the VERDICT + harness ablations in `plans/computed-a-pipeline-and-sdpa-oproj.md`, kept
as the design record): a barrier-light producer/consumer kernel — dedicated producer warps stream B
via TMA/cp.async, consumer warps mma off mbarriers, no CTA-wide syncthreads in the K loop. CUTLASS-
class hand-write; prototype on ONE shape first (mlp_down.m32.lin, the biggest single stream at 76 µs
for 118 MB = 1.55 TB/s; floor at 1.75 TB/s ≈ 67 µs) and only generalize if the prototype clears ~10%.
The scratchpad harness from the m256 session (`scratchpad/harness/`, driver-API cubin loader) is the
right test bed. This WS is allowed to conclude "not worth it" — the per-shape ceiling is ~12%.

## WS3 — sdpa→o_proj staging (bounded; `emmy run` whole-model, NOT serving)

Unchanged from the prior plan (its 2a): the split o_proj-as-consumer falls to gmem-direct/scalar
because the transposed flash-output A can't stage — materialize the transpose or extend the A-fill
closure to strided cp.async. Serving is unaffected (vLLM owns attention; the twin o_proj reads a
clean input); this fixes the whole-model `emmy run` path and unblocks the fused sdpa→o_proj golden
kind. Do after WS1.

## WS4 — the bench-harness eager-framing hang (bug, root-cause)

Reproducer: `emmy run --ir _tune/prefill-4k/post4096-global.json --bench` with warmup ≥ 1 hangs its
FIRST kernel at ANY o_proj tile when the geglu cut is deployed; `--warmup 0` (captured), plain runs,
memcheck, and serving raw launches are all clean. Probe: raise `_KERNEL_TIMEOUT_MS`, attach cuda-gdb
during the hang, dump warp PCs — expect either a stuck mbarrier/cp wait fed by a per-launch arg
assembled differently in the eager path, or an event/stream interaction in `iter_once`'s
record/launch/wait framing. Fixing it removes the `--warmup 0` caveat from the golden file comment.

## Non-goals

- Prefill: won (fm beats stock TTFT both workloads). Taking over vLLM's paged attention is the next
  prefill lever but a separate integration track — not this session.
- lm_head / sampler / scheduler overhead: shared with stock, no relative gain.
- More golden shapes (M=128 decode etc.): only if a workload demands them; the recipe is routine now.
