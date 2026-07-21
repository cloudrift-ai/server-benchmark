# Serving optimization roadmap (post gap-closure branch, 2026-07-21)

Successor to the executed gap-closure plan; state and A/B baselines in `plans/serving-gap-closure-findings.md`.
Ordered by expected impact per unit of work.

## 1. Embedding attention: why doesn't the whole-model trace reach flash? (root-cause FIRST, then varlen)

The embedding serving gap is one kernel class: `k_sdpa_linear_reduce`, the per-cell-serial O(S²) attention — 76%
of a batched (32, 512) step; stock leads 250 vs 2.46 req/s. Before building the cu_seqlens varlen lowering,
root-cause why the existing flash tiles never fire on the whole-model embedding trace: `flash_shape_eligible`
should accept the shape (batch and heads STATIC, only seq symbolic), masked-flash `.dynM` kernels exist and serve
gemma — yet the trace deploys the non-flash softmax reduction. Suspects, in order: the whole-model trace's
explicit additive-mask input on the classification chain (`_flash.py` mask-chain walk), the GQA head fold, a
fusion-order dependence (`is_flash_score_producer` deferral not surviving the full-model graph), or the
`sdpa+linear` fused consumer shape (o_proj fused into the P@V kernel changing the offer site). Diagnose with a
1-layer trace + `EMMY_DUMP_DIR` and the `_fuse_degraded` log lines. If it is a recognizer gap, fixing it may close
most of the embedding gap without new lowering; the varlen (ragged row→sequence + cu_seqlens mask derivation)
form then removes the padding waste as step 2. Exit: batched embedding bench within 2× stock at uniform-512.

## 2. Over-bucket decode goldens (M=64/128) — cheap, compounds the WS3.3 capture win

Over-bucket decode (concurrency > 32) now rides CAPTURED symbolic programs (+10.6% req/s at c=64), but those
shapes are cold-tuned: TPOT 35.4 ms at c=64 vs 20.7 at c=32 is sub-linear-but-soft. Capture decode twins at
M=64/128 (`scripts/capture_gen_twins.py --decode-bucket 64/128`), manual `--ab` sweep per the established method
(NOT the tuner), seed goldens on both cards. Exit: c=64 TPOT measurably below 35 ms in the A/B; no c=32
regression (protocol verbatim, `EMMY_GEN_DECODE_BUCKET=32`).

## 3. Research-class kernel residuals (tracked, unblocked only by pipeline work)

- Decode TPOT 1.14× stock = the fused gate⊗up computed-A edge (170 µs, 64% of the post twin) + down_proj
  (74 µs) at M=32 — memory-stall bound, knob optimum within 4% of floor
  (`plans/computed-a-pipeline-and-sdpa-oproj.md`). Needs async multi-stage weight prefetch, not tuning.
- Mixed prefill 0.72× stock req/s = the large-M computed-A pipeline + hd512 flash cold-unreachability (hd256
  lever port + symbolic split-KV, its own session per the round-2 findings).

## 4. Workflow: pack the A/B boots + re-baseline stock

- Wire `EMMY_PACK_DIR` into the serving A/B protocol (the gen runner has pack support; boots are ~11 min of
  evidence resolution — this cuts every future iteration).
- Re-run the stock decode arm once: its 473 ms median TTFT vs its own 95 ms baseline (TPOT matched exactly)
  looks like one-off warmup; confirm before quoting emmy's decode req/s lead.

## Non-goals (measured this session, recorded in the findings)

- Batched-shape tuning for the embedding trunk before #1 lands — attention dominance makes it noise.
- WS2.1 (RoPE into the pre-trace) — premise stale; revisit only with a flash B-track consumer for in-graph RoPE.
- WS3.2 (post-norm reduction epilogue) — the candidate edges are 1.1–3.5 µs kernels, under its own <2 µs stop bar.
