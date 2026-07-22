# Row-statistic sink (PLACE@stat) — session findings (2026-07-22, RTX 5090 + 4090)

Execution of the stat-sink workstream from the decode-TPOT research line (the ~0.5 ms/step bound the
WS2 corollary ranked as the top decode lever), branch `feature/gemma4-stat-sink`.

## What landed (the mechanism)

One general rewrite, applied per matched pair, never per named shape:

- **`RowAccum` leaf stmt** (`ir/stmt/leaves.py`): "accumulate `value` into `dst[flat / n]`". Renders as a
  hierarchical fold — warp shfl butterfly, smem stage + `__syncthreads`, ONE `atomicAdd` per block for a
  same-row run — in a **full-block** scope (`RenderCtx.full_block`, set by `Tile.render` when the tail
  guard is elided, cleared by `Cond.render`); a guarded/divergent scope falls back to the barrier-free
  warp fold, boundary warps to per-lane atomics. Registered in the stmt rewrite/simplify dispatch with
  every field (the `RegStore.atomic` silent-drop bug class has a regression test).
- **`AuxOutputOp`** (`ir/base.py`): a second produced buffer as its own graph node — no launch of its own;
  the producer's launch lists it in `outputs`/`arg_order`/`zero_outputs` (a `RowAccum` dst is detected
  exactly as `RegStore.atomic` and memset per launch), and the slab planner counts `zero_outputs` names
  as first writes.
- **`PLACE@stat` (fuse|sink)** beside cone/fold/tuple; offered by `010_recognize` on the fused norm form
  (`Map(body=[π…, sweep Loop], source=Reduction(PLANAR))` re-reading an in-graph producer's output, bound
  by `_sink.bind_sinkable_stat` — affine flat address, reduce coefficient 1, mixed-radix row coefficients,
  so `row = flat/N` is a bijection and the per-head qknorm form falls out for free). The sink siblings
  mirror EVERY local row; they are **evidence-only** (withheld from the model fallback in `greedy_decide`
  and `evidence_row_vouches`, same clauses as `PLACE@cone=cut`).
- **`025_sink_row_reduce`**: the pure realizer. Consumes B and its producer A (a two-node match via
  `match.consumed`/dict `match.output`), injects the contribution chain + `RowAccum` after A's
  complete-value store site — one write or a register-tiled write GROUP proven same-row (dense anchors
  `base..base+W-1`, `base ≡ 0 mod W`, `W | n`) — and re-emits B as an un-mapped `LoopOp` the restarted
  scan re-recognizes onto a wide 2-D grid. Waits structurally (`RuleSkipped`) for the producer to settle
  through 010/030; refuses atomic partials / mma epilogues / nested stores permanently — the picked sink
  row then deploys its own local-stat schedule (why the siblings mirror every row). Runs BEFORE 030 so a
  grid-split row on the norm can't be realized out from under the stamp.
- **`linear_norm` golden kind** (`LinearNormGoldenConfig`): the pair snippet (`F.linear` + trailing
  `F.rms_norm` [+ residual]) for seeding A/Bs; keyed at the NORM's fork (`kind="rms_norm"`), recording
  `{PLACE@stat: sink, REDUCE: <coop>}` — `emmy_us` is the fork's OWN realized sweep kernel (sorts ahead
  of the coop anchor at the same key; the pair-level verdict lives in the findings, not the row), the
  `REDUCE` constraint is the degrade anchor, and `_golden_matches_row`'s PLACE-refusal routes non-offer
  forks (input norms) to the plain anchor untouched.

## Epilogue engineering (measured, 5090, M=32 N=4096 finalize)

Per-warp atomics serialize (~128 same-row atomics: +2.5 µs, per-lane catastrophic 118 µs); the smem
block fold lands ~1 atomic/block. The residual per-site cost in eager per-kernel framing is dominated by
the per-launch 128-byte `memset_async` (~2 µs of stream/launch overhead; a CUDA-graph MEMSET node also
isn't free, ~1.3 µs isolated) — but in the pipelined e2e framing the memset overlaps and the site nets
positive. Isolated micro-benches of µs-class kernels are launch-bound past ~2-4 µs — the twins' e2e
line is the honest comparator; everything below is e2e.

## Where it pays — and where it doesn't (both cards, fm, whole-twin e2e)

| site | 5090 | 4090 | verdict |
| --- | --- | --- | --- |
| post-attn m32 (o_proj g8k finalize → k_mean) | 260.5-260.9 → 258.8-259.4; global 280.2 → 277.7 | 493.5 → 487.9; global 557.6 → 551.4 | **SEEDED** (k_mean 3.8→0.9 / 5.6→1.1) |
| qknorm ×3 (pre twin) | 30.8 → 39.6 pinned | — | LOSS — sweeps already wide post-reshape-fold |
| post_ff (needs down g4a→g4k) | 259.1/259.5 (no gain over one-site) | — | g4k's +0.9 eats the saving; down stays g4a |
| m64 | wash (±noise); global −0.7 | post64 1526.8 → 1542.1 (LOSS) | not seeded |
| input norm (`k_mean_b3bbda`) | no offer (producer = twin input) | — | correct refusal by construction |

The plan's ~0.5 ms/step family bound did NOT survive contact: the qknorm sites were already rescued by
the WS1.x reshape fold (their sweeps run wide at 1.1-1.5 µs), and the epilogue+memset floor (~1-2 µs
e2e per site) eats the small-site margins. What remains real is the post-attn site: ≈ −1.4 µs × 40
sliding + −2.5 × 8 global ≈ **−0.08 ms/step (5090)**, ≈ −5.1/−6.2 → **−0.25 ms/step (4090)**.

## Verification

- Unpinned twin verify from the seeded tier on both cards (numbers above); pre-twin untouched; the
  post_ff fork degrades to exactly the plain anchor's coop kernel (b512/b128).
- `emmy eval golden --in-model`: **MATCH 105 / DRIFT 0 / GAP 0 on BOTH cards** (was 103/103).
- Accuracy: twin runs PASS (max_diff 3.1e-5 class); snippet forms (plain / residual / grouped per-head)
  PASS vs eager. The stat is the same f32 sum reordered (warp/block tree over pre-round values).
- `make test` green (×3 through the session); new unit tests: RowAccum rewrite preservation, binding
  gates, pinned e2e (gelu producer, f4 write group), input-norm refusal.

## Serving A/B (4K-in/4K-out, c=1, 3 prompts, seed 0, fm, fresh packs, 5090)

Protocol: goldens tier + empty online (`{}`) + fresh `EMMY_PACK_DIR`, `EMMY_GEN_DECODE_BUCKET=32`,
mml 8448, mnbt 4096. The fresh-pack fm boot again exceeded the 1800 s bench health cap — the
documented no-bench warm boot (plain `emmy serve`, wait `/health`, kill) wrote the pack, and the
benched boot then rode it. The sink is IN the deployed pack: `__sq` appears in all 48 layers'
`post.decode` plans (192 refs).

| arm | out tok/s | TTFT mean/med (ms) | TPOT mean/med (ms) |
| --- | --: | --: | --: |
| stock vLLM | 57.01 | 561 / 547 | 17.41 / 17.42 |
| emmy fm (stat-sink seeded) | 49.57 | **478 / 465** | 20.06 / 20.07 |

The stock baseline reproduces the predecessor session's numbers exactly (56.97 / 563-548 / 17.42-17.41
— no drift), so the arms are comparable across sessions. Emmy keeps its TTFT lead (~15% under stock);
the decode TPOT residual is 20.06 vs the predecessor's post-session 19.83 — the m32 sink's expected
−0.08 ms/step is BELOW the boot-to-boot noise floor for this config (this boot also ran without the
predecessor's twins.db evidence tier), so the serving A/B neither confirms nor refutes the twin-level
win at c=1 on the 5090; the twins and the 4090 numbers (−0.25 ms/step class) are the evidence of
record. The remaining single-batch gap stays the computed-A / attention decode class, as before.

## Not done / follow-ups

- The **memset floor**: zeroing `__sq` inside the producer's PARTIAL kernel (launches strictly before
  the finalize) would erase the last fixed cost and might rescue the qknorm/m64 sites; requires 025 to
  also edit the partial (an mma-tier kernel) — not attempted.
- The mma `RegStore` epilogue arm (sink into an UNSPLIT matmul's store) — v2 per the design; the decode
  sites are covered by the finalize/pointwise arms.
- The bare-stat consumer (`k_mul_4__stat`, projection = `Write(stat)` only) — killing it requires
  splicing π into ITS consumers (v2).
- 4090 pre-twin qknorm sink never A/B'd (assumed loss by symmetry with the 5090).
