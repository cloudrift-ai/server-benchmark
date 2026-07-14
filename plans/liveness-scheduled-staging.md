# Liveness-scheduled operand staging — unify `staged_kloop` / `alternating_kloop`

Generalize the alternating single-slab pipeline (PR #358's `STAGE=d1/…/alt`) from a flash-shaped skeleton into a
liveness-driven scheduler over the streaming loop's dataflow graph, so ONE skeleton derives today's paired ring, the
alternating form, and the staged-Q prologue — with no per-kernel structure baked into `_twist` or `_stage`.

## Motivation (measured, 2026-07-14, PR #358 arc)

- The alternating form is the hd256 frontier on BOTH cards (5090 fm `d1/cp/alt` 29.7 µs = 1.03× vs torch SDPA; 4090
  33.8 = 1.21×) and the hd128 frontier on tma — but it is hand-assembled: `alternating_kloop` hardcodes the
  QK | softmax | PV phase split, and `_twist._stream_segments` exists only to feed it. That is a shape-adjacent
  skeleton in a codebase whose mandate is a purely algebraic moveset.
- The choice is genuinely perf-forked, not derivable: hd64 prefers the paired ring (alt 9.2 vs 8.6 — the extra sync
  outweighs when slabs already fit), hd128/hd256 prefer alt, and the transport flips per shape (hd128→tma,
  hd256→cp). Placement must be DERIVED; depth/transport must stay searchable rows.

## The abstraction

One streaming-loop iteration is already a dataflow graph in the tile IR: `Reduction.partial` is an ordered node
sequence (`Contraction(Q·K) | scale/mask/merge stmts | Contraction(P·V)`), and every staged operand names the exact
node(s) that read it. The staging schedule needs two facts per operand, both structural:

1. **Live range** — `[first reader, last reader]` in the body's node order (K = the head contraction only; V = the
   PV contraction only; a matmul's A and B = the whole chunk).
2. **Value delta** — how the slab's contents advance per iteration (K/V: δ=1; Q: δ=0, loop-invariant).

Everything the alt skeleton hand-codes becomes a derived consequence:

- δ=0 → fill ONCE in the prologue, no refill, no per-iter wait. Q-in-smem is this degenerate case (and the
  register-vs-smem residency of a δ=0 operand is the occupancy trade — the emitter piece already exists).
- δ=1 with a proper sub-interval live range → depth-1 slab, refill placed at the KILL POINT (`Sync` after the last
  reader, then fill; wait before the first reader). K refills under softmax+PV, V under the next QK — not because
  attention alternates, but because that is where their live ranges end.
- δ=1 spanning the whole body → the depth-2 ring, exactly today's `staged_kloop`. The matmul tier degenerates
  correctly with no special-casing in either direction.

Synchronization is derived too: TMA = one mbarrier per operand, parity per generation; cp.async `wait_group(N)` =
**the count of younger fills issued between this operand's fill and its wait point** — a static counting pass over
the placed schedule (the alternating K,V,K,V queue's uniform `wait_group(1)` was this count done by hand).

What stays a KNOB (evidence above): per-operand depth (1-with-inbody-refill vs ring) and transport. What is DERIVED
(the swizzle rule — layout, not eligibility): fill/wait/sync placement, prologue fills, wait-group counts.

## Design

- **Segment tagging.** Lift `_twist._stream_segments`'s ad-hoc 3-way split to a tagged form: the builder returns
  `[(stmts, reads: frozenset[str])]` segments keyed by which slab each reads (each `_frag_contraction` call already
  knows its `b_slab`; the softmax segment reads none; the staged-Q chunk loads read `_q_smem` inside the QK
  segment). The segment boundaries ARE the partial's node boundaries — no new IR.
- **`pipelined_kloop(segments, operands)`** in `_stage.py` replaces both skeletons. Each operand carries
  `(transport, depth, delta, first_seg, last_seg)`. The skeleton walks segments in order, inserting per-operand
  `wait` before `first_seg` and `Sync(); fill(next)` after `last_seg` (depth-1) or the ring-prefetch at the top
  (depth≥2), then runs the wait-group counting pass for cp.async operands. δ=0 operands emit prologue-only fills.
  `staged_kloop` remains as the derivation's output for the all-spanning case until byte-parity is proven, then
  folds in; `alternating_kloop` is deleted.
- **Codec unchanged.** `d1/…/alt` keeps meaning "per-operand depth-1, liveness-placed refills, Q staged"; `d2/…/ring`
  keeps the paired ring. A per-operand depth spelling is explicitly out of scope (no evidence any mixed-depth row
  wins; revisit only with a measured case).
- **Composition preserved**: the causal tile-skip `k_end`, the split-KV `Reduction.offset` window, and the golden
  pin contracts ride through untouched (they parameterize the loop bounds/coords, not the fill placement).

## Invariants (from the hand-derivation — these made alt correct)

- Every derived `Sync` is CTA-uniform, placed from the BODY ORDER, never per-warp (barrier-under-mask).
- A kill-point fill is legal only after a barrier all readers passed; a depth-1 δ=1 operand therefore costs exactly
  one extra `Sync` per iteration vs the ring — the measured hd64-vs-hd256 trade.
- Tail refills clamp onto the last needed chunk (re-fetched, never waited) — harmless, keeps the fills unguarded.
- cp.async commit-queue counting assumes ALL threads issue fills in the same body order (uniform code — holds by
  construction).

## Steps

1. Segment tagging in `_twist` (mechanical; the 3-way split becomes the tagged list). Verify: the concatenation is
   byte-identical to today's `_stream_step` output for ring configs.
2. `pipelined_kloop` + the wait-group counting pass. Verify: for every existing golden STAGE spelling (ring d2/d3,
   d1, tma/cp, alt tma/cp), the emitted CUDA is **byte-identical** to the current skeletons' output — the gold gate;
   a diff harness over the golden configs' dumped kernels makes this a one-command check.
3. Delete `alternating_kloop`; route `alt` through the scheduler. Re-run the alt e2e tests (4 cases) + the hd256/
   hd128 pinned benches (31.7/29.7 and 15.7/13.5 must reproduce) + `make test`.
4. (Optional, separate commit) δ=0 residency knob for OTHER loop-invariant staged operands (the fused norm→linear
   A-row is the candidate) — only if a measured case wants it.

## Out of scope

- Per-operand depth codec spellings; prior featurization of the alt/ring fork (needs tune data first).
- WSPEC × alt composition (still gated off), symbolic-kv alt (resolver still declines), cross-step reordering of
  compute segments (measured timing-neutral and racy — refuted 2026-07-14; this plan schedules FILLS only).
