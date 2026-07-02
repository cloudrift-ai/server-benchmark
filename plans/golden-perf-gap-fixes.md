# Golden perf-gap fixes — A/B integrity, shape-space reachability, analytic deploy, fp16 slab swizzle

Successor to `golden-sweep-rtx5090-findings.md` (sixth sweep): close its perf gaps **without** the learned-prior
trainer (deferred — `prior.json` stays deleted, greedy deploys stay analytic-only; the trainer investigation is a
separate future plan) and without kernel cutting / un-fusing. Ordering rationale: integrity gates first (they decide
whether any later "better/worse" call is real), then the CPU-only reachability + analytic work (cheap, and finding 1
proved the analytic-verify-first pattern), then the one GPU re-sweep, then the codegen gap.

## Current state (pass-2 numbers from the findings doc)

- **Deploy-pick gap**: 14/29 shapes "worse" at 1.02–1.49× the best-golden — the winning row is *in the space* but
  the analytic pick doesn't deploy it (honest static median golden rank 25). No catastrophic class since the
  tie-honest fitter fix; deploys read the analytic `_W_A` / `_W_A_DYN` only.
- **Reachability gap**: every static `.s512` golden and the fp32-square entries are `SKIPPED: recorded knobs not in
  the enumeration` — the recorded register tiles (`f2x14`, `f4x8`, `f4x10`, `f4x26`) are outside
  `search/space._SCALAR_REG`. These are exactly the 1.29–1.49× shapes; no patience can reach them.
- **fp16 codegen gap**: the fp16 squares sit above cuBLAS even at best-golden (1024: 1.56×, 2048: 1.37×, 4096:
  1.15×). The rebuilt transport stages plain NONE-swizzle slabs, so the `ldmatrix` drain eats smem bank conflicts;
  the pre-rebuild bar (`square.2048.fp16` 106.7 µs, 1.06× cuBLAS — `search/golden.py`) was set by swizzled slabs.
  Warp specialization (landed, PR #297) is orthogonal and measured neutral — it does not touch the drain.
- **A/B integrity**: a physically impossible `g2a` golden row (8.2 µs for 2048³, >2 PFLOP/s); fp16 golden rows
  swinging 21.4 ↔ 53.0 µs between passes; golden rows vanishing when greedy deploys a split partial+finalize pair;
  numbers living only in table text (hand parser rewritten three sweeps running).

## Phase 0 — A/B integrity gates (before any re-judging)

1. **Arithmetic-intensity floor**: the `run --bench --golden` A/B computes each row's FLOP/s from the shape and
   flags/discards rows above the device peak (the finding-4 gate); the `g2a` atomic-split re-bench additionally
   asserts its output against the greedy row (the 8.2 µs row was a silent wrong-answer or skipped-finalize bench).
2. **JSON A/B dump**: `--json` (or `--record`) on the A/B emitting greedy/golden/eager rows + flags — retires the
   `_tune/` table parser and hosts the floor; the confirm-twice rule becomes a field, not a habit.
3. **Golden rows attach to the shape, not the kernel node**: a split deploy (partial + finalize pair) must still
   print the shape's golden A/B rows (four s32 shapes lost them last sweep).
4. Finding 6 (reduce/pointwise goldens have no tune/A-B path, carried 2 sweeps): extend `_tune_targets` + the
   `--golden` registry to their existing `snippet()`s, or delete the five entries — decide at execution; either
   closes the recurring hole.

## Phase 1 — reachability: prove the space holds the good shapes (CPU-only)

1. Widen `search/space._SCALAR_REG` with the golden-informed deep-FM points (`(2,14)`, `(4,6)`, `(4,8)`, `(4,10)`,
   and the fp32 squares' `(4,26)` if the thread/register budget admits it) — keep the grid bounded and update the
   structural-coverage test that recomputes the product.
2. **Gate**: `emmy eval prior --dataset golden` reports ZERO `recorded knobs not in the enumeration` across the
   golden set (matmul shapes; modulo Phase 0.4 for reduce/pointwise).
3. **Permanence**: a test asserting every `GOLDEN_CONFIGS` row's `TILE` codec is a member of the enumerated moves
   (warp and scalar alike; `STAGE` checked as resolvable, since rows carry resolved spellings) — a future space
   edit can never silently orphan a golden again.

## Phase 2 — analytic pick quality (no trainer)

1. Re-run the tie-honest rank audit (`scripts/golden_knob_heuristics.py::rank_of_golden`) over the widened space;
   produce the per-shape golden-rank table + per-knob miss analysis (which `D_*` term flips each miss — the
   tune-golden skill's report format: rank, per-knob misses, recommendations).
2. Refit / hand-adjust `_W_A` / `_W_A_DYN` off that audit (the live `D_stage_tma/_ring` terms are precedent; expect
   register-tile-geometry terms to need the same treatment for the deep-FM rows). CPU-only loop: adjust → re-rank.
3. **Gates**: static median golden rank materially below the current 25 (target ≤ 10); no shape whose golden is
   in-space ranking outside the tune's per-shape bench budget; and on the next A/B pass the "worse" tally shrinks
   with no new catastrophic class (worst ratio stays under pass-2's 1.49).

## Phase 3 — re-sweep the previously unreachable families (GPU)

- `emmy tune --dataset golden` (warm resume, `setsid` + PID-file — the pattern that survived the kills) over the
  `.s512` statics + fp32 squares with the widened grid; A/B each through the Phase-0 JSON path, confirm-twice;
  record via the tune-golden workflow (better → replace, same → add, worse → leave). Only after this re-judge the
  1.29–1.49× family — per finding 3, these were reachability losses, not codegen losses.

## Phase 4 — fp16 staged-slab swizzle (the codegen gap)

- B64/B128 swizzle on the staged operand slab + the matching XOR in the staged `ldmatrix` drain
  (`_staged_inner_atom_loop`), TMA-first: the descriptor already carries a `swizzle` field (spelled `"NONE"` today)
  and the pre-demolition `050_use_tma` had the `pick_swizzle_atom` box-reshape to crib from git history; the
  cp.async fill's XOR write follows if TMA proves the win. Swizzle relocates smem bytes only — the staged kernel
  stays **bit-identical** to its unswizzled sibling (a strong verification property the staging tier already uses).
- **Gate**: `square.2048.fp16` at/near the pre-rebuild bar (106.7 µs / 1.06× cuBLAS) on the RTX 5090; the fp16
  squares' vs-cuBLAS column moves toward ≤ 1.1×. Afterwards re-A/B `WSPEC p1/p2` on those shapes — the split's
  value changes once the drain stops being conflict-bound.

## Verification

`make test` + `make lint` per phase; Phase 1's zero-SKIPPED gate and enumeration-membership test; Phase 2's rank
audit table checked into the findings report of the next sweep; Phase 3 A/B through the JSON path with the
intensity floor active; Phase 4's bit-identity + bar gate. The whole plan's end state: a seventh-sweep tally where
"worse" shapes are either genuinely codegen-bound (documented) or gone, and every judgment traceable to a flagged,
machine-readable A/B row.
