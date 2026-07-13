# Post-swizzle manual golden refresh — RTX 4090 (rented 176.124.69.204), 2026-07-12

**Method: manual by explicit direction — NO tuner, NO prior** (the prior is trained on pre-swizzle
measurements and its known deploy-evidence holes would steer the exploration; every variant was a pinned
`--ab` row benched live against the golden + eager rows in the same run). Three waves off the box's
`_tune/manual-refresh-4090/` logs: wave 0 replayed all 17 matmul goldens (27 entries) on the swizzle branch
(PR #351), wave 1 ran ~45 pinned exploration variants around the fp16 incumbents (k4 chunks, warp splits,
`/p2` reg double-buffer, d1/d3 depth, fm atom where it had been parity), wave 2 confirmed every candidate
2–3× plus dynM transfer probes. ~2 h wall. All new configs accuracy-checked (`EMMY_KNOBS`-pinned `run`);
post-stamp the whole file replays true.

**Tally: 15 `emmy_us` refreshes / 2 fm replaces / 4 fm adds / 0 std-lane config changes / fp32 squares
untouched** (scalar tier — `Load` drains, unswizzled by design — replayed within ±3%, confirming the
swizzle changed nothing there).

## Headline — the swizzle moved the fm optima to wide-N `f4x8` tiles; five more shapes now beat cuBLAS

The conflict-bound smem pipe had been taxing exactly the tiles with the widest B drains, so several
pre-swizzle "losers" flipped to winners — a region no prior trained on old data would revisit, which is why
this refresh had to be manual:

| shape | change | new best µs (total) | was | vs cuBLAS |
| --- | --- | --- | --- | --- |
| o_proj.h4096 | **fm ADD** `f16_f16/w2x2/f4x8/k2 g2k` (fm was parity pre-swizzle, unrecorded) | **88.4** | 108.7 (std/p2) | **0.74** |
| o_proj.h4096.dynM | **fm ADD** (static winner transfers, no masked penalty) | **89.3** | 119.5 (std) | **0.80** |
| square.2048.fp16 | **fm REPLACE** `w1x4/f4x4` → `w2x2/f4x8/k2` (old entry 29% slower, pruned) | **82.7** | 106.7 | **0.72** |
| square.4096.fp16 | **fm REPLACE** `w1x4/f4x8/k2` → `w2x2/f4x8/k2` (old 5.5% slower, pruned) | **603.1** | 639 | **0.73** |
| mlp_gate_up.h4096 | fm parity ADD: incumbent tile + `/p2` (−2.5%, inside the 3% gate) | 648.2 | 664.6 | 0.90 |
| square.512.fp16 | fm parity ADD: atom-swap of the `/p2` std golden (mirrors the 5090 file) | 4.8 | 4.9 | 0.83 |

With the pure `emmy_us` refreshes (gate_up fm 786.9→664.6, mlp_down fm 348.5→274.0 at 0.71×, qkv fm
323.5→267.7 at 0.82×, sq1024 fm 19.9→15.8, dynM twins in line), **the fm lane now beats cuBLAS HGEMM on 9
of the 11 fp16 matmul shapes** (parity on sq512/sq1024-class smalls); the std lane improved 2–11% where the
drain was conflict-bound (o_proj/p2 121.6→108.7 the largest) and its incumbent configs all still stand.

## Finding 1 — every incumbent config survived; the swizzle only re-priced, never invalidated, the std lane

No std-lane exploration row beat its incumbent beyond noise (k4/d1/d3 still lose everywhere — d1 766 vs 648
on gate_up fm, d3 755–966, k4 678–784: depth/chunk tradeoffs are occupancy-bound, not smem-bound, so the
swizzle didn't move them). The fm lane's flips are all one mechanism: `f4x8` B tiles read 128 B slab rows,
which were 8-way conflicted and are now clean.

## Finding 2 — `/p2` (smem→register double-buffer) is newly live on the biggest tiles

Pre-swizzle `/p2` was noise-to-negative on the fm winners; post-swizzle it's a consistent −2.5% on gate_up
(3×, 647.7–649.2 vs 664–668) and the o_proj std `/p2` entry gained 11%. With conflict replays gone, the
ldmatrix latency the double-buffer hides is now the visible term. Recorded as a parity-add on gate_up only
(inside the 3% gate); worth re-probing on other cards' cp shapes in their refreshes.

## Finding 3 — the one regression-shaped drift: `mlp_down.h4096.dynM` std (w2x4/f4x4)

Replays 424.6–425.2 (3 consistent passes) vs its recorded 387.3 — +10%, the only entry slower post-swizzle.
Its history already wobbled 387–425 across sessions (the 07-12 sweep refreshed it 412.9→387.3 the other
way), and its `f4x4` geometry makes a large conflict win impossible while the fill-side XOR adds a few
instructions — a small real regression on this specific tile is plausible but not separable from the
wobble. `emmy_us` refreshed to 425.2 (honest current value); the shape's fm twin (275.5) dominates it by
35%, so nothing deploys through it. If a future sweep needs this lane, re-derive the std config rather than
trusting either historical number.

## Fork sibling regret

Not run — this refresh deliberately used no tuner (`emmy tune` retrains the prior on -O1 rankings; the
user's direction was manual precisely because the prior is suspected broken, and the -O1 censoring issue
from the 07-12 5090 report still stands). The prior/node-store is now doubly stale: it has NO post-swizzle
measurements, and the fm optima moved. A future `--clean` golden tune (with the family-aware -O3 rebench
floor, still open) should re-seed it; until then the goldens are the only trustworthy source for these
shapes.

## Workflow notes

- Three nohup'd wave scripts + log scraping worked well (~2 h wall, zero babysitting); the `--ab` harness
  again caught everything (no declined pins this time — all 45 variants realized their knobs).
- The split-K finalize row still requires hand-summing totals per config — the `golden NAME (total)` row
  proposal from the 5090 report stands and would have saved the most error-prone step of this analysis.
- Refresh criterion used: ≥3 passes, spread <2%, drift >3% (the older "≥5%" rule alone would have kept
  stale values on tight 4–5% drifts like gate_up std 964.6→919.5 across 5 passes at <0.3% spread).
- `--ab` rows are bench-only; accuracy needed a second `EMMY_KNOBS`-pinned `run` per new config. A
  `--check-accuracy` flag on `--ab` rows (or an accuracy column) would fold those 6 extra invocations away.
- Replay wobble was small this time (most entries <1% across passes); the noisy ones remain sq2048 std
  (115–127) and gate_up.dynM std (856–922), both left unrefreshed where inconsistent.
