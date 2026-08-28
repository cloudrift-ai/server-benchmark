# Bind computed-B contraction operands so attention can materialize its K cone

## Problem

Fused attention kernels recompute producer work (RMSNorm, RoPE, statistics) per grid cell. Fold-tree
canonicalization already shares the cone once in the term, but term-level sharing cannot remove replication across
CTAs: a value every query block reads (the normalized/RoPE'd K vector) must be written to memory once and read
back, which is a kernel-boundary decision. This is why sequence-512 attention targets run hundreds of milliseconds
to seconds against tens of microseconds for `torch.compile` (see the 2026-08-28 checkpoint in
`experiments/golden-bench-2026/kernels/RESULTS.md`), and why no WORK/TILE/REDUCE/STAGE tuning can close the gap.

The transport options for a shared operand edge form a lattice by reuse scope: recompute (fuse), registers/smem
(reuse within one CTA), gmem workspace (reuse across the grid — the placement cut). For the attention K cone only
the gmem rung applies, and today the placement fork cannot offer it.

## Why the cut cannot be offered today

The cut machinery itself already supports a reduction-axis-swept workspace:

- `passes/lowering/tile/_tree.py` extends the axes-in-scope with the fold axis when descending into a
  contraction's operand edges, so a cone indexed by the fold axis can pass `_closed_at`.
- `_cut._workspace_axes` keeps captured axes; the producer piece sweeps them as free axes and the consumer's
  replacement `Load` reads the workspace inside the fold. The workspace dtype rule (fed-store / storage frontier),
  unit-axis preservation, and recursive re-entry of fresh pieces all apply unchanged.
- `cuttable_seams` already ranges over Fold-valued B edges (`node.a` and every `channel.b`).

The gap is one level upstream: the contraction binder in `ir/tile/normalize` declines to bind the attention K cone
as the dot's B operand edge. With no stored Fold edge there, the tree-path codec has no PLACE site to spell and the
seam cannot be offered. Cutting the neighboring sites instead promotes the key axis into the residue's free axes
(the A100 `a2` analysis in RESULTS.md: a 4.2M-block grid, 16x cost) because the value is recomputed, not indexed.

## Plan

1. **Binder extension** (`emmy/compiler/ir/tile/normalize.py`): recognize a computed cone that is swept by the
   contraction's fold axis and closed at it (plus enclosing free axes) and bind it as a proper B operand edge.
   The rule must stay boundary-derived and general — no operation-family or model recognition.
   Verify: unit tests over reduced RMSNorm→SDPA trees assert the edge exists, is alpha-stable across
   reconstruction, and appears as a PLACE site.
2. **Seam offer falls out**: confirm `cuttable_seams` now offers the K-cone seam with a workspace swept by the
   fold axis, and that `realize` produces a producer sweeping the reduction axis plus a consumer reading it as an
   ordinary load. Verify: a realization corpus case reduced from the Qwen3 attention target reaches `offered` and
   `realized` on any machine; `built`/`correct` on an exact-capability card.
3. **Fused-arm audit** (`passes/lowering/tile/_schedule.py`, `_staging.py`): the fill's computed-operand handling
   (stat seams, warp eligibility, the paired producer budget) is written around computed A. Ensure a computed B
   edge in the fused sibling replicates legally per cell without breaking enumeration or warp-plan offers.
   Verify: existing schedule-walk tests plus one computed-B fixture.
4. **Identity churn**: the new edge re-keys `structural_key` and path spellings for affected kernels. Restamp the
   realization corpus (`make test-corpus-regen`) and expect stale goldens for the affected attention targets.
5. **Receipt prerequisite for deploy evidence**: the resulting two-kernel route remains search/ranking evidence
   only — the measured-evidence tiers fail closed on `PLACE` rows until a durable receipt can bind the ordered
   exact child schedules (see the pipeline ARCHITECTURE evidence-tier rules). Making such receipts first-class
   golden evidence is its own follow-up; this plan does not relax the fail-closed contract.
6. **Retune afterwards, not before**: only after 1–4 land, rerun bounded tuning on the affected attention targets
   per card, then the normal archived experiment recipe before any paper claim.

## Success criteria

- The reduced attention target offers a `PLACE` seam at the K cone; cutting it yields a producer whose grid sweeps
  the key axis once and a consumer whose per-step work is a load, with direct eager correctness.
- Sequence-512 attention targets stop exhibiting the recompute blowup class (ms/s-scale) on at least one card,
  measured through `emmy run --golden-file ... --bench`, without any new benchmark script.
- No profitability filtering enters seam legality; the corpus ratchet and fail-closed evidence rules are preserved.
