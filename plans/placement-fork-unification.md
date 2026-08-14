# Placement fork unification: fusion → recognition → measured cuts

Working plan for branch `feature/placement-fork-unification`, which supersedes PR #504 and PR #505. Both PRs
attack the same boundary — fusion entangled with target-specific placement — from different campaigns (H200
tuning, V100 FP16 Llama). The layer contract this branch lands, each line enforceable by the pass contract test:

1. **Fusion** (`loop/fusion/`) — greedy-maximal, deterministic, algebra-only. Never reads `Context`, knob
   vocabulary, or hardware. Refusals limited to semantic preservation, the duplicated-contraction brake, and a
   compile-cost bound.
2. **Recognition** (`lowering/tile/`) — algebra-only, total over what it accepts; never reads `has_*`
   capabilities.
3. **Placement** (`_cut.py`) — single owner of fuse/cut. With no pin or routing golden, offers one structural
   fork: maximal fused form plus one option per realizable seam. Fork construction is target-independent. GPU
   product identity lives in exactly one function (`_routing_entry`), selecting measured evidence.
4. **Schedule** — enumerates the capability-legal domain via named `Context` predicates and
   `atom.available_on(ctx)`; enumeration does not rank.
5. **Policy** (`search/policy/`) — the only layer combining hardware facts with preference: deploy evidence
   hierarchy first, capability-derived cold leads when no evidence exists. #505's V100 knowledge lives here.

## State (2026-08-13)

- Branch = latest `main` + merged `codex/fix-llama-fp16-v100-kernels` (which already absorbed #504's original
  head and the fork reconciliation: fork owns placement, #505's cut rules are fork alternatives, FP16 repairs
  landed).
- #504's branch was rebased and grew a **parallel implementation** of exact placed-kernel plan persistence
  ("Add exact placed-kernel tuning") competing with this branch's `replay_plan.py` ("Persist exact placement and
  schedule plans"). One must win; the loser's orthogonal fixes (e.g. "Fix nested schedule site materialization",
  sync-transport tests) get ported.

## Remaining work on this branch

- [x] Replay verdict: kept this branch's transcript + `promote_replay_plans` design; ported from #504's head the
      nested-SSA sync-fill fix, the nested-only TILE ownership fix, strict-row accounting, the accuracy
      size-mismatch failure, `PLACE_cut_site_<path>` features, and the `no_exact_pin` fail-closed marker with
      stale-winner supersede. The rebase carried no other compiler drift.
- [x] Deleted `route_cut`; dropped the unused `computed_a_cover` alias.
- [x] Multi-channel sync staging: settled on the N-channel model (Volta sync-copy offered for materialized
      products, no forced sync transport); pinned the computed-operand transport capability boundary in
      `test_move_catalog`.
- [x] Placement cold lead landed (`greedy._cold_placement_lead` + deploy tier 4 wiring + tests).
- [x] Dropped the V100 Llama serving envelope (its golden was never captured; `serve-config-guard` rejects it).
      The envelope returns with the tuned inventory.
- [x] V100 validation (8× V100-SXM3 box): `make test` green (3458 passed; the two failures were the dropped
      envelope and a sync-method artifact); whole Llama 3.1 8B block compiles, runs, and passes accuracy vs
      eager at seq 1 AND seq 512. Three defects found and fixed on the way: the trace-native wide-product
      recomposition (120 GB workspace), cold option order hoisting value-seam cuts, and f16 carriers under a
      cross-thread combine. Cold perf without goldens is far behind eager (a single cold gate-up kernel is 87%
      of block time) — per design; the tuned V100 inventory is the follow-up that closes it.
- [x] Docs: deploy tier 4 + placement sections in `pipeline/ARCHITECTURE.md`; GLOSSARY *Structural fork* cold
      lead amendment.
- [ ] Open the superseding PR; close #504 and #505 pointing at it.

Not ported from #504's final head (recoverable from branch `codex/h200-cublas-kernel-optimization`,
`305b3de9`): the materialized-score flash lowering (`PLACE@dd=cut` → PV mma; `_twist`/`_flash`/`_schedule`
work entangled with its `lowering/schedule` pass-package split) and the `ranking.kernel_set` persistence form
superseded by `replay_plan`. The H200 attention placement search continues on top of this branch.

## Deferred (follow-up PRs, not this branch)

- Fold fusion's perf refusals (`_BLOWUP_FACTOR`, activation/product ordering) into stamped seam facts read by
  the placement cold lead; ships with a per-card golden re-verification (bigger regions re-key shapes).
- Move enumeration preference heuristics to cold leads: `_f16acc_allowed`'s `f16acc_is_faster` half, coop
  overrides (`_COOP_*`, `coop=32`), `_SPLITK_MAX_CTAS`.
- Weight-layout structural fork replacing `_fold_constant.py`'s `has_tma` gate + matvec exception.
- One `ShapeKey` kind classifier stamped at recognition (delete the `greedy._fork_shape_key` and
  `_cut._routing_entry` reconstructions).
- Delete the `or 170.0` SM-count featurizer fallbacks; derive `_MAX_DELEGATED_WORDS`; unify decode-bucket
  constants (`config` 16 vs `serving/twins` 32).
- Contract-test extensions: `from emmy.compiler import target` import loophole, recursive fusion glob, frontend
  decomposition scope, named-exemption list for numeric perf constants.
