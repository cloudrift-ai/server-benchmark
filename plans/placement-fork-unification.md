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

- [ ] Pick one exact-plan replay implementation; port the loser's unique fixes; take any non-replay drift from
      #504's rebase.
- [ ] Delete compat shims: `route_cut` (dead export), any dual paths found in the audit. No backwards compat.
- [ ] Verify multi-channel sync staging is legal without `cp.async` (V100 path) — the #504 mandatory-sync rule
      vs #505 N-channel emitters clash; "Fix cross-platform contraction legality" may already cover it.
- [ ] Structural-fork cold tier: `_pick_structural` still drops structural leaves on a cold prior. Add the
      capability-derived placement cold lead (rank, don't drop): computed-operand seam + every available atom
      `materialized_edges_only` → lead the cut (the reborn #505 V100 rule); no atom at all → lead fuse.
- [ ] The serving env references `goldens/v100_sm70_meta_llama_3_1_8b_instruct.yaml`, which does not exist —
      capture the V100 inventory during V100 validation or drop the reference.
- [ ] Test on local 5090 (`make test`, lint) and on the V100 box (focused sm_70 suites + Llama FP16 kernel
      compile/run vs eager at M=1/512).
- [ ] Docs: ARCHITECTURE placement sections (fork model is the base; keep the FP16 numeric-contract paragraph);
      GLOSSARY amendments (*Structural fork* cold tier; add *seam*, *placement*, *cold lead*).
- [ ] Open the superseding PR; close #504 and #505 pointing at it.

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
