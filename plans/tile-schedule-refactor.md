# Tile IR + schedule refactor — remaining work

The design landed: the λ-foldMap tile IR (two stored node kinds, no loop annotations, no stored step sequence), the
tree-path knob codec, the WORK/site-local value grammar with the mechanically re-spelled golden corpus, the α-invariant
term-hash kernel identity, and the phase-4 placement realizer with routing entries. The design as built is documented in
`emmy/compiler/pipeline/passes/ARCHITECTURE.md` (and CLAUDE.md's summary); every storage step was gated on
`scripts/digest_kernels.py` byte-identity. This file tracks only what remains.

## Remaining

1. **Evidence re-keying / re-seeding + the phase-5 consolidated parity gate** (Dmitry executes separately, the manual
   golden method — the tuner is not used for golden work): the tune DB / reservoir / online prior are REGENERATED, not
   migrated (the identity + wire re-keys discarded the old rows by design; pre-wipe µs are not evidence). Re-seed the
   retired PLACE goldens as routing + child schedule entries by hand-pinned `--ab` sweeps on both cards; then the gate:
   eval-golden MATCH across the board, twins deploy from tier, TPOT/TTFT within noise.
2. **`TilePlan.units` field deletion** (~150 consumer sites across the materializer): `TileOp.work` is the
   authoritative worker slot and `derive_workers` fails loudly on disagreement; the value object keeps its validated,
   `work`-agreeing `units` until one dedicated consumer-migration commit.
3. Cosmetic: newly recorded fused entries spell the cone stat's `REDUCE@<stat axis>` key explicitly (the axis name is
   trace-deterministic but ugly). The step-7 re-spell was value-level only, so the move to the path form
   (`REDUCE@a.fold.k`) waits for whenever such entries are next recorded.

## Blocked (live semantics, not legacy debt)

- **Flash's special-cased pin plumbing** (the greedy all-or-nothing `TILE@dd`+`TILE@pj` contract, golden.py's
  dynamic-attention bare-`TILE` schema arm, `_narrow_flash_forms`' keyed-only arm + the masked-flash bare-`TILE`
  fallback) **and the bare↔explicit any-of** (`pin_key_matches` / `family_value`): the dynamic-attention golden rows on
  every card record the PV plan on a bare `TILE` precisely because a symbolic trace resolves no stable axis key, and
  the bare row must match the masked fork's axis-keyed leaves any-of. These die only when symbolic-trace KEYED
  resolution exists (a codec that spells a stable site key off a symbolic-axis tree); until then the exceptions stay
  documented and tested, never silently widened.
