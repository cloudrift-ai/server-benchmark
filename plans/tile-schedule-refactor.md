# Tile IR + schedule refactor — remaining work

The design landed: the λ-foldMap tile IR (two stored node kinds, no loop annotations, no stored step sequence), the
tree-path knob codec, the WORK/site-local value grammar with the mechanically re-spelled golden corpus, the α-invariant
term-hash kernel identity, and the phase-4 placement realizer with routing entries. The design as built is documented in
`emmy/compiler/pipeline/passes/ARCHITECTURE.md` (and CLAUDE.md's summary); every storage step was gated on
`scripts/digest_kernels.py` byte-identity. This file tracks only what remains.

## Remaining

0. **Eliminate the legacy (embedded-worker) value grammar.** Step 7 flipped the STORED surface to the site grammar but
   left the internal enumeration speaking legacy, with adapters at the boundary. Detail + staging below.
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

## Item 0 in detail — killing the legacy value grammar

### The two grammars

| family      | legacy (embedded workers) | site |
| ----------- | ------------------------------ | ----------------------------------- |
| TILE warp   | `a:<atom>/w<M>x<N>/f../k..` | `<atom>/f../k..` + `WORK: w<M>x<N>` |
| TILE scalar | `n<M>x<N>/f..` | `f..` + `WORK: t<M>x<N>` |
| REDUCE      | `b<n>` (coop width) | `coop` + `WORK: t<n>` |

`g<n>[a|k]` splits and `r<n>` reg folds are IDENTICAL in both — the whole REDUCE delta is the `b<n>` coop widths. The
`a:scalar` / `a:none` pin-only aliases are a SEPARATE family (`has_scalar_atom_alias`), unaffected, keep them.

### Where legacy actually lives (surveyed 2026-07-31)

- **Produced — 5 sites, 2 files.** `space.py` `scalar_tile_moves` / `warp_tile_moves` (`TilePlan(...).spell()`) and
  `coop_reduce_moves` (literal `"b4".."b512"`, `"b32t"`, `"g8k/b128t"`); `_schedule.py`'s `["b32"]` matvec default and
  the two `_legacy_*` adapters. **Nothing else in `emmy/` spells legacy.**
- **Consumed** — `_schedule.py` (24 `TilePlan.parse`/`ReducePlan.parse`), `knob.py` (4), `golden.py` (1, a
  dual-grammar `_atom_of`), and the codec itself.
- **Stored — NONE.** 709 TILE + 363 REDUCE values across all six golden YAMLs are already site; the only legacy
  strings there are inside `#` provenance comments, which nothing parses. The tune DB is the unverifiable one
  (machine-local; pre-flip rows degrade to cache misses, not corruption).
- **Tests — 30 files**; docs — `pipeline/ARCHITECTURE.md:974` (documents the alias as supported) and
  `passes/ARCHITECTURE.md:335` (the `a:scalar` alias — different family, leave).

Two findings that make this bigger than a rename:

- **Legacy tolerance lives in the CODEC, not in `_schedule.py`.** `TilePlan.parse_site` / `ReducePlan.parse_site` /
  `resolve_site_tile` accept BOTH grammars and raise only when an embedded worker token CONTRADICTS `WORK`. So deleting
  the `_legacy_*` helpers leaves legacy unproducible but still accepted. Only step 3 below makes it impossible.
- **`knob.py:433/440` discriminates the grammars by legacy-parse FAILURE** (`# site values fail this parse ⇒ imply
  nothing`) to derive an implied `WORK`. Once legacy can't be produced this can never fire — it is code to DELETE, not
  to port.

### Staging (each stage green on `make test` before the next)

1. **Catalogs + `_tile_rows`** — one commit; the biggest win. Catalogs return `TilePlan` / `ReducePlan` objects
   constructed structurally (`ReducePlan.of(coop=128, coop_transposed=True)`, not parsed literals); `_tile_rows`
   assembles rows as site value + `WORK` inline; `_site_row`'s conversion collapses into `_filter_work`; `_legacy_tile`
   / `_legacy_reduce` delete; the pin path becomes `resolve_site_tile(pin, _pinned_workers())`.
   **TILE and REDUCE must flip together** — they share the `WORK` column, so `_site_row` cannot be half-flipped.
   VERIFIED 2026-07-31: the structural REDUCE catalog reproduces today's parsed literals exactly (`built == old`), so
   the enumeration is unchanged by construction.
   Watch: the flash arm has its own TILE handling (`_canon_tile_spec`, `_narrow_flash_forms`, `_demoted_warp_option`,
   `_twisted_warp_options`) that must flip in the same commit.
2. **`knob.py` + `golden.py` cleanup** — delete the now-dead implied-workers discriminator; `_atom_of` loses its
   legacy arm; `canon_family_value`'s TILE/REDUCE arms become identity.
3. **Codec** — `parse_site`/`spell_site` become `parse`/`spell` and the legacy pair is deleted, so an
   embedded-worker string RAISES. This is the step that makes legacy impossible, and the one that breaks
   `EMMY_REDUCE=b64` (site equivalent is two vars: `REDUCE=coop` + `WORK=t64`). It carries the 30 test files —
   `test_codec_roundtrip.py` / `test_codec_validation.py` have the legacy grammar as their SUBJECT and need
   rewriting, not find-and-replace.
4. **Docs** — drop `pipeline/ARCHITECTURE.md:974`'s "stays a validated alias" line with step 3.

Landed already: `69e87ec6` — the four option builders take resolved `TilePlan` / `ReducePlan` objects instead of
legacy strings and stamp `spell_site()` directly, so `_materialize` no longer spells structs down to legacy just to
have them re-parsed one call later.

Gate before merge: this branch stacks several commits whose real exercise is GPU-only (the warp-flash and mixed-dtype
paths). Run `make bench-kernels` plus a flash/attention compile on a 5090, and an `emmy tune` on a 4080 for the two
unmeasured `dit_xl_2.*` matmul goldens, before any of this merges.

## Blocked (live semantics, not legacy debt)

- **Flash's special-cased pin plumbing** (the greedy all-or-nothing `TILE@dd`+`TILE@pj` contract, golden.py's
  dynamic-attention bare-`TILE` schema arm, `_narrow_flash_forms`' keyed-only arm + the masked-flash bare-`TILE`
  fallback) **and the bare↔explicit any-of** (`pin_key_matches` / `family_value`): the dynamic-attention golden rows on
  every card record the PV plan on a bare `TILE` precisely because a symbolic trace resolves no stable axis key, and
  the bare row must match the masked fork's axis-keyed leaves any-of. These die only when symbolic-trace KEYED
  resolution exists (a codec that spells a stable site key off a symbolic-axis tree); until then the exceptions stay
  documented and tested, never silently widened.
