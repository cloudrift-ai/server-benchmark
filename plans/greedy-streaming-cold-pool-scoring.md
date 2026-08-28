# Greedy cold-pool scoring: stream the leaves, memoize the geometry block

## Problem

Greedy's cold model-tier pick (`search/policy/greedy.py`, `decide()` after the evidence tiers decline) materializes
the whole schedule pool three times over — `flatten_leaves()` (leaf `Fork` objects + knob dicts), the `live`
(leaf, knobs) pairs, and `rows = [{**base, **k} ...]` (60+ floats per row) — then featurizes per-dict at ~12k rows/s.
On the 486k-row pools (cold attention-mask policy) that is O(pool) memory (the OOM) and ~40s per cold fork (the time
wall). Known casualties, each marked at its site: `test_gen_prefill_device_gpu` (11.9s → >15min, SKIPPED),
`tests/test_quant.py:1098`, the `test_block.py` CUDA skips. Enumeration itself is not the problem — the recursive
walk streams ~500k rows in ~0.01s and the evidence tiers already descend without flattening (`_find_decided_leaf`,
decision memo); only the first cold model-tier decision per `pool_key` pays.

The prior never needed the flat list — it needs complete rows (partial branches are unfeaturizable: the `D_*`
geometry block requires the full codec set). Both model kinds already score matrices in one call
(`LinearModel.quality_rows` / `CatBoostModel.quality_rows`, `score_rows` over a packed `Group` — the pool-shaped
surface `prior/base.py` built because the per-dict path "would make an evaluation OOM for the same reason" it made
the fit OOM). Greedy is the one caller that never adopted it.

## WS1 — streaming chunked scoring (fixes memory; pick bit-identical)

Replace the flatten-and-score in `decide()`'s model tier with a single pass over `iter_leaves(fp.options)`:

- Accumulate chunks of ~4–8k leaves (thousands, not hundreds — CatBoost pays per-`predict` overhead).
- Per chunk: extract `_leaf_knobs`, drop blocklisted tiles (`_tile_blocked`), split out structural options
  (`_is_structural_option` — they go to `_priced_pick` / withdrawal exactly as today; they are few and never the
  pool), featurize, pack the chunk matrix, score.
- Evidence first, per chunk: `evidence_pick` / `_db_measured_pick` operate row-wise (sig-group joins), so run them
  on the chunk and keep a running best-evidence candidate separately from the running model-argmin; after the
  stream ends, apply today's hierarchy (evidence beats model prediction) to the two running bests. The
  disjoint-evidence warning needs the end-of-stream position too.
- Running argmin with ties falling to emission order (first seen wins) — this preserves the option-0 tie rule and
  makes the result invariant to chunk boundaries. Never retain a chunk after scoring; keep only the winning leaf
  object, its knobs row (for the decision memo), and its price.
- `base = {**fp.ctx.features(), **fp.root_op.knobs}` is constant per fork — featurize it once and merge per-row
  feature dicts, instead of building 486k merged knob dicts.

Memory becomes O(chunk). The pick is unchanged: argmin commutes with chunking and the walk's leaf order is
deterministic (the invariant `pool.py`'s reservoir already rests on).

## WS2 — memoize the geometry feature block (fixes time)

The ~12k rows/s is model-agnostic featurizer overhead, dominated by the schedule-geometry block: per row,
`knob_features` slices each schedule-bearing node (`_node_slice`) and computes `_schedule_node_features`
(area/occupancy/reuse ...). The *parses* are already memoized on spelling (`_resolved_tile`: "dozens of distinct
spellings beside ~100k rows per pool"); the feature-block computation is not.

Memoize `_schedule_node_features`' returned dict on the key it is a pure function of: the node slice's codec
spellings plus its structural extents (a hashable tuple of the slice's items, or the `_row_values`-style spelling
tuple extended with the `S_*` extents the block reads). A 486k-row pool has O(100s) distinct keys. Cautions:

- Treat the memoized dict as immutable — `knob_features` sum-pools it into `feats`, which already copies via
  `feats.get(...) + val`; verify no caller mutates the returned block.
- Size the lru generously (the key space is small; 64k is safe) and keep the memo module-level like the parse
  memos, so fit and eval passes share the win.
- Byte-identity gate: featurizer output must be exactly unchanged (same floats, same key set) — assert via a
  golden-pool featurization digest A/B before/after, same method as `scripts/digest_kernels.py` uses for kernels.

Expected effect: geometry work collapses to lookups; the residue is the trivial per-knob loop + dict merge,
plausibly 10–50× per-row. Combined with WS1, a 486k pool should score in low seconds with bounded memory.

## Non-goals

- No per-level / hierarchical priors: tree shape is a runtime artifact (single-value collapse, level splicing;
  stated invariant: the pick is invariant to level arrangement), partial rows are unfeaturizable, and a level-wise
  pick needs a value function the rank fit does not produce.
- No sampling of the cold pool (deterministic-reservoir cap stays a last resort — it changes the pick).
- No branch-and-bound: linear-only, and unnecessary if WS1+WS2 land the targets.

## Verification

1. Pick identity: on a set of recorded cold decisions (several pool sizes, both model kinds, with and without a
   blocklist), the streamed pick equals the flattened pick — leaf, row, and `fp.score` — exactly.
2. Featurizer digest A/B (WS2) over a golden pool: byte-identical feature vectors.
3. Un-skip `test_gen_prefill_device_gpu` and `test_quant.py:1098`; re-check the `test_block.py` CUDA skips; the
   prefill test must land near its former 11.9s.
4. Peak-RSS check on the 486k-row pool path (the attention-mask policy compile) — bounded, not O(pool).
5. `make test` green; timing drift recorded per `make test-durations` policy if worker balance moves.
