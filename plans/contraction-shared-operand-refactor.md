# Contraction generalization: let-bound shared operands, sibling nodes instead of fold channels

## Sequencing

Execute BEFORE the knob refactor's phases 2–4 (`knob-tree-path-codec.md`) — the PLACE realizers don't exist
on this branch yet, so the tree vocabulary can change while nothing consumes it. Knob phase 1 (codec core)
can proceed in parallel; its stampers (phase 2) then target the post-refactor nodes. This refactor also
DELIVERS the knob plan's nodification prerequisite #1 (the cone's stat reduce becomes a real node here).

## Problem

`Contraction.folds` bakes a fusion decision into node data: the `(b_load, acc)` channel tuple says both
"these N matmuls share one A operand" (a structural fact) and "they have been fused into one loop" (a
schedule/placement choice). The codebase now has THREE ad-hoc encodings of operand sharing:

1. `folds` channels — gate⊗up in one kernel, in-register A reuse (computed-A only, sync tier only);
2. the `035_merge_sibling_linears` concat rule (`source_parts`) — q/k/v sharing x, merged at graph level;
3. MIMO multi-output nodes (#433) — sharing across kernels.

Consequences of the conflation: the multi-channel carrier is *synthesized* inside `Contraction.loop` (the
N-component product-monoid state) instead of emerging from structure; `b_trans`-must-agree is a node assert
instead of a fusion-eligibility gate; the cross-CTA channel split (#389) needed bespoke handling and came
out correct-but-null; `Map.source` is single-valued so a combine over two folds can't be spelled.

## Design

Sharing becomes structural (a reference), fusing becomes a decision (later: the PLACE seam machinery).

```
bindings: x̂ = Map(body=scale, source=Reduction(stat over k))    # the cone, defined ONCE, now a real node tree
op:       Map(body=swiglu(acc_g, acc_u) + Write,
              sources=(Contraction(a=Ref(x̂), b=Wg → acc_g),
                       Contraction(a=Ref(x̂), b=Wu → acc_u)))
```

- **Let-tree, not implicit DAG.** `TileOp` gains a `bindings` table (name → node tree); a new leaf kind
  `Ref(name)` appears wherever a bound subtree is consumed. Every existing tree walk (`structural_key`,
  the `rewrite` registry, `_flatten_nodes`, `lower`, `pretty`, the materializer recursion) stays a tree
  walk + one `Ref` case; nothing needs DAG-aware traversal.
- **`Contraction` loses `folds`**: one `b_load`, one `acc` (today's properties become the fields). A fused
  multi-fold edge is a SIBLING GROUP — N contractions under one `Map.sources` tuple sharing an A `Ref`.
  `Map.source` → `sources: tuple[...]` (len 0/1 = today's forms; `source` stays as a len-1 compat property).
- **`a_operand: Load | Body` → `a: Load | Ref`.** The computed cone moves into `bindings` as a real
  `Map(source=Reduction)` tree — the stat reduce is now addressable (`REDUCE@a.reduce.k` per the knob plan)
  and `stat_prologue()`'s body-splitting becomes a read of the binding's structure (Map body = per-cell
  cone, Reduction = the stat; the k-seam split is the node boundary).
- **Fused sibling groups schedule as ONE unit**: a single shared `TilePlan` / `Stage` / `ReducePlan` row for
  the group (channels agreeing by construction, where today's `folds` shared them implicitly). So the knob
  spelling needs NO sibling ordinals — one `TILE@contraction.k` key per fused group; siblings only schedule
  separately after a future cut, when they are separate kernels (separate trees) anyway.
- **Group lowering must reproduce today's codegen bit-identically**: one A fragment reused across the
  per-channel mma chains, one C fragment per channel, the same synthesized N-component carrier for the
  cross-CTA split tier. The carrier is now DERIVED from the sibling group at the split/lower site instead
  of stored — same derive-never-store rule as `Reduction.loop`.

Out of scope (follow-ups this unlocks, not in this plan): re-expressing the `035` concat rule and MIMO
producer cuts as reference-driven decisions; sibling-group fuse/cut as a PLACE seam (knob plan phase 3
picks that up); LayerNorm multi-stat (N bindings, falls out of the vocabulary for free).

## Workstreams

### WS1 — IR (`ir/tile/ir.py`, `ir/tile/ops.py`)

- Add `Ref`; add `TileOp.bindings`; `Map.sources` tuple (+ `source` compat property); strip
  `Contraction.folds` → scalar `b_load` / `acc` fields; `a: Load | Ref`.
- `ops.lower` / `_flatten_nodes` / `pretty` / `axis_role` / `reduce_loop` resolve `Ref` through the owning
  `TileOp`'s bindings (thread a resolver, keep signatures otherwise); `rewrite` handler renames binding
  names like SSA names so `structural_key` canonicalizes sharing.
- The multi-channel `Contraction.loop` synthesis moves to a group-level helper (`ops.group_loop(map)`):
  same Loop, same carrier, same `op_cache_key` bytes as today's `folds` path. Assert byte-parity in a test
  against a captured pre-refactor key for the geglu + norm_linear shapes.

### WS2 — recognize / schedule (`010_recognize`, `_schedule._contraction_node`, `_atomize`)

- `_contraction_node` emits the binding + sibling group instead of stacking `folds`; the gate/up detection
  (shared lifted A over same (m,n,k)) becomes "second contraction binds the same A cone → same `Ref`".
- The cone is nodified at recognize time (Map(source=Reduction) into `bindings`) — this IS the knob plan's
  prerequisite #1, including its bare-`REDUCE` resolution guard (the binding's reduce stays out of the
  bare-eligible set; stored bare `REDUCE` keeps meaning the contraction fold).
- Scheduler stamps ONE shared plan per fused group; `b_trans` agreement moves from `Contraction.__post_init__`
  assert to a group-formation gate (disagreeing layouts simply don't group — they were never fusable).

### WS3 — materialize / factorize (`_factor`, `010_materialize`, split machinery)

- `factorize` reads the group (N b-loads/accs) off `Map.sources` + `Ref` identity instead of `folds`;
  the sync compute-fill tier's one-A-fragment/N-chain emission unchanged.
- `030_split_reduce`'s channel split derives the N-component carrier from the group; re-run the #389
  multichannel-split A/B — the null result may flip once the split is structural.
- `external_reads` / IO population: binding trees contribute their loads once (dedup by `Ref`).

### WS4 — compat + evidence stability

- Golden identity: fused kinds (`mlp_geglu`, `norm_linear`, fused `.lin` twins, `lm_head` fused) must keep
  their ShapeKeys and knob spellings byte-identical (short paths are canonical — nothing re-keys).
- Dump/kname stability: kernel names derive from realized ops — verify `<kname>.torch.json` reproducers
  still slice correctly with the cone in a binding.

## Verification (gates, in order)

1. Unit: Ref round-trip through `rewrite`/`structural_key` (sharing canonicalizes; two refs to one binding
   ≠ two copies); group_loop byte-parity vs captured pre-refactor `op_cache_key`s; group-formation gates
   (b_trans disagreement, non-shared A) fall back to separate recognition.
2. `make test` — full suite, including the fused computed-A tests and `test_golden_configs` permanence.
3. `emmy eval golden --in-model` both cards: MATCH/DRIFT/GAP counts identical to this branch's baseline
   (capture the baseline FIRST — the wipe branch's counts, not main's).
4. Kernel-dump diff (`EMMY_DUMP_DIR`) on the gemma-4 layer-0 geglu / norm_q / lm_head shapes: CUDA output
   byte-identical (or diff-explained: name churn only).
5. Accuracy: `emmy run --bench` on the geglu / norm_linear golden snippets + decode twin e2e sanity on the
   5090 (TPOT within noise of the recorded goldens).

## Risks

- **Byte-parity of the fused lowering** is the load-bearing gate — any drift re-keys kernel caches and
  invalidates golden µs. Mitigate: WS1's captured-key test lands before WS2 flips the producer.
- `Ref` resolution threading through `ops.*` helpers touches many call sites mechanically — keep it one
  commit, no behavior change, before the recognize-side flip.
- The cone nodification changes `--ir tile` dumps and any test asserting on `a_operand` Body shape — sweep
  `tests/compiler/passes/` for structural assertions early (test_structural_features, test_fuse_finalize
  remnants).
