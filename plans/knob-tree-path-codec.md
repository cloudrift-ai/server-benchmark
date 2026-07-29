# Knob refactor: tree-path addressing + PLACE as a per-seam edge property

## Context

This branch (`feature/remove-place-knob`) wiped the old PLACE machinery: the `020_cut_edge` /
`025_sink_row_reduce` / `032_fuse_finalize` realizers and `_sink.py` are deleted, `PLACE` is gone from
`search/space.py`, and the gemma golden YAMLs carry their old PLACE keys only as `# retired knob dropped:`
comments. The tree cannot express any placement decision right now — a clean slate for the redesign.

The redesign: schedule knobs stop being a flat per-kernel dict keyed by loose axis names and become addresses
into the **recognized algebra tree** (`Map` / `Reduction` / `Contraction`, `ir/tile/ir.py`). PLACE returns as
the one **edge property** — `cut | fuse` on every parent↔child seam — instead of four hand-named sites
(`@fold` / `@fin` / `@cone` / `@stat`).

## Grammar

```
FAMILY@<node-path>[.<axis>][<n>] = value
```

- **Families keep their names** (backwards compat): `TILE`, `REDUCE`, `STAGE` are node properties
  (TilePlan / ReducePlan / Stage); `PLACE` is the edge property; `RASTER`, `WSPEC`, `LOOPIFY` stay
  root-global and bare. The family selects *which property* of the addressed node — no `.property`
  suffix needed.
- **`<node-path>`** = node-kind segments from the tree root, lowercase, dot-separated: `map.reduce`,
  `map.contraction.a.reduce`. Edge labels where a kind repeats under one parent are the field names
  (`a` for `Contraction.a_operand`'s cone; `partial` children need no label — see axis rule).
- **`<axis>`** = the schedule-bearing axis (reduce / contraction-K), the leaf discriminator. Required for
  `TILE` / `REDUCE` / `STAGE`; absent for `PLACE` (its path names the seam's child node) and for a `Map`
  body tile (`TILE@map = f2`, no axis).
- **`<n>` ordinal** — emitted ONLY when kind + axis collide at the same path (LayerNorm's mean/var over one
  `k`). Ordinal is over the canonicalized (structural-key) traversal order.
- **Uniqueness invariant**: axis names must be unique among same-kind siblings at the same path; the stamp
  helper asserts it and emits the ordinal when violated.

### Sugar: shortest-unique suffix

`TILE@dd` ≡ `TILE@map.reduce.contraction.dd` whenever the suffix resolves uniquely in the kernel's tree; a
bare family (`REDUCE`) resolves to the unique eligible node (the current `resolve_axis` contract, one level
deeper). Rules:

- **Accept short, store long**: pins / hand-written YAML may use any unique suffix; the stamped knob row and
  all recorded evidence (DB, online prior, goldens) carry the canonical full path. Evidence *reads* match by
  suffix, so pre-refactor rows keyed `TILE@dd` / bare `REDUCE` keep matching without migration.
- An ambiguous suffix is a `ValueError` naming the candidates (extends `knob.resolve_axis`).
- **Axis names survive placement cuts** (see below), so a suffix key names the same node on both sides of a
  `cut` — pin strings stay valid across a cut/fuse A/B.

### Keys stamp against the pre-placement tree

All keys — schedule and placement — address the recognized tree *before* any cut rewrites it. Kernels are
derived from the cut set; a cut never re-keys downstream decisions. A cut child re-recognizes as its own
tree (its keys re-root: `map.contraction.a.reduce.k` → `reduce.k`), and the parent-path spelling of a child
decision MUST resolve to the same evidence as the child-tree spelling (suffix matching + preserved axis
names make this automatic).

## Spelling migration (gemma goldens, all ~580 live entries)

| Kind | Old | New canonical |
| --- | --- | --- |
| matmul | `TILE` / `REDUCE` / `STAGE` (bare) | `TILE@contraction.k` / `REDUCE@contraction.k` / `STAGE@contraction.k` |
| norm_linear, mlp_geglu | bare | `…@map.contraction.k` |
| flash | `TILE@dd` / `TILE@pj` / `REDUCE` / `STAGE` | `TILE@map.reduce.contraction.dd` / `…pj` / `REDUCE@map.reduce.kv` / `STAGE@map.reduce.kv` |
| rms_norm | `REDUCE` | `REDUCE@map.reduce.k` |
| bare reduce (`cut_cone_stat`) | `REDUCE` | `REDUCE@reduce.k` (child-tree anchor, unchanged role) |
| pointwise | `TILE` | `TILE@map` |
| root-globals | `RASTER` / `WSPEC` / `LOOPIFY` | unchanged, bare |

Every live spelling is already a valid unique suffix of its canonical form → **no YAML edit is required to
keep parsing**; the migration script (below) only *canonicalizes* stored spellings, mechanically.

## PLACE restoration under the new schema

`PLACE@<child-path> = cut | fuse` on in-tree seams. Old sites map to:

| Old | New | Semantics |
| --- | --- | --- |
| `PLACE: fuse` (flash) | `PLACE@map = fuse` | projection seam stays in-kernel |
| `PLACE@cone: cut` | `PLACE@map.contraction.a = cut` | cone → stat kernel + scale kernel + plain matmul |
| (new) | `PLACE@map = cut` + `REDUCE@map.reduce.k = g<n>k` | 3-kernel split reduce: partial → combine → separate elementwise projection (previously inexpressible) |
| `PLACE@fin: fuse` | **out of scope** (graph edge: inline finalize into consumers) | phase 5, if ever |
| `PLACE@stat: sink` | **out of scope** (graph edge: producer↔consumer tap) | phase 5, if ever |

Invariants carried over from the old design:

- **`cut` is the default on every seam; `fuse` (and any non-default cut) is evidence-only** — an unseeded
  site never pays. Exception: seams whose fused form IS the recognized default kernel (the `map` projection
  seam, the cone) default to the recognized form (`fuse`); verify each seam's pre-wipe default against the
  deleted realizers' gates in git history (`git show <pre-wipe>:.../020_cut_edge.py` etc.) before wiring.
- Accuracy: a cut materializes the seam value (f32 state for reduce seams, like `030_split_reduce`'s
  workspace); numerics gates unchanged.
- MIMO (#433) is the mechanism for a cut child that shares operands with siblings (multi-output node instead
  of recompute).

## Prerequisite: full nodification

Path addressing only reaches nodes. Two flat forms must become nodes first:

1. The computed-A cone's stat reduce (`Contraction.a_operand` body's annotated `Loop`) → a `Reduction` node
   under a new `a` edge (`stat_prologue()` already finds the seam). This unlocks addressing the norm→linear
   same-axis collision (`REDUCE@a.reduce.k` vs `REDUCE@contraction.k`) and the cone cut's child schedule.
2. `030_split_reduce`'s sliced partials riding flat `Map` bodies → keep as-is initially (they are
   post-rewrite artifacts, never key targets), but assert they never receive keys.

## Plan

### Phase 1 — codec core (`knob.py`, `search/keys.py`, `search/space.py`)

- Path type: parse/format `FAMILY@seg.seg….axis`, suffix-match resolver (generalize `resolve_axis` →
  `resolve_path(family, key, tree)`), per-level uniqueness check + ordinal emission.
- `family_of` / `axis_of` / `family_value` / `_FAMILY_ORDER` read through paths (family prefix unchanged, so
  featurizers and the prior's pooled reads survive; `axis_of` returns the leaf).
- Tree walker in `ir/tile/ops.py`: enumerate `(path, node, axis)` triples off a `TileOp.op` — the single
  source both the stamp helpers and the resolver use.
- Verify: unit tests — round-trip, suffix resolution incl. ambiguity errors, ordinal cases, and a
  compat test that every knob dict in ALL golden YAMLs (not just gemma) resolves against its kernel kind's
  tree shape unchanged.

### Phase 2 — stamp sites (`_schedule.py`, `010_recognize.py`)

- `_option` / `_at(REDUCE, raxis)` and the TILE/STAGE stampers emit canonical full paths (from the walker),
  not bare axis names. Evidence reads stay suffix-tolerant, so old DB/prior rows keep matching.
- Nodify the cone stat reduce (prerequisite #1); re-run the norm_linear/geglu fork enumeration and confirm
  identical option sets (keys re-spelled only).
- Verify: `emmy compile --golden <one per kind> --ir tile` on 4090+5090 golden sets deploys the recorded
  config for every kind (the pin-only offer audit, #435, must stay green).

### Phase 3 — PLACE realizer (new pass, replaces `020_cut_edge`)

- One generic edge-cut pass: given `PLACE@<path> = cut`, split the tree at that seam into producer/consumer
  kernels — materialize the seam value (MIMO node when operands are shared), re-recognize children.
  Seams in scope: `map` (projection off a reduce/contraction), `contraction.a` (the cone). The split-reduce
  finalize `map` seam composes with `REDUCE=g<n>k` for the 3-kernel form.
- `010_recognize` enumerates PLACE rows per seam (evidence-gated, default per seam as verified from
  pre-wipe gates); greedy pin precedence: exact path > suffix > bare, mirroring `narrow_at`.
- Verify: A/B the restored cuts against this branch's baseline on the recorded shapes —
  `cut_cone_stat.m256` + `cut_cone_scale.m256` pair ≈ 3.8 µs vs fused `F.rms_norm` 6.0 µs (5090 comments in
  the golden YAML); rms_norm `REDUCE@map.reduce.k=b256` unchanged; new 3-kernel split-reduce compiles and
  passes accuracy on the `--golden rms_norm.k3840` shape.

### Phase 4 — evidence migration + goldens

- Script: canonicalize stored knob spellings in golden YAMLs + tune DB rows + online prior (mechanical,
  suffix → full path; commented-out PLACE entries re-keyed to the new spellings and re-enabled ONLY behind
  a fresh `--ab` verification per entry — do not trust pre-wipe µs).
- Re-seed the retired PLACE goldens by hand-pinned `--ab` (the manual sweep method): flash `PLACE@map`,
  cone cuts on norm_linear/geglu at the recorded shapes, both cards.
- Verify: `eval golden` pin-only offer audit green; serving twins deploy from tier (decode TPOT / TTFT
  parity numbers within noise of the pre-wipe baselines recorded in the YAML comments).

### Phase 5 — deferred (graph-level placement)

`PLACE@fin=fuse` (consumer-inline) and `PLACE@stat=sink` (producer tap) cross graph edges — out of the
in-tree namespace. Re-introduce later as consumer-anchored graph rewrites if evidence warrants; the only
live golden impact today is one commented-out `stat: sink` entry (and `fin=fuse` was refuted e2e).

### Cleanup

- Docs: `pipeline/ARCHITECTURE.md` (knob/fork system), `passes/ARCHITECTURE.md` (tile lowering — PLACE as
  edge property), CLAUDE.md tile-lowering blurb if the node vocabulary changes.
- Delete this plan when landed.
