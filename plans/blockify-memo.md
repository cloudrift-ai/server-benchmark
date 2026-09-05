# Blocked SDPA: the three block loops, and the round trip that finally merged them

Working note from the blockify prototype (`feature/blockify-twisted-carrier`); never reference it from durable docs or
code. Rewritten 2026-09-04 after instrumenting — the first draft's premise ("the merge never receives the loops") was
wrong, and so was its advice to revert. Extended the same day: defect #3 is CLOSED, by a route none of the three
candidates below named.

## The symptom

`025_block` rewrites attention's twisted carrier into an outer block fold over three inner folds — the block pivot
(max), the P·V expectation, and the denominator. Each inner fold re-derives the score, so the emitted CUDA carries
three `a3` loops per block: `Q·K` runs 3× where FlashAttention-2 runs it 1×. The pivot must stay separate (the weight
reads its finished accumulator); the two channels should collapse to one loop.

## Three defects, in the order they bite

### 1. `Lambda.cone` resolved a param read through the combine's own state re-binding — FIXED

A stored combine is not SSA in its states: it reads `acc1` and writes `acc1` back on the way out (`acc1 =
copy(acc1__o__gn)`). `Body.backward_cone` resolves by name, so β's cone swallowed that trailing write, and
`_beta_cone` shipped it inside the weight. Downstream `eliminate_copy_aliases` then folded the pivot read into a
self-read and the kernel emitted

```c
float acc1__o__gn = fmaxf(acc1__o__gn, v1);   // uninitialized
```

— every blocked attention kernel was wrong, before any question of merging. `Lambda.cone` now takes the cone over the
body without the statements that re-bind a param; the kernel emits `expf(v1 - fmaxf(acc1__blk, v1))`. The same write
also masked the pivot→channel dependence from every uses-vs-defs reading, which let the repaired merge fuse the pivot
with the denominator until this was fixed.

### 2. The merge's gates could not fuse two alpha-equal copies of one cone — FIXED

The three loops ARE siblings in one `Body` (instrumented: `[Loop(a2_p,[acc1__blk]), Loop(a2_p,[acc5__sum__blk]),
Loop(a2_p,[acc3__blk]), …]` after `unify_sibling_reduce_axes`). The pairwise scan reached them every time; the gates
refused:

- the dependence gate asked `merged_defs & t.ssa_uses`, so a name `t` binds *itself* counted as a read — and two
  copies of one cone always share every spelling. Now one reading, `free_names(t)`, used by both the dependence gate
  and the between-statements gate.
- any def collision refused outright. A collision is not a dependence: the incoming body's copies rename apart. Only a
  name the incoming loop still binds after it closes cannot be renamed — `Loop.render` declares those (its immediate
  carriers, plus a nested `seed=False` loop's) ahead of the loop, everything else lives in the block it closes. That
  is `_carried_out`.
- `Loop.seed` was silently dropped when the merged loop was rebuilt; it is now carried and gated on.

`unify_sibling_reduce_axes` also had to see an affine index (`o·B + i`) to give the three loops one axis name — the
anchor and coefficient ride the key so `o·B + i` and `o·B + 32 + j` stay apart.

Verified end to end on `F.scaled_dot_product_attention(1,4,64,32)`: `DBG-MERGED a2_p ['acc5__sum__blk',
'acc3__blk'] << ['acc3__blk']`, the pivot refused. `tests/compiler/ir/stmt/test_merge_sibling_reduce_loops.py`
carries the scope shape as a case; it fails on the pre-fix pass.

### 3. The merge was not on the path that emits the kernel — CLOSED

`normalize_body` runs from `LoopOp.__post_init__` and `Body.structural_key`. `TileOp.__post_init__` normalizes the
TERM (`normalize_fold_tree`); `010_materialize` → `_factor.factorize` builds the `KernelOp` straight from
`Fold.lower`. So every merge observed above happened on a digest or on a Loop-IR op, and the emitted SDPA kernel
still carried three block loops. (The module docstring claiming `TileOp.__post_init__` runs these passes is what
seeded the first draft's wrong hypothesis; it is corrected.)

The three candidate closures listed here before — renormalize the materialized kernel body, teach `Fold.lower` to
share a reduce loop between alpha-equal binders, or emit one channel fold with a componentwise combine — are all
superseded by a fourth: put the kernel back in the dialect that already owns the merge.

## How it closed: `025_block` splices a `LoopOp` and the tile pass restarts

`025_block` no longer rebinds the blocked `TileOp` in place. It takes `TileOp.loop_body`, constructs a `LoopOp` —
whose `__post_init__` runs `normalize_body`, which is where the merge lives — and splices it as a `Graph` fragment.
The splice bumps `Cursor.n_applied`, so when the rule scan wraps it restarts at rule 0 of `lowering/tile` instead of
advancing, and `010_lift` re-derives the Fold tree from the merged nest. No engine change: this is the same mechanism
`030_cut` uses, and `pass_idx` never moves backwards, so the loop passes do not re-run (nor do they need to —
canonicalization is `LoopOp.__post_init__`'s, not `loop/canonicalize`'s).

Three things had to be fixed to make the round trip legal. All three are pre-existing and fire identically on the
UNBLOCKED SDPA tree; none had ever been observed, because nothing had constructed a `LoopOp` from a post-twist body.

- **`_validate` did not know `Const` is a binding site** (`ir/loop/ir.py`). `Const` is introduced by the twisted
  rewrite in `lowering/tile` — "the denominator's `1` must be a def the lift can return" — so no `LoopOp` had ever
  carried one, and every twisted body read an undefined name.
- **`hoist_loop_invariants` hoisted a consumer above the loop that defines what it reads**
  (`ir/stmt/normalize.py`). `v6 = reciprocal(acc3)` is invariant in the head-dim axis `a5` and moved out of it, but
  `acc3` is exported by a reduce loop PINNED inside `a5` (the value sweep reads `V[…,a5]`). The pass docstring's claim
  that axis-dependency closure makes an ordering check unnecessary does not hold when a pinned block exports an
  invariant name. Fixed by pinning any candidate that reads a name the remaining body still binds, iterated. Cannot
  move kernel identities: `Body.structural_key` normalizes with `hoist=False`.
- **`scan_from_loop` refused a `base`-`Accum`** (`lowering/tile/_fromloop.py`). That spelling IS the twisted carrier —
  `Fold.merge` lowers each component to `name = op(base, value)` with the rescale as `base` — so the lift could not
  read back what `Fold.lower` had just written. `_combine_from_merge` is the inverse: the rescale temps are the
  statements transitively reading a carried state, each `Accum` re-reads as the assignment it folds, original program
  order preserved (a combine is not SSA in its states). The injection is NOT the `Accum`'s `value`: a twisted merge
  rescales the incoming side first, so that operand names `acc__blk · β`, not the component. It is recovered as what
  the combine reads and does not define, in first-read order, and bound positionally as the combine's second operand
  — the contract `Fold.merge` re-applies at `lift.results`. A planar fold takes neither branch and reads back
  byte-identically to what it lowered from.

## What is measured

`F.scaled_dot_product_attention` f16, `EMMY_BLOCK=64`, `EMMY_REDUCE=` (serial outer), RTX 5090.

- **The round trip completes.** One `TileOp` in the terminal graph, no leftover `LoopOp`, and the outer fold reads
  back `twisted=True` — the carrier survives lowering and re-lifting.
- **The emitted kernel loses two loops**: `emmy compile --ir cuda | grep -c 'for ('` is **7 at HEAD, 5 with the round
  trip**, at both (1,4,128,32) and (1,8,512,64). The merge reaches the emitted kernel, which is exactly what #3 was.
- **The re-lifted tree** at (1,4,128,32): the twisted outer fold over a single-state pivot fold (whose score cone
  reads as a contraction) and a TWO-state merged channel fold (whose score cone also reads as a contraction). So the
  warning recorded against the third candidate above — that one channel fold with a componentwise combine likely
  costs the bilinear reading — does **not** hold: `as_contraction` is multi-channel by design (one shared ⊕, every
  lift result a two-argument product reading `operands[0]`), and the merged fold passes it.
- **Accuracy passes** at (1,4,128,32) and at the prototype commit's (1,8,512,64): `emmy run` exits 0 silently.
- 149 pass across `tests/compiler/ir/stmt/`, `tests/compiler/ir/pure/`, `tests/compiler/passes/test_twisted_rewrite.py`
  — including `test_sdpa_score_contraction_reaches_the_mma_tier`, which is red with the splice and no lift fix.

## Open

- **No mma reaches the emitted CUDA, and that is NOT the round trip.** `grep -ci mma` on `--ir cuda` at
  (1,8,512,64) is **0 at HEAD and 0 with the round trip**. The prototype commit's "takes the mma tile above" does not
  reproduce at HEAD under these env settings. The tile READING is a contraction on both arms, so the gap is between
  that reading and what the schedule search picks and emits — a separate investigation, and the next one.
- **Whether `normalize_body` should iterate its merge.** Instrumented on the blocked body directly, ONE
  `merge_sibling_reduce_loops` call takes 3 block loops → 2 but leaves 3 score cones; one further round takes those
  to 2. The full pipeline lands at 2/2, so something already runs the second round — worth confirming where, since
  the pass is called once and the fixpoint is not stated anywhere.
- **`Q·K` still runs 2×, not 1×.** The pivot must finish before the channels read it, so no sibling merge can share
  that cone; this needs the chained-score realization, unchanged from the prototype commit's own gap list.

## Blast radius, partly unverified

Not run: the rest of `tests/compiler/`, the realization corpus, `make test-goldens`, `scripts/digest_kernels.py`,
the perf lane. Kernel source changes wherever the merge now fuses, so the digest A/B and the corpus are both owed
before this goes near main; recorded golden rows are expected to go stale. `make lint` clean.

## Process note

Five rounds of inference, then one instrumented run refuted the premise; a second one refuted the repair. Instrument
first, and re-instrument after each fix — the second defect was invisible until the first was gone. The same held for
the round trip: two of its three blockers were only visible once the previous one was gone, and the "does the merged
fold still read as a contraction" question that looked like the decisive risk was answered in one probe.
