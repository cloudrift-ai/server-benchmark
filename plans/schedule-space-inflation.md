# The blocked carrier's schedule space is 35 million times bigger than the term it replaced

Working note. Measured on `feature/blockify-twisted-carrier` at the rebase onto `28a9f4f48`.

## The measurement

`F.scaled_dot_product_attention` f16 `(1, 8, 512, 128)`, sm_120, domains from `project_classic`
alone (no walk, no pool):

| | schedule space | node sites | contraction sites |
| --- | --- | --- | --- |
| unblocked term | 6.5 × 10⁹ | 4 | 1 |
| blocked term | 2.3 × 10¹⁷ | 9 | 3 |

`emmy compile --ir tile` on that shape takes **55 s**. The unblocked term reaches the same stage
in a few seconds.

Per-site, blocked:

```
map.1/twist                                27   (the carrier's REDUCE)
map.1/twist.1/reduce.1/inner              456   the PIVOT's score        <- Q·K
map.1/twist.2/inner                       460   the expectation channel  <- P·V
map.1/twist.2/inner.1/map.1/inner         456   the WEIGHT's score       <- Q·K again
```

## Where it comes from

Blocking turns one contraction site into three, and the space is their product. Two of the three
are the SAME VALUE.

`map.1/twist.1/reduce.1/inner` and `map.1/twist.2/inner.1/map.1/inner` are α-equal copies of the
block's `Q·K`, differing only in which binder they read — the pivot's pass binds `k_p`, the weight's
binds `k_c`. The emitter already knows this: `_fold_staged` rewrites the whole block into one
spelling and the two become one term, which is exactly what makes the score run ONCE per block
rather than twice. It then takes the weight's copy and never looks at the pivot's.

So one of those 456-choice domains is enumerated for a tile no kernel will ever read, and a row may
put a different tile on each of two copies of one value. That is a ~456× factor, and it also costs
the greedy a descent level and the codec two extra keys per row.

## What to do about it

**Option 1 — projection side.** A contraction site α-equal (modulo the block binder) to another
site inside the same block is offered ONE choice. Local to `_options`, no term change, and the
α-comparison already exists in `_fold_staged` (`binder` + `_reclosed`). Removes the redundant
factor and the dead key.

**Option 2 — term side.** The pivot and the weight read one shared score object, so there is one
site. Cleaner — the redundancy stops existing rather than being filtered — but it needs both to
bind the same coordinate, and the memo's reason for separate binders is that `Fold.lower` places
two folds over one axis in one loop, which would have the weight read a pivot that has not
finished accumulating. `_scope`'s dependence guard already refuses that fuse, so the reason may no
longer hold; that has to be checked rather than assumed.

**Neither is the whole factor.** After the duplicate goes, the space is still ~5 × 10¹⁴: the third
site (`P·V`) is genuine, and its STAGE edges multiply on top. The block's two cross-site equations
already cut the per-site domains hard (930 → 460 and 2282 → 456); what is left is the ordinary cost
of a kernel with three tiled contractions instead of one.

**A budget is not the fix.** The greedy already samples a budgeted subset when the pool is too
large, which is why the compile completes at all. The problem is that the subset is drawn over a
space where a whole dimension is noise.

## Not measured yet

- Whether the compile time is the domain product or the per-row work — 55 s for one kernel is a lot
  even for a sampled descent, and the profile has not been taken.
- The same numbers on the shapes that matter for serving (gemma4's heads), where the stream is
  longer and the block count higher.
