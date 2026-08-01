"""Tile-IR lowering: ``LoopOp`` → ``TileOp``.

1. **Recognize** (``010_recognize``) — the Loop-IR → Tile-IR boundary. Fuse flash attention, fuse
   online softmax, annotate each reduce ``Loop`` with its ``AxisRole`` (the algebra is the body),
   then **lift** the kernel to a ``TileOp`` carrying ONE op-tree — a ``Map`` / ``Fold`` /
   ``Contraction`` term — with an **unmapped** placement (its parallel ``free`` axes). After this
   nothing downstream traffics in ``LoopOp``. The ``_flash`` / ``_softmax`` helpers hold the
   pattern matchers, ``_atomize`` the algebra→atom binding, ``_cut`` the placement cut.
2. **Split** (``030_split_reduce``) — consume a cross-CTA ``GRID`` stage (``ReducePlan.needs_split``)
   as a **graph rewrite**: a partial kernel reduces each CTA's slice of the reduce axis and
   either ``atomicAdd``\\ s its (additive) state into the output (one kernel) or writes it to a
   ``__partial`` workspace folded by a sibling finalize kernel (the carrier's
   ``combine_states`` over the split axis — additive ``sum`` / split-K matmul AND the twisted
   flash ``(m, l, O)`` split-KV). The schedule carries the partition; the graph carries the
   kernel count, so ``lowering/kernel`` only ever sees single-launch kernels.

**The SCHEDULE step between them is currently ABSENT.** Deciding a ``TileOp``'s ``place`` (free
axes → grid) and its per-node ``schedule`` slices — enumerating the ``TILE`` / ``REDUCE`` /
``STAGE`` / ``WORK`` / ``RASTER`` families and offering them as a fork — was removed to clear the
ground for a demand-driven recursive enumerator over the stored term; the hand-written whole-tree
paths it replaces (contraction rows, reduce partitions, computed-A rows, the flash form fork, pin
narrowing, the demotion backtracking) went with it. Recognition above, the knob / path codec, the
move catalog (``search/space.py``) and the whole ``lowering/kernel`` materializer are untouched and
frozen — they are the contract the replacement meets. Until it lands nothing maps a ``TileOp``, so
every compile that reaches scheduling fails; those tests ride ``tests/xfail_registry.py``.

Recognition reads algebraic structure; scheduling is geometry; materialization back to
loop IR happens in ``lowering/kernel`` — so the tile passes work purely with algebra
primitives.
"""
