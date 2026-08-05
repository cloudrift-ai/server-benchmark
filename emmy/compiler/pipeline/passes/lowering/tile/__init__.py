"""Tile-IR lowering: ``LoopOp`` → ``TileOp``.

1. **Recognize** (``010_recognize``) — the Loop-IR → Tile-IR boundary. Fuse flash attention, fuse
   online softmax, annotate each reduce ``Loop`` with its ``AxisRole`` (the algebra is the body),
   then **lift** the kernel to a ``TileOp`` carrying ONE op-tree — a ``Fold`` /
   bilinear ``Fold`` term — with an **unmapped** placement (its parallel ``free`` axes). After this
   nothing downstream traffics in ``LoopOp``. The ``_flash`` / ``_softmax`` helpers hold the
   pattern matchers, ``_atomize`` the algebra→atom binding, ``_cut`` the placement cut.
2. **Schedule** (``020_schedule``) — decide that op's ``place`` (free axes → grid) and its per-node
   ``schedule`` slices, and offer them as a fork. ONE generic row enumerator (``_schedule``): the
   kernel's single ``WORK`` inventory is CHOSEN first, then a recursive walk of the site tree hands
   each family its typed candidate slices — dispatched on two stored-param predicates of the node
   (``axis is None`` / ``is_contraction``), NEVER the ``AxisRole``, which stays a loop annotation
   and a materializer read — the domain being the ``search/space.py`` move catalog. Each row spells
   every family ONCE at its canonical path key, and one ``build_fork_tree`` turns them into the lazy
   fork. Every role emits rows; none builds a ``TileOp`` directly.
3. **Split** (``030_split_reduce``) — consume a cross-CTA ``GRID`` stage (``ReducePlan.needs_split``)
   as a **graph rewrite**: a partial kernel reduces each CTA's slice of the reduce axis and
   either ``atomicAdd``\\ s its (additive) state into the output (one kernel) or writes it to a
   ``__partial`` workspace folded by a sibling finalize kernel (the carrier's
   ``combine_states`` over the split axis — additive ``sum`` / split-K matmul AND the twisted
   flash ``(m, l, O)`` split-KV). The schedule carries the partition; the graph carries the
   kernel count, so ``lowering/kernel`` only ever sees single-launch kernels.

**The schedule step is INCOMPLETE.** It covers every term whose operand edges are all MATERIALIZED —
the pointwise cell + the register-strip term reading, the reduce partition, and the contraction
(scalar + warp tiers, staging, split-K, raster) — plus the COMPUTED operand edge (the fused
norm→linear / gate⊗up cone), which arrives as a ``_site_values`` entry under the term-reading union
rather than as an emitter of its own — the flash streaming pair included, whose two sites
(the hoisted score edge and the derived P@V) enumerate their own halves of the twisted geometry and
reconcile at the stream. A term the enumeration cannot schedule yields NO rows and ``020`` leaves it
unmapped — the guardrail contract, ``[]`` and never a raise. The knob / path codec, the move catalog
and the whole ``lowering/kernel`` materializer are frozen — they are the contract the enumerator
meets.

Recognition reads algebraic structure; scheduling is geometry; materialization back to
loop IR happens in ``lowering/kernel`` — so the tile passes work purely with algebra
primitives.
"""
