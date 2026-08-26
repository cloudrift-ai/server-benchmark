"""Kernel identity — **the** home. Every question of the form "are these two kernels the same?"
is answered here or by something this module names, and nowhere else.

Identity has more than one useful meaning, and the bug this module exists to prevent is a second
codepath deriving one of them by hand. The meanings, and who answers each:

| question | answer | excludes |
| --- | --- | --- |
| is this the same ALGEBRA? | :meth:`TileOp.structural_key` (``_key.py``) | sizes, dtypes, placement, schedule, stores |
| is this the same KERNEL to deploy against? | :func:`deploy_identity` | knobs, hints, live pins |
| would this enumerate the same SCHEDULE SPACE? | :func:`pool_key` | nothing the enumeration reads |
| is this the same STRUCTURE for the DB / prior? | ``IdentityStrategy.op_sig`` (``passes/identity.py``) | — |

The term digest is deliberately narrow: it canonicalizes α-renaming, buffer spelling **and sizes**
away, so two kernels differing only in extent share it. That is right for the algebra and wrong
for everything downstream, which is why each coarser identity folds the excluded facts back in
EXPLICITLY. The fingerprints below are those facts, one function each, and they are the reason
this module is not just two digests.

**Adding a fact.** When something reads a `TileOp` fact the term does not carry and lets it change
what it produces, that fact belongs in a fingerprint here and in whichever identity is affected.
Deriving it at the call site instead is how ``pool_key`` came to omit per-axis extents while
``deploy_identity`` carried them: an ``8x64 @ 64x512`` matmul and a ``512x64 @ 64x8`` one agree on
the term, on the dtypes, and on the lossy ``S_ext_*`` knob summary (count / product / max), so they
shared one pool entry while their spaces were 57442 and 8280 candidates.

**Why this lives in ``ir/tile``.** Every reading here is a pure function of a `TileOp`. That keeps
the module below the pipeline (``ir/`` never imports ``pipeline/``), so the scheduler, the golden
tier and the greedy policy can all import it directly — they previously reached into
``lowering/tile/_schedule`` through deferred imports to break a cycle. The ONE ambient input,
the live env pins, is a required argument of :func:`pool_key` rather than a hidden read, so a
caller cannot forget it and no second spelling can appear.
"""

from __future__ import annotations

from emmy.compiler.dim import DEFAULT_SEQ_HINT
from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.stmt import Load
from emmy.compiler.ir.tile.ir import TileOp
from emmy.compiler.structural import digest

__all__ = [
    "deploy_identity",
    "dtype_fingerprint",
    "extent_fingerprint",
    "hint_extent",
    "hint_fingerprint",
    "pool_key",
    "shape_fingerprint",
    "store_fingerprint",
]


def _dims(shape) -> tuple[str, ...]:
    """A buffer shape rendered hint-free — a static dim as its integer, a symbolic one as ``sym``."""
    return tuple(str(d.as_static()) if d.is_static else "sym" for d in shape)


def hint_extent(ax) -> int:
    """An axis's static extent, or its ``Dim`` hint when symbolic."""
    e = ax.extent
    return e.as_static() if e.is_static else (e.hint or DEFAULT_SEQ_HINT)


def _walk_axes(tile: TileOp, note) -> None:
    """Visit the free grid axes, then every ``Fold`` axis, in stored walk order."""

    def walk(node) -> None:
        if not isinstance(node, Fold):
            return
        note(node.axis)
        for edge in node.operands:
            walk(edge)
        for stmt in node.lift.body:
            walk(stmt)

    for axis in tile.place.free:
        note(axis)
    walk(tile.op)


def hint_fingerprint(tile: TileOp) -> tuple[int, ...]:
    """The hint-resolved extents of the term's SYMBOLIC axes, in walk order.

    ``Dim.hint`` is deliberately excluded from identity (``Op.cache_key`` stays hint-independent),
    but a schedule SIZES against it (:func:`hint_extent` → which coop bands the reduce extent can
    feed), so a key over the schedule space carries it: two same-key ops traced at different
    ``--seq-len`` hints enumerate different spaces.
    """
    out: list[int] = []

    def note(ax) -> None:
        if ax is not None and not ax.extent.is_static:
            out.append(hint_extent(ax))

    _walk_axes(tile, note)
    return tuple(out)


def extent_fingerprint(tile: TileOp) -> tuple[str, ...]:
    """Every axis extent in walk order — the free grid, then each ``Fold`` axis.

    A static extent renders as its integer, a symbolic axis as the bare ``sym`` marker (identity
    stays hint-free — a symbolic record is the symbolic kernel's identity at every hint). Needed
    by every identity coarser than the term, because the α-invariant algebra digest canonicalizes
    sizes away: without it every same-algebra cone on a card shares one key, and the fastest record
    of ANY shape decides them all (an m32 scalar row deploying onto every M).
    """
    out: list[str] = []

    def note(ax) -> None:
        if ax is not None:
            out.append(str(ax.extent.as_static()) if ax.extent.is_static else "sym")

    _walk_axes(tile, note)
    return tuple(out)


def dtype_fingerprint(tile: TileOp) -> tuple[str, ...]:
    """The operand dtypes as a schedule reads them — each term ``Load``'s buffer dtype in first-use
    walk order, plus the output dtypes.

    NAME-FREE (a buffer's graph id never enters), so two same-shape kernels still share a key,
    while an f16 and an f32 trace of one shape — equal terms, different atom eligibility — key
    apart. Explicit rather than via the stamped ``S_dtype_*`` knobs because not every path that
    reaches scheduling carries the stamps.
    """
    seen: set[str] = set()
    out: list[str] = []

    def note_stmt(s) -> None:
        if isinstance(s, Fold):
            walk(s)
            return
        if isinstance(s, Load) and s.input not in seen:
            seen.add(s.input)
            t = tile.inputs.get(s.input)
            out.append(str(t.dtype) if t is not None else "?")
        for b in s.nested():
            for c in b:
                note_stmt(c)

    def walk(node) -> None:
        if not isinstance(node, Fold):
            return
        for e in node.operands:
            note_stmt(e)
        for s in node.lift.body:
            note_stmt(s)

    walk(tile.op)
    return (*out, "->", *(str(t.dtype) for t in tile.outputs.values()))


def _buffers(tile: TileOp) -> list:
    """Every buffer a schedule reads, in the ONE walk order: each term ``Load``'s buffer in
    first-use order, then the outputs. Shared by :func:`dtype_fingerprint` and
    :func:`shape_fingerprint` so the two can never disagree about which buffer a position names."""
    seen: set[str] = set()
    out: list = []

    def note_stmt(s) -> None:
        if isinstance(s, Fold):
            walk(s)
            return
        if isinstance(s, Load) and s.input not in seen:
            seen.add(s.input)
            out.append(tile.inputs.get(s.input))
        for b in s.nested():
            for c in b:
                note_stmt(c)

    def walk(node) -> None:
        if not isinstance(node, Fold):
            return
        for e in node.operands:
            note_stmt(e)
        for s in node.lift.body:
            note_stmt(s)

    walk(tile.op)
    return [*out, None, *tile.outputs.values()]  # ``None`` separates the operands from the results


def shape_fingerprint(tile: TileOp) -> tuple[tuple[str, ...] | str, ...]:
    """Every buffer's SHAPE, in :func:`_buffers` order. NAME-FREE, like the dtypes beside it.

    Not implied by :func:`extent_fingerprint`: the axes say how far a loop runs, the shape says how
    the buffer SPELLS that coordinate. A re-fused split axis reaches its buffer as a dim pair
    (``[…, f/Q, …, f%Q]``), which the fragment store can address only under a divisibility rule
    (``_legality.warp_split_store``) — so a ``(128, 128)`` output and a ``(4, 32, 128)`` one over
    the same iteration space do not offer the same tiers. They were reaching identical
    ``deploy_identity`` AND ``pool_key`` over spaces of 50538 and 10284 candidates, which put a
    golden measured on the flat kernel onto a kernel that cannot realize the row it names.
    """
    return tuple("->" if t is None else _dims(t.shape) for t in _buffers(tile))


def store_fingerprint(tile: TileOp) -> tuple:
    """The kernel-boundary stores' ADDRESSING — what ``TileOp.stores`` contributes that the term
    does not carry.

    Per store in order: the index expression, whether it is an ``atomicAdd``, its stored width, and
    the sweep axis's extent when it rides an output ``Loop``. Buffer and SSA names are excluded —
    those are spelling, and identity is name-free — while the index EXPR is kept whole, since it is
    exactly what :func:`~_legality.warp_split_store` reads to decide addressability.

    ``TileOp.structural_key`` excludes the stores by design (they are a kernel-boundary fact beside
    ``place``, not algebra), so any identity coarser than the term folds them back in here.
    """
    return tuple(
        (
            tuple(repr(e) for e in store.write.index),
            store.write.atomic,
            store.write.width,
            None if store.sweep is None else (str(store.sweep.extent.as_static()) if store.sweep.extent.is_static else "sym"),
            store.unroll,
        )
        for store in tile.stores
    )


def deploy_identity(tile: TileOp) -> str:
    """The verified-tier join key — what the kernel IS.

    The Fold tree's α/buffer-invariant algebra digest (:meth:`TileOp.structural_key`) folded with
    the operand/output dtype fingerprint and the axis extents the term deliberately omits. A golden
    record derives the SAME key from its own persisted program through the shared total lift
    (``_fromloop.lift_loop_op``), so the join is exact structural identity — no classified shape,
    no matching heuristic.

    Unlike :func:`pool_key` it excludes knobs, symbolic hints and live pins: identity is what the
    kernel is; the strict row decode (exact spelled-row equality) is what guarantees a record still
    realizes.
    """
    return digest(
        tile.structural_key(),
        dtype_fingerprint(tile),
        extent_fingerprint(tile),
        shape_fingerprint(tile),
        store_fingerprint(tile),
    )


def pool_key(tile: TileOp, *, pins: str) -> str:
    """The schedule-space key — would this enumerate the same candidates?

    ``tile.cache_key()`` covers the term and the knobs; every OTHER input the enumeration reads is
    folded in explicitly, because ``structural_key`` excludes it by design — the operand/output
    dtypes (atom eligibility), the per-axis extents, and the symbolic-axis hints. ``pins`` is the
    live env-pin fingerprint (``knob.schedule_pin_fingerprint``), passed in rather than read here
    so this module stays a pure function of a ``TileOp`` and below the pipeline layer.

    Target facts (smem cap, TMA, the f16acc gate) need no part: the pool cache lives ON the
    ``Context``, so one instance never spans two fact sets.
    """
    return digest(
        tile.cache_key(),
        dtype_fingerprint(tile),
        extent_fingerprint(tile),
        shape_fingerprint(tile),
        store_fingerprint(tile),
        hint_fingerprint(tile),
        pins,
    )
