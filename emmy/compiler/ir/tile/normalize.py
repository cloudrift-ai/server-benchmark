"""The tree-wide canonical forms of a Tile IR Fold tree.

Each :class:`Fold` owns every canonical form a node can state about itself at formation — its
lambda bodies' statement order, a bilinear term's A-first operand orientation. What remains here
is what only the WHOLE tree can decide: an identity projection dissolves into the one operand it
re-exposes (a closing rewrite can leave one behind, and it is what makes two occurrences of the
same computation compare unequal), and same-value cones become ONE shared object.

INVARIANT — normalization ends with same-value cones (alpha-equal, identical captures and
interface names) as ONE shared object (:func:`_share_common_cones`). Object identity is how the
placement machinery recognizes that two consumption sites read one value, so a rewrite that
copies a cone (the close rewrites do, by design) is only sound because this final pass restores
the sharing. Recompute elimination is a
Tile-level placement concern built on that identity: a duplicated value becomes one seam, and a
composed cut materializes it once for every reader. Do NOT patch recompute downstream — a Loop IR
fusion or emission workaround sees one kernel at a time and cannot know two kernels re-derive the
same value; fix the sharing or the seam offer here instead.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.ir.pure import Fold
from emmy.compiler.ir.stmt import Body
from emmy.compiler.structural import instance_memo


def _passthrough(node: Fold) -> Fold | None:
    """The single operand an identity projection merely re-exposes, or ``None``.

    A pass-through is shape noise — a closing rewrite can leave one behind — and it is what makes
    two occurrences of the same computation compare unequal, so normalization dissolves it
    wherever a projection is formed or revisited."""
    if node.axis is not None or node.lift.body or len(node.operands) != 1:
        return None
    (operand,) = node.operands
    if isinstance(operand, Fold) and node.lift.results == tuple(param for param, _, _ in node.bindings):
        return operand
    return None


def _normalize_fold(fold: Fold) -> Fold:
    operands = tuple(_normalize_fold(edge) for edge in fold.operands)
    node = replace(fold, operands=operands) if operands != fold.operands else fold
    if node.axis is None and (collapsed := _passthrough(node)) is not None:
        return collapsed
    return node


def _share_common_cones(root: Fold) -> Fold:
    """Restore object sharing between same-value cones — the tree-wide half of canonicalization.

    Fusion and the close rewrites inline one value into every consumption site, so a traced value
    consumed twice (attention's softmax statistics, read by the weight cone and the epilogue)
    reappears as equal-but-distinct copies. Everything downstream keys on object identity —
    ``cuttable_seams`` groups occurrences by it, ``realize`` replaces cut values by it — so a
    severed sharing silently turns one value into per-site recompute that no schedule can undo.
    This walk hash-conses every Fold bottom-up: copies UNIFY onto the first occurrence in walk
    order when they are alpha-equal with identical captures (the bucket key adds ``deps`` — the
    K-cone family, alpha-equal under DIFFERENT captures, stays value clustering's job) and
    identical interface names (``defines`` — what sibling members and the consuming lift read),
    so a copy that differs only in internal binder spelling still collapses where plain
    structural equality would silently sever the sharing. Emission is untouched in shape:
    lowering walks tree positions, and every position holds a term of the same value (a unified
    representative may change internal spelling). Identity-preserving off the replacement spine,
    like ``_replace_fold``."""
    canon: dict[tuple, Fold] = {}
    seen: dict[int, Fold] = {}

    def member(stmt):
        if isinstance(stmt, Fold):
            return visit(stmt)
        nested = stmt.nested()
        if not nested:
            return stmt
        bodies = tuple(Body(tuple(member(child) for child in body)) for body in nested)
        unchanged = all(
            len(body) == len(original) and all(piece is child for piece, child in zip(body, original, strict=True))
            for body, original in zip(bodies, nested, strict=True)
        )
        return stmt if unchanged else stmt.with_bodies(bodies)

    def visit(node: Fold) -> Fold:
        if id(node) in seen:
            return seen[id(node)]
        operands = tuple(visit(edge) if isinstance(edge, Fold) else edge for edge in node.operands)
        body = tuple(member(stmt) for stmt in node.lift.body)
        current = node
        if any(piece is not edge for piece, edge in zip(operands, node.operands, strict=True)):
            current = replace(current, operands=operands)
        if any(piece is not stmt for piece, stmt in zip(body, node.lift.body, strict=True)):
            current = replace(current, lift=replace(current.lift, body=Body(body)))
        # The canonical form IS the key: a dict lookup and an alpha-equality test are the same
        # operation, so there is no prefilter bucket and no pairwise rescan. The exposed names
        # ride beside it: a consumer's lift reads an edge's results BY NAME, so two values that
        # differ only in what they expose stay distinct — unifying them would re-spell every
        # consumer of the copy (softmax's two reads of one row, spelled ``in0`` and ``in1``).
        # The free coordinates ride the key by NAME: canonical renumbers them, so a cone over ``x[q, k]``
        # and one over ``x[m, k]`` spell the same canonical form and are two values all the same.
        prior = canon.setdefault((current.canonical(), current.exposes, frozenset(current.free_axes)), current)
        seen[id(node)] = prior
        return prior

    return visit(root)


def normalize_fold_tree(root):
    """Normalize a complete Tile IR tree bottom-up; ``None`` placeholders pass through.

    The reached fixpoint is STAMPED on the result (an
    :func:`~emmy.compiler.structural.instance_memo`): the term is immutable and the rewrite
    idempotent, so a reconstruction answers without re-walking — on a large fused tree the
    re-verification, once per ``TileOp`` construction, is what turns the pipeline quadratic."""
    if not isinstance(root, Fold):
        return root
    if instance_memo(root, "_memo_normal"):
        return root
    normalized = root
    while True:
        # One pass is not always the fixpoint (a collapse can expose the next pass's move), and the
        # stamp must mean the REACHED fixpoint, so iterate here rather than relying on the next
        # construction to finish the job.
        again = _normalize_fold(normalized)
        if again == normalized:
            break
        normalized = again
    result = root if normalized == root else normalized
    result = _share_common_cones(result)
    instance_memo(result, "_memo_normal")[()] = True
    return result


__all__ = [
    "normalize_fold_tree",
]
