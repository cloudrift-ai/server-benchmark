"""The PACKED-PAIR k-block operand reading — the NVFP4 weight's shape, recognized once.

One question, asked by three consumers that must not drift apart: the schedule's offer
(``_schedule._fill_values``), the stage resolver (``_legality.resolve_warp_stage``) and the
materializer (``kernel/_atom._staged``). Each asks :func:`match_packed_b_node` and gets the same
answer or ``None``.

It reads a SHAPE, never a checkpoint format: a packed-pair storage dtype (``logical_elems == 2``),
a data-dependent gather into a pair-value table, and a scale factor whose every ``k`` reference is
block-guarded. Any weight spelled that way is recognized; nothing here names a quantization scheme.

Kept beside the classification rather than inside it: the shape is a CONSUMER'S reading of an
already-bound contraction (``bind_bilinear`` binds the computed B as a plain projection, and this
asks what that projection contains), not one of the classification stages that build the tree.
"""

from __future__ import annotations

from dataclasses import dataclass

from emmy.compiler.ir.pure.fold import Fold, operand_body
from emmy.compiler.ir.stmt import Assign, Load
from emmy.compiler.ir.stmt.body import Body


def _idx_vars(index) -> set[str]:
    """Every free Var name across an index tuple's exprs."""
    return {v for e in index for v in e.free_vars()}


@dataclass(frozen=True)
class PackedKBlockB:
    """A computed B recognized as ``pair-decode(packed bits) x k-block scale``.

    ``bits`` is the packed-pair storage Load (its input tensor's dtype has
    ``logical_elems == 2``), ``table`` the data-dependent pair-value gather it feeds,
    ``factor`` the SSA name of the scale factor, and ``block`` the k extent the factor
    is constant on. The packed byte-slab offer consumes this: the bits stage raw, and
    the drain decodes pairs and multiplies the factor into the decoded values at the
    fragment load (one factor read per k block).
    """

    bits: Load
    table: Load
    factor: str
    block: int


def _k_block_guard(expr, k_name: str) -> tuple[bool, set[int]]:
    """Walk ``expr`` for the k-block-invariance proof: ``(k_seen_naked, guards)``.

    A ``k`` occurrence is GUARDED by ``(X + k) / B`` (a literal ``B``) when ``X`` is a
    provable multiple of ``B`` — only then is the floor constant on aligned k blocks of
    ``B``. Any shape this walk does not prove declines the match; the operand then stays
    a generic computed B, never a wrong one.
    """
    from emmy.compiler.ir.expr import BinaryExpr, CastExpr, Literal, Var  # noqa: PLC0415

    def multiple_of(e, b: int) -> bool:
        if isinstance(e, Literal):
            return isinstance(e.value, int) and e.value % b == 0
        if isinstance(e, BinaryExpr):
            if e.op == "*":
                return multiple_of(e.left, b) or multiple_of(e.right, b)
            if e.op == "+":
                return multiple_of(e.left, b) and multiple_of(e.right, b)
        return False

    def walk(e) -> tuple[bool, set[int]]:
        if isinstance(e, Var):
            return e.name == k_name, set()
        if isinstance(e, Literal) or e is None:
            return False, set()
        if isinstance(e, CastExpr):
            return walk(e.expr)
        if isinstance(e, BinaryExpr):
            ln, lg = walk(e.left)
            rn, rg = walk(e.right)
            if e.op == "/" and isinstance(e.right, Literal) and isinstance(e.right.value, int) and ln:
                b = e.right.value
                # (X + k) / B guards k iff the k-free remainder is a multiple of B.
                free = _k_free_addends(e.left, k_name)
                if free is not None and all(multiple_of(f, b) for f in free):
                    return rn, lg | rg | {b}
            return ln or rn, lg | rg
        return True, set()  # an unmodeled expr containing anything: treat as naked — decline upstream

    def _k_free_addends(e, k):
        # The additive terms of ``e`` that do not reference k; None if the k term is not
        # a bare ``Var(k)`` addend (a scaled k does not stride unit blocks).
        if isinstance(e, Var):
            return [] if e.name == k else None
        if isinstance(e, BinaryExpr) and e.op == "+":
            left, right = _k_free_addends(e.left, k), _k_free_addends(e.right, k)
            if left is None or right is None:
                return None
            return left + right
        naked, _ = walk(e)
        return None if naked else [e]

    return walk(expr)


def match_packed_kblock_b(cone: list, k_name: str, inputs) -> PackedKBlockB | None:
    """Recognize the packed-pair k-block shape in a computed-B cone.

    The shape (the NVFP4 speller's lowered form): a packed-pair bits Load feeds an
    index copy, a pair-table gather reads it by data-dependent index, and the final
    multiply combines the gathered value with a factor whose every ``k`` reference is
    block-guarded (:func:`_k_block_guard`). Everything else returns ``None``.
    """
    if not cone or inputs is None:
        return None
    loads = [st for st in cone if isinstance(st, Load)]
    packed = [ld for ld in loads if getattr(inputs.get(ld.input), "dtype", None) is not None and inputs[ld.input].dtype.logical_elems == 2]
    if len(packed) != 1:
        return None
    bits = packed[0]
    defined = {d for st in cone if isinstance(st, Assign) for d in st.defines()}
    gathers = [ld for ld in loads if _idx_vars(ld.index) & defined]
    if len(gathers) != 1:
        return None
    table = gathers[0]
    root = cone[-1]
    if not isinstance(root, Assign) or root.op.name != "multiply" or len(root.args) != 2:
        return None
    # One multiply arg's cone holds the gather; the other is the factor. The backward cone is the
    # tree-native reading — it replaced the old forward ``map_cone`` walk, and answers the same
    # question here: which stmts does this argument depend on.
    sides = {arg: list(Body(tuple(cone)).backward_cone([arg]).members) for arg in root.args}
    gather_args = [a for a, sub in sides.items() if any(st is table for st in sub) or a in table.names]
    if len(gather_args) != 1:
        return None
    factors = [a for a in root.args if a != gather_args[0]]
    if len(factors) != 1:
        return None  # multiply(x, x): the gathered value on both args leaves no factor
    factor = factors[0]
    blocks: set[int] = set()
    for st in sides[factor]:
        exprs = st.index if isinstance(st, Load) else ()
        for e in exprs:
            naked, guards = _k_block_guard(e, k_name)
            if naked:
                return None
            blocks |= guards
    if len(blocks) != 1:
        return None
    return PackedKBlockB(bits=bits, table=table, factor=factor, block=next(iter(blocks)))


def match_packed_b_node(node, inputs) -> PackedKBlockB | None:
    """The contraction ``node``'s packed-pair k-block B operand, or ``None``.

    The whole node shape the packed byte-slab stage stands on: one channel and a computed B whose
    cone is :func:`match_packed_kblock_b`'s. A may be materialized OR a producer cone — it rides
    whichever side the compute fill gives it (:func:`_atom._a_slab_operand`), so only B decides
    this. Asked here rather than spelled at each consumer, so the schedule's offer, the stage
    resolver and the materializer recognize one set of nodes and cannot drift apart. Everything
    else answers ``None`` and keeps the generic computed-B reading, which computes the same values
    through the smem compute fill.
    """
    if inputs is None or not isinstance(node, Fold) or node.axis is None:
        return None
    # A may be materialized OR a producer cone. A fused RMSNorm ahead of the projection is what a
    # serving program compiles, and refusing it there kept the packed weight off the whole serving
    # path. What the packed reading needs is the B cone's shape, not A's.
    if len(node.channels) != 1:
        return None
    b = node.channels[0].b
    if not isinstance(b, Fold) or b.axis is not None:
        return None
    return match_packed_kblock_b(list(operand_body(b)), node.axis.name, inputs)


@dataclass(frozen=True)
class BlockScaledOperand:
    """One side of the block-scaled reading, split into the three things the instruction wants.

    ``bits`` is the packed-pair storage Load, ``scale`` the RAW block-scale Load its factor
    decodes (a 1-byte, one-value-per-element storage dtype — what the instruction takes as its
    scale operand), and ``alpha`` the factor's k-invariant Loads, whose product the epilogue
    applies once per output element instead of once per k.

    The instruction consumes ``bits`` and ``scale`` and nothing else, so whatever the factor does
    BETWEEN the raw scale and the multiply — today a decode, a multiply by ``alpha`` and a round
    to the fragment dtype — has no counterpart in the emitted cell. That is the bounded gap the
    branch accepts here rather than restating the declared program (PR decision 18).
    """

    bits: Load
    scale: Load
    alpha: tuple[Load, ...]


@dataclass(frozen=True)
class BlockScaledPair:
    """Both contraction operands read as block-scaled packed pairs, over one block extent."""

    a: BlockScaledOperand
    b: BlockScaledOperand
    block: int


def _split_block_scale(read: PackedKBlockB, cone: list, k_name: str, inputs) -> BlockScaledOperand | None:
    """Split one operand's factor into its raw block-scale Load and its k-invariant residue.

    The factor's own backward cone must hold exactly one k-indexed Load — the block scale the
    k-block guard already proved constant per block — at a 1-byte storage dtype the instruction's
    scale operand can take; every other Load in it must be k-free, and those are the residue.
    Anything else declines, and the operand keeps the decode-based readings.
    """
    factor_cone = list(Body(tuple(cone)).backward_cone([read.factor]).members)
    loads = [st for st in factor_cone if isinstance(st, Load)]
    scales = [ld for ld in loads if k_name in _idx_vars(ld.index)]
    if len(scales) != 1:
        return None
    scale = scales[0]
    dt = getattr(inputs.get(scale.input), "dtype", None)
    if dt is None or dt.nbytes != 1 or dt.logical_elems != 1:
        return None  # the scale operand is one raw byte per block; nothing else spells it
    alpha = tuple(ld for ld in loads if ld is not scale)
    if any(_idx_vars(ld.index) for ld in alpha):
        return None  # the residue must be CELL-UNIFORM: the epilogue applies it once per output element
    return BlockScaledOperand(bits=read.bits, scale=scale, alpha=alpha)


def block_scaled_atom(atom) -> bool:
    """Whether ``atom`` multiplies PACKED PAIRS — the one cell whose operands are read as a pair
    rather than off an ``a`` edge's leaf.

    Asked by both the stage resolver and the materializer, because WHICH cell is being lowered
    decides which reading applies: two packed operands under a 16-bit atom are still the
    single-sided shape, whose drain decodes each into 16-bit fragments. The scalar tier's atom
    carries no operand dtypes at all and answers False.
    """
    dtype_of = getattr(atom, "operand_dtype", None)
    return dtype_of is not None and dtype_of("a").logical_elems == 2


def match_packed_pair_node(node, inputs) -> BlockScaledPair | None:
    """The contraction ``node`` read as a BLOCK-SCALED packed pair, or ``None``.

    The instruction's own node shape: one channel, both edges plain operand projections, each a
    :func:`match_packed_kblock_b` decode chain that splits into (packed codes, raw block-scale
    load, k-invariant residue), and both over the SAME block extent — the cell applies one scale
    per block per side and has one block size. A packed weight beside a 16-bit activation answers
    ``None`` here and keeps the single-sided reading (:func:`match_packed_b_node`), whose drain
    decodes into 16-bit fragments.

    Asked here rather than at each consumer for the same reason its single-sided sibling is: the
    schedule's offer, the stage resolver and the materializer must recognize one set of nodes.
    """
    if inputs is None or not isinstance(node, Fold) or node.axis is None or len(node.channels) != 1:
        return None
    edges = (node.a, node.channels[0].b)
    if not all(isinstance(e, Fold) and e.axis is None for e in edges):
        return None
    k_name = node.axis.name
    sides = []
    for edge in edges:
        cone = list(operand_body(edge))
        read = match_packed_kblock_b(cone, k_name, inputs)
        split = _split_block_scale(read, cone, k_name, inputs) if read is not None else None
        if split is None:
            return None
        sides.append((read, split))
    if len({read.block for read, _ in sides}) != 1:
        return None
    return BlockScaledPair(a=sides[0][1], b=sides[1][1], block=sides[0][0].block)


__all__ = [
    "BlockScaledOperand",
    "block_scaled_atom",
    "BlockScaledPair",
    "PackedKBlockB",
    "match_packed_b_node",
    "match_packed_kblock_b",
    "match_packed_pair_node",
]
