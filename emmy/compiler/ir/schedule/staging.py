r"""Operand-stage RESOLUTION — the sizing arithmetic behind the schedule's ``STAGE`` family.

Exactly three resolvers live here — :func:`resolve_warp_stage`, :func:`resolve_scalar_stage`,
:func:`resolve_fill_stage` — plus the compute fill's own node refusals. The classic scheduler
offers a staged transport only when it RESOLVES here, and the row carries the
resolved spelling, so the fork, the stamped knobs and the kernel agree. A resolver is not a
predicate and this is not a legality layer: for the shared-memory budget the legal answer is a
SIZE — the resolved ``bk_elems`` slab chunk and the deepest ring the budget affords — so each
returns a sized :class:`Stage` (or ``None`` for "this transport does not engage here"), and
handing back the largest legal stage is what keeps an over-budget row out of the fork instead of
failing at materialization.

The ``str | None`` functions beside the fill resolver (:func:`computed_operand_cover`,
:func:`computed_operand_copy_dtype`, with :func:`converting_a` naming their converting-A case)
are the NODE-dependent geometry of the move :func:`resolve_fill_stage` realizes — they sit here
because the fill is the move they filter, one statement each so the unpinned enumeration's drop
and a pin's raise share it. What deliberately does NOT live here: the transport/target rule
(MOVE×target — ``Stage.available_on``, filtered in the ``stage_moves`` catalog; the scheduler's
pin path reads its message through :func:`stage_target`) and the fragment-seam relation (including
the paired register bound), which lives in :class:`ClassicScheduleContext`. Nothing here ranks or
narrows for speed."""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.ir.address import BYTE_SLAB_PAD
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.packed import block_scaled_atom, match_packed_b_node, match_packed_pair_node
from emmy.compiler.ir.pure.fold import Fold, operand_name
from emmy.compiler.ir.schedule import ResolvedStage, Stage, Tile
from emmy.compiler.ir.stmt import Load

# TMA hardware: every box dim must fall in 1..256, and the swizzle-split box caps the operand rank
# at 4 so it stays within the 5-dim limit.
_TMA_MAX_BOX = 256
_TMA_ALIGN = 16  # the NONE-swizzle box-copy rule: 16 B-aligned inner dim and inner global stride

# cp.async needs a >= 4 B contiguous chunk, so a 2 B/elem slab's inner dim must be even.
_CP_ASYNC_MIN_ELEMS = 2


def _decline(why: list[str] | None, reason: str) -> None:
    """Record a resolver's decline reason for a caller that will report it (a pinned candidate)."""
    if why is not None:
        why.append(reason)


def stage_target(stage: Stage, ctx) -> str | None:
    """Why ``stage`` names a copy instruction family ``ctx`` lacks (``None`` when it has it) — the
    PIN path's message for the rule :meth:`Stage.available_on` states once (the catalog is already
    filtered through it, so only a pin can reach a transport the target lacks)."""
    if stage.available_on(ctx):
        return None
    need = "cp.async requires sm_80 or newer" if stage.transport == "smem-async" else "TMA requires sm_90 or newer"
    return f"STAGE {stage.spell()}: {need}"


def _clamp_depth(depth: int, slot_bytes: int, budget: int) -> int:
    """The deepest ring the smem ``budget`` affords at ``slot_bytes`` per ringed slot, never deeper
    than asked. Shared by the warp copy ring and the fill's B-slab ring; the scalar resolver
    instead honors the requested depth by SHRINKING its slab chunk (its budget rule is the chunk
    ladder's), so it does not end here."""
    return min(depth, budget // slot_bytes)


def _tma_operand_rank(index: tuple, tile_name: str, k_name: str) -> bool:
    """Whether TMA's box can encode this operand's gmem index. The data plane is the TRAILING 2
    dims; extra LEADING dims ride as extent-1 box dims whose origin is evaluated once per fill, so
    those exprs must not move with the tile or the K loop."""
    if not 2 <= len(index) <= 4:
        return False
    return all(not ({tile_name, k_name} & e.free_vars()) for e in index[:-2])


def _warp_vector_copy(k_axis: Axis, tile_n: int, bk_elems: int, mask_n: bool, b_trans: bool) -> bool:
    """Vector-copy staging: a STATIC, tile-divisible K, an unmasked N, and an even inner slab dim."""
    if mask_n or not k_axis.extent.is_static:
        return False
    if k_axis.extent.as_static() % bk_elems:
        return False
    return bk_elems % _CP_ASYNC_MIN_ELEMS == 0 and (b_trans or tile_n % _CP_ASYNC_MIN_ELEMS == 0)


def _warp_tma(k_axis: Axis, n_axis: Axis, tile_n: int, bk_elems: int, a_bytes: int, b_bytes: int, mask_n: bool, b_trans: bool) -> bool:
    """TMA staging: STATIC tile-divisible K and N, and 16 B-aligned inner dims — each operand's
    box inner dim and gmem inner stride at its OWN element width (a byte-staged fp8 B sizes at
    1 B). A transposed B boxes N-major, so N drops out of the alignment gate."""
    if mask_n or not (k_axis.extent.is_static and n_axis.extent.is_static):
        return False
    k, n = k_axis.extent.as_static(), n_axis.extent.as_static()
    if k % bk_elems:
        return False
    inner = (bk_elems * a_bytes, k * a_bytes) + ((bk_elems * b_bytes, k * b_bytes) if b_trans else (tile_n * b_bytes, n * b_bytes))
    return all(x % _TMA_ALIGN == 0 for x in inner)


# The packed byte-slab stage's fixed geometry. The drain decodes an N-major slab through the k16
# f16 B fragment map and reads one scale per 16 K elements, so the format's block, the atom's K
# step and this constant are all the same 16; a different block or atom keeps the generic reading.
_PACKED_BLOCK = 16

#: The fragment dtypes the packed drain is spelled for. Both hold every e2m1 value exactly, so the
#: decode is a constant-table read in either; a wider or narrower operand has no such table.
_PACKED_FRAGMENT_DTYPES = ("f16", "bf16")


def _packed_warp_stage(c: Fold, tile: Tile, stage: Stage, budget: int, packed, inputs) -> ResolvedStage | None:
    """Resolve the PACKED byte-slab stage for a packed-pair k-block B — the NVFP4 weight cone.

    The scoped shape, which is what the fragment drain is written for: a copy transport (cp.async or
    TMA), an
    N-major packed weight of 16-value blocks under an f16 or bf16 atom whose K step is that same
    16, and an A already carrying the atom's dtype. Everything outside it declines and stays on the generic
    computed-B reading, which computes the same values through the sync compute-fill.

    The sizing is the fp8 byte slab's rule restated in the format's own units. One stored byte is
    two K elements, so the bits row is ``bk_elems / 2`` BYTES plus the cp.async row pad, and it
    must be 16-divisible for the same reason the fp8 one is: the fill copies 16 B chunks and a
    chunk never straddles a row. The gmem rows those chunks stride are ``K / 2`` bytes, so that
    span is 16-divisible too. On top of the ring the budget carries ONE scale slab, ``tile_n ×
    bk_elems / block`` at the atom's element width — single-buffer, because it is compute-filled
    and ringing a compute fill buys no overlap.
    """
    atom = tile.atom
    if stage.transport not in ("smem-async", "smem-tma"):
        return None  # the sync compute fill has nothing to copy under, and split cuts a group this fold has one of
    if packed.block != _PACKED_BLOCK or atom.atom_k != _PACKED_BLOCK or atom.fragment_layout != "m16n8k16":
        return None
    a_dtype, b_dtype = atom.operand_dtype("a"), atom.operand_dtype("b")
    if a_dtype != b_dtype or a_dtype.name not in _PACKED_FRAGMENT_DTYPES:
        return None  # the drain has one value table and one scale multiply, both at the operand dtype
    bits = inputs.get(packed.bits.input)
    if bits is None:
        return None
    if isinstance(c.a, Load):
        a_tensor = inputs.get(c.a.input)
        if a_tensor is None or a_tensor.dtype != a_dtype:
            return None
    # A COMPUTED A has no gmem tensor to match: it evaluates into its slab at the atom's operand
    # dtype, converting on the store, which is the compute fill's own contract.

    if bits.dtype.logical_elems != 2 or len(bits.shape) != 2 or len(packed.bits.index) != 2:
        return None
    if c.axis.name not in packed.bits.index[-1].free_vars():
        return None  # a K-strided packed weight is not the N-major layout the drain reads
    if not c.axis.extent.is_static:
        return None
    k, bk_elems = c.axis.extent.as_static(), tile.bk * atom.atom_k
    if tile.n.mask or k % bk_elems or (bk_elems // 2) % 16 or (k // 2) % 16:
        return None
    # A TMA box deposits DENSE, so the byte rows carry no pad — the same split the fp8 byte slab
    # makes. Its extra demands are the hardware's: every box dim within the 256 limit, and a
    # 16 B-aligned inner span and gmem row stride per operand at its OWN width. The byte side is
    # already 16-divisible by the rule above; A's is ``bk_elems`` and ``k`` at two bytes each.
    pad = 0 if stage.transport == "smem-tma" else BYTE_SLAB_PAD
    if stage.transport == "smem-tma":
        if max(tile.m.tile, tile.n.tile, bk_elems, bk_elems // 2) > _TMA_MAX_BOX:
            return None
        if (bk_elems * a_dtype.nbytes) % _TMA_ALIGN or (k * a_dtype.nbytes) % _TMA_ALIGN:
            return None
    slot_bytes = tile.m.tile * bk_elems * a_dtype.nbytes + tile.n.tile * (bk_elems // 2 + pad)
    scale_bytes = tile.n.tile * (bk_elems // packed.block) * b_dtype.nbytes
    if scale_bytes + slot_bytes > budget:
        return None
    depth = _clamp_depth(stage.depth, slot_bytes, budget - scale_bytes)
    choice = replace(stage, depth=depth, reg_depth=min(stage.reg_depth, tile.bk))
    return ResolvedStage(choice, bk_elems=bk_elems)


def _row_major_k_inner(tensor, load, k_name: str) -> bool:
    """Whether a staged operand is ROW-MAJOR with the contraction axis innermost — the layout the
    byte gathers walk, asked without pinning a rank.

    A layer program and a weight constant carry the same operand at different ranks: a weight is
    ``[N, K/2]``, while an activation keeps its batch axis and its block axis as degenerate dims
    (``[1, M, K/2]``, ``[1, M, K/16, 1]``). Neither affects the address — a unit extent contributes
    no stride — so both are dropped before asking the one question that matters."""
    dims, idx = list(tensor.shape), list(load.index or ())
    if len(dims) != len(idx):
        return False
    while len(dims) > 2 and dims[-1].is_static and dims[-1].as_static() == 1 and not idx[-1].free_vars():
        dims.pop()
        idx.pop()
    while len(dims) > 2 and dims[0].is_static and dims[0].as_static() == 1:
        dims.pop(0)
        idx.pop(0)
    return len(dims) == 2 and all(d.is_static for d in dims) and k_name in idx[-1].free_vars()


def _block_scaled_warp_stage(c: Fold, tile: Tile, stage: Stage, budget: int, pair, inputs) -> ResolvedStage | None:
    """Resolve the FOUR-SLAB stage of a block-scaled packed pair — the native fp4 cell.

    The simplest staging in the tier, because no SCALE is computed: the packed byte-slab stage next
    door compute-fills its scale slab, while here the instruction takes the stored e4m3 byte itself,
    so that fill has nothing left to evaluate. Both sides' scales and every stored side's codes are
    therefore verbatim copies. The one slab still filled is an activation whose codes this very
    matmul computes — its quantize fused in, leaving no buffer to copy from. The weight side is
    always stored, which is what the ``pair.b`` check below requires of every channel.

    The scoped shape: cp.async (the four-descriptor TMA box copy is not built — a missing-code
    fact, stated where the code would live), a k64 cell over 16-element blocks, both code
    operands canonically laid out with k innermost, and a static k the tile divides.

    Sizing restates the byte-slab rule in the format's units, twice per side. A codes row is
    ``bk_elems / 2`` bytes and a scales row ``bk_elems / block``; the fill copies 16 B chunks and
    a chunk never straddles a row, so both spans — and the gmem rows they stride, ``k / 2`` and
    ``k / block`` — must be 16-divisible. That is what bounds the tile from below: at block 16 a
    scales row needs ``bk_elems`` to be a multiple of 256, so the narrow-k tiles decline here and
    keep the generic reading.
    """
    atom = tile.atom
    if stage.transport != "smem-async":
        return None
    if atom.atom_k != 64 or pair.block != _PACKED_BLOCK or atom.operand_dtype("a") != atom.operand_dtype("b"):
        return None
    if not c.axis.extent.is_static or tile.n.mask:
        return None  # an N tile the copy would clamp element-by-element along the contiguous span
    if any(op.bits is None for op in pair.b):
        return None  # only the ACTIVATION side's codes are ever computed here; a weight is stored
    k, bk_elems, block = c.axis.extent.as_static(), tile.bk * atom.atom_k, pair.block
    # Every channel's weight rides the SAME N tile and the same column geometry, so each one is
    # sized on its own terms and any refusal declines the whole node.
    for side, tile_side, atom_dim in ((pair.a, tile.m, 0), *((op, tile.n, 1) for op in pair.b)):
        scale = inputs.get(side.scale.input)
        if scale is None or not _row_major_k_inner(scale, side.scale, c.axis.name):
            return None
        if side.bits is not None:
            # STORED codes copy verbatim, so their gmem layout has to be the one the byte gathers
            # walk. COMPUTED codes have no gmem tensor at all — the fill writes the slab — so
            # there is nothing to lay out and the 16 B chunk rule below applies to the copied
            # slabs only.
            bits = inputs.get(side.bits.input)
            if bits is None or bits.dtype != atom.operand_dtype("a") or not _row_major_k_inner(bits, side.bits, c.axis.name):
                return None
            if (k // 2) % 16:
                return None
        if tile_side.tile % atom.shape[atom_dim]:
            return None
    if k % bk_elems or (bk_elems // 2) % 16 or (bk_elems // block) % 16 or (k // block) % 16:
        return None
    rows = (tile.m.tile, tile.n.tile)
    slot_bytes = sum(r * (bk_elems // 2 + BYTE_SLAB_PAD) + r * (bk_elems // block + BYTE_SLAB_PAD) for r in rows)
    if slot_bytes > budget:
        return None
    choice = replace(stage, depth=_clamp_depth(stage.depth, slot_bytes, budget), reg_depth=min(stage.reg_depth, tile.bk))
    return ResolvedStage(choice, bk_elems=bk_elems)


def resolve_warp_stage(
    c: Fold,
    tile: Tile,
    stage: Stage,
    budget: int,
    inputs=None,
    *,
    readings: tuple | None = None,
) -> ResolvedStage | None:
    """Resolve an operand ``Stage`` against the warp (mma) contraction ``c`` — synchronous copy,
    cp.async, TMA, or gmem-direct (``None``). The resolved stage carries ``bk_elems``, ``depth``
    clamped so the ring's slots fit ``budget``, and ``reg_depth`` clamped to ``bk``. A tile whose
    single depth-1 slot already exceeds ``budget`` DECLINES — unlike the scalar resolver it cannot
    shrink the slab.

    ``inputs`` (the per-buffer tensors) gates operand dtypes: the copy transports byte-copy into
    slabs sized at each operand's element width, so a slab is byte-copied verbatim and the drain
    reads exactly the bytes the fill deposited. Two dtype forms resolve: an operand traced AT the
    atom's operand dtype (the 16-bit family — ldmatrix drain), and a 1-byte (fp8-stored) operand —
    a B under a 16-bit atom stages as a RAW BYTE slab drained by the cooperative convert gather
    (W8A16), and the fp8 (k32) atoms stage both operands as byte slabs drained by the byte repack.
    Any other mismatch DECLINES and keeps the warp tier gmem-direct, whose fragment load converts
    per element. A byte slab's fill runs 16 B chunks and its cp.async row pad is 16 B
    (``address.BYTE_SLAB_PAD``), so its inner span — and, canonical-B, the gmem row stride N — must
    be 16-divisible.

    A PACKED-PAIR B (an NVFP4 weight's decode cone) is the one COMPUTED edge that resolves here,
    through :func:`_packed_warp_stage`: its bits copy verbatim like any byte slab, and the block
    scales the cone would otherwise recompute per element ride a small compute-filled slab beside
    them. Every other computed edge declines — a copy transport cannot evaluate a producer cone —
    and takes :func:`resolve_fill_stage` instead."""
    # Which cell is being resolved decides which reading applies, so the pair question is asked
    # only for the atom that consumes a pair. Both operands packed under a 16-BIT atom is still
    # the single-sided shape: that drain decodes each operand into 16-bit fragments, which is
    # correct — just not what the native cell does.
    # ``readings`` is the caller's memo of the two packed questions, both pure functions of the
    # node. Recomputing them here costs a backward cone per side PER CANDIDATE, and a warp site
    # has hundreds; the prescan asks once and hands the answer down (``_SiteFacts.packed``).
    single, pair = readings if readings is not None else (match_packed_b_node(c, inputs), match_packed_pair_node(c, inputs))
    if pair is not None and block_scaled_atom(tile.atom):
        return _block_scaled_warp_stage(c, tile, stage, budget, pair, inputs)
    if single is not None:
        return _packed_warp_stage(c, tile, stage, budget, single, inputs)
    atom = tile.atom
    sync_copy = stage.transport == "smem" and atom.sync_copy_staging
    bk_elems = tile.bk * atom.atom_k
    m, n = tile.m, tile.n
    a_nbytes, b_nbytes = atom.operand_dtype("a").nbytes, atom.operand_dtype("b").nbytes
    if inputs:
        for edge, role in ((c.a, "a"), (c.b, "b")):
            t = inputs.get(edge.input) if isinstance(edge, Load) else None
            if t is None or t.dtype == atom.operand_dtype(role):
                continue
            if sync_copy:
                return None  # the Volta shared gather consumes f16 slabs; synchronous copies do not convert
            if role == "b" and t.dtype.nbytes == 1 and t.dtype.logical_elems == 1 and b_nbytes == 2:
                b_nbytes = 1  # fp8-B under a 16-bit atom: byte slab, convert at the drain
                continue
            # A packed-pair byte (f4e2m1x2) is NOT an fp8 byte: one stored element is two
            # logical K elements, so the fp8 slab geometry would halve K. No slab takes it.
            return None
    for eb, inner, row_axis in (
        (a_nbytes, bk_elems, None),
        (b_nbytes, bk_elems if c.b_trans else n.tile, None if c.b_trans else n.axis),
    ):
        if eb != 1:
            continue
        if inner % 16:
            return None  # byte slab: 16 B chunks + the 16 B row pad need a 16-divisible inner span
        if row_axis is not None and (not row_axis.extent.is_static or row_axis.extent.as_static() % 16):
            return None  # canonical byte B: the 16 B gmem chunks stride rows of N bytes
    rank_ok = (
        isinstance(c.a, Load)
        and isinstance(c.b, Load)  # a descriptor needs a gmem address on BOTH edges
        and _tma_operand_rank(c.a.index, m.axis.name, c.axis.name)
        and _tma_operand_rank(c.b.index, n.axis.name, c.axis.name)
    )
    box_ok = max(m.tile, n.tile, bk_elems) <= _TMA_MAX_BOX
    tma_ok = (
        stage.transport == "smem-tma"
        and rank_ok
        and box_ok
        and _warp_tma(c.axis, n.axis, n.tile, bk_elems, a_nbytes, b_nbytes, n.mask, c.b_trans)
    )
    vector_copy_ok = _warp_vector_copy(c.axis, n.tile, bk_elems, n.mask, c.b_trans)
    cp_ok = stage.transport == "smem-async" and vector_copy_ok
    sync_ok = sync_copy and vector_copy_ok
    if not (tma_ok or cp_ok or sync_ok):
        return None
    pad_a, pad_b = (BYTE_SLAB_PAD if eb == 1 and cp_ok else 0 for eb in (a_nbytes, b_nbytes))
    b_rows, b_cols = (n.tile, bk_elems + pad_b) if c.b_trans else (bk_elems, n.tile + pad_b)
    slot_bytes = m.tile * (bk_elems + pad_a) * a_nbytes + b_rows * b_cols * b_nbytes
    if slot_bytes > budget:
        return None
    depth = _clamp_depth(stage.depth, slot_bytes, budget)
    choice = replace(stage, depth=depth, reg_depth=min(stage.reg_depth, tile.bk))
    return ResolvedStage(choice, bk_elems=bk_elems)


def resolve_scalar_stage(c: Fold, tile: Tile, stage: Stage, inputs, budget: int) -> ResolvedStage | None:
    """Resolve an operand ``Stage`` against the scalar register-tile contraction ``c``, or ``None``
    (gmem-direct). The slab K-chunk ``bk_elems`` is DERIVED to fit ``depth`` operand slots in the
    smem ``budget`` (the largest offered chunk dividing K) — not codec-spelled, so no schema change;
    when no chunk fits at the requested depth the depth steps down, single-buffer last."""
    if stage.transport not in ("smem-tma", "smem-async") or not c.axis.extent.is_static:
        return None
    # A masked-N B-slab fill would clamp a chunk-start column into a row-crossing gmem address and
    # hang on the misaligned copy; a transposed B has no scalar drain variant (the warp tier stages
    # it into an N-major slab).
    if tile.n.mask or c.b_trans:
        return None
    if not inputs or not isinstance(c.a, Load) or not isinstance(c.b, Load) or c.a.input not in inputs:
        return None
    # 1-byte (fp8) elements decline: the fill's chunk-width and alignment math below is written
    # for the 2/4-byte dtypes and is unaudited at nbytes == 1 — refusing keeps the tier
    # gmem-direct (correct, converts per element) instead of risking a mis-sized slab.
    if any(t is not None and t.dtype.nbytes < 2 for t in (inputs.get(c.a.input), inputs.get(c.b.input))):
        return None
    if stage.transport == "smem-tma" and not (
        _tma_operand_rank(c.a.index, tile.m.axis.name, c.axis.name) and _tma_operand_rank(c.b.index, tile.n.axis.name, c.axis.name)
    ):
        return None
    # Staging needs the CTA to BE one (tile_m x tile_n) output tile (the cooperative fill / drain
    # contract). A register-only tile launches the scalar default block over unrelated cells.
    if tile.launch_threads is None:
        return None
    if stage.transport == "smem-tma" and max(tile.m.tile, tile.n.tile) > _TMA_MAX_BOX:
        return None
    k = c.axis.extent.as_static()
    elem_bytes = inputs[c.a.input].dtype.nbytes
    # Every staged transport needs 16 B-aligned inner global strides — A's is K, B's is N.
    n_ext = tile.n.axis.extent
    if not n_ext.is_static or (k * elem_bytes) % _TMA_ALIGN or (n_ext.as_static() * elem_bytes) % _TMA_ALIGN:
        return None
    b_bytes = inputs[c.b.input].dtype.nbytes if c.b.input in inputs else elem_bytes
    depth, bk_elems = max(1, stage.depth), 0
    while depth >= 1:
        cap = budget // (depth * max(1, tile.m.tile * elem_bytes + tile.n.tile * b_bytes))
        bk_elems = next((v for v in (128, 64, 32, 16, 8, 4) if v <= cap and k % v == 0), 0)
        if bk_elems >= 4:
            break
        depth -= 1
    if bk_elems < 4:
        return None
    return ResolvedStage(replace(stage, depth=depth, reg_depth=1), bk_elems=bk_elems)


# ---- the smem compute fill --------------------------------------------------------------------- #


def converting_a(node: Fold, atom, inputs) -> bool:
    """Whether the ``a`` edge is a MATERIALIZED load whose dtype the atom cannot bind directly —
    the CONVERTING smem compute fill's case (an erased ``.float()`` cast ahead of an f16
    projection): the synchronous fill evaluates the load per slab cell and the typed slab store
    performs the conversion. A byte transport moves raw bits and cannot, so such an edge takes the
    fill or nothing. ``False`` for computed edges (the fill's native case), matching dtypes, and
    1-byte loads (the fp8 tiers move raw bits by design)."""
    if not isinstance(node.a, Load) or not inputs:
        return False
    if atom.operand_dtype("a").nbytes < 2:
        return False
    t = inputs.get(node.a.input)
    return t is not None and t.dtype.nbytes >= 2 and t.dtype != atom.operand_dtype("a")


def computed_operand_cover(c: Fold, tile: Tile, *, converting: bool = False, k_axis: Axis | None = None) -> str | None:
    """Geometry required by a smem compute-filled contraction operand.

    A computed A leaves B on the async-copy path, whose contiguous N-vector copy cannot clamp a
    partial inner row element-by-element, so N must be exact. A computed B leaves materialized A
    as the async operand; M is its *outer* slab row and can be safely clamped as a whole. Computed
    B's own per-cell fill clamps N before evaluating the generic producer cone.

    A SYMBOLIC K rides the fill's own K mask: the cone's reads clamp in-bounds and every slab lane
    whose k index reaches past the runtime extent stores the fold identity 0 (the bilinear reading
    pins ⊕ = add, so a zero A contributes nothing and the drain may read the whole chunk
    unconditionally). What that mask cannot cover is a BYTE-COPIED peer whose slab row is K-MAJOR —
    a materialized A, and a transposed B, both stage K as the slab's contiguous inner dim, so their
    cp.async chunk runs ALONG K and a clamped chunk START still copies past the extent. Those keep
    the refusal; a converting materialized A does not, since it rides the fill as a cone.

    ``k_axis`` overrides the stored axis for a derived unit-marker contraction whose enclosing
    Fold owns the actual K sweep."""
    if not (k_axis or c.axis).extent.is_static:
        if isinstance(c.a, Load) and not converting:
            return (
                "a materialized A stages K-major (K is the slab's contiguous row), so its cp.async "
                "chunk runs along K and cannot clamp a symbolic K's partial tail; the masked fill "
                "covers a COMPUTED (or converting) A only"
            )
        if c.b_trans:
            return (
                "a transposed B stages N-major (K contiguous), so its cp.async chunk runs along K "
                "and cannot clamp a symbolic K's partial tail; pin a canonical B layout"
            )
    materialized_b = [isinstance(ch.b, Load) for ch in c.channels]
    if any(materialized_b) and not all(materialized_b):
        return "the smem compute fill requires homogeneous B channels; mixed computed/materialized B layouts stay on the demoted reading"
    if tile.n.mask and any(isinstance(ch.b, Load) for ch in c.channels):
        return (
            f"a smem compute fill with a materialized B needs a TILE whose N width exactly covers "
            f"the static output columns (N={tile.n.axis.extent}; copied inner-row chunks cannot "
            f"clamp individual N cells); pick a dividing tile."
        )
    return None


def computed_operand_copy_dtype(c: Fold, tile: Tile, inputs, *, converting: bool = False) -> str | None:
    """Every BYTE-COPIED edge of a compute-filled contraction must already have the atom dtype.

    The ``smem`` stage evaluates computed (and converting) operands into their typed shared-memory
    slabs, but it *copies* every materialized peer byte-for-byte.  A copied f32 edge therefore
    cannot feed an f16 ``ldmatrix`` fragment merely because another edge is filled. Filled edges
    are exempt because their slab store performs the normal typed conversion — ``converting``
    marks a materialized ``a`` that rides the converting fill rather than the copy."""
    for edge, role in ((c.a, "a"), *((ch.b, "b") for ch in c.channels)):
        if not isinstance(edge, Load) or (role == "a" and converting):
            continue
        tensor = inputs.get(edge.input) if inputs else None
        # Structural scheduler fixtures intentionally do not carry Tensor metadata. Absence is not
        # evidence of an unsafe byte copy; the concrete lowering path always supplies inputs and
        # is where a known mismatch must be rejected.
        if tensor is None:
            continue
        want = tile.atom.operand_dtype(role)
        if tensor.dtype == want:
            continue
        return (
            f"smem compute fill: materialized {role.upper()} edge {edge.input!r} is {tensor.dtype}, but "
            f"atom {tile.atom.name} copies it into a {want} slab without conversion; only the "
            "``a`` role has a converting fill"
        )
    return None


def resolve_fill_stage(
    c: Fold,
    tile: Tile,
    budget: int,
    want_depth: int = 1,
    inputs=None,
    why: list[str] | None = None,
    seam: tuple | None = None,
    k_axis: Axis | None = None,
    producer: Fold | None = None,
) -> ResolvedStage | None:
    """The ``smem`` compute-fill :class:`Stage` for a computed-operand warp contraction under
    ``tile`` — MANDATORY for this form (the gmem-direct mma leaf refuses a computed A, and the
    byte-copy / cp.async / TMA transports move bytes and cannot evaluate a producer cone), so it
    has no gmem-direct ``""`` sibling and a ``STAGE`` pin can only choose its DEPTH. ``None`` when
    the slabs exceed ``budget``: one A slab, one B slab per channel, and one fp32 row per bridged
    statistic (``ops.cone_seam``'s ``stats`` — the same node boundary the materializer fills
    through).

    ``want_depth >= 2`` is the asymmetric B-only prefetch ring: only the B cp.async slabs ring
    (their copies for chunk ``i+d-1`` fly under chunk ``i``'s compute fill and drain), while the
    compute-filled A slab and the stat rows stay single-buffer — ringing a compute fill buys no
    overlap, it runs on the drain's own threads. Both depths are fork siblings, measured per shape.

    ``k_axis`` overrides the stored contraction axis when the contraction is a derived singleton
    marker whose enclosing Fold owns the actual K sweep. ``why`` collects the decline reason when
    the tier refuses, so a PINNED caller reports the gate it actually hit."""
    from emmy.compiler.ir.tile.ops import cone_seam  # noqa: PLC0415

    atom = tile.atom
    k_axis = k_axis or c.axis
    if atom.operand_dtype("a").nbytes < 2:
        # fp8 atoms: the compute fill's slab store + ldmatrix drain are 16-bit-only
        _decline(why, f"the smem compute fill is 16-bit-only, but this atom's a operand is {atom.operand_dtype('a').nbytes}-byte")
        return None
    bk_elems = tile.bk * atom.atom_k
    if k_axis.extent.is_static and k_axis.extent.as_static() % bk_elems:
        # the staged driver unrolls WHOLE K chunks — the same rule the copy transports state on their own
        _decline(
            why,
            f"the smem compute fill unrolls whole K chunks, but its {bk_elems}-element chunk "
            f"does not divide the contraction K={k_axis.extent.as_static()}",
        )
        return None
    a_nbytes = atom.operand_dtype("a").nbytes
    b_nbytes = atom.operand_dtype("b").nbytes
    _, _, stats = seam if seam is not None else cone_seam(c.a, c.axis.name) if not isinstance(c.a, Load) else ((), (), ())
    a_bytes = tile.m.tile * bk_elems * a_nbytes
    stat_bytes = len(stats) * tile.m.tile * 4
    sync_bytes = stat_bytes
    async_bytes = 0
    # A materialized A whose dtype the atom cannot bind rides the CONVERTING synchronous fill —
    # per-cell load + typed slab store — never the byte copy (which cannot convert).
    a_converts = converting_a(c, atom, inputs)
    if isinstance(c.a, Load) and not a_converts:
        async_bytes += a_bytes
    else:
        sync_bytes += a_bytes
    for ch in c.channels:
        if isinstance(ch.b, Load):
            async_bytes += tile.n.tile * bk_elems * b_nbytes
        else:
            sync_bytes += tile.n.tile * bk_elems * b_nbytes
    # A scheduled contraction producer contributes its own streamed and invariant operand slabs.
    # They do not ring: the streamed slab dies inside the block and the invariant slab does not
    # advance. Reserve both from the producer interface supplied by the scheduler.
    producer_k = producer.axis.extent.as_static() if producer is not None and producer.axis.extent.is_static else 0
    producer_bytes = producer_k * (bk_elems * b_nbytes + tile.m.tile * a_nbytes)
    if sync_bytes + async_bytes + producer_bytes > budget:
        _decline(why, f"the smem compute fill's slabs need {sync_bytes + async_bytes + producer_bytes} B, over the {budget} B smem budget")
        return None
    fixed = sync_bytes + producer_bytes
    # Only the asynchronous peer slabs ring (the compute-filled slab and stat rows stay
    # single-buffer), so the clamp budgets the ringed slot against what the fixed slabs leave.
    depth = _clamp_depth(want_depth, async_bytes, budget - fixed) if async_bytes else 1
    computed = [operand_name(c.a)] if a_converts or not isinstance(c.a, Load) else []
    computed.extend(operand_name(ch.b) for ch in c.channels if not isinstance(ch.b, Load))
    return ResolvedStage(Stage(depth=depth, transport="smem"), smem=tuple(computed), bk_elems=bk_elems)


__all__ = [
    "computed_operand_copy_dtype",
    "computed_operand_cover",
    "converting_a",
    "resolve_fill_stage",
    "resolve_scalar_stage",
    "resolve_warp_stage",
    "stage_target",
]
