"""Per-node schedule legality — **one predicate, one home, one severity**.

Every function here answers ONE question about whether a candidate is realizable against a
particular node, and returns the REFUSAL REASON as a string (or ``None`` for "legal"). The caller
picks the severity through :func:`enforce`: an env PIN raises the reason, the unpinned enumeration
silently drops the candidate. That split is why the predicates live together — the same rule stated
twice, once as a raise and once as a drop, is what produces "the pin says yes and the enumeration
says no".

The rules here are ORDINARY ARITHMETIC over facts read off the node — a thread budget, a K-step
that must tile a static extent, a 16 B-aligned inner stride. They are deliberately NOT expressed as
:mod:`~emmy.compiler.pipeline.search.domain` ``Bound``\\ s: that module GENERATES a candidate
domain, and nothing here is ever installed into a ``Space``. Routing a scalar divisibility test
through a constructed ``Bound`` shares the currency in name only and costs a second file open to
read ``k % step == 0``. If a legality rule ever does become a generated bound, constructing one is
a line.

What does NOT live here: anything that CHOOSES rather than checks. The coop / ILP partition catalog,
the atom ladder and the stage catalog are the schedule's. Nothing here ranks either — a predicate
answers "can this node realize the candidate", never "is it a good idea", which is why legality is
the only thing allowed to narrow an enumeration at all.

**With ONE stated exception: the stage RESOLVERS** (``resolve_warp_stage`` / ``resolve_scalar_stage``
/ ``resolve_fill_stage``). They return a SIZED :class:`Stage` rather
than a reason, because for the smem budget the legal answer is a size and not a yes/no — the
largest depth and slab chunk that fit. Handing back the largest legal stage is what keeps an
over-budget row out of the fork instead of failing at materialization, and splitting "how big may
this be" from "is this legal" would put the budget in two places. They are resolvers, they are
named so, and they clamp through the one shared :func:`clamp_depth`; everything else here refuses.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.dtype import BF16, F16
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr
from emmy.compiler.ir.pure.fold import Fold, operand_name
from emmy.compiler.ir.schedule import ReducePlan, Stage, TilePlan, WarpSpec
from emmy.compiler.ir.stmt import Body, Load, Loop, Write
from emmy.compiler.ir.stmt.passes import has_contraction_tail, projection_distributes
from emmy.compiler.ir.tile.ops import cone_seam
from emmy.compiler.pipeline.passes.lowering._addr import BYTE_SLAB_PAD, gmem_axis_step, split_pair
from emmy.compiler.pipeline.search.space import (
    MAX_BLOCK_THREADS,
    MAX_REGISTERS_PER_CTA,
    MAX_REGISTERS_PER_THREAD,
    WARP_LANES,
)

# TMA hardware: every box dim must fall in 1..256, and the swizzle-split box caps the operand rank
# at 4 so it stays within the 5-dim limit.
_TMA_MAX_BOX = 256
_TMA_ALIGN = 16  # the NONE-swizzle box-copy rule: 16 B-aligned inner dim and inner global stride

# cp.async needs a >= 4 B contiguous chunk, so a 2 B/elem slab's inner dim must be even.
_CP_ASYNC_MIN_ELEMS = 2


def decline(why: list[str] | None, reason: str) -> None:
    """Record a tier's refusal reason for a caller that will report it (a pinned candidate)."""
    if why is not None:
        why.append(reason)


def enforce(reason: str | None, *, pinned: bool) -> bool:
    """Resolve a refusal into a decision: ``True`` when legal, ``False`` when the candidate should
    be dropped, and a ``ValueError`` when it was PINNED — a pin names a specific kernel, so refusing
    it silently would deploy something the user did not ask for."""
    if reason is None:
        return True
    if pinned:
        raise ValueError(reason)
    return False


def _direct_atomic_output(outputs) -> str | None:
    """Whether direct cross-CTA partials avoid another low-precision rounding step.

    F16/BF16 destinations round once per CTA; the deferred finalize instead combines f32
    carrier state and rounds once at the output boundary.
    """
    lowp = sorted({str(t.dtype) for t in outputs.values() if t.dtype in (F16, BF16)})
    if not lowp:
        return None
    return (
        f"direct atomic REDUCE writes each partial into {'/'.join(lowp)} output storage; "
        "use the deferred f32 workspace finalize (REDUCE=g<n>k) so the output rounds once"
    )


def atomic_finalize(states: tuple[str, ...], tail, outputs) -> str | None:
    """Whether the cross-CTA split may take its DIRECT ``atomicAdd`` arm — every condition, once.

    The deferred workspace finalize (``REDUCE=g<n>k``) carries any carrier and any projection, so
    each refusal here names it as the alternative. Three things must hold, and they are stated
    together because the enumeration, the pin, and ``030_split_reduce``'s realization all have to
    agree: a pin that reaches the rewrite past a gate the enumeration applied would crash the
    emitter instead of refusing.

    - ONE additive state component. ``atomicAdd`` folds a scalar; a twisted carrier's
      ``(maximum, denominator, …)`` tuple has no atomic instruction at all.
    - An output the partials can round into once per CTA (:func:`_direct_atomic_output`).
    - A projection that DISTRIBUTES over the add. The atomic arm applies the epilogue per
      partition, before the combine, so anything but a linear-homogeneous map mis-scales each
      CTA's contribution.

    Storage is asked BEFORE the projection: a narrowing store spells its rounding as a conversion
    in that same epilogue (``loop/lifting/090_spell_store_rounding``), which no linear-homogeneous
    reading admits — so both refuse a low-precision output and the storage reason is the one that
    names why.
    """
    if len(states) != 1:
        return (
            f"atomic REDUCE folds ONE additive state component; this carrier has {len(states)} "
            f"({', '.join(states)}) — use the deferred f32 workspace finalize (REDUCE=g<n>k)"
        )
    storage = _direct_atomic_output(outputs)
    if storage is not None:
        return storage
    if tail and not projection_distributes(tuple(tail), states):
        return (
            "atomic REDUCE applies the projection epilogue per partition, so it must distribute "
            "over the add; this one does not (a fused bias / activation) — use the deferred "
            "workspace finalize (REDUCE=g<n>k), which projects once after the combine"
        )
    return None


# ---- thread budgets ---------------------------------------------------------------------------- #


def scalar_block_threads(plan: TilePlan) -> str | None:
    """A scalar tile launches one thread per parallel output cell, so its parallel widths spend
    ``par_n·par_m`` of the CTA budget."""
    if plan.units_n * plan.units_m <= MAX_BLOCK_THREADS:
        return None
    return (
        f"TILE parallel block {plan.units_n}x{plan.units_m}={plan.block_threads} threads exceeds the "
        f"{MAX_BLOCK_THREADS}-thread/CTA limit; shrink n/m or move work to the f register sub-tile."
    )


def producer_band(spec: WarpSpec, block_threads: int) -> str | None:
    """A dedicated producer band adds ``32·p`` threads ON TOP of the compute warps. Two budgets:
    the total fits the CTA limit, and the band does not outnumber the compute half.

    The only rule in this module that is NOT a bound — it mixes a sum with a product, which no
    coordinate change linearizes — so it stays an ordinary arithmetic predicate.
    """
    aux = WARP_LANES * spec.producer_warps
    if aux > block_threads:
        return f"producer band {aux} threads outnumbers the {block_threads} compute threads"
    if block_threads + aux > MAX_BLOCK_THREADS:
        return f"producer band {aux} + {block_threads} compute exceeds the {MAX_BLOCK_THREADS}-thread/CTA limit"
    return None


def producer_transport(stage: Stage | None) -> str | None:
    """What a producer band can actually drive: a RESOLVED TMA stage (the band arms the box-copy
    mbarrier ring — cp.async's wait-group is issuing-thread-scoped and a smem compute fill has no
    async load half).

    It used to also refuse a cross-CTA split as "not co-representable", and took the row's
    ``ReducePlan`` for that alone. A split now mints brand-new kernels that schedule themselves, so
    a split row's band never reaches a kernel at all, and a partial that wants one asks for it at
    its own fork like any other kernel."""
    if stage is None or stage.transport != "smem-tma":
        return "a producer band drives a resolved TMA stage; this row has none"
    return None


def coop_band_geometry(plan: ReducePlan, k: int | None, inner: Axis | None) -> str | None:
    """The TRANSPOSED coop band's structural requirements — the same fact for every tier that
    offers it. The band swaps the lane mapping so 32 lanes sweep the innermost FREE axis while
    each lane walks K serially, which fixes three things about the term: the coop width must be a
    whole number of warps, there must BE an innermost free axis to sweep, and a split composite
    must divide K. The swept extent itself is free — the grid is ``ceil(E/32)`` blocks and the
    emitter clamp-reads + store-guards an overhanging last block.

    One home because it had two: the reduce tier and the contraction tier each spelled a version of
    this inline, with different conditions and neither of them here — exactly the raise-vs-drop
    defect class this module exists to remove. The reduce tier's EXTRA requirement, on the shape of
    the epilogue, is :func:`coop_band_epilogue`. A plain (non-transposed) band answers ``None`` —
    it has no such geometry."""
    if not plan.coop_transposed:
        return None
    if plan.coop % WARP_LANES:
        return f"the transposed coop band sweeps whole warps — coop={plan.coop} is not a multiple of {WARP_LANES}"
    if inner is None:
        return "the transposed coop band needs an innermost free axis to sweep; this term has none"
    if plan.needs_split and (k is None or k % plan.cta):
        return f"the transposed split composite g{plan.cta} must divide a static K"
    return None


def coop_band_epilogue(tail) -> str | None:
    """The reduce tier's EXTRA requirement on a transposed band: the projection epilogue must be
    per-cell straight-line code. A swept lane owns output cells rather than a row, so a tail that
    is itself a loop — a sweep, or the nested contraction of a fused norm→linear — has no per-cell
    form for the swapped mapping to run, and the fused shape's shared-row slab is addressed for the
    plain band's row-per-CTA layout.

    A statement-SHAPE question, which is why it takes the tail rather than the term: the reduce
    tier used to ask it by calling the shared-row STAGE resolver and testing for ``None``, wiring a
    transport decision into a REDUCE candidate gate for a fact about the epilogue alone."""
    if any(isinstance(s, Loop) for s in tail):
        return "the transposed coop band needs a per-cell epilogue — this tail sweeps"
    if has_contraction_tail(tail):
        return "the transposed coop band needs a per-cell epilogue — this tail contracts over a new axis"
    return None


# ---- the warp tier's K step -------------------------------------------------------------------- #


def warp_atom_target(atom, ctx) -> str | None:
    """Whether ``atom`` belongs to the target's selected MMA instruction family."""
    if atom.available_on(ctx):
        return None
    return (
        f"warp TILE: atom {atom.name} requires target feature {atom.target_feature}, which is unavailable "
        f"on sm_{ctx.compute_capability[0]}{ctx.compute_capability[1]}"
    )


def warp_a_columns(node: Fold, tile: TilePlan, inputs) -> str | None:
    """Whether a materialized A edge has the contiguous K columns an mma fragment loader reads.

    A gmem fragment load always advances one element at a time within an ``atom_k``-wide row.  A
    blocked index is still representable when every fragment stays inside one of its contiguous
    runs; an unknown or strided index is not.  Computed A is exempt because its sync fill writes a
    dense shared-memory slab rather than asking a gmem loader to interpret the source index.
    """
    if not isinstance(node.a, Load):
        return None
    step = gmem_axis_step(node.a, node.axis.name, inputs)
    width = tile.atom.atom_k
    if step is not None and step[0] == 1 and (step[1] == 0 or step[1] % width == 0):
        return None
    motion = "unknown" if step is None else f"{step[0]} elements per column"
    return (
        f"warp TILE: A fragment loaders read {width} contraction columns CONTIGUOUSLY, but this "
        f"operand's gmem index moves {motion}; drop the atom token to use the scalar tier."
    )


def stage_target(stage: Stage, ctx) -> str | None:
    """Whether ``stage`` names a copy instruction family present on ``ctx``."""
    if stage.transport == "smem-async" and not ctx.has_cp_async:
        return f"STAGE {stage.spell()}: cp.async requires sm_80 or newer"
    if stage.transport == "smem-tma" and not ctx.has_tma:
        return f"STAGE {stage.spell()}: TMA requires sm_90 or newer"
    return None


def warp_k_step(node: Fold, plan: TilePlan) -> str | None:
    """Whether this atom's fragment loaders can reach the contraction K.

    The warp K-loop steps by ``atom_k`` and zero-fills the overhanging half of its final fragment
    (``_atom`` passes the loop's ``k_zero`` bound whenever the step does not tile K), so a K the
    step does not divide — static or symbolic — is masked and correct. The masked tail is the
    gmem-direct tier's; a STAGED row's K-chunk divisibility is the stage resolvers' own rule
    (``_warp_vector_copy`` / ``_warp_tma`` / :func:`resolve_fill_stage`), stated where the chunk
    width is.

    The fp8 atoms are the exception on both counts: their byte-gather fragment loaders have no
    masked-K zero-fill family, so they take an exact K — static, and tiled by the full K-step."""
    if not (plan.is_warp and plan.atom.operand_dtype("a").nbytes == 1):
        return None
    ext = node.axis.extent
    if not ext.is_static:
        return f"atom {plan.atom.name}: the fp8 byte-gather loaders have no masked-K zero-fill — a symbolic K stays off the fp8 tier"
    k, step = ext.as_static(), plan.atom.atom_k * plan.bk
    if k % step == 0:
        return None
    return (
        f"warp TILE K-step {step} (atom_k={plan.atom.atom_k}*bk={plan.bk}) does not divide the static "
        f"contraction K={k}, and atom {plan.atom.name}'s byte-gather loaders have no masked-K zero-fill; "
        f"pin a K that is a multiple of {step}, or drop the fp8 atom token."
    )


def _fragment_registers(atom, role: str) -> int:
    """The exact per-lane register count of one emitted mma fragment."""
    explicit = atom.fragment_nregs(role)
    if explicit is not None:
        return explicit
    m, n, k = atom.ptx_shape
    dtype = atom.operand_dtype(role)
    if role == "a":
        return m * k * dtype.nbytes // 128
    if role == "b":
        return n * k * dtype.nbytes // 128
    return m * n // (64 if dtype.nbytes == 2 else 32)


def paired_fragment_registers(node: Fold, producer: Fold | None, tile: TilePlan, stage: Stage | None) -> tuple[int, int] | None:
    """Return ``(required, available)`` peak registers/lane for two composed contractions.

    A computed fill keeps the consuming fragments live while it builds one scheduled producer
    block through the same mma atom. Count the exact ``RegFragment`` families emitted by
    ``_MmaOps.state`` for both contractions. This is a lower bound: scalar carrier state and
    address temporaries are intentionally absent, so the check rejects only rows whose fragments
    alone cannot fit the CTA register file.
    """
    if not (tile.is_warp and stage is not None and producer is not None):
        return None
    atom = tile.atom
    if stage.bk_elems % atom.atom_n:
        return None  # the producer fragment block does not realize this geometry
    a_regs = _fragment_registers(atom, "a")
    b_regs = _fragment_registers(atom, "b")
    c_regs = _fragment_registers(atom, "c")
    # The f16-accumulate atom keeps an additional f32 shadow C family.
    if atom.operand_dtype("c").nbytes == 2:
        c_regs += atom.atom_m * atom.atom_n // 32
    depth = max(1, stage.reg_depth)
    channels = len(node.channels)
    outer_c = channels * tile.reg_m * tile.reg_n * c_regs
    outer = tile.reg_m * depth * a_regs + channels * (tile.reg_n * depth * b_regs + tile.reg_m * tile.reg_n * c_regs)
    producer_n = stage.bk_elems // atom.atom_n
    producer_channels = len(producer.channels)
    producer_regs = tile.reg_m * a_regs + producer_channels * (producer_n * b_regs + tile.reg_m * producer_n * c_regs)
    available = min(MAX_REGISTERS_PER_THREAD, MAX_REGISTERS_PER_CTA // tile.block_threads)
    # Consumer A/B are first loaded in the drain after the producer block. Only the initialized consumer C
    # fragments span both regions; the two A/B families may reuse registers.
    return max(outer, outer_c + producer_regs), available


def paired_fragment_register_budget(node: Fold, producer: Fold | None, tile: TilePlan, stage: Stage | None) -> str | None:
    """Whether coexisting producer/consumer mma fragments fit the CTA register-file envelope."""
    counts = paired_fragment_registers(node, producer, tile, stage)
    if counts is None or counts[0] <= counts[1]:
        return None
    required, available = counts
    return (
        f"paired contractions require at least {required} live fragment registers/thread, over the "
        f"{available}-register envelope at {tile.block_threads} threads/CTA"
    )


def splitk_width(k_axis: Axis, width: int) -> str | None:
    """A cross-CTA split needs a STATIC reduce axis its width divides evenly — the σ-reindex
    reconstructs an absolute k from ``ksplit·(K/w) + kslice``, which is a bijection only when the
    extent is known and ``w`` divides it. Total over a symbolic axis rather than raising out of
    ``as_static``, so the enumeration drops the row and a pin reports the reason."""
    if not k_axis.extent.is_static:
        return f"cross-CTA split of the symbolic reduce axis {k_axis.name!r} is not built"
    big_k = k_axis.extent.as_static()
    if big_k % width == 0:
        return None
    return f"split-K width {width} does not divide K={big_k}; pick a dividing split width."


# ---- the register strip ------------------------------------------------------------------------ #


def strip_width(extent: int, width: int) -> str | None:
    """The pointwise strip hands each thread ``width`` CONTIGUOUS inner-axis elements, so the width
    must tile the inner free extent.

    Not an unimplemented mask — MEASURED. The one form that masks the overhang without breaking the
    strip's flat shape slides the last cell back onto the final full run (``min(cell·width,
    extent − width)``, idempotent because the cell is a pure map), and that slid base is no longer a
    provably aligned affine form, so ``050_vectorize_loads`` / ``080_vectorize_stores`` decline —
    which is the only thing the strip exists to buy. On a V100, gelu over 65536×255 (no width tiles
    255): the flat per-cell map runs 158.5 µs while the slid ``f2`` / ``f4`` / ``f8`` strips run
    199.5 / 220.2 / 390.8 µs. The refused rows are strictly worse than the row that remains."""
    if width <= 1 or (extent and extent % width == 0):
        return None
    return f"register strip width {width} does not divide the inner free extent {extent}"


# ---- the fragment epilogue --------------------------------------------------------------------- #


def fragment_epilogue(epilogue: Body) -> str | None:
    """The mma store folds the projection into a ``RegEpilogue`` whose leaf ``Load``\\ s are
    evaluated independently per fragment element, so a load whose INDEX reads a name an earlier
    epilogue stmt defined (an embedding gather) cannot be threaded through it. A nested output
    sweep is likewise not a per-fragment straight-line epilogue."""
    defs: set[str] = set()
    for s in epilogue:
        if isinstance(s, Loop):
            return "warp TILE: the projection epilogue contains an output sweep — use the scalar tier"
        if isinstance(s, Load) and {v for e in s.index for v in e.free_vars()} & defs:
            return (
                "warp TILE: the projection epilogue gathers through another epilogue load (a "
                "data-dependent index) — the fragment epilogue cannot thread it; drop the atom "
                "token to use the scalar tier."
            )
        defs.update(s.defines())
    return None


def _split_addressable(index: tuple, shape, name: str, atom_ext: int, trailing: bool) -> str | None:
    """Whether the fragment store's per-cell addressing — the cell base evaluated once at the atom
    origin, then ``+ col`` / ``+ row · ldm`` across the atom — is exact for an axis that reaches the
    buffer as a split dim pair. It is when the row-major flatten recomposes the pair to an affine
    address (the strides match the split), or when the ``%`` dim is the innermost carrier and its
    modulus is a multiple of the atom extent, so an aligned atom never straddles a ``Q`` boundary."""
    dims = [d for d, e in enumerate(index) if name in e.free_vars()]
    if len(dims) < 2:
        return None
    q = split_pair(index, name)
    if q is None:
        return "not a bare f/Q, f%Q pair"
    inner = index[dims[1]]
    if not (isinstance(inner, BinaryExpr) and inner.op == "%"):
        return "the quotient dim is the inner one"
    if trailing and dims[1] != len(index) - 1:
        return "the remainder dim is not the contiguous one"
    if shape is not None and len(shape) == len(index) and all(getattr(d, "is_static", False) for d in shape):
        stride, strides = 1, [1] * len(index)
        for d in reversed(range(len(index))):
            strides[d] = stride
            stride *= shape[d].as_static()
        if strides[dims[0]] == q * strides[dims[1]]:
            return None  # row-major recomposition — the address is affine in the axis
    if q % atom_ext:
        return f"Q={q} is not a multiple of the {atom_ext}-wide atom cell"
    return None


def warp_split_store(tail: list, free: tuple, atom_shape: tuple, shapes: dict) -> str | None:
    """The fragment epilogue addresses the output (and each epilogue load) per ATOM: the cell base
    is evaluated once at the atom origin and the lanes add ``col`` / ``row · ldm``. A re-fused
    split axis spells its coordinate across two buffer dims (``[…, f/Q, …, f%Q]``), and that is
    addressable only under :func:`_split_addressable`; otherwise the scalar tiers, which evaluate
    every element's index, are the kernel's tiers."""
    roles = [(free[-1].name, atom_shape[1], "n", True)]
    if len(free) >= 2:
        roles.append((free[-2].name, atom_shape[0], "m", False))
    for s in tail:
        if not isinstance(s, (Write, Load)):
            continue
        buf = s.output if isinstance(s, Write) else s.input
        for name, ext, role, trailing in roles:
            why = _split_addressable(s.index, getattr(shapes.get(buf), "shape", None), name, ext, trailing)
            if why is not None:
                return f"warp TILE: the {role} axis reaches {buf} as a split dim pair the fragment store cannot address ({why})"
    return None


# ---- operand staging --------------------------------------------------------------------------- #


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


def resolve_warp_stage(c: Fold, tile: TilePlan, stage: Stage, budget: int, inputs=None) -> Stage | None:
    """Resolve an operand ``Stage`` against the warp (mma) contraction ``c`` — synchronous copy,
    cp.async, TMA, or gmem-direct (``None``). The resolved stage carries ``bk_elems``, ``depth`` clamped so the ring's
    slots fit ``budget``, and ``reg_depth`` clamped to ``bk``. A tile whose single depth-1 slot
    already exceeds ``budget`` DECLINES — unlike the scalar resolver it cannot shrink the slab.

    A resolver rather than a predicate because the legal answer is a SIZE, not a yes/no: this is the
    one enforcement point for the smem budget, and returning the largest legal stage is what keeps
    an over-budget row out of the fork instead of failing at materialization.

    ``inputs`` (the per-buffer tensors) gates operand dtypes: the copy transports byte-copy into
    slabs sized at each operand's element width, so a slab is byte-copied verbatim and the drain
    reads exactly the bytes the fill deposited. Two dtype forms resolve: an operand traced AT the
    atom's operand dtype (the 16-bit family — ldmatrix drain), and a 1-byte (fp8-stored) operand —
    a B under a 16-bit atom stages as a RAW BYTE slab drained by the cooperative convert gather
    (W8A16), and the fp8 (k32) atoms stage both operands as byte slabs drained by the byte repack.
    Any other mismatch DECLINES and keeps the warp tier gmem-direct, whose fragment load converts
    per element (the same rule flash's ``stageable`` flag states). A byte slab's fill runs 16 B
    chunks and its cp.async row pad is 16 B (``_addr.BYTE_SLAB_PAD``), so its inner span — and,
    canonical-B, the gmem row stride N — must be 16-divisible."""
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
            if role == "b" and t.dtype.nbytes == 1 and b_nbytes == 2:
                b_nbytes = 1  # fp8-B under a 16-bit atom: byte slab, convert at the drain
                continue
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
    depth = clamp_depth(stage, slot_bytes, budget)
    return replace(stage, depth=depth, reg_depth=min(stage.reg_depth, tile.bk), bk_elems=bk_elems)


def clamp_depth(stage: Stage, slot_bytes: int, budget: int) -> int:
    """The deepest ring the smem ``budget`` affords at ``slot_bytes`` per slot, never deeper than
    the stage asked for. The ONE clamp every stage resolver ends in — a budget stated once, so a
    tier cannot drift into affording a slot the others refuse."""
    return min(stage.depth, budget // slot_bytes)


def warp_operand_dtype(c: Fold, tile: TilePlan, a_dtype) -> str | None:
    """A MATERIALIZED ``a`` edge must already carry the atom's own operand dtype: the transports that
    fill its fragment — gmem-direct ``ldmatrix``, cp.async, TMA — move raw BYTES and cannot convert.
    Only the smem compute fill converts (on the slab store), which is why a mixed pair reaches the
    warp tier through the mixed-A promotion's cone instead. A COMPUTED edge is exempt for exactly
    that reason."""
    if not isinstance(c.a, Load) or a_dtype == tile.atom.operand_dtype("a"):
        return None
    return (
        f"warp TILE: atom {tile.atom.name} takes a {tile.atom.operand_dtype('a')} A operand but this "
        f"contraction's A is {a_dtype} — a copy transport cannot convert it. The mma tier is reachable "
        f"through the converting smem compute fill, or drop the atom for the scalar tier."
    )


def computed_operand_cover(c: Fold, tile: TilePlan, *, converting_a: bool = False) -> str | None:
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
    """
    if not c.axis.extent.is_static:
        if isinstance(c.a, Load) and not converting_a:
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


def computed_operand_copy_dtype(c: Fold, tile: TilePlan, inputs, *, converting_a: bool = False) -> str | None:
    """Every BYTE-COPIED edge of a compute-filled contraction must already have the atom dtype.

    The ``smem`` stage evaluates computed (and converting) operands into their typed shared-memory
    slabs, but it *copies* every materialized peer byte-for-byte.  A copied f32 edge therefore
    cannot feed an f16 ``ldmatrix`` fragment merely because another edge is filled. Filled edges
    are exempt because their slab store performs the normal typed conversion — ``converting_a``
    marks a materialized ``a`` that rides the converting fill rather than the copy.
    """
    for edge, role in ((c.a, "a"), *((ch.b, "b") for ch in c.channels)):
        if not isinstance(edge, Load) or (role == "a" and converting_a):
            continue
        tensor = inputs.get(edge.input) if inputs else None
        # Structural scheduler fixtures (and a few pre-stamp inventory callers) intentionally do
        # not carry Tensor metadata.  Absence is not evidence of an unsafe byte copy; the concrete
        # lowering path always supplies inputs and is where a known mismatch must be rejected.
        if tensor is None:
            continue
        want = tile.atom.operand_dtype(role)
        if tensor.dtype == want:
            continue
        got = tensor.dtype
        return (
            f"smem compute fill: materialized {role.upper()} edge {edge.input!r} is {got}, but "
            f"atom {tile.atom.name} copies it into a {want} slab without conversion; only the "
            "``a`` role has a converting fill"
        )
    return None


def resolve_fill_stage(
    c: Fold,
    tile: TilePlan,
    budget: int,
    want_depth: int = 1,
    inputs=None,
    why: list[str] | None = None,
    seam: tuple | None = None,
    k_axis: Axis | None = None,
    producer: Fold | None = None,
) -> Stage | None:
    """The ``smem`` compute-fill :class:`Stage` for a computed-operand warp contraction under ``tile``
    — MANDATORY for this form (the gmem-direct mma leaf refuses a computed A, and the byte-copy /
    cp.async / TMA transports move bytes and cannot evaluate a producer cone), so it has no
    gmem-direct ``""`` sibling and a ``STAGE`` pin can only choose its DEPTH. ``None`` when the slabs exceed ``budget``: one A
    slab, one B slab per channel, and one fp32 row per bridged statistic (``ops.cone_seam``'s
    ``stats`` — the same node boundary the materializer fills through).

    ``want_depth >= 2`` is the **asymmetric B-only prefetch ring**: only the B cp.async slabs ring
    (their copies for chunk ``i+d-1`` fly under chunk ``i``'s compute fill and drain), while the
    compute-filled A slab and the stat rows stay single-buffer — ringing a compute fill buys no
    overlap, it runs on the drain's own threads. Measured on the gemma gate_up edge at M=512 (5090)
    the ring LOSES (897 vs 665 µs) — the extra B slot alone crosses the smem occupancy quantization
    — but at decode M (``tile_m ≤ 32``) the A slab and stat rows are tiny and the tradeoff inverts,
    so both depths are enumerated as fork siblings and measured per shape.

    ``k_axis`` overrides the stored contraction axis when the contraction is a derived singleton
    marker whose enclosing Fold owns the actual K sweep. ``why`` collects the decline reason when
    the tier refuses, so a PINNED caller reports the gate it actually hit instead of one catch-all
    sentence."""
    atom = tile.atom
    k_axis = k_axis or c.axis
    if atom.operand_dtype("a").nbytes < 2:
        # fp8 atoms: the compute fill's slab store + ldmatrix drain are 16-bit-only
        decline(why, f"the smem compute fill is 16-bit-only, but this atom's a operand is {atom.operand_dtype('a').nbytes}-byte")
        return None
    bk_elems = tile.bk * atom.atom_k
    if k_axis.extent.is_static and k_axis.extent.as_static() % bk_elems:
        # the staged driver unrolls WHOLE K chunks — the same rule the copy transports state on their own
        decline(
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
    a_tensor = inputs.get(c.a.input) if inputs and isinstance(c.a, Load) else None
    a_converts = a_tensor is not None and a_tensor.dtype != atom.operand_dtype("a")
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
        decline(why, f"the smem compute fill's slabs need {sync_bytes + async_bytes + producer_bytes} B, over the {budget} B smem budget")
        return None
    fixed = sync_bytes + producer_bytes
    depth = want_depth if want_depth >= 2 and async_bytes and fixed + want_depth * async_bytes <= budget else 1
    computed = [operand_name(c.a)] if a_converts or not isinstance(c.a, Load) else []
    computed.extend(operand_name(ch.b) for ch in c.channels if not isinstance(ch.b, Load))
    return Stage(depth=depth, transport="smem", smem=tuple(computed), bk_elems=bk_elems)


def resolve_scalar_stage(c: Fold, tile: TilePlan, stage: Stage, inputs, budget: int) -> Stage | None:
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
    return replace(stage, depth=depth, reg_depth=1, bk_elems=bk_elems)


__all__ = [
    "computed_operand_cover",
    "computed_operand_copy_dtype",
    "enforce",
    "fragment_epilogue",
    "producer_band",
    "producer_transport",
    "resolve_scalar_stage",
    "decline",
    "resolve_fill_stage",
    "resolve_warp_stage",
    "scalar_block_threads",
    "splitk_width",
    "stage_target",
    "strip_width",
    "warp_k_step",
    "warp_atom_target",
    "warp_a_columns",
    "warp_operand_dtype",
]
