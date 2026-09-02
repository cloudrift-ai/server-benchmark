r"""The per-atom codegen strategies — the one seam every tiled contraction dispatches through.

``_factor._bind`` (the output-tiled contraction arm) reads the tiling geometry off the contraction
node and asks this module for the two codegen halves: :func:`reduce_codegen` (the sink-agnostic
``(state_decls, reduce_region)`` — the accumulator/operand decls + the shared :func:`_contract_kloop`
K-loop) and :func:`store_sink` (the per-cell matmul sink). Both resolve through :func:`_atom_ops` to
one of the two concrete strategies — :class:`_MmaOps` (tensor-core fragment loads + ``mma.sync``, a
``RegStore`` sink) or :class:`_ScalarOps` (plain ``Load``\ s + an ``fma`` cell, the replicated-
``epilogue`` sink). The K-loop itself is ONE driver on the base strategy (:meth:`_AtomOps.reduce`),
deciding nothing: the **scheduler-resolved** ``stage`` (eligibility + sizing ran once in
the scheduler; ``None`` = gmem-direct)
picks its form — gmem-direct through the shared :func:`_contract_kloop` spine, or staged through
the shared :func:`_staged` fill→drain skeleton (over the one ``_stage.staged_kloop``) — and the
atom supplies only descriptor reads: the four gmem leaf constructors (:meth:`gmem_leaves`), the
slab drain leaf (:meth:`staged_drain`), and the slab element dtype. This IS the "atom as
descriptor" seam: one factory, one loop over atoms, no scattered ``isinstance``.

A Fold with scheduled contraction children uses the same atom seam.  Its carrier components are
assigned uniform, row, or fragment residence, and its stored lift/combine Lambdas are evaluated at
those residences.  A scheduled child callback emits its contraction and returns fragment values;
the ordinary staging skeleton then composes the producer block with the value contraction.  This
keeps operation-family recognition out of kernel materialization and leaves the canonical Fold tree
unchanged.

Leading ``_`` so the pass loader (globs ``*.py``, skips ``_``-prefixed) skips it."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from emmy.compiler.backend.cuda.dtype import cuda_name
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.ir.address import BYTE_SLAB_PAD
from emmy.compiler.ir.atom import AtomKind
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal, TernaryExpr, Var
from emmy.compiler.ir.kernel.ir import (
    FRAG,
    ROW,
    UNIFORM,
    BlockScaleLoad,
    EpilogueLoad,
    FragmentApply,
    FragmentPromote,
    FragmentRowReduce,
    LdmatrixLoad,
    MmaSyncPtx,
    RegEpilogue,
    RegFragment,
    RegStore,
    frag_layout,
)
from emmy.compiler.ir.pure import Lambda
from emmy.compiler.ir.pure.fold import Channel, Fold, is_contraction, operand_body, operand_name, subst_free
from emmy.compiler.ir.schedule import Side, Stage, Tile
from emmy.compiler.ir.schedule.packing import block_scaled_atom, match_packed_b_node, match_packed_pair_node
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Accum, Assign, Body, Cond, Init, Load, Loop, Select, SelectBranch, Stmt, StridedLoop, Write
from emmy.compiler.ir.stmt.passes import rename_free
from emmy.compiler.ir.tile.ops import cone_stat_dtypes, make_cone
from emmy.compiler.pipeline.passes.lowering._reduction import Reduction
from emmy.compiler.pipeline.passes.lowering.kernel._eval import Value, evaluate
from emmy.compiler.pipeline.passes.lowering.kernel._stage import (
    CpAsyncTransport,
    CtaTile,
    LeadSegment,
    Operand,
    SyncOperand,
    SyncTransport,
    TmaTransport,
    pick_swizzle_atom,
    pipelined_kloop,
    software_swizzle,
    staged_kloop,
    sync_stat_fill,
)
from emmy.compiler.pipeline.passes.lowering.kernel._tiling import AxisOffset
from emmy.compiler.pipeline.search.space import UNROLL

#: The contraction semiring — multiply ⊗ then accumulate ⊕ (add). The same multiply-add ``mma.sync``
#: realizes; in the scalar tier it is a plain scalar fma loop.
_MUL = ElementwiseImpl("multiply")
_ADD = ElementwiseImpl("add")


@dataclass(frozen=True)
class _ScheduledContraction:
    """A scheduled child contraction evaluated over its enclosing Fold's block axis.

    The child owns the semiring, operands, and output tile.  The enclosing Fold owns the streamed
    axis and the carrier component receiving the child's result.  This adapter is materialization
    data only; it never rebuilds or mutates the canonical Fold tree.
    """

    child: Fold
    axis: Axis
    channels: tuple[Channel, ...]

    @property
    def a(self):
        return self.child.a

    @property
    def b(self):
        return self.channels[0].b

    @property
    def acc(self) -> str:
        return self.channels[0].acc

    @property
    def semiring(self):
        return self.child.semiring

    @property
    def b_trans(self) -> bool:
        return isinstance(self.b, Load) and self.axis.name in self.b.index[-1].free_vars()


def scheduled_fold_contraction(fold: Fold, sched):
    """Return the scheduled child that contributes a fragment-resident carrier component.

    The relation is read from the derived Fold step: a tiled contraction result consumed by an
    ``Accum`` into one of the enclosing carrier names.  This is a generic producer/consumer edge,
    independent of the operations in either Fold.
    """

    states = set(fold.combine.results) if fold.combine is not None else set()
    steps = fold.step_stmts()
    for child in (stmt for stmt in steps if is_contraction(stmt)):
        tile = sched.tile_of(child)
        consumers = tuple(stmt for stmt in steps if isinstance(stmt, Accum) and stmt.name in states and stmt.value == child.out)
        stage = sched.get("STAGE", child) if tile is not None else None
        if tile is None or not tile.is_warp or stage is None or stage.transport != "smem" or not consumers:
            continue
        if len(child.channels) != len(consumers):
            return None
        channels = tuple(Channel(channel.b, consumer.name) for channel, consumer in zip(child.channels, consumers, strict=True))
        return _ScheduledContraction(child=child, axis=fold.axis, channels=channels), child, tile, stage
    return None


def unroll_ok(extent, cap: int | None = None) -> bool:
    """Whether a static-``extent`` loop should ``#pragma unroll`` — its trip count within the effective
    budget: the ``EMMY_UNROLL`` pin when set, else the site's natural ``cap`` (``None`` = unroll any
    static loop, the tensor-core K-chunk default). A non-static extent never unrolls."""
    if not extent.is_static:
        return False
    n = extent.as_static()
    return n <= UNROLL.read_int(n if cap is None else cap)


def unroll_ok_n(trips: int, cap: int | None = None) -> bool:
    """:func:`unroll_ok` for a loop whose static trip count ``trips`` is already a Python ``int``."""
    return trips <= UNROLL.read_int(trips if cap is None else cap)


# Shared per-cell helpers, used across this module (the atom-generic mma/scalar codegen).
def copy_cell(body, sigma, suffix: str, protected) -> list:
    """One copy of a tiled reduce ``body``: σ-substitute its indices (``sigma``) and suffix every
    per-copy SSA name (the shared grid / reduce / lane coordinates in ``protected`` pass through
    unrenamed). This is the **one** replication mechanic shared by the register tile (this module,
    one copy per output cell ``(i, j)`` → ``__c{i}_{j}``) and the ILP register fold (``_factor``
    ``_tile_reduce_axis``, one copy per accumulator chain ``r`` → ``__r{r}``); the caller supplies the per-copy
    ``sigma`` (the coordinate offset) and ``suffix`` (the SSA tag)."""
    rename = lambda n: n if n in protected else f"{n}{suffix}"  # noqa: E731
    return [s.rewrite(rename, sigma) for s in body]


# The per-axis masked-overhang helpers — one :class:`Side` in, so the codegen never re-derives a
# ``mask_m`` / ``m_ext`` scalar pair: ``_guard`` predicates a store, ``_wrap`` clamp-reads an operand.
def _guard(side: Side, coord: Expr):
    """The overhang guard for cell ``coord`` on ``side`` — ``(coord, extent)`` when the axis is masked
    (its store is predicated on it), else ``None``."""
    return (coord, side.ext) if side.mask else None


def _wrap(side: Side, coord: Expr) -> Expr:
    """Wrap ``coord`` in-bounds (``coord % extent``) on a masked ``side`` so an overhanging cell
    clamp-reads, else ``coord`` unchanged."""
    return BinaryExpr("%", coord, side.ext) if side.mask else coord


def _axis_dim(index: tuple, axis_name: str) -> int | None:
    """The innermost output-index position carrying ``axis_name``, before cell substitution.

    ``RegStore`` derives each fragment-axis stride from this dimension. The innermost occurrence
    is load-bearing for a re-fused split axis spelled ``[…, f/P, f%P, …]``: its trailing extents
    give the coefficient of the reconstructed flat ``f`` coordinate.
    """
    dims = [d for d, e in enumerate(index) if axis_name in e.free_vars()]
    return dims[-1] if dims else None


def _cells(mn: tuple, offset, i: int, j: int):
    """Yield ``(side, cell-base coord)`` for each present output axis of register cell ``(i, j)`` —
    ``(m, offset[0].base(i))`` then ``(n, offset[1].base(j))`` (``m`` skipped for a 1-D output)."""
    for side, off, r in ((mn[0], offset[0], i), (mn[1], offset[1], j)):
        if side is not None:
            yield side, off.base(r)


# ---- warp/mma tier ----------------------------------------------------------------------------- #
def _warp_roles(index, m_name: str, n_name: str) -> tuple[str, ...]:
    """Per-dim epilogue-load role: ``"m"`` / ``"n"`` for the dim the output row / col axis moves
    within the fragment cell, else ``"fixed"`` (batch / grid literal — uniform across the cell).
    Only the INNERMOST dim carrying an axis moves (the same reading as :func:`_row_dim`): a
    re-fused split axis reaches the load as ``[…, f/Q, …, f%Q]``, and within an atom the
    quotient dim is uniform — giving both dims the role would add the lane offset at two
    strides."""
    roles = ["fixed"] * len(index)
    for role, name in (("n", n_name), ("m", m_name)):
        dims = [d for d, e in enumerate(index) if name in e.free_vars()]
        if dims:
            roles[dims[-1]] = role
    return tuple(roles)


def _warp_epilogue(
    tail: list[Stmt], acc: str, m_name: str, n_name: str, sigma: Sigma, extra_accs: tuple[tuple[str, str], ...] = ()
) -> RegEpilogue | None:
    """Fold the projection (zero-axis) fold into a :class:`RegEpilogue` for cell ``sigma``. ``None`` when
    there is no projection (a bare ``Write`` of the accumulator). ``extra_accs`` binds a multi-fold
    node's additional ``(acc, C-fragment)`` pairs so the chain combines the channels per element.

    The projection is the post-reduce ``tail`` stmts: the leaf ``Load``s + pointwise ``Assign``s +
    an optional causal ``Select``. Each leaf ``Load`` becomes an :class:`EpilogueLoad` at the
    cell-base coordinate (σ-applied; the render adds the per-element row/col motion on the
    ``m``/``n`` dims); each ``Assign`` becomes an ``(name, op, args, dtype)`` op; a coord-predicated
    ``Select`` (causal mask) captures its σ-applied cell bases plus ``__M__`` / ``__N__``
    placeholders; the store substitutes only the element's row/col offsets. This keeps semantic
    source coordinates independent of a later store to a tile-local shared-memory slab. Keeping
    the optional dtype makes the register epilogue obey the scalar Loop tail's promotion and
    conversion rules."""
    loads, ops, selects = [], [], []
    write = None
    ph = {
        m_name: BinaryExpr("+", sigma.apply(Var(m_name)), Var("__M__")),
        n_name: BinaryExpr("+", sigma.apply(Var(n_name)), Var("__N__")),
    }
    for s in tail:
        if isinstance(s, Load):
            loads.append(
                EpilogueLoad(
                    name=s.names[0],
                    buffer=s.input,
                    index=tuple(sigma.apply(e) for e in s.index),
                    roles=_warp_roles(s.index, m_name, n_name),
                )
            )
        elif isinstance(s, Assign):
            ops.append((s.name, s.op.name, tuple(s.args), s.dtype))
        elif isinstance(s, Select):
            selects.append((s.name, tuple((br.select.substitute(ph), br.value) for br in s.branches)))
        elif isinstance(s, Write):
            write = s
    if write is None or (not ops and not selects):
        return None
    # Every chain-op arg must be BOUND in the render env (the accumulator(s), a leaf load, a
    # select, or an earlier op) — an unbound name means the variant's projection tail reads a
    # value this node does not compute (a mis-sliced multi-channel combine: the gemma GeGLU
    # tail on a single-fold row referenced the sibling channel's ``acc2`` and died with a
    # ``KeyError`` in the RegStore render). Decline the variant cleanly instead.
    from emmy.compiler.pipeline import RuleSkipped  # noqa: PLC0415 — avoid an import cycle

    bound = {acc, *(a for a, _ in extra_accs), *(ld.name for ld in loads), *(nm for nm, _ in selects)}
    for name, _op, args, _dtype in ops:
        unbound = [a for a in args if a not in bound]
        if unbound:
            raise RuleSkipped(f"projection epilogue reads {unbound} this node does not compute (mis-sliced multi-channel tail)")
        bound.add(name)
    return RegEpilogue(acc=acc, loads=tuple(loads), ops=tuple(ops), result=write.value, selects=tuple(selects), extra_accs=extra_accs)


# ---- operand staging (smem slab + ldmatrix drain) ---------------------------------------------- #
# The warp tier's smem operand pipeline, driven off the node's :class:`Stage`. Synchronous copy, cp.async, and TMA
# share the slab + the staged-``LdmatrixLoad`` drain (:func:`_staged_inner_atom_loop`) AND the
# slab swizzle modes (:meth:`_MmaOps.slab_swizzles`); only who applies the fill-side permutation
# differs — TMA swizzles in hardware during the box copy, a cp.async fill XORs its destination
# index in software — and the drain XOR undoes either. Staging is a **pure perf
# transform**: an ineligible kernel silently falls back to gmem-direct, and a staged kernel is
# bit-identical to its gmem-direct baseline. The transport primitives (the fill loops + the
# commit/wait / mbarrier handshakes) live in ``_stage.py``; these functions schedule them onto the
# K-loop off the contraction geometry.
def _fold_frag(base: str, fold: int) -> str:
    """The per-fold-channel fragment name — the primary channel keeps the historic bare spelling
    (``_b0`` / ``_c0_0``), extra channels suffix ``_x<f>`` (the multi-B gate/up node)."""
    return base if fold == 0 else f"{base}_x{fold}"


# The f16-accumulate (``_f16acc`` atom) chunked-promote scheme: the mma chain accumulates into
# packed f16 fragments at the full HMMA rate, and a periodic ``FragmentPromote`` folds them into
# f32 shadow fragments (which keep the ``_c{i}_{j}`` names the store / epilogue reads, so the sink
# is untouched). Staged rows promote once per bk chunk (the chunk IS the cadence); the gmem-direct
# K-loop promotes every ``_F16ACC_STEPS`` atom-K steps plus a final fold after the loop.
_F16ACC_STEPS = 4  # gmem-direct promote cadence: atom-K steps per f16 chunk (64 K-elems at k16)


def _f16acc(atom) -> bool:
    """Whether ``atom`` is an f16-accumulate mma cell (a 16-bit C operand) — selects the chunked
    f16→f32 promote scheme above."""
    return isinstance(atom, AtomKind) and atom.operand_dtype("c").nbytes == 2


def _mma_c_base(atom, i: int, j: int) -> str:
    """The mma-target C fragment base name for cell ``(i, j)`` — the packed f16 ``_ch{i}_{j}`` on
    an f16-accumulate atom (its f32 shadow keeps the ``_c{i}_{j}`` name the store reads), else the
    plain ``_c{i}_{j}``."""
    return f"_ch{i}_{j}" if _f16acc(atom) else f"_c{i}_{j}"


def _f16acc_promotes(m_reg: int, n_reg: int, n_folds: int, frag_ns: str = "") -> list[Stmt]:
    """One :class:`FragmentPromote` per C cell × fold channel — the f16 chunk fold into the f32
    shadows (also the FINAL fold: the shadows carry the full sum only after it runs)."""
    return [
        FragmentPromote(dst=_fold_frag(f"{frag_ns}_c{i}_{j}", f), src=_fold_frag(f"{frag_ns}_ch{i}_{j}", f))
        for f in range(n_folds)
        for i in range(m_reg)
        for j in range(n_reg)
    ]


def _staged_inner_atom_loop(
    *,
    slabs: tuple[str, ...],
    mn: tuple[Side, Side],
    atom,
    bk_elems,
    ki,
    reg_depth: int = 1,
    offs=None,
    swizzles=None,
    trans=None,
    byte_slabs=None,
    pads=None,
    frag_ns: str = "",
    scales=None,
) -> list[Stmt]:
    """The inner atom-K drain shared by every staged path: read the A/B ``slabs`` via
    ``LdmatrixLoad(staged=True)`` + ``MmaSyncPtx``. The leaf uses modern ``ldmatrix`` instructions
    or the Volta cooperative shared gather according to the atom fragment layout. ``slabs`` is ``(A, B…)`` — one B slab per fold
    channel (one for the ordinary matmul; N for the multi-B gate/up node, whose ONE ldmatrix'd A
    fragment feeds a per-channel mma chain into a per-channel C fragment). Slab-local indices —
    A[tile_m][bk_elems] (ldm=bk_elems), B[bk_elems][tile_n] (ldm=tile_n) — independent of which
    producer filled the (plain row-major, NONE-swizzle) slab; ``mn`` is the ``(m, n)`` :class:`Side`
    pair.

    ``reg_depth == 1`` (default): one ``StridedLoop`` over the ``bk`` atom-K steps, fragment-load-then-mma
    inline (the operand fragments ``_a{i}``/``_b{j}`` reused every step). ``reg_depth >= 2`` (the
    ``STAGE`` ``/p<n>`` smem→register double-buffer): the loop is **fully unrolled** into a software
    pipeline that ldmatrixes the next atom-K step into an alternate fragment slot (``_a{i}_s{slot}``)
    ``reg_depth-1`` steps ahead while the mma consumes the current slot — breaking the per-step WAR
    hazard on the operand fragments. Numerically identical to the inline form.

    ``offs`` (the gmem→smem ring, ``STAGE`` depth>1): the per-slab read SLOT row offsets — added to
    each slab's ROW (A's tile row / B's K row) so the drain reads the ring slot the producer already
    filled, while a later chunk prefetches into another slot.

    ``swizzles`` (per-slab, aligned with ``slabs``): the smem swizzle mode each slab was written
    with (TMA in-copy, or the cp.async fill's software XOR) — threaded onto each ``LdmatrixLoad``
    so its address XOR undoes the fill's chunk permutation (``"NONE"`` reads the plain row-major
    slab).

    ``trans`` (per-slab, aligned with ``slabs``): a transposed-B N-MAJOR slab (``Operand.trans``
    — ``tile_n`` rows × ``bk_elems`` K cols, the serving ``F.linear`` layout staged in its own
    gmem orientation): the B read takes A's row/col convention (tile coord the ROW, K the col,
    ``ldm = bk_elems``) and drains via the plain (no ``.trans``) ldmatrix — the
    ``LdmatrixLoad(b_trans=True)`` staged path. Always ``False`` for A.

    ``byte_slabs`` (per-slab, aligned with ``slabs``): a 1-byte (fp8) slab — ldmatrix is b16-only
    below sm_100a, so its drain is the cooperative per-lane byte gather instead
    (``LdmatrixLoad(byte_slab=True)`` — the gmem fragment-loader lane map pointed at the slab; a
    16-bit fragment converts per element, an fp8 fragment repacks raw bytes). NONE-swizzle by
    construction; ``pads`` (per-slab row pad in elements, the cp.async byte slabs'
    ``BYTE_SLAB_PAD``) rides the drain ``ldm`` so reads stride the padded rows.

    ``scales`` (per-slab, aligned with ``slabs``): a ``(scale slab, its row stride, the k block)``
    triple on a PACKED-PAIR byte slab (NVFP4 weights — one stored byte is two logical K elements
    sharing one block scale). The slab's K columns are then BYTE columns, so the drain halves its
    K coordinate to address them and reads the block's scale at ``K / block`` of the companion
    slab; the loader decodes both nibbles and applies the scale. ``None`` on every other slab."""
    (a_slab, *b_slabs), (m, n) = slabs, mn
    offs = offs if offs is not None else (None,) * len(slabs)
    swizzles = swizzles if swizzles is not None else ("NONE",) * len(slabs)
    trans = trans if trans is not None else (False,) * len(slabs)
    byte_slabs = byte_slabs if byte_slabs is not None else (False,) * len(slabs)
    pads = pads if pads is not None else (0,) * len(slabs)
    scales = scales if scales is not None else (None,) * len(slabs)
    atom_m, atom_n, atom_k = atom.shape
    n_steps = bk_elems // atom_k
    # Per-operand drain spec: (frag base fn, slab, ldm, tile-is-slab-row, reg count, warp-unit var,
    # atom dim, slot row off, swizzle, byte flag). A stacks the tile axis on the slab row (K the
    # col); B swaps (K the row, tile the col) — unless its slab is transposed (N-major: tile the
    # row, K the col, like A); the slot offset always lands on the ROW. All share ONE emission loop.
    # A packed-pair slab's K columns are BYTE columns — half as many as the chunk's logical K —
    # on either side; only the W4A16 shape (packed B beside a 16-bit A) leaves A at full width.
    a_cols = bk_elems // 2 if scales[0] is not None else bk_elems
    specs = [
        (
            lambda x: f"{frag_ns}_a{x}",
            "a",
            a_slab,
            a_cols + pads[0],
            True,
            m.reg,
            m.unit,
            atom_m,
            offs[0],
            swizzles[0],
            byte_slabs[0],
            scales[0],
            lambda x: f"_sfa{x}",
        )
    ]
    for f, bs in enumerate(b_slabs):
        frag_of = (lambda ff: lambda x: _fold_frag(f"{frag_ns}_b{x}", ff))(f)
        # The block-scale fragment is named PER CHANNEL, exactly as the data fragment is: with one
        # name for every channel the second channel would read the first channel's scales, which
        # is wrong by a per-block factor and invisible in the emitted source. The A side keeps one
        # name because it is genuinely shared.
        sf_of = (lambda ff: lambda x: _fold_frag(f"_sfb{x}", ff))(f)
        tr, sc = trans[1 + f], scales[1 + f]
        # A packed-pair slab's K columns are BYTE columns — half as many as the chunk's logical K.
        k_cols = bk_elems // 2 if sc is not None else bk_elems
        ldm_b = (k_cols if tr else n.tile) + pads[1 + f]
        specs.append((frag_of, "b", bs, ldm_b, tr, n.reg, n.unit, atom_n, offs[1 + f], swizzles[1 + f], byte_slabs[1 + f], sc, sf_of))

    # The BLOCK-SCALED cell (both multiplicands packed pairs): its scales do not fold into the
    # fragments — the instruction takes them as two more register operands and applies them
    # itself. So each side's scale slab feeds a ``BlockScaleLoad`` of its own instead of riding
    # the data drain, and the data drain is a plain packed-byte gather.
    block_scaled = block_scaled_atom(atom)

    def ldms(kexpr, suffix):  # every operand's ldmatrix reads at K position `kexpr`, into fragment slot `suffix`
        reads: list[Stmt] = []
        for frag_of, role, slab, ldm, is_row, reg, unit, adim, off, swz, b8, sc, sf_of in specs:
            assert not (b8 and swz != "NONE"), "a byte slab stays NONE-swizzle (the ldmatrix XOR is b16-indexed)"
            for x in range(reg):  # within-tile coord for register cell x: warp·(reg·adim) + x·adim
                prim = BinaryExpr("+", BinaryExpr("*", Var(unit), Literal(reg * adim, "int")), Literal(x * adim, "int"))
                row, col = (prim, kexpr) if is_row else (kexpr, prim)
                scale_slab, scale_index, scale_ldm = None, (), 0
                if sc is not None:
                    scale_slab, scale_ldm, block = sc
                    # The W4A16 drain's scale slab is COMPUTE-FILLED and single-buffer, so its row
                    # is the bare within-tile coord. The block-scaled cell's two scale slabs are
                    # copies riding the same ring as their codes, so they carry the slot row like
                    # any other copied operand — one row per slot, the same count, so the data
                    # slab's own offset serves. Without it the scales are read from slot 0 while
                    # the codes come from slot s, and a subset of cells is scaled by the wrong
                    # block (found live: most outputs exact, a minority off by a factor of 2-3).
                    scale_row = BinaryExpr("+", off, prim) if (block_scaled and off is not None) else prim
                    scale_index = (scale_row, BinaryExpr("/", kexpr, Literal(block, "int")))
                    col = BinaryExpr("/", col, Literal(2, "int"))
                    if block_scaled:
                        reads.append(
                            BlockScaleLoad(
                                frag=f"{sf_of(x)}{suffix}",
                                src_buffer=scale_slab,
                                src_index=scale_index,
                                role=role,
                                ldm=scale_ldm,
                            )
                        )
                        scale_slab, scale_index, scale_ldm = None, (), 0
                if off is not None:
                    row = BinaryExpr("+", off, row)
                frag = f"{frag_of(x)}{suffix}"
                reads.append(
                    LdmatrixLoad(
                        frag=frag,
                        src_buffer=slab,
                        src_index=(row, col),
                        role=role,
                        staged=True,
                        ldm=ldm,
                        swizzle=swz,
                        b_trans=role == "b" and is_row,
                        byte_slab=b8,
                        scale_buffer=scale_slab,
                        scale_index=scale_index,
                        scale_ldm=scale_ldm,
                        fragment_layout=atom.fragment_layout,
                    )
                )
        return reads

    def mmas(suffix):  # every fold channel × (i, j) cell's mma.sync over the `suffix`-slotted operand fragments
        return [
            MmaSyncPtx(
                c_frag=_fold_frag(f"{frag_ns}{_mma_c_base(atom, i, j)}", f),
                a_frag=f"{frag_ns}_a{i}{suffix}",
                b_frag=f"{_fold_frag(f'{frag_ns}_b{j}', f)}{suffix}",
                shape=atom.ptx_shape,
                ab_dtype=atom.ab_dtype,
                c_dtype=atom.operand_dtype("c").name,
                sfa_frag=f"_sfa{i}{suffix}" if block_scaled else None,
                sfb_frag=f"{_fold_frag(f'_sfb{j}', f)}{suffix}" if block_scaled else None,
            )
            for f in range(len(b_slabs))
            for i in range(m.reg)
            for j in range(n.reg)
        ]

    if reg_depth < 2 or n_steps < 2:  # single-buffer: the inline fragment-load → mma loop
        body = ldms(Var(ki), "") + mmas("")
        return [
            StridedLoop(
                axis=Axis(name=ki, extent=bk_elems),
                start=Literal(0, "int"),
                step=Literal(atom_k, "int"),
                body=Body(tuple(body)),
                unroll=True,
            )
        ]

    # reg_depth ≥ 2: the unrolled register double-buffer. ``slot = step % depth`` cycles the fragment
    # buffers; prefetch runs ``depth-1`` steps ahead of the consuming mma.
    depth = min(reg_depth, n_steps)
    kcol = lambda step: Literal(step * atom_k, "int")  # slab-local K col of atom-K step `step`  # noqa: E731
    stmts: list[Stmt] = []
    for s in range(depth - 1):  # prologue: prime the first depth-1 steps
        stmts += ldms(kcol(s), f"_s{s % depth}")
    for step in range(n_steps):
        nxt = step + depth - 1
        if nxt < n_steps:  # prefetch depth-1 ahead, into the slot the mma below frees
            stmts += ldms(kcol(nxt), f"_s{nxt % depth}")
        stmts += mmas(f"_s{step % depth}")
    return stmts


def clamp_last(idx: Expr, ext: Expr) -> Expr:
    """Clamp an overhanging gmem coordinate to the last valid index — the overhanging cell still
    reads an in-bounds (duplicate) operand, and its store is discarded by the guard (``RegStore`` /
    ``Cond``)."""
    return TernaryExpr(cond=BinaryExpr("<", idx, ext), if_true=idx, if_false=BinaryExpr("-", ext, Literal(1, "int")))


def _side_base(side: Side) -> Expr:
    """The CTA's tile-base coordinate on ``side`` — ``block·tile`` (always in-bounds: the block
    count is ``ceil(extent / tile)``, so ``block·tile < extent``)."""
    return BinaryExpr("*", Var(side.block), Literal(side.tile, "int"))


def _sibling_sigma(sibling: Side | None) -> dict[str, Expr]:
    """The σ entry for the operand's OTHER tiled output axis. A staged slab is CTA-shared across
    the sibling axis, so the drain contract already requires the operand's gmem address be
    sibling-invariant in VALUE — but the sibling var can still appear SYNTACTICALLY, through a
    flat-index reshape residue (a merged / reshaped weight row like ``(m·N + n) % N``, whose ``m``
    contribution is a multiple of the modulus and folds away). After the tile split the kernel
    decodes only the ``_b`` / ``_u`` split vars, never the bare axis name, so leaving the sibling
    unsubstituted emits an undefined identifier. Bind it to the CTA's block base — an in-bounds
    representative under which a value-dead residue evaluates unchanged."""
    return {} if sibling is None else {sibling.axis.name: _side_base(sibling)}


def _slab_index(operand_index, *, tile: Side, tile_base, k_axis, tile_is_row: bool, sibling: Side | None = None):
    """The **one** cp.async slab gmem-index factory, for either operand and either tier. The slab's
    inner (contiguous) dim maps to the contraction ``k_axis``, its outer dim to the stationary ``tile``
    axis (``m`` for A, ``n`` for B). For A the tile axis is the slab ROW (K the col); for B they swap
    (``slot[row][col] = A[row_base + row][k0 + col]`` / ``B[k0 + row][col_base + col]``). A masked tile
    coordinate is clamped in-bounds — the overhanging cell reads a duplicate and its store is guarded.
    A residual reference to the ``sibling`` output axis binds to its block base (:func:`_sibling_sigma`).
    Returns a ``k0 -> ((row, col) -> gmem index)`` map — one K-chunk offset per :func:`staged_kloop`
    fill."""

    def at(k0):
        def gmem(row, col):
            tc, kc = (row, col) if tile_is_row else (col, row)
            t = BinaryExpr("+", tile_base, tc)
            k = BinaryExpr("+", k0, kc)
            # A SYMBOLIC K's last chunk overhangs the extent. Here K is the slab's OUTER dim (the
            # contiguous copy chunk runs along the tile axis), so the overhanging row clamps to the
            # last valid one exactly as a masked tile row does — a duplicate row is copied and the
            # compute fill zeroes the matching A lanes, so it folds to nothing. The K-major
            # orientations have no such row and stay refused (``computed_operand_cover``).
            if not tile_is_row and not k_axis.extent.is_static:
                k = clamp_last(k, k_axis.extent_expr())
            sig = Sigma(
                {
                    tile.axis.name: clamp_last(t, tile.ext) if tile.mask else t,
                    k_axis.name: k,
                    **_sibling_sigma(sibling),
                }
            )
            return tuple(sig.apply(e) for e in operand_index)

        return gmem

    return at


def _tile_base(mn: tuple[Side, Side]) -> tuple[Expr, Expr]:
    """The CTA tile's ``(row_base, col_base)`` top-left origin — ``(m_b·tile_m, n_b·tile_n)``."""
    return tuple(_side_base(s) for s in mn)


def _box_origin(operand_index, *, tile: Side, tile_base: Expr, k_axis, sibling: Side | None = None):
    """The TMA box origin at K-chunk ``k0`` — the operand's OWN gmem index evaluated (σ) at the
    tile base and ``k0``, so an offset operand (a split-K partial's ``ksplit·(K/w) + k``) lands
    the box at its absolute coordinates. For a canonical operand this is exactly ``(tile_base,
    k0)`` (A, tile axis the slab row) / ``(k0, tile_base)`` (B). A residual reference to the
    ``sibling`` output axis binds to its block base (:func:`_sibling_sigma`)."""

    def at(k0):
        sig = Sigma({tile.axis.name: tile_base, k_axis.name: k0, **_sibling_sigma(sibling)})
        return tuple(sig.apply(e) for e in operand_index)

    return at


def _slab_operands(
    *,
    index_srcs: tuple,
    bufs: tuple[str, str],
    mn: tuple[Side, Side],
    k_axis,
    bk_elems: int,
    base: tuple[Expr, Expr],
    swizzles: tuple[str, str] = ("NONE", "NONE"),
    elems: tuple = (None, None),
    b_trans: bool = False,
    pads: tuple[int, int] = (0, 0),
):
    """The staged ``(A, B)`` :class:`Operand` pair — the one operand-geometry factory both tiers build,
    looped over the two operands. A is ``(tile_m × bk)`` indexed by the M tile axis (the slab ROW); B is
    ``(bk × tile_n)`` by the N tile axis (the slab COL) — ``is_row`` flips the slot shape + the TMA box
    origin. A transposed B (``b_trans``, the serving ``F.linear`` layout — K gmem-contiguous) takes
    A's geometry instead: an N-MAJOR ``(tile_n × bk)`` slab whose inner dim maps stride-1 to gmem K,
    so the fill's chunks stay contiguous; the ``Operand.trans`` stamp routes the drain to the plain
    (no ``.trans``) ldmatrix. ``base`` is the ``(row_base, col_base)`` CTA tile origin; ``index_srcs``
    are the operands' gmem index expressions (``load.index``); ``swizzles`` the per-operand smem
    swizzle modes (the mma tier's :meth:`_MmaOps.slab_swizzles` — ``("NONE", "NONE")`` everywhere
    else). ``elems`` are the per-operand element dtypes (``DataType`` or ``None`` = the
    transport-level dtype) — a mixed-dtype scalar contraction (fp32 A × fp16 B) must size each slab
    and fill by its OWN element width. ``pads`` are the per-operand slab row pads in elements
    (:data:`~emmy.compiler.ir.address.BYTE_SLAB_PAD` on a
    cp.async-staged byte slab; 0 everywhere else)."""
    ops: list[Operand] = []
    for i, (tag, is_row) in enumerate((("a", True), ("b", b_trans))):
        tile, tile_base, sibling = mn[i], base[i], mn[1 - i]
        shape = (tile.tile, bk_elems) if is_row else (bk_elems, tile.tile)
        elem = elems[i]
        # A >2-D operand (batched / unit-batch view) boxes as rank-N with leading extent-1 dims;
        # ``_box_origin`` already yields the full-rank origin (the leading index exprs ride
        # through σ untouched — the stage resolvers gated them tile/K-invariant). The flash K/V
        # ``(1, 1, bn, head_dim)`` convention, extended to the matmul tiers.
        box = (1,) * (len(index_srcs[i]) - 2) + shape if len(index_srcs[i]) > 2 else None
        ops.append(
            Operand(
                tag=tag,
                buf=bufs[i],
                shape=shape,
                box=box,
                coords=_box_origin(index_srcs[i], tile=tile, tile_base=tile_base, k_axis=k_axis, sibling=sibling),
                index=_slab_index(index_srcs[i], tile=tile, tile_base=tile_base, k_axis=k_axis, tile_is_row=is_row, sibling=sibling),
                swizzle=swizzles[i],
                dtype=cuda_name(elem) if elem is not None else None,
                elem_bytes=elem.nbytes if elem is not None else None,
                trans=i == 1 and b_trans,
                pad_cols=pads[i],
            )
        )
    return tuple(ops)


def _cta(mn: tuple[Side, Side], lanes: int, n_threads: int) -> CtaTile:
    """The staging :class:`CtaTile` for either atom — the intra-CTA linear thread id from the
    decoded unit axis vars (never a raw threadIdx.x): unit-major ``m_unit·units_n + n_unit``,
    ``·lanes + _lane`` when the unit is a warp (``lanes > 1``; a scalar unit IS one thread)."""
    m, n = mn
    tid = BinaryExpr("+", BinaryExpr("*", Var(m.unit), Literal(n.units, "int")), Var(n.unit))
    if lanes > 1:
        tid = BinaryExpr("+", BinaryExpr("*", tid, Literal(lanes, "int")), Var("_lane"))
    return CtaTile(linear_tid=tid, n_threads=n_threads)


def _stat_slab(name: str) -> str:
    """The smem row buffer bridging per-row statistic ``name`` from the sync prologue to the fill."""
    return f"_a_stat_{name}"


def _k_masked(stmts: list[Stmt], value: str, k: Expr, k_ext: Expr | None) -> tuple[list[Stmt], str]:
    """The smem compute fill's **K MASK** — the clamp-to-identity discipline the copy transports
    already follow, applied to the contraction axis.

    A SYMBOLIC K lets the last slab chunk overhang the runtime extent (a static K reaches the fill
    only through ``warp_k_step``, which already demands the chunk divide it). The drain reads the
    whole chunk unconditionally, so the overhang must be a REAL identity, not a skipped store: bind
    the fold identity and ``Select`` it whenever ``k >= K``. The identity is 0
    because the bilinear reading a compute fill exists for pins ⊕ = ``add`` (``Fold._contraction``),
    so a zero operand contributes nothing to the accumulator.

    Predicating the STORED value rather than the read is what makes it honest — the caller has
    already clamped the cone's own coordinate in-bounds so nothing loads out of range, but a clamped
    read produces a real (duplicate) value, and only this ``Select`` keeps it out of the fold.
    EVERY compute-filled edge takes it, not just ``a``: a zero A would already absorb a finite
    overhanging B, but ``0 · inf`` is a NaN, so a peer cone whose clamped evaluation is not finite
    needs its own identity. ``k_ext is None`` (a static K) returns the stmts untouched, so every
    static row stays bit-identical."""
    if k_ext is None:
        return stmts, value
    ident, masked = f"{value}__kid", f"{value}__km"
    return (
        [
            *stmts,
            Init(name=ident, identity=_ADD.identity, dtype=F32),
            Select(name=masked, branches=(SelectBranch(value, BinaryExpr("<", k, k_ext)), SelectBranch(ident, Literal(1, "int")))),
        ],
        masked,
    )


#: A nested contraction's slab tags — its own operand namespace, so ``_s_a_smem`` / ``_s_b_smem`` never
#: collide with the enclosing transport's ``_a_smem`` / ``_b_smem``.
_INVARIANT_TAG, _STREAM_TAG = "s_a", "s_b"


def _child_stream_operand(*, c: Fold, child: Fold, atom, m_name: str, cols: int) -> Operand | None:
    """A nested contraction's streamed operand, staged once per CTA instead of
    re-read per warp through the gmem fragment loaders, or ``None`` where the geometry does not.

    The two axes swap the roles the enclosing contraction's operands give them: what ADVANCES per
    chunk is the child's own N axis, and the child's own
    K is stationary and stages whole. So the slab is ``chunk × <child K>``, N-MAJOR
    — the operand's own gmem orientation, its K stride-1 in gmem and smem alike, which is what lets
    the ordinary staged ``LdmatrixLoad(b_trans=True)`` read it back.

    Declines a symbolic inner extent (the slab is a static shape), one whose row is not a whole number
    of ``cp.async`` chunks, a K-major streamed layout (the copy would stride, not run), an operand that
    indexes the invariant tile (the slab is CTA-shared across it), and a symbolic stream extent (whose
    last chunk overhangs, and only the gmem-direct read carries that guard)."""
    b = child.b
    if not (isinstance(b, Load) and child.b_trans and child.axis.extent.is_static and c.axis.extent.is_static):
        return None
    k_ext = child.axis.extent.as_static()
    if (k_ext * atom.operand_dtype("b").nbytes) % 16 or Body.coerce(()).depends_on(b, m_name):
        return None

    def index(k0):
        def gmem(row, col):
            return tuple(Sigma({c.axis.name: BinaryExpr("+", k0, row), child.axis.name: col}).apply(e) for e in b.index)

        return gmem

    def coords(k0):
        return tuple(Sigma({c.axis.name: k0, child.axis.name: Literal(0, "int")}).apply(e) for e in b.index)

    return Operand(
        tag=_STREAM_TAG,
        buf=b.input,
        shape=(cols, k_ext),
        index=index,
        coords=coords,
        trans=True,
        swizzle=software_swizzle(k_ext, atom.operand_dtype("b").nbytes),
    )


def _child_invariant_operand(*, child: Fold, atom, m: Side, k_name: str) -> Operand | None:
    """A nested contraction's invariant operand, staged once ahead of the chunk loop.

    Unlike the streamed operand, this edge does not advance with the chunk: the same ``tile_m × <child K>`` tile
    feeds every block of the sweep, so this is the loop-INVARIANT staged operand
    (:attr:`SyncTransport.invariant_operands`) — one fill, no ring, no live range to schedule. Its
    slab is the canonical A geometry (the tile axis the row, the child's K the contiguous column),
    read back by the ordinary staged ``LdmatrixLoad``. ``None`` where the geometry does not stage:
    a symbolic or non-chunk-aligned inner extent, a source whose K is not gmem-contiguous, or a
    masked invariant tile (whose overhang the slab fill would have to clamp)."""
    a = child.a
    if not (isinstance(a, Load) and child.axis.extent.is_static) or m.mask:
        return None
    k_ext = child.axis.extent.as_static()
    if (k_ext * atom.operand_dtype("a").nbytes) % 16 or k_name not in a.index[-1].free_vars():
        return None
    row_base = _side_base(m)

    def index(_k0):
        def gmem(row, col):
            return tuple(Sigma({m.axis.name: BinaryExpr("+", row_base, row), k_name: col}).apply(e) for e in a.index)

        return gmem

    def coords(_k0):
        return tuple(Sigma({m.axis.name: row_base, k_name: Literal(0, "int")}).apply(e) for e in a.index)

    return Operand(
        tag=_INVARIANT_TAG,
        buf=a.input,
        shape=(m.tile, k_ext),
        index=index,
        coords=coords,
        swizzle=software_swizzle(k_ext, atom.operand_dtype("a").nbytes),
    )


@dataclass(frozen=True)
class _BlockCols:
    """A nested tile's column offset — the enclosing K block it covers. Duck-types :class:`AxisOffset`'s one
    reader (:meth:`base`): a block column has no grid block and no unit, only the block base."""

    atom_dim: int
    k0: Expr

    def base(self, r: int) -> Expr:
        return BinaryExpr("+", self.k0, Literal(r * self.atom_dim, "int")) if r else self.k0


def _child_contraction_block(
    *,
    child: Fold,
    tile: Tile,
    k0: Expr,
    lead: tuple,
    ns: str,
    epilogue=None,
    slabs: tuple = (None, None),
):
    """One scheduled contraction block producing fragments for an enclosing Fold.

    Built out of the same atom strategy every tiled contraction dispatches through
    (:func:`_atom_ops` on the child node), so there is no second mma emitter here: its ``state``
    declares the fragments, its ``reduce`` emits the ``ldmatrix`` + ``mma.sync`` K-loop over the
    child's own axis, and its ``store`` — when the caller supplies an ``epilogue`` — writes them
    out. ``ns`` namespaces the fragments, so a nested emission never shadows the accumulators the
    enclosing drain carries across the same loop; ``slabs`` hands it the child's own staged
    operands (:func:`_child_invariant_operand` / :func:`_child_stream_operand`), each ``None`` reading
    gmem-direct instead.

    The rows are the enclosing tile's own, warp-partitioned and absolute; the columns are the
    enclosing block. Returns ``(ops, cells, offset, mn, stmts, frags)`` with ``frags[i]`` the
    register row ``i``'s column fragments in column order."""
    m, n = tile.mn
    ops = _atom_ops(child, tile, epilogue=epilogue, lead=lead, frag_ns=ns, slabs=slabs)
    offset = (
        AxisOffset(atom_dim=tile.atom.atom_m, reg=m.reg, block_var=m.block, unit_var=m.unit, unit_count=m.units),
        _BlockCols(atom_dim=tile.atom.atom_n, k0=k0),
    )
    cells = [(i, j) for i in range(m.reg) for j in range(n.reg)]
    decls = list(ops.state(cells))
    _, region = ops.reduce(cells, offset, tile.mn)
    frags = tuple(tuple(ops.frag(f"_c{i}_{j}") for j in range(n.reg)) for i in range(m.reg))
    return ops, cells, offset, tile.mn, [*decls, *region], frags


def _a_slab_operand(c: Fold, *, mn, bk_elems, cta, swizzle, seam, row_base, m_coord, k_coord, k_ext, inputs=None):
    """The A slab's operand, plus any statistic prologue it needs.

    A is COPIED when it is a materialized ``Load`` and COMPUTE-FILLED when it is a producer cone —
    a fused RMSNorm ahead of the projection, which is what a serving program's linears look like.

    Shared by the ``smem`` compute fill and the packed byte-slab stage. Those two differ in how B
    moves, never in how A does, and keeping one A side is what lets a packed weight sit behind a
    fused activation: the bits still copy verbatim while the activation evaluates into its slab.

    Returns ``(operand, copied, prologue)``."""
    pro, cell, stats = seam
    m_name, k_name = mn[0].axis.name, c.axis.name
    if isinstance(c.a, Load):
        shape = (mn[0].tile, bk_elems)
        op = Operand(
            tag="a",
            buf=c.a.input,
            shape=shape,
            # A >2-D operand boxes as rank-N with leading extent-1 dims — the convention
            # ``_slab_operands`` applies. ``_box_origin`` already yields the FULL-RANK origin, so
            # a box left at the 2-D shape gives the emitted copy more coordinates than the
            # descriptor's encoded rank, and TMA treats that as an invalid tensor map (measured
            # on a leading unit batch axis: UTMALDG.4D over a rank-3 map raises ILLEGAL
            # INSTRUCTION from the first thread).
            box=(1,) * (len(c.a.index) - 2) + shape if len(c.a.index) > 2 else None,
            coords=_box_origin(c.a.index, tile=mn[0], tile_base=row_base, k_axis=c.axis, sibling=mn[1]),
            index=_slab_index(c.a.index, tile=mn[0], tile_base=row_base, k_axis=c.axis, tile_is_row=True, sibling=mn[1]),
            swizzle=swizzle,
        )
        return op, True, []

    def a_value(k0, row, col):
        k = BinaryExpr("+", k0, col)
        # σ is hygienic (:func:`subst_free`): a cone statistic re-binding the contraction axis
        # name (attention's k-norm inside the K cone) keeps its own iteration var.
        sigma = Sigma({m_name: m_coord(row), k_name: k_coord(k)})
        stmts: list[Stmt] = [Load(names=(nm,), input=_stat_slab(nm), index=(row,)) for nm in stats]
        stmts += [subst_free(s, sigma) for s in cell]
        return _k_masked(stmts, operand_name(c.a), k, k_ext)

    prologue: list[Stmt] = []
    if stats:
        row_axis = Axis(name="_sr", extent=mn[0].tile)
        sigma = Sigma({m_name: m_coord(Var(row_axis.name))})
        row_body = [subst_free(s, sigma) for s in pro]
        prologue = sync_stat_fill(
            stats=stats,
            slab_of=_stat_slab,
            row_axis=row_axis,
            row_body=row_body,
            cta=cta,
            stat=Reduction.of_cone_stat(c.a),
            dtypes={nm: cuda_name(dt) for nm, dt in cone_stat_dtypes(pro, stats, inputs).items()},
        )
    return SyncOperand(tag="a", shape=(mn[0].tile, bk_elems), value=a_value, swizzle=swizzle), False, prologue


def _sync_operands(
    c: Fold,
    bk_elems: int,
    mn: tuple[Side, Side],
    cta: CtaTile,
    swizzles: tuple[str, str] = ("NONE", "NONE"),
    channels=(),
    seam: tuple = ((), (), ()),
    inputs=None,
) -> tuple[tuple, tuple[SyncOperand, ...], tuple[Operand, ...], list[Stmt]]:
    """The ``smem`` compute fill's drain-ordered, computed, copied, and prologue operands.

    Either contraction role may be a generic inline producer cone. A computed A evaluates at
    absolute ``(m, k)`` and fills the canonical ``tile_m × bk`` slab; a computed B evaluates at
    ``(k, n)`` and fills canonical ``bk × tile_n``. A materialized counterpart rides the same
    vectorized ``cp.async`` copy path used by ordinary staged matmul. Thus a decoded/derived B can
    stream into Tensor Core fragments without ever becoming a dense tensor or a storage-specific
    lowering op.

    A computed A cone with a
    row-invariant prologue (the fused
    norm→linear per-row statistic — its reduce ``Loop`` + scalar sweep) arrives already split at the
    K seam (``views.cone_seam`` reads the cone NODE's boundary; the scheduler sizes the stat rows off
    the same read): the prologue runs ONCE per tile row (:func:`sync_stat_fill`, returned as the transport
    prologue) and the per-cell fill reads the bridged values back from the stat smem rows. The
    schedule's eligibility guarantees exact cover on N; a masked / symbolic **M**
    clamp-reads the overhanging rows in-bounds (the A fill σ and the stat prologue σ — a duplicate
    of the last valid row is computed and its store discarded by the ``RegStore`` guard, the same
    contract the copy transports follow). A symbolic **K** is the same discipline applied to the
    contraction axis: :func:`_k_masked` clamps the cone's own reads and stores the fold identity
    into every slab lane past the runtime extent, so the drain still reads the whole chunk."""
    n_name, k_name = mn[1].axis.name, c.axis.name
    row_base, col_base = _tile_base(mn)
    k_ext = c.axis.extent_expr() if not c.axis.extent.is_static else None

    def m_coord(row) -> Expr:
        t = BinaryExpr("+", row_base, row)
        return clamp_last(t, mn[0].ext) if mn[0].mask else t

    def k_coord(k) -> Expr:
        return clamp_last(k, k_ext) if k_ext is not None else k

    def n_coord(col) -> Expr:
        t = BinaryExpr("+", col_base, col)
        return clamp_last(t, mn[1].ext) if mn[1].mask else t

    a_op, a_copied, prologue = _a_slab_operand(
        c,
        mn=mn,
        bk_elems=bk_elems,
        cta=cta,
        swizzle=swizzles[0],
        seam=seam,
        row_base=row_base,
        m_coord=m_coord,
        k_coord=k_coord,
        k_ext=k_ext,
        inputs=inputs,
    )
    # One B slab per fold channel (the multi-B node fills each projection's weights alongside the
    # one compute-filled A slab); drain order is (A, B0, B1, …) regardless of which fill each rides.
    # ``swizzles`` are the per-operand slab modes (the mma tier's ``slab_swizzles``; NONE elsewhere):
    # every fill kind applies the same flattened-index XOR — the compute fill through the
    # ``Write``'s ``swizzle``, the B cp.async fills through their ``Operand`` — and
    # the ldmatrix drain reads each slab back through its own mode. Unswizzled these slabs drain
    # 4-way (64 B A rows) / 8-way (128 B B rows) bank-conflicted — the measured megakernel residual
    # (294.9 M ld conflicts / 82.5 M LSU inst on the gemma-shape fused edge, 5090).
    channels = channels or ((c.b, c.acc),)
    drain: list = []
    sync_ops: list[SyncOperand] = []
    async_ops: list[Operand] = []

    (async_ops if a_copied else sync_ops).append(a_op)
    drain.append(a_op)

    for f, (bl, _) in enumerate(channels):
        tag = "b" if f == 0 else f"b_x{f}"
        if not isinstance(bl, Load):
            b_body = operand_body(bl)

            def b_value(k0, row, col, *, body=b_body, edge=bl):
                k = BinaryExpr("+", k0, row)
                sigma = Sigma({k_name: k_coord(k), n_name: n_coord(col)})
                return _k_masked([subst_free(s, sigma) for s in body], operand_name(edge), k, k_ext)

            op = SyncOperand(tag=tag, shape=(bk_elems, mn[1].tile), value=b_value, swizzle=swizzles[1])
            sync_ops.append(op)
            drain.append(op)
            continue
        # A transposed B stages N-major (``tile_n × bk`` — its own gmem orientation, K stride-1 in
        # gmem and smem alike), so its cp.async chunks are contiguous exactly like the canonical
        # K-major slab's (row-base alignment holds: B's row stride K is a multiple of ``bk_elems``).
        shape = (mn[1].tile, bk_elems) if c.b_trans else (bk_elems, mn[1].tile)
        op = Operand(
            tag=tag,
            buf=bl.input,
            shape=shape,
            coords=_box_origin(bl.index, tile=mn[1], tile_base=col_base, k_axis=c.axis, sibling=mn[0]),
            index=_slab_index(bl.index, tile=mn[1], tile_base=col_base, k_axis=c.axis, tile_is_row=c.b_trans, sibling=mn[0]),
            swizzle=swizzles[1],
            trans=c.b_trans,
        )
        async_ops.append(op)
        drain.append(op)
    return tuple(drain), tuple(sync_ops), tuple(async_ops), prologue


def _packed_operands(
    c: Fold,
    packed,
    bk_elems: int,
    mn: tuple[Side, Side],
    a_swizzle: str,
    bits_dtype,
    *,
    pad: int,
    cta: CtaTile,
    seam: tuple = ((), (), ()),
    inputs=None,
) -> tuple[tuple, tuple[SyncOperand, ...], tuple[Operand, ...], list[Stmt]]:
    """The staged operands of a PACKED-PAIR B contraction — the NVFP4 weight's byte-slab form.

    Three slabs where the ordinary matmul has two, because the weight arrives as two tensors that
    are cheapest to move apart and combine at the fragment: the packed BITS copy verbatim (one
    byte per two K elements, so the slab is half the K width of a 16-bit one and the copy moves
    half the traffic), and the block SCALES are decoded once per k block into their own small
    f16 slab. The drain reads both and does the decode-and-scale per fragment element; nothing
    ever materializes a decoded weight tile.

    A copied edge takes the index its gmem tensor really has, so BITS is addressed canonically —
    row ``n``, byte column ``k / 2`` over the checkpoint's ``[N, K/2]`` buffer — rather than
    through the decode cone's flattened reshape arithmetic, which says the same thing in a form no
    fill can chunk. The SCALES ride the sync compute-fill because decoding them is compute: the
    checkpoint stores e4m3 codes and one f32 per tensor, and the cone that combines them is
    :attr:`~...tile._packed.PackedKBlockB.factor`, evaluated at the block's own k. That the cone
    may be evaluated at ONE k per block — instead of at every k — is exactly the block-invariance
    the matcher proved.

    ``pad`` is the bits slab's row pad in bytes — ``BYTE_SLAB_PAD`` under cp.async, whose fill
    wants the bank spread, and zero under TMA, whose box deposits dense. The drain reads it back
    off ``Operand.pad_cols``, so the two cannot disagree.

    Returns ``(drain-ordered operands, sync operands, async operands)``. The scale slab is absent
    from the drain order: it is not a fragment source of its own, it is the bits drain's second
    input (``Operand.scale``).
    """
    m, n = mn
    row_base, col_base = _tile_base(mn)
    k_axis, block = c.axis, packed.block
    two = Literal(2, "int")

    def n_coord(row) -> Expr:
        t = BinaryExpr("+", col_base, row)
        return clamp_last(t, n.ext) if n.mask else t

    def m_coord(row) -> Expr:
        t = BinaryExpr("+", row_base, row)
        return clamp_last(t, m.ext) if m.mask else t

    k_ext = k_axis.extent_expr() if not k_axis.extent.is_static else None

    def k_coord(k) -> Expr:
        return clamp_last(k, k_ext) if k_ext is not None else k

    # A rides the SAME side the compute fill gives it: copied when materialized, compute-filled
    # when it is a cone. Only B differs here, and that is the whole point of the packed reading.
    a_op, a_copied, a_prologue = _a_slab_operand(
        c,
        mn=mn,
        bk_elems=bk_elems,
        cta=cta,
        swizzle=a_swizzle,
        seam=seam,
        row_base=row_base,
        m_coord=m_coord,
        k_coord=k_coord,
        k_ext=k_ext,
        inputs=inputs,
    )

    # Just the scale factor's own stmts, not the whole decode cone: the bits copy verbatim, so the
    # compute fill evaluates only what feeds the factor.
    factor_cone = list(Body(tuple(operand_body(c.b))).backward_cone([packed.factor]).members)

    def scale_value(k0, row, col):
        k = BinaryExpr("+", k0, BinaryExpr("*", col, Literal(block, "int")))
        # Hygienic like the compute fill's own substitution above: nothing in a block-scale factor
        # cone re-binds these names today, but the safe spelling costs nothing and the cone is
        # whatever the speller wrote.
        sigma = Sigma({n.axis.name: n_coord(row), k_axis.name: k})
        return [subst_free(s, sigma) for s in factor_cone], packed.factor

    scale_op = SyncOperand(tag="bs", shape=(n.tile, bk_elems // block), value=scale_value)

    # The bits address through the ORIGINAL ``Load``'s own index, σ-evaluated — never a fresh
    # spelling built from the chunk offset. That index carries whatever BASE the contraction axis
    # picked up: a split-K partition shrinks the axis and hangs the slice's absolute base on it
    # (``ksplit·(K/w) + k``), so a hand-built ``k0 / 2`` drops the base and every partition re-reads
    # the FIRST slice's bytes. The block scales never had the bug because they are evaluated by
    # rewriting the decode cone's own body, which carries the same index — this puts the bits on
    # that footing too, which is also what ``_box_origin`` / ``_slab_index`` do for every other
    # staged operand.
    #
    # One column of this slab is one BYTE, so a column step is TWO logical k: the σ substitutes
    # ``k0 + 2·col`` and the index's own ``k / 2`` turns that back into the byte offset.
    def _bits_at(k_expr: Expr, n_expr: Expr) -> tuple:
        sig = Sigma({n.axis.name: n_expr, k_axis.name: k_expr, **_sibling_sigma(m)})
        return tuple(sig.apply(e) for e in packed.bits.index)

    def bits_index(k0):
        def gmem(row, col):
            return _bits_at(BinaryExpr("+", k0, BinaryExpr("*", col, two)), n_coord(row))

        return gmem

    bits_op = Operand(
        tag="b",
        buf=packed.bits.input,
        shape=(n.tile, bk_elems // 2),
        coords=lambda k0: _bits_at(k0, col_base),
        index=bits_index,
        trans=True,
        pad_cols=pad,
        dtype=cuda_name(bits_dtype),
        elem_bytes=bits_dtype.nbytes,
        scale=(scale_op.slab, block),
    )
    # The scale slab is always compute-filled; A joins it there when it is a cone.
    filled = (scale_op,) if a_copied else (a_op, scale_op)
    copied = (a_op, bits_op) if a_copied else (bits_op,)
    return (a_op, bits_op), filled, copied, a_prologue


def _block_scaled_operands(
    c: Fold, pair, bk_elems: int, mn: tuple[Side, Side], bits_dtype, scale_dtype, *, pad: int
) -> tuple[tuple, tuple[Operand, ...], tuple[SyncOperand, ...]]:
    """The staged operands of a BLOCK-SCALED packed pair — the native fp4 cell's ``2 + 2N`` slabs.

    The shared A contributes its codes and its raw block scales; each of the N product channels
    adds its own two, so a plain matmul stages four and a fused gate⊗up MLP edge stages six. Where
    the packed byte-slab stage next door has two-and-a-fill, these are verbatim
    copies: both operands' codes and both operands' raw e4m3 block scales are stored bytes. The
    instruction applies the scales itself, so the fill that stage needs to evaluate a fused scale
    has nothing left to compute, and the drain never materializes a decoded value on either side.

    Every slab addresses through its own ``Load``'s index, σ-evaluated — never a fresh spelling
    built from the chunk offset. That index carries whatever BASE the contraction axis picked up:
    a split-K partition shrinks the axis and hangs the slice's absolute base on it, so a
    hand-built ``k0 / 2`` would drop the base and every partition would re-read the first slice's
    bytes (the defect that cost this branch a week on the W4A16 side).

    A codes column is one BYTE — two logical k — so its column step is 2 and the index's own
    ``k / 2`` turns that back into the byte offset. A scales column is one BLOCK, so its step is
    the block extent. Returns ``(drain-ordered operands, every copied operand)``; the two scale
    slabs are absent from the drain order because they are not fragment sources of their own —
    they reach the drain as each data operand's ``scale``.
    """
    m, n = mn
    row_base, col_base = _tile_base(mn)
    k_axis, block = c.axis, pair.block
    k_ext = k_axis.extent_expr() if not k_axis.extent.is_static else None

    def filled(side: Side, sibling: Side, base, codes, cone, tag: str, *, scale):
        """A compute-FILLED codes slab: this matmul's own kernel produces the byte, so there is no
        buffer to copy and the fill evaluates the cone at each slab cell instead. One cell is one
        byte — two logical k — and the cone derives both nibbles from a single k coordinate, so
        the σ substitutes the same ``k0 + 2·col`` the copied form's index arithmetic consumes."""

        def value(k0, row, col):
            k = BinaryExpr("+", k0, BinaryExpr("*", col, Literal(2, "int")))
            t = BinaryExpr("+", base, row)
            sigma = Sigma({side.axis.name: clamp_last(t, side.ext) if side.mask else t, k_axis.name: k, **_sibling_sigma(sibling)})
            return [st.rewrite(lambda nm: nm, sigma) for st in cone], codes

        return SyncOperand(tag=tag, shape=(side.tile, bk_elems // 2), value=value, scale=scale)

    def build(side: Side, sibling: Side, base, load, tag: str, *, cols: int, step: int, dtype, trans: bool, scale=None, pad=pad):
        def at(k_expr, row_expr):
            sig = Sigma({side.axis.name: row_expr, k_axis.name: k_expr, **_sibling_sigma(sibling)})
            return tuple(sig.apply(e) for e in load.index)

        def coord(row):
            t = BinaryExpr("+", base, row)
            return clamp_last(t, side.ext) if side.mask else t

        def index(k0):
            def gmem(row, col):
                k = BinaryExpr("+", k0, BinaryExpr("*", col, Literal(step, "int")))
                return at(clamp_last(k, k_ext) if k_ext is not None else k, coord(row))

            return gmem

        return Operand(
            tag=tag,
            buf=load.input,
            shape=(side.tile, cols),
            coords=lambda k0: at(k0, base),
            index=index,
            trans=trans,
            pad_cols=pad,
            dtype=cuda_name(dtype),
            elem_bytes=dtype.nbytes,
            scale=scale,
        )

    # The scale slabs are built first: each data operand names its own as the drain's second
    # source, exactly as the packed byte-slab drain names its compute-filled one.
    a_scale = build(m, n, row_base, pair.a.scale, "as", cols=bk_elems // block, step=block, dtype=scale_dtype, trans=False, pad=0)
    a_bits = (
        filled(m, n, row_base, pair.a.codes, pair.a.cone, "a", scale=(a_scale.slab, block))
        if pair.a.bits is None
        else build(m, n, row_base, pair.a.bits, "a", cols=bk_elems // 2, step=2, dtype=bits_dtype, trans=False, scale=(a_scale.slab, block))
    )
    # One codes + one scales slab per channel, over the shared A pair. Channel 0 keeps the bare
    # ``b`` / ``bs`` tags so a single-channel cell stages byte-identical slabs to before.
    b_scales, b_bits = [], []
    for i, side in enumerate(pair.b):
        tag = "b" if i == 0 else f"b{i}"
        scale = build(n, m, col_base, side.scale, f"{tag}s", cols=bk_elems // block, step=block, dtype=scale_dtype, trans=True, pad=0)
        b_scales.append(scale)
        b_bits.append(
            build(n, m, col_base, side.bits, tag, cols=bk_elems // 2, step=2, dtype=bits_dtype, trans=True, scale=(scale.slab, block))
        )
    copies = tuple(op for op in (a_bits, *b_bits, a_scale, *b_scales) if isinstance(op, Operand))
    fills = tuple(op for op in (a_bits,) if isinstance(op, SyncOperand))
    return (a_bits, *b_bits), copies, fills


def _staged(ops: _AtomOps, cells, offset, mn: tuple[Side, Side]):
    """The **one** STAGED K-loop driver, atom-agnostic — build the ``(A, B)`` operand pair, the
    :class:`Transport` (a cp.async prefetch ring or the TMA box-copy producer) and run the one
    :func:`staged_kloop`; the atom supplies only the slab drain leaf (:meth:`_AtomOps.staged_drain`
    — fragment load + ``mma.sync`` vs plain-``Load`` fma) and the slab element dtype. ``ops.stage``
    is the **scheduler-RESOLVED** stage (the scheduler
    ran eligibility + sizing once) — its ``transport`` / ``bk_elems`` / ``depth`` / ``reg_depth``
    are applied verbatim, no decision here. A pure perf transform, numerically identical to
    gmem-direct (mma: bit-identical). ``depth == 1`` is the single-buffer degenerate; ``depth >= 2``
    the gmem→smem ring; ``reg_depth`` composes the mma inner smem→register double-buffer. A masked
    M / N overhang is fill-side clamped (cp.async) or box zero-filled (TMA); the discard stays with
    the store guard."""
    c, stage, tile = ops.c, ops.stage, ops.tile
    k_axis = c.axis
    # The chunk stream. A static K unrolls a literal chunk count; a SYMBOLIC one hands the skeleton
    # its ``Dim`` and the runtime ``ceil(K / bk)`` — the fill masks the last chunk's tail to the
    # fold identity (:func:`_k_masked`), so the drain still reads whole chunks.
    bk, static_k = stage.bk_elems, k_axis.extent.is_static
    K = k_axis.extent.as_static() if static_k else k_axis.extent
    n_chunks = K // bk if static_k else Dim(BinaryExpr("/", BinaryExpr("+", K.expr, Literal(bk - 1, "int")), Literal(bk, "int")))
    elem = ops.slab_elem()
    cta = _cta(mn, tile.atom.lanes, tile.launch_threads)
    finalize: list[Stmt] = []
    # The BLOCK-SCALED pair, keyed on the ATOM: both operands packed under a 16-bit atom is still
    # the single-sided shape, whose drain decodes each into 16-bit fragments. cp.async only; the
    # four-descriptor TMA box copy is not built.
    native = stage.transport == "smem-async" and block_scaled_atom(tile.atom)
    bs_pair = match_packed_pair_node(c, ops.inputs) if native else None
    copy_transport = stage.transport in ("smem-async", "smem-tma")
    packed = None if bs_pair is not None else (match_packed_b_node(c, ops.inputs) if copy_transport else None)
    if bs_pair is not None:
        operands, copies, fills = _block_scaled_operands(
            c,
            bs_pair,
            stage.bk_elems,
            mn,
            ops.inputs[bs_pair.b[0].bits.input].dtype,
            ops.inputs[bs_pair.a.scale.input].dtype,
            pad=BYTE_SLAB_PAD,
        )
        common = dict(slab_dtype=cuda_name(elem), elem_bytes=elem.nbytes, cta=cta)
        # Pure copies when both operands' codes are stored; a fill underneath them when this
        # matmul computes its own A codes, which is the same two-group shape the packed
        # byte-slab stage takes for its scale fill.
        transport = (
            CpAsyncTransport(operands=copies, **common) if not fills else SyncTransport(operands=fills, copy_operands=copies, **common)
        )
        # The per-tensor scale levels, applied once per output element after the K-loop. The cell
        # multiplies the RAW e4m3 block scales into each block's sum and knows nothing of the
        # second level, so a factor that the operand's own chain applies per element has to land
        # here instead. It is synthesized rather than read off the term: the term spells that
        # factor INSIDE a rounding to the fragment dtype, which nothing can hoist out of, so no
        # epilogue at the tile layer names it. The emitted cell is therefore not a
        # rounding-for-rounding account of the declared program, and its parity is a tolerance
        # rather than the exact oracle.
        # One residue chain PER CHANNEL: the A-side factors are shared, each channel's weight-side
        # factors are its own. Folding them all into a single product would apply one channel's
        # per-tensor scale to the other, which is wrong by a scalar and invisible in the emitted
        # source — only numerical parity catches it.
        alpha_stmts: list[Stmt] = []
        alpha_of: list[str] = []
        counter = 0
        for side in bs_pair.b:
            alpha = ""
            for ld in (*bs_pair.a.alpha, *side.alpha):
                leaf = f"_bs_alpha_l{counter}"
                alpha_stmts.append(Load(name=leaf, input=ld.input, index=ld.index, dtype=ld.dtype))
                step = f"_bs_alpha{counter}"
                alpha_stmts.append(
                    Assign(name=step, op="copy", args=(leaf,)) if not alpha else Assign(name=step, op="multiply", args=(alpha, leaf))
                )
                alpha = step
                counter += 1
            alpha_of.append(alpha)
        finalize = [
            *alpha_stmts,
            *(
                FragmentApply(out=cf, op=ElementwiseImpl("multiply"), args=(cf, alpha_of[f]), kinds=(FRAG, UNIFORM), in_place=True)
                for f in range(len(ops.channels))
                for i in range(mn[0].reg)
                for j in range(mn[1].reg)
                for cf in (_fold_frag(ops.frag(f"_c{i}_{j}"), f),)
            ),
        ]
    elif packed is not None:
        # The packed-pair (NVFP4) weight: the bits copy beside A, the block scales decode into
        # their own slab, and the drain combines them at the fragment (:func:`_packed_operands`).
        tma = stage.transport == "smem-tma"
        operands, sync_ops, async_ops, packed_pro = _packed_operands(
            c,
            packed,
            stage.bk_elems,
            mn,
            ops.slab_swizzles(mn, elem.nbytes)[0],
            ops.inputs[packed.bits.input].dtype,
            pad=0 if tma else BYTE_SLAB_PAD,
            cta=cta,
            seam=ops.cone,
            inputs=ops.inputs,
        )
        common = dict(slab_dtype=cuda_name(elem), elem_bytes=elem.nbytes, cta=cta)
        if tma:
            # TWO operand groups, not one. cp.async can ride inside the ``sync`` producer because
            # both are issued by the same threads under one CTA barrier; a TMA copy is armed on an
            # mbarrier by one elected thread and waited on by parity, which no compute fill can be
            # folded into. The K-loop skeleton already schedules a LIST of groups, so the copies
            # and the scale fill are simply two of them over one drain segment: the box copies ring
            # at the stage's depth, the compute fill stays single-buffer as it always is.
            copies = TmaTransport(operands=async_ops, **common)
            fill = SyncTransport(operands=sync_ops, prologue_stmts=tuple(packed_pro), **common)
            slabs = frozenset(op.slab for op in (*async_ops, *sync_ops))
            return pipelined_kloop(
                operands=((copies, stage.depth), (fill, 1)),
                build_segments=lambda slots: [(ops.staged_drain(operands, slots[0], cells, offset, mn), slabs)],
                bk_elems=stage.bk_elems,
                n_chunks=K // stage.bk_elems,
                k_extent=K,
            )
        # cp.async: one ``sync`` producer whose copied peers are the two copied slabs — the
        # same shape the fused norm→linear edge takes.
        transport = SyncTransport(operands=sync_ops, copy_operands=async_ops, prologue_stmts=tuple(packed_pro), **common)
    elif stage.transport == "smem":
        # The synchronous fill: every inline edge is evaluated into its canonical slab (converting
        # on the store when dtypes differ); every materialized counterpart is COPIED underneath
        # that work — with ``cp.async``, or with the blocking vector copy on an atom whose target
        # has none. A term with no inline edge at all lands here too: then it is only the copy.
        operands, sync_ops, copy_ops, stat_pro = _sync_operands(
            c, stage.bk_elems, mn, cta, ops.slab_swizzles(mn, elem.nbytes), ops.channels, ops.cone, ops.inputs
        )
        transport = SyncTransport(
            operands=sync_ops,
            copy_operands=copy_ops,
            slab_dtype=cuda_name(elem),
            elem_bytes=elem.nbytes,
            cta=cta,
            prologue_stmts=tuple(stat_pro),
            copy_sync=tile.atom.sync_copy_staging,
        )
    else:
        assert len(ops.channels) == 1, "cp.async / TMA staging is single-fold — a multi-B node rides the smem compute fill"
        # A cp.async-staged 1-byte (fp8) slab pads its rows (`BYTE_SLAB_PAD`) so the cooperative
        # byte-gather drain spreads across banks; a TMA box deposit is dense, so its byte slab
        # stays unpadded (the resolver sized the budget with the same rule).
        elems = ops.slab_elems()
        pads = tuple(BYTE_SLAB_PAD if e.nbytes == 1 and stage.transport == "smem-async" else 0 for e in elems)
        operands = _slab_operands(
            index_srcs=(c.a.index, c.b.index),
            bufs=(c.a.input, c.b.input),
            mn=mn,
            k_axis=k_axis,
            bk_elems=stage.bk_elems,
            base=_tile_base(mn),
            swizzles=ops.slab_swizzles(mn, elem.nbytes),
            elems=elems,
            b_trans=c.b_trans,
            pads=pads,
        )
        common = dict(
            operands=operands,
            slab_dtype=cuda_name(elem),
            elem_bytes=elem.nbytes,
            cta=cta,
        )
        transport = TmaTransport(**common) if stage.transport == "smem-tma" else CpAsyncTransport(**common)

    def drain(slot):  # the atom's slab-reading leaf, over ring `slot`
        return ops.staged_drain(operands, slot, cells, offset, mn)

    pre, region = staged_kloop(
        transport=transport,
        drain=drain,
        depth=stage.depth,
        bk_elems=bk,
        n_chunks=n_chunks,
        k_extent=K,
        workers=ops.workers,
        block_threads=tile.launch_threads,
    )
    # The block-scaled cell's per-tensor scale levels land on the output fragments here — after
    # the K-loop, before the sink.
    return pre, [*region, *finalize]


def _contract_kloop(c, cells, *, read_row, read_col, contract, wrap):
    """The shared contraction K-loop skeleton — the ``read → ⊗ → fold`` spine both atoms lower through.

    Read each register ROW's A operand once and each register COL's B operand once (register-tile
    operand reuse — an A read is shared across the row's columns and vice versa), contract every
    ``(row, col)`` pair into its accumulator, then wrap the whole body in the reduce loop. The only
    per-atom variation is the four leaf constructors (the "atom factory"): ``read_row`` / ``read_col``
    build the operand read (``LdmatrixLoad`` fragment vs scalar ``Load``), ``contract`` the ⊗+accumulate
    (``MmaSyncPtx`` vs ``Assign``+``Accum``), ``wrap`` the K-loop (``StridedLoop`` step ``atom_k`` vs a
    unit ``Loop``). Returns ``(pre_decls, kloop_stmts)`` — no pre-decls here (accumulators ride
    ``state``)."""
    rows = sorted({i for i, _ in cells})
    cols = sorted({j for _, j in cells})
    body: list[Stmt] = []
    for i in rows:
        body += read_row(i)
    for j in cols:
        body += read_col(j)
    for i, j in cells:
        body += contract(i, j)
    return [], wrap(body)


# ---- scalar (register-tile) tier --------------------------------------------------------------- #
def _unroll_inner(axis) -> bool:
    """Mark the inner contraction loop for ``#pragma unroll`` when it's a small static reduce
    (≤ 64 trips, or the ``EMMY_UNROLL`` budget) — register-resident operand reuse + ILP, the
    scalar-SGEMM lever."""
    return unroll_ok(axis.extent, 64)


def _cell_varying(body, side: Side | None) -> bool:
    """Whether an operand's producing ``body`` varies along ``side`` — the OTHER output axis of the
    register tile (n for A, m for B).

    A gmem ``Load`` edge indexes its own axes only, so this is False and the tile's reuse holds: one
    A read per register row, one B read per column. A COMPUTED edge is free to read the sibling
    coordinate — the o_proj shape whose A cone is broadcast over n (``out[m, n] = Σ_k B[n, k] ·
    A[m, k, n]``) — and then the row read is BOTH the wrong value for every column but the first and
    a reference to a coordinate the kernel does not define: after the tile split only the ``_b`` /
    ``_u`` split vars are bound, so the per-copy rename (:func:`copy_cell`) suffixes the surviving
    axis name into an undefined identifier (nvcc: ``identifier "a1__ar9" is undefined``, the
    whole-model qwen3-0.6b layer-0 o_proj on sm_89). Such an operand is read per CELL instead."""
    return side is not None and Body(body).depends_on(body, side.name)


def _dedup_loads(stmts: list[Stmt]) -> list[Stmt]:
    """Collapse syntactically-identical scalar ``Load``s (same buffer + index) to one binding,
    rewriting the dropped names to the survivor — the operand reuse a register tile exists for (a
    load not referencing the ``m`` cell axis is shared across the ``n`` cells, and vice versa).

    Dedup itself is per-``stmts`` (this list only); the rewrite reaches nested scopes, since an
    epilogue cell's output sweep consumes the loads hoisted above it. It goes through
    :func:`~emmy.compiler.ir.stmt.passes.rename_free` so a nested scope re-binding a dropped
    name keeps its own binding."""
    seen: dict = {}
    rename: dict[str, str] = {}
    kept: list[Stmt] = []
    for s in stmts:
        if isinstance(s, Load) and s.is_scalar:
            sig = (s.input, tuple(e.pretty() for e in s.index))
            if sig in seen:
                rename[s.names[0]] = seen[sig]
                continue
            seen[sig] = s.names[0]
        kept.append(s)
    if rename:
        kept = [rename_free(s, rename) for s in kept]
    return kept


def _guard_writes(stmts: list[Stmt], cond) -> list[Stmt]:
    """Wrap each output ``Write`` in ``Cond(cond, …)`` — the masked tail cell computes (with
    clamp-read operands) but only stores when in bounds. Recurses into nested bodies: a projection
    tail carrying an output-sweep ``Loop`` (e.g. a decode epilogue) holds its ``Write`` inside that
    loop, and an unguarded overhang store would alias a valid row through the σ ``% extent`` wrap.
    Other stmts pass through."""
    if cond is None:
        return stmts
    out: list[Stmt] = []
    for s in stmts:
        if isinstance(s, Write):
            out.append(Cond(cond=cond, body=(s,)))
            continue
        bodies = s.nested()
        if bodies:
            s = s.with_bodies(tuple(Body(tuple(_guard_writes(list(b), cond))) for b in bodies))
        out.append(s)
    return out


def _scalar_sigma(mn, offset, i: int, j: int) -> Sigma:
    """σ mapping each present output axis to register cell ``(i, j)``'s real coordinate (the offset's
    block·tile + unit·reg + r), a **masked** axis wrapped in-bounds (``% extent``)."""
    return Sigma({side.name: _wrap(side, cell) for side, cell in _cells(mn, offset, i, j)})


def _scalar_bound(mn, offset, i: int, j: int):
    """The in-bounds predicate for cell ``(i, j)`` — ``base < extent`` anded over the masked axes,
    or ``None`` when nothing overhangs."""
    conds = [BinaryExpr("<", cell, side.ext) for side, cell in _cells(mn, offset, i, j) if side.mask]
    if not conds:
        return None
    cond = conds[0]
    for c in conds[1:]:
        cond = BinaryExpr("&&", cond, c)
    return cond


def _scalar_protected(c: Fold, tile: Tile, lead: tuple = (), *, body: Body | tuple = ()) -> frozenset[str]:
    """The shared iteration coordinates — the block / unit / loop / extent vars excluded from the
    per-cell SSA rename (everything else is suffixed ``__c{i}_{j}`` so each cell owns its names).
    ``lead`` is the kernel's leading (batch / ksplit) grid axes: one coordinate for the whole cell
    block, so renaming one would emit a reference no enclosing loop defines. ``body`` contributes
    projection-local loop coordinates, notably an output sweep that remains bound inside every
    replicated cell."""
    m, n, k_axis = tile.m, tile.n, c.axis
    prot = {k_axis.name}
    for s in (m, n):
        prot |= {s.block, s.unit}
    for a in lead:
        prot.add(a.name)
    for a in (m.axis, n.axis, k_axis, *lead):
        prot |= set(a.extent_expr().free_vars())
    prot.update(Body(body).axis_names)
    return frozenset(prot)


def _scalar_drain(
    c: Fold, cells, offset, slabs: tuple[str, str], ki: str, bk_elems: int, base: tuple[Expr, Expr], offs=(None, None)
) -> Loop:
    """The inner slab-drain reduce loop ``for ki: b = b_slab[ki, n_local]; a = a_slab[m_local, ki];
    v = a·b; acc += v`` — the scalar counterpart of the mma ``ldmatrix`` drain. Built per-cell directly
    (NOT via the masked gmem-direct σ, whose ``% extent`` wrap would corrupt the slab index for an
    overhanging cell): the slab is indexed by the **local** tile coordinate ``offset[{0,1}].base(...) −
    base`` (``m_uvar·reg_m + i`` ∈ [0, tile_m), always in-slab), so an overhanging cell reads a
    clamped / zero-filled slab row and its store is discarded by the guard. ``_dedup_loads`` still
    shares A across the n-cells and B across the m-cells exactly as gmem-direct does. **Seed-less**
    (the accumulators are pre-seeded once by :meth:`_ScalarOps.state` outside the
    outer slab loop, so the drain folds into them without re-seeding. ``offs`` (the gmem→smem ring,
    depth > 1) is the ``(a, b)`` read SLOT row offset pair, added to each slab's ROW — the same slot
    seam the mma drain rides."""
    (a_slab, b_slab), (row_base, col_base) = slabs, base
    off_a, off_b = offs
    b_name, a_name = operand_name(c.b), operand_name(c.a)
    body: list[Stmt] = []
    for i, j in cells:
        sfx = f"__c{i}_{j}"
        bn, an, vn, cn = f"{b_name}{sfx}", f"{a_name}{sfx}", f"{c.acc}__v{sfx}", f"{c.acc}{sfx}"
        m_local = BinaryExpr("-", offset[0].base(i), row_base)
        n_local = BinaryExpr("-", offset[1].base(j), col_base)
        k_row = Var(ki) if off_b is None else BinaryExpr("+", off_b, Var(ki))
        m_row = m_local if off_a is None else BinaryExpr("+", off_a, m_local)
        body.append(Load(names=(bn,), input=b_slab, index=(k_row, n_local)))
        body.append(Load(names=(an,), input=a_slab, index=(m_row, Var(ki))))
        body.append(Assign(name=vn, op=_MUL, args=(bn, an)))
        body.append(Accum(name=cn, value=vn, op=_ADD, axes=(ki,)))
    body = _dedup_loads(body)
    # seed=False: the accumulators are pre-seeded once by _ScalarOps.state outside the outer slab loop, so
    # this inner drain must NOT re-declare (re-zero) them each slab iteration.
    return Loop(axis=Axis(name=ki, extent=bk_elems), body=Body(tuple(body)), unroll=unroll_ok_n(bk_elems, 64), seed=False)


@dataclass(frozen=True)
class _AtomOps:
    """The per-atom codegen **strategy** — the one seam every tiled contraction dispatches through.
    Bound to the contraction ``c`` + its **scheduler-resolved** operand ``stage`` (``None`` =
    gmem-direct; eligibility + sizing already ran schedule-side) and ``inputs``, it supplies
    the three ``grid_tile`` callables — ``state(cells)`` (accumulator decls), :meth:`reduce` (the
    K-loop — **shared on this base**, one loop over atoms), ``store(i, j, offset, mn)`` (the
    per-cell sink; ``mn`` is the contraction's ``(m, n)`` :class:`Side` pair). The two concrete
    atoms (:class:`_MmaOps` / :class:`_ScalarOps`) supply only descriptor reads:
    :meth:`gmem_leaves` (the four gmem-direct leaf constructors), :meth:`staged_drain` (the
    slab-reading leaf) and :meth:`slab_elem` (the slab element dtype). This IS the "atom as
    descriptor" seam: one factory (:func:`_atom_ops`), no scattered ``isinstance``."""

    c: Fold  # the ALGEBRA — operand edges, K axis, channels
    tile: Tile  # the SCHEDULE slice, PLACED (``Tile.at``): the atom + the ``(m, n)`` geometry
    stage: Stage | None = None
    inputs: object = None
    workers: object = None  # the resolved WarpSpec (None = uniform SIMT) — consumed by _staged
    # The kernel's LEADING (batch / ksplit) grid axes — the coordinates shared by every register
    # cell, so the per-cell rename must pass them through (:func:`_scalar_protected`). They come
    # from the caller that owns the grid (``_factor._bind``), not off the ``tile`` slice: the slice
    # reads the tiled ``(m, n)`` cell, and what sits outside it is the grid's fact.
    lead: tuple = ()
    # The projection this node's store folds in — the wrapping zero-axis fold's lift body plus the grid-``Write``
    # glue, assembled by ``_factor._bind``. It is NOT a node field: every projection has ONE home,
    # the zero-axis ``Fold`` wrapper, and the store sink is where it lands.
    epilogue: Body = field(default_factory=Body)
    # The computed-A cone's ``(prologue, cell, stats)`` K seam, read off the NODE BOUNDARY
    # (``views.cone_seam``). ``None`` for a
    # plain gmem-``Load`` A — its whole body is the per-cell fill.
    seam: tuple | None = None
    # The register-fragment NAMESPACE. Empty for the kernel's own contraction; a nested producer
    # emitted inside the enclosing K-loop prefixes its fragments so its
    # ``_a`` / ``_b`` / ``_c`` do not shadow the accumulators the enclosing drain carries across
    # that same loop.
    frag_ns: str = ""
    # Per-operand ``(slab, ldm, swizzle)`` when the caller's own transport already STAGED it — the
    # nested contraction's invariant and streamed slabs, filled by the enclosing loop's pipeline
    # instead of re-read per warp through the gmem fragment loaders. Both keep the operand's own
    # gmem orientation (so ``ldm`` is the child's K extent either way), one chunk: the read is the ordinary staged
    # ``LdmatrixLoad``, only its slab comes from outside. ``swizzle`` is the mode the fill wrote the
    # slab with (:func:`~._stage.software_swizzle` on the child's K span) — the drain applies the matching
    # XOR, exactly like the enclosing contraction's own operands. ``None`` = gmem-direct.
    slabs: tuple[tuple[str, int, str] | None, tuple[str, int, str] | None] = (None, None)

    def frag(self, name: str) -> str:
        """``name`` in this emission's fragment namespace (:attr:`frag_ns`)."""
        return f"{self.frag_ns}{name}"

    @property
    def channels(self) -> tuple:
        """The ``(b, acc)`` pairs this emission folds — the node's product channels (one A
        fragment, N mma chains, one C fragment per channel at arity N)."""
        return tuple((ch.b, ch.acc) for ch in self.c.channels)

    @property
    def cone(self) -> tuple:
        """The A cone's ``(row-invariant prologue, per-cell body, bridged stats)`` — the node
        boundary, or the whole operand body when there is no cone to split."""
        return self.seam if self.seam is not None else ((), operand_body(self.c.a), ())

    def reduce(self, cells, offset, mn):
        """The contraction K-loop — the ONE driver both atoms flow through, deciding nothing: a
        resolved ``stage`` means staged (an smem operand slab over the one :func:`_staged`
        fill→drain skeleton), ``None`` means gmem-direct (the shared :func:`_contract_kloop`
        ``read → ⊗ → fold`` spine). Either way the atom contributes only leaves, never a loop."""
        if self.stage is not None:
            return _staged(self, cells, offset, mn)
        return _contract_kloop(self.c, cells, **self.gmem_leaves(offset, mn))

    def slab_swizzles(self, mn, elem_bytes: int) -> tuple[str, str]:  # noqa: ARG002
        """The per-operand TMA smem swizzle modes for the ``(A, B)`` slabs — ``NONE`` on this
        base: only a drain that applies the matching address XOR may read a swizzled slab, and
        the scalar tier's plain-``Load`` drain doesn't (:class:`_MmaOps` overrides)."""
        return ("NONE", "NONE")

    def slab_elems(self) -> tuple:
        """The per-operand ``(A, B)`` slab element dtypes. The mma tier's operands share the atom
        dtype (16-bit fragments, dtype-gated at enumeration), so this base returns the one
        :meth:`slab_elem` twice; :class:`_ScalarOps` overrides with each gmem operand's OWN dtype —
        a mixed fp32-A × fp16-B contraction (the norm→linear split-combine shape) must size each
        slab and its ``cp.async`` fill by its own element width, or the B fill issues 16 B chunks
        at fp32 spacing over fp16 memory (misaligned + overlapped — the Gemma bench_fail cluster)."""
        elem = self.slab_elem()
        return (elem, elem)


class _MmaOps(_AtomOps):
    """Tensor-core atom — layout-aware fragment reads + ``mma.sync``, a ``RegStore`` sink."""

    def slab_elem(self):
        """The slab element dtype — the mma A/B operand dtype (f16/bf16 fragments; the fp8
        atoms' byte operands)."""
        return self.tile.atom.operand_dtype("a")

    def slab_elems(self) -> tuple:
        """Per-operand slab element dtypes — the atom's operand dtype, except a 1-byte
        (fp8-stored) operand, whose slab keeps the STORAGE dtype: the raw bytes stage verbatim
        and the drain converts (W8A16) or repacks (the k32 atoms). Mirrors the resolver's dtype
        legality (``schedule.staging.resolve_warp_stage``), so a mismatch that staged would already
        have declined there."""
        out = []
        for edge, role in ((self.c.a, "a"), (self.c.b, "b")):
            dt = self.tile.atom.operand_dtype(role)
            t = self.inputs.get(edge.input) if self.inputs and isinstance(edge, Load) else None
            out.append(t.dtype if t is not None and t.dtype.nbytes == 1 else dt)
        return tuple(out)

    def staged_drain(self, operands, slot, cells, offset, mn):
        """The mma slab drain — the fragment-load + ``mma.sync`` leaf reading ring ``slot``
        (:func:`_staged_inner_atom_loop`; the cells ride ``mn``'s reg counts, so ``cells`` /
        ``offset`` are unused here). Each slab's swizzle mode rides its operand (``NONE`` on the
        sync transport's :class:`SyncOperand`, which has no swizzle field); a 1-byte operand slab
        (``Operand.elem_bytes == 1`` — staged fp8) drains through the cooperative byte gather
        instead of ldmatrix, its row pad riding the drain ``ldm``; a PACKED-PAIR operand
        (``Operand.scale``) additionally hands the drain its block-scale slab. An f16-accumulate
        atom promote-folds its packed f16 fragments into the f32 shadows once per drain — the
        bk chunk IS the promote cadence (the last chunk's fold doubles as the final one)."""
        stmts = _staged_inner_atom_loop(
            slabs=tuple(op.slab for op in operands),
            offs=tuple(op.slot_row(slot) for op in operands),
            mn=mn,
            atom=self.tile.atom,
            bk_elems=self.stage.bk_elems,
            ki="_ki",
            reg_depth=self.stage.reg_depth,
            swizzles=tuple(getattr(op, "swizzle", "NONE") for op in operands),
            trans=tuple(getattr(op, "trans", False) for op in operands),
            byte_slabs=tuple((getattr(op, "elem_bytes", None) or self.tile.atom.operand_dtype("a").nbytes) == 1 for op in operands),
            pads=tuple(getattr(op, "pad_cols", 0) for op in operands),
            frag_ns=self.frag_ns,
            scales=tuple(self._drain_scale(op) for op in operands),
        )
        if _f16acc(self.tile.atom):
            stmts = [*stmts, *_f16acc_promotes(mn[0].reg, mn[1].reg, len(self.channels), self.frag_ns)]
        return stmts

    def _drain_scale(self, op):
        """The drain's ``(scale slab, its row stride, the k block)`` for a PACKED-PAIR operand, or
        ``None``. The stride is the chunk's block count — the scale slab is ``tile × bk/block``."""
        scale = getattr(op, "scale", None)
        return None if scale is None else (scale[0], self.stage.bk_elems // scale[1], scale[1])

    def slab_swizzles(self, mn, elem_bytes: int) -> tuple[str, str]:  # noqa: ARG002 — per-operand widths come from slab_elems
        """The smem swizzle mode per operand slab, from each slab's inner (contiguous) row
        span — A's is ``bk_elems`` (the K chunk), B's ``tile_n``. TMA applies the mode in
        hardware (in-copy); the cp.async transport applies the identical XOR in software on
        each fill's destination index — both drains read back through the same ldmatrix XOR.
        The pre-rebuild bar for the fp16 squares (``square.2048.fp16`` at 106.7 µs / 1.06×
        cuBLAS) was set by swizzled slabs; the rebuilt NONE-swizzle transport left the
        ``ldmatrix`` drain bank-conflict-bound — and the never-swizzled cp path the same way
        (4-way on 64 B rows / 8-way on 128 B; conflict replays were 81% of the sm_89 gate_up fm
        golden's shared-mem wavefronts, the measured residual to cuBLAS). Modes are
        DERIVED, not tuned — the widest atom that divides the span (matching the pre-rebuild
        behavior: A → B64 on a 32-elem fp16 chunk, B → B128 on a 64-elem row). A transposed B
        stages N-major on EVERY transport (cp.async / TMA / the sync transport's async B fills),
        so its inner row span is the K chunk (``bk_elems``) like A's. A 1-byte (fp8) slab stays
        ``NONE`` — its cooperative byte-gather drain applies no address XOR (the ldmatrix XOR is
        b16-indexed); the cp.async byte slab's bank spread is the row pad instead."""
        if self.tile.atom.fragment_layout == "m8n8k4":
            return ("NONE", "NONE")
        b_inner = self.stage.bk_elems if self.c.b_trans else mn[1].tile
        # A TMA slab keeps the hardware spelling (the copy engine fixes its permutation, and the
        # descriptor splits its box down to the atom); every other transport writes the slab in
        # software, so its XOR reads the row index at the slab's OWN stride.
        hardware = self.stage.transport == "smem-tma"

        def mode(inner: int, nbytes: int) -> str:
            if nbytes == 1:
                return "NONE"
            return pick_swizzle_atom(inner, nbytes)[1] if hardware else software_swizzle(inner, nbytes)

        return tuple(mode(inner, e.nbytes) for e, inner in zip(self.slab_elems(), (self.stage.bk_elems, b_inner), strict=True))

    def state(self, cells):
        """The mma operand/accumulator register fragments — one ``_a``/``_b`` per register row/col and
        one ``_c`` accumulator per cell (held across the K-loop). A staged ``reg_depth >= 2`` slots the
        operand fragments (``_a{i}_s{slot}``) for the smem→register double-buffer's ping-pong."""
        atom, m, n = self.tile.atom, self.tile.m, self.tile.n
        reg_depth = self.stage.reg_depth if self.stage is not None else 1

        def frags(base_of, reg):  # reg-tile operand fragment names (slotted ``_s{s}`` when double-buffered)
            names = [base_of(i) for i in range(reg)]
            return [f"{nm}_s{s}" for nm in names for s in range(reg_depth)] if reg_depth >= 2 else names

        # One A fragment set; one B and one C fragment set PER fold channel (the multi-B node's
        # shared-A / per-channel-accumulate drain).
        n_folds = len(self.channels)
        decls: list[Stmt] = [
            RegFragment(
                name=nm,
                role="a",
                shape=atom.ptx_shape,
                dtype=atom.operand_dtype("a"),
                nregs=atom.fragment_nregs("a"),
            )
            for nm in frags(lambda i: self.frag(f"_a{i}"), m.reg)
        ]
        for f in range(n_folds):
            decls += [
                RegFragment(
                    name=nm,
                    role="b",
                    shape=atom.ptx_shape,
                    dtype=atom.operand_dtype("b"),
                    nregs=atom.fragment_nregs("b"),
                )
                for nm in frags(lambda i, ff=f: _fold_frag(self.frag(f"_b{i}"), ff), n.reg)
            ]
        decls += [
            RegFragment(
                name=_fold_frag(self.frag(_mma_c_base(atom, i, j)), f),
                role="c",
                shape=atom.ptx_shape,
                dtype=atom.operand_dtype("c"),
                nregs=atom.fragment_nregs("c"),
            )
            for f in range(n_folds)
            for i in range(m.reg)
            for j in range(n.reg)
        ]
        if _f16acc(atom):
            # The f32 shadow accumulators — they keep the ``_c{i}_{j}`` names the store reads;
            # the packed f16 mma targets above are the ``_ch{i}_{j}`` family FragmentPromote folds.
            decls += [
                RegFragment(name=_fold_frag(self.frag(f"_c{i}_{j}"), f), role="c", shape=atom.shape, dtype=F32)
                for f in range(n_folds)
                for i in range(m.reg)
                for j in range(n.reg)
            ]
        return decls

    def gmem_leaves(self, offset, mn):
        """The gmem-direct mma leaf constructors: ``ldmatrix`` each operand fragment straight from
        gmem, ``mma.sync`` every cell, the K-loop a ``StridedLoop`` of step ``atom_k`` (a symbolic /
        non-divisible K zero-fills the masked-K tail via the ``k_zero`` helper variants — canonical
        and transposed-B both have gmem-direct K zero-fill helpers)."""
        c = self.c
        atom, (m, n) = self.tile.atom, mn
        k_axis = c.axis
        assert isinstance(c.a, Load), "mma matmul arm: a register-resident (computed) A operand has no gmem-direct fragment loader here"
        assert len(self.channels) == 1, "gmem-direct mma is single-fold — a multi-B node rides the smem compute fill"
        a_load, b_load, b_trans = c.a, c.b, c.b_trans
        # The loop's final step overhangs K whenever ``atom_k`` does not tile it — a SYMBOLIC K
        # (unknown at compile time) or a static K with a remainder. Both mask the same way: the
        # loaders zero-fill the fragment halves past ``k_zero``'s bound, so the summed reduction
        # keeps its identity. An exactly tiled static K carries no bound (byte-identical output).
        k_exact = k_axis.extent.is_static and k_axis.extent.as_static() % atom.atom_k == 0
        k_zero = None if k_exact else (Var(k_axis.name), k_axis.extent_expr())

        def read_row(i):
            if self.slabs[0] is not None:
                # Already staged: the slab covers exactly this tile's M span, so the read is the
                # warp's own within-tile row and the K-loop's column.
                slab, ldm, swz = self.slabs[0]
                prim = BinaryExpr("+", BinaryExpr("*", Var(m.unit), Literal(m.reg * atom.atom_m, "int")), Literal(i * atom.atom_m, "int"))
                return [
                    LdmatrixLoad(
                        frag=self.frag(f"_a{i}"),
                        src_buffer=slab,
                        src_index=(prim, Var(k_axis.name)),
                        role="a",
                        staged=True,
                        ldm=ldm,
                        swizzle=swz,
                        fragment_layout=atom.fragment_layout,
                    )
                ]
            cell = offset[0].base(i)
            idx = tuple(Sigma({m.axis.name: cell}).apply(e) for e in a_load.index)
            return [
                LdmatrixLoad(
                    frag=self.frag(f"_a{i}"),
                    src_buffer=a_load.input,
                    src_index=idx,
                    role="a",
                    staged=False,
                    gmem_guard=_guard(m, cell),
                    k_zero=k_zero,
                    fragment_layout=atom.fragment_layout,
                )
            ]

        def read_col(j):
            if self.slabs[1] is not None:
                # Already staged: read the slab at the cell's LOCAL column (the slab covers exactly
                # this tile's N span, so the absolute base drops out) and the K-loop's own row.
                slab, ldm, swz = self.slabs[1]
                return [
                    LdmatrixLoad(
                        frag=self.frag(f"_b{j}"),
                        src_buffer=slab,
                        src_index=(Literal(j * atom.atom_n, "int"), Var(k_axis.name)),
                        role="b",
                        staged=True,
                        ldm=ldm,
                        b_trans=True,
                        swizzle=swz,
                        fragment_layout=atom.fragment_layout,
                    )
                ]
            cell = offset[1].base(j)
            idx = tuple(Sigma({n.axis.name: cell}).apply(e) for e in b_load.index)
            return [
                LdmatrixLoad(
                    frag=self.frag(f"_b{j}"),
                    src_buffer=b_load.input,
                    src_index=idx,
                    role="b",
                    staged=False,
                    b_trans=b_trans,
                    gmem_guard=_guard(n, cell),
                    k_zero=k_zero,
                    fragment_layout=atom.fragment_layout,
                )
            ]

        def contract(i, j):
            return [
                MmaSyncPtx(
                    c_frag=self.frag(_mma_c_base(atom, i, j)),
                    a_frag=self.frag(f"_a{i}"),
                    b_frag=self.frag(f"_b{j}"),
                    shape=atom.ptx_shape,
                    ab_dtype=atom.ab_dtype,
                    c_dtype=atom.operand_dtype("c").name,
                )
            ]

        def wrap(body):
            step = Literal(atom.atom_k, "int")
            stmts, tail = list(body), []
            if _f16acc(atom):
                # Promote every _F16ACC_STEPS atom-K steps (a compile-time-foldable modulo when
                # the loop unrolls), plus the unconditional final fold after the loop — it also
                # covers a symbolic / non-multiple K's partial last chunk.
                promotes = _f16acc_promotes(m.reg, n.reg, 1, self.frag_ns)
                period = atom.atom_k * _F16ACC_STEPS
                fire = BinaryExpr("==", BinaryExpr("%", Var(k_axis.name), Literal(period, "int")), Literal(period - atom.atom_k, "int"))
                stmts.append(Cond(cond=fire, body=tuple(promotes)))
                tail = promotes
            return [
                StridedLoop(axis=k_axis, start=Literal(0, "int"), step=step, body=Body(tuple(stmts)), unroll=unroll_ok(k_axis.extent)),
                *tail,
            ]

        return dict(read_row=read_row, read_col=read_col, contract=contract, wrap=wrap)

    def store(self, i, j, offset, mn):
        """Store cell ``(i, j)``'s ``_c`` fragment to the output, folding the projection ``tail`` into a
        :class:`RegEpilogue` and guarding overhanging M/N rows. A multi-fold node binds its extra C
        fragments as additional epilogue accumulators (the combine — SwiGLU — reads them per cell)."""
        atom = self.tile.atom
        m, n = mn
        mcell, ncell = offset[0].base(i), offset[1].base(j)
        tail = list(self.epilogue)
        sigma = Sigma({m.axis.name: mcell, n.axis.name: ncell})
        chans = self.channels
        accs = tuple(acc for _, acc in chans)
        frags = (self.frag(f"_c{i}_{j}"), *(_fold_frag(self.frag(f"_c{i}_{j}"), f) for f in range(1, len(chans))))
        writes = [s for s in tail if isinstance(s, Write)]
        body = Body(tail)
        out = []
        for write in writes:
            cone = body.backward_cone((write.value,))
            used = [f for f, acc in enumerate(accs) if acc in cone.external_reads]
            if not used:
                from emmy.compiler.pipeline import RuleSkipped  # noqa: PLC0415 — avoid an import cycle

                raise RuleSkipped(f"fragment projection for {write.output!r} reads no contraction accumulator")
            primary = used[0]
            extra = tuple((accs[f], frags[f]) for f in used[1:])
            epi = _warp_epilogue([*cone.members, write], accs[primary], m.axis.name, n.axis.name, sigma, extra_accs=extra)
            out.append(
                RegStore(
                    dst_buffer=write.output,
                    dst_index=tuple(sigma.apply(e) for e in write.index),
                    frag=frags[primary],
                    shape=atom.shape,
                    epilogue=epi,
                    m_guard=_guard(m, mcell),
                    n_guard=_guard(n, ncell),
                    atomic=write.atomic,
                    swizzle=write.swizzle,
                    fragment_layout=atom.fragment_layout,
                    row_dim=_axis_dim(write.index, m.axis.name),
                    col_dim=_axis_dim(write.index, n.axis.name),
                )
            )
        return out


class _ScalarOps(_AtomOps):
    """Scalar fma atom — plain ``Load``\\ s + an ``fma`` cell, the replicated-``epilogue`` sink."""

    def slab_elem(self):
        """The slab element dtype — the gmem operand's own dtype (fp32 SGEMM stages fp32)."""
        return self.inputs[self.c.a.input].dtype

    def slab_elems(self) -> tuple:
        """Each gmem operand's OWN dtype — A and B may differ on the scalar tier (fp32 split
        partials × fp16 weights); the drain's fma converts like the gmem-direct path does."""
        return (self.inputs[self.c.a.input].dtype, self.inputs[self.c.b.input].dtype)

    def staged_drain(self, operands, slot, cells, offset, mn):
        """The scalar slab drain — the plain-``Load`` fma leaf (:func:`_scalar_drain`), reading by
        LOCAL tile coords over ring ``slot`` (the ``depth >= 2`` gmem→smem ring offsets each slab's
        row by the slot, exactly as the mma drain does)."""
        a_op, b_op = operands
        offs = tuple(op.slot_row(slot) for op in operands)
        return [_scalar_drain(self.c, cells, offset, (a_op.slab, b_op.slab), "_ki", self.stage.bk_elems, _tile_base(mn), offs)]

    def state(self, cells):
        """The scalar accumulator seeds. Gmem-direct (unstaged): none — the accumulators are seeded
        inside the reduce ``Loop`` (the dissolved fold ``Accum``\\ s + ``Loop.render``). **Staged**: a
        per-cell ``Init(acc__c{i}_{j} = 0)`` emitted here, **outside** the outer slab loop, so the
        carrier-less :func:`_scalar_drain` folds across every slab without re-seeding (the nested-loop
        accumulator-lifetime split — the scalar analogue of :meth:`_MmaOps.state` declaring the ``_c``
        fragments outside the mma K-loop)."""
        c = self.c
        if self.stage is None:
            return []
        return [Init(name=f"{c.acc}__c{i}_{j}", identity=_ADD.identity, dtype=F32) for i, j in cells]

    def gmem_leaves(self, offset, mn):
        """The gmem-direct scalar leaf constructors: each register ROW reads its A operand once (a
        gmem ``Load`` or a computed register-resident body), each COL its
        B ``Load`` once, each ``(i, j)`` cell folds ``acc__c{i}_{j} += b·a``, and the K-loop is a unit
        ``Loop`` (``Loop.render`` seeds the accumulators; the store reads them). A masked axis wraps
        its read in-bounds (``% extent``) and the overhanging store is guarded (:meth:`store`).

        An operand that VARIES ALONG THE OTHER output axis (:func:`_cell_varying`) is read once per
        CELL instead — the row / column reuse is a property of the operand, not of the tier."""
        c = self.c
        assert len(self.channels) == 1, "the scalar tier is single-fold — a multi-B node rides the warp smem compute fill"
        k_axis = c.axis
        m, n = mn
        # The operand bodies contribute their OWN loop coordinates (a computed cone's internal
        # fold axes): a replicated read of such a coordinate must keep its name — the loop that
        # binds it is copied with the cell, so suffixing the reads (but never a Loop's binding)
        # emitted references no scope defines.
        prot = _scalar_protected(c, self.tile, self.lead, body=(*operand_body(c.a), *operand_body(c.b)))
        b_name, a_name = operand_name(c.b), operand_name(c.a)
        a_body, b_body = operand_body(c.a), operand_body(c.b)
        a_cell, b_cell = _cell_varying(a_body, n), _cell_varying(b_body, m)

        def at_m(i):  # register row ``i``'s m coordinate (a 1-D output has no m side)
            return {} if m is None else {m.name: _wrap(m, offset[0].base(i))}

        def at_n(j):  # register column ``j``'s n coordinate
            return {n.name: _wrap(n, offset[1].base(j))}

        # Each operand's σ also binds the SIBLING output axis: a value-dead reshape/broadcast
        # residue can keep a syntactic reference to it in the operand index (the same rule the
        # staged fills apply — the read is sibling-invariant in VALUE, so any in-bounds
        # representative evaluates it unchanged, and the bare axis name no longer exists after
        # the tile split).
        def read_row(i):
            return [] if a_cell else copy_cell(a_body, Sigma(at_m(i)), f"__ar{i}", prot)

        def read_col(j):
            return [] if b_cell else copy_cell(b_body, Sigma(at_n(j)), f"__bc{j}", prot)

        def contract(i, j):
            # A cell-varying operand's read lands here, σ-bound to BOTH coordinates and suffixed with
            # the full cell, so every coordinate it mentions is substituted and each cell owns its copy.
            cell = Sigma({**at_m(i), **at_n(j)})
            a_sfx = f"__ar{i}_{j}" if a_cell else f"__ar{i}"
            b_sfx = f"__bc{i}_{j}" if b_cell else f"__bc{j}"
            reads = [
                *(copy_cell(a_body, cell, a_sfx, prot) if a_cell else ()),
                *(copy_cell(b_body, cell, b_sfx, prot) if b_cell else ()),
            ]
            v = f"{c.acc}__v__c{i}_{j}"
            return [
                *reads,
                Assign(name=v, op=_MUL, args=(f"{b_name}{b_sfx}", f"{a_name}{a_sfx}")),
                Accum(name=f"{c.acc}__c{i}_{j}", value=v, op=_ADD, axes=(k_axis.name,)),
            ]

        def wrap(body):
            return [Loop(axis=k_axis, body=Body(tuple(body)), unroll=_unroll_inner(k_axis))]

        return dict(read_row=read_row, read_col=read_col, contract=contract, wrap=wrap)

    def store(self, i, j, offset, mn):
        """Replicate the projection ``tail`` for cell ``(i, j)`` — σ-offset, suffix the SSA names, guard
        the (overhanging) write, dedup shared operand loads."""
        c = self.c
        sigma = _scalar_sigma(mn, offset, i, j)
        cell = copy_cell(self.epilogue, sigma, f"__c{i}_{j}", _scalar_protected(c, self.tile, self.lead, body=self.epilogue))
        cell = _guard_writes(cell, _scalar_bound(mn, offset, i, j))
        return _dedup_loads(cell)


def _atom_ops(
    c: Fold,
    tile: Tile,
    stage: Stage | None = None,
    inputs=None,
    workers=None,
    epilogue: Body | None = None,
    seam=None,
    lead: tuple = (),
    frag_ns: str = "",
    slabs: tuple = (None, None),
) -> _AtomOps:
    """The **one** atom dispatch — select the codegen strategy off the atom kind. ``c`` is the
    stored algebra, ``tile`` the PLACED schedule slice (``Tile.at``) the geometry derives from.

    A CONVERTING materialized ``a`` — an ``smem`` stage on a load whose dtype differs from the
    atom's — is normalized to its one-``Load`` cone HERE, at the decode boundary: the synchronous
    fill then evaluates the load per slab cell and the typed slab store performs the conversion
    (the scheduler resolved the fill for exactly this edge; the tree itself is never rewritten)."""
    if (
        stage is not None
        and stage.transport == "smem"
        and isinstance(tile.atom, AtomKind)
        and isinstance(c.a, Load)
        and inputs is not None
        and (t := inputs.get(c.a.input)) is not None
        and t.dtype != tile.atom.operand_dtype("a")
    ):
        # Rebuilds thread the node's OWN semiring — never the constructor default.
        mul, plus = c.semiring
        c = Fold.contraction(k_axis=c.axis, a=make_cone([c.a], c.axis.name), channels=c.channels, product=mul, fold_op=plus)
    cls = _MmaOps if isinstance(tile.atom, AtomKind) else _ScalarOps
    return cls(c, tile, stage, inputs, workers, lead, Body(()) if epilogue is None else epilogue, seam, frag_ns, slabs)


def _row_value(value: Value, op: ElementwiseImpl, names: tuple[tuple[str, str], ...], group: int) -> tuple[list[Stmt], Value]:
    """Reduce a fragment-distributed value over its columns into one scalar per physical row."""
    assert value.kind == FRAG
    body = [
        FragmentRowReduce(top=names[i][0], bot=names[i][1], frags=tuple(value.data[i]), op=op, group=group) for i in range(len(value.data))
    ]
    return body, Value.row(names)


def _carrier_values(fold: Fold, ops: _AtomOps, cells) -> tuple[list[Stmt], dict[str, Value]]:
    """Declare a Fold carrier at the residence selected by its scheduled child edges."""
    lay = frag_layout(ops.tile.atom.atom_m, ops.tile.atom.atom_n)
    fragment_states = {channel.acc: index for index, channel in enumerate(ops.c.channels)}
    values: dict[str, Value] = {}
    decls = list(ops.state(cells))
    for name, identity in zip(fold.combine.results, fold.init, strict=True):
        if name in fragment_states:
            channel = fragment_states[name]
            values[name] = Value.frag(
                tuple(tuple(_fold_frag(ops.frag(f"_c{i}_{j}"), channel) for j in range(ops.tile.n.reg)) for i in range(ops.tile.m.reg))
            )
            continue
        rows = tuple(tuple(f"_{name}_r{i}_{r}" for r in range(lay.rows_per_lane)) for i in range(ops.tile.m.reg))
        values[name] = Value.row(rows)
        decls.extend(Init(name=row, identity=identity, dtype=F32) for pair in rows for row in pair)
    return decls, values


def _projection_finalize(tail: tuple, carried: dict[str, Value], target: Value) -> list[Stmt]:
    """Evaluate a pure post-fold projection at carrier residence and place it in ``target``."""
    writes = [stmt for stmt in tail if isinstance(stmt, Write)]
    pure = [stmt for stmt in tail if not isinstance(stmt, Write)]
    if not writes or not pure:
        return []
    result = writes[0].value
    lam = Lambda(params=tuple(carried), body=Body(tuple(pure)), results=(result,))
    body, (value,), _ = evaluate(lam, carried)
    if value.kind != FRAG or target.kind != FRAG:
        raise ValueError("a fragment-resident Fold projection must produce fragments")
    body.extend(
        FragmentApply(
            out=target.data[i][j],
            op=ElementwiseImpl("copy"),
            args=(value.data[i][j],),
            kinds=(FRAG,),
            in_place=True,
        )
        for i in range(len(target.data))
        for j in range(len(target.data[i]))
    )
    return body


def _fold_staged(
    ops: _AtomOps,
    fold: Fold,
    value_child: Fold,
    sched,
    cells,
    offset,
    mn: tuple[Side, Side],
    carried: dict[str, Value],
    projection: tuple,
):
    """Evaluate one scheduled Fold block through generic residence and child callbacks."""
    stage, c, tile = ops.stage, ops.c, ops.tile
    if stage is None or stage.transport != "smem" or not isinstance(tile.atom, AtomKind):
        raise ValueError(f"fragment Fold evaluation requires a scheduled synchronous MMA stage, got {stage!r}")
    if len(c.channels) != 1:
        raise ValueError("fragment Fold evaluation currently requires one fragment-resident carrier channel")
    steps = fold.step_stmts()
    producer = next(
        (stmt for stmt in steps if is_contraction(stmt) and stmt is not value_child and sched.tile_of(stmt) is not None),
        None,
    )
    producer_tile = sched.tile_of(producer) if producer is not None else None
    active_tile = producer_tile or tile
    lay = frag_layout(active_tile.atom.atom_m, active_tile.atom.atom_n)
    bk = stage.bk_elems
    cta = _cta(mn, tile.atom.lanes, tile.launch_threads)
    elem = ops.slab_elem()
    swizzles = ops.slab_swizzles(mn, elem.nbytes)
    m, n = mn
    row_base, col_base = _tile_base(mn)

    producer_stage = sched.get("STAGE", producer) if producer is not None else None
    stream_op = _child_stream_operand(c=c, child=producer, atom=producer_tile.atom, m_name=m.axis.name, cols=bk) if producer_stage else None
    invariant_op = (
        _child_invariant_operand(child=producer, atom=producer_tile.atom, m=producer_tile.m, k_name=producer.axis.name)
        if producer_stage
        else None
    )

    b = c.b
    assert isinstance(b, Load), "a fragment Fold child currently requires a materialized streamed operand"
    b_shape = (n.tile, bk) if c.b_trans else (bk, n.tile)
    b_op = Operand(
        tag="b",
        buf=b.input,
        shape=b_shape,
        coords=_box_origin(b.index, tile=n, tile_base=col_base, k_axis=c.axis, sibling=m),
        index=_slab_index(b.index, tile=n, tile_base=col_base, k_axis=c.axis, tile_is_row=c.b_trans, sibling=m),
        swizzle=swizzles[1],
        trans=c.b_trans,
    )

    k0 = Var("_ks")
    # Runtime bounds on the fragment coordinates: the reduced K overhang clamp-reads AND masks to
    # the fold identity (the fold consumes it); a masked M only clamp-reads — the overhanging row
    # computes a duplicate of the last valid row and its store is discarded by the ``RegStore``
    # guard, the same contract the copy transports follow.
    bounds: tuple[tuple[str, Expr, float | None], ...] = (
        () if fold.axis.extent.is_static else ((fold.axis.name, fold.axis.extent.expr, float(fold.init[0])),)
    )
    if m.mask:
        bounds += ((m.axis.name, m.ext, None),)
    if producer is not None:
        producer_slabs = (
            (invariant_op.slab, invariant_op.shape[1], invariant_op.swizzle) if invariant_op is not None else None,
            (stream_op.slab, stream_op.shape[1], stream_op.swizzle) if stream_op is not None else None,
        )
        _producer_ops, _producer_cells, producer_offset, _producer_mn, producer_body, producer_frags = _child_contraction_block(
            child=producer,
            tile=producer_tile,
            k0=k0,
            lead=ops.lead,
            ns=f"{ops.frag_ns}_fold",
            slabs=producer_slabs,
        )
        producer_value = Value.frag(producer_frags)
    else:
        producer_offset = (
            offset[0],
            _BlockCols(atom_dim=tile.atom.atom_n, k0=k0),
        )
        producer_body = []
        producer_value = None
    # The fragment column cells must cover the whole ``bk``-wide slab chunk the drain reads —
    # ``bk / atom_n`` cells, NOT the output tile's ``n.reg`` (the two agree only when
    # ``bk == n.reg · atom_n``; a deeper chunk left the slab tail unwritten). A scheduled
    # producer's grid is its own register tiling, so it must AGREE with that cover — refuse a
    # seam whose producer grid would leave (or overrun) the slab tail rather than emit it.
    # No modern atom can fail the divisibility — the k16/k32 families are 2·bk / 4·bk cells — and
    # the one that could (``mma_m8n8k4_f16_f32``: logical (16,16,4), so ``bk // atom_n`` is
    # ``tile.bk // 4`` and reads 0 at bk∈{1,2}) is unreachable today: sm70 computed-A rides the
    # per-cell sync fill, which never enters ``_fold_staged``. Kept so a future ladder or pin that
    # does reach it refuses loudly here rather than emitting a zero-width column grid over an
    # unwritten slab.
    if bk % active_tile.atom.atom_n:
        raise ValueError(
            f"the fragment Fold's {bk}-element slab chunk is not a whole number of "
            f"{active_tile.atom.atom_n}-column {active_tile.atom.name} fragment cells"
        )
    col_cells = bk // active_tile.atom.atom_n
    if producer is not None and active_tile.n.reg != col_cells:
        raise ValueError(
            f"fragment seam mismatch: the scheduled producer's {active_tile.n.reg}-cell "
            f"({active_tile.atom.atom_n}-column) fragment grid does not cover the "
            f"{bk}-element slab chunk the drain reads ({col_cells} cells required)"
        )
    bases = tuple(
        tuple((producer_offset[0].base(i), producer_offset[1].base(j)) for j in range(col_cells)) for i in range(active_tile.m.reg)
    )

    def child(node: Fold, env: dict[str, Value]):
        if node is producer:
            return [], producer_value
        if node.axis is None:
            body, results, _ = evaluate(
                node.lift,
                env,
                child=child,
                bases=bases,
                axes=(m.axis.name, fold.axis.name),
                bounds=bounds,
            )
            return body, results
        raise ValueError("only scheduled contraction children may execute inside a fragment Fold")

    bindings = {fold.axis.name: Value.uniform(fold.axis.name)}
    for param in fold.lift.params[1:]:
        bindings[param] = Value.uniform(param)
    prelude: list[Stmt] = []
    for node in (stmt for stmt in steps if isinstance(stmt, Fold) and stmt.axis is None):
        emitted, value = child(node, bindings)
        prelude.extend(emitted)
        values = value if isinstance(value, tuple) else (value,)
        bindings.update(zip(node.defines(), values, strict=True))
    lift_body, lifted, env = evaluate(
        fold.lift,
        bindings,
        child=child,
        bases=bases,
        axes=(m.axis.name, fold.axis.name),
        bounds=bounds,
    )
    pivot_accum = next(stmt for stmt in steps if isinstance(stmt, Accum) and stmt.name == fold.combine.results[0])
    raw_pivot_names = tuple(tuple(f"_fold_pivot_raw_{i}_{r}" for r in range(lay.rows_per_lane)) for i in range(active_tile.m.reg))
    pivot_body, raw_pivot = _row_value(lifted[0], pivot_accum.op, raw_pivot_names, lay.reduce_group)
    pivot_names = tuple(tuple(f"_fold_pivot_{i}_{r}" for r in range(lay.rows_per_lane)) for i in range(active_tile.m.reg))
    pivot_identity = "_fold_pivot_identity"
    pivot_body.append(Init(name=pivot_identity, identity=fold.init[0], dtype=F32))
    pivot_body.extend(
        Assign(name=pivot_names[i][r], op=pivot_accum.op, args=(raw_pivot.data[i][r], pivot_identity), dtype=F32)
        for i in range(active_tile.m.reg)
        for r in range(lay.rows_per_lane)
    )
    pivot = Value.row(pivot_names)

    env.update(carried)
    env[fold.combine.results[0]] = pivot
    for name, identity in zip(fold.combine.results[1:], fold.init[1:], strict=True):
        env[name] = Value.uniform(identity)
    value_at = steps.index(value_child)
    prefix = [stmt for stmt in steps[:value_at] if isinstance(stmt, (Assign, Select)) and stmt.defines()[0] not in env]
    if prefix:
        lam = Lambda(params=tuple(env), body=Body(tuple(prefix)), results=tuple(stmt.defines()[0] for stmt in prefix))
        prefix_body, _, env = evaluate(lam, env, bases=bases, axes=(m.axis.name, fold.axis.name), bounds=bounds)
    else:
        prefix_body = []
    weight_body, (weight,), env = evaluate(
        value_child.a.lift,
        env,
        child=child,
        bases=bases,
        axes=(m.axis.name, fold.axis.name),
        bounds=bounds,
    )
    if weight.kind != FRAG:
        raise ValueError("a fragment contraction producer must produce a fragment operand")

    partial: dict[str, Value] = {fold.combine.results[0]: pivot}
    producer_body = [*producer_body, *prelude, *lift_body, *pivot_body, *prefix_body, *weight_body]
    for index, stmt in enumerate(steps[:value_at]):
        if not isinstance(stmt, Accum) or stmt.name not in fold.combine.results or stmt.value not in env:
            continue
        value = env[stmt.value]
        if value.kind != FRAG:
            continue
        names = tuple(tuple(f"_fold_partial_{index}_{i}_{r}" for r in range(lay.rows_per_lane)) for i in range(active_tile.m.reg))
        reduced, partial[stmt.name] = _row_value(value, stmt.op, names, lay.reduce_group)
        producer_body.extend(reduced)
    producer_body.extend(
        RegStore(
            dst_buffer="_a_smem",
            dst_index=(
                BinaryExpr("-", bases[i][j][0], row_base),
                BinaryExpr("-", bases[i][j][1], k0),
            ),
            frag=weight.data[i][j],
            shape=active_tile.atom.shape,
            ldm=bk,
            swizzle=swizzles[0],
            fragment_layout=active_tile.atom.fragment_layout,
            row_dim=0,
            col_dim=1,
        )
        for i in range(len(weight.data))
        for j in range(len(weight.data[i]))
    )

    def unreachable(_k0, _row, _col):
        raise AssertionError("a whole-slab scheduled producer has no scalar value callback")

    a_op = SyncOperand(tag="a", shape=(m.tile, bk), value=unreachable, producer=lambda _k0: producer_body, swizzle=swizzles[0])
    operands = (a_op, b_op)
    transport = SyncTransport(
        operands=(a_op,),
        copy_operands=(b_op,),
        invariant_operands=(invariant_op,) if invariant_op is not None else (),
        slab_dtype=cuda_name(elem),
        elem_bytes=elem.nbytes,
        cta=cta,
        copy_sync=tile.atom.sync_copy_staging,
    )
    stream = CpAsyncTransport(operands=(stream_op,), slab_dtype=cuda_name(elem), elem_bytes=elem.nbytes, cta=cta) if stream_op else None
    lead_segment = LeadSegment(build=lambda: producer_body, transport=stream)
    block_ops = replace(ops, frag_ns=f"{ops.frag_ns}_partial")
    fragment_state = next(channel.acc for channel in c.channels)

    def drain(slot):
        block = Value.frag(tuple(tuple(block_ops.frag(f"_c{i}_{j}") for j in range(n.reg)) for i in range(m.reg)))
        partial[fragment_state] = block
        bindings = {
            **{param: carried[name] for param, name in zip(fold.combine.params[: len(carried)], fold.combine.results, strict=True)},
            **{param: partial[name] for param, name in zip(fold.combine.params[len(carried) :], fold.combine.results, strict=True)},
        }
        merged, _, _ = evaluate(fold.combine, bindings, targets=carried)
        return [*block_ops.state(cells), *block_ops.staged_drain(operands, slot, cells, offset, mn), *merged]

    extent = fold.axis.extent.as_static() if fold.axis.extent.is_static else fold.axis.extent
    n_chunks = (
        extent // bk
        if isinstance(extent, int)
        else Dim(BinaryExpr("/", BinaryExpr("+", extent.expr, Literal(bk - 1, "int")), Literal(bk, "int")))
    )
    pre, region = staged_kloop(
        transport=transport,
        drain=drain,
        depth=stage.depth,
        bk_elems=bk,
        n_chunks=n_chunks,
        k_extent=extent,
        workers=ops.workers,
        block_threads=tile.launch_threads,
        lead=lead_segment,
    )
    target = carried[fragment_state]
    return pre, [*region, *_projection_finalize(projection, carried, target)]


def reduce_codegen(
    c: Fold,
    tile: Tile,
    stage: Stage | None = None,
    inputs=None,
    workers=None,
    seam=None,
    lead: tuple = (),
    frag_ns: str = "",
    *,
    fold: Fold | None = None,
    value_child: Fold | None = None,
    sched=None,
    projection: tuple = (),
    carried: dict[str, Value] | None = None,
):
    """The reusable, **sink-agnostic** ``(state_decls, reduce_region)`` from the atom strategy — the
    accumulator decls + the contraction K-loop (the ONE :meth:`_AtomOps.reduce` driver: the shared
    :func:`_contract_kloop` spine gmem-direct, the shared :func:`_staged` fill→drain skeleton staged).
    ``stage`` / ``inputs`` bind operand staging (both atoms stage the same smem slab off it, differing
    only in the drain leaf — ``ldmatrix`` vs plain ``Load``); ``workers`` splits the staged phases
    across producer / compute warp bands (the resolved :class:`WarpSpec`; ``None`` = uniform)."""
    ops = _atom_ops(c, tile, stage, inputs, workers, seam=seam, lead=lead, frag_ns=frag_ns)
    if fold is None:
        return ops.state, ops.reduce
    if value_child is None or sched is None:
        raise ValueError("fragment Fold codegen requires its scheduled child and schedule")
    carried = {} if carried is None else carried

    def state(cells):
        decls, values = _carrier_values(fold, ops, cells)
        carried.update(values)
        return decls

    def reduce(cells, offset, mn):
        return _fold_staged(ops, fold, value_child, sched, cells, offset, mn, carried, projection)

    return state, reduce


def fold_store_tail(tail: tuple, fold: Fold, c: _ScheduledContraction) -> tuple:
    """Keep the boundary writes after a fragment Fold projection.

    Direct carrier writes retain their component name so the sink can store its physical
    residence.  A write of the projection result reads the fragment channel into which
    :func:`_projection_finalize` placed that result.
    """
    states = set(fold.combine.results)
    return tuple(
        replace(stmt, values=tuple(value if value in states else c.acc for value in stmt.values)) if isinstance(stmt, Write) else stmt
        for stmt in tail
        if not stmt.pure
    )


def fold_store_sink(
    tile: Tile,
    effects: tuple,
    carried: dict[str, Value],
    frag_ns: str = "",
):
    """Store each Fold carrier component from its selected physical residence."""
    atom = tile.atom
    layout = frag_layout(atom.atom_m, atom.atom_n)

    def store(i, j, offset, mn):
        m, n = mn
        mcell, ncell = offset[0].base(i), offset[1].base(j)
        sigma = Sigma({m.axis.name: mcell, n.axis.name: ncell})
        body: list[Stmt] = []
        for index, write in enumerate(stmt for stmt in effects if isinstance(stmt, Write)):
            if not write.is_scalar:
                raise ValueError("a fragment Fold boundary write must be scalar before vectorization")
            value = carried[write.value]
            if value.kind == FRAG:
                fragment = value.data[i][j]
            else:
                fragment = f"{frag_ns}_{write.value}_store_{index}_{i}_{j}"
                arg = value.data[i] if value.kind == ROW else value.data
                body.append(
                    FragmentApply(
                        out=fragment,
                        op=ElementwiseImpl("copy"),
                        args=(arg,),
                        kinds=(value.kind,),
                        layout=layout,
                    )
                )
            body.append(
                RegStore(
                    dst_buffer=write.output,
                    dst_index=tuple(sigma.apply(expr) for expr in write.index),
                    frag=fragment,
                    shape=atom.shape,
                    m_guard=_guard(m, mcell),
                    n_guard=_guard(n, ncell),
                    atomic=write.atomic,
                    swizzle=write.swizzle,
                    fragment_layout=atom.fragment_layout,
                    row_dim=_axis_dim(write.index, m.axis.name),
                    col_dim=_axis_dim(write.index, n.axis.name),
                )
            )
        return body

    return store


def store_sink(c: Fold, tile: Tile, epilogue: Body | None = None, lead: tuple = (), frag_ns: str = ""):
    """The default **matmul sink** — the per-cell ``store(i, j, offset, mn)`` from the atom strategy
    (an mma ``RegStore`` / the replicated scalar ``epilogue`` tail), folding in the ``epilogue`` (the
    projection off the node's zero-axis ``Fold`` wrapper + the store glue). A caller may replace
    the sink while reusing the shared ``reduce`` emission."""
    return _atom_ops(c, tile, epilogue=epilogue, lead=lead, frag_ns=frag_ns).store
