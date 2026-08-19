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

A computed-A cone that COMPOSES a score contraction (attention) reaches the tensor core through that
same seam, twice, without a second emitter: the **CHAINED A fill** (:func:`chain_a_fill`) computes
the slab with a nested contraction and folds the cone into its fragment store, and the **CHAINED
statistic** (:func:`chain_stat_fill`) sweeps the cone's per-row reduce at fragment residence over KV
blocks. Both are :func:`_atom_ops` on the SCORE node (:func:`_score_block`), namespaced by
:attr:`_AtomOps.frag_ns` so a nested emission never shadows the enclosing drain's accumulators; both
decline (and leave the per-cell / cooperative-row form standing) wherever the score does not read as
a contraction or the atom has no modeled C-fragment layout.

Leading ``_`` so the pass loader (globs ``*.py``, skips ``_``-prefixed) skips it."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from emmy.compiler.backend.cuda.dtype import cuda_name
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.ir.atom import AtomKind
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal, TernaryExpr, Var
from emmy.compiler.ir.kernel.ir import (
    FRAG,
    ROW,
    UNIFORM,
    EpilogueLoad,
    FragmentApply,
    FragmentPromote,
    FragmentRowReduce,
    LdmatrixLoad,
    MmaSyncPtx,
    RegEpilogue,
    RegFragment,
    RegStore,
    Smem,
    Sync,
    frag_layout,
)
from emmy.compiler.ir.pure.carrier import exp_merge
from emmy.compiler.ir.pure.fold import Fold, edge_refs_axis, is_contraction, operand_body, operand_name
from emmy.compiler.ir.schedule import Side, Stage, TilePlan
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Accum, Assign, Body, Cond, Init, Load, Loop, Select, SelectBranch, Stmt, StridedLoop, Write
from emmy.compiler.pipeline.passes.lowering._addr import BYTE_SLAB_PAD
from emmy.compiler.pipeline.passes.lowering._reduction import Reduction
from emmy.compiler.pipeline.passes.lowering.kernel._stage import (
    CpAsyncTransport,
    CtaTile,
    Operand,
    SyncOperand,
    SyncTransport,
    TmaTransport,
    pick_swizzle_atom,
    staged_kloop,
    sync_stat_fill,
)
from emmy.compiler.pipeline.passes.lowering.kernel._tiling import AxisOffset
from emmy.compiler.pipeline.search.space import UNROLL

#: The contraction semiring — multiply ⊗ then accumulate ⊕ (add). The same multiply-add ``mma.sync``
#: realizes; in the scalar tier it is a plain scalar fma loop.
_MUL = ElementwiseImpl("multiply")
_ADD = ElementwiseImpl("add")


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


def _cells(mn: tuple, offset, i: int, j: int):
    """Yield ``(side, cell-base coord)`` for each present output axis of register cell ``(i, j)`` —
    ``(m, offset[0].base(i))`` then ``(n, offset[1].base(j))`` (``m`` skipped for a 1-D output)."""
    for side, off, r in ((mn[0], offset[0], i), (mn[1], offset[1], j)):
        if side is not None:
            yield side, off.base(r)


# ---- warp/mma tier ----------------------------------------------------------------------------- #
def _warp_roles(index, m_name: str, n_name: str) -> tuple[str, ...]:
    """Per-dim epilogue-load role: ``"m"`` / ``"n"`` for a dim varying with the output row /
    col axis, else ``"fixed"`` (batch / grid literal — uniform across the fragment cell)."""
    roles = []
    for e in index:
        fv = e.free_vars()
        roles.append("m" if m_name in fv else "n" if n_name in fv else "fixed")
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
    ``m``/``n`` dims); each ``Assign`` becomes an ``(name, op, args)`` op; a coord-predicated
    ``Select`` (causal mask) rewrites its ``m``/``n`` coordinate vars to the ``__M__`` / ``__N__``
    placeholders the store substitutes with the element's own (row, col)."""
    loads, ops, selects = [], [], []
    write = None
    ph = {m_name: Var("__M__"), n_name: Var("__N__")}
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
            ops.append((s.name, s.op.name, tuple(s.args)))
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
    for name, _op, args in ops:
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


def _f16acc_promotes(m_reg: int, n_reg: int, n_folds: int) -> list[Stmt]:
    """One :class:`FragmentPromote` per C cell × fold channel — the f16 chunk fold into the f32
    shadows (also the FINAL fold: the shadows carry the full sum only after it runs)."""
    return [
        FragmentPromote(dst=_fold_frag(f"_c{i}_{j}", f), src=_fold_frag(f"_ch{i}_{j}", f))
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
    ``BYTE_SLAB_PAD``) rides the drain ``ldm`` so reads stride the padded rows."""
    (a_slab, *b_slabs), (m, n) = slabs, mn
    offs = offs if offs is not None else (None,) * len(slabs)
    swizzles = swizzles if swizzles is not None else ("NONE",) * len(slabs)
    trans = trans if trans is not None else (False,) * len(slabs)
    byte_slabs = byte_slabs if byte_slabs is not None else (False,) * len(slabs)
    pads = pads if pads is not None else (0,) * len(slabs)
    atom_m, atom_n, atom_k = atom.shape
    n_steps = bk_elems // atom_k
    # Per-operand drain spec: (frag base fn, slab, ldm, tile-is-slab-row, reg count, warp-unit var,
    # atom dim, slot row off, swizzle, byte flag). A stacks the tile axis on the slab row (K the
    # col); B swaps (K the row, tile the col) — unless its slab is transposed (N-major: tile the
    # row, K the col, like A); the slot offset always lands on the ROW. All share ONE emission loop.
    specs = [(lambda x: f"_a{x}", "a", a_slab, bk_elems + pads[0], True, m.reg, m.unit, atom_m, offs[0], swizzles[0], byte_slabs[0])]
    for f, bs in enumerate(b_slabs):
        frag_of = (lambda ff: lambda x: _fold_frag(f"_b{x}", ff))(f)
        tr = trans[1 + f]
        ldm_b = (bk_elems if tr else n.tile) + pads[1 + f]
        specs.append((frag_of, "b", bs, ldm_b, tr, n.reg, n.unit, atom_n, offs[1 + f], swizzles[1 + f], byte_slabs[1 + f]))

    def ldms(kexpr, suffix):  # every operand's ldmatrix reads at K position `kexpr`, into fragment slot `suffix`
        reads: list[Stmt] = []
        for frag_of, role, slab, ldm, is_row, reg, unit, adim, off, swz, b8 in specs:
            assert not (b8 and swz != "NONE"), "a byte slab stays NONE-swizzle (the ldmatrix XOR is b16-indexed)"
            for x in range(reg):  # within-tile coord for register cell x: warp·(reg·adim) + x·adim
                prim = BinaryExpr("+", BinaryExpr("*", Var(unit), Literal(reg * adim, "int")), Literal(x * adim, "int"))
                row, col = (prim, kexpr) if is_row else (kexpr, prim)
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
                        fragment_layout=atom.fragment_layout,
                    )
                )
        return reads

    def mmas(suffix):  # every fold channel × (i, j) cell's mma.sync over the `suffix`-slotted operand fragments
        return [
            MmaSyncPtx(
                c_frag=_fold_frag(_mma_c_base(atom, i, j), f),
                a_frag=f"_a{i}{suffix}",
                b_frag=f"{_fold_frag(f'_b{j}', f)}{suffix}",
                shape=atom.ptx_shape,
                ab_dtype=atom.ab_dtype,
                c_dtype=atom.operand_dtype("c").name,
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
    (:data:`~emmy.compiler.pipeline.passes.lowering._addr.BYTE_SLAB_PAD` on a
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


def _chain_of(*, c, score, mn, atom, bk_elems, slab, stats, lead):
    """The chained fill CLOSURE — ``chain(k0)`` for the transport, deferring the emission until the
    skeleton knows its chunk base."""

    def chain(k0):
        return chain_a_fill(c=c, score=score, mn=mn, atom=atom, bk_elems=bk_elems, k0=k0, slab=slab, stats=stats, lead=lead)

    return chain


def chain_edge(cone, k_name: str):
    """The cone's K-VARYING producer edge when it is a CONTRACTION over its own axis — the score
    the CHAINED fill realizes on the tensor core — or ``None`` (no such edge, or it is not a
    contraction, so only the per-cell evaluation exists)."""
    if not isinstance(cone, Fold) or cone.axis is not None:
        return None
    kv = [e for e in cone.operands if isinstance(e, Fold) and e.axis is not None and edge_refs_axis(e, k_name)]
    return kv[0] if len(kv) == 1 and is_contraction(kv[0]) and isinstance(kv[0].a, Load) else None


def _lane_group(lay) -> Expr:
    """The lane's fragment ROW GROUP (``_g``) written over ``_lane`` — the same value the fragment
    store's ``lane_decl`` binds, for a reader that is not a fragment store."""
    return BinaryExpr("/", BinaryExpr("%", Var("_lane"), Literal(32, "int")), Literal(lay.n_elems, "int"))


@dataclass(frozen=True)
class _ChainCols:
    """The score tile's COLUMN offset — the K block it covers. Duck-types :class:`AxisOffset`'s one
    reader (:meth:`base`): a block column has no grid block and no unit, only the block base."""

    atom_dim: int
    k0: Expr

    def base(self, r: int) -> Expr:
        return BinaryExpr("+", self.k0, Literal(r * self.atom_dim, "int")) if r else self.k0


def _score_block(*, n_axis: Axis, score: Fold, mn: tuple[Side, Side], atom, cols: int, k0: Expr, lead: tuple, ns: str, epilogue=None):
    """One BLOCK of the composed score, on the tensor core — ``mma(Q, Kᵀ)`` over the score's own
    axis into C fragments covering ``tile_m × cols`` of the enclosing contraction's ``(m, K)`` plane.

    Built out of the SAME atom strategy every tiled contraction dispatches through
    (:func:`_atom_ops` on the score node), so there is no second mma emitter here: its ``state``
    declares the fragments, its ``reduce`` emits the gmem-direct ``ldmatrix`` + ``mma.sync`` K-loop
    over the score's own axis, and its ``store`` — when the caller supplies an ``epilogue`` — writes
    them out. ``ns`` namespaces the fragments, so a nested emission never shadows the accumulators
    the enclosing drain carries across the same loop.

    The ROWS are the enclosing tile's own, warp-partitioned and ABSOLUTE (the score of a query row
    belongs to that row, whichever warp owns it); the COLUMNS are the block, ``cols / atom_n``
    register columns wide. Returns ``(ops, cells, offset, mn, stmts, frags)`` with ``frags[i]`` the
    register row ``i``'s column fragments in column order."""
    m, _ = mn
    n_reg = cols // atom.atom_n
    tile = TilePlan(atom=atom, units=(m.units, 1), regs=(m.reg, n_reg)).at(m.axis, replace(n_axis, extent=Dim(cols)))
    ops = _atom_ops(score, tile, epilogue=epilogue, lead=lead, frag_ns=ns)
    offset = (
        AxisOffset(atom_dim=atom.atom_m, reg=m.reg, block_var=m.block, unit_var=m.unit, unit_count=m.units),
        _ChainCols(atom_dim=atom.atom_n, k0=k0),
    )
    cells = [(i, j) for i in range(m.reg) for j in range(n_reg)]
    decls = list(ops.state(cells))
    _, region = ops.reduce(cells, offset, tile.mn)
    frags = tuple(tuple(ops.frag(f"_c{i}_{j}") for j in range(n_reg)) for i in range(m.reg))
    return ops, cells, offset, tile.mn, [*decls, *region], frags


def _frag_lift(body, frags: tuple[tuple[str, ...], ...], acc: str) -> list[Stmt] | None:
    """The score's own lift (its scale, a mask factor) re-expressed over the score FRAGMENTS — the
    fragment-tier sibling of the scalar ``Assign``, one :class:`FragmentApply` per fragment. The
    fragment operand is the score accumulator; every other argument is cell-uniform. Declines
    (empty) for a lift stmt that does not read the accumulator — nothing else is expressible here."""
    out: list[Stmt] = []
    frag_names = {acc}
    for s in body:
        if not (set(s.deps()) & frag_names):
            out.append(s)  # a cell-uniform stmt of the lift (the scale ``Load``) — scalar, once
            continue
        if not isinstance(s, Assign) or s.args[0] not in frag_names:
            return None  # a shape the fragment tier cannot express — the caller keeps the scalar sweep
        frag_names |= set(s.defines())
        for row in frags:
            for f in row:
                out.append(
                    FragmentApply(
                        out=f,
                        op=s.op,
                        args=tuple(f if a in frag_names else (a if isinstance(a, str) else repr(a)) for a in s.args),
                        kinds=tuple(FRAG if a in frag_names else UNIFORM for a in s.args),
                        in_place=True,
                    )
                )
    return out


def chain_a_fill(
    *, c: Fold, score: Fold, mn: tuple[Side, Side], atom, bk_elems: int, k0: Expr, slab: str, stats: tuple[str, ...], lead: tuple
):
    """The **CHAINED A fill** — the score contraction the cone composes, realized on the TENSOR CORE
    with the cone as its store epilogue: ``mma(Q, Kᵀ) → C fragments → exp(s − m)·(1/d) → the A slab``
    the ordinary ``ldmatrix`` drain then reads. The one fill that computes its slab with a nested
    contraction instead of per-cell scalar code.

    The statistics the cone reads arrive at each fragment element's own ROW, out of the stat rows
    the prologue bridged; the slab ``Write`` lands at the element's LOCAL ``(row, col)`` — σ folds
    the absolute cell coordinate back to the slab's own. Returns the fill stmts."""
    m, _ = mn
    row_base = _side_base(m)
    local_row = BinaryExpr("-", Var(m.axis.name), row_base)
    local_col = BinaryExpr("-", Var(c.axis.name), k0)
    epilogue = Body(
        (
            *(Load(names=(nm,), input=_stat_slab(nm), index=(local_row,)) for nm in stats),
            *c.a.body,
            Write(output=slab, index=(local_row, local_col), value=c.a.out),
        )
    )
    ops, cells, offset, smn, stmts, _ = _score_block(
        n_axis=c.axis, score=score, mn=mn, atom=atom, cols=bk_elems, k0=k0, lead=lead, ns="_s", epilogue=epilogue
    )
    return [*stmts, *(s for i, j in cells for s in ops.store(i, j, offset, smn))]


def chain_stat_fill(*, c: Fold, mn: tuple[Side, Side], atom, cols: int, stats: tuple[str, ...], lead: tuple) -> list[Stmt] | None:
    """The **CHAINED statistic** — the cone's per-row streaming reduce swept at FRAGMENT residence.

    The scalar prologue (:func:`sync_stat_fill`) folds one tile row per warp with the 32 lanes
    striding it: it computes the same score the weight cone computes, one element at a time. Here
    the sweep is BLOCKED and the score is a contraction like any other — per block the tile's
    scores land in mma C-fragments (:func:`_score_block`), the block's row statistic comes off those
    fragments through the layout's shuffle butterfly (:class:`FragmentRowReduce`), and the
    carrier's OWN generated program merges the block into the running state: ``exp_merge`` at a
    BLOCK singleton ``(rowmax, rowsum)`` instead of an element singleton — the same generator, the
    same stability certificate, no attention vocabulary.

    Each lane ends holding the statistic for the two rows the fragment layout gives it, so one lane
    per column group publishes them to the stat rows the weight fill reads. ``None`` when this is
    not that shape (no bindable producer, an expectation channel, a carrier the block merge cannot
    spell) — the scalar prologue stands."""
    cone = c.a
    if not (isinstance(cone, Fold) and cone.axis is None and cone.operands):
        return None
    pro_node = cone.operands[0]
    if not (isinstance(pro_node, Fold) and pro_node.axis is None and pro_node.operands):
        return None
    stat = pro_node.operands[0]
    score = stat.operands[0] if isinstance(stat, Fold) and stat.axis is not None and len(stat.operands) == 1 else None
    if not (isinstance(score, Fold) and score.axis is not None and is_contraction(score) and isinstance(score.a, Load)):
        return None
    names = tuple(stat.combine.results) if stat.combine is not None else ()
    if len(names) != 2 or len(stat.lift.results) != 2 or stat.lift.results[1] != 1.0:
        return None  # the (m, d) pair only — an expectation channel is the streaming form, not this
    if not stat.axis.extent.is_static or stat.axis.extent.as_static() % cols:
        return None  # the block sweep unrolls whole blocks
    m, _ = mn
    try:
        lay = frag_layout(atom.atom_m, atom.atom_n)
    except NotImplementedError:
        return None  # an atom whose C layout the fragment tier does not model — the scalar prologue stands
    if lay.rows_per_lane != 2:
        return None
    k0 = Var("_sb")
    _ops, _cells, _off, _smn, stmts, frags = _score_block(
        n_axis=stat.axis, score=score, mn=mn, atom=atom, cols=cols, k0=k0, lead=lead, ns="_t"
    )
    lift = _frag_lift(stat.lift.body, frags, score.acc)
    if lift is None:
        return None
    body = [*stmts, *lift]
    state = [[tuple(f"_ss{i}_{r}_{x}" for x in range(2)) for r in range(2)] for i in range(m.reg)]
    for i in range(m.reg):
        rmax, rsum, pw = (f"_rm{i}_0", f"_rm{i}_1"), (f"_rs{i}_0", f"_rs{i}_1"), tuple(f"_p{i}_{j}" for j in range(len(frags[i])))
        body += [
            FragmentRowReduce(top=rmax[0], bot=rmax[1], frags=frags[i], op=ElementwiseImpl("maximum"), group=lay.reduce_group),
            *(
                FragmentApply(out=pw[j], op=ElementwiseImpl("subtract"), args=(f, rmax), kinds=(FRAG, ROW), post=(ElementwiseImpl("exp"),))
                for j, f in enumerate(frags[i])
            ),
            FragmentRowReduce(top=rsum[0], bot=rsum[1], frags=pw, op=ElementwiseImpl("add"), group=lay.reduce_group),
        ]
        for r in range(2):
            body += list(exp_merge(state[i][r], (rmax[r], rsum[r]), key=state[i][r][0]))
    # The stat rows the weight fill reads back — declared here, as the scalar prologue declares
    # them; the carrier's own state is seeded by the sweep loop's ``Accum``\ s.
    out: list[Stmt] = [Smem(name=_stat_slab(nm), extents=(mn[0].tile,), dtype="float") for nm in stats]
    out.append(
        StridedLoop(
            axis=Axis(name=k0.name, extent=stat.axis.extent),
            start=Literal(0, "int"),
            step=Literal(cols, "int"),
            body=Body(tuple(body)),
            unroll=False,
        )
    )
    # Publish: the statistic's scalar epilogue per owned row, then one lane per column group writes
    # each bridged value into its stat row.
    writes: list[Stmt] = []
    for i in range(m.reg):
        for r in range(2):
            sfx = f"__r{i}{r}"
            ren = dict(zip(names, state[i][r], strict=True))
            for s in pro_node.body:
                ren.update({d: f"{d}{sfx}" for d in s.defines()})
                writes.append(s.rewrite(lambda nm, ren=ren: ren.get(nm, nm)))
            local = BinaryExpr("+", BinaryExpr("*", Var(m.unit), Literal(m.reg * atom.atom_m, "int")), Literal(i * atom.atom_m, "int"))
            # The layout's in-tile row offset, over ``_lane`` — this publish is a plain ``Write``,
            # not a fragment store, so the ``lane_decl`` locals its ``Expr``\ s name are not in scope.
            local = BinaryExpr("+", local, lay.row_off[r].substitute({"_g": _lane_group(lay)}))
            writes += [Write(output=_stat_slab(nm), index=(local,), value=ren.get(nm, nm)) for nm in stats]
    out.append(
        Cond(cond=BinaryExpr("==", BinaryExpr("%", Var("_lane"), Literal(lay.reduce_group, "int")), Literal(0, "int")), body=tuple(writes))
    )
    out.append(Sync())
    return out


def _sync_operands(
    c: Fold,
    bk_elems: int,
    mn: tuple[Side, Side],
    cta: CtaTile,
    swizzles: tuple[str, str] = ("NONE", "NONE"),
    channels=(),
    seam: tuple = ((), (), ()),
    atom=None,
    lead: tuple = (),
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
    K seam (``ops.cone_seam`` reads the cone NODE's boundary; the scheduler sizes the stat rows off
    the same read): the prologue runs ONCE per tile row (:func:`sync_stat_fill`, returned as the transport
    prologue) and the per-cell fill reads the bridged values back from the stat smem rows. The
    schedule's eligibility guarantees exact cover on N; a masked / symbolic **M**
    clamp-reads the overhanging rows in-bounds (the A fill σ and the stat prologue σ — a duplicate
    of the last valid row is computed and its store discarded by the ``RegStore`` guard, the same
    contract the copy transports follow). A symbolic **K** is the same discipline applied to the
    contraction axis: :func:`_k_masked` clamps the cone's own reads and stores the fold identity
    into every slab lane past the runtime extent, so the drain still reads the whole chunk."""
    m_name, n_name, k_name = mn[0].axis.name, mn[1].axis.name, c.axis.name
    row_base, col_base = _tile_base(mn)
    pro, cell, stats = seam
    k_ext = c.axis.extent_expr() if not c.axis.extent.is_static else None

    def m_coord(row) -> Expr:
        t = BinaryExpr("+", row_base, row)
        return clamp_last(t, mn[0].ext) if mn[0].mask else t

    def k_coord(k) -> Expr:
        return clamp_last(k, k_ext) if k_ext is not None else k

    def a_value(k0, row, col):
        k = BinaryExpr("+", k0, col)
        sigma = Sigma({m_name: m_coord(row), k_name: k_coord(k)})
        stmts: list[Stmt] = [Load(names=(nm,), input=_stat_slab(nm), index=(row,)) for nm in stats]
        stmts += [s.rewrite(lambda nm: nm, sigma) for s in cell]
        return _k_masked(stmts, operand_name(c.a), k, k_ext)

    def n_coord(col) -> Expr:
        t = BinaryExpr("+", col_base, col)
        return clamp_last(t, mn[1].ext) if mn[1].mask else t

    prologue: list[Stmt] = []
    if stats:
        # The CHAINED statistic first — the same score the weight fill mma's, swept at fragment
        # residence, so the prologue is not the one half of attention left at scalar residence.
        prologue = chain_stat_fill(c=c, mn=mn, atom=atom, cols=bk_elems, stats=stats, lead=lead) if isinstance(atom, AtomKind) else None
    if stats and prologue is None:
        row_axis = Axis(name="_sr", extent=mn[0].tile)
        sigma = Sigma({m_name: m_coord(Var(row_axis.name))})
        row_body = [s.rewrite(lambda nm: nm, sigma) for s in pro]
        prologue = sync_stat_fill(
            stats=stats, slab_of=_stat_slab, row_axis=row_axis, row_body=row_body, cta=cta, stat=Reduction.of_cone_stat(c.a)
        )
    prologue = prologue or []
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

    if isinstance(c.a, Load):
        a_op = Operand(
            tag="a",
            buf=c.a.input,
            shape=(mn[0].tile, bk_elems),
            coords=_box_origin(c.a.index, tile=mn[0], tile_base=row_base, k_axis=c.axis, sibling=mn[1]),
            index=_slab_index(c.a.index, tile=mn[0], tile_base=row_base, k_axis=c.axis, tile_is_row=True, sibling=mn[1]),
            swizzle=swizzles[0],
        )
        async_ops.append(a_op)
    else:
        # The CHAINED fill where the cone composes a score CONTRACTION and the atom can mma it: the
        # slab comes from one nested contraction (fragments → the cone's epilogue → the slab), not
        # per-cell scalar code. NONE-swizzle by construction — the fragment store applies no XOR, so
        # the drain must read the plain row-major slab.
        score = chain_edge(c.a, c.axis.name) if isinstance(atom, AtomKind) else None
        if score is not None:
            chain = _chain_of(c=c, score=score, mn=mn, atom=atom, bk_elems=bk_elems, slab="_a_smem", stats=stats, lead=lead)
            a_op = SyncOperand(tag="a", shape=(mn[0].tile, bk_elems), value=a_value, chain=chain, swizzle="NONE")
        else:
            a_op = SyncOperand(tag="a", shape=(mn[0].tile, bk_elems), value=a_value, swizzle=swizzles[0])
        sync_ops.append(a_op)
    drain.append(a_op)

    for f, (bl, _) in enumerate(channels):
        tag = "b" if f == 0 else f"b_x{f}"
        if not isinstance(bl, Load):
            b_body = operand_body(bl)

            def b_value(k0, row, col, *, body=b_body, edge=bl):
                k = BinaryExpr("+", k0, row)
                sigma = Sigma({k_name: k_coord(k), n_name: n_coord(col)})
                return _k_masked([s.rewrite(lambda nm: nm, sigma) for s in body], operand_name(edge), k, k_ext)

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
    if stage.transport == "smem":
        # The synchronous fill: every inline edge is evaluated into its canonical slab (converting
        # on the store when dtypes differ); every materialized counterpart is COPIED underneath
        # that work — with ``cp.async``, or with the blocking vector copy on an atom whose target
        # has none. A term with no inline edge at all lands here too: then it is only the copy.
        operands, sync_ops, copy_ops, stat_pro = _sync_operands(
            c, stage.bk_elems, mn, cta, ops.slab_swizzles(mn, elem.nbytes), ops.channels, ops.cone, tile.atom, ops.lead
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

    return staged_kloop(
        transport=transport,
        drain=drain,
        depth=stage.depth,
        bk_elems=bk,
        n_chunks=n_chunks,
        k_extent=K,
        workers=ops.workers,
        block_threads=tile.launch_threads,
    )


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
    load not referencing the ``m`` cell axis is shared across the ``n`` cells, and vice versa)."""
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
        kept = [s.rewrite(lambda nm: rename.get(nm, nm)) for s in kept]
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


def _scalar_protected(c: Fold, tile: TilePlan, lead: tuple = (), *, body: Body | tuple = ()) -> frozenset[str]:
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
    tile: TilePlan  # the SCHEDULE slice, PLACED (``TilePlan.at``): the atom + the ``(m, n)`` geometry
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
    # (``ops.cone_seam``). ``None`` for a
    # plain gmem-``Load`` A — its whole body is the per-cell fill.
    seam: tuple | None = None
    # The register-fragment NAMESPACE. Empty for the kernel's own contraction; a NESTED one (the
    # chained A fill's score, emitted inside the enclosing K-loop) prefixes its fragments so its
    # ``_a`` / ``_b`` / ``_c`` do not shadow the accumulators the enclosing drain carries across
    # that same loop.
    frag_ns: str = ""

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
        legality (``_legality.resolve_warp_stage``), so a mismatch that staged would already
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
        instead of ldmatrix, its row pad riding the drain ``ldm``. An f16-accumulate
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
        )
        if _f16acc(self.tile.atom):
            stmts = [*stmts, *_f16acc_promotes(mn[0].reg, mn[1].reg, len(self.channels))]
        return stmts

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
        return tuple(
            "NONE" if e.nbytes == 1 else pick_swizzle_atom(inner, e.nbytes)[1]
            for e, inner in zip(self.slab_elems(), (self.stage.bk_elems, b_inner), strict=True)
        )

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
                promotes = _f16acc_promotes(m.reg, n.reg, 1)
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
        c = self.c
        atom = self.tile.atom
        m, n = mn
        mcell, ncell = offset[0].base(i), offset[1].base(j)
        tail = list(self.epilogue)
        sigma = Sigma({m.axis.name: mcell, n.axis.name: ncell})
        chans = self.channels
        accs = tuple(acc for _, acc in chans)
        frags = (self.frag(f"_c{i}_{j}"), *(_fold_frag(self.frag(f"_c{i}_{j}"), f) for f in range(1, len(chans))))
        writes = [s for s in tail if isinstance(s, Write)]
        if len(chans) > 1 and len(tail) == len(writes) == len(chans) and {w.value for w in writes} == set(accs):
            # RAW per-channel stores — the split-K partial's epilogue is one workspace ``Write``
            # per accumulator, NO ⊗-combine (the finalize applies the projection once after the
            # cross-partition sums): each channel's C fragment stores to its own ws slice.
            by_acc = {w.value: w for w in writes}
            return [
                RegStore(
                    dst_buffer=by_acc[acc].output,
                    dst_index=tuple(sigma.apply(e) for e in by_acc[acc].index),
                    frag=frag,
                    shape=atom.shape,
                    epilogue=None,
                    m_guard=_guard(m, mcell),
                    n_guard=_guard(n, ncell),
                    atomic=by_acc[acc].atomic,
                    fragment_layout=atom.fragment_layout,
                )
                for acc, frag in zip(accs, frags, strict=True)
            ]
        write = next(s for s in writes)
        extra = tuple((acc, _fold_frag(self.frag(f"_c{i}_{j}"), f)) for f, (_, acc) in enumerate(chans[1:], 1))
        epi = _warp_epilogue(tail, c.acc, m.axis.name, n.axis.name, sigma, extra_accs=extra)
        assert len(chans) == 1 or epi is not None, "a fused sibling group's projection must combine the accumulators"
        return [
            RegStore(
                dst_buffer=write.output,
                dst_index=tuple(sigma.apply(e) for e in write.index),
                frag=self.frag(f"_c{i}_{j}"),
                shape=atom.shape,
                epilogue=epi,
                m_guard=_guard(m, mcell),
                n_guard=_guard(n, ncell),
                atomic=write.atomic,
                fragment_layout=atom.fragment_layout,
            )
        ]


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
        gmem ``Load`` — or the computed register-resident body, e.g. flash PV's ``P``), each COL its
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
    tile: TilePlan,
    stage: Stage | None = None,
    inputs=None,
    workers=None,
    epilogue: Body | None = None,
    seam=None,
    lead: tuple = (),
    frag_ns: str = "",
) -> _AtomOps:
    """The **one** atom dispatch — select the codegen strategy off the atom kind. ``c`` is the
    stored algebra, ``tile`` the PLACED schedule slice (``TilePlan.at``) the geometry derives from.

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
        from emmy.compiler.pipeline.passes.lowering.tile._atomize import make_cone  # noqa: PLC0415 — decode-boundary import

        c = Fold.contraction(k_axis=c.axis, a=make_cone([c.a], c.axis.name), channels=c.channels)
    cls = _MmaOps if isinstance(tile.atom, AtomKind) else _ScalarOps
    return cls(c, tile, stage, inputs, workers, lead, Body(()) if epilogue is None else epilogue, seam, frag_ns)


def reduce_codegen(c: Fold, tile: TilePlan, stage: Stage | None = None, inputs=None, workers=None, seam=None, lead: tuple = ()):
    """The reusable, **sink-agnostic** ``(state_decls, reduce_region)`` from the atom strategy — the
    accumulator decls + the contraction K-loop (the ONE :meth:`_AtomOps.reduce` driver: the shared
    :func:`_contract_kloop` spine gmem-direct, the shared :func:`_staged` fill→drain skeleton staged).
    ``stage`` / ``inputs`` bind operand staging (both atoms stage the same smem slab off it, differing
    only in the drain leaf — ``ldmatrix`` vs plain ``Load``); ``workers`` splits the staged phases
    across producer / compute warp bands (the resolved :class:`WarpSpec`; ``None`` = uniform)."""
    ops = _atom_ops(c, tile, stage, inputs, workers, seam=seam, lead=lead)
    return ops.state, ops.reduce


def store_sink(c: Fold, tile: TilePlan, epilogue: Body | None = None, lead: tuple = ()):
    """The default **matmul sink** — the per-cell ``store(i, j, offset, mn)`` from the atom strategy
    (an mma ``RegStore`` / the replicated scalar ``epilogue`` tail), folding in the ``epilogue`` (the
    projection off the node's zero-axis ``Fold`` wrapper + the store glue). ``factorize(c, store=…)`` swaps the
    sink (a flash sink that bridges the accumulator into the streaming-softmax twist), reusing the
    shared ``reduce`` emission."""
    return _atom_ops(c, tile, epilogue=epilogue, lead=lead).store
