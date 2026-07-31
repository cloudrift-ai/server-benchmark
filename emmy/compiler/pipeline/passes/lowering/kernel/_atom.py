r"""The per-atom codegen strategies — the one seam every tiled contraction dispatches through.

``_factor._bind`` (the output-tiled contraction arm) reads the tiling geometry off the :class:`Contraction`
node and asks this module for the two codegen halves: :func:`reduce_codegen` (the sink-agnostic
``(state_decls, reduce_region)`` — the accumulator/operand decls + the shared :func:`_contract_kloop`
K-loop) and :func:`store_sink` (the per-cell matmul sink). Both resolve through :func:`_atom_ops` to
one of the two concrete strategies — :class:`_MmaOps` (tensor-core ``ldmatrix`` + ``mma.sync``, a
``RegStore`` sink) or :class:`_ScalarOps` (plain ``Load``\ s + an ``fma`` cell, the replicated-
``epilogue`` sink). The K-loop itself is ONE driver on the base strategy (:meth:`_AtomOps.reduce`),
deciding nothing: the **scheduler-resolved** ``stage`` (eligibility + sizing ran once in
``020_schedule`` — ``_resolve_warp_stage`` / ``_resolve_scalar_stage``; ``None`` = gmem-direct)
picks its form — gmem-direct through the shared :func:`_contract_kloop` spine, or staged through
the shared :func:`_staged` fill→drain skeleton (over the one ``_stage.staged_kloop``) — and the
atom supplies only descriptor reads: the four gmem leaf constructors (:meth:`gmem_leaves`), the
slab drain leaf (:meth:`staged_drain`), and the slab element dtype. This IS the "atom as
descriptor" seam: one factory, one loop over atoms, no scattered ``isinstance``.

Leading ``_`` so the pass loader (globs ``*.py``, skips ``_``-prefixed) skips it."""

from __future__ import annotations

from dataclasses import dataclass, field

from emmy.compiler.backend.cuda.dtype import cuda_name
from emmy.compiler.dtype import F32
from emmy.compiler.ir.atom import AtomKind
from emmy.compiler.ir.axis import Axis, Window
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal, TernaryExpr, Var
from emmy.compiler.ir.kernel.ir import (
    EpilogueLoad,
    FragmentPromote,
    LdmatrixLoad,
    MmaSyncPtx,
    RegEpilogue,
    RegFragment,
    RegStore,
)
from emmy.compiler.ir.schedule import Stage
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Accum, Assign, Body, Cond, Init, Load, Loop, Select, Stmt, StridedLoop, Write
from emmy.compiler.ir.tile.ir import Side
from emmy.compiler.pipeline.passes.lowering._placed import Placed
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


# Shared axis-geometry helpers, used across this module (the atom-generic mma/scalar codegen) AND
# ``_factor.py`` (the tiling layer + the cooperative / ILP reduce tier).
def shrink_axis(axis: Axis, reg: int) -> Axis:
    """The grid (cell) axis for a register-tiled free axis: ``ceil(E / reg)`` cells, each a
    per-thread ``reg``-wide register sub-tile. ``Dim.ceil_div`` keeps a symbolic extent
    symbolic (``(seq_len+reg-1)//reg``) so the launch grid sizes from the runtime extent."""
    if reg <= 1:
        return axis
    return Axis(name=axis.name, extent=axis.extent.ceil_div(reg), window=Window(parent=axis.source_axis or axis))


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
    """Fold the projection ``Map`` into a :class:`RegEpilogue` for cell ``sigma``. ``None`` when
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
# The warp tier's smem operand pipeline, driven off the node's :class:`Stage`. cp.async and TMA
# share the slab + the staged-``LdmatrixLoad`` drain (:func:`_staged_inner_atom_loop`) AND the
# slab swizzle modes (:meth:`_MmaOps.slab_swizzles`); only who applies the fill-side permutation
# differs — TMA swizzles in hardware during the box copy, a cp.async fill XORs its destination
# index in software — and the drain XOR undoes either. Staging is a **pure perf
# transform**: an ineligible kernel silently falls back to gmem-direct, and a staged kernel is
# bit-identical to its gmem-direct baseline. The transport primitives (the fill loops + the
# commit/wait / mbarrier handshakes) live in ``_stage.py``; these functions schedule them onto the
# K-loop off the :class:`Contraction` geometry.
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
    *, slabs: tuple[str, ...], mn: tuple[Side, Side], atom, bk_elems, ki, reg_depth: int = 1, offs=None, swizzles=None, trans=None
) -> list[Stmt]:
    """The inner atom-K drain shared by the cp.async and TMA staged paths: read the A/B ``slabs`` via
    ``LdmatrixLoad(staged=True)`` + ``MmaSyncPtx``. ``slabs`` is ``(A, B…)`` — one B slab per fold
    channel (one for the ordinary matmul; N for the multi-B gate/up node, whose ONE ldmatrix'd A
    fragment feeds a per-channel mma chain into a per-channel C fragment). Slab-local indices —
    A[tile_m][bk_elems] (ldm=bk_elems), B[bk_elems][tile_n] (ldm=tile_n) — independent of which
    producer filled the (plain row-major, NONE-swizzle) slab; ``mn`` is the ``(m, n)`` :class:`Side`
    pair.

    ``reg_depth == 1`` (default): one ``StridedLoop`` over the ``bk`` atom-K steps, ldmatrix-then-mma
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
    ``LdmatrixLoad(b_trans=True)`` staged path. Always ``False`` for A."""
    (a_slab, *b_slabs), (m, n) = slabs, mn
    offs = offs if offs is not None else (None,) * len(slabs)
    swizzles = swizzles if swizzles is not None else ("NONE",) * len(slabs)
    trans = trans if trans is not None else (False,) * len(slabs)
    atom_m, atom_n, atom_k = atom.shape
    n_steps = bk_elems // atom_k
    # Per-operand drain spec: (frag base fn, slab, ldm, tile-is-slab-row, reg count, warp-unit var,
    # atom dim, slot row off, swizzle). A stacks the tile axis on the slab row (K the col); B swaps
    # (K the row, tile the col) — unless its slab is transposed (N-major: tile the row, K the col,
    # like A); the slot offset always lands on the ROW. All share ONE emission loop.
    specs = [(lambda x: f"_a{x}", "a", a_slab, bk_elems, True, m.reg, m.unit, atom_m, offs[0], swizzles[0])]
    for f, bs in enumerate(b_slabs):
        frag_of = (lambda ff: lambda x: _fold_frag(f"_b{x}", ff))(f)
        tr = trans[1 + f]
        specs.append((frag_of, "b", bs, bk_elems if tr else n.tile, tr, n.reg, n.unit, atom_n, offs[1 + f], swizzles[1 + f]))

    def ldms(kexpr, suffix):  # every operand's ldmatrix reads at K position `kexpr`, into fragment slot `suffix`
        reads: list[Stmt] = []
        for frag_of, role, slab, ldm, is_row, reg, unit, adim, off, swz in specs:
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
                    )
                )
        return reads

    def mmas(suffix):  # every fold channel × (i, j) cell's mma.sync over the `suffix`-slotted operand fragments
        return [
            MmaSyncPtx(
                c_frag=_fold_frag(_mma_c_base(atom, i, j), f),
                a_frag=f"_a{i}{suffix}",
                b_frag=f"{_fold_frag(f'_b{j}', f)}{suffix}",
                shape=atom.shape,
                ab_dtype=atom.ab_dtype,
                c_dtype=atom.operand_dtype("c").name,
            )
            for f in range(len(b_slabs))
            for i in range(m.reg)
            for j in range(n.reg)
        ]

    if reg_depth < 2 or n_steps < 2:  # single-buffer: the inline ldmatrix→mma loop
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


def _clamp_last(idx: Expr, ext: Expr) -> Expr:
    """Clamp an overhanging gmem coordinate to the last valid index — the overhanging cell still
    reads an in-bounds (duplicate) operand, and its store is discarded by the guard (``RegStore`` /
    ``Cond``)."""
    return TernaryExpr(cond=BinaryExpr("<", idx, ext), if_true=idx, if_false=BinaryExpr("-", ext, Literal(1, "int")))


def _slab_index(operand_index, *, tile: Side, tile_base, k_axis, tile_is_row: bool):
    """The **one** cp.async slab gmem-index factory, for either operand and either tier. The slab's
    inner (contiguous) dim maps to the contraction ``k_axis``, its outer dim to the stationary ``tile``
    axis (``m`` for A, ``n`` for B). For A the tile axis is the slab ROW (K the col); for B they swap
    (``slot[row][col] = A[row_base + row][k0 + col]`` / ``B[k0 + row][col_base + col]``). A masked tile
    coordinate is clamped in-bounds — the overhanging cell reads a duplicate and its store is guarded.
    Returns a ``k0 -> ((row, col) -> gmem index)`` map — one K-chunk offset per :func:`staged_kloop`
    fill."""

    def at(k0):
        def gmem(row, col):
            tc, kc = (row, col) if tile_is_row else (col, row)
            t = BinaryExpr("+", tile_base, tc)
            sig = Sigma({tile.axis.name: _clamp_last(t, tile.ext) if tile.mask else t, k_axis.name: BinaryExpr("+", k0, kc)})
            return tuple(sig.apply(e) for e in operand_index)

        return gmem

    return at


def _tile_base(mn: tuple[Side, Side]) -> tuple[Expr, Expr]:
    """The CTA tile's ``(row_base, col_base)`` top-left origin — ``(m_b·tile_m, n_b·tile_n)``."""
    return tuple(BinaryExpr("*", Var(s.block), Literal(s.tile, "int")) for s in mn)


def _box_origin(operand_index, *, tile: Side, tile_base: Expr, k_axis):
    """The TMA box origin at K-chunk ``k0`` — the operand's OWN gmem index evaluated (σ) at the
    tile base and ``k0``, so an offset operand (a split-K partial's ``ksplit·(K/w) + k``) lands
    the box at its absolute coordinates. For a canonical operand this is exactly ``(tile_base,
    k0)`` (A, tile axis the slab row) / ``(k0, tile_base)`` (B)."""

    def at(k0):
        sig = Sigma({tile.axis.name: tile_base, k_axis.name: k0})
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
    and fill by its OWN element width."""
    ops: list[Operand] = []
    for i, (tag, is_row) in enumerate((("a", True), ("b", b_trans))):
        tile, tile_base = mn[i], base[i]
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
                coords=_box_origin(index_srcs[i], tile=tile, tile_base=tile_base, k_axis=k_axis),
                index=_slab_index(index_srcs[i], tile=tile, tile_base=tile_base, k_axis=k_axis, tile_is_row=is_row),
                swizzle=swizzles[i],
                dtype=cuda_name(elem) if elem is not None else None,
                elem_bytes=elem.nbytes if elem is not None else None,
                trans=i == 1 and b_trans,
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


def _sync_operands(
    c: Placed,
    bk_elems: int,
    mn: tuple[Side, Side],
    cta: CtaTile,
    swizzles: tuple[str, str] = ("NONE", "NONE"),
    channels=(),
    seam: tuple = ((), (), ()),
) -> tuple[tuple, tuple[SyncOperand, ...], tuple[Operand, ...], list[Stmt]]:
    """The ``sync``-transport (fused-edge) operands + the one-shot prologue stmts, returned as
    ``(drain-ordered operands, compute-filled, cp.async-filled, prologue)``: A is **compute-filled**
    from the node's producer cone (``a`` is a ``Body`` — each thread evaluates the cone at
    the slab cell's absolute ``(m, k)`` coords and writes the result); each fold channel's B is a
    plain weight copy riding a vectorized ``cp.async`` :class:`Operand` that flies UNDER the
    compute fill — a canonical B as the K-major ``(bk × tile_n)`` slab, a transposed B
    (the serving ``F.linear`` layout, K gmem-contiguous) as the N-MAJOR ``(tile_n × bk)`` slab in
    its own gmem orientation (A's geometry; the ``Operand.trans`` stamp routes the drain to the
    plain no-``.trans`` ldmatrix — the same slab the copy transports stage). A cone with a
    row-invariant prologue (the fused
    norm→linear per-row statistic — its reduce ``Loop`` + scalar sweep) arrives already split at the
    K seam (``ops.cone_seam`` reads the cone NODE's boundary; the scheduler sizes the stat rows off
    the same read): the prologue runs ONCE per tile row (:func:`sync_stat_fill`, returned as the transport
    prologue) and the per-cell fill reads the bridged values back from the stat smem rows. The
    schedule's eligibility guarantees exact cover on N and K only; a masked / symbolic **M**
    clamp-reads the overhanging rows in-bounds (the A fill σ and the stat prologue σ — a duplicate
    of the last valid row is computed and its store discarded by the ``RegStore`` guard, the same
    contract the copy transports follow)."""
    m_name, k_name = c.m_axis.name, c.k_axis.name
    row_base, col_base = _tile_base(mn)
    pro, cell, stats = seam

    def m_coord(row) -> Expr:
        t = BinaryExpr("+", row_base, row)
        return _clamp_last(t, mn[0].ext) if mn[0].mask else t

    def a_value(k0, row, col):
        sigma = Sigma({m_name: m_coord(row), k_name: BinaryExpr("+", k0, col)})
        stmts: list[Stmt] = [Load(names=(nm,), input=_stat_slab(nm), index=(row,)) for nm in stats]
        stmts += [s.rewrite(lambda nm: nm, sigma) for s in cell]
        return stmts, c.a_name

    prologue: list[Stmt] = []
    if stats:
        row_axis = Axis(name="_sr", extent=mn[0].tile)
        sigma = Sigma({m_name: m_coord(Var(row_axis.name))})
        row_body = [s.rewrite(lambda nm: nm, sigma) for s in pro]
        prologue = sync_stat_fill(
            stats=stats, slab_of=_stat_slab, row_axis=row_axis, row_body=row_body, cta=cta, stat=Reduction.of_cone_stat(c.a)
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
    a_op = SyncOperand(tag="a", shape=(mn[0].tile, bk_elems), value=a_value, swizzle=swizzles[0])
    drain: list = [a_op]
    sync_ops: list[SyncOperand] = [a_op]
    async_ops: list[Operand] = []
    for f, (bl, _) in enumerate(channels):
        tag = "b" if f == 0 else f"b_x{f}"
        # A transposed B stages N-major (``tile_n × bk`` — its own gmem orientation, K stride-1 in
        # gmem and smem alike), so its cp.async chunks are contiguous exactly like the canonical
        # K-major slab's (row-base alignment holds: B's row stride K is a multiple of ``bk_elems``).
        shape = (mn[1].tile, bk_elems) if c.b_trans else (bk_elems, mn[1].tile)
        op = Operand(
            tag=tag,
            buf=bl.input,
            shape=shape,
            coords=_box_origin(bl.index, tile=mn[1], tile_base=col_base, k_axis=c.k_axis),
            index=_slab_index(bl.index, tile=mn[1], tile_base=col_base, k_axis=c.k_axis, tile_is_row=c.b_trans),
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
    — ``ldmatrix`` + ``mma.sync`` vs plain-``Load`` fma) and the slab element dtype. ``ops.stage``
    is the **scheduler-RESOLVED** stage (``_schedule._resolve_warp_stage`` / ``_resolve_scalar_stage``
    ran eligibility + sizing once) — its ``transport`` / ``bk_elems`` / ``depth`` / ``reg_depth``
    are applied verbatim, no decision here. A pure perf transform, numerically identical to
    gmem-direct (mma: bit-identical). ``depth == 1`` is the single-buffer degenerate; ``depth >= 2``
    the gmem→smem ring; ``reg_depth`` composes the mma inner smem→register double-buffer. A masked
    M / N overhang is fill-side clamped (cp.async) or box zero-filled (TMA); the discard stays with
    the store guard."""
    c, stage = ops.c, ops.stage
    k_axis = c.k_axis
    K = k_axis.extent.as_static()  # static K (the resolution eligibility rule)
    elem = ops.slab_elem()
    cta = _cta(mn, c.atom.lanes, c.block_threads)
    if stage.transport == "sync":
        # The fused-edge compute-fill: the A tile is COMPUTED into its slab (the producer cone);
        # every B (canonical K-major, or transposed staged N-major) rides a vectorized cp.async
        # that flies UNDER the compute fill — the mma
        # tier's ``sync`` transport; single-buffer, one wait + CTA barrier. A k-invariant cone
        # prefix (the fused norm→linear per-row statistic) rides the transport prologue, run once
        # ahead of the K-loop.
        operands, sync_ops, async_ops, stat_pro = _sync_operands(
            c, stage.bk_elems, mn, cta, ops.slab_swizzles(mn, elem.nbytes), ops.channels, ops.cone
        )
        transport = SyncTransport(
            operands=sync_ops,
            async_operands=async_ops,
            slab_dtype=cuda_name(elem),
            elem_bytes=elem.nbytes,
            cta=cta,
            prologue_stmts=tuple(stat_pro),
        )
    else:
        assert len(ops.channels) == 1, "cp.async / TMA staging is single-fold — a multi-B node rides the sync compute-fill"
        operands = _slab_operands(
            index_srcs=(c.a.index, c.b.index),
            bufs=(c.a.input, c.b.input),
            mn=mn,
            k_axis=k_axis,
            bk_elems=stage.bk_elems,
            base=_tile_base(mn),
            swizzles=ops.slab_swizzles(mn, elem.nbytes),
            elems=ops.slab_elems(),
            b_trans=c.b_trans,
        )
        common = dict(
            operands=operands,
            slab_dtype=cuda_name(elem),
            elem_bytes=elem.nbytes,
            cta=cta,
        )
        transport = TmaTransport(**common) if stage.transport == "tma" else CpAsyncTransport(**common)

    def drain(slot):  # the atom's slab-reading leaf, over ring `slot`
        return ops.staged_drain(operands, slot, cells, offset, mn)

    return staged_kloop(
        transport=transport,
        drain=drain,
        depth=stage.depth,
        bk_elems=stage.bk_elems,
        n_chunks=K // stage.bk_elems,
        k_extent=K,
        workers=ops.workers,
        block_threads=c.block_threads,
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
    clamp-read operands) but only stores when in bounds. Non-``Write`` stmts pass through."""
    if cond is None:
        return stmts
    return [Cond(cond=cond, body=(s,)) if isinstance(s, Write) else s for s in stmts]


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


def _scalar_protected(c: Placed) -> frozenset[str]:
    """The shared iteration coordinates — the block / unit / loop / extent vars excluded from the
    per-cell SSA rename (everything else is suffixed ``__c{i}_{j}`` so each cell owns its names)."""
    m, n, k_axis = c.m, c.n, c.k_axis
    prot = {k_axis.name}
    for s in (m, n):
        prot |= {s.block, s.unit}
    for a in c.lead_axes:
        prot.add(a.name)
    for a in (m.axis, n.axis, k_axis, *c.lead_axes):
        prot |= set(a.extent_expr().free_vars())
    return frozenset(prot)


def _scalar_drain(
    c: Placed, cells, offset, slabs: tuple[str, str], ki: str, bk_elems: int, base: tuple[Expr, Expr], offs=(None, None)
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
    b_name, a_name = c.b_name, c.a_name
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
    gmem-direct; eligibility + sizing already ran in ``020_schedule``) and ``inputs``, it supplies
    the three ``grid_tile`` callables — ``state(cells)`` (accumulator decls), :meth:`reduce` (the
    K-loop — **shared on this base**, one loop over atoms), ``store(i, j, offset, mn)`` (the
    per-cell sink; ``mn`` is the contraction's ``(m, n)`` :class:`Side` pair). The two concrete
    atoms (:class:`_MmaOps` / :class:`_ScalarOps`) supply only descriptor reads:
    :meth:`gmem_leaves` (the four gmem-direct leaf constructors), :meth:`staged_drain` (the
    slab-reading leaf) and :meth:`slab_elem` (the slab element dtype). This IS the "atom as
    descriptor" seam: one factory (:func:`_atom_ops`), no scattered ``isinstance``."""

    c: Placed
    stage: Stage | None = None
    inputs: object = None
    workers: object = None  # the resolved WarpSpec (None = uniform SIMT) — consumed by _staged
    # The projection this node's store folds in — the wrapping ``Map``'s body plus the grid-``Write``
    # glue, assembled by ``_factor._bind``. It is NOT a node field: every projection has ONE home,
    # the ``Map`` wrapper, and the store sink is where it lands.
    epilogue: Body = field(default_factory=Body)
    # The computed-A cone's ``(prologue, cell, stats)`` K seam, read off the NODE BOUNDARY
    # (``ops.cone_seam``). ``None`` for a
    # plain gmem-``Load`` A — its whole body is the per-cell fill.
    seam: tuple | None = None

    @property
    def channels(self) -> tuple:
        """The ``(b, acc)`` pairs this emission folds — the node's product channels (one A
        fragment, N mma chains, one C fragment per channel at arity N)."""
        return tuple((ch.b, ch.acc) for ch in self.c.channels)

    @property
    def cone(self) -> tuple:
        """The A cone's ``(row-invariant prologue, per-cell body, bridged stats)`` — the node
        boundary, or the whole operand body when there is no cone to split."""
        return self.seam if self.seam is not None else ((), tuple(self.c.a_body), ())

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
    """Tensor-core atom — ``ldmatrix`` fragment reads + ``mma.sync``, a ``RegStore`` sink."""

    def slab_elem(self):
        """The slab element dtype — the mma A/B operand dtype (f16/bf16 fragments)."""
        return self.c.atom.operand_dtype("a")

    def staged_drain(self, operands, slot, cells, offset, mn):
        """The mma slab drain — the ``ldmatrix`` + ``mma.sync`` leaf reading ring ``slot``
        (:func:`_staged_inner_atom_loop`; the cells ride ``mn``'s reg counts, so ``cells`` /
        ``offset`` are unused here). Each slab's swizzle mode rides its operand (``NONE`` on the
        sync transport's :class:`SyncOperand`, which has no swizzle field). An f16-accumulate
        atom promote-folds its packed f16 fragments into the f32 shadows once per drain — the
        bk chunk IS the promote cadence (the last chunk's fold doubles as the final one)."""
        stmts = _staged_inner_atom_loop(
            slabs=tuple(op.slab for op in operands),
            offs=tuple(op.slot_row(slot) for op in operands),
            mn=mn,
            atom=self.c.atom,
            bk_elems=self.stage.bk_elems,
            ki="_ki",
            reg_depth=self.stage.reg_depth,
            swizzles=tuple(getattr(op, "swizzle", "NONE") for op in operands),
            trans=tuple(getattr(op, "trans", False) for op in operands),
        )
        if _f16acc(self.c.atom):
            stmts = [*stmts, *_f16acc_promotes(mn[0].reg, mn[1].reg, len(self.channels))]
        return stmts

    def slab_swizzles(self, mn, elem_bytes: int) -> tuple[str, str]:
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
        so its inner row span is the K chunk (``bk_elems``) like A's."""
        b_inner = self.stage.bk_elems if self.c.b_trans else mn[1].tile
        return (
            pick_swizzle_atom(self.stage.bk_elems, elem_bytes)[1],
            pick_swizzle_atom(b_inner, elem_bytes)[1],
        )

    def state(self, cells):
        """The mma operand/accumulator register fragments — one ``_a``/``_b`` per register row/col and
        one ``_c`` accumulator per cell (held across the K-loop). A staged ``reg_depth >= 2`` slots the
        operand fragments (``_a{i}_s{slot}``) for the smem→register double-buffer's ping-pong."""
        c = self.c
        atom, m, n = c.atom, c.m, c.n
        reg_depth = self.stage.reg_depth if self.stage is not None else 1

        def frags(base_of, reg):  # reg-tile operand fragment names (slotted ``_s{s}`` when double-buffered)
            names = [base_of(i) for i in range(reg)]
            return [f"{nm}_s{s}" for nm in names for s in range(reg_depth)] if reg_depth >= 2 else names

        # One A fragment set; one B and one C fragment set PER fold channel (the multi-B node's
        # shared-A / per-channel-accumulate drain).
        n_folds = len(self.channels)
        decls: list[Stmt] = [
            RegFragment(name=nm, role="a", shape=atom.shape, dtype=atom.operand_dtype("a")) for nm in frags(lambda i: f"_a{i}", m.reg)
        ]
        for f in range(n_folds):
            decls += [
                RegFragment(name=nm, role="b", shape=atom.shape, dtype=atom.operand_dtype("b"))
                for nm in frags(lambda i, ff=f: _fold_frag(f"_b{i}", ff), n.reg)
            ]
        decls += [
            RegFragment(name=_fold_frag(_mma_c_base(atom, i, j), f), role="c", shape=atom.shape, dtype=atom.operand_dtype("c"))
            for f in range(n_folds)
            for i in range(m.reg)
            for j in range(n.reg)
        ]
        if _f16acc(atom):
            # The f32 shadow accumulators — they keep the ``_c{i}_{j}`` names the store reads;
            # the packed f16 mma targets above are the ``_ch{i}_{j}`` family FragmentPromote folds.
            decls += [
                RegFragment(name=_fold_frag(f"_c{i}_{j}", f), role="c", shape=atom.shape, dtype=F32)
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
        atom, (m, n) = c.atom, mn
        k_axis = c.k_axis
        assert not c.a_computed, (
            "mma matmul arm: a register-resident (computed) A operand lowers through the fragment realizer (_twist), not here"
        )
        assert len(self.channels) == 1, "gmem-direct mma is single-fold — a multi-B node rides the sync compute-fill"
        a_load, b_load, b_trans = c.a, c.b, c.b_trans
        k_static = k_axis.extent.is_static
        k_zero = None if k_static else (Var(k_axis.name), k_axis.extent_expr())

        def read_row(i):
            cell = offset[0].base(i)
            idx = tuple(Sigma({m.axis.name: cell}).apply(e) for e in a_load.index)
            return [
                LdmatrixLoad(
                    frag=f"_a{i}", src_buffer=a_load.input, src_index=idx, role="a", staged=False, gmem_guard=_guard(m, cell), k_zero=k_zero
                )
            ]

        def read_col(j):
            cell = offset[1].base(j)
            idx = tuple(Sigma({n.axis.name: cell}).apply(e) for e in b_load.index)
            return [
                LdmatrixLoad(
                    frag=f"_b{j}",
                    src_buffer=b_load.input,
                    src_index=idx,
                    role="b",
                    staged=False,
                    b_trans=b_trans,
                    gmem_guard=_guard(n, cell),
                    k_zero=k_zero,
                )
            ]

        def contract(i, j):
            return [
                MmaSyncPtx(
                    c_frag=_mma_c_base(atom, i, j),
                    a_frag=f"_a{i}",
                    b_frag=f"_b{j}",
                    shape=atom.shape,
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
        atom = c.atom
        m, n = mn
        mcell, ncell = offset[0].base(i), offset[1].base(j)
        tail = list(self.epilogue)
        sigma = Sigma({m.axis.name: mcell, n.axis.name: ncell})
        chans = self.channels
        accs = tuple(acc for _, acc in chans)
        frags = (f"_c{i}_{j}", *(_fold_frag(f"_c{i}_{j}", f) for f in range(1, len(chans))))
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
                )
                for acc, frag in zip(accs, frags, strict=True)
            ]
        write = next(s for s in writes)
        extra = tuple((acc, _fold_frag(f"_c{i}_{j}", f)) for f, (_, acc) in enumerate(chans[1:], 1))
        epi = _warp_epilogue(tail, c.acc, m.axis.name, n.axis.name, sigma, extra_accs=extra)
        assert len(chans) == 1 or epi is not None, "a fused sibling group's projection must combine the accumulators"
        return [
            RegStore(
                dst_buffer=write.output,
                dst_index=tuple(sigma.apply(e) for e in write.index),
                frag=f"_c{i}_{j}",
                shape=atom.shape,
                epilogue=epi,
                m_guard=_guard(m, mcell),
                n_guard=_guard(n, ncell),
                atomic=write.atomic,
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
        its read in-bounds (``% extent``) and the overhanging store is guarded (:meth:`store`)."""
        c = self.c
        assert len(self.channels) == 1, "the scalar tier is single-fold — a multi-B node rides the warp sync compute-fill"
        k_axis = c.k_axis
        m, n = mn
        prot = _scalar_protected(c)
        b_name, a_name = c.b_name, c.a_name

        def read_row(i):
            if m is None:
                return copy_cell(c.a_body, Sigma({}), f"__ar{i}", prot)
            return copy_cell(c.a_body, Sigma({m.name: _wrap(m, offset[0].base(i))}), f"__ar{i}", prot)

        def read_col(j):
            return copy_cell(c.b_body, Sigma({n.name: _wrap(n, offset[1].base(j))}), f"__bc{j}", prot)

        def contract(i, j):
            v = f"{c.acc}__v__c{i}_{j}"
            return [
                Assign(name=v, op=_MUL, args=(f"{b_name}__bc{j}", f"{a_name}__ar{i}")),
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
        cell = copy_cell(self.epilogue, sigma, f"__c{i}_{j}", _scalar_protected(c))
        cell = _guard_writes(cell, _scalar_bound(mn, offset, i, j))
        return _dedup_loads(cell)


def _atom_ops(c: Placed, stage: Stage | None = None, inputs=None, workers=None, epilogue: Body | None = None, seam=None) -> _AtomOps:
    """The **one** atom dispatch — select the codegen strategy off the atom kind."""
    cls = _MmaOps if isinstance(c.atom, AtomKind) else _ScalarOps
    return cls(c, stage, inputs, workers, Body(()) if epilogue is None else epilogue, seam)


def reduce_codegen(c: Placed, stage: Stage | None = None, inputs=None, workers=None, seam=None):
    """The reusable, **sink-agnostic** ``(state_decls, reduce_region)`` from the atom strategy — the
    accumulator decls + the contraction K-loop (the ONE :meth:`_AtomOps.reduce` driver: the shared
    :func:`_contract_kloop` spine gmem-direct, the shared :func:`_staged` fill→drain skeleton staged).
    ``stage`` / ``inputs`` bind operand staging (both atoms stage the same smem slab off it, differing
    only in the drain leaf — ``ldmatrix`` vs plain ``Load``); ``workers`` splits the staged phases
    across producer / compute warp bands (the resolved :class:`WarpSpec`; ``None`` = uniform)."""
    ops = _atom_ops(c, stage, inputs, workers, seam=seam)
    return ops.state, ops.reduce


def store_sink(c: Placed, epilogue: Body | None = None):
    """The default **matmul sink** — the per-cell ``store(i, j, offset, mn)`` from the atom strategy
    (an mma ``RegStore`` / the replicated scalar ``epilogue`` tail), folding in the ``epilogue`` (the
    projection off the node's ``Map`` wrapper + the store glue). ``factorize(c, store=…)`` swaps the
    sink (a flash sink that bridges the accumulator into the streaming-softmax twist), reusing the
    shared ``reduce`` emission."""
    return _atom_ops(c, epilogue=epilogue).store
