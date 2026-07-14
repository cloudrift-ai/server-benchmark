"""The fragment realizer — a TWISTED carrier realized at WARP-FRAGMENT residence.

The one ``_bind`` pipeline calls :func:`realize_warp_twist` when the reduce arm's tree carries a
warp tile (:func:`warp_source` — the stream's head :class:`Contraction` stamped with an mma
:class:`TilePlan` by ``_schedule`` (inside ``010_recognize``)): the streaming reduce keeps its per-step values in mma
C-fragments instead of scalars, so every piece of the scalar lowering is *realized* at the new
residence — the fragment row of the placement-keyed fold (a within-warp ``FragmentRowReduce``
``__shfl`` move where the scalar tier folds in-thread):

- the head contraction (Q@K) emits ``ldmatrix`` + ``mma.sync`` into score C-fragments off its node
  geometry (:func:`_frag_contraction` — operands / ``b_trans`` / guards from the node, the atom
  counts from its stamped ``TilePlan``);
- the score prologue (the scale ``Assign``, the causal ``Select``) is realized stmt-by-stmt
  (:func:`_realize_prologue`): a pointwise ``Assign`` → :class:`FragmentApply`, a
  coordinate-predicated ``Select`` → :class:`FragmentMask` (the keep-predicate negated), a
  loop-invariant scalar ``Load`` hoisted above the stream;
- the twisted carrier's streaming merge is regenerated FROM ITS CHANNEL SPEC (``twist.family ==
  "exp"``, ``channels = (pivot, …)``): the pivot's per-block fold is a ``FragmentRowReduce`` of its
  ``fold`` op (rowmax) + the running-stat update / rescale; a ``denom`` channel (no lift)
  row-reduces the exp-weight fragments (rowsum) into ``l = l·α + Σp``; an ``expect`` channel
  (``lift = multiply``) IS the P@V :class:`Contraction` node — its ⊗ lowers to ``mma.sync`` with
  the register-resident probability converted straight into its A-operand fragments by the C→A
  REGISTER repack (``FragmentRepack``, the ``AtomKind.c_to_a_repack`` lane-map compatibility —
  no smem round-trip, no sync; gated at schedule time);
- the projection tail (``O / l``) realizes as an in-place ``FragmentApply`` + the ``RegStore``
  output close;
- a resolved K/V ``Stage`` on the ``TileOp`` (``ctx.stage`` — ``_schedule._resolve_twisted_stage``,
  cp.async or TMA over a static block-divisible kv) re-parents the streaming step under the same
  ``staged_kloop`` skeleton the matmul tier runs: the K/V slabs fill per KV block (each in its
  operand's own layout — verbatim row copies, so staged stays bit-identical to gmem-direct) and
  the step's B fragments drain them via the staged ldmatrix variants (plain ``x2`` for the
  N-major K slab, ``x2.trans`` for the K-major V slab). cp.async fills pad the rows +16 B; TMA
  box-copies the batched operands via rank-N descriptors into dense hardware-swizzled slabs, and
  a stamped ``WSPEC`` split rides ``_wspec_kloop`` with the wrapped elected fill tid.

Nothing here keys on a kernel *identity* — the walk reads node structure, channel algebra, and the
stamped schedule; an unrealizable tree is rejected at schedule time (``_schedule._twisted_warp_
option`` never stamps the warp tile), so a raise here is an invariant breach, not a fallback.
Leading ``_`` so the pass loader skips this module."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from emmy.compiler.dtype import F32
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Builtin, Expr, Literal, TernaryExpr, Var
from emmy.compiler.ir.kernel.ir import (
    FRAG,
    FRAG_COL,
    FRAG_ROW,
    ROW,
    UNIFORM,
    FragmentApply,
    FragmentMask,
    FragmentPromote,
    FragmentRepack,
    FragmentRowReduce,
    LdmatrixLoad,
    MmaSyncPtx,
    Reassign,
    RegFragment,
    RegStore,
)
from emmy.compiler.ir.schedule import Fold, Level, ReduceStage
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Assign, Body, Cond, Init, Load, Select, Stmt, StridedLoop, Write
from emmy.compiler.ir.tile.ir import Contraction, Map, Reduction
from emmy.compiler.pipeline.passes.lowering.kernel._atom import _clamp_last, _f16acc, unroll_ok
from emmy.compiler.pipeline.passes.lowering.kernel._stage import (
    CpAsyncTransport,
    CtaTile,
    Operand,
    TmaTransport,
    cp_async_commit,
    cp_async_fill,
    cp_async_wait,
    pick_swizzle_atom,
    pipelined_kloop,
    slab_smem,
    staged_kloop,
)

_ADD = ElementwiseImpl("add")
_SUB = ElementwiseImpl("subtract")
_MUL = ElementwiseImpl("multiply")
_EXP = ElementwiseImpl("exp")

#: The 2 C-fragment rows per lane the m16n8 stats distribute over (the ``0`` / ``1`` suffixes).
_COMPS = ("0", "1")

#: Slab row padding, in halves (16 B — one ldmatrix row, so alignment holds). A power-of-two row
#: span (the 128 B head_dim/d_v rows) lands every ldmatrix row read on one bank group (8-way
#: replays, the measured 132 M-conflict profile); +16 B shifts consecutive rows across groups.
#: Applied to the cp.async K/V slabs — the cp.async-path counterpart of the TMA slab's hardware
#: swizzle.
_PAD = 8

#: The per-CTA warp index axis (bound by the ``Tile`` decode when the schedule maps more than one
#: warp per CTA — ``qk.tile.units[0] > 1``); each warp owns its own ``atom_m`` query-row block.
FLASH_WARP_AXIS = "_wq"

#: Negation of a comparison — a ``Select`` KEEPS where its predicate holds; a ``FragmentMask``
#: FILLS where its predicate holds, so the keep-predicate flips.
_NEGATE = {"<=": ">", "<": ">=", ">=": "<", ">": "<=", "==": "!="}


def warp_source(op) -> Contraction | None:
    """The warp-tiled stream-head :class:`Contraction` of a ``TWISTED`` :class:`Reduction` tree
    (``op`` bare or under a projecting :class:`Map`), or ``None`` — the structural schedule read
    the one binder keys the fragment realization on (like ``plan.coop`` keys the lane tiling)."""
    red = op.source if isinstance(op, Map) else op
    if not isinstance(red, Reduction) or len(red.partial) == 0:
        return None
    head = red.partial[0]
    if isinstance(head, Contraction) and head.tile.is_warp:
        return head
    return None


def _ext(axis: Axis) -> Expr:
    e = axis.extent
    return Literal(e.as_static(), "int") if e.is_static else e.expr


def _idx(load: Load, sub: dict[str, Expr]) -> tuple:
    """The operand gmem index at the fragment tile origin — σ on the load's own index exprs, so
    batch / GQA (``head // group``) terms ride through untouched."""
    return tuple(Sigma(sub).apply(e) for e in load.index)


def _stats(name: str, op: ElementwiseImpl, args: tuple[str, ...]) -> list[Stmt]:
    """One per-row (``0`` / ``1``) scalar-stat update — a fresh ``Assign`` per fragment row."""
    return [Assign(name=f"{name}{c}", op=op, args=tuple(f"{a}{c}" for a in args)) for c in _COMPS]


def _rebind(name: str, value: str) -> list[Stmt]:
    """Advance a running per-row stat in place (the enclosing ``Init`` is not shadowed)."""
    return [Reassign(name=f"{name}{c}", value=f"{value}{c}") for c in _COMPS]


def _row_pair(name: str) -> tuple[str, str]:
    return (f"{name}0", f"{name}1")


def _reads(s: Stmt) -> set[str]:
    """The SSA names a plain score-prologue / merge stmt reads (explicit per kind — the stop
    condition of the prologue walk)."""
    if isinstance(s, Assign):
        return set(s.args)
    if isinstance(s, Select):
        return {br.value for br in s.branches}
    return set()


def _frag_contraction(
    c: Contraction,
    tiles: list[tuple[tuple[str, ...], tuple[str, ...]]],
    *,
    n_sub,
    k_sub: Expr,
    mask: dict,
    b_slab: str | None = None,
    b_slot: Expr | None = None,
    b_ldm: int = 0,
    b_swizzle: str = "NONE",
    reg_depth: int = 1,
) -> list[Stmt]:
    """One warp-tiled :class:`Contraction` step as fragment codegen — the ``read → ⊗ → fold`` spine
    at fragment residence, geometry off the node (``b_trans`` / operand indices / ``ldm``) and its
    stamped :class:`TilePlan` (``regs`` / ``bk``). ``tiles`` is the ``(acc_frags, a_frags)`` pair
    per REGISTER QUERY TILE (the plan's ``regs[0]``): the A fragments are always resident (flash Q
    is hoisted ahead of the stream, flash P arrives via the C→A repack — one fragment per
    ``atom_k`` step), and every tile's mma chain consumes the SAME B fragment, loaded once per
    ``(step, t)`` — the matmul tier's ``m.reg`` operand sharing, and the source of the per-warp
    ILP (each tile is an independent dependency chain).
    ``n_sub(t)`` / ``k_sub`` give the B operand's tile-origin exprs; ``mask`` maps an axis name to
    its ``(coord, extent)`` overhang guard — a masked B column clamp-reads (transposed-B
    ``gmem_guard``) or zero-fills its overhanging K rows (canonical-B ``k_zero``).

    ``b_slab`` (the staged K/V stream): the B operand reads its smem slab instead of gmem — the
    slab keeps the operand's own layout (transposed-B N-major / canonical-B K-major, the stream
    axis on the slab ROW either way), ``b_slot`` is the ring-slot row offset, ``b_ldm`` the slab
    row stride, and the caller passes SLAB-LOCAL ``n_sub`` / ``k_sub`` origins (the resolver
    guaranteed exact cover, so no B-side masks ride the staged path)."""
    atom = c.tile.atom
    shape = atom.shape
    nt = c.tile.regs[1]  # output n-atoms per step (warp order (FM, FN))
    n_name, k_name = c.n_axis.name, c.k_axis.name
    b_trans = c.b_trans
    ldm_b = b_ldm if b_slab is not None else (c.k_axis.extent.as_static() if b_trans else c.n_axis.extent.as_static())

    bk = c.tile.bk
    # ``reg_depth == 2`` (the ``STAGE`` codec's ``p2``, staged drains with ≥ 2 atom-K steps): the
    # two-slot ldmatrix ping-pong — step ``s+1``'s B fragments load into the alternate slot while
    # step ``s``'s mmas consume the current one, breaking the WAR hazard on the fragment
    # registers. Load PLACEMENT only: the mma order (and every value) is unchanged, so the ``p2``
    # kernel stays bit-identical to its ``p1`` sibling.
    slots = 2 if (reg_depth >= 2 and b_slab is not None and bk >= 2) else 1

    def _bname(t: int, step: int) -> str:
        return f"_{c.acc}_b{t}" if slots == 1 else f"_{c.acc}_b{t}_s{step % 2}"

    def _k0(step: int) -> Expr:
        off = Literal(step * shape[2], "int")
        return k_sub if step == 0 else (off if isinstance(k_sub, Literal) and k_sub.value == 0 else BinaryExpr("+", k_sub, off))

    def _b_load(t: int, step: int) -> LdmatrixLoad:
        col, k0 = n_sub(t), _k0(step)
        if b_slab is not None:
            # The slab keeps the operand's own layout, stream axis on the ROW: transposed-B's
            # rows are its N (key) coords, canonical-B's its K coords — the slot offset lands
            # on the row either way.
            row0, col0 = (col, k0) if b_trans else (k0, col)
            if b_slot is not None:
                row0 = BinaryExpr("+", b_slot, row0)
            return LdmatrixLoad(
                frag=_bname(t, step),
                src_buffer=b_slab,
                src_index=(row0, col0),
                role="b",
                ldm=ldm_b,
                staged=True,
                b_trans=b_trans,
                swizzle=b_swizzle,
            )
        n_guard = mask.get(n_name)
        kz = mask.get(k_name) if not b_trans else None
        return LdmatrixLoad(
            frag=_bname(t, step),
            src_buffer=c.b_load.input,
            src_index=_idx(c.b_load, {n_name: col, k_name: k0}),
            role="b",
            ldm=ldm_b,
            staged=False,
            b_trans=b_trans,
            gmem_guard=(col, n_guard[1]) if (n_guard is not None and b_trans) else None,
            # The zero-fill origin advances with the K step (this slice's own base), so a
            # symbolic bound masks each ``atom_k`` chunk of a wide block correctly.
            k_zero=(k0, kz[1]) if kz is not None else None,
        )

    def _mmas(t: int, step: int) -> list[Stmt]:
        return [
            MmaSyncPtx(
                c_frag=acc_frags[t],
                a_frag=a_frags[step],
                b_frag=_bname(t, step),
                shape=shape,
                ab_dtype=atom.ab_dtype,
                c_dtype=atom.operand_dtype("c").name,
            )
            for acc_frags, a_frags in tiles
        ]

    out: list[Stmt] = []
    for t in range(nt):
        for sl in range(slots):
            out.append(
                RegFragment(
                    name=f"_{c.acc}_b{t}" if slots == 1 else f"_{c.acc}_b{t}_s{sl}", role="b", shape=shape, dtype=atom.operand_dtype("b")
                )
            )
    if slots == 1:
        for step in range(bk):
            for t in range(nt):
                out.append(_b_load(t, step))
                out += _mmas(t, step)
    else:
        out += [_b_load(t, 0) for t in range(nt)]  # prime slot 0
        for step in range(bk):
            if step + 1 < bk:
                out += [_b_load(t, step + 1) for t in range(nt)]  # prefetch the alternate slot
            for t in range(nt):
                out += _mmas(t, step)
    return out


def _realize_prologue(stmts, qk: Contraction, frags: tuple[str, ...], col_bases, row_base: Expr, state) -> tuple[list, list]:
    """Realize the score prologue (the plain stmts between the head contraction and the carrier's
    streaming merge) at fragment residence, stmt kind by stmt kind. Returns ``(hoisted, stream)`` —
    the loop-invariant scalar ``Load``\\ s (hoisted above the stream when a realized stmt reads
    them) and the in-stream fragment stmts. Stops at the first stmt touching a carrier state name —
    from there the merge is REGENERATED from the channel spec, not walked."""
    hoisted: list[Load] = []
    stream: list[Stmt] = []
    score = qk.acc
    m_name, n_name = qk.m_axis.name, qk.n_axis.name
    for s in stmts:
        if isinstance(s, Contraction):
            continue  # the expect contraction — regenerated from its channel
        if state & (set(s.defines()) | _reads(s)):
            break  # the dissolved merge — regenerated from the twist channels
        if isinstance(s, Load) and len(s.index) == 0:
            hoisted.append(s)  # a scalar constant (the 1/√d scale, the −inf fill) — loop-invariant
            continue
        if isinstance(s, Assign) and score in s.args:
            others = tuple(a for a in s.args if a != score)
            for f in frags:
                stream.append(FragmentApply(out=f, op=s.op, args=(f, *others), kinds=(FRAG, *(UNIFORM,) * len(others)), in_place=True))
            score = s.name
            continue
        if isinstance(s, Select) and s.branches[0].value == score:
            keep = s.branches[0].select  # a coordinate predicate over the (m, kv) axis vars
            sub = {m_name: Var(FRAG_ROW), n_name: Var(FRAG_COL)}
            mask_when = BinaryExpr(_NEGATE[keep.op], keep.left.substitute(sub), keep.right.substitute(sub))
            for t, f in enumerate(frags):
                stream.append(FragmentMask(frag=f, mask_when=mask_when, col_base=col_bases[t], row_base=row_base))
            score = s.name
            continue
        raise NotImplementedError(f"fragment realizer: unrealizable score-prologue stmt {type(s).__name__}")
    used = {a for st in stream for a in (st.deps() if hasattr(st, "deps") else ())}
    return [ld for ld in hoisted if set(ld.names) & used], stream


def _causal_stream(stmts, qk: Contraction) -> bool:
    """True when the score prologue carries the causal coordinate ``Select`` — keep ``kv ≤ m`` over
    the contraction's own axis vars. Structural: the triangular kv bound below derives from the
    predicate's shape, never from a kernel identity (an additive-mask or unmasked stream has no
    such ``Select`` and streams the full extent)."""
    for s in stmts:
        if isinstance(s, Select):
            keep = s.branches[0].select
            if (
                isinstance(keep, BinaryExpr)
                and keep.op == "<="
                and isinstance(keep.left, Var)
                and keep.left.name == qk.n_axis.name
                and isinstance(keep.right, Var)
                and keep.right.name == qk.m_axis.name
            ):
                return True
    return False


def _pv_streamed(pv: Contraction, kv_axis: Axis) -> Contraction:
    """The expect contraction with its singleton intra-block axis swapped for the STREAM axis — the
    scalar tree contracts one key per step (``k_axis = pj``, extent 1); the fragment tier contracts
    the whole block, whose keys ride the stream axis in the value ``Load``'s index (``tile.bk = 1``:
    the block is one atom-K step)."""
    return replace(pv, k_axis=kv_axis)


def realize_warp_twist(op, ctx, tail: tuple) -> tuple[list[Stmt], list[Stmt], list[Stmt]]:
    """Realize a warp-tiled ``TWISTED`` reduce tree at fragment residence — the ``(state, fold,
    close)`` triple the one ``_bind`` pipeline seals (state = the handoff slab + running stats +
    output fragments + hoisted scalars; fold = the streaming :class:`StridedLoop`; close = the
    realized projection + the fragment output store). See the module docstring for the walk."""
    red: Reduction = op.source if isinstance(op, Map) else op
    partial = list(red.partial)
    qk: Contraction = partial[0]
    pv: Contraction = next(s for s in partial[1:] if isinstance(s, Contraction))
    atom = qk.tile.atom
    shape = atom.shape
    atom_n = shape[1]
    nt = qk.tile.regs[1]  # score n-atoms per streaming block
    nd = pv.tile.regs[1]  # output n-atoms (d_v / atom_n)
    bn = nt * atom_n  # the streaming KV block width
    d_v = pv.n_axis.extent.as_static()

    # The ambient warp cell: the last grid axis is the shrunk query-block axis (``um`` warps per
    # CTA, each owning ``atom_m`` query rows — the ``FLASH_WARP_AXIS`` term is bound by the
    # ``Tile`` decode when ``um > 1``); the stream rides the reduce axis in ``bn``-key blocks.
    grid = tuple(ctx.grid)
    um = qk.tile.units[0]  # warps per CTA over the query rows
    fm = qk.tile.regs[0]  # register query tiles per warp (FA-2's in-flight ILP)
    m_blk: Expr = Var(grid[-1].name)
    if um > 1:
        m_blk = BinaryExpr("+", BinaryExpr("*", m_blk, Literal(um, "int")), Var(FLASH_WARP_AXIS))

    def _row_base(i: int) -> Expr:
        base: Expr = BinaryExpr("*", m_blk, Literal(fm * shape[0], "int"))
        return base if i == 0 else BinaryExpr("+", base, Literal(i * shape[0], "int"))

    kv_axis = red.axis
    kv0 = Axis(name=f"{kv_axis.name}0", extent=kv_axis.extent)
    kv0_var = Var(kv0.name)
    symbolic_k = not kv_axis.extent.is_static
    symbolic_q = not qk.m_axis.extent.is_static
    seq = _ext(kv_axis)
    # A cross-CTA slice partial (flash split-KV): the fold walks its LOCAL ``[0, B)`` window
    # (``seq`` is the slice length after ``030_split_reduce`` shrank the axis) and ``red.offset`` is the
    # slice's absolute base — added wherever the absolute key coordinate matters: the score-column
    # masks (``col_bases``), the gmem/TMA operand bases, and the causal bound below.
    kv_off = red.offset
    assert kv_off is None or not symbolic_k, "a cross-CTA kv slice is static (030_split_reduce refuses symbolic)"

    def _abs_kv(e: Expr) -> Expr:
        return BinaryExpr("+", e, kv_off) if kv_off is not None else e

    col_bases = tuple(_abs_kv(BinaryExpr("+", kv0_var, Literal(t * atom_n, "int"))) for t in range(nt))

    # Causal tile-skip: a triangular score prologue (keep ``kv ≤ m``) bounds the stream at the
    # CTA's last query row — ``kv_end = min(seq, (grid_m + 1) · um·fm·atom_m)``. Every skipped step
    # is the carrier's exact identity (all-masked scores: ``α = 1``, ``P = expf(−1e30 − m_i) = 0``),
    # so the early stop is bit-identical to the full stream. The bound is CTA-uniform (max over the
    # block's warps/query tiles, the raw grid var — NOT the per-warp ``m_blk``), which keeps the
    # staged path's in-loop barriers legal. On a slice partial the bound is slice-local (the base
    # subtracted; an above-the-diagonal slice goes negative and the loop runs zero steps).
    kv_end: Expr | None = None
    if _causal_stream(partial[1:], qk):
        cta_rows = Literal(um * fm * shape[0], "int")
        causal_rows = BinaryExpr("*", BinaryExpr("+", Var(grid[-1].name), Literal(1, "int")), cta_rows)
        if kv_off is not None:
            causal_rows = BinaryExpr("-", causal_rows, kv_off)
        kv_end = TernaryExpr(cond=BinaryExpr("<", causal_rows, seq), if_true=causal_rows, if_false=seq)

    # The channel spec (pivot first, one carried name per channel) — the merge is REGENERATED from
    # it at fragment residence. The expect channel's ⊗ is the pv node; the denom folds the weights.
    channels = red.carrier.twist.channels
    names = red.carrier.state.names
    pivot_name = names[0]
    expect_name = next(n for n, ch in zip(names, channels, strict=True) if ch.lift is not None)
    denom_name = next(n for n, ch in zip(names[1:], channels[1:], strict=True) if ch.lift is None)

    # ---- per-query-tile naming + state --------------------------------------------------------- #
    # Each of the warp's ``fm`` register query tiles carries its OWN running stats, score / output
    # fragments, hoisted Q fragments, and repacked P fragments (the ``_q<i>`` suffix; ``fm == 1``
    # keeps today's bare names). The tiles are independent ``(m, l, O)`` chains against shared K/V
    # fragments — the per-warp ILP that hides the mma → rowmax → exp → rescale dependency chain.
    pv_bk = max(1, pv.tile.bk)
    # An f16-accumulate PV atom (the ``_f16acc`` sibling — full-rate HMMA on the consumer dies):
    # the P@V mma targets packed f16 fragments (``_h<j>``) and each streaming block promote-folds
    # them into the f32 output shadows, which keep the ``ofrags`` names the rescale / projection /
    # store read — the KV block IS the promote cadence, and the partial is overflow-safe (Σp ≤ 1
    # per row, so a block's f16 partial is bounded by max|V|). Scores (Q@Kᵀ) stay f32-accumulate.
    pv_f16acc = _f16acc(pv.tile.atom)
    qtiles: list[SimpleNamespace] = []
    for i in range(fm):
        sfx = f"_q{i}" if fm > 1 else ""
        qtiles.append(
            SimpleNamespace(
                sfx=sfx,
                row_base=_row_base(i),
                pivot=f"{pivot_name}{sfx}",
                denom=f"{denom_name}{sfx}",
                sfrags=tuple(f"{qk.acc}{sfx}_f{t}" for t in range(nt)),
                pfrags=tuple(f"_p{sfx}_f{t}" for t in range(nt)),
                ofrags=tuple(f"{expect_name}{sfx}_f{j}" for j in range(nd)),
                ohfrags=tuple(f"{expect_name}{sfx}_h{j}" for j in range(nd)) if pv_f16acc else (),
                qa=tuple(f"_{qk.acc}{sfx}_a{s}" for s in range(qk.tile.bk)),
                pa=tuple(f"_pa{sfx}{s}" for s in range(pv_bk)) if pv_bk > 1 else (f"_pa{sfx}",),
            )
        )

    # Per-transport staging facts, visible to the stream builder's drains: cp.async slabs take the
    # +16 B row pad (the bank-conflict break), TMA slabs stay dense and take the hardware swizzle
    # (``pick_swizzle_atom`` per operand) with the drain's address XOR undoing it. The ALTERNATING
    # pipeline (``stage.alt``) additionally stages Q: a padded row-major smem tile filled once,
    # its A fragments ldmatrix'd per atom-K chunk INSIDE the stream — the freed resident registers
    # are what make the wide (64-key) block fit.
    stage = ctx.stage
    alt = stage is not None and stage.alt
    is_tma = stage is not None and stage.transport == "tma"
    elem_bytes = atom.operand_dtype("b").nbytes
    kv_pad = 0 if is_tma else _PAD
    k_swz = pick_swizzle_atom(qk.k_axis.extent.as_static(), elem_bytes)[1] if is_tma else "NONE"
    v_swz = pick_swizzle_atom(d_v, elem_bytes)[1] if is_tma else "NONE"
    head_dim_s = qk.k_axis.extent.as_static()
    q_ldm = head_dim_s + _PAD  # the staged-Q slab row stride (padded, like the cp.async K/V rows)

    def _q_frag_stmts(qt, i: int) -> list[Stmt]:
        """The query tile's ``bk`` A-fragment declarations + loads. Resident form (default): gmem
        loads hoisted ahead of the stream. Alternating form (``stage.alt``): per-block ldmatrix
        from the staged Q slab (slab-local rows — the tile's offset within the CTA's query block),
        block-scoped so the registers die each step. Same values either way: bit-identical."""
        stmts: list[Stmt] = []
        if alt:
            rel = Literal(i * shape[0], "int")
            if um > 1:
                rel = BinaryExpr("+", BinaryExpr("*", Var(FLASH_WARP_AXIS), Literal(fm * shape[0], "int")), rel)
            for s, qaf in enumerate(qt.qa):
                stmts.append(RegFragment(name=qaf, role="a", shape=shape, dtype=atom.operand_dtype("a")))
                stmts.append(
                    LdmatrixLoad(
                        frag=qaf, src_buffer="_q_smem", src_index=(rel, Literal(s * atom.atom_k, "int")), role="a", ldm=q_ldm, staged=True
                    )
                )
            return stmts
        # The Q A-fragments are loop-INVARIANT (Q's index carries the query/head-dim axes, never
        # the stream axis — structural: the contraction binder puts the m-carrying load on A), so
        # they load ONCE ahead of the stream — ``bk`` resident fragments per tile instead of
        # ``bk`` gmem fills per KV block. Same loads, same values: bit-identical to the in-loop form.
        q_guard = (qt.row_base, _ext(qk.m_axis)) if symbolic_q else None
        for s, qaf in enumerate(qt.qa):
            stmts.append(RegFragment(name=qaf, role="a", shape=shape, dtype=atom.operand_dtype("a")))
            stmts.append(
                LdmatrixLoad(
                    frag=qaf,
                    src_buffer=qk.a_operand.input,
                    src_index=_idx(qk.a_operand, {qk.m_axis.name: qt.row_base, qk.k_axis.name: Literal(s * atom.atom_k, "int")}),
                    role="a",
                    ldm=head_dim_s,
                    staged=False,
                    gmem_guard=q_guard,
                )
            )
        return stmts

    state: list[Stmt] = []
    for i, qt in enumerate(qtiles):
        for c in _COMPS:
            state.append(Init(name=f"{qt.pivot}{c}", identity=-1e30, dtype=F32))
            state.append(Init(name=f"{qt.denom}{c}", identity=0.0, dtype=F32))
        for f in qt.ofrags:
            state.append(RegFragment(name=f, role="c", shape=shape, dtype=F32))
        if not alt:
            state.extend(_q_frag_stmts(qt, i))

    # ---- the streaming step (a builder — the staged path re-parents it under the ring drain; the
    # alternating path hands the tagged segments to the liveness scheduler, which places the K/V
    # refills at their kill points) -------------------------------------------------------------- #
    def _stream_segments(
        k_slot: Expr | None = None, v_slot: Expr | None = None, staged: bool = False
    ) -> list[tuple[list[Stmt], frozenset[str]]]:
        stream: list[Stmt] = []
        for i, qt in enumerate(qtiles):
            if alt:
                stream.extend(_q_frag_stmts(qt, i))  # per-block A fragments off the staged Q slab
            for f in qt.sfrags:
                stream.append(RegFragment(name=f, role="c", shape=shape, dtype=F32))
            for f in qt.ohfrags:
                # The per-block packed f16 P@V targets — zero-inited per step (block-scoped, like
                # the score fragments); the promote below folds them into the f32 ``ofrags`` shadows.
                stream.append(RegFragment(name=f, role="c", shape=shape, dtype=pv.tile.atom.operand_dtype("c")))
        qk_mask: dict[str, tuple] = {}
        if symbolic_k:
            qk_mask[qk.n_axis.name] = (kv0_var, seq)
        stream += _frag_contraction(
            qk,
            [(qt.sfrags, qt.qa) for qt in qtiles],
            # Staged: the K slab is slot-local (its rows are this block's keys), so the score
            # column origin is the in-block offset; the absolute ``col_bases`` still name the
            # score columns for the masks below.
            n_sub=(lambda t: Literal(t * atom_n, "int")) if staged else (lambda t: col_bases[t]),
            k_sub=Literal(0, "int"),
            mask=qk_mask,
            b_slab="_k_smem" if staged else None,
            b_slot=k_slot,
            b_ldm=qk.k_axis.extent.as_static() + kv_pad if staged else 0,
            b_swizzle=k_swz if staged else "NONE",
            reg_depth=stage.reg_depth if staged else 1,
        )
        qk_seg, stream = stream, []

        for qi, qt in enumerate(qtiles):
            hoisted, pro = _realize_prologue(partial[1:], qk, qt.sfrags, col_bases, qt.row_base, set(names))
            if qi == 0:  # the hoisted scalar constants (the 1/√d scale) are tile-invariant
                state.extend(hoisted)
            stream += pro
            if symbolic_k:
                # The blocked stream may overrun a symbolic extent — clamp the overhanging keys to
                # the pivot's fold identity so they contribute nothing (the gmem reads were clamped).
                for t, f in enumerate(qt.sfrags):
                    stream.append(FragmentMask(frag=f, mask_when=BinaryExpr(">=", Var(FRAG_COL), seq), col_base=col_bases[t]))

        # ---- the merge, regenerated from the channel spec at fragment residence, per tile ------ #
        # The per-block fold move derives from the ONE placement-keyed selector: the streamed block
        # lives within one warp, so ReduceStage.combine(BLOCK, 32) yields the SHFL move — realized
        # here at FRAGMENT residence as the FragmentRowReduce __shfl butterfly (the same move
        # _factor.emit_combine realizes as a WarpShuffle over scalar registers). Each query tile's
        # merge is an independent chain — ptxas interleaves them (the reg_m ILP).
        (row_move,) = ReduceStage(Level.BLOCK, 32).combine(warp_size=32)
        assert row_move is Fold.SHFL, row_move
        for qt in qtiles:
            s = qt.sfx
            # pivot: the per-block fold (rowmax) then the running update mn = fold(m, rowmax(S)) + α.
            stream.append(FragmentRowReduce(top=f"_rmx{s}0", bot=f"_rmx{s}1", frags=qt.sfrags, op=channels[0].fold, group=4))
            stream += _stats(f"_mn{s}", channels[0].fold, (qt.pivot, f"_rmx{s}"))
            stream += _stats(f"_al{s}__d", _SUB, (qt.pivot, f"_mn{s}"))  # α = exp(m − mn)
            stream += _stats(f"_al{s}", _EXP, (f"_al{s}__d",))
            for sf, pf in zip(qt.sfrags, qt.pfrags, strict=True):  # the softmax weights P = exp(S − mn)
                stream.append(FragmentApply(out=pf, op=_SUB, args=(sf, _row_pair(f"_mn{s}")), kinds=(FRAG, ROW)))
                stream.append(FragmentApply(out=pf, op=_EXP, args=(pf,), kinds=(FRAG,), in_place=True))
            # denom (no lift): the per-block fold is the exp-weight rowsum; l = l·α + Σp.
            stream.append(FragmentRowReduce(top=f"_rsm{s}0", bot=f"_rsm{s}1", frags=qt.pfrags, op=channels[1].fold, group=4))
            stream += _stats(f"{qt.denom}__s", _MUL, (qt.denom, f"_al{s}"))
            stream += _stats(f"{qt.denom}__n", _ADD, (f"{qt.denom}__s", f"_rsm{s}"))
            stream += _rebind(qt.denom, f"{qt.denom}__n")
            # expect (lift = ⊗): rescale the accumulator, then the lift IS the P@V contraction —
            # the register-resident P repacked straight into its A-operand fragments (the C→A
            # register repack: ``atom.c_to_a_repack``, gated at schedule time; one A slice per
            # ``atom_k`` chunk, converted per lane from its two k-adjacent P C-fragments).
            for f in qt.ofrags:
                stream.append(FragmentApply(out=f, op=_MUL, args=(f, _row_pair(f"_al{s}")), kinds=(FRAG, ROW), in_place=True))
            for si, paf in enumerate(qt.pa):
                stream.append(RegFragment(name=paf, role="a", shape=shape, dtype=atom.operand_dtype("a")))
                stream.append(FragmentRepack(frag=paf, srcs=(qt.pfrags[2 * si], qt.pfrags[2 * si + 1]), ab_dtype=atom.ab_dtype))
        mid_seg, stream = stream, []
        pv_mask = {kv_axis.name: (kv0_var, seq)} if symbolic_k else {}
        stream += _frag_contraction(
            _pv_streamed(pv, kv_axis),
            [((qt.ohfrags if pv_f16acc else qt.ofrags), qt.pa) for qt in qtiles],
            n_sub=lambda j: Literal(j * atom_n, "int"),
            # Staged: the V slab is slot-local too (K rows are this block's keys). Gmem-direct
            # reads V at the absolute key (the slice base added on a split partial).
            k_sub=Literal(0, "int") if staged else _abs_kv(kv0_var),
            mask=pv_mask,
            b_slab="_v_smem" if staged else None,
            b_slot=v_slot,
            b_ldm=d_v + kv_pad if staged else 0,
            b_swizzle=v_swz if staged else "NONE",
            reg_depth=stage.reg_depth if staged else 1,
        )
        if pv_f16acc:
            # The block promote: fold the packed f16 P@V partials into the f32 output shadows
            # (O = α·O — applied above — + promote(block P@V)). The rescale never touches f16.
            for qt in qtiles:
                stream += [FragmentPromote(dst=of, src=oh) for of, oh in zip(qt.ofrags, qt.ohfrags, strict=True)]
        for qt in qtiles:
            stream += _rebind(qt.pivot, f"_mn{qt.sfx}")  # advance the running pivot: m = mn
        # Segments tagged by the slabs they READ — the liveness key ``pipelined_kloop`` derives
        # each staged operand's live range from. The QK contraction drains the K slab (and, under
        # ``alt``, the staged-Q slab — a loop-invariant prologue-only fill, never scheduled); the
        # softmax merge reads none; the P·V contraction drains the V slab. The boundaries ARE the
        # partial's node boundaries; gmem-direct segments read no slabs.
        qk_reads = frozenset(("_k_smem", *(("_q_smem",) if alt else ()))) if staged else frozenset()
        pv_reads = frozenset(("_v_smem",)) if staged else frozenset()
        return [(qk_seg, qk_reads), (mid_seg, frozenset()), (stream, pv_reads)]

    def _stream_step(k_slot: Expr | None = None, v_slot: Expr | None = None, staged: bool = False) -> list[Stmt]:
        return [s for stmts, _reads in _stream_segments(k_slot, v_slot, staged) for s in stmts]

    if stage is not None:
        # The staged K/V stream: the resolved ``Stage`` (``_schedule._resolve_twisted_stage`` —
        # static, block-divisible kv only) rides the SAME ``staged_kloop`` skeleton the matmul tier
        # runs, the streaming step as its drain. The K slab keeps K's own N-major layout (``bn`` key
        # rows × head_dim) and the V slab V's K-major one (``bn`` key rows × d_v) — the fills are
        # verbatim row copies, so the staged kernel stays bit-identical to its gmem-direct sibling.
        # cp.async fills cooperatively into padded rows; TMA box-copies the batched operands via
        # rank-N descriptors (leading extent-1 box dims, the load's own batch/head index exprs as
        # origin coords — GQA's ``h // group`` rides through) into dense hardware-swizzled slabs.
        # A symbolic kv stages under both: the streaming loop bound / last-chunk clamp ride the
        # symbolic ``Dim``; TMA zero-fills the box overhang, cp.async clamp-reads the tail's key
        # rows to the last valid key (``_key_row``). Either way the drain's tail masks zero the
        # overhanging P columns, so the staged kernel stays bit-identical to gmem-direct.
        kv_ext = kv_axis.extent if symbolic_k else kv_axis.extent.as_static()
        n_chunks = kv_axis.extent.ceil_div(bn) if symbolic_k else kv_axis.extent.as_static() // bn
        head_dim = qk.k_axis.extent.as_static()
        k_load, v_load = qk.b_load, pv.b_load
        kn, kk, vn = qk.n_axis.name, qk.k_axis.name, pv.n_axis.name

        def _key_row(k0: Expr, row) -> Expr:
            # cp.async has no OOB zero-fill — clamp a symbolic tail's key rows to the last valid
            # key. The drain's tail masks zero the overhanging P columns, so the duplicated rows
            # contribute exactly 0 (the cp.async counterpart of TMA's box zero-fill, which needs
            # no clamp). A slice partial reads at the absolute key (``_abs_kv`` — k0 is window-local).
            key = _abs_kv(BinaryExpr("+", k0, row))
            return _clamp_last(key, seq) if symbolic_k and not is_tma else key

        def _k_index(k0: Expr):
            return lambda row, col: _idx(k_load, {kn: _key_row(k0, row), kk: col})

        def _v_index(k0: Expr):
            return lambda row, col: _idx(v_load, {kv_axis.name: _key_row(k0, row), vn: col})

        k_batch, v_batch = tuple(k_load.index[:-2]), tuple(v_load.index[:-2])
        k_op = Operand(
            tag="k",
            buf=k_load.input,
            shape=(bn, head_dim),
            index=_k_index,
            coords=lambda k0: (*k_batch, _abs_kv(k0), Literal(0, "int")),
            pad_cols=kv_pad,
            swizzle=k_swz,
            box=(1,) * len(k_batch) + (bn, head_dim) if is_tma else None,
        )
        v_op = Operand(
            tag="v",
            buf=v_load.input,
            shape=(bn, d_v),
            index=_v_index,
            coords=lambda k0: (*v_batch, _abs_kv(k0), Literal(0, "int")),
            pad_cols=kv_pad,
            swizzle=v_swz,
            box=(1,) * len(v_batch) + (bn, d_v) if is_tma else None,
        )
        transport_cls = TmaTransport if is_tma else CpAsyncTransport
        # Under a WSPEC band split the fill's elected thread must be the PRODUCER band's first —
        # the wrapped linear tid (``threadIdx.x % block_threads``) maps aux thread ``block_threads``
        # to 0, so the transport's ``linear_tid == 0`` election lands there verbatim (the raw tid
        # would elect a compute thread and the producer branch would never fill).
        block_threads = 32 * um
        ltid: Expr = (
            BinaryExpr("%", Builtin("thread_idx.x"), Literal(block_threads, "int")) if ctx.workers is not None else Builtin("thread_idx.x")
        )
        cta = CtaTile(linear_tid=ltid, n_threads=block_threads)
        if alt:
            # The ALTERNATING single-slab pipeline (``d1/tma/alt``): one slab + one mbarrier per
            # operand, each a depth-1 group whose refill ``pipelined_kloop`` places at its KILL
            # POINT — K's live range ends at the QK segment so its refill lands under softmax +
            # P·V, V's at the P·V segment so its refill lands under the next step's Q·K (the FA-2
            # choreography, DERIVED from the tagged live ranges, not hand-assembled). Q stages
            # through smem: a padded row-major slab cooperatively cp.async-filled ONCE before the
            # stream (verbatim copy — bit-identical values; the δ=0 loop-invariant degenerate —
            # nothing advances, so its whole schedule is this prologue fill), its A fragments
            # ldmatrix'd per atom-K chunk inside the step (``_q_frag_stmts``), freeing the
            # resident Q registers.
            q_rows = um * fm * shape[0]
            cta_row0 = BinaryExpr("*", Var(grid[-1].name), Literal(q_rows, "int"))
            state.append(slab_smem("_q_smem", q_rows, q_ldm, _cuda(atom.operand_dtype("a")), align=16))
            state += cp_async_fill(
                slab="_q_smem",
                shape=(q_rows, head_dim_s),
                src=qk.a_operand.input,
                gmem_index=lambda row, col: _idx(qk.a_operand, {qk.m_axis.name: BinaryExpr("+", cta_row0, row), qk.k_axis.name: col}),
                cta=cta,
                elem_bytes=elem_bytes,
                name="q",
            )
            state += cp_async_commit()
            state += cp_async_wait(0)
            if is_tma:
                k_transport = TmaTransport(
                    operands=(k_op,), slab_dtype=_cuda(atom.operand_dtype("b")), elem_bytes=elem_bytes, cta=cta, mbar="_kbar"
                )
                v_transport = TmaTransport(
                    operands=(v_op,), slab_dtype=_cuda(atom.operand_dtype("b")), elem_bytes=elem_bytes, cta=cta, mbar="_vbar"
                )
            else:
                # cp.async: per-operand commit groups — the K,V,K,V commits queue per thread, and
                # the skeleton's uniform ``wait_group(1)`` completes exactly the older sibling.
                k_transport = CpAsyncTransport(operands=(k_op,), slab_dtype=_cuda(atom.operand_dtype("b")), elem_bytes=elem_bytes, cta=cta)
                v_transport = CpAsyncTransport(operands=(v_op,), slab_dtype=_cuda(atom.operand_dtype("b")), elem_bytes=elem_bytes, cta=cta)
            decls, fold = pipelined_kloop(
                operands=((k_transport, 1), (v_transport, 1)),
                build_segments=lambda slots: _stream_segments(k_op.slot_row(slots[0]), v_op.slot_row(slots[1]), staged=True),
                bk_elems=bn,
                n_chunks=n_chunks,
                k_extent=kv_ext,
                k0=kv0.name,
                k_end=kv_end,
            )
            state = decls + state
        else:
            transport = transport_cls(
                operands=(k_op, v_op),
                slab_dtype=_cuda(atom.operand_dtype("b")),
                elem_bytes=elem_bytes,
                cta=cta,
            )
            decls, fold = staged_kloop(
                transport=transport,
                drain=lambda slot: _stream_step(k_op.slot_row(slot), v_op.slot_row(slot), staged=True),
                depth=stage.depth,
                bk_elems=bn,
                n_chunks=n_chunks,
                k_extent=kv_ext,
                k0=kv0.name,
                workers=ctx.workers,
                block_threads=block_threads,
                k_end=kv_end,
            )
            state = decls + state
    else:
        static_small = unroll_ok(kv_axis.extent, 128) and kv_end is None
        body = Body(tuple(_stream_step()))
        fold = [StridedLoop(axis=kv0, start=Literal(0, "int"), step=Literal(bn, "int"), body=body, unroll=static_small, end=kv_end)]

    # ---- close: the projection tail realized on the output fragments + the store, per tile ---- #
    # A layout-aware output ``Write`` the node carries (flash's ``_out_store_index``) is the store
    # TEMPLATE — its index places the output at the buffer's real slots (a fused transpose, size-1
    # broadcast dims). Split it off the projection stmts; each fragment store reuses the template with
    # the row (m) axis → the query-tile base and the column (n) axis → the atom-n literal. Absent it,
    # fall back to the bare ``(batch…, row, n)`` grid store (the head-major identity).
    m_name, n_name = qk.m_axis.name, pv.n_axis.name

    # A cross-CTA slice partial stores the RAW carrier state instead of projecting (``030_split_reduce``
    # replaced the projection with one ``Write`` per state component into the ``__partial``
    # workspace): the expect component (O) is fragment-resident and rides the normal fragment
    # store; the d-invariant row stats (m, l) are written once per query row at their template's
    # pinned last slot, by the ``_t == 0`` lane of each row (the m16n8 C layout: lane ``g·4``
    # holds rows ``g`` / ``g+8``).
    if tail and all(isinstance(s, Write) for s in tail) and {s.value for s in tail} == set(names):
        by_state = {s.value: s for s in tail}
        lane = BinaryExpr("%", Builtin("thread_idx.x"), Literal(32, "int"))
        g_expr = BinaryExpr("/", lane, Literal(4, "int"))
        t_zero = BinaryExpr("==", BinaryExpr("%", lane, Literal(4, "int")), Literal(0, "int"))
        close = []
        for qt in qtiles:
            o_tmpl = by_state[expect_name]
            for j, f in enumerate(qt.ofrags):
                sub = {m_name: qt.row_base, n_name: Literal(j * atom_n, "int")}
                dst_index = tuple(sub.get(e.name, e) if isinstance(e, Var) else e for e in o_tmpl.index)
                close.append(RegStore(dst_buffer=ctx.output, dst_index=dst_index, frag=f, shape=shape, ldm=d_v))
            row_writes: list[Stmt] = []
            for comp in (pivot_name, denom_name):
                tmpl = by_state[comp]
                for c, row_off in zip(_COMPS, (g_expr, BinaryExpr("+", g_expr, Literal(8, "int"))), strict=True):
                    row_expr = BinaryExpr("+", qt.row_base, row_off)
                    idx = tuple(row_expr if (isinstance(e, Var) and e.name == m_name) else e for e in tmpl.index)
                    row_writes.append(Write(output=ctx.output, index=idx, value=f"{comp}{qt.sfx}{c}"))
            close.append(Cond(cond=t_zero, body=tuple(row_writes)))
        return state, fold, close

    store_tmpl = tail[-1] if tail and isinstance(tail[-1], Write) else None
    proj_tail = tail[:-1] if store_tmpl is not None else tail
    close: list[Stmt] = []
    batch_idx = tuple(qk.a_operand.index[:-2])  # (batch…, head) — passthrough grid vars (fallback)
    for qt in qtiles:
        for s in proj_tail:
            if isinstance(s, Assign) and expect_name in s.args:
                others = tuple(_row_pair(f"{a}{qt.sfx}") if a in names else a for a in s.args if a != expect_name)
                kinds = tuple(ROW if a in names else UNIFORM for a in s.args if a != expect_name)
                for f in qt.ofrags:
                    close.append(FragmentApply(out=f, op=s.op, args=(f, *others), kinds=(FRAG, *kinds), in_place=True))
                continue
            raise NotImplementedError(f"fragment realizer: unrealizable projection stmt {type(s).__name__}")
        m_guard = (qt.row_base, _ext(qk.m_axis)) if symbolic_q else None
        for j, f in enumerate(qt.ofrags):
            col_lit = Literal(j * atom_n, "int")
            if store_tmpl is not None:
                sub = {m_name: qt.row_base, n_name: col_lit}  # row/col axis motion; batch / broadcast slots pass through
                dst_index = tuple(sub.get(e.name, e) if isinstance(e, Var) else e for e in store_tmpl.index)
            else:
                dst_index = (*batch_idx, qt.row_base, col_lit)
            close.append(RegStore(dst_buffer=ctx.output, dst_index=dst_index, frag=f, shape=shape, ldm=d_v, m_guard=m_guard))
    return state, fold, close


def _cuda(dtype) -> str:
    from emmy.compiler.backend.cuda.dtype import cuda_name  # noqa: PLC0415

    return cuda_name(dtype)


__all__ = ["realize_warp_twist", "warp_source"]
