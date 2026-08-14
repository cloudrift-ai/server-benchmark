"""Flash-attention helper — recognition and construction.

The ``010_recognize`` pass calls :func:`try_flash`; everything flash lives here. Two
halves:

- **Recognition** — :func:`try_flash` matches the SDPA unit across either loop-fusion
  boundary: the original softmax-then-P@V kernel + scaled-QK producer, or gate-free fusion's
  P@V root + softmax producer containing repeated QK contractions. ``_recognize`` feeds both
  forms through the same extraction, eligibility, and fragment construction path.
- **Construction** — the ``flash_shape_eligible`` / ``gqa_group`` predicates and the fragment
  builder ``build_flash_frag``. It doesn't hand-assemble a kernel body — it builds the high-level
  **λ-spelled structural-node tree** (``ir/tile/ir``): flash is ``Fold.projection(body=[O/l projection],
  operands=(Fold(axis=kv, operands=(Fold(Σ_dd Q·K), Load(V)), lift=λ(kv, sacc, v_e) → (score, 1,
  v_e), init/combine = the exp-family ⊕),))`` — the ``(m,l,O)`` LSE streaming reduce over kv whose
  ``Σ_dd Q·K`` score is a HOISTED operand edge and whose **P@V** ``Σ_j P·V`` contraction (its A the
  register-resident softmax weight ``P``) is SYNTHESIZED into the DERIVED blocked evaluation
  (``ir._twisted_derived_step``), projected ``O/l`` after the loop. Both Q@K and P@V ride
  single walked ``step`` edge (no ``source`` asymmetry) and factorize through the one
  ``_factor`` contraction path; block=1 is the scalar streaming degenerate (``j`` a singleton reduce).
  ``build_flash_frag`` returns that zero-axis ``Fold`` UNLOWERED, on a ``TileOp`` with an UNMAPPED ``Placement``
  — the free ``(batch…, m, d)`` axes are the schedule's (like every recognizer), and ``_schedule``
  maps them onto the grid; ``materialize`` lowers the nodes (``Fold.loop`` splices the
  contraction's ``Σ Q·K`` loop ahead of the partial, the scalar tier expanding the same loop nest) +
  generates the output-store glue (the ``Write`` at the grid cell — not stored).

  A tensor-core flash tier is **not** a bespoke emitter: the mandate is that giving the Q@K / P@V
  contractions an mma ``TilePlan`` (a schedule field on the node) must route through the same
  ``_factor`` contraction path as any other mma matmul. Until that lands the whole way through the one
  emitter, flash lowers **only** on the scalar tier below — there is no divergent flash codegen path.

``is_flash_score_producer`` lets ``010_recognize`` defer the general lift of a standalone
scaled-QK score producer until its softmax-then-P@V consumer has fused.

Layout-agnostic on BOTH sides. The input loads permute the canonical ``(batch…, seq, last)`` index
back into each operand's own traced slot order (``_permute_idx`` — HF's per-layer Q/K/V arrive
seq-major); the output ``Write`` reproduces the root buffer's real layout — a fused output transpose
and size-1 broadcast / unsqueeze dims (HF's ``[b, h, s, 1, d]``) — via ``_out_store_index`` (the store
index each lowering tier uses, mapping grid axes onto the output slots by dim extent). A bare
grid-order store mis-strides a non-canonical output (all elements alias, the rest uninitialized → NaN).

(Online softmax — flash's softmax-stats half without the P@V — lives in ``_softmax``.)

Q/K may be plain loads, packed affine views, or closed computed operand edges with one
λ-representable statistic reduce (the existing computed-contraction spelling, covering fused
RMSNorm). An arbitrary score program — for example an unclosed or non-map RoPE cone — still
declines rather than being approximated. A pure V map cone is factored back into a canonical
workspace by the same fragment so the expectation contraction retains its materialized,
stageable B operand.

The fragment fuses scaled-dot-product attention into ONE kernel that tiles the KV
(reduce) axis and never materializes the ``[S_q, S_k]`` score matrix. The scalar tier
runs one independent streaming softmax per output element ``(…, m, d)`` — a correct, if
redundant, form::

    for *batch, m (query rows), d (value dim):       # free / grid
      Init (m_i = -inf, l_i = 0, O_i = 0)            # running (max, denom, out)
      for kv in 0..S_k:                              # streaming reduce (TWISTED)
        Init sacc = 0
        for dd in 0..head_dim: sacc += Q[…,m,dd]·K[…,kv,dd]   # Q@K score bilinear fold
        s = sacc · scale
        M = max(m_i, s); alpha = exp(m_i − M); P = exp(s − M)  # softmax stats (the derived exp merge)
        l_i = l_i·alpha + P
        for j in 0..1:  O_i__pv += P · V[…,kv,d]     # P@V bilinear fold (block=1: singleton j)
        O_i = O_i·alpha + O_i__pv                    # the LSE rescale + PV fold
      out[…,m,d] = O_i / l_i

Scope: static OR dynamic (symbolic ``seq_len`` on Q/K/V dim -2 — one cached kernel
carrying ``int seq_len`` serves every runtime size, the symbol landing on BOTH the
masked-row M and the symbolic reduce), causal or non-causal (causal masks the
score per element, ``kv ≤ m`` else −inf; the warp tier additionally tile-skips —
``_twist`` bounds the stream at the CTA's last query row off that ``Select``'s
shape), an optional broadcast additive mask (the HF ``(1,1,S,S)`` float bias),
and GQA (``q_heads == group · kv_heads``; the K/V head axis read at ``head //
group`` directly, no materialized broadcast). Fusion is unconditional — a recognized,
certifiable pair always fuses; an uncertifiable one falls back to the separate score
producer + softmax-then-P@V kernels.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.loop.ir import LoopOp
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Accum, Assign, Body, Lambda, Load, Loop, Select, SelectBranch, Write
from emmy.compiler.ir.stmt.carrier import exp_combine_states
from emmy.compiler.ir.tile import Channel, Fold, Placement, Store, TileOp
from emmy.compiler.pipeline.passes.lowering.tile._atomize import make_cone, map_cone
from emmy.compiler.pipeline.passes.lowering.tile._fromloop import fold_from_loop

if TYPE_CHECKING:
    from emmy.compiler.graph import Node
    from emmy.compiler.ir.stmt.base import Stmt

logger = logging.getLogger(__name__)


# The online-softmax (log-sum-exp) algebra for one streaming KV step — state ``(m, l, O)`` folds
# this key's ``(score, value)`` singleton::
#
#     m_new = max(m, s);   alpha = exp(m − m_new);   p = exp(s − m_new)
#     l = l·alpha + p;     O = O·alpha + p·v;         m = m_new   (last)
#
# — is no longer authored here at all: ``_flash_op`` stores the TRUE ⊕ (the generated
# ``exp_combine_states`` program) flat on the ``Fold``, and formation derives the carrier /
# streaming merge / blocked evaluation from it (``ir/stmt/carrier.py`` generates, the stabilizer
# recovers the max-rescale, ``ir._twisted_derived_step`` splits the PV contraction out).


def _static(d) -> int | None:
    """The static extent of a ``Dim``, or ``None`` when symbolic."""
    return d.as_static() if d.is_static else None


def _struct_features(batch: list[int], s_q: Dim, s_k: Dim, head_dim: int, d_v: int) -> dict[str, float]:
    """The ``S_ext_*`` structural skeleton for the fused flash kernel — the same extent features
    ``loop/stamp``'s ``020_stamp_structural_features`` would put on a ``LoopOp``, computed here
    because the fused fragment is BUILT as a ``TileOp`` (it never passes the loop-dialect stamp).
    Free = the grid ``(batch…, m, d)``, reduce = the streamed ``kv`` + the score ``dd``; a symbolic
    seq axis is excluded from the products and counted in ``S_ext_n_symbolic_axis`` (the flag the
    ``OfflinePrior`` selects its masked-tier weight set on). Riding the ``TileOp`` knob base, they
    reach every flash fork leaf row, so the prior's occupancy terms fire when ranking the forms."""
    free = [*batch, d_v]
    reduce_ = [head_dim]
    n_symbolic = 0
    for dim, bucket in ((s_q, free), (s_k, reduce_)):
        if dim.is_static:
            bucket.append(dim.as_static())
        else:
            n_symbolic += 1
    return {
        "S_ext_n_free_axis": float(len(free)),
        "S_ext_free_prod": float(math.prod(free)),
        "S_ext_free_max": float(max(free)),
        "S_ext_n_reduce_axis": float(len(reduce_)),
        "S_ext_reduce_prod": float(math.prod(reduce_)),
        "S_ext_reduce_max": float(max(reduce_)),
        "S_ext_n_symbolic_axis": float(n_symbolic),
    }


def gqa_group(q_shape: tuple, k_shape: tuple) -> int | None:
    """The grouped-query head ratio ``q_heads // kv_heads`` (1 when equal-head),
    or ``None`` when the head axis isn't statically divisible. The head axis is the
    last batch dim (``shape[-3]``); rank < 3 has no head (group 1)."""
    qh = _static(q_shape[-3]) if len(q_shape) >= 3 else 1
    kh = _static(k_shape[-3]) if len(k_shape) >= 3 else 1
    if qh is None or kh is None or kh == 0 or qh % kh != 0:
        return None
    return qh // kh


def flash_shape_eligible(q_shape: tuple, k_shape: tuple, v_shape: tuple, *, group: int, mask_shape: tuple | None) -> bool:
    """True iff the flash nest can serve this SDPA — static batch/head (only the
    seq axis may be symbolic), an optional broadcastable additive mask, and GQA
    where ``q_heads == group · kv_heads``. The K/V head axis is read at
    ``head // group`` directly in the nest (no materialized broadcast). The
    recognizer and this predicate MUST agree, so both call it."""
    if len(q_shape) < 2 or len(k_shape) < 2 or len(v_shape) < 2:
        return False
    q_batch = [_static(d) for d in q_shape[:-2]]
    k_batch = [_static(d) for d in k_shape[:-2]]
    v_batch = [_static(d) for d in v_shape[:-2]]
    if any(b is None for b in (*q_batch, *k_batch, *v_batch)):
        return False  # symbolic batch / head — only the seq axis may be dynamic
    if len(q_batch) != len(k_batch) or len(q_batch) != len(v_batch):
        return False
    if q_batch:
        # Leading (non-head) batch dims must match exactly; the head axis (last
        # batch dim) is q = group · kv.
        if q_batch[:-1] != k_batch[:-1] or q_batch[:-1] != v_batch[:-1]:
            return False
        if k_batch[-1] != v_batch[-1] or q_batch[-1] != group * k_batch[-1]:
            return False
    elif group != 1:
        return False  # no head axis but a non-trivial group makes no sense
    head_dim, d_v = _static(q_shape[-1]), _static(v_shape[-1])
    if head_dim is None or d_v is None:
        return False  # symbolic head_dim / value-dim
    if _static(k_shape[-1]) != head_dim:
        return False
    if v_shape[-2] != k_shape[-2]:  # V seq must match K seq
        return False
    if mask_shape is not None:
        # Per-(m, kv) additive bias: leading dims must be static 1 (indexed to 0),
        # the trailing two address the query / key seq.
        if len(mask_shape) < 2:
            return False
        if any(_static(d) != 1 for d in mask_shape[:-2]):
            return False
        if mask_shape[-2] != q_shape[-2] or mask_shape[-1] != k_shape[-2]:
            return False
    return True


def build_flash_frag(
    q_id: str,
    k_id: str,
    v_id: str,
    q_shape: tuple,
    k_shape: tuple,
    v_shape: tuple,
    out: Tensor,
    *,
    causal: bool,
    window: int | None = None,
    group: int = 1,
    mask: tuple[str, tuple] | None = None,
    layouts: tuple = (None, None, None),
    access_indices: tuple[tuple, tuple, tuple] | None = None,
    raw_shapes: tuple | None = None,
    out_index: tuple | None = None,
    scale: float | None = None,
    operand_edges: tuple[Load | Fold | None, Load | Fold | None] = (None, None),
    input_tensors: dict[str, Tensor] | None = None,
    materialized_v: Fold | None = None,
) -> Graph | None:
    """Build the fragment graph holding the fused flash ``TileOp`` (+ its scale /
    -inf constants), or ``None`` when the root's output layout can't be reproduced on the grid
    (:func:`_out_store_index` — the caller degrades to cut). The caller guarantees
    :func:`flash_shape_eligible`.

    The compute is the op tree itself — a zero-axis ``Fold`` whose body is the ``(m,l,O)`` LSE
    ``TWISTED`` reduce ``Loop`` then the ``O/l`` projection, carried unlowered on the ``TileOp`` with an empty
    schedule; the free ``(batch…, m, d)`` axes are the ``Placement``'s ``free`` (no
    free-axis loop nest). The schedule maps them onto the grid; ``materialize`` lowers
    the node and generates the output-store glue (the ``Write`` at the grid cell) — it
    isn't stored here.

    ``group`` is the GQA head ratio (K/V indexed at ``head // group``); ``mask`` is
    an optional ``(buffer_id, shape)`` additive bias loaded per ``(m, kv)``. ``q/k/v_shape``
    are CANONICAL ``(batch…, seq, last)``; ``layouts`` are the per-operand ``(seq_pos,
    last_pos)`` slot orders the loads permute back into (``None`` = head-major identity).
    ``access_indices``, when present, are exact canonical-variable Q/K/V load indices carried
    from affine views of one packed buffer (for example ``[..., head * D + d + qkv_offset]``);
    they take precedence over ``layouts``. ``raw_shapes`` are the operands' traced shapes for
    the fragment's input declarations.
    ``scale`` is the score producer's actual scale (an SDPA's captured ``scale=`` kwarg
    survives decomposition as the producer's scale constant); ``None`` = ``1/sqrt(head_dim)``.
    ``operand_edges`` optionally supplies already-canonical Q/K operand nodes recovered
    from inlined producers. A plain edge is a ``Load``; a computed edge is the existing
    structural ``Fold`` spelling (for example an RMSNorm statistic + per-element projection).
    ``input_tensors`` declares every original graph buffer those edges read. ``materialized_v``
    is an inlined pure V cone that the fragment factors into a canonical feeder workspace before
    flash, preserving the expectation contraction's materialized/stageable operand."""
    batch = [_static(d) for d in q_shape[:-2]]
    head_dim, d_v = _static(q_shape[-1]), _static(v_shape[-1])
    s_q_dim, s_k_dim = q_shape[-2], k_shape[-2]  # Dim instances — static int or symbolic seq_len
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    mask_buf, mask_shape = mask if mask is not None else (None, None)

    batch_axes = tuple(Axis(name=f"b{i}", extent=Dim(b)) for i, b in enumerate(batch))
    grid = (*batch_axes, Axis(name="m", extent=s_q_dim), Axis(name="d", extent=Dim(d_v)))

    # The output store must match the root buffer's real rank + layout (transpose / broadcast dims),
    # not the bare grid order. ``out_index`` is the root kernel's own output ``Write`` index; map its
    # per-axis vars onto the grid axes (:func:`_out_store_index`). ``None`` there = an un-reproducible
    # output layout → decline flash (the caller degrades to cut). With no ``out_index`` (isolated
    # fragment / tests), the materializer's bare grid-order glue stores the head-major identity.
    out_store: tuple[str, tuple] | None = None
    if out_index is not None:
        store_idx = _out_store_index(out_index, out.shape, grid)
        if store_idx is None:
            return None
        out_store = (out.name, store_idx)

    flash_op, flash_stores = _flash_op(
        q_id,
        k_id,
        v_id,
        batch,
        s_q_dim,
        s_k_dim,
        head_dim,
        d_v,
        causal=causal,
        window=window,
        group=group,
        mask_buf=mask_buf,
        mask_shape=mask_shape,
        layouts=layouts,
        access_indices=access_indices,
        operand_edges=operand_edges,
        out_store=out_store,
    )
    # The free axes are the schedule's, carried on the ``TileOp`` with an UNMAPPED grid —
    # like every other recognizer; the schedule maps ``free`` onto the grid.
    # The ``S_ext_*`` skeleton rides on ``knobs`` (:func:`_struct_features` — the fused fragment
    # never passes the loop-dialect stamp) so the prior can price the flash forms' occupancy.
    knobs = dict(_struct_features(batch, s_q_dim, s_k_dim, head_dim, d_v))
    tile = TileOp(op=flash_op, place=Placement(free=grid), knobs=knobs, stores=flash_stores)

    frag = Graph()
    in_shapes = raw_shapes if raw_shapes is not None else (q_shape, k_shape, v_shape)
    declared = {nid: Tensor(nid, shp, out.dtype) for nid, shp in zip((q_id, k_id, v_id), in_shapes, strict=True)}
    if mask_buf is not None:
        declared[mask_buf] = Tensor(mask_buf, mask_shape, out.dtype)
    declared.update(input_tensors or {})
    external_reads = tuple(n for n in flash_op.external_reads() if n not in {"_flash_scale", "_flash_ninf"})
    materialized_reads = (
        tuple(dict.fromkeys(stmt.input for stmt in Body(tuple(materialized_v.lower())).iter() if isinstance(stmt, Load)))
        if materialized_v is not None
        else ()
    )
    boundary_reads = tuple(dict.fromkeys((*external_reads, *materialized_reads)))
    for nid in boundary_reads:
        if materialized_v is not None and nid == v_id:
            continue
        tensor = declared.get(nid)
        if tensor is None:
            return None
        if nid not in frag.nodes:
            frag.add_node(op=InputOp(), inputs=[], output=Tensor(nid, tensor.shape, tensor.dtype), node_id=nid)
    inputs = list(external_reads)
    frag.add_node(
        # fp32 constant: the score scale accumulates into the fp32 carrier, so a half-precision
        # ``_flash_scale`` would only add an ``__half2float`` at every use (and round ``1/√d`` to fp16).
        op=ConstantOp(name="_flash_scale", value=scale),
        inputs=[],
        output=Tensor("_flash_scale", (1,), F32),
        node_id="_flash_scale",
    )
    inputs.append("_flash_scale")
    if causal or window is not None:
        # -inf bias for masked (key-after-query / outside-the-band) positions: exp(-inf)=0, so a
        # masked score contributes nothing to the streaming softmax / output.
        frag.add_node(
            op=ConstantOp(name="_flash_ninf", value=-1e30), inputs=[], output=Tensor("_flash_ninf", (1,), out.dtype), node_id="_flash_ninf"
        )
        inputs.append("_flash_ninf")
    if materialized_v is not None:
        v_batch = [_static(d) for d in v_shape[:-2]]
        if any(d is None for d in v_batch) or _static(v_shape[-1]) is None:
            return None
        v_grid = (
            *(Axis(name=f"b{i}", extent=Dim(d)) for i, d in enumerate(v_batch)),
            Axis(name="kv", extent=v_shape[-2]),
            Axis(name="d", extent=v_shape[-1]),
        )
        v_store = Store(write=Write(output=v_id, index=tuple(Var(axis.name) for axis in v_grid), value=materialized_v.out))
        v_tile = TileOp(op=materialized_v, name=v_id, place=Placement(free=v_grid), stores=(v_store,))
        frag.add_node(
            op=v_tile,
            inputs=list(materialized_reads),
            output=Tensor(v_id, v_shape, out.dtype),
            node_id=v_id,
        )
    frag.add_node(op=tile, inputs=inputs, output=Tensor(out.name, out.shape, out.dtype), node_id=out.name)
    frag.outputs = [out.name]
    return frag


def _batch_vars(n: int) -> tuple[Var, ...]:
    return tuple(Var(f"b{i}") for i in range(n))


# (The PV split — the ``O = O·α + v·P`` merge rewritten around a real ``Oblk = Σ_j P·V``
# contraction whose A is the register-resident softmax weight — is DERIVED at step 7:
# ``ir._split_expect`` / ``ir._derived_expect_fold`` synthesize it inside the twisted fold's
# blocked evaluation, byte-identical to the step element this module used to store.)


def _flash_op(
    q_buf: str,
    k_buf: str,
    v_buf: str,
    batch: list[int],
    s_q: Dim,
    s_k: Dim,
    head_dim: int,
    d_v: int,
    *,
    causal: bool = False,
    window: int | None = None,
    group: int = 1,
    mask_buf: str | None = None,
    mask_shape: tuple | None = None,
    layouts: tuple = (None, None, None),
    access_indices: tuple[tuple, tuple, tuple] | None = None,
    operand_edges: tuple[Load | Fold | None, Load | Fold | None] = (None, None),
    out_store: tuple[str, tuple] | None = None,
) -> Fold:
    """The per-output-element ``(…, m, d)`` compute as the structural-node tree: flash is
    ``Fold.projection(body=[O/l projection], operands=(Fold(role=TWISTED, axis=kv, step=[Fold.contraction(QK), …,
    Fold.contraction(PV)]))`` — the ``(m,l,O)`` LSE streaming reduce over ``kv`` whose per-step **partial**
    holds the NESTED ``Σ_dd Q·K`` contraction at its head (then scaled, optionally
    masked, the value read + the carrier's dissolved merge with the PV contraction), projected ``O/l``
    after the loop. :meth:`Fold.loop` flattens the head QK node the same way it flattens the
    embedded PV, so the scalar tier expands the same loop-in-body nest as before. The free
    ``(batch…, m, d)`` axes are the ``TileOp``'s grid, not loops here; the output store is glue
    generated at materialize.

    GQA: the K/V head axis (last batch dim) is read at ``head // group``, the same
    ``//group`` the upstream ``IndexMapOp`` encodes, moved into the load index so the
    kv_heads-many K/V are read without materializing the q_heads expansion. An additive
    ``mask_buf`` (broadcast leading dims) is added to the score; causal masking is a
    ``Select`` stmt (``kv ≤ m`` else −inf) in the score zero-axis ``Fold`` — the index predicate
    lives in the op tree, never in the carrier. Both make ``exp(s − m_new) = 0``, so
    masked keys contribute nothing."""
    bvars = _batch_vars(len(batch))
    head_axis = len(batch) - 1  # last batch dim is the head (when there is one)
    kv_bvars = tuple(BinaryExpr("/", bv, Literal(group, "int")) if (group > 1 and i == head_axis) else bv for i, bv in enumerate(bvars))
    q_layout, k_layout, v_layout = layouts
    if access_indices is None:
        q_idx = _permute_idx((*bvars, Var("m"), Var("dd")), q_layout)
        k_idx = _permute_idx((*kv_bvars, Var("kv"), Var("dd")), k_layout)
        v_idx = _permute_idx((*kv_bvars, Var("kv"), Var("d")), v_layout)
    else:
        q_idx, k_idx, v_idx = access_indices

    # s = Σ_dd Q·K — the inner contraction as a high-level contraction structural node, the
    # ``source`` of the streaming kv :class:`Fold`. Per-cell scalar (``TilePlan()``): the
    # redundant one-dot-per-output-element score. Its output axes are the score matrix ``[m, kv]``.
    # The scale / mask reads ``sacc``.
    q_edge, k_edge = operand_edges
    score_contraction = Fold.contraction(
        k_axis=Axis(name="dd", extent=Dim(head_dim)),
        a=q_edge if q_edge is not None else Load(name="q_e", input=q_buf, index=q_idx),
        channels=(Channel(b=k_edge if k_edge is not None else Load(name="k_e", input=k_buf, index=k_idx), acc="sacc"),),
    )
    score_post = [
        Load(name="scale_c", input="_flash_scale", index=()),
        Assign(name="s", op="multiply", args=("sacc", "scale_c")),
    ]
    score_name = "s"
    if mask_buf is not None:
        # Additive bias: leading dims broadcast (indexed to 0), trailing two are the
        # query row m and the streaming key kv.
        mask_idx = (*(Literal(0, "int") for _ in mask_shape[:-2]), Var("m"), Var("kv"))
        score_post += [Load(name="mask_e", input=mask_buf, index=mask_idx), Assign(name="s_masked", op="add", args=(score_name, "mask_e"))]
        score_name = "s_masked"
    if causal or window is not None:
        score_post.append(Load(name="ninf_c", input="_flash_ninf", index=()))
    if causal:
        # Causal mask: keep the score where key ≤ query (kv ≤ m), else −inf. Coexists with an
        # explicit bias on a stamped SDPA (bit-neutral there — the bias already masks the region).
        score_post.append(
            Select(
                name="s_causal",
                branches=(
                    SelectBranch(value=score_name, select=BinaryExpr("<=", Var("kv"), Var("m"))),
                    SelectBranch(value="ninf_c", select=Literal(1, "int")),
                ),
            )
        )
        score_name = "s_causal"
    if window is not None:
        # Sliding-window band: keep the score where kv > m − W, else −inf. The stream start
        # derives from this predicate (the band analogue of the causal stream end).
        score_post.append(
            Select(
                name="s_banded",
                branches=(
                    SelectBranch(value=score_name, select=BinaryExpr(">", Var("kv"), BinaryExpr("-", Var("m"), Literal(window, "int")))),
                    SelectBranch(value="ninf_c", select=Literal(1, "int")),
                ),
            )
        )
        score_name = "s_banded"
    # The (m,l,O) streaming fold over kv, λ-SPELLED (step 7 — the composed step dissolved): the
    # ``lift`` is ``λ(kv, sacc, v_e) → (score, 1, v_e)`` — its body the scale / mask stmts binding
    # ``score_name``, ι spelled in the results (the singleton state) — and the ``operands`` are the
    # ``Σ_dd Q·K`` score fold (an inline-node edge, hoisted; it reads the enclosing ``kv`` var —
    # legal, never state) and the value ``Load``. The flat ``(init, combine)`` is the TRUE
    # exp-family ⊕ over the real state names; formation derives the carrier and the DERIVED
    # blocked evaluation reproduces the retired step material exactly — the score edge at the
    # head, the lift body, then the generated merge with the PV contraction synthesized
    # (``ir._twisted_derived_step``), so the lowered nest is byte-identical.
    names = ("m_i", "l_i", "O_i")
    other = tuple(f"{n}__o" for n in names)
    combine = Lambda(params=names + other, body=Body(exp_combine_states(names, other)), results=names)
    lift = Lambda(params=("kv", "sacc", "v_e"), body=Body(tuple(score_post)), results=(score_name, 1.0, "v_e"))
    reduction = Fold(
        axis=Axis(name="kv", extent=s_k),
        operands=(score_contraction, Load(name="v_e", input=v_buf, index=v_idx)),
        lift=lift,
        init=(float("-inf"), 0.0, 0.0),
        combine=combine,
    )
    # φ projection: normalize the streamed (unnormalized) output by the LSE denominator —
    # O_i / l_i after the kv loop, the zero-axis ``Fold`` body over the reduction ``source``. When the caller
    # supplies ``out_store`` (``(buffer, index)``), the fragment ALSO carries the explicit output
    # ``Write`` at that index — the output-layout-aware store (:func:`_out_store_index`) that
    # reproduces the root's real buffer rank / transpose / broadcast dims — as a boundary
    # :class:`~emmy.compiler.ir.tile.ir.Store` (1q: never inside the term; ``TileOp.stores``).
    # Reconstituted into the tail it still short-circuits the materializer's bare grid-order
    # ``with_store`` glue (``has_write``). Absent it, the glue
    # writes the bare grid cell (the head-major identity path — isolated fragments, tests).
    proj: tuple[Stmt, ...] = (Assign(name="O_i__proj", op="divide", args=("O_i", "l_i")),)
    stores: tuple = ()
    if out_store is not None:
        out_buf, out_idx = out_store
        stores = (Store(write=Write(output=out_buf, index=out_idx, value="O_i__proj")),)
    return Fold.projection(body=Body(proj), operands=(reduction,)), stores


# --------------------------------------------------------------------------- #
# Recognition — match a softmax-then-P@V kernel (+ its clean scaled-QK producer)
# and emit the fused flash fragment. Called from ``lowering/tile/010_recognize``.
# --------------------------------------------------------------------------- #


def _is_sum(accum: Accum) -> bool:
    """The accum is the semiring additive reduce ``⊕`` (``add`` / ``sum``)."""
    return accum.op.reduce_canon == "add"


def _is_rowmax(accum: Accum) -> bool:
    """The accum is the softmax rowmax reduce (``maximum`` / ``amax``)."""
    return accum.op.reduce_canon == "maximum"


def _accum_loops(op: LoopOp) -> list[Loop]:
    """Loops whose immediate body folds an ``Accum`` (the matmul / softmax-stat reduces)."""
    return [lp for lp in op.body.iter_of_type(Loop) if any(isinstance(s, Accum) for s in lp.body)]


def _var_at(index: tuple, pos: int) -> str | None:
    """The plain axis-var name at ``index[pos]``, or None (literal / affine)."""
    if abs(pos) > len(index):
        return None
    e = index[pos]
    return e.name if isinstance(e, Var) else None


def _slot_of(index: tuple, name: str) -> int | None:
    """The position of the plain ``Var(name)`` in ``index``, or None."""
    for i, e in enumerate(index):
        if isinstance(e, Var) and e.name == name:
            return i
    return None


def _canon_shape(shape: tuple, layout: tuple[int, int] | None) -> tuple:
    """``shape`` reordered to the canonical ``(batch…, seq, last)`` given the operand's
    ``(seq_pos, last_pos)`` layout — identity for ``None`` (already head-major)."""
    if layout is None:
        return tuple(shape)
    s_pos, l_pos = layout
    rest = tuple(d for i, d in enumerate(shape) if i not in (s_pos, l_pos))
    return (*rest, shape[s_pos], shape[l_pos])


def _permute_idx(comps: tuple, layout: tuple[int, int] | None) -> tuple:
    """Place the canonical index components ``(batch…, seq, last)`` into the operand's own
    slot order — the trace layout a fused transpose baked into the load index (HF's
    per-layer Q/K/V arrive ``(b, s, h, d)``, seq-major). ``None`` = head-major identity."""
    if layout is None:
        return comps
    s_pos, l_pos = layout
    idx: list = [None] * len(comps)
    idx[s_pos], idx[l_pos] = comps[-2], comps[-1]
    rest = iter(comps[:-2])
    for i in range(len(comps)):
        if idx[i] is None:
            idx[i] = next(rest)
    return tuple(idx)


def _out_store_index(out_index: tuple, out_shape: tuple, grid: tuple) -> tuple | None:
    """The output ``Write`` index for the flash store — the OUTPUT counterpart to :func:`_permute_idx`.

    The store must match the output buffer's REAL rank + layout, not the bare grid order: the model's
    sdpa output can be transposed (a fused ``transpose``) and can carry size-1 broadcast / unsqueeze
    dims (e.g. HF's ``[b, h, s, 1, d]``), so a bare 4-var ``(batch…, m, d)`` grid write mis-strides
    against a 5-D buffer (all the outputs alias, the rest stays uninitialized → NaN downstream). This
    reproduces the ROOT kernel's own output ``Write`` (``out_index`` over ``out_shape``), substituting
    each per-axis loop ``Var`` with the fragment's grid axis of matching dim extent, and KEEPING every
    ``Literal`` slot (size-1 batch dims and pure broadcast dims — index 0). ``None`` when a ``Var``
    slot's extent has no unused grid axis (an un-reproducible layout — the caller declines to cut)."""
    from collections import defaultdict  # noqa: PLC0415

    buckets: dict = defaultdict(list)
    for ax in grid:
        buckets[ax.extent].append(Var(ax.name))
    used: dict = defaultdict(int)
    idx: list = []
    for pos, e in enumerate(out_index):
        if isinstance(e, Var):
            ext = out_shape[pos]
            cands = buckets.get(ext, ())
            k = used[ext]
            if k >= len(cands):
                return None  # no unused grid axis for this dim — decline (fall back to cut)
            used[ext] += 1
            idx.append(cands[k])
        else:
            idx.append(e)  # a Literal slot (size-1 batch / broadcast dim) — index 0, kept verbatim
    return tuple(idx)


def _extract_qk(xnode: Node) -> tuple[tuple[str, tuple[int, int]], tuple[str, tuple[int, int]]] | None:
    """From the scaled-QK^T producer of the score buffer, return
    ``((q_id, q_layout), (k_id, k_layout))``. Q vs K by index (fusion reorders the
    operands): the operand indexed by the score's row (M) var is Q — at ANY index slot,
    not just -2: a fused transpose bakes the trace layout into the load index (HF's
    per-layer Q/K arrive ``(b, s, h, d)``, seq-major). Each operand's ``(seq_pos,
    last_pos)`` layout rides along so the fragment builder emits loads in the operand's
    own slot order."""
    op = xnode.op
    if not isinstance(op, LoopOp):
        return None
    writes = [s for s in op.body.iter() if isinstance(s, Write)]
    if len(writes) != 1:
        return None
    m_var = _var_at(writes[0].index, -2)  # score [..., M (query), N (kv)] → row var
    n_var = _var_at(writes[0].index, -1)  # … and the streamed key (column) var
    if m_var is None or n_var is None:
        return None
    for lp in _accum_loops(op):
        loads = [s for s in lp.body if isinstance(s, Load)]
        accs = [s for s in lp.body if isinstance(s, Accum)]
        muls = [s for s in lp.body if isinstance(s, Assign) and s.op.semiring_product]
        if len(loads) == 2 and len(accs) == 1 and _is_sum(accs[0]) and muls:
            q = k = None
            for ld in loads:
                dd_pos = _slot_of(ld.index, lp.axis.name)
                if dd_pos is None:
                    continue
                q_pos = _slot_of(ld.index, m_var)
                k_pos = _slot_of(ld.index, n_var)
                if q_pos is not None:
                    q = (ld.input, (q_pos, dd_pos))
                elif k_pos is not None:
                    k = (ld.input, (k_pos, dd_pos))
            if q is not None and k is not None:
                return q, k
    return None


def _substitute_access(index: tuple, mapping: dict[str, Var], allowed: set[str]) -> tuple | None:
    """Rename an affine load index onto flash's canonical axes, rejecting stray trace vars."""
    result = tuple(expr.substitute(mapping) for expr in index)
    if any(expr.free_vars() - allowed for expr in result):
        return None
    return result


def _extract_packed_qkv(
    xnode: Node,
    root: Node,
    v_id: str,
) -> tuple[str, tuple, tuple, tuple, tuple, tuple, tuple] | None:
    """Recover exact Q/K/V accesses when three logical views share one packed backing buffer.

    A load-time-concatenated DiT projection is physically ``[B, S, 3H]`` while SDPA sees logical
    ``[B, heads, S, D]`` views. Their loop-IR loads retain the exact affine addressing:
    ``packed[b, seq, head*D + d + {0,H,2H}]``. The ordinary layout descriptor cannot represent
    either the flattened ``head*D+d`` coordinate or the QKV base offset, so carry those load
    indices directly after renaming trace axes to flash's canonical ``b*/m/kv/dd/d`` axes.

    Deliberately limited to equal-head packed Q=K=V. GQA and separate buffers keep using the
    established layout path.
    """
    if not isinstance(xnode.op, LoopOp) or not isinstance(root.op, LoopOp):
        return None
    score_writes = [s for s in xnode.op.body.iter() if isinstance(s, Write)]
    if len(score_writes) != 1:
        return None
    score_write = score_writes[0]
    m_var, n_var = _var_at(score_write.index, -2), _var_at(score_write.index, -1)
    if m_var is None or n_var is None:
        return None

    q_load = k_load = None
    head_dim = None
    for lp in _accum_loops(xnode.op):
        loads = [s for s in lp.body if isinstance(s, Load) and lp.axis.name in {v for e in s.index for v in e.free_vars()}]
        accs = [s for s in lp.body if isinstance(s, Accum)]
        if len(loads) != 2 or len(accs) != 1 or not _is_sum(accs[0]):
            continue
        q_load = next(
            (
                ld
                for ld in loads
                if m_var in {v for e in ld.index for v in e.free_vars()} and n_var not in {v for e in ld.index for v in e.free_vars()}
            ),
            None,
        )
        k_load = next(
            (
                ld
                for ld in loads
                if n_var in {v for e in ld.index for v in e.free_vars()} and m_var not in {v for e in ld.index for v in e.free_vars()}
            ),
            None,
        )
        if q_load is not None and k_load is not None:
            head_dim = lp.axis.extent
            dd_var = lp.axis.name
            break
    if q_load is None or k_load is None or q_load.input != k_load.input or q_load.input != v_id:
        return None

    score_shape = tuple(xnode.output.shape)
    if len(score_shape) < 2 or len(score_write.index) != len(score_shape):
        return None
    batch_shape = score_shape[:-2]
    batch_mapping: dict[str, Var] = {}
    for i, expr in enumerate(score_write.index[:-2]):
        if isinstance(expr, Var):
            batch_mapping[expr.name] = Var(f"b{i}")
        elif expr.free_vars():
            return None
    q_mapping = {**batch_mapping, m_var: Var("m"), dd_var: Var("dd")}
    k_mapping = {**batch_mapping, n_var: Var("kv"), dd_var: Var("dd")}
    q_allowed = {*(f"b{i}" for i in range(len(batch_shape))), "m", "dd"}
    k_allowed = {*(f"b{i}" for i in range(len(batch_shape))), "kv", "dd"}
    q_idx = _substitute_access(q_load.index, q_mapping, q_allowed)
    k_idx = _substitute_access(k_load.index, k_mapping, k_allowed)
    if q_idx is None or k_idx is None:
        return None

    out_writes = [s for s in root.op.body.iter() if isinstance(s, Write)]
    if len(out_writes) != 1:
        return None
    out_write = out_writes[0]
    d_var = _var_at(out_write.index, -1)
    if d_var is None:
        return None
    v_load = None
    kv_var = None
    for lp in _accum_loops(root.op):
        if not any(isinstance(s, Accum) and s.name == out_write.value and _is_sum(s) for s in lp.body):
            continue
        v_load = next(
            (
                ld
                for ld in lp.body
                if isinstance(ld, Load) and ld.input == v_id and lp.axis.name in {v for e in ld.index for v in e.free_vars()}
            ),
            None,
        )
        if v_load is not None:
            kv_var = lp.axis.name
            break
    if v_load is None or kv_var is None:
        return None

    axis_extents = {lp.axis.name: lp.axis.extent for lp in root.op.body.iter_of_type(Loop)}
    if d_var not in axis_extents:
        return None
    access_vars = {v for expr in v_load.index for v in expr.free_vars()} - {kv_var, d_var}
    ordered_access_vars: list[str] = []
    for expr in out_write.index:
        for name in sorted(expr.free_vars()):
            if name in access_vars and name not in ordered_access_vars:
                ordered_access_vars.append(name)
    if set(ordered_access_vars) != access_vars:
        return None

    v_mapping: dict[str, Var] = {kv_var: Var("kv"), d_var: Var("d")}
    available = list(range(len(batch_shape)))
    for name in ordered_access_vars:
        extent = axis_extents.get(name)
        choices = [i for i in available if extent is not None and batch_shape[i] == extent]
        if not choices:
            return None
        pos = choices[0]
        available.remove(pos)
        v_mapping[name] = Var(f"b{pos}")
    v_allowed = {*(f"b{i}" for i in range(len(batch_shape))), "kv", "d"}
    v_idx = _substitute_access(v_load.index, v_mapping, v_allowed)
    if v_idx is None:
        return None

    q_shape = (*batch_shape, score_shape[-2], head_dim)
    k_shape = (*batch_shape, score_shape[-1], head_dim)
    v_shape = (*batch_shape, score_shape[-1], axis_extents[d_var])
    return q_load.input, q_idx, k_idx, v_idx, q_shape, k_shape, v_shape


def _extract_scale(graph: Graph, xnode: Node) -> tuple[float | None] | None:
    """The score producer's scale value, from its multiply by a scalar-constant Load
    (the SDPA decomposition's ``{name}_scale`` constant — ``qk·scale`` is the only
    multiply-by-constant a clean producer carries). Returns ``(value,)``, ``(None,)``
    when the producer has no such multiply (hand-built IR — the builder's
    ``1/sqrt(d)`` default applies), or ``None`` when ambiguous (two distinct
    constant multiplies — decline the fuse rather than guess which is the scale)."""
    const_loads: dict[str, float] = {}
    for s in xnode.op.body.iter():
        if isinstance(s, Load):
            src = graph.producer(s.input)
            if src is not None and isinstance(src.op, ConstantOp) and src.op.value is not None:
                for nm in s.names:
                    const_loads[nm] = float(src.op.value)
    values = {
        const_loads[a]
        for s in xnode.op.body.iter()
        if isinstance(s, Assign) and s.op.name == "multiply"
        for a in s.args
        if a in const_loads
    }
    if len(values) > 1:
        return None
    return (values.pop(),) if values else (None,)


def _def(stmts: tuple[Stmt, ...], name: str) -> Stmt | None:
    """The statement in ``stmts`` (one loop body, flat) that defines SSA ``name``."""
    for s in stmts:
        if isinstance(s, Load) and name in s.names:
            return s
        if isinstance(s, (Assign, Select)) and s.name == name:
            return s
    return None


def _is_loopop(graph: Graph, buf: str) -> bool:
    node = graph.producer(buf)
    return node is not None and isinstance(node.op, LoopOp)


def _coord_select_kind(mdef: Select, kv: str, m: str) -> tuple[bool, int | None] | None:
    """Classify a mask-value ``Select`` (keep-branch first, as the decomposition emits) by its
    keep predicate over the streaming key var ``kv`` AGAINST the score's query-row var ``m``:
    ``kv ≤ m`` → causal ``(True, None)``; ``kv > m − W`` → sliding-window band ``(False, W)``.
    ``None`` for any other predicate — an unrecognized Select declines the fuse rather than
    silently classifying causal, and a coordinate compare against any var other than the row
    var (a head / batch var in hand-written IR) is exactly such an unrecognized predicate:
    ``_flash_op`` re-synthesizes the canonical ``kv ≤ m`` Selects, so classifying it would
    silently rewrite the mask's semantics."""
    keep = mdef.branches[0].select
    if not (isinstance(keep, BinaryExpr) and isinstance(keep.left, Var) and keep.left.name == kv):
        return None
    if keep.op == "<=" and isinstance(keep.right, Var) and keep.right.name == m:
        return (True, None)
    if (
        keep.op == ">"
        and isinstance(keep.right, BinaryExpr)
        and keep.right.op == "-"
        and isinstance(keep.right.left, Var)
        and keep.right.left.name == m
        and isinstance(keep.right.right, Literal)
    ):
        return (False, int(keep.right.right.value))
    return None


def _classify_rowmax(graph: Graph, lp: Loop) -> tuple[str, bool, int | None, str | None] | None:
    """For the rowmax reduce loop, return ``(score_buf, causal, window, mask_buf)``; else None.
    The value folded by the ``maximum`` Accum is the bare score Load or a CHAIN of ``add``\\ s on
    it — each add's other operand a mask: a coord ``Select`` (the causal keep ``kv ≤ m`` or the
    sliding-window band keep ``kv > m − W``) or a buffer ``Load`` (the explicit additive bias).
    A stamped SDPA carries coord masks alongside the bias; more than one bias declines."""
    max_accs = [s for s in lp.body if isinstance(s, Accum) and _is_rowmax(s)]
    if len(max_accs) != 1:
        return None
    causal, window, mask_buf = False, None, None
    coord_selects: list[Select] = []
    cur = _def(lp.body, max_accs[0].value)
    while isinstance(cur, Assign) and cur.op.name == "add" and len(cur.args) == 2:
        a, b = cur.args
        adef, bdef = _def(lp.body, a), _def(lp.body, b)
        # The mask side is the Select / non-score Load; the other side continues the chain.
        # Coord Selects are only COLLECTED here — classification needs the score's row var,
        # known once the chain bottoms out at the score Load below.
        nxt = None
        for sdef, mdef in ((adef, bdef), (bdef, adef)):
            if isinstance(mdef, Select):
                coord_selects.append(mdef)
                nxt = sdef
                break
            if isinstance(mdef, Load) and not _is_loopop(graph, mdef.input):
                if mask_buf is not None:
                    return None
                mask_buf = mdef.input
                nxt = sdef
                break
        if nxt is None:
            return None
        cur = nxt
    if isinstance(cur, Load) and _is_loopop(graph, cur.input):
        # Classify each coord Select against the score's own vars (``score[..., m, kv]`` — the
        # module's score-layout convention, as in ``_extract_qk``); any mismatch declines.
        m_var = _var_at(cur.index, -2)
        if coord_selects and (m_var is None or _var_at(cur.index, -1) != lp.axis.name):
            return None
        for mdef in coord_selects:
            kind = _coord_select_kind(mdef, lp.axis.name, m_var)
            if kind is None:
                return None
            if kind[0]:
                causal = True
            else:
                window = kind[1]
        return cur.input, causal, window, mask_buf
    return None


@dataclass(frozen=True)
class _FlashMatch:
    """The one flash recognition result, independent of where fusion left its boundary.

    ``inline`` is ``None`` for the original score-buffer → softmax/P×V form. When the generic
    loop merger has instead folded the repeated score reads into the softmax producer, it is
    ``(m_var, kv_var, rowmax_loop)`` and ``score`` names that probability producer itself.
    Both forms continue through the same Q/K extraction, eligibility, and fragment builder.
    """

    score: Node
    v_id: str
    causal: bool
    window: int | None
    mask_buf: str | None
    inline: tuple[str, str, Loop] | None = None
    v_inline: tuple[Loop, tuple[Stmt, ...], str, Load] | None = None


@dataclass(frozen=True)
class _InlineQk:
    """Canonical computed Q/K edges recovered from one inlined score cell.

    ``q_id`` / ``k_id`` are declaration anchors only; the edges retain every exact external
    load. Their logical shapes come from the SDPA iteration space rather than either anchor's
    physical storage shape, which may be one packed projection buffer.
    """

    q_id: str
    k_id: str
    q_edge: Load | Fold
    k_edge: Load | Fold
    head_dim: Dim


def _pv_loads(graph: Graph, node: Node) -> tuple[Load, Load, Loop, tuple[Stmt, ...], str] | None:
    """Return the probability load and V cone for a bare P×V root.

    The probability side is identified by provenance, not operand order: its producer is a
    ``LoopOp`` containing ``exp``. Its product argument may include the consumer-side reciprocal
    normalization (the sliding-window split leaves ``Σexp`` beside P×V). The V side may be a
    direct load or a closed pure-map cone; :func:`try_flash` factors the latter into a canonical
    feeder workspace so the expectation operand stays materialized and stageable.
    """
    if not isinstance(node.op, LoopOp):
        return None
    writes = [s for s in node.op.body.iter() if isinstance(s, Write)]
    if len(writes) != 1:
        return None
    d_var = _var_at(writes[0].index, -1)
    if d_var is None:
        return None
    for lp in _accum_loops(node.op):
        acc = next(
            (s for s in lp.body if isinstance(s, Accum) and s.name == writes[0].value and _is_sum(s)),
            None,
        )
        if acc is None:
            continue
        product = _def(lp.body, acc.value)
        if not isinstance(product, Assign) or not product.op.semiring_product or len(product.args) != 2:
            continue
        sides = [(arg, map_cone(list(lp.body), arg)) for arg in product.args]
        if any(not cone for _arg, cone in sides):
            continue
        probs = [
            (arg, cone, ld)
            for arg, cone in sides
            for ld in cone
            if isinstance(ld, Load)
            and (p := graph.producer(ld.input)) is not None
            and isinstance(p.op, LoopOp)
            and any(isinstance(s, Assign) and s.op.name == "exp" for s in p.op.body.iter())
        ]
        if len(probs) != 1:
            continue
        prob_arg, _prob_cone, prob = probs[0]
        value_side = next((side for side in sides if side[0] != prob_arg), None)
        if value_side is None:
            continue
        value_arg, value_cone = value_side
        value_loads = [
            ld
            for ld in value_cone
            if isinstance(ld, Load) and _slot_of(ld.index, lp.axis.name) is not None and _slot_of(ld.index, d_var) is not None
        ]
        if len(value_loads) != 1:
            continue
        return prob, value_loads[0], lp, tuple(value_cone), value_arg
    return None


def _classify_inline_rowmax(lp: Loop, m_var: str) -> tuple[bool, int | None, str | None] | None:
    """Classify the score/mask chain when Q×K is nested directly under rowmax.

    Unlike :func:`_classify_rowmax`, the chain bottoms out at ``scale * <sum-Accum>`` rather
    than a score-buffer ``Load``. The returned masks have the same meaning; Q/K recovery is a
    separate structural check over the nested contraction.
    """
    max_accs = [s for s in lp.body if isinstance(s, Accum) and _is_rowmax(s)]
    if len(max_accs) != 1:
        return None
    causal, window, mask_buf = False, None, None
    coord_selects: list[Select] = []
    cur = _def(lp.body, max_accs[0].value)
    while isinstance(cur, Assign) and cur.op.name == "add" and len(cur.args) == 2:
        a, b = cur.args
        adef, bdef = _def(lp.body, a), _def(lp.body, b)
        nxt = None
        for sdef, mdef in ((adef, bdef), (bdef, adef)):
            if isinstance(mdef, Select):
                coord_selects.append(mdef)
                nxt = sdef
                break
            if isinstance(mdef, Load):
                if mask_buf is not None:
                    return None
                mask_buf = mdef.input
                nxt = sdef
                break
        if nxt is None:
            return None
        cur = nxt

    sum_names = {s.name for s in lp.body.iter() if isinstance(s, Accum) and _is_sum(s)}
    if not (isinstance(cur, Assign) and cur.op.name == "multiply" and len(cur.args) == 2 and sum(a in sum_names for a in cur.args) == 1):
        return None
    for select in coord_selects:
        kind = _coord_select_kind(select, lp.axis.name, m_var)
        if kind is None:
            return None
        if kind[0]:
            causal = True
        else:
            window = kind[1]
    return causal, window, mask_buf


def _recognize(graph: Graph, node: Node) -> _FlashMatch | None:
    """Recognize the supported loop-fusion spellings of the existing flash unit.

    The original spelling has rowmax/softmax and P×V in ``node`` and reads a materialized
    scaled-QK score buffer. Gate-free loop fusion produces the other spelling: ``node`` is the
    bare P×V contraction and its probability input contains rowmax, softmax, and three repeated
    inlined Q×K contractions. Small problems may fuse that producer into the root as well. This
    function extends the same recognizer across those moved boundaries; it does not create a
    second flash path.
    """
    op = node.op
    if not isinstance(op, LoopOp):
        return None
    body = op.body
    writes = [s for s in body.iter() if isinstance(s, Write)]
    if len(writes) != 1:
        return None
    out_write = writes[0]
    if any(isinstance(s, Assign) and s.op.name == "exp" for s in body.iter()):
        x_buf: str | None = None
        causal, window = False, None
        mask_buf: str | None = None
        for lp in _accum_loops(op):
            cls = _classify_rowmax(graph, lp)
            if cls is not None:
                x_buf, causal, window, mask_buf = cls
                break
        if x_buf is not None:
            v_buf: str | None = None
            for lp in _accum_loops(op):
                if not any(isinstance(s, Accum) and s.name == out_write.value and _is_sum(s) for s in lp.body):
                    continue
                others = {s.input for s in lp.body if isinstance(s, Load)} - {x_buf, mask_buf}
                if len(others) == 1:
                    v_buf = next(iter(others))
            score = graph.producer(x_buf)
            if v_buf is not None and score is not None:
                return _FlashMatch(score, v_buf, causal, window, mask_buf)

        # A small enough SDPA can cross the final boundary too: the root then contains all
        # three repeated QK contractions, softmax, and P×V. Recover the direct V side of the
        # output contraction and use the rowmax copy as the canonical score spelling.
        m_var, d_var = _var_at(out_write.index, -2), _var_at(out_write.index, -1)
        if m_var is not None and d_var is not None:
            inline_pv: list[tuple[str, Loop]] = []
            for lp in _accum_loops(op):
                acc = next(
                    (s for s in lp.body if isinstance(s, Accum) and s.name == out_write.value and _is_sum(s)),
                    None,
                )
                if acc is None:
                    continue
                product = _def(lp.body, acc.value)
                if not isinstance(product, Assign) or not product.op.semiring_product or len(product.args) != 2:
                    continue
                direct = [_def(lp.body, arg) for arg in product.args]
                values = [
                    ld
                    for ld in direct
                    if isinstance(ld, Load) and _slot_of(ld.index, lp.axis.name) is not None and _slot_of(ld.index, d_var) is not None
                ]
                if len(values) == 1:
                    inline_pv.append((values[0].input, lp))
            if len(inline_pv) == 1:
                v_buf, pv_loop = inline_pv[0]
                matches = []
                for rowmax in _accum_loops(op):
                    if rowmax.axis.extent != pv_loop.axis.extent:
                        continue
                    masks = _classify_inline_rowmax(rowmax, m_var)
                    if masks is not None:
                        matches.append((rowmax, masks))
                if len(matches) == 1:
                    rowmax, (causal, window, mask_buf) = matches[0]
                    return _FlashMatch(node, v_buf, causal, window, mask_buf, (m_var, rowmax.axis.name, rowmax))

    pv = _pv_loads(graph, node)
    if pv is None:
        return None
    prob, value, kv_loop, value_cone, value_arg = pv
    score = graph.producer(prob.input)
    if score is None or not isinstance(score.op, LoopOp):
        return None
    kv_pos = _slot_of(prob.index, kv_loop.axis.name)
    if kv_pos is None or kv_pos == 0:
        return None
    m_var = _var_at(prob.index, kv_pos - 1)
    if m_var is None or _slot_of(out_write.index, m_var) is None:
        return None
    matches = []
    for rowmax in _accum_loops(score.op):
        if rowmax.axis.extent != kv_loop.axis.extent:
            continue
        masks = _classify_inline_rowmax(rowmax, m_var)
        if masks is not None:
            matches.append((rowmax, masks))
    if len(matches) != 1:
        return None
    rowmax, (causal, window, mask_buf) = matches[0]
    v_inline = (kv_loop, value_cone, value_arg, value) if len(value_cone) > 1 else None
    return _FlashMatch(score, value.input, causal, window, mask_buf, (m_var, rowmax.axis.name, rowmax), v_inline)


def _path_to_loop(body: Body, target: Loop) -> tuple[tuple[Body, Stmt], ...] | None:
    """The enclosing ``(body, child)`` chain down to ``target`` (identity-based)."""
    for stmt in body:
        if stmt is target:
            return ((body, stmt),)
        for nested in stmt.nested():
            suffix = _path_to_loop(nested, target)
            if suffix is not None:
                return ((body, stmt), *suffix)
    return None


def _qk_cell(rowmax: Loop, m_var: str, kv_var: str) -> tuple[Loop, list[Stmt], str, Load, list[Stmt], str, Load] | None:
    """Recover one Q×K contraction cell nested under ``rowmax``.

    Returns the contraction loop plus each product argument's pure backward cone, value name,
    and one declaration anchor. Classification uses each WHOLE cone's axis dependence: a packed
    affine view, coordinate ``Select``, normalization weight, and rotary inputs may contribute
    several sequence-bearing loads while still defining one closed scalar Q or K value. A
    K-normalization statistic is not mistaken for Q×K because it has no query-indexed side.
    """

    def expr_vars(stmts: list[Stmt]) -> set[str]:
        return {v for st in stmts for expr in st.exprs() for v in expr.free_vars()}

    found = []
    for lp in rowmax.body.iter_of_type(Loop):
        accs = [s for s in lp.body if isinstance(s, Accum) and _is_sum(s)]
        if len(accs) != 1:
            continue
        product = _def(lp.body, accs[0].value)
        if not isinstance(product, Assign) or not product.op.semiring_product or len(product.args) != 2:
            continue
        sides = []
        for value in product.args:
            cone = map_cone(list(lp.body), value)
            if not cone:
                break
            vars_ = expr_vars(cone)
            leaves = [
                s
                for s in cone
                if isinstance(s, Load)
                and lp.axis.name in {v for expr in s.index for v in expr.free_vars()}
                and ({m_var, kv_var} & {v for expr in s.index for v in expr.free_vars()})
            ]
            if not leaves or lp.axis.name not in vars_:
                break
            sides.append((cone, value, leaves[0], m_var in vars_, kv_var in vars_))
        if len(sides) != 2:
            continue
        q = next((s for s in sides if s[3] and not s[4]), None)
        k = next((s for s in sides if s[4] and not s[3]), None)
        if q is not None and k is not None:
            found.append((lp, q[0], q[1], q[2], k[0], k[1], k[2]))
    return found[0] if len(found) == 1 else None


def _rewrite_members(
    members: list[Stmt],
    *,
    prefix: str,
    sigma: Sigma,
    axis_name: str | None = None,
    old_axis: str | None = None,
    defined_names: set[str] | None = None,
) -> list[Stmt]:
    """Canonicalize axes and uniquify SSA names for one inline operand edge."""
    defs = defined_names or {name for stmt in members for name in Body((stmt,)).definitions}

    def rename(name: str) -> str:
        return f"{prefix}_{name}" if name in defs else name

    def axis_fn(axis: Axis) -> Axis:
        if old_axis is not None and axis.name == old_axis:
            return Axis(axis_name, axis.extent, window=axis.window)
        return axis

    return [stmt.rewrite(rename, sigma, axis_fn) for stmt in members]


def _inline_operand_edge(
    op: LoopOp,
    qk_loop: Loop,
    path: tuple[tuple[Body, Stmt], ...],
    cone: list[Stmt],
    value: str,
    *,
    prefix: str,
    outer_sigma: dict[str, Var],
) -> Load | Fold | None:
    """Turn one inlined Q/K value into its existing materialized/computed operand edge.

    A direct load stays a load. A pure map becomes a zero-axis computed edge. If the cell reads
    one enclosing statistic, its dependence cone must contain exactly one λ-representable reduce;
    that reduce and projection are stored with :func:`make_cone`, the same representation used by
    the general contraction recognizer for RMSNorm→linear.
    """
    cell_defs = {name for stmt in cone for name in stmt.defines()}
    pending = {name for stmt in cone for name in stmt.deps() if name not in cell_defs}
    axes = set(op.body.axis_names)
    pending -= axes
    groups: list[tuple[Stmt, ...]] = []
    for body, child in reversed(path):
        if not pending:
            break
        prefix_body = body[: body.index(child)]
        resolved = prefix_body.backward_cone(pending)
        if resolved.members:
            groups.append(resolved.members)
        pending = set(resolved.external_reads) - axes
    if pending:
        return None
    members = [stmt for group in reversed(groups) for stmt in group]
    loops = [stmt for stmt in members if isinstance(stmt, Loop)]
    if any(not (stmt.pure or isinstance(stmt, Loop)) for stmt in members):
        return None
    if len(loops) > 1:
        return None

    cell_sigma = Sigma({**outer_sigma, qk_loop.axis.name: Var("dd")})
    all_defs = {name for stmt in (*members, *cone) for name in Body((stmt,)).definitions}
    rewritten_cell = _rewrite_members(cone, prefix=prefix, sigma=cell_sigma, defined_names=all_defs)
    renamed_value = f"{prefix}_{value}"
    if not loops:
        if len(rewritten_cell) == 1 and isinstance(rewritten_cell[0], Load) and renamed_value in rewritten_cell[0].names:
            return rewritten_cell[0]
        return Fold.projection(body=Body(tuple(rewritten_cell)))

    stat_loop = loops[0]
    stat_name = f"{prefix}_stat"
    stat_sigma = Sigma({**outer_sigma, stat_loop.axis.name: Var(stat_name)})
    rewritten_stat = _rewrite_members(
        [stat_loop],
        prefix=prefix,
        sigma=stat_sigma,
        axis_name=stat_name,
        old_axis=stat_loop.axis.name,
        defined_names=all_defs,
    )[0]
    assert isinstance(rewritten_stat, Loop)
    stat = fold_from_loop(rewritten_stat)
    if stat is None:
        return None
    sweep_src = [stmt for stmt in members if stmt is not stat_loop]
    sweep = _rewrite_members(sweep_src, prefix=prefix, sigma=Sigma(outer_sigma), defined_names=all_defs)
    return make_cone(rewritten_cell, "dd", stat=stat, sweep=tuple(sweep))


def _extract_inline_qk(
    score: Node,
    root: Node,
    m_var: str,
    kv_var: str,
    rowmax: Loop,
) -> _InlineQk | None:
    """Recover canonical Q/K operand edges from an inlined softmax producer.

    Logical attention axes come from the consumer's iteration space. A physical declaration
    anchor cannot supply them when Q and K are affine views of one packed projection buffer.
    """
    found = _qk_cell(rowmax, m_var, kv_var)
    if found is None:
        return None
    qk_loop, q_cone, q_value, q_load, k_cone, k_value, k_load = found
    writes = [stmt for stmt in root.op.body.iter() if isinstance(stmt, Write)]
    if len(writes) != 1 or len(writes[0].index) != len(root.output.shape) or _var_at(writes[0].index, -2) != m_var:
        return None

    # The root write spells canonical ``(batch…, m, d)``. Its batch/head variables bind Q's
    # logical batch axes; K may carry the same head variable through affine GQA ``head/group``.
    outer_sigma: dict[str, Var] = {m_var: Var("m"), kv_var: Var("kv")}
    batch_rank = len(root.output.shape) - 2
    for i, expr in enumerate(writes[0].index[:-2]):
        if isinstance(expr, Var):
            outer_sigma[expr.name] = Var(f"b{i}")
        elif expr.free_vars():
            return None

    path = _path_to_loop(score.op.body, qk_loop)
    if path is None:
        return None
    q_edge = _inline_operand_edge(score.op, qk_loop, path, q_cone, q_value, prefix="flash_q", outer_sigma=outer_sigma)
    k_edge = _inline_operand_edge(score.op, qk_loop, path, k_cone, k_value, prefix="flash_k", outer_sigma=outer_sigma)
    if q_edge is None or k_edge is None:
        return None
    allowed = {*(f"b{i}" for i in range(batch_rank)), "m", "kv", "dd", "flash_q_stat", "flash_k_stat"}
    for edge in (q_edge, k_edge):
        for stmt in Body(tuple(edge.lower() if isinstance(edge, Fold) else (edge,))).iter():
            if any(expr.free_vars() - allowed for expr in stmt.exprs()):
                return None
    return _InlineQk(q_load.input, k_load.input, q_edge, k_edge, qk_loop.axis.extent)


def _extract_inline_v(root: Node, found: tuple[Loop, tuple[Stmt, ...], str, Load]) -> tuple[tuple[int, int], Fold] | None:
    """Canonicalize a closed pure-map V cone fused into the P×V root."""
    pv_loop, cone, value, v_load = found
    writes = [s for s in root.op.body.iter() if isinstance(s, Write)]
    if len(writes) != 1:
        return None
    d_var = _var_at(writes[0].index, -1)
    if d_var is None:
        return None
    layout = (_slot_of(v_load.index, pv_loop.axis.name), _slot_of(v_load.index, d_var))
    if None in layout:
        return None
    layout = (layout[0], layout[1])
    outer_sigma: dict[str, Var] = {pv_loop.axis.name: Var("kv"), d_var: Var("d")}
    batch_pos = [i for i in range(len(v_load.index)) if i not in layout]
    for i, pos in enumerate(batch_pos):
        expr = v_load.index[pos]
        if isinstance(expr, Var):
            outer_sigma[expr.name] = Var(f"b{i}")
        elif expr.free_vars():
            return None
    defs = {name for stmt in cone for name in stmt.defines()}
    if {name for stmt in cone for name in stmt.deps() if name not in defs} - set(root.op.body.axis_names):
        return None
    rewritten = _rewrite_members(list(cone), prefix="flash_v", sigma=Sigma(outer_sigma), defined_names=defs)
    if f"flash_v_{value}" not in {name for stmt in rewritten for name in Body((stmt,)).definitions}:
        return None
    return layout, Fold.projection(body=Body(tuple(rewritten)))


def _fuse_degraded(root: Node, reason: str) -> None:
    """A softmax-then-P@V kernel was recognized but cannot be certified for the fused flash
    form — it falls back to the un-fused tiers (a separate score producer + softmax-then-P@V).
    Debug-only: most kernels legitimately decline."""
    logger.debug("flash fuse of %r not certifiable (%s); keeping the un-fused kernels", root.id, reason)


def _extract_v_layout(root: Node, v_buf: str) -> tuple[int, int] | None:
    """The V operand's ``(kv_pos, d_pos)`` slot layout, from the P@V accum loop's own V
    ``Load``: ``kv`` is the accum loop's axis, ``d`` the out-write's last var. ``None`` when
    either var is not a plain index slot (an un-permutable V — the caller degrades)."""
    op = root.op
    writes = [s for s in op.body.iter() if isinstance(s, Write)]
    if len(writes) != 1:
        return None
    d_var = _var_at(writes[0].index, -1)
    if d_var is None:
        return None
    for lp in _accum_loops(op):
        if not any(isinstance(s, Accum) and s.name == writes[0].value and _is_sum(s) for s in lp.body):
            continue
        for ld in lp.body:
            if isinstance(ld, Load) and ld.input == v_buf:
                kv_pos, d_pos = _slot_of(ld.index, lp.axis.name), _slot_of(ld.index, d_var)
                if kv_pos is not None and d_pos is not None:
                    return (kv_pos, d_pos)
    return None


def try_flash(graph: Graph, root: Node) -> Graph | None:
    """Recognize SDPA on ``root`` and return the fused flash ``Graph`` fragment (a
    ``TileOp`` holding the flash zero-axis ``Fold`` (a ``TWISTED`` kv loop) + its scale / -inf constants), or ``None``
    if ``root`` is not a recognizable / eligible attention kernel."""
    found = _recognize(graph, root)
    if found is None:
        return None
    score_producer = found.score
    v_id, causal, window, mask_buf = found.v_id, found.causal, found.window, found.mask_buf
    operands = (score_producer.id, v_id, *((mask_buf,) if mask_buf is not None else ()))
    if any(graph.buffer(nid) is None for nid in operands):
        return None

    # The same Q/K extraction follows either accepted boundary. The original score-buffer
    # spelling uses plain/packed affine loads; the gate-free spelling recovers exact materialized
    # or computed operand edges from the QK contraction nested in the softmax producer.
    inline_qk = None
    if found.inline is not None:
        m_var, kv_var, rowmax = found.inline
        inline_qk = _extract_inline_qk(score_producer, root, m_var, kv_var, rowmax)
        qk = None
    else:
        qk = _extract_qk(score_producer)
    packed = _extract_packed_qkv(score_producer, root, v_id) if qk is None and inline_qk is None else None
    if qk is None and inline_qk is None and packed is None:
        _fuse_degraded(root, "score producer's Q/K are not certifiable operand edges")
        return None
    # A mask stranded on the standalone score producer (a coord Select / an additive bias add)
    # would be silently DROPPED by the canonical re-synthesis below. Decline the fuse rather
    # than mis-attend. Inline score masks have already been classified above.
    if found.inline is None and any(
        isinstance(s, Select) or (isinstance(s, Assign) and s.op.name == "add") for s in score_producer.op.body.iter()
    ):
        _fuse_degraded(root, "score producer carries mask stmts the flash re-synthesis cannot keep")
        return None
    access_indices = None
    q_edge = k_edge = None
    materialized_v = None
    if packed is not None:
        q_id, q_idx, k_idx, v_idx, q_shape, k_shape, v_shape = packed
        k_id = v_id = q_id
        q_layout = k_layout = v_layout = None
        access_indices = (q_idx, k_idx, v_idx)
    elif inline_qk is not None:
        q_id, k_id = inline_qk.q_id, inline_qk.k_id
        q_edge, k_edge = inline_qk.q_edge, inline_qk.k_edge
        q_layout = k_layout = None  # computed edges already carry exact canonicalized accesses
        if graph.buffer(q_id) is None or graph.buffer(k_id) is None:
            return None
    else:
        (q_id, q_layout), (k_id, k_layout) = qk
        if graph.buffer(q_id) is None or graph.buffer(k_id) is None:
            return None
    # The producer's actual scale — the flash re-synthesis re-applies it; assuming
    # 1/sqrt(d) here mis-scaled every explicit-``scale=`` SDPA (Gemma-nano's scale=1.0).
    scale = _extract_scale(graph, score_producer)
    if scale is None:
        _fuse_degraded(root, "score producer's scale constant is ambiguous")
        return None
    if packed is None:
        if found.v_inline is not None:
            inline_v = _extract_inline_v(root, found.v_inline)
            if inline_v is None:
                _fuse_degraded(root, "P@V's computed V cone is not closed or plainly indexed")
                return None
            v_layout, materialized_v = inline_v
        else:
            v_layout = _extract_v_layout(root, v_id)
            if v_layout is None:
                _fuse_degraded(root, "P@V's V load is not plainly indexed")
                return None
        # Shapes canonicalize to (batch…, seq, last) per each operand's traced layout — the
        # eligibility predicates and the fragment's grid work on canonical shapes; the LOAD
        # indices permute back to each operand's own slot order (``_permute_idx``).
        v_shape = _canon_shape(graph.buffer(v_id).shape, v_layout)
        if inline_qk is not None:
            # Q's batch/head + query axes are the consumer output's leading dimensions; K
            # shares V's batch/kv-head + sequence axes. Replace only the value dimension with
            # the score contraction's D. Physical q_id/k_id storage may be one packed buffer.
            q_shape = (*root.output.shape[:-1], inline_qk.head_dim)
            k_shape = (*v_shape[:-1], inline_qk.head_dim)
        else:
            q_shape = _canon_shape(graph.buffer(q_id).shape, q_layout)
            k_shape = _canon_shape(graph.buffer(k_id).shape, k_layout)
        if materialized_v is not None:
            v_id = f"{root.output.name}__flash_v"
            v_layout = None  # the recognizer-minted workspace is canonical by construction
    group = gqa_group(q_shape, k_shape)
    if group is None:
        _fuse_degraded(root, "head axis not statically GQA-divisible")
        return None
    mask_shape = graph.buffer(mask_buf).shape if mask_buf is not None else None
    if not flash_shape_eligible(q_shape, k_shape, v_shape, group=group, mask_shape=mask_shape):
        _fuse_degraded(root, "shape not flash-eligible")
        return None
    mask = (mask_buf, mask_shape) if mask_buf is not None else None
    # The root's own output ``Write`` index — the store's target layout (buffer rank, a fused output
    # transpose, size-1 broadcast dims). The fragment reproduces it (``_out_store_index``) instead of
    # a bare grid-order write, which would mis-stride a transposed / higher-rank output → NaN.
    out_writes = [s for s in root.op.body.iter() if isinstance(s, Write)]
    out_index = tuple(out_writes[0].index) if len(out_writes) == 1 else None
    frag = build_flash_frag(
        q_id,
        k_id,
        v_id,
        q_shape,
        k_shape,
        v_shape,
        root.output,
        causal=causal,
        window=window,
        group=group,
        mask=mask,
        layouts=(q_layout, k_layout, v_layout),
        access_indices=access_indices,
        raw_shapes=(
            tuple(graph.buffer(q_id).shape),
            tuple(graph.buffer(k_id).shape),
            tuple(v_shape) if materialized_v is not None else tuple(graph.buffer(v_id).shape),
        ),
        out_index=out_index,
        scale=scale[0],
        operand_edges=(q_edge, k_edge),
        input_tensors={
            nid: tensor
            for nid in (*score_producer.inputs, *root.inputs, q_id, k_id, v_id, *((mask_buf,) if mask_buf is not None else ()))
            if (tensor := graph.buffer(nid)) is not None
        },
        materialized_v=materialized_v,
    )
    if frag is None:
        _fuse_degraded(root, "output layout not reproducible on the flash grid")
    return frag


def fused_producer_ids(graph: Graph, root: Node) -> tuple[str, ...]:
    """The producer node ids a flash fusion of ``root`` would CONSUME — the score ``LoopOp``
    absent from the fused fragment — or ``()`` (``root`` is not a fusable softmax-then-P@V
    consumer). The two-level tuner's slicing reads this: the inner per-op slice for ``root``
    must carry the consumed producer as a REAL node (not a synthetic input boundary), otherwise
    ``try_flash`` can never re-fuse inside the slice and every tune trajectory silently loses the
    fused flash form."""
    frag = try_flash(graph, root)
    if frag is None:
        return ()
    return tuple(
        nid for nid in root.inputs if nid not in frag.nodes and (p := graph.producer(nid)) is not None and isinstance(p.op, LoopOp)
    )


def is_flash_score_producer(graph: Graph, root: Node) -> bool:
    """True iff ``root`` is the **score producer** (the scaled-QK matmul) of a flash kernel
    that :func:`try_flash` would fuse — i.e. some consumer of ``root`` is an eligible
    softmax-then-P@V kernel whose fusion **consumes** ``root``. The general lift in
    ``010_recognize`` defers such a node (leaving it a ``LoopOp``) until its consumer has a
    chance to fuse. The consumed score buffer is the one node absent from the fragment."""
    for consumer in graph.nodes.values():
        if root.id not in consumer.inputs:
            continue
        frag = try_flash(graph, consumer)
        if frag is not None and root.id not in frag.nodes:
            return True
    return False
