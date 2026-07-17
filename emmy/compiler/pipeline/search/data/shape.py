"""``ShapeKey`` — the cheap arithmetic structural identity of a tiled op.

A kernel's full structural signature is the ``S_*`` histogram the
``992_stamp_structural_features`` pass stamps (op counts, dtypes, loop depths,
extents). For grouping, the cold-start ``OfflinePrior`` ranking, and matching a
golden config to a kernel, only the *extents* matter — and those are derivable
arithmetically from a matmul's ``(M, N, K)`` with no compile. ``ShapeKey`` is that
arithmetic handle: ``(free_prod, reduce_max, is_warp)``, the same triple the prior
diagnostics and the per-kernel golden A/B already match on.

It deliberately carries only the extent keys. A *trained* ``OnlinePrior`` regresses
on the full ``S_*`` histogram, so the full set is derived (by compiling the snippet)
and cached on the :class:`~emmy.compiler.pipeline.search.data.sample.Sample`,
not here — see that module's ``compile_s_feats`` path.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShapeKey:
    """Arithmetic extent identity of a matmul-shaped op.

    ``free_prod`` is the product of the **static** output free dims (``M*N``; a
    symbolic axis is excluded, mirroring the ``992`` stamp — see :meth:`from_matmul`),
    ``reduce_max`` the reduce extent (``K``), ``is_warp`` whether it lowers on the
    tensor-core warp tier (any non-fp32 matmul), ``is_dyn`` whether an axis is
    symbolic — the split that keeps a dynamic golden and its static twin apart,
    exactly as ``is_warp`` keeps the fp32/fp16 twins apart. Hashable, so it keys
    ``Dataset`` groupings."""

    free_prod: int
    reduce_max: int
    is_warp: bool
    is_dyn: bool = False
    # The op-kind discriminator for the SWEEP kinds — ``""`` for a plain loop nest
    # (matmul / bare reduce / pointwise: ``S_loop_depth`` equals the axis count), else
    # ``"flash"`` / ``"softmax"`` / ``"rms_norm"`` / ``"fused"``, where a projection sweep
    # rides the reduction (``S_loop_depth < n_free + n_reduce + n_symbolic`` — the same
    # stamp identity the node-store plausibility gate uses). ``"fused"`` is the computed-A
    # contraction (the RMSNorm→linear / gate⊗up megakernel): a warp mma whose A cone carries
    # a statistic (rsqrt) prologue, so it stamps LIKE an ``rms_norm`` sweep but with a SECOND
    # reduce axis — the contraction — beside the statistic reduce (``S_ext_n_reduce_axis >= 2``).
    # Without the discriminator the fused op would join a real ``rms_norm`` (or bare ``mlp_gate_up``
    # matmul) golden even though it is a different kernel family; and its stamp's ``is_warp`` is
    # unreliable (the f32 statistic constants flip ``S_dtype_f32`` on), so ``from_s_features``
    # forces ``is_warp=True`` for the kind — a computed-A contraction is always a warp mma.
    kind: str = ""
    # The largest single static free extent (``S_ext_free_max`` / ``max(M, N)``) — the
    # ASPECT discriminator ``free_prod`` alone lacks: 32×8192 and 512×512 share the
    # product (the decode-twin vs square collision that let ``k_proj_global.s512``
    # silently shadow ``q_proj_global.m32``'s golden), but not the max. Participates
    # only for STATIC plain-nest keys (``kind == "" and not is_dyn``) — every other
    # key class normalizes it to 0 in ``__post_init__`` so the sweep-kind and
    # symbolic-axis joins (whose builders don't all see the same extent list the
    # stamp saw) stay bit-identical to the pre-``free_max`` behavior.
    free_max: int = 0

    def __post_init__(self) -> None:
        if (self.kind or self.is_dyn) and self.free_max:
            object.__setattr__(self, "free_max", 0)

    @classmethod
    def from_matmul(cls, M: int, N: int, K: int, dtype: str, *, dynamic: bool = False) -> ShapeKey:
        """The shape of an ``(M, K) @ (K, N)`` matmul. fp32 lowers on the scalar
        thread tier; fp16 / bf16 on the warp (MMA) tier. ``dynamic`` marks the M
        axis symbolic (the only symbolic-axis golden form today): the key then
        MIRRORS what ``992_stamp_structural_features`` puts on the op — symbolic
        axes are **excluded** from the extent products (``free_prod = N``, not the
        hint-sized ``M*N``) and flagged via ``S_ext_n_symbolic_axis`` — because
        the stamped histogram is the only identity the op side has (it doesn't
        know the hint), so a hint-sized golden key would never join it."""
        return cls(
            free_prod=N if dynamic else M * N,
            reduce_max=K,
            is_warp=dtype != "fp32",
            is_dyn=dynamic,
            free_max=0 if dynamic else max(M, N),
        )

    @classmethod
    def from_s_features(cls, s: dict) -> ShapeKey:
        """The key of a stamped ``S_*`` histogram — an op-group signature from
        ``Dataset.group_by_op``, a prior-reservoir row's knobs, or a ``CudaOp.knobs``
        dict carrying the stamped features. The op-side twin of :meth:`from_matmul`:
        every golden ↔ measured-data join must build BOTH sides through these two
        constructors, so a new key dimension (e.g. the planned symbolic-axis flag)
        lands in one place instead of per join site.

        ``is_warp`` derives from the operand-dtype multiset (``S_dtype_f32``), NOT
        ``S_n_mma``: the stamp pass runs at fusion end, before the tile tier emits
        ``Mma`` stmts, so ``S_n_mma`` is 0.0 on every stamped row — keying on it
        merged the fp32/fp16 twins (and silently dropped fp16 goldens from the
        diagnostics joins), the bug class this single constructor exists to
        prevent.

        ``kind`` classifies the sweep kinds off the stamped histogram (values measured
        by tracing each golden kind's snippet to the stamped op): a sweep op has
        ``S_loop_depth < n_free + n_reduce + n_symbolic`` (the projection sweep shares
        the reduce axis, so the loop nest is shallower than the axis count — matmul,
        bare reduce and pointwise are exactly equal); among sweeps, ``S_pw_rsqrt``
        marks the RMSNorm family, ``S_pw_exp`` the softmax family, and ``S_n_free_loop >= 3``
        (heads x rows x head_dim — histogram counts, so symbolic axes still count)
        separates flash attention from row softmax. Within the rsqrt family a SECOND reduce
        axis (``S_ext_n_reduce_axis >= 2``) means the statistic reduce rides a CONTRACTION —
        the computed-A ``"fused"`` megakernel (RMSNorm→linear / gate⊗up) — where a bare
        RMSNorm has just the one statistic reduce. The fused op's ``is_warp`` is forced True:
        it is a warp mma contraction, but its f32 statistic constants set ``S_dtype_f32``, so
        the dtype-multiset signal would wrongly read scalar. An unrecognized sweep keeps
        ``""`` — conservative: it can only stop a cross-kind join, never invent one."""
        n_axes = s.get("S_ext_n_free_axis", 0) + s.get("S_ext_n_reduce_axis", 0) + s.get("S_ext_n_symbolic_axis", 0)
        kind = ""
        if 0 < s.get("S_loop_depth", 0) < n_axes:
            if s.get("S_pw_rsqrt", 0):
                kind = "fused" if s.get("S_ext_n_reduce_axis", 0) >= 2 else "rms_norm"
            elif s.get("S_pw_exp", 0):
                kind = "flash" if s.get("S_n_free_loop", 0) >= 3 else "softmax"
        return cls(
            free_prod=int(s.get("S_ext_free_prod", 0)),
            reduce_max=int(s.get("S_ext_reduce_max", 0)),
            is_warp=kind == "fused" or not s.get("S_dtype_f32", 0),
            is_dyn=s.get("S_ext_n_symbolic_axis", 0) > 0,
            kind=kind,
            free_max=int(s.get("S_ext_free_max", 0)),
        )

    def s_features_arith(self) -> dict[str, float]:
        """The extent ``S_*`` features derivable without compiling — the exact set
        the cold ``OfflinePrior`` reads (``golden_eval._offline_scorer`` builds the
        same dict). For a matmul the reduce axis is a single contraction, so
        ``S_ext_reduce_prod == S_ext_reduce_max == K``. A dynamic key adds the
        ``S_ext_n_symbolic_axis`` flag (and its ``free_prod`` already excludes the
        symbolic axis), mirroring the stamped histogram so the arith fallback
        featurizes like the rows a DB-trained prior saw."""
        out = {
            "S_ext_free_prod": float(self.free_prod),
            "S_ext_reduce_prod": float(self.reduce_max),
            "S_ext_reduce_max": float(self.reduce_max),
        }
        if self.free_max:
            out["S_ext_free_max"] = float(self.free_max)
        if self.is_dyn:
            out["S_ext_n_symbolic_axis"] = 1.0
        return out
