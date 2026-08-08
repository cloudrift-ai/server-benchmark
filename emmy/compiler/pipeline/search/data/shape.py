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
    # silently shadow ``q_proj_global.m32``'s golden), but not the max.
    #
    # Participates for STATIC plain-nest (``kind == ""``) and ``kind == "fused"`` keys.
    # The other sweep kinds normalize it to 0 in ``__post_init__``: their ops are
    # MULTI-axis, so the stamp's largest single free extent is not the ``max(M, N)`` a
    # 2-D golden config can compute (a gemma-4 per-head qk-norm stamps
    # ``free_prod=1048576`` with ``free_max=256``), and the two builders would disagree —
    # the "builders don't all see the same extent list" hazard that scoped this field to
    # plain nests when it landed. A ``fused`` op is a plain ``(M, H) @ (H, N)`` contraction
    # with exactly two free axes, so both sides agree; measured across every gemma-4 fused
    # twin fork, the stamped ``S_ext_free_max`` equals the golden's ``max(M, N)`` exactly.
    # Including it there is required for correctness, not just precision: the M=256 global
    # norm→kv cone (256×512) and the M=32 local norm→q cone (32×4096) share
    # ``free_prod=131072`` AND ``reduce_max=3840``, so without the aspect the former
    # silently deployed the latter's golden at a fabricated 24.1 µs.
    # Dynamic keys always normalize to 0 — the symbolic axis has no static extent.
    free_max: int = 0
    # The STORAGE dtype class of the matmul's B side — ``""`` for the f32/f16/bf16
    # world (every pre-existing golden/DB key keeps its identity: the default
    # participates in equality/hash exactly as if the field didn't exist), ``"f8"``
    # when the B operand is stored as an fp8 dtype (``f8e4m3`` / ``f8e5m2``). An
    # fp8-stored weight is a different kernel family from its 16-bit twin at the
    # same ``(M, N, K)`` — half the B bytes, a decode convert in the staging path —
    # so the keys must not join, exactly as ``is_warp`` keeps the fp32/fp16 twins
    # apart (see the FP8 plan). f16 and bf16 deliberately still SHARE a key (same
    # bytes, same atoms); the class splits only where the kernel family splits.
    # ``dtype_class == "f8"`` forces ``is_warp=True`` in both constructors: the
    # fp8-B contraction is a warp-family kernel, and the op-side dtype-multiset
    # signal is unreliable there (the f32 scale constant flips ``S_dtype_f32`` on).
    # ``"trellis"`` is the same statement for a trellis-coded (EXL3) B: the operand is
    # an ``i16`` codes buffer decoded in-kernel, an eighth of the bytes and a wholly
    # different realization at both tiers (the warp tier's compute fill, the reduce
    # tier's decode band), so it must not join its f16 twin's rows in either direction.
    dtype_class: str = ""

    # Key classes that carry the aspect discriminator (see ``free_max``).
    _FREE_MAX_KINDS = ("", "fused")

    # Golden/CLI dtype spellings that name fp8 B-side storage (``fp8`` is the
    # generic golden spelling, defaulting to e4m3 — see ``golden_eval._DTYPES``).
    _F8_DTYPES = ("fp8", "f8e4m3", "f8e5m2")

    def __post_init__(self) -> None:
        if (self.is_dyn or self.kind not in self._FREE_MAX_KINDS) and self.free_max:
            object.__setattr__(self, "free_max", 0)

    @classmethod
    def from_matmul(cls, M: int, N: int, K: int, dtype: str, *, dynamic: bool = False) -> ShapeKey:
        """The shape of an ``(M, K) @ (K, N)`` matmul. fp32 lowers on the scalar
        thread tier; fp16 / bf16 on the warp (MMA) tier; an fp8-B spelling
        (``fp8`` / ``f8e4m3`` / ``f8e5m2``) is warp too and additionally carries
        ``dtype_class="f8"`` (see the field doc). ``dynamic`` marks the M
        axis symbolic (the only symbolic-axis golden form today): the key then
        MIRRORS what ``992_stamp_structural_features`` puts on the op — symbolic
        axes are **excluded** from the extent products (``free_prod = N``, not the
        hint-sized ``M*N``) and flagged via ``S_ext_n_symbolic_axis`` — because
        the stamped histogram is the only identity the op side has (it doesn't
        know the hint), so a hint-sized golden key would never join it."""
        f8 = dtype in cls._F8_DTYPES
        return cls(
            free_prod=N if dynamic else M * N,
            reduce_max=K,
            is_warp=f8 or dtype != "fp32",
            is_dyn=dynamic,
            free_max=0 if dynamic else max(M, N),
            dtype_class="f8" if f8 else "",
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
        ``""`` — conservative: it can only stop a cross-kind join, never invent one.

        ``dtype_class`` reads the stamped dtype multiset: an ``S_dtype_f8e4m3`` /
        ``S_dtype_f8e5m2`` load (the ``992`` stamp generates ``S_dtype_*`` from buffer
        dtype names, so fp8 stamps with no stamp-side change) marks the fp8-B storage
        class, which also forces ``is_warp=True`` — the fp8 kernel's f32 scale constant
        would otherwise flip the dtype-multiset signal to scalar, the same hazard the
        ``"fused"`` kind forces around. ``S_dtype_i16`` marks the trellis class the same
        way: ``i16`` is the packed-code carrier and nothing else reads one, so the stamp
        already separates a decoded B from its f16 twin at the same ``(M, N, K)``."""
        n_axes = s.get("S_ext_n_free_axis", 0) + s.get("S_ext_n_reduce_axis", 0) + s.get("S_ext_n_symbolic_axis", 0)
        kind = ""
        if 0 < s.get("S_loop_depth", 0) < n_axes:
            if s.get("S_pw_rsqrt", 0):
                kind = "fused" if s.get("S_ext_n_reduce_axis", 0) >= 2 else "rms_norm"
            elif s.get("S_pw_exp", 0):
                kind = "flash" if s.get("S_n_free_loop", 0) >= 3 else "softmax"
        f8 = any(s.get(f"S_dtype_{t}", 0) for t in ("f8e4m3", "f8e5m2"))
        trellis = bool(s.get("S_dtype_i16", 0))  # the packed-code carrier — see ``dtype_class``
        return cls(
            free_prod=int(s.get("S_ext_free_prod", 0)),
            reduce_max=int(s.get("S_ext_reduce_max", 0)),
            is_warp=kind == "fused" or f8 or not s.get("S_dtype_f32", 0),
            is_dyn=s.get("S_ext_n_symbolic_axis", 0) > 0,
            kind=kind,
            free_max=int(s.get("S_ext_free_max", 0)),
            dtype_class="f8" if f8 else ("trellis" if trellis else ""),
        )

    def joins(self, golden: ShapeKey) -> bool:
        """Whether this OP-side key (``from_s_features``) names the same shape as a
        GOLDEN-side key (``shape_key()``) — exact equality, except ``is_warp`` is
        ignored for the sweep kinds. Rationale: a sweep op's ``is_warp`` derives from
        the operand-dtype multiset, which is snippet-unstable — an all-fp16 golden
        snippet stamps fp16-pure (no ``S_dtype_f32`` → ``is_warp=True``) while the
        same norm in a served graph carries f32 statistic constants (→ ``False``), and
        the norm-kind goldens all record ``is_warp=False``. Extents + ``kind`` are the
        discriminating identity there; ``is_warp`` stays load-bearing exactly where it
        disambiguates real twins — the ``kind == ""`` fp32/fp16 contraction pair — and
        ``"fused"`` forces ``True`` on both sides already."""
        if golden.kind not in ("", "fused") and self.kind == golden.kind:
            from dataclasses import replace  # noqa: PLC0415

            return replace(self, is_warp=golden.is_warp) == golden
        return self == golden

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
