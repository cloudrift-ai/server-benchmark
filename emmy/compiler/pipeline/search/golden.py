"""Golden matmul configs — known-good ``knobs`` per shape, measured vs cuBLAS.

A *golden config* records, for one matmul shape on one GPU, the autotuned knob
set and the latencies of the emmy kernel vs cuBLAS (``torch.matmul``). A
config is ``golden`` when emmy lands within 95% of cuBLAS (or better),
i.e. ``ratio = cublas_us / emmy_us >= 0.95``.

The set is a ground truth for the tuning **prior**: it pins down what the
planner's first guess *should* land on for canonical shapes, and gives a
deployable-latency baseline to regression-test against.

A ``name`` is **not** unique — one shape may carry several golden configs (e.g. a
newly found faster knob set kept beside the old). Look configs up with
:func:`goldens_by_name` (returns a list) and never assume a single match. One sanctioned
duality is the **fast-math regime**: a sweep run under ``EMMY_FAST_MATH=1`` may record a
precision-trading winner (an f16-accumulate ``TILE`` atom / ``FAST_EXP: true``) BESIDE the
same shape's standard entry, never replacing it — a default (gate-off) compile can't reach
the fast-math config, so replacing would orphan the shape's deployable golden. The regime is
derived from the recorded knobs (``GoldenConfig.fast_math``), and the rank / reproduction
views compare each entry against its own regime's enumeration.

This module is import-light (no torch / cupy at module top) so passes and tests
can read :data:`GOLDEN_CONFIGS` cheaply. The data lives as **per-GPU YAML files**
under ``goldens/`` (e.g. ``goldens/rtx5090_sm120.yaml``): a ``gpu_name`` /
``compute_cap`` header plus a ``configs`` list, each tagged with a ``kernel``
discriminator (``matmul`` / ``reduce`` / ``pointwise`` / ``rms_norm`` / ``softmax`` /
``attention`` / ``norm_linear`` — the fused RMSNorm→linear computed-A megakernel — / ``mlp_geglu`` —
its multi-channel gate⊗up→GeGLU sibling — / ``rope`` / ``embedding`` — the fork-nothing memory-bound
regression anchors). A GPU may carry more than
one file (a themed set such as ``rtx5090_sm120_gemma4.yaml`` beside the card's main
file — same header, merged by the live-GPU ``(gpu_name, compute_cap)`` scoping).
:func:`_load_goldens` concatenates every file into :data:`GOLDEN_CONFIGS`. The set is hand-maintained via
the CLI golden workflow — ``emmy tune --golden NAME --bench`` records the
winning knobs / latencies into the GPU's YAML, ``emmy eval golden`` validates.
For the **fp32** configs the reference is
pinned to **true fp32** (``allow_tf32 = False``) so the ratio compares emmy's
CUDA-core FMA kernel against a real SGEMM, not the ~5-10x faster TF32 tensor-core
path. The **fp16** squares (``*.fp16``) instead ride the warp-tier tensor-core path
and compare against cuBLAS HGEMM (torch's default fp16 matmul) — same tensor-core
hardware on both sides, so the ratio is apples-to-apples vs cuBLAS. On sm_90+ the
autotuner lands these on the swizzled s16816 ``mma_m16n8k16_f16_f32`` (ldmatrix +
mma.sync) atom — the swizzled smem slab avoids shared-load bank conflicts (a
fragment load reading smem opaquely cannot), so mma.sync is the faster fp16
GEMM. On sm_120 the pre-rebuild bar (2048²: 106.7 µs / 1.06× on a 4-warp
warp-specialized CTA) was re-met and beaten by the rebuilt swizzled TMA tier
(2048²: 95.9 µs / 0.99× on ``w1x4/f4x2/k2 d4/tma/ring``, the 2026-07-02 seventh
sweep). Ranking lives in ``search/prior/OfflinePrior`` (the ``D_*`` geometry
features over ``features.knob_features``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from emmy.compiler.pipeline.knob import STRUCT_PREFIX

# Qwen3-Embedding-0.6B linear dims (mirrors ``tests/perf/cases.py``).
QWEN3_06B_HIDDEN = 1024  # hidden_size
QWEN3_06B_INTER = 3072  # intermediate_size (gate / up / down)
QWEN3_06B_Q_DIM = 2048  # fused Q-projection output (16 heads * 128)
QWEN3_06B_KV_DIM = 1024  # fused K/V-projection output (8 heads * 128)


def matmul_snippet(M: int, N: int, K: int, dtype: str = "fp32", trans_b: bool = False) -> str:
    """The torch expression a matmul golden config tunes / benches / reproduces.

    Single source of truth: the autotune / repro paths feed this to
    ``trace_inline_code`` (so the tuned graph *is* this expression), and each
    config reproduces from the same call. fp32 is ``torch.randn``'s default, so
    no dtype kwarg is emitted for fp32 — matching the canonical example
    ``torch.matmul(torch.randn(2048,2048), torch.randn(2048,2048))``.

    ``trans_b`` spells the **serving Linear layout** — B given ``(N, K)``,
    contracted as ``x @ w.T`` via ``F.linear``. The traced contraction carries
    ``b_trans``; the warp tier stages it like any canonical matmul (the
    N-major B slab — cp.async / TMA fill it in the operand's own orientation,
    the plain no-``.trans`` ldmatrix drains it), so the same ``STAGE``
    spellings realize on both layouts. The measured µs still differ per layout
    (different slab geometry and gmem walk), so a golden meant to decide a
    serving fork must still be TUNED on this layout — a canonical-form entry
    would deploy its config with a foreign µs."""
    if dtype == "fp32":
        if trans_b:
            return f"torch.nn.functional.linear(torch.randn({M},{K}), torch.randn({N},{K}))"
        return f"torch.matmul(torch.randn({M},{K}), torch.randn({K},{N}))"
    tdt = {"fp16": "torch.float16", "bf16": "torch.bfloat16"}[dtype]
    if trans_b:
        return f"torch.nn.functional.linear(torch.randn({M},{K},dtype={tdt}), torch.randn({N},{K},dtype={tdt}))"
    return f"torch.matmul(torch.randn({M},{K},dtype={tdt}), torch.randn({K},{N},dtype={tdt}))"


def fast_math_knobs(knobs: dict) -> bool:
    """Whether a realized knob dict carries a **precision-trading** realization — an
    f16-accumulate mma atom on any ``TILE``-family value, or ``FAST_EXP: true``. This is the
    golden entry's regime discriminator, DERIVED from the recorded knobs (never a stored flag,
    which could drift from them): a fast-math entry is only reachable by an enumeration run
    under the ``FAST_MATH`` / ``F16_MMA_F32_ACC`` gate, so regime-aware consumers (the
    ``eval`` rank / reproduction views) pin the gate on before comparing against it. Replay
    itself needs no gate — the recorded ``TILE`` pin is authoritative."""
    from emmy.compiler.ir.schedule import TilePlan, Workers  # noqa: PLC0415 — keep module import-light

    from .space import FAST_EXP  # noqa: PLC0415

    def _atom_of(spec: str):
        """The warp atom a ``TILE`` value names — the site form's bare leading atom, parsed under
        a dummy warp inventory (the units never reach the atom)."""
        try:
            plan = TilePlan.parse(spec, Workers(kind="warp", units=(1, 1)))
        except ValueError:
            return None  # an unparseable historic spelling can't name an f16acc atom
        return plan.atom if plan.is_warp else None

    for k, v in knobs.items():
        s = str(v).strip()
        if k.split("@", 1)[0] == "TILE" and s:
            atom = _atom_of(s)
            if atom is not None and atom.operand_dtype("c").nbytes == 2:
                return True
        if k == FAST_EXP.name and s.casefold() in {"true", "1", "yes", "on"}:
            return True
    return False


def _knobs_env(knobs: dict) -> str:
    """Render a knobs dict as a ``EMMY_KNOBS`` value: ``TILE=n32x8/f4x26,STAGE=d2/tma``.

    Structural-feature knobs (``STRUCT_PREFIX``) are dropped — a repro command
    pins tuning decisions, not the kernel's structural identity. ``WARPSPEC`` (the pre-rebuild
    boolean spelling still on old golden rows; the live codec is ``WSPEC``) rides through like
    any other knob."""
    return ",".join(f"{k}={v}" for k, v in knobs.items() if not k.startswith(STRUCT_PREFIX))


@dataclass(frozen=True, kw_only=True)
class GoldenConfig:
    """A kernel config measured within (or near) cuBLAS on a specific GPU.

    Only the two raw latencies are stored; :attr:`ratio` and :attr:`golden`
    derive from them so the record cannot drift out of sync.
    """

    name: str  # e.g. "square.2048" / "qwen3_06b.q_proj.s128"
    gpu_name: str = "NVIDIA GeForce RTX 5090"
    compute_cap: tuple[int, int] = (12, 0)
    knobs: dict = field(default_factory=dict)  # dict(cuda_op.knobs), verbatim
    emmy_us: float = 0.0
    cublas_us: float = 0.0
    dynamic: bool = False  # symbolic seq/row axis → masked-tile kernel (``.dynM`` name convention)
    # HF model id whose serving graph this shape came from (e.g. ``google/gemma-4-12B``) —
    # provenance, never part of any join key. Optional: a hand-picked benchmark shape has no
    # model. Model-tagged entries are what ``eval golden --in-model`` audits (it re-traces the
    # model's serving twins and checks each recorded golden still realizes in them).
    model: str | None = None

    @property
    def is_routing(self) -> bool:
        """A ROUTING entry (phase 4): its knobs are ``PLACE@<seam>`` cut decisions ONLY — it
        stores the cut set for its ``(kind, shape)`` and NO schedules (every resulting piece
        re-recognizes and resolves its OWN entry; ``emmy_us`` is the recorded pipeline total, a
        claim about the child entries as of seed time). The loader rejects a mixed entry
        (:func:`_require_routing_split`); the schedule-tier index skips routing entries and the
        routing consult skips schedule entries, so the two roles never cross."""
        return bool(self.knobs) and all(str(k).split("@", 1)[0] == "PLACE" for k in self.knobs)

    @property
    def ratio(self) -> float:
        """cuBLAS latency / emmy latency — 1.0 means parity, >1 means faster."""
        return self.cublas_us / self.emmy_us if self.emmy_us else 0.0

    @property
    def golden(self) -> bool:
        """Within 95% of cuBLAS or better."""
        return self.ratio >= 0.95

    @property
    def fast_math(self) -> bool:
        """Whether this entry's recorded knobs carry a precision-trading realization
        (:func:`fast_math_knobs`) — the golden's REGIME. A shape may carry a standard entry and
        a fast-math entry side by side under one ``name`` (names are non-unique by design); the
        regime-aware consumers compare each against its own regime's enumeration."""
        return fast_math_knobs(self.knobs)

    @property
    def sm_count(self) -> int | None:
        """The recording card's SM count, from the common GPU registry (by
        :attr:`gpu_name`) — ``None`` if the GPU isn't registered. This is what makes
        a same-``compute_cap`` pair distinguishable (RTX 5090 = 170 vs RTX PRO 6000
        = 188 SMs); :meth:`~...data.sample.Sample.from_golden` threads it into the
        reconstructed context so the golden featurizes with its own card's regime,
        not the live device's."""
        from emmy import gpu  # noqa: PLC0415

        spec = gpu.by_name(self.gpu_name)
        return spec.sm_count if spec else None

    def _require_dynamic_hint(self, axis_size: int) -> None:
        """A dynamic golden's symbolic axis is tiled / benched at the GLOBAL ``Dim`` hint, so its
        traced size must equal ``DEFAULT_SEQ_HINT`` — otherwise it silently measures a different
        shape than the one recorded. Each kind calls this from ``__post_init__`` with its seq/row axis."""
        if not self.dynamic:
            return
        from emmy.compiler.dim import DEFAULT_SEQ_HINT  # noqa: PLC0415 — int constant, import stays light

        if axis_size != DEFAULT_SEQ_HINT:
            raise ValueError(
                f"{self.name}: a dynamic golden's symbolic axis (traced size {axis_size}) must equal "
                f"DEFAULT_SEQ_HINT ({DEFAULT_SEQ_HINT}); the pipeline benches a symbolic axis at the hint"
            )

    def dynamic_specs(self) -> list[str]:
        """``--dynamic NAME@INPUT:AXIS`` spec strings for the tracer — empty when static.
        Each kind overrides with the axis(es) of its snippet's inputs that go symbolic."""
        return []

    def flops(self) -> float | None:
        """A NEVER-OVERESTIMATED FLOP count for this config's benched problem (the config's
        traced sizes — a dynamic golden's symbolic axis is sized at the hint it benches at, so
        no hint multiplier is ever needed). Feeds the arithmetic-intensity floor gate
        (``run``'s ``_intensity_floor_flag``): the floor flags a bench whose implied FLOP/s
        exceeds the device peak, so an OVERestimate false-fires on a correct bench — the exact
        bug this method retires (the old ShapeKey reconstruction multiplied the hint onto
        reduce-tier dyn keys whose ``free_prod`` already includes the symbolic axis, a 512×
        overcount flagging every reduce-tier ``.dynM`` replay "impossible"). ``None`` = the
        kind is ungateable."""
        return None


@dataclass(frozen=True, kw_only=True)
class MatmulGoldenConfig(GoldenConfig):
    """A golden config for a 2-D matmul ``(M, K) @ (K, N)``.

    When ``dynamic: true``, the M axis (``x0``, the lhs rows / seq) is symbolic: the shape
    compiles as a masked-tile kernel and M doubles as the ``Dim`` hint it is sized / benched at
    (so M must equal ``DEFAULT_SEQ_HINT``). A dynamic golden is a different deployment artifact
    than its static twin (boundary guards, masked tiers), so it gets its own ``.dynM`` name and
    is never merged with the static config. Only the M axis may be symbolic today (symbolic
    K/N is future work, kept out of the schema until the lowering exists)."""

    M: int
    N: int
    K: int
    dtype: str = "fp32"
    # The serving Linear layout: B given (N, K), contracted as ``x @ w.T`` (``F.linear``).
    # The warp tier stages this layout too (the N-major B slab; see ``matmul_snippet``), so
    # the same STAGE spellings realize on both layouts — but the measured µs differ per
    # layout, so a golden meant to decide a served model's linear fork must still be tuned
    # with this on. Same ShapeKey either way: layout twins coexist under one shape and the
    # shared bucket sorts by µs.
    # CAVEAT (ordering-protected): with staging realizable on EITHER layout, a stale or
    # missing twin lets the other layout's entry deploy its config with a foreign µs; the
    # real fix is a layout signal in the stamped ``S_*`` features + this key, which does not
    # exist yet. Until then, keep BOTH layout twins recorded and current together.
    trans_b: bool = False

    def __post_init__(self):
        self._require_dynamic_hint(self.M)

    def snippet(self) -> str:
        """The torch expression this config tunes / benches / reproduces."""
        return matmul_snippet(self.M, self.N, self.K, self.dtype, self.trans_b)

    def shape_key(self):
        """This config's :class:`~emmy.compiler.pipeline.search.data.ShapeKey` —
        the single golden-side join key. Import deferred to keep this module import-light."""
        from emmy.compiler.pipeline.search.data.shape import ShapeKey  # noqa: PLC0415

        return ShapeKey.from_matmul(self.M, self.N, self.K, self.dtype, dynamic=self.dynamic)

    def dynamic_specs(self) -> list[str]:
        return ["seq_len@x0:0"] if self.dynamic else []

    def flops(self) -> float:
        return 2.0 * self.M * self.N * self.K  # exact; M is the benched hint when dynamic

    def repro_command(self, ir: str = "cuda") -> str:
        """A runnable ``emmy`` command that rebuilds this config's kernel.

        e.g. ``EMMY_KNOBS="TILE=f4x26,WORK=t32x8,STAGE=d2/tma" emmy compile -c "torch.matmul(...)" --ir cuda``
        """
        dyn = "".join(f" --dynamic {s}" for s in self.dynamic_specs())
        return f'EMMY_KNOBS="{_knobs_env(self.knobs)}" emmy compile -c "{self.snippet()}"{dyn} --ir {ir}'


@dataclass(frozen=True, kw_only=True)
class ReduceGoldenConfig(GoldenConfig):
    """A golden config for a row-reduce ``(M, K) → (M,)`` (``torch.sum(dim=-1)``).

    The good config is cooperative: ``BR > 1`` threads reduce each row in parallel
    (then a WarpShuffle / TreeHalve combine), so the prior must rank cooperative
    ``BR`` above the serial ``BR=1`` tile — the signal the matmul-only fit lacked.
    Enumerated by ``priority_mode="reduce"`` (``E_N=1``, free=M, reduce=K)."""

    M: int
    K: int
    dtype: str = "fp32"  # the snippet is fp32-only; recorded so Sample.from_golden is kind-agnostic

    def __post_init__(self):
        self._require_dynamic_hint(self.M)

    def dynamic_specs(self) -> list[str]:
        return ["seq_len@x:0"] if self.dynamic else []

    def snippet(self) -> str:
        return f"torch.sum(torch.randn({self.M},{self.K}),dim=-1)"

    def flops(self) -> float:
        return float(self.M * self.K)  # one add per element

    def shape_key(self):
        """The reduce's arithmetic join key — free dims ``(M,)``, reduce extent ``K``,
        matching what ``992_stamp_structural_features`` stamps on the reduce kernel.
        The dynamic twin excludes the symbolic M (the ``(M,) → ()`` free product collapses
        to 1) and drops the aspect — the stamped-op convention every dynamic key follows;
        the pre-fix ``free_prod=M`` dynamic key could never join a stamped symbolic reduce
        (observed on the GeGLU cut's ``__stat`` fragment)."""
        from emmy.compiler.pipeline.search.data.shape import ShapeKey  # noqa: PLC0415

        free = 1 if self.dynamic else self.M
        fm = 0 if self.dynamic else self.M
        return ShapeKey(free_prod=free, reduce_max=self.K, is_warp=False, is_dyn=self.dynamic, free_max=fm)


@dataclass(frozen=True, kw_only=True)
class PointwiseGoldenConfig(GoldenConfig):
    """A golden config for an elementwise map ``(M, N) → (M, N)`` (``torch.relu``).

    Memory-bound: the good config is a wide coalesced tile (large ``BN`` / ``FM``,
    no reduce). Enumerated by ``priority_mode="pointwise"`` (``E_K=1``, free=M·N)."""

    M: int
    N: int
    dtype: str = "fp32"  # fp16/bf16 join the model's half-precision pointwise forks (is_warp=True keys)

    def __post_init__(self):
        self._require_dynamic_hint(self.M)

    def dynamic_specs(self) -> list[str]:
        return ["seq_len@x:0"] if self.dynamic else []

    def snippet(self) -> str:
        tdt = {"fp16": ",dtype=torch.float16", "bf16": ",dtype=torch.bfloat16", "fp32": ""}[self.dtype]
        return f"torch.relu(torch.randn({self.M},{self.N}{tdt}))"

    def flops(self) -> float:
        return float(self.M * self.N)  # one op per element

    def shape_key(self):
        """The pointwise map's arithmetic join key — free product ``M·N``, no reduce
        axis (``reduce_max=0``, the ``from_s_features`` default for an unstamped extent);
        the dynamic twin excludes the symbolic M (and drops the aspect, the ``kind=""``
        dynamic-key convention)."""
        from emmy.compiler.pipeline.search.data.shape import ShapeKey  # noqa: PLC0415

        free = self.N if self.dynamic else self.M * self.N
        fm = 0 if self.dynamic else max(self.M, self.N)
        return ShapeKey(free_prod=free, reduce_max=0, is_warp=self.dtype != "fp32", is_dyn=self.dynamic, free_max=fm)


@dataclass(frozen=True, kw_only=True)
class RmsNormGoldenConfig(GoldenConfig):
    """A golden config for RMSNorm ``(M, K) → (M, K)`` (``torch.nn.RMSNorm(K)``).

    RMSNorm is a ``Map(body=sweep, source=Fold)``: a per-row mean-of-squares reduce
    over ``K`` feeds an rsqrt that rescales every element of the row (the ``k_rms_norm``
    kernel). It is reduce-tier — the good config reduces each row cooperatively
    (``REDUCE`` coop>1), so it shares the reduce regime's arithmetic key (free=M, reduce=K)
    and ``priority_mode="reduce"`` enumeration. The reference is ``torch.nn.RMSNorm`` eager
    (fp32), so the ratio compares emmy's fused norm against PyTorch's, not cuBLAS."""

    M: int
    K: int
    dtype: str = "fp32"  # snippet is fp32; recorded so Sample.from_golden is kind-agnostic
    # Per-head norms (a model's q/k-norm over head_dim) keep the head axis as a STATIC free
    # dim beside the symbolic token axis: the op is ``(M, heads, K) → (M, heads, K)`` with
    # only M symbolic, so the dynamic key's free product is ``heads*K`` — a 2-D snippet can
    # never join it (its dynamic free is just ``K``). ``heads=1`` is the plain row norm.
    # Static per-head twins may fold heads into M (same key) or spell it — both join.
    heads: int = 1

    def __post_init__(self):
        self._require_dynamic_hint(self.M)

    def dynamic_specs(self) -> list[str]:
        return ["seq_len@x:0"] if self.dynamic else []

    def snippet(self) -> str:
        dims = f"{self.M},{self.heads},{self.K}" if self.heads > 1 else f"{self.M},{self.K}"
        return f"torch.nn.RMSNorm({self.K})(torch.randn({dims}))"

    def flops(self) -> float:
        return 2.0 * self.M * self.heads * self.K  # square+accumulate per element (the scale sweep is extra — stays an underestimate)

    def shape_key(self):
        """Keys the stamped RMSNorm sweep op (``kind="rms_norm"``): the OUTPUT is ``(M[, heads], K)`` —
        the sweep re-reads the row it normalizes, so K is a free axis of the output AND the reduce
        extent (``free_prod = M*heads*K``, measured off the stamped op; the old ``free_prod = M`` never
        matched what the 992 stamp writes). The dynamic twin excludes the symbolic M (keeping the
        static ``heads`` axis — the per-head q/k-norm join)."""
        from emmy.compiler.pipeline.search.data.shape import ShapeKey  # noqa: PLC0415

        free = self.heads * self.K if self.dynamic else self.M * self.heads * self.K
        return ShapeKey(free_prod=free, reduce_max=self.K, is_warp=False, is_dyn=self.dynamic, kind="rms_norm")


@dataclass(frozen=True, kw_only=True)
class LinearNormGoldenConfig(GoldenConfig):
    """A whole-pair golden for the producer→norm pair: ``F.linear(x, w)`` + a trailing
    ``F.rms_norm`` (+ residual add for the post-attn form). The snippet builds the PAIR so a
    seeding A/B measures the pair's true e2e cost, not the norm sweep alone.

    The recorded row lives at the NORM's fork (``shape_key`` mirrors
    :class:`RmsNormGoldenConfig` — ``kind="rms_norm"``, free ``M·K``, reduce ``K``), which it
    SHARES with the plain ``rms_norm`` anchors: a recorded entry here competes with them on the
    ordinary fastest-first ordering, so seed one only when the pair form genuinely measures
    faster at that key.

    NOTE: this kind was introduced to record the row-statistic sink placement
    (the retired ``PLACE`` knob's ``stat`` element). With that placement gone there is no
    sink-vs-local decision left to record, and every shipped entry is commented out in the
    golden YAMLs; the kind is kept for its snippet/keying, which the data samplers still use."""

    M: int
    K: int  # the norm width (= the linear's output N)
    H: int  # the linear's contraction extent
    dtype: str = "fp16"
    trans_b: bool = False  # serving F.linear layout for the producer matmul
    residual: bool = False  # the post-attn form: norm(linear(x)) + r

    def __post_init__(self):
        self._require_dynamic_hint(self.M)

    def dynamic_specs(self) -> list[str]:
        return ["seq_len@x:0"] if self.dynamic else []

    def snippet(self) -> str:
        tdt = {"fp16": ",dtype=torch.float16", "bf16": ",dtype=torch.bfloat16", "fp32": ""}[self.dtype]
        w_decl = f"w = torch.randn({self.K},{self.H}{tdt})\n" if self.trans_b else f"w = torch.randn({self.H},{self.K}{tdt})\n"
        lin = "F.linear(x, w)" if self.trans_b else "x @ w"
        r_decl = f"r = torch.randn({self.M},{self.K}{tdt})\n" if self.residual else ""
        tail = " + r" if self.residual else ""
        x_decl = f"x = torch.randn({self.M},{self.H}{tdt})\n"
        return f"nw = torch.randn({self.K}{tdt})\n{w_decl}{r_decl}{x_decl}F.rms_norm({lin}, ({self.K},), nw){tail}"

    def flops(self) -> float:
        return 2.0 * self.M * self.K * self.H  # the producer contraction dominates; the norm sweep is extra

    def shape_key(self):
        """Keys the NORM kernel's fork — identical to the plain row norm's key
        (:class:`RmsNormGoldenConfig` with ``heads=1``): the sweep's output is ``(M, K)`` so
        ``free_prod = M*K``, reduce ``K``, ``is_warp=False``, ``kind="rms_norm"``."""
        from emmy.compiler.pipeline.search.data.shape import ShapeKey  # noqa: PLC0415

        free = self.K if self.dynamic else self.M * self.K
        return ShapeKey(free_prod=free, reduce_max=self.K, is_warp=False, is_dyn=self.dynamic, kind="rms_norm")


def _require_routing_split(cfg) -> None:
    """The phase-4 golden-storage split, enforced at load: an entry either ROUTES (``PLACE``
    keys only — the cut set, no schedules) or SCHEDULES (no ``PLACE`` key at all). A mixed
    entry would re-create the retired single-namespace hazard (a cut row knob-identical to its
    fused twin, tying on evidence joins) — reject it loudly, never accept-and-hope."""
    fams = {str(k).split("@", 1)[0] for k in cfg.knobs}
    if "PLACE" in fams and fams != {"PLACE"}:
        raise ValueError(
            f"golden {cfg.name!r}: an entry mixes PLACE (routing) keys with schedule keys {sorted(fams - {'PLACE'})} — "
            f"a ROUTING entry stores the cut set only; each piece's schedule lives in that piece's own entry"
        )


def _require_cone_anchor(cfg) -> None:
    """Schema guard for the fused computed-A kinds (``NormLinearGoldenConfig`` /
    ``MlpGeGluGoldenConfig``): a RECORDED entry (non-empty knobs) must anchor itself to the cone
    fork with a ``d*/sync`` STAGE — the compute-fill only a computed-A contraction offers.
    The deploy join's over-fire safety rests on "a fused golden's config can't realize on a
    plain gmem-A matmul"; an entry recorded with a gmem-direct STAGE plus plain warp tiles WOULD
    realize there, deploying a cross-family config with its foreign µs — enforce the data
    convention at load, like the flash pin-form guard. Key-only instances (empty knobs — key
    computations and fixtures) and ROUTING entries (no schedules to anchor) are exempt."""
    if not cfg.knobs or cfg.is_routing:
        return
    knobs = {str(k): str(v) for k, v in cfg.knobs.items()}
    if any(k.split("@", 1)[0] == "STAGE" and "sync" in v.split("/") for k, v in knobs.items()):
        return
    raise ValueError(
        f"golden {cfg.name!r}: a fused computed-A entry must record a d*/sync STAGE — "
        f"its config would otherwise realize on a plain gmem-A matmul fork of coincident extents "
        f"(cross-family deploy); got knobs {cfg.knobs!r}"
    )


@dataclass(frozen=True, kw_only=True)
class NormLinearGoldenConfig(GoldenConfig):
    """A golden config for the fused RMSNorm→linear **computed-A** megakernel:
    ``rms_norm(x, (H,))·nw @ W`` in ONE mma kernel (``(M, H) → (M, N)``).

    This is the single-channel computed-A ``Contraction`` (``010_recognize``'s
    ``bind_prologue_contraction`` merge): the per-row RMSNorm statistic rides the A cone as a
    ``sync`` compute-fill prologue and the warp mma rows contract the scaled A against ``W`` — a
    different kernel family than the bare ``mlp_gate_up`` matmul (which round-trips ``xn`` through
    gmem) or a bare ``rms_norm`` sweep. It stamps LIKE an RMSNorm sweep (rsqrt, a sweep loop nest)
    but with a SECOND reduce axis — the contraction — beside the statistic reduce, so its
    :class:`ShapeKey` carries ``kind="fused"`` (``S_ext_n_reduce_axis >= 2``), keeping it off both
    the ``rms_norm`` and the ``mlp_gate_up`` matmul goldens. The reference is torch's UNFUSED
    decomposition (``F.rms_norm`` eager + ``@``), so the ratio compares emmy's one fused mma against
    PyTorch's norm-then-matmul. Its ONLY realizable configs are the sync compute-fill tiles
    (``d1/d2/sync``, no ``d2/tma/ring``) — record the tuned twin's ``record_knobs`` verbatim.

    The multi-channel gate⊗up (SwiGLU/GeGLU) megakernel shares ``kind="fused"`` but is a distinct
    snippet — see :class:`MlpGeGluGoldenConfig` (both matmuls must share ONE RMSNorm output, which a
    lambda binding — ``(lambda r: f(r@Wg)*(r@Wu))(rms_norm(x))`` — expresses; a preamble ``xn = ...``
    would precompute it to a constant and inlining ``rms(x)`` twice traces two un-shared norms)."""

    M: int
    H: int
    N: int
    dtype: str = "fp16"  # a computed-A contraction is a warp mma (fp16/bf16); fp32 has no fused form
    # The serving Linear layout: W given ``(N, H)``, contracted via ``F.linear`` — the fused edge a
    # SERVED model actually deploys (its ``b_trans`` channels stage N-major under the sync
    # transport's async fills). Same layout-blind ShapeKey as the canonical twin
    # (:class:`MatmulGoldenConfig.trans_b`'s caveat applies): keep BOTH layout twins recorded.
    trans_b: bool = False

    def __post_init__(self):
        self._require_dynamic_hint(self.M)
        _require_cone_anchor(self)

    def dynamic_specs(self) -> list[str]:
        return ["seq_len@x:0"] if self.dynamic else []

    def snippet(self) -> str:
        tdt = {"fp16": ",dtype=torch.float16", "bf16": ",dtype=torch.bfloat16", "fp32": ""}[self.dtype]
        w_decl = f"w = torch.randn({self.N},{self.H}{tdt})\n" if self.trans_b else f"w = torch.randn({self.H},{self.N}{tdt})\n"
        contract = f"F.linear(F.rms_norm(x,({self.H},),nw), w)" if self.trans_b else f"F.rms_norm(x,({self.H},),nw) @ w"
        return f"nw = torch.randn({self.H}{tdt})\n{w_decl}x = torch.randn({self.M},{self.H}{tdt})\n{contract}"

    def flops(self) -> float:
        return 2.0 * self.M * self.N * self.H  # the contraction; the norm/rsqrt prologue is extra (stays an underestimate)

    def shape_key(self):
        """Keys the stamped computed-A fused op (``kind="fused"``): the output is ``(M, N)`` so
        ``free_prod = M*N``, the reduce is the contraction extent ``H`` (the statistic reduce shares
        ``H``); the dynamic twin excludes the symbolic M. ``is_warp`` is always True — a computed-A
        contraction is a warp mma even though its f32 statistic constants would flip the stamp's
        dtype signal (:meth:`ShapeKey.from_s_features` forces it for the kind)."""
        from emmy.compiler.pipeline.search.data.shape import ShapeKey  # noqa: PLC0415

        free = self.N if self.dynamic else self.M * self.N
        return ShapeKey(
            free_prod=free,
            reduce_max=self.H,
            is_warp=True,
            is_dyn=self.dynamic,
            kind="fused",
            # The aspect, so 32x4096 (local norm->q) and 256x512 (global norm->kv) — equal
            # in free_prod AND reduce — stay distinct keys. Matches the op's stamped
            # ``S_ext_free_max``; dynamic keys normalize it away in ``__post_init__``.
            free_max=0 if self.dynamic else max(self.M, self.N),
        )


@dataclass(frozen=True, kw_only=True)
class MlpGeGluGoldenConfig(GoldenConfig):
    """The fused RMSNorm→gate⊗up→GeGLU **multi-channel computed-A** megakernel:
    ``gelu(rms_norm(x)·nw @ Wg) * (rms_norm(x)·nw @ Wu)`` in ONE mma kernel (``(M, H) → (M, inter)``).

    The MLP hot kernel gemma-4 actually deploys (``k_linear_mean_reduce``): the gate and up matmuls
    are ⊗-fold CHANNELS sharing ONE compute-filled A slab (one ldmatrix'd A fragment feeding per-fold
    B slabs / C fragments), the per-row RMSNorm statistic rides the A cone as a ``sync`` compute-fill
    prologue, and the GeGLU ⊗-combine rides the store's fragment epilogue. It shares
    :class:`NormLinearGoldenConfig`'s ``kind="fused"`` (rsqrt + a second reduce axis) but keys on the
    ``(M, inter)`` OUTPUT (``free_prod = M*inter``, reduce ``H``) — the geglu collapses the two channels
    to one inter-wide output. The snippet binds the shared ``rms_norm`` via a lambda (a torch expression
    cannot otherwise share it — a preamble precomputes, inlining duplicates). gemma-4 uses GeGLU
    (tanh-approx gelu), not SwiGLU. The reference is torch's UNFUSED decomposition, so the ratio compares
    emmy's one fused mma against PyTorch's norm→2 matmuls→gelu→multiply. Only the sync compute-fill tiles
    realize (``d1/d2/sync``); record the tuned twin's ``record_knobs`` verbatim."""

    M: int
    H: int
    inter: int
    dtype: str = "fp16"  # a computed-A contraction is a warp mma (fp16/bf16)
    # The serving Linear layout (gate/up given ``(inter, H)``, contracted via ``F.linear``) — see
    # :class:`NormLinearGoldenConfig.trans_b`; keep BOTH layout twins recorded.
    trans_b: bool = False

    def __post_init__(self):
        self._require_dynamic_hint(self.M)
        _require_cone_anchor(self)

    def dynamic_specs(self) -> list[str]:
        return ["seq_len@x:0"] if self.dynamic else []

    def snippet(self) -> str:
        tdt = {"fp16": ",dtype=torch.float16", "bf16": ",dtype=torch.bfloat16", "fp32": ""}[self.dtype]
        shape = f"{self.inter},{self.H}" if self.trans_b else f"{self.H},{self.inter}"
        combine = (
            "(lambda r: F.gelu(F.linear(r, wg), approximate='tanh') * F.linear(r, wu))"
            if self.trans_b
            else "(lambda r: F.gelu(r @ wg, approximate='tanh') * (r @ wu))"
        )
        return (
            f"wg = torch.randn({shape}{tdt})\n"
            f"wu = torch.randn({shape}{tdt})\n"
            f"nw = torch.randn({self.H}{tdt})\n"
            f"x = torch.randn({self.M},{self.H}{tdt})\n"
            f"{combine}(F.rms_norm(x,({self.H},),nw))"
        )

    def flops(self) -> float:
        return 4.0 * self.M * self.inter * self.H  # two K-contractions (gate + up) of the shared A; norm/gelu extra

    def shape_key(self):
        """Keys the stamped multi-channel computed-A fused op (``kind="fused"``): the GeGLU output is
        ``(M, inter)`` so ``free_prod = M*inter``, the reduce is the shared contraction extent ``H``;
        the dynamic twin excludes the symbolic M. ``is_warp`` is always True (a computed-A contraction
        is a warp mma — :meth:`ShapeKey.from_s_features` forces it for the kind)."""
        from emmy.compiler.pipeline.search.data.shape import ShapeKey  # noqa: PLC0415

        free = self.inter if self.dynamic else self.M * self.inter
        return ShapeKey(
            free_prod=free,
            reduce_max=self.H,
            is_warp=True,
            is_dyn=self.dynamic,
            kind="fused",
            free_max=0 if self.dynamic else max(self.M, self.inter),  # aspect — see NormLinearGoldenConfig
        )


@dataclass(frozen=True, kw_only=True)
class SoftmaxGoldenConfig(GoldenConfig):
    """Row-softmax ``(M, K) → (M, K)`` over the last axis (``torch.softmax(dim=-1)``).

    A twisted Fold (max / exp / sum) + a normalize sweep — the ``k_softmax`` kernel.
    Reduce-tier: free rows ``(M,)``, reduce extent ``K``, cooperative ``REDUCE`` over ``K``. The
    reference is torch softmax eager (fp32), so the ratio compares emmy's fused softmax vs PyTorch."""

    M: int
    K: int
    dtype: str = "fp32"

    def __post_init__(self):
        self._require_dynamic_hint(self.M)

    def dynamic_specs(self) -> list[str]:
        return ["seq_len@x:0"] if self.dynamic else []

    def snippet(self) -> str:
        return f"torch.softmax(torch.randn({self.M},{self.K}),dim=-1)"

    def flops(self) -> float:
        return 2.0 * self.M * self.K  # max+sum folds per element (exp/div extra — stays an underestimate)

    def shape_key(self):
        """Keys the stamped softmax sweep op (``kind="softmax"``): like RMSNorm, the normalize
        sweep makes K a free axis of the ``(M, K)`` output as well as the (max+sum) reduce extent —
        ``free_prod = M*K`` per the stamped op, symbolic M excluded on the dynamic twin."""
        from emmy.compiler.pipeline.search.data.shape import ShapeKey  # noqa: PLC0415

        free = self.K if self.dynamic else self.M * self.K
        return ShapeKey(free_prod=free, reduce_max=self.K, is_warp=False, is_dyn=self.dynamic, kind="softmax")


@dataclass(frozen=True, kw_only=True)
class AttentionGoldenConfig(GoldenConfig):
    """Scaled-dot-product (flash) attention over ``(1, n_heads, seq, head_dim)`` inputs, causal
    (``F.scaled_dot_product_attention(q, k, v, is_causal=True)``).

    The ``k_scaled_dot_product_attention`` kernel is a TWISTED streaming Fold over the KV
    axis (online-softmax flash) fused with the QKᵀ / ·V matmuls. When ``dynamic: true`` the seq
    axis (``x{0,1,2}:2`` of q / k / v) is symbolic — the masked-tile flash, the deployable
    artifact. The reference is torch SDPA (its own fused path), so the ratio is vs PyTorch."""

    n_heads: int
    seq: int
    head_dim: int
    dtype: str = "fp16"
    causal: bool = True

    def __post_init__(self):
        self._require_dynamic_hint(self.seq)
        # Flash TILE pinning is fragile — a golden must record the ONE form that reproduces its
        # measured config (a wrong form silently re-benches a scalar fallback 100–1000× slow: the
        # "flatten" pathology). The forms are shape-specific and there is no clean static rule (hd64
        # reproduces from a bare TILE, hd128 needs per-axis TILE@dd/TILE@pj — the two contractions
        # take different tiles there), so the recorder verifies each by re-bench. The ONE invariant we
        # can guard statically: a DYNAMIC (masked-flash) golden's axis-keyed pin never resolves (the
        # kernel keys its tile differently at pin time), so it MUST record a single bare TILE. A
        # dynamic FAST-MATH golden records the sibling-atom **PV plan** as that bare TILE (the exact
        # string its static twin stamps on TILE@<pv_k>) — the pinned branch of
        # ``_twisted_warp_options`` recovers the geometry from it and keeps scores f32-accumulate.
        keyed = [k for k in self.knobs if k.startswith("TILE@")]
        if self.dynamic and keyed:
            raise ValueError(
                f"{self.name}: dynamic attention golden has axis-keyed {keyed} — the masked-flash pin "
                f"doesn't resolve TILE@<axis>; record a single bare TILE that applies to both contractions."
            )

    def dynamic_specs(self) -> list[str]:
        return ["seq_len@x0:2", "seq_len@x1:2", "seq_len@x2:2"] if self.dynamic else []

    def flops(self) -> float:
        # Half the dense 4·h·s²·d flash count — never an overestimate under a causal mask.
        return 2.0 * self.n_heads * self.seq * self.seq * self.head_dim

    def snippet(self) -> str:
        tdt = {"fp32": "", "fp16": ",dtype=torch.float16", "bf16": ",dtype=torch.bfloat16"}[self.dtype]
        qkv = f"torch.randn(1,{self.n_heads},{self.seq},{self.head_dim}{tdt})"
        return f"F.scaled_dot_product_attention({qkv}, {qkv}, {qkv}, is_causal={self.causal})"

    def shape_key(self):
        """Keys the TWISTED flash op the stamp pass emits (``kind="flash"``): free = the output
        extents (heads x query rows x head_dim), reduce = the streamed KV axis. The dynamic twin
        mirrors the 992 stamp's symbolic-axis exclusion — q/k/v seq are ALL symbolic, so the free
        product keeps heads x head_dim and the reduce extent drops out entirely (``reduce_max=0``,
        measured off the stamped masked-flash op). The QKᵀ contraction the same trace emits keys as
        a plain contraction (``kind=""``) and never joins an attention golden."""
        from emmy.compiler.pipeline.search.data.shape import ShapeKey  # noqa: PLC0415

        free = self.n_heads * self.head_dim if self.dynamic else self.n_heads * self.seq * self.head_dim
        return ShapeKey(
            free_prod=free,
            reduce_max=0 if self.dynamic else self.seq,
            is_warp=self.dtype != "fp32",
            is_dyn=self.dynamic,
            kind="flash",
        )


@dataclass(frozen=True, kw_only=True)
class RopeGoldenConfig(GoldenConfig):
    """Rotary position embedding apply — ``q*cos + rotate_half(q)*sin`` over ``(1, n_heads, seq,
    head_dim)`` (the ``k_cat_slice_unsqueeze_pointwise`` kernel; ``rotate_half`` is ``cat(-q[…d/2:],
    q[…:d/2])``). A pure **memory-bound pointwise map** (no reduce, no contraction) applied to q and k
    every layer. It FORKS NOTHING — one coalesced elementwise config, so a golden cannot warm a cold
    deploy (there is no misdeploy to fix); it is a REGRESSION ANCHOR for ``emmy eval golden`` (the
    recorded latency vs torch eager catches a codegen slowdown). The reference is torch eager."""

    n_heads: int
    seq: int
    head_dim: int
    dtype: str = "fp16"

    def __post_init__(self):
        self._require_dynamic_hint(self.seq)

    def dynamic_specs(self) -> list[str]:
        return ["seq_len@q:2", "seq_len@cos:2", "seq_len@sin:2"] if self.dynamic else []

    def snippet(self) -> str:
        tdt = {"fp16": ",dtype=torch.float16", "bf16": ",dtype=torch.bfloat16", "fp32": ""}[self.dtype]
        h = self.head_dim // 2
        return (
            f"cos = torch.randn(1,1,{self.seq},{self.head_dim}{tdt})\n"
            f"sin = torch.randn(1,1,{self.seq},{self.head_dim}{tdt})\n"
            f"q = torch.randn(1,{self.n_heads},{self.seq},{self.head_dim}{tdt})\n"
            f"q*cos + torch.cat((-q[...,{h}:], q[...,:{h}]),dim=-1)*sin"
        )

    def flops(self) -> float:
        return float(self.n_heads * self.seq * self.head_dim)  # one fused-multiply-add per element

    def shape_key(self):
        """Keys the stamped pointwise map (``kind=""``, ``reduce_max=0``): free = the output extents
        (heads x seq x head_dim). The dynamic twin excludes the symbolic seq (``free_prod`` keeps
        heads x head_dim, ``free_max`` normalizes to 0 like every non-plain / dynamic key)."""
        from emmy.compiler.pipeline.search.data.shape import ShapeKey  # noqa: PLC0415

        free = self.n_heads * self.head_dim if self.dynamic else self.n_heads * self.seq * self.head_dim
        fm = 0 if self.dynamic else max(self.n_heads, self.seq, self.head_dim)
        return ShapeKey(free_prod=free, reduce_max=0, is_warp=self.dtype != "fp32", is_dyn=self.dynamic, free_max=fm)


@dataclass(frozen=True, kw_only=True)
class EmbeddingGoldenConfig(GoldenConfig):
    """Token embedding gather — ``embed_tokens[ids]`` over a ``(vocab, hidden)`` table (the
    ``k_embedding`` kernel; ``F.embedding``). A pure **memory-bound gather** (one hidden-wide row copy
    per token, no compute). Like :class:`RopeGoldenConfig` it FORKS NOTHING — a REGRESSION ANCHOR for
    ``emmy eval golden``, not a deploy warmer. The reference is torch eager (``F.embedding``)."""

    vocab: int
    seq: int
    hidden: int
    dtype: str = "fp16"

    def __post_init__(self):
        self._require_dynamic_hint(self.seq)

    def dynamic_specs(self) -> list[str]:
        return ["seq_len@ids:1"] if self.dynamic else []

    def snippet(self) -> str:
        tdt = {"fp16": ",dtype=torch.float16", "bf16": ",dtype=torch.bfloat16", "fp32": ""}[self.dtype]
        return f"ids = torch.randint(0,{self.vocab},(1,{self.seq}))\nw = torch.randn({self.vocab},{self.hidden}{tdt})\nF.embedding(ids, w)"

    def flops(self) -> float:
        return float(self.seq * self.hidden)  # one copy per gathered element (no arithmetic)

    def shape_key(self):
        """Keys the stamped gather (``kind=""``, ``reduce_max=0``): the output is ``(seq, hidden)`` so
        ``free_prod = seq*hidden`` (the vocab is the indexed axis, not a free extent); the dynamic twin
        excludes the symbolic seq."""
        from emmy.compiler.pipeline.search.data.shape import ShapeKey  # noqa: PLC0415

        free = self.hidden if self.dynamic else self.seq * self.hidden
        fm = 0 if self.dynamic else max(self.seq, self.hidden)
        return ShapeKey(free_prod=free, reduce_max=0, is_warp=self.dtype != "fp32", is_dyn=self.dynamic, free_max=fm)


_GOLDENS_DIR = Path(__file__).parent / "goldens"
_KERNEL_CLASSES = {
    "matmul": MatmulGoldenConfig,
    "reduce": ReduceGoldenConfig,
    "pointwise": PointwiseGoldenConfig,
    "rms_norm": RmsNormGoldenConfig,
    "linear_norm": LinearNormGoldenConfig,
    "norm_linear": NormLinearGoldenConfig,
    "mlp_geglu": MlpGeGluGoldenConfig,
    "rope": RopeGoldenConfig,
    "embedding": EmbeddingGoldenConfig,
    "softmax": SoftmaxGoldenConfig,
    "attention": AttentionGoldenConfig,
}


def _load_goldens() -> list[GoldenConfig]:
    """Load every per-GPU golden YAML under :data:`_GOLDENS_DIR` into one flat list.

    One file per GPU: a ``gpu_name`` / ``compute_cap`` header (stamped onto every
    config so it isn't repeated per entry), an optional ``model:`` provenance header
    (the HF model id the shapes came from — see :attr:`GoldenConfig.model`; overridable
    per entry), plus a ``configs`` list, each tagged with
    a ``kernel`` discriminator (``matmul`` / ``attention`` / ``softmax`` / ``reduce`` / ``rms_norm`` / ``norm_linear`` /
    ``mlp_geglu`` / ``rope`` / ``embedding`` / ``pointwise``) selecting the dataclass. All files are concatenated — a
    ``name`` may recur across GPUs.

    NOTE: ``compute_cap`` does **not** uniquely identify a GPU — two different cards
    can share a capability (e.g. ``rtx5090_sm120.yaml`` and
    ``rtxpro6000_sm120.yaml`` are both ``(12, 0)``). The full ``(gpu_name,
    compute_cap)`` pair is the GPU identity. The ``eval`` / ``tune --dataset golden``
    paths currently iterate this flat list without filtering to the live GPU, so a
    multi-GPU goldens dir intermixes cards in those views and ``ShapeKey`` joins
    (which key on shape, not GPU) merge same-shape entries across cards — filtering
    the consumers to the live ``(gpu_name, compute_cap)`` is a known TODO."""
    out: list[GoldenConfig] = []
    for path in sorted(_GOLDENS_DIR.glob("*.yaml")):
        doc = yaml.safe_load(path.read_text())
        gpu_name, cap = doc["gpu_name"], tuple(doc["compute_cap"])
        file_model = doc.get("model")  # optional provenance header; a per-entry ``model:`` overrides it
        for c in doc["configs"]:
            cfg = _KERNEL_CLASSES[c.pop("kernel")](gpu_name=gpu_name, compute_cap=cap, model=c.pop("model", file_model), **c)
            _require_routing_split(cfg)  # routing entries store cuts ONLY; schedule entries never spell PLACE
            out.append(cfg)
    return out


GOLDEN_CONFIGS: list[GoldenConfig] = _load_goldens()


def goldens_by_name(name: str) -> list[MatmulGoldenConfig]:
    """Every :class:`MatmulGoldenConfig` recorded under ``name`` — a **list**, not
    a single config: one shape may carry several golden knob sets (e.g. a newly
    found faster variant alongside the old one), so callers must not assume a name
    is unique. Empty when ``name`` is unknown. All entries share the shape (so any
    one's :meth:`~MatmulGoldenConfig.snippet` is interchangeable); they differ only
    in ``knobs`` / measured latency — **including across GPUs**: with multiple
    per-GPU files the same ``name`` recurs once per card, so the returned list can
    mix GPUs (use ``compute_cap`` **and** ``gpu_name`` to pick a specific card; cap
    alone is ambiguous for same-capability cards like RTX 5090 / RTX PRO 6000)."""
    return [g for g in GOLDEN_CONFIGS if isinstance(g, MatmulGoldenConfig) and g.name == name]


def goldens_for_live_gpu() -> list[GoldenConfig]:
    """:data:`GOLDEN_CONFIGS` narrowed to the **live** card's ``(gpu_name,
    compute_cap)`` when a CUDA device is visible, else the full flat list.

    The shape-keyed diagnostics / ``eval`` golden views (coverage, deploy-perf,
    prior rank) join on :class:`ShapeKey`, which is GPU-blind — so with multiple
    per-GPU files a name recurs once per card and same-shape entries collide.
    Two cards can even share ``compute_cap`` (RTX 5090 / RTX PRO 6000 are both
    ``(12, 0)``), so the live ``gpu_name`` is what disambiguates them. Filtering to
    the live card keeps those views about the card actually being tuned. Off-GPU
    (CI, pure-logic tests) there is no live card, so the unfiltered list is
    returned — callers that need determinism there should inject a single-GPU set.
    """
    live = live_recorded_goldens()
    return list(GOLDEN_CONFIGS) if live is None else (live or list(GOLDEN_CONFIGS))


def live_recorded_goldens() -> list[GoldenConfig] | None:
    """The live card's OWN recorded goldens — ``None`` when no CUDA device is visible
    (or the probe fails), an **empty list** for a live card with no golden file. Unlike
    :func:`goldens_for_live_gpu` there is no union fallback, so callers can tell an
    uncovered card from an off-GPU run: ``tune`` errors on the former rather than tune
    another card's config under this card's name (the cross-card shadowing bug)."""
    key = _live_gpu_key()
    if key is None:
        return None
    return [g for g in GOLDEN_CONFIGS if g.gpu_name == key[0] and tuple(g.compute_cap) == key[1]]


def _live_gpu_key() -> tuple[str, tuple[int, int]] | None:
    """The live card's ``(gpu_name, compute_cap)``, or ``None`` when no CUDA device
    is visible (or the probe fails) — the join key the per-GPU golden scoping filters on."""
    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_name(0), tuple(torch.cuda.get_device_capability(0))
    except Exception:  # noqa: BLE001 — any probe failure ⇒ no live filter
        return None
