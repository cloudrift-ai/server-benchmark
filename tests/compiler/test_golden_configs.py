"""Schema + invariants for the golden matmul config set.

These are pure-data checks (no GPU): the records load from the per-GPU YAML files,
the derived ``ratio`` / ``golden`` properties stay consistent, and ``matmul_snippet``
/ ``repro_command`` render the canonical form. The actual latencies are measured on
a CUDA device via ``emmy tune --golden NAME --bench`` and recorded into the YAML.
"""

from __future__ import annotations

import pytest

from emmy.compiler.pipeline.search.golden import (
    _GOLDENS_DIR,
    _KERNEL_CLASSES,
    GOLDEN_CONFIGS,
    AttentionGoldenConfig,
    EmbeddingGoldenConfig,
    GoldenConfig,
    MatmulGoldenConfig,
    MlpGeGluGoldenConfig,
    NormLinearGoldenConfig,
    PointwiseGoldenConfig,
    ReduceGoldenConfig,
    RmsNormGoldenConfig,
    RopeGoldenConfig,
    SoftmaxGoldenConfig,
    _load_goldens,
    matmul_snippet,
)


def test_matmul_snippet_fp32_has_no_dtype_kwarg():
    assert matmul_snippet(2048, 2048, 2048) == "torch.matmul(torch.randn(2048,2048), torch.randn(2048,2048))"
    # Non-square: lhs is (M,K), rhs is (K,N).
    assert matmul_snippet(32, 3072, 1024) == "torch.matmul(torch.randn(32,1024), torch.randn(1024,3072))"


def test_matmul_snippet_typed():
    assert "dtype=torch.float16" in matmul_snippet(128, 128, 128, "fp16")


@pytest.mark.parametrize(
    ("emmy_us", "cublas_us", "ratio", "golden"),
    [(100.0, 99.0, 0.99, True), (100.0, 95.0, 0.95, True), (100.0, 80.0, 0.80, False), (0.0, 99.0, 0.0, False)],
)
def test_ratio_and_golden_derive(emmy_us, cublas_us, ratio, golden):
    c = GoldenConfig(name="t", emmy_us=emmy_us, cublas_us=cublas_us)
    assert c.ratio == pytest.approx(ratio)
    assert c.golden is golden


def test_repro_command_round_trips_knobs_and_snippet():
    c = MatmulGoldenConfig(name="square.2048", M=2048, N=2048, K=2048, knobs={"TILE": "n32x8/f4x26", "STAGE": "d2/tma"})
    cmd = c.repro_command()
    assert 'EMMY_KNOBS="TILE=n32x8/f4x26,STAGE=d2/tma"' in cmd
    assert c.snippet() in cmd
    assert "--ir cuda" in cmd


def test_golden_configs_set_is_well_formed():
    for c in GOLDEN_CONFIGS:
        # Every regime: matmul (M,N,K), reduce/rms_norm/softmax (M,K), pointwise (M,N), attention.
        assert isinstance(
            c,
            (
                MatmulGoldenConfig,
                ReduceGoldenConfig,
                RmsNormGoldenConfig,
                NormLinearGoldenConfig,
                MlpGeGluGoldenConfig,
                RopeGoldenConfig,
                EmbeddingGoldenConfig,
                SoftmaxGoldenConfig,
                PointwiseGoldenConfig,
                AttentionGoldenConfig,
            ),
        ), c.name
        # The memory-bound anchors (rope / embedding) FORK NOTHING — one coalesced elementwise / gather
        # config, so they record empty knobs and serve as ``eval golden`` regression anchors, not deploy
        # warmers. Every other kind carries a scheduling knob set.
        fork_nothing = isinstance(c, (RopeGoldenConfig, EmbeddingGoldenConfig))
        if isinstance(c, MatmulGoldenConfig):
            assert c.M > 0 and c.N > 0 and c.K > 0, c.name
        elif isinstance(c, NormLinearGoldenConfig):
            # The fused RMSNorm→linear computed-A megakernel: a warp contraction (M,H)@(H,N), not the
            # reduce family — its REDUCE knob is a split-K (g4k), not a coop, so no coop>1 requirement.
            assert c.M > 0 and c.H > 0 and c.N > 0, c.name
        elif isinstance(c, MlpGeGluGoldenConfig):
            # The fused RMSNorm→gate⊗up→GeGLU multi-channel computed-A megakernel (M,H)→(M,inter).
            assert c.M > 0 and c.H > 0 and c.inter > 0, c.name
        elif isinstance(c, RopeGoldenConfig):
            assert c.n_heads > 0 and c.seq > 0 and c.head_dim > 0, c.name
        elif isinstance(c, EmbeddingGoldenConfig):
            assert c.vocab > 0 and c.seq > 0 and c.hidden > 0, c.name
        elif isinstance(c, (ReduceGoldenConfig, RmsNormGoldenConfig, SoftmaxGoldenConfig)):
            from emmy.compiler.ir.schedule import ReducePlan

            assert c.M > 0 and c.K > 0, c.name
            # The reduce knob is axis-qualified on a Map (``REDUCE@a1`` for rms/softmax); bare on a pure reduce.
            red = next((v for k, v in c.knobs.items() if k == "REDUCE" or k.startswith("REDUCE@")), None)
            coop = ReducePlan.parse(red).coop
            assert coop > 1, f"{c.name} reduce-family golden must be cooperative (REDUCE coop>1, got {coop})"
        elif isinstance(c, AttentionGoldenConfig):
            assert c.n_heads > 0 and c.seq > 0 and c.head_dim > 0, c.name
        else:
            assert c.M > 0 and c.N > 0, c.name
        assert c.emmy_us > 0 and c.cublas_us > 0, c.name
        assert c.ratio >= 0.0, c.name
        assert c.golden == (c.ratio >= 0.95), c.name
        assert fork_nothing or c.knobs, f"{c.name} has no recorded knobs"
        assert c.snippet(), c.name


def test_goldens_load_from_yaml():
    """The per-GPU YAML files are the source of truth: at least one file exists,
    every config carries a known ``kernel`` discriminator, and the loaded set is
    non-empty and identical to the import-time :data:`GOLDEN_CONFIGS`."""
    import yaml

    files = sorted(_GOLDENS_DIR.glob("*.yaml"))
    assert files, f"no golden YAML files under {_GOLDENS_DIR}"
    for path in files:
        doc = yaml.safe_load(path.read_text())
        assert isinstance(doc["gpu_name"], str) and len(doc["compute_cap"]) == 2, path.name
        for c in doc["configs"]:
            assert c["kernel"] in _KERNEL_CLASSES, f"{path.name}: unknown kernel {c.get('kernel')!r}"

    loaded = _load_goldens()
    assert loaded  # non-empty
    assert len(loaded) == len(GOLDEN_CONFIGS)


def test_goldens_speak_native_codecs_and_featurize():
    """Every golden's ``knobs`` parse cleanly through the native output-fragment / reduce / stage
    codecs (no legacy GEMM-letter vocabulary survives), and every **matmul** golden featurizes to a
    non-empty tile geometry family — the scalar path that silently produced empty ``D_*`` features
    while the goldens still spelled ``BM``/``BN`` (the bug the migration fixed)."""
    from emmy.compiler.ir.schedule import ReducePlan, Stage, TilePlan
    from emmy.compiler.pipeline.search import features

    legacy = {"BN", "BM", "FM", "FN", "BK", "BR", "FK", "SPLITK", "WN", "WM", "MMA", "RING", "TMA"}
    for g in GOLDEN_CONFIGS:
        assert not (legacy & set(g.knobs)), f"{g.name} still carries legacy knobs: {sorted(legacy & set(g.knobs))}"
        tile = g.knobs.get("TILE")
        if tile:
            TilePlan.parse(tile)  # parses (warp or scalar form), or raises
        ReducePlan.parse(g.knobs.get("REDUCE"))
        if g.knobs.get("STAGE"):
            Stage.parse(g.knobs["STAGE"])
        if isinstance(g, MatmulGoldenConfig):
            feats = features.knob_features(g.knobs)
            assert feats.get("D_threads", 0) > 0 and "D_tile_m" in feats, f"{g.name} featurized to empty tile geometry"


def test_golden_knobs_are_members_of_the_move_catalog():
    """Permanence: every recorded golden knob set stays REACHABLE by the search — its ``TILE``
    codec is a member of the enumerated move grids (scalar and warp tiers alike), its ``STAGE``
    is a resolved spelling the stage catalog offers, and its ``REDUCE`` partition is an
    enumerated split / coop move. A future space edit can never silently orphan a golden again
    (the sixth sweep's finding 3: the ``.s512`` goldens' ``f2x14``/``f4x8``/``f4x10``/``f4x26``
    register tiles fell out of ``_SCALAR_REG`` and no tune patience could reach them —
    exactly the 1.29-1.49× perf-gap shapes). Pointwise goldens are exempt (their kernel forks
    nothing today, so there is no catalog to be a member of); a reduce golden's ``TILE`` is
    likewise uncatalogued (the reduce fork partitions K only), so only its ``REDUCE`` is pinned."""
    from emmy.compiler.ir.schedule import TilePlan, is_warp_codec
    from emmy.compiler.pipeline.search.space import coop_reduce_moves, scalar_tile_moves, splitk_moves, stage_moves, warp_tile_moves

    scalar_moves = set(scalar_tile_moves())
    for g in GOLDEN_CONFIGS:
        where = f"{g.name} ({g.gpu_name})"
        if isinstance(g, (ReduceGoldenConfig, RmsNormGoldenConfig, SoftmaxGoldenConfig)):
            # The reduce knob is axis-qualified on a Map (``REDUCE@a1`` for rms/softmax); bare on a pure reduce.
            reduce_spec = next((v for k, v in g.knobs.items() if k == "REDUCE" or k.startswith("REDUCE@")), "")
            assert not reduce_spec or reduce_spec in coop_reduce_moves(), f"{where}: REDUCE {reduce_spec!r} not enumerable"
            continue
        if not isinstance(g, MatmulGoldenConfig):
            continue
        tile = g.knobs.get("TILE", "")
        warp = is_warp_codec(tile)
        if tile:
            plan = TilePlan.parse(tile)
            pool = set(warp_tile_moves((plan.atom.name,))) if warp else scalar_moves
            assert plan.spell() in pool, f"{where}: TILE {tile!r} not in the enumerated {'warp' if warp else 'scalar'} grid"
        stage = g.knobs.get("STAGE", "")
        assert not stage or stage in stage_moves(warp=warp), f"{where}: STAGE {stage!r} not a catalog spelling"
        reduce_spec = g.knobs.get("REDUCE", "")
        assert not reduce_spec or reduce_spec in splitk_moves(warp=warp) + coop_reduce_moves(), (
            f"{where}: REDUCE {reduce_spec!r} not enumerable"
        )


# --- dynamic (symbolic-axis) goldens -----------------------------------------


def _dyn(M=512):
    return MatmulGoldenConfig(name="matmul.square.512.dynM", M=M, N=512, K=512, dynamic=True)


def test_dynamic_golden_specs_snippet_and_repro():
    """A dynamic golden keeps the hint-shaped snippet (M doubles as the hint) and
    additionally carries the ``--dynamic NAME@INPUT:AXIS`` spec for the tracer;
    ``repro_command`` includes the flag so the repro rebuilds the masked-tile kernel."""
    c = _dyn()
    assert c.dynamic_specs() == ["seq_len@x0:0"]
    assert c.snippet() == matmul_snippet(512, 512, 512)  # unchanged hint-shaped code
    assert "--dynamic seq_len@x0:0" in c.repro_command()


def test_static_golden_has_no_dynamic_specs():
    c = MatmulGoldenConfig(name="matmul.square.512", M=512, N=512, K=512)
    assert c.dynamic_specs() == []
    assert "--dynamic" not in c.repro_command()


def test_dynamic_golden_hint_must_equal_default_seq_hint():
    """The pipeline tiles/benches a symbolic axis at the global ``DEFAULT_SEQ_HINT``, not the
    traced M — an M=1024 dynamic golden would silently be measured at 512. Rejected until
    per-Dim hints exist; the rule holds for every kind (matmul M / reduce rows / attention seq)."""
    from emmy.compiler.dim import DEFAULT_SEQ_HINT

    with pytest.raises(ValueError, match="DEFAULT_SEQ_HINT"):
        _dyn(M=1024)
    with pytest.raises(ValueError, match="DEFAULT_SEQ_HINT"):
        SoftmaxGoldenConfig(name="softmax.k2048.dynM", M=1024, K=2048, dynamic=True)
    with pytest.raises(ValueError, match="DEFAULT_SEQ_HINT"):
        AttentionGoldenConfig(name="attention.hd128.dynM", n_heads=8, seq=1024, head_dim=128, dynamic=True)
    _dyn(M=DEFAULT_SEQ_HINT)  # the only valid hint today
    SoftmaxGoldenConfig(name="softmax.k2048.dynM", M=DEFAULT_SEQ_HINT, K=2048, dynamic=True)


def test_dynamic_supported_for_all_kinds():
    """Every kind can be static or dynamic (a symbolic seq / row axis → masked-tile kernel),
    each emitting its own ``--dynamic`` spec: 1-input kinds mark ``x:0``, matmul marks its lhs
    ``x0:0``, attention marks the seq axis of all three q/k/v inputs (``x{0,1,2}:2``)."""
    one_input = [
        ReduceGoldenConfig(name="reduce.k2048.dynM", M=512, K=2048, dynamic=True),
        RmsNormGoldenConfig(name="rms_norm.k2048.dynM", M=512, K=2048, dynamic=True),
        SoftmaxGoldenConfig(name="softmax.k2048.dynM", M=512, K=2048, dynamic=True),
        PointwiseGoldenConfig(name="pointwise.n4096.dynM", M=512, N=4096, dynamic=True),
        NormLinearGoldenConfig(name="norm_linear.n4096.dynM", M=512, H=3840, N=4096, dynamic=True),
    ]
    assert all(c.dynamic_specs() == ["seq_len@x:0"] for c in one_input)
    att = AttentionGoldenConfig(name="attention.hd128.dynM", n_heads=8, seq=512, head_dim=128, dynamic=True)
    assert att.dynamic_specs() == ["seq_len@x0:2", "seq_len@x1:2", "seq_len@x2:2"]
    assert att.snippet().startswith("F.scaled_dot_product_attention(")
    # static kinds carry no specs
    assert ReduceGoldenConfig(name="reduce.k2048", M=512, K=2048).dynamic_specs() == []


def test_sample_from_golden_carries_dynamic_specs():
    from emmy.compiler.pipeline.search.data import Sample

    dyn = Sample.from_golden(_dyn())
    assert dyn.dynamic == ("seq_len@x0:0",)
    static = Sample.from_golden(MatmulGoldenConfig(name="matmul.square.512", M=512, N=512, K=512))
    assert static.dynamic is None


def test_shape_key_carries_dynamic_flag():
    """``shape_key()`` is the single golden-side join key: static keys are the full ``M*N``
    product; a dynamic config's key excludes the symbolic M and sets ``is_dyn``."""
    static = MatmulGoldenConfig(name="s", M=512, N=256, K=64)
    assert (static.shape_key().free_prod, static.shape_key().is_dyn) == (512 * 256, False)
    dyn = _dyn()  # M=N=K=512
    assert (dyn.shape_key().free_prod, dyn.shape_key().is_dyn) == (512, True)
    assert dyn.shape_key() != MatmulGoldenConfig(name="t", M=512, N=512, K=512).shape_key()


def _dup(knobs, us):
    return MatmulGoldenConfig(name="dup.512", M=512, N=512, K=512, knobs=knobs, emmy_us=us, cublas_us=14.0)


def test_goldens_by_name_returns_every_config_under_a_name(monkeypatch):
    """A name may carry several configs (e.g. a newly found faster variant beside
    the old one); ``goldens_by_name`` returns them all, empty for an unknown name."""
    from emmy.compiler.pipeline.search import golden as gmod

    a, b = _dup({"TILE": "n16x8/f2x2"}, 12.0), _dup({"TILE": "n16x8/f4x4"}, 14.0)
    monkeypatch.setattr(gmod, "GOLDEN_CONFIGS", [a, b])
    assert gmod.goldens_by_name("dup.512") == [a, b]
    assert gmod.goldens_by_name("nope") == []


def test_resolve_golden_arg_stashes_all_matches(monkeypatch):
    """``--golden NAME`` stashes *every* recorded config under NAME on
    ``args.golden_configs`` as :class:`Sample`s (all share the shape, so ``args.code``
    is the snippet of the first); no ``--golden`` leaves an empty list."""
    from argparse import Namespace

    from emmy.compiler.pipeline.search import golden as gmod

    a, b = _dup({"TILE": "n16x8/f2x2"}, 12.0), _dup({"TILE": "n16x8/f4x4"}, 14.0)
    # Dataset.from_golden reads golden.GOLDEN_CONFIGS via a lazy import, so this patch is seen.
    monkeypatch.setattr(gmod, "GOLDEN_CONFIGS", [a, b])

    from emmy.commands import compile as cmod

    args = Namespace(golden="dup.512", code=None, input=None, ir=None)
    cmod.resolve_golden_arg(args)
    assert [s.name for s in args.golden_configs] == ["dup.512", "dup.512"]
    assert [s.knobs for s in args.golden_configs] == [{"TILE": "n16x8/f2x2"}, {"TILE": "n16x8/f4x4"}]
    assert args.code == a.snippet()
    assert all(s.source == "golden" for s in args.golden_configs)

    none = Namespace(golden=None, code=None, input=None, ir=None)
    cmod.resolve_golden_arg(none)
    assert none.golden_configs == []


def test_resolve_golden_arg_applies_dynamic_spec(monkeypatch):
    """A dynamic golden's recorded spec lands on ``args.dynamic`` (so the trace goes
    symbolic); a CLI ``--dynamic`` next to ``--golden`` is rejected — the spec is
    part of the config, the same way ``--ir`` rejects it."""
    from argparse import Namespace

    from emmy.commands import compile as cmod
    from emmy.compiler.pipeline.search import golden as gmod

    dyn = MatmulGoldenConfig(
        name="dup.512.dynM",
        M=512,
        N=512,
        K=512,
        knobs={"TILE": "n16x8/f2x2"},
        emmy_us=12.0,
        cublas_us=14.0,
        dynamic=True,
    )
    monkeypatch.setattr(gmod, "GOLDEN_CONFIGS", [dyn])

    args = Namespace(golden="dup.512.dynM", code=None, input=None, ir=None, dynamic=None)
    cmod.resolve_golden_arg(args)
    assert args.dynamic == ["seq_len@x0:0"]
    assert args.code == dyn.snippet()

    clash = Namespace(golden="dup.512.dynM", code=None, input=None, ir=None, dynamic=["seq_len@x0:0"])
    with pytest.raises(SystemExit) as exc:
        cmod.resolve_golden_arg(clash)
    assert exc.value.code == 2


def test_norm_linear_golden_snippet_shape_key_and_flops():
    """The fused RMSNorm→linear computed-A golden: its snippet is the ``F.rms_norm(x)·nw @ w`` form
    that traces to the megakernel, its shape_key carries ``kind="fused"`` with ``is_warp=True`` (a
    computed-A contraction is a warp mma), and the dynamic twin excludes the symbolic M."""
    c = NormLinearGoldenConfig(name="nl", M=32, H=3840, N=4096, knobs={"TILE": "a:mma_m16n8k16_f16_f32/w1x8/f2x2/k2"})
    snip = c.snippet()
    assert "F.rms_norm(x,(3840,),nw)" in snip and snip.strip().endswith("@ w")
    assert "dtype=torch.float16" in snip  # fp16 default → warp mma
    sk = c.shape_key()
    assert (sk.free_prod, sk.reduce_max, sk.is_warp, sk.is_dyn, sk.kind) == (32 * 4096, 3840, True, False, "fused")
    assert c.flops() == 2.0 * 32 * 4096 * 3840
    dyn = NormLinearGoldenConfig(name="nl.dynM", M=512, H=3840, N=4096, dynamic=True, knobs={})
    dsk = dyn.shape_key()
    assert (dsk.free_prod, dsk.is_dyn, dsk.kind) == (4096, True, "fused")  # symbolic M excluded
    assert dyn.dynamic_specs() == ["seq_len@x:0"]
    # A dynamic golden's traced M must equal DEFAULT_SEQ_HINT (the axis is benched at the hint).
    from emmy.compiler.dim import DEFAULT_SEQ_HINT

    with pytest.raises(ValueError, match="DEFAULT_SEQ_HINT"):
        NormLinearGoldenConfig(name="nl.bad", M=DEFAULT_SEQ_HINT + 1, H=3840, N=4096, dynamic=True)


def test_mlp_geglu_golden_snippet_shape_key_and_flops():
    """The fused RMSNorm→gate⊗up→GeGLU multi-channel golden: the lambda-bound shared-``r`` snippet,
    ``kind="fused"`` keyed on the ``(M, inter)`` GeGLU output, and the dynamic twin excludes M."""
    c = MlpGeGluGoldenConfig(name="ggu", M=32, H=3840, inter=15360, knobs={"TILE": "a:mma_m16n8k16_f16_f32/w1x8/f2x2/k4"})
    snip = c.snippet()
    assert "lambda r:" in snip and "F.rms_norm(x,(3840,),nw)" in snip and "approximate='tanh'" in snip
    sk = c.shape_key()
    assert (sk.free_prod, sk.reduce_max, sk.is_warp, sk.is_dyn, sk.kind) == (32 * 15360, 3840, True, False, "fused")
    assert c.flops() == 4.0 * 32 * 15360 * 3840  # two shared-A contractions
    dyn = MlpGeGluGoldenConfig(name="ggu.dynM", M=512, H=3840, inter=15360, dynamic=True, knobs={})
    assert (dyn.shape_key().free_prod, dyn.shape_key().is_dyn) == (15360, True)  # symbolic M excluded
    assert dyn.dynamic_specs() == ["seq_len@x:0"]


def test_fast_math_regime_derived_from_knobs():
    """``GoldenConfig.fast_math`` derives the precision regime from the RECORDED knobs (never a
    stored flag): an f16-accumulate ``TILE`` atom (bare or axis-keyed) or ``FAST_EXP: true`` marks
    the entry fast-math; the acc-unspecified atom ALIAS resolves to f32-accumulate → standard."""
    from emmy.compiler.pipeline.search.golden import fast_math_knobs

    std = {"TILE": "a:mma_m16n8k16_f16_f32/w2x4/f2x4", "STAGE": "d4/tma/ring", "FAST_EXP": False}
    alias = {"TILE": "a:mma_m16n8k16_f16/w2x2/f2x2/k2"}  # historic spelling = f32-acc
    scalar = {"TILE": "n32x8/f4x4", "REDUCE": "b8"}
    assert not fast_math_knobs(std) and not fast_math_knobs(alias) and not fast_math_knobs(scalar)
    assert fast_math_knobs({"TILE@d": "a:mma_m16n8k16_f16_f16/w2x4/f2x4/k8"})
    assert fast_math_knobs({"TILE": "n32x8/f4x4", "FAST_EXP": True})
    fm_knobs = {"TILE@d": "a:mma_m16n8k16_f16_f16/w2x2/f2x2/k2"}
    assert MatmulGoldenConfig(name="square.128.fp16", M=128, N=128, K=128, dtype="fp16", knobs=fm_knobs).fast_math
    assert not MatmulGoldenConfig(name="square.128.fp16", M=128, N=128, K=128, dtype="fp16", knobs=std).fast_math


def test_f16acc_golden_tile_stays_reachable():
    """The permanence rule extends to a fast-math golden: the membership pool is built from the
    entry's OWN parsed atom, so an f16-accumulate spelling is a member of its warp grid without
    any gate."""
    from emmy.compiler.ir.schedule import TilePlan
    from emmy.compiler.pipeline.search.space import warp_tile_moves

    plan = TilePlan.parse("a:mma_m16n8k16_f16_f16/w2x2/f4x8/k2")
    assert plan.spell() in set(warp_tile_moves((plan.atom.name,)))


def test_fast_math_golden_ranks_in_gated_enumeration(monkeypatch):
    """``evaluate_golden`` self-gates for a fast-math golden: its row is found in the enumeration
    (rank not ``None``) with NO env set — the ``F16_MMA_F32_ACC`` pin is scoped to the call and
    restored after — while a standard golden still ranks against the default enumeration."""
    import os

    from emmy.compiler.context import Context
    from emmy.compiler.pipeline.search import golden_eval

    for var in ("EMMY_FAST_MATH", "EMMY_F16_MMA_F32_ACC"):
        monkeypatch.delenv(var, raising=False)
    ctx = Context.from_target((12, 0))
    fm = {"TILE": "a:mma_m16n8k16_f16_f16/w2x2/f2x2/k2", "STAGE": "", "REDUCE": ""}
    _, rank, pool = golden_eval.evaluate_golden(128, 128, 128, "fp16", fm, ctx)
    assert rank is not None and pool > 0, "fast-math golden must rank inside the self-gated enumeration"
    assert os.environ.get("EMMY_F16_MMA_F32_ACC") is None, "the scoped pin must restore"
    std = {"TILE": "a:mma_m16n8k16_f16_f32/w2x2/f2x2/k2", "STAGE": "", "REDUCE": ""}
    _, rank_std, pool_std = golden_eval.evaluate_golden(128, 128, 128, "fp16", std, ctx)
    assert rank_std is not None
    assert pool_std < pool, "the standard enumeration must not silently grow fast-math rows"


def test_golden_flops_never_overestimates_and_gates_dyn_reduce(monkeypatch):
    """Every golden kind's ``flops()`` is positive and hint-free (a dynamic golden's symbolic
    axis is already sized at the benched hint, so dyn == static counts), and the intensity-floor
    gate reads it instead of reconstructing from the ShapeKey — the reconstruction multiplied
    the hint onto reduce-tier dyn keys whose ``free_prod`` already includes the symbolic axis,
    flagging every reduce-tier ``.dynM`` replay "impossible" at its correct recorded value."""
    import importlib

    from emmy import gpu
    from emmy.compiler.pipeline.search.data.sample import Sample

    run_mod = importlib.import_module("emmy.commands.run")
    for cfg in GOLDEN_CONFIGS:
        f = cfg.flops()
        assert f is None or f > 0, cfg.name
    m = next(c for c in GOLDEN_CONFIGS if isinstance(c, MatmulGoldenConfig))
    assert m.flops() == 2.0 * m.M * m.N * m.K
    dyn_reduce = next(c for c in GOLDEN_CONFIGS if c.dynamic and type(c).__name__ == "RmsNormGoldenConfig")
    static_twin = next(c for c in GOLDEN_CONFIGS if not c.dynamic and type(c).__name__ == "RmsNormGoldenConfig" and c.K == dyn_reduce.K)
    assert dyn_reduce.flops() == static_twin.flops(), "dyn flops must not carry a hint multiplier"
    # The floor itself: a correct-value dynM reduce replay is CLEAN; a 100x-too-fast bench flags.
    # Build the Sample against the real GPU registry FIRST (``from_golden`` pins the recording
    # card's device features off it), then patch the live-device probe for the floor call only.
    s = Sample.from_golden(dyn_reduce)
    spec = type("S", (), {"peak_tflops": staticmethod(lambda dtype: 100.0)})()
    monkeypatch.setattr(gpu, "by_name", lambda name: spec)
    monkeypatch.setattr(gpu, "live_name", lambda: "test-gpu")
    assert run_mod._intensity_floor_flag(s, dyn_reduce.emmy_us) is None, "correct dynM replay must not flag"
    assert run_mod._intensity_floor_flag(s, dyn_reduce.emmy_us / 1e5) is not None, "an impossible bench must still flag"
