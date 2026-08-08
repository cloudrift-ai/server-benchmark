"""``ShapeKey`` round-trips for every golden kind — the golden-side ``shape_key()`` and the
op-side ``from_s_features`` (over the REAL stamped histogram of the kind's traced snippet) must
produce the same key, or the golden↔op joins (the deploy evidence tier, the diagnostics) silently
never fire. The attention trace emits TWO stamped ops (the QKᵀ contraction + the twisted flash
op); exactly the flash op must join, keyed ``kind="flash"``. Matmul keys must stay byte-identical
to the pre-``kind`` constructor (``kind=""``) so every existing matmul join is unchanged."""

from __future__ import annotations

import pytest

from emmy.compiler.pipeline.search.data.shape import ShapeKey
from emmy.compiler.pipeline.search.golden import (
    AttentionGoldenConfig,
    EmbeddingGoldenConfig,
    MatmulGoldenConfig,
    MlpGeGluGoldenConfig,
    NormLinearGoldenConfig,
    PointwiseGoldenConfig,
    ReduceGoldenConfig,
    RmsNormGoldenConfig,
    RopeGoldenConfig,
    SoftmaxGoldenConfig,
)

_CASES = [
    MatmulGoldenConfig(name="m", M=512, N=4096, K=3840, dtype="fp16", knobs={}),
    MatmulGoldenConfig(name="m.dyn", M=512, N=4096, K=3840, dtype="fp16", knobs={}, dynamic=True),
    AttentionGoldenConfig(name="a", n_heads=16, seq=512, head_dim=256, knobs={}),
    AttentionGoldenConfig(name="a.dyn", n_heads=16, seq=512, head_dim=256, knobs={}, dynamic=True),
    RmsNormGoldenConfig(name="r", M=512, K=3840, knobs={}),
    RmsNormGoldenConfig(name="r.dyn", M=512, K=3840, knobs={}, dynamic=True),
    NormLinearGoldenConfig(name="nl", M=512, H=3840, N=4096, knobs={}),
    NormLinearGoldenConfig(name="nl.dyn", M=512, H=3840, N=4096, knobs={}, dynamic=True),
    MlpGeGluGoldenConfig(name="ggu", M=512, H=3840, inter=15360, knobs={}),
    MlpGeGluGoldenConfig(name="ggu.dyn", M=512, H=3840, inter=15360, knobs={}, dynamic=True),
    ReduceGoldenConfig(name="red", M=512, K=4096, knobs={}),
    SoftmaxGoldenConfig(name="s", M=512, K=4096, knobs={}),
    PointwiseGoldenConfig(name="p", M=512, N=4096, knobs={}),
    RopeGoldenConfig(name="rope", n_heads=16, seq=512, head_dim=256, knobs={}),
    EmbeddingGoldenConfig(name="emb", vocab=262144, seq=512, hidden=3840, knobs={}),
]


def _stamped_keys(cfg) -> list[ShapeKey]:
    """Trace the golden's snippet (symbolic when dynamic) to the loop dialect and key
    every stamped op — the same capture ``compiled_s_features`` uses, per node."""
    from emmy.commands.trace import graph_from_code
    from emmy.compiler.pipeline import LOOP_PASSES, Pipeline
    from emmy.compiler.trace.dynamic import build_torch_dynamic_shapes, parse_position_specs

    dyn = cfg.dynamic_specs()
    ds = build_torch_dynamic_shapes(parse_position_specs(dyn)) if dyn else None
    graph, _, _ = graph_from_code(cfg.snippet(), dynamic_shapes=ds)
    compiled = Pipeline.build(LOOP_PASSES).run(graph)
    keys = []
    for n in compiled.nodes.values():
        s = {k: v for k, v in (getattr(n.op, "knobs", {}) or {}).items() if k.startswith("S_")}
        if s:
            keys.append(ShapeKey.from_s_features(s))
    return keys


@pytest.mark.parametrize("cfg", _CASES, ids=lambda c: c.name)
def test_golden_key_round_trips_through_the_stamped_op(cfg):
    keys = _stamped_keys(cfg)
    assert keys.count(cfg.shape_key()) == 1, f"{cfg.name}: golden key {cfg.shape_key()} not uniquely stamped in {keys}"


def test_matmul_keys_carry_kind_and_aspect():
    """Every matmul key — golden side and op side — carries ``kind=""`` plus the
    ``free_max`` aspect discriminator (``max(M, N)``); the round-trip tests above pin
    that the stamped side fills the same value, so the joins keep matching."""
    got = MatmulGoldenConfig(name="m", M=512, N=4096, K=3840, dtype="fp16", knobs={}).shape_key()
    assert got == ShapeKey(free_prod=512 * 4096, reduce_max=3840, is_warp=True, free_max=4096)
    assert got.kind == ""


def test_free_max_splits_the_aspect_collision():
    """32×8192 and 512×512 share ``free_prod`` (the decode-twin vs square collision that
    let ``k_proj_global.s512`` silently shadow ``q_proj_global.m32``'s golden) — the
    ``free_max`` discriminator keys them apart. Dynamic and sweep-kind keys normalize
    ``free_max`` to 0, keeping those join classes untouched."""
    thin = ShapeKey.from_matmul(32, 8192, 3840, "fp16")
    square = ShapeKey.from_matmul(512, 512, 3840, "fp16")
    assert thin.free_prod == square.free_prod and thin != square
    assert ShapeKey.from_matmul(512, 4096, 3840, "fp16", dynamic=True).free_max == 0
    assert ShapeKey(free_prod=1, reduce_max=1, is_warp=True, kind="flash", free_max=7).free_max == 0


def test_fused_key_never_collides_with_rms_norm_or_matmul():
    """The fused computed-A key (``kind="fused"``) must not equal a bare RMSNorm sweep nor a plain
    ``mlp_gate_up`` matmul that happens to share extents — the ``kind`` discriminator keeps the
    three kernel families apart at the golden↔op join."""
    fused = NormLinearGoldenConfig(name="nl", M=512, H=3840, N=4096, knobs={}).shape_key()
    assert fused.kind == "fused"
    # A bare RMSNorm whose free product coincides (M*K == M*N) keys rms_norm, never fused.
    rms = RmsNormGoldenConfig(name="r", M=512, K=4096, knobs={}).shape_key()
    assert rms.kind == "rms_norm" and rms != fused
    # A plain matmul of the same output/contraction extents keys "" (warp contraction), never fused.
    mm = MatmulGoldenConfig(name="m", M=512, N=4096, K=3840, dtype="fp16", knobs={}).shape_key()
    assert mm.kind == "" and mm != fused


def test_attention_qk_contraction_never_joins_the_flash_golden():
    """The attention trace's OTHER stamped op — the QKᵀ contraction — keys as a plain
    contraction and must not equal the attention golden's flash key."""
    cfg = AttentionGoldenConfig(name="a", n_heads=16, seq=512, head_dim=256, knobs={})
    keys = _stamped_keys(cfg)
    others = [k for k in keys if k != cfg.shape_key()]
    assert others and all(k.kind == "" for k in others)


def test_sweep_classifier_from_measured_stamps():
    """Pure-logic pin of the classifier on measured stamp literals (no trace): the sweep
    identity is ``S_loop_depth < n_free + n_reduce + n_symbolic``; rsqrt marks RMSNorm,
    exp splits flash (3 free loops) from softmax; equality keeps ``kind=""``."""
    flash_dyn = {
        "S_ext_free_prod": 4096.0,
        "S_ext_reduce_max": 0.0,
        "S_ext_n_free_axis": 2.0,
        "S_ext_n_reduce_axis": 0.0,
        "S_ext_n_symbolic_axis": 4.0,
        "S_loop_depth": 4.0,
        "S_pw_exp": 2.0,
        "S_n_free_loop": 3.0,
    }
    assert ShapeKey.from_s_features(flash_dyn).kind == "flash"
    rms = {
        "S_ext_free_prod": 1966080.0,
        "S_ext_reduce_max": 3840.0,
        "S_ext_n_free_axis": 2.0,
        "S_ext_n_reduce_axis": 1.0,
        "S_ext_n_symbolic_axis": 0.0,
        "S_loop_depth": 2.0,
        "S_pw_rsqrt": 1.0,
        "S_dtype_f32": 5.0,
    }
    assert ShapeKey.from_s_features(rms).kind == "rms_norm"
    # The fused computed-A megakernel stamps LIKE rms_norm (rsqrt sweep) but with a SECOND reduce
    # axis (the contraction). ``is_warp`` is forced True even though its f32 statistic constants set
    # S_dtype_f32 (the dtype-multiset signal would wrongly read scalar for a warp mma).
    fused = {
        "S_ext_free_prod": 131072.0,
        "S_ext_reduce_max": 3840.0,
        "S_ext_n_free_axis": 2.0,
        "S_ext_n_reduce_axis": 2.0,
        "S_ext_n_symbolic_axis": 0.0,
        "S_loop_depth": 3.0,
        "S_pw_rsqrt": 1.0,
        "S_dtype_f32": 2.0,
        "S_dtype_f16": 6.0,
    }
    got = ShapeKey.from_s_features(fused)
    assert got.kind == "fused" and got.is_warp is True
    softmax = {
        "S_ext_free_prod": 2097152.0,
        "S_ext_reduce_max": 4096.0,
        "S_ext_n_free_axis": 2.0,
        "S_ext_n_reduce_axis": 2.0,
        "S_ext_n_symbolic_axis": 0.0,
        "S_loop_depth": 2.0,
        "S_pw_exp": 2.0,
        "S_n_free_loop": 2.0,
        "S_dtype_f32": 3.0,
    }
    assert ShapeKey.from_s_features(softmax).kind == "softmax"
    matmul = {
        "S_ext_free_prod": 2097152.0,
        "S_ext_reduce_max": 3840.0,
        "S_ext_n_free_axis": 2.0,
        "S_ext_n_reduce_axis": 1.0,
        "S_ext_n_symbolic_axis": 0.0,
        "S_loop_depth": 3.0,
        "S_pw_multiply": 1.0,
    }
    assert ShapeKey.from_s_features(matmul).kind == ""
    # Bare rows without stamps (arithmetic bases, old reservoir rows) stay plain.
    assert ShapeKey.from_s_features({"S_ext_free_prod": 512.0, "S_ext_reduce_max": 4096.0}).kind == ""


def test_joins_tolerates_sweep_dtype_flip_but_stays_strict_on_contractions():
    """``ShapeKey.joins``: a sweep op's ``is_warp`` derives from the operand-dtype
    multiset, which flips between an all-fp16 golden snippet (fp16-pure stamps) and the
    same norm in a served graph (f32 statistic constants) — so for sweep kinds the join
    ignores it. For ``kind == ""`` it stays exact: ``is_warp`` is the fp32/fp16
    contraction-twin discriminator."""
    golden_rms = ShapeKey(free_prod=4096, reduce_max=64, is_warp=False, kind="rms_norm")
    op_rms_fp16 = ShapeKey(free_prod=4096, reduce_max=64, is_warp=True, kind="rms_norm")
    assert op_rms_fp16.joins(golden_rms)
    assert not ShapeKey(free_prod=4096, reduce_max=128, is_warp=True, kind="rms_norm").joins(golden_rms)  # extents still bind
    assert not ShapeKey(free_prod=4096, reduce_max=64, is_warp=True, kind="softmax").joins(golden_rms)  # kind still binds
    mm_fp16 = ShapeKey.from_matmul(512, 512, 512, "fp16")
    mm_fp32 = ShapeKey.from_matmul(512, 512, 512, "fp32")
    assert mm_fp16.joins(mm_fp16) and not mm_fp16.joins(mm_fp32)  # the real twin pair never merges


def test_f8_dtype_class_splits_the_storage_twins():
    """The ``dtype_class`` field (M2a of the FP8 plan): every pre-existing key keeps its
    identity (``""`` default), an fp8-B key differs from its bf16 twin at the same
    ``(M, N, K)``, and the two constructors agree on the class."""
    # Existing-key stability: a pre-M2a key spelled without the field equals the same
    # key spelled today, and the 16-bit family never stamps a class.
    assert ShapeKey.from_matmul(512, 4096, 3840, "fp16") == ShapeKey(free_prod=512 * 4096, reduce_max=3840, is_warp=True, free_max=4096)
    assert ShapeKey.from_matmul(512, 4096, 3840, "bf16").dtype_class == ""
    # fp8 splits from the 16-bit twins; f16/bf16 still share one key (same bytes, same atoms).
    fp8 = ShapeKey.from_matmul(512, 4096, 3840, "fp8")
    bf16 = ShapeKey.from_matmul(512, 4096, 3840, "bf16")
    assert fp8.dtype_class == "f8" and fp8.is_warp is True
    assert fp8 != bf16 and not fp8.joins(bf16)
    assert ShapeKey.from_matmul(512, 4096, 3840, "fp16") == bf16
    # Every fp8 spelling names the one class.
    for spelling in ("fp8", "f8e4m3", "f8e5m2"):
        assert ShapeKey.from_matmul(512, 4096, 3840, spelling).dtype_class == "f8"


def test_f8_key_agrees_between_from_matmul_and_from_s_features():
    """The op side reads the class off the stamped dtype multiset (``992`` generates
    ``S_dtype_*`` generically from buffer dtype names, so ``S_dtype_f8e4m3`` stamps with
    no stamp-side change) and forces ``is_warp`` — the fp8 kernel's f32 scale constant
    would otherwise flip the dtype-multiset warp signal, the ``"fused"`` hazard again."""
    s = {
        "S_ext_free_prod": 512.0 * 4096.0,
        "S_ext_free_max": 4096.0,
        "S_ext_reduce_max": 3840.0,
        "S_ext_n_free_axis": 2.0,
        "S_ext_n_reduce_axis": 1.0,
        "S_ext_n_symbolic_axis": 0.0,
        "S_loop_depth": 3.0,
        "S_pw_multiply": 1.0,
        "S_dtype_f8e4m3": 1.0,
        "S_dtype_f32": 1.0,  # the scale constant's load
        "S_dtype_f16": 1.0,
    }
    got = ShapeKey.from_s_features(s)
    assert got == ShapeKey.from_matmul(512, 4096, 3840, "fp8")
    assert got.is_warp is True and got.dtype_class == "f8"
    # e5m2 storage lands in the same class.
    s5 = dict(s)
    s5["S_dtype_f8e5m2"] = s5.pop("S_dtype_f8e4m3")
    assert ShapeKey.from_s_features(s5).dtype_class == "f8"


def test_code_rate_splits_two_rates_of_one_shape():
    """``k_bits`` (VQ Phase 4): an EXL3 "optimized" rung allocates bits per tensor by Hessian
    sensitivity, so ONE ``(M, N, K)`` appears at several rates in one checkpoint — the pinned
    GLM-4.5-Air 2.25 ships ``mlp.shared_experts.gate_proj`` at 2, 3 and 4 bits. The codes slab's
    size and the decode's per-element word math both scale with the rate, so those are different
    kernels; without the rate in the key they would fight over one golden."""
    k2 = ShapeKey.from_matmul(1, 11008, 4096, "trellis", k_bits=2)
    k3 = ShapeKey.from_matmul(1, 11008, 4096, "trellis", k_bits=3)
    assert (k2.k_bits, k3.k_bits) == (2, 3)
    assert k2 != k3 and not k2.joins(k3) and not k3.joins(k2)
    assert len({k2, k3}) == 2  # hashable and distinct — they index a golden dict separately


def test_code_rate_never_perturbs_a_non_coded_key():
    """The rate is normalized off every other storage class, so no shipped golden / DB row moves:
    a key spelled without the field equals the same key spelled with a stray rate, and neither
    ``from_matmul`` nor ``from_s_features`` can put a rate on an uncoded shape."""
    plain = ShapeKey(free_prod=512 * 4096, reduce_max=3840, is_warp=True, free_max=4096)
    assert ShapeKey(free_prod=512 * 4096, reduce_max=3840, is_warp=True, free_max=4096, k_bits=4) == plain
    for dtype in ("fp32", "fp16", "bf16", "fp8", "f8e4m3"):
        assert ShapeKey.from_matmul(512, 4096, 3840, dtype, k_bits=4).k_bits == 0
    # A stamped rate with no coded carrier (``S_dtype_i16``) is likewise dropped.
    s = {"S_ext_free_prod": 2097152.0, "S_ext_reduce_max": 3840.0, "S_dtype_f16": 1.0, "S_trellis_k_bits": 4.0}
    assert ShapeKey.from_s_features(s).k_bits == 0
    assert ShapeKey.from_s_features(s) == ShapeKey.from_s_features({k: v for k, v in s.items() if k != "S_trellis_k_bits"})


def test_code_rate_agrees_between_from_matmul_and_from_s_features():
    """The op side reads the rate off ``S_trellis_k_bits``, stamped over the same load walk that
    writes ``S_dtype_i16`` — so the two are present together and the golden ↔ measured join holds
    at the rate as well as the shape."""
    s = {
        "S_ext_free_prod": 11008.0,
        "S_ext_free_max": 11008.0,
        "S_ext_reduce_max": 4096.0,
        "S_ext_n_free_axis": 1.0,
        "S_ext_n_reduce_axis": 1.0,
        "S_loop_depth": 2.0,
        "S_dtype_i16": 1.0,
        "S_dtype_f16": 1.0,
        "S_trellis_k_bits": 3.0,
    }
    assert ShapeKey.from_s_features(s) == ShapeKey.from_matmul(1, 11008, 4096, "trellis", k_bits=3)
    assert not ShapeKey.from_s_features(s).joins(ShapeKey.from_matmul(1, 11008, 4096, "trellis", k_bits=2))
