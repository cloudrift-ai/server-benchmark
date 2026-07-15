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
    MatmulGoldenConfig,
    PointwiseGoldenConfig,
    ReduceGoldenConfig,
    RmsNormGoldenConfig,
    SoftmaxGoldenConfig,
)

_CASES = [
    MatmulGoldenConfig(name="m", M=512, N=4096, K=3840, dtype="fp16", knobs={}),
    MatmulGoldenConfig(name="m.dyn", M=512, N=4096, K=3840, dtype="fp16", knobs={}, dynamic=True),
    AttentionGoldenConfig(name="a", n_heads=16, seq=512, head_dim=256, knobs={}),
    AttentionGoldenConfig(name="a.dyn", n_heads=16, seq=512, head_dim=256, knobs={}, dynamic=True),
    RmsNormGoldenConfig(name="r", M=512, K=3840, knobs={}),
    RmsNormGoldenConfig(name="r.dyn", M=512, K=3840, knobs={}, dynamic=True),
    ReduceGoldenConfig(name="red", M=512, K=4096, knobs={}),
    SoftmaxGoldenConfig(name="s", M=512, K=4096, knobs={}),
    PointwiseGoldenConfig(name="p", M=512, N=4096, knobs={}),
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


def test_matmul_keys_are_byte_compatible_with_the_pre_kind_constructor():
    """Every matmul key — golden side and op side — carries ``kind=""``, so joins built
    before the discriminator existed keep matching (the whole point of the default)."""
    got = MatmulGoldenConfig(name="m", M=512, N=4096, K=3840, dtype="fp16", knobs={}).shape_key()
    assert got == ShapeKey(free_prod=512 * 4096, reduce_max=3840, is_warp=True)
    assert got.kind == ""


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
