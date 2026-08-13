"""Shared node-row builders for the search-package tests.

The dicts here are PHYSICS-CALIBRATED against the node-store plausibility gate and are
consumed by both ``test_node_gate.py`` (which tests the predicates directly) and
``test_freeze.py`` (whose filter composes them) — one copy, so a featurizer-vocabulary
or GPU-registry change cannot silently leave one suite exercising stale spellings.
"""

from __future__ import annotations

from emmy.compiler.pipeline.search.db import NodeRow

GPU_5090 = "NVIDIA GeForce RTX 5090"  # registry records fp32/fp16 peaks -> the plausibility gate is active

# The poisoned class's shape: symbolic-M mlp_down (free excludes the symbolic axis,
# benched at the dynamic hint), fp16 operands. 2*4096*14336*512 FLOPs. The stamps
# certify every loop multiplies the iteration space (depth 3 = 1 free + 1 reduce +
# 1 symbolic — the dynM matmul spelling), which is what licenses the free x red work
# formula: 9.17 us implies ~6500 TFLOP/s (implausible) while 500 us is honest.
F16_MATMUL_FEATS = {
    "S_ext_free_prod": 4096.0,
    "S_ext_reduce_max": 14336.0,
    "S_ext_n_free_axis": 1.0,
    "S_ext_n_reduce_axis": 1.0,
    "S_loop_depth": 3.0,
    "S_ext_n_symbolic_axis": 1.0,
    "S_dtype_f16": 2.0,
    "TILE@a2": "mma_m16n8k16_f16/f2x8/k8",
    "WORK": "w1x8",
    "REDUCE@a2": "g2k",
}


def impossible_staged_feats() -> dict:
    """The square.512.dynM residue: a cp.async-staged warp tile whose slab (139 KB for
    w1x8/f2x8/k8) exceeds the ~99 KB dynamic-smem cap could never launch — but the
    combine-only ~2 µs it left behind implies a LEGAL 133 TFLOP/s on that small shape,
    so only the kernel-validity check catches it. The same config unstaged is real."""
    return {
        **{k: v for k, v in F16_MATMUL_FEATS.items() if not k.startswith(("S_ext_free", "S_ext_reduce"))},
        "S_ext_free_prod": 512.0,
        "S_ext_reduce_max": 512.0,
        "STAGE@a2": "d1/cp",
    }


def node_row(key: str, *, value_us: float, features: dict | None = None, gpu: str = GPU_5090, **over) -> NodeRow:
    """A leaf ``NodeRow`` on a registry-known card, ``**over`` overriding any field."""
    kw = dict(
        node_key=key,
        parent_key=None,
        context_key="ctx",
        op_sig="op",
        features=dict(F16_MATMUL_FEATS if features is None else features),
        value_us=value_us,
        depth=5,
        gpu=gpu,
        visits=1,
        is_leaf=True,
        status="ok",
        run_id="run",
        measured_at="2026-07-09T00:00:00+00:00",
    )
    kw.update(over)
    return NodeRow(**kw)
