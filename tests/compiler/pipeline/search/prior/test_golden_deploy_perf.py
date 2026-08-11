"""``diagnostics.golden_deploy_perf`` — the no-rebench ``vs gold`` column for
``emmy eval prior``: per golden shape, the deployable (-O3) latency of the
prior's predicted-best reservoir config over the golden's recorded ``emmy_us``.

The load-bearing case is dtype separation: an fp32 square and its ``.fp16`` twin
share (free-dim product, reduce extent), so the shape key MUST split on
``S_dtype_f32`` — keying on an mma marker (which the fp16 row may not carry) merges
them and steals the fp16 latency for the fp32 row.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from emmy.compiler.pipeline.search import golden as golden_mod
from emmy.compiler.pipeline.search.data import ShapeKey
from emmy.compiler.pipeline.search.prior import diagnostics

_FP32_US = 10.0
_FP16_US = 20.0


@pytest.fixture(autouse=True)
def _single_gpu_goldens(monkeypatch):
    """Install the three shape records needed by these join tests."""
    records = [
        SimpleNamespace(
            name=name,
            gpu_name="NVIDIA GeForce RTX 5090",
            compute_cap=(12, 0),
            shape_key=ShapeKey.from_matmul(size, size, size, dtype),
            emmy_us=latency,
            fast_math=False,
            is_matmul=True,
        )
        for name, size, dtype, latency in (
            ("matmul.square.512", 512, "fp32", _FP32_US),
            ("matmul.square.512.fp16", 512, "fp16", _FP16_US),
            ("matmul.square.1024", 1024, "fp32", 30.0),
        )
    ]
    monkeypatch.setattr(golden_mod, "GOLDEN_RECORDS", records)
    monkeypatch.setattr(golden_mod, "live_recorded_goldens", lambda: None)


class _FakePrior:
    """A prior with a hand-built reservoir; ``mean_score`` is constant (each group
    here has one -O3 row, so the argmin pick is unambiguous). ``pick`` mirrors the
    real ``Prior.pick`` model-argmin fallback (constant scores → first row)."""

    def __init__(self, rows):
        self._dataset = rows  # list[(stamped_knobs, latency_us)]

    def mean_score(self, _feats):
        return 0.0

    def pick(self, rows):
        scores = [self.mean_score(r) for r in rows]
        best_i = min(range(len(scores)), key=scores.__getitem__)
        return best_i, scores[best_i]


def _row(free_prod, reduce_max, *, fp32, h_opt, latency):
    knobs = {
        "H_opt": float(h_opt),
        "S_ext_free_prod": float(free_prod),
        "S_ext_free_max": float(round(free_prod**0.5)),  # fixtures are square matmuls
        "S_ext_reduce_max": float(reduce_max),
        ("S_dtype_f32" if fp32 else "S_dtype_f16"): 2.0,
        # The matmul histogram markers (_matmul_sig): product → reduce-add, 2 inputs.
        "S_reduce_add": 1.0,
        "S_pw_multiply": 1.0,
        "S_n_distinct_input": 2.0,
        "BM": 8,
        "BN": 16,
    }
    return (knobs, latency)


def test_dtype_separation_and_o3_filter():
    fp = 512 * 512  # both square.512 and .fp16 share (free_prod, reduce)

    prior = _FakePrior(
        [
            _row(fp, 512, fp32=True, h_opt=3, latency=_FP32_US * 2.0),  # fp32 -O3 winner → ratio 2.0
            _row(fp, 512, fp32=False, h_opt=3, latency=_FP16_US * 0.5),  # fp16 -O3 winner → ratio 0.5
            _row(fp, 512, fp32=True, h_opt=1, latency=_FP32_US * 9.0),  # -O1 row must be IGNORED
        ]
    )
    perf = diagnostics.golden_deploy_perf(prior)

    # Each dtype matches its own -O3 row — no cross-contamination (the merge bug would
    # make the fp32 row pick the smaller fp16 latency).
    assert perf["matmul.square.512"] == pytest.approx(2.0)
    assert perf["matmul.square.512.fp16"] == pytest.approx(0.5)


def test_shape_without_o3_is_omitted():
    # square.1024 has only an -O1 row → no deployable measurement → omitted ('—').
    prior = _FakePrior([_row(1024 * 1024, 1024, fp32=True, h_opt=1, latency=50.0)])
    perf = diagnostics.golden_deploy_perf(prior)
    assert "matmul.square.1024" not in perf


def test_kernel_filter_restricts_shapes():
    prior = _FakePrior([_row(512 * 512, 512, fp32=True, h_opt=3, latency=_FP32_US)])
    assert set(diagnostics.golden_deploy_perf(prior, "matmul.square.512")) <= {"matmul.square.512"}


def test_non_matmul_group_with_colliding_extents_is_excluded():
    """A reduce-shaped op group that happens to share a matmul golden's
    (free_prod, reduce_max, dtype) must not satisfy the join — the index admits
    only matmul-histogram groups (``_matmul_sig``)."""
    knobs, latency = _row(512 * 512, 512, fp32=True, h_opt=3, latency=_FP32_US * 0.1)
    knobs.pop("S_pw_multiply")  # no product feeding the reduce → not a matmul body
    assert "matmul.square.512" not in diagnostics.golden_deploy_perf(_FakePrior([(knobs, latency)]))
