from __future__ import annotations

import pytest

from emmy.compiler.context import Context
from emmy.compiler.pipeline.search.policy.greedy import _cold_materialized_mma_lead

VOLTA = "mma_m8n8k4_f16_f32"


def _base(*, m: int = 512, n: int = 4096, k: int = 4096, opt: int = 3) -> dict:
    return {
        "H_opt": float(opt),
        "H_sm_count": 80.0,
        "S_ext_n_free_axis": 2.0,
        "S_ext_free_prod": float(m * n),
        "S_ext_free_max": float(max(m, n)),
        "S_ext_reduce_max": float(k),
    }


def _row(base: dict, work: str, tile: str, *, stage: str = "d1/sync", reduce: str = "") -> dict:  # noqa: A002
    return {**base, "WORK": work, "TILE": f"{VOLTA}/{tile}", "STAGE": stage, "REDUCE": reduce, "RASTER": ""}


def test_large_square_prefers_balanced_staged_volta_row() -> None:
    base = _base()
    rows = [
        _row(base, "w16x1", "f1x8/k8", stage=""),
        _row(base, "w2x4", "f4x2/k4"),
        _row(base, "w4x2", "f2x4/k4"),
        _row(base, "w2x2", "f4x4/k4"),
        _row(base, "w4x4", "f2x2/k4"),
    ]

    assert _cold_materialized_mma_lead(rows, Context.from_target((7, 0))) == 1


def test_narrow_output_keeps_enough_ctas_with_64_square() -> None:
    base = _base(n=1024)
    rows = [
        _row(base, "w8x1", "f1x8/k8", stage=""),
        _row(base, "w2x4", "f4x2/k4"),  # 128x128: only 32 output CTAs
        _row(base, "w2x2", "f2x2/k4"),  # 64x64: 128 output CTAs
        _row(base, "w1x4", "f4x1/k4"),
    ]

    assert _cold_materialized_mma_lead(rows, Context.from_target((7, 0))) == 2


def test_medium_m_derives_deep_atomic_split_and_wider_k_chunk() -> None:
    base = _base(m=32)
    rows = [
        _row(base, "w4x1", "f1x4/k8", stage="", reduce="g2a"),
        _row(base, "w2x2", "f1x1/k4"),
        _row(base, "w2x2", "f1x1/k8", reduce="g2a"),
        _row(base, "w2x2", "f1x1/k8", reduce="g4a"),
        _row(base, "w2x2", "f1x1/k8", reduce="g8a"),
    ]

    assert _cold_materialized_mma_lead(rows, Context.from_target((7, 0))) == 4


def test_medium_wide_n_uses_rectangular_tile_and_shallow_split() -> None:
    base = _base(m=32, n=14336)
    rows = [
        _row(base, "w2x2", "f1x1/k4", reduce="g2a"),
        _row(base, "w1x4", "f2x1/k4", reduce="g2a"),
        _row(base, "w1x4", "f2x1/k4", reduce="g8a"),
    ]

    assert _cold_materialized_mma_lead(rows, Context.from_target((7, 0))) == 1


def test_symbolic_free_axis_uses_scheduler_hint_geometry() -> None:
    # The fusion-time histogram excludes symbolic M (so it sees only N), while the scheduler
    # enumerates against Dim.hint.  Its typed facts must drive the identical M512 static rank.
    base = {
        **_base(),
        "S_ext_n_free_axis": 1.0,
        "S_ext_free_prod": 4096.0,
        "S_ext_free_max": 4096.0,
        "S_ext_n_symbolic_axis": 1.0,
        "S_hint_n_free_axis": 2.0,
        "S_hint_free_prod": float(512 * 4096),
        "S_hint_free_max": 4096.0,
    }
    rows = [
        _row(base, "w4x1", "f2x8/k2", stage="", reduce="g2a"),
        _row(base, "w2x4", "f4x2/k4"),
        _row(base, "w2x2", "f2x2/k4"),
    ]

    assert _cold_materialized_mma_lead(rows, Context.from_target((7, 0))) == 1


@pytest.mark.parametrize("reverse", [False, True])
def test_finalized_split_row_does_not_hide_unsplit_staged_candidate(reverse: bool) -> None:
    """Each structural alternative owns its geometry; catalog order cannot lend it row zero's."""
    split_base = {
        **_base(k=448),
        "S_ext_n_free_axis": 2.0,
        "S_ext_free_prod": float(32 * 4096),
        "S_ext_free_max": 4096.0,
        "S_ext_n_symbolic_axis": 1.0,
        "S_hint_n_free_axis": 3.0,
        "S_hint_free_prod": float(32 * 512 * 4096),
        "S_hint_free_max": 4096.0,
    }
    unsplit_base = {
        **_base(k=14336),
        "S_ext_n_free_axis": 1.0,
        "S_ext_free_prod": 4096.0,
        "S_ext_free_max": 4096.0,
        "S_ext_n_symbolic_axis": 1.0,
        "S_hint_n_free_axis": 2.0,
        "S_hint_free_prod": float(512 * 4096),
        "S_hint_free_max": 4096.0,
    }
    split = _row(split_base, "w16x2", "f8x2/k2", stage="")
    staged = _row(unsplit_base, "w2x4", "f4x2/k4")
    rows = [split, staged]
    if reverse:
        rows.reverse()

    lead = _cold_materialized_mma_lead(rows, Context.from_target((7, 0)))
    assert lead is not None and rows[lead] is staged


@pytest.mark.parametrize("base", [_base(m=1), _base(opt=1), _base(k=512)])
def test_decode_o1_and_short_k_retain_prior(base: dict) -> None:
    rows = [_row(base, "w2x4", "f4x2/k4")]

    assert _cold_materialized_mma_lead(rows, Context.from_target((7, 0))) is None


def test_other_target_atom_transport_retains_prior() -> None:
    base = _base()
    rows = [
        {
            **base,
            "WORK": "w2x4",
            "TILE": "mma_m16n8k16_f16_f32/f4x2/k2",
            "STAGE": "d2/cp",
            "REDUCE": "",
            "RASTER": "",
        }
    ]

    assert _cold_materialized_mma_lead(rows, Context.from_target((8, 0))) is None
    # A Volta-only atom in a row for another target is also ignored.
    assert _cold_materialized_mma_lead([_row(base, "w2x4", "f4x2/k4")], Context.from_target((8, 0))) is None
