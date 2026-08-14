from __future__ import annotations

import pytest

from emmy.compiler.context import Context
from emmy.compiler.pipeline.search.policy._cold_reduce import cold_grid_reduction_lead


def _base(*, output: int = 4096, reduce: int = 14336, opt: int = 3) -> dict:  # noqa: A002
    return {
        "H_opt": float(opt),
        "H_sm_count": 80.0,
        "S_ext_n_free_axis": 2.0,
        "S_ext_free_prod": float(output),
        "S_ext_free_max": float(output),
        "S_ext_n_reduce_axis": 1.0,
        "S_ext_reduce_max": float(reduce),
    }


def _row(base: dict, reduce: str = "", *, work: str = "t256", tile: str = "", stage: str = "") -> dict:  # noqa: A002
    return {**base, "WORK": work, "TILE": tile, "REDUCE": reduce, "STAGE": stage, "RASTER": ""}


def test_long_scalar_matvec_prefers_deepest_useful_kernel_split() -> None:
    base = _base()
    rows = [
        {**base, "WORK": "", "TILE": "", "REDUCE": "", "STAGE": "", "RASTER": ""},
        _row(base, "g2k/coop-t"),
        _row(base, "g4k/coop-t"),
        _row(base, "g8k/coop-t"),
        _row(base, "g16k/coop-t"),
        _row(base, "g32k/coop-t"),
    ]

    assert cold_grid_reduction_lead(rows, Context.from_target((7, 0))) == 5


def test_split_depth_stops_before_threads_outnumber_k_slice() -> None:
    base = _base(output=512, reduce=512)
    rows = [
        _row(base, "g2k/coop-t"),
        _row(base, "g4k/coop-t"),
        _row(base, "g8k/coop-t"),
    ]

    assert cold_grid_reduction_lead(rows, Context.from_target((7, 0))) == 0


def test_equal_k_parallelism_uses_smaller_split_workspace() -> None:
    base = _base(output=1024, reduce=4096)
    rows = [
        _row(base, "g16k/coop-t", work="t128"),
        _row(base, "g8k/coop-t", work="t256"),
    ]

    assert cold_grid_reduction_lead(rows, Context.from_target((7, 0))) == 1


@pytest.mark.parametrize("reverse", [False, True])
def test_symbolic_outer_output_with_resident_ctas_prefers_unsplit(reverse: bool) -> None:
    """A static inner histogram must not make a hinted multi-row output look like decode."""
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
    split = _row(base, "g32k/coop-t")
    unsplit = _row(base, "coop-t")
    rows = [split, unsplit]
    if reverse:
        rows.reverse()

    lead = cold_grid_reduction_lead(rows, Context.from_target((7, 0)))
    assert lead is not None and rows[lead]["REDUCE"] == "coop-t"


@pytest.mark.parametrize(
    "rows",
    [
        [_row(_base(), "g16a/coop-t")],
        [_row(_base(), "g16k/coop")],
        [_row(_base(), "g16k/coop-t", stage="d1/sync")],
        [_row(_base(output=128), "g16k/coop-t")],
        [_row(_base(output=16384), "g16k/coop-t")],
        [_row(_base(opt=1), "g16k/coop-t")],
        [{**_base(), "WORK": "w1x1", "TILE": "mma_m8n8k4_f16_f32/f1x1", "REDUCE": "", "STAGE": "", "RASTER": ""}],
    ],
)
def test_foreign_or_unamortized_rows_retain_the_prior(rows: list[dict]) -> None:
    assert cold_grid_reduction_lead(rows, Context.from_target((7, 0))) is None
