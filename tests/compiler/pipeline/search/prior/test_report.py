"""The evaluation report — cells, exclusions, and the two questions it keeps apart.

Everything here is synthetic: the report computes no metric of its own (``search/metrics.py`` owns those and
has its own tests) and reads no file. What it DOES own is which pools a number covers, and that is what these
tests pin — an aggregate that quietly averaged in the pools it could not measure would look healthier than the
model is, which is the failure this whole module exists to prevent.
"""

from __future__ import annotations

import numpy as np

from emmy.compiler.pipeline.search.data.group import GoldenGroup, MeasuredGroup
from emmy.compiler.pipeline.search.prior.report import EvalReport, golden_cells, measured_cells, pool_bucket


def _measured(key: str, latencies: list[float], *, gpu: str = "card-a", h_opt: float = 3.0) -> MeasuredGroup:
    """A benched pool whose rows carry one feature, ``D_a``, ascending — so a scorer that ranks by ``D_a``
    ranks them in a known order and the metric values are hand-checkable."""
    feats = [{"D_a": float(i)} for i in range(len(latencies))]
    return MeasuredGroup.from_measured(key, gpu, key, h_opt, latencies, feats)


def _golden(key: str, n_rows: int, golden_row: int, *, total: int | None = None, tier: str = "warp") -> GoldenGroup:
    feats = [{"D_a": float(i)} for i in range(n_rows)]
    return GoldenGroup.from_dicts(key, key, tier, "card-a", key, golden_row, feats, total)


def _by_d_a(group):
    """Ranks a pool by its ``D_a`` column DESCENDING — quality, higher = predicted faster."""
    return group.matrix(["D_a"])[:, 0]


def test_regret_is_the_ratio_the_pick_costs_over_the_measured_best():
    """A perfect ranking is 1.00x; a model that picks the second-best pays exactly that row's ratio."""
    # ``_by_d_a`` calls row 2 fastest. Make row 2 the measured best, then make it the measured worst.
    perfect = measured_cells("offline", [_measured("k", [30.0, 20.0, 10.0])], _by_d_a)[0]
    missed = measured_cells("offline", [_measured("k", [10.0, 20.0, 30.0])], _by_d_a)[0]

    assert perfect.metrics["regret1"]["median"] == 1.0
    assert missed.metrics["regret1"]["median"] == 3.0  # picked 30 µs where 10 µs was there
    assert perfect.metrics["spearman"] is not None  # computed only above the minimum pool size — see below


def test_a_pool_too_small_for_a_metric_is_excluded_and_the_count_says_so():
    """Each metric has its own minimum and its own group count. A cell holding a one-row pool, a three-row
    pool and a twelve-row pool covers a different number of pools with each metric, and the difference is
    the whole reason the counts travel with the values."""
    groups = [_measured("one", [10.0]), _measured("three", [30.0, 20.0, 10.0]), _measured("twelve", [float(12 - i) for i in range(12)])]
    cell = measured_cells("offline", groups, _by_d_a)[0]

    assert cell.groups == 3
    assert cell.metrics["regret1"]["groups"] == 2  # the one-row pool has no second candidate to be wrong about
    assert cell.metrics["spearman"]["groups"] == 1  # needs five rows
    assert cell.metrics["regret10"]["groups"] == 1  # needs eleven — a top-10 over ten rows is the whole pool


def test_a_pool_the_model_cannot_score_is_counted_not_dropped():
    """A linear model with no dynamic weight set answers ``None`` for every symbolic-axis pool. Dropping
    those silently would report a healthy static corpus with no sign that the rest is unmeasured."""
    groups = [_measured("a", [20.0, 10.0]), _measured("b", [20.0, 10.0])]
    cell = measured_cells("offline", groups, lambda g: None if g.key == "b" else _by_d_a(g))[0]

    assert (cell.groups, cell.unscored) == (2, 1)
    assert cell.metrics["regret1"]["groups"] == 1


def test_measured_cells_key_on_card_and_compile_regime():
    """Cards never pool, and neither do nvcc regimes — ``-O1`` and ``-O3`` reorder the same candidates."""
    groups = [
        _measured("a", [20.0, 10.0], gpu="card-a", h_opt=3.0),
        _measured("b", [20.0, 10.0], gpu="card-a", h_opt=1.0),
        _measured("c", [20.0, 10.0], gpu="card-b", h_opt=3.0),
    ]
    cells = measured_cells("offline", groups, _by_d_a)

    assert [(c.axes["gpu"], c.axes["H_opt"]) for c in cells] == [("card-a", "O1"), ("card-a", "O3"), ("card-b", "O3")]
    assert all(c.axes["half"] == "offline" for c in cells)


def test_golden_cells_stratify_by_pool_size():
    """A rank is only readable against the pool it was scored over: a corpus whose small pools rank
    perfectly and whose large ones rank at chance must not report one flattering middle number."""
    assert [pool_bucket(n) for n in (5, 99, 100, 999, 1_000, 9_999, 10_000, 10**6)] == [
        "<100",
        "<100",
        "<1k",
        "<1k",
        "<10k",
        "<10k",
        ">=10k",
        ">=10k",
    ]
    groups = [_golden("small", 3, 2, total=40), _golden("huge", 3, 0, total=200_000)]
    cells = golden_cells("offline", groups, _by_d_a)

    assert [c.axes["pool"] for c in cells] == ["<100", ">=10k"]
    assert cells[0].metrics["rank"]["median"] == 0  # ``_by_d_a`` ranks row 2 first, and row 2 is the golden
    assert cells[1].metrics["rank"]["median"] == 2  # the golden is row 0, which this scorer ranks last
    assert (cells[0].metrics["top1"]["count"], cells[1].metrics["top1"]["count"]) == (1, 0)


def test_a_pool_with_several_verified_rows_is_scored_on_the_best_of_them():
    """Deploy ships one config, so any acceptable one ranked first is the same win."""
    feats = [{"D_a": float(i)} for i in range(4)]
    both = GoldenGroup.from_dicts("k", "k", "warp", "card-a", "k", (0, 3), feats)
    (cell,) = golden_cells("offline", [both], _by_d_a)

    assert cell.metrics["rank"]["median"] == 0  # row 3 ranks first; row 0 ranking last does not count against it


def test_the_report_serializes_to_a_diffable_schema():
    """Two runs of the same evaluation must produce byte-identical JSON — it is compared with ``diff``."""
    cells = measured_cells("offline", [_measured("k", [30.0, 20.0, 10.0])], _by_d_a)
    report = EvalReport({"dataset": "nodes", "source": "freeze"}, cells)
    obj = report.to_json()

    assert obj["header"] == {"dataset": "nodes", "source": "freeze"}
    assert obj["cells"][0]["axes"] == {"half": "offline", "gpu": "card-a", "H_opt": "O3"}
    assert obj["cells"][0]["groups"] == 1 and obj["cells"][0]["unscored"] == 0
    assert EvalReport({"dataset": "nodes", "source": "freeze"}, cells).to_json() == obj


def test_both_halves_are_labelled_in_one_report():
    """The two halves fail for different reasons, so an unlabelled number destroys the diagnostic."""
    groups = [_measured("k", [30.0, 20.0, 10.0])]
    cells = measured_cells("offline", groups, _by_d_a) + measured_cells("online", groups, lambda g: -_by_d_a(g))

    assert [c.axes["half"] for c in cells] == ["offline", "online"]
    assert [c.metrics["regret1"]["median"] for c in cells] == [1.0, 3.0]  # each half's own answer


def test_only_the_metrics_with_a_size_minimum_carry_their_own_count():
    """A count that could only ever equal the cell's own scored total is noise repeated once per metric — and
    on the golden side that is every one of them, since no rank metric excludes a pool for being small."""
    (measured,) = measured_cells("offline", [_measured("k", [30.0, 20.0, 10.0])], _by_d_a)
    (golden,) = golden_cells("offline", [_golden("k", 3, 0)], _by_d_a)

    assert all("groups" in block for block in measured.metrics.values())
    assert not any("groups" in block for block in golden.metrics.values())


def test_quality_is_negated_into_a_cost_exactly_once():
    """The rank family takes quality (higher = faster) and the regret family a cost (lower = faster). Getting
    that backwards silently reverses every answer, so it is pinned on a pool where the two disagree."""
    group = _measured("k", [10.0, 20.0, 30.0])  # row 0 is fastest; ``_by_d_a`` calls row 2 fastest
    (cell,) = measured_cells("offline", [group], _by_d_a)

    assert cell.metrics["spearman"] is not None
    (flipped,) = measured_cells("offline", [group], lambda g: -_by_d_a(g))
    assert flipped.metrics["regret1"]["median"] == 1.0  # the flipped scorer agrees with the hardware
    assert np.isclose(cell.metrics["regret1"]["median"], 3.0)
