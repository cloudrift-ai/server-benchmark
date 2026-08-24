"""``group_measured`` — benched node rows packed into comparison sets.

A group here is the set of configs that genuinely competed: one op, one card, one nvcc regime. What makes
that non-trivial is that three plausible spellings of "one op" are wrong, and two kinds of row carry a
number that is not a measurement. Each is pinned below, because every one of them fails silently — a
misgrouped or mislabelled pool still produces a confident-looking correlation.
"""

from __future__ import annotations

from emmy.compiler.pipeline.search.data.group import GoldenGroup, Group, group_measured
from tests.compiler.pipeline.search.helpers import F16_MATMUL_FEATS
from tests.compiler.pipeline.search.helpers import GPU_5090 as _GPU
from tests.compiler.pipeline.search.helpers import node_row as _row

_GPU2 = "NVIDIA GeForce RTX 4090"


def _feats(*, opt: float = 3.0, **knobs) -> dict:
    """A row the admission filter accepts — the shared plausible fixture plus the regime stamps the filter
    requires. Inventing a smaller dict here would sidestep the very gate these tests are about."""
    return {**F16_MATMUL_FEATS, "H_cc": 120.0, "H_opt": opt, **knobs}


def test_configs_that_competed_land_in_one_group():
    rows = [_row(f"n{i}", value_us=200.0 + 100 * i, features=_feats(TILE=f"f2x{2 + i}")) for i in range(4)]
    (group,), dropped = group_measured(rows)
    assert not dropped
    assert group.labels.tolist() == [200.0, 300.0, 400.0, 500.0]
    assert len(group.feats) == 4 and group.gpu == _GPU


def test_the_key_separates_cards_ops_and_nvcc_regimes():
    """Three axes, three reasons. Cards never pool. Ops never pool — and ``op_sig`` is the DB's own identity,
    not the ``S_*`` signature, which descent stamps its own deltas into and which would therefore split one
    op's alternative schedules apart. Regimes never pool because ``-O1`` and ``-O3`` invert."""
    rows = [
        _row("a", value_us=200.0, features=_feats(opt=3.0)),
        _row("b", value_us=300.0, features=_feats(opt=1.0)),  # same op, other regime
        _row("c", value_us=400.0, features=_feats(opt=3.0), op_sig="other"),
        _row("d", value_us=500.0, features=_feats(opt=3.0), gpu=_GPU2),
    ]
    groups, dropped = group_measured(rows)
    assert not dropped
    assert len(groups) == 4 and all(len(g.feats) == 1 for g in groups)


def test_one_op_scheduled_two_ways_stays_one_group():
    """The reason the key is ``op_sig`` and not the structural signature: these two rows are alternative
    SCHEDULES for one op, and a key that split them would leave two groups of one row each — nothing to
    rank, and the comparison the metric exists for silently gone."""
    rows = [
        _row("a", value_us=200.0, features=_feats(TILE="f2x2", S_loop_depth=3.0)),
        _row("b", value_us=400.0, features=_feats(TILE="f8x8", S_loop_depth=2.0)),  # descent moved an S_* stamp
    ]
    (group,), _ = group_measured(rows)
    assert len(group.feats) == 2


def test_a_branchs_value_is_not_a_measurement():
    """``value_us`` on a branch is a min over its explored subtree — a coverage bound nobody benched.
    ``None`` is the pre-enrichment unknown and is not ``True`` either."""
    rows = [
        _row("leaf", value_us=500.0, features=_feats()),
        _row("branch", value_us=200.0, features=_feats(TILE="f4x4"), is_leaf=False),
        _row("old", value_us=300.0, features=_feats(TILE="f8x8"), is_leaf=None),
    ]
    (group,), dropped = group_measured(rows)
    assert dropped == {"non-leaf (branch or pre-enrichment row)": 2}
    assert group.labels.tolist() == [500.0]


def test_a_failed_bench_carries_a_sentinel_not_a_latency():
    """The watchdog sentinel is a huge POSITIVE number, so nothing downstream rejects it on sign — left in a
    group it is a row every model ranks last for free, inflating every correlation computed over the pool."""
    rows = [
        _row("ok", value_us=500.0, features=_feats()),
        _row("fail", value_us=1e9, features=_feats(TILE="f4x4"), status="bench_fail"),
    ]
    (group,), dropped = group_measured(rows)
    assert dropped == {"bench_fail": 1}
    assert group.labels.tolist() == [500.0]


def test_every_row_is_either_grouped_or_counted():
    """The counts are returned rather than logged because a report must publish them: "ρ=0.6 over 340 groups"
    means something different when a tenth of the corpus was dropped than when none was."""
    rows = [
        _row("a", value_us=500.0, features=_feats()),
        _row("b", value_us=1e9, features=_feats(TILE="f4x4"), status="bench_fail"),
        _row("c", value_us=200.0, features=_feats(TILE="f8x8"), is_leaf=False),
        _row("d", value_us=300.0, features=_feats(TILE="f2x8"), op_sig=""),
    ]
    groups, dropped = group_measured(rows)
    assert sum(len(g.feats) for g in groups) + sum(dropped.values()) == len(rows)
    assert dropped == {"bench_fail": 1, "non-leaf (branch or pre-enrichment row)": 1, "unkeyed": 1}


# --- the two label kinds -----------------------------------------------------------


def test_only_a_golden_pool_can_be_asked_which_rows_are_the_answer():
    """The two kinds differ in what can be ASKED of them, which is why they are two types. A measured pool
    has no ``golden_ids`` at all — not an empty one, not a raising one — so a rank metric that needs them
    says so by taking :class:`GoldenGroup`, and nothing has to check a flag at runtime."""
    rows = [_row(f"n{i}", value_us=200.0 + 100 * i, features=_feats(TILE=f"f2x{2 + i}")) for i in range(3)]
    (measured,), _ = group_measured(rows)
    golden = GoldenGroup.from_dicts("g/x", "x", "warp", "g", "x", 1, [{"D_a": float(i)} for i in range(3)])

    assert measured.labels.tolist() == [200.0, 300.0, 400.0]  # the measurement IS the label
    assert golden.golden_ids == (1,)
    assert isinstance(golden, Group) and not hasattr(measured, "golden_ids")


def test_goldens_go_in_as_row_indices_and_come_back_as_row_indices():
    """A group is built ONCE, already knowing every golden it holds, so there is no amend-after-construction
    path — several arrive together, in any order, with duplicates, and the marker encoding never leaves the
    class."""
    feats = [{"D_a": float(i)} for i in range(5)]
    assert GoldenGroup.from_dicts("g/x", "x", "warp", "g", "x", 2, feats).golden_ids == (2,)
    assert GoldenGroup.from_dicts("g/y", "y", "warp", "g", "y", (4, 2, 4), feats).golden_ids == (2, 4)


def test_a_row_the_freeze_would_refuse_is_refused_here_too():
    """Admission is ``freeze_reason``, not a second list of rules — which matters most for the two it adds.

    A row in a retired featurizer vocabulary means something different from what its numbers say, and an
    implausibly fast one is far worse than the ``bench_fail`` sentinel this module excludes by hand: the
    sentinel makes ONE row rank last, while a phantom optimum becomes the group's ``min`` and makes every
    other row in the pool look bad. Both are invisible on freeze input — a freeze passed this filter at
    write time — so only a live tune DB can produce them, and only a test like this one can show it."""
    rows = [
        _row("honest", value_us=500.0, features=_feats()),
        _row("stale", value_us=100.0, features=_feats(TILE="f4x4"), feat_ver=1),
        _row("phantom", value_us=0.05, features=_feats(TILE="f8x8")),
    ]
    groups, dropped = group_measured(rows)
    assert [g.labels.tolist() for g in groups] == [[500.0]]
    assert sorted(dropped) == ["implausible value", "stale feat_ver 1 != current 3"]
