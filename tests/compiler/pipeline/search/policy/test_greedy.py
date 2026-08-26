"""Focused tests for greedy schedule-space traversal."""

from types import SimpleNamespace

from emmy.compiler.pipeline.fork import Level, build_fork_tree
from emmy.compiler.pipeline.search.policy.greedy import _branch_schedule_pick, _direct_measured_pick, tile_identity


def test_schedule_pick_scores_complete_representatives_one_level_at_a_time() -> None:
    materialized = []
    rows = [{"TILE": str(tile), "STAGE": str(stage)} for tile in range(100) for stage in range(100)]
    tree = build_fork_tree(
        params=rows,
        levels=(Level(("TILE",), lambda row: (row["TILE"],)), Level(("STAGE",), lambda row: (row["STAGE"],))),
        materialize=lambda row: materialized.append(row),
    )
    batch_sizes = []

    class Prior:
        def mean_scores(self, offered):
            batch_sizes.append(len(offered))
            return [-(int(row["TILE"]) + int(row["STAGE"])) for row in offered]

    point = SimpleNamespace(
        options=[tree],
        node_id="node",
        root_op=SimpleNamespace(knobs={}),
        ctx=SimpleNamespace(features=lambda: {}),
    )
    leaf, knobs, price = _branch_schedule_pick(point, None, Prior())

    assert knobs == {"TILE": "99", "STAGE": "99"}
    assert leaf.knobs == knobs
    assert price == -198
    assert batch_sizes == [100, 100]
    assert materialized == []


def test_schedule_pick_descends_directly_to_complete_measured_row() -> None:
    materialized = []
    rows = [{"TILE": str(tile), "STAGE": str(stage)} for tile in range(100) for stage in range(100)]
    tree = build_fork_tree(
        params=rows,
        levels=(Level(("TILE",), lambda row: (row["TILE"],)), Level(("STAGE",), lambda row: (row["STAGE"],))),
        materialize=lambda row: materialized.append(row),
    )

    class Prior:
        def _o3_evidence(self):
            return {frozenset({("S_shape", 128)}): [({"TILE": "42", "STAGE": "73"}, 1.25)]}

    point = SimpleNamespace(
        options=[tree],
        node_id="node",
        root_op=SimpleNamespace(knobs={"S_shape": 128}),
        ctx=SimpleNamespace(features=lambda: {"H_opt": 3.0}),
    )
    leaf, knobs, price = _direct_measured_pick(point, None, Prior(), {})

    assert knobs == {"TILE": "42", "STAGE": "73"}
    assert leaf.knobs == knobs
    assert price == 1.25
    assert materialized == []


def test_schedule_pick_keeps_branch_with_an_unblocked_descendant() -> None:
    rows = [
        {"TILE": "0", "STAGE": "0"},
        {"TILE": "0", "STAGE": "1"},
        {"TILE": "1", "STAGE": "0"},
    ]
    tree = build_fork_tree(
        params=rows,
        levels=(Level(("TILE",), lambda row: (row["TILE"],)), Level(("STAGE",), lambda row: (row["STAGE"],))),
        materialize=lambda row: row,
    )

    class Prior:
        def mean_scores(self, offered):
            return [int(row["TILE"]) for row in offered]

    point = SimpleNamespace(options=[tree], node_id="node", root_op=SimpleNamespace(knobs={}), ctx=SimpleNamespace(features=lambda: {}))
    _, knobs, _ = _branch_schedule_pick(point, {"node": {tile_identity(rows[0])}}, Prior())

    assert knobs == rows[1]
