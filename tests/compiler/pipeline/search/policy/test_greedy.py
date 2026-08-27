"""Focused tests for greedy schedule-space traversal."""

from types import SimpleNamespace

from emmy.compiler.pipeline.fork import Level, build_fork_tree
from emmy.compiler.pipeline.search.policy.greedy import _direct_measured_pick


def test_schedule_pick_descends_directly_to_complete_measured_row() -> None:
    materialized = []
    rows = [{"TILE": str(tile), "STAGE": str(stage)} for tile in range(100) for stage in range(100)]
    tree = build_fork_tree(
        params=rows,
        levels=(Level(("TILE",), lambda row: (row["TILE"],)), Level(("STAGE",), lambda row: (row["STAGE"],))),
        materialize=lambda row: materialized.append(row),
    )

    point = SimpleNamespace(
        options=[tree],
        node_id="node",
        root_op=SimpleNamespace(knobs={"S_shape": 128}),
        ctx=SimpleNamespace(features=lambda: {"H_opt": 3.0}),
    )
    index = {frozenset({("S_shape", "128")}): [({"TILE": "42", "STAGE": "73"}, 1.25)]}
    leaf, knobs, price = _direct_measured_pick(point, None, index)

    assert knobs == {"TILE": "42", "STAGE": "73"}
    assert leaf.knobs == knobs
    assert price == 1.25
    assert materialized == []
