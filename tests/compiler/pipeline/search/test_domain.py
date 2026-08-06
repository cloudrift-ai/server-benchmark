"""The constrained-integer-set domain: declaration, pruning, enumeration and membership."""

from __future__ import annotations

from math import prod

import pytest

from emmy.compiler.pipeline.search.domain import Bound, Dimension, Space


def test_unconstrained_space_is_the_cartesian_product_in_declaration_order():
    """No bounds ⇒ every combination, first dimension slowest, last fastest."""
    space = Space(dims=(Dimension("a", (1, 2)), Dimension("b", (1, 2, 4))))
    assert list(space) == [
        {"a": 1, "b": 1},
        {"a": 1, "b": 2},
        {"a": 1, "b": 4},
        {"a": 2, "b": 1},
        {"a": 2, "b": 2},
        {"a": 2, "b": 4},
    ]


def test_le_bound_applies_the_coefficient():
    """``32·wm·wn ≤ 1024`` is the CTA thread budget: it keeps exactly the pairs with ``wm·wn ≤ 32``."""
    space = Space(
        dims=(Dimension("wm", (1, 2, 4, 8, 16)), Dimension("wn", (1, 2, 4, 8, 16))),
        bounds=(Bound(("wm", "wn"), limit=1024, coeff=32),),
    )
    assert all(p["wm"] * p["wn"] <= 32 for p in space)
    assert {(p["wm"], p["wn"]) for p in space} == {(m, n) for m in (1, 2, 4, 8, 16) for n in (1, 2, 4, 8, 16) if m * n <= 32}


def test_a_repeated_dimension_contributes_once_per_occurrence():
    """``x·x ≤ 16`` is a square budget, not a linear one."""
    space = Space(dims=(Dimension("x", (1, 2, 3, 4, 5, 6)),), bounds=(Bound(("x", "x"), limit=16),))
    assert [p["x"] for p in space] == [1, 2, 3, 4]


def test_non_power_of_two_values_are_ordinary_members():
    """The point of integer coordinates: 26 = 2·13 and 9 = 3² need no special casing — the golden
    ``f4x26`` / ``f2x9`` register tiles are members like any other."""
    space = Space(
        dims=(Dimension("reg_n", (1, 2, 4)), Dimension("reg_m", (1, 2, 4, 6, 8, 9, 10, 12, 14, 26))),
        bounds=(Bound(("reg_n", "reg_m"), limit=128),),
    )
    pts = list(space)
    assert {"reg_n": 4, "reg_m": 26} in pts
    assert {"reg_n": 2, "reg_m": 9} in pts


def test_pruning_makes_a_tightly_bounded_large_space_enumerable():
    """Five dimensions of 64 values is 64⁵ ≈ 1.07e9 combinations; the ``≤ 64`` product bound drops
    each prefix as it overruns, so the walk touches a few thousand. Without prefix pruning this
    test does not finish."""
    dims = tuple(Dimension(f"x{i}", tuple(range(1, 65))) for i in range(5))
    space = Space(dims=dims, bounds=(Bound(tuple(d.name for d in dims), limit=64),))
    pts = list(space)
    assert all(prod(p.values()) <= 64 for p in pts)
    assert len(pts) < 10_000
    assert {"x0": 64, "x1": 1, "x2": 1, "x3": 1, "x4": 1} in pts


def test_multiple_bounds_all_apply():
    space = Space(
        dims=(Dimension("wm", (1, 2, 4, 8)), Dimension("wn", (1, 2, 4, 8)), Dimension("fn", (1, 2, 4, 8))),
        bounds=(Bound(("wm", "wn"), limit=256, coeff=32), Bound(("wn", "fn"), limit=16)),
    )
    assert {(p["wm"], p["wn"], p["fn"]) for p in space} == {
        (m, n, f) for m in (1, 2, 4, 8) for n in (1, 2, 4, 8) for f in (1, 2, 4, 8) if m * n <= 8 and n * f <= 16
    }


def test_a_bound_violating_point_is_not_enumerated():
    """Membership has ONE definition — the walk. A point over the bound, or off a dimension's
    declared values, simply never appears in it."""
    space = Space(dims=(Dimension("a", (1, 2, 4)), Dimension("b", (1, 2, 4))), bounds=(Bound(("a", "b"), limit=4),))
    pts = list(space)
    assert {"a": 2, "b": 2} in pts
    assert {"a": 4, "b": 4} not in pts  # violates the bound
    assert {"a": 3, "b": 1} not in pts  # 3 is not a declared value
    assert all(set(p) == {"a", "b"} for p in pts)  # every point binds exactly the declared dims


def test_prefix_pruning_drops_no_legal_point():
    """The walk prunes a prefix the moment a running product overruns, which is only sound because
    every value is ``≥ 1``. Checked against the unpruned truth: the full cartesian product filtered
    by the bounds, computed here independently of the walk that produced them."""
    space = Space(
        dims=(Dimension("a", (1, 2, 3, 4)), Dimension("b", (1, 2, 3, 4)), Dimension("c", (1, 2, 3, 4))),
        bounds=(Bound(("a", "b"), limit=8), Bound(("b", "c", "c"), limit=12, coeff=2)),
    )
    brute = {(a, b, c) for a in (1, 2, 3, 4) for b in (1, 2, 3, 4) for c in (1, 2, 3, 4) if a * b <= 8 and 2 * b * c * c <= 12}
    assert {(p["a"], p["b"], p["c"]) for p in space} == brute


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"name": "a", "values": ()}, "declares no values"),
        ({"name": "a", "values": (1, 0, 2)}, "non-positive"),
        ({"name": "a", "values": (1, -4)}, "non-positive"),
    ],
)
def test_dimension_rejects_an_unusable_domain(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Dimension(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dims": (), "limit": 8}, "names no dimension"),
        ({"dims": ("a",), "limit": 0}, "positive limit"),
        ({"dims": ("a",), "limit": 8, "coeff": 0}, "positive limit"),
    ],
)
def test_bound_rejects_a_malformed_declaration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Bound(**kwargs)


def test_space_rejects_duplicate_and_undeclared_dimensions():
    with pytest.raises(ValueError, match="duplicate dimension name"):
        Space(dims=(Dimension("a", (1,)), Dimension("a", (2,))))
    with pytest.raises(ValueError, match="a space declares no dimensions"):
        Space(dims=())
    with pytest.raises(ValueError, match="undeclared dimension"):
        Space(dims=(Dimension("a", (1,)),), bounds=(Bound(("a", "b"), limit=4),))


def test_an_over_tight_bound_yields_the_empty_set():
    """An empty enumeration is a legal answer, not an error — the caller decides what to do."""
    space = Space(dims=(Dimension("a", (4, 8)),), bounds=(Bound(("a",), limit=3),))
    assert list(space) == []


def test_bound_spelling_reads_back():
    assert Bound(("wm", "wn"), limit=1024, coeff=32).spell() == "32*wm*wn <= 1024"
    assert Bound(("bk",), limit=512).spell() == "bk <= 512"
