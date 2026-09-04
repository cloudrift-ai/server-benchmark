"""The structural placement path codec — routes over the stored Fold tree.

What these pin: the grammar (``PLACE@<kind>.<i>/…/<kind>``: departures with the operand taken, the
arrival's kind last; bare is sugar); the walker (every non-slab node is one site at its first
position, its route the departures that reach it, a contraction's operands taken by stored
position); spelling (bare for a family's one site, the full route otherwise — no ordinal, no
axis name, no shortest-unique search); resolution (a route walked hop by hop, a bare key to the
one site or the primary, a stale key failing loudly at the segment whose kind is not what stands
there); and the round trip ``resolve(spell(node))`` being the node."""

from __future__ import annotations

import pytest

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.pure import Fold
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop
from emmy.compiler.ir.tile.path import MissingSiteError, canonical, family_sites, kind, parse_key, primary, resolve, sites, spell
from emmy.compiler.pipeline.passes.lowering.tile._fromloop import fold_from_loop
from tests.compiler.terms import contraction, projection, slab


def _row_stat(k_name: str = "k", *, load: str = "x", acc: str = "acc0", val: str = "v1") -> Fold:
    """``Σ_k x[m,k]²`` — a planar reduce formed by the lift, its load a slab operand."""
    body = Body(
        (
            Load(name=f"{load}_e", input=load, index=(Var("m"), Var(k_name))),
            Assign(name=val, op="multiply", args=(f"{load}_e", f"{load}_e")),
            Accum(name=acc, value=val, op=ElementwiseImpl("add"), axes=(k_name,)),
        )
    )
    return fold_from_loop(Loop(axis=Axis(k_name, 512), body=body))


def _norm_linear_tree() -> tuple[Fold, Fold, Fold]:
    """``root map → inner`` whose A is the normalize cone ``x · rsqrt(stat)`` and whose B is a slab:
    the norm→linear shape. Returns ``(root, product, stat)``."""
    stat = _row_stat("k")
    pro = projection((stat,), (Assign(name="rr", op="rsqrt", args=(stat.exposes[0],)),))
    cone = projection((pro, slab("xc", "x", "m", "k")), (Assign(name="xn", op="multiply", args=("xc", "rr")),))
    product = contraction("k", cone, (slab("b_e", "W", "k", "n"), "acc"))
    root = projection((product,), (Assign(name="o", op="relu", args=("acc",)),))
    return root, product, stat


# ---- the grammar --------------------------------------------------------------------------------- #


def test_a_key_is_departures_then_the_arrival() -> None:
    key = parse_key("PLACE@map.1/inner.2/map")
    assert key.family == "PLACE" and key.hops == (("map", 1), ("inner", 2)) and key.target == "map"
    assert parse_key("PLACE").bare and parse_key("PLACE@reduce").hops == () and parse_key("PLACE@reduce").target == "reduce"


@pytest.mark.parametrize("key", ["PLACE@in.a", "PLACE@in.b.fold.k", "PLACE@=x17"])
def test_reserved_forms_rejected(key: str) -> None:
    with pytest.raises(ValueError, match="reserved"):
        parse_key(key)


@pytest.mark.parametrize(
    ("key", "reason"),
    [
        ("PLACE@cone", "names the node arrived at by kind"),
        ("PLACE@map.1/fold", "names the node arrived at by kind"),
        ("PLACE@map/inner", "needs a 1-based operand index"),
        ("PLACE@map.0/inner", "needs a 1-based operand index"),
        ("PLACE@map.01/inner", "needs a 1-based operand index"),
        ("PLACE@a.1/inner", "unknown path segment"),
    ],
)
def test_a_segment_off_the_grammar_is_rejected(key: str, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        parse_key(key)


# ---- the walker ---------------------------------------------------------------------------------- #


def test_the_walker_routes_by_kind_and_stored_position() -> None:
    root, product, stat = _norm_linear_tree()
    by_node = {id(s.node): s for s in sites(root)}
    assert by_node[id(root)].hops == () and by_node[id(root)].path == "map"
    assert by_node[id(product)].hops == (("map", 1),) and by_node[id(product)].path == "map.1/inner"
    cone, weight = product.operands
    assert by_node[id(cone)].path == "map.1/inner.1/map"
    # A slab is no site: the product's B takes no route, and the cone's own slab operand neither.
    assert id(weight) not in by_node and all(s.node.as_slab() is None for s in by_node.values())
    # The stat sits under the cone's prologue, its route spelling every departure on the way.
    assert by_node[id(stat)].path == "map.1/inner.1/map.1/map.1/reduce" and by_node[id(stat)].axis == "k"
    assert {kind(s.node) for s in by_node.values()} == {"map", "inner", "reduce"}


def test_a_shared_subterm_is_one_site_at_its_first_position() -> None:
    stat = _row_stat("k")
    left = projection((stat,), (Assign(name="l", op="rsqrt", args=(stat.exposes[0],)),))
    right = projection((stat,), (Assign(name="r", op="exp", args=(stat.exposes[0],)),))
    root = projection((left, right), (Assign(name="o", op="add", args=("l", "r")),))
    routes = [s.path for s in sites(root) if s.node is stat]
    assert routes == ["map.1/map.1/reduce"]


def test_a_site_knows_what_it_lies_under() -> None:
    root, product, stat = _norm_linear_tree()
    by_node = {id(s.node): s for s in sites(root)}
    assert by_node[id(stat)].under(by_node[id(product)]) and by_node[id(stat)].under(by_node[id(root)])
    assert not by_node[id(product)].under(by_node[id(stat)]) and not by_node[id(product)].under(by_node[id(product)])


# ---- spelling and resolution --------------------------------------------------------------------- #


def test_a_familys_one_site_spells_bare_and_several_spell_their_routes() -> None:
    root, product, stat = _norm_linear_tree()
    cone = product.operands[0]
    assert spell(root, "PLACE", product) == "PLACE@map.1/inner"
    assert spell(root, "PLACE", cone) == "PLACE@map.1/inner.1/map"
    lone = projection((_row_stat("k"),), (Assign(name="o", op="rsqrt", args=("acc0",)),))
    assert spell(lone, "PLACE", lone.operands[0]) == "PLACE"


def test_resolution_round_trips_and_a_bare_key_names_the_primary() -> None:
    root, product, stat = _norm_linear_tree()
    all_sites = sites(root)
    for site in family_sites("PLACE", all_sites):
        assert resolve(root, spell(root, "PLACE", site.node)).node is site.node
        assert canonical(root, spell(root, "PLACE", site.node)) == spell(root, "PLACE", site.node)
    assert primary("PLACE", family_sites("PLACE", all_sites)).node is product
    assert resolve(root, "PLACE").node is product
    assert resolve(projection((), (Assign(name="y", op="relu", args=("x",)),)), "PLACE") is None


def test_a_stale_key_fails_at_the_segment_that_no_longer_stands() -> None:
    root, product, stat = _norm_linear_tree()
    with pytest.raises(MissingSiteError, match="segment 1 stands on a map, not a inner"):
        resolve(root, "PLACE@inner.1/map")
    with pytest.raises(MissingSiteError, match="takes operand 2 of 1"):
        resolve(root, "PLACE@map.2/inner")
    with pytest.raises(MissingSiteError, match="arrives at a inner, not a reduce"):
        resolve(root, "PLACE@map.1/reduce")
    with pytest.raises(MissingSiteError, match="arrives at a slab"):
        resolve(root, "PLACE@map.1/inner.2/map")
    with pytest.raises(MissingSiteError, match="the tree's sites: map.1/inner"):
        resolve(root, "PLACE@map.2/inner")


def test_a_bare_key_over_tied_shallowest_sites_is_ambiguous() -> None:
    stat = _row_stat("k")
    left = projection((stat,), (Assign(name="l", op="rsqrt", args=(stat.exposes[0],)),))
    right = projection((_row_stat("k", load="y", acc="acc1", val="v2"),), (Assign(name="r", op="exp", args=("acc1",)),))
    root = projection((left, right), (Assign(name="o", op="add", args=("l", "r")),))
    with pytest.raises(ValueError, match="PLACE is ambiguous: use PLACE@map.1/map or PLACE@map.2/map"):
        resolve(root, "PLACE")


def test_a_key_names_a_family_this_tree_does_not_know() -> None:
    root, *_ = _norm_linear_tree()
    with pytest.raises(ValueError, match="not a structural path family"):
        resolve(root, "TILE@map.1/inner")
