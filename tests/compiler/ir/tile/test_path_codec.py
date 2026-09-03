"""The structural PLACE path codec and its Fold-tree walker."""

from __future__ import annotations

import pytest

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.ir.tile.path import Site, _spellings, canonical, parse_key, resolve, sites, spell
from emmy.compiler.pipeline.passes.lowering.tile._fromloop import fold_from_loop
from tests.compiler.terms import contraction, projection


def _contraction_fold(k_name: str = "k", *, a=None, n_name: str = "n", acc: str = "acc0", w: str = "W") -> Fold:
    """A stored bilinear ``Fold`` — pure algebra, no placement/schedule fields."""
    return contraction(
        k_name,
        a if a is not None else Load(name="a_e", input="A", index=(Var("m"), Var(k_name))),
        (Load(name="b_e", input=w, index=(Var(k_name), Var(n_name))), acc),
    )


def _planar_fold(k_name: str = "k", *, acc: str = "s0", val: str = "v1", load: str = "x") -> Fold:
    """A λ-spelled PLANAR statistic fold (``acc += x²`` with loads inline)."""
    accum = Accum(name=acc, value=val, op="add", axes=(k_name,))
    body = Body(
        (
            Load(name=f"{load}_e", input=load, index=(Var("m"), Var(k_name))),
            Assign(name=val, op="multiply", args=(f"{load}_e", f"{load}_e")),
            accum,
        )
    )
    loop = Loop(axis=Axis(k_name, 512), body=body)
    fold = fold_from_loop(loop)
    assert fold.lift is not None
    return fold


def _cone(stat: Fold, out: str = "xn") -> Fold:
    """The norm→linear cone shape: the normalize cell over the sweep projection over the stat."""
    pro = projection((stat,), (Assign(name="rr", op="rsqrt", args=(stat.exposes[0],)),))
    cell = (
        Load(name="xc", input="x", index=(Var("m"), Var("k"))),
        Assign(name=out, op="multiply", args=("xc", "rr")),
    )
    return projection((pro,), cell)


def _norm_linear_tree() -> tuple[Fold, Fold, Fold]:
    """A root projection over the product fold, whose shared-A edge is the cone, whose prologue
    wraps the stat fold; the stat reduces the SAME ``k`` name the product contracts."""
    stat = _planar_fold("k")
    product = _contraction_fold("k", a=_cone(stat))
    root = projection((product,))
    return root, product, stat


# ---- canonical placement spellings -------------------------------------------------------------- #


def test_place_spelling_round_trips_for_nested_edges() -> None:
    root, product, stat = _norm_linear_tree()
    for node in (product, stat):
        key = spell(root, "PLACE", node)
        assert resolve(root, key).node is node
        assert canonical(root, key) == key


def test_schedule_families_are_not_structural_paths() -> None:
    root = _contraction_fold()
    for family in ("TILE", "REDUCE", "STAGE"):
        with pytest.raises(ValueError, match="not a structural path family"):
            resolve(root, family)


def test_deep_identical_placement_paths_take_the_ordinal() -> None:
    path = ("map", *("fold" for _ in range(1000)))
    first = Site(node=object(), axis="k", segments=path, ordinal=1)
    second = Site(node=object(), axis="k", segments=path, ordinal=2)

    assert _spellings("PLACE", second, (first, second)) == f"PLACE@{'.'.join(path)}.k2"


# ---- reserved grammar ---------------------------------------------------------------------------- #


@pytest.mark.parametrize("key", ["PLACE@in.a", "PLACE@in.b.fold.k", "PLACE@=x17"])
def test_reserved_graph_placement_forms_are_rejected(key: str) -> None:
    with pytest.raises(ValueError, match="reserved for graph-level placement"):
        parse_key(key)


def test_unknown_segment_rejected() -> None:
    with pytest.raises(ValueError, match="unknown path segment"):
        parse_key("PLACE@cone.fold.k")


# ---- the walker ---------------------------------------------------------------------------------- #


def test_walker_enumerates_paths_axes_and_ordinals() -> None:
    root, product, stat = _norm_linear_tree()
    by_node = {id(s.node): s for s in sites(root)}
    assert by_node[id(product)].segments == ("map", "fold")
    assert by_node[id(product)].axis == "k"
    assert by_node[id(stat)].segments == ("map", "fold", "a", "map", "fold")
    assert by_node[id(stat)].axis == "k"
    # A contraction's operands are labelled by ROLE, and a slab is no site: the product's A cone is
    # its one operand site, the B slab takes no path, and every site here is unique on its own.
    cone, b_slab = product.operands
    assert by_node[id(cone)].segments == ("map", "fold", "a") and by_node[id(cone)].ordinal == 1
    assert id(b_slab) not in by_node
    assert by_node[id(product)].ordinal == by_node[id(stat)].ordinal == 1


def test_site_is_a_frozen_value() -> None:
    s = Site(node=None, axis="k", segments=("fold",))
    assert s.depth == 1 and s.ordinal == 1


def test_exact_full_path_outranks_deeper_subsequence_matches() -> None:
    """A shallow site's full path is an anchored subsequence of every deeper same-axis path, so
    its canonical full-path + ordinal spelling admits the deeper sites too — the exact-path
    preference at the ambiguity point is what lets the spelling name its own site (the quant
    cone's nested a1 reduce chain)."""
    shallow1 = _planar_fold("a1", acc="s0", val="v1", load="x")
    shallow2 = _planar_fold("a1", acc="t0", val="w1", load="z")
    deep_inner = _planar_fold("a1", acc="u0", val="q1", load="y")
    deep = projection((deep_inner,), (Assign(name="d", op="add", args=(deep_inner.exposes[0], "one")),))
    root = projection(
        (shallow1, shallow2, deep),
        (Assign(name="o", op="add", args=(shallow1.exposes[0], shallow2.exposes[0])), Assign(name="p", op="add", args=("o", "d"))),
    )
    for node in (shallow1, shallow2, deep_inner):
        key = spell(root, "PLACE", node)
        assert resolve(root, key).node is node, key


def test_exact_full_path_uses_the_ordinal_reading_that_matched() -> None:
    """An axisless placement edge spells its final path segment plus ordinal as one component.

    Once the literal-axis reading fails, exact-path precedence must compare against the effective
    segment-plus-ordinal reading, not the original literal parse. Otherwise a deeper path that
    admits the shallow path as a subsequence makes the shallow site's canonical key ambiguous.
    """
    head = Site(node=object(), axis="a", segments=("map", "fold"))
    shallow = Site(node=object(), axis=None, segments=("map", "fold", "a", "fold", "b"))
    deep = Site(node=object(), axis=None, segments=("map", "fold", "a", "map", "fold", "fold", "b"))
    all_sites = (head, shallow, deep)

    key = _spellings("PLACE", shallow, all_sites)
    assert key == "PLACE@map.fold.a.fold.b1"
    assert resolve(None, key, all_sites=all_sites).node is shallow.node


def test_tile_axis_orientation_is_read_once_per_site(monkeypatch) -> None:
    """Candidate plans share a site's output-axis orientation: placement is a site fact, derived
    once and memoized, however many plans ask."""
    from emmy.compiler.ir.tile import Placement
    from emmy.compiler.ir.tile import ops as tile_ops

    root, _, _ = _norm_linear_tree()
    calls = []
    original = tile_ops.Sched._derive_mn

    def spy(sched, node):
        calls.append(node)
        return original(sched, node)

    monkeypatch.setattr(tile_ops.Sched, "_derive_mn", spy)
    axes = (Axis("m", 128), Axis("n", 256))
    tile = TileOp(op=root, axes=(*axes, Axis("k", 256)))
    product = tile_ops.head(tile.op)  # the identity wrapper over the product dissolves at construction
    sched = tile_ops.Sched(tile, place=Placement(free=axes, grid=axes))
    assert sched._mn_for(product) == axes
    assert sched._mn_for(product) == axes
    # both output axes are answered by ONE derivation, and the site memo keeps the second
    # ``_mn_for`` from asking again
    assert len(calls) == 1
