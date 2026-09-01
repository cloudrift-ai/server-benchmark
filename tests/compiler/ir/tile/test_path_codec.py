"""The structural PLACE path codec and its Fold-tree walker."""

from __future__ import annotations

import pytest

from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.pure.fold import Channel, Fold
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.ir.tile.path import Site, _spellings, canonical, family_sites, parse_key, primary, resolve, sites, spell
from emmy.compiler.pipeline.passes.lowering.tile._fromloop import fold_from_loop


def _contraction_fold(k_name: str = "k", *, a=None, n_name: str = "n", acc: str = "acc0", w: str = "W") -> Fold:
    """A stored a bilinear ``Fold`` (1s) — pure algebra, no placement/schedule fields."""
    return Fold.contraction(
        k_axis=Axis(k_name, 256),
        a=a if a is not None else Load(name="a_e", input="A", index=(Var("m"), Var(k_name))),
        channels=(Channel(b=Load(name="b_e", input=w, index=(Var(k_name), Var(n_name))), acc=acc),),
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
    loop = Loop(axis=Axis(k_name, 512), body=body, role=AxisRole.PLANAR)
    fold = fold_from_loop(loop)
    assert fold.lift is not None
    return fold


def _cone(stat: Fold, out: str = "xn") -> Fold:
    """The norm→linear cone shape: ``Fold.projection(body=normalize, operands=(Fold.projection(body=sweep, operands=(stat,)),))``."""
    pro = Fold.projection(body=Body((Assign(name="rr", op="rsqrt", args=(stat.out,)),)), operands=(stat,))
    cell = Body(
        (
            Load(name="xc", input="x", index=(Var("m"), Var("k"))),
            Assign(name=out, op="multiply", args=("xc", "rr")),
        )
    )
    return Fold.projection(body=cell, operands=(pro,))


def _norm_linear_tree() -> tuple[Fold, Fold, Fold]:
    """``Fold.projection(proj, operands=(product,))`` — the product fold's shared-A edge is the cone, whose
    prologue wraps the stat fold; the stat reduces the SAME ``k`` name the product contracts."""
    stat = _planar_fold("k")
    product = _contraction_fold("k", a=_cone(stat))
    root = Fold.projection(operands=(product,))
    return root, product, stat


def _flash_tree() -> tuple[Fold, Fold, Fold, Fold]:
    """The flash shape, λ-spelled (step 7 — mirroring ``_flash._flash_op``): the stream fold
    reduces ``kv`` with the QK (axis ``dd``) score fold hoisted as ``operands[0]`` and the value
    ``Load`` as ``operands[1]``; the PV (axis ``pj``) contraction is DERIVED — synthesized into
    the blocked evaluation, found among ``stream.step_stmts()``.

    Both are read by AXIS, not by position: the derived step PLACES an inline-node edge at the
    first read of its bound name, so the score sits after any lift stmt that precedes it (here the
    ``scale`` ``Load``)."""
    from emmy.compiler.ir.pure import Lambda
    from emmy.compiler.ir.pure.carrier import exp_combine_states

    qk = _contraction_fold("dd", acc="sacc", w="K")
    names = ("m_i", "l_i", "O_i")
    other = tuple(f"{n}__o" for n in names)
    lift = Lambda(
        params=("kv", "sacc", "v_e"),
        body=Body((Load(name="scale", input="_scale", index=()), Assign(name="s", op="multiply", args=("sacc", "scale")))),
        results=("s", 1.0, "v_e"),
    )
    stream = Fold(
        axis=Axis("kv", 512),
        operands=(qk, Load(name="v_e", input="V", index=(Var("kv"), Var("d")))),
        lift=lift,
        init=(float("-inf"), 0.0, 0.0),
        combine=Lambda(params=names + other, body=Body(exp_combine_states(names, other)), results=names),
    )
    pv = next(s for s in stream.step_stmts() if isinstance(s, Fold) and s.axis.name == "pj")  # the derived PV site
    root = Fold.projection(operands=(stream,))
    return root, stream, qk, pv


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
    assert all(s.ordinal == 1 for s in sites(root))


def test_family_sites_and_primary() -> None:
    root, stream, qk, _pv = _flash_tree()
    all_sites = sites(root)
    placements = family_sites("PLACE", all_sites)
    assert {id(s.node) for s in placements} == {id(stream), id(qk)}
    assert primary("PLACE", placements).node is stream


def test_root_placement_path_is_explicit_without_changing_the_bare_primary() -> None:
    root, stream, _qk, _pv = _flash_tree()

    assert spell(root, "PLACE", root) == "PLACE@root"
    assert resolve(root, "PLACE@root").node is root
    assert canonical(root, "PLACE@root") == "PLACE@root"
    assert resolve(root, "PLACE").node is stream


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
    deep = Fold.projection(body=Body((Assign(name="d", op="add", args=(deep_inner.out, "one")),)), operands=(deep_inner,))
    root = Fold.projection(
        body=Body((Assign(name="o", op="add", args=(shallow1.out, shallow2.out)), Assign(name="p", op="add", args=("o", "d")))),
        operands=(shallow1, shallow2, deep),
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


def test_identity_miss_is_loud_never_the_untiled_path() -> None:
    """A copied node (``dataclasses.replace`` — structurally equal, a different object) is NOT a
    site of the tree: addressing a schedule read through it must raise ``UnknownSiteError``, never
    resolve to ``None``/untiled. A node that IS a site but not of the asked family keeps its plain
    ``ValueError`` — the "family doesn't apply" decline the accessors swallow."""
    import dataclasses

    from emmy.compiler.ir.tile.ops import Sched
    from emmy.compiler.ir.tile.path import UnknownSiteError

    root, stream, qk, pv = _flash_tree()
    copy = dataclasses.replace(qk)
    assert copy == qk and copy is not qk
    sched = Sched(TileOp(op=root))
    with pytest.raises(UnknownSiteError):
        sched.site_of(copy)
    with pytest.raises(UnknownSiteError):
        sched.key("TILE", copy)
    assert sched.site_of(qk).node is qk
    plain = _planar_fold()
    wrapper = Fold.projection(body=Body((Assign(name="o", op="copy", args=(plain.out,)),)), operands=(plain,))
    assert Sched(TileOp(op=wrapper)).key("TILE", plain) is None  # a real site, family declines — not an identity miss


def test_tile_axis_orientation_is_read_once_per_site(monkeypatch) -> None:
    """Candidate plans share a site's output-axis orientation; the computed cone lowers once."""
    from emmy.compiler.ir.tile import Placement
    from emmy.compiler.ir.tile import ops as tile_ops

    root, product, _ = _norm_linear_tree()
    calls = []
    original = tile_ops.edge_free_axes

    def spy(edge):
        calls.append(edge)
        return original(edge)

    monkeypatch.setattr(tile_ops, "edge_free_axes", spy)
    axes = (Axis("m", 128), Axis("n", 256))
    sched = tile_ops.Sched(TileOp(op=root), place=Placement(free=axes, grid=axes))
    assert sched._mn_for(product) == axes
    assert sched._mn_for(product) == axes
    # both output axes are answered by ONE reading of the edge, and the site memo keeps the second
    # ``_mn_for`` from asking again
    assert len(calls) == 1
