"""The tree-path knob codec (phase 2) — walker, resolver, canonical speller.

Pins the codec contract: short paths are canonical (bare for the primary node, ``FAMILY@<axis>``
when the axis alone discriminates, the shortest anchored path subsequence otherwise), resolution
is total over the sugar levels and idempotent, ambiguity and broken stored keys fail LOUDLY, the
ordinal exists only for true same-path collisions, and the reserved graph-level placement grammar
is rejected — never reused."""

from __future__ import annotations

import pytest

from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.pure.fold import Channel, Fold
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop
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


# ---- canonical spellings ---------------------------------------------------------------------- #


def test_bare_is_canonical_on_a_single_fold_kernel() -> None:
    root = _contraction_fold()
    for fam in ("TILE", "REDUCE", "STAGE"):
        assert spell(root, fam, root) == fam
        assert resolve(root, fam).node is root
    # The axis sugar resolves to the same site; its canonical stays the bare spelling.
    assert resolve(root, "TILE@k").node is root
    assert canonical(root, "TILE@k") == "TILE"


def test_projected_fold_keeps_bare_canonical() -> None:
    fold = _contraction_fold()
    root = Fold.projection(operands=(fold,))
    assert spell(root, "REDUCE", fold) == "REDUCE"
    assert resolve(root, "REDUCE").node is fold


def test_reduce_kernel_has_no_tile_site() -> None:
    stat = _planar_fold()
    root = Fold.projection(operands=(stat,))
    assert resolve(root, "TILE") is None  # the family doesn't apply — decided-empty stamps stay bare
    assert canonical(root, "TILE") is None
    assert resolve(root, "REDUCE").node is stat


def test_pointwise_root_projection_is_the_tile_site() -> None:
    root = Fold.projection(
        body=Body(
            (
                Load(name="xc", input="x", index=(Var("m"),)),
                Assign(name="r", op="relu", args=("xc",)),
            )
        ),
        operands=(),
    )
    site = resolve(root, "TILE")
    assert site is not None and site.node is root and site.axis is None
    assert spell(root, "TILE", root) == "TILE"
    assert resolve(root, "REDUCE") is None


def test_flash_bare_tile_is_ambiguous_and_axis_keys_are_canonical() -> None:
    root, stream, qk, pv = _flash_tree()
    with pytest.raises(ValueError, match="TILE@dd or TILE@pj"):
        resolve(root, "TILE")
    assert spell(root, "TILE", qk) == "TILE@dd"
    assert spell(root, "TILE", pv) == "TILE@pj"
    assert resolve(root, "TILE@dd").node is qk
    assert resolve(root, "TILE@pj").node is pv
    # The stream is the primary REDUCE / STAGE site — bare keeps its meaning (today's stored rows).
    assert resolve(root, "REDUCE").node is stream
    assert resolve(root, "STAGE").node is stream
    assert spell(root, "REDUCE", stream) == "REDUCE"
    # The in-step folds stay explicitly addressable.
    assert resolve(root, "REDUCE@dd").node is qk
    assert canonical(root, "REDUCE@dd") == "REDUCE@dd"


def test_norm_linear_bare_resolves_to_the_product_and_the_stat_spells_the_path() -> None:
    root, product, stat = _norm_linear_tree()
    # The bare-resolution guard: bare REDUCE means the contraction's K fold (the primary).
    assert resolve(root, "REDUCE").node is product
    assert spell(root, "REDUCE", product) == "REDUCE"
    assert spell(root, "TILE", product) == "TILE"
    # The stat reduces the SAME k name — the axis cannot discriminate, the path does, and the
    # edge-label spelling wins among equal-length candidates.
    assert spell(root, "REDUCE", stat) == "REDUCE@a.fold.k"
    assert resolve(root, "REDUCE@a.fold.k").node is stat
    # An explicit same-named axis key is ambiguous between the product and the stat — loud.
    with pytest.raises(ValueError, match="ambiguous"):
        resolve(root, "REDUCE@k")


# ---- sugar levels / idempotence ---------------------------------------------------------------- #


def test_every_sugar_level_resolves_and_canonicalizes() -> None:
    root, product, stat = _norm_linear_tree()
    for key, node in [
        ("REDUCE", product),
        ("REDUCE@a.fold.k", stat),
        ("REDUCE@a.map.fold.k", stat),  # a longer unique suffix is accepted at pin time
        ("REDUCE@a.fold", stat),  # axis elided — the path alone is unique
    ]:
        site = resolve(root, key)
        assert site is not None and site.node is node, key
        # Idempotence: the canonical spelling resolves to the same site, and re-canonicalizes to itself.
        canon = canonical(root, key)
        assert resolve(root, canon).node is node
        assert canonical(root, canon) == canon


def test_broken_stored_key_fails_loudly() -> None:
    root = _contraction_fold()
    with pytest.raises(ValueError, match="names no site"):
        resolve(root, "TILE@dd")  # a structural change broke the stored short key — never silent


# ---- ordinals ----------------------------------------------------------------------------------- #


def test_true_same_path_collision_emits_ordinals() -> None:
    f1 = _planar_fold("k", acc="s0", val="v1", load="x")
    f2 = _planar_fold("k", acc="t0", val="w1", load="z")
    root = Fold.projection(body=Body((Assign(name="o", op="add", args=(f1.out, f2.out)),)), operands=(f1, f2))
    assert spell(root, "REDUCE", f1) == "REDUCE@map.fold.k1"
    assert spell(root, "REDUCE", f2) == "REDUCE@map.fold.k2"
    assert resolve(root, "REDUCE@map.fold.k1").node is f1
    assert resolve(root, "REDUCE@map.fold.k2").node is f2
    with pytest.raises(ValueError, match="ambiguous"):
        resolve(root, "REDUCE@map.fold.k")


def test_deep_identical_paths_take_the_ordinal_without_subsequence_search() -> None:
    path = ("map", *("fold" for _ in range(1000)))
    first = Site(node=object(), axis="k", segments=path, ordinal=1)
    second = Site(node=object(), axis="k", segments=path, ordinal=2)

    assert _spellings("REDUCE", second, (first, second)) == f"REDUCE@{'.'.join(path)}.k2"


def test_literal_axis_name_wins_over_an_ordinal_reading() -> None:
    # An axis literally named ``k2`` must never lose to the ``k`` + ordinal-2 reading.
    root = _contraction_fold("k2")
    assert resolve(root, "REDUCE@k2").node is root


def test_digit_suffixed_axis_round_trips_with_an_ordinal() -> None:
    f1 = _planar_fold("a2", acc="s0", val="v1", load="x")
    f2 = _planar_fold("a2", acc="t0", val="w1", load="z")
    root = Fold.projection(body=Body((Assign(name="o", op="add", args=(f1.out, f2.out)),)), operands=(f1, f2))
    assert spell(root, "REDUCE", f1) == "REDUCE@map.fold.a21"
    assert spell(root, "REDUCE", f2) == "REDUCE@map.fold.a22"
    assert resolve(root, "REDUCE@map.fold.a21").node is f1
    assert resolve(root, "REDUCE@map.fold.a22").node is f2


def test_ordinal_spelling_escapes_a_literal_axis_capture() -> None:
    """The round-trip law under collision: two identical-path ``a13`` folds beside a literal
    ``a131`` axis — the concatenated ordinal form (``a13`` + 1 → ``a131``) would resolve to the
    literal-axis site (resolve reads a final component as a literal axis first, by design), so
    the ordinal arm mints the explicit dotted form, and every site round-trips to itself."""
    f1 = _planar_fold("a13", acc="s0", val="v1", load="x")
    f2 = _planar_fold("a13", acc="t0", val="w1", load="z")
    f3 = _planar_fold("a131", acc="u0", val="q1", load="y")
    root = Fold.projection(
        body=Body((Assign(name="o", op="add", args=(f1.out, f2.out)), Assign(name="p", op="add", args=("o", f3.out)))),
        operands=(f1, f2, f3),
    )
    for node in (f1, f2, f3):
        key = spell(root, "REDUCE", node)
        assert resolve(root, key).node is node, key
    assert spell(root, "REDUCE", f1) == "REDUCE@map.fold.a13.1"  # a13 + 1 would be captured by the a131 axis
    assert spell(root, "REDUCE", f2) == "REDUCE@map.fold.a132"  # a13 + 2 collides with nothing — concat stays canonical


# ---- reserved grammar ---------------------------------------------------------------------------- #


@pytest.mark.parametrize("key", ["TILE@in.a", "REDUCE@in.b.fold.k", "STAGE@=x17"])
def test_reserved_graph_placement_forms_are_rejected(key: str) -> None:
    with pytest.raises(ValueError, match="reserved for graph-level placement"):
        parse_key(key)


def test_unknown_segment_rejected() -> None:
    with pytest.raises(ValueError, match="unknown path segment"):
        parse_key("REDUCE@cone.fold.k")


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
    root, stream, qk, pv = _flash_tree()
    all_sites = sites(root)
    tiles = family_sites("TILE", all_sites)
    assert {id(s.node) for s in tiles} == {id(qk), id(pv)}
    assert primary("TILE", tiles) is None
    reduces = family_sites("REDUCE", all_sites)
    assert {id(s.node) for s in reduces} == {id(stream), id(qk), id(pv)}
    assert primary("REDUCE", reduces).node is stream


def test_site_is_a_frozen_value() -> None:
    s = Site(node=None, axis="k", segments=("fold",))
    assert s.depth == 1 and s.ordinal == 1
