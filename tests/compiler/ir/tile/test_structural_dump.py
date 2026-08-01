"""The structural dump (``ops.pretty`` / ``TileOp.pretty_body``) — the STORED tree, as a tree.

The dump is the one place a reader meets the tile term directly, so what it shows has to be what
the term IS: each node's kind and stored params, an operand edge recursed into (a computed edge is
visibly a subtree, a materialized one visibly a leaf ``Load``), and the caller facts that live
BESIDE the term — placement, workers, boundary stores — in their own regions. The derived loop
reading is labelled ``derived`` and nothing else may masquerade as storage.

These pin: (a) every stored param of each node kind reaches the dump; (b) edges nest and are
labelled by inhabitant; (c) the derived reading is separated from the stored one; (d) schedule
slices annotate a node only when the owning ``TileOp`` supplies them — never from the term.
"""

from __future__ import annotations

from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.schedule import Placement, ReducePlan
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
from emmy.compiler.ir.tile import Channel, Contraction, Fold, Map, Store, TileOp
from emmy.compiler.ir.tile.ops import Sched, pretty


def _stat_fold() -> Fold:
    """RMSNorm's statistic — ``acc0 += x[m, k]²``, loads inline in the lift."""
    body = Body(
        (
            Load(name="in0", input="x", index=(Var("m"), Var("k"))),
            Assign(name="v1", op="multiply", args=("in0", "in0")),
            Accum(name="acc0", value="v1", op="add", axes=("k",)),
        )
    )
    return Fold.from_loop(Loop(axis=Axis("k", 512), body=body, role=AxisRole.PLANAR))


def _cone() -> Map:
    """A computed A edge — ``xhat = x[m, k] * w[k]``."""
    return Map(
        body=Body(
            (
                Load(name="xhat_e", input="x", index=(Var("m"), Var("k"))),
                Load(name="xhat_s", input="w", index=(Var("k"),)),
                Assign(name="xhat", op="multiply", args=("xhat_e", "xhat_s")),
            )
        )
    )


def _product(a=None) -> Contraction:
    """The gate⊗up shape — two channels over ONE shared ``a`` edge."""
    return Contraction(
        k_axis=Axis("k", 256),
        a=a if a is not None else Load(name="a_e", input="x", index=(Var("m"), Var("k"))),
        channels=tuple(
            Channel(b=Load(name=f"{acc}_b", input=w, index=(Var("k"), Var("n"))), acc=acc) for acc, w in (("acc_g", "Wg"), ("acc_u", "Wu"))
        ),
    )


# --- the stored params all reach the dump ------------------------------------------------------- #


def test_fold_dump_shows_every_stored_param() -> None:
    """A fold's storage IS ``axis`` + ``lift`` + ``(init, combine)`` + ``operands``; each is a
    labelled branch, and the role — DERIVED, never stored — rides the header."""
    text = "\n".join(pretty(_stat_fold()))
    assert text.splitlines()[0] == "Fold[k in 0..512] planar"
    assert "├─ init: (0)" in text
    assert "├─ lift: λ(k) -> (v1)" in text
    assert "├─ combine: λ(acc0, acc0__o) -> (acc0)" in text
    # The lift's own body nests under it — the stored program, not a synthesized one.
    assert "│  in0 = load x[m, k]" in text


def test_contraction_dump_shows_the_k_axis_and_every_channel() -> None:
    text = "\n".join(pretty(_product()))
    assert "Contraction [Σ k in 0..256]" in text
    assert "├─ a: a_e = load x[m, k]" in text
    # Sharing is arity: one ``a``, one branch per channel, each naming its own accumulator.
    assert "├─ channel[0] -> acc_g: acc_g_b = load Wg[k, n]" in text
    assert "└─ channel[1] -> acc_u: acc_u_b = load Wu[k, n]" in text


def test_map_dump_shows_the_binder_and_its_sources() -> None:
    m = Map(fn=None, sources=(_stat_fold(),), body=Body((Assign(name="o", op="rsqrt", args=("acc0",)),)))
    text = "\n".join(pretty(m))
    assert text.splitlines()[0].startswith("Map λ(acc0) -> (o)")
    assert "├─ source[0]: Fold[k in 0..512] planar" in text
    assert "└─ body" in text


def test_a_sourceless_map_is_marked_pointwise() -> None:
    assert "‹pointwise›" in "\n".join(pretty(_cone()))


# --- edges nest, and say which of the two inhabitants they are ---------------------------------- #


def test_a_computed_edge_nests_as_a_subtree_a_materialized_one_is_a_leaf() -> None:
    """The two inhabitants of an operand edge, told apart in the dump: the cone recurses into its
    own node, the gmem loads do not."""
    lines = pretty(_product(a=_cone()))
    (a_line,) = [ln for ln in lines if "├─ a:" in ln]
    assert "‹computed›" in a_line and "Map" in a_line
    assert any("‹materialized›" in ln and "load Wg" in ln for ln in lines)
    # The cone's own body is reached BELOW the a edge — the subtree is really rendered.
    assert any("xhat = multiply(xhat_e, xhat_s)" in ln for ln in lines)


# --- the derived reading is labelled, and suppressible ------------------------------------------ #


def test_the_derived_step_is_labelled_and_opt_out() -> None:
    fold = _stat_fold()
    assert "└─ derived step" in "\n".join(pretty(fold))
    assert "derived" not in "\n".join(pretty(fold, derived=False))


# --- schedule slices annotate from the TileOp, never from the term ------------------------------ #


def test_slices_annotate_a_node_only_when_the_owning_tileop_supplies_them() -> None:
    fold = _stat_fold()
    bare = TileOp(op=fold, name="k_stat")
    assert "REDUCE=" not in bare.pretty_body()

    scheduled = TileOp(op=fold, name="k_stat")
    Sched(scheduled.op, scheduled.schedule).put("REDUCE", fold, ReducePlan.of(reg=4))
    assert "⟨REDUCE=r4⟩" in scheduled.pretty_body()
    # The annotation is the TileOp's; the term is untouched by it.
    assert "REDUCE=" not in "\n".join(pretty(fold))


# --- the caller facts beside the term get their own regions ------------------------------------- #


def test_pretty_body_separates_placement_and_boundary_stores_from_the_term() -> None:
    m, n = Axis("m", 128), Axis("n", 64)
    tile = TileOp(
        op=Map(fn=None, sources=(_stat_fold(),), body=Body((Assign(name="o", op="rsqrt", args=("acc0",)),))),
        name="k_stat",
        place=Placement(free=(m, n), grid=(m,), mapped=True),
        stores=(Store(write=Write(output="y", index=(Var("m"), Var("n")), value="o"), sweep=n),),
    )
    text = tile.pretty_body()
    assert "place  free=(m, n)  grid=(m)" in text
    assert "stores" in text and "└─ sweep(n) y[m, n] = o" in text


def test_an_unmapped_placement_says_so() -> None:
    tile = TileOp(op=_stat_fold(), name="k_stat", place=Placement(free=(Axis("m", 128),)))
    assert "unmapped" in tile.pretty_body()
