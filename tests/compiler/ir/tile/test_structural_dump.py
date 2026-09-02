"""The structural dump (``ops.pretty`` / ``TileOp.pretty_body``) — the STORED tree, as a tree, and
NOTHING derived.

The dump is the one place a reader meets the tile term directly, so what it shows has to be what
the term IS: each node's kind and stored params, an operand edge recursed into (a computed edge is
visibly a subtree, a materialized one visibly a leaf ``Load``), and the caller facts that live
BESIDE the term — placement, workers, schedule, output specifications — in their own regions. A derived
evaluation (the per-cell step, the nodes synthesized inside it, the lowered nest) is a CONSEQUENCE
of the stored params and is never printed: showing it beside storage is the inversion the layer
exists to prevent, and it was the bulk of the output.

These pin: (a) every stored param of each node kind reaches the dump; (b) edges say which lambda
params they bind, nest, are labelled by inhabitant, and appear exactly once; (c) nothing derived
appears; (d) accepted schedule choices annotate a node only when the owning ``TileOp`` supplies
them — never from the term; (e) a λ that is not closed says what it captures.
"""

from __future__ import annotations

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.pure import Lambda
from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.schedule import Placement, Raster, Reduce, Schedule, Stage, Tile, Work
from emmy.compiler.ir.schedule.classic import (
    ClassicMaterialization,
    ClassicScheduleContext,
    EdgeSchedule,
    KernelSchedule,
    Projection,
    ProjectionSchedule,
    ReductionSchedule,
)
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
from emmy.compiler.ir.tile import OutputSpec, TileOp
from emmy.compiler.ir.tile._dump import pretty
from emmy.compiler.pipeline.passes.lowering.tile._fromloop import fold_from_loop


def _stat_fold() -> Fold:
    """RMSNorm's statistic — ``acc0 += x[m, k]²``, loads inline in the lift."""
    body = Body(
        (
            Load(name="in0", input="x", index=(Var("m"), Var("k"))),
            Assign(name="v1", op="multiply", args=("in0", "in0")),
            Accum(name="acc0", value="v1", op="add", axes=("k",)),
        )
    )
    return fold_from_loop(Loop(axis=Axis("k", 512), body=body))


def _cone() -> Fold:
    """A computed A edge — ``xhat = x[m, k] * w[k]``."""
    return Fold.projection(
        body=Body(
            (
                Load(name="xhat_e", input="x", index=(Var("m"), Var("k"))),
                Load(name="xhat_s", input="w", index=(Var("k"),)),
                Assign(name="xhat", op="multiply", args=("xhat_e", "xhat_s")),
            )
        )
    )


def _product(a=None) -> Fold:
    """The gate⊗up shape — two channels over ONE shared ``a`` edge."""
    return Fold.contraction(
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
    # Every param, the residual included: the body reads the enclosing row coordinate, so the
    # lift binds it after the iteration var and the signature says so.
    assert "├─ lift: λ(k, m) -> (v1)" in text
    assert "└─ combine: λ(acc0, acc0__o) -> (acc0)" in text
    # The lift's own body nests two under its signature — the stored program, read as the
    # binder's body rather than as a sibling of the branch labels.
    assert "│    in0 = load x[m, k]" in text


def test_contraction_dump_shows_the_k_axis_and_every_channel() -> None:
    text = "\n".join(pretty(_product()))
    assert "Fold[k in 0..256] contraction" in text
    assert "├─ operand[a_e]: load x[m, k]" in text
    # Sharing is arity: one ``a``, one branch per channel, each naming its own accumulator.
    assert "├─ operand[acc_g_b] -> acc_g: load Wg[k, n]" in text
    assert "├─ operand[acc_u_b] -> acc_u: load Wu[k, n]" in text
    # The binding labels connect role-ordered edges above to their actual positional lift params.
    assert "lift: λ(k, acc_g_b, a_e, acc_u_b) -> (acc_g__v, acc_u__v)" in text


def test_projection_dump_shows_the_binder_and_its_operands() -> None:
    """A zero-axis Fold stores its projection lambda and operands. The binder rides its own branch,
    next to the body it binds rather than in the header. Every lambda-valued field reads the same
    way: signature, then body."""
    m = Fold.projection(operands=(_stat_fold(),), body=Body((Assign(name="o", op="rsqrt", args=("acc0",)),)))
    text = "\n".join(pretty(m))
    assert text.splitlines()[0] == "Fold  free"
    assert "├─ operand[acc0]: Fold[k in 0..512] planar" in text
    assert "└─ lift: λ(acc0) -> (o)" in text
    assert "     o = rsqrt(acc0)" in text  # the body, indented two under the signature


def test_a_product_edge_shows_every_lambda_param_it_binds() -> None:
    """One operand edge can supply several result components, so the dump names every scalar
    substituted for that edge instead of leaving the positional binding implicit."""
    node = Fold.projection(operands=(_product(),), body=Body((Assign(name="o", op="add", args=("acc_g", "acc_u")),)))
    text = "\n".join(pretty(node))
    assert "operand[acc_g, acc_u]: Fold[k in 0..256] contraction" in text
    assert "lift: λ(acc_g, acc_u) -> (o)" in text


def test_the_fn_branch_survives_an_empty_body() -> None:
    """The branch carries the SIGNATURE, so it is emitted even with nothing to compute — an
    identity projection still binds, and dropping the branch would lose the binder entirely."""
    text = "\n".join(pretty(Fold.projection(operands=(_stat_fold(),), body=Body(()))))
    assert "└─ lift: λ(acc0) -> (acc0)" in text


def test_a_sourceless_map_is_marked_pointwise() -> None:
    assert "‹pointwise›" in "\n".join(pretty(_cone()))


# --- edges nest, and say which of the two inhabitants they are ---------------------------------- #


def test_a_computed_edge_nests_as_a_subtree_a_materialized_one_is_a_leaf() -> None:
    """The two inhabitants of an operand edge, told apart in the dump: the cone recurses into its
    own node, the gmem loads do not."""
    lines = pretty(_product(a=_cone()))
    (a_line,) = [ln for ln in lines if "operand[xhat]:" in ln]
    assert "‹computed›" in a_line and "Fold  free" in a_line
    assert any("‹materialized›" in ln and "load Wg" in ln for ln in lines)
    # The cone's own body is reached BELOW the a edge — the subtree is really rendered.
    assert any("xhat = multiply(xhat_e, xhat_s)" in ln for ln in lines)


# --- nothing DERIVED reaches the dump ------------------------------------------------------------ #


def _split_k() -> Fold:
    """Split-K's outer reduce — the identity-lift composition over one bilinear ``Fold`` edge. Its
    derived step embeds that very object, so a dump that printed the step would show it twice."""
    inner = Fold.contraction(
        k_axis=Axis("kslice", 128),
        a=Load(name="a_e", input="x", index=(Var("m"), Var("kslice"))),
        channels=(Channel(b=Load(name="b_e", input="w", index=(Var("kslice"), Var("n"))), acc="acc0"),),
    )
    return Fold(
        axis=Axis("ksplit", 4),
        operands=(inner,),
        lift=Lambda(params=("ksplit", "acc0"), body=Body(()), results=("acc0",)),
        init=(0.0,),
        combine=Lambda(
            params=("acc0", "acc0__o"),
            body=Body((Assign(name="acc0", op="add", args=("acc0", "acc0__o")),)),
            results=("acc0",),
        ),
    )


def test_the_derived_step_is_not_printed() -> None:
    """The step is a CONSEQUENCE of ``lift`` + ``combine``, as re-derivable as ``Fold.lower()``'s
    output — printing it beside storage is the inversion the dump exists to prevent, and it was
    the bulk of the output. ``--ir loop`` is where a body lives."""
    fold = _stat_fold()
    text = "\n".join(pretty(fold))
    assert "derived" not in text
    # The step's serial form — the combine specialized at the singleton, an ``Accum`` — is exactly
    # what must not appear: it is impure, so it is not even spellable as one of the stored λs.
    assert "acc0 <- add" not in text
    assert [type(s).__name__ for s in fold.step_stmts()][-1] == "Accum"  # the premise


def test_an_edge_is_rendered_once_not_once_per_derived_position() -> None:
    """The step embeds its operand edges at their derived positions, so printing it duplicated
    every edge. With storage only, each edge appears exactly once — under its own branch."""
    fold = _split_k()
    assert any(s is fold.operands[0] for s in fold.step_stmts())  # the premise: one object, two positions
    text = "\n".join(pretty(fold))
    assert text.count("Fold[kslice in 0..128] contraction") == 1
    assert text.count("operand[a_e]: load x[m, kslice]") == 1


# --- a λ that is not closed says so -------------------------------------------------------------- #


def _capturing_cone() -> Fold:
    """The flash ``P = exp(s − m)`` shape — a cone that READS a value the enclosing body defines
    (``m_run``, the carrier's running max) instead of producing it."""
    return Fold.projection(
        body=Body(
            (
                Load(name="p_e", input="x", index=(Var("m"), Var("k"))),
                Assign(name="p", op="subtract", args=("p_e", "m_run")),
            )
        )
    )


def test_a_lambda_binds_an_enclosing_value_as_a_trailing_param() -> None:
    """Being closed is exactly what decides whether a subtree can become an operand edge, so a λ
    that reads an enclosing value BINDS it — the dump shows it in the signature rather than in a
    separate capture annexe. ``m`` / ``k`` are iteration space (placement + the contraction's own
    axis) and ``m_run`` is a value, and binding everything is what removes that distinction
    instead of deciding it: all three arrive as params, in the order ``Lambda.closing`` appends."""
    node = _product(a=_capturing_cone())
    tile = TileOp(op=node, name="k_flashish", place=Placement(free=(Axis("m", 128),), grid=(Axis("m", 128),), mapped=True))
    text = tile.pretty_body()
    assert "│  └─ lift: λ(k, m, m_run) -> (p)" in text
    assert "captures" not in text


def test_iteration_vars_are_not_captures() -> None:
    """A λ reading an axis is not capturing — the nest binds it. This covers the three places an
    axis can come from: the term's own axes, the placement, and an output specification's sweep."""
    m, n = Axis("m", 128), Axis("n", 64)
    body = Body((Load(name="w_e", input="w", index=(Var("m"), Var("n"))), Assign(name="o", op="multiply", args=("acc0", "w_e"))))
    tile = TileOp(
        op=Fold.projection(operands=(_stat_fold(),), body=body),
        name="k_stat",
        place=Placement(free=(m, n), grid=(m,), mapped=True),
        output_specs=(OutputSpec(write=Write(output="y", index=(Var("m"), Var("n")), value="o"), sweep=n),),
    )
    # ``m`` comes from the placement, ``n`` from the output sweep — the sweep axis left the term
    # at 1q, so a dump reading only the term would wrongly call it captured.
    assert "captures" not in tile.pretty_body()


def test_the_capture_set_is_omitted_when_the_iteration_space_is_unknown() -> None:
    """A bare term has no placement, so its grid coordinates are indistinguishable from captured
    values. The dump declines to answer rather than claim the λ is closed."""
    assert "captures" not in "\n".join(pretty(_capturing_cone()))


# --- accepted choices annotate from the TileOp, never from the term ----------------------------- #


def test_slices_annotate_a_node_only_when_the_owning_tileop_supplies_them() -> None:
    fold = _stat_fold()
    bare = TileOp(op=fold, name="k_stat")
    assert "REDUCE=" not in bare.pretty_body()

    context = ClassicScheduleContext(bare)
    nodes = {
        site: ProjectionSchedule(Tile()) if isinstance(view, Projection) else ReductionSchedule(Tile(), Reduce.of(reg=4))
        for site, view in enumerate(context.tile_op.views)
    }
    classic = Schedule(
        KernelSchedule(Work(), Raster()),
        nodes,
        {edge: EdgeSchedule(Stage.direct()) for edge in context.tile_op.edge_sites},
    )
    scheduled = TileOp(
        op=fold,
        name="k_stat",
        schedule=classic,
        materialization=ClassicMaterialization({}, {}),
    )
    assert "⟨REDUCE=r4⟩" in scheduled.pretty_body()
    # The annotation is the TileOp's; the term is untouched by it.
    assert "REDUCE=" not in "\n".join(pretty(fold))


# --- the caller facts beside the term get their own regions ------------------------------------- #


def test_pretty_body_separates_placement_and_outputs_from_the_term() -> None:
    m, n = Axis("m", 128), Axis("n", 64)
    tile = TileOp(
        op=Fold.projection(operands=(_stat_fold(),), body=Body((Assign(name="o", op="rsqrt", args=("acc0",)),))),
        name="k_stat",
        place=Placement(free=(m, n), grid=(m,), mapped=True),
        output_specs=(OutputSpec(write=Write(output="y", index=(Var("m"), Var("n")), value="o"), sweep=n),),
    )
    text = tile.pretty_body()
    assert "place  free=(m, n)  grid=(m)" in text
    assert "outputs" in text and "└─ sweep(n) y[m, n] = o" in text


def test_an_unmapped_placement_says_so() -> None:
    tile = TileOp(op=_stat_fold(), name="k_stat", place=Placement(free=(Axis("m", 128),)))
    assert "unmapped" in tile.pretty_body()
