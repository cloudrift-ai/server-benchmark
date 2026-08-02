"""The structural dump (``ops.pretty`` / ``TileOp.pretty_body``) — the STORED tree, as a tree, and
NOTHING derived.

The dump is the one place a reader meets the tile term directly, so what it shows has to be what
the term IS: each node's kind and stored params, an operand edge recursed into (a computed edge is
visibly a subtree, a materialized one visibly a leaf ``Load``), and the caller facts that live
BESIDE the term — placement, workers, schedule, boundary stores — in their own regions. A derived
evaluation (the per-cell step, the nodes synthesized inside it, the lowered nest) is a CONSEQUENCE
of the stored params and is never printed: showing it beside storage is the inversion the layer
exists to prevent, and it was the bulk of the output.

These pin: (a) every stored param of each node kind reaches the dump; (b) edges nest, are labelled
by inhabitant, and appear exactly once; (c) nothing derived appears — including the one slice whose
site is synthesized, which lands in the schedule region rather than reconstructing its node;
(d) schedule slices annotate a node only when the owning ``TileOp`` supplies them — never from the
term; (e) a λ that is not closed says what it captures.
"""

from __future__ import annotations

from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.schedule import Placement, ReducePlan, TilePlan
from emmy.compiler.ir.stmt import Accum, Assign, Body, Lambda, Load, Loop, Write
from emmy.compiler.ir.tile import Channel, Fold, Store, TileOp
from emmy.compiler.ir.tile.ops import Sched, pretty, unplaced_slices


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
    assert "├─ lift: λ(k) -> (v1)" in text
    assert "└─ combine: λ(acc0, acc0__o) -> (acc0)" in text
    # The lift's own body nests two under its signature — the stored program, read as the
    # binder's body rather than as a sibling of the branch labels.
    assert "│    in0 = load x[m, k]" in text


def test_contraction_dump_shows_the_k_axis_and_every_channel() -> None:
    text = "\n".join(pretty(_product()))
    assert "Fold[k in 0..256] contraction" in text
    assert "├─ operand[a]: a_e = load x[m, k]" in text
    # Sharing is arity: one ``a``, one branch per channel, each naming its own accumulator.
    assert "├─ operand[b0] -> acc_g: acc_g_b = load Wg[k, n]" in text
    assert "├─ operand[b1] -> acc_u: acc_u_b = load Wu[k, n]" in text


def test_map_dump_shows_the_binder_and_its_sources() -> None:
    """A ``Map``'s storage is ``fn`` + ``sources``. The binder rides its OWN branch, next to the
    body it binds — not the header, which on a big fold sat a screenful above its stmts. Every
    λ-valued field reads the same way: ``lift:`` / ``combine:`` / ``fn:``, signature then body."""
    m = Fold.projection(fn=None, operands=(_stat_fold(),), body=Body((Assign(name="o", op="rsqrt", args=("acc0",)),)))
    text = "\n".join(pretty(m))
    assert text.splitlines()[0] == "Fold  free"
    assert "├─ operand[0]: Fold[k in 0..512] planar" in text
    assert "└─ lift: λ(acc0) -> (o)" in text
    assert "     o = rsqrt(acc0)" in text  # the body, indented two under the signature


def test_the_fn_branch_survives_an_empty_body() -> None:
    """The branch carries the SIGNATURE, so it is emitted even with nothing to compute — an
    identity projection still binds, and dropping the branch would lose the binder entirely."""
    text = "\n".join(pretty(Fold.projection(fn=None, operands=(_stat_fold(),), body=Body(()))))
    assert "└─ lift: λ(acc0) -> (acc0)" in text


def test_a_sourceless_map_is_marked_pointwise() -> None:
    assert "‹pointwise›" in "\n".join(pretty(_cone()))


# --- edges nest, and say which of the two inhabitants they are ---------------------------------- #


def test_a_computed_edge_nests_as_a_subtree_a_materialized_one_is_a_leaf() -> None:
    """The two inhabitants of an operand edge, told apart in the dump: the cone recurses into its
    own node, the gmem loads do not."""
    lines = pretty(_product(a=_cone()))
    (a_line,) = [ln for ln in lines if "operand[a]:" in ln]
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
    """The step is a CONSEQUENCE of ``lift`` + ``combine``, as re-derivable as ``lower()``'s
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
    assert text.count("operand[a]: a_e = load x[m, kslice]") == 1


def test_a_slice_keyed_against_derived_material_prints_in_the_schedule_region() -> None:
    """The one thing the derived branch carried that storage cannot: a slice whose site is a
    synthesized node. It is a SCHEDULE fact, so it lands in the schedule region — the term is not
    reconstructed to hang it on."""
    fold = _stat_fold()
    tile = TileOp(op=fold, name="k_stat")
    tile.schedule["TILE@pj"] = TilePlan(regs=(64, 1))  # a key no stored node claims
    assert unplaced_slices(tile) == [("TILE@pj", tile.schedule["TILE@pj"])]
    text = tile.pretty_body()
    assert "    schedule" in text and "└─ TILE@pj = f64" in text
    # A kernel whose every key lands on a stored node has no such region at all.
    plain = TileOp(op=fold, name="k_stat")
    Sched(plain.op, plain.schedule).put("REDUCE", fold, ReducePlan.of(reg=4))
    assert unplaced_slices(plain) == [] and "schedule" not in plain.pretty_body()


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


def test_a_lambda_that_captures_an_enclosing_value_shows_its_capture_set() -> None:
    """Without this the λ prints as though it were closed — and being closed is exactly what
    decides whether a subtree can become an operand edge. ``m`` / ``k`` here are iteration space
    (placement + the contraction's own axis); only ``m_run`` is a captured VALUE."""
    node = _product(a=_capturing_cone())
    tile = TileOp(op=node, name="k_flashish", place=Placement(free=(Axis("m", 128),), grid=(Axis("m", 128),), mapped=True))
    text = tile.pretty_body()
    assert "[captures m_run]" in text
    assert "captures k" not in text and "captures m," not in text


def test_iteration_vars_are_not_captures() -> None:
    """A λ reading an axis is not capturing — the nest binds it. This covers the three places an
    axis can come from: the term's own axes, the placement, and a boundary store's sweep."""
    m, n = Axis("m", 128), Axis("n", 64)
    body = Body((Load(name="w_e", input="w", index=(Var("m"), Var("n"))), Assign(name="o", op="multiply", args=("acc0", "w_e"))))
    tile = TileOp(
        op=Fold.projection(fn=None, operands=(_stat_fold(),), body=body),
        name="k_stat",
        place=Placement(free=(m, n), grid=(m,), mapped=True),
        stores=(Store(write=Write(output="y", index=(Var("m"), Var("n")), value="o"), sweep=n),),
    )
    # ``m`` comes from the placement, ``n`` from the output sweep — the sweep axis left the term
    # at 1q, so a dump reading only the term would wrongly call it captured.
    assert "captures" not in tile.pretty_body()


def test_the_capture_set_is_omitted_when_the_iteration_space_is_unknown() -> None:
    """A bare term has no placement, so its grid coordinates are indistinguishable from captured
    values. The dump declines to answer rather than claim the λ is closed."""
    assert "captures" not in "\n".join(pretty(_capturing_cone()))


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
        op=Fold.projection(fn=None, operands=(_stat_fold(),), body=Body((Assign(name="o", op="rsqrt", args=("acc0",)),))),
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
