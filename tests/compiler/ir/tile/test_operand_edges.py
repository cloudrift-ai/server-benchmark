"""Operand edges + the product-carrier :class:`ContractionView` — sharing is arity, not naming.

A computed operand is stored INLINE on its edge (there is no let table and no name-reference arm);
"these two matmuls read the same A" is ONE ``ContractionView`` with one ``a`` edge and N product
:class:`Channel`\\ s ``(b_i, acc_i)``. These pin the node's derived product loop (shared A lifted
once, N-component product-monoid carrier), the arity-vs-copies distinction, the inline-arm
canonicalization through ``rewrite``, and the ``captured_values`` closure predicate a placement cut
asks before lifting a subtree.
"""

from __future__ import annotations

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.schedule import TilePlan
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop
from emmy.compiler.ir.stmt.passes import rewrite
from emmy.compiler.ir.tile import Channel, ContractionView, Map, contraction_view
from emmy.compiler.ir.tile.ir import axis_names, captured_values, tree_nodes
from emmy.compiler.ir.tile.ops import lower


def _cone(name: str = "xhat") -> Map:
    """A minimal computed A-cone — ``xhat = x[m, k] * s[k]``, the shape the fused norm→linear edge's
    computed A takes (here without its statistic reduce, which the tree vocabulary does not need)."""
    load = Load(name=f"{name}_e", input="x", index=(Var("m"), Var("k")))
    scale = Load(name=f"{name}_s", input="w", index=(Var("k"),))
    return Map(body=Body((load, scale, Assign(name=name, op="multiply", args=(f"{name}_e", f"{name}_s")))))


def _node(a, *channels: tuple[str, str]) -> ContractionView:
    """A contraction over operand edge ``a`` with one ``(acc, weight-buffer)`` channel per arg."""
    return ContractionView(
        axes=(Axis("m", 128), Axis("n", 128)),
        k_axis=Axis("k", 256),
        a=a,
        channels=tuple(Channel(b=Load(name=f"{acc}_b", input=w, index=(Var("k"), Var("n"))), acc=acc) for acc, w in channels),
        tile=TilePlan(),
    )


def _product() -> ContractionView:
    """The gate⊗up shape: ONE product contraction, two channels over one inline cone."""
    return _node(_cone(), ("acc_g", "Wg"), ("acc_u", "Wu"))


# --- the product node: sharing is arity --------------------------------------------------------- #


def test_product_node_derives_one_fold_loop_with_the_shared_a_lifted_once() -> None:
    """The fused group lowers to ONE derived loop — the shared A evaluated once, each further
    channel splicing its ``b → ⊗ → ⊕`` triple after it — never one loop per channel."""
    stmts = lower(_product())
    assert len(stmts) == 1 and isinstance(stmts[0], Loop)
    body = list(stmts[0].body)
    assert sum(1 for s in body if isinstance(s, Assign) and s.name == "xhat") == 1
    assert [s.name for s in body if isinstance(s, Accum)] == ["acc_g", "acc_u"]


def test_product_carrier_is_the_n_component_product_monoid() -> None:
    loop = _product().loop
    assert loop.carrier.state.names == ("acc_g", "acc_u")
    assert loop.carrier.twist.family == "id"
    assert len(loop.carrier.twist.channels) == 2


def test_arity_is_not_two_copies() -> None:
    """One node with two channels ≢ two independent contractions each computing their own A: the
    former lifts the shared A once (one loop), the latter lower to two loops with a cone each."""
    fused = lower(Map(body=Body(()), sources=(_product(),)))
    copies = Map(body=Body(()), sources=(_node(_cone(), ("acc_g", "Wg")), _node(_cone("xhat2"), ("acc_u", "Wu"))))
    assert len([s for s in fused if isinstance(s, Loop)]) == 1
    assert len([s for s in lower(copies) if isinstance(s, Loop)]) == 2


def test_defines_and_out_read_the_channels() -> None:
    node = _product()
    assert node.defines() == ("acc_g", "acc_u")
    assert node.out == "acc_g" and node.acc == "acc_g"  # the primary channel
    assert node.b is node.channels[0].b  # the primary-channel read single-channel tiers use


def test_single_channel_node_is_the_plain_matmul() -> None:
    node = _node(Load(name="a_e", input="x", index=(Var("m"), Var("k"))), ("acc", "W"))
    (loop,) = lower(node)
    assert loop.carrier.state.names == ("acc",)
    assert not node.a_computed


# --- inline computed operands ------------------------------------------------------------------- #


def test_a_computed_operand_is_stored_inline_and_flattens_on_the_edge() -> None:
    node = _product()
    assert node.a_computed
    assert node.a_body == tuple(lower(node.a))
    assert node.a_name == "xhat"


def test_tree_nodes_walks_inline_operand_edges() -> None:
    fold = _product().as_fold()
    assert _product().a in list(tree_nodes(fold))  # the cone edge, walked off the STORED fold


def test_external_reads_cover_every_channel() -> None:
    assert set(_product().external_reads()) == {"x", "w", "Wg", "Wu"}


def test_pretty_prints_the_channels_once() -> None:
    (line,) = _product().pretty()
    assert "xhat @ Wg,Wu" in line and "-> acc_g,acc_u" in line


# --- closure: the predicate a placement cut asks ------------------------------------------------- #


def _capturing_cone(name: str = "xhat") -> Map:
    """A cone that READS a value the enclosing body defines (``m_run``) instead of producing it —
    the flash ``P = exp(s − m)`` shape, where the running max comes from the carrier merge."""
    load = Load(name=f"{name}_e", input="x", index=(Var("m"), Var("k")))
    return Map(body=Body((load, Assign(name=name, op="subtract", args=(f"{name}_e", "m_run")))))


def test_a_capturing_inline_operand_is_legal_but_reports_its_capture() -> None:
    """Flash's ``P`` is exactly this: an inline operand reading the running max its own loop step
    updates. Legal to build and lower (its one home is in scope) — just not cuttable, which
    ``captured_values`` is the predicate for."""
    node = _node(_capturing_cone(), ("acc_g", "Wg"))
    fold = node.as_fold()
    assert lower(fold)  # lowers fine — position in the enclosing body is what makes it legal
    cone = node.a
    axes = axis_names(fold) | {a.name for a in node.axes}  # the output axes are placement facts
    assert captured_values(cone, axes | axis_names(cone)) == ("m_run",)


def test_iteration_variables_are_not_captures() -> None:
    """The dominant free names in any cone are loop induction variables (``m`` / ``k``), bound by
    the enclosing nest — excluding them is what makes the predicate mean anything."""
    cone = _cone()
    assert captured_values(cone, set()) == ("k", "m")  # unfiltered: the axes show up
    assert captured_values(cone, {"m", "k"}) == ()  # filtered: closed


# --- canonicalization: inline arms rewrite like any other subtree -------------------------------- #


def test_rewrite_renames_channel_accs_and_inline_arms_in_lockstep() -> None:
    """The canonicalizer runs over STORED trees — the fold; the view of the renamed fold shows the
    accs and the inline cone renamed through one map."""
    fold = _product().as_fold()
    renamed = rewrite(fold, lambda n: {"xhat": "v0", "acc_g": "v1", "acc_g__v": "v1__v"}.get(n, n), Sigma({}), lambda a: a)
    view = contraction_view(renamed, Axis("m", 128), Axis("n", 128))
    assert view is not None
    assert view.channels[0].acc == "v1" and view.channels[1].acc == "acc_u"
    assert view.a.out == "v0"  # the inline cone canonicalized through the same map


def test_rewrite_reaches_a_channels_b_edge() -> None:
    fold = _product().as_fold()
    renamed = rewrite(fold, lambda n: {"acc_u_b": "vb"}.get(n, n), Sigma({}), lambda a: a)
    view = contraction_view(renamed, Axis("m", 128), Axis("n", 128))
    assert view is not None and view.channels[1].b.names == ("vb",)


# --- Map.fn: the binder (1n — the ``source`` compat read retired with the ``out`` convention) ---- #


def test_map_binder_binds_sources_positionally() -> None:
    single = Map(sources=(_product(),))
    # One param per source RESULT COMPONENT (1q params flattening) — the product source binds
    # BOTH channel accumulators, so a combine reading the second never sees a free name.
    assert single.fn.params == _product().defines()
    assert single.fn.results == single.fn.params[:1]  # empty-body wrap: result = the carried state
    assert single.out == _product().out
    assert Map().fn.params == () and Map().fn.results == ()
