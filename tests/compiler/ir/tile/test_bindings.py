"""``TileOp.bindings`` — the let table, its invariants, and the name-operand round trip.

A shared subtree lives ONCE, in ``TileOp.bindings``, keyed by the name its root defines; a consumer
references it by that plain SSA name in an operand field (``Contraction.a = "xhat"``). There
is no ``Ref`` node kind — names are the IR's one reference mechanism, so the rewrite rename map and
``structural_key`` canonicalization carry references and definitions in lockstep. These pin the
construction-time invariants (no dangling reference, a key IS its tree's ``out``, a binding name is
defined nowhere else) and the ``ops.resolve`` inlining every lowering walk goes through.
"""

from __future__ import annotations

import re

import pytest

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.schedule import TilePlan
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Assign, Body, Load, Write
from emmy.compiler.ir.stmt.passes import rewrite
from emmy.compiler.ir.tile import Contraction, Map, TileOp
from emmy.compiler.ir.tile.ir import axis_names, captured_values
from emmy.compiler.ir.tile.ops import group_loop, is_group, lower, resolve


def _cone(name: str = "xhat") -> Map:
    """A minimal bound A-cone — ``xhat = x[m, k] * s[k]``, the shape the fused norm→linear edge's
    computed A takes (here without its statistic reduce, which the tree vocabulary does not need).
    The scale is LOADED, not read free: a shared binding must be closed (``captured_values``)."""
    load = Load(name=f"{name}_e", input="x", index=(Var("m"), Var("k")))
    scale = Load(name=f"{name}_s", input="w", index=(Var("k"),))
    return Map(body=Body((load, scale, Assign(name=name, op="multiply", args=(f"{name}_e", f"{name}_s")))))


def _channel(acc: str, weight: str, a: str | Map = "xhat") -> Contraction:
    """One ⊗-fold channel over the shared A — ``acc = Σ_k A[m, k]·W[k, n]``."""
    return Contraction(
        axes=(Axis("m", 128), Axis("n", 128)),
        k_axis=Axis("k", 256),
        a=a,
        b=Load(name=f"{acc}_b", input=weight, index=(Var("k"), Var("n"))),
        acc=acc,
        tile=TilePlan(),
    )


def _shared_group() -> TileOp:
    """The gate⊗up shape: two sibling contractions referencing ONE bound cone."""
    cone = _cone()
    body = Body((Assign(name="y", op="multiply", args=("acc_g", "acc_u")), Write(output="out", index=(Var("m"), Var("n")), value="y")))
    return TileOp(op=Map(body=body, sources=(_channel("acc_g", "Wg"), _channel("acc_u", "Wu"))), bindings={"xhat": cone})


# --- construction-time invariants -------------------------------------------------------------- #


def test_a_dangling_reference_is_rejected_at_construction() -> None:
    with pytest.raises(AssertionError, match=re.escape("resolve to no binding")):
        TileOp(op=Map(sources=(_channel("acc", "W"),)))


def test_a_binding_key_must_be_its_trees_output_name() -> None:
    with pytest.raises(AssertionError, match="is not its tree's output name"):
        TileOp(op=Map(sources=(_channel("acc", "W", a="wrong"),)), bindings={"wrong": _cone()})


def test_a_binding_name_defined_elsewhere_is_rejected() -> None:
    """The one uniqueness rule the let table needs: a reference must be unambiguous, so a binding's
    key may not also be defined in ``op`` (or in another binding)."""
    shadow = Map(body=Body((Assign(name="xhat", op="copy", args=("z",)),)), sources=(_channel("acc", "W"),))
    with pytest.raises(AssertionError, match="is also defined in"):
        TileOp(op=shadow, bindings={"xhat": _cone()})


def test_a_wellformed_shared_group_validates() -> None:
    tile = _shared_group()
    assert set(tile.bindings) == {"xhat"}
    assert [c.a_ref for c in tile.op.sources] == ["xhat", "xhat"]


# --- closure: a subtree read from many places must be computable at every one of them ------------ #


def _capturing_cone(name: str = "xhat") -> Map:
    """A cone that READS a value the enclosing body defines (``m_run``) instead of producing it —
    the flash ``P = exp(s − m)`` shape, where the running max comes from the carrier merge."""
    load = Load(name=f"{name}_e", input="x", index=(Var("m"), Var("k")))
    return Map(body=Body((load, Assign(name=name, op="subtract", args=(f"{name}_e", "m_run")))))


def test_a_shared_binding_that_captures_an_enclosing_value_is_rejected() -> None:
    """A shared subtree has ONE home but N reading sites, so a name it captures would have to be in
    scope at all of them. Closure is what makes the seam liftable — the precondition a placement cut
    rests on."""
    op = Map(body=_shared_group().op.body, sources=(_channel("acc_g", "Wg"), _channel("acc_u", "Wu")))
    with pytest.raises(AssertionError, match=re.escape("captures ['m_run']")):
        TileOp(op=op, bindings={"xhat": _capturing_cone()})


def test_a_single_reference_binding_may_capture() -> None:
    """Flash's ``P`` is exactly this: one reference, reading the running max its own loop step
    updates. Legal to build and lower — just not cuttable, which ``captured_values`` is the
    predicate for."""
    tile = TileOp(op=Map(body=Body(()), sources=(_channel("acc_g", "Wg"),)), bindings={"xhat": _capturing_cone()})
    assert captured_values(tile.bindings["xhat"], axis_names(tile.op) | axis_names(tile.bindings["xhat"])) == ("m_run",)


def test_iteration_variables_are_not_captures() -> None:
    """The dominant free names in any cone are loop induction variables (``m`` / ``k``), bound by
    the enclosing nest — excluding them is what makes the predicate mean anything."""
    cone = _cone()
    assert captured_values(cone, set()) == ("k", "m")  # unfiltered: the axes show up
    assert captured_values(cone, {"m", "k"}) == ()  # filtered: closed


# --- resolution: the inlining every lowering walk goes through ---------------------------------- #


def test_resolve_inlines_the_bound_subtree_at_every_reference() -> None:
    tile = _shared_group()
    resolved = resolve(tile.op, tile.bindings)
    cone_stmts = tuple(lower(tile.bindings["xhat"]))
    for channel in resolved.sources:
        assert channel.a_ref is None
        assert channel.a_body == cone_stmts
    # The group lowers to ONE derived fold loop, then the projection — never one loop per channel.
    assert lower(tile.op, tile.bindings) == [group_loop(resolved.sources), *tile.op.body]


def test_two_references_to_one_binding_are_not_two_copies() -> None:
    """Sharing is part of the structural identity: the group lowers its shared A ONCE (the derived
    group loop), so it is distinguishable from two independent channels that each compute their own
    A — which lower to two separate fold loops, each carrying its own cone."""
    shared = _shared_group()
    fused = lower(shared.op, shared.bindings)
    cone = tuple(lower(shared.bindings["xhat"]))
    copies = TileOp(
        op=Map(
            body=shared.op.body,
            sources=(
                _channel("acc_g", "Wg", a=Map(body=Body(cone))),
                _channel("acc_u", "Wu", a=Map(body=Body(tuple(s.rewrite(lambda n: f"{n}__2", Sigma({})) for s in cone)))),
            ),
        )
    )
    assert is_group(shared.op)
    # One fold loop, one lifted A; the unshared spelling is two loops with a cone each.
    assert len([s for s in fused if s.__class__.__name__ == "Loop"]) == 1
    assert sum(1 for s in fused[0].body if isinstance(s, Assign) and s.name.startswith("xhat")) == 1
    assert len([s for s in lower(copies.op) if s.__class__.__name__ == "Loop"]) == 2


def test_lowering_without_the_table_fails_loudly() -> None:
    """A name operand cannot lower on its own — the walkers take the owning ``TileOp``'s table, and
    a caller that forgets it gets an assertion, never a silently missing operand."""
    tile = _shared_group()
    with pytest.raises(AssertionError, match="unresolved binding"):
        lower(tile.op)


def test_resolve_is_identity_without_a_table() -> None:
    """The unshared forms pay nothing: an empty table means a name-free tree (construction
    validates it), so resolution is a free identity."""
    node = Map(body=Body((Assign(name="y", op="relu", args=("x",)),)))
    assert resolve(node, {}) is node
    assert resolve(node, None) is node


# --- references rename with their definitions --------------------------------------------------- #


def test_a_reference_renames_through_the_same_map_as_the_definition() -> None:
    """``rewrite``'s rename map is what keeps a reference and its binding key in lockstep — the
    reason no ``Ref`` node kind is needed."""
    channel = _channel("acc", "W")
    renamed = rewrite(channel, lambda n: {"xhat": "v0", "acc": "v1"}.get(n, n), Sigma({}), lambda a: a)
    assert renamed.a_ref == "v0"
    assert renamed.acc == "v1"


# --- Map.sources: the fused sibling group ------------------------------------------------------- #


def test_source_is_the_len_le_one_compat_read() -> None:
    single = Map(sources=(_channel("acc", "W"),))
    assert single.source is single.sources[0]
    assert Map().source is None
    group = _shared_group().op
    with pytest.raises(AssertionError, match="read `sources`"):
        _ = group.source


def test_pretty_prints_each_binding_once_where_it_lives() -> None:
    tile = _shared_group()
    text = tile.pretty_body()
    assert text.count("let xhat =") == 1
    assert text.count("Contraction") == 2  # the two channels, each naming the shared operand
    assert "xhat @ Wg" in text and "xhat @ Wu" in text


# --- the B edge references bindings by the same mechanism as A ----------------------------------- #


def _b_ref_channel(acc: str, b: str) -> Contraction:
    """A channel whose **B** is a binding reference — the transposed sharing shape (one B value read
    by several contractions), expressible because both edges are the same kind of edge."""
    return Contraction(
        axes=(Axis("m", 128), Axis("n", 128)),
        k_axis=Axis("k", 256),
        a=Load(name=f"{acc}_a", input="x", index=(Var("m"), Var("k"))),
        b=b,
        acc=acc,
        tile=TilePlan(),
    )


def test_a_dangling_b_reference_is_rejected_at_construction() -> None:
    """The B edge goes through the same validation as A — one reference mechanism, not two."""
    with pytest.raises(AssertionError, match=re.escape("resolve to no binding")):
        TileOp(op=Map(sources=(_b_ref_channel("acc", "wn"),)))


def test_resolve_splices_the_b_edge_too() -> None:
    cone = _cone("wn")
    tile = TileOp(op=Map(body=Body(()), sources=(_b_ref_channel("acc", "wn"),)), bindings={"wn": cone})
    (channel,) = resolve(tile.op, tile.bindings).sources
    assert channel.b_ref is None and channel.b is cone
    assert channel.b_body == tuple(lower(cone))


def test_a_shared_b_binding_must_be_closed_too() -> None:
    """Closure is a property of the EDGE's target, so it is enforced identically whichever side the
    reference sits on."""
    op = Map(body=Body(()), sources=(_b_ref_channel("acc_g", "wn"), _b_ref_channel("acc_u", "wn")))
    with pytest.raises(AssertionError, match=re.escape("captures ['m_run']")):
        TileOp(op=op, bindings={"wn": _capturing_cone("wn")})
