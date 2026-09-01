"""``loop/canonicalize``: re-fuse adjacent free axes that a fused reshape split.

A view fused into a contraction iterates the post-view axes while the operand loads address the
producer's single axis through a composite index — which locks the kernel out of contraction
binding (the trailing free pair reads the wrong row and the weight load carries a third grid
axis). The canonicalization restores the single-axis spelling; these tests pin the fuse cases
(N split, M split, the pair across an intervening free loop — the transposed projection, the
permuted split store), the decline case (an axis addressed alone), the downstream classification
of the canonical nest, and the warp tier's split-store addressability.

The operand ROLE-PURITY section at the end was deleted with ``_classify.bind_bilinear`` and is
RESTORED against the canonical Fold tree. Its contracts are about correctness, not coverage: a
composite index that binds as a direct slab load emits code referencing an undefined iteration
variable, a grouped B address that varies with the output row is not one slab per tile, and trying
the opposite operand orientation is licensed only for a COMMUTATIVE product — reordering a
noncommutative one computes a different value."""

from __future__ import annotations

from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.address import split_addressable
from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import Pipeline

M, H, D, K = 8, 3, 4, 16  # N = H*D = 12


def _kloop(w_index: tuple) -> Loop:
    return Loop(
        axis=Axis("k", Dim(K)),
        body=Body(
            (
                Load(name="wv", input="w", index=w_index),
                Load(name="xv", input="x", index=(Literal(0, "int"), Var("a0"), Var("k"))),
                Assign(name="prod", op=ElementwiseImpl("multiply"), args=("wv", "xv")),
                Accum(name="acc", value="prod", op=ElementwiseImpl("add"), axes=("k",)),
            )
        ),
    )


def _graph(body: Body, out_shape: tuple) -> Graph:
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (1, M, K)), node_id="x")
    g.add_node(InputOp(), [], Tensor("w", (H * D, K)), node_id="w")
    g.add_node(LoopOp(body=body), ["x", "w"], Tensor("out", out_shape), node_id="out")
    g.inputs, g.outputs = ["x", "w"], ["out"]
    return g


def _split_n_body(write_index: tuple, tail=()) -> Body:
    """The fused-reshape spelling: free (m, h, d), the weight addressed ``w[h*D + d, k]``."""
    comp = BinaryExpr("+", BinaryExpr("*", Var("a1"), Literal(D, "int")), Var("a2"))
    cell = (_kloop((comp, Var("k"))), *tail, Write(output="out", index=write_index, value="acc"))
    nest = Loop(
        axis=Axis("a0", Dim(M)),
        body=Body((Loop(axis=Axis("a1", Dim(H)), body=Body((Loop(axis=Axis("a2", Dim(D)), body=Body(tuple(cell))),))),)),
    )
    return Body((nest,))


def _free_chain(op: LoopOp) -> list[Loop]:
    out, cur = [], op.body
    while len(cur) == 1 and isinstance(cur[0], Loop) and not cur[0].is_reduce:
        out.append(cur[0])
        cur = cur[0].body
    return out


def _run(g: Graph) -> LoopOp:
    return Pipeline.build(["loop/canonicalize"]).run(g).nodes["out"].op


def _lift(op: LoopOp, shape=(1,)) -> TileOp:
    graph = Graph()
    graph.add_node(op, [], Tensor("out", shape), node_id="out")
    graph.outputs = ["out"]
    return Pipeline.build(["lowering/tile"], select=["lift"]).run(graph).nodes["out"].op


def _reduce_name(op: LoopOp) -> str:
    return next(s.axis.name for s in op.body.iter() if isinstance(s, Loop) and s.is_reduce)


def test_split_n_pair_fuses():
    op = _run(_graph(_split_n_body((Literal(0, "int"), Var("a0"), Var("a1"), Var("a2"))), (1, M, H, D)))
    chain = _free_chain(op)
    assert [ln.axis.extent for ln in chain] == [Dim(M), Dim(H * D)], "the (h, d) pair must fuse into one N axis"
    n = chain[1].axis.name
    wv = next(s for s in op.body.iter() if isinstance(s, Load) and s.input == "w")
    assert wv.index == (Var(n), Var(_reduce_name(op))), "the composite operand index must collapse to the bare fused axis"
    wr = next(s for s in op.body.iter() if isinstance(s, Write))
    assert wr.index[2] == BinaryExpr("//", Var(n), Literal(D, "int"))
    assert wr.index[3] == BinaryExpr("%", Var(n), Literal(D, "int"))


def test_split_m_pair_fuses():
    """An M-side split (``view`` carving the row axis) fuses the same way: composite x load,
    split write rows."""
    comp = BinaryExpr("+", BinaryExpr("*", Var("a0"), Literal(4, "int")), Var("a1"))
    kloop = Loop(
        axis=Axis("k", Dim(K)),
        body=Body(
            (
                Load(name="wv", input="w", index=(Var("n"), Var("k"))),
                Load(name="xv", input="x", index=(Literal(0, "int"), comp, Var("k"))),
                Assign(name="prod", op=ElementwiseImpl("multiply"), args=("wv", "xv")),
                Accum(name="acc", value="prod", op=ElementwiseImpl("add"), axes=("k",)),
            )
        ),
    )
    cell = (kloop, Write(output="out", index=(Literal(0, "int"), Var("a0"), Var("a1"), Var("n")), value="acc"))
    nest = Loop(
        axis=Axis("a0", Dim(2)),
        body=Body((Loop(axis=Axis("a1", Dim(4)), body=Body((Loop(axis=Axis("n", Dim(H * D)), body=Body(cell)),))),)),
    )
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (1, M, K)), node_id="x")
    g.add_node(InputOp(), [], Tensor("w", (H * D, K)), node_id="w")
    g.add_node(LoopOp(body=Body((nest,))), ["x", "w"], Tensor("out", (1, 2, 4, H * D)), node_id="out")
    g.inputs, g.outputs = ["x", "w"], ["out"]
    op = _run(g)
    chain = _free_chain(op)
    assert [ln.axis.extent for ln in chain] == [Dim(M), Dim(H * D)]
    xv = next(s for s in op.body.iter() if isinstance(s, Load) and s.input == "x")
    assert xv.index == (Literal(0, "int"), Var(chain[0].axis.name), Var(_reduce_name(op)))


def _packed_kloop(w_index: tuple) -> Loop:
    """The NVFP4 weight read: two 4-bit codes per byte, so the operand address is the FLAT element
    offset divided by 2 and wrapped back into the packed row."""
    return Loop(
        axis=Axis("k", Dim(K)),
        body=Body(
            (
                Load(name="wv", input="w", index=w_index),
                Load(name="xv", input="x", index=(Literal(0, "int"), Var("a0"), Var("k"))),
                Assign(name="prod", op=ElementwiseImpl("multiply"), args=("wv", "xv")),
                Accum(name="acc", value="prod", op=ElementwiseImpl("add"), axes=("k",)),
            )
        ),
    )


def _packed_index(n_expr) -> tuple:
    """``w[n, ((n*K + k) / 2) % (K/2)]`` — the address a pair-packed constant carries."""
    flat = BinaryExpr("+", BinaryExpr("*", n_expr, Literal(K, "int")), Var("k"))
    return (n_expr, BinaryExpr("%", BinaryExpr("/", flat, Literal(2, "int")), Literal(K // 2, "int")))


def _packed_split_graph() -> Graph:
    """The deployed shape: a pair-packed weight under an output whose N axis a fused fp4-block
    reshape split into ``(H, D)``."""
    comp = BinaryExpr("+", BinaryExpr("*", Var("a1"), Literal(D, "int")), Var("a2"))
    cell = (
        _packed_kloop(_packed_index(comp)),
        Write(output="out", index=(Literal(0, "int"), Var("a0"), Var("a1"), Var("a2")), value="acc"),
    )
    nest = Loop(
        axis=Axis("a0", Dim(M)),
        body=Body((Loop(axis=Axis("a1", Dim(H)), body=Body((Loop(axis=Axis("a2", Dim(D)), body=Body(cell)),))),)),
    )
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (1, M, K)), node_id="x")
    g.add_node(InputOp(), [], Tensor("w", (H * D, K // 2)), node_id="w")
    g.add_node(LoopOp(body=Body((nest,))), ["x", "w"], Tensor("out", (1, M, H, D)), node_id="out")
    g.inputs, g.outputs = ["x", "w"], ["out"]
    return g


def test_packed_operand_split_pair_fuses():
    """A pair-packed weight address holds the row axis inside a division. The pair still fuses: the
    composite collapses to the bare axis, and the division that remains reads ``k`` alone."""
    op = _run(_packed_split_graph())
    chain = _free_chain(op)
    assert [ln.axis.extent for ln in chain] == [Dim(M), Dim(H * D)], "the packed pair must fuse"
    n = chain[1].axis.name
    wv = next(s for s in op.body.iter() if isinstance(s, Load) and s.input == "w")
    assert wv.index[0] == Var(n), "the packed row dim must collapse to the bare fused axis"
    assert n not in wv.index[1].free_vars(), f"the row axis stayed inside the packed offset: {wv.index[1].pretty()}"


def test_packed_split_nest_classifies_as_contraction():
    """Downstream: the fused packed nest binds the contraction with the weight as B — the tensor-core
    tier becomes reachable, which is what the split had locked out."""
    tile = _lift(_run(_packed_split_graph()))
    assert [a.extent for a in tile.place.free] == [Dim(M), Dim(H * D)]
    assert tile.op.axis is not None and tile.op.role is AxisRole.CONTRACTION


def test_axis_addressed_alone_declines():
    """A per-column bias read ``b[d]`` pins ``d``'s identity apart from ``h`` — the pair must
    not fuse (the residue ``b[f%D]`` would survive in a load)."""
    tail = (
        Load(name="bv", input="b", index=(Var("a2"),)),
        Assign(name="outv", op=ElementwiseImpl("add"), args=("acc", "bv")),
    )
    comp = BinaryExpr("+", BinaryExpr("*", Var("a1"), Literal(D, "int")), Var("a2"))
    cell = (_kloop((comp, Var("k"))), *tail, Write(output="out", index=(Literal(0, "int"), Var("a0"), Var("a1"), Var("a2")), value="outv"))
    nest = Loop(
        axis=Axis("a0", Dim(M)),
        body=Body((Loop(axis=Axis("a1", Dim(H)), body=Body((Loop(axis=Axis("a2", Dim(D)), body=Body(cell)),))),)),
    )
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (1, M, K)), node_id="x")
    g.add_node(InputOp(), [], Tensor("w", (H * D, K)), node_id="w")
    g.add_node(InputOp(), [], Tensor("b", (D,)), node_id="b")
    g.add_node(LoopOp(body=Body((nest,))), ["x", "w", "b"], Tensor("out", (1, M, H, D)), node_id="out")
    g.inputs, g.outputs = ["x", "w", "b"], ["out"]
    op = _run(g)
    assert [ln.axis.extent for ln in _free_chain(op)] == [Dim(M), Dim(H), Dim(D)], "the pair must decline"


def test_permuted_split_store_keeps_the_axes_separate():
    """Output-storage order is canonical, so a transposed ``[d, h]`` store keeps the pair split."""
    op = _run(_graph(_split_n_body((Literal(0, "int"), Var("a0"), Var("a2"), Var("a1"))), (1, M, D, H)))
    chain = _free_chain(op)
    assert [ln.axis.extent for ln in chain] == [Dim(M), Dim(D), Dim(H)]
    wr = next(s for s in op.body.iter() if isinstance(s, Write))
    assert wr.index == (Literal(0, "int"), *(Var(loop.axis.name) for loop in chain))


def _transposed_body() -> Body:
    """The attention projection's ``view(b, s, h, d).transpose(1, 2)``: free (h, s, d) — the
    split pair (h, d) is NOT adjacent, ``s`` sits between — with the weight addressed
    ``w[h*D + d, k]`` and the output stored ``[0, h, s, d]``."""
    comp = BinaryExpr("+", BinaryExpr("*", Var("a0"), Literal(D, "int")), Var("a2"))
    kloop = Loop(
        axis=Axis("k", Dim(K)),
        body=Body(
            (
                Load(name="wv", input="w", index=(comp, Var("k"))),
                Load(name="xv", input="x", index=(Literal(0, "int"), Var("a1"), Var("k"))),
                Assign(name="prod", op=ElementwiseImpl("multiply"), args=("wv", "xv")),
                Accum(name="acc", value="prod", op=ElementwiseImpl("add"), axes=("k",)),
            )
        ),
    )
    cell = (kloop, Write(output="out", index=(Literal(0, "int"), Var("a0"), Var("a1"), Var("a2")), value="acc"))
    nest = Loop(
        axis=Axis("a0", Dim(H)),
        body=Body((Loop(axis=Axis("a1", Dim(M)), body=Body((Loop(axis=Axis("a2", Dim(D)), body=Body(cell)),))),)),
    )
    return Body((nest,))


def test_split_pair_fuses_across_an_intervening_free_loop():
    """The transposed projection: the pair interchanges past ``s`` (free loops are parallel) and
    fuses under it — the weight load collapses to the bare axis, the store keeps the permuted
    ``[0, f//D, s, f%D]``."""
    op = _run(_graph(_transposed_body(), (1, H, M, D)))
    chain = _free_chain(op)
    assert [ln.axis.extent for ln in chain] == [Dim(M), Dim(H * D)], "s stays outer; the (h, d) pair fuses under it"
    n = chain[1].axis.name
    wv = next(s for s in op.body.iter() if isinstance(s, Load) and s.input == "w")
    assert wv.index == (Var(n), Var(_reduce_name(op)))
    wr = next(s for s in op.body.iter() if isinstance(s, Write))
    s_name = chain[0].axis.name
    assert wr.index == (
        Literal(0, "int"),
        BinaryExpr("//", Var(n), Literal(D, "int")),
        Var(s_name),
        BinaryExpr("%", Var(n), Literal(D, "int")),
    )


def test_transposed_canonical_nest_orders_free_by_the_remainder_dim():
    """Downstream: the lift positions an axis by the INNERMOST store dim carrying it, so under
    the permuted store the fused axis stays the trailing ``n`` (its ``%`` dim is the contiguous
    one) and the contraction binds with the weight as B. Positioning by the quotient dim would
    make the fused axis ``m`` and the stride-``D`` ``s`` the column — the mma store's ``+ col``
    would then address the wrong element (and did: a hang on the GPU)."""
    op = _run(_graph(_transposed_body(), (1, H, M, D)))
    tile = _lift(op)
    assert [a.extent for a in tile.place.free] == [Dim(M), Dim(H * D)]
    node = tile.op
    assert node.axis is not None and node.role is AxisRole.CONTRACTION
    assert isinstance(node.b, Load) and node.b.input == "w"


def test_canonical_nest_classifies_as_contraction():
    """Downstream of the canonicalization: the fused nest binds the contraction with the split
    store, and the free order keeps the fused N trailing (position read through the div/mod
    exprs, not bare Vars)."""
    n = Axis("n", Dim(H * D))
    kloop = Loop(
        axis=Axis("k", Dim(K)),
        body=Body(
            (
                Load(name="wv", input="w", index=(Var("n"), Var("k"))),
                Load(name="xv", input="x", index=(Literal(0, "int"), Var("a0"), Var("k"))),
                Assign(name="prod", op=ElementwiseImpl("multiply"), args=("wv", "xv")),
                Accum(name="acc", value="prod", op=ElementwiseImpl("add"), axes=("k",)),
            )
        ),
    )
    wr = Write(
        output="out",
        index=(
            Literal(0, "int"),
            Var("a0"),
            BinaryExpr("//", Var("n"), Literal(D, "int")),
            BinaryExpr("%", Var("n"), Literal(D, "int")),
        ),
        value="acc",
    )
    body = Body((Loop(axis=Axis("a0", Dim(M)), body=Body((Loop(axis=n, body=Body((kloop, wr))),))),))
    tile = _lift(LoopOp(body=body))
    assert [a.extent for a in tile.place.free] == [Dim(M), Dim(H * D)]
    node = tile.op
    assert node.axis is not None and node.role is AxisRole.CONTRACTION
    assert isinstance(node.b, Load) and node.b.input == "w"


# --- the warp tier's split-store addressability --------------------------------------------------- #


def _split_store_ok(index: tuple, shape: tuple, free_names=("m", "n"), atom=(16, 8, 16)) -> bool:
    """Whether an mma fragment store can address both free-axis roles of ``index``."""
    shape = tuple(Dim(extent) for extent in shape)
    roles = ((free_names[-1], atom[1], True),)
    if len(free_names) >= 2:
        roles = (*roles, (free_names[-2], atom[0], False))
    return all(split_addressable(index, shape, name, extent, trailing) for name, extent, trailing in roles)


def _pair(name: str, q: int, op: str):
    return BinaryExpr(op, Var(name), Literal(q, "int"))


def test_warp_split_store_legality():
    """The fragment store evaluates the cell base once per atom and adds ``col`` / ``row·ldm``:
    a split dim pair is addressable when the row-major flatten recomposes it (strides match the
    split, any ``Q``), or when the ``%`` dim is the innermost carrier, contiguous for ``n``, with
    ``Q`` a multiple of the atom extent — an aligned atom never straddles a ``Q`` boundary."""
    lit0 = Literal(0, "int")
    # #561's row-major N split: affine recomposition, Q=4 is fine.
    assert _split_store_ok((lit0, Var("m"), _pair("n", 4, "//"), _pair("n", 4, "%")), (1, 8, 3, 4))
    # The transposed projection: permuted strides, Q % 8 == 0.
    assert _split_store_ok((lit0, _pair("n", 32, "//"), Var("m"), _pair("n", 32, "%")), (1, 2, 8, 32))
    # Permuted with Q=12: an 8-wide atom straddles a head boundary.
    assert not _split_store_ok((lit0, _pair("n", 12, "//"), Var("m"), _pair("n", 12, "%")), (1, 4, 8, 12))
    # The within-pair transpose (quotient dim inner) never addresses — in either stride regime.
    assert not _split_store_ok((lit0, Var("m"), _pair("n", 32, "%"), _pair("n", 32, "//")), (1, 8, 32, 2))
    # An M-side permuted split (a batch dim between the row's pair) needs 16-row atoms inside
    # one ``P`` block.
    assert _split_store_ok((lit0, _pair("m", 32, "//"), Var("b"), _pair("m", 32, "%"), Var("n")), (1, 2, 3, 32, 16))
    assert not _split_store_ok((lit0, _pair("m", 8, "//"), Var("b"), _pair("m", 8, "%"), Var("n")), (1, 2, 3, 8, 16))


def test_warp_roles_move_only_the_innermost_carrier():
    """An epilogue load under a split store carries ``n`` in two dims; only the innermost (the
    ``%`` dim) moves within the atom — both dims moving would add the lane offset at two
    strides."""
    from emmy.compiler.pipeline.passes.lowering.kernel._atom import _warp_roles

    lit0 = Literal(0, "int")
    assert _warp_roles((lit0, Var("m"), _pair("n", 32, "//"), _pair("n", 32, "%")), "m", "n") == ("fixed", "m", "fixed", "n")
    assert _warp_roles((_pair("n", 32, "//"), Var("m"), _pair("n", 32, "%")), "m", "n") == ("fixed", "m", "n")
    assert _warp_roles((Var("b"), Var("m"), Var("n")), "m", "n") == ("fixed", "m", "n")


# --- operand role purity and product orientation (restored) --------------------------------------- #


def _bilinear_fold(w_index: tuple, x_index: tuple, product: str = "multiply"):
    from emmy.compiler.pipeline.passes.lowering.tile._fromloop import _stamp_axes, fold_from_loop

    loop = Loop(
        axis=Axis("k", Dim(K)),
        body=Body(
            (
                Load(name="wv", input="w", index=w_index),
                Load(name="xv", input="x", index=x_index),
                Assign(name="prod", op=ElementwiseImpl(product), args=("wv", "xv")),
                Accum(name="acc", value="prod", op=ElementwiseImpl("add"), axes=("k",)),
            )
        ),
    )
    return fold_from_loop(_stamp_axes(loop))


def _bind(fold, free_names):
    """Canonicalize the lifted fold under ``free_names``; return the contraction, or ``None`` when
    the tree keeps its PLANAR reading (the decline every negative case below asserts)."""
    from emmy.compiler.ir.pure.fold import Fold, is_contraction
    from emmy.compiler.ir.tile import Placement

    tile = TileOp(op=Fold.projection(body=Body((fold,))), place=Placement(free=tuple(Axis(n, Dim(4)) for n in free_names)))
    root = tile.op
    if is_contraction(root):
        return root
    inner = [s for s in root.lift.body if isinstance(s, Fold) and is_contraction(s)]
    inner += [o for o in root.operands if isinstance(o, Fold) and is_contraction(o)]
    return inner[0] if inner else None


def test_bilinear_batched_operand_still_binds():
    """Role purity must not over-reach: a batch offset riding a SEPARATE dim of the A load
    (batched GEMM) binds exactly as an unbatched one does.

    The batch dim is a grid offset, not a scheduling axis — ``loop/canonicalize`` folds a leading
    batch into the row axis before lowering, so the canonical term sees the ordinary ``(m, n)``
    pair with the offset still spelled in A's index."""
    con = _bind(_bilinear_fold((Var("n"), Var("k")), (Var("b"), Var("a0"), Var("k"))), ("a0", "n"))
    assert con is not None, "the batched GEMM demoted to PLANAR"
    assert isinstance(con.a, Load) and con.a.input == "x"


def test_bilinear_declines_composite_role_expr():
    """A third free axis composed into the SAME index expr as the role axis (the split-axis
    composite) must not bind as the direct B load — the mma slab template cannot address it.
    Before the per-expr purity check this bound and emitted code referencing an undefined
    iteration variable."""
    comp = BinaryExpr("+", BinaryExpr("*", Var("a1"), Literal(D, "int")), Var("n"))
    con = _bind(_bilinear_fold((comp, Var("k")), (Var("b"), Var("a0"), Var("k"))), ("b", "a1", "a0", "n"))
    if con is not None:
        for channel in con.channels:
            assert not isinstance(channel.b, Load), "the impure composite must not become a direct slab load"


def test_bilinear_binding_is_independent_of_the_product_argument_order():
    """A flattened GQA value row binds the same whichever way the commutative product is spelled.

    Fusion emits the value load first in the deployed attention cell, so an order-sensitive binder
    would bind one spelling and decline its twin. What the B edge becomes is NOT asserted here: the
    old binder answered slab addressability itself by forcing a computed cone, and that question
    now belongs to the scheduler's warp-atom gate. Order independence is the part that is still this rule's."""
    group = BinaryExpr("//", Var("h"), Literal(3, "int"))
    flat = BinaryExpr("+", BinaryExpr("*", group, Literal(D, "int")), Var("n"))
    w_index = (Literal(0, "int"), Var("k"), Literal(0, "int"), flat)
    x_index = (Var("h"), Var("m"), Var("k"))

    forward = _bind(_bilinear_fold(w_index, x_index), ("h", "m", "n"))
    assert forward is not None, "the grouped value row demoted to PLANAR"
    assert isinstance(forward.a, Load) and forward.a.input == "x"

    swapped = _bilinear_fold(w_index, x_index)
    reversed_products = _bind(swapped, ("h", "m", "n"))
    assert reversed_products is not None
    assert reversed_products.structural_key() == forward.structural_key()


def test_bilinear_rejects_grouped_b_that_changes_with_the_row():
    """A grouped value address that also reads the output row is not one B slab per tile. Trying
    the commutative product's other orientation must still fail closed."""
    group = BinaryExpr("//", Var("h"), Literal(3, "int"))
    row = BinaryExpr("*", Var("m"), Literal(H * D, "int"))
    flat = BinaryExpr("+", BinaryExpr("+", row, BinaryExpr("*", group, Literal(D, "int"))), Var("n"))
    fold = _bilinear_fold((Literal(0, "int"), Var("k"), Literal(0, "int"), flat), (Var("h"), Var("m"), Var("k")))

    assert _bind(fold, ("h", "m", "n")) is None


def test_bilinear_does_not_reorder_a_noncommutative_product():
    """Trying the opposite direct/computed role is licensed only for a COMMUTATIVE product.
    Reordering ``subtract`` computes a different value — a miscompile, not a missed schedule."""
    group = BinaryExpr("//", Var("h"), Literal(3, "int"))
    flat = BinaryExpr("+", BinaryExpr("*", group, Literal(D, "int")), Var("n"))
    fold = _bilinear_fold(
        (Literal(0, "int"), Var("k"), Literal(0, "int"), flat),
        (Var("h"), Var("m"), Var("k")),
        product="subtract",
    )

    assert _bind(fold, ("h", "m", "n")) is None
