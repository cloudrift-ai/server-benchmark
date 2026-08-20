"""``loop/canonicalize``: re-fuse adjacent free axes that a fused reshape split.

A view fused into a contraction iterates the post-view axes while the operand loads address the
producer's single axis through a composite index — which locks the kernel out of contraction
binding (the trailing free pair reads the wrong row and the weight load carries a third grid
axis). The canonicalization restores the single-axis spelling; these tests pin the fuse cases
(N split, M split), the decline cases (an axis addressed alone, a permuted-stride split), the
downstream classification of the canonical nest, and the binder's per-expr role purity."""

from __future__ import annotations

from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
from emmy.compiler.pipeline import Pipeline
from emmy.compiler.pipeline.passes.lowering.tile._classify import bind_bilinear
from emmy.compiler.pipeline.passes.lowering.tile._fromloop import _stamp_axes, fold_from_loop
from emmy.compiler.pipeline.passes.lowering.tile._lift import recognized_tile

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


def test_permuted_split_declines():
    """A store whose dims transpose the split (``[…, d, h]`` under an ``h*D + d`` operand index)
    has no matching stride — the flatten cannot fold the residue, so the pair declines."""
    op = _run(_graph(_split_n_body((Literal(0, "int"), Var("a0"), Var("a2"), Var("a1"))), (1, M, D, H)))
    assert [ln.axis.extent for ln in _free_chain(op)] == [Dim(M), Dim(H), Dim(D)]


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
    tile = recognized_tile(LoopOp(body=body))
    assert [a.extent for a in tile.place.free] == [Dim(M), Dim(H * D)]
    node = tile.op
    assert node.axis is not None and node.role is AxisRole.CONTRACTION
    assert isinstance(node.b, Load) and node.b.input == "w"


def _bilinear_fold(w_index: tuple, x_index: tuple):
    loop = Loop(
        axis=Axis("k", Dim(K)),
        body=Body(
            (
                Load(name="wv", input="w", index=w_index),
                Load(name="xv", input="x", index=x_index),
                Assign(name="prod", op=ElementwiseImpl("multiply"), args=("wv", "xv")),
                Accum(name="acc", value="prod", op=ElementwiseImpl("add"), axes=("k",)),
            )
        ),
    )
    return fold_from_loop(_stamp_axes(loop))


def test_bind_bilinear_batched_operand_still_binds():
    """Per-expr role purity must not over-reach: a batch axis riding a SEPARATE dim of the A
    load (batched GEMM) binds exactly as before."""
    f = _bilinear_fold((Var("n"), Var("k")), (Var("b"), Var("a0"), Var("k")))
    r = bind_bilinear(f, "a0", "n", frozenset({"b", "a0", "n"}))
    assert r is not None
    con, epi = r
    assert isinstance(con.a, Load) and con.a.input == "x" and not epi


def test_bind_bilinear_declines_composite_role_expr():
    """A third free axis composed into the SAME index expr as the role axis (the split-axis
    composite) must not bind as the direct B load — the mma slab template cannot address it.
    Before the per-expr purity check this bound and emitted code referencing an undefined
    iteration var."""
    comp = BinaryExpr("+", BinaryExpr("*", Var("a1"), Literal(D, "int")), Var("n"))
    f = _bilinear_fold((comp, Var("k")), (Var("b"), Var("a0"), Var("k")))
    r = bind_bilinear(f, "a0", "n", frozenset({"b", "a1", "a0", "n"}))
    if r is not None:
        con, _ = r
        for ch in con.channels:
            assert not isinstance(ch.b, Load), "the impure composite must not become a direct slab load"
