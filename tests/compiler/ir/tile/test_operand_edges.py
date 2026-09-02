"""Operand edges + the product-carrier a bilinear ``Fold`` — sharing is arity, not naming.

A computed operand is stored INLINE on its edge (there is no let table and no name-reference arm);
"these two matmuls read the same A" is ONE bilinear ``Fold`` with one ``a`` edge and N product
:class:`Channel`\\ s ``(b_i, acc_i)``. These pin the node's derived product loop (shared A lifted
once, N-component product-monoid carrier), the arity-vs-copies distinction, the inline-arm
and inline-arm canonicalization through ``rewrite``, and the CLOSURE predicate a placement cut asks
before lifting a subtree into its own kernel (``_cut._closed_at``, successor to the deleted
``_captured_values``).
"""

from __future__ import annotations

from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.pure.fold import Channel, Fold, operand_body, operand_name, splice_operands
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop
from emmy.compiler.ir.stmt.passes import rewrite


def _cone(name: str = "xhat") -> Fold:
    """A minimal computed A-cone — ``xhat = x[m, k] * s[k]``, the shape the fused norm→linear edge's
    computed A takes (here without its statistic reduce, which the tree vocabulary does not need)."""
    load = Load(name=f"{name}_e", input="x", index=(Var("m"), Var("k")))
    scale = Load(name=f"{name}_s", input="w", index=(Var("k"),))
    return Fold.projection(body=Body((load, scale, Assign(name=name, op="multiply", args=(f"{name}_e", f"{name}_s")))))


def _node(a, *channels: tuple[str, str]) -> Fold:
    """A contraction over operand edge ``a`` with one ``(acc, weight-buffer)`` channel per arg."""
    return Fold.contraction(
        k_axis=Axis("k", 256),
        a=a,
        channels=tuple(Channel(b=Load(name=f"{acc}_b", input=w, index=(Var("k"), Var("n"))), acc=acc) for acc, w in channels),
    )


def _product() -> Fold:
    """The gate⊗up shape: ONE product contraction, two channels over one inline cone."""
    return _node(_cone(), ("acc_g", "Wg"), ("acc_u", "Wu"))


# --- the product node: sharing is arity --------------------------------------------------------- #


def test_product_node_derives_one_fold_loop_with_the_shared_a_lifted_once() -> None:
    """The fused group lowers to ONE derived loop — the shared A evaluated once, each further
    channel splicing its ``b → ⊗ → ⊕`` triple after it — never one loop per channel."""
    stmts = _product().lower()
    assert len(stmts) == 1 and isinstance(stmts[0], Loop)
    body = list(stmts[0].body)
    assert sum(1 for s in body if isinstance(s, Assign) and s.name == "xhat") == 1
    assert [s.name for s in body if isinstance(s, Accum)] == ["acc_g", "acc_u"]


def test_product_loop_folds_the_n_component_product_state() -> None:
    loop = _product().loop
    accums = [s for s in loop.body if isinstance(s, Accum)]
    assert [a.name for a in accums] == ["acc_g", "acc_u"]
    assert all(a.op.reduce_canon == "add" for a in accums)  # the componentwise additive family


def test_arity_is_not_two_copies() -> None:
    """One node with two channels ≢ two independent contractions each computing their own A: the
    former lifts the shared A once (one loop), the latter lower to two loops with a cone each."""
    fused = Fold.projection(body=Body(()), operands=(_product(),)).lower()
    copies = Fold.projection(body=Body(()), operands=(_node(_cone(), ("acc_g", "Wg")), _node(_cone("xhat2"), ("acc_u", "Wu"))))
    assert len([s for s in fused if isinstance(s, Loop)]) == 1
    assert len([s for s in copies.lower() if isinstance(s, Loop)]) == 2


def test_defines_and_out_read_the_channels() -> None:
    node = _product()
    assert node.defines() == ("acc_g", "acc_u")
    assert node.out == "acc_g" and node.acc == "acc_g"  # the primary channel
    assert node.b is node.channels[0].b  # the primary-channel read single-channel tiers use


def test_single_channel_node_is_the_plain_matmul() -> None:
    node = _node(Load(name="a_e", input="x", index=(Var("m"), Var("k"))), ("acc", "W"))
    (loop,) = node.lower()
    assert [s.name for s in loop.body if isinstance(s, Accum)] == ["acc"]
    assert isinstance(node.a, Load)


# --- inline computed operands ------------------------------------------------------------------- #


def test_splice_orders_a_sibling_provider_before_its_consumer() -> None:
    """A provider used only by another operand still precedes that dependent operand."""
    provider = Fold.projection(body=Body((Load(name="raw", input="x", index=()), Assign(name="scale", op="rsqrt", args=("raw",)))))
    consumer = Fold.projection(body=Body((Assign(name="weighted", op="multiply", args=("value", "scale")),)))
    independent = Fold.projection(body=Body((Assign(name="offset", op="copy", args=("bias",)),)))
    projection = (Assign(name="out", op="add", args=("weighted", "offset")),)

    lowered = splice_operands((consumer, independent, provider), projection)

    assert [stmt.defines()[-1] for stmt in lowered] == ["offset", "raw", "scale", "weighted", "out"]


def test_a_computed_operand_is_stored_inline_and_flattens_on_the_edge() -> None:
    node = _product()
    assert not isinstance(node.a, Load)
    assert operand_body(node.a) == tuple(node.a.lower())
    assert operand_name(node.a) == "xhat"


def test_pretty_prints_the_channels_once() -> None:
    """One shared A edge, one branch per channel, each labelled by the lift param it binds."""
    from emmy.compiler.ir.tile._dump import pretty

    text = "\n".join(pretty(_product()))
    assert text.count("operand[xhat]:") == 1  # the SHARED A edge — printed once, not once per channel
    assert text.count("xhat = multiply") == 1  # and its cone body with it
    assert "operand[acc_g_b] -> acc_g" in text and "operand[acc_u_b] -> acc_u" in text


# --- canonicalization: inline arms rewrite like any other subtree -------------------------------- #


def test_rewrite_renames_channel_accs_and_inline_arms_in_lockstep() -> None:
    """The canonicalizer runs over STORED trees — the node: the accs and the inline cone rename
    through one map."""
    node = _product()
    renamed = rewrite(node, lambda n: {"xhat": "v0", "acc_g": "v1", "acc_g__v": "v1__v"}.get(n, n), Sigma({}), lambda a: a)
    assert renamed.channels[0].acc == "v1" and renamed.channels[1].acc == "acc_u"
    assert renamed.a.out == "v0"  # the inline cone canonicalized through the same map


def test_rewrite_reaches_a_channels_b_edge() -> None:
    node = _product()
    renamed = rewrite(node, lambda n: {"acc_u_b": "vb"}.get(n, n), Sigma({}), lambda a: a)
    assert renamed.channels[1].b.names == ("vb",)


# --- closure: the predicate a placement cut asks -------------------------------------------------- #


def _capturing_cone(name: str = "xhat") -> Fold:
    """A cone that READS a value the enclosing body defines (``m_run``) instead of producing it —
    the flash ``P = exp(s - m)`` shape, where the running max comes from the carrier merge."""
    load = Load(name=f"{name}_e", input="x", index=(Var("m"), Var("k")))
    return Fold.projection(body=Body((load, Assign(name=name, op="subtract", args=(f"{name}_e", "m_run")))))


def test_external_reads_cover_every_channel() -> None:
    """RESTORED: the node's derived loop reads every buffer it touches — the shared A's two and
    both channels' weights. A channel dropped from the reading is a kernel missing an argument.

    ``Fold.external_reads`` is gone with the recognition-era node API, so the reading comes off the
    derived loop, which is what materialization and kernel binding actually walk."""

    def buffers(stmts):
        for stmt in stmts:
            if isinstance(stmt, Load):
                yield stmt.input
            for body in stmt.nested():
                yield from buffers(body)

    assert set(buffers(_product().lower())) == {"x", "w", "Wg", "Wu"}


def test_a_capturing_inline_operand_is_legal_but_reports_its_capture() -> None:
    """Flash's ``P`` is exactly this: an inline operand reading the running max its own loop step
    updates. Legal to build and lower (its one home is in scope) — just not CUTTABLE, which is
    what the closure predicate is for. RESTORED: cutting a capturing cone into its own kernel
    produces a kernel that reads an undefined name."""
    from emmy.compiler.pipeline.passes.lowering.tile._cut import _closed_at

    node = _node(_capturing_cone(), ("acc_g", "Wg"))
    assert node.lower()  # lowers fine — position in the enclosing body is what makes it legal
    cone = node.a
    # The output axes are the CALLER's placement — never on the node — so the cut supplies them.
    assert not _closed_at(cone, (Axis("m", 256), Axis("k", 256))), "a cone capturing carrier state is not closed"


def test_contraction_deps_include_inline_operand_captures() -> None:
    node = _node(_capturing_cone(), ("acc_g", "Wg"))

    assert "m_run" in node.deps()


def test_cut_closure_does_not_confuse_a_sibling_loop_axis_for_scope() -> None:
    """A loop binding ``k`` in one operand does not scope a sibling operand's ``x[k]`` load."""
    from emmy.compiler.pipeline.passes.lowering.tile._cut import _closed_at
    from emmy.compiler.pipeline.passes.lowering.tile._fromloop import fold_from_loop

    loop = Loop(
        axis=Axis("k", 8),
        body=Body((Load(name="value", input="x", index=(Var("k"),)), Accum(name="total", value="value", op="add"))),
        role=AxisRole.PLANAR,
    )
    bound = fold_from_loop(loop)
    leak = Fold.projection(body=Body((Load(name="leak", input="x", index=(Var("k"),)),)), results=("leak",))
    root = Fold.projection(
        operands=(bound, leak),
        body=Body((Assign(name="out", op="add", args=("total", "leak")),)),
    )

    assert not _closed_at(root, ())


def test_cut_closure_includes_dead_but_emitted_axis_reads() -> None:
    """Until lowering removes a dead statement, its free axis still has to reach emitted CUDA."""
    from emmy.compiler.pipeline.passes.lowering.tile._cut import _closed_at

    root = Fold.projection(
        body=Body((Load(name="dead", input="x", index=(Var("m"),)), Load(name="live", input="y", index=()))),
        results=("live",),
    )

    assert not _closed_at(root, ())
    assert _closed_at(root, (Axis("m", 8),))


def test_iteration_variables_are_not_captures() -> None:
    """The dominant free names in any cone are loop induction variables (``m`` / ``k``), bound by
    the enclosing nest — excluding them is what makes the predicate mean anything."""
    from emmy.compiler.pipeline.passes.lowering.tile._cut import _closed_at

    cone = _cone()
    assert _closed_at(cone, (Axis("m", 256), Axis("k", 256))), "an ordinary cone over its own axes is closed"
    assert not _closed_at(cone, ()), "unfiltered, the axes themselves read as captures"


# --- the projection binder ----------------------------------------------------------------------- #


def test_a_shared_cone_is_typed_in_each_occurrence_scope() -> None:
    """``edge_dtypes`` answers per OCCURRENCE, not per object.

    Normalization shares one Fold object between structurally identical cones, so the same cone
    can sit under an f16 capture in one host and an f32 capture in another. Its result dtype is a
    property of the occurrence, and a cache keyed on identity alone handed every occurrence the
    first one's answer — which a cut then believes when it sizes the seam's workspace."""
    from emmy.compiler.dtype import get as get_dtype
    from emmy.compiler.ir.tile.ops import edge_dtypes
    from emmy.compiler.tensor import Tensor

    shared = Fold.projection(body=Body((Assign(name="y", op="relu", args=("x",)),)), results=("y",))
    inputs = {"a": Tensor("a", (4,), get_dtype("f16")), "b": Tensor("b", (4,), get_dtype("f32"))}

    def host(buf: str, out: str) -> Fold:
        source = Fold.projection(body=Body((Load(name="x", input=buf, index=(Var("i"),)),)), results=("x",))
        return Fold.projection(operands=(source, shared), body=Body((Assign(name=out, op="copy", args=("y",)),)), results=(out,))

    # Both hosts stay referenced for the whole check: the cache keys on object identity, which is
    # only stable while the keyed objects are alive.
    narrow_host, wide_host = host("a", "z0"), host("b", "z1")
    cache: dict = {}
    narrow = edge_dtypes(narrow_host, inputs, cache)
    wide = edge_dtypes(wide_host, inputs, cache)

    assert [dtype.name for dtype in narrow] == ["f16"]
    assert [dtype.name for dtype in wide] == ["f32"]
