"""The :class:`Fold` / a bilinear ``Fold`` / :class:`Map` structural tile-IR nodes — their
algebra/structure split and the round-trip invariant the materializer relies on.

A ``Fold`` is the typed successor of the bare annotated reduce ``Loop`` (holding no projection);
a projected reduce (softmax / RMSNorm) is a ``Map`` whose body IS the projection over a ``Fold``
``source``. A bilinear ``Fold`` is the tiled matmul node (built recognize-side at fork-emit), holding
its ``a`` edge + product channels + ``tile``; it synthesizes the canonical mul-add ``CONTRACTION``
loop on demand so ``Fold.lower`` / ``reduce_loop`` flatten it back to the loop nest the materializer
expands (via ``_factor.factorize``). These pin that contract plus the structural reads (``axis_role`` /
``reduce_loop`` / ``out``) dispatching on the nodes.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.schedule import TilePlan
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
from emmy.compiler.ir.tile import Channel, Fold, ReducePlan, TileOp
from emmy.compiler.ir.tile.ir import operand_body, operand_name
from emmy.compiler.ir.tile.ops import axis_role, cone_seam, reduce_loop, reduce_plan
from emmy.compiler.pipeline.passes.lowering.tile._fromloop import fold_from_loop


def _sum_loop() -> Loop:
    """A minimal annotated reduce ``Loop`` — ``acc += x[m, k]`` over ``k``, the way recognition
    stamps it (the role its ONLY annotation; the algebra is the body's fold ``Accum``, the
    ``axes`` stamp included — the canonical dissolved shape ``from_loop``'s byte gate
    reproduces)."""
    acc = Accum(name="acc", value="x_e", op="add", axes=("k",))
    body = Body((Load(name="x_e", input="x", index=(Var("m"), Var("k"))), acc))
    return Loop(axis=Axis("k", 1024), body=body, role=AxisRole.PLANAR)


def test_from_loop_reconstructs_the_loop_exactly() -> None:
    loop = _sum_loop()
    red = fold_from_loop(loop)
    assert red is not None
    # The synthesized loop is byte-identical to the captured one (axis / role / body).
    assert red.loop == loop
    assert reduce_loop(red) == loop
    assert axis_role(red) is AxisRole.PLANAR


def test_bare_reduction_lowers_to_just_the_loop() -> None:
    loop = _sum_loop()
    red = fold_from_loop(loop)
    assert red is not None
    assert red.lower() == [loop]
    # A bare reduce's grid ``Write`` is glue — ``out`` is the carried state's primary component.
    assert red.out == "acc"


def test_projected_reduce_is_a_map_over_the_reduction() -> None:
    loop = _sum_loop()
    proj = (Assign(name="rms", op="sqrt", args=("acc",)),)
    node = Fold.projection(body=Body(proj), operands=(fold_from_loop(loop),))
    # lower flattens the source's loop then the projection body — the bare-loop ``Map`` body.
    assert node.lower() == [loop, *proj]
    assert node.out == "rms"  # the projection's last def
    # The structural reads see straight through to the source's reduce.
    assert reduce_loop(node) == loop
    assert axis_role(node) is AxisRole.PLANAR


def test_map_over_reduction_matches_the_legacy_loop_in_body_form() -> None:
    """``Fold.projection(source=Fold).lower()`` equals the bare-loop ``Fold.projection(body=(loop, *proj))`` — the parity
    guarantee that keeps ``Op.cache_key`` stable across the lift."""
    loop = _sum_loop()
    proj = (Write(output="out", index=(Var("m"),), value="acc"),)
    node = Fold.projection(body=Body(proj), operands=(fold_from_loop(loop),))
    legacy = Fold.projection(body=(loop, *proj))
    assert node.lower() == legacy.lower()
    assert reduce_loop(node) == reduce_loop(legacy)
    assert axis_role(node) == axis_role(legacy)


def test_pure_pointwise_map_has_no_reduce() -> None:
    node = Fold.projection(body=(Load(name="x_e", input="x", index=(Var("m"),)), Assign(name="y", op="relu", args=("x_e",))))
    assert node.operands == ()
    assert reduce_loop(node) is None
    assert axis_role(node) is AxisRole.FREE
    assert node.out == "y"


def _tile(op) -> TileOp:
    """An unmapped :class:`TileOp` wrapping ``op`` — the reduce partition rides ``op``'s
    :class:`Fold` node (there is no residual ``TileOp.reduce`` field)."""
    return TileOp(op=op)


def test_reduce_plan_reads_the_partition_off_the_schedule_dict() -> None:
    from emmy.compiler.ir.tile.ops import sched_of

    plan = ReducePlan.of(coop=128)
    red = fold_from_loop(_sum_loop())
    assert red is not None
    # A bare reduce root and a projecting Map both surface the partition keyed on the fold.
    bare = _tile(red)
    sched_of(bare).put("REDUCE", red, plan)
    assert reduce_plan(bare) is plan
    wrapped = _tile(Fold.projection(body=Body((Assign(name="rms", op="sqrt", args=("acc",)),)), operands=(red,)))
    sched_of(wrapped).put("REDUCE", red, plan)
    assert reduce_plan(wrapped) is plan


def test_reduce_plan_is_none_for_a_flat_map_without_a_reduction_node() -> None:
    """A flat ``Map`` (pointwise, or a scalar per-cell contraction holding a loop-in-body with no
    ``Fold`` source) has NO partition — ``reduce_plan`` reads ``None``, not a residual field.
    Every partitioned reduce is nodified (``nodify_reduce``), so the fallback is gone."""
    legacy = Fold.projection(body=(_sum_loop(),))  # loop in the body, no Fold source
    assert reduce_plan(_tile(legacy)) is None
    pointwise = Fold.projection(body=(Load(name="x_e", input="x", index=(Var("m"),)),))
    assert reduce_plan(_tile(pointwise)) is None


def test_twisted_role_derives_from_the_combine_and_propagates() -> None:
    """The PLANAR/TWISTED half of the role is the stored combine's twist family — a fold whose
    body is the dissolved exp-family (online-softmax) merge derives ``TWISTED``; no stored role
    field, no side-band algebra (``from_loop`` reconstructs it from the body alone)."""
    from emmy.compiler.ir.stmt.carrier import exp_merge

    loop = Loop(
        axis=Axis("k", 1024),
        body=Body((Load(name="x_e", input="x", index=(Var("m"), Var("k"))), *exp_merge(("m_i", "l_i"), ("x_e", 1.0), key="m_i"))),
        role=AxisRole.TWISTED,
    )
    red = fold_from_loop(loop)
    assert red.role is AxisRole.TWISTED
    assert red.loop == loop  # the derivation reproduces the recognizer's annotation exactly
    assert axis_role(red) is AxisRole.TWISTED
    assert axis_role(Fold.projection(body=Body(()), operands=(red,))) is AxisRole.TWISTED


def _contraction() -> Fold:
    """A minimal tiled contraction node — ``acc = Σ_k A[m, k]·B[k, n]`` over a scalar tile."""
    a = Load(name="a_e", input="A", index=(Var("m"), Var("k")))
    b = Load(name="b_e", input="B", index=(Var("k"), Var("n")))
    return Fold.contraction(
        k_axis=Axis("k", 256),
        a=a,
        channels=(Channel(b=b, acc="acc"),),
    )


def test_contraction_synthesizes_the_mul_add_loop() -> None:
    c = _contraction()
    loop = c.loop
    assert loop.role is AxisRole.CONTRACTION
    assert loop.axis == c.axis
    # The derived contraction loop: B, A loads + the ⊗ lift + the additive fold.
    assert isinstance(loop.body[-1], Accum) and loop.body[-1].name == "acc"
    assert isinstance(loop.body[-2], Assign) and loop.body[-2].op.name == "multiply"


def test_contraction_dispatches_through_ops() -> None:
    c = _contraction()
    assert axis_role(c) is AxisRole.CONTRACTION
    assert reduce_loop(c) == c.loop
    assert c.lower() == [c.loop]  # bare: just the synthesized loop (the grid Write is materialize glue)
    assert c.out == "acc"


def test_a_projection_rides_the_map_wrapper_not_the_node() -> None:
    """ONE home for a projection: the wrapping ``Map``'s body. A bilinear ``Fold`` lowers to just its
    synthesized loop, and the projected form is the same ``project ∘ contract`` spelling the
    ``Fold`` tiers use — so the future cut realizer sees a single seam shape."""
    proj = (Assign(name="y", op="relu", args=("acc",)), Write(output="out", index=(Var("m"), Var("n")), value="y"))
    c = _contraction()
    node = Fold.projection(body=Body(proj), operands=(c,))  # the STORED form under the wrapper
    assert c.lower() == [c.loop]
    assert node.lower() == [c.loop, *proj]
    assert reduce_loop(node).role is AxisRole.CONTRACTION  # the projection doesn't hide the contraction


# --- nodify_reduce: the coop-K / split partial flat-Map → Fold node lift ------------------ #


def test_nodify_reduce_lifts_a_bare_loop_in_body_map_to_a_reduction() -> None:
    """A flat ``Map`` holding just the annotated reduce loop nodifies to a **bare** ``Fold`` node
    carrying the partition — ``lower`` byte-identical, ``reduce_plan`` reading the node."""
    from emmy.compiler.ir.tile.ops import sched_of
    from emmy.compiler.pipeline.passes.lowering.tile._fromloop import nodify_reduce

    loop = _sum_loop()
    flat = Fold.projection(body=(loop,))
    plan = ReducePlan.of(coop=4)
    node, fold = nodify_reduce(flat)
    assert isinstance(node, Fold) and fold is node
    assert node.lower() == flat.lower()  # bit-identical lowering
    t = _tile(node)
    sched_of(t).put("REDUCE", fold, plan)
    assert reduce_plan(t) is plan


def test_nodify_reduce_keeps_a_projection_tail_as_a_wrapping_map() -> None:
    """A fused-epilogue reduce (loop then projection) nodifies to ``Fold.projection(body=proj,
    source=Fold)`` — the tail rides the wrapping ``Map``, the partition the ``Fold``."""
    from emmy.compiler.pipeline.passes.lowering.tile._fromloop import nodify_reduce

    loop = _sum_loop()
    proj = (Assign(name="y", op="relu", args=("acc",)), Write(output="out", index=(Var("m"),), value="y"))
    flat = Fold.projection(body=(loop, *proj))
    node, fold = nodify_reduce(flat)
    assert (isinstance(node, Fold) and node.axis is None) and node.operands[0] is fold and isinstance(fold, Fold)
    assert tuple(node.body) == proj
    assert node.lower() == flat.lower()  # bit-identical
    assert axis_role(node) is AxisRole.PLANAR


# --- split-K: Fold ⊃ bilinear fold (E1) --------------------------------------------------- #


def test_splitk_reduction_over_contraction_is_no_double_reduce() -> None:
    """Split-K is the identity-lift composition ``Fold(axis=ksplit, operands=(Fold(k_axis=kslice),))``:
    the outer additive reduce sums partials across CTAs, the inner contraction folds its slice —
    the outer's λ spelling is the identity lift over the inner's exact accumulator state (step 7;
    the sliced fold is the ONE operand edge, the derived step embeds it verbatim). ``lower`` is a
    SINGLE ``for ksplit:[for kslice: mul-add]`` with DISTINCT axis names (not ``for k:[for k:]``),
    and it still classifies as a ``CONTRACTION`` carrying the GRID (cta) partition."""
    from emmy.compiler.dim import Dim
    from emmy.compiler.ir.expr import BinaryExpr, Literal
    from emmy.compiler.ir.sigma import Sigma
    from emmy.compiler.ir.stmt import Lambda
    from emmy.compiler.ir.stmt.algebra import M

    c = _contraction()  # k_axis = k(256)
    # The split-K factoring, spelled inline: ksplit (the partition index, a distinct name) x kslice
    # (the per-partition chunk, the ORIGINAL name), and the sigma reconstructing the absolute index.
    ksplit = Axis(name=f"{c.axis.name}_ks", extent=Dim(2))
    kslice = replace(c.axis, extent=Dim(128))
    sigma = Sigma({c.axis.name: BinaryExpr("+", BinaryExpr("*", Var(ksplit.name), Literal(128, "int")), Var(c.axis.name))})
    inner = Fold.contraction(
        k_axis=kslice,
        a=replace(c.a, index=tuple(sigma.apply(e) for e in c.a.index)),
        channels=(replace(c.channels[0], b=replace(c.b, index=tuple(sigma.apply(e) for e in c.b.index))),),
    )
    from emmy.compiler.ir.tile.ops import sched_of

    accs = inner.defines()
    init, combine = M(*(["add"] * len(accs)), names=accs)
    red = Fold(
        axis=ksplit,
        operands=(inner,),  # the sliced NODE rides the ONE operand edge — the identity-lift composition
        lift=Lambda(params=(ksplit.name, *accs), body=Body(()), results=accs),
        init=init,
        combine=combine,
    )

    # The outer fold is an ordinary additive reduce — it tiles nothing and has no operand pair, so
    # it derives PLANAR like any other. The reassociation it carries is a STRUCTURAL probe
    # (``Fold.composed``, what ``030_split_reduce`` consumes), never a role.
    assert axis_role(red) is AxisRole.PLANAR
    assert red.composed is inner
    t = _tile(red)
    sched_of(t).put("REDUCE", red, ReducePlan.of(cta=2, finalize="atomic"))
    assert reduce_plan(t).cta == 2
    lo = red.lower()
    assert len(lo) == 1 and isinstance(lo[0], Loop) and lo[0].axis.name == "k_ks"
    inner_loops = [s for s in lo[0].body if isinstance(s, Loop)]
    assert len(inner_loops) == 1 and inner_loops[0].axis.name == "k"  # distinct from ksplit — no double-reduce
    assert isinstance(inner_loops[0].body[-1], Accum) and inner_loops[0].body[-1].name == "acc"


# --- flash: a reduce partial composing a nested PV bilinear fold (tensor-core-flash seam) --------- #


def test_reduce_partial_flattens_a_nested_pv_contraction() -> None:
    """Flash composes TWO contractions — QK as the hoisted score OPERAND edge and PV **synthesized
    into the derived blocked evaluation** (step 7: the composed ``step`` dissolved). The QK loop
    lands at the derived head, the lift body follows, and the synthesized PV ``Fold`` (a ``Stmt``)
    flattens to its own loop in place — one recursion rule, so the scalar tier expands
    ``for kv:[QK loop; P; PV loop; fold]``. This is the structural seam warp-flash rides."""
    from emmy.compiler.ir.stmt import Lambda
    from emmy.compiler.ir.stmt.carrier import exp_combine_states

    qk = _contraction()  # Σ_k A·B -> acc (the score S)
    prob = Assign(name="p", op="exp", args=("acc",))  # the lift body — between QK and the merge
    names = ("m_i", "l_i", "O_i")
    other = tuple(f"{n}__o" for n in names)
    red = Fold(
        axis=Axis("kv", 128),
        operands=(qk, Load(name="v_e", input="V", index=(Var("kv"), Var("d")))),
        lift=Lambda(params=("kv", "acc", "v_e"), body=Body((prob,)), results=("p", 1.0, "v_e")),
        init=(float("-inf"), 0.0, 0.0),
        combine=Lambda(params=names + other, body=Body(exp_combine_states(names, other)), results=names),
    )
    assert red.role is AxisRole.TWISTED

    (kv_loop,) = red.lower()
    assert kv_loop.axis.name == "kv"
    body = list(kv_loop.body)
    assert not any(isinstance(s, Fold) for s in body), "the nested PV fold must be flattened, not left raw"
    (qk_loop,) = [s for s in body if isinstance(s, Loop) and s.axis.name == "k"]
    (pv_loop,) = [s for s in body if isinstance(s, Loop) and s.axis.name == "pj"]
    o_fold = next(s for s in body if isinstance(s, Accum) and s.name == "O_i")
    # QK (the hoisted edge) first, then the pre-PV probability, then the flattened PV loop, then the state fold.
    assert body.index(qk_loop) < body.index(prob) < body.index(pv_loop) < body.index(o_fold)
    assert pv_loop.role is AxisRole.CONTRACTION and isinstance(pv_loop.body[-1], Accum) and pv_loop.body[-1].name == "O_i__pv"


def test_flash_op_is_a_two_contraction_tree() -> None:
    """``_flash_op`` builds the blocked two-bilinear ``Fold`` tree: BOTH contractions ride the single
    walked edge — ``step`` — with QK (``Σ_dd Q·K`` score) at its head and PV — a
    **register-resident-A** contraction (``A = P``, the exp weight, a one-stmt cone INLINE on the
    edge) — spliced later (block=1: a singleton ``pj`` reduce). There is no second composition
    edge: a composed reduce spells it ONE way, so the walk reaches both QK and PV as nodes on
    ``step``. Its A is computed, not a gmem load; its O-fold consumes the PV output ``O_i__pv``,
    so the ⊗ is a contraction node, not an inline FMA."""
    from emmy.compiler.dim import Dim
    from emmy.compiler.pipeline.passes.lowering.tile._flash import _flash_op

    op, _stores = _flash_op("Q", "K", "V", [1, 2], Dim(16), Dim(16), 8, 8)  # (batch, s_q, s_k, head_dim, d_v)
    (red,) = op.operands  # the streaming Fold under the O/l projection Map
    assert red.role is AxisRole.TWISTED  # derived off the exp-family flash carrier
    folds = [s for s in red.step_stmts() if isinstance(s, Fold) and s.role is AxisRole.CONTRACTION]
    assert len(folds) == 2 and folds[0].out == "sacc", "the QK score fold is the derived evaluation's head node"
    assert folds[0] is red.operands[0], "the QK score is the hoisted operand edge (step 7)"
    # Algebra reads come straight off the stored node — placement is the ``Placed`` view's job.
    pv = folds[1]
    assert (not isinstance(pv.a, Load)) and operand_name(pv.a) == "O_i__p", "PV's A operand is the inline exp weight P"
    assert pv.a.out == "O_i__p"
    assert pv.acc == "O_i__pv"
    # A is the ONE-STMT node itself, not an identity ``Map`` wrapping a prologue in the cone shape:
    # the copy is the reference (an edge is a Load or an inline node — no name-reference arm), and
    # with an empty cell the wrapper carried nothing (``cone_seam`` bridges no stats either way).
    assert not pv.a.operands, "PV's A is the one-stmt cone itself — no empty-cell cone wrapper"
    assert [s.pretty()[0].strip() for s in operand_body(pv.a)] == ["O_i__p = copy(m_i__t5)"]
    assert cone_seam(pv.a) == ((), tuple(pv.a.body), ()), "an empty-cell wrapper bridged no stats — the seam is the node"
    # The reduce loop flattens BOTH folds; the O-fold reads the PV output (no inline v·P).
    (kv_loop,) = red.lower()
    o_fold = next(s for s in kv_loop.body if isinstance(s, Accum) and s.name == "O_i")
    assert o_fold.value == "O_i__pv", "the O-fold consumes the PV contraction's output, not an inline product"


def test_out_store_index_reproduces_output_layout() -> None:
    """``_out_store_index`` maps the fragment's grid axes onto the ROOT output buffer's real slots
    (by dim extent), keeping ``Literal`` slots (size-1 batch / broadcast dims). The bare grid order
    would mis-stride a higher-rank / transposed output → the Gemma model-trace flash NaN."""
    from emmy.compiler.dim import Dim
    from emmy.compiler.ir.expr import Literal
    from emmy.compiler.pipeline.passes.lowering.tile._flash import _out_store_index

    grid = (Axis("b0", Dim(1)), Axis("b1", Dim(16)), Axis("m", Dim(32)), Axis("d", Dim(256)))

    # Gemma's ``[b, h, s, 1, d]`` — a unit broadcast dim between seq and head_dim (Literal slots kept).
    out5 = (Literal(0, "int"), Var("h"), Var("s"), Literal(0, "int"), Var("hd"))
    idx = _out_store_index(out5, (Dim(1), Dim(16), Dim(32), Dim(1), Dim(256)), grid)
    assert idx == (Literal(0, "int"), Var("b1"), Var("m"), Literal(0, "int"), Var("d"))

    # A transposed ``[b, s, h, d]`` output — extent-match routes seq→m, head→b1 to their real slots.
    outT = (Literal(0, "int"), Var("s"), Var("h"), Var("hd"))
    idxT = _out_store_index(outT, (Dim(1), Dim(32), Dim(16), Dim(256)), grid)
    assert idxT == (Literal(0, "int"), Var("m"), Var("b1"), Var("d"))

    # No grid axis for a Var slot's extent → decline (the caller degrades flash to cut).
    assert _out_store_index((Var("x"),), (Dim(999),), grid) is None


# --- computed (register-resident) A operand: the tensor-core-flash PV crux ---------------------- #


def _pv_contraction() -> Fold:
    """A PV-style contraction whose **A operand is computed**, not a gmem ``Load``:
    ``O[m, d] = Σ_j P[m, j]·V[j, d]`` with ``P = exp(S[m, j])`` produced from an in-register score
    (the flash PV shape — its A is register-resident, so the operand edge holds the cone NODE
    inline)."""
    cone = Fold.projection(body=Body((Load(name="s_e", input="S", index=(Var("m"), Var("j"))), Assign(name="p", op="exp", args=("s_e",)))))
    return Fold.contraction(
        k_axis=Axis("j", 8),
        a=cone,
        channels=(Channel(b=Load(name="v_e", input="V", index=(Var("j"), Var("d"))), acc="oblk"),),
    )


def test_contraction_computed_a_operand_exposes_its_body() -> None:
    c = _pv_contraction()
    assert (not isinstance(c.a, Load)) and operand_name(c.a) == "p"
    assert isinstance(operand_body(c.a)[0], Load) and operand_body(c.a)[0].input == "S" and operand_body(c.a)[-1].op.name == "exp"
    # external reads are the A body's LOADED buffers + B — the computed ``p`` is an internal temp, not a read.
    assert set(c.external_reads()) == {"S", "V"}


def test_contraction_computed_a_lowers_into_the_k_loop() -> None:
    """The computed A body is spliced into the synthesized ``CONTRACTION`` loop AHEAD of the ⊗ multiply:
    ``for j: s_e = S[m, j]; p = exp(s_e); v_e = V[j, d]; oblk__v = v_e·p; oblk += oblk__v`` — the
    register-resident P produced per K-step, then multiplied by V and folded. Same builder a gmem-A
    contraction uses; the operand is just a body, not a leaf load."""
    loop = _pv_contraction().loop
    assert loop.role is AxisRole.CONTRACTION and loop.axis.name == "j"
    body = list(loop.body)
    exp_i = next(i for i, s in enumerate(body) if isinstance(s, Assign) and s.op.name == "exp")
    mul_i = next(i for i, s in enumerate(body) if isinstance(s, Assign) and s.op.name == "multiply")
    acc_i = next(i for i, s in enumerate(body) if isinstance(s, Accum))
    assert exp_i < mul_i < acc_i  # P computed, then P·V, then the additive fold
    assert "p" in body[mul_i].args  # the ⊗ multiplies the register-resident P, no gmem A load


def test_contraction_computed_a_factorizes_at_the_scalar_tier() -> None:
    """The scalar tier expands a computed-A contraction with **no gmem A address**: the register-tile
    replication treats ``P = exp(S)`` as ordinary K-loop body, so the emitted kernel carries the ``exp``
    inside the reduce loop feeding the accumulator. This is the standalone P@V the tensor-core-flash
    rebuild rests on (proved at the scalar tier; the mma tier reads the same operand as a fragment)."""
    from emmy.compiler.pipeline.passes.lowering.kernel._factor import factorize

    tile = factorize(TileOp(op=_pv_contraction()), root=None)
    exps = [s for s in tile.body.iter_of_type(Assign) if s.op.name == "exp"]
    assert exps, "the computed A operand (exp of the score) must survive into the scalar kernel body"


# --- composed steps: every structural node is a Stmt, so a body can hold one ---------------------- #


def test_there_is_exactly_one_stored_node_kind_and_it_is_a_stmt() -> None:
    """A composed step occupies a STATEMENT position in another node's body — flash's Q@K and P@V
    contractions in a reduce partial, split-K's sliced node — and ``Stmt``-hood is what makes that
    legal. After the collapse there is ONE stored kind to check: ``Map`` and bilinear ``Fold`` are
    derived READINGS (constructors returning a ``Fold``, ``isinstance`` answering the reading), so
    everything a term can hold is a ``Fold``."""
    from emmy.compiler.ir.stmt.base import Stmt

    assert issubclass(Fold, Stmt)
    for built in (_contraction(), Fold.projection(body=Body(()), operands=()), fold_from_loop(_sum_loop())):
        assert type(built) is Fold


def test_a_generic_body_walk_reaches_a_composed_nodes_children() -> None:
    """The protocol that matters: ``nested()`` exposes a composed node's body, so the generic deep
    walks (``deep_defines`` and friends) descend into it instead of stopping at — or crashing on —
    the node. This is the shape ``030_split_reduce`` produces: a ``Map`` group inside a partial."""
    from emmy.compiler.ir.tile.ir import deep_defines

    inner = Assign(name="g", op="copy", args=("x",))
    group = Fold.projection(body=Body((inner,)))
    assert group.nested() == (Body((inner,)),)
    assert "g" in deep_defines(Loop(axis=Axis("k", 8), body=Body((group,))))


def test_a_composed_step_keeps_its_position_when_flattened() -> None:
    """Position in the sequence is semantic — flash's P@V reads the softmax weight the merge stmts of
    that same loop step produce, so a composed step may not be hoisted ahead of them. ``_flatten_nodes``
    expands each node in place."""
    from emmy.compiler.ir.tile.ir import _flatten_nodes

    before, after = Assign(name="m", op="copy", args=("s",)), Assign(name="o", op="copy", args=("p",))
    flat = _flatten_nodes(Body((before, _pv_contraction(), after)))
    assert flat[0] is before and flat[-1] is after
    assert any(isinstance(s, Loop) and s.role is AxisRole.CONTRACTION for s in flat[1:-1])


# --- the B operand edge: same type as A, asymmetric only in the schedule ------------------------- #


def _computed_b_contraction(a_load: bool = True) -> Fold:
    """``O[m, n] = Σ_k A[m, k]·B'[k, n]`` where **B is computed**, not a gmem ``Load``:
    ``B' = exp(W[k, n])`` — the shape a fused per-column prologue takes (qk-norm / RoPE folded into a
    score, an on-the-fly dequant). The mirror of ``_pv_contraction``'s computed A."""
    cone = Fold.projection(body=Body((Load(name="w_e", input="W", index=(Var("k"), Var("n"))), Assign(name="wn", op="exp", args=("w_e",)))))
    return Fold.contraction(
        k_axis=Axis("k", 8),
        a=Load(name="a_e", input="A", index=(Var("m"), Var("k"))) if a_load else cone,
        channels=(Channel(b=cone, acc="o"),),
    )


def test_both_operand_edges_have_the_same_type() -> None:
    """A and B are ONE vocabulary: the algebra is symmetric in them, and the asymmetry that is real
    (A is M-resident and compute-fillable, B is the K×N operand the loop streams) is a SCHEDULE fact
    that lives in the tier gates, not in the structural type. After the collapse they are literally
    the same slot — entries in one ``operands`` tuple — and the A/B split is the stored ORDER,
    ``(b, a, b₁…)``, read back by the bilinear reading."""
    c = _contraction()
    assert c.operands == (c.b, c.a)
    assert c.a is c.operands[1] and c.channels[0].b is c.operands[0]


def test_computed_b_exposes_the_same_accessors_as_a() -> None:
    c = _computed_b_contraction()
    assert not isinstance(c.b, Load) and isinstance(c.a, Load)
    assert operand_name(c.b) == "wn"
    assert isinstance(operand_body(c.b)[0], Load) and operand_body(c.b)[-1].op.name == "exp"
    # Both edges' loaded buffers are external reads; the computed ``wn`` is an internal temp.
    assert set(c.external_reads()) == {"A", "W"}


def test_a_computed_b_has_no_gmem_layout() -> None:
    """``b_trans`` asks a gmem LAYOUT question, so it is meaningful only for a materialized B. Every
    tier that would act on the layout gates on ``isinstance(c.b, Load)`` first."""
    assert _computed_b_contraction().b_trans is False


def test_computed_b_lowers_into_the_k_loop() -> None:
    """The computed B body is spliced into the synthesized ``CONTRACTION`` loop ahead of the ⊗
    multiply, exactly as a computed A's is — the same derived loop, no B-specific
    path: ``for k: w_e = W[k, n]; wn = exp(w_e); a_e = A[m, k]; o__v = wn·a_e; o += o__v``."""
    loop = _computed_b_contraction().loop
    assert loop.role is AxisRole.CONTRACTION and loop.axis.name == "k"
    body = list(loop.body)
    exp_i = next(i for i, s in enumerate(body) if isinstance(s, Assign) and s.op.name == "exp")
    mul_i = next(i for i, s in enumerate(body) if isinstance(s, Assign) and s.op.name == "multiply")
    acc_i = next(i for i, s in enumerate(body) if isinstance(s, Accum))
    assert exp_i < mul_i < acc_i
    assert "wn" in body[mul_i].args  # the ⊗ multiplies the computed B, no gmem B load at the cell


def test_computed_b_factorizes_at_the_scalar_tier() -> None:
    """The widening is not dead vocabulary: the gmem-direct scalar tier genuinely executes a computed
    B, through the same register-tile replication a computed A rides. The staged / mma tiers decline
    (they need B's gmem address) — that decline is the schedule's, not the type's."""
    from emmy.compiler.pipeline.passes.lowering.kernel._factor import factorize

    tile = factorize(TileOp(op=_computed_b_contraction()), root=None)
    exps = [s for s in tile.body.iter_of_type(Assign) if s.op.name == "exp"]
    assert exps, "the computed B operand (exp of the weight) must survive into the scalar kernel body"


def test_both_edges_may_be_computed_at_once() -> None:
    """Nothing privileges one side: a contraction of two computed operands lowers too."""
    c = _computed_b_contraction(a_load=False)
    assert (not isinstance(c.a, Load)) and not isinstance(c.b, Load)
    assert any(isinstance(s, Accum) for s in c.loop.body)


def test_an_inline_b_edge_is_walked_like_any_other_node() -> None:
    """``path.sites`` descends every operand edge of the STORED fold — the one node walk in the
    layer, shared by the resolver, the stampers and the seam enumerator."""
    from emmy.compiler.ir.tile.path import sites

    c = _computed_b_contraction()
    assert c.b in [s.node for s in sites(c)]


# --- the ONE worker inventory (1r): TileOp.work derived from the TILE slices ---------------------- #


def test_workers_derive_from_tile_slices_and_disagreement_is_loud() -> None:
    """``derive_workers`` folds each TILE value's embedded worker geometry into the one
    kernel-global slot; two sites disagreeing on it is unrepresentable — the assembly FAILS LOUDLY
    (today an inconsistent ``TILE@dd``/``TILE@pj`` pin pair could only miss rows silently)."""
    import pytest

    from emmy.compiler.ir.schedule import Workers, derive_workers

    warp = TilePlan.parse("mma_m16n8k16_f16_f32/f1x2/k8", Workers.parse("w4x1"))
    assert derive_workers([warp, warp]) == Workers(kind="warp", units=(4, 1))
    assert derive_workers([TilePlan.parse("f4x8", Workers.parse("t16x8"))]) == Workers(kind="thread", units=(16, 8))
    assert derive_workers([TilePlan()]) is None  # per-cell — no inventory to factor
    with pytest.raises(ValueError, match="disagreeing worker geometry"):
        derive_workers([warp, TilePlan.parse("mma_m16n8k16_f16_f32/f1x2/k8", Workers.parse("w2x1"))])


def test_seal_workers_fills_the_slot_off_the_schedule_dict() -> None:
    from emmy.compiler.ir.schedule import Workers
    from emmy.compiler.ir.tile.ops import sched_of, seal_workers

    c = _contraction()
    t = _tile(c)
    sched_of(t).put("TILE", c, TilePlan.parse("f2", Workers.parse("t2")))
    seal_workers(t)
    assert t.work == Workers(kind="thread", units=(2, 1))
