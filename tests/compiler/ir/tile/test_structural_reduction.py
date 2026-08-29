"""Fold algebra, structural readings, and lowering back to Loop IR."""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.pure import Lambda
from emmy.compiler.ir.pure.fold import Channel, Fold, operand_body, operand_name
from emmy.compiler.ir.schedule import TilePlan
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
from emmy.compiler.ir.tile import OutputSpec, ReducePlan, TileOp, apply_output_specs
from emmy.compiler.ir.tile.ops import reduce_plan
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


def test_bare_reduction_lowers_to_just_the_loop() -> None:
    loop = _sum_loop()
    red = fold_from_loop(loop)
    assert red is not None
    assert red.lower() == [loop]
    # A bare reduce's grid ``Write`` is glue — ``out`` is the carried state's primary component.
    assert red.out == "acc"


def test_projected_reduce_is_a_zero_axis_fold_over_the_reduction() -> None:
    loop = _sum_loop()
    proj = (Assign(name="rms", op="sqrt", args=("acc",)),)
    node = Fold.projection(body=Body(proj), operands=(fold_from_loop(loop),))
    # Lower flattens the source's loop, then the zero-axis projection body.
    assert node.lower() == [loop, *proj]
    assert node.out == "rms"  # the projection's last def


def test_pure_pointwise_projection_has_no_reduce() -> None:
    node = Fold.projection(body=(Load(name="x_e", input="x", index=(Var("m"),)), Assign(name="y", op="relu", args=("x_e",))))
    assert node.operands == ()
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
    # A bare reduce root and a zero-axis projection both surface the partition keyed on the fold.
    bare = _tile(red)
    sched_of(bare).put("REDUCE", red, plan)
    assert reduce_plan(bare) is plan
    wrapped = _tile(Fold.projection(body=Body((Assign(name="rms", op="sqrt", args=("acc",)),)), operands=(red,)))
    sched_of(wrapped).put("REDUCE", red, plan)
    assert reduce_plan(wrapped) is plan


def test_reduce_plan_is_none_for_a_pointwise_projection() -> None:
    pointwise = Fold.projection(body=(Load(name="x_e", input="x", index=(Var("m"),)),))
    assert reduce_plan(_tile(pointwise)) is None


def test_twisted_role_derives_from_the_combine_and_propagates() -> None:
    """The PLANAR/TWISTED half of the role is the stored combine's twist family — a fold whose
    body is the dissolved exp-family (online-softmax) merge derives ``TWISTED``; no stored role
    field, no side-band algebra (``from_loop`` reconstructs it from the body alone)."""
    from emmy.compiler.ir.pure.carrier import exp_combine_states, exp_merge

    names = ("m_i", "l_i")
    other = tuple(f"{name}__o" for name in names)
    loop = Loop(
        axis=Axis("k", 1024),
        body=Body((Load(name="x_e", input="x", index=(Var("m"), Var("k"))), *exp_merge(("m_i", "l_i"), ("x_e", 1.0), key="m_i"))),
        role=AxisRole.TWISTED,
    )
    red = Fold(
        axis=loop.axis,
        lift=Lambda(
            params=("k",),
            body=Body((Load(name="x_e", input="x", index=(Var("m"), Var("k"))),)),
            results=("x_e", 1.0),
        ),
        init=(-1e30, 0.0),
        combine=Lambda(params=names + other, body=Body(exp_combine_states(names, other)), results=names),
    )
    assert red.role is AxisRole.TWISTED
    assert red.loop == loop  # the derivation reproduces the input annotation exactly
    assert Fold.projection(body=Body(()), operands=(red,)).operands[0].role is AxisRole.TWISTED


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
    assert c.role is AxisRole.CONTRACTION
    assert c.lower() == [c.loop]  # bare: just the synthesized loop (the grid Write is materialize glue)
    assert c.out == "acc"


def test_a_projection_rides_the_zero_axis_wrapper_not_the_node() -> None:
    """ONE home for a projection: the zero-axis Fold's body. A bilinear ``Fold`` lowers to just its
    synthesized loop, and the projected form is the same ``project ∘ contract`` spelling the Fold
    tiers use."""
    proj = (Assign(name="y", op="relu", args=("acc",)),)
    write = Write(output="out", index=(Var("m"), Var("n")), value="y")
    c = _contraction()
    node = Fold.projection(body=Body(proj), operands=(c,))  # the STORED form under the wrapper
    assert c.lower() == [c.loop]
    assert node.lower() == [c.loop, *proj]
    assert apply_output_specs(node.lower(), (OutputSpec(write=write),)) == [c.loop, *proj, write]
    assert node.operands[0].role is AxisRole.CONTRACTION


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
    from emmy.compiler.ir.pure import Lambda
    from emmy.compiler.ir.pure.algebra import M
    from emmy.compiler.ir.sigma import Sigma

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
    # (``Fold.composed``, the recognized split-K composition), never a role.
    assert red.role is AxisRole.PLANAR
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
    from emmy.compiler.ir.pure import Lambda
    from emmy.compiler.ir.pure.carrier import exp_combine_states

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


# --- term identity ------------------------------------------------------------------------------- #


def test_there_is_exactly_one_stored_node_kind_and_it_is_a_term() -> None:
    """ONE stored kind — ``Map`` and the bilinear reading are derived READINGS (constructors
    returning a ``Fold``, a predicate answering the reading), so everything a term can hold is a
    ``Fold`` — and that kind is a TERM, not a ``Stmt``. A composed step (flash's Q@K ahead of its
    P@V, split-K's sliced node) reaches the emitted stream through ``operands`` and the derivation
    that orders them, never by occupying a statement position; ``lower`` is the one crossing."""
    from emmy.compiler.ir.stmt.base import Stmt

    assert not issubclass(Fold, Stmt), "a pure term must not be a Stmt (ir/ARCHITECTURE.md)"
    for built in (_contraction(), Fold.projection(body=Body(()), operands=()), fold_from_loop(_sum_loop())):
        assert type(built) is Fold


def test_a_generic_walk_reaches_a_composed_nodes_children() -> None:
    """A term is not a statement, but it still has children, and the shared deep walks
    (``deep_defines`` and friends) have to descend into them rather than stop at — or crash on —
    the node. ``nested()`` is that structural accessor; it is a term operation sharing the stmt
    vocabulary's spelling so one walker serves both."""
    from emmy.compiler.ir.pure.fold import deep_defines

    inner = Assign(name="g", op="copy", args=("x",))
    group = Fold.projection(body=Body((inner,)))
    assert group.nested() == (Body((inner,)),)
    assert "g" in deep_defines(Loop(axis=Axis("k", 8), body=Body((group,))))


def test_a_composed_step_keeps_its_position_when_flattened() -> None:
    """Position in the sequence is semantic — flash's P@V reads the softmax weight the merge stmts of
    that same loop step produce, so a composed step may not be hoisted ahead of them. ``_flatten_nodes``
    expands each node in place."""
    from emmy.compiler.ir.pure.fold import _flatten_nodes

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


def test_output_tiled_contraction_keeps_a_sibling_provider_for_its_computed_b(monkeypatch) -> None:
    """Selecting an output-tiled contraction from a projection must retain a sibling Fold whose
    result its computed B edge reads. The sibling varies over the contraction's output column, so
    it belongs inside the per-cell compute fill; treating it as a post-contraction projection
    leaves the fill reading an undefined scalar."""
    from emmy.compiler.dtype import F16
    from emmy.compiler.graph import Tensor
    from emmy.compiler.ir.kernel.ir import RegStore
    from emmy.compiler.ir.schedule import Placement, Stage, Workers
    from emmy.compiler.ir.tile.ops import sched_of
    from emmy.compiler.pipeline.passes.lowering.kernel import _factor

    m, n, k, r = Axis("m", 16), Axis("n", 32), Axis("k", 16), Axis("r", 16)
    statistic = Fold(
        axis=r,
        lift=Lambda(
            params=("r",),
            body=Body(
                (
                    Load(name="stat_in", input="W", index=(Var("n"), Var("r"))),
                    Assign(name="square", op="multiply", args=("stat_in", "stat_in")),
                )
            ),
            results=("square",),
        ),
        init=(0.0,),
        combine=Lambda(
            params=("stat", "stat__o"),
            body=Body((Assign(name="stat", op="add", args=("stat", "stat__o")),)),
            results=("stat",),
        ),
    )
    provider = Fold.projection(
        operands=(statistic,),
        body=Body(
            (
                Assign(name="norm", op="rsqrt", args=("stat",)),
                Load(name="row_bias", input="Bias", index=(Var("m"),)),
            )
        ),
        results=("norm", "row_bias"),
    )
    computed_b = Fold.projection(
        body=Body(
            (
                Load(name="weight", input="W", index=(Var("n"), Var("k"))),
                Assign(name="scaled", op="multiply", args=("weight", "norm")),
            )
        ),
        results=("scaled",),
    )
    contraction = Fold.contraction(
        k_axis=k,
        a=Load(name="activation", input="A", index=(Var("m"), Var("k"))),
        channels=(Channel(b=computed_b, acc="out"),),
    )
    root = Fold.projection(
        operands=(provider, contraction),
        body=Body((Assign(name="biased", op="add", args=("out", "row_bias")),)),
        results=("biased",),
    )
    workers = Workers.parse("w1x1")
    plan = TilePlan.parse("mma_m16n8k16_f16_f32/f1x4/k1", workers).at(m, n)
    tile = TileOp(
        op=root,
        name="out",
        place=Placement(free=(m, n), grid=(m, n), mapped=True),
        output_specs=(OutputSpec(Write(output="out", index=(Var("m"), Var("n")), value="biased")),),
    )
    tile.inputs = {
        "A": Tensor("A", (16, 16), F16),
        "Bias": Tensor("Bias", (16,), F16),
        "W": Tensor("W", (32, 16), F16),
    }
    tile.outputs = {"out": Tensor("out", (16, 32), F16)}
    sched = sched_of(tile)
    sched.put("TILE", contraction, plan)
    sched.put("STAGE", contraction, Stage(depth=1, transport="smem", smem=("scaled",), bk_elems=16))

    sliced_results = []
    provider_slice = _factor._provider_slice

    def track_slice(edge, required):
        sliced_results.append(frozenset(required))
        return provider_slice(edge, required)

    monkeypatch.setattr(_factor, "_provider_slice", track_slice)
    lowered = _factor.factorize(tile, root=None)
    stmts = tuple(lowered.body.iter())
    first_def = {name: index for index, stmt in reversed(tuple(enumerate(stmts))) for name in stmt.defines()}
    norm_reads = [(index, name) for index, stmt in enumerate(stmts) for name in stmt.deps() if name.startswith("norm")]
    assert norm_reads
    assert all(first_def.get(name, len(stmts)) < index for index, name in norm_reads)
    assert all("m" not in expr.free_vars() for stmt in stmts for expr in stmt.exprs())
    stat_loads = [stmt for stmt in stmts if isinstance(stmt, Load) and stmt.name.startswith("stat_in")]
    assert len(stat_loads) == 8, "the statistic belongs only to the eight-column computed-B fill"
    definitions = [name for stmt in stmts for name in stmt.defines()]
    assert sum(name.startswith("norm") for name in definitions) == 8
    bias_loads = [load for stmt in stmts if isinstance(stmt, RegStore) for load in stmt.epilogue.loads if load.name == "row_bias"]
    assert len(bias_loads) == 4, "the unrelated provider result belongs only to the four output fragments"
    assert sliced_results == [frozenset(("norm",)), frozenset(("row_bias",))]


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
