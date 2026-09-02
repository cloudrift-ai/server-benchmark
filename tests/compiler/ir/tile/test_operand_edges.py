"""Operand edges — every edge is a TERM, and a term declares what a reader used to walk for.

A ``Fold``'s operands are ``Fold``s, without exception: a gmem read is a :meth:`Fold.slab`, a
wrapped ``Load`` whose index coordinates are its own :attr:`Fold.axes`. That one invariant is what
lets every per-edge question be an attribute rather than a helper branching on what the edge
happens to be — its coordinates (:attr:`Fold.index_space`), the names it binds
(:attr:`Fold.exposes`), whether it is a leaf (:attr:`Fold.is_slab`), and whether the term above it
is bilinear (:meth:`Fold.as_contraction`).

These pin the readings and the ONE lowering spelling built on them: operands lower before the body,
an operand that does not index the fold's axis lowers once ahead of the loop, and a combine folds
the step into an ``Accum`` per carried component.
"""

from __future__ import annotations

from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.pure import Fold, Lambda, M
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop

M_AXIS, N_AXIS, K_AXIS = Axis("m", Dim(256)), Axis("n", Dim(256)), Axis("k", Dim(256))
SCOPE = (M_AXIS, N_AXIS, K_AXIS)


def _slab(name: str, buffer: str, index: tuple) -> Fold:
    """One gmem read, as a term declaring the coordinates it indexes."""
    return Fold.slab(Load(name=name, input=buffer, index=index), SCOPE)


def _projection(operands: tuple, body: tuple, results: tuple) -> Fold:
    """A zero-axis term — the pointwise cell. Its lift binds one param per operand result."""
    bound = tuple(name for edge in operands for name in edge.exposes)
    return Fold(operands=operands, lift=Lambda.closing(bound, Body(body), results))


def _reduce(operands: tuple, body: tuple, accs: tuple[str, ...]) -> Fold:
    """A reducing term over ``k`` — one ``⊕`` component per accumulator."""
    bound = tuple(name for edge in operands for name in edge.exposes)
    init, combine = M(*([ElementwiseImpl("add")] * len(accs)), names=accs)
    return Fold(
        axes=(K_AXIS,),
        operands=operands,
        lift=Lambda.closing((K_AXIS.name, *bound), Body(body), tuple(f"{acc}__v" for acc in accs)),
        init=init,
        combine=combine,
    )


def _matmul() -> Fold:
    a = _slab("l", "x", (Var("m"), Var("k")))
    b = _slab("r", "w", (Var("k"), Var("n")))
    return _reduce((a, b), (Assign(name="acc__v", op="multiply", args=("l", "r")),), ("acc",))


# --- a slab declares what a walk used to discover ------------------------------------------------ #


def test_a_slab_declares_the_coordinates_it_indexes() -> None:
    """The leaf binds its own coordinates, so nothing above it has to scan an index expression."""
    slab = _slab("l", "x", (Var("m"), Var("k")))
    assert tuple(axis.name for axis in slab.axes) == ("m", "k")
    assert slab.index_space == {"m", "k"}
    assert slab.exposes == ("l",)
    assert slab.is_slab


def test_a_slab_lowers_to_exactly_its_load() -> None:
    """Wrapping is a declaration, not a layer: the emitted statements are unchanged."""
    load = Load(name="l", input="x", index=(Var("m"), Var("k")))
    assert Fold.slab(load, SCOPE).lower() == [load]


def test_a_slab_does_not_reduce() -> None:
    """``axes`` is an index space; ``combine`` is what makes an axis a REDUCTION."""
    slab = _slab("l", "x", (Var("m"), Var("k")))
    assert slab.axes and slab.axis is None and slab.role is AxisRole.FREE


def test_a_computed_cone_is_not_a_slab() -> None:
    cone = _projection(
        (_slab("e", "x", (Var("m"), Var("k"))), _slab("s", "w", (Var("k"),))),
        (Assign(name="xhat", op="multiply", args=("e", "s")),),
        ("xhat",),
    )
    assert not cone.is_slab
    assert cone.exposes == ("xhat",)
    assert cone.index_space == {"m", "k"}  # the union of its operands' declarations


# --- the bilinear reading is geometry ------------------------------------------------------------ #


def test_as_contraction_reads_the_shared_and_free_axes() -> None:
    """``a[m,k] × b[k,n]``: the shared axis is the reduction, the difference is the output."""
    view = _matmul().as_contraction
    assert view is not None
    assert view.axis is K_AXIS and {view.left, view.right} == {"m", "n"}


def test_a_scale_is_not_a_contraction() -> None:
    """An operand that brings no ``k`` makes the fold a scale, not a bilinear cell."""
    a = _slab("l", "x", (Var("m"), Var("k")))
    scale = _slab("s", "s", (Var("m"),))
    node = _reduce((a, scale), (Assign(name="acc__v", op="multiply", args=("l", "s")),), ("acc",))
    assert node.as_contraction is None
    assert node.role is AxisRole.PLANAR


def test_a_pointwise_term_has_no_view() -> None:
    """No axis to share, so nothing to read."""
    assert _projection((_slab("l", "x", (Var("m"),)),), (Assign(name="y", op="relu", args=("l",)),), ("y",)).as_contraction is None


# --- lowering ------------------------------------------------------------------------------------ #


def test_a_matmul_lowers_to_one_loop_with_both_operands_riding_the_step() -> None:
    """Both operands index ``k``, so both are re-read per step and nothing hoists."""
    (loop,) = _matmul().lower()
    assert isinstance(loop, Loop) and loop.axis is K_AXIS
    assert [stmt.input for stmt in loop.body if isinstance(stmt, Load)] == ["x", "w"]
    assert [stmt.name for stmt in loop.body if isinstance(stmt, Accum)] == ["acc"]


def test_an_operand_that_does_not_index_the_axis_lowers_once_ahead_of_the_loop() -> None:
    """The hoist is a DECLARATION compared against an axis — no body walked for free names."""
    a = _slab("l", "x", (Var("m"), Var("k")))
    scale = _slab("s", "s", (Var("m"),))
    stmts = _reduce((a, scale), (Assign(name="acc__v", op="multiply", args=("l", "s")),), ("acc",)).lower()
    hoisted, loop = stmts
    assert isinstance(hoisted, Load) and hoisted.input == "s"
    assert [stmt.input for stmt in loop.body if isinstance(stmt, Load)] == ["x"]


def test_the_combine_folds_one_accum_per_carried_component() -> None:
    """The fused gate⊗up shape: one loop, the shared A read once, an ``Accum`` per channel."""
    shared = _projection(
        (_slab("e", "x", (Var("m"), Var("k"))), _slab("sc", "w", (Var("k"),))),
        (Assign(name="xhat", op="multiply", args=("e", "sc")),),
        ("xhat",),
    )
    node = _reduce(
        (shared, _slab("bg", "Wg", (Var("k"), Var("n"))), _slab("bu", "Wu", (Var("k"), Var("n")))),
        (
            Assign(name="acc_g__v", op="multiply", args=("xhat", "bg")),
            Assign(name="acc_u__v", op="multiply", args=("xhat", "bu")),
        ),
        ("acc_g", "acc_u"),
    )
    (loop,) = node.lower()
    body = list(loop.body)
    assert sum(1 for stmt in body if isinstance(stmt, Assign) and stmt.name == "xhat") == 1
    assert [stmt.name for stmt in body if isinstance(stmt, Accum)] == ["acc_g", "acc_u"]
    assert all(stmt.op.reduce_canon == "add" for stmt in body if isinstance(stmt, Accum))


def test_a_zero_axis_term_lowers_to_its_operands_then_its_body() -> None:
    """No axis, no monoid: the step IS the answer."""
    stmts = _projection((_slab("l", "x", (Var("m"),)),), (Assign(name="y", op="relu", args=("l",)),), ("y",)).lower()
    assert [type(stmt).__name__ for stmt in stmts] == ["Load", "Assign"]


def test_every_buffer_the_term_touches_reaches_the_lowered_body() -> None:
    """A buffer dropped here is a kernel missing an argument."""

    def buffers(stmts):
        for stmt in stmts:
            if isinstance(stmt, Load):
                yield stmt.input
            for body in stmt.nested():
                yield from buffers(body)

    shared = _projection(
        (_slab("e", "x", (Var("m"), Var("k"))), _slab("sc", "w", (Var("k"),))),
        (Assign(name="xhat", op="multiply", args=("e", "sc")),),
        ("xhat",),
    )
    node = _reduce(
        (shared, _slab("bg", "Wg", (Var("k"), Var("n")))),
        (Assign(name="acc_g__v", op="multiply", args=("xhat", "bg")),),
        ("acc_g",),
    )
    assert set(buffers(node.lower())) == {"x", "w", "Wg"}


# --- closure: the predicate a placement cut asks -------------------------------------------------- #


def test_iteration_variables_are_not_captures() -> None:
    """The dominant names in any cone are induction variables, bound by the enclosing nest."""
    from emmy.compiler.pipeline.passes.lowering.tile._cut import _closed_at

    cone = _projection(
        (_slab("e", "x", (Var("m"), Var("k"))), _slab("s", "w", (Var("k"),))),
        (Assign(name="xhat", op="multiply", args=("e", "s")),),
        ("xhat",),
    )
    assert _closed_at(cone, (M_AXIS, K_AXIS)), "an ordinary cone over its own axes is closed"
    assert not _closed_at(cone, ()), "unfiltered, the axes themselves read as captures"


def test_the_closure_predicate_reads_the_declaration() -> None:
    """``_external_reads`` is :attr:`Fold.index_space` — asked of the term, not derived by lowering
    it and scanning the result."""
    from emmy.compiler.pipeline.passes.lowering.tile._cut import _external_reads

    assert _external_reads(_matmul()) == {"m", "n", "k"}


def test_a_reduce_under_an_output_sweep_lifts_to_an_operand_and_lowers_back_under_it() -> None:
    """Attention's ``Σ_k P·V`` per output column: the fold joins the projection's operands with its
    slabs declaring the sweep axis, the sweep keeps its store, and reconstitution wraps the fold's
    loop back inside the output loop. A sibling reduce ahead of the sweep keeps the sweep a sweep
    rather than a peeled free axis."""
    from emmy.compiler.ir.loop import LoopOp
    from emmy.compiler.ir.stmt import Write
    from emmy.compiler.ir.tile import lower_with_output_specs
    from emmy.compiler.pipeline.passes.lowering.tile._fromloop import lift_loop_op

    add = ElementwiseImpl("add")
    total = Loop(
        axis=K_AXIS,
        body=Body((Load(name="y0", input="y", index=(Var("m"), Var("k"))), Accum(name="tot", value="y0", op=add, axes=("k",)))),
    )
    weighted = Loop(
        axis=K_AXIS,
        body=Body(
            (
                Load(name="x0", input="x", index=(Var("m"), Var("n"), Var("k"))),
                Assign(name="d", op="subtract", args=("x0", "tot")),
                Assign(name="w", op="exp", args=("d",)),
                Accum(name="acc", value="w", op=add, axes=("k",)),
            )
        ),
    )
    sweep = Loop(axis=N_AXIS, body=Body((weighted, Write(output="out", index=(Var("m"), Var("n")), value="acc"))))
    tile = lift_loop_op(LoopOp(body=(Loop(axis=M_AXIS, body=Body((total, sweep))),)), name="k_sweep")

    # Canonical renumbering renames the axes; the sweep coordinate is one the fold's slab declares.
    # The sibling total is what the swept fold reads, so it arrives as that fold's operand — the
    # same object — and the projection keeps only the swept fold; the total lowers once, ahead.
    (swept,) = tile.op.operands
    (spec,) = tile.output_specs
    assert spec.sweep is not None and spec.sweep.name in swept.index_space
    assert any(edge.axis is not None and spec.sweep.name not in edge.index_space for edge in swept.operands)
    outer, loop = lower_with_output_specs(tile.op, tile.output_specs)
    assert isinstance(outer, Loop) and isinstance(loop, Loop) and loop.axis.name == spec.sweep.name
    assert [type(stmt).__name__ for stmt in loop.body] == ["Loop", "Write"]
