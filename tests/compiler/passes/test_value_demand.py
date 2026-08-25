"""Symbolic repeated-compute analysis for maximally fused Loop IR values."""

from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
from emmy.compiler.pipeline.passes.lowering.tile._demand import value_demands


def _x_f8_like(n_extent=None) -> LoopOp:
    """One quantization evaluated under ``n`` plus its once-per-``(m, k)`` live output."""
    n_extent = Dim(8) if n_extent is None else n_extent
    m = Axis("m", Dim(4))
    n = Axis("n", n_extent)
    k = Axis("k", n_extent)
    contraction = Loop(
        axis=k,
        body=Body(
            (
                Load(name="x_inline", input="x", index=(Var("m"), Var("k"))),
                Assign(name="q_inline", op="to_f8e4m3", args=("x_inline",)),
                Assign(name="decoded", op="from_f8e4m3", args=("q_inline",)),
                Load(name="weight", input="w", index=(Var("n"), Var("k"))),
                Assign(name="product", op="multiply", args=("decoded", "weight")),
                Accum(name="acc", value="product"),
            )
        ),
    )
    cell = Loop(
        axis=n,
        body=Body(
            (
                contraction,
                Load(name="x_output", input="x", index=(Var("m"), Var("n"))),
                Assign(name="q_output", op="to_f8e4m3", args=("x_output",)),
                Write(output="y", index=(Var("m"), Var("n")), value="acc"),
                Write(output="x_f8", index=(Var("m"), Var("n")), value="q_output"),
            )
        ),
    )
    return LoopOp(body=Body((Loop(axis=m, body=Body((cell,))),)))


def test_consumer_only_axis_proves_repeated_quantization() -> None:
    group = next(group for group in value_demands(_x_f8_like()) if group.live_outputs == ("x_f8",))
    assert len(group.occurrences) == 2
    inline = next(occurrence for occurrence in group.occurrences if occurrence.repeated)
    output = next(occurrence for occurrence in group.occurrences if not occurrence.repeated)
    assert len(inline.coordinate_axes) == 2
    assert len(inline.repeated_axes) == 1
    assert inline.replication_lower_bound == 8
    assert not output.repeated
    assert group.repeated


def test_matching_producer_and_demand_coordinates_do_not_repeat() -> None:
    m = Axis("m", Dim(4))
    k = Axis("k", Dim(8))
    op = LoopOp(
        body=Body(
            (
                Loop(
                    axis=m,
                    body=Body(
                        (
                            Loop(
                                axis=k,
                                body=Body(
                                    (
                                        Load(name="xv", input="x", index=(Var("m"), Var("k"))),
                                        Assign(name="q", op="to_f8e4m3", args=("xv",)),
                                        Write(output="x_f8", index=(Var("m"), Var("k")), value="q"),
                                    )
                                ),
                            ),
                        )
                    ),
                ),
            )
        )
    )
    group = next(group for group in value_demands(op) if group.live_outputs == ("x_f8",))
    assert len(group.occurrences) == 1
    assert group.occurrences[0].replication_lower_bound == 1
    assert not group.repeated


def test_symbolic_consumer_axis_keeps_the_structural_proof() -> None:
    group = next(group for group in value_demands(_x_f8_like(Dim("width"))) if group.live_outputs == ("x_f8",))
    inline = next(occurrence for occurrence in group.occurrences if occurrence.repeated_axes)
    assert inline.evaluations is None
    assert inline.coordinate_upper_bound is None
    assert inline.replication_lower_bound is None
    assert inline.repeated


def test_equal_fanout_demands_canonicalize_to_one_map() -> None:
    def branch(m_name: str, k_name: str, suffix: str) -> Loop:
        m = Axis(m_name, Dim(4))
        k = Axis(k_name, Dim(8))
        return Loop(
            axis=m,
            body=Body(
                (
                    Loop(
                        axis=k,
                        body=Body(
                            (
                                Load(name=f"x{suffix}", input="x", index=(Var(m_name), Var(k_name))),
                                Assign(name=f"q{suffix}", op="to_f8e4m3", args=(f"x{suffix}",)),
                                Write(output=f"out{suffix}", index=(Var(m_name), Var(k_name)), value=f"q{suffix}"),
                            )
                        ),
                    ),
                )
            ),
        )

    groups = value_demands(LoopOp(body=Body((branch("m0", "k0", "0"), branch("m1", "k1", "1")))))
    group = next(group for group in groups if set(group.live_outputs) == {"out0", "out1"})
    assert len(group.demands) == 1
    assert len(group.demands[0].occurrences) == 2
    assert group.repeated


def test_reduction_derived_value_fails_closed() -> None:
    k = Axis("k", Dim(8))
    op = LoopOp(
        body=Body(
            (
                Loop(
                    axis=k,
                    body=Body(
                        (
                            Load(name="xv", input="x", index=(Var("k"),)),
                            Accum(name="total", value="xv"),
                        )
                    ),
                ),
                Assign(name="projected", op="exp", args=("total",)),
                Write(output="out", index=(), value="projected"),
            )
        )
    )
    assert not any(group.live_outputs for group in value_demands(op))
