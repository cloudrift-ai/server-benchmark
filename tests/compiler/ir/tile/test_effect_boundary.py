"""The 1q effect boundary — ``OutputSpec`` decorations + the ``extract_output_specs`` / ``apply_output_specs``
round-trip.

Projection ``Write``\\ s (and the rms/softmax output-sweep ``Loop``) leave the root Fold lambda for
``TileOp.output_specs``; every consumer reconstitutes the effectful stmt stream via ``apply_output_specs``.
The conversion gate is byte-identity: ``extract_output_specs`` returns a spelling ONLY when
``apply_output_specs`` reproduces the captured stream exactly (the 1o construction-gate pattern), so
these tests pin the round-trip on each supported shape and the ``None`` declines.
"""

from __future__ import annotations

from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.stmt import Accum, Assign, Body, Init, Load, Loop, Write
from emmy.compiler.ir.tile import OutputSpec, ProjectionRegion, apply_output_specs, extract_output_specs


def _ax(name: str, n: int = 64) -> Axis:
    return Axis(name=name, extent=Dim(n))


# --- plain trailing root Write(s) ---------------------------------------------------------------- #


def test_trailing_write_splits_and_round_trips() -> None:
    stmts = [Assign(name="y", op="relu", args=("x",)), Write(output="o", index=(Var("m"),), value="y")]
    pure, stores = extract_output_specs(stmts)
    assert [type(s).__name__ for s in pure] == ["Assign"]
    assert len(stores) == 1 and stores[0].sweep is None and stores[0].write is stmts[1]
    assert apply_output_specs(pure, stores) == stmts


def test_trailing_write_run_keeps_order() -> None:
    """The register-strip shape: r loads, r computes, r stores — the Writes reattach in order."""
    stmts = [
        Load(name="x0", input="x", index=(Var("i"),)),
        Load(name="x1", input="x", index=(Var("i"),)),
        Assign(name="y0", op="relu", args=("x0",)),
        Assign(name="y1", op="relu", args=("x1",)),
        Write(output="o", index=(Var("i"),), value="y0"),
        Write(output="o", index=(Var("i"),), value="y1"),
    ]
    pure, stores = extract_output_specs(stmts)
    assert len(pure) == 4 and len(stores) == 2
    assert apply_output_specs(pure, stores) == stmts


def test_atomic_and_vector_write_fields_survive() -> None:
    """``OutputSpec`` holds the ``Write`` whole, so atomic / multi-value spellings are lossless."""
    w = Write(output="o", index=(Var("m"),), values=("a", "b"), atomic=False)
    aw = Write(output="o", index=(Var("m"),), value="a", atomic=True)
    for write in (w, aw):
        pure, stores = extract_output_specs([write])
        assert pure == () and stores[0].write == write
        assert apply_output_specs(pure, stores) == [write]


# --- the output-sweep Loop (rms/softmax projection) ----------------------------------------------- #


def _sweep_shape() -> list:
    """The rms_norm projection: a scalar epilogue, then the normalize sweep with the root store."""
    n = _ax("n")
    return [
        Assign(name="rs", op="rsqrt", args=("sacc",)),
        Loop(
            axis=n,
            body=Body(
                (
                    Load(name="xe", input="x", index=(Var("m"), Var("n"))),
                    Load(name="we", input="w", index=(Var("n"),)),
                    Assign(name="xhat", op="multiply", args=("xe", "rs")),
                    Write(output="o", index=(Var("m"), Var("n")), value="xhat"),
                )
            ),
        ),
    ]


def test_sweep_loop_splits_to_a_sweep_store_and_round_trips() -> None:
    stmts = _sweep_shape()
    pure, stores = extract_output_specs(stmts)
    assert [type(s).__name__ for s in pure] == ["Assign", "Load", "Load", "Assign"]
    assert len(stores) == 1 and stores[0].sweep is not None and stores[0].sweep.name == "n"
    assert apply_output_specs(pure, stores) == stmts


def test_sweep_loop_with_multiple_writes_round_trips() -> None:
    n = _ax("n")
    stmts = [
        Loop(
            axis=n,
            body=Body(
                (
                    Load(name="x", input="i", index=(Var("n"),)),
                    Write(output="o", index=(Var("n"),), value="x"),
                    Write(output="o", index=(Var("n"),), value="x"),
                )
            ),
        )
    ]

    pure, stores = extract_output_specs(stmts)

    assert len(stores) == 2 and all(store.sweep == n for store in stores)
    assert apply_output_specs(pure, stores) == stmts


def test_sweep_membership_is_the_trailing_axis_reading_run() -> None:
    """The scalar epilogue (no ``n`` read) stays outside the reconstituted loop; everything from
    the first ``n``-reading stmt on goes inside — the trailing-run rule."""
    stmts = _sweep_shape()
    pure, stores = extract_output_specs(stmts)
    rebuilt = apply_output_specs(pure, stores)
    assert isinstance(rebuilt[0], Assign) and rebuilt[0].name == "rs"
    assert isinstance(rebuilt[1], Loop) and len(rebuilt) == 2


def test_sweep_unroll_flag_survives() -> None:
    n = _ax("n")
    loop = Loop(axis=n, body=Body((Write(output="o", index=(Var("n"),), value="v"),)), unroll=True)
    pure, stores = extract_output_specs([loop])
    assert stores[0].unroll is True
    assert apply_output_specs(pure, stores) == [loop]


def test_sibling_output_loops_become_pure_projection_regions() -> None:
    """Different local output extents remain separate maps while writes stay at the boundary."""
    q, kv = _ax("q", 4), _ax("kv", 2)
    stmts = [
        Loop(
            axis=q,
            body=Body(
                (
                    Assign(name="qv", op="copy", args=("x",)),
                    Write(output="q_out", index=(Var("m"), Var("q")), value="qv"),
                )
            ),
        ),
        Loop(
            axis=kv,
            body=Body(
                (
                    Assign(name="kvv", op="copy", args=("x",)),
                    Write(output="k_out", index=(Var("m"), Var("kv")), value="kvv"),
                    Write(output="v_out", index=(Var("m"), Var("kv")), value="kvv"),
                )
            ),
        ),
    ]

    pure, specs = extract_output_specs(stmts)

    assert [member.axis.extent for member in pure if isinstance(member, ProjectionRegion)] == [Dim(4), Dim(2)]
    assert len(specs) == 3 and all(spec.sweep is None for spec in specs)
    assert all(member.pure for member in pure)
    assert apply_output_specs(pure, specs) == stmts


# --- declines (the caller keeps the computation in Loop IR) -------------------------------------- #


def test_reduce_loop_declines() -> None:
    """A reduce ``Loop`` (an ``Accum`` inside) is loop-carried state, never a sweep store."""
    k = _ax("k")
    red = Loop(axis=k, body=Body((Accum(name="acc", value="x", axes=("k",)),)))
    assert extract_output_specs([red, Write(output="o", index=(Var("m"),), value="acc")]) is None


def test_impure_residue_declines() -> None:
    """An ``Init`` seed (030's finalize shape) is not a projection stmt — no pure spelling."""
    stmts = [Init(name="acc", identity=0.0, dtype="f32"), Write(output="o", index=(Var("m"),), value="acc")]
    assert extract_output_specs(stmts) is None


def test_already_pure_stream_returns_no_stores() -> None:
    stmts = [Assign(name="y", op="relu", args=("x",))]
    pure, stores = extract_output_specs(stmts)
    assert pure == tuple(stmts) and stores == ()


def test_store_repr_round_trips_through_eval() -> None:
    """``TileOp.output_specs`` serializes as constructor reprs in graph dumps — eval must rebuild it."""
    from emmy.compiler.graph import _eval_stmt

    st = OutputSpec(write=Write(output="o", index=(Var("m"),), value="y"), sweep=_ax("n"), unroll=True)
    assert _eval_stmt(repr(st)) == st


def test_broadcast_sibling_loop_round_trips() -> None:
    """A sibling output loop may BROADCAST — write a value the enclosing scope computed, so its own
    body defines nothing and the region carries that value as a capture.

    This used to raise rather than round-trip: the capture set was read by building a lambda that
    binds the axis alone and asking what it left free, and ``Lambda`` rejects a result its body
    does not define. The capture set now comes off the body, where a broadcast result is free like
    any other name the body reads.
    """
    n = _ax("n", 16)
    stmts = [
        Assign(name="scale", op="copy", args=("x",)),
        Loop(axis=n, body=Body((Write(output="o", index=(Var("m"), Var("n")), value="scale"),))),
        Assign(name="tail", op="relu", args=("scale",)),
        Write(output="p", index=(Var("m"),), value="tail"),
    ]

    pure, specs = extract_output_specs(stmts)

    region = next(member for member in pure if isinstance(member, ProjectionRegion))
    assert region.lift.body == Body(()), "a broadcast region computes nothing of its own"
    assert "scale" in region.lift.params, "the broadcast value must arrive as a capture"
    assert all(member.pure for member in pure)
    assert apply_output_specs(pure, specs) == stmts
