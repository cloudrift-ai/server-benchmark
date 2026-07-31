"""The 1q effect boundary — ``Store`` decorations + the ``split_effects`` / ``effect_tail``
round-trip.

Projection ``Write``\\ s (and the rms/softmax output-sweep ``Loop``) leave ``Map.fn`` for
``TileOp.stores``; every consumer reconstitutes the effectful stmt stream via ``effect_tail``.
The conversion gate is byte-identity: ``split_effects`` returns a spelling ONLY when
``effect_tail`` reproduces the captured stream exactly (the 1o construction-gate pattern), so
these tests pin the round-trip on each recognized shape and the ``None`` declines.
"""

from __future__ import annotations

from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.stmt import Accum, Assign, Body, Init, Load, Loop, Write
from emmy.compiler.ir.tile import Store, effect_tail, split_effects


def _ax(name: str, n: int = 64) -> Axis:
    return Axis(name=name, extent=Dim(n))


# --- plain trailing root Write(s) ---------------------------------------------------------------- #


def test_trailing_write_splits_and_round_trips() -> None:
    stmts = [Assign(name="y", op="relu", args=("x",)), Write(output="o", index=(Var("m"),), value="y")]
    pure, stores = split_effects(stmts)
    assert [type(s).__name__ for s in pure] == ["Assign"]
    assert len(stores) == 1 and stores[0].sweep is None and stores[0].write is stmts[1]
    assert effect_tail(pure, stores) == stmts


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
    pure, stores = split_effects(stmts)
    assert len(pure) == 4 and len(stores) == 2
    assert effect_tail(pure, stores) == stmts


def test_atomic_and_vector_write_fields_survive() -> None:
    """``Store`` holds the ``Write`` whole, so atomic / multi-value spellings are lossless."""
    w = Write(output="o", index=(Var("m"),), values=("a", "b"), atomic=False)
    aw = Write(output="o", index=(Var("m"),), value="a", atomic=True)
    for write in (w, aw):
        pure, stores = split_effects([write])
        assert pure == () and stores[0].write == write
        assert effect_tail(pure, stores) == [write]


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
    pure, stores = split_effects(stmts)
    assert [type(s).__name__ for s in pure] == ["Assign", "Load", "Load", "Assign"]
    assert len(stores) == 1 and stores[0].sweep is not None and stores[0].sweep.name == "n"
    assert effect_tail(pure, stores) == stmts


def test_sweep_membership_is_the_trailing_axis_reading_run() -> None:
    """The scalar epilogue (no ``n`` read) stays outside the reconstituted loop; everything from
    the first ``n``-reading stmt on goes inside — the trailing-run rule."""
    stmts = _sweep_shape()
    pure, stores = split_effects(stmts)
    rebuilt = effect_tail(pure, stores)
    assert isinstance(rebuilt[0], Assign) and rebuilt[0].name == "rs"
    assert isinstance(rebuilt[1], Loop) and len(rebuilt) == 2


def test_sweep_unroll_flag_survives() -> None:
    n = _ax("n")
    loop = Loop(axis=n, body=Body((Write(output="o", index=(Var("n"),), value="v"),)), unroll=True)
    pure, stores = split_effects([loop])
    assert stores[0].unroll is True
    assert effect_tail(pure, stores) == [loop]


# --- declines (the caller keeps the raw-loop-IR spelling) ----------------------------------------- #


def test_reduce_loop_declines() -> None:
    """A reduce ``Loop`` (an ``Accum`` inside) is loop-carried state, never a sweep store."""
    k = _ax("k")
    red = Loop(axis=k, body=Body((Accum(name="acc", value="x", axes=("k",)),)))
    assert split_effects([red, Write(output="o", index=(Var("m"),), value="acc")]) is None


def test_impure_residue_declines() -> None:
    """An ``Init`` seed (030's finalize shape) is not a projection stmt — no pure spelling."""
    stmts = [Init(name="acc", identity=0.0, dtype="f32"), Write(output="o", index=(Var("m"),), value="acc")]
    assert split_effects(stmts) is None


def test_already_pure_stream_returns_no_stores() -> None:
    stmts = [Assign(name="y", op="relu", args=("x",))]
    pure, stores = split_effects(stmts)
    assert pure == tuple(stmts) and stores == ()


def test_store_repr_round_trips_through_eval() -> None:
    """``TileOp.stores`` serializes as constructor reprs in graph dumps — eval must rebuild it."""
    from emmy.compiler.graph import _eval_stmt

    st = Store(write=Write(output="o", index=(Var("m"),), value="y"), sweep=_ax("n"), unroll=True)
    assert _eval_stmt(repr(st)) == st
