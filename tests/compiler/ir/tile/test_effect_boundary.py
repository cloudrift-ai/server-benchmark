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
from emmy.compiler.ir.pure import Fold, Lambda, M
from emmy.compiler.ir.stmt import Accum, Assign, Body, Init, Load, Loop, Write
from emmy.compiler.ir.tile import OutputSpec, apply_output_specs, extract_output_specs, lower_with_output_specs


def _ax(name: str, n: int = 64) -> Axis:
    return Axis(name=name, extent=Dim(n))


# --- plain trailing root Write(s) ---------------------------------------------------------------- #


def test_trailing_write_splits_and_round_trips() -> None:
    stmts = [Assign(name="y", op="relu", args=("x",)), Write(output="o", index=(Var("m"),), value="y")]
    pure, stores = extract_output_specs(stmts)
    assert [type(s).__name__ for s in pure] == ["Assign"]
    assert len(stores) == 1 and stores[0].sweep is None and stores[0].write is stmts[1]
    assert apply_output_specs(pure, stores) == Body(stmts)


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
    assert apply_output_specs(pure, stores) == Body(stmts)


def test_atomic_and_vector_write_fields_survive() -> None:
    """``OutputSpec`` holds the ``Write`` whole, so atomic / multi-value spellings are lossless."""
    w = Write(output="o", index=(Var("m"),), values=("a", "b"), atomic=False)
    aw = Write(output="o", index=(Var("m"),), value="a", atomic=True)
    for write in (w, aw):
        pure, stores = extract_output_specs([write])
        assert pure == () and stores[0].write == write
        assert apply_output_specs(pure, stores) == Body((write,))


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
    assert apply_output_specs(pure, stores) == Body(stmts)


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
    assert apply_output_specs(pure, stores) == Body(stmts)


def test_sweep_membership_is_the_trailing_axis_reading_run() -> None:
    """The scalar epilogue (no ``n`` read) stays outside the reconstituted loop; everything from
    the first ``n``-reading stmt on goes inside — the trailing-run rule."""
    stmts = _sweep_shape()
    pure, stores = extract_output_specs(stmts)
    rebuilt = apply_output_specs(pure, stores)
    assert isinstance(rebuilt[0], Assign) and rebuilt[0].name == "rs"
    assert isinstance(rebuilt[1], Loop) and len(rebuilt) == 2


def test_sibling_output_loops_become_sweep_specs() -> None:
    """Different local output extents are sibling sweeps: one spec per write, each over its own
    axis, reconstituting the loops in order. Their per-cell projections are not the boundary's —
    the lift forms them as terms declaring the sweep axis before extraction sees the stream."""
    q, kv = _ax("q", 4), _ax("kv", 2)
    stmts = [
        Loop(axis=q, body=Body((Write(output="q_out", index=(Var("m"), Var("q")), value="qv"),))),
        Loop(
            axis=kv,
            body=Body(
                (
                    Write(output="k_out", index=(Var("m"), Var("kv")), value="kvv"),
                    Write(output="v_out", index=(Var("m"), Var("kv")), value="kvv"),
                )
            ),
        ),
    ]

    pure, specs = extract_output_specs(stmts)

    assert pure == ()
    assert [spec.sweep.name for spec in specs] == ["q", "kv", "kv"]
    assert apply_output_specs(pure, specs) == Body(stmts)


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

    st = OutputSpec(write=Write(output="o", index=(Var("m"),), value="y"), sweep=_ax("n"))
    assert _eval_stmt(repr(st)) == st


# --- the gate compares under construction normalization ------------------------------------------ #


def test_sweep_write_of_a_captured_accumulator_extracts() -> None:
    """A sweep may store an enclosing scope's value unchanged — ``o[j] = acc`` broadcasting an
    already-reduced accumulator. Nothing computes over ``j``, so no term opens it: the store's
    sweep spec is what binds the axis, wrapped around the writes alone."""
    first = Loop(
        axis=_ax("a1", 8),
        body=Body(
            (
                Load(name="x", input="src", index=(Var("a1"),)),
                Assign(name="y", op="relu", args=("x",)),
                Write(output="o1", index=(Var("a1"),), value="y"),
            )
        ),
    )
    second = Loop(
        axis=_ax("a2", 4),
        body=Body((Write(output="o2", index=(Var("a2"),), value="acc"),)),
    )
    split = extract_output_specs([first, second])
    assert split is not None
    pure, stores = split
    assert [store.write.output for store in stores] == ["o1", "o2"]
    assert [type(stmt).__name__ for stmt in pure] == ["Load", "Assign"], "the first sweep's pure prefix rejoins the stream"
    assert [store.sweep.name for store in stores] == ["a1", "a2"]
    assert apply_output_specs(pure, stores) == Body((first, second))


# --- a store lands at the scope binding its index ----------------------------------------------- #


def _slab(name: str, buffer: str, *index: str, scope: tuple[Axis, ...]) -> Fold:
    return Fold.slab(Load(name=name, input=buffer, index=tuple(Var(v) for v in index)), scope)


def _sum(operands: tuple[Fold, ...], body: tuple, acc: str, k: Axis) -> Fold:
    bound = tuple(name for edge in operands for name in edge.exposes)
    init, combine = M("add", names=(acc,))
    lift = Lambda.closing((k.name, *bound), Body(body), (f"{acc}__v",))
    return Fold(axes=(k,), operands=operands, lift=lift, init=init, combine=combine)


def test_a_store_lands_at_the_scope_binding_its_index() -> None:
    """The closed program binds ``m`` and ``n`` with the term's own loops, so the store of the
    ``[m, n]`` cell rides the ``n`` loop after the reduce; at kernel scope, where the grid binds
    both, the same store is the kernel tail."""
    m, n, k = _ax("m", 8), _ax("n", 4), _ax("k", 16)
    scope = (m, n, k)
    body = (Assign(name="acc__v", op="multiply", args=("l", "r")),)
    mm = _sum((_slab("l", "x", "m", "k", scope=scope), _slab("r", "w", "k", "n", scope=scope)), body, "acc", k)
    write = Write(output="out", index=(Var("m"), Var("n")), value="acc")
    (m_loop,) = lower_with_output_specs(mm, (OutputSpec(write=write),), frozenset())
    (n_loop,) = m_loop.body
    assert m_loop.axis is m and n_loop.axis is n
    assert [type(stmt).__name__ for stmt in n_loop.body] == ["Loop", "Write"] and n_loop.body[-1] is write
    assert lower_with_output_specs(mm, (OutputSpec(write=write),)) == Body((*mm.lower(), write))


def test_a_sweep_store_rides_the_loop_the_term_opened() -> None:
    """At kernel scope the term opens its output sweep itself, and the sweep store lands inside
    that loop rather than wrapping a second one around it; the row total, evaluated over ``m``
    alone, stays ahead of the sweep."""
    m, n, k = _ax("m", 8), _ax("n", 4), _ax("k", 16)
    scope = (m, n, k)
    total = _sum((_slab("y", "y", "m", "k", scope=scope),), (Assign(name="tot__v", op="copy", args=("y",)),), "tot", k)
    body = (Assign(name="d", op="subtract", args=("x", "tot")), Assign(name="acc__v", op="exp", args=("d",)))
    swept = _sum((_slab("x", "x", "m", "n", "k", scope=scope), total), body, "acc", k)
    write = Write(output="out", index=(Var("m"), Var("n")), value="acc")
    ahead, sweep = lower_with_output_specs(swept, (OutputSpec(write=write, sweep=n),))
    assert ahead.axis is k and sweep.axis is n
    assert [type(stmt).__name__ for stmt in sweep.body] == ["Loop", "Write"]
    assert sweep.body.axis_names == {"k"}, "the sweep loop the term opened is the one the store rides"


def test_a_store_lands_in_the_sibling_loop_binding_its_axis() -> None:
    """Two sweeps no term shares are sibling loops; a store descends the one whose axis its index
    reads, wherever it sits in the stream, rather than the trailing loop."""
    q, kv = _ax("q", 4), _ax("kv", 2)
    q_loop = Loop(axis=q, body=Body((Assign(name="qv", op="copy", args=("acc",)),)))
    kv_loop = Loop(axis=kv, body=Body((Assign(name="kvv", op="copy", args=("acc",)),)))
    write = Write(output="q_out", index=(Var("m"), Var("q")), value="qv")
    rebuilt = apply_output_specs([q_loop, kv_loop], (OutputSpec(write=write, sweep=q),))
    assert rebuilt == Body((Loop(axis=q, body=Body((*q_loop.body, write))), kv_loop))
