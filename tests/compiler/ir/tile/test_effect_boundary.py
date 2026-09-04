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
from emmy.compiler.ir.tile import OutputSpec, apply_output_specs, extract_output_specs


def _ax(name: str, n: int = 64) -> Axis:
    return Axis(name=name, extent=Dim(n))


# --- plain trailing root Write(s) ---------------------------------------------------------------- #


def test_trailing_write_splits_and_round_trips() -> None:
    stmts = [Assign(name="y", op="relu", args=("x",)), Write(output="o", index=(Var("m"),), value="y")]
    pure, stores = extract_output_specs(stmts)
    assert [type(s).__name__ for s in pure] == ["Assign"]
    assert len(stores) == 1 and stores[0].sweep == () and stores[0].write is stmts[1]
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
    assert len(stores) == 1 and [axis.name for axis in stores[0].sweep] == ["n"]
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

    assert len(stores) == 2 and all(store.sweep == (n,) for store in stores)
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
    assert [axis.name for spec in specs for axis in spec.sweep] == ["q", "kv", "kv"]
    assert apply_output_specs(pure, specs) == Body(stmts)


# --- nested output sweeps (DeepSeek-V4 post4096's boundary shapes) -------------------------------- #


def test_nested_write_only_sweep_round_trips() -> None:
    """A write-only sweep nest with no outer store (post4096's softmax pair): the specs carry the
    axis PATH, outermost first, and reconstitution reopens both loops."""
    a10, a17 = _ax("a10", 4), _ax("a17", 4)
    stmts = [
        Loop(
            axis=a10,
            body=Body(
                (
                    Loop(
                        axis=a17,
                        body=Body(
                            (
                                Write(output="s_sum", index=(Var("m"), Var("a10"), Var("a17")), value="acc"),
                                Write(output="s_exp", index=(Var("m"), Var("a10"), Var("a17")), value="v98"),
                            )
                        ),
                    ),
                )
            ),
        )
    ]

    split = extract_output_specs(stmts)

    assert split is not None
    pure, specs = split
    assert pure == ()
    assert [[axis.name for axis in spec.sweep] for spec in specs] == [["a10", "a17"], ["a10", "a17"]]
    assert apply_output_specs(pure, specs) == Body(stmts)


def test_nested_sweeps_beside_an_outer_write_round_trip() -> None:
    """An outer sweep carrying its own store beside nested write-only sweeps (post4096's gate
    stream: view_8 beside the mul_16 and matmul_1 nests). The outer loop is opened once, by the
    last spec whose path carries it; the nested loops sit inside it, writes trailing."""
    a20, a27, a27b, a35 = _ax("a20", 4), _ax("a27", 8), _ax("a27b", 4), _ax("a35", 8)
    inner1 = Loop(axis=a27, body=Body((Write(output="mul_16", index=(Var("a20"), Var("a27")), value="v216"),)))
    inner2 = Loop(
        axis=a27b,
        body=Body((Loop(axis=a35, body=Body((Write(output="matmul_1", index=(Var("a20"), Var("a27b"), Var("a35")), value="v223"),))),)),
    )
    outer = Loop(axis=a20, body=Body((inner1, inner2, Write(output="view_8", index=(Var("a20"),), value="v143"))))

    split = extract_output_specs([outer])

    assert split is not None
    pure, specs = split
    assert pure == ()
    assert [[axis.name for axis in spec.sweep] for spec in specs] == [["a20", "a27"], ["a20", "a27b", "a35"], ["a20"]]
    assert apply_output_specs(pure, specs) == Body((outer,))


def test_a_store_ahead_of_a_nested_sweep_keeps_its_source_position() -> None:
    """The outer store BEFORE the nest (the source order of post4096's gate stream): the outer
    group carries a prefix a later spec shares, so its write is appended bare and the last group
    carrying the outer axis wraps it together with the nest — store first, nest after, as written."""
    a20, a27 = _ax("a20", 4), _ax("a27", 8)
    inner = Loop(axis=a27, body=Body((Write(output="mul_16", index=(Var("a20"), Var("a27")), value="v216"),)))
    outer = Loop(axis=a20, body=Body((Write(output="view_8", index=(Var("a20"),), value="v143"), inner)))

    split = extract_output_specs([outer])

    assert split is not None
    pure, specs = split
    assert [[axis.name for axis in spec.sweep] for spec in specs] == [["a20"], ["a20", "a27"]]
    assert apply_output_specs(pure, specs) == Body((outer,))


def test_sibling_nested_sweeps_share_the_outer_loop() -> None:
    """Two write-only nests under one outer sweep with no outer store: the outer loop is still
    opened exactly once, by the last group whose path carries it."""
    a20, a27, a27b = _ax("a20", 4), _ax("a27", 8), _ax("a27b", 4)
    inner1 = Loop(axis=a27, body=Body((Write(output="o1", index=(Var("a20"), Var("a27")), value="x"),)))
    inner2 = Loop(axis=a27b, body=Body((Write(output="o2", index=(Var("a20"), Var("a27b")), value="y"),)))
    outer = Loop(axis=a20, body=Body((inner1, inner2)))

    split = extract_output_specs([outer])

    assert split is not None
    pure, specs = split
    assert [[axis.name for axis in spec.sweep] for spec in specs] == [["a20", "a27"], ["a20", "a27b"]]
    assert apply_output_specs(pure, specs) == Body((outer,))


def test_nested_sweep_pure_prefix_rejoins_the_stream() -> None:
    """A pure prefix inside a nested write-only sweep rejoins the stream, exactly as a single
    sweep's does, and the trailing-run rule re-wraps it on reconstitution."""
    a20, a27 = _ax("a20", 4), _ax("a27", 8)
    inner = Loop(
        axis=a27,
        body=Body(
            (
                Load(name="x", input="src", index=(Var("a20"), Var("a27"))),
                Assign(name="y", op="relu", args=("x",)),
                Write(output="o", index=(Var("a20"), Var("a27")), value="y"),
            )
        ),
    )
    outer = Loop(axis=a20, body=Body((inner,)))

    split = extract_output_specs([outer])

    assert split is not None
    pure, specs = split
    assert [type(stmt).__name__ for stmt in pure] == ["Load", "Assign"]
    assert [[axis.name for axis in spec.sweep] for spec in specs] == [["a20", "a27"]]
    assert apply_output_specs(pure, specs) == Body((outer,))


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

    st = OutputSpec(write=Write(output="o", index=(Var("m"),), value="y"), sweep=(_ax("n"),))
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
    assert [axis.name for store in stores for axis in store.sweep] == ["a1", "a2"]
    assert apply_output_specs(pure, stores) == Body((first, second))
