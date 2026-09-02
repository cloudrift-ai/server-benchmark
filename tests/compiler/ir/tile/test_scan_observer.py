"""The scan spelling — a ``Fold`` with a per-step observer.

Pins the observer contract end to end at the IR layer: formation (fresh results, positional
state binding, family observability, no observer where the stream is not a stream), identity (a
cumsum keys apart from a sum; α-invariance holds; observer-free digests are untouched), the
rename lockstep, recognition (a reduce loop with a per-step store lifts to an observed fold plus
a boundary store over the observed name), and reconstitution (the streamed store rides inside
the reduce loop, after the observer stmts)."""

from __future__ import annotations

import dataclasses

import pytest

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.pure import Lambda
from emmy.compiler.ir.pure.algebra import ExpFamily
from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
from emmy.compiler.ir.tile import OutputSpec, extract_output_specs, lower_with_output_specs, observed_result_names
from emmy.compiler.pipeline.passes.lowering.tile._fromloop import fold_from_loop, lift_loop_op, scan_from_loop


def _scan_loop(axis_name: str = "k", acc: str = "acc", out: str = "out") -> Loop:
    """The 025_lift_scan shape: a reduce loop that both folds and stores per iteration."""
    index = (Var("m"), Var(axis_name))
    return Loop(
        axis=Axis(axis_name, 16),
        body=Body(
            (
                Load(name="x0", input="x", index=index),
                Accum(name=acc, value="x0", op=ElementwiseImpl("add"), axes=(axis_name,)),
                Write(output=out, index=index, value=acc),
            )
        ),
    )


def _sum_fold(axis_name: str = "k", acc: str = "acc") -> Fold:
    loop = _scan_loop(axis_name, acc)
    pure = Loop(axis=loop.axis, body=Body(tuple(s for s in loop.body if not isinstance(s, Write))))
    return fold_from_loop(pure)


def _observed(fold: Fold, obs: str | None = None) -> Fold:
    (acc,) = fold.combine.results
    obs = obs or f"{acc}__obs"
    observe = Lambda(
        params=(fold.axis.name, acc),
        body=Body((Assign(name=obs, op="copy", args=(acc,)),)),
        results=(obs,),
    )
    return dataclasses.replace(fold, observe=observe)


# --- formation ----------------------------------------------------------------------------------- #


def test_observer_formation_gates() -> None:
    fold = _sum_fold()
    scan = _observed(fold)
    assert scan.observe is not None and fold.observe is None
    assert scan.defines() == ("acc", "acc__obs")

    with pytest.raises(AssertionError, match="zero-axis"):
        Fold(lift=Lambda(params=(), body=Body(()), results=()), observe=scan.observe)
    with pytest.raises(AssertionError, match="positionally"):
        # A closed λ over the same names, bound in the wrong ORDER: the observer's contract is
        # positional — the iteration var first, then the carried state.
        body = Body((Assign(name="t", op="copy", args=("acc",)),))
        dataclasses.replace(fold, observe=Lambda.closing(("acc", "k"), body, ("t",)))
    with pytest.raises(AssertionError, match="FRESH"):
        # Observing the state under its own name is ill-formed — the boundary distinguishes a
        # streamed store from a post-fold store by the name.
        dataclasses.replace(fold, observe=Lambda(params=("k", "acc"), body=Body(()), results=("acc",)))


def test_exp_family_declines_an_observer() -> None:
    names = ("m_i", "l_i")
    combine = ExpFamily().program(names)
    lift = Lambda(params=("k",), body=Body((Load(name="s0", input="s", index=(Var("k"),)),)), results=("s0", 1.0))
    fold = Fold(axis=Axis("k", 8), lift=lift, init=(float("-inf"), 0.0), combine=combine)
    observe = Lambda(params=("k", *names), body=Body((Assign(name="m__obs", op="copy", args=("m_i",)),)), results=("m__obs",))
    with pytest.raises(AssertionError, match="does not support a per-step observer"):
        dataclasses.replace(fold, observe=observe)


# --- identity ------------------------------------------------------------------------------------ #


def test_scan_keys_apart_from_sum_and_alpha_invariantly() -> None:
    plain = _sum_fold()
    scan = _observed(plain)
    assert plain.structural_key() != scan.structural_key(), "a cumsum is not a sum"
    renamed = _observed(_sum_fold(acc="total"), obs="total__obs")
    assert scan.structural_key() == renamed.structural_key(), "SSA spelling must not enter identity"
    other_axis = _observed(_sum_fold(axis_name="j"))
    assert scan.structural_key() == other_axis.structural_key(), "axis spelling must not enter identity"


def test_rewrite_threads_the_observer() -> None:
    scan = _observed(_sum_fold())
    # An SSA rename map carries SSA defines only — the iteration var and the enclosing row
    # coordinate are axis names, and they rename through ``axis_fn`` / σ, never through this.
    ssa = {"acc", "acc__obs", "x0"}
    renamed = scan.rewrite(lambda name: f"{name}_r" if name in ssa else name)
    assert renamed.observe is not None
    assert renamed.observe.params == ("k", "acc_r")
    assert renamed.observe.results == ("acc__obs_r",)
    assert renamed.combine.results == ("acc_r",)


# --- recognition --------------------------------------------------------------------------------- #


def test_scan_from_loop_lifts_the_per_step_store() -> None:
    fold, trailing = scan_from_loop(_scan_loop())
    assert fold.observe is not None and fold.observe.results == ("acc__obs",)
    assert len(trailing) == 1 and trailing[0].values == ("acc__obs",)
    with pytest.raises(ValueError, match="scan, not a pure reduction"):
        fold_from_loop(_scan_loop())
    # The derived step taps AFTER the combine: observer stmts trail the Accum.
    kinds = [type(s).__name__ for s in fold.step_stmts()]
    assert kinds.index("Accum") < kinds.index("Assign", kinds.index("Accum"))


def test_scan_from_loop_rejects_a_store_off_the_state() -> None:
    loop = _scan_loop()
    bad = Loop(axis=loop.axis, body=Body((*loop.body[:-1], Write(output="out", index=(Var("m"), Var("k")), value="x0"))))
    with pytest.raises(ValueError, match="may only observe the carried state"):
        scan_from_loop(bad)


# --- the boundary -------------------------------------------------------------------------------- #


def test_streamed_store_reconstitutes_inside_the_reduce_loop() -> None:
    fold, trailing = scan_from_loop(_scan_loop())
    stream = [fold, *trailing]
    split = extract_output_specs(stream)
    assert split is not None
    body, specs = split
    assert [type(s).__name__ for s in body] == ["Fold"] and len(specs) == 1
    assert observed_result_names(fold) == frozenset({"acc__obs"})

    lowered = lower_with_output_specs(fold, specs)
    (loop,) = [s for s in lowered if isinstance(s, Loop)]
    inner = list(loop.body)
    assert isinstance(inner[-1], Write) and inner[-1].values == ("acc__obs",), "the store rides each iteration, last"
    assert not any(isinstance(s, Write) for s in lowered if not isinstance(s, Loop))


def test_observer_free_reconstitution_is_untouched() -> None:
    """A plain reduce with a post-fold store keeps its kernel-tail position — the observed-name
    membership, not loop-defines, is what streams a store."""
    fold = _sum_fold()
    spec = OutputSpec(write=Write(output="out", index=(Var("m"),), value="acc"))
    lowered = lower_with_output_specs(fold, (spec,))
    assert isinstance(lowered[-1], Write), "the post-fold store stays the kernel tail"
    (loop,) = [s for s in lowered if isinstance(s, Loop)]
    assert not any(isinstance(s, Write) for s in loop.body)


# --- the schedule gate --------------------------------------------------------------------------- #


def test_lifted_scan_kernel_shape() -> None:
    """The whole 025 → lift path: free axis peeled, observed fold stored, streamed spec at the
    boundary — the TileOp the schedule walk and the materializer consume."""
    from emmy.compiler.ir.loop import LoopOp

    scan = _scan_loop()
    op = LoopOp(body=(Loop(axis=Axis("m", 4), body=Body((scan,))),))
    tile = lift_loop_op(op, name="k_scan")
    # Canonical renumbering renames the accumulator; the observed name follows in lockstep.
    (observed,) = observed_result_names(tile.op)
    assert observed.endswith("__obs")
    assert len(tile.output_specs) == 1 and tile.output_specs[0].write.values == (observed,)
