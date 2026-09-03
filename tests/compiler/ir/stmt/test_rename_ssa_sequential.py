"""Regression tests for ``rename_ssa_sequential`` SSA renumbering.

The renumber assigns Load names ``in0, in1, ...`` in definition order. A
gather (a Load whose *index* references another Load's SSA name) must keep
pointing at the producing Load after the renumber — even when the producer's
new name collides with a *surviving* old name elsewhere in the body.

This guards the ``in24``-undefined embedding-lookup bug: the renumber used to
publish each Load rename into both the SSA channel (``rename``) and the axis
channel (``sigma``). The Load/Write rewriter applies both to index exprs, so an
indirect index Var was substituted twice — and a rename chain (``in2_3 → in5``
while a pre-existing ``in5`` → ``in26``) collapsed transitively, wiring the
gather to the wrong row.
"""

from __future__ import annotations

from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.stmt.blocks import Loop
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.leaves import Accum, Load
from emmy.compiler.ir.stmt.normalize import rename_ssa_sequential
from emmy.compiler.pipeline.passes.lowering.tile._fromloop import fold_from_loop


def _find_load(body, input_buf: str) -> Load:
    for s in Body.coerce(body).iter():
        if isinstance(s, Load) and s.input == input_buf:
            return s
    raise AssertionError(f"no Load from {input_buf!r} in body")


def test_gather_index_survives_rename_chain_collision() -> None:
    # Definition order fixes the renumber: ids loads → in0/in1, gather → in2,
    # then a Load *literally named* ``in1`` (a surviving old name) → in3. The
    # gather's index references the second ids load (old name ``idx1``), whose
    # new name ``in1`` collides with that surviving ``in1`` — the exact shape
    # of the embedding + RMSNorm-weight kernel that triggered the bug.
    body = Body(
        (
            Load(name="idx0", input="ids", index=(Var("a0"),)),
            Load(name="idx1", input="ids", index=(Var("a0"),)),
            Load(name="gather", input="w", index=(Var("idx1"), Var("a0"))),
            Load(name="in1", input="weight", index=(Var("a0"),)),
        )
    )

    out = rename_ssa_sequential(body)

    idx1_new = _find_load(out, "ids").names  # first ids load → in0
    # The second ids load renumbers to in1; the gather's indirect index must
    # resolve to *that* in1, not to the renamed weight load (in3).
    gather = _find_load(out, "w")
    weight = _find_load(out, "weight")

    assert weight.names == ("in3",)
    assert gather.index[0] == Var("in1"), gather.index[0]
    # And the gather index must NOT have been double-substituted onto the weight.
    assert gather.index[0] != Var(weight.names[0])
    # Sanity: idx0 is the very first Load → in0.
    assert idx1_new == ("in0",)


def test_fold_combine_tracks_accum_rename() -> None:
    # The algebra lives on the ``Fold`` TERM — its stored combine/lift must track an SSA rename in
    # lockstep (the Fold rewrite handler renames the combine through ``Lambda.rename``), or the
    # cooperative combine reads a state name the renamed lift no longer defines (the M=1
    # cut-consumer's ``acc1``-undefined miscompile). A term is never a ``Body`` member, so the
    # rename reaches it through the rewrite registry directly.
    from emmy.compiler.ir.stmt.passes import rewrite

    loop = Loop(
        axis=Axis(name="k0", extent=Dim(8)),
        body=Body(
            (
                Load(name="v9", input="a", index=(Var("k0"),)),
                Accum(name="acc1", value="v9", axes=("k0",)),
            )
        ),
    )
    fold = fold_from_loop(loop)
    assert fold is not None

    renamed = rewrite(fold, lambda name: f"{name}_r")

    assert renamed.combine.results == ("acc1_r",), renamed.combine.results
    assert renamed.exposes == ("acc1_r",)
    (load,) = (stmt for edge in renamed.operands for stmt in edge.lift.body if isinstance(stmt, Load))
    assert load.names == ("v9_r",) and renamed.applied.results == ("v9_r",), (load.names, renamed.applied.results)
