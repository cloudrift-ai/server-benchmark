"""The reduce tier's ILP replication and the names it must NOT rename.

``_tile_reduce_axis`` copies the reduce body once per register chain, suffixing per-copy SSA
names. A name the body READS but does not DEFINE is one shared value (a hoisted scalar load, a
provider chain emitted ahead of the loop) — renaming its uses emits an undefined identifier at
nvcc. The Expr channel (``s.exprs()``) was already protected; ``Assign`` args travel the
SSA-deps channel (``s.deps()``), surfaced by DeepSeek-V4 post4096's two-cut piece
(``identifier "in3__r1" is undefined``).
"""

from __future__ import annotations

from types import SimpleNamespace

from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.schedule import Reduce
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop
from emmy.compiler.pipeline.passes.lowering.kernel._factor import Ctx, _tile_reduce_axis
from emmy.compiler.pipeline.passes.lowering.tile._fromloop import fold_from_loop


def _names_read(stmts) -> set[str]:
    out: set[str] = set()
    for s in stmts:
        out.update(s.deps())
        for e in s.exprs():
            out.update(e.free_vars())
        for b in s.nested():
            out |= _names_read(b)
    return out


def test_an_external_ssa_read_is_shared_by_every_ilp_copy() -> None:
    body = Body(
        (
            Load(name="x_e", input="x", index=(Var("m"), Var("k"))),
            Assign(name="scaled", op="multiply", args=("x_e", "alpha")),  # ``alpha`` defined ahead of the loop
            Accum(name="acc", value="scaled", op="add", axes=("k",)),
        )
    )
    red = fold_from_loop(Loop(axis=Axis("k", 128), body=body, role=AxisRole.PLANAR))
    assert red is not None
    sched = SimpleNamespace(get=lambda *_: None)
    ctx = Ctx(grid=(Axis("m", 4),), inputs={}, output="o", sched=sched)

    _state, fold, _close, _lane = _tile_reduce_axis(red, Reduce.of(reg=2), ctx, tail=(), out_val="acc")

    read = _names_read(fold)
    assert "alpha__r1" not in read, "the shared external value must not be renamed per copy"
    assert "alpha" in read
    assert any(name.endswith("__r1") for name in read), "the per-copy chains themselves still rename"
