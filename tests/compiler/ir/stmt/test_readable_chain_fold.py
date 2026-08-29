"""The readability scalar-chain fold's straddle guard (``render_body``).

The fold moves a single-use ``Assign`` temp's computation from its def site to its sole read
site — sound only when no statement in between redefines an operand the folded expression
reads (a carrier commit / ``Accum`` is a same-body redefinition). These tests pin the guard,
including the transitive case where a chained fold carries an earlier temp's base operands
across the redefinition.
"""

from __future__ import annotations

from emmy.compiler.dtype import F16, F32, I32, U32
from emmy.compiler.ir.expr import Literal
from emmy.compiler.ir.stmt.base import RenderCtx, render_body
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Load


def _render(*stmts) -> list[str]:
    return render_body(Body(tuple(stmts)), RenderCtx())


def test_fold_inlines_single_use_temp(monkeypatch) -> None:
    """Positive control: with no redefinition in between, the single-use temp folds away."""
    monkeypatch.setenv("EMMY_READABLE", "1")
    lines = _render(
        Assign(name="t", op="subtract", args=("m", "s")),
        Assign(name="y", op="exp", args=("t",)),
    )
    assert not any(" t =" in ln for ln in lines), f"t must fold into its consumer: {lines}"
    assert any("(m - s)" in ln for ln in lines), f"y must carry the folded expression: {lines}"


def test_fold_keeps_target_rendered_bitcast_named(monkeypatch) -> None:
    """A bitcast is rendered by the target-aware ``Assign`` path, not ``op_to_expr``."""
    monkeypatch.setenv("EMMY_READABLE", "1")
    lines = _render(
        Assign(name="bits", op="bitcast", args=("x",), dtype=U32),
        Assign(name="y", op="copy", args=("bits",), dtype=U32),
    )
    assert any("unsigned int bits = emmy_bitcast<unsigned int>(x);" in line for line in lines)
    assert any(" y = bits;" in line for line in lines)


def test_fold_preserves_integer_dtype_for_bitwise_consumer(monkeypatch) -> None:
    """Inlining a typed shift must not turn its consumer's integer mask into logical AND."""
    monkeypatch.setenv("EMMY_READABLE", "1")
    ctx = RenderCtx(ssa_dtypes={"packed": "i32", "shift": "i32", "mask": "i32"})
    lines = render_body(
        Body(
            (
                Assign(name="shifted", op="right_shift", args=("packed", "shift"), dtype=I32),
                Assign(name="nibble", op="bitwise_and", args=("shifted", "mask"), dtype=I32),
            )
        ),
        ctx,
    )
    assert lines == ["    int nibble = (packed >> shift) & mask;"]


def test_fold_keeps_mixed_dtype_conversion_named(monkeypatch) -> None:
    """A readability fold must not bypass the target-aware conversion in ``Assign.render``."""
    monkeypatch.setenv("EMMY_READABLE", "1")
    lines = render_body(
        Body(
            (
                Load(name="scale", input="scale_buffer", index=(Literal(0, "int"),), dtype=F16),
                Assign(name="scaled", op="multiply", args=("acc", "scale"), dtype=F32),
                Assign(name="out", op="add", args=("bias", "scaled"), dtype=F32),
            )
        ),
        RenderCtx(ssa_dtypes={"acc": "f32", "bias": "f32"}, shapes={"scale_buffer": (1,)}),
    )
    assert lines == [
        "    __half scale = scale_buffer[0];",
        "    float scaled = acc * __half2float(scale);",
        "    float out = bias + scaled;",
    ]


def test_fold_declines_across_operand_redefinition(monkeypatch) -> None:
    """An interposed redefinition of the temp's operand (``Accum m``) must keep the temp named —
    folding would evaluate ``m - s`` after ``m`` changed."""
    monkeypatch.setenv("EMMY_READABLE", "1")
    lines = _render(
        Assign(name="t", op="subtract", args=("m", "s")),
        Accum(name="m", value="z"),
        Assign(name="y", op="exp", args=("t",)),
    )
    assert any(" t =" in ln for ln in lines), f"t must stay named across the m redefinition: {lines}"


def test_fold_declines_transitively_across_redefinition(monkeypatch) -> None:
    """A chained fold carries the earlier temp's base operands: ``t1 = m - s`` folds into
    ``t2 = exp(t1)``, so ``t2``'s fold into ``y`` must decline when ``m`` is redefined between
    ``t2``'s def and ``y`` — even though ``t2``'s own args never name ``m``."""
    monkeypatch.setenv("EMMY_READABLE", "1")
    lines = _render(
        Assign(name="t1", op="subtract", args=("m", "s")),
        Assign(name="t2", op="exp", args=("t1",)),
        Accum(name="m", value="z"),
        Assign(name="y", op="multiply", args=("t2", "c")),
    )
    assert any(" t2 =" in ln for ln in lines), f"t2 must stay named across the m redefinition: {lines}"
