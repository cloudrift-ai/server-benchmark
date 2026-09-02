"""Generic integer elementwise typing and CUDA statement rendering."""

from __future__ import annotations

from importlib import import_module

import pytest

from emmy.compiler.dtype import F32, U32
from emmy.compiler.graph import Node, Tensor
from emmy.compiler.ir.expr import Literal
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.stmt import Assign, Body, Load, RenderCtx, Write


@pytest.mark.parametrize(
    ("op", "symbol"),
    [
        ("floor_divide", "/"),
        ("remainder", "%"),
        ("left_shift", "<<"),
        ("right_shift", ">>"),
        ("bitwise_and", "&"),
        ("bitwise_or", "|"),
        ("bitwise_xor", "^"),
    ],
)
def test_integer_assign_renders_natively(op, symbol):
    ctx = RenderCtx(ssa_dtypes={"a": "u32", "b": "u32"})
    line = Assign(name="result", op=op, args=("a", "b")).render(ctx)[0]
    assert line == f"    unsigned int result = a {symbol} b;"
    assert ctx.ssa_dtypes["result"] == "u32"


def test_f32_bitwise_names_remain_logical_masks():
    ctx = RenderCtx(ssa_dtypes={"a": "f32", "b": "f32"})
    assert Assign(name="both", op="bitwise_and", args=("a", "b")).render(ctx) == ["    float both = a && b;"]
    assert Assign(name="either", op="bitwise_or", args=("a", "b")).render(ctx) == ["    float either = a || b;"]


@pytest.mark.parametrize(
    "op",
    ["add", "subtract", "multiply", "floor_divide", "remainder", "left_shift", "right_shift", "bitwise_and", "bitwise_or", "bitwise_xor"],
)
def test_stamp_types_preserves_integer_assign_and_write(op):
    idx = (Literal(0, "int"),)
    body = Body(
        (
            Load(name="a", input="lhs", index=idx),
            Load(name="b", input="rhs", index=idx),
            Assign(name="result", op=op, args=("a", "b")),
            Write(output="out", index=idx, value="result"),
        )
    )
    tensor = Tensor("lhs", (1,), U32)
    out = Tensor("out", (1,), U32)
    kernel = KernelOp(
        body=body,
        name="integer_elementwise",
        inputs={"lhs": tensor, "rhs": Tensor("rhs", (1,), U32)},
        outputs={"out": out},
    )
    root = Node(id="out", op=kernel, inputs=["lhs", "rhs"], outputs=(out,))

    stamped = import_module("emmy.compiler.pipeline.passes.lowering.kernel.030_stamp_types").rewrite(root)
    assert stamped is not None
    assign = next(stmt for stmt in stamped.body if isinstance(stmt, Assign))
    write = next(stmt for stmt in stamped.body if isinstance(stmt, Write))
    assert assign.dtype == U32
    assert write.value_dtype == U32


def test_stamp_types_repairs_a_stale_float_stamp_on_integer_algebra():
    idx = (Literal(0, "int"),)
    body = Body(
        (
            Load(name="a", input="lhs", index=idx),
            Load(name="b", input="rhs", index=idx),
            Assign(name="result", op="left_shift", args=("a", "b"), dtype=F32),
            Write(output="out", index=idx, value="result"),
        )
    )
    tensor = Tensor("lhs", (1,), U32)
    out = Tensor("out", (1,), U32)
    root = Node(
        id="out",
        op=KernelOp(
            body=body,
            name="integer_elementwise",
            inputs={"lhs": tensor, "rhs": Tensor("rhs", (1,), U32)},
            outputs={"out": out},
        ),
        inputs=["lhs", "rhs"],
        outputs=(out,),
    )

    stamped = import_module("emmy.compiler.pipeline.passes.lowering.kernel.030_stamp_types").rewrite(root)
    assign = next(stmt for stmt in stamped.body if isinstance(stmt, Assign))
    assert assign.dtype == U32


def test_an_integer_constant_load_keeps_its_dtype_when_it_folds_to_a_literal():
    """A one-element constant buffer renders as a literal binding rather than a buffer read. An
    INTEGER one must keep its own dtype: the binding's SSA dtype decides the CONSUMER's arithmetic,
    and a shift amount folded to f32 sends its shift down the f32 path, which has no `<<`."""
    ctx = RenderCtx(literal_constants={"shift_c": 4.0})
    load = Load(name="shift", input="shift_c", index=(Literal(0, "int"),), dtype=U32)
    assert load.render(ctx) == ["    unsigned int shift = 4;"]
    assert ctx.ssa_dtypes["shift"] == "u32"
    ctx.ssa_dtypes["odd"] = "u32"
    assert Assign(name="hi", op="left_shift", args=("odd", "shift")).render(ctx) == ["    unsigned int hi = odd << shift;"]


def test_a_float_constant_load_still_binds_as_a_float_literal():
    """The float case is untouched — every kernel reading a scalar float constant renders as before."""
    ctx = RenderCtx(literal_constants={"scale_c": 0.125})
    load = Load(name="scale", input="scale_c", index=(Literal(0, "int"),), dtype=F32)
    assert load.render(ctx) == ["    float scale = 0.125f;"]
    assert ctx.ssa_dtypes["scale"] == "f32"


def test_the_literal_prefold_leaves_an_integer_constant_named():
    """`render_body`'s pre-pass drops a literal-constant Load and inlines its value at every use
    site — as a FLOAT literal, since that is what its map holds. An integer constant stays out of
    it and keeps its named local, or the bit operation reading it lands back in f32."""
    from emmy.compiler.ir.stmt.base import render_body

    ctx = RenderCtx(literal_constants={"shift_c": 4.0, "scale_c": 0.125})
    body = Body(
        (
            Load(name="shift", input="shift_c", index=(Literal(0, "int"),), dtype=U32),
            Load(name="scale", input="scale_c", index=(Literal(0, "int"),), dtype=F32),
            Assign(name="hi", op="left_shift", args=("odd", "shift")),
            Assign(name="lo", op="multiply", args=("odd", "scale")),
        )
    )
    ctx.ssa_dtypes["odd"] = "u32"
    lines = render_body(body, ctx)
    assert "    unsigned int shift = 4;" in lines, lines
    assert not any("scale = 0.125f" in ln for ln in lines), "a float constant still inlines at its use site"
    assert any("odd << shift" in ln for ln in lines), lines
