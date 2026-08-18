"""Generic integer elementwise typing and CUDA statement rendering."""

from __future__ import annotations

from importlib import import_module

import pytest

from emmy.compiler.dtype import F32, I32, U32
from emmy.compiler.graph import Node, Tensor
from emmy.compiler.ir.expr import Literal, SimplifyCtx, Var
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.stmt import Assign, Body, Load, RenderCtx, Select, Write
from emmy.compiler.ir.stmt.leaves import SelectBranch
from emmy.compiler.ir.stmt.passes import simplify


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


def test_typed_select_is_structural_and_survives_rewrites():
    branches = (SelectBranch("a", Var("p")), SelectBranch("b", Literal(1, "int")))
    integer = Select(name="v", branches=branches, dtype=I32)
    floating = Select(name="v", branches=branches, dtype=F32)

    assert Body((integer,)).structural_key() != Body((floating,)).structural_key()
    assert integer.rewrite(lambda name: f"{name}_r").dtype == I32
    assert simplify(integer, SimplifyCtx.empty()).dtype == I32

    ctx = RenderCtx(ssa_dtypes={"a": "i32", "b": "i32"})
    assert integer.render(ctx) == ["    int v = ((p) ? (((int)(a))) : (((int)(b))));"]
    assert ctx.ssa_dtypes["v"] == "i32"


def test_stamp_types_infers_untyped_select_from_branches():
    idx = (Literal(0, "int"),)
    body = Body(
        (
            Load(name="a", input="lhs", index=idx),
            Load(name="b", input="rhs", index=idx),
            Select(name="selected", branches=(SelectBranch("a", Var("p")), SelectBranch("b", Literal(1, "int")))),
            Write(output="out", index=idx, value="selected"),
        )
    )
    tensor = Tensor("lhs", (1,), I32)
    kernel = KernelOp(body=body, name="integer_select", inputs={"lhs": tensor, "rhs": Tensor("rhs", (1,), I32)}, outputs={"out": tensor})
    root = Node(id="out", op=kernel, inputs=["lhs", "rhs"], outputs=(tensor,))

    stamped = import_module("emmy.compiler.pipeline.passes.lowering.kernel.030_stamp_types").rewrite(root)
    assert stamped is not None
    select = next(stmt for stmt in stamped.body if isinstance(stmt, Select))
    assert select.dtype == I32
