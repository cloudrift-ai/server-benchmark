"""Generic integer elementwise typing and CUDA statement rendering."""

from __future__ import annotations

from importlib import import_module

import pytest

from emmy.compiler.dtype import U32
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
