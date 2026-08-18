"""The e2m1 encode — writing NVFP4 rather than reading it.

The staged path reads a packed weight a checkpoint already quantized. Producing fp4 is the other
direction, and the hardware's block-scaled mma needs it: that instruction wants BOTH multiplicands
packed, and an activation arrives as f16 or bf16.

``to_f4e2m1`` is the encode op. It cannot work the way the fp8 encodes do. Those render as an
identity and let the conversion to an fp8 type do the rounding; these codes land on an ordinary
integer carrier, where that same identity would truncate the value instead of encoding it. So the
render spells the rounding itself, which leaves the CUDA spelling and ``dtype.encode_f4`` as two
independent implementations of one contract — what the device test below holds together.
"""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler.dtype import decode_f4, encode_f4
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.tensor.ir import ElementwiseOp
from tests.compiler.helpers import requires_cuda

#: Every exact midpoint of the e2m1 grid, where the rounding rule is decided.
MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def _encode_graph(n: int) -> Graph:
    """``codes = to_f4e2m1(x)`` over a flat f32 vector — the op on its own, no matmul around it."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (n,), "f32"), node_id="x")
    g.add_node(op=ElementwiseOp(op="to_f4e2m1"), inputs=["x"], output=Tensor("codes", (n,), "u16"), node_id="codes")
    g.inputs = ["x"]
    g.outputs = ["codes"]
    return g


def test_to_f4e2m1_is_registered_with_the_numpy_encode():
    op = ElementwiseImpl("to_f4e2m1")
    assert op.arity == 1
    got = op(np.array([1.0, -1.5, 6.0, 0.0], dtype=np.float32))
    assert got.dtype == np.uint8
    np.testing.assert_array_equal(decode_f4(got), [1.0, -1.5, 6.0, 0.0])


def test_the_render_emits_the_encode_helper_rather_than_a_dtype_conversion():
    """The fp8 encodes render as an identity and let the result dtype convert. This one must not:
    a raw uint8 store would move the wrong bits entirely."""
    from emmy.compiler.ir.expr import FuncCallExpr, Var
    from emmy.compiler.ir.stmt.base import op_to_expr

    expr = op_to_expr("to_f4e2m1", [Var("v")])
    assert isinstance(expr, FuncCallExpr)
    assert expr.name == "emmy_to_f4e2m1"


@requires_cuda
@pytest.mark.xdist_group("cuda")
def test_the_cuda_encode_matches_the_numpy_encode():
    """A real kernel over a dense sweep plus every midpoint and both saturation ends. The two
    implementations share no code, so agreement here is evidence rather than tautology."""
    from emmy.compiler.backend.cuda.backend import CudaBackend

    mids = np.array(MIDPOINTS, dtype=np.float32)
    x = np.concatenate(
        [
            np.linspace(-8.0, 8.0, 4001, dtype=np.float32),
            mids,
            -mids,
            np.array([0.0, -0.0, 1e9, -1e9], dtype=np.float32),
        ]
    )
    backend = CudaBackend()
    compiled = backend.compile(_encode_graph(x.size))
    source = next(s for node in compiled.nodes.values() if (s := getattr(node.op, "kernel_source", None)))
    assert "emmy_to_f4e2m1" in source, "the encode helper never reached the kernel"

    result, _ = backend.run(compiled, input_data={"x": x})
    got = np.asarray(result.outputs[compiled.outputs[0]]).astype(np.uint8).ravel()
    np.testing.assert_array_equal(got, encode_f4(x))


@requires_cuda
@pytest.mark.xdist_group("cuda")
def test_the_cuda_encode_rounds_ties_to_even():
    """Pinned separately from the sweep because it is the one rule a plain nearest-value encode
    would get wrong, and it would still pass a tolerance-based comparison."""
    from emmy.compiler.backend.cuda.backend import CudaBackend

    x = np.array(MIDPOINTS, dtype=np.float32)
    backend = CudaBackend()
    compiled = backend.compile(_encode_graph(x.size))
    result, _ = backend.run(compiled, input_data={"x": x})
    got = np.asarray(result.outputs[compiled.outputs[0]]).astype(np.uint8).ravel()
    # Codes 0, 2, 2, 4, 4, 6, 6 — every tie lands on the even code, never the odd one below it.
    np.testing.assert_array_equal(got, [0, 2, 2, 4, 4, 6, 6])
