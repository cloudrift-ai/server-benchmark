"""Lower static tensor casts and same-width bitcasts to generic CUDA copy kernels.

These minimal tensor ops can remain after decomposition when their producer is a runtime input.
Their shapes are static in the checkpoint-reconstruction lane; dynamic conversion kernels remain
the responsibility of Loop lifting once that dialect grows an explicit conversion statement.
"""

from emmy.compiler.backend.cuda.dtype import cuda_includes, cuda_name
from emmy.compiler.graph import Node
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.ir.tensor.ir import BitcastOp, CastOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.structural import digest

PATTERN = [Pattern("root", (CastOp, BitcastOp))]

_BLOCK = 256


def rewrite(match: Match, root: Node) -> CudaOp:
    inp = match.graph.buffer(root.inputs[0])
    if inp is None:
        raise RuleSkipped("conversion input has no tensor")
    if any(not dim.is_static for dim in root.output.shape):
        raise RuleSkipped("direct conversion lowering currently requires a static shape")
    size = 1
    for dim in root.output.shape:
        size *= dim.as_static()
    src_type = cuda_name(inp.dtype)
    dst_type = cuda_name(root.output.dtype)
    bitcast = isinstance(root.op, BitcastOp)
    if bitcast and inp.dtype.nbytes != root.output.dtype.nbytes:
        raise ValueError(f"BitcastOp requires equal element widths, got {inp.dtype} and {root.output.dtype}")
    kind = "bitcast" if bitcast else "cast"
    name = f"k_{kind}_{digest(inp.dtype.name, root.output.dtype.name, size)[:16]}"
    includes = "".join(f"#include {header}\n" for header in cuda_includes((inp.dtype, root.output.dtype)))
    value = f"reinterpret_cast<const {dst_type}*>(inp)[i]" if bitcast else f"static_cast<{dst_type}>(inp[i])"
    source = f"""{includes}extern "C" __global__
__launch_bounds__({_BLOCK}) void {name}(const {src_type}* inp, {dst_type}* out) {{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < {size}ULL) out[i] = {value};
}}
"""
    return CudaOp(
        kernel_source=source,
        kernel_name=name,
        arg_order=(root.inputs[0], root.id),
        grid=(((size + _BLOCK - 1) // _BLOCK or 1,), (1,), (1,)),
        block=((_BLOCK,), (1,), (1,)),
        comment=name,
    )
