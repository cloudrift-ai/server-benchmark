"""Lower a static :class:`RangeOp` directly to one generic CUDA fill kernel.

``RangeOp`` has no tensor inputs: its values come from the output coordinate itself.  Loop IR
intentionally models SSA values separately from coordinate expressions, so manufacturing a
fake input merely to carry the coordinate would obscure that distinction.  Keep the operation
generic through decomposition/fusion, then render the coordinate-derived fill at the final
backend boundary.
"""

from emmy.compiler.backend.cuda.dtype import cuda_includes, cuda_name
from emmy.compiler.graph import Node
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.ir.tensor.ir import RangeOp
from emmy.compiler.pipeline import Pattern
from emmy.compiler.structural import digest

PATTERN = [Pattern("root", RangeOp)]

_BLOCK = 256


def rewrite(root: Node) -> CudaOp:
    op: RangeOp = root.op
    size = len(range(op.start, op.stop, op.step))
    dtype = root.output.dtype
    ctype = cuda_name(dtype)
    name = f"k_range_{digest(op.start, op.stop, op.step, dtype.name)[:16]}"
    includes = "".join(f"#include {header}\n" for header in cuda_includes((dtype,)))
    source = f"""{includes}extern "C" __global__
__launch_bounds__({_BLOCK}) void {name}({ctype}* out) {{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < {size}ULL) {{
        long long value = {op.start}LL + (long long)i * {op.step}LL;
        out[i] = static_cast<{ctype}>(value);
    }}
}}
"""
    return CudaOp(
        kernel_source=source,
        kernel_name=name,
        arg_order=(root.id,),
        grid=(((size + _BLOCK - 1) // _BLOCK or 1,), (1,), (1,)),
        block=((_BLOCK,), (1,), (1,)),
        comment=name,
    )
