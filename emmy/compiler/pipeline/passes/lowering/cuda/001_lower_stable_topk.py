"""Lower deterministic stable top-k directly to one CUDA row kernel."""

from emmy.compiler.graph import Node
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.ir.tensor.ir import StableTopKOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.lowering.cuda._routing import row_launch

PATTERN = [Pattern("root", StableTopKOp)]


def rewrite(match: Match, root: Node) -> CudaOp:
    del match
    op = root.op
    ranking, payload = (root.op.inputs[name] for name in root.inputs)
    weights, indices = root.outputs
    if ranking.dtype.name != "f32" or payload.dtype.name != "f32":
        raise RuleSkipped("StableTopKOp CUDA lowering requires f32 ranking and payload")
    if weights.dtype.name != "f32" or indices.dtype.name != "i32":
        raise RuleSkipped("StableTopKOp CUDA lowering requires f32 weights and i32 indices")
    candidates = ranking.shape[1].as_static()
    rows, grid_factor, runtime_args = row_launch(weights)
    runtime_param = "" if not runtime_args else f", int {runtime_args[0]}"
    normalize = "true" if op.normalize else "false"
    name = f"k_stable_topk_{root.id}"
    source = f"""extern "C" __global__ void {name}(
    const float* ranking, const float* payload, float* weights, int* indices{runtime_param}) {{
  const int row = (int)blockIdx.x;
  if (row >= {rows}) return;
  int selected[{op.k}];
  float values[{op.k}];
  float total = 0.0f;
#pragma unroll
  for (int slot = 0; slot < {op.k}; ++slot) {{
    int best_index = -1;
    float best_value = -3.402823466e+38F;
#pragma unroll
    for (int candidate = 0; candidate < {candidates}; ++candidate) {{
      bool seen = false;
#pragma unroll
      for (int prior = 0; prior < slot; ++prior) seen |= selected[prior] == candidate;
      const float value = ranking[row * {candidates} + candidate];
      if (!seen && (best_index < 0 || value > best_value)) {{
        best_value = value;
        best_index = candidate;
      }}
    }}
    selected[slot] = best_index;
    values[slot] = payload[row * {candidates} + best_index];
    total += values[slot];
  }}
  float factor = {float(op.scale):.9g}f;
  if ({normalize}) factor /= total > 0.0f ? total : 1.0f;
#pragma unroll
  for (int slot = 0; slot < {op.k}; ++slot) {{
    const int offset = row * {op.k} + slot;
    weights[offset] = values[slot] * factor;
    indices[offset] = selected[slot];
  }}
}}
"""
    return CudaOp(
        kernel_source=source,
        kernel_name=name,
        arg_order=(*root.inputs, *root.buffer_names()),
        grid=((grid_factor,), (1,), (1,)),
        block=((1,), (1,), (1,)),
        runtime_args=runtime_args,
        comment=name,
    )
