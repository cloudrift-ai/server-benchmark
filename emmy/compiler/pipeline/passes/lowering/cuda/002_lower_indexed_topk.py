"""Lower indexed fixed-route normalization directly to one CUDA row kernel."""

from emmy.compiler.graph import Node
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.ir.tensor.ir import IndexedTopKOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.lowering.cuda._routing import row_launch

PATTERN = [Pattern("root", IndexedTopKOp)]


def rewrite(match: Match, root: Node) -> CudaOp:
    del match
    op = root.op
    payload, table, row_indices = (root.op.inputs[name] for name in root.inputs)
    weights, indices = root.outputs
    if payload.dtype.name != "f32" or table.dtype.name != "i32":
        raise RuleSkipped("IndexedTopKOp CUDA lowering requires f32 payload and i32 table")
    if row_indices.dtype.name not in {"i32", "i64"}:
        raise RuleSkipped("IndexedTopKOp CUDA lowering requires i32 or i64 row indices")
    if weights.dtype.name != "f32" or indices.dtype.name != "i32":
        raise RuleSkipped("IndexedTopKOp CUDA lowering requires f32 weights and i32 indices")
    candidates = payload.shape[1].as_static()
    entries = table.shape[0].as_static()
    rows, grid_factor, runtime_args = row_launch(weights)
    runtime_param = "" if not runtime_args else f", int {runtime_args[0]}"
    index_type = "int" if row_indices.dtype.name == "i32" else "long long"
    normalize = "true" if op.normalize else "false"
    name = f"k_indexed_topk_{root.id}"
    source = f"""extern "C" __global__ void {name}(
    const float* payload, const int* table, const {index_type}* row_indices,
    float* weights, int* indices{runtime_param}) {{
  const int row = (int)blockIdx.x;
  if (row >= {rows}) return;
  const int lane = (int)threadIdx.x;
  const {index_type} table_row = row_indices[row];
  if (table_row < 0 || table_row >= {entries}) {{
    if (lane == 0) {{
#pragma unroll
      for (int slot = 0; slot < {op.k}; ++slot) {{
        weights[row * {op.k} + slot] = 0.0f;
        indices[row * {op.k} + slot] = 0;
      }}
    }}
    return;
  }}
  float values[{op.k}];
  float total = 0.0f;
  bool valid = true;
#pragma unroll
  for (int slot = 0; slot < {op.k}; ++slot) {{
    const int candidate = table[table_row * {op.k} + slot];
    const bool candidate_valid = candidate >= 0 && candidate < {candidates};
    valid &= candidate_valid;
    values[slot] = candidate_valid ? payload[row * {candidates} + candidate] : 0.0f;
    const int owner = candidate_valid ? (candidate % {op.reduction_lanes * op.lane_chunk}) / {op.lane_chunk} : -1;
    if (lane == owner) total += values[slot];
  }}
  if (!valid) {{
    if (lane == 0) {{
#pragma unroll
      for (int slot = 0; slot < {op.k}; ++slot) {{
        weights[row * {op.k} + slot] = 0.0f;
        indices[row * {op.k} + slot] = 0;
      }}
    }}
    return;
  }}
  if ({op.reduction_lanes} > 1) {{
#pragma unroll
    for (int mask = {op.reduction_lanes // 2}; mask > 0; mask /= 2)
      total += __shfl_xor_sync(0xffffffff, total, mask, {op.reduction_lanes});
  }}
  float factor = {float(op.scale):.9g}f;
  if ({normalize}) factor /= total > 0.0f ? total : 1.0f;
#pragma unroll
  for (int slot = 0; slot < {op.k}; ++slot) {{
    const int candidate = table[table_row * {op.k} + slot];
    const int owner = (candidate % {op.reduction_lanes * op.lane_chunk}) / {op.lane_chunk};
    if (lane == owner) {{
      const int offset = row * {op.k} + slot;
      indices[offset] = candidate;
      weights[offset] = values[slot] * factor;
    }}
  }}
}}
"""
    return CudaOp(
        kernel_source=source,
        kernel_name=name,
        arg_order=(*root.inputs, *root.buffer_names()),
        grid=((grid_factor,), (1,), (1,)),
        block=((op.reduction_lanes,), (1,), (1,)),
        runtime_args=runtime_args,
        comment=name,
    )
