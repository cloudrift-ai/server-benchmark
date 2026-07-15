"""Split the dtype conversion out of a dtype-changing ``IndexMapOp``.

A traced view op can carry a dtype cast (torch stamps the cast on the view's output —
gemma's V path: ``mul_10`` f32 → ``transpose_2`` f16 feeding SDPA). Downstream, a view is
"pure plumbing": ``010_compose_indexmaps`` composes it into neighboring maps and loop
fusion splices it into consumers as a re-index. Both treatments silently DROP the dtype
boundary — the consumer kernel ends up loading the wide source buffer directly, and every
buffer-dtype gate downstream sees the wrong dtype (the flash K/V staging gate's
``kv_stage_ok`` reads the operand BUFFER dtype, so gemma's flash V stayed gmem-direct
forever — the 2026-07-15 layer-0 findings' biggest lockout).

The rewrite makes the cast a first-class COMPUTE node at the SOURCE's own shape:

    IndexMapOp(src@f32 → out@f16)  ⇒  copy(src@f32) → cast@f16 ; IndexMapOp(cast → out)

The elementwise ``copy`` carries the conversion (an ``Assign``-bearing kernel, so the
flash-operand materialization guards in loop fusion treat it as compute and keep it a real
buffer at flash offer sites), while the residual IndexMapOp is dtype-preserving plumbing
that composes and splices as before. A pointwise producer with fan-out 1 (gemma's V-norm
``mul``) then fuses INTO the cast at loop fusion, so the boundary usually costs no extra
kernel — the producer simply writes the narrow dtype.

Runs before ``010_compose_indexmaps`` so a cast-carrying view never composes into wider
plumbing (composing first would put the cast on the composed map's broadcast-shaped
output, materializing the broadcast instead of the source-shaped tensor).
"""

from __future__ import annotations

from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.tensor.ir import ElementwiseOp, IndexMapOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.frontend.decomposition._helpers import open_fragment

PATTERN = [Pattern("root", IndexMapOp)]


def rewrite(match: Match, root: Node) -> Graph | None:
    graph = match.graph
    op: IndexMapOp = root.op
    out = root.output
    mismatched = [i for i in {s.input_idx for s in op.sources} if graph.nodes[root.inputs[i]].output.dtype.name != out.dtype.name]
    if not mismatched:
        raise RuleSkipped("index map preserves dtype — nothing to split")

    frag = open_fragment(graph, list(root.inputs))
    new_inputs = list(root.inputs)
    for i in mismatched:
        src = graph.nodes[root.inputs[i]]
        cast_id = frag.add_node(
            op=ElementwiseOp(op="copy"),
            inputs=[src.id],
            output=Tensor(f"{out.name}_cast{i if len(mismatched) > 1 else ''}", tuple(src.output.shape), out.dtype),
        )
        new_inputs[i] = cast_id
    new_node = frag.add_node(
        op=IndexMapOp(out_shape=op.out_shape, sources=op.sources),
        inputs=new_inputs,
        output=Tensor(out.name, tuple(out.shape), out.dtype),
    )
    frag.outputs = [new_node]
    return frag
