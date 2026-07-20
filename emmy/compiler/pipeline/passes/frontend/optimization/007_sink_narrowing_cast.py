"""Sink a narrowing ``copy`` cast into its sole producer, so the producer writes the narrow dtype.

``005_split_cast_from_indexmap`` turns a dtype-changing view into a real ``copy`` compute node,
expecting that "a pointwise producer with fan-out 1 then fuses INTO the cast at loop fusion, so the
producer simply writes the narrow dtype". Loop fusion is free to merge in either direction, and on
gemma-4 it consistently goes the other way: the cheap cast is spliced into its CONSUMERS instead, so
the wide producer buffer survives and every consumer loads it wide.

Measured cost of that on the gemma-4 post twins: the pre-FF RMSNorm wrote ``float* mul_3`` and the
gate/up projections read ``const float* mul_3``. A mixed-dtype A operand has no plain mma tier — the
copy transports move raw bytes and cannot convert — so `_demote_mixed_a` routed both projections onto
the ``sync`` compute-fill, which has no weight-prefetch ring. They streamed their 118 MB weight at
1.12 TB/s where the neighbouring down_proj, whose A is a clean f16 buffer and which therefore rides
``d2/tma/ring``, hit 1.61 TB/s on the identical weight footprint.

The rewrite is a dtype retype, not a numeric change: an ``ElementwiseOp`` computes in its inputs'
promoted precision and rounds on store, so folding ``copy(x@f32) -> f16`` into ``x`` — retyping only
the OUTPUT tensor — produces bit-identical values to computing f32 and copying. The inputs are
untouched, so nothing upstream (the RMSNorm statistic, which must stay f32 — squaring gemma
activations in f16 overflows above |x| = 256) narrows.

Fires only when the producer is a pointwise ``ElementwiseOp`` whose SOLE consumer is the cast: with
any other consumer the wide value is genuinely needed, and retyping would narrow it for everyone.
"""

from __future__ import annotations

from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.tensor.ir import ElementwiseOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped


def _op_name(op) -> str:
    """The elementwise op's canonical name — ``ElementwiseOp.op`` is an ``ElementwiseImpl``, not a str."""
    return getattr(op.op, "name", op.op)


PATTERN = [
    Pattern("producer", ElementwiseOp),
    Pattern("cast", ElementwiseOp),
]


def rewrite(match: Match, producer: Node, cast: Node) -> Graph | None:
    graph = match.graph
    if not isinstance(cast.op, ElementwiseOp) or _op_name(cast.op) != "copy":
        raise RuleSkipped("consumer is not a copy cast")
    if len(cast.inputs) != 1 or cast.inputs[0] != producer.id:
        raise RuleSkipped("cast does not read the producer")
    if not isinstance(producer.op, ElementwiseOp) or _op_name(producer.op) == "copy":
        raise RuleSkipped("producer is not a pointwise compute node")
    if cast.output.dtype.name == producer.output.dtype.name:
        raise RuleSkipped("copy preserves dtype — nothing to sink")
    if tuple(cast.output.shape) != tuple(producer.output.shape):
        raise RuleSkipped("cast reshapes — not a pure dtype boundary")
    # The wide value must be needed by NOTHING else: retyping the producer narrows it for every
    # consumer, so a second reader would silently lose precision it may depend on.
    if graph.users(producer.id) != {cast.id}:
        raise RuleSkipped(f"producer has {len(graph.users(producer.id))} consumers — the wide value is still live")

    frag = Graph()
    for inp in producer.inputs:
        if inp in frag.nodes:
            continue
        ext = graph.nodes.get(inp)
        frag.add_node(InputOp(), [], Tensor(inp, ext.output.shape if ext else (), ext.output.dtype if ext else "f32"), node_id=inp)
    out = frag.add_node(
        producer.op,
        list(producer.inputs),
        Tensor(cast.output.name, tuple(cast.output.shape), cast.output.dtype),
    )
    frag.outputs = [out]

    match.output = cast.id
    match.consumed = {producer.id, cast.id}
    return frag
