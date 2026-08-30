"""Spell the rounding a narrowing store performs as a statement.

A ``Write`` to a buffer narrower than the value it stores ROUNDS — that rounding is part of what
the source program computed, and it is the reason an unfused f16 matmul and its fused twin used to
disagree. Left implicit in ``Write.output``'s buffer dtype, it is a fact about a BUFFER, and fusion
deletes buffers: the merged kernel reads the producer's value directly and the rounding silently
disappears.

So it is spelled here instead, once, as an ordinary ``Assign`` conversion ahead of the store. After
this pass the rounding is a VALUE fact, and every later transform preserves it for free — the
splicer inlines the conversion through its ordinary plain-stmt path, total lift carries it into the
projection epilogue, and no pass needs a private channel for "this edge also converts".

**Which stores round.** A store whose value is an ``Accum``: a reduce accumulates in f32 until
``030_stamp_types`` stamps a dtype, so its value is provably wider than a 16-bit destination. The
same boundary can appear one node later when a decomposition writes the accumulator to a transient
shape buffer and a pass-through LoopOp copies it to the public buffer. The transient never stored,
so that direct load still carries the accumulator's width and the public copy performs the source
program's rounding. An ``Assign`` chain already carries its inputs' width, so narrowing it here
would introduce a rounding the reference never performed.

**Which stores need it spelled.** Only one whose buffer something READS. The rounding has to
survive fusion, and fusion is what deletes the store — but a buffer with no consumer is never
fused away, so its ``Write`` rounds on its own exactly as it always did. Spelling it there is not
merely redundant: a bare reduce would gain a one-statement projection it did not have, and that
wrapper is a cuttable ``PLACE`` seam (the reduce stops being the root). A cold fork took it, and
an f16 ``sum`` came out as two kernels — the reduce writing f32 to a workspace and a second kernel
doing nothing but the convert.

**Which buffers are boundaries.** Only a non-transient one. A transient buffer
(:class:`~emmy.compiler.tensor.Tensor` — decided in :meth:`Graph.splice`) never existed outside the
rewrite fragment that minted it, so its dtype carries shape rather than storage and there is no
rounding to spell. That is what keeps a decomposition's private intermediate (RMSNorm's mean,
softmax's denominator) computing at full width.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.dtype import F32
from emmy.compiler.graph import Node
from emmy.compiler.ir.loop import Accum, Assign, Load, LoopOp, Write
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped

PATTERN = [Pattern("root", LoopOp)]


def _accum_dtype(graph, meta, write: Write):
    """The accumulator dtype reaching ``write`` directly or through one private copy."""
    value = meta.defs.get(write.value)
    if isinstance(value, Accum):
        return value.dtype or F32
    if not isinstance(value, Load):
        return None
    source = graph.buffer(value.input)
    producer = graph.producer(value.input)
    if source is None or not source.transient or producer is None or not isinstance(producer.op, LoopOp):
        return None
    producer_meta = producer.op.analyze()
    produced = [candidate for candidate, _scope in producer_meta.writes if candidate.output == value.input]
    if len(produced) != 1 or not produced[0].is_scalar:
        return None
    accumulated = producer_meta.defs.get(produced[0].value)
    return (accumulated.dtype or F32) if isinstance(accumulated, Accum) else None


def rewrite(match: Match, root: Node) -> LoopOp:
    graph = match.graph
    op: LoopOp = root.op
    meta = op.analyze()
    taken = {name for stmt in op.body.iter() for name in stmt.defines()}
    edits: dict[int, tuple[Assign, Write]] = {}

    for write, _scope in meta.writes:
        if not write.is_scalar:
            continue
        value_dtype = _accum_dtype(graph, meta, write)
        if value_dtype is None:
            continue
        buffer = graph.buffer(write.output)
        if buffer is None or buffer.transient or buffer.dtype == value_dtype:
            continue
        if not graph.buffer_users(write.output):
            continue  # nothing reads it, so no fusion can delete the store that rounds
        name = f"{write.output}__st"
        while name in taken:
            name += "_"
        taken.add(name)
        edits[id(write)] = (
            Assign(name=name, op="copy", args=(write.value,), dtype=buffer.dtype),
            replace(write, values=(name,)),
        )

    if not edits:
        raise RuleSkipped("no narrowing store to spell")
    return replace(op, body=op.body.map(lambda stmt: edits.get(id(stmt), stmt) if isinstance(stmt, Write) else stmt))
