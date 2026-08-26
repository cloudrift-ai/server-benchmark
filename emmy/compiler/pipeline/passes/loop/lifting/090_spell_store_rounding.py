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

**Which stores round.** Only a store whose value is an ``Accum``: a reduce accumulates in f32 until
``030_stamp_types`` stamps a dtype, so its value is provably wider than a 16-bit destination. An
``Assign`` chain already carries its inputs' width, so narrowing it here would introduce a rounding
the reference never performed.

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
from emmy.compiler.ir.loop import Accum, Assign, LoopOp, Write
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped

PATTERN = [Pattern("root", LoopOp)]


def rewrite(match: Match, root: Node) -> LoopOp:
    graph = match.graph
    op: LoopOp = root.op
    meta = op.analyze()
    taken = {name for stmt in op.body.iter() for name in stmt.defines()}
    edits: dict[int, tuple[Assign, Write]] = {}

    for write, _scope in meta.writes:
        if not write.is_scalar:
            continue
        value = meta.defs.get(write.value)
        if not isinstance(value, Accum):
            continue
        buffer = graph.buffer(write.output)
        # ``value.dtype or F32``: an Accum is unstamped in canonical Loop IR, and unstamped IS
        # f32 — the accumulation dtype, not a missing value.
        if buffer is None or buffer.transient or buffer.dtype == (value.dtype or F32):
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
