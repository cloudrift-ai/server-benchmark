"""Delegate an atomic accumulator's per-launch zero-init to a predecessor kernel (WS2).

Every atomic accumulator (atomic ``Write`` / ``RegStore`` output, a ``RowAccum`` aux buffer)
pays a per-launch memset — a CUDA-graph MEMSET node per site (~1.3 µs isolated, 3-5 per gemma-4
decode layer). The zero only has to happen-before the accumulating launch IN THE SAME STREAM, so
it can ride any launch that precedes it: this rule injects a ``ZeroPrologue`` stmt (CTA 0 writes
raw zero words) into a dataflow-PREDECESSOR kernel — a producer of one of the accumulator
kernel's inputs, which topological launch order puts strictly earlier — and marks the buffer
``zero_delegated`` on the accumulator so ``010_lower_kernelop`` drops it from
``CudaOp.zero_outputs``. Runs before ``010`` in this pass (file order), while both nodes are
still ``KernelOp``\\ s.

Correctness:

- **Happen-before**: single-stream serialization — every CTA of the predecessor (including the
  zeroing CTA 0) completes before the accumulator kernel starts. Across capture replays the
  previous step's consumers of the buffer also precede this step's predecessor in stream order,
  so the zero never wipes live data.
- **No graph edge**: the target buffer is the DOWNSTREAM kernel's own output — an input edge on
  the predecessor would be a cycle. The buffer joins the predecessor's ``outputs`` dict (and so
  its signature / ``arg_names``) via the stmt's ``external_writes``; the slab planner learns the
  earlier first-write through ``CudaOp.zero_prologues`` (its live interval must START at the
  zeroing launch, or a slab-slot neighbor could alias the bytes the prologue writes).
- **First launch keeps its memset**: a kernel with no ``KernelOp`` producer (all inputs are
  graph inputs / constants) is skipped — nothing precedes it in the capture.
- **Static shapes only**: the word count bakes into the source; a symbolic-shaped accumulator
  keeps the runtime memset.
- **Name re-suffix**: kernel names were content-stamped BEFORE this rewrite, and launches
  resolve kernels by name (first source wins) — two same-named predecessors delegated
  different-sized buffers would run one body with the other's baked word count. The suffix
  carries the total word count, so equal-suffix collisions are positionally safe (params bind
  by position; equal structure + equal counts).
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Node, Tensor
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.stmt import Body, ZeroPrologue
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.lowering.cuda._helpers import atomic_outputs

PATTERN = [Pattern("root", KernelOp)]


def rewrite(match: Match, root: Node) -> KernelOp | None:
    graph = match.graph
    kernel: KernelOp = root.op
    if not isinstance(kernel, KernelOp):
        raise RuleSkipped("root is no longer a KernelOp")
    todo = [b for b in atomic_outputs(kernel) if b not in kernel.zero_delegated]
    if not todo:
        raise RuleSkipped("no undelegated atomic outputs")

    pnode = next((n for i in root.inputs if (n := graph.nodes.get(i)) is not None and isinstance(n.op, KernelOp)), None)
    if pnode is None:
        raise RuleSkipped("no KernelOp predecessor — the capture's first launch keeps its memset")

    already = {s.dst for s in pnode.op.body.iter() if isinstance(s, ZeroPrologue)}
    prologues: list[ZeroPrologue] = []
    delegated: list[str] = []
    for b in todo:
        t = kernel.outputs.get(b)
        if t is None or b in already:
            continue
        if not all(d.is_static for d in t.shape):
            continue  # symbolic accumulator — the runtime memset resolves its size
        nbytes = t.dtype.nbytes
        for d in t.shape:
            nbytes *= d.as_static()
        if nbytes % 4:
            continue  # raw-word zeroing needs 4-byte multiples; every real accumulator is one
        prologues.append(ZeroPrologue(dst=b, words=nbytes // 4))
        delegated.append(b)
    if not prologues:
        raise RuleSkipped("every atomic output is symbolic / already delegated")

    # In-place predecessor mutation (the engine's Op-rewrite path only touches the root; the
    # fragment alternative would splice two kernels for a body prepend). The prologues go FIRST
    # in the body; the target buffers join ``outputs`` with placeholder Tensors — the next
    # match's ``populate_io`` swaps in the real graph tensors (the buffers are graph nodes).
    total_words = sum(s.words for s in prologues) + sum(s.words for s in pnode.op.body.iter() if isinstance(s, ZeroPrologue))
    base = pnode.op.name.rsplit("__zp", 1)[0] if "__zp" in pnode.op.name else pnode.op.name
    new_p = replace(pnode.op, body=Body((*prologues, *pnode.op.body)), name=f"{base}__zp{total_words}")
    new_p.inputs = dict(pnode.op.inputs)
    new_p.outputs = {**pnode.op.outputs, **{b: Tensor(b, (), "f32") for b in delegated}}
    pnode.op = new_p

    new_k = replace(kernel, zero_delegated=(*kernel.zero_delegated, *delegated))
    new_k.inputs = dict(kernel.inputs)
    new_k.outputs = dict(kernel.outputs)
    return new_k
