"""Re-roll parallel per-fragment ``FragmentApply`` runs into ``#pragma unroll`` loops (``LOOPIFY`` —
pin-only, off by default).

The flash mma epilogue spends ~45% of its emitted body on element-wise 4-slot C-fragment arithmetic
laid out as straight-line statements — the ``O_i_f0..7 *= α`` rescale and the final ``O_i_f0..7 /= l``
divide are each an 8-long parallel run over a contiguous fragment family. This pass detects such a run,
**arrays** the family into one declaration (``float O_i_f[8][4]``, indexed ``O_i_f[t]`` kernel-wide via
the generic SSA rename), and **fuses** the run into a single ``StridedLoop`` over ``O_i_f[_t]``. The
result is ~30% fewer listing lines with **identical SASS** — nvcc unrolls the pragma — and handles every
op uniformly (``expf`` included, which a float4 vectorize pass cannot). This is a readability experiment
for blog / ``--ir cuda`` inspection, decoupled from production codegen.

Modeled on ``085_fast_exp`` (the pin-only kernel-IR peephole mold): ``PATTERN = [Pattern("root",
KernelOp)]`` and a ``rewrite`` that stamps the knob and returns the body unchanged when off — so the
**entire default pipeline runs LOOPIFY-off and is byte-identical** (no golden / snapshot churn). The knob
is an INT = the minimum run length to re-roll (``LOOPIFY.read_int(0)``): ``0`` / unset / ``< 2`` → off;
``EMMY_LOOPIFY=4`` catches the 8-long O rescale / divide (most of the win) while skipping the 2-long QK
scale; ``EMMY_LOOPIFY=2`` re-rolls every run ≥ 2. The fused loop's ``#pragma unroll`` rides the
``EMMY_UNROLL`` budget (``unroll_ok_n``), so ``EMMY_UNROLL=0`` keeps it rolled.

A run is re-rollable iff every stmt is an **in-place** ``FragmentApply`` whose FRAG-kind operands are all
the in-place target (the ``O_i_f{t} *= …`` shape), the targets form a **contiguous 0..N-1 family**
(``O_i_f0 … O_i_f{N-1}``, one common stem), and the op / kinds / layout / non-FRAG args are identical
across the run — the guard against a false-positive "looks parallel but isn't a clean family". The
interleaved ``subtract``→``exp`` per fragment is not a single-op run (each ``exp`` has ``subtract``
neighbors), so it never matches and is left for a later widening step.
"""

from __future__ import annotations

import re

from emmy.compiler.graph import Node
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Literal
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.kernel.ir import FRAG, FragmentApply, RegFragment
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Body, StridedLoop
from emmy.compiler.ir.stmt.base import _axis_identity
from emmy.compiler.ir.stmt.passes import rewrite as _rewrite
from emmy.compiler.pipeline import Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.lowering.kernel._atom import unroll_ok_n
from emmy.compiler.pipeline.search.space import LOOPIFY

PATTERN = [Pattern("root", KernelOp)]

_SUFFIX = re.compile(r"^(.*?)(\d+)$")


def rewrite(root: Node) -> KernelOp | None:
    op: KernelOp = root.op
    if LOOPIFY.name in op.knobs:
        raise RuleSkipped("LOOPIFY already decided (idempotence via knob)")
    n = LOOPIFY.read_int(0)
    knobs = {**op.knobs, LOOPIFY.name: n}
    if n < 2:  # 0 / unset / a lone iteration → off, byte-identical
        return KernelOp(body=op.body, name=op.name, knobs=knobs)
    families: dict[str, int] = {}
    _scan(op.body, n, families)
    if not families:
        return KernelOp(body=op.body, name=op.name, knobs=knobs)
    body = _rewrite_tree(op.body, n, families)
    # Point every remaining scalar reference at the arrayed decl: ``O_i_f{j}`` → ``O_i_f[j]`` (the
    # mma / ldmatrix / row-reduce / repack / store stmts + any non-re-rolled FragmentApply). The
    # arrayed RegFragment (name ``O_i_f``) and the fused loop's ``O_i_f[_t]`` are absent from the
    # map, so they pass through untouched.
    rename_map = {f"{stem}{j}": f"{stem}[{j}]" for stem, count in families.items() for j in range(count)}
    rn = lambda x: rename_map.get(x, x)  # noqa: E731
    body = Body(tuple(_rewrite(s, rn, Sigma.IDENTITY, _axis_identity) for s in body))
    return KernelOp(body=body, name=op.name, knobs=knobs)


def _split(name: str) -> tuple[str, int] | None:
    """``"O_i_f7"`` → ``("O_i_f", 7)``; ``None`` when the name has no trailing integer."""
    m = _SUFFIX.match(name)
    return (m.group(1), int(m.group(2))) if m else None


def _shape(s: object) -> tuple[str, int, tuple] | None:
    """The ``(stem, index, key)`` of a re-rollable ``FragmentApply``, else ``None``. ``key`` is the
    run-invariant signature (op / kinds / layout / non-FRAG args); two stmts join a run iff they
    share a ``key`` and a ``stem`` with consecutive indices."""
    if not isinstance(s, FragmentApply) or not s.in_place:
        return None
    if any(k == FRAG and a != s.out for a, k in zip(s.args, s.kinds, strict=True)):
        return None  # a multi-family / non-in-place FRAG op — not the ``O_i_f{t} *= …`` shape
    sp = _split(s.out)
    if sp is None:
        return None
    stem, idx = sp
    nonfrag = tuple(a for a, k in zip(s.args, s.kinds, strict=True) if k != FRAG)
    return stem, idx, (s.op.name, s.kinds, s.layout, nonfrag)


def _runs(stmts: list, min_run: int) -> list[tuple[int, int, str, int]]:
    """Maximal ``(start, end, stem, length)`` runs of consecutive re-rollable ``FragmentApply`` whose
    targets are a contiguous ``stem0..stem{L-1}`` family (index starting at 0) and whose length is
    ``>= max(min_run, 2)`` — a loop needs ≥ 2 iterations."""
    runs: list[tuple[int, int, str, int]] = []
    i, n = 0, len(stmts)
    while i < n:
        info = _shape(stmts[i])
        if info is None or info[1] != 0:  # a run's family must start at index 0
            i += 1
            continue
        stem, _, key = info
        j, expect = i + 1, 1
        while j < n:
            nxt = _shape(stmts[j])
            if nxt is None or nxt[0] != stem or nxt[1] != expect or nxt[2] != key:
                break
            j, expect = j + 1, expect + 1
        length = j - i
        if length >= min_run and length >= 2:
            runs.append((i, j, stem, length))
        i = max(j, i + 1)
    return runs


def _scan(body: Body, min_run: int, families: dict[str, int]) -> None:
    """Collect every family (stem → member count) that a qualifying run arrays, recursing into
    nested bodies (the O rescale rides the KV ``StridedLoop``; the divide rides the ``Tile`` body)."""
    stmts = list(body)
    for _start, _end, stem, count in _runs(stmts, min_run):
        families[stem] = max(families.get(stem, 0), count)
    for s in stmts:
        for b in s.nested():
            _scan(b, min_run, families)


def _fuse(template: FragmentApply, stem: str, count: int) -> StridedLoop:
    """The re-rolled run: one ``#pragma unroll`` ``StridedLoop`` over ``_t`` whose body is the
    template ``FragmentApply`` retargeted at ``stem[_t]`` (its FRAG operands index the arrayed
    family; the ROW / UNIFORM args are loop-invariant and unchanged)."""
    tgt = f"{stem}[_t]"
    args = tuple(tgt if k == FRAG else a for a, k in zip(template.args, template.kinds, strict=True))
    body = FragmentApply(out=tgt, op=template.op, args=args, kinds=template.kinds, in_place=template.in_place, layout=template.layout)
    return StridedLoop(
        axis=Axis(name="_t", extent=count),
        start=Literal(0, "int"),
        step=Literal(1, "int"),
        body=Body((body,)),
        unroll=unroll_ok_n(count),
    )


def _rewrite_tree(body: Body, min_run: int, families: dict[str, int]) -> Body:
    """Collapse each arrayed family's N ``RegFragment`` decls into one arrayed decl and replace each
    qualifying run with its fused loop, recursing into nested bodies."""
    stmts = list(body)
    runs = {start: (end, stem, count) for start, end, stem, count in _runs(stmts, min_run) if stem in families}
    out: list = []
    i = 0
    while i < len(stmts):
        if i in runs:
            end, stem, count = runs[i]
            out.append(_fuse(stmts[i], stem, count))
            i = end
            continue
        s = stmts[i]
        if isinstance(s, RegFragment):
            sp = _split(s.name)
            if sp is not None and sp[0] in families:
                stem, idx = sp
                if idx == 0:  # array at the family's first decl; drop the siblings (idx > 0)
                    out.append(RegFragment(name=stem, role=s.role, shape=s.shape, dtype=s.dtype, count=families[stem]))
                i += 1
                continue
        nested = s.nested()
        if nested:
            s = s.with_bodies(tuple(_rewrite_tree(b, min_run, families) for b in nested))
        out.append(s)
        i += 1
    return Body(tuple(out))
