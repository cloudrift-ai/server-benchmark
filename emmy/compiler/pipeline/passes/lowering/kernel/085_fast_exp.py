"""Lower ``exp`` through the SFU fast path (``FAST_EXP`` — pin-only, off by default).

libm ``expf`` compiles to a range-checked multi-instruction sequence; ``__expf`` is one FMUL +
``MUFU.EX2``. On an exp-dense fold (flash / online softmax: ~34 exponentials per lane per
streaming step) the sequence cost is a first-order term once the memory path is clean. The
rewrite swaps every ``exp`` :class:`ElementwiseImpl` for ``exp_fast`` — an op whose HOST
semantics are identical (``np.exp`` / libm on the loop runner, so references stay accurate) and
whose CUDA render is ``__expf`` (f32; the f16 spellings keep ``hexp``, already hardware-native).

This is the ONE policy knob that is **not bit-exact** (~2 ulp vs correctly-rounded ``expf``) —
numerically benign for the softmax family (the α rescale never amplifies, the carrier stays
fp32), but it must be a deliberate, pinnable choice: ``hints=(False,)``, never enumerated, so
only ``EMMY_FAST_EXP=1`` turns it on. The knob is stamped either way (the realized config
records the policy; idempotence rides the stamp, like ``095_interleave_loads``).

Why a kernel-IR pass and not a Loop-IR rewrite: the warp-flash softmax merge never comes from
Loop IR — the fragment realizer REGENERATES it from the carrier's channel spec — so only a
post-materialize rewrite reaches every tier's emitted body uniformly.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.stmt import Body, Stmt
from emmy.compiler.pipeline import Pattern, RuleSkipped
from emmy.compiler.pipeline.search.space import FAST_EXP

PATTERN = [Pattern("root", KernelOp)]

_FAST = ElementwiseImpl("exp_fast")


def rewrite(root: Node) -> KernelOp | None:
    op: KernelOp = root.op
    if FAST_EXP.name in op.knobs:
        raise RuleSkipped("FAST_EXP already decided (idempotence via knob)")
    if not FAST_EXP.narrow((False,))[0]:
        return KernelOp(body=op.body, name=op.name, knobs={**op.knobs, FAST_EXP.name: False})
    new_body, _ = _walk(op.body)
    return KernelOp(body=new_body, name=op.name, knobs={**op.knobs, FAST_EXP.name: True})


def _walk(body: Body) -> tuple[Body, bool]:
    stmts: list[Stmt] = []
    changed = False
    for s in body:
        nested = s.nested()
        if nested:
            new_bodies = []
            sub = False
            for b in nested:
                nb, c = _walk(b)
                new_bodies.append(nb)
                sub = sub or c
            if sub:
                s = s.with_bodies(tuple(new_bodies))
                changed = True
        op_attr = getattr(s, "op", None)
        if isinstance(op_attr, ElementwiseImpl) and op_attr.name == "exp":
            s = replace(s, op=_FAST)
            changed = True
        stmts.append(s)
    return Body(tuple(stmts)), changed
