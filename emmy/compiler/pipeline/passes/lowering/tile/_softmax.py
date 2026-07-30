"""Online-softmax shared helper — the carrier builder + the recognition that fuses a
standalone two-pass softmax into it.

The classic softmax reads its input three times: a row-max reduce, a ``Σ exp(x − max)``
reduce, then a normalize. The **online-softmax** trick (flash's softmax-stats half,
without the P@V value accumulator) collapses the two reduces into ONE streaming pass
over a ``(m, d)`` log-sum-exp ``TWISTED`` :class:`Algebra` — running row-max ``m`` and exp-sum
denominator ``d`` — so only two reads of ``x`` remain (the normalize pass downstream is
untouched, reading the final ``m`` + ``1/d``).

``online_softmax_combine`` builds the ``(m, d)`` carrier; :func:`try_online_softmax`
recognizes an adjacent ``(rowmax, Σexp)`` reduce pair over the same input + reduce
extent in a ``LoopOp`` body and rewrites it to the fused streaming loop. The carried
``(m, d)`` states fold through ``base``-``Accum``\\ s, so when the cell is lifted (the reduce
``Loop`` annotated ``TWISTED`` with its ``Algebra``) the seed is derived from ``op.identity`` by
``Loop.render``; explicit
``Init`` stmts are emitted before the loop as well, load-bearing only on the flat-``Map``
fallback (a cell kept as loop-IR verbatim). Recognition is called from
``lowering/tile/010_recognize``
(after flash, before the plain-reduce normalize — each later step consumes the
``Accum``\\ s an earlier one matches).
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.axis import AxisRole
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import Accum, Algebra, Assign, Body, Load, Loop


def online_softmax_combine(m: str, d: str, s: str) -> Algebra:
    """The standalone **online-softmax** :class:`Algebra` — flash's softmax-stats half without the
    P@V accumulator (expectation component). State ``(m, d)`` (running row max / exp-sum
    denominator) folds this element's score partial ``s`` in ONE streaming pass::

        m_new = max(m, s);   alpha = exp(m − m_new);   p = exp(s − m_new)
        d = d·alpha + p;     m = m_new   (last)

    Built as the exp-family algebra over ``(m, d)`` with injected terms ``(s, 1.0)`` (pivot +
    denominator); ``merge`` / ``combine_states`` are *generated* (see ``ir/stmt/carrier.py``). The
    downstream normalize pass reads the final ``m`` and ``1/d``."""
    return Algebra.exp_family(s, (1.0,), (m, d))


def _rowmax(loop: Loop) -> tuple[str, str, tuple] | None:
    """``(acc, input, index)`` if ``loop`` is a row-max reduce of a single ``Load``."""
    body = list(loop.body)
    maxes = [s for s in body if isinstance(s, Accum) and s.op.reduce_canon == "maximum"]
    if len(maxes) != 1:
        return None
    acc = maxes[0]
    ld = next((s for s in body if isinstance(s, Load) and s.name == acc.value), None)
    return (acc.name, ld.input, ld.index) if ld is not None else None


def _sumexp(loop: Loop, maxacc: str, input_buf: str) -> str | None:
    """The sum ``Accum`` name if ``loop`` is a ``Σ exp(x − maxacc)`` reduce over
    ``input_buf`` — folds ``add`` over ``exp(subtract(load(input_buf, …), maxacc))``."""
    body = list(loop.body)
    sums = [s for s in body if isinstance(s, Accum) and s.op.reduce_canon == "add"]
    if len(sums) != 1:
        return None
    acc2 = sums[0]
    expa = next((s for s in body if isinstance(s, Assign) and s.name == acc2.value and s.op.name == "exp"), None)
    if expa is None:
        return None
    suba = next((s for s in body if isinstance(s, Assign) and s.name == expa.args[0] and s.op.name == "subtract"), None)
    if suba is None or maxacc not in suba.args:
        return None
    ld = next((s for s in body if isinstance(s, Load) and s.name == suba.args[0] and s.input == input_buf), None)
    return acc2.name if ld is not None else None


def _fuse(body: Body) -> tuple[Body, bool]:
    """Recurse into nested ``Loop`` bodies; fuse any adjacent ``(rowmax, sum-of-exp)``
    reduce pair over the same input + reduce extent into one streaming online-softmax loop —
    a ``TWISTED`` reduce ``Loop`` carrying the exp-family :class:`Algebra`, its body the score
    ``Load`` + the carrier's dissolved streaming ``merge`` (``base``-``Accum`` folds + ψ
    rescales)."""
    stmts = list(body)
    out: list = []
    changed = False
    i = 0
    while i < len(stmts):
        s = stmts[i]
        if isinstance(s, Loop) and i + 1 < len(stmts) and isinstance(stmts[i + 1], Loop):
            nxt = stmts[i + 1]
            mx = _rowmax(s)
            if mx is not None and s.axis.extent == nxt.axis.extent:
                maxacc, input_buf, index = mx
                sumacc = _sumexp(nxt, maxacc, input_buf)
                if sumacc is not None:
                    src = f"{maxacc}__osin"
                    carrier = online_softmax_combine(maxacc, sumacc, src)
                    # The carrier's streaming ``merge`` (``base``-``Accum`` folds + ψ rescales) sits
                    # in the loop body directly; the loop is stamped TWISTED + the carrier. No
                    # explicit ``Init`` seeds — ``Loop.render`` seeds each fold ``Accum`` from
                    # ``op.identity`` ((−inf, 0)).
                    fused = Loop(
                        axis=s.axis,
                        body=Body.coerce((Load(name=src, input=input_buf, index=index), *carrier.dissolve())),
                        role=AxisRole.TWISTED,
                        carrier=carrier,
                    )
                    out.append(fused)
                    changed = True
                    i += 2
                    continue
        if isinstance(s, Loop):
            nb, ch = _fuse(s.body)
            if ch:
                s = replace(s, body=nb)
                changed = True
        out.append(s)
        i += 1
    return Body.coerce(out), changed


def try_online_softmax(root: Node) -> LoopOp | None:
    """Fuse any ``(rowmax, Σexp)`` reduce pair in ``root``'s body into one streaming
    online-softmax ``TWISTED`` ``Loop``. Returns the rewritten ``LoopOp``, or ``None`` if
    there is nothing to fuse."""
    new_body, changed = _fuse(root.op.body)
    if not changed:
        return None
    return replace(root.op, body=new_body)
