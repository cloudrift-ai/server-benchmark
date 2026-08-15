"""Online-softmax recognition — fuse a standalone two-pass softmax into the streaming form.

The classic softmax reads its input three times: a row-max reduce, a ``Σ exp(x − max)``
reduce, then a normalize. The **online-softmax** trick (flash's softmax-stats half,
without the P@V value accumulator) collapses the two reduces into ONE streaming pass
over a ``(m, d)`` log-sum-exp ``TWISTED`` state — running row-max ``m`` and exp-sum
denominator ``d`` — so only two reads of ``x`` remain (the normalize pass downstream is
untouched, reading the final ``m`` + ``1/d``).

``online_softmax_combine`` builds the ``(m, d)`` carrier; :func:`try_online_softmax`
recognizes an adjacent ``(rowmax, Σexp)`` reduce pair over the same input + reduce
extent in a ``LoopOp`` body and rewrites it to the fused streaming loop. The pair is
recognized on the FOLD ALGEBRA — each loop reads through :func:`fold_from_loop`, the same
byte-identity-gated λ parser every other recognition step consumes, and the condition is
stated on the folds' combine canon and lifted values; the merge itself is the exp-family
product monoid (``exp_merge``), so no stmt pattern survives here. The carried
``(m, d)`` states fold through ``base``-``Accum``\\ s, so when the cell is lifted (the reduce
``Loop`` annotated ``TWISTED``) the seed is derived from ``op.identity`` by
``Loop.render``; explicit
``Init`` stmts are emitted before the loop as well, load-bearing only on the flat-zero-axis ``Fold``
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
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, component_ops
from emmy.compiler.ir.stmt.carrier import exp_merge
from emmy.compiler.pipeline.passes.lowering.tile._fromloop import fold_from_loop


def _fold_of(loop: Loop, canon: str):
    r"""The loop read as a single-state fold of combine-canon ``canon`` through the ONE loop→node
    parser (:func:`fold_from_loop` — the byte-identity-gated λ reading every recognition step
    shares), or ``None``. Returns ``(state name, lifted value name, lift body stmts)``.

    The pairing runs on the RAW pre-annotation body, whose ``Accum``\ s have no fold axes stamped
    yet; the derivation regenerates them as ``(loop.axis,)``, so the raw loop is brought to that
    canonical dissolved spelling before the byte-identity gate compares."""
    dissolved = replace(
        loop,
        body=Body(tuple(replace(s, axes=(loop.axis.name,)) if isinstance(s, Accum) and not s.axes else s for s in loop.body)),
    )
    f = fold_from_loop(dissolved)
    if f is None or len(f.combine.results) != 1 or len(f.lift.results) != 1:
        return None
    ops = component_ops(f.combine)
    if ops is None or [op.reduce_canon for op in ops] != [canon]:
        return None
    return f.combine.results[0], f.lift.results[0], list(f.lift.body)


def _rowmax(loop: Loop) -> tuple[str, str, tuple] | None:
    """``(state, input, index)`` if ``loop`` reads as a row-max fold of ONE loaded value."""
    read = _fold_of(loop, "maximum")
    if read is None:
        return None
    state, value, body = read
    ld = next((s for s in body if isinstance(s, Load) and s.name == value), None)
    return (state, ld.input, ld.index) if ld is not None else None


def _sumexp(loop: Loop, maxacc: str, input_buf: str) -> str | None:
    """The sum state name if ``loop`` reads as a ``Σ exp(x − maxacc)`` fold over ``input_buf`` —
    an additive fold whose lifted value is ``exp(subtract(load(input_buf, …), maxacc))``."""
    read = _fold_of(loop, "add")
    if read is None:
        return None
    state, value, body = read
    expa = next((s for s in body if isinstance(s, Assign) and s.name == value and s.op.name == "exp"), None)
    if expa is None:
        return None
    suba = next((s for s in body if isinstance(s, Assign) and s.name == expa.args[0] and s.op.name == "subtract"), None)
    if suba is None or maxacc not in suba.args:
        return None
    ld = next((s for s in body if isinstance(s, Load) and s.name == suba.args[0] and s.input == input_buf), None)
    return state if ld is not None else None


def _fuse(body: Body) -> tuple[Body, bool]:
    """Recurse into nested ``Loop`` bodies; fuse any adjacent ``(rowmax, sum-of-exp)``
    reduce pair over the same input + reduce extent into one streaming online-softmax loop —
    a ``TWISTED`` reduce ``Loop``, its body the score
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
                    # The generated streaming merge (``base``-``Accum`` folds + ψ rescales — the
                    # exp-family program over ``(m, d)`` with injected terms ``(s, 1.0)``) sits in
                    # the loop body directly; the loop is stamped TWISTED, and the algebra is the
                    # body itself (``Fold.from_loop`` reconstructs it). No explicit ``Init``
                    # seeds — ``Loop.render`` seeds each fold ``Accum`` from ``op.identity``
                    # ((−inf, 0)).
                    fused = Loop(
                        axis=s.axis,
                        body=Body.coerce(
                            (Load(name=src, input=input_buf, index=index), *exp_merge((maxacc, sumacc), (src, 1.0), key=maxacc))
                        ),
                        role=AxisRole.TWISTED,
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
