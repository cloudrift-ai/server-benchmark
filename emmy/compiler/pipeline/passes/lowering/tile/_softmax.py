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

import re
from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.axis import AxisRole
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Select, component_ops
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


def _score_cone(body: list, result: str) -> tuple[list, Load] | None:
    """The pure elementwise cone producing ``result`` from a pair-loop body: the
    :meth:`Body.backward_cone` members (body order) restricted to ``Load``/``Assign``/``Select``
    reaching exactly ONE ``Load``. Names defined outside the loop (an enclosing-scope value, an
    axis var) surface as external reads and stay free. ``None`` when the value depends on
    anything else — the pairing then declines and the cell keeps its current reading."""
    cone = Body.coerce(tuple(s for s in body if not isinstance(s, Accum))).backward_cone([result])
    stmts = list(cone.members)
    if not stmts or any(not isinstance(s, (Load, Assign, Select)) for s in stmts):
        return None
    loads = [s for s in stmts if isinstance(s, Load)]
    if len(loads) != 1:
        return None
    return stmts, loads[0]


def _cone_canon(stmts: list, result: str, axis_name: str) -> str:
    """An α-renamed rendering of a score cone — locally-defined names canonicalize in definition
    order, the loop axis to a fixed placeholder, free names verbatim — so the two pair loops'
    recomputed score cones compare by content, exactly as the byte-identity λ gate compares
    lift bodies."""
    mapping = {axis_name: "_ax"}
    for s in stmts:
        for n in s.defines():
            mapping.setdefault(n, f"_c{len(mapping)}")
    text = repr(tuple(stmts)) + "|" + mapping.get(result, result)
    for old, new in mapping.items():
        text = re.sub(f"'{re.escape(old)}'", f"'{new}'", text)
    return text


def _rowmax(loop: Loop) -> tuple[str, str, list, Load] | None:
    """``(state, score name, cone stmts, load)`` if ``loop`` reads as a row-max fold of a score
    cone over ONE loaded value (the bare ``Load`` is the degenerate cone; a masked score —
    ``Select`` + ``add`` over the load — reads the same way)."""
    read = _fold_of(loop, "maximum")
    if read is None:
        return None
    state, value, body = read
    cone = _score_cone(body, value)
    if cone is None:
        return None
    return state, value, *cone


def _sumexp(loop: Loop, maxacc: str) -> tuple[str, str, list] | None:
    """``(state, score name, cone stmts)`` if ``loop`` reads as a ``Σ exp(score − maxacc)``
    fold — an additive fold whose lifted value is ``exp(subtract(<score cone>, maxacc))``."""
    read = _fold_of(loop, "add")
    if read is None:
        return None
    state, value, body = read
    expa = next((s for s in body if isinstance(s, Assign) and s.name == value and s.op.name == "exp"), None)
    if expa is None:
        return None
    suba = next((s for s in body if isinstance(s, Assign) and s.name == expa.args[0] and s.op.name == "subtract"), None)
    if suba is None or len(suba.args) != 2 or suba.args[1] != maxacc:
        return None
    cone = _score_cone(body, suba.args[0])
    if cone is None:
        return None
    return state, suba.args[0], cone[0]


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
                maxacc, score, cone, ld = mx
                se = _sumexp(nxt, maxacc)
                if se is not None and _cone_canon(cone, score, s.axis.name) == _cone_canon(se[2], se[1], nxt.axis.name):
                    sumacc = se[0]
                    # The generated streaming merge (``base``-``Accum`` folds + ψ rescales — the
                    # exp-family program over ``(m, d)`` with injected terms ``(s, 1.0)``) sits in
                    # the loop body directly; the loop is stamped TWISTED, and the algebra is the
                    # body itself (``Fold.from_loop`` reconstructs it). No explicit ``Init``
                    # seeds — ``Loop.render`` seeds each fold ``Accum`` from ``op.identity``
                    # ((−inf, 0)). A bare-``Load`` score keeps the historical renamed spelling; a
                    # composite score cone rides verbatim as the streaming prefix.
                    if len(cone) == 1:
                        src = f"{maxacc}__osin"
                        head: tuple = (Load(name=src, input=ld.input, index=ld.index),)
                    else:
                        src = score
                        head = tuple(cone)
                    fused = Loop(
                        axis=s.axis,
                        body=Body.coerce((*head, *exp_merge((maxacc, sumacc), (src, 1.0), key=maxacc))),
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
