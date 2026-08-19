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

The carrier is N-channel (:func:`exp_merge` takes a names tuple), so the pairing joins any
number of EXPECTATION channels beyond the ``(m, d)`` pair: a further sibling additive fold
over the same extent whose lifted value is (the pair's own per-element weight
``exp(score − m)``) × (a value cone free of the pair's states) folds into the SAME twisted
loop as one more carried state — pivot ``m``, denominator ``d``, one expectation per joined
fold. A loop-invariant multiplicative factor on such a fold (softmax's hoisted
``1/d`` normalize) is split off first (``Σₖ c·xₖ = c·Σₖ xₖ`` — :func:`split_invariant_factors`)
and multiplies the state back after the loop, so a fused softmax·V region streams as one
``(m, d, o…)`` pass and the probability matrix never materializes.

A channel joins only where it is a SIBLING of the pair. When the expectation folds sit inside a
following free output sweep instead (the fused-matmul spelling — one fold per output column), the
pair stays at its own level and is that sweep's per-ROW statistic: the sweep then reads as ONE
computed-A contraction whose cone is ``exp(score − m)·(1/d)`` and whose source is the pair, through
the same binding the norm→linear edge uses (``_atomize.bind_prologue_contraction``). That is what
puts the region on the contraction schedule catalog — the warp/mma tier, the staged transports and
split-K — instead of a per-cell serial fold, and it is why the pair must NOT be pushed into the
sweep: inside it the statistic is recomputed once per output column.

A pair's score is whatever its step produces. It is usually a ``Load``; when the step COMPOSES the
score instead (loop fusion spliced the ``Q·Kᵀ`` producer into it), the cone's one source is that
producer NODE and the pairing compares the two passes' cones by CONTENT — every bound name and
each nested loop's iteration var α-renamed in walk order (:func:`_cone_canon`), since two
separately-traced copies of one score differ in exactly those. The fused stream then carries ONE
copy of the producer where the pair carried two.
"""

from __future__ import annotations

import re
from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.axis import AxisRole
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.pure import component_ops
from emmy.compiler.ir.pure.carrier import exp_merge
from emmy.compiler.ir.pure.fold import Fold, operand_name
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Select
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
    # The step's own producer NODES ride back as body members (the COLLAPSE view,
    # :meth:`Fold.demoted` — each edge inline as the structural node, before its first read): a
    # composed step (attention's per-key score contraction) reads as an operand edge, and the
    # pairing's cone walk reads a producer the same way whether it is a ``Load`` or a node.
    return f.combine.results[0], f.lift.results[0], list(f.demoted().lift.body)


def _score_cone(body: list, result: str) -> tuple[list, object] | None:
    """The pure elementwise cone producing ``result`` from a pair-loop body: the
    :meth:`Body.backward_cone` members (body order) restricted to ``Load``/``Assign``/``Select``
    reaching exactly ONE SOURCE — a ``Load``, or a producer NODE the step composes (attention's
    per-key score contraction), which leads the returned cone. Names defined outside the loop (an
    enclosing-scope value, an axis var) surface as external reads and stay free. ``None`` when the value depends on
    anything else — the pairing then declines and the cell keeps its current reading."""
    nodes = {operand_name(s): s for s in body if isinstance(s, Fold)}
    if result in nodes:
        return [nodes[result]], nodes[result]  # the bare node score — the degenerate cone
    cone = Body.coerce(tuple(s for s in body if not isinstance(s, (Accum, Fold)))).backward_cone([result])
    stmts = list(cone.members)
    if not stmts or any(not isinstance(s, (Load, Assign, Select)) for s in stmts):
        return None
    used = [n for n in nodes if n in {a for s in stmts for a in s.deps()}]
    srcs = [s for s in stmts if isinstance(s, Load)] + [nodes[n] for n in used]
    if len(srcs) != 1:
        return None
    return [*(nodes[n] for n in used), *stmts], srcs[0]


def _cone_names(stmts: list, mapping: dict) -> None:
    """Number every name a cone BINDS, deep — the stmts' defs plus each nested loop's iteration
    var — in walk order. A composed cone's producer node carries its own axis and temps, and two
    separately-traced copies of one score differ in exactly those, so identity has to see through
    them (the key digest cannot: axis names are recognition-canonical and part of it)."""
    for s in stmts:
        if isinstance(s, Loop):
            mapping.setdefault(s.axis.name, f"_c{len(mapping)}")
        for n in s.defines():
            mapping.setdefault(n, f"_c{len(mapping)}")
        for b in s.nested():
            _cone_names(list(b), mapping)


def _cone_canon(stmts: list, result: str, axis_name: str) -> str:
    """An α-renamed rendering of a score cone — every bound name canonicalizes in walk order
    (:func:`_cone_names`), the loop axis to a fixed placeholder, free names verbatim — so the two
    pair loops' recomputed score cones compare by content, exactly as the byte-identity λ gate
    compares lift bodies. A producer NODE in the cone renders through its own lowered nest, so a
    composed score compares like any other."""
    flat = [t for s in stmts for t in (s.lower() if isinstance(s, Fold) else [s])]
    mapping = {axis_name: "_ax"}
    _cone_names(flat, mapping)
    text = repr(tuple(flat)) + "|" + mapping.get(result, result)
    for old, new in mapping.items():
        text = re.sub(f"'{re.escape(old)}'", f"'{new}'", text)
    return text


def same_score_cone(a, b, a_axis: str, b_axis: str) -> bool:
    """Whether two score cones are the SAME program modulo their own bound names and the axis each
    streams — the pairing's own content comparison (:func:`_cone_canon`), asked of two nodes that
    reached lowering separately.

    The online-softmax pair carries one score per pass; the pairing matched them at recognition, but
    a later pass may rewrite one side alone (a split-K partition re-indexes the weight's keys while
    the statistic keeps spanning the whole axis), and a lowering that folds the two passes into one
    sweep may only do so while they still read the same keys."""
    return _cone_canon([a], operand_name(a), a_axis) == _cone_canon([b], operand_name(b), b_axis)


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


def split_invariant_factors(body: list, value: str, axis_name: str) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """The general additive-fold factor split ``Σₖ c·xₖ = c·Σₖ xₖ``: flatten the two-arg
    ``multiply`` spine defining ``value`` over a reduce-loop body and split the leaf factor
    names into ``(c — the loop-invariant factors, names defined outside the body; the
    loop-varying leaves)``, left-to-right. The loop axis itself counts as loop-varying. The
    spine must be private to the product — a spine temp read by any other body stmt (or a
    non-binary multiply) returns ``None``, and the caller keeps the loop's current reading.
    A bare leaf is the degenerate product: ``((), (value,))``."""
    defs: dict[str, object] = {n: s for s in body for n in s.defines()}
    spine: list[str] = []
    leaves: list[str] = []

    def flatten(n: str) -> bool:
        d = defs.get(n)
        if isinstance(d, Assign) and d.op.name == "multiply":
            if len(d.args) != 2:
                return False
            spine.append(n)
            return flatten(d.args[0]) and flatten(d.args[1])
        leaves.append(n)
        return True

    if not flatten(value):
        return None
    spine_reads = {n for n in spine if n != value}
    for s in body:
        if not (isinstance(s, Assign) and s.name in spine) and set(s.deps()) & spine_reads:
            return None
    inv = tuple(n for n in leaves if n not in defs and n != axis_name)
    return inv, tuple(n for n in leaves if n in defs or n == axis_name)


def _channel(loop: Loop, maxacc: str, sumacc: str, canon: str) -> tuple[str, tuple, tuple, tuple] | None:
    """Read ``loop`` as an EXPECTATION channel joining an online-softmax pair: an additive
    single-state fold whose lifted value is a product of the pair's own per-element weight —
    ``exp(score − maxacc)`` with a score cone α-equal to the pair's (``canon``) — optional
    loop-invariant factors (split off by :func:`split_invariant_factors`; they multiply the
    state back after the loop), and a residual value cone free of the pair's states. Returns
    ``(state, invariant factors, value-cone stmts, value factor names)``, or ``None`` — the
    fold then keeps its current reading."""
    read = _fold_of(loop, "add")
    if read is None:
        return None
    state, value, body = read
    split = split_invariant_factors(body, value, loop.axis.name)
    if split is None:
        return None
    inv, local = split
    defs = {n: s for s in body for n in s.defines()}
    weights: list[str] = []
    values: list[str] = []
    for n in local:
        d = defs.get(n)
        sub = defs.get(d.args[0]) if isinstance(d, Assign) and d.op.name == "exp" else None
        if (
            isinstance(sub, Assign)
            and sub.op.name == "subtract"
            and len(sub.args) == 2
            and sub.args[1] == maxacc
            and (cone := _score_cone(body, sub.args[0])) is not None
            and _cone_canon(cone[0], sub.args[0], loop.axis.name) == canon
        ):
            weights.append(n)
        else:
            values.append(n)
    if len(weights) != 1:
        return None
    vcone = Body.coerce(tuple(s for s in body if not isinstance(s, Accum))).backward_cone(values)
    stmts = list(vcone.members)
    if any(not isinstance(s, (Load, Assign, Select)) for s in stmts):
        return None
    if {maxacc, sumacc} & {d for s in stmts for d in s.deps()}:
        return None  # the value must be free of the pair's running states
    return state, inv, tuple(stmts), tuple(values)


def _deep_reads(stmts) -> set[str]:
    """Every SSA name read anywhere in ``stmts`` (deep — through ``deps`` + nested bodies)."""
    out: set[str] = set()
    for s in stmts:
        out.update(s.deps())
        for b in s.nested():
            out |= _deep_reads(list(b))
    return out


def _score_head(maxacc: str, score: str, cone: list, ld) -> tuple[str, tuple]:
    """The fused loop's streaming score prefix. A bare-``Load`` score keeps the historical
    renamed spelling; a composite score cone — and a cone whose one source is a producer NODE
    (attention's per-key score contraction) — rides verbatim."""
    if len(cone) == 1 and isinstance(ld, Load):
        src = f"{maxacc}__osin"
        return src, (Load(name=src, input=ld.input, index=ld.index),)
    # The emitted prefix is LOOP IR: a producer node in the cone rides as its own loop nest (a
    # ``Fold`` may only be a child of a ``Fold``, never a member of a raw ``Loop`` body). The
    # twisted reading hoists it straight back to an operand edge (``_fromloop._hoist_step_nodes``).
    return score, tuple(t for s in cone for t in (s.lower() if isinstance(s, Fold) else [s]))


def _twist(s: Loop, maxacc: str, sumacc: str, score: str, cone: list, ld, canon: str, rest: list) -> tuple[list, int]:
    """Emit the fused streaming loop for a matched ``(rowmax, Σexp)`` pair, joining every
    EXPECTATION channel that follows. Returns ``(emitted stmts, extra stmts consumed from
    ``rest``)``.

    The generated streaming merge (``base``-``Accum`` folds + ψ rescales — the exp-family
    program over the carried states with injected terms ``(score, 1.0, value…)``) sits in
    the loop body directly; the loop is stamped TWISTED, and the algebra is the body itself
    (``Fold.from_loop`` reconstructs it). No explicit ``Init`` seeds — ``Loop.render`` seeds
    each fold ``Accum`` from ``op.identity``.

    The channel scan walks the adjacent siblings: a same-extent additive fold reading as an
    expectation channel joins; pure stmts are held back and re-emitted after the fused loop
    (they may read the pair's final states — the hoisted normalize); anything else stops the
    scan. A pair with no adjacent channels keeps the plain ``(m, d)`` spelling — byte-identical
    to the historical pair emission — and STAYS AT ITS OWN LEVEL: when the expectation folds sit
    inside a following free output sweep (the fused-matmul spelling), the pair is that sweep's
    per-ROW statistic, and the sweep binds as one computed-A contraction over it
    (``_atomize.bind_prologue_contraction`` — the same seam the norm→linear edge uses)."""
    region: list = []
    consumed = 0
    nchan = 0
    for idx, t in enumerate(rest):
        if isinstance(t, Loop):
            if not (t.is_reduce and t.axis.extent == s.axis.extent):
                break
            ch = _channel(t, maxacc, sumacc, canon)
            if ch is None:
                break
            region.append((t, ch))
            nchan += 1
            consumed = idx + 1
        elif isinstance(t, (Load, Assign, Select)):
            region.append((None, t))
        else:
            break
    if nchan:
        emitted = _emit_channels(s, maxacc, sumacc, score, cone, ld, region[: _last_channel(region)])
        if emitted is not None:
            return emitted, consumed
    src, head = _score_head(maxacc, score, cone, ld)
    fused = Loop(
        axis=s.axis,
        body=Body.coerce((*head, *exp_merge((maxacc, sumacc), (src, 1.0), key=maxacc))),
        role=AxisRole.TWISTED,
    )
    return [fused], 0


def _last_channel(region: list) -> int:
    """Index just past the last joined channel — trailing held-back pure stmts stay outside."""
    return max(i + 1 for i, (t, _) in enumerate(region) if t is not None)


def _emit_channels(s: Loop, maxacc: str, sumacc: str, score: str, cone: list, ld, region: list) -> list | None:
    """Build the N-channel fused loop + its epilogue for a pair and its joined channels.
    ``region`` interleaves held-back pure stmts (re-emitted after the loop, original order)
    with ``(channel loop, channel read)`` entries. Declines (``None``) when a held-back
    stmt's binding is read inside the fused loop — re-emitting it after would break SSA."""
    src, head = _score_head(maxacc, score, cone, ld)
    states = [maxacc, sumacc]
    terms: list = [src, 1.0]
    prefix = list(head)
    epilogue: list = []
    for t, entry in region:
        if t is None:
            epilogue.append(entry)
            continue
        state, inv, vstmts, values = entry
        rename = {t.axis.name: s.axis.name}
        sigma = Sigma({t.axis.name: Var(s.axis.name)})
        prefix += [st.rewrite(lambda n: rename.get(n, n), sigma) for st in vstmts]  # noqa: B023
        values = tuple(rename.get(n, n) for n in values)
        if not values:
            term: str | float = 1.0
        elif len(values) == 1:
            term = values[0]
        else:  # a multi-factor value cone rebuilds its product in the streaming prefix
            term = values[0]
            for k, v in enumerate(values[1:]):
                nm = f"{state}__v{k}"
                prefix.append(Assign(nm, "multiply", (term, v)))
                term = nm
        st_name = f"{state}__sum" if inv else state
        states.append(st_name)
        terms.append(term)
        cur = st_name  # the invariant factors multiply the carried sum back (Σ c·x = c·Σ x)
        for k, c in enumerate(inv):
            nm = state if k == len(inv) - 1 else f"{state}__c{k}"
            epilogue.append(Assign(nm, "multiply", (cur, c)))
            cur = nm
    fused = Loop(axis=s.axis, body=Body.coerce((*prefix, *exp_merge(tuple(states), tuple(terms), key=maxacc))), role=AxisRole.TWISTED)
    held_defs = {d for t, entry in region if t is None for d in entry.defines()}
    if held_defs & _deep_reads(list(fused.body)):
        return None
    return [fused, *epilogue]


def _fuse(body: Body) -> tuple[Body, bool]:
    """Recurse into nested ``Loop`` bodies; fuse any adjacent ``(rowmax, sum-of-exp)``
    reduce pair over the same input + reduce extent — together with every expectation
    channel that joins it (:func:`_twist`) — into one streaming online-softmax loop: a
    ``TWISTED`` reduce ``Loop``, its body the score cone + value cones + the carrier's
    dissolved streaming ``merge`` (``base``-``Accum`` folds + ψ rescales)."""
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
                canon = _cone_canon(cone, score, s.axis.name)
                if se is not None and canon == _cone_canon(se[2], se[1], nxt.axis.name):
                    emitted, extra = _twist(s, maxacc, se[0], score, cone, ld, canon, stmts[i + 2 :])
                    out.extend(emitted)
                    changed = True
                    i += 2 + extra
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
