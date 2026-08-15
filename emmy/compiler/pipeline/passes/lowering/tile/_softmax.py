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
``(m, d, o…)`` pass and the probability matrix never materializes. When the expectation
folds sit inside a following free output sweep (the fused-matmul spelling — one fold per
output column), the pair and its held-back pure companions sink into that sweep first, and
the per-cell join proceeds there.
"""

from __future__ import annotations

import re
from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.axis import AxisRole
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.sigma import Sigma
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


def _score_head(maxacc: str, score: str, cone: list, ld: Load) -> tuple[str, tuple]:
    """The fused loop's streaming score prefix. A bare-``Load`` score keeps the historical
    renamed spelling; a composite score cone rides verbatim."""
    if len(cone) == 1:
        src = f"{maxacc}__osin"
        return src, (Load(name=src, input=ld.input, index=ld.index),)
    return score, tuple(cone)


def _twist(s: Loop, nxt: Loop, maxacc: str, sumacc: str, score: str, cone: list, ld: Load, canon: str, rest: list) -> tuple[list, int]:
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
    scan. A pair with no adjacent channels tries the sink (:func:`_sink`) before falling back
    to the plain ``(m, d)`` spelling — byte-identical to the historical pair emission."""
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
    sunk = _sink(s, nxt, maxacc, sumacc, canon, rest)
    if sunk is not None:
        return sunk
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


def _emit_channels(s: Loop, maxacc: str, sumacc: str, score: str, cone: list, ld: Load, region: list) -> list | None:
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


def _sink(s: Loop, nxt: Loop, maxacc: str, sumacc: str, canon: str, rest: list) -> tuple[list, int] | None:
    """Sink a pair whose expectation channels sit inside a following free output sweep (the
    fused-matmul spelling: one same-extent additive fold per output column) into that sweep,
    then fuse per cell. The pair loops and the held-back pure siblings (the hoisted normalize)
    move to the sweep's head; commits only when the per-cell pairing actually joins a channel
    (≥3 carried states), and only when nothing after the sweep reads a moved binding."""
    pending: list = []
    for idx, t in enumerate(rest):
        if isinstance(t, (Load, Assign, Select)):
            pending.append(t)
            continue
        if isinstance(t, Loop) and not t.is_reduce:
            joinable = any(
                isinstance(u, Loop) and u.is_reduce and u.axis.extent == s.axis.extent and _channel(u, maxacc, sumacc, canon) is not None
                for u in t.body
            )
            if not joinable:
                return None
            moved = {maxacc, sumacc} | {d for p in pending for d in p.defines()}
            if moved & _deep_reads(rest[idx + 1 :]):
                return None
            nb, ch = _fuse(Body.coerce((s, nxt, *pending, *t.body)))
            fused = next((st for st in nb if isinstance(st, Loop) and st.role is AxisRole.TWISTED), None)
            if not ch or fused is None or sum(1 for st in fused.body if isinstance(st, Accum)) < 3:
                return None
            return [replace(t, body=nb)], idx + 1
        return None
    return None


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
                    emitted, extra = _twist(s, nxt, maxacc, se[0], score, cone, ld, canon, stmts[i + 2 :])
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
