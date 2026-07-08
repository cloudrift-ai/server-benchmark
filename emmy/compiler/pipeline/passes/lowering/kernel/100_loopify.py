"""Re-roll parallel per-fragment statement runs into ``#pragma unroll`` loops (``LOOPIFY`` — pin-only,
off by default).

A generic **loop re-roller**, iterated to a fixpoint so nested runs collapse to nested loops, plus a
pin-gated **pointwise-chain fusion** (:func:`_fuse_chains`). The flash mma body emits long runs of
near-identical straight-line statements — the per-fragment ``FragmentApply`` epilogue (``O_i_f{t} *= α``
/ ``/= l``, the ``sacc_f`` QK scale), the ``P@V`` ``load``+``mma`` pairs, the fragment ``RegStore``s,
the A-fragment loads, and the nested ``QK`` score contraction (4 K-chunks × 2 N-atoms). Each run is N
repetitions of a fixed K-statement
window that differ ONLY in (a) a **contiguous fragment family** index (``O_i_f0 … O_i_f{N-1}``,
``_sacc_a0 …``, or an already-arrayed ``fam[0] … fam[N-1]``) and (b) **affine address offsets** (a
store's / load's ``+ t*8`` stride). This pass detects such a run, arrays each family into one
declaration, and folds the run into a single ``StridedLoop`` over ``_r{depth}`` — family refs become
``fam[_r]``, affine offsets ``base + _r*step``.

**Why it is always correct:** a congruent run, unrolled, executes its iterations in the SAME order as the
original straight-line statements — so the ``#pragma unroll`` loop is byte-for-byte the same computation
(nvcc unrolls the pragma → identical SASS). Correctness reduces to the structural matcher :func:`_reroll`
succeeding: it returns a parameterized template ONLY when every window w is window 0 with each family
suffix → ``w`` and each affine literal → ``base + w*step``. Nothing about data dependence enters — the
transform is a readability-only re-spelling for blog / ``--ir cuda`` inspection, decoupled from
production codegen.

The matcher is affine-aware, which is what lets it see through the codegen's **constant-folded zero**:
fragment 0's KV offset ``kv0`` is really ``kv0 + 0*8`` with the ``+ 0`` simplified away, so it is
structurally *different* from fragment 1's ``kv0 + 8`` — :func:`_reroll` peels a trailing ``± Literal``
off both (``(kv0, 0)`` vs ``(kv0, 8)``) and reconciles them as one affine family. That, plus the fixpoint
(each pass gets a fresh ``_r{depth}`` and re-parses already-arrayed ``fam[i]`` refs), turns the nested QK
contraction into two nested ``#pragma unroll`` loops: pass k re-rolls the inner N-atoms, pass k+1 sees
the resulting sibling loops as a run and wraps them.

Modeled on ``085_fast_exp`` (the pin-only kernel-IR peephole mold): ``PATTERN = [Pattern("root",
KernelOp)]`` and a ``rewrite`` that stamps the knob and returns the body unchanged when off — so the
**entire default pipeline runs LOOPIFY-off and is byte-identical** (no golden / snapshot churn). The knob
is an INT = the minimum run length ``N`` to re-roll (``LOOPIFY.read_int(0)``): ``0`` / unset / ``< 2`` →
off; ``EMMY_LOOPIFY=4`` catches the 8-long runs while skipping the 2-long QK scale and ``subtract``→
``exp`` and the K-chunk × N-atom nest; ``EMMY_LOOPIFY=2`` re-rolls every run ≥ 2. The fused loop's
``#pragma unroll`` rides the ``EMMY_UNROLL`` budget (``unroll_ok_n``).

Only ``RegFragment``-declared families are arrayable ``float[4]`` / ``unsigned[N]`` fragments; a run over
scalar carriers (``m_i`` / the ``_rmx`` row-reduce temps) or an inline-declared fragment (``_p_f``) is
rejected. The arrayed decl is sized to the family's member count, never a single run's length.

Chain fusion (also pin-gated, run before the re-roll): a ``FragmentApply`` whose result is immediately
consumed by an in-place UNARY ``FragmentApply`` on the same fragment (``p <- s − m`` then ``p *= exp(p)``)
folds into one node carrying the tail on its ``post`` field, so — with the element-loop render each
``FragmentApply`` already uses — the softmax exp renders as ``p[_e] = expf(s[_e] − m)`` rather than a
subtract loop feeding an exp loop. Valid because the tail reads only ``p`` and the next stmt is its sole
consumer (no intermediate use to preserve).
"""

from __future__ import annotations

import re
from dataclasses import fields, is_dataclass, replace
from itertools import count

from emmy.compiler.graph import Node
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal, Var
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.kernel.ir import FRAG, FragmentApply, RegFragment
from emmy.compiler.ir.stmt import Body, StridedLoop
from emmy.compiler.pipeline import Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.lowering.kernel._atom import unroll_ok_n
from emmy.compiler.pipeline.search.space import LOOPIFY

PATTERN = [Pattern("root", KernelOp)]

_KMAX = 2  # max statements per iteration window (1 = FragmentApply / RegStore; 2 = load+mma, subtract+exp)
_SUFFIX = re.compile(r"^(.*?)(?:\[(\d+)\]|(\d+))$")  # matches ``fam7`` and the arrayed ``fam[7]``
_FAIL = object()  # distinct failure sentinel — ``None`` is a legitimate invariant field value (a guard / epilogue)


def rewrite(root: Node) -> KernelOp | None:
    op: KernelOp = root.op
    if LOOPIFY.name in op.knobs:
        raise RuleSkipped("LOOPIFY already decided (idempotence via knob)")
    n = LOOPIFY.read_int(0)
    knobs = {**op.knobs, LOOPIFY.name: n}
    if n < 2:  # 0 / unset / a lone iteration → off, byte-identical
        return KernelOp(body=op.body, name=op.name, knobs=knobs)
    # (2) Fuse each pointwise fragment chain (``p <- s − m`` then ``p *= exp(p)``) into one node
    # carrying a ``post`` tail, so it renders as ``p[_e] = expf(s[_e] − m)`` instead of two loops.
    body = _fuse_chains(op.body)
    # Iterate to a fixpoint: each pass re-rolls the runs it can see, then the next pass (fresh loop var)
    # sees the loops it produced as a new run and nests them. Terminates — every pass either shrinks the
    # flat statement count or leaves the body unchanged.
    for depth in count():
        loopvar = f"_r{depth}"
        new_body = _one_pass(body, n, loopvar)
        if new_body == body:
            break
        body = new_body
    return KernelOp(body=body, name=op.name, knobs=knobs)


def _is_unary_post(s, out: str, layout) -> bool:
    """``s`` is an in-place ``FragmentApply`` that applies ONE unary op to ``out`` (``out *= g(out)``)
    — the fusable tail of a pointwise chain (the softmax ``p *= exp(p)`` after ``p <- s − m``)."""
    return isinstance(s, FragmentApply) and s.in_place and s.out == out and s.layout == layout and s.kinds == (FRAG,) and s.args == (out,)


def _fuse_chains(body: Body) -> Body:
    """Merge a maximal chain of same-fragment pointwise ops — ``p <- f(args)`` immediately followed by
    ``p *= g(p)`` (, then ``p *= h(p)`` …) — into one ``FragmentApply`` whose ``post`` carries the
    unary tail. Correct because each tail op reads ONLY ``p`` and the next stmt is its sole consumer,
    so there is no intermediate use to preserve. Recurses into nested bodies."""
    stmts = list(body)
    out: list = []
    i = 0
    while i < len(stmts):
        s = stmts[i]
        nested = s.nested()
        if nested:
            s = s.with_bodies(tuple(_fuse_chains(b) for b in nested))
        if isinstance(s, FragmentApply):
            while i + 1 < len(stmts) and _is_unary_post(stmts[i + 1], s.out, s.layout):
                nxt = stmts[i + 1]
                s = replace(s, post=s.post + (nxt.op, *nxt.post))
                i += 1
        out.append(s)
        i += 1
    return Body(tuple(out))


def _one_pass(body: Body, min_run: int, loopvar: str) -> Body:
    sizes: dict[str, int] = {}  # fragment stem → member count
    already: set[str] = set()  # stems already collapsed to a ``count``-arrayed RegFragment
    _frag_info(body, sizes, already)
    frag_stems = set(sizes)
    used: set[str] = set()  # the fragment families some run re-rolls
    _scan(body, min_run, frag_stems, loopvar, used)
    to_array = {stem: sizes[stem] for stem in used if stem not in already}  # newly-arrayed this pass
    new = _rewrite_tree(body, min_run, frag_stems, to_array, loopvar)
    if to_array:
        # Point every scalar reference OUTSIDE a re-rolled loop at the arrayed decl (``fam{j}`` →
        # ``fam[j]`` — the row-reduce / repack / non-re-rolled fragment ops). The arrayed RegFragment
        # (name ``fam``) and the loops' ``fam[_r]`` are absent from the map.
        from emmy.compiler.ir.sigma import Sigma  # noqa: PLC0415
        from emmy.compiler.ir.stmt.base import _axis_identity  # noqa: PLC0415
        from emmy.compiler.ir.stmt.passes import rewrite as _rw  # noqa: PLC0415

        rename_map = {f"{stem}{j}": f"{stem}[{j}]" for stem, cnt in to_array.items() for j in range(cnt)}
        rn = lambda x: rename_map.get(x, x)  # noqa: E731
        new = Body(tuple(_rw(s, rn, Sigma.IDENTITY, _axis_identity) for s in new))
    return new


def _split(name: str) -> tuple[str, int] | None:
    """``"O_i_f7"`` / ``"O_i_f[7]"`` → ``("O_i_f", 7)``; ``None`` when there is no trailing integer
    (a loop-var-indexed ``fam[_r0]`` is not a family member)."""
    m = _SUFFIX.match(name)
    if m is None:
        return None
    return m.group(1), int(m.group(2) if m.group(2) is not None else m.group(3))


# --------------------------------------------------------------------------- #
# The matcher: given the SAME position across N windows, return the template
# value that generalizes them (family index / affine offset / recursed
# container), or None if they don't form a re-rollable pattern.
# --------------------------------------------------------------------------- #


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _peel(v):
    """Split an Expr into ``(base, trailing_int_const)`` — ``kv0`` → ``(kv0, 0)``, ``kv0 + 8`` →
    ``(kv0, 8)`` — so a constant-folded ``+ 0`` reconciles with a sibling's ``+ c``."""
    if isinstance(v, BinaryExpr) and v.op == "+" and isinstance(v.right, Literal) and _is_int(v.right.value):
        return v.left, v.right.value
    return v, 0


def _reroll(vals: list, loopvar: str, frag_stems: set[str]):
    """The template value generalizing this position across the run, or :data:`_FAIL`."""
    v0 = vals[0]
    if all(v == v0 for v in vals):
        return v0  # invariant across the run (may itself be ``None`` — a guard / epilogue)
    if all(isinstance(v, str) for v in vals):
        parsed = [_split(v) for v in vals]
        if all(p is not None for p in parsed) and len({p[0] for p in parsed}) == 1 and [p[1] for p in parsed] == list(range(len(vals))):
            stem = parsed[0][0]
            return f"{stem}[{loopvar}]" if stem in frag_stems else _FAIL  # only arrayable fragment families
        return _FAIL
    if all(isinstance(v, Literal) for v in vals):
        if all(isinstance(v.value, int) and not isinstance(v.value, bool) for v in vals) and len({v.dtype for v in vals}) == 1:
            base, step = vals[0].value, vals[1].value - vals[0].value
            if all(v.value == base + i * step for i, v in enumerate(vals)):
                return _affine(base, step, loopvar, vals[0].dtype)  # step != 0 here (values differ), never None
        return _FAIL
    if all(isinstance(v, Body) for v in vals) and len({len(v) for v in vals}) == 1:
        return _recurse_seq(vals, loopvar, frag_stems, Body)
    if all(isinstance(v, tuple) for v in vals) and len({len(v) for v in vals}) == 1:
        return _recurse_seq(vals, loopvar, frag_stems, tuple)
    if all(type(v) is type(v0) for v in vals) and (isinstance(v0, Expr) or is_dataclass(v0)):
        kw = {}
        for f in fields(v0):
            if not f.init:
                continue
            sub = _reroll([getattr(v, f.name) for v in vals], loopvar, frag_stems)
            if sub is _FAIL:
                return _FAIL
            kw[f.name] = sub
        return type(v0)(**kw)
    # folded-zero / affine tail: each val is ``base`` or ``base + Literal(c)`` — peel and reconcile.
    peeled = [_peel(v) for v in vals]
    consts = [c for _, c in peeled]
    if any(c != 0 for c in consts):
        base_t = _reroll([b for b, _ in peeled], loopvar, frag_stems)
        step = consts[1] - consts[0]
        if base_t is not _FAIL and all(c == consts[0] + i * step for i, c in enumerate(consts)):
            tail = _affine(consts[0], step, loopvar, "int")
            return base_t if tail is None else BinaryExpr("+", base_t, tail)
    return _FAIL


def _recurse_seq(vals: list, loopvar: str, frag_stems: set[str], ctor):
    cols = [_reroll([v[k] for v in vals], loopvar, frag_stems) for k in range(len(vals[0]))]
    if any(c is _FAIL for c in cols):
        return _FAIL
    return ctor(cols)


def _affine(base: int, step: int, loopvar: str, dtype: str):
    """``base + _r*step`` as an Expr — ``_r*step`` when ``base == 0``, ``Literal(base)`` when ``step ==
    0``, ``None`` when both are zero (a pure ``+ 0`` that folds away)."""
    if step == 0:
        return None if base == 0 else Literal(base, dtype)
    term = BinaryExpr("*", Var(loopvar), Literal(step, "int"))
    return term if base == 0 else BinaryExpr("+", Literal(base, dtype), term)


def _families_in(tmpl, loopvar: str, out: set[str]) -> None:
    """The family stems a template re-rolls — every ``{stem}[{loopvar}]`` string leaf."""
    suffix = f"[{loopvar}]"
    if isinstance(tmpl, str):
        if tmpl.endswith(suffix):
            out.add(tmpl[: -len(suffix)])
    elif isinstance(tmpl, tuple):
        for v in tmpl:
            _families_in(v, loopvar, out)
    elif isinstance(tmpl, Expr) or is_dataclass(tmpl):
        for f in fields(tmpl):
            if f.init:
                _families_in(getattr(tmpl, f.name), loopvar, out)


# --------------------------------------------------------------------------- #
# Detection + rewrite.
# --------------------------------------------------------------------------- #


def _has_regfrag(stmts: list, start: int, count_: int) -> bool:
    return any(isinstance(stmts[start + t], RegFragment) for t in range(count_) if start + t < len(stmts))


def _frag_info(body: Body, sizes: dict[str, int], already: set[str]) -> None:
    """Fragment stem → member count, and the stems already collapsed to a ``count``-arrayed decl."""
    for s in body:
        if isinstance(s, RegFragment):
            if s.count > 1:  # already arrayed — the name IS the stem
                sizes[s.name] = s.count
                already.add(s.name)
            else:
                sp = _split(s.name)
                if sp is not None:
                    sizes[sp[0]] = max(sizes.get(sp[0], 0), sp[1] + 1)
        for b in s.nested():
            _frag_info(b, sizes, already)


def _detect(stmts: list, min_run: int, frag_stems: set[str], loopvar: str) -> list[tuple[int, int, int, object]]:
    """Maximal re-rollable runs ``(start, K, N, template)`` with ``N >= max(2, min_run)``, each varying
    at least one fragment family. Runs start on the smallest window size ``K``; ``RegFragment`` decls are
    transparent (collapsed by arraying, never looped)."""
    runs: list[tuple[int, int, int, object]] = []
    i, total = 0, len(stmts)
    while i < total:
        if isinstance(stmts[i], RegFragment):
            i += 1
            continue
        chosen = None
        for k in range(1, _KMAX + 1):
            if i + 2 * k > total or _has_regfrag(stmts, i, 2 * k):
                continue
            best_n, best_t = 0, None
            for cand in range(2, (total - i) // k + 1):
                if _has_regfrag(stmts, i, cand * k):
                    break
                windows = [tuple(stmts[i + w * k : i + (w + 1) * k]) for w in range(cand)]
                t = _reroll(windows, loopvar, frag_stems)
                if t is _FAIL:
                    break
                best_n, best_t = cand, t
            fams: set[str] = set()
            if best_t is not None:
                _families_in(best_t, loopvar, fams)
            if best_n >= 2 and fams:  # a real run must vary a fragment family
                chosen = (k, best_n, best_t)
                break
        if chosen is None:
            i += 1
            continue
        k, n, tmpl = chosen
        if n >= min_run:
            runs.append((i, k, n, tmpl))
        i += k * n
    return runs


def _scan(body: Body, min_run: int, frag_stems: set[str], loopvar: str, used: set[str]) -> None:
    stmts = list(body)
    for _i, _k, _n, tmpl in _detect(stmts, min_run, frag_stems, loopvar):
        _families_in(tmpl, loopvar, used)
    for s in stmts:
        for b in s.nested():
            _scan(b, min_run, frag_stems, loopvar, used)


def _rewrite_tree(body: Body, min_run: int, frag_stems: set[str], to_array: dict[str, int], loopvar: str) -> Body:
    """Replace each run with its fused loop, collapse each newly-arrayed family's scalar ``RegFragment``
    decls into one ``count``-arrayed decl, and recurse into nested bodies."""
    stmts = list(body)
    runs = {i: (k, n, tmpl) for i, k, n, tmpl in _detect(stmts, min_run, frag_stems, loopvar)}
    out: list = []
    i = 0
    while i < len(stmts):
        if i in runs:
            k, n, tmpl = runs[i]
            axis = Axis(name=loopvar, extent=n)
            out.append(StridedLoop(axis=axis, start=Literal(0, "int"), step=Literal(1, "int"), body=Body(tmpl), unroll=unroll_ok_n(n)))
            i += k * n
            continue
        s = stmts[i]
        if isinstance(s, RegFragment) and s.count == 1:
            sp = _split(s.name)
            if sp is not None and sp[0] in to_array:
                stem, idx = sp
                if idx == 0:  # array at the family's first decl; drop the siblings (idx > 0)
                    out.append(RegFragment(name=stem, role=s.role, shape=s.shape, dtype=s.dtype, count=to_array[stem]))
                i += 1
                continue
        nested = s.nested()
        if nested:
            s = s.with_bodies(tuple(_rewrite_tree(b, min_run, frag_stems, to_array, loopvar) for b in nested))
        out.append(s)
        i += 1
    return Body(tuple(out))
