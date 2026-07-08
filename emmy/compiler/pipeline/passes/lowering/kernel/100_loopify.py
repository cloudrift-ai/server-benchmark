"""Re-roll parallel per-fragment statement runs into ``#pragma unroll`` loops (``LOOPIFY`` — pin-only,
off by default).

A generic **loop re-roller**: the flash mma body emits long runs of near-identical straight-line
statements — the per-fragment ``FragmentApply`` epilogue (``O_i_f{t} *= α`` / ``/= l``, the ``sacc_f``
QK scale, the interleaved ``subtract``→``exp``), the ``P@V`` ``load``+``mma`` pairs (``O_i_f{t} += P @
V{t}``), and the 8 fragment ``RegStore``s. Each run is N repetitions of a fixed K-statement window that
differ ONLY in (a) a **contiguous fragment family** index (``O_i_f0 … O_i_f{N-1}``, ``_O_i__pv_b0 …``)
and (b) **affine address offsets** (the ``+ t*8`` N-column stride in a store's ``dst_index`` or a load's
``src_index``). This pass detects such a run, arrays each family into one declaration, and folds the run
into a single ``StridedLoop`` over ``_r`` — family refs become ``fam[_r]``, affine offsets ``base +
_r*step``.

**Why it is always correct:** a congruent run, unrolled, executes its iterations in the SAME order as the
original straight-line statements — so the ``#pragma unroll`` loop is byte-for-byte the same computation
(nvcc unrolls the pragma → identical SASS). Correctness therefore reduces to a purely structural check:
every window ``w`` must equal window 0 with each family suffix bumped to ``w`` and each affine literal to
``base + w*step`` (``_window_ok``). Nothing about data dependence enters — the transform is a
readability-only re-spelling for blog / ``--ir cuda`` inspection, decoupled from production codegen.

Modeled on ``085_fast_exp`` (the pin-only kernel-IR peephole mold): ``PATTERN = [Pattern("root",
KernelOp)]`` and a ``rewrite`` that stamps the knob and returns the body unchanged when off — so the
**entire default pipeline runs LOOPIFY-off and is byte-identical** (no golden / snapshot churn). The knob
is an INT = the minimum run length (iteration count ``N``) to re-roll (``LOOPIFY.read_int(0)``): ``0`` /
unset / ``< 2`` → off; ``EMMY_LOOPIFY=4`` catches the 8-long runs (O rescale / divide, P@V, stores)
while skipping the 2-long QK scale and ``subtract``→``exp``; ``EMMY_LOOPIFY=2`` re-rolls every run ≥ 2.
The fused loop's ``#pragma unroll`` rides the ``EMMY_UNROLL`` budget (``unroll_ok_n``).

The algorithm is node-type-agnostic: it walks each candidate window's Stmt/Expr trees generically
(``_collect`` / ``_transform``), so ``FragmentApply``, ``RegStore``, ``LdmatrixLoad`` and ``MmaSyncPtx``
runs are all handled by the same detect → verify → fuse core. ``RegFragment`` decls are never looped —
they are collapsed into one ``count``-arrayed decl for every family a run references.
"""

from __future__ import annotations

import re
from dataclasses import fields, is_dataclass

from emmy.compiler.graph import Node
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal, Var
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.kernel.ir import RegFragment
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Body, StridedLoop
from emmy.compiler.ir.stmt.base import _axis_identity
from emmy.compiler.ir.stmt.passes import rewrite as _rewrite
from emmy.compiler.pipeline import Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.lowering.kernel._atom import unroll_ok_n
from emmy.compiler.pipeline.search.space import LOOPIFY

PATTERN = [Pattern("root", KernelOp)]

_LOOPVAR = "_r"  # the re-roll index — distinct from the fragment lane vars ``_t`` / ``_g`` a RegStore re-declares
_KMAX = 2  # max statements per iteration window (1 = FragmentApply / RegStore; 2 = load+mma, subtract+exp)
_SUFFIX = re.compile(r"^(.*?)(\d+)$")


def rewrite(root: Node) -> KernelOp | None:
    op: KernelOp = root.op
    if LOOPIFY.name in op.knobs:
        raise RuleSkipped("LOOPIFY already decided (idempotence via knob)")
    n = LOOPIFY.read_int(0)
    knobs = {**op.knobs, LOOPIFY.name: n}
    if n < 2:  # 0 / unset / a lone iteration → off, byte-identical
        return KernelOp(body=op.body, name=op.name, knobs=knobs)
    # Only RegFragment-declared names are arrayable ``float[4]`` / ``unsigned[N]`` fragments — a run
    # over scalar SSA carriers (``m_i`` / ``l_i`` / the ``_rmx`` row-reduce temps) or a fragment a
    # ``FragmentApply`` declares inline (``_p_f``) must NOT be re-rolled (it would emit ``float
    # _rmx[0]`` / ``float _p_f[_r][4]`` — a nonsense array-indexed declaration).
    frag_sizes: dict[str, int] = {}  # fragment stem → member count (from its RegFragment decls)
    _frag_stems(op.body, frag_sizes)
    frag_stems = set(frag_sizes)
    used: set[str] = set()  # the fragment families some run actually re-rolls
    _scan(op.body, n, frag_stems, used)
    # The arrayed decl size is the FAMILY's member count, never a single run's length — a partial
    # run (a staged N-atom pair over an 8-member family) must still array the whole family so every
    # member (looped or not) indexes the same declaration.
    arrayed = {stem: frag_sizes[stem] for stem in used}
    body = _rewrite_tree(op.body, n, frag_stems, arrayed)
    if arrayed:
        # Point every scalar reference OUTSIDE a re-rolled loop at the arrayed decl: ``fam{j}`` →
        # ``fam[j]`` (the row-reduce / repack / any non-re-rolled fragment op). The arrayed
        # ``RegFragment`` (name ``fam``) and the loops' ``fam[_r]`` are absent from the map.
        rename_map = {f"{stem}{j}": f"{stem}[{j}]" for stem, count in arrayed.items() for j in range(count)}
        rn = lambda x: rename_map.get(x, x)  # noqa: E731
        body = Body(tuple(_rewrite(s, rn, Sigma.IDENTITY, _axis_identity) for s in body))
    return KernelOp(body=body, name=op.name, knobs=knobs)


# --------------------------------------------------------------------------- #
# Generic structural walk over flat Stmt / Expr trees (windows hold no nested
# bodies — only leaf compute stmts). ``_collect`` flattens to ordered atoms for
# comparison; ``_transform`` rebuilds with per-position substitutions. Both visit
# leaves in the same order, so a position index means the same thing to each.
# --------------------------------------------------------------------------- #


def _collect(value, names: list, lits: list, skel: list) -> None:
    """Flatten ``value`` into ordered ``names`` (str leaves), ``lits`` ((value, dtype) of ``Literal``
    leaves) and ``skel`` (everything structural: node types, tuple lengths, int/bool/enum scalars,
    opaque ops). Two windows are congruent only if their ``skel`` lists are equal."""
    if isinstance(value, Literal):
        lits.append((value.value, value.dtype))
    elif isinstance(value, str):
        names.append(value)
    elif isinstance(value, tuple):
        skel.append(("tup", len(value)))
        for v in value:
            _collect(v, names, lits, skel)
    elif isinstance(value, Expr) or is_dataclass(value):
        skel.append(("cls", type(value).__name__))
        for f in fields(value):
            if f.init:
                _collect(getattr(value, f.name), names, lits, skel)
    else:  # int / float / bool / enum / None / opaque (ElementwiseImpl) — structural, compared by value
        skel.append(("val", value))


def _transform(value, on_name, on_lit, ctr: dict):
    """Rebuild ``value``, replacing the ``ctr['n']``-th str leaf via ``on_name(idx, s)`` and the
    ``ctr['l']``-th ``Literal`` via ``on_lit(idx, lit)`` — the same traversal order as ``_collect``."""
    if isinstance(value, Literal):
        idx = ctr["l"]
        ctr["l"] += 1
        return on_lit(idx, value)
    if isinstance(value, str):
        idx = ctr["n"]
        ctr["n"] += 1
        return on_name(idx, value)
    if isinstance(value, tuple):
        return tuple(_transform(v, on_name, on_lit, ctr) for v in value)
    if isinstance(value, Expr) or is_dataclass(value):
        return type(value)(**{f.name: _transform(getattr(value, f.name), on_name, on_lit, ctr) for f in fields(value) if f.init})
    return value


def _split(name: str) -> tuple[str, int] | None:
    """``"O_i_f7"`` → ``("O_i_f", 7)``; ``None`` when the name has no trailing integer."""
    m = _SUFFIX.match(name)
    return (m.group(1), int(m.group(2))) if m else None


# --------------------------------------------------------------------------- #
# Run detection: a maximal ``(start, K, N)`` where the K-stmt windows w = 0..N-1
# are congruent under one parameterization (fam_pos: name index → family stem;
# affine: literal index → (base, step)).
# --------------------------------------------------------------------------- #


def _win(stmts: list, i: int, K: int) -> tuple[list, list, list]:
    names, lits, skel = [], [], []
    for s in stmts[i : i + K]:
        _collect(s, names, lits, skel)
    return names, lits, skel


def _has_regfrag(stmts: list, start: int, count: int) -> bool:
    return any(isinstance(stmts[start + t], RegFragment) for t in range(count) if start + t < len(stmts))


def _frag_stems(body: Body, sizes: dict[str, int]) -> None:
    """Every fragment-family stem → its member count — the ``RegFragment`` decls that make a name an
    arrayable array (``float O_i_f0..7`` → ``{"O_i_f": 8}``)."""
    for s in body:
        if isinstance(s, RegFragment):
            sp = _split(s.name)
            if sp is not None:
                sizes[sp[0]] = max(sizes.get(sp[0], 0), sp[1] + 1)
        for b in s.nested():
            _frag_stems(b, sizes)


def _classify(w0, w1) -> tuple[dict, dict] | None:
    """Derive ``(fam_pos, affine)`` from windows 0 and 1, or ``None`` if any difference is neither a
    stride-1 start-0 family suffix nor an integer affine literal step."""
    n0, l0, s0 = w0
    n1, l1, s1 = w1
    if s0 != s1 or len(n0) != len(n1) or len(l0) != len(l1):
        return None
    fam_pos: dict[int, str] = {}
    for idx, (a, b) in enumerate(zip(n0, n1, strict=True)):
        if a == b:
            continue
        pa, pb = _split(a), _split(b)
        if pa is None or pb is None or pa[0] != pb[0] or pa[1] != 0 or pb[1] != 1:
            return None  # not a contiguous fragment family (``fam0`` then ``fam1``)
        fam_pos[idx] = pa[0]
    affine: dict[int, tuple[int, int]] = {}
    for idx, ((v0, d0), (v1, d1)) in enumerate(zip(l0, l1, strict=True)):
        if v0 == v1 and d0 == d1:
            continue
        if d0 != d1 or isinstance(v0, bool) or isinstance(v1, bool) or not isinstance(v0, int) or not isinstance(v1, int):
            return None  # only integer literals fold to ``base + _r*step``
        affine[idx] = (v0, v1 - v0)
    return fam_pos, affine


def _window_ok(stmts: list, i: int, K: int, w: int, w0, fam_pos: dict, affine: dict) -> bool:
    """Window ``w`` equals window 0 with each family suffix → ``w`` and each affine literal → ``base + w*step``."""
    n, ell, s = _win(stmts, i + w * K, K)
    n0, l0, s0 = w0
    if s != s0 or len(n) != len(n0) or len(ell) != len(l0):
        return False
    for idx, name in enumerate(n):
        want = f"{fam_pos[idx]}{w}" if idx in fam_pos else n0[idx]
        if name != want:
            return False
    for idx, (v, d) in enumerate(ell):
        if idx in affine:
            base, step = affine[idx]
            if v != base + w * step or d != l0[idx][1]:
                return False
        elif (v, d) != l0[idx]:
            return False
    return True


def _detect(stmts: list, min_run: int, frag_stems: set[str]) -> list[tuple[int, int, int, dict, dict]]:
    """Maximal re-rollable runs ``(start, K, N, fam_pos, affine)`` with ``N >= max(2, min_run)``. Runs
    start on the smallest window size ``K`` that yields ≥ 2 congruent windows; ``RegFragment`` decls
    are transparent (collapsed by arraying, never looped). A run must vary at least one fragment
    family, and EVERY family it varies must be ``RegFragment``-declared — a run over scalar carriers
    or an inline-declared fragment is not arrayable and is rejected."""
    runs: list[tuple[int, int, int, dict, dict]] = []
    i, total = 0, len(stmts)
    while i < total:
        if isinstance(stmts[i], RegFragment):
            i += 1
            continue
        chosen = None
        for k in range(1, _KMAX + 1):
            if i + 2 * k > total or _has_regfrag(stmts, i, 2 * k):
                continue
            w0 = _win(stmts, i, k)
            params = _classify(w0, _win(stmts, i + k, k))
            if params is None:
                continue
            fam_pos, affine = params
            if not fam_pos or any(stem not in frag_stems for stem in fam_pos.values()):
                continue  # no fragment family, or a non-arrayable (scalar / inline-declared) one
            n = 2
            while i + (n + 1) * k <= total and not _has_regfrag(stmts, i + n * k, k) and _window_ok(stmts, i, k, n, w0, fam_pos, affine):
                n += 1
            chosen = (k, n, fam_pos, affine)
            break
        if chosen is None:
            i += 1
            continue
        k, n, fam_pos, affine = chosen
        if n >= min_run:
            runs.append((i, k, n, fam_pos, affine))
        i += k * n
    return runs


# --------------------------------------------------------------------------- #
# Rewrite: collapse each family's decls, fuse each run into its loop, recurse.
# --------------------------------------------------------------------------- #


def _scan(body: Body, min_run: int, frag_stems: set[str], used: set[str]) -> None:
    """Collect every fragment family some run re-rolls, across all nested bodies."""
    stmts = list(body)
    for _i, _k, _n, fam_pos, _affine in _detect(stmts, min_run, frag_stems):
        used.update(fam_pos.values())
    for s in stmts:
        for b in s.nested():
            _scan(b, min_run, frag_stems, used)


def _build_loop(window: list, fam_pos: dict, affine: dict, n: int) -> StridedLoop:
    """The re-rolled run: one ``#pragma unroll`` ``StridedLoop`` over ``_r`` whose body is the window
    with family names indexed ``fam[_r]`` and affine literals folded to ``base + _r*step``."""

    def on_name(idx: int, s: str) -> str:
        return f"{fam_pos[idx]}[{_LOOPVAR}]" if idx in fam_pos else s

    def on_lit(idx: int, lit: Literal) -> Expr:
        if idx not in affine:
            return lit
        base, step = affine[idx]
        term = BinaryExpr("*", Var(_LOOPVAR), Literal(step, "int"))
        return term if base == 0 else BinaryExpr("+", Literal(base, lit.dtype), term)

    ctr = {"n": 0, "l": 0}
    fused = tuple(_transform(s, on_name, on_lit, ctr) for s in window)
    return StridedLoop(
        axis=Axis(name=_LOOPVAR, extent=n), start=Literal(0, "int"), step=Literal(1, "int"), body=Body(fused), unroll=unroll_ok_n(n)
    )


def _rewrite_tree(body: Body, min_run: int, frag_stems: set[str], arrayed: dict[str, int]) -> Body:
    """Replace each run with its fused loop, collapse each arrayed family's ``RegFragment`` decls into
    one ``count``-arrayed decl, and recurse into nested bodies."""
    stmts = list(body)
    runs = {i: (k, n, fam_pos, affine) for i, k, n, fam_pos, affine in _detect(stmts, min_run, frag_stems)}
    out: list = []
    i = 0
    while i < len(stmts):
        if i in runs:
            k, n, fam_pos, affine = runs[i]
            out.append(_build_loop(stmts[i : i + k], fam_pos, affine, n))
            i += k * n
            continue
        s = stmts[i]
        if isinstance(s, RegFragment):
            sp = _split(s.name)
            if sp is not None and sp[0] in arrayed:
                stem, idx = sp
                if idx == 0:  # array at the family's first decl; drop the siblings (idx > 0)
                    out.append(RegFragment(name=stem, role=s.role, shape=s.shape, dtype=s.dtype, count=arrayed[stem]))
                i += 1
                continue
        nested = s.nested()
        if nested:
            s = s.with_bodies(tuple(_rewrite_tree(b, min_run, frag_stems, arrayed) for b in nested))
        out.append(s)
        i += 1
    return Body(tuple(out))
