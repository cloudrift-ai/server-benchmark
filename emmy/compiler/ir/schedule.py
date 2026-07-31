"""Tile schedule type system — how a kernel's axes bind to the hardware — plus the codec engine that
ser/des the schedule knobs and the warp-spec role registry they build on.

This root ``ir`` module is the merge of the former ``ir/tile/{codec,role,schedule}.py``. The schedule
value types are used by both the tile IR and the kernel materializer, so they live at the ir root
beside :mod:`~emmy.compiler.ir.atom`, not under ``ir/tile``.

**The schedule is separate from the combine.** The combine (the ⊕) lives in the op tree
(:mod:`emmy.compiler.ir.stmt.algebra` + :mod:`~emmy.compiler.ir.tile.ir`); the
schedule — which axes are parallel, how the reduce axis partitions across hardware levels — is the
**codec value types** here (:class:`ReducePlan` / :class:`TilePlan` / :class:`Stage` /
:class:`WarpSpec` + :class:`Placement`). The per-node slices live in ``TileOp.schedule`` (1r —
keyed by the tree-path codec, read through ``ops.Sched``; the derived ``Contraction`` carries
them as view fields) beside the thin root :class:`~emmy.compiler.ir.tile.ir.TileOp`
fields (``place`` / ``work`` / ``workers``).

A reduction's only freedom is **how the reduce axis is partitioned across hardware levels**
(:class:`ReducePlan`); the combine *mechanism* at each level is **derived** from the level
(:meth:`ReduceStage.combine`), and the combine *algebra* rides the carrier (the ``Twist``). So the
same op + the same materializer extend across kernel kinds — only the carrier and the partition
change.

The schedule is **flat and kind-free**: a kernel's structure is read from its annotated reduce loop's
:class:`~emmy.compiler.ir.axis.AxisRole` (``ops.axis_role``), not a Python type, so a pointwise
cell, a ``PLANAR`` / ``TWISTED`` reduce, and a ``CONTRACTION`` contraction all schedule through the
same value types and use the subset their axes admit. Warp specialization is **orthogonal** — an
optional ``workers: WarpSpec | None`` root field (``None`` = uniform SIMT), a role→warp-count split
*over* the fixed pipeline.

**The codec values are SITE-LOCAL**: a kernel's worker inventory is spelled exactly once, in the
``WORK`` family (:class:`Workers`), so a ``TILE`` value carries no unit widths
(``<atom>/f<FM>x<FN>[/k<bk>]`` warp | ``f<fn>[x<fm>]`` scalar) and a ``REDUCE`` value no coop width
(``[g<n>[a|k]][/coop[-t]][/r<n>]``). Both therefore ``parse`` **against** a :class:`Workers` and are
hand-written here; :func:`resolve_site_tile` is the one rule resolving the single ambiguity that
spelling has (an empty ``TILE`` beside a thread inventory). There is no second, self-contained
reading — the retired embedded-worker grammar (``a:<atom>/w../f..``, ``n../f..``, ``b<n>``) raises.

**The codec engine** (:func:`decode` / :func:`encode` / :class:`Schema` / :class:`Field`) is the
ser/de source of truth for the inventory-free codecs (``STAGE`` / ``WSPEC``): one ``/``-separated
string of ``TOKEN`` nodes (grammar at :func:`desugar` / :func:`decode`), declared as a
:class:`Schema` of typed :class:`Field`\\s so the value type keeps only its own semantics
(``is_async`` / validation). The tunable knob *declarations* that spell these codecs (``REDUCE`` /
``TILE`` / ``STAGE`` / ``WSPEC``) live one layer up in
:mod:`emmy.compiler.pipeline.search.space` — they are ``Knob`` instances, a search concern.

**Warp-spec roles** (:class:`RoleKind` / :data:`ROLE_REGISTRY`) are the worker bands a CTA's warps
split into; the ``WSPEC`` schema is built from the registry so a new role needs no codec edit. The
COMPUTE / mma-consumer role is implicit (sized by ``TilePlan.units``), never registered.
"""

from __future__ import annotations

import enum
import re
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

from emmy.compiler.ir.atom import SCALAR_ATOM, Atom, AtomKind, atom_for
from emmy.compiler.ir.axis import Axis

# ===========================================================================================
# Codec engine — the ser/de source of truth for the inventory-free schedule codecs below (STAGE /
# WSPEC). The site-local TILE / REDUCE values parse against a Workers inventory and are hand-written.
# ===========================================================================================

_TOKEN_RE = re.compile(r"^([A-Za-z]+)(.*)$")


def _codec_width(num: str, *, tok: str, codec: str) -> int:
    """Parse a codec field's positive-integer width, rejecting empty / non-numeric / ``< 1``
    values with a clear message. A ``1`` width is the legal identity (the level is off); only
    ``0`` / negatives / non-digits are rejected. The ``codec`` name rides the message so a bad
    pin names the knob it came from (``bad REDUCE token ...`` / ``bad TILE token ...``)."""
    if not num.isdigit() or int(num) < 1:
        raise ValueError(f"bad {codec} token {tok!r}: expected a positive integer width, got {num!r}")
    return int(num)


class FieldKind(enum.Enum):
    """The five leaf shapes a codec field can take."""

    TUPLE = "tuple"  # int(xint)* — arity-1 scalar or arity-n pair (the STAGE d/p, the WSPEC role warp counts)
    NAME = "name"  # a registry name resolved to an object (an extension point; no live codec uses it)
    CHOICE = "choice"  # a bare token that IS the value (the STAGE transport sync|cp|tma)
    FLAG = "flag"  # a valueless bare token → bool (STAGE ring)
    GROUP = "group"  # a value plus named sub-fields (the WSPEC roles, p<np>[:<param>,...])


class Emit(enum.Enum):
    """When :func:`encode` renders a field (the byte-identity lever — mirrors the old ``spell``)."""

    ALWAYS = "always"  # render unconditionally (the STAGE depth / transport)
    NONDEFAULT = "nondefault"  # render iff the value differs from the field default (the STAGE p, the WSPEC roles)
    TRUE = "true"  # render iff truthy (the ring flag)


@dataclass(frozen=True)
class Field:
    """One node slot in a codec :class:`Schema`."""

    token: str
    kind: FieldKind
    arity: int = 1  # TUPLE: number of int elements (1 = scalar int value, ≥2 = tuple value)
    emit: Emit = Emit.NONDEFAULT
    suppress_trailing: bool = True  # TUPLE arity≥2: drop trailing ``x<m>`` when it equals 1
    required: bool = False  # decode raises if the node is absent
    suffix: tuple[tuple[str, str], ...] = ()  # TUPLE arity-1 enum letter: ((codec, canonical), ...)
    suffix_default: str = ""  # the canonical suffix when the letter is omitted
    choices: tuple[tuple[str, str], ...] = ()  # CHOICE: ((codec-token, canonical), ...)
    default: Any = None  # CHOICE: the canonical default; others derive via :func:`_default`
    params: tuple[Field, ...] = ()  # GROUP sub-fields

    @property
    def is_bare(self) -> bool:
        """True for the no-leading-letter-token kinds matched whole (CHOICE / FLAG)."""
        return self.kind in (FieldKind.CHOICE, FieldKind.FLAG)


@dataclass(frozen=True)
class Schema:
    """A codec's field list + its name (rides parse errors) + the ``expect`` hint."""

    name: str
    fields: tuple[Field, ...]
    expect: str = ""

    def by_token(self) -> dict[str, Field]:
        return {f.token: f for f in self.fields}


def _default(field: Field) -> Any:
    """The value of an absent field (decode fills these so the IR glue reads every token)."""
    if field.kind is FieldKind.TUPLE:
        if field.suffix:
            return (1, field.suffix_default)
        return 1 if field.arity == 1 else tuple([1] * field.arity)
    if field.kind is FieldKind.FLAG:
        return False
    if field.kind is FieldKind.CHOICE:
        return field.default
    if field.kind is FieldKind.NAME:
        return None
    return None  # GROUP absent → None


def desugar(node: str) -> str:
    """Rewrite a glued node into canonical ``TOKEN:value[,params]`` form.

    ``w2x2`` → ``w:2x2``; ``g2a`` → ``g:2a``; ``p2:q8`` → ``p:2,q8``. A node with no glued value
    (``cp`` / ``ring`` / ``a:mma...`` — already colon-form) is returned unchanged."""
    m = _TOKEN_RE.match(node)
    if not m:
        return node
    token, rest = m.group(1), m.group(2)
    if rest and rest[0].isdigit():
        value, _, tail = rest.partition(":")
        return f"{token}:{value}" + (f",{tail}" if tail else "")
    return node


def _split_top(s: str) -> list[str]:
    return [t.strip() for t in s.split("/") if t.strip()]


def _decode_tuple(field: Field, body: str, *, tok: str, codec: str) -> Any:
    """Decode an ``int(xint)*`` value (peeling a suffix letter first when the field has one)."""
    if field.suffix:
        suf = dict(field.suffix)
        tag = field.suffix_default
        if body and body[-1] in suf:
            tag = suf[body[-1]]
            body = body[:-1]
        return (_codec_width(body, tok=tok, codec=codec), tag)
    parts = body.split("x") if body else [""]
    if len(parts) > field.arity:
        raise ValueError(f"bad {codec} token {tok!r}: expected ≤{field.arity} dims, got {len(parts)}")
    nums = [_codec_width(p, tok=tok, codec=codec) for p in parts]
    nums += [1] * (field.arity - len(nums))
    return nums[0] if field.arity == 1 else tuple(nums)


def _decode_value(field: Field, body: str, *, tok: str, schema: Schema) -> Any:
    if field.kind is FieldKind.TUPLE:
        return _decode_tuple(field, body, tok=tok, codec=schema.name)
    if field.kind is FieldKind.NAME:
        return atom_for(body)  # raises ValueError on an unknown name
    if field.kind is FieldKind.GROUP:
        return _decode_group(field, body, tok=tok, schema=schema)
    raise ValueError(f"bad {schema.name} token {tok!r} ({schema.expect})")


def _decode_group(field: Field, body: str, *, tok: str, schema: Schema) -> dict[str, Any]:
    """Decode a GROUP body ``2,q8`` into ``{"": <own value>, <sub-token>: <sub value>, ...}``,
    binding the single bare-tuple positional to the field's own value and named params by token."""
    sub_by_token = {p.token: p for p in field.params}
    out: dict[str, Any] = {"": 1 if field.arity == 1 else tuple([1] * field.arity)}
    out.update({p.token: _default(p) for p in field.params})
    for raw in body.split(",") if body else []:
        part = raw.strip()
        if not part:
            continue
        if part[0].isdigit():  # the positional own-value tuple
            out[""] = _decode_tuple(field, part, tok=tok, codec=schema.name)
            continue
        sub = desugar(part)
        sub_tok, _, sub_body = sub.partition(":")
        pf = sub_by_token.get(sub_tok)
        if pf is None:
            raise ValueError(f"bad {schema.name} role param {part!r} on {tok!r} ({schema.expect})")
        out[sub_tok] = _decode_value(pf, sub_body, tok=part, schema=schema)
    return out


def _match_bare(schema: Schema, node: str) -> Field | None:
    """Match a whole bare node against a FLAG token or a CHOICE codec-token."""
    for f in schema.fields:
        if f.kind is FieldKind.FLAG and node == f.token:
            return f
        if f.kind is FieldKind.CHOICE and node in dict(f.choices):
            return f
    return None


def decode(schema: Schema, spec: str | None) -> dict[str, Any]:
    """Decode a codec string into ``{token: value}`` (defaults filled for absent fields).

    Raises ``ValueError`` — and only ``ValueError`` — on any malformed input (the featurizer
    degrades on ``ValueError``); the message names ``schema.name`` so a bad pin names its knob."""
    result: dict[str, Any] = {f.token: _default(f) for f in schema.fields}
    present: set[str] = set()
    by_token = schema.by_token()
    for node in _split_top((spec or "").strip()):
        bare = _match_bare(schema, node)
        if bare is not None:
            result[bare.token] = True if bare.kind is FieldKind.FLAG else dict(bare.choices)[node]
            present.add(bare.token)
            continue
        token, _, body = desugar(node).partition(":")
        field = by_token.get(token)
        if field is None or field.is_bare:
            raise ValueError(f"bad {schema.name} token {node!r} ({schema.expect})")
        result[field.token] = _decode_value(field, body, tok=node, schema=schema)
        present.add(field.token)
    for f in schema.fields:
        if f.required and f.token not in present:
            raise ValueError(f"{schema.name} codec {spec!r} names no {f.token} ({schema.expect})")
    return result


def _is_default(field: Field, value: Any) -> bool:
    if field.kind is FieldKind.TUPLE:
        if field.suffix:
            return value[0] == 1  # the width drives presence; the suffix rides an emitted token
        return value == 1 if field.arity == 1 else all(v == 1 for v in value)
    if field.kind is FieldKind.CHOICE:
        return value == field.default
    return not value


def _render_tuple(field: Field, value: Any) -> str:
    if field.suffix:
        width, tag = value
        canonical_to_codec = {canonical: codec for codec, canonical in field.suffix}
        # A ``suffix_default`` canonical that has NO codec letter (e.g. the ``b`` field's
        # "interleaved") renders bare — only the marked variants carry a letter. Fields
        # listing every canonical (``g``: a/k) keep re-emitting theirs.
        return f"{width}{canonical_to_codec.get(tag, '')}"
    if field.arity == 1:
        return str(value)
    nums = list(value)
    if field.suppress_trailing:
        while len(nums) > 1 and nums[-1] == 1:
            nums.pop()
    return "x".join(str(n) for n in nums)


def _encode_field(field: Field, value: Any) -> str | None:
    if field.emit is Emit.TRUE:
        return field.token if value else None
    if field.emit is Emit.NONDEFAULT and _is_default(field, value):
        return None
    if field.kind is FieldKind.FLAG:
        return field.token if value else None
    if field.kind is FieldKind.CHOICE:
        return dict((t, c) for c, t in field.choices)[value]
    if field.kind is FieldKind.NAME:
        return f"{field.token}:{value.name}"
    if field.kind is FieldKind.GROUP:
        return _encode_group(field, value)
    return f"{field.token}{_render_tuple(field, value)}"


def _encode_group(field: Field, value: dict[str, Any]) -> str:
    """Render ``p<own>`` (glued, no params) or ``p<own>:<param>,...`` — the positional value is
    glued to the token; emitted named params follow after a ``:`` (matching the ``WSPEC`` spelling)."""
    own = _render_tuple(field, value.get("", 1 if field.arity == 1 else tuple([1] * field.arity)))
    params = [frag for p in field.params if (frag := _encode_field(p, value.get(p.token, _default(p)))) is not None]
    return f"{field.token}{own}" + (":" + ",".join(params) if params else "")


def encode(schema: Schema, values: dict[str, Any]) -> str:
    """Encode a ``{token: value}`` map into the canonical codec string (fields in schema order)."""
    toks: list[str] = []
    for field in schema.fields:
        frag = _encode_field(field, values.get(field.token, _default(field)))
        if frag is not None:
            toks.append(frag)
    return "/".join(toks)


def field_default(field: Field) -> Any:
    """The value of an absent ``field`` — the public handle on the engine's default (used to drop
    default-valued params when building a structured codec object, so they don't spell back)."""
    return _default(field)


# ===========================================================================================
# Warp-specialization roles — the worker bands the WSPEC codec / WarpSpec build on.
# ===========================================================================================


def _always(_sched: object) -> bool:
    return True


def _never(_sched: object) -> bool:
    return False


def _drives_tma_stage(sched: object) -> bool:
    """The producer band is only meaningful when the pipeline stages operands over a transport a
    dedicated warp can drive for other warps — TMA this cut: the box copy is issued by one elected
    thread and lands on the slot mbarrier any thread can parity-wait, so the fill moves warps
    freely. ``cp.async``'s wait-group is issuing-thread-scoped (a consumer cannot observe a
    producer's commit groups without the ``cp.async.mbarrier.arrive`` bridge, which the kernel IR
    does not carry) and a ``sync`` compute-fill has no async load half — both stay uniform."""
    stage = getattr(sched, "stage", None)
    return stage is not None and getattr(stage, "transport", None) == "tma"


@dataclass(frozen=True)
class RoleKind:
    """One warp-specialized worker role.

    ``token`` is the ``WSPEC`` codec letter (``p`` producer, ``s`` sfu, ...); ``params`` is the
    per-role param schema (extra :class:`~emmy.compiler.ir.schedule.Field`\\s after the
    warp-count value — e.g. the producer's in-flight op window ``q``); ``legal`` decides whether
    the role is meaningful for a given uniform schedule (the producer needs a ``stage`` to drive).
    Frozen + hashable so it rides on a frozen ``WarpSpec``."""

    token: str
    help: str = ""
    params: tuple[Field, ...] = ()
    legal: Callable[[object], bool] = dc_field(default=_always)


#: The registered warp-spec roles, keyed by the ``WSPEC`` codec token. COMPUTE (the mma consumer)
#: is implicit — sized by ``TilePlan.units``, never registered. PRODUCER drives the ``Stage`` load
#: half; SFU is a stub example of the role-extension path (a transcendental epilogue band).
ROLE_REGISTRY: dict[str, RoleKind] = {
    "p": RoleKind(
        "p",
        help="producer warps — drive the Stage gmem→smem load half (TMA transport)",
        params=(Field("q", FieldKind.TUPLE),),  # in-flight op window (producer-local; RESERVED — a q pin degrades to uniform)
        legal=_drives_tma_stage,
    ),
    "s": RoleKind("s", help="sfu / transcendental combine warps — reserved example role (does not materialize)", legal=_never),
}


def role_for(token: str) -> RoleKind:
    """The registered :class:`RoleKind` for ``token`` (a ``WSPEC`` codec role token)."""
    try:
        return ROLE_REGISTRY[token]
    except KeyError:
        raise ValueError(f"unknown warp-spec role {token!r} (have {sorted(ROLE_REGISTRY)})") from None


# ===========================================================================================
# Schedule value types — the codec value objects that ride the structural nodes + TileOp.
# ===========================================================================================


class Level(enum.Enum):
    """One hardware level the reduce axis can be partitioned across, coarse→fine."""

    GRID = "grid"  # across CTAs (split-K) — emitted by 030_split_reduce, never the in-kernel walk
    BLOCK = "block"  # cooperative threads within a CTA (warp shuffle / smem tree)
    REG = "reg"  # ILP register-fold accumulators
    SERIAL = "serial"  # the per-thread serial remainder (never spelled — derived)


class FoldMove(enum.Enum):
    """The per-level combine *mechanism* — the placement-keyed fold MOVE, derived from the
    :class:`Level` (where the reduced axis sits), never tuned and never re-decided at a consumer.
    :meth:`ReduceStage.combine` is the ONE selector; every fold emitter consumes its output —
    ``_factor.emit_combine`` (SHFL butterfly / SMEM tree at scalar residence), ``_twist``'s
    streaming merge (the same SHFL move realized as a ``FragmentRowReduce`` at fragment
    residence), and ``030_split_reduce`` (the cross-CTA ATOMIC / KERNEL finalize as a graph rewrite)."""

    SERIAL = "serial"  # no cross-unit combine (the serial / reg remainder)
    REG = "reg"  # register tree (ILP) — TODO(reg)
    SHFL = "shfl"  # lane-level ``__shfl_xor_sync`` butterfly (within-warp)
    SMEM = "smem"  # cross-warp / block-wide smem tree-halve (within-block)
    ATOMIC = "atomic"  # cross-CTA ``atomicAdd`` finalize (030_split_reduce's one-kernel arm)
    KERNEL = "kernel"  # cross-CTA workspace + deferred sibling combine kernel (030_split_reduce)


@dataclass(frozen=True)
class ReduceStage:
    """One level's **tuned** partition: a ``width`` of partials at a hardware ``level``.

    The combine *mechanism* is **derived** (:meth:`combine`), not stored — the level
    implies the fold, and a BLOCK width derives warp-shuffle vs hierarchical-smem from the
    warp size. ``width`` is power-of-two for BLOCK (the butterfly / tree reorder).

    ``finalize`` is meaningful only at ``GRID``: how ``030_split_reduce`` realizes the cross-CTA
    combine — ``"atomic"`` (the partial kernel ``atomicAdd``\\ s into the output, one kernel,
    additive carriers only) or ``"kernel"`` (a deferred sibling combine kernel over a
    workspace, the only legal arm for a twisted carrier). The ``g<n>[a|k]`` codec letter."""

    level: Level
    width: int = 1
    finalize: str = "kernel"  # GRID only: "atomic" | "kernel" (the g<n> finalize letter)
    # BLOCK only (the ``b<n>t`` codec letter): swap the cooperative thread mapping for a
    # k-major B operand — warp lanes sweep the OUTPUT axis (contiguous B loads at every k
    # step) and the k partition rides the upper thread bits; the combine becomes the
    # lane-indexed smem tree across k-slices (no shuffle stage — each lane holds a
    # different output). The interleaved default keeps lanes on the reduce axis.
    transposed: bool = False

    def combine(self, *, warp_size: int, segmented: bool = False) -> tuple[FoldMove, ...]:
        """The derived per-level combine fold(s), fine→coarse within this stage — the ONE
        placement-keyed move selector every fold emitter consumes (see :class:`FoldMove`).

        - ``SERIAL`` / ``REG`` → ``()`` (no cross-unit combine; REG-fold is TODO(reg)).
        - ``GRID`` → ``(ATOMIC,)`` or ``(KERNEL,)`` per the ``finalize`` letter (the split-K
          cross-CTA move — consumed by ``030_split_reduce``; the carrier / projection legality raises
          stay with the graph rewrite, which alone holds the context).
        - ``BLOCK`` → the intra-CTA hierarchy: a lone ``SHFL`` when ``segmented`` (the
          per-row segmented butterfly for strided-cooperative rows) or ``width ≤ warp``
          (one warp); ``(SHFL, SMEM)`` when ``width`` is a clean warp multiple (lanes then
          the cross-warp tree); else a standalone ``(SMEM,)`` block tree. Power-of-two
          ``width`` required."""
        if self.level in (Level.SERIAL, Level.REG):
            return ()
        if self.level is Level.GRID:
            return (FoldMove.ATOMIC,) if self.finalize == "atomic" else (FoldMove.KERNEL,)
        # BLOCK.
        w = self.width
        if w & (w - 1):
            raise ValueError(f"BLOCK reduce width must be a power of two, got {w}")
        if segmented or w <= warp_size:
            return (FoldMove.SHFL,)
        if w % warp_size == 0:
            return (FoldMove.SHFL, FoldMove.SMEM)
        return (FoldMove.SMEM,)


@dataclass(frozen=True)
class ReducePlan:
    """The kernel's single reduce partition — the **tuned widths only**, coarse→fine.

    There is one reduce fold per kernel (1:1 and singular — the fold owns the axis),
    so the plan holds no axis; the per-thread ``serial`` remainder is derived by the
    materializer as ``ceil(extent / parallel)``. ``stages=()`` is the scalar serial fold
    (today's one-thread-per-cell tier)."""

    stages: tuple[ReduceStage, ...] = ()

    @classmethod
    def of(cls, *, cta: int = 1, coop: int = 1, reg: int = 1, finalize: str = "kernel", coop_transposed: bool = False) -> ReducePlan:
        """Build a plan from per-level widths (1 = absent). Order is coarse→fine:
        GRID (cta) → BLOCK (coop) → REG (reg). ``finalize`` rides the GRID stage;
        ``coop_transposed`` rides the BLOCK stage (the ``b<n>t`` k-major lane swap)."""
        stages: list[ReduceStage] = []
        if cta > 1:
            stages.append(ReduceStage(Level.GRID, cta, finalize=finalize))
        if coop > 1:
            stages.append(ReduceStage(Level.BLOCK, coop, transposed=coop_transposed))
        if reg > 1:
            stages.append(ReduceStage(Level.REG, reg))
        return cls(tuple(stages))

    def spell(self) -> str:
        """The ``REDUCE`` codec value for this plan — the pipeline coarse→fine, SITE-LOCAL: the
        coop WIDTH lives in the kernel's ``WORK`` inventory, never here, so the value is
        ``[g<n>[a|k]][/coop[-t]][/r<n>]``. ``""`` for the scalar serial fold (the per-thread
        serial remainder is never spelled — it derives as ``ceil(extent / parallel)``). The GRID
        finalize letter IS kept: ``a``/``k`` is the atomic-vs-deferred finalize MODE, a site-local
        fact — ``g4a`` and ``g2k`` are semantically different rows, both live in the golden
        corpus."""
        parts: list[str] = []
        if self.cta > 1:
            parts.append(f"g{self.cta}{'a' if self.finalize == 'atomic' else 'k'}")
        if self.coop > 1:
            parts.append("coop-t" if self.coop_transposed else "coop")
        if self.reg > 1:
            parts.append(f"r{self.reg}")
        return "/".join(parts)

    @classmethod
    def parse(cls, spec: str | None, work: Workers | None) -> ReducePlan:
        """Decode a ``REDUCE`` value against the kernel's ``WORK`` inventory (inverse of
        :meth:`spell` — the coop width is ``work.count``, never the string). Empty / ``None`` =
        the scalar serial fold. An unknown token raises, so a width-carrying spelling (the retired
        ``b<n>`` embedded-worker grammar) is a loud error, not a silent second reading."""
        s = (spec or "").strip()
        cta, coop, reg, finalize, transposed = 1, 1, 1, "kernel", False
        for t in s.split("/") if s else ():
            if t.startswith("g"):
                body = t[1:]
                if body.endswith(("a", "k")):
                    finalize = "atomic" if body[-1] == "a" else "kernel"
                    body = body[:-1]
                cta = _codec_width(body, tok=t, codec="REDUCE")
            elif t in ("coop", "coop-t"):
                if work is None or work.kind != "thread":
                    raise ValueError(f"REDUCE {spec!r}: 'coop' requires a thread WORK inventory (t<N>)")
                coop, transposed = work.count, t.endswith("-t")
            elif t.startswith("r") and t[1:].isdigit():
                reg = _codec_width(t[1:], tok=t, codec="REDUCE")
            else:
                raise ValueError(f"REDUCE {spec!r}: unknown token {t!r} (expect g<n>[a|k] / coop[-t] / r<n>)")
        return cls.of(cta=cta, coop=coop, reg=reg, finalize=finalize, coop_transposed=transposed)

    @property
    def parallel(self) -> int:
        """The total parallel degree = ∏ stage widths (the lane/CTA fan-out the serial
        remainder divides into)."""
        p = 1
        for s in self.stages:
            p *= s.width
        return p

    @property
    def needs_split(self) -> bool:
        """True iff any stage is a cross-CTA GRID split (``030_split_reduce`` territory)."""
        return any(s.level is Level.GRID for s in self.stages)

    def _width(self, level: Level) -> int:
        for s in self.stages:
            if s.level is level:
                return s.width
        return 1

    @property
    def coop(self) -> int:
        """The BLOCK (cooperative-thread) width, or 1 if no BLOCK stage."""
        return self._width(Level.BLOCK)

    @property
    def coop_transposed(self) -> bool:
        """True iff the BLOCK stage carries the ``b<n>t`` k-major lane swap."""
        s = self.block_stage
        return s is not None and s.transposed

    @property
    def cta(self) -> int:
        """The GRID (cross-CTA split) width, or 1 if no GRID stage."""
        return self._width(Level.GRID)

    @property
    def finalize(self) -> str:
        """The GRID stage's cross-CTA finalize — ``"atomic"`` | ``"kernel"`` (``"kernel"``
        if no GRID stage; the value is only meaningful when :attr:`needs_split`)."""
        for s in self.stages:
            if s.level is Level.GRID:
                return s.finalize
        return "kernel"

    @property
    def reg(self) -> int:
        """The REG (ILP fold) width, or 1 if no REG stage."""
        return self._width(Level.REG)

    @property
    def block_stage(self) -> ReduceStage | None:
        """The single BLOCK :class:`ReduceStage`, or ``None`` (scalar serial)."""
        for s in self.stages:
            if s.level is Level.BLOCK:
                return s
        return None


#: Pin-input aliases for the scalar tier — an explicit ``a:``-prefixed spelling of the (atom-less)
#: scalar output tile, symmetric with the warp form's ``a:<atom>``. ``a:scalar`` names the scalar atom
#: (:data:`~emmy.compiler.ir.atom.SCALAR_ATOM`'s name); ``a:none`` is an alias. **Pin vocabulary only**
#: — a producer never emits these: :meth:`TilePlan.parse` strips them, so the alias never rides a
#: stored knob dict. Lets a pin force the scalar tier explicitly instead of the invisible empty
#: string.
_SCALAR_ATOM_ALIASES = frozenset({"a:scalar", "a:none"})


def has_scalar_atom_alias(spec: str | None) -> bool:
    """True iff any ``/``-token of ``spec`` is an explicit scalar-atom alias (``a:scalar`` / ``a:none``)."""
    return bool(spec) and any(t.strip() in _SCALAR_ATOM_ALIASES for t in spec.split("/"))


def _strip_scalar_atom(spec: str | None) -> str:
    """Drop any ``a:scalar`` / ``a:none`` token from ``spec``, returning the bare scalar body
    (``""`` when the alias was the only token — the per-cell tier)."""
    if not spec:
        return spec or ""
    return "/".join(t for t in spec.split("/") if t.strip() not in _SCALAR_ATOM_ALIASES)


@dataclass(frozen=True)
class TilePlan:
    """The contraction's output tile — **one descriptor for both tiers**, discriminated by
    :attr:`atom`: a tensor-core :class:`AtomKind` (the warp mma tile) or the scalar
    :class:`~emmy.compiler.ir.atom.ScalarAtom` (the register sub-tile, the ``Scalar``
    fragment). Each tiled output axis splits into a **unit** width (warps for mma / parallel threads
    for scalar — :attr:`units`) and a **register** width (atom sub-cells / register cells per unit —
    :attr:`regs`); ``bk`` chunks the K (contraction) axis (mma only). The all-``1`` scalar tile (a
    scalar ``atom``, ``units``/``regs`` ``(1, 1)``) is the per-cell tier — one thread per output cell.

    ``units`` / ``regs`` are stored in each tier's **native codec order** — warp ``(WM, WN)`` /
    ``(FM, FN)`` (m-then-n), scalar ``(par_n, par_m)`` / ``(reg_n, reg_m)`` (n-then-m, the
    featurizer's inner/coalesced ``n`` vs outer ``m``); the :attr:`units_m` / :attr:`units_n` /
    :attr:`reg_m` / :attr:`reg_n` accessors normalize that order. Spelled by the unified ``TILE``
    knob — the warp form ``<atom>/f<FM>x<FN>[/k<bk>]`` or the scalar ``f<fn>[x<fm>]``; the unit
    widths ride the kernel's ``WORK`` inventory, so :meth:`parse` needs it and :attr:`is_warp`
    discriminates the tier on the object. Decided in ``_schedule`` (inside ``010_recognize``)."""

    atom: Atom = SCALAR_ATOM
    units: tuple[int, int] = (1, 1)  # warp (WM, WN) m-then-n / scalar (par_n, par_m) n-then-m
    regs: tuple[int, int] = (1, 1)  # warp (FM, FN) m-then-n / scalar (reg_n, reg_m) n-then-m
    bk: int = 1  # K-chunk per inner mma step, in atom_k units (mma only; 1 for scalar)

    def spell(self) -> str:
        """The ``TILE`` codec value for this tile — SITE-LOCAL: the worker tokens live in the
        kernel-global ``WORK`` family, so the warp form is ``<atom>/f<FM>x<FN>[/k<bk>]`` (the atom
        name bare — the tier discriminator IS the worker kind, never a per-``TILE`` spelling) and
        the scalar form ``f<fn>[x<fm>]`` (``""`` for the per-cell tier). Never the pin-only
        ``a:scalar`` alias."""
        if self.is_warp:
            k = f"/k{self.bk}" if self.bk > 1 else ""
            return f"{self.atom.name}/f{self.regs[0]}x{self.regs[1]}{k}"
        if not any(v > 1 for v in self.regs):
            return ""
        return f"f{self.regs[0]}" if self.regs[1] == 1 else f"f{self.regs[0]}x{self.regs[1]}"

    @classmethod
    def parse(cls, spec: str | None, work: Workers | None) -> TilePlan:
        """Decode a ``TILE`` value against the kernel's ``WORK`` inventory (inverse of
        :meth:`spell` — the unit widths come from ``work``, never the string): the warp form
        ``<atom>/f<FM>x<FN>[/k<bk>]`` or the scalar ``f<fn>[x<fm>]``. Empty / ``None`` is the
        per-cell tier — beside a thread inventory it can also be the unit-register thread tile,
        the ONE ambiguity :func:`resolve_site_tile` resolves. The pin-only ``a:scalar`` /
        ``a:none`` aliases (an explicit spelling of the scalar tier) are stripped first, so
        ``a:scalar`` alone is per-cell and ``a:scalar/f4x8`` a register-tiled scalar. An unknown
        token raises, so the retired embedded-worker spellings (``a:<atom>/w../f..``, ``n../f..``)
        are a loud error, not a silent second reading."""
        s = _strip_scalar_atom((spec or "").strip()).strip("/").strip()
        if not s:
            return cls()
        tokens = s.split("/")
        units = tuple(work.units) if work is not None else (1, 1)
        atom = None
        if not tokens[0].startswith(("f", "k")):
            try:  # a non-atom leading token is a retired ``n<N>x<M>`` / ``a:<atom>`` spelling
                atom = atom_for(tokens[0])
            except ValueError as e:
                raise ValueError(f"TILE {spec!r}: {e}") from None
            if work is None or work.kind != "warp":
                raise ValueError(f"TILE {spec!r}: a warp atom requires a warp WORK inventory (w<M>x<N>)")
            tokens = tokens[1:]
        regs, bk = (1, 1), 1
        for t in tokens:
            if t.startswith("f") and t[1:2].isdigit():
                d = t[1:].split("x")
                if len(d) > 2:
                    raise ValueError(f"bad TILE token {t!r}: expected ≤2 dims, got {len(d)}")
                widths = [_codec_width(x, tok=t, codec="TILE") for x in d]
                regs = (widths[0], widths[1] if len(widths) == 2 else 1)
            elif atom is not None and t.startswith("k") and t[1:].isdigit():
                bk = _codec_width(t[1:], tok=t, codec="TILE")
            else:
                raise ValueError(f"TILE {spec!r}: unknown token {t!r} (expect [<atom>/]f<fn>[x<fm>][/k<bk>])")
        return cls(units=units, regs=regs) if atom is None else cls(atom=atom, units=units, regs=regs, bk=bk)

    @property
    def is_warp(self) -> bool:
        """True iff this is the tensor-core (mma) tile — :attr:`atom` is a real :class:`AtomKind`."""
        return isinstance(self.atom, AtomKind)

    @property
    def is_tiled(self) -> bool:
        """True iff this materializes a tile: a warp tile always does; a scalar tile only when some
        unit / register width > 1 (else it is the per-cell tier — one thread per output cell)."""
        return self.is_warp or any(v > 1 for v in (*self.units, *self.regs))

    @property
    def units_m(self) -> int:
        """Units on the outer (m) output axis — ``WM`` (warp) / ``par_m`` (scalar)."""
        return self.units[0] if self.is_warp else self.units[1]

    @property
    def units_n(self) -> int:
        """Units on the inner (n) output axis — ``WN`` (warp) / ``par_n`` (scalar)."""
        return self.units[1] if self.is_warp else self.units[0]

    @property
    def reg_m(self) -> int:
        """Register sub-cells on the outer (m) axis — ``FM`` (warp) / ``reg_m`` (scalar)."""
        return self.regs[0] if self.is_warp else self.regs[1]

    @property
    def reg_n(self) -> int:
        """Register sub-cells on the inner (n) axis — ``FN`` (warp) / ``reg_n`` (scalar)."""
        return self.regs[1] if self.is_warp else self.regs[0]

    @property
    def block_threads(self) -> int:
        """The per-CTA thread count = ``units_m · units_n · atom.lanes`` (mma: ``WM·WN·32``;
        scalar: ``par_n·par_m·1``)."""
        return self.units[0] * self.units[1] * self.atom.lanes

    @property
    def tile_m(self) -> int:
        """The per-CTA output rows = ``units_m · reg_m · atom_m``."""
        return self.units_m * self.reg_m * self.atom.atom_m

    @property
    def tile_n(self) -> int:
        """The per-CTA output cols = ``units_n · reg_n · atom_n``."""
        return self.units_n * self.reg_n * self.atom.atom_n


@dataclass(frozen=True)
class Workers:
    """The kernel's ONE worker inventory (1r in-memory; the step-7 ``WORK`` wire family) — the
    per-site ``w``/``n`` worker tokens factored out of the ``TILE`` values into a single
    kernel-global slot (``w4x1`` warps for the mma tier / ``t16x8`` threads for the scalar tier /
    ``t512`` the 1-D cooperative width), plus the ``+p<n>`` PRODUCER band the retired ``WSPEC``
    family spelled (aux warps are inventory too). Derived at option assembly from the TILE
    slices (:func:`derive_workers`), FAILING LOUDLY when two sites disagree — an inconsistent
    ``TILE@dd``/``TILE@pj`` pin pair could only miss rows silently; the slot makes the
    disagreement unrepresentable. Kernel-global-ness encodes today's TRUE invariant:
    fragment-resident dataflow between composed folds shares the warp map."""

    kind: str  # "warp" (mma tier) | "thread" (scalar / coop tiers)
    units: tuple[int, int]  # the native codec order of the TILE value the slot came from
    producer: int = 0  # dedicated producer warps (the WSPEC absorb — ``+p<n>``; warp kind only)

    @property
    def count(self) -> int:
        """Workers in the inventory (warps or threads, by ``kind``) — the COMPUTE band; the
        producer band is on top."""
        return self.units[0] * self.units[1]

    def spell(self) -> str:
        """The ``WORK`` codec string: ``w<M>x<N>[+p<np>]`` (warps; both dims always) /
        ``t<N>x<M>`` (the scalar thread tile, native n-then-m order) / ``t<N>`` (the 1-D
        cooperative width — a trailing ``x1`` is suppressed for threads only)."""
        if self.kind == "warp":
            base = f"w{self.units[0]}x{self.units[1]}"
            return f"{base}+p{self.producer}" if self.producer else base
        if self.units[1] == 1:
            return f"t{self.units[0]}"
        return f"t{self.units[0]}x{self.units[1]}"

    @classmethod
    def parse(cls, spec: str | None) -> Workers | None:
        """Decode a ``WORK`` codec string (inverse of :meth:`spell`); ``""`` / ``None`` = no
        inventory (the per-cell / pure-reduce forms keep their derived launch geometry)."""
        if not spec:
            return None
        s = spec.strip()
        producer = 0
        if "+" in s:
            s, _, band = s.partition("+")
            if not band.startswith("p") or not band[1:].isdigit():
                raise ValueError(f"WORK {spec!r}: expected a +p<n> producer band, got {band!r}")
            producer = int(band[1:])
        kind = {"w": "warp", "t": "thread"}.get(s[:1])
        if kind is None:
            raise ValueError(f"WORK {spec!r}: expected w<M>x<N>[+p<n>] or t<N>[x<M>]")
        dims = s[1:].split("x")
        if not all(d.isdigit() for d in dims) or len(dims) > 2:
            raise ValueError(f"WORK {spec!r}: bad worker dims {s[1:]!r}")
        if kind == "warp" and (len(dims) != 2 or producer < 0):
            raise ValueError(f"WORK {spec!r}: the warp inventory spells both dims (w<M>x<N>)")
        if kind == "thread" and producer:
            raise ValueError(f"WORK {spec!r}: a producer band needs the warp inventory")
        units = (int(dims[0]), int(dims[1]) if len(dims) == 2 else 1)
        return cls(kind=kind, units=units, producer=producer)


def resolve_site_tile(spec: str | None, work: Workers | None, reduce_spec: str | None = "") -> TilePlan:
    """The :class:`TilePlan` a row denotes — :meth:`TilePlan.parse` plus the ONE ambiguity the
    site spelling has: an EMPTY ``TILE`` beside a THREAD ``WORK`` inventory is the unit-register
    parallel thread tile (a tile whose register sub-tile is ``f1x1`` spells ``""``), UNLESS the
    row's ``REDUCE`` value claims those threads as a cooperative band (a ``coop`` token) — then
    ``TILE`` stays the per-cell tier. Every reader that reconstructs a plan from a row (the
    scheduler's materialize, the featurizers) resolves through this one rule."""
    plan = TilePlan.parse(spec, work)
    if (spec or "").strip() or work is None or work.kind != "thread":
        return plan
    try:
        red = ReducePlan.parse(reduce_spec, work)
    except ValueError:
        return plan
    if red.coop > 1:
        return plan  # the thread inventory is the coop band — TILE stays per-cell
    return TilePlan(units=tuple(work.units))


def plan_workers(plan: TilePlan | None) -> Workers | None:
    """The worker inventory ONE tile plan implies — ``None`` for an untiled plan or a 1-thread
    thread inventory (a bare register strip: the per-cell forms keep their derived launch
    geometry)."""
    if plan is None or not plan.is_tiled:
        return None
    w = Workers(kind="warp" if plan.is_warp else "thread", units=tuple(plan.units))
    return None if (w.kind == "thread" and w.count == 1) else w


def derive_workers(tiles) -> Workers | None:
    """Fold the worker inventory out of a kernel's resolved ``TILE`` slices (``TilePlan`` values,
    any iterable) — ``None`` when no slice tiles (the per-cell / pure-reduce forms keep their
    derived launch geometry). Raises on DISAGREEMENT between sites: one kernel has ONE worker
    inventory (the flash QK/PV pair shares the warp map by construction; a pin pair that
    disagrees must fail loudly, never be half-ignored)."""
    work: Workers | None = None
    for plan in tiles:
        w = plan_workers(plan)
        if w is None:
            continue
        if work is None:
            work = w
        elif work != w:
            raise ValueError(f"disagreeing worker geometry across TILE sites: {work.spell()} vs {w.spell()} — one kernel, one inventory")
    return work


@dataclass(frozen=True)
class Placement:
    """Kind-neutral free-axis → grid binding (the parallel output axes and their grid
    mapping). ``010_recognize`` builds an UNMAPPED placement (just ``free``);
    ``_schedule`` (inside ``010_recognize``) maps every free axis onto ``grid`` (the per-cell tier)."""

    free: tuple[Axis, ...] = ()
    grid: tuple[Axis, ...] = ()

    @property
    def is_mapped(self) -> bool:
        """True once the free axes are bound (``grid`` set) — or there were none to bind
        (a scalar-output kernel materializes on an empty grid)."""
        return bool(self.grid) or not self.free

    def on_grid(self) -> Placement:
        """The scalar-tier mapping: bind every free axis onto the thread grid."""
        return Placement(free=self.free, grid=self.free)


# --------------------------------------------------------------------------- #
# The operand-transport + warp-split descriptors. ``Stage`` (sync / cp.async / TMA) is the operand
# pipeline; ``WarpSpec`` (the WSPEC worker split) partitions the CTA's warps into producer /
# compute bands over that fixed pipeline — both resolved scheduler-side and applied verbatim by
# the materializer (the liveness-scheduled ``lowering/kernel/_stage.pipelined_kloop``, via its
# whole-body ``staged_kloop`` entry).
# --------------------------------------------------------------------------- #


#: The codec transport token (``cp``) vs the canonical stored value (``cp.async``).
_TRANSPORT_CODEC = {"sync": "sync", "cp": "cp.async", "tma": "tma"}
_TRANSPORT_SPELL = {v: k for k, v in _TRANSPORT_CODEC.items()}

#: The ``STAGE`` codec schema: ``d<depth>`` and the transport always spell; ``ring`` only when set;
#: ``p<reg_depth>`` only at ≥ 2. The transport choices derive from ``_TRANSPORT_CODEC`` (one source).
_STAGE_SCHEMA = Schema(
    "STAGE",
    (
        Field("d", FieldKind.TUPLE, emit=Emit.ALWAYS),
        Field("transport", FieldKind.CHOICE, choices=tuple(_TRANSPORT_CODEC.items()), default="sync", emit=Emit.ALWAYS),
        Field("ring", FieldKind.FLAG, emit=Emit.TRUE),
        Field("alt", FieldKind.FLAG, emit=Emit.TRUE),
        Field("p", FieldKind.TUPLE),
    ),
    expect="expect d<depth> / sync|cp|tma / ring / alt / p<reg_depth>",
)


@dataclass(frozen=True)
class Stage:
    """One operand-transport pipeline over the serial reduce loop — one ``Stage`` per reduce
    loop (a reduce ``Loop`` ⇒ one reduce axis ⇒ one pipeline). The schedule's
    operand-staging knob, decided in ``_schedule`` (inside ``010_recognize``) and materialized in
    ``010_materialize``.

    A constructed ``Stage`` means staging is **on** (the reused gmem operands ride a shared-
    memory slab); ``schedule.stage is None`` is the register / gmem-direct baseline (no
    slab). Spelled by the ``STAGE`` codec ``d<depth>/sync|cp|tma[/ring][/alt][/p<reg_depth>]``
    (decided in ``_schedule`` (inside ``010_recognize``)). A stage stamped on a ``TileOp`` is **RESOLVED**, not the raw
    pin: the scheduler runs eligibility + sizing against the node ONCE
    (``_schedule._resolve_warp_stage`` / ``_resolve_scalar_stage`` for a contraction's operand
    pipeline, ``_row_stage`` for the reduce tier's shared row) and stamps the result — or ``None``
    when the pin can't engage (gmem-direct) — so the materializer applies it verbatim, deciding
    nothing. Two fields are derived at resolution and never spelled by the codec: ``smem`` (empty
    for a contraction pipeline — both operands stage; the ONE detected row buffer for the
    shared-row stage) and ``bk_elems`` (the contraction slab's K-chunk in elements — the
    codec-spelled ``TilePlan.bk · atom_k`` on the warp tier, the fit-to-smem derivation on the
    scalar tier; 0 for the 1-D shared row).

    The pipeline has two buffering levels down the memory hierarchy, each with its own depth:
    ``depth`` is the **gmem→smem** ring (the cp.async / TMA prefetch over the serial reduce
    loop), ``reg_depth`` is the **smem→register** double-buffer (the ldmatrix ping-pong over
    the inner atom-K steps, breaking the WAR hazard on the operand fragments). They are
    orthogonal — ``d3/cp/p2`` is a 3-deep gmem ring feeding a 2-deep register ping-pong.
    ``reg_depth = 1`` (the default) is the "optional register" OFF point (no inner prefetch).
    The slab K-*granularity* (how much K is resident) is ``TilePlan.bk``, NOT a third depth
    here — granularity and buffer depth are kept distinct."""

    depth: int = 1  # gmem→smem ring depth over the reduce loop (1 = single buffer, no prefetch)
    transport: str = "sync"  # sync | cp.async | tma (the gmem→smem producer)
    smem: tuple[str, ...] = ()  # operands staged through smem (derived at resolution; not in the codec)
    ring: bool = False  # ring buffer vs static double-buffer
    reg_depth: int = 1  # smem→register double-buffer depth (1 = no inner ldmatrix prefetch)
    bk_elems: int = 0  # contraction slab K-chunk, elements (derived at resolution; not in the codec)
    # The ALTERNATING single-slab pipeline (the warp-flash stream's FA-2 choreography): each operand
    # keeps ONE slab and its own mbarrier, and the refills interleave with the phases that no longer
    # read them — K refills under softmax + P·V, V under the next step's Q·K — so a wide (64-key)
    # streaming block overlaps its copies within HALF the paired ring's smem. Implies the A (query)
    # operand stages through smem too (the freed resident fragments are what make the wide block's
    # registers fit). ``d1/tma/alt`` / ``d1/cp/alt``; flash stream only — the matmul resolvers
    # decline it.
    alt: bool = False

    def __post_init__(self) -> None:
        if self.transport not in _TRANSPORT_SPELL:
            raise ValueError(f"bad Stage transport {self.transport!r} (expect sync | cp.async | tma)")
        if self.depth < 1:
            raise ValueError(f"Stage depth must be ≥ 1, got {self.depth}")
        if self.ring and self.depth < 2:
            raise ValueError("a ring buffer needs depth ≥ 2 (nothing to cycle at depth 1)")
        if self.reg_depth < 1:
            raise ValueError(f"Stage reg_depth must be ≥ 1, got {self.reg_depth}")
        if self.alt and (self.depth != 1 or self.ring):
            raise ValueError("the alternating pipeline is single-slab (d1, no ring) — its overlap comes from the phase split")

    @classmethod
    def parse(cls, spec: str | None) -> Stage:
        """Decode the ``STAGE`` knob codec into a stage: ``/``-separated tokens —
        ``d<depth>`` (gmem→smem ring depth), ``sync`` | ``cp`` | ``tma`` (the transport), an
        optional ``ring`` flag, and an optional ``p<reg_depth>`` (smem→register double-buffer
        depth). Empty / ``None`` = the depth-1 ``sync`` default (the caller maps an empty
        ``STAGE`` to ``stage=None``, the gmem-direct baseline — ``parse`` is only reached on a
        non-empty spec). ``smem`` is filled in later by the scheduler. Ser/de routes through
        :func:`decode`; ``cls(...)`` then runs :meth:`__post_init__` (the depth / ring semantics)."""
        v = decode(_STAGE_SCHEMA, spec)
        return cls(depth=v["d"], transport=v["transport"], ring=v["ring"], alt=v["alt"], reg_depth=v["p"])

    def spell(self) -> str:
        """The ``STAGE`` codec string for this stage (inverse of :meth:`parse`). ``smem`` is
        derived, so it is not spelled; ``reg_depth`` is spelled only when ≥ 2 (the ``p1``
        default is omitted, so an unstaged-register config round-trips byte-identical)."""
        return encode(
            _STAGE_SCHEMA,
            {"d": self.depth, "transport": self.transport, "ring": self.ring, "alt": self.alt, "p": self.reg_depth},
        )

    @property
    def is_async(self) -> bool:
        """True for the asynchronous-copy transports (``cp.async`` / ``tma``) — the ones
        that issue a commit/wait or mbarrier handshake rather than a plain ``__syncthreads``."""
        return self.transport in ("cp.async", "tma")


#: The ``WSPEC`` codec schema — one GROUP field per registered :class:`RoleKind` (token =
#: warp-count value, params = the role's per-role param schema). Built from ``ROLE_REGISTRY`` so a
#: new role needs no edit here. The COMPUTE role is implicit (``TilePlan.units``), never a field.
_WSPEC_SCHEMA = Schema(
    "WSPEC",
    tuple(Field(r.token, FieldKind.GROUP, params=r.params) for r in ROLE_REGISTRY.values()),
    expect="expect <role><warps>[:<param>,...]",
)


@dataclass(frozen=True)
class RoleAlloc:
    """One warp-specialized band: a registered :class:`RoleKind`, its dedicated warp count, and its
    per-role param values (canonical-sorted, default-valued params dropped so they don't re-spell)."""

    role: RoleKind
    warps: int = 1
    params: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class WarpSpec:
    """The worker-mapping pin — a role→warp-count allocation over the fixed pipeline, ORTHOGONAL to
    it (``reduce`` / ``tile`` / ``stage``): it adds no pipeline parameter, only the warp split. The
    COMPUTE (mma-consumer) role is implicit (sized by ``TilePlan.units``, never listed); each
    :class:`RoleAlloc` is a band of dedicated warps split off the uniform pipeline, so the CTA
    launches ``TilePlan.block_threads + 32·aux_warps`` threads. ``workers is None`` on the schedule
    is uniform SIMT (every warp does every role's work, software-pipelined in-warp); a constructed
    ``WarpSpec`` means specialization is on. Spelled by the ``WSPEC`` codec ``<token><np>[:<param>,
    ...]`` per role (``p2`` / ``p2:q8`` / ``p2:q8/s1``), decided in ``_schedule`` (inside ``010_recognize``).

    Materialized by the staged K-loop (``lowering/kernel/_stage.staged_kloop``): the producer band
    rides at the TAIL of the thread block (``threadIdx.x >= block_threads``), so the compute warps'
    grid decode is untouched; the producer drives the TMA fill (one elected thread arms the slot
    mbarrier), the compute warps parity-wait, drain and release the slot on an "empty" mbarrier
    ring, with ``SetMaxNReg`` register redistribution between the branches."""

    roles: tuple[RoleAlloc, ...] = ()

    @classmethod
    def parse(cls, spec: str | None) -> WarpSpec:
        """Decode the ``WSPEC`` codec into a role allocation (``""`` / ``None`` = no roles). Ser/de
        routes through :func:`decode` (``_WSPEC_SCHEMA``)."""
        v = decode(_WSPEC_SCHEMA, spec)
        allocs: list[RoleAlloc] = []
        for token, role in ROLE_REGISTRY.items():
            group = v[token]
            if group is None:  # role absent from the codec
                continue
            params = tuple(sorted((p.token, group[p.token]) for p in role.params if group[p.token] != field_default(p)))
            allocs.append(RoleAlloc(role=role, warps=group[""], params=params))
        return cls(tuple(allocs))

    def spell(self) -> str:
        """The ``WSPEC`` codec string for this allocation (inverse of :meth:`parse`); ``""`` when
        there are no roles (uniform SIMT)."""
        values: dict[str, object] = {token: None for token in ROLE_REGISTRY}
        for a in self.roles:
            values[a.role.token] = {"": a.warps, **dict(a.params)}
        return encode(_WSPEC_SCHEMA, values)

    @property
    def aux_warps(self) -> int:
        """Total dedicated (non-COMPUTE) warps the split adds on top of the ``TilePlan`` warp grid."""
        return sum(a.warps for a in self.roles)

    def is_legal(self, sched: object) -> bool:
        """True iff every allocated role is meaningful for ``sched`` (a producer needs a ``stage``).
        A pin failing this degrades to uniform (``workers=None``) — the same pin-validity rule the
        other codecs follow."""
        return all(a.role.legal(sched) for a in self.roles)


@dataclass(frozen=True)
class Raster:
    """The CTA rasterization order — the ``RASTER`` codec, kernel-scoped like ``WSPEC`` (one launch
    order per grid; no ``@<axis>`` key). It changes NO per-CTA work, layout, or schedule — only
    which flat CTA id maps to which ``(m, n)`` output block tile: ``gm<G>`` iterates ``G`` M
    block-tiles fastest within each launch stripe (consecutive CTAs then share the streamed B
    slab, so its repeat reads hit L2 instead of DRAM — the 2026-07-12 4090 NCU finding measured
    the flat order streaming B once per M-row, ``A + C + B×2`` exactly); ``gn<G>`` is the
    transpose (A the streamed operand). ``""`` / ``None`` = the flat N-fastest row-major order.
    Eligible only on a 2-D-tiled contraction grid (``Tile.raster_axes``, stamped by the kernel
    materializer's ``grid_tile`` seal)."""

    orient: str  # "m" (group M block-tiles) | "n" (group N block-tiles)
    group: int  # stripe group size, ≥ 2

    @classmethod
    def parse(cls, spec: str | None) -> Raster | None:
        """Decode ``gm<G>`` / ``gn<G>`` (``""`` / ``None`` → ``None``, the flat order). Raises
        ``ValueError`` on any other spelling — the loud pin contract."""
        if not spec:
            return None
        m = _RASTER_RE.fullmatch(spec)
        if m is None or int(m.group(2)) < 2:
            raise ValueError(f"RASTER: expected gm<G> / gn<G> with G >= 2 (or empty for the flat order), got {spec!r}")
        return cls(orient=m.group(1), group=int(m.group(2)))

    def spell(self) -> str:
        """The ``RASTER`` codec string (inverse of :meth:`parse`)."""
        return f"g{self.orient}{self.group}"


_RASTER_RE = re.compile(r"g([mn])(\d+)")


__all__ = [
    "AtomKind",
    "Emit",
    "Field",
    "FieldKind",
    "FoldMove",
    "Level",
    "Placement",
    "ROLE_REGISTRY",
    "ReducePlan",
    "ReduceStage",
    "RoleAlloc",
    "Raster",
    "RoleKind",
    "Schema",
    "Stage",
    "TilePlan",
    "WarpSpec",
    "Workers",
    "_codec_width",
    "atom_for",
    "decode",
    "derive_workers",
    "desugar",
    "encode",
    "field_default",
    "has_scalar_atom_alias",
    "role_for",
]
