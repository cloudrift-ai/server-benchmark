"""Knob descriptor — the canonical schema for a tunable / forkable rule parameter.

Each ``Knob`` declares one tuning dimension: its name (also the env-var
key), its type (driving parse + pretty), the autotune candidate hints,
and a short help string. Rules emit knob values into ``TileOp.knobs``
dicts; the env layer reads ``EMMY_<NAME>`` to pin a knob across
the whole run; pretty-printing routes through the descriptor so
display matches storage.

This module owns the ``Knob`` *descriptor* (the dataclass), the
registry, and the env plumbing — **not** the concrete knob
declarations (those live in :mod:`emmy.compiler.pipeline.search.space`,
beside the enumeration value grids) and **not** the featurizers (those
live in :mod:`emmy.compiler.pipeline.search.features`). INVARIANT:
every ``Knob`` *instance* is declared in ``search/space.py``, and rules
import the knobs they decide from there. Registration is construction:
``Knob.__post_init__`` records every instance in the registry the
moment its declaration executes — no ``register(...)`` wrapper, no
manual bookkeeping, and no dependence on what else the process has
imported. :func:`registry` imports ``space.py`` before answering, so
the canonical set is complete in ANY process that can ask — the
pre-2026-07-28 module-scan discovery silently dropped a knob's
features in processes that featurized without the pipeline loaded.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any

from emmy import config
from emmy.compiler.ir.tile.path import SLICE_FAMILIES

# Reserved prefix for the structural-feature knobs stamped by
# ``loop/stamp/020_stamp_structural_features`` — distinct from any tuning Knob
# name, so ``format_tuning_knobs`` drops them from the tuning view and
# ``knob_features`` passes them through as-is. Declared here (rather than with
# the producing pass, which is loaded under a bare module stem) so every
# consumer can import it.
STRUCT_PREFIX = "S_"

# Reserved prefix for host/hardware-regime features injected from
# :meth:`Context.features` (GPU compute capability + nvcc opt level).
# Treated like ``STRUCT_PREFIX``: dropped from the tuning view, passed straight
# through ``knob_features`` as floats — they describe the regime a row was
# measured in, letting one global prior span every GPU / opt level.
CTX_PREFIX = "H_"


class _Unset:
    """Sentinel for ``Knob.off`` meaning "no OFF value declared" — the knob is
    expected to always be stamped by its owning pass, so
    :func:`apply_off_defaults` never auto-fills it."""

    def __repr__(self) -> str:
        return "<unset>"


_UNSET = _Unset()


class KnobType(Enum):
    """Knob value type — drives ``Knob.parse`` and ``Knob.pretty``."""

    INT = "int"
    BOOL = "bool"
    BINMASK = "binmask"
    STR = "str"


@dataclass(frozen=True)
class Knob:
    """Schema for one tunable parameter.

    ``name`` is used as the key in ``TileOp.knobs`` dicts and to derive
    the env override ``EMMY_<NAME>``. ``hints`` is the autotune
    candidate list (a guideline, not a constraint — rules apply their
    own structural validity gates). ``help`` is a short docstring shown
    by ``emmy eval knobs``."""

    name: str
    type: KnobType
    hints: tuple = ()
    help: str = ""
    # Optional custom featurizer for the online-prior feature vector: maps this
    # knob's value to a ``dict[str, float]`` of sub-features. Declared at the
    # knob (e.g. ``MMA`` expands an atom kind into physical cell/dtype props) so
    # ``knob_features`` needs no per-knob special-casing. ``None`` → encode by
    # ``type`` (INT → float, BOOL → 0/1, BINMASK → popcount/width/frac).
    features: Callable[[Any], dict[str, float]] | None = None
    # The "unused / declined" OFF value. When a knob doesn't apply to a variant
    # (a tier-foreign knob like ``WM`` on a scalar kernel) or its owning pass
    # declines / is skipped, :func:`apply_off_defaults` stamps this value so the
    # realized variant carries an *explicit* decision rather than an absent key —
    # letting the online prior tell "decided: unused" (an OFF value) from
    # "not-yet-decided" (still absent → NaN-filled). ``_UNSET`` (the default)
    # means the knob is always stamped by its pass and is never auto-filled.
    off: Any = _UNSET
    # An **unfeatured** knob: it must never enter the online-prior feature vector
    # (``knob_features`` skips it). Two families qualify: a *cosmetic* policy that changes only how
    # the emitted source is spelled, never the compiled kernel (SASS-identical — e.g. ``LOOPIFY``'s
    # ``--ir`` re-rolling; batch-enabled via ``EMMY_READABLE`` rather than tuned), and a *meta /
    # umbrella* pin that gates which forks are OFFERED without being a kernel property itself
    # (``FAST_MATH`` — the realized fork is already fully identified by the knobs it enables, e.g.
    # the ``TILE`` codec's atom token). Either way the value cannot disambiguate kernels, and
    # letting it perturb the features would split one perf class into spurious variants.
    unfeatured: bool = False

    def __post_init__(self) -> None:
        # Construction IS registration: a module-level ``Knob`` declaration lands in the registry
        # the moment it executes, wherever it lives. First-seen wins, matching the old module-scan
        # collapse rule (duplicate declarations should agree on type / hints anyway).
        _REGISTRY.setdefault(self.name, self)

    @property
    def env(self) -> str:
        return config.knob_var(self.name)

    def raw(self) -> str | None:
        """Live env pin (``EMMY_<NAME>``); ``None`` when unset."""
        return config.knob_raw(self.name)

    def read_int(self, default: int) -> int:
        """Read this knob's ``EMMY_<NAME>`` env pin as an int (empty /
        unset / unparseable → ``default``).

        Reads a knob's env override through its canonical descriptor rather
        than re-spelling the name as a bare string, so a rename of the constant
        fails loudly instead of silently reading a dead env var. ``INT`` knobs
        only."""
        if self.type is not KnobType.INT:
            raise ValueError(f"Knob.read_int only valid for INT knobs ({self.name!r} is {self.type})")
        return config.int_env(self.env, default)

    def parse(self, raw: str, *, width: int | None = None) -> Any:
        """Decode an env-string value per ``type``. ``width`` is required
        for ``BINMASK`` (number of candidate buffers in the current tile).

        BINMASK accepts a binary string ``"101"`` (char ``i`` selects
        ranked-buffer ``i``; length must match ``width``), the keywords
        ``"all"`` / ``"none"``, or a decimal / ``0x``-hex int (clamped
        to ``width`` bits)."""
        s = raw.strip()
        if self.type is KnobType.INT:
            return int(s, 0)
        if self.type is KnobType.BOOL:
            low = s.lower()
            if low in {"1", "true", "yes", "on"}:
                return True
            if low in {"", "0", "false", "no", "off"}:
                return False
            # Anything else (a typo like ``ture`` / ``banana``, or a stray ``2``) used to
            # coerce silently to False, disabling the knob with no diagnostic. Fail loudly.
            raise ValueError(f"bad BOOL value for knob {self.name!r}: {raw!r} (expect 1/true/yes/on or 0/false/no/off)")
        if self.type is KnobType.BINMASK:
            if width is None:
                raise ValueError(f"BINMASK knob {self.name!r} parse needs width")
            if s == "all":
                return (1 << width) - 1
            if s == "none":
                return 0
            if len(s) == width and all(c in "01" for c in s):
                return sum(int(c) << i for i, c in enumerate(s))
            return int(s, 0) & ((1 << width) - 1)
        if self.type is KnobType.STR:
            return s
        raise ValueError(f"unhandled knob type: {self.type}")

    def pretty(self, value: Any, *, width: int | None = None) -> str:
        """Render ``value`` for display / storage. ``BINMASK`` produces
        a fixed-width binary string (char ``i`` = bit ``i``); other types
        round-trip through ``str()``."""
        if self.type is KnobType.BINMASK:
            if width is None:
                raise ValueError(f"BINMASK knob {self.name!r} pretty needs width")
            return "".join("1" if (value >> i) & 1 else "0" for i in range(width))
        return str(value)

    def narrow(self, candidates: Iterable[Any]) -> tuple:
        """Override ``candidates`` with the env pin ``EMMY_<NAME>``.

        Folds env-driven pinning into the same iteration that produces
        hint-driven candidates, so callers don't enumerate-then-filter:

            reduce_choices = REDUCE.narrow(cands)  # 1-tuple if pinned

        Returns ``tuple(candidates)`` unchanged when the env var is
        absent; otherwise a 1-tuple ``(pinned,)`` — *authoritative*,
        even if ``pinned`` is not in ``candidates``. Hints are guidance,
        not constraint: the user knows what they're doing. Downstream
        structural gates (divisibility, threads-per-CTA budget, etc.)
        still apply, so a structurally invalid pin yields an empty
        enumeration; callers that need a peer-kernel fallback (empty
        after structural gates → invalid pin for *this* shape →
        re-enumerate without pins) implement that policy rule-side.

        ``BINMASK`` isn't supported — it would need ``width``, and no
        rule enumerates BINMASK candidates today."""
        raw = self.raw()
        if raw is None:
            return tuple(candidates)
        if self.type is KnobType.BINMASK:
            raise ValueError(f"Knob.narrow not supported for BINMASK ({self.name!r})")
        pinned = self.parse(raw)
        return (pinned,)

    @contextmanager
    def pinned(self, value: str):
        """Temporarily pin ``EMMY_<NAME>=value`` for the enclosed block, restoring the prior
        state (previous value or absence) on exit — the scoped form of the env pin, for
        regime-aware diagnostics that must evaluate under a gate the caller didn't set (e.g.
        ranking a fast-math golden inside the ``F16_MMA_F32_ACC``-gated enumeration). Process-
        global like every env pin: not safe around concurrently-compiling threads."""
        prev = self.raw()
        config.set_knob(self.name, value)
        try:
            yield
        finally:
            if prev is None:
                config.unset_knob(self.name)
            else:
                config.set_knob(self.name, prev)

    def pin_at(self, element: str) -> str | None:
        """The env pin for this knob's ``<NAME>@<element>`` key ALONE — no bare fallback. The
        EXPLICIT half of :meth:`narrow_at`: a bare pin cannot say which site of a multi-site family
        it meant, so a caller that must tell an authoritative site pin from a fanned-out bare one
        asks this first. The ``@``-suffixed key is not a valid shell var name, so it rides the
        ``EMMY_KNOBS`` aggregate; :func:`parse_knob_spec` canonicalizes its element to lowercase,
        matching the lowercase element spelling producers use."""
        return config.knob_raw(f"{self.name}@{element}")

    def narrow_at(self, element: str) -> str | None:
        """The env pin for this knob's ``<NAME>@<element>`` key, falling back to the bare
        ``<NAME>`` pin — the per-element read mirroring :func:`family_value`'s precedence
        (``TILE@d`` > bare ``TILE``). Returns the raw pin string (``None`` when neither is
        set); the caller resolves vocabulary."""
        value = self.pin_at(element)
        if value is not None:
            return value
        return self.raw()


# --- Axis-named schedule keys ----------------------------------------------
#
# A per-node schedule codec is keyed ``FAMILY@<axis>`` — ``TILE@<k_axis>`` / ``STAGE@<axis>`` /
# ``REDUCE@<axis>`` — so a multi-node kernel (flash) can address each schedule-bearing node by the
# reduce/contraction axis it schedules. The **bare** form (``TILE`` with no suffix) stays first-class:
# it resolves to the unique eligible axis for that family, so the common single-node kernel — and every
# existing pin / recipe / golden — keeps working unchanged (the suffix disambiguates only a kernel with
# two eligible nodes). ``TILE`` / ``STAGE`` / ``REDUCE`` all carry the suffix: the schedule reduce
# partition IS the axis-named reduce decision (there is no separate native ``REDUCE@`` family — the
# reduce/split-K partition is the one reduce family). ``WORK`` / ``RASTER`` stay root-global (always
# bare). Readers use :func:`family_value` so a bare and a suffixed key featurize / match identically.

# The per-node schedule codec families that carry an ``@<axis>`` element (``WORK`` / ``RASTER`` are
# root-global, always bare).
_AXIS_FAMILIES = SLICE_FAMILIES  # the one list, defined in ``ir/tile/path.py``


def family_of(key: str) -> str:
    """The knob family — the part before an ``@<axis>`` suffix (``TILE@d`` → ``TILE``); the whole key
    when unsuffixed."""
    return key.split("@", 1)[0]


def axis_of(key: str) -> str | None:
    """The ``@<axis>`` element of a knob key (``TILE@d`` → ``d``), or ``None`` when bare."""
    return key.split("@", 1)[1] if "@" in key else None


def family_value(knobs: dict, family: str):
    """The value of a per-node schedule codec ``family`` in ``knobs``, keyed bare (``TILE``) or
    axis-suffixed (``TILE@<axis>``). ``None`` when absent. A single-node kernel has exactly one match;
    a multi-node kernel (flash) has one key per node — this returns the first (a pooled read; a
    per-node featurizer reads each node's slice via a group-by-axis loop)."""
    v = knobs.get(family)
    if v is not None:
        return v
    prefix = family + "@"
    for k, val in knobs.items():
        if k.startswith(prefix):
            return val
    return None


# --- Registry --------------------------------------------------------------

#: ``name → Knob`` — filled at declaration time by ``Knob.__post_init__``, first-seen wins.
_REGISTRY: dict[str, Knob] = {}


def registry() -> dict[str, Knob]:
    """``{name: Knob}`` — every declared knob, registered at construction time.

    Completeness is guaranteed by importing ``search/space.py`` (the single home of every
    ``Knob`` declaration) before answering: any process that can ask for a knob sees the full
    canonical set, no matter how little of the pipeline it has loaded. The import is deferred
    (``space`` imports this module at its top level)."""
    from emmy.compiler.pipeline.search import space  # noqa: F401, PLC0415 — deferred: space imports knob

    return _REGISTRY


def get(name: str) -> Knob | None:
    """Lookup a registered ``Knob`` by name. ``None`` if no rule has
    declared it."""
    return registry().get(name)


def apply_off_defaults(knobs: dict, declared: Iterable[Knob]) -> dict:
    """Fill any ``declared`` knob with a defined ``off`` value into ``knobs``
    when it is absent — the "every emitted variant carries an explicit value for
    every declared knob" rule. Mutates and returns ``knobs``.

    Used in two places: the pipeline stamps a pass's declared OFF knobs on the
    realized variant at the pass boundary (so a declined / skipped / no-variant
    pass still records its OFF decision), and the partition planner stamps the
    tier-foreign knobs (``WM``/``WN``/``MMA`` on a scalar row; ``BM``/``BN``/…
    on a warp row) on each enumerated row so the OFF value rides the variant
    identity from enumeration → score → DB → materialized ``TileOp.knobs``.
    Idempotent: a knob already present (real value *or* a prior OFF fill) is left
    untouched; knobs whose ``off`` is ``_UNSET`` are never filled."""
    for knob in declared:
        if knob.off is _UNSET:
            continue
        if knob.name not in knobs:
            knobs[knob.name] = knob.off
    return knobs


def pin_key_matches(pinned: str, realized: str) -> bool:
    """Whether a realized same-family key satisfies a pinned key: exact, or one side
    bare — a bare env pin fans out to every eligible axis, and a bare golden spelling
    matches whatever axis the lowering stamped (``TILE`` ↔ ``TILE@dd``). Two
    *differing* explicit axes never match."""
    return pinned == realized or axis_of(pinned) is None or axis_of(realized) is None


def canon_family_value(name: str, value) -> str:
    """The canonical spelling of a ``TILE`` / ``REDUCE`` / ``STAGE`` value — the codec's own normal
    form. The two site families decode under a DUMMY inventory (the worker widths never reach a site
    spelling, so the dummy cannot leak) and re-spell: ``f64x1`` ≡ ``f64``, the pin-only ``a:scalar``
    alias ≡ ``""``. ``STAGE`` needs no inventory and normalizes token ORDER, which binds order-free
    but spells in schema order — so a hand pin ``cp/d2`` matches the realized ``d2/cp`` instead of
    failing verification against its own value on the deploy path. Any other family — and any
    unparseable value — passes through untouched (the caller's own equality applies)."""
    from emmy.compiler.ir.schedule import ReducePlan, Stage, TilePlan, Workers  # noqa: PLC0415

    fam = family_of(name)
    v = str(value).strip()
    if not v:
        return v
    try:
        if fam == "TILE":
            return TilePlan.parse(v, Workers(kind="warp", units=(1, 1))).spell()
        if fam == "REDUCE":
            return ReducePlan.parse(v, Workers(kind="thread", units=(2, 1))).spell()
        if fam == "STAGE":
            return Stage.parse(v).spell()
    except ValueError:
        return v
    return v


def values_equal(name: str, want, got) -> bool:
    """Pinned-vs-realized value equality for knob ``name``: casefolded ``str`` equality
    (the env round-trip stringifies), else both sides decoded through the registered
    knob's canonical :meth:`Knob.parse` — a BOOL pinned ``1``/``yes``/``on`` matches a
    realized ``True``, a hex INT its decimal, a BINMASK spelling the stamped binary
    string (width taken from the realized binary spelling — the :meth:`Knob.pretty`
    storage convention). ``TILE`` / ``REDUCE`` / ``STAGE`` values canonicalize through
    their codec's normal form (:func:`canon_family_value`), so an atom-ALIAS pin
    (``mma_m16n8k16_f16/…``) keeps matching the canonically-stamped atom, the
    pin-only ``a:scalar`` alias matches the per-cell row, and an out-of-order
    ``cp/d2`` matches the realized ``d2/cp``. ``WORK``
    compares through its own codec. An unregistered family compares by string only."""
    w, g = str(want).strip(), str(got).strip()
    if w.casefold() == g.casefold():
        return True
    if family_of(name) in _AXIS_FAMILIES:
        return canon_family_value(name, w) == canon_family_value(name, g)
    if family_of(name) == "WORK":
        from emmy.compiler.ir.schedule import Workers  # noqa: PLC0415

        try:
            return Workers.parse(w) == Workers.parse(g)
        except ValueError:
            return False
    kn = get(family_of(name))
    if kn is None:
        return False
    width = None
    if kn.type is KnobType.BINMASK:
        if not (g and all(c in "01" for c in g)):
            return False  # realized side isn't the canonical binary spelling — no width to decode with
        width = len(g)
    try:
        return kn.parse(w, width=width) == kn.parse(g, width=width)
    except ValueError:
        return False


def is_off_value(family: str, value) -> bool:
    """Whether ``value`` is ``family``'s declared OFF value — :func:`apply_off_defaults`
    stamps it on every variant the knob doesn't apply to (``""`` for the codec knobs,
    ``False`` for the BOOL policies), so it means "declined / not applicable", never a
    conflicting realization. ``False`` for an unregistered family or a knob with no
    declared OFF."""
    kn = get(family)
    return kn is not None and kn.off is not _UNSET and str(value).strip().casefold() == str(kn.off).strip().casefold()


# --- Aggregate env var ------------------------------------------------------


def apply_knobs_env(raw: str | None = None) -> dict[str, str]:
    """Splat ``EMMY_KNOBS="K1=V1,K2=V2,..."`` into individual
    ``EMMY_<K>=V`` env vars.

    Convenience for setting many knobs from one place (CLI invocation,
    Makefile recipe, pytest ``monkeypatch.setenv``). Individual
    ``EMMY_<K>`` vars take precedence: this only writes a key when
    the per-knob env var is unset, so callers can pin one knob from
    the command line and supply the rest via ``EMMY_KNOBS``
    without surprise.

    Whitespace is tolerant; empty entries are skipped. A malformed
    entry (no ``=``) raises ``ValueError``. Returns the dict of keys
    actually applied (for tests / diagnostics).

    Runs at module import time on the live process environment. Pass
    ``raw`` explicitly to apply a different aggregate without touching
    ``EMMY_KNOBS`` (useful in tests).
    """
    if raw is None:
        raw = config.knobs_aggregate()
    applied: dict[str, str] = {}
    for key, value in parse_knob_spec(raw).items():
        # Individual per-knob env vars win — don't clobber an explicit
        # ``EMMY_BK=4`` with whatever the aggregate says.
        if config.set_knob(key, value, overwrite=False):
            applied[config.knob_var(key)] = value
    return applied


def parse_knob_spec(raw: str) -> dict[str, str]:
    """Parse the shared ``K1=V1,K2=V2`` knob-spec grammar (the ``EMMY_KNOBS``
    aggregate, ``run --ab``) into an ordered ``{NAME: value}`` dict — whitespace
    tolerated, empty entries skipped. A malformed entry (missing ``=`` / empty
    KEY) raises ``ValueError``. Case is canonicalized per key part: the family
    (before any ``@``) uppercases, the element (after it) lowercases — so
    ``tile@D=…`` pins ``TILE@d`` instead of being mangled to ``TILE@D``."""
    out: dict[str, str] = {}
    for entry in (raw or "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(f"knob spec entry {entry!r} is missing '=' (expected KEY=VALUE)")
        key, _, value = entry.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"knob spec entry {entry!r} has empty KEY")
        fam, at, element = key.partition("@")
        key = fam.upper() + (at + element.lower() if at else "")
        out[key] = value.strip()
    return out


# Splat ``EMMY_KNOBS`` once at import so every later per-knob reader
# (``config.knob_raw`` / ``config.int_env`` — knob.py is imported transitively
# by every pipeline pass) sees the individual ``EMMY_<NAME>`` keys.
apply_knobs_env()


# --- Rendering -------------------------------------------------------------

# Canonical display order for tuning knobs. The axis-named schedule codecs (``TILE@`` /
# ``REDUCE@`` / ``STAGE@``) lead, each family's keys sorted by element; the bare exact-name knobs
# follow in ``KNOB_ORDER``, unknown knobs last (alpha). Shared by the ``run --bench`` kernel table
# and the ``emmy eval`` tables so columns read stably.
_FAMILY_ORDER = ("TILE@", "REDUCE@", "STAGE@")
KNOB_ORDER = ("WORK", "TILE", "REDUCE", "STAGE")
_KNOB_RANK = {k: i for i, k in enumerate(KNOB_ORDER)}

# The greedy-fillable schedule codec families a golden RECORDING must pin explicitly. An entry that
# omits one leaves that family to the planner's fill at replay time, and the fill drifts as the
# planner evolves — the recurring unpinned-``REDUCE`` phantom-regression class (a recorded un-split
# config replaying with a surprise ``g2k`` fill). :func:`stamp_schedule_families` is the recording
# view that closes the gap: every family explicit, OFF spelling (``""`` = decided unused) included.
SCHEDULE_FAMILIES = ("WORK", "TILE", "REDUCE", "STAGE", "RASTER")


def schedule_pin_fingerprint() -> tuple[tuple[str, str], ...]:
    """Every live env pin the schedule enumeration can read — the :data:`SCHEDULE_FAMILIES`
    knobs, bare and ``@``-keyed — as sorted ``(env var, raw value)`` pairs. The schedule pool
    cache folds this into its key: a pin changes which rows enumerate, so two pin states must
    never share a pool. The environ scan is this module's to make — the ``EMMY_<KNOB>``
    namespace is knob.py-owned (the one exception to ``config.py``'s env ownership), and the
    ``@``-keyed pins land there via the ``EMMY_KNOBS`` splat."""
    import os  # noqa: PLC0415 — the one environ read outside ``config``, per the ownership note above

    prefixes = tuple(config.knob_var(fam) for fam in SCHEDULE_FAMILIES)
    return tuple(sorted((var, val) for var, val in os.environ.items() if any(var == p or var.startswith(p + "@") for p in prefixes)))


def knob_sort_key(name: str) -> tuple[int, str]:
    """Sort key: native ``MOVE@element`` families first (by :data:`_FAMILY_ORDER`, then
    element), then the codec knobs in :data:`KNOB_ORDER`, unknown knobs last (alpha)."""
    for i, prefix in enumerate(_FAMILY_ORDER):
        if name.startswith(prefix):
            return (i, name[len(prefix) :])
    return (len(_FAMILY_ORDER) + _KNOB_RANK.get(name, len(KNOB_ORDER)), name)


def format_tuning_knobs(knobs: dict) -> str:
    """Render ``knobs`` as a compact ``key=value`` string, dropping
    pass-marker booleans. Empty after filtering → ``-``.

    A registered ``Knob`` of type ``BOOL`` is treated as a marker and
    dropped; unregistered boolean values are also dropped (forward-compat).
    ``BINMASK`` values are already stored as binary strings in
    ``op.knobs`` (rules stamp via ``Knob.pretty``), so ``str(v)`` here
    round-trips correctly. ``STRUCT_PREFIX`` knobs (the structural-feature
    stamp from ``992_stamp_structural_features``) are facts about the kernel, not
    tuning decisions, so they are dropped from this tuning-knob view.
    """
    items = tuning_knob_items(knobs)
    return ", ".join(f"{k}={v}" for k, v in items) if items else "-"


def tuning_knob_items(knobs: dict) -> list[tuple[str, str]]:
    """The filtered, canonically-ordered ``(name, str(value))`` tuning knobs —
    the same view :func:`format_tuning_knobs` renders, but as items so callers can
    build aligned columns. ``STRUCT_PREFIX`` / ``CTX_PREFIX`` features and marker
    booleans are dropped; the rest is sorted by :func:`knob_sort_key`. The unified
    ``TILE`` output-fragment knob is one column for both the scalar and warp tiers
    (the value self-describes), so there are no tier-foreign OFF knobs to hide.

    Keys render AS STORED — since phase 3 the stampers spell the canonical codec key (bare for the
    primary node, ``TILE@dd``-style where the tree needs it), so the view needs no bare-collapse:
    the stamped row IS the stored/golden spelling. (The old ``@<axis>``→bare display collapse died
    with the axis-suffix codec; pre-phase-3 evidence is regenerated, not migrated.)"""
    rendered: list[tuple[str, str]] = []
    for k, v in knobs.items():
        if k.startswith(STRUCT_PREFIX) or k.startswith(CTX_PREFIX):
            continue
        knob = get(k)
        if knob is not None and knob.type is KnobType.BOOL:
            continue
        if knob is None and isinstance(v, bool):
            continue
        rendered.append((k, str(v)))
    return sorted(rendered, key=lambda kv: knob_sort_key(kv[0]))


def canonical_row_key(knobs: dict) -> tuple[tuple[str, str], ...]:
    """Content-based ORDER over candidate knob rows — the :func:`tuning_knob_items`
    view as a comparable tuple. Every deploy pick (model argmin, measured-evidence
    argmin, golden realization) breaks score / µs ties with this key, so the selected
    variant is a function of the candidates' CONTENT, never of their enumeration
    order. Without it, a tie falls to whichever leaf a fork emitted first — an order
    that can shift across processes — and the deployed kernel flips boot to boot (the
    2026-07 gemma-4 5090 image's bimodal cubin set: the offline prior scored 8
    same-featurized tiles identically at the drifted m16 forks, and emission order
    chose among them)."""
    return tuple(tuning_knob_items(knobs))


def evidence_row_vouches(cand_tun: dict, row_tun: dict) -> bool:
    """Whether a measured evidence row can vouch for a candidate under value-of-position
    semantics: every tunable knob the candidate specifies must match the row, a key absent
    from the ROW reading as free (a later pass decides it)."""
    return not any(k in row_tun and row_tun[k] != v for k, v in cand_tun.items())


def stamp_schedule_families(knobs: dict) -> dict[str, str]:
    """The ready-to-record knob map for one realized kernel: its tuning knobs
    (:func:`tuning_knob_items`) plus an explicit OFF value for every :data:`SCHEDULE_FAMILIES`
    family the realized dict never stamped. The pass-boundary OFF fill covers a pass that ran
    and declined, but a family whose pass never loads for the target (e.g. ``STAGE`` when no
    transport resolves) can be absent from ``op.knobs`` entirely — a recording must still pin it
    as declined, or the entry drifts when a later planner starts filling it. A family with no
    registered OFF is skipped rather than invented."""
    out = dict(tuning_knob_items(knobs))
    present = {family_of(k) for k in out}
    for fam in SCHEDULE_FAMILIES:
        if fam in present:
            continue
        kn = get(fam)
        if kn is not None and kn.off is not _UNSET:
            out[fam] = str(kn.off)
    return dict(sorted(out.items(), key=lambda kv: knob_sort_key(kv[0])))
