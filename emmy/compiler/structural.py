"""Structural identity protocol — one convention for every type whose
instances we want to compare or dedup by their structure rather than by
Python identity / field equality.

Implementers return a hex sha256 digest folded over the bits of state
that affect downstream behavior (codegen output, dataflow semantics,
compilation result). Bits that are name-only / advisory / ambient I/O
are deliberately excluded — see each implementer's docstring for the
exact include/exclude list.

Implementers today:

- :class:`emmy.compiler.graph.Graph` — Merkle digest over op kinds,
  body structure, output shapes/dtypes, input wiring; excludes node ids,
  Tensor names, and Hints.
- :class:`emmy.compiler.ir.stmt.body.Body` — canonicalized body
  rendering with SSA / axis / commutative-arg / external-buffer names
  normalized away.
 - :class:`emmy.compiler.ir.pure.fold.Fold` — the tile term's α-invariant
   identity: the exact-flavor canonical digest of the Loop-IR body the term
   lowers to (the term is pure algebra; its body is its normal form);
   excludes placement, classic schedule, materialization and workers.
- :class:`emmy.compiler.context.Context` — codegen-affecting
  compilation knobs (compute capability today; tuning overrides as they
  land). Excludes ambient I/O fields (dump dirs, verbosity, the session
  cache).

``Op.identity_key`` layers on these: one lattice over the op's canonical
Loop-IR body, folding in the io fingerprint and the knob dict for the
deploy join (the deploy identity (``identity_key(with_io=True)``)) and the tuning / cubin caches
(``identity_key(with_io=True, with_knobs=True)``).

The cache layer in the autotuning loop keys candidates by these digests,
so adding a field to an implementer is an explicit decision: include it
in the digest only if it changes generated code or dataflow semantics.

``digest(*parts)`` is the canonical fold helper — pass a mix of strings,
ints, bytes, and pre-computed child digests; the helper canonicalizes
via ``repr`` and returns a hex sha256 string. Composite implementers use
it to fold child digests with their own discriminating fields.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Protocol, runtime_checkable


@runtime_checkable
class Structural(Protocol):
    """Anything whose structural identity is comparable as a hex digest.

    Two instances that should be treated as equivalent for caching /
    dedup purposes return the same string from :meth:`structural_key`;
    two instances that differ in any codegen- or dataflow-relevant way
    return different strings.

    **Inherit it and you get the default**: :func:`form` over the instance, which is the right
    answer for any frozen dataclass whose fields ARE its identity (every ``Stmt``). Override it
    when the type owns a canonicalization the field walk cannot know — ``Fold``'s α-invariance,
    ``Body``'s normalize-and-collapse, ``Graph``'s Merkle walk over its nodes.

    :func:`form` reads that distinction directly (``type(x).structural_key is
    Structural.structural_key``) and delegates only to an override, which is what keeps the
    default from calling itself forever. There is no marker to set and none to forget.
    """

    def structural_key(self) -> str:
        return digest(form(self))


def instance_memo(obj, slot: str) -> dict:
    """The named per-instance memo table riding an IMMUTABLE object — the structural-key
    pattern, as one mechanism: a derived read caches on the term it derives from (the table is
    created on first use via ``object.__setattr__``, so frozen dataclasses take it), and the
    owner's ``__getstate__`` strips it so no cache — an id-keyed one especially — crosses a
    process boundary. The retired bottom-up term hasher originated the pattern with its
    single-slot form; the normalize fixpoint stamp (``ir/tile/normalize.py``) and the codec's
    spelling tables (``ir/tile/path.py``) go through this table form. A memo holds ONLY values
    derivable from the object; never decisions, never mutable policy."""
    table = obj.__dict__.get(slot)
    if table is None:
        table = {}
        object.__setattr__(obj, slot, table)
    return table


def form(value: object) -> object:
    """The STRUCTURAL rendering of ``value`` — a nested ``(class name, *fields)`` tuple.

    The one alternative to ``repr`` for anything that enters an identity. Every IR value object is
    a frozen dataclass (``Stmt``, ``Expr``, ``Axis``, ``Dim``, ``DataType``, ``Window``), so the
    field walk is generic over all of them and needs no per-class list: a new node kind is covered
    the day it is added, and a ``__repr__`` edit never moves a key. That matters because how a
    statement PRINTS is a presentation choice, and presentation choices must not re-key every
    stored golden.

    Rendering rules, in order:

    - primitives and ``None`` pass through, so the leaves stay comparable by value;
    - tuples / lists (``Body`` included — it is a ``tuple`` subclass) render elementwise;
    - sets render SORTED, because their iteration order is not stable and an unsorted rendering
      would key one object two ways;
    - a value that OVERRIDES :meth:`Structural.structural_key` renders as ``(class, its key)`` —
      it owns a canonicalization this walk cannot know. A ``Fold`` is the case that matters:
      it is a dataclass, so the field walk below would happily render it by SSA spelling, while
      its own key is α-invariant. Two renderings of one type is what this module exists to
      prevent, so the type that knows better is asked. A type that merely INHERITS the default
      falls through to the field walk below — that is the default's definition, and delegating
      would recurse. Checked AFTER the container rules, so a ``Body`` still renders elementwise
      rather than through its own aggressive normalize-and-collapse key;
    - dataclasses render as their fields, in declaration order;
    - anything else exposing a ``str`` ``name`` renders as ``(class, name)`` — the
      ``ElementwiseImpl`` case, whose name IS its identity (its algebraic traits are looked up by
      it) and which is not a dataclass;
    - anything remaining RAISES. There is no ``repr`` fallback: a silent one would be the very
      thing this function exists to remove, and it would hide the moment a new field type started
      keying kernels on its ``__repr__``. A type that lands here needs a rule above — one line,
      decided by whoever knows what part of that value is identity and what part is incidental.

    ``DataType`` takes the dataclass route, so its numpy dtype rides along as a leaf. That is
    stable — ``numpy.dtype`` repr is a public API and equal dtypes render equal — just noisier
    than the name alone.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (tuple, list)):
        return tuple(form(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((form(v) for v in value), key=repr))
    own_key = getattr(type(value), "structural_key", None)
    if own_key is not None and own_key is not Structural.structural_key:
        return (type(value).__name__, value.structural_key())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (type(value).__name__, *(form(getattr(value, f.name)) for f in dataclasses.fields(value)))
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return (type(value).__name__, name)
    raise TypeError(
        f"structural.form has no rule for {type(value).__name__}, so this value cannot enter an "
        "identity. Add a rule in structural.form: render the part of it that IS the identity, and "
        "drop the part that is incidental. Do not fall back to repr — that is what this replaced."
    )


def digest(*parts: object) -> str:
    """Fold ``parts`` into a hex sha256 digest. Each part is rendered
    via ``repr`` and joined; pass child digests (already strings),
    primitive fields (ints, tuples, names), or any ``repr``-stable
    object. Order is significant — callers control canonicalization."""
    return hashlib.sha256(repr(parts).encode()).hexdigest()
