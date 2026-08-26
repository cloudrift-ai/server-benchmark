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
- :class:`emmy.compiler.ir.pure.fold.Fold` /
  :class:`emmy.compiler.ir.tile.ir.TileOp` — the tile term's α-invariant
  identity, digested bottom-up from per-node canonical content plus the
  children's cached keys (``ir/tile/_key.py``); excludes placement,
  schedule slices, workers and stores.
- :class:`emmy.compiler.context.Context` — codegen-affecting
  compilation knobs (compute capability today; tuning overrides as they
  land). Excludes ambient I/O fields (dump dirs, verbosity, the session
  cache).

``Op.cache_key`` layers on these: each kernel-bearing dialect folds its
content identity with the op's knob dict for the tuning / cubin caches.

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
    """

    def structural_key(self) -> str: ...


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
    - dataclasses render as their fields, in declaration order;
    - anything else exposing a ``str`` ``name`` renders as ``(class, name)`` — the
      ``ElementwiseImpl`` case, whose name IS its identity (its algebraic traits are looked up by
      it) and which is not a dataclass;
    - anything remaining falls back to ``(class, repr)``. That escape hatch is the one place a
      ``__repr__`` still reaches a key, so a new field type that lands there deserves a rule above
      rather than a shrug.

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
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (type(value).__name__, *(form(getattr(value, f.name)) for f in dataclasses.fields(value)))
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return (type(value).__name__, name)
    return (type(value).__name__, repr(value))


def digest(*parts: object) -> str:
    """Fold ``parts`` into a hex sha256 digest. Each part is rendered
    via ``repr`` and joined; pass child digests (already strings),
    primitive fields (ints, tuples, names), or any ``repr``-stable
    object. Order is significant — callers control canonicalization."""
    return hashlib.sha256(repr(parts).encode()).hexdigest()
