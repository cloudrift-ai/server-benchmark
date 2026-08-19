"""Engine events + the :class:`Strategy` protocol + discovery.

The rewrite engine is IR-dialect-agnostic: it emits a small fixed set of EVENTS and never
branches on pass names, dialects, or per-concern flags. Every cross-cutting concern — provenance
threading, kernel structural identity, a tune session's kernel inventory — is a strategy class
implementing the event methods it cares about. Extension is a new strategy over the existing
events (or a new event field), never a new engine parameter.

Two binding scopes share the protocol:

- **Discovered** (build-scoped): strategy modules are plain ``.py`` files at the top level of the
  ``passes/`` directory; :func:`discovered_strategies` imports them and instantiates every
  :class:`Strategy` subclass they define. Instances are shared across runs and candidates, so
  they hold immutable config only — never trajectory state. Dispatch order is deterministic
  (class-name sort) but MUST NOT be load-bearing: no strategy may depend on another having
  handled an event first.
- **Run-scoped** (``Run.observers``): instances with per-run state (e.g. the two-level tuner's
  ``KernelInventory``), installed by the caller that owns the run and notified after the
  discovered set.

Events fire at the engine's own moments — ``Run.drive`` / ``Run.resolve`` entry,
``Candidate.apply``'s Graph splice (before and after), and ``Cursor.advance``'s pass completion —
and carry payload objects so signatures never churn.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emmy.compiler.context import Context
    from emmy.compiler.graph import Graph, SpliceReceipt
    from emmy.compiler.ir.base import Op
    from emmy.compiler.pipeline.pipeline import Match


class Strategy:
    """Base class for engine-event strategies — the event protocol's one authoritative
    declaration. Subclasses defined in a ``passes/`` top-level module are DISCOVERED and
    instantiated at ``Pipeline.build`` (see module docstring); run-scoped instances are
    installed via ``Run.observers``. Every handler below is a concrete no-op (never abstract:
    each strategy cares about a subset) — override the events you act on."""

    def on_run_start(self, e: RunStartEvent) -> None:  # noqa: B027 — optional hook, no-op default
        """A loop (``Run.drive`` / ``Run.resolve``) starts driving a graph."""

    def on_splice(self, e: SpliceEvent) -> None:  # noqa: B027 — optional hook, no-op default
        """Before a ``Graph`` fragment splices in (op identities stable, pre-id-promotion).
        Handlers may mutate fragment OPS — never the graph or the cursor."""

    def on_spliced(self, e: SplicedEvent) -> None:  # noqa: B027 — optional hook, no-op default
        """After the splice, with its :class:`~emmy.compiler.graph.SpliceReceipt`."""

    def on_pass_end(self, e: PassEndEvent) -> None:  # noqa: B027 — optional hook, no-op default
        """A named pass completed (quiescent scan)."""


@dataclass
class RunStartEvent:
    """A loop (``Run.drive`` / ``Run.resolve``) starts driving ``graph``. ``passes`` names the
    pipeline's pass list — a strategy keyed to a pass boundary reads it to handle partial
    pipelines that enter after its boundary (e.g. a loop-stage IR resume never runs
    ``loop/stamp``, so identity stamps at entry instead)."""

    graph: Graph
    ctx: Context
    passes: tuple[str, ...]


@dataclass
class SpliceEvent:
    """Emitted by ``Candidate.apply`` BEFORE a ``Graph`` fragment splices in. Fragment op
    identities are stable (pre-splice, pre-id-promotion); ``graph`` is the candidate's graph,
    still holding the consumed nodes. Strategies may mutate fragment OPS (stamp identity,
    thread attribution) — never the graph or the cursor."""

    match: Match
    fragment: Graph
    root_op: Op
    pass_name: str
    graph: Graph


@dataclass
class SplicedEvent:
    """Emitted by ``Candidate.apply`` AFTER the splice; ``receipt`` is what the splice did
    (see :class:`emmy.compiler.graph.SpliceReceipt`)."""

    graph: Graph
    pass_name: str
    receipt: SpliceReceipt


@dataclass
class PassEndEvent:
    """Emitted by ``Cursor.advance`` when a named pass completes with a quiescent scan.
    ``passes`` names the pipeline's full pass list, so a strategy keyed to a boundary can
    compute it per event instead of holding per-run state (build-scoped strategies are shared
    across concurrent runs)."""

    pass_name: str
    graph: Graph
    ctx: Context
    passes: tuple[str, ...]


_STRATEGY_DIR = Path(__file__).resolve().parent / "passes"
_DISCOVERED: tuple[Strategy, ...] | None = None


def discovered_strategies() -> tuple[Strategy, ...]:
    """Import every top-level ``passes/*.py`` strategy module and return one shared instance of
    each :class:`Strategy` subclass they define, class-name-sorted. Cached — the same instances
    serve every pipeline build (they are stateless by contract)."""
    global _DISCOVERED
    if _DISCOVERED is None:
        modules: set[str] = set()
        for path in sorted(_STRATEGY_DIR.glob("*.py")):
            if path.name == "__init__.py" or path.name.startswith("_"):
                continue
            # Canonical import path (passes/ is a package), so a strategy class has exactly one
            # class object whether reached through discovery or a plain import.
            name = f"emmy.compiler.pipeline.passes.{path.stem}"
            importlib.import_module(name)
            modules.add(name)
        classes = [cls for cls in Strategy.__subclasses__() if cls.__module__ in modules]
        _DISCOVERED = tuple(cls() for cls in sorted(classes, key=lambda c: c.__name__))
    return _DISCOVERED


def emit(strategies, event_name: str, event) -> None:
    """Notify every strategy in ``strategies``. Strategies derive from :class:`Strategy`, so
    every event method exists (a no-op unless overridden) — a missing attribute is a loud
    error, not a silently ignored observer."""
    for strat in strategies:
        getattr(strat, event_name)(event)
