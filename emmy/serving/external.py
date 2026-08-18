"""Build compiler programs whose live inputs and outputs belong to a host runtime."""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

_PACK_PROGRAM = "external"


class ExternalProgramNotRealized(RuntimeError):
    """The exact external-program execution-plan pack is unavailable."""


def _external_pack_key(graph, pins: dict[str, str] | None, tune_db: str | None, symbolic_values: dict[str, int] | None) -> dict:
    """Return the complete persistent identity of one external program."""
    return {
        "model": "external-program",
        "graph": graph.to_dict(),
        "pins": dict(sorted((pins or {}).items())),
        "tune_db": None if tune_db is None else str(tune_db),
        "symbolic_values": dict(sorted((symbolic_values or {}).items())),
    }


@contextmanager
def _pack_lock(path: Path):
    """Serialize a first pack build across serving worker processes."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _override_symbolic_hints(plan, values: dict[str, int] | None):
    if not values:
        return plan

    from dataclasses import replace

    unknown = set(values) - set(plan.symbolic_hints)
    if unknown:
        raise KeyError(f"external program symbolic values name unknown dimensions: {sorted(unknown)}")
    hints = dict(plan.symbolic_hints)
    for name, raw_value in values.items():
        value = int(raw_value)
        cap = plan.symbolic_caps.get(name)
        if value < 1 or (cap is not None and value > cap):
            raise ValueError(f"external program symbolic value {name!r}={value} is outside [1,{cap or 'unbounded'}]")
        hints[name] = value
    return replace(plan, symbolic_hints=hints)


def _resolve_external_plan(graph, pins, tune_db, symbolic_values, compile_plan):
    """Load one external plan pack or serialize its first process-wide build."""
    from emmy import config
    from emmy.compiler.backend.pack import load_pack, pack_path, save_pack

    pack_root = config.pack_dir()
    if pack_root is None:
        return compile_plan()

    key = _external_pack_key(graph, pins, tune_db, symbolic_values)
    # Fail before touching the filesystem if a new graph field is not part of
    # the stable JSON wire form expected by the pack manifest.
    json.dumps(key, sort_keys=True)
    pack_at = pack_path(pack_root, key)
    loaded = load_pack(pack_at, key=key)
    plan = loaded.get(_PACK_PROGRAM) if loaded is not None else None
    if plan is not None:
        return plan

    lock_at = pack_at.with_name(f"{pack_at.name}.lock")
    with _pack_lock(lock_at):
        # Another TP/PP worker may have completed the same program while this
        # process waited. Only the first worker performs graph passes.
        loaded = load_pack(pack_at, key=key)
        plan = loaded.get(_PACK_PROGRAM) if loaded is not None else None
        if plan is not None:
            return plan
        plan = compile_plan()
        try:
            save_pack(pack_at, {_PACK_PROGRAM: plan}, key=key, provenance={"kind": "external-program"})
        except Exception:  # noqa: BLE001 -- a pack miss must not disable a correct live program
            logger.warning("external program pack save failed at %s; continuing with the compiled plan", pack_at, exc_info=True)
        return plan


def _load_external_plan(graph, pins, tune_db, symbolic_values):
    """Load one exact external plan without compiling on a miss."""
    from emmy import config
    from emmy.compiler.backend.pack import load_pack, pack_path

    pack_root = config.pack_dir()
    if pack_root is None:
        raise ExternalProgramNotRealized("external-program runtime loading requires EMMY_PACK_DIR")

    key = _external_pack_key(graph, pins, tune_db, symbolic_values)
    json.dumps(key, sort_keys=True)
    pack_at = pack_path(pack_root, key)
    loaded = load_pack(pack_at, key=key)
    plan = loaded.get(_PACK_PROGRAM) if loaded is not None else None
    if plan is None:
        raise ExternalProgramNotRealized(f"external-program pack is missing or invalid at {pack_at}")
    return plan


def _compile_external_plan(graph, pins, tune_db, symbolic_values):
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.backend.plan import plan_from_graph
    from emmy.compiler.pipeline.search.pins import pinned_knobs

    with pinned_knobs(pins or {}):
        compiled = plan_from_graph(CudaBackend(tune_db=tune_db).compile(graph))
    return _override_symbolic_hints(compiled, symbolic_values)


def realize_external_plan(
    graph,
    *,
    pins: dict[str, str] | None = None,
    tune_db: str | None = "auto",
    symbolic_values: dict[str, int] | None = None,
):
    """Compile and persist one external plan, then prove its strict pack load."""
    from emmy import config

    if config.pack_dir() is None:
        raise ExternalProgramNotRealized("external-program realization requires EMMY_PACK_DIR")
    _resolve_external_plan(
        graph,
        pins,
        tune_db,
        symbolic_values,
        lambda: _compile_external_plan(graph, pins, tune_db, symbolic_values),
    )
    return _load_external_plan(graph, pins, tune_db, symbolic_values)


def _program_from_plan(plan):
    from emmy.compiler.backend.cuda.program import CompiledProgram
    from emmy.compiler.backend.gpu_lock import gpu_lock

    external = frozenset((*plan.inputs, *plan.outputs))
    with gpu_lock():
        return CompiledProgram.build_from_plan(plan, external_buffers=external), plan


def build_external_program(
    graph,
    *,
    pins: dict[str, str] | None = None,
    tune_db: str | None = "auto",
    symbolic_values: dict[str, int] | None = None,
):
    """Compile or load a graph with no private copies of its live boundary buffers."""
    plan = _resolve_external_plan(
        graph,
        pins,
        tune_db,
        symbolic_values,
        lambda: _compile_external_plan(graph, pins, tune_db, symbolic_values),
    )
    return _program_from_plan(plan)


def load_external_program(
    graph,
    *,
    pins: dict[str, str] | None = None,
    tune_db: str | None = "auto",
    symbolic_values: dict[str, int] | None = None,
):
    """Load a realized external program without any compile-on-miss path."""
    return _program_from_plan(_load_external_plan(graph, pins, tune_db, symbolic_values))
