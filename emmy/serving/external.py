"""Build compiler programs whose live inputs and outputs belong to a host runtime."""

from __future__ import annotations


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


def build_external_program(
    graph,
    *,
    pins: dict[str, str] | None = None,
    tune_db: str | None = "auto",
    symbolic_values: dict[str, int] | None = None,
):
    """Compile a graph with no private copies of its live boundary buffers."""
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.backend.cuda.program import CompiledProgram
    from emmy.compiler.backend.gpu_lock import gpu_lock
    from emmy.compiler.backend.plan import plan_from_graph
    from emmy.compiler.pipeline.search.pins import pinned_knobs

    with pinned_knobs(pins or {}):
        plan = plan_from_graph(CudaBackend(tune_db=tune_db).compile(graph))
    plan = _override_symbolic_hints(plan, symbolic_values)
    external = frozenset((*plan.inputs, *plan.outputs))
    with gpu_lock():
        return CompiledProgram.build_from_plan(plan, external_buffers=external), plan
