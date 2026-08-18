"""Build compiler programs whose live inputs and outputs belong to a host runtime."""

from __future__ import annotations


def build_external_program(graph, *, pins: dict[str, str] | None = None, tune_db: str | None = "auto"):
    """Compile a graph with no private copies of its live boundary buffers."""
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.backend.cuda.program import CompiledProgram
    from emmy.compiler.backend.gpu_lock import gpu_lock
    from emmy.compiler.backend.plan import plan_from_graph
    from emmy.compiler.pipeline.search.pins import pinned_knobs

    with pinned_knobs(pins or {}):
        plan = plan_from_graph(CudaBackend(tune_db=tune_db).compile(graph))
    external = frozenset((*plan.inputs, *plan.outputs))
    with gpu_lock():
        return CompiledProgram.build_from_plan(plan, external_buffers=external), plan
