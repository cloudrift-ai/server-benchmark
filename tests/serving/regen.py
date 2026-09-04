"""Regenerate the serving-generation lane's golden — ``python -m tests.serving.regen``.

Builds every runner in :data:`helpers.RUNNERS` once, capturing the graph handed to each plan
compile, and writes them as one inventory completed to **one entry per kernel of the set** — the
same ``helpers.complete`` the realization corpus authors its cases with, reading the replay's
realized row per kernel.

Run it when a runner shape changes, when a new one joins the table, or when strict evidence starts
reporting a kernel the golden does not decide (a kernel identity or schedule codec moved). Those
reports are the signal that this needs running; they are never something to work around by
loosening the scope.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _captured_graphs() -> dict:
    """Every distinct graph this lane compiles, keyed by a readable name.

    The runner shapes come through the plan cache — ``PlanTemplateCache.resolve`` sees each graph on
    its way to the compiler, the one seam every split's plan passes through — and the attention-split
    wrappers are traced directly, since they never reach a runner.

    """
    from emmy.compiler.backend.plan_cache import PlanTemplateCache
    from emmy.serving.gen_runner import EmmyGenRunner
    from tests.serving import helpers

    graphs: dict = {}

    class _Capturing(PlanTemplateCache):
        def resolve(self, graph, compile_plan):
            missed = self.misses
            plan = super().resolve(graph, compile_plan)
            if self.misses > missed:  # a hit is the same binding-neutral graph, already recorded
                graphs[f"g{len(graphs):03d}"] = graph.copy()
            return plan

    # ONE cache across every runner, the way the lane's session fixture shares it: two shapes with
    # a layer program in common contribute it once, and the capture sees each distinct graph once.
    cache = _Capturing()
    for runner_id, (make_model, kwargs) in helpers.RUNNERS.items():
        logger.info("[regen] building %s ...", runner_id)
        before = len(graphs)
        EmmyGenRunner.from_model(make_model(), plan_cache=cache, **kwargs)
        logger.info("[regen]   %s contributed %d new graph(s) (%d total)", runner_id, len(graphs) - before, len(graphs))
    for case_id in helpers.WRAPPERS:
        logger.info("[regen] tracing wrapper %s ...", case_id)
        graphs[f"w-{case_id}"] = helpers.wrapper_graph(case_id)
    return graphs


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from emmy.compiler.context import Context
    from emmy.compiler.pipeline.search.golden import dump_golden_file, load_golden_file
    from emmy.compiler.pipeline.search.working_golden import write_trace_inventories
    from tests.compiler.realization import helpers as corpus
    from tests.serving import helpers

    destination = Path(helpers.GOLDEN)
    graphs = _captured_graphs()
    logger.info("[regen] %d distinct graph(s); writing the inventory", len(graphs))
    destination.parent.mkdir(parents=True, exist_ok=True)
    scratch = destination.with_suffix(".inventory.tmp")
    scratch.unlink(missing_ok=True)
    write_trace_inventories(graphs, scratch, ctx=Context.probe())
    document = load_golden_file(scratch)
    scratch.unlink(missing_ok=True)

    for index, entry in enumerate(document["configs"]):
        single = {**document, "configs": [entry]}
        document["configs"][index] = corpus.complete(single)["configs"][0]
        logger.info("[regen]   target %d: %d realization(s)", index, len(document["configs"][index]["realizations"]))
    dump_golden_file(document, destination, overwrite=True)
    logger.info("[regen] wrote %s", destination)
    return 0


if __name__ == "__main__":
    sys.exit(main())
