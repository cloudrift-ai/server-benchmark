"""The release gate's compile — does the golden file decide every fork of every reachable kernel?

Compiles each graph greedily with the golden records as the ONLY golden evidence in scope, the
machine-local evidence out of the way (the online-prior file pointed at a nonexistent path,
``config.online_file_override``; a fresh in-memory tune store), the deployable nvcc regime forced,
the golden file's own card targeted through ``Context.from_target``, and strict evidence on: a fork
no golden row decides is an ``EvidenceError`` naming the kernel and the fork
(:func:`~.policy.greedy._require_evidence`), never a prediction the prior makes. That is the same
question the pick answers on every deploy, asked of the whole matrix at once, and it is
machine-independent by construction — it runs identically on a GPU-less CI box and the recording
host.

Consumer: ``emmy eval golden --golden GOLDEN_YAML --serving-config PATH`` — the release gate, which
re-traces the pinned model's serving twins weight-free and compiles the exact file-scoped
realization matrix through :func:`audit_card`, one precision lane at a time.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emmy.compiler.graph import Graph

logger = logging.getLogger(__name__)


def audit_card(
    graphs: dict[str, Graph],
    gpu_name: str,
    compute_cap: tuple[int, int],
    *,
    goldens: list | None = None,
) -> dict[str, str | None]:
    """Compile every graph in ``graphs`` (name → traced Graph) under strict evidence with
    ``goldens`` as the only golden scope (installed as the pick's ``records_for_card`` corpus — the
    loader the pick actually reads), off-GPU-safe. Returns name → ``None`` when every fork was
    decided by a golden row, else the failure: the ``EvidenceError`` naming the kernel and fork
    nothing measured decided, or the compile error."""
    from emmy import config  # noqa: PLC0415
    from emmy.compiler import target  # noqa: PLC0415
    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.compiler.pipeline import CUDA_PASSES, Pipeline  # noqa: PLC0415
    from emmy.compiler.pipeline.search import golden  # noqa: PLC0415

    cap = tuple(compute_cap)
    out: dict[str, str | None] = {}
    with tempfile.TemporaryDirectory(prefix="emmy-golden-audit-") as tmp:
        # A guaranteed-nonexistent online file: the verdict must not depend on this machine's
        # tune history. Under strict evidence the prior decides nothing, so nothing else can either.
        absent = Path(tmp) / "absent-online.json"
        prev_target = target._OVERRIDE  # noqa: SLF001 — save/restore around the audit
        with (
            config.nvcc_flags_override(""),
            config.online_file_override(absent),
            config.strict_evidence_override(True),
            golden.records_override(goldens),
        ):
            target.set_target(cap)
            try:
                # Built inside the overrides: ``compile_flags`` (→ ``H_opt=3``, the deployable
                # regime) reads the env at construction time.
                ctx = Context.from_target(cap, gpu_name=gpu_name)
                for name, graph in graphs.items():
                    try:
                        Pipeline.build(CUDA_PASSES).run(graph, ctx=ctx)
                        out[name] = None
                    except Exception as ex:  # noqa: BLE001 — one bad graph must not sink the audit
                        logger.error("golden audit: %s: %s", name, ex)
                        out[name] = " ".join(f"{type(ex).__name__}: {ex}".split())
            finally:
                target.set_target(prev_target)
    return out
