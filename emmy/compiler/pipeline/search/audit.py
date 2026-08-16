"""Verified-tier drift audit — do the recorded goldens still decide against a set of graphs?

Compiles each graph greedily with the verified tier as the ONLY evidence — no tune DB, the
online-prior file pointed at a nonexistent path (``config.online_file_override``), the
repo-shipped offline prior resolving whatever the records don't — and collects one verdict per
verified-tier consultation through the :func:`~.policy.greedy.golden_audit` seam:

  MATCH — a record carrying the fork's ``deploy_identity`` decided it: the record's spelled row
          (``knob.schedule_row_key``) equalled exactly one enumerated leaf
  DRIFT — records carry that identity but NO offered leaf equals any of their rows (a graph /
          enumeration change invalidated the recording; the tier is fail-closed and the deploy
          falls through)
  GAP   — no record carries the fork's identity (coverage information, not a defect)

DRIFT is always a defect: the recorded config claims a µs the deploy can no longer produce. The
join is strict structural identity and the decode is exact row equality — there is no shape
classification and no prefix acceptance anywhere in the verdicts, so a MATCH means the deploy
really did realize that recording.

The audit is machine-independent by construction — it forces the deployable nvcc regime (records
are -O3 truth; under ``make test``'s ``-Xcicc -O1`` lane the tier is not consulted at all) and
targets the golden file's own card via ``Context.from_target``, so it runs identically on a
GPU-less CI box and the 5090 host.

Consumer: ``emmy eval golden GOLDEN_YAML --serving-config PATH`` — the release gate, which
re-traces the pinned model's serving twins weight-free, audits the exact file-scoped realization
matrix, and ratchets :func:`consultation_counts` against the serving config's checked-in
``SERVE_CONSULT_BASELINE``.
"""

from __future__ import annotations

import logging
import tempfile
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emmy.compiler.graph import Graph

logger = logging.getLogger(__name__)

#: The one non-MATCH/DRIFT/GAP verdict: the greedy compile itself failed, so no fork of
#: that graph was audited. Always a gate failure — an uncompilable serving twin is worse
#: than a drifted record.
COMPILE_FAIL = "COMPILE_FAIL"


def audit_graph(graph: Graph, ctx=None) -> list[dict]:
    """One greedy compile of ``graph`` under ``ctx`` (``None`` probes the live device),
    returning the verified-tier verdict records. This is the primitive — it does NOT isolate
    evidence or force the deployable regime; use :func:`audit_card` for the reproducible
    whole-card audit."""
    from emmy.compiler.pipeline import CUDA_PASSES, Pipeline  # noqa: PLC0415

    from .policy.greedy import golden_audit  # noqa: PLC0415

    records: list[dict] = []
    with golden_audit(records):
        Pipeline.build(CUDA_PASSES).run(graph, ctx=ctx)
    return records


def audit_card(
    graphs: dict[str, Graph],
    gpu_name: str,
    compute_cap: tuple[int, int],
    *,
    goldens: list | None = None,
) -> dict[str, list[dict]]:
    """Audit every graph in ``graphs`` (name → traced Graph) against the records of
    ``(gpu_name, compute_cap)``, off-GPU-safe. ``goldens`` optionally scopes the audit to one
    file or precision lane (installed as the tier's ``records_for_card`` corpus — the loader the
    tier actually reads). Returns name → verdict records; a graph whose compile raises yields a
    single :data:`COMPILE_FAIL` record with the error."""
    from emmy import config  # noqa: PLC0415
    from emmy.compiler import target  # noqa: PLC0415
    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.compiler.pipeline.search import golden  # noqa: PLC0415

    cap = tuple(compute_cap)
    out: dict[str, list[dict]] = {}
    with tempfile.TemporaryDirectory(prefix="emmy-golden-audit-") as tmp:
        # A guaranteed-nonexistent online file: verdicts must not depend on this
        # machine's tune history. The offline prior (repo artifact) still ranks the
        # non-verified forks, so the compile follows the real cold-deploy path.
        absent = Path(tmp) / "absent-online.json"
        prev_target = target._OVERRIDE  # noqa: SLF001 — save/restore around the audit
        scoped = golden.RECORDS_OVERRIDE
        with config.nvcc_flags_override(""), config.online_file_override(absent):
            target.set_target(cap)
            if goldens is not None:
                golden.RECORDS_OVERRIDE = goldens
            try:
                # Built inside the overrides: ``compile_flags`` (→ ``H_opt=3``, the
                # deployable regime the tier is gated on) reads the env at construction time.
                ctx = Context.from_target(cap, gpu_name=gpu_name)
                for name, graph in graphs.items():
                    try:
                        out[name] = audit_graph(graph, ctx)
                    except Exception as ex:  # noqa: BLE001 — one bad graph must not sink the audit
                        logger.error("golden audit: %s failed to compile: %s", name, ex)
                        out[name] = [{"node": None, "key": None, "verdict": COMPILE_FAIL, "golden": None, "us": None, "error": str(ex)}]
            finally:
                golden.RECORDS_OVERRIDE = scoped
                target.set_target(prev_target)
    return out


def summarize(records_by_graph: dict[str, list[dict]]) -> Counter:
    """Total verdict counts across an :func:`audit_card` result."""
    return Counter(r["verdict"] for records in records_by_graph.values() for r in records)


def consultation_counts(records_by_graph: dict[str, list[dict]]) -> dict[str, int]:
    """Verified-tier consultations per graph — every MATCH/DRIFT/GAP record (:data:`COMPILE_FAIL`
    is not a consultation). This count is the signal the verdicts cannot carry: a pass change that
    removes a kernel's schedule fork entirely (e.g. a merged kernel whose lowering stops
    enumerating candidates) deploys it single-option with NO consultation, so its recorded MATCHes
    silently vanish instead of turning DRIFT. ``emmy eval golden`` ratchets these counts per twin
    and lane against the serving config's checked-in baseline (``SERVE_CONSULT_BASELINE``); a drop
    is a gate failure naming the twin."""
    return {name: sum(r["verdict"] != COMPILE_FAIL for r in records) for name, records in records_by_graph.items()}


def gap_keys(records_by_graph: dict[str, list[dict]]) -> set:
    """Every distinct GAP identity across an :func:`audit_card` result — the full coverage view
    the release gate ratchets on (the records must cover ALL kernel forks in the model:
    contractions, reductions/norms, pointwise alike). The identities are opaque digests; they
    are counted and compared, never classified by shape."""
    return {r["key"] for records in records_by_graph.values() for r in records if r["verdict"] == "GAP" and r["key"] is not None}
