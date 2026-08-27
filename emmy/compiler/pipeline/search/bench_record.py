"""Bench-to-node recording — ``run --bench`` A/B measurements become node-store leaves.

The tune engine was the node table's only writer, so the manual golden/``--ab`` sweeps
that found optima the search missed (the fm-lane story) never became training data — the
region around each golden stayed censored in every freeze. This module closes that loop:
a ``run --bench`` invocation that benched pinned rows records each clean measurement as a
parentless leaf ``NodeRow`` in the canonical tune DB, where the measurement freeze and
the offline-prior fit pick it up like any tune leaf.

Recording is **default-on behind a quality bar** (:func:`meets_quality_bar` — the tuner's
own pinned-bench standard; ``run --no-record-nodes`` opts out). What records:

- every cleanly-benched pinned golden / ``--ab`` row as an ``ok`` leaf;
- a realized config whose bench failed as a ``bench_fail`` negative (its ``value_us`` is
  :data:`FAIL_SENTINEL_US`, not a measurement — consumers read ``status``), and only when
  the config lowered to ONE kernel: a failure belongs to the variant, and spreading one
  sentinel across several kernels would file a number none of them measured;
- the greedy pick, from its ``greedy (isolated)`` re-bench — pinned-comparable by
  construction, so every benched pool self-anchors the prior's argmax.

Never recorded: ``pin_unmatched`` rows (the claimed config never ran — and "not offered"
is not "doesn't launch"), rows carrying an integrity flag (wrong-answer / intensity
floor: the measurement is untrue), a whole run whose greedy execution computed the wrong
answer, a cross-target (``--gpu-arch``) run (the cubin is assembled for the LIVE device, so
the timings are this card's while the row would key under the target's capability), and
anything from a direct ``--ir`` input (serialization drops ``op.knobs`` AND ``op.source``,
so a row would name only the knobs the short tail happened to re-decide). A ``--golden``
replay is NOT a direct ``--ir`` input: it re-lowers an in-memory program through the full
pipeline, so ``loop/stamp`` stamps every kernel and the rebind knob-merge carries the
realized knobs onto the terminal op — its rows are as honest as the ``--code`` path's, and
they record. The caller (``emmy/commands/run.py``) owns those exclusions; this module
records what it is given.

Pool fidelity: **one row describes one kernel**, the same rule the tune walk records by
(``policy/mcts._measured_kernel_rows``) — its own knobs, its own launch time, and its own
identity, so a kernel benched here and the same kernel tuned by the search meet on one
``node_key`` and rank against each other in one candidate pool. A benched graph is
therefore a per-kernel map, with nothing grouped and no latency summed: a variant that
lowered to several kernels contributes several rows, because no kernel ran at their total.

Identity comes from :func:`~...passes.identity.chain_op_sig`, never from the realized op's
own stamps — tile materialization merges ``S_warp_eligible`` onto the op it builds, so the
op carries a stamp its kernel was not born with. A kernel whose chain carries no stamp at
all is not recordable: there is no identity to file it under, and it is dropped with a
debug note rather than attributed to a neighbour.

Rows are keyed with the tune's own recipes (same ``node_key`` / ``op_sig`` /
``context_key``), parentless with ``depth=0`` — the no-tree-schema marker the fork
diagnostics skip — and stamped with a ``bench-…`` ``run_id`` so freeze headers show the
provenance.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from emmy.compiler.context import Context

logger = logging.getLogger(__name__)

# The tuner's pinned-bench measurement standard (``CudaBackend.bench_pinned_async``
# defaults). A run benched below it is a quick look, not a measurement — recording it
# would let newest-wins replace tune-grade leaves with drive-by numbers.
MIN_RECORD_WARMUP = 5
MIN_RECORD_ITERS = 20

# A ``bench_fail`` leaf's ``value_us`` — NOT a measurement (the tune stores the bench
# watchdog's sentinel there; consumers read ``status`` and every metric excludes fails).
FAIL_SENTINEL_US = 1e9


@dataclass(frozen=True)
class BenchLeaf:
    """One benched KERNEL's measurement, extracted from a benched compiled graph."""

    op_sig: str  # the kernel's identity — the stamp it was born with (``chain_op_sig``)
    knobs: dict  # the kernel's own realized knob dict (S_* stamps + tunables)
    value_us: float  # the kernel's own launch time; sentinel on fail
    variance: float | None
    n_samples: int | None
    status: str  # 'ok' | 'bench_fail'


def meets_quality_bar(warmup: int, iters: int) -> bool:
    """Whether a ``run --bench`` invocation measures well enough to record."""
    return warmup >= MIN_RECORD_WARMUP and iters >= MIN_RECORD_ITERS


def mint_bench_run_id() -> str:
    """A sortable, unique bench-session id — the ``bench-`` prefix distinguishes
    recorded-bench provenance from tune sessions in freeze headers / row audits."""
    return f"bench-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"


def bench_leaves(compiled, bench, *, status: str = "ok") -> list[BenchLeaf]:
    """One :class:`BenchLeaf` per benched CUDA kernel of ``compiled``.

    Kernels pair with ``bench.per_launch`` by topological order (the launch order) and each
    keeps its own launch time: nothing is grouped and nothing is summed, because a row that
    carried several kernels' total would describe a cost no kernel ran at — and the deploy
    prices a multi-kernel realization by summing a per-kernel price, so one kernel's cost is
    the quantity a row must hold.

    ``status="bench_fail"`` (with ``bench=None``) emits ONE sentinel leaf, and only for a
    single-kernel graph: the failure is the variant's, and there is no honest way to divide
    it among several kernels. A kernel whose chain carries no stamp has no identity to file
    under and is skipped with a debug note; if that is true of every kernel in the graph, the
    silence is loud, since it means a lowering path stopped preserving provenance."""
    from emmy.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415
    from emmy.compiler.pipeline.passes.identity import chain_op_sig  # noqa: PLC0415

    nids = [nid for nid in compiled.topological_order() if isinstance(compiled.nodes[nid].op, CudaOp)]
    per_launch = list(getattr(bench, "per_launch", None) or []) if bench is not None else []
    if status != "ok" and len(nids) != 1:
        logger.debug("[record-nodes] %d-kernel variant failed to bench — the failure is the variant's, no row recorded", len(nids))
        return []

    leaves: list[BenchLeaf] = []
    skipped = 0
    for idx, nid in enumerate(nids):
        op = compiled.nodes[nid].op
        sig = chain_op_sig(op)
        if sig is None:
            skipped += 1
            logger.debug("[record-nodes] kernel %s carries no structural stamp in its chain — skipped", nid)
            continue
        knobs = dict(op.knobs or {})
        if status != "ok":
            leaves.append(BenchLeaf(op_sig=sig, knobs=knobs, value_us=FAIL_SENTINEL_US, variance=None, n_samples=None, status=status))
            continue
        launch = per_launch[idx] if idx < len(per_launch) else None
        if launch is None:
            logger.debug("[record-nodes] kernel %s has no per-launch timing — skipped", nid)
            continue
        variance = n_samples = None
        if launch.samples:
            samples_us = [sample * 1000.0 for sample in launch.samples]
            n_samples = len(samples_us)
            variance = statistics.pvariance(samples_us) if n_samples >= 2 else None
        leaves.append(
            BenchLeaf(op_sig=sig, knobs=knobs, value_us=launch.time_ms * 1000.0, variance=variance, n_samples=n_samples, status="ok")
        )
    if skipped and not leaves:
        # Silence must never read as success: a graph whose EVERY kernel lost its stamp means a
        # provenance gap in some lowering path, not "nothing to record".
        logger.warning(
            "[record-nodes] none of the %d kernel(s) carries a structural stamp — nothing recorded "
            "(a lowering path is not preserving op provenance; please report)",
            skipped,
        )
    return leaves


def record_bench_leaves(db_path: Path | str, ctx: Context, leaves: list[BenchLeaf], *, run_id: str | None = None) -> int:
    """Key ``leaves`` with the tune's own recipes and upsert them into the node store
    at ``db_path`` — parentless ``depth=0`` leaf rows under the live context's regime
    (``run --bench`` compiles at the deployable flags, so these land in the -O3 lane
    the store is censored in). Returns the number of rows offered to
    :meth:`SearchDB.record_nodes` (its plausibility gate and quality-aware leaf
    replacement still apply per row)."""
    from emmy.compiler.pipeline.search.db import NodeRow, SearchDB, node_key  # noqa: PLC0415

    if not leaves:
        return 0
    run_id = run_id or mint_bench_run_id()
    ctx_key, gpu, h_feats = ctx.structural_key(), ctx.hardware_id(), ctx.features()
    rows = []
    for leaf in leaves:
        features = {**h_feats, **leaf.knobs}
        rows.append(
            NodeRow(
                node_key=node_key(ctx_key, gpu, leaf.op_sig, features),
                parent_key=None,
                context_key=ctx_key,
                op_sig=leaf.op_sig,
                features=features,
                value_us=leaf.value_us,
                depth=0,  # a bench leaf has no search tree above it
                gpu=gpu,
                visits=1,
                is_leaf=True,
                variance=leaf.variance,
                n_samples=leaf.n_samples,
                status=leaf.status,
                run_id=run_id,
            )
        )
    db = SearchDB(Path(db_path))
    try:
        db.record_nodes(rows)
    finally:
        db.close()
    return len(rows)
