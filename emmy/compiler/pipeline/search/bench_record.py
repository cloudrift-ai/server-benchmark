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
  :data:`FAIL_SENTINEL_US`, not a measurement — consumers read ``status``);
- the greedy pick, from its ``greedy (isolated)`` re-bench — pinned-comparable by
  construction, so every benched pool self-anchors the prior's argmax.

Never recorded: ``pin_unmatched`` rows (the claimed config never ran — and "not offered"
is not "doesn't launch"), rows carrying an integrity flag (wrong-answer / intensity
floor: the measurement is untrue), and anything from the ``--ir`` path (serialization
drops ``op.knobs``, so there is no honest feature dict). The caller
(``emmy/commands/run.py``) owns those exclusions; this module records what it is given.

Pool fidelity: the tune keys a pool by ``op_sig`` — a digest over the **pre-descent**
offer op's ``S_*`` stamps, NOT the terminal kernel's (descent stamps further ``S_*``
deltas, so the two differ for most ops). :func:`bench_leaves` recovers the offer site
from each compiled kernel via ``source_chain`` (the ``two_level`` decomposition idiom:
deepest loop-dialect ancestor carrying ``S_*`` knobs) and groups a variant's kernels
(e.g. a split-K main + combine pair) under one site — one leaf per (variant, op), valued
at the group's summed per-launch time, exactly the tune's whole-variant leaf semantics.
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
    """One (variant, op) measurement extracted from a benched compiled graph."""

    op_sig: str  # the tune's pool key — pre-descent offer-site S_* digest
    knobs: dict  # realized knob dict (S_* stamps + tunables) of the group's main kernel
    value_us: float  # whole-variant latency for this op (summed launches); sentinel on fail
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


def _offer_site(op) -> object | None:
    """The pre-descent offer op a compiled kernel lowered from. Preferred: the deepest
    loop-dialect ancestor carrying ``S_*`` stamps (the ``two_level`` decomposition-row
    idiom). The tensor-core (mma) tile-lowering does NOT preserve a ``LoopOp`` in the
    ``.source`` chain — it bottoms at a ``TileOp`` — so with no loop ancestor the
    deepest tile-dialect ``S_*``-carrying ancestor stands in: it holds the
    recognize-time ``S_*`` set unchanged (descent's extra ``S_*`` stamps land on
    kernel/cuda-dialect ops), so its digest equals the tune-written ``op_sig``
    (verified against real tune rows on an RTX 4090 — without the fallback, every
    mma-path kernel was silently unrecordable, i.e. exactly the fast tensor-core
    variants this module exists to capture). ``None`` when neither exists."""
    from emmy.compiler.pipeline.search.keys import dialect_of, source_chain  # noqa: PLC0415

    site = fallback = None
    for anc in source_chain(op):
        if not any(k.startswith("S_") for k in getattr(anc, "knobs", {}) or {}):
            continue
        dialect = dialect_of(anc)
        if dialect == "loop":
            site = anc
        elif dialect == "tile":
            fallback = anc
    return site or fallback


def bench_leaves(compiled, bench, *, status: str = "ok") -> list[BenchLeaf]:
    """Extract one :class:`BenchLeaf` per (variant, op) from a benched compiled graph.

    Kernels are paired with ``bench.per_launch`` by topological order (the launch
    order) and grouped by their offer site: a multi-kernel variant (split-K main +
    combine) contributes ONE leaf valued at the group's summed launch time — the
    tune's whole-variant leaf semantics; a fragment kernel's own tiny latency never
    becomes a row (the pre-#330 poison class). Bench stats (variance / n_samples)
    carry only for single-kernel groups — per-launch windows replay each kernel
    back-to-back, so cross-kernel samples don't align iter-wise and a summed variance
    would be fiction. ``status="bench_fail"`` (with ``bench=None``) emits sentinel
    leaves for a realized config that failed to bench. A kernel with no recoverable
    offer site is skipped with a debug note."""
    from emmy.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415
    from emmy.compiler.structural import digest  # noqa: PLC0415

    ops = [compiled.nodes[nid].op for nid in compiled.topological_order() if isinstance(compiled.nodes[nid].op, CudaOp)]
    per_launch = list(getattr(bench, "per_launch", None) or []) if bench is not None else []
    groups: dict[str, dict] = {}
    skipped = 0
    for idx, op in enumerate(ops):
        site = _offer_site(op)
        if site is None:
            skipped += 1
            logger.debug("[record-nodes] kernel %d has no recoverable offer site — skipped", idx)
            continue
        sig = digest(*sorted((k, v) for k, v in site.knobs.items() if k.startswith("S_")))
        launch = per_launch[idx] if idx < len(per_launch) else None
        g = groups.setdefault(sig, {"ops": [], "launches": []})
        g["ops"].append(op)
        g["launches"].append(launch)
    if skipped and not groups:
        # Silence must never read as success: a graph whose EVERY kernel lost its offer
        # site means a provenance gap in some lowering path, not "nothing to record".
        logger.warning(
            "[record-nodes] none of the %d kernel(s) has a recoverable offer site — nothing recorded "
            "(a lowering path is not preserving op provenance; please report)",
            skipped,
        )
    leaves = []
    for sig, g in groups.items():
        # The main kernel carries the variant's descent stamps; auxiliaries (a split-K
        # combine) carry a subset — the most-tunables op is the variant's knob identity.
        main = max(g["ops"], key=lambda o: sum(1 for k in (o.knobs or {}) if not k.startswith(("S_", "H_"))))
        knobs = dict(main.knobs or {})
        if status != "ok":
            leaves.append(BenchLeaf(op_sig=sig, knobs=knobs, value_us=FAIL_SENTINEL_US, variance=None, n_samples=None, status=status))
            continue
        if any(launch is None for launch in g["launches"]):
            logger.debug("[record-nodes] op %s missing per-launch timings — skipped", sig[:12])
            continue
        value_us = sum(launch.time_ms for launch in g["launches"]) * 1000.0
        variance = n_samples = None
        if len(g["launches"]) == 1 and g["launches"][0].samples:
            samples_us = [s * 1000.0 for s in g["launches"][0].samples]
            n_samples = len(samples_us)
            variance = statistics.pvariance(samples_us) if n_samples >= 2 else None
        leaves.append(BenchLeaf(op_sig=sig, knobs=knobs, value_us=value_us, variance=variance, n_samples=n_samples, status="ok"))
    return leaves


def record_bench_leaves(db_path: Path | str, ctx: Context, leaves: list[BenchLeaf], *, run_id: str | None = None) -> int:
    """Key ``leaves`` with the tune's own recipes and upsert them into the node store
    at ``db_path`` — parentless ``depth=0`` leaf rows under the live context's regime
    (``run --bench`` compiles at the deployable flags, so these land in the -O3 lane
    the store is censored in). Returns the number of rows offered to
    :meth:`SearchDB.record_nodes` (its plausibility gate and quality-aware leaf
    replacement still apply per row)."""
    from emmy.compiler.pipeline.search.db import NodeRow, SearchDB  # noqa: PLC0415
    from emmy.compiler.structural import digest  # noqa: PLC0415

    if not leaves:
        return 0
    run_id = run_id or mint_bench_run_id()
    ctx_key, gpu, h_feats = ctx.structural_key(), ctx.hardware_id(), ctx.features()
    rows = []
    for leaf in leaves:
        features = {**h_feats, **leaf.knobs}
        tun = tuple(sorted((k, str(v)) for k, v in features.items() if not k.startswith(("S_", "H_"))))
        rows.append(
            NodeRow(
                node_key=digest(ctx_key, gpu, leaf.op_sig, tun),
                parent_key=None,
                context_key=ctx_key,
                op_sig=leaf.op_sig,
                features=features,
                value_us=leaf.value_us,
                depth=0,  # no tree schema — the marker the fork diagnostics skip
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
