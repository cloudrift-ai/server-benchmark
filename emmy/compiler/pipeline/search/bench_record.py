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

Declarative identity (``run --bench --record-shape``): the caller may pass a
golden-spelled shape spec (:func:`parse_record_shape`) naming what the snippet benches.
It is attached PER LEAF, not blanket: a leaf gets the spec only when its own stamped
``S_*`` extents key to the spec's :class:`~.data.shape.ShapeKey` — the sanctioned
golden↔measured join — so the secondary ops a multi-op snippet realizes (a
``linear_norm`` snippet's producer matmul, an attention trace's plain QKᵀ contraction)
stay identity-less like legacy rows. Identity-carrying rows are what the goldens-format
measurement freeze (``data/freeze.py``) keeps.

Pool fidelity: the tune keys a pool by ``op_sig`` — a digest over the **pre-descent**
offer op's ``S_*`` stamps, NOT the terminal kernel's (descent stamps further ``S_*``
deltas, so the two differ for most ops). :func:`bench_leaves` recovers the offer site
from each compiled kernel via ``source_chain`` (the ``two_level`` decomposition idiom:
deepest loop-dialect ancestor carrying ``S_*`` knobs, tile-dialect fallback for the mma
path) and groups a variant's kernels (e.g. a split-K main + combine pair) under one
site — an auxiliary kernel with no provenance at all attributes to its nearest sited
producer through the graph edges — one leaf per (variant, op), valued at the group's
summed per-launch time, exactly the tune's whole-variant leaf semantics.
Rows are keyed with the tune's own recipes (same ``node_key`` / ``op_sig`` /
``context_key``), parentless with ``depth=0`` — the no-tree-schema marker the fork
diagnostics skip — and stamped with a ``bench-…`` ``run_id`` so freeze headers show the
provenance.
"""

from __future__ import annotations

import json
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


@dataclass(frozen=True)
class RecordShape:
    """A validated ``--record-shape`` spec: the golden-spelled identity dict verbatim
    (what lands on ``NodeRow.shape_spec``) plus the :class:`~.data.shape.ShapeKey` the
    spec's kind class derives — the golden-side half of the per-leaf identity join."""

    spec: dict
    key: object  # ShapeKey (kept untyped here to avoid a module-level data import)


def parse_record_shape(text: str) -> RecordShape:
    """Parse and validate a ``--record-shape`` JSON spec (the golden YAML entry spelling
    minus knobs/latencies: ``{"kernel": ..., <shape fields>, "dynamic": ...}``).

    Validation IS instantiation: the spec round-trips through the golden kind dataclass
    (``_KERNEL_CLASSES``), so unknown kinds, missing/extra fields, and kind-specific
    ``__post_init__`` invariants (e.g. the dynamic-hint check) all raise here, and the
    returned ``key`` comes from the same ``shape_key()`` every golden consumer uses.
    Raises ``ValueError`` with the underlying reason on any invalid spec."""
    from emmy.compiler.pipeline.search.golden import _KERNEL_CLASSES  # noqa: PLC0415

    try:
        spec = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"--record-shape is not valid JSON: {e}") from e
    if not isinstance(spec, dict) or "kernel" not in spec:
        raise ValueError('--record-shape must be a JSON object with a "kernel" field')
    cls = _KERNEL_CLASSES.get(spec["kernel"])
    if cls is None:
        raise ValueError(f"--record-shape: unknown kernel kind {spec['kernel']!r} (expected one of {sorted(_KERNEL_CLASSES)})")
    try:
        cfg = cls(name="record-shape", knobs={}, **{k: v for k, v in spec.items() if k != "kernel"})
    except (TypeError, ValueError) as e:
        raise ValueError(f"--record-shape: invalid {spec['kernel']!r} spec: {e}") from e
    return RecordShape(spec=spec, key=cfg.shape_key())


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
    site = fallback = None
    for anc in op.source_chain():
        if not any(k.startswith("S_") for k in getattr(anc, "knobs", {}) or {}):
            continue
        dialect = anc.dialect
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

    nids = [nid for nid in compiled.topological_order() if isinstance(compiled.nodes[nid].op, CudaOp)]
    per_launch = list(getattr(bench, "per_launch", None) or []) if bench is not None else []
    entries = []  # (nid, op, launch, sig-or-None) in launch order
    sig_by_nid: dict[str, str] = {}
    for idx, nid in enumerate(nids):
        op = compiled.nodes[nid].op
        site = _offer_site(op)
        sig = digest(*sorted((k, v) for k, v in site.knobs.items() if k.startswith("S_"))) if site is not None else None
        if sig is not None:
            sig_by_nid[nid] = sig
        entries.append((nid, op, per_launch[idx] if idx < len(per_launch) else None, sig))

    def attributed(nid: str, hops: int = 4) -> str | None:
        """The site group of an orphan kernel's nearest sited PRODUCER. An auxiliary
        kernel synthesized at materialize (a split-K combine) carries no ``S_*``
        provenance anywhere in its chain, but it consumes its main kernel's output —
        the graph edge is the attribution the source chain lost. Without this the
        combine was silently dropped and a split-K variant recorded a partial-only
        value: systematically fast-biased against the tune's whole-slice leaves in
        the same pool (found by the 2026-07-16 4090 verification)."""
        frontier = [nid]
        for _ in range(hops):
            nxt: list[str] = []
            for cur in frontier:
                node = compiled.producer(cur)
                for pid in getattr(node, "inputs", None) or []:
                    if pid in sig_by_nid:
                        return sig_by_nid[pid]
                    nxt.append(pid)
            frontier = nxt
        return None

    groups: dict[str, dict] = {}
    skipped = 0
    for nid, op, launch, sig in entries:
        sig = sig or attributed(nid)
        if sig is None:
            skipped += 1
            logger.debug("[record-nodes] kernel %s has no recoverable offer site and no sited producer — skipped", nid)
            continue
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


def record_bench_leaves(
    db_path: Path | str, ctx: Context, leaves: list[BenchLeaf], *, run_id: str | None = None, shape: RecordShape | None = None
) -> int:
    """Key ``leaves`` with the tune's own recipes and upsert them into the node store
    at ``db_path`` — parentless ``depth=0`` leaf rows under the live context's regime
    (``run --bench`` compiles at the deployable flags, so these land in the -O3 lane
    the store is censored in). Returns the number of rows offered to
    :meth:`SearchDB.record_nodes` (its plausibility gate and quality-aware leaf
    replacement still apply per row).

    ``shape`` (from ``--record-shape``) stamps the declarative identity onto exactly
    the leaves whose stamped ``S_*`` extents key to ``shape.key`` (see the module
    docstring); a spec that matches NO leaf warns loudly — a systematic key mismatch
    must not silently produce an identity-less sweep."""
    from emmy.compiler.pipeline.search.data.shape import ShapeKey  # noqa: PLC0415
    from emmy.compiler.pipeline.search.db import NodeRow, SearchDB  # noqa: PLC0415
    from emmy.compiler.structural import digest  # noqa: PLC0415

    if not leaves:
        return 0
    run_id = run_id or mint_bench_run_id()
    ctx_key, gpu, h_feats = ctx.structural_key(), ctx.hardware_id(), ctx.features()
    rows = []
    n_tagged = 0
    for leaf in leaves:
        features = {**h_feats, **leaf.knobs}
        tun = tuple(sorted((k, str(v)) for k, v in features.items() if not k.startswith(("S_", "H_"))))
        spec = None
        if shape is not None and ShapeKey.from_s_features(leaf.knobs).joins(shape.key):
            spec = shape.spec
            n_tagged += 1
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
                shape_spec=spec,
            )
        )
    if shape is not None and n_tagged == 0:
        logger.warning(
            "[record-nodes] --record-shape %s matched none of the %d benched leaf/leaves — rows record "
            "identity-less and will not enter goldens-format freezes (shape-key mismatch; please report)",
            shape.spec.get("kernel"),
            len(rows),
        )
    db = SearchDB(Path(db_path))
    try:
        db.record_nodes(rows)
    finally:
        db.close()
    return len(rows)
