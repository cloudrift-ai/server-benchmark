r"""The training corpus for a best-latency estimate: one row per candidate pool, labelled with the
fastest latency any golden recorded for that kernel on that card.

**One row per pool, not per golden.** ``commands/fit.py`` states the unit — "a group is a
candidate pool, not a golden" — and several goldens can land on one pool: the same shape recorded
twice, or under two names. Their labels are competing measurements of one kernel's best, so the
row takes the minimum. That is also what makes a duplicate recording harmless: the label means
"the best anyone has achieved", so a second, slower recording of the same kernel cannot drag it.

**Nothing here enumerates a candidate pool**, which is the whole reason this is cheap and is not
merely an optimization. The ranking fit must enumerate a pool and locate the golden's own row
inside it, which is why ``fit/cv.py`` marks whole kernel kinds — attention, rms_norm, softmax — as
"not group-buildable" and drops them. This estimate needs only the kernel and its measured best,
so those kinds are ordinary rows here, and roughly a fifth of the corpus becomes usable that was
not before.

It sits in ``data/`` because that is what it is — a read-view over one of the measurement-data
sources, turning golden configs into one labelled row apiece, which is the package's stated job.
It is NOT beside ``build_golden_groups`` in ``commands/`` because it does not need what put that
function there: the snippet tracer, which only the enumerating half uses. Reconstruction here
stops at ``LOOP_PASSES``.

Every row is built from the kernel as the loop passes leave it — the same op an ordinary lowering
carries forward, since lifting to Tile IR inherits the stamps rather than recomputing them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from emmy import gpu
from emmy.compiler.pipeline.knob import SCHEDULE_FAMILIES, family_of
from emmy.compiler.pipeline.search.kernel_cost import kernel_row

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CostRow:
    """One kernel on one card: what it looks like, what its best measured latency was, and the two
    keys a fit needs to group it correctly."""

    #: The pool this row stands for — ``GoldenRecord.pool_group``, i.e. (card, kernel identity,
    #: pins). The row identity, and the reason a fast-math kernel and its standard sibling are
    #: separate rows: the pin is part of the key.
    key: tuple
    #: The cross-validation grouping (:attr:`~..data.shape.ShapeKey.fold_key`) — card-blind, and
    #: folding a ``.dynM`` kernel together with the static twin whose pool it shares. Holding out
    #: on anything finer lets a model see a kernel on one card and be scored on it on another.
    fold: str
    gpu: str
    name: str
    #: The kernel family, for grouping a report — ``""`` when the stamps match no sweep kind.
    #: Carried rather than re-derived from ``features``: the builder holds the ``ShapeKey``
    #: already, and a reader should not have to reconstruct one from the model's input vector.
    kind: str
    features: dict[str, float]
    #: The label: the fastest ``emmy_us`` any golden in this pool recorded.
    best_us: float
    #: How many goldens shared this pool. A DIAGNOSTIC, deliberately not a training weight: every
    #: label is the best *found*, never the best possible, and the gap is unmeasurable from
    #: goldens alone. What this supports is asking whether that censoring tracks how much
    #: exploration a kernel got, by splitting a bias report on singleton versus multi-record rows.
    members: int = 1


def _records_by_pool(records) -> dict[tuple, list]:
    """Records grouped by the pool each describes, keyed on ``GoldenRecord.pool_group`` — "the ONE
    place that question is answered", as its own docstring puts it."""
    pools: dict[tuple, list] = {}
    for r in records:
        pools.setdefault(r.pool_group, []).append(r)
    return pools


def _records_no_schedule(members) -> bool:
    """Every member records NO schedule family at all — the fabricated-label class.

    Such a record cannot say which candidate it verified, so it silently matches option 0 of
    whatever pool it opens and a fit trains on that as the verified optimum. Distinct from a
    record that spells the families and sets them OFF (``TILE: '' WORK: '' ...``), which is an
    honest scalar-tier realization and stays: the difference is whether the keys are present."""
    return not any(any(family_of(k) in SCHEDULE_FAMILIES for k in r.knobs) for r in members)


def build_rows(records: list | None = None) -> tuple[list[CostRow], list[tuple[str, str, str]]]:
    """The corpus, plus what was left out of it as ``(gpu, name, reason)``.

    Every golden that does not become a row is reported rather than dropped quietly, so a caller
    can say how much of the corpus it is actually fitting on.

    Exclusions are by **validity, never by magnitude**. The slowest labels in the corpus are
    multi-second V100 kernels realized on the scalar tier, and they are honest measurements of
    genuinely catastrophic kernels — exactly the signal a fuse-or-cut decision needs, since the
    whole question is whether one big kernel beats several small ones. Dropping outliers here
    would delete the examples the estimate exists to learn from. What does get dropped is a label
    that cannot be trusted to describe anything at all: a record naming no schedule.

    Deliberately NOT re-checked here: whether a latency is physically possible. That gate exists
    (``db.py:implausible_value_reason`` for measured rows, ``run.py:_intensity_floor_flag`` for a
    golden A/B) and belongs with the recording, not with a reader — it rejects 0 of 1285 goldens
    and can judge only 593 of them, so running it again here would be machinery that has never
    had an opinion."""
    # Deferred like ``dataset.py``'s: ``golden.py`` imports this package for ``ShapeKey``, so a
    # module-level import here would close the cycle.
    from emmy.compiler.pipeline.search.golden import GOLDEN_RECORDS, _target_kernel_nodes  # noqa: PLC0415

    rows: list[CostRow] = []
    skipped: list[tuple[str, str, str]] = []
    for members in _records_by_pool(GOLDEN_RECORDS if records is None else records).values():
        namer = members[0]
        if _records_no_schedule(members):
            skipped.extend((m.gpu_name, m.name, "no schedule family recorded") for m in members)
            continue
        try:
            # Once per POOL, not per member: every member spells the same kernel, which is what
            # the pool key means. Note the corpus is still lowered TWICE overall — ``pool_group``
            # lowers each record to compose its key (~10 s), and this lowers each pool again
            # (~10 s), because ``_target_kernel_nodes`` caches nothing and memoizing it would be
            # unsafe: ``_lifted_target`` mutates the graph it is handed. ~20 s for the whole
            # corpus is not worth unpicking that.
            lowered, nodes = _target_kernel_nodes(namer)
        except Exception as exc:  # noqa: BLE001 — a record that no longer resolves is data, not a crash
            skipped.extend((m.gpu_name, m.name, f"does not lower: {exc}") for m in members)
            continue
        if len(nodes) != 1:
            # The label measures the whole target, so several kernels under one record would need
            # dividing between them — and nothing records how. The plan measured exactly one node
            # for all 1285 records; if that ever stops holding, drop the record loudly rather than
            # price the first kernel and silently discard the rest of the work and traffic.
            skipped.extend((m.gpu_name, m.name, f"target resolves to {len(nodes)} kernels") for m in members)
            continue
        node = nodes[0]
        op = node.op.with_io(lowered, node)
        spec = gpu.by_name(namer.gpu_name)
        best_us = min(m.emmy_us for m in members)
        features = kernel_row(op, spec)
        rows.append(
            CostRow(
                key=namer.pool_group,
                fold=namer.shape_key.fold_key,
                gpu=namer.gpu_name,
                name=namer.name,
                kind=namer.shape_key.kind,
                features=features,
                best_us=best_us,
                members=len(members),
            )
        )
    logger.info("cost dataset: %d rows over %d fold groups (%d goldens skipped)", len(rows), len({r.fold for r in rows}), len(skipped))
    return rows, skipped
