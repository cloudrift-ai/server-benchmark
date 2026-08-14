"""Cross-validated golden-rank evaluation for offline-prior fits — the fold machinery
behind :func:`~.run.run_fit` (the ``emmy fit`` run harness).

Works entirely on pre-built :class:`~.group.Group` lists (the command layer owns case
building, which needs the snippet tracer ``pipeline/`` must not import) and produces the
run's metrics dict, so every piece here is testable on synthetic cases with no tracing.

The report structure (one :func:`run_axis` block per fold axis) is standard grouped
cross-validation: individual folds are construction machinery; the pooled tables are the
result. Each golden is held out exactly ONCE per axis — its ``holdout`` rank comes from
the single fold model that never trained on it, its ``train`` rank is the median across
the fold models that did — and the per-card ``gap`` (holdout median − train median)
separates overfitting (train good, holdout bad) from a too-weak model class (both bad,
small gap). Per-fold medians survive only in ``fold_detail`` as the spread/noise view,
plus ``excluded`` folds with reasons: a fold whose training slice can't fit a weight set
the holdout needs is dropped loudly, never scored with a stale or empty vector.

Aggregates are per card ONLY — pooling cards is the failure mode the 2026-07 sweeps
documented (a pooled win trading one arch against the other), so the shape forbids it.

Fold models are seeded from ZEROS (not the incumbent artifact): the incumbent's weights
were themselves fit on every golden, so seeding folds from them would leak each held-out
golden into its own holdout model. The full-train model (the shippable artifact) keeps
the incumbent seeding — the difference is recorded in the metrics header.
"""

from __future__ import annotations

import statistics

from emmy.compiler.pipeline.search.prior.fit.group import Group
from emmy.compiler.pipeline.search.prior.fit.rank import dual_rank

TOP_KS = (1, 10, 25, 50, 100)

# The ``skipped`` reason for golden kinds the fitter has no case builder for
# (attention / rms_norm / softmax) — counted per card as ``out_of_scope``, distinct from
# ``unranked`` (a case-buildable golden whose enumeration failed to contain it).
OUT_OF_SCOPE = "kernel kind not case-buildable"


def fold_key(case: Group, axis: str) -> str:
    if axis not in ("op_family", "gpu"):
        raise ValueError(f"unknown fold axis {axis!r}")
    return case.gpu if axis == "gpu" else case.family


def _unfittable(trainer, train: list[Group], hold: list[Group]) -> str | None:
    """Why this fold cannot be scored, or ``None`` — the TRAINER's answer, since what makes a
    training slice unfittable is a property of the model class. The linear trainer has two such
    constraints (its weight sets); a tree fit has none, and a trainer that declares no
    ``unfittable`` (the test stubs, and any model class with no structural requirement on its
    training slice) is taken at its word.

    A fold that IS unfittable is dropped with its reason recorded, never scored with a stale or
    empty model."""
    check = getattr(trainer, "unfittable", None)
    return check(train, hold) if check is not None else None


def case_ranks(case: Group, model) -> tuple[int, int] | None:
    """The case's golden ``(rank, rank_optimistic)`` under a fitted model — any object
    with the trainer protocol's ``score_rows(group) -> scores | None`` (higher =
    predicted faster; the linear model's weight-set selection lives inside it).
    ``None`` when the model can't score the group (an unfittable fold — callers
    exclude it up front)."""
    scores = model.score_rows(case)
    if scores is None:
        return None
    return dual_rank(scores, case.pinned_idx)


def _median(vals: list[float]) -> float:
    return round(float(statistics.median(vals)), 2)


def _per_card(entries: list[tuple[Group, tuple[int, int]]]) -> dict:
    """Per-card aggregates over ``(case, (rank, rank_optimistic))`` rows: count, median
    and top-k coverage under both tie conventions. Cards never pool."""
    by_gpu: dict[str, list[tuple[int, int]]] = {}
    for case, ranks in entries:
        by_gpu.setdefault(case.gpu, []).append(ranks)
    out = {}
    for gpu, ranks in sorted(by_gpu.items()):
        pes, opt = [r for r, _ in ranks], [o for _, o in ranks]
        out[gpu] = {
            "n": len(ranks),
            "median": _median(pes),
            "median_optimistic": _median(opt),
            "top": {str(k): sum(r < k for r in pes) for k in TOP_KS},
            "top_optimistic": {str(k): sum(r < k for r in opt) for k in TOP_KS},
        }
    return out


def evaluate_full_train(cases: list[Group], model) -> dict:
    """The ``full_train`` metrics block: every case ranked under the shippable model."""
    entries = [(c, r) for c in cases if (r := case_ranks(c, model)) is not None]
    per_golden = {c.key: {"rank": r, "rank_optimistic": o, "pool": len(c.feats)} for c, (r, o) in entries}
    return {"per_golden": per_golden, "per_card": _per_card(entries)}


def run_axis(cases: list[Group], axis: str, *, trainer) -> dict:
    """One fold axis's full cross-validation → its ``cv.<axis>`` metrics block.

    ``trainer`` is any object with ``fit(groups) -> model`` where the model satisfies
    :func:`case_ranks`' protocol. ONE instance serves every fold: the trainer is immutable and
    its ``fit`` is pure, so a fold's fit never depends on how many folds ran before it — adding a
    golden family changes that family's fold and nothing else, keeping cross-run diffs meaningful.
    The caller passes the FOLD trainer, seeded from zeros rather than the incumbent, since the
    incumbent's weights were fit on every golden and would leak each held-out golden into its own
    holdout model. Guard: a fold is excluded (with a recorded reason) when its training slice has
    no static cases (the dynamic stage seeds from the static fit, so nothing is fittable) or when
    its holdout needs the dynamic set and the training slice has no dynamic cases."""
    folds = sorted({fold_key(c, axis) for c in cases})
    holdout: list[tuple[Group, tuple[int, int]]] = []
    holdout_fold: dict[str, str] = {}
    train_acc: dict[str, tuple[Group, list[int], list[int]]] = {}
    fold_medians: dict[str, dict] = {}
    excluded: dict[str, str] = {}

    for f in folds:
        train = [c for c in cases if fold_key(c, axis) != f]
        hold = [c for c in cases if fold_key(c, axis) == f]
        if reason := _unfittable(trainer, train, hold):
            excluded[f] = reason
            continue
        model = trainer.fit(train)
        hold_entries = [(c, r) for c in hold if (r := case_ranks(c, model)) is not None]
        holdout.extend(hold_entries)
        holdout_fold.update({c.key: f for c, _ in hold_entries})
        for c in train:
            if (r := case_ranks(c, model)) is not None:
                acc = train_acc.setdefault(c.key, (c, [], []))
                acc[1].append(r[0])
                acc[2].append(r[1])
        fold_medians[f] = {"median": _median([r for _, (r, _) in hold_entries]) if hold_entries else None, "n": len(hold_entries)}

    train_entries = [(c, (_median(pes), _median(opt))) for c, pes, opt in train_acc.values()]
    holdout_cards = _per_card(holdout)
    train_cards = _per_card(train_entries)
    return {
        "holdout": {
            "per_golden": {c.key: {"rank": r, "rank_optimistic": o, "fold": holdout_fold[c.key]} for c, (r, o) in holdout},
            "per_card": holdout_cards,
        },
        "train": {
            "per_golden": {c.key: {"rank": r, "rank_optimistic": o} for c, (r, o) in train_entries},
            "per_card": train_cards,
        },
        "gap": {gpu: round(holdout_cards[gpu]["median"] - train_cards[gpu]["median"], 2) for gpu in holdout_cards if gpu in train_cards},
        "fold_detail": {"holdout_medians": fold_medians, "excluded": excluded},
    }


def build_metrics(header: dict, cases: list[Group], skipped: list[tuple[str, str, str]], full_model, cv: dict[str, dict]) -> dict:
    """Assemble the run's complete metrics dict (JSON-ready, deterministic — no
    timestamps or host info; the caller serializes with sorted keys). ``skipped`` rows
    are ``(gpu, name, reason)`` for goldens that never became cases: enumeration
    failures count per card as ``unranked`` (never silently dropped), kinds the fitter
    doesn't case-build (attention / rms_norm / softmax) as ``out_of_scope``."""
    full = evaluate_full_train(cases, full_model)
    for gpu, _name, reason in skipped:
        card = full["per_card"].setdefault(gpu, {"n": 0, "median": None, "median_optimistic": None, "top": {}, "top_optimistic": {}})
        kind = "out_of_scope" if reason == OUT_OF_SCOPE else "unranked"
        card[kind] = card.get(kind, 0) + 1
    for card in full["per_card"].values():
        card.setdefault("unranked", 0)
        card.setdefault("out_of_scope", 0)
    return {"header": header, "full_train": full, "cv": cv}
