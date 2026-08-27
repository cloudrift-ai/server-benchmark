"""Cross-validated golden-rank evaluation for offline-prior fits — the fold machinery
behind :func:`~.run.run_fit` (the ``emmy fit`` run harness).

Works entirely on pre-built :class:`GoldenGroup` lists (the command layer owns group
building, which needs the snippet tracer ``pipeline/`` must not import) and produces the
run's metrics dict, so every piece here is testable on synthetic groups with no tracing.

The report is standard grouped cross-validation: individual folds are construction machinery;
the pooled tables are the result. Each GROUP is held out exactly ONCE — its ``holdout`` rank
comes from the single fold model that never trained on it, its ``train`` rank is the median
across the fold models that did — and the per-card ``gap`` (holdout median − train median)
separates overfitting (train good, holdout bad) from a too-weak model class (both bad,
small gap). Per-fold medians survive only in ``fold_detail`` as the spread/noise view,
plus ``excluded`` folds with reasons: a fold whose training slice can't fit a weight set
the holdout needs is dropped loudly, never scored with a stale or empty vector.

A group is ONE candidate pool, which the builder may have matched several goldens into. Its rank
is then the best over those positives (:func:`~..metrics.best_dual_rank`), and every ``per_golden``
row carries ``positives`` so a merged group is never mistaken for a single-golden one — the count
moves when the corpus grows a sibling, and two metrics files are only comparable knowing that.

**Folds group by SHAPE** (:attr:`GoldenGroup.shape` — the extent identity), five of them, balanced by
group count. That is the deploy question stated as an experiment: a new shape, on a card we have
data for, no measurements. Two retired axes are worth naming so they do not come back:

- ``op_family``, keyed on the golden's NAME family, both leaked and barely folded. What decides
  whether two goldens compete over one candidate pool is their extent identity, not their name:
  178 shape groups spanned more than one family, covering 695 of 1385 goldens, so a "held-out"
  golden was routinely scored by a model trained on its own pool. And at 891 families over 951
  names it was nearly leave-one-out — ~891 refits, which is why cross-validation was never run.
- ``gpu`` (leave-one-card-out) asked a transfer question the deploy path does not face, and with
  720/348/297/10/10 goldens per card mostly measured sample-size imbalance.

Note the difference in kind: a shape group must be held out on EVERY card at once (173 groups span
more than one), because the same shape's answer on another card is still the answer.

Aggregates are per card ONLY — pooling cards is the failure mode the 2026-07 sweeps
documented (a pooled win trading one arch against the other), so the shape forbids it. That is a
REPORT axis and always was; it is independent of what the folds group by.

Fold models are seeded from ZEROS (not the incumbent artifact): the incumbent's weights
were themselves fit on every golden, so seeding folds from them would leak each held-out
golden into its own holdout model. The full-train model (the shippable artifact) keeps
the incumbent seeding — the difference is recorded in the metrics header.
"""

from __future__ import annotations

import statistics

from emmy.compiler.pipeline.search.data.group import GoldenGroup
from emmy.compiler.pipeline.search.metrics import best_dual_rank
from emmy.compiler.pipeline.search.prior.report import Summary, rank_metrics

# The ``skipped`` reason for golden kinds the fitter has no group builder for
# (attention / rms_norm / softmax) — counted per card as ``out_of_scope``, distinct from
# ``unranked`` (a group-buildable golden whose enumeration failed to contain it).
OUT_OF_SCOPE = "kernel kind not group-buildable"


DEFAULT_FOLDS = 5


def assign_folds(groups: list[GoldenGroup], k: int = DEFAULT_FOLDS) -> dict[str, int]:
    """Shape group → fold index, balanced by group count.

    Groups are wildly uneven — on the golden dataset the largest holds 122 groups, the next 14, and 162 are
    singletons — so groups are assigned largest-first to whichever fold is currently smallest. Random or hashed
    assignment would let one fold carry the 122-group group and leave the rest thin, and per-fold medians over a
    thin fold are noise. Measured on the real dataset this lands 277 groups in every one of 5 folds.

    Ties break on the group key, so the assignment is a pure function of the group list and a re-run reproduces
    it exactly."""
    sizes: dict[str, int] = {}
    for c in groups:
        sizes[c.shape] = sizes.get(c.shape, 0) + 1
    loads = [0] * k
    out: dict[str, int] = {}
    for shape, n in sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0])):
        f = min(range(k), key=lambda i: (loads[i], i))
        out[shape] = f
        loads[f] += n
    return out


def _unfittable(trainer, train: list[GoldenGroup], hold: list[GoldenGroup]) -> str | None:
    """Why this fold cannot be scored, or ``None`` — the TRAINER's answer, since what makes a
    training slice unfittable is a property of the model class. The linear trainer has two such
    constraints (its weight sets); a tree fit has none, and a trainer that declares no
    ``unfittable`` (the test stubs, and any model class with no structural requirement on its
    training slice) is taken at its word.

    A fold that IS unfittable is dropped with its reason recorded, never scored with a stale or
    empty model."""
    check = getattr(trainer, "unfittable", None)
    return check(train, hold) if check is not None else None


def group_ranks(group: GoldenGroup, model) -> tuple[int, int] | None:
    """The group's best golden ``(rank, rank_optimistic)`` under a fitted model — any object
    with the trainer protocol's ``score_rows(group) -> scores | None`` (higher =
    predicted faster; the linear model's weight-set selection lives inside it).
    ``None`` when the model can't score the group (an unfittable fold — callers
    exclude it up front). A pool with several verified configs is scored on the best
    of them: deploy ships one, so any of them ranked first is the same win."""
    scores = model.score_rows(group)
    if scores is None:
        return None
    return best_dual_rank(scores, group.golden_ids)


def _median(vals: list[float]) -> float:
    """One decimal, matching :func:`~..report.rank_metrics` — so every median in the file, summary or
    fold detail, is quoted to the same precision. Ranks are integers, so the second digit this used to
    carry was padding everywhere except the median-of-medians the ``train`` block takes."""
    return round(float(statistics.median(vals)), 1)


def _per_card(entries: list[tuple[GoldenGroup, tuple[int, int]]], *, cv_split: str) -> list[Summary]:
    """Per-card :class:`~..report.Summary`s over ``(group, (rank, rank_optimistic))`` rows. Cards never pool.

    Exactly the summary ``emmy eval prior`` emits — same four fields, same :func:`~..report.rank_metrics` — so a
    fit's metrics file and an eval report state the golden screen identically instead of two summarisers
    agreeing by coincidence. ``split`` rides in the axes because one file carries three of these (the
    shippable model's, the holdout, and the fold models' training ranks) and a row has to say which it is.
    Not "stage": this package already spends that word on the linear fit's static/dynamic stages, and the
    compiler spends it again on pipeline stages and the ``STAGE`` knob family.

    ``unscored`` is 0 by construction — a group the model cannot score is dropped by :func:`group_ranks`
    before it reaches here. Goldens that never became a group at all are a fact about the CORPUS, not about
    a scored card, and are counted beside the summaries by :func:`build_metrics`."""
    by_gpu: dict[str, list[tuple[int, int]]] = {}
    for group, ranks in entries:
        by_gpu.setdefault(group.gpu, []).append(ranks)
    return [
        Summary(axes={"cv_split": cv_split, "gpu": gpu}, groups=len(ranks), unscored=0, metrics=rank_metrics(ranks))
        for gpu, ranks in sorted(by_gpu.items())
    ]


def evaluate_full_train(groups: list[GoldenGroup], model) -> dict:
    """The ``full_train`` metrics block: every group ranked under the shippable model."""
    entries = [(c, r) for c in groups if (r := group_ranks(c, model)) is not None]
    # ``pool`` is the pool's TRUE size and ``sampled`` how many of its rows this fit saw; the rank is
    # the raw rank within those rows, never scaled up to the pool. They differ only under sampling.
    per_golden = {
        c.key: {"rank": r, "rank_optimistic": o, "pool": c.total, "sampled": len(c.feats), "positives": len(c.golden_ids)}
        for c, (r, o) in entries
    }
    return {"per_golden": per_golden, "summaries": [c.to_json() for c in _per_card(entries, cv_split="full_train")]}


def run_folds(groups: list[GoldenGroup], *, trainer, k: int = DEFAULT_FOLDS) -> dict:
    """The full cross-validation → the ``cv`` metrics block.

    ``trainer`` is any object with ``fit(groups) -> model`` where the model satisfies
    :func:`group_ranks`' protocol. ONE instance serves every fold: the trainer is immutable and
    its ``fit`` is pure, so a fold's fit never depends on how many folds ran before it, keeping
    cross-run diffs meaningful. The caller passes the FOLD trainer, seeded from zeros rather than
    the incumbent, since the incumbent's weights were fit on every golden and would leak each
    held-out golden into its own holdout model. Guard: a fold is excluded (with a recorded reason)
    when the trainer declares its training slice unfittable (:func:`_unfittable`).

    Folds group by :attr:`GoldenGroup.shape` (:func:`assign_folds`), so every golden sharing a candidate
    pool is held out together — on any card. Splitting them is not a bias, it is a hole: the fold
    model would be scored on a pool it had already been given the answer to."""
    by_shape = assign_folds(groups, k)
    holdout: list[tuple[GoldenGroup, tuple[int, int]]] = []
    holdout_fold: dict[str, int] = {}
    train_acc: dict[str, tuple[GoldenGroup, list[int], list[int]]] = {}
    fold_medians: dict[str, dict] = {}
    excluded: dict[str, str] = {}

    for f in range(k):
        train = [c for c in groups if by_shape[c.shape] != f]
        hold = [c for c in groups if by_shape[c.shape] == f]
        if not hold:
            continue
        if reason := _unfittable(trainer, train, hold):
            excluded[str(f)] = reason
            continue
        model = trainer.fit(train)
        hold_entries = [(c, r) for c in hold if (r := group_ranks(c, model)) is not None]
        holdout.extend(hold_entries)
        holdout_fold.update({c.key: f for c, _ in hold_entries})
        for c in train:
            if (r := group_ranks(c, model)) is not None:
                acc = train_acc.setdefault(c.key, (c, [], []))
                acc[1].append(r[0])
                acc[2].append(r[1])
        fold_medians[str(f)] = {"median": _median([r for _, (r, _) in hold_entries]) if hold_entries else None, "n": len(hold_entries)}

    train_entries = [(c, (_median(pes), _median(opt))) for c, pes, opt in train_acc.values()]
    holdout_cards = _per_card(holdout, cv_split="holdout")
    train_cards = _per_card(train_entries, cv_split="train")
    # ``gap`` = holdout median − train median, per card: it separates overfitting (train good, holdout bad)
    # from a model class too weak for the problem (both bad, small gap). Read off the summaries, so the two
    # numbers it subtracts are the ones the file actually reports.
    hold_med = {c.axes["gpu"]: c.metrics["rank"]["median"] for c in holdout_cards}
    train_med = {c.axes["gpu"]: c.metrics["rank"]["median"] for c in train_cards}
    return {
        "summaries": [c.to_json() for c in holdout_cards + train_cards],
        "holdout_per_golden": {
            c.key: {"rank": r, "rank_optimistic": o, "positives": len(c.golden_ids), "fold": holdout_fold[c.key]} for c, (r, o) in holdout
        },
        "train_per_golden": {c.key: {"rank": r, "rank_optimistic": o, "positives": len(c.golden_ids)} for c, (r, o) in train_entries},
        "gap": {gpu: round(hold_med[gpu] - train_med[gpu], 1) for gpu in hold_med if gpu in train_med},
        "fold_detail": {"holdout_medians": fold_medians, "excluded": excluded},
    }


def build_metrics(header: dict, groups: list[GoldenGroup], skipped: list[tuple[str, str, str]], full_model, cv: dict[str, dict]) -> dict:
    """Assemble the run's complete metrics dict (JSON-ready, deterministic — no
    timestamps or host info; the caller serializes with sorted keys). ``skipped`` rows
    are ``(gpu, name, reason)`` for goldens that never became groups: enumeration
    failures count per card as ``unranked`` (never silently dropped), kinds the fitter
    doesn't group-build (attention / rms_norm / softmax) as ``out_of_scope``."""
    full = evaluate_full_train(groups, full_model)
    # Beside the summaries, not inside them: a golden that never became a group has no pool, no rank and nothing
    # to score, so it is a fact about the CORPUS rather than about a scored card. Folding it into a summary
    # would also give the fit a five-key summary where an eval report has four, which is exactly the schema
    # split this step exists to close. A card whose every golden was skipped still appears — as a key here,
    # with no summary — so a card going missing stays loud.
    counts = {c["axes"]["gpu"]: {"unranked": 0, "out_of_scope": 0} for c in full["summaries"]}
    for gpu, _name, reason in skipped:
        per_card = counts.setdefault(gpu, {"unranked": 0, "out_of_scope": 0})
        per_card["out_of_scope" if reason == OUT_OF_SCOPE else "unranked"] += 1
    full["skipped"] = dict(sorted(counts.items()))
    return {"header": header, "full_train": full, "cv": cv}
