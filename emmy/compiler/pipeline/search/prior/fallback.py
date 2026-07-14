"""``FallbackPrior`` — composes the online + offline priors into the single
ranking surface the search consumes.

``mean_score`` / ``mean_scores`` / ``pick`` (the deploy + eval + diagnostics
surface) use the :class:`OnlinePrior` once it's ``trustworthy`` —
``fitted`` AND passing the reservoir calibration gate (``Prior.trustworthy``;
a fitted-but-mis-calibrated model is quarantined to offline instead of owning
deploys) — and fall
back to the :class:`OfflinePrior` (cold-start heuristic) otherwise — so the
policies always get a usable ranking and no longer special-case "cold → emission
order". :meth:`score` (the MCTS *selection* signal — see :mod:`policy.mcts`) is
the one surface that BLENDS the two even when trusted, so PUCT explores the region
the heuristic prices well but the data-poor online model buries (the golden-sweep
fp16 finding). Everything else (training: ``add_rows`` / ``maybe_refit`` /
``checkpoint`` / ``fit`` / ``to_json``, and inspection: ``_dataset`` /
``trajectory`` / ``summary`` for diagnostics) delegates to the online half, so
``tune`` trains and checkpoints CatBoost exactly as before and ``fitted`` reflects
whether a *trained* model exists.

The two priors are on **different scales**: CatBoost regresses ``log(latency µs)``
so its ``score`` is calibrated µs, whereas the offline prior is fit by
learning-to-rank (``scripts/golden_knob_heuristics.py``) so its ``score`` —
``exp(-0.1·quality)`` — is an *ordinal* proxy with arbitrary magnitude (only its
order is meaningful; its neutral "no opinion" value is exactly ``1.0``). So
:meth:`score` keeps the online µs as the scale and folds the offline in as a
dimensionless **multiplier** ``offline**W`` centered at that neutral 1.0 — the
online half owns the per-shape µs anchor, the offline half contributes only its
ranking. ``W`` (``config.offline_tilt``) sizes the nudge: small enough to
perturb the online order, not replace it.
"""

from __future__ import annotations

from emmy import config
from emmy.compiler.pipeline.search.prior.base import Prior
from emmy.compiler.pipeline.search.prior.offline import OfflinePrior
from emmy.compiler.pipeline.search.prior.online import OnlinePrior


class FallbackPrior(Prior):
    """Online prior with an offline cold-start fallback. Not a dataset owner —
    its training / inspection surface is the ``online`` prior's. ``mean_score`` /
    ``mean_scores`` fall back to offline while cold or quarantined
    (``trustworthy`` — the calibration gate); ``score`` (the MCTS
    selection signal) blends the offline multiplier in even when trusted."""

    def __init__(self, online: Prior, offline: Prior | None = None) -> None:
        # Deliberately NOT calling super().__init__() — this prior holds no
        # dataset of its own; every stateful attribute delegates to ``online``
        # (see __getattr__), so there's no second reservoir to diverge.
        self.online = online
        self.offline = offline if offline is not None else OfflinePrior()

    @property
    def fitted(self) -> bool:
        # Reflects the ONLINE model (so diagnostics / `eval online` report whether
        # real tuning data exists). The policies no longer gate on this — they
        # always call score()/mean_score(), which fall back to offline when cold.
        return self.online.fitted

    @property
    def trustworthy(self) -> bool:
        # The promotion gate: fitted AND not demonstrably mis-calibrated
        # (``Prior.trustworthy`` — reservoir Spearman vs ``CALIBRATION_MIN``).
        # ``fitted`` alone let a garbage model own deploys silently (the RTX 5090
        # sweep-7 finding 3); a quarantined model keeps training and checkpointing
        # while decisions stay offline.
        return self.online.trustworthy

    def score(self, knobs: dict) -> float:
        # MCTS-selection signal ONLY (deploy/eval go through mean_score/pick).
        # Cold or quarantined: the offline prior IS the ranking. Trusted: keep the
        # online µs as the scale and tilt it by the offline's dimensionless ranking
        # multiplier (``offline**W``, neutral 1.0) so PUCT still explores a region the
        # heuristic prices well but the online model — having never measured it —
        # buries (the fp16 small-BK warp tiles at large squares). ``W=0`` recovers
        # pure-online selection. Scale-correct because the offline factor is
        # centered at its no-opinion 1.0, so a config the heuristic has no view on
        # leaves the online prediction untouched.
        if not self.trustworthy:
            return self.offline.score(knobs)
        w = config.offline_tilt()
        online = self.online.score(knobs)
        if w == 0.0 or online <= 0.0:
            return online
        return online * self.offline.score(knobs) ** w

    def mean_score(self, knobs: dict) -> float:
        return self.online.mean_score(knobs) if self.trustworthy else self.offline.mean_score(knobs)

    def mean_scores(self, knobs_list: list[dict]) -> list[float]:
        return self.online.mean_scores(knobs_list) if self.trustworthy else self.offline.mean_scores(knobs_list)

    # Features-seam surface (attribution / ablation) — routed through the same
    # trustworthy gate as mean_score/mean_scores, so the diagnostics decompose the
    # model that actually owns decisions.
    def mean_score_features(self, feats: dict) -> float:
        return self.online.mean_score_features(feats) if self.trustworthy else self.offline.mean_score_features(feats)

    def mean_scores_features(self, feats_list: list[dict]) -> list[float]:
        return self.online.mean_scores_features(feats_list) if self.trustworthy else self.offline.mean_scores_features(feats_list)

    def explain_features(self, feats: dict) -> dict[str, float] | None:
        return self.online.explain_features(feats) if self.trustworthy else self.offline.explain_features(feats)

    def pick(self, rows: list[dict]) -> tuple[int, float]:
        # Measured -O3 evidence lives in the ONLINE half's reservoir (the
        # offline prior has no dataset), and applies even while the model is
        # cold — a freshly-seeded reservoir below ``min_rows`` still holds real
        # measurements worth deploying. The score fallback below then ranks the
        # evidence-less case through whichever half is live.
        ev = self.online.evidence_pick(rows)
        if ev is not None:
            return ev
        scores = self.mean_scores(rows)
        best_i = min(range(len(scores)), key=scores.__getitem__)
        return best_i, scores[best_i]

    # --- training + inspection: delegate to the online half ------------------
    def fit(self) -> None:
        self.online.fit()

    def add_rows(self, rows) -> None:
        self.online.add_rows(rows)

    def maybe_refit(self, *, force: bool = False) -> bool:
        return self.online.maybe_refit(force=force)

    def checkpoint(self) -> None:
        self.online.checkpoint()

    def to_json(self) -> dict | None:
        return self.online.to_json()

    def record_bench(self, knobs: dict, median: float, status: str) -> None:
        self.online.record_bench(knobs, median, status)

    def summary(self, label: str) -> str:
        return self.online.summary(label)

    def __getattr__(self, name: str):
        # Read-through for anything not defined here (``_dataset`` / ``trajectory``
        # / ``_path`` / counters that diagnostics + the batch refit (`tune --refit`) inspect).
        # ``__getattr__`` only fires for genuinely-missing attributes, so the
        # explicit overrides above always win.
        return getattr(self.online, name)


def load_prior(*, seed: int = 0, path=None) -> FallbackPrior:
    """The one global prior the search loads: the ``OnlinePrior``
    (warm-started from its checkpoint if present) wrapped with an
    ``OfflinePrior`` cold-start fallback."""
    return FallbackPrior(OnlinePrior.load(seed=seed, path=path), OfflinePrior())
