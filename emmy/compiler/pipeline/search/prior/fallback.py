"""``FallbackPrior`` — composes the online + offline priors into the single
ranking surface the search consumes.

The composition itself — who answers a deploy ranking, and how the two combine for
PUCT — is a swappable :class:`~emmy.compiler.pipeline.search.prior.blend.Blend`
strategy, so the interaction can be A/B'd rather than assumed. The class keeps its
name because "fall back to the cold half" is what the default strategies do:
``mean_score`` / ``mean_scores`` / ``pick`` (the deploy + eval + diagnostics
surface) use the :class:`OnlinePrior` once it's ``trustworthy`` — ``fitted`` AND
passing the reservoir calibration gate (``Prior.trustworthy``; a
fitted-but-mis-calibrated model is quarantined to offline instead of owning
deploys) — and fall back to the :class:`OfflinePrior` (cold-start heuristic)
otherwise, so the policies always get a usable ranking and no longer special-case
"cold → emission order". :meth:`policy` (the MCTS *selection* signal — see
:mod:`policy.mcts`) is the surface where a strategy may genuinely combine the two,
as the default ``tilt`` does so PUCT explores the region the heuristic prices well
but the data-poor online model buries (the golden-sweep fp16 finding).

Everything else (training: ``add_rows`` / ``maybe_refit`` / ``checkpoint`` / ``fit``
/ ``to_json``, and inspection: ``_dataset`` / ``trajectory`` / ``summary`` for
diagnostics) delegates to the online half, so ``tune`` trains and checkpoints
CatBoost exactly as before and ``fitted`` reflects whether a *trained* model exists.

The two priors are on **different scales**: the online model regresses
``log(latency µs)`` so its prediction is calibrated µs, whereas the offline prior is
fit by learning-to-rank (``emmy fit``) so its score is an *ordinal* proxy with
arbitrary magnitude — only its order is meaningful. That is why the halves meet only
in ``policy``, where ``Prior.policy`` has already reduced both to the same
sibling-relative scale, and why the deploy surface hands the whole question to ONE
half instead of mixing (``Blend.deploy_half``).
"""

from __future__ import annotations

from emmy.compiler.pipeline.search.prior.base import Prior
from emmy.compiler.pipeline.search.prior.blend import Blend, load_blend
from emmy.compiler.pipeline.search.prior.offline import OfflinePrior
from emmy.compiler.pipeline.search.prior.online import OnlinePrior


class FallbackPrior(Prior):
    """Online prior with an offline cold-start fallback. Not a dataset owner —
    its training / inspection surface is the ``online`` prior's. Which half answers
    a deploy ranking, and how the two combine for PUCT, is the ``blend``
    strategy's call (default ``tilt``; ``EMMY_PRIOR_BLEND`` swaps it)."""

    def __init__(self, online: Prior, offline: Prior | None = None, *, blend: Blend | None = None) -> None:
        # Deliberately NOT calling super().__init__() — this prior holds no
        # dataset of its own; every stateful attribute delegates to ``online``
        # (see __getattr__), so there's no second reservoir to diverge.
        self.online = online
        self.offline = offline if offline is not None else OfflinePrior()
        self.blend = blend if blend is not None else load_blend()

    @property
    def _deploy(self) -> Prior:
        """The half that owns the deploy ranking right now — ONE reading of the
        strategy, shared by every scoring surface below so they cannot disagree
        about which model is answering."""
        return self.blend.deploy_half(self.online, self.offline, trusted=self.trustworthy)

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

    def policy(self, knobs_list: list[dict]) -> list[float]:
        # MCTS-selection signal ONLY (deploy/eval go through mean_score/pick), and the
        # one surface where a strategy may combine both halves — safely, because
        # ``Prior.policy`` normalizes each within the sibling set first.
        return self.blend.policy(self.online, self.offline, knobs_list, trusted=self.trustworthy)

    # The deploy surface: ONE half answers, and every entry point below reads that
    # choice from the same place, so the diagnostics always decompose the model that
    # actually owns decisions.
    def mean_score(self, knobs: dict) -> float:
        return self._deploy.mean_score(knobs)

    def mean_scores(self, knobs_list: list[dict]) -> list[float]:
        return self._deploy.mean_scores(knobs_list)

    def mean_score_features(self, feats: dict) -> float:
        return self._deploy.mean_score_features(feats)

    def mean_scores_features(self, feats_list: list[dict]) -> list[float]:
        return self._deploy.mean_scores_features(feats_list)

    def explain_features(self, feats: dict) -> dict[str, float] | None:
        return self._deploy.explain_features(feats)

    @property
    def masking_exact(self) -> bool:
        return self._deploy.masking_exact

    def pick(self, rows: list[dict]) -> tuple[int, float]:
        # Measured -O3 evidence lives in the ONLINE half's reservoir (the
        # offline prior has no dataset), and applies even while the model is
        # cold — a freshly-seeded reservoir below ``min_rows`` still holds real
        # measurements worth deploying. The score fallback below then ranks the
        # evidence-less case through whichever half is live.
        ev = self.online.evidence_pick(rows)
        if ev is not None:
            return ev
        from emmy.compiler.pipeline.knob import canonical_row_key  # noqa: PLC0415

        # Score ties break by candidate content (``canonical_row_key``), never by
        # enumeration order — see ``Prior.pick``.
        scores = self.mean_scores(rows)
        best_i = min(range(len(scores)), key=lambda i: (scores[i], canonical_row_key(rows[i])))
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


def load_prior(*, seed: int = 0, path=None, blend: str | None = None) -> FallbackPrior:
    """The one global prior the search loads: the ``OnlinePrior``
    (warm-started from its checkpoint if present) wrapped with an
    ``OfflinePrior`` cold-start fallback, composed by the named ``blend``
    strategy (default: ``EMMY_PRIOR_BLEND``, else ``tilt``)."""
    return FallbackPrior(OnlinePrior.load(seed=seed, path=path), OfflinePrior(), blend=load_blend(blend))
