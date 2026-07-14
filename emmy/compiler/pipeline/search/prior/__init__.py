"""Online tuning prior: a :class:`Prior` base (bounded reservoir dataset, batched
refit, end-of-run sanity stats) and the concrete :class:`OnlinePrior`."""

from __future__ import annotations

from emmy.compiler.pipeline.search.prior.base import Prior
from emmy.compiler.pipeline.search.prior.fallback import FallbackPrior, load_prior
from emmy.compiler.pipeline.search.prior.offline import OfflinePrior
from emmy.compiler.pipeline.search.prior.online import OnlinePrior


def prior_from_json(obj: dict) -> Prior:
    """Reconstruct a fitted prior from a persisted :meth:`Prior.to_json` dict
    (used to load a tuned prior into a later ``compile`` / ``run``)."""
    return OnlinePrior.from_json(obj)


__all__ = ["OfflinePrior", "OnlinePrior", "FallbackPrior", "Prior", "load_prior", "prior_from_json"]
