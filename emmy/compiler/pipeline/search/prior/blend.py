"""How the online and offline priors interact — as a named, swappable strategy rather than a hardcoded expression.

:class:`~emmy.compiler.pipeline.search.prior.fallback.FallbackPrior` holds both halves and owns the calibration
gate; a :class:`Blend` decides what to *do* with them. Two questions, and they are genuinely different:

- **Who owns the deploy ranking** (:meth:`Blend.deploy_half`) — the greedy argmin, ``pick``, and the featurized
  diagnostics surface. This returns one whole prior rather than a mixed score vector, because ``pick``'s score
  feeds ``greedy._pick_structural`` as an absolute µs cost estimate: mixing an ordinal proxy into that scale is
  precisely what went wrong on the selection side. Blending here would mean reopening that seam deliberately.
- **What PUCT explores by** (:meth:`Blend.policy`) — where the two halves genuinely combine, safely, because
  ``Prior.policy`` has already normalized both to the same sibling-relative scale.

Why a strategy at all: the blend has never been measurable. The shipped one multiplied predicted µs by a clamped
``offline**W``, and measurement found 255 of 261 goldens pinned to the identical clamped constant — a documented
mechanism contributing nothing (HISTORY.md: "The inert offline tilt"). Naming the alternatives makes the
comparison an ordinary A/B: ``EMMY_PRIOR_BLEND=gate`` measures what the tilt is worth, and ``offline`` / ``online``
isolate one half so a change to *that* half can be measured without the other masking it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from emmy.compiler.pipeline.search.prior.base import Prior


class Blend(ABC):
    """One way the two halves interact. Stateless and immutable — a strategy holds no per-run state, so one
    instance serves every prior and swapping strategies swaps nothing else."""

    name: str

    @abstractmethod
    def deploy_half(self, online: Prior, offline: Prior, *, trusted: bool) -> Prior:
        """Which half owns the deploy ranking — ``mean_score(s)``, the featurized scoring surface,
        ``explain_features``, and the tie-break behind ``pick``."""

    def policy(self, online: Prior, offline: Prior, rows: list[dict], *, trusted: bool) -> list[float]:
        """PUCT's ``P`` over one fork's sibling set (higher = better, best sibling ``1.0``). The default is no
        interaction at all — the half that owns deploys also owns selection — so only a strategy that genuinely
        combines the two has anything to override."""
        return self.deploy_half(online, offline, trusted=trusted).policy(rows)


class GateBlend(Blend):
    """No interaction: whichever half is live answers both questions. The online model owns everything once it is
    ``trustworthy``, the offline one while it is cold or quarantined.

    This is the honest null hypothesis for the tilt — run it against :class:`TiltBlend` on the same shape and the
    difference IS the tilt's contribution, which has never been measured."""

    name = "gate"

    def deploy_half(self, online: Prior, offline: Prior, *, trusted: bool) -> Prior:
        return online if trusted else offline


class TiltBlend(GateBlend):
    """The default: :class:`GateBlend` at deploy, and at selection the online policy *tilted* by the offline
    one — ``p ∝ p_online · p_offline**W``, renormalized.

    The point of the tilt is that PUCT should still explore a region the cold heuristic rates well but the
    data-poor online model buries, having never measured it (the golden-sweep fp16 finding). ``W`` is
    ``config.offline_tilt`` (``EMMY_OFFLINE_TILT``, default 0.3); ``W=0`` recovers pure-online selection and
    ``W=1`` gives the two halves equal say.

    Both factors are sibling-normalized into ``(0, 1]`` before they meet, which is the whole fix: the retired
    spelling multiplied *predicted µs* by ``offline**W`` and had to clamp the offline factor to ``e**±8`` to keep
    it from swamping the µs scale — a clamp nearly every candidate saturated, leaving the term a constant."""

    name = "tilt"

    def policy(self, online: Prior, offline: Prior, rows: list[dict], *, trusted: bool) -> list[float]:
        if not trusted:
            return offline.policy(rows)
        from emmy import config  # noqa: PLC0415 — module-level would make config import order load-bearing

        w = config.offline_tilt()
        p_on = online.policy(rows)
        if w == 0.0:
            return p_on
        p_off = offline.policy(rows)
        # Renormalize: two vectors that each peak at 1.0 have a product that need not (the halves rarely agree on
        # which sibling is best), and PUCT scales the exploration term against ``Q``. An all-zero product — both
        # halves underflowing on disjoint siblings — falls back to uniform rather than dividing by zero.
        raw = [a * b**w for a, b in zip(p_on, p_off, strict=True)]
        top = max(raw, default=0.0)
        return [r / top for r in raw] if top > 0 else [1.0] * len(raw)


class SingleBlend(Blend):
    """One half answers everything, calibration gate ignored — the A/B arms.

    ``offline`` is how a change to the *cold* model class (its features, its fit, its model class) is measured on
    real hardware without the online half quietly taking over the moment it looks trained. ``online`` is the
    mirror. Neither is a deploy default: they deliberately discard the guard that keeps a mis-calibrated online
    model from owning decisions."""

    def __init__(self, name: str, *, use_online: bool) -> None:
        self.name = name
        self._use_online = use_online

    def deploy_half(self, online: Prior, offline: Prior, *, trusted: bool) -> Prior:  # noqa: ARG002 — the point is to ignore it
        return online if self._use_online else offline


BLENDS = {b.name: b for b in (TiltBlend(), GateBlend(), SingleBlend("online", use_online=True), SingleBlend("offline", use_online=False))}


def load_blend(name: str | None = None) -> Blend:
    """The named strategy, or the configured one (``EMMY_PRIOR_BLEND``) when ``name`` is None. An unknown name is
    a hard error naming the alternatives — a silently-defaulted A/B arm would report the default's numbers under
    the arm's label."""
    from emmy import config  # noqa: PLC0415

    name = name or config.prior_blend()
    if name not in BLENDS:
        raise ValueError(f"unknown prior blend {name!r} — expected one of {sorted(BLENDS)}")
    return BLENDS[name]
