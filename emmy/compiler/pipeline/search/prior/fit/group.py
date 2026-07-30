"""``Group`` — the fit pipeline's dataset representation: one candidate pool plus its labels.

A group is one shape's featurized candidate pool on one card, with whatever supervision exists for it. Today
that supervision is a single pinned verified-optimum row (``pinned_idx`` — the golden's index in the pool);
the freeze-trained cells add per-row measured labels when they land, so builders other than the golden case
builder can populate groups from measurement sources. Groups are built by plain functions (the golden builder
lives in ``emmy/commands/fit.py`` — case building needs the snippet tracer, which ``pipeline/`` must not
import) and consumed by trainers and the CV harness through this one shape; there is no iterator/batching
layer — the whole dataset is a small in-memory list.

``key`` is ``"<gpu>/<name>"``, disambiguated by the builder when one name records several parity entries
(``#2``, ``#3``, … in dataset order). ``tier`` is the fit's case tier (``thread`` / ``warp`` / ``dyn`` /
``reduce`` / ``pointwise``); ``dyn`` groups score with the dynamic weight set, everything else with the
static one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# A golden name's trailing size/dtype/variant segments (``512``, ``fp16``, ``dynM``, ``h4096``, ``s2048``,
# ``n16384``, ``k8192``, ``hd128``) — stripped by :func:`op_family`.
_VARIANT_SEG = re.compile(r"fp16|dynM|(?:hd|[hsnk])?\d+")


@dataclass(frozen=True)
class Group:
    """One golden's featurized candidate pool, plus the identity the fold axes and metrics keys need."""

    key: str
    name: str
    tier: str
    gpu: str
    pinned_idx: int
    feats: list[dict[str, float]] = field(repr=False)

    @property
    def fit_case(self) -> tuple[str, str, int, list[dict[str, float]]]:
        """The 4-tuple shape :func:`linear.fit_weights` consumes."""
        return (self.name, self.tier, self.pinned_idx, self.feats)

    @property
    def family(self) -> str:
        return op_family(self.name)


def op_family(name: str) -> str:
    """The golden's op family — its dot-name with trailing size/dtype/variant segments stripped:
    ``matmul.square.512.fp16`` → ``matmul.square``, ``gemma4_12b.q_proj.s2048`` → ``gemma4_12b.q_proj``,
    ``reduce.k2048.dynM`` → ``reduce``. The leave-one-family-out axis holds out every size/dtype/dynamic
    variant of one op shape together, so the holdout fold measures generalization to an unseen shape family,
    not interpolation between its own sizes. (Model-prefixed names keep the model tag: ``gemma4_12b.mlp_down``
    and ``matmul.mlp_down`` are distinct families — different shape geometry.)"""
    segs = name.split(".")
    while len(segs) > 1 and _VARIANT_SEG.fullmatch(segs[-1]):
        segs.pop()
    return ".".join(segs)
