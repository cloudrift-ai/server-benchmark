"""``Group`` — the fit pipeline's dataset representation: one candidate pool plus its labels.

A group is one shape's featurized candidate pool on one card, with whatever supervision exists for it. Today
that supervision is a single pinned verified-optimum row (``pinned_idx`` — the golden's index in the pool);
the freeze-trained cells add per-row measured labels when they land, so builders other than the golden case
builder can populate groups from measurement sources. Groups are built by plain functions (the golden builder
lives in ``emmy/commands/fit.py`` — case building needs the snippet tracer, which ``pipeline/`` must not
import) and consumed by trainers and the CV harness through this one shape; there is no iterator/batching
layer — the whole dataset is a small in-memory list.

Rows are ndarray-backed, not dict-backed: ``feats`` is one float64 matrix (rows × ``feat_names``), packed
once by :meth:`Group.from_dicts` from the builder's transient per-row feature dicts. A per-row dict of ~63
floats costs ~4 KB; the full golden dataset is ~2.5 M rows, and the dict spelling (~10 GB) OOM-killed whole
fit runs — the matrix spelling is ~20× smaller and :meth:`matrix` reproduces ``feats.get(k, 0.0)`` semantics
bit-identically (an absent feature is a zero column).

``key`` is ``"<gpu>/<name>"``, disambiguated by the builder when one name records several parity entries
(``#2``, ``#3``, … in dataset order). ``tier`` is the fit's case tier (``thread`` / ``warp`` / ``dyn`` /
``reduce`` / ``pointwise``); ``dyn`` groups score with the dynamic weight set, everything else with the
static one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

# A golden name's trailing size/dtype/variant segments (``512``, ``fp16``, ``dynM``, ``h4096``, ``s2048``,
# ``n16384``, ``k8192``, ``hd128``) — stripped by :func:`op_family`.
_VARIANT_SEG = re.compile(r"fp16|dynM|(?:hd|[hsnk])?\d+")


def feature_matrix(feats: list[dict[str, float]], names: list[str]) -> np.ndarray:
    """Feature-dict rows as a dense float64 matrix over ``names`` — absent key = 0.0."""
    return np.array([[f.get(n, 0.0) for n in names] for f in feats], dtype=float)


@dataclass(frozen=True)
class Group:
    """One golden's featurized candidate pool, plus the identity the fold axes and metrics keys need."""

    key: str
    name: str
    tier: str
    gpu: str
    pinned_idx: int
    feat_names: tuple[str, ...]
    feats: np.ndarray = field(repr=False)

    @classmethod
    def from_dicts(cls, key: str, name: str, tier: str, gpu: str, pinned_idx: int, feats: list[dict[str, float]]) -> Group:
        """Pack per-row feature dicts into the matrix spelling: ``feat_names`` is the sorted
        union of the pool's keys, the matrix a column per name (absent key = 0.0). Callers
        drop the dicts after this — the matrix is the stored representation."""
        names = tuple(sorted({k for f in feats for k in f}))
        return cls(key, name, tier, gpu, pinned_idx, names, feature_matrix(feats, list(names)))

    def matrix(self, names: list[str]) -> np.ndarray:
        """The pool projected onto ``names`` — column ``j`` is the stored ``names[j]`` column,
        or zeros when the pool never saw that feature: exactly ``feats.get(k, 0.0)`` per row,
        so scoring/fitting against any name vocabulary matches the dict spelling bit for bit."""
        idx = {n: j for j, n in enumerate(self.feat_names)}
        out = np.zeros((len(self.feats), len(names)))
        for j, n in enumerate(names):
            if n in idx:
                out[:, j] = self.feats[:, idx[n]]
        return out

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
