"""``Sample`` — one measured-or-recorded ``(config, latency, identity)`` row,
the common currency over all three measurement-data sources.

A golden config, a tune-DB ``perf`` row, and a online-prior reservoir row are all
the same thing once normalized: a tunable-knob dict, a measured latency, a
structural identity, and (for golden) a reference latency. ``Sample`` is that
normal form. The split into ``knobs`` (tunable) / ``context`` (``H_*``) /
``s_features`` (``S_*``) is by key prefix and therefore lossless — :meth:`all_knobs`
re-merges them to the exact original dict, and :meth:`features` runs the single
featurizer (:func:`features.knob_features`) on that merge, so a ``Sample`` reproduces
the feature vector each source built inline today.

Featurization fidelity (the load-bearing invariant): a trained ``OnlinePrior``
regresses on the full ``S_*`` histogram stamped by
``992_stamp_structural_features``. DB / prior rows carry that histogram inline;
golden rows derive it by lowering their embedded frontend program and selecting
the target through provenance. Neither the histogram nor ``ShapeKey`` is part of
the stable golden format.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from emmy.compiler.pipeline.knob import CTX_PREFIX, STRUCT_PREFIX
from emmy.compiler.pipeline.search.data.shape import ShapeKey
from emmy.compiler.pipeline.search.features import knob_features

# The C identifier of a CUDA kernel, parsed from ``cuda_op.pretty`` — the grouping
# key for the per-knob regret analysis. Kept here so the DB-row adapter and the
# regret grouping share one source. Anchored on the ``__global__`` entry point
# (``__launch_bounds__`` sits between it and ``void``) — MMA/TMA kernel sources
# open with ``__device__`` helper preludes (``emmy_ldmatrix_x4``, ``mbarrier_init``),
# so a bare first-``void`` match would name the helper, collapsing distinct kernels
# into one leaderboard bucket and hiding them from ``--kernel`` filters.
KERNEL_NAME_RE = re.compile(r"__global__\s+(?:__launch_bounds__\([^)]*\)\s+)?void\s+(\w+)\s*\(")


def _split_by_prefix(knobs: dict) -> tuple[dict, dict, dict]:
    """Split a stamped knob dict into ``(tunable, context H_*, structural S_*)`` by
    key prefix. Disjoint prefixes → re-merging is lossless."""
    ctx = {k: v for k, v in knobs.items() if k.startswith(CTX_PREFIX)}
    s = {k: v for k, v in knobs.items() if k.startswith(STRUCT_PREFIX)}
    tunable = {k: v for k, v in knobs.items() if not k.startswith((CTX_PREFIX, STRUCT_PREFIX))}
    return tunable, ctx, s


@dataclass(frozen=True)
class Sample:
    """One ``(config, latency, identity)`` row, normalized across sources.

    ``knobs`` holds *only* tunable knobs (``S_*`` / ``H_*`` live in ``s_full`` /
    ``context``); ``pins`` holds the input knob regime for a golden replay and is
    empty for measurement rows from other sources. ``shape`` is the arithmetic
    identity; ``ref_us`` is the cuBLAS / torch reference (golden only, ``None``
    elsewhere); ``pretty`` / ``name`` carry the kernel C identifier for DB rows.
    ``source`` ∈ ``{"golden", "db", "prior"}`` marks provenance for the
    orthogonality fail-fast (``dataset_args.require_source``)."""

    knobs: dict
    latency_us: float
    shape: ShapeKey | None = None
    name: str | None = None
    dtype: str | None = None
    ref_us: float | None = None
    pins: dict = field(default_factory=dict)
    context: dict = field(default_factory=dict)
    pretty: str | None = None
    source: str = "db"
    s_full: dict | None = None  # full compiled/derived S_* histogram when known
    error: str | None = None  # bench_fail failure text (db rows only; None on ok rows)
    # Optional exact work count. The intensity-floor gate reads THIS, not a ShapeKey reconstruction: the join key
    # excludes symbolic axes on the matmul side but includes them on the reduce-tier side, so no
    # one hint-multiplier formula over it can be right for both.
    flops: float | None = None

    def s_features(self) -> dict[str, float]:
        """The ``S_*`` features this sample featurizes on: the full stamped histogram
        when known, else the cheap arithmetic extents, else nothing."""
        if self.s_full is not None:
            return self.s_full
        return self.shape.s_features_arith() if self.shape is not None else {}

    def all_knobs(self) -> dict:
        """The original stamped dict — ``context ∪ s_features ∪ knobs``. For a DB
        row this re-merges to exactly the recorded ``perf.knobs``; the per-knob
        regret analysis iterates this so its output is unchanged."""
        return {**self.context, **self.s_features(), **self.knobs}

    def features(self) -> dict[str, float]:
        """The flat numeric feature vector the priors regress on — the single
        featurizer over the merged dict. Merge order ``context, s_*, knobs`` matches
        the inline construction the eval / prior code used (knobs win on collision,
        though the prefixes are disjoint)."""
        return knob_features(self.all_knobs())

    @classmethod
    def from_golden(cls, cfg, *, compile_s_feats: bool = False) -> Sample:
        """A program-backed golden record as a normalized measurement sample.

        ``compile_s_feats`` remains an accepted no-op for callers that used to
        request snippet compilation. Structural features are lazily derived from the
        embedded program and provenance target.
        """
        from emmy.compiler.context import Context  # noqa: PLC0415

        tunable, _ctx, _s = _split_by_prefix(cfg.knobs)
        return cls(
            knobs=tunable,
            latency_us=cfg.emmy_us,
            shape=cfg.shape_key,
            name=cfg.name,
            dtype=cfg.dtype,
            ref_us=cfg.reference_us,
            pins=cfg.pin_map,
            # gpu_name pins the device-physical features (H_sm_count / smem / …) to
            # the golden's OWN card's memorized specs, not the live device's — so a
            # PRO 6000 golden ranked on a 5090 (both cc 12.0) gets 188 SMs, not 170.
            context=Context.from_target(cfg.compute_cap, gpu_name=cfg.gpu_name).features(),
            source="golden",
            s_full=dict(cfg.structural_features),
        )

    @classmethod
    def from_perf_sample(cls, ps) -> Sample:
        """A tune-DB ``perf ⋈ cuda_op`` row (:class:`db.PerfSample`) as a ``Sample``.
        Splits the recorded knob dict by prefix; the kernel C identifier (for
        per-knob regret grouping) is parsed from ``cuda_op.pretty``."""
        tunable, ctx, s = _split_by_prefix(ps.knobs)
        m = KERNEL_NAME_RE.search(ps.pretty or "")
        return cls(
            knobs=tunable,
            latency_us=ps.latency_us,
            name=m.group(1) if m else None,
            context=ctx,
            pretty=ps.pretty,
            source="db",
            s_full=s,
            error=ps.error,
        )

    @classmethod
    def from_prior_row(cls, knobs: dict, latency_us: float) -> Sample:
        """A online-prior reservoir row ``(stamped_knobs, latency)`` as a ``Sample``.
        The reservoir dicts already carry ``S_*`` / ``H_*`` inline (stamped by the
        live pipeline), so the split + re-merge is lossless for grouping / scoring."""
        tunable, ctx, s = _split_by_prefix(knobs)
        return cls(knobs=tunable, latency_us=latency_us, context=ctx, source="prior", s_full=s)
