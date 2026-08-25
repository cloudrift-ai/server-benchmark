"""The fit's rank table — the one rendering both trainers log and write into artifact provenance.

Apart from :mod:`~..metrics` because that module is defined as string-free, and apart from the report cells
(``prior/report.py``) on purpose. This is trainer TELEMETRY, not a report: it is printed per fit round while a
trainer converges, and carries ``mean_log2`` — the fit's own objective, which no report cell wants — plus a
``k=5`` rung a report has no use for and a converging trainer does. The JSON twin that used to live in
``fit/cv.py`` is gone: a fit's metrics file now carries report cells, built by ``report.rank_metrics``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def topk_table(ranks: Sequence[int], ks: Sequence[int] = (1, 5, 10, 25, 50, 100)) -> str:
    """``topN=hits/total`` per cutoff, then the median and the mean log2 rank — one line.

    ``mean_log2`` is over ``rank + 1``, so a top-1 rank contributes 0 and the number is the fit's own
    objective rather than a raw mean that one 300k-rank pool would dominate. The median is the upper one
    at even ``n``, which is what this line has always reported."""
    n = len(ranks)
    parts = [f"top{k}={sum(r < k for r in ranks)}/{n}" for k in ks]
    med = sorted(ranks)[n // 2]
    return "  ".join(parts) + f"   median={med}  mean_log2={np.mean([math.log2(r + 1) for r in ranks]):.2f}"
