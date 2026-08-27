"""The perf lane: every realization-corpus case this card can run, measured against torch.

One bench per case, two outputs. The row joins the session-end table — emmy beside eager and
`torch.compile`, sorted worst-first — and the same measurement is compared against the case's
stored latency for this card, so a regression is caught without paying for a second sweep.

**A regression is reported, never enforced.** A slower case does not fail the run: it prints a
finding that the timing-refresh workflow turns into a labelled pull request a human accepts or
declines. A lane that goes red because one legitimate correctness fix cost latency is a lane
nobody reads. What DOES fail here is a case that cannot be benched at all, because that is a
broken measurement rather than a slow kernel.

The case list is the corpus itself. It used to be a hand-curated table of twelve
Qwen3-Embedding kernels, which drifted — it still claimed "Emmy currently emits FP32 only" long
after that stopped being true. Those shapes now live in `cases/qwen3emb/`, where the program is
stored rather than described, so the same drift cannot recur.
"""

from __future__ import annotations

import pytest

from tests.compiler.helpers import device_compute_capability, requires_cuda
from tests.compiler.realization import helpers

pytestmark = [pytest.mark.perf, requires_cuda]


def _cases():
    """Every closed case, ordered so a family reads together in the table.

    Open cases are excluded: their schedule never realizes, so there is nothing to time and
    demanding a number for one would be the same false attribution the corpus rejects elsewhere.
    """
    cases = [helpers.load_case(path) for path in helpers.case_files()]
    return [pytest.param(case, id=case.id) for case in cases if case.xfail_stage is None]


@pytest.mark.parametrize("case", _cases())
def test_corpus_perf(case, bench_pair):
    if device_compute_capability() != case.compute_cap:
        pytest.skip(f"case declares sm_{''.join(map(str, case.compute_cap))}")
    bench_pair(case)
