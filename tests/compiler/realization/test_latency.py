"""Stage 5 — is a realized schedule still as fast as it was, and how does it compare to torch.

Its own lane, `perf`-marked and opt-in, for two reasons the tree already establishes. `make test`
compiles at `-Xcicc -O1` — the correctness lane, which is not a measurement lane — so a latency
assertion there would measure the wrong regime entirely. And `tests/perf/` is deliberately
non-asserting today; turning that stance around belongs in its own lane rather than as a side
effect of a correctness suite.

**A performance regression is reported, never enforced.** A slower case does not fail a run: it
produces a finding that the timing-refresh workflow turns into a labelled pull request a human
accepts or declines. A nightly that goes permanently red because one legitimate correctness fix
cost latency is a nightly nobody reads.
"""

from __future__ import annotations

import pytest

from tests.compiler.helpers import device_compute_capability, requires_cuda
from tests.compiler.realization import helpers

pytestmark = [pytest.mark.perf, requires_cuda]


def _closed_cases():
    """Only closed cases. An open case's schedule never runs, so demanding a latency for it would
    be the same false attribution the corpus rejects everywhere else."""
    cases = []
    for path in helpers.case_files():
        case = helpers.load_case(path)
        if case.xfail_stage is None:
            cases.append(pytest.param(case, id=case.id))
    return cases


@pytest.mark.parametrize("case", _closed_cases())
def test_latency(case, record_property):
    """Compare the best of three runs against this card's recorded latency."""
    if device_compute_capability() != case.compute_cap:
        pytest.skip(f"case declares sm_{''.join(map(str, case.compute_cap))}")
    hardware_id = helpers.live_hardware_id()
    recorded = helpers.recorded_latency(case, hardware_id)
    if recorded is None:
        # Coverage grows by being asked once, on a card that can answer. Reported at session end
        # rather than here, so one run names every gap and the command that closes them.
        record_property("missing_latency", hardware_id)
        pytest.skip(f"no recorded latency for {hardware_id}")

    stored = float(recorded["emmy_us"])
    samples, tcompile_us = helpers.measure(case, within=stored * (1 + helpers.LATENCY_BAND))
    best = min(samples)
    if best > stored * (1 + helpers.LATENCY_BAND):
        # A finding, not a failure — see the module docstring.
        print(
            f"\nSLOWER {case.id} on {hardware_id}: {best:.2f} us against a recorded {stored:.2f} us "
            f"(+{100 * (best / stored - 1):.1f}%, band {100 * helpers.LATENCY_BAND:.0f}%); "
            f"samples {', '.join(f'{sample:.2f}' for sample in samples)}. "
            f"{f'torch.compile {tcompile_us:.2f} us. ' if tcompile_us else ''}"
            f"Accept the new baseline with `emmy run --golden-file {case.path} --golden {case.record.name} "
            "--bench --bench-backends eager,tcompile,emmy --record`."
        )
