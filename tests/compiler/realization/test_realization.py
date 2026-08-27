"""One parametrized walker over ``cases/`` — the realization corpus.

A case is a checked-in minimized reproducer of one failure class: a schedule that should be
realizable is not. Its expectation is its filename (see ``ARCHITECTURE.md``); there is no manifest.

Every case is asked four questions in order, one test node each, so an ``_xfail_<stage>`` suffix
lands on exactly the stage it names rather than on a walker that could have failed anywhere.

``offered`` and ``realized`` are GPU-free and run at the case's DECLARED capability, so an sm_70
lockout is exercised on any box. ``built`` and ``correct`` need the card itself and run only when
the live capability equals the declared one — a pinned schedule is a claim about one capability,
never about a merely newer card.
"""

from __future__ import annotations

import pytest

from tests.compiler.helpers import device_compute_capability, requires_cuda
from tests.compiler.realization import helpers
from tests.compiler.realization.helpers import STAGES, CaseError

#: ``built`` hands its compiled graph to ``correct`` so one nvcc invocation serves both nodes when
#: they run on the same worker — which they do, since the root conftest routes CUDA onto one chain.
_COMPILED: dict[str, object] = {}


def _parameters():
    """One parameter per (case, stage), carrying the case's expectation as a marker.

    A case file that cannot be loaded becomes a single failing parameter rather than no parameters
    at all: silently collecting nothing is how a corpus stops asserting.
    """
    parameters = []
    for path in helpers.case_files():
        case_id = path.relative_to(helpers.CASES_DIR).as_posix()
        try:
            case = helpers.load_case(path)
        except CaseError as exc:
            parameters.append(pytest.param(str(exc), "unusable", id=f"{case_id}-unusable"))
            continue
        for stage in STAGES:
            # `built` and `correct` issue in-process CUDA work, so they must carry the marker the
            # root conftest routes on: without it they scatter across xdist workers, each opening
            # its own context, which is the OOM-and-cascade the serial chain exists to prevent.
            marks = [requires_cuda] if stage in ("built", "correct") else []
            if case.xfail_stage == stage:
                marks.append(pytest.mark.xfail(strict=True, reason=f"known gap — {helpers.evidence_line(path)}"))
            elif case.xfail_stage is not None and STAGES.index(stage) > STAGES.index(case.xfail_stage):
                # The schedule never realizes, so the stages past the gap have nothing to run.
                marks.append(pytest.mark.skip(reason=f"open case: the gap at {case.xfail_stage} blocks {stage}"))
            parameters.append(pytest.param(case, stage, id=f"{case_id}-{stage}", marks=marks))
    return parameters


@pytest.mark.parametrize(("case", "stage"), _parameters())
def test_realization(case, stage):
    """Assert one stage of one case."""
    if stage == "unusable":
        raise AssertionError(case)
    if stage == "offered":
        assert (reason := helpers.offered(case)) is None, reason
        return
    if stage == "realized":
        assert (reason := helpers.realized(case)) is None, reason
        return
    live = device_compute_capability()
    if live != case.compute_cap:
        pytest.skip(f"case declares sm_{_spell(case.compute_cap)}; this card is sm_{_spell(live)}")
    if stage == "built":
        _COMPILED[case.id] = helpers.built(case)
        return
    compiled = _COMPILED.pop(case.id, None)
    helpers.correct(case, helpers.built(case) if compiled is None else compiled)


def _spell(cap: tuple[int, int]) -> str:
    return "".join(str(part) for part in cap)


# --- staleness --------------------------------------------------------------------------------
#
# Detection is a test, not a command: the check is GPU-free and roughly 0.1 s per case, so codec
# and kernel-identity drift is caught on the pull request that causes it rather than weeks later
# on a GPU box. ``make test-corpus-regen`` only APPLIES the fix — the same split as
# ``ruff format --check`` / ``make format`` and the session-end durations gate.


@pytest.mark.parametrize("path", helpers.case_files(), ids=lambda path: path.relative_to(helpers.CASES_DIR).as_posix())
def test_case_derived_half_is_current(path):
    """The stored program wire, target, realization name, identity and canonical knobs still equal
    what this compiler derives from the case's own program."""
    case = helpers.load_case(path)
    assert helpers.regenerate(case.document) == case.document, (
        f"{path.name} is stale — a kernel identity or a schedule codec moved under it. "
        "Run `make test-corpus-regen` to restamp it; that command refuses to write when a case's "
        "verdict also changed, which is a review conversation rather than a mechanical step."
    )
