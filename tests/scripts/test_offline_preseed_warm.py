"""The initial serving-image warm can be intentionally offline.

This is distinct from the later fixpoint boots, which are always offline.  A release may
start from a locally assembled immutable HF snapshot when the publishing namespace is not
reachable from the build host; dropping the caller's ``HF_HUB_OFFLINE=1`` on the first
container would turn that supported path into a network lookup for a different artifact.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WARM_SCRIPT = PROJECT_ROOT / "docker" / "vllm-emmy-serve" / "warm.sh"


def test_initial_warm_forwards_offline_mode_to_the_container():
    body = WARM_SCRIPT.read_text()
    initial, fixpoint = body.split("fixpoint()", 1)

    assert "-e HF_HUB_OFFLINE" in initial
    assert 'if [ "${HF_HUB_OFFLINE:-}" = "1" ]' in initial
    assert "-e HF_HUB_OFFLINE=1" in fixpoint
