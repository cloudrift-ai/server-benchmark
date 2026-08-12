"""Fail-closed command publication provenance."""

from emmy.benchmark.execution import _apply_command_provenance_gate
from tests.benchmark.test_results import SYSTEM_INFO_RAW


def test_required_command_provenance_flips_success_to_failure():
    info = {}

    assert _apply_command_provenance_gate(True, info, "", None, required=True, dry_run=False) is False
    assert info["provenance_errors"] == [
        "staged source manifest",
        "GPU provenance",
        "CUDA compiler provenance",
    ]


def test_complete_command_provenance_preserves_status():
    info = {}
    source = {"source_id": "abc", "files": {"emmy/a.py": "hash"}}

    assert _apply_command_provenance_gate(True, info, SYSTEM_INFO_RAW, source, required=True, dry_run=False) is True
    assert info == {}


def test_dry_run_does_not_require_real_host_provenance():
    info = {}

    assert _apply_command_provenance_gate(True, info, "", None, required=True, dry_run=True) is True
    assert info == {}
