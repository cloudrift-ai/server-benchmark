"""Tests for scripts/bench_serve_sweep.py — the SSE hardening and the poll summary.

The hardening is a regression test for a defect that FABRICATES numbers rather than raising:
`vllm bench serve` splits server-sent events on "\\n\\n", so a CRLF-framed stream survives only
through its single-JSON fallback. Under concurrency, coalesced reads wedge the parser for the
rest of the request and the client then reports empty generations with plausible-looking TPOT.
Measured against tabbyAPI at concurrency 16: 51 of 64 requests lost silently
(plans/vq-phase0-findings.md §5.1). A keepalive comment (": ping - ...") wedges it the same way.

These run against the installed vLLM's real handler class, so an upstream fix — or an upstream
rename — surfaces here rather than in a benchmark run.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("vllm", reason="the SSE handler under test lives in vllm.benchmarks")


@pytest.fixture
def handler_cls():
    """The real handler class, hardened, and restored afterwards."""
    scripts_dir = str(Path(__file__).resolve().parents[2] / "scripts")
    sys.path.insert(0, scripts_dir)
    try:
        import bench_serve_sweep
        from vllm.benchmarks.lib.endpoint_request_func import StreamedResponseHandler
    finally:
        sys.path.remove(scripts_dir)

    original = StreamedResponseHandler.add_chunk
    bench_serve_sweep._harden_sse()
    yield StreamedResponseHandler
    StreamedResponseHandler.add_chunk = original


def test_coalesced_crlf_events_are_split(handler_cls):
    """Two CRLF-framed events in one socket read. Unhardened this returns nothing and the
    buffer never drains again."""
    handler = handler_cls()
    messages = handler.add_chunk(b'data: {"a":1}\r\n\r\ndata: {"b":2}\r\n\r\n')

    assert messages == ['data: {"a":1}', 'data: {"b":2}']
    assert handler.buffer == ""


def test_keepalive_comment_is_dropped_and_does_not_wedge(handler_cls):
    """sse-starlette's 15 s ping lands whenever queueing exceeds it. The caller json.loads()
    whatever it is handed, so a comment must never reach it — and must not block the events
    behind it either."""
    handler = handler_cls()
    messages = handler.add_chunk(b': ping - 2026-08-07\r\n\r\ndata: {"c":3}\r\n\r\n')

    assert messages == ['data: {"c":3}']
    assert handler.buffer == ""


def test_plain_lf_framing_is_unchanged(handler_cls):
    """The vLLM-served lanes emit LF and no pings; the hardening must not perturb them."""
    handler = handler_cls()
    messages = handler.add_chunk(b'data: {"d":4}\n\ndata: [DONE]\n\n')

    assert messages == ['data: {"d":4}', "data: [DONE]"]


def test_partial_event_waits_for_the_rest(handler_cls):
    """A JSON body split across two reads must not be emitted twice or dropped."""
    handler = handler_cls()

    assert handler.add_chunk(b'data: {"e":') == []
    assert handler.add_chunk(b"5}\r\n\r\n") == ['data: {"e":5}']


def test_poll_summary_reports_power_and_peak_vram(tmp_path):
    scripts_dir = str(Path(__file__).resolve().parents[2] / "scripts")
    sys.path.insert(0, scripts_dir)
    try:
        from bench_serve_sweep import _summarize_poll
    finally:
        sys.path.remove(scripts_dir)

    csv = tmp_path / "p.smi.csv"
    csv.write_text("2026/08/07 10:00:00.0, 210.5, 29061\n2026/08/07 10:00:01.0, 390.5, 31299\n")
    summary = _summarize_poll(csv)

    assert summary["samples"] == 2
    assert summary["power_mean_w"] == pytest.approx(300.5)
    assert summary["power_max_w"] == pytest.approx(390.5)
    assert summary["vram_peak_mib"] == pytest.approx(31299)


def test_poll_summary_is_empty_without_a_file(tmp_path):
    """No nvidia-smi on the box is a degraded run, not a crashed one."""
    scripts_dir = str(Path(__file__).resolve().parents[2] / "scripts")
    sys.path.insert(0, scripts_dir)
    try:
        from bench_serve_sweep import _summarize_poll
    finally:
        sys.path.remove(scripts_dir)

    assert _summarize_poll(tmp_path / "missing.csv") == {}
