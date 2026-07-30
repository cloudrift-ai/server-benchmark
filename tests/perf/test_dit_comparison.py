"""Opt-in pretrained DiT block correctness and three-way benchmark acceptance."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from tests.compiler.conftest import requires_cuda

pytestmark = [pytest.mark.perf, requires_cuda]


def test_dit_xl_layer0_pretrained_comparison(tmp_path):
    """Layer 0 passes the fatal eager gate and records eager/compile/Emmy latency."""
    if os.environ.get("EMMY_RUN_DIT_PRETRAINED") != "1":
        pytest.skip("set EMMY_RUN_DIT_PRETRAINED=1 to download and benchmark facebook/DiT-XL-2-256")

    output = tmp_path / "dit-layer0.json"
    command = [
        sys.executable,
        "-m",
        "emmy.emmy",
        "run",
        "facebook/DiT-XL-2-256",
        "--adapter",
        "dit",
        "--layer",
        "0",
        "--bench",
        "--bench-backends",
        "eager,tcompile,emmy",
        "--warmup",
        "10",
        "--iters",
        "100",
        "--json",
        str(output),
    ]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=3600, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    payload = json.loads(output.read_text())
    assert set(payload["backends"]) == {"Eager PyTorch", "torch.compile", "Emmy"}
    for row in payload["backends"].values():
        assert row["latency_us"] > 0
        assert row["speedup_vs_eager"] > 0
