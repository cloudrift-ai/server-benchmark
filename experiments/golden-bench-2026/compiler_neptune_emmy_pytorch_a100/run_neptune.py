#!/usr/bin/env python3
"""Run the pinned Neptune CLI on the A100 80GB using its published A100 target."""

import runpy
import sys
from pathlib import Path

import torch


_A100_40GB = "NVIDIA A100-SXM4-40GB"
_A100_80GB = "NVIDIA A100-SXM4-80GB"
_get_device_name = torch.cuda.get_device_name


def _artifact_device_name(device=None):
    name = _get_device_name(device)
    return _A100_40GB if name == _A100_80GB else name


def main() -> None:
    actual_name = _get_device_name(0)
    if actual_name != _A100_80GB:
        raise RuntimeError(f"Expected {_A100_80GB}, found {actual_name}")
    # Neptune recognizes only the 40GB product string. Both cards use its sm_80 A100 target;
    # keep the pinned source unchanged and narrow the alias to this artifact process.
    torch.cuda.get_device_name = _artifact_device_name
    sys.path.insert(0, str(Path.cwd()))
    from scripts.neptune_bench import ours

    # The pinned CLI contains a truncated `.e(...)` call for the prefill SoftCap runner. Current
    # upstream spells the call with this existing factory; supply only the missing alias.
    ours.NeptuneGQARunner.e = ours.NeptuneGQARunner.create_flex_from_schedulers
    runpy.run_module("scripts.neptune_bench", run_name="__main__")


if __name__ == "__main__":
    main()
