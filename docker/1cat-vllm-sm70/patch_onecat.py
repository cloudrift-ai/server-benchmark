#!/usr/bin/env python3
"""Apply the pinned pipeline-parallel profile fix missing from 1Cat-vLLM 1.2.2."""

from __future__ import annotations

import sysconfig
from pathlib import Path


def replace_once(source: str, old: str, new: str, label: str) -> str:
    """Replace one exact upstream block and fail closed if the pinned wheel drifts."""
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one {label} block, found {count}")
    return source.replace(old, new)


purelib = Path(sysconfig.get_paths()["purelib"])
runner_path = purelib / "vllm" / "v1" / "worker" / "gpu_model_runner.py"
runner = runner_path.read_text(encoding="utf-8")

runner = replace_once(
    runner,
    '    ) -> tuple[torch.Tensor, torch.Tensor]:\n        """\n        Run a dummy forward pass to warm up/profile run or capture the\n',
    "    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:\n"
    '        """\n'
    "        Run a dummy forward pass to warm up/profile run or capture the\n",
    "_dummy_run return annotation",
)

runner = replace_once(
    runner,
    "        logit_indices = np.cumsum(num_scheduled_tokens) - 1\n"
    "        logit_indices_device = torch.from_numpy(logit_indices).to(\n"
    "            self.device, non_blocking=True\n"
    "        )\n"
    "        _sm70_profile_trace(\n"
    '            "_dummy_run return hidden_shape=%s sampled_count=%s",\n'
    "            tuple(hidden_states.shape),\n"
    "            len(logit_indices),\n"
    "        )\n"
    "        return hidden_states, hidden_states[logit_indices_device]\n",
    runner_tail := (
        "        if not get_pp_group().is_last_rank:\n"
        '            _sm70_profile_trace("_dummy_run return non-last PP rank")\n'
        "            return None, None\n"
        "\n"
        "        logit_indices = np.cumsum(num_scheduled_tokens) - 1\n"
        "        logit_indices_device = torch.from_numpy(logit_indices).to(\n"
        "            self.device, non_blocking=True\n"
        "        )\n"
        "        _sm70_profile_trace(\n"
        '            "_dummy_run return hidden_shape=%s sampled_count=%s",\n'
        "            tuple(hidden_states.shape),\n"
        "            len(logit_indices),\n"
        "        )\n"
        "        return hidden_states, hidden_states[logit_indices_device]\n"
    ),
    "_dummy_run pipeline output",
)

runner = replace_once(
    runner,
    "        _sm70_profile_trace(\n"
    '            "profile_run dummy_run exit hidden_shape=%s last_hidden_shape=%s",\n'
    "            tuple(hidden_states.shape),\n"
    "            tuple(last_hidden_states.shape),\n"
    "        )\n"
    "        if get_pp_group().is_last_rank:\n"
    "            if self.is_pooling_model:\n",
    "        if get_pp_group().is_last_rank:\n"
    "            assert hidden_states is not None\n"
    "            assert last_hidden_states is not None\n"
    "            _sm70_profile_trace(\n"
    '                "profile_run dummy_run exit hidden_shape=%s last_hidden_shape=%s",\n'
    "                tuple(hidden_states.shape),\n"
    "                tuple(last_hidden_states.shape),\n"
    "            )\n"
    "            if self.is_pooling_model:\n",
    "profile_run pipeline output",
)

# The exact replacement above intentionally leaves the existing non-last-rank `else` path in
# place. Assert the distinguishing blocks before writing so a partial edit never reaches an image.
if runner.count(runner_tail) != 1 or runner.count("assert last_hidden_states is not None") != 1:
    raise RuntimeError("pipeline-parallel profile patch validation failed")

runner_path.write_text(runner, encoding="utf-8")
