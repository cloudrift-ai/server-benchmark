"""Tests for the GPU power/VRAM sampler (fake run_cmd, no nvidia-smi needed)."""

import re

import pytest

from emmy.benchmark.gpu_sampler import gpu_sampling, parse_gpu_samples

# Two GPUs, two ticks, interleaved as nvidia-smi emits them.
MULTI_GPU_CSV = """\
0, 100.0, 1000
1, 200.0, 2000
0, 300.0, 3000
1, 400.0, 6000
"""


def make_fake_run_cmd(responses):
    """A fake run_cmd returning canned (rc, stdout, stderr) tuples in order.

    Returns (run_cmd, calls); calls records dicts of the issued command and kwargs.
    """
    calls = []

    async def run_cmd(command, stream=True, timeout=600):
        calls.append({"command": command, "stream": stream, "timeout": timeout})
        return responses[len(calls) - 1]

    return run_cmd, calls


def remote_file_of(start_command):
    match = re.search(r"> (/tmp/emmy_gpu_samples_[0-9a-f]+\.csv) 2>&1", start_command)
    assert match, start_command
    return match.group(1)


async def test_command_sequence():
    run_cmd, calls = make_fake_run_cmd([(0, "12345\n", ""), (0, MULTI_GPU_CSV, "")])
    async with gpu_sampling(run_cmd):
        pass

    assert len(calls) == 2
    start, stop = calls[0]["command"], calls[1]["command"]
    assert start.startswith("nohup nvidia-smi --query-gpu=index,power.draw,memory.used")
    assert "--format=csv,noheader,nounits -lms 500" in start
    assert start.endswith("& echo $!")
    remote_file = remote_file_of(start)
    assert "kill 12345" in stop
    assert f"cat {remote_file}" in stop
    assert f"rm -f {remote_file}" in stop
    # stdout must be captured, not streamed.
    assert all(not c["stream"] for c in calls)


async def test_multi_gpu_summary():
    run_cmd, _ = make_fake_run_cmd([(0, "1\n", ""), (0, MULTI_GPU_CSV, "")])
    async with gpu_sampling(run_cmd) as sampler:
        pass

    summary = sampler.summary()
    assert summary is not None
    gpu0, gpu1 = summary.gpus
    assert (gpu0.index, gpu0.avg_power_w, gpu0.peak_memory_mib, gpu0.samples) == (0, 200.0, 3000.0, 2)
    assert (gpu1.index, gpu1.avg_power_w, gpu1.peak_memory_mib, gpu1.samples) == (1, 300.0, 6000.0, 2)
    assert summary.total_avg_power_w == 500.0
    assert summary.peak_memory_mib == 6000.0


async def test_summary_to_dict():
    run_cmd, _ = make_fake_run_cmd([(0, "1\n", ""), (0, "0, 50.0, 1000\n", "")])
    async with gpu_sampling(run_cmd) as sampler:
        pass

    assert sampler.summary().to_dict() == {
        "gpus": [{"index": 0, "avg_power_w": 50.0, "peak_memory_mib": 1000.0, "samples": 1}],
        "total_avg_power_w": 50.0,
        "peak_memory_mib": 1000.0,
    }


async def test_na_values_and_truncated_last_line():
    csv = (
        "0, [N/A], 1000\n"  # power N/A: memory still counts
        "1, [N/A], [N/A]\n"  # GPU with no valid samples at all
        "0, 50.0, [N/A]\n"
        "\n"
        "0, 12"  # truncated mid-write: 2 fields, skipped
    )
    run_cmd, _ = make_fake_run_cmd([(0, "1\n", ""), (0, csv, "")])
    async with gpu_sampling(run_cmd) as sampler:
        pass

    summary = sampler.summary()
    gpu0, gpu1 = summary.gpus
    assert (gpu0.avg_power_w, gpu0.peak_memory_mib, gpu0.samples) == (50.0, 1000.0, 2)
    assert (gpu1.avg_power_w, gpu1.peak_memory_mib, gpu1.samples) == (None, None, 1)
    assert summary.total_avg_power_w == 50.0
    assert summary.peak_memory_mib == 1000.0


async def test_empty_sample_file():
    run_cmd, calls = make_fake_run_cmd([(0, "1\n", ""), (0, "", "")])
    async with gpu_sampling(run_cmd) as sampler:
        pass
    assert len(calls) == 2
    assert sampler.summary() is None


async def test_start_failure_skips_collection():
    run_cmd, calls = make_fake_run_cmd([(1, "", "")])
    async with gpu_sampling(run_cmd) as sampler:
        pass
    assert len(calls) == 1  # no kill/cat when the sampler never started
    assert sampler.summary() is None


async def test_start_without_pid_skips_collection():
    run_cmd, calls = make_fake_run_cmd([(0, "not a pid", "")])
    async with gpu_sampling(run_cmd) as sampler:
        pass
    assert len(calls) == 1
    assert sampler.summary() is None


async def test_collection_failure():
    run_cmd, _ = make_fake_run_cmd([(0, "1\n", ""), (1, "", "")])
    async with gpu_sampling(run_cmd) as sampler:
        pass
    assert sampler.summary() is None


async def test_collects_when_block_raises():
    run_cmd, calls = make_fake_run_cmd([(0, "1\n", ""), (0, MULTI_GPU_CSV, "")])
    with pytest.raises(RuntimeError, match="bench failed"):
        async with gpu_sampling(run_cmd) as sampler:
            raise RuntimeError("bench failed")
    assert len(calls) == 2  # kill + cleanup still ran
    assert sampler.summary().total_avg_power_w == 500.0


async def test_unique_remote_file_per_session():
    run_cmd1, calls1 = make_fake_run_cmd([(0, "1\n", ""), (0, "", "")])
    run_cmd2, calls2 = make_fake_run_cmd([(0, "1\n", ""), (0, "", "")])
    async with gpu_sampling(run_cmd1):
        pass
    async with gpu_sampling(run_cmd2):
        pass
    assert remote_file_of(calls1[0]["command"]) != remote_file_of(calls2[0]["command"])


def test_parse_rejects_non_csv_noise():
    assert parse_gpu_samples("bash: nvidia-smi: command not found\n") is None
