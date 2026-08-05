"""Detection of a prebuilt image's baked HF cache (``_baked_hf_cache``).

The per-model serving images bake the model snapshot under their own HF_HOME and set
HF_HUB_OFFLINE=1. Deploy used to point HF_HOME at a host cache regardless, which hid the
snapshot while offline mode stayed on — the download step then failed with "Local entry
not found ... offline mode is enabled" and no deploy was possible from a prebuilt image.
These pin the signal that turns the override off.
"""

import pytest

from emmy.deploy.orchestrate import _baked_hf_cache


def _fake_run_cmd(rc, out):
    async def run_cmd(cmd, timeout=None, **kwargs):
        assert "docker image inspect" in cmd
        return rc, out, ""

    return run_cmd


ENV_OFFLINE = "PATH=/usr/bin\nHF_HOME=/opt/emmy/hf\nHF_HUB_OFFLINE=1\n"


@pytest.mark.asyncio
async def test_detects_offline_image_cache():
    assert await _baked_hf_cache(_fake_run_cmd(0, ENV_OFFLINE), "img") == "/opt/emmy/hf"


@pytest.mark.asyncio
async def test_hf_home_alone_is_not_a_baked_cache():
    """A stock image may set HF_HOME without shipping weights — it can still download,
    so the host cache override stays."""
    assert await _baked_hf_cache(_fake_run_cmd(0, "HF_HOME=/root/.cache/huggingface\n"), "img") is None


@pytest.mark.asyncio
async def test_offline_without_hf_home_is_none():
    assert await _baked_hf_cache(_fake_run_cmd(0, "HF_HUB_OFFLINE=1\n"), "img") is None


@pytest.mark.asyncio
async def test_plain_image_returns_none():
    assert await _baked_hf_cache(_fake_run_cmd(0, "PATH=/usr/bin\nLANG=C\n"), "img") is None


@pytest.mark.asyncio
async def test_inspect_failure_falls_back_to_download():
    """An un-inspectable image must not silently skip the download."""
    assert await _baked_hf_cache(_fake_run_cmd(1, ""), "img") is None
    assert await _baked_hf_cache(_fake_run_cmd(0, ""), "img") is None


@pytest.mark.asyncio
async def test_value_containing_equals_survives():
    out = "HF_HOME=/opt/emmy/hf\nHF_HUB_OFFLINE=1\nOTHER=a=b\n"
    assert await _baked_hf_cache(_fake_run_cmd(0, out), "img") == "/opt/emmy/hf"
