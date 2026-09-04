"""Gen-runner pack round-trip (``EMMY_PACK_DIR``): the first ``from_model`` boot writes the
pack, the second boots from it (no trace / compile) and produces identical layer outputs.
Needs CUDA + cupy (skips itself otherwise); tiny random Qwen3, same pattern as
``test_gen_runner_gpu``."""

import logging

import numpy as np
import pytest

pytestmark = [pytest.mark.xdist_group("cuda")]


def test_gen_pack_second_boot_hits_and_matches(tmp_path, monkeypatch, caplog):
    """A pack round-trip: the first boot compiles and writes one pack, the second loads it and
    reproduces the first boot's outputs byte for byte.

    This carried an ``xfail`` while the lane compiled cold — the refit steered its tiny fp32 shape
    onto a TMA-staged pick with run-to-run last-ulp instability, so byte equality was not a
    property two boots could hold. The golden decides the pick now, both boots land on one
    schedule, and the equality holds. Neither boot takes the session plan cache: the claim is that
    the FIRST compiles and the second hits the pack, which a warm template cache would decide.
    """
    pytest.importorskip("cupy")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from tests.serving import helpers

    monkeypatch.setenv("EMMY_PACK_DIR", str(tmp_path))
    # No session plan cache: this test's whole claim is that the FIRST boot compiles and the
    # second loads the pack, and a warm template cache would decide the first boot for it.
    shape = "qwen3.l2.b16"
    model = helpers.RUNNERS[shape][0]()

    caplog.set_level(logging.INFO, logger="emmy.serving.gen_runner")
    first = helpers.build(shape, model=model)
    manifests = list(tmp_path.glob("*/manifest.json"))
    assert len(manifests) == 1, "the full-compile boot must write exactly one pack"
    assert not any("pack hit" in r.message for r in caplog.records)

    caplog.clear()
    second = helpers.build(shape, model=model)
    assert any("pack hit" in r.message for r in caplog.records), "second boot must load the pack"
    assert second.has_device_decode == first.has_device_decode

    t = 7
    hidden = first.embed(list(range(1, t + 1)))
    position_ids = torch.arange(t).unsqueeze(0)
    q1, k1, v1 = first.forward_layer_pre(0, hidden, position_ids)
    q2, k2, v2 = second.forward_layer_pre(0, hidden, position_ids)
    np.testing.assert_array_equal(q2, q1)
    np.testing.assert_array_equal(k2, k1)
    np.testing.assert_array_equal(v2, v1)
    attn = np.ascontiguousarray(np.random.default_rng(0).standard_normal((t, 4 * 16)).astype(np.float32) * 0.1)
    np.testing.assert_array_equal(second.forward_layer_post(0, attn, hidden), first.forward_layer_post(0, attn, hidden))


def test_gen_pack_key_separates_quantized_rungs(tmp_path, monkeypatch):
    """Two rungs of one coded repo must not share a pack.

    Their architecture configs are identical — the rungs differ only in the per-tensor bit
    allocation, which the twin's config no longer carries (``load_quantized_split`` strips
    ``quantization_config`` before ``from_config``). So the config hash alone put both boots in
    ONE directory, where the second warm overwrote the first and either boot could load plans
    built for the other rung's coded extents. Same model, two checkpoints ⇒ two packs."""
    pytest.importorskip("cupy")
    import json

    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from tests.serving import helpers

    monkeypatch.setenv("EMMY_PACK_DIR", str(tmp_path))
    shape = "qwen3.l1.b16"
    model = helpers.RUNNERS[shape][0]()

    def rung(name, bits):
        d = tmp_path / name
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "exl3", "bits": bits}}))
        return {"dir": str(d), "trunk": "values", "fmt": None, "layers": {}}

    # The store moves the pack KEY; the dense twin it compiles is the table's shape either way.
    helpers.build(shape, model=model, expert_store=rung("r200", 2.0))
    helpers.build(shape, model=model, expert_store=rung("r225", 2.26))
    keys = [json.loads(m.read_text())["key"] for m in tmp_path.glob("*/manifest.json")]
    assert len(keys) == 2, f"each rung needs its own pack, got {keys}"
    assert keys[0]["config_sha"] == keys[1]["config_sha"], "the aliasing precondition: the config hash cannot tell them apart"
    assert keys[0]["quant_sha"] != keys[1]["quant_sha"]
