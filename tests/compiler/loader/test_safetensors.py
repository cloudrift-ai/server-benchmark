"""Checkpoint-directory resolution: the ``<repo>@<revision>`` spelling and the guard on an
unpinned multi-branch repo.

These pin the contract that a pinned rung survives every hop from a caller's model id down to a
snapshot directory. It was broken end to end before: ``--revision`` reached vLLM, and the runner
then re-resolved the checkpoint off the repo's DEFAULT branch — so a repo publishing one EXL3 rung
per branch traced one rung's geometry against another rung's weights.
"""

import json
import logging

import pytest


def test_candidate_keys_accepts_laguna_shared_expert_alias():
    from emmy.compiler.loader.safetensors import _candidate_keys

    plural = "model.layers.1.mlp.shared_experts.gate_proj.weight"
    singular = "model.layers.1.mlp.shared_expert.gate_proj.weight"
    assert singular in _candidate_keys(plural)
    assert plural in _candidate_keys(singular)


def test_split_revision_forms(tmp_path):
    from emmy.compiler.loader.safetensors import split_revision

    assert split_revision("turboderp/GLM-4.5-Air-exl3@2.25bpw") == ("turboderp/GLM-4.5-Air-exl3", "2.25bpw")
    assert split_revision("turboderp/GLM-4.5-Air-exl3@6a309ed6") == ("turboderp/GLM-4.5-Air-exl3", "6a309ed6")
    assert split_revision("Qwen/Qwen3-0.6B") == ("Qwen/Qwen3-0.6B", None)
    # A local directory whose name contains '@' is a path, not a tagged id.
    tagged_dir = tmp_path / "snap@2.0bpw"
    tagged_dir.mkdir()
    assert split_revision(str(tagged_dir)) == (str(tagged_dir), None)


def test_resolve_model_dir_forwards_the_tagged_revision(tmp_path, monkeypatch):
    """The tagged half must reach ``snapshot_download``; the repo half must arrive stripped."""
    import huggingface_hub

    from emmy.compiler.loader.safetensors import _resolve_model_dir

    seen = {}

    def fake_snapshot_download(repo, revision=None, **kw):
        seen["repo"], seen["revision"] = repo, revision
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    _resolve_model_dir("acme/coded@2.25bpw")
    assert seen == {"repo": "acme/coded", "revision": "2.25bpw"}

    # An explicit argument wins over the tagged one.
    _resolve_model_dir("acme/coded@2.25bpw", "2.0bpw")
    assert seen == {"repo": "acme/coded", "revision": "2.0bpw"}


def test_resolve_model_dir_warns_when_a_multi_branch_repo_is_unpinned(tmp_path, monkeypatch, caplog):
    """The library half of ``warm.sh``'s refusal: taking the default branch of a repo that
    publishes several must never be silent."""
    import huggingface_hub

    from emmy.compiler.loader import safetensors as sf

    sf.warn_if_unpinned.cache_clear()
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo, revision=None, **kw: str(tmp_path))
    monkeypatch.setattr(
        huggingface_hub,
        "list_repo_refs",
        lambda repo: type("Refs", (), {"branches": [type("B", (), {"name": n}) for n in ("main", "2.0bpw", "2.25bpw")]}),
    )
    caplog.set_level(logging.WARNING, logger="emmy.compiler.loader.safetensors")
    sf._resolve_model_dir("acme/coded")
    assert any("no revision was pinned" in r.getMessage() for r in caplog.records)
    assert any("2.25bpw" in r.getMessage() for r in caplog.records), "the warning must name the branches"

    # A pinned id says nothing.
    caplog.clear()
    sf.warn_if_unpinned.cache_clear()
    sf._resolve_model_dir("acme/coded@2.25bpw")
    assert not caplog.records
    sf.warn_if_unpinned.cache_clear()


def test_quantized_checkpoint_dir_reads_config_at_the_pinned_revision(tmp_path, monkeypatch):
    """Detection reads ``config.json`` from the hub — at the DEFAULT branch before this fix, so a
    tagged id both mis-detected the scheme and snapshot-downloaded the wrong rung."""
    import huggingface_hub

    from emmy.compiler.trace.huggingface import quantized_checkpoint_dir

    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "exl3", "bits": 2.26}}))
    seen = {}

    def fake_hf_hub_download(repo, filename, revision=None, **kw):
        seen["cfg"] = (repo, filename, revision)
        return str(snap / filename)

    def fake_snapshot_download(repo, revision=None, **kw):
        seen["snapshot"] = (repo, revision)
        return str(snap)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    assert quantized_checkpoint_dir("acme/coded@2.25bpw") == snap
    assert seen["cfg"] == ("acme/coded", "config.json", "2.25bpw")
    assert seen["snapshot"] == ("acme/coded", "2.25bpw")


def test_pinned_model_id_tags_the_vllm_revision():
    """The seam where ``--revision`` crosses into emmy: vLLM keeps the repo id and the revision in
    two fields, and only the id used to reach the runner."""
    pytest.importorskip("vllm")
    from emmy.serving.vllm_model import pinned_model_id

    def mc(model, revision):
        return type("MC", (), {"model": model, "revision": revision})

    assert pinned_model_id(mc("turboderp/GLM-4.5-Air-exl3", "6a309ed6")) == "turboderp/GLM-4.5-Air-exl3@6a309ed6"
    assert pinned_model_id(mc("Qwen/Qwen3-0.6B", None)) == "Qwen/Qwen3-0.6B"
    assert pinned_model_id(mc("/local/snapshot/dir", None)) == "/local/snapshot/dir"
