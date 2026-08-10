"""Canonical serving-image publication rules and Docker execution."""

import json
import subprocess

import pytest

from emmy.publish import (
    PublishError,
    build_publish_plan,
    load_publish_recipe,
    parse_publish_reference,
    publish_recipe_image,
)
from emmy.recipe import Recipe, VllmConfig


def _recipe(image="cloudriftai/vllm-emmy-model-name:0.23.0-abcdef123", model="org/Model Name"):
    recipe = Recipe()
    recipe.model.huggingface = model
    recipe.engine.llm.vllm = VllmConfig(image=image)
    return recipe


def _labels(family="vllm-emmy", version="0.23.0", revision="abcdef1234567890abcdef1234567890abcdef12"):
    return {
        "ai.emmy.publish.family": family,
        "org.opencontainers.image.version": version,
        "org.opencontainers.image.revision": revision,
        "ai.emmy.model.id": "org/Model Name",
    }


class RecordingRunner:
    def __init__(self, labels=None, repo_digests=(), remote_digest=None, pushed_digest=None, remote_error=None):
        self.labels = labels if labels is not None else _labels()
        self.repo_digests = list(repo_digests)
        self.remote_digest = remote_digest
        self.pushed_digest = pushed_digest or f"sha256:{'a' * 64}"
        self.remote_error = remote_error
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        stdout = ""
        if command[:3] == ["docker", "image", "inspect"]:
            stdout = json.dumps(
                [
                    {
                        "Id": f"sha256:{'d' * 64}",
                        "Config": {"Labels": self.labels},
                        "RepoDigests": self.repo_digests,
                    }
                ]
            )
        elif command[:4] == ["docker", "buildx", "imagetools", "inspect"]:
            if self.remote_error:
                return subprocess.CompletedProcess(command, 1, stdout="", stderr=self.remote_error)
            if self.remote_digest is None:
                return subprocess.CompletedProcess(command, 1, stdout="", stderr="manifest unknown")
            stdout = json.dumps(self.remote_digest)
        elif command[:2] == ["docker", "push"]:
            self.remote_digest = self.pushed_digest
            self.repo_digests = [f"cloudriftai/model@{self.pushed_digest}"]
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")


@pytest.mark.parametrize(
    "image,family,version,revision",
    [
        ("cloudriftai/vllm-emmy-model-name:0.23.0-abcdef1", "vllm-emmy", "0.23.0", "abcdef1"),
        ("cloudriftai/1cat-vllm-model-name:1.2.3-d76126608155", "1cat-vllm", "1.2.3", "d76126608155"),
    ],
)
def test_parse_publish_reference_accepts_both_families(image, family, version, revision):
    parsed = parse_publish_reference(image, "org/Model Name")

    assert parsed.family == family
    assert parsed.runtime_version == version
    assert parsed.source_revision == revision


@pytest.mark.parametrize(
    "image",
    [
        "cloudriftai/vllm-emmy-model-name:latest",
        "cloudriftai/onecat-vllm-model-name:1.2.3-abcdef1",
        "cloudriftai/1cat-vllm-model-name:sm70-abcdef1",
        "cloudriftai/1cat-vllm-model-name:1.2.3-abcdef1-jitfree",
        "cloudriftai/vllm-emmy-model-name:v0.23.0-abcdef1",
        "cloudriftai/vllm-emmy-model-name:0.23.0-abc123",
        "cloudriftai/vllm-emmy-model-name:0.23.0-abcdef1234567",
    ],
)
def test_parse_publish_reference_rejects_noncanonical_tags(image):
    with pytest.raises(PublishError, match="recipe image must match"):
        parse_publish_reference(image, "org/Model Name")


def test_parse_publish_reference_requires_the_recipe_model_slug():
    with pytest.raises(PublishError, match="does not match model"):
        parse_publish_reference("cloudriftai/vllm-emmy-other:0.23.0-abcdef1", "org/Model Name")


def test_build_publish_plan_retags_an_explicit_local_source():
    plan = build_publish_plan(_recipe(), "local/deepseek:baked")

    assert plan.source_image == "local/deepseek:baked"
    assert plan.commands == (
        ("docker", "tag", "local/deepseek:baked", "cloudriftai/vllm-emmy-model-name:0.23.0-abcdef123"),
        ("docker", "push", "cloudriftai/vllm-emmy-model-name:0.23.0-abcdef123"),
    )


def test_dry_run_inspects_metadata_but_does_not_mutate_images():
    runner = RecordingRunner()

    plan = publish_recipe_image(_recipe(), source_image="local/deepseek:baked", dry_run=True, runner=runner)

    assert plan.destination.image == "cloudriftai/vllm-emmy-model-name:0.23.0-abcdef123"
    assert plan.source_image_id == f"sha256:{'d' * 64}"
    assert plan.registry_digest is None
    assert [call[0] for call in runner.calls] == [
        ["docker", "image", "inspect", "local/deepseek:baked"],
        [
            "docker",
            "buildx",
            "imagetools",
            "inspect",
            "cloudriftai/vllm-emmy-model-name:0.23.0-abcdef123",
            "--format",
            "{{json .Manifest.Digest}}",
        ],
    ]


def test_publish_requires_explicit_noninteractive_confirmation():
    runner = RecordingRunner()

    with pytest.raises(PublishError, match="without --yes"):
        publish_recipe_image(_recipe(), runner=runner)

    assert runner.calls == []


def test_publish_retags_then_pushes_after_metadata_validation():
    runner = RecordingRunner()

    publish_recipe_image(_recipe(), source_image="local/deepseek:baked", yes=True, runner=runner)

    assert [call[0] for call in runner.calls] == [
        ["docker", "image", "inspect", "local/deepseek:baked"],
        [
            "docker",
            "buildx",
            "imagetools",
            "inspect",
            "cloudriftai/vllm-emmy-model-name:0.23.0-abcdef123",
            "--format",
            "{{json .Manifest.Digest}}",
        ],
        ["docker", "tag", "local/deepseek:baked", "cloudriftai/vllm-emmy-model-name:0.23.0-abcdef123"],
        ["docker", "push", "cloudriftai/vllm-emmy-model-name:0.23.0-abcdef123"],
        ["docker", "image", "inspect", "cloudriftai/vllm-emmy-model-name:0.23.0-abcdef123"],
        [
            "docker",
            "buildx",
            "imagetools",
            "inspect",
            "cloudriftai/vllm-emmy-model-name:0.23.0-abcdef123",
            "--format",
            "{{json .Manifest.Digest}}",
        ],
    ]


def test_publish_skips_retag_when_the_recipe_image_is_already_local():
    runner = RecordingRunner()

    publish_recipe_image(_recipe(), yes=True, runner=runner)

    assert [call[0] for call in runner.calls] == [
        ["docker", "image", "inspect", "cloudriftai/vllm-emmy-model-name:0.23.0-abcdef123"],
        [
            "docker",
            "buildx",
            "imagetools",
            "inspect",
            "cloudriftai/vllm-emmy-model-name:0.23.0-abcdef123",
            "--format",
            "{{json .Manifest.Digest}}",
        ],
        ["docker", "push", "cloudriftai/vllm-emmy-model-name:0.23.0-abcdef123"],
        ["docker", "image", "inspect", "cloudriftai/vllm-emmy-model-name:0.23.0-abcdef123"],
        [
            "docker",
            "buildx",
            "imagetools",
            "inspect",
            "cloudriftai/vllm-emmy-model-name:0.23.0-abcdef123",
            "--format",
            "{{json .Manifest.Digest}}",
        ],
    ]


@pytest.mark.parametrize(
    "labels,message",
    [
        ({}, "missing required label"),
        (_labels(family="1cat-vllm"), "family label"),
        (_labels(version="0.22.0"), "version label"),
        (_labels(revision="1234567"), "does not match destination revision"),
        (_labels(revision="not-a-git-revision"), "lowercase hexadecimal Git revision"),
    ],
)
def test_publish_rejects_missing_or_mismatched_metadata(labels, message):
    runner = RecordingRunner(labels)

    with pytest.raises(PublishError, match=message):
        publish_recipe_image(_recipe(), dry_run=True, runner=runner)

    assert len(runner.calls) == 1


def test_publish_requires_the_exact_recipe_model_label():
    labels = _labels()
    labels["ai.emmy.model.id"] = "another/model"

    with pytest.raises(PublishError, match="model label"):
        publish_recipe_image(_recipe(), dry_run=True, runner=RecordingRunner(labels))


def test_existing_destination_is_only_idempotent_for_the_same_repo_digest():
    digest = f"sha256:{'b' * 64}"
    runner = RecordingRunner(repo_digests=[f"some/source@{digest}"], remote_digest=digest)

    plan = publish_recipe_image(_recipe(), source_image="local/deepseek:baked", dry_run=True, runner=runner)

    assert plan.already_published is True
    assert plan.registry_digest == digest
    assert plan.commands == ()


def test_existing_destination_with_an_unproven_digest_is_rejected():
    runner = RecordingRunner(remote_digest=f"sha256:{'b' * 64}")

    with pytest.raises(PublishError, match="refusing to overwrite"):
        publish_recipe_image(_recipe(), dry_run=True, runner=runner)


def test_registry_lookup_failure_is_not_treated_as_an_absent_destination():
    runner = RecordingRunner(remote_error="authorization failed")

    with pytest.raises(PublishError, match="could not inspect registry destination"):
        publish_recipe_image(_recipe(), dry_run=True, runner=runner)


def test_missing_buildx_is_not_treated_as_an_absent_destination():
    runner = RecordingRunner(remote_error="docker-buildx: executable file not found in $PATH")

    with pytest.raises(PublishError, match="could not inspect registry destination"):
        publish_recipe_image(_recipe(), dry_run=True, runner=runner)


def test_post_push_digest_must_match_the_local_repo_digest():
    class MismatchedPushRunner(RecordingRunner):
        def __call__(self, command, **kwargs):
            result = super().__call__(command, **kwargs)
            if command[:2] == ["docker", "push"]:
                self.repo_digests = [f"cloudriftai/model@sha256:{'c' * 64}"]
            return result

    with pytest.raises(PublishError, match="post-push registry digest"):
        publish_recipe_image(_recipe(), yes=True, runner=MismatchedPushRunner())


def test_load_publish_recipe_resolves_one_matrix_variant(tmp_path):
    (tmp_path / "recipe.yaml").write_text(
        """
model:
  huggingface: org/Model Name
engine:
  llm:
    vllm:
      image: cloudriftai/vllm-emmy-model-name:0.23.0-abcdef123
matrices:
  deploy.gpu: NVIDIA V100
  deploy.gpu_count: 8
"""
    )

    recipe = load_publish_recipe(tmp_path)

    assert recipe.deploy.gpu == "NVIDIA V100"
    assert recipe.deploy.gpu_count == 8


def test_load_publish_recipe_rejects_multiple_matrix_variants(tmp_path):
    (tmp_path / "recipe.yaml").write_text(
        """
model:
  huggingface: org/Model Name
engine:
  llm:
    vllm:
      image: cloudriftai/vllm-emmy-model-name:0.23.0-abcdef123
matrices:
  deploy.gpu: [NVIDIA V100, NVIDIA H100]
"""
    )

    with pytest.raises(PublishError, match="matrix expands to 2 variants"):
        load_publish_recipe(tmp_path)


def test_publish_rejects_command_recipes():
    recipe = Recipe.from_dict({"command": {"run": "true"}})

    with pytest.raises(PublishError, match="only inference recipes"):
        build_publish_plan(recipe)
