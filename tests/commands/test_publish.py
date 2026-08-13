"""CLI registration for serving-image publication."""

import argparse
import json
import os
import subprocess
import sys

from emmy.commands.publish import handle_publish, register_publish_command


def test_publish_command_parses_recipe_source_and_safety_flags():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_publish_command(subparsers)

    args = parser.parse_args(["publish", "recipes/model", "--source-image", "local/model:baked", "--dry-run"])

    assert args.command == "publish"
    assert args.recipe == "recipes/model"
    assert args.source_image == "local/model:baked"
    assert args.dry_run is True
    assert args.yes is False
    assert args.func is handle_publish


def test_publish_cli_dry_run_uses_fake_docker_without_mutation(tmp_path, project_root):
    recipe_dir = tmp_path / "recipe"
    recipe_dir.mkdir()
    (recipe_dir / "recipe.yaml").write_text(
        """
model:
  huggingface: org/Model Name
engine:
  llm:
    vllm:
      image: cloudriftai/vllm-emmy-model-name:0.23.0-abcdef1
matrices:
  deploy.gpu: NVIDIA V100
  deploy.gpu_count: 8
"""
    )
    docker_log = tmp_path / "docker.log"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    metadata = json.dumps(
        [
            {
                "Id": f"sha256:{'d' * 64}",
                "Config": {
                    "Labels": {
                        "ai.emmy.publish.family": "vllm-emmy",
                        "ai.emmy.model.id": "org/Model Name",
                        "org.opencontainers.image.version": "0.23.0",
                        "org.opencontainers.image.revision": "abcdef1234567890abcdef1234567890abcdef12",
                    }
                },
                "RepoDigests": [],
            }
        ]
    )
    fake_docker.write_text(
        f"""#!/bin/sh
printf '%s\\n' "$*" >> "$FAKE_DOCKER_LOG"
if [ "$1 $2 $3" = "image inspect local/model:baked" ]; then
    printf '%s\\n' '{metadata}'
    exit 0
fi
if [ "$1 $2 $3" = "buildx imagetools inspect" ]; then
    echo 'manifest unknown' >&2
    exit 1
fi
exit 99
"""
    )
    fake_docker.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    env["FAKE_DOCKER_LOG"] = str(docker_log)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "emmy.emmy",
            "publish",
            str(recipe_dir),
            "--source-image",
            "local/model:baked",
            "--dry-run",
        ],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert f"Image ID:    sha256:{'d' * 64}" in result.stdout
    assert "Registry:    destination is absent" in result.stdout
    assert "DRY RUN: docker tag" in result.stdout
    assert "DRY RUN: docker push" in result.stdout
    docker_calls = docker_log.read_text()
    assert "image inspect local/model:baked" in docker_calls
    assert "buildx imagetools inspect" in docker_calls
    assert "tag local/model:baked" not in docker_calls
    assert "push cloudriftai" not in docker_calls
