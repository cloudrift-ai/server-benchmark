"""Canonical Docker image publication for serving recipes."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emmy.recipe import Recipe

PUBLISH_FAMILIES = ("vllm-emmy", "1cat-vllm")

_IMAGE_RE = re.compile(
    rf"^cloudriftai/(?P<family>{'|'.join(map(re.escape, PUBLISH_FAMILIES))})-(?P<slug>[a-z0-9][a-z0-9._-]*):"
    r"(?P<version>[0-9]+\.[0-9]+\.[0-9]+)-(?P<revision>[0-9a-f]{7,12})$"
)
_SLUG_JUNK_RE = re.compile(r"[^a-z0-9._-]+")
_HEX_REVISION_RE = re.compile(r"[0-9a-f]{7,40}")


class PublishError(ValueError):
    """A recipe image is not safe to publish under the canonical convention."""


@dataclass(frozen=True)
class PublishReference:
    """Parsed canonical publication reference."""

    image: str
    family: str
    model_slug: str
    runtime_version: str
    source_revision: str


@dataclass(frozen=True)
class PublishPlan:
    """Validated source and destination plus the mutating Docker commands."""

    source_image: str
    destination: PublishReference
    commands: tuple[tuple[str, ...], ...]
    already_published: bool = False
    source_image_id: str | None = None
    registry_digest: str | None = None


@dataclass(frozen=True)
class ImageMetadata:
    """Publication labels and registry digests carried by a local image."""

    labels: dict[str, str]
    repo_digests: frozenset[str]
    image_id: str


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def model_slug(model: str) -> str:
    """Return the Docker-safe model slug shared by release tooling and recipes."""
    name = model.rsplit("/", 1)[-1].lower()
    slug = _SLUG_JUNK_RE.sub("-", name).strip("._-")
    if not slug:
        raise PublishError(f"{model!r} sanitizes to an empty model slug")
    return slug


def parse_publish_reference(image: str, model: str) -> PublishReference:
    """Parse and validate a recipe's canonical publication destination."""
    match = _IMAGE_RE.fullmatch(image)
    if match is None:
        raise PublishError(
            "recipe image must match cloudriftai/(vllm-emmy|1cat-vllm)-<model-slug>:<runtime-version>-<7..12 hex source sha>"
        )

    expected_slug = model_slug(model)
    if match["slug"] != expected_slug:
        raise PublishError(f"recipe image slug {match['slug']!r} does not match model {model!r} (expected {expected_slug!r})")

    return PublishReference(
        image=image,
        family=match["family"],
        model_slug=match["slug"],
        runtime_version=match["version"],
        source_revision=match["revision"],
    )


def _inspect_image(source_image: str, runner: CommandRunner) -> ImageMetadata:
    result = runner(
        ["docker", "image", "inspect", source_image],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        inspected = json.loads(result.stdout)
        image = inspected[0]
        image_id = image["Id"]
        labels = image["Config"]["Labels"]
        repo_digests = image.get("RepoDigests") or []
    except (IndexError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise PublishError(f"docker inspect returned invalid metadata for {source_image!r}") from exc
    if not isinstance(labels, dict):
        raise PublishError(f"source image {source_image!r} has no Docker labels")
    if not isinstance(image_id, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None:
        raise PublishError(f"docker inspect returned an invalid image ID for {source_image!r}")
    if not isinstance(repo_digests, list):
        raise PublishError(f"docker inspect returned invalid RepoDigests for {source_image!r}")
    digests = frozenset(value.rsplit("@", 1)[-1] for value in repo_digests if isinstance(value, str) and "@" in value)
    return ImageMetadata(
        labels={str(key): str(value) for key, value in labels.items()},
        repo_digests=digests,
        image_id=image_id,
    )


def _remote_digest(destination: str, runner: CommandRunner) -> str | None:
    command = [
        "docker",
        "buildx",
        "imagetools",
        "inspect",
        destination,
        "--format",
        "{{json .Manifest.Digest}}",
    ]
    result = runner(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        error = (result.stderr or "").lower()
        if any(marker in error for marker in ("manifest unknown", "no such manifest")) or error.strip().endswith(": not found"):
            return None
        raise PublishError(f"could not inspect registry destination {destination!r}: {(result.stderr or '').strip()}")
    try:
        digest = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise PublishError(f"registry returned an invalid digest for {destination!r}") from exc
    if not isinstance(digest, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
        raise PublishError(f"registry returned an invalid digest for {destination!r}: {digest!r}")
    return digest


def _require_label(labels: dict[str, str], key: str, source_image: str) -> str:
    value = labels.get(key, "").strip()
    if not value:
        raise PublishError(f"source image {source_image!r} is missing required label {key!r}")
    return value


def validate_publish_labels(
    source_image: str,
    reference: PublishReference,
    model: str,
    labels: dict[str, str],
) -> None:
    """Require publication labels and compare them with the destination tag."""
    family = _require_label(labels, "ai.emmy.publish.family", source_image)
    version = _require_label(labels, "org.opencontainers.image.version", source_image)
    revision = _require_label(labels, "org.opencontainers.image.revision", source_image)
    model_id = _require_label(labels, "ai.emmy.model.id", source_image)

    if family != reference.family:
        raise PublishError(f"source image family label is {family!r}; destination requires {reference.family!r}")
    if version != reference.runtime_version:
        raise PublishError(f"source image version label is {version!r}; destination requires {reference.runtime_version!r}")
    if _HEX_REVISION_RE.fullmatch(revision) is None:
        raise PublishError("source image revision label must be a 7..40 character lowercase hexadecimal Git revision")
    if not revision.startswith(reference.source_revision):
        raise PublishError(f"source image revision label {revision!r} does not match destination revision {reference.source_revision!r}")
    if model_id != model:
        raise PublishError(f"source image model label is {model_id!r}; recipe requires {model!r}")


def load_publish_recipe(recipe_dir: str | Path) -> Recipe:
    """Load one concrete recipe, rejecting unresolved benchmark matrices."""
    from emmy.recipe import _load_raw_config, _validate_and_build, build_override, deep_merge, expand_matrix

    raw = _load_raw_config(str(recipe_dir))
    matrices = raw.pop("matrices", None)
    if matrices is None:
        return _validate_and_build(raw)
    combinations = expand_matrix(matrices)
    if len(combinations) != 1:
        raise PublishError(f"publish requires one concrete recipe; the matrix expands to {len(combinations)} variants")
    return _validate_and_build(deep_merge(raw, build_override(combinations[0])))


def build_publish_plan(recipe: Recipe, source_image: str | None = None) -> PublishPlan:
    """Build the immutable Docker command plan for one inference recipe."""
    if recipe.kind != "inference":
        raise PublishError("only inference recipes can publish serving images")
    if not recipe.model_name:
        raise PublishError("recipe model.huggingface must name the model being published")

    destination = parse_publish_reference(recipe.engine.llm.image, recipe.model_name)
    source = source_image or destination.image
    commands: list[tuple[str, ...]] = []
    if source != destination.image:
        commands.append(("docker", "tag", source, destination.image))
    commands.append(("docker", "push", destination.image))
    return PublishPlan(source_image=source, destination=destination, commands=tuple(commands))


def publish_recipe_image(
    recipe: Recipe,
    *,
    source_image: str | None = None,
    dry_run: bool = False,
    yes: bool = False,
    runner: CommandRunner = subprocess.run,
) -> PublishPlan:
    """Validate, optionally retag, and push the serving image named by a recipe."""
    plan = build_publish_plan(recipe, source_image)
    if not dry_run and not yes:
        raise PublishError("refusing to push without --yes; use --dry-run to validate and preview")

    source_metadata = _inspect_image(plan.source_image, runner)
    validate_publish_labels(plan.source_image, plan.destination, recipe.model_name, source_metadata.labels)

    existing_digest = _remote_digest(plan.destination.image, runner)
    plan = replace(plan, source_image_id=source_metadata.image_id, registry_digest=existing_digest)
    if existing_digest is not None:
        if existing_digest not in source_metadata.repo_digests:
            known = ", ".join(sorted(source_metadata.repo_digests)) or "none"
            raise PublishError(
                f"destination already exists at {existing_digest}, but source RepoDigests are {known}; refusing to overwrite"
            )
        return replace(plan, commands=(), already_published=True)

    if dry_run:
        return plan

    for command in plan.commands:
        runner(list(command), check=True)

    published_metadata = _inspect_image(plan.destination.image, runner)
    published_digest = _remote_digest(plan.destination.image, runner)
    if published_digest is None or published_digest not in published_metadata.repo_digests:
        known = ", ".join(sorted(published_metadata.repo_digests)) or "none"
        raise PublishError(f"post-push registry digest {published_digest or 'missing'} does not match local RepoDigests {known}")
    return replace(plan, registry_digest=published_digest)


def _model_slug_main(argv: Sequence[str]) -> int:
    """Small stable entry point used by the shell release tooling."""
    if len(argv) != 1:
        return 2
    try:
        slug = model_slug(argv[0])
    except PublishError:
        return 1
    sys.stdout.write(f"{slug}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_model_slug_main(sys.argv[1:]))
