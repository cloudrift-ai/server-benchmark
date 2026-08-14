"""Recipe inventory and onboarding-stub creation."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from emmy import gpu as gpu_registry
from emmy.recipe.lifecycle import ONBOARDING_TAG, UNTESTED_TAG, validate_recipe_tags
from emmy.recipe.matrix import build_override, expand_matrix
from emmy.recipe.recipe import deep_merge

HF_ID = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
DEPLOYMENT_FIELDS = frozenset({"deploy.gpu", "deploy.gpu_count"})
MAX_STUB_DEPLOYMENTS = 3


def recipe_catalog(root: str | Path) -> dict[str, dict]:
    """Load model recipes below root, keyed by Hugging Face model ID."""
    root = Path(root)
    records = {}
    for path in sorted(root.glob("*/recipe.yaml")):
        config = yaml.safe_load(path.read_text()) or {}
        if not isinstance(config, dict):
            raise ValueError(f"Recipe must contain a YAML object: {path}")
        tags = validate_recipe_tags(config.get("tags"))
        model_id = (config.get("model") or {}).get("huggingface")
        if not model_id:
            continue
        if not isinstance(model_id, str) or not HF_ID.fullmatch(model_id):
            raise ValueError(f"Invalid Hugging Face model ID in {path}: {model_id!r}")
        if model_id in records:
            raise ValueError(f"Multiple recipes use Hugging Face model ID {model_id}")
        records[model_id] = {"path": path, "config": config, "tags": tags}
    return records


def deployment_setups(config: dict) -> list[dict[str, object]]:
    """Return the unique GPU/count combinations produced by one recipe matrix."""
    base = {key: value for key, value in config.items() if key != "matrices"}
    matrices = config.get("matrices")
    variants = [base]
    if matrices:
        variants = [deep_merge(base, build_override(combination)) for combination in expand_matrix(matrices)]

    setups = []
    seen = set()
    for variant in variants:
        deploy = variant.get("deploy") or {}
        gpu_name = deploy.get("gpu")
        gpu_count = deploy.get("gpu_count", 1)
        key = (gpu_name, gpu_count)
        valid_count = isinstance(gpu_count, int) and not isinstance(gpu_count, bool) and gpu_count >= 1
        if not isinstance(gpu_name, str) or not valid_count or key in seen:
            continue
        seen.add(key)
        setups.append({"deploy.gpu": gpu_name, "deploy.gpu_count": gpu_count})
    return setups


def recipe_inventory(root: str | Path, tags: tuple[str, ...] = ()) -> list[dict]:
    """Return compact, JSON-ready metadata for recipes carrying every requested tag."""
    records = recipe_catalog(root)
    inventory = []
    for model_id, record in records.items():
        if not set(tags).issubset(record["tags"]):
            continue
        config = record["config"]
        model = config.get("model") or {}
        try:
            display_path = record["path"].relative_to(Path.cwd())
        except ValueError:
            display_path = record["path"]
        inventory.append(
            {
                "path": str(display_path),
                "model_id": model_id,
                "tags": list(record["tags"]),
                "task": model.get("task", "generate"),
                "deployments": deployment_setups(config),
                "rationale": model.get("rationale"),
            }
        )
    return inventory


def validate_stub_deployments(value: object, model_id: str) -> list[dict[str, object]]:
    """Validate the one-to-three native matrix entries allowed in a recipe stub."""
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_STUB_DEPLOYMENTS:
        raise ValueError(f"Onboarding model {model_id} needs one to {MAX_STUB_DEPLOYMENTS} deployments")
    deployments = []
    seen = set()
    for deployment in value:
        if not isinstance(deployment, dict) or set(deployment) != DEPLOYMENT_FIELDS:
            raise ValueError(f"Each deployment must contain exactly: {', '.join(sorted(DEPLOYMENT_FIELDS))}")
        gpu_name = deployment["deploy.gpu"]
        gpu_count = deployment["deploy.gpu_count"]
        spec = gpu_registry.by_name(gpu_name) if isinstance(gpu_name, str) else None
        if spec is None or spec.name != gpu_name:
            raise ValueError(f"Onboarding model {model_id} selected unknown GPU {gpu_name!r}")
        if not isinstance(gpu_count, int) or isinstance(gpu_count, bool) or gpu_count < 1:
            raise ValueError(f"Onboarding model {model_id} needs a positive deploy.gpu_count")
        key = (gpu_name, gpu_count)
        if key in seen:
            raise ValueError(f"Onboarding model {model_id} contains a duplicate deployment")
        seen.add(key)
        deployments.append(deployment)
    return deployments


def create_recipe_stub(
    root: str | Path,
    model_id: str,
    rationale: str,
    task: str,
    deployments: list[dict[str, object]],
) -> Path:
    """Create an onboarding/untested recipe stub and return its recipe path."""
    root = Path(root)
    if not isinstance(model_id, str) or not HF_ID.fullmatch(model_id):
        raise ValueError(f"Invalid onboarding Hugging Face model ID: {model_id!r}")
    if model_id in recipe_catalog(root):
        raise ValueError(f"A recipe already exists for onboarding model {model_id}")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError(f"Onboarding model {model_id} needs a rationale")
    if task not in ("generate", "embed"):
        raise ValueError(f"Invalid task for {model_id}: {task!r}")
    deployments = validate_stub_deployments(deployments, model_id)

    organization, name = model_id.split("/", 1)
    name_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")
    directory = root / name_slug
    if directory.exists():
        organization_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", organization).strip("-")
        directory = root / f"{organization_slug}--{name_slug}"
    if directory.exists():
        raise ValueError(f"Cannot choose a unique recipe directory for {model_id}")

    directory.mkdir(parents=True)
    recipe = directory / "recipe.yaml"
    config = {
        "tags": [ONBOARDING_TAG, UNTESTED_TAG],
        "model": {"huggingface": model_id, "rationale": rationale.strip(), "task": task},
        "matrices": deployments,
    }
    recipe.write_text(yaml.safe_dump(config, sort_keys=False, width=116))
    return recipe
