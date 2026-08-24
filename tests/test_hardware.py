"""Unit tests for hardware lookup tables."""

from pathlib import Path

import yaml

from emmy.hardware import GPU_INSTANCE_TYPES, GPU_SHORT_NAMES, gpu_short_name, resolve_instance_type
from emmy.recipe.catalog import deployment_setups

# ── resolve_instance_type ────────────────────────────────────────


def test_resolve_cloudrift_single_gpu():
    assert resolve_instance_type("cloudrift", "rtx49-10c-kn", 1) == "rtx49-10c-kn.1"


def test_resolve_cloudrift_multi_gpu():
    assert resolve_instance_type("cloudrift", "rtx49-10c-kn", 4) == "rtx49-10c-kn.4"


def test_resolve_gcp_standard():
    assert resolve_instance_type("gcp", "a3-highgpu", 8) == "a3-highgpu-8g"


def test_resolve_gcp_g4_standard():
    assert resolve_instance_type("gcp", "g4-standard", 4) == "g4-standard-192"


def test_resolve_gcp_g4_standard_single():
    assert resolve_instance_type("gcp", "g4-standard", 1) == "g4-standard-48"


# ── GPU_INSTANCE_TYPES table ────────────────────────────────────


def test_all_gpus_have_at_least_one_provider():
    for gpu_name, entries in GPU_INSTANCE_TYPES.items():
        assert len(entries) >= 1, f"GPU '{gpu_name}' has no provider entries"


def test_known_gpu_lookup():
    entries = GPU_INSTANCE_TYPES["NVIDIA GeForce RTX 4090"]
    assert entries[0][0] == "cloudrift"
    assert entries[0][1] == "rtx49-10c-kn"


def test_gcp_gpu_lookup():
    entries = GPU_INSTANCE_TYPES["NVIDIA H100 80GB"]
    assert entries[0] == ("gcp", "a3-highgpu")


# ── gpu_short_name ────────────────────────────────────────────────


def test_gpu_short_name_known():
    assert gpu_short_name("NVIDIA GeForce RTX 5090") == "rtx5090"
    assert gpu_short_name("NVIDIA H200 141GB") == "h200"
    assert gpu_short_name("NVIDIA RTX PRO 6000 Blackwell Server Edition") == "pro6000"


def test_gpu_short_name_unknown_fallback():
    result = gpu_short_name("Some Unknown GPU 9000")
    assert result == "someunknowngpu9000"


def test_gpu_short_names_covers_all_instance_types():
    """Every GPU in GPU_INSTANCE_TYPES has a short name."""
    for gpu_name in GPU_INSTANCE_TYPES:
        assert gpu_name in GPU_SHORT_NAMES, f"GPU '{gpu_name}' missing from GPU_SHORT_NAMES"


def test_every_declared_recipe_deployment_is_provisionable():
    """A recipe deployment naming a GPU absent from the table can never be rented or selected."""
    recipes = Path(__file__).parents[1] / "recipes"
    for recipe_path in sorted(recipes.glob("*/recipe.yaml")):
        config = yaml.safe_load(recipe_path.read_text()) or {}
        for setup in deployment_setups(config):
            gpu_name = setup["deploy.gpu"]
            assert gpu_name in GPU_INSTANCE_TYPES, f"{recipe_path} declares '{gpu_name}', missing from GPU_INSTANCE_TYPES"
