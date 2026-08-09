"""Compatibility contract for self-contained golden YAML."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from emmy.compiler.pipeline.search.golden import (
    GOLDEN_RECORDS,
    GoldenEntryState,
    GoldenFileValidation,
    golden_entry_state,
    load_golden_file,
    load_golden_records,
    validate_golden_file,
)

_GOLDENS = Path(__file__).parents[2] / "emmy/compiler/pipeline/search/goldens"
_FILES = sorted(_GOLDENS.rglob("*.yaml"))


@pytest.mark.parametrize("path", _FILES, ids=lambda path: path.name)
def test_repository_golden_deserializes(path: Path) -> None:
    document = load_golden_file(path, validation=GoldenFileValidation.REPOSITORY)
    records = load_golden_records(document)
    assert records
    assert "format_version" not in document
    assert isinstance(document["programs"], list)
    assert all("ir_version" not in program for program in document["programs"])
    for record in records:
        assert record.program.nodes
        assert record.origins
        assert set(record.origins) <= set(record.program.nodes)


def test_repository_index_is_the_flat_corpus() -> None:
    count = sum(len(load_golden_file(path, validation=GoldenFileValidation.REPOSITORY)["configs"]) for path in _FILES)
    assert len(GOLDEN_RECORDS) == count
    assert count > 0


def test_repository_format_has_no_legacy_or_derived_target_fields() -> None:
    forbidden = {"kernel", "reproducer", "emmy_us", "cublas_us", "provisional", "shape_key", "structural_features"}
    for path in _FILES:
        document = load_golden_file(path, validation=GoldenFileValidation.REPOSITORY)
        for entry in document["configs"]:
            assert not (forbidden & set(entry)), path
            assert set(entry["target"]) == {"origins"}, path


def test_production_tree_has_no_retired_golden_surfaces() -> None:
    root = Path(__file__).parents[2]
    forbidden = ("GoldenConfig", "GOLDEN_CONFIGS", "_KERNEL_CLASSES", "golden_sidecar_dir", "golden_v2", ".torch.json")
    paths = [*sorted((root / "emmy").rglob("*.py")), *sorted((root / "scripts").glob("*.py")), *sorted((root / "scripts").glob("*.sh"))]
    for path in paths:
        text = path.read_text()
        assert not [token for token in forbidden if token in text], path


def test_record_derived_measurement_properties() -> None:
    record = GOLDEN_RECORDS[0]
    assert record.emmy_us > 0
    assert record.reference_us > 0
    assert record.ratio == record.reference_us / record.emmy_us
    assert record.dynamic == record.shape_key.is_dyn


def test_working_states_and_repository_requires_measurements() -> None:
    document = load_golden_file(_FILES[0], validation=GoldenFileValidation.REPOSITORY)
    inventory = copy.deepcopy(document)
    inventory["configs"] = [copy.deepcopy(inventory["configs"][0])]
    entry = inventory["configs"][0]
    entry.pop("knobs")
    entry.pop("measurements")
    assert golden_entry_state(entry) == GoldenEntryState.INVENTORY
    validate_golden_file(inventory)
    with pytest.raises(ValueError, match="requires knobs and paired positive timings"):
        validate_golden_file(inventory, validation=GoldenFileValidation.REPOSITORY)

    entry["knobs"] = {}
    assert golden_entry_state(entry) == GoldenEntryState.PROPOSAL
    validate_golden_file(inventory)


def test_program_index_must_resolve_in_document() -> None:
    document = copy.deepcopy(load_golden_file(_FILES[0], validation=GoldenFileValidation.REPOSITORY))
    document["configs"][0]["program"] = len(document["programs"])
    with pytest.raises(ValueError, match="does not resolve"):
        validate_golden_file(document)
