"""Compatibility contract for self-contained golden YAML."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from emmy.compiler.pipeline.search.golden import (
    GoldenEntryState,
    GoldenFileValidation,
    _LazyGoldenRecords,
    golden_entry_state,
    load_golden_file,
    load_golden_records,
    validate_golden_file,
)

_GOLDENS = Path(__file__).parents[2] / "emmy/compiler/pipeline/search/goldens"
_FILES = sorted(_GOLDENS.rglob("*.yaml"))
_FORBIDDEN_ENTRY_FIELDS = {
    "cublas_us",
    "emmy_us",
    "kernel",
    "provisional",
    "reproducer",
    "shape_key",
    "structural_features",
}


@pytest.mark.parametrize("path", _FILES, ids=lambda path: path.name)
def test_repository_golden_deserializes(path: Path) -> None:
    document = load_golden_file(path, validation=GoldenFileValidation.REPOSITORY)
    records = load_golden_records(document)
    assert records
    assert "format_version" not in document
    assert isinstance(document["programs"], list)
    assert all("ir_version" not in program for program in document["programs"])
    for entry in document["configs"]:
        assert not (_FORBIDDEN_ENTRY_FIELDS & set(entry)), path
        assert set(entry["target"]) in ({"origins"}, {"loop"}), path
        assert entry["realizations"]
        assert all("pins" in realization for realization in entry["realizations"]), path
    for record in records:
        assert record.program.nodes
        assert bool(record.origins) != (record.loop_wire is not None)
        if record.loop_wire is not None:
            assert record.target_program.nodes
            assert record.target_program.outputs
        else:
            assert set(record.origins) <= set(record.program.nodes)


def test_production_tree_has_no_retired_golden_surfaces() -> None:
    root = Path(__file__).parents[2]
    forbidden = ("GoldenConfig", "GOLDEN_CONFIGS", "_KERNEL_CLASSES", "golden_sidecar_dir", "golden_v2", ".torch.json")
    paths = [*sorted((root / "emmy").rglob("*.py")), *sorted((root / "scripts").glob("*.py")), *sorted((root / "scripts").glob("*.sh"))]
    for path in paths:
        text = path.read_text()
        assert not [token for token in forbidden if token in text], path


def test_record_derived_measurement_properties() -> None:
    document = load_golden_file(_FILES[0], validation=GoldenFileValidation.REPOSITORY)
    record = load_golden_records(document)[0]
    assert record.emmy_us > 0
    assert record.reference_us > 0
    assert record.ratio == record.reference_us / record.emmy_us
    assert record.dynamic == record.shape_key.is_dyn


def test_repository_index_loads_once_on_first_access() -> None:
    calls = 0

    def load() -> list:
        nonlocal calls
        calls += 1
        return ["record"]

    records = _LazyGoldenRecords(load)
    assert calls == 0
    assert len(records) == 1
    assert records[0] == "record"
    assert list(records) == ["record"]
    assert calls == 1


def test_legacy_dynamic_attention_record_stays_flash_keyed() -> None:
    """Gate-free fusion moves exp histograms off the P×V consumer, but recognition still
    certifies the unit; old bare-TILE dynamic records must retain their stable flash key."""
    document = load_golden_file(_GOLDENS / "rtx5090_sm120.yaml", validation=GoldenFileValidation.REPOSITORY)
    record = next(r for r in load_golden_records(document) if r.name == "attention.hd64.dynM" and "TILE" in r.knobs)
    assert not any(key.startswith("TILE@") for key in record.knobs)
    assert record.shape_key.kind == "flash"


def test_working_states_and_repository_requires_measurements() -> None:
    document = load_golden_file(_FILES[0], validation=GoldenFileValidation.REPOSITORY)
    inventory = copy.deepcopy(document)
    inventory["configs"] = [copy.deepcopy(inventory["configs"][0])]
    inventory["configs"][0]["realizations"] = [copy.deepcopy(inventory["configs"][0]["realizations"][0])]
    entry = inventory["configs"][0]["realizations"][0]
    entry.pop("knobs")
    entry.pop("measurements")
    assert golden_entry_state(entry) == GoldenEntryState.INVENTORY
    validate_golden_file(inventory)
    with pytest.raises(ValueError, match="requires knobs and paired positive timings"):
        validate_golden_file(inventory, validation=GoldenFileValidation.REPOSITORY)

    entry["knobs"] = {}
    assert golden_entry_state(entry) == GoldenEntryState.PROPOSAL
    validate_golden_file(inventory)


def test_realization_pins_are_registered_and_typed() -> None:
    document = copy.deepcopy(load_golden_file(_FILES[0], validation=GoldenFileValidation.REPOSITORY))
    document["configs"] = [copy.deepcopy(document["configs"][0])]
    document["configs"][0]["realizations"] = [copy.deepcopy(document["configs"][0]["realizations"][0])]
    realization = document["configs"][0]["realizations"][0]

    realization["pins"] = {"FAST_MATH": False, "F16_MMA_F32_ACC": True}
    validate_golden_file(document)
    record = load_golden_records(document)[0]
    assert record.pin_map == {"F16_MMA_F32_ACC": True, "FAST_MATH": False}

    realization["pins"] = {"NOT_A_KNOB": False}
    with pytest.raises(ValueError, match="names unknown knob 'NOT_A_KNOB'"):
        validate_golden_file(document)

    realization["pins"] = {"FAST_MATH": "false"}
    with pytest.raises(ValueError, match=r"pins\.FAST_MATH must be a bool value"):
        validate_golden_file(document)

    realization["pins"] = {"WORK": "w1x1"}
    realization["knobs"] = {"WORK": "w2x2"}
    with pytest.raises(ValueError, match="conflicting input pins and measured knobs for WORK"):
        validate_golden_file(document)


def test_promotion_rejects_mixed_bare_and_axis_scoped_schedule_keys() -> None:
    document = copy.deepcopy(load_golden_file(_FILES[0], validation=GoldenFileValidation.REPOSITORY))
    document["configs"] = [copy.deepcopy(document["configs"][0])]
    document["configs"][0]["realizations"] = [copy.deepcopy(document["configs"][0]["realizations"][0])]
    document["configs"][0]["realizations"][0]["knobs"] = {"REDUCE": "", "REDUCE@a1": "coop"}

    validate_golden_file(document)
    with pytest.raises(ValueError, match="mixes bare and axis-scoped REDUCE"):
        validate_golden_file(document, validation=GoldenFileValidation.PROMOTION)


def test_program_index_must_resolve_in_document() -> None:
    document = copy.deepcopy(load_golden_file(_FILES[0], validation=GoldenFileValidation.REPOSITORY))
    document["configs"][0]["program"] = len(document["programs"])
    with pytest.raises(ValueError, match="does not resolve"):
        validate_golden_file(document)
