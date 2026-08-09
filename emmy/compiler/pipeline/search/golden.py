"""Repository index for self-contained, program-backed golden records."""

from __future__ import annotations

from emmy.compiler.pipeline.search.golden_v2 import (
    _GOLDENS_DIR,
    GoldenEntryState,
    GoldenFileValidation,
    GoldenRecord,
    dump_golden_file,
    fast_math_knobs,
    golden_entry_state,
    golden_record_from_entry,
    is_repository_golden_path,
    load_golden_file,
    load_golden_records,
    validate_golden_file,
)


def _load_goldens() -> list[GoldenRecord]:
    records: list[GoldenRecord] = []
    for path in sorted(_GOLDENS_DIR.rglob("*.yaml")):
        document = load_golden_file(path, validation=GoldenFileValidation.REPOSITORY)
        records.extend(load_golden_records(document))
    return records


GOLDEN_RECORDS: list[GoldenRecord] = _load_goldens()


def goldens_by_name(name: str) -> list[GoldenRecord]:
    """Every record with an exact name; names need not be unique."""
    return [record for record in GOLDEN_RECORDS if record.name == name]


def goldens_for_live_gpu() -> list[GoldenRecord]:
    """Goldens for the live card, or all records when no card is visible."""
    live = live_recorded_goldens()
    return list(GOLDEN_RECORDS) if live is None else (live or list(GOLDEN_RECORDS))


def live_recorded_goldens() -> list[GoldenRecord] | None:
    """The live card's own records, ``None`` when no CUDA card is visible."""
    key = _live_gpu_key()
    if key is None:
        return None
    return [record for record in GOLDEN_RECORDS if record.gpu_name == key[0] and record.compute_cap == key[1]]


def _live_gpu_key() -> tuple[str, tuple[int, int]] | None:
    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            return None
        name = torch.cuda.get_device_name(0)
        from emmy.gpu import by_name  # noqa: PLC0415

        gpu = by_name(name)
        return (gpu.name if gpu is not None else name), tuple(torch.cuda.get_device_capability(0))
    except Exception:  # noqa: BLE001
        return None


__all__ = [
    "GOLDEN_RECORDS",
    "GoldenEntryState",
    "GoldenFileValidation",
    "GoldenRecord",
    "dump_golden_file",
    "fast_math_knobs",
    "golden_entry_state",
    "golden_record_from_entry",
    "goldens_by_name",
    "goldens_for_live_gpu",
    "is_repository_golden_path",
    "live_recorded_goldens",
    "load_golden_file",
    "load_golden_records",
    "validate_golden_file",
]
