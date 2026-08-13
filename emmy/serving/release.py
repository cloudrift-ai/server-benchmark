"""Pinned serving configuration and its compiler realization matrix."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path

from emmy.compiler.trace.dynamic import DYNAMIC_DIM_MAX
from emmy.publish import model_slug

_KEY = re.compile(r"[A-Z][A-Z0-9_]*")
_HEX_REVISION = re.compile(r"[0-9a-f]{7,40}")


@dataclass(frozen=True)
class ServingRealization:
    """One symbolic or statically bound compiler lane used by serving."""

    name: str
    bindings: tuple[tuple[str, int], ...]
    pins: tuple[tuple[str, object], ...]

    def to_golden(self) -> dict:
        return {"name": self.name, "bindings": dict(self.bindings), "pins": dict(self.pins)}


@dataclass(frozen=True)
class ServingConfig:
    """The release identity and realized compiler matrix from one pinned env file."""

    source: Path
    model: str
    revision: str | None
    gpu_name: str
    golden_file: Path
    realizations: tuple[ServingRealization, ...]
    static_only: bool
    #: Checked-in per-twin golden-consultation counts (JSON) the release audit ratchets
    #: against; ``None`` (no ``SERVE_CONSULT_BASELINE``) skips that check.
    consult_baseline: Path | None = None

    @property
    def model_provenance(self) -> str:
        return f"{self.model}@{self.revision}" if self.revision else self.model

    @property
    def static_widths(self) -> tuple[int, ...]:
        return tuple(sorted({dict(row.bindings)["num_tokens"] for row in self.realizations if row.bindings}))


def _read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{line_number}: expected KEY=value")
        key, encoded = line.split("=", 1)
        if _KEY.fullmatch(key) is None:
            raise ValueError(f"{path}:{line_number}: invalid field name {key!r}")
        if key in values:
            raise ValueError(f"{path}:{line_number}: duplicate field {key}")
        try:
            parsed = shlex.split(encoded, posix=True)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid quoted value for {key}: {exc}") from exc
        if encoded == "":
            value = ""
        elif len(parsed) == 1:
            value = parsed[0]
        else:
            raise ValueError(f"{path}:{line_number}: {key} must be one shell-free value (quote values containing spaces)")
        if "$" in value or "`" in value:
            raise ValueError(f"{path}:{line_number}: {key} cannot contain shell expansion")
        values[key] = value
    return values


def _integer(values: dict[str, str], field: str, *, default: int | None = None, minimum: int = 0) -> int:
    raw = values.get(field)
    if raw in (None, ""):
        if default is None:
            raise ValueError(f"missing required {field}")
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{field} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{field} must be >= {minimum}, got {value}")
    return value


def _root(source: Path) -> Path:
    for parent in (source.parent, *source.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def _lane_realizations(
    *,
    decode: int,
    prefill: int,
    max_tokens: int,
    m1: bool,
    pins: tuple[tuple[str, object], ...],
    static_only: bool,
) -> list[ServingRealization]:
    suffix = ".fm" if dict(pins).get("FAST_MATH") else ""
    widths = set()
    if decode:
        widths.add(decode)
    if m1 and decode > 1:
        widths.add(1)
    if prefill > decode:
        widths.add(prefill)
    rows = [
        ServingRealization(
            name=f"m{width}{suffix}",
            bindings=(("num_tokens", width),),
            pins=pins,
        )
        for width in sorted(widths)
    ]
    if max_tokens and not static_only:
        rows.append(
            ServingRealization(
                name=f"dynamic{suffix}",
                bindings=(),
                pins=pins,
            )
        )
    return rows


def _realizations(values: dict[str, str], source: Path, *, static_only: bool) -> tuple[ServingRealization, ...]:
    pinned_decode = _integer(values, "SERVE_DECODE_BUCKET", minimum=0)
    max_tokens = _integer(values, "SERVE_MAX_NUM_BATCHED_TOKENS", minimum=1)
    capacity = _integer(values, "SERVE_PREFILL_CAPACITY", default=DYNAMIC_DIM_MAX, minimum=1)
    pinned_prefill = _integer(values, "SERVE_PREFILL_BUCKET", default=capacity, minimum=0)
    m1_value = _integer(values, "SERVE_M1_TIER", default=1, minimum=0)
    if m1_value not in {0, 1}:
        raise ValueError(f"{source}: SERVE_M1_TIER must be 0 or 1")
    m1 = bool(m1_value)
    lanes = [(pinned_decode, pinned_prefill, max_tokens, (("FAST_MATH", False),))]
    for spec in values.get("SERVE_WARM_SHAPES", "").split():
        fields = spec.split(":")
        if len(fields) not in {3, 4}:
            raise ValueError(f"{source}: invalid SERVE_WARM_SHAPES entry {spec!r}")
        decode = int(fields[0]) if fields[0] else pinned_decode
        prefill = int(fields[1]) if fields[1] else pinned_prefill
        maximum = int(fields[2]) if fields[2] else max_tokens
        lane = fields[3] if len(fields) == 4 else ""
        if lane not in {"", "fm"}:
            raise ValueError(f"{source}: warm shape {spec!r} has unsupported lane {lane!r}")
        if min(decode, prefill) < 0 or maximum <= 0:
            raise ValueError(f"{source}: warm shape {spec!r} requires non-negative buckets and positive max tokens")
        lanes.append((decode, prefill, maximum, (("FAST_MATH", lane == "fm"),)))

    standard_m1 = (1, 0, 1, (("FAST_MATH", False),))
    if static_only and any(lane != standard_m1 for lane in lanes):
        raise ValueError(f"{source}: static-only release warm shapes must stay in the standard M=1 envelope")

    rows = {
        row
        for decode, prefill, maximum, pins in lanes
        for row in _lane_realizations(
            decode=decode,
            prefill=prefill,
            max_tokens=maximum,
            m1=m1,
            pins=pins,
            static_only=static_only,
        )
    }
    return tuple(sorted(rows, key=lambda row: (row.pins, not row.bindings, row.bindings, row.name)))


def load_serving_config(path: str | Path) -> ServingConfig:
    """Parse a pinned release env without evaluating it as shell code."""
    source = Path(path).resolve()
    values = _read_env(source)
    model = values.get("SERVE_MODEL", "").strip()
    gpu_name = values.get("SERVE_GPU", "").strip()
    golden = values.get("SERVE_GOLDEN_FILE", "").strip()
    if not model or not gpu_name or not golden:
        raise ValueError(f"{source}: SERVE_MODEL, SERVE_GPU, and SERVE_GOLDEN_FILE are required")
    revision = values.get("SERVE_REVISION", "").strip() or None
    static_only = values.get("SERVE_STATIC_ONLY", "0") == "1"
    if values.get("SERVE_STATIC_ONLY", "0") not in {"0", "1"}:
        raise ValueError(f"{source}: SERVE_STATIC_ONLY must be 0 or 1")
    if static_only:
        required = {
            "SERVE_MAX_NUM_BATCHED_TOKENS": "1",
            "SERVE_DECODE_BUCKET": "1",
            "SERVE_PREFILL_CAPACITY": "1",
            "SERVE_PREFILL_BUCKET": "0",
            "SERVE_M1_TIER": "1",
            "SERVE_CAPTURE_SIZES": "[1]",
        }
        wrong = {key: values.get(key) for key, expected in required.items() if values.get(key) != expected}
        if wrong:
            raise ValueError(f"{source}: unsafe static-only serving envelope: {wrong!r}")
    golden_path = Path(golden)
    if not golden_path.is_absolute():
        golden_path = _root(source) / golden_path
    consult_baseline = None
    consult = values.get("SERVE_CONSULT_BASELINE", "").strip()
    if consult:
        consult_baseline = Path(consult)
        if not consult_baseline.is_absolute():
            consult_baseline = _root(source) / consult_baseline
        consult_baseline = consult_baseline.resolve()
    realizations = _realizations(values, source, static_only=static_only)
    if static_only and realizations != (ServingRealization(name="m1", bindings=(("num_tokens", 1),), pins=(("FAST_MATH", False),)),):
        raise ValueError(f"{source}: static-only release warm shapes must stay in the standard M=1 lane")
    return ServingConfig(
        source=source,
        model=model,
        revision=revision,
        gpu_name=gpu_name,
        golden_file=golden_path.resolve(),
        realizations=realizations,
        static_only=static_only,
        consult_baseline=consult_baseline,
    )


def revision_matches(golden_revision: str | None, serving_revision: str | None) -> bool:
    if golden_revision is None:
        return True
    if serving_revision is None:
        return False
    if golden_revision == serving_revision:
        return True
    return bool(
        _HEX_REVISION.fullmatch(golden_revision)
        and _HEX_REVISION.fullmatch(serving_revision)
        and (golden_revision.startswith(serving_revision) or serving_revision.startswith(golden_revision))
    )


def split_revision(model: str) -> tuple[str, str | None]:
    repo, separator, revision = model.rpartition("@")
    return (repo, revision or None) if separator and repo else (model, None)


def model_matches(golden_model: str | None, config: ServingConfig) -> bool:
    """Match a base checkpoint to an instruction-tuned sibling, plus revision."""
    if not golden_model:
        return False
    repo, revision = split_revision(golden_model)
    golden_slug = model_slug(repo)
    serving_slug = model_slug(config.model)
    return (golden_slug == serving_slug or serving_slug.startswith(f"{golden_slug}-")) and revision_matches(revision, config.revision)


__all__ = ["ServingConfig", "ServingRealization", "load_serving_config", "model_matches", "revision_matches", "split_revision"]
