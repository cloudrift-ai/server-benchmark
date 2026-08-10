"""Self-contained golden YAML records and repository index.

Golden YAML is a persistence format, not a Python class discriminator.  Every
record points at a stable Torch IR program in the same document and carries the
target identity needed by search consumers directly as data.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from numbers import Real
from pathlib import Path

import yaml

from emmy.compiler.loop_wire import loop_graph_from_wire, validate_loop_program_pool
from emmy.compiler.pipeline.search.data.shape import ShapeKey
from emmy.compiler.torch_wire import graph_from_wire, validate_program_pool

_GOLDENS_DIR = Path(__file__).parent / "goldens"
_PROGRAM_GRAPH_CACHE: dict[int, tuple[dict, object]] = {}
_LOOP_GRAPH_CACHE: dict[int, tuple[dict, object]] = {}


class _FlowSequence(list):
    """A YAML sequence rendered inline without changing the loaded schema."""


class _FlowMapping(dict):
    """A YAML mapping rendered inline without changing the loaded schema."""


class _GoldenDumper(yaml.SafeDumper):
    pass


_GoldenDumper.add_representer(
    _FlowSequence,
    lambda dumper, value: dumper.represent_sequence("tag:yaml.org,2002:seq", value, flow_style=True),
)
_GoldenDumper.add_representer(
    _FlowMapping,
    lambda dumper, value: dumper.represent_mapping("tag:yaml.org,2002:map", value, flow_style=True),
)


def _flow(value):
    if isinstance(value, list):
        return _FlowSequence(_flow(item) for item in value)
    if isinstance(value, Mapping):
        return _FlowMapping((key, _flow(item)) for key, item in value.items())
    return value


def _short_flow(value: object) -> bool:
    if isinstance(value, Mapping):
        if "__program__" in value:
            return False
        return len(repr(value)) <= 120 and all(_short_flow(item) for item in value.values())
    if isinstance(value, list):
        return len(repr(value)) <= 120 and all(_short_flow(item) for item in value)
    return value is None or isinstance(value, (str, int, float, bool))


def _style_wire_value(value):
    if isinstance(value, Mapping):
        if set(value) == {"__program__"}:
            return {"__program__": _style_program(value["__program__"])}
        styled = {key: _style_wire_value(item) for key, item in value.items()}
        return _flow(styled) if _short_flow(value) else styled
    if isinstance(value, list):
        return [_style_wire_value(item) for item in value]
    return value


def _style_program(program: Mapping) -> dict:
    styled_nodes = []
    for source in program["nodes"]:
        node = dict(source)
        if "attrs" in node:
            node["attrs"] = _style_wire_value(node["attrs"])
        if "inputs" in node:
            node["inputs"] = _flow(node["inputs"])
        node["outputs"] = _flow(node["outputs"])
        styled_nodes.append(node)
    styled = {
        "inputs": _flow(program["inputs"]),
        "outputs": _flow(program["outputs"]),
        "nodes": styled_nodes,
    }
    if "hints" in program:
        styled["hints"] = _flow(program["hints"])
    return styled


class GoldenEntryState(StrEnum):
    INVENTORY = "inventory"
    PROPOSAL = "proposal"
    VERIFIED = "verified"


class GoldenFileValidation(StrEnum):
    WORKING = "working"
    PROMOTION = "promotion"
    REPOSITORY = "repository"


def fast_math_knobs(knobs: Mapping) -> bool:
    """Whether recorded knobs select a precision-trading realization."""
    from emmy.compiler.ir.schedule import TilePlan, Workers  # noqa: PLC0415
    from emmy.compiler.pipeline.search.space import FAST_EXP  # noqa: PLC0415

    for key, value in knobs.items():
        spelling = str(value).strip()
        if str(key).split("@", 1)[0] == "TILE" and spelling:
            try:
                plan = TilePlan.parse(spelling, Workers(kind="warp", units=(1, 1)))
            except ValueError:
                plan = None
            if plan is not None and plan.is_warp and plan.atom.operand_dtype("c").nbytes == 2:
                return True
        if key == FAST_EXP.name and spelling.casefold() in {"true", "1", "yes", "on"}:
            return True
    return False


def golden_entry_state(entry: Mapping) -> GoldenEntryState:
    has_knobs = "knobs" in entry
    has_measurements = "measurements" in entry
    if not has_knobs and not has_measurements:
        return GoldenEntryState.INVENTORY
    if has_knobs and not has_measurements:
        return GoldenEntryState.PROPOSAL
    if has_measurements and not has_knobs:
        raise ValueError("measurements require knobs")
    measurements = entry["measurements"]
    if not isinstance(measurements, Mapping):
        raise ValueError("measurements must be a mapping")
    required = {"emmy_us", "reference_us", "reference_backend"}
    missing = required - set(measurements)
    if missing:
        raise ValueError(f"measurements missing {', '.join(sorted(missing))}")
    return GoldenEntryState.VERIFIED


@dataclass(frozen=True)
class GoldenRecord:
    name: str
    gpu_name: str
    compute_cap: tuple[int, int]
    model: str | None
    program_index: int
    program_wire: dict
    origins: tuple[str, ...]
    knobs: dict
    measurements: dict | None
    ranking: dict | None
    loop_index: int | None = None
    loop_wire: dict | None = None

    @cached_property
    def program(self):
        """Decode the stable Torch IR payload once per embedded program."""
        key = id(self.program_wire)
        cached = _PROGRAM_GRAPH_CACHE.get(key)
        if cached is None or cached[0] is not self.program_wire:
            graph = graph_from_wire(self.program_wire)
            _PROGRAM_GRAPH_CACHE[key] = (self.program_wire, graph)
            return graph
        return cached[1]

    @cached_property
    def target_program(self):
        """Derive the disposable standalone program selected by this record."""
        if self.loop_wire is not None:
            key = id(self.loop_wire)
            cached = _LOOP_GRAPH_CACHE.get(key)
            if cached is None or cached[0] is not self.loop_wire:
                graph = loop_graph_from_wire(self.loop_wire)
                _LOOP_GRAPH_CACHE[key] = (self.loop_wire, graph)
            else:
                graph = cached[1]
            return graph.copy()

        from emmy.compiler.pipeline import CompilerDump  # noqa: PLC0415

        return CompilerDump.frontend_reproducer_from_origins(self.program, set(self.origins))

    @property
    def target_key(self) -> tuple:
        """Document-local identity shared by candidate rows for one target."""
        return ("loop", self.loop_index) if self.loop_index is not None else ("origins", *self.origins)

    @cached_property
    def shape_key(self) -> ShapeKey:
        """Deployment join key derived by lowering the stable frontend target."""
        return ShapeKey.from_s_features(self.structural_features)

    @cached_property
    def structural_features(self) -> dict[str, float]:
        """Current compiler features, derived lazily through target provenance."""
        return dict(_derive_structural_features(self))

    @cached_property
    def origin_ops(self) -> tuple[str, ...]:
        if not self.origins:
            return ()
        by_id = {node["id"]: node["op"] for node in self.program_wire["nodes"]}
        return tuple(by_id[origin] for origin in self.origins)

    @cached_property
    def dtype(self) -> str:
        """Public dtype spelling derived from the selected frontend operation."""
        if self.loop_wire is not None:
            graph = self.target_program
            tensor = graph.buffer(graph.outputs[0])
            if tensor is None:
                raise ValueError(f"{self.name}: Loop IR target has no output tensor")
            output_dtype = tensor.dtype.name
            return {"f16": "fp16", "f32": "fp32"}.get(output_dtype, output_dtype)
        by_id = {node["id"]: node for node in self.program_wire["nodes"]}
        order = {node["id"]: index for index, node in enumerate(self.program_wire["nodes"])}
        terminal = max(self.origins, key=order.__getitem__)
        output_dtype = by_id[terminal]["outputs"][0][1]
        return {"f16": "fp16", "f32": "fp32"}.get(output_dtype, output_dtype)

    @property
    def is_matmul(self) -> bool:
        """Whether this target is a plain frontend contraction."""
        return self.shape_key.kind == "" and any(op in {"torch.matmul", "torch.linear"} for op in self.origin_ops)

    @property
    def emmy_us(self) -> float:
        return float(self.measurements["emmy_us"]) if self.measurements else 0.0

    @property
    def reference_us(self) -> float:
        return float(self.measurements["reference_us"]) if self.measurements else 0.0

    @property
    def reference_backend(self) -> str | None:
        return str(self.measurements["reference_backend"]) if self.measurements else None

    @property
    def ratio(self) -> float:
        return self.reference_us / self.emmy_us if self.emmy_us else 0.0

    @property
    def golden(self) -> bool:
        return self.ratio >= 0.95

    @property
    def fast_math(self) -> bool:
        return fast_math_knobs(self.knobs)

    @property
    def is_routing(self) -> bool:
        return bool(self.knobs) and all(str(key).split("@", 1)[0] == "PLACE" for key in self.knobs)

    @property
    def dynamic(self) -> bool:
        return self.shape_key.is_dyn

    @property
    def sm_count(self) -> int | None:
        from emmy import gpu  # noqa: PLC0415

        spec = gpu.by_name(self.gpu_name)
        return spec.sm_count if spec else None


def _require_keys(value: Mapping, allowed: set[str], where: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{where}: unknown field(s): {', '.join(sorted(unknown))}")


def _positive_number(value, where: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or value <= 0:
        raise ValueError(f"{where} must be a positive number")


def _validate_target(target: object, *, index: int, program_wire: dict, loops: list[dict]) -> None:
    where = f"configs[{index}].target"
    if not isinstance(target, Mapping):
        raise ValueError(f"{where} must be a mapping")
    _require_keys(target, {"origins", "loop"}, where)
    if set(target) == {"origins"}:
        origins = target["origins"]
        if not isinstance(origins, list) or not origins or not all(isinstance(origin, str) and origin for origin in origins):
            raise ValueError(f"{where}.origins must be a non-empty list of node ids")
        node_ids = {node["id"] for node in program_wire["nodes"]}
        missing_origins = set(origins) - node_ids
        if missing_origins:
            raise ValueError(f"{where}.origins reference unknown program node(s): {', '.join(sorted(missing_origins))}")
        return
    if set(target) == {"loop"}:
        loop_ref = target["loop"]
        if isinstance(loop_ref, bool) or not isinstance(loop_ref, int) or not 0 <= loop_ref < len(loops):
            raise ValueError(f"{where}.loop does not resolve in this document: {loop_ref!r}")
        return
    raise ValueError(f"{where} must contain exactly one of origins or loop")


def validate_golden_file(
    document: object,
    *,
    validation: GoldenFileValidation = GoldenFileValidation.WORKING,
) -> None:
    if not isinstance(document, Mapping):
        raise ValueError("golden document must be a mapping")
    _require_keys(document, {"gpu_name", "compute_cap", "model", "programs", "loops", "configs"}, "golden document")
    gpu_name = document.get("gpu_name")
    if gpu_name is not None and (not isinstance(gpu_name, str) or not gpu_name):
        raise ValueError("gpu_name must be a non-empty string")
    cap = document.get("compute_cap")
    if not isinstance(cap, list) or len(cap) != 2 or not all(isinstance(item, int) for item in cap):
        raise ValueError("compute_cap must be a two-integer list")
    if validation == GoldenFileValidation.REPOSITORY and not gpu_name:
        raise ValueError("repository golden requires gpu_name")
    if document.get("model") is not None and not isinstance(document["model"], str):
        raise ValueError("model must be a string")
    try:
        programs = validate_program_pool(document.get("programs"))
    except ValueError as exc:
        raise ValueError(f"programs: {exc}") from exc
    try:
        loops = validate_loop_program_pool(document.get("loops"))
    except ValueError as exc:
        raise ValueError(f"loops: {exc}") from exc
    configs = document.get("configs")
    if not isinstance(configs, list) or not configs:
        raise ValueError("configs must be a non-empty list")
    strict = validation in (GoldenFileValidation.PROMOTION, GoldenFileValidation.REPOSITORY)
    for index, entry in enumerate(configs):
        where = f"configs[{index}]"
        if not isinstance(entry, Mapping):
            raise ValueError(f"{where} must be a mapping")
        _require_keys(entry, {"name", "model", "program", "target", "knobs", "measurements", "ranking"}, where)
        if not isinstance(entry.get("name"), str) or not entry["name"]:
            raise ValueError(f"{where}.name must be a non-empty string")
        if entry.get("model") is not None and not isinstance(entry["model"], str):
            raise ValueError(f"{where}.model must be a string")
        program_ref = entry.get("program")
        if isinstance(program_ref, bool) or not isinstance(program_ref, int) or not 0 <= program_ref < len(programs):
            raise ValueError(f"{where}.program does not resolve in this document: {program_ref!r}")
        try:
            graph_from_wire(programs[program_ref])
        except ValueError as exc:
            raise ValueError(f"{where}.program {program_ref}: {exc}") from exc
        _validate_target(entry.get("target"), index=index, program_wire=programs[program_ref], loops=loops)
        if "knobs" in entry and not isinstance(entry["knobs"], Mapping):
            raise ValueError(f"{where}.knobs must be a mapping")
        if "knobs" in entry:
            families = {str(key).split("@", 1)[0] for key in entry["knobs"]}
            if "PLACE" in families and families != {"PLACE"}:
                raise ValueError(f"{where} mixes PLACE routing knobs with schedule knobs")
        if "ranking" in entry and not isinstance(entry["ranking"], Mapping):
            raise ValueError(f"{where}.ranking must be a mapping")
        if strict and "ranking" in entry:
            raise ValueError(f"{where} working ranking metadata cannot be promoted")
        try:
            state = golden_entry_state(entry)
        except ValueError as exc:
            raise ValueError(f"{where} ({entry.get('name', '?')}): {exc}") from exc
        if strict and state != GoldenEntryState.VERIFIED:
            raise ValueError(f"{where} repository promotion requires knobs and paired positive timings")
        if state == GoldenEntryState.VERIFIED:
            measurements = entry["measurements"]
            _require_keys(measurements, {"emmy_us", "reference_us", "reference_backend"}, f"{where}.measurements")
            _positive_number(measurements["emmy_us"], f"{where}.measurements.emmy_us")
            _positive_number(measurements["reference_us"], f"{where}.measurements.reference_us")
            if not isinstance(measurements["reference_backend"], str) or not measurements["reference_backend"]:
                raise ValueError(f"{where}.measurements.reference_backend must be a non-empty string")


def load_golden_file(
    path: str | Path,
    *,
    validation: GoldenFileValidation = GoldenFileValidation.WORKING,
) -> dict:
    source = Path(path)
    try:
        document = yaml.safe_load(source.read_text())
        validate_golden_file(document, validation=validation)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise ValueError(f"invalid golden file {source}: {exc}") from exc
    return document


def golden_record_from_entry(document: Mapping, entry: Mapping) -> GoldenRecord:
    target = entry["target"]
    loop_index = target.get("loop")
    return GoldenRecord(
        name=entry["name"],
        gpu_name=document.get("gpu_name") or "",
        compute_cap=tuple(document["compute_cap"]),
        model=entry.get("model", document.get("model")),
        program_index=entry["program"],
        program_wire=document["programs"][entry["program"]],
        origins=tuple(target.get("origins", ())),
        loop_index=loop_index,
        loop_wire=document.get("loops", [])[loop_index] if loop_index is not None else None,
        knobs=dict(entry.get("knobs") or {}),
        measurements=dict(entry["measurements"]) if entry.get("measurements") is not None else None,
        ranking=dict(entry["ranking"]) if entry.get("ranking") is not None else None,
    )


def load_golden_records(document: Mapping) -> list[GoldenRecord]:
    return [golden_record_from_entry(document, entry) for entry in document["configs"]]


_STRUCTURAL_CACHE: dict[tuple, tuple[tuple[str, float], ...]] = {}
_PROGRAM_TARGET_CACHE: dict[
    tuple[int, tuple[int, int], str],
    dict[frozenset[str], set[tuple[tuple[str, float], ...]]],
] = {}


def _derive_structural_features(record: GoldenRecord) -> tuple[tuple[str, float], ...]:
    """Lower one persisted frontend target and recover its unique ``S_*`` row."""
    payload_id = id(record.loop_wire) if record.loop_wire is not None else id(record.program_wire)
    key = (payload_id, record.target_key, record.compute_cap)
    cached = _STRUCTURAL_CACHE.get(key)
    if cached is not None:
        return cached

    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.compiler.ir.loop import LoopOp  # noqa: PLC0415
    from emmy.compiler.pipeline import LOOP_PASSES, Pipeline  # noqa: PLC0415
    from emmy.compiler.pipeline.knob import STRUCT_PREFIX  # noqa: PLC0415

    if record.loop_wire is not None:
        ctx = Context.from_target(record.compute_cap, gpu_name=record.gpu_name or None)
        lowered = Pipeline.build(LOOP_PASSES).run(record.target_program, ctx=ctx)
        output_nodes = {
            node.id for output in lowered.outputs if (node := lowered.producer(output)) is not None and isinstance(node.op, LoopOp)
        }
        signatures = {
            tuple(
                sorted(
                    (name, float(value))
                    for name, value in (getattr(lowered.nodes[node_id].op, "knobs", {}) or {}).items()
                    if name.startswith(STRUCT_PREFIX)
                )
            )
            for node_id in output_nodes
        }
        signatures.discard(())
        if len(signatures) != 1:
            raise ValueError(f"{record.name}: Loop IR target resolves to {len(signatures)} structural targets")
        result = next(iter(signatures))
        _STRUCTURAL_CACHE[key] = result
        return result

    from emmy.compiler import provenance  # noqa: PLC0415
    from emmy.compiler.pipeline.passes.lowering.tile._flash import fused_producer_ids  # noqa: PLC0415

    wanted = set(record.origins)
    program_key = (id(record.program_wire), record.compute_cap, record.gpu_name)
    target_index = _PROGRAM_TARGET_CACHE.get(program_key)
    if target_index is None:
        graph = record.program.copy()
        provenance.seed(graph)
        ctx = Context.from_target(record.compute_cap, gpu_name=record.gpu_name or None)
        lowered = Pipeline.build(LOOP_PASSES).run(graph, ctx=ctx)
        absorbed_ids = {
            producer for node in lowered.nodes.values() if isinstance(node.op, LoopOp) for producer in fused_producer_ids(lowered, node)
        }
        target_index = {}
        for node_id in lowered.topological_order():
            node = lowered.nodes[node_id]
            if not isinstance(node.op, LoopOp) or node_id in absorbed_ids:
                continue
            absorbed = [lowered.nodes[producer] for producer in fused_producer_ids(lowered, node) if producer in lowered.nodes]
            origins = frozenset(origin for item in (node, *absorbed) for origin in provenance.get(item) if origin in record.program.nodes)
            features = tuple(
                sorted(
                    (name, float(value)) for name, value in (getattr(node.op, "knobs", {}) or {}).items() if name.startswith(STRUCT_PREFIX)
                )
            )
            if origins and features:
                target_index.setdefault(origins, set()).add(features)
        _PROGRAM_TARGET_CACHE[program_key] = target_index

    signatures = target_index.get(frozenset(wanted), set())
    if not signatures:
        raise ValueError(f"{record.name}: provenance target {sorted(wanted)} no longer resolves after lowering")
    if len(signatures) != 1:
        raise ValueError(
            f"{record.name}: provenance target {sorted(wanted)} resolves to {len(signatures)} structural targets; "
            "the stable target selector is ambiguous"
        )
    result = next(iter(signatures))
    _STRUCTURAL_CACHE[key] = result
    return result


def dump_golden_file(
    document: Mapping,
    path: str | Path,
    *,
    validation: GoldenFileValidation = GoldenFileValidation.WORKING,
    overwrite: bool = False,
) -> Path:
    validate_golden_file(document, validation=validation)
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} already exists; pass overwrite=True to replace it")
    destination.parent.mkdir(parents=True, exist_ok=True)
    styled = dict(document)
    styled["programs"] = [_style_program(program) for program in document["programs"]]
    if "loops" in document:
        styled["loops"] = [_style_program(program) for program in document["loops"]]
    payload = yaml.dump(styled, Dumper=_GoldenDumper, sort_keys=False, width=140)
    temporary = None
    mode = destination.stat().st_mode & 0o777 if destination.exists() else 0o644
    try:
        with tempfile.NamedTemporaryFile("w", dir=destination.parent, prefix=f".{destination.name}.", delete=False) as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
            temporary = Path(output.name)
        temporary.chmod(mode)
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination


def is_repository_golden_path(path: str | Path) -> bool:
    resolved = Path(path).resolve()
    repository = _GOLDENS_DIR.resolve()
    return resolved == repository or repository in resolved.parents


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
