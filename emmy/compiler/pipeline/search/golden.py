"""Self-contained golden YAML records and repository index.

Golden YAML is a persistence format, not a Python class discriminator.  Every
record points at a stable Torch IR program in the same document and carries the
target identity needed by search consumers directly as data.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from numbers import Real
from pathlib import Path

import yaml

from emmy.compiler.loop_wire import loop_graph_from_wire, validate_loop_program_pool
from emmy.compiler.pipeline.search.data.shape import ShapeKey
from emmy.compiler.structural import digest
from emmy.compiler.torch_wire import graph_from_wire, validate_program_pool
from emmy.recipe.bundled import default_recipe_root

_HARDWARE_GOLDENS_DIR = Path(__file__).parent / "goldens"
_RECIPE_GOLDEN_DIR = "golden"
_PROGRAM_GRAPH_CACHE: dict[int, tuple[dict, object]] = {}
_LOOP_GRAPH_CACHE: dict[int, tuple[dict, object]] = {}
_SAFE_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


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


def precision_trading_pins(pins: Mapping) -> bool:
    """Whether input pins enable any precision-trading enumeration path."""
    umbrella = bool(pins.get("FAST_MATH", False))
    return any(bool(pins.get(name, umbrella)) for name in ("FAST_EXP", "F16_MMA_F32_ACC", "FP8_MMA"))


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
    bindings: tuple[tuple[str, int], ...]
    pins: tuple[tuple[str, object], ...]
    knobs: dict
    measurements: dict | None
    ranking: dict | None
    loop_index: int | None = None
    loop_wire: dict | None = None

    @cached_property
    def pool_key(self) -> tuple:
        """Which candidate pool this record belongs to — the ONE place that question is answered, so every
        consumer that groups goldens groups them the same way.

        Derived today, because nothing records it: ``enumerate_graph(self.target_program, ctx)`` under
        ``self.pin_map`` reads the card, the wire the target specializes from, which node it selects, the
        bindings and the pins, and records agreeing on all five run the same enumeration. Two consequences
        worth knowing before relying on it. It is SUFFICIENT, not necessary — it never fuses two pools that
        differ, but it splits two recordings of one program made in different sessions, whose node ids differ
        and whose pools do not. And it keys on what the enumeration READS, never on what it produced, so it
        does not go stale when the scheduler changes.

        When a group identity is recorded with the golden instead, this property returns it and its callers
        do not change."""
        wire = self.loop_wire if self.loop_wire is not None else self.program_wire
        kind = "loop" if self.loop_wire is not None else "prog"
        digest = hashlib.blake2b(json.dumps(wire, sort_keys=True).encode(), digest_size=16).digest()
        return (self.gpu_name, tuple(self.compute_cap), kind, digest, tuple(self.origins), tuple(self.bindings), self.pin_key)

    @cached_property
    def pin_key(self) -> tuple:
        """This record's pins as a hashable tuple — already sorted, as the loader stores them."""
        return tuple((k, str(v)) for k, v in self.pins)

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
        from emmy.compiler.specialize import specialize_program  # noqa: PLC0415

        if self.loop_wire is not None:
            key = id(self.loop_wire)
            cached = _LOOP_GRAPH_CACHE.get(key)
            if cached is None or cached[0] is not self.loop_wire:
                graph = loop_graph_from_wire(self.loop_wire)
                _LOOP_GRAPH_CACHE[key] = (self.loop_wire, graph)
            else:
                graph = cached[1]
            return specialize_program(graph, dict(self.bindings), loop=True)

        from emmy.compiler.pipeline import CompilerDump  # noqa: PLC0415

        graph = CompilerDump.frontend_reproducer_from_origins(self.program, set(self.origins))
        return specialize_program(graph, dict(self.bindings))

    @property
    def target_key(self) -> tuple:
        """Document-local identity shared by candidate rows for one target."""
        return ("loop", self.loop_index) if self.loop_index is not None else ("origins", *self.origins)

    @property
    def binding_map(self) -> dict[str, int]:
        return dict(self.bindings)

    @property
    def pin_map(self) -> dict[str, object]:
        return dict(self.pins)

    @cached_property
    def shape_key(self) -> ShapeKey:
        """The arithmetic-identity descriptor for eval / diagnostics grouping, derived from the
        lowered target's stamped histogram. NOT the deploy join key — that is
        :func:`kernel_identity` (strict structural identity); this key only groups eval rows."""
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
        """Whether this target is a plain frontend contraction — read off the STORED origin
        operations alone (a fused norm→linear names its norm origin too, so the subset test
        separates them), never by lowering the target: an eval listing must classify a record
        the current compiler can no longer lower."""
        return bool(self.origin_ops) and set(self.origin_ops) <= {"torch.matmul", "torch.linear"}

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
    _require_keys(
        document,
        {"gpu_name", "compute_cap", "model", "model_quant_digest", "programs", "loops", "configs"},
        "golden document",
    )
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
    quant_digest = document.get("model_quant_digest")
    if quant_digest is not None and (not isinstance(quant_digest, str) or re.fullmatch(r"[0-9a-f]{16}", quant_digest) is None):
        raise ValueError("model_quant_digest must be a 16-character lowercase hexadecimal digest")
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
        _require_keys(entry, {"model", "program", "target", "realizations"}, where)
        if entry.get("model") is not None and not isinstance(entry["model"], str):
            raise ValueError(f"{where}.model must be a string")
        program_ref = entry.get("program")
        if isinstance(program_ref, bool) or not isinstance(program_ref, int) or not 0 <= program_ref < len(programs):
            raise ValueError(f"{where}.program does not resolve in this document: {program_ref!r}")
        # The pool check above already decoded every program. A whole-model inventory points
        # hundreds of configurations at a handful of programs, so do not decode again per config.
        _validate_target(entry.get("target"), index=index, program_wire=programs[program_ref], loops=loops)
        realizations = entry.get("realizations")
        if not isinstance(realizations, list) or not realizations:
            raise ValueError(f"{where}.realizations must be a non-empty list")
        for realization_index, realization in enumerate(realizations):
            realization_where = f"{where}.realizations[{realization_index}]"
            if not isinstance(realization, Mapping):
                raise ValueError(f"{realization_where} must be a mapping")
            _require_keys(realization, {"name", "bindings", "pins", "knobs", "measurements", "ranking"}, realization_where)
            if not isinstance(realization.get("name"), str) or not realization["name"]:
                raise ValueError(f"{realization_where}.name must be a non-empty string")
            bindings = realization.get("bindings")
            if not isinstance(bindings, Mapping):
                raise ValueError(f"{realization_where}.bindings must be a mapping")
            for name, size in bindings.items():
                if not isinstance(name, str) or not name or type(size) is not int or size <= 0:
                    raise ValueError(f"{realization_where}.bindings must map non-empty names to positive integers")
            pins = realization.get("pins")
            if not isinstance(pins, Mapping):
                raise ValueError(f"{realization_where}.pins must be a mapping")
            from emmy.compiler.pipeline.knob import KnobType, family_of, get  # noqa: PLC0415

            for name, value in pins.items():
                descriptor = get(family_of(name)) if isinstance(name, str) and name else None
                if descriptor is None:
                    raise ValueError(f"{realization_where}.pins names unknown knob {name!r}")
                valid = {
                    KnobType.BOOL: type(value) is bool,
                    KnobType.INT: type(value) is int,
                    KnobType.STR: isinstance(value, str),
                    KnobType.BINMASK: type(value) is int or isinstance(value, str),
                }[descriptor.type]
                if not valid:
                    raise ValueError(f"{realization_where}.pins.{name} must be a {descriptor.type.value} value, got {value!r}")
            if "knobs" in realization and not isinstance(realization["knobs"], Mapping):
                raise ValueError(f"{realization_where}.knobs must be a mapping")
            if "knobs" in realization:
                from emmy.compiler.pipeline.knob import values_equal  # noqa: PLC0415

                conflicts = [
                    name
                    for name, value in pins.items()
                    if name in realization["knobs"] and not values_equal(name, value, realization["knobs"][name])
                ]
                if conflicts:
                    raise ValueError(
                        f"{realization_where} gives conflicting input pins and measured knobs for {', '.join(sorted(conflicts))}"
                    )
                families = {str(key).split("@", 1)[0] for key in realization["knobs"]}
                if "PLACE" in families and families != {"PLACE"}:
                    raise ValueError(f"{realization_where} mixes PLACE routing knobs with schedule knobs")
                if strict:
                    for family in families:
                        scoped = [str(key) for key in realization["knobs"] if str(key).split("@", 1)[0] == family]
                        # A stamped row legitimately carries the primary node's bare key beside
                        # axis-scoped site decisions (``STAGE: d2/smem`` + ``STAGE@a1: ''``) — that
                        # IS the canonical codec spelling. The ambiguous shape is a bare OFF next
                        # to scoped keys: replaying it would fan OFF across every eligible site,
                        # so ``stamp_schedule_families`` drops it and a recording must not store it.
                        if family in scoped and any("@" in key for key in scoped) and str(realization["knobs"][family]) == "":
                            raise ValueError(
                                f"{realization_where}.knobs mixes bare and axis-scoped {family} keys with a bare OFF; "
                                "the scoped spelling is the site decision — drop the bare OFF"
                            )
            if "ranking" in realization and not isinstance(realization["ranking"], Mapping):
                raise ValueError(f"{realization_where}.ranking must be a mapping")
            if strict and "ranking" in realization:
                raise ValueError(f"{realization_where} working ranking metadata cannot be promoted")
            try:
                state = golden_entry_state(realization)
            except ValueError as exc:
                raise ValueError(f"{realization_where} ({realization.get('name', '?')}): {exc}") from exc
            if strict and state != GoldenEntryState.VERIFIED:
                raise ValueError(f"{realization_where} repository promotion requires knobs and paired positive timings")
            if state == GoldenEntryState.VERIFIED:
                measurements = realization["measurements"]
                _require_keys(measurements, {"emmy_us", "reference_us", "reference_backend"}, f"{realization_where}.measurements")
                _positive_number(measurements["emmy_us"], f"{realization_where}.measurements.emmy_us")
                _positive_number(measurements["reference_us"], f"{realization_where}.measurements.reference_us")
                if not isinstance(measurements["reference_backend"], str) or not measurements["reference_backend"]:
                    raise ValueError(f"{realization_where}.measurements.reference_backend must be a non-empty string")


def load_golden_file(
    path: str | Path,
    *,
    validation: GoldenFileValidation = GoldenFileValidation.WORKING,
) -> dict:
    source = Path(path)
    try:
        document = yaml.load(source.read_text(), Loader=_SAFE_LOADER)
        validate_golden_file(document, validation=validation)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise ValueError(f"invalid golden file {source}: {exc}") from exc
    return document


def golden_record_from_entry(document: Mapping, entry: Mapping, realization: Mapping) -> GoldenRecord:
    target = entry["target"]
    loop_index = target.get("loop")
    return GoldenRecord(
        name=realization["name"],
        gpu_name=document.get("gpu_name") or "",
        compute_cap=tuple(document["compute_cap"]),
        model=entry.get("model", document.get("model")),
        program_index=entry["program"],
        program_wire=document["programs"][entry["program"]],
        origins=tuple(target.get("origins", ())),
        bindings=tuple(sorted(realization["bindings"].items())),
        pins=tuple(sorted(realization["pins"].items())),
        loop_index=loop_index,
        loop_wire=document.get("loops", [])[loop_index] if loop_index is not None else None,
        knobs=dict(realization.get("knobs") or {}),
        measurements=dict(realization["measurements"]) if realization.get("measurements") is not None else None,
        ranking=dict(realization["ranking"]) if realization.get("ranking") is not None else None,
    )


def load_golden_records(document: Mapping) -> list[GoldenRecord]:
    return [
        golden_record_from_entry(document, entry, realization) for entry in document["configs"] for realization in entry["realizations"]
    ]


_STRUCTURAL_CACHE: dict[tuple, tuple[tuple[str, float], ...]] = {}
_IDENTITY_CACHE: dict[tuple, str | None] = {}
#: The persisted identity memo: {record fingerprint: identity | None}, valid only under one
#: compiler fingerprint. Purely derived data — a stale or missing store just re-derives.
_IDENTITY_STORE: dict | None = None
_IDENTITY_STORE_DIRTY: bool = False
_WIRE_DIGESTS: dict[int, str] = {}


def _compiler_fingerprint() -> str:
    """A cheap fingerprint of the compiler tree — (path, mtime, size) of every ``emmy/compiler``
    source file. Any edit invalidates the persisted identity memo, so a derivation can never be
    replayed across compiler versions."""
    import emmy.compiler as _pkg  # noqa: PLC0415

    root = Path(_pkg.__file__).parent
    parts = []
    for path in sorted(root.rglob("*.py")):
        st = path.stat()
        parts.append(f"{path.relative_to(root)}:{st.st_mtime_ns}:{st.st_size}")
    return digest("\n".join(parts))


def _identity_store() -> dict:
    global _IDENTITY_STORE
    if _IDENTITY_STORE is None:
        import json  # noqa: PLC0415

        from emmy import config  # noqa: PLC0415

        fingerprint = _compiler_fingerprint()
        entries: dict = {}
        verdicts: dict = {}
        try:
            payload = json.loads(config.golden_identity_cache_path().read_text())
            if payload.get("fingerprint") == fingerprint:
                entries = payload.get("entries", {})
                verdicts = payload.get("verdicts", {})
        except (OSError, ValueError):
            pass
        _IDENTITY_STORE = {"fingerprint": fingerprint, "entries": entries, "verdicts": verdicts}
    return _IDENTITY_STORE


def flush_identity_store() -> None:
    """Persist newly derived identities (atomic replace; concurrent writers last-win — a lost
    write only re-derives later)."""
    global _IDENTITY_STORE_DIRTY
    if not _IDENTITY_STORE_DIRTY or _IDENTITY_STORE is None:
        return
    import json  # noqa: PLC0415

    from emmy import config  # noqa: PLC0415

    path = config.golden_identity_cache_path()
    try:
        # MERGE with the on-disk state before writing: concurrent processes (xdist workers each
        # walking one golden set) flush independently, and overwrite-last-wins silently dropped
        # every other worker's derivations.
        try:
            on_disk = json.loads(path.read_text())
        except (OSError, ValueError):
            on_disk = None
        if on_disk is not None and on_disk.get("fingerprint") == _IDENTITY_STORE["fingerprint"]:
            for section in ("entries", "verdicts"):
                merged = dict(on_disk.get(section, {}))
                merged.update(_IDENTITY_STORE.get(section, {}))
                _IDENTITY_STORE[section] = merged
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as out:
            json.dump(_IDENTITY_STORE, out)
            temporary = Path(out.name)
        temporary.replace(path)
        _IDENTITY_STORE_DIRTY = False
    except OSError:
        pass  # the store is a memo; failing to persist only costs a re-derivation


def _record_fingerprint(record: GoldenRecord) -> str:
    """A stable content digest for one record's TARGET (identity depends on nothing else): the
    persisted wire payload, the target selector, bindings, card. Wire digests are memoized per
    payload object — one document's records share their program pool."""
    import json  # noqa: PLC0415

    wire = record.loop_wire if record.loop_wire is not None else record.program_wire
    wd = _WIRE_DIGESTS.get(id(wire))
    if wd is None:
        wd = _WIRE_DIGESTS.setdefault(id(wire), digest(json.dumps(wire, sort_keys=True, default=str)))
    return digest(wd, str(record.target_key), str(record.bindings), str(record.compute_cap), record.gpu_name or "")


#: Enumerated rows per (target, pins) — sibling realizations of one config decode against one
#: enumeration (the tripwire and the migration validator walk whole files) — and ONE Context per
#: card, so same-shape targets share the schedule pool cache across configs and files.
_DECODE_ROWS_CACHE: dict[tuple, list[dict]] = {}
_DECODE_CTX_CACHE: dict[tuple, object] = {}


def _record_cache_key(record: GoldenRecord) -> tuple:
    payload_id = id(record.loop_wire) if record.loop_wire is not None else id(record.program_wire)
    return (payload_id, record.target_key, record.compute_cap, record.bindings)


def _target_kernel_nodes(record: GoldenRecord):
    """The record's target kernels in the CURRENT compiler: lower the persisted program through
    the loop passes and select the target's ``LoopOp`` node(s) — every output kernel for a Loop IR
    target, the provenance-selected ones for a frontend target. Returns ``(lowered graph, nodes)``.
    Raises when the selector no longer resolves — the strict tripwire's loud case."""
    from emmy.compiler import provenance  # noqa: PLC0415
    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.compiler.ir.loop import LoopOp  # noqa: PLC0415
    from emmy.compiler.pipeline import LOOP_PASSES, Pipeline  # noqa: PLC0415

    ctx = Context.from_target(record.compute_cap, gpu_name=record.gpu_name or None)
    graph = record.target_program.copy()
    if record.loop_wire is None:
        provenance.seed(graph)
    lowered = Pipeline.build(LOOP_PASSES).run(graph, ctx=ctx)
    if record.loop_wire is not None:
        nodes = [node for output in lowered.outputs if (node := lowered.producer(output)) is not None and isinstance(node.op, LoopOp)]
    else:
        wanted = frozenset(record.origins)
        nodes = []
        for node_id in lowered.topological_order():
            node = lowered.nodes[node_id]
            if not isinstance(node.op, LoopOp):
                continue
            origins = frozenset(origin for origin in provenance.get(node) if origin in record.program.nodes)
            if origins == wanted:
                nodes.append(node)
    if not nodes:
        raise ValueError(f"{record.name}: the persisted target selects no kernel after lowering")
    return lowered, nodes


def _recognized_target(record: GoldenRecord):
    """The record's ONE recognized tile (loud): the target must select exactly one kernel, and the
    shared recognition core must lift it. Returns the ``TileOp``."""
    from emmy.compiler.pipeline.passes.lowering.tile._lift import recognized_tile  # noqa: PLC0415

    lowered, nodes = _target_kernel_nodes(record)
    if len(nodes) != 1:
        raise ValueError(f"{record.name}: target lowers to {len(nodes)} kernels — a row decorates exactly one")
    node = nodes[0]
    node.op.populate_io(lowered, node)
    tile = recognized_tile(node.op, name=node.id)
    # The live fork's root op has its io populated by the matcher; mirror it here so the dtype
    # half of the identity (``deploy_identity``) reads the same output fingerprint.
    tile.outputs = {node.output.name: node.output}
    return tile


def decode_record(record: GoldenRecord) -> str | None:
    """STRICTLY decode one record against the current compiler — ``None`` on success, else the
    failure reason. This is the replayability contract the nightly onboarding job gates: the persisted
    program selects exactly one kernel; a ROUTING record's ``PLACE`` keys resolve to legal cut
    seams on the recognized tree; a SCHEDULE record's spelled row equals EXACTLY ONE enumerated
    leaf (``canonical_row_key`` equality under the record's own pins) — no prefix matching, no
    any-of, no classified shape."""
    from emmy.compiler.ir.tile.path import resolve, sites  # noqa: PLC0415
    from emmy.compiler.pipeline.knob import schedule_row_key  # noqa: PLC0415
    from emmy.compiler.pipeline.passes.lowering.tile._classify import fused_view  # noqa: PLC0415
    from emmy.compiler.pipeline.passes.lowering.tile._cut import cuttable_seams  # noqa: PLC0415

    try:
        tile = _recognized_target(record)
    except Exception as exc:  # noqa: BLE001 — the reason IS the product here
        return f"{type(exc).__name__}: {exc}"
    if record.is_routing:
        verdict_key = digest(_record_fingerprint(record), str(sorted(record.knobs.items())), str(record.pins))
        store = _identity_store()
        verdicts = store.setdefault("verdicts", {})
        if verdict_key in verdicts:
            return verdicts[verdict_key]
        pro = fused_view(tile)
        route_tree, route_free, route_stores = (
            (pro[0], (*tile.place.free, pro[1]), pro[2]) if pro is not None else (tile.op, tile.place.free, tile.stores)
        )
        seams = cuttable_seams(route_tree, route_stores, route_free)
        all_sites = sites(route_tree)
        for key, value in record.knobs.items():
            if str(value) != "cut":
                return _remember_verdict(verdict_key, f"routing value {key}={value!r} is not a cut")
            if str(key) == "PLACE":
                # The bare family key IS the codec's "shallowest cuttable seam" spelling
                # (``route_cut``'s pin semantics) — it decodes iff any seam is legal.
                if not seams:
                    return _remember_verdict(verdict_key, "bare PLACE=cut recorded, but the recognized tree has no legal cut seam")
                continue
            try:
                site = resolve(route_tree, str(key), all_sites=all_sites)
            except ValueError as exc:
                return _remember_verdict(verdict_key, f"routing key {key!r} does not resolve: {exc}")
            if site is None or site not in seams:
                return _remember_verdict(verdict_key, f"routing key {key!r} names no legal cut seam on the recognized tree")
        return _remember_verdict(verdict_key, None)
    verdict_key = digest(_record_fingerprint(record), str(sorted(record.knobs.items())), str(record.pins))
    store = _identity_store()
    verdicts = store.setdefault("verdicts", {})
    if verdict_key in verdicts:
        return verdicts[verdict_key]
    candidates = _candidate_row_keys(record)
    if schedule_row_key(record.knobs) in candidates:
        reason = None
    else:
        reason = f"no enumerated row equals the recording ({len(candidates)} candidate rows)"
    verdicts[verdict_key] = reason
    global _IDENTITY_STORE_DIRTY
    _IDENTITY_STORE_DIRTY = True
    return reason


def _remember_verdict(key: str, reason: str | None) -> str | None:
    global _IDENTITY_STORE_DIRTY
    _identity_store().setdefault("verdicts", {})[key] = reason
    _IDENTITY_STORE_DIRTY = True
    return reason


def _candidate_row_keys(record: GoldenRecord) -> frozenset:
    """Every schedule-row identity the record's target can realize under its pins: the fork
    leaves' rows, PLUS each resolved kernel's own realized row — a forkless kernel (the schedule
    space collapsed to one row, often the all-OFF anchor) never opens a fork, so its one row is
    read off the resolved op instead."""
    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.compiler.ir.tile import TileOp  # noqa: PLC0415
    from emmy.compiler.pipeline import TILE_PASSES, Pipeline  # noqa: PLC0415
    from emmy.compiler.pipeline.fork import flatten_leaves  # noqa: PLC0415
    from emmy.compiler.pipeline.knob import schedule_row_key  # noqa: PLC0415
    from emmy.compiler.pipeline.pipeline import Run, _is_structural_option  # noqa: PLC0415
    from emmy.compiler.pipeline.search.pins import pinned_knobs  # noqa: PLC0415

    cache_key = (_record_cache_key(record), record.pins)
    cached = _DECODE_ROWS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    ctx_key = (record.compute_cap, record.gpu_name or None)
    ctx = _DECODE_CTX_CACHE.get(ctx_key)
    if ctx is None:
        ctx = _DECODE_CTX_CACHE.setdefault(ctx_key, Context.from_target(ctx_key[0], gpu_name=ctx_key[1]))
    keys: set = set()

    def decide(fp):
        leaves = flatten_leaves(fp.options)
        ops = [o for o in leaves if not _is_structural_option(o)]
        for leaf in ops:
            row = dict(getattr(leaf, "knobs", None) or {})
            if row:
                keys.add(schedule_row_key(row))
        return ops[0] if ops else leaves[0]

    with pinned_knobs(record.pin_map):
        out, _ = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=ctx).resolve(record.target_program.copy(), decide)
    for node in out.nodes.values():
        if isinstance(node.op, TileOp):
            keys.add(schedule_row_key(dict(node.op.knobs or {})))
    result = frozenset(keys)
    _DECODE_ROWS_CACHE[cache_key] = result
    return result


def kernel_identity(record: GoldenRecord) -> str | None:
    """The record's kernel identity under the CURRENT compiler — the verified-tier join key
    (``_schedule.deploy_identity``) of the recognized tile of the record's ONE target kernel,
    derived through the exact recognition core the live compile uses (``_lift.recognized_tile``).
    ``None`` when the record cannot carry a deploy identity: the target lowers to several kernels
    (a schedule row decorates exactly one), or the selector/recognition fails — best-effort here
    (a corpus row must never break a compile); nightly strict decoding is where failure is loud."""
    global _IDENTITY_STORE_DIRTY
    key = _record_cache_key(record)
    if key in _IDENTITY_CACHE:
        return _IDENTITY_CACHE[key]
    store = _identity_store()
    fingerprint = _record_fingerprint(record)
    if fingerprint in store["entries"]:
        identity = store["entries"][fingerprint]
        _IDENTITY_CACHE[key] = identity
        return identity
    try:
        from emmy.compiler.pipeline.passes.lowering.tile._schedule import deploy_identity  # noqa: PLC0415

        identity = deploy_identity(_recognized_target(record))
    except Exception:  # noqa: BLE001 — see the docstring; the decode tripwire re-derives loudly
        identity = None
    _IDENTITY_CACHE[key] = identity
    store["entries"][fingerprint] = identity
    _IDENTITY_STORE_DIRTY = True
    return identity


_PROGRAM_TARGET_CACHE: dict[
    tuple[int, tuple[int, int], str, tuple[tuple[str, int], ...]],
    dict[frozenset[str], set[tuple[tuple[str, float], ...]]],
] = {}


def _derive_structural_features(record: GoldenRecord) -> tuple[tuple[str, float], ...]:
    """Lower one persisted frontend target and recover its unique ``S_*`` row."""
    payload_id = id(record.loop_wire) if record.loop_wire is not None else id(record.program_wire)
    key = (payload_id, record.target_key, record.compute_cap, record.bindings)
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

    wanted = set(record.origins)
    program_key = (id(record.program_wire), record.compute_cap, record.gpu_name, record.bindings)
    target_index = _PROGRAM_TARGET_CACHE.get(program_key)
    if target_index is None:
        from emmy.compiler.specialize import specialize_program  # noqa: PLC0415

        graph = specialize_program(record.program, record.binding_map)
        provenance.seed(graph)
        ctx = Context.from_target(record.compute_cap, gpu_name=record.gpu_name or None)
        lowered = Pipeline.build(LOOP_PASSES).run(graph, ctx=ctx)
        target_index = {}
        for node_id in lowered.topological_order():
            node = lowered.nodes[node_id]
            if not isinstance(node.op, LoopOp):
                continue
            origins = frozenset(origin for origin in provenance.get(node) if origin in record.program.nodes)
            feature_map = {
                name: float(value) for name, value in (getattr(node.op, "knobs", {}) or {}).items() if name.startswith(STRUCT_PREFIX)
            }
            features = tuple(sorted(feature_map.items()))
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


_POOL_KEYS = ("programs", "loops")
_POOL_CACHE: tuple[list, dict[str, str]] | None = None


def _dump_block(key: str, value: object) -> str:
    """One top-level key as YAML. Block-style keys concatenate into exactly the document
    ``yaml.dump`` writes for the whole mapping, so a block can be reused verbatim."""
    return yaml.dump({key: value}, Dumper=_GoldenDumper, sort_keys=False, width=140)


def _pool_blocks(document: Mapping, *, reuse: bool) -> dict[str, str]:
    """The serialized program and loop pools, kept across repeated persists of one document.

    Keyed by pool identity, and the cache holds the pools it serialized: a document whose pools
    are the very objects that were dumped last has the same bytes for them.
    """
    global _POOL_CACHE

    pools = [document.get(key) for key in _POOL_KEYS]
    if reuse and _POOL_CACHE is not None and all(cached is pool for cached, pool in zip(_POOL_CACHE[0], pools, strict=True)):
        return _POOL_CACHE[1]
    blocks = {key: _dump_block(key, [_style_program(program) for program in document[key]]) for key in _POOL_KEYS if key in document}
    if reuse:
        _POOL_CACHE = (pools, blocks)
    return blocks


def dump_golden_file(
    document: Mapping,
    path: str | Path,
    *,
    validation: GoldenFileValidation = GoldenFileValidation.WORKING,
    overwrite: bool = False,
    incremental: bool = False,
) -> Path:
    """Write one golden document, atomically.

    ``incremental`` is for the repeated working-golden persists of a single loaded document that
    ``tune`` makes as it records ranking feedback per target: those persists mutate realizations
    only, so the program and loop pools keep the text they were serialized to instead of being
    reserialized once per target. The whole document is still validated, and the bytes written are
    the ones a full dump writes. Canonical and promotion dumps leave it off and reserialize everything.
    """
    validate_golden_file(document, validation=validation)
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} already exists; pass overwrite=True to replace it")
    destination.parent.mkdir(parents=True, exist_ok=True)
    blocks = _pool_blocks(document, reuse=incremental)
    payload = "".join(blocks[key] if key in blocks else _dump_block(key, value) for key, value in document.items())
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
    hardware_root = _HARDWARE_GOLDENS_DIR.resolve()
    if resolved == hardware_root or hardware_root in resolved.parents:
        return True
    with default_recipe_root() as recipe_root:
        if recipe_root is None:
            return False
        try:
            relative = resolved.relative_to(recipe_root.resolve())
        except ValueError:
            return False
        return len(relative.parts) >= 2 and relative.parts[1] == _RECIPE_GOLDEN_DIR


@contextmanager
def _repository_golden_paths():
    """Yield model-agnostic hardware goldens plus recipe-local model goldens."""
    with default_recipe_root() as recipe_root:
        paths = list(_HARDWARE_GOLDENS_DIR.glob("*.yaml"))
        if recipe_root is not None:
            paths.extend(recipe_root.glob(f"*/{_RECIPE_GOLDEN_DIR}/*.yaml"))
        yield sorted(paths)


def _file_gpu_name(path: Path) -> str | None:
    """The document's ``gpu_name`` read off the file HEAD without parsing the body — the dump
    writes it as the first key, so a card-scoped consumer can skip foreign multi-megabyte files
    (the whole-corpus parse is the dominant first-evidence cost). ``None`` when the head does not
    carry it — the caller falls back to the full parse."""
    try:
        head = path.open("r").read(256)
    except OSError:
        return None
    for line in head.splitlines():
        if line.startswith("gpu_name:"):
            return str(yaml.load(line, Loader=_SAFE_LOADER)["gpu_name"])
    return None


#: Optional scope override for :func:`records_for_card` — the corpus the deploy tier reads.
#: ``None`` (the default, and the only value a real deploy ever sees) reads the repository files.
#: The drift audit (``search/audit.py``) installs one file's / one precision lane's records here so
#: its verdicts judge exactly that set, the way the release gate needs them scoped.
RECORDS_OVERRIDE: list[GoldenRecord] | None = None


def records_for_card(gpu_name: str, compute_cap: tuple[int, int]) -> list[GoldenRecord]:
    """The repository records for ONE card, loading only that card's files (header sniff) — the
    deploy tier's loader. ``GOLDEN_RECORDS`` stays the full corpus for the eval / fit consumers;
    both share the per-path document memo so nothing parses twice. :data:`RECORDS_OVERRIDE`
    replaces the repository corpus when the audit has scoped it."""
    if RECORDS_OVERRIDE is not None:
        return [r for r in RECORDS_OVERRIDE if r.gpu_name == gpu_name and tuple(r.compute_cap) == tuple(compute_cap)]
    records: list[GoldenRecord] = []
    with _repository_golden_paths() as paths:
        for path in paths:
            head_gpu = _file_gpu_name(path)
            if head_gpu is not None and head_gpu != gpu_name:
                continue
            records.extend(r for r in _records_of(path) if r.gpu_name == gpu_name and tuple(r.compute_cap) == tuple(compute_cap))
    return records


_DOCUMENT_MEMO: dict[Path, list[GoldenRecord]] = {}


def _records_of(path: Path) -> list[GoldenRecord]:
    cached = _DOCUMENT_MEMO.get(path)
    if cached is None:
        document = load_golden_file(path, validation=GoldenFileValidation.REPOSITORY)
        cached = _DOCUMENT_MEMO.setdefault(path, load_golden_records(document))
    return cached


def _load_goldens() -> list[GoldenRecord]:
    records: list[GoldenRecord] = []
    with _repository_golden_paths() as paths:
        for path in paths:
            records.extend(_records_of(path))
    return records


class _LazyGoldenRecords(Sequence[GoldenRecord]):
    """Load the repository corpus only when a consumer asks for evidence."""

    def __init__(self, loader: Callable[[], list[GoldenRecord]]) -> None:
        self._loader = loader

    @cached_property
    def _records(self) -> tuple[GoldenRecord, ...]:
        return tuple(self._loader())

    def __getitem__(self, index: int | slice) -> GoldenRecord | tuple[GoldenRecord, ...]:
        return self._records[index]

    def __iter__(self) -> Iterator[GoldenRecord]:
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)


GOLDEN_RECORDS: Sequence[GoldenRecord] = _LazyGoldenRecords(_load_goldens)


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
    "precision_trading_pins",
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
