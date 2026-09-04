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
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import cached_property
from numbers import Real
from pathlib import Path
from typing import NamedTuple

import yaml

from emmy import config
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
    from emmy.compiler.ir.schedule import Tile, Work  # noqa: PLC0415
    from emmy.compiler.pipeline.search.space import FAST_EXP  # noqa: PLC0415

    for key, value in knobs.items():
        spelling = str(value)
        if str(key).split("@", 1)[0] == "TILE" and spelling:
            try:
                plan = Tile.parse(spelling, Work(kind="warp", units=(1, 1)))
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


def pins_freeze_cut(pins: Mapping) -> bool:
    """Whether the input pins freeze any placement cut (a ``PLACE…=cut`` pin) — the ONE spelling
    of the predicate behind both the loader's receipt validation and :attr:`GoldenRecord.is_receipt`."""
    from emmy.compiler.pipeline.knob import family_of  # noqa: PLC0415

    return any(family_of(str(name)) == "PLACE" and str(value) == "cut" for name, value in pins.items())


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
    #: The record's stored deploy identity (``identity_key(with_io=True)``, see :func:`kernel_identity`), when the
    #: file keeps one. Model inventories mostly do not; the realization corpus does, because a new
    #: fingerprint fact must show up as a diff there rather than silently re-key a checked-in
    #: reproducer. A stored identity is the strict decode's kernel selector, and it is how a
    #: **child-identity schedule receipt** names its kernel: a record whose pins freeze a cut lowers
    #: to several kernels, and only the stored identity says which child this row's schedule
    #: decorates (and so which kernel's ``S_*`` signature its row is evidence under).
    identity: str | None = None
    #: Measured microseconds per ``Context.hardware_id``: ``{card: {emmy_us, tcompile_us}}``. A
    #: model golden is one file per card and uses the flat ``measurements`` block instead; a corpus
    #: case is one file across many cards, which a flat block cannot hold.
    latency: dict | None = None

    @property
    def is_routing(self) -> bool:
        """Whether this row records a kernel-placement decision rather than a kernel schedule."""
        return bool(self.knobs) and all(str(key).split("@", 1)[0] == "PLACE" for key in self.knobs)

    @property
    def is_receipt(self) -> bool:
        """Whether this row is a child-identity schedule receipt: a schedule row recorded behind
        pinned cut(s), whose stored ``identity`` names the child kernel the row decorates."""
        return self.identity is not None and not self.is_routing and pins_freeze_cut(dict(self.pins))

    @property
    def route(self) -> dict[str, str]:
        """The placement this record carries — every ``PLACE`` key of its pins and knobs, spelled
        as recorded. A routing row keeps it in ``knobs``; a receipt, a corpus case or an ``--ab``
        row freezes it in ``pins``. Empty for a plain schedule row, which says the kernel it
        decorates ran fused."""
        route = {str(key): str(value) for key, value in self.pins if str(key).split("@", 1)[0] == "PLACE"}
        route.update((str(key), str(value)) for key, value in self.knobs.items() if str(key).split("@", 1)[0] == "PLACE")
        return route

    @property
    def schedule_row(self) -> dict[str, str]:
        """The schedule half of the record — its decided tuning knobs minus the route, as the evidence
        index carries them (an OFF ``''`` is a decided value and stays)."""
        from emmy.compiler.pipeline.knob import tuning_knob_items  # noqa: PLC0415

        return {key: value for key, value in tuning_knob_items(self.knobs) if key.split("@", 1)[0] != "PLACE"}

    @cached_property
    def pool_group(self) -> tuple:
        """Which candidate pool this record belongs to — the ONE place that question is answered, so every
        consumer that groups goldens groups them the same way. (A grouping key over RECORDS —
        distinct from the scheduler's per-compile ``pool_id`` stamp.)

        Composed from the target kernels' identity keys — the one identity function — around the
        card and the record's pin regime: per fused kernel, the structural variant key
        (``identity_key(with_io=True, with_knobs=True)`` — cluster siblings share a schedule
        space, so they rightly share a pool) folded with the symbolic-dim hints the enumeration
        sizes against. Node-id spelling never enters, so two recordings of one program made in
        different sessions FUSE — the wire-digest key this replaces split them — and any fact
        that changes the kernels shows up in their keys, so the key stays sufficient. It keys on
        what the enumeration READS, never on what it produced, so it does not go stale when the
        scheduler changes; bindings stay out (they bind replay values, not the space).

        Best-effort like every record-side derivation: a target the current compiler no longer
        lowers falls back to the persisted wire's digest, so a stale record still groups
        deterministically (alone) instead of breaking a fit."""
        from emmy.compiler.dim import DEFAULT_SEQ_HINT  # noqa: PLC0415

        try:
            _lowered, nodes = _target_kernel_nodes(self)
            kernels = tuple(
                sorted(
                    digest(
                        op.identity_key(with_io=True, with_knobs=True) or "",
                        tuple(
                            d.hint or DEFAULT_SEQ_HINT
                            for t in (*op.inputs.values(), *op.outputs.values())
                            for d in t.shape
                            if not d.is_static
                        ),
                    )
                    for op in (node.op for node in nodes)
                )
            )
        except Exception:  # noqa: BLE001 — a stale record must never break the fit's dataset build
            wire = self.loop_wire if self.loop_wire is not None else self.program_wire
            kernels = (hashlib.blake2b(json.dumps(wire, sort_keys=True).encode(), digest_size=16).digest(), tuple(self.origins))
        return (self.gpu_name, tuple(self.compute_cap), kernels, self.pin_key)

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


#: What one card's ``latency`` entry records. ``emmy_us`` is required — it is the ratchet, and a
#: case without it stores nothing. ``tcompile_us`` is the "are we ahead of or behind torch" half
#: and is OPTIONAL, because some targets have no torch twin to compile: a provenance-reconstructed
#: frontend program benches against eager and Emmy only. Refusing to store the ratchet because the
#: comparison is unavailable would discard the more important number of the two.
LATENCY_FIELDS = ("emmy_us", "tcompile_us")
_REQUIRED_LATENCY_FIELDS = ("emmy_us",)


def _validate_latency(latency: object, where: str) -> None:
    if not isinstance(latency, Mapping) or not latency:
        raise ValueError(f"{where} must be a non-empty mapping of hardware id to timings")
    for card, timings in latency.items():
        if not isinstance(card, str) or not card:
            raise ValueError(f"{where} keys must be non-empty hardware ids")
        if not isinstance(timings, Mapping):
            raise ValueError(f"{where}.{card} must be a mapping")
        _require_keys(timings, set(LATENCY_FIELDS), f"{where}.{card}")
        for field in _REQUIRED_LATENCY_FIELDS:
            if field not in timings:
                raise ValueError(f"{where}.{card} missing {field}")
        for field in LATENCY_FIELDS:
            if field in timings:
                _positive_number(timings[field], f"{where}.{card}.{field}")


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
            _require_keys(
                realization,
                {"name", "bindings", "pins", "knobs", "measurements", "ranking", "identity", "latency"},
                realization_where,
            )
            if not isinstance(realization.get("name"), str) or not realization["name"]:
                raise ValueError(f"{realization_where}.name must be a non-empty string")
            if "identity" in realization:
                identity = realization["identity"]
                if not isinstance(identity, str) or re.fullmatch(r"[0-9a-f]{64}", identity) is None:
                    raise ValueError(f"{realization_where}.identity must be a 64-character lowercase hexadecimal digest")
            if "latency" in realization:
                _validate_latency(realization["latency"], f"{realization_where}.latency")
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
                if strict and family_of(name) in {"WORK", "TILE", "REDUCE", "STAGE", "RASTER"}:
                    from emmy.compiler.pipeline.knob import validate_family_value  # noqa: PLC0415

                    try:
                        validate_family_value(name, value)
                    except ValueError as exc:
                        raise ValueError(f"{realization_where}.pins.{name}: {exc}") from exc
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
                if families and "PLACE" not in families and pins_freeze_cut(pins) and "identity" not in realization:
                    raise ValueError(
                        f"{realization_where} schedules a kernel behind pinned cut(s) without naming it; "
                        "a child-identity schedule receipt must store the child kernel's identity"
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
        identity=realization.get("identity"),
        latency=dict(realization["latency"]) if realization.get("latency") is not None else None,
    )


def load_golden_records(document: Mapping) -> list[GoldenRecord]:
    return [
        golden_record_from_entry(document, entry, realization) for entry in document["configs"] for realization in entry["realizations"]
    ]


def regime_pins(record: GoldenRecord) -> dict:
    """The record's INPUT pin regime — its pins minus the route: the precision knobs (``FAST_MATH``
    and friends) a replay publishes to the environment so the record reads as live evidence
    (:func:`regime_live`). The schedule row and the route never travel this way; they are
    measured rows the evidence pick joins to the kernel they were recorded for."""
    return {str(key): value for key, value in record.pins if str(key).split("@", 1)[0] != "PLACE"}


def shared_regime_pins(records: Sequence[GoldenRecord]) -> dict:
    """The one input regime every record shares, or ``{}`` when they disagree — a compile publishes
    a regime only when the records it replays agree on it, because choosing one would silently
    change which realization was requested."""
    regimes = {tuple(sorted(regime_pins(record).items())) for record in records}
    return dict(regimes.pop()) if len(regimes) == 1 else {}


_STRUCTURAL_CACHE: dict[tuple, tuple[tuple[str, float], ...]] = {}
_IDENTITY_CACHE: dict[tuple, str | None] = {}
#: The persisted identity memo: {record fingerprint: identity | None}, valid only under one
#: compiler fingerprint. Purely derived data — a stale or missing store just re-derives.
_IDENTITY_STORE: dict | None = None
_IDENTITY_STORE_DIRTY: bool = False
#: Wire payload digests, memoized per payload OBJECT. The value holds the wire itself, not just
#: its digest: an ``id()``-keyed memo whose entry outlives the object it describes answers for
#: whatever later lands at that address, and this memo feeds a record's identity fingerprint.
#: Keeping the reference is what makes the address stable, and it matches how every sibling
#: cache in this module (``_PROGRAM_GRAPH_CACHE``, ``_LOOP_GRAPH_CACHE``) is written.
_WIRE_DIGESTS: dict[int, tuple[dict, str]] = {}


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
        sections: dict = {"entries": {}, "verdicts": {}, "replays": {}}
        try:
            payload = json.loads(config.golden_identity_cache_path().read_text())
            if payload.get("fingerprint") == fingerprint:
                sections = {name: payload.get(name, {}) for name in sections}
        except (OSError, ValueError):
            pass
        _IDENTITY_STORE = {"fingerprint": fingerprint, **sections}
    return _IDENTITY_STORE


def flush_identity_store() -> None:
    """Persist newly derived identities, decode verdicts and evidence replays (atomic replace;
    concurrent writers merge — a lost write only re-derives later). Called once an evidence
    import has derived what it needed (:func:`evidence_rows`), so the next process on this
    machine and compiler reads the derivations instead of replaying every record."""
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
            for section in ("entries", "verdicts", "replays"):
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
    cached = _WIRE_DIGESTS.get(id(wire))
    if cached is None or cached[0] is not wire:
        cached = (wire, digest(json.dumps(wire, sort_keys=True, default=str)))
        _WIRE_DIGESTS[id(wire)] = cached
    return digest(cached[1], str(record.target_key), str(record.bindings), str(record.compute_cap), record.gpu_name or "")


#: One :class:`_Replay` per exact target, pins and spelled knobs — the tripwire and the evidence
#: import walk whole files, and sibling realizations that spell the same kernel-set decisions share
#: one replay. Context construction is also shared per card; neither cache changes schedule-space
#: membership.
_REPLAY_CACHE: dict[tuple, _Replay] = {}
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
        # One kernel per PRODUCER, not per output: a multi-output kernel (an NVFP4 re-encode emits
        # packed codes beside their block scales) produces several of the graph's outputs, and
        # counting it once per output made a single-kernel target read as "lowers to N kernels".
        producers = (lowered.producer(output) for output in lowered.outputs)
        nodes = list({node.id: node for node in producers if node is not None and isinstance(node.op, LoopOp)}.values())
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


def _lifted_target(record: GoldenRecord):
    """Lift the record's single selected kernel to Tile IR — the tree the cut pass schedules: the
    lift, then the twist rewrite, exactly as ``lowering/tile`` runs them. A placement key is
    spelled on that tree, so decoding it against the lift alone would name sites the fused
    single-pass carrier no longer has."""
    from emmy.compiler.pipeline.passes.lowering.tile._fromloop import lift_loop_op  # noqa: PLC0415
    from emmy.compiler.pipeline.passes.lowering.tile._twist import rewrite_twisted  # noqa: PLC0415

    lowered, nodes = _target_kernel_nodes(record)
    if len(nodes) != 1:
        raise ValueError(f"{record.name}: target lowers to {len(nodes)} kernels — a row decorates exactly one")
    node = nodes[0]
    node.op = node.op.with_io(lowered, node)
    tile = lift_loop_op(node.op, name=node.id)
    tile = replace(tile, op=rewrite_twisted(tile.op, tile.axes))
    # A fork's root op is always matcher-refreshed (``_match_at`` runs ``with_io`` on every matched
    # node before the rule that offers the fork), so the record side mirrors the io through that
    # same call rather than a hand-rolled map: a multi-output kernel — an NVFP4 re-encode emits
    # packed codes beside their block scales — is bound to every one of the node's output buffers,
    # and the dtype half of the deploy identity (``identity_key(with_io=True)``) reads the same
    # output fingerprint on both sides.
    return tile.with_io(lowered, node)


def decode_record(record: GoldenRecord, siblings: Sequence[GoldenRecord] = ()) -> str | None:
    """STRICTLY decode one record against the current compiler — ``None`` on success, else the
    failure reason. This is the replayability contract the nightly onboarding job gates: the persisted
    program selects exactly one kernel, except that a child-identity schedule receipt may select its
    kernel from a multi-kernel target by stored identity; a routing record's every cut key names a
    seam the cut pass offers on the replay (:func:`_replay`); a SCHEDULE record's spelled row equals
    one enumerated leaf (``canonical_row_key`` equality under the record's own pins) — no prefix
    matching, no any-of, no classified shape. A receipt's
    identity must equal one kernel resolved under the record's pins, and the spelled row must equal
    one of THAT kernel's rows — a sibling child's row must not vouch for it."""
    from emmy.compiler.pipeline.knob import schedule_row_key  # noqa: PLC0415

    verdict_key = digest(_record_fingerprint(record), str(sorted(record.knobs.items())), str(record.pins), record.identity or "")
    store = _identity_store()
    verdicts = store.setdefault("verdicts", {})
    if verdict_key in verdicts:
        return verdicts[verdict_key]
    tile = None
    try:
        tile = _lifted_target(record)
    except Exception as exc:  # noqa: BLE001 — the reason IS the product here
        if not record.is_receipt:
            return _remember_verdict(verdict_key, f"{type(exc).__name__}: {exc}")
    replay = _replay(record, siblings=siblings, exhaustive=True)
    if record.is_routing:
        reason = f"routing key {replay.unresolved[0]!r} does not resolve to an offered cut seam" if replay.unresolved else None
        return _remember_verdict(verdict_key, reason)
    candidates = replay.rows
    row = schedule_row_key(record.knobs)
    if record.is_receipt and (tile is None or record.identity != tile.identity_key(with_io=True)):
        child_rows = candidates.get(record.identity)
        if child_rows is None:
            reason = f"stored identity equals none of the {len(candidates)} kernel identities resolved under the record's pins"
        elif row in child_rows:
            reason = None
        else:
            reason = f"no enumerated row of the identified kernel equals the recording ({len(child_rows)} candidate rows)"
    else:
        pooled = frozenset().union(*candidates.values()) if candidates else frozenset()
        reason = None if row in pooled else f"no enumerated row equals the recording ({len(pooled)} candidate rows)"
    verdicts[verdict_key] = reason
    global _IDENTITY_STORE_DIRTY
    _IDENTITY_STORE_DIRTY = True
    return reason


def _remember_verdict(key: str, reason: str | None) -> str | None:
    global _IDENTITY_STORE_DIRTY
    _identity_store().setdefault("verdicts", {})[key] = reason
    _IDENTITY_STORE_DIRTY = True
    return reason


class _Replay(NamedTuple):
    """One replay of a record's target through the tile passes under the record's pins, following
    the record's knobs at every kernel-set fork (:func:`~emmy.compiler.pipeline.search.pins.spelled_arm`).

    ``rows`` — the EXHAUSTIVE replay's answer: every schedule-row identity each kernel can realize,
    bucketed by the kernel's deploy identity (``identity_key(with_io=True)``; ``None`` for forks
    whose root is not a recognized ``TileOp``): the fork leaves' rows, PLUS each resolved kernel's
    own realized row — a forkless kernel (the schedule space collapsed to one row, often the all-OFF
    anchor) never opens a fork, so its one row is read off the resolved op instead. Behind a cut the
    buckets are exactly the pieces, which is what lets a child-identity receipt decode against its
    own kernel only. ``holders`` — the evidence replay's answer to the one question the index
    needs of the same enumeration: the kernels whose enumeration admits the record's piece row
    (:func:`piece_row`), found by the deploy's own descent (``fork.leaf_for``) instead of by
    flattening every pool. ``signatures`` — each resolved kernel's ``S_*`` signature by identity:
    how a row is keyed as evidence for the kernel it decorates. ``arms`` — the arm the record's
    route and knobs spelled at each kernel-set fork it decided (a cut seam, a cross-CTA plan),
    keyed by the signature of the kernel that fork was offered on: the record's route rows.
    ``unresolved`` — the record's scoped cut keys no offered seam carried, the strict decode's
    routing failure."""

    rows: dict[str | None, frozenset]
    holders: frozenset[str]
    #: The kernels the replay scheduled — those that reached a schedule fork or were resolved
    #: without one; a kernel a cut or split consumed is not among them.
    kernels: frozenset[str]
    signatures: dict[str, frozenset]
    arms: tuple[tuple[frozenset, dict[str, str]], ...]
    unresolved: tuple[str, ...]
    #: Each scheduled kernel's realized schedule row (``schedule_row_key`` families), by identity —
    #: what a per-kernel entry for a kernel the set leaves undescribed would record.
    realized: dict[str, dict[str, str]]


def piece_row(row: Mapping[str, str]) -> dict[str, str]:
    """A record's schedule row as a piece of its kernel set can carry it: a ``REDUCE`` value reduced
    to what a piece can still stamp (:func:`~emmy.compiler.pipeline.search.pins.stampable_reduce`),
    since the cross-CTA split it names was the parent's decision."""
    from emmy.compiler.pipeline.knob import family_of  # noqa: PLC0415
    from emmy.compiler.pipeline.search.pins import stampable_reduce  # noqa: PLC0415

    out = {str(key): str(value) for key, value in row.items()}
    for key, value in list(out.items()):
        if family_of(key) == "REDUCE" and (rest := stampable_reduce(value)) is not None:
            del out[key]
            if rest:
                out[key] = rest
    return out


def siblings_of(record: GoldenRecord, records: Sequence[GoldenRecord]) -> tuple[GoldenRecord, ...]:
    """The other records of ``record``'s target among ``records`` — same persisted target, bindings
    and input regime: the entries that walk one kernel set together (a case's per-kernel entries,
    a golden config's receipts). The first of them in ``records`` order is the set's lead."""
    key = _set_key(record)
    return tuple(other for other in records if other is not record and _set_key(other) == key)


def lead_of(record: GoldenRecord, records: Sequence[GoldenRecord]) -> GoldenRecord:
    """The set's leading entry — the first record of ``record``'s target in ``records`` order, the
    target's own entry: it decides every fork no entry names by identity."""
    key = _set_key(record)
    return next(other for other in records if _set_key(other) == key)


def _set_key(record: GoldenRecord) -> tuple:
    from emmy.compiler.pipeline.knob import family_of  # noqa: PLC0415

    regime = tuple(sorted((str(k), str(v)) for k, v in record.pin_map.items() if family_of(str(k)) != "PLACE"))
    return (_record_cache_key(record), regime)


def _replay(
    record: GoldenRecord, *, siblings: Sequence[GoldenRecord] = (), lead: GoldenRecord | None = None, exhaustive: bool = False
) -> _Replay:
    """Replay ``record``'s target through the tile passes — see :class:`_Replay`. The record's input
    pins are the regime it was measured under and go to the environment; its route (the ``PLACE``
    keys of its pins and knobs) and its knobs are the decisions it took at forks and are followed
    fork by fork. A piece a cut or split mints is a brand-new kernel: it inherits nothing, and the
    record's remaining keys are read against its own offers, exactly as the deploy reads a row of
    its signature.

    ``siblings`` are the other entries of the same target (:func:`siblings_of`) and ``lead`` the
    set's leading entry (:func:`lead_of`; the record itself when absent). A fork offered on a kernel
    one entry names by ``identity`` is decided by THAT entry's spelling; every other fork by the
    lead's — never by an entry that does not own it, whose row would say "fused" or "unsplit" of a
    kernel it never described. So a set of per-kernel entries — the parent's cut, each piece's
    row — walks one path together, and the record's own rows are what this replay reports.
    ``exhaustive`` flattens every schedule pool for ``rows`` (the strict decode's question); the
    evidence import asks only ``holders`` and descends."""
    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.compiler.ir.tile import TileOp  # noqa: PLC0415
    from emmy.compiler.pipeline import TILE_PASSES, Pipeline  # noqa: PLC0415
    from emmy.compiler.pipeline.fork import flatten_leaves, fork_signature, iter_leaves, leaf_for, leaf_knobs  # noqa: PLC0415
    from emmy.compiler.pipeline.knob import (  # noqa: PLC0415
        canonical_row_key,
        evidence_row_vouches,
        family_of,
        schedule_pin_fingerprint,  # noqa: PLC0415
        schedule_row_key,
    )
    from emmy.compiler.pipeline.pipeline import Run, _is_structural_option  # noqa: PLC0415
    from emmy.compiler.pipeline.search.pins import pinned_knobs, spelled_arm  # noqa: PLC0415

    def _spelling(entry: GoldenRecord) -> dict[str, str]:
        return {**entry.route, **{str(key): str(value) for key, value in entry.knobs.items()}}

    lead = record if lead is None else lead
    named = {entry.identity: entry for entry in siblings if entry.identity is not None}
    if record.identity is not None:
        named[record.identity] = record
    set_digest = digest(
        f"lead:{sorted(_spelling(lead).items())}",
        *(f"{identity}:{sorted(_spelling(entry).items())}" for identity, entry in sorted(named.items())),
    )
    cache_key = (_record_cache_key(record), record.pins, canonical_row_key(record.knobs), set_digest, exhaustive)
    cached = _REPLAY_CACHE.get(cache_key)
    if cached is not None:
        return cached
    # The evidence replay is a pure function of the record, its set, the compiler and the live
    # enumeration pins, so it persists beside identities and verdicts; the exhaustive one stays in
    # memory.
    store = _identity_store()["replays"]
    store_key = digest(
        _record_fingerprint(record),
        str(sorted(record.knobs.items())),
        str(record.pins),
        record.identity or "",
        set_digest,
        str(schedule_pin_fingerprint()),
    )
    if not exhaustive and (kept := store.get(store_key)) is not None:
        result = _Replay(
            {},
            frozenset(kept["holders"]),
            frozenset(kept["kernels"]),
            {identity: frozenset(tuple(pair) for pair in signature) for identity, signature in kept["signatures"].items()},
            tuple((frozenset(tuple(pair) for pair in signature), dict(arm)) for signature, arm in kept["arms"]),
            tuple(kept["unresolved"]),
            {identity: dict(row) for identity, row in kept["realized"].items()},
        )
        _REPLAY_CACHE[cache_key] = result
        return result
    ctx_key = (record.compute_cap, record.gpu_name or None)
    ctx = _DECODE_CTX_CACHE.get(ctx_key)
    if ctx is None:
        ctx = _DECODE_CTX_CACHE.setdefault(ctx_key, Context.from_target(ctx_key[0], gpu_name=ctx_key[1]))
    spelled = _spelling(record)
    regime = {key: value for key, value in record.pin_map.items() if family_of(str(key)) != "PLACE"}
    pending = {key for key, value in spelled.items() if family_of(key) == "PLACE" and value == "cut"}
    piece = piece_row(record.schedule_row)
    buckets: dict[str | None, set] = {}
    holders: set[str] = set()
    kernels: set[str] = set()
    signatures: dict[str, frozenset] = {}
    realized: dict[str, dict[str, str]] = {}
    arms: list[tuple[frozenset, dict[str, str]]] = []

    def _identity_of(op) -> str | None:
        return op.identity_key(with_io=True) if isinstance(op, TileOp) else None

    def _note(op, signature: frozenset) -> None:
        identity = _identity_of(op)
        if identity is not None:
            signatures.setdefault(identity, signature)

    def decide(fp):
        # The kernel's signature as the deploy reads it at this fork — an op resolved without a
        # fork is keyed below, off its own stamp.
        signature = fork_signature(fp.root_op, fp.options, ctx)
        _note(fp.root_op, signature)
        identity = _identity_of(fp.root_op)
        owner = named.get(identity) if identity is not None else None
        decider = owner if owner is not None else lead
        if fp.structural:
            arm = spelled_arm(fp.options, spelled if decider is record else _spelling(decider))
            if arm is not None:
                option, knobs = arm
                if _is_structural_option(option) and decider is record:
                    # A cut consumes the key that spelled it (a bare ``PLACE=cut`` its one root-most
                    # cut), so the pieces are read against what the record has left to say.
                    arms.append((signature, dict(knobs)))
                    for key in (*(pending & set(knobs)), *(("PLACE",) if spelled.get("PLACE") == "cut" else ())):
                        pending.discard(key)
                        spelled.pop(key, None)
                return option
        if identity is not None:
            kernels.add(identity)
        if not exhaustive:
            asked = piece if decider is record else piece_row(decider.schedule_row)
            hit = leaf_for(fp.options, asked) if asked else None
            if hit is not None and identity is not None and decider is record:
                holders.add(identity)
            return hit[0] if hit is not None else next(iter_leaves(fp.options))
        leaves = flatten_leaves(fp.options)
        ops = [o for o in leaves if not _is_structural_option(o)]
        for leaf in ops:
            row = leaf_knobs(leaf)
            if row:
                buckets.setdefault(identity, set()).add(schedule_row_key(row))
        return ops[0] if ops else leaves[0]

    with pinned_knobs(regime):
        out, _ = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=ctx).resolve(record.target_program.copy(), decide)
    for node in out.nodes.values():
        if isinstance(node.op, TileOp):
            identity = _identity_of(node.op)
            row = schedule_row_key(dict(node.op.knobs or {}))
            if exhaustive:
                buckets.setdefault(identity, set()).add(row)
            if identity is not None:
                kernels.add(identity)
                realized[identity] = dict(row)
                if piece and (named.get(identity) or lead) is record and evidence_row_vouches(dict(row), piece):
                    holders.add(identity)  # a forkless kernel: its one row is the resolved op's
            _note(node.op, fork_signature(node.op, (), ctx))
    result = _Replay(
        {identity: frozenset(rows) for identity, rows in buckets.items()},
        frozenset(holders),
        frozenset(kernels),
        signatures,
        tuple(arms),
        tuple(sorted(pending)),
        realized,
    )
    _REPLAY_CACHE[cache_key] = result
    if not exhaustive:
        global _IDENTITY_STORE_DIRTY
        store[store_key] = {
            "holders": sorted(holders),
            "kernels": sorted(kernels),
            "signatures": {identity: sorted(signature) for identity, signature in signatures.items()},
            "arms": [[sorted(signature), arm] for signature, arm in arms],
            "unresolved": sorted(pending),
            "realized": realized,
        }
        _IDENTITY_STORE_DIRTY = True
    return result


def kernel_identity(record: GoldenRecord) -> str | None:
    """The record's kernel identity under the CURRENT compiler — the strict decode's and the drift
    key (``identity_key(with_io=True)``). A STORED identity is returned as-is: it is how a
    child-identity receipt names the one split child its schedule decorates (the target's own lift
    stops at the pre-cut kernel and cannot say), and a stale stored identity selects nothing — the
    strict decode is where that fails loudly. Without one, the identity is derived as the lift of the
    record's ONE target kernel, through the exact total lift the live compile uses
    (``_fromloop.lift_loop_op``). ``None`` when the record cannot carry a deploy identity: the target
    lowers to several kernels (a schedule row decorates exactly one), or selection/lifting fails —
    best-effort here (a corpus row must never break a compile); nightly strict decoding is where
    failure is loud. Deploy never joins on this key: a record deploys as measured rows, matched by
    ``S_*`` signature (:func:`evidence_rows`)."""
    global _IDENTITY_STORE_DIRTY
    if record.identity is not None:
        return record.identity
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
        identity = _lifted_target(record).identity_key(with_io=True)
    except Exception:  # noqa: BLE001 — see the docstring; the decode tripwire re-derives loudly
        identity = None
    _IDENTITY_CACHE[key] = identity
    store["entries"][fingerprint] = identity
    _IDENTITY_STORE_DIRTY = True
    return identity


def _derive_structural_features(record: GoldenRecord) -> tuple[tuple[str, float], ...]:
    """Lower the exact replay target and recover its unique ``S_*`` row."""
    payload_id = id(record.loop_wire) if record.loop_wire is not None else id(record.program_wire)
    key = (payload_id, record.target_key, record.compute_cap, record.bindings)
    cached = _STRUCTURAL_CACHE.get(key)
    if cached is not None:
        return cached

    from emmy.compiler.pipeline.knob import STRUCT_PREFIX  # noqa: PLC0415

    _lowered, nodes = _target_kernel_nodes(record)
    signatures = {
        tuple(
            sorted((name, float(value)) for name, value in (getattr(node.op, "knobs", {}) or {}).items() if name.startswith(STRUCT_PREFIX))
        )
        for node in nodes
    }
    signatures.discard(())
    if len(signatures) != 1:
        raise ValueError(f"{record.name}: target resolves to {len(signatures)} structural targets")
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


#: Optional scope override for :func:`records_for_card` — the golden rows the evidence index loads.
#: ``None`` (the default) reads ``EMMY_GOLDEN_FILE`` when set, else the repository files. Every
#: command that names a golden scopes the rows here: ``run`` / ``compile`` install the selected
#: records in-process, the release gate (``eval golden --serving-config``) one precision lane's
#: records through :func:`sole_evidence`, and ``serve --golden`` reaches the same loader through
#: the env var because the vLLM child is another process. Set it through :func:`records_override`,
#: never by hand.
RECORDS_OVERRIDE: list[GoldenRecord] | None = None


@contextmanager
def records_override(records: list[GoldenRecord] | None):
    """Scope the golden rows :func:`records_for_card` supplies to the evidence index, restoring
    the previous scope after. ``[]`` hides every record — how a caller that must measure without
    golden evidence (the tuner) says so; ``None`` is a no-op, leaving whatever scope is already
    installed.

    **The body must not ``await``.** This swaps a module global, so it is only atomic with respect
    to other coroutines while the block stays synchronous — and it is used inside concurrently
    gathered tune targets, which share one event loop."""
    global RECORDS_OVERRIDE  # noqa: PLW0603 — the documented scope seam, one owner
    if records is None:
        yield
        return
    prev = RECORDS_OVERRIDE
    RECORDS_OVERRIDE = records
    try:
        yield
    finally:
        RECORDS_OVERRIDE = prev


@contextmanager
def sole_evidence(records: list[GoldenRecord]):
    """``records`` as a compile's ONLY evidence, strictly: the golden scope is these rows
    (:func:`records_override`), the machine-local online prior and its reservoir are out of the
    way (``EMMY_ONLINE_FILE`` at a nonexistent path) and strict evidence is on, so a fork none of
    the rows decides is an ``EvidenceError`` naming the kernel instead of a prediction; a
    ``Pipeline.run`` given no ``db`` consults no tune DB either. The release gate (``eval golden
    --serving-config``) and the realization corpus ask their question inside this, which is what
    makes the answer the same on every machine that holds the same rows."""
    with tempfile.TemporaryDirectory(prefix="emmy-evidence-") as tmp:
        with (
            records_override(records),
            config.online_file_override(Path(tmp) / "absent-online.json"),
            config.strict_evidence_override(True),
        ):
            yield


def scope_explicit() -> bool:
    """Whether a caller scoped the golden evidence to records of its own choosing — an in-process
    override or ``EMMY_GOLDEN_FILE`` — rather than the repository corpus."""
    return RECORDS_OVERRIDE is not None or config.golden_scope() is not None


def scope_token() -> object:
    """A hashable stamp of the installed golden scope, for the evidence index's process memo."""
    if RECORDS_OVERRIDE is not None:
        return ("override", id(RECORDS_OVERRIDE), len(RECORDS_OVERRIDE))
    path = config.golden_file()
    return ("file", str(path)) if path is not None else ("repository",)


def _scoped(records: Sequence[GoldenRecord], gpu_name: str, compute_cap: tuple[int, int]) -> list[GoldenRecord]:
    """An explicit scope's records for one card: the capability must agree; a record that names
    no card (a working golden traced off-GPU) applies to whichever card compiles it."""
    return [r for r in records if tuple(r.compute_cap) == tuple(compute_cap) and (not r.gpu_name or r.gpu_name == gpu_name)]


def records_for_card(gpu_name: str, compute_cap: tuple[int, int]) -> list[GoldenRecord]:
    """The golden records the evidence index loads for ONE card: the installed scope when one is set
    (:data:`RECORDS_OVERRIDE`, else ``EMMY_GOLDEN_FILE`` — a file, or none when set empty), otherwise
    the repository files, loading only that card's (header sniff). ``GOLDEN_RECORDS`` stays the full corpus for the eval / fit
    consumers; both share the per-path document memo so nothing parses twice."""
    if RECORDS_OVERRIDE is not None:
        return _scoped(RECORDS_OVERRIDE, gpu_name, compute_cap)
    if (scope := config.golden_scope()) is not None:
        # A path scopes the evidence to that file; the empty form (``EMMY_GOLDEN_FILE=``) is no golden evidence.
        return _scoped(_records_of(Path(scope), validation=GoldenFileValidation.WORKING), gpu_name, compute_cap) if scope else []
    records: list[GoldenRecord] = []
    with _repository_golden_paths() as paths:
        for path in paths:
            head_gpu = _file_gpu_name(path)
            if head_gpu is not None and head_gpu != gpu_name:
                continue
            records.extend(r for r in _records_of(path) if r.gpu_name == gpu_name and tuple(r.compute_cap) == tuple(compute_cap))
    return records


#: The precision-trading pin universe the regime check covers in BOTH directions — a record
#: that omits one of these was measured with it OFF, and must not deploy when it is live-ON.
_PRECISION_PINS = ("FAST_MATH", "FAST_EXP", "F16_MMA_F32_ACC", "FP8_MMA")


def regime_live(record: GoldenRecord) -> bool:
    """Whether the record's input-pin regime IS the live one — exact per pin: a BOOL pin compares
    against the live env pin (unset = the knob's off state), anything else against the raw env
    string. Strict BOTH ways: a record measured under FAST_MATH is no evidence for a standard
    deploy, and a standard record none under a live precision-trading pin — the precision universe
    (:data:`_PRECISION_PINS`, umbrella semantics per ``space.precision_pin``) is compared even for
    pins the record omits (omitted = measured OFF). ``PLACE`` pins are the record's route, not a
    regime."""
    from emmy.compiler.pipeline.knob import KnobType, family_of, registry  # noqa: PLC0415
    from emmy.compiler.pipeline.search.space import precision_pin  # noqa: PLC0415

    knobs = registry()
    pins = record.pin_map
    for name, value in pins.items():
        if family_of(str(name)) == "PLACE":
            continue
        kn = knobs.get(str(name))
        raw = kn.raw() if kn is not None else config.knob_raw(str(name))
        if kn is not None and kn.type is KnobType.BOOL:
            live = kn.parse(raw) if raw is not None else False
            if bool(value) != live:
                return False
        elif (raw or "") != str(value):
            return False
    umbrella = bool(pins.get("FAST_MATH", False))
    for name in _PRECISION_PINS:
        recorded = bool(pins.get(name, umbrella))
        kn = knobs.get(name)
        live = bool(precision_pin(kn)) if kn is not None else False
        if recorded != live:
            return False
    return True


def evidence_rows(gpu_name: str, compute_cap: tuple[int, int]) -> list[tuple[frozenset, dict, float, str]]:
    """The golden rows in scope as measured-evidence rows for one card: ``(S_* signature, tuning
    knobs, µs, record name)``, the same shape the tune DB's ``perf`` rows take in the deploy's
    evidence index. Only a MEASURED record in the live input regime (:func:`regime_live`) is
    evidence; a proposal has no µs to rank with and deploys once ``run --golden PATH --bench``
    has measured it.

    Every row is keyed by the kernel it decides — a piece a cut or split mints is a brand-new
    kernel, so nothing a record says about the kernel it was offered on reaches the pieces. A
    record that decorates the target's one kernel (no stored identity, no route, no cross-CTA
    split) is that kernel's schedule row under the target's signature. Any other record is read through its
    replay (:func:`_replay`): each kernel-set arm it spelled is a route row under the signature of
    the kernel that fork was offered on, and its schedule row is keyed under the kernel its stored
    identity names when that kernel is one the replay resolved. Otherwise the row speaks for the
    kernel set collectively — a row the tuner merged with the parent's split, a case authored
    behind a cut — and is keyed under every piece whose enumerated rows it vouches for
    (``evidence_row_vouches``), its ``REDUCE`` value reduced to what a piece can still stamp
    (:func:`~emmy.compiler.pipeline.search.pins.stampable_reduce`): the split it names was the
    parent's decision. A row no kernel of the replay enumerates is stale and is no evidence.
    Best-effort per record: a record the current compiler cannot lower is skipped, since the
    strict decode is where that is loud."""
    from emmy.compiler.pipeline.knob import family_of  # noqa: PLC0415
    from emmy.compiler.pipeline.search.pins import parse_reduce  # noqa: PLC0415

    rows: list[tuple[frozenset, dict, float, str]] = []
    records = records_for_card(gpu_name, compute_cap)
    for record in records:
        if record.measurements is None or record.emmy_us <= 0 or not regime_live(record):
            continue
        row = record.schedule_row
        split = any(family_of(k) == "REDUCE" and (plan := parse_reduce(v)) is not None and plan.needs_split for k, v in row.items())
        if record.identity is None and not record.route and not split:
            try:
                signature = frozenset((key, str(value)) for key, value in record.structural_features.items())
            except Exception:  # noqa: BLE001 — a stale record is no evidence, not an error
                continue
            if row:
                rows.append((signature, row, record.emmy_us, record.name))
            continue
        try:
            replay = _replay(record, siblings=siblings_of(record, records), lead=lead_of(record, records))
        except Exception:  # noqa: BLE001 — see above
            continue
        rows.extend((signature, arm, record.emmy_us, record.name) for signature, arm in replay.arms)
        if not row:
            continue
        if record.identity in replay.kernels:
            kernels = [record.identity]
        else:
            row, kernels = piece_row(row), sorted(replay.holders)
        rows.extend((replay.signatures[kernel], row, record.emmy_us, record.name) for kernel in kernels if kernel in replay.signatures)
    flush_identity_store()
    return rows


_DOCUMENT_MEMO: dict[Path, list[GoldenRecord]] = {}


def _records_of(path: Path, *, validation: GoldenFileValidation = GoldenFileValidation.REPOSITORY) -> list[GoldenRecord]:
    path = Path(path)
    cached = _DOCUMENT_MEMO.get(path)
    if cached is None:
        document = load_golden_file(path, validation=validation)
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
