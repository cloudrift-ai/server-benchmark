"""Mutable working-golden inventories, candidates, and ranking feedback.

This module owns the untrusted side of the golden YAML workflow: trace inventory
generation, target reconstruction, exact proposal measurement, and atomic ranking
persistence. CLI commands only validate argument combinations and report errors.
"""

from __future__ import annotations

import copy
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from emmy.compiler.pipeline.search.golden import (
    GoldenEntryState,
    dump_golden_file,
    golden_entry_state,
    golden_record_from_entry,
    is_repository_golden_path,
    load_golden_file,
)


@dataclass
class WorkingGoldenTarget:
    """One deduplicated working-file target and its candidate rows."""

    label: str
    code: str | None
    input: str | None
    dynamic: list[str] | None
    bindings: dict[str, int] = field(default_factory=dict)
    pins: dict[str, object] = field(default_factory=dict)
    program: object | None = None
    entry_indexes: list[tuple[int, int]] = field(default_factory=list)
    proposals: list[tuple[tuple[int, int], dict]] = field(default_factory=list)


@dataclass(frozen=True)
class TraceInventoryResult:
    """Artifacts written for one trace-generated working inventory."""

    path: Path
    target_count: int


def preflight_trace_inventory(path: str | Path) -> Path:
    """Resolve a fresh trace-inventory destination and reject replacement."""
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing golden artifact: {destination}")
    return destination


def write_trace_inventory(
    graph,
    path: str | Path,
    *,
    model: str | None = None,
    ctx=None,
    force_loop_targets: bool = False,
    realizations: list[dict] | None = None,
    model_quant_digest: str | None = None,
) -> TraceInventoryResult:
    """Lower a trace through fusion and write a self-contained target inventory."""
    destination = preflight_trace_inventory(path)
    from emmy.compiler.context import Context  # noqa: PLC0415

    ctx = ctx or Context.probe()
    programs: list[dict] = []
    loops: list[dict] = []
    entries: list[dict] = []
    _append_trace_inventory(
        graph,
        ctx=ctx,
        programs=programs,
        loops=loops,
        entries=entries,
        force_loop_targets=force_loop_targets,
        realizations=realizations,
    )
    _dump_trace_inventory(
        destination,
        ctx=ctx,
        model=model,
        model_quant_digest=model_quant_digest,
        programs=programs,
        loops=loops,
        entries=entries,
    )
    return TraceInventoryResult(path=destination, target_count=len(entries))


def write_trace_inventories(
    graphs: dict[str, object],
    path: str | Path,
    *,
    model: str | None = None,
    ctx=None,
    realizations: list[dict] | None = None,
    model_quant_digest: str | None = None,
) -> TraceInventoryResult:
    """Combine named traces into one exact-Loop-IR working inventory.

    Serving capture emits many independent pre/post/expert graphs.  A directory of
    one-file-per-graph inventories is awkward to tune and, more importantly, easy to
    promote only partially.  This writer interns all of their programs and Loop IR
    targets into one self-contained artifact.  Identical Loop programs are recorded
    once: they are the same tuning target even when several serving twins consult it.
    """
    destination = preflight_trace_inventory(path)
    if not graphs:
        raise ValueError("cannot write an empty trace inventory")
    from emmy.compiler.context import Context  # noqa: PLC0415

    ctx = ctx or Context.probe()
    programs: list[dict] = []
    loops: list[dict] = []
    entries: list[dict] = []
    seen_loops: set[int] = set()
    for name in sorted(graphs):
        _append_trace_inventory(
            graphs[name],
            ctx=ctx,
            programs=programs,
            loops=loops,
            entries=entries,
            force_loop_targets=True,
            name_prefix=name,
            seen_loops=seen_loops,
            realizations=realizations,
        )
    _dump_trace_inventory(
        destination,
        ctx=ctx,
        model=model,
        model_quant_digest=model_quant_digest,
        programs=programs,
        loops=loops,
        entries=entries,
    )
    return TraceInventoryResult(path=destination, target_count=len(entries))


def _append_trace_inventory(
    graph,
    *,
    ctx,
    programs: list[dict],
    loops: list[dict],
    entries: list[dict],
    force_loop_targets: bool,
    name_prefix: str | None = None,
    seen_loops: set[int] | None = None,
    realizations: list[dict] | None = None,
) -> None:
    """Append one lowered graph to shared trace-inventory pools."""
    from emmy.compiler import provenance  # noqa: PLC0415
    from emmy.compiler.ir.loop import LoopOp  # noqa: PLC0415
    from emmy.compiler.loop_wire import intern_loop_program  # noqa: PLC0415
    from emmy.compiler.pipeline import LOOP_PASSES, Pipeline  # noqa: PLC0415
    from emmy.compiler.pipeline.passes.lowering.tile._flash import fused_producer_ids  # noqa: PLC0415
    from emmy.compiler.pipeline.search.slice import single_node_graph  # noqa: PLC0415
    from emmy.compiler.torch_wire import intern_program  # noqa: PLC0415

    # A birth-time speller may mark an internal storage value that has to remain
    # materialized for a faithful target inventory (dynamic activation bits and
    # scale are the first use). Promoting only the inventory copy to an auxiliary
    # graph output preserves the boundary without changing normal model outputs.
    for traced_node in graph.nodes.values():
        if traced_node.hints.get("trace.materialize") and traced_node.id not in graph.outputs:
            graph.outputs.append(traced_node.id)

    # Torch tracing and checkpoint spelling may hand us a graph that has already crossed one
    # compiler pipeline and therefore carries implementation-piece provenance. The stable wire
    # deliberately does not persist provenance hints, so retaining those here would write
    # selectors from one provenance universe and replay them after a fresh per-node seed — they
    # cannot match.
    # Re-seed the pristine frontend graph exactly as the wire decoder will.
    for traced_node in graph.nodes.values():
        traced_node.hints.remove(provenance.PROV)
    provenance.seed(graph)
    input_graph = graph.copy()
    fused = Pipeline.build(LOOP_PASSES).run(graph, ctx=ctx)

    # Match two-level tuning's fold-aware target set. A flash score producer is
    # part of its consuming attention target, not a second inventory row.
    absorbed: dict[str, str] = {}
    for node_id in fused.topological_order():
        node = fused.nodes[node_id]
        if not isinstance(node.op, LoopOp):
            continue
        for producer_id in fused_producer_ids(fused, node):
            absorbed[producer_id] = node_id

    targets: list[tuple[str, object]] = []
    for node_id in fused.topological_order():
        node = fused.nodes[node_id]
        if not isinstance(node.op, LoopOp) or node_id in absorbed:
            continue
        targets.append((node_id, node))

    # Persist the pristine program once.  Per-target frontend slices are useful
    # ephemeral tuning views, but they can change fusion when independently
    # lowered (especially sibling linears and computed-A cones).  Provenance
    # selectors must therefore resolve against the original trace context.
    program_ref: int | None = None
    inventory = []
    for node_id, node in targets:
        folded = [fused.nodes[producer_id] for producer_id, consumer in absorbed.items() if consumer == node_id]
        target_prov = provenance.union(*(provenance.get(item) for item in (node, *folded)))
        origins = tuple(sorted(origin for origin in target_prov if origin in input_graph.nodes))
        inventory.append((node_id, node, folded, origins))
    origin_counts = Counter(origins for _node_id, _node, _folded, origins in inventory if origins)
    used_names = {
        realization["name"]
        for entry in entries
        for realization in entry.get("realizations", [])
        if isinstance(realization, dict) and isinstance(realization.get("name"), str)
    }

    for node_id, node, folded, origins in inventory:
        key = node.op.cache_key()
        suffix = key[:12] if key is not None else node_id
        name = f"{node.op.name or node_id}.{suffix}"
        if name_prefix:
            name = f"{name_prefix}.{name}"
        if name in used_names:
            # One kernel body/cache key can occur at multiple exact Loop
            # targets whose boundary shapes or checkpoint sources differ.
            # ``emmy run --golden`` resolves by name, so retaining the bare
            # duplicate makes the generated file impossible to replay. Node
            # ids are deterministic within the persisted source program and
            # distinguish these otherwise same-bodied target sites.
            base = f"{name}.{node_id}"
            name = base
            duplicate = 2
            while name in used_names:
                name = f"{base}.{duplicate}"
                duplicate += 1
        used_names.add(name)
        if origins and origin_counts[origins] == 1 and not force_loop_targets:
            target = {"origins": list(origins)}
        else:
            loop_graph = single_node_graph(fused, node_id, absorb=frozenset(item.id for item in folded))
            loop_ref = intern_loop_program(loops, loop_graph)
            if seen_loops is not None and loop_ref in seen_loops:
                continue
            if seen_loops is not None:
                seen_loops.add(loop_ref)
            target = {"loop": loop_ref}
        if program_ref is None:
            program_ref = intern_program(programs, input_graph)
        if realizations is None:
            rows = [{"name": name, "bindings": {}, "pins": {"FAST_MATH": False}}]
        else:
            rows = []
            for template in realizations:
                row = copy.deepcopy(template)
                suffix = row.pop("name")
                row["name"] = f"{name}.{suffix}" if suffix else name
                rows.append(row)
        entry = {"program": program_ref, "target": target, "realizations": rows}
        entries.append(entry)


def _dump_trace_inventory(
    destination: Path,
    *,
    ctx,
    model: str | None,
    model_quant_digest: str | None,
    programs: list[dict],
    loops: list[dict],
    entries: list[dict],
) -> None:
    """Write shared trace-inventory pools with their card and model provenance."""
    document: dict = {
        "compute_cap": list(ctx.compute_capability),
        "programs": programs,
        "configs": entries,
    }
    if loops:
        document["loops"] = loops
    if ctx.gpu_name:
        document["gpu_name"] = ctx.gpu_name
    if model_quant_digest:
        document["model_quant_digest"] = model_quant_digest
    if model:
        document["model"] = model

    dump_golden_file(document, destination)


def load_working_targets(path: str | Path, *, kernel: str | None = None) -> tuple[dict, list[WorkingGoldenTarget]]:
    """Load a mutable YAML and reconstruct its deduplicated tune targets."""
    source = Path(path)
    if is_repository_golden_path(source):
        raise ValueError(f"working golden cannot point inside the canonical repository goldens: {source}")
    document = load_golden_file(source)

    by_source: dict[tuple[int, tuple, tuple[tuple[str, int], ...], tuple[tuple[str, object], ...]], WorkingGoldenTarget] = {}
    for index, entry in enumerate(document["configs"]):
        for realization_index, realization in enumerate(entry["realizations"]):
            if kernel and kernel not in realization["name"]:
                continue
            record = golden_record_from_entry(document, entry, realization)
            key = (record.program_index, record.target_key, record.bindings, record.pins)
            target = by_source.get(key)
            if target is None:
                target = WorkingGoldenTarget(
                    label=realization["name"],
                    code=None,
                    input=None,
                    dynamic=None,
                    bindings=record.binding_map,
                    pins=record.pin_map,
                    program=record.target_program,
                )
                by_source[key] = target
            path = (index, realization_index)
            target.entry_indexes.append(path)
            if "knobs" in realization:
                target.proposals.append((path, dict(realization["knobs"])))

    if not by_source:
        raise ValueError(f"no working golden targets matched --kernel {kernel!r}")
    return document, list(by_source.values())


def validate_working_gpu(document: dict, ctx) -> None:
    """Reject a working file recorded for a different concrete GPU."""
    file_cap = document.get("compute_cap")
    file_gpu = document.get("gpu_name")
    if file_cap is not None and tuple(file_cap) != (0, 0) and tuple(file_cap) != tuple(ctx.compute_capability):
        raise ValueError(
            f"working golden targets compute capability {tuple(file_cap)}, but the live GPU is {tuple(ctx.compute_capability)}"
        )
    if file_gpu and ctx.gpu_name and file_gpu != ctx.gpu_name:
        raise ValueError(f"working golden targets {file_gpu}, but the live GPU is {ctx.gpu_name}")


def realized_tuning_knobs(graph) -> dict[str, str] | None:
    """Conflict-free canonical tuning knobs across every CudaOp in ``graph``."""
    from emmy.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415
    from emmy.compiler.pipeline.knob import stamp_schedule_families  # noqa: PLC0415

    rows = [stamp_schedule_families(node.op.knobs) for node in graph.nodes.values() if isinstance(node.op, CudaOp)]
    if not rows:
        return None
    merged: dict[str, str] = {}
    for row in rows:
        for key, value in row.items():
            if key in merged and str(merged[key]) != str(value):
                return None
            merged[key] = value
    return merged


async def measure_proposals(graph, proposals, *, backend, db, ctx, max_candidates: int | None, prior=None) -> list[dict]:
    """Measure working-file candidates exactly, in file order, before MCTS."""
    from emmy.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415
    from emmy.compiler.pipeline import CUDA_PASSES, Pipeline, TuningSearch  # noqa: PLC0415
    from emmy.compiler.pipeline.search.pins import pinned_knobs, unreproducible_pin_flag  # noqa: PLC0415

    rankings: list[dict] = []
    limit = len(proposals) if max_candidates is None else min(len(proposals), max_candidates)
    for proposal_index, (_entry_index, pins) in enumerate(proposals):
        if proposal_index >= limit:
            rankings.append(
                {
                    "status": "skipped_budget",
                    "latency_us": None,
                    "compile_flags": ctx.compile_flags,
                    "measured_knobs": {},
                }
            )
            continue
        search = TuningSearch(
            patience=1,
            max_visits=1,
            max_measurements=1,
            prior_model=prior,
            base_knobs=ctx.features(),
        )
        terminal = None
        with pinned_knobs(pins):
            async for candidate in Pipeline.build(CUDA_PASSES).tune_async(graph.copy(), search=search, ctx=ctx, backend=backend, db=db):
                terminal = candidate
        if prior is not None:
            prior.add_rows(search._collect_rows() + search.o3_rows)
            prior.maybe_refit()
        raw_rows = [node.op.knobs for node in terminal.graph.nodes.values() if isinstance(node.op, CudaOp)] if terminal else []
        measured_knobs = realized_tuning_knobs(terminal.graph) if terminal is not None else None
        pin_error = unreproducible_pin_flag(pins, raw_rows) if raw_rows else "proposal produced no CUDA kernel"
        knob_error = None
        if raw_rows and measured_knobs is None:
            knob_error = f"proposal lowered to {len(raw_rows)} CUDA kernels with conflicting tuning knobs"
        status = "pin_unmatched" if pin_error else ("ambiguous_multi_kernel" if knob_error else (search.last_status or "bench_fail"))
        latency = search.last_stats.median if search.last_stats is not None and search.last_status == "ok" else None
        ranking = {
            "status": status,
            "latency_us": latency,
            "compile_flags": ctx.compile_flags,
            "measured_knobs": measured_knobs,
        }
        if pin_error or knob_error:
            ranking["error"] = pin_error or knob_error
        rankings.append(ranking)
    return rankings


def persist_proposal_rankings(path: str | Path, document: dict, target: WorkingGoldenTarget, rankings: list[dict]) -> None:
    """Atomically persist measured proposal feedback."""
    configs = document["configs"]
    for ((entry_index, realization_index), _pins), ranking in zip(target.proposals, rankings, strict=True):
        realization = configs[entry_index]["realizations"][realization_index]
        if golden_entry_state(realization) == GoldenEntryState.VERIFIED:
            continue
        realization["ranking"] = {**ranking, "source": "proposal"}
    dump_golden_file(document, path, overwrite=True)


def _kernel_set_knobs(value: object, *, where: str, placement: bool) -> dict:
    """Validate and copy one canonical string-valued knob map from a working tune result."""
    from emmy.compiler.pipeline.knob import KnobType, family_of, get  # noqa: PLC0415

    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be a mapping")
    out = {}
    for name, item in value.items():
        family = family_of(name) if isinstance(name, str) and name else ""
        descriptor = get(family)
        if descriptor is None:
            raise ValueError(f"{where} names unknown knob {name!r}")
        if (family == "PLACE") != placement:
            expected = "PLACE" if placement else "schedule"
            raise ValueError(f"{where} must contain only {expected} knobs")
        if not isinstance(item, str):
            raise ValueError(f"{where}.{name} must be a canonical string value, got {item!r}")
        if descriptor.type in {KnobType.BOOL, KnobType.INT}:
            try:
                descriptor.parse(item)
            except ValueError as exc:
                raise ValueError(f"{where}.{name} has invalid {descriptor.type.value} spelling {item!r}") from exc
        out[name] = item
    return out


def parse_kernel_set_ranking(ranking: object) -> dict:
    """Validate and normalize one exact working-only ``ranking.kernel_set``."""
    if not isinstance(ranking, Mapping):
        raise ValueError("ranking must be a mapping")
    if ranking.get("source") != "tune" or ranking.get("status") != "ok" or ranking.get("tune_winner") is not True:
        raise ValueError("kernel-set winner requires source=tune, status=ok, and tune_winner=true")
    if not isinstance(ranking.get("compile_flags"), str):
        raise ValueError("kernel-set winner compile_flags must be a string")
    if "measured_knobs" in ranking:
        raise ValueError("kernel-set winner cannot carry scalar measured_knobs")
    latency = ranking.get("latency_us")
    if isinstance(latency, bool) or not isinstance(latency, int | float) or not math.isfinite(latency) or latency <= 0:
        raise ValueError("kernel-set winner latency_us must be finite and positive")
    raw = ranking.get("kernel_set")
    if not isinstance(raw, Mapping):
        raise ValueError("ranking.kernel_set must be a mapping")
    unknown = set(raw) - {"placement", "kernels"}
    if unknown:
        raise ValueError(f"ranking.kernel_set has unknown field(s): {', '.join(sorted(unknown))}")
    placement = _kernel_set_knobs(raw.get("placement"), where="ranking.kernel_set.placement", placement=True)
    rows = raw.get("kernels")
    if not isinstance(rows, list) or not rows:
        raise ValueError("ranking.kernel_set.kernels must be a non-empty list")

    kernels = []
    seen = set()
    weighted = 0.0
    allowed = {"op_key", "multiplicity", "pins", "latency_us", "cuda_record_knobs"}
    for index, row in enumerate(rows):
        where = f"ranking.kernel_set.kernels[{index}]"
        if not isinstance(row, Mapping):
            raise ValueError(f"{where} must be a mapping")
        unknown = set(row) - allowed
        if unknown:
            raise ValueError(f"{where} has unknown field(s): {', '.join(sorted(unknown))}")
        op_key = row.get("op_key")
        if not isinstance(op_key, str) or not op_key:
            raise ValueError(f"{where}.op_key must be a non-empty string")
        if op_key in seen:
            raise ValueError(f"{where}.op_key duplicates {op_key!r}")
        seen.add(op_key)
        multiplicity = row.get("multiplicity")
        if type(multiplicity) is not int or multiplicity <= 0:
            raise ValueError(f"{where}.multiplicity must be a positive integer")
        component_us = row.get("latency_us")
        if (
            isinstance(component_us, bool)
            or not isinstance(component_us, int | float)
            or not math.isfinite(component_us)
            or component_us <= 0
        ):
            raise ValueError(f"{where}.latency_us must be finite and positive")
        pins = _kernel_set_knobs(row.get("pins"), where=f"{where}.pins", placement=False)
        cuda_rows = row.get("cuda_record_knobs")
        if not isinstance(cuda_rows, list) or not cuda_rows:
            raise ValueError(f"{where}.cuda_record_knobs must be a non-empty list")
        normalized_cuda = [
            _kernel_set_knobs(item, where=f"{where}.cuda_record_knobs[{cuda_index}]", placement=False)
            for cuda_index, item in enumerate(cuda_rows)
        ]
        weighted += float(component_us) * multiplicity
        kernels.append(
            {
                "op_key": op_key,
                "multiplicity": multiplicity,
                "pins": pins,
                "latency_us": float(component_us),
                "cuda_record_knobs": normalized_cuda,
            }
        )
    if not math.isclose(float(latency), weighted, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError(f"kernel-set winner latency_us {latency} does not equal weighted direct latency {weighted}")
    return {"placement": placement, "kernels": kernels}


def persist_tune_winner(
    path: str | Path,
    document: dict,
    target: WorkingGoldenTarget,
    winner: object | None,
    *,
    compile_flags: str,
) -> None:
    """Atomically persist one unambiguous directly searched winner, or fail closed.

    Scalar winners retain their legacy flat wire. A multi-kernel winner records exact placement and
    per-operation schedules. Only a tune with neither representation receives a target-scoped marker,
    so a strict publication run cannot mistake an older winner or unchanged inventory for success.
    """
    from emmy.compiler.pipeline.knob import canonical_row_key  # noqa: PLC0415
    from emmy.compiler.pipeline.search.two_level import KernelSetWinner  # noqa: PLC0415

    configs = document["configs"]
    marker_path = kernel_set_path = None
    for entry_path in target.entry_indexes:
        realization = configs[entry_path[0]]["realizations"][entry_path[1]]
        if golden_entry_state(realization) == GoldenEntryState.VERIFIED:
            continue
        ranking = realization.get("ranking")
        if not isinstance(ranking, dict) or ranking.get("source") != "tune":
            continue
        if winner is None and ranking.get("status") == "no_exact_pin" and marker_path is None:
            marker_path = entry_path
            continue
        if isinstance(winner, KernelSetWinner) and "kernel_set" in ranking and kernel_set_path is None:
            kernel_set_path = entry_path
            continue
        if ranking.get("tune_winner") is True or ranking.get("status") == "no_exact_pin":
            realization["ranking"] = {**ranking, "status": "superseded"}
            realization["ranking"].pop("tune_winner", None)

    if isinstance(winner, KernelSetWinner):
        winner_ranking = {
            "source": "tune",
            "status": "ok",
            "tune_winner": True,
            "compile_flags": compile_flags,
            "latency_us": winner.latency_us,
            "kernel_set": {
                "placement": dict(winner.placement),
                "kernels": [
                    {
                        "op_key": kernel.op_key,
                        "multiplicity": kernel.multiplicity,
                        "pins": dict(kernel.pins),
                        "latency_us": kernel.latency_us,
                        "cuda_record_knobs": [dict(row) for row in kernel.cuda_record_knobs],
                    }
                    for kernel in winner.kernels
                ],
            },
        }
        parse_kernel_set_ranking(winner_ranking)
        if kernel_set_path is None:
            config_index, realization_index = target.entry_indexes[0]
            seed = copy.deepcopy(configs[config_index]["realizations"][realization_index])
            for key in ("knobs", "measurements", "ranking"):
                seed.pop(key, None)
            seed["ranking"] = winner_ranking
            configs[config_index]["realizations"].append(seed)
            kernel_set_path = (config_index, len(configs[config_index]["realizations"]) - 1)
            target.entry_indexes.append(kernel_set_path)
        else:
            configs[kernel_set_path[0]]["realizations"][kernel_set_path[1]]["ranking"] = winner_ranking
    elif winner is not None:
        winner_knobs, winner_us = winner
        winner_ranking = {
            "status": "ok",
            "latency_us": winner_us,
            "compile_flags": compile_flags,
            "measured_knobs": winner_knobs,
            "source": "tune",
        }
        winner_key = canonical_row_key(winner_knobs)
        matching = [
            path
            for path in target.entry_indexes
            if "knobs" in configs[path[0]]["realizations"][path[1]]
            and canonical_row_key(configs[path[0]]["realizations"][path[1]]["knobs"]) == winner_key
        ]
        writable = next(
            (path for path in matching if golden_entry_state(configs[path[0]]["realizations"][path[1]]) != GoldenEntryState.VERIFIED),
            None,
        )
        if writable is not None:
            realization = configs[writable[0]]["realizations"][writable[1]]
            realization["ranking"] = {
                **winner_ranking,
                "tune_winner": True,
            }
        elif not matching:
            config_index, realization_index = target.entry_indexes[0]
            seed = copy.deepcopy(configs[config_index]["realizations"][realization_index])
            for key in ("knobs", "measurements", "ranking"):
                seed.pop(key, None)
            seed["knobs"] = winner_knobs
            seed["ranking"] = {**winner_ranking, "tune_winner": True}
            configs[config_index]["realizations"].append(seed)
            target.entry_indexes.append((config_index, len(configs[config_index]["realizations"]) - 1))
    else:
        marker = {
            "source": "tune",
            "status": "no_exact_pin",
            "compile_flags": compile_flags,
            "latency_us": None,
            "measured_knobs": None,
            "error": "tune produced no exact replayable winner",
        }
        if marker_path is None:
            config_index, realization_index = target.entry_indexes[0]
            seed = copy.deepcopy(configs[config_index]["realizations"][realization_index])
            for key in ("knobs", "measurements", "ranking"):
                seed.pop(key, None)
            seed["ranking"] = marker
            configs[config_index]["realizations"].append(seed)
            marker_path = (config_index, len(configs[config_index]["realizations"]) - 1)
            target.entry_indexes.append(marker_path)
        else:
            configs[marker_path[0]]["realizations"][marker_path[1]]["ranking"] = marker
    dump_golden_file(document, path, overwrite=True)
