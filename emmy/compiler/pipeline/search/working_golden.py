"""Mutable working-golden inventories, candidates, and ranking feedback.

This module owns the untrusted side of the golden YAML workflow: trace inventory
generation, target reconstruction, exact proposal measurement, and atomic ranking
persistence. CLI commands only validate argument combinations and report errors.
"""

from __future__ import annotations

import copy
from collections import Counter
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
    program: object | None = None
    entry_indexes: list[int] = field(default_factory=list)
    proposals: list[tuple[int, dict]] = field(default_factory=list)


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
) -> TraceInventoryResult:
    """Lower a trace through fusion and write a self-contained target inventory."""
    from emmy.compiler import provenance  # noqa: PLC0415
    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.compiler.ir.loop import LoopOp  # noqa: PLC0415
    from emmy.compiler.loop_wire import intern_loop_program  # noqa: PLC0415
    from emmy.compiler.pipeline import LOOP_PASSES, Pipeline  # noqa: PLC0415
    from emmy.compiler.pipeline.passes.lowering.tile._flash import fused_producer_ids  # noqa: PLC0415
    from emmy.compiler.pipeline.search.slice import single_node_graph  # noqa: PLC0415
    from emmy.compiler.torch_wire import intern_program  # noqa: PLC0415

    destination = preflight_trace_inventory(path)
    provenance.seed(graph)
    input_graph = graph.copy()
    ctx = ctx or Context.probe()
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

    programs: list[dict] = []
    # Persist the pristine program once.  Per-target frontend slices are useful
    # ephemeral tuning views, but they can change fusion when independently
    # lowered (especially sibling linears and computed-A cones).  Provenance
    # selectors must therefore resolve against the original trace context.
    program_ref = intern_program(programs, input_graph)
    loops: list[dict] = []
    entries: list[dict] = []
    inventory = []
    for node_id, node in targets:
        folded = [fused.nodes[producer_id] for producer_id, consumer in absorbed.items() if consumer == node_id]
        target_prov = provenance.union(*(provenance.get(item) for item in (node, *folded)))
        origins = tuple(sorted(origin for origin in target_prov if origin in input_graph.nodes))
        inventory.append((node_id, node, folded, origins))
    origin_counts = Counter(origins for _node_id, _node, _folded, origins in inventory if origins)

    for node_id, node, folded, origins in inventory:
        key = node.op.cache_key()
        suffix = key[:12] if key is not None else node_id
        name = f"{node.op.name or node_id}.{suffix}"
        if origins and origin_counts[origins] == 1:
            target = {"origins": list(origins)}
        else:
            loop_graph = single_node_graph(fused, node_id, absorb=frozenset(item.id for item in folded))
            target = {"loop": intern_loop_program(loops, loop_graph)}
        entry = {
            "name": name,
            "program": program_ref,
            "target": target,
        }
        entries.append(entry)

    document: dict = {
        "compute_cap": list(ctx.compute_capability),
        "programs": programs,
        "configs": entries,
    }
    if loops:
        document["loops"] = loops
    if ctx.gpu_name:
        document["gpu_name"] = ctx.gpu_name
    if model:
        document["model"] = model

    dump_golden_file(document, destination)
    return TraceInventoryResult(path=destination, target_count=len(entries))


def load_working_targets(path: str | Path, *, kernel: str | None = None) -> tuple[dict, list[WorkingGoldenTarget]]:
    """Load a mutable YAML and reconstruct its deduplicated tune targets."""
    source = Path(path)
    if is_repository_golden_path(source):
        raise ValueError(f"working golden cannot point inside the canonical repository goldens: {source}")
    document = load_golden_file(source)

    by_source: dict[tuple[int, tuple], WorkingGoldenTarget] = {}
    for index, entry in enumerate(document["configs"]):
        if kernel and kernel not in entry["name"]:
            continue
        record = golden_record_from_entry(document, entry)
        key = (record.program_index, record.target_key)
        target = by_source.get(key)
        if target is None:
            target = WorkingGoldenTarget(label=entry["name"], code=None, input=None, dynamic=None, program=record.target_program)
            by_source[key] = target
        target.entry_indexes.append(index)
        if "knobs" in entry:
            target.proposals.append((index, dict(entry["knobs"])))

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
    for (entry_index, _pins), ranking in zip(target.proposals, rankings, strict=True):
        entry = configs[entry_index]
        if golden_entry_state(entry) == GoldenEntryState.VERIFIED:
            continue
        entry["ranking"] = {**ranking, "source": "proposal"}
    dump_golden_file(document, path, overwrite=True)


def persist_tune_winner(
    path: str | Path,
    document: dict,
    target: WorkingGoldenTarget,
    winner: tuple[dict[str, str], float] | None,
    *,
    compile_flags: str,
) -> None:
    """Atomically persist one unambiguous directly searched winner."""
    from emmy.compiler.pipeline.knob import canonical_row_key  # noqa: PLC0415

    configs = document["configs"]
    if winner is not None:
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
            index
            for index in target.entry_indexes
            if "knobs" in configs[index] and canonical_row_key(configs[index]["knobs"]) == winner_key
        ]
        writable = next(
            (index for index in matching if golden_entry_state(configs[index]) != GoldenEntryState.VERIFIED),
            None,
        )
        if writable is not None:
            previous = dict(configs[writable].get("ranking") or {})
            configs[writable]["ranking"] = {
                **winner_ranking,
                "source": previous.get("source", "tune"),
                "tune_winner": True,
            }
        elif not matching:
            seed = copy.deepcopy(configs[target.entry_indexes[0]])
            for key in ("knobs", "measurements", "ranking"):
                seed.pop(key, None)
            seed["knobs"] = winner_knobs
            seed["ranking"] = {**winner_ranking, "tune_winner": True}
            configs.append(seed)
            target.entry_indexes.append(len(configs) - 1)
    dump_golden_file(document, path, overwrite=True)
