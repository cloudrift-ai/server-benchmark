"""Mutable working-golden inventories, candidates, and ranking feedback.

This module owns the untrusted side of the golden YAML workflow: trace inventory
generation, target reconstruction, exact proposal measurement, and atomic ranking
persistence. CLI commands only validate argument combinations and report errors.
"""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path

from emmy.compiler.pipeline.search.golden import (
    GoldenEntryState,
    dump_golden_file,
    golden_entry_state,
    golden_record_from_entry,
    is_repository_golden_path,
    load_golden_file,
)
from emmy.compiler.pipeline.strategy import PipelineStrategy


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

    targets: list[tuple[str, object]] = []
    for node_id in fused.topological_order():
        node = fused.nodes[node_id]
        if not isinstance(node.op, LoopOp):
            continue
        targets.append((node_id, node))

    # Persist the pristine program once.  Per-target frontend slices are useful
    # ephemeral tuning views, but they can change fusion when independently
    # lowered (especially sibling linears and computed-A cones).  Provenance
    # selectors must therefore resolve against the original trace context.
    program_ref: int | None = None
    inventory = []
    for node_id, node in targets:
        origins = tuple(sorted(origin for origin in provenance.get(node) if origin in input_graph.nodes))
        inventory.append((node_id, node, origins))
    origin_counts = Counter(origins for _node_id, _node, origins in inventory if origins)
    used_names = {
        realization["name"]
        for entry in entries
        for realization in entry.get("realizations", [])
        if isinstance(realization, dict) and isinstance(realization.get("name"), str)
    }

    for node_id, node, origins in inventory:
        key = node.op.identity_key(with_io=True, with_knobs=True)
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
            loop_graph = single_node_graph(fused, node_id)
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
    from emmy.compiler.pipeline.knob import complete_kernel_row  # noqa: PLC0415

    merged: dict[str, str] = {}
    try:
        for row in (complete_kernel_row(node.op.knobs) for node in graph.nodes.values() if isinstance(node.op, CudaOp)):
            for key, value in row.items():
                if key in merged and str(merged[key]) != str(value):
                    return None
                merged[key] = value
    except ValueError:
        return None
    return merged or None


class _ProposalLoopIdentity(PipelineStrategy):
    """Capture the finalized Loop target and any measured structural parent."""

    def __init__(self) -> None:
        self.value: tuple[str, str, dict] | None = None
        self.structural_parents: list[tuple[dict[str, str], str, dict]] = []

    def _capture(self, graph) -> None:
        if self.value is not None:
            return
        from emmy.compiler.ir.loop import LoopOp  # noqa: PLC0415
        from emmy.compiler.pipeline.knob import STRUCT_PREFIX  # noqa: PLC0415
        from emmy.compiler.pipeline.passes.identity import IdentityStrategy  # noqa: PLC0415
        from emmy.compiler.pipeline.strategy import discovered_strategies  # noqa: PLC0415

        loops = [node.op for node in graph.nodes.values() if isinstance(node.op, LoopOp)]
        if len(loops) != 1:
            return
        identity = next(strategy for strategy in discovered_strategies() if isinstance(strategy, IdentityStrategy))
        stamped = {key: float(value) for key, value in loops[0].knobs.items() if key.startswith(STRUCT_PREFIX)}
        cache_key = loops[0].identity_key(with_io=True, with_knobs=True)
        if stamped and cache_key is not None:
            self.value = identity.op_sig(loops[0], graph), cache_key, stamped

    def on_run_start(self, event) -> None:
        self._capture(event.graph)

    def on_pass_end(self, event) -> None:
        self._capture(event.graph)

    def on_splice(self, event) -> None:
        """Capture the consumed parent whose cross-CTA route changes the kernel set."""
        from emmy.compiler.pipeline import TuningSearch  # noqa: PLC0415

        parent_knobs = {**(getattr(event.root_op, "knobs", None) or {}), **getattr(event, "knobs", {})}
        route = TuningSearch._structural_row(parent_knobs)
        if route is None:
            return
        parent = replace(event.root_op, knobs={**parent_knobs, **route})
        if not any(key.startswith("S_") for key in parent.knobs):
            return
        key = parent.identity_key(with_io=True, with_knobs=True)
        if key is None:
            return
        receipt = (dict(route), key, dict(parent.knobs))
        if receipt not in self.structural_parents:
            self.structural_parents.append(receipt)

    def structural_parent(self, route: dict) -> tuple[str, dict] | None:
        """The one consumed parent that realized ``route``, or ``None`` if ambiguous."""
        from emmy.compiler.pipeline import TuningSearch  # noqa: PLC0415

        wanted = TuningSearch._structural_row(route)
        matches = {(key, tuple(sorted(knobs.items()))) for got, key, knobs in self.structural_parents if got == wanted}
        if len(matches) != 1:
            return None
        key, knob_items = matches.pop()
        return key, dict(knob_items)


async def measure_proposals(
    graph, proposals, *, backend, db, ctx, max_candidates: int | None, prior=None, run_id: str | None = None
) -> list[dict]:
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
        loop_identity = _ProposalLoopIdentity()
        terminal = None
        with pinned_knobs(pins):
            pipeline = Pipeline.build(CUDA_PASSES).with_strategies(loop_identity)
            async for candidate in pipeline.tune_async(graph.copy(), search=search, ctx=ctx, backend=backend, db=db):
                terminal = candidate
        if loop_identity.value is not None:
            search._base_knobs.update(loop_identity.value[2])
        raw_rows = [node.op.knobs for node in terminal.graph.nodes.values() if isinstance(node.op, CudaOp)] if terminal else []
        pin_error = unreproducible_pin_flag(pins, raw_rows) if raw_rows else "proposal produced no CUDA kernel"
        validated_route = pins if pin_error is None and loop_identity.value is not None else None
        searched = search.best_realized(validated_input_route=validated_route)
        structural = searched if searched is not None and searched[3] else None
        structural_parent = loop_identity.structural_parent(structural[0]) if structural is not None else None
        if structural_parent is not None:
            search._base_knobs.update({key: value for key, value in structural_parent[1].items() if key.startswith("S_")})
        if prior is not None:
            prior.add_rows(search._collect_rows())
            prior.maybe_refit()
        if loop_identity.value is not None:
            db.record_nodes(
                search._collect_node_records(
                    context_key=ctx.structural_key(),
                    op_sig=loop_identity.value[0],
                    gpu=ctx.hardware_id(),
                    run_id=run_id or "",
                    validated_input_route=validated_route,
                )
            )
        measured_knobs = dict(structural[0]) if structural is not None else (realized_tuning_knobs(terminal.graph) if terminal else None)
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


def record_latency(
    path: str | Path,
    document: dict,
    name: str,
    *,
    hardware_id: str,
    emmy_us: float,
    tcompile_us: float | None,
    knobs: dict | None = None,
    pins: dict | None = None,
) -> None:
    """Write one card's measured latencies back into a working golden's realization.

    The per-card block is keyed by ``Context.hardware_id`` — the identity that already separates
    same-die SKUs like H100 from H200, which a free-text card name does not. It is separate from
    the flat ``measurements`` block because a model golden is one file per card (so the card is
    implied by the file) while a file measured on several cards needs a row each.

    Both numbers where both exist, because the block answers two questions and only one of them is
    a ratchet: ``emmy_us`` against its own stored value says *did we regress*, and ``tcompile_us``
    beside it says *are we ahead of or behind torch*, per case, per card. ``tcompile_us`` is
    omitted rather than faked when the target has no torch twin to compile.
    """
    destination = Path(path)
    if is_repository_golden_path(destination):
        raise ValueError(f"refusing to write measurements into a canonical repository golden: {destination}")
    from emmy.compiler.pipeline.knob import canonical_row_key  # noqa: PLC0415

    wanted_knobs = canonical_row_key(knobs) if knobs is not None else None
    wanted_pins = tuple(sorted((key, str(value)) for key, value in pins.items())) if pins is not None else None
    matches = []
    for entry in document["configs"]:
        for realization in entry["realizations"]:
            if realization["name"] != name:
                continue
            if wanted_knobs is not None and canonical_row_key(realization.get("knobs", {})) != wanted_knobs:
                continue
            got_pins = tuple(sorted((key, str(value)) for key, value in realization.get("pins", {}).items()))
            if wanted_pins is not None and got_pins != wanted_pins:
                continue
            matches.append(realization)
    if len(matches) != 1:
        raise ValueError(f"{destination} resolves {name!r} to {len(matches)} latency rows; exact knobs and pins must select one")
    timings = {"emmy_us": float(emmy_us)}
    if tcompile_us:
        timings["tcompile_us"] = float(tcompile_us)
    matches[0].setdefault("latency", {})[hardware_id] = timings
    dump_golden_file(document, destination, overwrite=True, incremental=True)


def persist_proposal_rankings(path: str | Path, document: dict, target: WorkingGoldenTarget, rankings: list[dict]) -> None:
    """Atomically persist measured proposal feedback for one target of a loaded document."""
    configs = document["configs"]
    for ((entry_index, realization_index), _pins), ranking in zip(target.proposals, rankings, strict=True):
        realization = configs[entry_index]["realizations"][realization_index]
        if golden_entry_state(realization) == GoldenEntryState.VERIFIED:
            continue
        realization["ranking"] = {**ranking, "source": "proposal"}
    dump_golden_file(document, path, overwrite=True, incremental=True)


def persist_tune_winner(
    path: str | Path,
    document: dict,
    target: WorkingGoldenTarget,
    winner: tuple[dict[str, str], float] | None,
    *,
    compile_flags: str,
) -> None:
    """Atomically persist one unambiguous directly searched winner into a loaded document."""
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
            for key in ("knobs", "measurements", "ranking", "latency"):
                seed.pop(key, None)
            seed["knobs"] = winner_knobs
            seed["ranking"] = {**winner_ranking, "tune_winner": True}
            configs[config_index]["realizations"].append(seed)
            target.entry_indexes.append((config_index, len(configs[config_index]["realizations"]) - 1))
    dump_golden_file(document, path, overwrite=True, incremental=True)
