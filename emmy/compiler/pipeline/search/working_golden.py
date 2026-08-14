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
    bindings: dict[str, int] = field(default_factory=dict)
    pins: dict[str, object] = field(default_factory=dict)
    program: object | None = None
    entry_indexes: list[tuple[int, int]] = field(default_factory=list)
    # Proposal payload is either a legacy flat knob row or an exact replay plan.
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
    """Append one maximally fused graph to shared trace-inventory pools.

    Placement is a tuner-visible structural fork, so inventory capture must stop before it.
    Each retained Loop target enters the outer placement search with all legal fused/cut
    alternatives available instead of baking one machine's current routing evidence into the
    target ABI.
    """
    from emmy.compiler import provenance  # noqa: PLC0415
    from emmy.compiler.ir.loop import LoopOp  # noqa: PLC0415
    from emmy.compiler.loop_wire import intern_loop_program  # noqa: PLC0415
    from emmy.compiler.pipeline import LOOP_PASSES, Pipeline  # noqa: PLC0415
    from emmy.compiler.pipeline.passes.lowering.tile._cut import _product_reduction_pair  # noqa: PLC0415
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

    # Match two-level tuning's context-complete target set. A flash score producer is
    # part of its consuming attention target, not a second inventory row. Likewise, a
    # placement residue whose expanded product has one additive-reduction consumer must
    # travel with that consumer: the outer fork needs both endpoints to compare
    # ``PLACE@product=fuse`` recomposition with retaining the materialized pair.
    absorbed: dict[str, str] = {}
    for node_id in fused.topological_order():
        node = fused.nodes[node_id]
        if not isinstance(node.op, LoopOp):
            continue
        for producer_id in fused_producer_ids(fused, node):
            absorbed[producer_id] = node_id
        if (consumer := _product_reduction_pair(fused, node)) is not None:
            absorbed[node.id] = consumer.id

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
            elif "replay_plan" in realization:
                target.proposals.append((path, {"replay_plan": copy.deepcopy(realization["replay_plan"])}))

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


def _verification_input_maps(graphs: tuple, *, seed: int) -> tuple[dict, ...]:
    """Build graph-local bindings from one canonical deterministic input source.

    Exact replay and cold greedy may lay constants out differently, so sharing a
    final ``input_data`` mapping is not generally valid. Instead, activations are
    keyed by stable input id and loadable constants by source path; each graph then
    applies its own recorded load transforms to those identical source values.
    """
    import numpy as np  # noqa: PLC0415

    from emmy.compiler.dim import DEFAULT_SEQ_HINT, Dim  # noqa: PLC0415
    from emmy.compiler.dtype import decode_f8  # noqa: PLC0415
    from emmy.compiler.dtype import get as get_dtype
    from emmy.compiler.ir.base import ConstantOp, InputOp  # noqa: PLC0415
    from emmy.compiler.loader.binder import bind_constants  # noqa: PLC0415

    sym_env: dict[str, int] = {}
    for graph in graphs:
        for node in graph.nodes.values():
            for dim in node.output.shape:
                if isinstance(dim, Dim) and not dim.is_static:
                    for name in dim.expr.free_vars():
                        sym_env.setdefault(name, dim.hint or DEFAULT_SEQ_HINT)

    def static(shape) -> tuple[int, ...]:
        return tuple(
            (dim.as_static() if dim.is_static else int(dim.expr.eval(sym_env))) if isinstance(dim, Dim) else int(dim) for dim in shape
        )

    def spec(shape, dtype) -> tuple[tuple[int, ...], str]:
        return static(shape), get_dtype(dtype or "f32").name

    def merge(target: dict, key: str, value, kind: str) -> None:
        previous = target.setdefault(key, value)
        if previous != value:
            raise ValueError(f"replay verification {kind} {key!r} has inconsistent declarations: {previous!r} != {value!r}")

    inputs: dict[str, tuple[tuple[int, ...], str]] = {}
    sources: dict[str, tuple[tuple[int, ...], str]] = {}
    synthetic: dict[str, tuple[tuple[int, ...], str]] = {}
    for graph in graphs:
        for node_id, node in graph.nodes.items():
            op = node.op
            if isinstance(op, InputOp):
                merge(inputs, node_id, spec(node.output.shape, node.output.dtype), "input")
            elif isinstance(op, ConstantOp) and op.value is None:
                dtype = op.source_dtype or node.output.dtype
                if op.source_path:
                    merge(sources, op.source_path, spec(op.source_shape or node.output.shape, dtype), "constant source")
                for path, shape in op.source_parts:
                    merge(sources, path, spec(shape, dtype), "constant source")
                if not op.source_path and not op.source_parts:
                    merge(synthetic, node_id, spec(node.output.shape, node.output.dtype), "synthetic constant")

    rng = np.random.default_rng(seed)

    def random_values(shape: tuple[int, ...], dtype: str, *, scale: float):
        if dtype in {"f8e4m3", "f8e5m2"}:
            bits = rng.integers(0, 256, shape, dtype=np.uint8)
            bits[~np.isfinite(decode_f8(bits, dtype))] = np.uint8(0)
            return bits
        return rng.standard_normal(shape, dtype=np.float32) * scale

    input_values = {key: random_values(*value, scale=1.0) for key, value in sorted(inputs.items())}
    source_values = {key: random_values(*value, scale=0.02) for key, value in sorted(sources.items())}
    synthetic_values = {key: random_values(*value, scale=0.02) for key, value in sorted(synthetic.items())}

    bindings = []
    for graph in graphs:
        data: dict[str, object] = {}
        for node_id, node in graph.nodes.items():
            if isinstance(node.op, InputOp):
                data[node_id] = input_values[node_id]
            elif isinstance(node.op, ConstantOp) and node.op.value is not None:
                data[node_id] = [float(node.op.value)]
        data.update(bind_constants(graph, source_values))
        for node_id, node in graph.nodes.items():
            if not isinstance(node.op, ConstantOp) or node.op.value is not None or node_id in data:
                continue
            if node_id not in synthetic_values:
                raise ValueError(f"replay verification could not bind constant {node_id!r}")
            data[node_id] = synthetic_values[node_id].flatten().tolist()
        bindings.append(data)
    return tuple(bindings)


def _verification_us(bench, *, label: str) -> tuple[float, str]:
    """Require captured deploy timing and return ``(us, semantics)``."""
    import math  # noqa: PLC0415

    if getattr(bench, "captured", None) is not True:
        raise ValueError(f"{label}: replay verification requires CUDA-graph-captured timing")
    launches = int(getattr(bench, "num_launches", 0) or len(getattr(bench, "per_launch", None) or ()))
    value = getattr(bench, "e2e_min_ms", None)
    if launches > 1:
        if value is None:
            raise ValueError(f"{label}: multi-launch replay verification requires whole-program e2e_min_ms")
        value = float(value) * 1000.0
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{label}: invalid whole-program timing {value!r} us")
        return value, "whole_program_e2e"
    if value is not None:
        value = float(value) * 1000.0
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{label}: invalid whole-program timing {value!r} us")
        return value, "whole_program_e2e"
    value = getattr(bench, "min_ms", None)
    if value is None:
        value = bench.time_ms
    value = float(value) * 1000.0
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label}: invalid single-launch timing {value!r} us")
    return value, "single_launch"


def _verification_output_proof(outputs: dict, reference: dict) -> dict:
    """The normal same-input configuration-parity gate plus audit statistics."""
    import numpy as np  # noqa: PLC0415

    if set(outputs) != set(reference):
        return {
            "status": "fail",
            "relative_tolerance": 0.05,
            "error": f"output names differ: exact={sorted(outputs)}, greedy={sorted(reference)}",
        }
    worst = 0.0
    max_abs = 0.0
    abs_sum = 0.0
    elements = 0
    for name, ref in reference.items():
        actual = np.asarray(outputs[name], dtype=np.float64)
        wanted = np.asarray(ref, dtype=np.float64)
        if actual.shape != wanted.shape:
            return {
                "status": "fail",
                "relative_tolerance": 0.05,
                "error": f"output {name!r} shape {actual.shape} != greedy {wanted.shape}",
            }
        if not np.isfinite(actual).all() or not np.isfinite(wanted).all():
            return {
                "status": "fail",
                "relative_tolerance": 0.05,
                "error": f"output {name!r} contains non-finite values",
            }
        delta = np.abs(actual - wanted)
        denom = float(np.abs(wanted).max()) or 1.0
        worst = max(worst, float(delta.max()) / denom)
        max_abs = max(max_abs, float(delta.max()))
        abs_sum += float(delta.sum())
        elements += int(delta.size)
    passed = worst <= 0.05
    proof = {
        "status": "pass" if passed else "fail",
        "relative_tolerance": 0.05,
        "max_abs_error": max_abs,
        "mean_abs_error": abs_sum / max(elements, 1),
        "max_relative_error": worst,
    }
    if not passed:
        proof["error"] = f"rel err {worst:.3f} vs greedy output"
    return proof


async def _verify_replay_pair(
    exact,
    greedy,
    *,
    backend,
    seed: int,
    warmup: int,
    iters: int,
    label: str,
) -> tuple[dict, dict]:
    """Fresh-bench one exact/cold pair and require same-input output parity."""
    import uuid  # noqa: PLC0415

    exact_inputs, greedy_inputs = _verification_input_maps((exact, greedy), seed=seed)
    pair_id = uuid.uuid4().hex
    exact_bench, exact_outputs = await backend.bench_pinned_async(
        exact,
        run_inputs=exact_inputs,
        run_inputs_key=f"replay-exact-{pair_id}",
        warmup=warmup,
        num_iters=iters,
    )
    greedy_bench, greedy_outputs = await backend.bench_pinned_async(
        greedy,
        run_inputs=greedy_inputs,
        run_inputs_key=f"replay-greedy-{pair_id}",
        warmup=warmup,
        num_iters=iters,
    )
    if exact_outputs is None or greedy_outputs is None:
        raise ValueError(f"{label}: replay verification backend returned no same-input outputs")
    proof = _verification_output_proof(exact_outputs, greedy_outputs)
    if proof["status"] != "pass":
        raise ValueError(f"{label}: wrong-answer: {proof['error']}")
    exact_us, exact_semantics = _verification_us(exact_bench, label=f"{label}: exact")
    greedy_us, greedy_semantics = _verification_us(greedy_bench, label=f"{label}: emmy-greedy")
    measurement = {
        "emmy_us": exact_us,
        "reference_us": greedy_us,
        "reference_backend": "emmy-greedy",
    }
    return measurement, {
        "label": label,
        "emmy_us": exact_us,
        "reference_us": greedy_us,
        "exact_timing_semantics": exact_semantics,
        "reference_timing_semantics": greedy_semantics,
        "captured": True,
        "correctness": proof,
    }


async def verify_replay_plans(
    document: dict,
    *,
    backend,
    ctx,
    warmup: int,
    iters: int,
    seed: int = 0,
    kernel: str | None = None,
) -> tuple[dict, list[dict]]:
    """Audit every matching replay plan against a fresh cold-greedy reference.

    The returned document is a copy. Callers persist it only after every whole
    program and every independent child passed, so a timeout, stale transcript,
    or wrong answer cannot partially verify a working file. ``total_us`` and child
    ``latency_us`` remain the tune-time ranking observations; verification writes
    only the canonical measurement records consumed by promotion.
    """
    if warmup < 10 or iters < 50:
        raise ValueError("replay verification requires --warmup >= 10 and --iters >= 50")
    flags = str(getattr(ctx, "compile_flags", "") or "")
    if any(f"-O{level}" in flags for level in range(3)):
        raise ValueError(f"replay verification requires deployable O3 codegen, not {flags}")
    validate_working_gpu(document, ctx)

    from emmy.compiler.pipeline import CUDA_PASSES, Pipeline  # noqa: PLC0415
    from emmy.compiler.pipeline.search.pins import pinned_knobs  # noqa: PLC0415
    from emmy.compiler.pipeline.search.replay_plan import (  # noqa: PLC0415
        replay_child_source,
        replay_child_tuning_plan,
        replay_outer_tuning_plan,
        replay_tuning_plan,
    )
    from emmy.compiler.pipeline.search.two_level import LOWERING_PASSES  # noqa: PLC0415

    verified = copy.deepcopy(document)
    reports: list[dict] = []
    matched = 0
    for config in verified["configs"]:
        for realization in config["realizations"]:
            plan = realization.get("replay_plan")
            if plan is None or (kernel and kernel not in realization["name"]):
                continue
            matched += 1
            record = golden_record_from_entry(verified, config, realization)
            source = record.target_program
            with pinned_knobs(record.pin_map):
                exact = replay_tuning_plan(source.copy(), plan, ctx=ctx)
                greedy = Pipeline.build(CUDA_PASSES).run(source.copy(), ctx=ctx)
                measurement, report = await _verify_replay_pair(
                    exact,
                    greedy,
                    backend=backend,
                    seed=seed,
                    warmup=warmup,
                    iters=iters,
                    label=f"{realization['name']}: whole",
                )
                realization["measurements"] = measurement
                reports.append(
                    {
                        **report,
                        "scope": "whole",
                        "name": realization["name"],
                        "outer_terminal_key": plan["outer"]["terminal_key"],
                        "terminal_key": plan["lowering"]["terminal_key"],
                    }
                )

                outer = replay_outer_tuning_plan(source.copy(), plan, ctx=ctx)
                for index, child in enumerate(plan["lowering"]["children"]):
                    child_source = replay_child_source(outer, child)
                    exact_child = replay_child_tuning_plan(outer, child, ctx=ctx)
                    greedy_child = Pipeline.build(LOWERING_PASSES).run(child_source.copy(), ctx=ctx)
                    child_measurement, child_report = await _verify_replay_pair(
                        exact_child,
                        greedy_child,
                        backend=backend,
                        seed=seed,
                        warmup=warmup,
                        iters=iters,
                        label=f"{realization['name']}: child {index} {child['key']}",
                    )
                    child["measurements"] = child_measurement
                    reports.append(
                        {
                            **child_report,
                            "scope": "child",
                            "name": realization["name"],
                            "child": index,
                            "key": child["key"],
                            "outer_terminal_key": plan["outer"]["terminal_key"],
                            "terminal_key": child["terminal_key"],
                        }
                    )
    if not matched:
        suffix = f" matching --kernel {kernel!r}" if kernel else ""
        raise ValueError(f"working golden contains no replay_plan{suffix}")
    return verified, reports


async def verify_and_persist_replay_plans(path: str | Path, document: dict, **kwargs) -> list[dict]:
    """Verify a full copy, then atomically write its YAML and adjacent audit."""
    import json  # noqa: PLC0415
    import os  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    verified, reports = await verify_replay_plans(document, **kwargs)
    ctx = kwargs["ctx"]
    audit = {
        "version": 1,
        "working_file": str(Path(path).resolve()),
        "gpu_name": getattr(ctx, "gpu_name", None),
        "compute_cap": list(getattr(ctx, "compute_capability", ()) or ()),
        "context_key": ctx.structural_key(),
        "compile_flags": str(getattr(ctx, "compile_flags", "") or ""),
        "warmup": kwargs["warmup"],
        "iters": kwargs["iters"],
        "seed": kwargs.get("seed", 0),
        "reference_backend": "emmy-greedy",
        "records": reports,
    }
    audit_path = Path(path).with_suffix(Path(path).suffix + ".replay-verify.json")
    payload = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=audit_path.parent, prefix=f".{audit_path.name}.", delete=False) as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
            temporary = Path(output.name)
        dump_golden_file(verified, path, overwrite=True)
        temporary.replace(audit_path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return reports


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
    for proposal_index, (_entry_index, proposal) in enumerate(proposals):
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
        replay_plan = proposal.get("replay_plan")
        if replay_plan is not None:
            from emmy.compiler.pipeline.search.replay_plan import replay_tuning_plan  # noqa: PLC0415

            try:
                terminal_graph = replay_tuning_plan(graph.copy(), replay_plan, ctx=ctx)
                bench = await backend.benchmark_async(terminal_graph)
                e2e_ms = getattr(bench, "e2e_min_ms", None)
                latency = float(e2e_ms if e2e_ms is not None else bench.time_ms) * 1000.0
                rankings.append(
                    {
                        "status": "ok",
                        "latency_us": latency,
                        "compile_flags": ctx.compile_flags,
                        "measured_plan_key": replay_plan["lowering"]["terminal_key"],
                    }
                )
            except Exception as exc:  # noqa: BLE001 - stale working plans are ranking failures, not command aborts
                rankings.append(
                    {
                        "status": "replay_stale",
                        "latency_us": None,
                        "compile_flags": ctx.compile_flags,
                        "measured_plan_key": None,
                        "error": str(exc),
                    }
                )
            continue
        pins = proposal
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
    for ((entry_index, realization_index), _proposal), ranking in zip(target.proposals, rankings, strict=True):
        realization = configs[entry_index]["realizations"][realization_index]
        if golden_entry_state(realization) == GoldenEntryState.VERIFIED:
            continue
        realization["ranking"] = {**ranking, "source": "proposal"}
    dump_golden_file(document, path, overwrite=True)


def persist_tune_winner(
    path: str | Path,
    document: dict,
    target: WorkingGoldenTarget,
    winner: tuple[dict[str, str], float] | None,
    *,
    compile_flags: str,
    replay_plan: dict | None = None,
) -> None:
    """Atomically persist a searched winner, including heterogeneous plans, or fail closed.

    A tune that produced neither an exact plan nor a scalar winner writes a target-scoped
    ``no_exact_pin`` marker, and every stale ``tune_winner`` / marker ranking from an earlier
    tune retires to ``superseded`` — so a strict publication run cannot mistake an older winner
    or unchanged inventory for this run's success.
    """
    from emmy.compiler.pipeline.knob import canonical_row_key  # noqa: PLC0415

    configs = document["configs"]
    plan_key = replay_plan["lowering"]["terminal_key"] if replay_plan is not None else None
    winner_key = canonical_row_key(winner[0]) if replay_plan is None and winner is not None else None

    def claimed(realization: dict) -> bool:
        """Whether the current result (re)writes this realization below."""
        if plan_key is not None:
            return realization.get("replay_plan", {}).get("lowering", {}).get("terminal_key") == plan_key
        if winner_key is not None:
            return "knobs" in realization and canonical_row_key(realization["knobs"]) == winner_key
        return (realization.get("ranking") or {}).get("status") == "no_exact_pin"

    marker_path = None
    for entry_path in target.entry_indexes:
        realization = configs[entry_path[0]]["realizations"][entry_path[1]]
        if golden_entry_state(realization) == GoldenEntryState.VERIFIED:
            continue
        ranking = realization.get("ranking")
        if not isinstance(ranking, dict) or ranking.get("source") != "tune":
            continue
        if claimed(realization):
            if plan_key is None and winner_key is None and marker_path is None:
                marker_path = entry_path
            continue
        if ranking.get("tune_winner") is True or ranking.get("status") == "no_exact_pin":
            realization["ranking"] = {**ranking, "status": "superseded"}
            realization["ranking"].pop("tune_winner", None)

    if replay_plan is not None:
        ranking = {
            "status": "ok",
            "latency_us": replay_plan["total_us"],
            "compile_flags": compile_flags,
            "measured_plan_key": plan_key,
            "source": "tune",
            "tune_winner": True,
        }
        matching = [
            entry_path
            for entry_path in target.entry_indexes
            if configs[entry_path[0]]["realizations"][entry_path[1]].get("replay_plan", {}).get("lowering", {}).get("terminal_key")
            == plan_key
        ]
        writable = next(
            (
                entry_path
                for entry_path in matching
                if golden_entry_state(configs[entry_path[0]]["realizations"][entry_path[1]]) != GoldenEntryState.VERIFIED
            ),
            None,
        )
        if writable is None and not matching:
            config_index, realization_index = target.entry_indexes[0]
            seed = copy.deepcopy(configs[config_index]["realizations"][realization_index])
            for key in ("knobs", "replay_plan", "measurements", "ranking"):
                seed.pop(key, None)
            seed["replay_plan"] = copy.deepcopy(replay_plan)
            seed["ranking"] = ranking
            configs[config_index]["realizations"].append(seed)
            target.entry_indexes.append((config_index, len(configs[config_index]["realizations"]) - 1))
        elif writable is not None:
            realization = configs[writable[0]]["realizations"][writable[1]]
            realization["replay_plan"] = copy.deepcopy(replay_plan)
            realization["ranking"] = ranking
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
            previous = dict(realization.get("ranking") or {})
            realization["ranking"] = {
                **winner_ranking,
                "source": previous.get("source", "tune"),
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
            for key in ("knobs", "replay_plan", "measurements", "ranking"):
                seed.pop(key, None)
            seed["ranking"] = marker
            configs[config_index]["realizations"].append(seed)
            target.entry_indexes.append((config_index, len(configs[config_index]["realizations"]) - 1))
        else:
            configs[marker_path[0]]["realizations"][marker_path[1]]["ranking"] = marker
    dump_golden_file(document, path, overwrite=True)


def promote_replay_plans(document: dict) -> dict:
    """Split audited working plans into canonical routing and child rows.

    Repository goldens intentionally never mix placement with schedules.  This
    transform replays each working transcript, writes one PLACE-only aggregate
    routing row on the original target, and interns one exact Loop-IR target with
    a schedule-only row per independently tuned recognized child.
    """
    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.compiler.ir.loop import LoopOp  # noqa: PLC0415
    from emmy.compiler.loop_wire import intern_loop_program  # noqa: PLC0415
    from emmy.compiler.pipeline import CUDA_PASSES, Pipeline  # noqa: PLC0415
    from emmy.compiler.pipeline.search.pins import pinned_knobs  # noqa: PLC0415
    from emmy.compiler.pipeline.search.replay_plan import (  # noqa: PLC0415
        _cuda_inventory,
        replay_child_tuning_plan,
        replay_outer_tuning_plan,
    )
    from emmy.compiler.pipeline.search.slice import single_node_graph  # noqa: PLC0415

    promoted = copy.deepcopy(document)
    loops = promoted.setdefault("loops", [])
    additions: list[dict] = []
    ctx = Context.from_target(tuple(promoted["compute_cap"]), gpu_name=promoted.get("gpu_name"))

    for config in promoted["configs"]:
        kept = []
        for realization in config["realizations"]:
            plan = realization.get("replay_plan")
            if plan is None:
                kept.append(realization)
                continue
            record = golden_record_from_entry(promoted, config, realization)
            outer = replay_outer_tuning_plan(record.target_program.copy(), plan, ctx=ctx)
            aggregate_measurement = realization.get("measurements")
            if aggregate_measurement is None:
                raise ValueError(
                    f"replay_plan {realization['name']!r} remains a proposal: promotion requires an exact whole-plan "
                    "Emmy measurement and an honest eager/cuBLAS (or same-input emmy-greedy) reference"
                )
            aggregate_measurement = copy.deepcopy(aggregate_measurement)
            if plan["outer"]["placement"]:
                kept.append(
                    {
                        "name": f"{realization['name']}.routing",
                        "bindings": copy.deepcopy(realization["bindings"]),
                        "pins": copy.deepcopy(realization["pins"]),
                        "knobs": copy.deepcopy(plan["outer"]["placement"]),
                        "measurements": aggregate_measurement,
                    }
                )

            promoted_children = []
            for child_index, child in enumerate(plan["lowering"]["children"]):
                if "measurements" not in child:
                    raise ValueError(
                        f"replay_plan child {child['key']} remains a proposal: promotion requires its exact Emmy "
                        "measurement and an honest eager/cuBLAS or same-input emmy-greedy reference"
                    )
                replay_child_tuning_plan(outer, child, ctx=ctx)
                node_id = next(
                    (nid for nid in outer.topological_order() if outer.nodes[nid].op.cache_key() == child["key"]),
                    None,
                )
                if node_id is None:
                    raise ValueError(f"promotion lost recognized child {child['key']}")
                child_loop = single_node_graph(outer, node_id)
                for node in child_loop.nodes.values():
                    if node.op.dialect not in {"tile", "loop"}:
                        continue
                    loop_source = next((source for source in node.op.source_chain() if isinstance(source, LoopOp)), None)
                    if loop_source is None:
                        raise ValueError(f"promotion child {child['key']} has no Loop IR source boundary")
                    node.op = loop_source
                loop_ref = intern_loop_program(loops, child_loop)

                # The promoted flat row must itself reproduce the independently
                # audited transcript. Scoped schedule keys make heterogeneous
                # children representable without pooling their choices.
                with pinned_knobs(child["knobs"]):
                    compiled = Pipeline.build(CUDA_PASSES).run(child_loop.copy(), ctx=ctx)
                if _cuda_inventory(compiled) != child["kernels"]:
                    raise ValueError(f"promotion child {child['key']} cannot be represented by its canonical schedule row")
                promoted_children.extend(child["kernels"] * child["multiplicity"])
                additions.append(
                    {
                        "program": config["program"],
                        "target": {"loop": loop_ref},
                        "realizations": [
                            {
                                "name": f"{realization['name']}.child{child_index}.{child['key'][:12]}",
                                "bindings": copy.deepcopy(realization["bindings"]),
                                "pins": copy.deepcopy(realization["pins"]),
                                "knobs": copy.deepcopy(child["knobs"]),
                                "measurements": copy.deepcopy(child["measurements"]),
                            }
                        ],
                    }
                )
            from emmy.compiler.pipeline.knob import canonical_row_key  # noqa: PLC0415

            expected = sorted(
                ((kernel["key"], canonical_row_key(kernel["knobs"])) for kernel in plan["lowering"]["kernels"]),
                key=repr,
            )
            actual = sorted(((kernel["key"], canonical_row_key(kernel["knobs"])) for kernel in promoted_children), key=repr)
            if actual != expected:
                raise ValueError("promoted child rows do not reconstruct the replay_plan CUDA terminal")
        config["realizations"] = kept
    promoted["configs"] = [config for config in promoted["configs"] if config["realizations"]]
    promoted["configs"].extend(additions)
    return promoted
