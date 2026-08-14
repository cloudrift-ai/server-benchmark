"""Exact, auditable replay of a two-level placement + scheduling winner.

The online prior deliberately pools placement features, and ordinary golden rows
deliberately keep one schedule row per kernel.  Neither representation is an
execution plan.  A working golden therefore persists the compiler's ordered fork
transcript until the result is verified and promoted into those canonical rows.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from emmy.compiler.pipeline.fork import flatten_leaves
from emmy.compiler.pipeline.knob import stamp_schedule_families
from emmy.compiler.structural import digest

if TYPE_CHECKING:
    from emmy.compiler.context import Context
    from emmy.compiler.graph import Graph
    from emmy.compiler.pipeline.pipeline import Decision, ForkPoint
    from emmy.compiler.pipeline.search.db import SearchDB


REPLAY_PLAN_VERSION = 1


def _exact_row_key(row: Mapping) -> tuple:
    return tuple(sorted((str(name), type(value).__name__, repr(value)) for name, value in row.items()))


def decision_to_wire(decision: Decision) -> dict:
    """Stable YAML spelling of one resolved fork."""
    return {
        "rule": decision.rule_name,
        "node": decision.node_id,
        "kind": decision.chosen_kind,
        "knobs": dict(decision.knob_delta),
        "options": decision.n_options,
    }


def _recognized_inventory(graph: Graph) -> list[dict]:
    from emmy.compiler.ir.loop import LoopOp  # noqa: PLC0415
    from emmy.compiler.ir.tile import TileOp  # noqa: PLC0415

    counts: Counter[str] = Counter()
    order: list[str] = []
    for node_id in graph.topological_order():
        op = graph.nodes[node_id].op
        if not isinstance(op, (LoopOp, TileOp)) or (key := op.cache_key()) is None:
            continue
        if key not in counts:
            order.append(key)
        counts[key] += 1
    return [{"key": key, "multiplicity": counts[key]} for key in order]


def _cuda_inventory(graph: Graph) -> list[dict]:
    from emmy.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415

    out = []
    for node_id in graph.topological_order():
        op = graph.nodes[node_id].op
        if not isinstance(op, CudaOp):
            continue
        key = op.cache_key()
        if key is None:
            raise ValueError(f"CUDA plan child {node_id!r} has no structural key")
        out.append({"key": key, "knobs": stamp_schedule_families(op.knobs)})
    return out


def _inventory_key(rows: Sequence[Mapping]) -> str:
    return digest(*(tuple(sorted((str(k), str(v)) for k, v in row.items())) for row in rows))


def _placement_pins(graph: Graph) -> dict[str, str]:
    """Full, unpooled scoped PLACE row carried by a winning outer terminal."""
    pins: dict[str, str] = {}
    for node in graph.nodes.values():
        for name, value in (getattr(node.op, "decision_knobs", None) or {}).items():
            if str(name).split("@", 1)[0] != "PLACE":
                continue
            if name in pins and str(pins[name]) != str(value):
                raise ValueError(f"placement terminal carries conflicting {name} decisions: {pins[name]!r} and {value!r}")
            pins[str(name)] = str(value)
    return pins


def _child_schedule_knobs(trace: Sequence[Decision], kernels: Sequence[Mapping]) -> dict[str, str]:
    """Canonical flat replay row for one independently lowered outer child."""
    from emmy.compiler.pipeline.knob import tuning_knob_items  # noqa: PLC0415

    row: dict[str, str] = {}
    for decision in trace:
        for name, value in tuning_knob_items(decision.knob_delta):
            if name.split("@", 1)[0] == "PLACE":
                continue
            if name in row and row[name] != value:
                raise ValueError(f"child schedule transcript changes {name} from {row[name]!r} to {value!r}")
            row[name] = value
    # Deterministic pass-boundary OFF stamps do not appear in a fork trace. Add
    # only values all emitted CUDA pieces agree on; distinct scoped rows remain
    # independent in ``kernels`` and are audited separately.
    if kernels:
        shared = set(kernels[0]["knobs"])
        for kernel in kernels[1:]:
            shared &= set(kernel["knobs"])
        for name in shared:
            values = {str(kernel["knobs"][name]) for kernel in kernels}
            if len(values) == 1 and name not in row:
                row[name] = values.pop()
    return stamp_schedule_families(row)


def _capture_children(outer: Graph, *, ctx: Context, db: SearchDB) -> list[dict]:
    """Resolve every unique recognized child independently, as inner tuning does."""
    from emmy.compiler.pipeline import Pipeline  # noqa: PLC0415
    from emmy.compiler.pipeline.passes.lowering.tile._flash import fused_producer_ids  # noqa: PLC0415
    from emmy.compiler.pipeline.pipeline import Run  # noqa: PLC0415
    from emmy.compiler.pipeline.search.policy.greedy import greedy_decide  # noqa: PLC0415
    from emmy.compiler.pipeline.search.slice import single_node_graph  # noqa: PLC0415
    from emmy.compiler.pipeline.search.two_level import LOWERING_PASSES  # noqa: PLC0415

    absorbed: dict[str, str] = {}
    nodes = {node["key"]: node for node in _recognized_inventory(outer)}
    reps: dict[str, tuple[str, frozenset[str]]] = {}
    for node_id in outer.topological_order():
        node = outer.nodes[node_id]
        producers = frozenset(fused_producer_ids(outer, node)) if node.op.cache_key() in nodes else frozenset()
        for producer in producers:
            absorbed[producer] = node_id
    for node_id in outer.topological_order():
        node = outer.nodes[node_id]
        key = node.op.cache_key()
        if key not in nodes or node_id in absorbed:
            continue
        producers = frozenset(producer for producer, consumer in absorbed.items() if consumer == node_id)
        if node_id not in absorbed and key not in reps:
            reps[key] = (node_id, producers)

    out = []
    for key, (node_id, producers) in reps.items():
        sub = single_node_graph(outer, node_id, absorb=producers)
        terminal, trace = Run(pipeline=Pipeline.build(LOWERING_PASSES), ctx=ctx, db=db).resolve(sub, greedy_decide(db=db))
        kernels = _cuda_inventory(terminal)
        if not kernels:
            raise ValueError(f"recognized child {key} produced no CUDA kernels")
        out.append(
            {
                "key": key,
                "multiplicity": nodes[key]["multiplicity"],
                # Filled by ``run_two_level_tune`` from an exact replay bench.
                "latency_us": 0.0,
                "decisions": [decision_to_wire(item) for item in trace],
                "terminal_key": _inventory_key(kernels),
                "knobs": _child_schedule_knobs(trace, kernels),
                "kernels": kernels,
            }
        )
    return out


def replay_outer_tuning_plan(graph: Graph, plan: Mapping, *, ctx: Context) -> Graph:
    """Replay and audit only the structural half of a plan."""
    from emmy.compiler.pipeline.search.two_level import outer_pipeline  # noqa: PLC0415

    outer = plan["outer"]
    terminal, _ = _resolve_transcript(graph, pipeline=outer_pipeline(), ctx=ctx, transcript=outer["decisions"])
    recognized = _recognized_inventory(terminal)
    if recognized != outer["recognized"] or _inventory_key(recognized) != outer["terminal_key"]:
        raise ValueError("replay_plan outer terminal fingerprint is stale")
    if _placement_pins(terminal) != outer["placement"]:
        raise ValueError("replay_plan outer scoped PLACE metadata is stale")
    return terminal


def replay_child_tuning_plan(outer: Graph, child: Mapping, *, ctx: Context) -> Graph:
    """Replay one recognized child's independent schedule transcript."""
    from emmy.compiler.pipeline import Pipeline  # noqa: PLC0415
    from emmy.compiler.pipeline.passes.lowering.tile._flash import fused_producer_ids  # noqa: PLC0415
    from emmy.compiler.pipeline.search.slice import single_node_graph  # noqa: PLC0415
    from emmy.compiler.pipeline.search.two_level import LOWERING_PASSES  # noqa: PLC0415

    node_id = next(
        (nid for nid in outer.topological_order() if outer.nodes[nid].op.cache_key() == child["key"]),
        None,
    )
    if node_id is None:
        raise ValueError(f"replay_plan child {child['key']} no longer exists in outer terminal")
    producers = frozenset(fused_producer_ids(outer, outer.nodes[node_id]))
    sub = single_node_graph(outer, node_id, absorb=producers)
    terminal, _ = _resolve_transcript(
        sub,
        pipeline=Pipeline.build(LOWERING_PASSES),
        ctx=ctx,
        transcript=child["decisions"],
    )
    kernels = _cuda_inventory(terminal)
    if kernels != child["kernels"] or _inventory_key(kernels) != child["terminal_key"]:
        raise ValueError(f"replay_plan child {child['key']} CUDA fingerprint or schedule rows are stale")
    return terminal


def _trace_decide(expected: Sequence[Mapping]):
    """Build a strict Run.resolve callback consuming ``expected`` in order."""
    from emmy.compiler.graph import Graph  # noqa: PLC0415
    from emmy.compiler.pipeline.pipeline import _choice_knobs, _concrete_option  # noqa: PLC0415

    index = 0

    def decide(fp: ForkPoint):
        nonlocal index
        if index >= len(expected):
            raise ValueError(f"replay transcript ended before fork {fp.match.rule.name!r} at {fp.node_id!r}")
        want = expected[index]
        where = f"replay decision {index}"
        actual_site = (fp.match.rule.name, fp.node_id, len(fp.options))
        wanted_site = (want["rule"], want["node"], want["options"])
        if actual_site != wanted_site:
            raise ValueError(f"{where} is stale: expected site {wanted_site!r}, compiler offered {actual_site!r}")
        matches = []
        for choice in flatten_leaves(fp.options):
            option = _concrete_option(choice)
            if option is None:
                continue
            kind = "graph" if isinstance(option, Graph) else "op"
            knobs = _choice_knobs(choice, option, fp.root_op)
            if kind == want["kind"] and _exact_row_key(knobs) == _exact_row_key(want["knobs"]):
                matches.append(choice)
        if len(matches) != 1:
            raise ValueError(
                f"{where} is stale: expected one {want['kind']} option with knobs {dict(want['knobs'])!r}, found {len(matches)}"
            )
        index += 1
        return matches[0]

    def finish() -> None:
        if index != len(expected):
            raise ValueError(f"replay transcript has {len(expected) - index} unconsumed decision(s)")

    return decide, finish


def _resolve_transcript(graph: Graph, *, pipeline, ctx: Context, transcript: Sequence[Mapping], dump=None) -> tuple[Graph, list[Decision]]:
    from emmy.compiler.pipeline.pipeline import Run  # noqa: PLC0415

    decide, finish = _trace_decide(transcript)
    terminal, trace = Run(pipeline=pipeline, ctx=ctx, dump=dump).resolve(graph.copy(), decide)
    finish()
    actual = [decision_to_wire(item) for item in trace]
    if actual != list(transcript):
        raise ValueError("replay transcript resolved but did not round-trip byte-for-byte")
    return terminal, trace


def capture_replay_plan(
    source: Graph,
    outer_terminal: Graph,
    *,
    outer_trace: Sequence[Mapping],
    ctx: Context,
    db: SearchDB,
    dump=None,
) -> tuple[Graph, dict]:
    """Capture the selected placement and every independently scheduled CUDA child."""
    from emmy.compiler.pipeline import Pipeline  # noqa: PLC0415
    from emmy.compiler.pipeline.pipeline import Run  # noqa: PLC0415
    from emmy.compiler.pipeline.search.policy.greedy import greedy_decide  # noqa: PLC0415
    from emmy.compiler.pipeline.search.two_level import LOWERING_PASSES, outer_pipeline  # noqa: PLC0415

    placement = _placement_pins(outer_terminal)
    replayed_outer, recovered_trace = _resolve_transcript(
        source,
        pipeline=outer_pipeline(),
        ctx=ctx,
        transcript=outer_trace,
    )
    wanted_recognized = _recognized_inventory(outer_terminal)
    got_recognized = _recognized_inventory(replayed_outer)
    if got_recognized != wanted_recognized:
        raise ValueError("winning scoped PLACE row does not reconstruct its recognized kernel set")
    if _placement_pins(replayed_outer) != placement:
        raise ValueError("winning scoped PLACE transcript lost or changed a consumed seam decision during replay")

    assembled, lowering_trace = Run(pipeline=Pipeline.build(LOWERING_PASSES), ctx=ctx, db=db, dump=dump).resolve(
        replayed_outer.copy(), greedy_decide(db=db)
    )
    cuda = _cuda_inventory(assembled)
    if not cuda:
        raise ValueError("two-level winner produced no CUDA kernels")
    plan = {
        "version": REPLAY_PLAN_VERSION,
        # Filled only after the exact assembled transcript is benchmarked. A
        # per-op DB sum may name different rows and is not a plan measurement.
        "total_us": 0.0,
        "outer": {
            "placement": placement,
            "decisions": [decision_to_wire(item) for item in recovered_trace],
            "terminal_key": _inventory_key(got_recognized),
            "recognized": got_recognized,
        },
        "lowering": {
            "decisions": [decision_to_wire(item) for item in lowering_trace],
            "terminal_key": _inventory_key(cuda),
            "kernels": cuda,
            "children": _capture_children(replayed_outer, ctx=ctx, db=db),
        },
    }
    return assembled, plan


def replay_tuning_plan(graph: Graph, plan: Mapping, *, ctx: Context, dump=None) -> Graph:
    """Exactly replay and audit one working-golden plan."""
    from emmy.compiler.pipeline import Pipeline  # noqa: PLC0415
    from emmy.compiler.pipeline.search.two_level import LOWERING_PASSES  # noqa: PLC0415

    if plan.get("version") != REPLAY_PLAN_VERSION:
        raise ValueError(f"unsupported replay_plan version {plan.get('version')!r}")
    terminal = replay_outer_tuning_plan(graph, plan, ctx=ctx)
    lowered, _ = _resolve_transcript(
        terminal,
        pipeline=Pipeline.build(LOWERING_PASSES),
        ctx=ctx,
        transcript=plan["lowering"]["decisions"],
        dump=dump,
    )
    cuda = _cuda_inventory(lowered)
    if cuda != plan["lowering"]["kernels"] or _inventory_key(cuda) != plan["lowering"]["terminal_key"]:
        raise ValueError("replay_plan CUDA terminal fingerprint or independent schedule rows are stale")
    return lowered
