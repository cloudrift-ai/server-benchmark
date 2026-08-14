"""Scoped tuning-knob pins and realized-pin validation."""

from __future__ import annotations

import contextlib
import os
from collections import Counter

from emmy import config
from emmy.compiler.pipeline.knob import axis_of, family_of, get, is_off_value, pin_key_matches, values_equal


class KernelSetReplayError(ValueError):
    """An exact working kernel set no longer matches the compiler's offers."""


def unreproducible_pin_flag(pinned: dict, kernel_knobs: list[dict]) -> str | None:
    """Describe pins not realized by any compiled CUDA kernel, or return ``None``.

    A registered family with no realized key is ungateable because serialized IR
    can omit knob stamps. Declared OFF values mean not-applicable rather than a
    conflicting realization; an unknown absent family remains a likely typo.
    """
    if not any(kernel_knobs):
        return None
    misses: list[str] = []
    for name, want in pinned.items():
        fam = family_of(name)
        if fam == "PLACE":
            continue  # a realized cut is visible structurally, not as a knob stamp
        others: list[str] = []
        saw_off = False
        hit = False
        for raw in kernel_knobs:
            for key, got in raw.items():
                if family_of(key) != fam:
                    continue
                if pin_key_matches(name, key) and values_equal(name, want, got):
                    hit = True
                elif is_off_value(fam, got):
                    saw_off = True
                else:
                    spell = f"{key}={got}" if key != name else str(got)
                    if spell not in others:
                        others.append(spell)
            if hit:
                break
        if hit:
            continue
        if not others and not saw_off and get(fam) is not None:
            continue
        ran = "/".join(others) if others else ("(off)" if saw_off else "(unset)")
        misses.append(f"{name}={want} realized {ran}")
    return f"unreproducible pin: {'; '.join(misses)}" if misses else None


@contextlib.contextmanager
def pinned_knobs(knobs: dict):
    """Temporarily publish ``knobs`` as authoritative environment pins.

    Axis-scoped keys ride both their programmatic ``EMMY_<KNOB@site>`` splat and the raw
    ``EMMY_KNOBS`` aggregate. Schedule readers consume the splat after import, while placement
    routing reads the aggregate directly because ``@`` is not a portable shell-variable name.
    """
    saved: dict[str, str | None] = {}
    try:
        scoped = []
        for name, value in knobs.items():
            key = config.knob_var(name)
            saved[key] = os.environ.get(key)
            os.environ[key] = str(value)
            if axis_of(name) is not None:
                scoped.append(f"{name}={value}")
        if scoped:
            saved[config.KNOBS] = os.environ.get(config.KNOBS)
            aggregate = config.knobs_aggregate()
            os.environ[config.KNOBS] = ",".join(part for part in (aggregate, *scoped) if part)
        yield
    finally:
        for key, previous in saved.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


def _placed_kernel_nodes(graph) -> list[tuple[str, object]]:
    """The same recognized-kernel inventory the two-level inner search tunes."""
    from emmy.compiler.pipeline.passes.lowering.tile._flash import fused_producer_ids  # noqa: PLC0415
    from emmy.compiler.pipeline.search.two_level import _kernel_nodes  # noqa: PLC0415

    eligible = dict(_kernel_nodes(graph))
    absorbed = {producer for nid in eligible for producer in fused_producer_ids(graph, graph.nodes[nid])}
    return [(nid, eligible[nid]) for nid in graph.topological_order() if nid in eligible and nid not in absorbed]


def _placement_knobs(graph) -> dict[str, str]:
    """Conflict-free realized placement rows on one placed kernel set."""
    out: dict[str, str] = {}
    for node in graph.nodes.values():
        for key, value in node.op.decision_knobs.items():
            if family_of(key) != "PLACE":
                continue
            value = str(value)
            if key in out and out[key] != value:
                raise KernelSetReplayError(f"placement {key} realized conflicting values {out[key]!r} and {value!r}")
            out[key] = value
    return out


def _leaf_knobs(leaf: object) -> dict:
    """The direct knob row carried by one flattened schedule leaf."""
    from emmy.compiler.graph import Graph  # noqa: PLC0415
    from emmy.compiler.pipeline.fork import Fork  # noqa: PLC0415

    if isinstance(leaf, Fork):
        return dict(leaf.knobs)
    if isinstance(leaf, Graph):
        return {}
    return dict(getattr(leaf, "knobs", None) or {})


def _exact_schedule_decide(kernel_set: dict):
    """Build a deterministic schedule resolver keyed by pre-schedule ``op_key``."""
    from emmy.compiler.pipeline.fork import flatten_leaves  # noqa: PLC0415
    from emmy.compiler.pipeline.knob import stamp_schedule_families  # noqa: PLC0415
    from emmy.compiler.pipeline.search.policy.greedy import PARTITION_RULE  # noqa: PLC0415

    by_key = {row["op_key"]: row for row in kernel_set["kernels"]}

    def decide(fp):
        if fp.structural or fp.match.rule.name != PARTITION_RULE:
            raise KernelSetReplayError(f"unexpected exact-replay fork {fp.match.rule.name!r} at {fp.node_id}")
        op_key = fp.root_op.cache_key()
        component = by_key.get(op_key)
        if component is None:
            raise KernelSetReplayError(f"schedule offered unknown op_key {op_key!r} at {fp.node_id}")
        want = {key: str(value) for key, value in component["pins"].items()}
        matches = []
        for leaf in flatten_leaves(fp.options):
            got = stamp_schedule_families({**dict(fp.root_op.knobs or {}), **_leaf_knobs(leaf)})
            if got == want:
                matches.append(leaf)
        if len(matches) != 1:
            raise KernelSetReplayError(f"op_key {op_key!r} expected one schedule matching {want}, got {len(matches)}")
        return matches[0]

    return decide


def _cuda_record_knobs(graph) -> list[dict[str, str]]:
    """Ordered canonical Cuda rows for one lowered graph."""
    from emmy.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415
    from emmy.compiler.pipeline.knob import stamp_schedule_families  # noqa: PLC0415

    return [
        stamp_schedule_families(graph.nodes[nid].op.knobs or {})
        for nid in graph.topological_order()
        if isinstance(graph.nodes[nid].op, CudaOp)
    ]


def _validate_component_cuda_rows(op_key: str, actual: list[dict[str, str]], expected: list[dict[str, str]]) -> None:
    """Require one component's direct Cuda launch order to remain exact."""
    if actual != expected:
        raise KernelSetReplayError(f"op_key {op_key!r} CUDA record-knob mismatch: expected {expected}, realized {actual}")


def _validate_cuda_inventory(actual: list[dict[str, str]], kernel_set: dict) -> None:
    """Require every target Cuda row without imposing cross-component order."""
    expected = Counter(
        tuple(sorted((key, str(value)) for key, value in row.items()))
        for component in kernel_set["kernels"]
        for _ in range(component["multiplicity"])
        for row in component["cuda_record_knobs"]
    )
    realized = Counter(tuple(sorted(row.items())) for row in actual)
    if realized != expected:
        raise KernelSetReplayError(f"CUDA record-knob inventory mismatch: expected {dict(expected)}, realized {dict(realized)}")


def _lower_exact_schedule(graph, kernel_set: dict, tail_passes: list[str], ctx):
    """Resolve the schedule fork exactly and reject any residual compiler IR."""
    from emmy.compiler.ir.kernel import KernelOp  # noqa: PLC0415
    from emmy.compiler.ir.loop import LoopOp  # noqa: PLC0415
    from emmy.compiler.ir.tile import TileOp  # noqa: PLC0415
    from emmy.compiler.pipeline import Pipeline  # noqa: PLC0415
    from emmy.compiler.pipeline.pipeline import Run  # noqa: PLC0415

    run = Run(pipeline=Pipeline.build(tail_passes), ctx=ctx, rejections=[])
    compiled, _trace = run.resolve(graph, _exact_schedule_decide(kernel_set))
    unlowered = [nid for nid, node in compiled.nodes.items() if isinstance(node.op, (LoopOp, TileOp, KernelOp))]
    if unlowered:
        raise KernelSetReplayError(f"exact replay left unlowered kernel node(s): {unlowered}")
    return compiled


def lower_exact_kernel_set(graph, kernel_set: dict, *, ctx=None):
    """Replay exact placement, schedule leaves, and realized Cuda rows."""
    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.compiler.pipeline import CUDA_PASSES, Pipeline  # noqa: PLC0415
    from emmy.compiler.pipeline.passes.lowering.tile._flash import fused_producer_ids  # noqa: PLC0415
    from emmy.compiler.pipeline.search.slice import single_node_graph  # noqa: PLC0415

    ctx = ctx or Context.probe()
    schedule_index = CUDA_PASSES.index("lowering/schedule")
    placement_passes = CUDA_PASSES[:schedule_index]
    tail_passes = CUDA_PASSES[schedule_index:]
    expected_placement = {key: str(value) for key, value in kernel_set["placement"].items()}
    with pinned_knobs(expected_placement):
        placed = Pipeline.build(placement_passes).run(graph, ctx=ctx)

    actual_placement = _placement_knobs(placed)
    if actual_placement != expected_placement:
        raise KernelSetReplayError(f"placement mismatch: expected {expected_placement}, realized {actual_placement}")

    components = {row["op_key"]: row for row in kernel_set["kernels"]}
    placed_nodes = _placed_kernel_nodes(placed)
    actual_counts = Counter(op.cache_key() for _, op in placed_nodes)
    expected_counts = Counter({row["op_key"]: row["multiplicity"] for row in kernel_set["kernels"]})
    if actual_counts != expected_counts:
        raise KernelSetReplayError(f"kernel inventory mismatch: expected {dict(expected_counts)}, realized {dict(actual_counts)}")

    absorbed = {producer: consumer for consumer, _op in placed_nodes for producer in fused_producer_ids(placed, placed.nodes[consumer])}
    checked = set()
    for nid, op in placed_nodes:
        op_key = op.cache_key()
        if op_key in checked:
            continue
        checked.add(op_key)
        absorb = frozenset(producer for producer, consumer in absorbed.items() if consumer == nid)
        sub = single_node_graph(placed, nid, absorb=absorb)
        lowered_sub = _lower_exact_schedule(sub, kernel_set, tail_passes, ctx)
        actual_component = _cuda_record_knobs(lowered_sub)
        expected_component = [{key: str(value) for key, value in row.items()} for row in components[op_key]["cuda_record_knobs"]]
        _validate_component_cuda_rows(op_key, actual_component, expected_component)

    compiled = _lower_exact_schedule(placed.copy(), kernel_set, tail_passes, ctx)
    _validate_cuda_inventory(_cuda_record_knobs(compiled), kernel_set)
    return compiled
