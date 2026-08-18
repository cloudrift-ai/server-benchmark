"""Offline execution-plan realization for the complete DeepSeek V4 1Cat adapter inventory."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import partial
from typing import Any

logger = logging.getLogger(__name__)

_CAPACITY = 4096


@dataclass(frozen=True)
class ExternalProgramProfile:
    """One exact graph and build contract required by a live 1Cat adapter."""

    name: str
    family: str
    rows: int
    symbolic: bool
    expected_inputs: tuple[str, ...]
    output_count: int
    graph_factory: Callable[[], Any] = field(compare=False, repr=False)
    pins: tuple[tuple[str, str], ...] = ()
    symbolic_values: tuple[tuple[str, int], ...] = ()
    launch_count: int | None = None

    def validate(self, plan: Any) -> None:
        inputs = tuple(plan.inputs)
        outputs = tuple(plan.outputs)
        if inputs != self.expected_inputs or len(outputs) != self.output_count:
            raise RuntimeError(
                f"external-program ABI changed for {self.name}: "
                f"expected {self.expected_inputs!r} and {self.output_count} output(s), got {inputs!r} -> {outputs!r}"
            )
        if self.launch_count is not None and len(plan.launches) != self.launch_count:
            raise RuntimeError(
                f"external-program launch count changed for {self.name}: expected {self.launch_count}, got {len(plan.launches)}"
            )


def _profile(
    name: str,
    family: str,
    rows: int,
    symbolic: bool,
    expected_inputs: tuple[str, ...],
    output_count: int,
    graph_factory: Callable[[], Any],
    *,
    pins: dict[str, str] | None = None,
    symbolic_values: dict[str, int] | None = None,
    launch_count: int | None = None,
) -> ExternalProgramProfile:
    return ExternalProgramProfile(
        name=name,
        family=family,
        rows=rows,
        symbolic=symbolic,
        expected_inputs=expected_inputs,
        output_count=output_count,
        graph_factory=graph_factory,
        pins=tuple(sorted((pins or {}).items())),
        symbolic_values=tuple(sorted((symbolic_values or {}).items())),
        launch_count=launch_count,
    )


def deepseek_external_program_manifest() -> tuple[ExternalProgramProfile, ...]:
    """Return every exact external-program profile reachable by the broad adapter opt-in."""
    from emmy.compiler.loader import onecat_sm70 as physical_loader
    from emmy.serving import deepseek as deepseek_traces
    from emmy.serving import deepseek_experts as expert_traces
    from emmy.serving import mhc as mhc_traces
    from emmy.serving import onecat
    from emmy.serving import onecat_deepseek as deepseek_adapter
    from emmy.serving import onecat_experts as experts
    from emmy.serving import onecat_indexer as indexer
    from emmy.serving import onecat_linear as linear
    from emmy.serving import onecat_mhc as mhc
    from emmy.serving import onecat_output as output
    from emmy.serving import onecat_vocab as vocab

    profiles = [
        _profile(
            "final_rms_norm.symbolic.m4096",
            "final_rms_norm",
            _CAPACITY,
            True,
            ("x", "weight"),
            1,
            onecat._rms_norm_graph,
            pins={"WORK": "t256", "REDUCE": "coop"},
            symbolic_values={"num_tokens": _CAPACITY},
            launch_count=1,
        ),
        _profile(
            "qkv_rms_norm.symbolic.m4096",
            "qkv_rms_norm",
            _CAPACITY,
            True,
            ("fused_q_kv", "q_weight", "kv_weight"),
            2,
            partial(deepseek_traces.trace_fused_q_kv_rmsnorm, rows=_CAPACITY, dynamic=True),
            symbolic_values={"num_tokens": _CAPACITY},
        ),
        _profile(
            "inverse_rope.symbolic.m4096",
            "inverse_rope",
            _CAPACITY,
            True,
            ("x", "positions", "cos_sin_cache"),
            1,
            partial(deepseek_traces.trace_inverse_rope, rows=_CAPACITY, dynamic=True),
            symbolic_values={"num_tokens": _CAPACITY},
        ),
        _profile(
            "qnorm_rope.symbolic.m4096",
            "qnorm_rope",
            _CAPACITY,
            True,
            ("q", "positions", "cos_sin_cache"),
            1,
            partial(deepseek_traces.trace_qnorm_rope, rows=_CAPACITY, dynamic=True),
            pins=deepseek_adapter._QNORM_ROPE_PINS,
            symbolic_values={"num_tokens": _CAPACITY},
            launch_count=1,
        ),
    ]

    linear_contracts = (
        (64, False),
        (256, True),
        (512, True),
        (1024, True),
        (2048, True),
    )
    for width, output_fp32 in linear_contracts:
        dtype = "fp32" if output_fp32 else "fp16"
        for rows in linear.PROFILE_ROWS:
            program_profile = linear._LinearProfile(width, output_fp32, rows, symbolic=False)
            profiles.append(
                _profile(
                    f"linear.n{width}.{dtype}.static.m{rows}",
                    "linear",
                    rows,
                    False,
                    ("x", "weight"),
                    1,
                    partial(linear._linear_graph, program_profile),
                )
            )
        program_profile = linear._LinearProfile(width, output_fp32, _CAPACITY, symbolic=True)
        profiles.append(
            _profile(
                f"linear.n{width}.{dtype}.symbolic.m4096",
                "linear",
                _CAPACITY,
                True,
                ("x", "weight"),
                1,
                partial(linear._linear_graph, program_profile),
                symbolic_values={"num_tokens": _CAPACITY},
            )
        )

    for spec in physical_loader.PROJECTION_SPECS:
        for rows in physical_loader.PROFILE_ROWS:
            program_profile = physical_loader.ProjectionProfile(spec, rows)
            profiles.append(
                _profile(
                    f"retained_fp8.{spec.name}.static.m{rows}",
                    "retained_fp8",
                    rows,
                    False,
                    physical_loader.expected_inputs(),
                    1,
                    partial(physical_loader.projection_graph, program_profile),
                    pins=physical_loader.PROFILE_PINS,
                    launch_count=1,
                )
            )
        program_profile = physical_loader.ProjectionProfile(spec, _CAPACITY, symbolic=True)
        profiles.append(
            _profile(
                f"retained_fp8.{spec.name}.symbolic.m4096",
                "retained_fp8",
                _CAPACITY,
                True,
                physical_loader.expected_inputs(),
                1,
                partial(physical_loader.projection_graph, program_profile),
                pins=physical_loader.PROFILE_PINS,
                symbolic_values={"num_tokens": _CAPACITY},
                launch_count=1,
            )
        )

    mhc_builders = {
        "broadcast": mhc_traces.trace_mhc_broadcast,
        "pre": mhc_traces.trace_mhc_pre,
        "fused": mhc_traces.trace_mhc_fused,
        "post": mhc_traces.trace_mhc_post,
        "head": mhc_traces.trace_hc_head,
    }
    for kind, builder in mhc_builders.items():
        output_count = 4 if kind in ("broadcast", "fused") else 3 if kind == "pre" else 1
        for rows in mhc.PROFILE_ROWS:
            profiles.append(
                _profile(
                    f"mhc.{kind}.static.m{rows}",
                    "mhc",
                    rows,
                    False,
                    mhc._PLAN_INPUTS[kind],
                    output_count,
                    partial(builder, rows=rows),
                )
            )
        profiles.append(
            _profile(
                f"mhc.{kind}.symbolic.m4096",
                "mhc",
                _CAPACITY,
                True,
                mhc._PLAN_INPUTS[kind],
                output_count,
                partial(mhc._trace_symbolic_prefill, kind),
                symbolic_values={"num_tokens": _CAPACITY},
            )
        )

    output_graphs = {
        "embedding": output._embedding_graph,
        "lm_head": output._lm_head_graph,
        "clamp_swiglu": output._clamp_swiglu_graph,
    }
    for kind, graph_factory in output_graphs.items():
        profiles.append(
            _profile(
                f"output.{kind}.symbolic.m4096",
                "output",
                _CAPACITY,
                True,
                output._EXPECTED_INPUTS[kind],
                1,
                graph_factory,
                symbolic_values={"num_tokens": _CAPACITY},
            )
        )

    for rank in range(vocab._TP):
        profiles.extend(
            (
                _profile(
                    f"vocab.embedding.rank{rank}.symbolic.m4096",
                    "vocab_embedding",
                    _CAPACITY,
                    True,
                    vocab._EXPECTED_INPUTS["embedding"],
                    1,
                    partial(vocab._tp_embedding_graph, rank),
                    symbolic_values={"num_tokens": _CAPACITY},
                ),
                _profile(
                    f"vocab.local_top1.rank{rank}.symbolic.m4096",
                    "local_top1",
                    _CAPACITY,
                    True,
                    vocab._EXPECTED_INPUTS["local_top1"],
                    1,
                    partial(vocab._local_top1_graph, rank),
                    symbolic_values={"num_tokens": _CAPACITY},
                ),
            )
        )
    profiles.append(
        _profile(
            "vocab.rank_top1.symbolic.m4096",
            "rank_top1",
            _CAPACITY,
            True,
            vocab._EXPECTED_INPUTS["rank_top1"],
            1,
            vocab._rank_top1_graph,
            symbolic_values={"num_tokens": _CAPACITY},
        )
    )

    for kind, launches in (("learned", 3), ("hash", 2)):
        for rows in experts.PROFILE_ROWS:
            profiles.append(
                _profile(
                    f"route.{kind}.static.m{rows}",
                    f"route_{kind}",
                    rows,
                    False,
                    ("router_logits", "bias") if kind == "learned" else ("router_logits", "table", "input_ids"),
                    2,
                    partial(expert_traces.trace_deepseek_route, rows=rows, kind=kind),
                    launch_count=launches,
                )
            )
        profiles.append(
            _profile(
                f"route.{kind}.symbolic.m4096",
                f"route_{kind}",
                _CAPACITY,
                True,
                ("router_logits", "bias") if kind == "learned" else ("router_logits", "table", "input_ids"),
                2,
                partial(expert_traces.trace_deepseek_route, rows=128, kind=kind, symbolic=True),
                symbolic_values={"num_tokens": _CAPACITY},
                launch_count=launches,
            )
        )

    profiles.append(
        _profile(
            "experts.direct.symbolic.m4096",
            "routed_experts",
            _CAPACITY,
            True,
            ("x", "route_weights", "route_ids", "w13", "w2", "w13_scale", "w2_scale"),
            1,
            partial(expert_traces.trace_deepseek_experts, rows=128, symbolic=True),
            symbolic_values={"num_tokens": _CAPACITY},
            launch_count=4,
        )
    )
    for rows, (shards, _groups) in expert_traces.WIDE_EXPERT_PROFILES.items():
        profiles.extend(
            (
                _profile(
                    f"experts.grouped.bucket.m{rows}",
                    "routed_experts_wide",
                    rows,
                    False,
                    ("route_ids",),
                    3,
                    partial(expert_traces.trace_expert_bucket, rows=rows),
                    launch_count=1,
                ),
                _profile(
                    f"experts.grouped.pack.m{rows}",
                    "routed_experts_wide",
                    rows,
                    False,
                    ("x", "grouped_routes"),
                    1,
                    partial(expert_traces.trace_grouped_input, rows=rows),
                    launch_count=1,
                ),
                _profile(
                    f"experts.grouped.w13.m{rows}",
                    "routed_experts_wide",
                    rows,
                    False,
                    ("grouped_x", "w13", "group_experts", "w13_scale"),
                    1,
                    partial(expert_traces.trace_grouped_w13, rows=rows),
                    pins=experts._W13_PINS,
                    launch_count=1,
                ),
                _profile(
                    f"experts.grouped.activation.m{rows}",
                    "routed_experts_wide",
                    rows,
                    False,
                    ("gate_up",),
                    1,
                    partial(expert_traces.trace_grouped_activation, rows=rows),
                    launch_count=1,
                ),
                _profile(
                    f"experts.grouped.w2.m{rows}",
                    "routed_experts_wide",
                    rows,
                    False,
                    ("intermediate", "w2", "group_experts", "w2_scale"),
                    1,
                    partial(expert_traces.trace_grouped_w2, rows=rows),
                    pins=experts._W2_PINS,
                    launch_count=1,
                ),
                _profile(
                    f"experts.grouped.combine.m{rows}",
                    "routed_experts_wide",
                    rows,
                    False,
                    ("partials", "weights"),
                    1,
                    partial(expert_traces.trace_weighted_route_sum, rows=rows),
                    launch_count=1,
                ),
            )
        )
        for shard in range(shards):
            profiles.append(
                _profile(
                    f"experts.grouped.unbucket{shard}.m{rows}",
                    "routed_experts_wide",
                    rows,
                    False,
                    ("base", "grouped", "inverse"),
                    1,
                    partial(expert_traces.trace_expert_unbucket, rows=rows, shard_index=shard),
                    launch_count=1,
                )
            )
    profiles.append(
        _profile(
            "indexer.q_rope_weights.symbolic.m4096",
            "indexer_q",
            _CAPACITY,
            True,
            indexer._EXPECTED_INPUTS,
            2,
            indexer._indexer_q_graph,
            symbolic_values={"num_tokens": _CAPACITY},
        )
    )

    ordered = tuple(sorted(profiles, key=lambda profile: profile.name))
    names = tuple(profile.name for profile in ordered)
    if len(names) != len(set(names)):
        raise RuntimeError("DeepSeek external-program manifest contains duplicate profile names")
    return ordered


def prewarm_deepseek_external_programs(
    profiles: Iterable[ExternalProgramProfile] | None = None,
    *,
    realize: Callable[..., Any] | None = None,
) -> tuple[str, ...]:
    """Realize and verify the exact manifest, raising on the first incomplete pack."""
    if realize is None:
        from emmy.serving.external import realize_external_plan

        realize = realize_external_plan

    manifest = tuple(profiles) if profiles is not None else deepseek_external_program_manifest()
    names = tuple(profile.name for profile in manifest)
    if names != tuple(sorted(names)) or len(names) != len(set(names)):
        raise ValueError("DeepSeek external-program profiles must have unique names in deterministic order")

    for index, profile in enumerate(manifest, start=1):
        logger.info("1Cat prewarm: realizing %d/%d %s", index, len(manifest), profile.name)
        try:
            plan = realize(
                profile.graph_factory(),
                pins=dict(profile.pins),
                symbolic_values=dict(profile.symbolic_values),
            )
            profile.validate(plan)
        except Exception as exc:
            raise RuntimeError(f"1Cat prewarm failed at profile {profile.name} ({index}/{len(manifest)})") from exc
    logger.info("1Cat prewarm: realized and verified all %d external-program profiles", len(manifest))
    return names


def main() -> None:
    """Run the release-blocking DeepSeek external-program realization pass."""
    from emmy.logging_setup import setup_cli_logging

    setup_cli_logging()
    prewarm_deepseek_external_programs()


if __name__ == "__main__":
    main()
