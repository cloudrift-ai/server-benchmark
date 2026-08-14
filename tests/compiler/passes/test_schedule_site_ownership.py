"""Schedule-site ownership regressions."""

from __future__ import annotations

import torch


class _NormalizedGqa(torch.nn.Module):
    def forward(self, q, k, v):
        q = (q.float() * torch.rsqrt(torch.mean(q.float() ** 2, dim=-1, keepdim=True) + 1e-6)).half()
        k = (k.float() * torch.rsqrt(torch.mean(k.float() ** 2, dim=-1, keepdim=True) + 1e-6)).half()
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, enable_gqa=True)


def test_nested_only_thread_work_does_not_invent_root_tile():
    """A nested cooperative reduce owns WORK without creating a TILE slice at its root."""
    from emmy.compiler import target as target_mod  # noqa: PLC0415
    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.compiler.pipeline import CUDA_PASSES, LOOP_PASSES, Pass, Pipeline  # noqa: PLC0415
    from emmy.compiler.pipeline.fork import flatten_leaves  # noqa: PLC0415
    from emmy.compiler.pipeline.passes.lowering.tile._schedule import schedule  # noqa: PLC0415
    from emmy.compiler.trace.torch import trace_module  # noqa: PLC0415

    target_mod.set_target((9, 0))
    try:
        ctx = Context.from_target((9, 0), gpu_name="NVIDIA H200")
        q = torch.randn(1, 16, 128, 128, dtype=torch.float16)
        k, v = (torch.randn(1, 8, 128, 128, dtype=torch.float16) for _ in range(2))
        graph = trace_module(_NormalizedGqa(), (q, k, v))
        if "lowering/schedule" in CUDA_PASSES:
            placed = Pipeline.build(CUDA_PASSES[: CUDA_PASSES.index("lowering/schedule")]).run(graph, ctx=ctx)
        else:
            prefix = Pipeline.build(LOOP_PASSES)
            recognize = Pass.load("lowering/tile", len(prefix.passes), {"010_recognize"})
            placed = Pipeline([*prefix.passes, recognize]).run(graph, ctx=ctx)
        tile = placed.nodes["scaled_dot_product_attention"].op
        leaves = flatten_leaves([schedule(tile, "scaled_dot_product_attention", {}, ctx)])
        matching = [
            leaf
            for leaf in leaves
            if leaf.knobs.get("WORK") == "t128"
            and leaf.knobs.get("REDUCE") == ""
            and leaf.knobs.get("REDUCE@flash_k_stat") == "coop"
            and "TILE" not in leaf.knobs
        ]

        assert matching, "the fused GQA schedule must offer the nested-only cooperative row"
        materialized = matching[0].expand()[0]
        assert "TILE" not in materialized.schedule
        assert materialized.schedule["REDUCE@flash_k_stat"].coop == 128
    finally:
        target_mod.set_target(None)
