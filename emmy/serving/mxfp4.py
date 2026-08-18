"""Packed expert-stage programs for the 1Cat serving adapter.

Checkpoint spelling stays in :mod:`emmy.compiler.loader.quant`.  This module
only constructs the ordinary Gather + batched-matmul graph that consumes the
resulting byte, integer, and floating-point algebra.
"""

from __future__ import annotations


class RoutedMxfp4StageModule:
    """Select one compact expert per already-permuted row and multiply it."""

    def __new__(cls):
        import torch

        class Module(torch.nn.Module):
            def forward(self, x, weight, expert_ids):
                selected = weight[expert_ids]
                return torch.bmm(x.unsqueeze(1), selected.transpose(1, 2)).squeeze(1)

        return Module()


class GroupedRowsModule:
    """Pad compact expert-sorted rows into fixed per-group tiles."""

    def __init__(self, rows_per_group: int):
        self.rows_per_group = int(rows_per_group)

    def module(self):
        import torch

        rows_per_group = self.rows_per_group

        class Module(torch.nn.Module):
            def forward(self, x, offsets):
                local_rows = torch.arange(rows_per_group, device=x.device, dtype=offsets.dtype)
                rows = offsets[:-1, None] + local_rows[None, :]
                valid = rows < offsets[1:, None]
                safe_rows = torch.where(valid, rows, 0)
                grouped = x[safe_rows]
                return torch.where(valid.unsqueeze(-1), grouped, 0.0)

        return Module()


class GroupedMxfp4StageModule:
    """Multiply fixed row tiles by one selected compact expert per group."""

    def __new__(cls):
        import torch

        class Module(torch.nn.Module):
            def forward(self, grouped_x, weight, expert_ids):
                selected = weight[expert_ids]
                return torch.matmul(grouped_x, selected.transpose(1, 2))

        return Module()


class CompactGroupedRowsModule:
    """Return fixed grouped tiles to their compact expert-sorted row order."""

    def __init__(self, rows: int, rows_per_group: int):
        self.rows = int(rows)
        self.rows_per_group = int(rows_per_group)

    def module(self):
        import torch

        rows = self.rows
        rows_per_group = self.rows_per_group

        class Module(torch.nn.Module):
            def forward(self, grouped, offsets):
                row_ids = torch.arange(rows, device=grouped.device, dtype=offsets.dtype)
                group_ids = (row_ids[:, None] >= offsets[1:][None, :]).to(torch.int32).sum(dim=-1)
                local_rows = row_ids - offsets[group_ids]
                padded_rows = group_ids * rows_per_group + local_rows
                return grouped.flatten(0, 1)[padded_rows]

        return Module()


def trace_routed_mxfp4_stage(
    *,
    rows: int,
    experts: int,
    out_features: int,
    in_features: int,
    storage=None,
):
    """Trace and birth-spell one routed packed-weight stage.

    Rows are static so fusion can prove the Gather narrows the expert store
    before the compact decode cone enters the contraction.  A symbolic row
    extent can make materializing the decoded expert store look cheaper than
    replaying its decode per row; that is not a legal serving program.
    """
    import torch

    from emmy.compiler.loader.quant import spell_mxfp4_inputs
    from emmy.compiler.trace.torch import trace_module

    examples = (
        torch.empty((rows, in_features), dtype=torch.float16, device="meta"),
        torch.empty((experts, out_features, in_features), dtype=torch.float16, device="meta"),
        torch.empty((rows,), dtype=torch.int32, device="meta"),
    )
    graph = trace_module(RoutedMxfp4StageModule(), examples)
    storage = storage or (
        (experts, out_features, in_features // 2),
        (experts, out_features, in_features // 32),
    )
    spell_mxfp4_inputs(graph, {"weight": storage})
    return graph


def trace_grouped_mxfp4_stage(
    *,
    groups: int,
    rows_per_group: int,
    experts: int,
    out_features: int,
    in_features: int,
    storage=None,
):
    """Trace the contraction in a fixed grouped-row representation."""
    import torch

    from emmy.compiler.loader.quant import spell_mxfp4_inputs
    from emmy.compiler.trace.torch import trace_module

    graph = trace_module(
        GroupedMxfp4StageModule(),
        (
            torch.empty((groups, rows_per_group, in_features), dtype=torch.float16, device="meta"),
            torch.empty((experts, out_features, in_features), dtype=torch.float16, device="meta"),
            torch.empty((groups,), dtype=torch.int32, device="meta"),
        ),
    )
    storage = storage or (
        (experts, out_features, in_features // 2),
        (experts, out_features, in_features // 32),
    )
    spell_mxfp4_inputs(graph, {"weight": storage})
    return graph


def trace_grouped_rows(*, rows: int, groups: int, rows_per_group: int, features: int):
    """Trace the fixed decode-row packing step around a grouped expert stage."""
    import torch

    from emmy.compiler.trace.torch import trace_module

    return trace_module(
        GroupedRowsModule(rows_per_group).module(),
        (
            torch.empty((rows, features), dtype=torch.float16, device="meta"),
            torch.empty((groups + 1,), dtype=torch.int32, device="meta"),
        ),
    )


def trace_compact_grouped_rows(*, rows: int, groups: int, rows_per_group: int, features: int):
    """Trace the inverse packing step that restores compact sorted-row order."""
    import torch

    from emmy.compiler.trace.torch import trace_module

    return trace_module(
        CompactGroupedRowsModule(rows, rows_per_group).module(),
        (
            torch.empty((groups, rows_per_group, features), dtype=torch.float16, device="meta"),
            torch.empty((groups + 1,), dtype=torch.int32, device="meta"),
        ),
    )
