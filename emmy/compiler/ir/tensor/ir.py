"""Minimal tensor IR — the dialect that survives decomposition.

After decomposition rewrites the frontend ops (``LinearOp``, ``MatmulOp``,
``SdpaOp``, ``MeanOp``, ``UnsqueezeOp``, ``TransposeOp``, ``ReshapeOp``,
``SliceOp``, ``CatOp``) into their primitives, only this set of ops should
remain in the graph:

- ``ElementwiseOp`` — scalar function per element (add, mul, exp, silu, ...).
- ``CastOp`` / ``BitcastOp`` — numeric conversion / same-width bit reinterpretation.
- ``RangeOp`` — a static one-dimensional integer sequence.
- ``FixedSinkhornOp`` — bounded static FP32 Sinkhorn normalization.
- ``RowRmsNormRopeOp`` — one row RMSNorm followed by partial interleaved RoPE.
- ``StableTopKOp`` / ``IndexedTopKOp`` — deterministic fixed-width selection.
- ``ExpertBucketOp`` / ``RouteUnbucketOp`` / ``WeightedRouteSumOp`` — stable routed-row layout and combine.
- ``ReduceOp`` — collapse one axis via an associative binary op.
- ``ScanOp`` — cumulative variant of ``ReduceOp``.
- ``GatherOp`` / ``ScatterOp`` — data-dependent reads/writes along an axis.
- ``IndexMapOp`` — unified layout-only op (subsumes slice / cat / transpose
  / reshape / unsqueeze) described by affine coord arithmetic over
  placeholder vars from ``ir.expr``.

Plus the boundary sentinels ``InputOp`` and ``ConstantOp`` from ``ir.base``.
The ``lifting/`` pass wraps each tensor op in a trivial ``ir.loop.LoopOp``
and the ``fusion/`` pass splices adjacent LoopOp pairs via the
tree-splicer in ``ir/loop/splicer.py``.

Op metadata (arity / commutative / reducer identity) lives on
``ir.expr.ElementwiseImpl`` — the single source of truth shared across
elementwise, reduce, scan, and accumulator use sites; read straight from
``op.op.arity`` / ``op.op.identity`` etc.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from emmy.compiler.dim import Dim, to_dim
from emmy.compiler.dtype import get as get_dtype
from emmy.compiler.ir.base import Op, _keepdim_axis
from emmy.compiler.ir.elementwise import _REDUCE_SPELLING, ElementwiseImpl

# ---------------------------------------------------------------------------
# Value construction / conversion
# ---------------------------------------------------------------------------


@dataclass
class RangeOp(Op):
    """Construct the static one-dimensional sequence ``[start, stop)`` by ``step``.

    This is the tensor-IR counterpart of ``range`` / ``arange``. The dtype is explicit
    because the op has no input from which the interpreter could derive it.
    """

    start: int = 0
    stop: int = 0
    step: int = 1
    dtype: str = "i64"

    def __post_init__(self) -> None:
        if self.step == 0:
            raise ValueError("RangeOp step must be non-zero")
        get_dtype(self.dtype)

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        return (len(range(self.start, self.stop, self.step)),)

    def forward(self, *inputs):
        return np.arange(self.start, self.stop, self.step, dtype=get_dtype(self.dtype).np)


@dataclass
class CastOp(Op):
    """Numerically convert every input element to ``dtype``."""

    dtype: str = "f32"

    def __post_init__(self) -> None:
        get_dtype(self.dtype)

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        return tuple(input_shapes[0])

    def forward(self, *inputs):
        return np.asarray(inputs[0]).astype(get_dtype(self.dtype).np)


@dataclass
class BitcastOp(Op):
    """Reinterpret every input element as a same-width ``dtype`` without changing bits."""

    dtype: str = "u16"

    def __post_init__(self) -> None:
        get_dtype(self.dtype)

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        return tuple(input_shapes[0])

    def forward(self, *inputs):
        source = np.ascontiguousarray(inputs[0])
        target = get_dtype(self.dtype).np
        if source.dtype.itemsize != target.itemsize:
            raise ValueError(f"BitcastOp requires equal element widths, got {source.dtype} and {target}")
        return source.view(target)


# ---------------------------------------------------------------------------
# Elementwise / reduce / scan
# ---------------------------------------------------------------------------


@dataclass
class FixedSinkhornOp(Op):
    """Bounded static FP32 Sinkhorn normalization over square matrix batches.

    The operation starts with a stable row softmax plus ``eps``, performs one
    column normalization, then ``iterations - 1`` row/column pairs. Static
    bounds keep its Loop-IR lowering finite enough to remain register-resident.
    """

    MAX_SIZE: ClassVar[int] = 8
    MAX_ITERATIONS: ClassVar[int] = 32

    eps: float = 1e-6
    iterations: int = 20

    def __post_init__(self) -> None:
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError(f"FixedSinkhornOp eps must be finite and positive, got {self.eps}")
        if not 1 <= self.iterations <= self.MAX_ITERATIONS:
            raise ValueError(f"FixedSinkhornOp iterations must be in [1,{self.MAX_ITERATIONS}], got {self.iterations}")

    def matrix_size(self, shape: tuple) -> int:
        if len(shape) != 3:
            raise ValueError(f"FixedSinkhornOp requires rank-3 [M,N,N] input, got shape {shape}")
        dims = tuple(to_dim(dim) for dim in shape)
        if any(not dim.is_static for dim in dims[-2:]):
            raise ValueError(f"FixedSinkhornOp requires static matrix dimensions, got {shape[-2:]}")
        rows, cols = (dim.as_static() for dim in dims[-2:])
        if rows != cols:
            raise ValueError(f"FixedSinkhornOp requires square matrices, got {rows}x{cols}")
        if not 1 <= rows <= self.MAX_SIZE:
            raise ValueError(f"FixedSinkhornOp matrix size must be in [1,{self.MAX_SIZE}], got {rows}")
        return rows

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        shape = tuple(input_shapes[0])
        self.matrix_size(shape)
        return shape

    def forward(self, *inputs):
        values = np.asarray(inputs[0])
        self.matrix_size(values.shape)
        if values.dtype != np.float32:
            raise TypeError(f"FixedSinkhornOp requires float32 input, got {values.dtype}")
        eps = np.float32(self.eps)
        values = np.exp(values - np.max(values, axis=-1, keepdims=True))
        values = values / np.sum(values, axis=-1, keepdims=True) + eps
        values = values / (np.sum(values, axis=-2, keepdims=True) + eps)
        for _ in range(self.iterations - 1):
            values = values / (np.sum(values, axis=-1, keepdims=True) + eps)
            values = values / (np.sum(values, axis=-2, keepdims=True) + eps)
        return values


@dataclass
class RowRmsNormRopeOp(Op):
    """Per-row RMSNorm followed by GPT-J interleaved RoPE on a suffix."""

    rope_dim: int = 64
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.rope_dim <= 0 or self.rope_dim % 2:
            raise ValueError(f"RowRmsNormRopeOp rope_dim must be positive and even, got {self.rope_dim}")
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError(f"RowRmsNormRopeOp eps must be finite and positive, got {self.eps}")

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        if len(input_shapes) != 3:
            raise ValueError(f"RowRmsNormRopeOp requires Q, positions, and cache inputs, got {len(input_shapes)}")
        q_shape, positions_shape, cache_shape = (tuple(shape) for shape in input_shapes)
        if len(q_shape) != 3:
            raise ValueError(f"RowRmsNormRopeOp requires rank-3 Q, got {q_shape}")
        head_dim = to_dim(q_shape[-1])
        if not head_dim.is_static or self.rope_dim >= head_dim.as_static():
            raise ValueError(f"RowRmsNormRopeOp requires static head_dim > rope_dim, got {q_shape[-1]} and {self.rope_dim}")
        if positions_shape != (q_shape[0],):
            raise ValueError(f"RowRmsNormRopeOp positions must match Q rows, got {positions_shape} and {q_shape}")
        if len(cache_shape) != 2 or to_dim(cache_shape[-1]) != to_dim(self.rope_dim):
            raise ValueError(f"RowRmsNormRopeOp cache must end in rope_dim={self.rope_dim}, got {cache_shape}")
        return q_shape

    def forward(self, *inputs):
        q, positions, cache = inputs
        q = np.asarray(q)
        shape = self.infer_output_shape([q.shape, np.asarray(positions).shape, np.asarray(cache).shape])
        values = q.astype(np.float32)
        rrms = np.float32(1.0) / np.sqrt(np.mean(values * values, axis=-1, keepdims=True) + np.float32(self.eps))
        normalized = values * rrms
        rotary = np.asarray(cache)[np.asarray(positions, dtype=np.int64)]
        cos, sin = np.split(rotary.astype(np.float32), 2, axis=-1)
        cos = cos[:, None, :]
        sin = sin[:, None, :]
        pairs = normalized[..., -self.rope_dim :].reshape(*shape[:-1], self.rope_dim // 2, 2)
        even, odd = pairs[..., 0], pairs[..., 1]
        rotated = np.stack((even * cos - odd * sin, odd * cos + even * sin), axis=-1)
        output = np.concatenate((normalized[..., : -self.rope_dim], rotated.reshape(*shape[:-1], self.rope_dim)), axis=-1)
        return output.astype(q.dtype)


@dataclass
class StableTopKOp(Op):
    """Select the highest ``k`` row values with stable lower-index ties.

    The first input ranks candidates and the second supplies returned values.
    This separation preserves graph-level correction arithmetic exactly.
    """

    k: int = 1
    scale: float = 1.0
    normalize: bool = True

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError(f"StableTopKOp k must be positive, got {self.k}")
        if not math.isfinite(self.scale):
            raise ValueError(f"StableTopKOp scale must be finite, got {self.scale}")

    def _shape(self, input_shapes: list[tuple]) -> tuple:
        if len(input_shapes) != 2 or tuple(input_shapes[0]) != tuple(input_shapes[1]):
            raise ValueError(f"StableTopKOp requires equal rank/payload matrices, got {input_shapes}")
        shape = tuple(input_shapes[0])
        if len(shape) != 2:
            raise ValueError(f"StableTopKOp requires rank-2 [rows,candidates] inputs, got {shape}")
        candidates = to_dim(shape[1])
        if not candidates.is_static or self.k > candidates.as_static():
            raise ValueError(f"StableTopKOp requires static candidates >= k, got {shape[1]} and k={self.k}")
        return (shape[0], self.k)

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        return self._shape(input_shapes)

    def forward(self, *inputs):
        ranking = np.asarray(inputs[0], dtype=np.float32)
        payload = np.asarray(inputs[1], dtype=np.float32)
        out_shape = self._shape([ranking.shape, payload.shape])
        weights = np.empty(out_shape, dtype=np.float32)
        indices = np.empty(out_shape, dtype=np.int32)
        scale = np.float32(self.scale)
        for row in range(ranking.shape[0]):
            selected: list[int] = []
            total = np.float32(0.0)
            for slot in range(self.k):
                best_index = -1
                best_value = np.float32(-np.inf)
                for candidate in range(ranking.shape[1]):
                    if candidate in selected:
                        continue
                    value = ranking[row, candidate]
                    if best_index < 0 or value > best_value:
                        best_value = value
                        best_index = candidate
                selected.append(best_index)
                indices[row, slot] = best_index
                weights[row, slot] = payload[row, best_index]
                total = np.float32(total + weights[row, slot])
            denominator = total if self.normalize and total > 0 else np.float32(1.0)
            factor = np.float32(scale / denominator) if self.normalize else scale
            for slot in range(self.k):
                weights[row, slot] = np.float32(weights[row, slot] * factor)
        return weights, indices


@dataclass
class IndexedTopKOp(Op):
    """Gather fixed row candidates and normalize with an explicit FP32 order.

    ``reduction_lanes`` and ``lane_chunk`` define observable addition order:
    contiguous candidate chunks accumulate per lane, then an XOR tree reduces
    lanes. They are numerical semantics, not performance-selection knobs.
    """

    k: int = 1
    scale: float = 1.0
    normalize: bool = True
    reduction_lanes: int = 1
    lane_chunk: int = 1

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError(f"IndexedTopKOp k must be positive, got {self.k}")
        if not math.isfinite(self.scale):
            raise ValueError(f"IndexedTopKOp scale must be finite, got {self.scale}")
        if self.reduction_lanes not in {1, 2, 4, 8, 16, 32} or self.lane_chunk < 1:
            raise ValueError("IndexedTopKOp reduction_lanes must be a power of two <=32 and lane_chunk must be positive")

    def _shape(self, input_shapes: list[tuple]) -> tuple:
        if len(input_shapes) != 3:
            raise ValueError(f"IndexedTopKOp requires payload, table, and row indices, got {input_shapes}")
        payload, table, row_indices = (tuple(shape) for shape in input_shapes)
        if len(payload) != 2 or len(table) != 2 or len(row_indices) != 1 or payload[0] != row_indices[0]:
            raise ValueError(f"IndexedTopKOp requires [M,E], [V,K], [M], got {input_shapes}")
        table_width = to_dim(table[1])
        if not table_width.is_static or table_width.as_static() != self.k:
            raise ValueError(f"IndexedTopKOp table width must equal k={self.k}, got {table[1]}")
        return (payload[0], self.k)

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        return self._shape(input_shapes)

    def forward(self, *inputs):
        payload = np.asarray(inputs[0], dtype=np.float32)
        table = np.asarray(inputs[1])
        row_indices = np.asarray(inputs[2])
        out_shape = self._shape([payload.shape, table.shape, row_indices.shape])
        weights = np.empty(out_shape, dtype=np.float32)
        selected = np.empty(out_shape, dtype=np.int32)
        scale = np.float32(self.scale)
        period = self.reduction_lanes * self.lane_chunk
        for row in range(payload.shape[0]):
            lane_totals = np.zeros((self.reduction_lanes,), dtype=np.float32)
            for slot in range(self.k):
                candidate = int(table[int(row_indices[row]), slot])
                selected[row, slot] = candidate
                weights[row, slot] = payload[row, candidate]
                lane = (candidate % period) // self.lane_chunk
                lane_totals[lane] = np.float32(lane_totals[lane] + weights[row, slot])
            mask = self.reduction_lanes // 2
            while mask:
                prior = lane_totals.copy()
                for lane in range(self.reduction_lanes):
                    lane_totals[lane] = np.float32(prior[lane] + prior[lane ^ mask])
                mask //= 2
            total = lane_totals[0]
            denominator = total if self.normalize and total > 0 else np.float32(1.0)
            factor = np.float32(scale / denominator) if self.normalize else scale
            for slot in range(self.k):
                weights[row, slot] = np.float32(weights[row, slot] * factor)
        return weights, selected


@dataclass
class ExpertBucketOp(Op):
    """Group routes by expert into fixed-width tiles.

    Padding routes are ``-1``. ``inverse`` maps each original route to its
    flattened grouped-row position, so a later operation can restore the
    original route order without depending on within-expert tile order.
    """

    experts: int = 1
    routes: int = 1
    rows_per_group: int = 16

    def __post_init__(self) -> None:
        if self.experts < 1 or self.routes < 1 or self.rows_per_group < 1:
            raise ValueError("ExpertBucketOp experts, routes, and rows_per_group must be positive")

    def output_shapes(self, input_shape: tuple) -> tuple[tuple, tuple, tuple]:
        shape = tuple(input_shape)
        if len(shape) != 2:
            raise ValueError(f"ExpertBucketOp requires rank-2 [rows,routes] input, got {shape}")
        rows, routes = (to_dim(dim) for dim in shape)
        if not rows.is_static or not routes.is_static or routes.as_static() != self.routes:
            raise ValueError(f"ExpertBucketOp requires static routes={self.routes}, got {shape}")
        total = rows.as_static() * self.routes
        nonempty = min(total, self.experts)
        groups = nonempty + max(0, total - nonempty + self.rows_per_group - 1) // self.rows_per_group
        return (groups, self.rows_per_group), (groups,), shape

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        return self.output_shapes(tuple(input_shapes[0]))[0]

    def forward(self, *inputs):
        ids = np.asarray(inputs[0])
        grouped_shape, experts_shape, inverse_shape = self.output_shapes(ids.shape)
        grouped_routes = np.full(grouped_shape, -1, dtype=np.int32)
        group_experts = np.zeros(experts_shape, dtype=np.int32)
        inverse = np.empty(inverse_shape, dtype=np.int32)
        flat_ids = ids.reshape(-1)
        if np.any(flat_ids < 0) or np.any(flat_ids >= self.experts):
            raise ValueError(f"ExpertBucketOp expert IDs must be in [0,{self.experts})")
        counts = np.bincount(flat_ids, minlength=self.experts)
        group_base = np.empty((self.experts,), dtype=np.int32)
        next_group = 0
        for expert, count in enumerate(counts):
            group_base[expert] = next_group
            expert_groups = (int(count) + self.rows_per_group - 1) // self.rows_per_group
            group_experts[next_group : next_group + expert_groups] = expert
            next_group += expert_groups
        seen = np.zeros((self.experts,), dtype=np.int32)
        for route, expert_value in enumerate(flat_ids):
            expert = int(expert_value)
            position = int(seen[expert])
            seen[expert] += 1
            group = int(group_base[expert]) + position // self.rows_per_group
            lane = position % self.rows_per_group
            grouped_routes[group, lane] = route
            inverse.reshape(-1)[route] = group * self.rows_per_group + lane
        return grouped_routes, group_experts, inverse


@dataclass
class RouteUnbucketOp(Op):
    """Restore one shard of grouped route outputs into a route-ordered tensor."""

    rows_per_group: int = 16
    shard_index: int = 0

    def __post_init__(self) -> None:
        if self.rows_per_group < 1 or self.shard_index < 0:
            raise ValueError("RouteUnbucketOp rows_per_group must be positive and shard_index must be nonnegative")

    def _shape(self, input_shapes: list[tuple]) -> tuple:
        if len(input_shapes) != 3:
            raise ValueError(f"RouteUnbucketOp requires base, grouped values, and inverse indices, got {input_shapes}")
        base, grouped, inverse = (tuple(shape) for shape in input_shapes)
        if len(base) != 2 or len(grouped) != 3 or len(inverse) != 2:
            raise ValueError(f"RouteUnbucketOp requires [R,H], [G,P,H], [M,K], got {input_shapes}")
        if grouped[1] != self.rows_per_group or grouped[2] != base[1] or inverse[0] * inverse[1] != base[0]:
            raise ValueError(f"RouteUnbucketOp shapes are incompatible: {input_shapes}")
        return base

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        return self._shape(input_shapes)

    def forward(self, *inputs):
        base, grouped, inverse = (np.asarray(value) for value in inputs)
        self._shape([base.shape, grouped.shape, inverse.shape])
        output = base.copy()
        grouped_rows = grouped.shape[0] * self.rows_per_group
        shard_start = self.shard_index * grouped_rows
        inverse_flat = inverse.reshape(-1)
        for route, grouped_row_value in enumerate(inverse_flat):
            local_row = int(grouped_row_value) - shard_start
            if 0 <= local_row < grouped_rows:
                output[route] = grouped.reshape(grouped_rows, grouped.shape[-1])[local_row]
        return output


@dataclass
class WeightedRouteSumOp(Op):
    """Combine fixed route slots in FP32 order and narrow once to FP16."""

    routes: int = 1

    def __post_init__(self) -> None:
        if self.routes < 1:
            raise ValueError(f"WeightedRouteSumOp routes must be positive, got {self.routes}")

    def _shape(self, input_shapes: list[tuple]) -> tuple:
        if len(input_shapes) != 2:
            raise ValueError(f"WeightedRouteSumOp requires partials and weights, got {input_shapes}")
        partials, weights = (tuple(shape) for shape in input_shapes)
        if len(partials) != 3 or len(weights) != 2 or partials[:2] != weights or partials[1] != self.routes:
            raise ValueError(f"WeightedRouteSumOp requires [M,{self.routes},H] and [M,{self.routes}], got {input_shapes}")
        return partials[0], partials[2]

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        return self._shape(input_shapes)

    def forward(self, *inputs):
        partials = np.asarray(inputs[0])
        weights = np.asarray(inputs[1], dtype=np.float32)
        shape = self._shape([partials.shape, weights.shape])
        output = np.zeros(shape, dtype=np.float32)
        for slot in range(self.routes):
            output = np.asarray(output + partials[:, slot].astype(np.float32) * weights[:, slot, None], dtype=np.float32)
        return output.astype(np.float16)


@dataclass
class _ElementwiseImplOp(Op):
    """Shared base for ops carrying an ``ElementwiseImpl`` combine in ``op``.

    Centralizes the str→``ElementwiseImpl`` coercion and the ``name`` / ``fn``
    accessors that ``ElementwiseOp`` / ``ReduceOp`` / ``ScanOp`` all expose.
    Subclasses redeclare ``op`` only to change its default spelling.
    """

    op: ElementwiseImpl = field(default_factory=lambda: ElementwiseImpl("copy"))

    def __post_init__(self) -> None:
        if isinstance(self.op, str):
            object.__setattr__(self, "op", ElementwiseImpl(self.op))

    @property
    def name(self) -> str:
        """String name of the inner ElementwiseImpl — convenient for readers + tests."""
        return self.op.name

    @property
    def fn(self) -> str:
        """Alias for ``name`` — kept for pattern-matcher ``constraints={"fn": ...}``."""
        return self.op.name


@dataclass
class ElementwiseOp(_ElementwiseImplOp):
    """Apply a scalar function independently to each element.

    The ``op`` field is an ``ElementwiseImpl`` carrying the function's name +
    arity + commutativity + (for reducer use) identity.
    """

    @property
    def arity(self) -> int:
        return self.op.arity

    @property
    def commutative(self) -> bool:
        return self.op.commutative

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        """Elementwise is rank-preserving with no implicit broadcasting:
        every input must have shape equal to the output. Broadcasts must be
        expressed as explicit ``IndexMapOp`` wrappers upstream (the
        decomposition rules use the ``broadcast_to`` helper for this).
        """
        if not input_shapes:
            return ()
        head = tuple(input_shapes[0])
        for s in input_shapes[1:]:
            if tuple(s) != head:
                shapes_fmt = [tuple(s) for s in input_shapes]
                raise ValueError(
                    f"ElementwiseOp({self.op.name!r}) input shapes must all match output; "
                    f"got {shapes_fmt}. Wrap in IndexMapOp (pipeline/passes/frontend/decomposition/_broadcast.broadcast_to)."
                )
        return head

    def forward(self, *inputs):
        # No shape check here — inside a LoopOp body, forward is called
        # per-iteration on scalar values, so a tensor-level match assert
        # doesn't apply. infer_output_shape enforces it at the graph level.
        return self.op(*inputs)


@dataclass
class ReduceLikeOp(_ElementwiseImplOp):
    """Shared base for the axis-folding ops (``ReduceOp`` / ``ScanOp``): a ``sum``-default
    ``ElementwiseImpl`` combine over one ``axis``, resolved to its numpy spelling."""

    op: ElementwiseImpl = field(default_factory=lambda: ElementwiseImpl("sum"))
    axis: int | str = 0

    def _spelling(self):
        """The ``ReduceSpelling`` for this combine, or raise if it has none."""
        spelling = _REDUCE_SPELLING.get(self.op.reduce_canon)
        if spelling is None:
            raise NotImplementedError(f"{type(self).__name__}.forward: unknown fn {self.op.name!r}")
        return spelling


@dataclass
class ReduceOp(ReduceLikeOp):
    """Collapse one or more dimensions via an associative binary op.

    ``op`` is the combine (``sum`` / ``max`` / ``prod`` / …); ``axis`` is
    the reduced dimension (concrete int or symbolic name).
    """

    @property
    def identity(self) -> float:
        return self.op.identity if self.op.identity is not None else 0.0

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        return _keepdim_axis(input_shapes[0], self.axis)

    def forward(self, *inputs):
        return self._spelling().np_reduce(inputs[0], axis=self.axis, keepdims=True)


@dataclass
class ScanOp(ReduceLikeOp):
    """Cumulative application of an associative binary op along an axis."""

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        return tuple(input_shapes[0])  # scan preserves shape

    def forward(self, *inputs):
        spelling = self._spelling()
        if spelling.np_scan is None:
            raise NotImplementedError(f"ScanOp.forward: unknown fn {self.op.name!r}")
        return spelling.np_scan(inputs[0], axis=self.axis)


# ---------------------------------------------------------------------------
# Gather / scatter
# ---------------------------------------------------------------------------


@dataclass
class GatherOp(Op):
    """Read elements from arbitrary positions along an axis."""

    axis: int | str = 0

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        # Output shape = input shape with the gather axis sized by the index input.
        # Conservative fallback: keep input shape (callers should pre-size if needed).
        return tuple(input_shapes[0])

    def forward(self, *inputs):

        data, indices = inputs[0], inputs[1].astype(np.intp)
        axis = self.axis if self.axis >= 0 else data.ndim + self.axis
        # Three semantics share this op (see ``lift_gather``):
        # - ``torch.gather`` — idx and data same rank with matching non-axis
        #   dims; one idx value per output cell. Use ``take_along_axis``.
        # - ``embedding`` / ``index_select`` — output rank is ``idx.ndim +
        #   data.ndim - 1`` with idx contributing the slice axes. Use
        #   ``np.take`` on the gather axis.
        same_rank = data.ndim == indices.ndim
        if same_rank and all(indices.shape[k] == data.shape[k] for k in range(data.ndim) if k != axis):
            return np.take_along_axis(data, indices, axis=axis)
        return np.take(data, indices, axis=axis)


@dataclass
class ScatterOp(Op):
    """Write (or reduce) values into arbitrary positions along an axis."""

    axis: int | str = 0
    reduce_fn: str | None = None  # None = overwrite, "sum" = scatter-add

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        return tuple(input_shapes[0])  # scatter preserves the destination shape

    def forward(self, *inputs):

        dest, indices, values = inputs[0].copy(), inputs[1].astype(np.intp), inputs[2]
        if self.reduce_fn == "sum":
            np.add.at(dest, (np.arange(dest.shape[0])[:, None], indices), values)
        else:
            np.put_along_axis(dest, indices, values, axis=self.axis)
        return dest


# ---------------------------------------------------------------------------
# Unified layout op (subsumes Slice/Cat/Transpose/Reshape/Unsqueeze)
# ---------------------------------------------------------------------------


@dataclass
class IndexSource:
    """One input source for an IndexMapOp.

    ``coord_map[i]`` is an ``Expr`` producing the input's i-th index from
    placeholder vars ``Var("out_coord_0")``, ``Var("out_coord_1")``, ...
    See ``emmy.compiler.ir.expr`` for the placeholder convention and
    substitution helpers.

    ``select`` is None for single-source ops; for multi-source IndexMaps
    (cat) it's a boolean ``Expr`` selecting which output positions read
    this source.
    """

    input_idx: int  # position in IndexMapOp's input list
    coord_map: tuple  # tuple[Expr, ...] — kept untyped to avoid forward-reference clutter
    select: object | None = None  # Expr | None


@dataclass
class IndexMapOp(Op):
    """Compute output by reindexing inputs via affine coord arithmetic.

    Subsumes Slice, Cat, Transpose, Reshape, Unsqueeze — every layout-only
    op is a function from output coordinates to input coordinates.
    Multi-source forms (cat) use ``select`` on each source to pick which
    output positions read which input.
    """

    out_shape: tuple[Dim, ...] = ()
    sources: tuple[IndexSource, ...] = ()

    def __post_init__(self) -> None:
        if any(not isinstance(d, Dim) for d in self.out_shape):
            self.out_shape = tuple(to_dim(d) for d in self.out_shape)

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        return tuple(self.out_shape)

    def forward(self, *inputs):
        shape = tuple(d.as_static() for d in self.out_shape)
        if not self.sources:
            return np.empty(shape, dtype=np.float32)

        # Sparse coordinate grids make every Expr evaluate over whole axes without
        # materializing one dense grid per dimension. NumPy advanced indexing then
        # broadcasts the derived input coordinates to the output shape. This is the
        # same clipping/select semantics as the old np.ndindex reference, expressed
        # once per source instead of once per output element.
        grids = np.ogrid[tuple(slice(0, extent) for extent in shape)]
        env = {f"out_coord_{i}": grid for i, grid in enumerate(grids)}
        output = np.empty(shape, dtype=inputs[0].dtype)
        remaining = np.ones(shape, dtype=bool)
        for source in self.sources:
            input_tensor = inputs[source.input_idx]
            coords = tuple(
                np.clip(np.asarray(expr.eval(env), dtype=np.intp), 0, input_tensor.shape[i] - 1) for i, expr in enumerate(source.coord_map)
            )
            values = input_tensor[coords]
            select = remaining if source.select is None else remaining & np.broadcast_to(source.select.eval(env), shape)
            np.copyto(output, values, where=select)
            remaining &= ~select
        return output

    def is_identity(self, input_shape: tuple) -> bool:
        """True when this IndexMap is a pure pointer alias of its single input."""
        from emmy.compiler.ir.expr import PLACEHOLDER_PREFIX, Var

        if len(self.sources) != 1:
            return False
        src = self.sources[0]
        if src.select is not None:
            return False
        if tuple(self.out_shape) != tuple(input_shape):
            return False
        for i, c in enumerate(src.coord_map):
            if not isinstance(c, Var) or c.name != f"{PLACEHOLDER_PREFIX}{i}":
                return False
        return True
