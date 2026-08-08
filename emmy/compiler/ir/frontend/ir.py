"""Frontend (Torch) IR — ops captured directly from PyTorch tracing.

These ops exist in the graph between tracing and decomposition. Every one
of them has a decomposition rule in ``compiler/pipeline/passes/frontend/decomposition/`` that
rewrites it into ``ir.tensor.ir`` primitives (elementwise + reduce + indexmap
+ constants). After the decomposition pass completes, none of these ops
should remain in the graph.

Three groups:

1. **Layout-only ops** — ``TransposeOp``, ``ReshapeOp``, ``SliceOp``,
   ``CatOp``, ``UnsqueezeOp``. Rewritten to a single ``IndexMapOp`` each.
2. **Compound math ops** — ``LinearOp``, ``MatmulOp``, ``SdpaOp``,
   ``MeanOp``. Rewritten to elementwise/reduce chains (sometimes with
   inserted ``IndexMapOp`` unsqueezes so the broadcast contraction works).
3. **Storage-decode ops** — ``TrellisDecodeOp``, ``HadamardOp``. ``TrellisDecodeOp``'s full
   (``hadamard=True``) form is not decomposed: it only ever appears inside a constant-only cone
   that ``032_fold_constant_subgraphs`` collapses into a bind-time ``source_graph`` record,
   where its numpy ``forward`` runs through the reference NumPy backend. The HAT-BASIS form
   (``hadamard=False``) has a kernel realization: it lifts to a ``LoopOp`` of per-element
   ``TrellisLoad`` reads (``loop/lifting/050_lift_trellis_decode``) so loop fusion can inline
   it into its consuming matmul as a computed-B cone. ``HadamardOp`` is a zero-input generator
   that only ever lives inside such a bind record.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from emmy.compiler.dim import Dim
from emmy.compiler.ir.base import Op, _keepdim_axis

# ---------------------------------------------------------------------------
# Layout-only ops (decomposed to IndexMapOp)
# ---------------------------------------------------------------------------


@dataclass
class TransposeOp(Op):
    """Permute dimensions.

    ``axes`` either lists a full permutation (``len(axes) == ndim``) or
    names two axes to swap (``len(axes) == 2``), matching torch's
    ``permute``/``transpose`` overloads.
    """

    axes: tuple[int, ...]

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        in_shape = input_shapes[0]
        ndim = len(in_shape)
        if len(self.axes) == 2:
            # Tracer convention: 2-tuple is always a swap (aten.transpose).
            a, b = self.axes[0] % ndim, self.axes[1] % ndim
            out = list(in_shape)
            out[a], out[b] = out[b], out[a]
            return tuple(out)
        return tuple(in_shape[a] for a in self.axes)

    def forward(self, *inputs):

        a = inputs[0]
        ndim = a.ndim
        if len(self.axes) == 2:
            ax0, ax1 = self.axes[0] % ndim, self.axes[1] % ndim
            return np.swapaxes(a, ax0, ax1)
        return np.transpose(a, self.axes)


@dataclass
class ReshapeOp(Op):
    """Reshape tensor without changing data."""

    shape: tuple[int | str, ...]

    @staticmethod
    def _numel(shape) -> Dim:
        """Product of a shape's extents as a single ``Dim``. Accumulator starts
        at ``Dim(1)`` so bare-int shape elements (from tests / loader scratch
        shapes) are promoted to ``Dim`` on the first multiply; pure-Dim inputs
        fold exactly via ``Expr.simplify``."""
        out = Dim(1)
        for d in shape:
            out = out * d
        return out

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        if -1 not in self.shape:
            return tuple(self.shape)
        known = tuple(d for d in self.shape if d != -1)
        # Inferred axis takes whatever the remaining product needs. ``Dim``
        # arithmetic eager-folds the all-static case to a Literal int; mixed
        # static/symbolic produces a ``BinaryExpr`` Dim that resolves at launch.
        # ``_numel`` bootstraps the accumulator with ``Dim(1)``, so a raw-int
        # input shape still yields a ``Dim`` result.
        numel = self._numel
        inferred = numel(input_shapes[0]) // numel(known) if known else numel(input_shapes[0])
        return tuple(inferred if d == -1 else d for d in self.shape)

    def forward(self, *inputs):

        return np.reshape(inputs[0], self.shape)


@dataclass
class SliceOp(Op):
    """Extract a sub-tensor along a dimension.

    ``dim`` / ``start`` are recorded by the tracer from the raw FX args —
    the constant-input convention below can't represent them when the FX
    ``start`` is ``None`` or the ``end`` is a SymInt (``_resolve_inputs``
    drops both, leaving the surviving constants positionally ambiguous).
    The end never needs recording: ``shape`` already carries the output
    extent (symbolic or static).

    Legacy inputs convention (pre-field IR dumps): [tensor, dim_const,
    start_const, end_const] where the constants are scalar ConstantOps
    from the tracer; consumers fall back to it when ``dim is None``.
    """

    shape: tuple[int | str, ...]
    dim: int | None = None
    start: int | None = None

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        return tuple(self.shape)

    def forward(self, *inputs):
        tensor = inputs[0]
        if self.dim is not None:
            dim, start = self.dim, self.start or 0
            if start < 0:
                start += tensor.shape[dim]  # Python-negative start counts back from the end
            extent = self.shape[dim]
            end = start + int(extent) if isinstance(extent, int) else tensor.shape[dim]
        else:
            dim = int(inputs[1].flat[0]) if len(inputs) > 1 else 0
            start = int(inputs[2].flat[0]) if len(inputs) > 2 else 0
            end = int(inputs[3].flat[0]) if len(inputs) > 3 else tensor.shape[dim]
        slices = [slice(None)] * tensor.ndim
        slices[dim] = slice(start, end)
        return tensor[tuple(slices)]


@dataclass
class CatOp(Op):
    """Concatenate tensors along a dimension.

    Inputs: [dim_const, tensor_1, tensor_2, ...] where dim_const
    is a scalar ConstantOp indicating the concat axis.
    """

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        # Tensor inputs are all but the trailing scalar dim-constant.
        # Find them by skipping shape-(1,) inputs at the tail.
        def _is_scalar_constant(s):
            return len(s) == 1 and s[0] == 1

        tensor_shapes = [s for s in input_shapes if len(s) > 1 or (len(s) == 1 and not _is_scalar_constant(s))]
        if not tensor_shapes:
            return tuple(input_shapes[0])
        # Cat along the last dim by default (matches CatOp tracer convention).
        # Bootstrap ``total`` with ``Dim(0)`` so the cat-axis result is always
        # a ``Dim`` (eager-fold collapses the static-input case to a Literal
        # int-backed Dim). Other axes flow through as whatever the caller
        # passed in; ``Tensor.__post_init__`` coerces on assignment.
        last = len(tensor_shapes[0]) - 1
        out = list(tensor_shapes[0])
        total: Dim = Dim(0)
        for s in tensor_shapes:
            total = total + s[last]
        out[last] = total
        return tuple(out)

    def forward(self, *inputs):

        arrays = []
        dim = -1
        for inp in inputs:
            if inp.ndim == 0 or (inp.ndim == 1 and inp.size == 1):
                dim = int(inp.flat[0])
            else:
                arrays.append(inp)
        return np.concatenate(arrays, axis=dim)


@dataclass
class UnsqueezeOp(Op):
    """PyTorch aten.unsqueeze: add a size-1 dimension."""

    dim: int = 0

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        in_shape = list(input_shapes[0])
        d = self.dim if self.dim >= 0 else len(in_shape) + 1 + self.dim
        in_shape.insert(d, 1)
        return tuple(in_shape)

    def forward(self, *inputs):

        return np.expand_dims(inputs[0], axis=self.dim)


# ---------------------------------------------------------------------------
# Compound math ops (decomposed to elementwise + reduce chains)
# ---------------------------------------------------------------------------


@dataclass
class LinearOp(Op):
    """PyTorch aten.linear: output = x @ weight.T [+ bias]."""

    has_bias: bool = False

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        x_shape = input_shapes[0]
        w_shape = input_shapes[1]  # (out_features, in_features)
        return tuple(x_shape[:-1]) + (w_shape[-2],)

    def forward(self, *inputs):
        x, w = inputs[0], inputs[1]
        result = x @ w.T
        if self.has_bias:
            result = result + inputs[2]
        return result


@dataclass
class MatmulOp(Op):
    """PyTorch aten.mm/matmul/addmm: output = A @ B [+ bias]."""

    has_bias: bool = False

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        a_shape = input_shapes[0]
        b_shape = input_shapes[1]
        # Standard matmul: A(..., M, K) @ B(..., K, N) → (..., M, N)
        return tuple(a_shape[:-1]) + (b_shape[-1],)

    def forward(self, *inputs):
        a, b = inputs[0], inputs[1]
        result = a @ b
        if self.has_bias:
            result = result + inputs[2]
        return result


@dataclass
class SdpaOp(Op):
    """PyTorch scaled_dot_product_attention(Q, K, V, ...).

    ``sliding_window`` is a trace-time stamp (the HF wrapper knows the config's
    per-layer attention type): the attention is banded — a query at row ``m``
    attends keys ``kv ∈ [m − W + 1, m]``. Semantically it asserts the mask
    (explicit operand or ``is_causal``) keeps at most that band, so the lowering
    may skip key blocks wholly outside it; an explicit mask operand still
    applies (it may mask more, e.g. padding), a stamped mask-less SDPA computes
    the band itself.

    ``scale`` is torch's ``scale=`` kwarg: the score multiplier. ``None`` means
    the torch default ``1/sqrt(head_dim)``. Gemma-nano (E2B/E4B) passes an
    explicit ``scale=1.0`` (its q_norm absorbs the scaling) — dropping the kwarg
    silently re-scales the logits by ``1/sqrt(d)`` and redistributes the whole
    softmax."""

    is_causal: bool = False
    sliding_window: int | None = None
    scale: float | None = None

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        # SDPA output mirrors Q's batch+heads+seq dims, with V's last (head_dim).
        q_shape = input_shapes[0]
        v_shape = input_shapes[2]
        return tuple(q_shape[:-1]) + (v_shape[-1],)

    def forward(self, *inputs):

        q, k, v = inputs[0], inputs[1], inputs[2]
        # Align ndims: pad K/V with leading 1s to match Q's rank.
        while k.ndim < q.ndim:
            k = np.expand_dims(k, 0)
        while v.ndim < q.ndim:
            v = np.expand_dims(v, 0)
        # GQA: if Q has more heads than K/V, expand K/V by repeating heads.
        if q.ndim >= 3 and k.shape[-3] != q.shape[-3]:
            group = q.shape[-3] // k.shape[-3]
            k = np.repeat(k, group, axis=-3)
            v = np.repeat(v, group, axis=-3)
        d_k = q.shape[-1]
        scale = self.scale if self.scale is not None else 1.0 / np.sqrt(d_k)
        scores = q @ np.swapaxes(k, -2, -1) * scale
        if self.is_causal:
            seq_len = scores.shape[-2]
            kv_len = scores.shape[-1]
            causal_mask = np.triu(np.ones((seq_len, kv_len), dtype=scores.dtype), k=1)
            scores = scores - causal_mask * 1e9
        if self.sliding_window is not None:
            seq_len = scores.shape[-2]
            kv_len = scores.shape[-1]
            band_mask = np.tril(np.ones((seq_len, kv_len), dtype=scores.dtype), k=-self.sliding_window)
            scores = scores - band_mask * 1e9
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        return attn @ v


@dataclass
class MeanOp(Op):
    """PyTorch aten.mean.dim: reduction that averages along an axis.

    Kept as its own op so the tracer does a faithful 1:1 capture; a
    decomposition rule rewrites it into sum + div.
    """

    axis: int | str = -1

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        return _keepdim_axis(input_shapes[0], self.axis)

    def forward(self, *inputs):
        return np.mean(inputs[0], axis=self.axis)


@dataclass
class RmsNormOp(Op):
    """PyTorch aten.rms_norm: ``x * rsqrt(mean(x*x) + eps) * weight``.

    Inputs are ``(x, weight [, eps_const])``; ``eps`` falls back to the
    default when the optional constant isn't present. Decomposed by
    ``passes/frontend/decomposition/080_rms_norm.py``.
    """

    eps: float = 1e-6

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        return tuple(input_shapes[0])

    def forward(self, *inputs):
        x, weight = inputs[0], inputs[1]
        rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * weight


@dataclass
class LayerNormOp(Op):
    """PyTorch aten.layer_norm: ``(x - mean(x)) * rsqrt(var(x) + eps) * weight + bias``.

    Inputs are ``(x [, weight [, bias]])`` — the affine params are optional
    (``elementwise_affine=False`` drops both, ``bias=False`` drops the bias);
    the tracer peels the trailing ``eps`` constant into the op's own field.
    Decomposed by ``passes/frontend/decomposition/085_layer_norm.py``.
    """

    eps: float = 1e-5

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        return tuple(input_shapes[0])

    def forward(self, *inputs):
        x = inputs[0]
        mu = np.mean(x, axis=-1, keepdims=True)
        var = np.mean((x - mu) * (x - mu), axis=-1, keepdims=True)
        out = (x - mu) / np.sqrt(var + self.eps)
        if len(inputs) >= 2:
            out = out * inputs[1]
        if len(inputs) >= 3:
            out = out + inputs[2]
        return out


@dataclass
class SoftmaxOp(Op):
    """PyTorch aten.softmax.int: ``exp(x - max(x, dim)) / sum(exp(...), dim)``.

    Decomposed by ``passes/frontend/decomposition/100_softmax.py``.
    """

    axis: int | str = -1

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        return tuple(input_shapes[0])

    def forward(self, *inputs):
        x = inputs[0]
        m = np.max(x, axis=self.axis, keepdims=True)
        e = np.exp(x - m)
        return e / np.sum(e, axis=self.axis, keepdims=True)


# ---------------------------------------------------------------------------
# Storage-decode ops (bind-time only — folded by 032_fold_constant_subgraphs)
# ---------------------------------------------------------------------------


@dataclass
class TrellisDecodeOp(Op):
    """Decode one EXL3 trellis-coded linear's weight, in the traced ``(out, in)`` orientation.

    Inputs, in order: the packed codes (int16 ``(in/16, out/16, 16*K)``, one tail-biting
    trellis walk per 16x16 tile), ``suh`` (f16 ``(in,)``) and ``svh`` (f16 ``(out,)``) — the
    per-channel multipliers whose 128-block Hadamard fold restores the original basis. ``cb``
    selects the computed codebook (resolved at spell time from the ``mcg``/``mul1`` marker
    siblings' PRESENCE in the checkpoint index). The decode math lives in
    ``emmy/compiler/loader/exl3.py`` (imported lazily at evaluation time — this op is a
    bind-time construct, not a lowered kernel).

    ``out_features`` / ``in_features`` are the LOGICAL weight dims: EXL3 pads both dims to
    multiples of 128 at encode time, and slicing the decoded padded weight back down is
    exactly the reference math (exllamav3 zero-pads the input activations and slices the
    output, which reads only that submatrix). ``None`` keeps the full padded extent.

    Two forms, discriminated by ``hadamard``:

    - ``hadamard=True`` (the checkpoint-basis form, spelled by
      ``loader.quant.spell_trellis_constants``): inputs ``(codes, suh, svh)``, output the
      original-basis weight. It only ever appears inside a constant-only cone spelled at
      birth; ``032_fold_constant_subgraphs`` collapses that cone into a bind-time
      ``ConstantOp(source_graph=record)``, and ``forward`` is the record's evaluation through
      the reference NumPy backend. No decomposition or lowering rule exists for it — it must
      never survive the fold into the pipeline.
    - ``hadamard=False`` (the HAT-BASIS form — the raw per-tile decode, no channel vectors and
      no Hadamard fold): input ``(codes,)`` alone, output ``W_hat.T`` sliced to the logical
      dims. This is the form with an in-kernel realization — it lifts to a ``LoopOp`` of
      per-element :class:`~emmy.compiler.ir.stmt.TrellisLoad` reads, which loop fusion inlines
      into the consuming matmul as a computed-B cone (the warp tier's compute fill decodes the
      B tile in-kernel and the codes stay compressed in device memory). The activation-side
      basis restore (``x·suh`` / the 128-block Hadamard / ``·svh``) is spelled as separate
      graph algebra by its caller.
    """

    cb: int = 0
    out_features: int | None = None
    in_features: int | None = None
    hadamard: bool = True

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        # (n, k) — the HF ``nn.Linear`` weight orientation the trace promised;
        # sized by the suh/svh channel vectors (inputs 1 and 2) minus the encode padding.
        if not self.hadamard:
            assert self.out_features is not None and self.in_features is not None, "the hat-basis form sizes off its own fields"
            return (self.out_features, self.in_features)
        return (self.out_features or input_shapes[2][0], self.in_features or input_shapes[1][0])

    def forward(self, *inputs):
        from emmy.compiler.loader.exl3 import decode_trellis, fold_hadamard

        if not self.hadamard:
            (trellis,) = inputs
            w_hat = decode_trellis(np.asarray(trellis), self.cb).T
            return w_hat[: self.out_features, : self.in_features]
        trellis, suh, svh = inputs
        # fp16 is the decode's canonical precision (the checkpoint's own storage dtype and
        # exllamav3's reconstruction surface); the interpreter casts to the graph dtype after.
        w = fold_hadamard(decode_trellis(np.asarray(trellis), self.cb), suh, svh).T
        return w[: self.out_features, : self.in_features]


@dataclass
class HadamardOp(Op):
    """The natural-order Sylvester Hadamard matrix of size ``size``: plain ±1, UNSCALED.

    A ZERO-input generator: the matrix is a compile-time constant with no checkpoint source, so
    it is spelled as a ``ConstantOp(source_graph=…)`` bind record holding this one node and the
    loader evaluates it through the reference NumPy backend (``evaluate_source_graph``). Like
    the checkpoint-basis ``TrellisDecodeOp`` it has no decomposition or lowering rule — it must
    never appear in the live graph.

    Unscaled because ±1 is exact in every float width: the transform's ``1/sqrt(size)`` rides a
    separate multiply in the chain that spells it (``loader.quant``), which also gets to split
    the factor into exact powers of two. The matrix is symmetric, so the same constant serves
    ``x @ H`` and ``H @ x`` — the ``LinearOp`` spelling (``x @ W.T``) needs no transpose.
    """

    size: int = 128

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        return (self.size, self.size)

    def forward(self, *inputs):
        from emmy.compiler.loader.exl3 import sylvester_hadamard  # noqa: PLC0415

        return sylvester_hadamard(self.size).astype(np.float32)
