"""Device-resident EXL3 output head over the compiler's native static-M1 operation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Exl3HeadSpec:
    bits: int
    codebook: int
    hidden_size: int
    stored_vocab_size: int
    vocab_size: int

    @classmethod
    def from_source(
        cls,
        source: dict,
        *,
        hidden_size: int,
        vocab_size: int,
        tensor_parallel_size: int,
        tied: bool,
        compute_capability: tuple[int, int] | None = None,
    ) -> Exl3HeadSpec | None:
        """Return the native head contract, or ``None`` for the decoded fallback."""
        if tied or tensor_parallel_size != 1:
            return None
        if "lm_head.mcg" in source and "lm_head.mul1" in source:
            raise ValueError("EXL3 coded lm_head carries both mcg and mul1 codebook markers")
        codebook = 2 if "lm_head.mul1" in source else 1 if "lm_head.mcg" in source else 0
        if codebook != 2:
            return None
        trellis = np.asarray(source["lm_head.trellis"])
        suh = np.asarray(source["lm_head.suh"])
        svh = np.asarray(source["lm_head.svh"])
        if trellis.dtype != np.int16 or trellis.ndim != 3 or trellis.shape[-1] % 16:
            raise ValueError(f"EXL3 coded lm_head has invalid trellis storage {trellis.dtype}{trellis.shape}")
        bits = trellis.shape[-1] // 16
        stored_hidden, stored_vocab = trellis.shape[0] * 16, trellis.shape[1] * 16
        if suh.dtype != np.float16 or suh.shape != (stored_hidden,):
            raise ValueError(f"EXL3 coded lm_head has invalid suh storage {suh.dtype}{suh.shape}")
        if svh.dtype != np.float16 or svh.shape != (stored_vocab,):
            raise ValueError(f"EXL3 coded lm_head has invalid svh storage {svh.dtype}{svh.shape}")
        if stored_hidden != hidden_size or stored_vocab < vocab_size:
            raise ValueError(f"EXL3 coded lm_head stores {(stored_vocab, stored_hidden)}, expected at least {(vocab_size, hidden_size)}")
        if stored_hidden % 128 or stored_vocab % 256:
            return None
        if compute_capability is None:
            from emmy.compiler.target import compute_capability as live_compute_capability  # noqa: PLC0415

            compute_capability = live_compute_capability()
        if compute_capability < (7, 0):
            return None
        if compute_capability < (8, 0) and bits not in (5, 6, 7):
            return None
        return cls(bits, codebook, stored_hidden, stored_vocab, vocab_size)


class Exl3CodedHead:
    """One persistent coded head; checkpoint leaves upload once and remain pointer-stable."""

    def __init__(self, spec: Exl3HeadSpec, source: dict):
        from emmy.compiler.backend.cuda.backend import CudaBackend
        from emmy.compiler.backend.cuda.program import CompiledProgram
        from emmy.compiler.backend.gpu_lock import gpu_lock
        from emmy.compiler.backend.plan import plan_from_graph
        from emmy.compiler.graph import Graph, Tensor
        from emmy.compiler.ir.base import ConstantOp, InputOp
        from emmy.compiler.ir.frontend.ir import Exl3GemvOp

        graph = Graph()
        graph.add_node(InputOp(), [], Tensor("x", (1, spec.hidden_size), "f16"), node_id="x")
        arrays = {}
        for leaf, dtype in (("trellis", "i16"), ("suh", "f16"), ("svh", "f16")):
            path = f"lm_head.{leaf}"
            value = np.ascontiguousarray(source[path])
            arrays[path] = value
            graph.add_node(
                ConstantOp(name=path, source_path=path, source_shape=value.shape, source_dtype=dtype),
                [],
                Tensor(path, value.shape, dtype),
                node_id=path,
            )
        graph.add_node(
            Exl3GemvOp(
                bits=spec.bits,
                codebook=spec.codebook,
                weight_shape=(spec.stored_vocab_size, spec.hidden_size),
                residual=True,
            ),
            ["x", "lm_head.trellis", "lm_head.suh", "lm_head.svh"],
            Tensor("logits", (1, spec.stored_vocab_size), "f32"),
            node_id="logits",
        )
        graph.inputs, graph.outputs = ["x"], ["logits"]
        plan = plan_from_graph(CudaBackend(tune_db="auto").compile(graph))
        if len(plan.launches) != 1 or not plan.launches[0].scalar_args:
            raise RuntimeError("EXL3 coded lm_head did not lower to one native static-M1 launch")
        feed = {"x": np.zeros((1, spec.hidden_size), dtype=np.float16), **arrays}
        with gpu_lock():
            self.program = CompiledProgram.build_from_plan(plan, feed)
        self.spec = spec

    def __call__(self, hidden_states):
        import cupy as cp
        import torch

        from emmy.compiler.backend.gpu_lock import gpu_lock

        rows = hidden_states.reshape(-1, self.spec.hidden_size)
        if rows.dtype != torch.float16 or not rows.is_cuda:
            raise ValueError(f"EXL3 coded lm_head requires CUDA fp16 input, got {rows.device}/{rows.dtype}")
        if torch.cuda.is_current_stream_capturing() and rows.shape[0] != 1:
            raise ValueError("captured EXL3 coded lm_head requires exactly one sampled row")
        output = []
        with gpu_lock(), cp.cuda.Stream.from_external(torch.cuda.current_stream()):
            for row in rows:
                self.program.upload_prefix_device({"x": cp.from_dlpack(row[None].contiguous())})
                self.program.run_once()
                logits = torch.from_dlpack(self.program.output_prefix_device()["logits"])[:, : self.spec.vocab_size]
                output.append(logits if torch.cuda.is_current_stream_capturing() else logits.clone())
        return output[0] if len(output) == 1 else torch.cat(output, dim=0)
