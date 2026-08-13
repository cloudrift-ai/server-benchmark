"""Tests for the streaming DeepSeek V4 serving trace summarizer."""

import gzip
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "summarize_deepseek_v4_profile", PROJECT_ROOT / "scripts" / "summarize_deepseek_v4_profile.py"
)
assert _spec is not None and _spec.loader is not None
summarizer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(summarizer)


def _event(category: str, name: str, timestamp_us: float, duration_us: float) -> str:
    return f'{{"ph": "X", "cat": "{category}", "name": "{name}", "pid": 1,\n"ts": {timestamp_us}, "dur": {duration_us}}}\n'


def test_summarize_trace_attributes_kernels_and_preserves_segments(tmp_path):
    trace = tmp_path / "dp0_pp0_tp0_worker_rank0.pt.trace.json.gz"
    payload = "".join(
        (
            _event("kernel", "ncclKernel_AllReduce", 0, 10),
            _event("kernel", "_sm70_sparse_paged_fp8_kernel", 10, 20),
            _event("kernel", "quantize_and_insert_k_kernel", 30, 30),
            _event("kernel", "mxfp4_moe_grouped_gemm", 60, 40),
            _event("user_annotation", "execute_context_2(512)_generation_0(0)", 100, 70),
            _event("user_annotation", "execute_context_0(0)_generation_2(2)", 180, 20),
        )
    )
    with gzip.open(trace, "wt") as output:
        output.write(payload)

    result = summarizer.summarize_trace(trace)

    assert result["kernel_ms"] == {
        "attention": 0.02,
        "kv": 0.03,
        "moe": 0.04,
        "tp_collectives": 0.01,
    }
    assert result["microbatches"] == [
        {
            "concurrency": 2,
            "context_tokens": 512,
            "iteration_count": 2,
            "active_ms": 0.09,
            "span_ms": 0.1,
            "pp_utilization": 0.9,
            "pp_bubble_ms": 0.01,
            "pp_bubble_fraction": 0.1,
        }
    ]


def test_kernel_category_checks_kv_before_attention():
    assert summarizer.kernel_category("_fused_kv_compress_norm_rope_insert_sparse_attn") == "kv"
