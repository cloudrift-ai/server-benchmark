"""Fast CPU tests for ``gen_runner`` helpers (no GPU/model). The decode-bucket compile +
correctness are covered on GPU by ``test_gen_runner_gpu.py`` / ``test_vllm_plugin_gen_gpu.py``."""

import numpy as np
import pytest

from emmy.serving.gen_runner import EmmyGenRunner, _pad_rows, _program_config_sha, _static_decode_covers_capacity


class _Config:
    def __init__(self, data):
        self.data = data

    def to_dict(self):
        import copy

        return copy.deepcopy(self.data)


def test_program_config_identity_ignores_only_generation_eos_policy():
    base = {
        "model_type": "gemma4_unified_text",
        "text_config": {"hidden_size": 3840, "intermediate_size": 15360},
        "eos_token_id": 1,
    }
    instruction = {**base, "eos_token_id": [1, 106]}
    assert _program_config_sha(_Config(base)) == _program_config_sha(_Config(instruction))

    different_architecture = {**instruction, "model_type": "gemma4_other_text"}
    assert _program_config_sha(_Config(base)) != _program_config_sha(_Config(different_architecture))

    different_geometry = {
        **instruction,
        "text_config": {**instruction["text_config"], "hidden_size": 4096},
    }
    assert _program_config_sha(_Config(base)) != _program_config_sha(_Config(different_geometry))


def test_pad_rows_pads_with_zeros_and_preserves_real_rows():
    a = np.arange(6, dtype=np.float16).reshape(3, 2)
    out = _pad_rows(a, 5)
    assert out.shape == (5, 2)
    assert out.dtype == np.float16
    np.testing.assert_array_equal(out[:3], a)  # real rows intact
    assert (out[3:] == 0).all()  # padding is zeros (computed then sliced away)


def test_pad_rows_is_passthrough_when_already_at_bucket():
    a = np.ones((4, 8), dtype=np.float16)
    assert _pad_rows(a, 4) is a  # no copy when t == bucket


@pytest.mark.parametrize(
    ("max_tokens", "decode_bucket", "prefill_bucket", "expected"),
    [
        (None, 16, 0, False),
        (1, 1, 0, True),
        (1, 16, 0, True),
        (16, 16, 0, True),
        (17, 16, 0, False),
        (1, 0, 0, False),
        (1, 16, 32, False),
    ],
)
def test_static_decode_capacity_proof(max_tokens, decode_bucket, prefill_bucket, expected):
    assert _static_decode_covers_capacity(max_tokens, decode_bucket, prefill_bucket) is expected


def test_static_only_runner_counts_layers_without_symbolic_programs():
    runner = EmmyGenRunner(
        embed_weight=np.empty((1, 1), dtype=np.float16),
        norm=None,
        pre=[],
        post=[],
        attn_meta=[(1, 1, 1, 1.0), (1, 1, 1, 1.0)],
        np_dtype=np.dtype("float16"),
        pre_decode=[object(), object()],
        post_decode=[object(), object()],
        decode_bucket=1,
        prefill_capacity=1,
    )
    assert runner.num_layers == 2
    assert runner.global_layer_id(0) == 0
    assert runner.global_layer_id(1) == 1
    assert runner.prefill_capacity == 1
    assert runner.has_device_decode
    with pytest.raises(RuntimeError, match="token width 2 exceeds static-only capacity 1"):
        runner.forward_layer_pre(0, np.zeros((2, 1), dtype=np.float16))
    with pytest.raises(RuntimeError, match="token width 2 exceeds static-only capacity 1"):
        runner.forward_layer_post(
            0,
            np.zeros((2, 1), dtype=np.float16),
            np.zeros((2, 1), dtype=np.float16),
        )


def test_float32_residual_moe_rider_keeps_normalized_activation_in_float16():
    """The two-output fp32-residual post rider must not allocate ``xn`` in residual dtype."""
    torch = pytest.importorskip("torch")

    class PostProgram:
        output_names = ("hidden", "moe_xn")

        def run_device(self, inputs, *, out):
            _attn_out, residual = inputs
            out[0].copy_(residual)
            out[1].copy_(residual.to(out[1].dtype))

    runner = EmmyGenRunner.__new__(EmmyGenRunner)
    runner._post_m1 = None
    runner._post_decode = [PostProgram()]
    runner._post_prefill = [PostProgram()]
    runner._post = []
    runner._pre_decode = [object()]
    runner._pre_prefill = [object()]
    runner._decode_bucket = 2
    runner._prefill_bucket = 4
    runner._activation_dtype = torch.float16

    residual = torch.randn(6, 8, dtype=torch.float32)
    hidden, normalized = runner._route_post_device(0, torch.randn(6, 8, dtype=torch.float16), residual)

    assert hidden.dtype == torch.float32
    assert normalized.dtype == torch.float16
    assert torch.nn.Linear(8, 2, dtype=torch.float16)(normalized).dtype == torch.float16


def test_fork_attention_pre_rider_sizes_its_destination_at_hidden_width():
    """A chunk step carrying decode riders splits across two programs into one joint destination.
    The classic seam sizes those from ``(q, k, v)``; the fork-attention seam has no projections
    here — its ``pre`` returns one hidden-width activation, and sizing that from the attention
    metadata instead (``num_heads * head_dim``, 32768 on DeepSeek V4 against a hidden size of
    4096) makes every rider-width step fail its copy."""
    torch = pytest.importorskip("torch")

    class PreProgram:
        output_names = ("x",)

        def run_device(self, inputs, *, out):
            out[0].copy_(inputs[0])

    runner = EmmyGenRunner.__new__(EmmyGenRunner)
    runner._pre_m1 = None
    runner._pre_decode = [PreProgram()]
    runner._pre_prefill = [PreProgram()]
    runner._decode_bucket = 2
    runner._prefill_bucket = 4
    runner._hidden_size = 8
    runner._activation_dtype = torch.float16
    # The attention metadata a fork-attention layer still carries: 4 heads x head_dim 16 is
    # 64 wide, eight times the seam's real width — the shape the old sizing would have used.
    runner._attn_meta = [(16, 4, 1, 0.5)]

    (x,) = runner.forward_layer_pre_device(0, torch.randn(6, 8, dtype=torch.float16))

    assert x.shape == (6, 8)


def test_pipeline_runner_tracks_absolute_layers_and_boundary_ownership():
    runner = EmmyGenRunner(
        embed_weight=None,
        norm=None,
        hidden_size=8,
        layer_ids=[7, 8],
        pre=[],
        post=[],
        attn_meta=[(2, 4, 1, 0.5), (2, 4, 1, 0.5)],
        np_dtype=np.dtype("float16"),
    )

    assert runner.num_layers == 2
    assert runner.global_layer_id(0) == 7
    assert runner.global_layer_id(1) == 8
    with pytest.raises(RuntimeError, match="does not own the token embedding"):
        runner.embed([0])
    with pytest.raises(RuntimeError, match="does not own the final norm"):
        runner.final_norm(np.zeros((1, 8), dtype=np.float16))


@pytest.mark.parametrize(
    ("quant_method", "coded_trunk"),
    [("exl3", True), ("awq", True), ("fp8", False), ("modelopt", True)],
)
def test_create_keeps_storage_coded_trunks_packed(tmp_path, monkeypatch, quant_method, coded_trunk):
    """EXL3/AWQ/NVFP4 stay checkpoint-coded; FP8 preserves the decoded trunk lane.

    NVFP4 (``modelopt``) sat in the decoded column while two defects made a coded trunk compute
    silently wrong numbers — a packed operand that dropped its split-K slice base, and a plan-keyed
    constant read that decoded the e4m3 block scales a second time. Both are fixed, and serving
    parity against eager torch is what moved this row: the coded and decoded trunks agree
    bit-for-bit on a layer's q/k/v, and both match eager torch on the rest of the layer.
    """
    import json

    from emmy.compiler.loader import safetensors
    from emmy.compiler.trace import huggingface
    from emmy.serving.gen_runner import EmmyGenRunner

    quant_config = {"quant_method": quant_method}
    if quant_method == "modelopt":
        quant_config["quant_algo"] = "NVFP4"
    if quant_method == "awq":
        quant_config.update(bits=4, version="gemm", zero_point=True)
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": quant_config}))
    seen = {}
    fake_model = object()
    fake_store = {"fmt": quant_method}

    monkeypatch.setattr(safetensors, "warn_if_unpinned", lambda _model_id: None)
    monkeypatch.setattr(huggingface, "quantized_checkpoint_dir", lambda _model_id: tmp_path)

    def fake_load(path, dtype, *, compress_trunk=False, layer_range=None, include_embed=True, include_norm=True, expert_range=None):
        seen.update(
            path=path,
            dtype=dtype,
            compress_trunk=compress_trunk,
            layer_range=layer_range,
            include_embed=include_embed,
            include_norm=include_norm,
            expert_range=expert_range,
        )
        return fake_model, fake_store

    monkeypatch.setattr(huggingface, "load_quantized_split", fake_load)

    def fake_from_model(cls, model, **kwargs):
        seen.update(cls=cls, model=model, kwargs=kwargs)
        return "runner"

    monkeypatch.setattr(EmmyGenRunner, "from_model", classmethod(fake_from_model))

    assert EmmyGenRunner.create(str(tmp_path), dtype_str="float16") == "runner"
    assert seen["path"] == tmp_path
    assert seen["compress_trunk"] is coded_trunk
    assert seen["model"] is fake_model
    assert seen["kwargs"]["expert_store"] is fake_store


class _StampedGraph(Exception):
    """Carries the graph out of ``_compile_split`` at the backend seam — the last point where the
    loader's spellings are all in, and the first that would need a GPU."""

    def __init__(self, graph):
        super().__init__("captured the stamped split graph")
        self.graph = graph


def test_compile_split_spells_static_fp4_activations_for_a_marked_nvfp4_checkpoint(tmp_path, monkeypatch):
    """A checkpoint declaring static 4-bit input activations must reach serving's split programs as
    the declared W4A4 algebra: a ``to_f4e2m1`` encode ahead of the marked linear and a packed
    ``f4e2m1x2`` activation buffer beside the packed weight. ``emmy compile`` runs the activation
    speller after the weight speller; serving's stamp inside ``_compile_split`` must do the same,
    or the coded trunk computes 4-bit weights against 16-bit activations — the W4A16 scaffolding,
    which is a different program from the one the checkpoint declares.

    The graph is captured at the CUDA backend seam, so nothing here compiles or runs a kernel."""
    torch = pytest.importorskip("torch")

    from emmy.compiler.loader.synthesize import write_quantized_checkpoint
    from emmy.serving.gen_runner import _compile_split, trace_split

    class _Split(torch.nn.Module):
        """One marked linear — the shape a serving split's ``pre`` carries per projection."""

        def __init__(self, hidden, inner):
            super().__init__()
            self.q_proj = torch.nn.Linear(hidden, inner, bias=False)

        def forward(self, x):
            return self.q_proj(x)

    wrapper = _Split(64, 32)
    example = (torch.randn(8, 64),)
    ckpt = write_quantized_checkpoint(trace_split(wrapper, example, None), (wrapper, example, {}), tmp_path / "ckpt")
    # What the runner threads as ``ckpt``: the checkpoint dir plus parameter identity → its key.
    id_to_key = {id(wrapper.q_proj.weight): "l0.weight"}

    class _CaptureBackend:
        def __init__(self, **_kwargs):
            pass

        def compile(self, graph):
            raise _StampedGraph(graph)

    monkeypatch.setattr("emmy.compiler.backend.cuda.backend.CudaBackend", _CaptureBackend)
    with pytest.raises(_StampedGraph) as caught:
        _compile_split(wrapper, list(example), None, np.dtype("float32"), ckpt=(str(ckpt), id_to_key))
    graph = caught.value.graph

    packed_weights = [n for n in graph.nodes.values() if n.output.dtype.name == "f4e2m1x2" and type(n.op).__name__ == "ConstantOp"]
    assert packed_weights, "the weight speller never fired — the fixture is not a marked NVFP4 checkpoint"

    encodes = [n for n in graph.nodes.values() if type(n.op).__name__ == "ElementwiseOp" and n.op.name == "to_f4e2m1"]
    packed_activations = [n for n in graph.nodes.values() if n.output.dtype.name == "f4e2m1x2" and n not in packed_weights]
    assert encodes, "no to_f4e2m1 encode ahead of the marked linear: serving still runs 16-bit activations"
    assert packed_activations, "no packed f4e2m1x2 activation buffer: only the weight side is spelled"


def test_create_passes_the_expert_shard_through_to_the_loader(tmp_path, monkeypatch):
    """A tensor-parallel rank's expert shard must reach the checkpoint read, not just the routing:
    holding every expert is what does not fit the card in the first place."""
    import json

    from emmy.compiler.loader import safetensors
    from emmy.compiler.trace import huggingface
    from emmy.serving.gen_runner import EmmyGenRunner

    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "fp8"}}))
    seen = {}
    monkeypatch.setattr(safetensors, "warn_if_unpinned", lambda _model_id: None)
    monkeypatch.setattr(huggingface, "quantized_checkpoint_dir", lambda _model_id: tmp_path)

    def fake_load(path, dtype, **kwargs):
        seen.update(kwargs)
        return object(), {"fmt": "mxfp4"}

    monkeypatch.setattr(huggingface, "load_quantized_split", fake_load)
    monkeypatch.setattr(EmmyGenRunner, "from_model", classmethod(lambda cls, model, **kwargs: kwargs))

    built = EmmyGenRunner.create(model_id=str(tmp_path), expert_range=(64, 96))

    assert seen["expert_range"] == (64, 96), "the shard never reached the checkpoint read"
    assert built["expert_range"] == (64, 96), "the shard never reached the runner"
