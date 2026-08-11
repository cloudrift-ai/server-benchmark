"""Hermetic equivalence tests for the MoE third-seam carve (no GPU/vLLM).

Proves the carve is correct on a tiny random OLMoE layer: ``post_attn`` (o_proj + residual +
post-attention norm, both ``h`` and ``xn`` out) plus the runner-style torch half (HF router
module → per-expert ``expert(x, w_gate_up, w_down)`` launches → weighted ``index_add_``)
reproduces the eager block tail exactly. Pure eager, CPU, fp32 — no compile.
"""

import pytest

from tests.support.checkpoints import exl3_linear_tensors


def _tiny_olmoe_config(transformers):
    return transformers.OlmoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=8,
        num_experts_per_tok=2,
        norm_topk_prob=False,
        max_position_embeddings=64,
    )


def _tiny_laguna_config(transformers):
    return transformers.LagunaConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_attention_heads_per_layer=[4, 4],
        num_key_value_heads=2,
        head_dim=16,
        num_experts=4,
        num_experts_per_tok=2,
        layer_types=["full_attention", "full_attention"],
        mlp_layer_types=["dense", "sparse"],
        moe_routed_scaling_factor=2.5,
        max_position_embeddings=64,
        gating=True,
    )


def _combine(gate, experts, expert, xn):
    """The runner's torch half with the eager expert wrapper in place of the compiled program —
    the routing math itself is the SHARED ``combine_routed_experts`` serving runs."""
    from emmy.serving.gen_runner import combine_routed_experts

    return combine_routed_experts(xn, gate(xn), lambda e, rows: expert(rows, experts.gate_up_proj[e], experts.down_proj[e]))


def test_moe_split_matches_eager_block_tail():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers.models.olmoe.modeling_olmoe import OlmoeDecoderLayer

    from emmy.compiler.trace.huggingface import build_moe_split_wrapper, moe_block_parts

    torch.manual_seed(0)
    cfg = _tiny_olmoe_config(transformers)
    block = OlmoeDecoderLayer(cfg, layer_idx=0).eval()
    for p in block.parameters():
        torch.nn.init.normal_(p, std=0.2)
    block.mlp.routed_scaling_factor = 2.5

    _, post_attn, expert = build_moe_split_wrapper(block)
    gate, experts = moe_block_parts(block.mlp)

    t = 5
    attn_width = cfg.num_attention_heads * (cfg.hidden_size // cfg.num_attention_heads)
    attn_out = torch.randn(t, attn_width)
    residual = torch.randn(t, cfg.hidden_size)
    with torch.no_grad():
        h, xn = post_attn(attn_out, residual)
        got = h + _combine(gate, experts, expert, xn)
        h_ref = residual + block.self_attn.o_proj(attn_out)
        ref = h_ref + block.mlp(block.post_attention_layernorm(h_ref).unsqueeze(0)).squeeze(0) * 2.5
    assert torch.allclose(ref, got, atol=1e-5)


def test_moe_expert_shares_one_shape_across_experts():
    """Every expert must be servable by ONE wrapper: the expert forward takes the weights as
    forward ARGUMENTS (they trace as inputs), and dim-0 slices of the 3-D expert tensors all
    have the same shape."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers.models.olmoe.modeling_olmoe import OlmoeDecoderLayer

    from emmy.compiler.trace.huggingface import build_moe_split_wrapper, moe_block_parts

    torch.manual_seed(0)
    cfg = _tiny_olmoe_config(transformers)
    block = OlmoeDecoderLayer(cfg, layer_idx=0).eval()
    for p in block.parameters():  # bare-layer params are torch.empty — init or the assert below is nondeterministic
        torch.nn.init.normal_(p, std=0.2)
    _, _, expert = build_moe_split_wrapper(block)
    _, experts = moe_block_parts(block.mlp)

    assert len(list(expert.parameters())) == 0, "expert wrapper must carry NO parameters (weights are inputs)"
    x = torch.randn(3, cfg.hidden_size)
    with torch.no_grad():
        y0 = expert(x, experts.gate_up_proj[0], experts.down_proj[0])
        y1 = expert(x, experts.gate_up_proj[1], experts.down_proj[1])
    assert y0.shape == y1.shape == x.shape
    assert not torch.allclose(y0, y1), "different experts' weights must produce different outputs"


def test_moe_split_folds_shared_experts_into_post_attn():
    """DeepSeek/GLM lineage blocks carry an always-on ``shared_experts`` MLP beside the routed
    ones; the carve folds it into ``post_attn``'s returned ``h`` (over the same normed ``xn``),
    so the runner's route-and-combine half needs no seam change."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers.models.olmoe.modeling_olmoe import OlmoeDecoderLayer

    from emmy.compiler.trace.huggingface import build_moe_split_wrapper

    torch.manual_seed(0)
    block = OlmoeDecoderLayer(_tiny_olmoe_config(transformers), layer_idx=0).eval()
    block.mlp.shared_experts = torch.nn.Linear(64, 64)
    _, post_attn, _ = build_moe_split_wrapper(block)
    attn_out = torch.randn(5, 64)
    residual = torch.randn(5, 64)
    with torch.no_grad():
        h, xn = post_attn(attn_out, residual)
        h_ref = residual + block.self_attn.o_proj(attn_out)
        xn_ref = block.post_attention_layernorm(h_ref)
    assert torch.allclose(xn, xn_ref, atol=1e-6)
    assert torch.allclose(h, h_ref + block.mlp.shared_experts(xn_ref), atol=1e-6)


def test_moe_split_rejects_gated_shared_expert():
    """Qwen-MoE's shared expert is GATED (``shared_expert_gate``); the ungated fold would
    silently drop the gate, so it must reject LOUDLY."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers.models.olmoe.modeling_olmoe import OlmoeDecoderLayer

    from emmy.compiler.trace.huggingface import build_moe_split_wrapper

    torch.manual_seed(0)
    block = OlmoeDecoderLayer(_tiny_olmoe_config(transformers), layer_idx=0).eval()
    block.mlp.shared_expert = torch.nn.Linear(64, 64)
    block.mlp.shared_expert_gate = torch.nn.Linear(64, 1)
    with pytest.raises(NotImplementedError, match="shared_expert_gate"):
        build_moe_split_wrapper(block)


def _tiny_glm4_moe_config(transformers):
    return transformers.Glm4MoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        first_k_dense_replace=1,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        use_qk_norm=False,
        partial_rotary_factor=0.5,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
    )


def test_glm4_moe_split_matches_eager_block_tail():
    """GLM-4.5-Air's block layout (Glm4Moe: 3-tuple router, packed v5 experts, always-on
    shared expert) through the carve: ``post_attn`` (shared experts folded in) + the runner's
    route-and-combine reproduces the eager MoE block tail."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeDecoderLayer

    from emmy.compiler.trace.huggingface import build_moe_split_wrapper, moe_block_parts

    torch.manual_seed(0)
    cfg = _tiny_glm4_moe_config(transformers)
    block = Glm4MoeDecoderLayer(cfg, layer_idx=1).eval()  # layer 1 = the MoE form
    for p in block.parameters():
        torch.nn.init.normal_(p, std=0.2)

    _, post_attn, expert = build_moe_split_wrapper(block)
    gate, experts = moe_block_parts(block.mlp)
    assert gate is block.mlp.gate  # the 3-tuple Glm4MoeTopkRouter; combine reads its LAST two entries

    t = 5
    attn_width = cfg.num_attention_heads * cfg.head_dim
    attn_out = torch.randn(t, attn_width)
    residual = torch.randn(t, cfg.hidden_size)
    with torch.no_grad():
        h, xn = post_attn(attn_out, residual)
        got = h + _combine(gate, experts, expert, xn)
        h_ref = residual + block.self_attn.o_proj(attn_out)
        ref = h_ref + block.mlp(block.post_attention_layernorm(h_ref).unsqueeze(0)).squeeze(0)
    assert torch.allclose(ref, got, atol=1e-5)


def test_laguna_moe_split_preserves_attention_gate_and_routed_scale():
    """Laguna combines two model-specific factors at this seam: a softplus per-head attention
    gate before ``o_proj`` and a 2.5 multiplier on the routed (but not shared) expert term."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers.models.laguna.modeling_laguna import LagunaDecoderLayer

    from emmy.compiler.trace.huggingface import build_moe_split_wrapper, moe_block_parts

    cfg = _tiny_laguna_config(transformers)
    torch.manual_seed(0)
    block = LagunaDecoderLayer(cfg, layer_idx=1).eval()
    for parameter in block.parameters():
        torch.nn.init.normal_(parameter, std=0.2)
    _, post_attn, expert = build_moe_split_wrapper(block)
    gate, experts = moe_block_parts(block.mlp)

    residual = torch.randn(5, cfg.hidden_size)
    attn_out = torch.randn(5, cfg.num_attention_heads * cfg.head_dim)
    with torch.no_grad():
        h, xn = post_attn(attn_out, residual)
        got = h + _combine(gate, experts, expert, xn)

        normalized = block.input_layernorm(residual)
        attn_gate = torch.nn.functional.softplus(block.self_attn.g_proj(normalized).float()).to(attn_out.dtype)
        gated = (attn_out.view(5, cfg.num_attention_heads, cfg.head_dim) * attn_gate.unsqueeze(-1)).view(5, -1)
        h_ref = residual + block.self_attn.o_proj(gated)
        ref = h_ref + block.mlp(block.post_attention_layernorm(h_ref).unsqueeze(0)).squeeze(0)

    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)


def test_exl3_laguna_routed_scale_matches_reference_architecture():
    """Laguna EXL3 stores routed up projections with the reference runtime's 1/128 scale."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM

    from emmy.compiler.trace.huggingface import _apply_exl3_laguna_routed_scale, build_moe_split_wrapper, moe_block_parts
    from emmy.serving.gen_runner import combine_routed_experts

    cfg = _tiny_laguna_config(transformers)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg).eval()

    dense = model.model.layers[0].mlp
    sparse = model.model.layers[1].mlp
    hidden = torch.randn(5, cfg.hidden_size)
    routed = sparse.gate(hidden)
    original_gate = sparse.gate
    _pre, _post, original_expert = build_moe_split_wrapper(model.model.layers[1], split_gate_up=True)
    _gate, experts = moe_block_parts(sparse)
    reference = combine_routed_experts(
        hidden,
        routed,
        lambda e, rows: original_expert(rows, *experts.gate_up_proj[e].chunk(2, dim=0), experts.down_proj[e]),
    )
    assert moe_block_parts(dense) is None
    assert sparse.routed_scaling_factor == 2.5

    _apply_exl3_laguna_routed_scale(model, exl3=False)
    assert sparse.gate is original_gate
    _apply_exl3_laguna_routed_scale(model, exl3=True)

    assert moe_block_parts(dense) is None
    assert sparse.routed_scaling_factor == 2.5
    scaled_gate = sparse.gate
    assert scaled_gate._emmy_routed_accumulate_float32 is True
    assert scaled_gate._emmy_routed_base_scale_folded is True
    assert scaled_gate._emmy_exl3_laguna_scale == 320.0
    scaled = scaled_gate(hidden)
    torch.testing.assert_close(scaled[-2], routed[-2] * 320.0, rtol=0, atol=0)
    torch.testing.assert_close(scaled[-1], routed[-1], rtol=0, atol=0)
    assert scaled[-2].dtype == routed[-2].dtype

    _pre, _post, expert = build_moe_split_wrapper(model.model.layers[1], split_gate_up=True)
    assert expert._emmy_output_float32 is True
    combined = combine_routed_experts(
        hidden,
        scaled_gate(hidden),
        lambda e, rows: expert(rows, *experts.gate_up_proj[e].chunk(2, dim=0), experts.down_proj[e]),
    )
    assert torch.isfinite(combined).all()
    torch.testing.assert_close(combined, reference * 128.0, rtol=1e-5, atol=1e-5)

    _apply_exl3_laguna_routed_scale(model, exl3=True)

    assert sparse.gate is scaled_gate
    assert sparse.gate.router is original_gate


def test_exl3_laguna_shared_expert_uses_marked_float32_cone():
    """The late-layer shared activation can overflow in fp16; the marked graph keeps only
    that checkpoint-provenanced cone in fp32 and returns the residual in model dtype."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM

    from emmy.compiler.ir.frontend.ir import LinearOp
    from emmy.compiler.trace.huggingface import (
        _apply_exl3_laguna_routed_scale,
        build_moe_split_wrapper,
        promote_shared_expert_float32,
    )
    from emmy.serving.gen_runner import trace_split

    cfg = _tiny_laguna_config(transformers)
    model = AutoModelForCausalLM.from_config(cfg).half().eval()
    _apply_exl3_laguna_routed_scale(model, exl3=True)
    _pre, post, _expert = build_moe_split_wrapper(
        model.model.layers[1],
        split_gate_up=True,
        float32_residual=True,
    )
    assert post._emmy_shared_expert_float32 is True
    with torch.no_grad():
        post_outputs = post(
            torch.zeros(2, cfg.hidden_size, dtype=torch.float16),
            torch.zeros(2, cfg.hidden_size, dtype=torch.float32),
        )
    assert len(post_outputs) == 3

    args = [
        torch.zeros(2, cfg.hidden_size, dtype=torch.float16),
        torch.zeros(2, cfg.hidden_size, dtype=torch.float32),
    ]
    graph = trace_split(post, args, None)
    shared = {}
    for node in graph.nodes.values():
        source = getattr(node.op, "source_path", "") or ""
        if "shared_experts" not in source:
            continue
        kind = next(name for name in ("gate", "up", "down") if source.endswith(f".{name}_proj.weight"))
        users = [graph.nodes[uid] for uid in graph.consumers(node.id) if isinstance(graph.nodes[uid].op, LinearOp)]
        assert len(users) == 1
        shared[kind] = users[0]
    assert set(shared) == {"gate", "up", "down"}
    assert {node.output.dtype.name for node in shared.values()} == {"f16"}

    promote_shared_expert_float32(graph)

    assert {node.output.dtype.name for node in shared.values()} == {"f32"}
    cone = set()
    queue = [shared["gate"].id, shared["up"].id]
    while queue:
        nid = queue.pop()
        if nid in cone or nid == shared["down"].id:
            continue
        cone.add(nid)
        queue.extend(graph.consumers(nid))
    assert cone
    assert {graph.nodes[nid].output.dtype.name for nid in cone} == {"f32"}
    assert graph.nodes[graph.outputs[0]].output.dtype.name == "f32"
    assert graph.nodes[graph.outputs[1]].output.dtype.name == "f16"
    assert graph.nodes[graph.outputs[2]].output.dtype.name == "f32"
    non_shared = [node for node in graph.nodes.values() if isinstance(node.op, LinearOp) and node not in shared.values()]
    assert non_shared
    assert {node.output.dtype.name for node in non_shared} == {"f16"}

    gate = torch.full((1,), 300.0, dtype=torch.float16)
    up = torch.full((1,), 300.0, dtype=torch.float16)
    assert not torch.isfinite(torch.nn.functional.silu(gate) * up).all()
    assert torch.isfinite(torch.nn.functional.silu(gate.float()) * up.float()).all()


def test_exl3_laguna_routed_down_uses_marked_float32_output():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM

    from emmy.compiler.ir.frontend.ir import LinearOp
    from emmy.compiler.trace.huggingface import (
        _apply_exl3_laguna_routed_scale,
        build_moe_split_wrapper,
        moe_block_parts,
        promote_expert_output_float32,
    )
    from emmy.serving.gen_runner import trace_split

    cfg = _tiny_laguna_config(transformers)
    model = AutoModelForCausalLM.from_config(cfg).half().eval()
    _apply_exl3_laguna_routed_scale(model, exl3=True)
    _pre, _post, expert = build_moe_split_wrapper(model.model.layers[1], split_gate_up=True)
    assert expert._emmy_output_float32 is True
    _gate, experts = moe_block_parts(model.model.layers[1].mlp)
    w_gate, w_up = experts.gate_up_proj[0].chunk(2, dim=0)
    args = [torch.zeros(2, cfg.hidden_size, dtype=torch.float16), w_gate, w_up, experts.down_proj[0]]
    graph = trace_split(expert, args, None)

    weight = graph.nodes["w_down"]
    down = [graph.nodes[uid] for uid in graph.consumers(weight.id) if isinstance(graph.nodes[uid].op, LinearOp)]
    assert len(down) == 1
    assert graph.outputs == [down[0].id]
    assert down[0].output.dtype.name == "f16"

    promote_expert_output_float32(graph)
    assert down[0].output.dtype.name == "f32"

    graph.inputs.remove("w_down")
    with pytest.raises(RuntimeError, match="w_down graph input"):
        promote_expert_output_float32(graph)


def test_exl3_laguna_dense_and_sparse_blocks_preserve_float32_residuals():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM

    from emmy.compiler.ir.frontend.ir import LinearOp
    from emmy.compiler.trace.huggingface import (
        _apply_exl3_laguna_routed_scale,
        build_attention_split_wrapper,
        build_moe_split_wrapper,
        promote_laguna_exl3_post_float32,
        promote_shared_expert_float32,
    )
    from emmy.serving.gen_runner import _retarget_constants, trace_split

    cfg = _tiny_laguna_config(transformers)
    model = AutoModelForCausalLM.from_config(cfg).half().eval()
    _apply_exl3_laguna_routed_scale(model, exl3=True)
    id_to_key = {
        id(tensor): path
        for path, tensor in list(model.named_parameters(remove_duplicate=False)) + list(model.named_buffers(remove_duplicate=False))
    }

    def trace_block(layer, *, sparse):
        block = model.model.layers[layer]
        wrappers = (
            build_moe_split_wrapper(block, split_gate_up=True, float32_residual=True)
            if sparse
            else (*build_attention_split_wrapper(block, float32_residual=True), None)
        )
        pre, post = wrappers[:2]
        pre_graph = trace_split(pre, [torch.zeros(2, cfg.hidden_size, dtype=torch.float32)], None)
        assert pre_graph.nodes[pre_graph.inputs[0]].output.dtype.name == "f32"
        assert {pre_graph.nodes[nid].output.dtype.name for nid in pre_graph.outputs} == {"f16"}

        post_graph = trace_split(
            post,
            [
                torch.zeros(2, cfg.num_attention_heads * cfg.head_dim, dtype=torch.float16),
                torch.zeros(2, cfg.hidden_size, dtype=torch.float32),
            ],
            None,
        )
        _retarget_constants(post_graph, post, id_to_key)
        promote_laguna_exl3_post_float32(post_graph, "sparse" if sparse else "dense")
        if sparse:
            promote_shared_expert_float32(post_graph)
        return post_graph

    dense = trace_block(0, sparse=False)
    sparse = trace_block(1, sparse=True)

    def linear_dtype(graph, suffix):
        weight = next(node for node in graph.nodes.values() if (getattr(node.op, "source_path", "") or "").endswith(suffix))
        users = [graph.nodes[uid] for uid in graph.consumers(weight.id) if isinstance(graph.nodes[uid].op, LinearOp)]
        assert len(users) == 1
        return users[0].output.dtype.name

    assert linear_dtype(dense, ".self_attn.o_proj.weight") == "f32"
    assert linear_dtype(dense, ".mlp.down_proj.weight") == "f32"
    assert [dense.nodes[nid].output.dtype.name for nid in dense.outputs] == ["f32"]
    assert linear_dtype(sparse, ".self_attn.o_proj.weight") == "f32"
    assert linear_dtype(sparse, ".mlp.shared_experts.down_proj.weight") == "f32"
    assert [sparse.nodes[nid].output.dtype.name for nid in sparse.outputs] == ["f32", "f16", "f32"]


def test_float32_residual_final_norm_returns_activation_dtype():
    import numpy as np

    torch = pytest.importorskip("torch")

    from emmy.serving.gen_runner import EmmyGenRunner

    runner = object.__new__(EmmyGenRunner)
    runner._norm = torch.nn.Identity()
    runner._residual_float32 = True
    runner._activation_dtype = torch.float16
    assert runner.residual_dtype == torch.float32
    result = runner.final_norm(np.ones((2, 4), dtype=np.float32))
    assert result.dtype == np.float16


def test_marked_shared_expert_requires_exact_checkpoint_provenance():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM

    from emmy.compiler.trace.huggingface import (
        _apply_exl3_laguna_routed_scale,
        build_moe_split_wrapper,
        promote_shared_expert_float32,
    )
    from emmy.serving.gen_runner import trace_split

    cfg = _tiny_laguna_config(transformers)
    model = AutoModelForCausalLM.from_config(cfg).half().eval()
    _apply_exl3_laguna_routed_scale(model, exl3=True)
    _pre, post, _expert = build_moe_split_wrapper(model.model.layers[1], split_gate_up=True)
    graph = trace_split(post, [torch.zeros(2, cfg.hidden_size, dtype=torch.float16)] * 2, None)
    down = next(
        node for node in graph.nodes.values() if (getattr(node.op, "source_path", "") or "").endswith("shared_experts.down_proj.weight")
    )
    down.op.source_path = "shared_experts.missing.weight"
    with pytest.raises(RuntimeError, match="gate/up/down checkpoint provenance"):
        promote_shared_expert_float32(graph)


def test_unmarked_shared_expert_retains_model_dtype():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM

    from emmy.compiler.trace.huggingface import build_moe_split_wrapper
    from emmy.serving.gen_runner import trace_split

    cfg = _tiny_laguna_config(transformers)
    model = AutoModelForCausalLM.from_config(cfg).half().eval()
    _pre, post, expert = build_moe_split_wrapper(model.model.layers[1], split_gate_up=True)
    assert post._emmy_shared_expert_float32 is False
    assert expert._emmy_output_float32 is False
    graph = trace_split(post, [torch.zeros(2, cfg.hidden_size, dtype=torch.float16)] * 2, None)
    shared_linears = [
        graph.nodes[uid]
        for node in graph.nodes.values()
        if "shared_experts" in (getattr(node.op, "source_path", "") or "")
        for uid in graph.consumers(node.id)
    ]
    assert len(shared_linears) == 3
    assert {node.output.dtype.name for node in shared_linears} == {"f16"}


def test_exl3_routed_scale_leaves_other_architectures_untouched():
    from types import SimpleNamespace

    from emmy.compiler.trace.huggingface import _apply_exl3_laguna_routed_scale

    model = SimpleNamespace(config=SimpleNamespace(model_type="glm4_moe"), marker=object())
    marker = model.marker
    _apply_exl3_laguna_routed_scale(model, exl3=True)
    assert model.marker is marker


def test_laguna_exl3_precision_contract_invalidates_old_pack_key(tmp_path, monkeypatch):
    from emmy.compiler.backend import pack
    from emmy.serving.gen_runner import _generation_precision_contract

    monkeypatch.setattr(pack, "_environment", lambda: {})
    old_key = {"kind": "gen-split", "model": "laguna", "quant_sha": "same-checkpoint"}
    contract = _generation_precision_contract("laguna", {"fmt": "exl3"})
    assert contract == "laguna-exl3-precision-v4"
    assert pack.pack_path(tmp_path, old_key) != pack.pack_path(tmp_path, {**old_key, "precision_contract": contract})
    assert _generation_precision_contract("laguna", {"fmt": "fp8"}) is None
    assert _generation_precision_contract("glm4_moe", {"fmt": "exl3"}) is None


def test_combine_casts_fp32_router_scores():
    """Mixtral-family routers return fp32 scores; the combine must cast them to the activation
    dtype or ``index_add_`` crashes under the forced-fp16 serving lane."""
    torch = pytest.importorskip("torch")

    from emmy.serving.gen_runner import combine_routed_experts

    torch.manual_seed(0)
    xn = torch.randn(5, 8, dtype=torch.float16)
    scores = torch.rand(5, 2, dtype=torch.float32)
    indices = torch.randint(0, 4, (5, 2))
    out = combine_routed_experts(xn, (None, scores, indices), lambda e, rows: rows * (e + 1))
    assert out.dtype == torch.float16
    ref = torch.zeros_like(xn)
    for t in range(5):
        for j in range(2):
            ref[t] += (xn[t] * (indices[t, j].item() + 1)) * scores[t, j].to(torch.float16)
    assert torch.allclose(out, ref, atol=1e-2)


def test_marked_moe_contributions_preserve_the_float32_residual():
    """Large routed and shared terms remain finite in Laguna's float32 residual stream."""
    torch = pytest.importorskip("torch")

    from emmy.serving.gen_runner import _combine_moe_output, _combine_slot_partials, combine_routed_experts

    xn = torch.zeros(1, 1, dtype=torch.float16)
    scores = torch.tensor([[128.0, 128.0]], dtype=torch.float32)
    indices = torch.tensor([[0, 1]])
    partials = torch.tensor([[1024.0], [-1024.0]], dtype=torch.float16)
    gated = (None, scores, indices)

    def run_expert(expert, rows):
        return partials[expert].expand_as(rows)

    assert not torch.isfinite(combine_routed_experts(xn, gated, run_expert)).all()
    routed = combine_routed_experts(xn, gated, run_expert, accumulate_float32=True)
    slots = _combine_slot_partials(scores, partials, xn.dtype, accumulate_float32=True)

    torch.testing.assert_close(routed, torch.zeros_like(xn, dtype=torch.float32), rtol=0, atol=0)
    torch.testing.assert_close(slots, torch.zeros_like(xn, dtype=torch.float32), rtol=0, atol=0)

    h = torch.tensor([[1000.0]], dtype=torch.float32)
    shared = torch.tensor([[73000.0]], dtype=torch.float32)
    routed = torch.tensor([[-1000.0]], dtype=torch.float32)
    result = _combine_moe_output(h, routed, shared)
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, torch.tensor([[73000.0]], dtype=torch.float32), rtol=0, atol=0)


def test_moe_block_parts_rejects_dense_mlp():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from emmy.compiler.trace.huggingface import moe_block_parts

    config = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
    )
    torch.manual_seed(0)
    model = transformers.Qwen3ForCausalLM(config).eval()
    assert moe_block_parts(model.model.layers[0].mlp) is None


def test_flat_qk_norm_pre_matches_eager_projection():
    """OLMoE normalizes the FLAT q/k projections BEFORE the head reshape (norm width == the
    projection width, not head_dim) — the pre carve must reproduce that placement."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers.models.olmoe.modeling_olmoe import OlmoeDecoderLayer

    from emmy.compiler.trace.huggingface import build_moe_split_wrapper

    torch.manual_seed(0)
    cfg = _tiny_olmoe_config(transformers)
    block = OlmoeDecoderLayer(cfg, layer_idx=0).eval()
    for p in block.parameters():
        torch.nn.init.normal_(p, std=0.2)
    pre, _, _ = build_moe_split_wrapper(block)

    t, hd = 5, block.self_attn.head_dim
    nh = block.self_attn.q_proj.out_features // hd
    hidden = torch.randn(t, cfg.hidden_size)
    with torch.no_grad():
        q2, k2, _ = pre(hidden)
        hn = block.input_layernorm(hidden)
        q_ref = block.self_attn.q_norm(block.self_attn.q_proj(hn))
        k_ref = block.self_attn.k_norm(block.self_attn.k_proj(hn))
    assert torch.allclose(q2, q_ref, atol=1e-6)
    assert torch.allclose(k2, k_ref, atol=1e-6)
    assert q2.shape == (t, nh * hd)


# ===================================================================
# EXL3 (trellis-coded) experts: the checkpoint's per-expert MODULES, gate/up kept SEPARATE
# (each coded linear carries its own input-side channel vector), and the E-stacked store the
# expert programs feed from.
# ===================================================================


def test_expert_slot_maps_both_expert_layouts():
    """The v5 E-stacked params and the per-expert-module (EXL3) lineage land on the same input
    names; a shared expert is NOT a routed expert tensor."""
    from emmy.compiler.trace.huggingface import _expert_slot

    assert _expert_slot("model.layers.3.mlp.experts.gate_up_proj") == (3, "w_gate_up", None)
    assert _expert_slot("model.layers.3.mlp.experts.7.gate_proj.trellis") == (3, "w_gate", 7)
    assert _expert_slot("model.layers.3.mlp.experts.7.up_proj.suh") == (3, "w_up_suh", 7)
    assert _expert_slot("model.layers.3.mlp.experts.7.down_proj.svh") == (3, "w_down_svh", 7)
    assert _expert_slot("model.layers.3.mlp.shared_experts.gate_proj.trellis") is None
    assert _expert_slot("model.layers.3.self_attn.q_proj.trellis") is None


def test_laguna_checkpoint_names_map_to_builtin_transformers_modules():
    from emmy.compiler.trace.huggingface import _checkpoint_to_model_key

    assert _checkpoint_to_model_key("model.layers.3.mlp.shared_expert.gate_proj.weight") == (
        "model.layers.3.mlp.shared_experts.gate_proj.weight"
    )
    assert _checkpoint_to_model_key("model.layers.3.mlp.experts.e_score_correction_bias") == (
        "model.layers.3.mlp.gate.e_score_correction_bias"
    )


def test_split_gate_up_expert_wrapper_matches_the_merged_form():
    """``split_gate_up`` takes gate and up as separate forward args — the EXL3 shape, where the
    merged weight has no single activation-side basis. Same math as the chunk-half form."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeDecoderLayer

    from emmy.compiler.trace.huggingface import build_moe_split_wrapper, moe_block_parts

    torch.manual_seed(0)
    cfg = _tiny_glm4_moe_config(transformers)
    block = Glm4MoeDecoderLayer(cfg, layer_idx=1).eval()
    for p in block.parameters():
        torch.nn.init.normal_(p, std=0.2)
    _, _, merged = build_moe_split_wrapper(block)
    _, _, split = build_moe_split_wrapper(block, split_gate_up=True)
    _gate, experts = moe_block_parts(block.mlp)

    x = torch.randn(5, cfg.hidden_size)
    w_gate_up, w_down = experts.gate_up_proj[0], experts.down_proj[0]
    w_gate, w_up = w_gate_up.chunk(2, dim=0)
    with torch.no_grad():
        torch.testing.assert_close(split(x, w_gate, w_up, w_down), merged(x, w_gate_up, w_down))


def _exl3_moe_checkpoint(dirpath, cfg, *, routing_bias=None, omit_routing_bias=False):
    """A tiny GLM-MoE EXL3 checkpoint: every layer linear trellis-coded (the routed experts as
    per-expert modules, the way the real checkpoint stores them), norms / router / embeddings
    fp16. Returns the decoded reference state dict, keyed as the checkpoint stores it."""
    import json

    import numpy as np
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_config(cfg).to(torch.float16).eval()
    if routing_bias is not None:
        # Make expert choice depend only on the nonzero correction bias.  This gives the loader
        # test below a deterministic routing-parity assertion, not merely a tensor-presence pin.
        model.model.layers[1].mlp.gate.weight.data.zero_()
        model.model.layers[1].mlp.gate.e_score_correction_bias.copy_(routing_bias)
    tensors: dict = {}
    ref: dict = {}
    for name, t in model.state_dict().items():
        t = t.detach().cpu()
        if name.endswith(".weight") and t.ndim == 2 and ".layers." in name and "mlp.gate.weight" not in name:
            n, k = t.shape
            coded, dec = exl3_linear_tensors(name[: -len(".weight")], n, k)
            tensors.update(coded)
            ref[name] = torch.from_numpy(np.ascontiguousarray(dec))
        else:
            checkpoint_name = name
            if name.endswith(".mlp.gate.e_score_correction_bias"):
                checkpoint_name = name.replace(".mlp.gate.", ".mlp.experts.")
                if omit_routing_bias:
                    ref[name] = t
                    continue
            tensors[checkpoint_name] = t
            ref[name] = t
    # The v5 3-D expert params never appear in an EXL3 checkpoint — it stores per-expert modules.
    per_expert = {}
    for key in [k for k in tensors if ".experts." in k and k.split(".")[-2].endswith("_proj")]:
        per_expert[key] = tensors.pop(key)
    for key in [k for k in list(tensors) + list(ref) if ".experts." in k and k.endswith(("gate_up_proj", "down_proj"))]:
        tensors.pop(key, None)
        ref.pop(key, None)
    for e in range(cfg.n_routed_experts):
        for layer in range(cfg.first_k_dense_replace, cfg.num_hidden_layers):
            for proj, (n, k) in (
                ("gate_proj", (cfg.moe_intermediate_size, cfg.hidden_size)),
                ("up_proj", (cfg.moe_intermediate_size, cfg.hidden_size)),
                ("down_proj", (cfg.hidden_size, cfg.moe_intermediate_size)),
            ):
                base = f"model.layers.{layer}.mlp.experts.{e}.{proj}"
                coded, dec = exl3_linear_tensors(base, n, k)
                tensors.update(coded)
                ref[base + ".weight"] = torch.from_numpy(np.ascontiguousarray(dec))
    save_file({k: v.clone() for k, v in tensors.items()}, str(dirpath / "model.safetensors"))
    cfg_dict = cfg.to_dict()
    cfg_dict["quantization_config"] = {"quant_method": "exl3", "version": "0.0.5", "bits": 2.0}
    (dirpath / "config.json").write_text(json.dumps(cfg_dict))
    return ref


def test_load_quantized_split_preserves_nonzero_laguna_router_bias(tmp_path):
    """The real EXL3 source spelling aliases onto the built-in router without losing routing."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeTopkRouter

    from emmy.compiler.trace.huggingface import load_quantized_split

    cfg = _tiny_glm4_moe_config(transformers)
    correction = torch.tensor([-0.25, 0.75, 0.125, 0.5], dtype=torch.float32)
    ref = _exl3_moe_checkpoint(tmp_path, cfg, routing_bias=correction)
    model, _store = load_quantized_split(tmp_path, torch.float16, compress_trunk=True)
    loaded = model.model.layers[1].mlp.gate

    reference = Glm4MoeTopkRouter(cfg).eval()
    reference.weight.data.copy_(ref["model.layers.1.mlp.gate.weight"])
    reference.e_score_correction_bias.copy_(ref["model.layers.1.mlp.gate.e_score_correction_bias"])
    hidden = torch.randn(6, cfg.hidden_size, dtype=torch.float16)
    with torch.no_grad():
        expected = reference(hidden)
        actual = loaded(hidden)

    torch.testing.assert_close(loaded.e_score_correction_bias, correction.to(torch.float16), rtol=0, atol=0)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)


def test_load_quantized_split_rejects_missing_laguna_router_bias(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from emmy.compiler.trace.huggingface import load_quantized_split

    cfg = _tiny_glm4_moe_config(transformers)
    correction = torch.tensor([-0.25, 0.75, 0.125, 0.5], dtype=torch.float32)
    _exl3_moe_checkpoint(tmp_path, cfg, routing_bias=correction, omit_routing_bias=True)
    with pytest.raises(ValueError, match="missing routing correction bias"):
        load_quantized_split(tmp_path, torch.float16, compress_trunk=True)


def test_load_quantized_split_keeps_exl3_experts_coded(tmp_path):
    """The serving load: the dense trunk DECODES to values on the twin (it binds off the module,
    which has no checkpoint path), while each routed expert keeps its packed codes — stacked
    E-leading, with full padded channel vectors matching the generic speller's inputs."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from emmy.compiler.trace.huggingface import load_quantized_split

    cfg = _tiny_glm4_moe_config(transformers)
    torch.manual_seed(0)
    ref = _exl3_moe_checkpoint(tmp_path, cfg)
    model, store = load_quantized_split(tmp_path, torch.float16)

    assert store["fmt"] == "exl3"
    assert store["codebooks"] == {1: {"w_gate": 0, "w_up": 0, "w_down": 0}}
    layer = store["layers"][1]
    e = cfg.n_routed_experts
    # Both extents are under one Hadamard block here, so the codes sit at the 128-padded shape.
    assert tuple(layer["w_gate"].shape) == (e, 8, 8, 32) and layer["w_gate"].dtype == torch.int16
    assert tuple(layer["w_gate_suh"].shape) == (e, 128)
    assert tuple(layer["w_gate_svh"].shape) == (e, 128)
    assert tuple(layer["w_down"].shape) == (e, 8, 8, 32)
    assert tuple(layer["w_down_svh"].shape) == (e, 128)
    assert model.model.layers[1].mlp.experts.gate_up_proj.is_meta  # experts never load onto the twin

    sd = model.state_dict()
    for key in ("model.layers.0.self_attn.q_proj.weight", "model.layers.1.mlp.shared_experts.gate_proj.weight"):
        assert not sd[key].is_meta
        torch.testing.assert_close(sd[key], ref[key], rtol=0, atol=0)


def test_load_quantized_split_compress_trunk_leaves_the_trunk_coded(tmp_path):
    """``compress_trunk=True`` is the serving lane: no trunk linear decodes, the twin carries
    placeholders at the declared shapes (so the trace still works), and the store says which
    lane produced it plus where the checkpoint lives — the two things the runner needs to
    re-source those constants from the shards."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from emmy.compiler.trace.huggingface import load_quantized_split

    cfg = _tiny_glm4_moe_config(transformers)
    torch.manual_seed(0)
    ref = _exl3_moe_checkpoint(tmp_path, cfg)
    model, store = load_quantized_split(tmp_path, torch.float16, compress_trunk=True)

    assert store["trunk"] == "codes" and store["dir"] == str(tmp_path)
    sd = model.state_dict()
    for key in ("model.layers.0.self_attn.q_proj.weight", "model.layers.1.mlp.shared_experts.gate_proj.weight"):
        assert not sd[key].is_meta and sd[key].shape == ref[key].shape  # shaped, but NOT decoded
    # The unquantized trunk tensors still load their real values.
    torch.testing.assert_close(sd["model.embed_tokens.weight"], ref["model.embed_tokens.weight"], rtol=0, atol=0)
    _, values_store = load_quantized_split(tmp_path, torch.float16)
    assert values_store["trunk"] == "values"


def test_serving_trunk_constants_retarget_to_checkpoint_keys(tmp_path):
    """The fit blocker, at its root: a split wrapper's traced constants carry WRAPPER-relative
    parameter paths (``q_proj.weight``), which never reach the checkpoint index, so the trellis
    speller cannot fire and the trunk binds decoded values. ``_retarget_constants`` re-addresses
    them by tensor identity; only then does the speller rewrite the linear into the compressed
    chain, and every remaining constant names a real checkpoint key."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from emmy.compiler.loader.quant import spell_trellis_constants
    from emmy.compiler.loader.safetensors import _build_index, _candidate_keys
    from emmy.compiler.trace.huggingface import build_attention_split_wrapper, load_quantized_split
    from emmy.serving.gen_runner import _retarget_constants, trace_split

    cfg = _tiny_glm4_moe_config(transformers)
    torch.manual_seed(0)
    _exl3_moe_checkpoint(tmp_path, cfg)
    model, store = load_quantized_split(tmp_path, torch.float16, compress_trunk=True)
    block = model.model.layers[0]
    pre_w, _post_w = build_attention_split_wrapper(block)
    example = [torch.zeros(4, cfg.hidden_size, dtype=torch.float16)]

    bare = trace_split(pre_w, example, None)
    assert spell_trellis_constants(bare, str(tmp_path)) == 0, "wrapper-relative paths must reach nothing"

    graph = trace_split(pre_w, example, None)
    id_to_key = {
        id(t): path for path, t in list(model.named_parameters(remove_duplicate=False)) + list(model.named_buffers(remove_duplicate=False))
    }
    _retarget_constants(graph, pre_w, id_to_key)
    assert spell_trellis_constants(graph, str(tmp_path)) == 3  # q/k/v
    assert sum(op.source_path is not None and op.source_path.endswith(".trellis") for _nid, op in graph.loadable_constants()) == 3
    index = _build_index(tmp_path)
    for nid, op in graph.loadable_constants():
        if op.source_path is None:
            continue
        assert any(c in index for c in _candidate_keys(op.source_path)), f"{nid}: {op.source_path} is not a checkpoint key"
