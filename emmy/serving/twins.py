"""Weight-free serving-twin capture — the graphs serving compiles, without the weights.

``EmmyGenRunner`` compiles two programs per decoder layer (the ``pre`` and ``post``
attention halves) at three widths (static decode bucket, static prefill chunk, symbolic).
Capturing those graphs normally means downloading the full checkpoint
(``scripts/capture_gen_twins.py``), but nothing in a TRACE reads a weight value — only
config-derived shapes and dtypes matter. So this module builds a random-init skeleton from
the model's ``config.json`` alone (a few-KB fetch, or a checked-in fixture directory) and
traces the twins through the exact ``build_attention_split_wrapper`` / ``trace_split``
path serving uses.

The skeleton is constructed on ``meta`` and one representative of every distinct per-layer
structure is materialized: attention type, dense-vs-MoE MLP type, and per-layer head count.
The vocab shrinks to a stub since the twins never touch the embedding or lm_head. This is two
profiles for Gemma-4 (local/global), but three for Laguna (dense/full, sparse/sliding,
sparse/full). Routed expert weights remain graph inputs and are never materialized.

A coded checkpoint gets one more loader-owned step. The weight-free inventory of which modules
are coded, at what rate, and with what sibling shapes comes from the checkpoint allocation
sidecar. The same generic birth-time reconstruction builder used by deployed model tracing is
applied here, so checkpoint metadata never escapes into compiler IR.

Because an "optimized" rung allocates bits per tensor, one traced layer does not represent the
whole trunk: GLM-4.5-Air 2.25 stores q/k/v at 4 bits on 42 layers and 3 bits on 4. The rate is
recorded in the inventory, so a coded twin is emitted once per distinct rate profile and named
``<twin>@b<rates>`` before equivalent generic targets are deduplicated.

Consumed by ``emmy eval golden GOLDEN_YAML --serving-config PATH`` and the serving-image release gate. Note the traced
graph tracks the installed ``transformers`` modeling code: a transformers bump that
changes the model's forward changes these twins — exactly as it would change serving.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from emmy.compiler.loader.exl3 import coded_tensor_storage

if TYPE_CHECKING:
    from emmy.compiler.graph import Graph

logger = logging.getLogger(__name__)

#: Serving's default static widths (``EmmyGenRunner`` / ``capture_gen_twins`` conventions).
DECODE_BUCKET = 32
PREFILL_BUCKET = 256
#: DeepSeek V4's canonical compiler inventory width. Four complete HCA windows and
#: 128 complete CSA windows exercise both compressor families without tracing a whole model.
DEEPSEEK_V4_AUDIT_WIDTH = 512


def capture_in_model_graphs(model: str, *, extra_widths: tuple[int, ...] = ()) -> dict[str, Graph]:
    """Capture the sound graph provider for an in-model golden audit.

    Most generative models use the exact serving ``pre`` / ``post`` / expert twins below.
    DeepSeek V4 cannot: its HCA/CSA compressors and hyper-connection residual streams do
    not fit the classic external-attention seam. Audit that architecture through the exact
    representative full-layer inventory path instead. Both providers are config-only and
    track the installed Transformers modeling code.
    """
    from transformers import AutoConfig  # noqa: PLC0415

    from emmy.compiler.loader.safetensors import split_revision  # noqa: PLC0415

    if any(width <= 0 for width in extra_widths):
        raise ValueError(f"in-model audit widths must be positive, got {extra_widths}")
    repo, revision = split_revision(model)
    cfg = AutoConfig.from_pretrained(repo, revision=revision)
    text = getattr(cfg, "text_config", cfg)
    if getattr(text, "model_type", None) == "deepseek_v4":
        if extra_widths:
            raise ValueError(
                "DeepSeek V4 in-model audit uses the fixed full-layer architecture width "
                f"{DEEPSEEK_V4_AUDIT_WIDTH}; additional serving-twin widths {extra_widths} cannot be claimed by this provider"
            )
        return _capture_deepseek_v4_architecture_graphs(model, text)
    return capture_twin_graphs(model, extra_widths=extra_widths)


def _capture_deepseek_v4_architecture_graphs(model: str, config, *, seq_len: int = DEEPSEEK_V4_AUDIT_WIDTH) -> dict[str, Graph]:
    """Trace one exact full decoder layer per DeepSeek V4 structural profile."""
    import torch  # noqa: PLC0415

    from emmy.compiler.loader.safetensors import split_revision  # noqa: PLC0415
    from emmy.compiler.trace.huggingface import (  # noqa: PLC0415
        find_text_decoder,
        load_architecture_trace_twin,
        trace_selected_layer,
    )

    repo, revision = split_revision(model)
    graphs: dict[str, Graph] = {}
    for layer, attention_type, mlp_type in _deepseek_v4_profiles(config):
        twin = load_architecture_trace_twin(repo, torch.float16, layer, revision=revision)
        decoder = find_text_decoder(twin)
        if not bool(getattr(getattr(decoder.layers[layer], "mlp", None), "_emmy_traceable_expert", False)):
            raise NotImplementedError(
                f"DeepSeek V4 in-model profile at layer {layer} lacks confirmed representative routed-expert replacement"
            )
        graph, _bundle = trace_selected_layer(twin, layer, seq_len, torch.float16)
        name = f"layer{seq_len}-{_attention_label(attention_type)}-{mlp_type.replace('_', '-')}"
        if name in graphs:
            raise ValueError(f"DeepSeek V4 in-model profile name collision for {name!r}")
        graphs[name] = graph
    return graphs


def _deepseek_v4_profiles(config) -> list[tuple[int, str, str]]:
    """First layer index for every installed DeepSeek V4 attention/MLP pairing."""
    count = getattr(config, "num_hidden_layers", None)
    attention_types = list(getattr(config, "layer_types", None) or [])
    mlp_types = list(getattr(config, "mlp_layer_types", None) or [])
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError(f"DeepSeek V4 in-model audit requires positive num_hidden_layers, got {count!r}")
    if len(attention_types) != count or len(mlp_types) != count:
        raise ValueError(
            "DeepSeek V4 in-model audit requires one attention and MLP type per layer: "
            f"layers={count}, attention={len(attention_types)}, mlp={len(mlp_types)}"
        )
    allowed_attention = {"sliding_attention", "heavily_compressed_attention", "compressed_sparse_attention"}
    allowed_mlp = {"hash_moe", "moe"}
    unknown_attention = set(attention_types) - allowed_attention
    unknown_mlp = set(mlp_types) - allowed_mlp
    if unknown_attention or unknown_mlp:
        raise NotImplementedError(
            "DeepSeek V4 in-model audit does not recognize installed layer signatures: "
            f"attention={sorted(unknown_attention)}, mlp={sorted(unknown_mlp)}"
        )

    profiles = []
    seen = set()
    for layer, pair in enumerate(zip(attention_types, mlp_types, strict=True)):
        if pair in seen:
            continue
        seen.add(pair)
        profiles.append((layer, *pair))
    return profiles


def _serving_twin_buckets(
    decode_bucket: int,
    prefill_bucket: int,
    extra_widths: tuple[int, ...],
    *,
    symbolic: bool,
    static_only: bool,
) -> list[tuple[str, int | None]]:
    """Return exactly the requested serving widths and optional symbolic template."""
    if static_only:
        if decode_bucket != 1 or prefill_bucket != 0 or extra_widths:
            raise ValueError("static-only serving capture requires decode_bucket=1, prefill_bucket=0, and no extra widths")
        return [("1", 1)]

    widths = [decode_bucket, prefill_bucket, *extra_widths]
    buckets: list[tuple[str, int | None]] = [(str(m), m) for m in sorted({w for w in widths if w})]
    if symbolic:
        buckets.append(("-sym", None))
    return buckets


def capture_twin_graphs(
    model: str,
    *,
    decode_bucket: int = DECODE_BUCKET,
    prefill_bucket: int = PREFILL_BUCKET,
    extra_widths: tuple[int, ...] = (),
    symbolic: bool = True,
    dtype: str = "float16",
    static_only: bool = False,
) -> dict[str, Graph]:
    """Trace every serving twin of ``model`` from its config alone (no weights).

    ``model`` is an HF hub id or a local directory holding ``config.json``; a hub id may pin a
    branch or commit as ``<repo>@<revision>`` (a coded checkpoint's rung lives on a branch, and
    the rungs differ in exactly the bit allocation the keys carry). Returns
    ``{"pre32": Graph, "post32": …, "pre256": …, "pre-sym": …}`` plus ``-global``
    variants of each when the model has ``full_attention`` layers — the same names
    ``scripts/capture_gen_twins.py`` writes. ``extra_widths`` adds release-specific decode or
    prefill buckets. On an EXL3 checkpoint each
    twin holding coded weights is replaced by its spelled forms, one per rate profile
    (``…@b4``). ``static_only`` is the deliberate exception: it accepts only the proven
    decode-1/prefill-0 envelope and emits M=1 without any standard or symbolic twins."""
    import torch  # noqa: PLC0415
    from transformers import AutoConfig, AutoModel  # noqa: PLC0415

    from emmy.compiler.loader.quant import strip_engine_quant_config  # noqa: PLC0415
    from emmy.compiler.loader.safetensors import split_revision  # noqa: PLC0415
    from emmy.compiler.trace.huggingface import (  # noqa: PLC0415
        build_attention_split_wrapper,
        build_moe_split_wrapper,
        moe_block_parts,
    )
    from emmy.serving.gen_runner import trace_split  # noqa: PLC0415

    model, revision = split_revision(model)
    cfg = AutoConfig.from_pretrained(model, revision=revision)
    text = getattr(cfg, "text_config", cfg)
    if getattr(text, "model_type", None) == "deepseek_v4":
        raise NotImplementedError(
            "DeepSeek V4 cannot be captured as classic serving twins: HCA/CSA compressors and hyper-connection "
            "residuals do not fit the (q, k, v) seam; use capture_in_model_graphs for its full-layer audit provider"
        )
    storage = coded_tensor_storage(model, cfg, revision=revision)
    strip_engine_quant_config(text)
    text.vocab_size = 32  # the twins are decoder halves — the embedding/lm_head are never traced
    if (text.pad_token_id or 0) >= text.vocab_size:
        text.pad_token_id = 0  # the stub vocab must still contain the padding index (nn.Embedding asserts it)
    td = getattr(torch, dtype)
    # Keep every original layer index: Laguna's per-layer head counts, attention types, and
    # dense/sparse MLP types are three parallel arrays. Collapsing only ``layer_types`` silently
    # constructs the wrong architecture (and materializing all 256 experts per sparse layer is
    # several GiB). Meta construction costs no weight memory; only the small split wrappers below
    # are materialized on CPU.
    with torch.device("meta"):
        try:
            trunk = AutoModel.from_config(text, dtype=td).eval()
        except ValueError as exc:
            if "trust_remote_code" not in str(exc):
                raise
            trunk = AutoModel.from_config(text, dtype=td, trust_remote_code=True).eval()
    hidden = text.hidden_size

    # Every static width a serving mode deploys gets a twin, so the in-model golden audit
    # sees the SAME forms the packs compile: the decode bucket (32), the documented c=4/c=8
    # decode-bucket knob (8) and the c=64 one (64), the MTP c=64 verify bucket (192 — the
    # serving_mtp_rtx5090 depth-2 lane; it had no records of any kind until the 2026-07-30/31
    # m192 seeding, which is the misdeploy class this list exists to make visible), the legacy
    # prefill bucket (256), the c=4/c=8 chunk-quantum knob (2048 — EMMY_GEN_PREFILL_BUCKET=2048
    # + mnbt 2056), and the chunked-prefill static width (4096 — the --max-num-batched-tokens
    # chunk every 4K+ prompt rides). The 2026-07-23 lane regressions (m64 decode TPOT 74 ms,
    # m4096 TTFT +260 ms) were merged-key coverage gaps INVISIBLE to the audit precisely
    # because these widths were missing here — MATCH 74/0/0 while serving cold-resolved.
    if any(width <= 0 for width in extra_widths):
        raise ValueError(f"serving twin widths must be positive, got {extra_widths}")
    buckets = _serving_twin_buckets(
        decode_bucket,
        prefill_bucket,
        extra_widths,
        symbolic=symbolic,
        static_only=static_only,
    )

    signatures = _layer_signatures(trunk, text)
    layers = _profile_layers(trunk, text)

    graphs: dict[str, Graph] = {}
    layer_scopes: dict[str, set[int]] = {}
    for layer_idx, block, suffix in layers:
        members = {i for i, signature in enumerate(signatures) if signature == signatures[layer_idx]}
        parts = moe_block_parts(block.mlp)
        if parts is None:
            pre_w, post_w = build_attention_split_wrapper(block)
            expert_w = None
        else:
            pre_w, post_w, expert_w = build_moe_split_wrapper(block, split_gate_up=bool(storage))
        # ``pre`` and ``post`` reference only attention/dense/shared-expert parameters. Calling
        # to_empty on the wrappers avoids materializing the packed routed-expert table that still
        # hangs off ``block.mlp`` (4.8 GiB for one Laguna layer).
        pre_w.to_empty(device="cpu").to(td)
        post_w.to_empty(device="cpu").to(td)
        _num_heads, attn_width = _attention_query_layout(block.self_attn)
        for name, m in buckets:
            # The symbolic program traces at the example width serving uses (8) and ties
            # axis-0 to a ``num_tokens`` Dim; a static twin traces at its bucket, no Dim.
            rows = 8 if m is None else m
            halves = [
                ("pre", pre_w, [torch.zeros(rows, hidden, dtype=td)], ["hidden"] if m is None else None),
                (
                    "post",
                    post_w,
                    [torch.zeros(rows, attn_width, dtype=td), torch.zeros(rows, hidden, dtype=td)],
                    ["attn_out", "residual"] if m is None else None,
                ),
            ]
            for half, wrapper, example_args, argnames in halves:
                with torch.device("cpu"):
                    twin_name = f"{half}{name}{suffix}"
                    graphs[twin_name] = trace_split(wrapper, example_args, argnames)
                    layer_scopes[twin_name] = members

            if expert_w is not None:
                rows = 8 if m is None else m
                examples = _expert_examples(parts[1], rows, hidden, td, split_gate_up=bool(storage))
                argnames = ["x"] if m is None else None
                graph = trace_split(expert_w, examples, argnames)
                expert_name = f"expert{name}{suffix}"
                if storage:
                    graphs.update(_spell_expert_twins(expert_name, graph, storage))
                else:
                    graphs[expert_name] = graph
    return _spell_coded_twins(graphs, storage, layer_scopes=layer_scopes) if storage else graphs


def _profile_layers(trunk, config) -> list[tuple[int, object, str]]:
    """One decoder layer per distinct serving structure.

    Historically twins distinguished only local vs global attention and kept the local profile's
    empty suffix plus ``-global``. Preserve those names for existing models. Architectures with
    heterogeneous MLP type or per-layer head counts (Laguna) get explicit, reviewable suffixes:
    ``-dense-full``, ``-sparse-sliding``, and ``-sparse-full`` for Laguna-S-2.1.
    """
    signatures = _layer_signatures(trunk, config)

    selected: list[tuple[int, tuple[str, str, int]]] = []
    seen = set()
    for i, signature in enumerate(signatures):
        if signature not in seen:
            seen.add(signature)
            selected.append((i, signature))

    rich = len({m for m, _a, _h in signatures}) > 1 or len({h for _m, _a, h in signatures}) > 1
    if not rich:
        out = []
        for i, (_mlp, attn, _heads) in selected:
            out.append((i, trunk.layers[i], "-global" if "full" in attn and any("full" not in a for _m, a, _h in signatures) else ""))
        return out

    pairs = [(m, _attention_label(a)) for m, a, _h in (sig for _i, sig in selected)]
    out = []
    for i, (mlp, attn, nheads) in selected:
        pair = (mlp, _attention_label(attn))
        suffix = f"-{pair[0]}-{pair[1]}"
        if pairs.count(pair) > 1:
            suffix += f"-h{nheads}"
        out.append((i, trunk.layers[i], suffix))
    return out


def _layer_signatures(trunk, config) -> list[tuple[str, str, int]]:
    """The construction-relevant per-layer fields that select a serving program."""
    from emmy.compiler.trace.huggingface import moe_block_parts  # noqa: PLC0415

    types = list(getattr(config, "layer_types", None) or [])
    mlp_types = list(getattr(config, "mlp_layer_types", None) or [])
    heads = list(getattr(config, "num_attention_heads_per_layer", None) or [])

    def at(values, i, default):
        return values[i] if i < len(values) else default

    out = []
    for i, block in enumerate(trunk.layers):
        attn = at(types, i, "homogeneous")
        mlp = at(mlp_types, i, "sparse" if moe_block_parts(block.mlp) is not None else "dense")
        inferred_heads, _width = _attention_query_layout(block.self_attn)
        nheads = at(heads, i, inferred_heads)
        out.append((str(mlp), str(attn), int(nheads)))
    return out


def _attention_query_layout(attn) -> tuple[int, int]:
    """Return ``(num_heads, flattened query width)`` for a supported attention signature.

    Classic attention exposes one ``q_proj``. DeepSeek V4 instead exposes a low-rank
    ``q_a_proj`` / ``q_b_proj`` path and a single-head shared ``kv_proj``. Validate the
    complete shape relation before accepting the latter; a partial unfamiliar signature
    must fail instead of producing a plausible but wrong post-attention example tensor.
    """
    head_dim = getattr(attn, "head_dim", None)
    if not isinstance(head_dim, int) or isinstance(head_dim, bool) or head_dim <= 0:
        raise NotImplementedError(f"serving twin attention {type(attn).__name__} requires a positive integer head_dim, got {head_dim!r}")

    q_proj = getattr(attn, "q_proj", None)
    if q_proj is not None:
        width = getattr(q_proj, "out_features", None)
        if not isinstance(width, int) or isinstance(width, bool) or width <= 0 or width % head_dim:
            raise ValueError(
                f"serving twin attention {type(attn).__name__} q_proj width {width!r} is not a positive multiple of head_dim={head_dim}"
            )
        num_heads = width // head_dim
        declared = getattr(attn, "num_heads", None)
        if declared is not None and declared != num_heads:
            raise ValueError(
                f"serving twin attention {type(attn).__name__} declares num_heads={declared!r} but q_proj/head_dim imply {num_heads}"
            )
        return num_heads, width

    deepseek_fields = ("q_a_proj", "q_a_norm", "q_b_proj", "q_b_norm", "kv_proj", "kv_norm", "o_a_proj", "o_b_proj")
    if not all(getattr(attn, field, None) is not None for field in deepseek_fields):
        raise NotImplementedError(
            f"serving twin attention {type(attn).__name__} has neither q_proj nor the complete DeepSeek low-rank query/shared-KV signature"
        )
    num_heads = getattr(attn, "num_heads", None)
    if not isinstance(num_heads, int) or isinstance(num_heads, bool) or num_heads <= 0:
        raise ValueError(f"serving twin DeepSeek attention requires positive integer num_heads, got {num_heads!r}")
    width = getattr(attn.q_b_proj, "out_features", None)
    kv_width = getattr(attn.kv_proj, "out_features", None)
    if width != num_heads * head_dim or kv_width != head_dim:
        raise ValueError(
            f"serving twin DeepSeek attention shape mismatch: q_b_proj={width!r}, kv_proj={kv_width!r}, "
            f"num_heads={num_heads}, head_dim={head_dim}"
        )
    return num_heads, width


def _attention_label(layer_type: str) -> str:
    return layer_type.removesuffix("_attention").replace("_", "-")


def _expert_examples(experts, rows: int, hidden: int, dtype, *, split_gate_up: bool):
    """Shape-only forward arguments for one routed expert; no packed expert is materialized."""
    import torch  # noqa: PLC0415

    from emmy.compiler.trace.huggingface import moe_expert_layout  # noqa: PLC0415

    gate_up = tuple(int(d) for d in experts.gate_up_proj.shape[1:])
    down = tuple(int(d) for d in experts.down_proj.shape[1:])
    args = [torch.zeros(rows, hidden, dtype=dtype)]
    if split_gate_up:
        if gate_up[0] % 2:
            raise ValueError(f"coded expert gate/up output dimension must be even, got {gate_up}")
        args += [torch.zeros((gate_up[0] // 2, gate_up[1]), dtype=dtype) for _ in range(2)]
    else:
        args.append(torch.zeros(gate_up, dtype=dtype))
    args.append(torch.zeros(down, dtype=dtype))
    _transposed, _interleaved, has_bias = moe_expert_layout(experts)
    if has_bias:
        args += [
            torch.zeros(tuple(experts.gate_up_proj_bias.shape[1:]), dtype=dtype),
            torch.zeros(tuple(experts.down_proj_bias.shape[1:]), dtype=dtype),
        ]
    return args


def _spell_expert_twins(name: str, graph: Graph, storage: dict) -> dict[str, Graph]:
    """Spell each distinct EXL3 routed-expert allocation profile into input-sourced twins."""
    from emmy.compiler.loader.quant import spell_trellis_inputs  # noqa: PLC0415

    groups: dict[tuple[str, int, int], dict[str, str]] = {}
    for base in storage:
        head, sep, rest = base.partition(".layers.")
        fields = rest.split(".")
        if not sep or len(fields) < 5 or not fields[0].isdigit() or fields[1:3] != ["mlp", "experts"] or not fields[3].isdigit():
            continue
        proj = fields[4]
        if proj in {"gate_proj", "up_proj", "down_proj"}:
            groups.setdefault((head, int(fields[0]), int(fields[3])), {})[proj] = base

    out: dict[str, Graph] = {}
    seen = set()
    for modules in groups.values():
        if set(modules) != {"gate_proj", "up_proj", "down_proj"}:
            continue
        specs = {}
        profile = []
        for inp, proj in (("w_gate", "gate_proj"), ("w_up", "up_proj"), ("w_down", "down_proj")):
            base = modules[proj]
            args = _spell_args(base, storage[base])
            specs[inp] = (args["cb"], args["shapes"]["trellis"])
            profile.append((inp, int(storage[base]["bits_per_weight"]), specs[inp]))
        frozen = tuple(profile)
        if frozen in seen:
            continue
        candidate = graph.copy()
        try:
            spell_trellis_inputs(candidate, specs)
        except ValueError:
            continue  # a different expert width/layout; another structural twin owns it
        seen.add(frozen)
        rates = "-".join(str(b) for b in sorted({bits for _inp, bits, _spec in profile}))
        out[f"{name}@b{rates}"] = candidate
    if not out:
        raise ValueError(f"no coded routed-expert allocation matches serving twin {name!r}")
    return out


def _spell_coded_twins(graphs: dict[str, Graph], storage: dict, *, layer_scopes: dict[str, set[int]] | None = None) -> dict[str, Graph]:
    """Replace every twin holding coded weights by its spelled forms, one per rate profile.

    A twin traces ONE representative decoder layer, and its constant paths are wrapper-relative
    and partly flattened (``q_proj.weight`` for the checkpoint's ``…self_attn.q_proj``), so a
    module is matched to a checkpoint entry by dotted-suffix within one layer. Each layer then
    offers one candidate allocation; distinct allocations are distinct kernels now that the rate
    keys, so each is emitted, suffixed with its rates (``pre32@b4`` / ``pre32@b3``). Twins with
    no coded weight pass through untouched under their original name."""
    by_layer = _layers(storage)
    out: dict[str, Graph] = {}
    for name, graph in graphs.items():
        spelled = _spell_one(graph, storage, by_layer, allowed_layers=None if layer_scopes is None else layer_scopes.get(name))
        out.update({f"{name}@b{rates}": g for rates, g in spelled.items()} if spelled else {name: graph})
    return out


def _layers(storage: dict) -> list[tuple[int, list[str]]]:
    """``storage``'s coded module names grouped by decoder layer, in layer order (a string sort
    would run ``layers.10`` before ``layers.2``, so the first profile would not be layer 0's)."""
    groups: dict[tuple, list[str]] = {}
    for name in storage:
        head, sep, rest = name.partition(".layers.")
        idx = rest.split(".", 1)[0]
        if sep and idx.isdigit():
            groups.setdefault((head, int(idx)), []).append(name)
    return [(key[1], groups[key]) for key in sorted(groups)]


def _spell_one(
    graph: Graph, storage: dict, by_layer: list[tuple[int, list[str]]], *, allowed_layers: set[int] | None = None
) -> dict[str, Graph]:
    """``{rate label: spelled copy}`` for one twin — empty when it holds no coded weight."""
    from emmy.compiler.loader.quant import _spell_trellis_linear  # noqa: PLC0415

    weights = {op.source_path[: -len(".weight")]: nid for nid, op in graph.loadable_constants() if _is_weight(op)}
    out: dict[str, Graph] = {}
    seen: set[tuple] = set()
    for layer_idx, names in by_layer:
        if allowed_layers is not None and layer_idx not in allowed_layers:
            continue
        matched = {mod: base for mod in weights if (base := _match(names, mod))}
        # A suffix may exist on a structurally different profile (Laguna full-attention q/o
        # projections have different N/K from its sliding layers). Never emit a partially coded
        # twin: every matched module must reproduce the traced logical weight shape.
        if any(not _storage_matches_weight(storage[base], graph.nodes[weights[mod]].op) for mod, base in matched.items()):
            continue
        hits = matched
        profile = tuple(sorted((mod, int(storage[base]["bits_per_weight"])) for mod, base in hits.items()))
        if not profile or profile in seen:
            continue
        seen.add(profile)
        g = graph.copy()
        for mod, base in hits.items():
            _spell_trellis_linear(g, weights[mod], **_spell_args(base, storage[base]))
        out["-".join(str(b) for b in sorted({b for _, b in profile}))] = g
    return out


def _storage_matches_weight(entry: dict, op) -> bool:
    shape = tuple(int(d) for d in (op.source_shape or ()))
    if len(shape) != 2:
        return False
    n, k = shape
    stored = entry["stored_tensors"]
    base = next(iter(stored)).rsplit(".", 1)[0]
    suh = tuple(stored[f"{base}.suh"]["shape"])
    svh = tuple(stored[f"{base}.svh"]["shape"])
    return suh == (-(-k // 128) * 128,) and svh == (-(-n // 128) * 128,)


def _match(names: list[str], mod: str) -> str | None:
    """The one module in this layer whose checkpoint name ends with the traced module path
    ``mod`` — ``mlp.gate_proj`` hits the dense MLP and never ``mlp.experts.3.gate_proj``. An
    ambiguous suffix names no module: guessing there would spell the wrong rate."""

    # Original Laguna checkpoints use ``shared_expert`` while built-in Transformers exposes
    # ``shared_experts``. Match through that load-boundary alias without changing provenance.
    def canonical(name: str) -> str:
        return name.replace(".shared_expert.", ".shared_experts.")

    hits = [name for name in names if canonical(name).endswith("." + canonical(mod))]
    return hits[0] if len(hits) == 1 else None


def _is_weight(op) -> bool:
    return bool(op.source_path) and op.source_path.endswith(".weight")


def _spell_args(base: str, entry: dict) -> dict:
    """``base`` / ``cb`` / ``shapes`` for one coded module, read off ``tensor_storage`` — the
    same three arguments the checkpoint-driven speller derives from the safetensors index."""
    stored = entry["stored_tensors"]
    return {
        "base": base,
        # Marker-sibling PRESENCE selects the codebook, exactly as ``quant._exl3_codebook`` reads
        # it off the index; the stored values are never read.
        "cb": 1 if f"{base}.mcg" in stored else 2 if f"{base}.mul1" in stored else 0,
        "shapes": {leaf: tuple(stored[f"{base}.{leaf}"]["shape"]) for leaf in ("trellis", "suh", "svh")},
    }
