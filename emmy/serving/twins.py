"""Weight-free serving-twin capture — the graphs serving compiles, without the weights.

``EmmyGenRunner`` compiles two programs per decoder layer (the ``pre`` and ``post``
attention halves) at three widths (static decode bucket, static prefill chunk, symbolic).
Capturing those graphs normally means downloading the full checkpoint
(``scripts/capture_gen_twins.py``), but nothing in a TRACE reads a weight value — only
config-derived shapes and dtypes matter. So this module builds a random-init skeleton from
the model's ``config.json`` alone (a few-KB fetch, or a checked-in fixture directory) and
traces the twins through the exact ``build_attention_split_wrapper`` / ``trace_split``
path serving uses.

The skeleton is trimmed before construction: ``layer_types`` collapses to one local layer
plus (when the model has any) one ``full_attention`` layer — per-layer structure is what's
traced, not depth — and the vocab shrinks to a stub since the twins never touch the
embedding or lm_head. Gemma-4's global layers use a different head_dim, so the ``-global``
twins genuinely differ from the local ones.

A CODED checkpoint (EXL3 trellis) gets one more step. Serving retargets each traced constant to
its checkpoint key and runs ``spell_trellis_constants(…, expand=True)``, so a deployed trunk
linear is not an f16 matmul at all — it is the activation-side basis restore around a hat-basis
coded contraction. A twin that skipped that would key every trunk projection as its uncompressed
twin and report GAP on every coded golden. The weight-free inventory of which modules are coded,
at what rate and with what sibling shapes comes from the loader
(:func:`~emmy.compiler.loader.exl3.coded_tensor_storage`, which reads the checkpoint's small
allocation sidecar) — that is everything
:func:`~emmy.compiler.loader.quant._spell_trellis_activation_one` needs. Calling that same
function is deliberate: a re-implementation of the spelling here would drift from the deployed
one, which is the "matched but never deploys" class.

Because an "optimized" rung allocates bits per tensor, one traced layer does not represent the
whole trunk: GLM-4.5-Air 2.25 stores q/k/v at 4 bits on 42 layers and 3 bits on 4. The rate is
part of the ``ShapeKey``, so both must be audited — a coded twin is therefore emitted once per
distinct RATE PROFILE, named ``<twin>@b<rates>``.

Consumed by ``emmy eval golden --in-model`` and the golden drift CI gate. Note the traced
graph tracks the installed ``transformers`` modeling code: a transformers bump that
changes the model's forward changes these twins — exactly as it would change serving.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from emmy.compiler.loader.exl3 import coded_tensor_storage

if TYPE_CHECKING:
    from emmy.compiler.graph import Graph

logger = logging.getLogger(__name__)

#: Serving's default static widths (``EmmyGenRunner`` / ``capture_gen_twins`` conventions).
DECODE_BUCKET = 32
PREFILL_BUCKET = 256


def capture_twin_graphs(
    model: str,
    *,
    decode_bucket: int = DECODE_BUCKET,
    prefill_bucket: int = PREFILL_BUCKET,
    symbolic: bool = True,
    dtype: str = "float16",
) -> dict[str, Graph]:
    """Trace every serving twin of ``model`` from its config alone (no weights).

    ``model`` is an HF hub id or a local directory holding ``config.json``; a hub id may pin a
    branch or commit as ``<repo>@<revision>`` (a coded checkpoint's rung lives on a branch, and
    the rungs differ in exactly the bit allocation the keys carry). Returns
    ``{"pre32": Graph, "post32": …, "pre256": …, "pre-sym": …}`` plus ``-global``
    variants of each when the model has ``full_attention`` layers — the same names
    ``scripts/capture_gen_twins.py`` writes. On an EXL3 checkpoint each twin holding coded
    weights is replaced by its spelled forms, one per rate profile (``…@b4``)."""
    import torch  # noqa: PLC0415
    from transformers import AutoConfig, AutoModel  # noqa: PLC0415

    from emmy.compiler.trace.huggingface import build_attention_split_wrapper  # noqa: PLC0415
    from emmy.serving.gen_runner import trace_split  # noqa: PLC0415

    model, revision = _split_revision(model)
    cfg = AutoConfig.from_pretrained(model, revision=revision)
    text = getattr(cfg, "text_config", cfg)
    types = list(getattr(text, "layer_types", None) or [])
    local = next((t for t in types if "full" not in t), None)
    glob = next((t for t in types if "full" in t), None)
    if types:
        text.layer_types = [t for t in (local, glob) if t]
        text.num_hidden_layers = len(text.layer_types)
    else:
        text.num_hidden_layers = 1  # homogeneous model (no layer_types): one layer suffices
    text.vocab_size = 32  # the twins are decoder halves — the embedding/lm_head are never traced
    if (text.pad_token_id or 0) >= text.vocab_size:
        text.pad_token_id = 0  # the stub vocab must still contain the padding index (nn.Embedding asserts it)
    td = getattr(torch, dtype)
    with torch.device("cpu"):
        trunk = AutoModel.from_config(text).eval().to(td)
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
    widths = [1, 8, decode_bucket, 64, 192, prefill_bucket, 2048, 4096]  # 1 = the EMMY_GEN_M1_TIER gemv twins
    buckets: list[tuple[str, int | None]] = [(str(m), m) for m in sorted({w for w in widths if w})]
    if symbolic:
        buckets.append(("-sym", None))

    layers = [(trunk.layers[0], "")]
    if glob is not None and len(trunk.layers) > 1:
        layers.append((trunk.layers[1], "-global"))

    graphs: dict[str, Graph] = {}
    for block, suffix in layers:
        pre_w, post_w = build_attention_split_wrapper(block)
        attn_width = block.self_attn.q_proj.out_features  # this layer's num_heads * head_dim
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
                    graphs[f"{half}{name}{suffix}"] = trace_split(wrapper, example_args, argnames)
    storage = coded_tensor_storage(model, cfg, revision=revision)
    return _spell_coded_twins(graphs, storage) if storage else graphs


def _split_revision(model: str) -> tuple[str, str | None]:
    """``"repo@rev"`` → ``("repo", "rev")``; anything else (including a local path, which is
    checked first so a directory whose name contains ``@`` still resolves) → ``(model, None)``."""
    if Path(model).is_dir() or "@" not in model:
        return model, None
    repo, _, rev = model.rpartition("@")
    return repo, rev


def _spell_coded_twins(graphs: dict[str, Graph], storage: dict) -> dict[str, Graph]:
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
        spelled = _spell_one(graph, storage, by_layer)
        out.update({f"{name}@b{rates}": g for rates, g in spelled.items()} if spelled else {name: graph})
    return out


def _layers(storage: dict) -> list[list[str]]:
    """``storage``'s coded module names grouped by decoder layer, in layer order (a string sort
    would run ``layers.10`` before ``layers.2``, so the first profile would not be layer 0's)."""
    groups: dict[tuple, list[str]] = {}
    for name in storage:
        head, sep, rest = name.partition(".layers.")
        idx = rest.split(".", 1)[0]
        if sep and idx.isdigit():
            groups.setdefault((head, int(idx)), []).append(name)
    return [groups[k] for k in sorted(groups)]


def _spell_one(graph: Graph, storage: dict, by_layer: list[list[str]]) -> dict[str, Graph]:
    """``{rate label: spelled copy}`` for one twin — empty when it holds no coded weight."""
    from emmy.compiler.loader.quant import _spell_trellis_activation_one  # noqa: PLC0415

    weights = {op.source_path[: -len(".weight")]: nid for nid, op in graph.loadable_constants() if _is_weight(op)}
    out: dict[str, Graph] = {}
    seen: set[tuple] = set()
    for names in by_layer:
        hits = {mod: base for mod in weights if (base := _match(names, mod))}
        profile = tuple(sorted((mod, int(storage[base]["bits_per_weight"])) for mod, base in hits.items()))
        if not profile or profile in seen:
            continue
        seen.add(profile)
        g = graph.copy()
        # The kernel-path hint serving's ``expand=True`` spelling stamps: without it ``032``
        # folds the hat-basis cone into one bind-time constant, which a weight-free twin cannot
        # evaluate — and which would key the projection as a plain f16 matmul even if it could.
        g.hints.set("trellis.expand", True)
        spelled = sum(_spell_trellis_activation_one(g, weights[mod], **_spell_args(base, storage[base])) for mod, base in hits.items())
        if spelled < len(hits):
            # Serving falls back to the folded checkpoint-basis cone here; a weight-free twin
            # cannot (nothing may read a value), so the weight stays f16 and its fork keys
            # uncoded. Visible as a GAP rather than a silent wrong MATCH.
            logger.warning("coded twin: %d of %d coded weights declined the activation-side spelling", len(hits) - spelled, len(hits))
        out["-".join(str(b) for b in sorted({b for _, b in profile}))] = g
    return out


def _match(names: list[str], mod: str) -> str | None:
    """The one module in this layer whose checkpoint name ends with the traced module path
    ``mod`` — ``mlp.gate_proj`` hits the dense MLP and never ``mlp.experts.3.gate_proj``. An
    ambiguous suffix names no module: guessing there would spell the wrong rate."""
    hits = [name for name in names if name.endswith("." + mod)]
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
