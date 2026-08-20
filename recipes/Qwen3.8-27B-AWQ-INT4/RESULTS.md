# Qwen3.8-27B W4A16 on one RTX 4090

Status: serving-qualified with stock vLLM on the text-only path at an 8,192-token context. Emmy is **ineligible** for
this checkpoint: its compiler frontend cannot lower 48 of the model's 64 layers, so this recipe has no Emmy lane and
no comparison column.

## Qualified deployment

| Item | Value |
| --- | --- |
| Model | `philbert440/Qwen3.8-27B-W4A16-AWQ` |
| Model revision | `7908d42a71077a5e4dc458f273682b12dfe384a0` |
| Base checkpoint | `Qwen/Qwen3.8-27B` (W4A16 `compressed-tensors`, pack-quantized, group size 128) |
| Hardware | 1x NVIDIA GeForce RTX 4090, compute capability 8.9 |
| Driver | 580.159.03 |
| Engine | stock vLLM `0.1.dev19754+g3a0914114` (2026-08-19 nightly) |
| Image | `vllm/vllm-openai@sha256:dae7af23ea9b66b4f15de3d5e4ddebfdafa7be636be91d400184c1666f1b1462` |
| Serving shape | TP1, context 8,192, text-only, prefix caching off |
| Repository revision | `a20b10790824a04d195c707d0dda3d8fa5e1cf68` |

Weights occupy 20.7 GiB of the 24 GiB card. That is high for a 4-bit 27B because this checkpoint leaves
`embed_tokens` and `lm_head` unquantized and untied, at 248,320 x 5,120 each — about 5.1 GiB before any transformer
weight is counted. The remaining KV pool measures 2.68 GiB, or 32,182 tokens, which is what caps the context.

## Serving performance

Measured on the recipe's exact engine and serving shape: greedy decoding, ignored EOS, seed 0, no prefix caching.
All rows completed with zero failed requests.

| Input / output tokens | Concurrency | Output throughput | Median TTFT | Median TPOT | Duration |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 / 128 | 4 | 172.1 tokens/s | 313 ms | 21.0 ms | 23.8 s |
| 1,024 / 256 | 4 | 150.3 tokens/s | 1,316 ms | 21.6 ms | 27.2 s |
| 4,096 / 512 | 2 | 76.4 tokens/s | 2,294 ms | 21.7 ms | 53.6 s |

Decode cost is essentially flat at ~21 ms per token across all three points, so on this card the model is
weight-streaming-bound in decode and the differences between points are prefill and batching effects. The recipe keeps
concurrency 4, which is the largest setting the 32,182-token KV pool supports at full 8,192-token sequences.

## Validated capabilities

- **Context 8,192** — verified by retrieving a planted marker from a 7,768-token prompt, not from startup alone. The
  native 262,144-token context does not fit this card at any concurrency.
- **Tool calling** — returns structured `tool_calls` with the correct function name and arguments, and
  `finish_reason: tool_calls`, using `--tool-call-parser qwen3_coder`.
- **Thinking toggle** — `enable_thinking: false` suppresses reasoning cleanly (`reasoning_tokens: 0`); the default
  thinking path produces correct answers.
- **Text-only** — qualified with `--language-model-only`. This is an `any-to-any` checkpoint, but the vision tower is
  unquantized in every available W4A16 build and does not fit alongside a usable KV pool.

### Known limitation: reasoning content is not surfaced

With `--reasoning-parser qwen3`, thinking is correctly stripped from `content` and counted in
`completion_tokens_details.reasoning_tokens`, but `reasoning_content` is returned empty on every request. The chat
template does pre-fill `<think>\n`, which is the layout the parser expects, and no request-level flag changes the
result. No Qwen3.8-specific reasoning parser exists in this image. Clients that need the reasoning text should not
rely on this field until an upstream parser lands.

## Emmy eligibility: ineligible

The blocking gate is compiler coverage. Qwen3.8 is a hybrid checkpoint: 48 of its 64 layers are `linear_attention`
(`Qwen3_5GatedDeltaNet`) and 16 are `full_attention`. The Gated DeltaNet path opens with a depthwise causal
convolution, and Emmy's frontend has no mapping for `aten.conv1d` — there is no convolution operation anywhere in the
Tensor IR, so the operation falls through to the elementwise fallback and raises.

Traced against the real modules built from the pinned configuration, at FP16 and sequence length 512:

| Coverage manifest entry | Result |
| --- | --- |
| `embed_tokens` | traces (3 nodes) |
| final `RMSNorm` | traces (20 nodes) |
| `lm_head` | traces (3 nodes) |
| dense MLP | traces (9 nodes) |
| decoder layer 3 — full attention, 16 of 64 layers | traces (196 nodes) |
| decoder layer 0 — linear attention, 48 of 64 layers | **fails** on `aten.conv1d` |

Reproducer, which needs no GPU because the gap is in the frontend:

```bash
emmy compile --ir torch -c "F.conv1d(torch.randn(1,10240,512), torch.randn(10240,1,4), groups=10240, padding=3)"
```

```
ValueError: cannot map fallback aten.conv1d as elementwise for output shape (1, 10240, 515)
```

The shape is the model's own: `conv_dim` is 10,240 (`2 x 16 x 128` key channels plus `48 x 128` value channels) and
the kernel is 4 wide with padding 3.

Coverage is therefore **partial**, so this recipe intentionally has no `golden/` directory — a partial inventory is
not repository evidence. Because gate 4 fails, gates 3 and 5 (the W4A16 loader path and representative kernel
correctness) were not evaluated for this checkpoint.

### Second, independent gap

`aten.einsum` is not decomposed into permute plus matmul, so it fails broadcasting even for a plain batched product,
while the identical contraction written as `torch.matmul` traces. This one is a missing decomposition rather than a
missing capability:

```bash
emmy compile --ir torch -c "torch.einsum('bij,bjk->bik', torch.randn(48,512,128), torch.randn(48,128,512))"
```

Both gaps sit on the linear-attention path, so closing `conv1d` alone may not be sufficient to qualify the model;
closing it is the prerequisite that unblocks measuring what follows.

## Reproduce

```bash
emmy bench experiments/Qwen3.8-27B-AWQ-INT4/serving --ssh USER@HOST
```

## Limits

- The image is a dated nightly, not a release tag. Re-resolve it before any new performance run, and re-check whether
  a release tag has superseded it.
- The 24 GB card holds 8,192 tokens of context at concurrency 4. Raising either requires a build that quantizes
  `embed_tokens`/`lm_head`, which none of the current community W4A16 repositories do.
- `reasoning_content` is unusable; see the limitation above.
- Multimodal input is out of scope for this platform and untested here.
