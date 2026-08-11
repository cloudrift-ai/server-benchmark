# DeepSeek V4 Flash 0731 on 16x V100 SXM3 32GB

Status: serving-qualified under the canonical local image name; Docker Hub publication still requires separate
approval and namespace credentials.

Emmy eligibility: **ineligible**. The complete static-sequence-512 compiler golden qualifies the architecture on
V100, but DeepSeek V4's full-layer provider is audit-only because its HCA/CSA compressor and hyper-connection path
does not fit the current external-attention serving ABI. `emmy serve --generate` therefore cannot execute this exact
checkpoint through the Emmy runner. The first unmet eligibility gate is a real serving path; this is a qualified 1Cat
lane, not an Emmy comparison, and no Emmy serving image is claimed.

Measured on 2026-08-10 with model revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062`, sixteen 32 GB V100
SXM3 GPUs, and 1Cat commit `d76126608155c334df7c2fb9b75096f879624859`. The baked native-SM70 cache image is
`cloudriftai/1cat-vllm-deepseek-v4-flash-0731:1.2.3-d76126608`, with local image ID
`sha256:276240257b224097876b5b6db8f0d32484dff6a6f168d6b03d6df188e5c65bc1`. The supporting 1Cat changes are in
[cloudrift-ai/1Cat-vLLM pull request 2](https://github.com/cloudrift-ai/1Cat-vLLM/pull/2).

## Recommended configuration

The checkpoint loaded all 48 weight shards with TP8 and PP2. Pipeline stages held 22 and 21 model layers. The
qualified lane uses FP16 activations, FP8 KV cache, the SM70 sparse MLA route, TurboMind W8A16 dense and grouped-BMM
paths, and the TurboMind MXFP4 MoE path. Model loading took 19.83 seconds and engine profiling, KV-cache creation,
and warmup took 84.05 seconds. The healthy service used about 27.0 GiB per GPU on PP0 and 28.5 GiB per GPU on PP1.

The recipe keeps the tested 4096-token context and eight-request concurrency. It removes `--enforce-eager`; vLLM
selected `FULL_AND_PIECEWISE` graph mode and captured decode sizes 1, 2, 4, 8, and 16. Graph mode improved TPOT by
1.41–4.24x across the 16-cell qualification matrix.

## Accuracy and capability checks

Deterministic chat probes returned `Paris`, `4`, `323`, and `OK` for terse factual and arithmetic requests. A tool
probe emitted an OpenAI-compatible `multiply` call with arguments `{"a": 17, "b": 19}` and finish reason
`tool_calls`. Repeating the direct 17-by-19 prompt produced the correct answer first, but then repeated malformed
reasoning markup until the 32-token limit. This is a response-formatting caveat, not a numerical mismatch; concise
answer-only prompts stopped normally.

Request-time warming reached 19 Triton functions across prefill, decode, batching, and tool-call shapes. Greedy
logprobs and sampling added two more functions plus six specializations of existing functions. The rebuilt active-off
image passed the complete 16-cell matrix, 3968-token boundary cases, greedy and sampling probes, and the structured
tool call from a fresh container while fail-closed compiler guards stayed empty and all cache manifests remained
unchanged. An active-expert B1 route cut TPOT by 39–58%, but changed the second-token ranking and is therefore not
enabled in the recipe. Exact probes, route evidence, and artifact locations are in the
[serving experiment](../../experiments/DeepSeek-V4-Flash-0731/serving_v100_sxm3/RESULTS.md).
Compiler coverage and tuning evidence are documented separately in the
[compiler experiment](../../experiments/DeepSeek-V4-Flash-0731/compiler_v100_sxm3/RESULTS.md).

## 2026-08-11 recipe revalidation

The current recipe's immutable `model.revision` drove both Hugging Face prefetch and the engine. Prefetch reused
`/home/riftuser/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062`
without downloading checkpoint data, and the live engine reported the same model and tokenizer revision. A fresh
deployment passed Emmy's smoke test, returned `Paris` and arithmetic result `323`, and emitted the expected
`multiply` tool call with `{"a": 17, "b": 19}`.

The representative 1024-input/64-output, concurrency-8 experiment ran three identical client repeats against one
deployment. All 24 requests completed with exact token counts and no failures. The three-repeat mean was 198.74 ms
TPOT, 31.71 output tokens/s, 539.05 total tokens/s, and 3,857.82 ms TTFT. The first repeat had higher client warmup
variance; repeats two and three measured 194.58 and 190.98 ms TPOT. The durable raw result is
`experiments/DeepSeek-V4-Flash-0731/serving_v100_sxm3/2026-08-11_02-35-06_f8e924ce/v100x16_vllm_benchmark.json`.
The deployment and the earlier compiler-replay task were both stopped; no container or GPU compute process remained.

## Image publication

An earlier source-image push failed with `insufficient_scope`; no Docker credential was created or left on the VM.
The canonical image and checkpoint remain on `riftuser@185.165.50.61`. Publication now goes only through the recipe
gate. First run its read-only check and show the exact image ID, destination, labels, and collision result:

```bash
emmy publish recipes/DeepSeek-V4-Flash-0731 \
  --source-image cloudriftai/1cat-vllm-deepseek-v4-flash-0731:1.2.3-d76126608 \
  --dry-run
```

Only after a separate human approval should an operator obtain a short-lived Docker Hub token, log in with tracing
disabled, replace `--dry-run` with `--yes`, verify the registry digest, and log out. Use a Docker Hub token for this
operation, not `HF_TOKEN`.
