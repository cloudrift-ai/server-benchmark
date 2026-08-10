# DeepSeek V4 Flash 0731 on 16x V100 SXM3 32GB

Status: serving-qualified; Docker Hub publication is blocked on namespace credentials.

Measured on 2026-08-10 with model revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062`, sixteen 32 GB V100
SXM3 GPUs, and 1Cat commit `d76126608155c334df7c2fb9b75096f879624859`. The local native-SM70 image is
`cloudriftai/onecat-vllm-deepseek-v4-flash-0731:sm70-d76126608`; its image ID and local manifest digest are
`sha256:d08f373fe9558def8b6ab6589b4d80e5b021caa3e1e40687192305a49c5b11c6`, and its size is 10,968,934,035
bytes. The supporting 1Cat changes are in
[cloudrift-ai/1Cat-vLLM pull request 2](https://github.com/cloudrift-ai/1Cat-vLLM/pull/2).

## Recommended configuration

The checkpoint loaded all 48 weight shards with TP8 and PP2. Pipeline stages held 22 and 21 model layers. The
qualified lane uses FP16 activations, FP8 KV cache, the SM70 sparse MLA route, TurboMind W8A16 dense and grouped-BMM
paths, and the TurboMind MXFP4 MoE path. Model loading took 19.83 seconds and engine profiling, KV-cache creation,
and warmup took 84.05 seconds. The healthy service used about 27.0 GiB per GPU on PP0 and 28.5 GiB per GPU on PP1.

The recipe keeps the tested 4096-token context, one-request concurrency, and eager execution. The
`VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH` setting was present in the qualified launch but is inactive under eager
execution; it remains in the recipe to preserve the exact launch environment.

## Accuracy and capability checks

Deterministic chat probes returned `Paris`, `4`, `323`, and `OK` for terse factual and arithmetic requests. A tool
probe emitted an OpenAI-compatible `multiply` call with arguments `{"a": 17, "b": 19}` and finish reason
`tool_calls`. Repeating the direct 17-by-19 prompt produced the correct answer first, but then repeated malformed
reasoning markup until the 32-token limit. This is a response-formatting caveat, not a numerical mismatch; concise
answer-only prompts stopped normally.

The first live request JIT-compiled 16 Triton kernels that the image warmup did not cover. Later serialized `OK`
requests all returned HTTP 200 with two completion tokens in 0.879345, 1.160202, and 1.152433 seconds. This bounded
smoke workload establishes a coherent service, not representative throughput. Exact probes, route evidence, and
artifact locations are in the
[serving experiment](../../experiments/DeepSeek-V4-Flash-0731/serving_v100_sxm3/RESULTS.md).
Compiler coverage and tuning evidence are documented separately in the
[compiler experiment](../../experiments/DeepSeek-V4-Flash-0731/compiler_v100_sxm3/RESULTS.md).

## Image publication

The Docker Hub push failed with `insufficient_scope`; no Docker credential was created or left on the VM. The exact
image and checkpoint remain on `riftuser@185.165.50.61`. After creating the Docker Hub repository and obtaining a
Docker Hub personal access token with write access to the `cloudriftai` namespace, retry on that VM with:

```bash
read -r DOCKERHUB_USER
read -rs DOCKERHUB_PAT
printf '%s' "${DOCKERHUB_PAT}" | sudo docker login --username "${DOCKERHUB_USER}" --password-stdin
sudo docker push cloudriftai/onecat-vllm-deepseek-v4-flash-0731:sm70-d76126608
sudo docker logout
unset DOCKERHUB_PAT DOCKERHUB_USER
```

Use a Docker Hub token for this operation, not `HF_TOKEN`.
