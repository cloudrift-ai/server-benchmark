# Gemma 4 12B base-checkpoint qualification on RTX 5090

Status: text-completion serving qualified at 16,384 tokens. The base checkpoint is not instruction-tuned, so the
recommended recipe uses `/v1/completions` and a semantic completion smoke test rather than treating an empty chat
answer as a serving failure.

## Scope

- Date: 2026-08-11
- Model: `google/gemma-4-12B`
- Immutable revision: `023679ed352de9bb66cc873c9009ce3482585c08`
- Hardware: 1× `NVIDIA GeForce RTX 5090`, compute capability 12.0, driver 580.173.02
- Serving image: `vllm/vllm-openai:v0.23.0`, digest
  `sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f`
- Precision: FP16 weights and KV cache
- Recommended context: 16,384 tokens, one concurrent request, `gpu_memory_utilization: 0.95`

## Fit and modality boundary

The checkpoint advertises a 262,144-token native context. An exploratory boot at 0.90 memory utilization failed
before a request: vLLM required 5.08 GiB of KV cache but had 4.81 GiB available, estimating a 244,544-token ceiling
at that setting. The later 0.95 serving run does not establish whether the full native context fits at its higher
budget, so that envelope remains unqualified rather than claimed impossible. The recommended 16,384-token envelope
boots cleanly while leaving the active graphical session intact; 0.96 fails the same startup headroom check
documented by the article reproduction.

The text completion `2 + 2 =` returns the correct answer through `/v1/completions`. Image and video requests reached
the model forward path but produced empty base-model continuations, so they are not qualified as useful generation.
Audio requests fail before the model because the stock image does not include vLLM's optional audio dependencies.
The durable recipe is therefore text-only; it does not claim multimodal output quality.

## Serving benchmark

The final recipe runs 256 input tokens, 128 output tokens, four prompts, concurrency 1, greedy decoding, and ignored
EOS. The clean final run completed 4/4 requests at 60.15 output tok/s, 180.45 total tok/s, 56.60 ms median TTFT, and
16.18 ms median TPOT. The semantic completion smoke passed before the benchmark. The structured result and full log
are preserved under [`2026-08-11_02-03-46_f32ab819`](2026-08-11_02-03-46_f32ab819).

## Compiler evidence

`emmy/compiler/pipeline/search/goldens/rtx5090_sm120_gemma4.yaml` is the shared Gemma 4 geometry golden used by the
base checkpoint and its instruction-tuned sibling. It currently contains 281 self-contained programs and all stored
RTX 5090 realizations deserialize under the current schema. The file deliberately remains revision-untagged because
the release gate uses the base geometry for the `-it` sibling; serving reproducibility comes from the immutable
revision in this recipe.

## Decision

Promote `recipes/gemma-4-12B/recipe.yaml` as the stock FP16 text-completion configuration. Do not describe it as a
chat or audio recipe, and do not advertise the full native context on one 32 GB RTX 5090.
