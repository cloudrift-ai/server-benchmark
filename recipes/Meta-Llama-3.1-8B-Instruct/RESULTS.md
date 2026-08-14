# Llama 3.1 8B Instruct FP16 on one 32 GB V100

Status: serving-qualified at 65,536 tokens on one Tesla V100-SXM3-32GB with the pinned SM70 1Cat/vLLM runtime.

## Qualified deployment

| Item | Value |
| --- | --- |
| Hardware | 1× Tesla V100-SXM3-32GB, SM70 |
| Context | 65,536 tokens |
| Max sequences | 8 |
| Tensor parallelism | 1 |
| Driver / CUDA | 580.159.03 / 13.0 |
| Model revision | `d10aef7999a2b5ba950ab3974312feeedbfe0b77` |
| Image digest | `sha256:8405bb60d24610417d0d6da278a753e2c968bfd1e0d7ff7f79cd6601a038b2be` |
| KV capacity reported at boot | 99,120 tokens |

The official Meta repository is gated, so the recipe pins the public NousResearch mirror. Its `config.json` and all
four safetensor shard SHA-256 values were checked against Meta revision
`0e9e39f249a16976918f6564b8830bc894c89659` and are identical. The model has 8,030,261,248 parameters and occupies
about 15 GiB in FP16. The mirror omits Meta's tool-aware tokenizer template, so the recipe pins the tokenizer from
the qualified AWQ repository; its tokenizer config is byte-identical to Meta's.

The native 131,072-token context failed the measured KV fit gate: 11.42 GiB was available while 16.0 GiB was needed,
with a boot estimate of 93,584 maximum tokens. Halving to 65,536 loaded with 1.51× reported concurrency.

## Serving performance

Measured 2026-08-13 with three repeats of 32 requests, 512 input tokens, 256 forced output tokens, concurrency 8,
greedy decoding, and two warmup requests. All 96 measured requests succeeded.

| Output tok/s | Total tok/s | Requests/s | Mean TTFT | Mean TPOT / ITL |
| ---: | ---: | ---: | ---: | ---: |
| 280.80 ± 9.64 | 842.40 ± 28.92 | 1.093 ± 0.038 | 333.18 ms | 27.30 / 27.30 ms |

Cold model load and warmup took 108.1 seconds. The three output-throughput repeats were 267.17, 287.98, and
287.24 tokens/s.

## Chat, tools, and context

- OpenAI-compatible chat completion returned the correct answer to `2 + 2`.
- Tool use emitted a parsed `get_weather` call with `{"city":"Paris"}`.
- A material 60,041-token prompt plus three generated tokens completed in 84.93 seconds and returned `OK.` without OOM.

## Emmy compiler qualification

The FP16 layer path was evaluated separately from serving. At sequence length 1, layer 0 compiled into five Emmy
kernels and passed direct strict eager comparison at `rtol=atol=1e-3` with 0.000977 maximum absolute error. Emmy took
223,882 µs versus 773 µs eager, a 290× regression. At the required 512-token prefill shape, generated CUDA failed to
compile because an `f2x26` tiled reduction referenced undefined unrolled-axis variables. Emmy-generated kernels are
therefore not promoted; the qualified deployment uses the pinned 1Cat/vLLM Volta kernels.

## Reproduce

```bash
emmy bench --ssh riftuser@66.172.10.131 \
  experiments/Meta-Llama-3.1-8B-Instruct/serving_v100_sxm3_32gb
```

The benchmark writes ignored local output. `RESULTS.md` is the only retained measurement artifact.

## Limits

- This FP16 profile is qualified through 65,536 tokens; native 131,072 did not fit the measured KV-cache budget.
- The tool-aware tokenizer is pinned separately because the byte-identical model mirror lacks Meta's template.
- The exact image is already present on the qualified host. A registry pull by this locally resolved digest returned
  `manifest unknown`, so preload or republish it before using the recipe on a new host.
- Requalify after changing the image, model or tokenizer revision, driver, context length, or attention backend.
- Emmy compiler serving is not qualified for this model on V100.
