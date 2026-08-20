# Model VRAM Fit

The single shared definition of "does this checkpoint fit this GPU platform?". The `discover-models` and
`onboard-model` skills both follow this contract so a proposed deployment and a measured one describe the same
quantity. No repository code estimates fit; this reasoning is the only place it is decided.

## Read the checkpoint, not the name

A model ID is not evidence of size. Before proposing or accepting any `(GPU name, GPU count)`, read the checkpoint's
`config.json` and model card and record:

- **total parameters** — every weight that must be resident, including every expert of a Mixture of Experts model;
- **active parameters** — for a Mixture of Experts model, the subset routed per token;
- **dtype and quantization**, and which quantized repositories actually exist for this checkpoint;
- **native context length**.

A name like `...-17B-16E` reports active parameters and expert count, never the total. Resolve the total from
`config.json` (`num_experts` × expert width, plus the dense remainder) or from an explicit model-card statement.

## Compute the footprint

```
weight VRAM ≈ total_params(B) × bytes_per_param
    bytes_per_param:  BF16/FP16 = 2  ·  FP8 = 1  ·  AWQ/INT4 ≈ 0.5
min-to-serve ≈ weight VRAM × 1.3        # + CUDA graphs, activations, a small KV cache at modest context
long-context / high-concurrency ≈ weight VRAM × 1.5+   (KV cache grows with context × concurrency)
```

Two traps:

- **A Mixture of Experts model is sized by TOTAL parameters.** Every expert loads into memory; active parameters
  drive throughput, not residency. A 109B-total / 17B-active checkpoint needs 109B worth of weights resident.
- **Quantization decides the GPU.** The same checkpoint in BF16 versus AWQ is a 4× difference. Only count a
  quantization whose repository exists — a hypothetical FP8 repacking is not a deployment.

**Multi-GPU:** when one card cannot hold `min-to-serve`, tensor-parallel across N GPUs; per-GPU need is
`min-to-serve / N` plus per-GPU overhead. N must divide the model's attention head count; prefer 2, 4, or 8.

## Compare against real GPU capacity

`emmy/gpu.py` is the authority on canonical GPU names and their `vram_mib`. Read it; do not work from remembered
capacities or a table copied into a skill. Total platform capacity is that GPU's `vram_mib` × `gpu_count`.

A deployment is admissible only when `min-to-serve` fits the total platform capacity. Tensor parallelism divides one
model across GPUs — it never lowers the total that must be resident, so it cannot rescue a platform whose combined
VRAM is below the weight footprint.

## Record the arithmetic

State the numbers you used wherever the fit conclusion lands — a recipe rationale, a shortlist row, or a failure
report: total parameters, bytes per parameter and the quantization they come from, the resulting `min-to-serve`, and
the total capacity of the proposed platform. A fit claim without these numbers cannot be checked or disagreed with,
and a later measurement on real hardware must be comparable to the estimate that proposed it.

When the smallest admissible platform is outside the available fleet, say so plainly and propose no deployment.
Proposing a platform that cannot hold the weights spends rented GPU hours to rediscover arithmetic.
