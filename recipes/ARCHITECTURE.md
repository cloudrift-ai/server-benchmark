# recipes/ — the recommended serving configuration, one per model

A recipe answers exactly one question: **how should this model be served?** It is the config you deploy, not a
config you measure with.

```bash
emmy deploy local --recipe recipes/<model>          # this machine
emmy deploy ssh   --recipe recipes/<model> --ssh user@host
emmy deploy cloud --recipe recipes/<model>          # provisions the VM first
```

`emmy deploy` leaves the stack running and prints the endpoint (`--teardown` is the opt-in that stops it instead), so
maintained and best-effort recipes are deployable artifacts. Obsolete recipes and untested onboarding shells also live
here for lifecycle continuity, but their tags disable deployment. The recipe format itself — lifecycle tags, matrices,
`cross`/`zip`, deep merge, `extra_args` validation, `docker_options`, command recipes — is documented in
[`emmy/recipe/ARCHITECTURE.md`](../emmy/recipe/ARCHITECTURE.md); this file is about **what belongs here** and why.

Every runnable `recipe.yaml` here also ships inside the published wheel, so `pip install emmy-ml` can deploy without
a checkout: `--recipe <model>` (a bare name, no path) copies the bundled recipe into the current directory and uses
that. Obsolete recipes and onboarding shells remain repository-only. Only the recipe files travel — `RESULTS.md` and
local benchmark output do not.

## Lifecycle

Discovery keeps exactly ten fully configured recipes tagged `maintained` for periodic testing and optimization. Other
useful complete recipes are tagged `best-effort`: they remain runnable and bundled, but are not selected for periodic
work. `obsolete` is reserved for a recipe with an all-around better replacement for the same task at a comparable or
lower practical VRAM footprint. Obsolete recipes are retained rather than deleted, so their configuration and evidence
stay available and a later reassessment can return one to the maintained or best-effort set. New discoveries start as
minimal shells:

```yaml
tags:
  - onboarding
  - untested
model:
  huggingface: org/model
  task: generate
discovery:
  target_gpu: NVIDIA H200 141GB
  target_gpu_count: 1
  rationale: Brief evidence-backed reason.
```

The shell is a handoff to model onboarding, not a serving claim. It becomes runnable only after qualification replaces
the shell with a complete configuration and the `best-effort` lifecycle tag.

## recipes/ vs experiments/

The two directories use the same YAML format and are easy to confuse. The distinction is intent, and it decides
where a file belongs:

| | `recipes/` | `experiments/` |
| --- | --- | --- |
| answers | "how do I serve this model well?" | "which configuration is better, and by how much?" |
| variants | **one** — every value is a decision | many — a workload grid is the point |
| `benchmark:` block | none | required; it defines the measurement |
| lanes | the winner only | the winner *and* the baselines it beat |
| consumed by | `emmy deploy` | `emmy bench` |
| lifetime | updated when a better config is found | configuration retained while it remains useful; output stays local |

This boundary erodes in one direction: a recipe grows a `matrices:` sweep "just to compare two settings", and stops
being a deployment config. That is what happened to the gemma-4-12B recipes — three of them (stock, emmy,
fast-math) sharing a nine-point workload grid, which is an A/B experiment living in the wrong directory. The grids
now live in `experiments/gemma-4-12B/` and `recipes/gemma-4-12B-it` is a single serving variant. If you want to
compare configurations, add an experiment; then fold the winner back into the recipe.

## What a serving recipe should pin

- **The engine and image.** For emmy-accelerated models this is the prebuilt per-model image built by
  [`docker/vllm-emmy-serve/`](../docker/vllm-emmy-serve/ARCHITECTURE.md) — the model snapshot, warmed cubins and the
  execution-plan pack are baked in, so a cold start pays no download, no `nvcc`, and no compiler frontend. Pin its
  canonical immutable reference; the recipe is the publication input to `emmy publish` as well as the deployment
  input.
- **The serving shape**, matching what that image was warmed at. The pack is keyed on the shape (model,
  max-model-len, max-num-batched-tokens, decode bucket), so a recipe that drifts off it still deploys but re-runs
  the compiler frontend per program on every boot — measured at ~50 min of host CPU, which can exceed the compose
  healthcheck window and get the deploy killed as unhealthy. **Changing one of those values means re-warming the
  image**, not just editing a flag. Treat the recipe and `docker/vllm-emmy-serve/models/<slug>.env` as two halves of
  one decision; a test asserts they agree.
- **Tuning knobs as `extra_env`** (`EMMY_FAST_MATH`, `EMMY_GEN_DECODE_BUCKET`, …). These reach the container as real
  environment variables in the generated compose. An Emmy-backed recipe uses `EMMY_FAST_MATH: "1"` by default after
  exact-shape accuracy checks show no quality regression; retain standard Emmy when they do. Mainstream-only vLLM or
  SGLang recipes do not set Emmy knobs.
- **The target hardware**, as a single-entry `matrices:` block. `deploy` resolves it against the detected GPU and
  aborts early if the host cannot satisfy it.

## What limits the values you can pin

Two ceilings decide most of a serving config, and both are measured rather than derived:

- **KV capacity caps context.** Weights come off the card first; whatever remains is the KV pool, shared across
  concurrent requests. vLLM refuses to boot when `max-model-len` exceeds it. For gemma-4-12B FP16 on a 32 GB 5090
  the measured pool is ~25,131 tokens — so its full native context does not fit, and the recipe's context is a
  choice about how many full-length sequences you want resident at once.
- **The prefill-chunk cap is a compiler limit.** `--max-num-batched-tokens` rides the 4096 dynamic-dim cap (plus
  bucket-sized rider headroom); long inputs stream through chunked prefill under it. Context length is not bounded
  by the compiler — RoPE and paged KV are vLLM-owned on this path.
- **A quantized KV cache moves the first ceiling, and on a tight fit it is what makes the config possible at all.**
  `--kv-cache-dtype fp8_e4m3` halves the bytes per token, so it roughly doubles the pool and therefore the context.
  The cache is vLLM-owned on this path (emmy owns no KV), so it is a plain `extra_args` flag. It is a quality
  decision as well as a capacity one: the served model's accuracy is no longer the checkpoint's, so a recipe that
  turns it on owes a quality measurement taken with it on. `recipes/GLM-4.5-Air-EXL3` is the case where nothing fits
  without it.

Both numbers come from the headroom sweep in the release workflow, which is why adding a model means running that
sweep rather than guessing (`release-serving-image` skill, Step 4).

## Adding a model

1. Qualify an image with `make serve-goldens/serve-warm/serve-image/serve-verify MODEL=<hf-id>`, pin its canonical
   reference in the recipe, then validate and publish it with `emmy publish <recipe>` after approval — see
   [`docker/vllm-emmy-serve/ARCHITECTURE.md`](../docker/vllm-emmy-serve/ARCHITECTURE.md). The headroom sweep there
   produces the shape the recipe must match.
2. Add `recipes/<model>/recipe.yaml` pinning that image and that shape, one variant, no `benchmark:` block.
3. If a configuration choice needs justifying, put the A/B in `experiments/<model>/<name>/` and reference the
   finding from the recipe's header comment — not the grid itself.

Every qualified final recipe has one compact, self-contained `RESULTS.md` beside it. It embeds the best complete
measurement for the recipe's selected engine and exact configuration, not a comparison table or a link to raw output.
If a newer complete run regresses, keep the best result in the main section and add one final `## Current regression`
section; remove that section on recovery. Models without a valid final recipe do not get a repository `RESULTS.md`.

Models without emmy kernels are still perfectly good recipes: they pin a stock vLLM or SGLang image and its flags.
`recipes/gemma-4-31B-it` is one — the per-model emmy image serves 12B only.
