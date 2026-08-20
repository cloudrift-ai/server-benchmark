# Automated Lifecycle Selection

Use the attached discovery task as the exhaustive recipe inventory. Its `recipe_batches` partition every existing
recipe exactly once, and `maintained_count` is the required maintained-set size. Do not rebuild the inventory or
reconstruct an existing model ID.

## Research and scoring

1. Invoke `discover-reddit`, `discover-huggingface`, and `discover-openrouter` once and in parallel. Treat their
   results as independent candidate sources. Reddit may surface a candidate absent from the other two sources.
2. Merge and deduplicate their evidence. Use at most three parent-agent public-web calls for targeted identity or
   capability verification. Verify the exact open-weight Hugging Face ID of every new candidate.
3. Invoke `discover-scorer` once per recipe batch and in parallel. Give each scorer the complete contents of the
   attached `score-recipes.md`, the shared source evidence, and exactly one batch. Scorers only return heat scores and
   rationales; they do not choose lifecycle states or new models.
4. Merge the batch results, compare scores globally, and adjust only when needed to keep the heat bands consistent.

## Lifecycle decisions

Select exactly `maintained_count` fully configured recipes for periodic testing and optimization. Existing recipes
tagged `onboarding` are untested shells: score them, but never select them as maintained or obsolete. The workflow
will preserve their task and deployment matrix deterministically.

Prefer current community demand, serving value, architecture coverage, and a useful spread of sizes in the maintained
set. Every unselected complete recipe defaults to best-effort; do not return best-effort IDs because the workflow
derives them mechanically.

Propose an obsolete recipe only when a named all-around better complete replacement for the same task is available at
a comparable or lower practical VRAM footprint sized by the attached `model-fit.md`, or when a concrete technical
limitation means the recipe should no longer be used. Before proposing a replacement, read both recipe files and
confirm that the old recipe retains no advantage in configured context, concurrency, quantization, hardware support,
model capability, latency, throughput, operating cost, modality, or licensing. A replacement that is merely comparable
is not all-around better. Low demand, age, and exclusion from the maintained set are not sufficient. Existing obsolete
tags are prior proposals, not evidence; reassess them under this policy. Put the exact replacement ID and VRAM
comparison in the obsolete model's score rationale. When no successor is appropriate, the rationale must state the
concrete technical limitation.

Add every genuinely promising newly discovered model to `new_onboarding_models`; there is no candidate-count limit.
Use task `embed` only for embedding models. Give each new model one to three useful deployments containing only
canonical `deploy.gpu` and positive `deploy.gpu_count` values. Derive every deployment from the attached
`model-fit.md`: read the checkpoint's total parameters, size the footprint, and compare it against that platform's
real capacity. No repository code checks this, so an unservable deployment reaches rented hardware unchallenged. Put
the total parameter count, the quantization, and the resulting footprint in the model's rationale. Do not claim an
onboarding shell was deployed or benchmarked. Never put an existing recipe ID in `new_onboarding_models`, including an
existing onboarding shell; an existing ID appears only in `scores`.

## Output

Return exactly one JSON object without prose or a Markdown fence:

```json
{
  "scores": [
    {"model_id": "exact/existing-id", "rationale": "Evidence-backed rationale of at most 20 words.", "heat": 75}
  ],
  "maintained_model_ids": ["exact/existing-id"],
  "obsolete_models": [
    {"model_id": "exact/old-id", "replacement_model_id": "exact/replacement-id"},
    {"model_id": "exact/technically-broken-id"}
  ],
  "new_onboarding_models": [
    {
      "model_id": "exact/new-id",
      "task": "generate",
      "rationale": "Why this model is worth onboarding.",
      "heat": 90,
      "deployments": [{"deploy.gpu": "NVIDIA H200 141GB", "deploy.gpu_count": 1}]
    }
  ]
}
```

`scores` must contain every existing recipe from every batch exactly once, including onboarding shells. Copy each
existing `model_id` letter for letter. Each score contains exactly `model_id`, `rationale`, and `heat`; heat is an
integer from 0 through 100. `maintained_model_ids` contains exact IDs only. Each obsolete entry contains `model_id`
and optionally `replacement_model_id`. `new_onboarding_models` contains new models only.

Before returning, verify that the scores cover every batch row and that the maintained count is exact. If OpenCode
requests the final response, return the best complete selection immediately without another tool call.

Do not edit the repository, rent hardware, deploy a model, or return the final lifecycle manifest. Repository code
validates this compact selection, restores existing onboarding data, derives best-effort decisions, and assembles the
manifest.
