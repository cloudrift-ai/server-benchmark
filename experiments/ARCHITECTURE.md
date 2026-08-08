# experiments/ — reproducible measurements grouped by model

An experiment answers a comparison or qualification question. It uses the recipe format, but it is run with
`emmy bench`; the recommended serving configuration belongs in `recipes/`.

## Directory convention

Group model-specific experiments under one model directory:

```text
experiments/<model>/
  <workload_or_question>_<hardware>/
    recipe.yaml
```

Use the model's established repository slug and a short `snake_case` child name. Name the workload or question
first and add the hardware suffix when the result is hardware-specific, as in
`experiments/gemma-4-12B/serving_rtx5090/`. Each child directory owns one `recipe.yaml`; committed result files, when
they are publication evidence, live beside that recipe.

Related command workloads also stay under the model. Image qualification, kernel smoke tests, serving smoke tests,
and final benchmarks for one model should not become unrelated top-level directories. A genuinely cross-model
compiler experiment may remain at the top level.

## Lifetime

Keep experiments that reproduce a published comparison, protect a durable qualification gate, or are needed for a
planned final measurement. Delete exploratory sweeps, intermediate candidates, duplicate run snapshots, and their
one-off helper scripts after the result is encoded in a recipe, golden config, test, or durable architecture note.
Plans and local run artifacts are not experiment deliverables.

See [`recipes/ARCHITECTURE.md`](../recipes/ARCHITECTURE.md) for the serving-recipe boundary and
[`emmy/recipe/ARCHITECTURE.md`](../emmy/recipe/ARCHITECTURE.md) for the YAML format.
