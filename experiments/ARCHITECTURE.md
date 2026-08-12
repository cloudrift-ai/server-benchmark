# experiments/ — reproducible measurement configurations grouped by model

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
`experiments/gemma-4-12B/serving_rtx5090/`. Each child directory owns one `recipe.yaml`.

Measured output does not belong here by default. Benchmark JSON/TXT/logs, plots, dated recipe snapshots, compiler run
summaries, and experiment `RESULTS.md` files stay ignored or outside the checkout. After a configuration is selected,
embed its compact best result in `recipes/<model>/RESULTS.md` beside the final deployment recipe. Retain a measured
experiment artifact only when the caller explicitly requests that exact file as durable publication evidence.

Related command workloads also stay under the model. Image qualification, kernel smoke tests, serving smoke tests,
and final benchmarks for one model should not become unrelated top-level directories. A genuinely cross-model
compiler experiment may remain at the top level.

## Result analysis

`emmy bench` is an experiment-agnostic runner. Recipes define workloads and neutral matrix labels; they do not define
semantic gates, log predicates, or output comparisons. A short self-contained `aggregate.run` command may perform
readable mechanical post-processing, but it may not invoke an external script or generate the experiment report.
Tests verify the intended configuration before measurement. After a run, an agent examines every raw result, failure,
log, and artifact against the experiment protocol and writes the model-specific report.

## Lifetime

Keep experiment configurations that reproduce a published comparison, protect a qualification gate, or are needed
for a planned measurement. Delete exploratory configurations, intermediate candidates, duplicate run snapshots, and
their one-off helper scripts after the result is encoded in a final recipe, its `RESULTS.md`, a golden config, a test,
or a durable architecture note. Plans and local run artifacts are not experiment deliverables.

See [`recipes/ARCHITECTURE.md`](../recipes/ARCHITECTURE.md) for the serving-recipe boundary and
[`emmy/recipe/ARCHITECTURE.md`](../emmy/recipe/ARCHITECTURE.md) for the YAML format.
