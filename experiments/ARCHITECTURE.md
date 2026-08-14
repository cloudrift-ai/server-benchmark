# experiments/ — reproducible measurements grouped by model

An experiment answers a comparison or qualification question. It uses the recipe format and runs through
`emmy bench`; the recommended serving configuration belongs in `recipes/`.

## Directory convention

```text
experiments/<model>/<workload_or_question>_<hardware>/
  recipe.yaml
  results/                    # raw evidence from the last run
  <row>.experiment.yaml      # assembled record for each last-run row
  RESULTS.md                  # factual system/status/artifact index
```

Use the model's established repository slug and a short `snake_case` experiment name. Keep one protocol in one recipe
when platforms differ only by hardware allocation or a small control; use a zipped matrix rather than copied command
bodies. Split directories only when the workload or raw evidence set differs.

## Last-run artifacts

`emmy bench` replaces `results/` for every actual invocation and writes one YAML experiment record per expanded row
there. It keeps raw client/server logs and every declared command result beside those records. It never writes legacy
JSON/TXT wrappers, a task/instance manifest, or a report. Dry runs leave the preceding snapshot untouched.

Use the repository `run-experiment` skill to finish a requested run. The skill checks matrix-row coverage, system
information, artifact presence, and terminal status; moves the records beside `recipe.yaml`; overwrites `RESULTS.md`;
force-stages the ignored artifacts; and commits the complete snapshot. Existing records, raw results, and artifact
index always describe the same most recent run; do not accumulate dated run directories.

`RESULTS.md` is a factual artifact index. It identifies the run, machine, protocol, row status, failures, and raw
artifacts, but contains no benchmark values, comparisons, conclusions, performance language, or scientific
interpretation. Recipes cannot contain post-processing or report-generation commands; command blocks are only the
measured workload.

## Lifetime

Keep configurations that reproduce a published comparison, support a durable qualification, or are needed for a
planned measurement. Delete exploratory configurations and one-off analysis helpers after their conclusion is encoded
in a final recipe, golden configuration, test, or architecture note.

See [`recipes/ARCHITECTURE.md`](../recipes/ARCHITECTURE.md) for the serving-recipe boundary and
[`emmy/recipe/ARCHITECTURE.md`](../emmy/recipe/ARCHITECTURE.md) for the YAML format.
