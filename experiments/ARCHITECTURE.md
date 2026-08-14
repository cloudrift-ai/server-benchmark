# experiments/ — reproducible measurements grouped by model

An experiment answers a comparison or qualification question. It uses the recipe format and runs through
`emmy bench`; the recommended serving configuration belongs in `recipes/`.

## Directory convention

```text
experiments/<model>/<workload_or_question>_<hardware>/
  recipe.yaml
  <YYYY-MM-DD_HH-MM-SS>/      # temporary ignored raw output
  results.tar.gz              # Git LFS archive of the last run directory
  <row>.experiment.yaml      # assembled record for each last-run row
  RESULTS.md                  # thoughtful interpretation of the last run
```

Use the model's established repository slug and a short `snake_case` experiment name. Keep one protocol in one recipe
when platforms differ only by hardware allocation or a small control; use a zipped matrix rather than copied command
bodies. Split directories only when the workload or raw evidence set differs.

## Last-run artifacts

`emmy bench` creates a timestamped directory for every actual invocation and writes one YAML experiment record per
expanded row there. It keeps raw client/server logs and every declared command result beside those records. It never
writes legacy JSON/TXT wrappers, a task/instance manifest, or a report. Dry runs do not create a directory.

Use the repository `run-experiment` skill to finish a requested run. The skill checks matrix-row coverage, system
information, declared command-result presence, and terminal status; copies the records beside `recipe.yaml`; replaces
the Git LFS-backed `results.tar.gz`; overwrites `RESULTS.md`; and commits the complete durable snapshot. Once the
archive has been extracted or byte-checked against the raw files, the ignored timestamped directory may be deleted;
the archive is the durable raw copy. Existing records, archive, and report always describe the same most recent run.

`RESULTS.md` is an intelligent review of the last run. It reports the protocol, measurements, repeat variation,
comparisons, conclusion, limitations, system, status, and archive location, with every claim grounded in the raw
files. Recipes and repository scripts cannot contain result interpretation, post-processing, or report generation;
command blocks are only the measured workload.

## Lifetime

Keep configurations that reproduce a published comparison, support a durable qualification, or are needed for a
planned measurement. Delete exploratory configurations and one-off analysis helpers after their conclusion is encoded
in a final recipe, golden configuration, test, or architecture note.

See [`recipes/ARCHITECTURE.md`](../recipes/ARCHITECTURE.md) for the serving-recipe boundary and
[`emmy/recipe/ARCHITECTURE.md`](../emmy/recipe/ARCHITECTURE.md) for the YAML format.
