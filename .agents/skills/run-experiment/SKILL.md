---
name: run-experiment
description: >-
  Run or rerun Emmy experiment recipes, including requests to adjust an experiment harness before running it, then
  preserve the latest raw results, YAML experiment records, and a human-readable RESULTS.md as committed repository
  artifacts. Use for requests such as "run this experiment", "benchmark this recipe", "rerun the comparison on this
  GPU", or "customize and execute this Emmy experiment".
---

# Run Experiment

Produce one durable snapshot of the last requested run. Use Emmy for execution and use judgment for the report; do not
add a result-conversion, plotting, manifest, or report-generation script.

## Prepare

1. Read the repository `AGENTS.md`, `experiments/ARCHITECTURE.md`, the selected `recipe.yaml`, and any protocol note in
   the experiment directory.
2. Confirm the exact experiment directories and hardware source from the request. Use an existing host when supplied;
   otherwise use the normal Emmy cloud allocation.
3. Create a feature branch from `main` when the checkout is still on `main`. Preserve unrelated worktree changes.
4. If the request changes the harness, make only those recipe or measurement-script edits needed for the requested
   comparison. Validate the expanded rows with `emmy bench ... --dry-run`; a dry run must not change prior results.

## Run

Run the selected directories in one Emmy invocation when practical:

```bash
./venv/bin/emmy bench experiments/<model>/<experiment> [host and filter flags]
```

Stay with the run until every selected row reaches a terminal state. Do not hide failed rows or rerun only failures
unless the user requests that change. Emmy replaces `<experiment>/results/` and writes one
`results/*.experiment.yaml` per expanded matrix row alongside raw logs and declared command artifacts.

For `--no-teardown`, clean up with `emmy teardown <experiment-dir>` after evidence collection unless the user asked to
retain the machine. Verify the records were updated after cleanup.

## Assemble the durable snapshot

For each selected experiment directory:

1. Load `recipe.yaml` and every `results/*.experiment.yaml`. Verify that the records cover the expected filtered matrix
   rows, use one run ID, parse as YAML, and have a terminal `succeeded` or `failed` status. Treat a missing row as a run
   failure.
2. Inspect every record and relevant raw artifact. Do not infer success from file presence, and do not discard partial
   evidence from failed rows.
3. Remove only the prior top-level `*.experiment.yaml` files for that exact experiment. Move the new record files from
   `results/` into the experiment directory; their artifact paths remain rooted at `results/`.
4. Overwrite `<experiment>/RESULTS.md` with a free-form, human-readable account of the latest run. State the timestamp,
   run ID, machine, protocol, result rows, failures, important raw-artifact references, and conclusions justified by
   the evidence. Prefer a compact table where rows share comparable metrics. Clearly distinguish observation from
   interpretation.
5. Ensure the final experiment contains `recipe.yaml`, the raw `results/` folder, the top-level experiment records,
   and `RESULTS.md`. Do not retain records or reports from an earlier run.

## Verify and commit

Run proportionate recipe tests plus `make lint`; run `make test` when harness code changed. Check each record's artifact
paths after moving it and ensure no secret appears in records, logs, or the report.

Force-stage only the requested experiments' durable snapshot because `experiments/` is ignored by default:

```bash
git add -f experiments/<model>/<experiment>/results \
  experiments/<model>/<experiment>/*.experiment.yaml \
  experiments/<model>/<experiment>/RESULTS.md
```

Stage harness edits normally, review the staged diff, and commit once with a concise subject such as
`Record <experiment> results`. Do not push or open a pull request unless the user requested submission or repository
instructions require it. Report the commit, all failed rows, retained infrastructure, and the durable artifact paths.
