---
name: run-experiment
description: >-
  Run or rerun Emmy experiment recipes, including requests to adjust an experiment harness before running it, then
  preserve per-platform compressed raw results with system-only YAML experiment records and a thoughtful cumulative
  RESULTS.md interpretation. Use for requests such as "run this experiment", "benchmark this recipe", "rerun this on
  a GPU", or "customize and execute this Emmy experiment". Interpret results through intelligent review, never
  repository code.
---

# Run Experiment

Produce one durable snapshot for each requested exact GPU platform. Use Emmy for execution. Preserve measurements as
raw evidence and review them yourself; do not add a result-conversion, plotting, manifest, analysis, or
report-generation script.

## Prepare

1. Read the repository `AGENTS.md`, `experiments/ARCHITECTURE.md`, the selected `recipe.yaml`, and any protocol note in
   the experiment directory.
2. Confirm the exact experiment directories and hardware source from the request. Use an existing host when supplied;
   otherwise use the normal Emmy cloud allocation.
3. Create a feature branch from `main` when the checkout is still on `main`. Preserve unrelated worktree changes.
4. If the request changes the harness, make only the requested recipe or measurement-script edits. Validate expanded
   rows with `emmy bench ... --dry-run`; a dry run must not change prior results.

## Run

Run the selected directories in one Emmy invocation per exact GPU name/count when practical. Separate invocations
keep each platform archive self-contained:

```bash
./venv/bin/emmy bench experiments/<model>/<experiment> [host and filter flags]
```

Stay with the run until every selected row reaches a terminal state. Do not hide failed rows or rerun only failures
unless the user requests that change. Emmy creates one `<experiment>/<YYYY-MM-DD_HH-MM-SS>/` directory per invocation
and writes one `*.experiment.yaml` per expanded matrix row alongside raw logs and declared command results. Keep this
timestamped directory locally for inspection; it remains ignored by Git.

For `--no-teardown`, clean up with `emmy teardown <experiment-dir>` after evidence collection unless the user asked to
retain the machine. Verify the records were updated after cleanup.

## Assemble the durable snapshot

For each selected experiment directory and exact GPU name/count:

1. Load `recipe.yaml` and every `<timestamp>/*.experiment.yaml` from the run just completed. Verify that records cover
   the expected filtered rows, use one run ID, parse as YAML, contain generic system information, and have a terminal
   `succeeded` or `failed` status. Treat a missing row as a run failure.
2. Check declared command results for presence and scan records and raw files for secrets. Read the raw measurements,
   compare the intended lanes, calculate only quantities needed for a clear interpretation, and inspect repeat
   stability, failures, correctness evidence, and protocol limitations. Do this as intelligent review, not with code
   added to the experiment recipe or repository.
3. Derive the platform key as `<gpu-short>x<gpu-count>` with `emmy.hardware.gpu_short_name`, for example `rtx4090x1`.
   Preserve the latest system-only records inside the timestamped raw directory exactly as Emmy produced them. Do not
   copy them beside `recipe.yaml`.
4. Replace `<experiment>/results_<platform-key>.tar.gz` with a gzip-compressed tar archive whose root member is the
   latest timestamped directory, including its system-only records. Verify that archive contains every expected row
   record. Remove legacy top-level `<platform-key>*.experiment.yaml` files only after this verification. Keep the local
   directory through archive extraction or byte verification and never delete another platform's archive. If the
   caller requires an artifact-only checkout, delete the task-owned local directory only after that archive
   verification. Track `experiments/**/results_*.tar.gz` with Git LFS.
5. Update the current platform section in `<experiment>/RESULTS.md` with a thoughtful, evidence-backed interpretation.
   Preserve other platform sections. Include the question, protocol, result summary, repeat variation, comparisons,
   conclusion, limitations, timestamp, run ID, machine and software information, row status, failures, archive path,
   and member names. Distinguish direct comparisons from directional ones and avoid claims the harness does not
   support.
6. Ensure the durable experiment contains `recipe.yaml`, one cumulative `RESULTS.md`, and one named archive for every
   retained platform. Each archive contains that platform's matching records. The current platform's archive and
   report section must describe the same most recent run; do not modify another platform's snapshot.

## Verify and commit

Run proportionate recipe tests plus `make lint`; run `make test` when harness code changed. List and test the archive,
verify its Git LFS attribute, ensure no secret appears in records, logs, filenames, or `RESULTS.md`, and trace every
reported value and conclusion back to the raw files and protocol.

Force-stage only the requested experiments' durable snapshot because `experiments/` is ignored by default:

```bash
git check-attr filter -- experiments/<model>/<experiment>/results_<gpu-short>x<gpu-count>.tar.gz
git add -f -A -- experiments/<model>/<experiment>/recipe.yaml \
  experiments/<model>/<experiment>/results_<gpu-short>x<gpu-count>.tar.gz \
  experiments/<model>/<experiment>/RESULTS.md
```

When the platform update removes tracked legacy top-level records, stage those exact deleted paths separately with
`git add -u -- <paths>`.

The archive must report `filter: lfs`. If the repository pattern is missing and the caller permits infrastructure
changes, add it with `git lfs track "experiments/**/results_*.tar.gz"` and stage `.gitattributes`. If the caller says
LFS is configured locally, do not modify or list `.gitattributes`; the caller owns that file.

Stage harness edits normally, review the staged diff, and commit once with a concise subject such as
`Record <experiment> run`. Never stage the timestamped raw directory. Do not push or open a pull request unless the
user requested submission or repository instructions require it. Report the commit, failed rows, retained
infrastructure, local run directory, durable snapshot paths, and the main result with its strongest limitation.
