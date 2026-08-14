---
name: run-experiment
description: >-
  Run or rerun Emmy experiment recipes, including requests to adjust an experiment harness before running it, then
  preserve the latest compressed raw results, system-only YAML experiment records, and a thoughtful RESULTS.md
  interpretation. Use for requests such as "run this experiment", "benchmark this recipe", "rerun this on a GPU", or
  "customize and execute this Emmy experiment". Interpret results through intelligent review, never repository code.
---

# Run Experiment

Produce one durable snapshot of the last requested run. Use Emmy for execution. Preserve measurements as raw evidence
and review them yourself.

## Boundary

- Emmy, experiment recipes, and repository scripts MUST NOT interpret experiment measurements or assemble
  human-readable reports.
- Experiment records MUST NOT contain measurements, comparisons, conclusions, or workload output. They contain only
  row identity, execution lifecycle, Git provenance, and generic system information.
- Raw result files MUST NOT be rewritten or normalized.
- Review the raw evidence yourself and write the thoughtful interpretation in `RESULTS.md`.

## Prepare

1. Read the repository `CLAUDE.md`, `experiments/ARCHITECTURE.md`, the selected `recipe.yaml`, and any protocol note in
   the experiment directory.
2. Confirm the exact experiment directories and hardware source from the request. Use an existing host when supplied;
   otherwise use the normal Emmy cloud allocation.
3. Create a feature branch from `main` when the checkout is still on `main`. Preserve unrelated worktree changes.
4. If the request changes the harness, make only the requested recipe or measurement-script edits. Validate expanded
   rows with `emmy bench ... --dry-run`; a dry run must not change prior results.

## Run

Run the selected directories in one Emmy invocation when practical:

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

For each selected experiment directory:

1. Load `recipe.yaml` and every `<timestamp>/*.experiment.yaml` from the run just completed. Verify that records cover
   the expected filtered rows, use one run ID, parse as YAML, contain generic system information, and have a terminal
   `succeeded` or `failed` status. Treat a missing row as a run failure.
2. Check declared command results for presence and scan records and raw files for secrets. Read the raw measurements,
   compare the intended lanes, calculate only quantities needed for a clear interpretation, and inspect repeat
   stability, failures, correctness evidence, and protocol limitations.
3. Remove only the prior top-level `*.experiment.yaml` files for that exact experiment. Copy the latest records beside
   `recipe.yaml`, preserving the timestamped local directory exactly as Emmy produced it.
4. Replace `<experiment>/results.tar.gz` with a gzip-compressed tar archive whose root member is the latest timestamped
   directory. Do not delete that local directory. Track `experiments/**/results.tar.gz` with Git LFS.
5. Overwrite `<experiment>/RESULTS.md` with a thoughtful, evidence-backed interpretation. Include the question,
   protocol, result summary, repeat variation, comparisons, conclusion, limitations, timestamp, run ID, machine and
   software information, row status, failures, archive path, and member names. Distinguish direct comparisons from
   directional ones and avoid claims the harness does not support.
6. Ensure the durable experiment contains `recipe.yaml`, `results.tar.gz`, top-level experiment records, and
   `RESULTS.md`. Do not retain durable records, archives, or reports from an earlier run.

## Verify and commit

Run proportionate recipe tests plus `make lint`; run `make test` when harness code changed. List and test the archive,
verify its Git LFS attribute, ensure no secret appears in records, logs, filenames, or `RESULTS.md`, and trace every
reported value and conclusion back to the raw files and protocol.

Force-stage only the requested experiments' durable snapshot because `experiments/` is ignored by default:

```bash
git lfs track "experiments/**/results.tar.gz"
git add .gitattributes
git add -f experiments/<model>/<experiment>/results.tar.gz \
  experiments/<model>/<experiment>/*.experiment.yaml \
  experiments/<model>/<experiment>/RESULTS.md
```

Stage harness edits normally, review the staged diff, and commit once with a concise subject such as
`Record <experiment> run`. Never stage the timestamped raw directory. Do not push or open a pull request unless the
user requested submission or repository instructions require it. Report the commit, failed rows, retained
infrastructure, local run directory, durable snapshot paths, and the main result with its strongest limitation.
