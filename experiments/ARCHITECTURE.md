# experiments/ — reproducible measurements grouped by model

An experiment answers a comparison or qualification question. It uses the recipe format and runs through
`emmy bench`; the recommended serving configuration belongs in `recipes/`.

## Directory convention

```text
experiments/<model>/<workload_or_question>/
  recipe.yaml
  golden/                                   # optional pre-tuned goldens the recipe replays
  <YYYY-MM-DD_HH-MM-SS>/                    # temporary ignored raw output
  results_<gpu-short>x<gpu-count>.tar.gz    # one Git LFS archive per exact platform, including row records
  RESULTS.md                                # one interpretation across all platforms
```

Use the model's established repository slug and a short `snake_case` experiment name. Derive `<gpu-short>` with
`emmy.hardware.gpu_short_name`; `results_rtx4090x1.tar.gz` is the archive for one RTX 4090. Keep one protocol in one
recipe when platforms differ only by hardware allocation or a small control; use a zipped matrix rather than copied
command bodies. Split directories only when the workload or raw evidence set differs.

## Pre-tuned goldens as a recipe input

An experiment that measures tuned Emmy does not tune: searching a schedule is judgment work owned by the
`tune-kernels` skill, and a recipe that scripted it would encode that judgment in the harness. Instead the skill
produces the golden files, they are committed under `golden/`, and the recipe replays them. The recipe still owns the
program definition — a checked-in snippet or trace input the skill reads — so the tuned program and the benched
program cannot drift apart.

The committed files remain search state, so the measuring lane re-measures every schedule they pin and its own
records stay the experiment's evidence. They are per-card and are retuned when the platform changes, which is why a
compiler change is re-measured by rerunning the recipe alone.

Give each lane and each measured operator (or other workload split) its own matrix parameter so `--filter` can
re-measure one slice. Command rows for one GPU share a single execution group and therefore a single VM, so splitting
a long sweep into rows costs staging, not hardware.

## Last-run artifacts

`emmy bench` creates a timestamped directory for every actual invocation and writes one YAML experiment record per
expanded row there. It keeps raw client/server logs and every declared command result beside those records. It never
writes legacy JSON/TXT wrappers, a task/instance manifest, or a report. Dry runs do not create a directory.

Use the repository `run-experiment` skill to finish a requested run. The skill checks matrix-row coverage, system
information, declared command-result presence, and terminal status; retains the platform's records inside the raw-run
tree; replaces its Git LFS-backed named archive; updates its section in `RESULTS.md`; and commits the complete durable
snapshot. Once the archive has been extracted or byte-checked against the raw files, the ignored timestamped directory
may be deleted; the archive is the durable raw copy. Each platform's records inside that archive and its report section
describe the same most recent run. Updating one platform preserves every other platform snapshot.

`RESULTS.md` is an intelligent review across the retained platform runs. Each platform section reports the protocol,
measurements, repeat variation, comparisons, conclusion, limitations, system, status, and archive location, with every
claim grounded in its raw files. Recipes and repository scripts cannot contain result interpretation, post-processing,
or report generation; command blocks are only the measured workload.

## Lifetime

Keep configurations that reproduce a published comparison, support a durable qualification, or are needed for a
planned measurement. Delete exploratory configurations and one-off analysis helpers after their conclusion is encoded
in a final recipe, golden configuration, test, or architecture note.

See [`recipes/ARCHITECTURE.md`](../recipes/ARCHITECTURE.md) for the serving-recipe boundary and
[`emmy/recipe/ARCHITECTURE.md`](../emmy/recipe/ARCHITECTURE.md) for the YAML format.
