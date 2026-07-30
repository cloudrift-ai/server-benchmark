---
name: reproduce-article-benchmarks
description: Use this skill when the user asks to "re-run the benchmarks from the article", "reproduce the blog post numbers", "validate that <article url> still holds", "check the latest code still performs like the published post", or otherwise wants a published article's benchmarks re-measured with emmy and compared against what was published. Works for any article — it reads the article itself, finds the experiments/recipes in this repo that back it, reports whether the code has moved since any pinned docker image was published (and offers to rebuild), asks where the hardware comes from (auto-provisioned cloud VM, a pre-allocated remote host, or a local GPU), runs the benchmarks, and reports the deltas.
version: 0.1.0
---

# Reproduce a published article's benchmarks

A published article's benchmarks already live in this repo as experiments. This skill re-runs them and diffs the
result against the article — it does not re-derive the benchmarks and it does not invent new ones.

**Read the article at run time. Never work from a transcribed copy of its numbers.** No baseline files, no cached
tables, no numbers memorized into this skill. A snapshot of an article diverges from it silently, and a stale
reference number invents regressions that were never there. The article is the reference; fetch it every run.

The deliverable is one comparison: measured vs published, per cell, with the setup that produced it stated plainly.
**A regression is a finding to hand back, never something to diagnose or fix inside the run** — a repro session's
entire value is that it is a clean, uncontaminated measurement.

## Step 1 — Read the article

`WebFetch` the URL and extract, explicitly:

- every results table, with its **exact numbers, units, and column meaning** (tok/s vs ms; mean vs median vs p99);
- the **lanes** each table compares (stock engine / emmy / a precision fork / a third-party engine / a speculation
  depth) — these become the run's control lanes;
- the **workload points** (input/output lengths, concurrency, batch sizes, kernel shapes) and the client settings;
- the **hardware and stack** it was measured on (GPU, driver, CUDA, torch, engine version);
- any **methodology caveat the article states about itself** — "compare rows of this table only with each other",
  "this grid ran on a different machine", "this cell failed and here is why". These are load-bearing; a caveat
  ignored turns a correct result into a false regression.

If a table's meaning is ambiguous after reading, say so rather than guessing — an assumed column definition is the
easiest way to manufacture a fake delta.

## Step 2 — Find what backs it in this repo

Recipes cite the article they back in their header comment, but **not by URL** — expect to search on the title, the
model or kernel name, and topic keywords, and expect the mapping to be fuzzy:

```bash
grep -rl "<title fragment>" experiments/ recipes/ scripts/ tests/ --include='*.yaml' --include='*.py' --include='*.md'
grep -rli "<model or kernel name>" experiments/ scripts/            # topic fallback
ls experiments/*/                                 # experiments are experiments/<model>/<name>/recipe.yaml
head -50 experiments/<model>/<name>/recipe.yaml   # the header states its lanes, knobs, protocol, repro command
```

Not every article is backed by an `experiments/` recipe — a kernel-level article may be backed by a `scripts/`
benchmark helper, a `tests/perf` case, or a golden-set filter instead. Follow the same rule wherever it lands: find
the thing that claims the article, and read it.

**Also look for prior canonical runs.** An experiment directory often keeps committed result directories plus a
`RESULTS.md` / `report.md` naming which run backs which of the article's tables, and recording the stack it ran on
(driver, CUDA, library versions). That is a **first-party reference and usually the better one**: it shares the
harness, the field names and the recipe with your run, where the article's prose has been through a rewrite. Use
both — the article for what was claimed publicly, the prior run for a like-for-like diff.

Read every candidate header in full before running anything. They are the durable record of *why* each recipe is
shaped the way it is — which knobs are the article's equal-tuning protocol, which cells are expected to fail, which
metric is the honest one to read. That context is what keeps the comparison honest.

**If the mapping is uncertain, confirm it with the user before spending GPU hours.** Show which recipes you believe
back which tables and let them correct you; a wrong mapping produces a confidently-wrong regression report.

Then reconcile the two sides and **say what you found**:

- tables with a backing recipe → runnable;
- tables with **no** backing recipe → not runnable as-is. Say so. Do not improvise a substitute benchmark and
  present its numbers against a published table; an approximation of the article is not a reproduction of it.
- recipe cells with **no** published counterpart → runnable, but report them as unmatched. Recipes deliberately
  carry diagnostic points the article's tables omit.

If nothing in the repo backs the article, stop and report that. Building the experiment is a separate task.

## Step 3 — Classify each recipe: does it test current code, or an image?

This decides whether the run answers the user's question.

- **Command recipes** (a `command:` block with a `stage:` list) copy `emmy/`, `scripts/` etc. from the **local
  checkout** to the host and build a venv there. They measure the **current working tree**. No image, no drift.
- **Deploy recipes** (`model:` + `engine:`) pull a **docker image** and measure whatever code is inside it.

State the split. It often makes the whole image question moot for what the user actually wants to know.

## Step 4 — Drift report (whenever a deploy recipe is in scope)

Emmy image tags embed the emmy commit they were built from — `<repo>:<engine-version>-<short-sha>`; see the
`Makefile` tag variables. Both drifts are therefore mechanical, and you report **both**:

**(a) Recipe pin vs latest published tag** — what the recipes pull, vs what has been released:

```bash
grep -rho 'cloudriftai/[^"]*' <the recipes in scope> | sort -u
curl -s "https://hub.docker.com/v2/repositories/<org>/<repo>/tags?page_size=25&ordering=last_updated" \
  | python3 -c "import json,sys; [print(t['name'], t['digest'][:19], t['last_updated']) for t in json.load(sys.stdin)['results']]"
```

A pin behind the newest tag is a **recipe bug** — report it, offer to bump it as its own one-line PR rather than
folding it into the run. Compare **digests**, not names: `latest` is usually the same digest as the newest
`<ver>-<sha>` tag, and pinning the sha tag is always correct over pinning `latest`. A repo that 404s is private or
was never pushed — say that plainly instead of assuming the tag is fine.

**(b) Published image vs current code** — the tag suffix is a commit:

```bash
git fetch origin main
git log --oneline "<tag sha>..origin/main" -- emmy/ docker/ scripts/ | cat
```

Report it concretely — "the image was baked at `<sha>` (<date>); `origin/main` is N commits ahead, M touching
`emmy/`" — then state what it means, because this is usually the actual question:

> A deploy recipe run on the published image measures the **image's** code, not HEAD. It answers "does the released
> artifact still perform as published" — valid, and the cheap answer. It does **not** answer "does current main
> still perform as published". Only a locally built image answers that.

**When (b) is non-empty and a deploy recipe is in scope, this is a decision point, not a note.** Either the image
matches the code under test, or the user tells you to proceed anyway knowing what is being measured. Present both
and ask; never start a rebuild unprompted (it is a multi-hour GPU session), and never quietly run the stale image
and let the report imply it covered current code.

## Step 5 — Image selection

Three paths; default to the first.

1. **Published tag (default).** Run as pinned, after any bump from Step 4a. Fastest and reproducible.
2. **User-supplied reference.** Verify it exists (`docker manifest inspect <ref>`) before provisioning anything.
3. **Build locally** — the path for testing unreleased or uncommitted changes. Delegate to the repo's release skill
   for that image (`release-serving-image`, parameterized by `MODEL`) and run it **through its verify gate, stopping before
   `docker login` / the push**; the local tag `make` produces is what the recipes then reference on that host. State
   the cost up front (hours of wall time, tens of GB of disk) and note it wants the **same host** the benchmarks
   will run on. Its gates apply unchanged — a preflight or parity failure aborts the session and is itself the
   finding. Two things to record either way:
   - a locally built image is not pushed, so the run is **not reproducible elsewhere** until someone releases it;
   - `git rev-parse` tags the image with the **committed** sha even on a dirty tree, so capture `git status --short`
     or the tag misrepresents what was measured.

A prebuilt image is warmed at one serving shape and one kernel fork; lanes outside that set fall back to compiling
on boot. Check the image's `ARCHITECTURE.md` for what its warm actually covers, and treat any gap as **boot cost,
not numerics** — the kernels are the same, the first boot is just slow, and it can exceed a healthcheck window.

**Overriding an image without editing the repo.** Recipes hardcode `engine.llm.vllm.image` and `emmy bench` has no
override flag. Copy the experiment to the scratch dir and rewrite it there, so the checkout stays clean:

```bash
cp -r experiments/<model>/<suite> "$SCRATCH/<suite>"
sed -i 's|<pinned ref>|<new ref>|g' "$SCRATCH/<suite>/recipe.yaml"
grep image: "$SCRATCH/<suite>/recipe.yaml"       # confirm the substitution took
./venv/bin/emmy bench "$SCRATCH/<suite>" --local
```

**Rewrite only the emmy lanes.** Baseline lanes (stock vLLM, SGLang, llama.cpp) are the run's control and stay
exactly as pinned — swapping a control converts a comparison into a single-lane measurement. And a silently no-op
`sed` means you benchmark the old image and report the delta as a code regression, so always confirm.

## Step 6 — Hardware

Ask where the hardware comes from; each recipe pins `deploy.gpu` and `bench` aborts up front if no host satisfies a
group.

- **Auto-provision** (default when the user has cloud creds and no card in hand) — plain `emmy bench <recipe>`
  provisions, deploys, benches, tears down. Use `--no-teardown` only if they want to inspect afterwards, and then
  remember `emmy teardown <run_dir>`: a forgotten GPU VM bills indefinitely.
- **A pre-allocated remote host** — `emmy bench <recipe> --ssh user@host` (repeatable). Fixed hosts are never
  deleted by the run. The `start-remote-server` skill covers renting one to manage by hand.
- **A local GPU** — `emmy bench <recipe> --local`. This is what each recipe header's "Repro on a pre-allocated card"
  line shows.

Confirm the host actually carries what the recipes pin, and name what that excludes:

```bash
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader     # locally or over ssh
```

Articles commonly span several GPUs. If only one is available, run its recipes and **report the rest as not-run** —
never substitute a different card and present the numbers against the article's table for another one. Command
recipes also need host toolchain: `nvcc` on `PATH` at a version the target arch requires, `python3.12-venv`, a
system-wide `ninja`, and `cmake` where a lane builds a third-party engine from source.

Before committing the user to a run, **estimate and state the cost** — recipe `timeout:` values and the variant
count (`emmy bench <recipe> --dry-run`) give the order of magnitude. These are multi-hour, billable runs.

## Step 7 — Run

Launch each recipe **detached** and poll; a multi-hour run must survive an SSH drop:

```bash
nohup ./venv/bin/emmy bench <recipe> --local > "$SCRATCH/<suite>.log" 2>&1 &
```

Safeguards:

- **One recipe at a time per GPU.** Concurrent runs contend for the card and both sets of numbers become
  meaningless.
- **Cap the wall clock** at roughly double the estimate. On overrun: capture the log tail, stop, tear down, report
  how far it got.
- **Do not "fix" the expected failures** the recipe headers and the article call out. A cell that fails by design is
  a passing run.
- **Never edit a recipe to improve a number.** The article's numbers were measured under that recipe's protocol, and
  the protocol *is* the experiment. Changing a knob mid-run produces a number that compares to nothing.
- **Keep partial results.** If a run dies, a partial table plus an honest "died at cell X, log attached" beats
  burning another six hours.

## Step 8 — Report

Build the comparison from the run's own outputs against the tables you read in Step 1, and against any prior
canonical run found in Step 2. Each `*_benchmark.json` carries both the recipe and the metrics, so every measured
cell identifies its own lane and workload — derive the lane from the recipe (image, env knobs, speculative config),
never from a filename. Command recipes leave whatever their tool writes; read it directly. Show measured,
published, and the delta per cell.

Where a prior committed run exists, diff against it too and show both columns. It is the like-for-like comparison,
and when the two disagree — matching the prior run but not the article, or vice versa — that disagreement is itself
the finding, and worth more than either number alone.

Judge deltas honestly:

- **Lead with the setup**, because it determines what the numbers mean: which image (published tag / local build),
  which commit, which host and card, which recipes ran and which did not.
- **Have a noise floor and state it.** Run-to-run and box-to-box spread is real; a small delta is not a finding.
  Where a recipe runs repeats, its own stddev is the best available estimate. Where the article states an error bar
  (an eval's standard error, say), use the article's.
- **Compare lanes only within one run on one box**, and honor whatever the article says about its own tables — a
  table published from a different machine compares internally, against its own baseline rows, not across boxes.
- **A flagged cell is a lead, not a conclusion.** Check the run's own control: did the *baseline* lane move too? A
  control lane shares the harness, client and box but none of emmy's kernels, so a control moving alongside emmy
  points at the box/driver/client, while a steady control beside a moving emmy localises the fault to emmy. A driver
  or engine-version change since publication does the same thing — check the article's stated stack against the
  host's before blaming the code.
- **List every deviation from the article's procedure**, with what forced it — an API or CLI change, a knob that no
  longer exists, a lane that wouldn't boot, a host stack that differs from the published one. State which
  comparisons each deviation weakens. A best-effort reproduction is a legitimate result; an undisclosed one is not.
- **Say what you did not measure.** Recipes skipped, cells that failed, tables with no backing recipe, a
  command-recipe-only run that never touched the image question. Silence reads as coverage.

## Common pitfalls

**A docker-backed experiment silently measuring old code.** This is the one that quietly ruins a report. A deploy
recipe runs whatever is inside its image, so if the image predates the code you were asked about, the run answers a
different question than the one asked — and the report reads as if it didn't. Before running any deploy recipe,
establish that the image corresponds to the code under test, or **get an explicit decision from the user** to
proceed with a known-stale image. Then carry that decision into the report: name the image, its sha, and how far
`origin/main` has moved past it, on the same line as the numbers. "vLLM+Emmy hit X tok/s" is not a claim you can
make without saying which Emmy.

**Assuming the recipe still runs.** These recipes were written against the emmy of their day. Flags get renamed,
commands get restructured, env knobs get retired, defaults move. Dry-run first (`emmy bench <recipe> --dry-run`)
and read the failure honestly — a recipe that no longer loads is a finding about the repo, not a reason to guess.

**Refusing to deviate when the API has moved.** If emmy's CLI, API, or behavior has changed such that the article's
exact procedure is no longer expressible, **a best-effort reproduction is the right call** — do not abandon the run,
and do not silently fake the old path. Translate the intent to what current emmy actually supports, keeping the
article's protocol (same workload points, same lanes, same equal-tuning discipline) wherever you still can. Then
say so, precisely: what changed, what you ran instead, and which comparisons that weakens. A deviation stated
plainly is a usable result; a deviation buried in a table is a false regression waiting to be quoted.

**Treating a deviation as a small footnote.** If the substitution touches something load-bearing — a different
knob, a different metric, a lane you couldn't reproduce at all — that belongs in the report's summary, not only in
the row. The reader's question is "can I trust this number against the published one", and only you know the answer.

**Reproducing an article whose stack no longer exists.** The article pins a driver, CUDA, torch and engine version.
If the host or the current repo has moved past them, the delta includes that move. Check the article's stated stack
against the host's before attributing anything to emmy — and if they differ materially, say the comparison is
against a different stack rather than presenting it as a clean A/B.

**Editing recipes in place.** Recipes are the protocol. Override images and knobs in a scratch copy (Step 5), never
in the checkout, and never to make a number look better.

## Teardown

Always, including on every abort path: `emmy teardown <run_dir>` for anything left by `--no-teardown`,
`docker logout` if a build logged in, and `emmy vm delete` for anything this session provisioned. Report rental time.
