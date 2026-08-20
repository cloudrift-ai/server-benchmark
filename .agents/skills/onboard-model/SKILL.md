---
name: onboard-model
description: >-
  Onboard or periodically reverify and benchmark a Hugging Face model on an exact target GPU platform. Use when asked
  to add a model recipe, refresh a maintained recipe on a supplied GPU server, benchmark serving, create reproducible
  experiments and a durable results report, fully qualify and tune the model's Emmy compiler inventory even when
  serving is blocked, or publish a prebuilt CloudRiftAI serving image.
---

# Onboard or reverify a model

Turn a Hugging Face model ID, an operation mode, and an exact `(GPU name, GPU count)` into reviewed repository
artifacts:

- one recommended serving recipe under `recipes/<model>/recipe.yaml`;
- one reusable serving experiment under `experiments/<model>/` with a cumulative report and one Git LFS evidence
  archive, including the system-only row records, per exact GPU platform;
- one compact, self-contained `recipes/<model>/RESULTS.md` only when a valid final deployment recipe exists;
- a complete compiler golden under `recipes/<model>/golden/` when full coverage qualifies;
- when Emmy is eligible, tuned kernels and a verified, prebuilt
  `cloudriftai/vllm-emmy-<model-slug>:<tag>` image.

Repository storage retains reproducibility input and durable evidence for each qualified serving platform. In the
serving experiment root, commit `recipe.yaml`, one cumulative `RESULTS.md`, and
`results_<gpu-short>x<gpu-count>.tar.gz` for each measured exact GPU name/count. Store the system-only experiment
records inside that platform archive as part of the timestamped raw run; do not commit them as top-level files. Derive
`<gpu-short>` with `emmy.hardware.gpu_short_name`; for example, a single RTX 4090 uses
`results_rtx4090x1.tar.gz`. Track these archives
with Git LFS. When the caller says LFS is configured locally, verify the archive attribute but do not modify or list
`.gitattributes`; the caller owns that infrastructure file. Do not commit the ignored dated run directory, loose
benchmark JSON/TXT/logs, plots, compiler run summaries, partial working goldens, or onboarding-summary files.

Use one serving experiment root for all GPU platforms that share the protocol. On a platform-specific run, replace
only that platform's archive, update only its section of the shared experiment `RESULTS.md`, and preserve every other
platform archive and report section. Remove legacy top-level records only for the current platform after verifying
they are retained inside its archive. Reuse the existing serving experiment root when it already represents the
protocol; otherwise create `experiments/<model>/serving/`.

Use only the supplied SSH server. The caller owns VM creation and deletion; this skill owns deployed workloads and
must tear them down before returning. Never switch GPU type, count, provider, model quantization, or model checkpoint
to rescue a failed run.

## Required inputs

Require:

1. an operation mode, exactly `onboarding` or `verification`;
2. the exact Hugging Face model ID;
3. the exact hardware name used by `emmy/hardware.py` and GPU count;
4. an SSH target accepted by `emmy deploy ssh` / `emmy bench --ssh`;
5. a wall-clock deadline;
6. whether publication is authorized and, when needed, Docker Hub credentials supplied without logging them;
7. whether a multimodal model should be qualified as multimodal, text-only, or `auto` (infer the advertised modality
   from the checkpoint and qualify that path);
8. for a non-interactive run, an absolute output path for the machine-readable summary.

In an interactive run, ask only for missing inputs. In a non-interactive run, fail immediately on a missing or
ambiguous input. The workflow-dispatch request is the publication authorization in CI: do not add a conversational
approval gate, but do not publish unless the input explicitly authorizes it.

When the caller supplies an environment file, source it in the same shell invocation as each `emmy` command. Source
`.env` first only when it exists, then the explicitly named overlay. In CI, use the injected process environment when
no overlay was supplied; never guess an overlay file. Check whether required variables are present without printing
their values, and never echo a token.

Before changing files, verify that the model exists on Hugging Face, the target host reports exactly the requested
GPU name and count, and the checkout is on a feature branch. Model and GPU selection, VM rental, VM cleanup, git, and
pull-request operations belong to the caller, not this skill.

In `onboarding` mode, use the existing `onboarding`/`untested` recipe shell when present; a direct caller may instead
authorize a new recipe. Preserve the discovery-managed `model.heat` value and replace the pending tags with
`best-effort` only after qualification and artifact completion. In `verification` mode, begin with the existing active
recipe, refresh its measurements and durable artifacts, and preserve its supplied lifecycle tag and `model.heat`.
The mode explicitly authorizes updates to that recipe; never use verification to silently change its model checkpoint,
target GPU/count, lifecycle, or heat. For any other existing recipe update, require explicit caller authorization or
fail instead of overwriting it.

## 1. Research the serving path

Use the model card, `config.json`, and current primary engine documentation to record:

- architecture, modality, total and active parameter counts, dtype and quantization;
- the VRAM footprint on the requested platform, following [`prompts/model-fit.md`](../../../prompts/model-fit.md) —
  the shared fit contract `discover-models` also follows, so the platform a recipe proposes and the platform measured
  here mean the same thing. Record the arithmetic it requires; when the weights cannot fit the requested GPU name and
  count, report the failed fit gate with those numbers instead of deploying;
- the immutable Hugging Face commit resolved for this run;
- native context length and any documented practical cap;
- current vLLM or SGLang support and the first pinned image version that supports it;
- required tool-call, reasoning, tokenizer, and multimodal flags;
- known model-specific launch or correctness issues.

Prefer vLLM when both engines support the checkpoint. If a configured image cannot be pulled, do not stop at the first
missing tag. Check the official registry, release notes, engine documentation, and upstream repository for a renamed
image repository or a current compatible tag. A registry manifest check or short clean deployment may establish the
replacement. Use a moving image tag only for diagnosis, then pin the exact working tag or digest. Never substitute an
unofficial image without explicit caller authorization. Read `emmy/recipe/ARCHITECTURE.md` before authoring YAML;
named recipe fields must not be duplicated in `extra_args`.

## 2. Create and validate a conservative baseline

Create or update the shared experiment recipe under the model's serving experiment root. Start with the smallest
tensor-parallel size that fits the weights on the requested GPU count, conservative memory utilization and
concurrency, and a short benchmark. Keep the current target matrix row exact and preserve the other platform rows.

Deploy before benchmarking:

```bash
emmy deploy ssh --recipe experiments/<model>/serving --ssh <target>
```

Require all of these before measuring performance:

- the image and weights load and the server reaches health;
- a real chat or completion request returns coherent output;
- advertised tool calling returns structured `tool_calls`;
- advertised reasoning is separated into the engine's reasoning field;
- requested multimodal behavior is exercised, or multimodal inputs are explicitly disabled for a text-only run;
- the largest claimed context is tested with an input that materially fills it.

Start context testing at the model's native maximum. On an out-of-memory or invalid-capacity failure, halve it and
retry from a clean deployment. Do not claim a context length from startup alone. Search current upstream issues for
an unfamiliar error before changing engine or flags, and make one evidence-backed change per retry.

When useful, delegate up to three independent read-only investigations to the caller-provided onboarding subagent.
Good tasks are model protocol research, official engine/image compatibility research, and diagnosis of one unfamiliar
failure from already captured logs. The parent agent owns all commands, edits, measurements, and final conclusions;
never let subagent work consume the time reserved for artifacts and cleanup.

You may implement a small, model-agnostic compatibility fix when it is necessary to qualify the exact requested
checkpoint and platform. Keep it within `emmy/` plus focused tests and the nearest `ARCHITECTURE.md` updates, prefer an
existing mechanism, and validate the reproducer before resuming qualification. Do not perform a broad refactor,
dependency upgrade, validation bypass, workflow change, or model-specific hard-code. If the fix is not bounded, does
not validate, or cannot finish before the deadline, preserve diagnostics outside the repository and report the gate
as failed.

Use `emmy ... --teardown` after every failed deployment and before changing a serving configuration. Do not mutate
the remote checkout or containers manually over SSH; read-only inspection of `nvidia-smi`, container state, and logs
is allowed.

## 3. Fully qualify the compiler inventory

Invoke the `tune-kernels` skill on the supplied host for every model, even when memory fit, engine support,
checkpoint representation, or another serving gate has already failed. Compiler qualification is an independent,
durable deliverable and must not be skipped because the serving stack is ineligible.

First build a coverage manifest from the immutable configuration and instantiated architecture. It must map every
layer index and non-layer seam to a traced representative, including:

- token or input embeddings, final normalization, and the output head;
- every attention type, sliding/global pattern, compressor, or recurrent path;
- dense MLP, routed-expert, shared-expert, and MTP paths;
- layer-local gates, residual layouts, and materially different checkpoint storage or quantization layouts.

Prefer one architecture-only whole-model trace. When that is not bounded for a very large checkpoint, trace each
distinct path separately and merge the self-contained programs into one working golden. Deduplicate only identical
programs and target identities; never merge by kernel name alone. Keep the full layer-to-program mapping in the
caller-supplied external summary and put only coverage counts and gaps in the final recipe report. A representative
expert is valid only where Emmy deliberately keeps routing/sort/combine in host orchestration; do not stub an
unsupported GPU operation merely to claim coverage.

Call the inventory **complete** only when every manifest path emits a non-empty post-fusion target set and every
retained target reconstructs and lowers for the exact compute capability. Otherwise call it **partial**, preserve all
successful targets, and record the first unsupported operation and exact reproducer for every uncovered path. Fix
tractable compiler gaps with focused regressions, then retrace before accepting the coverage result.

Use `tune-kernels` to run its equal-budget model-proposal-versus-MCTS workflow over every traceable target, followed
by repeated O3 correctness and finalist checks. For complete coverage, finish one deployable O3 measurement and one
positive reference-backend measurement for every retained target. When search finds no acceptable winner, measure a
correct greedy fallback instead of dropping the target. Strip working `ranking` metadata and commit the resulting
self-contained, fully verified document beside the model's recipe:

`recipes/<model>/golden/<gpu-slug>_<compute-cap>.yaml`

Keep exactly one file per exact GPU model and compute capability. If the model has no recipe directory yet, create an
`onboarding`/`untested` recipe shell first, preserving the discovery-managed lifecycle fields when they are known.

That complete model golden is required even when serving, checkpoint loading, or full-model accuracy is blocked. It
must contain every traced target, explicit knobs (an empty mapping is valid for a forkless anchor), paired positive
`emmy_us` / reference timings, the exact GPU identity and compute capability, and the immutable model ID as provenance.
Validate it with repository-golden validation and lower every entry again from the committed file.

Do not put a partially traced model in the recipe's `golden/` directory. Preserve a partial working inventory and its
diagnostics only in the caller-supplied external output or an ignored task directory; do not commit them or create an
experiment `RESULTS.md` unless the caller explicitly requests that exact evidence. If no path emits a target, do not
invent an empty file. Trace incompleteness is the only model-coverage reason to omit the usual golden YAML; tuning
failures must fall back to a correct measured configuration, and serving failures do not block a complete golden.

## 4. Decide Emmy eligibility

Do not infer Emmy support from a similar model. Mark the checkpoint eligible only when every gate passes on the
requested hardware and the exact checkpoint quantization:

1. the live compute capability is accepted by Emmy's CUDA backend;
2. the architecture has a real trace/runner path in the current checkout;
3. the checkpoint quantization is handled by the matching Emmy loader and serving path, with its stored representation
   preserved in the deployed graph; reference-only dequantization does not establish deployment support;
4. the compiler qualification above has complete architecture coverage and its working golden reconstructs and
   lowers on the target compute capability;
5. representative kernel correctness succeeds, and `emmy serve --generate` or the applicable embedding path can
   serve the checkpoint.

Record `eligible` or `ineligible` plus the first failed gate in the final recipe report. An ineligible model may still
receive a mainstream-engine experiment configuration, serving recipe, and recipe report when that engine qualifies.
When no engine can produce a valid final recipe, create no repository report; put the failure in the external summary.
Never create an empty Emmy serving lane or imply that a mainstream-only qualification is an Emmy comparison.

## 5. Release when eligible

Treat `EMMY_FAST_MATH=1` as the default candidate for every Emmy-backed final recipe. Before release, compare it with
standard Emmy on the exact checkpoint, hardware, serving shape, capability probes, and a checkpoint-appropriate
accuracy suite with a predeclared tolerance. Select FAST_MATH unless a correctness, capability, or quality regression
is observed; if one is observed, retain standard Emmy and record the failed accuracy gate in the recipe report.
FAST_MATH failure does not by itself change Emmy eligibility. Mainstream-only vLLM or SGLang recipes must not set
`EMMY_FAST_MATH`.

After the required compiler qualification, invoke the `release-serving-image` workflow on the same supplied server
and follow `docker/vllm-emmy-serve/ARCHITECTURE.md`. The CI dispatch's explicit publication input satisfies the human
approval pause described by that workflow; skip only that conversational pause. Keep every mechanical gate: golden
coverage, toolchain preflight, headroom sweep, HF parity, warm convergence, offline zero-recompile verification, and
image push. A failed gate means no push.

When FAST_MATH is selected, its exact serving shape is a required release lane: warm it to convergence, verify the
FAST_MATH execution-plan pack is a hit with offline zero recompilation, and set
`engine.llm.vllm.extra_env.EMMY_FAST_MATH: "1"` in the final recipe. A missing or invalid FAST_MATH pack blocks release
of a FAST_MATH recipe even if the standard pack succeeds.

Use the repository slug helper for both config and image name. Publish only the verified tag under
`cloudriftai/vllm-emmy-<model-slug>:<tag>`, update the recipe to that exact tag, pass credentials through
`docker login --password-stdin`, and run `docker logout` on success and failure.

## 6. Measure the final serving configuration

Keep each request-producing benchmark variant within 20 minutes. Setup and model loading have a separate 30-minute
cap. Stop a benchmark that exceeds its cap, halve request count or concurrency, tear down, and retry. Keep at least
15 minutes of the caller's deadline reserved for artifact collection and cleanup; do not start a stage that cannot
finish within the remaining budget.

When Emmy is eligible and the image was verified, the final experiment must compare on the same model, target GPU
count, workload, context, request count, client concurrency, warm-up, and precision policy:

- the pinned mainstream vLLM or SGLang image;
- the pinned prebuilt Emmy image.

When Emmy is ineligible, run only the pinned mainstream lane. The experiment may contain a context/concurrency grid
needed to justify the recommended recipe, but it must not contain a dummy Emmy lane.

Use comparison lanes to select and explain the recommended configuration. Benchmark the precision lane selected in
section 5. Include the comparisons that matter for this model's decision and clearly identify the engine and
configuration selected by `recipes/<model>/recipe.yaml`; do not force unrelated models into one table layout.

Commit the canonical experiment `recipe.yaml` when the comparison configuration remains useful. Follow the
repository's `run-experiment` skill for the final measurement. Store the run as
`results_<gpu-short>x<gpu-count>.tar.gz`, including the system-only experiment records inside the timestamped raw run,
and update its section in the shared factual experiment `RESULTS.md`, including failed-row evidence. Remove
current-platform legacy top-level records only after verifying the archive, and preserve every other platform
snapshot. Fold the measured winner into a single-variant serving recipe under
`recipes/<model>/recipe.yaml`; recipes have no `benchmark:` block.

## 7. Write the durable report

Create `recipes/<model>/RESULTS.md` only beside a valid recommended recipe. If serving qualification cannot produce a
valid recipe, create no `RESULTS.md` in the repository; the caller-supplied external summary is the failure record.
Creating or updating a valid serving recipe without creating or refreshing its `RESULTS.md` is incomplete.

Do not expect `emmy bench`, its experiment record, or the experiment `RESULTS.md` artifact index to interpret the run.
For the separate recommended-recipe report, inspect complete server logs, failures, and declared raw artifacts.
Reconcile them with the recipe matrix and the model's protocol before deciding what the evidence supports. Assemble
the recipe report directly; do not add a result-analysis script.

Before writing performance numbers, find a successful experiment row and raw artifacts for the exact selected recipe
configuration: model revision, engine image tag or digest, GPU name/count, precision policy, context, concurrency,
workload, and engine knobs must all match. Re-run that lane with the existing `emmy bench` experiment when evidence
is missing, stale, or does not identify the selected engine. Update the current platform section of the recipe report
from the new measurement while preserving still-valid sections for other platforms, but do not copy measurements
into the experiment record or experiment artifact index.
Never estimate a missing value, copy a competing engine's result, or combine metrics from different runs.

Choose the report structure that makes this model's evidence easy to understand. A dense model, a quantized model, a
multimodal model, and an Emmy compiler qualification may need different sections, metrics, and comparisons. There is
no required heading order or universal table. Prefer a compact narrative and only the tables that clarify the result.
Use measurements only, compare complete runs without cherry-picking, and include the relevant subset of:

- date, repository revision, model revision, GPU name/count, driver, CUDA, and pinned image tags or digests;
- exact workload used for the selected measurement;
- validated context, modality, tool-call and reasoning-parser results;
- request/output throughput, TTFT, TPOT or ITL, failure count, and benchmark duration;
- Emmy eligibility and the evidence for the decision;
- complete/partial/none compiler coverage, tuned target counts, O3 verification, and every remaining compiler gap;
  link the repository golden only when coverage is complete;
- the selected recipe engine's performance lane, identified by the exact image and relevant engine knobs, plus any
  comparison needed to justify why it was selected;
- one clean `emmy bench experiments/<model>/<name> ...` reproduction command using a retained experiment recipe;
  filter a comparison recipe to the selected engine and precision lane;
- for an Emmy recipe, its serving result, kernel-tuning summary, published image tag, and the accuracy result that
  authorized FAST_MATH or the quality regression that retained standard Emmy;
- for a vLLM or SGLang recipe without Emmy, that engine's serving result without a comparison column;
- the recommended serving configuration, limitations, and any unresolved upstream issue.

Keep the recipe report useful without the working directory. The recipe is the decision and its report embeds the
selected lane's compact, best qualified measurements. The experiment directory separately retains reproducibility
input, per-platform archives containing raw results and per-row system-only experiment records, and its factual
last-run artifact index.

## 8. Verify and hand off

Before reporting success:

```bash
emmy bench --dry-run experiments/<model>/serving
emmy deploy ssh --dry-run --recipe recipes/<model> --ssh <target>
```

Also verify:

- the compiler coverage manifest accounts for every layer and non-layer seam; a complete trace has a repository
  golden whose every entry has paired positive O3/reference measurements and reconstructs and lowers on the requested
  compute capability, while a partial trace has no file under the recipe's `golden/` directory;
- nightly verification is the checked-in golden gate: repository-validate and strictly decode every realization in
  every `recipes/<model>/golden/*.yaml` file for the model, then run the file-scoped audit and exact pinned replay for
  the file matching this job's supplied GPU; nightly orchestration must schedule every recipe/GPU golden pair;
- every reported tuning winner identifies its O1 ranking lane, and only repeated O3 rows are described as deployable;
- when a serving recipe exists, at least one successful result for that exact recipe lane exists and the report embeds
  only numbers from that complete run;
- the recipe and experiment pin immutable engine images;
- the recipe targets exactly the requested GPU name/count;
- the published Emmy image, when applicable, passed offline zero-recompile verification, and a FAST_MATH recipe has a
  verified FAST_MATH pack for the exact serving shape;
- `EMMY_FAST_MATH=1` appears in an Emmy recipe unless its accuracy gate regressed, and never appears in a
  mainstream-only recipe;
- the experiment snapshot contains the shared `recipe.yaml` and `RESULTS.md` plus the exact platform's LFS archive;
  the archive contains the platform's records, no current-platform record remains at the experiment root, and other
  platform archives and report sections remain unchanged;
- no ignored dated run directory, loose benchmark output, or onboarding summary is staged;
- tracked artifacts contain no credentials, absolute scratch paths, or VM identifiers;
- deployed workloads are torn down and `docker logout` has run.

In the summary, `cleanup.docker_logout: true` means no Docker credential remains: either logout completed after a
login, or no Docker login was performed because the stock-only path did not require one.

For a successful run, include one-line `deployment_summary` and `performance_summary` values of at most 1000
characters each. The deployment summary names the selected engine/image and the important serving configuration.
The performance summary reports the selected recipe lane's measured workload and key throughput/latency/failure
numbers. Use only measurements from the same exact successful lane that supports the final recipe; never estimate or
copy numbers from a competing lane. These values are used in the workflow notification and do not replace either
durable `RESULTS.md` report.

Write this JSON object atomically to the caller-supplied summary path outside the repository and print that path as the
final line. Include the requested mode on success and failure. List every repository file created, modified, or deleted
by the qualification run in `artifacts`. List only a complete repository golden in `compiler_artifacts`. In
`experiment_artifacts`, list only the shared experiment recipe and report plus the current platform's compressed
archive. List deleted legacy records for the current platform in `artifacts`, but not in `experiment_artifacts`. List
any bounded implementation fix and its focused tests in `artifacts`. Do not list unchanged artifacts from other
platforms or the ignored dated run directory.

```json
{
  "status": "success",
  "mode": "onboarding",
  "model_id": "org/model",
  "target": {"gpu": "exact hardware.py name", "gpu_count": 1, "ssh": "user@host"},
  "deployment_summary": "vLLM 0.22.1, 32K context, concurrency 8",
  "performance_summary": "100 requests, 2,400 output tok/s, p50 TTFT 42 ms, p50 TPOT 7.1 ms, 0 failures",
  "recipe": "recipes/<model>/recipe.yaml",
  "experiment": "experiments/<model>/serving/recipe.yaml",
  "artifacts": [
    "recipes/<model>/recipe.yaml",
    "recipes/<model>/RESULTS.md",
    "recipes/<model>/golden/<gpu-slug>_<compute-cap>.yaml",
    "experiments/<model>/serving/recipe.yaml",
    "experiments/<model>/serving/RESULTS.md",
    "experiments/<model>/serving/results_<gpu-short>x<gpu-count>.tar.gz"
  ],
  "compiler_artifacts": [
    "recipes/<model>/golden/<gpu-slug>_<compute-cap>.yaml"
  ],
  "experiment_artifacts": [
    "experiments/<model>/serving/recipe.yaml",
    "experiments/<model>/serving/RESULTS.md",
    "experiments/<model>/serving/results_<gpu-short>x<gpu-count>.tar.gz"
  ],
  "report": "recipes/<model>/RESULTS.md",
  "compiler": {
    "coverage": "complete",
    "golden": "recipes/<model>/golden/<gpu-slug>_<compute-cap>.yaml",
    "traced_targets": 42,
    "tuned_targets": 42,
    "blocked_paths": []
  },
  "emmy": {"eligible": true, "reason": "all eligibility gates passed", "image": "cloudriftai/...:tag"},
  "cleanup": {"workloads": "complete", "docker_logout": true},
  "failure": null
}
```

Use `status: "failed"`, nullable serving artifact fields, `deployment_summary: null`, `performance_summary: null`,
`report: null`, and a `failure` object with `gate`, `message`, and boolean `regression` when no valid serving recipe is
produced. Set `regression: true` only when a previously qualified behavior or measured lane no longer meets its prior
contract and the bounded fix attempt could not restore it. Keep the message concise and credential-free because the
workflow sends it to the configured Discord channel as a prominent regression notice. A new model that has never
qualified is a failure, not a regression. Keep a complete golden populated if compiler qualification succeeded. Put
partial inventories and useful diagnostics in the external output location, not the repository. Always write the
summary, then return nonzero for a failed run. Do not delete the VM. The caller uses the SSH target and its separately
captured provider/instance handle to perform and verify VM cleanup.

On any failure, tear down workloads, report the first failed gate and external diagnostic paths, remove task-owned
measurement output from the checkout, and return nonzero. Never claim partial onboarding as success.
