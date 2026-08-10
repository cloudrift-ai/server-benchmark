---
name: onboard-model
description: >-
  Onboard and benchmark a Hugging Face model on an exact target GPU platform. Use when asked to add a model recipe,
  validate a new model on a supplied GPU server, benchmark serving, create reproducible experiments and a durable
  results report, fully qualify and tune the model's Emmy compiler inventory even when serving is blocked, or publish
  a prebuilt CloudRiftAI serving image.
---

# Onboard a model

Turn a Hugging Face model ID and an exact `(GPU name, GPU count)` into reviewed repository artifacts:

- one recommended serving recipe under `recipes/<model>/recipe.yaml`;
- reproducible qualification or comparison recipes under `experiments/<model>/` with their measured results;
- `recipes/<model>/RESULTS.md`, which records the serving evidence behind the recipe;
- a complete compiler golden under `emmy/compiler/pipeline/search/goldens/`, or an explicitly partial compiler report
  under `experiments/<model>/compiler_<hardware-slug>/`, even when serving cannot be delivered;
- when Emmy is eligible, tuned kernels and a verified, prebuilt
  `cloudriftai/vllm-emmy-<model-slug>:<tag>` image.

Use only the supplied SSH server. The caller owns VM creation and deletion; this skill owns deployed workloads and
must tear them down before returning. Never switch GPU type, count, provider, model quantization, or model checkpoint
to rescue a failed run.

## Required inputs

Require:

1. the exact Hugging Face model ID;
2. the exact hardware name used by `emmy/hardware.py` and GPU count;
3. an SSH target accepted by `emmy deploy ssh` / `emmy bench --ssh`;
4. a wall-clock deadline;
5. whether publication is authorized and, when needed, Docker Hub credentials supplied without logging them;
6. whether a multimodal model should be qualified as multimodal, text-only, or `auto` (infer the advertised modality
   from the checkpoint and qualify that path);
7. for a non-interactive run, an absolute output path for the machine-readable summary.

In an interactive run, ask only for missing inputs. In a non-interactive run, fail immediately on a missing or
ambiguous input. The workflow-dispatch request is the publication authorization in CI: do not add a conversational
approval gate, but do not publish unless the input explicitly authorizes it.

When the caller supplies an environment file, source it in the same shell invocation as each `emmy` command. Source
`.env` first only when it exists, then the explicitly named overlay. In CI, use the injected process environment when
no overlay was supplied; never guess an overlay file. Check whether required variables are present without printing
their values, and never echo a token.

Before changing files, verify that the model exists on Hugging Face, the target host reports exactly the requested
GPU name and count, and the checkout is on a feature branch. If `recipes/<model>/` already exists, use it as a
starting point only when the caller explicitly allowed updates; otherwise fail instead of overwriting it.

## 1. Research the serving path

Use the model card, `config.json`, and current primary engine documentation to record:

- architecture, modality, total and active parameter counts, dtype and quantization;
- the immutable Hugging Face commit resolved for this run;
- native context length and any documented practical cap;
- current vLLM or SGLang support and the first pinned image version that supports it;
- required tool-call, reasoning, tokenizer, and multimodal flags;
- known model-specific launch or correctness issues.

Prefer vLLM when both engines support the checkpoint. Use a moving image tag only for diagnosis, then pin the exact
working tag or digest. Read `emmy/recipe/ARCHITECTURE.md` before authoring YAML; named recipe fields must not be
duplicated in `extra_args`.

## 2. Create and validate a conservative baseline

Create a provisional experiment recipe under
`experiments/<model>/serving_<hardware-slug>/recipe.yaml`. Start with the smallest tensor-parallel size that fits the
weights on the requested GPU count, conservative memory utilization and concurrency, and a short benchmark. Keep
the target matrix exact.

Deploy before benchmarking:

```bash
emmy deploy ssh --recipe experiments/<model>/serving_<hardware-slug> --ssh <target>
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
programs and target identities; never merge by kernel name alone. Record the full layer-to-program mapping in the
compiler report. A representative expert is valid only where Emmy deliberately keeps routing/sort/combine in host
orchestration; do not stub an unsupported GPU operation merely to claim coverage.

Call the inventory **complete** only when every manifest path emits a non-empty post-fusion target set and every
retained target reconstructs and lowers for the exact compute capability. Otherwise call it **partial**, preserve all
successful targets, and record the first unsupported operation and exact reproducer for every uncovered path. Fix
tractable compiler gaps with focused regressions, then retrace before accepting the coverage result.

Use `tune-kernels` to run its equal-budget model-proposal-versus-MCTS workflow over every traceable target, followed
by repeated O3 correctness and finalist checks. For complete coverage, finish one deployable O3 measurement and one
positive reference-backend measurement for every retained target. When search finds no acceptable winner, measure a
correct greedy fallback instead of dropping the target. Strip working `ranking` metadata and commit the resulting
self-contained, fully verified document to the usual location:

`emmy/compiler/pipeline/search/goldens/<gpu-slug>_<compute-cap>_<model-slug>.yaml`

That complete model golden is required even when serving, checkpoint loading, or full-model accuracy is blocked. It
must contain every traced target, explicit knobs (an empty mapping is valid for a forkless anchor), paired positive
`emmy_us` / reference timings, the exact GPU identity and compute capability, and the immutable model ID as provenance.
Validate it with repository-golden validation and lower every entry again from the committed file.

Do not put a partially traced model in the repository golden directory. When any manifest path cannot be traced,
commit only these ignored experiment artifacts:

- `experiments/<model>/compiler_<hardware-slug>/partial_traced_working.yaml`, preserving every successful target for
  future compiler work without presenting it as complete deploy evidence;
- `experiments/<model>/compiler_<hardware-slug>/RESULTS.md` with the coverage manifest, tuning measurements, exact
  compiler gaps, and the reason canonical promotion was withheld.

If no path emits a target, commit the compiler report but do not invent an empty file. Trace incompleteness is the
only model-coverage reason to omit the usual golden YAML; tuning failures must fall back to a correct measured
configuration, and serving failures do not block golden publication.

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

Record `eligible` or `ineligible` plus the first failed gate in `RESULTS.md`. An ineligible model still receives a
mainstream-engine experiment, report, serving recipe, and the compiler artifacts from section 3. Never create an
empty Emmy serving lane or imply that a stock-only qualification is an Emmy comparison.

## 5. Release when eligible

After the required compiler qualification, invoke the `release-serving-image` workflow on the same supplied server
and follow `docker/vllm-emmy-serve/ARCHITECTURE.md`. The CI dispatch's explicit publication input satisfies the human
approval pause described by that workflow; skip only that conversational pause. Keep every mechanical gate: golden
coverage, toolchain preflight, headroom sweep, HF parity, warm convergence, offline zero-recompile verification, and
image push. A failed gate means no push.

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

Commit the experiment recipe and the raw successful result files that support the report. Delete exploratory sweeps,
failed run directories, caches, credentials, and unrelated logs. Fold the measured winner into a single-variant
serving recipe under `recipes/<model>/recipe.yaml`; recipes have no `benchmark:` block.

The repository ignores `experiments/` by default. Force-add only the intended new experiment recipe and successful
result files during submission; never force-add the whole directory or exploratory output.

## 7. Write the durable report

Create `recipes/<model>/RESULTS.md` beside the recipe. Use measurements only; never fill gaps with estimates. Include:

- date, repository revision, model revision, GPU name/count, driver, CUDA, and pinned image tags or digests;
- exact workload and raw experiment-result paths;
- validated context, modality, tool-call and reasoning-parser results;
- request/output throughput, TTFT, TPOT or ITL, failure count, and benchmark duration;
- Emmy eligibility and the evidence for the decision;
- complete/partial/none compiler coverage, the repository-golden or partial-inventory and compiler-report paths,
  tuned target counts, O3 verification, and every remaining compiler gap; link the usual repository golden when
  coverage is complete, otherwise link only the partial working inventory;
- for an Emmy run, equal-workload mainstream-versus-Emmy results, kernel-tuning summary, and published image tag;
- for a stock-only run, the qualification result with no Emmy comparison;
- the recommended serving configuration, limitations, and any unresolved upstream issue.

Keep the report useful without the working directory. Link only committed experiment artifacts. The recipe is the
decision; the report and experiment are its evidence.

## 8. Verify and hand off

Before reporting success:

```bash
emmy bench --dry-run experiments/<model>/serving_<hardware-slug>
emmy deploy ssh --dry-run --recipe recipes/<model> --ssh <target>
```

Also verify:

- the compiler coverage manifest accounts for every layer and non-layer seam; a complete trace has a repository
  golden whose every entry has paired positive O3/reference measurements and reconstructs and lowers on the requested
  compute capability, while a partial trace has no file under the repository golden directory;
- every reported tuning winner identifies its O1 ranking lane, and only repeated O3 rows are described as deployable;
- at least one successful serving result exists and every reported number points to it;
- the recipe and experiment pin immutable engine images;
- the recipe targets exactly the requested GPU name/count;
- the published Emmy image, when applicable, passed offline zero-recompile verification;
- tracked artifacts contain no credentials, absolute scratch paths, or VM identifiers;
- deployed workloads are torn down and `docker logout` has run.

In the summary, `cleanup.docker_logout: true` means no Docker credential remains: either logout completed after a
login, or no Docker login was performed because the stock-only path did not require one.

Write this JSON object atomically to the caller-supplied summary path and print that path as the final line:

List every repository file created, modified, or deleted by the onboarding run in `artifacts`. List the committed
compiler report plus the repository golden (complete coverage) or partial working inventory in `compiler_artifacts`,
and the serving recipe and every successful raw result in `experiment_artifacts`; the caller uses these manifests to
reject unrequested worktree changes and force-add only the intended ignored experiment files.

```json
{
  "status": "success",
  "model_id": "org/model",
  "target": {"gpu": "exact hardware.py name", "gpu_count": 1, "ssh": "user@host"},
  "recipe": "recipes/<model>/recipe.yaml",
  "experiment": "experiments/<model>/serving_<hardware-slug>/recipe.yaml",
  "artifacts": [
    "recipes/<model>/recipe.yaml",
    "recipes/<model>/RESULTS.md",
    "emmy/compiler/pipeline/search/goldens/<gpu-slug>_<compute-cap>_<model-slug>.yaml",
    "experiments/<model>/compiler_<hardware-slug>/RESULTS.md",
    "experiments/<model>/serving_<hardware-slug>/recipe.yaml",
    "experiments/<model>/serving_<hardware-slug>/<run>/result.json"
  ],
  "compiler_artifacts": [
    "emmy/compiler/pipeline/search/goldens/<gpu-slug>_<compute-cap>_<model-slug>.yaml",
    "experiments/<model>/compiler_<hardware-slug>/RESULTS.md"
  ],
  "experiment_artifacts": [
    "experiments/<model>/serving_<hardware-slug>/recipe.yaml",
    "experiments/<model>/serving_<hardware-slug>/<run>/result.json"
  ],
  "report": "recipes/<model>/RESULTS.md",
  "compiler": {
    "coverage": "complete",
    "golden": "emmy/compiler/pipeline/search/goldens/<gpu-slug>_<compute-cap>_<model-slug>.yaml",
    "report": "experiments/<model>/compiler_<hardware-slug>/RESULTS.md",
    "traced_targets": 42,
    "tuned_targets": 42,
    "blocked_paths": []
  },
  "emmy": {"eligible": true, "reason": "all eligibility gates passed", "image": "cloudriftai/...:tag"},
  "cleanup": {"workloads": "complete", "docker_logout": true},
  "failure": null
}
```

Use `status: "failed"`, nullable serving artifact fields, and a `failure` object with `gate` and `message` on failure.
Keep successful compiler fields and artifacts populated even when the failure is a serving gate. Always write the
summary, then return nonzero for a failed run. Do not delete the VM. The caller uses the SSH target and its separately
captured provider/instance handle to perform and verify VM cleanup.

On any failure, preserve useful tracked candidates and untracked logs, tear down workloads, report the first failed
gate and artifact paths, and return nonzero. Never claim partial onboarding as success.
