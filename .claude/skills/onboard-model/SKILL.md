---
name: onboard-model
description: >-
  Onboard and benchmark a Hugging Face model on an exact target GPU platform. Use when asked to add a model recipe,
  validate a new model on a supplied GPU server, benchmark serving, create reproducible experiments and a durable
  results report, or optionally tune Emmy kernels and publish a prebuilt CloudRiftAI serving image.
---

# Onboard a model

Turn a Hugging Face model ID and an exact `(GPU name, GPU count)` into reviewed repository artifacts:

- one recommended serving recipe under `recipes/<model>/recipe.yaml`;
- reproducible qualification or comparison recipes under `experiments/<model>/` with their measured results;
- `recipes/<model>/RESULTS.md`, which records the serving evidence behind the recipe;
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

## 3. Decide Emmy eligibility

Do not infer Emmy support from a similar model. Mark the checkpoint eligible only when every gate passes on the
requested hardware and the exact checkpoint quantization:

1. the live compute capability is accepted by Emmy's CUDA backend;
2. the architecture has a real trace/runner path in the current checkout;
3. the checkpoint quantization is handled by the matching Emmy loader and serving path, with its stored representation
   preserved in the deployed graph; reference-only dequantization does not establish deployment support;
4. `emmy trace <model> --golden-output <working.yaml>` emits a non-empty post-fusion kernel inventory for the chosen
   trace profile;
5. representative kernel correctness succeeds, and `emmy serve --generate` or the applicable embedding path can
   serve the checkpoint.

Record `eligible` or `ineligible` plus the first failed gate in `RESULTS.md`. An ineligible model still receives a
mainstream-engine experiment, report, and serving recipe. Never create an empty Emmy lane or imply that a stock-only
qualification is an Emmy comparison.

## 4. Tune and release when eligible

Invoke the `tune-kernels` skill with the model ID, exact target GPU/count, supplied SSH target, trace profile, and the
remaining deadline. Let it create the working golden when none exists, propose candidates from same-GPU and related
canonical goldens, and run the equal-budget hybrid-versus-MCTS comparison. Only reviewed, repeated O3 measurements
may update a canonical golden.

After tuning, invoke the `release-serving-image` workflow on the same supplied server and follow
`docker/vllm-emmy-serve/ARCHITECTURE.md`. The CI dispatch's explicit publication input satisfies the human approval
pause described by that workflow; skip only that conversational pause. Keep every mechanical gate: golden coverage,
toolchain preflight, headroom sweep, HF parity, warm convergence, offline zero-recompile verification, and image
push. A failed gate means no push.

Use the repository slug helper for both config and image name. Publish only the verified tag under
`cloudriftai/vllm-emmy-<model-slug>:<tag>`, update the recipe to that exact tag, pass credentials through
`docker login --password-stdin`, and run `docker logout` on success and failure.

## 5. Measure the final serving configuration

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

## 6. Write the durable report

Create `recipes/<model>/RESULTS.md` beside the recipe. Use measurements only; never fill gaps with estimates. Include:

- date, repository revision, model revision, GPU name/count, driver, CUDA, and pinned image tags or digests;
- exact workload and raw experiment-result paths;
- validated context, modality, tool-call and reasoning-parser results;
- request/output throughput, TTFT, TPOT or ITL, failure count, and benchmark duration;
- Emmy eligibility and the evidence for the decision;
- for an Emmy run, equal-workload mainstream-versus-Emmy results, kernel-tuning summary, and published image tag;
- for a stock-only run, the qualification result with no Emmy comparison;
- the recommended serving configuration, limitations, and any unresolved upstream issue.

Keep the report useful without the working directory. Link only committed experiment artifacts. The recipe is the
decision; the report and experiment are its evidence.

## 7. Verify and hand off

Before reporting success:

```bash
emmy bench --dry-run experiments/<model>/serving_<hardware-slug>
emmy deploy ssh --dry-run --recipe recipes/<model> --ssh <target>
```

Also verify:

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
experiment recipe and every successful raw result again in `experiment_artifacts`; the caller uses these manifests
to reject unrequested worktree changes and force-add only the intended ignored experiment files.

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
    "experiments/<model>/serving_<hardware-slug>/recipe.yaml",
    "experiments/<model>/serving_<hardware-slug>/<run>/result.json"
  ],
  "experiment_artifacts": [
    "experiments/<model>/serving_<hardware-slug>/recipe.yaml",
    "experiments/<model>/serving_<hardware-slug>/<run>/result.json"
  ],
  "report": "recipes/<model>/RESULTS.md",
  "emmy": {"eligible": true, "reason": "all eligibility gates passed", "image": "cloudriftai/...:tag"},
  "cleanup": {"workloads": "complete", "docker_logout": true},
  "failure": null
}
```

Use `status: "failed"`, nullable artifact fields, and a `failure` object with `gate` and `message` on failure. Always
write the summary, then return nonzero for a failed run. Do not delete the VM. The caller uses the SSH target and its
separately captured provider/instance handle to perform and verify VM cleanup.

On any failure, preserve useful tracked candidates and untracked logs, tear down workloads, report the first failed
gate and artifact paths, and return nonzero. Never claim partial onboarding as success.
