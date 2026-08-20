# Non-Interactive Model Qualification

Use the attached onboarding task as the complete run request. Its fields are the only source of the model, hardware,
credentials path, and deadline; never select, substitute, or infer any of them. The attached exact-workflow-SHA copies
of the `onboard-model`, `tune-kernels`, and `run-experiment` skills are authoritative — follow them instead of any
older skill copy in the checkout. Do not ask follow-up questions; a missing or ambiguous task field is an immediate
failure.

## Task fields

| Field | Use |
| --- | --- |
| `mode` | exactly `onboarding` or `verification`; drives the recipe policy below and is echoed in the summary |
| `model_id` | the exact Hugging Face model ID to qualify |
| `gpu` / `gpu_count` | the exact target platform; never change the GPU name, count, quantization, or checkpoint |
| `ssh_target`, `ssh_host`, `ssh_user`, `ssh_port` | the supplied server; the only host this run may use |
| `ssh_key` | pass `--ssh-key <value>` to every Emmy remote command |
| `deadline` | absolute wall-clock deadline for the whole run |
| `multimodal_mode` | `auto`, `multimodal`, or `text-only` qualification path |
| `publish_image` | `true` authorizes publishing a verified prebuilt Emmy image; `false` forbids publication |
| `summary_path` | absolute path for the atomic machine-readable summary, outside the repository |
| `expected_lifecycle` | in `verification` mode, the lifecycle tag the refreshed recipe must keep |

The task's `publish_image` value is the publication authorization for this run. Do not add a conversational approval
pause, and do not publish when it is `false`.

## Recipe policy per mode

In `onboarding` mode, use the existing `onboarding`/`untested` recipe shell when present, or create the recipe for a
direct manual request. Preserve `model.heat` and replace `onboarding` and `untested` with `best-effort` only after
qualification and artifact completion.

In `verification` mode, begin from the existing recipe's current configuration, refresh its measured artifacts, and
preserve its `expected_lifecycle` tag and `model.heat`.

## Boundaries

Do not select a model or GPU, provision or delete the VM, commit, push, or open or update a pull request. The caller
owns those steps. Tear down every deployed workload before returning.

For a missing image or an unfamiliar launch failure, investigate current official registries, release notes, engine
documentation, and upstream issues. Test an evidence-backed current repository or tag when the configured image moved
or disappeared, then pin the exact working tag or digest. You may implement a bounded, model-agnostic compatibility
fix with focused tests when it is necessary for this exact qualification.

Delegate only bounded, independent read-only research or failure diagnosis to the `onboard-investigator` subagent,
giving it the complete contents of the attached `investigate.md` and exactly one question. Retain responsibility for
every edit, command, measurement, and conclusion.

## Repository artifacts

Allowed areas are `recipes/` (including the model's `golden/` subdirectory), `experiments/`,
`docker/vllm-emmy-serve/models/`, and a bounded small fix under `emmy/` with its focused tests and nearest
`ARCHITECTURE.md` updates.

Every platform shares one serving experiment root, `experiments/<model>/serving/`; reuse it when it exists and
never create a platform-suffixed root. The platform appears in the archive filename, not in a directory name. The
summary's `experiment` field is that root's `recipe.yaml` path, never a directory.

Replace only this platform's `results_<gpu-short>x<gpu-count>.tar.gz` archive and preserve every other platform
archive and `RESULTS.md` section. The archive must contain the platform's system-only experiment records; do not
retain those records as top-level files. Update the final recipe's `RESULTS.md` for this platform while preserving
still-valid measurements for its other platforms. Do not retain the ignored dated run directory or qualification
summaries in the checkout.

Git LFS is already configured through the caller's local attributes. Verify that the named archive reports
`filter: lfs`, but do not run `git lfs track` and do not modify or list `.gitattributes` in the summary.

## Output

Always write the skill's atomic summary to `summary_path`, on success and on failure, with `mode` set to the task's
mode. List every intended created, modified, or deleted repository file in `artifacts` and no exploratory output.
Include `experiment_artifacts` with the shared experiment recipe and `RESULTS.md` plus the exact platform archive, and
one-line `deployment_summary` and `performance_summary` values drawn from the exact selected recipe lane.

On failure, set `failure.regression` to `true` only when a previously qualified behavior or measured performance lane
regressed and the bounded fix attempt could not restore it. Keep the message concise and credential-free; the caller
sends it to a chat notification. Print the summary path as the final line and return nonzero for a failed run.
