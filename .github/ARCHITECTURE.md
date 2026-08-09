# GitHub Actions architecture

GitHub Actions owns pull-request checks, package publication, cloud experiment execution, and automated model
discovery and onboarding. Workflows that only inspect or build the repository use GitHub-hosted runners. Agent-driven
model work uses the self-hosted `agents` runner because it can exceed ordinary hosted-runner limits and needs the
tracked skills and CloudRift inference endpoint.

## Workflow overview

| Workflow | Trigger | Runner | Result |
| --- | --- | --- | --- |
| **Tests** | Pull request to `main` | GitHub-hosted | Runs Ruff, the test suite, and GitHub helper tests. |
| **Publish to PyPI** | Manual dispatch or published GitHub release | GitHub-hosted | Tests, builds, publishes to PyPI, and optionally creates the release. |
| **Run Experiment** | Authorized `/run-experiment` PR comment | GitHub-hosted | Runs selected cloud experiments and commits results to the PR branch. |
| **Onboard model** | Manual dispatch | Self-hosted `agents` | Produces measured model artifacts on an exact GPU target and updates a PR. |
| **Discover model** | Nightly schedule or manual dispatch | Self-hosted `agents` | Selects one model and opens a capped plan PR without renting a VM. |

## Pull-request checks

**Tests** installs the CI dependency set on Python 3.13. Its lint job runs Ruff check and format verification; its test
job runs `make test` plus the tests for helpers under `.github/scripts/`. Hugging Face downloads used by tests are
cached because anonymous shared-runner traffic is rate-limited. This workflow has no write permission and does not
use deployment credentials.

## Package publication

**Publish to PyPI** has two entry paths:

- Manual dispatch reads the version from `pyproject.toml`, refuses an already-tagged version, publishes to PyPI, then
  creates the matching tag and GitHub release.
- A manually published GitHub release must already have a tag matching `pyproject.toml`; the workflow validates and
  publishes that version without creating another release.

Both paths run lint and tests before building. `scripts/prepare_dist.py` stages bundled recipes and rewrites
repository-relative README links for PyPI. The build artifact moves between jobs through GitHub artifacts. PyPI uses
trusted publishing through the `pypi` environment and an OIDC token, so the repository stores no PyPI password. The
manual path creates its GitHub release only after a successful upload, preventing a failed publication from leaving a
release behind.

## Cloud experiment execution

**Run Experiment** responds only to `/run-experiment` comments on pull requests. The gate requires the commenter to
have repository write or admin access, resolves the PR head, detects the requested experiment changes, and verifies
the benchmark command with `--dry-run` before provisioning anything. It acknowledges an accepted run on the PR.

The benchmark job creates an ephemeral SSH key, configures optional GCP access, and runs the selected `emmy bench`
command with result commits enabled. CloudRift and Hugging Face credentials are present only for the benchmark step.
Long runs use renewable GitHub App credentials when committing results, so the one-hour lifetime of a token cannot
strand a six-hour experiment. The job force-adds any remaining timestamped result directories, uploads them as a
workflow artifact, and posts a formatted result comment even when benchmarking fails. A separate failure job reports
gate or benchmark failures on the PR.

Experiment selection and result formatting are owned by `.github/scripts/detect-experiments.py` and
`.github/scripts/format-results.py`. The workflow does not accept an arbitrary shell command from a PR comment; the
detector constructs the supported Emmy command from validated experiment paths and filters.

## Model discovery and onboarding

All discovery paths use the tracked `discover-models` skill and select at most one open-weight Hugging Face model
without an existing recipe for one exact GPU name and count. Discovery is read-only: the workflow checks that the
agent did not modify the checkout, then `.github/scripts/discovery_selection.py` validates the model ID, target,
rationale, and absence of an existing recipe.

The repo-owned agent runtime in `.github/scripts/cloudrift_agent.py` calls a configurable OpenAI-compatible CloudRift
endpoint. It provides bounded public-web search and fetch tools while rejecting private, link-local, and metadata
addresses. Search results, redirects, response sizes, extracted text, command output, and the final transcript are
bounded. Discovery never provisions hardware.

### Direct onboarding

**Onboard model** accepts an exact Hugging Face model ID, exact GPU name/count, multimodal qualification mode, optional
existing onboarding PR, and explicit image-publication authorization. The job has a six-hour limit and gives the
agent an earlier deadline so artifact validation and cleanup retain time.

The workflow resolves an existing labeled onboarding PR or creates a new artifact branch, provisions exactly the
requested platform through CloudRift or optional GCP, and passes the resulting SSH target to the tracked
`onboard-model` skill. If neither provider can supply the exact target, the workflow fails; it does not silently change
GPU type or count. The skill produces the recommended recipe, serving report, reproducible experiment and raw results,
plus an Emmy-tuned prebuilt image only when the architecture, quantization, trace, serving, and release gates pass.

The agent returns an atomic manifest. `.github/scripts/onboarding_artifacts.py` accepts only declared changes under the
allowed recipe, experiment, serving-image, canonical-golden, and matching plan paths. Unmanifested or exploratory
output is rejected. The workflow then commits those artifacts, updates or opens the onboarding PR, and uses renewable
GitHub App credentials for the long-running push path.

### Discovery and plan PR

**Discover model** runs nightly or by manual dispatch. It defaults to one H200, with repository variables and manual
inputs able to select another exact target. It creates no plan when three `model-onboarding` PRs are already open.
After discovery it queries open PRs again, both to reject a duplicate model and to close the race in which another
run reached the cap during discovery.

When allowed, the workflow commits one `plans/onboard-<model>.md` file on a new branch and opens a draft,
`model-onboarding`-labeled PR. It never rents a VM. Running **Onboard model** from that branch, or with the PR number,
replaces the plan with measured artifacts and removes the plan file.

## Credentials, VM ownership, and cleanup

Agent workflows transfer the CloudRift inference key through a mode-`0600`, one-use file, replace the secret-bearing
shell, and unlink the file before the first agent tool call. Agent tool subprocesses do not inherit CloudRift, GCP, or
GitHub credentials. Onboarding retains only the explicitly required Hugging Face and Docker Hub credentials. The
self-hosted runner must not carry unrelated ambient cloud credentials.

`.github/scripts/onboarding_vm.py` writes a run-owned lease as soon as CloudRift returns an instance ID or before GCP
starts the named instance. The lease binds the provider handle, exact request, repository, run attempt, and SSH target.
Cleanup accepts only that lease, retries deletion, and audits only the recorded handle; it never enumerates or deletes
unrelated VMs. Cleanup runs from both the agent shell trap and an `if: always()` workflow step, and the job fails if its
owned VM is still active.

GitHub App credentials are used for long-lived branch writes and PR operations. Private keys and temporary provider
configuration live only under run-specific `/tmp/emmy-*` paths and are removed by unconditional cleanup steps.

## Repository configuration

Agent and experiment workflows use these repository secrets as applicable:

- `CLOUDRIFT_API_KEY` for model discovery, CloudRift provisioning, and cloud experiments;
- `EXPERIMENT_APP_ID` and `EXPERIMENT_APP_PRIVATE_KEY` for renewable Git and pull-request credentials;
- `HF_TOKEN` for gated checkpoints;
- `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` for an eligible verified prebuilt image;
- optional `GCP_SERVICE_ACCOUNT_KEY` and `GCP_SERVICE_ACCOUNT` for GCP capacity.

`ONBOARD_AGENT_MODEL` selects the discovery/onboarding model and defaults to `Qwen/Qwen3.6-35B-A3B-FP8`.
`CLOUDRIFT_INFERENCE_URL` selects its OpenAI-compatible endpoint and defaults to
`https://inference.cloudrift.ai/v1`. Scheduled discovery defaults can be changed with `DISCOVERY_TARGET_GPU` and
`DISCOVERY_TARGET_GPU_COUNT`.
