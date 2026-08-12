# GitHub Actions architecture

GitHub Actions owns pull-request checks, package publication, cloud experiment execution, and automated model
discovery and onboarding. Workflows that only inspect or build the repository use GitHub-hosted runners. Agent-driven
model work uses the self-hosted `agents` runner because it can exceed ordinary hosted-runner limits and needs the
tracked skills and CloudRift inference endpoint.

## Workflow overview

| Workflow | Trigger | Runner | Result |
| --- | --- | --- | --- |
| **Tests** | Pull request to `main` | GitHub-hosted | Runs Ruff and the complete test suite. |
| **Publish to PyPI** | Manual dispatch or published GitHub release | GitHub-hosted | Tests, builds, publishes to PyPI, and optionally creates the release. |
| **Run Experiment** | Authorized `/run-experiment` PR comment | GitHub-hosted | Runs selected cloud experiments and commits results to the PR branch. |
| **Onboard model** | Manual dispatch | Self-hosted `agents` | Produces measured model artifacts on an exact GPU target and updates a PR. |
| **Discover model** | Nightly schedule or manual dispatch | Self-hosted `agents` | Refreshes recipe lifecycle tags and onboarding shells in one rolling PR without renting a VM. |

## Pull-request checks

**Tests** installs the CI dependency set on Python 3.13. Its lint job runs Ruff check and format verification; its test
job runs `make test`, including `tests/github/` coverage for helpers under `.github/scripts/`. Hugging Face downloads
used by tests are cached because anonymous shared-runner traffic is rate-limited. This workflow has no write
permission and does not use deployment credentials.

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

All discovery paths use the tracked `discover-models` skill. The agent selects exactly ten existing, fully configured
recipes for the maintained set, classifies the remaining complete recipes, and supplies a rationale for every
lifecycle decision. It keeps at most three total onboarding shells for open-weight Hugging Face models. Each new shell
contains one to three proposed deployment entries made only from `deploy.gpu` and `deploy.gpu_count`; existing shells
consume the three-shell limit. Discovery remains read-only: the workflow checks that the agent did not modify the
checkout, then
`.github/scripts/discovery_lifecycle.py` validates and applies its lifecycle manifest. The helper tolerates a model
reasoning wrapper around the JSON object, but requires exactly the four expected top-level fields before validating
their contents. The agent writes that manifest through the runner to one explicitly allowed temporary path; it cannot
write recipe changes itself. Once a nonempty manifest exists, an inference failure during the optional confirmation
turn does not discard it; the repository validator remains the authoritative completion gate.

The repo-owned `emmy agent run` command calls a configurable OpenAI-compatible CloudRift endpoint. It provides bounded
public-web search and fetch tools while rejecting private, link-local, and metadata addresses. Search results,
redirects, response sizes, extracted text, command output, and the final transcript are bounded. Discovery never
provisions hardware. The reusable runner contract is documented in `emmy/agent/ARCHITECTURE.md`.

### Direct onboarding

**Onboard model** accepts an exact Hugging Face model ID, exact GPU name/count, multimodal qualification mode, optional
existing onboarding PR, and explicit image-publication authorization. The job has a six-hour limit and gives the
agent an earlier deadline so artifact validation and cleanup retain time. When the PR contains a discovery shell, the
qualified recipe replaces it and changes `onboarding`/`untested` to `best-effort`; later discovery runs can promote it
to the maintained set.

The workflow uses `gh` to resolve an existing labeled onboarding PR or prepares a new artifact branch, provisions
exactly the requested platform through CloudRift or optional GCP, and passes the resulting SSH target to the tracked
`onboard-model` skill. If neither provider can supply the exact target, the workflow fails; it does not silently change
GPU type or count. The skill produces the recommended recipe, its compact serving report, and reproducible experiment
YAML. Raw benchmark output, experiment reports, dated run snapshots, and onboarding summaries are not repository
artifacts. An Emmy-tuned prebuilt image is produced only when the architecture, quantization, trace, serving, and
release gates pass.

The agent returns an atomic manifest. `.github/scripts/onboarding_artifacts.py` accepts only declared changes under the
allowed recipe, experiment, serving-image, canonical-golden, and matching plan paths. Unmanifested or exploratory
output is rejected. The workflow then commits those artifacts, updates or opens the onboarding PR, and uses renewable
GitHub App credentials for the long-running push path.

### Discovery lifecycle PR

**Discover model** runs nightly or by manual dispatch. It updates one rolling draft PR rather than opening one PR per
model. A legacy discovery plan PR is adopted as the rolling PR, and the workflow fails closed if more than one rolling
discovery PR exists. It also adopts one unpaired discovery branch left by an interrupted PR-creation step, while
failing closed if multiple such branches would make ownership ambiguous.

The validated manifest tags the ten selected complete recipes `maintained`, keeps other useful recipes runnable as
`best-effort`, and uses `obsolete` only when the rationale names an all-around better maintained or best-effort
replacement for the same task at a comparable or lower practical VRAM footprint, or gives a technical reason the
recipe should no longer be used. The manifest must classify every complete recipe at most once; the validator
conservatively assigns an omitted complete recipe to `best-effort`. For decisions with a replacement, it compares
qualified targets and demotes the proposal to `best-effort` unless the replacement is active, serves the same task,
and its smallest deployment uses no more total physical GPU memory than the old recipe's smallest deployment. Unknown
lower-priority model IDs are ignored so the corresponding real, omitted recipes also default to `best-effort`;
unknown maintained IDs still fail validation because all ten selections must be exact. The agent must use
`best-effort` when the old model retains any material capability or operating advantage. Every complete recipe stores
the current rationale directly under `model`. Obsolete recipes remain in git but cannot be deployed, benchmarked,
published, or bundled; a later reassessment may return one to the maintained or best-effort set.

The workflow creates `onboarding`/`untested` shells up to the three-shell total. Each shell stores its rationale under
`model` and a list of one to three candidate deployment entries under `matrices`; it does not claim qualification. The
workflow removes superseded `plans/onboard-*.md` files, commits the lifecycle update to the rolling branch, and uses
the API-only `make setup-agent` target for repository setup plus `gh` for rolling-PR discovery and updates. It never
rents a VM. Discovery uses a bounded research prompt and a workflow-specific model-turn cap so the manifest is written
before the agent transcript reaches the inference endpoint's context ceiling. The lifecycle script pre-renders one
compact inventory of recipe identity, lifecycle, rationale, task, and deployment setups into the prompt rather than
making the agent load complete serving configurations, and the shared runner retains only a bounded recent history.
Discovery reserves 4,096 output tokens because its only durable model output is the atomic manifest; onboarding
retains the larger general-purpose runner default.

## Credentials, VM ownership, and cleanup

Agent workflows transfer the CloudRift inference key through a mode-`0600`, one-use file, replace the secret-bearing
shell, and unlink the file before the first agent tool call. Agent tool subprocesses do not inherit CloudRift, GCP, or
GitHub credentials. Onboarding retains only the explicitly required Hugging Face and Docker Hub credentials. The
self-hosted runner must not carry unrelated ambient cloud credentials.

`emmy vm create gpu --lease` writes a run-owned lease as soon as CloudRift returns an instance ID or GCP creates the
named instance. The lease binds the provider handle, exact request, workflow owner, and SSH target. Cleanup through
`emmy vm delete lease` accepts only that lease, retries deletion, and audits only the recorded handle; it never
enumerates or deletes unrelated VMs. Cleanup runs from both the agent shell trap and an `if: always()` workflow step,
then `emmy vm audit lease` fails the job if its owned VM is still active.

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
`https://inference.cloudrift.ai/v1`.
