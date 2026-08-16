# GitHub Actions architecture

GitHub Actions owns pull-request checks, package publication, and automated model discovery and onboarding.
Pull-request lint and packaging use GitHub-hosted runners, while the compiler-heavy test job uses the organization-level
`ubuntu-runners` builder group. Package publication remains GitHub-hosted. Agent-driven model work uses the separate
`agent-runners` group with the `agents` label because it can exceed ordinary hosted-runner limits and needs the tracked
skills and CloudRift inference endpoint.

## Workflow overview

| Workflow | Trigger | Runner | Result |
| --- | --- | --- | --- |
| **Tests** | Pull request to `main` | GitHub-hosted + `ubuntu-runners` | Runs Ruff, the complete test suite, and a PyPI package dry run. |
| **Publish to PyPI** | Manual dispatch or published GitHub release | GitHub-hosted | Tests, builds, publishes to PyPI, and optionally creates the release. |
| **Verify or onboard model** | Nightly schedule or manual dispatch | `agent-runners` / `agents` | Qualifies one available exact model/GPU deployment and updates the rolling lifecycle PR. |
| **Discover model** | Nightly schedule or manual dispatch | `agent-runners` / `agents` | Refreshes recipe lifecycle tags and onboarding shells in one rolling PR without renting a VM. |

There is no generic experiment workflow or GitHub dispatch input for `emmy bench`. Requested experiment runs start
from a developer checkout through the tracked `.agents/skills/run-experiment` skill.

## Pull-request checks

**Tests** runs three parallel jobs and installs the CI dependency set on Python 3.13. A newer commit cancels the
previous run for the same pull request. The GitHub-hosted lint job runs Ruff check and format verification. The
compiler-heavy job uses `ubuntu-runners` for `make test`, including `tests/github/` coverage for helpers under
`.github/scripts/`. Hugging Face downloads used by tests are cached because anonymous shared-runner traffic is
rate-limited. A separate GitHub-hosted bare-Python job runs `make pypi-dist`, the exact non-publishing build path used
by the release workflow, and requires one wheel and one source distribution. This workflow has no write permission
and does not use deployment credentials.

## Package publication

**Publish to PyPI** has two entry paths:

- Manual dispatch reads the version from `pyproject.toml`, refuses an already-tagged version, publishes to PyPI, then
  creates the matching tag and GitHub release.
- A manually published GitHub release must already have a tag matching `pyproject.toml`; the workflow validates and
  publishes that version without creating another release.

Both paths run lint and tests before building. `make pypi-dist` first installs its minimal build dependencies, then
uses `scripts/prepare_dist.py` to stage bundled recipes and rewrite repository-relative README links for PyPI before
building the wheel and source distribution. The build artifact moves between jobs through GitHub artifacts. PyPI
uses trusted publishing through the `pypi` environment and an OIDC token, so the repository stores no PyPI password.
The manual path creates its GitHub release only after a successful upload, preventing a failed publication from
leaving a release behind.

## Model discovery and onboarding

All discovery paths use the tracked `discover-models` skill. The agent selects exactly ten existing, fully configured
recipes for the maintained set, classifies the remaining complete recipes, and supplies a rationale for every
lifecycle decision. Every existing recipe and selected new model receives a 0-100 heat score for current onboarding
priority. Each promising new open-weight Hugging Face model becomes an onboarding shell with one to three proposed
deployment entries made only from `deploy.gpu` and `deploy.gpu_count`; there is no shell-count limit. Existing shells
remain in the manifest and must retain their task and deployment matrix while discovery refreshes their heat and
rationale. `emmy recipe list --json` supplies the versioned compact agent inventory under its `recipes` field, and
recipe queries enforce the maintained count after application.
The workflow checks that the agent did not modify the checkout, then validates and applies its lifecycle manifest. Its
artifact worktree remains on the rolling lifecycle branch, while the catalog, lifecycle helper, OpenCode agent and
plugin directory, and attached discovery skill come from the exact `github.sha` that started the run. This lets a
manual dispatch test a workflow PR without copying its implementation commits into the rolling branch or silently
using an older manifest contract. The helper tolerates a model reasoning wrapper around the JSON object, but requires
exactly the four expected top-level fields before validating their contents. The named OpenCode discovery agent denies
repository edits, permits only the tracked discovery skill, public-web tools, repository reads, read-only Git
inspection, and the three named read-only source subagents, and caps parent work at 64 agentic steps. The Reddit,
Hugging Face, and OpenRouter/Arena investigators run as independent bounded sources; Reddit can surface a candidate
before an exact Hugging Face identity is known. The parent agent alone reconciles identities, assigns heat, and writes
the lifecycle manifest. The last complete lifecycle object in OpenCode's final completed text event becomes the
temporary manifest and is logged before validation so a rejected decision remains inspectable; the repository
validator remains the authoritative completion gate. The project provider configuration selects the configurable
CloudRift model through an OpenAI-compatible Chat Completions endpoint and disables the model's chat-template thinking
mode for the concise JSON result. Discovery never provisions hardware.

OpenCode is provisioned on the self-hosted runners rather than maintained inside Emmy. `opencode.json` owns the model
provider alias, while `.opencode/agents/` owns the separate discovery and onboarding limits and permissions. The
tracked `.agents/skills/` remain the canonical task definitions. Compatibility symlinks under `.claude/skills/`
expose the same packages through OpenCode's native skill tool.

### Nightly verification and direct onboarding

**Verify or onboard model** runs nightly and retains a manual exact model/GPU dispatch. Its selector uses the
`emmy recipe query` command against the rolling branch's recipe root, with the command implementation loaded from the
exact workflow SHA. Manual dispatch supplies one exact external candidate; scheduled dispatch queries declared
deployments. A filtered-out manual candidate is an error, while no scheduled match is a successful no-op.
The query's filters and sorts read CloudRift VM variant availability without filtering on public-IP supply and consider
only declared deployments with an available exact CloudRift GPU count. Pending `onboarding`/`untested` recipes are the
first priority, ordered by descending heat, then model ID and deployment declaration order. If none can run, the
selector performs a second generic query for a `maintained` recipe whose committed `RESULTS.md` has the oldest
last-change timestamp; a missing report is oldest. No eligible deployment is a successful no-op.

The workflow requires the repository's `CLOUDRIFT_TEAM_ID` variable to contain the exact Robots team UUID. Before it
checks capacity, it validates that `CLOUDRIFT_API_KEY` can act for that UUID through a team-scoped account request;
every rent then includes the UUID and requests a public IP so the GitHub runner can reach the VM over SSH. It attaches
`emmy`, workflow, and GitHub job tags,
makes at most three workflow-level rental attempts for the same selection, and sweeps a failed attempt by the complete
tag set before retrying. Only V100 rentals set CloudRift's admin-only billing exemption; every other GPU is a regular
team rental. The workflow never falls back to GCP or changes the selected GPU type/count.

Unconditional teardown sends the complete tag-scoped terminate request before its bounded status audit and lease
audit. This ordering gives cancellation cleanup a short critical path inside GitHub's cancellation grace period while
retaining the owned lease as an independent verification handle.

The workflow passes the resulting SSH target and an explicit `onboarding` or `verification` mode to the tracked
`onboard-model` skill. Onboarding replaces the discovery shell and changes `onboarding`/`untested` to `best-effort`.
Verification begins from the active recipe, refreshes measurements and durable artifacts, and preserves its existing
lifecycle tag. Both modes preserve discovery-managed `model.heat`. Before the agent starts, the workflow installs the
small remote Python/rsync prerequisite set and
requires `$HOME/.cache/emmy` to be durable storage with at least 8 GiB free. Compiler staging keeps its checkout,
venv, cache, and build temporary files there rather than on a small `/tmp` tmpfs. The job has a 24-hour limit and gives
the agent a 23.5-hour deadline so artifact validation and cleanup retain 30 minutes. The shared serving experiment
retains one LFS archive and top-level row-record set per exact GPU platform plus one cumulative `RESULTS.md`; a run
replaces only its platform snapshot. Ignored dated run directories, loose benchmark output, and qualification
summaries are not repository artifacts. An Emmy-tuned prebuilt image is produced only when every release gate passes.
Nightly image publication is disabled unless
`NIGHTLY_ONBOARD_PUBLISH_IMAGE` is `true`; manual dispatch retains an explicit input.

The artifact worktree stays on the rolling lifecycle branch, while Python control code is loaded from the exact
`github.sha` whose workflow definition started the job. This keeps normal scheduled runs reproducible and lets a
manual dispatch test a workflow PR without leaking that PR's implementation commits into the model-artifact branch.
The selector runs the exact-SHA catalog logic against the rolling worktree's `recipes/` directory so lifecycle mode
and priority always reflect the branch that the agent will update.
The workflow also attaches the exact-SHA `onboard-model`, `tune-kernels`, and `run-experiment` skills as authoritative
agent inputs; older copies on the rolling branch cannot silently override a proposed artifact contract.

The agent returns an atomic manifest. `.github/scripts/onboarding_artifacts.py` accepts only declared changes under the
allowed recipe, experiment, serving-image, and canonical-golden paths. The validator requires the shared experiment
recipe and report, the exact `results_<gpu-short>x<gpu-count>.tar.gz` archive, and matching current-platform row
records.
It rejects changes to another platform snapshot and requires the current archive to be created or updated. Optional
outputs must remain in `artifacts`; unmanifested or exploratory output is rejected. The job installs a pinned,
checksum-verified Git LFS binary in the runner's temporary directory when needed, then configures LFS locally before
staging so the normal push uploads the archive object with the rolling branch. After the agent returns, the workflow
requires each task-local timestamp directory to be a root member of the declared platform archive before removing
that ignored local directory; only the durable archive proceeds to staging.
The workflow writes the agent's final completed text event to the job log before checking its exit status, preserving
the failure explanation after temporary output cleanup. The validator also checks the requested mode, exact recipe
model, expected lifecycle tag, and compact deployment and measured-performance summaries from the selected recipe
lane. The workflow then commits those artifacts, rebases on the latest default branch, and updates or opens the
rolling model lifecycle PR using renewable GitHub App credentials for the long-running push path.

Both lifecycle workflows finish with a separate GitHub-hosted notification job. Discovery groups only recipe entries
actually modified by the run under their resulting lifecycle, includes each current heat score, and links the run and
rolling PR. Onboarding includes the selected model, target, operation mode, serving deployment, and measured
performance from its validated atomic summary. Because the notification job is independent of the self-hosted agent
job, it still runs after a failure, cancellation, or timeout. Discord delivery retries three times, remains
non-blocking, and disables all mentions; the workflow run, durable reports, and rolling PR retain the complete
evidence.

### Discovery lifecycle PR

**Discover model** runs nightly or by manual dispatch. Discovery and qualification share a static concurrency group
and one rolling draft PR rather than opening one PR per model. Each workflow fails closed if more than one rolling PR
exists. It also adopts one unpaired
discovery branch left by an interrupted PR-creation step, while
failing closed if multiple such branches would make ownership ambiguous. Before rendering inventory or running the
agent, it rebases an existing rolling branch onto the latest default branch. The rebase push uses the exact original
remote head as its force-with-lease expectation; a conflict, a stale checkout, or a concurrent branch update stops the
run before any lifecycle changes are applied.

The validated manifest tags the ten selected complete recipes `maintained`, keeps other useful recipes runnable as
`best-effort`, and uses `obsolete` only when the rationale names the exact ID of an all-around better maintained or
best-effort replacement for the same task at a comparable or lower practical VRAM footprint, or gives a technical
reason the recipe should no longer be used. The manifest must classify and score every complete recipe exactly once.
For decisions with a replacement, the validator compares
qualified targets and demotes the proposal to `best-effort` unless the replacement is active, serves the same task,
and its smallest deployment uses no more total physical GPU memory than the old recipe's smallest deployment. A
replacement described as merely comparable, or whose recipe reduces configured context or concurrency, also defaults
to `best-effort` while retaining the supplied heat. Unknown or malformed lower-priority model IDs cannot stand in for
an omitted recipe because every real recipe must still be scored. A checkpoint name is normalized across a missing or
incorrect organization only when it uniquely identifies one existing recipe; ambiguous or unknown maintained IDs
still fail validation because all ten selections must resolve exactly. The agent must use
`best-effort` when the old model retains any material capability or operating advantage. Every complete recipe stores
the current rationale and heat immediately after `model.huggingface`. Obsolete recipes remain in git but cannot be
deployed, benchmarked, published, or bundled; a later reassessment may return one to the maintained or best-effort
set.

The workflow creates every selected `onboarding`/`untested` shell through the same catalog library that backs
`emmy recipe create`. Each shell stores its rationale and heat under `model` and a list of one to three candidate
deployment entries under `matrices`; subsequent runs validate the same task and setups while refreshing heat and
rationale. A shell does not claim qualification. The workflow commits lifecycle updates to the rolling branch and
uses the API-only `make setup-agent` target for repository helpers plus `gh` for rolling-PR discovery and updates. It
never rents a VM. Network operations use bounded retries, and discovery keeps research, prompt inventory, retained
history, and final output within the inference endpoint's context limit. The lifecycle helper retains only
classification policy and manifest application.

## Credentials, VM ownership, and cleanup

OpenCode inherits the CloudRift inference key only for provider requests. The project plugin removes CloudRift, GCP,
GitHub Actions, and GitHub CLI credentials from every agent shell subprocess. Onboarding retains only the explicitly
required Hugging Face and Docker Hub credentials. The self-hosted runner must not carry unrelated ambient cloud
credentials.

`emmy vm create gpu --lease` writes a run-owned lease as soon as CloudRift returns an instance ID. The lease binds the
provider handle, exact request, workflow owner, and SSH target. Cleanup first deletes and audits that handle, then
lists and terminates every still-active CloudRift VM carrying the complete run-unique tag set. The tag audit catches a
VM created before the lease was durable without selecting another job's rentals. An `if: always()` step performs both
paths after OpenCode exits and fails the job if either ownership audit leaves a VM active.

GitHub App credentials are used for long-lived branch writes and PR operations. Private keys and temporary provider
configuration live only under run-specific `/tmp/emmy-*` paths and are removed by unconditional cleanup steps.

## Repository configuration

Agent workflows use these repository secrets as applicable:

- `CLOUDRIFT_API_KEY` for model discovery, Robots-team resolution, availability, and CloudRift provisioning;
- `DISCORD_EMMY_ROBOTS_WEBHOOK_URL` for non-pinging model discovery, verification, and onboarding summaries;
- `HF_TOKEN` for gated checkpoints;
- `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` for an eligible verified prebuilt image.

`ONBOARD_AGENT_MODEL` selects the discovery/onboarding model and defaults to `Qwen/Qwen3.6-35B-A3B-FP8`.
`CLOUDRIFT_TEAM_ID` must be the exact Robots team UUID; the verification/onboarding workflow fails before capacity
selection if the variable is absent, malformed, or inaccessible to `CLOUDRIFT_API_KEY`.
`CLOUDRIFT_INFERENCE_URL` selects its OpenAI-compatible endpoint and defaults to
`https://inference.cloudrift.ai/v1`.
`NIGHTLY_ONBOARD_PUBLISH_IMAGE=true` authorizes a nightly qualification to publish an otherwise eligible image; it is
false when unset.
