# Automated model onboarding

Three workflows use the self-hosted `agents` runner:

- **Onboard model** takes an exact Hugging Face ID and exact GPU name/count, rents that platform, runs the tracked
  `onboard-model` skill, commits its measured artifacts, and opens or updates a pull request.
- **Discover and onboard model** runs `discover-new-models`, selects at most one model without a recipe for the exact
  target, and calls **Onboard model**.
- **Discover model nightly** is read-only until it finds one candidate. It opens a draft `model-onboarding` pull
  request with a concrete plan and never rents a VM. It creates no new plan PR while three are already open.

Run **Onboard model** from a nightly plan PR's branch, or pass its PR number, to replace the plan on that same branch
with the measured recipe, experiment, report, and optional verified prebuilt image. There is no conversational
approval gate; `publish_image` is the explicit CI authorization for image publication.

## Configuration

The workflows require these repository secrets:

- `CLOUDRIFT_API_KEY` for the Chat Completions agent and CloudRift VM provisioning;
- `EXPERIMENT_APP_ID` and `EXPERIMENT_APP_PRIVATE_KEY` for renewable Git and pull-request credentials;
- `HF_TOKEN` for gated model downloads when needed;
- `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` for an eligible prebuilt Emmy image.

`GCP_SERVICE_ACCOUNT_KEY` and `GCP_SERVICE_ACCOUNT` are optional. When configured, exact GCP capacity is tried after
eligible CloudRift candidates. GCP activation uses a run-scoped config directory; both it and the service-account key
are removed before the model starts, then recreated only inside the always-cleanup step. Provider credentials are not
inherited by the agent's shell tools. The self-hosted runner must not carry unrelated ambient cloud credentials. The
agent model defaults to `Qwen/Qwen3.6-35B-A3B-FP8`; override it with `ONBOARD_AGENT_MODEL`. The endpoint defaults
to `https://inference.cloudrift.ai/v1`; override it with `CLOUDRIFT_INFERENCE_URL`. Nightly discovery defaults to one
H200 and accepts `DISCOVERY_TARGET_GPU` and `DISCOVERY_TARGET_GPU_COUNT` repository-variable overrides.

## VM ownership and cleanup

Provisioning writes a run-owned JSON lease as soon as CloudRift returns an instance ID or before GCP starts the named
instance. The lease records the repository, workflow run/attempt, exact GPU request, provider handle, zone, and SSH
target. Cleanup refuses a lease owned by another run, retries deletion, and audits only that exact CloudRift or GCP
handle. It never enumerates or deletes unrelated VMs. Cleanup runs from both a shell trap and an `if: always()` step;
the workflow fails if its owned VM is still active.

The repo-owned agent runner calls CloudRift's Chat Completions API directly. The workflow transfers its API key
through a mode-`0600` one-use file, replaces the secret-bearing shell, and unlinks the file before the first model
tool call. It removes CloudRift, GCP, and GitHub credentials from every tool subprocess while retaining the explicitly
needed Hugging Face and Docker Hub credentials. Long jobs use the existing renewable GitHub App credential helper
before pushing, then refresh the app token immediately before creating or updating the PR.

Discovery has bounded, keyless public-web search and page-fetch tools. Search queries, result counts, redirects,
response bytes, extracted text, and tool output are capped; private, link-local, and metadata addresses are rejected.
