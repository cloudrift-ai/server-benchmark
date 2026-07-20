---
name: release-gemma4-image
description: Use this skill when the user asks to "release the gemma-4 image", "build the prebuilt gemma-4 image", "run the gemma-4 image release session", "warm and bake the gemma-4 kernels", "create/publish the gemma-4 docker image with prebuilt kernels", or otherwise wants the vllm-emmy-gemma4 release pipeline executed end to end. Runs the documented workflow (docker/vllm-emmy-gemma4/ARCHITECTURE.md) on a rented RTX 5090 (via start-remote-server) or a local one — base image, toolchain preflight, memory-headroom sweep that pins config.env, HF-parity validation gate, warm, bake, offline zero-recompile verify, push — with hard abort gates at every step, a human approval pause before the push, and guaranteed teardown.
version: 0.1.0
---

# Release the prebuilt gemma-4-12B serving image

The steps, their rationale, and the cache-key parity contract live in
[`docker/vllm-emmy-gemma4/ARCHITECTURE.md`](../../../docker/vllm-emmy-gemma4/ARCHITECTURE.md) — read it first; this
skill does not restate it. What the skill adds is **orchestration**: mode selection (rental vs local 5090), detached
execution that survives SSH drops, hard PASS/FAIL gates that abort-and-teardown instead of rationalizing past a bad
signal, the headroom-sweep policy that finalizes `config.env`, secret hygiene for the push, and the release
side-effects (config PR, recipe retag).

Budget: ~2–3 h wall on a rental (image build ~20 min, headroom sweep ~30–60 min, validate ~30 min, warm ~30 min,
bake/verify/push ~30 min). **Hard cap: 4 h** — if the session exceeds it, capture logs, tear down, report.

## Inputs to confirm

Ask only for what the user hasn't already given:

1. **Mode** — rental (default) or local. Local requires a 5090 in `nvidia-smi` on this machine; if the box is
   multi-GPU, resolve the 5090's index and export `GPU_DEVICE=<index>` for every warm/verify invocation. Check
   ≥100 GB free disk (base image + `warm/` + baked weight layer + BuildKit's transient ~24 GB context copy of
   `warm/` — the measured figure; the earlier ~60 GB estimate predates it and hits ENOSPC mid-bake) before
   starting local mode.
2. **`HF_TOKEN`** — needed for the gated gemma-4 download during headroom/validate/warm. Local mode with the model
   already in `~/.cache/huggingface` can skip it by pre-seeding (ARCHITECTURE.md "Running it locally").
3. **Docker Hub credentials for the push** — ask the user for a **short-lived access token** (not their password;
   `docker login -u <user> --password-stdin`). Never echo it; `docker logout` in teardown regardless of outcome.
4. **Env file** (CloudRift creds, rental mode) — default `.env`, per the `start-remote-server` sourcing rules.

## Step 0 — Provision (rental mode; skip in local mode)

Delegate to the `start-remote-server` skill: `emmy vm create gpu --gpu "NVIDIA GeForce RTX 5090" --gpu-count 1`
(CloudRift). Capture `REMOTE` and the teardown handle. Then prepare the host (CloudRift image quirks: `apt update`,
`python3.12-venv`/`python3.12-dev`, `nvcc` via `CUDA_HOME=/usr/local/cuda`; Docker + NVIDIA runtime are preinstalled
on CloudRift images — verify with `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi` or the
vllm-emmy image itself). Rsync the working tree at the **release commit** and `make setup` +
`./venv/bin/pip install -e ".[serving]"` + cupy (the venv steps back the headroom sweep and the validate script).
Host toolchain: **CUDA >= 12.9** (FlashInfer refuses sm_120 below it — the misleading error is "requires sm75 or
higher"); on non-CloudRift hosts verify the NVIDIA container toolkit actually works (`docker run --gpus all ...
nvidia-smi`) — a registered runtime with missing binaries fails with `could not select device driver`. The
venv-based steps (headroom sweep, validate) additionally need `CUDA_HOME`/nvcc exported in the step's own
environment (emmy refuses to serve without nvcc) and a **system-wide** `ninja` (`apt install ninja-build` —
FlashInfer's JIT subprocess won't see a venv-local one). For the model download use `hf download` with a retry
loop and `HF_HUB_DISABLE_XET=1` — `huggingface-cli` is a deprecated no-op stub now, and the Xet transfer path
flakes; note `warm.sh` resolves `warm/` relative to its own directory (`docker/vllm-emmy-gemma4/warm/`), so
pre-seed the snapshot there, not at the repo root. Budget
**~100 GB disk** (base images + venv + HF cache + BuildKit's transient context copy of warm/), or `rm -rf venv`
before the bake — nothing after the warm needs it.

**Run every long step detached** (`nohup … > step.log 2>&1 &` or `emmy`'s detached patterns) and poll the log — an
SSH SIGHUP mid-compile must not kill the step or eat its traceback.

## Step 1 — Base image

`make wheel && make vllm-emmy-image` (~20 min, detached). A pulled `cloudriftai/vllm-emmy:TAG` is acceptable only if
pushed from the same commit (the wheel is part of the cubin cache key) — when in doubt, build.

## Step 2 — Toolchain preflight (GATE)

Run the preflight inside the freshly built image (exact command in ARCHITECTURE.md step 2).
**Gate: the output must end `== preflight done: <N> OK, 0 FAIL` with `<N>` nonzero.** `<N>` is the golden count and
grows as goldens land — the gate is **0 FAIL and at least one OK**, not a specific N (the script itself hard-fails
an empty golden enumeration, so `0 OK, 0 FAIL` can no longer exit 0). Anything else → capture the failing `*.log`s from
the preflight out dir, abort, teardown, report. Do not attempt to fix compiler/toolchain bugs inside the release
session.

## Step 3 — Headroom sweep → pin `config.env`

Policy (decode bucket stays at 16): try `--max-model-len`/`--max-num-batched-tokens` at 256 → 512 → 1024 → 2048 →
4096 (stop at 4096 — the dynamic-dim cap), `--gpu-memory-utilization 0.97`, each via a detached
`./venv/bin/emmy serve --generate google/gemma-4-12B --bench --max-model-len N --max-num-batched-tokens N`. A config
**passes** when the server reaches `/health`, the bench completes, AND the serve log has no
`EngineCore encountered a fatal error` (the exit code alone hides tail crashes — a drained bench can die after its
metrics print; grep the log). It **fails** on CUDA OOM, death before health, or a logged engine fatal. Keep the largest passing N. If even 256 fails with the decode bucket on, retry 256 with
`EMMY_GEN_DECODE_BUCKET=0` and report that the memory fixes regressed (that is a finding, not a config).

Write the winner into `docker/vllm-emmy-gemma4/config.env`. **The config is sealed from here on** — any later change
invalidates the warm.

## Step 4 — Correctness gate (GATE + human pause)

`./venv/bin/python scripts/validate_gemma4_serve.py --model google/gemma-4-12B --max-model-len <pinned>
--max-num-batched-tokens <pinned> --gpu-mem-util <pinned>` (detached; ~30 min — it runs HF eager refs then the
served A/B).

**Gate: overall PASS (first-token match per prompt).** On FAIL: capture the per-prompt side-by-side, abort,
teardown, report — never warm from a server that mismatches HF.

**Human pause: show the user the per-prompt side-by-side output and get an explicit go-ahead before continuing.**
First-token match is the mechanical gate; a human eyeballs the continuations for coherence before anything gets
published. Rental billing note: the pause costs rental time — say so when asking.

## Step 5 — Warm

`HF_TOKEN=… make gemma4-warm` (detached; the script itself polls `/health`, fires one completion, stops the
container). Confirm the summary line reports a non-zero cubin count and `warm/hf` holds the model snapshot.

## Step 6 — Bake, verify (GATE), push

```bash
make gemma4-serve-image
make gemma4-serve-verify     # GATE: must print "PASS — served offline with zero new cubins"
docker login -u <user> --password-stdin   # the short-lived token from Inputs
make gemma4-serve-push
docker logout
```

A verify FAIL means cache-key drift between warm and bake — capture the printed cubin diff (it names the recompiled
kernels), abort, teardown, report. Do not push a partially-hitting image.

## Step 7 — Release side-effects + teardown

1. On a feature branch: commit the finalized `config.env`, point the gemma-4 recipe image tags at the pushed
   `cloudriftai/vllm-emmy-gemma4:<ver>-<sha>` tag, and open a PR (repo contribution checklist applies). If the plan
   file `plans/gemma4-prebuilt-kernel-bench-image.md` still exists, its success criteria are now met — delete it in
   the same PR.
2. **Teardown (rental mode) — always, including on every abort path above**: `emmy vm delete …` with the captured
   handle, after `docker logout`. Report the total rental time.
3. Report: pushed tag, pinned config, validate summary, verify line, and any findings.

## Failure handling

Every gate failure follows the same shape: capture the relevant log tail (not the whole log), `docker logout` if
logged in, teardown the rental, and report what failed with the evidence — the fix happens in a normal dev session,
never inline in the release session. The skill's first real invocation doubles as its shakedown: if a step's
mechanics disagree with reality (a flag, a path, a timing), patch the skill/scripts in the follow-up PR alongside
the fix.
