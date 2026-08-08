---
name: release-serving-image
description: Use this skill when the user asks to "release the serving image for <model>", "build the prebuilt image for <HF model id>", "warm and bake the kernels for <model>", "create/publish the vllm-emmy docker image with prebuilt kernels", "release the gemma-4 image", or otherwise wants the per-model vllm-emmy serving image release pipeline executed end to end. Runs the documented workflow (docker/vllm-emmy-serve/ARCHITECTURE.md) on the model's target GPU — rented (via start-remote-server) or local — covering the golden coverage gate, base image, toolchain preflight, memory-headroom sweep that pins the model config, HF-parity validation gate, warm, bake, offline zero-recompile verify, and push, with hard abort gates at every step, a human approval pause before the push, and guaranteed teardown.
version: 0.2.0
---

# Release a prebuilt serving image for one model

The steps, their rationale, and the cache-key parity contract live in
[`docker/vllm-emmy-serve/ARCHITECTURE.md`](../../../docker/vllm-emmy-serve/ARCHITECTURE.md) — read it first; this
skill does not restate it. What the skill adds is **orchestration**: model and mode selection, detached execution
that survives SSH drops, hard PASS/FAIL gates that abort-and-teardown instead of rationalizing past a bad signal,
the headroom-sweep policy that finalizes the model config, secret hygiene for the push, and the release
side-effects.

Every step is parameterized by `MODEL` (an HF id). The slug derived from it names both the published image and the
pinned config, so nothing else needs naming:

```bash
make serve-config MODEL=google/gemma-4-12B-it   # model / slug / config path / image tag / target GPU
make serve-models                               # which models already have a pinned config
```

Budget: ~2–3 h wall on a rental for a 12B-class model (image build ~20 min, headroom sweep ~30–60 min, validate
~30 min, warm ~30 min, bake/verify/push ~30 min). **Hard cap: 4 h** — if the session exceeds it, capture logs, tear
down, report. A larger checkpoint scales this up; re-estimate before starting rather than inheriting these numbers.

## Inputs to confirm

Ask only for what the user hasn't already given:

1. **Model** — the HF id. Required; there is no default worth guessing. Run `make serve-config MODEL=<id>` and show
   the resolved slug, config path, image tag and target GPU before anything else happens.
2. **Mode** — rental (default) or local. Local requires the config's `SERVE_GPU` in `nvidia-smi` on this machine; if
   the box is multi-GPU, resolve that card's index and export `GPU_DEVICE=<index>` for every warm/verify
   invocation. Check ≥100 GB free disk for a 12B-class model (base image + `warm/` + baked weight layer +
   BuildKit's transient ~24 GB context copy of `warm/` — the measured figure) before starting local mode; scale it
   with the checkpoint.
3. **`HF_TOKEN`** — needed for a gated download during headroom/validate/warm. Local mode with the model already in
   `~/.cache/huggingface` can skip it by pre-seeding (ARCHITECTURE.md "Running it locally").
4. **Docker Hub credentials for the push** — ask the user for a **short-lived access token** (not their password;
   `docker login -u <user> --password-stdin`) **and, in the same question, the username the token belongs to**: a PAT
   authenticates only its owning account, which is usually a personal user with push rights to the `cloudriftai` org,
   not `cloudriftai` itself — the wrong username fails as "incorrect username or password". Never echo the token;
   `docker logout` in teardown regardless of outcome.
5. **Env file** (CloudRift creds, rental mode) — default `.env`, per the `start-remote-server` sourcing rules.

## Step 0 — Golden coverage (GATE, and it comes first)

```bash
make serve-goldens MODEL=<id>      # -> scripts/check_serving_goldens.py against the config's SERVE_GPU
```

**Why this gates everything, before a single GPU-hour is spent.** Emmy's greedy compile resolves every fork through
the deploy evidence hierarchy, and the live card's **recorded goldens are its top tier** — they are what seeds the
picks with tuned kernel schedules. Warm a model with no goldens for its shapes and cold greedy chooses instead,
which on unseeded projection shapes is not "a bit slower": it deploys a scalar tile ~770× off cuBLAS, and on some
shapes picks a kernel that hangs outright. Those picks then get frozen into the shipped cubins **and the
execution-plan pack**, where no later boot revisits them. A golden-less release ships that permanently.

**On FAIL, stop and ask the user.** Do not decide this alone. Report which case it is — the script distinguishes
them — and offer the real options:

- *the card has no goldens at all* → the card is untuned; releasing from it is not sensible. Record goldens first
  (the `tune-golden` skill), or release on a card that has them.
- *the card is tuned, but not for this model* → the script lists which models it IS tuned for. Options: run a golden
  sweep for this model's shapes first (the honest fix, hours of tuning); release anyway and accept cold-greedy picks
  (only defensible for a throwaway or experimental image, and it must be said out loud in the release notes); or
  release on a different card.
- *a related checkpoint is covered but the slug missed* — e.g. a quantized variant whose base has goldens. The
  matcher deliberately refuses to guess, because a quantized or resized model does not share the base's kernel
  shapes. If the user knows the shapes really are identical, adding this model to the golden file's `model:`
  provenance is the fix — not a bypass flag.

Whichever the user chooses, **carry it into the release notes**. "Released without golden coverage" is a property of
the artifact, not a detail of the session.

## Step 1 — Provision (rental mode; skip in local mode)

Delegate to the `start-remote-server` skill, asking for the config's `SERVE_GPU`, e.g.
`emmy vm create gpu --gpu "NVIDIA GeForce RTX 5090" --gpu-count 1` (CloudRift). On vast.ai instead: rent a **VM
instance** template (e.g. "Ubuntu 22.04 VM", vastai/kvm, `vms_enabled=true` in search) — plain docker-container
templates cannot run the host-level Docker builds; the account SSH key may not be injected, self-inject via
`--onstart-cmd`. Capture `REMOTE` and the teardown handle. Then prepare the host (CloudRift image quirks:
`apt update`, `python3.12-venv`/`python3.12-dev`, `nvcc` via `CUDA_HOME=/usr/local/cuda`; Docker + NVIDIA runtime
are preinstalled on CloudRift images — verify with `docker run --rm --gpus all
nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi` or the vllm-emmy image itself). Also install `git-lfs` (the repo's
hooks require it — a checkout without it dies mid-rebuild) and, when operating as root over a synced tree,
`git config --global --add safe.directory <repo>` (without it `git rev-parse` returns empty and the image tag
degrades; `make` now guards this).

**Ship the source at the pinned release commit — never rsync the live working tree.** The working tree is shared
mutable state: a branch switch in another session mid-release yields a hybrid tree whose wheel fails in confusing,
distant ways. Pin `origin/main`'s sha up front, get the tree onto the host (fresh clone, or rsync of the
`git ls-files` list plus `.git`), then `git checkout -f <sha>` there — and re-verify `git rev-parse HEAD` before any
rebuild. Then `make setup` + `./venv/bin/pip install -e ".[serving]"` + cupy (the venv steps back the headroom
sweep and the validate script). Host toolchain: a CUDA version whose nvcc supports the target arch (**>= 12.9** for
sm_120 — FlashInfer refuses it below that, and the misleading error is "requires sm75 or higher"); on non-CloudRift
hosts verify the NVIDIA container toolkit actually works (`docker run --gpus all ... nvidia-smi`) — a registered
runtime with missing binaries fails with `could not select device driver`. The venv-based steps (headroom sweep,
validate) additionally need `CUDA_HOME`/nvcc exported in the step's own environment (emmy refuses to serve without
nvcc) and a **system-wide** `ninja` (`apt install ninja-build` — FlashInfer's JIT subprocess won't see a venv-local
one). For the model download use `hf download` with a retry loop and `HF_HUB_DISABLE_XET=1` — `huggingface-cli` is a
deprecated no-op stub now, and the Xet transfer path flakes; note `warm.sh` resolves `warm/` relative to its own
directory (`docker/vllm-emmy-serve/warm/`), so pre-seed the snapshot there, not at the repo root. Budget disk for
several copies of the checkpoint (base images + venv + HF cache + BuildKit's transient context copy of `warm/`), or
`rm -rf venv` before the bake — nothing after the warm needs it.

**Run every long step detached** (`nohup … > step.log 2>&1 &` or `emmy`'s detached patterns) and poll the log — an
SSH SIGHUP mid-compile must not kill the step or eat its traceback. Detached means no tty: pass `--batch --yes` to
any `gpg` in installer snippets or it dies silently. And never `export TOKEN=$(cat …)` inside a `set -x` script —
xtrace prints the expanded secret into the log; wrap secret reads in `set +x` … `set -x`.

## Step 2 — Base image

`make wheel && make vllm-emmy-image` (~20 min, detached). A pulled `cloudriftai/vllm-emmy:TAG` is acceptable only if
pushed from the same commit (the wheel is part of the cubin cache key) — when in doubt, build.

## Step 3 — Toolchain preflight (GATE)

```bash
MODEL=<id> ARCH=<target arch> scripts/preflight_serving_kernels.sh
```

Run it **inside the freshly built image** so it uses the image's exact nvcc (exact command in ARCHITECTURE.md). It
renders every golden shape recorded for this model and `nvcc --cubin`s each one — the golden set is the enumeration
precisely because those are the picks the warm will deploy.

**Gate: the output must end `== preflight done: <N> OK, 0 FAIL` with `<N>` nonzero.** `<N>` is this model's golden
count and grows as goldens land — the gate is **0 FAIL and at least one OK**, not a specific N (the script
hard-fails an empty enumeration, so `0 OK, 0 FAIL` cannot exit 0). Anything else → capture the failing `*.log`s from
the preflight out dir, abort, teardown, report. Do not attempt to fix compiler or toolchain bugs inside the release
session.

## Step 4 — Headroom sweep → pin the model config

For a **new model** this step *creates* `docker/vllm-emmy-serve/models/<slug>.env`; for an existing one it
re-validates it. Policy (decode bucket stays at the model's tuned default): try
`--max-model-len`/`--max-num-batched-tokens` at 256 → 512 → 1024 → 2048 → 4096 (stop at 4096 — the dynamic-dim cap),
`--gpu-memory-utilization 0.97`, each via a detached
`./venv/bin/emmy serve --generate <model> --bench --max-model-len N --max-num-batched-tokens N`. A config **passes**
when the server reaches `/health`, the bench completes, AND the serve log has no `EngineCore encountered a fatal
error` (the exit code alone hides tail crashes — a drained bench can die after its metrics print; grep the log). It
**fails** on CUDA OOM, death before health, or a logged engine fatal. Keep the largest passing N. If even 256 fails
with the decode bucket on, retry 256 with `EMMY_GEN_DECODE_BUCKET=0` and report that the memory fixes regressed
(that is a finding, not a config).

The sweep's first boot compiles every kernel and can exceed `emmy serve`'s 30-min health cap (`_HEALTH_TIMEOUT_S` in
`emmy/commands/serve.py`) on a slow host — bumping it with a VM-local sed is fine; it does not affect the shipped
artifacts.

Write the winner into `models/<slug>.env`. Authoring rules, because this file is read **two ways** — `include`d by
the Makefile and `source`d by `warm.sh`/`verify.sh` — so its syntax is the intersection of make and bash:

- `VAR=value`, one per line, no `export`, no command substitution, comments on their own line.
- **Any value containing spaces must be double-quoted**, or bash runs its second word as a command (`SERVE_GPU`
  is the one that bites: unquoted, it dies with `GeForce: command not found`). Make keeps the quotes in the value,
  which is why the Makefile strips them once into `SERVE_GPU_NAME` rather than at each use site.
- Required keys: `SERVE_MODEL`, `SERVE_MAX_MODEL_LEN`, `SERVE_MAX_NUM_BATCHED_TOKENS`, `SERVE_GPU_MEM_UTIL`,
  `SERVE_DECODE_BUCKET`, `SERVE_GPU`. A test asserts all six are present and that `SERVE_MODEL`'s slug equals the
  filename — a file named for a different slug is simply unreachable from `make serve-* MODEL=`.
- `SERVE_GPU` is the card actually swept. `warm.sh` and `verify.sh` compare the live card against it and refuse a
  mismatch (`GPU_DEVICE=<index>` selects the card on a multi-GPU box; `SKIP_GPU_CHECK=1` overrides, and wanting to
  use it is a sign something is wrong).
- Optional per-checkpoint keys, each defaulting to a dense unquantized default-branch release (`serve.sh` documents
  them in full): `SERVE_REVISION` the commit sha to serve — `warm.sh` REFUSES an unpinned revision on any repo with
  more than one branch, because the default branch may be a different variant entirely; `SERVE_QUANT=exl3` for a
  checkpoint whose quantization method vLLM does not have; `SERVE_CAPTURE_SIZES` for the cudagraph ladder, which an
  MoE model must cap at `[1]`; `SERVE_EXTRA_ARGS` for further pinned flags (e.g. `--kv-cache-dtype fp8_e4m3`).
  Set these BEFORE the headroom sweep — they change what the sweep measures — and sweep with the same
  `--revision <sha>` so `emmy serve` derives the same arms from the same checkpoint.

After writing it, run `make serve-config MODEL=<id>` and confirm every line reads back as intended — that is the
cheap check that both readers agree before a multi-hour warm depends on it.

**The config is sealed from here on** — any later change invalidates the warm.

## Step 5 — Correctness gate (GATE + human pause)

```bash
./venv/bin/python scripts/validate_serve.py --model <id> \
    --max-model-len <pinned> --max-num-batched-tokens <pinned> --gpu-mem-util <pinned>
```

Detached; ~30 min for a 12B-class model — it runs HF eager refs, then the served A/B.

**Gate: overall PASS (first-token match per prompt).** On FAIL: capture the per-prompt side-by-side, abort,
teardown, report — never warm from a server that mismatches HF, because the mismatch gets baked in.

**Human pause: show the user the per-prompt side-by-side output and get an explicit go-ahead before continuing.**
First-token match is the mechanical gate; a human eyeballs the continuations for coherence before anything gets
published. Rental billing note: the pause costs rental time — say so when asking.

## Step 6 — Warm

`HF_TOKEN=… make serve-warm MODEL=<id>` (detached; the script polls `/health`, fires one completion, stops the
container, then runs the offline fixpoint passes). Confirm the summary line reports a non-zero cubin count and that
`warm/hf` holds the model snapshot. A non-converged warm is a hard failure by design — do not paper over it.

`warm/` is a single shared directory, so **one model at a time**: warming a second model into a populated `warm/`
bakes a mixture of two models' caches. Clear it between models.

## Step 7 — Bake, verify (GATE), push

```bash
make serve-image  MODEL=<id>
make serve-verify MODEL=<id>     # GATE: must print "PASS — served offline with zero new cubins"
docker login -u <user> --password-stdin   # the short-lived token from Inputs
make serve-push   MODEL=<id>
docker logout
```

A verify FAIL means cache-key drift between warm and bake — capture the printed cubin diff (it names the recompiled
kernels), abort, teardown, report. Do not push a partially-hitting image.

## Step 8 — Release side-effects + teardown

1. On a feature branch: commit the finalized `models/<slug>.env`, point any recipe that should use the new image at
   the pushed `cloudriftai/vllm-emmy-<slug>:<ver>-<sha>` tag, and open a PR (the repo contribution checklist
   applies).
2. **Teardown (rental mode) — always, including on every abort path above**: `emmy vm delete …` with the captured
   handle, after `docker logout`. Report the total rental time.
3. Report: pushed tag, pinned config, **the Step 0 golden coverage status**, validate summary, verify line, and any
   findings.

## Failure handling

Every gate failure follows the same shape: capture the relevant log tail (not the whole log), `docker logout` if
logged in, teardown the rental, and report what failed with the evidence — the fix happens in a normal dev session,
never inline in the release session. If a step's mechanics disagree with reality (a flag, a path, a timing), patch
the skill or scripts in the follow-up PR alongside the fix.
