# vllm-emmy-gemma4 — the prebuilt gemma-4-12B serving image

A release image for one model on one GPU: **gemma-4-12B served by `EmmyGenModel` on an RTX 5090**, with the compiled
CUDA kernels (**cubins**) and the **HF model snapshot** baked in. Cold-start pays zero `nvcc` compiles and zero HF
downloads — `docker run --gpus all --ipc=host -p 8000:8000 cloudriftai/vllm-emmy-gemma4:TAG` serves with no
`HF_TOKEN` and no network dependency on HuggingFace (`HF_HUB_OFFLINE=1` is baked). The plain
[`vllm-emmy`](../vllm-emmy/) image stays the general-purpose base (any model, compile-on-boot); this image trades a
~35 GB pull (~24 GB of weights on the ~10 GB base) for a deterministic, tokenless boot.

## Why a "warm" step exists: the cache-key parity contract

A prebuilt cubin is reused iff all five of `sha1(source, name, arch, toolkit_tag, flags)` (see
`emmy/compiler/backend/cuda/nvcc.py`) match what the server would regenerate. That forces how the cache is produced:

- `source` embeds the **live-probed GPU featurization** — the warm must run on a real 5090, not a cross-compile
  (a locally-produced cache would silently miss on the real card and recompile).
- `toolkit_tag` is the compiling nvcc — the warm must run **inside the image**, not on the host toolchain.
- `flags` — production `-O3`: never warm with `EMMY_NVCC_FLAGS` set (tune's `-Xcicc -O1` would poison the key).
- The serving config (model / dtype / max-model-len / max-num-batched-tokens / decode bucket) changes **which
  programs exist and their shapes** — warm and release must use identical values.

The machinery enforces the last two points structurally: **`config.env` is the single source** of the serving config
(make passes it to both the warm run and the bake), and **`serve.sh` is the single serve invocation** — the warm run
bind-mounts it into the plain image and the baked image ships it, so the warmed and released servers execute
literally the same script with the same env.

## Files

- `config.env` — the pinned serving config. Every value is cache-key-relevant; it must be **final before warming**
  (re-measure memory headroom on the card first — see the workflow). Make-includable `VAR=value` syntax.
- `serve.sh` — the frozen generative serve invocation (the arg set `emmy serve --generate` builds: `--runner
  generate --enforce-eager --dtype float16 --hf-overrides EmmyGenModel` + the `GEMMA4_*` config).
- `warm.sh` — runs the **plain** `vllm-emmy` image on the target GPU with `./warm` mounted at `/opt/emmy`, waits for
  `/health`, issues one completion (covers prefill + decode kernels), stops. Result: `warm/hf` (the model snapshot —
  the gated download happens here, once, via `HF_TOKEN`) and `warm/cubin` (every compiled kernel).
- `Dockerfile` — `FROM` the plain image, `COPY warm/hf` + `COPY warm/cubin` to `/opt/emmy`, bakes the config env and
  `HF_HUB_OFFLINE=1`, entrypoint `serve.sh`. The caches live at **`/opt/emmy`** on purpose: compose/recipes
  bind-mount the host HF cache over `/root/.cache/huggingface`, which would shadow anything baked there.
- `verify.sh` — cold-starts the **baked** image with no token, issues one completion, and diffs the cubin file set
  before/after: an empty diff proves 100% cache hit (zero compiles), and the offline boot proves zero downloads.
- `warm/` — gitignored; the warm output that the bake copies in.

## Workflow

Only the boxed steps need the physical 5090; the bake and push are GPU-free (but see the note below on where to run
them):

```
make vllm-emmy-image   →   ┌ make gemma4-warm ┐   →   make gemma4-serve-image   →   ┌ make gemma4-serve-verify ┐   →   make gemma4-serve-push
    (anywhere)             │    RTX 5090      │            (anywhere)               │        RTX 5090          │           (anywhere)
                           └──────────────────┘                                     └──────────────────────────┘
```

The full release session on a rented 5090 (each step from the repo checkout; host prereqs for steps 3–4:
`make setup` + `pip install -e ".[serving]"` + cupy + `export HF_TOKEN=…`):

1. Build the base image the warm will compile inside of:

   ```bash
   make wheel && make vllm-emmy-image        # or: docker pull cloudriftai/vllm-emmy:TAG
   ```

   A pulled tag must come from the SAME commit you release from (the wheel is part of the cubin `source`); when in
   doubt, build on the rental. To pin a non-default tag for every later target: `make VLLM_EMMY_TAG=… <target>`.
2. Preflight the toolchain with the **image's** nvcc — mount just the script; it needs no repo or GPU in-container
   (the emmy wheel + nvcc are in the image; it hides CUDA and resolves goldens off-GPU):

   ```bash
   docker run --rm --entrypoint bash -v "$PWD/scripts/preflight_gemma4_sm120.sh":/preflight.sh \
       cloudriftai/vllm-emmy:TAG /preflight.sh          # expect: 34 OK, 0 FAIL
   ```

3. **Re-measure memory headroom** and finalize `config.env` — the config seals the cache key, so it cannot change
   after this point without re-warming. Step `--max-model-len` / `--max-num-batched-tokens` up from the old floor
   (256) with the decode bucket ON, watching for OOM, e.g.:

   ```bash
   ./venv/bin/emmy serve --generate google/gemma-4-12B --bench \
       --max-model-len 2048 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.97
   ```

   Write the largest passing values into `config.env`.
4. Correctness gate at the pinned config (A/B vs HF eager; the Phase-A exit check):

   ```bash
   ./venv/bin/python scripts/validate_gemma4_serve.py --model google/gemma-4-12B \
       --max-model-len <pinned> --max-num-batched-tokens <pinned> --gpu-mem-util <pinned>
   ```

5. `HF_TOKEN=… make gemma4-warm` — fills `warm/` (first boot downloads the model + compiles all layers; minutes).
6. `make gemma4-serve-image` → `make gemma4-serve-verify` (expect `PASS — served offline with zero new cubins`) →
   `make gemma4-serve-push`.
7. Point the gemma-4 recipes at the new tag; tear down the rental.

**Where to run bake/push:** although `gemma4-serve-image` and `-push` are GPU-free, run them on the rental anyway —
moving `warm/` off-datacenter means a ~24 GB download plus a ~35 GB Docker Hub upload from a home link, and the
verify step needs the card between bake and push regardless. Never "top up" the cache on a non-5090: any kernel
compiled elsewhere is a dead cache entry the real card never hits.

## Licensing

gemma-4 is **Apache 2.0** — public redistribution of the weights in a Docker Hub tag is permitted. The bake copies
the unmodified HF snapshot, which carries its LICENSE/NOTICE files — keep them (that is the attribution obligation),
and don't imply Google endorsement in the tag name or description.
