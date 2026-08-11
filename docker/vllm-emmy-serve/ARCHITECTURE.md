# vllm-emmy-serve — prebuilt per-model serving images

The release pipeline for **one model on one GPU**, served by `EmmyGenModel`, with the compiled CUDA kernels
(**cubins**), the **HF model snapshot**, and the **execution-plan pack** (`emmy/compiler/backend/pack.py`) baked in.
Cold-start pays zero `nvcc` compiles, zero HF downloads, and — on a pack hit — none of the compiler frontend either
(no trace / pass pipeline / fork resolution / codegen; boot collapses from ~25 min to ~weight-load time), so
`docker run --gpus all --ipc=host -p 8000:8000 cloudriftai/vllm-emmy-<slug>:TAG` serves with no `HF_TOKEN` and no
network dependency on HuggingFace (`HF_HUB_OFFLINE=1` is baked). The plain [`vllm-emmy`](../vllm-emmy/) image stays
the general-purpose base (any model, compile-on-boot); a baked image trades a large pull (the weights on top of the
~10 GB base) for a deterministic, tokenless boot.

**One (model, GPU, config) triple per image, and the naming schema says which.** `model_slug.sh` maps an HF id to a
docker-safe slug — org dropped, lowercased, junk collapsed to `-` — and that one slug names **both** the published
image and the pinned config:

| HF model id | slug | config | image |
| --- | --- | --- | --- |
| `google/gemma-4-12B-it` | `gemma-4-12b-it` | `models/gemma-4-12b-it.env` | `cloudriftai/vllm-emmy-gemma-4-12b-it` |
| `Qwen/Qwen3-Embedding-0.6B` | `qwen3-embedding-0.6b` | `models/qwen3-embedding-0.6b.env` | `cloudriftai/vllm-emmy-qwen3-embedding-0.6b` |

One implementation of that mapping on purpose (a shell script both `make` and the container scripts call): a slug
that disagreed between the warm and the bake would silently load two different configs, which is exactly the
cache-key parity failure the rest of this document exists to prevent. Onboarding a model is therefore one new
`models/<slug>.env` — no new Makefile targets, no new Dockerfile, no new scripts.

**The slug does not encode the revision.** An HF id that publishes several variants under one repo — an EXL3
checkpoint carries one branch per bit rate, a base-model repo carries training-checkpoint branches — maps every one
of them to the same slug, the same config file and the same image name. `SERVE_REVISION` is what separates them, and
it separates them only inside the config: two rungs of one repo cannot be released side by side without distinct
image names. Pin it by **commit sha**, never a branch name — a branch can be re-cut under the same name (which
would move the weights under a config that claims to be sealed), and an offline boot resolves `snapshots/<sha>`
directly. A config that pins nothing warms whatever the default branch points at, so `warm.sh` refuses an unpinned
revision on any repo with more than one branch.

Every step is `make <target> MODEL=<hf-id>`; `make serve-config MODEL=<id>` prints the resolved slug, config, tag
and target GPU, and `make serve-models` lists the models that already have a pinned config.

## Why a "warm" step exists: the cache-key parity contract

**Why not just cross-compile?** `nvcc` targets any arch from any machine (that's exactly what the preflight script
does), but the cache is **content-addressed, not searched**: at boot the server generates each kernel's source,
computes `sha1(source, name, arch, toolkit_tag, flags)` (see `emmy/compiler/backend/cuda/nvcc.py`), and looks up
that exact file. A cubin built from source that differs by one character has a different hash and is simply never
found — locally-produced cubins aren't "slightly worse", they're invisible. So the image doesn't need "kernels that
run on the target card"; it needs **the exact kernels the released server will ask for**, and two of the hash inputs can't be
reproduced off the card:

- the kernel **source** — the compiler picks each kernel's schedule partly from the **live-probed** GPU features; a
  memorized-spec cross-compile can drift on any feature that steers a pick, changing the source text and the hash;
- the kernel **set** — the programs are enumerated by an actual `emmy serve --generate` boot (48 layers × symbolic +
  decode twins at the pinned config), which must load the ~24 GB model — a serving run, not a compile.

Hence the warm: one real serving run on the target card, inside the image. The full contract it satisfies:

- `source` — live-probed target-card featurization + the real program enumeration (above).
- `toolkit_tag` is the compiling nvcc — the warm must run **inside the image**, not on the host toolchain.
- `flags` — production `-O3`: never warm with `EMMY_NVCC_FLAGS` set (tune's `-Xcicc -O1` would poison the key).
- The serving config (model / revision / dtype / max-model-len / max-num-batched-tokens / decode bucket / cudagraph
  capture ladder / any pinned flag that moves the plugin's path, such as `--kv-cache-dtype`) changes **which
  programs exist and their shapes** — warm and release must use identical values. The revision is the one whose
  absence is silent: the wrong rung boots, verifies and publishes with nothing flagging it, which is why it has its
  own gate in `warm.sh` and its own image-vs-config comparison in `verify.sh`.

The preflight script is the flip side: the *toolchain acceptance* question ("does this nvcc compile every
kernel family for `sm_120`?") IS answerable by cross-compilation, so that part runs anywhere, before the rental.

The machinery enforces the last two points structurally: **`models/<slug>.env` is the single source** of the serving config
(make passes it to both the warm run and the bake), and **`serve.sh` is the single serve invocation** — the warm run
bind-mounts it into the plain image and the baked image ships it, so the warmed and released servers execute
literally the same script with the same env.

**The warm runs to a fixpoint under the release environment.** One online boot is not enough: the shipped image
serves offline (`HF_HUB_OFFLINE=1`, the model resolved to its snapshot path), and a handful of kernels only
materialize on an offline boot; independently, a fork pick can flip between boots, selecting between two stable
kernel variants (2026-07 5090 session: 6 offline-only kernels + 2 bimodal ones on gemma-4-12B — each variant's
source is byte-stable, so the union converges). After the online pass, `warm.sh` re-boots offline against the
accumulated cache until a boot compiles nothing new (max 5 passes); **non-convergence after 5 passes is a hard
failure** (`warm.sh` exits 1 — with the cubin set still growing, verify's single boot passing is a coin flip and
customer boots would recompile at runtime). The boot-to-boot pick flip is a real compiler bug (fork resolution
should be deterministic across processes); the fixpoint warm contains it but does not excuse it.

**The pack changes the fixpoint's role.** The online pass also writes the execution-plan pack (`warm/pack`, keyed
on the model **config hash** + serving shape — not the id/path, precisely so the offline boots share it; a
compressed checkpoint adds a digest of its compression declaration, which the config hash cannot see, so two rungs
of one conversion never share a pack), and every
subsequent boot — the offline fixpoint passes, verify, customer boots — loads it: fork picks are frozen in the
artifact, so the bimodal-pick class vanishes and the fixpoint converges immediately whenever the pack takes. The
loop stays as a safety net for the case where the pack write was skipped (a weight outside the pack vocabulary) or
the load falls back — then the old cubin-union behavior is exactly what runs.

**A serving-shape override at run time is a pack miss, and it is expensive.** The pack is keyed on the serving shape,
of which `EMMY_GEN_DECODE_BUCKET` is part, so a deployment that overrides the bucket the image was warmed at gets no
pack hit for the decode programs — the boot re-runs the whole compiler frontend (trace, pass pipeline, fork
resolution, codegen) once per program. Measured at bucket 256 against a 32-warmed image: ~50 minutes of pure host CPU,
with the cubins already baked (`compile+alloc=0.00s`) so nothing is saved by the cubin cache. That exceeds the
compose healthcheck's own boot window (`start_period` 1200s + 180 × 10s probes = 3000s), which makes a bucket
override a coin flip on host core count rather than a supported configuration. Recipes that need a width outside the
warmed set should either re-warm the image at that width or expect the deploy to be killed as unhealthy.

**So an image bakes a pack per serving shape it is meant to serve, not just the pinned one.** `SERVE_WARM_SHAPES` in
`models/<slug>.env` lists the extra shapes as `<decode_bucket>:<prefill_bucket>:<max_num_batched_tokens>[:fm]` (an
empty field keeps the pinned value); `warm.sh` runs the offline fixpoint once per shape and each writes its own pack
directory, and `verify.sh` boots the baked image once per shape and asserts the pack HIT plus zero new cubins. Packs
are a few MB, so the cost is warm time, not image size. The `:fm` suffix exists because the precision gate is part of
the pack's environment key — a `EMMY_FAST_MATH=1` boot can never hit a standard-lane pack, and before that key
existed it silently *did*, serving standard kernels under the fast-math label (fixed 2026-08-01). What this is worth:
in the 2026-08-01 article reproduction 24 of 28 emmy benchmark cells missed the pack at ~12 min of frontend each,
about 4.5 hours of the session — and every customer following the documented per-workload knobs pays the same on
every boot. An extra shape that will not converge is reported loudly but does **not** fail the release: the pinned
shape is the contract, and the degraded outcome for the others is a cold boot, not a broken image.

## Files

- `models/<slug>.env` — the pinned serving config, one file per model (the filename IS the slug). Every value is cache-key-relevant; it must be **final before warming**
  (re-measure memory headroom on the card first — see the workflow). Make-includable `VAR=value` syntax. Six keys
  are required (`SERVE_MODEL`, `SERVE_MAX_MODEL_LEN`, `SERVE_MAX_NUM_BATCHED_TOKENS`, `SERVE_GPU_MEM_UTIL`,
  `SERVE_DECODE_BUCKET`, `SERVE_GPU`); the rest are per-checkpoint opt-ins whose defaults reproduce a dense,
  unquantized, default-branch release exactly — `SERVE_WARM_SHAPES`, `SERVE_REVISION`, `SERVE_QUANT`,
  `SERVE_CAPTURE_SIZES`, `SERVE_EXTRA_ARGS`, plus the runner memory/shape lane
  (`SERVE_EMBED_HOST`, `SERVE_PREFILL_CAPACITY`, `SERVE_PREFILL_BUCKET`, `SERVE_M1_TIER`), and the release-gate scope
  opt-in `SERVE_STATIC_ONLY`. The latter four map
  immutably to their `EMMY_GEN_*` variables in initial warm, every shape fixpoint, the baked image, and verify;
  an extra warm shape's prefill field overrides the pinned bucket. A test rejects any other key, because a
  misspelled one reads as a value nothing consumes.
  `SERVE_STATIC_ONLY=1` narrows only the golden audit, and is fail-closed: it requires runner capacity, decode bucket,
  and scheduler maximum all equal to one, prefill disabled, the M1 tier enabled, capture sizes exactly `[1]`, and no
  warm-shape override outside that same envelope. Without it the audit retains all standard widths plus symbolic.
- `serve.sh` — the frozen generative serve invocation (the arg set `emmy serve --generate` builds: `--runner
  generate --dtype float16 --hf-overrides EmmyGenModel`, the `FULL_DECODE_ONLY` whole-step decode-cudagraph
  compilation-config with the forced fused `rotary_embedding` CustomOp, `--no-enable-prefix-caching`, + the
  `SERVE_*` config; keep in sync with `_gen_graph_args` / `build_serve_cmd` in `emmy/commands/serve.py`). What the
  CLI decides by probing the checkpoint, this script reads from the config, because the config is what the bake
  seals: `SERVE_QUANT=exl3` adds `"quantization_config": null` beside the architectures override (vLLM has no EXL3
  quantization method and refuses the boot at config parsing, though nothing in the engine needs one — emmy owns
  every coded weight), and `SERVE_CAPTURE_SIZES` replaces the power-of-two capture ladder, which an **MoE model must
  cap at `[1]`**: single-token steps ride the runner's fixed-slot expert dispatch (fixed launch set, capture-legal)
  while wider decode steps keep the routed dispatch, which host-syncs and stays eager.
- `warm.sh` — runs the **plain** `vllm-emmy` image on the target GPU with `./warm` mounted at `/opt/emmy`, waits for
  `/health`, issues one completion (covers prefill + decode kernels), stops. Result: `warm/hf` (the model snapshot —
  the download happens here, once), `warm/cubin` (every compiled kernel), and `warm/pack`
  (the execution-plan pack the first boot writes). Before any of it, two refusals: the live GPU against `SERVE_GPU`,
  and an unpinned `SERVE_REVISION` against the repo's branch list (one HTTP call to the HF refs API; unreachable
  refs skip the check rather than fail it, since warming offline off a pre-seeded snapshot is supported).
  `SKIP_REVISION_CHECK=1` overrides, for the case where the default branch really is the target.
- `Dockerfile` — `FROM` the plain image, `COPY warm/hf` + `COPY warm/cubin` + `COPY warm/pack` to `/opt/emmy`, bakes
  the config env, `EMMY_PACK_DIR` and `HF_HUB_OFFLINE=1`, entrypoint `serve.sh`. The caches live at **`/opt/emmy`**
  on purpose: compose/recipes bind-mount the host HF cache over `/root/.cache/huggingface`, which would shadow
  anything baked there. The baked `HF_HOME` + `HF_HUB_OFFLINE=1` pair is also the signal `emmy deploy` reads
  (`orchestrate._baked_hf_cache`): an image that declares itself self-contained keeps its own `HF_HOME` in the
  generated compose and skips the download step entirely. Deploy used to override `HF_HOME` unconditionally, which
  hid the baked snapshot while offline mode stayed on — the download then failed outright and no deploy from a
  prebuilt image was possible.
- `verify.sh` — compares the image's baked `SERVE_REVISION` against the config's (a tag built from an older config
  serves different weights and still passes every check below), then cold-starts the **baked** image with no token,
  issues one completion, and diffs the cubin file set before/after: an empty diff proves 100% cache hit (zero
  compiles), and the offline boot proves zero downloads.
  When a pack is baked, it also asserts the boot **hit** it (a silent fallback to the full compile would still pass
  the cubin check while re-paying the frontend on every customer boot). The hit signal is the runner's "pack hit"
  line grepped from `docker logs` — reachable because `emmy.serving.register()` self-attaches a log handler under
  the bare vLLM entrypoint (2026-07-23: without it emmy INFO logs never surfaced and the gate false-FAILed a boot
  that demonstrably hit the pack). The container is removed by an EXIT trap on every path, pass or fail.
- `warm/` — gitignored; the warm output that the bake copies in.

## Workflow

Only the highlighted steps need the physical target card; the bake and push are GPU-free (but see the note below on
where to run them):

```mermaid
flowchart LR
    gold["make serve-goldens<br/><i>anywhere</i>"]
    base["make vllm-emmy-image<br/><i>anywhere</i>"]
    warm["make serve-warm<br/><b>target GPU</b>"]
    bake["make serve-image<br/><i>anywhere</i>"]
    verify["make serve-verify<br/><b>target GPU</b>"]
    push["make serve-push<br/><i>anywhere</i>"]
    gold -- "coverage OK" --> base --> warm -- "warm/ (hf + cubin)" --> bake --> verify -- PASS --> push
    classDef gpu fill:#76b900,color:#fff,stroke:#4e7a00
    class warm,verify gpu
```

The `release-serving-image` skill (`.claude/skills/release-serving-image/`) automates this whole session — rental or
local mode, abort gates per step, a human approval pause before the push, guaranteed teardown. The manual steps:

The full release session on a rented card (each step from the repo checkout; host prereqs for steps 4–5:
`make setup` + `pip install -e ".[serving]"` + cupy + `export HF_TOKEN=…`):

0. **Golden coverage** — assert this (model, card) pair has recorded goldens before spending anything:

   ```bash
   make serve-goldens MODEL=google/gemma-4-12B-it
   ```

   The goldens are the **top tier of the fork-resolution evidence hierarchy**: they are what seeds each kernel with
   a tuned schedule. Warm a model with no goldens for its shapes and cold greedy picks instead — on unseeded
   projection shapes that is a scalar tile ~770× off cuBLAS, and on some shapes a kernel that hangs outright — and
   those picks are then frozen into the shipped cubins **and the pack**, where no later boot revisits them. So a
   golden-less release does not ship a slightly slower image; it ships a permanently bad one.

   Matching is by the golden's recorded `model:` provenance, in two halves. The **repo** half compares as slugs,
   with a `-`-boundary prefix rule so a base checkpoint's goldens cover its instruction-tuned sibling (same layer
   geometry, same kernel shapes) while a quantized or resized variant correctly misses. The **revision** half
   compares against `SERVE_REVISION`, which `make serve-goldens` forwards: since the slug deliberately does not
   encode the revision (above) and a repo's revisions do NOT share kernel shapes — an EXL3 rung differs in exactly
   the per-tensor bit allocation the shape keys carry — a golden may tag its provenance `<repo>@<revision>`, and
   such an entry covers that revision and no other. An **untagged** golden makes no revision claim and covers every
   revision of its repo, which is how every non-coded golden file behaves. Tagged goldens plus a release that named
   no revision is **unevaluable**: the gate fails saying so, rather than reporting zero coverage for a card that
   plainly has some. Revisions compare as exact strings (an abbreviated hex sha matches the full one it prefixes);
   a branch name and a commit sha never match, because nothing offline can resolve one to the other.

   On FAIL the script distinguishes "this card has no goldens at all" from "this card is tuned, but for other
   models" from "it is tuned for this model, at another revision" and names them — that difference decides what to
   do next, so it is a question for a human, not something to proceed through. On PASS it still reports any
   repo-matching goldens the revision rule excluded, since the fork resolution will consult them regardless (the
   `model:` tag is provenance, never a join key).

1. Build the base image the warm will compile inside of:

   ```bash
   make wheel && make vllm-emmy-image        # or: docker pull cloudriftai/vllm-emmy:TAG
   ```

   A pulled tag must come from the SAME commit you release from (the wheel is part of the cubin `source`); when in
   doubt, build on the rental. To pin a non-default tag for every later target: `make VLLM_EMMY_TAG=… <target>`.
2. Preflight the toolchain with the **image's** nvcc — mount the `scripts/` directory (the preflight imports its
   sibling `check_serving_goldens`, so a single-file mount dies on `ModuleNotFoundError`); it needs no repo root or
   GPU in-container (the emmy wheel + nvcc are in the image; it hides CUDA and resolves goldens off-GPU):

   ```bash
   docker run --rm --entrypoint bash -v "$PWD/scripts":/scripts:ro \
       -e MODEL=google/gemma-4-12B-it -e ARCH=sm_120 \
       cloudriftai/vllm-emmy:TAG /scripts/preflight_serving_kernels.sh   # expect: <N> OK, 0 FAIL
   ```

   The enumeration is this model's golden set (the same matcher as step 0, revision half included — pass
   `-e REVISION=$SERVE_REVISION` when the goldens are tagged), so the preflight covers exactly the picks the warm
   will deploy. `<N>` grows as goldens land — the gate is **0 FAIL with at least one OK**, never a specific count.

3. **Re-measure memory headroom** and finalize `models/<slug>.env` — the config seals the cache key, so it cannot change
   after this point without re-warming. Step `--max-model-len` / `--max-num-batched-tokens` up from the old floor
   (256) with the decode bucket ON, watching for OOM, e.g.:

   ```bash
   ./venv/bin/emmy serve --generate google/gemma-4-12B-it --bench \
       --max-model-len 2048 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.97
   ```

   Write the largest passing values into `models/<slug>.env`, including `SERVE_GPU` — and, before the sweep rather
   than after it, the per-checkpoint keys, since they decide what the sweep is even measuring: `SERVE_REVISION` (the
   commit sha; sweep the rung you will ship), `SERVE_QUANT`, `SERVE_CAPTURE_SIZES`, `SERVE_EXTRA_ARGS`. The sweep
   command above is `emmy serve --generate`, which derives the same three from the checkpoint itself — pass
   `--revision <sha>` there so it derives them from the right one. Then `make serve-config MODEL=<id>` prints the
   whole resolved config, revision included; read it back before starting a multi-hour warm.
4. Correctness gate at the pinned config (A/B vs HF eager; the Phase-A exit check):

   ```bash
   ./venv/bin/python scripts/validate_serve.py --model google/gemma-4-12B-it \
       --max-model-len <pinned> --max-num-batched-tokens <pinned> --gpu-mem-util <pinned>
   ```

5. `HF_TOKEN=… make serve-warm MODEL=<id>` — fills `warm/` (first boot downloads the model + compiles all layers; minutes).
6. `make serve-image MODEL=<id>` → `make serve-verify MODEL=<id>` (expect `PASS — served offline with zero new cubins`) →
   `make serve-push MODEL=<id>`.
7. Point the recipes that should use it at the new tag; tear down the rental.

**Where to run bake/push:** although `serve-image` and `serve-push` are GPU-free, run them on the rental anyway —
moving `warm/` off-datacenter means a ~24 GB download plus a ~35 GB Docker Hub upload from a home link, and the
verify step needs the card between bake and push regardless. Never "top up" the cache on a
different card: any kernel compiled elsewhere is a dead cache entry the real card never hits, which is why
`warm.sh` and `verify.sh` check the live GPU against the config's `SERVE_GPU` and refuse a mismatch.

### Running it locally (the target card in the box)

The steps are identical — the machinery doesn't care where the card lives; steps 1–7 run as written, minus the
rental/teardown. The local-only deltas:

- **Pin the GPU on a multi-GPU box.** `warm.sh` / `verify.sh` default to `--gpus all` and vLLM takes device 0 — on
  a box that also holds another card, set `GPU_DEVICE=<index>` (both scripts, and `make` passes env through:
  `GPU_DEVICE=1 make serve-warm MODEL=<id>`). Warming on the wrong card produces a dead cache: wrong arch, wrong
  featurization, zero hits on the real card. `warm.sh` checks the live GPU against `SERVE_GPU` and refuses a
  mismatch, but a multi-GPU box still needs `GPU_DEVICE` pointed at the right index.
- **Skip the 24 GB download by pre-seeding the snapshot.** If the model is already in the local HF cache, copy it
  into the warm dir before warming — the hub client finds it and downloads nothing, and `HF_TOKEN` becomes
  unnecessary (`warm.sh` only requires the token when the snapshot is absent). Add `HF_HUB_OFFLINE=1` to make the
  initial boot fail closed if that exact snapshot is incomplete or its pinned revision cannot be resolved locally:

  ```bash
  mkdir -p docker/vllm-emmy-serve/warm/hf/hub
  cp -r ~/.cache/huggingface/hub/models--google--gemma-4-12B docker/vllm-emmy-serve/warm/hf/hub/
  HF_HUB_OFFLINE=1 make serve-warm MODEL=google/gemma-4-12B-it
  ```

- **Disk budget: ~100 GB free** (measured, not the earlier ~60 GB estimate): base images (~21 GB unpacked) +
  `warm/` (~24 GB) + the baked weight layer (~24 GB) + BuildKit's transient context copy of `warm/` during the bake
  (another ~24 GB) + the host venv/HF cache if present. Freeing the venv before the bake is safe — nothing after
  the warm needs it.
- **The push is the slow part.** `make serve-push` uploads ~35 GB over your uplink — hours on a residential
  connection vs minutes from a datacenter. It's the main reason the rental flow exists; locally, just let it run.
- **The snapshot ships re-sharded, as four image layers.** Docker Hub rejects blobs past ~10 GB (upload initiation
  503s forever), and gemma-4-12B ships ONE consolidated 23 GB `model.safetensors` — a single file cannot be split
  across layers by COPY. `make serve-image MODEL=<id>` therefore first runs `reshard_snapshot.py` (inside the base
  image), rewriting the consolidated file as standard HF shards + `model.safetensors.index.json` — per-tensor
  bytes identical, loader-transparent — then `split_hf.sh` balances the tree into four hardlinked sub-10 GB parts
  that the Dockerfile COPYs back into `/opt/emmy/hf` (the split asserts completeness — every source file lands in
  exactly one part — and that each part stays under the ~10 GB blob cap). Kernel cache-key parity is unaffected
  (weights are runtime constants, not source). The reshard verifies every tensor byte-identical against the
  consolidated source BEFORE deleting it — the post-bake verify gate only proves the shards load and the cubin set
  is closed, not that the weights survived.

## Licensing

gemma-4 is **Apache 2.0** — public redistribution of the weights in a Docker Hub tag is permitted. The bake copies
the HF snapshot (resharded, per-tensor identical — see above), which carries its LICENSE/NOTICE files — keep them
(that is the attribution obligation), and don't imply Google endorsement in the tag name or description.
