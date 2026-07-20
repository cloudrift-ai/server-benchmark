# Debug plan: M=4096 prefill-chunk twin cudaErrorIllegalAddress (RTX 5090, bare venv)

Companion to `plans/gemma4-12b-postfix-serving-verification-5090.md`, which found the bug. Status: reproduced and
bisected to pre-existing (predates the 2026-07-20 review-fix batch); not yet root-caused.

## What is known

- **Symptom**: `emmy serve --generate` on gemma-4-12B at mnbt 4096 (the prefill twin's default width) dies during
  vLLM's profiling pass: the twin's FIRST captured-graph launch (`_Program.capture_program_graph` → `graph.launch`)
  raises `cudaErrorIllegalAddress`. Both the pre and post twins fault, at layer 0 already.
- **Standalone repro** (no vLLM): build `EmmyGenRunner.create(model_id="google/gemma-4-12B", decode_bucket=16,
  max_tokens=4096, prefill_bucket=4096)`, call `forward_layer_pre_device(0, zeros(4096, 3840, fp16, cuda))` — the
  first twin `run_device` faults. Full-model build costs ~20 min (CPU pipeline; cubins cached).
- **Pre-existing**: identical crash at the pre-batch branch tip `a18a8b5c` (worktree bisect on the same box), so
  none of the review-fix commits caused it. The branch's serving benches all ran mnbt 512 and never hit this width.
- **Possibly environment-sensitive**: the docker release-image validation (#397) passed the same model/config on a
  5090 — inside the image toolchain. The failing box runs a bare venv: torch 2.13.0+cu130, CUDA 13.0, driver-side
  cupy graphs. Env note for the box: CUDA lives at `/usr/local/cuda` and is NOT on PATH in non-interactive SSH.
- Decode twins (M=16) and the M=256 twin paths (GPU tests) are fine on the same box; symbolic `run_device_sym` at
  T=4096 served the whole 612 s bench cleanly — the fault is specific to the static twin programs at M=4096.

## Fix plan

1. **Fast repro loop (~2 min/iter)**: 1-layer model via `AutoConfig` + `AutoModelForCausalLM.from_config` (random
   weights — the crash needs shapes, not values; skips the 23 GB load), truncate `trunk.layers` to one layer,
   `EmmyGenRunner.from_model(..., prefill_bucket=M)`, run pre+post at T=M. Confirm the 1-layer repro still faults
   at 4096. Also try the first `full_attention` layer (global head_dim differs) — the sliding/global split may
   matter.
2. **Width bisect**: M ∈ {256, 512, 1024, 2048, 3072, 4096}. An onset threshold is the strongest hint:
   - fails only ≥ some power of two → suspect int32 index arithmetic or a grid-dimension limit (gridDim.y/z cap
     65535; a 2-D block grid at M=4096 with small tiles can overflow a folded axis);
   - fails at every width in the 1-layer build but not the full build (or vice versa) → suspect the shared
     `BufferArena` (all programs share one arena — a static-twin buffer aliasing a capacity buffer sized for a
     different program is exactly an illegal-address shape).
3. **Attribute the faulting kernel**: `compute-sanitizer --tool memcheck` on the smallest failing repro (names the
   kernel + OOB address even inside graph launches). Fallback: bypass graph capture (call the program's eager
   `run_once` launch sequence) under `CUDA_LAUNCH_BLOCKING=1`. In parallel, `EMMY_DUMP_DIR` the twin build and keep
   the per-kernel sources + execution plan for inspection.
4. **Root-cause in the kernel/plan source**: check, in order — flattened index arithmetic at M=4096 extents
   (products like `m·N` near/past 2³¹ in intermediate int math), grid folding vs the 65535 y/z caps, the arena
   slot sizes vs the twin's buffer shapes (compare the dumped execution plan's buffer table against the arena's
   allocations), and the deploy picks unique to M=4096 (the `--json` A/B record names the chosen knob rows — a
   config only this width picks would explain the width specificity).
5. **Fix + regression test**: the fix lands wherever the root cause sits (codegen guard, arena sizing, or a
   scheduler legality gate). Add the cheapest test that pins it: a compile-level test if it's plan/codegen math, a
   `requires_cuda` twin test at the failing width (1-layer `from_config` build keeps it minutes-cheap) if it needs
   the device to manifest.
6. **Validate end-to-end**: 1-layer repro green → full-model standalone repro green → re-run the serving bench
   with the twin ON at mnbt 4096 (drop `EMMY_GEN_PREFILL_BUCKET=0`) and compare against the twin-off numbers in
   the verification findings; the twin should improve the full-chunk prefill steps or at minimum not crash.
7. **Reconcile with the docker validation**: if the root cause turns out env-sensitive (toolkit/driver codegen),
   record the sensitivity in `docker/vllm-emmy-gemma4/ARCHITECTURE.md`'s toolchain notes and consider a preflight
   check; if it reproduces inside the image too, the release validation gate has a hole — re-check #397's 4/4
   validation config against the twin path actually exercised.
