# Volta MMA support and Qwen3.5-122B-A10B on 16×V100

## Goal

Add a real Volta tensor-core path to Emmy, run `Qwen/Qwen3.5-122B-A10B` in FP16 on 16×32 GB V100s, create and
measure the model's V100 golden configurations, and publish a reproducible article in the CloudRift blog. The article's
claim is deliberately narrow: **we made a current 122B MoE model work on Volta**. A competitive engine baseline is useful
when one exists, but is not required and must not be manufactured from unlike hardware, precision, or model variants.

This is an execution plan, not a design commitment. Validate every hardware, model, and engine assumption on the target
machine before building on it.

Delivery is split deliberately:

- **Phase 1 — implementation on the current PCIe-only host:** compiler/runtime work, model bring-up, correctness, golden
  configurations, and a dry-run benchmark harness. No number from this host is a final article result.
- **Phase 2 — final evidence on a replacement NVLink host:** topology qualification, final end-to-end benchmarks, article,
  integration, and the last rebase onto main. Phase 2 starts when the user supplies the new SSH target.

## Definition of done

1. `emmy compile --target sm_70` cannot emit instructions unavailable on Volta, and existing SM80+ kernels remain
   unchanged except where a reviewed, target-specific change is necessary.
2. Representative FP16 matmuls execute correctly on a V100 through
   `mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32`; PTX and disassembly both prove the tensor-core path was used.
3. Qwen3.5-122B-A10B loads in FP16 across all 16 V100s and completes a short, deterministic generation through Emmy.
4. Every serving-critical Emmy matmul shape has a reviewed V100 golden configuration measured at `-O3`. The checked-in
   V100 YAML passes the golden configuration tests and is selected by the live-card deploy evidence hierarchy.
5. A Phase 2 result manifest records correctness, memory, NVLink topology, absolute serving results, and any honest baseline
   that ran. All article tables and claims are derived from that manifest; Phase 1 diagnostic serving numbers are excluded.
6. A CloudRift article, benchmark plan, and scripts are added beside the existing Gemma 4 article pattern. The article
   clearly separates measured results, implementation facts, limitations, and absent baselines.
7. The completed Emmy work is rebased onto the latest `origin/main`, and full verification plus the Phase 2 benchmark suite
   pass on the rebased revision that the article names.

## Scope and fixed decisions

- Target checkpoint: `Qwen/Qwen3.5-122B-A10B`, cast to FP16 at load. Do not silently substitute an AWQ, FP8, or smaller
  model in the final results.
- Phase 1 host: `ssh riftuser@185.165.50.65` (`riftvm`), with 16× Tesla V100-SXM3-32GB on one Linux host.
- Observed topology on 2026-08-07: this is a KVM/Q35 guest with every GPU pair reported as `PHB`.
  `nvidia-smi nvlink --status` exits successfully but returns zero link entries. Treat NVLink as unavailable unless the
  provider changes GPU exposure. This is acceptable for Phase 1 and disqualifies the host only from final serving numbers.
- Observed host state: driver 580.159.03, Docker 29.5.3, no system `nvcc`, about 6.1 TB free disk, 1.3 TiB RAM, and no
  active GPU processes. Revalidate all of these at execution time.
- Phase 2 host: supplied later by the user. It must expose working NVLink paths between the GPUs used by the selected
  parallel layout; the exact SSH target, GPU identity, and topology are Phase 2 inputs, not Phase 1 blockers.
- Toolchain: pin CUDA 12.8 or 12.9. CUDA 13 does not target Volta, so the current CUDA 13 serving image is not a valid
  base for this work.
- First Volta atom: FP16 A/B with FP32 accumulation. An FP16-accumulate sibling is a separate, optional accuracy tradeoff
  and is out of scope until the standard path is correct.
- Initial context target: 32K or the largest length that leaves measured KV-cache headroom. The model's advertised maximum
  context is not a launch requirement.
- Initial serving target: text-only, batch/concurrency 1. Add higher concurrency only after the basic path is stable.
- Scalar kernels are acceptable as correctness fallbacks for unsupported operators, but completing this plan requires the
  model's eligible matmuls to use Volta MMA instructions.
- No performance claim is a success criterion. Correct execution, transparent measurements, and reproducibility are.

The checkpoint is roughly 244 GB at two bytes per parameter before runtime overhead, so 512 GB total VRAM is sufficient
on paper. The actual allocation, KV cache, communication buffers, and framework overhead are measured gates, not assumed
headroom. On the Phase 1 all-`PHB` host, test PP16 first: the 48 layers divide evenly into three layers per stage, and
pipeline boundaries move far less data than TP16 collectives on every layer. If the serving stack cannot run PP16, try
PP8×TP2, then PP4×TP4. These Phase 1 choices prove functionality only. Phase 2 chooses its final parallel layout from the
new host's NVLink/NVSwitch matrix and measured NCCL results.

## Execution protocol

The orchestrating agent owns milestone gates, commits, shared files, and final integration. Sub-agents receive bounded
work packages with named inputs, owned paths, required artifacts, and a return condition. Do not start broad tuning or
article work while the target instruction path or end-to-end model is still unproved.

1. Continue on the current Emmy branch, `feature/vq-weight-compression`; it contains useful work for this project. Do not
   create a replacement feature branch or discard unrelated branch changes. Rebase onto the latest `origin/main` only in
   the final milestone.
2. Use `ssh riftuser@185.165.50.65` only for Phase 1. Add the user-provided replacement SSH target to `status.md` at the
   Phase 2 handoff. Record the local and remote Emmy revisions used by every run.
3. Keep all generated evidence inside `_tune/volta-qwen35/`, never `/tmp`. Remote jobs run in named `tmux` sessions and
   write logs continuously so another agent can resume them.
4. Maintain `_tune/volta-qwen35/status.md` with the current milestone, exact command, host, session name, last successful
   artifact, and next action. Keep candidate measurements in append-only JSONL so interrupted searches can skip completed
   schedule strings.
5. The orchestrator is the only writer to shared coordination files, golden YAML, durable documentation, and Git history.
   Sub-agents do not stage, commit, rebase, or overwrite another agent's artifacts. Give each remote worker a disjoint GPU
   set and output directory before it starts.
6. Commit at the milestone boundaries below. Each commit should have one verifiable purpose and targeted tests, followed
   by the repository's mandatory pre-commit `make test` and `make lint` checks. Do not mix compiler support, model
   integration, measurements, and article prose in one commit.
7. Record failures as evidence. After a command fails, preserve its command, log, and environment before changing the
   stack. Never convert a failed or timed-out result into a performance number.
8. Re-read the closest `ARCHITECTURE.md`, `STYLE.md`, `README.md`, `AGENTS.md`, and `GLOSSARY.md` before the relevant edit.
   Update durable documentation only with stable facts, and never link this plan from code or durable docs.

Expected evidence layout:

```text
_tune/volta-qwen35/
  status.md
  environment.txt
  topology.txt
  subagents/
    assignments.md
    handoffs/
  stack-probes/
  mma-smoke/
  kernel-inventory.json
  kernel-inventory.md
  golden-seed/candidates.jsonl
  golden-sweep/tune.log
  golden-sweep/results/
  correctness/
  serving/
  phase1-handoff.md
  final-bench/
  article-manifest.json
```

## Sub-agent execution map

The orchestrator writes `_tune/volta-qwen35/subagents/assignments.md` before each wave. Every assignment names the input
revision, allowed files, remote GPU IDs, `tmux` session, output directory, timeout, success condition, and facts that need
to be returned. A sub-agent finishes by writing a short handoff under `subagents/handoffs/`; it does not merely report that
it “looked into” the task.

| Phase/wave | Owner | Bounded work | Owned output | Depends on |
| --- | --- | --- | --- | --- |
| P1/A | host agent | M0 PCIe host, CUDA container, NCCL diagnostics, stack probes | `_tune/volta-qwen35/{environment,topology,stack-probes}/` | none |
| P1/A | compiler-audit agent | Locate SM80 assumptions and propose the minimal M1–M3 edit/test set | `subagents/handoffs/compiler-audit.md` | none |
| P1/A | model-stack agent | Verify Qwen config, loader, PP support, and exact external-engine failure/success | `subagents/handoffs/model-stack.md` | none |
| P1/B | compiler owner | Implement M1–M3; sole writer to compiler/lowering files during this wave | code, tests, `mma-smoke/` | P1/A handoffs |
| P1/C | model owner | Implement M4 and produce the actual serving kernel inventory | serving code/tests, `kernel-inventory.*` | M3 gate |
| P1/D | golden owner | Seed M5, run Path A, own the YAML and final A/B decisions | golden YAML, sweep log, findings | M4 gate |
| P1/D | manual-search workers | Search disjoint inventory-shape shards on disjoint GPUs if Path A fails | `golden-seed/shard-*/candidates.jsonl` | seed entries |
| P1/E | harness agent | Implement and dry-run M7 without producing publishable results | recipes/scripts, schema tests | M6 gate |
| P1/F | orchestrator | Freeze the implementation revision and Phase 1 handoff | `phase1-handoff.md`, commits | M7 gate |
| P2/A | host agent | Qualify the replacement NVLink machine and final parallel layout | `final-bench/{environment,topology,nccl}/` | new SSH target |
| P2/B | benchmark agent | Execute the frozen harness and produce the final manifest | `final-bench/raw/`, `article-manifest.json` | M8 gate |
| P2/C | article agent | Draft only from the final manifest and technical handoffs | CloudRift article directory | M9 gate |
| P2/D | orchestrator | Review, docs, checks, cleanup, final rebase, rerun, and submission | commits and PR evidence | all gates |

Phase 1 wave A may use three sub-agents in parallel because their writes are disjoint. M1–M4 use one product-code owner at
a time; review agents may run read-only tests but must not patch the same modules. Path A tuning has one owner and one
shared process so within-sweep prior transfer is preserved. Path B may fan out only by disjoint shape names and GPU sets;
the golden owner merges candidate records and is the sole YAML writer. Stop all agents at the Phase 1 gate until the user
provides the new host. In Phase 2, the article agent starts only after `article-manifest.json` is frozen.

## Phase 1 — implementation on the current PCIe-only V100 host

Phase 1 is independently executable now. It ends with implementation, correctness, reviewed golden configurations, and a
dry-run final-benchmark harness. The current host's serving performance is diagnostic and must not enter the article.

### M0 — validate the supplied machine and prove the serving stack

#### Work

1. Re-capture `nvidia-smi -q`, `nvidia-smi topo -m`, `nvidia-smi nvlink --status`, driver packages, GPU memory, CPU, RAM,
   storage, OS, NCCL, PCIe tree, NUMA placement, and container runtime into `environment.txt` and `topology.txt`.
2. Record the confirmed KVM/Q35 guest boundary and zero NVLink entries. Ask for no provider-side change as a prerequisite;
   run a small NCCL peer-to-peer and all-reduce matrix to establish the real PCIe communication baseline and proceed.
3. Do not reinstall the driver or spend Phase 1 time recovering NVLink. The user will provide a replacement NVLink host for
   Phase 2. Preserve the working 580.159.03 installation unless ordinary CUDA correctness requires a driver change.
4. Build or select a CUDA 12.8/12.9 container and prove it can compile and run a trivial `-arch=sm_70` program. Do not depend
   on a system `nvcc`; none was present during the planning inspection.
5. Select a Transformers/vLLM-compatible stack that understands Qwen3.5 and Volta. Test the latest known Volta-focused
   vLLM fork first rather than relying on the older pinned image in the AWQ recipe. Record its source revision and image
   digest. A missing `SharedFusedMoE.sm70_hidden_logical_size` or another SM70-specific failure is a stack failure, not an
   Emmy compiler failure.
6. Download the official checkpoint once to the host's local storage. Inspect its config and tensor dtypes, then perform
   FP16 load-only probes in topology order: PP16, PP8×TP2, PP4×TP4, and TP16 only if needed. Record peak allocated and
   reserved memory per rank plus inter-stage/collective failures.
7. If an external engine can generate with this exact checkpoint, precision, topology, and context, preserve it as the
   candidate baseline configuration for Phase 2. Any current-host latency is diagnostic only. Limit the investigation to
   three clearly different stack revisions/configurations; then record incompatibility and continue without that baseline.

#### Exit gate

- The 16 GPUs are healthy, the PCIe/NVLink state is recorded, the topology-informed parallel layout is fixed, CUDA 12.x
  builds SM70 code, and the exact checkpoint loads in FP16 with enough measured headroom for a short generation.
- NVLink is not a Phase 1 exit requirement. Record the PCIe limitation and continue with the best functional PP/TP layout.
- If the exact model cannot load, stop. A smaller or quantized model may be used for compiler smoke tests but cannot satisfy
  the project goal or support publication.

#### Commit

No product commit is required. Preserve the environment and stack-probe manifests for later article reproduction.

### M1 — make target capabilities explicit and keep SM70 legal

#### Work

1. Add target capability predicates in the compiler context for the instruction families the scheduler needs to decide:
   Volta MMA, `ldmatrix`, `cp.async`, TMA, BF16 MMA, and FP8 MMA. Do not scatter raw compute-capability comparisons across
   lowering passes.
2. Audit schedule enumeration and lowering so SM70 never offers or realizes `m16n8k16`, `ldmatrix`, `cp.async`, TMA, BF16
   MMA, or FP8 MMA. A user-pinned illegal choice must fail with a precise target/feature error.
3. Ensure `--target sm_70` reaches source generation, NVCC, cubin caching, and any source/cubin identity key. A binary built
   for another target must not be reused on V100.
4. Add CPU/source tests for the capability matrix and illegal pin failures. Add a digest or source comparison proving that
   representative SM80+ output did not change from capability plumbing alone.

#### Exit gate

- Cross-compilation tests show that SM70 source contains none of the forbidden instructions and that SM80, SM89, SM90, and
  SM120 still expose their previous legal paths.

#### Commit

`compiler: make GPU instruction capabilities target-aware`

### M2 — add the Volta `m8n8k4` MMA family

Volta's `m8n8k4` form is not a smaller spelling of the existing `m16n8k16` atom. A warp issues four independent MMA
operations, with different fragment registers and lane ownership, and Volta has no `ldmatrix`. Model it as its own atom
family so later passes cannot accidentally apply Ampere layout assumptions.

#### Work

1. Extend the atom and fragment model with an explicit Volta FP16/FP32-accumulate atom. Encode operand shape, accumulator
   shape, warp multiplicity, register types, lane mapping, and store mapping as atom data rather than renderer conditionals.
2. Render `mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32` with all warp lanes participating uniformly. Add focused tests
   for operand packing, accumulator initialization, inline PTX constraints, and output coordinates.
3. Add a Volta operand-load implementation that does not use `ldmatrix`:
   - first, a direct/global fragment gather that favors obvious correctness;
   - then, if needed for usable performance, synchronous shared-memory staging using ordinary loads/stores and
     `__syncthreads()`.
4. Generalize or gate every shape-specific consumer of the current atom, including C-to-A register repacking, paired
   `ldmatrix` loads, fragment stores, the fused norm/linear path, and twisted attention lowering. Unsupported composite
   paths should decline the MMA schedule and use the existing materialized fallback, not miscompile.
5. Run live V100 tests for small and irregular matmuls: transposed and canonical B, odd M/N tails, several K multiples,
   static and symbolic M, and FP16 output with FP32 accumulation. Compare against PyTorch/cuBLAS tolerances.
6. For representative shapes, save generated CUDA/PTX and `cuobjdump` or `nvdisasm` output. Assert the compiled kernel has
   HMMA instructions and benchmark it against Emmy's scalar path. A vendor baseline is optional; the MMA path should at
   least beat Emmy's own scalar implementation on representative tensor-core-friendly shapes.

#### Exit gate

- Live V100 accuracy passes for the test matrix, disassembly proves HMMA execution, and compute-sanitizer reports no access
  or synchronization errors on the focused suite.

#### Commit

`compiler: lower FP16 contractions to Volta m8n8k4 MMA`

### M3 — integrate the Volta atom with schedule enumeration

#### Work

1. Add the Volta tile spelling to the existing `TILE` codec and permitted-move catalog. Derive legal output fragments,
   K steps, warp geometry, register budget, and shared-memory use from the atom family. Do not add model-shape branches.
2. Restrict SM70 `STAGE` choices to global/direct or synchronous shared-memory movement. The existing `d<depth>/cp` and
   `d<depth>/tma` choices must not appear. Gate the paired-`ldmatrix` peephole entirely on the relevant atom capability.
3. Extend prior features only where atom geometry is currently conflated with the Ampere shape. It is acceptable for the
   first SM70 choice to be conservative or manually seeded; do not retrain the global prior before measurements show that
   is needed.
4. Test schedule parse/spell round trips, enumeration bounds, exact pin realization, K/tail legality, transposed B,
   symbolic M, split-K choices, register limits, and shared-memory limits.
5. Re-run kernel-source digests for representative existing targets. Investigate any non-SM70 change before proceeding.

#### Exit gate

- The scheduler enumerates valid Volta schedule strings without exposing newer instruction families; a pinned string is
  realized exactly in the emitted kernel; invalid strings fail loudly; existing target digests stay stable.

#### Commit

`compiler: enumerate legal Volta MMA schedules`

### M4 — inventory and close Qwen3.5 execution coverage

Golden configurations are shape-specific deploy evidence, not one model-wide setting. Inventory actual serving programs
before deciding which entries the V100 file needs.

#### Work

1. Trace the exact FP16 checkpoint through the intended generation path without materializing duplicate weights. Inventory
   each unique Emmy kernel program by layer class, operation, M/N/K, B layout, static or symbolic dimension, and dtype.
2. Include all Qwen3.5 paths exercised by short generation: routed and shared MoE projections, dense projections, full
   attention, linear-attention support code, normalization, embeddings, and logits. Do not infer coverage solely from the
   architecture config; use emitted serving programs.
3. Choose a small evidence set of actual widths after observing the server: decode widths such as 1/8/16/32 and prefill
   widths such as 256/512/2048/4096, plus `.dynM` where serving can select it. Dynamic golden entries use Emmy's sequence
   hint of 512.
4. Write `kernel-inventory.json` and a readable Markdown rendering. Each entry records the program that produced it, the
   expected golden name, whether the Volta MMA family is eligible, and the measured fallback when it is not.
5. Close unsupported graphs surgically. Use existing materialization or PyTorch/vLLM seams for non-matmul work; do not
   broaden this project into a new attention or MoE implementation unless the exact serving path requires it.
6. Verify correctness in increasing scope: primitive kernels, one representative layer of each class, full-model logits
   on short inputs, then deterministic short generations against the selected reference stack. Store tolerances and outputs.

#### Exit gate

- The exact model completes a deterministic short generation on all 16 V100s. The inventory has no unexplained program,
  every eligible matmul uses the Volta atom when pinned, and every non-MMA path has a documented implementation owner.

#### Commit

`serving: run Qwen3.5-122B-A10B in FP16 on Volta`

### M5 — bootstrap the V100 golden configuration file

`emmy tune --dataset golden` cannot bootstrap a live GPU that has no golden entries. Seed reviewed entries from the actual
Qwen serving programs first; only then attempt the normal dataset sweep.

#### Work

1. Create `emmy/compiler/pipeline/search/goldens/v100_sm70_qwen35_122b.yaml` with the exact live GPU name, compute capability
   `[7, 0]`, model provenance, and one entry per serving-critical inventory shape/regime.
2. Obtain initial schedule strings from the scheduler's actual variant output or schedule codec. Never guess a spelling in
   the YAML. Start with the simplest legal Volta MMA schedule and no optional reduction/raster/staging move, then add one
   legal move at a time.
3. Pin each candidate on the actual in-model program. Isolated matmul snippets are useful diagnostics but are not the
   authority when their fork tree, B layout, fusion, or memory behavior differs from serving.
4. Benchmark the candidate at deployable `-O3`, record all schedule families including explicit off values, and add
   `emmy_us` plus the same-lane PyTorch/cuBLAS latency when it exists. Hand-edit the YAML; do not round-trip it through
   PyYAML.
5. Validate the file with:
   - `./venv/bin/pytest tests/compiler/test_golden_configs.py -q`
   - the golden drift gate and relevant compiler end-to-end tests;
   - `emmy eval golden --in-model --model Qwen/Qwen3.5-122B-A10B` on the V100 host;
   - `scripts/check_serving_goldens.py` for the captured program set.
6. Recompile without `EMMY_KNOBS` and prove each program selects the checked-in entry from the live-card golden tier. A
   pinned candidate is not deployment evidence until this unpinned check passes.

#### Exit gate

- The V100 YAML loads, every inventory shape is covered, recorded strings realize exactly, and an unpinned live compile
  selects the expected entries.

#### Commit

`compiler: seed V100 goldens for Qwen3.5-122B-A10B`

### M6 — tune the goldens, with manual schedule discovery as the guaranteed fallback

The normal `tune-golden` workflow is the first attempt after M5 because it gives consistent candidate accounting and prior
diagnostics. It is not a dependency: SM70 may expose assumptions in the tuner, or the isolated golden snippet may not
represent an actual Qwen program. In either case, switch to the bounded manual path below rather than debugging the tuner
indefinitely.

This work stays in Phase 1 because a single-GPU kernel schedule does not depend on inter-GPU NVLink. Phase 2 still rechecks
live-card identity, representative `-O3` results, and unpinned selection before using the entries for final serving.

#### Path A: normal golden sweep

1. Smoke one or two entries on one GPU. Then use homogeneous V100 workers, increasing to all 16 only after compile and
   memory pressure are stable:

   ```bash
   mkdir -p _tune/volta-qwen35/golden-sweep
   ./venv/bin/emmy tune --dataset golden --clean --gpus 16 \
     2>&1 | tee _tune/volta-qwen35/golden-sweep/tune.log
   ```

2. Treat the tuner's `-Xcicc -O1` measurements as ranking evidence only. Recompile finalists at default `-O3` and record
   only the `emmy run --bench --golden NAME` or same-lane `--ab` results.
3. Compare the greedy winner against the best recorded golden in the same precision and compiler lane. Re-run marginal
   winners; below 5% is presumed noise unless repeated evidence is tight, and small shapes may need a wider 10–13% bar.
4. Preserve the rank of the winning schedule, failures, compile times, and per-knob misses for the findings report.

#### Path B: manual schedule discovery

Switch immediately when the sweep produces no valid candidates, repeatedly times out, fails before measuring, enumerates
newer instruction families, cannot reproduce a pin, or ranks an isolated snippet differently from the actual model program.

1. Ask the scheduler/variant diagnostic for all legal strings for the exact captured program. Save the raw rows and filter
   only by target legality, resource limits, and exact realization. Do not hand-invent codec values.
2. Enumerate a bounded grid in this order so partial progress remains useful:
   - legal `WORK` warp grids;
   - Volta `TILE` fragment and K-step choices;
   - off/direct staging, then legal synchronous shared-memory `STAGE` choices;
   - legal `REDUCE` choices;
   - `RASTER` last.
3. Pin complete schedule families through `EMMY_KNOBS` on the actual in-model program. Use a short correctness probe before
   every benchmark, a hard per-candidate timeout, and per-launch medians rather than whole-model noise for initial ranking.
4. The golden owner partitions the inventory by complete shape name and assigns each manual-search worker disjoint GPU IDs,
   a `volta-M6-search-<shard>` `tmux` session, and `golden-seed/shard-<N>/candidates.jsonl`. Never let two workers search
   the same shape or write one tuning DB/JSONL file concurrently.
5. Append `{shape, program, schedule, compiler flags, correctness, latency, failure, revision}` to the assigned shard file.
   Resume by skipping exact strings already recorded. Never discard failed rows.
6. Carry the best few candidates into `-O3`, run at least three interleaved repetitions against the current golden, and
   accept only a repeatable same-lane improvement. If a snippet winner and serving-program winner differ, record the actual
   serving-program schedule.

#### Shared completion work

1. Hand-edit genuine winners into the V100 YAML with every recorded knob, including off values. Leave the current golden in
   place when no candidate clears the evidence bar.
2. Re-run M5 validation and an unpinned in-model compile after every YAML batch.
3. Write `plans/golden-sweep-v100-qwen35-findings.md` with a per-shape A/B table, prior rank, knob misses, failures, manual
   search notes, and recommendations. When the implementation lands, delete this execution plan before adding the findings
   report so `plans/` remains within its ten-file cap.

#### Exit gate

- Every eligible inventory shape has the best reviewed schedule found within the bounded budget, all recorded numbers are
  `-O3`, and both checked-in validation and unpinned live selection pass. The gate does not require the tuner to work or a
  golden to beat cuBLAS.

#### Commit

`compiler: tune Qwen3.5 V100 golden configurations`

### M7 — implement and dry-run the final benchmark harness

#### Work

1. Add checked-in experiment recipes or scripts for environment capture, topology/NCCL capture, correctness prompts,
   kernel A/Bs, external-engine probes, and Emmy serving. Pin image digests, revisions, model revision, engine arguments,
   parallel layout, seeds, warmups, repetitions, and timeouts.
2. Define the raw result schema and a deterministic builder for `article-manifest.json`. The schema must represent missing
   baselines with a null value plus reason, retain every repetition, and keep diagnostic Phase 1 runs distinct from final
   Phase 2 runs.
3. Encode the final workload set without collecting publication numbers: boot/time-to-ready, per-rank memory and KV-cache
   headroom, short decode at concurrency 1, a medium 4K-input/256-output point, maximum tested context, TTFT, TPOT, output
   and request throughput, errors, per-kernel MMA coverage, and compile/tuning time.
4. Dry-run every command on the current host at the smallest useful workload. Prove that timeouts, cleanup, raw outputs,
   manifest generation, and table generation work. Mark every output `phase: 1`, `final: false`; do not compare or publish
   these serving numbers.
5. Write `_tune/volta-qwen35/phase1-handoff.md` with the exact branch revision, remote revision, container digest, model
   cache location, working parallel layout, golden coverage, known failures, final commands, expected runtime, and Phase 2
   entry requirements.

#### Exit gate

- A clean Phase 1 deployment completes a deterministic short generation, and the frozen harness can run unattended on a
  different 16×V100 host and regenerate its manifest/tables. No final serving result has been claimed.

#### Commit

`experiments: prepare Qwen3.5 V100 benchmark harness`

### Phase 1 completion gate and pause

Phase 1 is complete only when M0–M7 pass, every intended implementation change is committed on
`feature/vq-weight-compression`, `make test` and `make lint` pass, the V100 MMA disassembly/correctness gate passes, the
model generates on the current host, and `phase1-handoff.md` is complete. Stop here and release the current machine if the
user wants. Do not start article prose or treat current-host serving measurements as final. Resume only after the user
provides the replacement NVLink host.

## Phase 2 — final evidence on the replacement NVLink V100 host

Phase 2 has one external input: an SSH-accessible 16×V100 machine with NVLink exposed for the intended parallel layout.
The implementation and harness arrive frozen from Phase 1; host qualification may uncover bugs, but it should not reopen
unrelated compiler design work.

### M8 — qualify the replacement host and freeze the final layout

#### Work

1. Add the new SSH target to `status.md` and capture GPU identity, UUIDs, clocks, power limits, ECC, driver, CUDA, OS,
   CPU/RAM/storage, container runtime, `nvidia-smi topo -m`, `nvidia-smi nvlink --status`, NUMA placement, and NCCL version
   under `_tune/volta-qwen35/final-bench/`.
2. Require `NV#` paths for the GPU groups used by the final layout and active links in the NVLink status output. Run
   peer-to-peer, all-reduce, and all-gather tests across the exact groups. If the intended path still traverses only `PHB`,
   stop Phase 2 and request another host; do not fall back to the Phase 1 machine for final numbers.
3. Confirm the new GPUs have the same live golden identity (`gpu_name`, compute capability, and relevant resource limits).
   If the identity differs, repeat M5–M6 for a new GPU-specific file before benchmarking. If it matches, re-bench a
   representative golden subset at `-O3` and prove unpinned live selection; retune only reproducible regressions.
4. Sync the exact Phase 1 revision without stale cubins, reproduce the `m8n8k4` accuracy/disassembly gate, and complete the
   deterministic short generation.
5. Evaluate topology-compatible layouts with short diagnostic probes, then freeze one final layout. Prefer TP groups that
   remain within the observed NVLink/NVSwitch domain; use pipeline parallelism between domains when the matrix is split.
   Record the choice before any final run.

#### Exit gate

- The replacement host exposes working NVLink for the selected layout, focused correctness/golden checks pass from a clean
  cache, the exact model generates, and the final layout plus environment manifest are frozen.

### M9 — collect the final benchmark evidence

#### Baseline hierarchy

Use the first applicable baseline and label it exactly:

1. The same Qwen checkpoint, FP16 precision, context, topology, and workload on a Volta-capable external engine.
2. PyTorch/cuBLAS for isolated kernels only; do not turn kernel results into serving claims.
3. Emmy's pre-MMA scalar path as an internal implementation comparison.
4. If none is representative, publish absolute Emmy results with no speedup column.

An engine that uses AWQ, fewer GPUs, a different Qwen variant, or materially different context is background information,
not a baseline.

#### Work

1. Run the frozen M7 harness from a clean deployment. Save all outputs under `final-bench/raw/` with the exact host,
   topology, branch revision, image digest, model revision, layout, and command.
2. Run warmups and at least three measured repetitions in an interleaved order. Preserve request-level results and failed
   repetitions; do not select only the fastest run.
3. Re-run deterministic correctness prompts after performance measurements, restart from the checked-in recipe, and
   reproduce the result without manual pins or an untracked cache.
4. Build `article-manifest.json` exclusively from Phase 2 raw data, with null/reason fields for baselines that could not run.
   Regenerate every planned table and verify that no Phase 1 serving number entered the manifest.

#### Exit gate

- A clean deployment on the NVLink host produces the recorded result, raw Phase 2 data regenerates every table, and all
  claims have an exact manifest field or source artifact.

#### Commit

`experiments: reproduce Qwen3.5 FP16 results on NVLink V100s`

### M10 — write the CloudRift article

Create a new directory under `packages/blog/content/blog/`, using
`optimizing-gemma-4-12b-rtx/{index.md,benchmark-plan.md,benchmark-scripts/}` as the structural reference. A working slug is
`running-qwen35-122b-on-v100`; use the site's established naming and front-matter conventions at implementation time.

#### Article outline

1. **What we made work** — exact Qwen checkpoint, FP16, 16×V100, tested context, and final NVLink topology.
2. **Why Volta needed a separate path** — `m8n8k4`, four MMA operations per warp, distinct fragment layout, and the lack
   of `ldmatrix`, `cp.async`, and TMA.
3. **Compiler work** — target capability gates, Volta atom/lane mapping, synchronous movement, and schedule enumeration.
4. **Model work** — memory fit, hybrid attention/MoE coverage, parallel layout, and the actual serving kernel inventory.
5. **Finding schedules** — golden configuration bootstrap, normal tuning results, and manual schedule-string search if it
   was required.
6. **Correctness and results** — absolute memory and Phase 2 serving measurements first; baseline columns only where
   comparable.
7. **Limitations** — toolchain pin, tested context, final topology, unsupported/fallback paths, compile/tune cost, and why a
   modern serving baseline may be absent.
8. **Reproduction** — exact commands, scripts, revisions, image digests, model revision, expected runtime, and raw result
   schema.

#### Writing gates

- Prefer “run,” “support,” or “made it work.” Use performance language only when matched measurements support it.
- Explain that FP16 is a runtime cast of the named checkpoint if that remains the implementation; do not imply an official
  FP16 artifact exists.
- Do not generalize one topology's numbers to all V100 clusters.
- Link the benchmark plan and scripts from the article, not this ephemeral Emmy plan.
- Generate every numeric table from `article-manifest.json`; manually written numbers fail review.
- Keep the tone and structure of the Gemma article while making the thesis distinct: hardware enablement rather than a
  vendor-baseline performance win.

#### Exit gate

- A reader can reproduce the reported Phase 2 run from a clean NVLink host, and every performance sentence is traceable to
  a matched result or explicitly marked as an observation/limitation.

#### Commit

Use a separate CloudRift branch and PR: `blog: run Qwen3.5-122B-A10B FP16 on 16 V100s`.

### M11 — integration, durable docs, and pre-rebase checks

1. Update the nearest compiler and pipeline `ARCHITECTURE.md` files with the stable Volta atom/capability invariants and key
   entry points. Update serving/recipe docs only if their contracts changed.
2. Read and compare `STYLE.md`, `README.md`, and `AGENTS.md`; update only when their durable guidance became inaccurate.
   Check every new term against `GLOSSARY.md` and add glossary text only for a genuinely stable concept.
3. Run targeted GPU tests on V100, then `make test` and `make lint` in the required repository lane. Run the relevant
   serving golden coverage and kernel digest checks separately because the default test suite is not a V100 performance
   test.
4. Remove this completed plan, enforce the ten-file `plans/` cap, and confirm no durable document or code comment points to
   any `plans/*.md` file.
5. Confirm all remote `tmux` jobs have finished, each handoff names its exact revision, and the worktree contains no
   generated evidence or sub-agent scratch files intended to remain untracked.

### M12 — rebase the current branch onto latest main, refresh final evidence, and submit

This is intentionally the last milestone. Do not rebase between long remote measurements because their recorded revision
must remain available and unambiguous.

1. Confirm `feature/vq-weight-compression` is clean and every intended change is committed. Preserve unrelated branch work;
   do not squash or drop existing VQ commits as part of the Volta cleanup.
2. Fetch the latest main and rebase the current branch:

   ```bash
   git fetch origin main
   git rebase origin/main
   ```

3. Resolve conflicts in favor of the current durable contracts plus the reviewed Volta behavior. Re-run targeted compiler,
   golden configuration, serving, and kernel-digest tests after each substantive resolution.
4. Run `make test` and `make lint`, sync the exact rebased revision to the NVLink host, clear pre-rebase cubins, and repeat
   the focused MMA test plus deterministic generation. Because published results must name the final revision, rerun the
   complete M9 harness on the rebased commit and regenerate `article-manifest.json` and article tables.
5. If refreshed results change a claim, update the article before submission. Push with `--force-with-lease` only if the
   branch history was already published, then open/update the Emmy PR and the separate CloudRift PR with the final revision
   and evidence.

#### Exit gate

- The current branch is based on the latest `origin/main`; all checks pass; and the exact rebased revision produced the
  final NVLink benchmark manifest, tables, MMA proof, and short generation.

## Main risks and bounded responses

| Risk | Detection | Response |
| --- | --- | --- |
| CUDA/framework versions that know Qwen do not support SM70 | M0 image build/load probes | Pin CUDA 12.x and a compatible framework revision; record the exact fork. Do not patch Emmy around a broken external image. |
| FP16 cast is numerically unstable for the BF16 checkpoint | Layer/logit/generation comparisons | Quantify the first divergence; fix incorrect kernels first. If instability is inherent to FP16, report it and stop the final claim rather than changing precision silently. |
| `m8n8k4` lane mapping or four-operation warp semantics are wrong | Focused coordinate tests and sanitizer | Reduce to one tile, validate every output coordinate, and land no tuning work until it is correct. |
| Ordinary loads make the MMA path memory-bound | Scalar/MMA kernel evidence | Add synchronous shared-memory staging and tune it. Performance can remain modest; real HMMA execution and correctness are the hard gates. |
| Existing Ampere assumptions leak into Volta | SM70 illegal-source tests and SM80+ digests | Centralize capability checks and gate shape-specific passes; avoid target checks scattered through renderers. |
| Replacement host is not yet available | Phase 2 entry check | Finish and freeze Phase 1, write the handoff, then pause without inventing final results. |
| Replacement host still lacks usable NVLink | M8 topology and NCCL gates | Stop Phase 2 and request a corrected host; never substitute the Phase 1 PCIe machine for final numbers. |
| Final GPU identity differs from Phase 1 | M8 live-card identity check | Repeat M5–M6 into a GPU-specific file before benchmarking; do not replay foreign golden evidence. |
| PCIe collectives obscure Phase 1 bring-up | Phase 1 NCCL tests and serving traces | Prefer PP16, then PP8×TP2 or PP4×TP4; treat all current-host serving latency as diagnostic. |
| Golden tuner cannot search SM70 | M6 Path A smoke | Use bounded manual variant enumeration on actual in-model programs and preserve every candidate in JSONL. |
| No external engine runs the exact final setup | Three bounded Phase 1/2 probes | Publish absolute Phase 2 results and internal scalar/MMA evidence; frame the article around enablement. |
| Full model still cannot generate after compiler smoke tests | M4 scope ladder | Stop before tuning/article publication. A partial kernel demo is useful engineering evidence but does not satisfy the goal. |

## Primary technical references

- NVIDIA PTX ISA, warp-level `mma` instructions:
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma>
- NVIDIA Volta Tuning Guide: <https://docs.nvidia.com/cuda/archive/12.9.1/volta-tuning-guide/index.html>
- Qwen model card: <https://huggingface.co/Qwen/Qwen3.5-122B-A10B>
- Emmy compiler and pipeline contracts: `emmy/compiler/ARCHITECTURE.md`,
  `emmy/compiler/pipeline/ARCHITECTURE.md`, and `emmy/compiler/pipeline/passes/ARCHITECTURE.md`
- Article structure reference:
  `/Users/dikobraz/Projects/cloudrift-landing/packages/blog/content/blog/optimizing-gemma-4-12b-rtx`
