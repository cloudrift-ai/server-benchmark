# V100 model-onboarding compiler gaps

Goal: make Emmy trace and preserve the exact computation and checkpoint representation of Laguna S 2.1 FP8,
DeepSeek V4 Flash 0731, and Ling 3.0 Flash FP8 on SM70. Keep model serving and distributed execution in 1Cat-vLLM;
the compiler owns trace semantics, kernel generation, tuning evidence, and a narrow kernel ABI at that boundary.

This plan follows the complete-model coverage contract added to the onboarding skill. A model earns a repository
golden only when every layer path and non-layer seam traces and lowers. A partial trace remains useful tuning and
compiler evidence under its experiment directory, but must not be presented as complete deploy evidence.

## 1. Current boundary and retained evidence

This PR already closes three gaps found during qualification:

- output liveness now prunes a dead local mutation while preserving a write through an alias of a returned value;
- Volta warp scheduling rejects a swept boundary-store epilogue that the fragment store cannot realize;
- repository goldens may retain a self-contained Loop IR target when frontend provenance cannot reconstruct the
  fused target, with the same measurement and validation requirements as an origin target.

Retain the exact model revisions, coverage manifests, failing FX slices, working goldens, target dumps, and SM70
lowering logs from the three experiments. Reduce each remaining failure to the smallest stable graph before changing
the compiler. Do not use a model-specific stub, discard a live operation, or dequantize a checkpoint merely to make
coverage appear complete.

Gate: every milestone below starts with a focused failing test derived from retained evidence and ends with the
relevant complete-model trace rerun. A local unit test alone does not establish model coverage.

## 2. M0 - functional view and mutation semantics

The exact Ling MTP token-shift seam exports `roll -> select -> fill_` and returns the mutated roll storage. Alias-aware
liveness correctly retains the write, but tracing currently stops at the rank-reducing `aten.select` result.

DeepSeek layer 2 separately exports `copy_` writes through destination slices and later reads the original base.
The current tracer now rejects that observable alias mutation instead of silently returning the pre-write base. This
milestone must make both mutation patterns functional before either model can earn a repository golden.

1. Lower `aten.select` as an IndexMap view that removes exactly one source axis and preserves the selected storage
   relation in trace metadata. Record each view-to-root affine index map, not only the shared storage root.
2. Functionalize `fill_` and the other observed scalar writes into an explicit updated value. A returned base value
   must observe a write through any chain of aliases; an unrelated local mutation must remain removable.
   Apply the same rule to DeepSeek's slice `copy_`: rebuild or remap the destination base so every later base or view
   read observes the update, including broadcast sources and multiple provably non-overlapping writes.
   Introduce an affine update/base-reassembly operation, or an FX functionalization pass that emits existing slice
   and concatenate operations. Version the root after each write and rebind later base reads and views created before
   the write while preserving untouched destination elements. Keep direct-return `copy_` as its narrower broadcast
   and cast operation.
3. Decompose `aten.roll` into finite-domain index algebra, or into slice and concatenate operations already
   representable by Graph IR. Cover positive and negative shifts, non-zero dimensions, unit extents, and wraparound.
4. Use exported FX execution order as the mutation sequence. Reject overlapping writes until their ordering and
   update semantics have an explicit representation; do not infer order from node names or user sets. M0 owns the
   static slice/select update subset, while general indexed scatter remains in M2.

Verification:

- eager, exported Torch, Numpy backend, Loop IR, and CUDA agree for the exact Ling token-shift geometry;
- eager, exported Torch, Numpy backend, Loop IR, and CUDA agree for the exact DeepSeek layer-2 slice updates;
- focused tests cover a returned alias mutation, a dead local mutation, a view-of-view write, and an unsupported
  overlapping write;
- focused tests cover two sequential non-overlapping writes, a later read through a pre-write view, destination
  casting and source broadcasting, and preservation of untouched base regions;
- the immutable Ling 3.0 revision traces the real token-shift seam without the diagnostic slice/concatenate adapter.

Exit: all Ling and DeepSeek layer classes and model seams lower on SM70, or each next exact unsupported operation is
preserved as a new partial-coverage boundary.

## 3. M1 - bounded recurrent-attention frontend

Ling's KDA path enters external FLA autograd and Triton autotuning during `torch.export`. The experiment used
equivalent Torch algebra to inventory the downstream kernels, but that diagnostic replacement is not yet a durable
frontend contract.

1. Define trace adapters for the exact ShortConvolution, KDA recurrence, and gated RMS normalization entry points.
   Each adapter must be ordinary Torch algebra or an opaque operation with a separately implemented Emmy semantic;
   it may not execute a Triton autotuner or read a FakeTensor data pointer during export.
2. Select adapters by architecture and callable contract, not by checkpoint repository text or a model-name branch
   in a lowering pass.
3. Compare each adapter against the immutable remote-code implementation on a live supported GPU over the deployed
   dtypes, sequence regimes, gates, initial states, and boundary values.
4. Keep the recurrent state as an explicit input and output so prefill, decode, and chunked execution cannot silently
   share different state semantics.

Gate: FP16 results meet the existing compiler tolerance, state tensors have identical shapes and update order, and
the trace contains no host data-pointer access or hidden runtime compilation.

## 4. M2 - live multi-output and indexed-update operations

DeepSeek's observed sparse-attention `topk -> scatter_` branch was dead and is now correctly removed. Live instances
of the same operations remain unsupported and need ordinary compiler semantics rather than a DeepSeek-only rewrite.

1. Extend the trace value map to represent tuple-valued operations and `operator.getitem` without losing dtype,
   shape, or provenance for either result.
2. Add a deterministic `topk` representation for the required values-and-indices contract, including tie behavior.
   If exact Torch tie ordering cannot be guaranteed, reject the operation instead of claiming parity.
3. Lower indexed scatter/update only for statically provable non-overlapping writes, or define the required atomic or
   ordered semantics explicitly. Preserve the destination initializer and scalar/source broadcast rules.
4. Add liveness tests proving that a dead impure branch is removed while a live values-only, indices-only, or
   values-and-indices consumer remains.

Exit: representative live sparse-attention and routing slices pass backend parity on SM70. Dead branches must still
emit no target and consume no tuning budget.

## 5. M3 - exact quantized checkpoint representation

Architecture-only traces currently prove the model programs and shapes, but not all stored formats. DeepSeek V4 uses
packed MXFP4 expert values with E8M0 scales and block-scaled FP8 dense weights. Laguna stores routed experts as
per-expert gate, up, and down tensors with independent block scales. Emmy must not describe either path as serving-
eligible while replacing those values with FP16 representative constants.

1. Extend constant spelling and checkpoint-key resolution for E8M0 block scales, packed MXFP4/FP4 expert values,
   and Laguna's per-expert projection layout. Keep scale tensors attached to the exact value block they describe.
2. Add typed Graph IR constants or quantized load operations that retain packed storage through lowering. Conversion
   to an accumulator dtype belongs in the generated kernel, not in checkpoint loading.
3. Define SM70 schedules for the retained representations only where the generated code has an executable Volta
   path. Reuse 1Cat's proven layout contracts where possible, but do not copy an engine-specific dispatch policy into
   the compiler.
4. Validate constant bytes, scale addressing, dequantized reference values, and end-to-end expert outputs against an
   independent loader on adversarial blocks, including zeros, saturation, subnormal scales, and tail blocks.

Exit: the exact checkpoint shards spell every traced quantized constant without a representation-changing fallback,
and representative dense, expert, and shared-expert outputs pass live SM70 parity. Until then, canonical architecture
goldens remain compiler coverage evidence rather than checkpoint-serving eligibility evidence.

## 6. M4 - trace-inventory reconstruction and promotion

Some fused Laguna targets had unique frontend origins but could not reconstruct the same fused target after reload.
Self-contained Loop IR made the working and canonical goldens honest; the trace writer should select that fallback
automatically rather than relying on post-processing.

1. During trace inventory writing, reload each proposed origin slice, run the normal frontend and Loop passes, and
   require exactly the expected structural target identity.
2. Persist origins only when that round trip succeeds. Otherwise persist the already-produced single-target Loop IR
   and record the fallback count in the trace summary.
3. Make target identity independent of incidental node IDs while retaining enough provenance for diagnostics.
4. Validate every written target by loading the final YAML and lowering it for the requested compute capability.

Exit: Laguna and DeepSeek complete inventories can be produced directly by `emmy trace` with no repair script, and a
round-trip test covers both origin and Loop IR targets.

## 7. M5 - measured 1Cat-vLLM kernel boundary

Emmy cannot replace 1Cat's scheduler, TP/PP collectives, paged KV cache, expert routing, or checkpoint loader. It may
generate missing leaf kernels once the exact quantized representation is supported.

The qualified DeepSeek TP8-by-PP2 server already reaches every intended native 1Cat route. Its 16 first-request
Triton compilations are a warm-cache coverage gap, not evidence of missing kernels. First add a 1Cat-owned image
warm/bake lane for representative prefill, decode, and tool-call shapes, then verify that those requests trigger no
runtime compilation. Emmy replacement remains optional and begins only after per-operation profiling.

1. Inventory the 1Cat DeepSeek and Laguna execution paths after a successful model load. Classify each missing or
   slow operation by inputs, outputs, layouts, streams, workspace, graph-capture requirements, and TP/PP ownership.
   Do not classify an operation as missing or slow from JIT events or whole-request latency alone.
2. Select one leaf operation with no hidden side effects and whose complete tensor contract already exists in Emmy.
   Only schema-declared output mutations are permitted. Avoid attention state updates, collectives, routing, or an
   operation whose layout is private to a fused 1Cat pipeline.
3. Define a versioned CUDA launch/cubin descriptor and the exact 1Cat Torch custom-operation schema used at the
   callsite. Include parameter order and types, input/output aliases and mutations, alignment, workspace, current
   device and stream rules, shape/layout constraints, capture safety, and both compiler and operator-contract
   revisions in the cache key. The caller supplies buffers and stream; the kernel performs no allocation.
4. Compare the Emmy kernel with the existing 1Cat path on identical live tensors, then measure the complete serving
   step under CUDA graph capture. A faster isolated kernel that regresses the server does not land.
5. Add an engine fallback so a missing, rejected, or stale Emmy artifact uses the native 1Cat implementation.

Exit: the image warm/bake lane verifies zero request-time compilation, and one profiled leaf operation passes
accuracy, capture, concurrency, and end-to-end serving performance gates. Only then generalize the boundary.

## 8. Requalification and delivery order

Run the milestones in this order:

1. M0 and M1, then rerun Ling complete trace. Promote a Ling repository golden only if every exact path and seam now
   passes; otherwise update its partial inventory and first blocker.
2. M4, then require the trace CLI to reproduce Laguna's existing 36 structural target identities and preserve its
   verified O3 rows. Regenerate or retune only if identities change. Retrace and remeasure DeepSeek after M0 because
   its excluded semantic target may change the downstream inventory; otherwise retain its partial experiment.
3. M3, then repeat checkpoint spelling and component-accuracy gates for Laguna and DeepSeek. This establishes Emmy
   representation parity, not Emmy-backed serving eligibility.
4. M2 when a retained model path or a minimal generic workload needs live tuple/index semantics.
5. Start M5 native-route inventory, profiling, and image warm-cache work from the already qualified 1Cat DeepSeek
   baseline. Gate a quantized Emmy replacement on M3; a representation-independent pure leaf may proceed earlier.

For each complete model, run the onboarding skill's equal-budget hybrid-versus-MCTS search over every retained target,
repeat deployable O3 correctness measurements, use a measured greedy fallback for search misses, and commit the final
golden under `emmy/compiler/pipeline/search/goldens/`. Reports must distinguish architecture coverage, checkpoint-
representation parity, component accuracy, and full serving accuracy; none implies the next.

## 9. Stop conditions

- Do not weaken liveness to discard a live mutation or unsupported operation.
- Do not add a model-name special case below a frontend adapter.
- Do not call an FP16/dequantized trace exact FP8 or FP4 checkpoint support.
- Do not move TP/PP, KV-cache, routing, or API responsibilities from 1Cat into Emmy to make a kernel demo work.
- Do not publish a repository golden for a partial exact trace.
- Stop a milestone when the remaining failure belongs to an external runtime contract; preserve its minimal
  reproducer and move it to the owning integration plan.
