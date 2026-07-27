# Plan: first-class multi-output nodes — the MIMO graph foundation

## Status / context

Decision (2026-07-23): before the tap-only stat fusion (`stat-tap-loop-fusion.md`), build proper multi-input,
multi-output node support as a standalone foundation. Multi-INPUT already exists (`Node.inputs` is a list); the work
is first-class multi-OUTPUT and the retirement of the two workarounds that stand in for it today:

- `AuxOutputOp` (`ir/base.py`) — a zero-launch sentinel node minted per extra buffer (`025_sink_row_reduce`'s `__sq`),
  so the buffer gets a `BufferSpec`/lifetime while the PRODUCER's launch actually writes it — dual bookkeeping nothing
  validates;
- component packing — `030_split_reduce` fattens one workspace with a leading `comp` axis for flash's `(m, l, O)`
  state precisely because "no multi-output kernel" (its own comment).

Clients this unlocks, in order: (1) the stat tap (`T` + `T__sq` as two outputs of one node — the rebased tap plan);
(2) fusing producers that are graph outputs — `merge_loop_ops` today refuses them ("producer is a graph output — it
must stay materialized", the escaping-residual `(rms_norm(x + r), x + r)` shape) because the splice consumes the
node wholesale; with MIMO the merged kernel keeps both buffers; (3) the shelved whole-B merge / megakernel direction;
(4) optionally, typed per-component split workspaces instead of the `comp`-axis pack.

## Survey findings that shape the design (2026-07-23 session, three parallel code surveys)

The single-output assumption is a **convention, not an enforced invariant**, and it concentrates:

- `Node.output: Tensor` is the only singular field (`graph.py:119`). `add_node` merely DEFAULTS `node_id` to
  `output.name` when free (`graph.py:420-424`); nothing asserts equality. `Graph.splice` already has the dict-form
  multi-redirect (`output={old_id: frag_out_id}`, `graph.py:571-598`) and an id-promotion step; `Match.output`
  already speaks the same dict form (three sites use it today).
- The `.output` census: 235 sites in `emmy/compiler` — only **27 are identity** (`.output.name`); ~125 are mechanical
  shape/dtype reads; ~83 pass the Tensor through. Producer lookup is `graph.nodes[buf]` (~38 traversal sites) and
  `graph.users`/`consumers` (10 sites, node-granular, buffer-blind).
- **The launch/runtime layer is already buffer-plural**: `LaunchSpec.arg_names` / `zero_outputs` / `zero_prologues`,
  the `arrays` by-name binding, the memset loop, and `compute_live_intervals` all operate on buffer-name sets;
  `zero_outputs` already counts as a buffer's first write. The singular seams downstream of the graph are exactly:
  `BufferSpec` minted one-per-node (`plan.py:145-148`), `LaunchSpec.node_id` as the first-write key
  (`_planner.py:51`), `single_node_graph`'s sole-output slice (`slice.py:112`), and the eager accuracy reference
  collapsing tuples to `out[0]` (`run.py:1999-2000`).
- **Fork/evidence identity is safe**: `op_cache_key` is name-invariant (kernel name replaced by `_K_`; Loop/Tile keys
  are `digest(type, body.structural_key(), knobs)`), so golden/tune rows key on structure + knobs, never node ids.
  Knobs ride the `Op`; the 1:1 rebind merge-forward (`candidate.py:192-197`) is node-count-agnostic.
- The strongest conflations, i.e. the NON-mechanical sites: `Op.populate_io` hardcoding
  `outputs = {node.id: node.output}` (`base.py:73`); `BodyOp.populate_io` requiring every written buffer to BE a
  graph node (`stmt/ir.py:120-123`, the `stray_out` check) — the very rule that forces `AuxOutputOp` to exist;
  `add_node`'s id defaulting; `_rename_buf_in_op` / `replace_node`; the buffer-role derivation `graph.node_role`
  being per-NODE when a MIMO node can produce one graph-output buffer and one scratch buffer.
- Already plural-shaped and needing nothing: `Op.outputs` (a dict), `BodyOp`'s multi-`Write` output derivation, the
  loop splicer's per-`(node, buffer)` target selection (`splicer.py:353-365`), `restamp_structural_features`
  (counts `len(body.writes)`), provenance/reproducer op→kernel association (already many-ops-per-kernel,
  buffer-agnostic), multiple GRAPH outputs (tracer + oracle handle them today).

## Design

### Representation

- `Node.outputs: tuple[Tensor, ...]` (non-empty, ordered). `Node.output` becomes a read-only property returning
  `outputs[0]` — the **primary** output — so ~208 of the 235 call sites don't change at all.
- **Buffer names are the edge currency.** `Node.inputs`, `Graph.inputs`, `Graph.outputs`, `Load.input`,
  `Write.output` all keep their string type; their meaning shifts from "node id" to "buffer name" (identical today).
  The IR dialects are untouched.
- **Primary-name convention, kept and enforced**: `node.id == node.outputs[0].name` in steady state (splice's
  transient fragment-id window stays as today). This preserves kernel naming (`k_<node.id>` fallback), dump paths,
  and log readability. Non-primary buffers carry their own names (`<primary>__sq` style).
- New graph indexes, maintained in lockstep by every mutation method exactly like `_users` is today:
  - `_producers: dict[str, tuple[str, int]]` — buffer name → (node id, output slot). API: `graph.producer(buf) ->
    Node`, `graph.buffer(buf) -> Tensor`.
  - `_users` becomes buffer-keyed: `dict[str, set[str]]` buffer → consumer node ids. API: `graph.buffer_users(buf)`;
    `graph.users(node_id)` stays and returns the UNION over the node's buffers (today's semantics for single-output
    nodes, so existing call sites keep their behavior — each of the 10 sites gets audited for which granularity it
    means; `007_sink_narrowing_cast`'s "wide value still live" check is the one that must become buffer-granular).
- Graph-level SSA invariant: every buffer name has exactly one `(node, slot)` producer; `add_node` rejects a
  duplicate buffer name anywhere, not just a duplicate node id.

### Op protocol

- `Op.populate_io` default: `outputs = {t.name: t for t in node.outputs}` (the dict was always plural-shaped).
- `BodyOp.populate_io`'s `stray_out` check inverts to the single-source-of-truth rule: every `Write` buffer must be
  one of the NODE's declared output buffers (or a matcher-known external); for body-bearing ops, `node.outputs` is
  derived/validated from the body's Writes. This is the "derived, not dual-bookkept" hardening — the producer's
  written-buffer set, its `arg_order`/`zero_outputs`, and the graph's view can no longer disagree.
- `infer_output_shape` → `infer_output_shapes -> tuple[tuple, ...]` with a default that wraps the old single-shape
  hook, so the ~30 frontend ops don't all churn.
- Oracle `forward` may return a tuple, matched positionally to `node.outputs`; `Backend.run` stores per-buffer
  `values[buf]` (today `values[nid]` — same key for single-output nodes).

### Rewrite protocol (matcher / splice)

- `Match.consumed` stays node ids; `Match.output` redirect keys/values become buffer names — backward compatible,
  since today every buffer name IS a node id. The dict form gains the one new meaning MIMO needs: a consumed node's
  individual buffers can be redirected to different fragment buffers.
- `Graph.splice` consumer rebind (`replace_node` + `_rename_buf_in_op`) rewires per-buffer instead of per-node —
  mechanical, the machinery already renames buf refs inside `LoopOp` bodies by string.
- The chain walker (`consumers[0] if len(consumers) == 1`) keeps node-granular semantics unchanged: a multi-output
  producer with two distinct consumers is a fan-out and ends the chain, which is correct — rules that want a specific
  buffer's edge enumerate `buffer_users(buf)` themselves (the `005_split_shared_indexmap` idiom).
- `frag.add_node(..., outputs=[...])`: fragment rules may now emit true multi-output nodes; `wrap_merge_fragment` /
  `rename_write_output` rename per buffer.

### Backend / planner

- `BufferSpec` minted per BUFFER: iterate `node.outputs`; role computed per buffer (`buf in graph.outputs` →
  "output", else by the producing op's role) — replacing the per-node `node_role` at this call site.
- `LaunchSpec` gains `writes: tuple[str, ...]` (the produced buffer names); `compute_live_intervals`' first-write
  test becomes `buffer in ln.writes` (the `zero_outputs`-as-first-write rule is untouched). `node_id` stays for
  naming/diagnostics only.
- `AuxOutputOp` is deleted (phase 2); the `plan.py` skip-list shrinks; the aux buffer becomes output slot 1 of its
  producer, and everything downstream already works by name.
- Accuracy: `_eager_output` stops collapsing to `out[0]` — compare each graph output buffer against the eager
  tuple positionally. (A latent correctness gap today; fixing it is independent value.)
- `single_node_graph` reproducer slices list ALL of the node's buffers as slice outputs.

### Serialization / dumps

- Node schema: `"output": {...}` → `"outputs": [{...}, ...]`; `from_dict` reads BOTH forms (old dumps stay
  loadable); `to_dict` writes the new form only. `structural_key` folds all outputs in slot order.
- Kernel naming, provenance, `<kname>.torch.json` slicing: unchanged by design (all name- or provenance-keyed).

### Deliberately unchanged

The fork/knob engine, `op_cache_key`/golden keys, `restamp_structural_features`, the `Match` dataclass shape and
`is_alive` identity snapshots, the pass-scan fixpoint, `Load`/`Write` stmt types, and the graph-outputs handling in
the tracer/oracle. No pass changes behavior in phases 0–1; goldens must not notice this refactor happened.

## Decision points (resolve during phase 0, in this order)

1. **`graph.users` audit** — for each of the 10 call sites: node-union or buffer-granular? Default union (behavior-
   preserving); flag-and-fix the sites where union is semantically wrong once MIMO nodes exist (known: `007`'s
   liveness check; the fusion walker is correct as union).
2. **`inputs` as buffer names vs `(node, slot)` refs** — buffer names, firmly: the IR already speaks them, and the
   `_producers` index recovers the slot. Revisit only if buffer renames become a bottleneck (they won't; renames
   already exist).
3. **Where `Graph.validate()` runs** — there is no validator today; this plan adds one (SSA-per-buffer, producers/
   users index consistency, primary-name convention, body-Writes ≡ declared outputs). Run it in tests always and
   behind a debug env var at pass boundaries; never in production compile paths.

## Migration phases (each lands green: `make test` + `make lint`, goldens `MATCH 105/0/0` where GPU-checked)

- **Phase 0 — alias + indexes, zero behavior change.** `Node.outputs` tuple + `output` property; `add_node` accepts
  `output=` (wraps) or `outputs=`; build `_producers` and buffer-keyed `_users`; introduce `graph.producer(buf)` /
  `graph.buffer_users(buf)` and migrate the ~38 producer-lookup sites and the audited `users` sites; add
  `Graph.validate()` + unit tests. Every graph is still single-output-per-node; serialized form unchanged.
- **Phase 1 — plural plumbing.** Serialization (`outputs` list + read-compat); per-buffer `BufferSpec` + per-buffer
  role; `LaunchSpec.writes` + planner first-write; oracle per-buffer values + tuple `forward`;
  `infer_output_shapes`; `populate_io` defaults; `BodyOp` `stray_out` against declared outputs; the eager accuracy
  multi-output fix. Still no multi-output node exists in any real graph — covered by synthetic unit graphs.
- **Phase 2 — first real client + workaround retirement.** Port `025_sink_row_reduce` to emit `T__sq` as output
  slot 1 of the producer node (dict-redirect exercised; the manual `match._identities` poke and `AuxOutputOp` node
  both disappear); delete `AuxOutputOp`; `single_node_graph` multi-buffer slice; GPU gates: golden eval on both
  cards, stat-sink twin e2e retention (post-attn m32), accuracy twins.
- **Phase 3 — unlocks (separate plans).** Rebase `stat-tap-loop-fusion.md` on true multi-output (its WS1/WS2 aux
  bookkeeping simplifies to "second output"); relax `merge_loop_ops`' graph-output-producer refusal (fuse
  `(rms_norm(x+r), x+r)` keeping both outputs — needs the splicer to keep the producer's `Write`, a small
  `splice_loop_ops` extension); optionally revisit `030`'s `comp`-axis packing. Each is its own measured change,
  not part of this foundation.

## Risks

- **The `users`-union ambiguity** is the subtle-bug class: a site that meant "who reads THIS buffer" silently reads
  "who reads any of them" once nodes have two. The phase-0 audit + validator are the mitigation; grep-able because
  all 10 sites are enumerated.
- **Old-dump compat**: `from_dict` dual-read covers `EMMY_DUMP_DIR` artifacts and `emmy compare` across the
  boundary; tune DB / goldens are structural-keyed and unaffected.
- **CUDA graph capture**: per-buffer specs can change slab layout ordering (`(-size, name)` is deterministic but the
  name set changes when aux nodes disappear); capture already invalidates on rebind — verify the pack-hit gate on
  the gemma-4 verify flow in phase 2.
- **Silent behavior drift in phase 0's mechanical sweep**: 27 identity sites + 38 traversals is small enough to
  review by hand; the census (this plan + session transcript) is the checklist.

## Sizing

Census totals: 235 `.output` reads (27 identity / ~125 metadata / ~83 pass-through), ~38 producer traversals, 10
`users`/`consumers` sites, 8 `match.output` sites, 5 backend chokepoints, ~30 `infer_output_shape` implementors.
Phases 0–1 are the bulk and are mechanical against the enumerated lists; phase 2 is one pass port plus deletion.
The nontrivial-by-hand list is exactly: `add_node`/`splice`/`replace_node`/`_rename_buf_in_op` (graph.py),
`Op.populate_io` + `BodyOp.stray_out`, `BufferSpec`/`_planner` first-write, per-buffer roles, eager accuracy, and
the `users` audit.
