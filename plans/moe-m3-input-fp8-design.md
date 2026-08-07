# M3 design note — fp8 expert weights as program INPUTS through the serving seam

Context: in the MoE serving seam the expert weights are forward-argument inputs of the expert program
(`InputOp`s), not `ConstantOp`s — one program per layer kind, per-expert 2-D slices fed per launch. The fp8
lane's birth-time spelling (`emmy/compiler/loader/quant.py::spell_quantized_constants`) walks
`graph.loadable_constants()` only, so it can never fire on them. The question: how do fp8 bits + scales reach
the kernels when the weight is an input? Checkpoint under discussion:
`~/checkpoints/gptoss20b-fp8-emmy` (fp8e4m3, per-out-channel bf16 scales, compressed-tensors layout).

## Option (i) — dequantize at load into f16 device tensors

Upload `decode_f8(bits) * scale` as the bf16/f16 3-D tensors the runner uploads today. Zero compiler work.

**Rejected**: gpt-oss-20b is 20.9 B parameters; at 2 bytes/param that is ~41.8 GB of weights alone — misses
the 32 GB RTX 5090 (and the 24 GB 4090) before any KV cache or activations. It also erases the entire point
of the 22.1 GB checkpoint: fp8 storage exists to fit the card, not just the disk.

## Option (ii) — fp8 bits + scales as ADDITIONAL program inputs, dequant spelled in-graph  ← RECOMMENDED

Upload the fp8 bits (1 byte/element, uint8 carrier) and the per-out-channel scale tensors as extra program
inputs, and spell the dequant in the expert graph: `x @ (from_f8e4m3(w_bits) * scale)` (gpt-oss weights are
(in, out)-oriented, applied as `x @ W`). The W8A16 mul-hoist binding then absorbs the cone: the decode is
taken by the load's storage dtype and the scale multiply commutes out of the k-fold onto the accumulator
epilogue — the weight streams through the kernel as 1-byte elements.

VRAM: experts 19.1 GB (bits) + scales ~0.02 GB + non-expert bf16 ~2.9 GB ≈ 22 GB → fits the 32 GB card with
KV headroom.

### The exact gate in the mul-hoist binding — and why it already works for inputs

`emmy/compiler/pipeline/passes/lowering/tile/_atomize.py::_hoist_k_invariant_factors` (line 105) is
**provenance-blind**: it recognizes the loop-body STRUCTURE, never the graph-op kind. Its gates:

1. The lift operand's cone must factor as `(storage decode of that side's load) ⊗ k-invariant factors`,
   where ⊗ is a 2-arg multiply/divide chain (divide only with the weight on the left).
2. The chain's leaf is recognized by TRAIT: `ElementwiseImpl.decodes` — `_DECODES = {"from_f8e4m3":
   "f8e4m3", "from_f8e5m2": "f8e5m2"}` (`emmy/compiler/ir/elementwise.py` line 111) — with `st.args ==
   (bits,)` and `leaf.dtype is None or leaf.dtype.name == storage` (line 172–177). So the **Load's dtype
   must be the f8 storage dtype**, which holds iff the graph-level bits tensor carries dtype `f8e4m3` —
   true for an `InputOp` exactly as for a `ConstantOp`.
3. The factor must be k-free in the loop body (line 167) — per-out-channel scales are k-invariant ✓; a 2-D
   block scale is k-indexed and correctly declines to PLANAR.
4. Full body coverage (line 179–184) — every stmt accounted to a bound cone.

Nothing in `_atomize.py` needs to change. By the time the tile lowering runs, weights are `Load`s from
buffers; whether the buffer is constant- or input-backed is invisible. Per-launch expert slices keep the
kernel-facing operands 2-D (bits slice `(H, 2I)`, scale slice `(1, 2I)` riding the same per-expert slicing),
so the W8A16 warp-tier contract is untouched.

### What DOES need to change — the exact code gates

1. **Graph birth — an input analog of the constant spelling.** `spell_quantized_constants`
   (`loader/quant.py` line 288) iterates `graph.loadable_constants()`; `_spell_one` (line 159) builds the
   dequant fragment with two `ConstantOp` leaves. Needed: after tracing the expert program (traced with bf16
   weight inputs, as today), rewrite each quantized weight `InputOp` into a bits `InputOp` (dtype `f8e4m3`)
   + a scale `InputOp` + the same decode-cast / broadcast-multiply cone — `_spell_one`'s fragment with the
   two leaf ops swapped for `InputOp`s, everything else identical. Spelling it in the torch wrapper instead
   is NOT viable today: the tracer maps a dtype-changing `.to()` to an identity `IndexMapOp`
   (`trace/torch.py`, the "Pass-through" branch, ~line 838), never to the decode `ElementwiseOp`, and
   `_get_dtype` (line 320) would stamp the raw torch token `float8_e4m3fn`, not emmy's `f8e4m3`.
2. **Constant folding — no change, and no env var.** `032_fold_constant_subgraphs` dissolves CONSTANT dequant
   cones unless `EMMY_FP8_EXPAND` keeps them; an input-rooted cone is not a constant subgraph, so it stays
   in-graph unconditionally. The input path gets W8A16 semantics for free, with no `EMMY_FP8_EXPAND` analog.
3. **Runner input feed — per-input dtypes.** `emmy/serving/gen_runner.py::_compile_split` feeds every plan
   input through `t.detach().cpu().to(torch.float32).numpy().astype(np_dtype)` (lines 254, 266) — ONE
   homogeneous model dtype for all inputs. Bits inputs must bind as the uint8 carrier (a
   `torch.float8_e4m3fn` tensor `.view(torch.uint8)`), scale inputs as their own float dtype. The
   convention already exists on the constant side: `loader/safetensors.py` binds RAW BITS on a uint8 carrier
   whenever the graph dtype is an f8 token (line 159 comment) — the input-binding layer needs the same rule.
4. **Checkpoint pairing — non-`.weight` scale resolution.** Both `load_dequantized_state_dict`
   (`loader/quant.py` line 146–148) and `spell_quantized_constants` (line 292–298) are `.weight`-suffix
   bound: they require `key.endswith(".weight")` and pair `prefix + ".weight_scale"`. The gpt-oss expert
   params have no `.weight` leaf; the checkpoint pairs `<param>` + `<param>_scale`
   (`…experts.gate_up_proj` + `…experts.gate_up_proj_scale`). The general rule `scale_key = key + "_scale"`
   (and `+ "_scale_inv"`) subsumes the existing `.weight` → `.weight_scale` case — one relaxation covers
   both. The seam's own weight read (the runner pulling `experts.gate_up_proj` for upload) must read
   bits+scale from the shards rather than the transformers twin, since the twin's 3-D params cannot hold fp8.

## Option (iii) — variants considered

- **On-device dequant at load** into a bf16 arena: same 41.8 GB VRAM miss as (i).
- **Per-expert constant programs** (bits as constants, E programs/layer): already rejected by the plan —
  compile-count blowup, and it forfeits the one-program-per-layer-kind seam.
- **W8A8 for the expert matmuls**: quantize activations per-token and use the native fp8 mma tier (PR
  #470). `_hoist_k_invariant_factors` explicitly binds BOTH sides ("the W8A8 double-cone" — A-side factors
  are the activation scale), so the read side is covered; but the activation ENCODE (dynamic per-token
  quantize with a runtime absmax) has no in-graph spelling — round/clamp/absmax are outside the
  multiplicative decode form the arm binds. A real extension, not a freebie. Sequence it after (ii): (ii)
  is strictly upstream (same bits/scale inputs), and W8A8 then only adds the A-side cone.

## Recommendation

**Option (ii).** The expensive half — the warp-tier binding that turns the in-graph dequant into a 1-byte
weight stream with an epilogue scale — already exists and is provenance-blind (verified against the gate
conditions above). The remaining work is plumbing at known, narrow points: an InputOp variant of the
birth-time spelling (gate 1), per-input dtype binding in the runner feed (gate 3), and the `<param>_scale`
pairing relaxation (gate 4). Gate 2 costs nothing.
