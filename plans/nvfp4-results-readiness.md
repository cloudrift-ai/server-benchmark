# NVFP4 results readiness, by command

Kept current on the `feature/nvfp4` branch. PR #499 links here rather than carrying the table inline, because the
table is wider than a pull-request description renders comfortably.


Every deliverable of this PR, expressed as the command that demonstrates it: what the command should produce when the
PR is done, and what it produces today. Cells carry a marker so a reader can tell evidence from expectation.
**OBSERVED** — seen to run, with the run named. **CHECKED** — a fast probe or artifact inspection. **CITED** —
recorded earlier in this PR or its notes, not re-verified. **SPECULATED** — read from code, with the grounding file
named. *unknown, not checked* where that is the truth.

Pre-PR main is one fact repeated, so it is stated once here instead of in a column: main has no NVFP4 at all — zero
`f4e2m1x2` in `emmy/compiler/dtype.py`, zero `nvfp4` in `emmy/compiler/loader/quant.py`, which recognizes only fp8,
AWQ and EXL3. Every row below would therefore fail on main at load, on the packed `[N, K/2]` shape. That is SPECULATED
from those two files, not run.

Branch observations come from an RTX 5090 (sm_120, CUDA 13.0) unless noted. The `--ir` rows cite a six-file layer-0
dump set under `tmp-claude/ir-dumps/`, produced by earlier IR-explainer work and inspected directly for this table.
Sync #5 with main brought #634, which reprints the torch and tensor stages as Rust-like pseudocode, so the token
counts the `--ir torch` and `--ir tensor` rows cite are spelled differently now; what those rows expect is unchanged,
and rendering the NVFP4 decode cone through the new printer confirmed it. The `loop`, `tile`, `kernel` and `cuda`
citations are untouched.
`emmy compile -c "<torch expr>"` cannot express these rows: an inline expression has no checkpoint, and NVFP4 operands
exist only because the birth-time speller rewrites checkpoint constants — so the narrow rows sharpen through `--layer`
and `--seq-len` instead.

| area | command | when the PR is done | now, on the branch |
| --- | --- | --- | --- |
| loading | `emmy pull nvidia/Qwen3-8B-NVFP4` | downloads and classifies as NVFP4/modelopt, proving the loader reads both config conventions | **CHECKED** — `is_nvfp4_checkpoint=True`, summary `nvfp4 modelopt`, 6 GB cached |
| compiling | `emmy compile nvidia/Qwen3-8B-NVFP4 --layer 0 --ir torch` | packed constants at checkpoint shapes, unwidened — the stored element is the packed byte pair | **CHECKED** — `layer0.torch.txt` |
| compiling | `emmy compile nvidia/Qwen3-8B-NVFP4 --layer 0 --ir tensor` | the decode is ordinary graph algebra — a pair-table gather, no bespoke op | **CHECKED** — `layer0.tensor.txt`: 7 `f4e2m1x2`, 7 `gather`, 7 `from_f8e4m3`, 14 `f4_pairs` |
| compiling | `emmy compile nvidia/Qwen3-8B-NVFP4 --layer 0 --ir loop` | the packed dtype survives fusion; no full-width weight materialises | **CHECKED** — `layer0.loop.txt`; **0** `(4096,4096) f16` / `(12288,4096) f16` buffers |
| compiling | `emmy compile nvidia/Qwen3-8B-NVFP4 --layer 0 --ir tile` | resolved pins show the packed byte-slab stage chosen | **CHECKED** — `layer0.tile.txt`: `mma_m16n8k16_f16_f32` x4, `d1/smem-tma` x3, `w1x4` x3, `w8x2`, `d1/smem` |
| compiling | `emmy compile nvidia/Qwen3-8B-NVFP4 --layer 0 --ir kernel` | three-slab staging visible as packed-drain annotations | **CHECKED** — `layer0.kernel.txt`: 35 `LdmatrixLoad` drains |
| compiling | `emmy compile nvidia/Qwen3-8B-NVFP4 --layer 0 --ir cuda` | emitted CUDA carries the `_f4s_` drain and the 16-entry e2m1 LUT, with the scale applied in the drain | **CHECKED** — `layer0.cuda.txt`: `EMMY_F4_LUT_F16[16]` (line 404), `emmy_mma_load_b_smem_trans_f4s_f16` (line 409); 13 `f4s_`, 9 LUT hits across 3501 lines |
| compiling | `emmy compile nvidia/Qwen3-8B-NVFP4 --layer 0 --seq-len 512 --ir tile` | at a prefill width the pin names the block-scaled atom `mma_m16n8k64_e2m1_f32`, with **both** operands `f4e2m1x2` — the activation-quantize step visible ahead of the matmul | **not yet** — the atom exists and is device-verified, but nothing offers it to the scheduler (see the PENDING DECISION entry under Known gaps). `_warp_atoms` declines any operand whose dtype has `logical_elems != 1` |
| compiling | `emmy compile nvidia/Qwen3-8B-NVFP4 --layer 0 --seq-len 512 --ir cuda` | emitted CUDA carries the PTX `mma.sync.aligned.m16n8k64...kind::mxf4nvf4.block_scale...ue4m3`, both multiplicands e2m1 | **not yet** — same PENDING DECISION. The wrapper and its scale-fragment loaders exist in `ir/kernel/render.py` and are device-verified; nothing reaches them |
| compiling | `emmy compile nvidia/Qwen3-8B-NVFP4` | all 36 layers compile | **unknown, not checked** — serving exercised this code; the command itself never ran |
| parity | `emmy run nvidia/Qwen3-8B-NVFP4 --layer 0` | per-layer output matches torch | **OBSERVED by another route** — parity ran through `EmmyGenRunner`: post half 1.95e-3, pre half q/k/v bit-identical. This command is **unknown, not checked**; its arity mismatch is listed under Known gaps |
| parity | `./venv/bin/pytest tests/compiler/passes/test_nvfp4_staged.py -k "packed_drain"` | the drain equals `dequantize_nvfp4` on device, including split-K slices, over both copy transports | **OBSERVED** — `..._matches_the_decoded_oracle` and `..._addresses_its_own_split_k_slice` pass |
| benching | `emmy run nvidia/Qwen3-8B-NVFP4 --bench` | per-kernel packed-vs-fill speedups against eager and `torch.compile` | **CITED** — the RTX 4090 and 5090 tables below; not re-run for this table |
| benching | `./venv/bin/python scripts/profile_gen_decode.py --model nvidia/Qwen3-8B-NVFP4 --bucket 16` | a coded-trunk decode step that completes and approaches the decoded path; today's 58431x-over-roofline pick becomes a near-roofline one after tuning | **OBSERVED, half** — decoded trunk 160.03 ms/step, GPU 146.17 ms (91.3%), host 13.86 ms. Coded trunk: **no number**, two runs (50 and 90 minutes) timed out, 2483 log lines of one program window past 69 s |
| serving | `emmy serve nvidia/Qwen3-8B-NVFP4 --generate` | the runner picks the coded trunk unprompted, the packed drain reaches the kernels, output matches torch, and the server returns coherent tokens | **OBSERVED for the first three** — chose `compress_trunk=True`, trunk `codes`; 4 of 35 kernels carry `f4s_`; post half 1.95e-3, rel 3.67e-4. Coherent generated text **unknown, not checked** for this table (**CITED** from an earlier run) |
| serving | `emmy serve nvidia/Qwen3-8B-NVFP4 --generate --bench` | TPOT approaches or beats stock vLLM | **OBSERVED — not obtainable** while the coded path cannot be profiled. Whether stock vLLM loads this checkpoint at all is **unknown, not checked** |
| tuning | `emmy tune <TWIN>.json --bench` (TWIN from `scripts/capture_gen_twins.py`) | DB rows and an online prior for the packed serving programs, moving deploy picks toward the roofline | **CHECKED** — not started; nothing tuned on this GPU. The gap is **OBSERVED**: the untuned pick sits **58431x** over the roofline floor |
| tuning | `emmy eval knobs` (siblings: `emmy eval online`, `emmy eval golden`) | packed rows visible in the priors, distinguished without a new row stamp | **unknown, not checked** — all three subcommands exist in `emmy/commands/eval.py` |
| hybrid model | `emmy compile nvidia/Qwen3.6-27B-NVFP4 --layer <N>` (N: a full-attention layer and a linear-attention layer) | both carves compile and reproduce real layers exactly | **CITED** — max_abs 0.0, per this PR's own record; not re-verified. Checkpoint **CHECKED** present, 21 GB |
| hybrid model | `emmy run nvidia/Qwen3.6-27B-NVFP4 --layer <N>` | parity for both layer kinds | **unknown, not checked** |
| hybrid model | `emmy serve nvidia/Qwen3.6-27B-NVFP4 --generate` | full-attention layers on emmy, linear-attention layers on stock, generating coherently | **SPECULATED** (`emmy/serving/vllm_model_gen.py` plus the Known gaps entry) — the serving-side half is unwritten |

**What the counts say.** Of twenty-one rows, nine rest on runs from this session and seven on artifacts inspected
directly; six are unknown and three are inherited claims not re-verified here. Four of the green rows are green
because *serving* exercises that code rather than because the named command ran — `emmy compile`, `emmy run` and `emmy
eval` have not themselves run against NVFP4. Every correctness and structure property has a row that would demonstrate
it; the efficiency column is the thin one, and the coded-trunk TPOT stays unobtainable until tuning lands.

**Coverage.** Phase 1 maps to the `--ir torch` row; phase 2 to `emmy pull`; phase 3 to `--ir cuda`, the pytest oracle
and `emmy run --bench`; phase 4 to the two `--seq-len 512` rows; phase 5's model side to the hybrid compile row and
its serving side to the hybrid serve row; phase 6 to the serving row. Among the design decisions, 1, 2, 4, 6 and 7 map
onto the `--ir` rows, 3 onto the pytest oracle, 8 onto `emmy eval knobs`, 9 onto hybrid compile, 14 onto the two
block-scaled rows, and 15 onto the serving row, since that fix is what makes it correct.

Four decisions are deliberately not command-expressible — 5 (recognition is conservative and marker-first), 11 (one
flag names the arch-suffixed target), 12 (probes settled the scale-fragment layout) and 13 (adopting main's #513).
They are design positions and merge choices, attested by code and tests rather than by any command's output. The
golden re-seed and the digest-baseline regeneration ride with the `emmy tune` row; release imaging is outside this PR.

So the implication holds: when every *when the PR is done* cell above is green, this PR's stated goals are met, with
those four decisions attested by review rather than by running anything.
