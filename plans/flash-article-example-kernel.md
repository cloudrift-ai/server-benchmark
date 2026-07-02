# Flash article example kernel: K/V staging + pipelining on the tensor-core flash tier, pinned knobs

Goal: land the two remaining schedule-side flash optimizations — K/V operand staging (article Move 4) and async
transport + software pipelining (Move 5) — on the warp (mma) flash tier, so ONE example SDPA kernel with fp16
operands / fp32 accumulation runs the full move stack for the Part-2 article. The optimal config is found by a
MANUAL pinned-knob sweep; no prior / search / OptionFork work anywhere in this plan.

## The example kernel

- **Listing shape** (the article's running example, in fp16):
  `q=k=v=torch.randn(1,4,128,64, dtype=torch.float16); F.scaled_dot_product_attention(q,k,v)`.
  Warp-eligible today post-#300: `head_dim 64 % atom_k 16 == 0`, `d_v 64 % atom_n 8 == 0`, static `kv=m=128`
  divides every `(um, nt)` grid point (`_FLASH_WARPS × _FLASH_KEY_ATOMS = {1,2,4} × {2,4,8,16}`).
- **Perf shape** for benches: `(1, 8, 4096, 64)` f16 — same divisibility, enough work for stable numbers.
- **fp32 accumulation needs no work**: the `mma_m16n8k16_f16` atom accumulates into fp32 C fragments and the
  `(m, l, O)` carrier is fp32 (`_flash.py`). The article asserts it; step 4's accuracy table evidences it.

## Current gap (verified on `feature/flash-form-fork`)

`_schedule._twisted_warp_options` stamps only `TILE@<qk_k>` / `TILE@<pv_k>` / `REDUCE@<kv>` — no `STAGE` key —
and `_twist.realize_warp_twist` reads the K and V fragments gmem-direct on every streaming step. Every warp in
the CTA re-reads the same KV tile from global memory. `passes/ARCHITECTURE.md` names this Stage seam (and the
causal tile-skip) as the flash follow-ups.

## Step 1 — scheduler: resolve + stamp a KV `Stage` on the warp-flash rows

- New `_resolve_twisted_stage(red, head, pv, stage, budget)` in `_schedule.py`, mirroring `_resolve_warp_stage`:
  the slabs are the K tile `(bn × head_dim)` and V tile `(bn × d_v)` in the operand dtype (2 B), so
  `slot_bytes = bn · (head_dim + d_v) · 2`; clamp `depth` to the smem budget (dropping `ring` when nothing
  cycles), `reg_depth` to the resident chunk. cp.async eligibility mirrors `_can_stage_warp` with the STREAMING
  axis in the K-axis role (static, block-divisible kv). **TMA is rank-gated out** — the batched K/V `Load` index
  has more gmem dims than the 2-D descriptor box — so the transport is cp.async, which is exactly the article's
  sm_120 rung ("this rung is `cp.async` + `mma.sync`").
- Stamp `STAGE@<kv>` (the resolved spelling) on each warp row; stamp decided-empty `""` on the gmem-direct warp
  row and on the chain / coop / scalar escape rows, so every flash leaf spells the same key set (the fork
  invariant `_twisted_warp_options` already documents for TILE/REDUCE).
- Unpinned rows enumerate `["", "d1/cp", "d2/cp/ring", "d3/cp/ring", "d2/cp/ring/p2"]` resolver-gated (a subset
  of `stage_moves(warp=True)`; no TMA rows, see above). The `EMMY_STAGE` pin stays authoritative and follows
  the standard pin-validity degrade with a log line.

Verify: `--ir tile -vv` on the listing shape shows the staged rows with resolved spellings; a non-divisible-kv
shape and an fp32 shape show no staged rows.

## Step 2 — materializer: the staged streaming K-loop in `_twist.py`

- Thread the resolved `Stage` into `realize_warp_twist`: per kv step, fill the K/V slabs through the existing
  `fill → commit → wait → drain → Sync` seam (`CpAsyncTransport`, `_stage.py`) and repoint the Q@K B drain and
  the P@V V drain at the slab via the existing ldmatrix slab drain (`_staged_inner_atom_loop`) — coalesced
  slab reads replace the per-lane gmem fragment loads.
- `d1` single-buffer is the Move-4 listing (stage); `d2+/ring` is the Move-5 listing (transport + pipeline):
  prologue primes the ring, steady state prefetches KV tile `i+1` under tile `i`'s mma work, epilogue drains —
  the `staged_kloop` phases composed over the streaming reduce instead of a contraction K-loop.
- **Do not fork a second pipeline skeleton.** If `staged_kloop` can't be driven wholesale over the twisted
  reduce (its skeleton assumes a contraction drain), extract its phase helpers and call them from the twist
  realizer's kv loop — one `Transport` seam either way.
- Invariant carried over from the matmul tier: staging is a pure perf transform — the staged kernel is
  bit-identical to its gmem-direct sibling.
- Smem budget note: the C→A handoff slab (`flash_pv_smem`) shares the pool; at `(um=1, nt=16)` (bn=128) a d2
  ring is 64 KiB and must clamp to d1 under the 48 KiB static floor — the resolver clamp from step 1 handles
  it (or the dynamic opt-in cap when a `Context` reaches the schedule).

## Step 3 — tests

- Extend `tests/compiler/e2e/test_attention_coverage.py`: staged warp-flash accuracy vs torch across
  {`d1/cp`, `d2/cp/ring`} × {static, symbolic seq} × {plain, causal, gqa}; one bit-identity check
  staged-vs-unstaged; decline cases (fp32, non-divisible kv, additive mask) still lower via their old tiers.
- `make test` (the -O1 correctness lane), `make lint`.

## Step 4 — manual sweep → the article's pinned config + numbers

On the RTX 5090 (sm_120), perf shape, via env pins (flash fusion is the `PLACE` default — no gate pin needed;
the keyed pins ride `EMMY_KNOBS="TILE@<qk_k>=a:mma_m16n8k16_f16/…,STAGE@<kv>=d2/cp/ring"` since `@` is not
shell-legal), `emmy run --bench` per point:

- Geometry `(um, nt) ∈ {1,2,4} × {2,4,8,16}` × stage `{"", d1/cp, d2/cp/ring, d3/cp/ring, d2/cp/ring/p2}`
  (resolver-declined points skip themselves). Record the µs table in this plan as it fills in.
- The winner becomes the article's `$KNOBS`. Produce the per-move latency ladder the article narrates:
  Move 1 scalar streaming → Move 2 mma gmem-direct → Move 4 staged d1 → Move 5 pipelined d2+,
  each vs torch SDPA (flash backend), eager and `torch.compile`; `flash_attn` (FA-2) if installed.
- Accuracy table for the Numerics placeholder: scalar fp32 flash vs f16-mma flash, max abs error vs an fp64
  reference, listing + perf shapes.
- NCU on the winner (the tune-model skill's per-kernel flow) to attribute any remaining gap. The C→A smem
  handoff is the known suspect — **measure and report, don't fix**, unless it dominates the profile.

## Step 5 — docs + article handoff

- Update `passes/ARCHITECTURE.md` + `lowering/kernel/ARCHITECTURE.md`: the Stage seam now covers the twisted
  tier; the flash follow-up list shrinks to the causal tile-skip; CLAUDE.md's tile-lowering blurb likewise.
- Hand the article (cloudrift-landing Part 2) its data: the knob strings, the latency ladder, the accuracy
  table — plus three stale-claim fixes: static block-divisible shapes DO atomize now (the "static-shape …
  scalar" sentence); the cited test path is `tests/compiler/e2e/test_attention_coverage.py`; and flash fusion
  is no longer gated behind `DEPLODOCK_FLASH=1` — `PLACE`'s built-in `auto` resolves to fuse everywhere, so
  greedy ships the fused kernel by default and `PLACE@fold=cut` is the multi-kernel escape.
- Delete this plan file when landed.

## Out of scope

TMA + WSPEC on flash (rank-gated / needs TMA — the article's sm_90 story), causal tile-skip (the running
example is non-causal), split-KV / the decode-combine kernel (Move 6 is demonstrated on the matmul tier),
the C→A register-shuffle handoff (measured only), and ALL prior / search / two-level-tune work.
