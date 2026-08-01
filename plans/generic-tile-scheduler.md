# Generic tile scheduler — demand-driven recursive enumeration

The tile IR is compositional (`Map` / `Fold` / `Contraction`, operand edges materialized or computed inline) and the
schedule output is compositional (`TileOp.schedule`, per-node slices keyed by the tree-path codec: `TILE@dd`,
`TILE@pj`, `REDUCE@<axis>`). Only the middle is not: `_schedule.schedule()` dispatches once on the root's role and
then each composite tree shape owns a hand-written whole-tree path — `twisted_pair` + `_twisted_warp_options` for
flash, `bind_prologue_contraction` + `_computed_a_rows` for the fused-cone contraction, `_demoted_warp_option` /
`_demote_planar` as scattered backtracking, and pin narrowing re-implemented per path (`_narrow_flash_forms`, the
REDUCE-pin escapes). Every new composition (nested folds, a second computed operand, gated attention) means a fourth
hand-written path. This plan replaces the whole-tree paths with ONE recursion over the stored term.

## Design

Schedule the tree the way the IR is built: per-kind local move generators, a typed interface between parent and
child, and a unification step for the kernel-global inventory.

```
def candidates(node, demand, ctx) -> list[NodeSchedule]:
    local = MOVES[type(node)](node, demand, ctx)          # per-kind move generator
    out = []
    for cand in local:
        subs = [candidates(child, cand.demand_for(edge), ctx)
                for edge, child in computed_edges(node)]   # recurse into computed operand edges
        out += [unify(cand, combo) for combo in product(*subs) if unify(cand, combo)]
    return out
```

### `Demand` — the edge interface

What a parent requires of a child's result. The concept the current code has implicitly but never names; naming it is
the whole refactor. Fields:

- **bound axes** — the placement slice at this point (already modeled: `TilePlan.at` binds the caller's `(m, n)`).
- **residency** — where the result must land: gmem cell / smem slab / register fragment / per-thread scalar.
- **step granularity** — whole-axis, or *per-block of the enclosing fold's axis* (block width a candidate parameter).
- **budgets** — remaining smem / register / thread headroom (the CTA-wide inventory constraint, shared down the walk).

Flash is nothing but the fold offering a streaming candidate whose QK-edge demand is "a (bm × bn) score block per
kv-block, fragment residency"; norm→linear is a contraction whose computed-A demand is "a k-block slab via the sync
compute-fill". Today those demands are baked into `_twisted_warp_options` and `_computed_a_rows` respectively; named,
they are two instantiations of one interface.

### `NodeSchedule` — one composed candidate

A candidate carries: the schedule slices it decides, each keyed by its node's tree path (the existing codec — the
stamped spelling IS the stored/golden spelling, unchanged); its **worker demand** (`Workers`: kind, units, producer
band); its **smem cost**; and the demand it imposes per computed edge. Composition = slice-dict union + worker join +
budget subtraction.

### Per-kind move generators (local, tree-shape-blind)

These mostly exist and need only the tree-shape awareness stripped:

- **`Map`** — grid-map the free axes + the strip tile (`_map_strip_fork`). A `Map` with a node source delegates: its
  `fn` is the projection the source's candidate must be able to carry (the fragment-epilogue gather check
  `_fragment_epilogue_ok` is a local filter here).
- **`Fold`** — the reduce-partition family: serial / coop `t<n>` / ILP `r<n>` / cross-CTA `g<n>` / **streaming**. The
  flash schedule is a fold MOVE, not a shape. Legality reads only the node + ctx: axis static/symbolic and extent,
  carrier additivity (atomic `g`) vs combine-relocatability (kernel-finalize `g` — one move unifies today's split-K
  and split-KV), output cardinality, the occupancy heuristics (`_pick_coop`). The combine is already carrier-generic
  (`_coop_carrier`'s own contract), so twisted-vs-planar changes combine cost, never eligibility.
- **`Contraction`** — the tile family (scalar register tile / warp atom, dtype-gated via `_warp_atoms`) × per-edge
  stage family (materialized edge: gmem-direct / cp.async / TMA; computed edge: the sync compute-fill) × raster. The
  WS5 layout gates, `%32` transposed-band conditions, and `_f16acc_allowed` stay as filters inside the generator —
  they read only the node's own loads plus ctx, so they compose.

### Unification — the one composition constraint

`WORK` is kernel-global and smem is one budget. Each candidate declares its worker demand; a composed row is legal iff
the demands join (`plan_workers` / `seal_workers` already do exactly this, ad hoc and pairwise). This is a tiny
solver — kinds {thread, warp}, units, `+p` band — and it is also why `WORK` keeps leading the fork-tree levels: the
generic tree's levels become `[WORK, *per-node-path families in canonical path order, RASTER]`, which
`build_fork_tree` already supports.

### Pin narrowing — generic for free

Pins are already spelled per path (`TILE@dd`), so narrowing becomes "filter each node's local candidate list at its
path before taking the product". Retires `_narrow_flash_forms`, the per-path `_pinned_tile` fast paths, and the
REDUCE-pin escapes at the top of the twisted branch. ONE standing exception: the dynamic-attention golden rows record
the PV plan on a bare `TILE` because a symbolic trace resolves no stable axis key, so the bare↔axis-keyed any-of
(`pin_key_matches` / `family_value`) stays until symbolic-trace keyed resolution exists — the generic narrower must
keep that arm, documented and tested, never silently widened.

### Backtracking — one edge-collapse rule

If a computed edge has no candidate under the demand, collapse the edge inline (the cone as a structural node in the
lift — legal, a term is a value) and re-enumerate the parent. This is what `_demote_planar` already does structurally
and what `_demoted_warp_option` and `020_schedule`'s fallback re-implement at other depths; one rule replaces all
three.

### Explicitly out of scope

Materialization is untouched: `_warp_option` / `_tile_option` / `_splitk_option` and the whole `lowering/kernel`
materializer stay as the `materialize` callbacks the fork tree already calls. `030_split_reduce` keeps consuming the
same schedule slices. This refactor changes candidate *enumeration and composition* only.

## Worked examples — every shape in `plans/loop-vs-tile-ir-examples.md` under the new walk

The eight dumps in that file are the acceptance corpus: for each, the new walk must enumerate the SAME rows (same
spellings, same order — option-0 is semantic, it is cold greedy's pick). Each listing below is that file's tile dump
with the axis → grid/warp/thread mapping annotated inline (`←`), for the schedule the dump shows (or, for flash, the
candidate families). Codec reminders for reading the annotations: thread `WORK` is `t<N>x<M>` and scalar `TILE` is
`f<fn>[x<fm>]` — both **n-then-m**; warp `WORK` `w<M>x<N>` and warp `TILE` `<atom>/f<FM>x<FN>` are m-then-n.

### 1. Bare reduction — `Fold`, planar

```python
torch.randn(8, 512, 1024).sum(dim=-1)
```

```
=== 0: k_sum_1_reduce ===
    place  free=(a0, a1)  grid=(a0, a1)           ← a0, a1 → blockIdx: one CTA per output cell (8·512 = 4096 CTAs)
    work   t128                                   ← the coop row's worker demand: a 128-thread cooperative band
    Fold[a2 in 0..1024] planar   ⟨REDUCE=coop⟩    ← a2 → lane ⊗ serial: lane l folds elements l, l+128, … (8 each),
    ├─ init: (0)                                     then a 7-step cross-lane tree combine
    ├─ lift: λ(a2) -> (in0)
    │    in0 = load x[a0, a1, a2]                 ← adjacent lanes read adjacent a2 — coalesced band read
    └─ combine: λ(acc0, acc0__o) -> (acc0)
         acc0 = add(acc0, acc0__o)                ← the SAME λ is the cross-lane merge: combine is carrier-generic
```

Depth-0 walk. Root demand: one gmem scalar per grid cell. The `Fold` generator emits `_reduce_specs`' rows verbatim:
serial option-0 (this free grid is past `_FREE_CAP`, so the heuristic stays scalar), then the coop catalog — each
`t<n>` row's worker demand IS its `WORK` inventory, nothing to unify against — then guarded `g<n>`/`r<n>`. The `t128`
row shown is the evidence/prior pick among those catalog rows, exactly as today.

### 2. RMSNorm — `Map` over a `Fold`, plus a sweep store

```python
torch.nn.functional.rms_norm(torch.randn(4, 512, 1024), (1024,), torch.randn(1024))
```

```
=== 0: k_rms_norm_6175eb ===
    place  free=(a0, a1)  grid=(a0, a1)               ← one CTA per row: 4·512 = 2048 CTAs
    work   t128
    Map
    ├─ source[0]: Fold[a2 in 0..1024] planar   ⟨REDUCE=coop⟩   ← a2 → 128 lanes × 8 serial + tree combine
    │  ├─ init: (0)
    │  ├─ lift: λ(a2) -> (v1)
    │  │    in2 = load x0[a0, a1, a2]                 ← coalesced band read of the row
    │  │    v1 = multiply(in2, in2)
    │  └─ combine: λ(acc0, acc0__o) -> (acc0)
    │       acc0 = add(acc0, acc0__o)
    └─ fn: λ(acc0) -> (v6)                            ← runs once per OUTPUT cell of the sweep, on its owning lane
         v0 = reciprocal(1024)
         v2 = multiply(acc0, v0)
         v3 = add(1e-06, v2)
         v4 = rsqrt(v3)
         in3 = load x0[a0, a1, a3]                    ← the TWICE-READ edge (also read in the lift): the shared-row
         v5 = multiply(in3, v4)                          stage move's site — its benefit gate (a contraction tail)
         in4 = load x1[a3]                               declines on plain rmsnorm, so this stays gmem-direct
         v6 = multiply(in4, v5)
    stores
    └─ sweep(a3) rms_norm[a0, a1, a3] = v6            ← a3 → the SAME 128 lanes: each writes 1024/128 = 8 cells
```

The `Map` delegates to the fold (demand: the per-cell scalar state); the sweep distributes the output row over the
coop band. The shared-row `sync` stage stops being the `_shared_row_buf` shape-matcher and becomes a stage move on a
twice-read materialized edge — offered only when the tail contracts over a new axis (the fused norm→linear shape,
example 6's depth-1 form), which is exactly today's `_has_contraction_tail` gate, kept for row equality.

### 3. Softmax — `Fold`, twisted

```python
torch.nn.functional.softmax(torch.randn(8, 512, 512), dim=-1)
```

```
=== 0: k_softmax_efaac1 ===
    place  free=(a0, a1)  grid=(a0, a1)               ← one CTA per row: 8·512 = 4096 CTAs
    work   t128
    Map
    ├─ source[0]: Fold[a2 in 0..512] twisted   ⟨REDUCE=coop⟩   ← a2 → 128 lanes × 4 serial; each lane's running
    │  ├─ init: (-inf, 0)                                        state is the PAIR (m, l)
    │  ├─ lift: λ(a2) -> (acc0__osin, 1)
    │  │    acc0__osin = load x[a0, a1, a2]
    │  └─ combine: λ(acc0, acc1, acc0__o, acc1__o) -> (acc0, acc1)
    │       acc0__o__t0 = maximum(acc0, acc0__o)      ← the cross-lane tree merge runs THIS λ on (m, l) pairs —
    │       …                                            the twisted rescale, same code as the serial step
    │       acc0 = copy(acc0__o__t0)
    └─ fn: λ(acc0, acc1) -> (v5)                      ← normalize per output cell, on its owning lane
         …
    stores
    └─ sweep(a3) softmax[a0, a1, a3] = v5             ← a3 → the same lanes: 512/128 = 4 cells each
```

Identical walk to example 2 — twisted changes what the combine COSTS, never which fold moves are legal, so no new
path exists for softmax at all. The example file's own point ("same node kind as example 1, only the monoid arity
differs") finally holds for the scheduler too.

### 4. Matmul — `Contraction`

```python
torch.randn(1024, 4096) @ torch.randn(4096, 4096)
```

```
=== 0: k_matmul_c1b5ce ===
    place  free=(a0, a1)  grid=(a0, a1)     ← block-tiled: a0 (m) in 224-row tiles, a1 (n) in 128-col tiles
    work   t64x16                           ← 1024 threads/CTA as a 64(n) × 16(m) unit grid
    Contraction [Σ a2 in 0..4096] x0 @ x1 -> acc0   ⟨TILE=f2x14 STAGE=d2/cp/ring⟩
    │                                       ← a2 (K) → SERIAL per thread, chunked by the depth-2 cp.async ring;
    │                                          f2x14: each thread owns a 14(m) × 2(n) register fragment, so the
    │                                          CTA tile is (16·14) × (64·2) = 224 × 128 → ⌈1024/224⌉ × 32 CTAs
    ├─ a: in1 = load x0[a0, a2]   ‹materialized›   ← staged: A slab → smem each ring step, loads fan across the CTA
    └─ b: in0 = load x1[a2, a1]   ‹materialized›   ← staged: B slab → smem, double-buffered against compute
```

Depth-0: the `Contraction` generator's tile × stage × reduce × raster product — `_tile_rows` almost unchanged, minus
the `contraction_view` shape-probe around it. The warp sibling family maps the same axes differently: `WORK=w<M>x<N>`
puts 32-lane warps on an (m, n) warp grid, `TILE=<atom>/f<FM>x<FN>[/k<bk>]` gives each warp an
(FM·atom_m) × (FN·atom_n) fragment tile (per-lane fragments per the mma layout), and a2 advances in `atom_k`-element
mma steps, `bk` per smem stage. Split-K (`g<n>`) would additionally map a2 across `n` CTAs plus a finalize.

### 5. Epilogue fusion — `Map` over a `Contraction`

```python
torch.relu(torch.randn(512, 1024) @ torch.randn(1024, 1024) + torch.randn(1024))
```

```
=== 0: k_matmul_9f4c41 ===
    place  free=(a0, a1)  grid=(a0, a1)     ← a0 in 64-row tiles, a1 in 64-col tiles → 8 × 16 = 128 CTAs
    work   t32x8                            ← 256 threads/CTA as a 32(n) × 8(m) unit grid
    Map
    ├─ source[0]: Contraction [Σ a2 in 0..1024] x0 @ x1 -> acc0   ⟨TILE=f2x8 STAGE=d2/cp/ring⟩
    │  │                                    ← a2 serial per thread over the cp.async ring; f2x8: 8(m) × 2(n)
    │  │                                       registers per thread → CTA tile (8·8) × (32·2) = 64 × 64
    │  ├─ a: in1 = load x0[a0, a2]   ‹materialized›
    │  └─ b: in2 = load x1[a2, a1]   ‹materialized›
    └─ fn: λ(acc0) -> (v2)                  ← runs per REGISTER CELL after the K loop, before the store —
         in0 = load x2[a1]                     16 cells per thread, no extra parallel structure of its own
         v1 = add(acc0, in0)
         v2 = relu(v1)
    stores
    └─ relu[a0, a1] = v2                    ← each thread stores its 8×2 fragment
```

Example 4's product with one local filter at the `Map`: the fragment-epilogue gather check (a warp row folds `fn`
into the per-fragment `RegEpilogue`, so a data-dependent gather index refuses the warp tier). The source subtree is
byte-identical to example 4's node — under the generic walk the candidate enumeration is identical too, by
construction rather than by parallel code.

### 6. SwiGLU — the concatenated gate⊗up projection, then a pointwise kernel

```python
class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(1024, 2816, bias=False)
        self.up = nn.Linear(1024, 2816, bias=False)

    def forward(self, x):
        return F.silu(self.gate(x)) * self.up(x)
```

```
=== 0: k_linear_reduce_6eecae ===
    place  free=(a0, a1)  grid=(a0, a1)     ← a0 (m) in 160-row tiles, a1 (n) in 128-col tiles
    work   t32x16                           ← 512 threads/CTA as a 32(n) × 16(m) unit grid
    Contraction [Σ a2 in 0..1024] x @ linear__cat__linear_1_w trans -> acc0   ⟨TILE=f4x10⟩
    │                                       ← a2 serial per thread, gmem-direct (no STAGE picked here);
    │                                          f4x10: 10(m) × 4(n) registers → CTA tile 160 × 128
    ├─ a: in1 = load x[a0, a2]   ‹materialized›
    └─ b: in0 = load linear__cat__linear_1_w[a1, a2]   ‹materialized›
    │                                       ← trans: B is N-major (F.linear layout), K contiguous per row — the
    │                                          WS5 layout gates and the warp tier's N-major slab read this
    │                                          locally off the node's own edge
    stores
    └─ linear__cat__linear_1_reduce[a0, 0, a1] = acc0
```

Kernel 1 (silu·mul) is example 7's shape. This trace concatenated gate|up pre-recognition, so the node has arity 1;
the **arity-2 form** (one `a` cone edge, `channel[0] → acc_g`, `channel[1] → acc_u`) is the depth-1 case: the
contraction's computed-A demand is "per-k-block A slab via the sync compute-fill" — a producer phase where the CTA's
threads cooperatively evaluate the cone (the normalized/projected A row) for the current k-block into smem, sync, then
consume it from both channels' mma/scalar steps. The cone's inner stat `Fold` answers with its own `REDUCE@<axis>`
slice from its own generator (retiring `prologue_knob_bases`' hand-threading), and the `020` MONOID-producer merge
(two term readings of one loop) reduces to "run `candidates` on each reading, union the rows", the decided-empty
stamps computed generically as "families whose site the other tree lacks".

### 7. Pure pointwise — `Map` with no sources

```python
torch.nn.functional.silu(torch.randn(8, 512, 1024)) * torch.randn(8, 512, 1024)
```

```
=== 0: k_mul_pointwise ===
    place  free=(a0, a1, a2)  grid=(a0, a1, a2)   ← EVERY axis → grid; one THREAD per grid cell (per-cell tier —
    Map  ‹pointwise›                                 no work line, launch geometry derived by the materializer)
    └─ fn: λ() -> (v5__u3)
         in1__u0 = load x0[a0, a1, ((a2 * 4) + 0)]   ← ×4 unroll strip: a2's grid extent is 1024/4 = 256 and each
         in2__u0 = load x1[a0, a1, ((a2 * 4) + 0)]      thread owns 4 CONSECUTIVE cells — a vector-width access
         …                                              per lane, lanes still coalesced across a2
    stores
    ├─ mul[a0, a1, ((a2 * 4) + 0)] = v5__u0
    ├─ mul[a0, a1, ((a2 * 4) + 1)] = v5__u1
    ├─ mul[a0, a1, ((a2 * 4) + 2)] = v5__u2
    └─ mul[a0, a1, ((a2 * 4) + 3)] = v5__u3
```

Trivial depth-0: the sourceless-`Map` generator's grid-map + strip moves (`_map_strip_fork` verbatim).

### 8. Causal SDPA — the flash rewrite

```python
F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (1, 8, 512, 64) each
```

```
=== 0: scaled_dot_product_attention -> scaled_dot_product_attention ===
    place  free=(b0, b1, m, d)  grid=(b0, b1, m)   ← b0·b1 (batch · heads) → blockIdx on EVERY family; how m and d
    Map                                               map is what the fork decides (annotated per family below)
    ├─ source[0]: Fold[kv in 0..512] twisted       ← kv NEVER maps to the grid un-split: it is the STREAM — serial
    │  ├─ init: (-inf, 0, 0)                          bn-blocks per CTA; the g<n>k move splits it across n CTAs
    │  ├─ lift: λ(kv, sacc, v_e) -> (s_causal, 1, v_e)   ← (m, l, O) running state stays register/fragment-resident
    │  │    scale_c = load _flash_scale[]                   across the whole stream — the demand the fold puts on
    │  │    s = multiply(sacc, scale_c)                     ITSELF; sacc / v_e bind the operand edges positionally
    │  │    ninf_c = load _flash_ninf[]
    │  │    s_causal = s when ((kv <= m))          ← causal mask per streamed block (whole blocks skippable when
    │  │               ninf_c when (1)                kv-block > m-block — a raster/legality fact, not algebra)
    │  ├─ combine: λ(m_i, l_i, O_i, m_i__o, l_i__o, O_i__o) -> (m_i, l_i, O_i)
    │  │    …                                      ← the per-block rescale; on the coop family the SAME λ is the
    │  │                                              cross-lane merge of (m, l, O) triples
    │  ├─ operand[0]: Contraction [Σ dd in 0..64] x0 @ x1 trans -> sacc   ‹computed›
    │  │  │                                        ← the streaming demand lands HERE: "produce a (bm × bn) score
    │  │  │                                           block per kv-block, fragment residency"
    │  │  ├─ a: q_e = load x0[b0, b1, m, dd]   ‹materialized›   ← warp family: the Q tile loads ONCE and stays
    │  │  └─ b: k_e = load x1[b0, b1, kv, dd]  ‹materialized›      resident across the stream
    │  │                                        ← warp family: K slab staged per kv-block (cp.async / TMA);
    │  │                                           dd → serial mma k-steps (64 / atom_k = 4)
    │  └─ operand[1]: v_e = load x2[b0, b1, kv, d]   ‹materialized›   ← V slab staged per kv-block, consumed by
    │                                                                    the DERIVED PV contraction below the seam
    └─ fn: λ(m_i, l_i, O_i) -> (O_i__proj)
         O_i__proj = divide(O_i, l_i)            ← once per (m, d) output cell, at stream end
    schedule
    └─ TILE@pj = f64                             ← the derived PV site's slice: d (64 cols) as a per-thread
    stores                                          register vector — the CHAIN reading's spelling
    └─ scaled_dot_product_attention[0, b1, m, d] = O_i__proj
```

The dump is pre-schedule (no `work` line), so the annotation above shows where each family's decision lands. The
fold's move families and their axis mappings:

- **WARP streaming** (`_twisted_warp_options` today; dtype-gated OFF for this f32 `randn` trace — joins on f16/bf16
  models): grid = (b0·b1, m/bm). m → bm query rows held as warp mma fragments (bm = WM·FM·atom_m from `WORK` +
  `TILE@dd`); kv → serial stream, bn keys per step (bn = WN·FN·atom_n — the kv-block ↔ score-tile coupling
  `_stamp_twisted_split` hand-computes IS the unification check on the QK edge); dd → mma k-steps; d → PV fragment
  columns via `TILE@pj`. ONE `WORK=w<M>x<N>` inventory shared by the QK child and the derived PV — the unification.
- **CHAIN** (FA-2 scalar, `_twisted_chain_option`): grid = (b0, b1, m) — one THREAD per query row; d → a per-thread
  register vector (`TILE@pj=f64`, legal since d = 64 ≤ the register budget); kv → serial per thread, the score
  computed ONCE per key and shared across all 64 columns.
- **Per-cell**: grid = (b0, b1, m, d) — one thread per output cell; kv serial; the QK edge collapses inline (the
  edge-collapse rule applied as a move), so the score recomputes per d — the redundant form the chain exists to beat.
- **Coop**: grid = (b0, b1, m, d), a `t<n>` band splits kv across lanes within the CTA; the cross-lane merge of
  (m, l, O) triples is the fold's own combine λ — carrier-generic, same machinery as examples 1–3.
- **Split-KV** (`g<n>k`): composes with the warp rows — kv → n CTAs × serial stream; each partial keeps fragment
  residency; `030_split_reduce` realizes the partial + LSE-combine finalize. This is the generic kernel-finalize
  `g<n>` fold move (the twisted carrier is combine-relocatable), the same move as matmul split-K — legality (kv
  divides, slices block-whole) plus unification replace `_stamp_twisted_split`.

Pins (`TILE@dd` / `TILE@pj` / `REDUCE@<kv>`) narrow at their paths like every other kernel; `_narrow_flash_forms` and
the warp/stage-pin routing block at the top of the twisted branch retire.

### Summary — what each hand-written path becomes

| Today | Under the walk |
| --- | --- |
| `_tile_rows` (contraction product) | the `Contraction` generator, depth-0 |
| `_reduce_specs` + `_coop_carrier` | the `Fold` generator, depth-0 |
| `_row_stage` / `_shared_row_buf` shape-match | stage move on a twice-read materialized edge (gate kept) |
| `_computed_a_rows` + `prologue_knob_bases` | depth-1 recursion; child fold spells its own slices |
| `020`'s MONOID-producer merge | candidate-union of two term readings; decided-empty generic |
| `twisted_pair` + `_twisted_warp_options` | `Fold` streaming moves + child recursion + unification |
| `_twisted_chain_option` | a fold move |
| `_stamp_twisted_split` + matmul split-K | ONE `g<n>` fold move (atomic / kernel-finalize legality) |
| `_narrow_flash_forms` + per-path pin escapes | per-path narrowing of local candidate lists |
| `_demote_planar` / `_demoted_warp_option` | the one edge-collapse backtracking rule |

## Invariants that must not move

- **Knob keys and value spellings are frozen.** Keys are already path-based; the golden corpus, priors, and recorded
  evidence key on them. Zero spelling changes is the definition of a correct refactor.
- **Row ORDER is semantic.** Option-0 is the conservative pick cold greedy deploys; gates compare ordered row lists,
  not sets.
- **`WORK` leads the fork-tree levels**; `RASTER` closes.
- **Node identity is untouched** — the walk reads terms, never mutates them; `term_key` / `op_cache_key` unchanged.
- **The bare-`TILE` dynamic-attention pin any-of** stays until symbolic keyed resolution exists.

## Status — the ground is cleared

The old scheduler is DELETED (`_schedule.py`, `_view.py`, the `020_schedule` rule). Recognition, the codec, the move
catalog and the materializer are untouched — the sections above describing the hand-written paths now read as the
behavior to REPRODUCE, not code to refactor. Row-list equality against the old enumeration is therefore no longer
available as a phase gate; what replaces it is `tests/xfail_registry.py`: 111 exact node ids, every one an
acceptance obligation, measured on a CPU-only box (GPU-gated ids get appended when observed). A phase is done when
the ids it restores are deleted from the registry; the refactor is done when the registry is empty. Byte-identity
via `scripts/digest_kernels.py` still applies for pinned/golden rows, against kernels dumped before the removal.

## Migration — one phase per recursion depth, each gated

Verification harness first, then port shallowest-to-deepest. Per-phase gates, all CPU-only except the last:

- **Registry shrink**: the ids the phase restores are deleted from `tests/xfail_registry.py` and the suite is green
  without them. Row ORDER is still semantic — assert the ordered row list per corpus shape (the eight example
  expressions above plus every golden shape) against the spellings recorded in the golden corpus.
- **`scripts/digest_kernels.py` byte-identity** on the materialized kernels for pinned/golden rows.
- `make test`, `make lint`.

1. **Name the interface.** `Demand` + `NodeSchedule` + the `MOVES` registry + `candidates` / `unify` in a new
   `passes/lowering/tile/_compose.py`; generators initially thin wrappers over the existing helpers. No behavior
   change; harness runs green trivially.
2. **Depth-0 contraction** (examples 4, 5, 6-k0): port `_tile_rows` + the materialized-edge stage candidates into the
   `Contraction` generator; `schedule()`'s CONTRACTION arm calls the walk.
3. **Depth-0 fold + map** (examples 1, 2, 3, 7): port `_reduce_specs`, `_map_strip_fork`; re-derive the shared-row
   stage from the twice-read-edge read (benefit gate kept).
4. **Depth-1 computed-A** (example 6's arity-2 form, norm→linear): port `_computed_a_rows`; the 020 merge becomes
   candidate-union; `prologue_knob_bases` retires.
5. **The twisted fold** (example 8): streaming + chain + per-cell + split moves; retire `twisted_pair` consumers,
   `_twisted_warp_options`' outer plumbing, `_stamp_twisted_split`, `_narrow_flash_forms`.
6. **Collapse.** `schedule()` becomes walk → compose → `build_fork_tree`; delete the role dispatch, the demotion
   special cases (fold into the edge-collapse rule), and the dead pin escapes.

Gate before merge (GPU): `make bench-kernels`, a flash/attention compile + tune probe on the 5090, and an eval-golden
MATCH sweep — the enumeration is the single choke point every tier resolves through, so a silent row loss shows up as
a golden miss, and that check is cheap.

## Risks / honest caveats

- The real content of flash does not disappear — it moves into the fold's streaming move generator and the demand it
  emits. The win is not fewer lines on day one; it is that the NEXT composition is a new local move or demand, not a
  fifth whole-tree path.
- The unification solver must stay tiny. If it grows past workers + smem + block-divisibility, that is a design smell:
  push the fact into a local legality filter instead.
- Ordered-row equality may surface latent order accidents in today's paths (rows equal as sets, ordered by code-path
  happenstance). Resolve each one explicitly — either the order was semantic (keep it, encode it in the generator) or
  it was not (document the diff in the phase's commit); never let the harness pass by sorting.
