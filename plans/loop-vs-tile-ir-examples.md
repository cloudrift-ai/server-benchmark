# Loop IR vs Tile IR — worked examples

Eight representative operations, compiled on this branch (`feature/tile-ir-helper-relocation`) and dumped at both
stages. Every listing below is verbatim output of

```bash
emmy compile -c "<expr>" --ir loop     # LoopOp — explicit nest, accumulators as `acc <- ⊕(acc, …)`
emmy compile -c "<expr>" --ir tile     # TileOp — the stored algebraic tree + its caller facts
```

No GPU is needed for either stage.

## How to read the two forms

**Loop IR** is the literal nest: one `for` per axis, loads spelled at their index expressions, reductions as
`acc0 <- add(acc0, …)`, and the output as a plain indexed assignment. It is what the frontend/fusion passes produce and
it says nothing about how work is split across a GPU.

**Tile IR** stores exactly three node kinds (`emmy/compiler/ir/tile/ir.py`):

| Node | Stored params | Header in a dump |
| --- | --- | --- |
| `Fold` | `axis`, `lift` λ, the monoid's `(init, combine)`, `operands` | `Fold[<axis> in 0..N] <role>` |
| `Contraction` | `k_axis`, the shared `a` edge, the product `Channel`s `(b_i, acc_i)` | `Contraction [Σ k in 0..N] a @ b -> acc` |
| `Map` | `fn: Lambda`, `sources` | `Map` (its λ rides the `fn:` branch) |

The tile dump renders that tree *as a tree*. Reading it:

- **Branches are stored params.** A `Fold` shows `init` / `lift` / `combine` / `operand[i]`; a `Contraction` shows `a`
  and one branch per channel; a `Map` shows `source[i]` and `fn`. Every λ-valued field reads the same way — the
  signature labels the branch and its body nests two under it, so a binder is always adjacent to what it binds.
- **Operand edges recurse and are tagged.** `‹materialized›` is a leaf gmem `Load`; `‹computed›` is an inline node,
  and its own subtree is printed below it. Those are the only two inhabitants of an edge.
- **A λ signature names what it captures.** `[captures …]` lists the free names that are not iteration vars. Empty
  (so omitted) means closed — and closed is what lets a subtree hoist to an operand edge.
- **Nothing derived is printed** — not the per-cell step (`Fold.step_stmts()`), not the nodes synthesized inside it,
  not the lowered nest. All of it follows from the params above, and the structure is already complete in the edges and
  their nesting. `--ir loop` is where a body lives. `role` on a `Fold` is derived too (`Fold.role`), never stored:
  `TWISTED` off the twist family, `CONTRACTION` off a composed split-K operand, `PLANAR` otherwise — it rides the
  header because it is a one-word reading of the params, not a program.
- **The regions are the owners.** `place` / `work` above the term is geometry; the tree is algebra; `schedule` and
  `stores` below are the schedule dict and the kernel boundary. Slices normally print as `⟨TILE=… REDUCE=… STAGE=…⟩`
  beside the node they key against (from `TileOp.schedule`, via the tree-path codec — never from the term itself); a
  slice whose site is *derived* has no stored node to annotate, so it lands in the `schedule` region instead. An empty
  region is omitted.

---

## 1. Bare reduction — `Fold`, planar

```python
torch.randn(8, 512, 1024).sum(dim=-1)
```

**Loop IR**

```
=== 0: k_sum_1_reduce ===
    for a0 in 0..8
        for a1 in 0..512
            for a2 in 0..1024
                in0 = load x[a0, a1, a2]
                acc0 <- add(acc0, in0)
            sum_1[a0, a1] = acc0
```

**Tile IR**

```
=== 0: k_sum_1_reduce ===
    place  free=(a0, a1)  grid=(a0, a1)
    work   t128
    Fold[a2 in 0..1024] planar   ⟨REDUCE=coop⟩
    ├─ init: (0)
    ├─ lift: λ(a2) -> (in0)
    │    in0 = load x[a0, a1, a2]
    └─ combine: λ(acc0, acc0__o) -> (acc0)
         acc0 = add(acc0, acc0__o)
```

The simplest case, and it already shows the whole vocabulary: identity lift, the `(0, add)` monoid spelled as an `init`
seed plus a two-argument `combine`. The per-cell step — that combine specialized at the singleton, `acc0 <- add(acc0,
in0)` — is derived and therefore not printed; it is also impure (an `Accum` mutating loop-carried state), which is why
it could never have been one of the stored λs. The batch axes `a0`/`a1` are gone from the term too — they are the
kernel's grid, a `place` fact. Note this fold has *no* operand branch: the load stayed inline in the lift.

## 2. RMSNorm — `Map` over a `Fold`, plus a sweep store

```python
torch.nn.functional.rms_norm(torch.randn(4, 512, 1024), (1024,), torch.randn(1024))
```

**Loop IR**

```
=== 0: k_rms_norm_6175eb ===
    v0 = reciprocal(1024)
    for a0 in 0..4
        for a1 in 0..512
            for a2 in 0..1024
                in2 = load x0[a0, a1, a2]
                v1 = multiply(in2, in2)
                acc0 <- add(acc0, v1)
            v2 = multiply(acc0, v0)
            v3 = add(1e-06, v2)
            v4 = rsqrt(v3)
            for a3 in 0..1024
                in3 = load x0[a0, a1, a3]
                v5 = multiply(in3, v4)
                in4 = load x1[a3]
                v6 = multiply(in4, v5)
                rms_norm[a0, a1, a3] = v6
```

**Tile IR**

```
=== 0: k_rms_norm_6175eb ===
    place  free=(a0, a1)  grid=(a0, a1)
    work   t128
    Map
    ├─ source[0]: Fold[a2 in 0..1024] planar   ⟨REDUCE=coop⟩
    │  ├─ init: (0)
    │  ├─ lift: λ(a2) -> (v1)
    │  │    in2 = load x0[a0, a1, a2]
    │  │    v1 = multiply(in2, in2)
    │  └─ combine: λ(acc0, acc0__o) -> (acc0)
    │       acc0 = add(acc0, acc0__o)
    └─ fn: λ(acc0) -> (v6)
         v0 = reciprocal(1024)
         v2 = multiply(acc0, v0)
         v3 = add(1e-06, v2)
         v4 = rsqrt(v3)
         in3 = load x0[a0, a1, a3]
         v5 = multiply(in3, v4)
         in4 = load x1[a3]
         v6 = multiply(in4, v5)
    stores
    └─ sweep(a3) rms_norm[a0, a1, a3] = v6
```

The `Map`'s binder makes the wiring explicit: `fn: λ(acc0) -> (v6)` says the projection consumes exactly the fold's one
accumulator and produces `v6` — and it labels the branch holding the body it binds, so the two are read together. The statistic is the `Fold` (square lift, sum monoid); the second `for a3` nest of the
loop form is gone — the per-cell normalize is `Map.fn`, and the output sweep is a `Store` decoration, reconstituted on
demand by `effect_tail`. That is the shape of the "cone": row-invariant prologue above the seam, per-cell body below.

## 3. Softmax — `Fold`, twisted

```python
torch.nn.functional.softmax(torch.randn(8, 512, 512), dim=-1)
```

**Loop IR** — three passes over the row (max, then exp-sum, then normalize):

```
=== 0: k_softmax_efaac1 ===
    for a0 in 0..8
        for a1 in 0..512
            for a2 in 0..512
                in0 = load x[a0, a1, a2]
                acc0 <- maximum(acc0, in0)
            for a2 in 0..512
                in1 = load x[a0, a1, a2]
                v0 = subtract(in1, acc0)
                v1 = exp(v0)
                acc1 <- add(acc1, v1)
            v2 = reciprocal(acc1)
            for a3 in 0..512
                in2 = load x[a0, a1, a3]
                v3 = subtract(in2, acc0)
                v4 = exp(v3)
                v5 = multiply(v2, v4)
                softmax[a0, a1, a3] = v5
```

**Tile IR** — one twisted fold:

```
=== 0: k_softmax_efaac1 ===
    place  free=(a0, a1)  grid=(a0, a1)
    work   t128
    Map
    ├─ source[0]: Fold[a2 in 0..512] twisted   ⟨REDUCE=coop⟩
    │  ├─ init: (-inf, 0)
    │  ├─ lift: λ(a2) -> (acc0__osin, 1)
    │  │    acc0__osin = load x[a0, a1, a2]
    │  └─ combine: λ(acc0, acc1, acc0__o, acc1__o) -> (acc0, acc1)
    │       acc0__o__t0 = maximum(acc0, acc0__o)
    │       acc0__o__t1 = subtract(acc0, acc0__o__t0)
    │       acc0__o__t2 = exp(acc0__o__t1)
    │       acc0__o__t3 = multiply(acc1, acc0__o__t2)
    │       acc0__o__t4 = subtract(acc0__o, acc0__o__t0)
    │       acc0__o__t5 = exp(acc0__o__t4)
    │       acc0__o__t6 = multiply(acc1__o, acc0__o__t5)
    │       acc1 = add(acc0__o__t3, acc0__o__t6)
    │       acc0 = copy(acc0__o__t0)
    └─ fn: λ(acc0, acc1) -> (v5)
         v2 = reciprocal(acc1)
         in2 = load x[a0, a1, a3]
         v3 = subtract(in2, acc0)
         v4 = exp(v3)
         v5 = multiply(v2, v4)
    stores
    └─ sweep(a3) softmax[a0, a1, a3] = v5
```

Same node kind as example 1 — only the monoid arity differs. The tree makes the difference legible in one line:
`init: (-inf, 0)` is the two-component state `(m, l)`, and `lift: λ(a2) -> (acc0__osin, 1)` is the singleton `(x, 1)`
— the ι injection spelled as a literal result, which is why it has no def to name.

The `combine` branch is the actual novelty. It is a genuine two-argument `S × S → S` monoid over `(m, l)`: take the
componentwise max, rescale *both* sides by `exp(m_side − m_new)`, add. The streaming form the loop IR shows is that
same program specialized at the singleton, where the right operand's rescale collapses — derived, so the dump does not
restate it. Three loop passes collapse to one streaming pass, and `twisted` is derived from the combine's shape, not
stored.

## 4. Matmul — `Contraction`

```python
torch.randn(1024, 4096) @ torch.randn(4096, 4096)
```

**Loop IR**

```
=== 0: k_matmul_c1b5ce ===
    for a0 in 0..1024
        for a1 in 0..4096
            for a2 in 0..4096
                in0 = load x1[a2, a1]
                in1 = load x0[a0, a2]
                v0 = multiply(in0, in1)
                acc0 <- add(acc0, v0)
            matmul[a0, a1] = acc0
```

**Tile IR**

```
=== 0: k_matmul_c1b5ce ===
    place  free=(a0, a1)  grid=(a0, a1)
    work   t64x16
    Contraction [Σ a2 in 0..4096] x0 @ x1 -> acc0   ⟨TILE=f2x14 STAGE=d2/cp/ring⟩
    ├─ a: in1 = load x0[a0, a2]   ‹materialized›
    └─ b: in0 = load x1[a2, a1]   ‹materialized›
```

Four lines, and they are the entire kernel. The stored node is *only* `k_axis=a2`, the `a` edge and one
`Channel(b=x1, acc=acc0)` — both edges materialized. The `(m, n)` axes `a0`/`a1` are **not** on the node; they are
caller placement, which is why they appear on the `place` line and the tile geometry appears as a `⟨TILE=…⟩`
annotation rather than a node field. `TilePlan.at(m, n)` binds the two at the point of use. That separation is what
makes the node's identity (`==` / `hash` / `term_key`) its algebra alone, so the same term can be re-placed and
re-scheduled — here it picked an `f2x14` register tile behind a depth-2 `cp.async` ring — without changing.

## 5. Epilogue fusion — `Map` over a `Contraction`

```python
torch.relu(torch.randn(512, 1024) @ torch.randn(1024, 1024) + torch.randn(1024))
```

**Loop IR**

```
=== 0: k_matmul_9f4c41 ===
    for a0 in 0..512
        for a1 in 0..1024
            in0 = load x2[a1]
            for a2 in 0..1024
                in1 = load x0[a0, a2]
                in2 = load x1[a2, a1]
                v0 = multiply(in1, in2)
                acc0 <- add(acc0, v0)
            v1 = add(acc0, in0)
            v2 = relu(v1)
            relu[a0, a1] = v2
```

**Tile IR**

```
=== 0: k_matmul_9f4c41 ===
    place  free=(a0, a1)  grid=(a0, a1)
    work   t32x8
    Map
    ├─ source[0]: Contraction [Σ a2 in 0..1024] x0 @ x1 -> acc0   ⟨TILE=f2x8 STAGE=d2/cp/ring⟩
    │  ├─ a: in1 = load x0[a0, a2]   ‹materialized›
    │  └─ b: in2 = load x1[a2, a1]   ‹materialized›
    └─ fn: λ(acc0) -> (v2)
         in0 = load x2[a1]
         v1 = add(acc0, in0)
         v2 = relu(v1)
    stores
    └─ relu[a0, a1] = v2
```

`Map(fn=bias+relu, sources=(Contraction,))`. Compare with example 4: the `source[0]` subtree is structurally identical
— same header, same two edges, its own `⟨TILE=…⟩` — and everything the epilogue added lives in the `fn` branch and
the `stores` region. Fusion here is literally a wrapper, not an edit to the contraction.

## 6. SwiGLU — the concatenated gate⊗up projection, then a pointwise kernel

```python
# module built outside the traced expression so its init doesn't get traced
class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(1024, 2816, bias=False)
        self.up = nn.Linear(1024, 2816, bias=False)

    def forward(self, x):
        return F.silu(self.gate(x)) * self.up(x)
```

**Loop IR** — two kernels; the two projections already concatenated into one N=5632 reduce:

```
=== 0: k_linear_reduce_6eecae ===
    for a0 in 0..512
        for a1 in 0..5632
            for a2 in 0..1024
                in0 = load linear__cat__linear_1_w[a1, a2]
                in1 = load x[a0, a2]
                v0 = multiply(in0, in1)
                acc0 <- add(acc0, v0)
            linear__cat__linear_1_reduce[a0, 0, a1] = acc0

=== 1: k_linear_pointwise_253985 ===
    for a0 in 0..512
        for a1 in 0..2816
            in1 = load linear__cat__linear_1_reduce[a0, 0, a1]
            v0 = negative(in1)
            v1 = exp(v0)
            v2 = add(1, v1)
            v3 = reciprocal(v2)
            v4 = multiply(in1, v3)
            in2 = load linear__cat__linear_1_reduce[a0, 0, (a1 + 2816)]
            v5 = multiply(in2, v4)
            mul[a0, a1] = v5
```

**Tile IR** — kernel 0 is a `Contraction` with the `trans` B reading:

```
=== 0: k_linear_reduce_6eecae ===
    place  free=(a0, a1)  grid=(a0, a1)
    work   t32x16
    Contraction [Σ a2 in 0..1024] x @ linear__cat__linear_1_w trans -> acc0   ⟨TILE=f4x10⟩
    ├─ a: in1 = load x[a0, a2]   ‹materialized›
    └─ b: in0 = load linear__cat__linear_1_w[a1, a2]   ‹materialized›
    stores
    └─ linear__cat__linear_1_reduce[a0, 0, a1] = acc0
```

Note `trans` — a gmem *layout* fact of the stored `b` edge, the one schedule-flavoured thing that does live on the
node. This trace concatenated gate and up into a single wide projection before recognition, so the node has arity 1.
The arity-2 form — one `a` edge, `channel[0] -> acc_g` and `channel[1] -> acc_u` — is what *sharing is arity* means:
one cone edge read across two channels, no privileged operand slot and no reference arm. Kernel 1 is the pointwise
shape of example 7.

## 7. Pure pointwise — `Map` with no sources

```python
torch.nn.functional.silu(torch.randn(8, 512, 1024)) * torch.randn(8, 512, 1024)
```

**Loop IR**

```
=== 0: k_mul_pointwise ===
    for a0 in 0..8
        for a1 in 0..512
            for a2 in 0..1024
                in1 = load x0[a0, a1, a2]
                v0 = negative(in1)
                v1 = exp(v0)
                v2 = add(1, v1)
                v3 = reciprocal(v2)
                v4 = multiply(in1, v3)
                in2 = load x1[a0, a1, a2]
                v5 = multiply(in2, v4)
                mul[a0, a1, a2] = v5
```

**Tile IR** (abridged — `__u0`…`__u3` are the four unrolled lanes):

```
=== 0: k_mul_pointwise ===
    place  free=(a0, a1, a2)  grid=(a0, a1, a2)
    Map  ‹pointwise›
    └─ fn: λ() -> (v5__u3)
         in1__u0 = load x0[a0, a1, ((a2 * 4) + 0)]
         in2__u0 = load x1[a0, a1, ((a2 * 4) + 0)]
         …
         v0__u0 = negative(in1__u0)
         v1__u0 = exp(v0__u0)
         v2__u0 = add(1, v1__u0)
         v3__u0 = reciprocal(v2__u0)
         v4__u0 = multiply(in1__u0, v3__u0)
         v5__u0 = multiply(in2__u0, v4__u0)
         …
    stores
    ├─ mul[a0, a1, ((a2 * 4) + 0)] = v5__u0
    ├─ mul[a0, a1, ((a2 * 4) + 1)] = v5__u1
    ├─ mul[a0, a1, ((a2 * 4) + 2)] = v5__u2
    └─ mul[a0, a1, ((a2 * 4) + 3)] = v5__u3
```

No node at all — a sourceless `Map`, marked `‹pointwise›`, with an empty binder (`λ()`: nothing to bind, since there
is no source to consume) plus four root `Store`s. This is the one example where the tile form is *longer* than the
loop form, because the schedule's unroll factor has already been applied to the cell.

## 8. Causal SDPA — the flash rewrite

```python
F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (1, 8, 512, 64) each
```

**Loop IR** — two kernels, materializing the full S×S score matrix and reading it back three times:

```
=== 0: k_sdpa_reduce_775196 ===
    for a0 in 0..8
        for a1 in 0..512
            for a2 in 0..512
                for a3 in 0..64
                    in1 = load x1[0, a0, a2, a3]
                    in2 = load x0[0, a0, a1, a3]
                    v0 = multiply(in1, in2)
                    acc0 <- add(acc0, v0)
                v1 = multiply(acc0, 0.125)
                scaled_dot_product_attention_scaled[0, a0, a1, a2] = v1

=== 1: k_sdpa_reduce_e95a7c ===
    for a0 in 0..8
        for a1 in 0..512
            for a2 in 0..512                      # pass 1: row max over the masked scores
                v0 = 0 when ((a2 <= a1))
                     -1e+09 when ((a2 > a1))
                in2 = load scaled_dot_product_attention_scaled[0, a0, a1, a2]
                v1 = add(in2, v0)
                acc0 <- maximum(acc0, v1)
            for a2 in 0..512                      # pass 2: exp-sum
                …
                acc1 <- add(acc1, v5)
            v6 = reciprocal(acc1)
            for a3 in 0..64                       # pass 3: re-read scores, PV
                for a4 in 0..512
                    …
                    in5 = load x2[0, a0, a4, a3]
                    v12 = multiply(in5, v11)
                    acc2 <- add(acc2, v12)
                scaled_dot_product_attention[0, a0, a1, a3] = acc2
```

**Tile IR** — one kernel, one twisted fold over the kv stream:

```
=== 0: scaled_dot_product_attention -> scaled_dot_product_attention ===
    place  free=(b0, b1, m, d)  grid=(b0, b1, m)
    Map
    ├─ source[0]: Fold[kv in 0..512] twisted
    │  ├─ init: (-inf, 0, 0)
    │  ├─ lift: λ(kv, sacc, v_e) -> (s_causal, 1, v_e)
    │  │    scale_c = load _flash_scale[]
    │  │    s = multiply(sacc, scale_c)
    │  │    ninf_c = load _flash_ninf[]
    │  │    s_causal = s when ((kv <= m))
    │  │               ninf_c when (1)
    │  ├─ combine: λ(m_i, l_i, O_i, m_i__o, l_i__o, O_i__o) -> (m_i, l_i, O_i)
    │  │    m_i__o__t0 = maximum(m_i, m_i__o)
    │  │    m_i__o__t1 = subtract(m_i, m_i__o__t0)
    │  │    m_i__o__t2 = exp(m_i__o__t1)
    │  │    m_i__o__t3 = multiply(l_i, m_i__o__t2)
    │  │    m_i__o__t4 = subtract(m_i__o, m_i__o__t0)
    │  │    m_i__o__t5 = exp(m_i__o__t4)
    │  │    m_i__o__t6 = multiply(l_i__o, m_i__o__t5)
    │  │    m_i__o__t7 = multiply(O_i, m_i__o__t2)
    │  │    m_i__o__t8 = multiply(O_i__o, m_i__o__t5)
    │  │    l_i = add(m_i__o__t3, m_i__o__t6)
    │  │    O_i = add(m_i__o__t7, m_i__o__t8)
    │  │    m_i = copy(m_i__o__t0)
    │  ├─ operand[0]: Contraction [Σ dd in 0..64] x0 @ x1 trans -> sacc   ‹computed›
    │  │  ├─ a: q_e = load x0[b0, b1, m, dd]   ‹materialized›
    │  │  └─ b: k_e = load x1[b0, b1, kv, dd]   ‹materialized›
    │  └─ operand[1]: v_e = load x2[b0, b1, kv, d]   ‹materialized›
    └─ fn: λ(m_i, l_i, O_i) -> (O_i__proj)
         O_i__proj = divide(O_i, l_i)
    schedule
    └─ TILE@pj = f64
    stores
    └─ scaled_dot_product_attention[0, b1, m, d] = O_i__proj
```

This is the payoff example, and the tree earns its keep here:

- The whole thing is *the same `Fold` node kind* as `sum(dim=-1)`, on the streaming schedule with the softmax twist —
  a twisted monoid is a monoid, selected structurally, not a distinct kind. `init: (-inf, 0, 0)` and
  `lift: λ(kv, sacc, v_e) -> (s_causal, 1, v_e)` say the state is the triple `(m, l, O)` with singleton `(s, 1, v)`.
- **QK is a stored operand edge; PV is not** — and only QK appears. QK is `operand[0]`, a `‹computed›` `Contraction`
  with its own two-edge subtree, because it is *closed*: it reads `q` and `k` and the enclosing iteration vars, nothing
  else, so it may hoist to an edge and is a PLACE site. PV reads `P = exp(s − m)`, i.e. the **running state**, so it
  can never be an edge; it is synthesized inside the derived evaluation, and since the dump shows only storage it does
  not appear at all. "Edge iff closed" is exactly the line between what is printed and what is not.
- The lift's params `(kv, sacc, v_e)` bind the operand edges **positionally** — `sacc` is `operand[0]`'s output,
  `v_e` is `operand[1]`. That binding is the whole reason there is no let table and no name-reference arm.
- There is no stored `step` sequence. `_flash_scale` / `_flash_ninf` are constant loads, and
  `O_i__proj = divide(O_i, l_i)` in the `Map` body is the projection λ.
- **`schedule` holds the one slice with no stored home.** `TILE@pj = f64` is a decided tile for the synthesized PV
  contraction — a real schedule site (`Site.derived`), below the seam lattice and excluded from PLACE. Its node is
  derived, so rather than reconstruct it inside the term the dump reports the slice where it actually lives: the
  schedule dict. Every other kernel here has all its keys on stored nodes, so none of them show this region.

Also note what is *absent*: no `work` line and no `⟨TILE=…⟩` on the outer fold, because this dump is taken before the
flash schedule is decided. The dump shows decided facts only — an undecided slice is simply not printed.

---

## Summary

| Op | Loop IR | Tile IR |
| --- | --- | --- |
| `sum(dim=-1)` | 3 nested `for` + `acc <-` | `Fold` planar |
| `rms_norm` | reduce nest + normalize nest | `Fold` planar + `Map` + sweep `Store` |
| `softmax` | 3 row passes | `Fold` twisted (1 pass) + `Map` + sweep `Store` |
| `matmul` | 3 nested `for` | `Contraction`, two materialized edges |
| `relu(x@w + b)` | nest + epilogue | `Map` over `Contraction` |
| SwiGLU | reduce kernel + pointwise kernel | `Contraction` (trans) + pointwise `Map` |
| pointwise | 3 nested `for` | `Map(sources=())` + `Store`s, unrolled ×4 |
| causal SDPA | 2 kernels, S×S materialized, 3 row passes | 1 kernel: `Fold` twisted over kv, QK edge + derived PV |

The recurring pattern: **loop IR spells iteration, tile IR spells algebra.** Everything the loop form encodes about
*how* to iterate — passes over a row, materialized intermediates, batch nesting, unroll — is either derived on demand
from the tile node (`op.lower()`, `Fold.step_stmts()`, `Fold.composed`) — and so absent from the dump — or moved off
the term entirely onto the
`TileOp`'s `place` / `schedule` / `stores`. That separation is what lets the schedule search mutate the plan while the
term stays immutable and its α-invariant `term_key` stays the kernel's identity.

## Reproducing

```bash
for S in loop tile; do
  ./venv/bin/emmy compile -c "torch.randn(8,512,1024).sum(dim=-1)" --ir $S
done
```

Substitute any expression from the sections above. For examples needing `nn.Module` parameters, build the module in an
importable file (module construction inside `-c` traces the parameter-init ops and fails on `uniform_`).
