# Reading a graph dump

`emmy compile <model> --ir torch` and `--ir tensor` print the graph as Rust-shaped pseudocode. Nothing parses it
back; it exists so a reader can follow what the compiler holds at that point. The renderer is
`emmy/compiler/pretty.py`, and `EMMY_DUMP_DIR` writes every stage's graph through the same code.

## The program

Three small ones first, from `emmy compile --code`. A whole-tensor operation, with its own fields after the
operands:

```rust
// 3 nodes, 2 inputs, 1 outputs

pub fn main(
    x0: f32[4,64],
    x1: f32[16,64],
) -> f32[4,16] {
    let linear: f32[4,16] = linear(x0, x1, has_bias=False);

    linear
}
```

A scalar added to a tensor. The scalar becomes a constant, and reaching the tensor's shape takes an explicit
node — the closure ignores both indices, which is what a broadcast looks like:

```rust
// 4 nodes, 1 inputs, 1 outputs

let add_c1: f32[1] = 1.0;

pub fn main(
    x: f32[4,8],
) -> f32[4,8] {
    let add_c1_bc: f32[4,8] = emmy::tensor_from_fn(|_i, _j| add_c1[0]);
    let add: f32[4,8] = add(x, add_c1_bc);

    add
}
```

Two operations, and the constants the trace captured along the way — `transpose_c1` and `softmax_c1` end up
bound and unused, since the fields carry the same information:

```rust
// 6 nodes, 1 inputs, 1 outputs

let transpose_c1: f32[1] = 1.0;
let transpose_c2: f32[1] = 2.0;
let softmax_c1: f32[1] = -1.0;

pub fn main(
    x: f32[2,4,8],
) -> f32[2,8,4] {
    let transpose: f32[2,8,4] = transpose(x, axes=(1, 2));
    let softmax: f32[2,8,4] = softmax(transpose, axis=-1);

    softmax
}
```

A model layer has the same shape, with more of it. This is `--layer 0` of a Llama-architecture model, cut down
to its landmarks:

```rust
// 106 nodes, 4 inputs, 1 outputs

let p_input_layernorm_weight: f16[2048] = load("input_layernorm.weight");
let p_attn_q_proj_weight: f16[2048,2048] = load("self_attn.q_proj.weight");
// … 33 constants in all

pub fn main(
    hidden_states: f16[1,512,2048],
    position_embeddings_0: f16[1,512,64],
    position_embeddings_1: f16[1,512,64],
    attention_mask: f32,
) -> f16[1,512,2048] {
    let p_input_layernorm_weight_bc: f16[1,512,2048]
        = emmy::tensor_from_fn(|_i, _j, k| p_input_layernorm_weight[k]);
    let to: f32[1,512,2048] = emmy::cast(hidden_states);
    // … the norm, then the q / k / v projections
    let scaled_dot_product_attention: f16[1,32,512,64]
        = sdpa(add_1, add_2, transpose_2, is_causal=True, sliding_window=None, scale=0.125);
    let linear_3: f16[1,512,2048] = linear(reshape, p_attn_o_proj_weight, has_bias=False);
    let add_3: f16[1,512,2048] = add(hidden_states, linear_3);
    // … the post-attention norm and the feed-forward block
    let add_5: f16[1,512,2048] = add(add_3, linear_6);

    add_5
}
```

The graph's constants become module-level bindings, its inputs become the parameters of `pub fn main`, every
compute node becomes one `let`, and its outputs become the tail expression. A graph with several outputs
returns an `Outputs` struct, declared after the function; a graph with none returns `()`.

```rust
pub fn main(
    x0: f32[4,8],
    x1: f32[4,8],
) -> Outputs {
    let add: f32[4,8] = add(x0, add_c1_bc);
    let mul: f32[4,8] = multiply(x1, mul_c1_bc);

    Outputs { add, mul }
}

struct Outputs {
    add: f32[4,8],
    mul: f32[4,8],
}
```

**Types** carry the shape: `f16[1,512,4096]`. A rank-0 tensor is just its dtype (`f32`). Dtype names are the
repo's own, including the narrow float formats (`f8e4m3`).

**Order** is a dependency order, not an execution schedule.

**Trailing `key=value` arguments** are the operation's own fields, not tensor operands:
`linear(mul_1, p_attn_q_proj_weight, has_bias=False)`, `reshape(x, shape=(1, 512, -1))`. A `-1` inside `shape=`
is the request the trace captured; the declared type on the left shows the extent it resolved to.

**Constants named `<node>_c<N>`** are scalars the trace captured, named after the node that consumes them and
the argument position they sat at: `add_c1` is the epsilon inside the node called `add`.

**Binding names are node ids.** Every node carries two labels: the id the graph stores it under and refers to
it by, and its output tensor's name. They are usually the same string, because a new node takes its tensor name
as its id when that id is free. When it is taken, the id falls back to `n0`, `n1`, … and the tensor keeps its
name — `broadcast_to` names its result after its source, so one source broadcast to two different shapes yields
one name twice. Bindings print the id, so no two can collide, and a differing tensor name follows in a comment:

```rust
let n0: f16[1,8,512,128] = emmy::tensor_from_fn(|i, _j, k, l| unsqueeze[i, 0, k, l]);  // aka tensor name `unsqueeze_bc`
let unsqueeze_bc: f16[1,32,512,128] = emmy::tensor_from_fn(|i, _j, k, l| unsqueeze[i, 0, k, l]);
```

**Memory, loops, threads, kernels, fusion and hardware choices are absent** because these levels cannot express
them, not because the printer hides them. They arrive with the later dialects, which have their own printed
forms.

**Four fields are hidden.** Every operation carries `source` (its predecessor in a rewrite chain), `knobs` (the
tuning choices passes stamp on it), and a map of its own input and output tensors. They record where a node
came from rather than what it computes — `Graph.to_dict` skips the same four when it serializes a graph — and
at the torch stage they are empty anyway, since no pass has run yet.

When `EMMY_DUMP_DIR` dumps a later stage, whose nodes hold whole statement trees, an operation prints its own
body instead:

```rust
    let linear: f32[4,16]
        = loop(x1, x0) {
        for a0 in 0..4
            for a1 in 0..16
                for a2 in 0..64
                    in0 = load x1[a1, a2]
                    ...
    };
```

## Where a constant's value comes from

| Right-hand side | Meaning |
| --- | --- |
| `1e-06` | a literal fixed at trace time |
| `load("model.layers.0…")` | a tensor read by path, from the checkpoint or from the traced wrapper |
| `load("…").reshape(shape=(1, 1))` | layout operations the loader applies after reading, in order |
| `emmy::const_eval()` | a tensor the loader computes by evaluating a small graph, rather than reading one |
| `context(num_tokens)` | a scalar bound at launch from the symbolic-dimension environment |
| `emmy::input_data()` | a tensor the caller supplies at run time |

Two notes on paths. A quantized checkpoint's dump mixes two naming schemes: parameters are rewritten to
absolute model paths (`model.layers.0.self_attn.q_proj.weight`), while buffers keep the traced wrapper's own
names (`causal_mask`). Unquantized dumps leave both wrapper-relative.

The last three rows are rare. `context(...)` needs a decomposition pass and a symbolic dimension, so it cannot
appear at `--ir torch` or in a static-shape dump. A caller-supplied tensor and the multi-path form
(`load("…a", "…b")`, concatenated on axis 0) exist in the renderer, but no traced model produces them.

## Operations

A bare name means what torch or numpy already means by it: `multiply`, `add`, `pow`, `mean`, `sum`, `linear`,
`silu`, `softmax`, `reshape`, `slice`, `cat`, `transpose`. Four cases need care.

A **reduction** prints as its combine's name with an `axis=` argument — `sum(pow_1, axis=-1)`,
`maximum(x, axis=-1)`. That is the same spelling as the elementwise operation of the same name, so the `axis=`
argument is the only mark; the reduced axis stays in the result at size 1.

**`transpose`** carries an `axes` field that is either a pair to swap, as torch's `transpose` takes, or a full
permutation, as numpy's reads. Both print identically, so check the declared result type.

**`slice`** also carries the scalar constants the trace captured, which duplicate its own fields. **`cat`** ends
in a captured scalar too, but that one is not a duplicate: `CatOp` has no fields, so the constant is the only
record of the concatenation axis.

**`rms_norm`** and **`layer_norm`** are respellings: the class names behind them would otherwise lowercase to
`rmsnorm` and `layernorm`. Every other operation without an `emmy::` prefix prints under its own name, `sdpa`
included — emmy's short spelling of scaled dot product attention.

An `emmy::` name is emmy's own:

| Name | Meaning |
| --- | --- |
| `emmy::tensor_from_fn(\|i, j\| …)` | build a tensor from a function of its indices |
| `emmy::cast(x)` | a dtype change (see the gotchas below — no node computes it) |
| `emmy::bitcast(x)` | reinterpret same-width elements as another dtype |
| `emmy::index_map(…)` | a reindexing the closure form cannot hold |
| `emmy::gather(data, idx, axis=n)` | pick one element per output position |
| `emmy::gather_by_axis(data, idx, axis=n)` | look up whole slices along an axis, by index |

A second namespace, `emmy::intrinsics::`, holds the scalar functions emmy defines itself, the ones with no
torch or numpy function behind them. `emmy/compiler/ir/elementwise.py` holds that table, and everything in it
that torch or numpy does not already name prints under the prefix:

| Name | Meaning |
| --- | --- |
| `emmy::intrinsics::from_f8e4m3(x)`, `emmy::intrinsics::to_f8e5m2(x)`, … | decode or encode a narrow float format |
| `emmy::intrinsics::bitcast(x)` | the elementwise spelling of a same-width reinterpretation |
| `emmy::intrinsics::gelu_tanh(x)` | gelu's tanh approximation, torch's `approximate="tanh"` |
| `emmy::intrinsics::exp_fast(x)` | `exp` with the fast-math CUDA spelling, from a kernel-stage pass |

### What the `emmy::` helpers mean

Written as definitions in the same pseudocode, with three conventions.

`T` and `U` are element types. A shape is one const parameter holding all the axis sizes — `const d: usize[n]`
— so a definition can index it (`d[axis]`), slice it (`d[0..axis]`) and join pieces with `++`.

An index into a rank-`n` tensor is itself a vector of `n` numbers, so `x[i]` with `i: usize[n]` is one element,
and `x[i0, i1]` is the same thing written out. Slicing and joining work on an index too: `i[0..axis]` is the
coordinates before the axis, and `i[0..axis] ++ [v] ++ i[axis+1..n]` is `i` with the coordinate at `axis`
replaced by `v`.

```rust
// Build a tensor from a function of its indices. The result type fixes how many
// parameters the closure takes and what each one ranges over. Rank six is the
// limit, because the printer has six index names.
fn emmy::tensor_from_fn<T, const n: usize, const d: usize[n]>(f: |usize, …, usize| -> T) -> T[d]
    // produces the tensor whose element at i is f(i), for every index i with
    // i[k] < d[k]

// A dtype change. No node computes it: the declared result type is the whole
// operation, and the backend converts when it materializes that type. The values
// can still change, since a narrowing conversion rounds.
fn emmy::cast<T, U, const n: usize, const d: usize[n]>(x: T[d]) -> U[d]

// Reinterpret each element's bits as another type of the same width.
fn emmy::bitcast<T, U, const n: usize, const d: usize[n]>(x: T[d]) -> U[d]
    where size_of::<U>() == size_of::<T>()

// One graph node covers the next two, and the operand shapes decide which: the
// per-element reading applies when `idx` matches `data` in rank and in every axis
// but `axis`, and the slice lookup applies otherwise.

// Pick one element per output position. `idx` has the result's shape, and each of
// its entries says which position along `axis` that one output element reads; the
// other coordinates come from the output position itself. Close to torch's
// `gather`, but not the same — see the gotchas.
fn emmy::gather<T, const n: usize, const d: usize[n], const e: usize[n]>(
    data: T[d],
    idx: i64[e],
    axis: usize,
) -> T[e]
    where e[k] == d[k] for every k != axis
    // produces the tensor whose element at i is
    // data[i[0..axis] ++ [idx[i]] ++ i[axis+1..n]]

// Look up whole slices by index. Each entry of `idx` names a position along `axis`,
// and the slice of `data` sitting there is copied out; `idx`'s own axes take the
// place of `axis`. A token embedding is this: one row of a table per token id.
fn emmy::gather_by_axis<T, const n: usize, const m: usize, const d: usize[n], const e: usize[m]>(
    data: T[d],
    idx: i64[e],
    axis: usize,
) -> T[d[0..axis] ++ e ++ d[axis+1..n]]
    // produces the tensor whose element at i, writing j for the index tensor's own
    // coordinates i[axis..axis+m], is
    // data[i[0..axis] ++ [idx[j]] ++ i[axis+m..n+m-1]]

// A reindexing the closure form cannot hold: no source, several sources, a
// selection deciding which source feeds which output position, or rank above six.
fn emmy::index_map<T, const n: usize, const d: usize[n]>(sources: …) -> T[d]

// Decode or encode a narrow float format, element by element. Unlike `cast`, these
// have a real implementation behind them (`emmy/compiler/ir/elementwise.py`).
fn emmy::intrinsics::from_f8e4m3<const n: usize, const d: usize[n]>(x: f8e4m3[d]) -> f32[d]
fn emmy::intrinsics::to_f8e4m3<const n: usize, const d: usize[n]>(x: f32[d]) -> f8e4m3[d]
```

`emmy::const_eval()` and `emmy::input_data()` are not operations. They stand for where a constant's value comes
from and appear only on the right of a module-level binding. `const_eval` means the loader computes the value
by running a small graph — a chain of layout operations over other constants that folding collapsed into one.
`input_data` means the constant has no value and no address at all, so whoever runs the graph must supply it by
name.

### Reading `emmy::tensor_from_fn`

The closure takes one parameter per axis of the **result**, and the type gives their ranges. Inside, a tensor
is read with brackets. An axis the body never uses takes Rust's `_` prefix, which is how a broadcast looks:

```rust
let p_input_layernorm_weight_bc: f16[1,512,4096]
    = emmy::tensor_from_fn(|_i, _j, k| p_input_layernorm_weight[k]);
```

One number per column, repeated over batch and tokens. A read whose index is itself a tensor value — a table
lookup — prints the same way, with the inner bracket producing the row:

```rust
let pairs: f16[4096,2048,2] = emmy::tensor_from_fn(|i, j, k| pair_table[idx[i, j], k]);
```

Valid: every index variable in the body comes from the closure's parameters, or is a literal, or is another
tensor read. Invalid: a free variable, or a parameter count that disagrees with the result's rank — if you see
either, the printer has a bug.

## Gotchas

These are oddities of the compiler that the new spelling makes visible rather than hiding.

**A conversion has no operation of its own.** It is an identity node whose output tensor *declares* the new
dtype, and the backend converts when it materializes that type. `emmy::cast` is the printer naming a node it
recognized by comparing the declared types.

**`CastOp` exists and nothing builds it.** `emmy/compiler/ir/tensor/ir.py` defines a cast operation carrying an
explicit dtype. `torch_wire.py` can rebuild one from a stored graph and the constant-folding pass knows it, but
no production code creates one, so every real conversion arrives as the identity node above. `copy` and
`bitcast` are the two elementwise names whose implementation is the identity; the narrow-float decoders do
compute.

**A binding can be a plain rebinding.** An identity node whose declared dtype already matches its input prints
as its argument alone — `005_split_cast_from_indexmap.py` leaves one behind whenever it splits a conversion out
of an index map, so a `--ir tensor` dump carries several:

```rust
let to: f32[1,512,1024] = to_cast;
```

**`emmy::gather` is not `torch.gather`.** The formula is torch's, and on the inputs emmy accepts the two agree
element for element. But torch requires only `index.size(d) <= input.size(d)` on every axis other than the
gathered one, while emmy demands equality there. A narrower index is a legal `torch.gather` that emmy reads as
the other operation: data `[4,8]` with an index `[4,3]` on axis 0 gives `[4,3]` in torch and `[4,3,8]` here.
Four ATen operators collapse onto one node and only the axis survives, so operand shapes are the only evidence
of which was meant.

**Some captured constants go unused.** `transpose` and `softmax` leave the scalars the trace captured bound at
module level, with nothing referring to them.
