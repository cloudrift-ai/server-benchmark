# Reading a graph dump

`emmy compile <model> --ir torch` and `--ir tensor` print the graph as pseudocode with Rust-like syntax. Nothing parses it
back; it exists so a reader can follow what the compiler holds at that point. The renderer is
`emmy/compiler/pretty.py`, and `EMMY_DUMP_DIR` writes every stage's graph through the same code.

Torch and Tensor IR represent a very special kind of programs that work in a restricted model.
There is no branching, loops or any other control flow.
The whole program is a static (not changing at execution time dynamically) sequence of applying operations to values,
  starting from input values, and finally arriving at the output values.
Each such operation can be expressed as an assignment of an operation result to a name: `<result_name> = <operation>(<arg1>, ... <argn>)`,
  where args are names of either program inputs or previous operation results.
Once a name `<result_name>` is assigned like that, its value cannot be changed.
Some names are marked as outputs.
Their values at the end of program execution define the program's result.

The pseudocode, in which emmy dumps torch and tensor IRs, looks like a subset of Rust with extra types and primitives for working with tensors.
It aims to concisely express the IR code, while also making it easy to read intuitively for someone who is familiar with Rust.
The following sections introduce by example.

## IR Dump Examples

Three small ones first. A whole-tensor operation, with its own fields after the operands:

```
emmy compile -c 'F.linear(torch.randn(4,64), torch.randn(16,64))' --ir torch
```

```rust
// 3 nodes, 2 inputs, 1 outputs

struct Dynamic {}

struct Inputs<dynamic: Dynamic> {
    x0: f32[4,64],
    x1: f32[16,64],
}

struct Outputs<dynamic: Dynamic> {
    linear: f32[4,16],
}

fn main(dynamic: Dynamic, inputs: Inputs<dynamic>) -> Outputs<dynamic> {
  let linear: f32[4,16] = linear(inputs.x0, inputs.x1, has_bias=False);

  Outputs { linear }
}
```



A scalar added to a tensor. The scalar becomes a constant, and reaching the tensor's shape takes an explicit
node — the closure ignores both indices, which is what a broadcast looks like:

```
emmy compile -c 'torch.randn(4,8) + 1' --ir torch
```

```rust
// 4 nodes, 1 inputs, 1 outputs

struct Dynamic {}

struct Inputs<dynamic: Dynamic> {
    x: f32[4,8],
}

struct Outputs<dynamic: Dynamic> {
    add: f32[4,8],
}

fn main(dynamic: Dynamic, inputs: Inputs<dynamic>) -> Outputs<dynamic> {
  // constants: checkpoint tensors, and the literals the trace captured
  let add_c1: f32[1] = 1.0;

  let add_c1_bc: f32[4,8] = emmy::tensor_from_fn(|_i, _j| add_c1[0]);
  let add: f32[4,8] = add(inputs.x, add_c1_bc);

  Outputs { add }
}
```

Two operations, and three constants that nothing refers to. Resolving a traced call's inputs materializes every
scalar argument as a constant, and a handler that folds the scalar into a field instead — `transpose`'s `axes`,
`softmax`'s `axis` — never wires the constant up. They are real nodes, so they print; the count in the header
includes them:

```
emmy compile -c 'F.softmax(torch.randn(2,4,8).transpose(1,2), dim=-1)' --ir torch
```

```rust
// 6 nodes, 1 inputs, 1 outputs

struct Dynamic {}

struct Inputs<dynamic: Dynamic> {
    x: f32[2,4,8],
}

struct Outputs<dynamic: Dynamic> {
    softmax: f32[2,8,4],
}

fn main(dynamic: Dynamic, inputs: Inputs<dynamic>) -> Outputs<dynamic> {
  // constants: checkpoint tensors, and the literals the trace captured
  let transpose_c1: f32[1] = 1.0;
  let transpose_c2: f32[1] = 2.0;
  let softmax_c1: f32[1] = -1.0;

  let transpose: f32[2,8,4] = transpose(inputs.x, axes=(1, 2));
  let softmax: f32[2,8,4] = softmax(transpose, axis=-1);

  Outputs { softmax }
}
```

Those leftovers do not survive the pipeline. One Qwen3-0.6B layer carries 33 such constants at `--ir torch`
and 4 at `--ir tensor`, and the four that remain are the norm epsilons, which really are operands. A `slice`
keeps its captured scalars as operands too, redundantly with its own fields.

A model layer has the same shape, with more of it. Below is one layer of a Llama-architecture model, cut down
to its landmarks:

```
emmy compile TinyLlama/TinyLlama-1.1B-Chat-v1.0 --layer 0 --ir torch
```

```rust

// 106 nodes, 4 inputs, 1 outputs

struct Dynamic {}

struct Inputs<dynamic: Dynamic> {
    hidden_states: f16[1,512,2048],
    position_embeddings_0: f16[1,512,64],
    position_embeddings_1: f16[1,512,64],
    attention_mask: f32,
}

struct Outputs<dynamic: Dynamic> {
    add_5: f16[1,512,2048],
}

fn main(dynamic: Dynamic, inputs: Inputs<dynamic>) -> Outputs<dynamic> {
  // constants: checkpoint tensors, and the literals the trace captured
  let p_attn_q_proj_weight: f16[2048,2048] = load("self_attn.q_proj.weight");
  // … 38 constants in all

  let p_input_layernorm_weight_bc: f16[1,512,2048] = emmy::tensor_from_fn(|_i, _j, k| p_input_layernorm_weight[k]);
  let to: f32[1,512,2048] = emmy::cast(inputs.hidden_states);
  // … the norm, then the q / k / v projections
  let scaled_dot_product_attention: f16[1,32,512,64]
      = sdpa(add_1, add_2, transpose_2, is_causal=True, sliding_window=None, scale=0.125);
  let linear_3: f16[1,512,2048] = linear(reshape, p_attn_o_proj_weight, has_bias=False);
  let add_3: f16[1,512,2048] = add(inputs.hidden_states, linear_3);
  // … the post-attention norm and the feed-forward block
  let add_5: f16[1,512,2048] = add(add_3, linear_6);

  Outputs { add_5 }
}
```

Three types come first. `Dynamic` carries the symbolic extents a `--dynamic` trace introduced. `Inputs` and
`Outputs` both take it as a parameter, because a shape on either side can be a function of those extents —
`x: f32[2,dynamic.seq_len,4]` — and that parameter is what binds the name an extent refers to. Every dump
prints all three, empty ones included, so one shape fits every graph.

Then `main`. Its body opens with the graph's constants, in their own block under a comment; each compute node
becomes one more `let`; and the tail expression builds `Outputs`. An input is reached through its struct,
`inputs.hidden_states`, while a constant or an earlier result is a plain name.

**Types** carry the shape: `f16[1,512,4096]`. A rank-0 tensor is just its dtype (`f32`). Dtype names are the
repo's own, including the narrow float formats (`f8e4m3`). Under `--dynamic` an extent can be a name or an
expression rather than a number — `f32[2,seq_len,4]`, `f32[1,(2 * seq_len),4]`.

**Order** is a dependency order, not an execution schedule.

**Trailing `key=value` arguments** are the operation's own fields, not tensor operands:
`linear(mul_1, p_attn_q_proj_weight, has_bias=False)`, `reshape(x, shape=(1, 512, -1))`. A `-1` inside `shape=`
is the request the trace captured; the declared type on the left shows the extent it resolved to.

**Constants named `<node>_c<N>`** are scalars the trace captured, named after the traced call they came from
and their position in that call's resolved operand list: `add_c1` is the epsilon inside the node called `add`.

**A node that writes several buffers destructures.** `let (nid, buf1): (f32[4,8], f32[4,8]) = op(x);` — slot 0
travels under the node id, the rest under their own buffer names, so every name a later line can refer to is
bound. Only the later dialects build such nodes; a torch or tensor dump has none.

**Binding names are node ids.** Every node carries two labels: the id the graph stores it under and refers to
it by, and its output tensor's name. They are usually the same string, because a new node takes its tensor name
as its id when that id is free. When it is taken, the id falls back to `n0`, `n1`, … and the tensor keeps its
name — `broadcast_to` names its result after its source, so one source broadcast to two different shapes yields
one name twice. Bindings print the id, so no two can collide, and a differing tensor name follows in a comment:

```rust
let n0: f16[1,4,512,64] = emmy::tensor_from_fn(|i, _j, k, l| unsqueeze[i, 0, k, l]);  // aka tensor name `unsqueeze_bc`
let unsqueeze_bc: f16[1,32,512,64] = emmy::tensor_from_fn(|i, _j, k, l| unsqueeze[i, 0, k, l]);
```

**Memory, loops, threads, kernels, fusion and hardware choices are absent** because these levels cannot express
them, not because the printer hides them. They arrive with the later dialects, which have their own printed
forms.

**Four fields are hidden.** Every operation carries `source` (its predecessor in a rewrite chain), `knobs` (the
tuning choices passes stamp on it), and `inputs` / `outputs`, a derived view of the tensors around it that the
matcher fills in. None of them says what the node computes, and `Graph.to_dict` skips them too when it
serializes a graph. At the torch stage they are empty anyway, since no pass has run yet.

When `EMMY_DUMP_DIR` dumps a later stage, whose nodes hold whole statement trees, an operation prints its own
body instead:

```rust
  let linear: f32[4,16]
      = loop(inputs.x1, inputs.x0) {
        for a0 in 0..4
            for a1 in 0..16
                for a2 in 0..64
                    in0 = load x1[a1, a2]
                    in1 = load x0[a0, a2]
                    v0 = multiply(in0, in1)
                    acc0 <- add(acc0, v0)
                linear[a0, a1] = acc0
  };
```

`acc0 <- add(acc0, v0)` is that dialect's accumulate, not an assignment this notation defines.

## Two stages, one notation

`--ir torch` and `--ir tensor` print through the same renderer, and every reading rule below applies to both.
What changes is the vocabulary of operations, because decomposition runs between them.

A torch dump is the trace as recorded, so it holds model-level operations — `linear`, `sdpa`, `mean`,
`softmax`, `reshape`, `transpose`, `slice`, `cat`, `unsqueeze` — each carrying a whole idea in one node
(`emmy/compiler/ir/frontend/ir.py`). A tensor dump has none of them. Decomposition rewrites every one into the
generic set in `emmy/compiler/ir/tensor/ir.py`: elementwise operations, reductions, scans, the two gathers,
scatter, index maps, casts.

The same Qwen3-0.6B layer at both stages:

```
--ir torch    136 nodes    linear x7, mean x4, transpose x4, slice x4, reshape x4, sdpa, softmax
--ir tensor   153 nodes    none of those; sum x11, divide x5, exp x2, emmy::tensor_from_fn x51
```

A `linear` becomes a broadcast, an elementwise multiply and a `sum` over the contracted axis. A `mean` becomes
that `sum` times a reciprocal. An `sdpa` becomes masked scores, the three-pass softmax spelled out, then the
second contraction. The layout operations become index maps carrying real arithmetic, which is why the builder
count more than doubles: what sat inside a `slice` node's fields at the torch stage is
`linear_4__cat__linear_5[i, j, (k + 3072)]` at the tensor stage.

One node becoming several is why the count grows, even though the pipeline sweeps dead nodes on the way.

## Where a constant's value comes from

| Right-hand side | Meaning |
| --- | --- |
| `1e-06` | a literal fixed at trace time |
| `load("self_attn.q_proj.weight")` | a tensor read by path, from the checkpoint or from the traced wrapper |
| `load("a.weight", "b.weight")` | several tensors read and concatenated along axis 0 |
| `load("x.weight_scale_2").reshape(shape=(1, 1))` | layout operations the loader applies after reading |
| `emmy::const_eval()` | a tensor the loader computes by evaluating a small graph, rather than reading one |
| `context(num_tokens)` | a scalar bound at launch from the symbolic-dimension environment |
| `emmy::input_data()` | a tensor the caller supplies at run time |

A path is relative to whatever wrapper the trace ran over. A single layer gives short names
(`self_attn.q_proj.weight`); a whole model gives long ones (`model.model.layers.0.input_layernorm.weight`), and
buffers keep their own registered names (`causal_mask`) beside them. So one dump can carry both shapes.

The multi-path form appears once passes have run: `035_merge_sibling_linears` fuses the sibling projections, so
a tensor dump reads `load("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight")`.
`emmy::const_eval()` needs a constant the loader computes rather than reads — constant folding collapsing a
chain, or a quantized checkpoint's decode table. `context(...)` needs a decomposition pass and a symbolic
dimension, so it cannot appear at `--ir torch` or in a static-shape dump. `emmy::input_data()` is the fallback
when a constant has no value and no address of any kind, which also covers constants a pass synthesized and
the loader never sees.

## Operations

A bare name means what torch or numpy already means by it. The model-level ones appear only in a torch dump —
`linear`, `mean`, `softmax`, `reshape`, `slice`, `cat`, `transpose`, `sdpa`, `silu` — while `multiply`, `add`,
`pow`, `sum`, `divide`, `exp` and the rest of the scalar and reduction names appear at both stages. Five names
need care.

A **reduction** prints as its combine's name with an `axis=` argument — `sum(pow_1, axis=-1)`,
`maximum(x, axis=-1)`. That is the same spelling as the elementwise operation of the same name, so the `axis=`
argument is the only mark; the reduced axis stays in the result at size 1. A running reduction prints
`scan_sum(x, axis=-1)`, a name neither library has — nothing in the repo builds one today.

**`transpose`** carries an `axes` field, and its length tells you which reading applies: two entries are the
pair torch's `transpose` swaps, more are a full permutation, as numpy's reads.

**`slice`** takes three captured scalars as operands. Two duplicate its `dim` and `start` fields; the third is
the end, which the declared result shape already carries. **`cat`** ends in a captured scalar too, and that one
is not a duplicate: `CatOp` has no fields, so the constant is the only record of the concatenation axis.

**`rms_norm`** and **`layer_norm`** are respellings: the class names behind them would otherwise lowercase to
`rmsnorm` and `layernorm`. Every other operation without an `emmy::` prefix prints under its own name, `sdpa`
included — emmy's short spelling of scaled dot product attention.

An `emmy::` name is emmy's own:

| Name | Meaning |
| --- | --- |
| `emmy::tensor_from_fn(\|i, j\| body)` | build a tensor from a function of its indices, here rank two |
| `emmy::cast(x)` | a dtype change (see the gotchas below — no node computes it) |
| `emmy::bitcast(x)` | reinterpret same-width elements as another dtype |
| `emmy::index_map(a, b)` | a reindexing `emmy::tensor_from_fn` cannot hold |
| `emmy::gather(data, idx, axis=n)` | pick one element per output position |
| `emmy::gather_by_axis(data, idx, axis=n)` | look up whole slices along an axis, by index |

A second namespace, `emmy::intrinsics::`, holds the scalar functions emmy defines itself, the ones with no
torch or numpy function behind them. `emmy/compiler/ir/elementwise.py` holds that table, and everything in it
that torch or numpy does not already name prints under the prefix:

| Name | Meaning |
| --- | --- |
| `emmy::intrinsics::from_f8e4m3(x)`, `from_f8e5m2`, `to_f8e4m3`, `to_f8e5m2` | decode or encode a narrow float format |
| `emmy::intrinsics::bitcast(x)` | the same reinterpretation as `emmy::bitcast`, reached through the intrinsic table |
| `emmy::intrinsics::gelu_tanh(x)` | gelu's tanh approximation, torch's `approximate="tanh"` |
| `emmy::intrinsics::exp_fast(x)` | `exp` with the fast-math CUDA spelling, from a kernel-stage pass |

### What the `emmy::` helpers mean

Written as definitions in the same pseudocode, with three conventions.

`T` and `U` are element types. A shape is one const parameter holding all the axis sizes —
`const d: usize[rank]` — so a definition can index it (`d[axis]`), slice it (`d[0..axis]`) and join pieces
with `++`.

An index into a rank-`r` tensor is itself a vector of `r` numbers, so `x[i]` with `i: usize[r]` is one element,
and `x[i0, i1]` is the same thing written out. Slicing and joining work on an index too: `i[0..axis]` is the
coordinates before the axis, and `i[0..axis] ++ [v] ++ i[axis+1..rank]` is `i` with the coordinate at `axis`
replaced by `v`. An index *tensor* holds `i64`, the dtype a traced index carries; splicing one of its values
into an index vector converts it.

```rust
// Build a tensor from a function of its indices. The closure takes one index
// vector; a printed line names its components instead, `|i, j, k|` for rank three,
// which is why rank six is the limit — the printer has six index names.
fn emmy::tensor_from_fn<T, const rank: usize, const d: usize[rank]>(f: |usize[rank]| -> T) -> T[d]
    // produces the tensor whose element at i is f(i), for every index i with
    // i[k] < d[k]

// A dtype change. No node computes it: the declared result type is the whole
// operation, and the backend converts when it materializes that type. The values
// can still change, since a narrowing conversion rounds.
fn emmy::cast<T, U, const rank: usize, const d: usize[rank]>(x: T[d]) -> U[d]

// Reinterpret each element's bits as another type of the same width.
fn emmy::bitcast<T, U, const rank: usize, const d: usize[rank]>(x: T[d]) -> U[d]
    where size_of::<U>() == size_of::<T>()

// One graph node covers the next two, and the operand shapes decide which: the
// per-element reading applies when `idx` matches `data` in rank and in every axis
// but `axis`, and the slice lookup applies otherwise.

// Pick one element per output position. `idx` has the result's shape, and each of
// its entries says which position along `axis` that one output element reads; the
// other coordinates come from the output position itself. Close to torch's
// `gather`, but not the same — see the gotchas.
fn emmy::gather<T, const rank: usize, const d: usize[rank], const e: usize[rank]>(
    data: T[d],
    idx: i64[e],
    axis: usize,
) -> T[e]
    where e[k] == d[k] for every k != axis
    // produces the tensor whose element at i is
    // data[i[0..axis] ++ [idx[i]] ++ i[axis+1..rank]]

// Look up whole slices by index. Each entry of `idx` names a position along `axis`,
// and the slice of `data` sitting there is copied out; `idx`'s own axes take the
// place of `axis`. A token embedding is this: one row of a table per token id.
fn emmy::gather_by_axis<T, const rank: usize, const irank: usize, const d: usize[rank], const e: usize[irank]>(
    data: T[d],
    idx: i64[e],
    axis: usize,
) -> T[d[0..axis] ++ e ++ d[axis+1..rank]]
    // produces the tensor whose element at i, writing j for the index tensor's own
    // coordinates i[axis..axis+irank], is
    // data[i[0..axis] ++ [idx[j]] ++ i[axis+irank..rank+irank-1]]

// A reindexing the builder above cannot hold: no source, several sources, a
// selection deciding which source feeds which output position, or rank above six.
// A printed call lists the operand tensors; the coordinate map that pairs each one
// with the output index, and the condition under which it supplies a position, stay
// inside the node.
fn emmy::index_map<T, const rank: usize, const d: usize[rank]>(operands: [tensor]) -> T[d]

// Decode or encode a narrow float format, element by element. Unlike `cast`, these
// have a real implementation behind them (`emmy/compiler/ir/elementwise.py`).
fn emmy::intrinsics::from_f8e4m3<const rank: usize, const d: usize[rank]>(x: f8e4m3[d]) -> f32[d]
fn emmy::intrinsics::to_f8e4m3<const rank: usize, const d: usize[rank]>(x: f32[d]) -> f8e4m3[d]
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
let p_input_layernorm_weight_bc: f16[1,512,2048] = emmy::tensor_from_fn(|_i, _j, k| p_input_layernorm_weight[k]);
```

One number per column, repeated over batch and tokens. Coordinates can also be arithmetic over the parameters,
which is how a slice or a dropped axis reads:

```rust
let linear_5: f16[1,512,3072] = emmy::tensor_from_fn(|i, j, k| linear_4__cat__linear_5[i, j, (k + 3072)]);
let linear_3: f16[1,512,1024] = emmy::tensor_from_fn(|i, j, k| linear_3_reduce[i, j, 0, k]);
```

The first reads the upper half of a concatenated tensor, the second drops a size-1 axis.

Valid: every coordinate is arithmetic over the closure's parameters, literals, and — under `--dynamic` — the
symbolic dimension names, which appear free in the body because the launch binds them. A coordinate is never a
value read from another tensor; that is what `emmy::gather` and `emmy::gather_by_axis` are for. Invalid: a
parameter count that disagrees with the result's rank, which would mean the printer has a bug.

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
gathered one, while emmy demands equality there. A narrower index is a legal `torch.gather` that prints as the
other operation: `torch.gather` on data `[4,8]` with an index `[4,3]` at axis 0 prints
`let gather: f32[4,3] = emmy::gather_by_axis(x0, x1, axis=0);`. That line is the one place a dump contradicts
itself — the declared type is torch's answer, while `emmy::gather_by_axis` as defined above would give
`[4,3,8]`, and that is also what emmy computes. Four ATen operators collapse onto one node and only the axis
survives, so operand shapes are the only evidence of which was meant.

**Some captured constants go unused.** The layer above captures 29 scalars and refers to 18 of them. The 11
left over come from `transpose` (8), `unsqueeze` (2) and `sdpa` (1) — the handlers that read a scalar argument
off the trace and store it in a field, rather than keeping the constant the resolver had already made.
