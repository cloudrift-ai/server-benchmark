# Reading a graph dump

`emmy compile <model> --ir torch` and `--ir tensor` print the graph as pseudocode with Rust-like syntax. Nothing parses it
back; it exists so a reader can follow what the compiler holds at that point. The renderer is
`emmy/compiler/pretty.py`, and `EMMY_DUMP_DIR` writes every stage's graph through the same code.

Torch and Tensor IR represent a very special kind of programs that work in a restricted model.  There is no branching,
loops or any other control flow.  The whole program is a static (not changing at execution time dynamically) sequence of
applying pre-defined operations to values.  Values in this model are typically tensors, and the operations are common
pytorch/numpy primitives, plus a few ones Emmy adds for convenience.

## Overview

The pseudocode in which emmy dumps torch and tensor IRs is a *dependently-typed*, *purely functional*, *array*
programming language.  Despite using Rust syntax, it removes some of the things Rust can do and adds a few things not
possible in Rust:

- No mutation: no `mut` keyword, once a name is bound with `let`, it can't be rebound.
- No I/O, no side effects. Whole program is `main` function that deterministically maps its arguments to its return value.
- No control flow and no pattern matching: `if`, `for`, `while`, `match`.
- No traits.
- No ownership or borrow checker.
- First-class typed tensors support. Example: `let x : f32[16,8,64] = ...` is a three-dimensional tensor.
- Tensor's shape is given by its type. You may not omit the dimensions that go in `[]` of `f32[16,8,64]`.
- Types (of a struct or a function) can be polymorphic over values. This is particularly useful to support `--dynamic`
  parameters. Example: if some *value* `dyn_len: usize` is in scope, you can declare a tensor whose *type-level* shape
  depends on that value `let x : [16,dyn_len,64]`.

These features are hand-picked from existing functional programming languages (Agda, Idris, Futhark) to concisely
and unambiguously express programs in tensor IR's programming model.  Using Rust syntax allows us to piggyback on a
programmer's knowledge of Rust, and enable them to intuitively read 90% of the code and ease the learning curve.  The
following sections introduce the pseudocode by example.

## Examples

High-level structure of the IR dump is demonstrated below.

```rust
struct Dynamic {
  // dynamic arguments
}

struct Inputs<dynamic: Dynamic> {
  // input tensor shapes, may depend on `dynamic`
}

struct Outputs<dynamic: Dynamic> {
  // output tensor shapes, may depend on `dynamic`
}

// the actual computations of tensor IR program
fn main(dynamic: Dynamic, inputs: Inputs<dynamic>) -> Outputs<dynamic> {
  // constants list
  let constant1 : Type = ...;
  ...

  // assignments defining nodes, each with a tensor operation
  let result1 : Type = operation(arg1, arg2);
  ...

  // final collection of the IR program's outputs, each comes from one of the assignment above
  Outputs { output1, output2, ... }
}
```

### Main function

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

This is a program with two inputs that applies `F.linear` to them.
It does not use any dynamic parameters, so the `Dynamic` and `dynamic` can be disregarded here, they're just noise.
The program's main meaning is in `main`: it maps its `inputs.x0` and `inputs.x1` to `linear` output.

Note that `linear` also takes a named argument `has_bias=False`.
The `False` value does not come from a previous assignment, declared constant or input, this value is inlined in the `linear` call.

### Scalars and constants

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

The `add_c1` in this example is a constant, it does not depend on `inputs`, but some later operations use it as an
argument.  The torch tracing logic has decided that `torch.randn(4,8)` is an input called `inputs.x`, while the plain
literal `1` is a constant `add_c1`.  This example also features the `emmy::tensor_from_fn` primitive, it is used to fill
up an array by given a function of its indices.  Let's look at its application more closely.

```
  let add_c1_bc: f32[4,8] = emmy::tensor_from_fn(|_i, _j| add_c1[0]);
  //                 ^ emmy::tensor_from_fn infers the result size from here, it's not unambiguous,
  //                 so it knows it has to iterate _i over 0..=3 and _j over 0..=7
```

This produces an $$4\times8$$ tensor filled with the same constant `add_c1`.  The `emmy::tensor_from_fn`
primitive is polymorphic over the shape of the tensor it produces, so its type signature is something like `fn
emmy::tensor_from_fn<T, n: usize, m:usize>(...) -> T[n, m]` so it may be called by the same name `emmy::tensor_from_fn`
to produce tensors of different shapes and dimensions. Since the result's type is always annotated, the shape is never
unambiguous.

In practice, tensor IR can't express `emmy::tensor_from_fn` calls with arbitrary lambdas, only some specific restricted
forms.  But it's not a problem for you, since you only read these lambdas and never write them.  For the precise
definition of what lambdas can occur in `emmy::tensor_from_fn` calls, check the tensor IR definition and the code that
generates it.

### Dynamic parameters

```
emmy compile -c 'torch.nn.RMSNorm(4)(torch.randn(2,8,4))' --dynamic 'seq_len@x:1' --ir torch
```

```rust
// 3 nodes, 1 inputs, 1 outputs

struct Dynamic {
    seq_len: usize,
}

struct Inputs<dynamic: Dynamic> {
    x: f32[2,dynamic.seq_len,4],
}

struct Outputs<dynamic: Dynamic> {
    rms_norm: f32[2,dynamic.seq_len,4],
}

fn main(dynamic: Dynamic, inputs: Inputs<dynamic>) -> Outputs<dynamic> {
  // constants: checkpoint tensors, and the literals the trace captured
  let p_weight: f32[4] = load("weight");

  let rms_norm: f32[2,dynamic.seq_len,4] = rms_norm(inputs.x, p_weight, eps=1e-06);

  Outputs { rms_norm }
}
```

Three types come first. `Dynamic` carries the symbolic extents a `--dynamic` trace introduced. `Inputs` and
`Outputs` both take it as a parameter, because a shape on either side can be a function of those extents —
`x: f32[2,dynamic.seq_len,4]` — and that parameter is what binds the name an extent refers to. Every dump
prints all three, empty ones included, so one shape fits every graph.

**Types** carry the shape: `f16[1,512,4096]`. A rank-0 tensor is just its dtype (`f32`). Dtype names are the
repo's own, including the narrow float formats (`f8e4m3`). Under `--dynamic` an extent can be a name or an
expression rather than a number — `f32[2,seq_len,4]`, `f32[1,(2 * seq_len),4]`.

### Detructuring

No traced model produces a node with several outputs — `aten.chunk` becomes one `slice` per piece, and
`aten.split` and `aten.max.dim` are refused. So this listing starts from a graph written straight to an IR
file, which `emmy compile` takes as readily as a model id:

```
python - > /tmp/split.json <<'PY'
import json, sys
from emmy.compiler.graph import Graph
from emmy.compiler.tensor import Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.tensor.ir import ElementwiseOp

g = Graph()
g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4, 8), "f32"), node_id="x")
g.inputs = ["x"]
g.add_node(op=ElementwiseOp(op="negative"), inputs=["x"], node_id="split",
           outputs=[Tensor("lo", (4, 8), "f32"), Tensor("hi", (4, 8), "f32")])
g.outputs = ["split", "hi"]
json.dump(g.to_dict(), sys.stdout)
PY

emmy compile /tmp/split.json --ir torch
```

```rust
// 2 nodes, 1 inputs, 2 outputs

struct Dynamic {}

struct Inputs<dynamic: Dynamic> {
    x: f32[4,8],
}

struct Outputs<dynamic: Dynamic> {
    split: f32[4,8],
    hi: f32[4,8],
}

fn main(dynamic: Dynamic, inputs: Inputs<dynamic>) -> Outputs<dynamic> {
  let (split, hi): (f32[4,8], f32[4,8]) = negative(inputs.x);

  Outputs { split, hi }
}
```

**A node that writes several buffers destructures.** `let (nid, buf1): (f32[4,8], f32[4,8]) = op(x);` — slot 0
travels under the node id, the rest under their own buffer names, so every name a later line can refer to is
bound. Only the later dialects build such nodes; a torch or tensor dump has none.

## Constants

In the simplest case, a constant's value is given by a literal, e.g. `let add_c1: f32[1] = 1.0;`.  But in some cases
they may be loaded from disk (like model weights) or be declared externally in some other way; the key is that all these
values are known before the values of `inputs` tensors are chosen, and such constants don't depend on inputs.  This
section summarizes a few examples of syntax our pseudocode uses for that.

| Right-hand side | Meaning |
| --- | --- |
| `1e-06` | a literal fixed at trace time |
| `load("self_attn.q_proj.weight")` | a tensor read by path, from the checkpoint or from the traced wrapper |
| `load("a.weight", "b.weight")` | several tensors read and concatenated along axis 0 |
| `load("x.weight_scale_2").reshape(shape=(1, 1))` | layout operations the loader applies after reading |
| `context(num_tokens)` | a scalar bound at launch from the symbolic-dimension environment |
| `emmy::const_eval()` | a tensor the loader computes by evaluating a small graph, rather than reading one |
| `emmy::input_data()` | a tensor the caller supplies at run time |

The last two are the less common ones; feel free to skip them if they don't make sense at first.

## Operations: PyTorch/NumPy and Emmy

We've seen a few examples of applying operations to tensors inside `main`, like

```rust
let linear: f32[4,16] = linear(inputs.x0, inputs.x1, has_bias=False);
```

At this point, you may be wondering where do operation names like `linear` come from, and whether there's an exhaustive
list of operation names one might see in tensor IR. They come from PyTorch/NumPy operations that are valid in PyTorch
graph, as well as a few of Emmy's own operations. We mark Emmy's own operations with `emmy::` namespace. Operations that
don't have a namespace like `linear`, `add` or `mul` are from either PyTorch or NumPy. We can't tell exactly which of the
two is meant in each case, because the tracing code does not thoroughly record the source.

Below, we overview the `emmy::` operations.

| Name | Meaning |
| --- | --- |
| `emmy::tensor_from_fn(\|i, j\| body)` | build a tensor from a function of its indices (rank 2 in this example for simplicity, in principle this supports any rank) |
| `emmy::cast(x)` | a dtype change, the IR uses it to make the implicit casts in graph IR explicit |
| `emmy::bitcast(x)` | reinterpret same-width elements as another dtype |
| `emmy::index_map(a, b)` | a reindexing `emmy::tensor_from_fn` cannot hold |
| `emmy::gather(data, idx, axis=n)` | pick one element per output position |
| `emmy::gather_by_axis(data, idx, axis=n)` | look up whole slices along an axis, by index |

The `emmy::gather` and `emmy::gather_by_axis` are internally represented as one operation `"gather"` in graph IR.
Depending on the shapes of inputs and outputs, the `"gather"` in graph IR does one of two unrelated things.  We
make the two cases easy to distinguish by differentiating them at IR dump time, and calling them `emmy::gather` and
`emmy::gather_by_axis`. The `emmy::gather` is very similar to PyTorch `gather`, but has some slight differences, so we
keep it distinct.

A second namespace, `emmy::scalar::`, holds functions on a single number, applied to scalars. They are
the entries of one table — `_NAME_TO_FN` in `emmy/compiler/ir/elementwise.py` — minus the ones torch or numpy
already names, which print bare.

| Name | Meaning |
| --- | --- |
| `emmy::scalar::from_f8e4m3(x)`, `from_f8e5m2`, `to_f8e4m3`, `to_f8e5m2` | decode or encode a narrow float format |
| `emmy::scalar::bitcast(x)` | the same reinterpretation as `emmy::bitcast`, reached through that table |
| `emmy::scalar::gelu_tanh(x)` | gelu's tanh approximation, torch's `approximate="tanh"` |
| `emmy::scalar::exp_fast(x)` | `exp` with the fast-math CUDA spelling, from a kernel-stage pass |

The following section presents `emmy::` helper operations in more detail.
We do not discuss the PyTorch- and NumPy-based operations in detail, consult those repsective projects for that.

### The `emmy::` helpers

In this section, we will have a deeper look at the `emmy::*` tensor opreations.
Before we start, let's recap our notation for tensor types and shapes we use in torch IR.

A tensor is an $$n$$-dimensional array.
When $$n=1$$, it's just a vector indexed by a single value; when $$n=2$$ it's a 2D matrix.
In our IR, a tensor value's type must specify its size across every dimension.

```rust
fn main(dynamic: Dynamic, inputs: Inputs<dynamic>) -> Outputs<dynamic> {
  // examples of declaring 1D and 2D tensors
  let tensor_1d: f32[16] = ...
  let tensor_2d: f32[8, 32] = ...;
}
```

Most of the time, tensor's size across each dimensions is just a constant literal like 16, 8 or 32 above.
Sometimes, a tensor's size across some of the dimensions can be give by a `dynamic` variable or some expression of `dynamic` variables.

```rust
fn main(dynamic: Dynamic, inputs: Inputs<dynamic>) -> Outputs<dynamic> {
	let var_len_tensor: f32[dynamic.dyn_len] = ...;
}
```

The number of tensor's dimensions though is always fixed and known inside `main`.  You can never have a tensor value
whose number of dimensions varies with tensor IR program inputs.  But the `emmy::*` operations we present here are
re-usable across many different programs, and we'd like them to work with tensors of any size and dimensionality.  So in
order to show their types correctly and concisely, we empower notation with tensors of variable dimensionality.  This
extension of our Rust pseudocode is only needed to present the `emmy::` helper operations in this section, it's not used
in actual IR dumps.  In the following, we define this pseudocode extension and present the `emmy::` tensor helpers using
it.

A 1-dimensional tensor is just a vector. Its shape is given by a single number: vector's length.  A 2-dimensional tensor
is a 2D matrix, its shape is a pair of numbers that specify the matrix height and width.  Likewise, the shape of an
$$n$$-dimensional tensor is a tuple of $$n$$ integers, specifying its size across each of $$n$$ axes.  Let `n` be some
integer. Then:

1. `shape: usize[n]` is a 1D vector of `n` integers.
2. `t: f32[shape]` denodes an `n`-dimensional tensor of `f32` whose size across axis `i` is `shape[i]`.
3. `i: usize[n]` is another 1D vector on `n` integers such that `i[k] < shape[k]` for all `k`.
4. `t[i]: f32` is a scalar in `t` addressed by `i`. Note that `i` that serves as an index into the tensor has the same
   type `usize[n]` as the tensor shape.

We use standard notation for slicing vectors/tensors and concatenating them. For example, `t[[i[0]+1] + i[1..]]` is the
element just one position "up" from `t[i]` in `t` across axis 0.

The following sections present `emmy::` tensor operation helpers. Their type signatures use this tensor notation
heavily, this helps precisely pin down the types and shapes of tensors they consume and produce. We don't show their
implementation as code, but intead define it in prose.

#### `emmy::cast`

A dtype change. The graph IR does not have an explicit node for cast operation: casts there are implicit, and assumed
to happen whenever an tensor of one type is assigned to a tensor of a different type. In tensor IR pseudocode dumps, we
print them explicitly since such conversions actually change the data.

```rust
fn emmy::cast<T, U, const rank: usize, const d: usize[rank]>(x: T[d]) -> U[d]
```

Note that the dtype changes from `T` to `U`, while the shape `d` stays the same.

#### `emmy::bitcast`

Reinterprets each element's bits as another type of the same width.

```rust
fn emmy::bitcast<T, U, const rank: usize, const d: usize[rank]>(x: T[d]) -> U[d]
    where size_of::<U>() == size_of::<T>()
```

#### `emmy::tensor_from_fn`

Builds a tensor from a lambda function that maps tensor indices to values. The lambda takes one index vector; a printed
line names its components instead, `|i, j, k|` for rank three, which is why rank six is the limit — the printer has
six index names.

```rust
fn emmy::tensor_from_fn<T, const rank: usize, const d: usize[rank]>(f: |usize[rank]| -> T) -> T[d]
    // produces the tensor whose element at i is f(i), for every index i with
    // i[k] < d[k]
```

Inside the closure, a tensor is read with brackets. An axis the body never uses takes Rust's `_` prefix, which
is how a broadcast looks:

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

#### `emmy::gather`

Picks one element per output position. `idx` has the result's shape, and each of its entries says which
position along `axis` that one output element reads; the other coordinates come from the output position
itself.

Close to torch's `gather`, but stricter. Torch allows `idx` to be smaller than `data` on the axes it does not
gather; `emmy::gather` requires them equal there.

```rust
fn emmy::gather<T, const rank: usize, const d: usize[rank], const e: usize[rank]>(
    data: T[d],
    idx: i64[e],
    axis: usize,
) -> T[e]
    where e[k] == d[k] for every k != axis
    // result[i] = data[i[0..axis] ++ [idx[i]] ++ i[axis+1..rank]]
```

#### `emmy::gather_by_axis`

Looks up whole slices by index. Each entry of `idx` names a position along `axis`, and the slice of `data`
sitting there is copied out; `idx`'s own axes take the place of `axis`. A token embedding is this: one row of a
table per token id.

```rust
fn emmy::gather_by_axis<T, const rank: usize, const irank: usize, const d: usize[rank], const e: usize[irank]>(
    data: T[d],
    idx: i64[e],
    axis: usize,
) -> T[d[0..axis] ++ e ++ d[axis+1..rank]]
    // result[i] = data[i[0..axis] ++ [idx[i[axis..axis+irank]]] ++ i[axis+irank..rank+irank-1]]
```

#### `emmy::index_map`

Reads values from several tensors into one result, choosing per output position which tensor to read and where.
A source is three things: an operand, a map from an output index to an index into that operand, and a condition
selecting the output positions it supplies. Sources are ordered, and the first whose condition holds at a
position supplies it.

```rust
struct IndexSource<T, const rank: usize> {
  operand_rank: usize,
  operand_d: usize[operand_rank],
  operand: T[operand_d],
  coord: |usize[rank]| -> usize[operand_rank],
  select: |usize[rank]| -> bool,
}

fn emmy::index_map<T, const rank: usize, const d: usize[rank]>(
    sources: IndexSource<T, rank>[],
) -> T[d]
    // result[i] = s.operand[s.coord(i)]
    //   where s is the first source with s.select(i)
```

A `cat` is the everyday case:

```
emmy compile -c 'torch.cat([torch.randn(2,3), torch.randn(2,5)], dim=1)' --ir tensor
```

```rust
let cat: f32[2,8]
    = emmy::index_map([
    IndexSource { operand: inputs.x0, coord: |i, j| [i, ((j < 3) ? j : 0)], select: |_i, j| (j < 3) },
    IndexSource { operand: inputs.x1, coord: |i, j| [i, ((j < 3) ? 0 : (j - 3))] },
  ]);
```

The first source supplies the columns below 3 and reads `x0` there; the second has no condition, so it takes
what is left. Both coordinate maps are guarded, so each one stays inside its operand even at the positions the
other source supplies.

A source carries its operand's rank and shape as its own fields because the operands need not agree on either
— `f32[2,3]` and `f32[2,5]` above, and a mask built from two `f16[1]` scalars. They do agree on the element
type: `Graph.validate` rejects a node whose sources disagree on it, since converting is `emmy::cast`'s job.

One node class covers both spellings. `emmy::tensor_from_fn` and `emmy::index_map` are the same underlying
operation, which is general enough to read from several tensors under a condition. Almost no node needs that:
of the 154 index maps in a TinyLlama and a Qwen3-0.6B layer, 148 read one tensor with no condition, and those
are the broadcasts, reshapes, transposes, slices and unsqueezes a reader meets on nearly every line. Printing
the general form for all of them would bury a one-line coordinate map in the record around it, so the simple
case gets the simple spelling and `emmy::index_map` prints only what the closure form cannot hold.
