# Glossary

This file defines the project-specific and technical language used in the Emmy course and docs. The definitions
describe how a term is used in Emmy; they are not meant to replace a full textbook definition.

## Machine learning and model serving

- **Large language model (LLM)** — A program trained on a large amount of text to predict or generate language. In
  this course, an LLM is mainly a chain of tensor operations that must run on a GPU.
- **Model** — A collection of mathematical operations and learned numbers called weights. The operations describe
  what to compute; the weights contain information learned during training.
- **Weight** — A number learned during model training. A real model contains millions or billions of weights,
  usually stored in tensors.
- **Tensor** — A rectangular collection of numbers. A scalar is a zero-dimensional tensor, a list is one-dimensional,
  a table is two-dimensional, and higher-dimensional tensors can represent batches, tokens, or model features.
- **Shape** — The size of every tensor dimension. A tensor with shape `(2, 8, 16)` has 2 groups, 8 rows per group,
  and 16 values per row. The meaning of each dimension depends on the operation.
- **Dtype (data type)** — The format used to store each tensor value, such as 32-bit floating point (`float32`) or
  16-bit floating point (`float16`). Smaller formats use less memory and can be faster but may lose precision.
- **Inference** — Using an already-trained model to produce an answer. It is different from training, which changes
  the weights.
- **Token** — A small piece of text represented by an integer. A tokenizer converts text into tokens before an LLM
  processes it.
- **Embedding** — A list of numbers representing the meaning or features of an input. Similar inputs often have
  nearby embeddings.
- **Batch** — Several inputs processed together. Batching often makes better use of a GPU.
- **vLLM** — An external model-serving system. It owns the HTTP API, request scheduling, tokenization, batching, and
  other serving work. Emmy can plug compiled model operations into that system.
- **Transformer** — The neural-network architecture used by most modern LLMs. It processes token representations
  through repeated layers containing attention and other tensor operations.
- **KV cache (key/value cache)** — GPU memory that stores attention information for tokens a model has already
  processed. Reusing it avoids recalculating the entire earlier sequence during text generation. A paged KV cache
  manages this memory in fixed-size pieces.
- **Hugging Face** — An ecosystem for publishing and loading models, datasets, and related tools. Emmy can trace
  compatible models loaded through Hugging Face libraries.

## Compiler concepts

- **Compiler** — A program that translates code from one form into another. Emmy translates PyTorch model operations
  into CUDA code.
- **Operation (op)** — One unit of computation, such as add, matrix multiplication, softmax, or RMSNorm.
- **Graph** — A representation of a program as nodes connected by data-flow edges. A node usually represents an
  operation; an edge means that one operation uses a value produced by another.
- **Node** — One item in a graph. In Emmy, it stores an operation, the names of the input buffers it reads (usually
  the producing node's ID), one or more output tensors, and optional hints.
- **Intermediate representation (IR)** — A compiler's internal description of a program. Emmy uses several IR
  stages. Early stages resemble PyTorch; later stages explicitly describe loops, GPU threads, memory, and CUDA
  source.
- **Dialect** — The vocabulary allowed in one IR stage. For example, Tensor IR uses tensor primitives, while Kernel
  IR uses hardware-oriented statements.
- **Frontend / backend** — The frontend reads and represents the source program, such as PyTorch operations. The
  backend turns a lower-level representation into executable code for a target such as CUDA.
- **Pass** — One ordered compiler phase. A pass searches for known patterns and rewrites them into a form suitable
  for the next phase.
- **Rewrite rule** — A small compiler transformation. It recognizes a pattern, such as RMSNorm, and replaces it with
  equivalent lower-level operations.
- **Pipeline** — An ordered sequence of compiler passes. The output of one stage becomes the input to the next.
- **Decomposition** — Breaking a complex operation into simpler operations. Emmy decomposes recognizable PyTorch
  operations into a small set of tensor primitives.
- **Primitive** — A basic operation used as a building block, such as elementwise arithmetic, reduction, or index
  mapping.
- **Elementwise operation** — An operation independently applied to corresponding tensor elements, such as adding
  two tensors.
- **Reduction** — Combining many values into fewer values, such as summing a row or finding its maximum.
- **Broadcasting** — Reusing a smaller tensor across a larger shape. For example, one weight per column can be
  reused for every row.
- **Index map** — A description of how output coordinates correspond to input coordinates. Emmy uses index maps for
  broadcasting, transposing, slicing, and reshaping.
- **Tracing** — Observing a PyTorch program with example inputs to capture the operations and data flow that it
  performs.
- **Lowering** — Moving from a high-level representation to a more detailed, machine-oriented one while preserving
  the program's meaning.
- **Runtime** — The code and state involved while a compiled program is executing. Emmy's CUDA runtime allocates
  buffers, resolves dynamic sizes, and launches kernels.
- **Fusion** — Combining operations so that one GPU kernel performs work that would otherwise require several
  kernels. Fusion can avoid writing temporary values to slower global memory.
- **Static shape** — A tensor shape known when compiling, such as exactly 512 tokens.
- **Dynamic or symbolic shape** — A shape containing a named value, such as `seq_len`, whose actual size is supplied
  when the program runs.
- **Provenance** — Metadata that records where generated code came from. It helps connect a final CUDA kernel back
  to the original model operation when debugging.
- **Metadata** — Information about program data rather than the data itself, such as a tensor's shape, a kernel's
  original operation, or the number of threads needed for launch.
- **Mutable / immutable** — A mutable object can be changed after creation. An immutable object cannot; code creates
  a replacement instead. Emmy's graph is mutable, while many nested compiler statements are immutable.
- **Structural identity / structural key** — A fingerprint based on computation and data flow rather than cosmetic
  names. It lets Emmy recognize equivalent compiler candidates.
- **Idempotent rule** — A rule that does not keep changing its own output when applied again. Compiler rewrite rules
  need this property because a pipeline may revisit earlier rules.
- **JIT compilation (just-in-time compilation)** — Compiling code during program execution, shortly before it is
  needed.

## GPU and CUDA concepts

- **GPU (graphics processing unit)** — Hardware designed to run many similar calculations in parallel. GPUs were
  developed for graphics and are now widely used for machine learning.
- **CUDA** — NVIDIA's platform and programming model for running general-purpose code on NVIDIA GPUs.
- **Kernel** — A function executed on the GPU by many parallel workers. In this course, "kernel" means a GPU
  function, not an operating-system kernel.
- **Thread** — One GPU worker executing a kernel. Threads are organized into blocks.
- **Block / cooperative thread array (CTA)** — A group of GPU threads that can cooperate using fast shared memory
  and synchronization.
- **Warp** — A small group of GPU threads — 32 on current NVIDIA hardware — that execute instructions together.
- **Grid** — All thread blocks launched for one kernel call.
- **Shared memory** — Fast memory shared by threads in one block. It is limited in size and must be managed
  explicitly.
- **Global memory** — The GPU's large main memory. It has much higher capacity than shared memory but is generally
  slower to access.
- **Register** — Very fast storage private to one GPU thread.
- **Tile** — A small rectangular piece of a larger tensor computation. GPU programs process tiles so data can fit in
  fast memory and be shared by nearby threads.
- **Schedule** — The plan for executing correct mathematics on hardware: tile sizes, thread assignments, memory
  movement, and synchronization. Different schedules can compute the same answer at very different speeds.
- **Staging** — Moving a piece of data into faster memory before computing with it, often from GPU global memory to
  shared memory.
- **Synchronization** — Making workers wait until a required event or memory update is complete. It prevents threads
  from reading data before other threads have finished producing it.
- **Contraction** — A multiply-and-combine operation over one or more dimensions. Matrix multiplication is the most
  familiar example.
- **Carrier** — The temporary state maintained by a reduction and the rule for combining partial states. A sum's
  carrier can be one accumulated number; stable softmax needs several related values.
- **Tensor core** — Specialized GPU hardware for performing small matrix operations efficiently.
- **nvcc / NVRTC** — Two NVIDIA CUDA compilers. nvcc compiles ahead of execution; NVRTC compiles CUDA source at
  runtime.
- **CUDA graph capture** — Recording a sequence of GPU launches so the sequence can be replayed with less CPU
  overhead.
- **DLPack** — A standard for sharing tensor memory between libraries without copying the values through CPU memory.

## Deployment and benchmarking

- **CLI (command-line interface)** — The commands and options used from a terminal, such as `emmy compile`.
- **YAML** — A human-readable configuration format. Emmy recipes are written in YAML.
- **Recipe** — A version-controlled plan describing what model or command to run, how to deploy it, which hardware
  and benchmark settings to use, which variants to compare, and how to aggregate results. Commands such as
  `emmy deploy` and `emmy bench` consume recipes; a recipe does not execute by itself.
- **Variant** — One concrete combination of recipe settings. A matrix can expand one recipe into many variants.
- **Matrix** — A recipe section that describes several values to test. `cross` creates every combination; `zip`
  pairs values by position.
- **Benchmark** — A controlled measurement of speed, latency, throughput, or resource use.
- **Latency** — The time needed to complete one operation or request. Lower latency is faster.
- **Throughput** — The amount of work completed per unit of time, such as requests per second. Higher throughput is
  better.
- **VM (virtual machine)** — A computer created by software, often rented from a cloud provider. A GPU VM includes
  one or more GPUs.
- **Provisioning** — Obtaining and preparing a machine. This may include creating a cloud VM, installing software,
  and returning connection details.
- **Deployment** — Placing and starting the application on a machine that is already available.
- **Docker container** — An isolated package containing an application and its dependencies.
- **Docker Compose** — A configuration format and tool for defining and starting several related containers.
- **Smoke test** — A quick check that the deployed service starts and can answer a basic request before expensive
  benchmarks begin.
- **Control plane** — The code that coordinates work — choosing machines, starting services, and collecting
  results — rather than performing the model computation itself.

## Search and tuning

- **Autotuning** — Trying several valid GPU schedules, measuring them, and keeping evidence about which ones are
  fast.
- **Fork** — The point where a rewrite rule returns several correct alternatives instead of one result, because which
  is fastest depends on the hardware and shapes. Each alternative is identified by the knob values it pins. An
  ordinary fork chooses settings within one kernel (tile size, staging, …); these are the forks the prior decides —
  it ranks the options directly whenever no measurement already answers the choice.
- **Structural fork** — A fork whose alternatives change which kernels exist — for example, keeping operations fused
  in one kernel versus splitting them apart. The prior is never asked to rank these options. Instead, the compiler
  compares the total estimated cost of each resulting kernel set; the prior contributes only per-kernel cost
  estimates inside that comparison. Only a trusted online prior — trained and passing calibration — may supply those
  estimates; on the offline prior or a quarantined online prior, the default kernel set is kept.
- **Knob** — A named tuning choice, such as a tile size or memory-staging strategy.
- **Candidate** — One complete set of choices that the compiler could use.
- **Greedy selection** — Choosing the candidate that currently appears best without exploring alternatives during
  normal compilation.
- **MCTS (Monte Carlo tree search)** — A search method that treats decisions as a tree and balances trying promising
  branches with exploring less-tested ones.
- **Prior** — In Emmy, a ranker that estimates which schedule will be fast before the current candidate is measured.
  The offline prior is fitted ahead of time (by `emmy fit`, on the golden dataset) and ships with the repo; the
  online prior learns from collected measurements.
- **Golden configuration** — A reviewed, GPU-specific schedule and latency for a standard problem shape. It is
  trusted deployment evidence and a regression reference.
- **Evidence** — A compatible recorded measurement used to select between schedule candidates.
- **Calibration** — A check of whether a learned model ranks measured candidates well enough to influence
  compilation.
- **Quarantine** — The state in which an online model may continue learning but is not trusted to choose deployed
  schedules.
- **CatBoost** — The machine-learning library Emmy uses for its online schedule-ranking model.
- **SQLite** — A small database stored in one local file. Emmy uses it to persist tuning measurements.

## Common mathematical terms

- **Matrix multiplication (matmul)** — An operation that combines rows of one matrix with columns of another. It is
  one of the most important computations in machine learning.
- **RMSNorm (root mean square normalization)** — An operation that rescales each row according to the root mean
  square of its values. It helps keep values at a stable scale inside many language models.
- **Softmax** — An operation that converts a list of scores into positive values that sum to one. Models use it to
  turn scores into relative weights or probabilities.
- **Attention** — A mechanism that lets each token combine information from other tokens. It uses matrix
  multiplication and softmax as major building blocks.
- **Associative operation** — An operation for which grouping does not change the mathematical result, such as
  `(a + b) + c = a + (b + c)`. Associativity lets a reduction be split across parallel workers. Floating-point
  rounding means the bit-level result can still differ slightly.
- **Numerical precision** — How accurately a number format represents real values. Faster or smaller data types can
  introduce more rounding error.
