# NVFP4 goldens and the measurement trail

Branch-lifetime working note for PR #499. Two jobs: record how the goldens were tuned (honestly — very little
search was involved), and hold the COMPLETE command trail behind every number in the PR description. The
description carries one short command per headline figure; everything else lives here.

Host for every number below: the rented RTX 5090 box (sm_120, CUDA 13.0/12.9, 47 GB). Every command ran under
`run-serial.sh`, a `flock` wrapper that keeps one heavy job on the GPU at a time — GPU contention does not crash
a timing run, it silently corrupts it.

Common environment, exported once per shell:

```
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda PYTHONPATH=<worktree>
```

`CUDA_HOME` and `nvcc` on `PATH` are not optional: without them the suite silently SKIPS every CUDA test and
"passes" in 48 s. That happened once here and the run had to be thrown away.

## 1. Freeze both arms from one traced module

Both programs are frozen once, through the ordinary `--quantize` path, so the arms differ only in the declared
scheme and no re-quantization variance sits between them:

```
emmy compile -c "torch.nn.Linear(4096, 4096, bias=False).half()(torch.randn(512, 4096).half())" \
    --quantize nvfp4       --ir torch -o w4a4.ir  --dump-dir dump-nvfp4
emmy compile -c "torch.nn.Linear(4096, 4096, bias=False).half()(torch.randn(512, 4096).half())" \
    --quantize nvfp4-w4a16 --ir torch -o w4a16.ir --dump-dir dump-nvfp4-w4a16
```

`--ir torch -o` writes the pretty-printed stage, which is NOT what the later commands read; they read the dump's
`dump-<scheme>/00_input.json`, which is `Graph.to_dict()` form. `emmy run <file>.json` loads it via
`Graph.from_dict`; the golden wire decoder (`graph_from_wire`) does not — it wants a list of nodes, not a dict.

## 2. The two recorded goldens

| golden | kernel | knobs | recorded `emmy_us` |
| --- | --- | --- | --- |
| native W4A4 cell | `k_linear_744c36` | `WORK=w2x4 TILE=mma_m16n8k64_e2m1_f32/f4x4/k8 STAGE=d1/smem-async` | 34.6 |
| W4A16 arm | `k_linear_27b9c0` | `WORK=w2x8 TILE=mma_m16n8k16_f16_f32/f4x2/k8 STAGE=d1/smem-tma` | 126.8 |

Produced by the canonical flow, per arm:

```
emmy trace dump-nvfp4/00_input.json --target sm_120 -o golden-nvfp4.yaml
emmy run --golden-file golden-nvfp4.yaml --bench --record --iters 100 --warmup 10
```

and replayed (this is the description's repro command) with:

```
emmy run --golden-file golden-nvfp4.yaml --bench --iters 200 --warmup 20
```

## 3. The tuning that did NOT happen, and why that is fine

**Neither recorded configuration came from a search.** Both are what greedy selects at default on a clean tuning
DB — the native cell fires cold, with no pin and no prior evidence, and so does the W4A16 tier. `--record` only
measured what default selection already produced. Calling that "tuned" would overstate it.

**The one shape that needed a search was left unrecorded.** The activation quantize kernel
(`k_reshape_reduce_e3e181`) costs 3970.6 us compiled whole-program, where it fuses onto a grid of 2. It was swept
exhaustively and prior-free — 125 axis-scoped `REDUCE@a2/a3/a4` combinations and 33 kernel-global `WORK` values,
158 pinned configurations, each compiled and benched — and nothing moved it off grid 2 (3957-3990 us throughout).
No per-kernel winner exists to record.

What fixes it is not a knob: resolving the target on its own, which a working golden does and the deploy path
does anyway. The same kernel identity costs 7.6 us that way. So the honest record is the golden plus the finding,
not a fabricated winner. The only whole-program lever that helps, `PLACE=cut`, is a trade rather than a win — the
quantize drops to 1420 us while the matmul loses its native cell (34.3 -> 127.4 us, worse than the W4A16 arm).

## 4. The sweep driver

The candidate space came from the enumerator, not from guessing at the grammar:

```
python - <<'EOF'
from emmy.compiler.graph import Graph
from emmy.compiler.context import Context
from emmy.compiler.pipeline.search.golden_eval import enumerate_graph
import json
g = Graph.from_dict(json.load(open("dump-nvfp4/00_input.json")))
rows = enumerate_graph(g, Context.from_target((12, 0))).rows      # 3201 whole-program rows
EOF
```

Distinct reduce-side tuples: 232. Distinct matmul-side tuples: 380. The measured sweep then pinned each candidate
through `--ab`, which is repeatable and benches each pinned variant beside the greedy pick:

```
emmy run dump-nvfp4/00_input.json --bench --iters 30 --warmup 3 \
    --ab "REDUCE@a2=r4,REDUCE@a3=r2" --ab "REDUCE@a2=coop,REDUCE@a3=coop" ...   # 25 per invocation
emmy run dump-nvfp4/00_input.json --bench --iters 30 --warmup 3 \
    --ab "WORK=w2x4" --ab "WORK=t256" ...                                        # 11 per invocation
```

Two grammar facts the sweep design turns on. Axis-scoped families (`REDUCE@a2`) isolate per kernel, because axis
names differ between kernels. `WORK` and `RASTER` are KERNEL-GLOBAL and have no `@<axis>` key, so `WORK@a2` is an
unreproducible pin — a global `WORK` pin hits every site at once. And `PLACE` is a PIN, not a knob, so
`enumerate_graph` never returns it: the sweep had that blind spot by construction and `PLACE=cut` was found by
hand afterwards, via `EMMY_PLACE=cut emmy run ...`.

## 5. Clean-DB discipline

An MCTS `emmy tune` run was started and then abandoned; it had already written evidence rows, and greedy reads
them. The same kernel measured 51.4 us before and 34.4 us after, for no code reason. Every number that survived
into the description was re-taken either against a moved-aside DB or through explicit pins, which bypass the
prior entirely. The reset, and the restore afterwards:

```
cp  ~/.cache/emmy/autotune.db  autotune-backup.db          # take the copy FIRST
mv  ~/.cache/emmy/autotune.db  autotune-contaminated.db
mv  ~/.cache/emmy/online.json  online-contaminated.json
emmy run dump-nvfp4/00_input.json --bench --iters 100 --warmup 10 --json clean-nvfp4.json
cp  autotune-backup.db  ~/.cache/emmy/autotune.db          # restore; the DB holds prior work too
```

`--record` stored the Emmy latency alone for both goldens (`no torch.compile timing for <kernel>`): these are
quantized programs torch.compile has no kernel-level equivalent for, so there is no paired reference to store.
The external comparisons are therefore taken separately, at whole-program level, and labelled by question.

## 6. External baselines

Three different questions, three different baselines. All CUDA-graph captured, best-of-5 over 200 replays, same
shape.

**No-quantization reference** — what the shape costs without any of this, through emmy's own bench flow:

```
emmy run -c "torch.nn.Linear(4096, 4096, bias=False).half().cuda()(torch.randn(512, 4096).half().cuda())" \
    --bench --bench-backends eager,tcompile,emmy --iters 200 --warmup 20
```

Eager 94 us, torch.compile 94 us.

**Same-program baseline** — the declared quantize->dequantize->matmul in f16, which is what emmy's W4A4 program
computes. `emmy run -c` cannot host it (the tracer has no `aten.bucketize`), so it is timed directly, with the
same capture harness as the arms above: eager 2720.6 us, `torch.compile` 145.6 us. Driver kept at
`perf/fq.py` on the box; it quantizes to the e2m1 grid through a LUT `bucketize`, per 16-element block along K.

**Same-instruction baseline** — cuBLASLt's own NVFP4 GEMM through `torch._scaled_mm`, on packed `float4_e2m1fn_x2`
operands with `float8_e4m3fn` block scales, `out_dtype=torch.float16`: 16.9 us. Driver at `perf/sm.py`. Note it is
fed synthetic codes and scales and was NOT checked for numerical equivalence with our program — it is a
throughput reference for the same instruction class, not a verified-equivalent computation.

**vLLM's CUTLASS fp4 GEMM: not measured.** vLLM is not importable on the measurement host, by design — the
`serving` extra pins its own torch and would replace the one every compiler test runs against.

## 7. What could not be verified here

`emmy serve` wraps `vllm serve`, so the live serving command cannot run on this box for the same reason. The
serving evidence in the description is therefore the test suite, which does run:

```
pytest tests/serving -q -p no:randomly -n 8 --dist=loadgroup      # 223 passed, 29 skipped, 2 xpassed
```
