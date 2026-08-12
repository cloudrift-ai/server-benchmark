# EXL3 fused MoE CUDA source

The `upstream/` files are the minimal quoted-include closure of ExLlamaV3 v1.4.0 commit
`791c83073f7f90c44f765a0ceeab7a05fa15b96b`. They remain under the upstream MIT license in
`LICENSE.exllamav3`. Emmy flattens this closure into one self-contained translation unit and compiles only the
requested bit rate, codebook, and output tile through its existing content-addressed cubin cache.

The small route-preparation kernel is Emmy code. It prepares the fixed single-token routing ABI and clears the fp32
output before the pinned fused kernel runs. The static-M1 GEMV keeps upstream's staged-weight path on SM80 and newer;
below SM80 its translation-unit builder selects the same header's global-load/DP4A narrow unit because `cp.async` is
not available. No ExLlamaV3 Python or binary dependency is used at runtime.
