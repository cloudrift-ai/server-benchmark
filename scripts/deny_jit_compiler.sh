#!/usr/bin/env bash
set -euo pipefail

# Fail a serving qualification run if a request reaches a backend compiler.
# Mount this script under each compiler name at the front of PATH.

log_path=${JIT_GUARD_LOG:-/tmp/jit-compiler-invocations.log}
compiler=$(basename "$0")

# Cache-key discovery may query compiler versions even when every binary is already cached.
if [[ $# -eq 1 && $1 == --version ]]; then
    case "${compiler}" in
        ptxas)
            printf '%s\n' \
                'ptxas: NVIDIA (R) Ptx optimizing assembler' \
                'Copyright (c) 2005-2025 NVIDIA Corporation' \
                'Built on Fri_Feb_21_20:21:21_PST_2025' \
                'Cuda compilation tools, release 12.8, V12.8.93' \
                'Build cuda_12.8.r12.8/compiler.35583870_0'
            ;;
        nvcc)
            printf '%s\n' \
                'nvcc: NVIDIA (R) Cuda compiler driver' \
                'Copyright (c) 2005-2025 NVIDIA Corporation' \
                'Built on Tue_May_27_02:21:03_PDT_2025' \
                'Cuda compilation tools, release 12.9, V12.9.86' \
                'Build cuda_12.9.r12.9/compiler.36037853_0'
            ;;
        ninja)
            printf '%s\n' '1.13.0.git.kitware.jobserver-pipe-1'
            ;;
    esac
    exit 0
fi

printf '%s compiler=%s args=' "$(date --iso-8601=seconds)" "$(basename "$0")" >>"${log_path}"
printf ' %q' "$@" >>"${log_path}"
printf '\n' >>"${log_path}"

exit 86
