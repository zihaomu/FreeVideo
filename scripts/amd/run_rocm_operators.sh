#!/usr/bin/env bash
# Experimental BF16 hipBLASLt replacements; see r9700-operator-replacement.md.
set -euo pipefail
fv_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export FV_ROCM_GATE_BLAS="${FV_ROCM_GATE_BLAS:-cublaslt}"
export FV_ROCM_STATE_BLAS="${FV_ROCM_STATE_BLAS:-cublaslt}"
exec bash "$fv_script_dir/run_rocm_fast.sh" "$@"
