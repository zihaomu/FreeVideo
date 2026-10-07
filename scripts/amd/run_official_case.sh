#!/usr/bin/env bash
# Default workload: official current 1344x768, 10 seconds, 8+3 two-pass.
set -euo pipefail
fv_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export FV_ROCM_SPATIAL_CONV="${FV_ROCM_SPATIAL_CONV:-triton}"
export FV_ROCM_ATTENTION="${FV_ROCM_ATTENTION:-triton-window}"
export FV_ROCM_VIDEO_BLAS="${FV_ROCM_VIDEO_BLAS:-cublaslt}"
export FV_ROCM_AUDIO_CONV="${FV_ROCM_AUDIO_CONV:-native}"
export FV_ROCM_GATE_BLAS="${FV_ROCM_GATE_BLAS:-default}"
export FV_ROCM_STATE_BLAS="${FV_ROCM_STATE_BLAS:-default}"
exec python3 "$fv_script_dir/run_case.py" "$@"
