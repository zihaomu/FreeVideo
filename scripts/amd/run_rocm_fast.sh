#!/usr/bin/env bash
# Validated gfx1201 inference and audio decode; see docs/zh-CN/r9700-bottlenecks.md.
set -euo pipefail
fv_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export FV_ROCM_SPATIAL_CONV="${FV_ROCM_SPATIAL_CONV:-triton}"
export FV_ROCM_ATTENTION="${FV_ROCM_ATTENTION:-triton-window}"
export FV_ROCM_VIDEO_BLAS="${FV_ROCM_VIDEO_BLAS:-cublaslt}"
export FV_ROCM_AUDIO_CONV="${FV_ROCM_AUDIO_CONV:-native}"
exec bash "$fv_script_dir/run_rocm.sh" "$@"
