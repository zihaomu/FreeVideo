#!/usr/bin/env bash
# One isolated R9700; all writable state stays on the data disk.
set -euo pipefail
fv_checkout="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$fv_checkout/scripts/amd/storage.sh"
fv_image="${FV_ROCM_IMAGE:-sha256:55bf8baa2a513b1c05bd256119fbc57a6ca64170e6cf5fe519b1cdd0c458cfd9}"
fv_render="${FV_RENDER_DEVICE:-/dev/dri/renderD130}"
fv_gpu_uuid="${FV_GPU_UUID:-GPU-b11d6bcf3a61a551}"
fv_cache_group="${FV_CACHE_GROUP:-shared}"
[[ "$fv_cache_group" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
fv_python=python3
fv_pythonpath="$fv_container_experiment_root/envs/r9700-site:/workspace"
if [[ -x "$fv_experiment_root/envs/rocm/bin/python" ]]; then
  fv_python="$fv_container_experiment_root/envs/rocm/bin/python"
  fv_pythonpath=/workspace
fi
mkdir -p "$fv_experiment_root"/{runtime,reports,outputs,cache,kernel-cache,envs}
mkdir -p "$fv_experiment_root/cache/miopen" "$fv_experiment_root/cache/config"
exec docker run --rm --pull=never --read-only \
  --device=/dev/kfd --device="$fv_render" \
  --user="$(id -u):$(id -g)" \
  --group-add="$(stat -c '%g' "$fv_render")" --group-add="$(stat -c '%g' /dev/kfd)" \
  --tmpfs /tmp:rw,exec,size=2g --shm-size=2g \
  --env ROCR_VISIBLE_DEVICES="$fv_gpu_uuid" \
  --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 \
  --env OMP_NUM_THREADS=8 --env MKL_NUM_THREADS=8 \
  --env TORCH_BLAS_PREFER_HIPBLASLT="${FV_TORCH_BLAS_PREFER_HIPBLASLT:-0}" \
  --env ROCBLAS_USE_HIPBLASLT="${FV_ROCBLAS_USE_HIPBLASLT:-0}" \
  --env TORCH_COMPILE_DISABLE="${FV_TORCH_COMPILE_DISABLE:-0}" \
  --env FREEVIDEO_ROCM_VIDEO_BLAS="${FV_ROCM_VIDEO_BLAS:-default}" \
  --env FREEVIDEO_ROCM_SPATIAL_CONV="${FV_ROCM_SPATIAL_CONV:-miopen}" \
  --env FREEVIDEO_ROCM_ATTENTION="${FV_ROCM_ATTENTION:-aotriton}" \
  --env FREEVIDEO_ROCM_AUDIO_CONV="${FV_ROCM_AUDIO_CONV:-miopen}" \
  --env FV_STORAGE_ROOT=/data --env FV_PREFLIGHT_IMAGE_ID="$fv_image" \
  --env TRITON_CACHE_DIR="$fv_container_experiment_root/kernel-cache/$fv_cache_group/triton" \
  --env TORCHINDUCTOR_CACHE_DIR="$fv_container_experiment_root/kernel-cache/$fv_cache_group/inductor" \
  --env XDG_CACHE_HOME="$fv_container_experiment_root/cache" --env HF_HOME=/data/models/.hf-cache \
  --env XDG_CONFIG_HOME="$fv_container_experiment_root/cache/config" \
  --env MIOPEN_USER_DB_PATH="$fv_container_experiment_root/cache/miopen" \
  --env MIOPEN_CUSTOM_CACHE_DIR="$fv_container_experiment_root/kernel-cache/$fv_cache_group/miopen" \
  --env FREEVIDEO_HOME="$fv_container_experiment_root/runtime" \
  --env FREEVIDEO_VDN_ROOT="$fv_container_experiment_root/vendor/vdn" \
  --env FREEVIDEO_COMFY_ROOT="$fv_container_experiment_root/vendor/h3-text-encoder" \
  --env FREEVIDEO_MODEL_ROOT=/data/models/vdn \
  --env PYTHONPATH="$fv_pythonpath" \
  --mount "type=bind,src=$fv_checkout,dst=/workspace,readonly" \
  --mount "type=bind,src=$fv_storage_root,dst=/data" \
  --mount type=bind,src=/etc/passwd,dst=/etc/passwd,readonly \
  --mount type=bind,src=/etc/group,dst=/etc/group,readonly \
  --workdir /workspace --entrypoint "$fv_python" "$fv_image" "$@"
