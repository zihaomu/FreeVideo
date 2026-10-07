#!/usr/bin/env bash
set -euo pipefail
fv_checkout="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$fv_checkout/scripts/amd/storage.sh"
fv_report_dir="${FV_REPORT_DIR:-$fv_experiment_root/reports}"
fv_image="${FV_ROCM_IMAGE:-sha256:55bf8baa2a513b1c05bd256119fbc57a6ca64170e6cf5fe519b1cdd0c458cfd9}"
fv_gpu_uuid="${FV_GPU_UUID:-GPU-b11d6bcf3a61a551}"
fv_render="${FV_RENDER_DEVICE:-/dev/dri/renderD130}"
fv_report_name="${FV_REPORT_NAME:-rocm-preflight.json}"
[[ "$fv_report_name" != */* ]] || { printf 'Report name must be a filename\n' >&2; exit 2; }
[[ -c "$fv_render" && -c /dev/kfd ]] || { printf 'GPU device node missing\n' >&2; exit 2; }
docker image inspect "$fv_image" >/dev/null
mkdir -p "$fv_report_dir"
mkdir -p "$fv_experiment_root/envs/r9700-site" "$fv_experiment_root/kernel-cache" \
  "$fv_experiment_root/cache" "$fv_experiment_root/runtime"
[[ -f "$fv_experiment_root/vendor/vdn/src/models/hybrid_attention.py" ]] || \
  { printf 'Pinned VDN source is missing from %s/vendor/vdn\n' "$fv_experiment_root" >&2; exit 2; }
fv_report_dir="$(realpath "$fv_report_dir")"
fv_image_id="$(docker image inspect "$fv_image" --format '{{.Id}}')"
fv_render_group="$(stat -c '%g' "$fv_render")"
fv_kfd_group="$(stat -c '%g' /dev/kfd)"
docker run --rm --pull=never --read-only --network=none \
  --device=/dev/kfd --device="$fv_render" \
  --user="$(id -u):$(id -g)" --group-add="$fv_render_group" --group-add="$fv_kfd_group" \
  --tmpfs /tmp:rw,exec,size=512m --shm-size=256m \
  --env ROCR_VISIBLE_DEVICES="$fv_gpu_uuid" \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env TRITON_CACHE_DIR="$fv_container_experiment_root/kernel-cache/triton" \
  --env TORCHINDUCTOR_CACHE_DIR="$fv_container_experiment_root/kernel-cache/inductor" \
  --env XDG_CACHE_HOME="$fv_container_experiment_root/cache" \
  --env HF_HOME=/data/models/.hf-cache \
  --env FREEVIDEO_HOME="$fv_container_experiment_root/runtime" \
  --env FREEVIDEO_VDN_ROOT="$fv_container_experiment_root/vendor/vdn" \
  --env FREEVIDEO_MODEL_ROOT=/data/models/vdn \
  --env FV_PREFLIGHT_IMAGE_ID="$fv_image_id" \
  --env FV_STORAGE_ROOT=/data \
  --env PYTHONPATH="$fv_container_experiment_root/envs/r9700-site:/workspace" \
  --mount "type=bind,src=$fv_checkout,dst=/workspace,readonly" \
  --mount "type=bind,src=$fv_storage_root,dst=/data" \
  --mount "type=bind,src=$fv_report_dir,dst=/reports" \
  --workdir /workspace --entrypoint python3 "$fv_image_id" \
  scripts/amd/r9700_preflight.py --out "/reports/$fv_report_name"
