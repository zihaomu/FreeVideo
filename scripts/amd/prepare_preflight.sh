#!/usr/bin/env bash
set -euo pipefail
fv_checkout="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$fv_checkout/scripts/amd/storage.sh"
fv_image="${FV_ROCM_IMAGE:-sha256:55bf8baa2a513b1c05bd256119fbc57a6ca64170e6cf5fe519b1cdd0c458cfd9}"
docker image inspect "$fv_image" >/dev/null
mkdir -p "$fv_experiment_root/envs/r9700-site"
fv_free_bytes="$(df -B1 --output=avail "$fv_experiment_root" | tail -n 1 | tr -d ' ')"
(( fv_free_bytes >= 268435456 )) || { printf 'Need 256MiB free for preflight dependencies\n' >&2; exit 2; }
docker run --rm --pull=never --read-only --user="$(id -u):$(id -g)" \
  --tmpfs /tmp:rw,exec,size=256m --env PYTHONDONTWRITEBYTECODE=1 \
  --env PIP_DISABLE_PIP_VERSION_CHECK=1 \
  --mount "type=bind,src=$fv_checkout,dst=/workspace,readonly" \
  --mount "type=bind,src=$fv_storage_root,dst=/data" \
  --workdir /workspace --entrypoint python3 "$fv_image" \
  -m pip install --no-cache-dir --target "$fv_container_experiment_root/envs/r9700-site" \
  --requirement scripts/amd/requirements-preflight.lock.txt
