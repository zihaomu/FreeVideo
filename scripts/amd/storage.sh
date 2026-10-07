# Shared host/container paths. Source after setting fv_checkout.
fv_storage_root="${FV_STORAGE_ROOT:-/dc1/zihaomu/free_token_mapping}"
[[ -d "$fv_storage_root" ]] || { printf 'Storage root is missing: %s\n' "$fv_storage_root" >&2; exit 2; }
fv_storage_root="$(realpath "$fv_storage_root")"
fv_experiment_root="$fv_storage_root/experiments/freevideo-r9700"
fv_container_experiment_root=/data/experiments/freevideo-r9700
