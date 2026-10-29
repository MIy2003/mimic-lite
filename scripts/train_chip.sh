#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "${script_dir}/../../active-adaptation/venv/mjlab" ]]; then
    framework_root="$(cd "${script_dir}/../../active-adaptation" && pwd)"
else
    framework_root="$(cd "${script_dir}/../../.." && pwd)"
fi
cd "${framework_root}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY=1
# Avoid forking data-loader workers after CUDA initialization during FK caching.
export ANY4HDMI_CACHE_BUILD_NUM_WORKERS="${ANY4HDMI_CACHE_BUILD_NUM_WORKERS:-0}"
config_args=()
if [[ "${1:-}" == "--config-dir" ]]; then
    config_args=("$1" "${2:?Missing config directory}")
    shift 2
fi
exec uv --project venv/mjlab run --no-sync projects/mimic-lite/scripts/train.py \
  "${config_args[@]}" task=chip backend=mjlab \
  task.num_envs="${CHIP_NUM_ENVS:-4096}" \
  total_iters="${CHIP_TOTAL_ITERS:-4000}" "$@"
