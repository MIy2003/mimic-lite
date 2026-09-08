#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
framework_root="$(cd "${script_dir}/../../.." && pwd)"
cd "${framework_root}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY=1
# Avoid forking data-loader workers after CUDA initialization during FK caching.
export ANY4HDMI_CACHE_BUILD_NUM_WORKERS="${ANY4HDMI_CACHE_BUILD_NUM_WORKERS:-0}"
exec uv --project venv/mjlab run --no-sync projects/mimic-lite/scripts/train.py \
  task=chip backend=mjlab \
  task.num_envs="${CHIP_NUM_ENVS:-4096}" \
  total_iters="${CHIP_TOTAL_ITERS:-4000}" "$@"
