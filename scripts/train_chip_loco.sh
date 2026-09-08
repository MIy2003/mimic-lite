#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
framework_root="$(cd "${script_dir}/../../.." && pwd)"
cd "${framework_root}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY=1
uv --project venv/mjlab run "${script_dir}/prepare_chip_loco.py" \
  --root "${CHIP_LOCO_ROOT:-${framework_root}/../loco_manip_physical_rollout_accepted_v1}" \
  --output "${framework_root}/.cache/chip_loco"
exec bash "${script_dir}/train_chip.sh" \
  --config-dir "${framework_root}/.cache/chip_loco" task=chip_loco_train "$@"
