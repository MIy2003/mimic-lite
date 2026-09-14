#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
framework_root="$(cd "${script_dir}/../../.." && pwd)"
cd "$framework_root"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY=1
export ANY4HDMI_CACHE_BUILD_NUM_WORKERS="${ANY4HDMI_CACHE_BUILD_NUM_WORKERS:-0}"
export CHIP_CHECKPOINT="${CHIP_CHECKPOINT:-$framework_root/../checkpoints/lambda-ik0008iq-20000/checkpoint_20000.pt}"
if [[ ! -f "$CHIP_CHECKPOINT" || ! -f "$(dirname "$CHIP_CHECKPOINT")/cfg.yaml" ]]; then
  echo "Missing checkpoint or sibling cfg.yaml: $CHIP_CHECKPOINT" >&2
  exit 1
fi
uv --project venv/mjlab run --no-sync "$script_dir/prepare_chip_loco.py" \
  --root "${CHIP_LOCO_ROOT:-$framework_root/../loco_manip_physical_rollout_accepted_v1}" \
  --output "$framework_root/.cache/chip_loco"
exec uv --project venv/mjlab run --no-sync "$script_dir/play.py" \
  --config-name replay_chip_loco --config-dir "$framework_root/.cache/chip_loco" "$@"
