#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
framework_root="$(cd "$script_dir/../../.." && pwd)"
episode="${1:?Usage: bash replay_chip_episode.sh EPISODE_DIR [Hydra overrides]}"
shift
cd "$framework_root"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" HF_HUB_DISABLE_TELEMETRY=1
export ANY4HDMI_CACHE_BUILD_NUM_WORKERS=0
export CHIP_CHECKPOINT="${CHIP_CHECKPOINT:-$framework_root/../checkpoints/lambda-ik0008iq-20000/checkpoint_20000.pt}"
uv --project venv/mjlab run --no-sync "$script_dir/prepare_chip_episode.py" "$episode"
exec uv --project venv/mjlab run --no-sync "$script_dir/play.py" \
  --config-name replay_chip_loco \
  --config-dir "$framework_root/.cache/chip_episode/$(basename "$episode")" \
  task=chip_episode "$@"
